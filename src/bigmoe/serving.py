"""What the model servers (glm53, dsv41) share around exllamav3's AsyncGenerator: OpenAI-style errors, context checks,
generator liveness, usage with prefix-cache hits and server-side timings, a log line per request, and the prefix-cache
tiers. Stdlib only at import time (FastAPI is imported inside install()), so it is testable without the GPU stack.

What it guards against:
- A prompt that does not fit the cache. exllamav3 asserts in Generator.enqueue ("Job requires N pages"); raised inside
  a streaming response that tore the connection down mid-stream, so clients saw a transport error and retried the same
  oversized request. Now it is a 400 `context_length_exceeded` before anything is streamed, worded the way OpenAI
  words it, which agent clients (DeepSeek Harness) recognize and answer by compacting the conversation.
- Any other failure before the first token: a JSON error with a status code instead of a dropped stream.
- A failure after streaming started: an SSE `error` event (the OpenAI SDKs raise it as an APIError), then [DONE].
- A dead generator loop (an exception in Generator.iterate, e.g. CUDA out of memory). exllamav3 then fails every later
  job while the process stays up and /health kept answering ok. /health now reports 503 and, after a short grace for
  in-flight requests to get their error, the process exits so a supervisor can start a fresh one.
"""
import asyncio
import json
import os
import sys
import time
import traceback

# A job needs room for at least this many new tokens (or the requested max_tokens if smaller), else the request is
# refused as too long: a 10-token answer cut off by the cache end helps nobody, a context error lets the client compact
MIN_OUTPUT_TOKENS = 1024
EXIT_CODE_GENERATOR_DEAD = 70
EXIT_GRACE_S = 3.0          # in-flight requests get their error before a dead-generator exit
GiB = 1024 ** 3


class RequestError(Exception):
    """An error with an HTTP status and an OpenAI-shaped body: {"error": {message, type, param, code}}."""

    def __init__(self, status: int, message: str, type: str = "invalid_request_error", code: str = None,
                 param: str = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def context_exceeded(prompt_tokens: int, context: int, max_tokens: int = None, reserve: int = 0) -> RequestError:
    asked = f" and {max_tokens} in the completion" if max_tokens else ""
    return RequestError(
        400, f"This model's maximum context length is {context} tokens. However, you requested "
             f"{prompt_tokens + (max_tokens or 0)} tokens ({prompt_tokens} in the messages{asked}); {reserve} more are "
             f"reserved by the server. Please reduce the length of the messages or the completion.",
        code="context_length_exceeded", param="messages")


def output_budget(prompt_tokens: int, context: int, reserve: int, max_tokens=None) -> int:
    """max_new_tokens for a job: the requested max_tokens clamped to the room left in the cache. Raises RequestError
    (400, context_length_exceeded) when the room is below min(max_tokens, MIN_OUTPUT_TOKENS)."""
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            raise RequestError(400, f"max_tokens must be an integer, got {max_tokens!r}", param="max_tokens")
        if max_tokens < 1:
            raise RequestError(400, f"max_tokens must be at least 1, got {max_tokens}", param="max_tokens")
    room = context - prompt_tokens - reserve
    need = min(max_tokens, MIN_OUTPUT_TOKENS) if max_tokens else MIN_OUTPUT_TOKENS
    if room < need:
        raise context_exceeded(prompt_tokens, context, max_tokens, reserve)
    return room if max_tokens is None else min(max_tokens, room)


def job_timings(result: dict) -> dict:
    """Server-side timings from exllamav3's final job result (eos): queue / prefill / decode wall time, cached prompt
    tokens (prefix reuse) and accepted speculative tokens."""
    def ms(key):
        v = result.get(key)
        return round(float(v) * 1000, 1) if v is not None else None
    return {k: v for k, v in {
        "queue_ms": ms("time_enqueued"), "prefill_ms": ms("time_prefill"), "decode_ms": ms("time_generate"),
        "cached_tokens": result.get("cached_tokens"), "new_tokens": result.get("new_tokens"),
        "accepted_draft_tokens": result.get("accepted_draft_tokens"),
    }.items() if v is not None}


def log_request(prompt_tokens: int, completion_tokens: int, fin: dict = None):
    """One line per finished job in the server log: where a request's time went (prefill vs decode, prefix reuse)."""
    t = (fin or {}).get("timings") or {}
    sec = lambda k: f"{t[k] / 1000:.1f}s" if k in t else "-"
    print(f" -- request: prompt {prompt_tokens} (cached {t.get('cached_tokens', '-')}), out {completion_tokens}, "
          f"queue {sec('queue_ms')}, prefill {sec('prefill_ms')}, decode {sec('decode_ms')}, "
          f"finish {(fin or {}).get('reason', '-')} [{time.strftime('%H:%M:%S')}]", flush=True)


def usage(prompt_tokens: int, completion_tokens: int, fin: dict = None) -> dict:
    """OpenAI usage plus prompt_tokens_details.cached_tokens (prefix-cache hits, shown by clients) and, non-standard
    like llama.cpp's, `timings` (job_timings)."""
    u = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
         "total_tokens": prompt_tokens + completion_tokens}
    t = (fin or {}).get("timings")
    if t:
        if t.get("cached_tokens") is not None:
            u["prompt_tokens_details"] = {"cached_tokens": int(t["cached_tokens"])}
        u["timings"] = t
    return u


def as_request_error(exc: BaseException, prompt_tokens: int = None, context: int = None) -> RequestError:
    """Map an exception raised while creating or running a job to a RequestError."""
    if isinstance(exc, RequestError):
        return exc
    text = f"{type(exc).__name__}: {exc}"
    if "cannot be enqueued" in str(exc) or "pages (only" in str(exc):   # exllamav3's cache-size assertion
        if prompt_tokens is not None and context is not None:
            return context_exceeded(prompt_tokens, context)
        return RequestError(400, f"The request does not fit the model's maximum context length ({text})",
                            code="context_length_exceeded", param="messages")
    if "out of memory" in str(exc).lower():
        return RequestError(503, f"The server ran out of GPU memory for this request ({text})", type="server_error",
                            code="server_overloaded")
    return RequestError(500, text, type="server_error")


async def read_json(req, require: tuple = ()) -> dict:
    """The request body as a JSON object (400 if it is not one, or lacks a required field)."""
    try:
        body = await req.json()
    except Exception as e:
        raise RequestError(400, f"request body is not valid JSON ({type(e).__name__}: {e})")
    if not isinstance(body, dict):
        raise RequestError(400, "request body must be a JSON object")
    for field in require:
        if body.get(field) is None:
            raise RequestError(400, f"missing required field {field!r}", param=field)
    return body


def sse_error(exc: BaseException) -> str:
    return "data: " + json.dumps(as_request_error(exc).body()) + "\n\n"


async def guard_stream(chunks):
    """Pass SSE chunks through. Once a response has started its status cannot change, so an exception becomes an
    in-band error event (OpenAI SDKs raise it as an APIError) followed by [DONE], instead of a torn connection."""
    try:
        async for c in chunks:
            yield c
    except Exception as e:
        err = as_request_error(e)
        if err.status >= 500:
            traceback.print_exception(e)
        yield sse_error(e)
        yield "data: [DONE]\n\n"


def generator_failure(gen):
    """The exception that killed the exllamav3 AsyncGenerator's iteration loop, or None while it is alive."""
    if gen is None:
        return None
    err = getattr(gen, "error", None)
    if err is not None:
        return err
    task = getattr(gen, "iteration_task", None)
    if task is not None and task.done():
        if task.cancelled():
            return RuntimeError("generator iteration task was cancelled")
        return task.exception() or RuntimeError("generator iteration task ended")
    return None


def prefix_cache_tiers(args) -> dict:
    """exllamav3 Generator arguments for the prefix-cache tiers behind the GPU cache (-ccs / -dcd, the backends'
    cpu_cache_gb / disk_cache options): K/V pages the GPU cache evicts spill to pinned RAM (reserved by the engine like
    -rcs) and on to disk, so a conversation's prefix survives other conversations and restarts. Disk writes happen only
    when pages leave the GPU or the server goes idle (write interval 0): while a request runs the NVMe serves the
    expert reads."""
    return {"cpu_cache_size": int((args.cpu_cache_size or 0) * GiB), "disk_cache_dir": args.disk_cache_dir,
            "disk_cache_size": int(args.disk_cache_size * GiB), "disk_cache_write_interval": 0.0}


def recurrent_cache_stats(gen) -> dict | None:
    """Occupancy of exllamav3's recurrent-state checkpoint cache (prefix reuse for models with recurrent layers)."""
    rc = getattr(getattr(gen, "generator", None), "recurrent_cache", None)
    if rc is None:
        return None
    n = len(rc)
    return {"checkpoints": n, "bytes": int(rc.current_size), "max_bytes": int(rc.max_size),
            "bytes_per_checkpoint": int(rc.current_size / n) if n else None, **dict(getattr(rc, "metrics", {}))}


def gpu_memory_stats() -> dict | None:
    """GPU memory in MiB as torch sees it: held by live tensors, reserved by its caching allocator, the peak since the
    start, and what the device has free in total (torch imported here: this module stays stdlib-only at import)."""
    import torch
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    mib = lambda b: round(b / 2 ** 20)
    return {"allocated_mib": mib(torch.cuda.memory_allocated()), "reserved_mib": mib(torch.cuda.memory_reserved()),
            "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()), "free_mib": mib(free), "total_mib": mib(total)}


def install(app, get_gen, name: str = "server", exit_on_dead: bool = True):
    """Error handlers, a liveness-aware /health and the dead-generator exit for a FastAPI app. get_gen() returns the
    AsyncGenerator (None until startup). Registers GET /health: define no other /health route. exit_on_dead=False
    (tests) keeps the process alive when the generator loop dies."""
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(RequestError)
    async def _request_error(_req: Request, exc: RequestError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError):
        return JSONResponse(RequestError(400, str(exc)).body(), status_code=400)

    @app.exception_handler(Exception)
    async def _any_error(_req: Request, exc: Exception):
        err = as_request_error(exc)
        if err.status >= 500:
            traceback.print_exception(exc)
        return JSONResponse(err.body(), status_code=err.status)

    @app.get("/health")
    async def health():
        gen = get_gen()
        if gen is None:
            return JSONResponse({"ok": False, "status": "starting"}, status_code=503)
        dead = generator_failure(gen)
        if dead is not None:
            return JSONResponse({"ok": False, "status": "generator dead", "error": f"{type(dead).__name__}: {dead}"},
                                status_code=503)
        return JSONResponse({"ok": True})

    @app.on_event("startup")
    async def _watch_generator():
        if not exit_on_dead:
            return

        async def watch():
            while True:
                await asyncio.sleep(1.0)
                dead = generator_failure(get_gen())
                if dead is None:
                    continue
                print(f" !! {name}: the generator loop died ({type(dead).__name__}: {dead}); in-flight requests get "
                      f"the error, exiting in {EXIT_GRACE_S:g} s so the server can be restarted", flush=True)
                traceback.print_exception(dead)
                await asyncio.sleep(EXIT_GRACE_S)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(EXIT_CODE_GENERATOR_DEAD)

        app.state.generator_watch = asyncio.create_task(watch())
