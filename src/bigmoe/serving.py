"""What the servers share around the exllamav3 executor: OpenAI-style errors, the context rule, timings and a log
line per request, the prefix-cache tiers, and out-of-GPU-memory cleanup. Stdlib only at import time.

- A prompt that does not fit the cache is a 400 `context_length_exceeded`, worded as OpenAI words it, before anything
  is streamed; agent clients recognize it and compact the conversation.
- A job that runs out of GPU memory: the exception's traceback keeps the failed forward's activations alive, in
  reference cycles that only a full garbage collection frees. release_failed frees them; the executor runs a job that
  failed before its first token once more with half-size prefill chunks (vllm/engine.py).
"""
import gc
import time
import traceback

# A job needs room for at least this many new tokens (or the requested max_tokens if smaller), else the request is
# refused as too long: a 10-token answer cut off by the cache end helps nobody, a context error lets the client compact
MIN_OUTPUT_TOKENS = 1024
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


def is_cuda_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


def release_failed(exc: BaseException):
    """Free what a failed forward left behind: the locals of the finished frames in the exception's traceback (and in
    its causes), whatever only cyclic garbage collection reclaims, and torch's cached blocks."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
