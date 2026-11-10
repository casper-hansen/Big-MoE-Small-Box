"""The servers' routes beside the OpenAI API, on vLLM's server (a vllm.endpoint_plugins entry point,
allowlisted by serve.py through VLLM_PLUGINS), plus the SSE keepalive middleware.

  /stats /server_info /v1/models    engine statistics (requests, draft acceptance, tiers, vision, GPU memory), the
                                    exllamav3 arguments, the model card with its input modalities
  /generate                         SGLang-shaped generation on token ids (bigmoe bench, verify) and teacher-forced
                                    logprobs (return_logprob)
  /kl_capture /kl_force /kl_teacher quality tools (models/dsv41/tools)
  /nv_verify /nv2_rate /cpu         shared-engine diagnostics

The engine lives in the EngineCore process: every route reaches it through engine_client.collective_rpc("bigmoe_call",
...) (worker.BoxWorker.bigmoe_call), which EngineCore runs between two steps.
"""
import asyncio
import json
import time
import uuid

KEEPALIVE_S = 15.0
SSE_KEEPALIVE = b": keepalive\n\n"


class KeepaliveMiddleware:
    """An SSE comment on a streaming response that has been silent for KEEPALIVE_S (queued, or in a long prefill):
    HTTP clients drop a response body that stays silent too long. Clients' idle watchdogs see events, not comments,
    so they still catch a server that has stopped. Pure ASGI: only text/event-stream responses are touched."""

    def __init__(self, app, interval: float = KEEPALIVE_S):
        self.app, self.interval = app, interval

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        lock = asyncio.Lock()
        st = {"last": time.monotonic(), "done": False, "task": None}

        async def ping():
            while not st["done"]:
                await asyncio.sleep(max(0.05, self.interval - (time.monotonic() - st["last"])))
                if st["done"] or time.monotonic() - st["last"] < self.interval:
                    continue
                async with lock:
                    if st["done"]:
                        return
                    try:
                        await send({"type": "http.response.body", "body": SSE_KEEPALIVE, "more_body": True})
                    except Exception:
                        st["done"] = True
                        return
                    st["last"] = time.monotonic()

        async def wrapped(msg):
            async with lock:
                if msg["type"] == "http.response.start":
                    ctype = b""
                    for k, v in msg.get("headers") or ():
                        if k.lower() == b"content-type":
                            ctype = v
                    await send(msg)
                    st["last"] = time.monotonic()
                    if b"text/event-stream" in ctype:
                        st["task"] = asyncio.create_task(ping())
                    return
                if msg["type"] == "http.response.body":
                    if not msg.get("more_body", False):
                        st["done"] = True
                    await send(msg)
                    st["last"] = time.monotonic()
                    return
                await send(msg)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            st["done"] = True
            t = st["task"]
            if t is not None and not t.done():
                t.cancel()


keepalive_middleware = KeepaliveMiddleware


class ModelAliasMiddleware:
    """One model per server: a request may name any `model` (agent clients send their own aliases); vLLM answers 404
    for a name it does not serve. A JSON POST to /v1/... whose model is missing or unknown gets the served name."""

    def __init__(self, app):
        import os
        self.app = app
        self.name = os.environ.get("BIGMOE_SERVED_NAME") or ""
        self.known = {self.name}

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" or not self.name or \
                not scope.get("path", "").startswith("/v1/"):
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] != "http.request":
                return await self.app(scope, _replay([msg]), send)
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        try:
            obj = json.loads(body) if body else None
            if isinstance(obj, dict) and obj.get("model") not in self.known:
                obj["model"] = self.name
                body = json.dumps(obj).encode()
                scope = dict(scope, headers=[(k, v) for k, v in scope.get("headers", ())
                                             if k.lower() != b"content-length"]
                             + [(b"content-length", str(len(body)).encode())])
        except ValueError:
            pass
        return await self.app(scope, _replay([{"type": "http.request", "body": body, "more_body": False}],
                                             receive), send)


def _replay(msgs, receive=None):
    q = list(msgs)

    async def r():
        if q:
            return q.pop(0)
        if receive is not None:
            return await receive()
        return {"type": "http.disconnect"}
    return r


model_alias_middleware = ModelAliasMiddleware


class ContextErrorMiddleware:
    """vLLM phrases overlong-prompt errors like OpenAI but uses numeric code 400. Set OpenAI's
    `context_length_exceeded` code on small JSON 400 responses so OpenAI/agent clients can compact and retry."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/v1/"):
            return await self.app(scope, receive, send)
        status = [None]
        start = [None]
        chunks = []

        async def wrapped(msg):
            if msg["type"] == "http.response.start":
                if msg["status"] != 400:
                    await send(msg)
                else:
                    start[0] = msg
                status[0] = msg["status"]
                return
            if msg["type"] == "http.response.body" and status[0] == 400:
                chunks.append(msg)
                return
            await send(msg)

        await self.app(scope, receive, wrapped)
        if status[0] != 400:
            return
        body = b"".join(m.get("body", b"") for m in chunks)
        try:
            obj = json.loads(body)
            err = obj.get("error") if isinstance(obj, dict) else None
            if isinstance(err, dict) and "maximum context length" in str(err.get("message", "")):
                err["code"], err["param"] = "context_length_exceeded", "messages"
                body = json.dumps(obj).encode()
        except ValueError:
            pass
        msg = dict(start[0], headers=[(k, v) for k, v in start[0].get("headers", ())
                                      if k.lower() != b"content-length"] + [(b"content-length", str(len(body)).encode())])
        await send(msg)
        await send({"type": "http.response.body", "body": body, "more_body": False})


context_error_middleware = ContextErrorMiddleware


class BigmoeRoutes:
    """vllm.endpoint_plugins factory (EndpointPlugin protocol)."""
    name = "bigmoe_routes"
    required_tasks = ("generate",)

    def attach_router(self, app):
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        r = APIRouter()

        def client(req: Request):
            return req.app.state.engine_client

        async def call(req: Request, method: str, *args, **kwargs):
            out = await client(req).collective_rpc("bigmoe_call", args=(method, *args), kwargs=kwargs or None)
            return out[0] if isinstance(out, list) else out

        async def wait_idle(req):
            while await call(req, "has_work"):
                await asyncio.sleep(0.05)

        @r.get("/stats")
        async def stats(req: Request):
            return JSONResponse(await call(req, "summary"))

        @r.get("/server_info")
        async def server_info(req: Request):
            return JSONResponse(await call(req, "server_info"))

        @r.get("/v1/models")
        async def models(req: Request):
            info = await call(req, "model_card")
            names = req.app.state.openai_serving_models.base_model_paths
            return JSONResponse({"object": "list", "data": [
                dict({"id": m.name, "object": "model", "created": int(time.time()), "owned_by": "local",
                      "root": m.model_path, "parent": None}, **info) for m in names]})

        @r.get("/kl_capture")
        async def kl_capture(req: Request, on: int = -1):
            return JSONResponse(await call(req, "kl_capture", on))

        @r.post("/kl_teacher")
        async def kl_teacher(req: Request):
            b = await req.json()
            await wait_idle(req)
            return JSONResponse(await call(req, "kl_teacher", [int(x) for x in b["ids"]], int(b["start"]),
                                           int(b["count"])))

        @r.post("/kl_force")
        async def kl_force(req: Request):
            """Decode `ref` after `prompt` with every output token forced (KL capture on): per-step decode logits of
            this engine configuration on a fixed token path, comparable across configurations."""
            b = await req.json()
            prev = (await call(req, "kl_capture", 1, keep=True))["on"]
            t0 = time.time()
            try:
                n, _text, fin = 0, "", None
                async for _text, n, fin in generate_ids(req, [int(x) for x in b["prompt"]],
                                                       {"temperature": 0, "max_new_tokens": len(b["ref"])},
                                                       force=[int(x) for x in b["ref"]]):
                    pass
            finally:
                cap = await call(req, "kl_capture", 1 if prev else 0)
            return JSONResponse({"n": n, "secs": time.time() - t0, "prompt": cap["prompt"], "gen": cap["gen"],
                                 "steps": cap["steps"], "timings": (fin or {}).get("timings")})

        @r.get("/nv_verify")
        async def nv_verify(req: Request, n: int = 32):
            return JSONResponse(await call(req, "nv_verify", n))

        @r.get("/nv2_rate")
        async def nv2_rate(req: Request, mbps: float = 0.0):
            return JSONResponse(await call(req, "nv2_rate", mbps))

        @r.get("/cpu")
        async def cpu_switch(req: Request, on: int = 1):
            return JSONResponse(await call(req, "cpu_switch", on))

        @r.post("/generate")
        async def generate(req: Request):
            """SGLang-shaped: {"input_ids" | "text", "sampling_params", "stream", "return_logprob"}."""
            b = await req.json()
            sp = b.get("sampling_params") or {}
            ids = b.get("input_ids")
            if ids is None:
                tok = await client(req).get_tokenizer()
                ids = tok.encode(b["text"], add_special_tokens=True)
            ids = [int(x) for x in ids]
            if b.get("return_logprob"):
                await wait_idle(req)
                t = await call(req, "teacher_forced", ids, int(b.get("logprob_start_len", 0)),
                               int(b.get("top_logprobs_num", 20)), b.get("token_ids_logprob") or [])
                meta = {k: t[k] for k in ("input_top_logprobs", "input_token_ids_logprobs", "output_top_logprobs",
                                          "output_token_ids_logprobs")}
                meta.update({"prompt_tokens": len(ids), "completion_tokens": 1, "finish_reason": {"type": "length"}})
                return JSONResponse({"text": "", "output_ids": [t["next_token"]], "meta_info": meta})
            P = len(ids)
            try:
                from .. import serving
                info = getattr(req.app.state, "bigmoe_server_info", None)
                if info is None:
                    info = await call(req, "server_info")
                    req.app.state.bigmoe_server_info = info
                ndt = info["server_args"].get("num_draft_tokens") or 8
                sp = dict(sp, max_new_tokens=serving.output_budget(P, int(info["context_length"]), ndt + 2,
                                                                  sp.get("max_new_tokens")))
            except serving.RequestError as e:
                return JSONResponse(e.body(), status_code=e.status)
            if b.get("stream"):
                async def events():
                    async for text, n, fin in generate_ids(req, ids, sp):
                        yield "data: " + json.dumps({"text": text, "meta_info": {
                            "prompt_tokens": P, "completion_tokens": n, "finish_reason": fin}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(events(), media_type="text/event-stream")
            text, n, fin = "", 0, None
            async for text, n, fin in generate_ids(req, ids, sp):
                pass
            return JSONResponse({"text": text, "meta_info": {"prompt_tokens": P, "completion_tokens": n,
                                                             "finish_reason": fin}})

        async def generate_ids(req, ids, sp, force=None):
            """(text so far, completion tokens, finish or None) per engine output."""
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt
            t = sp.get("temperature")
            params = SamplingParams(temperature=1.0 if t is None else float(t), top_p=float(sp.get("top_p") or 1.0),
                                    top_k=int(sp.get("top_k") or 0), min_p=float(sp.get("min_p") or 0.0),
                                    max_tokens=sp.get("max_new_tokens"), skip_special_tokens=True,
                                    extra_args={"bigmoe_force": force} if force else None)
            rid = "generate-" + uuid.uuid4().hex
            async for out in client(req).generate(TokensPrompt(prompt_token_ids=ids), params, rid):
                c = out.outputs[0]
                fin = None
                if c.finish_reason is not None:
                    fin = {"type": "length" if c.finish_reason == "length" else "stop",
                           "reason": str(c.stop_reason if c.stop_reason is not None else c.finish_reason),
                           "timings": {"cached_tokens": out.num_cached_tokens}}
                yield c.text, len(c.token_ids), fin

        app.include_router(r)

    async def init_state(self, engine_client, state, args):
        return None
