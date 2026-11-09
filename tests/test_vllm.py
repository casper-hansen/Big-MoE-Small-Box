"""bigmoe.vllm without the GPU: the serve command line, the GLM template overlay, the ASGI middlewares, the image
placeholder splice, the context rule, and (with vLLM installed) the scheduler / worker loop over the FakeEngine."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigmoe.vllm import serve
from bigmoe.vllm.engine import FakeEngine, Progress


def args(**kw):
    a = dict(model_dir="/m", cache_size=131072, max_batch_size=4, host="0.0.0.0", port=30000, served_name="ds",
             num_draft_tokens=None)
    a.update(kw)
    return SimpleNamespace(**a)


def test_vllm_argv():
    argv = serve.vllm_argv("dsv41", args(), ["--enable-log-requests"])
    assert argv[:2] == ["serve", "/m"]
    get = lambda k: argv[argv.index(k) + 1]
    assert get("--max-model-len") == "131072" and get("--max-num-seqs") == "4"
    assert int(get("--max-num-batched-tokens")) == 131072 * 4          # the budget never binds
    assert get("--worker-cls") == "bigmoe.vllm.worker.BoxWorker"
    assert get("--scheduler-cls") == "bigmoe.vllm.scheduler.BoxScheduler"
    assert json.loads(get("--hf-overrides")) == {"architectures": ["BigmoeDeepseekV41ForCausalLM"]}
    assert get("--tool-call-parser") == "deepseek_v41" and get("--reasoning-parser") == "deepseek_v41"
    assert json.loads(get("--default-chat-template-kwargs")) == {"enable_thinking": False}
    assert get("--chat-template-content-format") == "openai"
    assert "--disable-prefix-caching" in argv and argv[-1] == "--enable-log-requests"
    g = serve.vllm_argv("glm53", args(served_name=None), [], tokenizer="/tok")
    assert g[g.index("--tool-call-parser") + 1] == "glm47" and g[g.index("--tokenizer") + 1] == "/tok"
    assert "--default-chat-template-kwargs" not in g and g[g.index("--served-model-name") + 1] == "glm53"


def test_split_argv():
    assert serve.split_argv(["-m", "x", "--", "--foo"]) == (["-m", "x"], ["--foo"])
    assert serve.split_argv(["-m", "x"]) == (["-m", "x"], [])


GLM_TAIL = ('{%- macro visible_text(content) -%}{%- for item in content -%}'
            '{%- if item.type == "text" -%}{{- item.text }}{%- elif item.type in ["image", "image_url"] -%}'
            '{%- set media_type = item.type | replace("_url", "") -%}' + serve.GLM_REMINDER +
            '{%- endif -%}{%- endfor -%}{%- endmacro -%}'
            '[gMASK]<sop>{%- for m in messages -%}<|{{ m.role }}|>{{ visible_text(m.content) }}{%- endfor -%}'
            '{%- if add_generation_prompt -%}<|assistant|>{{- "<think>" -}}{%- endif -%}')


def test_glm_template_overlay(tmp_path):
    jinja2 = pytest.importorskip("jinja2")
    src = tmp_path / "glm"
    src.mkdir()
    (src / "chat_template.jinja").write_text(GLM_TAIL)
    (src / "tokenizer.json").write_text("{}")
    (src / "model-00001.safetensors").write_text("x")
    d = Path(serve.tokenizer_overlay("glm53", str(src), str(tmp_path / "home")))
    assert (d / "tokenizer.json").is_symlink() and not (d / "model-00001.safetensors").exists()
    t = jinja2.Environment().from_string((d / "chat_template.jinja").read_text())
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look "}, {"type": "image"}]}]
    on = t.render(messages=msgs, add_generation_prompt=True)
    off = t.render(messages=msgs, add_generation_prompt=True, enable_thinking=False)
    assert on.endswith("<|assistant|><think>") and off.endswith("<|assistant|><think></think>")
    assert "look <|begin_of_image|><|image|><|end_of_image|>" in on and "reminder" not in on
    # idempotent: the same directory, nothing rewritten
    assert serve.tokenizer_overlay("glm53", str(src), str(tmp_path / "home")) == str(d)
    assert serve.tokenizer_overlay("dsv41", str(src), str(tmp_path / "home")) is None
    # a same-named checkpoint elsewhere must not inherit this checkpoint's tokenizer symlinks
    other = tmp_path / "other" / "glm"
    other.mkdir(parents=True)
    (other / "chat_template.jinja").write_text(GLM_TAIL)
    (other / "tokenizer.json").write_text('{"other": true}')
    d2 = Path(serve.tokenizer_overlay("glm53", str(other), str(tmp_path / "home")))
    assert d2 != d and (d2 / "tokenizer.json").resolve() == (other / "tokenizer.json").resolve()
    assert (d / "tokenizer.json").resolve() == (src / "tokenizer.json").resolve()


def test_glm_overlay_without_images(monkeypatch):
    monkeypatch.setenv("BIGMOE_MM", "0")
    out = serve.glm_template(GLM_TAIL, images=False)
    assert serve.GLM_REMINDER in out and "enable_thinking" in out


async def _run_app(mw_cls, app, scope, body=b"", **kw):
    sent, msgs = [], [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return msgs.pop(0) if msgs else {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)
    await mw_cls(app, **kw)(scope, receive, send)
    return sent


def test_keepalive_middleware():
    from bigmoe.vllm.routes import KeepaliveMiddleware, SSE_KEEPALIVE

    async def slow_sse(scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await asyncio.sleep(0.35)
        await send({"type": "http.response.body", "body": b"data: x\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def json_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await asyncio.sleep(0.35)
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    sse = asyncio.run(_run_app(KeepaliveMiddleware, slow_sse, {"type": "http"}, interval=0.1))
    bodies = [m.get("body") for m in sse if m["type"] == "http.response.body"]
    assert bodies.count(SSE_KEEPALIVE) >= 2 and bodies[-2:] == [b"data: x\n\n", b""]
    plain = asyncio.run(_run_app(KeepaliveMiddleware, json_app, {"type": "http"}, interval=0.1))
    assert [m.get("body") for m in plain if m["type"] == "http.response.body"] == [b"{}"]


def test_model_alias_middleware(monkeypatch):
    from bigmoe.vllm.routes import ModelAliasMiddleware
    monkeypatch.setenv("BIGMOE_SERVED_NAME", "ds")
    seen = []

    async def app(scope, receive, send):
        m = await receive()
        seen.append((json.loads(m["body"]), dict(scope["headers"]).get(b"content-length")))

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"content-type", b"application/json"), (b"content-length", b"99")]}
    for model, want in (("default", "ds"), (None, "ds"), ("ds", "ds")):
        body = json.dumps({"messages": [], **({"model": model} if model else {})}).encode()
        asyncio.run(_run_app(ModelAliasMiddleware, app, scope, body))
        obj, clen = seen.pop()
        assert obj["model"] == want
        if model != "ds":
            assert int(clen) == len(json.dumps(obj).encode())
    asyncio.run(_run_app(ModelAliasMiddleware, app, dict(scope, path="/generate"), b'{"model": "x"}'))
    assert seen.pop()[0]["model"] == "x"


def test_context_error_code_middleware():
    from bigmoe.vllm.routes import ContextErrorMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 400,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", b"999")]})
        await send({"type": "http.response.body", "body": json.dumps(
            {"error": {"message": "This model's maximum context length is 131072 tokens", "code": 400,
                       "param": "input_tokens"}}).encode(), "more_body": False})
    sent = asyncio.run(_run_app(ContextErrorMiddleware, app, {"type": "http", "path": "/v1/chat/completions"}))
    body = json.loads(sent[-1]["body"])
    assert body["error"]["code"] == "context_length_exceeded" and body["error"]["param"] == "messages"
    assert int(dict(sent[0]["headers"])[b"content-length"]) == len(sent[-1]["body"])


def test_splice():
    mm = pytest.importorskip("bigmoe.vllm.mm", exc_type=ImportError)
    F = 1_000_000_000
    ds = SimpleNamespace(token_list=[F, F + 1, F + 2, F + 3], first_index=F)
    glm = SimpleNamespace(token_list=[7, F + 10, F + 11, 8], first_index=F + 10)
    prompt = [1, 9, 9, 9, 9, 2, 7, 5, 5, 8, 3]
    out = mm.splice(prompt, [("a", 1, 4, None, None), ("b", 7, 2, None, None)], [ds, glm])
    assert out == [1, F, F + 1, F + 2, F + 3, 2, 7, F + 10, F + 11, 8, 3]
    with pytest.raises(ValueError):
        mm.splice(prompt, [("a", 1, 3, None, None)], [ds])


def test_unsupported_sampling_options_are_rejected():
    from bigmoe.serving import RequestError
    from bigmoe.vllm.engine import validate_sampling_params
    validate_sampling_params(SimpleNamespace())
    validate_sampling_params(SimpleNamespace(structured_outputs=SimpleNamespace(
        all_non_structural_tag_constraints_none=lambda: True)))   # native tool-parser constraint, not response_format
    for field, value in (("logprobs", 5), ("prompt_logprobs", 1),
                         ("structured_outputs", SimpleNamespace(all_non_structural_tag_constraints_none=lambda: False)),
                         ("allowed_token_ids", [1]), ("bad_words", ["x"])):
        with pytest.raises(RequestError) as err:
            validate_sampling_params(SimpleNamespace(**{field: value}))
        assert err.value.status == 400 and field.split("_")[0] in err.value.message


def test_generate_route_defaults_to_remaining_context():
    pytest.importorskip("vllm")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from vllm.outputs import CompletionOutput, RequestOutput
    from bigmoe.vllm.routes import BigmoeRoutes

    class Client:
        seen = []

        async def collective_rpc(self, method, args=None, kwargs=None):
            return [{"context_length": 5000, "server_args": {"num_draft_tokens": 8}, "load_seconds": 0}]

        async def generate(self, prompt, params, rid):
            self.seen.append(params.max_tokens)
            c = CompletionOutput(index=0, text="ok", token_ids=[1, 2], cumulative_logprob=None, logprobs=None,
                                 finish_reason="stop")
            yield RequestOutput(request_id=rid, prompt=None, prompt_token_ids=[1], prompt_logprobs=None,
                                outputs=[c], finished=True)

    app = FastAPI()
    client = Client()
    app.state.engine_client = client
    BigmoeRoutes().attach_router(app)
    http = TestClient(app)
    r = http.post("/generate", json={"input_ids": list(range(10)), "sampling_params": {"temperature": 0}})
    assert r.status_code == 200 and r.json()["text"] == "ok" and client.seen == [4980]
    r = http.post("/generate", json={"input_ids": list(range(10)),
                                     "sampling_params": {"max_new_tokens": 5}, "stream": True})
    assert r.status_code == 200 and client.seen[-1] == 5 and '"completion_tokens": 2' in r.text


def test_forced_output_is_constrained_after_enqueue():
    from bigmoe.vllm.engine import ExlEngine, Request
    calls = []

    class Gen:
        def enqueue(self, job):
            calls.append("enqueue")

    class Job:
        def constrain_output_now(self, ids):
            assert ids.tolist() == [[4, 5]]
            calls.append("constrain")
    e = ExlEngine("dsv41", [])
    e.gen = Gen()
    e.torch = pytest.importorskip("torch")
    r = Request("r", [1], SimpleNamespace(), force=[4, 5], job=Job())
    e._enqueue(r)
    assert calls == ["enqueue", "constrain"]


def test_oom_before_first_token_retries_with_smaller_chunks(monkeypatch):
    from bigmoe.vllm.engine import ExlEngine, Request
    e = ExlEngine("dsv41", [])
    e.ctx = 131072
    jobs = []

    class Gen:
        max_chunk_size = 8192

        def enqueue(self, job):
            jobs.append(job)

    e.gen = Gen()
    e._make_job = lambda r: object()
    monkeypatch.setattr("bigmoe.serving.release_failed", lambda exc: None)
    r = Request("r", [1, 2, 3], SimpleNamespace())
    e.reqs["r"] = r

    class OutOfMemoryError(Exception):
        pass
    e._failed(r, OutOfMemoryError("CUDA out of memory"))
    assert e.gen.max_chunk_size == 4096 and r.retried and not r.finished and len(jobs) == 1
    e._failed(r, OutOfMemoryError("CUDA out of memory"))
    assert e.gen.max_chunk_size == 8192 and r.finished == "error" and r.status == 503


def test_cached_prefix_measured_per_job_allocation(monkeypatch):
    from bigmoe.vllm.engine import ExlEngine, Request
    pytest.importorskip("exllamav3")
    import exllamav3.generator
    metrics = {"alloc_cached_pages": 10}

    class Job:
        def __init__(self, **kwargs):
            self.cached_tokens = 0

        def allocate_pages(self):
            metrics["alloc_cached_pages"] += 32
    monkeypatch.setattr(exllamav3.generator, "Job", Job)
    e = ExlEngine("dsv41", [])
    e.torch = pytest.importorskip("torch")
    e.eos = []
    e.gen = SimpleNamespace(pagetable=SimpleNamespace(metrics=metrics))
    sp = SimpleNamespace(max_tokens=1, min_tokens=0, seed=None, ignore_eos=False, extra_args=None, temperature=0.0)
    r = Request("r", [1, 2], sp)
    r.job = e._make_job(r)
    r.job.allocate_pages()
    assert r.cached == 8192
    # a later final result with requeue-local zero must not overwrite the allocation measurement
    r.n_out = 1
    e._eos(r, {"eos_reason": "max_new_tokens", "cached_tokens": 0})
    assert r.cached == 8192


def test_cached_prefix_comes_from_final_job_result():
    from bigmoe.vllm.engine import ExlEngine, Request
    torch = pytest.importorskip("torch")
    e = ExlEngine("dsv41", [])
    e.torch = torch
    job = SimpleNamespace(cached_pages=32, cached_tokens=0, draft_stats=None)
    r = Request("r", list(range(16384)), SimpleNamespace(), job=job)
    e.reqs["r"] = r
    e.by_job[id(job)] = r
    results = [{"job": job, "stage": "started", "eos": False},
               {"job": job, "stage": "streaming", "eos": True, "eos_reason": "max_new_tokens",
                "token_ids": torch.tensor([[123]]), "cached_tokens": 8192, "prompt_tokens": 16384}]
    e.gen = SimpleNamespace(num_remaining_jobs=lambda: 1, iterate=lambda: results)
    _tokens, progress = e.step()["r"]
    assert r.cached == 8192 and progress.cached == 8192 and progress.finished == "length"


def test_fake_engine_progress():
    sp = SimpleNamespace(max_tokens=None)
    e = FakeEngine(chunk=4, answer=3, spec=True)
    e.add("r", list(range(10, 20)), sp)
    seen = []
    for _ in range(10):
        toks, p = e.step()["r"]
        seen.append((toks, p.computed, p.finished))
        if p.finished:
            break
    assert [s[1] for s in seen[:3]] == [4, 8, 9]                  # prefill: chunks, never the last prompt token
    assert sum((s[0] for s in seen), []) == [20, 21, 22] and seen[-1][2] == "length"
    assert seen[3] == ([20], 10, None)                             # first token: all but it computed


def test_context_room(monkeypatch):
    pytest.importorskip("vllm")
    from bigmoe import vllm as bv
    from vllm.renderers.params import TokenizeParams
    monkeypatch.setenv("BIGMOE_CTX_RESERVE", "10")
    monkeypatch.setattr(TokenizeParams, "_bigmoe_room", None, raising=False)
    orig = TokenizeParams.max_input_tokens
    try:
        bv.reserve_output_room()
        assert TokenizeParams(max_total_tokens=131072, max_output_tokens=None).max_input_tokens == 131072 - 10 - 1024
        assert TokenizeParams(max_total_tokens=131072, max_output_tokens=50).max_input_tokens == 131072 - 10 - 50
    finally:
        TokenizeParams.max_input_tokens = orig
        TokenizeParams._bigmoe_room = None


def test_scheduler_loop_over_fake_engine(monkeypatch):
    """vLLM's LLM (in-process EngineCore) with BoxScheduler + BoxWorker over the FakeEngine: chunked prefill progress,
    speculative multi-token steps, max_tokens and engine-side finishes."""
    pytest.importorskip("vllm")
    model = os.environ.get("BIGMOE_TEST_MODEL_DIR")
    if not model or not Path(model, "config.json").is_file():
        pytest.skip("BIGMOE_TEST_MODEL_DIR (a DeepSeek-V4.1 checkpoint directory) not set")
    monkeypatch.setenv("BIGMOE_VLLM_ENGINE", "fake")
    monkeypatch.setenv("BIGMOE_FAKE_ENGINE", json.dumps({"chunk": 64, "answer": 12, "spec": True, "cached": 32}))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from bigmoe.vllm import register
    register()
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, hf_overrides={"architectures": ["BigmoeDeepseekV41ForCausalLM"]},
              worker_cls="bigmoe.vllm.worker.BoxWorker", scheduler_cls="bigmoe.vllm.scheduler.BoxScheduler",
              distributed_executor_backend="uni", max_model_len=4096, max_num_seqs=4, max_num_batched_tokens=16384,
              enforce_eager=True, async_scheduling=False, enable_prefix_caching=False, tokenizer_mode="hf",
              gpu_memory_utilization=0.01, skip_mm_profiling=True)
    outs = llm.generate([{"prompt_token_ids": list(range(10, 300))}, {"prompt_token_ids": list(range(5, 50))}],
                        SamplingParams(temperature=0, max_tokens=7))
    assert [o.outputs[0].token_ids for o in outs] == [list(range(300, 307)), list(range(50, 57))]
    assert all(o.outputs[0].finish_reason == "length" and o.num_cached_tokens == 32 for o in outs)
    outs = llm.generate([{"prompt_token_ids": list(range(10, 100))}], SamplingParams(temperature=0, max_tokens=12))
    assert outs[0].outputs[0].token_ids == list(range(100, 112)) and outs[0].outputs[0].finish_reason == "length"
    assert outs[0].num_cached_tokens == 32   # engine EOS and vLLM max_tokens on the same step keep the prefix stats
    # an engine-side rejection is an error with its message, not an empty successful completion
    rejected = llm.generate([{"prompt_token_ids": [999, 1, 2]}], SamplingParams(temperature=0, max_tokens=2))
    assert rejected[0].outputs[0].finish_reason == "error"
    assert rejected[0].outputs[0].stop_reason == "rejected by the engine (test)"
