"""bigmoe.vllm without the GPU: the serve command line, the GLM template overlay, the ASGI middlewares, the image
placeholder splice, the context rule, the executor's chunk plan and job handling, and (with vLLM installed and
BIGMOE_TEST_MODEL_DIR set) vLLM's scheduler driving the FakeEngine: admission by page capacity, chunked prefill under the
token budget, prefix-cache hits, speculative tokens scheduled and accounted by vLLM, engine-side failures."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigmoe.vllm import serve
from bigmoe.vllm.engine import plan_chunk


def args(**kw):
    a = dict(model_dir="/m", cache_size=131072, max_batch_size=4, host="0.0.0.0", port=30000, served_name="ds",
             num_draft_tokens=None, chunk_size=8192, mtp=True, draft_model_dir=None, ngram_match_min=0,
             hybrid_draft_tokens=0, autosplit_max_batch_size=4)
    a.update(kw)
    return SimpleNamespace(**a)


def test_vllm_argv():
    argv = serve.vllm_argv("dsv41", args(), ["--enable-log-requests"])
    assert argv[:2] == ["serve", "/m"]
    get = lambda k: argv[argv.index(k) + 1]
    assert get("--max-model-len") == "131072" and get("--max-num-seqs") == "4"
    assert int(get("--max-num-batched-tokens")) == 8192 + 4 * (1 + 8)  # one chunk beside every decode + drafts
    assert get("--long-prefill-token-threshold") == "8192" and get("--scheduling-policy") == "priority"
    spec = json.loads(get("--speculative-config"))                       # vLLM schedules the engine's drafts
    assert spec["method"] == "ngram" and spec["num_speculative_tokens"] == 8
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
    one = serve.vllm_argv("dsv41", args(autosplit_max_batch_size=1), [])   # one recurrent-state slot: one sequence
    assert one[one.index("--max-num-seqs") + 1] == "1" and one[one.index("--max-num-batched-tokens") + 1] == "8201"
    plain = serve.vllm_argv("glm53", args(mtp=False, chunk_size=4096), [])
    assert "--speculative-config" not in plain and plain[plain.index("--max-num-batched-tokens") + 1] == str(4096 + 4)
    ng = serve.vllm_argv("glm53", args(mtp=False, ngram_match_min=2, num_draft_tokens=5), [])
    assert json.loads(ng[ng.index("--speculative-config") + 1])["num_speculative_tokens"] == 5


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
    # structured outputs (response_format, the tool parsers' structural tags) are honoured: grammar_filter
    validate_sampling_params(SimpleNamespace(structured_outputs=SimpleNamespace(
        all_non_structural_tag_constraints_none=lambda: False)))
    for field, value in (("logprobs", 5), ("prompt_logprobs", 1), ("allowed_token_ids", [1]), ("bad_words", ["x"]),
                         ("n", 2)):
        with pytest.raises(RequestError) as err:
            validate_sampling_params(SimpleNamespace(**{field: value}))
        assert err.value.status == 400 and field.split("_")[0] in err.value.message


def test_grammar_rows_reach_the_jobs_filter():
    """vLLM's bitmask (a stride of rows per structured request, 1 + its spec tokens used) becomes the masks the
    job's filter hands exllamav3's sampler, position by position; drafts past num_acceptable_drafts are not
    verified."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("exllamav3")
    from bigmoe.vllm.engine import Request, grammar_filter
    e = exl_engine(FakeGen())
    r = e.reqs["s"] = Request("s", [1, 2], SimpleNamespace(), grammar=grammar_filter(None))
    e.reqs["t"] = Request("t", [1, 2], SimpleNamespace())
    r.grammar.vocab_size = 64
    bm = np.arange(2 * 4 * 2, dtype=np.int32).reshape(8, 2)          # 2 requests x stride 4 rows x 2 words
    out = SimpleNamespace(structured_output_request_ids=["x", "s"], grammar_bitmask=bm, num_acceptable_drafts=[3, 1])
    limit = e._grammar_rows(out, {"s": [5, 6]})
    assert limit == {"s": 1}
    assert [m.tolist() for m in r.grammar.rows] == [[[8, 9]], [[10, 11]], [[12, 13]]]
    assert r.grammar.get_next_logit_mask().tolist() == [[8, 9]]
    e._grammar_rows(None, {})
    assert r.grammar.get_next_logit_mask().tolist() == [[-1, -1]]     # no bitmask this step: nothing masked


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


def test_plan_chunk():
    # a chunk ends on a page and inside the max chunk; the last one reaches the sequence end (its last token is
    # the sampling forward's input); decodes (one token + drafts) pass unchanged
    assert plan_chunk(0, 8192, 20000, 8192, False, None) == 8192
    assert plan_chunk(0, 8000, 20000, 8192, False, None) == 7936
    assert plan_chunk(0, 100, 20000, 8192, False, None) == 0
    assert plan_chunk(16384, 3616, 20000, 8192, False, None) == 3616            # final: through the last token
    assert plan_chunk(0, 20000, 20000, 8192, False, None) == 8192              # too long to be final in one chunk
    assert plan_chunk(19999, 6, 20000, 8192, False, None) == 6                  # decode with 5 drafts
    # recurrent state: a short last partial page is a forward of its own (engine/last_page.py), a long one is not
    assert plan_chunk(16384, 3616, 20000, 8192, True, 85) == 3584               # tail 31 tokens: split
    assert plan_chunk(19968, 32, 20000, 8192, True, 85) == 32                   # then the tail
    assert plan_chunk(16384, 3700, 20084, 8192, True, 85) == 3700               # tail 179 tokens: one forward
    assert plan_chunk(16384, 3700, 20084, 8192, True, None) == 3584             # exllamav3's stock rule: always
    assert plan_chunk(0, 64, 290, 64, False, None, page=16) == 64
    assert plan_chunk(16384, 3616, 20000, 8192, False, None, hold_last=True) == 3615   # drafting: hold the last token
    assert plan_chunk(19999, 6, 20000, 8192, False, None, hold_last=True) == 6


class FakeJob:
    def __init__(self, n=1, pages=1):
        self.sequences = [SimpleNamespace(kv_position=0, sequence_ids=[0] * 10)] * n
        self.pages, self.activated, self.allocated = pages, False, False

    def current_new_pages_required(self):
        return self.pages

    def activate(self):
        self.activated = True

    def allocate_pages(self):
        self.allocated = True

    def is_prefill_done(self):
        return False

    def prefill(self, results, chunk_size):
        assert chunk_size == 0
        self.sequences[0].kv_position = 512             # two cached pages skipped, no forward


class FakeGen:
    max_batch_size, max_chunk_size = 4, 8192
    recurrent_cache = disk_page_cache = draft_model = None
    ngram_match_min = hybrid_draft_tokens = 0

    def __init__(self, free_pages=100):
        self.pending_jobs, self.active_jobs = [], []
        self.pagetable = SimpleNamespace(num_unreferenced_pages=lambda: free_pages, max_pages=200)
        self.reaped = []

    def enqueue(self, job):
        self.pending_jobs.append(job)

    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def reap_failed_job(self, job, e, results):
        self.reaped.append(e)
        self.active_jobs.remove(job)

    def cancel(self, job):
        for q in (self.pending_jobs, self.active_jobs):
            if job in q:
                q.remove(job)


def exl_engine(gen):
    from bigmoe.vllm.engine import ExlEngine
    e = ExlEngine("dsv41", [])
    e.gen, e.ctx, e.eos = gen, 131072, []
    e.torch = pytest.importorskip("torch")
    e._make_job = lambda r: FakeJob(pages=r.prompt[0])
    return e


def vreq(rid, prompt, **sp):
    params = SimpleNamespace(max_tokens=sp.get("max_tokens"), min_tokens=0, seed=None, ignore_eos=False,
                             extra_args=sp.get("extra_args"), temperature=0.0, logprobs=None, prompt_logprobs=None,
                             structured_outputs=None, allowed_token_ids=None, bad_words=None, n=1)
    return SimpleNamespace(request_id=rid, all_token_ids=list(prompt), num_output_tokens=sp.get("out", 0),
                           sampling_params=params, mm_features=[], num_prompt_tokens=len(prompt))


def test_reserve_admits_only_with_free_pages():
    """vLLM's prefix lookup reserves the job's pages and learns the cached prefix; without free pages the request
    stays waiting with its job prepared, and is admitted once pages free up."""
    gen = FakeGen(free_pages=10)
    e = exl_engine(gen)
    assert e.reserve(vreq("a", [8, 1, 2])) == 512 and gen.active_jobs[0].allocated
    big = vreq("b", [50, 1, 2])
    assert e.reserve(big) is None and e.ensure(big) is False and len(gen.pending_jobs) == 1
    job = gen.pending_jobs[0]
    assert e.reserve(big) is None and gen.pending_jobs == [job]                # prepared once
    gen.pagetable.num_unreferenced_pages = lambda: 60
    assert e.ensure(big) is True and gen.active_jobs[-1] is job and e.cached_of("b") == 512
    e.release("a")
    assert len(gen.active_jobs) == 1 and e.reqs.keys() == {"b"}


def test_reserve_refuses_what_the_executor_cannot_run():
    gen = FakeGen()
    e = exl_engine(gen)
    bad = vreq("x", [1, 2])
    bad.sampling_params.logprobs = 5
    assert e.reserve(bad) == 0                     # admitted, so the error reaches the client after one step
    _toks, p = e.execute({"x": (0, 2, 2)}, {})["x"]
    assert p.finished == "error" and p.status == 400 and "logprobs" in p.error and not gen.active_jobs


def test_forced_output_is_constrained_after_enqueue():
    gen = FakeGen()
    e = exl_engine(gen)
    calls = []
    gen.enqueue = lambda job: calls.append("enqueue") or gen.pending_jobs.append(job)
    FakeJob.constrain_output_now = lambda self, ids: calls.append(("constrain", ids.tolist()))
    try:
        e.reserve(vreq("r", [1, 2], extra_args={"bigmoe_force": [4, 5]}))
    finally:
        del FakeJob.constrain_output_now
    assert calls == ["enqueue", ("constrain", [[4, 5]])]


def test_resumed_request_continues_its_answer():
    """A request vLLM preempted comes back with its output so far: the job's prompt is the whole sequence and only the
    rest of max_tokens is left."""
    e = exl_engine(FakeGen())
    e.reserve(vreq("r", [1, 2, 3, 70, 71], max_tokens=10, out=2))
    r = e.reqs["r"]
    assert r.prompt == [1, 2, 3, 70, 71] and r.max_new == 8


def test_oom_before_first_token_retries_with_smaller_chunks(monkeypatch):
    from bigmoe.vllm.engine import Request
    gen = FakeGen()
    e = exl_engine(gen)
    monkeypatch.setattr("bigmoe.serving.release_failed", lambda exc: None)
    r = Request("r", [1, 2, 3], SimpleNamespace())
    e.reqs["r"] = r

    class OutOfMemoryError(Exception):
        pass
    e._failed(r, OutOfMemoryError("CUDA out of memory"))
    assert gen.max_chunk_size == 4096 and r.retried and not r.finished and gen.active_jobs == [r.job]
    assert e._position(r) == 512              # the new job resumes from the cached prefix
    e._failed(r, OutOfMemoryError("CUDA out of memory"))
    assert gen.max_chunk_size == 8192 and r.finished == "error" and r.status == 503


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


def fake_llm(monkeypatch, *, chunk=64, batch=4, spec=0, policy="fcfs", **cfg):
    """vLLM's LLM (in-process EngineCore) with BoxScheduler + BoxWorker over a FakeEngine, configured as serve.py
    configures the real servers (budget: one chunk beside every decode and its drafts)."""
    pytest.importorskip("vllm")
    model = os.environ.get("BIGMOE_TEST_MODEL_DIR")
    if not model or not Path(model, "config.json").is_file():
        pytest.skip("BIGMOE_TEST_MODEL_DIR (a DeepSeek-V4.1 checkpoint directory) not set")
    monkeypatch.setenv("BIGMOE_VLLM_ENGINE", "fake")
    monkeypatch.setenv("BIGMOE_FAKE_ENGINE", json.dumps(dict(cfg, chunk=chunk, spec=bool(spec), max_batch=batch)))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from bigmoe.vllm import register, state
    register()
    state.CORRECTIONS.clear()
    from vllm import LLM
    kw = {}
    if spec:
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": spec, "prompt_lookup_min": 1,
                                    "prompt_lookup_max": 1}
    llm = LLM(model=model, hf_overrides={"architectures": ["BigmoeDeepseekV41ForCausalLM"]},
              worker_cls="bigmoe.vllm.worker.BoxWorker", scheduler_cls="bigmoe.vllm.scheduler.BoxScheduler",
              distributed_executor_backend="uni", max_model_len=4096, max_num_seqs=batch,
              max_num_batched_tokens=chunk + batch * (1 + spec), long_prefill_token_threshold=chunk,
              enforce_eager=True, async_scheduling=False, enable_prefix_caching=False, tokenizer_mode="hf",
              gpu_memory_utilization=0.01, skip_mm_profiling=True, scheduling_policy=policy, **kw)
    return llm, state.engine(), state


def prompts(*ranges):
    return [{"prompt_token_ids": list(range(a, b))} for a, b in ranges]


def test_vllm_schedules_chunks_and_drafts(monkeypatch):
    """Chunked prefill sized by vLLM's budget and long-prefill threshold (page-aligned by the plan), a second request
    admitted into the leftover budget, drafts proposed by the engine, scheduled as spec tokens and accounted by vLLM;
    the engine computes exactly what was scheduled (no corrections)."""
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, spec=2, answer=12, page=16)
    outs = llm.generate(prompts((10, 300), (5, 50)), SamplingParams(temperature=0, max_tokens=7))
    assert [o.outputs[0].token_ids for o in outs] == [list(range(300, 307)), list(range(50, 57))]
    assert all(o.outputs[0].finish_reason == "length" for o in outs)
    steps = eng.trace()
    for st in steps:
        for rid, (c, n, total, drafts) in st.items():
            # page-aligned chunks; the last one stops a token short (the first decode verifies the first drafts)
            assert c + n == total - 1 or (n <= 64 and (c + n) % 16 == 0) or c >= total - 1
            assert c + n != total or c == total - 1
    assert sum(1 for st in steps for v in st.values() if v[3]) >= 4               # spec tokens were scheduled
    assert 0 < eng.stats["accepted"] < eng.stats["verified"]                       # first draft right, second wrong
    assert len(steps) < 7 + 290 // 64 + 2                                          # drafts saved steps
    first = [st[rid] for st in steps for rid in st if st[rid][0] == st[rid][2] - 1 and st[rid][2] in (290, 45)]
    assert first and all(v[3] for v in first)              # the first decode step verifies drafts already
    assert dict(state.CORRECTIONS) == {}


def test_long_prefill_does_not_stall_a_decoding_request(monkeypatch):
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, answer=40, page=16)
    llm.generate(prompts((10, 30), (100, 900)), SamplingParams(temperature=0, max_tokens=40))
    steps = eng.trace()
    a, b = (sorted({rid for st in steps for rid in st}, key=lambda r: int(r.split("-")[0])))
    both = [st for st in steps if b in st and st[b][0] < st[b][2] - 1]           # b prefilling
    assert both and all(a in st for st in both if any(a in s2 for s2 in steps[steps.index(st):]))
    assert all(st[b][1] <= 64 for st in both)                                      # one chunk per step at most
    assert dict(state.CORRECTIONS) == {}


def test_admission_waits_for_pages(monkeypatch):
    """With room for one request's pages, vLLM admits the others only as pages are released."""
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, answer=8, page=16, pages=10)
    outs = llm.generate(prompts((10, 110), (200, 300), (400, 500)), SamplingParams(temperature=0, max_tokens=8))
    assert [o.outputs[0].token_ids for o in outs] == [list(range(110, 118)), list(range(300, 308)),
                                                     list(range(500, 508))]
    assert all(len(st) == 1 for st in eng.trace()) and eng.stats["no_pages"] > 0
    assert dict(state.CORRECTIONS) == {}


def test_prefix_hits_are_vllms_cached_tokens(monkeypatch):
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, answer=4, page=16, cached=32)
    first = llm.generate(prompts((10, 100)), SamplingParams(temperature=0, max_tokens=4))
    again = llm.generate([{"prompt_token_ids": list(range(10, 42)) + [7] * 50}],
                         SamplingParams(temperature=0, max_tokens=4))
    assert first[0].num_cached_tokens == 0 and again[0].num_cached_tokens == 32
    second = [st for st in eng.trace() if len(st) == 1 and list(st.values())[0][2] == 82]
    assert list(second[0].values())[0][:2] == (32, 50)                            # vLLM starts after the hit
    assert dict(state.CORRECTIONS) == {}


def test_engine_side_errors(monkeypatch):
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, answer=4, page=16)
    # an engine-side rejection is an error with its message, not an empty successful completion
    rejected = llm.generate([{"prompt_token_ids": [999, 1, 2]}], SamplingParams(temperature=0, max_tokens=2))
    assert rejected[0].outputs[0].finish_reason == "error"
    assert rejected[0].outputs[0].stop_reason == "rejected by the engine (test)"
    ok = llm.generate(prompts((10, 20)), SamplingParams(temperature=0, max_tokens=3))
    assert ok[0].outputs[0].token_ids == [20, 21, 22]


class PagedJob:
    """Job.prefill's rule: a forward ends at min(page-aligned max-chunk window, the last token), and a chunk_size
    bounds it to whole pages."""
    def __init__(self, n, chunk=8192):
        self.sequences = [SimpleNamespace(kv_position=0, sequence_ids=[0] * n)]
        self.chunk, self.calls = chunk, []

    def current_new_pages_required(self):
        return 1

    def activate(self):
        pass

    def allocate_pages(self):
        pass

    def is_prefill_done(self):
        s = self.sequences[0]
        return s.kv_position == len(s.sequence_ids) - 1

    def prefill(self, results, chunk_size):
        s = self.sequences[0]
        start, last = s.kv_position, len(s.sequence_ids) - 1
        end = min((start + self.chunk) // 256 * 256, last)
        if chunk_size is not None:
            end = min(end, (start + chunk_size) // 256 * 256)
        self.calls.append(chunk_size)
        s.kv_position = max(start, end)


def test_executor_follows_the_plan_through_the_last_chunk():
    """Each planned chunk (plan_chunk, drafting: the last one stops a token short) is exactly what Job.prefill
    computes; the request then waits for its first verify step."""
    gen = FakeGen()
    gen.sample_calls = []
    e = exl_engine(gen)
    e._make_job = lambda r: PagedJob(len(r.prompt))
    e.reserve(vreq("r", list(range(700))))
    c = 0
    for _ in range(4):
        n = plan_chunk(c, 600, 700, 8192, True, 85, hold_last=True) if c < 699 else 0
        if not n:
            break
        _toks, p = e.execute({"r": (c, n, 700)}, {})["r"]
        assert p.computed == c + n and not p.finished
        c += n
    assert c == 699 and e.reqs["r"].job.is_prefill_done()


def test_executor_fails_a_request_that_cannot_progress():
    gen = FakeGen()
    e = exl_engine(gen)
    e._make_job = lambda r: PagedJob(len(r.prompt))
    e.reserve(vreq("r", list(range(700))))
    from bigmoe.vllm import engine as E
    for _ in range(E.STALL_STEPS):
        _toks, p = e.execute({"r": (0, 100, 700)}, {})["r"]     # never a whole page: Job.prefill computes nothing
    assert p.finished == "error" and "no progress" in p.error


def test_priority_orders_the_waiting_queue(monkeypatch):
    """One sequence at a time (as DeepSeek's single recurrent slot): queued requests start by priority (lower
    first), equal priorities in arrival order."""
    from vllm import SamplingParams
    llm, eng, state = fake_llm(monkeypatch, chunk=64, batch=1, answer=4, page=16, policy="priority")
    outs = llm.generate(prompts((10, 40), (100, 130), (200, 230), (300, 330)),
                        SamplingParams(temperature=0, max_tokens=4), priority=[0, 5, -3, 0])
    first = []
    for st in eng.trace():
        for rid in st:
            if rid not in first:
                first.append(rid)
    order = [int(rid.split("-")[0]) for rid in first]
    assert order == [2, 0, 3, 1] and all(len(o.outputs[0].token_ids) == 4 for o in outs)


def test_a_failed_sampling_forward_ends_its_requests_not_the_engine():
    from bigmoe.vllm.engine import Request
    gen = FakeGen()
    e = exl_engine(gen)
    job = FakeJob()
    r = e.reqs["r"] = Request("r", [1, 2], SimpleNamespace(), job=job)
    e.by_job[id(job)] = r
    gen.active_jobs.append(job)
    gen._draft_conf_round = None

    def boom(results, draft_tokens):
        raise AssertionError("speculative pass of 4 tokens x 1 rows exceeds the cache's max_history 0")
    gen.iterate_gen = boom
    gen.reap_failed_job = lambda job, err, results: (gen.active_jobs.remove(job),
                                                     results.append({"job": job, "stage": "error", "error": err}))
    results = []
    e._sample([r], {}, results)
    e._results(results)
    assert r.finished == "error" and "max_history" in r.error and not gen.active_jobs
