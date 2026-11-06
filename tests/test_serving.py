"""bigmoe.serving: request hardening shared by the model servers. No GPU; the FastAPI part skips without fastapi."""
import asyncio
import json
import re

import pytest

from bigmoe import serving
from bigmoe.serving import RequestError

# what DeepSeek Harness (dsh-llm isContextWindowExceededError) matches to compact the conversation and retry
DSH_OVERFLOW = [re.compile(r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-](?:exceed(?:ed|s)?|overflow(?:ed)?|"
                           r"limit[\s_-]exceeded)(?:$|[^a-z0-9])", re.I),
                re.compile(r"\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b", re.I)]


def test_output_budget_clamps_to_the_room_left():
    assert serving.output_budget(1000, 65536, 10) == 65536 - 1010
    assert serving.output_budget(1000, 65536, 10, 500) == 500
    assert serving.output_budget(60000, 65536, 10, 32768) == 65536 - 60010
    assert serving.output_budget(65000, 65536, 10, 100) == 100, "a small max_tokens only needs that much room"


def test_output_budget_refuses_prompts_that_do_not_fit():
    for prompt, max_tokens in ((74000, 32768), (65530, None), (65000, 4096), (64600, None)):
        with pytest.raises(RequestError) as e:
            serving.output_budget(prompt, 65536, 10, max_tokens)
        err = e.value
        assert err.status == 400 and err.code == "context_length_exceeded"
        assert "65536" in err.message and str(prompt) in err.message
        detail = " ".join([err.code, err.type, err.message])
        assert all(r.search(detail) for r in DSH_OVERFLOW), detail


def test_output_budget_validates_max_tokens():
    for bad in ("many", 0, -5):
        with pytest.raises(RequestError) as e:
            serving.output_budget(10, 1000, 0, bad)
        assert e.value.status == 400 and e.value.param == "max_tokens"


def test_exllamav3_enqueue_assertion_becomes_a_context_error():
    exc = AssertionError("Job requires 291 pages (only 256 available) and cannot be enqueued. Total cache allocated "
                         "is 256 * 256 = 65536 tokens")
    err = serving.as_request_error(exc, 74300, 65536)
    assert (err.status, err.code) == (400, "context_length_exceeded") and "74300" in err.message
    assert serving.as_request_error(exc).code == "context_length_exceeded"
    oom = serving.as_request_error(RuntimeError("CUDA out of memory. Tried to allocate 276.00 MiB"))
    assert oom.status == 503
    other = serving.as_request_error(ValueError("boom"))
    assert (other.status, other.type, other.message) == (500, "server_error", "ValueError: boom")


def test_usage_reports_cached_tokens_and_timings():
    eos = {"time_enqueued": 0.002, "time_prefill": 14.712, "time_generate": 3.5, "cached_pages": 55,
           "cached_tokens": 14080, "new_tokens": 63, "accepted_draft_tokens": 40, "eos": True}
    u = serving.usage(14197, 63, {"type": "stop", "timings": serving.job_timings(eos)})
    assert u["prompt_tokens_details"] == {"cached_tokens": 14080}
    assert u["timings"] == {"queue_ms": 2.0, "prefill_ms": 14712.0, "decode_ms": 3500.0, "cached_tokens": 14080,
                            "new_tokens": 63, "accepted_draft_tokens": 40}
    assert serving.usage(10, 2) == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_guard_stream_turns_a_late_failure_into_an_error_event():
    async def chunks():
        yield "data: {\"a\": 1}\n\n"
        raise AssertionError("Job requires 300 pages (only 256 available) and cannot be enqueued.")

    async def collect():
        return [c async for c in serving.guard_stream(chunks())]
    out = asyncio.run(collect())
    assert out[0] == "data: {\"a\": 1}\n\n" and out[-1] == "data: [DONE]\n\n"
    assert json.loads(out[1][len("data: "):])["error"]["code"] == "context_length_exceeded"


class FakeTask:
    def __init__(self, done=False, exc=None):
        self._done, self._exc = done, exc

    def done(self):
        return self._done

    def cancelled(self):
        return False

    def exception(self):
        return self._exc


class FakeGen:
    def __init__(self, error=None, task=None):
        self.error, self.iteration_task = error, task or FakeTask()


def test_generator_failure():
    assert serving.generator_failure(None) is None
    assert serving.generator_failure(FakeGen()) is None
    boom = RuntimeError("CUDA error: an illegal memory access was encountered")
    assert serving.generator_failure(FakeGen(error=boom)) is boom
    assert serving.generator_failure(FakeGen(task=FakeTask(done=True, exc=boom))) is boom


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    app = FastAPI()
    state = {"gen": None}
    serving.install(app, lambda: state["gen"], "test", exit_on_dead=False)

    @app.post("/chat")
    async def chat(req: Request):
        body = await serving.read_json(req, require=("messages",))
        serving.output_budget(int(body["prompt_tokens"]), 65536, 10, body.get("max_tokens"))
        return {"ok": True}

    @app.get("/crash")
    async def crash():
        raise RuntimeError("something broke")

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, state


def test_health_reflects_the_generator(client):
    c, state = client
    assert c.get("/health").status_code == 503 and c.get("/health").json()["status"] == "starting"
    state["gen"] = FakeGen()
    assert c.get("/health").status_code == 200 and c.get("/health").json() == {"ok": True}
    state["gen"] = FakeGen(error=RuntimeError("CUDA out of memory"))
    r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "generator dead" and "CUDA" in r.json()["error"]


def test_errors_are_openai_shaped_json(client):
    c, _ = client
    r = c.post("/chat", json={"messages": [], "prompt_tokens": 74000, "max_tokens": 32768})
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert c.post("/chat", json={"messages": [], "prompt_tokens": 100}).json() == {"ok": True}
    r = c.post("/chat", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "not valid JSON" in r.json()["error"]["message"]
    r = c.post("/chat", json={"prompt_tokens": 1})
    assert r.status_code == 400 and r.json()["error"]["param"] == "messages"
    r = c.get("/crash")
    assert r.status_code == 500 and r.json()["error"] == {"message": "RuntimeError: something broke",
                                                          "type": "server_error", "param": None, "code": None}


def test_prefix_cache_tiers():
    class Args:
        cpu_cache_size, disk_cache_dir, disk_cache_size = 1.0, "/pc", 50.0
    assert serving.prefix_cache_tiers(Args) == {"cpu_cache_size": 1024 ** 3, "disk_cache_dir": "/pc",
                                                "disk_cache_size": 50 * 1024 ** 3, "disk_cache_write_interval": 0.0}
    Args.cpu_cache_size, Args.disk_cache_dir = None, None   # exllamav3's -ccs default when the option is not passed
    assert serving.prefix_cache_tiers(Args)["cpu_cache_size"] == 0


def test_gpu_memory_stats():
    torch = pytest.importorskip("torch")
    s = serving.gpu_memory_stats()
    if not torch.cuda.is_available():
        assert s is None
    else:
        assert set(s) == {"allocated_mib", "reserved_mib", "peak_allocated_mib", "free_mib", "total_mib"}
        assert 0 <= s["free_mib"] <= s["total_mib"]


class OutOfMemoryError(RuntimeError):   # stands in for torch.OutOfMemoryError (matched by name)
    pass


class FakeAsyncJob:
    def __init__(self, script):
        self.script, self.cancelled = script, False

    async def __aiter__(self):
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            yield item

    async def cancel(self):
        self.cancelled = True


def collect(rj):
    async def run():
        return [r async for r in rj]
    return asyncio.run(run())


STREAM = {"stage": "streaming", "text": "ok"}


def test_retrying_job_retries_oom_before_the_first_token():
    gen = type("G", (), {"max_chunk_size": 8192})()
    chunks, logs = [], []
    jobs = [FakeAsyncJob([OutOfMemoryError("CUDA out of memory")]), FakeAsyncJob([{"stage": "prefill"}, STREAM])]

    def remake():
        chunks.append(gen.max_chunk_size)
        return jobs.pop(0)
    rj = serving.RetryingJob(jobs.pop(0), remake, gen, log=logs.append)
    assert collect(rj) == [{"stage": "prefill"}, STREAM]
    assert chunks == [4096] and gen.max_chunk_size == 8192 and rj.retried and "4096-token" in logs[0]


def test_retrying_job_gives_up():
    gen = type("G", (), {"max_chunk_size": 8192})()
    twice = [FakeAsyncJob([OutOfMemoryError("CUDA out of memory")]), FakeAsyncJob([OutOfMemoryError("again")])]
    rj = serving.RetryingJob(twice.pop(0), lambda: twice.pop(0), gen, log=lambda m: None)
    with pytest.raises(OutOfMemoryError):
        collect(rj)
    assert gen.max_chunk_size == 8192            # restored after the failed retry
    late = serving.RetryingJob(FakeAsyncJob([STREAM, OutOfMemoryError("mid-stream")]), lambda: 1 / 0, gen)
    with pytest.raises(OutOfMemoryError):        # tokens were streamed: no retry
        collect(late)
    other = serving.RetryingJob(FakeAsyncJob([ValueError("bad")]), lambda: 1 / 0, gen)
    with pytest.raises(ValueError):
        collect(other)


def test_release_failed_clears_the_failed_frames():
    pytest.importorskip("torch")

    def forward():
        activations = bytearray(10)   # what a failed forward holds
        raise OutOfMemoryError("CUDA out of memory")
    try:
        forward()
    except OutOfMemoryError as e:
        err = e
    tb = err.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert "activations" in tb.tb_frame.f_locals
    serving.release_failed(err)
    assert "activations" not in tb.tb_frame.f_locals
