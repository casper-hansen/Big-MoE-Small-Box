"""bigmoe.serving: request hardening shared by the servers (bigmoe/vllm). No GPU."""
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
