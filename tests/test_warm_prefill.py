"""engine/warm_prefill.py selection and masking (CPU tensors, no GPU) and bench/prefill.py prompt construction."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bigmoe.engine import warm_prefill as warm  # noqa: E402
from bigmoe.bench import prefill  # noqa: E402


def test_cold_set_is_neither_vram_nor_ram():
    slot = np.array([[5, -1, -1, -1]], dtype=np.int32)       # expert 0 in VRAM
    kst = np.array([0, 2, 1, 0], dtype=np.int8)               # expert 1 in RAM, 2 in flight, 3 on NVMe only
    assert warm.cold_set(slot, kst).tolist() == [[False, False, True, True]]


def test_rank_cold_by_summed_weight():
    cold = torch.tensor([True, True, False, True, True])
    sel = torch.tensor([[0, 2], [1, 3], [3, 2]])
    w = torch.tensor([[0.5, 0.5], [0.2, 0.8], [0.3, 0.7]])
    # cold weights: 0 -> 0.5, 1 -> 0.2, 3 -> 1.1, 4 -> never picked; 2 is warm
    assert warm.rank_cold(sel, w, cold, 0, 8).tolist() == [3, 0, 1]
    assert warm.rank_cold(sel, w, cold, 0, 2).tolist() == [3, 0]
    assert warm.rank_cold(sel, w, cold, 0, 0).tolist() == []
    assert warm.rank_cold(sel + 10, w, cold, 10, 1).tolist() == [3]   # routing_first offset


def test_drop_cold_sentinel_and_renormalization():
    cold = torch.tensor([False, True, False, False])
    sel = torch.tensor([[0, 1], [2, 3], [1, 1]])
    w = torch.tensor([[0.6, 0.2], [0.5, 0.5], [0.3, 0.3]], dtype=torch.half)
    s2, w2, m = warm.drop_cold(sel, w, cold, 0)
    assert s2.tolist() == [[0, 4], [2, 3], [4, 4]]            # sentinel = first + E, never run by the MoE kernel
    assert m.tolist() == [[False, True], [False, False], [True, True]]
    assert w2.dtype == torch.half
    assert torch.allclose(w2.float(), torch.tensor([[0.8, 0.0], [0.5, 0.5], [0.0, 0.0]]), atol=1e-3)
    s3, _, _ = warm.drop_cold(sel + 7, w, cold, 7)
    assert s3[0, 1].item() == 11


class _Result:
    def __init__(self, fn=None):
        self.fn, self.calls = fn, 0

    def result(self):
        self.calls += 1
        if self.fn is not None:
            self.fn()


class _Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, ev):
        self.waits.append(ev)

    def wait_stream(self, stream):
        self.waits.append(stream)


def _ahead(wp, *, done, inflight, queue, closed=False, layer=1, b=0, used=4):
    import threading
    a = object.__new__(warm.Ahead)
    a.layer, a.b, a.used = layer, b, used
    a.done, a.inflight, a.queue, a.closed = dict(done), dict(inflight), list(queue), closed
    a.lock, a.event = threading.Lock(), object()
    a.fut = _Result()
    wp.ahead = a
    return a


def _prefill_for_take():
    wp = object.__new__(warm.WarmPrefill)
    wp.st = {"ahead_read": 0, "ahead_hit": 0, "ahead_wasted": 0, "ahead_wait_ms": 0.0}
    wp.stream = _Stream()
    return wp


def test_take_ahead_retargets_unread_guesses_and_fills_each_true_expert_once():
    wp = _prefill_for_take()
    a = _ahead(wp, done={1: 4}, inflight={2: 5}, queue=[(3, 6), (9, 7)])

    def finish():
        # the worker consumes the retargeted queue, not the guesses discarded by the consumer
        a.done.update(a.inflight)
        a.done.update(a.queue)
        a.inflight, a.queue, a.closed = {}, [], True
    a.fut.fn = finish
    cur = _Stream()
    pairs = wp._take_ahead(1, 0, 4, 3, [2, 4, 1], cur)
    assert pairs == [(2, 5), (1, 4), (4, 6)]
    assert a.queue == [] and a.closed
    assert wp.st["ahead_read"] == 2 and wp.st["ahead_hit"] == 2 and wp.st["ahead_missed"] == 1
    assert cur.waits == [a.event, wp.stream]


def test_take_ahead_reads_misses_itself_after_the_worker_closed():
    wp = _prefill_for_take()
    a = _ahead(wp, done={1: 4, 2: 5}, inflight={}, queue=[], closed=True)
    read = []
    wp._read_into = lambda li, b, pairs: read.append((li, b, pairs))
    pairs = wp._take_ahead(1, 0, 4, 3, [3, 1], _Stream())
    assert pairs == [(1, 4), (3, 5)] and read == [(1, 0, [(3, 5)])]   # an unused guess's slot is reused
    assert wp.st["ahead_hit"] == 1 and wp.st["ahead_wasted"] == 1 and wp.st["ahead_missed"] == 1


def test_drop_ahead_discards_unread_work_and_orders_buffer_reuse_after_inflight(monkeypatch):
    import types
    wp = _prefill_for_take()
    wp.pool = types.SimpleNamespace(device="cpu")
    a = _ahead(wp, done={1: 4}, inflight={2: 5}, queue=[(3, 6)])
    cur = _Stream()
    monkeypatch.setattr(warm.torch.cuda, "current_stream", lambda device: cur)
    wp.drop_ahead()
    assert a.queue == [] and a.fut.calls == 1 and wp.ahead is None
    assert cur.waits == [a.event]       # Future completion means copies queued; the event orders their completion.


def test_before_stage_drains_only_a_conflicting_buffer(monkeypatch):
    import types
    wp = _prefill_for_take()
    wp.pool = types.SimpleNamespace(device="cpu")
    cur = _Stream()
    monkeypatch.setattr(warm.torch.cuda, "current_stream", lambda device: cur)
    a = _ahead(wp, done={}, inflight={2: 5}, queue=[], b=1)
    wp.before_stage(0)                         # normal sequential layer: other parity, no synchronization
    assert wp.ahead is a and a.fut.calls == 0 and cur.waits == []
    wp.before_stage(1)                         # out-of-order reuse: finish it before cstream can be submitted
    assert wp.ahead is None and a.fut.calls == 1 and cur.waits == [a.event]


def test_mismatched_ahead_still_finishes_old_copies_before_returning():
    wp = _prefill_for_take()
    a = _ahead(wp, done={1: 4}, inflight={2: 5}, queue=[(3, 6)], layer=9)
    cur = _Stream()
    assert wp._take_ahead(1, 0, 4, 3, [], cur) is None
    assert a.queue == [] and a.fut.calls == 1 and cur.waits == [a.event]


def test_begin_always_orders_new_staging_after_the_warm_stream(monkeypatch):
    import types
    wp = _prefill_for_take()
    wp.ahead = None
    wp.pool = types.SimpleNamespace(device="cpu")
    wp.st["forwards"] = 0
    cur = _Stream()
    monkeypatch.setattr(warm.torch.cuda, "current_stream", lambda device: cur)
    ext = types.SimpleNamespace(state=lambda: (None, torch.tensor([0, 0], dtype=torch.int8)))
    out = wp.begin(ext, np.array([[5, -1]], dtype=np.int32))
    assert out.tolist() == [[5, 0]] and cur.waits == [wp.stream]

    # A failed worker must not skip the recovery barrier either.
    a = _ahead(wp, done={}, inflight={2: 5}, queue=[])
    a.fut.fn = lambda: (_ for _ in ()).throw(RuntimeError("read failed"))
    with pytest.raises(RuntimeError, match="read failed"):
        wp.begin(ext, np.array([[5, -1]], dtype=np.int32))
    assert cur.waits[-1] == wp.stream and wp.ahead is None


def test_start_ahead_uses_next_layers_router_and_allows_a_wider_guess_window(monkeypatch):
    import types
    wp = _prefill_for_take()
    wp.cold_np = np.ones((2, 5), dtype=bool)
    wp.cold = torch.ones((2, 5), dtype=torch.bool)
    wp.st["ahead_guessed"] = 0
    calls = []

    def route(bsz, cfg, z, params):
        calls.append((bsz, cfg, z, params))
        return torch.tensor([[0, 1], [2, 3]]), torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    module = types.SimpleNamespace(_bigmoe_route=route, routing_cfg="cfg")
    pool = types.SimpleNamespace(pending=(1, 1, np.array([2]), None, None), stage_n=10, first=0,
                                 mods=[None, module], stage=[object(), object()])
    wp.pool = pool
    submitted = []
    wp.ahead_pool = types.SimpleNamespace(submit=lambda job: submitted.append(job) or _Result())

    class FakeAhead:
        def __init__(self, layer, b, used, queue):
            self.layer, self.b, self.used, self.queue = layer, b, used, queue
    monkeypatch.setattr(warm, "Ahead", FakeAhead)
    monkeypatch.setattr(warm, "AHEAD_N", 3)
    wp._start_ahead(1, "hidden", {"p": 1}, 2, object())
    assert calls == [(2, "cfg", "hidden", {"p": 1})]
    assert wp.ahead.queue == [(1, 1), (2, 2), (3, 3)]       # predicted weights .8, .7, .3; RAM expert used slot 0
    assert len(submitted) == 1 and wp.st["ahead_guessed"] == 3


def test_route_starts_read_ahead_only_at_the_configured_chunk_size(monkeypatch):
    import types
    wp = _prefill_for_take()
    wp.ahead = None
    wp.cold = torch.zeros((2, 5), dtype=torch.bool)
    wp.acc = torch.zeros(4, dtype=torch.float64)
    wp.pool = types.SimpleNamespace(device="cpu", staged={}, stage_n=100, first=0, mods=[None, None])
    started = []
    wp._start_ahead = lambda *args: started.append(args)
    monkeypatch.setattr(warm, "AHEAD", True)
    monkeypatch.setattr(warm, "AHEAD_MIN", 1024)
    monkeypatch.setattr(warm.torch.cuda, "current_stream", lambda device: _Stream())

    class Event:
        def record(self, stream):
            pass
    monkeypatch.setattr(warm.torch.cuda, "Event", Event)
    sel = torch.tensor([[0, 1]])
    w = torch.tensor([[0.5, 0.5]])
    wp.route(0, sel, w, z=torch.zeros(1), params={}, bsz=1023)
    assert started == []
    wp.route(0, sel, w, z=torch.zeros(1), params={}, bsz=1024)
    assert len(started) == 1 and started[0][0] == 1


class FakeClient:
    def tokenize(self, text):
        return [ord(c) for c in text]


def test_prefill_prompts_share_the_prefix_and_nothing_else(tmp_path):
    pkg = tmp_path / "exllamav3"
    pkg.mkdir()
    (pkg / "a.py").write_text("x = 1\n" * 20000)
    warmup, reqs = prefill.prompts(FakeClient(), tmp_path, (8, 20, 40), prefix=512, nonce="n1")
    pre = warmup[:512]
    assert len(warmup) == 513 and pre[:20] == FakeClient().tokenize("[prefill benchmark n1]\n")[:20]
    tails = [warmup[512:]]
    for n, ids in reqs.items():
        assert len(ids) == 512 + n and ids[:512] == pre
        tails.append(ids[512:])
    offsets = [0, 8, 28, 68]
    body = FakeClient().tokenize(prefill.corpus(10 ** 6, tmp_path))[512 - len("[prefill benchmark n1]\n"):]
    assert tails[1:] == [body[a:b] for a, b in zip(offsets, offsets[1:])] and tails[0] == body[68:69]


def test_prefill_compare_runs():
    a = {"runs": {"8": {"ttft_s": 1.0}, "200": {"ttft_s": 15.0}}}
    b = {"runs": {"8": {"ttft_s": 1.0}, "200": {"ttft_s": 4.5}}}
    rows = prefill.compare_runs(a, b)
    assert len(rows) == 2 and "-70%" in rows[1]
