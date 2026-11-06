"""engine/tiers.py: a forward that raises inside an elastic prefill hands the lent VRAM back right away."""
import pytest

torch = pytest.importorskip("torch")
from bigmoe.engine import tiers


class FakePool:
    elastic, stage_n, topk, stage_min = 1.0, 384, 6, 512

    def __init__(self):
        self.in_prefill, self.calls = False, []

    def _elastic_enter(self, n):
        self.in_prefill = True
        self.calls.append(("enter", n))

    def _elastic_exit(self):
        self.in_prefill = False
        self.calls.append(("exit",))


def test_failed_prefill_exits_elastic_mode(monkeypatch):
    class Model:
        def prefill(self, input_ids, params=None):
            raise torch.OutOfMemoryError("CUDA out of memory")

        def forward(self, input_ids, params=None):
            return "ok"

    pool = FakePool()
    monkeypatch.setattr(tiers, "POOLS", [pool])
    tiers._hook_prefill(Model)
    with pytest.raises(torch.OutOfMemoryError):
        Model().prefill(torch.zeros((1, 4096), dtype=torch.long))
    assert pool.calls == [("enter", 4096), ("exit",)] and not pool.in_prefill
    assert Model().forward(torch.zeros((1, 1), dtype=torch.long)) == "ok"   # a single token does not stage
    assert pool.calls == [("enter", 4096), ("exit",)]
