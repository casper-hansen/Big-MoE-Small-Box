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
