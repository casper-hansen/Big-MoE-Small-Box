"""Warm prefill: staged forwards run on the experts that are already in VRAM or RAM, plus the few NVMe-only experts that
matter most to each layer of the chunk.

A staged forward (a prefill chunk, >= EC_STAGE_MIN picks per layer) used to stage every expert of every layer, which
reads every NVMe-only expert once: ~15 s per forward on a 3090 box whatever the chunk's length, the floor of every
agent step that adds a tool result. Here the forward starts from the warm set (VRAM-resident and RAM-tier experts at
its start). Once a layer's routing is known, its cold (NVMe-only) experts are ranked by the routing weight the chunk
gives them; the top FETCH_PER_LAYER are read from the store in parallel into free slots of the layer's staging buffer
and run exactly. The picks of the remaining cold experts are dropped (sentinel id, weight 0) and each token's weights
are rescaled to their original sum. Decode is unchanged.

Against exact prefill: DeepSeek-V4.1-Flash on five steps of a DeepSeek Harness agent turn (14k to 36k tokens, decode
exact) prefills in 77.5 s instead of 116.9 s with top-1 agreement 99.7 % and mean KL 0.0021 on 330 forced answer
positions; teacher-forced (prompt and answer through warm prefill) DeepSeek keeps 100 % top-1 (mean KL 0.023) and
GLM-5.3-Flash 94.5 % (0.042). Mode nvme only, as the backends' warm_prefill option (DeepSeek: on by default,
GLM: off); mode nvme-exact prefills exactly.
"""
import concurrent.futures as cf
import time

import numpy as np
import torch

FETCH_PER_LAYER = 64      # cold experts read per layer of a staged forward (64 of 384: ~7 % of the routing weight left)
READERS = 16              # parallel store reads (the NVMe needs a deep queue for its full bandwidth)
BATCH = 16                # experts per read batch; two batches of pinned slots alternate (read one, copy the other)
RAM_RESIDENT = 2          # nv2 host state of an expert in the RAM tier (KS_RES)


def cold_set(slot_np: np.ndarray, kst: np.ndarray) -> np.ndarray:
    """[L, E] bool: experts neither in VRAM (slot >= 0) nor resident in the RAM tier."""
    return (slot_np < 0) & (kst.reshape(slot_np.shape) != RAM_RESIDENT)


def rank_cold(sel: torch.Tensor, w: torch.Tensor, cold: torch.Tensor, first: int, limit: int) -> torch.Tensor:
    """The (at most `limit`) cold experts of one layer call with the largest routing weight summed over its tokens, by
    weight, as local expert ids. sel / w: [tokens, top_k]; cold: [E] bool."""
    E = cold.numel()
    loc = sel.reshape(-1).long() - first
    valid = (loc >= 0) & (loc < E)
    score = torch.zeros(E, dtype=torch.float32, device=sel.device)
    score.index_add_(0, loc.clamp(0, E - 1), w.reshape(-1).float() * valid)
    score = torch.where(cold, score, torch.zeros_like(score))
    top = torch.topk(score, min(limit, E))
    return top.indices[top.values > 0]


def drop_cold(sel: torch.Tensor, w: torch.Tensor, cold: torch.Tensor, first: int):
    """Routing without the cold experts: their picks get the sentinel id first + E (exllamav3's MoE counts it in its
    out-of-slice bucket and never runs it) and weight 0; each token's remaining weights are rescaled to the token's
    original sum. Returns (sel, w, dropped mask)."""
    E = cold.numel()
    loc = (sel.long() - first).clamp(0, E - 1)
    m = cold[loc]
    wf = w.float()
    kept = wf.masked_fill(m, 0.0)
    total, left = wf.sum(-1, keepdim=True), kept.sum(-1, keepdim=True)
    kept = kept * torch.where(left > 0, total / left.clamp_min(1e-12), torch.ones_like(left))
    return sel.masked_fill(m, first + E), kept.to(w.dtype), m


class WarmPrefill:
    """Per Pool. Created before nv2 sizes the RAM tier, so the pinned read slots are already charged to the cgroup."""

    def __init__(self, pool):
        from .tiers import HostArena
        self.pool = pool
        self.slot = -(-pool.rec // 4096) * 4096                  # O_DIRECT reads: 4 KiB-aligned slots
        dev = pool.device.index or 0
        self.arena = HostArena(2 * BATCH, self.slot, BATCH, dev, "warm prefill")
        self.readers = cf.ThreadPoolExecutor(READERS, thread_name_prefix="warm-read")
        self.stream = torch.cuda.Stream(pool.device)             # H2D of fetched experts, overlapping the next reads
        self.copied = [torch.cuda.Event() for _ in range(2)]     # per batch of slots: its last H2D is done
        self.cold = None
        self.acc = torch.zeros(4, dtype=torch.float64, device=pool.device)   # picks, dropped, weight, dropped weight
        self.st = {"forwards": 0, "fetched": 0, "fetched_bytes": 0, "read_ms": 0.0}

    def begin(self, ext, slot_np: np.ndarray) -> np.ndarray:
        """Start of a staged forward: the cold set of every layer, and the residency snapshot for stage_begin /
        stage_submit with the cold experts marked resident, so they are neither read nor staged. Their pointer-table
        rows are never used: route() fetches or drops every pick of theirs."""
        cold = cold_set(slot_np, ext.state()[1].numpy())
        self.cold = torch.from_numpy(cold).to(self.pool.device)
        self.st["forwards"] += 1
        return np.where(cold, 0, slot_np).astype(slot_np.dtype)

    def route(self, li: int, sel: torch.Tensor, w: torch.Tensor):
        """One layer call of the staged forward: fetch the heaviest cold experts, drop the other cold picks."""
        if self.cold is None:
            return sel, w
        p = self.pool
        b, staged = p.staged.get(li, (li % 2, None))
        used = 0 if staged is None else int(staged.numel())
        fetch = rank_cold(sel, w, self.cold[li], p.first, min(FETCH_PER_LAYER, p.stage_n - used)).cpu().tolist()
        if fetch:
            self._fetch(li, b, used, staged, fetch)
        sel2, w2, m = drop_cold(sel, w, self.cold[li], p.first)
        wf = w.float()
        self.acc[0] += m.numel()
        self.acc[1] += m.sum()
        self.acc[2] += wf.sum()
        self.acc[3] += (wf * m).sum()
        return sel2, w2

    def _fetch(self, li, b, used, staged, experts):
        """Read `experts` of layer li into staging buffer b after its `used` warm slots and point the tables at them."""
        from . import nv2 as N2
        p, ext, A = self.pool, N2.ext(), self.arena
        cur = torch.cuda.current_stream(p.device)
        free = torch.cuda.Event()
        free.record(cur)                       # the buffer's previous layer (li - 2) has run its MoE by this point
        self.stream.wait_event(free)
        stage, keys = p.stage[b], [li * p.E + e for e in experts]
        t0 = time.perf_counter()
        for c in range(0, len(keys), BATCH):
            half = (c // BATCH) % 2
            self.copied[half].synchronize()    # these pinned slots' previous H2D is done
            batch = keys[c:c + BATCH]
            slots = [half * BATCH + i for i in range(len(batch))]
            got = list(self.readers.map(lambda ks: ext.read_sync(ks[0], int(A.addr[ks[1]])), zip(batch, slots)))
            with torch.cuda.stream(self.stream):
                for i, (key, s) in enumerate(zip(batch, slots)):
                    n = int(p.nv2.klen[key])
                    if got[i] != n:
                        raise RuntimeError(f"warm prefill: short read of expert {key} ({got[i]} of {n} bytes)")
                    stage[used + c + i, :n].copy_(A.view(s, n), non_blocking=True)
                    self.st["fetched_bytes"] += n
            self.copied[half].record(self.stream)
        cur.wait_stream(self.stream)
        self.st["read_ms"] += (time.perf_counter() - t0) * 1e3
        self.st["fetched"] += len(keys)
        idx = torch.tensor(experts, dtype=torch.long, device=p.device)
        a = stage.data_ptr() + (used + torch.arange(len(experts), dtype=torch.int64, device=p.device)) * p.rec
        _, _, _, ou, od = p.g[li]
        pg, pu, pd, gu = p.keep[li]
        pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + ou); pd.index_copy_(0, idx, a + od)
        if gu is not None:
            gu.index_copy_(0, idx, torch.stack([a, a + ou], 1))
        p.staged[li] = (b, idx if staged is None else torch.cat([staged, idx]))   # restored with the staged experts
        self.cold[li, idx] = False

    def summary(self) -> dict:
        picks, dropped, weight, dropped_w = self.acc.tolist()
        return dict(self.st, fetch_per_layer=FETCH_PER_LAYER, fetched_gb=round(self.st["fetched_bytes"] / 1e9, 2),
                    read_ms=round(self.st["read_ms"]), dropped_pick_frac=round(dropped / picks, 4) if picks else None,
                    dropped_weight_frac=round(dropped_w / weight, 4) if weight else None)
