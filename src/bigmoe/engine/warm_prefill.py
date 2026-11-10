"""Warm prefill (DeepSeek-V4.1, mode nvme): a staged forward reads from NVMe only the experts its routing uses.

A staged forward starts from the warm set: the experts in VRAM or the RAM tier when it begins. Once a layer's routing is
known, its cold (NVMe-only) experts are ranked by the routing weight the forward gives them and read from the store in
parallel into free slots of the layer's staging buffer:

  forwards of at most EXACT_ROWS rows read every cold expert their routing touches and drop nothing, so the MoE runs
      the same experts on the same bytes as staging all of them: the logits are identical.
  longer forwards read the FETCH_PER_LAYER heaviest; the picks of the other cold experts are dropped (sentinel id,
      weight 0) and each token's weights are rescaled to their original sum.

Read-ahead (longer forwards): while layer i computes, layer i+1's cold experts are guessed with its own router on layer
i's router input and read on a background thread, most likely first. Once layer i+1's routing is known the job is
retargeted: guesses not yet read are dropped and the true experts not yet read are queued. The experts used are the
same as without read-ahead, so the result is unchanged.

Decode is unchanged. Mode nvme-exact and GLM-5.3-Flash stage every expert.
"""
import concurrent.futures as cf
import threading
import time

import numpy as np
import torch

FETCH_PER_LAYER = 64      # cold experts read per layer of a longer staged forward
EXACT_ROWS = 1024         # staged forwards of at most this many rows read every routed cold expert (exact)
READERS = 16              # parallel store reads (the NVMe needs a deep queue for its full bandwidth)
BATCH = 16                # experts per read batch; two batches of pinned slots alternate (read one, copy the other)
RAM_RESIDENT = 2          # nv2 host state of an expert in the RAM tier (KS_RES)


def fetch_limit(rows: int, room: int) -> int:
    """Cold experts a staged layer call of `rows` rows reads at most, with `room` free staging slots."""
    if rows <= EXACT_ROWS:
        return room                         # every routed cold expert (they always fit: staged + cold <= stage_n)
    return min(FETCH_PER_LAYER, room)


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


class Ahead:
    """A read-ahead of one layer: its queue of (expert, position) still to read (retargeted by the consumer), what is
    being read and what has landed, under one lock; the event the copies of the job end with."""

    def __init__(self, layer, b, used, queue):
        self.layer, self.b, self.used, self.queue = layer, b, used, queue
        self.inflight, self.done, self.closed = {}, {}, False
        self.lock = threading.Lock()
        self.event = torch.cuda.Event()
        self.fut = None


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
        self.st = {"forwards": 0, "exact_calls": 0, "fetched": 0, "fetched_bytes": 0, "read_ms": 0.0, "ahead_guessed": 0,
                   "ahead_read": 0, "ahead_hit": 0, "ahead_wasted": 0, "ahead_missed": 0, "ahead_wait_ms": 0.0,
                   "ahead_ms": 0.0}
        self.ahead = None            # Ahead of the next layer
        self.flip = 0                # alternating half of the pinned arena
        self.ahead_pool = cf.ThreadPoolExecutor(1, thread_name_prefix="warm-ahead")

    def begin(self, ext, slot_np: np.ndarray) -> np.ndarray:
        """Start of a staged forward: the cold set of every layer, and the residency snapshot for stage_begin /
        stage_submit with the cold experts marked resident, so they are neither read nor staged. Their pointer-table
        rows are never used: route() fetches or drops every pick of theirs."""
        try:
            self.drop_ahead()
        finally:
            # Also cover a worker that failed after _take_ahead had dropped the only Ahead reference: no new staging
            # may overlap any H2D ever queued on the warm stream.
            torch.cuda.current_stream(self.pool.device).wait_stream(self.stream)
        cold = cold_set(slot_np, ext.state()[1].numpy())
        self.cold = torch.from_numpy(cold).to(self.pool.device)
        self.cold_np = cold
        self.st["forwards"] += 1
        return np.where(cold, 0, slot_np).astype(slot_np.dtype)

    def route(self, li: int, sel: torch.Tensor, w: torch.Tensor, z=None, params=None, bsz=None):
        """One layer call of the staged forward: fetch its cold experts (every routed one in a short forward, the
        heaviest in a longer one; those read ahead have landed already), drop any other cold picks, and start reading
        ahead for the next layer."""
        if self.cold is None:
            return sel, w
        p = self.pool
        cur = torch.cuda.current_stream(p.device)
        free = torch.cuda.Event()
        free.record(cur)                       # buffer li % 2 was last used by layer li - 2, buffer (li+1) % 2 by li - 1
        b, staged = p.staged.get(li, (li % 2, None))
        used = 0 if staged is None else int(staged.numel())
        rows = int(sel.shape[0])
        limit = fetch_limit(rows, p.stage_n - used)
        if rows <= EXACT_ROWS:
            self.st["exact_calls"] += 1
        fetch = rank_cold(sel, w, self.cold[li], p.first, limit).cpu().tolist()
        pairs = self._take_ahead(li, b, used, limit, fetch, cur) if self.ahead is not None else None
        if pairs is None and fetch:           # no read-ahead for this layer: read them now
            t0 = time.perf_counter()
            pairs = [(e, used + i) for i, e in enumerate(fetch)]
            self.stream.wait_event(free)
            self._read_into(li, b, pairs)
            cur.wait_stream(self.stream)
            self.st["read_ms"] += (time.perf_counter() - t0) * 1e3
        if pairs:
            self._point(li, b, staged, pairs)
        sel2, w2, m = drop_cold(sel, w, self.cold[li], p.first)
        wf = w.float()
        self.acc[0] += m.numel()
        self.acc[1] += m.sum()
        self.acc[2] += wf.sum()
        self.acc[3] += (wf * m).sum()
        # after this layer's routing is consumed: small row counts share exllamav3's routing buffers across layers
        if z is not None and bsz is not None and bsz > EXACT_ROWS and li + 1 < len(p.mods):
            self._start_ahead(li + 1, z, params, bsz, free)
        return sel2, w2

    # ---- read-ahead ---------------------------------------------------------------------------------------------
    def _start_ahead(self, nl, z, params, bsz, free):
        """Guess layer nl's heaviest cold experts with its own router on this layer's input and start reading them, most
        likely first, into its staging buffer after the RAM experts the pool stages there (positions from `used`)."""
        p = self.pool
        h = p.pending
        if h is None or h[0] != nl or not self.cold_np[nl].any():
            return
        b, used = h[1], len(h[2])
        limit = fetch_limit(int(bsz), p.stage_n - used)
        route = getattr(p.mods[nl], "_bigmoe_route", None)
        if limit <= 0 or route is None:
            return
        if p.stage is None:
            return
        psel, pw = route(bsz, p.mods[nl].routing_cfg, z, params)
        # only experts selected by the real routing are pointed at the staging buffer and run
        guess = rank_cold(psel, pw, self.cold[nl], p.first, limit).cpu().tolist()
        if not guess:
            return
        a = Ahead(nl, b, used, [(e, used + i) for i, e in enumerate(guess)])
        self.st["ahead_guessed"] += len(guess)

        def job():
            t0 = time.perf_counter()
            with torch.inference_mode():       # thread-local: the staging buffers are inference tensors
                self.stream.wait_event(free)
                while True:
                    with a.lock:
                        batch, a.queue = a.queue[:BATCH], a.queue[BATCH:]
                        a.inflight = dict(batch)
                        if not batch:
                            a.closed = True
                            break
                    self._read_into(nl, b, batch)
                    with a.lock:
                        a.done.update(a.inflight)
                        a.inflight = {}
                a.event.record(self.stream)
            self.st["ahead_ms"] += (time.perf_counter() - t0) * 1e3
        a.fut = self.ahead_pool.submit(job)
        self.ahead = a

    def _take_ahead(self, li, b, used, limit, fetch, cur):
        """Retarget the read-ahead of layer li to its true experts `fetch` and wait for them: (expert, position) pairs,
        or None if there is none for this layer."""
        a, self.ahead = self.ahead, None
        if (a.layer, a.b, a.used) != (li, b, used):   # the forward went another way (not in a staged forward)
            with a.lock:
                a.queue = []
            a.fut.result()
            cur.wait_event(a.event)           # even with no true experts to fetch, old copies must finish first
            return None
        t0 = time.perf_counter()
        with a.lock:
            have = {**a.done, **a.inflight}
            a.queue = []                                   # guesses not read yet: dropped
            hits = {e: have[e] for e in fetch if e in have}
            taken = set(hits.values())
            spare = (q for q in range(used, used + limit) if q not in taken)
            misses = [(e, next(spare)) for e in fetch if e not in have]
            if not a.closed:
                a.queue = list(misses)                     # the job reads them next, in the same pipeline
                misses_left = []
            else:
                misses_left = misses
        a.fut.result()
        if misses_left:
            self._read_into(li, b, misses_left)
        cur.wait_event(a.event)
        cur.wait_stream(self.stream)
        self.st["ahead_wait_ms"] += (time.perf_counter() - t0) * 1e3
        self.st["ahead_read"] += len(have)
        self.st["ahead_hit"] += len(hits)
        self.st["ahead_wasted"] += len(have) - len(hits)
        self.st["ahead_missed"] += len(misses)
        return list(hits.items()) + misses

    def before_stage(self, b: int):
        """Ordinary cstream staging is about to reuse buffer b. Sequential forwards alternate buffers, so this is free;
        an out-of-order recovery must drain a conflicting read before stage_submit records cstream's dependency."""
        if self.ahead is not None and self.ahead.b == b:
            self.drop_ahead()

    def drop_ahead(self):
        a, self.ahead = self.ahead, None
        if a is not None:
            with a.lock:
                a.queue = []
            a.fut.result()                    # the copies are queued by now, not necessarily complete
            # An abandoned forward can be followed by ordinary cstream staging in the same buffer. Order the compute
            # stream after the abandoned H2Ds; its next stage_submit event then orders cstream after them too.
            torch.cuda.current_stream(self.pool.device).wait_event(a.event)

    # ---- reads ----------------------------------------------------------------------------------------------------
    def _read_into(self, li, b, pairs):
        """Read the (expert, position) pairs of layer li from the store into staging buffer b on self.stream (after
        whatever it waits on), through the pinned arena in alternating batches."""
        from . import nv2 as N2
        p, ext, A = self.pool, N2.ext(), self.arena
        if p.stage is None:
            raise RuntimeError("warm prefill: staging buffers freed under a read")
        stage = p.stage[b]
        for c in range(0, len(pairs), BATCH):
            half = self.flip
            self.flip ^= 1
            self.copied[half].synchronize()    # these pinned slots' previous H2D is done
            batch = pairs[c:c + BATCH]
            slots = [half * BATCH + i for i in range(len(batch))]
            keys = [li * p.E + e for e, _ in batch]
            got = list(self.readers.map(lambda ks: ext.read_sync(ks[0], int(A.addr[ks[1]])), zip(keys, slots)))
            with torch.cuda.stream(self.stream):
                for i, (key, s, (_e, pos)) in enumerate(zip(keys, slots, batch)):
                    n = int(p.nv2.klen[key])
                    if got[i] != n:
                        raise RuntimeError(f"warm prefill: short read of expert {key} ({got[i]} of {n} bytes)")
                    stage[pos, :n].copy_(A.view(s, n), non_blocking=True)
                    self.st["fetched_bytes"] += n
            self.copied[half].record(self.stream)
        self.st["fetched"] += len(pairs)

    def _point(self, li, b, staged, pairs):
        """Point layer li's tables at the staged positions of `pairs` and mark those experts warm for drop_cold."""
        p = self.pool
        idx = torch.tensor([e for e, _ in pairs], dtype=torch.long, device=p.device)
        pos = torch.tensor([q for _, q in pairs], dtype=torch.int64, device=p.device)
        a = p.stage[b].data_ptr() + pos * p.rec
        _, _, _, ou, od = p.g[li]
        pg, pu, pd, gu = p.keep[li]
        pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + ou); pd.index_copy_(0, idx, a + od)
        if gu is not None:
            gu.index_copy_(0, idx, torch.stack([a, a + ou], 1))
        p.staged[li] = (b, idx if staged is None else torch.cat([staged, idx]))   # restored with the staged experts
        self.cold[li, idx] = False

    def summary(self) -> dict:
        picks, dropped, weight, dropped_w = self.acc.tolist()
        st = self.st
        covered = st["ahead_hit"] + st["ahead_missed"]
        return dict(st, fetched_gb=round(st["fetched_bytes"] / 1e9, 2), read_ms=round(st["read_ms"]),
                    ahead_ms=round(st["ahead_ms"]), ahead_wait_ms=round(st["ahead_wait_ms"]),
                    ahead_hit_frac=round(st["ahead_hit"] / st["ahead_read"], 3) if st["ahead_read"] else None,
                    ahead_cover_frac=round(st["ahead_hit"] / covered, 3) if covered else None,
                    dropped_pick_frac=round(dropped / picks, 4) if picks else None,
                    dropped_weight_frac=round(dropped_w / weight, 4) if weight else None)
