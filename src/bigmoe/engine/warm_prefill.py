"""Warm prefill: staged forwards run on the experts that are already in VRAM or RAM, plus the few NVMe-only experts that
matter most to each layer of the chunk.

A staged forward (a prefill chunk, >= EC_STAGE_MIN picks per layer) used to stage every expert of every layer, which
reads every NVMe-only expert once: ~15 s per forward on a 3090 box whatever the chunk's length, the floor of every
agent step that adds a tool result. Here the forward starts from the warm set (VRAM-resident and RAM-tier experts at
its start). Once a layer's routing is known, its cold (NVMe-only) experts are ranked by the routing weight the chunk
gives them; the top FETCH_PER_LAYER are read from the store in parallel into free slots of the layer's staging buffer
and run exactly. The picks of the remaining cold experts are dropped (sentinel id, weight 0) and each token's weights
are rescaled to their original sum. Decode is unchanged.

Read-ahead (EC_WARM_AHEAD, on by default for chunks of at least EC_WARM_AHEAD_MIN=1024 rows): a layer's reads would
otherwise stall the GPU before its MoE. While layer i computes, layer i+1's cold experts are guessed with its own router
on layer i's router input (exact for DeepSeek's hash-routed layers, which route by token id) and read on a background
thread into that layer's staging buffer, most likely first. Once layer i+1's true routing is known the job is retargeted:
guesses it has not read yet are dropped and the true experts it has not read are queued, so wasted reads are only those
done before the routing was known. The experts used are the same top FETCH_PER_LAYER as without read-ahead, so the
forward's result is unchanged.

Against exact prefill: DeepSeek-V4.1-Flash on five steps of a DeepSeek Harness agent turn (14k to 36k tokens, decode
exact) prefills in 77.5 s instead of 116.9 s with top-1 agreement 99.7 % and mean KL 0.0021 on 330 forced answer
positions; teacher-forced (prompt and answer through warm prefill) DeepSeek keeps 100 % top-1 (mean KL 0.023) and
GLM-5.3-Flash 94.5 % (0.042). Mode nvme only, as the backends' warm_prefill option (DeepSeek: on by default,
GLM: off); mode nvme-exact prefills exactly.
"""
import concurrent.futures as cf
import threading
import time

import numpy as np
import torch

from .env import env

# cold experts read per layer of a staged forward (64 of 384: ~7 % of the routing weight left); EC_WARM_FETCH overrides
FETCH_PER_LAYER = int(env("EC_WARM_FETCH", "64"))
READERS = 16              # parallel store reads (the NVMe needs a deep queue for its full bandwidth)
BATCH = 16                # experts per read batch; two batches of pinned slots alternate (read one, copy the other)
RAM_RESIDENT = 2          # nv2 host state of an expert in the RAM tier (KS_RES)
AHEAD = env("EC_WARM_AHEAD", "1") == "1"
AHEAD_N = int(env("EC_WARM_AHEAD_N", "0")) or None   # guesses read ahead per layer (default FETCH_PER_LAYER)
# Extra routing / ranking does not pay on short chunks (matched A/B, 2026-11-09); 0 enables it everywhere.
AHEAD_MIN = int(env("EC_WARM_AHEAD_MIN", "1024"))


def set_ahead(on: bool, n: int | None = None) -> dict:
    """Runtime switch (A/B on one server, /bigmoe/rpc warm_ahead): read-ahead on/off and its guesses per layer."""
    global AHEAD, AHEAD_N
    AHEAD, AHEAD_N = bool(on), (int(n) or None) if n is not None else AHEAD_N
    return {"ahead": AHEAD, "ahead_n": AHEAD_N}


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
        self.st = {"forwards": 0, "fetched": 0, "fetched_bytes": 0, "read_ms": 0.0, "ahead_read": 0, "ahead_hit": 0,
                   "ahead_wasted": 0, "ahead_wait_ms": 0.0, "ahead_ms": 0.0}
        self.enabled = True          # runtime switch (set_enabled): off = exact staged prefill on this server
        self.ahead = None            # Ahead of the next layer
        self.flip = 0                # alternating half of the pinned arena
        self.ahead_pool = cf.ThreadPoolExecutor(1, thread_name_prefix="warm-ahead")

    def set_enabled(self, on: bool) -> bool:
        """Runtime switch between forwards (quality A/B on one server): off stages every expert, exactly as a server
        started with warm_prefill=False. The prefill ring borrows its RAM slots again while off (warm prefill keeps it
        from evicting RAM experts that a forward counts as warm)."""
        from . import nv2 as N2
        self.drop_ahead()
        self.enabled = bool(on)
        nv = self.pool.nv2
        if getattr(nv, "ring_borrow", False):
            N2.ext().set_ring_borrow(0 if self.enabled else int(nv.ring_pf))
        return self.enabled

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
        """One layer call of the staged forward: fetch the heaviest cold experts (the ones read ahead already landed),
        drop the other cold picks, and start reading ahead for the next layer."""
        if self.cold is None:
            return sel, w
        p = self.pool
        cur = torch.cuda.current_stream(p.device)
        free = torch.cuda.Event()
        free.record(cur)                       # buffer li % 2 was last used by layer li - 2, buffer (li+1) % 2 by li - 1
        b, staged = p.staged.get(li, (li % 2, None))
        used = 0 if staged is None else int(staged.numel())
        limit = min(FETCH_PER_LAYER, p.stage_n - used)
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
        if AHEAD and z is not None and bsz is not None and bsz >= AHEAD_MIN and li + 1 < len(p.mods):
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
        limit = min(FETCH_PER_LAYER, p.stage_n - used)
        route = getattr(p.mods[nl], "_bigmoe_route", None)
        if limit <= 0 or route is None:
            return
        if p.stage is None:
            return
        psel, pw = route(bsz, p.mods[nl].routing_cfg, z, params)
        # A wider guess window can contain more of the true top `limit`; positions beyond that window are harmless:
        # only experts selected by the real routing below are pointed at the staging buffer and run.
        guess_limit = min(AHEAD_N or limit, p.stage_n - used)
        guess = rank_cold(psel, pw, self.cold[nl], p.first, guess_limit).cpu().tolist()
        if not guess:
            return
        a = Ahead(nl, b, used, [(e, used + i) for i, e in enumerate(guess)])
        self.st["ahead_guessed"] = self.st.get("ahead_guessed", 0) + len(guess)

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
        self.st["ahead_missed"] = self.st.get("ahead_missed", 0) + len(misses)
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
        return dict(self.st, fetch_per_layer=FETCH_PER_LAYER, fetched_gb=round(self.st["fetched_bytes"] / 1e9, 2),
                    read_ms=round(self.st["read_ms"]), ahead=AHEAD, ahead_min=AHEAD_MIN,
                    ahead_ms=round(self.st["ahead_ms"]),
                    ahead_wait_ms=round(self.st["ahead_wait_ms"]),
                    ahead_hit_frac=round(self.st["ahead_hit"] / self.st["ahead_read"], 3) if self.st["ahead_read"]
                    else None,
                    ahead_cover_frac=round(self.st["ahead_hit"] / (self.st["ahead_hit"] + self.st.get("ahead_missed", 0)), 3)
                    if self.st["ahead_hit"] + self.st.get("ahead_missed", 0) else None,
                    dropped_pick_frac=round(dropped / picks, 4) if picks else None,
                    dropped_weight_frac=round(dropped_w / weight, 4) if weight else None)
