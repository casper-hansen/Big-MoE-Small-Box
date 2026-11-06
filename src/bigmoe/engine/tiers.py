"""Expert tiers: store manifest, loader patch, VRAM pool, attach (with nv2.py: RAM tier, NVMe, planner, CPU lane).

  loader   install() before model load: the routed experts' trellis tensors become views of a VRAM bounce slot (the
           checkpoint bytes are never read; the store has them) and the fused path's gu pointer table is captured
  Pool     ONE CLOCK cache over every routed layer in the free VRAM. Slots are uniform (the largest layer's expert); an
           expert of layer li fills its slot in li's compact layout (gate | up @ off_u | down @ off_d = the store
           record image), so per-layer kernel calls pass geom(li). Prefill chunks stage every non-VRAM expert of the
           next layer into two VRAM buffers on a copy stream; in elastic mode those buffers (and the activations) come
           from the last chunks of the cache, which are handed back to torch for the prefill and reclaimed for decode.
           In mode nvme with EC_WARM_PREFILL=1 a staged forward stages the warm experts plus each layer's heaviest
           cold ones (warm_prefill.py; the backends' warm_prefill option).
  attach   pool + nv2 + routing / combine hooks + bounce proxies for exllamav3's per-expert paths

Knobs (<prefix>NAME, see env.py): NV_STORE (store .bin), EC_RESERVE_GB (VRAM left to torch, 2.5), EC_MAX_SLOTS,
EC_ADMIT_MAX (64), EC_CHUNK_SLOTS (32), EC_STAGE_SLOTS / EC_STAGE_GB (staging buffer size; 0 = no staged prefill),
EC_STAGE_MIN (picks per layer call that stage, 8192), EC_ELASTIC_GB (0 = off), EC_ELASTIC_DYN (1),
EC_ACT_MB_PER_TOK (0.6), EC_COPY_GRID, EC_WARM (JSON {layer key: [score x E]}), EC_WARM_PREFILL (0; mode nvme only).
"""
import ctypes
import json
import mmap
import os
import sys
import time

import numpy as np
import torch

from .ec import ext as _ext
from .env import env

GiB = 1024 ** 3
_BC_ARGS = {}             # id(multi_gate.ptrs_trellis) -> (gate ptrs, up ptrs, gu_trellis_ptr) from BC_BlockSparseMLP(...)
_BOUNCE = {}              # device index -> (tensor, base address)
_LOAD = {"active": None, "marked_linears": 0, "skipped_bytes": 0, "files": set()}
_T0 = time.time()
POOLS = []
MAN = None


def al(x, a=256):
    return (x + a - 1) // a * a


# --------------------------------------------------------------------------------------------------------------------
# store manifest (v1: one record size; v2: per-layer record sizes) -> per-layer layout
# --------------------------------------------------------------------------------------------------------------------
def manifest():
    """{"_store", "mats": [gate, up, down], "layers": [{li, key, base, rec_bytes, trellis_bytes, off_u, off_d, fields,
    routed}], "experts_per_layer"}; v1 stores (GLM-5.3) are normalised to the v2 per-layer form."""
    global MAN
    if MAN is not None:
        return MAN
    store = env("NV_STORE")
    assert store, f"{env.__module__}: <prefix>NV_STORE (expert store .bin) is required"
    m = json.load(open(store[:-4] + ".json"))
    E = m["experts_per_layer"]
    if m["format"] == "exl3-expert-store-v2":
        layers = [dict(L, routed=True) for L in m["layers"]]
        mats = list(m["mats"])
    else:
        assert m["format"] == "glm53-expert-store-v1", m.get("format")
        vs, R = m["vram_slot"], m["record_bytes"]
        mats = [f["name"].split(".")[0] for f in m["fields"] if f["name"].endswith(".trellis")]
        layers = [{"li": L["li"], "key": L["key"], "model_layer": L.get("model_layer"), "base": L["li"] * E * R,
                   "rec_bytes": R, "trellis_bytes": vs["slot_bytes"], "off_u": vs["off_u"], "off_d": vs["off_d"],
                   "fields": m["fields"], "routed": L["li"] != m.get("mtp_layer_li")} for L in m["layers"]]
    MAN = {"_store": store, "format": m["format"], "mats": mats, "experts_per_layer": E, "layers": layers,
           "total_bytes": m.get("total_bytes")}
    return MAN


def routed_layers():
    return [L for L in manifest()["layers"] if L["routed"]]


# --------------------------------------------------------------------------------------------------------------------
# memory helpers
# --------------------------------------------------------------------------------------------------------------------
def cgroup_mem():
    """memcg numbers (GiB) of this process's cgroup v2 + host MemAvailable (GiB)."""
    out = {}
    try:
        cg = "/sys/fs/cgroup"
        if not os.path.exists(f"{cg}/memory.max"):
            try:
                rel = [l.split(":", 2)[2].strip() for l in open("/proc/self/cgroup") if l.startswith("0::")][0]
                cg = "/sys/fs/cgroup" + rel.rstrip("/")
            except (OSError, IndexError):
                pass
        out["current"] = int(open(f"{cg}/memory.current").read()) / GiB
        if os.path.exists(f"{cg}/memory.peak"):
            out["peak"] = int(open(f"{cg}/memory.peak").read()) / GiB
        mx = open(f"{cg}/memory.max").read().strip()
        out["max"] = int(mx) / GiB if mx != "max" else None
        for line in open(f"{cg}/memory.stat"):
            k, v = line.split()
            if k in ("anon", "file"):
                out[k] = int(v) / GiB
    except OSError:
        pass
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            out["host_avail"] = int(line.split()[1]) / 1048576
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def log(msg):
    m = cgroup_mem()
    print(f" -- tiers [{time.time() - _T0:7.1f}s] {msg} | memcg cur {m.get('current', -1):.2f} peak {m.get('peak', -1):.2f}"
          f" GiB, host avail {m.get('host_avail', -1):.1f}", flush=True)


_log = log


def _rcs_bytes():
    """exllamav3's host-RAM caches, left out of the RAM tier: recurrent / SWA state checkpoints (-rcs GiB, default
    0.5) and the second-tier K/V page cache in pinned memory (-ccs GiB, default 0)."""
    a, rcs, ccs = sys.argv, 0.5, 0.0
    for i, x in enumerate(a):
        if x in ("-rcs", "--recurrent_cache_size") and i + 1 < len(a):
            rcs = float(a[i + 1])
        if x in ("-ccs", "--cpu_cache_size") and i + 1 < len(a):
            ccs = float(a[i + 1])
    return int((rcs + ccs) * GiB)


class HostArena:
    """Page-aligned anon memory, cudaHostRegister(Mapped) per chunk (charged to the memcg when registered)."""

    def __init__(self, nslots, slot_bytes, chunk_slots, dev_idx, tag):
        self.slot, self.n, self.cs = slot_bytes, nslots, chunk_slots
        self.maps, self.bases = [], []
        from exllamav3.ext import exllamav3_ext as ext
        left = nslots
        while left > 0:
            k = min(left, chunk_slots)
            mm = mmap.mmap(-1, k * slot_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            assert a % 4096 == 0
            with torch.cuda.device(dev_idx):
                r = torch.cuda.cudart().cudaHostRegister(a, k * slot_bytes, 2)
            assert int(r) == 0, f"cudaHostRegister failed ({tag}): {r}"
            t = torch.frombuffer(mm, dtype=torch.uint8, count=4096)
            assert ext.pinned_cuda_view(t, dev_idx).data_ptr() == a, "UVA alias != host address"
            self.maps.append(mm); self.bases.append(a)
            left -= k
        self.addr = np.array([self.bases[s // chunk_slots] + (s % chunk_slots) * slot_bytes for s in range(nslots)],
                             dtype=np.int64)

    def view(self, s, n=None):
        mm = self.maps[s // self.cs]
        return torch.frombuffer(mm, dtype=torch.uint8, count=n or self.slot, offset=(s % self.cs) * self.slot)


# --------------------------------------------------------------------------------------------------------------------
# loader: routed-expert trellis -> bounce views, nothing read
# --------------------------------------------------------------------------------------------------------------------
def install():
    """Before model load: (1) capture gu_trellis_ptr from the BC constructor, (2) make the loader hand every routed
    expert's trellis a view of the bounce slot instead of reading it."""
    from exllamav3.ext import exllamav3_ext as ext
    import exllamav3.modules.block_sparse_mlp as bsm
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.linear import Linear
    from exllamav3.loader.safetensors import SafetensorsCollection

    real = ext.BC_BlockSparseMLP

    def wrapped(*a, **k):
        obj = real(*a, **k)
        # gate ptrs (identity key), up ptrs, gu_trellis_ptr, the shared expert's output rows (X16 lane reduction)
        _BC_ARGS[id(a[16])] = (a[16], a[22], a[43], a[9])
        return obj
    ext.BC_BlockSparseMLP = wrapped
    bsm.ext.BC_BlockSparseMLP = wrapped

    man = manifest()
    gate, up, down = man["mats"]
    by_key = {L["key"]: L for L in routed_layers()}
    slot_max = max(L["trellis_bytes"] for L in by_key.values())
    dtmap = {"torch.int16": torch.int16, "I16": torch.int16}

    def bounce(dev_idx):
        if dev_idx not in _BOUNCE:
            t = torch.zeros(slot_max, dtype=torch.uint8, device=f"cuda:{dev_idx}")
            _BOUNCE[dev_idx] = (t, t.data_ptr())
        return _BOUNCE[dev_idx]

    orig_split = BlockSparseMLP.cpu_maybe_split_load

    def split_then_mark(self, device, **kwargs):
        orig_split(self, device, **kwargs)
        if device is None or torch.device(device).type != "cuda" or self.key not in by_key:
            return
        assert self.cpu_split_first is None and not self.cpu_offload, "tiers: no stock CPU expert split / offload"
        L = by_key[self.key]
        f = {x["name"]: x for x in L["fields"]}
        dev_idx = torch.device(device).index or 0
        bounce(dev_idx)
        for mat, off, group in ((gate, 0, self.gates), (up, L["off_u"], self.ups), (down, L["off_d"], self.downs)):
            for l in group:
                assert l.key.endswith(f".{mat}"), (l.key, mat)
                l._nv_slot = (dev_idx, off, f[mat + ".trellis"]["bytes"])
                _LOAD["marked_linears"] += 1

    BlockSparseMLP.cpu_maybe_split_load = split_then_mark
    orig_lin_load = Linear.load

    def lin_load(self, device, **kwargs):
        if getattr(self, "_nv_slot", None) is None:
            return orig_lin_load(self, device, **kwargs)
        _LOAD["active"] = (self.key + ".trellis", self._nv_slot)
        try:
            return orig_lin_load(self, device, **kwargs)
        finally:
            _LOAD["active"] = None

    Linear.load = lin_load
    orig_get = SafetensorsCollection.get_tensor

    def get_tensor(self, key, device=None, *args, **kwargs):
        a = _LOAD["active"]
        if a is None or key != a[0]:
            fn = self.tensor_file_map.get(key) if hasattr(self, "tensor_file_map") else None
            if fn:
                _LOAD["files"].add(os.path.join(self.directory, fn) if not os.path.isabs(fn) else fn)
            return orig_get(self, key, device, *args, **kwargs)
        dev_idx, off, sz = a[1]
        meta = self.find_stc(key).get_tensor_meta(key)[key]
        assert meta["n_bytes"] == sz, (key, meta, sz)
        t, _ = _BOUNCE[dev_idx]
        _LOAD["skipped_bytes"] += sz
        return t[off: off + sz].view(dtmap[meta["dtype"]]).view(meta["shape"])

    SafetensorsCollection.get_tensor = get_tensor
    _ext()   # build now (fail fast, before the long load)
    log(f"installed: store {man['_store']} ({man['format']}), {len(by_key)} routed layers, loader skips expert trellis "
        f"bytes (bounce slot {slot_max} B)")


def _drop_checkpoint_cache():
    nd = 0
    for f in sorted(_LOAD["files"]):
        try:
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); nd += 1
        except OSError:
            pass
    return nd


# --------------------------------------------------------------------------------------------------------------------
# VRAM pool
# --------------------------------------------------------------------------------------------------------------------
class Pool:
    def __init__(self, device, mods, reserve, max_slots, admit_max, stage_n=0, stage_min=8192, elastic=0.0):
        self.device, self.mods, self.admit_max = device, mods, admit_max
        self.nv2 = None
        self.stage_min = stage_min
        m0 = mods[0]
        self.E = m0.num_local_experts or m0.num_experts
        self.first = int(m0.routing_first or 0)
        self.topk = int(m0.num_experts_per_tok)
        nb = lambda l: l.inner.trellis.numel() * l.inner.trellis.element_size()
        self.g = []
        for m in mods:
            assert (m.num_local_experts or m.num_experts) == self.E and int(m.routing_first or 0) == self.first
            sg, su, sd = nb(m.multi_gate.linears[0]), nb(m.multi_up.linears[0]), nb(m.multi_down.linears[0])
            for mm, s in ((m.multi_gate, sg), (m.multi_up, su), (m.multi_down, sd)):
                for l in mm.linears:
                    assert nb(l) == s and s % 16 == 0, (m.key, nb(l), s)
            self.g.append((sg, su, sd, al(sg), al(sg) + al(su)))
        self.rec = max(od + al(sd) for sg, su, sd, ou, od in self.g)      # uniform slot: the largest layer's expert
        self.uniform = len(set(self.g)) == 1
        self.sg, self.su, self.sd, self.off_u, self.off_d = self.g[0]
        torch.cuda.synchronize(device); torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        self.stage_n = int(stage_n)
        self.elastic = float(elastic)
        self.stage = None
        if self.stage_n and not self.elastic:
            self.stage = [torch.empty((self.stage_n, self.rec), dtype=torch.uint8, device=device) for _ in range(2)]
            free -= 2 * self.stage_n * self.rec
        S = max(0, int((free - reserve) // self.rec))
        if max_slots is not None:
            S = min(S, max_slots)
        # arena in chunks of spc slots: flex chunks (the last ones) are handed back to torch during prefill (elastic)
        self.spc = int(env("EC_CHUNK_SLOTS", "32"))
        self.chunks = []
        for _ in range((S + self.spc - 1) // self.spc):
            try:
                self.chunks.append(torch.empty((self.spc, self.rec), dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:
                break
        self.S = S = len(self.chunks) * self.spc
        self.cbase = torch.tensor([t.data_ptr() for t in self.chunks] or [0], dtype=torch.int64, device=device)
        nflex = min(len(self.chunks), int(-(-self.elastic // (self.spc * self.rec)))) if self.elastic and self.chunks else 0
        self.flex_all = list(range(len(self.chunks) - nflex, len(self.chunks)))
        self.flex = list(self.flex_all)
        self.in_prefill = False
        self.elastic_stats = {"enter": 0, "exit": 0, "realloc_fail": 0}
        L = len(mods)
        i32 = dict(dtype=torch.int32, device=device)
        self.slotof = torch.full((L, self.E), -1, **i32)
        self.owner = torch.full((max(S, 1),), -1, **i32)
        self.refb = torch.zeros((max(S, 1),), **i32)
        self.pin = torch.zeros((max(S, 1),), **i32)
        self.ctl = torch.zeros((4,), **i32)
        self.jobs = torch.zeros((2 * 1024,), **i32)
        self.stats = torch.zeros((8,), dtype=torch.int64, device=device)
        home, tabs, self.keep = [], [], []
        for m in mods:
            pg, pu, pd = m.multi_gate.ptrs_trellis, m.multi_up.ptrs_trellis, m.multi_down.ptrs_trellis
            cap = _BC_ARGS.get(id(pg))
            gu = None
            m._x16_sh_out = cap[3] if cap is not None else None
            if cap is not None:
                assert cap[0] is pg and cap[1] is pu
                gu = cap[2]
                assert gu.shape == (self.E, 2) and gu.dtype == torch.long and gu.device == pg.device
                assert torch.equal(gu[:, 0], pg) and torch.equal(gu[:, 1], pu), m.key
            else:
                print(f" !! tiers: no gu table for {m.key} (fused bsz-N path would read home)", flush=True)
            for t in (pg, pu, pd):
                assert t.dtype == torch.long and t.is_contiguous() and t.device == torch.device(device)
            home.append(torch.stack([pg, pu, pd], dim=1))
            tabs.append([pg.data_ptr(), pu.data_ptr(), pd.data_ptr(), gu.data_ptr() if gu is not None else 0])
            self.keep.append((pg, pu, pd, gu))
        self.home = torch.stack(home).contiguous()                    # [L, E, 3] current homes
        self.tabs = torch.tensor(tabs, dtype=torch.int64, device=device)
        self.grid = int(env("EC_COPY_GRID", "0")) or torch.cuda.get_device_properties(device).multi_processor_count * 8
        self.size_hist = {}
        if self.stage_n:
            self.cstream = torch.cuda.Stream(device)
            self.ev_ready = [torch.cuda.Event() for _ in range(2)]
            self.staged = {}          # li -> (buffer idx, expert idx tensor on device)
            self.last_li = None
            self.slot_np = None
            self.pending = None
            self._st_open = False
            self.stage_stats = {"forwards": 0, "layers": 0, "experts": 0}
        self.warm_prefill = None      # warm_prefill.WarmPrefill in mode nvme (set by attach)
        for li, m in enumerate(mods):
            m._ec = (self, li)

    def geom(self, li):
        """(sg, su, sd, off_u, off_d) of layer li's experts inside a slot."""
        return self.g[li]

    def step(self, li, sel, admit):
        sg, su, sd, ou, od = self.g[li]
        n = sel.numel()
        b = 1 << max(0, (n - 1).bit_length())
        self.size_hist[b] = self.size_hist.get(b, 0) + 1
        e = _ext()
        sel = sel.reshape(-1)
        if sel.dtype != torch.long or not sel.is_contiguous():
            sel = sel.long().contiguous()
        e.ec_step(sel, self.first, li, self.E, self.S, int(admit), self.slotof, self.home, self.tabs, self.owner, self.refb,
                  self.pin, self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, ou, od)
        if admit:
            e.ec_copy(self.ctl, self.jobs, li, self.E, self.home, self.cbase.data_ptr(), self.spc, self.rec, sg, su, sd,
                      ou, od, self.grid)

    def warm(self, scores):
        """Initial fill: top S/L RAM-resident experts per layer by score (admission path, tables stay consistent)."""
        k = self.S // max(1, len(self.mods))
        if k == 0 or not scores or self.nv2 is None:
            return
        from . import nv2 as N2
        ks = N2.ext().state()[1].numpy()
        with torch.cuda.device(self.device):
            for li, m in enumerate(self.mods):
                sc = scores.get(m.key)
                if sc is None:
                    continue
                loc = range(self.first, self.first + self.E)
                top = [x for x in sorted(loc, key=lambda x: -sc[x])[:k] if ks[li * self.E + x - self.first] == 2]
                for c in range(0, len(top), 256):
                    self.step(li, torch.tensor(top[c:c + 256], dtype=torch.long, device=self.device), True)
            torch.cuda.synchronize(self.device)
        self.stats.zero_()

    # ---- staged prefill ----------------------------------------------------------------------------------------------
    def _ec_hits(self, li, sel):
        """Staged forwards: CLOCK hit marking only (no admission)."""
        sg, su, sd, ou, od = self.g[li]
        _ext().ec_step(sel, self.first, li, self.E, self.S, 0, self.slotof, self.home, self.tabs, self.owner, self.refb,
                       self.pin, self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, ou, od)

    def _restore_nv2(self, li):
        from . import nv2 as N2
        b, idx = self.staged.pop(li)
        N2.ext().nv_restore(idx, li, self.E, self.nv2.a["home"], self.tabs)

    def _stage_hook_nv2(self, li, n_picks):
        """Per forward the host engine pins the layer's RAM-resident experts and reads the NVMe-only ones through a FIFO
        ring several layers ahead; per layer the copy stream H2Ds them into staging buffer li % 2 (GIL-free job on a
        background thread, one layer ahead of the compute stream)."""
        from . import nv2 as N2
        e = N2.ext()
        prefill = n_picks >= self.stage_min
        for l in [k for k in self.staged if k != li]:
            self._restore_nv2(l)                      # stream order: after that layer's MoE
        if not prefill:
            if self.last_li is not None or self._st_open:
                if self.pending is not None:
                    self.nv2.stage_wait(self.pending)
                    self.pending = None
                e.stage_end()
                self._st_open = False
            self.last_li = None
            if self.in_prefill:
                self._elastic_exit()
            return
        cur = torch.cuda.current_stream(self.device)
        if self.last_li is None or li <= self.last_li:  # new forward pass
            if self.pending is not None:
                self.nv2.stage_wait(self.pending)
                self.pending = None
            if self.elastic and not self.in_prefill:
                self._elastic_enter(n_picks // self.topk)
            self.slot_np = self.slotof.cpu().numpy()    # stable during prefill (no admission)
            if self.warm_prefill is not None:
                self.slot_np = self.warm_prefill.begin(e, self.slot_np)
            e.stage_begin(torch.from_numpy(np.ascontiguousarray(self.slot_np)), li)
            self._st_open = True
            self.stage_stats["forwards"] += 1
            self.pending = None
        self.last_li = li
        if self.pending is None or self.pending[0] != li:
            if self.pending is not None:
                self.nv2.stage_wait(self.pending)
            self.pending = self.nv2.stage_submit(li, li % 2)
        h = self.pending
        self.nv2.stage_wait(h)
        _, b, take, _f, _ev = h
        cur.wait_event(self.ev_ready[b])
        if len(take):
            idx = torch.from_numpy(take.astype(np.int64)).to(self.device)
            sg, su, sd, ou, od = self.g[li]
            a = self.stage[b].data_ptr() + torch.arange(len(take), dtype=torch.int64, device=self.device) * self.rec
            pg, pu, pd, gu = self.keep[li]
            pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + ou); pd.index_copy_(0, idx, a + od)
            if gu is not None:
                gu.index_copy_(0, idx, torch.stack([a, a + ou], 1))
            self.staged[li] = (b, idx)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += len(take)
        self.pending = self.nv2.stage_submit(li + 1, (li + 1) % 2)   # next layer, other buffer, overlaps this layer

    def _elastic_enter(self, ntok=None):
        """Prefill: evict + disable the flex chunks' slots, hand their memory back to torch (activations of big
        chunks), allocate the staging buffers from it. Short forwards take only what they need (EC_ELASTIC_DYN)."""
        e = _ext()
        self.flex = list(self.flex_all)
        if ntok is not None and self.nv2 is not None and env("EC_ELASTIC_DYN", "1") == "1":
            need = 2 * self.stage_n * self.rec + ntok * float(env("EC_ACT_MB_PER_TOK", "0.6")) * 1e6 + 0.3e9
            nf = min(len(self.flex_all), int(-(-need // (self.spc * self.rec))))
            self.flex = self.flex_all[len(self.flex_all) - nf:]
        if self.nv2 is not None:
            self.elastic_stats["wb"] = self.elastic_stats.get("wb", 0) + self.nv2.elastic_writeback(self.flex)
        for c in self.flex:
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 1, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        torch.cuda.synchronize(self.device)
        for c in self.flex:
            self.chunks[c] = None
        self.stage = [torch.empty((self.stage_n, self.rec), dtype=torch.uint8, device=self.device) for _ in range(2)]
        self.in_prefill = True
        self.elastic_stats["enter"] += 1

    def _elastic_exit(self):
        """Back to decode: free staging, reallocate the flex chunks, re-enable their slots (empty)."""
        torch.cuda.synchronize(self.device)
        self.stage = None
        torch.cuda.empty_cache()
        e = _ext()
        for c in self.flex:
            try:
                t = torch.empty((self.spc, self.rec), dtype=torch.uint8, device=self.device)
            except torch.OutOfMemoryError:
                self.elastic_stats["realloc_fail"] += 1
                continue
            self.chunks[c] = t
            self.cbase[c] = t.data_ptr()
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 0, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        self.in_prefill = False
        self.elastic_stats["exit"] += 1

    def summary(self):
        s = self.stats.tolist()
        return {"device": str(self.device), "layers": len(self.mods), "slots": self.S, "slot_bytes": self.rec,
                "arena_gb": round(self.S * self.rec / 1e9, 2), "uniform_geometry": self.uniform,
                "elastic": dict(self.elastic_stats, flex_chunks=len(self.flex), in_prefill=self.in_prefill)
                if self.elastic else None,
                "hits": s[0], "misses": s[1], "admitted": s[2], "no_victim": s[3], "evictions": s[4],
                "hit_rate": round(s[0] / max(1, s[0] + s[1]), 4),
                "stage": dict(self.stage_stats, per_buffer=self.stage_n) if self.stage_n else None,
                "warm_prefill": self.warm_prefill.summary() if self.warm_prefill is not None else None,
                "picks_hist": dict(sorted(self.size_hist.items()))}


class BCProxy:
    """Wraps BC_BlockSparseMLP: the per-expert paths (DQ reconstruct, single-expert graph) read the per-expert Linear
    trellis, which is the bounce slot. Copy the expert's live bytes there first (same stream)."""

    def __init__(self, bc, li, pool):
        object.__setattr__(self, "_bc", bc)
        object.__setattr__(self, "_li", li)
        object.__setattr__(self, "_pool", pool)

    def _fill(self, e):
        p = self._pool
        _ext().nv_bounce(p.tabs, self._li, int(e), _BOUNCE[p.device.index or 0][1], *p.geom(self._li))

    def run_single_expert_dq(self, y, e, *a):
        self._pool.nv2.st["dq_bounce_calls"] += 1
        self._fill(e)
        return self._bc.run_single_expert_dq(y, e, *a)

    def run_single_expert(self, y, e, *a):
        self._pool.nv2.st["graph_bounce_calls"] += 1
        self._fill(e)
        return self._bc.run_single_expert(y, e, *a)

    def __getattr__(self, k):
        return getattr(self._bc, k)


# --------------------------------------------------------------------------------------------------------------------
# attach
# --------------------------------------------------------------------------------------------------------------------
def find_moe(modules):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    seen = []

    def walk(mm):
        if isinstance(mm, BlockSparseMLP):
            seen.append(mm)
        for c in getattr(mm, "modules", []) or []:
            walk(c)
    for top in modules:
        walk(top)
    return [m for m in seen if getattr(m, "multi_up", None) is not None and m.device is not None
            and (m.num_local_experts is None or m.num_local_experts > 0)]


def _hook_prefill(cls):
    """Elastic mode: hand the flex chunks back BEFORE the first layer of a big prefill chunk runs (the layers ahead of
    the first MoE layer already need the activation memory)."""
    if getattr(cls, "_bigmoe_prefill_hooked", False):
        return
    for name in ("prefill", "forward"):
        orig = getattr(cls, name)

        def wrapped(self, input_ids, params=None, _orig=orig):
            try:
                n = int(input_ids.shape[-1]) * int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
            except Exception:
                n = 0
            for p in POOLS:
                if p.elastic and p.stage_n and not p.in_prefill and n * p.topk >= p.stage_min:
                    p._elastic_enter(n)
            try:
                return _orig(self, input_ids, params)
            except BaseException:
                # a failed forward (e.g. CUDA out of memory) ends the prefill: free the staging buffers and take the
                # flex chunks back now instead of holding them until the next request's first small forward
                for p in POOLS:
                    if p.in_prefill:
                        p._elastic_exit()
                raise
        setattr(cls, name, wrapped)
    cls._bigmoe_prefill_hooked = True


def attach(model):
    """One Pool over every routed layer + the nv2 engine (device-side publish / reply, host planner, exclusive RAM
    tier, NVMe reads, optional CPU lane) + routing / combine hooks + bounce proxies."""
    from . import nv2 as N2
    by_key = {L["key"]: L for L in routed_layers()}
    mods = [m for m in find_moe(model.modules) if m.key in by_key]
    assert len(mods) == len(by_key), f"{len(mods)} MoE modules on the GPU vs {len(by_key)} store layers"
    mods.sort(key=lambda m: by_key[m.key]["li"])
    dev = torch.device(str(mods[0].device))
    reserve = float(env("EC_RESERVE_GB", "2.5")) * GiB
    mx = env("EC_MAX_SLOTS")
    admit_max = int(env("EC_ADMIT_MAX", "64"))
    stage_min = int(env("EC_STAGE_MIN", "8192"))
    elastic = float(env("EC_ELASTIC_GB", "0")) * GiB
    slots = env("EC_STAGE_SLOTS")
    stage_gb = float(env("EC_STAGE_GB", "0"))
    slot = max(al(L["trellis_bytes"]) for L in by_key.values())
    stage_n = int(slots) if slots else (int(stage_gb * GiB // slot) if stage_gb > 0 else 0)
    p = Pool(dev, mods, reserve, int(mx) if mx else None, admit_max, stage_n, stage_min, elastic)
    for li, m in enumerate(mods):
        Ly = by_key[m.key]
        sg, su, sd, ou, od = p.geom(li)
        assert (Ly["off_u"], Ly["off_d"]) == (ou, od) and sg + su + sd <= Ly["trellis_bytes"] <= p.rec, (m.key, Ly, p.geom(li))
    # mode nvme (approximate, with the CPU lane) also prefills approximately: warm prefill and the last-page split (which
    # changes how the recurrent state's prefill is chunked, so its rounding); mode nvme-exact keeps exllamav3's prefill
    approx_prefill = bool(p.stage_n) and env("NV_CPU", "0") == "1"
    if approx_prefill and env("EC_WARM_PREFILL", "0") == "1":   # the backend's warm_prefill option
        # created before nv2 sizes the RAM tier from the cgroup's free memory, which then excludes its pinned slots
        from .warm_prefill import WarmPrefill
        p.warm_prefill = WarmPrefill(p)
    POOLS.append(p)
    N2.attach_pool(p)
    if p.warm_prefill is not None:
        N2.ext().set_ring_borrow(0)   # the prefill ring would evict RAM experts that the forward counts as warm
    warm = env("EC_WARM")
    scores = json.load(open(warm)) if warm else None
    if scores:
        p.nv2.warm(scores)      # RAM first: the VRAM warm copies from RAM
        p.warm(scores)
    p.nv2.post_warm()
    p.nv2.cpu_setup()
    p.nv2.x16_setup()
    for m in mods:
        orig = m.routing_fn

        def routed(bsz, cfg, z, params, _orig=orig, _m=m):
            nv = _m._ec[0].nv2
            if nv.pfside and not params.get("autosplit_measure") and not params.get("tp_warmup"):
                nv.predict_early(_m._ec[1], z, bsz)
            sel, w = _orig(bsz, cfg, z, params)
            _m._nv2_cpu = False
            _m._x16 = None
            if not params.get("autosplit_measure") and not params.get("tp_warmup"):
                pool, li = _m._ec
                staged = pool.warm_prefill is not None and sel.numel() >= pool.stage_min
                # X16 lane: a single-token call of one of its layers keeps the original selection for the reduction
                x16 = nv.x16 is not None and bsz == 1 and li < nv.x16_hi
                orig_sw = (sel.clone(), w.clone(), z) if x16 else None
                sel, w, _m._nv2_cpu = nv.layer(li, sel, w, z, bsz)
                if staged:   # warm prefill: staged (prefill) calls only, never a single-token X16 call
                    sel, w = pool.warm_prefill.route(li, sel, w)
                if x16 and nv.stepped and _m._nv2_cpu:
                    _m._x16 = (orig_sw, sel)
                    nv.x16_start(_m)
            return sel, w
        m.routing_fn = routed
        orig_comb = m.cpu_split_combine

        def comb(fhs, cpu_partial, cpu_pending, shape, _orig=orig_comb, _m=m):
            if getattr(_m, "_nv2_cpu", False):
                _m._nv2_cpu = False
                if getattr(_m, "_x16", None) is not None:
                    fhs = _m._ec[0].nv2.x16_finish(_m, fhs)
                    _m._x16 = None
                else:
                    fhs = _m._ec[0].nv2.combine(fhs)
            return _orig(fhs, cpu_partial, cpu_pending, shape)
        m.cpu_split_combine = comb
    if p.elastic and p.stage_n:
        _hook_prefill(type(model))
    if approx_prefill:
        from . import last_page
        log(last_page.apply(p.stage_min, p.topk))
    log(f"pool: {len(mods)} MoE layers (experts {p.first}..{p.first + p.E - 1}), {p.S} slots x {p.rec / 1e6:.2f} MB = "
        f"{p.S * p.rec / GiB:.2f} GiB ({p.S / len(mods):.1f}/layer), {'uniform' if p.uniform else 'per-layer'} geometry, "
        f"admit <= {admit_max} picks, stage 2 x {p.stage_n} experts (>= {stage_min} picks), elastic "
        f"{len(p.flex_all)} chunks{', warm ' + warm if warm else ''}")
    N2.finish(model)
    return POOLS


def summary():
    from . import nv2 as N2
    return [dict(vram=p.summary(), nv2=N2.summary()) for p in POOLS]


def verify(n=32):
    return [p.nv2.verify(n) for p in POOLS]
