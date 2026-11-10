"""nv2: RAM tier, NVMe reads, planner and CPU lane of the shared engine (exact mode: every pick on the GPU path with a
device-side stall; mode nvme: + CPU lane on RAM-resident / streamed experts). Builds on tiers.py's loader (expert
trellis -> one VRAM bounce slot, nothing read) and VRAM CLOCK pool. Knobs below are <prefix>NAME (env.py).

Per MoE layer call with <= EC_STAGE_MIN picks (decode, small forwards), no host sync:
  nv_pub   device publishes the layer's unique picks (+ the MoE input rows for a CPU job) to a mapped request ring
  host     controller thread (nv2_host.cpp) drains the device's write-back / admission logs, runs plan_layer (moetier
           semantics: lanes gpu | zerocopy (= GPU-admitted RAM expert) | cpu | nvme->gpu | nvme->cpu, greedy min-max,
           NVMe misses stall = exact), starts O_DIRECT reads into fresh RAM slots, replies with the lanes, hands the
           CPU job to the persistent CPU worker
  nv_step  device waits for the reply (bounded), masks CPU-lane picks (id -1, weight 0), CLOCK admission of GPU-lane
           misses; VRAM victims get a fresh RAM landing slot (exclusive RAM tier), admissions are logged
  nv_copy  gathers admitted experts: RAM sources first, NVMe sources once their landed flag is up (per key, set by
           the host after the read completes); victim bytes go to the landing slot in the same pass
  MoE      exllamav3's kernels, unchanged, on the live pointer tables
  combine  waits for the CPU job (bounded), adds its fp32 partial
Prefill (>= stage_min picks): staging buffers filled on the copy stream from RAM slots (pinned for the forward) and,
for NVMe-only experts, from a pinned FIFO ring that a reader fills several layers ahead (NV_PF_RING slots).
Env: NV_RAM_GB (auto), NV_MARGIN_GB (1), NV_THREADS (16 readers), NV_PIECE_KB (2304),
     NV_PF_RING (128), NV_PF_QD (24), NV_EXCL (1), NV_ELASTIC_WB (1), NV_LAND (64),
     NV_FREE_LO (96), NV_CPU (0: S2; 1: S3 CPU tier), NV_NVCPU (0), NV_CPU_CPUS (2-23),
     NV_CPU_MODE (-1 auto), NV_CLAMP (1: swiglu clamp on the CPU share, B004 fix), NV_POL (cost model
     "g0,thit,tzc,ca,cb,ctok,push,nvlat,nvdeep,nvone"), NV_MAXCPU (32), NV_CTL_CPU (25),
     NV_READER_CPUS (26-39), NV_TIMEOUT_S (20), NV_COPY_GRID (48)
2026-10-20 (nv2_host.cpp): NV_TNZC (0.6 ms: GPU cost of a zero-copy NVMe pick after landing; < 0 legacy model),
     NV_PARTIAL (1: an NVMe->CPU job binds at gate+up landing, down units wait), NV_PRIO (1: reader classes
     demand > prefetch > bulk), NV_HI_MB (12) / NV_LO_MB (6) in-flight caps, NV_PF_LIFO (1),
     NV_DPIECE_KB (512: decode read pieces), NV_WB_EARLY (1: write-backs drained at nv_copy completion),
     NV_PF_MULTI (2 prefetch hints per verify-window token), NV_RZC1 (1: verify windows do not admit RAM picks
     of one token), NV_STG / NV_STG_MAX (0 / 3: layer-ahead DMA staging pool, experimental, off) with
     NV_STG_GATE (2: DMA after the copy kernel unless the MoE reads zero-copy, then after the MoE kernel),
     NV_PF_RING_BORROW (1: the prefill ring borrows RAM-tier slots during a staged prefill).
     NV_SLRU (0: the RAM tier evicts its least recently used expert; N: segmented, N % of the slots are for experts
     used again after they landed, and the ones that only landed go first. DeepSeek in mode nvme sets 80).
     Round 3: NV_STREAM (1: an NVMe->CPU pick is bound when its read starts and each gate / up / down band of the
     CPU lane waits only for the read pieces that hold it), NV_TRACK_QD (0: pieces of one decode read in flight,
     uncapped), NV_CBT (planner: CPU time left for a streamed NVMe->CPU pick after its last byte lands, ms; < 0
     legacy), NV_CTL_MWX (20000) / NV_LANE_MWX (0): MWAITX timer cycles for the controller's / CPU lane's waits
     on AMD (0 = PAUSE spins). timeline() also reports the CPU tail after the last landing of any NVMe pick and the
     lane's time per RAM expert in layers without NVMe picks.
     Diagnostics: timeline() (per-layer breakdown in summary()), ext().trace_set / trace_take (routing trace)
N137 B70 expert tier (needs the CPU tier, NV_CPU=1): B70 (0), B70_RING (/run/local-ai/shared/b70.ring,
     created by b70tier/b70srv.py on the B70), B70_N (3000 experts), B70_ORDER (score: warm-score order after the VRAM warm
     set | rr: the RAM tier's round-robin rank), B70_TIMEOUT_S (2). Decode picks of B70-resident experts are masked
     on the 3090 and computed on the B70; their RAM copies are dropped (exclusive) and RAM refills with colder experts.
"""
import collections, concurrent.futures as cf, ctypes, json, mmap, os, signal, threading, time
import numpy as np
import torch

from .env import env, build_dir
from . import tiers as NT

GiB = 1024 ** 3
_EXT = None
NV = None
VERIFY = [False]   # BCP: set by the server around generator verify forwards (lossy-policy eligible)


def _cpus(spec):
    out = []
    for part in spec.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _vbmi_on():
    try:
        return bool(ext().cpu_knob("vbmi", -1))
    except Exception:
        return False


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        root = os.path.dirname(os.path.abspath(__file__))
        here = os.path.join(root, "kernels", "nv2")   # nv2_host.cpp, nv2_dev.cu, nv2_shared.h, ft_core.h, ft_n135.h
        d = build_dir("NV2", os.path.join(root, "build", "nv2"))
        os.makedirs(d, exist_ok=True)
        _EXT = load(name="bigmoe_nv2", sources=[os.path.join(here, "nv2_host.cpp"), os.path.join(here, "nv2_dev.cu")],
                    build_directory=d, extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-mtune=znver3", "-std=c++17",
                                                     "-I" + here],
                    extra_cuda_cflags=["-O3", "-I" + here], extra_ldflags=["-lpthread"], verbose=False)
    return _EXT


def _mapped(nbytes, dev_idx, tag):
    """One page-aligned anon region, cudaHostRegister(Mapped); UVA alias == host address (checked)."""
    from exllamav3.ext import exllamav3_ext as xe
    nbytes = (nbytes + 4095) // 4096 * 4096
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    with torch.cuda.device(dev_idx):
        r = torch.cuda.cudart().cudaHostRegister(a, nbytes, 2)
    assert int(r) == 0, f"cudaHostRegister failed ({tag}): {r}"
    t = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)
    assert xe.pinned_cuda_view(t[:4096], dev_idx).data_ptr() == a, "UVA alias != host address"
    t.zero_()
    return mm, a, t


class _Arenas:
    """Several HostArenas as one slot id space (RAM slot classes): addr[s], view(s, n)."""

    def __init__(self, arenas):
        self.arenas = arenas
        self.addr = np.concatenate([a.addr for a in arenas]) if arenas else np.zeros(0, np.int64)
        self.n = len(self.addr)
        self.base = np.cumsum([0] + [a.n for a in arenas])

    def view(self, s, n=None):
        i = int(np.searchsorted(self.base, s, side="right") - 1)
        a = self.arenas[i]
        return a.view(s - int(self.base[i]), n or a.slot)


class Nv2:
    def __init__(self, pool):
        e = ext()
        sz = e.sizes()
        self.SREQ, self.SREP, self.SLOG, self.MAXU, self.MAXP, self.RQ, self.LG, self.LR, self.MAXB, HCN, DCN, STN, DBG, DBGW = sz
        self.DBGW = DBGW
        self.pool = pool
        self.dev = pool.device
        dev_idx = self.dev.index or 0
        man = NT.manifest()
        self.store = man["_store"]
        self.slot = pool.rec                      # uniform slot: the largest layer's trellis (store v2)
        assert pool.first == 0 and pool.E == man["experts_per_layer"]
        li_of = {L["key"]: L for L in man["layers"]}
        assert [li_of[m.key]["li"] for m in pool.mods] == list(range(len(pool.mods))), "pool layer order != store order"
        geom = []
        for li, m in enumerate(pool.mods):
            Ly = li_of[m.key]
            sg, su, sd, ou, od = pool.geom(li)
            assert (Ly["trellis_bytes"], Ly["off_u"], Ly["off_d"]) == (sg + su + sd, ou, od), (m.key, Ly, pool.geom(li))
            geom.append([Ly["base"], Ly["rec_bytes"], Ly["trellis_bytes"], ou, od])
        self.geomt = torch.tensor(geom, dtype=torch.int64)
        self.klen = np.repeat(self.geomt[:, 2].numpy(), pool.E)
        self.L, self.E = len(pool.mods), pool.E
        self.K = self.L * self.E
        m0 = pool.mods[0]
        self.H, self.I = m0.hidden_size, m0.intermediate_size
        self.act_limit = float(getattr(m0, "act_limit", 0.0) or 0.0)
        self.topk = int(m0.num_experts_per_tok)
        # predictor activation per layer, from the routing function exllamav3 picked (before the engine hooks it)
        from exllamav3.modules import block_sparse_mlp_routing as R
        self.route_act = [getattr(R, "ROUTING_ACT_SQRTSP", 1) if "sqrtsp" in getattr(m.routing_fn, "__name__", "") else
                          R.ROUTING_ACT_SIGMOID for m in pool.mods]
        g = env
        self.cpu_on = g("NV_CPU", "0") == "1"
        self.x16, self.x16_hi, self.stepped = None, 0, False   # X16 lane (x16_setup)
        self.b70 = g("B70", "0") == "1"
        assert not self.b70 or self.cpu_on, "B70=1 needs the CPU tier (NV_CPU=1): the CPU worker runs the B70 lane"
        self.b70_ring = g("B70_RING", "/run/local-ai/shared/b70.ring")
        self.b70_timeout = float(g("B70_TIMEOUT_S", "2"))
        self.b70_keys = []
        self.timeout_ns = int(float(g("NV_TIMEOUT_S", "5")) * 1e9)
        self.grid = int(g("NV_COPY_GRID", "48"))
        self.wb_on = g("NV_EXCL", "1") == "1"
        self.prefetch = g("NV_PREFETCH", "0") == "1"
        # N136: the layer-ahead routing guess runs on a side stream concurrently with this layer's router (both read the
        # same MoE input; 36-block GEMVs on an 82-SM card), instead of after it on the compute stream (NV_PFSIDE=1;
        # single-row decode only, see predict_early)
        self.pfside = self.prefetch and g("NV_PFSIDE", "0") == "1"
        self._pfe = None
        if self.pfside:
            self.side = torch.cuda.Stream(self.dev)
            self._pfev = [torch.cuda.Event() for _ in range(8)]
            self._pfev_i = 0
        self._pfb = {}
        self._pfb = {}
        self.elastic_wb = self.wb_on and g("NV_ELASTIC_WB", "1") == "1"
        self.fd = os.open(self.store, os.O_RDONLY | os.O_DIRECT)
        self.ph_t = torch.zeros(self.slot, dtype=torch.uint8, device=self.dev)
        self.ph = self.ph_t.data_ptr()
        self.bounce_t, self.bounce = NT._BOUNCE[dev_idx]
        self.st = collections.Counter()
        # ---- CPU tier scale copies first (charged to the cap before the RAM tier is sized)
        self.scales = None
        if self.cpu_on:
            sc = lambda lins, attr: torch.stack([getattr(l.inner, attr).view(-1) for l in lins]).half().cpu().contiguous()
            self.scales = [(sc(m.multi_gate.linears, "suh"), sc(m.multi_gate.linears, "svh"), sc(m.multi_up.linears, "suh"),
                            sc(m.multi_up.linears, "svh"), sc(m.multi_down.linears, "suh"), sc(m.multi_down.linears, "svh"))
                           for m in pool.mods]
            NT._log(f"CPU tier scale copies: {sum(t.numel() * 2 for s in self.scales for t in s) / GiB:.2f} GiB")
        # ---- mapped control region
        H = self.H
        lay = [("hc", HCN * 8), ("req", self.RQ * self.SREQ), ("rep", self.RQ * self.SREP), ("wbl", self.LG * self.SLOG),
               ("adml", self.LG * self.SLOG), ("land", self.LR * 4), ("home", self.K * 24), ("res", self.K * 4),
               ("hx", self.MAXB * H * 2), ("hout", self.MAXB * H * 4)]
        off, self.off = 0, {}
        for k, n in lay:
            self.off[k] = off
            off += (n + 4095) // 4096 * 4096
        self.cmm, self.cbase_addr, self.ct = _mapped(off, dev_idx, "control")
        A = lambda k: self.cbase_addr + self.off[k]
        self.a = {k: A(k) for k, _ in lay}
        self.hc = self.ct[self.off["hc"]: self.off["hc"] + HCN * 8].view(torch.int64)
        self.home = self.ct[self.off["home"]: self.off["home"] + self.K * 24].view(torch.int64)
        self.res = self.ct[self.off["res"]: self.off["res"] + self.K * 4].view(torch.int32)
        # ---- RAM tier + prefill ring sizes
        ring_pf = self.ring_pf = int(g("NV_PF_RING", "128"))
        # the prefill ring borrows RAM-tier slots for the duration of a staged prefill instead of owning its own memory
        self.ring_borrow = g("NV_PF_RING_BORROW", "1") == "1"
        ring_own = 1 if self.ring_borrow else ring_pf
        margin = float(g("NV_MARGIN_GB", "1")) * GiB
        mem = NT.cgroup_mem()
        want = g("NV_RAM_GB", "auto")
        if want == "auto":
            assert mem.get("max"), "NV_RAM_GB=auto needs a memory-capped container"
            avail = (mem["max"] - mem["current"]) * GiB - margin - ring_own * self.slot - NT._rcs_bytes() - 0.3 * GiB
        else:
            avail = float(want) * GiB
        avail = max(0, avail)
        # RAM slot classes (NV_RAMCLS=1): full slots for the large-record layers, small slots for the layers whose
        # trellis fits the smallest layer size, the same number of slots per layer either way
        lens = self.geomt[:, 2].tolist()
        self.small_bytes = int(min(lens))
        n_small_layers = sum(1 for x in lens if x <= self.small_bytes)
        if g("NV_RAMCLS", "1") == "1" and self.small_bytes < self.slot and n_small_layers < len(lens):
            per_layer = avail / (self.slot * (len(lens) - n_small_layers) + self.small_bytes * n_small_layers)
            n_full, n_small = int(per_layer * (len(lens) - n_small_layers)), int(per_layer * n_small_layers)
        else:
            n_full, n_small, self.small_bytes = int(avail // self.slot), 0, 0
        nram = n_full + n_small
        NT._log(f"nv2: rcs reserve {NT._rcs_bytes() / GiB:.1f} GiB, margin {margin / GiB:.1f} GiB, prefill ring {ring_pf} "
                f"slots ({'borrowed from the RAM tier' if self.ring_borrow else f'{ring_pf * self.slot / GiB:.2f} GiB'}) -> "
                f"RAM tier {nram} slots ({n_full} x {self.slot / 1e6:.2f} MB + {n_small} x {self.small_bytes / 1e6:.2f} MB = "
                f"{(n_full * self.slot + n_small * self.small_bytes) / GiB:.2f} GiB)")
        self.ram = _Arenas([NT.HostArena(n_full, self.slot, 512, dev_idx, "ram")] +
                           ([NT.HostArena(n_small, self.small_bytes, 512, dev_idx, "ram_small")] if n_small else []))
        self.slot_cls = torch.tensor([0] * n_full + [1] * n_small, dtype=torch.int8)
        self.nram = nram
        self.pf = NT.HostArena(ring_own, self.slot, ring_own, dev_idx, "pfring")
        NT._log(f"nv2: RAM arena + prefill ring registered")
        # VRAM victim ring (write-back source: nv_copy parks the victim's bytes here D2D, the host drains it D2H on a
        # copy engine; SM stores to host memory run at ~4 GB/s vs 25 GB/s for the DMA engine)
        nvr = int(g("NV_VRING", "32")) if self.wb_on else 0
        self.vring = torch.empty((max(nvr, 1), self.slot), dtype=torch.uint8, device=self.dev) if nvr else None
        self.vring_addr = [self.vring[i].data_ptr() for i in range(nvr)] if nvr else []
        self.ram_addr_dev = torch.tensor(self.vring_addr or [0], dtype=torch.int64, device=self.dev)
        # layer-ahead VRAM staging pool (NV_STG slots, NV_STG_MAX per layer; 0 = off)
        nstg = int(g("NV_STG", "0"))
        self.stg = torch.empty((max(nstg, 1), self.slot), dtype=torch.uint8, device=self.dev) if nstg else None
        self.stg_addr = [self.stg[i].data_ptr() for i in range(nstg)] if nstg else []
        self.stg_addr_dev = torch.tensor(self.stg_addr or [0], dtype=torch.int64, device=self.dev)
        # ---- device scratch
        D = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=self.dev)
        self.dc, self.uidx, self.uexp, self.ulane = D(DCN, torch.int64), D(self.E), D(self.MAXU), D(self.MAXU)
        self.jobx, self.jobnv = D(pool.jobs.numel() // 2), D(pool.jobs.numel() // 2)
        self.jobsrc = D(pool.jobs.numel() // 2, torch.int64)
        # dflag[0]: this call's CPU job seq (0 = none); [1 + li]: nv_step's per-layer request records (X16 lane)
        self.dflag, self.dstats = D(128, torch.int64), D(STN, torch.int64)
        self.dbg = D(DBG * DBGW, torch.int64)
        self.sel_out = torch.empty(16384, dtype=torch.int64, device=self.dev)
        self.w_out = torch.empty(16384, dtype=torch.half, device=self.dev)
        # ---- scores (tie-break: colder first to the CPU)
        sp = g("EC_WARM", "")
        self.scores = json.load(open(sp)) if sp else {}
        sc = np.zeros(self.K, np.float32)
        for li, m in enumerate(pool.mods):
            s = self.scores.get(m.key)
            if s is not None:
                sc[li * self.E:(li + 1) * self.E] = np.asarray(s[:self.E], np.float32)
        # ---- host engine
        rcpus = _cpus(g("NV_READER_CPUS", "26-39"))
        e.init(self.L, self.E, H, self.I, self.geomt, self.slot, int(g("NV_PIECE_KB", "2304")) * 1024,
               pool.off_u, pool.off_d, self.fd, self.ph, self.a["hc"], self.a["req"], self.a["rep"], self.a["wbl"],
               self.a["adml"], self.a["land"], self.a["home"], self.a["res"], self.a["hx"], self.a["hout"],
               torch.tensor(self.ram.addr.tolist(), dtype=torch.int64), torch.tensor(self.pf.addr.tolist(), dtype=torch.int64),
               torch.from_numpy(sc), int(g("NV_THREADS", "16")), rcpus, int(g("NV_CTL_CPU", "25")),
               int(self.wb_on), int(g("NV_LAND", "64")), int(g("NV_FREE_LO", "96")),
               torch.tensor(self.vring_addr, dtype=torch.int64), self.slot_cls, int(self.small_bytes))
        e.set_stage_qd(int(g("NV_PF_QD", "24")))
        if self.ring_borrow:
            e.set_ring_borrow(ring_pf)
        if self.stg_addr:
            e.stg_init(torch.tensor(self.stg_addr, dtype=torch.int64), int(g("NV_STG_MAX", "3")))
        pol = [float(x) for x in g("NV_POL", "0.17,0.023,0.38,0.145,0.165,0.040,0.40,0.12,0.395,0.75").split(",")]
        self.pol = pol + [1.0 if self.cpu_on else 0.0, 1.0 if g("NV_NVCPU", "1" if self.cpu_on else "0") == "1" else 0.0,
                          float(g("NV_MAXCPU", "32")), 1.0 if g("NV_NOADMIT", "1" if self.cpu_on else "0") == "1" else 0.0]
        e.set_pol(self.pol)
        # ---- the pool reads homes from the mapped table from now on (ec_step / ec_copy / ec_evict take its address)
        pool.home = self.home.view(self.L, self.E, 3)
        pool.home_cpu = pool.home
        pool.nv2 = self
        e.nv_rows_all(self.L, self.E, pool.slotof, self.a["home"], pool.tabs)
        torch.cuda.synchronize(self.dev)
        if getattr(pool, "ev_ready", None) is not None:
            for ev in pool.ev_ready:
                ev.record(torch.cuda.current_stream(self.dev))
        self.bg = cf.ThreadPoolExecutor(1, thread_name_prefix="nv2-stage")
        e.start()
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()
        NT._log(f"nv2 engine up: {self.L} layers x {self.E}, readers {g('NV_THREADS', '16')} on {rcpus[:3]}.., "
                f"exclusive {self.wb_on}, CPU tier {self.cpu_on}")

    # ------------------------------------------------------------------------------------------------------------
    def _watchdog(self):
        last, quiet = None, 0
        while True:
            time.sleep(1.0)
            pub = int(self.hc[0])
            quiet = quiet + 1 if pub == last else 0
            last = pub
            if quiet == 10 and int(self.hc[0]) != int(self.hc[1]):
                print(f" !! nv2 watchdog: no progress 10 s; host diag {ext().diag()} dc {ext().peek(self.dc.data_ptr(), 8)}", flush=True)
            err = int(self.hc[2])
            if err:
                print(f" !! nv2 ERROR flag {err} (1 reply timeout, 2 ring full, 3 fix wait, 4 copy wait, 5 CPU wait, 16-19 "
                      f"read/alloc); host diag {ext().diag()} dc {ext().peek(self.dc.data_ptr(), 8)} counters {self.counters()} "
                      f"dev {ext().peek(self.dstats.data_ptr(), 16)}", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(30)
                os._exit(70)

    def counters(self):
        names = ["reqs", "dec_reqs", "small_reqs", "picks", "vram", "ram", "nvme", "inflight_hit", "lane_zc", "lane_cpu",
                 "lane_nvg", "lane_nvc", "reads", "read_bytes", "wb_landed", "wb_dup", "adm_freed", "adm_pinned",
                 "evictions", "cpu_jobs", "cpu_experts", "cpu_tokens", "stage_ram", "stage_nv", "stage_reads", "read_err",
                 "alloc_fail", "land_pushed", "prefetch", "plan_ns", "ctl_busy_ns", "cpu_busy_ns", "cpu_wait_land_ns",
                 "read_ms_x1000", "stage_wait_ns", "stage_job_ns", "wb_bulk", "dec_tok", "notice_ns", "notice_max_ns", "serve_max_ns", "reply_ns", "wb_issued", "wb_drop", "vring_hits", "lane_b70", "b70_jobs", "b70_rows", "b70_wait_ns", "b70_rtt_ns", "b70_err", "b70_rtt_max_ns", "pf_keys", "pf_skip", "pf_used", "pf_wasted", "lane_skip", "lossy_calls", "lossy_tok", "lossy_tok_skip", "lossy_cold_x1e6", "lossy_skip_x1e6", "lossy_coldu", "lossy_want", "lossy_bg_reads", "promoted", "stg_issued", "stg_ready", "stg_hit", "stg_late",
                 "stg_nofree", "stg_dup", "stg_waste", "wb_early", "partial_lands", "partial_binds", "wb_lat_us",
                 "wb_lat_max_us", "ring_low", "ring_low_vfree", "ring_borrowed",
                 "x16_jobs", "x16_experts", "x16_gu_ns", "x16_wait_ns", "x16_down_ns",
                 "resident", "inflight", "free", "landing"]
        return dict(zip(names, ext().counters()))

    # ---- warm start: RAM by score (incl. the VRAM warm set), VRAM warm, exclusive: drop VRAM copies, refill RAM
    def warm(self, scores):
        p = self.pool
        order = []
        for li, m in enumerate(p.mods):
            sc = scores.get(m.key)
            o = sorted(range(self.E), key=lambda x: -sc[x]) if sc is not None else list(range(self.E))
            order.append([li * self.E + x for x in o])
        rank = []    # round-robin by rank across layers (top-1 of every layer first)
        for r in range(self.E):
            rank += [order[li][r] for li in range(self.L)]
        # per RAM slot class: at most that class's slot count (a class never evicts its own warm reads)
        lens = self.geomt[:, 2].numpy()
        ncls = [int((self.slot_cls == 0).sum()), int((self.slot_cls == 1).sum())]
        cls_of = lambda k: 1 if (ncls[1] and lens[k // self.E] <= self.small_bytes) else 0
        take, used = [], [0, 0]
        for k in rank:
            c = cls_of(k)
            if used[c] < ncls[c]:
                take.append(k); used[c] += 1
        t = time.perf_counter()
        n1 = ext().warm_read(torch.tensor(take, dtype=torch.int64))
        ext().nv_rows_all(self.L, self.E, p.slotof, self.a["home"], p.tabs)
        torch.cuda.synchronize(self.dev)
        NT._log(f"nv2 RAM warm: {n1} experts in {time.perf_counter() - t:.1f} s")
        self._rank = rank

    def post_warm(self):
        p = self.pool
        torch.cuda.synchronize(self.dev)
        so = p.slotof.flatten().cpu().numpy()
        vk = np.nonzero(so >= 0)[0]
        self.b70_keys = self._b70_pick(so) if self.b70 else []
        b70set = set(self.b70_keys)
        nd = 0
        if self.wb_on:
            drop = np.concatenate([vk.astype(np.int64), np.asarray(self.b70_keys, np.int64)])
            nd = ext().drop_keys(torch.from_numpy(drop))
            rest = [k for k in self._rank if so[k] < 0 and k not in b70set]
            st = ext().state()
            ks = st[1].numpy()
            rest = [k for k in rest if ks[k] == 0][: nd]
            if rest:
                ext().warm_read(torch.tensor(rest, dtype=torch.int64))
        ext().nv_rows_all(self.L, self.E, p.slotof, self.a["home"], p.tabs)
        torch.cuda.synchronize(self.dev)
        if self.b70:
            ms = ext().b70_attach(self.b70_ring, torch.tensor(self.b70_keys, dtype=torch.int64), 900.0)
            ext().b70_set(1, self.b70_timeout)
            NT._log(f"nv2 B70 tier: {len(self.b70_keys)} experts loaded on the B70 server in {ms / 1e3:.1f} s ({self.b70_ring})")
        ext().reset_counters()
        self.dstats.zero_()
        c = self.counters()
        NT._log(f"nv2 post-warm: {len(vk)} VRAM-resident, {nd} RAM copies of them dropped (exclusive), RAM resident "
                f"{c['resident']}, free {c['free']}")

    def _b70_pick(self, so):
        """B70 key set: the next experts after the 3090's VRAM warm set, by warm score (or the RAM tier's rank order)."""
        n = int(env("B70_N", "3000"))
        if env("B70_ORDER", "score") == "rr":
            order = self._rank
        else:
            sc = np.full(self.K, -1.0)
            for li, m in enumerate(self.pool.mods):
                s = self.scores.get(m.key)
                if s is not None:
                    sc[li * self.E:(li + 1) * self.E] = np.asarray(s[:self.E], np.float64)
            order = [int(k) for k in np.argsort(-sc, kind="stable") if sc[k] >= 0]
        return [k for k in order if so[k] < 0][:n]

    def set_b70(self, on):
        """Runtime switch (decode-KL harness): on=False -> B70-resident picks take the normal RAM/NVMe lanes (exact)."""
        if self.b70:
            ext().b70_set(int(bool(on)), self.b70_timeout)

    # ---- per MoE layer call (decode / small forwards) -----------------------------------------------------------
    def layer(self, li, sel, w, z, bsz):
        p = self.pool
        e = ext()
        n = sel.numel()
        if p.stage_n and n >= p.stage_min:
            p._stage_hook_nv2(li, n)
            s = sel.reshape(-1)
            if s.dtype != torch.long or not s.is_contiguous():
                s = s.long().contiguous()
            p._ec_hits(li, s)
            self.st["staged_calls"] += 1
            self.stepped = False
            return sel, w, False
        if p.stage_n:
            p._stage_hook_nv2(li, n)
        dec = n <= p.admit_max
        s = sel.reshape(-1)
        if s.dtype != torch.long or not s.is_contiguous():
            s = s.long().contiguous()
        ww = w.reshape(-1)
        if ww.dtype != torch.half or not ww.is_contiguous():
            ww = ww.half().contiguous()
        send_x = self.cpu_on and dec and bsz <= self.MAXB and n <= self.MAXP
        zp = 0
        if send_x:
            zz = z.view(bsz, -1)
            if zz.dtype != torch.half or not zz.is_contiguous():
                zz = zz.half().contiguous()
            self._zkeep = zz
            zp = zz.data_ptr()
        pf, npf, son = 0, 0, p.slotof[li]
        pre, self._pfe = self._pfe, None
        if self.prefetch and dec and li + 1 < self.L and bsz <= 8:
            if pre is not None and pre[0] == li and pre[1] == bsz:
                torch.cuda.current_stream(self.dev).wait_event(pre[3])
                pf = pre[2]
                self.st["pf_side"] += 1
            else:
                pf = self._predict(li + 1, z, bsz)
            npf = bsz * self.topk
            son = p.slotof[li + 1]
        lgp = self._lg_ptr(li) if (dec and bsz == 1) else 0
        plgp = self._pfb[1][0].data_ptr() if (pf and bsz == 1 and 1 in self._pfb) else 0
        e.nv_pub(s, ww, zp, n, bsz, self.H, li, self.E, p.first, p.slotof[li], (2 if VERIFY[0] else 0) if dec else 1, int(send_x),
                 self.a["hc"], self.a["req"], self.a["hx"], self.dc, self.uidx, self.uexp, self.timeout_ns, pf, npf, son, lgp, plgp)
        so, wo = self.sel_out[:n], self.w_out[:n]
        sg_, su_, sd_, ou_, od_ = p.geom(li)
        e.nv_step(s, ww, so, wo, n, p.first, li, self.E, p.S, int(dec), p.slotof, self.a["home"], self.a["res"], p.tabs,
                  p.owner, p.refb, p.pin, p.ctl, p.jobs, self.jobx, self.jobnv, self.dstats, p.cbase.data_ptr(), p.spc,
                  p.rec, ou_, od_, self.a["hc"], self.a["rep"], self.dc, self.uidx, self.ulane, self.a["wbl"],
                  self.a["adml"], self.a["land"], self.dflag, self.timeout_ns, int(self.wb_on), self.dbg,
                  self.stg_addr_dev.data_ptr() if self.stg_addr else 0, self.jobsrc)
        if self.x16 is not None and li < self.x16_hi and bsz == 1:
            # X16 epilogue gate: this call's nv_step is done (its masked selection and request record exist)
            b = self.x16
            b["cur_ev"] = b["ev"][b["evi"]]; b["evi"] = (b["evi"] + 1) % len(b["ev"])
            b["cur_ev"].record(torch.cuda.current_stream(self.dev))
        e.nv_copy(p.ctl, p.jobs, self.jobx, self.jobnv, li, self.E, self.a["home"], self.a["res"], p.cbase.data_ptr(), p.spc,
                  p.rec, sg_, su_, sd_, ou_, od_, self.ram_addr_dev, self.dc, self.uexp, self.ulane, p.slotof,
                  p.tabs, self.ph, self.a["hc"], self.timeout_ns, self.dstats, self.grid, self.dbg, self.jobsrc,
                  int(p.uniform))
        self.st["dec_calls" if dec else "small_calls"] += 1
        self.stepped = True
        return so.view(sel.shape), wo.view(w.shape), send_x

    def _lg_ptr(self, li):
        """Device address of layer li's single-row router logits (exllamav3 routing_dots writes them for bsz 1)."""
        lp = getattr(self, "_lgp", None)
        if lp is None:
            lp = self._lgp = []
            for m in self.pool.mods:
                t = getattr(getattr(m, "routing_cfg", None), "router_logits_bsz1", None)
                ok = t is not None and t.is_cuda and t.dtype == torch.half and t.numel() >= self.E and t.is_contiguous()
                lp.append(t.data_ptr() if ok else 0)
        return lp[li]

    def predict_early(self, li, z, bsz):
        """N136: launch layer li+1's routing guess on the side stream before layer li's router runs (decode only)."""
        # bsz 1 only: for more rows exllamav3's routing projection is routing_gemm_det, whose decode-class calls share
        # per-device static workspaces, so a concurrent guess on the side stream would corrupt the layer's own routing
        # (N136 AB1: C4 streams looped). The single-row GEMV path has no shared scratch.
        if not self.pfside or li + 1 >= self.L or bsz != 1:
            return
        main = torch.cuda.current_stream(self.dev)
        e0 = self._pfev[self._pfev_i]; e1 = self._pfev[self._pfev_i + 1]
        self._pfev_i = (self._pfev_i + 2) % len(self._pfev)
        e0.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(e0)
            ptr = self._predict(li + 1, z, bsz)
            e1.record(self.side)
        z.record_stream(self.side)
        if self._pfz is not z:
            self._pfz.record_stream(self.side)
        self._pfe = (li, bsz, ptr, e1)

    def _predict(self, lj, z, bsz):
        """Layer-ahead routing guess: layer lj's router (sigmoid + selection bias, top-8) on layer lj-1's MoE input,
        one fused exllamav3 routing kernel into private buffers. Residency hint only (never changes the computation)."""
        from exllamav3.modules import block_sparse_mlp_routing as R
        cfg = self.pool.mods[lj].routing_cfg
        b = self._pfb.get(bsz)
        if b is None:
            b = self._pfb[bsz] = (torch.empty((bsz, self.E), dtype=torch.half, device=self.dev),
                                  torch.empty((bsz, self.topk), dtype=torch.long, device=self.dev),
                                  torch.empty((bsz, self.topk), dtype=torch.half, device=self.dev))
        zz = z.view(bsz, -1)
        if zz.dtype != torch.half or not zz.is_contiguous():
            zz = zz.half().contiguous()
        R._gate_t(cfg)
        R.ext.routing_ds3_nogroup(zz, cfg.gate_tensor, b[0], R._esb_h(cfg), b[1], b[2], cfg.routed_scaling_factor,
                                  cfg.gate_tensor_t, self.route_act[lj], cfg.gate_i8, cfg.gate_sb)
        self._pfz = zz
        return b[1].data_ptr()

    def combine(self, fhs):
        f = fhs if fhs.is_contiguous() else fhs.contiguous()
        ext().nv_combine(f, self.dflag, self.a["hc"] + 6 * 8, self.a["hout"], self.dstats, self.a["hc"], self.timeout_ns,
                         self.dbg, self.dc)
        return f

    # ---- CPU tier -------------------------------------------------------------------------------------------------
    def cpu_setup(self):
        if not self.cpu_on:
            return
        e = ext()
        cpus = _cpus(env("NV_CPU_CPUS", "2-23"))
        threads = int(env("NV_CPU_THREADS", "0")) or len(cpus)
        mode = int(env("NV_CPU_MODE", "-1"))
        lim = self.act_limit if env("NV_CLAMP", "1") == "1" else 0.0
        e.cpu_init(threads, cpus, mode, lim)
        for li, s in enumerate(self.scales):
            m = self.pool.mods[li]
            e.cpu_add_layer(li, *s, [float(m.multi_gate.K), float(m.multi_up.K), float(m.multi_down.K)])
        self.selftest = self._selftest(lim)
        NT._log(f"nv2 CPU tier: {threads} threads on {cpus[0]}-{cpus[-1]}, mode {mode}, swiglu clamp {lim}, selftest "
                f"{self.selftest}")
        e.cpu_start(cpus[0])
        self.cpu_mode, self.cpu_lim = mode, lim

    # ---- X16 lane: bit-exact CPU experts in the first NV_X16_HI MoE layers (x16.py) -----------------------------------
    def x16_setup(self):
        want = int(env("NV_X16_HI", "0"))
        if not want:
            return
        if not self.cpu_on:
            NT._log("nv2 X16 lane: off (needs the CPU tier, NV_CPU=1)")
            return
        from . import x16 as X
        hi = 0
        for li in range(min(want, self.L)):
            why = X.unsupported(self.pool.mods[li])
            if why:
                NT._log(f"nv2 X16 lane: layer {li} not supported ({why})")
                break
            hi = li + 1
        cfg = self.pool.mods[0].experts_cfg
        I, Ho, k = cfg.interm_g.shape[-1], cfg.out_d.shape[-1], self.topk
        pin = lambda n, dt: torch.zeros(n, dtype=dt, pin_memory=True)
        b = dict(gg=pin(k * I, torch.half), gu=pin(k * I, torch.half), act=pin(k * I, torch.half),
                 d=pin(k * Ho, torch.float), flags=pin(4, torch.int64), I=I, Ho=Ho)
        cost = int(env("NV_X16_COST", "550"))
        hi = ext().x16_init(b["gg"].data_ptr(), b["gu"].data_ptr(), b["act"].data_ptr(), b["d"].data_ptr(),
                            b["flags"].data_ptr(), hi, cost) if hi else 0
        if not hi:
            NT._log("nv2 X16 lane: off (no supported layers: needs AVX-512 VBMI and EXL3 K 2 / 2.5 / 3 mul1 experts)")
            return
        D = lambda n, dt=torch.int64: torch.zeros(n, dtype=dt, device=self.dev)
        b.update(err=D(1, torch.int32), last=D(128), started=D(1), ticket=0, timeout=self.timeout_ns,
                 side=torch.cuda.ExternalStream(X.ext().x16_stream(self.dev.index), device=self.dev),
                 ev=[torch.cuda.Event() for _ in range(4)], evi=0)
        # the epilogue launches pair with nv_step calls from here on: start past the records written so far
        torch.cuda.synchronize(self.dev)
        b["last"][:127].copy_(self.dflag[1:].abs() // 64)
        t = time.perf_counter()
        for li in range(hi):   # every layer: its expert formats / sizes may differ
            ok, info = self.x16_selftest(b, li)
            if not ok:
                ext().x16_init(0, 0, 0, 0, 0, 0, cost)
                NT._log(f"nv2 X16 lane: OFF, self-test not bit-exact ({info})")
                return
        self.x16, self.x16_hi = b, hi
        NT._log(f"nv2 X16 lane: MoE layers 0-{hi - 1} (bit-exact CPU experts, planner cost {cost} %), self-test: every "
                f"layer bit-identical with {(self.topk + 1) // 2} of {self.topk} picks on the CPU "
                f"({time.perf_counter() - t:.1f} s)")

    def _x16_mod(self, m):
        """Per-module constants of the epilogue / reduction (cached on the module)."""
        c = getattr(m, "_x16_c", None)
        if c is None:
            from . import x16 as X
            shg = getattr(m, "shared_gate", None) if getattr(m, "bc_sh_exp", False) else None
            sho = getattr(m, "_x16_sh_out", None) if getattr(m, "bc_sh_exp", False) else None
            c = m._x16_c = dict(act=X.ACT[m.activation_fn], lim=float(m.act_limit or 0.0), gated=bool(m.gated),
                                shw=shg.inner.weight.reshape(-1) if shg is not None else None,
                                sho=sho.reshape(-1, self.H)[0] if sho is not None else None)
        return c

    def x16_start(self, m):
        """Right after a single-token X16-layer call's publish / step (a CPU job may be planned): exllamav3's activation
        epilogue for the CPU's picks, as ONE block on a high-priority non-blocking side stream gated by this call's
        nv_step, plus a main-stream gate that holds the layer's MoE kernels until that block is resident (so it answers as
        soon as the CPU's gate / up rows land; it is never resident outside this layer call, exllamav3's prefill GEMMs
        are cooperative launches). The block finds its call through nv_step's request record dflag[1 + li]."""
        from . import x16 as X
        (sel, w, _z), msel = m._x16
        b, c = self.x16, self._x16_mod(m)
        li = m._ec[1]
        sel, msel, w = sel.reshape(-1), msel.reshape(-1), w.reshape(-1)
        if w.dtype != torch.half:
            w = w.half()
        f = b["flags"].data_ptr()
        b["ticket"] += 1
        st = b["started"].data_ptr()
        with torch.cuda.stream(b["side"]):
            b["side"].wait_event(b["cur_ev"])
            X.ext().x16_epa(b["gg"].data_ptr(), b["gu"].data_ptr(), b["I"], sel, msel, w, m.multi_gate.ptrs_svh,
                            m.multi_up.ptrs_svh, m.multi_down.ptrs_suh, c["act"], c["lim"], c["gated"],
                            b["act"].data_ptr(), self.topk, self.dflag.data_ptr() + 8 * (1 + li),
                            b["last"].data_ptr() + 8 * li, st, b["ticket"], f, b["timeout"], f + 8, b["err"])
        X.ext().x16_gate(st, b["ticket"], b["timeout"], b["err"])
        for t in (sel, w):
            t.record_stream(b["side"])

    def x16_finish(self, m, fhs):
        """After the MoE kernel of an X16-layer call with CPU picks: exllamav3's token reduction over every slot (the
        GPU's down outputs and the CPU's) into fhs; x16_red waits for the CPU's down rows."""
        from . import x16 as X
        (sel, w, z), msel = m._x16
        b, c = self.x16, self._x16_mod(m)
        sel, msel, w = sel.reshape(-1), msel.reshape(-1), w.reshape(-1)
        if w.dtype != torch.half:
            w = w.half()
        zz = z.reshape(-1)
        if zz.dtype != torch.half:
            zz = zz.half()
        out = fhs if fhs.is_contiguous() else fhs.contiguous()
        X.ext().x16_red(m.experts_cfg.out_d.reshape(-1, b["Ho"]), b["d"].data_ptr(), msel, sel, w, self.topk, self.H,
                        b["Ho"], self.H, m.multi_down.ptrs_svh, zz, c["shw"], c["sho"], out.reshape(-1, self.H),
                        self.dflag, b["flags"].data_ptr() + 16, b["timeout"], b["err"])
        return out

    @torch.inference_mode()
    def x16_selftest(self, b, li):
        """One token through layer li with the first half of its picks on the X16 path (CPU GEMVs + x16_epa + x16_red,
        synchronously), the rest on the GPU kernel, against the GPU kernel on all picks: must be bit-identical."""
        from . import x16 as X
        e, xe = ext(), X.ext()
        m = self.pool.mods[li]
        st = e.state()[1].numpy()
        res = [x for x in range(self.E) if st[li * self.E + x] == 2]
        k = self.topk
        if len(res) < k:
            return False, f"layer {li}: only {len(res)} RAM-resident experts"
        g = torch.Generator(device="cpu").manual_seed(16)
        pick = [res[i] for i in torch.randperm(len(res), generator=g)[:k].tolist()]
        z = (torch.randn(1, self.H, generator=g) * 0.5).half().to(self.dev)
        sel = torch.tensor([pick], dtype=torch.long, device=self.dev)
        rw = (torch.rand(1, k, generator=g) * 0.5 + 0.1).half().to(self.dev)
        cfg = m.experts_cfg
        m.bc.run_bszN(z, sel, rw)
        ref = cfg.out_bszn[:1].clone()
        xs = list(range((k + 1) // 2))
        msel, mw = sel.clone(), rw.clone()
        msel[0, xs] = -1
        mw[0, xs] = 0
        cfg.out_d.zero_()
        m.bc.run_bszN(z, msel, mw)
        torch.cuda.synchronize(self.dev)
        I, Ho = b["I"], b["Ho"]
        gg, gu = b["gg"].view(k, I), b["gu"].view(k, I)
        act, d = b["act"].view(k, I), b["d"].view(k, Ho)
        zf = z[0].float().cpu()
        for j in xs:
            gg[j] = e.x16_gemv(li, pick[j], 0, zf, 1).half()
            gu[j] = e.x16_gemv(li, pick[j], 1, zf, 1).half()
        c = self._x16_mod(m)
        seq = 1
        flags = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        flags[0] = seq
        D = lambda *v: torch.tensor(list(v), dtype=torch.int64, device=self.dev)
        req, last, started, dflag = D(seq * 64 + k), D(0), D(0), D(seq)
        err = torch.zeros(1, dtype=torch.int32, device=self.dev)
        xe.x16_epa(gg.data_ptr(), gu.data_ptr(), I, sel[0], msel[0], rw[0], m.multi_gate.ptrs_svh, m.multi_up.ptrs_svh,
                   m.multi_down.ptrs_suh, c["act"], c["lim"], c["gated"], act.data_ptr(), k, req.data_ptr(),
                   last.data_ptr(), started.data_ptr(), 1, flags.data_ptr(), 2_000_000_000, flags.data_ptr() + 8, err)
        torch.cuda.synchronize(self.dev)
        for j in xs:
            d[j] = e.x16_gemv(li, pick[j], 2, act[j].float(), 0)
        flags[2] = seq
        out = torch.zeros(1, self.H, dtype=torch.float, device=self.dev)
        xe.x16_red(cfg.out_d.reshape(-1, Ho), d.data_ptr(), msel[0], sel[0], rw[0], k, self.H, Ho, self.H,
                   m.multi_down.ptrs_svh, z.reshape(-1), c["shw"], c["sho"], out, dflag, flags.data_ptr() + 16,
                   2_000_000_000, err)
        torch.cuda.synchronize(self.dev)
        same = float((out.view(torch.int32) == ref.view(torch.int32)).float().mean())
        ok = same == 1.0 and int(err.item()) == 0
        return ok, f"layer {li}: {same * 100:.3f} % of outputs equal, err {int(err.item())}"

    @torch.inference_mode()
    def _selftest(self, lim, n_layers=3, n_exp=4):
        """CPU kernel on RAM slots vs an fp64 reference from exllamav3's own dequantised weights (the expert's bytes are
        copied into the bounce slot so LinearEXL3.get_weight_tensor decodes them; same swiglu clamp)."""
        p = self.pool
        st = ext().state()
        ks = st[1].numpy()
        g = torch.Generator(device="cpu").manual_seed(0)
        out = {}
        for li in list(range(0, self.L, max(1, self.L // n_layers)))[:n_layers]:
            m = p.mods[li]
            res = [x for x in range(self.E) if ks[li * self.E + x] == 2][:n_exp]
            if len(res) < n_exp:
                continue
            for mt in (1, 3):
                x = (torch.randn(mt, self.H, generator=g) * 0.35).half()
                ref = torch.zeros(mt, self.H, dtype=torch.float64)
                xd = x.double()
                for ex in res:
                    sl = int(st[0][li * self.E + ex])
                    rv = self.ram.view(sl)
                    self.bounce_t[: rv.numel()].copy_(rv.to(self.dev))  # selftest (CPU lane)
                    torch.cuda.synchronize(self.dev)
                    Wg = m.gates[ex].inner.get_weight_tensor().cpu().double(); Wu = m.ups[ex].inner.get_weight_tensor().cpu().double()
                    a1, a2 = torch.nn.functional.silu(xd @ Wg), xd @ Wu
                    if lim:
                        a1, a2 = a1.clamp(max=lim), a2.clamp(-lim, lim)
                    a = (a1 * a2).half().double()
                    Wd = m.downs[ex].inner.get_weight_tensor().cpu().double()
                    ref += 0.25 * (a @ Wd)
                sel = torch.tensor([res] * mt, dtype=torch.int32)
                w = torch.full((mt, len(res)), 0.25)
                for mode in (1, 0, 2):
                    cpu = ext().cpu_forward(li, x.float(), sel, w, mode).double()
                    k = {1: "EXACT", 0: "AFFINE", 2: "I16"}[mode]
                    out[k] = max(out.get(k, 0.0), round(float((cpu - ref).norm() / ref.norm()), 6))
                    k3 = float(m.multi_gate.K) == 3 and float(m.multi_up.K) == 3 and float(m.multi_down.K) == 3
                    if mode == 2 and _vbmi_on() and k3:   # the VBMI I16 kernel must match the AVX2 one bit for bit (K 3 only: no AVX2 decoder for K 2 / 2.5)
                        ext().cpu_knob("vbmi", 0)
                        avx2 = ext().cpu_forward(li, x.float(), sel, w, mode).double()
                        ext().cpu_knob("vbmi", 1)
                        out["vbmi_diff"] = max(out.get("vbmi_diff", 0.0), float((cpu - avx2).abs().max()))
        self.bounce_t.zero_()
        torch.cuda.synchronize(self.dev)
        return out

    def set_cpu(self, on, nvcpu=None, mode=None, clamp=None):
        """Runtime switch (decode-KL harness): on=False -> every pick on the GPU path (exact)."""
        nvc = self.pol[11] if nvcpu is None else float(nvcpu)
        md = self.cpu_mode if mode is None else mode
        lim = self.cpu_lim if clamp is None else (self.act_limit if clamp else 0.0)
        ext().cpu_set(int(bool(on) and self.cpu_on), int(nvc), md, lim)

    # ---- elastic write-back (exclusive tier) ----------------------------------------------------------------------
    def elastic_writeback(self, chunks):
        p = self.pool
        if not self.elastic_wb or not chunks:
            return 0
        torch.cuda.synchronize(self.dev)
        ext().drain_all()
        own = p.owner.cpu().numpy()
        cb = p.cbase.cpu().tolist()
        keys, src = [], []
        for c in chunks:
            for s in range(c * p.spc, (c + 1) * p.spc):
                if own[s] >= 0:
                    keys.append(int(own[s])); src.append(cb[c] + (s % p.spc) * p.rec)
        t = time.perf_counter()
        n = ext().wb_bulk(torch.tensor(keys, dtype=torch.int64), torch.tensor(src, dtype=torch.int64)) if keys else 0
        self.st["elastic_wb"] += n; self.st["elastic_wb_ms"] += (time.perf_counter() - t) * 1e3
        return n

    # ---- staged prefill (called from Pool._stage_hook_nv2 on the main thread) --------------------------------------
    def stage_submit(self, li, b):
        p = self.pool
        if li >= self.L:
            return None
        miss = np.nonzero(p.slot_np[li] < 0)[0]
        assert len(miss) <= p.stage_n, f"staging overflow {len(miss)} > {p.stage_n}"
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.dev))
        fut = self.bg.submit(ext().stage_layer, li, p.stage[b].data_ptr(), p.cstream.cuda_stream, ev.cuda_event,
                             p.ev_ready[b].cuda_event)
        return (li, b, miss, fut, ev)

    def stage_wait(self, h):
        t = time.perf_counter()
        h[3].result()
        self.st["stage_wait_ms"] += (time.perf_counter() - t) * 1e3

    # ---- checks / stats -----------------------------------------------------------------------------------------
    def verify(self, n=64, seed=0):
        """I1: sampled VRAM slots and RAM slots byte-compared against fresh O_DIRECT store reads; pointer tables, homes,
        landed flags consistent with the host's residency."""
        import random
        p = self.pool
        torch.cuda.synchronize(self.dev)
        ext().drain_all()
        rnd = random.Random(seed)
        owner = p.owner.cpu().numpy()
        vs = [s for s in range(p.S) if owner[s] >= 0 and p.chunks[s // p.spc] is not None]
        so_k, ks, pc, ow = [t.numpy() for t in ext().state()]
        rk = [k for k in range(self.K) if ks[k] == 2]
        if getattr(self, "_vtmp", None) is None:
            self._vtmp = NT.HostArena(1, self.slot, 1, self.dev.index or 0, "verify")
        tmp = self._vtmp
        bad_v = bad_r = 0
        for s in rnd.sample(vs, min(n, len(vs))):
            ext().read_sync(int(owner[s]), int(tmp.addr[0]))
            kl = int(self.klen[int(owner[s])])
            bad_v += int(not torch.equal(p.chunks[s // p.spc][s % p.spc][:kl].cpu(), tmp.view(0, kl)))
        for k in rnd.sample(rk, min(n, len(rk))):
            ext().read_sync(k, int(tmp.addr[0]))
            kl = int(self.klen[k])
            bad_r += int(not torch.equal(self.ram.view(int(so_k[k]), kl), tmp.view(0, kl)))
        cb = p.cbase.cpu().tolist()
        slotof = p.slotof.flatten().cpu().numpy()
        home = self.home.view(-1, 3).numpy()
        res = self.res.numpy()
        bad_t = stale = 0       # rows of non-VRAM experts are don't-care (set at use: nv_copy fix / staging)
        for li in range(self.L):
            pg = p.keep[li][0].cpu().numpy()
            for x in range(self.E):
                k = li * self.E + x
                s = int(slotof[k])
                if s >= 0:
                    bad_t += int(pg[x] != cb[s // p.spc] + (s % p.spc) * p.rec)
                else:
                    stale += int(pg[x] != int(home[k, 0]))
        addr = self.ram.addr
        hres = sum(1 for k in rk if home[k, 0] != addr[so_k[k]] or res[k] != 1)
        hnon = sum(1 for k in range(self.K) if ks[k] == 0 and (home[k, 0] != self.ph or res[k] != 0))
        both = int(sum(1 for k in rk if slotof[k] >= 0))
        return {"vram_checked": min(n, len(vs)), "vram_bad": bad_v, "ram_checked": min(n, len(rk)), "ram_bad": bad_r,
                "table_bad": int(bad_t), "rows_stale_nonvram": int(stale), "home_bad_resident": int(hres), "home_bad_nonresident": int(hnon),
                "ram_resident": len(rk), "vram_and_ram": both, "pinned": int((pc > 0).sum()),
                "dev_bad_rows": int(self.dstats[8].item())}

    def latency(self):
        """Per-layer-call timeline from the debug rings. Device (globaltimer): publish, step start / reply seen / step end,
        copy start / end, admitted jobs; host (CLOCK_REALTIME): notice, reply. Host-device offset removed with the
        minimum of (host notice - device publish), so the host-side gaps are relative to the fastest observed."""
        d = ext().peek(self.dbg.data_ptr(), self.dbg.numel())
        d = np.asarray(d, np.int64).reshape(-1, self.DBGW)
        hs, hn, hr = [t.numpy() for t in ext().dbg_host()[:3]]
        ok = (d[:, 0] > 0) & (hs == d[:, 0]) & (d[:, 6] > 0) & (d[:, 4] > 0)
        if ok.sum() < 8:
            return {"n": int(ok.sum())}
        o = np.argsort(d[ok, 0])
        d, hn, hr = d[ok][o], hn[ok][o], hr[ok][o]
        off = np.min(hn - d[:, 1])
        q = lambda x: {"p50": round(float(np.percentile(x, 50)) / 1e3, 1), "p90": round(float(np.percentile(x, 90)) / 1e3, 1),
                       "p99": round(float(np.percentile(x, 99)) / 1e3, 1), "mean": round(float(x.mean()) / 1e3, 1)}
        consec = np.diff(d[:, 0]) == 1
        gap = (d[1:, 1] - d[:-1, 6])[consec]                       # copy end -> next publish (MoE + attention + glue)
        nj = d[:, 7].astype(np.float64)
        cms = (d[:, 6] - d[:, 5]).astype(np.float64)
        rate = (nj[nj > 0] * self.slot) / (cms[nj > 0] + 1)      # bytes per ns = GB/s
        return {"n": int(len(d)), "us_pub_to_notice_rel": q(hn - d[:, 1] - off), "us_host_serve": q(hr - hn),
                "us_step_wait": q(d[:, 3] - d[:, 2]), "us_pub_to_seen": q(d[:, 3] - d[:, 1]),
                "us_step_after_reply": q(d[:, 4] - d[:, 3]), "us_copy": q(d[:, 6] - d[:, 5]),
                "copy_gbps_p50": round(float(np.median(rate)), 2) if len(rate) else None,
                "jobs_per_call": round(float(nj.mean()), 2), "us_gap_copy_to_next_pub": q(gap) if len(gap) else None,
                "us_layer_total": q(np.diff(d[:, 1])[consec]) if consec.any() else None}

    def timeline(self):
        """Mean per-layer breakdown of consecutive decode layer calls (us). Device: pub -> reply seen (A), -> copy start
        (B), copy kernel (C; C_fix = block 0 done fixing rows incl. NVMe landing waits), copy end -> combine start (D:
        MoE kernel), combine wait for the CPU job (E), combine end -> next pub (F: glue, next attention, router).
        Host: CPU job span, CPU end relative to combine start, demand-read landing relative to copy start."""
        d = ext().peek(self.dbg.data_ptr(), self.dbg.numel())
        d = np.asarray(d, np.int64).reshape(-1, self.DBGW)
        hh = [t.numpy() for t in ext().dbg_host()]
        hs, hn, hr, cs, ce, nr, npf, inf, lm = hh[:9]
        la, nc = (hh[9], hh[10]) if len(hh) > 10 else (np.zeros_like(lm), np.zeros_like(lm))
        ok = (d[:, 0] > 0) & (hs == d[:, 0]) & (d[:, 6] > 0) & (d[:, 4] > 0) & (d[:, 9] >= d[:, 8]) & (d[:, 8] >= d[:, 6])
        if ok.sum() < 16:
            return {"n": int(ok.sum())}
        o = np.argsort(d[ok, 0])
        d, hn, hr, cs, ce, nr, npf, inf, lm, la, nc = [x[ok][o] for x in (d, hn, hr, cs, ce, nr, npf, inf, lm, la, nc)]
        off = np.min(hn - d[:, 1])
        consec = np.diff(d[:, 0]) == 1
        i = np.nonzero(consec)[0]
        nxt = d[i + 1, 1]
        D = d[i]
        tot = (nxt - D[:, 1]).astype(np.float64)
        good = (tot > 0) & (tot < 50e6)
        i, D, nxt, tot = i[good], D[good], nxt[good], tot[good]
        m = lambda x: round(float(np.mean(x)) / 1e3, 1)
        parts = {"A_pub_to_reply": D[:, 3] - D[:, 1], "B_step": D[:, 5] - D[:, 3], "C_copy": D[:, 6] - D[:, 5],
                 "D_moe": D[:, 8] - D[:, 6], "E_cpu_wait": D[:, 9] - D[:, 8], "F_rest": nxt - D[:, 9]}
        out = {"n": int(len(i)), "us_layer_mean": m(tot), "parts_us_mean": {k: m(v) for k, v in parts.items()},
               "C_fix_us_mean": m(np.where(D[:, 10] > 0, D[:, 10] - D[:, 5], 0)), "jobs_per_call": round(float(D[:, 7].mean()), 2)}
        has = (cs[i] >= hn[i]) & (ce[i] >= cs[i]) & (cs[i] - hn[i] < 50e6)
        cpu_span = (ce[i] - cs[i])[has]
        out["cpu_job_us_mean"] = m(cpu_span) if len(cpu_span) else None
        late = ((ce[i] - off) - D[:, 8])[has]
        out["cpu_end_minus_combine_start_us_mean"] = m(late) if len(late) else None
        out["cpu_start_minus_pub_us_mean"] = m(((cs[i] - off) - D[:, 1])[has]) if has.any() else None
        dr = nr[i] > 0
        out["frac_layers_demand_read"] = round(float(dr.mean()), 3)
        out["frac_layers_inflight_hit"] = round(float((inf[i] > 0).mean()), 3)
        if dr.any():
            lmv = lm[i][dr]
            okl = lmv > 0
            out["demand_land_minus_copy_start_us_mean"] = m(((lmv - off) - D[dr][:, 5])[okl]) if okl.any() else None
            out["layer_us_mean_with_demand"] = m(tot[dr]); out["layer_us_mean_without"] = m(tot[~dr]) if (~dr).any() else None
            out["parts_with_demand"] = {k: m(v[dr]) for k, v in parts.items()}
            out["parts_without"] = {k: m(v[~dr]) for k, v in parts.items()} if (~dr).any() else None
        # any NVMe pick (demand read or in-flight prefetch hit): CPU end / combine go after the last landing
        anyn = (nr[i] > 0) | (inf[i] > 0)
        okla = anyn & (la[i] > 0) & has
        out["frac_layers_any_nvme"] = round(float(anyn.mean()), 3)
        if okla.any():
            out["tail_cpu_end_minus_last_land_us_mean"] = m((ce[i] - la[i])[okla])
            out["combine_go_minus_last_land_us_mean"] = m((D[:, 9] - (la[i] - off))[okla])
            out["last_land_minus_copy_start_us_mean"] = m(((la[i] - off) - D[:, 5])[okla])
            out["layer_us_mean_any_nvme"] = m(tot[okla])
        ncr, ncn = nc[i] % 1000, nc[i] // 1000
        pure = has & ~anyn & (ncr > 0) & (ncn == 0)
        out["cpu_ram_per_call"] = round(float(ncr.mean()), 2)
        out["cpu_nvc_per_call"] = round(float(ncn.mean()), 2)
        if pure.any():
            out["frac_layers_ram_only"] = round(float(pure.mean()), 3)
            out["cpu_experts_ram_only_mean"] = round(float(ncr[pure].mean()), 2)
            out["cpu_us_per_expert_ram_only"] = m(((ce[i] - cs[i]) / np.maximum(ncr, 1))[pure])
            out["layer_us_mean_ram_only"] = m(tot[pure])
            out["parts_ram_only"] = {k: m(v[pure]) for k, v in parts.items()}
        return out

    def summary(self):
        c = self.counters()
        d = self.dstats.tolist()
        dn = ["hit", "miss", "admit", "novict", "evict", "wb", "wb_noslot", "fix", "bad_rows", "reply_wait_ns", "reply_waits",
              "nv_wait_ns", "nv_waits", "timeouts", "noslot_ring_x1e6", "noslot_jobs"]
        dev = dict(zip(dn, d))
        steps = max(1, c["dec_reqs"] // self.L)
        toks = max(1, c["dec_tok"] // self.L)
        r = lambda x: round(x, 3)
        out = {"ram_slots": self.nram, "ram_gib": r(self.nram * self.slot / GiB), "pf_ring": self.pf.n, "exclusive": self.wb_on,
               "slru_kept_pct": min(99, max(0, int(env("NV_SLRU", "0") or 0))),
               "cpu_tier": self.cpu_on, "pol": self.pol, "host": c, "dev": dev, "py": dict(self.st),
               "decode_per_step": {"steps_est": steps, "tokens_est": toks,
                                   "picks": r(c["picks"] / steps) if c["small_reqs"] == 0 else None,
                                   "vram_hits": r(c["vram"] / steps), "ram_hits": r(c["ram"] / steps),
                                   "nvme": r(c["nvme"] / steps), "cpu_experts": r(c["cpu_experts"] / steps),
                                   "zc_admits": r(c["lane_zc"] / steps), "nvme_gpu": r(c["lane_nvg"] / steps),
                                   "nvme_cpu": r(c["lane_nvc"] / steps),
                                   "reply_wait_ms": r(d[9] / 1e6 / steps), "nvme_wait_ms": r(d[11] / 1e6 / steps),
                                   "cpu_busy_ms": r(c["cpu_busy_ns"] / 1e6 / steps),
                                   "b70_experts": r(c["lane_b70"] / steps), "b70_wait_ms": r(c["b70_wait_ns"] / 1e6 / steps)},
               "b70": {"on": self.b70, "keys": len(self.b70_keys), "jobs": c["b70_jobs"], "rows": c["b70_rows"],
                       "rtt_us_mean": r(c["b70_rtt_ns"] / 1e3 / max(1, c["b70_jobs"])), "rtt_us_max": r(c["b70_rtt_max_ns"] / 1e3),
                       "wait_us_mean": r(c["b70_wait_ns"] / 1e3 / max(1, c["b70_jobs"])), "err": c["b70_err"]},
               "memcg": NT.cgroup_mem()}
        ps = out["decode_per_step"]
        out["decode_per_token"] = {k: (r(v * steps / toks) if isinstance(v, float) else v) for k, v in ps.items()
                                   if k not in ("steps_est", "tokens_est", "picks")}
        if getattr(self, "selftest", None) is not None:
            out["cpu_selftest"] = self.selftest
        try:
            out["latency"] = self.latency()
        except Exception as ex:
            out["latency"] = str(ex)
        try:
            out["timeline"] = self.timeline()
        except Exception as ex:
            out["timeline"] = repr(ex)
        return out


def attach_pool(pool):
    global NV
    nd = NT._drop_checkpoint_cache()
    NT._log(f"model loaded + VRAM pool built; DONTNEED on {nd} checkpoint files")
    NV = Nv2(pool)
    return NV


def finish(model):
    nv = NV
    for li, m in enumerate(nv.pool.mods):
        assert m.bc is not None and m.support_quant_paths, m.key
        m.bc = NT.BCProxy(m.bc, li, nv.pool)
    nd = NT._drop_checkpoint_cache()
    NT._log(f"nv2 ready: {NT._LOAD['marked_linears']} expert linears on the bounce slot, "
            f"{NT._LOAD['skipped_bytes'] / 1e9:.1f} GB of checkpoint expert bytes never read, DONTNEED on {nd} files")
    if env("NV_VERIFY_START", "1") == "1":
        print(f" -- nv2 startup verify: {nv.verify(32)}", flush=True)


def summary():
    return NV.summary() if NV is not None else None
