"""Expert tiers for DeepSeek-V4.1-Flash on stock exllamav3 1.6.6 (bit-exact by construction).

Generalised from the GLM-5.3 engine (expert_cache.py + nv_tier.py, NV=1 path) for checkpoints whose routed experts
come in several record sizes (store v2 size classes: 2.5 bpw outer layers, 2.0 bpw middle layers at 2.25 bpw).

Every routed expert has one of three homes, resolved per MoE layer before exllamav3's expert kernels run:
  VRAM slot  Pool of its size class: CLOCK cache over one arena of uniform slots (ec_step / ec_copy, device side)
  RAM slot   NvTier of its size class: pinned host arena (anon mmap + cudaHostRegister), host LRU
  NVMe       expert store v2 record (base[layer] + e * rec), read with O_DIRECT into a RAM slot on demand
exllamav3 reads each expert through its live pointer tables (multi_gate / multi_up / multi_down .ptrs_trellis and the
fused path's gu_trellis_ptr), so the arithmetic and the bytes are identical to an all-VRAM run; only addresses differ.
The checkpoint's routed-expert trellis bytes are never read: the loader hands every expert Linear a view of a VRAM
"bounce" slot, and the per-expert paths (single-expert DQ / graph) get the expert's live bytes copied there first.

Env (DSV41_*): NV_STORE (store .bin), EC_RESERVE_GB (VRAM left to torch, default 3.0), EC_MAX_SLOTS (cap, total),
EC_ADMIT_MAX (picks per layer call that admit into VRAM, default 64), NV_RAM_GB (auto = cgroup max - current - margin),
NV_MARGIN_GB (3), NV_THREADS (16), NV_PIECE_KB (1024), NV_CHECK (1), EC_WARM / NV_WARM (JSON {layer key: [score x E]}).
"""
import collections
import ctypes
import json
import mmap
import os
import time

import numpy as np
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unistd.h>
#include <errno.h>
namespace py = pybind11;

// ================= VRAM expert cache (CLOCK), one pool = one size class =================
__global__ void ec_step_k(const int64_t* __restrict__ sel, int n_picks, int first, int li, int E, int S, int admit,
    int* slotof_all, const int64_t* home_all, const int64_t* tabs, int* owner, int* refb, int* pin, int* ctl,
    int* jobs, int maxj, unsigned long long* stats, const long long* __restrict__ cbase, int spc, long long rec,
    long long off_u, long long off_d)
{
    __shared__ unsigned mask[32];
    __shared__ int misses[1024];
    __shared__ int nmiss, nhit, epoch;
    int t = threadIdx.x;
    if (t < 32) mask[t] = 0;
    if (t == 0) { nmiss = 0; nhit = 0; epoch = ctl[1] + 1; ctl[1] = epoch; ctl[2] = 0; }
    __syncthreads();
    for (int i = t; i < n_picks; i += blockDim.x) {
        long long e = sel[i] - first;
        if (e >= 0 && e < E) atomicOr(&mask[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    int* slotof = slotof_all + (long long) li * E;
    for (int e = t; e < E; e += blockDim.x) {
        if (mask[e >> 5] & (1u << (e & 31))) {
            int s = slotof[e];
            if (s >= 0) { refb[s] = 1; pin[s] = epoch; atomicAdd(&nhit, 1); }
            else { int k = atomicAdd(&nmiss, 1); misses[k] = e; }
        }
    }
    __syncthreads();
    if (t != 0) return;
    stats[0] += nhit; stats[1] += nmiss;
    if (!admit || S == 0) return;
    int hand = ctl[0], nj = 0;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    for (int k = 0; k < nmiss && nj < maxj; k++) {
        int e = misses[k], found = -1;
        for (int step = 0; step < 2 * S; step++) {
            int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
            if (pin[s] == epoch || pin[s] < 0) continue;
            if (refb[s]) { refb[s] = 0; continue; }
            found = s; break;
        }
        if (found < 0) { stats[3]++; break; }
        int s = found, old = owner[s];
        if (old >= 0) {
            int ol = old / E, oe = old % E;
            slotof_all[(long long) ol * E + oe] = -1;
            const int64_t* h = home_all + ((long long) ol * E + oe) * 3;
            int64_t* og = (int64_t*) tabs[ol * 4 + 0]; int64_t* ou = (int64_t*) tabs[ol * 4 + 1];
            int64_t* od = (int64_t*) tabs[ol * 4 + 2]; int64_t* ogu = (int64_t*) tabs[ol * 4 + 3];
            og[oe] = h[0]; ou[oe] = h[1]; od[oe] = h[2];
            if (ogu) { ogu[2 * oe] = h[0]; ogu[2 * oe + 1] = h[1]; }
            stats[4]++;
        }
        owner[s] = li * E + e; slotof[e] = s; refb[s] = 1; pin[s] = epoch;
        long long a = cbase[s / spc] + (long long) (s % spc) * rec;
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
        jobs[2 * nj] = e; jobs[2 * nj + 1] = s; nj++;
        stats[2]++;
    }
    ctl[0] = hand; ctl[2] = nj;
}

__global__ void ec_copy_k(const int* __restrict__ ctl, const int* __restrict__ jobs, int li, int E,
    const int64_t* __restrict__ home_all, const long long* __restrict__ cbase, int spc, long long rec, long long vg,
    long long vu, long long vd, long long off_u, long long off_d)
{
    long long nj = ctl[2];
    long long vpj = vg + vu + vd;
    long long total = nj * vpj;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long base = (long long) blockIdx.x * blockDim.x + threadIdx.x; base < total; base += 4 * stride) {
        int4 v[4]; int4* dst[4]; int n = 0;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            long long idx = base + u * stride;
            if (idx >= total) break;
            long long j = idx / vpj, r = idx - j * vpj;
            int e = jobs[2 * j], s = jobs[2 * j + 1];
            const int64_t* h = home_all + ((long long) li * E + e) * 3;
            const int4* src; long long doff;
            if (r < vg) { src = (const int4*) h[0] + r; doff = r * 16; }
            else if (r < vg + vu) { src = (const int4*) h[1] + (r - vg); doff = off_u + (r - vg) * 16; }
            else { src = (const int4*) h[2] + (r - vg - vu); doff = off_d + (r - vg - vu) * 16; }
            v[u] = *src;
            dst[u] = (int4*) (cbase[s / spc] + (long long) (s % spc) * rec + doff);
            n++;
        }
        for (int u = 0; u < n; u++) *dst[u] = v[u];
    }
}

void ec_step(at::Tensor sel, int64_t first, int64_t li, int64_t E, int64_t S, int64_t admit, at::Tensor slotof, at::Tensor home,
             at::Tensor tabs, at::Tensor owner, at::Tensor refb, at::Tensor pin, at::Tensor ctl, at::Tensor jobs,
             at::Tensor stats, int64_t arena, int64_t spc, int64_t rec, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous(), "sel int64 contiguous");
    TORCH_CHECK(E <= 1024, "E <= 1024");
    ec_step_k<<<1, 512, 0, st>>>(sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E, (int) S, (int) admit,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), owner.data_ptr<int>(),
        refb.data_ptr<int>(), pin.data_ptr<int>(), ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) (jobs.numel() / 2),
        (unsigned long long*) stats.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, off_u, off_d);
}

void ec_copy(at::Tensor ctl, at::Tensor jobs, int64_t li, int64_t E, at::Tensor home, int64_t arena, int64_t spc, int64_t rec,
             int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t grid)
{
    c10::cuda::CUDAGuard g(ctl.device());
    auto st = at::cuda::getCurrentCUDAStream(ctl.device().index());
    ec_copy_k<<<(int) grid, 256, 0, st>>>(ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) li, (int) E,
        home.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, sg / 16, su / 16, sd / 16, off_u, off_d);
}

// slots [s0, s1): owners back to their home rows; disable = 1 takes the slots out of the CLOCK (elastic prefill)
__global__ void ec_evict_k(int s0, int s1, int disable, int E, int* owner, int* slotof_all, const int64_t* home_all,
    const int64_t* tabs, int* refb, int* pin)
{
    int s = s0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= s1) return;
    int o = owner[s];
    if (o >= 0) {
        int l = o / E, e = o % E;
        slotof_all[(long long) l * E + e] = -1;
        const int64_t* h = home_all + ((long long) l * E + e) * 3;
        int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
        int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
        pg[e] = h[0]; pu[e] = h[1]; pd[e] = h[2];
        if (gu) { gu[2 * e] = h[0]; gu[2 * e + 1] = h[1]; }
        owner[s] = -1;
    }
    refb[s] = 0;
    pin[s] = disable ? -1 : 0;
}

void ec_evict(int64_t s0, int64_t s1, int64_t disable, int64_t E, at::Tensor owner, at::Tensor slotof, at::Tensor home,
              at::Tensor tabs, at::Tensor refb, at::Tensor pin)
{
    c10::cuda::CUDAGuard g(owner.device());
    auto st = at::cuda::getCurrentCUDAStream(owner.device().index());
    int n = (int) (s1 - s0);
    if (n <= 0) return;
    ec_evict_k<<<(n + 255) / 256, 256, 0, st>>>((int) s0, (int) s1, (int) disable, (int) E, owner.data_ptr<int>(),
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), refb.data_ptr<int>(), pin.data_ptr<int>());
}

// ================= homes / pointer rows of non-VRAM experts =================
__global__ void nv_apply_k(const int64_t* __restrict__ upd, int n, int64_t* home, const int* __restrict__ slotof,
    const int64_t* __restrict__ tabs, int E, long long off_u, long long off_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long key = upd[2 * i]; long long a = upd[2 * i + 1];
    int l = (int) (key / E), e = (int) (key % E);
    home[key * 3 + 0] = a; home[key * 3 + 1] = a + off_u; home[key * 3 + 2] = a + off_d;
    if (slotof[key] < 0) {
        int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
        int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
    }
}

void nv_apply(at::Tensor upd, at::Tensor home, at::Tensor slotof, at::Tensor tabs, int64_t E, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(home.device());
    auto st = at::cuda::getCurrentCUDAStream(home.device().index());
    int n = (int) upd.size(0);
    if (n == 0) return;
    nv_apply_k<<<(n + 255) / 256, 256, 0, st>>>(upd.data_ptr<int64_t>(), n, home.data_ptr<int64_t>(), slotof.data_ptr<int>(),
        tabs.data_ptr<int64_t>(), (int) E, off_u, off_d);
}

// out[0] += picks not VRAM-resident whose home is the placeholder; out[1] += picks whose row != VRAM slot / home
__global__ void nv_check_k(const int64_t* __restrict__ sel, int n, int first, int li, int E, const int* __restrict__ slotof,
    const int64_t* __restrict__ home, const int64_t* __restrict__ tabs, long long ph, unsigned long long* out)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        long long e = sel[i] - first;
        if (e < 0 || e >= E) continue;
        long long key = (long long) li * E + e;
        if (slotof[key] >= 0) continue;
        const int64_t* pg = (const int64_t*) tabs[li * 4 + 0];
        if (home[key * 3] == ph) atomicAdd(&out[0], 1ull);
        if (pg[e] != home[key * 3]) atomicAdd(&out[1], 1ull);
    }
}

void nv_check(at::Tensor sel, int64_t first, int64_t li, int64_t E, at::Tensor slotof, at::Tensor home, at::Tensor tabs,
              int64_t ph, at::Tensor out)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    nv_check_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), ph,
        (unsigned long long*) out.data_ptr<int64_t>());
}

// expert e of layer li, from its live pointer rows (VRAM slot / RAM), into the bounce slot
__global__ void nv_bounce_k(const int64_t* __restrict__ tabs, int li, int e, long long dst, long long ng, long long nu,
    long long nd, long long off_u, long long off_d)
{
    const int4* sg = (const int4*) ((const int64_t*) tabs[li * 4 + 0])[e];
    const int4* su = (const int4*) ((const int64_t*) tabs[li * 4 + 1])[e];
    const int4* sd = (const int4*) ((const int64_t*) tabs[li * 4 + 2])[e];
    long long total = ng + nu + nd;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        if (i < ng) ((int4*) dst)[i] = sg[i];
        else if (i < ng + nu) ((int4*) (dst + off_u))[i - ng] = su[i - ng];
        else ((int4*) (dst + off_d))[i - ng - nu] = sd[i - ng - nu];
    }
}

void nv_bounce(at::Tensor tabs, int64_t li, int64_t e, int64_t dst, int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(tabs.device());
    auto st = at::cuda::getCurrentCUDAStream(tabs.device().index());
    nv_bounce_k<<<164, 256, 0, st>>>(tabs.data_ptr<int64_t>(), (int) li, (int) e, dst, sg / 16, su / 16, sd / 16, off_u, off_d);
}

// ================= O_DIRECT pread pool (persistent threads) =================
struct Piece { int fd; long long off; long long dst; long long len; };
static std::vector<std::thread> g_thr;
static std::mutex g_mu, g_call;
static std::condition_variable g_cv, g_done_cv;
static std::vector<Piece> g_q;
static size_t g_next = 0;
static long long g_pending = 0;
static int g_err = 0;

static void worker() {
    for (;;) {
        Piece p;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait(lk, [] { return g_next < g_q.size(); });
            p = g_q[g_next++];
        }
        long long done = 0; int err = 0;
        while (done < p.len) {
            ssize_t r = pread(p.fd, (void*) (p.dst + done), (size_t) (p.len - done), (off_t) (p.off + done));
            if (r < 0) { if (errno == EINTR) continue; err = errno; break; }
            if (r == 0) { err = -1; break; }
            done += r;
        }
        std::lock_guard<std::mutex> lk(g_mu);
        if (err) g_err = err;
        if (--g_pending == 0) g_done_cv.notify_all();
    }
}

void io_start(int64_t n) {
    std::lock_guard<std::mutex> c(g_call);
    if (!g_thr.empty()) return;
    for (int i = 0; i < n; i++) { g_thr.emplace_back(worker); g_thr.back().detach(); }
}

int64_t io_read(int64_t fd, at::Tensor offs, at::Tensor dsts, int64_t len, int64_t piece) {
    // read len bytes at file offset offs[i] into host address dsts[i], split into `piece` reads, piece-major
    TORCH_CHECK(offs.dtype() == at::kLong && dsts.dtype() == at::kLong && offs.numel() == dsts.numel());
    const int64_t* o = offs.data_ptr<int64_t>(); const int64_t* d = dsts.data_ptr<int64_t>();
    int64_t n = offs.numel();
    std::lock_guard<std::mutex> c(g_call);
    TORCH_CHECK(!g_thr.empty(), "io_start first");
    if (n == 0) return 0;
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_q.clear(); g_next = 0; g_err = 0;
        for (int64_t s = 0; s < len; s += piece)
            for (int64_t i = 0; i < n; i++)
                g_q.push_back({(int) fd, o[i] + s, d[i] + s, std::min(piece, len - s)});
        g_pending = (long long) g_q.size();
    }
    g_cv.notify_all();
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_done_cv.wait(lk, [] { return g_pending == 0; });
        g_q.clear(); g_next = 0;
    }
    TORCH_CHECK(g_err == 0, "io_read failed, errno ", g_err);
    return n * len;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ec_step", &ec_step);
    m.def("ec_copy", &ec_copy);
    m.def("ec_evict", &ec_evict);
    m.def("nv_apply", &nv_apply);
    m.def("nv_check", &nv_check);
    m.def("nv_bounce", &nv_bounce);
    m.def("io_start", &io_start);
    m.def("io_read", &io_read, py::call_guard<py::gil_scoped_release>());
}
"""

GiB = 1024 ** 3
_EXT = None
_BC_ARGS = {}             # id(multi_gate.ptrs_trellis) -> (gate ptrs, up ptrs, gu_trellis_ptr) from BC_BlockSparseMLP(...)
_BOUNCE = {}              # device index -> (tensor, base address)
_LOAD = {"active": None, "marked_linears": 0, "skipped_bytes": 0, "files": set()}
_T0 = time.time()
POOLS = []                # one Pool per (device, size class)
MAN = None                # store v2 manifest


def env(name, default=None):
    return os.environ.get("DSV41_" + name, default)


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = env("TIERS_BUILD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "tiers"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "tiers.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="dsv41_tiers", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


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


def manifest():
    global MAN
    if MAN is None:
        store = env("NV_STORE")
        assert store, "DSV41_NV_STORE (expert store v2 .bin) is required"
        MAN = json.load(open(store[:-4] + ".json"))
        assert MAN["format"] == "exl3-expert-store-v2", MAN.get("format")
        MAN["_store"] = store
    return MAN


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
        _BC_ARGS[id(a[16])] = (a[16], a[22], a[43])   # gate ptrs (identity key), up ptrs, gu_trellis_ptr
        return obj
    ext.BC_BlockSparseMLP = wrapped
    bsm.ext.BC_BlockSparseMLP = wrapped

    man = manifest()
    gate, up, down = man["mats"]
    by_key = {L["key"]: L for L in man["layers"]}
    slot_max = max(L["trellis_bytes"] for L in man["layers"])

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
        assert meta["dtype"] in ("torch.int16", "I16"), meta
        t, _ = _BOUNCE[dev_idx]
        _LOAD["skipped_bytes"] += sz
        return t[off: off + sz].view(torch.int16).view(meta["shape"])

    SafetensorsCollection.get_tensor = get_tensor
    _ext()   # build now (fail fast, before the long load)
    log(f"installed: store {man['_store']}, {len(by_key)} routed layers in {len(man['classes'])} size classes, loader "
        f"skips expert trellis bytes (bounce slot {slot_max} B)")


# --------------------------------------------------------------------------------------------------------------------
# VRAM pool (one size class)
# --------------------------------------------------------------------------------------------------------------------
class Pool:
    def __init__(self, device, mods, cls, budget, admit_max):
        self.device, self.mods, self.cls, self.admit_max = device, mods, cls, admit_max
        self.nv = self.nv2 = None
        self.stage_n = 0
        m0 = mods[0]
        self.E = m0.num_local_experts or m0.num_experts
        self.first = int(m0.routing_first or 0)
        nb = lambda l: l.inner.trellis.numel() * l.inner.trellis.element_size()
        self.sg, self.su, self.sd = nb(m0.multi_gate.linears[0]), nb(m0.multi_up.linears[0]), nb(m0.multi_down.linears[0])
        # slot image = store record's trellis part: gate | up | down, no padding (store v2 field offsets)
        self.off_u, self.off_d = self.sg, self.sg + self.su
        self.rec = self.sg + self.su + self.sd
        assert self.rec == cls["trellis_bytes"] and self.off_u == cls["off_u"] and self.off_d == cls["off_d"], \
            (self.sg, self.su, self.sd, cls)
        for m in mods:
            assert (m.num_local_experts or m.num_experts) == self.E and int(m.routing_first or 0) == self.first
            for mm, s in ((m.multi_gate, self.sg), (m.multi_up, self.su), (m.multi_down, self.sd)):
                for l in mm.linears:
                    assert nb(l) == s and s % 16 == 0, (m.key, nb(l), s)
        S = max(0, int(budget // self.rec))
        self.spc = int(env("EC_CHUNK_SLOTS", "32"))
        self.chunks = []
        for _ in range((S + self.spc - 1) // self.spc):
            try:
                self.chunks.append(torch.empty((self.spc, self.rec), dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:
                break
        self.S = S = len(self.chunks) * self.spc
        self.cbase = torch.tensor([t.data_ptr() for t in self.chunks] or [0], dtype=torch.int64, device=device)
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
        for li, m in enumerate(mods):
            m._ec = (self, li)

    def geom(self, li):
        """(sg, su, sd, off_u, off_d) of layer li's experts inside a slot."""
        return self.sg, self.su, self.sd, self.off_u, self.off_d

    def step(self, li, sel, admit):
        n = sel.numel()
        b = 1 << max(0, (n - 1).bit_length())
        self.size_hist[b] = self.size_hist.get(b, 0) + 1
        e = _ext()
        sel = sel.reshape(-1)
        if sel.dtype != torch.long or not sel.is_contiguous():
            sel = sel.long().contiguous()
        e.ec_step(sel, self.first, li, self.E, self.S, int(admit), self.slotof, self.home, self.tabs, self.owner, self.refb,
                  self.pin, self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, self.off_u,
                  self.off_d)
        if admit:
            e.ec_copy(self.ctl, self.jobs, li, self.E, self.home, self.cbase.data_ptr(), self.spc, self.rec, self.sg,
                      self.su, self.sd, self.off_u, self.off_d, self.grid)

    def warm(self, scores):
        """Initial fill: top S/L RAM-resident experts per layer by score (admission path, tables stay consistent)."""
        k = self.S // max(1, len(self.mods))
        if k == 0 or not scores:
            return
        with torch.cuda.device(self.device):
            for li, m in enumerate(self.mods):
                sc = scores.get(m.key)
                if sc is None:
                    continue
                top = sorted(range(self.E), key=lambda x: -sc[x])[:k]
                top = [x for x in top if self.nv.ram_slot[li * self.E + x] >= 0]
                for c in range(0, len(top), 256):
                    self.step(li, torch.tensor([x + self.first for x in top[c:c + 256]], dtype=torch.long,
                                               device=self.device), True)
            torch.cuda.synchronize(self.device)
        self.stats.zero_()

    def _arena(self, budget, mods):
        """Slot arena + CLOCK state + live pointer tables of `mods` (self.rec = slot bytes must be set)."""
        device = self.device
        S = max(0, int(budget // self.rec))
        self.spc = int(env("EC_CHUNK_SLOTS", "32"))
        self.chunks = []
        for _ in range((S + self.spc - 1) // self.spc):
            try:
                self.chunks.append(torch.empty((self.spc, self.rec), dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:
                break
        self.S = S = len(self.chunks) * self.spc
        self.cbase = torch.tensor([t.data_ptr() for t in self.chunks] or [0], dtype=torch.int64, device=device)
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
            if cap is not None:
                assert cap[0] is pg and cap[1] is pu
                gu = cap[2]
                assert gu.shape == (self.E, 2) and gu.dtype == torch.long and gu.device == pg.device
                assert torch.equal(gu[:, 0], pg) and torch.equal(gu[:, 1], pu), m.key
            else:
                print(f" !! tiers: no gu table for {m.key} (fused bsz-N path would read home)", flush=True)
            home.append(torch.stack([pg, pu, pd], dim=1))
            tabs.append([pg.data_ptr(), pu.data_ptr(), pd.data_ptr(), gu.data_ptr() if gu is not None else 0])
            self.keep.append((pg, pu, pd, gu))
        self.home = torch.stack(home).contiguous()
        self.tabs = torch.tensor(tabs, dtype=torch.int64, device=device)
        self.grid = int(env("EC_COPY_GRID", "0")) or torch.cuda.get_device_properties(device).multi_processor_count * 8
        self.size_hist = {}

    def summary(self):
        s = self.stats.tolist()
        return {"device": str(self.device), "class_layers": len(self.mods), "slots": self.S, "slot_bytes": self.rec,
                "arena_gb": round(self.S * self.rec / 1e9, 2), "hits": s[0], "misses": s[1], "admitted": s[2],
                "no_victim": s[3], "evictions": s[4], "hit_rate": round(s[0] / max(1, s[0] + s[1]), 4),
                "picks_hist": dict(sorted(self.size_hist.items()))}


class UPool(Pool):
    """nv2 mode: ONE CLOCK pool over every routed layer. Slots are uniform (the largest layer's trellis); an expert of
    layer li fills the first sg+su+sd bytes of its slot in that layer's compact layout (gate | up @ off_u | down @ off_d,
    the store v2 record image), so every per-layer kernel call passes its own geometry (geom(li))."""

    def __init__(self, device, mods, budget, admit_max, slot_bytes):
        self.device, self.mods, self.admit_max = device, mods, admit_max
        self.nv = self.nv2 = None
        self.stage_n = 0
        self.cls = None
        m0 = mods[0]
        self.E = m0.num_local_experts or m0.num_experts
        self.first = int(m0.routing_first or 0)
        nb = lambda l: l.inner.trellis.numel() * l.inner.trellis.element_size()
        self.g = []
        for m in mods:
            assert (m.num_local_experts or m.num_experts) == self.E and int(m.routing_first or 0) == self.first
            sg, su, sd = nb(m.multi_gate.linears[0]), nb(m.multi_up.linears[0]), nb(m.multi_down.linears[0])
            for mm, s in ((m.multi_gate, sg), (m.multi_up, su), (m.multi_down, sd)):
                for l in mm.linears:
                    assert nb(l) == s and s % 16 == 0, (m.key, nb(l), s)
            assert sg + su + sd <= slot_bytes
            self.g.append((sg, su, sd, sg, sg + su))
        self.rec = slot_bytes
        self.sg, self.su, self.sd, self.off_u, self.off_d = self.g[0]   # layer 0 (diagnostics only)
        self._arena(budget, mods)
        for li, m in enumerate(mods):
            m._ec = (self, li)

    def geom(self, li):
        return self.g[li]

    # ---- staged prefill (nv2): GLM-5.3 engine's Pool._stage_hook_nv2 / elastic VRAM, per-layer slot geometry -----------
    def stage_init(self, stage_min, elastic_bytes, stage_n):
        """Prefill forwards with >= stage_min picks per layer stage every non-VRAM expert of the next layer into a VRAM
        buffer (two buffers, stage_n slots each) on a copy stream. Elastic: the last VRAM-cache chunks (elastic_bytes) are
        evicted and handed to torch for the prefill (activations + the staging buffers) and come back for decode."""
        self.stage_min, self.stage_n = stage_min, stage_n
        self.elastic = elastic_bytes
        self.stage = None
        nflex = min(len(self.chunks), int(-(-elastic_bytes // (self.spc * self.rec)))) if elastic_bytes else 0
        self.flex_all = list(range(len(self.chunks) - nflex, len(self.chunks)))
        self.flex = list(self.flex_all)
        self.in_prefill = False
        self.elastic_stats = {"enter": 0, "exit": 0, "realloc_fail": 0}
        self.cstream = torch.cuda.Stream(self.device)
        self.ev_ready = [torch.cuda.Event() for _ in range(2)]
        for ev in self.ev_ready:
            ev.record(torch.cuda.current_stream(self.device))
        self.staged = {}
        self.last_li = None
        self.pending = None
        self.slot_np = None
        self._st_open = False
        self.stage_stats = {"forwards": 0, "layers": 0, "experts": 0}
        if not elastic_bytes and stage_n:
            self.stage = [torch.empty((stage_n, self.rec), dtype=torch.uint8, device=self.device) for _ in range(2)]

    def _point_staged(self, li, take, b):
        idx = torch.from_numpy(take.astype(np.int64)).to(self.device)
        sg, su, sd, ou, od = self.g[li]
        base = self.stage[b].data_ptr()
        a = base + torch.arange(len(take), dtype=torch.int64, device=self.device) * self.rec
        pg, pu, pd, gu = self.keep[li]
        pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + ou); pd.index_copy_(0, idx, a + od)
        if gu is not None:
            gu.index_copy_(0, idx, torch.stack([a, a + ou], 1))
        self.staged[li] = (b, idx)

    def _restore_nv2(self, li):
        from . import nv2 as N2
        b, idx = self.staged.pop(li)
        N2.ext().nv_restore(idx, li, self.E, self.nv2.a["home"], self.tabs)

    def _stage_hook_nv2(self, li, n_picks):
        from . import nv2 as N2
        e = N2.ext()
        prefill = n_picks >= self.stage_min
        for l in [k for k in self.staged if k != li]:
            self._restore_nv2(l)
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
        if self.last_li is None or li <= self.last_li:      # new forward pass
            if self.pending is not None:
                self.nv2.stage_wait(self.pending)
                self.pending = None
            if self.elastic and not self.in_prefill:
                self._elastic_enter()
            self.slot_np = self.slotof.cpu().numpy()
            e.stage_begin(torch.from_numpy(np.ascontiguousarray(self.slot_np)), li)
            self._st_open = True
            self.stage_stats["forwards"] += 1
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
            self._point_staged(li, take, b)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += len(take)
        self.pending = self.nv2.stage_submit(li + 1, (li + 1) % 2)

    def _ec_hits(self, li, sel):
        sg, su, sd, ou, od = self.g[li]
        _ext().ec_step(sel, self.first, li, self.E, self.S, 0, self.slotof, self.home, self.tabs, self.owner, self.refb,
                       self.pin, self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, ou, od)

    def _elastic_enter(self):
        e = _ext()
        self.flex = list(self.flex_all)
        if self.nv2 is not None:
            self.elastic_stats["wb"] = self.elastic_stats.get("wb", 0) + self.nv2.elastic_writeback(self.flex)
        for c in self.flex:
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 1, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        torch.cuda.synchronize(self.device)
        for c in self.flex:
            self.chunks[c] = None
        torch.cuda.empty_cache()
        self.stage = [torch.empty((self.stage_n, self.rec), dtype=torch.uint8, device=self.device) for _ in range(2)]
        self.in_prefill = True
        self.elastic_stats["enter"] += 1

    def _elastic_exit(self):
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

    def step(self, li, sel, admit):
        sg, su, sd, ou, od = self.g[li]
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
        """VRAM warm through nv2's admission path is done by Nv2; here: top S/L RAM-resident experts per layer."""
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
                top = [x for x in sorted(range(self.E), key=lambda x: -sc[x])[:k] if ks[li * self.E + x] == 2]
                for c in range(0, len(top), 256):
                    self.step(li, torch.tensor([x + self.first for x in top[c:c + 256]], dtype=torch.long,
                                               device=self.device), True)
            torch.cuda.synchronize(self.device)
        self.stats.zero_()


# GLM-engine compatible names used by nv2.py
_log = log
_LOADED = _LOAD


def _rcs_bytes():
    import sys
    a, v = sys.argv, 0.5
    for i, x in enumerate(a):
        if x in ("-rcs", "--recurrent_cache_size") and i + 1 < len(a):
            v = float(a[i + 1])
    return int(v * GiB)


# --------------------------------------------------------------------------------------------------------------------
# RAM tier + NVMe (one size class)
# --------------------------------------------------------------------------------------------------------------------
class NvTier:
    def __init__(self, pool, nram, fd):
        self.pool = pool
        self.dev = pool.device
        man = manifest()
        self.slot = pool.rec
        li_of = {L["key"]: L for L in man["layers"]}
        self.base = np.array([li_of[m.key]["base"] for m in pool.mods], dtype=np.int64)
        self.rec_bytes = li_of[pool.mods[0].key]["rec_bytes"]
        assert all(li_of[m.key]["rec_bytes"] == self.rec_bytes for m in pool.mods)
        assert all(li_of[m.key]["trellis_bytes"] == self.slot for m in pool.mods)
        assert pool.first == 0 and pool.E == man["experts_per_layer"]
        self.L, self.E = len(pool.mods), pool.E
        self.K = self.L * self.E
        self.fd = fd
        self.piece = int(env("NV_PIECE_KB", "1024")) * 1024
        assert self.piece % 4096 == 0
        dev_idx = self.dev.index or 0
        self.ph_t = torch.zeros(self.slot, dtype=torch.uint8, device=self.dev)   # placeholder home (never real bytes)
        self.ph = self.ph_t.data_ptr()
        self.bounce_t, self.bounce = _BOUNCE[dev_idx]
        self.ram = HostArena(nram, self.slot, 256, dev_idx, f"ram{id(pool)}") if nram else None
        self.nram = nram
        self.ram_slot = np.full(self.K, -1, dtype=np.int32)
        self.owner = np.full(max(nram, 1), -1, dtype=np.int64)
        self.free = list(range(nram - 1, -1, -1))
        self.lru = collections.OrderedDict()          # key -> slot, oldest first
        self.h_sel = torch.empty(65536, dtype=torch.int64, pin_memory=True)
        self.h_slot = torch.empty(self.E, dtype=torch.int32, pin_memory=True)
        self.check = env("NV_CHECK", "1") == "1"
        self.pf_transient = env("NV_PF_TRANSIENT", "1") == "1"
        self.chk = torch.zeros(4, dtype=torch.int64, device=self.dev)
        self.counts = {"dec": np.zeros((self.L, self.E), np.int64), "pf": np.zeros((self.L, self.E), np.int64)}
        self.st = collections.Counter()
        self.lat = collections.defaultdict(list)
        self._apply([(k, self.ph) for k in range(self.K)])
        torch.cuda.synchronize(self.dev)
        pool.nv = self

    def offset(self, k):
        li, e = divmod(int(k), self.E)
        return int(self.base[li]) + e * self.rec_bytes

    def _apply(self, upd):
        if not upd:
            return
        u = torch.tensor(upd, dtype=torch.int64).to(self.dev)
        p = self.pool
        _ext().nv_apply(u, p.home, p.slotof, p.tabs, self.E, p.off_u, p.off_d)

    def _read(self, keys, dsts, phase):
        t = time.perf_counter()
        n = _ext().io_read(self.fd, torch.tensor([self.offset(k) for k in keys], dtype=torch.int64),
                           torch.tensor([int(d) for d in dsts], dtype=torch.int64), self.slot, self.piece)
        ms = (time.perf_counter() - t) * 1e3
        self.st[phase + "_nvme_reads"] += len(keys); self.st[phase + "_nvme_bytes"] += n
        self.st[phase + "_nvme_ms"] += ms; self.st[phase + "_nvme_batches"] += 1
        if len(self.lat[phase]) < 200000:
            self.lat[phase].append(round(ms, 3))
        return ms

    def _alloc(self, n, protect):
        slots, ev = [], []
        while len(slots) < n and self.free:
            slots.append(self.free.pop())
        guard = 0
        while len(slots) < n:
            k, s = self.lru.popitem(last=False)
            if k in protect:
                self.lru[k] = s
                guard += 1
                assert guard <= len(self.lru) + 1, "RAM tier smaller than one layer's working set"
                continue
            self.ram_slot[k] = -1; self.owner[s] = -1
            ev.append(k); slots.append(s)
        return slots, ev

    def _fill(self, keys, protect, phase):
        slots, ev = self._alloc(len(keys), protect)
        upd = [(k, self.ph) for k in ev]
        self.st[phase + "_ram_evictions"] += len(ev)
        ms = 0.0
        for c in range(0, len(keys), 256):
            ms += self._read(keys[c:c + 256], [self.ram.addr[s] for s in slots[c:c + 256]], phase)
        for k, s in zip(keys, slots):
            self.ram_slot[k] = s; self.owner[s] = k; self.lru[k] = s
            upd.append((k, int(self.ram.addr[s])))
        return upd, ms

    def warm(self, scores):
        if not self.nram or not scores:
            return
        k = self.nram // self.L
        keys = []
        for li, m in enumerate(self.pool.mods):
            sc = scores.get(m.key)
            order = sorted(range(self.E), key=lambda x: -sc[x]) if sc is not None else list(range(self.E))
            keys += [li * self.E + e for e in order[:k]]
        t = time.perf_counter()
        upd, ms = self._fill(keys, set(), "warm")
        self._apply(upd)
        torch.cuda.synchronize(self.dev)
        log(f"RAM warm class {self.slot}: {len(keys)} experts ({k}/layer) in {time.perf_counter() - t:.1f} s "
            f"({len(keys) * self.slot / max(ms, 1e-3) / 1e6:.1f} GB/s)")

    def post_warm(self):
        so = self.pool.slotof.flatten().cpu().numpy()
        for k in list(self.lru.keys()):
            if so[k] >= 0:
                self.lru.move_to_end(k, last=False)
        self.st.clear(); self.lat.clear(); self.chk.zero_()

    def ensure(self, li, sel):
        """Host sync; every picked expert that is not VRAM-resident is made RAM-resident (NVMe -> RAM, stall)."""
        p = self.pool
        n = sel.numel()
        phase = "dec" if n <= p.admit_max else "pf"
        t0 = time.perf_counter()
        flat = sel.reshape(-1)
        if n <= self.h_sel.numel():
            self.h_sel[:n].copy_(flat, non_blocking=True)
            self.h_slot.copy_(p.slotof[li], non_blocking=True)
            torch.cuda.current_stream(self.dev).synchronize()
            s_np, row = self.h_sel[:n].numpy(), self.h_slot.numpy()
        else:
            s_np, row = flat.cpu().numpy(), p.slotof[li].cpu().numpy()
        t1 = time.perf_counter()
        ex = np.unique(s_np) - p.first
        ex = ex[(ex >= 0) & (ex < self.E)]
        np.add.at(self.counts[phase][li], ex, 1)
        miss = ex[row[ex] < 0]
        keys = li * self.E + miss
        rs = self.ram_slot[keys]
        hit = keys[rs >= 0]
        nv = [int(k) for k in keys[rs < 0]]
        for k in hit:
            self.lru.move_to_end(int(k))
        self.st[phase + "_calls"] += 1
        self.st[phase + "_picks_unique"] += len(ex)
        self.st[phase + "_vram_hits"] += len(ex) - len(miss)
        self.st[phase + "_ram_hits"] += len(hit)
        self.st[phase + "_nvme_misses"] += len(nv)
        if nv:
            upd, _ = self._fill(nv, set(int(k) for k in hit), phase)
            self._apply(upd)
            if phase == "pf" and self.pf_transient:     # prefill fills are evicted first (decode working set kept)
                for k in nv:
                    self.lru.move_to_end(k, last=False)
        if self.check:
            _ext().nv_check(flat if flat.dtype == torch.long else flat.long(), p.first, li, self.E, p.slotof, p.home,
                            p.tabs, self.ph, self.chk)
        t2 = time.perf_counter()
        self.st[phase + "_sync_ms"] += (t1 - t0) * 1e3
        self.st[phase + "_ensure_ms"] += (t2 - t0) * 1e3

    def verify(self, n=32, seed=0):
        """Sampled VRAM and RAM slots byte-compared against fresh O_DIRECT store reads; pointer rows / homes checked."""
        import random
        p = self.pool
        torch.cuda.synchronize(self.dev)
        rnd = random.Random(seed)
        owner = p.owner.cpu().numpy()
        vs = [s for s in range(p.S) if owner[s] >= 0]
        rk = list(self.lru.keys())
        if getattr(self, "_vtmp", None) is None:
            self._vtmp = HostArena(1, al4k(self.slot), 1, self.dev.index or 0, "verify")
        tmp = self._vtmp
        bad_v = bad_r = 0
        for s in rnd.sample(vs, min(n, len(vs))):
            self._read([int(owner[s])], [tmp.addr[0]], "verify")
            bad_v += int(not torch.equal(p.chunks[s // p.spc][s % p.spc].cpu(), tmp.view(0, self.slot)))
        for k in rnd.sample(rk, min(n, len(rk))):
            self._read([k], [tmp.addr[0]], "verify")
            bad_r += int(not torch.equal(self.ram.view(int(self.ram_slot[k])), tmp.view(0, self.slot)))
        cb = p.cbase.cpu().tolist()
        slotof = p.slotof.flatten().cpu().numpy()
        home = p.home.view(-1, 3).cpu().numpy()
        bad_t = 0
        for li in range(self.L):
            pg = p.keep[li][0].cpu().numpy()
            for e in range(self.E):
                k = li * self.E + e
                s = int(slotof[k])
                want = (cb[s // p.spc] + (s % p.spc) * p.rec) if s >= 0 else int(home[k, 0])
                bad_t += int(pg[e] != want)
        rsl = self.ram_slot
        hres = sum(1 for k in range(self.K) if rsl[k] >= 0 and home[k, 0] != self.ram.addr[rsl[k]])
        hnon = sum(1 for k in range(self.K) if rsl[k] < 0 and home[k, 0] != self.ph)
        return {"slot_bytes": self.slot, "vram_checked": min(n, len(vs)), "vram_bad": bad_v,
                "ram_checked": min(n, len(rk)), "ram_bad": bad_r, "table_bad": int(bad_t),
                "home_bad_resident": int(hres), "home_bad_nonresident": int(hnon)}

    def summary(self):
        st = dict(self.st)
        c = self.chk.tolist()
        r = lambda x: round(x, 3)
        out = {"slot_bytes": self.slot, "ram_slots": self.nram, "ram_gib": r(self.nram * self.slot / GiB),
               "ram_resident": len(self.lru), "violations_placeholder_reads": c[0], "violations_table_rows": c[1],
               "counters": {k: (r(v) if isinstance(v, float) else v) for k, v in st.items()}}
        for ph, lt in self.lat.items():
            if lt:
                s = sorted(lt)
                out.setdefault("nvme_batch_ms", {})[ph] = {"n": len(s), "p50": s[len(s) // 2],
                                                          "p99": s[int(0.99 * (len(s) - 1))], "max": s[-1]}
        return out


def al4k(x):
    return (x + 4095) // 4096 * 4096


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

    def _st(self):
        p = self._pool
        return p.nv.st if p.nv is not None else p.nv2.st

    def run_single_expert_dq(self, y, e, *a):
        self._st()["dq_bounce_calls"] += 1
        self._fill(e)
        return self._bc.run_single_expert_dq(y, e, *a)

    def run_single_expert(self, y, e, *a):
        self._st()["graph_bounce_calls"] += 1
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


def _drop_checkpoint_cache():
    nd = 0
    for f in sorted(_LOAD["files"]):
        try:
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); nd += 1
        except OSError:
            pass
    return nd


def attach(model):
    """After model load: one Pool + NvTier per (device, size class); VRAM / RAM budgets split so that every layer gets
    about the same number of slots; routing hooks make every pick VRAM- or RAM-resident before the expert kernels."""
    man = manifest()
    by_key = {L["key"]: L for L in man["layers"]}
    mods = [m for m in find_moe(model.modules) if m.key in by_key]
    assert len(mods) == len(by_key), f"{len(mods)} MoE modules on the GPU vs {len(by_key)} store layers"
    nd = _drop_checkpoint_cache()
    groups = {}
    for m in mods:
        groups.setdefault((str(m.device), by_key[m.key]["cls"]), []).append(m)
    devs = sorted({d for d, _ in groups})
    assert len(devs) == 1, "tiers: one GPU"
    dev = torch.device(devs[0])
    admit_max = int(env("EC_ADMIT_MAX", "64"))
    # VRAM budget
    torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(dev)
    reserve = float(env("EC_RESERVE_GB", "3.0")) * GiB
    vram = max(0, free - reserve)
    mx = env("EC_MAX_SLOTS")
    weight = {g: len(ms) * man["classes"][g[1]]["trellis_bytes"] for g, ms in groups.items()}
    wsum = sum(weight.values())
    # RAM budget
    margin = float(env("NV_MARGIN_GB", "3")) * GiB
    want = env("NV_RAM_GB", "auto")
    m = cgroup_mem()
    if want == "auto":
        assert m.get("max"), "DSV41_NV_RAM_GB=auto needs a memory-capped cgroup"
        ram = max(0, (m["max"] - m["current"]) * GiB - margin)
    else:
        ram = float(want) * GiB
    log(f"model loaded; DONTNEED on {nd} checkpoint files; VRAM free {free / GiB:.2f} GiB (reserve {reserve / GiB:.1f}), "
        f"RAM tier budget {ram / GiB:.2f} GiB")
    store = man["_store"]
    fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
    _ext().io_start(int(env("NV_THREADS", "16")))
    warm = env("EC_WARM")
    scores = json.load(open(warm)) if warm else None
    for g in sorted(groups):
        ms = groups[g]
        cls = man["classes"][g[1]]
        budget = vram * weight[g] / wsum
        if mx:
            budget = min(budget, int(mx) * weight[g] / wsum * cls["trellis_bytes"])
        p = Pool(dev, ms, cls, budget, admit_max)
        nram = int(ram * weight[g] / wsum // cls["trellis_bytes"])
        NvTier(p, nram, fd)
        p.nv.warm(scores)
        p.warm(scores)
        p.nv.post_warm()
        POOLS.append(p)
        log(f"class {g[1]}: {len(ms)} layers, slot {cls['trellis_bytes'] / 1e6:.2f} MB, VRAM {p.S} slots "
            f"({p.S / len(ms):.1f}/layer, {p.S * p.rec / GiB:.2f} GiB), RAM {nram} slots ({nram / len(ms):.1f}/layer, "
            f"{nram * p.rec / GiB:.2f} GiB)")
    for m in mods:
        pool, li = m._ec
        orig = m.routing_fn

        def routed(bsz, cfg, z, params, _orig=orig, _m=m):
            sel, w = _orig(bsz, cfg, z, params)
            if not params.get("autosplit_measure") and not params.get("tp_warmup"):
                pool, li = _m._ec
                pool.nv.ensure(li, sel)
                pool.step(li, sel, sel.numel() <= pool.admit_max)
            return sel, w
        m.routing_fn = routed
        assert m.bc is not None and m.support_quant_paths, m.key
        m.bc = BCProxy(m.bc, li, pool)
    nd = _drop_checkpoint_cache()
    log(f"ready: {_LOAD['marked_linears']} expert linears on the bounce slot, {_LOAD['skipped_bytes'] / 1e9:.1f} GB of "
        f"checkpoint expert bytes never read, DONTNEED on {nd} files")
    if env("NV_VERIFY_START", "1") == "1":
        print(f" -- tiers startup verify: {verify(16)}", flush=True)
    return POOLS


def attach_nv2(model):
    """nv2 mode (DSV41_NV=2): one UPool over every routed layer (uniform slots of the largest layer's trellis) + the nv2
    engine (device-side publish / reply, host planner, exclusive RAM tier, NVMe reads, optional CPU lane)."""
    from . import nv2 as N2
    man = manifest()
    by_key = {L["key"]: L for L in man["layers"]}
    mods = [m for m in find_moe(model.modules) if m.key in by_key]
    assert len(mods) == len(by_key), f"{len(mods)} MoE modules on the GPU vs {len(by_key)} store layers"
    mods.sort(key=lambda m: by_key[m.key]["li"])
    dev = torch.device(str(mods[0].device))
    admit_max = int(env("EC_ADMIT_MAX", "64"))
    torch.cuda.synchronize(dev); torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(dev)
    reserve = float(env("EC_RESERVE_GB", "3.0")) * GiB
    slot = max(L["trellis_bytes"] for L in man["layers"])
    budget = max(0, free - reserve)
    mx = env("EC_MAX_SLOTS")
    if mx:
        budget = min(budget, int(mx) * slot)
    p = UPool(dev, mods, budget, admit_max, slot)
    if env("NV_STAGE", "1") == "1":
        stage_n = int(env("EC_STAGE_SLOTS", str(p.E)))           # every expert of a layer may be non-VRAM
        elastic = float(env("EC_ELASTIC_GB", "0")) * GiB or (2 * stage_n * slot + 0.5 * GiB)
        p.stage_init(int(env("EC_STAGE_MIN", "512")), elastic, stage_n)
        log(f"staged prefill: >= {p.stage_min} picks per layer, 2 x {stage_n} slots ({2 * stage_n * slot / GiB:.2f} GiB) "
            f"from {len(p.flex_all)} elastic chunks ({len(p.flex_all) * p.spc * slot / GiB:.2f} GiB)")
    POOLS.append(p)
    N2.attach_pool(p)
    warm = env("EC_WARM")
    scores = json.load(open(warm)) if warm else None
    if scores:
        p.nv2.warm(scores)
        p.warm(scores)
    p.nv2.post_warm()
    p.nv2.cpu_setup()
    for m in mods:
        orig = m.routing_fn

        def routed(bsz, cfg, z, params, _orig=orig, _m=m):
            nv = _m._ec[0].nv2
            if nv.pfside and not params.get("autosplit_measure") and not params.get("tp_warmup"):
                nv.predict_early(_m._ec[1], z, bsz)
            sel, w = _orig(bsz, cfg, z, params)
            _m._nv2_cpu = False
            if not params.get("autosplit_measure") and not params.get("tp_warmup"):
                sel, w, _m._nv2_cpu = nv.layer(_m._ec[1], sel, w, z, bsz)
            return sel, w
        m.routing_fn = routed
        orig_comb = m.cpu_split_combine

        def comb(fhs, cpu_partial, cpu_pending, shape, _orig=orig_comb, _m=m):
            if getattr(_m, "_nv2_cpu", False):
                _m._nv2_cpu = False
                fhs = _m._ec[0].nv2.combine(fhs)
            return _orig(fhs, cpu_partial, cpu_pending, shape)
        m.cpu_split_combine = comb
    log(f"nv2 pool: {len(mods)} MoE layers, {p.S} slots x {slot / 1e6:.2f} MB = {p.S * slot / GiB:.2f} GiB "
        f"({p.S / len(mods):.1f}/layer), admit <= {admit_max} picks{', warm ' + warm if warm else ''}")
    N2.finish(model)
    return POOLS


def summary():
    if POOLS and POOLS[0].nv2 is not None:
        from . import nv2 as N2
        return [dict(vram=POOLS[0].summary(), nv2=N2.summary())]
    return [dict(vram=p.summary(), ram=p.nv.summary()) for p in POOLS]


def expert_counts(phase="dec"):
    """{layer key: [unique-pick count per expert]} of every routed layer (warm-start scores, routing statistics)."""
    out = {}
    for p in POOLS:
        if p.nv is None:
            continue
        for li, m in enumerate(p.mods):
            out[m.key] = p.nv.counts[phase][li].tolist()
    return out


def verify(n=32):
    if POOLS and POOLS[0].nv2 is not None:
        return [POOLS[0].nv2.verify(n)]
    return [p.nv.verify(n) for p in POOLS]
