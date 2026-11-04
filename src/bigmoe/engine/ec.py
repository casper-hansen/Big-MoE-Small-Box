"""CUDA kernels of the VRAM expert pool: CLOCK step (hit marking + admission), admission copy, slot eviction (elastic
prefill), and the bounce copy for exllamav3's per-expert paths."""
import os

from .env import build_dir

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py = pybind11;

// ================= VRAM expert cache (CLOCK) =================
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ec_step", &ec_step);
    m.def("ec_copy", &ec_copy);
    m.def("ec_evict", &ec_evict);
    m.def("nv_bounce", &nv_bounce);
}
"""

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = build_dir("EC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "ec"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "bigmoe_ec.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="bigmoe_ec", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT
