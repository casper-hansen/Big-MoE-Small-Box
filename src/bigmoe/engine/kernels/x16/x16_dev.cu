// X16 lane, device side (engine/x16.py): the parts of exllamav3's fused decode MoE (exl3_moe_coop_kernel.cuh, identical
// in exllamav3 1.5.1 and 1.6.x on Ampere) that follow its GEMVs, re-run for the slots whose GEMVs the CPU computed
// bit-exactly (nv2/ft_x16.h):
//   x16_epa  kernel A's chunk epilogue (output rotation * svh, activation, * down suh, rotation -> fp16 down input) on the
//            CPU's fp16 gate / up rows
//   x16_red  kernel B's token reduction (rotation * svh of every slot's down output, routing weights in slot order, the
//            shared expert through its sigmoid gate) over ALL slots: the GPU's own down outputs for its slots, the CPU's
//            for the X16 slots, so the layer output is the one the full kernel would have produced
// The helpers are exllamav3's, verbatim, and this file is built with exllamav3's nvcc flags (-O3 --use_fast_math), so the
// arithmetic compiles to the same instructions (exllamav3: MIT licence, (c) turboderp).
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {

constexpr int WK = 16;
constexpr int THREADS = WK * 32;
constexpr float R_SCALE = 0.088388347648f;      // 1/sqrt(128)
#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

// ---- exllamav3 helpers (verbatim) ----------------------------------------------------------------------------------
__device__ __forceinline__ float tanh_opt(float x)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
        float r;
        asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
        return r;
    #else
        return tanhf(x);
    #endif
}

__device__ inline void shuffle_had_f4x32(float& h0, float& h1, float& h2, float& h3, const int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        uint32_t i0 = __float_as_uint(h0);
        uint32_t i1 = __float_as_uint(h1);
        uint32_t i2 = __float_as_uint(h2);
        uint32_t i3 = __float_as_uint(h3);
        uint64_t h01 =  (uint64_t) i0 | (((uint64_t) i1) << 32);
        uint64_t h23 =  (uint64_t) i2 | (((uint64_t) i3) << 32);
        uint64_t ph01 = __shfl_xor_sync(0xffffffff, h01, i);
        uint64_t ph23 = __shfl_xor_sync(0xffffffff, h23, i);
        float ph0 = __uint_as_float((uint32_t) (ph01 & 0xffffffff));
        float ph1 = __uint_as_float((uint32_t) (ph01 >> 32));
        float ph2 = __uint_as_float((uint32_t) (ph23 & 0xffffffff));
        float ph3 = __uint_as_float((uint32_t) (ph23 >> 32));
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        i0 ^= sfm & 0x80000000;
        i1 ^= sfm & 0x80000000;
        i2 ^= sfm & 0x80000000;
        i3 ^= sfm & 0x80000000;
        h0 = __uint_as_float(i0) + ph0;
        h1 = __uint_as_float(i1) + ph1;
        h2 = __uint_as_float(i2) + ph2;
        h3 = __uint_as_float(i3) + ph3;
    }
}

__device__ __forceinline__ float act_silu(float x)
{
    float e = __expf(-x);
    return x * __fdividef(1.0f, 1.0f + e);
}

__device__ __forceinline__ float act_gelu(float x)
{
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_opt(t));
}

__device__ __forceinline__ float act_relu2(float x)
{
    x = fmaxf(0.0f, x);
    return x * x;
}

__device__ __forceinline__ float act_oai_swiglu(float g, float u, float limit)
{
    if (limit != 0.0f)
    {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    float glu = g * (1.0f / (1.0f + __expf(-1.702f * g)));
    return (u + 1.0f) * glu;
}

__device__ __forceinline__ float act_gate(int act, bool gated, float g, float u, float limit)
{
    if (act == MOE_COOP_ACT_SILU_OAI) return act_oai_swiglu(g, u, limit);
    float x;
    if (!gated) x = fmaxf(0.0f, u);
    else if (act == MOE_COOP_ACT_SILU) x = act_silu(g);
    else if (act == MOE_COOP_ACT_GELU) x = act_gelu(g);
    else x = act_relu2(g);
    if (limit != 0.0f)
    {
        u = fminf(fmaxf(u, -limit), limit);
        x = fminf(x, limit);
    }
    return x * u;
}

__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v0 = h0 * R_SCALE;
    v1 = h1 * R_SCALE;
    v2 = h2 * R_SCALE;
    v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void unpack_h4(uint2 u, float& v0, float& v1, float& v2, float& v3)
{
    half2 a = *reinterpret_cast<half2*>(&u.x);
    half2 b = *reinterpret_cast<half2*>(&u.y);
    v0 = __low2float(a);
    v1 = __high2float(a);
    v2 = __low2float(b);
    v3 = __high2float(b);
}

__device__ __forceinline__ void load_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(*((const uint2*) p), v0, v1, v2, v3);
}

__device__ __forceinline__ void load_h4_cg(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(__ldcg((const uint2*) p), v0, v1, v2, v3);
}

__device__ __forceinline__ void load_f4_cg(const float* p, float& v0, float& v1, float& v2, float& v3)
{
    float4 f = __ldcg((const float4*) p);
    v0 = f.x; v1 = f.y; v2 = f.z; v3 = f.w;
}

__device__ __forceinline__ void store_h4(half* p, float v0, float v1, float v2, float v3)
{
    half2 a = __floats2half2_rn(v0, v1);
    half2 b = __floats2half2_rn(v2, v3);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *((uint2*) p) = u;
}

__device__ __forceinline__ void scale_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3;
}

__device__ __forceinline__ void add_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 += s0; v1 += s1; v2 += s2; v3 += s3;
}

// ---- waits --------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long gtime()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// thread 0 spins until *flag >= seq (bounded); returns false on timeout (block-uniform through shared memory)
__device__ __forceinline__ bool wait_flag(const volatile long long* flag, long long seq, long long timeout_ns, int* sh_ok, int* err)
{
    if (threadIdx.x == 0)
    {
        const unsigned long long t0 = gtime();
        int ok = 1;
        while (*flag < seq)
            if ((long long) (gtime() - t0) > timeout_ns) { ok = 0; atomicExch(err, 1); break; }
        *sh_ok = ok;
        __threadfence_system();
    }
    __syncthreads();
    return *sh_ok;
}

// ---- kernel A epilogue on CPU gate / up rows ------------------------------------------------------------------------
// ONE block of THREADS (16 warps): warp w takes the (X16 slot, 128-column chunk) items w, w + 16, ... Rows of gu_g / gu_u /
// act_out are slot indices (host-mapped). Launched on a high-priority non-blocking side stream behind an event on this
// call's nv_step; on entry it publishes *started = ticket, and the main stream's x16_gate kernel (before the MoE kernels)
// waits for that, so the block is resident before the copy / MoE kernels fill the GPU. It reads this call's seq from
// nv_step's per-layer record req = dflag[1 + li] (+-(seq * 64 + picks), + with a CPU job), then waits for the CPU's
// gate / up rows (flag >= seq). Only X16 slots (masked sel < 0, original weight != 0) do work.
__global__ void x16_epa_k(const half* gu_g, const half* gu_u, int I, int topk, const int64_t* sel, const int64_t* msel,
                          const half* rw, const int64_t* g_svh, const int64_t* u_svh, const int64_t* d_suh,
                          int act, float act_limit, bool gated, half* act_out, const volatile long long* req,
                          long long* last, volatile long long* started, long long ticket, const volatile long long* flag,
                          long long timeout_ns, volatile long long* done, int* err)
{
    __shared__ int sh_ok;
    __shared__ long long sh_seq;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x == 0)
    {
        *started = ticket;
        __threadfence();
        const long long prev = *last;
        long long v, sq = 0;
        const unsigned long long t0 = gtime();
        while (true)
        {
            v = *req;
            const long long a = v < 0 ? -v : v;
            sq = a / 64;
            if (sq > prev) break;
            if ((long long) (gtime() - t0) > timeout_ns) { atomicExch(err, 2); v = 0; sq = 0; break; }
        }
        sh_seq = v > 0 ? sq : -sq;
        if (v < 0) *last = sq;
    }
    __syncthreads();
    const long long seq = sh_seq;
    if (seq <= 0) return;
    if (!wait_flag(flag, seq, timeout_ns, &sh_ok, err)) return;
    const int nch = I / 128;
    for (int it = warp; it < topk * nch; it += WK)
    {
        const int j = it / nch, chunk = it % nch;
        if (!(msel[j] < 0 && __half2float(rw[j]) != 0.0f)) continue;
        const int lc = (int) sel[j];
        const int col = chunk * 128 + lane * 4;
        const size_t off = (size_t) j * I + col;
        auto load_sum = [&] (const half* buf, float& v0, float& v1, float& v2, float& v3)
        {
            v0 = v1 = v2 = v3 = 0.0f;
            float t0, t1, t2, t3;
            load_h4_cg(buf + off, t0, t1, t2, t3);
            v0 += t0; v1 += t1; v2 += t2; v3 += t3;
        };
        float u0, u1, u2, u3;
        load_sum(gu_u, u0, u1, u2, u3);
        had128(u0, u1, u2, u3, lane);
        scale_h4(((const half*) u_svh[lc]) + col, u0, u1, u2, u3);
        float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
        if (gated)
        {
            load_sum(gu_g, g0, g1, g2, g3);
            had128(g0, g1, g2, g3, lane);
            scale_h4(((const half*) g_svh[lc]) + col, g0, g1, g2, g3);
        }
        float a0 = act_gate(act, gated, g0, u0, act_limit);
        float a1 = act_gate(act, gated, g1, u1, act_limit);
        float a2 = act_gate(act, gated, g2, u2, act_limit);
        float a3 = act_gate(act, gated, g3, u3, act_limit);
        scale_h4(((const half*) d_suh[lc]) + col, a0, a1, a2, a3);
        had128(a0, a1, a2, a3, lane);
        store_h4(act_out + off, a0, a1, a2, a3);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *last = seq;
        __threadfence_system();
        *done = seq;
    }
}

// main stream, before the layer's MoE kernels: wait (bounded) until the epilogue block of this call is resident
__global__ void x16_gate_k(const volatile long long* started, long long ticket, long long timeout_ns, int* err)
{
    const unsigned long long t0 = gtime();
    while (*started < ticket)
        if ((long long) (gtime() - t0) > timeout_ns) { atomicExch(err, 3); break; }
}

// ---- kernel B reduction over all slots of token row 0 ----------------------------------------------------------------
// grid Ho / 128 x THREADS. Slot k's down output: d_cpu row k (host-mapped) for X16 slots (masked sel < 0), else the
// GPU's d_out row k. sel / rw: the token's ORIGINAL selection (global ids, min_expert -1) and fp16 routing weights.
__global__ void x16_red_k(const float* d_gpu, const float* d_cpu, const int64_t* msel, const int64_t* sel, const half* rw,
                          int topk, int H, int Ho, int H_out, const int64_t* d_svh, const half* x,
                          const half* sh_gate_w, const float* sh_out, float* out, const long long* dflag,
                          const volatile long long* flag, long long timeout_ns, int* err)
{
    __shared__ float sh_part[WK * 128];
    __shared__ float sh_gate;
    __shared__ int sh_ok;
    const long long seq = dflag[0];
    if (seq == 0) return;
    if (!wait_flag(flag, seq, timeout_ns, &sh_ok, err)) return;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int chunk = blockIdx.x;
    const int row = 0;
    const int col = chunk * 128 + lane * 4;

    if (sh_gate_w)
    {
        const half* xr = x + (size_t) row * H;
        float dot = 0.0f;
        for (int i = threadIdx.x * 2; i < H; i += THREADS * 2)
        {
            half2 xv = *((const half2*) (xr + i));
            half2 wv = *((const half2*) (sh_gate_w + i));
            dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) sh_part[warp] = dot;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            float tt = 0.0f;
            for (int w = 0; w < WK; ++w) tt += sh_part[w];
            sh_gate = 1.0f / (1.0f + __expf(-tt));
        }
        __syncthreads();
    }

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int k = warp; k < topk; k += WK)
    {
        const int sk = row * topk + k;
        const float w = __half2float(rw[sk]);
        if (!(w != 0.0f)) continue;
        const int lc = (int) sel[sk];
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        {
            float t0, t1, t2, t3;
            if (msel[sk] < 0) load_f4_cg(d_cpu + (size_t) k * Ho + col, t0, t1, t2, t3);
            else load_f4_cg(d_gpu + (size_t) sk * Ho + col, t0, t1, t2, t3);
            v0 += t0; v1 += t1; v2 += t2; v3 += t3;
        }
        had128(v0, v1, v2, v3, lane);
        scale_h4(((const half*) d_svh[lc]) + col, v0, v1, v2, v3);
        o0 += w * v0;
        o1 += w * v1;
        o2 += w * v2;
        o3 += w * v3;
    }
    float* part = sh_part + warp * 128 + lane * 4;
    __syncthreads();   // the gate dot's sh_part[0..WK) reads are done
    part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
    __syncthreads();
    if (warp == 0)
    {
        o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
        #pragma unroll
        for (int w = 0; w < WK; ++w)
        {
            const float* q = sh_part + w * 128 + lane * 4;
            o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
        }
        if (sh_out)
        {
            const float gv = sh_gate_w ? sh_gate : 1.0f;
            const float* sh = sh_out + (size_t) row * H + col;
            if (col + 0 < H) o0 += gv * sh[0];
            if (col + 1 < H) o1 += gv * sh[1];
            if (col + 2 < H) o2 += gv * sh[2];
            if (col + 3 < H) o3 += gv * sh[3];
        }
        float* dst = out + col;
        if (col + 3 < H_out)
            *((float4*) dst) = make_float4(o0, o1, o2, o3);
        else
        {
            if (col + 0 < H_out) dst[0] = o0;
            if (col + 1 < H_out) dst[1] = o1;
            if (col + 2 < H_out) dst[2] = o2;
        }
    }
}

}  // namespace

static int64_t P(const c10::optional<at::Tensor>& t) { return t.has_value() ? (int64_t) t->data_ptr() : 0; }

void x16_epa(int64_t gu_g, int64_t gu_u, int64_t I, at::Tensor sel, at::Tensor msel, at::Tensor rw, at::Tensor g_svh,
             at::Tensor u_svh, at::Tensor d_suh, int64_t act, double act_limit, bool gated, int64_t act_out, int64_t topk,
             int64_t req, int64_t last, int64_t started, int64_t ticket, int64_t flag, int64_t timeout_ns, int64_t done,
             at::Tensor err)
{
    auto stream = c10::cuda::getCurrentCUDAStream(sel.device().index()).stream();
    x16_epa_k<<<1, THREADS, 0, stream>>>(
        (const half*) gu_g, (const half*) gu_u, (int) I, (int) topk, sel.data_ptr<int64_t>(), msel.data_ptr<int64_t>(),
        (const half*) rw.data_ptr(), (const int64_t*) g_svh.data_ptr(), (const int64_t*) u_svh.data_ptr(),
        (const int64_t*) d_suh.data_ptr(), (int) act, (float) act_limit, gated, (half*) act_out,
        (const volatile long long*) req, (long long*) last, (volatile long long*) started, ticket,
        (const volatile long long*) flag, timeout_ns, (volatile long long*) done, err.data_ptr<int>());
}

void x16_gate(int64_t started, int64_t ticket, int64_t timeout_ns, at::Tensor err)
{
    auto stream = c10::cuda::getCurrentCUDAStream(err.device().index()).stream();
    x16_gate_k<<<1, 1, 0, stream>>>((const volatile long long*) started, ticket, timeout_ns, err.data_ptr<int>());
}

void x16_red(at::Tensor d_gpu, int64_t d_cpu, at::Tensor msel, at::Tensor sel, at::Tensor rw, int64_t topk, int64_t H,
             int64_t Ho, int64_t H_out, at::Tensor d_svh, at::Tensor x, c10::optional<at::Tensor> sh_gate_w,
             c10::optional<at::Tensor> sh_out, at::Tensor out, at::Tensor dflag, int64_t flag, int64_t timeout_ns,
             at::Tensor err)
{
    auto stream = c10::cuda::getCurrentCUDAStream(out.device().index()).stream();
    x16_red_k<<<(unsigned) (Ho / 128), THREADS, 0, stream>>>(
        d_gpu.data_ptr<float>(), (const float*) d_cpu, msel.data_ptr<int64_t>(), sel.data_ptr<int64_t>(),
        (const half*) rw.data_ptr(), (int) topk, (int) H, (int) Ho, (int) H_out, (const int64_t*) d_svh.data_ptr(),
        (const half*) x.data_ptr(), (const half*) P(sh_gate_w), (const float*) P(sh_out), out.data_ptr<float>(),
        (const long long*) dflag.data_ptr(), (const volatile long long*) flag, timeout_ns, err.data_ptr<int>());
}

// a high-priority NON-BLOCKING stream for x16_epa: torch's compute stream is the legacy default stream, which
// implicitly serializes with every blocking stream (torch.cuda.Stream), so the epilogue would hold the layer's MoE kernels
int64_t x16_stream(int64_t device)
{
    int lo = 0, hi = 0;
    cudaSetDevice((int) device);
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t st;
    TORCH_CHECK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, hi) == cudaSuccess, "x16_stream failed");
    return (int64_t) st;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("x16_stream", &x16_stream);
    m.def("x16_epa", &x16_epa);
    m.def("x16_gate", &x16_gate);
    m.def("x16_red", &x16_red);
}
