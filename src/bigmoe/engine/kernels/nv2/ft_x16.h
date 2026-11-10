// X16: bit-exact CPU emulation of exllamav3's fused decode GEMV (exl3_moe_coop_kernel.cuh gemv_tile; Ampere, WIDE
// geometry, ksplit 1, mul1 codebook), for the X16 lane (engine/x16.py): CPU-computed experts whose gate / up / down
// GEMV outputs are bit-identical to the GPU kernel's.
//
// Semantics of mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 on Ampere (determined on an RTX 3090; the start-up
// self-test checks them): every k8 half-step turns the fp16 accumulator c into
//     RN16( sum_i trunc0(t_i, 2^(emax - 24)) ),   t = { c, x_k * w_k for the 8 rows k },
//     emax = max( e(c), e(x_k) + e(w_k) )
// where e is the input's exponent (floor(log2 |v|), fp16 subnormals -14, zero terms excluded): each term is truncated
// toward zero onto the grid 24 bits below the largest ALIGNMENT exponent (input exponent sum, not the product's own),
// the truncated terms are summed exactly and the sum is rounded to fp16 once (nearest even).
// The tile: c restarts at 0 every 64 k (FOLD 4 k-slices of 16) and each fold is added in fp32 to its warp's
// accumulator; X16_WKK = 4 warps split k into equal contiguous ranges, each summing its folds in order from 0, and the
// 4 warp sums are added in warp order from 0. gate / up store that sum as fp16, down as fp32.
//
// Implementation (AVX-512 VBMI): the weights are decoded exactly as the GPU does (16-bit trellis window, bytesum of
// state * MUL1, hfma(1024 + s, KINV, KBIAS) rounded to fp16); the products are exact in fp32 (11 x 11 bits);
// trunc0(t * 2^(24 - emax)) = cvtt(scalef(t)) is exact in int32; the int32 sum is rounded to fp16 by round-to-odd to
// fp32 then nearest-even to fp16 (correct, 24 >= 11 + 2 bits).
// Lane layout: step q (0..7) of a tile decodes rows 2q', 2q' + 1 (q' = q & 3, + 8 for q >= 4) of its 16 columns; zmm A
// lane 4m + z holds column m (z = 0, 1: even / odd row) and m + 8 (z = 2, 3), zmm B columns 4 + m and 12 + m.

#define N135_X16_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi,avx512dq,f16c,fma")))

namespace {
namespace n135 {

constexpr int X16_WKK = 4;
alignas(64) inline uint8_t X16_PIDX[TF_N][8][64];
alignas(64) inline uint8_t X16_MSH[TF_N][8][64];
inline bool x16_fmt_ok[TF_N] = { false, false, false };

// byte gather (permutex2var) + bit extraction (multishift) tables per trellis format: qword m of step q holds the four
// 16-bit states j = {0,1,4,5} (low rows) or {2,3,6,7} (high rows) of trellis group 4m + q' (columns m and m + 8)
inline void init_x16()
{
    for (int f = 0; f < TF_N; ++f)
    {
        bool ok = true;
        for (int q = 0; q < 8; ++q)
            for (int m = 0; m < 8; ++m)
            {
                const int t = 4 * m + (q & 3);
                const int js[4] = { q < 4 ? 0 : 2, q < 4 ? 1 : 3, q < 4 ? 4 : 6, q < 4 ? 5 : 7 };
                int lo = 1 << 30;
                for (int k = 0; k < 4; ++k) lo = std::min(lo, tf_s(f, 8 * t + js[k]) - 16);
                const int qs = lo >= 0 ? lo / 8 : -((-lo + 7) / 8);
                for (int i = 0; i < 8; ++i) X16_PIDX[f][q][8 * m + i] = uint8_t(mem_of_stream_tb(qs + 7 - i, TF_TB[f]));
                for (int k = 0; k < 4; ++k)
                {
                    const int r = tf_s(f, 8 * t + js[k]) - 16 - 8 * qs;
                    if (r < 0 || r > 48) ok = false;
                    X16_MSH[f][q][8 * m + 2 * k] = uint8_t(48 - r);
                    X16_MSH[f][q][8 * m + 2 * k + 1] = uint8_t(56 - r);
                }
            }
        x16_fmt_ok[f] = ok && vbmi_ok;
    }
}

// alignment exponent of an fp16 input: floor(log2 |x|), fp16 subnormals -14 (exponent field 0), zero -inf (excluded)
inline float x16_exp(float x) { return x != 0.0f ? float(std::max(std::ilogb(x), -14)) : -INFINITY; }
// lane l of a tile's 32 output lanes (A 0..15, B 16..31; even lanes, the odd ones duplicate them) -> column 0..15
inline int x16_col(int l) { const int z = l >= 16, e = l & 15, m = e >> 2, j = e & 3; return 4 * z + m + ((j >= 2) ? 8 : 0); }

// RN16(S * 2^(m - 24)) for int32 S: round-to-odd to fp32, exact scale, nearest-even to fp16
N135_X16_TARGET __attribute__((always_inline)) inline __m512 x16_rn16(__m512i S, __m512 m)
{
    const __m512 frz = _mm512_cvt_roundepi32_ps(S, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __mmask16 stk = _mm512_cmpneq_epi32_mask(_mm512_cvttps_epi32(frz), S);
    const __m512 fodd = _mm512_castsi512_ps(_mm512_mask_or_epi32(_mm512_castps_si512(frz), stk, _mm512_castps_si512(frz),
                                                                 _mm512_set1_epi32(1)));
    const __m512 v = _mm512_scalef_ps(fodd, _mm512_add_ps(m, _mm512_set1_ps(-24.0f)));
    return _mm512_cvtph_ps(_mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

// Tile rows [r0, r0 + R) (R a multiple of 4: whole folds) of tiles [t0, t0 + C) (C <= MAXCT) of mat (native layout).
// x: activations in k order (fp32 holding fp16 values, the GEMV's input), ex: x16_exp(x).
// The folds are added in order to wsum[t * 32 + lane] (C * 32 floats, zeroed here): the GPU warp's fp32 accumulator
// when [r0, r0 + R) is exactly one warp's k range (k / 16 / X16_WKK tile rows).
template <int F>
N135_X16_TARGET void rect_x16_f(const Mat& mat, int r0, int R, int t0, int C, const float* x, const float* ex, float* wsum)
{
    constexpr int tb = TF_TB[F];
    const __m512i c1 = _mm512_set1_epi16(short(MUL1 & 0xffff)), c2 = _mm512_set1_epi16(short(MUL1 >> 16));
    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512 ka = _mm512_set1_ps(KINV * 16777216.0f), kb = _mm512_set1_ps(CAFF);
    const __m512 f24 = _mm512_set1_ps(24.0f), lo30 = _mm512_set1_ps(-1e30f), m14 = _mm512_set1_ps(-14.0f);
    constexpr __mmask64 hmask = tb > 64 ? ((1ull << (tb - 64)) - 1) : 0;
    const size_t row_stride = size_t(mat.n / 16) * tb;
    const uint8_t* base = mat.tr + size_t(t0) * tb;
    const int lines = (C * tb + 63) / 64, pfr = K_.pfr, kt = mat.k / 16;
    auto pf = [&](const uint8_t* q) { const char* c = reinterpret_cast<const char*>(q); for (int l = 0; l < lines; ++l) _mm_prefetch(c + l * 64, _MM_HINT_T0); };
    if (pfr > 0 && K_.pfpro) for (int r = r0; r < std::min(r0 + pfr, r0 + R); ++r) pf(base + size_t(r) * row_stride);
    static thread_local float* cst = nullptr;   // fp16 accumulators of the open fold, per tile
    if (!cst) cst = static_cast<float*>(std::aligned_alloc(64, size_t(MAXCT) * 32 * 4));
    std::memset(cst, 0, size_t(C) * 32 * 4);
    std::memset(wsum, 0, size_t(C) * 32 * 4);
    for (int r = r0; r < r0 + R; ++r)
    {
        const uint8_t* p = base + size_t(r) * row_stride;
        if (pfr > 0 && r + pfr < kt) pf(p + size_t(pfr) * row_stride);
        const float* xr = x + size_t(r) * 16; const float* er = ex + size_t(r) * 16;
        #pragma GCC unroll 1
        for (int t = 0; t < C; ++t)
        {
            const uint8_t* tile = p + size_t(t) * tb;
            const __m512i ta = _mm512_loadu_si512(tile);
            const __m512i tbv = _mm512_maskz_loadu_epi8(hmask, tile + 64);
            __m512 cA = _mm512_load_ps(cst + t * 32), cB = _mm512_load_ps(cst + t * 32 + 16);
            #pragma GCC unroll 2
            for (int h = 0; h < 2; ++h)
            {
                __m512 pA[4], pB[4];
                __m512 mA = _mm512_getexp_ps(cA), mB = _mm512_getexp_ps(cB);   // e(c), zero excluded (-inf)
                mA = _mm512_mask_max_ps(mA, _mm512_cmp_ps_mask(cA, _mm512_setzero_ps(), _CMP_NEQ_OQ), mA, m14);
                mB = _mm512_mask_max_ps(mB, _mm512_cmp_ps_mask(cB, _mm512_setzero_ps(), _CMP_NEQ_OQ), mB, m14);
                #pragma GCC unroll 4
                for (int qq = 0; qq < 4; ++qq)
                {
                    const int q = 4 * h + qq;
                    const __m512i src = _mm512_permutex2var_epi8(ta, _mm512_load_si512(X16_PIDX[F][q]), tbv);
                    const __m512i st = _mm512_multishift_epi64_epi8(_mm512_load_si512(X16_MSH[F][q]), src);
                    const __m512i lo = _mm512_mullo_epi16(st, c1);
                    const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(st, c1), _mm512_mullo_epi16(st, c2));
                    const __m512i sm = _mm512_add_epi16(_mm512_maddubs_epi16(lo, one8), _mm512_maddubs_epi16(hi, one8));
                    __m512 w0 = _mm512_cvtph_ps(_mm512_castsi512_si256(sm));
                    __m512 w1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(sm, 1));
                    w0 = _mm512_fmadd_ps(w0, ka, kb); w1 = _mm512_fmadd_ps(w1, ka, kb);
                    w0 = _mm512_cvtph_ps(_mm512_cvtps_ph(w0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    w1 = _mm512_cvtph_ps(_mm512_cvtps_ph(w1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    const int rr = 2 * qq + 8 * h;
                    double xp2, ep2; std::memcpy(&xp2, xr + rr, 8); std::memcpy(&ep2, er + rr, 8);
                    const __m512 xb = _mm512_castpd_ps(_mm512_set1_pd(xp2)), eb = _mm512_castpd_ps(_mm512_set1_pd(ep2));
                    pA[qq] = _mm512_mul_ps(w0, xb); pB[qq] = _mm512_mul_ps(w1, xb);
                    mA = _mm512_max_ps(mA, _mm512_add_ps(_mm512_getexp_ps(w0), eb));
                    mB = _mm512_max_ps(mB, _mm512_add_ps(_mm512_getexp_ps(w1), eb));
                }
                // lanes (2i, 2i + 1) hold the same column (even / odd rows): one emax, c counted in the even lane
                mA = _mm512_max_ps(mA, _mm512_permute_ps(mA, 0xB1)); mB = _mm512_max_ps(mB, _mm512_permute_ps(mB, 0xB1));
                const __m512 sA = _mm512_sub_ps(f24, mA), sB = _mm512_sub_ps(f24, mB);
                __m512i SA = _mm512_cvttps_epi32(_mm512_scalef_ps(_mm512_maskz_mov_ps(0x5555, cA), sA));
                __m512i SB = _mm512_cvttps_epi32(_mm512_scalef_ps(_mm512_maskz_mov_ps(0x5555, cB), sB));
                #pragma GCC unroll 4
                for (int qq = 0; qq < 4; ++qq)
                {
                    SA = _mm512_add_epi32(SA, _mm512_cvttps_epi32(_mm512_scalef_ps(pA[qq], sA)));
                    SB = _mm512_add_epi32(SB, _mm512_cvttps_epi32(_mm512_scalef_ps(pB[qq], sB)));
                }
                SA = _mm512_add_epi32(SA, _mm512_shuffle_epi32(SA, _MM_PERM_CDAB));
                SB = _mm512_add_epi32(SB, _mm512_shuffle_epi32(SB, _MM_PERM_CDAB));
                const __mmask16 vA = _mm512_cmp_ps_mask(mA, lo30, _CMP_GT_OQ), vB = _mm512_cmp_ps_mask(mB, lo30, _CMP_GT_OQ);
                cA = _mm512_maskz_mov_ps(vA, x16_rn16(SA, mA)); cB = _mm512_maskz_mov_ps(vB, x16_rn16(SB, mB));
            }
            if ((r & 3) == 3)
            {
                float* f = wsum + size_t(t) * 32;
                _mm512_storeu_ps(f, _mm512_add_ps(_mm512_loadu_ps(f), cA));
                _mm512_storeu_ps(f + 16, _mm512_add_ps(_mm512_loadu_ps(f + 16), cB));
                cA = _mm512_setzero_ps(); cB = _mm512_setzero_ps();
            }
            _mm512_store_ps(cst + t * 32, cA); _mm512_store_ps(cst + t * 32 + 16, cB);
        }
    }
}

inline void rect_x16(const Mat& mat, int r0, int R, int t0, int C, const float* x, const float* ex, float* wsum)
{
    switch (mat.tf)
    {
        case TF_K2: rect_x16_f<TF_K2>(mat, r0, R, t0, C, x, ex, wsum); break;
        case TF_K25: rect_x16_f<TF_K25>(mat, r0, R, t0, C, x, ex, wsum); break;
        default: rect_x16_f<TF_K3>(mat, r0, R, t0, C, x, ex, wsum); break;
    }
}

// the warp sums of a whole matrix (wsum[w][C tiles][32 lanes]) -> GEMV output y[t * 16 + column], added in warp order
inline void x16_reduce(const float* ws, int C, int t0, int t1, float* y)
{
    for (int t = t0; t < t1; ++t)
        for (int l = 0; l < 32; l += 2)
        {
            float tot = 0.0f;
            for (int w = 0; w < X16_WKK; ++w) tot += ws[(size_t(w) * C + t) * 32 + l];
            y[t * 16 + x16_col(l)] = tot;
        }
}

// a matrix X16 can compute: native layout, a supported trellis format, k a multiple of 64 * X16_WKK, n of 16
inline bool x16_mat_ok(const Mat& m)
{
    return !m.swz && m.tf >= 0 && m.tf < TF_N && x16_fmt_ok[m.tf] && m.k % (64 * X16_WKK) == 0 && m.n % 16 == 0
           && m.n / 16 <= 8 * MAXCT;
}

// single-threaded whole-matrix GEMV (the X16 self-test): x fp32 [k] (the GEMV's input) -> y fp32 [n]
inline void x16_gemv(const Mat& M, const float* x, float* y)
{
    const int C = M.n / 16, kpw = M.k / 16 / X16_WKK, CG = 64;
    std::vector<float> ex(M.k), ws(size_t(X16_WKK) * C * 32);
    for (int i = 0; i < M.k; ++i) ex[i] = x16_exp(x[i]);
    for (int w = 0; w < X16_WKK; ++w)
        for (int c0 = 0; c0 < C; c0 += CG)
            rect_x16(M, w * kpw, kpw, c0, std::min(CG, C - c0), x, ex.data(), ws.data() + (size_t(w) * C + c0) * 32);
    x16_reduce(ws.data(), C, 0, C, y);
}

}  // namespace n135
}  // namespace
