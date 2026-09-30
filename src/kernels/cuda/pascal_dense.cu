// src/kernels/cuda/pascal_dense.cu - see include/strata/kernels/pascal_dense.hpp.
//
// Layout of one launch (T = NC activation columns):
//   * a block of 4 or 8 warps owns 4 rows per warp; each half-warp (16 lanes) owns TWO rows and each of its lanes a
//     32-value "unit" (one weight group) of both rows at a time, so every staged activation chunk read from shared
//     memory feeds two rows;
//   * the K dimension is tiled: all T columns of a K-slice of KT values are staged in shared memory as fp16
//     (q8_1 value q as q/128, exact; the per-32 scale d*128 as a float), __syncthreads, every lane multiplies its
//     units, repeat.  KT is a multiple of 512 = 16 lanes x 32 values, so on n_in % 512 == 0 every lane of a row pair
//     gets the same number of units per tile.  Staged bytes per block <= ~18.5 KB, so 3 blocks fit per SM;
//   * a unit: its weights are decoded once per row into half2 (8 values per step) and multiplied into one half2
//     accumulator per (row, column); at the end of each scale group (32 values, 16 for Q6_K) the accumulator is
//     folded into an fp32 sum with the block/sub-block scale times the activation's per-32 scale.
//
// FP16 RANGE.  Staged activations satisfy |x| = |q|/128 <= 1.  Between two flushes one half lane of an accumulator
// sums at most 16 products (8 for Q6_K), each bounded by the format's largest magnitude BEFORE any scale:
//     Q8_0 |q| <= 128           -> 16 * 128 = 2048
//     IQ4_NL / IQ4_XS |kv| <= 127 -> 16 * 127 = 2032
//     Q4_K |q - 8| <= 8          -> 16 * 8   =  128   (the min is applied in fp32 from the staged q8_1 sums)
//     Q5_K |q - 16| <= 16        -> 16 * 16  =  256   (likewise)
//     Q6_K |q - 32| <= 32         ->  8 * 32  =  256   (per 16-value sub-block: its int8 scale is applied in fp32)
// and the flush adds the two half lanes in fp16 (<= 4096).  Every fp16 partial sum stays far below 65504, whatever
// the block scales (d, dmin, the K-quant sub-block scales and the q8_1 d are all applied in fp32).
#include "strata/kernels/pascal_dense.hpp"

#if defined(__HIPCC__)
// AMD has dot-product instructions and no Pascal: the int8 path in native_mmvq.cu stays.
namespace strata::kernels {
bool pascal_dense_mmvq(int, const void*, const void*, float*, int, int, int, void*) { return false; }
void pascal_dense_fp16_override(int) {}
}  // namespace strata::kernels
#else

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int LPR = 16;   // lanes per row pair: a half warp

static_assert(sizeof(block_q8_0) == 34 && sizeof(block_q4_K) == 144 && sizeof(block_q5_K) == 176 &&
              sizeof(block_q6_K) == 210 && sizeof(block_iq4_nl) == 18 && sizeof(block_iq4_xs) == 136 &&
              sizeof(block_q8_1) == 36, "block sizes");

// The staged K-slice per column: <= 8192 values over all columns (16 KB of fp16), a multiple of 512, <= 4096.
// T = 2: 4096, 3: 2560, 4: 2048, 5: 1536, 6..8: 1024.
__host__ __device__ constexpr int kt_of(int nc) {
    return (8192 / nc) / 512 * 512 > 4096 ? 4096 : (8192 / nc) / 512 * 512;
}

// ------------------------------------------------------------------------------------------------ formats
template<int TY> struct Fmt;
// BE values and BB bytes per block; SUMS: the format has a min (needs the staged per-32 activation sums);
// IL: the staged activation of a 32-group is interleaved as half2 (x[j], x[j + 16]) (IQ4 nibble order);
// Q6: the Q6_K chunk swizzle (see swz).
template<> struct Fmt<8>  { static constexpr int BE = 32,  BB = 34;  static constexpr bool SUMS = false, IL = false, Q6 = false; };
template<> struct Fmt<12> { static constexpr int BE = 256, BB = 144; static constexpr bool SUMS = true,  IL = false, Q6 = false; };
template<> struct Fmt<13> { static constexpr int BE = 256, BB = 176; static constexpr bool SUMS = true,  IL = false, Q6 = false; };
template<> struct Fmt<14> { static constexpr int BE = 256, BB = 210; static constexpr bool SUMS = false, IL = false, Q6 = true;  };
template<> struct Fmt<20> { static constexpr int BE = 32,  BB = 18;  static constexpr bool SUMS = false, IL = true,  Q6 = false; };
template<> struct Fmt<23> { static constexpr int BE = 256, BB = 136; static constexpr bool SUMS = false, IL = true,  Q6 = false; };

// Staged activations are addressed in 16-byte chunks (8 halves).  The chunk index is XOR-swizzled so the 8 lanes of
// a quarter warp, which read 8 consecutive units, hit 8 different bank quads in one LDS.128:
//   * one unit = one 32-group = chunks 4u..4u+3: XOR bits 3-4 into bits 0-1;
//   * Q6_K: unit (n, j, h) of a superblock reads chunks 16n + 4h + 2j (+1, +8, +9): XOR bit 4 (n) into bit 0.
template<int TY> __device__ __forceinline__ int swz(int c) {
    if constexpr (Fmt<TY>::Q6) return c ^ ((c >> 4) & 1);
    else return c ^ ((c >> 3) & 3);
}

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ half2 as_h2(uint32_t u) { half2 h; memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t as_u(half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return __ldg((const unsigned short*) p); }
__device__ __forceinline__ uint32_t ld32a(const uint8_t* p) { return __ldg((const unsigned int*) p); }   // 4-aligned
__device__ __forceinline__ float ldh(const uint8_t* p) { return __half2float(__ushort_as_half((unsigned short) ld16(p))); }
// 16 bytes at a 2-byte aligned address: aligned 32-bit loads and a funnel shift.  Reads only [p - 2, p + 16): the
// word before p is inside the buffer because the buffer (and so every 4-aligned word) starts 4-aligned.
__device__ __forceinline__ void ld16B(const uint8_t* p, uint32_t (&o)[4]) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t* base = (const uint32_t*) (a & ~(uintptr_t) 3);
    const uint32_t sh = (uint32_t) (a & 2) * 8;
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = __ldg(base + i);
    w[4] = sh ? ld16((const uint8_t*) (base + 4)) : 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = __funnelshift_r(w[i], w[i + 1], sh);
}
// four unsigned bytes -> (b0,b1), (b2,b3) as exact halves minus a bias: 0x64XX is 1024 + XX, `k` = 1024 + bias
__device__ __forceinline__ void u8x4_h2(uint32_t g, uint32_t k, half2& lo, half2& hi) {
    lo = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4140)), as_h2(k));
    hi = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4342)), as_h2(k));
}
// four SIGNED bytes: bias by 128 first, then 1024 + 128 + b - 1152
__device__ __forceinline__ void s8x4_h2(uint32_t g, half2& lo, half2& hi) {
    u8x4_h2(g ^ 0x80808080u, 0x64806480u, lo, hi);
}
// four signed q8_1 bytes -> q/128, exact: (1152 + q) / 128 - 9 in one HFMA2 (the exact result is representable)
__device__ __forceinline__ void s8x4_q128(uint32_t g, half2& lo, half2& hi) {
    const uint32_t b = g ^ 0x80808080u;
    const half2 s = as_h2(0x20002000u), o = as_h2(0xC880C880u);   // 1/128, -9
    lo = __hfma2(as_h2(__byte_perm(b, 0x64646464u, 0x4140)), s, o);
    hi = __hfma2(as_h2(__byte_perm(b, 0x64646464u, 0x4342)), s, o);
}
__device__ __forceinline__ float hsum(half2 a) { return __half2float(__hadd(__low2half(a), __high2half(a))); }
__device__ __forceinline__ int sum_s8x4(uint32_t g) {
    return (int) (int8_t) g + (int) (int8_t) (g >> 8) + (int) (int8_t) (g >> 16) + (int) (int8_t) (g >> 24);
}

// ------------------------------------------------------------------------------------------------ staging
// ng 32-groups of every column starting at group g0.  One item = (column, group, w): q8_1 words w and w + 4 of the
// group (values 4w..4w+3 and 16+4w..16+4w+3).  ng is a multiple of 8, so a warp's items are all valid or all past
// the end and the 4-lane shuffle of the sums is uniform.
template<int TY, int NC>
__device__ __forceinline__ void stage(const block_q8_1* __restrict__ x, int xs, int g0, int ng, half* s_x,
                                      float* s_xd, float* s_xs) {
    constexpr int KT = kt_of(NC), G = KT / 32;
    const int per_col = ng * 4, items = NC * per_col;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int c = it / per_col, rem = it - c * per_col, g = rem >> 2, w = rem & 3;
        const block_q8_1* b = x + (size_t) c * xs + g0 + g;
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(b->qs);
        const uint32_t q0 = __ldg(qw + w), q1 = __ldg(qw + 4 + w);
        half2 a0, a1, b0, b1;
        s8x4_q128(q0, a0, a1);
        s8x4_q128(q1, b0, b1);
        half* sx = s_x + c * KT;
        if constexpr (Fmt<TY>::IL) {
            // half2 slot 4w + i of the group = (x[4w + i], x[16 + 4w + i]): chunk 4g + w
            const uint4 v = make_uint4(as_u(__lows2half2(a0, b0)), as_u(__highs2half2(a0, b0)),
                                       as_u(__lows2half2(a1, b1)), as_u(__highs2half2(a1, b1)));
            *reinterpret_cast<uint4*>(sx + 8 * swz<TY>(4 * g + w)) = v;
        } else {
            *reinterpret_cast<uint2*>(sx + 8 * swz<TY>(4 * g + (w >> 1)) + 4 * (w & 1)) = make_uint2(as_u(a0), as_u(a1));
            *reinterpret_cast<uint2*>(sx + 8 * swz<TY>(4 * g + 2 + (w >> 1)) + 4 * (w & 1)) = make_uint2(as_u(b0), as_u(b1));
        }
        const float d = __low2float(__ldg(&b->ds));
        if (w == 0) s_xd[c * G + g] = d * 128.0f;
        if constexpr (Fmt<TY>::SUMS) {
            // the q8_1 integer sum times d, as the int8 path's dp4a(0x01010101, u) * d8
            int s = sum_s8x4(q0) + sum_s8x4(q1);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            if (w == 0) s_xs[c * G + g] = d * (float) s;
        }
    }
}

// ------------------------------------------------------------------------------------------------ unit helpers
// A unit's weights, decoded: wv[r][4p + i] is half2 i of chunk p (8 values) of row r.  Per column: chunk p is
// multiplied with the staged activation chunk at off[p] (in halves) into one half2 accumulator per row, folded into
// fp32 with the weight scale times the activation's per-32 scale at the end of each scale group.  Q6 (two 16-value
// sub-blocks): chunks 0-1 fold with wsA and activation group xgA, chunks 2-3 with wsB / xgB; otherwise all four fold
// once with wsB / xgB.  Column-outer order: only two fp16 accumulators are live at a time.
template<int NC, bool Q6>
__device__ __forceinline__ void cols(float (&acc)[2][NC], const half2 (&wv)[2][16], const int (&off)[4],
                                     const float (&wsA)[2], int xgA, const float (&wsB)[2], int xgB,
                                     const half* s_x, const float* s_xd) {
    constexpr int KT = kt_of(NC), G = KT / 32;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const half* sx = s_x + c * KT;
        half2 h[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint4 xv = *reinterpret_cast<const uint4*>(sx + off[p]);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                h[r] = __hfma2(wv[r][4 * p + 0], as_h2(xv.x), h[r]);
                h[r] = __hfma2(wv[r][4 * p + 1], as_h2(xv.y), h[r]);
                h[r] = __hfma2(wv[r][4 * p + 2], as_h2(xv.z), h[r]);
                h[r] = __hfma2(wv[r][4 * p + 3], as_h2(xv.w), h[r]);
            }
            if (Q6 && p == 1) {
                const float xd = s_xd[c * G + xgA];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    acc[r][c] = fmaf(wsA[r] * xd, hsum(h[r]), acc[r][c]);
                    h[r] = __float2half2_rn(0.0f);
                }
            }
        }
        const float xd = s_xd[c * G + xgB];
#pragma unroll
        for (int r = 0; r < 2; ++r) acc[r][c] = fmaf(wsB[r] * xd, hsum(h[r]), acc[r][c]);
    }
}
// one 32-group per unit: chunks 4 gl .. 4 gl + 3
template<int TY> __device__ __forceinline__ void group_offs(int gl, int (&off)[4]) {
#pragma unroll
    for (int p = 0; p < 4; ++p) off[p] = 8 * swz<TY>(4 * gl + p);
}

// ------------------------------------------------------------------------------------------------ units
// gg: the unit's 32-group index in the row (global); gl: its index in the staged tile.
template<int TY, int NC> struct Unit;

// Q6_K (dequantize_row_q6_K): block = ql[128] qh[64] scales[16] d.  Half n (128 values) of a superblock, value l of
// quadrant qu (0..3), l in 0..31, is  (ql[64n + l + 32(qu & 1)] >> 4(qu >> 1)) & 0xF | ((qh[32n + l] >> 2qu) & 3) << 4,
// minus 32, times d * scales[8n + 2qu + l / 16], at position 128n + 32qu + l.  A unit (n, j, h) takes l = 16j..16j+15
// of quadrants h and h + 2: 16 bytes of ql (low nibbles -> quadrant h, high -> h + 2) and 16 bytes of qh, two
// 16-value sub-blocks with their own scale and activation group.
template<int NC> struct Unit<14, NC> {
    __device__ static __forceinline__ void run(const uint8_t* const (&wr)[2], int gg, int gl, const half* s_x,
                                               const float* s_xd, const float*, const uint32_t*, float (&acc)[2][NC]) {
        const int sub = gg & 7, n = sub >> 2, j = (sub >> 1) & 1, hq = sub & 1;
        const uint32_t shA = 4 - 2 * hq, shB = 2 * hq;
        half2 wv[2][16];
        float wsA[2], wsB[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint8_t* b = wr[r] + (unsigned) (gg >> 3) * 210u;
            uint32_t A[4], H[4];
            ld16B(b + 64 * n + 32 * hq + 16 * j, A);
            ld16B(b + 128 + 32 * n + 16 * j, H);
            const float d = ldh(b + 208);
            wsA[r] = d * (float) (int8_t) __ldg(b + 192 + 8 * n + 2 * hq + j);
            wsB[r] = d * (float) (int8_t) __ldg(b + 196 + 8 * n + 2 * hq + j);
#pragma unroll
            for (int k = 0; k < 4; ++k) {   // word k: l = 16j + 4k .. +3; chunk k/2 of quadrant h, 2 + k/2 of h + 2
                const uint32_t lo = (A[k] & 0x0F0F0F0Fu) | ((H[k] << shA) & 0x30303030u);
                const uint32_t hi = ((A[k] >> 4) & 0x0F0F0F0Fu) | ((H[k] >> shB) & 0x30303030u);
                u8x4_h2(lo, 0x64206420u, wv[r][2 * k], wv[r][2 * k + 1]);          // q - 32
                u8x4_h2(hi, 0x64206420u, wv[r][8 + 2 * k], wv[r][8 + 2 * k + 1]);
            }
        }
        const int lsb = gl >> 3, cb = 32 * lsb + 16 * n + 4 * hq + 2 * j, xg = 8 * lsb + 4 * n + hq;
        const int off[4] = {8 * swz<14>(cb), 8 * swz<14>(cb + 1), 8 * swz<14>(cb + 8), 8 * swz<14>(cb + 9)};
        cols<NC, true>(acc, wv, off, wsA, xg, wsB, xg + 2, s_x, s_xd);
    }
};

// Q4_K / Q5_K (dequantize_row_q4_K / q5_K): group g of a superblock (j64 = g / 2, hi = g & 1), value l in 0..31 is
//   q = (qs[32 j64 + l] >> 4hi) & 0xF  (+ 16 * ((qh[l] >> g) & 1) for Q5_K);  w = d * sc_g * q - dmin * m_g
// with (sc_g, m_g) = get_scale_min_k4(g, scales).  The code is centred, c = q - 8 (Q5_K: q - 16), which halves the
// fp16 partial sums at no cost (the bias is in the byte->half constant):
//   sum w x = d sc_g xd sum(c * x/xd) - (dmin m_g - 8 d sc_g) sum(x)        (16 for Q5_K)
template<int TY, int NC> struct UnitK {
    __device__ static __forceinline__ void run(const uint8_t* const (&wr)[2], int gg, int gl, const half* s_x,
                                               const float* s_xd, const float* s_xs, const uint32_t*,
                                               float (&acc)[2][NC]) {
        constexpr bool Q5 = TY == 13;
        constexpr unsigned BB = Q5 ? 176u : 144u;
        constexpr int QS = Q5 ? 48 : 16;
        const int g = gg & 7, j64 = g >> 1, hi = g & 1;
        half2 wv[2][16];
        float ws[2], mn[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint8_t* b = wr[r] + (unsigned) (gg >> 3) * BB;
            const half2 dm = as_h2(ld32a(b));
            const uint32_t s0 = ld32a(b + 4), s1 = ld32a(b + 8), s2 = ld32a(b + 12);
            const uint32_t sh = 8 * (g & 3);
            const uint32_t b0 = (s0 >> sh) & 0xFF, b1 = (s1 >> sh) & 0xFF, b2 = (s2 >> sh) & 0xFF;
            const uint32_t sc = g < 4 ? (b0 & 63) : ((b2 & 0xF) | ((b0 >> 6) << 4));
            const uint32_t m = g < 4 ? (b1 & 63) : ((b2 >> 4) | ((b1 >> 6) << 4));
            ws[r] = __low2float(dm) * (float) sc;
            mn[r] = __high2float(dm) * (float) m - (Q5 ? 16.0f : 8.0f) * ws[r];
#pragma unroll
            for (int k = 0; k < 8; ++k) {   // word k: l = 4k .. 4k + 3
                uint32_t v = (ld32a(b + QS + 32 * j64 + 4 * k) >> (4 * hi)) & 0x0F0F0F0Fu;
                if constexpr (Q5) v |= ((ld32a(b + 16 + 4 * k) >> g) & 0x01010101u) << 4;
                u8x4_h2(v, Q5 ? 0x64106410u : 0x64086408u, wv[r][2 * k], wv[r][2 * k + 1]);   // q - 16 / q - 8
            }
        }
        int off[4];
        group_offs<TY>(gl, off);
        cols<NC, false>(acc, wv, off, ws, gl, ws, gl, s_x, s_xd);
        constexpr int G = kt_of(NC) / 32;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float xs = s_xs[c * G + gl];
#pragma unroll
            for (int r = 0; r < 2; ++r) acc[r][c] = fmaf(-mn[r], xs, acc[r][c]);
        }
    }
};
template<int NC> struct Unit<12, NC> : UnitK<12, NC> {};
template<int NC> struct Unit<13, NC> : UnitK<13, NC> {};

// Q8_0: w = d * (int8) qs[l]
template<int NC> struct Unit<8, NC> {
    __device__ static __forceinline__ void run(const uint8_t* const (&wr)[2], int gg, int gl, const half* s_x,
                                               const float* s_xd, const float*, const uint32_t*, float (&acc)[2][NC]) {
        half2 wv[2][16];
        float ws[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint8_t* b = wr[r] + (unsigned) gg * 34u;
            ws[r] = ldh(b);
            uint32_t q[8];
            ld16B(b + 2, *reinterpret_cast<uint32_t(*)[4]>(q));
            ld16B(b + 18, *reinterpret_cast<uint32_t(*)[4]>(q + 4));
#pragma unroll
            for (int k = 0; k < 8; ++k) s8x4_h2(q[k], wv[r][2 * k], wv[r][2 * k + 1]);
        }
        int off[4];
        group_offs<8>(gl, off);
        cols<NC, false>(acc, wv, off, ws, gl, ws, gl, s_x, s_xd);
    }
};

// IQ4_NL / IQ4_XS (dequantize_row_iq4_nl / iq4_xs): byte j of a 32-group's 16 code bytes holds value j (low nibble)
// and j + 16 (high nibble); s_nl[byte] is (kvalues[lo], kvalues[hi]) and the activation is staged interleaved to
// match, so each byte is one table load.  IQ4_XS group ib: scale d * (ls - 32) with
// ls = ((scales_l[ib / 2] >> 4(ib % 2)) & 0xF) | (((scales_h >> 2ib) & 3) << 4).
template<int TY, int NC> struct UnitIQ4 {
    __device__ static __forceinline__ void run(const uint8_t* const (&wr)[2], int gg, int gl, const half* s_x,
                                               const float* s_xd, const float*, const uint32_t* s_nl,
                                               float (&acc)[2][NC]) {
        half2 wv[2][16];
        float ws[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            uint32_t q[4];
            if constexpr (TY == 20) {
                const uint8_t* b = wr[r] + (unsigned) gg * 18u;
                ws[r] = ldh(b);
                ld16B(b + 2, q);
            } else {
                const uint8_t* b = wr[r] + (unsigned) (gg >> 3) * 136u;
                const int ib = gg & 7;
                const uint32_t hd = ld32a(b), sl = ld32a(b + 4);
                const int ls = (int) (((sl >> (4 * ib)) & 0xF) | (((hd >> (16 + 2 * ib)) & 3) << 4));
                ws[r] = __low2float(as_h2(hd)) * (float) (ls - 32);
#pragma unroll
                for (int k = 0; k < 4; ++k) q[k] = ld32a(b + 8 + 16 * ib + 4 * k);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k)
#pragma unroll
                for (int i = 0; i < 4; ++i) wv[r][4 * k + i] = as_h2(s_nl[(q[k] >> (8 * i)) & 0xFF]);
        }
        int off[4];
        group_offs<TY>(gl, off);
        cols<NC, false>(acc, wv, off, ws, gl, ws, gl, s_x, s_xd);
    }
};
template<int NC> struct Unit<20, NC> : UnitIQ4<20, NC> {};
template<int NC> struct Unit<23, NC> : UnitIQ4<23, NC> {};

// ------------------------------------------------------------------------------------------------ kernel
template<int TY, int NC>
__global__ void __launch_bounds__(256, 3) pdense_kernel(const uint8_t* __restrict__ w,
                                                        const block_q8_1* __restrict__ x, float* __restrict__ y,
                                                        int n_in, int n_out, unsigned row_bytes) {
    using F = Fmt<TY>;
    constexpr int KT = kt_of(NC), G = KT / 32;
    __shared__ __align__(16) half s_x[NC * KT];
    __shared__ float s_xd[NC * G];
    __shared__ float s_xs[F::SUMS ? NC * G : 1];
    __shared__ uint32_t s_nl[F::IL ? 256 : 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, lu = lane & (LPR - 1);
    const int row = blockIdx.x * (blockDim.x / 8) + warp * 4 + (lane >> 4) * 2;   // this half warp: row, row + 1
    if constexpr (F::IL) {
        for (int v = threadIdx.x; v < 256; v += blockDim.x)
            s_nl[v] = as_u(__halves2half2(__int2half_rn(kvalues_iq4nl[v & 15]), __int2half_rn(kvalues_iq4nl[v >> 4])));
    }
    // rows past the end compute on the last row and are not stored (the whole warp stays in the shuffles)
    const uint8_t* const wr[2] = {w + (size_t) min(row, n_out - 1) * row_bytes,
                                  w + (size_t) min(row + 1, n_out - 1) * row_bytes};
    float acc[2][NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[0][c] = acc[1][c] = 0.0f;
    const int xs = n_in / 32;
    for (int k0 = 0; k0 < n_in; k0 += KT) {
        const int g0 = k0 / 32, ng = min(KT, n_in - k0) / 32;
        __syncthreads();   // the previous tile consumed (the first time: s_nl filled)
        stage<TY, NC>(x, xs, g0, ng, s_x, s_xd, s_xs);
        __syncthreads();
        for (int gl = lu; gl < ng; gl += LPR) Unit<TY, NC>::run(wr, g0 + gl, gl, s_x, s_xd, s_xs, s_nl, acc);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lu == 0 && row + r < n_out) y[(size_t) c * n_out + row + r] = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ host
int g_override = -1;

bool enabled_here() {
    if (g_override >= 0) return g_override == 1;
    static int env = -1;
    if (env < 0) {
        const char* v = std::getenv("STRATA_PASCAL_FP16");
        env = (v != nullptr && std::strcmp(v, "0") == 0) ? 0 : 1;
    }
    if (env == 0) return false;
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (cc[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cc[dev] = major > 0 ? major : 99;
    }
    return cc[dev] == 6;
}

template<int TY, int NC>
void launch_nc(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    // 8 warps (32 rows) per block only for tall matrices (the LM head): 4 warps keep enough blocks for 56 SMs on the
    // 2560..8192-row projections.  Not measured (no sm_60 card in the dev loop): a first guess.
    const int nw = n_out >= 16384 ? 8 : 4, rows = nw * 4;
    const unsigned grid = (unsigned) ((n_out + rows - 1) / rows);
    const unsigned row_bytes = (unsigned) (n_in / Fmt<TY>::BE * Fmt<TY>::BB);
    pdense_kernel<TY, NC><<<grid, nw * 32, 0, s>>>((const uint8_t*) w, (const block_q8_1*) x, y, n_in, n_out,
                                                   row_bytes);
}

template<int TY>
void launch_ty(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    switch (ncols) {
        case 2: launch_nc<TY, 2>(w, x, y, n_in, n_out, s); break;
        case 3: launch_nc<TY, 3>(w, x, y, n_in, n_out, s); break;
        case 4: launch_nc<TY, 4>(w, x, y, n_in, n_out, s); break;
        case 5: launch_nc<TY, 5>(w, x, y, n_in, n_out, s); break;
        case 6: launch_nc<TY, 6>(w, x, y, n_in, n_out, s); break;
        case 7: launch_nc<TY, 7>(w, x, y, n_in, n_out, s); break;
        default: launch_nc<TY, 8>(w, x, y, n_in, n_out, s); break;
    }
}

bool covered(int ty) { return ty == 14 || ty == 12 || ty == 13 || ty == 23 || ty == 20 || ty == 8; }

}  // namespace

void pascal_dense_fp16_override(int on) { g_override = on; }

bool pascal_dense_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                       void* stream) {
    if (ncols < 2 || ncols > 8 || !covered(ggml_type)) return false;
    if (n_in <= 0 || n_in % 256 != 0 || n_out <= 0) return false;
    if (!enabled_here()) return false;
    cudaStream_t s = (cudaStream_t) stream;
    switch (ggml_type) {
        case 8: launch_ty<8>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 12: launch_ty<12>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 13: launch_ty<13>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 14: launch_ty<14>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
        case 20: launch_ty<20>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
        default: launch_ty<23>(w, x_q8_1, y, n_in, n_out, ncols, s); break;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "pascal_dense_mmvq: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::kernels
#endif  // !__HIPCC__
