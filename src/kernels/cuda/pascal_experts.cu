// src/kernels/cuda/pascal_experts.cu - see include/strata/kernels/pascal_experts.hpp.
//
// Block layouts and codebooks are llama.cpp's (ggml-common.h, via iq_kernels.cu's include); each decoder below
// follows the matching vec_dot_*_q8_1 in iq_kernels.cu index for index, with the int8 dp4a replaced by half2 FMA.
//
// Two kernels per call:
//   pgu_kernel<TG>   gate + up + SwiGLU.  A warp owns row pair r (gate row r on lanes 0-15, up row r on lanes 16-31),
//                    a block 8 row pairs, grid.x = 80.  Writes h[entry][r] = silu(gate . x) * (up . x).
//   pdown_kernel<TD> down.  4 lanes per output row (5 of its 20 32-value items each), 8 rows per warp, 64 per block,
//                    grid.x = 40.
// Token tiling: a group is one expert and its entries are the verify window's tokens routed to it (1..8).  Every
// 32-value weight sub-block is loaded and decoded to half2 ONCE and multiplied against all entries of the group (in
// chunks of up to NE_MAX = 8 entries; the chunk size is a template parameter, so a 1-entry group runs 1-entry code).
// The entries' activations are staged in shared memory a K-slice at a time - gate/up: 512 values (16 sub-blocks, one
// per half-warp lane) of every entry of the chunk, 1 KB each, 5 slices per row; down: all 640, 1.25 KB each - in an
// [entry][quarter][sub-block] uint4 layout, so the lanes of a warp read consecutive 16-byte words (no bank conflicts;
// the two half-warps / the 8 row groups read the same words: broadcast).  The weights of the next slice / item are
// loaded before the barriers (gate/up) or before the current item's FMAs (down).
// Grids: grid.y is the number of group rows that fills the SMs once (occupancy API); blocks loop over the groups
// g = blockIdx.y + k * gridDim.y < *n_groups (read on the device), so an empty call is one short wave of blocks.
//
// Numerics.  Weights decode exactly to fp16 (codebook integers; the per-32 / per-16 scales stay in fp32).
//   gate/up: x = q/128 exactly (q8_1, |q| <= 127, so |x| < 1).  A half2 lane accumulates 16 products (IQ2_S: 8 per
//     scale part) in fp16: |partial| <= 16 * 62 = 992 (IQ3_XXS grid <= 62, IQ3_S <= 15, IQ2_S <= 43), far inside
//     fp16's 65504; rounding <= 2^-11 relative per step.  Then fp32: acc += (d * 128) * (w_scale * (lo + hi)).
//   down: h / s with s the power of two >= the 32-block's max |h| (exact division), rounded to fp16 (|x| <= 1): a
//     half lane's 16 products are <= 16 * 127 = 2032 (IQ4_NL; Q2_0 <= 32).  Then fp32: acc += s * (d * (lo + hi)).
//   Every entry goes through the same operations in the same order whatever the chunk size or grouping (explicit
//   roundings, fixed shuffle trees), so results per entry are independent of grouping, and deterministic.
#include "strata/kernels/pascal_experts.hpp"

#if defined(__HIPCC__)
// AMD has dot-product instructions and no Pascal: the int8 path in iq_kernels.cu stays.
namespace strata::kernels {
bool pascal_expert_grouped(const NativeExpertLayout&, const unsigned long long*, const int32_t*, const int32_t*,
                           const int32_t*, const int32_t*, int64_t, int64_t, const void*, void*, float*, void*) {
    return false;
}
void pascal_expert_fp16_override(int) {}
}  // namespace strata::kernels
#else

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int H = 2560;          // n_embd
constexpr int FF = 640;          // expert n_ff
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int NE_MAX = 8;        // entries per chunk: a verify window's tokens (a group with more runs in chunks)

// gate/up: lanes 0-15 of a warp take gate row r, lanes 16-31 up row r; per staged K-slice each lane owns one
// 32-value sub-block of its row
constexpr int SLICE = 512;                     // K values per staging round
constexpr int SLICE_SB = SLICE / 32;           // 16 sub-blocks: one per half-warp lane
constexpr int N_SLICE = H / SLICE;             // 5
constexpr int GU_RPW = 1;                      // row pairs (gate r, up r) per warp (8 fp32 accumulators per lane)
constexpr int GU_RP = WARPS * GU_RPW;          // 8 row pairs per block
// down: 4 lanes per output row, 5 of the row's 20 items (32 values) each; 8 rows per warp
constexpr int D_ITEMS = FF / 32;               // 20
constexpr int D_LPR = 4;
constexpr int D_IPL = D_ITEMS / D_LPR;         // 5 items per lane
constexpr int D_ROWS = WARPS * (32 / D_LPR);   // 64 rows per block
static_assert(H % SLICE == 0 && FF % GU_RP == 0 && D_ITEMS % D_LPR == 0 && H % D_ROWS == 0, "tiling");

constexpr int B_IQ3XXS = 98, B_IQ3S = 110, B_IQ2S = 82, B_IQ4NL = 18, B_Q20 = 18;   // bytes per block
static_assert(sizeof(block_iq3_xxs) == B_IQ3XXS && sizeof(block_iq3_s) == B_IQ3S && sizeof(block_iq2_s) == B_IQ2S &&
              sizeof(block_iq4_nl) == B_IQ4NL && sizeof(block_q2_0) == B_Q20, "block sizes");

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ half2 as_h2(uint32_t u) { half2 h; memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t as_u(half2 h) { uint32_t u; memcpy(&u, &h, 4); return u; }
// the blocks are 2-byte aligned (odd sizes / fp16 first), so words are read as two halves, like get_int_b2;
// through the read-only cache (the weight pointers come from a device array, so the compiler cannot tell)
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return __ldg((const unsigned short*) p); }
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ float ldh(const uint8_t* p) { return __half2float(__ushort_as_half((unsigned short) ld16(p))); }
// four UNSIGNED bytes -> (b0,b1), (b2,b3) as exact halves: 0x64XX is 1024 + XX
__device__ __forceinline__ void u8x4_h2(uint32_t g, half2& lo, half2& hi) {
    const half2 k = as_h2(0x64006400u);
    lo = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4140)), k);
    hi = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4342)), k);
}
__device__ __forceinline__ half2 sgn(half2 w, uint32_t m) { return as_h2(as_u(w) ^ m); }
// the two halves of an fp16 accumulator, summed in fp32 (the conversions are exact: HADD2.F32 on sm_60)
__device__ __forceinline__ float h2sum(half2 a) { return __fadd_rn(__low2float(a), __high2float(a)); }

// ------------------------------------------------------------------------------------------------ shared memory
// s_sgn[s]: the four half2 sign masks for 8 values whose sign bits are s (bit j negates value j)
__shared__ uint4 s_sgn[256];
__shared__ uint2 s_g3xxs[256];              // IQ3 grids pre-converted: four halves per entry
__shared__ uint2 s_g3s[512];
__shared__ uint2 s_g2s[1024];
__shared__ uint2 s_q2[256];                 // Q2_0: code byte -> four values (code - 1) as two half2
__shared__ uint32_t s_nl[256];              // IQ4_NL: code byte -> (kvalues[lo nibble], kvalues[hi nibble]) half2
// gate/up: one K-slice of every entry of the chunk as q/128 in fp16, [entry][quarter l][sub-block] (8 KB), and the
// per-32 scale d * 128
__shared__ __align__(16) uint4 s_xs[NE_MAX * 4 * SLICE_SB];
__shared__ float s_xsd[NE_MAX * SLICE_SB];
// down: every entry's SwiGLU output / its per-32 power-of-two scale, [entry][quarter k][item] (10 KB), and the scale
__shared__ __align__(16) uint4 s_hs[NE_MAX * 4 * D_ITEMS];
__shared__ float s_hsd[NE_MAX * D_ITEMS];

__device__ __forceinline__ void fill_sgn(int t) {
    for (int s = t; s < 256; s += THREADS) {
        uint32_t m[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) m[k] = (((s >> (2 * k)) & 1u) << 15) | (((s >> (2 * k + 1)) & 1u) << 31);
        s_sgn[s] = make_uint4(m[0], m[1], m[2], m[3]);
    }
}

template<int TG>
__device__ __forceinline__ void fill_gu_tables(int t) {
    fill_sgn(t);
    if constexpr (TG == 18 || TG == 21) {
        const int n = TG == 18 ? 256 : 512;
        for (int i = t; i < n; i += THREADS) {
            half2 lo, hi;
            u8x4_h2(TG == 18 ? iq3xxs_grid[i] : iq3s_grid[i], lo, hi);
            (TG == 18 ? s_g3xxs : s_g3s)[i] = make_uint2(as_u(lo), as_u(hi));
        }
    } else {
        for (int i = t; i < 1024; i += THREADS) {
            const uint64_t v = iq2s_grid[i];
            s_g2s[i] = make_uint2((uint32_t) v, (uint32_t) (v >> 32));
        }
    }
}

template<int TD>
__device__ __forceinline__ void fill_down_tables(int t) {
    if constexpr (TD == 20) {
        for (int v = t; v < 256; v += THREADS)
            s_nl[v] = as_u(__halves2half2(__int2half_rn(kvalues_iq4nl[v & 15]), __int2half_rn(kvalues_iq4nl[v >> 4])));
    } else {
        for (int v = t; v < 256; v += THREADS) {
            half vals[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) vals[j] = __int2half_rn((int) ((v >> (2 * j)) & 3) - 1);
            s_q2[v] = make_uint2(as_u(__halves2half2(vals[0], vals[1])), as_u(__halves2half2(vals[2], vals[3])));
        }
    }
}

// ------------------------------------------------------------------------------------------------ gate/up decoders
// GuW<TG>: one 32-value sub-block of a row: sub-block ib (0..7) of the 256-value block at b.  load() reads its raw
// words (global, 2-byte aligned); quarter(l) decodes values 8l..8l+7 to four sign-applied half2; scale(k) is the fp32
// scale of the k-th of NS equal parts.
template<int TG> struct GuW;

template<> struct GuW<18> {   // IQ3_XXS (vec_dot_iq3_xxs)
    static constexpr int NS = 1, BYTES = B_IQ3XXS;
    float d;
    uint32_t q0, q1, aux;
    __device__ __forceinline__ void load(const uint8_t* b, int ib) {
        const uint8_t *p8 = b + 8 * ib, *p4 = b + 4 * ib;   // one address per stride, fields at immediates
        d = ldh(b);
        q0 = ld32(p8 + 2);
        q1 = ld32(p8 + 6);
        aux = ld32(p4 + 66);
    }
    __device__ __forceinline__ void quarter(int l, half2 (&w)[4]) const {
        const uint32_t qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const uint32_t s7 = (aux >> (7 * l)) & 127u;
        const uint4 m = s_sgn[s7 | ((__popc(s7) & 1u) << 7)];
        const uint2 ga = s_g3xxs[(qq >> sh) & 0xFF], gb = s_g3xxs[(qq >> (sh + 8)) & 0xFF];
        w[0] = sgn(as_h2(ga.x), m.x); w[1] = sgn(as_h2(ga.y), m.y);
        w[2] = sgn(as_h2(gb.x), m.z); w[3] = sgn(as_h2(gb.y), m.w);
    }
    __device__ __forceinline__ float scale(int) const { return d * (0.5f + (float) (aux >> 28)) * 0.5f; }
};

template<> struct GuW<21> {   // IQ3_S (vec_dot_iq3_s)
    static constexpr int NS = 1, BYTES = B_IQ3S;
    float d;
    uint32_t q0, q1, qh, sg, sc;
    __device__ __forceinline__ void load(const uint8_t* b, int ib) {
        const uint8_t *p8 = b + 8 * ib, *p4 = b + 4 * ib, *p1 = b + ib;
        d = ldh(b);
        q0 = ld32(p8 + 2);
        q1 = ld32(p8 + 6);
        qh = __ldg(p1 + 66);
        sg = ld32(p4 + 74);
        sc = (__ldg(b + 2 + 64 + 8 + 32 + (ib >> 1)) >> (4 * (ib & 1))) & 0xF;
    }
    __device__ __forceinline__ void quarter(int l, half2 (&w)[4]) const {
        const uint32_t qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const uint32_t i0 = ((qq >> sh) & 0xFF) | ((qh << (8 - 2 * l)) & 0x100);
        const uint32_t i1 = ((qq >> (sh + 8)) & 0xFF) | ((qh << (7 - 2 * l)) & 0x100);
        const uint4 m = s_sgn[(sg >> (8 * l)) & 0xFF];
        const uint2 ga = s_g3s[i0], gb = s_g3s[i1];
        w[0] = sgn(as_h2(ga.x), m.x); w[1] = sgn(as_h2(ga.y), m.y);
        w[2] = sgn(as_h2(gb.x), m.z); w[3] = sgn(as_h2(gb.y), m.w);
    }
    __device__ __forceinline__ float scale(int) const { return d * (float) (1 + 2 * sc); }
};

template<> struct GuW<22> {   // IQ2_S (vec_dot_iq2_s): two 16-value halves with their own 4-bit scales
    static constexpr int NS = 2, BYTES = B_IQ2S;
    float d;
    uint32_t gi, sg, qh, sc;
    __device__ __forceinline__ void load(const uint8_t* b, int ib) {
        const uint8_t *p4 = b + 4 * ib, *p1 = b + ib;
        d = ldh(b);
        gi = ld32(p4 + 2);
        sg = ld32(p4 + 34);
        qh = __ldg(p1 + 66);
        sc = __ldg(p1 + 74);
    }
    __device__ __forceinline__ void quarter(int l, half2 (&w)[4]) const {
        const uint32_t idx = ((gi >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
        const uint2 gr = s_g2s[idx];
        const uint4 m = s_sgn[(sg >> (8 * l)) & 0xFF];
        half2 w0, w1, w2, w3;
        u8x4_h2(gr.x, w0, w1);
        u8x4_h2(gr.y, w2, w3);
        w[0] = sgn(w0, m.x); w[1] = sgn(w1, m.y); w[2] = sgn(w2, m.z); w[3] = sgn(w3, m.w);
    }
    __device__ __forceinline__ float scale(int k) const { return d * 0.25f * (0.5f + (float) ((sc >> (4 * k)) & 0xF)); }
};

// One decoded sub-block against NE staged entries: acc[j] += x_scale[j] * w_scale * sum(w * x_j).  xs / xd point at
// the lane's sub-block in s_xs / s_xsd.  Each entry's operations are the same whatever NE is (explicit roundings,
// no contraction freedom), so an entry's result does not depend on the grouping.
//
// Bounds: x = q/128 with |q| <= 127, so |x| < 1; the codebook magnitudes are <= 62 (IQ3_XXS), 15 (IQ3_S), 43 (IQ2_S).
// A half lane accumulates 16 products (IQ2_S: 8 per part), so |partial| <= 16 * 62 = 992, far below fp16's 65504;
// the fp16 rounding error of the chain is <= 16 * 2^-11 * 992 < 8 absolute, ~2^-11 relative per step.
template<int TG, int NE>
__device__ __forceinline__ void gu_dot(const GuW<TG>& w, const uint4* __restrict__ xs, const float* __restrict__ xd,
                                       float (&acc)[NE]) {
    constexpr int NS = GuW<TG>::NS;
    half2 a[NE][NS];
#pragma unroll
    for (int j = 0; j < NE; ++j)
#pragma unroll
        for (int k = 0; k < NS; ++k) a[j][k] = __float2half2_rn(0.f);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        half2 q[4];
        w.quarter(l, q);
        const int k = NS == 1 ? 0 : l >> 1;
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const uint4 x = xs[(j * 4 + l) * SLICE_SB];
            a[j][k] = __hfma2(q[0], as_h2(x.x), a[j][k]);
            a[j][k] = __hfma2(q[1], as_h2(x.y), a[j][k]);
            a[j][k] = __hfma2(q[2], as_h2(x.z), a[j][k]);
            a[j][k] = __hfma2(q[3], as_h2(x.w), a[j][k]);
        }
    }
    float ws[NS];
#pragma unroll
    for (int k = 0; k < NS; ++k) ws[k] = w.scale(k);
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        float v = __fmul_rn(ws[0], h2sum(a[j][0]));
        if constexpr (NS == 2) v = __fmaf_rn(ws[1], h2sum(a[j][1]), v);
        acc[j] = __fmaf_rn(xd[j * SLICE_SB], v, acc[j]);
    }
}

// ------------------------------------------------------------------------------------------------ down decoders
// DnW<TD>: 32 values of a down row (one IQ4_NL block, half a Q2_0 block): item it lives in the block at(row, it),
// and a lane's next item (it + D_LPR) STEP bytes further.  quarter(k) = values of uint4 k of the staged activation.
template<int TD> struct DnW;

template<> struct DnW<20> {   // IQ4_NL
    // byte j of the block holds values j (low nibble) and j + 16 (high nibble); the staged activation of an IQ4_NL
    // block is interleaved to match - half2 j = (x[j], x[j + 16]) - so each byte is one table load and one HFMA2
    static constexpr int STEP = D_LPR * B_IQ4NL;   // a lane's items are D_LPR blocks apart
    float d;
    uint32_t w[4];
    static __device__ __forceinline__ const uint8_t* at(const uint8_t* row, int it) { return row + it * B_IQ4NL; }
    __device__ __forceinline__ void load(const uint8_t* b, int) {
        d = ldh(b);
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = ld32(b + 2 + 4 * k);
    }
    __device__ __forceinline__ void quarter(int k, half2 (&q)[4]) const {
        q[0] = as_h2(s_nl[w[k] & 0xFF]);
        q[1] = as_h2(s_nl[(w[k] >> 8) & 0xFF]);
        q[2] = as_h2(s_nl[(w[k] >> 16) & 0xFF]);
        q[3] = as_h2(s_nl[w[k] >> 24]);
    }
};

template<> struct DnW<42> {   // Q2_0: value = code - 1, 16 codes per word, LSB first
    static constexpr int STEP = D_LPR / 2 * B_Q20;
    float d;
    uint32_t w[2];
    static __device__ __forceinline__ const uint8_t* at(const uint8_t* row, int it) { return row + (it >> 1) * B_Q20; }
    __device__ __forceinline__ void load(const uint8_t* b, int it) {   // it: the item (its parity picks the half)
        d = ldh(b);
        const uint8_t* qs = b + 2 + 8 * (it & 1);
        w[0] = ld32(qs);
        w[1] = ld32(qs + 4);
    }
    __device__ __forceinline__ void quarter(int k, half2 (&q)[4]) const {
        const uint32_t v = w[k >> 1] >> (16 * (k & 1));
        const uint2 w0 = s_q2[v & 0xFF], w1 = s_q2[(v >> 8) & 0xFF];
        q[0] = as_h2(w0.x); q[1] = as_h2(w0.y); q[2] = as_h2(w1.x); q[3] = as_h2(w1.y);
    }
};

// Bounds: the staged h is h / s with s the power of two >= the 32-block's max |h|, so |x| <= 1; IQ4_NL magnitudes
// are <= 127 (Q2_0: 2), so a half lane's 16 products stay <= 2032 < 65504.
template<int TD, int NE>
__device__ __forceinline__ void down_dot(const DnW<TD>& w, const uint4* __restrict__ xs, const float* __restrict__ xd,
                                         float (&acc)[NE]) {
    half2 a[NE];
#pragma unroll
    for (int j = 0; j < NE; ++j) a[j] = __float2half2_rn(0.f);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        half2 q[4];
        w.quarter(k, q);
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const uint4 x = xs[(j * 4 + k) * D_ITEMS];
            a[j] = __hfma2(q[0], as_h2(x.x), a[j]);
            a[j] = __hfma2(q[1], as_h2(x.y), a[j]);
            a[j] = __hfma2(q[2], as_h2(x.z), a[j]);
            a[j] = __hfma2(q[3], as_h2(x.w), a[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < NE; ++j) acc[j] = __fmaf_rn(xd[j * D_ITEMS], __fmul_rn(w.d, h2sum(a[j])), acc[j]);
}

// ------------------------------------------------------------------------------------------------ chunks
// NE entries (tokens tok[0..NE)) of one group through gate and up rows r0 .. r0+GU_RPW-1, fused with SwiGLU:
// hc[j * FF + r] = silu(gate_r . x_j) * (up_r . x_j).  Block-uniform: every thread calls it with the same NE, and its
// barriers are unconditional.
template<int TG, int NE>
__device__ __forceinline__ void gu_chunk(const uint8_t* __restrict__ mat, size_t gu_row, int r0,
                                         const block_q8_1* __restrict__ xq, const int32_t* __restrict__ tok,
                                         float* __restrict__ hc, int t, int lane) {
    static_assert(GU_RPW == 1 && SLICE_SB == 16 && THREADS == 256, "the lane / staging maps below");
    using W = GuW<TG>;
    // staging map: thread t stages word v = t & 127 (sub-block b = v >> 3, word q = v & 7) of entries
    // jg, jg + 2, ... (jg = t >> 7): source offsets in 32-bit words of xq (a block_q8_1 is 9 words: ds, then qs)
    constexpr int KW = (NE + 1) / 2;
    const int jg = t >> 7, v = t & 127, b = v >> 3, q = v & 7;
    uint32_t src[KW];
#pragma unroll
    for (int k = 0; k < KW; ++k)
        src[k] = (jg + 2 * k < NE) ? ((uint32_t) __ldg(tok + jg + 2 * k) * (H / 32) + b) * 9u : 0u;
    uint2* const sdst = reinterpret_cast<uint2*>(s_xs) + jg * 128 + ((q >> 1) * SLICE_SB + b) * 2 + (q & 1);
    float* const sdd = s_xsd + jg * SLICE_SB + b;
    const uint32_t* const xw = reinterpret_cast<const uint32_t*>(xq);
    // compute map: lanes 0-15 gate row r0, 16-31 up row r0 (mat already points at the lane's matrix); lane sl owns
    // sub-block s * 16 + sl of slice s, i.e. sub-block sl & 7 of the row's block 2s + (sl >> 3)
    const int sl = lane & 15, ib = sl & 7;
    const uint8_t* wb = mat + (size_t) r0 * gu_row + (sl >> 3) * W::BYTES;
    W w;
    w.load(wb, ib);   // slice 0: in flight across the first barriers
    float acc[NE];
#pragma unroll
    for (int j = 0; j < NE; ++j) acc[j] = 0.0f;
#pragma unroll 1
    for (int s = 0; s < N_SLICE; ++s) {
        __syncthreads();   // tables filled / the previous slice consumed
#pragma unroll
        for (int k = 0; k < KW; ++k) {
            if (jg + 2 * k < NE) {   // odd NE: the last entry has only threads 0-127
                const uint32_t o = src[k] + (uint32_t) (s * SLICE_SB * 9);
                // q/128 exactly: byte_perm makes 1024 + (b + 128) (b the int8, sign flipped); * 1/128 - 9 = b/128
                const uint32_t g = __ldg(xw + o + 1 + q) ^ 0x80808080u;
                const half2 k128 = as_h2(0x20002000u), m9 = as_h2(0xC880C880u);   // 1/128, -9
                sdst[256 * k] = make_uint2(as_u(__hfma2(as_h2(__byte_perm(g, 0x64646464u, 0x4140)), k128, m9)),
                                           as_u(__hfma2(as_h2(__byte_perm(g, 0x64646464u, 0x4342)), k128, m9)));
                if (q == 0) sdd[2 * SLICE_SB * k] = ldh(reinterpret_cast<const uint8_t*>(xw + o)) * 128.0f;
            }
        }
        __syncthreads();
        gu_dot<TG, NE>(w, s_xs + sl, s_xsd + sl, acc);
        wb += (SLICE_SB / 8) * W::BYTES;
        if (s + 1 < N_SLICE) w.load(wb, ib);   // the next slice's words: in flight across the next barriers
    }
    // each half-warp sums its 16 lanes (fixed butterfly order); lane j < NE of the gate half takes entry j's gate
    // and up sums and writes its SwiGLU output
    float g = 0.0f, u = 0.0f;
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        float v = acc[j];
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        const float vu = __shfl_xor_sync(0xffffffffu, v, 16);
        if (lane == j) { g = v; u = vu; }
    }
    if (lane < NE) hc[(size_t) lane * FF + r0] = (g / (1.0f + __expf(-g))) * u;
}

template<int TD, int NE>
__device__ __forceinline__ void down_chunk(const uint8_t* __restrict__ mat, size_t d_row, int r,
                                           const float* __restrict__ hc, const int32_t* __restrict__ dst,
                                           float* __restrict__ out, int t, int lane) {
    using W = DnW<TD>;
    const int q = lane & (D_LPR - 1);   // items q, q + 4, ..., q + 16 of row r
    const uint8_t* wb = W::at(mat + (size_t) r * d_row, q);
    W w;
    w.load(wb, q);      // the first item: in flight across the barriers
    __syncthreads();    // tables filled / the previous chunk consumed
    // stage: whole warps (FF % 32 == 0), a warp covers one aligned 32-block of one entry
    const int it_s = t >> 5, jj = t & 31;
    // IQ4_NL: value jj of a 32-block goes to half 2jj for jj < 16, 2(jj-16) + 1 otherwise (see DnW<20>)
    const int slot = TD == 20 ? (jj < 16 ? 2 * jj : 2 * (jj - 16) + 1) : jj;
    half* const sdst = reinterpret_cast<half*>(s_hs) + ((slot >> 3) * D_ITEMS + it_s) * 8 + (slot & 7);
#pragma unroll
    for (int j = 0; j < NE; ++j) {
#pragma unroll
        for (int k = 0; k * THREADS < FF; ++k) {
            if (k * THREADS + t < FF) {   // warp-uniform
                const float v = hc[j * FF + k * THREADS + t];
                float amax = fabsf(v);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                // the next power of two >= amax: dividing by it is exact, and |h / s| <= 1 keeps fp16 sums in range
                const float sc = amax > 1e-30f ? __int_as_float((__float_as_int(amax) + 0x007FFFFF) & 0x7F800000) : 1.0f;
                sdst[(j * 4 * D_ITEMS + k * (THREADS / 32)) * 8] = __float2half_rn(v * (1.0f / sc));
                if (jj == 0) s_hsd[j * D_ITEMS + k * (THREADS / 32) + it_s] = sc;
            }
        }
    }
    __syncthreads();
    float acc[NE];
#pragma unroll
    for (int j = 0; j < NE; ++j) acc[j] = 0.0f;
#pragma unroll 1
    for (int m = 0; m < D_IPL; ++m) {   // the next item's words are loaded while this one is multiplied
        const int it = q + D_LPR * m;
        const W cur = w;
        wb += W::STEP;
        if (m + 1 < D_IPL) w.load(wb, it + D_LPR);
        down_dot<TD, NE>(cur, s_hs + it, s_hsd + it, acc);
    }
#pragma unroll
    for (int j = 0; j < NE; ++j) {
        float v = acc[j];
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
        if (q == (j & (D_LPR - 1))) out[(size_t) __ldg(dst + j) * H + r] = v;
    }
}

// ------------------------------------------------------------------------------------------------ kernels
// Groups g = blockIdx.y, blockIdx.y + gridDim.y, ... < *n_groups; every block-level branch depends only on the
// group / chunk, so the barriers are block-uniform.
template<int TG>
__global__ void __launch_bounds__(THREADS, 4) pgu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_tok,
                                                         const block_q8_1* __restrict__ xq, size_t up_off,
                                                         size_t gu_row, float* __restrict__ h) {
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    fill_gu_tables<TG>(t);   // the chunk's first barrier publishes them
    const int r0 = blockIdx.x * GU_RP + warp * GU_RPW;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* mat = (const uint8_t*) grp_ptr[g] + (lane >= 16 ? up_off : 0);   // gate | up half-warp
        const int e1 = grp_start[g + 1];
        for (int c = grp_start[g]; c < e1; c += NE_MAX) {
            const int32_t* tk = ent_tok + c;
            float* d = h + (size_t) c * FF;
            switch (min(NE_MAX, e1 - c)) {
                case 1: gu_chunk<TG, 1>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 2: gu_chunk<TG, 2>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 3: gu_chunk<TG, 3>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 4: gu_chunk<TG, 4>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 5: gu_chunk<TG, 5>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 6: gu_chunk<TG, 6>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                case 7: gu_chunk<TG, 7>(mat, gu_row, r0, xq, tk, d, t, lane); break;
                default: gu_chunk<TG, 8>(mat, gu_row, r0, xq, tk, d, t, lane); break;
            }
        }
    }
}

template<int TD>
__global__ void __launch_bounds__(THREADS, 4) pdown_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_dst,
                                                           const float* __restrict__ h, size_t down_off,
                                                           size_t d_row, float* __restrict__ out) {
    const int ng = *n_groups;
    if ((int) blockIdx.y >= ng) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    fill_down_tables<TD>(t);   // the chunk's first barrier publishes them
    const int r = blockIdx.x * D_ROWS + warp * (32 / D_LPR) + lane / D_LPR;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* mat = (const uint8_t*) grp_ptr[g] + down_off;
        const int e1 = grp_start[g + 1];
        for (int c = grp_start[g]; c < e1; c += NE_MAX) {
            const float* hc = h + (size_t) c * FF;
            const int32_t* ds = ent_dst + c;
            switch (min(NE_MAX, e1 - c)) {
                case 1: down_chunk<TD, 1>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 2: down_chunk<TD, 2>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 3: down_chunk<TD, 3>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 4: down_chunk<TD, 4>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 5: down_chunk<TD, 5>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 6: down_chunk<TD, 6>(mat, d_row, r, hc, ds, out, t, lane); break;
                case 7: down_chunk<TD, 7>(mat, d_row, r, hc, ds, out, t, lane); break;
                default: down_chunk<TD, 8>(mat, d_row, r, hc, ds, out, t, lane); break;
            }
        }
    }
}

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

// Grid rows (groups in flight) for a kernel whose grid has gx columns: as many as keep every SM filled with the
// blocks it can hold at once (one wave; a block loops over groups g = blockIdx.y + k * gridDim.y), so an empty or
// small call launches that one wave instead of gx * cap_groups mostly-empty blocks.  Per device, cached.
template<class Kernel>
int wave_rows(Kernel kernel, int slot, int gx) {
    static int cache[64][5] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 1; }
    int& c = cache[dev][slot];
    if (c == 0) {
        int sms = 0, per_sm = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, 0) != cudaSuccess) {
            cudaGetLastError();
            per_sm = 1;
        }
        c = sms * per_sm / gx > 1 ? sms * per_sm / gx : 1;
    }
    return c;
}

template<int TG>
void launch_gu(int64_t cap_groups, cudaStream_t s, const unsigned long long* gp, const int32_t* gs, const int32_t* ng,
               const int32_t* tok, const block_q8_1* x, const NativeExpertLayout& L, float* h) {
    constexpr int gx = FF / GU_RP;
    const int gy = wave_rows(pgu_kernel<TG>, TG == 18 ? 0 : TG == 21 ? 1 : 2, gx);
    const dim3 grid((unsigned) gx, (unsigned) (cap_groups < gy ? cap_groups : gy));
    pgu_kernel<TG><<<grid, THREADS, 0, s>>>(gp, gs, ng, tok, x, L.up_off, L.gu_row, h);
}
template<int TD>
void launch_down(int64_t cap_groups, cudaStream_t s, const unsigned long long* gp, const int32_t* gs,
                 const int32_t* ng, const int32_t* dst, const float* h, const NativeExpertLayout& L, float* out) {
    constexpr int gx = H / D_ROWS;
    const int gy = wave_rows(pdown_kernel<TD>, TD == 20 ? 3 : 4, gx);
    const dim3 grid((unsigned) gx, (unsigned) (cap_groups < gy ? cap_groups : gy));
    pdown_kernel<TD><<<grid, THREADS, 0, s>>>(gp, gs, ng, dst, h, L.down_off, L.d_row, out);
}

}  // namespace

void pascal_expert_fp16_override(int on) { g_override = on; }

bool pascal_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (L.n_embd != H || L.n_ff != FF) return false;
    if (L.gu_type != 18 && L.gu_type != 21 && L.gu_type != 22) return false;
    if (L.d_type != 20 && L.d_type != 42) return false;
    if (!enabled_here()) return false;
    if (cap_groups <= 0 || cap_entries <= 0) return true;
    cudaStream_t s = (cudaStream_t) stream;
    // native_expert_grouped's scratch: gate | up | h | hq; only h is used here (gate/up are fused into SwiGLU)
    const size_t f = (size_t) cap_entries * FF * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    // both kernels: one wave of blocks (the grid rows loop over the device-side group count), whatever the caps
    switch (L.gu_type) {
        case 18: launch_gu<18>(cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, h); break;
        case 21: launch_gu<21>(cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, h); break;
        default: launch_gu<22>(cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, h); break;
    }
    if (L.d_type == 20) launch_down<20>(cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, h, L, out);
    else launch_down<42>(cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, h, L, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "pascal_expert_grouped: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::kernels
#endif  // !__HIPCC__
