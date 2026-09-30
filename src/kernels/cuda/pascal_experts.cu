// src/kernels/cuda/pascal_experts.cu - see include/strata/kernels/pascal_experts.hpp.
//
// Block layouts and codebooks are llama.cpp's (ggml-common.h, via iq_kernels.cu's include); each decoder below
// follows the matching vec_dot_*_q8_1 in iq_kernels.cu index for index, with the int8 dp4a replaced by half2 FMA.
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
constexpr int GU_ROWS = 32;      // rows per block (4 per warp): the staged activation serves 32 rows
constexpr int D_ROWS = 32;

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
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// four UNSIGNED bytes -> (b0,b1), (b2,b3) as exact halves: 0x64XX is 1024 + XX
__device__ __forceinline__ void u8x4_h2(uint32_t g, half2& lo, half2& hi) {
    const half2 k = as_h2(0x64006400u);
    lo = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4140)), k);
    hi = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4342)), k);
}
// four SIGNED bytes: bias by 128 first, then 1024 + 128 + b - 1152
__device__ __forceinline__ void s8x4_h2(uint32_t g, half2& lo, half2& hi) {
    const half2 k = as_h2(0x64806480u);   // 1152
    g ^= 0x80808080u;
    lo = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4140)), k);
    hi = __hsub2(as_h2(__byte_perm(g, 0x64646464u, 0x4342)), k);
}
__device__ __forceinline__ half2 sgn(half2 w, uint32_t m) { return as_h2(as_u(w) ^ m); }
__device__ __forceinline__ float h2sum(half2 a) { return __low2float(a) + __high2float(a); }

// ------------------------------------------------------------------------------------------------ shared tables
// s_sgn[s]: the four half2 sign masks for 8 values whose sign bits are s (bit j negates value j)
__shared__ uint4 s_sgn[256];
__shared__ uint2 s_g3xxs[256];              // IQ3 grids pre-converted: four halves per entry
__shared__ uint2 s_g3s[512];
__shared__ uint2 s_g2s[1024];
__shared__ uint2 s_q2[256];                 // Q2_0: code byte -> four values (code - 1) as two half2
__shared__ uint32_t s_nl[256];              // IQ4_NL: code byte -> (kvalues[lo nibble], kvalues[hi nibble]) half2
__shared__ __align__(16) half2 s_x[H / 2];  // the current entry's activation, fp16
__shared__ float s_xd[H / 32];              // its per-32 scale
__shared__ __align__(16) half2 s_h[FF / 2];
__shared__ float s_hd[FF / 32];

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

// ------------------------------------------------------------------------------------------------ gate/up decoders
// One 32-value sub-block sb (of the row's H/32) against the staged activation: sum of w * x without the activation's
// scale.  x values are q/128, |x| < 1; the codebook magnitudes are <= 62, so a half lane's 16 products stay < 1000.
template<int TG> __device__ __forceinline__ float gu_sub(const uint8_t* row, int sb);

template<> __device__ __forceinline__ float gu_sub<18>(const uint8_t* row, int sb) {   // IQ3_XXS (vec_dot_iq3_xxs)
    const uint8_t* b = row + (unsigned) (sb >> 3) * B_IQ3XXS;
    const int ib = sb & 7;
    const float d = ldh(b);
    const uint32_t q0 = ld32(b + 2 + 8 * ib), q1 = ld32(b + 2 + 8 * ib + 4);
    const uint32_t aux = ld32(b + 2 + 64 + 4 * ib);
    const uint4* xv = reinterpret_cast<const uint4*>(s_x + sb * 16);
    half2 acc = __float2half2_rn(0.f);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const uint32_t s7 = (aux >> (7 * l)) & 127u;
        const uint4 m = s_sgn[s7 | ((__popc(s7) & 1u) << 7)];
        const uint2 ga = s_g3xxs[(qq >> sh) & 0xFF], gb = s_g3xxs[(qq >> (sh + 8)) & 0xFF];
        const half2 w0 = as_h2(ga.x), w1 = as_h2(ga.y), w2 = as_h2(gb.x), w3 = as_h2(gb.y);
        const uint4 x = xv[l];
        acc = __hfma2(sgn(w0, m.x), as_h2(x.x), acc);
        acc = __hfma2(sgn(w1, m.y), as_h2(x.y), acc);
        acc = __hfma2(sgn(w2, m.z), as_h2(x.z), acc);
        acc = __hfma2(sgn(w3, m.w), as_h2(x.w), acc);
    }
    return d * (0.5f + (float) (aux >> 28)) * 0.5f * h2sum(acc);
}

template<> __device__ __forceinline__ float gu_sub<21>(const uint8_t* row, int sb) {   // IQ3_S (vec_dot_iq3_s)
    const uint8_t* b = row + (unsigned) (sb >> 3) * B_IQ3S;
    const int ib = sb & 7;
    const float d = ldh(b);
    const uint32_t q0 = ld32(b + 2 + 8 * ib), q1 = ld32(b + 2 + 8 * ib + 4);
    const uint32_t qh = __ldg(b + 2 + 64 + ib);
    const uint32_t sg = ld32(b + 2 + 64 + 8 + 4 * ib);
    const uint32_t sc = (__ldg(b + 2 + 64 + 8 + 32 + (ib >> 1)) >> (4 * (ib & 1))) & 0xF;
    const uint4* xv = reinterpret_cast<const uint4*>(s_x + sb * 16);
    half2 acc = __float2half2_rn(0.f);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const uint32_t i0 = ((qq >> sh) & 0xFF) | ((qh << (8 - 2 * l)) & 0x100);
        const uint32_t i1 = ((qq >> (sh + 8)) & 0xFF) | ((qh << (7 - 2 * l)) & 0x100);
        const uint4 m = s_sgn[(sg >> (8 * l)) & 0xFF];
        const uint2 ga = s_g3s[i0], gb = s_g3s[i1];
        const half2 w0 = as_h2(ga.x), w1 = as_h2(ga.y), w2 = as_h2(gb.x), w3 = as_h2(gb.y);
        const uint4 x = xv[l];
        acc = __hfma2(sgn(w0, m.x), as_h2(x.x), acc);
        acc = __hfma2(sgn(w1, m.y), as_h2(x.y), acc);
        acc = __hfma2(sgn(w2, m.z), as_h2(x.z), acc);
        acc = __hfma2(sgn(w3, m.w), as_h2(x.w), acc);
    }
    return d * (float) (1 + 2 * sc) * h2sum(acc);
}

template<> __device__ __forceinline__ float gu_sub<22>(const uint8_t* row, int sb) {   // IQ2_S (vec_dot_iq2_s)
    const uint8_t* b = row + (unsigned) (sb >> 3) * B_IQ2S;
    const int ib = sb & 7;
    const float d = ldh(b);
    const uint32_t gi = ld32(b + 2 + 4 * ib);
    const uint32_t sg = ld32(b + 2 + 32 + 4 * ib);
    const uint32_t qh = __ldg(b + 2 + 64 + ib);
    const uint32_t sc = __ldg(b + 2 + 64 + 8 + ib);
    const uint4* xv = reinterpret_cast<const uint4*>(s_x + sb * 16);
    half2 a0 = __float2half2_rn(0.f), a1 = a0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t idx = ((gi >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
        const uint2 gr = s_g2s[idx];
        const uint4 m = s_sgn[(sg >> (8 * l)) & 0xFF];
        half2 w0, w1, w2, w3;
        u8x4_h2(gr.x, w0, w1);
        u8x4_h2(gr.y, w2, w3);
        const uint4 x = xv[l];
        half2& a = l < 2 ? a0 : a1;
        a = __hfma2(sgn(w0, m.x), as_h2(x.x), a);
        a = __hfma2(sgn(w1, m.y), as_h2(x.y), a);
        a = __hfma2(sgn(w2, m.z), as_h2(x.z), a);
        a = __hfma2(sgn(w3, m.w), as_h2(x.w), a);
    }
    return d * 0.25f * ((0.5f + (float) (sc & 0xF)) * h2sum(a0) + (0.5f + (float) (sc >> 4)) * h2sum(a1));
}

// ------------------------------------------------------------------------------------------------ down decoders
// 32 values (one IQ4_NL block, half a Q2_0 block) against the staged SwiGLU output (|h/scale| <= 1): IQ4_NL
// magnitudes are <= 127, so a half lane's 16 products stay < 2048.
template<int TD> __device__ __forceinline__ float down_item(const uint8_t* row, int it);

template<> __device__ __forceinline__ float down_item<20>(const uint8_t* row, int it) {   // IQ4_NL
    // byte j of the block holds values j (low nibble) and j + 16 (high nibble); the staged activation of an IQ4_NL
    // block is interleaved to match - half2 j = (x[j], x[j + 16]) - so each byte is one table load and one HFMA2
    const uint8_t* b = row + (unsigned) it * B_IQ4NL;
    const float d = ldh(b);
    const uint4* xv = reinterpret_cast<const uint4*>(s_h + it * 16);
    half2 acc = __float2half2_rn(0.f);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t w = ld32(b + 2 + 4 * k);
        const uint4 x = xv[k];
        acc = __hfma2(as_h2(s_nl[w & 0xFF]), as_h2(x.x), acc);
        acc = __hfma2(as_h2(s_nl[(w >> 8) & 0xFF]), as_h2(x.y), acc);
        acc = __hfma2(as_h2(s_nl[(w >> 16) & 0xFF]), as_h2(x.z), acc);
        acc = __hfma2(as_h2(s_nl[w >> 24]), as_h2(x.w), acc);
    }
    return d * h2sum(acc);
}

template<> __device__ __forceinline__ float down_item<42>(const uint8_t* row, int it) {   // Q2_0: value = code - 1
    const uint8_t* b = row + (unsigned) (it >> 1) * B_Q20;
    const float d = ldh(b);
    const uint8_t* qs = b + 2 + 8 * (it & 1);
    const uint4* xv = reinterpret_cast<const uint4*>(s_h + it * 16);
    half2 acc = __float2half2_rn(0.f);
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const uint32_t w = ld32(qs + 4 * k);                                 // 16 codes, LSB first
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint2 w0 = s_q2[(w >> (16 * j)) & 0xFF], w1 = s_q2[(w >> (16 * j + 8)) & 0xFF];
            const uint4 x = xv[2 * k + j];
            acc = __hfma2(as_h2(w0.x), as_h2(x.x), acc);
            acc = __hfma2(as_h2(w0.y), as_h2(x.y), acc);
            acc = __hfma2(as_h2(w1.x), as_h2(x.z), acc);
            acc = __hfma2(as_h2(w1.y), as_h2(x.w), acc);
        }
    }
    return d * h2sum(acc);
}

// ------------------------------------------------------------------------------------------------ kernels
template<int TG>
__global__ void __launch_bounds__(THREADS) pgu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                      const int32_t* __restrict__ grp_start,
                                                      const int32_t* __restrict__ n_groups,
                                                      const int32_t* __restrict__ ent_tok,
                                                      const block_q8_1* __restrict__ xq, size_t up_off, size_t gu_row,
                                                      float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    fill_gu_tables<TG>(t);
    const int row0 = blockIdx.x * GU_ROWS;               // 32 rows: all gate or all up (FF % 32 == 0)
    const bool is_up = row0 >= FF;
    const uint8_t* mat = (const uint8_t*) grp_ptr[g] + (is_up ? up_off : 0);
    float* dst = is_up ? up : gate;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        __syncthreads();                                   // tables filled / the previous entry consumed
        const block_q8_1* xe = xq + (size_t) ent_tok[e] * (H / 32);
        for (int i = t; i < H; i += THREADS) {             // q/128: exact in fp16
            const int8_t q = xe[i >> 5].qs[i & 31];
            reinterpret_cast<half*>(s_x)[i] = __float2half_rn((float) q * (1.0f / 128.0f));
            if ((i & 31) == 0) s_xd[i >> 5] = __low2float(xe[i >> 5].ds) * 128.0f;
        }
        __syncthreads();
        for (int rr = warp; rr < GU_ROWS; rr += WARPS) {
            const int r = (row0 + rr) - (is_up ? FF : 0);
            const uint8_t* wr = mat + (size_t) r * gu_row;
            float s = 0.0f;
            for (int sb = lane; sb < H / 32; sb += 32) s += s_xd[sb] * gu_sub<TG>(wr, sb);
            s = warp_sum(s);
            if (lane == 0) dst[(size_t) e * FF + r] = s;
        }
    }
}

__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                              long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(THREADS) pdown_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_dst,
                                                        const float* __restrict__ h, size_t down_off, size_t d_row,
                                                        float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if constexpr (TD == 20) {
        for (int v = t; v < 256; v += THREADS)
            s_nl[v] = as_u(__halves2half2(__int2half_rn(kvalues_iq4nl[v & 15]), __int2half_rn(kvalues_iq4nl[v >> 4])));
    }
    if constexpr (TD == 42) {
        for (int v = t; v < 256; v += THREADS) {
            half vals[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) vals[j] = __int2half_rn((int) ((v >> (2 * j)) & 3) - 1);
            s_q2[v] = make_uint2(as_u(__halves2half2(vals[0], vals[1])), as_u(__halves2half2(vals[2], vals[3])));
        }
    }
    const int row0 = blockIdx.x * D_ROWS;
    const uint8_t* mat = (const uint8_t*) grp_ptr[g] + down_off;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        __syncthreads();
        const float* he = h + (size_t) e * FF;
        for (int i = t; i < FF; i += THREADS) {           // a warp covers 32 aligned values: one scale block
            const float v = he[i];
            float amax = fabsf(v);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            // the next power of two >= amax: dividing by it is exact, and |h / s| <= 1 keeps fp16 sums in range
            const float s = amax > 1e-30f ? __int_as_float((__float_as_int(amax) + 0x007FFFFF) & 0x7F800000) : 1.0f;
            // IQ4_NL: value j of a 32-block goes to half (2j) for j < 16, (2(j-16) + 1) otherwise (see down_item<20>)
            const int j = i & 31, slot = TD == 20 ? (i & ~31) + (j < 16 ? 2 * j : 2 * (j - 16) + 1) : i;
            reinterpret_cast<half*>(s_h)[slot] = __float2half_rn(v * (1.0f / s));
            if ((i & 31) == 0) s_hd[i >> 5] = s;
        }
        __syncthreads();
        for (int rr = warp; rr < D_ROWS; rr += WARPS) {
            const int r = row0 + rr;
            const uint8_t* wr = mat + (size_t) r * d_row;
            // 20 items of 32 values (lanes 0..19), as the int8 kernel
            const float s = lane < FF / 32 ? s_hd[lane] * down_item<TD>(wr, lane) : 0.0f;
            const float sum = warp_sum(s);
            if (lane == 0) out[(size_t) ent_dst[e] * H + r] = sum;
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

template<int TG>
void launch_gu(dim3 grid, cudaStream_t s, const unsigned long long* gp, const int32_t* gs, const int32_t* ng,
               const int32_t* tok, const block_q8_1* x, const NativeExpertLayout& L, float* gate, float* up) {
    pgu_kernel<TG><<<grid, THREADS, 0, s>>>(gp, gs, ng, tok, x, L.up_off, L.gu_row, gate, up);
}
template<int TD>
void launch_down(dim3 grid, cudaStream_t s, const unsigned long long* gp, const int32_t* gs, const int32_t* ng,
                 const int32_t* dst, const float* h, const NativeExpertLayout& L, float* out) {
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
    // native_expert_grouped's scratch: gate | up | h | (hq, unused here)
    const size_t f = (size_t) cap_entries * FF * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) (2 * FF / GU_ROWS), (unsigned) cap_groups);
    switch (L.gu_type) {
        case 18: launch_gu<18>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        case 21: launch_gu<21>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        default: launch_gu<22>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    }
    const long long nh = (long long) cap_entries * FF;
    swiglu_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
    const dim3 gd((unsigned) (H / D_ROWS), (unsigned) cap_groups);
    if (L.d_type == 20) launch_down<20>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, h, L, out);
    else launch_down<42>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, h, L, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "pascal_expert_grouped: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::kernels
#endif  // !__HIPCC__
