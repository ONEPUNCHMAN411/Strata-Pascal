// src/kernels/pascal_dense_parity.cpp - the Pascal FP16 token-tiled dense kernels (pascal_dense.cu), synthetic, no model.
//
// For each covered format (Q6_K, Q4_K, Q5_K, IQ4_XS, IQ4_NL, Q8_0), n_in in {2560, 6144} and ncols in {2, 3, 4, 6, 8}:
// random blocks with sane fp16 scales, one q8_1 quantization of normal activations, then native_mmvq's
// multi-column call twice - the FP16 path forced on, and forced off (the upstream int8 path) - each against an fp64
// reference computed from the dequantized weights (dequant_f32) and the dequantized q8_1 activations.  The FP16
// relative L2 error must stay below 1e-2 and below 2x the int8 path's + 2e-3.  Also: every output finite, the
// row tail (n_out not a multiple of the rows per block) and the 8-warp layout (n_out >= 16384) exercised, and the
// fall-through cases (ncols 1, an uncovered type) launch nothing.
//
// Runs on any CUDA card: the override forces the FP16 path on whatever the compute capability.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/pascal_dense.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

namespace {

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

struct Ty { int id; const char* name; int be, bb; std::vector<int> scale_offs; };

// fp16 bits of a normal value in +-[2^-10, 2^-5): random bytes there would put inf/NaN in the outputs
uint16_t sane_h(std::mt19937& rng) {
    const uint16_t exp = (uint16_t) (5 + rng() % 5);        // biased exponents 5..9: 2^-10 .. 2^-6 * (1.x)
    return (uint16_t) (((rng() & 1u) << 15) | (exp << 10) | (rng() & 0x3FFu));
}

std::vector<uint8_t> random_rows(const Ty& t, int rows, int n, std::mt19937& rng) {
    const size_t bb = (size_t) t.bb, nb = (size_t) rows * (n / t.be);
    std::vector<uint8_t> w(nb * bb);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t o = 0; o < w.size(); o += bb)
        for (int off : t.scale_offs) {
            const uint16_t h = sane_h(rng);
            std::memcpy(&w[o + off], &h, 2);
        }
    return w;
}

float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else { float v = std::ldexp((float) m, -24); return s ? -v : v; }
    } else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e - 15 + 127) << 23) | (m << 13);
    float v;
    std::memcpy(&v, &f, 4);
    return v;
}

double rel_l2(const std::vector<float>& y, const std::vector<double>& ref, size_t& nonfinite) {
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) { ++nonfinite; continue; }
        const double d = (double) y[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1));
}

bool run_case(const Ty& t, int n_in, int n_out, const std::vector<int>& cols_list, std::mt19937& rng,
              cudaStream_t s) {
    const int max_cols = 8;
    const auto w = random_rows(t, n_out, n_in, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) max_cols * n_in);
    for (auto& v : x) v = nd(rng);
    void *d_w, *d_xq;
    float *d_x, *d_y, *d_deq;
    CK(cudaMalloc(&d_w, w.size()));
    CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, native_q8_1_bytes(n_in, max_cols)));
    CK(cudaMalloc(&d_y, (size_t) max_cols * n_out * 4));
    CK(cudaMalloc(&d_deq, (size_t) n_out * n_in * 4));
    native_quantize_q8_1(d_x, d_xq, n_in, max_cols, s);
    dequant_f32(t.id, d_w, 0, n_out, n_in, d_deq, s);
    CK(cudaStreamSynchronize(s));
    std::vector<float> deq((size_t) n_out * n_in);
    std::vector<uint8_t> xq(native_q8_1_bytes(n_in, max_cols));
    CK(cudaMemcpy(deq.data(), d_deq, deq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xq.data(), d_xq, xq.size(), cudaMemcpyDeviceToHost));
    // the q8_1 activation as the kernels read it: d * q
    std::vector<double> xd((size_t) max_cols * n_in);
    for (size_t b = 0; b < xd.size() / 32; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &xq[b * 36], 2);
        const double d = h2f(dh);
        for (int i = 0; i < 32; ++i) xd[b * 32 + i] = d * (double) (int8_t) xq[b * 36 + 4 + i];
    }
    bool ok = true;
    for (int nc : cols_list) {
        std::vector<double> ref((size_t) nc * n_out);
        for (int c = 0; c < nc; ++c)
            for (int r = 0; r < n_out; ++r) {
                const float* wr = &deq[(size_t) r * n_in];
                const double* xc = &xd[(size_t) c * n_in];
                double a = 0;
                for (int k = 0; k < n_in; ++k) a += (double) wr[k] * xc[k];
                ref[(size_t) c * n_out + r] = a;
            }
        std::vector<float> y16((size_t) nc * n_out), yup(y16.size());
        CK(cudaMemset(d_y, 0xFF, (size_t) nc * n_out * 4));        // NaN: an unwritten output fails
        pascal_dense_fp16_override(1);
        const bool taken = pascal_dense_mmvq(t.id, d_w, d_xq, d_y, n_in, n_out, nc, s);
        CK(cudaMemset(d_y, 0xFF, (size_t) nc * n_out * 4));
        native_mmvq(t.id, d_w, d_xq, d_y, n_in, n_out, nc, s);      // through the hook in native_mmvq.cu
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(y16.data(), d_y, y16.size() * 4, cudaMemcpyDeviceToHost));
        pascal_dense_fp16_override(0);
        CK(cudaMemset(d_y, 0xFF, (size_t) nc * n_out * 4));
        native_mmvq(t.id, d_w, d_xq, d_y, n_in, n_out, nc, s);
        CK(cudaStreamSynchronize(s));
        CK(cudaMemcpy(yup.data(), d_y, yup.size() * 4, cudaMemcpyDeviceToHost));
        size_t nf16 = 0, nfup = 0;
        const double e16 = rel_l2(y16, ref, nf16), eup = rel_l2(yup, ref, nfup);
        const bool pass = taken && nf16 == 0 && nfup == 0 && e16 <= 1e-2 && e16 <= 2 * eup + 2e-3;
        std::printf("%-7s n_in=%5d n_out=%6d ncols=%d  fp16 rel L2 %.3e  int8 %.3e  %s%s\n", t.name, n_in, n_out, nc,
                    e16, eup, pass ? "OK" : "FAIL", taken ? "" : " (fp16 path not taken)");
        if (nf16 || nfup) std::printf("        non-finite outputs: fp16 %zu, int8 %zu\n", nf16, nfup);
        ok = ok && pass;
    }
    pascal_dense_fp16_override(-1);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_y); cudaFree(d_deq);
    return ok;
}

}  // namespace

int main() {
#if defined(__HIP_PLATFORM_AMD__)
    std::printf("pascal_dense_parity: skipped (HIP build: the FP16 Pascal path is compiled out)\n");
    return 0;
#endif
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(20260930);
    const std::vector<Ty> types = {
        {14, "Q6_K", 256, 210, {208}},
        {12, "Q4_K", 256, 144, {0, 2}},
        {13, "Q5_K", 256, 176, {0, 2}},
        {23, "IQ4_XS", 256, 136, {0}},
        {20, "IQ4_NL", 32, 18, {0}},
        {8, "Q8_0", 32, 34, {0}},
    };
    const std::vector<int> cols = {2, 3, 4, 6, 8};
    bool ok = true;
    for (const auto& t : types)
        for (int n_in : {2560, 6144}) ok = run_case(t, n_in, 300, cols, rng, s) && ok;   // 300: a partial block
    // the 8-warp layout (tall matrices, the LM head)
    ok = run_case(types[0], 2560, 16400, {2, 8}, rng, s) && ok;
    ok = run_case(types[5], 2560, 16400, {5}, rng, s) && ok;

    // fall-through: nothing launched, false returned
    pascal_dense_fp16_override(1);
    const bool ft = !pascal_dense_mmvq(14, nullptr, nullptr, nullptr, 2560, 64, 1, s) &&   // ncols 1
                    !pascal_dense_mmvq(11, nullptr, nullptr, nullptr, 2560, 64, 4, s) &&   // Q3_K: not covered
                    !pascal_dense_mmvq(8, nullptr, nullptr, nullptr, 2560 + 32, 64, 4, s);  // n_in % 256
    pascal_dense_fp16_override(0);
    const bool off = !pascal_dense_mmvq(14, nullptr, nullptr, nullptr, 2560, 64, 4, s);     // forced off
    pascal_dense_fp16_override(-1);
    std::printf("fall-through cases: %s\n", ft && off ? "OK" : "FAIL");
    ok = ok && ft && off;
    cudaStreamDestroy(s);
    std::printf("pascal_dense_parity: %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
