// src/kernels/pascal_iq_parity.cpp - the Pascal (sm_6x) tuning in iq_kernels.cu, on synthetic blocks (no model).
//
//   1. native_expert_grouped: one group holding every entry (the multi-entry path shares each weight block across
//      entries) must equal, bitwise, the same entries each in a group of its own.
//   2. iq_mmvq (the grid read from shared memory on sm_6x) must match the dequantized matrix (grid read from global
//      memory) times the dequantized q8_1 activation, within the activation rounding.
//
// Runs on any CUDA card; on RTX builds it checks the upstream code paths the same way.
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int QK_K = 256;
constexpr int N_EMBD = 2560, N_FF = 640;   // the model's expert shape

#define CK(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

int qk_of(int ty) { return ty == 20 ? 32 : ty == 42 ? 64 : QK_K; }

// Random blocks with a sane fp16 scale (a random one is NaN/inf 1 time in 32).
std::vector<uint8_t> random_rows(int ty, int rows, int n, std::mt19937& rng) {
    const size_t rb = iq_row_bytes(ty, n), bb = rb / (n / qk_of(ty));
    std::vector<uint8_t> w(rb * rows);
    for (auto& b : w) b = (uint8_t) rng();
    const uint16_t d16 = 0x2C00;   // 0.0625
    for (size_t o = 0; o < w.size(); o += bb) {
        if (ty == 29) {            // IQ1_M: the fp16 scale is the top nibbles of its four 16-bit scale words
            uint16_t sc[4];
            std::memcpy(sc, &w[o + bb - 8], 8);
            for (int i = 0; i < 4; ++i) sc[i] = (uint16_t) ((sc[i] & 0x0FFF) | (((d16 >> (4 * i)) & 0xF) << 12));
            std::memcpy(&w[o + bb - 8], sc, 8);
        } else {
            std::memcpy(&w[o], &d16, 2);
        }
    }
    return w;
}

bool grouped_test(int gu, int dn, std::mt19937& rng) {
    const int T = 6, G = 3;                 // a verify window of 6 tokens, 3 distinct experts
    const NativeExpertLayout L = native_expert_layout(gu, dn, N_EMBD, N_FF);
    std::vector<uint8_t> blobs;
    for (int g = 0; g < G; ++g) {
        auto a = random_rows(gu, 2 * N_FF, N_EMBD, rng);
        auto b = random_rows(dn, N_EMBD, N_FF, rng);
        blobs.insert(blobs.end(), a.begin(), a.end());
        blobs.insert(blobs.end(), b.begin(), b.end());
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) T * N_EMBD);
    for (auto& v : x) v = nd(rng);
    // entries: expert 0 takes all 6 tokens, expert 1 tokens 0-3, expert 2 tokens 2 and 5
    const std::vector<std::vector<int>> toks = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}, {2, 5}};
    int n_ent = 0;
    for (auto& t : toks) n_ent += (int) t.size();

    uint8_t* d_blob; float* d_x; void* d_xq; void* d_scr; float *d_out1, *d_out2;
    int32_t* d_i;
    unsigned long long* d_p;
    CK(cudaMalloc(&d_blob, blobs.size()));
    CK(cudaMemcpy(d_blob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, (size_t) T * (N_EMBD / 32) * 36));
    quantize_q8_1_rows(d_x, T, N_EMBD, d_xq, nullptr);
    CK(cudaMalloc(&d_scr, native_expert_scratch_bytes(n_ent, N_FF)));
    CK(cudaMalloc(&d_out1, (size_t) n_ent * N_EMBD * 4));
    CK(cudaMalloc(&d_out2, (size_t) n_ent * N_EMBD * 4));
    CK(cudaMalloc(&d_i, 4 * 256));
    CK(cudaMalloc(&d_p, 8 * 64));

    auto run = [&](bool split, float* out) {
        std::vector<unsigned long long> ptr;
        std::vector<int32_t> start, dst, tok;
        int e = 0;
        for (int g = 0; g < G; ++g) {
            const unsigned long long p = (unsigned long long) (d_blob + (size_t) g * L.bytes);
            if (!split) { ptr.push_back(p); start.push_back(e); }
            for (int t : toks[g]) {
                if (split) { ptr.push_back(p); start.push_back(e); }
                dst.push_back(e++);
                tok.push_back(t);
            }
        }
        start.push_back(e);
        const int ng = (int) ptr.size();
        // [n_groups | grp_start (ng+1) | ent_dst | ent_tok]
        std::vector<int32_t> ints = {ng};
        ints.insert(ints.end(), start.begin(), start.end());
        ints.insert(ints.end(), dst.begin(), dst.end());
        ints.insert(ints.end(), tok.begin(), tok.end());
        CK(cudaMemcpy(d_i, ints.data(), ints.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_p, ptr.data(), ptr.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemset(out, 0, (size_t) n_ent * N_EMBD * 4));
        native_expert_grouped(L, d_p, d_i + 1, d_i, d_i + 1 + (ng + 1), d_i + 1 + (ng + 1) + n_ent, n_ent, n_ent, d_xq,
                              d_scr, out, nullptr);
        CK(cudaDeviceSynchronize());
    };
    run(false, d_out1);
    run(true, d_out2);
    std::vector<float> o1((size_t) n_ent * N_EMBD), o2(o1.size());
    CK(cudaMemcpy(o1.data(), d_out1, o1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), d_out2, o2.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0, nonzero = 0;
    for (size_t i = 0; i < o1.size(); ++i) {
        bad += std::memcmp(&o1[i], &o2[i], 4) != 0;
        nonzero += o1[i] != 0.f && std::isfinite(o1[i]);
    }
    const bool ok = bad == 0 && nonzero > o1.size() / 2;
    std::printf("grouped  gu=%2d down=%2d  %d entries in %d groups vs 1 each: %zu of %zu differ, %zu finite nonzero  %s\n",
                gu, dn, n_ent, G, bad, o1.size(), nonzero, ok ? "OK" : "FAIL");
    cudaFree(d_blob); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_scr); cudaFree(d_out1); cudaFree(d_out2);
    cudaFree(d_i); cudaFree(d_p);
    return ok;
}

bool mmvq_test(int ty, std::mt19937& rng) {
    const int rows = 64, n = N_EMBD, cols = 3;
    const auto w = random_rows(ty, rows, n, rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) cols * n);
    for (auto& v : x) v = nd(rng);
    void *d_w, *d_xq;
    float *d_x, *d_y, *d_deq;
    CK(cudaMalloc(&d_w, w.size()));
    CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, (size_t) cols * (n / 32) * 36));
    CK(cudaMalloc(&d_y, (size_t) cols * rows * 4));
    CK(cudaMalloc(&d_deq, (size_t) rows * n * 4));
    quantize_q8_1_rows(d_x, cols, n, d_xq, nullptr);
    iq_mmvq(ty, d_w, d_xq, d_y, n, rows, cols, nullptr);
    iq_dequant_f32(ty, d_w, (int64_t) rows * n, d_deq, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> y((size_t) cols * rows), deq((size_t) rows * n);
    std::vector<uint8_t> xq((size_t) cols * (n / 32) * 36);
    CK(cudaMemcpy(y.data(), d_y, y.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(deq.data(), d_deq, deq.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(xq.data(), d_xq, xq.size(), cudaMemcpyDeviceToHost));
    auto h2f = [](uint16_t h) {
        const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
        const float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 0x400), e - 25);
        return (h & 0x8000) ? -v : v;
    };
    double worst = 0;
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r) {
            double ref = 0, mag = 0;
            for (int b = 0; b < n / 32; ++b) {
                const uint8_t* blk = &xq[((size_t) c * (n / 32) + b) * 36];
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const float d = h2f(d16);
                for (int i = 0; i < 32; ++i) {
                    const double t = (double) deq[(size_t) r * n + b * 32 + i] * d * (int8_t) blk[4 + i];
                    ref += t;
                    mag += std::fabs(t);
                }
            }
            worst = std::fmax(worst, std::fabs(y[(size_t) c * rows + r] - ref) / (mag + 1e-30));
        }
    // integer scale rounding in the dot products (e.g. IQ2_XS's (s*ls + s/2)/4) keeps this ~1e-3 of the magnitude
    const bool ok = worst < 1e-2;
    std::printf("mmvq     type=%2d  %d rows x %d cols vs dequant: worst |err|/sum|terms| = %.2e  %s\n", ty, rows, cols,
                worst, ok ? "OK" : "FAIL");
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_xq); cudaFree(d_y); cudaFree(d_deq);
    return ok;
}

}  // namespace

int main() {
    int dev = 0;
    cudaDeviceProp p{};
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    std::printf("%s, compute capability %d.%d\n", p.name, p.major, p.minor);
    std::mt19937 rng(1234);
    int fail = 0;
    for (int gu : {16, 17, 18, 21, 22, 23, 29, 42}) fail += !grouped_test(gu, 20, rng);
    fail += !grouped_test(18, 42, rng);   // (IQ4_XS down needs n_ff % 256 == 0; the model's 640 is not)
    for (int ty : {16, 17, 18, 20, 21, 22, 23, 29, 42}) fail += !mmvq_test(ty, rng);
    std::printf("%s (%d failed)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
