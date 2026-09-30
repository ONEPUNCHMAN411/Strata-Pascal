// include/strata/kernels/pascal_dense.hpp - FP16 token-tiled dense matrix-vector products for Pascal (sm_6x).
//
// sm_6x has no dp4a (each emulated one is ~10 instructions) but runs half2 FMA at twice the fp32 rate.  For the
// multi-column case of native_mmvq (speculative verify windows of T = 2..8 tokens: dense projections and the LM
// head) these kernels decode each 32-value weight group ONCE to half2 and multiply it against all T staged
// activation columns, two rows per lane, instead of one emulated-dp4a dot product per (row, column).
//
// Formats: Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), IQ4_NL (20), IQ4_XS (23) - the dense tensors of the IQ3_S
// model.  Block layouts are llama.cpp's (ggml-common.h); each decoder follows the reference dequantizer index for
// index (see dequant_bf16.cu's group32 and the vec_dot_*_q8_1 in native_mmvq.cu).
//
// Numerics: NOT bitwise the int8 path, and therefore NOT bitwise equal to a single-column call either (the
// multi_exact contract of native_mmvq_set_multi_exact does not hold while this path is on).  The activation is read
// exactly (q8_1 values staged as q/128 in fp16, d*128 kept as a per-32 float scale); the products of at most 16
// values per half lane accumulate in fp16 before the block/sub-block scales are applied in fp32.
// `pascal_dense_parity` measures both paths against an fp64 reference.  STRATA_PASCAL_FP16=0 in the environment
// turns this path off (the same switch as pascal_experts).
#pragma once

namespace strata::kernels {

/// native_mmvq's multi-column contract: `x_q8_1` holds `ncols` columns of q8_1 blocks (36 bytes per 32 values,
/// column j at block j * n_in / 32); `y` is column-major, y[j * n_out + row].  Returns false, launching nothing, when
/// the current device is not sm_6x, the path is disabled, ncols is outside 2..8, the type is not covered, or n_in is
/// not a multiple of 256.  Pointers and stream as native_mmvq (device pointers, 4-byte aligned, explicit stream).
bool pascal_dense_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                       void* stream);

/// Force the path on (1) or off (0) for this process, overriding STRATA_PASCAL_FP16 and the device check (tests);
/// -1 restores the default.
void pascal_dense_fp16_override(int on);

}  // namespace strata::kernels
