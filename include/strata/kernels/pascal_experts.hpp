// include/strata/kernels/pascal_experts.hpp - FP16 grouped experts for Pascal (sm_6x) cards.
//
// sm_6x has no dp4a (each emulated one is ~10 instructions) but runs half2 FMA at twice the fp32 rate.  These
// kernels decode the IQ3_S model's expert formats (gate/up IQ3_XXS, IQ3_S, IQ2_S; down IQ4_NL, Q2_0) straight to
// half2 and multiply with half2 activations, instead of the int8 dot products of iq_kernels.cu.  Each weight
// sub-block is decoded once per group and multiplied against all of the group's entries (up to 8 at a time); gate,
// up and SwiGLU run in one kernel; the grids are one wave of blocks looping over the device-side group count, so
// an empty call (n_groups = 0) costs two short launches whatever the caps.
//
// Numerics: NOT bitwise the int8 path.  Gate/up read the same q8_1 activation (exactly, as q/128 in fp16); the down
// projection reads the fp32 SwiGLU output scaled per 32 by a power of two (11-bit mantissa instead of q8_1's 8 bits).
// Products of 32 values accumulate in fp16 (16 per half lane, |partial| <= 992 gate/up, <= 2032 down), then in fp32.
// An entry's result does not depend on how entries are grouped (bitwise), and runs are deterministic.
// `pascal_iq_parity` measures both paths against an fp64 reference.  STRATA_PASCAL_FP16=0 in the environment turns
// this path off.
#pragma once

#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {

/// `native_expert_grouped`'s contract.  Returns false, launching nothing, when the current device is not sm_6x,
/// the path is disabled, or the layout/types are not covered (n_embd 2560, n_ff 640; gate/up 18/21/22, down 20/42).
bool pascal_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream);

/// Force the path on (1) or off (0) for this process, overriding STRATA_PASCAL_FP16 (tests); -1 restores it.
void pascal_expert_fp16_override(int on);

}  // namespace strata::kernels
