# Where decode time goes on 2x P100 (IQ3_S) - analysis from the code

No GPU was available for this; every number below is derived from the model's shapes, the engine's structure and
P100 specs, and should be checked against `STRATA_DECODE_TIMING` / `STRATA_VERIFY_PROFILE` output from the server.
Upstream's only Pascal measurement (4 GP100-class dies, IQ3_XXS, all experts resident): 21.3 tok/s decode, which
at 2.4-3.2 accepted tokens per verify window is ~110-150 ms per window. The estimates here add up to less than
that (they assume ~40% of peak issue rate), so treat them as a ranking, not a prediction.

## How one verify window runs

One CUDA graph covers all 48 layers (`Verifier::record_window`, src/core/verify.cpp). Per layer the GPU:
1. `pre(l)`: hyper-connection read (3 kernels), the mixer - GDN on 36 layers (~8 kernels) or QSA on 12 (~20 + 2 per
   token) - a second hyper-connection read, the router (bf16 GEMV + top-10), the shared expert (~10 kernels), the
   q8_1 activation quantize, then **rings a doorbell** in mapped host memory.
2. The host thread (spinning) sees it, plans the layer (which routed experts are in VRAM / go over PCIe / go to the
   CPU), publishes the plan (flag A), computes the CPU experts, raises the CPU flag.
3. `post(l)`: a spin kernel waits for flag A, runs the VRAM experts (4 kernels), waits for the PCIe share, runs
   it, waits for the CPU rows, combines.

Every layer depends on the previous one, so the window is a serial chain of ~48 x (40-70) kernels plus 48 GPU->host
->GPU round trips. With `--gpus 0,1` card 0 runs layers 0..K-1, synchronizes, then card 1 runs the rest (one
hand-off per window); the cards never work at the same time. After the window: sample, commit graph, then the MTP
drafter drafts the next window token by token.

## Estimated cost per window (T ~ 5 tokens), largest first

| # | Where | Why it costs on a P100 | Estimate / window | Fix |
|---|---|---|---|---|
| 1 | **Dense quantized GEMVs** (attn_qkv 26M, gate 16M, ssm_out 16M per GDN layer; q/k/v/out on QSA; shared expert) | ~60M weights/layer x 48 x T = ~15B multiply-adds, all through **emulated dp4a** (~10 instr each). The multi-column kernel uses 1 row per 128-thread block (10,240 tiny blocks for attn_qkv) | largest single item, ~15-25 ms | FP16 token-tiled dense kernels (in progress), fatter blocks |
| 2 | **Routed experts** (VRAM hits) | ~10 experts x T entries x 4.9M weights x 48 layers = ~10B multiply-adds; int8 path re-decodes weights per token | ~10-16 ms | FP16 expert kernels (done), token tiling + fused gate/up/SwiGLU (in progress) |
| 3 | **Kernel launch/latency floor** | ~2,500 dependent kernels per window; Pascal has no fast graph launch path, ~3-5 us each even when tiny | ~8-12 ms | fusion (below) |
| 4 | **LM head** (248,320 x 2560, Q6_K = 520 MB) | read once per window and multiplied against T tokens on the emulated path, on the last card | ~3-6 ms | covered by FP16 dense (Q6_K) |
| 5 | **CPU experts on misses** | the GPU waits for the CPU rows before combining; Haswell AVX2 at DDR3-1600 | 0-10 ms, depends on hit rate | pin fix (done: PCIe share now possible on all layers), skip-low-weight misses, more VRAM |
| 6 | **Per-layer host round trip** | doorbell -> host plan -> flag, polled over PCIe, x48 | ~1-3 ms | `STRATA_VERIFY_DEVICE_PLAN=1` (exists, off by default) plans all-resident layers on the GPU |
| 7 | **Drafting** (MTP layer per draft token, sequential) | ~T small forward passes after each window | ~1-3 ms | fine for now |
| 8 | **Card hand-off + commit** | stream sync, host copy, second graph launch | ~0.5-1 ms | pipelining idea (large) |

**Correction to earlier chat estimates:** I had said the expert kernels were probably not the bottleneck (~2-3 ms per
token). That missed the dense projections and the LM head, and the fact that a verify window multiplies every weight
by T tokens. Counted properly, emulated-int8 GEMV arithmetic (items 1, 2, 4) is plausibly half or more of the window,
which matches upstream's own note that Pascal decode is "bounded by Pascal's per-layer arithmetic".

## What to do, in order

1. **FP16 token-tiled dense kernels incl. the head** (agent in progress) - biggest expected win.
2. **FP16 token-tiled + fused expert kernels** (agent in progress).
3. **Skip the q8_1 quantize for FP16 paths**: the FP16 kernels re-expand the int8 activation to half; reading the
   fp32 activation directly removes ~5 quantize kernels per layer (~240 per window) and is more precise. Needs
   `verify.cpp` to skip `native_quantize_q8_1` on Pascal and pass the fp32 buffer.
4. **Batch the per-token QSA kernels** (`kv_append_q8_step`, `native_qsa_indexer_append` run once per token: 2T
   launches per QSA layer) into one launch each.
5. **Free knobs to A/B on the server**: `STRATA_VERIFY_DEVICE_PLAN=1`; `./setup.sh --calibrate`; `--spec 6/8`.
6. Only then, structural ideas: both cards busy (cross-window pipelining), per-card dense weights, skip-low-weight
   misses.

## What to send back to confirm this

```
"env": {"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1", "STRATA_VERIFY_PROFILE": "1"}
```
One ~500-token request; the per-window lines split GPU wait vs host plan vs CPU pool vs staging per card, and
`STRATA_VERIFY_PROFILE` gives GPU time per stage (VRAM experts, PCIe, CPU wait, combine, head) for GDN vs QSA
layers. If "GPU wait" dominates and the VRAM-expert/mixer stages are large, items 1-2 are right; if the CPU pool
dominates, misses are the problem instead.
