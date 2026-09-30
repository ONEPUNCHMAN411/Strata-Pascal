# Strata-Pascal handoff (2026-09-30)

Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for **2x Tesla P100 16 GB** running
**Qwen3.8-Flash-Next IQ3_S**. Work branch: `claude/p100-server-nvme-migration-9wn0sg` (no PR: it is the repo's only
branch). Nothing below has run on a GPU yet - everything is compiled/verified in a GPU-less container only.

## The server
VenkataAI (192.168.68.111, user `venkata`): Xeon E5-2678 v3 (12C/24T, AVX2, no AVX-512, 1 socket), 128 GB DDR3-1600
quad channel, 2x P100 PCIe 3.0 x16 (same root, P2P OK), driver 580 / CUDA 12.8 installed, 512 GB NVMe (Inland TN320)
at /home/venkata/models, repo cloned at ~/Strata-Pascal.

## Run everything (one command, ~1-2 h + 84 GB download the first time)
    cd ~/Strata-Pascal && git pull && ./pascal/run-all.sh
Update, hardware report, install IQ3_S on GPUs 0,1 (compiled from this code), GPU correctness tests, calibration,
A/B benchmark of every change + option (winners saved into the config with a backup), one profiled request.
Send back `~/p100-report-*.tar.gz` (or `summary.txt`). Stop other GPU services (router/llama) first for clean numbers.

## Done (all pushed)
| Change | Where | Switch |
|---|---|---|
| Pascal build path: setup accepts cc 6.x, CUDA 12.x, `-DSTRATA_EXPERIMENTAL_SM60` | setup.py, CMakeLists, device.cu | - |
| **Whole expert arena pinned with 2 GPUs on Linux** (upstream capped at 8 GiB = no PCIe share on ~41 of 48 layers) | generate.cpp | `STRATA_PIN_LIMIT_GIB=N` |
| **Free other stages' dense matrices** after the split (~1.8 GB/card -> more cached experts) | native_dense.cpp, generate.cpp | `STRATA_SPLIT_KEEP_DENSE=1` |
| **FP16 expert kernels** (IQ3_XXS/IQ3_S/IQ2_S gate/up, IQ4_NL/Q2_0 down; not bit-exact) | pascal_experts.cu | `STRATA_PASCAL_FP16=0` |
| fused_gr down in one launch for 8-token windows | fused_gr.cu | - |
| IQ codebooks in shared memory, multi-entry weight reuse (int8 path) | iq_kernels.cu | - |
| THP for the arena, qsa smem cap, unused shared-expert bf16 copy removed | pinned.cu, qsa.cu, verify.cpp | - |
| Tools: run-all.sh, ab_bench.py (+`--apply`), build-p100.sh, server-checks.sh (status/hwinfo/coldload/crashrun), pascal_iq_parity test | pascal/, src/kernels/ | - |

Docs: `pascal/README.md` (setup + tuning), `pascal/PERF_ANALYSIS.md` (where time goes, ranked fixes).

## In progress (NOT pushed - lost if the cloud session ends)
1. **FP16 token-tiled dense matmuls incl. the LM head** (`pascal_dense.cu`, test `pascal_dense_parity`) - highest
   expected win: dense GEMVs on emulated dp4a are likely the largest cost per window.
   If gone, re-create it from PERF_ANALYSIS.md item 1.

**Merged since:** token-tiled + fused (gate/up/SwiGLU) FP16 expert kernels with one-wave grids looping over the
device-side group count (`pascal_experts.cu`, commit 95ed800): per 32 weights x 8 entries ~3.5x (gate/up) / ~4x
(down) fewer instructions; an empty PCIe-share call is two short launches. Unrun on a GPU: `pascal_iq_parity`
checks it (run-all.sh runs it).

## Next steps, in order
1. Run `run-all.sh`; read `summary.txt`: per-variant tok/s, `profile.log` timing lines (GPU wait vs CPU pool vs
   staging per card), startup lines (arena pinned? PCIe probe, experts cached, hit rate).
2. If GPU time dominates: finish items 1-2 above. If the CPU pool dominates: misses matter -> more VRAM, skip
   low-weight missed experts (needs routing weights plumbed into `expert_pool_dispatch_multi`), sticky-miss ring.
3. Then: skip the q8_1 quantize for FP16 paths (~240 kernels/window), cross-window pipelining across the two cards.

## Other open items from the original session
- BigBang-MTP segfault (libggml-base.so): retry after the DIMM swap: `./pascal/server-checks.sh crashrun <cmd>`.
- Timed NVMe cold-load test: `./pascal/server-checks.sh coldload <file on NVMe> <same file in models_old_hdd>`.
- Delete `/home/venkata/models_old_hdd` once the NVMe copy is trusted.
- Laptop: delete the duplicate `neuralorganism_v3*` folders (~3.4 GB) and the Cisco-CCNA duplicate (user approved).
- Q4_K_M rejected: 119 GB, n-gram must be GPU-resident in llama.cpp; IQ3_S matches BF16 on ISTA's benchmarks.
