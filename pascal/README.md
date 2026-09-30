# Strata on Tesla P100 (sm_60)

Upstream Strata carries an experimental Pascal build (`-DSTRATA_EXPERIMENTAL_SM60=ON`, commit `01bca8a`). This fork adds:

- `setup.py`: accepts compute capability 6.x cards, always compiles for them (no ready-made sm_6x engine), and uses a
  CUDA 12.x toolkit (`cuda-toolkit-12-8`) because CUDA 13 dropped Pascal. A Pascal card cannot share an engine with an RTX 50.
- `src/core/device.cu`: the device check (`strata-device`, self-tests) accepts 6.0 in a Pascal build.
- `pascal/build-p100.sh`: standalone sm_60 build + `strata-device` and `pascal_iq_parity` on each GPU.
- `pascal/server-checks.sh`: `status` (stray processes, GPUs, EDAC per DIMM, mounts), `coldload FILE..` (uncached read
  timing), `crashrun CMD..` (run with core dumps + dmesg capture, for the BigBang-MTP segfault).

## P100 kernel tuning (`src/kernels/cuda/iq_kernels.cu`, sm_6x builds only)

- **Codebook grids in shared memory.** The i-quant dot products look up `iq2xs/iq2s/iq3xxs/iq3s/iq1m` grids; on GP100
  those 1-8 KB tables share the 24 KB L1 with the streaming weights and get evicted, so lookups fall to L2.
- **Grouped experts reuse each weight block across tokens.** `native_gu_kernel` / `native_down_kernel` visited the
  whole row once per token routed to the expert; now each block is visited once for all of them (up to 8, the verify
  window). Per-token sums live in shared memory so registers/occupancy stay at the baseline (ptxas: IQ3_XXS gate/up
  48 -> 46 regs, IQ4_NL down 40 -> 40).
- **Hyper-connection read in one launch** (`fused_gr.cu`): sm_6x only launches 48 KB per block, so the 2560-float
  tile carried 4 tokens and a 6-token verify window read the 6.6 MB `w_down` twice (96 times per window). The 1280
  tile carries all 8; each lane keeps the same chunk order, so bit-identical.
- `qsa.cu` (legacy attention, not the default path): the shared-memory check uses the limit Pascal actually launches.
- **FP16 expert kernels** (`src/kernels/cuda/pascal_experts.cu`, on by default on sm_6x; `STRATA_PASCAL_FP16=0` turns
  them off for A/B): the IQ3_S model's expert formats (gate/up IQ3_XXS, IQ3_S, IQ2_S; down IQ4_NL, Q2_0) decoded
  to half2 and multiplied with HFMA2 (P100: 2x the fp32 rate, no dp4a). Codebooks pre-converted to fp16 in shared
  memory; IQ4_NL via a byte -> half2 table against an interleaved activation. Static SASS per row vs the int8 path:
  IQ3_XXS ~1.45x, IQ3_S ~1.25x, IQ2_S ~1.15x fewer instructions, IQ4_NL down ~2x. **Not bit-exact**: fp16 products,
  sums of 32 in fp16 then fp32; the down projection reads the fp32 SwiGLU output at 11-bit precision instead of int8.
  `pascal_iq_parity` reports both paths' error against an fp64 reference and fails if fp16 is > 2x the int8 error.
  Layers with other types (the one IQ4_XS gate/up layer) keep the int8 kernels.
- Output of the other changes is intended to be **bit-identical** to upstream (same terms, same order, same FMUL+FFMA). Check on the P100:
  `build-p100/pascal_iq_parity` (synthetic, no model) and, with a model fixture, `iq_parity`.
- Not done: no measured speedup yet (no GPU here). Expected gains are modest for the expert kernels; per-token math
  suggests decode time is dominated by kernel latency / host sync, so profile with
  `nsys profile --stats=true <strata command>` before deeper work.

Verified in a GPU-less container: full build with CUDA 12.8 for sm_60 succeeds (54 sm_60 cubins, no errors);
CPU-only tests pass; GPU tests need the real cards.

Install on the server (IQ3_S: 54.8 GB of weights + a 28.8 GB n-gram shard that can stay memory-mapped on the NVMe;
matches BF16 on ISTA's AIME25 / GPQA-D / LCB v6 benchmarks):

    ./setup.sh --model IQ3_S --gpus 0,1 --build --models-dir /home/venkata/models

IQ3_S tensor types (ISTA allocation file): expert gate/up IQ2_S x20, IQ3_XXS x17, IQ3_S x10, IQ4_XS x1 layers; expert
down IQ4_NL x39, Q2_0 x9; dense mostly Q6_K / Q4_K / Q5_K / IQ4_XS / IQ4_NL. All run in the kernels tuned above.

## Using the MoE and both P100s well (this server: E5-2678 v3, 128 GB DDR4-1600, 2x P100 PCIe 3.0 x16)

How Strata runs a token here: each of the 48 layers routes to 10 of 512 experts. Experts in VRAM run on the GPU;
missed ones are either copied over PCIe and run on the GPU (a share set by a PCIe probe, ~0.25 at ~12 GB/s) or
computed by the CPU (AVX2 on this Xeon, bounded by ~35-40 GB/s of DDR4-1600). With `--gpus 0,1` the layers are
split between the cards (each holds its own copy of the dense weights and a cache of its layers' hottest experts,
~21 GB of experts in total vs ~11 GB on one card); a verify window runs card 0's layers, then card 1's.

Fixed in this fork (Linux):
- **The whole expert arena is pinned with two GPUs.** Upstream capped pinning at 8 GiB whenever more than one GPU was
  used - a Windows (WDDM) workaround applied on Linux too. With IQ3_S that left ~7 of 48 layers pinned: every other
  layer had **no PCIe share** (all misses on the CPU) and its adaptive cache swaps copied from pageable memory.
  `STRATA_PIN_LIMIT_GIB=N` restores a cap if pinning ~50 GB ever causes trouble.
- **Transparent huge pages for the arena** when no hugetlb pool is reserved (Ubuntu's THP mode is `madvise`), so the
  CPU expert kernels stop missing the TLB across 50 GB.

After installing, check the startup log:
- `expert arena: cudaHostRegister PORTABLE ok; ... MADV_HUGEPAGE` - the whole arena is pinned and THP-backed
- `PCIe probe: ~12 GB/s -> pcie_frac ...` on each card
- `layer split auto: K=..`, `expert cache auto: .. slots` per card, `decode expert cache hit rate` per request

Then tune for this CPU/PCIe balance (the defaults were measured on a Ryzen 7600 + RTX 5070 on PCIe 5):

    ./setup.sh --calibrate          # measures --pcie-frac, --spec-min-p, --pool-workers and saves the winners

Optional:
- n-gram table in RAM instead of NVMe reads: add `"--ple-io", "ram"` to `"args"` in `strata-*.json` (28.8 GB locked;
  arena 50 GB + table 29 GB + your other services must fit in 125 GB).
- An expert profile from your own traffic (better VRAM hit rate for your workload): run with
  `"--dump-routing", "routing.bin"` for a while, then `python tools/make_profile.py` (see its header) and point
  `--expert-profile` at the result.

Measure every change at once: `.venv/bin/python pascal/ab_bench.py` restarts the engine once per variant (base =
fork's switchable changes off, pin, fp16, fork, devplan, spec6, spec8) and prints tok/s per variant vs base.

Where the time goes (send me this): add `"env": {"STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1"}` to the
config, send one ~500-token request, and copy the per-window timing lines (GPU wait vs CPU pool vs staging per card).
