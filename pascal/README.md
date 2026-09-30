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
- Output is intended to be **bit-identical** to upstream (same terms, same order, same FMUL+FFMA). Check on the P100:
  `build-p100/pascal_iq_parity` (synthetic, no model) and, with a model fixture, `iq_parity`.
- Not done: no measured speedup yet (no GPU here). Expected gains are modest for the expert kernels; per-token math
  suggests decode time is dominated by kernel latency / host sync, so profile with
  `nsys profile --stats=true <strata command>` before deeper work.

Verified in a GPU-less container: full build with CUDA 12.8 for sm_60 succeeds (54 sm_60 cubins, no errors);
CPU-only tests pass; GPU tests need the real cards.

Normal install on the server: `./setup.sh --gpus 0,1` (add `--models-dir DIR` for the NVMe).
