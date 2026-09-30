# Strata on Tesla P100 (sm_60)

Upstream Strata carries an experimental Pascal build (`-DSTRATA_EXPERIMENTAL_SM60=ON`, commit `01bca8a`). This fork adds:

- `setup.py`: accepts compute capability 6.x cards, always compiles for them (no ready-made sm_6x engine), and uses a
  CUDA 12.x toolkit (`cuda-toolkit-12-8`) because CUDA 13 dropped Pascal. A Pascal card cannot share an engine with an RTX 50.
- `src/core/device.cu`: the device check (`strata-device`, self-tests) accepts 6.0 in a Pascal build.
- `pascal/build-p100.sh`: standalone sm_60 build + `strata-device` on each GPU.
- `pascal/server-checks.sh`: `status` (stray processes, GPUs, EDAC per DIMM, mounts), `coldload FILE..` (uncached read
  timing), `crashrun CMD..` (run with core dumps + dmesg capture, for the BigBang-MTP segfault).

Verified in a GPU-less container: full build with CUDA 12.8 for sm_60 succeeds (54 sm_60 cubins, no errors);
CPU-only tests pass; GPU tests need the real cards.

Normal install on the server: `./setup.sh --gpus 0,1` (add `--models-dir DIR` for the NVMe).
