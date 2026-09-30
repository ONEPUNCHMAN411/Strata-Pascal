#!/usr/bin/env bash
# Build Strata for Tesla P100 (sm_60) with a CUDA 12.x toolkit (CUDA 13 dropped Pascal), then run strata-device
# on every GPU so the compute-capability check and memory report are exercised before any model is loaded.
#   CUDA_HOME=/usr/local/cuda-12.8 ./pascal/build-p100.sh [build-dir] [--with-engine]
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -d .venv/bin ]] && export PATH="$PWD/.venv/bin:$PATH"   # setup.py installs cmake and ninja there
BUILD=build-p100
[[ -n ${1:-} && ${1:-} != --* ]] && BUILD=$1
if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -1 || true)
fi
NVCC="${CUDA_HOME:-}/bin/nvcc"
[[ -x "$NVCC" ]] || { echo "no CUDA 12.x nvcc found; install one: sudo apt-get install cuda-toolkit-12-8"; exit 1; }
VER=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
[[ ${VER%%.*} -eq 12 ]] || { echo "nvcc $VER at $NVCC: Pascal needs CUDA 12.x"; exit 1; }
echo "== nvcc $VER ($NVCC)"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader

cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_COMPILER="$NVCC"
# the device check and the Pascal parity tests (setup.py --build compiles the engine itself)
TARGETS=()
ALL=$(ninja -C "$BUILD" -t targets all 2>/dev/null || true)
for t in strata-device pascal_iq_parity pascal_dense_parity; do
  grep -q "^$t:" <<<"$ALL" && TARGETS+=("$t")
done
[[ ${1:-} == --with-engine || ${2:-} == --with-engine ]] && TARGETS+=(strata)
cmake --build "$BUILD" --target "${TARGETS[@]}" -j "$(nproc)"

FAIL=0
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  for t in "${TARGETS[@]}"; do
    [[ $t == strata ]] && continue
    echo "== $t on GPU $i"
    CUDA_VISIBLE_DEVICES=$i "$BUILD/$t" || { echo "!! $t FAILED on GPU $i"; FAIL=1; }
  done
done
echo "== GPU tests: $([[ $FAIL == 0 ]] && echo all passed || echo SOME FAILED)"
exit $FAIL
