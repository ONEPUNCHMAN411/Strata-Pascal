#!/usr/bin/env bash
# Build Strata for Tesla P100 (sm_60) with a CUDA 12.x toolkit (CUDA 13 dropped Pascal), then run strata-device
# on every GPU so the compute-capability check and memory report are exercised before any model is loaded.
#   CUDA_HOME=/usr/local/cuda-12.8 ./pascal/build-p100.sh [build-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build-p100}
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
cmake --build "$BUILD" --target strata strata-device pascal_iq_parity -j "$(nproc)"

for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  echo "== strata-device on GPU $i"
  CUDA_VISIBLE_DEVICES=$i "$BUILD/strata-device" || echo "!! strata-device failed on GPU $i"
  CUDA_VISIBLE_DEVICES=$i "$BUILD/pascal_iq_parity" || echo "!! pascal_iq_parity failed on GPU $i"
done
echo "== built: $BUILD/strata"
