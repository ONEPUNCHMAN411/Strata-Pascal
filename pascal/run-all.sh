#!/usr/bin/env bash
# Everything on the P100 server in one go: update, hardware report, build + GPU correctness tests, install/update
# IQ3_S across both cards, calibrate, A/B benchmark of every change (the winning options are saved into the config,
# with a backup), one profiled request - into one report folder.
#
#   cd ~/Strata-Pascal && ./pascal/run-all.sh
#
# Takes ~1-2 h (plus the 84 GB IQ3_S download the first time; it resumes if interrupted). Safe to re-run: finished
# steps (download, build) are skipped. Env: MODELS_DIR (default /home/venkata/models).
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS_DIR=${MODELS_DIR:-/home/venkata/models}
R="$HOME/p100-report-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$R"
exec > >(tee "$R/run.log") 2>&1
step() { echo; echo "=================== $* ($(date +%H:%M:%S))"; }
ok=1

step "1/7 update the code"
git pull --ff-only || echo "!! git pull failed - continuing with the current checkout"
git log --oneline -1

step "2/7 hardware and server state"
LOG_DIR="$R" ./pascal/server-checks.sh hwinfo
LOG_DIR="$R" ./pascal/server-checks.sh status
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if (( used > 1500 )); then
  echo "!! a GPU already has ${used} MiB in use - other services on the cards shrink Strata's expert cache and skew"
  echo "   the benchmark. Stop them first (e.g. the router / llama services) and re-run for clean numbers."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

step "3/7 install / update IQ3_S on GPUs 0,1 (compiles the engine; downloads the model if missing)"
if [[ -x .venv/bin/python ]] && ls strata-*.json >/dev/null 2>&1; then
  echo "an install exists: re-running setup so the engine is recompiled from this code"
fi
./setup.sh --model IQ3_S --gpus 0,1 --build --vision no --models-dir "$MODELS_DIR" --yes --no-start \
  || { echo "!! setup failed - stopping here"; ok=0; }

step "4/7 build and GPU correctness tests"
./pascal/build-p100.sh build-p100 || { echo "!! GPU tests failed or build broke (see above)"; ok=0; }

if (( ok )); then
  step "5/7 calibrate (PCIe share, draft floor, CPU workers) for this machine"
  ./setup.sh --calibrate --build --yes --no-start || echo "!! calibration failed - benchmarking with defaults"

  step "6/7 A/B benchmark of every change + profiled request"
  .venv/bin/python pascal/ab_bench.py --out "$R" --profile --apply || echo "!! benchmark failed"
fi

step "7/7 collect"
cp strata-*.json "$R"/ 2>/dev/null
{
  echo "== commit: $(git log --oneline -1)"
  for f in "$R"/engine-fork.log "$R"/profile.log; do
    [[ -f $f ]] || continue
    echo "== $(basename "$f")"
    grep -E "expert arena|PCIe probe|layer split|expert cache auto|pre-filled|MiB of VRAM free|pool workers|window up to|hit rate|tok/s" "$f" | head -40
  done
  ls "$R"/ab_results-*.json >/dev/null 2>&1 && cat "$R"/ab_results-*.json
} > "$R/summary.txt"
tar czf "$R.tar.gz" -C "$(dirname "$R")" "$(basename "$R")"
echo
echo "Done. Send back:  $R.tar.gz   (or paste $R/summary.txt)"
