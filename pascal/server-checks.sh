#!/usr/bin/env bash
# P100 server follow-ups from the NVMe/DIMM maintenance. Read-only except `sudo drop_caches` in the load test.
#   ./pascal/server-checks.sh status                 GPU/process/memory-error/mount state
#   ./pascal/server-checks.sh coldload FILE [FILE..]  timed uncached read of model files (NVMe vs old HDD copy)
#   ./pascal/server-checks.sh crashrun CMD...         run CMD (e.g. the BigBang-MTP launch) with core dumps + logs
#   ./pascal/server-checks.sh hwinfo                  CPU / NUMA / PCIe / memory facts that decide Strata's placement
set -uo pipefail
LOG_DIR=${LOG_DIR:-$HOME/p100-checks}; mkdir -p "$LOG_DIR"
ts() { date +%Y%m%d-%H%M%S; }

status() {
  echo "== r3_wikilm processes (expect none)"; pgrep -af r3_wikilm || echo "none"
  echo "== GPUs"; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  echo "== services"; systemctl --no-pager --type=service --state=failed | head -20
  echo "== memory errors (EDAC, per DIMM)"
  if [[ -d /sys/devices/system/edac/mc ]]; then
    grep -H . /sys/devices/system/edac/mc/mc*/dimm*/dimm_{ce,ue}_count 2>/dev/null | sed 's|/sys/devices/system/edac/mc/||'
  else echo "EDAC not loaded"; fi
  echo "== mounts"; findmnt /home/venkata/models; df -h /home/venkata/models /home/venkata/models_old_hdd 2>/dev/null
}

coldload() {
  [[ $# -gt 0 ]] || { echo "usage: coldload FILE [FILE..]"; exit 1; }
  local out="$LOG_DIR/coldload-$(ts).txt"
  for f in "$@"; do
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
    local sz; sz=$(stat -c %s "$f")
    local t0 t1; t0=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M status=none; t1=$(date +%s.%N)
    awk -v f="$f" -v s="$sz" -v a="$t0" -v b="$t1" \
      'BEGIN{d=b-a; printf "%-70s %8.2f GB %7.2f s %8.1f MB/s\n", f, s/1e9, d, s/1e6/d}' | tee -a "$out"
  done
  echo "saved: $out"
}

crashrun() {
  [[ $# -gt 0 ]] || { echo "usage: crashrun CMD..."; exit 1; }
  local tag; tag="$LOG_DIR/crash-$(ts)"
  ulimit -c unlimited
  sudo dmesg --clear 2>/dev/null || true
  echo "== running: $*" | tee "$tag.log"
  "$@" >>"$tag.log" 2>&1; local rc=$?
  echo "== exit code $rc" | tee -a "$tag.log"
  if (( rc >= 128 )); then
    echo "== killed by signal $((rc-128)); core dumps:" | tee -a "$tag.log"
    coredumpctl list --no-pager 2>/dev/null | tail -3 | tee -a "$tag.log"
    echo "   backtrace: coredumpctl gdb   (then: bt, info sharedlibrary)"
  fi
  sudo dmesg 2>/dev/null | grep -Ei 'segfault|EDAC|mce|NVRM|Xid' | tee "$tag.dmesg"
  echo "saved: $tag.log $tag.dmesg"
}

hwinfo() {
  local out="$LOG_DIR/hwinfo-$(ts).txt"
  {
    echo "== CPU"; lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|MHz|L3'
    echo "avx512: $(grep -o -m1 'avx512[a-z]*' /proc/cpuinfo | sort -u | tr '\n' ' ')  avx2: $(grep -c -m1 avx2 /proc/cpuinfo)"
    echo "== NUMA"; numactl --hardware 2>/dev/null || echo "numactl not installed (sudo apt install numactl)"
    echo "== memory"; free -g; grep -E 'HugePages_Total|Hugepagesize' /proc/meminfo
    sudo dmidecode -t memory 2>/dev/null | grep -E '^\s+(Size|Speed|Configured Memory Speed|Locator):' | paste - - - - | head -16
    echo "== GPUs and PCIe links"
    nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.total --format=csv
    echo "== GPU <-> CPU topology"; nvidia-smi topo -m
    echo "== P2P"; nvidia-smi topo -p2p r 2>/dev/null
    echo "== NVMe"; lsblk -d -o NAME,MODEL,SIZE,ROTA | grep -v loop
    echo "== driver / CUDA"; nvidia-smi | head -4 | tail -2; ls -d /usr/local/cuda-* 2>/dev/null
  } | tee "$out"
  echo "saved: $out"
}

cmd=${1:-status}; shift || true
case "$cmd" in status|coldload|crashrun|hwinfo) "$cmd" "$@" ;; *) sed -n 2,5p "$0"; exit 1 ;; esac
