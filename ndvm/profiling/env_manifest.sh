#!/usr/bin/env bash
# Capture the exact measurement environment for a reproducible NDVM benchmark (MLSys artifact requirement).
# Run on the SAME node as the measurement (an HPC compute node, not the login node). Emits a manifest to
# stdout and to the path given as $1 (default env_manifest.txt). Every runtime figure in the paper should
# cite the manifest captured on its node.
set -uo pipefail
out="${1:-env_manifest.txt}"
here="$(cd "$(dirname "$0")" && pwd)"

emit() {
  echo "# NDVM measurement environment manifest"
  echo "captured_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"

  echo; echo "## source"
  echo "git_commit: $(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git_dirty_files: $(git -C "$here" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"

  echo; echo "## cpu"
  if command -v lscpu >/dev/null; then
    lscpu | grep -iE "model name|^cpu\(s\)|thread|core|socket|cache|mhz|numa node\(s\)|flags" | sed 's/^/  /'
  else sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/^/  model: /'; fi
  echo "  governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"

  echo; echo "## memory + numa"
  free -h 2>/dev/null | sed 's/^/  /' || echo "  (free unavailable)"
  command -v numactl >/dev/null && numactl --hardware 2>/dev/null | grep -iE "available|node . size" | sed 's/^/  /'

  echo; echo "## compilers"
  for c in gcc g++ clang++ nvcc; do command -v $c >/dev/null && echo "  $c: $($c --version 2>/dev/null | head -1)"; done
  echo "  cmake_build_flags: see ndvm/CMakeLists.txt (Release = -O3); ext: ndvm/python/setup.py"

  echo; echo "## blas / numpy / torch"
  PY="${NDVM_PY:-python3}"
  $PY - <<'PYEOF' 2>/dev/null | sed 's/^/  /' || echo "  (python introspection unavailable)"
import os
try:
    import numpy as np
    cfg = []
    np.__config__.show() if False else None
    print("numpy:", np.__version__)
    try: print("numpy_blas:", (np.__config__.get_info("blas_opt_info") or {}).get("libraries"))
    except Exception: pass
except Exception as e: print("numpy: n/a", e)
try:
    import torch
    print("torch:", torch.__version__, "| threads:", torch.get_num_threads())
    print("torch_parallel_info_head:", torch.__config__.parallel_info().splitlines()[0])
except Exception as e: print("torch: n/a", e)
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
PYEOF

  echo; echo "## os"
  uname -a | sed 's/^/  /'
  [ -f /etc/os-release ] && grep -E "^PRETTY_NAME" /etc/os-release | sed 's/^/  /'

  echo; echo "## perf access (for hardware-counter runs)"
  echo "  perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"
  command -v perf >/dev/null && echo "  perf: $(perf --version 2>/dev/null)" || echo "  perf: not found"
}

emit | tee "$out"
