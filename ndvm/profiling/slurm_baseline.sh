#!/bin/bash
#SBATCH --job-name=ndvm_phase0
#SBATCH --partition=sheneman
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=ndvm/profiling/results/phase0_%j.out
#SBATCH --error=ndvm/profiling/results/phase0_%j.err

# NDVM Phase 0 baseline: lock the current PyTorch DMCI cost model on an HPC compute node.
# Runs in the ISOLATED NDVM checkout (/mnt/ceph/sheneman/src/nncompile-ndvm) -- never the ICLR
# tree (/mnt/ceph/sheneman/src/nncompile). Excludes n113 (active ICLR job) + known-bad nodes.
#
# The venv was copied from the ICLR tree; invoke its python by ABSOLUTE PATH rather than
# `source activate` (activate bakes in the original VIRTUAL_ENV path). neural_compiler is NOT
# installed into the venv, so it imports from this dir (REPO_ROOT is prepended in the profiler).

set -euo pipefail
ROOT=/mnt/ceph/sheneman/src/nncompile-ndvm
SRC=/mnt/ceph/sheneman/src/nncompile
cd "$ROOT"
# Prefer the fully-isolated copied venv; if the copy is still finalizing, fall back to READ-ONLY
# use of the source venv. Read-only use on a non-ICLR node with no bytecode writes does not impact
# the concurrent ICLR job. pyvenv.cfg is among the last files copied, so its presence => copy done.
if [ -f "$ROOT/.venv/pyvenv.cfg" ]; then
  PY="$ROOT/.venv/bin/python"; echo "venv: isolated copy ($PY)"
else
  PY="$SRC/.venv/bin/python"; echo "venv: source READ-ONLY fallback ($PY) -- isolated copy not ready"
fi
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1     # do not scatter .pyc into the copied venv
export OMP_NUM_THREADS=8
mkdir -p ndvm/profiling/results

echo "host=$(hostname)  python=$($PY -c 'import platform;print(platform.python_version())')  torch=$($PY -c 'import torch;print(torch.__version__)')"

# Full baseline + per-bucket decomposition (B=1). Batches probe the overhead-bound / D-independent claim.
$PY -u ndvm/profiling/profile_dmci_baseline.py \
    --iters 30 \
    --batches 1 8 64 256 1024 \
    --kalman-T 80 \
    --loop-n 16 \
    --decompose
