#!/bin/bash
#SBATCH --job-name=cosearch_rec
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/cosearch_rec_%j.out
#SBATCH --error=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/cosearch_rec_%j.err
set -e
cd /mnt/ceph/sheneman/src/nncompile-ndvm
module load gcc/12.1.0 || true
PY=/mnt/ceph/sheneman/src/nncompile-ndvm/.venv/bin/python

echo "=== HOST $(hostname) $(date) ==="
echo "=== STAGE 1: propose + compile-validate recurrence-heavy candidates (LLM, offline) ==="
$PY ndvm/profiling/cosearch_rec_propose.py --stop-valid 80 --patience 400 --workers 16 \
    --model qwen/qwen3.6-27b \
    --out ndvm/profiling/results/cosearch_rec_candidates_valid.jsonl

echo "=== STAGE 2: fixed-budget replay (DMCI vs NDVM), 900s/backend ==="
$PY ndvm/profiling/cosearch_rec_e2e.py --run \
    --infile ndvm/profiling/results/cosearch_rec_candidates_valid.jsonl \
    --budget 900 --iters 50

echo "=== DONE $(date) ==="
wc -l ndvm/profiling/results/cosearch_rec_candidates_valid.jsonl || true
