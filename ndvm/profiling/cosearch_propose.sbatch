#!/bin/bash
#SBATCH --job-name=cosearch_propose
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=240:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/propose_%j.out
#SBATCH --error=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/propose_%j.err
set -e
module load gcc/12.1.0
REPO=/mnt/ceph/sheneman/src/nncompile-ndvm; cd $REPO
$REPO/.venv/bin/python ndvm/profiling/cosearch_propose.py --stop-valid 250 --patience 300 --workers 12 \
    --model qwen/qwen3.6-27b --out ndvm/profiling/results/cosearch_candidates_valid.jsonl
