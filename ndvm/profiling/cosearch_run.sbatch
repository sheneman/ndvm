#!/bin/bash
#SBATCH --job-name=cosearch_run
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=45:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/cosearch_run_%j.out
set -e
module load gcc/12.1.0
REPO=/mnt/ceph/sheneman/src/nncompile-ndvm; cd $REPO
$REPO/.venv/bin/python ndvm/profiling/cosearch_e2e.py --run \
  --infile ndvm/profiling/results/cosearch_candidates_valid.jsonl --budget 900
