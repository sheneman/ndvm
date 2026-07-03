#!/bin/bash
#SBATCH --job-name=ndvm_mc_med
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/mcmed_%j.out
#SBATCH --error=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/mcmed_%j.err
set -e
module load gcc/12.1.0
REPO=/mnt/ceph/sheneman/src/nncompile-ndvm; cd $REPO/ndvm
PY=$REPO/.venv/bin/python
PAR=/tmp/ndvm_par_$$; WORK=/tmp/mc_$$; mkdir -p $WORK
echo "host=$(hostname)"
g++ -std=c++17 -O3 -pthread -Isrc src/sexpr.cpp src/interp.cpp src/interp_linalg.cpp src/interp_tape.cpp src/parallel.cpp tools/ndvm_par.cpp -o $PAR
$PY profiling/dump_programs.py $WORK
KP=$WORK/kalman2d_T80.scm; KB=$WORK/kalman2d_T80.binds
RES=$REPO/ndvm/profiling/results/multicore_scaling_n128.txt
{
echo "# NDVM multicore candidate-level scaling, Kalman NLL+grad, NDVM_PAR_N=16000, MEDIAN of 5 sweeps, host=$(hostname), $(date)"
echo "# threads  median_wall_ms  speedup_vs_1  efficiency"
} > $RES
declare -A MED
for T in 1 2 4 8 16 32 64; do
  w=$(for rep in 1 2 3 4 5; do NDVM_THREADS=$T NDVM_PAR_N=16000 $PAR $KP $KB 2>/dev/null | sed -n 's/.*wall=\([0-9.]*\) ms.*/\1/p'; done | sort -g | sed -n '3p')
  MED[$T]=$w
  sp=$($PY -c "print(f'{${MED[1]}/$w:.2f}')")
  ef=$($PY -c "print(f'{100*(${MED[1]}/$w)/$T:.1f}')")
  printf '%-8s %-15s %-12s %s%%\n' "$T" "$w" "$sp" "$ef" >> $RES
done
echo "DONE"; cat $RES
rm -f $PAR; rm -rf $WORK
