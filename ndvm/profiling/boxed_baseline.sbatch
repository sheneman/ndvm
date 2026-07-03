#!/bin/bash
#SBATCH --job-name=boxed_baseline
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/boxed_%j.out
#SBATCH --error=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/boxed_%j.err
set -e
module load gcc/12.1.0
REPO=/mnt/ceph/sheneman/src/nncompile-ndvm
cd $REPO/ndvm
PY=$REPO/.venv/bin/python
BX=/tmp/boxed_run_$$
ND=/tmp/ndvm_run_$$
WORK=/tmp/bx_$$
mkdir -p $WORK $REPO/ndvm/profiling/results
echo "host=$(hostname)"; g++ --version | head -1
g++ -std=c++17 -O3 -Isrc src/sexpr.cpp tools/boxed_run.cpp -o $BX
g++ -std=c++17 -O3 -Isrc src/sexpr.cpp src/interp.cpp src/interp_linalg.cpp src/interp_tape.cpp tools/ndvm_run.cpp -o $ND
echo "built boxed_run + ndvm_run"
$PY profiling/dump_programs.py $WORK
PROGS="scalar_mul_add michaelis_menten damped_oscillator logistic_map_loop kalman2d_T80"
declare -A N=( [scalar_mul_add]=20000 [michaelis_menten]=20000 [damped_oscillator]=20000 [logistic_map_loop]=5000 [kalman2d_T80]=300 )
RES=$REPO/ndvm/profiling/results/boxed_baseline_n128.txt
{
echo "# boxed-C++ (no split) vs NDVM-C++ (split) forward, host=$(hostname), $(date)"
echo "# columns: program | boxed_result | ndvm_result | boxed_ms(median5) | ndvm_ms(median5) | N"
} > $RES
for p in $PROGS; do
  bxr=$($BX $WORK/$p.scm $WORK/$p.binds 2>/dev/null | awk '/^result|^vec/{print $2; exit}')
  ndr=$($ND $WORK/$p.scm $WORK/$p.binds 2>/dev/null | awk '/^result|^vec/{print $2; exit}')
  n=${N[$p]}
  bxms=$(for i in 1 2 3 4 5; do BOXED_BENCH=$n $BX $WORK/$p.scm $WORK/$p.binds 2>/dev/null | awk '{print $2}'; done | sort -g | sed -n '3p')
  ndms=$(for i in 1 2 3 4 5; do NDVM_BENCH=$n $ND $WORK/$p.scm $WORK/$p.binds 2>/dev/null | awk '{print $2}'; done | sort -g | sed -n '3p')
  printf '%-20s  boxed=%-14s ndvm=%-14s  boxed_ms=%-12s ndvm_ms=%-12s N=%s\n' "$p" "$bxr" "$ndr" "$bxms" "$ndms" "$n"
  printf '%s  %s  %s  %s  %s  %s\n' "$p" "$bxr" "$ndr" "$bxms" "$ndms" "$n" >> $RES
done
echo "DONE; results:"; cat $RES
rm -f $BX $ND; rm -rf $WORK
