#!/bin/bash
#SBATCH --job-name=compiled_kalman
#SBATCH --partition=sheneman
#SBATCH --nodelist=n128
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=15:00
#SBATCH --output=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/compiled_%j.out
#SBATCH --error=/mnt/ceph/sheneman/src/nncompile-ndvm/ndvm/profiling/results/compiled_%j.err
set -e
module load gcc/12.1.0
REPO=/mnt/ceph/sheneman/src/nncompile-ndvm
cd $REPO/ndvm
PY=$REPO/.venv/bin/python
CK=/tmp/compiled_kalman_$$
ND=/tmp/ndvm_run_$$
WORK=/tmp/ck_$$; mkdir -p $WORK
RES=$REPO/ndvm/profiling/results
echo "host=$(hostname)"; g++ --version | head -1
g++ -std=c++17 -O3 tools/compiled_kalman.cpp -o $CK
g++ -std=c++17 -O3 -Isrc src/sexpr.cpp src/interp.cpp src/interp_linalg.cpp src/interp_tape.cpp tools/ndvm_run.cpp -o $ND
$PY profiling/dump_programs.py $WORK
KB=$WORK/kalman2d_T80.binds; KP=$WORK/kalman2d_T80.scm
echo "=== compiled C++ Kalman (forward + exact dual gradient) ==="
CKOUT=$($CK $KB); echo "$CKOUT"
echo "=== NDVM reverse-mode gradient (reference) ==="
NDG=$(NDVM_GRAD=1 $ND $KP $KB 2>/dev/null | grep -E '^grad|^result'); echo "$NDG"
echo "=== timing (median of 5), n128 ==="
ckf=$(for i in 1 2 3 4 5; do COMPILED_BENCH=200000 COMPILED_MODE=fwd  $CK $KB | awk '/^bench/{print $3}'; done | sort -g | sed -n '3p')
ckg=$(for i in 1 2 3 4 5; do COMPILED_BENCH=100000 COMPILED_MODE=grad $CK $KB | awk '/^bench/{print $3}'; done | sort -g | sed -n '3p')
echo "compiled_fwd_ms=$ckf  compiled_fwdgrad_ms=$ckg"
RESF=$RES/compiled_kalman_n128.txt
{
  echo "# compiled-differentiable C++ Kalman (program-as-code; hand-coded fwd + forward-mode dual grad), host=$(hostname), $(date)"
  echo "# forward+grad output:"; echo "$CKOUT"
  echo "# NDVM reverse-mode gradient reference:"; echo "$NDG"
  echo "# timing median-of-5 (ms/eval):"
  echo "compiled_fwd_ms  $ckf"
  echo "compiled_fwdgrad_ms  $ckg"
} > $RESF
echo "DONE"; cat $RESF
rm -f $CK $ND; rm -rf $WORK
