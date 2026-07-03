#!/usr/bin/env bash
# One-command smoke test for the NDVM native runtime: builds the C++ interpreter and checks forward values
# and a reverse-mode gradient against known answers, in a few seconds, with NO Python/torch dependency.
# This is the fast "does it work at all" check; the full artifact (forward + gradient equivalence vs the
# PyTorch DMCI oracle, batching, fuzzing) is in ndvm/profiling/ and ndvm/tests/ per the REPRODUCE.md artifact guide.
#
#   bash ndvm/smoke_test.sh
#
# Exit code 0 = PASS. Uses g++ if available (the deployment compiler), else clang++.
set -u
cd "$(dirname "$0")"                       # ndvm/
CXX=$(command -v g++ || command -v clang++)
[ -z "$CXX" ] && { echo "SMOKE FAIL: no C++ compiler"; exit 1; }
BIN=$(mktemp -u /tmp/ndvm_smoke.XXXX)
WORK=$(mktemp -d /tmp/ndvm_smoke.XXXX.d)
trap 'rm -f "$BIN"; rm -rf "$WORK"' EXIT

echo "building native runtime with $CXX ..."
"$CXX" -std=c++17 -O2 -Isrc src/sexpr.cpp src/interp.cpp src/interp_linalg.cpp src/interp_tape.cpp \
    tools/ndvm_run.cpp -o "$BIN" || { echo "SMOKE FAIL: build error"; exit 1; }

fail=0
check() {  # check <name> <expected> <got>
  if awk -v e="$2" -v g="$3" 'BEGIN{d=e-g; if(d<0)d=-d; exit !(d<1e-3)}'; then
    echo "  PASS $1 (= $3)"
  else
    echo "  FAIL $1: expected $2 got $3"; fail=1
  fi
}

# forward: (+ (* alpha x) beta), alpha=2 x=1.5 beta=1 -> 4.0 ; Michaelis-Menten -> 1.5
printf '(+ (* alpha x) beta)\n' > "$WORK/p1.scm"
printf 'scalar alpha 2.0\nscalar x 1.5\nscalar beta 1.0\n' > "$WORK/p1.binds"
r=$("$BIN" "$WORK/p1.scm" "$WORK/p1.binds" 2>/dev/null | awk '/^result/{print $2}')
check "forward scalar"    4.0 "$r"

printf '(/ (* Vmax S) (+ Km S))\n' > "$WORK/p2.scm"
printf 'scalar Vmax 2.0\nscalar S 1.5\nscalar Km 0.5\n' > "$WORK/p2.binds"
r=$("$BIN" "$WORK/p2.scm" "$WORK/p2.binds" 2>/dev/null | awk '/^result/{print $2}')
check "forward michaelis" 1.5 "$r"

# reverse-mode gradient: d/dalpha (+ (* alpha x) beta) = x = 1.5 ; d/dbeta = 1.0
ga=$(NDVM_GRAD=1 "$BIN" "$WORK/p1.scm" "$WORK/p1.binds" 2>/dev/null | awk '/^grad alpha/{print $3}')
gb=$(NDVM_GRAD=1 "$BIN" "$WORK/p1.scm" "$WORK/p1.binds" 2>/dev/null | awk '/^grad beta/{print $3}')
check "grad d/dalpha"     1.5 "$ga"
check "grad d/dbeta"      1.0 "$gb"

if [ "$fail" -eq 0 ]; then echo "SMOKE PASS"; exit 0; else echo "SMOKE FAIL"; exit 1; fi
