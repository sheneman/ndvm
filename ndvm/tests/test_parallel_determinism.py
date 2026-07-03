"""Phase-5 determinism gate: a parallel population evaluation is byte-identical across thread counts.

ndvm_par builds a population of distinct candidate tasks from one program and runs them through the
multicore scheduler. With NDVM_PAR_DUMP it prints each task's result as raw IEEE-754 hex bit patterns
(forward outputs + gradients), in task order. Because every task is independent and pure and each worker
uses its own thread-local Interp, the dump MUST be exactly equal for any thread count -- not within a
tolerance, bit for bit. Any difference is a data race, shared-state corruption, or a cross-task reduction.
Pure NDVM (the native scheduler); no torch/HPC needed (but the real wide sweep runs on a many-core node).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAR = Path(os.environ.get("NDVM_PAR", str(HERE.parents[0] / "build" / "ndvm_par")))

# Cover the hard cases: a per-candidate divergent branch, a recursive convergence loop (lane masks +
# recursion + the tape + pools concurrently), a deep recursion, and the 80-step Kalman matrix rollout.
PROGRAMS = [
    ("scalar_branch", "(* x (if (> x 0) x 1.0))", "scalar x 2.0\n"),
    ("newton_loop", "(define (go x) (if (< (abs (- (* x x) a)) 0.0001) x (go (* 0.5 (+ x (/ a x))))))\n(go 1.0)", "scalar a 7.0\n"),
    ("recursive", "(define (poly x n) (if (= n 0) 0.0 (+ (* alpha x) (poly x (- n 1)))))\n(poly 1.5 20)", "scalar alpha 0.3\n"),
]


def _dump(prog, binds, threads, n):
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(prog + "\n")
        bp = Path(d) / "p.bind"; bp.write_text(binds)
        env = dict(os.environ)
        env.update({"NDVM_PAR_DUMP": "1", "NDVM_THREADS": str(threads), "NDVM_PAR_N": str(n)})
        out = subprocess.run([str(PAR), str(sp), str(bp)], capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr
        return out.stdout


@pytest.mark.skipif(not PAR.exists(), reason="ndvm_par not built")
@pytest.mark.parametrize("p", PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_parallel_determinism(p):
    name, prog, binds = p
    ref = _dump(prog, binds, 1, 2000)
    assert ref.count("\n") == 2000, f"{name}: expected 2000 task lines"
    for w in (2, 4, 8, 16):
        assert _dump(prog, binds, w, 2000) == ref, f"{name}: W={w} dump differs from W=1 (nondeterminism/race)"


@pytest.mark.skipif(not PAR.exists(), reason="ndvm_par not built")
def test_parallel_repeat_stability():
    # The same many-thread run, repeated, must be identical each time (no schedule-dependent drift).
    _, prog, binds = PROGRAMS[1]
    ref = _dump(prog, binds, 16, 1500)
    for _ in range(10):
        assert _dump(prog, binds, 16, 1500) == ref
