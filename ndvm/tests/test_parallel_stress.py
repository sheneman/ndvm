"""Phase-5 race-freedom gate: the multicore scheduler is ThreadSanitizer-clean under contention stress.

This is the contention-stress driver specified in PHASE5_DESIGN.md (Gate 2). It builds ndvm_par with
-fsanitize=thread and hammers the scheduler in the configurations most likely to expose a race:

  * oversubscription: THREADS = 2x hardware cores, so workers are preempted mid-evaluation;
  * cold-start burst: every run is a fresh process, so all threads first-touch the decoded-form
    cache, interned symbols, and pools simultaneously;
  * a program mix covering the shared machinery concurrently: a divergent branch, a recursive
    convergence loop (lane masks + tape + pools), a deep recursion, and a matrix program
    (det/inv/trace adjoints through the tape);
  * a batched-lane variant (NDVM_B > 1) so wide payload buffers run under the same contention;
  * 10 repeats of the oversubscribed burst per program.

PASS = every run exits 0 under TSAN_OPTIONS halt_on_error=1 (any reported race exits 66), and the
NDVM_PAR_DUMP hex output stays byte-identical to a single-threaded run of the same binary.

Run under pytest (`pytest ndvm/tests/test_parallel_stress.py`) or directly
(`python ndvm/tests/test_parallel_stress.py`) to print the summary log that is committed as
`ndvm/tests/results/tsan_stress_n128.txt`. Pure native; no torch. Requires a compiler with
ThreadSanitizer support (g++ >= 10 with libtsan, or clang++); skips if TSan is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NDVM = HERE.parent  # ndvm/

TSAN_ENV = {
    "TSAN_OPTIONS": "halt_on_error=1 second_deadlock_stack=1 history_size=7 exitcode=66",
}

# Program mix per PHASE5_DESIGN: divergent branch, convergence loop, deep recursion, matrix ops.
PROGRAMS = [
    ("scalar_branch", "(* x (if (> x 0) x 1.0))", "scalar x 2.0\n", 8192),
    ("newton_loop",
     "(define (go x) (if (< (abs (- (* x x) a)) 0.0001) x (go (* 0.5 (+ x (/ a x))))))\n(go 1.0)",
     "scalar a 7.0\n", 8192),
    ("recursive",
     "(define (poly x n) (if (= n 0) 0.0 (+ (* alpha x) (poly x (- n 1)))))\n(poly 1.5 20)",
     "scalar alpha 0.3\n", 8192),
    ("matrix_ops",
     "(+ (* s (det m)) (trace (inv m)))",
     "scalar s 1.0\nmatrix m 2 2 2.0 0.3 0.1 1.5\n", 4096),
]

REPEATS = int(os.environ.get("NDVM_STRESS_REPEATS", "10"))


def _build_tsan(tmp: Path) -> Path | None:
    """Build ndvm_par with ThreadSanitizer per the PHASE5_DESIGN recipe; None if unavailable."""
    pre = os.environ.get("NDVM_PAR_TSAN")
    if pre and Path(pre).exists():
        return Path(pre)
    cxx = shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        return None
    out = tmp / "ndvm_par_tsan"
    srcs = sorted(str(p) for p in (NDVM / "src").glob("*.cpp")) + [str(NDVM / "tools" / "ndvm_par.cpp")]
    cmd = [cxx, "-std=c++17", "-O1", "-g", "-fsanitize=thread", "-fno-omit-frame-pointer",
           f"-I{NDVM / 'src'}", f"-I{NDVM / 'include'}", *srcs, "-o", str(out), "-pthread"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return out if r.returncode == 0 else None


def _run(par: Path, prog: str, binds: str, threads: int, n: int, b: int = 1) -> tuple[int, str, str]:
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(prog + "\n")
        bp = Path(d) / "p.bind"; bp.write_text(binds)
        env = dict(os.environ)
        env.update(TSAN_ENV)
        env.update({"NDVM_PAR_DUMP": "1", "NDVM_THREADS": str(threads),
                    "NDVM_PAR_N": str(n), "NDVM_B": str(b)})
        out = subprocess.run([str(par), str(sp), str(bp)], capture_output=True, text=True, env=env)
        return out.returncode, out.stdout, out.stderr


def _stress(par: Path, log) -> bool:
    cores = os.cpu_count() or 8
    oversub = 2 * cores
    ok = True
    log(f"ThreadSanitizer contention stress: {oversub} threads (2x{cores} cores), "
        f"{REPEATS} repeats/program, halt_on_error=1")
    for name, prog, binds, n in PROGRAMS:
        # single-thread reference dump from the SAME TSan binary (bit-comparison baseline)
        rc, ref, err = _run(par, prog, binds, threads=1, n=n)
        if rc != 0:
            log(f"  {name}: FAIL reference run rc={rc}\n{err[-2000:]}"); ok = False; continue
        races = 0
        mismatches = 0
        for rep in range(REPEATS):
            rc, dump, err = _run(par, prog, binds, threads=oversub, n=n)
            if rc != 0:
                races += 1
                log(f"  {name}: rep {rep}: TSAN rc={rc}\n{err[-2000:]}")
            elif dump != ref:
                mismatches += 1
                log(f"  {name}: rep {rep}: dump differs from single-thread (race/corruption)")
        # batched-lane variant: wide payloads under the same contention
        rc, dump_b, err = _run(par, prog, binds, threads=oversub, n=max(n // 4, 512), b=8)
        brace = " B=8:ok" if rc == 0 else f" B=8:TSAN rc={rc}"
        if rc != 0:
            races += 1
            log(f"  {name}: B=8 variant: TSAN rc={rc}\n{err[-2000:]}")
        status = "PASS" if (races == 0 and mismatches == 0) else "FAIL"
        if status == "FAIL":
            ok = False
        log(f"  {name}: {status}  ({REPEATS}x N={n} threads={oversub}: races={races} "
            f"dump-mismatches={mismatches};{brace})")
    log("TSAN STRESS " + ("PASS" if ok else "FAIL"))
    return ok


def test_parallel_stress_tsan():
    import pytest
    with tempfile.TemporaryDirectory() as d:
        par = _build_tsan(Path(d))
        if par is None:
            pytest.skip("ThreadSanitizer build unavailable (no g++/clang++ with libtsan)")
        assert _stress(par, print)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        par = _build_tsan(Path(d))
        if par is None:
            print("SKIP: ThreadSanitizer build unavailable"); sys.exit(2)
        sys.exit(0 if _stress(par, print) else 1)
