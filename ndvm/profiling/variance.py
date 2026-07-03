#!/usr/bin/env python3
"""Variance / confidence intervals on the forward decomposition timings.

Reviewer #5 asked for variance, not bare medians, on the runtime measurements that back the paper's
decomposition table. This harness re-runs the same three forward timings used in residual_e2e.py --
tagged DMCI backend, tuned-eager (payload-only, engine_pv), native NDVM -- on the four scalar+recursive
baseline programs, N>=12 measured reps each (warmups discarded), and reports per program per backend:

    median (ms),  the 25-75 inter-quartile range (IQR),  and the coefficient of variation (std/mean, %).

The point is to show the decomposition RATIOS (tagged/eager, tagged/ndvm, eager/ndvm) are far larger than
the run-to-run variance, so the ordering in the paper table is unambiguous.

Run on an HPC compute node, pinned to one core of n128:
    NDVM_REPS controls measured reps (default 16); NDVM_WARMUP controls discarded warmups (default 3).
    The 80-step Kalman is skipped by default (set NDVM_KALMAN=1 to include it with few reps).
    srun --partition=sheneman --cpus-per-task=4 --mem=12G .venv/bin/python ndvm/profiling/variance.py
"""
from __future__ import annotations
import sys, os, time, statistics
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "python"))

import torch
from neural_compiler.dmci import compile_dmci
from neural_compiler.evaluator import engine as ENG_T, engine_pv as ENG_P
from neural_compiler.runtime import tagged_value as TV, payload_value as PV
from profile_dmci_baseline import build_programs, EVAL_KW

# Reuse the residual_e2e payload builders verbatim so we time exactly the same call paths.
from residual_e2e import make_matrices, binds, fwd_val, ndvm_val

REPS = int(os.environ.get("NDVM_REPS", 16))
WARMUP = int(os.environ.get("NDVM_WARMUP", 3))


def sample(fn, n, warmup):
    """Return a sorted list of measured wall-clock times (ms); first `warmup` reps discarded."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)


def stats(ts):
    n = len(ts)
    med = ts[n // 2]
    # linear-interpolated 25/75 quantiles for a stable IQR on small N
    qs = statistics.quantiles(ts, n=4, method="inclusive")
    q25, q75 = qs[0], qs[2]
    mean = statistics.fmean(ts)
    sd = statistics.stdev(ts) if n > 1 else 0.0
    cv = (sd / mean * 100.0) if mean > 0 else 0.0
    return dict(median=med, q25=q25, q75=q75, mean=mean, cv=cv)


def main():
    names = sys.argv[1:] or ["scalar_mul_add", "michaelis_menten", "damped_oscillator", "logistic_map_loop"]
    if os.environ.get("NDVM_KALMAN") == "1":
        names = names + ["kalman2d_T80"]
    progs = build_programs(16, 80)

    print(f"# variance.py  reps={REPS} warmup={WARMUP}  (single pinned core, n128)")
    print(f"{'program':20} {'backend':12} {'median_ms':>10} {'q25_ms':>9} {'q75_ms':>9} {'CV%':>7}")
    rows = {}
    for name in names:
        prog = progs[name]
        mats = make_matrices(prog)
        g = compile_dmci(prog["src"])
        # correctness gate (same tolerances as residual_e2e)
        vt = fwd_val(ENG_T, TV, prog, g, mats)
        vp = fwd_val(ENG_P, PV, prog, g, mats)
        vn = ndvm_val(prog, mats)
        assert abs(vt - vp) < 1e-4 and abs(vt - vn) < 1e-3, f"{name} backends disagree {vt}/{vp}/{vn}"

        reps = REPS if name != "kalman2d_T80" else max(6, REPS // 3)
        backs = [
            ("tagged",      lambda: ENG_T.evaluate(g, binds(prog, TV, mats), **EVAL_KW)),
            ("tuned-eager", lambda: ENG_P.evaluate(g, binds(prog, PV, mats), **EVAL_KW)),
            ("ndvm",        lambda: ndvm_val(prog, mats)),
        ]
        rows[name] = {}
        for tag, fn in backs:
            s = stats(sample(fn, reps, WARMUP))
            rows[name][tag] = s
            print(f"{name:20} {tag:12} {s['median']:10.4f} {s['q25']:9.4f} {s['q75']:9.4f} {s['cv']:7.2f}")

    # ratio stability summary: medians and the max CV across all backends
    print("\n# ratio stability (medians) and worst-case CV per program")
    print(f"{'program':20} {'eager/tag':>10} {'ndvm/tag':>10} {'eager/ndvm':>11} {'maxCV%':>8}")
    all_cv = []
    for name in names:
        r = rows[name]
        mt, mp, mn = r["tagged"]["median"], r["tuned-eager"]["median"], r["ndvm"]["median"]
        maxcv = max(r[t]["cv"] for t in ("tagged", "tuned-eager", "ndvm"))
        all_cv.extend(r[t]["cv"] for t in ("tagged", "tuned-eager", "ndvm"))
        print(f"{name:20} {mt/mp:9.2f}x {mt/mn:9.1f}x {mp/mn:10.1f}x {maxcv:8.2f}")
    print(f"\n# overall CV range across all backends/programs: "
          f"{min(all_cv):.2f}% .. {max(all_cv):.2f}%")


if __name__ == "__main__":
    main()
