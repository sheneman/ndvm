#!/usr/bin/env python3
"""Three-way forward comparison and the NDVM-vs-tuned-eager residual.

The thesis gate showed a tuned-eager payload-only interpreter removes ~5-7x of the current backend's cost in
eager Python. The open question the reframed paper turns on: how much does the NATIVE runtime still earn OVER
a competent eager encoding? NDVM is native C++; tuned-eager is still a Python interpreter, so it keeps the
Python eval-loop overhead NDVM removes. This harness times forward three ways on each program -- tagged
backend, tuned-eager (payload-only, engine_pv), native NDVM -- validates all three agree, and reports:
  - tuned-eager speedup  = tagged / payload          (what the eager encoding alone buys)
  - NDVM speedup         = tagged / ndvm             (what the native runtime buys vs the current backend)
  - RESIDUAL             = payload / ndvm            (what the native runtime earns OVER tuned-eager)

Run on an HPC compute node (torch + DMCI + the NDVM ext). The first NDVM call may JIT-compile the extension.
    python3 ndvm/profiling/residual_e2e.py [program ...]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "python"))   # ndvm/python -> ndvm_autograd, ndvm_native

import torch
from neural_compiler.dmci import compile_dmci, as_matrix
from neural_compiler.evaluator import engine as ENG_T, engine_pv as ENG_P
from neural_compiler.runtime import tagged_value as TV, payload_value as PV
from profile_dmci_baseline import build_programs, EVAL_KW
from ndvm_autograd import ndvm_forward


def make_matrices(prog):
    mats = {}
    for name, (kind, shape) in prog.get("matrix", {}).items():
        g = torch.Generator().manual_seed(0)
        mats[name] = torch.randn(*shape, generator=g) if kind == "randn" else torch.zeros(*shape)
    return mats


def binds(prog, mod, mats):
    b = {}
    for k, v in prog["params"].items():
        b[k] = mod.make_float(torch.tensor(float(v), requires_grad=True))
    for k, v in prog["inputs"].items():
        b[k] = mod.make_float(torch.tensor(float(v)))
    for name, t in mats.items():
        b[name] = mod.as_matrix(t)
    return b


def fwd_val(eng, mod, prog, g, mats):
    out = eng.evaluate(g, binds(prog, mod, mats), **EVAL_KW)
    y = mod.unwrap_number(out)
    return float((y if isinstance(y, torch.Tensor) else torch.tensor(float(y))).reshape(()).item())


def ndvm_val(prog, mats):
    params = {k: torch.tensor(float(v), requires_grad=True) for k, v in prog["params"].items()}
    params.update({k: torch.tensor(float(v)) for k, v in prog["inputs"].items()})
    mdict = {n: (t.shape[0], t.shape[1] if t.dim() > 1 else 1, t.reshape(-1)) for n, t in mats.items()}
    return float(ndvm_forward(prog["src"], params, mdict or None).reshape(()).item())


import os
def timed(fn, n=None):
    n = n or int(os.environ.get("NDVM_REPS", 15))
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1e3


def main():
    names = sys.argv[1:] or ["scalar_mul_add", "michaelis_menten", "damped_oscillator", "logistic_map_loop"]
    progs = build_programs(16, 80)
    print(f"{'program':20} {'tagged_ms':>10} {'eager_ms':>9} {'ndvm_ms':>9} "
          f"{'eager_vs_tag':>12} {'ndvm_vs_tag':>11} {'RESIDUAL':>9}  match")
    for name in names:
        prog = progs[name]
        mats = make_matrices(prog)
        g = compile_dmci(prog["src"])
        try:
            vt = fwd_val(ENG_T, TV, prog, g, mats)
            vp = fwd_val(ENG_P, PV, prog, g, mats)
            vn = ndvm_val(prog, mats)
            match = abs(vt - vp) < 1e-4 and abs(vt - vn) < 1e-3
            mt = timed(lambda: ENG_T.evaluate(g, binds(prog, TV, mats), **EVAL_KW))
            mp = timed(lambda: ENG_P.evaluate(g, binds(prog, PV, mats), **EVAL_KW))
            mn = timed(lambda: ndvm_val(prog, mats))
            print(f"{name:20} {mt:10.2f} {mp:9.2f} {mn:9.3f} {mt/mp:11.2f}x {mt/mn:10.1f}x "
                  f"{mp/mn:8.1f}x  {match} ({vt:.4f}/{vp:.4f}/{vn:.4f})")
        except Exception as e:
            print(f"{name:20} ERROR: {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    main()
