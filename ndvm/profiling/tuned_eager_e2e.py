#!/usr/bin/env python3
"""End-to-end tuned-eager interpreter: run a program through engine_pv (payload-only values) and validate
forward value + gradient against the tagged oracle, then time forward. The decisive thesis-gate confirmation
(resolves the boxing-share ambiguity by measuring forward wall time directly) and the source of the
NDVM-vs-tuned-eager residual the reframed paper needs. Run on an HPC compute node.

    python3 ndvm/profiling/tuned_eager_e2e.py [program_name]
"""
from __future__ import annotations
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import torch
from neural_compiler.dmci import compile_dmci
from neural_compiler.evaluator import engine as ENG_T          # tagged oracle
from neural_compiler.evaluator import engine_pv as ENG_P        # tuned-eager payload-only
from neural_compiler.runtime import tagged_value as TV
from neural_compiler.runtime import payload_value as PV
from profile_dmci_baseline import build_programs, EVAL_KW


def binds(prog, mod):
    b, leaves = {}, {}
    for k, v in prog["params"].items():
        leaf = torch.tensor(float(v), requires_grad=True)
        b[k] = mod.make_float(leaf); leaves[k] = leaf
    for k, v in prog["inputs"].items():
        b[k] = mod.make_float(torch.tensor(float(v)))
    return b, leaves


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scalar_mul_add"
    prog = build_programs(16, 80)[name]
    g = compile_dmci(prog["src"])
    pk = list(prog["params"])

    bt, lt = binds(prog, TV)
    yt = TV.unwrap_number(ENG_T.evaluate(g, bt, **EVAL_KW))
    print(f"tagged   forward: {float(yt.reshape(()).item()):.8f}")

    bp, lp = binds(prog, PV)
    yp = PV.unwrap_number(ENG_P.evaluate(g, bp, **EVAL_KW))
    yp_t = yp if isinstance(yp, torch.Tensor) else torch.tensor(float(yp))
    print(f"payload  forward: {float(yp_t.reshape(()).item()):.8f}")

    match = torch.allclose(yt.reshape(()), yp_t.reshape(()), atol=1e-6)
    print(f"forward match: {match}")
    gt = torch.autograd.grad(yt.sum(), [lt[k] for k in pk], retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(yp_t.sum(), [lp[k] for k in pk], allow_unused=True)
    gmatch = all((a is None and b is None) or torch.allclose(a, b, atol=1e-5) for a, b in zip(gt, gp))
    print(f"gradient match: {gmatch}  tagged={[None if x is None else round(float(x),5) for x in gt]} "
          f"payload={[None if x is None else round(float(x),5) for x in gp]}")

    def tfwd(mod, eng, n=20):
        bb, _ = binds(prog, mod)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); eng.evaluate(g, bb, **EVAL_KW); ts.append(time.perf_counter() - t0)
        return sorted(ts)[n // 2] * 1e3
    mt, mp = tfwd(TV, ENG_T), tfwd(PV, ENG_P)
    print(f"\n{name} forward ms: tagged {mt:.2f}  payload {mp:.2f}  tuned-eager speedup {mt/mp:.2f}x")


if __name__ == "__main__":
    main()
