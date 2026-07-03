#!/usr/bin/env python3
"""De-risk: does NDVM compute correct gradients when the BATCH axis is the DATA grid and the fitted
parameters are SHARED across all data points? (This is the opposite of NDVM's population batching, where
each lane has its own params.) We pass x as a [B] per-lane input and each scalar parameter as
param * ones(B) (a broadcast view), so torch sums the per-lane grads back into the scalar leaf. We then
check NDVM's parameter gradients of an MSE regression loss against the PyTorch DMCI oracle to float32.
Run on a compute node: srun ... .venv/bin/python ndvm/profiling/cosearch_gradcheck.py
"""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[0] / "python"))
import torch
from neural_compiler.dmci import compile_dmci
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number
from ndvm_autograd import ndvm_forward

EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)

def target(x):  # damped oscillator, inside the candidate space
    return 1.0 * torch.exp(-0.3 * x) * torch.cos(2.0 * x)

CANDS = [
    "(+ a (* b x))",
    "(+ (* a x) (* b (sin (* c x))))",
    "(* a (* (exp (- 0 (* b x))) (cos (* c x))))",
    "(/ (* a x) (+ b x))",
]

def dmci_loss_grad(src, params, x, y):
    B = x.shape[0]
    g = compile_dmci(src)
    leaves = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    binds = {"x": make_float(x)}
    for k, t in leaves.items():
        binds[k] = make_float(t * torch.ones(B))
    out = unwrap_number(evaluate(g, binds, **EVAL_KW)).reshape(-1)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    return float(loss), {k: float(t.grad) for k, t in leaves.items()}

def ndvm_loss_grad(src, params, x, y):
    B = x.shape[0]
    leaves = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    pd = {"x": x}
    for k, t in leaves.items():
        pd[k] = t * torch.ones(B)
    out = ndvm_forward(src, pd, None).reshape(-1)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    return float(loss), {k: float(t.grad) for k, t in leaves.items()}

def main():
    x = torch.linspace(0.1, 6.0, 48)
    y = target(x)
    print(f"{'candidate':46} fwd_match  grad_match  (max|dgrad|)")
    allok = True
    for src in CANDS:
        used = [p for p in "abcd" if p in src.split()] or [p for p in "abcd" if p in src]
        params = {p: 0.5 for p in used}
        try:
            ld, gd = dmci_loss_grad(src, params, x, y)
            ln, gn = ndvm_loss_grad(src, params, x, y)
            fwd_ok = abs(ld - ln) < 1e-4 * (1 + abs(ld))
            dg = max(abs(gd[k] - gn[k]) for k in params)
            gscale = max(1.0, max(abs(gd[k]) for k in params))
            grad_ok = dg < 2e-3 * gscale
            allok = allok and fwd_ok and grad_ok
            print(f"{src[:46]:46} {str(fwd_ok):9} {str(grad_ok):10} ({dg:.2e}) loss {ld:.4f}/{ln:.4f}")
        except Exception as e:
            allok = False
            print(f"{src[:46]:46} ERROR: {str(e)[:80]}")
    print("ALL OK" if allok else "FAILURES PRESENT")

if __name__ == "__main__":
    main()
