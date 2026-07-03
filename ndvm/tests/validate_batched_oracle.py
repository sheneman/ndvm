#!/usr/bin/env python3
"""Batched NDVM vs the PyTorch DMCI oracle, per lane (Phase-3 cross-check, run on an HPC compute node).

For a few programs, sweep each parameter across B lanes, compute the oracle's per-lane forward output
[B] and per-lane gradient [B] (bind params as [B] leaf tensors, evaluate once, loss=output.sum();
loss.backward() -> param.grad is the per-lane gradient since lanes are independent), then run the native
ndvm_run in batched mode (NDVM_B=B, scalarb bindings, shared matrix broadcast) and compare per lane.
This confirms batched NDVM == oracle directly (complementing the local self-consistency test), under
whatever compiler built ndvm_run (point NDVM_RUN at the g++ build to repeat the Phase-2 cross-compiler
check). Skips a program if the oracle cannot compile it.

    NDVM_RUN=/path/to/ndvm_run python3 ndvm/tests/validate_batched_oracle.py
"""
from __future__ import annotations
import json, math, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
RUN = os.environ.get("NDVM_RUN", str(HERE.parents[0] / "build" / "ndvm_run"))
REFS = json.loads((HERE / "results" / "oracle_refs.json").read_text())["programs"]

B = 4
FACTOR = 0.02   # gentle per-lane spread (keeps loop/branch control uniform; logistic r stays < 4)
NAMES = ["scalar_mul_add", "michaelis_menten", "damped_oscillator", "power_law",
         "logistic_map_loop", "logdet_grad", "kalman2d_T80"]
# per-program (abs, rel) gradient tolerance (LU linalg looser, matching the de-risk pilot)
GTOL = {"kalman2d_T80": (5.0, 2e-2), "logdet_grad": (1e-3, 1e-3), "logistic_map_loop": (1e-3, 1e-3)}


def sweep(base):
    return {k: [base[k] * (1.0 + FACTOR * b) for b in range(B)] for k in base}


def oracle(p, sw):
    import torch
    from neural_compiler.dmci import compile_dmci, as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    g = compile_dmci(p["src"])
    leaves = {k: torch.tensor(sw[k], requires_grad=True) for k in sw}   # each [B]
    binds = {k: make_float(leaves[k]) for k in sw}
    if p["matrix"]:
        m = p["matrix"]; binds[m["name"]] = as_matrix(torch.tensor(m["data"]).reshape(m["rows"], m["cols"]))
    out = unwrap_number(evaluate(g, binds, max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)).reshape(-1)
    fwd = out.detach().tolist()
    out.sum().backward()
    grads = {k: leaves[k].grad.reshape(-1).tolist() for k in sw}
    return fwd, grads


def ndvm(p, sw):
    d = tempfile.mkdtemp(); sp = Path(d) / "p.scm"; sp.write_text(p["src"] + "\n"); bp = Path(d) / "p.bind"
    lines = ["scalarb " + k + " " + " ".join(repr(v) for v in sw[k]) for k in sw]
    if p["matrix"]:
        m = p["matrix"]; lines.append("matrix %s %d %d %s" % (m["name"], m["rows"], m["cols"], " ".join(repr(x) for x in m["data"])))
    bp.write_text("\n".join(lines) + "\n")
    env = dict(os.environ); env["NDVM_B"] = str(B); env["NDVM_GRAD"] = "1"
    o = subprocess.run([RUN, str(sp), str(bp)], capture_output=True, text=True, env=env)
    if o.returncode != 0:
        raise RuntimeError(o.stderr.strip())
    fwd, grads = None, {}
    for ln in o.stdout.splitlines():
        t = ln.split()
        if t and t[0] == "result": fwd = [float(x) for x in t[1:]]
        elif t and t[0] == "grad": grads[t[1]] = [float(x) for x in t[2:]]
    return fwd, grads


def close(a, b, atol, rtol):
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= atol + rtol * abs(b)


def main():
    npass = nfail = 0
    for name in NAMES:
        p = next((x for x in REFS if x["name"] == name), None)
        if p is None or "oracle_result" not in p or not p.get("scalars"):
            print(f"  [SKIP] {name}"); continue
        sw = sweep(p["scalars"])
        try:
            ofwd, ograd = oracle(p, sw)
            nfwd, ngrad = ndvm(p, sw)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}: {e}"); nfail += 1; continue
        gat, grt = GTOL.get(name, (1e-4, 1e-4))
        ok = all(close(nfwd[b], ofwd[b], 1e-3, 1e-3) for b in range(B))
        for k in sw:
            ok = ok and all(close(ngrad[k][b], ograd[k][b], gat, grt) for b in range(B))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} fwd lane0 ndvm={nfwd[0]:.5g} oracle={ofwd[0]:.5g} "
              f"| grads {'all-match' if ok else 'MISMATCH'} over B={B} lanes")
        npass += ok; nfail += (not ok)
    print(f"\n=== batched vs oracle: {npass}/{npass + nfail} programs match per-lane ===")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
