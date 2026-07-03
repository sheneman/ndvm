"""Phase-5 torch-boundary population path: evaluate_population (parallel) == serial ndvm_forward.

evaluate_population packs a whole population of independent candidates into one GIL-released native call
that fans them across worker threads. Each candidate's (forward output, gradients) must be byte-identical
to evaluating it serially through ndvm_forward -- which the Phase-2/3 boundary tests already pin to the
PyTorch DMCI oracle. So parallel population == serial == oracle, per candidate. Needs the prebuilt
extension (HPC compute node).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

torch = pytest.importorskip("torch")
try:
    import ndvm_autograd
except Exception as e:  # noqa: BLE001
    pytest.skip(f"ndvm_autograd import failed: {e}", allow_module_level=True)


def _serial(src, params, matrices=None):
    leaves = {k: torch.tensor(float(v), requires_grad=True) for k, v in params.items()}
    out = ndvm_autograd.ndvm_forward(src, leaves, matrices or {})
    out.backward()
    return out.item(), {k: leaves[k].grad.item() for k in leaves}


def test_population_branch_matches_serial():
    # 60 candidates of a per-candidate divergent branch (different x -> different taken side).
    src = "(* x (if (> x 0) x 1.0))"
    cands = [(src, {"x": -3.0 + 0.1 * k}, {}) for k in range(60)]
    pop = ndvm_autograd.evaluate_population(cands, nthreads=8)
    assert len(pop) == len(cands)
    for (s, params, _), r in zip(cands, pop):
        assert r["ok"], r["err"]
        sout, sgrad = _serial(s, params)
        assert r["outs"][0] == sout, f"x={params['x']}: parallel {r['outs'][0]} != serial {sout}"
        assert r["grads"]["x"][0] == sgrad["x"], f"x={params['x']}: grad mismatch"


def test_population_kalman_matches_serial():
    # A small population of Kalman NLL fits with distinct (q, r); each has a shared obs matrix.
    import json
    refs = HERE / "results" / "oracle_refs.json"
    if not refs.exists():
        pytest.skip("oracle_refs.json missing")
    kal = next((p for p in json.loads(refs.read_text())["programs"] if p["name"].startswith("kalman")), None)
    if kal is None or not kal.get("matrix"):
        pytest.skip("no kalman program with matrix in refs")
    m = kal["matrix"]
    mats = {m["name"]: (m["rows"], m["cols"], m["data"])}
    base = kal["scalars"]
    cands = [(kal["src"], {k: base[k] * (1.0 + 0.03 * j) for k in base}, mats) for j in range(8)]
    pop = ndvm_autograd.evaluate_population(cands, nthreads=8)
    for (s, params, mm), r in zip(cands, pop):
        assert r["ok"], r["err"]
        sout, sgrad = _serial(s, params, mm)
        assert r["outs"][0] == sout, f"kalman fwd parallel {r['outs'][0]} != serial {sout}"
        for k in params:
            assert r["grads"][k][0] == sgrad[k], f"kalman d/d{k}: {r['grads'][k][0]} != {sgrad[k]}"
