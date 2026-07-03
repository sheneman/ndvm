"""Phase-2 PyTorch autograd-boundary test: NDVMFunction wires native gradients into torch autograd.

For each program with reference gradients (results/oracle_refs.json), build the parameters as
requires_grad leaf tensors, evaluate through `ndvm_forward`, call `.backward()`, and check that each
leaf's `.grad` equals the oracle's autograd gradient within tolerance. Also runs a short Adam descent
on the Kalman NLL to confirm an external optimizer can drive NDVM end to end.

Requires torch + a C++17 compiler (the extension JIT-compiles on first use), so this runs on an HPC
compute node, not the login node / Mac. Skips cleanly if torch or the references are unavailable.

    PYTHONPATH=<repo> python3 -m pytest ndvm/tests/test_autograd_boundary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))
REFS = HERE / "results" / "oracle_refs.json"

torch = pytest.importorskip("torch")
try:
    import ndvm_autograd  # noqa: E402  (from ndvm/python, added to sys.path above)
except Exception as e:  # noqa: BLE001
    pytest.skip(f"ndvm_autograd import failed: {e}", allow_module_level=True)

if not REFS.exists():
    pytest.skip("oracle_refs.json not generated", allow_module_level=True)

# Reuse the gradient tolerances from the equivalence comparator.
sys.path.insert(0, str(HERE))
from compare_equivalence import GTOL, GDEFAULT  # type: ignore  # noqa: E402

_PROGRAMS = [p for p in json.loads(REFS.read_text())["programs"] if p.get("grads")]


def _matrices(p):
    m = p.get("matrix")
    return {m["name"]: (m["rows"], m["cols"], m["data"])} if m else {}


@pytest.mark.parametrize("p", _PROGRAMS, ids=[p["name"] for p in _PROGRAMS])
def test_autograd_boundary_gradients(p):
    params = {k: torch.tensor(float(v), requires_grad=True) for k, v in p["scalars"].items()}
    out = ndvm_autograd.ndvm_forward(p["src"], params, _matrices(p))
    out.backward()
    gatol, grtol = GTOL.get(p["name"], GDEFAULT)
    for k, gw in p["grads"].items():
        gn = params[k].grad.item()
        assert abs(gn - gw) <= gatol + grtol * abs(gw), f"{p['name']} d/d{k}: ndvm={gn!r} oracle={gw!r}"


def test_optimizer_step_reduces_kalman_nll():
    kal = next((p for p in _PROGRAMS if p["name"].startswith("kalman")), None)
    if kal is None:
        pytest.skip("no kalman program in refs")
    q = torch.tensor(0.30, requires_grad=True)
    r = torch.tensor(0.30, requires_grad=True)
    opt = torch.optim.Adam([q, r], lr=0.05)
    mats = _matrices(kal)
    L0 = None
    for _ in range(15):
        opt.zero_grad()
        nll = ndvm_autograd.ndvm_forward(kal["src"], {"q": q, "r": r}, mats)
        if L0 is None:
            L0 = nll.item()
        nll.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(min=1e-4); r.clamp_(min=1e-4)
    assert nll.item() < L0 - 1.0, f"NLL did not decrease: {L0} -> {nll.item()}"
