"""Phase-3b PyTorch boundary test: the BATCHED autograd boundary fits B lanes in one walk.

`ndvm_forward` now accepts per-lane [B] parameter tensors and returns a [B] output whose per-lane
gradients flow back through one native structural walk. This is validated by self-consistency: a batched
[B] call (forward AND per-lane .grad) must equal B independent B=1 `ndvm_forward` calls -- which the
Phase-2 boundary test already checks against the oracle. Plus a B-lane Adam descent driving B independent
Kalman fits through a single batched op (the co-search throughput payoff, now reachable from torch).

Requires torch + the prebuilt extension (setup.py build_ext --inplace) on an HPC compute node.

    PYTHONPATH=<repo> python3 -m pytest ndvm/tests/test_autograd_batched.py
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
    import ndvm_autograd  # noqa: E402
except Exception as e:  # noqa: BLE001
    pytest.skip(f"ndvm_autograd import failed: {e}", allow_module_level=True)

if not REFS.exists():
    pytest.skip("oracle_refs.json not generated", allow_module_level=True)

ATOL, RTOL = 1e-4, 2e-4
_PROGRAMS = [p for p in json.loads(REFS.read_text())["programs"] if p.get("grads") and p.get("scalars")]


def _matrices(p):
    m = p.get("matrix")
    return {m["name"]: (m["rows"], m["cols"], m["data"])} if m else {}


def _close(a, b):
    import math
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= ATOL + RTOL * abs(b)


@pytest.mark.parametrize("p", _PROGRAMS, ids=[p["name"] for p in _PROGRAMS])
def test_batched_boundary_self_consistency(p):
    base = p["scalars"]
    names = list(base)
    mats = _matrices(p)
    B = 4
    # Multiplicative per-lane sweep keeps control flow lane-uniform (these are the Phase-2 programs).
    sweep = {n: [base[n] * (1.0 + 0.05 * b) for b in range(B)] for n in names}

    bparams = {n: torch.tensor(sweep[n], requires_grad=True) for n in names}  # [B] leaves
    bout = ndvm_autograd.ndvm_forward(p["src"], bparams, mats)                # [B]
    assert bout.shape == (B,), f"{p['name']}: batched output shape {tuple(bout.shape)} != ({B},)"
    bout.sum().backward()  # out[b] depends only on lane b -> .grad[b] is the per-lane gradient
    bgrad = {n: bparams[n].grad.clone() for n in names}

    for b in range(B):
        lp = {n: torch.tensor(sweep[n][b], requires_grad=True) for n in names}  # scalar leaves
        lout = ndvm_autograd.ndvm_forward(p["src"], lp, mats)                   # scalar
        lout.backward()
        assert _close(bout[b].item(), lout.item()), \
            f"{p['name']} lane {b}: fwd batched={bout[b].item()} scalar={lout.item()}"
        for n in names:
            assert _close(bgrad[n][b].item(), lp[n].grad.item()), \
                f"{p['name']} lane {b} d/d{n}: batched={bgrad[n][b].item()} scalar={lp[n].grad.item()}"


def test_batched_optimizer_drives_all_lanes():
    kal = next((p for p in _PROGRAMS if p["name"].startswith("kalman")), None)
    if kal is None:
        pytest.skip("no kalman program in refs")
    mats = _matrices(kal)
    B = 3
    q = torch.tensor([0.30, 0.22, 0.38], requires_grad=True)
    r = torch.tensor([0.30, 0.38, 0.22], requires_grad=True)
    opt = torch.optim.Adam([q, r], lr=0.05)
    L0 = None
    for _ in range(15):
        opt.zero_grad()
        nll = ndvm_autograd.ndvm_forward(kal["src"], {"q": q, "r": r}, mats)  # [B]
        if L0 is None:
            L0 = nll.detach().clone()
        nll.sum().backward()   # independent lanes -> each lane's grad is its own dNLL
        opt.step()
        with torch.no_grad():
            q.clamp_(min=1e-4); r.clamp_(min=1e-4)
    final = nll.detach()
    assert torch.all(final < L0 - 1.0), f"some lane's NLL did not decrease: {L0.tolist()} -> {final.tolist()}"
