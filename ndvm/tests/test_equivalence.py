"""NDVM forward-equivalence vs the PyTorch DMCI oracle (design section 15).

Phase 1 (forward, no AD yet): for every program in results/oracle_refs.json (generated on HPC by
oracle_refs.py from the frozen DMCI backend), run the native NDVM runtime (ndvm_run, built via CMake)
on the IDENTICAL source + bindings and assert the forward output matches within tolerance. Skips
cleanly when the native binary or the reference artifact is absent (e.g. on a machine where neither
has been built/generated). Gradient equivalence is Phase 2.

    cmake -S ndvm -B ndvm/build && cmake --build ndvm/build
    python3 ndvm/tests/oracle_refs.py        # on an HPC compute node (needs torch)
    pytest ndvm/tests/test_equivalence.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

try:
    import pytest
except Exception:  # noqa: BLE001
    pytest = None

from compare_equivalence import run_ndvm, TOL, DEFAULT_TOL, GTOL, GDEFAULT, DEFAULT_RUN  # type: ignore

HERE = Path(__file__).resolve().parent
REFS = HERE / "results" / "oracle_refs.json"


def _cases():
    if not REFS.exists() or not Path(DEFAULT_RUN).exists():
        return []
    data = json.loads(REFS.read_text())
    return [p for p in data["programs"] if "oracle_result" in p]


_CASES = _cases()

if pytest is not None:
    @pytest.mark.skipif(not _CASES, reason="ndvm_run not built or oracle_refs.json not generated")
    @pytest.mark.parametrize("prog", _CASES, ids=[p["name"] for p in _CASES])
    def test_forward_equivalence(prog):
        oracle_grads = prog.get("grads") or {}
        got, _ = run_ndvm(DEFAULT_RUN, prog["src"], prog["scalars"], prog.get("matrix"), grad=False)
        want = prog["oracle_result"]
        atol, rtol = TOL.get(prog["name"], DEFAULT_TOL)
        if not math.isfinite(want) or not math.isfinite(got):
            assert (math.isnan(want) and math.isnan(got)) or got == want, f"{prog['name']}: {got} vs {want}"
        else:
            assert abs(got - want) <= atol + rtol * abs(want), \
                f"{prog['name']}: ndvm={got!r} oracle={want!r}"

    @pytest.mark.skipif(not _CASES, reason="ndvm_run not built or oracle_refs.json not generated")
    @pytest.mark.parametrize("prog", [p for p in _CASES if p.get("grads")],
                             ids=[p["name"] for p in _CASES if p.get("grads")])
    def test_gradient_equivalence(prog):
        _, ggrads = run_ndvm(DEFAULT_RUN, prog["src"], prog["scalars"], prog.get("matrix"), grad=True)
        gatol, grtol = GTOL.get(prog["name"], GDEFAULT)
        for k, gw in prog["grads"].items():
            gn = ggrads.get(k)
            assert gn is not None, f"{prog['name']}: ndvm missing grad d/d{k}"
            assert abs(gn - gw) <= gatol + grtol * abs(gw), \
                f"{prog['name']} d/d{k}: ndvm={gn!r} oracle={gw!r}"
