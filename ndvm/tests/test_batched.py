"""Phase-3 batch-native validation: one evaluator walk over B lanes == B independent single-lane walks.

NDVM at B=1 is already forward- and gradient-equivalent to the PyTorch DMCI oracle (test_equivalence,
82/82 grads). This test proves the batched path by self-consistency: run a program at B=K with K
DISTINCT per-lane parameter sets in ONE batched walk, and check each lane's forward output AND per-lane
gradients equal the B=1 run with that lane's parameters. Batched-lane == scalar-lane, combined with
scalar == oracle, gives batched == oracle. It also demonstrates the structural walk is shared across
lanes (the co-search throughput multiplier). Pure NDVM (the native binary); no torch/HPC needed.

Stage 1 covers scalar + recursive programs (no matrix inputs); matrix batching is Stage 2.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("NDVM_RUN", str(HERE.parents[0] / "build" / "ndvm_run")))
REFS = HERE / "results" / "oracle_refs.json"
B = 8
ATOL, RTOL = 1e-5, 1e-4


def _run(src, scalars_per_lane, n, grad, matrix=None):
    """scalars_per_lane: {name: [v0..v(n-1)]}. matrix is bound shared across lanes. Returns (result[n], grads)."""
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(src + "\n")
        bp = Path(d) / "p.bind"
        lines = []
        for k, vals in scalars_per_lane.items():
            if n == 1:
                lines.append(f"scalar {k} {vals[0]!r}")
            else:
                lines.append("scalarb " + k + " " + " ".join(repr(v) for v in vals))
        if matrix:
            lines.append("matrix %s %d %d %s" % (matrix["name"], matrix["rows"], matrix["cols"],
                                                  " ".join(repr(x) for x in matrix["data"])))
        bp.write_text("\n".join(lines) + "\n")
        env = dict(os.environ)
        env["NDVM_B"] = str(n)
        if grad:
            env["NDVM_GRAD"] = "1"
        out = subprocess.run([str(RUN), str(sp), str(bp)], capture_output=True, text=True, env=env)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        result, grads = None, {}
        for line in out.stdout.splitlines():
            t = line.split()
            if t and t[0] == "result":
                result = [float(x) for x in t[1:]]
            elif t and t[0] == "grad":
                grads[t[1]] = [float(x) for x in t[2:]]
        return result, grads


def _close(a, b):
    # Batched lane b uses the identical per-lane ops as the scalar run, so even chaotic/divergent
    # values match exactly, including non-finite ones (e.g. a logistic-map lane with r>4 -> inf).
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= ATOL + RTOL * abs(b)


def _programs():
    if not REFS.exists() or not RUN.exists():
        return []
    progs = json.loads(REFS.read_text())["programs"]
    # Any program with differentiable scalar parameters (incl. vec/mat ops and matrix-input programs
    # like the Kalman rollout, whose shared matrix is broadcast across lanes).
    return [p for p in progs if p.get("scalars") and "oracle_result" in p]


_PROGS = _programs()


@pytest.mark.skipif(not _PROGS, reason="ndvm_run not built or oracle_refs.json not generated")
@pytest.mark.parametrize("p", _PROGS, ids=[p["name"] for p in _PROGS])
def test_batched_matches_per_lane(p):
    base = p["scalars"]
    matrix = p.get("matrix")
    # B distinct per-lane sweeps (lane 0 == base); multiplicative so control flow stays lane-uniform.
    sweep = {k: [base[k] * (1.0 + 0.07 * b) for b in range(B)] for k in base}
    bout, bgrad = _run(p["src"], sweep, B, grad=True, matrix=matrix)
    assert bout is not None and len(bout) == B, f"{p['name']}: batched result not [B]"
    for b in range(B):
        lane = {k: [sweep[k][b]] for k in base}
        sout, sgrad = _run(p["src"], lane, 1, grad=True, matrix=matrix)
        assert _close(bout[b], sout[0]), f"{p['name']} lane {b}: fwd batched={bout[b]} scalar={sout[0]}"
        for k in base:
            assert _close(bgrad[k][b], sgrad[k][0]), \
                f"{p['name']} lane {b} d/d{k}: batched={bgrad[k][b]} scalar={sgrad[k][0]}"
