"""Phase-4 cross-call reuse validation: a REUSED Interp == a FRESH Interp, byte for byte.

run() caches the parsed+macro-expanded+decoded program by source, and begin_forward()/reset_state() clear
the per-forward state (arena, env, heap, tape, active set) while keeping that program + its warm decode
cache + the symbol table. This is the co-search win (parse/expand/decode paid once, not per eval), so it
MUST be exactly transparent: re-running the same Interp N times must reproduce, bit for bit, a fresh
single run -- forward outputs AND gradients, for scalar / recursive / matrix / batched / divergent
programs. ndvm_run does the reuse loop under NDVM_REUSE=<N> and prints the last forward; this compares its
stdout to a fresh run's. Pure NDVM; no torch/HPC needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("NDVM_RUN", str(HERE.parents[0] / "build" / "ndvm_run")))
REFS = HERE / "results" / "oracle_refs.json"


def _stdout(src, binds, env):
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(src + "\n")
        bp = Path(d) / "p.bind"; bp.write_text(binds)
        e = dict(os.environ); e.update(env)
        out = subprocess.run([str(RUN), str(sp), str(bp)], capture_output=True, text=True, env=e)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        return out.stdout


def _refs_programs():
    if not REFS.exists() or not RUN.exists():
        return []
    return [p for p in json.loads(REFS.read_text())["programs"] if p.get("scalars")]


_PROGS = _refs_programs()


@pytest.mark.skipif(not _PROGS, reason="ndvm_run not built or oracle_refs.json missing")
@pytest.mark.parametrize("p", _PROGS, ids=[p["name"] for p in _PROGS])
def test_reuse_byte_identical_to_fresh(p):
    lines = [f"scalar {k} {v!r}" for k, v in p["scalars"].items()]
    m = p.get("matrix")
    if m:
        lines.append("matrix %s %d %d %s" % (m["name"], m["rows"], m["cols"], " ".join(repr(x) for x in m["data"])))
    binds = "\n".join(lines) + "\n"
    fresh = _stdout(p["src"], binds, {"NDVM_GRAD": "1"})
    reuse = _stdout(p["src"], binds, {"NDVM_GRAD": "1", "NDVM_REUSE": "3"})
    assert fresh == reuse, f"{p['name']}: reuse output differs from fresh\nfresh={fresh!r}\nreuse={reuse!r}"


# Batched + divergent (per-lane control flow) through the reuse path: the active set, actset pool, and
# select-merge tape must all reset cleanly between forwards.
_BATCHED = [
    {"name": "div_scalar_branch", "src": "(* x (if (> x 0) x 1.0))", "binds": "scalarb x 2 -3 5 -0.5\n", "B": 4},
    {"name": "div_newton_loop",
     "src": "(define (go x) (if (< (abs (- (* x x) a)) 0.0001) x (go (* 0.5 (+ x (/ a x))))))\n(go 1.0)",
     "binds": "scalarb a 2 4 9 16\n", "B": 4},
]


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
@pytest.mark.parametrize("p", _BATCHED, ids=[p["name"] for p in _BATCHED])
def test_reuse_batched_divergent(p):
    base = {"NDVM_GRAD": "1", "NDVM_B": str(p["B"])}
    fresh = _stdout(p["src"], p["binds"], base)
    reuse = _stdout(p["src"], p["binds"], {**base, "NDVM_REUSE": "4"})
    assert fresh == reuse, f"{p['name']}: reuse differs\nfresh={fresh!r}\nreuse={reuse!r}"
