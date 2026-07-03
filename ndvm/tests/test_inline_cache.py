"""Phase-4 inline variable-lookup cache: cache-ON must equal cache-OFF (the pure scan), byte for byte.

Each DK_VAR node caches its lexical address (parent-hops, slot); a hit jumps straight to the binding,
validated by binds[slot]==symbol. NDVM_NO_INLINE forces the scanning lookup (the original, oracle-
validated path). This test drives adversarial SHADOWING programs -- where a naive lexical-address cache
could pick the wrong binding -- and asserts the cached result is byte-identical to the scan, forward AND
gradient. (The oracle-equivalence suite already pins cache-ON to the oracle on 33 programs; this pins the
trickier shadowing cases to the scan directly.)
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("NDVM_RUN", str(HERE.parents[0] / "build" / "ndvm_run")))


def _stdout(src, binds, env):
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(src + "\n")
        bp = Path(d) / "p.bind"; bp.write_text(binds)
        e = dict(os.environ); e.update(env)
        out = subprocess.run([str(RUN), str(sp), str(bp)], capture_output=True, text=True, env=e)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        return out.stdout


# Adversarial shadowing / scoping programs. Each binds a,b and is run with gradients.
SHADOW = [
    # inner let shadows an outer binding of the same name
    {"name": "nested_let_shadow", "src": "(let ((x a)) (let ((x b)) (+ x x)))"},
    # the inner binding's RHS references the OUTER same-named variable (frame exists but x not yet bound)
    {"name": "rhs_refs_outer_same_name", "src": "(let ((x a)) (let ((x (* x b))) x))"},
    # outer binding is used AFTER an inner shadow goes out of scope
    {"name": "use_outer_after_inner", "src": "(let ((x a)) (+ (let ((x b)) x) x))"},
    # sibling inner scopes reuse the same name at the same lexical depth (distinct nodes)
    {"name": "sibling_scopes", "src": "(+ (let ((x a)) x) (let ((x b)) (* x b)))"},
    # a free variable resolved at constant depth across recursion levels, accumulated in a loop
    {"name": "recursion_free_var",
     "src": "(define (f n acc) (if (= n 0) acc (f (- n 1) (+ acc a))))\n(f 7.0 b)"},
    # deep let nesting: the innermost x must resolve to the innermost binding, the body a to the global
    {"name": "deep_nesting",
     "src": "(let ((x a)) (let ((y (+ x 1.0))) (let ((z (* y b))) (let ((x z)) (+ x (* a y))))))"},
]


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
@pytest.mark.parametrize("p", SHADOW, ids=[p["name"] for p in SHADOW])
def test_inline_cache_matches_scan(p):
    binds = "scalar a 1.5\nscalar b 2.5\n"
    on = _stdout(p["src"], binds, {"NDVM_GRAD": "1"})                       # inline cache (default)
    off = _stdout(p["src"], binds, {"NDVM_GRAD": "1", "NDVM_NO_INLINE": "1"})  # pure scan
    assert on == off, f"{p['name']}: inline-cache output differs from scan\non ={on!r}\noff={off!r}"


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
@pytest.mark.parametrize("p", SHADOW, ids=[p["name"] for p in SHADOW])
def test_inline_cache_stable_under_reuse(p):
    # The address cache lives on the program AST and persists across forwards on a reused Interp; running
    # several reuse iterations must still match the pure scan (the cache must not drift between forwards).
    binds = "scalar a 1.5\nscalar b 2.5\n"
    reuse_on = _stdout(p["src"], binds, {"NDVM_GRAD": "1", "NDVM_REUSE": "5"})
    off = _stdout(p["src"], binds, {"NDVM_GRAD": "1", "NDVM_NO_INLINE": "1"})
    assert reuse_on == off, f"{p['name']}: reuse+inline differs from scan\non ={reuse_on!r}\noff={off!r}"
