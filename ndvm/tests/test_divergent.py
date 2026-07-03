"""Phase-3b validation: per-lane divergent control flow == B independent single-lane runs.

The DMCI/heap-backed oracle RAISES on per-lane-divergent control flow (engine.py _batch_branch_decision),
so there is no batched-oracle reference. Phase 3b is validated by LANE DECOMPOSITION: a batched run over
B per-lane parameter sets must reproduce, lane by lane (forward AND per-lane gradient), B independent B=1
runs with those parameters. Each B=1 run already matches the PyTorch oracle (test_equivalence, 82/82), so
batched-divergent == oracle per lane by transitivity. Pure NDVM (the native binary); no torch/HPC needed.

Covers: D1 non-recursive scalar branch; D2 per-lane convergence loop (Newton sqrt); D3 NaN/Inf-in-dead-lane
stressor (a terminated lane's continued recur overflows -> must not poison its gradient); D4 nested >2
termination levels (nested SELECTs); D6 VecCell-output divergence. Raise cases: D5 per-lane structural
divergence (different heap cells), D6' vector shape mismatch -- both MUST raise, not silently mis-merge.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUN = Path(os.environ.get("NDVM_RUN", str(HERE.parents[0] / "build" / "ndvm_run")))
ATOL, RTOL = 1e-4, 2e-4


def _run(src, scalars_per_lane, n, grad):
    """scalars_per_lane: {name: [v0..v(n-1)]}. Returns (result[n], {name: grads[n]}) or raises."""
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(src + "\n")
        bp = Path(d) / "p.bind"
        lines = []
        for k, vals in scalars_per_lane.items():
            if n == 1:
                lines.append(f"scalar {k} {vals[0]!r}")
            else:
                lines.append("scalarb " + k + " " + " ".join(repr(v) for v in vals))
        bp.write_text("\n".join(lines) + "\n")
        env = dict(os.environ); env["NDVM_B"] = str(n)
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
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= ATOL + RTOL * abs(b)


# Divergent programs: control flow differs per lane via the swept parameters.
DIVERGENT = [
    # D1: non-recursive scalar branch.
    {"name": "D1_scalar_branch",
     "src": "(* x (if (> x 0) x 1.0))",
     "lanes": [{"x": 2.0}, {"x": -3.0}, {"x": 5.0}, {"x": -0.5}]},
    # D2: per-lane convergence loop (Newton sqrt; lanes converge at different iteration counts).
    {"name": "D2_newton_sqrt",
     "src": "(define (go x) (if (< (abs (- (* x x) a)) 0.0001) x (go (* 0.5 (+ x (/ a x))))))\n(go 1.0)",
     "lanes": [{"a": 2.0}, {"a": 4.0}, {"a": 9.0}, {"a": 16.0}, {"a": 30.0}]},
    # D3: NaN/Inf-in-dead-lane stressor. n is a per-lane iteration count; a terminated lane keeps being
    # squared in the (still-active) recur branch and overflows to inf -- its gradient must stay clean.
    {"name": "D3_dead_lane_inf",
     "src": "(define (go i x) (if (< i n) (go (+ i 1) (* x x)) x))\n(go 0 x)",
     "lanes": [{"x": 1.05, "n": 3.0}, {"x": 1.03, "n": 10.0}]},
    # D4: nested >2 termination levels -> nested SELECTs (lanes terminate at iters 1,2,3).
    {"name": "D4_nested_levels",
     "src": "(define (go i x) (if (< i n) (go (+ i 1) (* x x)) x))\n(go 0 x)",
     "lanes": [{"x": 1.1, "n": 1.0}, {"x": 1.2, "n": 2.0}, {"x": 1.3, "n": 3.0}]},
    # D6: VecCell-output divergence (both branches return a length-2 vector -> SELECT over slabs).
    {"name": "D6_vec_branch",
     "src": "(vsum (if (> s 0) (scale a (vec a b)) (vec b a)))",
     "lanes": [{"a": 1.5, "b": 2.0, "s": 1.0}, {"a": 0.7, "b": 3.0, "s": -1.0},
               {"a": 2.0, "b": 1.0, "s": 1.0}]},
    # D7: cond with >2 clauses diverging across lanes (lowered to nested-if -> nested SELECTs). Lanes
    # take clause 1 (>2), clause 2 (>0), or else; per-lane test laziness keeps each clause's test scoped.
    {"name": "D7_cond_3way",
     "src": "(cond ((> x 2) (* x 10)) ((> x 0) x) (else (- x)))",
     "lanes": [{"x": 3.0}, {"x": 1.0}, {"x": -2.0}, {"x": 5.0}, {"x": -0.5}]},
    # The following stress cases came from the Phase-3b adversarial implementation review (all confirmed
    # to match B=1). They cover gaps D1-D7 missed: matrix-VJP gating with a singular DEAD lane, shared
    # leaves across branches/actsets, nested if (not just loops), and INT/FLOAT merge.
    # R1: INV backward reads a cached inverse slab for ALL lanes; the dead lane's slab is Inf -> must be
    # gated out so grad x stays clean.
    {"name": "R1_inv_singular_dead_lane",
     "src": "(if (> s 0) (trace (inv (scale x (eye 2)))) x)",
     "lanes": [{"s": 1.0, "x": 2.0}, {"s": -1.0, "x": 0.5}]},
    # R2: LOGDET (the Kalman-flagship op) backward under divergence with a singular not-taken lane.
    {"name": "R2_logdet_singular_dead_lane",
     "src": "(if (> s 0) (logdet (scale x (eye 2))) x)",
     "lanes": [{"s": 1.0, "x": 2.0}, {"s": -1.0, "x": 0.5}]},
    # R3: shared scalar leaf x feeds a kept branch AND a dead divergent DIV that is Inf for that lane.
    {"name": "R3_shared_leaf_dead_inf",
     "src": "(* x (if (> s 0) 2.0 (/ 1.0 (- x 1.0))))",
     "lanes": [{"s": 1.0, "x": 1.0}, {"s": -1.0, "x": 3.0}]},
    # R4: a let-bound intermediate read by a FULL-actset node and two nodes under disjoint reduced actsets.
    {"name": "R4_shared_payload_multi_actset",
     "src": "(let ((u (* x x))) (if (> s 0) (* u x) (* u x x)))",
     "lanes": [{"s": 1.0, "x": 2.0}, {"s": -1.0, "x": 3.0}]},
    # R5: nested ifs with DISTINCT leaves per arm; the outer-else lane must contribute zero to inner leaves.
    {"name": "R5_nested_ifs_distinct_leaves",
     "src": "(if (> p 0) (if (> q 0) (* a a) (* b b)) (* c c))",
     "lanes": [{"p": 1.0, "q": 1.0, "a": 2.0, "b": 5.0, "c": 8.0},
               {"p": 1.0, "q": -1.0, "a": 3.0, "b": 6.0, "c": 9.0},
               {"p": -1.0, "q": 1.0, "a": 4.0, "b": 7.0, "c": 10.0}]},
    # R6: asymmetric nested if partitioning the batch into 4 singletons; shared leaf w gets one bucket each.
    {"name": "R6_asymmetric_nested_4lane",
     "src": "(if (> s 0) (if (> t 0) (* w 2.0) (* w 3.0)) (if (> t 0) (* w 5.0) (* w 7.0)))",
     "lanes": [{"s": 1.0, "t": 1.0, "w": 2.0}, {"s": 1.0, "t": -1.0, "w": 2.0},
               {"s": -1.0, "t": 1.0, "w": 2.0}, {"s": -1.0, "t": -1.0, "w": 2.0}]},
    # R7: a divergent loop whose (full-active) SELECT output seeds a shared downstream multiply.
    {"name": "R7_loop_then_shared_downstream",
     "src": "(define (go i x) (if (< i n) (go (+ i 1) (* x x)) x))\n(* w (go 0 x))",
     "lanes": [{"x": 1.1, "n": 1.0, "w": 2.0}, {"x": 1.2, "n": 2.0, "w": 0.5},
               {"x": 1.3, "n": 3.0, "w": 3.0}, {"x": 1.05, "n": 4.0, "w": 1.5}]},
    # R8: INT vs FLOAT merge (both number?, interchangeable) must merge to [2, 5], NOT raise.
    {"name": "R8_int_float_merge",
     "src": "(* 2.0 (if (> t 0) 1 2.5))",
     "lanes": [{"t": 1.0}, {"t": -1.0}]},
    # R9: VEC SELECT with the 1e-8 NORMALIZE clamp subgradient under divergence (dead-lane slab harmless).
    {"name": "R9_vec_normalize_dead_lane",
     "src": "(vsum (if (> s 0) (normalize (vec x x)) (scale x (vec 1.0 1.0))))",
     "lanes": [{"s": 1.0, "x": 0.0}, {"s": -1.0, "x": 3.0}]},
]


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
@pytest.mark.parametrize("p", DIVERGENT, ids=[p["name"] for p in DIVERGENT])
def test_divergent_lane_decomposition(p):
    lanes = p["lanes"]; B = len(lanes)
    keys = list(lanes[0].keys())
    batched = {k: [lane[k] for lane in lanes] for k in keys}
    bout, bgrad = _run(p["src"], batched, B, grad=True)
    assert bout is not None and len(bout) == B, f"{p['name']}: batched result not [B]"
    for b in range(B):
        sout, sgrad = _run(p["src"], {k: [lanes[b][k]] for k in keys}, 1, grad=True)
        assert _close(bout[b], sout[0]), f"{p['name']} lane {b}: fwd batched={bout[b]} scalar={sout[0]}"
        for k in keys:
            assert _close(bgrad[k][b], sgrad[k][0]), \
                f"{p['name']} lane {b} d/d{k}: batched={bgrad[k][b]} scalar={sgrad[k][0]}"


# Structural divergence: a single scalar-aux Val cannot carry different heap cells / shapes per lane.
# These MUST raise (not silently mis-merge to one branch's structure).
MUST_RAISE = [
    {"name": "D5_pair_divergence",
     "src": "(if (= t 1) (cons 1 2) (cons 3 4))",
     "lanes": {"t": [1.0, 0.0]}, "match": "per-lane-divergent structural value unsupported"},
    {"name": "D6p_vec_shape_mismatch",
     "src": "(vsum (if (> s 0) (vec a b) (vec a b a)))",
     "lanes": {"a": [1.0, 1.0], "b": [2.0, 2.0], "s": [1.0, -1.0]},
     "match": "per-lane-divergent structural value unsupported"},
    # Boolean vs number is observable tag divergence (boolean?/number?) -> must raise, not collapse to FLOAT.
    {"name": "R10_bool_vs_number",
     "src": "(boolean? (if (> t 0) (= 1 1) 5))",
     "lanes": {"t": [1.0, -1.0]}, "match": "per-lane-divergent structural value unsupported"},
    # Per-lane structural list (different lengths) shares one scalar aux -> must raise (the loop form of B1).
    {"name": "R11_list_accumulator_loop",
     "src": "(define (f self acc k) (if (= k 0) acc (self self (cons k acc) (- k 1))))\n(f f (quote ()) k)",
     "lanes": {"k": [2.0, 4.0]}, "match": "per-lane-divergent structural value unsupported"},
    # A per-lane VARYING constructor size is structural divergence (VecCell shape is uniform) -> must raise.
    {"name": "R12_nonuniform_constructor_size",
     "src": "(vsum (ones n))",
     "lanes": {"n": [2.0, 5.0]}, "match": "size must be uniform"},
]


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
@pytest.mark.parametrize("p", MUST_RAISE, ids=[p["name"] for p in MUST_RAISE])
def test_divergence_raises(p):
    B = len(next(iter(p["lanes"].values())))
    with pytest.raises(RuntimeError, match=p["match"]):
        _run(p["src"], p["lanes"], B, grad=True)


@pytest.mark.skipif(not RUN.exists(), reason="ndvm_run not built")
def test_nonterminating_lane_capped():
    # A counter that DEcreases never satisfies (< i n), so the loop never terminates; the eval-step cap
    # (NDVM_MAX_STEPS) must raise loudly rather than hang -- the Phase-3b guard for batched divergence
    # where the loop trampolines until the LAST lane terminates.
    src = "(define (go i x) (if (< i n) (go (- i 1) x) x))\n(go 0 x)"
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "p.scm"; sp.write_text(src + "\n")
        bp = Path(d) / "p.bind"; bp.write_text("scalar x 1.0\nscalar n 3.0\n")
        env = dict(os.environ); env["NDVM_MAX_STEPS"] = "200000"
        out = subprocess.run([str(RUN), str(sp), str(bp)], capture_output=True, text=True, env=env)
        assert out.returncode != 0 and "eval step budget exceeded" in out.stderr, out.stderr
