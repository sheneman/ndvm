#!/usr/bin/env python3
"""Randomized DIFFERENTIAL testing driver for the NDVM native runtime (reviewer weakness 8).

Generates a deterministic corpus of well-typed scalar programs over the DMCI-supported subset
(genprog.gen_corpus) and runs THREE gates on each:

  G1 forward      : NDVM ndvm_forward(...)  vs  DMCI oracle evaluate(...)        (float32 tolerance)
  G2 gradient     : NDVM per-parameter d/dp vs  DMCI oracle reverse-mode d/dp   (float32 tolerance)
  G3 fd-gradient  : NDVM per-parameter d/dp vs  central finite differences      (atol ~1e-2)

Every program failing any gate is logged with its source + the disagreeing values and classified:
  real-bug      : NDVM disagrees with BOTH the oracle and finite differences (NDVM is wrong);
  oracle-bug    : NDVM agrees with finite differences but the oracle does not (oracle is wrong);
  tolerance     : disagreement within a few ULP / near a non-smooth point (abs/if branch, large grad);
  generator     : the program tripped an unsupported shape (oracle or NDVM raised / produced 0-output).

Deterministic + reproducible (fixed seed). Freezes the corpus to results/fuzz_corpus.jsonl and the
full per-program gate log to results/fuzz_report.json. Prints a coverage table (feature family x
#programs x per-gate pass rate) and the headline N x gates summary.

Run on an HPC compute node (torch + DMCI + the NDVM extension).
    python3 ndvm/tests/run_fuzz.py --n 200 --seed 1234
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # ndvm/tests -> ndvm -> repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "python"))   # ndvm/python -> ndvm_autograd

from genprog import gen_corpus, FEATURES, GenConfig  # noqa: E402

EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)

# gate tolerances (float32 single-scalar arithmetic)
G1_ATOL, G1_RTOL = 1e-4, 1e-4          # forward value
G2_ATOL, G2_RTOL = 2e-3, 2e-3          # NDVM grad vs oracle grad
G3_ATOL, G3_RTOL = 1e-2, 5e-2          # NDVM grad vs central finite difference (coarse by nature)
FD_EPS = 1e-3


def _close(a, b, atol, rtol):
    if a is None or b is None:
        return a is None and b is None
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= atol + rtol * abs(b)


# ---------------------------------------------------------------------------
# Oracle (DMCI) forward + reverse-mode gradient
# ---------------------------------------------------------------------------
def oracle_eval(src, params, want_grad):
    import torch
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    graph = compile_dmci(src)
    leaves, binds = {}, {}
    for k, v in params.items():
        t = torch.tensor(float(v), requires_grad=want_grad)
        leaves[k] = t
        binds[k] = make_float(t)
    y = unwrap_number(evaluate(graph, binds, **EVAL_KW)).reshape(())
    val = float(y)
    grads = None
    if want_grad:
        gs = torch.autograd.grad(y, [leaves[k] for k in params], allow_unused=True)
        grads = {k: (0.0 if g is None else float(g)) for k, g in zip(params, gs)}
    return val, grads


def oracle_forward_only(src, params):
    """Forward value with no autograd (for finite differences)."""
    import torch
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    graph = compile_dmci(src)
    binds = {k: make_float(torch.tensor(float(v))) for k, v in params.items()}
    y = unwrap_number(evaluate(graph, binds, **EVAL_KW)).reshape(())
    return float(y)


# ---------------------------------------------------------------------------
# NDVM native forward + gradient
# ---------------------------------------------------------------------------
def ndvm_eval(src, params):
    import torch
    from ndvm_autograd import ndvm_forward
    leaves = {k: torch.tensor(float(v), requires_grad=True) for k, v in params.items()}
    out = ndvm_forward(src, leaves, None)
    val = float(out.reshape(()))
    out.backward()
    grads = {k: (0.0 if leaves[k].grad is None else float(leaves[k].grad)) for k in params}
    return val, grads


def ndvm_forward_only(src, params):
    import torch
    from ndvm_autograd import ndvm_forward
    leaves = {k: torch.tensor(float(v)) for k, v in params.items()}
    out = ndvm_forward(src, leaves, None)
    return float(out.reshape(()))


# ---------------------------------------------------------------------------
# Central finite difference of the NDVM forward (gate G3 uses the NDVM forward itself, so G3 tests
# NDVM's backward against NDVM's own forward -- the proper self-consistency of the native tape).
# ---------------------------------------------------------------------------
def ndvm_fd_grad(src, params):
    fd = {}
    for k in params:
        bp = dict(params); bp[k] = params[k] + FD_EPS
        fp = ndvm_forward_only(src, bp)
        bm = dict(params); bm[k] = params[k] - FD_EPS
        fm = ndvm_forward_only(src, bm)
        fd[k] = (fp - fm) / (2 * FD_EPS)
    return fd


# ---------------------------------------------------------------------------
# Classify a failing program
# ---------------------------------------------------------------------------
def classify(rec):
    """rec has g1/g2/g3 booleans + the value dicts. Decide why it failed."""
    g1, g2, g3 = rec["g1"], rec["g2"], rec["g3"]
    if rec.get("err"):
        return "generator"  # oracle or NDVM raised on this shape
    if g1 and g2 and g3:
        return "pass"
    # forward disagreement
    if not g1:
        # if NDVM forward agrees with NDVM-fd-consistency but not oracle -> could be oracle/parity
        return "real-bug" if _forward_real_bug(rec) else "tolerance"
    # gradient disagreement
    # NDVM grad vs oracle (g2) and NDVM grad vs NDVM-fd (g3)
    if not g3 and not g2:
        return "real-bug"          # NDVM backward inconsistent with its own forward AND the oracle
    if not g2 and g3:
        return "oracle-or-tol"     # NDVM self-consistent (fd ok) but differs from oracle -> oracle/parity
    if not g3 and g2:
        return "tolerance"         # fd is coarse near non-smooth points; oracle agrees -> fd artifact
    return "tolerance"


def _forward_real_bug(rec):
    # NaN/inf mismatch or a gross relative difference is a real forward bug
    o = rec["fwd_oracle"]; n = rec["fwd_ndvm"]
    if o is None or n is None:
        return True
    if not (math.isfinite(o) and math.isfinite(n)):
        return not ((math.isnan(o) and math.isnan(n)) or o == n)
    denom = max(1.0, abs(o))
    return abs(o - n) / denom > 1e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--p-loop", type=float, default=0.30)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    sys.setrecursionlimit(200_000)

    cfg = GenConfig(p_loop=args.p_loop)
    corpus = gen_corpus(args.n, seed=args.seed, cfg=cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # freeze the corpus immediately (reproducibility)
    with (out_dir / "fuzz_corpus.jsonl").open("w") as f:
        for r in corpus:
            f.write(json.dumps(r) + "\n")

    records = []
    # per-feature, per-gate counters: feat -> {"n", "g1", "g2", "g3"}
    feat_stats = {feat: {"n": 0, "g1": 0, "g2": 0, "g3": 0} for feat in FEATURES}
    n_g1 = n_g2 = n_g3 = 0
    n_eval = 0
    class_counts = defaultdict(int)

    for r in corpus:
        src, params, feats = r["src"], r["params"], r["features"]
        rec = {"id": r["id"], "src": src, "params": params, "features": feats,
               "g1": False, "g2": False, "g3": False, "err": None}
        try:
            ov, og = oracle_eval(src, params, want_grad=True)
            nv, ng = ndvm_eval(src, params)
            fdg = ndvm_fd_grad(src, params)
            rec["fwd_oracle"] = ov
            rec["fwd_ndvm"] = nv
            rec["grad_oracle"] = og
            rec["grad_ndvm"] = ng
            rec["grad_fd"] = fdg

            rec["g1"] = _close(nv, ov, G1_ATOL, G1_RTOL)
            rec["g2"] = all(_close(ng[k], og[k], G2_ATOL, G2_RTOL) for k in params)
            rec["g3"] = all(_close(ng[k], fdg[k], G3_ATOL, G3_RTOL) for k in params)
        except Exception as e:  # noqa: BLE001
            rec["err"] = f"{type(e).__name__}: {str(e)[:200]}"

        n_eval += 1
        if rec["err"] is None:
            n_g1 += rec["g1"]; n_g2 += rec["g2"]; n_g3 += rec["g3"]
            for feat in feats:
                if feat in feat_stats:
                    fs = feat_stats[feat]
                    fs["n"] += 1
                    fs["g1"] += rec["g1"]; fs["g2"] += rec["g2"]; fs["g3"] += rec["g3"]
        cls = classify(rec)
        rec["class"] = cls
        class_counts[cls] += 1
        records.append(rec)

    # ---- report ----
    host = socket.gethostname()
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = "?"

    report = {
        "host": host, "torch": tv, "seed": args.seed, "n": args.n,
        "tolerances": {"g1": [G1_ATOL, G1_RTOL], "g2": [G2_ATOL, G2_RTOL],
                       "g3": [G3_ATOL, G3_RTOL], "fd_eps": FD_EPS},
        "summary": {"n_eval": n_eval, "g1_pass": n_g1, "g2_pass": n_g2, "g3_pass": n_g3},
        "class_counts": dict(class_counts),
        "feature_coverage": feat_stats,
        "records": records,
    }
    (out_dir / "fuzz_report.json").write_text(json.dumps(report, indent=2))

    # ---- console: coverage table ----
    print("=" * 78)
    print(f"NDVM differential fuzz  host={host} torch={tv} seed={args.seed} N={args.n}")
    print("=" * 78)
    print(f"{'feature family':<24} {'#prog':>6} {'G1 fwd':>9} {'G2 grad':>9} {'G3 fd':>9}")
    print("-" * 62)
    for feat in FEATURES:
        fs = feat_stats[feat]
        n = fs["n"]
        if n == 0:
            print(f"{feat:<24} {0:>6} {'--':>9} {'--':>9} {'--':>9}")
            continue
        print(f"{feat:<24} {n:>6} {fs['g1']/n:>8.1%} {fs['g2']/n:>8.1%} {fs['g3']/n:>8.1%}")
    print("-" * 62)
    ne = max(1, n_eval)
    print(f"{'TOTAL (evaluated)':<24} {n_eval:>6} {n_g1/ne:>8.1%} {n_g2/ne:>8.1%} {n_g3/ne:>8.1%}")
    print(f"\ngate pass counts: G1={n_g1}/{n_eval}  G2={n_g2}/{n_eval}  G3={n_g3}/{n_eval}")
    print(f"class counts: {dict(class_counts)}")

    # ---- console: failing programs (non-pass, non-generator) ----
    fails = [r for r in records if r["class"] not in ("pass",)]
    real = [r for r in records if r["class"] == "real-bug"]
    print(f"\nfailing programs (not full-pass): {len(fails)}  | real-bug candidates: {len(real)}")
    shown = 0
    for r in sorted(fails, key=lambda x: (x["class"] != "real-bug",)):
        if shown >= 25:
            print(f"  ... ({len(fails) - shown} more in fuzz_report.json)")
            break
        shown += 1
        if r["err"]:
            print(f"  [{r['class']:<12}] id={r['id']:<4} ERR {r['err']}")
            print(f"      src: {r['src']}")
            continue
        print(f"  [{r['class']:<12}] id={r['id']:<4} g1={r['g1']} g2={r['g2']} g3={r['g3']}")
        print(f"      src: {r['src']}")
        print(f"      params: {r['params']}")
        print(f"      fwd oracle={r.get('fwd_oracle')} ndvm={r.get('fwd_ndvm')}")
        if not (r["g2"] and r["g3"]):
            print(f"      grad oracle={r.get('grad_oracle')}")
            print(f"      grad ndvm  ={r.get('grad_ndvm')}")
            print(f"      grad fd    ={r.get('grad_fd')}")

    print(f"\nwrote {out_dir / 'fuzz_report.json'}")
    print(f"wrote {out_dir / 'fuzz_corpus.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
