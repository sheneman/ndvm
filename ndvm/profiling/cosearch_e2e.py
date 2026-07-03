#!/usr/bin/env python3
"""End-to-end co-search frontier experiment (fixed wall-clock, LLM out of the timed loop).

A cached stream of LLM-proposed candidate model programs (cosearch_propose.py -> validated here) is
replayed through BOTH backends under the SAME fixed wall-clock budget on a symbolic-regression task.
Because NDVM's gradients are bit-identical to the DMCI oracle, a given candidate fits to the SAME
parameters on either backend; the only thing that differs is HOW MANY candidates each backend gets
through in the budget. So this measures whether the inner-loop speedup moves the SEARCH FRONTIER:
candidates calibrated, successful fits, best held-out loss, and structural diversity of discoveries.

Two modes:
  --validate : load raw candidates, keep those that compile on BOTH backends, agree forward to float32,
               use >=1 free parameter, and produce finite predictions. Writes the validated stream.
  --run      : fixed wall-clock per backend; calibrate the stream (cycling with fresh restarts when
               exhausted); report frontier metrics for DMCI vs NDVM.

Run on a compute node (torch + DMCI + NDVM ext).
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[0] / "python"))
import torch
from neural_compiler.dmci import compile_dmci
from neural_compiler.evaluator import evaluate
from neural_compiler.runtime.tagged_value import make_float, unwrap_number
from ndvm_autograd import ndvm_forward

EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)
PARAMS = "abcd"
RES = HERE / "results"

# ---- symbolic-regression task: a damped oscillator (inside the candidate space), train/held-out split ----
def make_task(n=64, noise=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.1, 6.0, n)
    y = 1.0 * torch.exp(-0.3 * x) * torch.cos(2.0 * x) + noise * torch.randn(n, generator=g)
    idx = torch.randperm(n, generator=g)
    tr, ho = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    return x[tr], y[tr], x[ho], y[ho]

def params_used(src):
    toks = set(re.findall(r"[a-z]+", src))
    return [p for p in PARAMS if p in toks]

def skeleton(src):
    s = re.sub(r"-?\d+\.?\d*", "C", src)          # numeric literals -> C
    s = re.sub(r"\b[abcd]\b", "P", s)             # params -> P
    return re.sub(r"\s+", " ", s).strip()

# ---- one full calibration on a backend; returns (held_mse, train_loss) or (inf, inf) on failure ----
def calibrate(backend, src, used, xtr, ytr, xho, yho, iters=50, lr=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    leaves = {p: torch.nn.Parameter(0.3 * torch.randn((), generator=g)) for p in used}
    opt = torch.optim.Adam(list(leaves.values()), lr=lr)
    B = xtr.shape[0]; Bh = xho.shape[0]
    try:
        G = compile_dmci(src) if backend == "dmci" else None    # compile ONCE per candidate (fair to both)
        def fwd(xb, Bn):
            if backend == "dmci":
                binds = {"x": make_float(xb)}
                for p, t in leaves.items():
                    binds[p] = make_float(t * torch.ones(Bn))
                return unwrap_number(evaluate(G, binds, **EVAL_KW)).reshape(-1)
            pd = {"x": xb}
            for p, t in leaves.items():
                pd[p] = t * torch.ones(Bn)
            return ndvm_forward(src, pd, None).reshape(-1)
        for _ in range(iters):
            opt.zero_grad()
            loss = ((fwd(xtr, B) - ytr) ** 2).mean()
            if not torch.isfinite(loss):
                return float("inf"), float("inf")
            loss.backward(); opt.step()
        with torch.no_grad():
            hm = float(((fwd(xho, Bh) - yho) ** 2).mean())
        return (hm if hm == hm else float("inf")), float(loss.detach())
    except Exception:
        return float("inf"), float("inf")

def load_raw(path):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("src"):
            out.append(r["src"])
    return out

def do_validate(args):
    xtr, ytr, xho, yho = make_task()
    raw = load_raw(args.infile)
    seen, valid = set(), []
    B = xtr.shape[0]
    for src in raw:
        used = params_used(src)
        if not used:
            continue
        skel = skeleton(src)
        try:
            G = compile_dmci(src)
            binds = {"x": make_float(xtr)}
            leaves = {p: torch.tensor(0.5) for p in used}
            for p, t in leaves.items():
                binds[p] = make_float(t * torch.ones(B))
            pd = unwrap_number(evaluate(G, binds, **EVAL_KW)).reshape(-1)
            pn = ndvm_forward(src, {"x": xtr, **{p: leaves[p] * torch.ones(B) for p in used}}, None).reshape(-1)
            if not (torch.isfinite(pd).all() and torch.isfinite(pn).all()):
                continue
            if float((pd - pn).abs().max()) > 1e-3 * (1 + float(pd.abs().max())):
                continue
        except Exception:
            continue
        if skel in seen:        # dedup by structural skeleton
            continue
        seen.add(skel)
        valid.append({"src": src, "params": used, "skeleton": skel})
    outp = Path(args.out)
    outp.write_text("\n".join(json.dumps(v) for v in valid) + "\n")
    print(f"validated {len(valid)}/{len(raw)} raw -> {len(valid)} unique-structure candidates; wrote {outp}")

def do_run(args):
    xtr, ytr, xho, yho = make_task()
    cands = [json.loads(l) for l in Path(args.infile).read_text().splitlines() if l.strip()]
    var_y = float(((yho - yho.mean()) ** 2).mean())
    tau = 0.1 * var_y                       # success = held-out R^2 > 0.9
    print(f"loaded {len(cands)} validated candidates; var(y_holdout)={var_y:.4f} success tau(MSE)<{tau:.4f} (R2>0.9)")
    # warm both backends
    calibrate("dmci", cands[0]["src"], cands[0]["params"], xtr, ytr, xho, yho, iters=2)
    calibrate("ndvm", cands[0]["src"], cands[0]["params"], xtr, ytr, xho, yho, iters=2)
    checkpoints = [t for t in (1, 2, 5, 10, 20, 30, 60, 120, 180, 240, 300, 420, 600, 900, 1200, 1800) if t <= args.budget]
    report = {"budget_s": args.budget, "tau": tau, "var_y": var_y, "n_candidates": len(cands), "backends": {}}
    for backend in ("dmci", "ndvm"):
        t0 = time.perf_counter(); n = 0; best = float("inf"); cycle = 0
        succ_struct = set(); traj = []; ci = 0; t_first_success = None
        while time.perf_counter() - t0 < args.budget:
            c = cands[n % len(cands)]
            seed = 1000 * cycle + (n % len(cands))
            hm, _ = calibrate(backend, c["src"], c["params"], xtr, ytr, xho, yho, iters=args.iters, seed=seed)
            n += 1
            if hm < best:
                best = hm
            if hm < tau:
                if t_first_success is None:
                    t_first_success = round(time.perf_counter() - t0, 2)
                succ_struct.add(c["skeleton"])
            if n % len(cands) == 0:
                cycle += 1
            el = time.perf_counter() - t0
            while ci < len(checkpoints) and el >= checkpoints[ci]:   # record discovery-vs-wallclock trajectory
                traj.append({"t": checkpoints[ci], "calibrated": n,
                             "distinct_success": len(succ_struct),
                             "best_r2": (None if best == float("inf") else round(1 - best / var_y, 4))})
                ci += 1
            if backend == "dmci" and n >= args.dmci_cap:
                break
        elapsed = time.perf_counter() - t0
        report["backends"][backend] = {
            "elapsed_s": round(elapsed, 1), "calibrated": n, "successful_distinct": len(succ_struct),
            "best_held_mse": (None if best == float("inf") else round(best, 6)),
            "best_held_r2": (None if best == float("inf") else round(1 - best / var_y, 4)),
            "t_first_success_s": t_first_success, "trajectory": traj,
        }
        print(f"[{backend}] {elapsed:.0f}s: calibrated={n} distinct_successes={len(succ_struct)} "
              f"best_R2={1-best/var_y:.3f} t_first_success={t_first_success}s")
    rb = report["backends"]
    if rb["dmci"]["calibrated"]:
        report["frontier_shift_candidates"] = round(rb["ndvm"]["calibrated"] / rb["dmci"]["calibrated"], 1)
        print(f"FRONTIER SHIFT: NDVM calibrated {report['frontier_shift_candidates']}x more candidates in the same budget")
    RES.mkdir(exist_ok=True)
    (RES / "cosearch_e2e_n128.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {RES / 'cosearch_e2e_n128.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--infile", default=str(RES / "cosearch_candidates_valid.jsonl"),
                    help="candidate stream; --run replays the committed validated cache (default). "
                         "--validate expects the raw propose output: pass --infile results/cosearch_candidates_raw.jsonl")
    ap.add_argument("--out", default=str(RES / "cosearch_candidates_valid.jsonl"))
    ap.add_argument("--budget", type=float, default=600.0, help="wall-clock seconds per backend")
    ap.add_argument("--iters", type=int, default=50, help="Adam steps per calibration")
    ap.add_argument("--dmci-cap", type=int, default=100000, help="safety cap on DMCI calibrations")
    args = ap.parse_args()
    if args.validate:
        do_validate(args)
    elif args.run:
        do_run(args)
    else:
        ap.error("choose --validate or --run")

if __name__ == "__main__":
    main()
