#!/usr/bin/env python3
"""End-to-end co-search frontier experiment, SECOND task (recurrence-heavy candidate class).

Same fixed-wall-clock, LLM-out-of-the-timed-loop protocol as cosearch_e2e.py, but the cached candidate
stream is the recurrence-heavy one produced by cosearch_rec_propose.py (every candidate is built around a
deep bounded iterated map). The target is a sinusoid-plus-trend, discoverable inside the candidate space.
Because each candidate is a deep rollout, the per-candidate interpreter cost is far higher than the first
task's mostly-flat scalar expressions, so this probes whether the frontier shift grows with rollout depth.

    srun -p sheneman -w n128 .venv/bin/python ndvm/profiling/cosearch_rec_e2e.py --run \
         --infile ndvm/profiling/results/cosearch_rec_candidates_valid.jsonl --budget 900 --iters 50
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

# ---- second task target: a sinusoid with a linear trend (smooth, in-family, discoverable by a driven map) ----
def make_task(n=64, noise=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.1, 6.0, n)
    y = torch.sin(1.2 * x) + 0.3 * x + noise * torch.randn(n, generator=g)
    idx = torch.randperm(n, generator=g)
    tr, ho = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    return x[tr], y[tr], x[ho], y[ho]

def params_used(src):
    toks = set(re.findall(r"[a-z]+", src))
    return [p for p in PARAMS if p in toks]

def skeleton(src):
    s = re.sub(r"-?\d+\.?\d*", "C", src)
    s = re.sub(r"\b[abcd]\b", "P", s)
    return re.sub(r"\s+", " ", s).strip()

def calibrate(backend, src, used, xtr, ytr, xho, yho, iters=50, lr=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    leaves = {p: torch.nn.Parameter(0.3 * torch.randn((), generator=g)) for p in used}
    opt = torch.optim.Adam(list(leaves.values()), lr=lr)
    B = xtr.shape[0]; Bh = xho.shape[0]
    try:
        G = compile_dmci(src) if backend == "dmci" else None
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

def do_validate(args):
    xtr, ytr, xho, yho = make_task()
    raw = [json.loads(l) for l in Path(args.infile).read_text().splitlines() if l.strip()]
    seen, valid = set(), []
    B = xtr.shape[0]
    for r in raw:
        src = r["src"] if isinstance(r, dict) else r
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
        if skel in seen:
            continue
        seen.add(skel)
        valid.append({"src": src, "params": used, "skeleton": skel})
    outp = Path(args.out)
    outp.write_text("\n".join(json.dumps(v) for v in valid) + "\n")
    print(f"validated {len(valid)}/{len(raw)} raw -> {len(valid)} unique-structure candidates; wrote {outp}")

def do_run(args):
    xtr, ytr, xho, yho = make_task()
    cands = [json.loads(l) for l in Path(args.infile).read_text().splitlines() if l.strip()]
    if not cands:
        print(f"no validated candidates in {args.infile}; aborting run"); return
    var_y = float(((yho - yho.mean()) ** 2).mean())
    tau = 0.1 * var_y
    print(f"loaded {len(cands)} validated recurrence candidates; var(y_holdout)={var_y:.4f} success tau(MSE)<{tau:.4f} (R2>0.9)")
    calibrate("dmci", cands[0]["src"], cands[0]["params"], xtr, ytr, xho, yho, iters=2)
    calibrate("ndvm", cands[0]["src"], cands[0]["params"], xtr, ytr, xho, yho, iters=2)
    checkpoints = [t for t in (1, 2, 5, 10, 20, 30, 60, 120, 180, 240, 300, 420, 600, 900, 1200, 1800) if t <= args.budget]
    report = {"task": "recurrence-heavy (deep iterated maps); target sin(1.2x)+0.3x",
              "budget_s": args.budget, "iters": args.iters, "tau": tau, "var_y": var_y,
              "n_candidates": len(cands), "backends": {}}
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
            while ci < len(checkpoints) and el >= checkpoints[ci]:
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
              f"best_R2={(1-best/var_y) if best!=float('inf') else float('nan'):.3f} t_first_success={t_first_success}s", flush=True)
    rb = report["backends"]
    if rb["dmci"]["calibrated"]:
        report["frontier_shift_candidates"] = round(rb["ndvm"]["calibrated"] / rb["dmci"]["calibrated"], 1)
        print(f"FRONTIER SHIFT: NDVM calibrated {report['frontier_shift_candidates']}x more candidates in the same budget")
    RES.mkdir(exist_ok=True)
    (RES / "cosearch_rec_e2e_n128.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {RES / 'cosearch_rec_e2e_n128.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--infile", default=str(RES / "cosearch_rec_candidates_valid.jsonl"))
    ap.add_argument("--out", default=str(RES / "cosearch_rec_candidates_valid.jsonl"))
    ap.add_argument("--budget", type=float, default=900.0, help="wall-clock seconds per backend")
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
