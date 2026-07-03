#!/usr/bin/env python3
"""Structure-cached staging baseline (MLSys review residual weakness #1).

The strongest objection to the program-as-data argument is a staging system that REUSES a compiled graph
across candidates that share a program skeleton and differ only in numeric constants: it pays the XLA
compile once per distinct structure rather than once per candidate. The paper previously conceded this
regime without measuring it. This benchmark measures it directly, exactly as requested: K closed-form
skeleton families, each instantiated with varying constants (reuse rate r = candidates per skeleton),
compared across the four conditions the reviewer named

  (A) per-candidate staging    : XLA-compile every candidate (no structural reuse)         -> cost ~ stage + run
  (B) structure-cached staging : XLA-compile once per skeleton, reuse across its r settings -> cost ~ stage/r + run
  (C) NDVM program-as-data     : no compile, fit each candidate through the interpreter      -> cost ~ ndvm_run
  (D) NDVM batched lanes       : no compile, one structural walk over the r settings as lanes-> cost ~ ndvm_batched/cand

reporting the crossover reuse rate r* at which the strongest staging (B) overtakes the strongest NDVM (D).

Method. We measure clean per-skeleton primitives (the JAX fit's XLA compile cost `stage`, its compiled
per-candidate run `staged_run`, the NDVM unbatched per-candidate fit, and the NDVM batched per-candidate
fit at each reuse rate) and report the per-candidate cost of each condition as a function of r, in the same
amortization style as the paper's existing staging crossover. Because varying constants within a skeleton
IS a parameter batch, the cached-staging regime is precisely the regime NDVM's batch amortization targets.

Validation gate: every family's JAX transcription agrees with the NDVM interpreter forward to float32, and
the batched NDVM gradient equals the unbatched one, before any timing. Problem scale (D data points, iters
Adam steps) matches the paper's co-search task. Honest caveat reported with the result: the per-candidate
numeric work here is small (low-dimensional, few steps), the regime co-search actually occupies; a much
larger per-candidate compute or many gradient steps would shift the crossover toward staging.

    srun -p sheneman -w n128 .venv/bin/python ndvm/profiling/cosearch_cached_staging.py --iters 50
"""
from __future__ import annotations
import argparse, json, platform, socket, statistics, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE.parents[0] / "python"))
RES = HERE / "results"

D = 44        # data points per fit (matches the co-search task: n=64, 70% train)
LR = 0.05


def make_families():
    import jax.numpy as jnp
    return {
        "linear":      {"src": "(+ (* a x) b)",
                        "jax": lambda th, x: th[0] * x + th[1], "p": ["a", "b"], "true": [1.3, -0.5]},
        "quadratic":   {"src": "(+ (+ (* a (* x x)) (* b x)) c)",
                        "jax": lambda th, x: th[0] * x * x + th[1] * x + th[2], "p": ["a", "b", "c"], "true": [0.4, -0.7, 1.0]},
        "exp_decay":   {"src": "(* a (exp (* b x)))",
                        "jax": lambda th, x: th[0] * jnp.exp(th[1] * x), "p": ["a", "b"], "true": [2.0, -0.5]},
        "damped_osc":  {"src": "(* a (* (exp (* b x)) (cos (* c x))))",
                        "jax": lambda th, x: th[0] * jnp.exp(th[1] * x) * jnp.cos(th[2] * x), "p": ["a", "b", "c"], "true": [1.5, -0.3, 2.0]},
        "mm_rational": {"src": "(/ (* a x) (+ b x))",
                        "jax": lambda th, x: (th[0] * x) / (th[1] + x), "p": ["a", "b"], "true": [3.0, 1.0]},
        "sin_lin":     {"src": "(+ (* a (sin (* b x))) (* c x))",
                        "jax": lambda th, x: th[0] * jnp.sin(th[1] * x) + th[2] * x, "p": ["a", "b", "c"], "true": [1.2, 1.5, 0.3]},
    }


def gen_candidates(fam, r, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor(fam["true"])
    return [(base + 0.3 * torch.randn(len(fam["true"]), generator=g)).tolist() for _ in range(r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50, help="Adam steps per candidate fit")
    ap.add_argument("--reuse", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(RES / "cached_staging_n128.json"))
    args = ap.parse_args()

    import torch, jax
    import jax.numpy as jnp
    from ndvm_autograd import ndvm_forward
    jax.config.update("jax_platform_name", "cpu")

    fams = make_families()
    xnp = torch.linspace(0.2, 5.0, D)
    xj = jnp.asarray(xnp.numpy(), dtype=jnp.float32)
    targets = {k: torch.tensor([float(v) for v in fam["jax"](jnp.asarray(fam["true"], dtype=jnp.float32), xj)])
               for k, fam in fams.items()}
    yj = {k: jnp.asarray(targets[k].numpy(), dtype=jnp.float32) for k in fams}

    def median(xs):
        return sorted(xs)[len(xs) // 2]

    # ---- NDVM fit of B candidate constant-settings of a family (batched=one walk over B*D lanes) ----
    def ndvm_fit(fam_key, consts, iters, batched):
        fam = fams[fam_key]; B = len(consts); pnames = fam["p"]; y = targets[fam_key]
        if batched:
            th = torch.tensor(consts, requires_grad=True)
            opt = torch.optim.Adam([th], lr=LR)
            x_lane = xnp.repeat(B); ytar = y.repeat(B)
            for _ in range(iters):
                opt.zero_grad()
                binds = {"x": x_lane}
                for j, nm in enumerate(pnames):
                    binds[nm] = th[:, j].repeat_interleave(D)
                pred = ndvm_forward(fam["src"], binds, None).reshape(-1)
                loss = ((pred - ytar) ** 2).mean()
                loss.backward(); opt.step()
            return float(loss.detach())
        last = 0.0
        for c in range(B):
            thc = torch.tensor(consts[c], requires_grad=True)
            o = torch.optim.Adam([thc], lr=LR)
            for _ in range(iters):
                o.zero_grad()
                binds = {"x": xnp}
                for j, nm in enumerate(pnames):
                    binds[nm] = thc[j] * torch.ones(D)
                pred = ndvm_forward(fam["src"], binds, None).reshape(-1)
                loss = ((pred - y) ** 2).mean()
                loss.backward(); o.step()
            last = float(loss.detach())
        return last

    # ---- JAX: the full Adam fit compiled into one graph (lax.scan); single-candidate, reusable ----
    def make_jfit(fam_key, iters):
        fam = fams[fam_key]; y = yj[fam_key]
        def loss(th):
            return jnp.mean((fam["jax"](th, xj) - y) ** 2)
        def fit_one(th0):
            def body(carry, _):
                th, m, v, t = carry
                val, g = jax.value_and_grad(loss)(th)
                t = t + 1
                m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
                th = th - LR * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
                return (th, m, v, t), val
            z = jnp.zeros_like(th0)
            _, vals = jax.lax.scan(body, (th0, z, z, jnp.float32(0.0)), None, length=iters)
            return vals[-1]
        return jax.jit(fit_one)

    # ---- validation gate ----
    print("=== validation gate (forward agreement + batched-vs-unbatched grad) ===", flush=True)
    for k, fam in fams.items():
        th = torch.tensor(fam["true"]); binds = {"x": xnp}
        for j, nm in enumerate(fam["p"]):
            binds[nm] = th[j] * torch.ones(D)
        pn = ndvm_forward(fam["src"], binds, None).reshape(-1)
        pj = torch.tensor([float(v) for v in fam["jax"](jnp.asarray(fam["true"], dtype=jnp.float32), xj)])
        fwd_err = float((pn - pj).abs().max())
        cs = gen_candidates(fam, 3, seed=99)
        thb = torch.tensor(cs, requires_grad=True); xl = xnp.repeat(3); yl = targets[k].repeat(3)
        bb = {"x": xl}
        for j, nm in enumerate(fam["p"]):
            bb[nm] = thb[:, j].repeat_interleave(D)
        lb = ((ndvm_forward(fam["src"], bb, None).reshape(-1) - yl) ** 2).reshape(3, D).mean(1).sum()
        lb.backward(); gb = thb.grad.clone(); gu = torch.zeros_like(gb)
        for c in range(3):
            thc = torch.tensor(cs[c], requires_grad=True); bu = {"x": xnp}
            for j, nm in enumerate(fam["p"]):
                bu[nm] = thc[j] * torch.ones(D)
            lu = ((ndvm_forward(fam["src"], bu, None).reshape(-1) - targets[k]) ** 2).mean()
            lu.backward(); gu[c] = thc.grad
        grad_err = float((gb - gu).abs().max())
        ok = fwd_err < 1e-3 and grad_err < 1e-3
        print(f"  {k:12s} fwd_err={fwd_err:.2e} grad_err={grad_err:.2e} {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            print("VALIDATION FAILED -- aborting", flush=True); return 1

    # ---- warmup (NDVM ext load + JAX backend init) ----
    print("=== warmup ===", flush=True)
    wc = {k: gen_candidates(fams[k], 4, seed=7) for k in fams}
    for k in fams:
        ndvm_fit(k, wc[k], 3, batched=True); ndvm_fit(k, wc[k], 3, batched=False)
        jf = make_jfit(k, 3); jf(jnp.asarray(fams[k]["true"], dtype=jnp.float32)).block_until_ready()

    # ---- per-skeleton primitives: JAX stage (compile) + staged_run, NDVM unbatched per-candidate ----
    print(f"=== primitives (iters={args.iters}, D={D}, median of {args.repeats}) ===", flush=True)
    prim = {}
    for k in fams:
        jf = make_jfit(k, args.iters)
        th0 = jnp.asarray(fams[k]["true"], dtype=jnp.float32)
        t0 = time.perf_counter(); jf(th0).block_until_ready(); first = (time.perf_counter() - t0) * 1e3
        runs = []
        for _ in range(max(args.repeats, 5)):
            t0 = time.perf_counter(); jf(th0).block_until_ready(); runs.append((time.perf_counter() - t0) * 1e3)
        staged_run = median(runs); stage = max(first - staged_run, 0.0)
        c1 = gen_candidates(fams[k], 1, seed=3)
        nd_times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter(); ndvm_fit(k, c1, args.iters, False); nd_times.append((time.perf_counter() - t0) * 1e3)
        ndvm_data = median(nd_times)
        prim[k] = {"stage_ms": round(stage, 3), "staged_run_ms": round(staged_run, 4), "ndvm_data_ms": round(ndvm_data, 4)}
        print(f"  {k:12s} stage={stage:8.2f}ms staged_run={staged_run:7.4f}ms ndvm_data={ndvm_data:7.4f}ms", flush=True)

    K = len(fams)
    stage_tot = sum(prim[k]["stage_ms"] for k in fams)
    staged_run_tot = sum(prim[k]["staged_run_ms"] for k in fams)
    ndvm_data_tot = sum(prim[k]["ndvm_data_ms"] for k in fams)

    # ---- reuse sweep: directly time NDVM batched per skeleton over r settings; model staging from primitives ----
    print(f"\n=== reuse-rate sweep (K={K} skeletons) ===", flush=True)
    sweep = []
    for r in args.reuse:
        cand = {k: gen_candidates(fams[k], r, seed=1000 + r) for k in fams}
        N = K * r
        # NDVM batched: one fit over r lanes per skeleton (directly timed)
        ndvm_batched_tot = 0.0
        for k in fams:
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter(); ndvm_fit(k, cand[k], args.iters, batched=True); ts.append(time.perf_counter() - t0)
            ndvm_batched_tot += median(ts) * 1e3
        # condition totals (ms) for the whole stream of N candidates
        percand_staging = N * (stage_tot / K) + N * (staged_run_tot / K)   # compile EVERY candidate (no reuse)
        cached_staging = stage_tot + N * (staged_run_tot / K)              # compile once per skeleton, reuse
        ndvm_data = N * (ndvm_data_tot / K)                                # serial interpreter fits
        ndvm_batched = ndvm_batched_tot                                    # one walk per skeleton over r lanes
        row = {"reuse_r": r, "n_candidates": N,
               "per_candidate_staging_ms": round(percand_staging, 2),
               "cached_staging_ms": round(cached_staging, 2),
               "ndvm_program_as_data_ms": round(ndvm_data, 2),
               "ndvm_batched_ms": round(ndvm_batched, 2),
               "cached_over_ndvm_batched": round(cached_staging / ndvm_batched, 3)}
        sweep.append(row)
        print(f"  r={r:4d} N={N:5d} | A per-cand-stage {percand_staging:10.1f} | B cached {cached_staging:9.1f} | "
              f"C ndvm-data {ndvm_data:9.1f} | D ndvm-batched {ndvm_batched:9.1f} ms | B/D={cached_staging/ndvm_batched:6.2f}", flush=True)

    rstar = next((row["reuse_r"] for row in sweep if row["cached_staging_ms"] <= row["ndvm_batched_ms"]), None)
    meta = {"host": socket.gethostname(), "python": platform.python_version(), "jax": jax.__version__,
            "torch": torch.__version__, "iters": args.iters, "D": D, "K_skeletons": K, "lr": LR,
            "families": list(fams.keys()), "repeats": args.repeats,
            "note": "A=per-candidate staging (compile every candidate), B=structure-cached staging (compile "
                    "once per skeleton, reuse across constants), C=NDVM program-as-data, D=NDVM batched lanes. "
                    "Staging totals modeled from measured per-skeleton primitives (stage compile + staged run); "
                    "NDVM batched directly timed per reuse rate. r* = candidates-per-skeleton where B overtakes "
                    "D. Co-search streams are structurally distinct (r approx 1). Small per-candidate compute "
                    "(low-dim, few steps) is the co-search regime; larger compute/more steps shift r* toward staging."}
    out = {"meta": meta, "primitives": prim, "crossover_reuse_rstar_B_over_D": rstar, "sweep": sweep}
    RES.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\ncrossover r* (cached staging B overtakes NDVM batched D) = {rstar}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
