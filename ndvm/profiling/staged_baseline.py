#!/usr/bin/env python3
"""Staged-graph baseline + amortization curve (Track 3, reviewer weakness 2/10).

A "staged" baseline pays a one-off cost to lower a known program into a fast executable graph, then each
gradient step is cheap. NDVM and the tuned-eager interpreter pay NOTHING to stage: they consume the
program as data and start producing gradients immediately. This harness measures, per program:

  * STAGING compile cost (paid once): we use two practitioner staging paths and report whichever applies
      - sympy.lambdify for the closed-form scalar programs (symbolic -> a numpy callable), AND
      - jax.jit trace+XLA-compile for ALL programs (works on the loop/Kalman programs sympy can't fold),
        which is the dominant, honestly-measured staging cost (XLA compilation of forward+grad).
  * PER-STEP grad eval cost after staging.
  * The CROSSOVER: how many gradient steps per candidate before the one-off staging cost pays off, i.e.
        staging + n * per_step_staged  <  n * per_step_unstaged
    solved as n* = staging / (per_step_unstaged - per_step_staged).

Two "unstaged" references are read from prior baseline runs (or measured live here):
  - the DMCI tagged oracle per-step fwd+grad (the current backend), and
  - NDVM per-step fwd+grad (native runtime), via ndvm_autograd.

The deliverable is the AMORTIZATION CURVE data: total wall vs number-of-gradient-steps n, three lines per
program (staged-JAX = staging + n*per_step; DMCI = n*per_step; NDVM = n*per_step), with the crossover n*
where staged-JAX overtakes NDVM. The point: in co-search every candidate is fitted for only a HANDFUL of
steps before being replaced, so n is small and staging usually never amortizes; NDVM wins in that regime.

Run on an HPC compute node (needs torch + DMCI + jax + the NDVM .so). jax is optional: if it is missing,
we fall back to sympy-only staging for the closed-form programs and report that.

    python3 ndvm/profiling/staged_baseline.py
"""
from __future__ import annotations

import json
import platform
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "python"))  # ndvm/python -> ndvm_autograd

KALMAN_T = 80
LOOP_N = 16
EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# JAX staging (forward+grad), reusing the hand-written transcriptions from jax_baseline.
# ---------------------------------------------------------------------------

def _jax_stage_and_time(name, prog, iters):
    """Return (staging_ms, per_step_ms) for jit-compiling value_and_grad and then running it.

    staging_ms = wall to trace + XLA-compile (first call, blocked); per_step_ms = median steady-state.
    """
    import jax
    import jax.numpy as jnp
    from jax_baseline import _make_jax_programs, _obs_matrix

    fn = _make_jax_programs()[name]
    pnames = list(prog["params"])
    p0 = jnp.asarray([float(prog["params"][k]) for k in pnames], dtype=jnp.float32)
    inp = {k: jnp.float32(float(v)) for k, v in prog.get("inputs", {}).items()}
    mats = {}
    for mname, spec in prog.get("matrix", {}).items():
        mats[mname] = _obs_matrix(spec[1][0])

    f = lambda p: fn(p, inp, mats)
    vg = jax.jit(jax.value_and_grad(f))

    t0 = time.perf_counter()
    val, grad = vg(p0)
    val.block_until_ready()
    grad.block_until_ready()
    staging_ms = (time.perf_counter() - t0) * 1e3  # includes trace + XLA compile

    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        v, g = vg(p0)
        v.block_until_ready()
        g.block_until_ready()
        ts.append(time.perf_counter() - t0)
    per_step_ms = sorted(ts)[iters // 2] * 1e3
    return staging_ms, per_step_ms, float(val)


# ---------------------------------------------------------------------------
# sympy staging (closed-form scalar programs only) -- the lambdify path.
# ---------------------------------------------------------------------------

def _sympy_stage_and_time(name, prog, iters):
    """sympy.lambdify staging for the closed-form scalar programs. Returns (staging_ms, per_step_ms, val)
    or None if the program is not a simple closed form we transcribe symbolically here."""
    import sympy as sp

    specs = {
        "scalar_mul_add": (lambda al, be, x: al * x + be, ["alpha", "beta"], {"x": 1.5}),
        "michaelis_menten": (lambda vm, km, S: (vm * S) / (km + S), ["Vmax", "Km"], {"S": 1.5}),
        "damped_oscillator": (lambda A, b, om, t: A * (sp.exp(-(b * t)) * sp.cos(om * t)),
                              ["A", "b", "omega"], {"t": 1.5}),
    }
    if name not in specs:
        return None
    build, pnames, inputs = specs[name]

    t0 = time.perf_counter()
    syms = sp.symbols(pnames + list(inputs))
    psyms = syms[:len(pnames)]
    isyms = syms[len(pnames):]
    expr = build(*psyms, *isyms)
    # forward + gradient expressions, lowered to numpy callables (the staged graph)
    grads = [sp.diff(expr, s) for s in psyms]
    fwd_cb = sp.lambdify(list(syms), expr, "numpy")
    grad_cbs = [sp.lambdify(list(syms), gexpr, "numpy") for gexpr in grads]
    staging_ms = (time.perf_counter() - t0) * 1e3

    pvals = [float(prog["params"][k]) for k in pnames]
    ivals = [float(v) for v in inputs.values()]
    args = pvals + ivals

    val = float(fwd_cb(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fwd_cb(*args)
        _ = [cb(*args) for cb in grad_cbs]
        ts.append(time.perf_counter() - t0)
    per_step_ms = sorted(ts)[iters // 2] * 1e3
    return staging_ms, per_step_ms, val


# ---------------------------------------------------------------------------
# Unstaged references: DMCI oracle per-step (fwd+grad) and NDVM per-step (fwd+grad).
# ---------------------------------------------------------------------------

def _oracle_step_time(prog, iters):
    import torch
    from neural_compiler.dmci import compile_dmci, as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number

    g = compile_dmci(prog["src"])
    pnames = list(prog["params"])

    def step():
        leaves = {k: torch.tensor(float(prog["params"][k]), requires_grad=True) for k in pnames}
        binds = {k: make_float(leaves[k]) for k in pnames}
        for k, v in prog.get("inputs", {}).items():
            binds[k] = make_float(torch.tensor(float(v)))
        for name, spec in prog.get("matrix", {}).items():
            kind, shape = spec
            gen = torch.Generator().manual_seed(0)
            t = torch.randn(*shape, generator=gen) if kind == "randn" else torch.zeros(*shape)
            binds[name] = as_matrix(t)
        y = unwrap_number(evaluate(g, binds, **EVAL_KW))
        loss = y.reshape(-1).sum()
        loss.backward()
        return float(loss.detach().reshape(-1)[0])

    val = step()  # warmup
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[iters // 2] * 1e3, val


def _ndvm_step_time(prog, iters):
    import torch
    from ndvm_autograd import ndvm_forward

    pnames = list(prog["params"])
    mats = {}
    for name, spec in prog.get("matrix", {}).items():
        kind, shape = spec
        gen = torch.Generator().manual_seed(0)
        t = torch.randn(*shape, generator=gen) if kind == "randn" else torch.zeros(*shape)
        mats[name] = (t.shape[0], t.shape[1] if t.dim() > 1 else 1, t.reshape(-1))

    def step():
        params = {k: torch.tensor(float(prog["params"][k]), requires_grad=True) for k in pnames}
        params.update({k: torch.tensor(float(v)) for k, v in prog.get("inputs", {}).items()})
        out = ndvm_forward(prog["src"], params, mats or None)
        out.reshape(()).backward()
        return float(out.reshape(()).item())

    val = step()  # warmup (also triggers first-call ext load)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        step()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[iters // 2] * 1e3, val


def _crossover(staging_ms, staged_per_step, unstaged_per_step):
    """n* gradient steps before staging pays off vs an unstaged backend:
       staging + n*staged  <  n*unstaged   ->  n* = staging / (unstaged - staged).
    Returns None if the unstaged backend is not slower per step (staging never pays off)."""
    denom = unstaged_per_step - staged_per_step
    if denom <= 0:
        return None
    return staging_ms / denom


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--curve-max", type=int, default=2000, help="max n (grad steps) for the amortization curve")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.setrecursionlimit(100_000)
    if not _have("torch"):
        print("[staged_baseline] no torch; run on an HPC compute node.")
        return 1
    have_jax = _have("jax")
    have_sympy = _have("sympy")
    have_ndvm = True
    try:
        from ndvm_autograd import ndvm_forward  # noqa: F401
    except Exception as e:  # noqa: BLE001
        have_ndvm = False
        print(f"[staged_baseline] NDVM unavailable: {type(e).__name__}: {str(e)[:80]}")

    from profile_dmci_baseline import build_programs
    programs = build_programs(LOOP_N, KALMAN_T)

    rows = []
    for name, prog in programs.items():
        rec = {"name": name, "regime": prog["regime"]}

        # unstaged references
        dmci_ms, dmci_val = _oracle_step_time(prog, args.iters)
        rec["dmci_per_step_ms"] = dmci_ms
        rec["dmci_val"] = dmci_val
        if have_ndvm:
            try:
                ndvm_ms, ndvm_val = _ndvm_step_time(prog, args.iters)
                rec["ndvm_per_step_ms"] = ndvm_ms
                rec["ndvm_val"] = ndvm_val
            except Exception as e:  # noqa: BLE001
                rec["ndvm_per_step_ms"] = None
                rec["ndvm_err"] = f"{type(e).__name__}: {str(e)[:80]}"

        # staged: JAX (all programs) + sympy (closed-form only)
        if have_jax:
            try:
                s_ms, ps_ms, jval = _jax_stage_and_time(name, prog, args.iters)
                rec["jax_staging_ms"] = s_ms
                rec["jax_per_step_ms"] = ps_ms
                rec["jax_val"] = jval
                rec["jax_match"] = abs(jval - dmci_val) <= 1e-3 + 1e-3 * abs(dmci_val)
                rec["crossover_vs_dmci"] = _crossover(s_ms, ps_ms, dmci_ms)
                if rec.get("ndvm_per_step_ms"):
                    rec["crossover_vs_ndvm"] = _crossover(s_ms, ps_ms, rec["ndvm_per_step_ms"])
            except Exception as e:  # noqa: BLE001
                rec["jax_err"] = f"{type(e).__name__}: {str(e)[:120]}"
        if have_sympy:
            sp_res = _sympy_stage_and_time(name, prog, args.iters)
            if sp_res is not None:
                s_ms, ps_ms, sval = sp_res
                rec["sympy_staging_ms"] = s_ms
                rec["sympy_per_step_ms"] = ps_ms
                rec["sympy_val"] = sval
                rec["sympy_match"] = abs(sval - dmci_val) <= 1e-3 + 1e-3 * abs(dmci_val)
                rec["sympy_crossover_vs_dmci"] = _crossover(s_ms, ps_ms, dmci_ms)
                if rec.get("ndvm_per_step_ms"):
                    rec["sympy_crossover_vs_ndvm"] = _crossover(s_ms, ps_ms, rec["ndvm_per_step_ms"])

        # amortization curve points (a few representative n)
        ns = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, args.curve_max]
        curve = {}
        if "jax_staging_ms" in rec:
            curve["staged_jax"] = [rec["jax_staging_ms"] + n * rec["jax_per_step_ms"] for n in ns]
        curve["dmci"] = [n * dmci_ms for n in ns]
        if rec.get("ndvm_per_step_ms"):
            curve["ndvm"] = [n * rec["ndvm_per_step_ms"] for n in ns]
        rec["curve_ns"] = ns
        rec["curve_total_ms"] = curve
        rows.append(rec)

        msg = (f"[staged_baseline] {name:18s} dmci/step={dmci_ms:8.3f}ms "
               f"ndvm/step={rec.get('ndvm_per_step_ms', float('nan')):8.4f}ms ")
        if "jax_staging_ms" in rec:
            msg += (f"| JAX stage={rec['jax_staging_ms']:8.2f}ms step={rec['jax_per_step_ms']:8.4f}ms "
                    f"X_vs_ndvm={rec.get('crossover_vs_ndvm')}")
        print(msg, flush=True)

    meta = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "have_jax": have_jax,
        "have_sympy": have_sympy,
        "have_ndvm": have_ndvm,
        "iters": args.iters,
        "kalman_T": KALMAN_T,
        "loop_n": LOOP_N,
        "note": "Staged-graph baseline + amortization; staging=JAX jit compile / sympy lambdify, "
                "unstaged refs=DMCI oracle and NDVM native.",
    }
    if have_jax:
        import jax
        meta["jax"] = jax.__version__
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else results_dir / f"staged_baseline_{meta['host']}.json"
    out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"[staged_baseline] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
