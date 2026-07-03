#!/usr/bin/env python3
"""JAX hand-written baseline for the 5 NDVM profiling programs (Track 3, reviewer weakness 2/10).

This is the "why not just write the model directly in JAX?" baseline. For each of the 5 representative
object programs we hand-transcribe the SAME math into jnp, differentiate it with jax.grad, jit-compile
forward and forward+grad, batch the parameter axis with jax.vmap, and validate the forward value against
the DMCI tagged oracle to float32. We report forward and forward+grad ms per program.

The honest framing (returned alongside the numbers): a hand-written JAX program is expected to be far
faster than any interpreter on a KNOWN program, because JAX has the program's source and can fuse and
compile it. That is exactly the condition co-search does NOT enjoy: the candidate program is not known in
advance, it is proposed and revised inside the search loop, so there is nothing to hand-write and nothing
to pay a one-off compile for. NDVM/tuned-eager consume the program AS DATA and pay zero staging cost per
candidate. The JAX column therefore sharpens, rather than weakens, the program-as-data argument: it is the
upper bound you reach only after you already know the model.

Run on an HPC compute node (the venv has jax==0.10.1 CPU; the Mac/login node lacks torch + jax):

    python3 ndvm/profiling/jax_baseline.py
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

# Kalman rollout length (matches profile_dmci_baseline default kalman_T=80)
KALMAN_T = 80
LOOP_N = 16


def _try_imports():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        import torch  # noqa: F401
        from neural_compiler.dmci import compile_dmci, as_matrix  # noqa: F401
        from neural_compiler.evaluator import evaluate  # noqa: F401
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number  # noqa: F401
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Hand-written JAX transcriptions of the 5 programs.
# Each takes a 1-D param vector p (param-major, matching prog["params"] order) and the fixed scalar/matrix
# inputs, and returns a scalar. The obs matrix is generated with the SAME seed-0 torch.randn the oracle
# harness uses, then handed in as a jnp array, so the forward values are directly comparable.
# ---------------------------------------------------------------------------

def _make_jax_programs():
    import jax.numpy as jnp

    def scalar_mul_add(p, inp, mats):
        alpha, beta = p[0], p[1]
        x = inp["x"]
        return alpha * x + beta

    def michaelis_menten(p, inp, mats):
        Vmax, Km = p[0], p[1]
        S = inp["S"]
        return (Vmax * S) / (Km + S)

    def damped_oscillator(p, inp, mats):
        A, b, omega = p[0], p[1], p[2]
        t = inp["t"]
        return A * (jnp.exp(-(b * t)) * jnp.cos(omega * t))

    def logistic_map_loop(p, inp, mats):
        r = p[0]
        x = inp["x0"]
        for _ in range(LOOP_N):
            x = r * (x * (1.0 - x))
        return x

    def kalman2d(p, inp, mats):
        q, r = p[0], p[1]
        obs = mats["obs"]  # [T, 2]
        T = obs.shape[0]
        I2 = jnp.eye(2)
        x = jnp.zeros(2)
        P = jnp.eye(2)
        L = 0.0
        Q = q * I2
        R = r * I2
        for k in range(T):
            Ppred = P + Q
            y = obs[k]
            e = y - x
            S = Ppred + R
            Sinv = jnp.linalg.inv(S)
            Kg = Ppred @ Sinv
            x = x + Kg @ e
            P = (I2 - Kg) @ Ppred
            nll = jnp.log(jnp.linalg.det(S)) + e @ (Sinv @ e)
            L = L + nll
        return L

    return {
        "scalar_mul_add": scalar_mul_add,
        "michaelis_menten": michaelis_menten,
        "damped_oscillator": damped_oscillator,
        "logistic_map_loop": logistic_map_loop,
        f"kalman2d_T{KALMAN_T}": kalman2d,
    }


def _obs_matrix(T):
    """Same seed-0 torch.randn the oracle/NDVM harness uses, as a jnp array."""
    import jax.numpy as jnp
    import torch
    g = torch.Generator().manual_seed(0)
    t = torch.randn(T, 2, generator=g)
    return jnp.asarray(t.numpy())


# ---------------------------------------------------------------------------
# Oracle value (DMCI) for the float32 validation.
# ---------------------------------------------------------------------------

def _oracle_value(prog):
    import torch
    from neural_compiler.dmci import compile_dmci, as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)
    g = compile_dmci(prog["src"])
    binds = {k: make_float(torch.tensor(float(v))) for k, v in prog["params"].items()}
    for k, v in prog.get("inputs", {}).items():
        binds[k] = make_float(torch.tensor(float(v)))
    for name, spec in prog.get("matrix", {}).items():
        kind, shape = spec
        gen = torch.Generator().manual_seed(0)
        t = torch.randn(*shape, generator=gen) if kind == "randn" else torch.zeros(*shape)
        binds[name] = as_matrix(t)
    y = unwrap_number(evaluate(g, binds, **EVAL_KW))
    return float(y.reshape(()).item())


def _median_ms(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = fn()
        # block until the device computation is done (CPU XLA is sync, but be explicit)
        try:
            r.block_until_ready()
        except AttributeError:
            pass
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1e3


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256, help="vmap batch size for the batched columns")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.setrecursionlimit(100_000)
    ok, err = _try_imports()
    if not ok:
        print(f"[jax_baseline] cannot import jax + torch + neural_compiler: {err}")
        print("[jax_baseline] run on an HPC compute node.")
        return 1

    import jax
    import jax.numpy as jnp
    from profile_dmci_baseline import build_programs

    programs = build_programs(LOOP_N, KALMAN_T)
    jax_fns = _make_jax_programs()

    rows = []
    for name, prog in programs.items():
        fn = jax_fns[name]
        pnames = list(prog["params"])
        p0 = jnp.asarray([float(prog["params"][k]) for k in pnames], dtype=jnp.float32)
        inp = {k: jnp.float32(float(v)) for k, v in prog.get("inputs", {}).items()}
        mats = {}
        for mname, spec in prog.get("matrix", {}).items():
            kind, shape = spec
            mats[mname] = _obs_matrix(shape[0])

        # forward and grad closures over fixed inputs/mats
        f = lambda p: fn(p, inp, mats)
        gradf = jax.grad(f)
        f_jit = jax.jit(f)
        gradf_jit = jax.jit(gradf)

        # validate forward value vs oracle (float32)
        oracle = _oracle_value(prog)
        jval = float(f_jit(p0).block_until_ready())
        match = abs(jval - oracle) <= 1e-3 + 1e-3 * abs(oracle)

        # validate grad is finite (sanity); we do not have an analytic grad oracle handed in here, the
        # oracle-equivalence claim is on the FORWARD value; jax.grad is the gold AD so its grad is reference.
        gval = gradf_jit(p0)
        gval.block_until_ready()
        grad_finite = bool(jnp.all(jnp.isfinite(gval)))

        # warmup (jit compile) -- excluded from timing
        f_jit(p0).block_until_ready()
        gradf_jit(p0).block_until_ready()

        fwd_ms = _median_ms(lambda: f_jit(p0), args.iters)
        # forward+grad: value_and_grad jitted, the practitioner's fit-step cost
        vg = jax.jit(jax.value_and_grad(f))
        vg(p0)[0].block_until_ready()
        fwdgrad_ms = _median_ms(lambda: vg(p0)[0], args.iters)

        # batched (vmap over param axis B) -- the co-search "many candidates" axis
        B = args.batch
        pb = jnp.broadcast_to(p0, (B, p0.shape[0]))
        vmap_f = jax.jit(jax.vmap(f))
        vmap_vg = jax.jit(jax.vmap(jax.value_and_grad(f)))
        try:
            vmap_f(pb).block_until_ready()
            vmap_vg(pb)[0].block_until_ready()
            bfwd_ms = _median_ms(lambda: vmap_f(pb), args.iters)
            bfwdgrad_ms = _median_ms(lambda: vmap_vg(pb)[0], args.iters)
        except Exception as e:  # noqa: BLE001
            bfwd_ms = bfwdgrad_ms = float("nan")
            print(f"[jax_baseline] {name} vmap failed: {type(e).__name__}: {str(e)[:80]}")

        rec = {
            "name": name,
            "regime": prog["regime"],
            "fwd_ms": fwd_ms,
            "fwdgrad_ms": fwdgrad_ms,
            "batched_fwd_ms": bfwd_ms,
            "batched_fwdgrad_ms": bfwdgrad_ms,
            "batch": B,
            "jax_value": jval,
            "oracle_value": oracle,
            "oracle_match_f32": match,
            "grad_finite": grad_finite,
        }
        rows.append(rec)
        print(f"[jax_baseline] {name:18s} fwd={fwd_ms:8.4f}ms fwd+grad={fwdgrad_ms:8.4f}ms "
              f"(B={B}: fwd={bfwd_ms:8.4f} fwd+grad={bfwdgrad_ms:8.4f})  "
              f"match={match} (jax={jval:.5f} oracle={oracle:.5f})", flush=True)

    meta = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "iters": args.iters,
        "kalman_T": KALMAN_T,
        "loop_n": LOOP_N,
        "note": "Hand-written JAX baseline; forward values validated vs DMCI oracle float32.",
    }
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else results_dir / f"jax_baseline_{meta['host']}.json"
    out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"[jax_baseline] all_match={all(r['oracle_match_f32'] for r in rows)}")
    print(f"[jax_baseline] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
