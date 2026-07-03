#!/usr/bin/env python3
"""NDVM Phase 0: the profiling contract (design section 18, Phase 0).

Locks the baseline cost model of the CURRENT PyTorch DMCI backend before any native code is written,
so every NDVM speedup is measured against a fixed reference. For each representative object program
(supplied to the compiled evaluator as DATA; only the bound parameters are differentiated) it collects:

  * clean forward / backward wall-clock per iteration (no profiler attached -> authoritative ms);
  * the batch-scaling curve (wall vs payload batch size B) -- tests the autopsy's "overhead-bound,
    D-independent" claim: the interpreter walk is paid once, batch rides the dense payload;
  * (optional, --decompose) the per-bucket cost DECOMPOSITION via cProfile, separately for the
    forward and backward passes -- tagged-value boxing vs evaluator graph-walking vs heap vs dispatch
    vs raw arithmetic vs autograd -- plus per-bucket CALL COUNTS (wrap/unwrap counts, heap ops, ...).

The bucket classification mirrors experiments/exp_a/profile_decomposition.py (the perf autopsy that
produced experiments/exp_a/results/profile_decomposition.txt: forward ~61% tagged-value boxing,
~25% graph-walking, ~8% heap, ~6% dispatch, ~1% raw arithmetic, ~0.5% autograd). It is VENDORED here
(not imported) so this profiler is self-contained and does not depend on the experiments/ tree -- the
ndvm/ subtree is a standalone runtime project. Keep the two rule sets in sync if exp_a changes.

Real oracle API (validated against experiments/pilots/kalman_detinv_pilot.py):

    from neural_compiler.dmci import compile_dmci, as_matrix
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    from neural_compiler.evaluator import evaluate
    g = compile_dmci(src)                              # free vars auto-detected; program baked as data
    tagged = {k: make_float(torch.tensor(v)) for ...}  # scalars (batched [B] tensors allowed)
    tagged["obs"] = as_matrix(obs)                     # matrix/vector inputs (Strategy B, heap-stored)
    y = unwrap_number(evaluate(g, tagged, **EVAL_KW))  # tagged [.,14] -> numeric payload tensor
    y.sum().backward()

Requires torch + the DMCI core (neural_compiler). Per project convention, run on an HPC COMPUTE node
(the login node / Mac lacks torch). Writes results/baseline_<host>.json and a human-readable .txt.

    python3 ndvm/profiling/profile_dmci_baseline.py --iters 50 --batches 1 8 64 256 --decompose
"""
from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # ndvm/profiling -> ndvm -> repo root
sys.path.insert(0, str(REPO_ROOT))

# Match the flagship pilots: tail-call trampolining keeps the interpreter's own eval loop flat, but
# long matrix rollouts still allocate heap linearly, so give it room. (kalman_detinv_pilot.py uses
# the same envelope.) Non-tail recursion would consume Python stack; bump the limit defensively.
EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)


# ---------------------------------------------------------------------------
# Representative object programs spanning the regimes the cost model cares about.
# Each program is DATA fed to the compiled evaluator; `params` are the differentiated bindings
# (batched over B for the scaling curve), `inputs` are non-differentiated scalar feeds, and an
# optional `matrix` factory binds a [.,.] tensor via as_matrix (Strategy B). `steps` records the
# rollout length so per-rollout-step time can be reported for the recursive/matrix regimes.
# ---------------------------------------------------------------------------

def _kalman_src(T: int) -> str:
    # 2D local-level Kalman filter NLL folded through the interpreter (the flagship regime; this is
    # where the "~250 ms/rollout-step" figure lives). Verbatim shape from kalman_detinv_pilot.py.
    return f"""
(loop ((k 0)
       (x  (vec 0.0 0.0))
       (P  (mat (vec 1.0 0.0) (vec 0.0 1.0)))
       (L  0.0))
  (if (= k {T})
      L
      (let* ((Q     (scale q (eye 2)))
             (R     (scale r (eye 2)))
             (Ppred (+ P Q))
             (y     (ref obs k))
             (e     (- y x))
             (S     (+ Ppred R))
             (Sinv  (inv S))
             (Kg    (matmul Ppred Sinv))
             (xnew  (+ x (matvec Kg e)))
             (Pnew  (matmul (- (eye 2) Kg) Ppred))
             (nll   (+ (log (det S)) (dot e (matvec Sinv e)))))
        (recur (+ k 1) xnew Pnew (+ L nll)))))
"""


def build_programs(loop_n: int, kalman_T: int) -> dict:
    return {
        # --- scalar closed-form (autopsy program; direct comparability to profile_decomposition.txt)
        "scalar_mul_add": {
            "regime": "scalar",
            "src": "(+ (* alpha x) beta)",
            "params": {"alpha": 2.0, "beta": 1.0},
            "inputs": {"x": 1.5},
            "batchable": True,
            "steps": 1,
        },
        # --- scalar closed-form with division (Exp B M03 Michaelis-Menten)
        "michaelis_menten": {
            "regime": "scalar",
            "src": "(/ (* Vmax S) (+ Km S))",
            "params": {"Vmax": 2.0, "Km": 0.5},
            "inputs": {"S": 1.5},
            "batchable": True,
            "steps": 1,
        },
        # --- transcendental closed-form (Exp B M05 damped oscillator)
        "damped_oscillator": {
            "regime": "scalar_transcendental",
            "src": "(* A (* (exp (- 0 (* b t))) (cos (* omega t))))",
            "params": {"A": 1.0, "b": 0.3, "omega": 2.0},
            "inputs": {"t": 1.5},
            "batchable": True,
            "steps": 1,
        },
        # --- iterative / loop-heavy: logistic map rolled loop_n steps (data-independent counter branch)
        "logistic_map_loop": {
            "regime": "recursive_scalar",
            "src": f"(loop ((x x0) (k 0)) (if (= k {loop_n}) x (recur (* r (* x (- 1.0 x))) (+ k 1))))",
            "params": {"r": 3.2},
            "inputs": {"x0": 0.4},
            "batchable": True,
            "steps": loop_n,
        },
        # --- matrix / linear-algebra rollout: the flagship Kalman NLL (T steps, obs matrix input)
        f"kalman2d_T{kalman_T}": {
            "regime": "matrix_rollout",
            "src": _kalman_src(kalman_T),
            "params": {"q": 0.05, "r": 0.10},
            "inputs": {},
            "matrix": {"obs": ("randn", (kalman_T, 2))},
            "batchable": False,  # matrix ops over a batched payload axis are out of Phase-0 scope
            "steps": kalman_T,
        },
    }


# ---------------------------------------------------------------------------
# cProfile cost-bucket classification.
# VENDORED from experiments/exp_a/profile_decomposition.py (CATEGORY_RULES) -- single source of truth
# for what counts as boxing vs graph-walk vs heap vs dispatch. Keep in sync if the oracle changes.
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    # Tagged-value wrap/unwrap (the #1 NDVM target: ~61% of forward in the autopsy)
    ("tagged_value.py", "make_float", "tagged_value"),
    ("tagged_value.py", "make_int", "tagged_value"),
    ("tagged_value.py", "make_bool", "tagged_value"),
    ("tagged_value.py", "make_nil", "tagged_value"),
    ("tagged_value.py", "make_pair", "tagged_value"),
    ("tagged_value.py", "make_symbol", "tagged_value"),
    ("tagged_value.py", "make_char", "tagged_value"),
    ("tagged_value.py", "make_closure", "tagged_value"),
    ("tagged_value.py", "make_vector", "tagged_value"),
    ("tagged_value.py", "_make", "tagged_value"),
    ("tagged_value.py", "from_scalar", "tagged_value"),
    ("tagged_value.py", "unwrap_number", "tagged_value"),
    ("tagged_value.py", "unwrap_bool", "tagged_value"),
    ("tagged_value.py", "unwrap_char", "tagged_value"),
    ("tagged_value.py", "unwrap_symbol_id", "tagged_value"),
    ("tagged_value.py", "unwrap_pair_addrs", "tagged_value"),
    ("tagged_value.py", "unwrap_closure", "tagged_value"),
    ("tagged_value.py", "extract_tag", "tagged_value"),
    ("tagged_value.py", "extract_payload", "tagged_value"),
    ("tagged_value.py", "type_index", "tagged_value"),
    ("tagged_value.py", "is_nil", "tagged_value"),
    ("tagged_value.py", "is_pair", "tagged_value"),
    ("tagged_value.py", "is_number", "tagged_value"),
    ("tagged_value.py", "is_symbol", "tagged_value"),
    ("tagged_value.py", "is_closure", "tagged_value"),
    ("tagged_value.py", "is_bool", "tagged_value"),
    ("tagged_value.py", "is_type", "tagged_value"),
    ("tagged_value.py", "to_scalar", "tagged_value"),
    # Dispatch (soft branching)
    ("tagged_value.py", "tagged_if", "dispatch"),
    ("tagged_value.py", "soft_select", "dispatch"),
    # Heap operations
    ("heap.py", "cons", "heap"),
    ("heap.py", "car", "heap"),
    ("heap.py", "cdr", "heap"),
    ("heap.py", "read", "heap"),
    ("heap.py", "write", "heap"),
    ("heap.py", "store", "heap"),
    ("heap.py", "build_list", "heap"),
    ("heap.py", "reset", "heap"),
    ("heap.py", "allocated", "heap"),
    # Tagged ops (cons/car/cdr wrappers, list ops, type predicates via tagged_ops)
    ("tagged_ops.py", "evaluate_tagged_op", "tagged_ops"),
    ("tagged_ops.py", "_tagged_arith", "tagged_ops"),
    ("tagged_ops.py", "_tagged_compare", "tagged_ops"),
    ("tagged_ops.py", "_tagged_logic", "tagged_ops"),
    ("tagged_ops.py", "_op_cons", "heap"),
    ("tagged_ops.py", "_op_car", "heap"),
    ("tagged_ops.py", "_op_cdr", "heap"),
    ("tagged_ops.py", "_op_list", "heap"),
    ("tagged_ops.py", "_op_length", "heap"),
    ("tagged_ops.py", "_op_append", "heap"),
    ("tagged_ops.py", "_op_reverse", "heap"),
    ("tagged_ops.py", "_op_null_p", "tagged_value"),
    ("tagged_ops.py", "_op_pair_p", "tagged_value"),
    ("tagged_ops.py", "_op_number_p", "tagged_value"),
    ("tagged_ops.py", "_op_boolean_p", "tagged_value"),
    ("tagged_ops.py", "_op_symbol_p", "tagged_value"),
    ("tagged_ops.py", "_op_char_p", "tagged_value"),
    ("tagged_ops.py", "_op_procedure_p", "tagged_value"),
    ("tagged_ops.py", "_op_eq", "dispatch"),
    ("tagged_ops.py", "_op_eqv", "dispatch"),
    ("tagged_ops.py", "_op_equal", "dispatch"),
    ("tagged_ops.py", "_deep_equal", "dispatch"),
    ("tagged_ops.py", "materialize_quote", "tagged_value"),
    # Raw arithmetic primitives (and linear-algebra kernels: matmul/inv/det live in primitives)
    ("primitives.py", "evaluate_op", "arithmetic"),
    ("primitives.py", "_op_add", "arithmetic"),
    ("primitives.py", "_op_sub", "arithmetic"),
    ("primitives.py", "_op_mul", "arithmetic"),
    ("primitives.py", "_op_div", "arithmetic"),
    ("primitives.py", "_op_if", "arithmetic"),
    ("primitives.py", "_op_eq", "arithmetic"),
    ("primitives.py", "_op_lt", "arithmetic"),
    ("primitives.py", "_op_gt", "arithmetic"),
    ("primitives.py", "_op_le", "arithmetic"),
    ("primitives.py", "_op_ge", "arithmetic"),
    ("primitives.py", "_op_not", "arithmetic"),
    # Evaluator engine (graph walking)
    ("engine.py", "_eval_lazy_tagged", "evaluator"),
    ("engine.py", "_eval_graph_tagged", "evaluator"),
    ("engine.py", "_evaluate_tagged", "evaluator"),
    ("engine.py", "_eval_call_tagged", "evaluator"),
    ("engine.py", "_eval_call_lazy_tagged", "evaluator"),
    ("engine.py", "_eval_dynamic_call", "evaluator"),
    ("engine.py", "_eval_loop_tagged", "evaluator"),
    ("engine.py", "_eval_loop_lazy_tagged", "evaluator"),
    ("engine.py", "_trace_loop_root_lazy", "evaluator"),
    ("engine.py", "_pack_env", "evaluator"),
    ("engine.py", "_unpack_env", "evaluator"),
    ("engine.py", "_list_to_vec", "evaluator"),
    ("engine.py", "_func_name_to_id", "evaluator"),
    ("engine.py", "_func_id_to_name", "evaluator"),
    ("engine.py", "evaluate", "evaluator"),
    ("engine.py", "_eval_graph", "evaluator"),
    ("engine.py", "_eval_lazy", "evaluator"),
    ("engine.py", "_eval_call", "evaluator"),
    ("engine.py", "_to_tensor", "evaluator"),
    # Symbols
    ("symbols.py", None, "tagged_value"),
]

CATEGORY_LABELS = {
    "tagged_value": "Tagged-value wrap/unwrap",
    "heap": "Heap operations (cons/car/cdr)",
    "dispatch": "Dispatch (tagged_if/eq?/select)",
    "tagged_ops": "Tagged arithmetic wrappers",
    "arithmetic": "Raw arithmetic / linalg primitives",
    "evaluator": "Evaluator graph walking",
    "torch_runtime": "PyTorch runtime / autograd",
}
CATEGORY_ORDER = ["tagged_value", "heap", "dispatch", "tagged_ops", "arithmetic", "evaluator", "torch_runtime"]


def classify_function(filename: str, funcname: str):
    for file_pat, func_pat, category in CATEGORY_RULES:
        if file_pat in filename and (func_pat is None or funcname == func_pat):
            return category
    if "torch" in filename or "autograd" in filename:
        return "torch_runtime"
    return None


def aggregate_categories(stats: pstats.Stats):
    """Aggregate cProfile stats -> {category: {"time": cumulative tottime s, "ncalls": int}}."""
    out: dict[str, dict] = {}
    for (filename, _lineno, funcname), (ncalls, _tot, tottime, _cum, _callers) in stats.stats.items():
        cat = classify_function(filename, funcname)
        if cat is None:
            continue
        rec = out.setdefault(cat, {"time": 0.0, "ncalls": 0})
        rec["time"] += tottime
        rec["ncalls"] += int(ncalls)
    return out


# ---------------------------------------------------------------------------
# Oracle import + binding
# ---------------------------------------------------------------------------

def _try_import_dmci():
    try:
        import torch  # noqa: F401
        from neural_compiler.dmci import compile_dmci, as_matrix  # noqa: F401
        from neural_compiler.evaluator import evaluate  # noqa: F401
        from neural_compiler.runtime.tagged_value import make_float, unwrap_number  # noqa: F401
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _make_params(prog, batch):
    """Differentiated leaf tensors for the program's params (batched [B] when batch>1)."""
    import torch
    params = {}
    for name, val in prog["params"].items():
        if batch > 1:
            params[name] = torch.full((batch,), float(val), requires_grad=True)
        else:
            params[name] = torch.tensor(float(val), requires_grad=True)
    return params


def _make_bindings(prog, params, batch):
    """Tagged-value input dict for evaluate(): params (make_float) + scalar inputs + matrix inputs."""
    import torch
    from neural_compiler.dmci import as_matrix
    from neural_compiler.runtime.tagged_value import make_float
    binds = {name: make_float(p) for name, p in params.items()}
    for name, val in prog.get("inputs", {}).items():
        binds[name] = make_float(torch.tensor(float(val)))
    for name, spec in prog.get("matrix", {}).items():
        kind, shape = spec
        g = torch.Generator().manual_seed(0)
        t = torch.randn(*shape, generator=g) if kind == "randn" else torch.zeros(*shape)
        binds[name] = as_matrix(t)
    return binds


# ---------------------------------------------------------------------------
# Profiling one (program, batch)
# ---------------------------------------------------------------------------

def profile_program(name, prog, iters, batch, decompose):
    import torch
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import unwrap_number

    t = time.perf_counter()
    graph = compile_dmci(prog["src"])
    compile_s = time.perf_counter() - t

    params = _make_params(prog, batch)

    def forward():
        binds = _make_bindings(prog, params, batch)
        y = unwrap_number(evaluate(graph, binds, **EVAL_KW))
        return y.reshape(-1).sum()  # scalar loss; .sum() collapses any batch axis

    def zero_grads():
        for p in params.values():
            p.grad = None

    # Warmup (uncounted): exercises lazy graph construction / interning caches.
    loss = forward(); loss.backward(); zero_grads()

    # --- Phase A: clean wall-clock timing (NO profiler attached -> authoritative ms/iter) ---
    fwd_s = bwd_s = 0.0
    last_loss = None
    for _ in range(iters):
        s = time.perf_counter(); loss = forward(); fwd_s += time.perf_counter() - s
        s = time.perf_counter(); loss.backward(); bwd_s += time.perf_counter() - s
        last_loss = float(loss.detach().reshape(-1)[0])
        zero_grads()

    rec = {
        "name": name,
        "regime": prog["regime"],
        "batch": batch,
        "iters": iters,
        "steps": prog.get("steps", 1),
        "compile_s": compile_s,
        "fwd_ms_per_iter": 1e3 * fwd_s / iters,
        "bwd_ms_per_iter": 1e3 * bwd_s / iters,
        "fwd_bwd_ms_per_iter": 1e3 * (fwd_s + bwd_s) / iters,
        "final_loss": last_loss,
    }
    steps = max(1, prog.get("steps", 1))
    if steps > 1:
        rec["fwd_ms_per_rollout_step"] = rec["fwd_ms_per_iter"] / steps
        rec["fwd_bwd_ms_per_rollout_step"] = rec["fwd_bwd_ms_per_iter"] / steps

    # --- Phase B: per-bucket decomposition via cProfile (forward and backward separately) ---
    if decompose:
        rec["decomposition"] = {
            "forward": _decompose(forward, zero_grads, backward=False, iters=max(10, iters // 2)),
            "backward": _decompose(forward, zero_grads, backward=True, iters=max(10, iters // 2)),
        }
    return rec


def _decompose(forward, zero, backward, iters):
    """cProfile one pass type (forward-only or backward-only) and attribute tottime + ncalls per
    bucket. The un-profiled pass runs OUTSIDE the enable/disable window so the attribution is clean."""
    prof = cProfile.Profile()
    for _ in range(iters):
        if backward:
            loss = forward()                              # forward outside the profiled window
            prof.enable(); loss.backward(); prof.disable()
        else:
            prof.enable(); loss = forward(); prof.disable()
            loss.backward()                               # backward outside the profiled window
        zero()
    stats = pstats.Stats(prof)
    cats = aggregate_categories(stats)
    total_time = sum(c["time"] for c in cats.values())
    buckets = {}
    for cat in CATEGORY_ORDER:
        c = cats.get(cat, {"time": 0.0, "ncalls": 0})
        buckets[cat] = {
            "ms_per_iter": 1e3 * c["time"] / iters,
            "frac_of_profiled": (c["time"] / total_time) if total_time > 0 else 0.0,
            "calls_per_iter": c["ncalls"] / iters,
        }
    return {"buckets": buckets, "profiled_ms_per_iter": 1e3 * total_time / iters}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_decomp(label, decomp):
    lines = [f"  {label}:"]
    lines.append(f"    {'Component':<36s} {'ms/iter':>10s} {'frac':>8s} {'calls/iter':>12s}")
    lines.append(f"    {'-'*36} {'-'*10} {'-'*8} {'-'*12}")
    for cat in CATEGORY_ORDER:
        b = decomp["buckets"][cat]
        lines.append(f"    {CATEGORY_LABELS[cat]:<36s} {b['ms_per_iter']:10.3f} "
                     f"{b['frac_of_profiled']:8.1%} {b['calls_per_iter']:12.1f}")
    lines.append(f"    {'(profiled total)':<36s} {decomp['profiled_ms_per_iter']:10.3f}")
    return "\n".join(lines)


def render_report(meta, rows):
    out = []
    out.append("=" * 78)
    out.append("NDVM Phase 0 baseline cost model -- current PyTorch DMCI backend")
    out.append(f"host={meta['host']}  python={meta['python']}  torch={meta['torch']}  device=cpu")
    out.append(f"iters={meta['iters']}  decompose={meta['decompose']}")
    out.append("=" * 78)
    for r in rows:
        if "error" in r:
            out.append(f"\n[{r['name']:18s} B={r['batch']:<5d}] ERROR: {r['error']}")
            continue
        line = (f"\n[{r['name']:18s} B={r['batch']:<5d}] regime={r['regime']:22s} "
                f"fwd={r['fwd_ms_per_iter']:8.2f}ms  bwd={r['bwd_ms_per_iter']:8.2f}ms  "
                f"fwd+bwd={r['fwd_bwd_ms_per_iter']:8.2f}ms")
        out.append(line)
        if r.get("steps", 1) > 1:
            out.append(f"  rollout steps={r['steps']}  -> fwd {r['fwd_ms_per_rollout_step']:.3f} ms/step  "
                       f"fwd+bwd {r['fwd_bwd_ms_per_rollout_step']:.3f} ms/step")
        if "decomposition" in r:
            out.append(_fmt_decomp("FORWARD decomposition", r["decomposition"]["forward"]))
            out.append(_fmt_decomp("BACKWARD decomposition", r["decomposition"]["backward"]))
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="NDVM Phase 0 baseline profiler for PyTorch DMCI")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64, 256])
    ap.add_argument("--programs", nargs="+", default=None, help="subset of program names (default: all)")
    ap.add_argument("--loop-n", type=int, default=16, help="logistic_map_loop rollout length")
    ap.add_argument("--kalman-T", type=int, default=80, help="Kalman rollout length")
    ap.add_argument("--decompose", action="store_true", help="run the per-bucket cProfile decomposition (B=1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.setrecursionlimit(100_000)

    ok, err = _try_import_dmci()
    if not ok:
        print(f"[phase0] cannot import torch + neural_compiler: {err}")
        print("[phase0] run on an HPC compute node (login node / Mac lacks torch). Skeleton intact.")
        return 1

    import torch
    programs = build_programs(args.loop_n, args.kalman_T)
    names = args.programs or list(programs)

    rows = []
    for name in names:
        if name not in programs:
            print(f"[phase0] unknown program {name!r}; have {list(programs)}", flush=True)
            continue
        prog = programs[name]
        batches = args.batches if prog.get("batchable", True) else [1]
        for b in batches:
            # decomposition only at B=1 (cleanest attribution; batch just scales payload width)
            decompose = args.decompose and b == 1
            try:
                rec = profile_program(name, prog, args.iters, b, decompose)
                rows.append(rec)
                extra = ""
                if rec.get("steps", 1) > 1:
                    extra = f"  ({rec['fwd_ms_per_rollout_step']:.3f} ms/step x{rec['steps']})"
                print(f"[phase0] {name:18s} B={b:<5d} fwd={rec['fwd_ms_per_iter']:8.2f}ms "
                      f"bwd={rec['bwd_ms_per_iter']:7.2f}ms{extra}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[phase0] FAIL {name} B={b}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                rows.append({"name": name, "batch": b, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    meta = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "iters": args.iters,
        "decompose": args.decompose,
        "loop_n": args.loop_n,
        "kalman_T": args.kalman_T,
        "note": "NDVM Phase-0 baseline; bucket rules mirror experiments/exp_a/profile_decomposition.py",
    }
    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else results_dir / f"baseline_{meta['host']}.json"
    out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    report = render_report(meta, rows)
    out_txt = out_json.with_suffix(".txt")
    out_txt.write_text(report)
    print(report)
    print(f"[phase0] wrote {out_json}")
    print(f"[phase0] wrote {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
