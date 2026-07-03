#!/usr/bin/env python3
"""Generate forward reference values from the PyTorch DMCI oracle for the Phase-1 equivalence gate.

For each Phase-0 program (same source + params/inputs as the profiler), compile via compile_dmci,
bind scalars with make_float and matrices with as_matrix, evaluate, and record the raw forward output
unwrap_number(...).item(). Emits results/oracle_refs.json containing, per program: the exact source
string (which NDVM parses identically), the scalar bindings, any matrix binding (flattened, so NDVM is
fed bit-identical inputs), and the oracle's scalar result. Run on an HPC compute node (needs torch).
"""
from __future__ import annotations
import json, sys, socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def kalman_src(T):
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


def main():
    import torch
    from neural_compiler.dmci import compile_dmci, as_matrix
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)

    T = 80
    g = torch.Generator().manual_seed(0)
    obs = torch.randn(T, 2, generator=g)

    progs = [
        # --- the five Phase-0 regime programs ---
        {"name": "scalar_mul_add", "src": "(+ (* alpha x) beta)",
         "scalars": {"alpha": 2.0, "beta": 1.0, "x": 1.5}, "matrix": None},
        {"name": "michaelis_menten", "src": "(/ (* Vmax S) (+ Km S))",
         "scalars": {"Vmax": 2.0, "Km": 0.5, "S": 1.5}, "matrix": None},
        {"name": "damped_oscillator", "src": "(* A (* (exp (- 0 (* b t))) (cos (* omega t))))",
         "scalars": {"A": 1.0, "b": 0.3, "omega": 2.0, "t": 1.5}, "matrix": None},
        {"name": "logistic_map_loop",
         "src": "(loop ((x x0) (k 0)) (if (= k 16) x (recur (* r (* x (- 1.0 x))) (+ k 1))))",
         "scalars": {"r": 3.2, "x0": 0.4}, "matrix": None},
        {"name": "kalman2d_T80", "src": kalman_src(T),
         "scalars": {"q": 0.05, "r": 0.10},
         "matrix": {"name": "obs", "rows": T, "cols": 2, "data": obs.reshape(-1).tolist()}},
        # --- adversarial coverage: more Exp-B equations ---
        {"name": "coulomb", "src": "(/ (* k (* q1 q2)) (* r r))",
         "scalars": {"k": 8.99, "q1": 1.6, "q2": -1.6, "r": 2.0}, "matrix": None},
        {"name": "arrhenius", "src": "(* A (exp (- 0 (* Ea T))))",
         "scalars": {"A": 1.5, "Ea": 0.5, "T": 2.0}, "matrix": None},
        {"name": "power_law", "src": "(* a (pow x b))",
         "scalars": {"a": 2.0, "x": 3.0, "b": 1.5}, "matrix": None},
        {"name": "logistic_growth", "src": "(/ K (+ 1.0 (* (- K 1.0) (exp (- 0 (* rr t))))))",
         "scalars": {"K": 2.0, "rr": 0.5, "t": 3.0}, "matrix": None},
        # --- control flow + binding forms ---
        {"name": "factorial_letrec",
         "src": "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))",
         "scalars": {}, "matrix": None},
        {"name": "higher_order_twice",
         "src": "((lambda (f x) (f (f x))) (lambda (y) (+ y a)) z)",
         "scalars": {"a": 1.5, "z": 2.0}, "matrix": None},
        {"name": "cond_abs", "src": "(cond ((< x 0.0) (- 0 x)) (else x))",
         "scalars": {"x": -3.0}, "matrix": None},
        {"name": "nested_let", "src": "(let ((u (+ a b))) (let ((v (* u u))) (- v a)))",
         "scalars": {"a": 2.0, "b": 3.0}, "matrix": None},
        {"name": "begin_seq", "src": "(begin (+ a 1) (* a 2))",
         "scalars": {"a": 3.0}, "matrix": None},
        {"name": "min_max_abs", "src": "(+ (min a b) (+ (max a b) (abs (- 0 a))))",
         "scalars": {"a": -2.0, "b": 3.0}, "matrix": None},
        {"name": "list_car_cdr", "src": "(car (cdr (list a b c)))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0}, "matrix": None},
        {"name": "recursive_define",
         "src": "(define (poly x n) (if (= n 0) 0.0 (+ (* alpha x) (poly x (- n 1)))))\n(poly x 3)",
         "scalars": {"alpha": 2.0, "x": 1.5}, "matrix": None},
        # --- float32 parity edge cases (clamp / non-finite) ---
        {"name": "sqrt_clamp_neg", "src": "(sqrt (- 0 1.0))", "scalars": {}, "matrix": None},
        {"name": "log_clamp_neg", "src": "(log (- 0 1.0))", "scalars": {}, "matrix": None},
        {"name": "div_by_zero", "src": "(/ 1.0 0.0)", "scalars": {}, "matrix": None},
        # --- vector ops that return scalars (so unwrap_number is meaningful) ---
        {"name": "cross_vsum", "src": "(vsum (cross (vec 1.0 2.0 3.0) (vec 4.0 5.0 6.0)))",
         "scalars": {}, "matrix": None},
        {"name": "normalize_ref", "src": "(ref (normalize (vec 3.0 4.0 0.0)) 1)",
         "scalars": {}, "matrix": None},
        # --- gradient-coverage programs: exercise VJPs the above set does not (params -> grads) ---
        {"name": "logdet_grad", "src": "(logdet (mat (vec a b) (vec b c)))",
         "scalars": {"a": 2.0, "b": 0.5, "c": 1.5}, "matrix": None},
        {"name": "trace_matmul", "src": "(trace (matmul (mat (vec a b) (vec c d)) (mat (vec a b) (vec c d))))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "matrix": None},
        {"name": "transpose_sq", "src": "(trace (matmul (transpose (mat (vec a b) (vec c d))) (mat (vec a b) (vec c d))))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "matrix": None},
        {"name": "norm_grad", "src": "(norm (vec a b))",
         "scalars": {"a": 3.0, "b": 4.0}, "matrix": None},
        {"name": "normalize_grad", "src": "(ref (normalize (vec a b)) 0)",
         "scalars": {"a": 3.0, "b": 4.0}, "matrix": None},
        {"name": "cross_grad", "src": "(vsum (cross (vec a b c) (vec d e f)))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0}, "matrix": None},
        {"name": "outer_trace", "src": "(trace (outer (vec a b) (vec c d)))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "matrix": None},
        {"name": "ewmul_vsum", "src": "(vsum (* (vec a b) (vec c d)))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "matrix": None},
        {"name": "vec_vsum", "src": "(vsum (vec a b c))",
         "scalars": {"a": 1.5, "b": 2.5, "c": 3.5}, "matrix": None},
        {"name": "mat_trace", "src": "(trace (mat (vec a b) (vec c d)))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "matrix": None},
        {"name": "matvec_dot", "src": "(dot (matvec (mat (vec a b) (vec c d)) (vec e f)) (vec e f))",
         "scalars": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 0.5, "f": 1.5}, "matrix": None},
        {"name": "mod_rem_grad", "src": "(+ (modulo a b) (remainder a b))",
         "scalars": {"a": 7.5, "b": 2.3}, "matrix": None},
    ]

    FD_EPS = 1e-3

    rows = []
    for p in progs:
        rec = {"name": p["name"], "src": p["src"], "scalars": p["scalars"], "matrix": p["matrix"]}
        try:
            graph = compile_dmci(p["src"])

            def forward(scalar_vals, want_grad):
                binds, leaves = {}, {}
                for k, v in scalar_vals.items():
                    t = torch.tensor(float(v), requires_grad=want_grad)
                    leaves[k] = t
                    binds[k] = make_float(t)
                if p["matrix"]:
                    binds[p["matrix"]["name"]] = as_matrix(obs)
                y = unwrap_number(evaluate(graph, binds, **EVAL_KW)).reshape(())
                return y, leaves

            y, _ = forward(p["scalars"], want_grad=False)
            rec["oracle_result"] = float(y)

            # reverse-mode autograd gradient d(output)/d(param) -- the tight reference
            grads = {}
            if p["scalars"]:
                yg, leaves = forward(p["scalars"], want_grad=True)
                yg.backward()
                grads = {k: (float(leaves[k].grad) if leaves[k].grad is not None else 0.0)
                         for k in p["scalars"]}
            rec["grads"] = grads

            # central finite-difference gradient -- secondary sanity cross-check
            fd = {}
            for k in p["scalars"]:
                base = dict(p["scalars"])
                base[k] = p["scalars"][k] + FD_EPS
                fp = float(forward(base, False)[0])
                base[k] = p["scalars"][k] - FD_EPS
                fm = float(forward(base, False)[0])
                fd[k] = (fp - fm) / (2 * FD_EPS)
            rec["fd_grads"] = fd

            gstr = " ".join(f"d/d{k}={v:.5g}" for k, v in grads.items())
            print(f"[oracle] {p['name']:18s} = {rec['oracle_result']:.9g}  {gstr}", flush=True)
        except Exception as ex:  # noqa: BLE001
            import traceback; traceback.print_exc()
            rec["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            print(f"[oracle] FAIL {p['name']}: {rec['error']}", flush=True)
        rows.append(rec)

    out = HERE / "results"; out.mkdir(exist_ok=True)
    (out / "oracle_refs.json").write_text(json.dumps(
        {"host": socket.gethostname(), "torch": torch.__version__, "programs": rows}, indent=2))
    print(f"[oracle] wrote {out / 'oracle_refs.json'}")


if __name__ == "__main__":
    main()
