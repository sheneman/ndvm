#!/usr/bin/env python3
"""Co-search proposal stage for the SECOND end-to-end task: a RECURRENCE-HEAVY candidate class.

Identical machinery to cosearch_propose.py (LLM proposes, every candidate is compile-validated on BOTH
backends offline and skeleton-deduped, the LLM is out of the later timed loop), but every candidate MUST be
built around a deep bounded iterated map (loop trip count N in 6..16). This is the rollout-heavy regime in
which the per-candidate interpreter cost is dominated by the recursive walk (the logistic-map loop residual
is 415x in the paper's decomposition vs 8x for a flat scalar expression), so it is deliberately distinct from
the first task's mixed/mostly-flat scalar stream and from the fixed Kalman example.

    srun -p sheneman -w n128 .venv/bin/python ndvm/profiling/cosearch_rec_propose.py --stop-valid 150 --workers 16
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dotenv, openai

for p in ("/mnt/ceph/sheneman/src/nncompile/.env",
          str(Path(__file__).resolve().parents[2] / ".env"), str(Path.home() / ".env")):
    if Path(p).exists():
        dotenv.load_dotenv(p); break

BASE_URL = os.environ.get("MINDROUTER_BASE_URL", "https://mindrouter.uidaho.edu/v1")
MODEL = os.environ.get("MINDROUTER_MODEL", "qwen/qwen3.6-27b")
PARAMS = "abcd"

SYSTEM = """You propose candidate DYNAMICAL-SYSTEM programs for regression over 1-D data y = f(x).
Output a SINGLE Scheme expression that predicts y from the input variable x and free scalar parameters, and
that is built around a bounded ITERATED MAP (a discrete recurrence run for a fixed number of steps).

HARD CONSTRAINTS (the target interpreter is restricted; violating these makes the program useless):
- The only input variable is x. Free parameters are single lowercase letters chosen from a, b, c, d (use 1 to 4 of them).
- Every arithmetic operator +, -, *, / takes EXACTLY TWO arguments. Write (* a (* b x)), NEVER (* a b x).
- Allowed operators/functions ONLY: + - * / exp log sqrt sin cos abs, plus decimal numeric literals (e.g. 1.0, 0.5, 2.0).
- You MUST use a bounded iterated map of the EXACT form
      (loop ((s INIT) (k 0)) (if (= k N) s (recur NEW-S (+ k 1))))
  where N is an integer literal between 6 and 16, INIT is the initial state (may use x and parameters), and
  NEW-S is the update rule g(s, x, parameters) you design (the recurrence). The loop returns the final state.
- You MAY use (let* ((v expr) ...) body) to name shared subexpressions, and you MAY wrap the loop's result in a
  small closed-form transform (for example multiply by an exponential of x, or add an offset parameter).
- The recurrence NEW-S MUST genuinely depend on s (use s in the update), so the program is a real iteration.
- Do NOT use: define, lambda, letrec, list, cons, car, cdr, null?, pair?, and, or, set!, quote, vectors, matrices, or any other names.
- Keep arguments to log and sqrt positive by construction where possible.
- Return ONLY the Scheme expression. No prose, no comments, no code fence."""

HINTS = [
    "an affine iterated map s <- a*s + b driven by x in the initial state",
    "an affine map whose update adds a term in sin(x)", "an affine map whose update adds a term in cos(x)",
    "a logistic iterated map s <- a*s*(1-s) seeded from a function of x",
    "a Newton-style fixed-point iteration for a root that depends on x",
    "a damped recurrence s <- a*s + b*x that relaxes toward a fixed point",
    "an iterated map driven by an exponential of x", "an exponential relaxation s <- s + a*(b - s)",
    "a cobweb iteration of a unimodal map as a function of x",
    "an iterated map whose result is multiplied by a decaying exponential of x",
    "a two-parameter affine recurrence with an x-dependent forcing term",
    "an iterated rotation-like map combining sin and cos of the state",
    "a saturating recurrence using a/(1+s) style updates", "a quadratic map s <- a*s*s + b driven by x",
    "an iterated average s <- 0.5*s + 0.5*g(x) for several steps", "a contraction map toward a*x",
    "an iterated map plus a linear-in-x trend added afterward",
    "an iterated sine map s <- a*sin(s) + b*x", "an iterated map of x through a polynomial update",
    "a relaxation toward a sinusoid in x", "an iterated map with an exp-of-x initial condition",
    "a logistic-like growth iteration toward a carrying capacity that depends on x",
    "an iterated map whose output approximates a decaying oscillation in x",
    "a Mann-style averaged fixed-point iteration", "an iterated affine-plus-bias recurrence",
    "a damped Newton iteration for sqrt-like behavior in x",
]

# A recurrence candidate introduces local bindings (loop state/counter, let* names), so a strict
# allowlist of barewords would reject every valid loop program. We pre-filter with a DENYLIST of the
# forbidden special forms only; the heavy validate() (compile on BOTH backends + float32 agreement) is
# the real gate that decides whether a candidate is usable.
FORBIDDEN = {"define", "lambda", "letrec", "list", "cons", "car", "cdr", "null?", "pair?",
             "set!", "quote", "and", "or", "vector", "matrix", "cond", "case", "begin", "when", "unless"}


def extract(resp: str) -> str:
    if not resp:
        return ""
    m = re.search(r"```(?:scheme|lisp)?\s*\n?(.*?)```", resp, re.DOTALL)
    s = (m.group(1) if m else resp).strip().replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()


def one(client, hint, idx):
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=8192, temperature=1.0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Propose one candidate iterated-map model expression for: {hint}. "
                                                  f"Use parameters a,b,c as needed and make it structurally distinct."}])
        msg = r.choices[0].message
        return {"idx": idx, "hint": hint, "src": extract(msg.content or getattr(msg, "reasoning_content", None) or "")}
    except Exception as e:
        return {"idx": idx, "hint": hint, "src": "", "error": str(e)[:160]}


def params_used(src):
    toks = set(re.findall(r"[a-z]+", src))
    return [p for p in PARAMS if p in toks]


def skeleton(src):
    s = re.sub(r"-?\d+\.?\d*", "C", src)
    s = re.sub(r"\b[abcd]\b", "P", s)
    return re.sub(r"\s+", " ", s).strip()


def light_ok(src):
    if src.count("(") != src.count(")") or "(" not in src:
        return False
    if "loop" not in src or "recur" not in src:    # this task REQUIRES an iterated map
        return False
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_*?!]*", src))
    if toks & FORBIDDEN:                            # reject forbidden special forms; allow local var names
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="max proposal requests (cap)")
    ap.add_argument("--stop-valid", type=int, default=150, help="stop after this many distinct VALID candidates")
    ap.add_argument("--patience", type=int, default=400, help="stop early if no new valid-distinct candidate in this many requests")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "cosearch_rec_candidates_valid.jsonl"))
    args = ap.parse_args()
    global MODEL
    if args.model:
        MODEL = args.model
    key = os.environ.get("MINDROUTER_API_KEY")
    if not key:
        sys.exit("MINDROUTER_API_KEY not found")

    import torch
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ndvm" / "python"))
    from neural_compiler.dmci import compile_dmci
    from neural_compiler.evaluator import evaluate
    from neural_compiler.runtime.tagged_value import make_float, unwrap_number
    from ndvm_autograd import ndvm_forward
    EVAL_KW = dict(max_iter=2_000_000, max_depth=2_000_000, max_heap=8_000_000)
    xg = torch.linspace(0.1, 6.0, 48); B = xg.shape[0]

    def validate(src):
        used = params_used(src)
        if not used or not light_ok(src):
            return None
        try:
            G = compile_dmci(src)
            binds = {"x": make_float(xg)}
            for p in used:
                binds[p] = make_float(0.5 * torch.ones(B))
            pd = unwrap_number(evaluate(G, binds, **EVAL_KW)).reshape(-1)
            pn = ndvm_forward(src, {"x": xg, **{p: 0.5 * torch.ones(B) for p in used}}, None).reshape(-1)
            if not (torch.isfinite(pd).all() and torch.isfinite(pn).all()):
                return None
            if float((pd - pn).abs().max()) > 1e-3 * (1 + float(pd.abs().max())):
                return None
        except Exception:
            return None
        return {"src": src, "params": used, "skeleton": skeleton(src)}

    client = openai.OpenAI(base_url=BASE_URL, api_key=key, timeout=120, max_retries=2)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    seen_skel, n_req, n_nonempty, n_valid, last_improve = set(), 0, 0, 0, 0
    t0 = time.perf_counter()
    fout = outp.open("w")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, client, HINTS[i % len(HINTS)], i): i for i in range(args.n)}
        for f in as_completed(futs):
            n_req += 1
            src = f.result().get("src", "")
            if src:
                n_nonempty += 1
                v = validate(src)
                if v and v["skeleton"] not in seen_skel:
                    seen_skel.add(v["skeleton"]); n_valid += 1; last_improve = n_req
                    fout.write(json.dumps(v) + "\n"); fout.flush()
            if n_req % 20 == 0:
                print(f"  req {n_req} | non-empty {n_nonempty} | VALID-distinct {n_valid} | {time.perf_counter()-t0:.0f}s", flush=True)
            if n_valid >= args.stop_valid:
                print(f"  reached target {args.stop_valid} valid-distinct", flush=True); break
            if n_req - last_improve >= args.patience and n_valid > 0:
                print(f"  plateau: no new valid-distinct in {args.patience} requests (diversity saturated at {n_valid})", flush=True); break
        for fu in futs:
            fu.cancel()
    fout.close()
    print(f"DONE: {n_req} requests, {n_nonempty} non-empty, {n_valid} valid-distinct -> {outp} "
          f"({time.perf_counter()-t0:.0f}s, model={MODEL})")


if __name__ == "__main__":
    main()
