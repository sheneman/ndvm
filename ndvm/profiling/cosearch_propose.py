#!/usr/bin/env python3
"""Co-search proposal stage (OFFLINE, untimed): use the LLM to propose candidate model programs for a 1-D
symbolic-regression task, COMPILE-TEST every one on BOTH backends, and cache the valid, structurally-
distinct ones. The cached stream is later replayed under a fixed wall-clock budget (cosearch_e2e.py) with
the LLM fully out of the timed loop, so the measured frontier shift is clean of any LLM/network dependency.

Proposer: qwen/qwen3.6-27b via MindRouter (dense model, better at valid Scheme), 12-16 way concurrency.
Every candidate is validated inline: it must parse, use >=1 free parameter, compile on the DMCI oracle AND
the NDVM native runtime, produce finite predictions, and have DMCI and NDVM agree forward to float32. Only
validated candidates with a NEW structural skeleton are written. Stops at --stop-valid distinct valid
candidates. The compute nodes reach MindRouter and carry the venv, so generation+validation run in one job.

    srun -p sheneman -w n128 .venv/bin/python ndvm/profiling/cosearch_propose.py --stop-valid 250 --workers 16
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

SYSTEM = """You propose candidate model programs for symbolic regression over 1-D data y = f(x).
Output a SINGLE Scheme expression that predicts y from the input variable x and free scalar parameters.

HARD CONSTRAINTS (the target interpreter is restricted; violating these makes the program useless):
- The only input variable is x. Free parameters are single lowercase letters chosen from a, b, c, d (use 1 to 4 of them).
- Every arithmetic operator +, -, *, / takes EXACTLY TWO arguments. Write (* a (* b x)), NEVER (* a b x).
- Allowed operators/functions ONLY: + - * / exp log sqrt sin cos abs, plus decimal numeric literals (e.g. 1.0, 0.5, 2.0).
- You MAY use (let* ((v expr) ...) body) to name shared subexpressions.
- You MAY use ONE bounded loop of the exact form (loop ((s init) (k 0)) (if (= k N) s (recur new-s (+ k 1)))) where N is a small integer literal (2..8), for an iterated map.
- Do NOT use: define, lambda, letrec, list, cons, car, cdr, null?, pair?, and, or, set!, quote, vectors, matrices, or any other names.
- Keep arguments to log and sqrt positive by construction where possible.
- Return ONLY the Scheme expression. No prose, no comments, no code fence."""

HINTS = [
    "a straight line", "a quadratic", "a cubic polynomial", "a rational function (ratio of two linear terms)",
    "an exponential decay", "an exponential growth", "a sum of two exponentials with different rates",
    "a decaying sinusoid (damped oscillation)", "a growing sinusoid", "a pure sine wave with phase",
    "a sum of two sinusoids at different frequencies", "a Gaussian-like bump using exp of a negative square",
    "a logistic / saturating S-curve", "a Michaelis-Menten saturation", "a power law x^p via exp(p*log x)",
    "a square-root growth", "a logarithmic growth", "a product of a polynomial and an exponential",
    "a sinusoid whose amplitude grows linearly", "a sinusoid whose amplitude decays exponentially",
    "a damped oscillation plus a constant offset", "an exponential approach to an asymptote",
    "a ratio of an exponential to a polynomial", "a chirp (sinusoid with x-dependent frequency)",
    "an absolute-value V shape", "a smooth step using a logistic", "a sum of a line and a sine",
    "a product of two sines", "a hyperbolic-tangent-like curve built from exponentials",
    "an iterated logistic map run a few steps as a function of x", "an iterated affine map a few steps",
    "a Newton-like fixed-point iteration of a few steps", "a polynomial divided by an exponential",
    "a sine of a polynomial", "an exponential of a sine", "a damped chirp",
    "a two-term Fourier-style series", "a saturating exponential (1 - exp form)",
    "a reciprocal plus a linear term", "a log of a quadratic",
]

ALLOWED = set("+ - * / exp log sqrt sin cos abs let* loop recur if = ( )".split()) | set(PARAMS) | {"x"}


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
                      {"role": "user", "content": f"Propose one candidate model expression for: {hint}. "
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
    # token-level allowlist: every bareword must be an allowed op/var; reject foreign identifiers
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_*?]*", src):
        if tok not in ALLOWED and tok not in ("let", "abs"):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="max proposal requests (cap)")
    ap.add_argument("--stop-valid", type=int, default=250, help="stop after this many distinct VALID candidates")
    ap.add_argument("--patience", type=int, default=300, help="stop early if no new valid-distinct candidate in this many requests")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "cosearch_candidates_valid.jsonl"))
    args = ap.parse_args()
    global MODEL
    if args.model:
        MODEL = args.model
    key = os.environ.get("MINDROUTER_API_KEY")
    if not key:
        sys.exit("MINDROUTER_API_KEY not found")

    # validators (compile/agreement test on EVERY candidate) -- compute node has torch + NDVM
    import torch
    ROOT = Path(__file__).resolve().parents[2]            # repo root (holds neural_compiler/)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ndvm" / "python"))     # ndvm_autograd
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
