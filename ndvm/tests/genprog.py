#!/usr/bin/env python3
"""Typed random generator over the DMCI-SUPPORTED subset, for differential testing (reviewer
weakness 8: correctness empirical + suite too small).

CRITICAL constraint (the infinite-recursion footgun): the DMCI interpreter SILENTLY RETURNS 0 for
unsupported ops, and only supports the comparison set ``= < > <= >=``. An unsupported op in a loop
guard therefore never terminates. So this generator emits ONLY the proven-supported surface:

  numeric literals; bound variables (the K named params + locals);
  binary  + - * /  ; unary  sin cos exp sqrt log abs ;
  let / let* (scalar bindings) ; if with a comparison guard (= < > <= >=) ;
  BOUNDED recursion via the loop / recur form, mirroring the logistic_map_loop /
  kalman2d shapes in profile_dmci_baseline.py: a counter ``k`` that counts up to a
  COMPILE-TIME-CONSTANT bound N and a single scalar accumulator updated each step.

Every generated program is a WELL-TYPED, SCALAR-VALUED expression over the K named params. Numeric
domains are kept finite-and-differentiable so a NaN/inf does not defeat the differential gates:
  * ``/`` denominators are wrapped to ``(+ 1.0 (* d d))`` (>= 1, smooth) -- never zero;
  * ``sqrt`` / ``log`` arguments are wrapped to ``(+ 1.0 (* a a))`` (>= 1, smooth) -- always in-domain;
  * ``exp`` arguments are squashed by a leading ``0.1 *`` and a tanh-free bound is unnecessary because
    inputs stay small; we additionally cap literal/param magnitude so exp does not overflow float32.

The generator is deterministic given a seed. ``gen_corpus(n, seed)`` returns a list of records
``{"id", "src", "params", "features"}`` where ``params`` maps each used param name to a sampled value
in a benign range, and ``features`` is the set of feature families the program exercises (for the
coverage table). Programs that do not reference any param are rejected (a gradient gate needs at least
one differentiated leaf).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Supported surface. DO NOT add anything the DMCI interpreter silently 0-returns.
# ---------------------------------------------------------------------------
BINOPS = ["+", "-", "*", "/"]
UNOPS = ["sin", "cos", "exp", "sqrt", "log", "abs"]
COMPARES = ["=", "<", ">", "<=", ">="]

# feature-family tags used for the coverage table
FEATURES = [
    "literal", "param", "binop", "unary_transcendental", "unary_other",
    "div_guarded", "let", "let_star", "if_compare", "loop_recur",
]


@dataclass
class GenConfig:
    n_params: int = 3            # K named params per program (some may go unused -> trimmed)
    depth: int = 4              # expression depth cap
    p_loop: float = 0.30         # probability the program is wrapped in a bounded loop
    loop_min: int = 2
    loop_max: int = 12          # loop bound N is a compile-time constant in [loop_min, loop_max]
    param_lo: float = 0.2        # sampled param values kept small + positive-ish so exp/log stay finite
    param_hi: float = 1.5
    max_let_binds: int = 2


@dataclass
class _Ctx:
    rng: random.Random
    cfg: GenConfig
    param_names: list
    used_params: set = field(default_factory=set)
    feats: set = field(default_factory=set)
    counter: list = field(default_factory=lambda: [0])

    def fresh(self, prefix="t"):
        self.counter[0] += 1
        return f"{prefix}{self.counter[0]}"


def _lit(ctx: _Ctx) -> str:
    ctx.feats.add("literal")
    # small magnitudes; keep exp/log args benign. Include a couple of "nice" constants.
    v = ctx.rng.choice([
        round(ctx.rng.uniform(-1.5, 1.5), 3),
        round(ctx.rng.uniform(0.2, 1.5), 3),
        1.0, 0.5, 2.0,
    ])
    return f"{float(v)}"


def _var(ctx: _Ctx, locals_in_scope: list) -> str:
    # prefer params (so the program depends on differentiated leaves), but locals are valid too
    pool = list(ctx.param_names)
    if locals_in_scope and ctx.rng.random() < 0.4:
        name = ctx.rng.choice(locals_in_scope)
    else:
        name = ctx.rng.choice(pool)
    if name in ctx.param_names:
        ctx.used_params.add(name)
        ctx.feats.add("param")
    return name


def _atom(ctx: _Ctx, locals_in_scope: list) -> str:
    if ctx.rng.random() < 0.55:
        return _var(ctx, locals_in_scope)
    return _lit(ctx)


def _guarded_pos(ctx: _Ctx, expr: str) -> str:
    """Wrap expr into (+ 1.0 (* expr expr)): smooth, >= 1, keeps sqrt/log/div in-domain."""
    return f"(+ 1.0 (* {expr} {expr}))"


def _gen_expr(ctx: _Ctx, depth: int, locals_in_scope: list, no_let_star: bool = False) -> str:
    """no_let_star: forbid emitting a let* node here. Set when we are directly inside a let* binding
    RHS, because a let* nested as a let* binding's RHS is mishandled by BOTH backends (the DMCI
    compiler raises UnsupportedOperatorError on it and NDVM silently evaluates it to 0). Documented
    in run_fuzz.py as a found cross-backend bug; excluded from the corpus to keep generated programs
    inside the surface both runtimes agree on."""
    if depth <= 0:
        return _atom(ctx, locals_in_scope)

    # weighted choice of node kind
    kinds = ["atom", "binop", "unary", "let", "if"]
    weights = [0.18, 0.34, 0.22, 0.13, 0.13]
    kind = ctx.rng.choices(kinds, weights=weights)[0]

    if kind == "atom":
        return _atom(ctx, locals_in_scope)

    if kind == "binop":
        op = ctx.rng.choice(BINOPS)
        a = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        b = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        ctx.feats.add("binop")
        if op == "/":
            ctx.feats.add("div_guarded")
            return f"(/ {a} {_guarded_pos(ctx, b)})"
        return f"({op} {a} {b})"

    if kind == "unary":
        op = ctx.rng.choice(UNOPS)
        a = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        if op in ("sqrt", "log"):
            ctx.feats.add("unary_transcendental")
            return f"({op} {_guarded_pos(ctx, a)})"
        if op in ("sin", "cos", "exp"):
            ctx.feats.add("unary_transcendental")
            if op == "exp":
                # squash to keep exp arg bounded (float32-safe)
                return f"(exp (* 0.1 {a}))"
            return f"({op} {a})"
        ctx.feats.add("unary_other")  # abs
        return f"(abs {a})"

    if kind == "let":
        # if forbidden from emitting let* here, force a plain let
        star = (not no_let_star) and ctx.rng.random() < 0.5
        nb = ctx.rng.randint(1, ctx.cfg.max_let_binds)
        names, binds = [], []
        scope = list(locals_in_scope)
        for _ in range(nb):
            nm = ctx.fresh("u")
            # in let*, later binds may see earlier ones; in let they may not
            rhs_scope = scope if star else locals_in_scope
            # The ENTIRE RHS subtree of a let* binding must contain NO let* anywhere (the DMCI
            # compiler's let* desugaring does not recurse into its own binding RHS, so any nested
            # let* there raises UnsupportedOperatorError; NDVM silently 0s it). Propagate the ban
            # into the whole RHS subtree (no_let_star sticks through every child) when star.
            rhs = _gen_expr(ctx, depth - 1, rhs_scope, no_let_star=(no_let_star or star))
            binds.append(f"({nm} {rhs})")
            names.append(nm)
            if star:
                scope.append(nm)
        body_scope = locals_in_scope + names
        # a let* body is NOT a binding RHS, so let* is allowed there (unless an enclosing ban applies)
        body = _gen_expr(ctx, depth - 1, body_scope, no_let_star=no_let_star)
        ctx.feats.add("let_star" if star else "let")
        kw = "let*" if star else "let"
        return f"({kw} ({' '.join(binds)}) {body})"

    if kind == "if":
        cmp = ctx.rng.choice(COMPARES)
        lhs = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        rhs = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        then = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        els = _gen_expr(ctx, depth - 1, locals_in_scope, no_let_star)
        ctx.feats.add("if_compare")
        return f"(if ({cmp} {lhs} {rhs}) {then} {els})"

    return _atom(ctx, locals_in_scope)


def _gen_loop(ctx: _Ctx) -> str:
    """Bounded loop/recur mirroring logistic_map_loop: counter k up to constant N, one scalar
    accumulator x updated by a depth-limited body expression each step, returns x.

      (loop ((x <init>) (k 0))
        (if (= k N) x (recur <body(x)> (+ k 1))))
    """
    N = ctx.rng.randint(ctx.cfg.loop_min, ctx.cfg.loop_max)
    # init references params; body references x (the local accumulator) and params
    init = _gen_expr(ctx, max(1, ctx.cfg.depth - 2), [])
    # keep the per-step update contractive-ish so the rollout stays finite in float32:
    # x_next = a*x*(1-x)-style or a guarded combination. Use a generated body but tame it.
    body = _gen_expr(ctx, max(1, ctx.cfg.depth - 1), ["x"])
    # tame: blend the generated body with x so magnitude does not blow up over N steps
    update = f"(* 0.5 (+ x {body}))"
    ctx.feats.add("loop_recur")
    return (f"(loop ((x {init}) (k 0)) "
            f"(if (= k {N}) x (recur {update} (+ k 1))))")


def gen_one(rng: random.Random, cfg: GenConfig, pid: int) -> dict:
    param_names = [f"p{i}" for i in range(cfg.n_params)]
    ctx = _Ctx(rng=rng, cfg=cfg, param_names=param_names)
    if rng.random() < cfg.p_loop:
        body = _gen_loop(ctx)
    else:
        body = _gen_expr(ctx, cfg.depth, [])

    # Guarantee the OUTPUT genuinely depends on a param leaf (not just references one in a dead
    # binding). A gradient gate needs a live differentiated path to the output, else NDVM .backward()
    # has no grad_fn and the oracle grad is 0 -- a generator artifact, not a runtime signal. So the
    # root always ADDS a guaranteed-live param term.
    live = param_names[0]
    ctx.used_params.add(live)
    ctx.feats.add("param")
    ctx.feats.add("binop")
    src = f"(+ {body} {live})"

    used = sorted(ctx.used_params)
    params = {nm: round(rng.uniform(cfg.param_lo, cfg.param_hi), 4) for nm in used}
    return {
        "id": pid,
        "src": src,
        "params": params,
        "features": sorted(ctx.feats),
    }


# ---------------------------------------------------------------------------
# Structural acceptance filter. The `no_let_star` threading inside _gen_expr is a best-effort
# preventer, but a few rare deep shapes still slip a `let*` into a `let*` binding RHS (an empirically
# confirmed cross-backend hazard: the DMCI compiler RAISES UnsupportedOperatorError on that exact shape
# while NDVM silently evaluates the inner let* to 0; verified on HPC with minimal probes). Rather than
# rely solely on the recursive ban, every generated program is parsed and HARD-REJECTED if it contains
# a `let*` anywhere inside a `let*` binding RHS, so the frozen corpus stays inside the surface BOTH
# runtimes agree on. This is what makes the gate pass-rates a clean signal instead of being polluted by
# generator-emitted unsupported shapes.
# ---------------------------------------------------------------------------
def _sexpr_parse(src: str):
    toks = src.replace("(", " ( ").replace(")", " ) ").split()

    def rd(t):
        x = t.pop(0)
        if x == "(":
            lst = []
            while t and t[0] != ")":
                lst.append(rd(t))
            if t:
                t.pop(0)  # ")"
            return lst
        return x

    return rd(toks)


def _contains_letstar(node) -> bool:
    if not isinstance(node, list):
        return False
    if node and node[0] == "let*":
        return True
    return any(_contains_letstar(c) for c in node)


def _has_letstar_in_letstar_rhs(node) -> bool:
    """True iff some `let*` node has a `let*` anywhere inside one of its binding RHS subtrees."""
    if not isinstance(node, list):
        return False
    if len(node) >= 2 and node[0] == "let*" and isinstance(node[1], list):
        for b in node[1]:
            if isinstance(b, list) and len(b) >= 2 and _contains_letstar(b[1]):
                return True
    return any(_has_letstar_in_letstar_rhs(c) for c in node)


def is_dmci_supported_shape(src: str) -> bool:
    """Reject the one cross-backend-disagreeing shape (let* inside a let* binding RHS)."""
    try:
        ast = _sexpr_parse(src)
    except Exception:  # noqa: BLE001
        return False
    return not _has_letstar_in_letstar_rhs(ast)


def gen_corpus(n: int = 200, seed: int = 1234, cfg: GenConfig | None = None) -> list:
    """Deterministic corpus of n well-typed, scalar-valued, param-dependent programs.

    Each candidate is hard-filtered through ``is_dmci_supported_shape`` so the frozen corpus contains
    only programs inside the surface both DMCI and NDVM agree on (no silently-0 / RAISE shapes).
    """
    cfg = cfg or GenConfig()
    rng = random.Random(seed)
    out = []
    pid = 0
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        # vary structural knobs per program for breadth of coverage
        c = GenConfig(
            n_params=rng.randint(2, 4),
            depth=rng.randint(2, 5),
            p_loop=cfg.p_loop,
            loop_min=cfg.loop_min,
            loop_max=cfg.loop_max,
            param_lo=cfg.param_lo,
            param_hi=cfg.param_hi,
            max_let_binds=cfg.max_let_binds,
        )
        rec = gen_one(rng, c, pid)
        if not is_dmci_supported_shape(rec["src"]):
            continue  # skip the rare let*-in-let*-RHS shape (regenerate); keeps ids dense
        rec["id"] = pid
        out.append(rec)
        pid += 1
    return out


if __name__ == "__main__":
    import json
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    corpus = gen_corpus(n, seed=1234)
    for r in corpus:
        print(json.dumps(r))
