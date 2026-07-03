############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# dmci.py: Compile a program via Differentiable Meta-Circular Interpretation (DMCI). In direct compilation, a Scheme...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Compile a program via Differentiable Meta-Circular Interpretation (DMCI).

In direct compilation, a Scheme program is compiled straight to a differentiable graph. In
DMCI, the *self-hosted Scheme evaluator* (``bootstrap/compiler.scm``) is compiled instead, and
your program is handed to it as quoted **data** -- so gradients flow from the loss, through the
compiled interpreter, to the program's numeric constants, and any new program runs through the
same compiled evaluator without recompilation. This is the method behind Paper 2.

``compile_dmci(source, ...)`` returns the ComputeGraph of ``<evaluator> + (scheme-eval-program
'<your program> <env>)``; it evaluates to the same value as direct compilation (Theorem 1) but
exercises the meta-circular path.
"""
from __future__ import annotations

from pathlib import Path

from neural_compiler.compiler import compile_program
from neural_compiler.graph.builder import ComputeGraph
# Re-export the tensor-payload input helpers so callers can write
#   evaluate(compile_dmci(src), {"obs": as_matrix(obs_tensor)})
from neural_compiler.runtime.tagged_value import TensorInput, as_vector, as_matrix  # noqa: F401

_BOOTSTRAP = Path(__file__).resolve().parent.parent / "bootstrap" / "compiler.scm"


class UnsupportedOperatorError(ValueError):
    """A program handed to the DMCI interpreter uses an operator or special form the
    meta-circular evaluator (``bootstrap/compiler.scm``) does not implement.

    Without this static check such an operator would hit eval-apply's ``(#t 0)``
    fallthrough and *silently evaluate to 0*, corrupting results or causing
    non-termination (a recursion guard that never fires).
    """


# Operators dispatched by ``eval-apply`` in ``bootstrap/compiler.scm``. This is the
# source of truth for "what a program run THROUGH the interpreter may call"; it is a
# subset of the directly-compiled primitive set. KEEP IN SYNC with eval-apply's cond
# clauses -- ``tests/integration/test_self_hosting.py`` asserts the two match.
INTERPRETER_OPS = {
    "+", "-", "*", "/",
    "=", "<", ">", "<=", ">=",
    "cons", "car", "cdr",
    "null?", "pair?", "number?", "boolean?", "symbol?",
    "eq?", "not", "list",
    "sin", "cos", "exp", "sqrt", "log", "abs", "pow",
    "min", "max", "modulo", "remainder",
    # Strategy B tensor-payload vector/matrix ops (native in eval-apply, dispatched to
    # the vectorized torch primitives via tagged_ops.VEC_OPS).
    "vec", "mat", "ref", "dot", "cross", "norm", "normalize", "vsum", "vlen",
    "scale", "matvec", "matmul", "transpose", "trace", "det", "logdet", "inv", "outer",
    "eye", "zeros", "ones",
}

# Special forms recognized by ``scheme-eval`` (compiler.scm). ``define`` is handled at
# program top level by ``scheme-eval-program``; ``else`` is cond-clause syntax.
INTERPRETER_SPECIAL_FORMS = {
    "quote", "if", "cond", "let", "letrec", "lambda", "begin", "define", "else",
}


def _is_number_token(tok: object) -> bool:
    if not isinstance(tok, str):
        return False
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _scan_form(node: object, bound: set, heads: set) -> None:
    """Walk one program datum (nested lists of token strings), collecting locally bound
    names and the symbols that appear in head/operator position. Quoted data is skipped
    -- it is data, not code -- so symbols inside ``(quote ...)`` are never flagged."""
    if not isinstance(node, list) or not node:
        return
    head = node[0]
    if head == "quote":
        return
    if head == "lambda":
        params = node[1] if len(node) > 1 else []
        if isinstance(params, list):
            bound.update(p for p in params if isinstance(p, str))
        for sub in node[2:]:
            _scan_form(sub, bound, heads)
        return
    if head in ("let", "letrec"):
        binds = node[1] if len(node) > 1 else []
        if isinstance(binds, list):
            for b in binds:
                if isinstance(b, list) and b and isinstance(b[0], str):
                    bound.add(b[0])
                    for v in b[1:]:
                        _scan_form(v, bound, heads)
        for sub in node[2:]:
            _scan_form(sub, bound, heads)
        return
    if head == "define":
        target = node[1] if len(node) > 1 else None
        if isinstance(target, list):  # (define (f a b) body) -- f and params are bound
            bound.update(s for s in target if isinstance(s, str))
        elif isinstance(target, str):  # (define name val)
            bound.add(target)
        for sub in node[2:]:
            _scan_form(sub, bound, heads)
        return
    # generic application (and if/cond/begin, whose heads are allowed special forms)
    if isinstance(head, str):
        heads.add(head)
    else:
        _scan_form(head, bound, heads)  # e.g. ((lambda (x) ...) arg)
    for sub in node[1:]:
        _scan_form(sub, bound, heads)


def unsupported_interpreter_ops(program) -> set:
    """Return the set of head-position symbols in ``program`` that the DMCI interpreter
    does not implement (and would therefore silently evaluate to 0). Empty == safe."""
    bound: set = set()
    heads: set = set()
    for form in program_datum(program):  # program_datum macro-expands first
        _scan_form(form, bound, heads)
    known = INTERPRETER_OPS | INTERPRETER_SPECIAL_FORMS
    return {h for h in heads
            if isinstance(h, str) and h not in known and h not in bound
            and not _is_number_token(h)}


def check_interpreter_supported(program) -> None:
    """Raise :class:`UnsupportedOperatorError` if ``program`` uses operators or special
    forms the DMCI interpreter lacks (which would otherwise silently return 0)."""
    unknown = unsupported_interpreter_ops(program)
    if not unknown:
        return
    from neural_compiler.parser.ast_nodes import PRIMITIVES
    direct_only = sorted(o for o in unknown if o in PRIMITIVES)
    hint = ""
    if direct_only:
        hint = (f" {direct_only} are supported by direct compilation but not the "
                f"meta-circular interpreter -- add an eval-apply clause in "
                f"bootstrap/compiler.scm (and to INTERPRETER_OPS) to use them under DMCI.")
    raise UnsupportedOperatorError(
        f"program uses operator(s)/form(s) the DMCI interpreter does not implement: "
        f"{sorted(unknown)}. They would silently evaluate to 0.{hint}")


def evaluator_source() -> str:
    """The self-hosted Scheme evaluator source compiled by DMCI."""
    return _BOOTSTRAP.read_text()


def split_top_level_forms(source: str) -> list[str]:
    """Split Scheme source into its top-level S-expressions (skips ``;`` line comments)."""
    forms: list[str] = []
    depth = 0
    cur: list[str] = []
    i, n = 0, len(source)
    while i < n:
        c = source[i]
        if c == ";":  # line comment to end of line
            while i < n and source[i] != "\n":
                i += 1
            continue
        cur.append(c)
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                form = "".join(cur).strip()
                if form:
                    forms.append(form)
                cur = []
        i += 1
    tail = "".join(cur).strip()
    if tail:
        forms.append(tail)
    return forms


def _make_env(names: list[str]) -> str:
    """Build the Scheme environment association list: ``(list (cons 'name name) ...)``."""
    pairs = " ".join(f"(cons '{name} {name})" for name in names)
    return f"(list {pairs})"


def dmci_wrap(source: str, names: list[str]) -> str:
    """Wrap a program as quoted data evaluated by the compiled self-hosted interpreter.

    ``names`` are the program's free variables (its inputs / learnable constants); they are
    bound in the runtime environment and become the inputs of the compiled DMCI graph.

    The program is run through ``program_datum`` (macro expansion to core forms), so this
    string path matches the datum path used by ``evaluate_program``/
    ``check_interpreter_supported`` -- one source of truth.
    """
    forms = program_datum(source)
    quoted = "\n    ".join("'" + _datum_to_scheme(f) for f in forms)
    return (
        evaluator_source()
        + "\n(scheme-eval-program\n  (list\n    "
        + quoted
        + ")\n  "
        + _make_env(names)
        + ")\n"
    )


def _detect_free_vars(source: str, prelude: bool = False) -> list[str]:
    """Compile ``source`` directly, discovering its free variables as inputs (the compiler
    reports each undefined variable). Returns the input names in declaration order."""
    inputs: dict[str, None] = {}
    seen: set[str] = set()
    while True:
        try:
            g = compile_program(source, inputs=inputs or None, prelude=prelude)
            return list(g.input_names)
        except KeyError as e:
            msg = str(e.args[0]) if e.args else str(e)
            if "Undefined variable:" not in msg:
                raise
            name = msg.split("Undefined variable:")[-1].strip().strip("'\"")
            if not name or name in seen:
                raise
            seen.add(name)
            inputs[name] = None


def compile_dmci(
    source: str,
    input_names: list[str] | None = None,
    prelude: bool = False,
) -> ComputeGraph:
    """Compile ``source`` via the meta-circular interpreter (program-as-data).

    ``input_names`` are the program's free variables; if omitted they are auto-detected.
    The returned graph is tagged (heap-backed) and evaluates the program through the compiled
    evaluator. The tagged evaluator trampolines tail calls -- including the interpreter's own
    ``scheme-eval``/``eval-apply`` loop -- so tail-recursive programs run in constant Python
    stack; only structural (non-tail) recursion consumes stack proportional to the program's
    genuine recursion depth.

    Raises :class:`UnsupportedOperatorError` if the program uses an operator or special form
    the interpreter does not implement (which would otherwise silently evaluate to 0).
    """
    check_interpreter_supported(source)
    if input_names is None:
        input_names = _detect_free_vars(source, prelude=prelude)
    wrapped = dmci_wrap(source, input_names)
    return compile_program(wrapped, inputs={n: None for n in input_names} or None, prelude=prelude)


# ---------------------------------------------------------------------------
# Bare, distributable interpreter (program supplied as runtime data)
# ---------------------------------------------------------------------------
# ``compile_dmci`` bakes ONE program into the graph (as a ``quote_const``), so each
# program yields its own ``.ncg``. ``compile_interpreter`` instead compiles the evaluator
# ONCE with the program and environment as runtime INPUTS: the result is a single,
# backend-agnostic differentiable artifact that runs ANY program handed to it as data --
# no per-program recompilation and no Scheme toolchain on the consumer's side. Compile it
# once, ``neural_compiler.save_compiled`` it to a portable ``.ncg``, distribute it, then run
# arbitrary programs through it with :func:`evaluate_program`.


def compile_interpreter(prelude: bool = True) -> ComputeGraph:
    """Compile the bare meta-circular interpreter as a reusable, distributable artifact.

    Returns the ComputeGraph of ``<evaluator> + (scheme-eval-program program env)`` in which
    ``program`` (a list of quoted forms) and ``env`` (a binding association list) are runtime
    INPUTS rather than embedded constants. Serialize once with ``neural_compiler.save_compiled``
    and run any program through the loaded graph with :func:`evaluate_program`.
    """
    src = evaluator_source() + "\n(scheme-eval-program program env)\n"
    return compile_program(src, inputs={"program": None, "env": None}, prelude=prelude)


def _read_datum(form: str):
    """Parse one S-expression string into the nested-list datum that materialize_quote
    consumes -- the same representation the compiler bakes into a ``quote_const`` node."""
    from neural_compiler.parser.scheme_parser import tokenize, _parse_sexpr
    datum, _ = _parse_sexpr(tokenize(form), 0)
    return datum


# ---------------------------------------------------------------------------
# Macro expansion: lower sugar to the core forms the interpreter implements,
# BEFORE the program reaches the interpreter. This keeps bootstrap/compiler.scm
# minimal and keeps INTERPRETER_OPS / the prescan unchanged (expansion runs
# first, so the interpreter only ever sees core forms). Mirrors the direct
# compiler's parser desugaring (scheme_parser.py) so DMCI matches its language.
# ---------------------------------------------------------------------------

_gensym_counter = [0]


def _gensym(prefix: str) -> str:
    _gensym_counter[0] += 1
    return f"{prefix}{_gensym_counter[0]}"


def _begin_wrap(forms: list):
    """A body (list of forms) as a SINGLE expression. The interpreter's
    let/letrec/lambda/cond bodies take one expr, so multi-expr bodies must be
    begin-wrapped or all but the last form are silently dropped."""
    forms = list(forms)
    if len(forms) == 1:
        return forms[0]
    return ["begin"] + forms


def _expand_let_star(binds, body):
    """(let* (b1 b2 ...) body...) -> nested single-binding lets."""
    if len(binds) <= 1:
        return ["let", list(binds), _begin_wrap([expand_macros(x) for x in body])]
    return ["let", [_expand_binding(binds[0])], _expand_let_star(binds[1:], body)]


def _expand_binding(b):
    if isinstance(b, list) and len(b) >= 2:
        return [b[0], expand_macros(b[1])]
    return b


def _rewrite_recur(expr, selfsym, arity, tail):
    """Replace a tail-position (recur a...) with the self-application (selfsym selfsym a...).
    `loop` lowers to a self-passing CLOSURE (see _expand_loop): the loop function is reached by
    re-applying `selfsym` (the closure passed to itself) rather than a letrec'd defined-fn. The
    closure uses its FIXED captured env, so each iteration re-binds in a constant-size env --
    O(1) env-lookup, O(T) total -- avoiding the defined-fn path's growing call-site env (O(T^2),
    compiler.scm:238-241). Raise on a non-tail recur or arity mismatch (a non-tail recur would
    silently degrade from constant-stack trampolining to depth-bounded recursion)."""
    if not isinstance(expr, list) or not expr:
        return expr
    head = expr[0]
    if head == "quote":
        return expr
    if head == "recur":
        if not tail:
            raise UnsupportedOperatorError(
                "recur must be in tail position inside a loop (non-tail recur unsupported)")
        args = expr[1:]
        if len(args) != arity:
            raise UnsupportedOperatorError(
                f"recur takes {arity} arg(s) to match the loop variables, got {len(args)}")
        return [selfsym, selfsym] + [_rewrite_recur(a, selfsym, arity, False) for a in args]
    if head == "if":
        out = ["if", _rewrite_recur(expr[1], selfsym, arity, False)]
        if len(expr) > 2:
            out.append(_rewrite_recur(expr[2], selfsym, arity, tail))
        if len(expr) > 3:
            out.append(_rewrite_recur(expr[3], selfsym, arity, tail))
        return out
    if head == "begin":
        forms = expr[1:]
        return ["begin"] + [_rewrite_recur(f, selfsym, arity, tail and i == len(forms) - 1)
                            for i, f in enumerate(forms)]
    if head == "cond":
        out = ["cond"]
        for clause in expr[1:]:
            if isinstance(clause, list) and clause:
                test = clause[0]
                test2 = test if test == "else" else _rewrite_recur(test, selfsym, arity, False)
                rest = clause[1:]
                out.append([test2] + [_rewrite_recur(c, selfsym, arity, tail and j == len(rest) - 1)
                                      for j, c in enumerate(rest)])
            else:
                out.append(clause)
        return out
    if head in ("let", "letrec"):
        binds = expr[1] if len(expr) > 1 else []
        new_binds = [[b[0], _rewrite_recur(b[1], selfsym, arity, False)]
                     if isinstance(b, list) and len(b) >= 2 else b for b in binds]
        out = [head, new_binds]
        if len(expr) > 2:
            out.append(_rewrite_recur(expr[2], selfsym, arity, tail))
        return out
    if head == "lambda":
        # recur does not cross into a nested lambda; any recur inside is non-tail -> error
        out = ["lambda", expr[1] if len(expr) > 1 else []]
        if len(expr) > 2:
            out.append(_rewrite_recur(expr[2], selfsym, arity, False))
        return out
    # generic application: operator and operands are all non-tail
    return [_rewrite_recur(x, selfsym, arity, False) for x in expr]


def _expand_loop(node):
    """(loop ((v init)...) body...) ->
       (let ((f (lambda (self v...) body[recur->(self self ...)]))) (f f init...)).
    Self-passing CLOSURE form (NOT letrec/defined-fn): a closure captures its definition env
    (compiler.scm:233-237) and `self` is passed explicitly, so each iteration re-binds the loop
    vars in a FIXED-size env -- O(1) env-lookup, O(T) total. The earlier letrec/defined-fn form
    re-bound in the GROWING call-site env (compiler.scm:238-241), making a T-step loop O(T^2)."""
    binds = node[1] if len(node) > 1 else []
    names = [b[0] for b in binds]
    inits = [expand_macros(b[1]) for b in binds]
    fname = _gensym("__loop_")
    selfsym = _gensym("__self_")
    body_expr = _begin_wrap([expand_macros(x) for x in node[2:]])  # expand sugar first
    rewritten = _rewrite_recur(body_expr, selfsym, len(names), True)
    lam = ["lambda", [selfsym] + names, rewritten]
    return ["let", [[fname, lam]], [fname, fname] + inits]


def expand_macros(node):
    """Recursively lower sugar (let*/when/unless/loop/recur, vec/mat, multi-expr bodies)
    to core forms. Quoted data is left untouched."""
    if not isinstance(node, list) or not node:
        return node
    head = node[0]
    if head == "quote":
        return node
    if head == "let*":
        return _expand_let_star(node[1] if len(node) > 1 else [], node[2:])
    if head == "when":
        return ["if", expand_macros(node[1]), _begin_wrap([expand_macros(x) for x in node[2:]]), "#f"]
    if head == "unless":
        return ["if", expand_macros(node[1]), "#f", _begin_wrap([expand_macros(x) for x in node[2:]])]
    if head == "loop":
        return _expand_loop(node)
    if head in ("vec", "mat"):
        # Strategy B: vec/mat are native tensor-payload constructors. Lower the variadic
        # surface to a fixed-arity call taking the element list -- (vec a b c) -> (vec (list a b c))
        # -- so the eval-apply clause is fixed-arity and the VEC_OPS handler stacks the elements.
        return [head, ["list"] + [expand_macros(x) for x in node[1:]]]
    if head in ("let", "letrec"):
        binds = node[1] if len(node) > 1 else []
        return [head, [_expand_binding(b) for b in binds],
                _begin_wrap([expand_macros(x) for x in node[2:]])]
    if head == "lambda":
        return ["lambda", node[1] if len(node) > 1 else [],
                _begin_wrap([expand_macros(x) for x in node[2:]])]
    if head == "cond":
        out = ["cond"]
        for clause in node[1:]:
            if isinstance(clause, list) and clause:
                test = clause[0]
                test2 = test if test == "else" else expand_macros(test)
                rest = [expand_macros(c) for c in clause[1:]]
                out.append([test2, _begin_wrap(rest)] if rest else [test2])
            else:
                out.append(clause)
        return out
    if head == "define":
        target = node[1] if len(node) > 1 else None
        if isinstance(target, list):  # (define (f a..) body...)
            return ["define", target, _begin_wrap([expand_macros(x) for x in node[2:]])]
        return ["define", target] + [expand_macros(x) for x in node[2:]]
    return [expand_macros(x) for x in node]


def _datum_to_scheme(d) -> str:
    """Serialize a nested-list datum back to Scheme source (inverse of _read_datum)."""
    if isinstance(d, list):
        return "(" + " ".join(_datum_to_scheme(x) for x in d) + ")"
    return str(d)


def program_datum(program) -> list:
    """Normalize ``program`` (source string, or pre-parsed list of form-data) into the list
    of macro-expanded top-level form data that ``scheme-eval-program`` expects."""
    if isinstance(program, str):
        forms = [_read_datum(f) for f in split_top_level_forms(program)]
    elif isinstance(program, list):
        forms = program
    else:
        forms = [program]
    return [expand_macros(f) for f in forms]


def _shared_symtab():
    """The process-wide symbol table used by ``materialize_quote``. Interning the runtime
    program's and environment's symbols through it guarantees they share symbol ids with the
    interpreter's own quoted symbols (so ``eq?`` dispatch on operators/keywords works)."""
    from neural_compiler.ops.tagged_ops import materialize_quote
    from neural_compiler.runtime.symbols import SymbolTable
    if not hasattr(materialize_quote, "_symtab"):
        materialize_quote._symtab = SymbolTable()
    return materialize_quote._symtab


def evaluate_program(
    interp_graph: ComputeGraph,
    program,
    bindings: dict | None = None,
    *,
    max_iter: int = 10000,
    max_depth: int = 10000,
):
    """Run ``program`` through a pre-compiled bare interpreter (:func:`compile_interpreter`).

    Args:
        interp_graph: a graph from :func:`compile_interpreter` (or one loaded from ``.ncg``).
        program: Scheme source (one or more top-level forms) or a pre-parsed list of form-data.
        bindings: maps each free variable to its value -- a Python number, a scalar tensor
            (e.g. a learnable parameter; gradients flow back to it), or a batched ``[N]`` tensor
            (the whole batch is evaluated in one interpreter walk). Already-tagged
            ``(*, VALUE_DIM)`` tensors are passed through unchanged.

    The program AST and the environment association list are materialized onto a fresh heap
    and bound to the interpreter's ``program``/``env`` inputs. Returns the raw TaggedValue
    tensor (read it with ``neural_compiler.runtime.tagged_value.unwrap_number``). No
    recompilation occurs -- the same compiled interpreter evaluates every program.
    """
    import torch
    from neural_compiler.evaluator.engine import _evaluate_tagged
    from neural_compiler.runtime.heap import TensorHeap
    from neural_compiler.ops.tagged_ops import materialize_quote
    from neural_compiler.runtime.tagged_value import (
        make_symbol, make_float, make_vector, VALUE_DIM, TensorInput)

    check_interpreter_supported(program)
    bindings = bindings or {}
    heap = TensorHeap()
    symtab = _shared_symtab()

    prog_root = materialize_quote(program_datum(program), heap)

    entries = []
    for name, val in bindings.items():
        if isinstance(val, TensorInput):  # bind a raw tensor as a VECTOR/MATRIX payload
            tagged_val = make_vector(float(heap.store(val.tensor)), float(val.feature_ndim),
                                     device=heap.device)
        elif isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[-1] == VALUE_DIM:
            tagged_val = val
        elif isinstance(val, torch.Tensor):
            tagged_val = make_float(val)
        else:
            tagged_val = make_float(torch.tensor(float(val)))
        entries.append(heap.cons(make_symbol(symtab.intern(name)), tagged_val))
    env_root = heap.build_list(entries)

    return _evaluate_tagged(
        interp_graph,
        {"program": prog_root, "env": env_root},
        max_iter,
        max_depth,
        heap=heap,
    )
