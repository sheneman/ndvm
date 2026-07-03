############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# transform.py: Transform an AST into A-Normal Form. The key property: in ANF, all arguments to function applications and the...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Transform an AST into A-Normal Form.

The key property: in ANF, all arguments to function applications
and the test expression in conditionals are trivial (Const or Var).
Compound subexpressions are let-bound to fresh temporaries.
"""

from __future__ import annotations
from neural_compiler.parser.ast_nodes import (
    ASTNode,
    Begin,
    Const,
    Define,
    If,
    Lambda,
    Let,
    Letrec,
    Loop,
    Program,
    Quote,
    Recur,
    SoftChoice,
    App,
    Var,
)
from neural_compiler.anf.anf_nodes import (
    ANFNode,
    ANFBegin,
    ANFConst,
    ANFSoftChoice,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
    ANFLetrec,
    ANFLoop,
    ANFQuote,
    ANFRecur,
)


class _NameGen:
    """Generate unique temporary variable names."""

    def __init__(self) -> None:
        self._counter = 0

    def fresh(self, prefix: str = "t") -> str:
        name = f"__{prefix}{self._counter}"
        self._counter += 1
        return name


def _is_trivial(node: ANFNode) -> bool:
    return isinstance(node, (ANFConst, ANFVar))


def _normalize(node: ASTNode, gen: _NameGen) -> ANFNode:
    """Convert an AST node to ANF."""
    if isinstance(node, Const):
        return ANFConst(node.value)

    if isinstance(node, Var):
        return ANFVar(node.name)

    if isinstance(node, Lambda):
        return ANFLambda(
            params=node.params,
            body=_normalize(node.body, gen),
        )

    if isinstance(node, Let):
        result = _normalize(node.body, gen)
        for name, expr in reversed(node.bindings):
            rhs = _normalize(expr, gen)
            result = ANFLet(name=name, rhs=rhs, body=result)
        return result

    if isinstance(node, If):
        test_anf = _normalize(node.test, gen)
        then_anf = _normalize(node.then_, gen)
        else_anf = _normalize(node.else_, gen)

        if _is_trivial(test_anf):
            return ANFIf(test=test_anf, then_=then_anf, else_=else_anf)

        name = gen.fresh("cond")
        return ANFLet(
            name=name,
            rhs=test_anf,
            body=ANFIf(test=ANFVar(name), then_=then_anf, else_=else_anf),
        )

    if isinstance(node, App):
        func_anf = _normalize(node.func, gen)
        args_anf = [_normalize(a, gen) for a in node.args]

        bindings: list[tuple[str, ANFNode]] = []

        if not _is_trivial(func_anf):
            fname = gen.fresh("fn")
            bindings.append((fname, func_anf))
            func_anf = ANFVar(fname)

        trivial_args: list[ANFNode] = []
        for arg in args_anf:
            if _is_trivial(arg):
                trivial_args.append(arg)
            else:
                aname = gen.fresh("arg")
                bindings.append((aname, arg))
                trivial_args.append(ANFVar(aname))

        result: ANFNode = ANFApp(func=func_anf, args=tuple(trivial_args))

        for name, rhs in reversed(bindings):
            result = ANFLet(name=name, rhs=rhs, body=result)

        return result

    if isinstance(node, Loop):
        init_anf = [_normalize(expr, gen) for _, expr in node.bindings]
        params = tuple(name for name, _ in node.bindings)

        bindings: list[tuple[str, ANFNode]] = []
        trivial_inits: list[ANFNode] = []
        for init in init_anf:
            if _is_trivial(init):
                trivial_inits.append(init)
            else:
                iname = gen.fresh("init")
                bindings.append((iname, init))
                trivial_inits.append(ANFVar(iname))

        body_anf = _normalize(node.body, gen)
        result: ANFNode = ANFLoop(
            params=params, inits=tuple(trivial_inits), body=body_anf
        )

        for name, rhs in reversed(bindings):
            result = ANFLet(name=name, rhs=rhs, body=result)
        return result

    if isinstance(node, Letrec):
        norm_bindings = []
        for name, expr in node.bindings:
            rhs = _normalize(expr, gen)
            norm_bindings.append((name, rhs))
        body_anf = _normalize(node.body, gen)
        return ANFLetrec(bindings=tuple(norm_bindings), body=body_anf)

    if isinstance(node, Recur):
        args_anf = [_normalize(a, gen) for a in node.args]

        bindings_r: list[tuple[str, ANFNode]] = []
        trivial_args: list[ANFNode] = []
        for arg in args_anf:
            if _is_trivial(arg):
                trivial_args.append(arg)
            else:
                aname = gen.fresh("rec")
                bindings_r.append((aname, arg))
                trivial_args.append(ANFVar(aname))

        result_r: ANFNode = ANFRecur(args=tuple(trivial_args))

        for name, rhs in reversed(bindings_r):
            result_r = ANFLet(name=name, rhs=rhs, body=result_r)
        return result_r

    if isinstance(node, SoftChoice):
        options_anf = tuple(_normalize(o, gen) for o in node.options)
        weights_anf = _normalize(node.weights, gen)
        bindings: list[tuple[str, ANFNode]] = []
        if not _is_trivial(weights_anf):
            wname = gen.fresh("w")
            bindings.append((wname, weights_anf))
            weights_anf = ANFVar(wname)
        result: ANFNode = ANFSoftChoice(options=options_anf, weights=weights_anf)
        for name, rhs in reversed(bindings):
            result = ANFLet(name=name, rhs=rhs, body=result)
        return result

    if isinstance(node, Quote):
        return ANFQuote(datum=node.datum)

    if isinstance(node, Begin):
        if len(node.exprs) == 1:
            return _normalize(node.exprs[0], gen)
        result = _normalize(node.exprs[-1], gen)
        for expr in reversed(node.exprs[:-1]):
            name = gen.fresh("seq")
            result = ANFLet(name=name, rhs=_normalize(expr, gen), body=result)
        return result

    if isinstance(node, Define):
        raise TypeError("Define must be inside a Program (use parse_program)")

    if isinstance(node, Program):
        if not node.forms:
            raise TypeError("Empty program")
        defines = [f for f in node.forms if isinstance(f, Define)]
        exprs = [f for f in node.forms if not isinstance(f, Define)]
        if not exprs:
            raise TypeError("Program must end with an expression")
        body: ASTNode
        if len(exprs) == 1:
            body = exprs[0]
        else:
            body = Begin(exprs=tuple(exprs))
        lambda_defs = [(d.name, d.value) for d in defines if isinstance(d.value, Lambda)]
        value_defs = [(d.name, d.value) for d in defines if not isinstance(d.value, Lambda)]
        if lambda_defs:
            body = Letrec(bindings=tuple(lambda_defs), body=body)
        for name, value in reversed(value_defs):
            body = Let(bindings=((name, value),), body=body)
        return _normalize(body, gen)

    raise TypeError(f"Unknown AST node type: {type(node)}")


def to_anf(node: ASTNode) -> ANFNode:
    """Convert a parsed AST to A-Normal Form."""
    return _normalize(node, _NameGen())
