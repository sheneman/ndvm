############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# tco.py: Tail-call optimization: convert self-tail-recursive letrec to loop/recur. When a letrec binds a single function...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tail-call optimization: convert self-tail-recursive letrec to loop/recur.

When a letrec binds a single function whose only self-calls are in tail
position, the entire construct can be replaced with a loop/recur — giving
O(1) stack instead of O(depth).

Two cases:
  1. Letrec body is a direct call (f args...) → eliminate letrec, produce ANFLoop
  2. Letrec body uses f indirectly → keep letrec but replace lambda body with
     an internal loop so the function no longer recurses
"""

from __future__ import annotations
from neural_compiler.anf.anf_nodes import (
    ANFNode,
    ANFConst,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
    ANFLetrec,
    ANFLoop,
    ANFRecur,
)
from neural_compiler.parser.ast_nodes import PRIMITIVES


def optimize_tco(node: ANFNode) -> ANFNode:
    """Walk the ANF tree and optimize eligible letrec nodes to loop/recur."""
    if isinstance(node, ANFLetrec):
        result = _try_tco_letrec(node)
        return _walk(result)
    return _walk(node)


def _walk(node: ANFNode) -> ANFNode:
    """Recursively apply TCO to all sub-expressions."""
    if isinstance(node, (ANFConst, ANFVar)):
        return node

    if isinstance(node, ANFLet):
        return ANFLet(
            name=node.name,
            rhs=_walk(node.rhs),
            body=_walk(node.body),
        )

    if isinstance(node, ANFIf):
        return ANFIf(
            test=node.test,
            then_=_walk(node.then_),
            else_=_walk(node.else_),
        )

    if isinstance(node, ANFApp):
        return node

    if isinstance(node, ANFLambda):
        return ANFLambda(params=node.params, body=_walk(node.body))

    if isinstance(node, ANFLetrec):
        result = _try_tco_letrec(node)
        if not isinstance(result, ANFLetrec):
            return _walk(result)
        new_bindings = tuple(
            (name, ANFLambda(params=lam.params, body=_walk(lam.body)))
            for name, lam in result.bindings
        )
        return ANFLetrec(bindings=new_bindings, body=_walk(result.body))

    if isinstance(node, ANFLoop):
        return ANFLoop(
            params=node.params,
            inits=node.inits,
            body=_walk(node.body),
        )

    if isinstance(node, ANFRecur):
        return node

    return node


def _try_tco_letrec(node: ANFLetrec) -> ANFNode:
    """Try to optimize a letrec. Returns the original node if not eligible."""
    if len(node.bindings) == 1:
        return _try_tco_single(node)
    result = _try_tco_mutual(node)
    if not isinstance(result, ANFLetrec):
        return result
    return _try_tco_per_binding(result)


def _try_tco_single(node: ANFLetrec) -> ANFNode:
    """TCO for single-binding letrec (self-recursive tail calls)."""
    name, lam = node.bindings[0]

    ok, transformed_body = _replace_tail_calls(name, lam.body, in_tail=True)
    if not ok:
        return node

    if _is_direct_call(name, node.body):
        inits = node.body.args
        return ANFLoop(params=lam.params, inits=inits, body=transformed_body)

    loop_body = ANFLoop(
        params=lam.params,
        inits=tuple(ANFVar(p) for p in lam.params),
        body=transformed_body,
    )
    new_lam = ANFLambda(params=lam.params, body=loop_body)
    return ANFLetrec(
        bindings=((name, new_lam),),
        body=_inline_calls(name, lam.params, node.body),
    )


def _try_tco_mutual(node: ANFLetrec) -> ANFNode:
    """TCO for multi-binding letrec (mutual tail recursion → dispatch loop).

    All mutual/self calls must be in tail position. Produces a single
    loop with a dispatch tag selecting which function body to execute.
    """
    names = [name for name, _ in node.bindings]
    name_set = frozenset(names)

    for name, lam in node.bindings:
        ok = _check_mutual_tail_only(name_set, lam.body, in_tail=True)
        if not ok:
            return node

    max_arity = max(len(lam.params) for _, lam in node.bindings)
    tag_param = "__mtco_tag"
    unified_params = tuple(f"__mtco_p{i}" for i in range(max_arity))
    all_loop_params = (tag_param,) + unified_params

    transformed_bodies = []
    for idx, (name, lam) in enumerate(node.bindings):
        body = _rename_params(lam.body, lam.params, unified_params[:len(lam.params)])
        body = _mutual_calls_to_recur(body, node.bindings, unified_params, max_arity)
        transformed_bodies.append(body)

    dispatch = transformed_bodies[-1]
    for i in range(len(transformed_bodies) - 2, -1, -1):
        test_name = f"__mtco_eq_{i}"
        dispatch = ANFLet(
            name=test_name,
            rhs=ANFApp(ANFVar("="), (ANFVar(tag_param), ANFConst(float(i)))),
            body=ANFIf(test=ANFVar(test_name), then_=transformed_bodies[i], else_=dispatch),
        )

    for idx, name in enumerate(names):
        if _is_direct_call(name, node.body):
            call_args = node.body.args
            padded = call_args + tuple(ANFConst(0.0) for _ in range(max_arity - len(call_args)))
            inits = (ANFConst(float(idx)),) + padded
            return ANFLoop(params=all_loop_params, inits=inits, body=dispatch)

    return node


def _try_tco_per_binding(node: ANFLetrec) -> ANFLetrec:
    """Apply self-TCO to individual bindings within a multi-binding letrec.

    When mutual TCO fails (some bindings have non-tail cross-calls),
    each binding that is only self-tail-recursive still gets its body
    wrapped in a loop/recur.
    """
    new_bindings = []
    changed = False
    for name, lam in node.bindings:
        ok, transformed_body = _replace_tail_calls(name, lam.body, in_tail=True)
        if ok and _contains_call_to(name, lam.body):
            loop_body = ANFLoop(
                params=lam.params,
                inits=tuple(ANFVar(p) for p in lam.params),
                body=transformed_body,
            )
            new_bindings.append((name, ANFLambda(params=lam.params, body=loop_body)))
            changed = True
        else:
            new_bindings.append((name, lam))
    if not changed:
        return node
    return ANFLetrec(bindings=tuple(new_bindings), body=node.body)


def _replace_tail_calls(
    name: str, node: ANFNode, in_tail: bool
) -> tuple[bool, ANFNode]:
    """Replace tail self-calls with ANFRecur. Returns (success, new_node).

    Fails if any self-call is in a non-tail position.
    """
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name == name:
            if not in_tail:
                return False, node
            return True, ANFRecur(args=node.args)
        return True, node

    if isinstance(node, (ANFConst, ANFVar)):
        return True, node

    if isinstance(node, ANFIf):
        if _contains_call_to(name, node.test):
            return False, node
        ok_then, new_then = _replace_tail_calls(name, node.then_, in_tail)
        if not ok_then:
            return False, node
        ok_else, new_else = _replace_tail_calls(name, node.else_, in_tail)
        if not ok_else:
            return False, node
        return True, ANFIf(test=node.test, then_=new_then, else_=new_else)

    if isinstance(node, ANFLet):
        if _contains_call_to(name, node.rhs):
            return False, node
        ok_body, new_body = _replace_tail_calls(name, node.body, in_tail)
        if not ok_body:
            return False, node
        return True, ANFLet(name=node.name, rhs=node.rhs, body=new_body)

    if isinstance(node, ANFRecur):
        return True, node

    return True, node


def _contains_call_to(name: str, node: ANFNode) -> bool:
    """Check if node or any descendant contains a call to `name`."""
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name == name:
            return True
        return False

    if isinstance(node, (ANFConst, ANFVar)):
        return False

    if isinstance(node, ANFIf):
        return (
            _contains_call_to(name, node.test)
            or _contains_call_to(name, node.then_)
            or _contains_call_to(name, node.else_)
        )

    if isinstance(node, ANFLet):
        return _contains_call_to(name, node.rhs) or _contains_call_to(name, node.body)

    if isinstance(node, ANFLambda):
        return _contains_call_to(name, node.body)

    return False


def _is_direct_call(name: str, node: ANFNode) -> bool:
    """Check if node is a direct application of `name`."""
    return (
        isinstance(node, ANFApp)
        and isinstance(node.func, ANFVar)
        and node.func.name == name
    )


def _inline_calls(
    name: str, params: tuple[str, ...], node: ANFNode
) -> ANFNode:
    """Replace calls to `name` in the letrec body with inline ANFLoop invocations.

    For case 2 (indirect use), each call site (f args...) becomes
    (loop ((params args)) body) — but since the lambda body already
    contains the loop, the call just invokes the non-recursive lambda.
    No transformation needed here — the letrec binding is kept and the
    lambda's body handles the looping.
    """
    return node


def _check_mutual_tail_only(
    names: frozenset[str], node: ANFNode, in_tail: bool
) -> bool:
    """Check that all calls to any name in `names` are in tail position."""
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name in names:
            return in_tail
        return True

    if isinstance(node, (ANFConst, ANFVar)):
        return True

    if isinstance(node, ANFIf):
        if _contains_any_call(names, node.test):
            return False
        return (_check_mutual_tail_only(names, node.then_, in_tail) and
                _check_mutual_tail_only(names, node.else_, in_tail))

    if isinstance(node, ANFLet):
        if _contains_any_call(names, node.rhs):
            return False
        return _check_mutual_tail_only(names, node.body, in_tail)

    if isinstance(node, ANFRecur):
        return True

    return True


def _contains_any_call(names: frozenset[str], node: ANFNode) -> bool:
    """Check if node contains a call to any name in `names`."""
    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name in names:
            return True
        return any(_contains_any_call(names, a) for a in node.args)

    if isinstance(node, (ANFConst, ANFVar)):
        return False

    if isinstance(node, ANFIf):
        return (_contains_any_call(names, node.test) or
                _contains_any_call(names, node.then_) or
                _contains_any_call(names, node.else_))

    if isinstance(node, ANFLet):
        return (_contains_any_call(names, node.rhs) or
                _contains_any_call(names, node.body))

    if isinstance(node, ANFLambda):
        return _contains_any_call(names, node.body)

    return False


def _rename_params(
    node: ANFNode, old_params: tuple[str, ...], new_params: tuple[str, ...]
) -> ANFNode:
    """Rename parameter references in an expression."""
    mapping = dict(zip(old_params, new_params))
    return _subst(node, mapping)


def _subst(node: ANFNode, mapping: dict[str, str]) -> ANFNode:
    """Substitute variable names according to mapping."""
    if isinstance(node, ANFConst):
        return node
    if isinstance(node, ANFVar):
        return ANFVar(mapping.get(node.name, node.name))
    if isinstance(node, ANFLet):
        new_rhs = _subst(node.rhs, mapping)
        inner = {k: v for k, v in mapping.items() if k != node.name}
        return ANFLet(name=node.name, rhs=new_rhs, body=_subst(node.body, inner))
    if isinstance(node, ANFIf):
        return ANFIf(
            test=_subst(node.test, mapping),
            then_=_subst(node.then_, mapping),
            else_=_subst(node.else_, mapping),
        )
    if isinstance(node, ANFApp):
        return ANFApp(
            func=_subst(node.func, mapping),
            args=tuple(_subst(a, mapping) for a in node.args),
        )
    if isinstance(node, ANFRecur):
        return ANFRecur(args=tuple(_subst(a, mapping) for a in node.args))
    if isinstance(node, ANFLambda):
        inner = {k: v for k, v in mapping.items() if k not in node.params}
        return ANFLambda(params=node.params, body=_subst(node.body, inner))
    if isinstance(node, ANFLoop):
        return ANFLoop(
            params=node.params,
            inits=tuple(_subst(i, mapping) for i in node.inits),
            body=_subst(node.body, {k: v for k, v in mapping.items()
                                    if k not in node.params}),
        )
    return node


def _mutual_calls_to_recur(
    node: ANFNode,
    bindings: tuple[tuple[str, ANFLambda], ...],
    unified_params: tuple[str, ...],
    max_arity: int,
) -> ANFNode:
    """Replace tail calls to any bound function with recur (tag, padded args)."""
    name_to_idx = {name: i for i, (name, _) in enumerate(bindings)}
    name_to_arity = {name: len(lam.params) for name, lam in bindings}

    def rewrite(n: ANFNode, in_tail: bool) -> ANFNode:
        if isinstance(n, ANFApp):
            if (isinstance(n.func, ANFVar) and n.func.name in name_to_idx and in_tail):
                idx = name_to_idx[n.func.name]
                arity = name_to_arity[n.func.name]
                padded = n.args + tuple(ANFConst(0.0) for _ in range(max_arity - arity))
                return ANFRecur(args=(ANFConst(float(idx)),) + padded)
            return n

        if isinstance(n, (ANFConst, ANFVar, ANFRecur)):
            return n

        if isinstance(n, ANFIf):
            return ANFIf(
                test=n.test,
                then_=rewrite(n.then_, in_tail),
                else_=rewrite(n.else_, in_tail),
            )

        if isinstance(n, ANFLet):
            return ANFLet(
                name=n.name,
                rhs=n.rhs,
                body=rewrite(n.body, in_tail),
            )

        return n

    return rewrite(node, in_tail=True)
