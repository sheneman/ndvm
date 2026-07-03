############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# engine_generic.py: Backend-agnostic evaluation engine. Mirrors the logic in engine.py but uses a NumpyFamilyBackend for all array...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Backend-agnostic evaluation engine.

Mirrors the logic in engine.py but uses a NumpyFamilyBackend for all
array operations instead of PyTorch directly. Supports numpy, jax, cupy.
"""

from __future__ import annotations

from neural_compiler.graph.builder import ComputeGraph
from neural_compiler.parser.ast_nodes import SchemeChar
from neural_compiler.backend.base import NumpyFamilyBackend, GenericHeap

DEFAULT_MAX_ITERATIONS = 10000
DEFAULT_MAX_RECURSION_DEPTH = 10000


class _TailCall:
    """Trampoline sentinel: represents a tail call through a closure."""
    __slots__ = ("closure_val", "arg_vals")
    def __init__(self, closure_val, arg_vals):
        self.closure_val = closure_val
        self.arg_vals = arg_vals


def _func_name_to_id(graph: ComputeGraph, func_name: str) -> int:
    return list(graph.functions.keys()).index(func_name)


def _func_id_to_name(graph: ComputeGraph, func_id: int) -> str:
    return list(graph.functions.keys())[func_id]


def _pack_env(capture_vals: list, heap: GenericHeap, backend: NumpyFamilyBackend) -> float:
    if not capture_vals:
        return -1.0
    env = heap.build_list(capture_vals)
    p = backend.extract_payload(env)
    return float(p[0])


def _unpack_env(env_addr: float, num_captures: int,
                heap: GenericHeap, backend: NumpyFamilyBackend) -> list:
    if num_captures == 0 or env_addr < 0:
        return []
    captures = []
    addr = int(env_addr)
    for _ in range(num_captures):
        car_val = heap.read(addr)
        captures.append(car_val)
        cdr_val = heap.read(addr + 1)
        p = backend.extract_payload(cdr_val)
        addr = int(float(p[0]))
    return captures


def _list_to_vec(lst, heap: GenericHeap, backend: NumpyFamilyBackend) -> list:
    result = []
    cur = lst
    while backend.is_pair(cur):
        result.append(heap.car(cur))
        cur = heap.cdr(cur)
    return result


def evaluate_generic(
    graph: ComputeGraph,
    inputs: dict,
    backend: NumpyFamilyBackend,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    raw_result: bool = False,
):
    inputs = inputs or {}
    for name in graph.input_names:
        if name not in inputs:
            raise ValueError(f"Missing input: {name}")

    if graph.uses_tagged_values:
        return _evaluate_tagged(graph, inputs, backend, max_iter, max_depth)

    return _evaluate_scalar(graph, inputs, backend, max_iter, max_depth, raw_result)


def _evaluate_scalar(
    graph: ComputeGraph,
    inputs: dict,
    backend: NumpyFamilyBackend,
    max_iter: int,
    max_depth: int,
    raw_result: bool = False,
):
    xp = backend.xp

    def to_arr(val):
        if hasattr(val, "shape"):
            return val
        return xp.float32(float(val))

    values: dict[int, object] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            val = node.value.code_point if isinstance(node.value, SchemeChar) else node.value
            values[nid] = xp.float32(float(val))

        elif node.op_type == "input":
            values[nid] = to_arr(inputs[node.name])

        elif node.op_type in ("loop_param", "func_param"):
            values[nid] = to_arr(inputs[node.name])

        elif node.op_type == "if":
            arg_vals = [values[eid] for eid in node.input_edges]
            values[nid] = backend.evaluate_op("if", arg_vals)

        elif node.op_type == "recur":
            values[nid] = xp.float32(0.0)

        elif node.op_type == "loop":
            values[nid] = _eval_loop_scalar(node, graph, values, backend, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_scalar(node, graph, values, backend, max_iter, max_depth, 0)

        else:
            arg_vals = [values[eid] for eid in node.input_edges]
            values[nid] = backend.evaluate_op(node.op_type, arg_vals)

    if graph.root_id is None:
        raise ValueError("Graph has no root node")

    result = values[graph.root_id]
    if raw_result:
        return result
    if hasattr(result, "item"):
        return result.item()
    return float(result)


def _eval_loop_scalar(loop_node, graph, values, backend, max_iter):
    xp = backend.xp
    loop_body = graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params = {}
    for i, param in enumerate(params):
        current_params[param] = values[loop_node.input_edges[i]]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = values[loop_node.input_edges[len(params) + i]]

    for _ in range(max_iter):
        body_values = {}
        for nid in loop_body.body_graph.topological_order():
            node = loop_body.body_graph.nodes[nid]
            if node.op_type == "const":
                val = node.value.code_point if isinstance(node.value, SchemeChar) else node.value
                body_values[nid] = xp.float32(float(val))
            elif node.op_type in ("loop_param", "func_param", "input"):
                body_values[nid] = current_params[node.name]
            elif node.op_type == "if":
                arg_vals = [body_values[eid] for eid in node.input_edges]
                body_values[nid] = backend.evaluate_op("if", arg_vals)
            elif node.op_type == "recur":
                body_values[nid] = xp.float32(0.0)
            else:
                arg_vals = [body_values[eid] for eid in node.input_edges]
                body_values[nid] = backend.evaluate_op(node.op_type, arg_vals)

        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                current_params[param] = body_values[root_node.input_edges[i]]
        elif root_node.op_type == "if":
            if _check_if_recurs_scalar(root_nid, loop_body.body_graph, body_values,
                                       params, current_params):
                continue
            return body_values[root_nid]
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Loop did not terminate after {max_iter} iterations")


def _check_if_recurs_scalar(nid, graph, values, params, current_params):
    node = graph.nodes[nid]
    if node.op_type != "if":
        return False
    test_val = float(values[node.input_edges[0]])
    taken_nid = node.input_edges[1] if test_val != 0.0 else node.input_edges[2]
    taken_node = graph.nodes[taken_nid]
    if taken_node.op_type == "recur":
        for i, param in enumerate(params):
            current_params[param] = values[taken_node.input_edges[i]]
        return True
    if taken_node.op_type == "if":
        return _check_if_recurs_scalar(taken_nid, graph, values, params, current_params)
    return False


def _eval_call_scalar(call_node, graph, values, backend, max_iter, max_depth, depth):
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")
    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    args = {}
    for i, param in enumerate(func_body.params):
        args[param] = values[call_node.input_edges[i]]
    memo = {}
    return _eval_lazy_scalar(func_body.body_graph.root_id, func_body.body_graph,
                             args, memo, backend, max_iter, max_depth, depth + 1)


def _eval_lazy_scalar(nid, graph, inputs, memo, backend, max_iter, max_depth, depth):
    if nid in memo:
        return memo[nid]
    xp = backend.xp
    node = graph.nodes[nid]

    if node.op_type == "const":
        val = node.value.code_point if isinstance(node.value, SchemeChar) else node.value
        result = xp.float32(float(val))
    elif node.op_type in ("func_param", "input", "loop_param"):
        result = inputs[node.name]
    elif node.op_type == "if":
        test = _eval_lazy_scalar(node.input_edges[0], graph, inputs, memo, backend,
                                 max_iter, max_depth, depth)
        if float(test) != 0.0:
            result = _eval_lazy_scalar(node.input_edges[1], graph, inputs, memo, backend,
                                       max_iter, max_depth, depth)
        else:
            result = _eval_lazy_scalar(node.input_edges[2], graph, inputs, memo, backend,
                                       max_iter, max_depth, depth)
    elif node.op_type == "call":
        result = _eval_call_lazy_scalar(node, graph, inputs, memo, backend,
                                        max_iter, max_depth, depth)
    elif node.op_type == "recur":
        result = xp.float32(0.0)
    elif node.op_type == "loop":
        loop_body = graph.loops[nid]
        params = loop_body.params
        current_params = {}
        for i, param in enumerate(params):
            current_params[param] = _eval_lazy_scalar(
                node.input_edges[i], graph, inputs, memo, backend, max_iter, max_depth, depth)
        for i, cap_name in enumerate(loop_body.captures):
            current_params[cap_name] = _eval_lazy_scalar(
                node.input_edges[len(params) + i], graph, inputs, memo, backend,
                max_iter, max_depth, depth)
        result = _eval_loop_scalar(
            type("_N", (), {"node_id": nid, "input_edges": node.input_edges})(),
            graph, {**{eid: _eval_lazy_scalar(eid, graph, inputs, memo, backend,
                       max_iter, max_depth, depth) for eid in node.input_edges},
                    **inputs},
            backend, max_iter)
    else:
        arg_vals = [_eval_lazy_scalar(e, graph, inputs, memo, backend,
                                      max_iter, max_depth, depth)
                    for e in node.input_edges]
        result = backend.evaluate_op(node.op_type, arg_vals)

    memo[nid] = result
    return result


def _eval_call_lazy_scalar(call_node, graph, inputs, memo, backend, max_iter, max_depth, depth):
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")
    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    args = {}
    for i, param in enumerate(func_body.params):
        args[param] = _eval_lazy_scalar(call_node.input_edges[i], graph, inputs, memo,
                                        backend, max_iter, max_depth, depth)
    body_memo = {}
    return _eval_lazy_scalar(func_body.body_graph.root_id, func_body.body_graph,
                             args, body_memo, backend, max_iter, max_depth, depth + 1)


# ===================================================================== #
# Tagged evaluation
# ===================================================================== #

def _evaluate_tagged(graph, inputs, backend, max_iter, max_depth):
    heap = backend.create_heap()

    tagged_inputs = {}
    for name, val in inputs.items():
        tagged_inputs[name] = backend.from_scalar(val) if not hasattr(val, "__len__") else val

    values = _eval_graph_tagged(graph, tagged_inputs, heap, backend, max_iter, max_depth)

    if graph.root_id is None:
        raise ValueError("Graph has no root node")
    return values[graph.root_id]


def _eval_graph_tagged(graph, inputs, heap, backend, max_iter, max_depth, depth=0):
    values = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            if isinstance(node.value, bool):
                values[nid] = backend.make_bool(node.value)
            elif isinstance(node.value, SchemeChar):
                values[nid] = backend.make_char(node.value.code_point)
            else:
                values[nid] = backend.make_float(node.value)

        elif node.op_type == "quote_const":
            values[nid] = backend.materialize_quote(node.value, heap)

        elif node.op_type in ("input", "loop_param", "func_param"):
            val = inputs[node.name]
            if hasattr(val, "__len__") and len(val) == 14:
                values[nid] = val
            else:
                values[nid] = backend.from_scalar(val)

        elif node.op_type == "if":
            test_val = values[node.input_edges[0]]
            then_val = values[node.input_edges[1]]
            else_val = values[node.input_edges[2]]
            values[nid] = backend.tagged_if(test_val, then_val, else_val)

        elif node.op_type == "recur":
            values[nid] = backend.make_nil()

        elif node.op_type == "loop":
            values[nid] = _eval_loop_tagged(node, graph, values, heap, backend, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_tagged(node, graph, values, heap, backend,
                                            max_iter, max_depth, depth)

        elif node.op_type == "make_closure":
            func_name = node.call_target
            func_id = _func_name_to_id(graph, func_name)
            capture_vals = [values[eid] for eid in node.input_edges]
            env_addr = _pack_env(capture_vals, heap, backend)
            values[nid] = backend.make_closure(func_id, env_addr)

        elif node.op_type == "dynamic_call":
            closure_val = values[node.input_edges[0]]
            arg_vals = [values[eid] for eid in node.input_edges[1:]]
            values[nid] = _eval_dynamic_call(
                closure_val, arg_vals, graph, heap, backend, max_iter, max_depth, depth)

        elif node.op_type == "apply":
            closure_val = values[node.input_edges[0]]
            args_list = values[node.input_edges[1]]
            arg_vals = _list_to_vec(args_list, heap, backend)
            values[nid] = _eval_dynamic_call(
                closure_val, arg_vals, graph, heap, backend, max_iter, max_depth, depth)

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = backend.evaluate_tagged_op(node.op_type, arg_tensors, heap)

    return values


def _eval_dynamic_call(closure_val, arg_vals, graph, heap, backend,
                       max_iter, max_depth, depth):
    if depth >= max_depth:
        raise RecursionError(f"Maximum recursion depth {max_depth} exceeded")

    for _ in range(max_iter):
        func_id, env_addr = backend.unwrap_closure(closure_val)
        func_id = int(func_id)
        func_name = _func_id_to_name(graph, func_id)
        func_body = graph.functions[func_name]

        body_inputs = {}
        for i, param in enumerate(func_body.params):
            body_inputs[param] = arg_vals[i]

        capture_vals = _unpack_env(env_addr, len(func_body.captures), heap, backend)
        for i, cap_name in enumerate(func_body.captures):
            body_inputs[cap_name] = capture_vals[i]

        body_memo = {}
        result = _eval_lazy_tagged(
            func_body.body_graph.root_id, func_body.body_graph,
            body_inputs, body_memo, heap, backend, max_iter, max_depth, depth + 1,
            tail_position=True,
        )

        if isinstance(result, _TailCall):
            closure_val = result.closure_val
            arg_vals = result.arg_vals
            continue

        return result

    raise RuntimeError(f"Trampoline did not terminate after {max_iter} iterations")


def _eval_loop_tagged(loop_node, graph, values, heap, backend, max_iter):
    loop_body = graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params = {}
    for i, param in enumerate(params):
        current_params[param] = values[loop_node.input_edges[i]]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = values[loop_node.input_edges[len(params) + i]]

    for _ in range(max_iter):
        body_values = _eval_graph_tagged(
            loop_body.body_graph, current_params, heap, backend,
            max_iter, DEFAULT_MAX_RECURSION_DEPTH)
        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                current_params[param] = body_values[root_node.input_edges[i]]
        elif root_node.op_type == "if":
            if _check_if_recurs_tagged(root_nid, loop_body.body_graph, body_values,
                                       params, current_params, backend):
                continue
            return body_values[root_nid]
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")


def _check_if_recurs_tagged(nid, graph, values, params, current_params, backend):
    node = graph.nodes[nid]
    if node.op_type != "if":
        return False

    test_val = backend.unwrap_number(values[node.input_edges[0]])
    taken_nid = node.input_edges[1] if test_val != 0.0 else node.input_edges[2]
    taken_node = graph.nodes[taken_nid]

    if taken_node.op_type == "recur":
        for i, param in enumerate(params):
            current_params[param] = values[taken_node.input_edges[i]]
        return True
    if taken_node.op_type == "if":
        return _check_if_recurs_tagged(taken_nid, graph, values, params,
                                       current_params, backend)
    return False


def _eval_call_tagged(call_node, graph, values, heap, backend,
                      max_iter, max_depth, depth):
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")
    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    args = {}
    for i, param in enumerate(func_body.params):
        args[param] = values[call_node.input_edges[i]]

    memo = {}
    return _eval_lazy_tagged(func_body.body_graph.root_id, func_body.body_graph,
                             args, memo, heap, backend, max_iter, max_depth, depth + 1)


def _eval_lazy_tagged(nid, graph, inputs, memo, heap, backend, max_iter, max_depth, depth,
                      tail_position=False):
    if nid in memo:
        return memo[nid]

    node = graph.nodes[nid]

    if node.op_type == "const":
        if isinstance(node.value, bool):
            result = backend.make_bool(node.value)
        elif isinstance(node.value, SchemeChar):
            result = backend.make_char(node.value.code_point)
        else:
            result = backend.make_float(node.value)

    elif node.op_type == "quote_const":
        result = backend.materialize_quote(node.value, heap)

    elif node.op_type in ("func_param", "input", "loop_param"):
        val = inputs[node.name]
        if hasattr(val, "__len__") and len(val) == 14:
            result = val
        else:
            result = backend.from_scalar(val)

    elif node.op_type == "if":
        test_val = _eval_lazy_tagged(node.input_edges[0], graph, inputs, memo, heap,
                                     backend, max_iter, max_depth, depth)
        if backend.unwrap_number(test_val) != 0.0:
            result = _eval_lazy_tagged(node.input_edges[1], graph, inputs, memo, heap,
                                       backend, max_iter, max_depth, depth,
                                       tail_position=tail_position)
        else:
            result = _eval_lazy_tagged(node.input_edges[2], graph, inputs, memo, heap,
                                       backend, max_iter, max_depth, depth,
                                       tail_position=tail_position)

    elif node.op_type == "call":
        result = _eval_call_lazy_tagged(node, graph, inputs, memo, heap, backend,
                                        max_iter, max_depth, depth)

    elif node.op_type == "make_closure":
        func_name = node.call_target
        func_id = _func_name_to_id(graph, func_name)
        capture_vals = [
            _eval_lazy_tagged(eid, graph, inputs, memo, heap, backend,
                              max_iter, max_depth, depth)
            for eid in node.input_edges
        ]
        env_addr = _pack_env(capture_vals, heap, backend)
        result = backend.make_closure(func_id, env_addr)

    elif node.op_type == "dynamic_call":
        closure_val = _eval_lazy_tagged(
            node.input_edges[0], graph, inputs, memo, heap, backend,
            max_iter, max_depth, depth)
        arg_vals = [
            _eval_lazy_tagged(eid, graph, inputs, memo, heap, backend,
                              max_iter, max_depth, depth)
            for eid in node.input_edges[1:]
        ]
        if tail_position:
            return _TailCall(closure_val, arg_vals)
        result = _eval_dynamic_call(closure_val, arg_vals, graph, heap, backend,
                                    max_iter, max_depth, depth)

    elif node.op_type == "apply":
        closure_val = _eval_lazy_tagged(
            node.input_edges[0], graph, inputs, memo, heap, backend,
            max_iter, max_depth, depth)
        args_list = _eval_lazy_tagged(
            node.input_edges[1], graph, inputs, memo, heap, backend,
            max_iter, max_depth, depth)
        arg_vals = _list_to_vec(args_list, heap, backend)
        if tail_position:
            return _TailCall(closure_val, arg_vals)
        result = _eval_dynamic_call(closure_val, arg_vals, graph, heap, backend,
                                    max_iter, max_depth, depth)

    elif node.op_type == "recur":
        result = backend.make_nil()

    elif node.op_type == "loop":
        loop_body = graph.loops[nid]
        params = loop_body.params
        current_params = {}
        for i, param in enumerate(params):
            current_params[param] = _eval_lazy_tagged(
                node.input_edges[i], graph, inputs, memo, heap, backend,
                max_iter, max_depth, depth)
        for i, cap_name in enumerate(loop_body.captures):
            current_params[cap_name] = _eval_lazy_tagged(
                node.input_edges[len(params) + i], graph, inputs, memo, heap,
                backend, max_iter, max_depth, depth)
        result = _eval_loop_tagged_body(loop_body, params, current_params,
                                        heap, backend, max_iter)
    else:
        arg_vals = [
            _eval_lazy_tagged(e, graph, inputs, memo, heap, backend,
                              max_iter, max_depth, depth)
            for e in node.input_edges
        ]
        result = backend.evaluate_tagged_op(node.op_type, arg_vals, heap)

    memo[nid] = result
    return result


def _eval_call_lazy_tagged(call_node, graph, inputs, memo, heap, backend,
                           max_iter, max_depth, depth):
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")
    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    args = {}
    for i, param in enumerate(func_body.params):
        args[param] = _eval_lazy_tagged(call_node.input_edges[i], graph, inputs, memo,
                                        heap, backend, max_iter, max_depth, depth)
    body_memo = {}
    return _eval_lazy_tagged(func_body.body_graph.root_id, func_body.body_graph,
                             args, body_memo, heap, backend, max_iter, max_depth, depth + 1)


def _eval_loop_tagged_body(loop_body, params, current_params, heap, backend, max_iter):
    for _ in range(max_iter):
        body_values = _eval_graph_tagged(
            loop_body.body_graph, current_params, heap, backend,
            max_iter, DEFAULT_MAX_RECURSION_DEPTH)
        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                current_params[param] = body_values[root_node.input_edges[i]]
        elif root_node.op_type == "if":
            if _check_if_recurs_tagged(root_nid, loop_body.body_graph, body_values,
                                       params, current_params, backend):
                continue
            return body_values[root_nid]
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")
