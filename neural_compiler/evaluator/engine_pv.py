############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# engine.py: Evaluation engine: execute a ComputeGraph with concrete inputs. Walks the graph in topological order. Each node...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Evaluation engine: execute a ComputeGraph with concrete inputs.

Walks the graph in topological order. Each node computes its value
from its input edges using the fixed-weight primitive operations.

For loops, the engine iteratively evaluates the loop body graph
until it produces a non-recur result. Each iteration feeds the
new parameter values back as inputs to the body graph.
"""

from __future__ import annotations
import torch
from neural_compiler.graph.builder import ComputeGraph
from neural_compiler.ops.primitives import evaluate_op
from neural_compiler.runtime.heap_pv import TensorHeap
from neural_compiler.parser.ast_nodes import SchemeChar
from neural_compiler.runtime.payload_value import (
    PV,
    VALUE_DIM, NIL, BOOL, INT, FLOAT, PAIR, SYMBOL, CLOSURE,
    make_nil, make_bool, make_int, make_float, make_char, make_pair, make_symbol,
    make_vector, TensorInput,
    make_closure as make_closure_val,
    extract_payload, type_index, unwrap_number, unwrap_closure,
    is_nil, is_pair, is_number, is_symbol, is_closure, tagged_if,
    from_scalar,
)
from neural_compiler.ops.tagged_ops_pv import evaluate_tagged_op, materialize_quote

DEFAULT_MAX_ITERATIONS = 10000
DEFAULT_MAX_RECURSION_DEPTH = 10000
DEFAULT_SOFT_CHOICE_TAU = 1.0

_soft_choice_tau = DEFAULT_SOFT_CHOICE_TAU
_soft_choice_use_gumbel = True
_soft_choice_hard = False


def set_soft_choice_tau(tau: float) -> None:
    """Set the Gumbel-softmax temperature for soft-choice nodes."""
    global _soft_choice_tau
    _soft_choice_tau = tau


def set_soft_choice_gumbel(use_gumbel: bool) -> None:
    """Toggle Gumbel noise. Disable for deterministic softmax (e.g. at test time)."""
    global _soft_choice_use_gumbel
    _soft_choice_use_gumbel = use_gumbel


def set_soft_choice_hard(hard: bool) -> None:
    """Toggle straight-through estimation: argmax forward, softmax backward."""
    global _soft_choice_hard
    _soft_choice_hard = hard


def _soft_choice_weights(logits: torch.Tensor) -> torch.Tensor:
    """Compute soft-choice weights from logits using current tau and gumbel setting."""
    if _soft_choice_use_gumbel:
        return torch.nn.functional.gumbel_softmax(
            logits, tau=_soft_choice_tau, hard=_soft_choice_hard,
        )
    soft = torch.nn.functional.softmax(logits / _soft_choice_tau, dim=0)
    if _soft_choice_hard:
        idx = soft.argmax()
        one_hot = torch.zeros_like(soft)
        one_hot[idx] = 1.0
        return one_hot - soft.detach() + soft  # straight-through
    return soft


def _soft_choice_combine(option_vals: list[torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
    """Combine option values via Gumbel-softmax weighted sum over logits."""
    k = len(option_vals)
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)
    logits = logits[:k]
    weights = _soft_choice_weights(logits)
    result = torch.zeros_like(option_vals[0])
    for i in range(k):
        result = result + weights[i] * option_vals[i]
    return result


def _soft_choice_combine_tagged(option_vals: list[torch.Tensor], logits_payload: torch.Tensor) -> torch.Tensor:
    """Combine tagged option values via Gumbel-softmax weighted sum.

    Each option is a tagged value [TAG_DIM + PAYLOAD_DIM]. The combination
    blends both tag and payload — at low temperature the dominant option's
    tag wins, giving a valid typed result.
    """
    k = len(option_vals)
    if logits_payload.dim() == 0:
        logits_payload = logits_payload.unsqueeze(0)
    logits = logits_payload[:k]
    weights = _soft_choice_weights(logits)
    result = torch.zeros_like(option_vals[0])
    for i in range(k):
        result = result + weights[i] * option_vals[i]
    return result


class _TailCall:
    """Trampoline sentinel: represents a tail call through a closure (dynamic dispatch)."""
    __slots__ = ("closure_val", "arg_vals")
    def __init__(self, closure_val, arg_vals):
        self.closure_val = closure_val
        self.arg_vals = arg_vals


class _TailCallNamed:
    """Trampoline sentinel: a tail call to a named (``letrec``) function.

    Returning this instead of recursing keeps the Python stack flat for tail-recursive named
    functions -- crucially the self-hosted evaluator's own recursion (``scheme-eval`` ->
    ``eval-apply`` -> ``scheme-eval`` ...), which makes recursive DMCI cheap rather than
    nesting thousands of Python frames per interpreted step.
    """
    __slots__ = ("func_name", "arg_vals")
    def __init__(self, func_name, arg_vals):
        self.func_name = func_name
        self.arg_vals = arg_vals


def _func_name_to_id(graph: ComputeGraph, func_name: str) -> int:
    names = list(graph.functions.keys())
    return names.index(func_name)


def _func_id_to_name(graph: ComputeGraph, func_id: int) -> str:
    names = list(graph.functions.keys())
    return names[func_id]


def _pack_env(capture_vals: list[torch.Tensor], heap: TensorHeap) -> float:
    """Pack captured values into a list on the heap, return the env address."""
    if not capture_vals:
        return -1.0
    env = heap.build_list(capture_vals)
    payload = extract_payload(env)
    return payload[0].item()


def _unpack_env(env_addr: float, num_captures: int, heap: TensorHeap) -> list[torch.Tensor]:
    """Unpack captured values from a heap-stored environment list."""
    if num_captures == 0 or env_addr < 0:
        return []
    captures = []
    addr = int(env_addr)
    for _ in range(num_captures):
        car_val = heap.read(addr)
        captures.append(car_val)
        cdr_val = heap.read(addr + 1)
        addr = int(extract_payload(cdr_val)[0].item())
    return captures


def _run_trampoline(
    state,
    graph: ComputeGraph,
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Unified tail-call trampoline. ``state`` is a ``_TailCallNamed`` (tail call to a named
    ``letrec`` function) or a ``_TailCall`` (tail call through a closure / dynamic dispatch).

    Both kinds bounce in this single flat loop -- including mutual recursion that crosses the
    named<->closure boundary -- so *every* tail call runs in constant Python stack and constant
    ``depth``. This is what makes recursive DMCI cheap: the self-hosted evaluator's own
    ``scheme-eval -> eval-apply -> scheme-eval`` loop trampolines instead of nesting a Python
    frame per interpreted step. ``depth`` is the genuine non-tail nesting at entry (guarded
    against ``max_depth``); tail bounces do not consume it -- the loop is bounded by ``max_iter``.
    """
    if depth >= max_depth:
        raise RecursionError(f"Maximum recursion depth {max_depth} exceeded")

    for _ in range(max_iter):
        if isinstance(state, _TailCallNamed):
            func_body = graph.functions[state.func_name]
            body_inputs: dict[str, torch.Tensor] = {
                param: state.arg_vals[i] for i, param in enumerate(func_body.params)
            }
        else:  # _TailCall: dispatch through the closure (function id + captured environment)
            func_id_t, env_addr_t = unwrap_closure(state.closure_val)
            func_name = _func_id_to_name(graph, int(func_id_t.item()))
            func_body = graph.functions[func_name]
            body_inputs = {param: state.arg_vals[i] for i, param in enumerate(func_body.params)}
            capture_vals = _unpack_env(env_addr_t.item(), len(func_body.captures), heap)
            for i, cap_name in enumerate(func_body.captures):
                body_inputs[cap_name] = capture_vals[i]

        body_memo: dict[int, torch.Tensor] = {}
        result = _eval_lazy_tagged(
            func_body.body_graph.root_id, func_body.body_graph,
            body_inputs, body_memo, heap, max_iter, max_depth, depth + 1,
            tail_position=True,
        )
        if isinstance(result, (_TailCall, _TailCallNamed)):  # bounce: stay flat, same depth
            state = result
            continue
        return result

    raise RuntimeError(f"Trampoline did not terminate after {max_iter} iterations")


def _eval_dynamic_call(
    closure_val: torch.Tensor,
    arg_vals: list[torch.Tensor],
    graph: ComputeGraph,
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Apply a closure value to ``arg_vals`` -- non-tail entry into the unified trampoline."""
    return _run_trampoline(
        _TailCall(closure_val, arg_vals), graph, heap, max_iter, max_depth, depth)


def _list_to_vec(lst: torch.Tensor, heap: TensorHeap) -> list[torch.Tensor]:
    """Unpack a Scheme list into a Python list of tagged values."""
    result = []
    cur = lst
    while is_pair(cur).item() > 0.5:
        result.append(heap.car(cur))
        cur = heap.cdr(cur)
    return result


def _to_tensor(val) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val
    return torch.tensor(val, dtype=torch.float32)


def _eval_graph(
    graph: ComputeGraph,
    inputs: dict,
    outer_values: dict[int, torch.Tensor] | None = None,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    depth: int = 0,
) -> dict[int, torch.Tensor]:
    """Evaluate a compute graph, returning values for all nodes."""
    values: dict[int, torch.Tensor] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            val = node.value.code_point if isinstance(node.value, SchemeChar) else node.value
            values[nid] = torch.tensor(float(val), dtype=torch.float32)

        elif node.op_type == "input":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "loop_param":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "func_param":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "if":
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op("if", arg_tensors)

        elif node.op_type == "recur":
            values[nid] = torch.tensor(0.0)

        elif node.op_type == "loop":
            values[nid] = _eval_loop(node, graph, values, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_from_eager(node, graph, values, max_iter, max_depth, depth)

        elif node.op_type == "soft_choice":
            num_opts = node.value
            opt_vals = [values[node.input_edges[i]] for i in range(num_opts)]
            logits = values[node.input_edges[num_opts]]
            values[nid] = _soft_choice_combine(opt_vals, logits)

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op(node.op_type, arg_tensors)

    return values


def _run_loop_body(
    body_graph: ComputeGraph,
    params: tuple[str, ...],
    current_params: dict[str, float],
    max_iter: int,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    depth: int = 0,
) -> torch.Tensor:
    """Run a loop body iteratively until termination."""
    for _ in range(max_iter):
        body_values = _eval_graph(
            body_graph, current_params, max_iter=max_iter, max_depth=max_depth, depth=depth
        )
        root_nid = body_graph.root_id
        root_node = body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = root_node.input_edges[i]
                current_params[param] = body_values[arg_nid]
        elif root_node.op_type == "if":
            result = body_values[root_nid]
            if _check_if_recurs(root_nid, body_graph, body_values, params, current_params):
                continue
            return result
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Loop did not terminate after {max_iter} iterations")


def _eval_loop(
    loop_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Evaluate a loop by iterating the body graph."""
    loop_body = outer_graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params = {}
    for i, param in enumerate(params):
        init_nid = loop_node.input_edges[i]
        current_params[param] = outer_values[init_nid]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

    return _run_loop_body(loop_body.body_graph, params, current_params, max_iter)


def _check_if_recurs(
    nid: int,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    params: tuple[str, ...],
    current_params: dict,
) -> bool:
    """Check if the result of an if-expression is a recur node.

    When the loop body is: (if test (recur ...) result) or (if test result (recur ...)),
    we need to determine which branch was taken and extract recur args if applicable.
    """
    node = graph.nodes[nid]
    if node.op_type != "if":
        return False

    test_nid = node.input_edges[0]
    then_nid = node.input_edges[1]
    else_nid = node.input_edges[2]

    test_val = values[test_nid].item()
    taken_nid = then_nid if test_val != 0.0 else else_nid
    taken_node = graph.nodes[taken_nid]

    if taken_node.op_type == "recur":
        for i, param in enumerate(params):
            arg_nid = taken_node.input_edges[i]
            current_params[param] = values[arg_nid]
        return True

    if taken_node.op_type == "if":
        return _check_if_recurs(taken_nid, graph, values, params, current_params)

    return False


def _eval_lazy(
    nid: int,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Demand-driven evaluation: only compute nodes actually needed.

    Unlike topological evaluation, if-nodes evaluate lazily — only the taken
    branch is evaluated. This prevents infinite recursion in base cases.
    """
    if nid in memo:
        return memo[nid]

    node = graph.nodes[nid]

    if node.op_type == "const":
        result = torch.tensor(node.value, dtype=torch.float32)

    elif node.op_type in ("func_param", "input", "loop_param"):
        result = _to_tensor(inputs[node.name])

    elif node.op_type == "if":
        test_val = _eval_lazy(node.input_edges[0], graph, inputs, memo, max_iter, max_depth, depth)
        if test_val.item() != 0.0:
            result = _eval_lazy(node.input_edges[1], graph, inputs, memo, max_iter, max_depth, depth)
        else:
            result = _eval_lazy(node.input_edges[2], graph, inputs, memo, max_iter, max_depth, depth)

    elif node.op_type == "loop":
        loop_body = graph.loops[nid]
        params = loop_body.params
        current_params = {}
        for i, param in enumerate(params):
            current_params[param] = _eval_lazy(
                node.input_edges[i], graph, inputs, memo, max_iter, max_depth, depth
            )
        for i, cap_name in enumerate(loop_body.captures):
            current_params[cap_name] = _eval_lazy(
                node.input_edges[len(params) + i], graph, inputs, memo, max_iter, max_depth, depth
            )
        result = _run_loop_body(
            loop_body.body_graph, params, current_params, max_iter, max_depth, depth
        )

    elif node.op_type == "call":
        result = _eval_call(node, graph, inputs, memo, max_iter, max_depth, depth)

    elif node.op_type == "recur":
        result = torch.tensor(0.0)

    elif node.op_type == "soft_choice":
        num_opts = node.value
        opt_vals = [_eval_lazy(node.input_edges[i], graph, inputs, memo, max_iter, max_depth, depth) for i in range(num_opts)]
        logits = _eval_lazy(node.input_edges[num_opts], graph, inputs, memo, max_iter, max_depth, depth)
        result = _soft_choice_combine(opt_vals, logits)

    else:
        arg_vals = [_eval_lazy(e, graph, inputs, memo, max_iter, max_depth, depth) for e in node.input_edges]
        result = evaluate_op(node.op_type, arg_vals)

    memo[nid] = result
    return result


def _eval_call_from_eager(
    call_node,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Bridge from eager topological eval into lazy function body eval."""
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    body_graph = func_body.body_graph

    args = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = values[arg_nid]

    body_memo: dict[int, torch.Tensor] = {}
    return _eval_lazy(
        body_graph.root_id, body_graph, args, body_memo, max_iter, max_depth, depth + 1
    )


def _eval_call(
    call_node,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Evaluate a recursive function call within lazy evaluation."""
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    body_graph = func_body.body_graph

    args = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = _eval_lazy(arg_nid, graph, inputs, memo, max_iter, max_depth, depth)

    body_memo: dict[int, torch.Tensor] = {}
    return _eval_lazy(
        body_graph.root_id, body_graph, args, body_memo, max_iter, max_depth, depth + 1
    )


def evaluate(
    graph: ComputeGraph,
    inputs: dict = None,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    max_heap: int | None = None,
):
    """Evaluate a compute graph with the given input values.

    Args:
        graph: The compiled compute graph.
        inputs: Map of input variable names to values (float or tensor).
        max_iter: Maximum loop iterations before raising an error.
        max_depth: Maximum non-tail recursion depth before raising an error.
        max_heap: Cap on heap cells for tagged (heap-backed) programs. ``None`` keeps the
            default (:data:`~neural_compiler.runtime.heap.DEFAULT_MAX_HEAP`). The heap is a
            bump allocator with no garbage collection, so recursive DMCI conses linearly with
            interpreted steps -- raise this for recursive/deep meta-circular programs (the heap
            is dict-backed, so a larger cap costs nothing until those cells are actually used).

    Returns:
        Scalar float for scalar programs, or torch.Tensor for vector/matrix programs.
        For tagged-value programs, returns the raw TaggedValue tensor.
    """
    inputs = inputs or {}
    for name in graph.input_names:
        if name not in inputs:
            raise ValueError(f"Missing input: {name}")

    if graph.uses_tagged_values:
        return _evaluate_tagged(graph, inputs, max_iter, max_depth, max_heap)

    values = _eval_graph(graph, inputs, max_iter=max_iter, max_depth=max_depth)

    if graph.root_id is None:
        raise ValueError("Graph has no root node")

    result = values[graph.root_id]
    if result.dim() == 0:
        return result.item()
    return result


def _evaluate_tagged(
    graph: ComputeGraph,
    inputs: dict,
    max_iter: int,
    max_depth: int,
    max_heap: int | None = None,
    heap: "TensorHeap | None" = None,
) -> torch.Tensor:
    """Evaluate a tagged-value compute graph. Returns raw TaggedValue tensor.

    ``heap`` may be a pre-populated TensorHeap -- e.g. carrying a program-as-data input
    for the bare meta-circular interpreter (see ``neural_compiler.dmci.evaluate_program``).
    When ``None`` (the default) a fresh heap is allocated, preserving existing behaviour."""
    if heap is None:
        heap = TensorHeap() if max_heap is None else TensorHeap(max_size=max_heap)

    tagged_inputs = {}
    for name, val in inputs.items():
        # A TensorInput (as_vector/as_matrix) binds a raw tensor as a VECTOR/MATRIX payload:
        # store it on THIS heap and bind a VECTOR ref, so (ref name k)/matvec/... use it and
        # gradients flow. (Without this it would hit make_float -> a batch of scalars.)
        if isinstance(val, TensorInput):
            addr = heap.store(val.tensor)
            tagged_inputs[name] = make_vector(float(addr), float(val.feature_ndim),
                                              device=heap.device)
        # Accept an already-tagged value, scalar [VALUE_DIM] OR batched [N, VALUE_DIM].
        # Batched leaves flow through the heap-backed evaluator unchanged: structural
        # values (pairs/symbols/AST/counters) stay scalar and data-independent, only
        # numeric payloads carry the batch dimension, and arithmetic broadcasts.
        elif isinstance(val, PV):
            tagged_inputs[name] = val
        else:
            tagged_inputs[name] = from_scalar(val) if not isinstance(val, torch.Tensor) else make_float(val)

    values = _eval_graph_tagged(graph, tagged_inputs, heap, max_iter, max_depth)

    if graph.root_id is None:
        raise ValueError("Graph has no root node")

    return values[graph.root_id]


def _batch_branch_decision(test_val: torch.Tensor) -> bool:
    """Reduce a (possibly batched) tagged truth value to a single branch decision.

    The recursive/lazy tagged evaluator must descend into exactly one branch per call
    (this is what makes recursion terminate), so it needs a scalar decision. Data-
    independent control flow -- the interpreter's own structural dispatch, loop
    counters, etc. -- yields a truth that is uniform across the batch and decides
    cleanly (returning a scalar, or a batched-but-uniform vector). If the batch elements
    *disagree*, the program's control flow depends on the batched input and cannot be
    vectorized through the recursive interpreter; we raise a clear error instead of the
    generic "Tensor with N elements cannot be converted to Scalar"."""
    truth = unwrap_number(test_val)
    if not isinstance(truth, torch.Tensor):          # payload structural truth is a Python scalar
        return float(truth) != 0.0
    if truth.numel() == 1:
        return truth.flatten()[0].item() != 0.0
    nonzero = truth != 0.0
    if bool(nonzero.all()):
        return True
    if bool((~nonzero).all()):
        return False
    raise ValueError(
        "Batched DMCI requires data-independent control flow: this program's branch "
        "decision differs across the batch (some elements would take the 'then' branch "
        "and others 'else'). The recursive interpreter descends into a single branch "
        "per call, so a branch whose test depends on the batched input cannot be "
        "vectorized this way. Evaluate such programs one input at a time (unbatched), "
        "or -- for non-recursive arithmetic -- use the heap-free batched path, where a "
        "data-dependent `if` is soft-masked per element."
    )


def _eval_graph_tagged(
    graph: ComputeGraph,
    inputs: dict,
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int = 0,
) -> dict[int, torch.Tensor]:
    """Evaluate a tagged-value compute graph, returning TaggedValue tensors."""
    values: dict[int, torch.Tensor] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            if isinstance(node.value, bool):
                values[nid] = make_bool(node.value)
            elif isinstance(node.value, SchemeChar):
                values[nid] = make_char(node.value.code_point)
            else:
                values[nid] = make_float(node.value)

        elif node.op_type == "quote_const":
            values[nid] = materialize_quote(node.value, heap)

        elif node.op_type in ("input", "loop_param", "func_param"):
            val = inputs[node.name]
            if isinstance(val, PV):
                values[nid] = val
            else:
                values[nid] = from_scalar(val) if not isinstance(val, torch.Tensor) else make_float(val)

        elif node.op_type == "if":
            test_val = values[node.input_edges[0]]
            then_val = values[node.input_edges[1]]
            else_val = values[node.input_edges[2]]
            values[nid] = tagged_if(test_val, then_val, else_val)

        elif node.op_type == "recur":
            values[nid] = make_nil()

        elif node.op_type == "loop":
            values[nid] = _eval_loop_tagged(node, graph, values, heap, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_tagged(node, graph, values, heap, max_iter, max_depth, depth)

        elif node.op_type == "make_closure":
            func_name = node.call_target
            func_id = _func_name_to_id(graph, func_name)
            capture_vals = [values[eid] for eid in node.input_edges]
            env_addr = _pack_env(capture_vals, heap)
            values[nid] = make_closure_val(func_id, env_addr)

        elif node.op_type == "dynamic_call":
            closure_val = values[node.input_edges[0]]
            arg_vals = [values[eid] for eid in node.input_edges[1:]]
            values[nid] = _eval_dynamic_call(
                closure_val, arg_vals, graph, heap, max_iter, max_depth, depth
            )

        elif node.op_type == "apply":
            closure_val = values[node.input_edges[0]]
            args_list = values[node.input_edges[1]]
            arg_vals = _list_to_vec(args_list, heap)
            values[nid] = _eval_dynamic_call(
                closure_val, arg_vals, graph, heap, max_iter, max_depth, depth
            )

        elif node.op_type == "soft_choice":
            num_opts = node.value
            opt_vals = [values[node.input_edges[i]] for i in range(num_opts)]
            logits = values[node.input_edges[num_opts]]
            logits_scalar = extract_payload(logits)
            result = _soft_choice_combine_tagged(opt_vals, logits_scalar)
            values[nid] = result

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_tagged_op(node.op_type, arg_tensors, heap)

    return values


def _eval_loop_tagged(
    loop_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    heap: TensorHeap,
    max_iter: int,
) -> torch.Tensor:
    """Evaluate a tagged-value loop."""
    loop_body = outer_graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params = {}
    for i, param in enumerate(params):
        init_nid = loop_node.input_edges[i]
        current_params[param] = outer_values[init_nid]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

    for _ in range(max_iter):
        body_values = _eval_graph_tagged(
            loop_body.body_graph, current_params, heap, max_iter, DEFAULT_MAX_RECURSION_DEPTH
        )
        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = root_node.input_edges[i]
                current_params[param] = body_values[arg_nid]
        elif root_node.op_type == "if":
            result = body_values[root_nid]
            if _check_if_recurs_tagged(root_nid, loop_body.body_graph, body_values, params, current_params):
                continue
            return result
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")


def _check_if_recurs_tagged(
    nid: int,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    params: tuple[str, ...],
    current_params: dict,
) -> bool:
    """Check if tagged if-expression result is a recur. Uses tagged truth value."""
    node = graph.nodes[nid]
    if node.op_type != "if":
        return False

    test_nid = node.input_edges[0]
    then_nid = node.input_edges[1]
    else_nid = node.input_edges[2]

    taken_nid = then_nid if _batch_branch_decision(values[test_nid]) else else_nid
    taken_node = graph.nodes[taken_nid]

    if taken_node.op_type == "recur":
        for i, param in enumerate(params):
            arg_nid = taken_node.input_edges[i]
            current_params[param] = values[arg_nid]
        return True

    if taken_node.op_type == "if":
        return _check_if_recurs_tagged(taken_nid, graph, values, params, current_params)

    return False


def _eval_call_tagged(
    call_node,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Evaluate a function call in tagged mode using lazy evaluation."""
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    body_graph = func_body.body_graph

    args = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = values[arg_nid]

    memo: dict[int, torch.Tensor] = {}
    return _eval_lazy_tagged(body_graph.root_id, body_graph, args, memo, heap, max_iter, max_depth, depth + 1)


def _eval_lazy_tagged(
    nid: int,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
    tail_position: bool = False,
) -> torch.Tensor:
    """Demand-driven tagged evaluation for recursive functions."""
    if nid in memo:
        return memo[nid]

    node = graph.nodes[nid]

    if node.op_type == "const":
        if isinstance(node.value, bool):
            result = make_bool(node.value)
        elif isinstance(node.value, SchemeChar):
            result = make_char(node.value.code_point)
        else:
            result = make_float(node.value)
    elif node.op_type == "quote_const":
        result = materialize_quote(node.value, heap)
    elif node.op_type in ("func_param", "input", "loop_param"):
        val = inputs[node.name]
        if isinstance(val, PV):
            result = val
        else:
            result = from_scalar(val) if not isinstance(val, torch.Tensor) else make_float(val)
    elif node.op_type == "if":
        test_val = _eval_lazy_tagged(node.input_edges[0], graph, inputs, memo, heap, max_iter, max_depth, depth)
        branch = node.input_edges[1] if _batch_branch_decision(test_val) else node.input_edges[2]
        result = _eval_lazy_tagged(branch, graph, inputs, memo, heap, max_iter, max_depth, depth, tail_position=tail_position)
        if isinstance(result, (_TailCall, _TailCallNamed)):
            # tail-position branch produced a pending tail call: hand it up to the trampoline
            # unchanged -- never memoize a control-flow sentinel as if it were a value.
            return result
    elif node.op_type == "call":
        if tail_position:
            # bounce instead of recursing: evaluate args here, hand the call to the trampoline.
            func_body = graph.functions[node.call_target]
            arg_vals = [
                _eval_lazy_tagged(node.input_edges[i], graph, inputs, memo, heap,
                                  max_iter, max_depth, depth)
                for i in range(len(func_body.params))
            ]
            return _TailCallNamed(node.call_target, arg_vals)
        result = _eval_call_lazy_tagged(node, graph, inputs, memo, heap, max_iter, max_depth, depth)
    elif node.op_type == "make_closure":
        func_name = node.call_target
        func_id = _func_name_to_id(graph, func_name)
        capture_vals = [
            _eval_lazy_tagged(eid, graph, inputs, memo, heap, max_iter, max_depth, depth)
            for eid in node.input_edges
        ]
        env_addr = _pack_env(capture_vals, heap)
        result = make_closure_val(func_id, env_addr)
    elif node.op_type == "dynamic_call":
        closure_val = _eval_lazy_tagged(
            node.input_edges[0], graph, inputs, memo, heap, max_iter, max_depth, depth
        )
        arg_vals = [
            _eval_lazy_tagged(eid, graph, inputs, memo, heap, max_iter, max_depth, depth)
            for eid in node.input_edges[1:]
        ]
        if tail_position:
            return _TailCall(closure_val, arg_vals)
        result = _eval_dynamic_call(
            closure_val, arg_vals, graph, heap, max_iter, max_depth, depth
        )
    elif node.op_type == "apply":
        closure_val = _eval_lazy_tagged(
            node.input_edges[0], graph, inputs, memo, heap, max_iter, max_depth, depth
        )
        args_list = _eval_lazy_tagged(
            node.input_edges[1], graph, inputs, memo, heap, max_iter, max_depth, depth
        )
        arg_vals = _list_to_vec(args_list, heap)
        if tail_position:
            return _TailCall(closure_val, arg_vals)
        result = _eval_dynamic_call(
            closure_val, arg_vals, graph, heap, max_iter, max_depth, depth
        )
    elif node.op_type == "recur":
        result = make_nil()
    elif node.op_type == "loop":
        loop_body = graph.loops[nid]
        params = loop_body.params
        current_params = {}
        for i, param in enumerate(params):
            current_params[param] = _eval_lazy_tagged(
                node.input_edges[i], graph, inputs, memo, heap, max_iter, max_depth, depth
            )
        for i, cap_name in enumerate(loop_body.captures):
            current_params[cap_name] = _eval_lazy_tagged(
                node.input_edges[len(params) + i], graph, inputs, memo, heap, max_iter, max_depth, depth
            )
        result = _eval_loop_lazy_tagged(loop_body, params, current_params, heap, max_iter, max_depth, depth)
    elif node.op_type == "soft_choice":
        num_opts = node.value
        opt_vals = [
            _eval_lazy_tagged(node.input_edges[i], graph, inputs, memo, heap, max_iter, max_depth, depth)
            for i in range(num_opts)
        ]
        logits_tagged = _eval_lazy_tagged(node.input_edges[num_opts], graph, inputs, memo, heap, max_iter, max_depth, depth)
        logits_scalar = extract_payload(logits_tagged)
        result = _soft_choice_combine_tagged(opt_vals, logits_scalar)
    else:
        arg_vals = [_eval_lazy_tagged(e, graph, inputs, memo, heap, max_iter, max_depth, depth) for e in node.input_edges]
        result = evaluate_tagged_op(node.op_type, arg_vals, heap)

    memo[nid] = result
    return result


def _trace_loop_root_lazy(
    nid: int,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> tuple[int, dict[int, torch.Tensor]]:
    """Trace through if-nodes lazily to find the terminal node (recur or value)."""
    node = graph.nodes[nid]
    if node.op_type == "if":
        test_val = _eval_lazy_tagged(node.input_edges[0], graph, inputs, memo, heap, max_iter, max_depth, depth)
        if _batch_branch_decision(test_val):
            return _trace_loop_root_lazy(node.input_edges[1], graph, inputs, memo, heap, max_iter, max_depth, depth)
        else:
            return _trace_loop_root_lazy(node.input_edges[2], graph, inputs, memo, heap, max_iter, max_depth, depth)
    return nid, memo


def _eval_loop_lazy_tagged(
    loop_body,
    params: tuple[str, ...],
    current_params: dict,
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Evaluate a loop body lazily — only evaluates the taken branch of if-nodes."""
    body_graph = loop_body.body_graph
    for _ in range(max_iter):
        memo: dict[int, torch.Tensor] = {}
        terminal_nid, memo = _trace_loop_root_lazy(
            body_graph.root_id, body_graph, current_params, memo, heap, max_iter, max_depth, depth
        )
        terminal_node = body_graph.nodes[terminal_nid]
        if terminal_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = terminal_node.input_edges[i]
                current_params[param] = _eval_lazy_tagged(
                    arg_nid, body_graph, current_params, memo, heap, max_iter, max_depth, depth
                )
        else:
            return _eval_lazy_tagged(terminal_nid, body_graph, current_params, memo, heap, max_iter, max_depth, depth)
    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")


def _trampoline_named(
    func_name: str,
    arg_vals: list[torch.Tensor],
    graph: ComputeGraph,
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Apply a named (``letrec``) function to ``arg_vals`` -- non-tail entry into the unified
    trampoline (see :func:`_run_trampoline`)."""
    return _run_trampoline(
        _TailCallNamed(func_name, arg_vals), graph, heap, max_iter, max_depth, depth)


def _eval_call_lazy_tagged(
    call_node,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    heap: TensorHeap,
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    arg_vals = [
        _eval_lazy_tagged(call_node.input_edges[i], graph, inputs, memo, heap,
                          max_iter, max_depth, depth)
        for i in range(len(func_body.params))
    ]
    return _trampoline_named(func_name, arg_vals, graph, heap, max_iter, max_depth, depth)


def _eval_loop_tagged_body(
    loop_body,
    params: tuple[str, ...],
    current_params: dict,
    heap: TensorHeap,
    max_iter: int,
) -> torch.Tensor:
    for _ in range(max_iter):
        body_values = _eval_graph_tagged(
            loop_body.body_graph, current_params, heap, max_iter, DEFAULT_MAX_RECURSION_DEPTH
        )
        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = root_node.input_edges[i]
                current_params[param] = body_values[arg_nid]
        elif root_node.op_type == "if":
            if _check_if_recurs_tagged(root_nid, loop_body.body_graph, body_values, params, current_params):
                continue
            return body_values[root_nid]
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")


# ============================================================
# Batched evaluation — vectorized forward pass over N inputs
# ============================================================

_HEAP_OPS = frozenset({
    "cons", "car", "cdr", "list", "length", "append", "reverse",
    "null?", "pair?", "symbol?", "char?", "string?", "vector?",
    "eq?", "eqv?", "equal?", "make_closure", "dynamic_call", "apply",
    "quote_const",
})


def _graph_uses_heap(graph) -> bool:
    """True if the graph or any of its function/loop bodies uses heap operations.

    The eager heap-free batched evaluator (_eval_graph_tagged_batched) cannot run
    such graphs; the meta-circular interpreter (DMCI) is the canonical case — it
    materializes program-as-data via quote_const and builds the environment with
    cons/list. These are routed to the heap-backed evaluator, which batches natively.
    """
    def _has(g) -> bool:
        return any(g.nodes[nid].op_type in _HEAP_OPS for nid in g.nodes)
    if _has(graph):
        return True
    for fb in getattr(graph, "functions", {}).values():
        if _has(fb.body_graph):
            return True
    for lb in getattr(graph, "loops", {}).values():
        if _has(lb.body_graph):
            return True
    return False


def evaluate_batched(
    graph: ComputeGraph,
    inputs: dict[str, torch.Tensor],
    max_iter: int = DEFAULT_MAX_ITERATIONS,
) -> torch.Tensor:
    """Evaluate a ComputeGraph over a batch of inputs.

    For tagged graphs: inputs should be (N, VALUE_DIM) or (VALUE_DIM,).
    For non-tagged graphs: inputs should be (N,) or scalar tensors.
    Scalars/unbatched broadcast to match batch dimensions.

    Heap-using tagged graphs (cons/car/cdr/list/quote_const, i.e. the meta-circular
    interpreter / DMCI) are routed to the heap-backed evaluator, which batches natively:
    structural values stay scalar and data-independent while only numeric leaves carry
    the batch dimension. Heap-free graphs take the fast eager batched walk.
    """
    if graph.root_id is None:
        raise ValueError("Graph has no root node")

    if graph.uses_tagged_values:
        if _graph_uses_heap(graph):
            # Heap-backed programs (DMCI / meta-circular interpreter) batch natively
            # through the heap-backed evaluator: structural values stay scalar and
            # data-independent, only numeric leaves carry the batch dimension. The
            # eager heap-free path below cannot run them (quote_const/cons/etc.).
            return _evaluate_tagged(graph, inputs, max_iter, DEFAULT_MAX_RECURSION_DEPTH)
        from neural_compiler.runtime.payload_value import VALUE_DIM
        tagged_inputs: dict[str, torch.Tensor] = {}
        for name, val in inputs.items():
            if isinstance(val, PV):
                tagged_inputs[name] = val
            elif isinstance(val, torch.Tensor):
                if val.dim() >= 1 and val.shape[-1] == VALUE_DIM:
                    tagged_inputs[name] = val
                else:
                    tagged_inputs[name] = make_float(val)
            else:
                tagged_inputs[name] = from_scalar(val)
        values = _eval_graph_tagged_batched(graph, tagged_inputs, max_iter)
    else:
        tensor_inputs: dict[str, torch.Tensor] = {}
        for name, val in inputs.items():
            tensor_inputs[name] = _to_tensor(val)
        values = _eval_graph_batched(graph, tensor_inputs, max_iter)

    return values[graph.root_id]


def compile_batched(
    graph: ComputeGraph,
    input_names: list[str] | None = None,
    mode: str = "reduce-overhead",
) -> callable:
    """Return a torch.compiled version of evaluate_batched for a fixed graph.

    Args:
        graph: The compiled compute graph (must not change between calls).
        input_names: Ordered list of input names. If None, uses graph.input_names.
        mode: torch.compile mode ('reduce-overhead', 'default', 'max-autotune').

    Returns:
        A callable that takes positional tensor arguments in input_names order
        and returns the batched result tensor.
    """
    names = input_names or list(graph.input_names)

    def _forward(*args):
        inputs = {name: arg for name, arg in zip(names, args)}
        return evaluate_batched(graph, inputs)

    return torch.compile(_forward, mode=mode, dynamic=True)


def _eval_graph_tagged_batched(
    graph: ComputeGraph,
    inputs: dict[str, torch.Tensor],
    max_iter: int,
) -> dict[int, torch.Tensor]:
    """Topological evaluation with batched tagged values — no heap."""
    from neural_compiler.runtime.payload_value import VALUE_DIM
    from neural_compiler.ops.tagged_ops_pv import ARITH_OPS, COMPARE_OPS, LOGIC_OPS

    values: dict[int, torch.Tensor] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            if isinstance(node.value, bool):
                values[nid] = make_bool(node.value)
            elif isinstance(node.value, SchemeChar):
                values[nid] = make_char(node.value.code_point)
            else:
                values[nid] = make_float(node.value)

        elif node.op_type in ("input", "loop_param", "func_param"):
            val = inputs[node.name]
            if isinstance(val, PV):
                values[nid] = val
            else:
                values[nid] = from_scalar(val) if not isinstance(val, torch.Tensor) else make_float(val)

        elif node.op_type == "if":
            test_val = values[node.input_edges[0]]
            then_val = values[node.input_edges[1]]
            else_val = values[node.input_edges[2]]
            values[nid] = tagged_if(test_val, then_val, else_val)

        elif node.op_type == "recur":
            values[nid] = make_nil()

        elif node.op_type == "loop":
            values[nid] = _eval_loop_tagged_batched(node, graph, values, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_tagged_batched(node, graph, values, max_iter)

        elif node.op_type in _HEAP_OPS:
            raise NotImplementedError(
                f"Batched evaluation does not support heap operation '{node.op_type}'. "
                f"Only arithmetic/comparison/logic programs are supported."
            )

        elif node.op_type == "soft_choice":
            num_opts = node.value
            opt_vals = [values[node.input_edges[i]] for i in range(num_opts)]
            logits = values[node.input_edges[num_opts]]
            logits_scalar = extract_payload(logits)
            result = _soft_choice_combine_tagged(opt_vals, logits_scalar)
            values[nid] = result

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = _evaluate_tagged_op_batched(node.op_type, arg_tensors)

    return values


def _evaluate_tagged_op_batched(op_type: str, args: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate a primitive op on batched tagged values without heap."""
    from neural_compiler.ops.tagged_ops_pv import ARITH_OPS, COMPARE_OPS, LOGIC_OPS

    if op_type in ARITH_OPS:
        raw_args = [unwrap_number(a) for a in args]
        result = evaluate_op(op_type, raw_args)
        return make_float(result)
    if op_type in COMPARE_OPS:
        raw_args = [unwrap_number(a) for a in args]
        result = evaluate_op(op_type, raw_args)
        return make_bool(result)
    if op_type in LOGIC_OPS:
        raw_args = [unwrap_number(a) for a in args]
        result = evaluate_op(op_type, raw_args)
        return make_bool(result)
    if op_type in ("number?", "boolean?", "procedure?"):
        from neural_compiler.runtime.payload_value import is_number, is_bool, is_closure
        checks = {"number?": is_number, "boolean?": is_bool, "procedure?": is_closure}
        return make_bool(checks[op_type](args[0]))
    raise ValueError(f"Unsupported batched operation: {op_type}")


def _eval_loop_tagged_batched(
    loop_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Batched loop: iterate until all elements terminate."""
    loop_body = outer_graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params: dict[str, torch.Tensor] = {}
    for i, param in enumerate(params):
        init_nid = loop_node.input_edges[i]
        current_params[param] = outer_values[init_nid]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

    for _ in range(max_iter):
        body_values = _eval_graph_tagged_batched(
            loop_body.body_graph, current_params, max_iter
        )
        root_nid = loop_body.body_graph.root_id
        root_node = loop_body.body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = root_node.input_edges[i]
                current_params[param] = body_values[arg_nid]
        elif root_node.op_type == "if":
            if _check_if_recurs_tagged_batched(
                root_nid, loop_body.body_graph, body_values, params, current_params
            ):
                continue
            return body_values[root_nid]
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Tagged loop did not terminate after {max_iter} iterations")


def _check_if_recurs_tagged_batched(
    if_nid: int,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    params: tuple[str, ...],
    current_params: dict[str, torch.Tensor],
) -> bool:
    """Check if an if-node's taken branch is a recur (batched scalar path)."""
    node = graph.nodes[if_nid]
    test_val = values[node.input_edges[0]]
    truth = unwrap_number(test_val)
    if truth.dim() == 0:
        taken = truth.item() != 0.0
    else:
        taken = truth.flatten()[0].item() != 0.0

    branch_nid = node.input_edges[1] if taken else node.input_edges[2]
    branch_node = graph.nodes[branch_nid]

    if branch_node.op_type == "recur":
        for i, param in enumerate(params):
            arg_nid = branch_node.input_edges[i]
            current_params[param] = values[arg_nid]
        return True
    return False


def _eval_call_tagged_batched(
    call_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Batched function call — inlines the callee body graph."""
    func_name = call_node.call_target
    if func_name not in outer_graph.functions:
        raise ValueError(f"Batched eval: unknown function '{func_name}'")
    func_body = outer_graph.functions[func_name]
    body_graph = func_body.body_graph

    args: dict[str, torch.Tensor] = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = outer_values[arg_nid]

    body_values = _eval_graph_tagged_batched(body_graph, args, max_iter)
    return body_values[body_graph.root_id]


# --- Non-tagged batched evaluation (bare tensors, PyTorch broadcasting) ---

def _eval_graph_batched(
    graph: ComputeGraph,
    inputs: dict[str, torch.Tensor],
    max_iter: int,
) -> dict[int, torch.Tensor]:
    """Topological evaluation with batched bare tensors."""
    values: dict[int, torch.Tensor] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            val = node.value.code_point if isinstance(node.value, SchemeChar) else node.value
            values[nid] = torch.tensor(float(val), dtype=torch.float32)

        elif node.op_type in ("input", "loop_param", "func_param"):
            values[nid] = inputs[node.name]

        elif node.op_type == "if":
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op("if", arg_tensors)

        elif node.op_type == "recur":
            values[nid] = torch.tensor(0.0)

        elif node.op_type == "loop":
            values[nid] = _eval_loop_batched(node, graph, values, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_batched(node, graph, values, max_iter)

        elif node.op_type == "soft_choice":
            num_opts = node.value
            opt_vals = [values[node.input_edges[i]] for i in range(num_opts)]
            logits = values[node.input_edges[num_opts]]
            values[nid] = _soft_choice_combine(opt_vals, logits)

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op(node.op_type, arg_tensors)

    return values


def _broadcast_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Reshape a per-element ``(B,)`` boolean mask to broadcast against ``value``'s trailing
    feature dims (e.g. ``(B,)`` -> ``(B, 1)`` for a ``(B, F)`` value)."""
    if value.dim() <= mask.dim():
        return mask
    return mask.reshape(mask.shape + (1,) * (value.dim() - mask.dim()))


def _eval_loop_batched(
    loop_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Batched loop for non-tagged graphs.

    When the loop body is the common ``(if test (recur ...) result)`` shape, this performs
    *masked padded iteration*: each element of the batch terminates on its own iteration (when
    its branch decision flips to the terminal branch), its result is frozen via ``torch.where``
    (autograd-safe), and still-active elements keep iterating until every element has
    terminated. This makes data-dependent loop bounds correct per batch element. Other body
    shapes (unconditional ``recur`` or a non-``if`` body) fall back to the uniform path.
    """
    loop_body = outer_graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params: dict[str, torch.Tensor] = {}
    for i, param in enumerate(params):
        current_params[param] = outer_values[loop_node.input_edges[i]]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

    body = loop_body.body_graph
    root_nid = body.root_id
    root_node = body.nodes[root_nid]

    # Identify a single top-level (if test recur/result) so we can mask per element.
    masked = None  # (recur_node, recur_is_then, terminal_nid) or None
    if root_node.op_type == "if":
        then_node = body.nodes[root_node.input_edges[1]]
        else_node = body.nodes[root_node.input_edges[2]]
        if then_node.op_type == "recur":
            masked = (then_node, True, root_node.input_edges[2])
        elif else_node.op_type == "recur":
            masked = (else_node, False, root_node.input_edges[1])

    done: torch.Tensor | None = None
    result: torch.Tensor | None = None
    for _ in range(max_iter):
        body_values = _eval_graph_batched(body, current_params, max_iter)

        if root_node.op_type == "recur":  # unconditional recur (uniform)
            for i, param in enumerate(params):
                current_params[param] = body_values[root_node.input_edges[i]]
            continue
        if masked is None:  # non-if body, or nested-if recur: uniform fallback
            if root_node.op_type == "if":
                if _check_if_recurs_batched(root_nid, body, body_values, params, current_params):
                    continue
            return body_values[root_nid]

        recur_node, recur_is_then, terminal_nid = masked
        test = body_values[root_node.input_edges[0]]
        terminal_value = body_values[terminal_nid]
        recur_args = {p: body_values[recur_node.input_edges[i]] for i, p in enumerate(params)}

        recurs = (test != 0.0) if recur_is_then else (test == 0.0)  # (B,) bool (or scalar)
        if recurs.dim() == 0:  # uniform scalar test: no per-element divergence
            if bool(recurs):
                for p in params:
                    current_params[p] = recur_args[p]
                continue
            return terminal_value

        if done is None:
            done = torch.zeros_like(recurs, dtype=torch.bool)
            result = torch.zeros_like(terminal_value)
        newly_term = (~recurs) & (~done)
        result = torch.where(_broadcast_mask(newly_term, result), terminal_value, result)
        done = done | (~recurs)
        if bool(done.all()):
            return result
        for p in params:
            keep = _broadcast_mask(done, current_params[p])
            current_params[p] = torch.where(keep, current_params[p], recur_args[p])

    if done is not None:  # ran out of iterations: return what terminated, leave the rest as-is
        return result
    raise RuntimeError(f"Batched loop did not terminate after {max_iter} iterations")


def _check_if_recurs_batched(
    if_nid: int,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    params: tuple[str, ...],
    current_params: dict[str, torch.Tensor],
) -> bool:
    """Check if an if-node's taken branch is a recur (non-tagged batched)."""
    node = graph.nodes[if_nid]
    test_val = values[node.input_edges[0]]
    if test_val.dim() == 0:
        taken = test_val.item() != 0.0
    else:
        taken = test_val.flatten()[0].item() != 0.0

    branch_nid = node.input_edges[1] if taken else node.input_edges[2]
    branch_node = graph.nodes[branch_nid]

    if branch_node.op_type == "recur":
        for i, param in enumerate(params):
            arg_nid = branch_node.input_edges[i]
            current_params[param] = values[arg_nid]
        return True
    return False


def _eval_call_batched(
    call_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Batched function call for non-tagged graphs."""
    func_name = call_node.call_target
    if func_name not in outer_graph.functions:
        raise ValueError(f"Batched eval: unknown function '{func_name}'")
    func_body = outer_graph.functions[func_name]
    body_graph = func_body.body_graph

    args: dict[str, torch.Tensor] = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = outer_values[arg_nid]

    body_values = _eval_graph_batched(body_graph, args, max_iter)
    return body_values[body_graph.root_id]
