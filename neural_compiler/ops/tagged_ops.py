############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# tagged_ops.py: Tagged-value primitive operations for full Scheme. These operations work on TaggedValue tensors (shape...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tagged-value primitive operations for full Scheme.

These operations work on TaggedValue tensors (shape [VALUE_DIM]) and
interact with the TensorHeap for cons cells.
"""

from __future__ import annotations
import torch
from torch import Tensor

from neural_compiler.runtime.tagged_value import (
    VALUE_DIM, TAG_DIM,
    NIL, BOOL, INT, FLOAT, CHAR, SYMBOL, PAIR, STRING, CLOSURE, VECTOR,
    make_nil, make_bool, make_int, make_float, make_pair, make_symbol, make_vector,
    extract_tag, extract_payload, type_index, unwrap_number, unwrap_bool,
    unwrap_pair_addrs, unwrap_symbol_id,
    is_nil, is_pair, is_number, is_symbol, is_closure, is_bool,
    is_char, is_string, is_vector_type,
    tagged_if, from_scalar,
)
from neural_compiler.runtime.heap import TensorHeap


def _op_cons(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return heap.cons(args[0], args[1])


def _op_car(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return heap.car(args[0])


def _op_cdr(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return heap.cdr(args[0])


def _op_null_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_nil(args[0]).item() > 0.5)


def _op_pair_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_pair(args[0]).item() > 0.5)


def _op_number_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_number(args[0]).item() > 0.5)


def _op_boolean_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_bool(args[0]).item() > 0.5)


def _op_symbol_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_symbol(args[0]).item() > 0.5)


def _op_char_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_char(args[0]).item() > 0.5)


def _op_procedure_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_closure(args[0]).item() > 0.5)


def _op_string_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_string(args[0]).item() > 0.5)


def _op_vector_p(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return make_bool(is_vector_type(args[0]).item() > 0.5)


def _op_eq(args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Identity comparison. For symbols: compare interned IDs. For numbers: value equality."""
    a, b = args
    ta, tb = type_index(a), type_index(b)
    if ta != tb:
        return make_bool(False)
    if ta == NIL:
        return make_bool(True)
    if ta == BOOL:
        return make_bool(unwrap_bool(a).item() == unwrap_bool(b).item())
    if ta in (INT, FLOAT, CHAR):
        return make_bool(unwrap_number(a).item() == unwrap_number(b).item())
    if ta == SYMBOL:
        return make_bool(unwrap_symbol_id(a).item() == unwrap_symbol_id(b).item())
    if ta == PAIR:
        ca, da = unwrap_pair_addrs(a)
        cb, db = unwrap_pair_addrs(b)
        return make_bool(ca.item() == cb.item() and da.item() == db.item())
    return make_bool(False)


def _op_eqv(args: list[Tensor], heap: TensorHeap) -> Tensor:
    return _op_eq(args, heap)


def _op_equal(args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Structural equality — recursive comparison through pairs."""
    a, b = args
    return _deep_equal(a, b, heap)


def _deep_equal(a: Tensor, b: Tensor, heap: TensorHeap) -> Tensor:
    ta, tb = type_index(a), type_index(b)
    if ta != tb:
        return make_bool(False)
    if ta == NIL:
        return make_bool(True)
    if ta in (BOOL, INT, FLOAT, SYMBOL, CHAR):
        return make_bool(unwrap_number(a).item() == unwrap_number(b).item())
    if ta == PAIR:
        car_eq = _deep_equal(heap.car(a), heap.car(b), heap)
        if unwrap_bool(car_eq).item() == 0.0:
            return make_bool(False)
        return _deep_equal(heap.cdr(a), heap.cdr(b), heap)
    return make_bool(False)


def _op_list(args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Build a proper list from arguments."""
    return heap.build_list(args)


def _op_length(args: list[Tensor], heap: TensorHeap) -> Tensor:
    lst = args[0]
    count = 0
    while is_pair(lst).item() > 0.5:
        count += 1
        lst = heap.cdr(lst)
    return make_float(float(count))


def _op_append(args: list[Tensor], heap: TensorHeap) -> Tensor:
    if len(args) == 0:
        return make_nil()
    if len(args) == 1:
        return args[0]
    result = args[-1]
    for lst in reversed(args[:-1]):
        elements = []
        cur = lst
        while is_pair(cur).item() > 0.5:
            elements.append(heap.car(cur))
            cur = heap.cdr(cur)
        for elem in reversed(elements):
            result = heap.cons(elem, result)
    return result


def _op_reverse(args: list[Tensor], heap: TensorHeap) -> Tensor:
    result = make_nil()
    cur = args[0]
    while is_pair(cur).item() > 0.5:
        result = heap.cons(heap.car(cur), result)
        cur = heap.cdr(cur)
    return result


# --- Numeric ops wrapped for tagged values ---

def _tagged_arith(op_name: str, args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Wrap a numeric primitive for tagged values.

    Scalar args -> scalar (FLOAT) result, as before. But if ANY arg is a VECTOR-tagged
    tensor payload, do ELEMENTWISE tensor arithmetic via torch broadcasting and return a
    VECTOR ref. This makes +, -, *, /, min, max, exp, ... work on vectors/matrices under
    DMCI (e.g. P F^T + Q in a Kalman filter, A C + b in a compartmental model) and matches
    the direct-compilation path, where these primitives already broadcast over tensors --
    restoring direct == DMCI for tensor arithmetic. Scalars broadcast against tensors; a
    batched scalar [N] gets trailing singleton feature dims so it broadcasts over [*, feat]."""
    from neural_compiler.ops.primitives import evaluate_op
    vec_flags = [bool(is_vector_type(a).reshape(-1)[0].item() > 0.5) for a in args]
    if any(vec_flags):
        out_ndim = max(_feature_ndim(a) for a, f in zip(args, vec_flags) if f)
        raw = []
        for a, f in zip(args, vec_flags):
            if f:
                raw.append(_unwrap_tensor(a, heap))
            else:
                s = unwrap_number(a)
                if s.dim() > 0:  # batched scalar [N] -> [N, 1(, 1)] to broadcast over feature dims
                    s = s.reshape(*s.shape, *([1] * out_ndim))
                raw.append(s)
        return _wrap_tensor(evaluate_op(op_name, raw), heap, out_ndim)
    raw_args = [unwrap_number(a) for a in args]
    result = evaluate_op(op_name, raw_args)
    return make_float(result)


def _tagged_compare(op_name: str, args: list[Tensor], heap: TensorHeap) -> Tensor:
    from neural_compiler.ops.primitives import evaluate_op
    raw_args = [unwrap_number(a) for a in args]
    result = evaluate_op(op_name, raw_args)
    return make_bool(result)


TAGGED_OP_TABLE: dict[str, callable] = {
    "cons": _op_cons,
    "car": _op_car,
    "cdr": _op_cdr,
    "list": _op_list,
    "length": _op_length,
    "append": _op_append,
    "reverse": _op_reverse,
    "null?": _op_null_p,
    "pair?": _op_pair_p,
    "number?": _op_number_p,
    "boolean?": _op_boolean_p,
    "symbol?": _op_symbol_p,
    "char?": _op_char_p,
    "procedure?": _op_procedure_p,
    "string?": _op_string_p,
    "vector?": _op_vector_p,
    "eq?": _op_eq,
    "eqv?": _op_eqv,
    "equal?": _op_equal,
}

ARITH_OPS = {"+", "-", "*", "/", "pow", "abs", "min", "max", "modulo", "remainder",
             "sin", "cos", "exp", "sqrt", "log"}
COMPARE_OPS = {"=", "<", ">", "<=", ">="}
LOGIC_OPS = {"not", "and", "or"}


# --- Strategy B: tensor-payload vector/matrix ops (v2, batched) ---
# A vector/matrix is a raw torch tensor stored in ONE heap slot; the tagged value is a
# VECTOR(9) ref carrying [heap_addr, feature_ndim, length]. The stored tensor is laid out
# [*batch, *feature]: LEADING dims are batch, the TRAILING feature_ndim dims (1=vector,
# 2=matrix) are the payload. The primitives use dim=-1/-2 conventions, so they operate on
# the feature dims and broadcast over batch for free. A batched scalar is a FLOAT-tagged
# [N, VALUE_DIM] value that unwraps to [N]; batch vs feature is disambiguated by the TAG
# (VECTOR vs FLOAT) and the ref's feature_ndim, never by shape -- so coincidental N==n is safe.
VEC_OPS = {
    "vec", "mat", "ref", "dot", "cross", "norm", "normalize", "vsum", "vlen",
    "scale", "matvec", "matmul", "transpose", "trace", "det", "logdet", "inv", "outer",
    "eye", "zeros", "ones",
}
# Output feature rank per op: 0 -> scalar (make_float, possibly batched [N]); 1 -> vector ref;
# 2 -> matrix ref. `ref` is special-cased (output ndim = input feature_ndim - 1).
_VEC_OUT_NDIM = {
    "dot": 0, "norm": 0, "vsum": 0, "vlen": 0, "trace": 0, "det": 0, "logdet": 0,
    "vec": 1, "cross": 1, "normalize": 1, "scale": 1, "matvec": 1, "zeros": 1, "ones": 1,
    "mat": 2, "matmul": 2, "transpose": 2, "outer": 2, "inv": 2, "eye": 2,
}


def _wrap_tensor(t: Tensor, heap: TensorHeap, feature_ndim: int) -> Tensor:
    """Raw torch tensor -> VECTOR-tagged ref. feature_ndim = trailing non-batch rank
    (1=vector, 2=matrix); leading dims are batch and not recorded (read from the tensor)."""
    addr = heap.store(t)
    length = float(t.shape[-1]) if t.dim() > 0 else 0.0
    return make_vector(float(addr), float(feature_ndim), length, device=heap.device)


def _unwrap_tensor(tv: Tensor, heap: TensorHeap) -> Tensor:
    """VECTOR ref -> the raw stored tensor (identity preserved for autograd)."""
    return heap.read(extract_payload(tv)[..., 0])


def _feature_ndim(tv: Tensor) -> int:
    """Trailing feature rank recorded in a VECTOR ref (payload slot 1)."""
    return int(extract_payload(tv)[..., 1].reshape(-1)[0].item())


def _heap_list_elems(lst: Tensor, heap: TensorHeap) -> list[Tensor]:
    """Collect the element tagged values of a heap cons-list (for vec/mat constructors)."""
    out: list[Tensor] = []
    cur = lst
    while bool(is_pair(cur).reshape(-1)[0].item() > 0.5):
        out.append(heap.car(cur))
        cur = heap.cdr(cur)
    return out


def _tagged_vec(op_name: str, args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Dispatch a vector/matrix op on tensor-payload tagged values, reusing primitives.py.
    Batched: leading dims of a stored tensor are batch; a FLOAT scalar may be batched [N]."""
    from neural_compiler.ops.primitives import evaluate_op

    # --- constructors: broadcast batched/unbatched elements, then stack on the feature axis ---
    if op_name == "vec":  # heap list of scalar tagged values -> [*batch, n]
        elems = list(torch.broadcast_tensors(*[unwrap_number(e)
                                               for e in _heap_list_elems(args[0], heap)]))
        return _wrap_tensor(torch.stack(elems, dim=-1), heap, 1)
    if op_name == "mat":  # heap list of VECTOR row refs -> [*batch, m, n]
        rows = list(torch.broadcast_tensors(*[_unwrap_tensor(e, heap)
                                              for e in _heap_list_elems(args[0], heap)]))
        return _wrap_tensor(torch.stack(rows, dim=-2), heap, 2)

    # --- ref: dispatched by the input's feature rank (NOT by tensor.dim(), which is ambiguous) ---
    if op_name == "ref":
        in_ndim = _feature_ndim(args[0])
        vec = _unwrap_tensor(args[0], heap)
        idx = unwrap_number(args[1])
        if idx.dim() != 0:  # per-batch (gather) index deferred
            raise ValueError("ref index must be a scalar (data-independent); per-batch "
                             "ref indices are deferred in this version.")
        i = int(idx.item())
        if in_ndim == 1:                       # vector element -> scalar
            return make_float(vec[..., i])
        if in_ndim == 2:                       # matrix row -> vector
            return _wrap_tensor(vec[..., i, :], heap, 1)
        raise ValueError(f"ref on unsupported feature_ndim={in_ndim}")

    # --- general ops: unwrap (tensor for VECTOR refs, scalar for FLOATs) ---
    raw, feats = [], []
    for a in args:
        if bool(is_vector_type(a).reshape(-1)[0].item() > 0.5):
            raw.append(_unwrap_tensor(a, heap)); feats.append(_feature_ndim(a))
        else:
            raw.append(unwrap_number(a)); feats.append(0)

    if op_name == "scale":  # (scalar, vector|matrix) in either order: align the scalar's
        si = 0 if feats[0] == 0 else 1        # batch axis against the feature tensor's trailing dims
        fnd = feats[1 - si]
        if raw[si].dim() > 0:                  # batched scalar [N] -> [N, 1(, 1)]
            raw[si] = raw[si].reshape(*raw[si].shape, *([1] * fnd))
        return _wrap_tensor(evaluate_op("scale", [raw[0], raw[1]]), heap, fnd)

    result = evaluate_op(op_name, raw)
    out_ndim = _VEC_OUT_NDIM[op_name]
    if out_ndim == 0:
        # batch-aware rank guard: a scalar result is 0-d (unbatched) or [N] (batched); a
        # trailing feature axis (dim>=2) means a real rank error -> fail loudly, do not let
        # make_float silently broadcast the feature axis as a batch.
        if result.dim() >= 2:
            raise ValueError(f"vector op '{op_name}' expected a scalar/batched-scalar result "
                             f"but got shape {tuple(result.shape)} (rank error).")
        return make_float(result)
    return _wrap_tensor(result, heap, out_ndim)


def evaluate_tagged_op(op_type: str, args: list[Tensor], heap: TensorHeap) -> Tensor:
    """Evaluate a primitive operation on tagged values."""
    if op_type in TAGGED_OP_TABLE:
        return TAGGED_OP_TABLE[op_type](args, heap)
    if op_type in ARITH_OPS:
        return _tagged_arith(op_type, args, heap)
    if op_type in COMPARE_OPS:
        return _tagged_compare(op_type, args, heap)
    if op_type in LOGIC_OPS:
        return _tagged_logic(op_type, args, heap)
    if op_type in VEC_OPS:
        return _tagged_vec(op_type, args, heap)
    raise ValueError(f"Unknown tagged operation: {op_type}")


def _tagged_logic(op_name: str, args: list[Tensor], heap: TensorHeap) -> Tensor:
    from neural_compiler.ops.primitives import evaluate_op
    raw_args = [unwrap_number(a) for a in args]
    result = evaluate_op(op_name, raw_args)
    return make_bool(result)


def materialize_quote(datum: object, heap: TensorHeap) -> Tensor:
    """Convert a raw quoted datum (from parser) into a TaggedValue on the heap.

    Atoms become tagged scalars. Lists become cons-cell chains on the heap.
    """
    if datum is None or (isinstance(datum, list) and len(datum) == 0):
        return make_nil()

    if isinstance(datum, str):
        try:
            return make_int(int(datum))
        except ValueError:
            pass
        try:
            return make_float(float(datum))
        except ValueError:
            pass
        if datum == "#t":
            return make_bool(True)
        if datum == "#f":
            return make_bool(False)
        # Treat as symbol
        from neural_compiler.runtime.symbols import SymbolTable
        if not hasattr(materialize_quote, "_symtab"):
            materialize_quote._symtab = SymbolTable()
        sym_id = materialize_quote._symtab.intern(datum)
        return make_symbol(sym_id)

    if isinstance(datum, (int, float)):
        return make_float(float(datum))

    if isinstance(datum, bool):
        return make_bool(datum)

    if isinstance(datum, list):
        elements = [materialize_quote(item, heap) for item in datum]
        return heap.build_list(elements)

    raise TypeError(f"Cannot quote datum of type {type(datum)}: {datum}")
