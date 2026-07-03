############################################################
# DMCI tuned-eager payload-only value representation.
#
# The MLSys thesis gate (see paper-ndvm/MLSYS_REMEDIATION_PLAN.md): a competent eager encoding in which the
# type TAG is a native Python int and a tensor is created ONLY for a numeric, gradient-carrying payload.
# Structural values (nil, bool, symbol, char, pair addresses, closure refs, string/vector refs) carry plain
# Python numbers and allocate NO tensor at all. This is the strongest-fair eager baseline: it removes the
# per-value [14]-float tagged-tensor boxing that dominates the current backend (61-66% of forward time)
# WITHOUT the native C++ runtime, so it tests whether the speedup the paper attributes to the representation
# survives a tuned eager encoding.
#
# Interface mirrors neural_compiler.runtime.tagged_value so the two can be swapped op-for-op in a faithful
# boxing replay. Tags use the SAME integer codes as tagged_value.
############################################################
from __future__ import annotations

import torch
from torch import Tensor

# Same tag codes as tagged_value (one-hot positions there; native ints here).
NIL, BOOL, INT, FLOAT, CHAR, SYMBOL, PAIR, STRING, CLOSURE, VECTOR = range(10)
TYPE_NAMES = ["nil", "bool", "int", "float", "char", "symbol", "pair", "string", "closure", "vector"]

# Layout constants kept only for drop-in import compatibility with the tagged backend (the engine uses them
# for input-shape checks). A payload-only value is a PV object, not a [14]-float tensor, so these do not
# describe its storage; the engine's `shape[-1] == VALUE_DIM` checks are replaced by `isinstance(v, PV)`.
TAG_DIM = 10
PAYLOAD_DIM = 4
VALUE_DIM = TAG_DIM + PAYLOAD_DIM  # 14


class _PS(float):
    """A structural-payload scalar: a Python float that also answers ``.item()`` and ``.numel()``, so the
    tagged engine's tensor-style payload access (e.g. ``unwrap_symbol_id(x).item()``) works on a structural
    value with NO tensor allocated. Behaves as a float everywhere else (arithmetic, comparison, int())."""
    __slots__ = ()

    def item(self):
        return float(self)

    def numel(self):
        return 1


class PV:
    """A payload-only tagged value: a native int tag plus up to two payload slots.

    For numeric values (INT, FLOAT) slot ``a`` is a torch scalar tensor, so autograd flows through it
    exactly as it does through the numeric payload of a tagged tensor, but no [14]-float wrapper is built.
    For structural values the slots are ``_PS`` scalars and NO tensor is allocated."""
    __slots__ = ("tag", "a", "b")

    def __init__(self, tag: int, a=0.0, b=0.0):
        self.tag = tag
        self.a = a if isinstance(a, Tensor) else _PS(a)
        self.b = b if isinstance(b, Tensor) else _PS(b)


# --- constructors (the "boxing" the gate measures) -------------------------------------------------------
# Numeric constructors keep the payload as a tensor (gradient-carrying) but allocate no wrapper tensor.
# A bare Python number leaf is promoted to a 0-d tensor once, exactly as the tagged path's make_float does.

def make_nil(device=None) -> PV:
    return PV(NIL)


def make_bool(val, device=None) -> PV:
    if isinstance(val, Tensor):
        return PV(BOOL, (val != 0.0).float())
    return PV(BOOL, 1.0 if val else 0.0)


def make_int(val, device=None) -> PV:
    return PV(INT, val if isinstance(val, Tensor) else torch.tensor(float(val)))


def make_float(val, device=None) -> PV:
    return PV(FLOAT, val if isinstance(val, Tensor) else torch.tensor(float(val)))


def make_char(codepoint: int, device=None) -> PV:
    return PV(CHAR, float(codepoint))


def make_symbol(interned_id: int, device=None) -> PV:
    return PV(SYMBOL, float(interned_id))


def make_pair(car_addr, cdr_addr, device=None) -> PV:
    a = float(car_addr) if not isinstance(car_addr, Tensor) else car_addr
    b = float(cdr_addr) if not isinstance(cdr_addr, Tensor) else cdr_addr
    return PV(PAIR, a, b)


def make_string(heap_addr, length, device=None) -> PV:
    return PV(STRING, float(heap_addr), float(length))


def make_closure(func_id, env_addr, device=None) -> PV:
    a = func_id if isinstance(func_id, Tensor) else float(func_id)
    b = env_addr if isinstance(env_addr, Tensor) else float(env_addr)
    return PV(CLOSURE, a, b)


def make_vector(heap_addr, feature_ndim: float = 1.0, length: float = 0.0, device=None) -> PV:
    return PV(VECTOR, float(heap_addr), float(feature_ndim))


# --- extractors / predicates (native int compare, no tensor indexing) ------------------------------------

def type_index(pv: PV) -> int:
    return pv.tag


def type_name(pv: PV) -> str:
    return TYPE_NAMES[pv.tag]


def is_type(pv: PV, t: int) -> bool:
    return pv.tag == t


def extract_tag(pv: PV) -> int:
    """The native int tag (the tagged backend returns a one-hot tensor here; the exact tag is the int)."""
    return pv.tag


def extract_payload(pv: PV):
    """Positional payload access, compatible with the tagged backend's extract_payload(v)[i]. Returns a
    2-slot list of the native payloads (Python numbers for structural values, tensors for numeric)."""
    return [pv.a, pv.b]


def is_nil(pv: PV) -> bool: return pv.tag == NIL
def is_bool(pv: PV) -> bool: return pv.tag == BOOL
def is_number(pv: PV) -> bool: return pv.tag == INT or pv.tag == FLOAT
def is_char(pv: PV) -> bool: return pv.tag == CHAR
def is_symbol(pv: PV) -> bool: return pv.tag == SYMBOL
def is_pair(pv: PV) -> bool: return pv.tag == PAIR
def is_string(pv: PV) -> bool: return pv.tag == STRING
def is_closure(pv: PV) -> bool: return pv.tag == CLOSURE
def is_vector_type(pv: PV) -> bool: return pv.tag == VECTOR


def unwrap_number(pv: PV):
    return pv.a


def unwrap_bool(pv: PV):
    return pv.a


def unwrap_symbol_id(pv: PV) -> float:
    return pv.a


def unwrap_pair_addrs(pv: PV):
    return pv.a, pv.b


def unwrap_closure(pv: PV):
    return pv.a, pv.b


def from_scalar(val) -> PV:
    if isinstance(val, bool):
        return make_bool(val)
    if isinstance(val, int):
        return make_int(val)
    if isinstance(val, float):
        return make_float(val)
    if isinstance(val, Tensor):
        return make_bool(val.float()) if val.dtype == torch.bool else make_float(val)
    raise TypeError(f"cannot convert {type(val)}")


def to_scalar(pv: PV) -> float:
    return float(pv.a.item() if isinstance(pv.a, Tensor) else pv.a)


# --- exact dispatch (native branch; the one-hot soft_select becomes a hard switch) -----------------------

def select(pv: PV, branches: dict):
    """Exact type dispatch. The tagged backend writes this as a weighted sum over one-hot tag
    probabilities (soft_select); with exact tags that is a hard switch, which is what NDVM and this
    payload-only baseline both do, with no tensor allocated for the selector."""
    return branches[pv.tag]


def hard_if(test: PV, then_val, else_val):
    truth = test.a
    if isinstance(truth, Tensor):
        sel = (truth != 0.0).float()
        return sel * then_val + (1.0 - sel) * else_val
    return then_val if truth != 0.0 else else_val


# tagged backend name for the soft MUX; with exact tags it is the hard select above.
tagged_if = hard_if


def soft_select(pv: PV, branches: dict):
    return select(pv, branches)


# --- Tensor-payload inputs (Strategy B data ingestion), mirrored from tagged_value for drop-in use -------

from dataclasses import dataclass  # noqa: E402


@dataclass(eq=False)
class TensorInput:
    tensor: "Tensor"
    feature_ndim: int = 1


def as_vector(t: "Tensor") -> "TensorInput":
    return TensorInput(t, 1)


def as_matrix(t: "Tensor") -> "TensorInput":
    return TensorInput(t, 2)
