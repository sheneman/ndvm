############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# tagged_value.py: Tagged value representation for full Scheme: every value is a fixed-size tensor. Layout: [TAG_DIM one-hot type...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tagged value representation for full Scheme: every value is a fixed-size tensor.

Layout: [TAG_DIM one-hot type tag | PAYLOAD_DIM payload] = VALUE_DIM floats total.

Type tags (one-hot position):
  0=nil, 1=bool, 2=int, 3=float, 4=char, 5=symbol, 6=pair, 7=string, 8=closure, 9=vector

Payload interpretation depends on type:
  nil:     [0, 0, 0, 0]
  bool:    [0.0 or 1.0, 0, 0, 0]
  int:     [value, 0, 0, 0]
  float:   [value, 0, 0, 0]
  char:    [codepoint, 0, 0, 0]
  symbol:  [interned_id, 0, 0, 0]
  pair:    [car_addr, cdr_addr, 0, 0]
  string:  [heap_addr, length, 0, 0]
  closure: [func_id, env_addr, 0, 0]
  vector:  [heap_addr, length, 0, 0]
"""

from __future__ import annotations

import torch
from torch import Tensor
from dataclasses import dataclass

TAG_DIM = 10
PAYLOAD_DIM = 4
VALUE_DIM = TAG_DIM + PAYLOAD_DIM  # 14

NIL = 0
BOOL = 1
INT = 2
FLOAT = 3
CHAR = 4
SYMBOL = 5
PAIR = 6
STRING = 7
CLOSURE = 8
VECTOR = 9

TYPE_NAMES = [
    "nil", "bool", "int", "float", "char",
    "symbol", "pair", "string", "closure", "vector",
]


def _make(type_idx: int, payload: list[float], device: torch.device | None = None) -> Tensor:
    tag = [0.0] * TAG_DIM
    tag[type_idx] = 1.0
    pad = payload + [0.0] * (PAYLOAD_DIM - len(payload))
    return torch.tensor(tag + pad, dtype=torch.float32, device=device)


def make_nil(device: torch.device | None = None) -> Tensor:
    return _make(NIL, [], device=device)


def make_bool(val: bool | float | Tensor, device: torch.device | None = None) -> Tensor:
    if isinstance(val, Tensor):
        if val.dim() == 0:
            tag = torch.zeros(TAG_DIM, dtype=torch.float32, device=val.device)
            tag[BOOL] = 1.0
            payload = torch.zeros(PAYLOAD_DIM, dtype=torch.float32, device=val.device)
            payload[0] = (val != 0.0).float()
            return torch.cat([tag, payload])
        tag = torch.zeros(*val.shape, TAG_DIM, dtype=torch.float32, device=val.device)
        tag[..., BOOL] = 1.0
        payload = torch.zeros(*val.shape, PAYLOAD_DIM, dtype=torch.float32, device=val.device)
        payload[..., 0] = (val != 0.0).float()
        return torch.cat([tag, payload], dim=-1)
    v = 1.0 if val else 0.0
    return _make(BOOL, [v], device=device)


def make_int(val: int | float | Tensor, device: torch.device | None = None) -> Tensor:
    if isinstance(val, Tensor):
        if val.dim() == 0:
            tag = torch.zeros(TAG_DIM, dtype=torch.float32, device=val.device)
            tag[INT] = 1.0
            payload = torch.zeros(PAYLOAD_DIM, dtype=torch.float32, device=val.device)
            payload[0] = val.float()
            return torch.cat([tag, payload])
        tag = torch.zeros(*val.shape, TAG_DIM, dtype=torch.float32, device=val.device)
        tag[..., INT] = 1.0
        payload = torch.zeros(*val.shape, PAYLOAD_DIM, dtype=torch.float32, device=val.device)
        payload[..., 0] = val.float()
        return torch.cat([tag, payload], dim=-1)
    return _make(INT, [float(val)], device=device)


def make_float(val: float | Tensor, device: torch.device | None = None) -> Tensor:
    if isinstance(val, Tensor):
        if val.dim() == 0:
            tag = torch.zeros(TAG_DIM, dtype=torch.float32, device=val.device)
            tag[FLOAT] = 1.0
            payload = torch.zeros(PAYLOAD_DIM, dtype=torch.float32, device=val.device)
            payload[0] = val
            return torch.cat([tag, payload])
        tag = torch.zeros(*val.shape, TAG_DIM, dtype=torch.float32, device=val.device)
        tag[..., FLOAT] = 1.0
        payload = torch.zeros(*val.shape, PAYLOAD_DIM, dtype=torch.float32, device=val.device)
        payload[..., 0] = val
        return torch.cat([tag, payload], dim=-1)
    return _make(FLOAT, [float(val)], device=device)


def make_char(codepoint: int, device: torch.device | None = None) -> Tensor:
    return _make(CHAR, [float(codepoint)], device=device)


def make_symbol(interned_id: int, device: torch.device | None = None) -> Tensor:
    return _make(SYMBOL, [float(interned_id)], device=device)


def make_pair(car_addr: float | Tensor, cdr_addr: float | Tensor,
              device: torch.device | None = None) -> Tensor:
    if isinstance(car_addr, Tensor) or isinstance(cdr_addr, Tensor):
        dev = car_addr.device if isinstance(car_addr, Tensor) else cdr_addr.device
        tag = torch.zeros(TAG_DIM, dtype=torch.float32, device=dev)
        tag[PAIR] = 1.0
        payload = torch.zeros(PAYLOAD_DIM, dtype=torch.float32, device=dev)
        ca = car_addr if isinstance(car_addr, Tensor) else torch.tensor(car_addr, device=dev)
        cd = cdr_addr if isinstance(cdr_addr, Tensor) else torch.tensor(cdr_addr, device=dev)
        payload[0] = ca
        payload[1] = cd
        return torch.cat([tag, payload])
    return _make(PAIR, [float(car_addr), float(cdr_addr)], device=device)


def make_string(heap_addr: float, length: float,
                device: torch.device | None = None) -> Tensor:
    return _make(STRING, [heap_addr, length], device=device)


def make_closure(func_id: int | float | Tensor, env_addr: float | Tensor,
                 device: torch.device | None = None) -> Tensor:
    if isinstance(func_id, Tensor) or isinstance(env_addr, Tensor):
        dev = func_id.device if isinstance(func_id, Tensor) else env_addr.device
        tag = torch.zeros(TAG_DIM, dtype=torch.float32, device=dev)
        tag[CLOSURE] = 1.0
        payload = torch.zeros(PAYLOAD_DIM, dtype=torch.float32, device=dev)
        fid = func_id if isinstance(func_id, Tensor) else torch.tensor(float(func_id), device=dev)
        ea = env_addr if isinstance(env_addr, Tensor) else torch.tensor(float(env_addr), device=dev)
        payload[0] = fid
        payload[1] = ea
        return torch.cat([tag, payload])
    return _make(CLOSURE, [float(func_id), float(env_addr)], device=device)


def make_vector(heap_addr: float, feature_ndim: float = 1.0, length: float = 0.0,
                device: torch.device | None = None) -> Tensor:
    # Strategy B payload: [heap_addr, feature_ndim, length]. feature_ndim is the trailing
    # non-batch rank (1=vector, 2=matrix); leading dims of the stored tensor are batch.
    return _make(VECTOR, [heap_addr, feature_ndim, length], device=device)


# --- Extractors ---

def extract_tag(tv: Tensor) -> Tensor:
    return tv[..., :TAG_DIM]


def extract_payload(tv: Tensor) -> Tensor:
    return tv[..., TAG_DIM:]


def type_index(tv: Tensor) -> int:
    return int(tv[:TAG_DIM].argmax().item())


def type_name(tv: Tensor) -> str:
    return TYPE_NAMES[type_index(tv)]


def is_type(tv: Tensor, type_idx: int) -> Tensor:
    return tv[..., type_idx]


def is_nil(tv: Tensor) -> Tensor:
    return is_type(tv, NIL)


def is_bool(tv: Tensor) -> Tensor:
    return is_type(tv, BOOL)


def is_number(tv: Tensor) -> Tensor:
    return is_type(tv, INT) + is_type(tv, FLOAT)


def is_char(tv: Tensor) -> Tensor:
    return is_type(tv, CHAR)


def is_symbol(tv: Tensor) -> Tensor:
    return is_type(tv, SYMBOL)


def is_pair(tv: Tensor) -> Tensor:
    return is_type(tv, PAIR)


def is_string(tv: Tensor) -> Tensor:
    return is_type(tv, STRING)


def is_closure(tv: Tensor) -> Tensor:
    return is_type(tv, CLOSURE)


def is_vector_type(tv: Tensor) -> Tensor:
    return is_type(tv, VECTOR)


def unwrap_number(tv: Tensor) -> Tensor:
    return extract_payload(tv)[..., 0]


# --- Tensor-payload inputs (Strategy B data ingestion) ---

@dataclass(eq=False)
class TensorInput:
    """Marker for binding a raw tensor as a DMCI VECTOR/MATRIX payload INPUT.

    Without it, ``evaluate``/``evaluate_program`` run a raw tensor through ``make_float``,
    which reads it as a *batch of scalars* (shape ``[N, VALUE_DIM]``) -- wrong for a real
    ``[T, D]`` observation matrix or ``[T]`` forcing series. Wrapping it (``as_vector`` /
    ``as_matrix``) makes the binding code ``heap.store`` the tensor on the evaluation heap and
    bind a ``make_vector`` ref, so inside the program ``(ref obs k)`` gathers from it and
    gradients flow back to the tensor. ``feature_ndim`` is the trailing non-batch rank
    (1 = vector, 2 = matrix); leading dims are indexable/batch."""
    tensor: "Tensor"
    feature_ndim: int = 1


def as_vector(t: "Tensor") -> "TensorInput":
    """Bind ``t`` as a VECTOR payload input: ``(ref t k)`` gathers element/row ``k``."""
    return TensorInput(t, 1)


def as_matrix(t: "Tensor") -> "TensorInput":
    """Bind ``t`` as a MATRIX payload input: ``(ref t k)`` gathers row ``k`` as a vector."""
    return TensorInput(t, 2)


def unwrap_bool(tv: Tensor) -> Tensor:
    return extract_payload(tv)[..., 0]


def unwrap_char(tv: Tensor) -> Tensor:
    return extract_payload(tv)[..., 0]


def unwrap_symbol_id(tv: Tensor) -> Tensor:
    return extract_payload(tv)[..., 0]


def unwrap_pair_addrs(tv: Tensor) -> tuple[Tensor, Tensor]:
    p = extract_payload(tv)
    return p[..., 0], p[..., 1]


def unwrap_closure(tv: Tensor) -> tuple[Tensor, Tensor]:
    p = extract_payload(tv)
    return p[..., 0], p[..., 1]


# --- Soft type dispatch ---

def soft_select(tv: Tensor, branches: dict[int, Tensor]) -> Tensor:
    """Weighted sum of branch results based on type tag probabilities.

    Each branch value must have the same shape. The type tag weights
    select which branch contributes to the result. Fully differentiable.
    """
    tag = extract_tag(tv)
    result = torch.zeros_like(next(iter(branches.values())))
    for type_idx, val in branches.items():
        weight = tag[type_idx]
        while weight.dim() < val.dim():
            weight = weight.unsqueeze(-1)
        result = result + weight * val
    return result


def tagged_if(test: Tensor, then_val: Tensor, else_val: Tensor) -> Tensor:
    """Soft MUX on tagged values. test is a TaggedValue (bool or number).

    Extracts the numeric truth value and blends then_val/else_val.
    """
    truth = unwrap_number(test)
    sel = (truth != 0.0).float()
    while sel.dim() < then_val.dim():
        sel = sel.unsqueeze(-1)
    return sel * then_val + (1.0 - sel) * else_val


# --- Conversion from bare tensors ---

def from_scalar(val: float | int | bool | Tensor) -> Tensor:
    """Convert a Python/torch scalar to a tagged value (auto-detect type)."""
    if isinstance(val, bool):
        return make_bool(val)
    if isinstance(val, int):
        return make_int(val)
    if isinstance(val, float):
        return make_float(val)
    if isinstance(val, Tensor):
        if val.dtype == torch.bool:
            return make_bool(val.float())
        return make_float(val)
    raise TypeError(f"Cannot convert {type(val)} to TaggedValue")


def to_scalar(tv: Tensor) -> float:
    """Extract scalar value from a numeric tagged value."""
    return unwrap_number(tv).item()
