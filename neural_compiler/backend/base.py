############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# base.py: NumPy-family backend base class. Provides tagged value construction, heap management, and primitive ops using a...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""NumPy-family backend base class.

Provides tagged value construction, heap management, and primitive ops
using a generic array module (numpy, jax.numpy, cupy). Concrete backends
subclass and set ``self.xp``.
"""

from __future__ import annotations

from types import ModuleType
from neural_compiler.runtime.symbols import SymbolTable

TAG_DIM = 10
PAYLOAD_DIM = 4
VALUE_DIM = TAG_DIM + PAYLOAD_DIM

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


class GenericHeap:
    """Array-backed heap for cons cells. Backend-agnostic."""

    def __init__(self, backend: NumpyFamilyBackend, max_size: int = 65536):
        self.backend = backend
        self.xp = backend.xp
        self.max_size = max_size
        self.storage = self.xp.zeros((max_size, VALUE_DIM), dtype=self.xp.float32)
        self.alloc_ptr = 0

    def cons(self, car, cdr):
        if self.alloc_ptr + 2 > self.max_size:
            raise RuntimeError(f"Heap overflow at {self.alloc_ptr}, max={self.max_size}")
        car_addr = self.alloc_ptr
        cdr_addr = self.alloc_ptr + 1
        self.alloc_ptr += 2
        self.storage = self.backend.heap_set(self.storage, car_addr, car)
        self.storage = self.backend.heap_set(self.storage, cdr_addr, cdr)
        return self.backend.make_pair(float(car_addr), float(cdr_addr))

    def read(self, addr):
        if hasattr(addr, "item"):
            idx = int(addr.item())
        else:
            idx = int(addr)
        if idx < 0 or idx >= self.alloc_ptr:
            raise IndexError(f"Heap read OOB: addr={idx}, allocated={self.alloc_ptr}")
        return self.storage[idx]

    def write(self, addr, val):
        if hasattr(addr, "item"):
            idx = int(addr.item())
        else:
            idx = int(addr)
        self.storage = self.backend.heap_set(self.storage, idx, val)

    def car(self, pair_val):
        p = self.backend.extract_payload(pair_val)
        return self.read(p[0])

    def cdr(self, pair_val):
        p = self.backend.extract_payload(pair_val)
        return self.read(p[1])

    def build_list(self, elements):
        result = self.backend.make_nil()
        for elem in reversed(elements):
            result = self.cons(elem, result)
        return result

    def allocated(self) -> int:
        return self.alloc_ptr


class NumpyFamilyBackend:
    """Base backend for numpy-compatible array libraries."""

    name: str = "numpy_family"
    supports_autograd: bool = False
    xp: ModuleType

    def __init__(self, xp: ModuleType):
        self.xp = xp
        self._symtab = SymbolTable()

    # ------------------------------------------------------------------ #
    # Array helpers
    # ------------------------------------------------------------------ #

    def heap_set(self, storage, idx: int, val):
        storage[idx] = val
        return storage

    def item(self, arr) -> float:
        if hasattr(arr, "item"):
            return float(arr.item())
        return float(arr)

    # ------------------------------------------------------------------ #
    # Tagged value constructors
    # ------------------------------------------------------------------ #

    def _make(self, type_idx: int, payload: list[float]):
        tag = [0.0] * TAG_DIM
        tag[type_idx] = 1.0
        pad = payload + [0.0] * (PAYLOAD_DIM - len(payload))
        return self.xp.array(tag + pad, dtype=self.xp.float32)

    def make_nil(self):
        return self._make(NIL, [])

    def make_bool(self, val):
        v = 1.0 if val else 0.0
        return self._make(BOOL, [v])

    def make_float(self, val):
        return self._make(FLOAT, [float(val)])

    def make_int(self, val):
        return self._make(INT, [float(val)])

    def make_char(self, codepoint: int):
        return self._make(CHAR, [float(codepoint)])

    def make_symbol(self, interned_id: int):
        return self._make(SYMBOL, [float(interned_id)])

    def make_pair(self, car_addr: float, cdr_addr: float):
        return self._make(PAIR, [car_addr, cdr_addr])

    def make_closure(self, func_id, env_addr):
        return self._make(CLOSURE, [float(func_id), float(env_addr)])

    # ------------------------------------------------------------------ #
    # Tagged value accessors
    # ------------------------------------------------------------------ #

    def extract_tag(self, tv):
        return tv[:TAG_DIM]

    def extract_payload(self, tv):
        return tv[TAG_DIM:]

    def type_index(self, tv) -> int:
        return int(self.xp.argmax(tv[:TAG_DIM]))

    def unwrap_number(self, tv) -> float:
        return float(tv[TAG_DIM])

    def unwrap_bool(self, tv) -> float:
        return float(tv[TAG_DIM])

    def unwrap_closure(self, tv):
        return float(tv[TAG_DIM]), float(tv[TAG_DIM + 1])

    def unwrap_pair_addrs(self, tv):
        return float(tv[TAG_DIM]), float(tv[TAG_DIM + 1])

    def unwrap_symbol_id(self, tv) -> float:
        return float(tv[TAG_DIM])

    def is_nil(self, tv) -> bool:
        return float(tv[NIL]) > 0.5

    def is_pair(self, tv) -> bool:
        return float(tv[PAIR]) > 0.5

    def is_number(self, tv) -> bool:
        return float(tv[INT]) + float(tv[FLOAT]) > 0.5

    def is_symbol(self, tv) -> bool:
        return float(tv[SYMBOL]) > 0.5

    def is_closure(self, tv) -> bool:
        return float(tv[CLOSURE]) > 0.5

    def is_bool_type(self, tv) -> bool:
        return float(tv[BOOL]) > 0.5

    def is_char(self, tv) -> bool:
        return float(tv[CHAR]) > 0.5

    def is_string(self, tv) -> bool:
        return float(tv[STRING]) > 0.5

    def is_vector_type(self, tv) -> bool:
        return float(tv[VECTOR]) > 0.5

    def from_scalar(self, val):
        if isinstance(val, bool):
            return self.make_bool(val)
        if isinstance(val, int):
            return self.make_int(val)
        if isinstance(val, float):
            return self.make_float(val)
        raise TypeError(f"Cannot convert {type(val)} to TaggedValue")

    def tagged_if(self, test, then_val, else_val):
        truth = self.unwrap_number(test)
        if truth != 0.0:
            return then_val
        return else_val

    # ------------------------------------------------------------------ #
    # Heap
    # ------------------------------------------------------------------ #

    def create_heap(self, max_size: int = 65536) -> GenericHeap:
        return GenericHeap(self, max_size)

    # ------------------------------------------------------------------ #
    # Scalar primitive ops (bare float arrays)
    # ------------------------------------------------------------------ #

    def evaluate_op(self, op_name: str, args: list):
        xp = self.xp
        if op_name == "+":
            return args[0] + args[1]
        if op_name == "-":
            return -args[0] if len(args) == 1 else args[0] - args[1]
        if op_name == "*":
            return args[0] * args[1]
        if op_name == "/":
            return args[0] / args[1]
        if op_name == "=":
            return xp.float32(1.0 if float(args[0]) == float(args[1]) else 0.0)
        if op_name == "<":
            return xp.float32(1.0 if float(args[0]) < float(args[1]) else 0.0)
        if op_name == ">":
            return xp.float32(1.0 if float(args[0]) > float(args[1]) else 0.0)
        if op_name == "<=":
            return xp.float32(1.0 if float(args[0]) <= float(args[1]) else 0.0)
        if op_name == ">=":
            return xp.float32(1.0 if float(args[0]) >= float(args[1]) else 0.0)
        if op_name == "not":
            return xp.float32(1.0 if float(args[0]) == 0.0 else 0.0)
        if op_name == "and":
            return xp.float32(1.0 if float(args[0]) != 0.0 and float(args[1]) != 0.0 else 0.0)
        if op_name == "or":
            return xp.float32(1.0 if float(args[0]) != 0.0 or float(args[1]) != 0.0 else 0.0)
        if op_name == "modulo":
            return xp.float32(float(args[0]) % float(args[1]))
        if op_name == "remainder":
            import math
            return xp.float32(math.remainder(float(args[0]), float(args[1])))
        if op_name == "abs":
            return xp.abs(args[0])
        if op_name == "min":
            return xp.minimum(args[0], args[1])
        if op_name == "max":
            return xp.maximum(args[0], args[1])
        if op_name == "sin":
            return xp.sin(args[0])
        if op_name == "cos":
            return xp.cos(args[0])
        if op_name == "exp":
            return xp.exp(args[0])
        if op_name == "sqrt":
            return xp.sqrt(xp.maximum(args[0], xp.float32(1e-8)))
        if op_name == "log":
            return xp.log(xp.maximum(args[0], xp.float32(1e-8)))
        if op_name == "pow":
            return args[0] ** args[1]
        if op_name == "if":
            test, then_val, else_val = args
            return then_val if float(test) != 0.0 else else_val
        raise ValueError(f"Unknown scalar op: {op_name}")

    # ------------------------------------------------------------------ #
    # Tagged primitive ops
    # ------------------------------------------------------------------ #

    def evaluate_tagged_op(self, op_name: str, args: list, heap: GenericHeap):
        if op_name == "cons":
            return heap.cons(args[0], args[1])
        if op_name == "car":
            return heap.car(args[0])
        if op_name == "cdr":
            return heap.cdr(args[0])
        if op_name == "list":
            return heap.build_list(args)
        if op_name == "null?":
            return self.make_bool(self.is_nil(args[0]))
        if op_name == "pair?":
            return self.make_bool(self.is_pair(args[0]))
        if op_name == "number?":
            return self.make_bool(self.is_number(args[0]))
        if op_name == "boolean?":
            return self.make_bool(self.is_bool_type(args[0]))
        if op_name == "symbol?":
            return self.make_bool(self.is_symbol(args[0]))
        if op_name == "char?":
            return self.make_bool(self.is_char(args[0]))
        if op_name == "procedure?":
            return self.make_bool(self.is_closure(args[0]))
        if op_name == "string?":
            return self.make_bool(self.is_string(args[0]))
        if op_name == "vector?":
            return self.make_bool(self.is_vector_type(args[0]))
        if op_name == "eq?":
            return self._op_eq(args[0], args[1], heap)
        if op_name == "eqv?":
            return self._op_eq(args[0], args[1], heap)
        if op_name == "equal?":
            return self._deep_equal(args[0], args[1], heap)
        if op_name == "length":
            return self._op_length(args[0], heap)
        if op_name == "append":
            return self._op_append(args, heap)
        if op_name == "reverse":
            return self._op_reverse(args[0], heap)
        if op_name == "not":
            v = self.unwrap_number(args[0])
            return self.make_bool(v == 0.0)

        if op_name in ("+", "-", "*", "/", "pow", "abs", "min", "max",
                       "modulo", "remainder", "sin", "cos", "exp", "sqrt", "log"):
            return self._tagged_arith(op_name, args)

        if op_name in ("=", "<", ">", "<=", ">="):
            return self._tagged_compare(op_name, args)

        if op_name in ("and", "or"):
            return self._tagged_logic(op_name, args)

        raise ValueError(f"Unknown tagged op: {op_name}")

    def _tagged_arith(self, op_name, args):
        raw = [self.xp.float32(self.unwrap_number(a)) for a in args]
        result = self.evaluate_op(op_name, raw)
        return self.make_float(float(result))

    def _tagged_compare(self, op_name, args):
        raw = [self.xp.float32(self.unwrap_number(a)) for a in args]
        result = self.evaluate_op(op_name, raw)
        return self.make_bool(float(result) != 0.0)

    def _tagged_logic(self, op_name, args):
        raw = [self.xp.float32(self.unwrap_number(a)) for a in args]
        result = self.evaluate_op(op_name, raw)
        return self.make_bool(float(result) != 0.0)

    def _op_eq(self, a, b, heap):
        ta, tb = self.type_index(a), self.type_index(b)
        if ta != tb:
            return self.make_bool(False)
        if ta == NIL:
            return self.make_bool(True)
        if ta == BOOL:
            return self.make_bool(self.unwrap_bool(a) == self.unwrap_bool(b))
        if ta in (INT, FLOAT, CHAR):
            return self.make_bool(self.unwrap_number(a) == self.unwrap_number(b))
        if ta == SYMBOL:
            return self.make_bool(self.unwrap_symbol_id(a) == self.unwrap_symbol_id(b))
        if ta == PAIR:
            ca, da = self.unwrap_pair_addrs(a)
            cb, db = self.unwrap_pair_addrs(b)
            return self.make_bool(ca == cb and da == db)
        return self.make_bool(False)

    def _deep_equal(self, a, b, heap):
        ta, tb = self.type_index(a), self.type_index(b)
        if ta != tb:
            return self.make_bool(False)
        if ta == NIL:
            return self.make_bool(True)
        if ta in (BOOL, INT, FLOAT, SYMBOL, CHAR):
            return self.make_bool(self.unwrap_number(a) == self.unwrap_number(b))
        if ta == PAIR:
            car_eq = self._deep_equal(heap.car(a), heap.car(b), heap)
            if self.unwrap_bool(car_eq) == 0.0:
                return self.make_bool(False)
            return self._deep_equal(heap.cdr(a), heap.cdr(b), heap)
        return self.make_bool(False)

    def _op_length(self, lst, heap):
        count = 0
        while self.is_pair(lst):
            count += 1
            lst = heap.cdr(lst)
        return self.make_float(float(count))

    def _op_append(self, args, heap):
        if len(args) == 0:
            return self.make_nil()
        if len(args) == 1:
            return args[0]
        result = args[-1]
        for lst in reversed(args[:-1]):
            elements = []
            cur = lst
            while self.is_pair(cur):
                elements.append(heap.car(cur))
                cur = heap.cdr(cur)
            for elem in reversed(elements):
                result = heap.cons(elem, result)
        return result

    def _op_reverse(self, lst, heap):
        result = self.make_nil()
        cur = lst
        while self.is_pair(cur):
            result = heap.cons(heap.car(cur), result)
            cur = heap.cdr(cur)
        return result

    # ------------------------------------------------------------------ #
    # Quote materialization
    # ------------------------------------------------------------------ #

    def materialize_quote(self, datum, heap: GenericHeap):
        if datum is None or (isinstance(datum, list) and len(datum) == 0):
            return self.make_nil()
        if isinstance(datum, str):
            try:
                return self.make_int(int(datum))
            except ValueError:
                pass
            try:
                return self.make_float(float(datum))
            except ValueError:
                pass
            if datum == "#t":
                return self.make_bool(True)
            if datum == "#f":
                return self.make_bool(False)
            sid = self._symtab.intern(datum)
            return self.make_symbol(sid)
        if isinstance(datum, (int, float)):
            return self.make_float(float(datum))
        if isinstance(datum, bool):
            return self.make_bool(datum)
        if isinstance(datum, list):
            elements = [self.materialize_quote(item, heap) for item in datum]
            return heap.build_list(elements)
        raise TypeError(f"Cannot quote datum of type {type(datum)}: {datum}")
