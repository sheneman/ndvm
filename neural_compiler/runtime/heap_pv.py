############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# heap.py: Dict-backed heap for cons cells and compound data structures. Each slot stores one TaggedValue tensor in a...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Dict-backed heap for cons cells and compound data structures.

Each slot stores one TaggedValue tensor in a Python dict, keyed by
integer address.  Cons cells occupy two consecutive slots (car at addr,
cdr at addr+1).

Differentiability: reads return the exact tensor that was written —
no in-place tensor mutations, so autograd chains are preserved even
across many intervening cons calls.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neural_compiler.runtime.payload_value import (
    VALUE_DIM,
    make_pair,
    make_nil,
    extract_payload,
)

DEFAULT_MAX_HEAP = 65536


class TensorHeap:
    """Dict-backed heap for storing compound Scheme values."""

    def __init__(self, max_size: int = DEFAULT_MAX_HEAP,
                 device: torch.device | None = None):
        self.max_size = max_size
        self.device = device
        self._slots: dict[int, Tensor] = {}
        self.alloc_ptr = 0

    def reset(self) -> None:
        self._slots.clear()
        self.alloc_ptr = 0

    def cons(self, car: Tensor, cdr: Tensor) -> Tensor:
        """Allocate a pair cell. Returns a pair-tagged value with heap addresses."""
        if self.alloc_ptr + 2 > self.max_size:
            raise RuntimeError(
                f"Heap overflow: tried to allocate at {self.alloc_ptr}, max_size={self.max_size}. "
                f"A typical (terminating) recursive program uses very little heap, so this most "
                f"often means a NON-TERMINATING program -- e.g. a base case that never fires "
                f"because it uses an operator the interpreter does not support. For a genuinely "
                f"large list/recursion workload, raise the cap via evaluate(..., max_heap=N) "
                f"(the heap is dict-backed, so a larger cap is free until used)."
            )
        car_addr = self.alloc_ptr
        cdr_addr = self.alloc_ptr + 1
        self.alloc_ptr += 2

        self._slots[car_addr] = car
        self._slots[cdr_addr] = cdr

        return make_pair(float(car_addr), float(cdr_addr), device=self.device)

    def store(self, t: Tensor) -> int:
        """Stash a raw tensor (e.g. a Strategy-B tensor-payload vector/matrix) in ONE fresh
        slot and return its integer address. Identity-preserving (no clone/detach) so autograd
        flows through a later ``read``. A VECTOR-tagged value points here via its payload addr."""
        if self.alloc_ptr + 1 > self.max_size:
            raise RuntimeError(
                f"Heap overflow: tried to allocate a tensor slot at {self.alloc_ptr}, "
                f"max_size={self.max_size}. Raise the cap via evaluate(..., max_heap=N) "
                f"(the heap is dict-backed, so a larger cap is free until used)."
            )
        addr = self.alloc_ptr
        self.alloc_ptr += 1
        self._slots[addr] = t
        return addr

    def read(self, addr: Tensor | int | float) -> Tensor:
        """Read a TaggedValue from the heap at the given address."""
        if isinstance(addr, Tensor):
            idx = int(addr.item())
        else:
            idx = int(addr)
        if idx not in self._slots:
            raise IndexError(
                f"Heap read out of bounds: addr={idx}, allocated={self.alloc_ptr}"
            )
        return self._slots[idx]

    def write(self, addr: Tensor | int | float, val: Tensor) -> None:
        """Write a TaggedValue to the heap at the given address."""
        if isinstance(addr, Tensor):
            idx = int(addr.item())
        else:
            idx = int(addr)
        if idx < 0 or idx >= self.max_size:
            raise IndexError(
                f"Heap write out of bounds: addr={idx}, max_size={self.max_size}"
            )
        self._slots[idx] = val

    def car(self, pair_val: Tensor) -> Tensor:
        """Extract car from a pair tagged value."""
        payload = extract_payload(pair_val)
        car_addr = payload[0]
        return self.read(car_addr)

    def cdr(self, pair_val: Tensor) -> Tensor:
        """Extract cdr from a pair tagged value."""
        payload = extract_payload(pair_val)
        cdr_addr = payload[1]
        return self.read(cdr_addr)

    def to(self, device: torch.device) -> "TensorHeap":
        """Move heap to a device. Returns self for chaining."""
        self.device = device
        self._slots = {k: v.to(device) for k, v in self._slots.items()}
        return self

    def allocated(self) -> int:
        """Number of slots currently allocated."""
        return self.alloc_ptr

    def build_list(self, elements: list[Tensor]) -> Tensor:
        """Build a proper Scheme list from a sequence of tagged values.

        (list a b c) = (cons a (cons b (cons c nil)))
        """
        result = make_nil(device=self.device)
        for elem in reversed(elements):
            result = self.cons(elem, result)
        return result
