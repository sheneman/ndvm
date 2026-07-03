#!/usr/bin/env python3
"""A second client of the NDVM value representation: a differentiable stack-bytecode VM.

The MLSys resubmission (reviewer weakness 1: a single measured client) needs evidence that the
structural/numeric-split value representation generalizes beyond the DMCI tree-walking evaluator. This
module is a SECOND, independent front end -- a small stack-bytecode VM with its own dispatch model
(linear instruction stream, explicit operand stack) -- that shares the same value contract as
neural_compiler.runtime.payload_value (PV: a native int tag plus torch-scalar payloads, gradient-carrying
ONLY for numeric values; structural/control values are plain Python). It contains NO DMCI parser or
evaluator code. The point: engine reuse across two dispatch models (tree-walking + bytecode), so the
representation -- not the interpreter -- is what earns the speedup.

Two value backends, switchable per run, exercise the SAME instruction semantics:

  * backend="payload" -- the split: numeric stack values are PV(tag, torch_scalar); structural / control
    values (loop counters, branch selectors used as control) are plain Python. Autograd flows only through
    the numeric payload. This is the representation the paper measures.

  * backend="eager"   -- the naive baseline: every stack value is ONE [2]-tensor [tag, payload]; even a
    purely structural value allocates a tensor and rides the autograd tape. This is the strongest-fair
    "tag and payload fused in a tensor" encoding, the thing the split is meant to beat. Identical control
    flow and arithmetic; the only difference is the value box.

15 instructions: PUSH, LOAD, ADD, SUB, MUL, DIV, NEG, EXP, LOG, SIN, COS, DUP, LOOP, BRANCH, DOT.
A program is a list of (op, *args). LOOP/BRANCH take nested sub-bytecode as args, so control flow is
explicit and the VM never recurses into a parser.

Batch-independence: the structural walk over the instruction stream is paid ONCE; numeric payloads may be
[B] tensors, so one walk evaluates B lanes. Per-lane cost is therefore roughly flat in B.

Run/validate via ndvm/profiling/bytecode_vm_e2e.py on an HPC compute node (the Mac lacks torch).
"""
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any

import torch
from torch import Tensor

# Reuse the SHARED value module's tag codes and numeric payload box. We import the split value type and
# its numeric constructors directly -- the VM does not redefine the value contract, it consumes it. This
# is the engine reuse the second-client claim rests on.
from neural_compiler.runtime.payload_value import (
    PV, FLOAT, _PS,
    make_float as pv_make_float,
    is_number as pv_is_number,
    unwrap_number as pv_unwrap_number,
)

Instr = Tuple  # (op:str, *args)
Program = List[Instr]


# ---------------------------------------------------------------------------------------------------------
# Backend A: payload-only (the structural/numeric split). Numeric values are PV with a torch-scalar payload
# (gradient-carrying); control/structural values are plain Python floats. NO tensor for structural values.
# ---------------------------------------------------------------------------------------------------------

class PayloadBackend:
    """Stack values are PV (shared split value). Numeric payload is a torch scalar/[B] tensor; structural
    values (loop counts, branch selectors used as control) stay plain Python. Reuses payload_value.PV."""
    name = "payload"

    def num(self, t):
        # promote a python number or tensor to the SHARED numeric box exactly as the tree-walker does
        return pv_make_float(t)

    def is_num(self, v):
        return isinstance(v, PV) and pv_is_number(v)

    def payload(self, v):
        # numeric payload as a tensor (autograd-carrying); a structural value answers via _PS.item()
        p = pv_unwrap_number(v) if isinstance(v, PV) else v
        return p if isinstance(p, Tensor) else torch.as_tensor(float(p))

    def to_control(self, v):
        # collapse a numeric stack value to a plain Python float for control-flow tests (no tape)
        p = self.payload(v)
        return float(p.reshape(-1)[0].item()) if isinstance(p, Tensor) else float(p)

    # arithmetic builds a new numeric PV around the torch op (gradient flows through the payload only)
    def binop(self, f, a, b):
        return self.num(f(self.payload(a), self.payload(b)))

    def unop(self, f, a):
        return self.num(f(self.payload(a)))


# ---------------------------------------------------------------------------------------------------------
# Backend B: eager-tensor naive encoding. EVERY stack value is one [2] (or [2,B]) tensor [tag, payload].
# Even a structural value (a loop counter, a branch selector) allocates a tensor and rides the tape. Same
# control flow + arithmetic; the only change is the value box. This is the cost the split removes.
# ---------------------------------------------------------------------------------------------------------

class EagerBackend:
    """Stack values are a fused [tag, payload] tensor (the naive 'one tensor per value' encoding). Tag and
    payload are both tensor elements; structural values allocate a tensor too. Strongest-fair baseline."""
    name = "eager"

    def _box(self, payload):
        # payload may be scalar or [B]; tag row broadcast to match so the value is [2] or [2,B]
        if not isinstance(payload, Tensor):
            payload = torch.as_tensor(float(payload))
        tag = torch.full_like(payload.reshape(payload.shape) if payload.dim() else payload,
                              float(FLOAT))
        # stack tag and payload along a new leading axis -> [2] or [2, B]
        return torch.stack([tag, payload], dim=0)

    def num(self, t):
        if isinstance(t, Tensor):
            return self._box(t)
        return self._box(torch.as_tensor(float(t)))

    def is_num(self, v):
        return isinstance(v, Tensor)

    def payload(self, v):
        if isinstance(v, Tensor):
            return v[1]
        return torch.as_tensor(float(v))

    def to_control(self, v):
        p = self.payload(v)
        return float(p.reshape(-1)[0].item())

    def binop(self, f, a, b):
        return self.num(f(self.payload(a), self.payload(b)))

    def unop(self, f, a):
        return self.num(f(self.payload(a)))


# ---------------------------------------------------------------------------------------------------------
# The VM. One linear walk over the instruction stream; an explicit operand stack of backend values.
# DOT consumes vector-valued operands (lists of numeric payloads) for the small matvec/dot workload.
# ---------------------------------------------------------------------------------------------------------

# instruction set (15 instructions; the count is asserted in the harness LOC/inventory table).
INSTRUCTIONS = [
    "PUSH", "LOAD", "ADD", "SUB", "MUL", "DIV", "NEG",
    "EXP", "LOG", "SIN", "COS", "DUP", "LOOP", "BRANCH", "DOT",
]


class BytecodeVM:
    """Differentiable stack-bytecode VM. ``run(program, params)`` returns the single numeric value left on
    the stack (its payload tensor), so a caller can ``.backward()`` straight through it. ``backend`` is a
    PayloadBackend or EagerBackend instance -- the value box is the only thing that changes between them."""

    def __init__(self, backend):
        self.b = backend

    def run(self, program: Program, params: Dict[str, Any]):
        stack: List[Any] = []
        self._exec(program, stack, params)
        if not stack:
            raise RuntimeError("VM finished with empty stack")
        return self.b.payload(stack[-1])

    def _exec(self, program: Program, stack: List[Any], params: Dict[str, Any]):
        b = self.b
        for instr in program:
            op = instr[0]
            if op == "PUSH":
                stack.append(b.num(instr[1]))
            elif op == "LOAD":
                v = params[instr[1]]
                # a vector operand (for DOT) is carried as a plain Python list of payloads; a scalar
                # param is promoted into the backend's numeric box
                if isinstance(v, (list, tuple)):
                    stack.append([b.num(x) for x in v])
                else:
                    stack.append(b.num(v))
            elif op == "ADD":
                y = stack.pop(); x = stack.pop(); stack.append(b.binop(torch.add, x, y))
            elif op == "SUB":
                y = stack.pop(); x = stack.pop(); stack.append(b.binop(torch.sub, x, y))
            elif op == "MUL":
                y = stack.pop(); x = stack.pop(); stack.append(b.binop(torch.mul, x, y))
            elif op == "DIV":
                y = stack.pop(); x = stack.pop(); stack.append(b.binop(torch.div, x, y))
            elif op == "NEG":
                x = stack.pop(); stack.append(b.unop(torch.neg, x))
            elif op == "EXP":
                x = stack.pop(); stack.append(b.unop(torch.exp, x))
            elif op == "LOG":
                x = stack.pop(); stack.append(b.unop(torch.log, x))
            elif op == "SIN":
                x = stack.pop(); stack.append(b.unop(torch.sin, x))
            elif op == "COS":
                x = stack.pop(); stack.append(b.unop(torch.cos, x))
            elif op == "DUP":
                stack.append(stack[-1])
            elif op == "LOOP":
                # LOOP(count, body): repeat body `count` times. `count` is a structural constant (the loop
                # bound); the body accumulates on the stack. The counter is plain Python -- in the payload
                # backend it allocates NO tensor; this is exactly the structural walk paid once over a
                # numeric payload, so a [B] payload makes one walk evaluate B lanes.
                count = int(instr[1]); body = instr[2]
                for _ in range(count):
                    self._exec(body, stack, params)
            elif op == "BRANCH":
                # BRANCH(then_body, else_body): pop a selector value; if its (control) payload != 0 run
                # then_body else else_body. The selector is reduced to a plain Python truth value, so the
                # branch decision never rides the tape (matches the hard-dispatch in payload_value.select).
                sel = stack.pop()
                then_body, else_body = instr[1], instr[2]
                if b.to_control(sel) != 0.0:
                    self._exec(then_body, stack, params)
                else:
                    self._exec(else_body, stack, params)
            elif op == "DOT":
                # DOT: pop two vector operands (lists of numeric values) -> scalar dot product. The small
                # matvec workload reuses this per output row.
                v2 = stack.pop(); v1 = stack.pop()
                acc = None
                for a, c in zip(v1, v2):
                    term = b.binop(torch.mul, a, c)
                    acc = term if acc is None else b.binop(torch.add, acc, term)
                stack.append(acc)
            else:
                raise ValueError(f"unknown instruction {op!r}")


# ---------------------------------------------------------------------------------------------------------
# Workloads as bytecode. Each returns (program, params, reference_fn) where reference_fn(params)->torch
# scalar is a plain-torch oracle for finite-difference / autograd validation. params hold differentiable
# leaf tensors (scalars or [B] batched), bound by name.
# ---------------------------------------------------------------------------------------------------------

def w1_scalar_expr():
    """W1 scalar expression:  exp(-(a*x)) * cos(w*x) + b   (damped-oscillator-like closed form)."""
    program = [
        ("LOAD", "a"), ("LOAD", "x"), ("MUL",), ("NEG",), ("EXP",),     # exp(-(a*x))
        ("LOAD", "w"), ("LOAD", "x"), ("MUL",), ("COS",),               # cos(w*x)
        ("MUL",),                                                       # product
        ("LOAD", "b"), ("ADD",),                                        # + b
    ]

    def ref(p):
        return torch.exp(-(p["a"] * p["x"])) * torch.cos(p["w"] * p["x"]) + p["b"]
    return program, ref


def w2_counted_loop(n=12):
    """W2 counted loop: logistic-map-style accumulation, x <- r*x*(1-x), repeated n times from x0.
    Implemented with the structural LOOP whose body reads/rewrites the top of stack via params is not
    possible (params are read-only), so we thread the running value on the stack: body computes
    r*x*(1-x) consuming the current top and pushing the next."""
    body = [
        ("DUP",),                                   # x, x
        ("PUSH", 1.0), ("SUB",), ("NEG",),          # x, (1 - x)   [ (x-1) negated ]
        ("MUL",),                                   # x*(1-x)
        ("LOAD", "r"), ("MUL",),                    # r*x*(1-x)
    ]
    program = [("LOAD", "x0")] + [("LOOP", n, body)]

    def ref(p):
        x = p["x0"]
        for _ in range(n):
            x = p["r"] * x * (1.0 - x)
        return x
    return program, ref


def w3_branch():
    """W3 branch: select(g>0 ? a*x : b/x). The selector is a numeric value computed from params; BRANCH
    reduces it to a control decision (hard dispatch). To make the selected path differentiable we run the
    selected arm with autograd, the (data-independent) decision itself is structural."""
    program = [
        ("LOAD", "g"),                              # selector
        ("BRANCH",
         [("LOAD", "a"), ("LOAD", "x"), ("MUL",)],  # then: a*x
         [("LOAD", "b"), ("LOAD", "x"), ("DIV",)]), # else: b/x
    ]

    def ref(p):
        g = float(p["g"].reshape(-1)[0].item())
        if g != 0.0:
            return p["a"] * p["x"]
        return p["b"] / p["x"]
    return program, ref


def w4_matvec(rows=3, cols=3):
    """W4 small matrix/dot: y = sum_i (M[i] . v), the summed entries of a matvec. M rows and v are vector
    params (lists of scalars). Exercises DOT and accumulation."""
    program = []
    # push each row . v, summing
    for i in range(rows):
        program += [("LOAD", f"m{i}"), ("LOAD", "v"), ("DOT",)]
        if i > 0:
            program += [("ADD",)]

    def ref(p):
        total = None
        for i in range(rows):
            row = torch.stack(p[f"m{i}"]) if isinstance(p[f"m{i}"], list) else p[f"m{i}"]
            vv = torch.stack(p["v"]) if isinstance(p["v"], list) else p["v"]
            d = (row * vv).sum()
            total = d if total is None else total + d
        return total
    return program, ref


def all_workloads():
    return {
        "W1_scalar_expr": w1_scalar_expr(),
        "W2_counted_loop": w2_counted_loop(),
        "W3_branch": w3_branch(),
        "W4_matvec": w4_matvec(),
    }
