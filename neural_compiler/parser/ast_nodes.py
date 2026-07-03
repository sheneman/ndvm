############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# ast_nodes.py: AST node types for Scheme. Node types: Const(value) — numeric or boolean literal Var(name) — variable reference...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""AST node types for Scheme.

Node types:
  Const(value)           — numeric or boolean literal
  Var(name)              — variable reference
  If(test, then_, else_) — conditional
  Lambda(params, body)   — fixed-arity function abstraction
  Let(bindings, body)    — let bindings: [(name, expr), ...]
  App(func, args)        — function/primitive application
  Loop(bindings, body)   — tail-recursive loop: (loop ((var init) ...) body)
  Recur(args)            — tail call back to enclosing loop: (recur expr ...)
  Letrec(bindings, body) — recursive bindings: (letrec ((f (lambda ...)) ...) body)
  Quote(datum)           — quoted literal: (quote datum) or 'datum
  Begin(exprs)           — sequencing: (begin e1 e2 ... en)
  Define(name, value)    — top-level definition: (define name expr)
  Program(forms)         — sequence of top-level forms
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class SchemeChar:
    """Wrapper to distinguish character values from integers in the AST pipeline."""
    code_point: int


@dataclass(frozen=True)
class ASTNode:
    pass


@dataclass(frozen=True)
class Const(ASTNode):
    value: Union[int, float, bool, SchemeChar]


@dataclass(frozen=True)
class Var(ASTNode):
    name: str


@dataclass(frozen=True)
class If(ASTNode):
    test: ASTNode
    then_: ASTNode
    else_: ASTNode


@dataclass(frozen=True)
class Lambda(ASTNode):
    params: tuple[str, ...]
    body: ASTNode


@dataclass(frozen=True)
class Let(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


@dataclass(frozen=True)
class App(ASTNode):
    func: ASTNode
    args: tuple[ASTNode, ...]


@dataclass(frozen=True)
class Loop(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


@dataclass(frozen=True)
class Recur(ASTNode):
    args: tuple[ASTNode, ...]


@dataclass(frozen=True)
class Letrec(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


@dataclass(frozen=True)
class Quote(ASTNode):
    datum: object


@dataclass(frozen=True)
class Begin(ASTNode):
    exprs: tuple[ASTNode, ...]


@dataclass(frozen=True)
class Define(ASTNode):
    name: str
    value: ASTNode


@dataclass(frozen=True)
class SoftChoice(ASTNode):
    options: tuple[ASTNode, ...]
    weights: ASTNode

@dataclass(frozen=True)
class Program(ASTNode):
    forms: tuple[ASTNode, ...]


PRIMITIVES = {
    "+", "-", "*", "/",
    "=", "<", ">", "<=", ">=",
    "not", "and", "or",
    "min", "max", "abs",
    "modulo", "remainder",
    "sin", "cos", "exp", "sqrt", "log", "pow",
    # Vector operations
    "vec", "ref", "dot", "cross", "norm", "normalize", "vsum", "scale", "vlen",
    # Matrix operations
    "mat", "matmul", "matvec", "transpose", "trace", "det", "logdet", "inv",
    "outer", "eye", "zeros", "ones",
    # Cons cells and list operations
    "cons", "car", "cdr", "list", "length", "append", "reverse",
    # Apply
    "apply",
    # Type predicates
    "null?", "pair?", "number?", "boolean?", "symbol?", "char?",
    "procedure?", "string?", "vector?",
    # Identity and equality
    "eq?", "eqv?", "equal?",
}
