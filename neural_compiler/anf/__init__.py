############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# __init__.py: A-Normal Form transform: flatten compound subexpressions into let bindings.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""A-Normal Form transform: flatten compound subexpressions into let bindings."""

from neural_compiler.anf.transform import to_anf
from neural_compiler.anf.anf_nodes import (
    ANFNode,
    ANFBegin,
    ANFConst,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
    ANFLetrec,
    ANFLoop,
    ANFQuote,
    ANFRecur,
)

__all__ = [
    "to_anf",
    "ANFNode",
    "ANFBegin",
    "ANFConst",
    "ANFVar",
    "ANFLet",
    "ANFIf",
    "ANFApp",
    "ANFLambda",
    "ANFLetrec",
    "ANFLoop",
    "ANFQuote",
    "ANFRecur",
]
