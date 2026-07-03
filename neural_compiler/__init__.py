############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# __init__.py: Neural Compiler: a compiler from a self-hosting Scheme subset to differentiable PyTorch computation graphs...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Neural Compiler: a compiler from a self-hosting Scheme subset to differentiable
PyTorch computation graphs (Differentiable Meta-Circular Interpretation).

Convenience re-exports of the most common entry points:

    from neural_compiler import compile_program, run_scheme, save_compiled, load_compiled

Evaluation lives in :mod:`neural_compiler.evaluator` (``evaluate``, ``evaluate_batched``),
which is imported lazily to keep ``import neural_compiler`` light.
"""

from neural_compiler.compiler import (
    compile_program,
    compile_scheme,
    run_program,
    run_scheme,
)
from neural_compiler.dmci import compile_dmci, compile_interpreter, evaluate_program
from neural_compiler.serialize import load_compiled, save_compiled

__all__ = [
    "compile_scheme",
    "compile_program",
    "run_scheme",
    "run_program",
    "compile_dmci",
    "compile_interpreter",
    "evaluate_program",
    "save_compiled",
    "load_compiled",
]
