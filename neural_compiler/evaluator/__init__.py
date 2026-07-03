############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# __init__.py: Evaluator backends for compiled compute graphs. Four execution modes: - evaluate(): Sequential Python loop over...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Evaluator backends for compiled compute graphs.

Four execution modes:
  - evaluate(): Sequential Python loop over nodes (default: torch)
  - evaluate(backend="numpy"): NumPy backend (no autograd)
  - evaluate(backend="jax"): JAX backend (functional autograd)
  - evaluate(backend="cupy"): CuPy backend (GPU-accelerated NumPy)
  - SchemeGNN: torch.nn.Module wrapper for sequential evaluation
  - DirectModule: Flat instruction execution (~1-3x overhead)
"""

from neural_compiler.evaluator.engine import evaluate as _torch_evaluate
from neural_compiler.evaluator.engine import evaluate_batched, compile_batched
from neural_compiler.evaluator.engine import set_soft_choice_tau, set_soft_choice_gumbel, set_soft_choice_hard
from neural_compiler.evaluator.gnn_module import SchemeGNN
from neural_compiler.evaluator.direct_module import DirectModule


def evaluate(graph, inputs=None, backend=None, **kwargs):
    """Evaluate a compiled compute graph.

    Args:
        graph: ComputeGraph from compile_program/compile_scheme.
        inputs: Map of input variable names to values.
        backend: Backend name. None or "torch" uses PyTorch (default).
                 Also supports "numpy", "jax", "cupy".
        **kwargs: Passed to the evaluator (max_iter, max_depth).
    """
    if backend is None or backend == "torch":
        return _torch_evaluate(graph, inputs, **kwargs)

    from neural_compiler.backend import get_backend
    from neural_compiler.evaluator.engine_generic import evaluate_generic

    be = get_backend(backend)
    return evaluate_generic(graph, inputs, be, **kwargs)


def jax_grad(graph, inputs, wrt):
    """Compute gradient of a compiled scalar program via jax.grad.

    See :func:`neural_compiler.backend.jax_backend.jax_grad` for details.
    """
    from neural_compiler.backend.jax_backend import jax_grad as _jg
    return _jg(graph, inputs, wrt)


def jax_value_and_grad(graph, inputs, wrt):
    """Compute value and gradient via jax.value_and_grad.

    See :func:`neural_compiler.backend.jax_backend.jax_value_and_grad`.
    """
    from neural_compiler.backend.jax_backend import jax_value_and_grad as _jvg
    return _jvg(graph, inputs, wrt)


__all__ = [
    "evaluate", "evaluate_batched", "compile_batched", "SchemeGNN", "DirectModule",
    "jax_grad", "jax_value_and_grad",
    "set_soft_choice_tau", "set_soft_choice_gumbel", "set_soft_choice_hard",
]
