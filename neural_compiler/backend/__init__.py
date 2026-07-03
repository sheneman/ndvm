############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# __init__.py: Multi-backend support: torch (default), numpy, jax, cupy. SCOPE: this abstraction covers the **scalar,...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Multi-backend support: torch (default), numpy, jax, cupy.

SCOPE: this abstraction covers the **scalar, straight-line direct-compile** path
only (a ComputeGraph whose control flow does not depend on the differentiated
constants), lowered through ``evaluator/engine_generic.py``. The **differentiable
meta-circular interpreter** (heap + tagged values + the variable-length
trampolined recur loop) is **torch-only** and routes straight to
``evaluator/engine.py``; it is NOT served by this abstraction. See
``backend/README.md`` for why -- in short, PyTorch's define-by-run tape
differentiates data-dependent control flow directly, whereas JAX's
``lax.while_loop`` is not reverse-mode differentiable, so a JAX port is a
fixed-length ``lax.scan`` + masking re-architecture (future work). NumPy is the
forward-only reference oracle; NumPy/CuPy have no autograd.

Usage:
    from neural_compiler.evaluator import evaluate
    result = evaluate(graph, inputs, backend="numpy")
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neural_compiler.backend.base import NumpyFamilyBackend

_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get_backend(name: str = "numpy") -> NumpyFamilyBackend:
    if name not in _REGISTRY:
        _lazy_import(name)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend: {name!r}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def _lazy_import(name: str) -> None:
    if name == "numpy":
        from neural_compiler.backend.numpy_backend import NumpyBackend  # noqa: F401
    elif name == "jax":
        from neural_compiler.backend.jax_backend import JaxBackend  # noqa: F401
    elif name == "cupy":
        from neural_compiler.backend.cupy_backend import CupyBackend  # noqa: F401
    else:
        raise ValueError(f"Unknown backend: {name!r}")


def available_backends() -> list[str]:
    names = []
    for name in ("numpy", "jax", "cupy"):
        try:
            _lazy_import(name)
            if name in _REGISTRY:
                names.append(name)
        except (ImportError, ValueError):
            pass
    return names
