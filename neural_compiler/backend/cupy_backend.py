############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# cupy_backend.py: CuPy backend for the neural compiler (GPU-accelerated NumPy).
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""CuPy backend for the neural compiler (GPU-accelerated NumPy)."""

from __future__ import annotations

from neural_compiler.backend import register_backend
from neural_compiler.backend.base import NumpyFamilyBackend

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


class CupyBackend(NumpyFamilyBackend):
    name = "cupy"
    supports_autograd = False

    def __init__(self):
        if not _CUPY_AVAILABLE:
            raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")
        super().__init__(cp)


if _CUPY_AVAILABLE:
    register_backend("cupy", CupyBackend)
