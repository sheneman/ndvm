############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# numpy_backend.py: NumPy backend for the neural compiler.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""NumPy backend for the neural compiler."""

from __future__ import annotations

import numpy as np
from neural_compiler.backend import register_backend
from neural_compiler.backend.base import NumpyFamilyBackend


class NumpyBackend(NumpyFamilyBackend):
    name = "numpy"
    supports_autograd = False

    def __init__(self):
        super().__init__(np)


register_backend("numpy", NumpyBackend)
