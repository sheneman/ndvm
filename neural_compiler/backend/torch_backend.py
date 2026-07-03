############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# torch_backend.py: PyTorch backend — delegates to the existing evaluator engine. This exists for registry completeness;...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""PyTorch backend — delegates to the existing evaluator engine.

This exists for registry completeness; ``backend="torch"`` (or ``None``)
routes directly to ``engine.py`` without going through the generic
evaluator, preserving autograd support and full backward compatibility.
"""

from __future__ import annotations

from neural_compiler.backend import register_backend


class TorchBackend:
    name = "torch"
    supports_autograd = True


register_backend("torch", TorchBackend)
