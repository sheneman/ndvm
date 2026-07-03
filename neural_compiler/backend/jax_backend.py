############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# jax_backend.py: JAX backend for the neural compiler. JAX arrays are immutable, so heap mutation uses ``.at[].set()`` instead of...
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""JAX backend for the neural compiler.

JAX arrays are immutable, so heap mutation uses ``.at[].set()`` instead
of plain index assignment. JAX autograd is supported via ``jax.grad``.

SCOPE: this backend serves the **scalar, straight-line direct-compile path only**.
The differentiable meta-circular interpreter is torch-only: its variable-length
trampolined ``recur`` loop would need JAX's ``lax.while_loop``, which is NOT
reverse-mode differentiable, so a JAX port requires rewriting the trampoline as a
fixed-length ``lax.scan(max_iter)`` + masking (future work; payoff is ``jit``
fusion of the per-step interpreter overhead + ``vmap`` over the population path).
See ``backend/README.md``.
"""

from __future__ import annotations

from neural_compiler.backend import register_backend
from neural_compiler.backend.base import NumpyFamilyBackend

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


class JaxBackend(NumpyFamilyBackend):
    name = "jax"
    supports_autograd = True

    def __init__(self):
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
        super().__init__(jnp)

    def heap_set(self, storage, idx: int, val):
        return storage.at[idx].set(val)


def jax_grad(graph, inputs, wrt):
    """Compute gradient of a compiled scalar program via jax.grad.

    Works for programs where control flow does not depend on the
    differentiated variables (straight-line arithmetic, math ops).

    Root cause of the constraint: JAX traces to a functional jaxpr that forbids
    data-dependent Python branching on traced values (a concretization error;
    would need ``lax.cond``) and whose ``lax.while_loop`` is not reverse-mode
    differentiable. Programs with data-dependent control flow -- the tagged
    meta-circular interpreter, or any data-dependent loop trip count -- must use
    the torch path instead (``evaluator/engine.py``).

    Args:
        graph: ComputeGraph (scalar, not tagged).
        inputs: dict mapping input names to float values.
        wrt: str or list[str] — input name(s) to differentiate w.r.t.

    Returns:
        Single float gradient if wrt is a string, else tuple of floats.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for jax_grad")
    from neural_compiler.evaluator.engine_generic import evaluate_generic

    backend = JaxBackend()
    scalar_wrt = isinstance(wrt, str)
    if scalar_wrt:
        wrt = [wrt]

    def fn(*diff_vals):
        full_inputs = dict(inputs)
        for name, val in zip(wrt, diff_vals):
            full_inputs[name] = val
        return evaluate_generic(graph, full_inputs, backend, raw_result=True)

    diff_inputs = [jnp.float32(float(inputs[name])) for name in wrt]
    argnums = 0 if scalar_wrt else tuple(range(len(wrt)))
    grad_fn = jax.grad(fn, argnums=argnums)
    grads = grad_fn(*diff_inputs)

    if scalar_wrt:
        return float(grads)
    return tuple(float(g) for g in grads)


def jax_value_and_grad(graph, inputs, wrt):
    """Compute both value and gradient via jax.value_and_grad.

    Same constraints as jax_grad: scalar program, control flow must not
    depend on the differentiated variables.

    Returns:
        (value, grad) if wrt is a string.
        (value, (grad1, grad2, ...)) if wrt is a list.
    """
    if not _JAX_AVAILABLE:
        raise ImportError("JAX is required for jax_value_and_grad")
    from neural_compiler.evaluator.engine_generic import evaluate_generic

    backend = JaxBackend()
    scalar_wrt = isinstance(wrt, str)
    if scalar_wrt:
        wrt = [wrt]

    def fn(*diff_vals):
        full_inputs = dict(inputs)
        for name, val in zip(wrt, diff_vals):
            full_inputs[name] = val
        return evaluate_generic(graph, full_inputs, backend, raw_result=True)

    diff_inputs = [jnp.float32(float(inputs[name])) for name in wrt]
    argnums = 0 if scalar_wrt else tuple(range(len(wrt)))
    vg_fn = jax.value_and_grad(fn, argnums=argnums)
    val, grads = vg_fn(*diff_inputs)

    val = float(val)
    if scalar_wrt:
        return val, float(grads)
    return val, tuple(float(g) for g in grads)


if _JAX_AVAILABLE:
    register_backend("jax", JaxBackend)
