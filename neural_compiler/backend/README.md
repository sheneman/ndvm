# Backends: scope and the two-path split

This package is the backend abstraction for the compiler. It is important to
understand what it covers and what it deliberately does **not**.

## Two paths

1. **Scalar, straight-line direct-compile path** - *this abstraction's scope.*
   A `ComputeGraph` of scalar arithmetic/math whose control flow does not depend
   on the differentiated constants is lowered through `evaluator/engine_generic.py`
   to any of `torch`, `jax`, `numpy`, `cupy`. `jax_backend.jax_grad` /
   `jax_value_and_grad` differentiate this path with `jax.grad`.

2. **The differentiable meta-circular interpreter** (`evaluator/engine.py`) -
   **torch-only.** Dictionary heap, tagged values, and the variable-length
   trampolined `recur` loop. This is *not* routed through the generic backend
   abstraction; `backend="torch"` (or `None`) goes straight to `engine.py`.

## Why the interpreter is torch-only (by design, not oversight)

DMCI must differentiate through **data-dependent control flow**: which eval–apply
clause fires, the `(ref obs k)` index, and the loop-termination test are all
decided by reading structural ints/bools off tagged values via `.item()`. That
`.item()` call is a deliberate **non-differentiable boundary** between structural
control (discrete, program-determined) and the numeric dataflow carried on-tape
as tensors. Note that *neither* torch nor JAX differentiates the discrete branch
*choice* itself (a step discontinuity); both give the gradient **along the
executed path**. The difference is what each AD substrate *allows you to write*:

- **PyTorch** - define-by-run (tape) autograd records the ops that *actually*
  execute, so it differentiates the realized trajectory - including the
  data-dependent-length trampoline - with no restructuring.
- **JAX** - traces to a functional `jaxpr`: data-dependent Python branching on
  traced values raises a concretization error (must use `lax.cond`), and
  `lax.while_loop` is **not reverse-mode differentiable** (data-dependent trip
  count). Porting the interpreter would require rewriting the trampoline as a
  fixed-length `lax.scan(max_iter)` + masking. This is **future work** (payoff:
  `jit` fusion of the ~90 ms/step interpreter overhead + `vmap` over the
  population path), not a drop-in swap - which is why `jax_grad` is intentionally
  limited to path (1). (The float32 log-det ceiling we hit would have been a
  one-line `jax_enable_x64` flag in JAX; we fixed it cleanly in torch with a
  `logdet`/`slogdet` primitive.)
- **NumPy / CuPy** - no autodiff → forward-only. NumPy is the **reference
  oracle**: the validation gate checks every DMCI result against the NumPy twin,
  which is exactly why a non-autodiff backend earns a place in the system. CuPy is
  redundant with `torch.cuda` and loses gradients; the single-filter interpreter
  walk is latency/dispatch-bound (sequential tiny ops + `.item()` syncs), so GPU
  does not help it regardless.

The batched Strategy-B tensor/matrix ops and the `logdet`-based determinant are
torch-only for the same reason.
