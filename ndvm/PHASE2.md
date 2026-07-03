# NDVM Phase 2: native reverse-mode AD (gradient-equivalent to the DMCI oracle)

Status: **complete**. A native define-by-run tape and backward pass compute gradients through the
forward runtime, and they match the PyTorch DMCI oracle's `autograd` gradients on 33 programs (82
per-parameter gradients), ASan-clean. NDVM now differentiates, which is the property the whole
approach is about. CPU, single core, float32. The PyTorch `autograd.Function` boundary is wired and
validated (a torch optimizer can drive NDVM end to end). The implementation was hardened by a
multi-agent adversarial review and by building on a second compiler (see "Hardening").

## What was built

- **Tape + adjoint buffers** (`src/interp.hpp`, `src/interp_tape.cpp`): a vec-aware tape (`Op`, `Ref`,
  `TNode`) records one node per differentiable numeric op; structural ops record nothing. Adjoints live
  in `adj_` (parallel to the scalar payload table) and `vadj_` (one slab per `VecCell`), sized lazily at
  `backward()`. The structural/numeric split now carries adjoints, not just primals.
- **Backward replay** (`dispatch_adjoint`): seeds the scalar output adjoint to 1 and replays the tape in
  reverse, applying the per-op VJP and accumulating with `+=` (so multi-use values sum correctly). The
  VJP catalog covers scalar arithmetic (variadic folds), `sin/cos/exp`, the `sqrt`/`log` 1e-8 clamp
  subgradient (grad 0 below the clamp), `pow`, `abs`, `min`/`max` (tie-splitting), `modulo`/`remainder`,
  elementwise vec/mat `+ - * /`, `scale`, `dot`, `vsum`, `norm`, `normalize`, `cross`, `matmul`,
  `matvec`, `transpose`, `trace`, `outer`, `det`, `logdet`, `inv`, and the `vec`/`mat`/`ref`
  gather/scatter. `det`/`logdet` cache the LU inverse from the forward (their adjoints need `A^{-1}`).
- **Trace-constant control**: comparisons and `if`/`cond` predicates emit no tape node, so gradients do
  not flow through branch selection (the realized branch carries ordinary gradients).
- **CLI + harness** (`tools/ndvm_run.cpp`, `tests/`): `NDVM_GRAD=1 ndvm_run prog binds` prints
  `d(result)/d(param)` for each bound scalar; `oracle_refs.py` now emits per-param autograd gradients
  and central finite differences; `compare_equivalence.py`/`test_equivalence.py` gate both forward and
  gradient equivalence.
- **PyTorch autograd boundary** (`python/ndvm_ext.cpp`, `python/ndvm_autograd.py`, `python/setup.py`): a
  pybind11 extension exposes the native forward+backward; `NDVMFunction(torch.autograd.Function)` makes
  NDVM a differentiable torch op (params as autograd leaves in, scalar output out, `loss.backward()`
  routes native gradients back). Built ahead-of-time via `setup.py build_ext --inplace` (no ninja
  needed). `test_autograd_boundary.py` (28 tests) checks every program's `params.grad` against the
  oracle and runs an Adam descent that reduces the Kalman NLL, confirming an external optimizer can
  drive NDVM end to end.

## Gradient equivalence (the deliverable)

**33/33 forward and 82/82 per-parameter gradients match the PyTorch DMCI oracle** within tolerance;
`pytest tests/test_equivalence.py` is green (60 tests) and the full sweep is ASan-clean. Most scalar and
structural gradients match the oracle's `autograd` to **0 or ~1e-10**. The headline matrix case, the
80-step 2x2 Kalman-filter NLL, matches **dNLL/dq to abs 2.9e-6 and dNLL/dr to 3.1e-7** through the
`inv`/`det`/`matmul`/`matvec`/`scale` adjoints. Coverage that the gradient suite exercises against the
oracle: every VJP listed above, including `logdet`, `transpose`, `trace`, `outer`, `norm`, `normalize`,
`cross`, elementwise `*`, `vec`/`mat`, `matvec`/`dot`, the 16-step recursive logistic gradient, the
`min`/`max` subgradient (d/da=0, d/db=1), `cond`-branch gradients, and gradient through `car`/`cdr`
(flows only to the selected list element).

float32 tolerances follow the de-risk pilot: scalar/structural gradients are effectively exact; LU-based
linear algebra (`det`/`logdet`/`inv` and the Kalman path) is held to ~1e-3 relative (LU vs LAPACK
reduction order), never bit-exact.

## Hardening (adversarial review outcomes)

A multi-agent review (VJP math, tape memory safety, gaps) produced 24 findings. Acted on: added
square-matrix validation to `det`/`logdet`/`inv` and rectangular-rows validation to `mat` (undefined
inputs that could otherwise read out of bounds; torch raises on these too), and added a `modulo`/
`remainder` gradient test so those VJPs are now empirically validated (d/da=2, d/db=-6). Reviewed and
intentionally left unchanged: the `pow` gradient for base $\le 0$ (it propagates nan/inf exactly as
torch.pow does, so guarding it would diverge from the oracle; it is outside the validated regime). The
det/logdet inverse-cache `aux` is always set when a node is recorded (recording is a no-op when taping
is off), so it cannot be read uninitialized. `run()` clears the tape per call, so repeated
forward+backward (an optimizer loop) is safe.

Building the autograd extension with **g++** (the prior validation was clang-only, on the Mac and under
ASan) surfaced a real bug that clang and ASan had hidden: `vec` constructed its payload via
`mk_vec(1, 1, d.size(), std::move(d))`, where the order of evaluating `d.size()` and `std::move(d)` is
unspecified in C++. g++ moved `d` first, so `d.size()` read the moved-from vector and every `(vec ...)`
got length 0 (collapsing matrices to empty). Fixed by hoisting the length into a local before the move.
The lesson: validate on the deployment compiler, not only on the development one.

## Not done in Phase 2 (next)

- **Phase 3:** batch-native payloads (the co-search throughput multiplier). The adjoint buffers are
  per-element today; Phase 3 adds the batch axis.
- The like-for-like metacircular forward/backward speedup (run `compiler.scm` natively).
- GPU (Phase 6) and MLIR/Enzyme (Phase 7).

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
NDVM_GRAD=1 ndvm/build/ndvm_run <program.scm> [bindings]   # prints grad <param> <d(result)/d(param)>
python3 ndvm/tests/oracle_refs.py                          # HPC compute node (needs torch): refs + grads
pytest ndvm/tests/test_equivalence.py                      # 60: forward + gradient equivalence vs oracle

# PyTorch autograd boundary (HPC compute node: needs torch + a C++17 compiler):
(cd ndvm/python && python setup.py build_ext --inplace)    # builds ndvm_native.*.so (no ninja)
pytest ndvm/tests/test_autograd_boundary.py                # 28: NDVMFunction grads == oracle + optimizer step
```
