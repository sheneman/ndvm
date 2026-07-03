# NDVM Phase 3: batch-native execution (one walk over B lanes)

Status: **complete**. The structural walk (tags, heap, dispatch, control flow, the tape) runs **once**;
numeric payloads carry a batch axis `B`, so a single evaluator walk fits `B` parameter vectors
(restarts / cells / data points) at once. Per-lane forward outputs and per-lane gradients match the
PyTorch DMCI oracle, including the 80-step Kalman flagship. This is the co-search throughput multiplier.
CPU, single core, float32.

## What was built

- **B-strided value model** (`src/interp.hpp`): an engine-wide batch width `B_`. Scalar payloads are
  B-strided (`primal_at(pid)` / `adj_at(pid)` span B lanes); `VecCell.data` is batch-leading
  `[B, rows*cols]`. `Val` stays scalar `{tag, aux, pid}`: one structural value addresses all B lanes, so
  tags, heap addresses, closures, environments, and the tape are allocated once and shared.
- **B-loop kernels** (`src/interp.cpp`, `src/interp_linalg.cpp`): every numeric forward primitive
  (scalar arithmetic/transcendental/comparison, and the vec/mat ops `vec`/`mat`/`ref`/`dot`/`matvec`/
  `matmul`/`transpose`/`trace`/`outer`/`scale`/elementwise, with an independent partial-pivot LU per lane
  for `det`/`inv`/`logdet`) loops over the B lanes; the backward VJPs (`src/interp_tape.cpp`) do the same.
  The structural walk and dispatch stay scalar. Kernels read all inputs before any allocation, so no
  pointer dangles across a buffer growth (the Phase-2 lesson, applied).
- **Control-flow uniformity** (`truthy`): a batched branch test reduces to one decision (all lanes
  nonzero -> true, all zero -> false, mixed -> error), matching the oracle's `_batch_branch_decision`.
  Population batching over parameter vectors keeps control flow lane-uniform (loop counters are
  structural), so it reduces cleanly; genuinely data-dependent control raises the same `BatchError`.
- **Batched bindings + per-lane gradients**: `bind_scalar_batched(name, [B] values)` binds one value per
  lane; `bind_matrix` broadcasts a shared matrix to all lanes. Backward seeds the output adjoint to 1 on
  all B lanes (the native equivalent of `loss = output.sum(); loss.backward()`), so each parameter's
  adjoint is its per-lane gradient. `ndvm_run` gains `NDVM_B`, `scalarb` bindings, and per-lane output.

## Validation

- **B=1 regression: byte-identical to Phase 2** (33/33 forward, 82/82 gradients), under both clang and
  g++. B is a multiplier that collapses to the Phase-2 layout at B=1.
- **Batched self-consistency: 26/26** (`test_batched.py`). For each program, one batched walk over B=8
  distinct per-lane parameter sets reproduces, lane by lane, the B=1 run with that lane's parameters
  (forward and gradients), under clang and g++. Batched-lane == scalar-lane, and scalar == oracle, so
  batched == oracle.
- **Batched vs oracle: 7/7 per-lane** (`validate_batched_oracle.py`, on HPC under g++): batched NDVM
  matches the oracle's per-lane forward and per-lane gradients directly, including `kalman2d_T80` (all
  lanes' forward and dNLL/dq, dNLL/dr). Local `pytest` is 86 green (60 equivalence + 26 batched).

## Throughput (the point)

One walk over B lanes amortizes the per-walk structural overhead. On the 80-step 2x2 Kalman NLL with
gradients (`ndvm_run`, single core), per-lane cost falls as B grows:

| B | per-lane cost |
|---|--:|
| 1 | 2.96 ms |
| 64 | 0.082 ms |
| 256 | 0.049 ms |

About a 60x drop in per-lane cost from B=1 to B=256: the structural walk is paid once and the batch
rides the dense payloads. This is exactly the population-fitting pattern OpenEvolve-style co-search needs
(one candidate program, many restarts/cells), and it is the empirical payoff of the structural/numeric
split that the Phase-0 cost model predicted.

## Hardening (adversarial review outcomes)

A multi-agent review produced 16 findings. Acted on: (1) `ref` now checks that its index is uniform
across the batch and raises otherwise, rather than silently using lane 0 (a structural index that varies
per lane is data-dependent indexing, a Phase-3b feature; this matches the control-flow uniformity rule);
(2) the local batched self-consistency test now includes matrix-input programs (the Kalman rollout, with
its shared matrix broadcast across lanes), so the matrix B-loop kernels are covered by `pytest`, not only
by the HPC oracle script (27/27). The remaining findings (runtime shape-uniformity assertions; extra
coverage for >2-operand batched division and the normalize clamp boundary) are defensive/coverage notes;
the math is validated at B=1 and the B-loops are mechanical, so they are not bugs.

## Not done (next)

- **Lane masks / trace bucketing** (design 9.3-9.4): divergent (data-dependent) control flow across the
  batch currently raises `BatchError`. Per-lane masked execution or trace bucketing is Phase 3b.
- **Batched PyTorch boundary**: the `NDVMFunction` autograd boundary is currently B=1 (the native
  runtime batches; exposing batched params through the torch boundary is a thin follow-on).
- **SIMD / multicore / GPU**: the structural-walk amortization already delivers the throughput win;
  SIMD over the lane loops (Phase 4), the multicore scheduler (Phase 5), and the GPU backend (Phase 6)
  are additive. Intra-lane reduction order is kept identical to the scalar path to preserve parity.

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
NDVM_B=8 NDVM_GRAD=1 ndvm/build/ndvm_run prog.scm binds   # scalarb <name> v0..v7 ; per-lane result + grads
pytest ndvm/tests/test_batched.py                          # 26: batched lane == per-lane scalar (clang/g++)
python3 ndvm/tests/validate_batched_oracle.py              # HPC: batched NDVM == oracle per-lane (7/7)
```
