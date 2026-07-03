# NDVM Phase 3b: lane masks / per-lane divergent control flow

Status: **complete.** One structural walk over B lanes now handles control flow that differs *per lane*
(per-lane convergence loops, per-lane branches), which Phase 3 rejected with `BatchError`. This is
purely additive: uniform control flow and B=1 stay byte-identical to Phase 3. CPU, single core, float32.

The design is locked in `PHASE3B_DESIGN.md` (strategy S1, confirmed against two alternatives by an
adversarial design review). This file reports the realized implementation and validation.

## What was built

- **Active-lane set as dynamic scope** (`src/interp.hpp`): `active_full_` (all B lanes, the fast path) +
  `active_lanes_` (a sorted index list when narrowed). A divergent `if`/`cond` narrows the set around its
  branch recursion through an RAII `ActiveGuard`, so a thrown error restores the engine's active set on
  unwind. The arena stays uniformly `B_`-strided; only *which lanes participate* changes.
- **`classify` replaces `truthy`** (`src/interp.cpp`): reduces a test over the **active** lanes to
  THEN / ELSE / MIXED. B=1 and uniform batches always agree -> never MIXED -> the existing trampoline
  (TCO preserved, byte-identical).
- **Divergent `if`/`cond` split + merge**: on MIXED, partition the active set into `then_lanes` /
  `else_lanes` (each a strict subset, so nested splits are <= B-1 deep), evaluate each branch under its
  subset, then `Op::SELECT` merges per lane into one full-(parent-)active value. `cond` is lowered to
  right-nested `if` (`eval_cond_tail`), reusing the one split+select path and preserving per-lane clause
  laziness. Nested divergence composes: each lane's value is owned by the SELECT at the level where it
  terminated.
- **`Op::SELECT`** (the only adjoint-routing node): forward assembles the output per lane from the two
  branch results; backward routes the output adjoint to `v_then` on `then_lanes` and to `v_else` on the
  **parent-active complement** (`actset_pool_[n.actset] \ then_lanes`, NOT the full-B complement -- so a
  lane terminated at an OUTER select gets zero adjoint at an INNER one). Scalars and VecCell slabs both
  supported. Mergeability is strict: scalars merge same-tag (INT/FLOAT interchangeable as numbers; a
  BOOLEAN only with a BOOLEAN); VecCells need equal `ndim`/`rows`/`cols`; PAIR/CLOSURE/SYMBOL need equal
  `aux`. Any real per-lane structural divergence raises `per-lane-divergent structural value unsupported`
  (matching the oracle, which raises on B>1 divergent control flow).
- **Backward gating** (`src/interp_tape.cpp`): `TNode` carries an `actset` id snapshotted at record time;
  every VJP replays via an `each` lambda that loops `0..B_` when FULL (byte-identical to Phase 3) or only
  the recorded lanes otherwise.
- **Bounded-iteration guard**: batched divergence trampolines until the LAST lane terminates, so a single
  non-terminating lane is capped by `max_eval_steps_` (settable via `NDVM_MAX_STEPS`) and raises loudly
  rather than hanging the batch. `ref`/`eye`/`zeros`/`ones` structural args are read from the first
  active lane and checked uniform across the active batch.

### A deliberate deviation from the spec (gate backward only)

The locked spec gates both forward and backward kernels. The implementation gates **only the backward
VJPs**; forward kernels keep looping `0..B_` (compute all lanes). This is correct and was the central
target of the implementation review: forward kernels are per-lane independent (lane i never affects lane
j), SELECT discards inactive lanes forward, and the gated backward never reads an inactive lane's primal,
so an inactive (terminated) lane's stale/`Inf`/`NaN` value can never poison a shared leaf's gradient
(`0*NaN`). Recursion still terminates because `classify` reduces over the active set only. The benefit:
every forward kernel stays byte-identical to Phase 3 (zero churn, zero byte-identity risk). The cost:
forward does some wasted work on terminated lanes in narrow divergent buckets (not work-optimal); this is
the natural place for a future compaction optimization. The review confirmed the deviation sound,
including matrix ops (`inv`/`logdet`/`det`) going singular in a dead lane.

## Validation

Validated under **clang + ASan + UBSan** locally and **g++ 12.1.0 on an HPC compute node**.

- **B=1 regression byte-identical**: 33/33 forward, 82/82 gradients (`compare_equivalence.py`). B is a
  multiplier that collapses to the Phase-2 layout at B=1.
- **Uniform batched self-consistency unchanged**: 27/27 (`test_batched.py`).
- **Divergent lane decomposition** (`test_divergent.py`, 21 cases): a batched run over B per-lane
  parameter sets reproduces, per lane (forward AND per-lane gradient), B independent B=1 runs -- and each
  B=1 run already matches the PyTorch oracle (the oracle RAISES on batched divergence, so lane
  decomposition is the gold standard). Covers: non-recursive scalar branches; a Newton-sqrt convergence
  loop (lanes converge at different iterations); a dead-lane-`Inf` stressor (a terminated lane's continued
  recur overflows -- gradient stays clean); nested 3- and 4-level termination (nested SELECTs, exercising
  the parent-active-complement rule); VecCell-output divergence; a 3-way `cond`; matrix-VJP gating with a
  singular dead lane (`inv`, the Kalman-flagship `logdet`); shared leaves across branches and actsets;
  INT/FLOAT merge. Plus 5 structural-divergence raises (per-lane pair, list accumulator, vector shape
  mismatch, boolean-vs-number tag divergence, non-uniform constructor size) and the eval-step cap.

## Hardening (adversarial implementation review)

A 4-lens review generated 24 candidate-breaking programs. Three lenses were sound (the review empirically
confirmed the forward-all-lanes deviation, the nested SELECT else-set, and matrix/shared-leaf gating all
match B=1). It found **two real bugs**, both fixed and now regression-tested:

1. **BOOLEAN-vs-number tag collapse**: `select_merge` had lumped BOOLEAN with INT/FLOAT as
   scalar-mergeable, so `(boolean? (if (> t 0) (= 1 1) 5))` collapsed to one tag and returned `[0,0]`
   instead of `[1,0]`. Fixed: a BOOLEAN merges only with a BOOLEAN; boolean-vs-number raises (test R10).
2. **Non-uniform constructor size**: `eye`/`zeros`/`ones` read a size from one lane without a uniformity
   check, so `(ones n)` with per-lane n silently used one lane's size. Fixed: the structural size must be
   uniform across the active batch or it raises (test R12). (Pre-existing; `ref` already checked.)

## Batched PyTorch boundary (done)

`NDVMFunction` / `ndvm_forward` (`python/`) now batch: a parameter maps to a `[B]` per-lane tensor (a
scalar broadcasts), one native walk fits all B lanes, the output is `[B]`, and `param.grad[i, b] =
grad_out[b] * d(out_b)/d(param_i lane b)` routes the per-lane gradient back (each lane's output depends
only on its own lane's params, so NDVM's all-B adjoint seed yields exactly that). B == 1 still returns a
scalar (the Phase-2 boundary, unchanged). The native `eval_and_grad_batched` entry binds per-lane scalars
and shares bound matrices across lanes. This exposes the Phase-3 / 3b batch-native throughput through the
torch API the co-search actually runs on. Validated on HPC (built with `setup.py build_ext --inplace`):
the B=1 boundary regression stays 28/28, and the batched suite (`tests/test_autograd_batched.py`) is 28/28
-- per-lane forward + gradients match B independent B=1 calls, and a 3-lane Adam descent drives three
independent Kalman fits through one batched op.

## Not done (next)

- **Forward-kernel gating / lane compaction**: make divergent buckets work-optimal (today forward
  recomputes terminated lanes). Low value per the Phase-0 cost model -- the recompute hits only the <1%
  numeric kernels; the dominant structural walk is shared regardless of active-lane count.
- **Per-lane gather**: `ref` with a per-lane-distinct index still raises (data-dependent indexing). Matrix
  inputs through the torch boundary are still shared across lanes (no per-lane matrices).
- **Phase 4**: structural caches (decoded-form / inline caches), the real lever on the ~85% boxing+walk.

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
pytest ndvm/tests/test_divergent.py                 # 21: per-lane divergence lane-decomposition + raises
pytest ndvm/tests/test_batched.py                   # 27: uniform batched self-consistency (unchanged)
python3 ndvm/tests/compare_equivalence.py           # 33 fwd / 82 grad vs stored oracle refs (B=1)
```
