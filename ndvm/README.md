# NDVM: Native Differentiable Virtual Machine for DMCI

A high-performance, batching-first, reverse-mode differentiable virtual machine for executing the
compiled DMCI Scheme evaluator natively, while preserving the DMCI thesis:

```
compile the evaluator once -> supply arbitrary object programs as data ->
differentiate through the evaluator to continuous parameters -> no per-program recompilation
```

**Full design spec:** [`../docs/NDVM_native_differentiable_vm.md`](../docs/NDVM_native_differentiable_vm.md)

This is a separate research/engineering track (targeting its own paper). It develops on the `ndvm`
branch, based off `complete_scheme` (the canonical DMCI core). The PyTorch DMCI backend in
`neural_compiler/` is the **frozen reference oracle**: NDVM validates forward + gradient equivalence
against it and must not modify it.

## Defining invariant (do not violate)

> Make the interpreter fast **without compiling away the interpreted program.**

NDVM executes the *evaluator* natively; object programs remain runtime S-expression data. It is **not**
a Futamura specializer and does not residualize each candidate program into its own graph. The core
representation change is the **structural/numeric split**: scalar native tags + heap addresses for
structure; dense differentiable payload buffers (carrying the batch dimension) for numbers. This kills
the current backend's dominant cost (the perf autopsy found ~61% of forward time is tagged-value boxing).

## Layout

```
ndvm/
  README.md                         this file
  CMakeLists.txt                    C++ build (Phase 1+; stub)
  include/ndvm/                     core data-structure contracts (from design sections 4,5,7)
    value.hpp                       Tag + Value (scalar structural value)
    payload.hpp                     differentiable numeric payload table (SoA primal/adjoint)
    heap.hpp                        immutable arena heap (pairs/closures/vectors)
    tape.hpp                        native reverse-mode AD tape (adjoint bytecode)
  src/                              native runtime implementation (stubs)
  python/                           Python/PyTorch API boundary (design section 14)
    ndvm_autograd.py                torch.autograd.Function wrapper (stub)
  profiling/                        Phase 0: the profiling contract (baseline cost model)
    profile_dmci_baseline.py        instruments the current PyTorch DMCI; runnable where torch is available
  tests/                            forward + gradient equivalence vs the PyTorch DMCI oracle
    test_equivalence.py             scaffold
```

## Phase roadmap (see design section 18)

- **Phase 0** — profiling contract (baseline cost model of current DMCI). **Done:** `profiling/BASELINE.md`.
- **Phase 1** — native forward runtime (value/heap/env, direct-threaded eval, lazy if, tail calls). **Done:** `PHASE1.md` (forward-equivalent to the oracle on 21 programs).
- **Phase 2** — native reverse-mode tape (payload + adjoint buffers, backward) + PyTorch `autograd.Function` boundary. **Done:** `PHASE2.md` (gradient-equivalent to the oracle: 82/82 per-param grads incl. Kalman dNLL/dq,dr; NDVMFunction lets a torch optimizer drive NDVM end to end).
- **Phase 3** — batch-native execution (batched payloads, population batching). **Done:** `PHASE3.md` (one walk over B lanes; per-lane forward + gradients match the oracle incl. Kalman; ~60x lower per-lane cost at B=256).
- **Phase 3b** — lane masks / per-lane divergent control flow. **Done:** `PHASE3B.md` (an active-lane set + `Op::SELECT` merge let one walk handle branches/loops that diverge per lane, which Phase 3 rejected with `BatchError`; validated by lane decomposition vs B independent B=1 runs, incl. convergence loops, dead-lane-`Inf`, nested SELECTs, matrix-VJP gating, under clang+ASan+UBSan and g++12). Forward-kernel compaction / per-lane gather deferred.
- **Phase 4** — structural caches (decoded-form cache, inline caches) without program compilation. **Done:** `PHASE4.md` (decoded-form cache + cross-call parse caching + inline variable-lookup cache + `let*` env-flattening + frame/args allocation pooling; byte-identical, ~3.2x on the 80-step Kalman rollout (reused) and ~30x on small co-search programs).
- **Phase 5** — multi-core scheduler (thread-local heaps/tapes, candidate-level parallelism). **Done:** `PHASE5.md` (a population of independent candidate evaluations fans across worker threads, each with a thread-local `Interp` (shared-nothing); byte-identical to serial for any thread count, ThreadSanitizer-clean, ~15x on 16 cores (93% efficiency, median of 5). NUMA / work-stealing deferred).
- **Phase 6** — GPU backend (persistent kernels only). **Design locked + scoped POC done; full backend deferred:** `PHASE6_DESIGN.md`, `PHASE6.md`. A feasibility study found a GPU NDVM beats the 64-core CPU only in the large-D, long-rollout, large-$B$ regime (no committed consumer). The scoped POC (`gpu/kalman_poc.cu`) then measured the *forward numeric ceiling* on an RTX 4090: the dense D-dim Kalman rollout beats a 64-thread CPU by 2--19x (correct to float32), clearing the $\geq$2x bar at all D. That is the upper bound (no interpreter overhead, forward-only); the full D2 interpreter + backward + a committed consumer remain the GO gate. CPU stays primary.
- **Phase 7** — optional MLIR/Enzyme integration.

## Status

Phase 3b complete. The native CPU runtime (`src/sexpr.*`, `src/interp*.cpp`, `src/interp_tape.cpp`) is a
batch-native differentiable interpreter: one structural walk fits B parameter vectors at once, including
control flow that **diverges per lane** (per-lane convergence loops / branches) via an active-lane set +
`Op::SELECT` merge. Forward outputs and reverse-mode gradients match the PyTorch DMCI oracle at B=1 (33
fwd / 82 grad) and per-lane batched (incl. the 80-step Kalman dNLL/dq,dr); divergent control is validated
by lane decomposition (a batched run == B independent B=1 runs), under clang+ASan+UBSan and g++12
(`PHASE1.md`..`PHASE3B.md`). Build with CMake (`build/ndvm_run`; `NDVM_GRAD=1` for gradients, `NDVM_B=<B>`
+ `scalarb` for batches). The PyTorch `autograd.Function` boundary (`python/`, built via `setup.py
build_ext --inplace`, B=1) makes NDVM a differentiable torch op. Gates: `tests/test_equivalence.py` (B=1
vs oracle), `tests/test_batched.py` (uniform batched self-consistency), `tests/test_divergent.py`
(per-lane divergence). Per-lane cost drops ~60x from B=1 to B=256. Next: forward-kernel compaction for
work-optimal divergence, a batched torch boundary, and structural caches.

## Build (Phase 1+, not yet functional)

```
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release
cmake --build ndvm/build -j
```
