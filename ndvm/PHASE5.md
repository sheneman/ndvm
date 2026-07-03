# NDVM Phase 5: multicore scheduler (candidate-level parallelism)

Status: **complete.** A population of independent `(program, bindings)` candidate evaluations runs across a
pool of worker threads, each with a **thread-local `Interp`** -- so 100% of the engine's mutable state is
per-thread and nothing is shared. Each task is a complete deterministic serial NDVM forward + reverse pass;
results are placed by task index, so a parallel run is **byte-identical to a serial one for any thread
count**. The B-lane (batched) axis is orthogonal and untouched: a task may itself be batched, and candidates
are parallel across cores. CPU, float32. Design locked in `PHASE5_DESIGN.md` (chosen over two alternatives
by a design judge-panel + a race-surface recon).

## What was built

- **`src/parallel.{hpp,cpp}`**: `evaluate_batch(tasks, nthreads)` over an atomic-index thread pool. Each
  worker owns a `thread_local` LRU `Interp` cache keyed on `(src, B)` and reused across the tasks it handles
  (warm per-worker program / decode / pools). `eval_one` is the per-task `begin_forward` -> bind -> run ->
  backward pipeline, wrapped so any `InterpError` (structural divergence, eval-step budget) is caught and
  marshaled into `Result{ok=false, err}` -- no exception ever unwinds through a worker thread.
- **`tools/ndvm_par.cpp`**: the determinism + scaling driver (`NDVM_THREADS` / `NDVM_PAR_N` / `NDVM_PAR_DUMP`).
- **PyTorch boundary** (`python/`): `evaluate_batch_py` runs the whole native population with the **GIL
  released** (workers touch no Python object) and returns `[(ok, err, outs, grads)]` in order; the Python
  `evaluate_population(candidates, nthreads)` helper packs a co-search population into one call.
- **CMake**: `find_package(Threads)` + link; the existing source glob picks up `parallel.cpp`.

## The race surface (and why it is closed)

The design recon found that the engine has **no process-global mutable state**, so the entire race surface
is four hazards outside the `Interp`'s own state, all closed:

| # | hazard | fix |
|---|--------|-----|
| H1 | the boundary's process-global static `Interp` cache | now `thread_local` (single-candidate path) + a separate per-worker `thread_local` pool (population path) |
| H2 | the per-`Datum` decode / inline-var caches (race **iff** an AST is shared) | **eliminated by construction**: each worker parses its OWN value-typed `Datum` AST, so the caches are written and read by exactly one thread on nodes it alone owns. **Zero change to the hot `eval`/`lookup_var`/`decode` path** -- the Phase-4 byte-identical guarantees carry over verbatim |
| H3 | `g_gensym` file-static counter (reached from `expand_macros`) | `static thread_local`, and reset per program load so the sequence is schedule-independent |
| H4 | an `InterpError` crossing a thread boundary | caught at the task boundary, marshaled into `Result` |

## Determinism guarantee

The gate is **exact IEEE-754 bit equality** of every task's outputs and gradients, independent of thread
count and schedule. It holds structurally: tasks are independent and pure (no cross-task floating-point
reduction exists in the engine); within a task the arithmetic is the unchanged Phase-4 serial walk on a
private `Interp`; results are placed by `task.index` (never append-on-completion); `begin_forward` /
`reset_state` restores ctor-equivalent per-forward state (including `taping_`) so task N never depends on
task N-1; the eval-step cap is step-based not wall-clock; and gradients are read by name, not map-iteration
order. So `evaluate_batch(tasks)` equals the serial evaluation, bit for bit.

## Validation

All gates pass under **g++ 12.1.0 on the 64-core n128 node**, plus clang + ThreadSanitizer locally.

- **Single-core suites byte-identical**: 33/33 forward + 82/82 gradients, batched 27/27, divergent 21/21,
  reuse 30/30, inline-cache 12/12 (the `g_gensym thread_local` change does not alter any result).
- **Gate 1 -- determinism**: the `ndvm_par` hex bit-pattern dump (4000 tasks: Kalman, plus a divergent
  convergence loop and a deep recursion) is byte-identical for `W in {1, 2, 4, 8, 16, 24, 36}`, and a
  16-thread run repeated 10x is identical each time.
- **Gate 2 -- race-freedom**: ThreadSanitizer reports **0 data races** on the contention stress (72 threads
  oversubscribed, 8192 tasks all first-touching the same program, Kalman and divergent), 10 repeats.
- **Gate 3 -- boundary**: 58 PyTorch-boundary tests pass through the threaded path, including
  `evaluate_population` (parallel) matching serial `ndvm_forward` per candidate (which match the DMCI
  oracle), so parallel population == serial == oracle.

## Scaling (the payoff)

Candidate throughput on the 80-step Kalman NLL with gradients (16000 tasks, n128), tasks/sec:

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| tasks/s | 2008 | 4255 | 8538 | 16129 | 24162 | **31776** | 66003 |
| speedup | 1.0 | 2.1 | 4.3 | 8.0 | 12.0 | **15.8** | 32.9 |

Near-linear to 16 cores (**15.8x = 98.9% efficiency**), and 32.9x at 64 logical cores (the node is ~32
physical + SMT, so this is near the physical-core ceiling). This is the co-search throughput axis: one
worker fits a population of restarts/cells, candidates fan across cores, and the inner B-lane batching
multiplies underneath each.

## Hardening (adversarial race review)

A 4-lens review (data races / memory model; determinism; exception-GIL-resource safety; boundary + the
H1-H4 fixes) found **no blocking issues** and three should-fixes, all folded in: (1) `taping_` was not reset
by `begin_forward`, a latent (currently masked) cross-task carryover -- now reset, and `set_taping` is set
unconditionally per task; (2) the `std::thread` construction loop was unguarded -- now clamped to a sane
ceiling and wrapped so a thread-creation failure degrades to fewer workers (still correct, the atomic pool
self-balances) instead of `std::terminate`; (3) the raw `evaluate_batch` had no bounds check on a task's
`svals`/matrix arrays -- now validated (a malformed task raises a caught error rather than UB). Two
clang-vs-g++ header-portability gaps surfaced in the deployment build (`<system_error>` for
`std::system_error`; a parameter-name typo cmake could not catch because it does not build the ext) --
fixed; **always build the torch extension on the deployment compiler, not only cmake/clang.**

## Cluster lesson

The `eight` partition is **heterogeneous**: some nodes' CPUs are older than the g++ module's default ISA, so
an `-O3` binary built there SIGILLs at runtime (and `-march=x86-64-v2` made it worse, not better -- those
nodes predate v2). Run NDVM on a known-good node; the user's dedicated `sheneman` partition (n128, 64 cores)
is reliable and is what the Phase-5 numbers above use.

## Not done (deferred per the spec's decision rules)

- **Work-stealing** (Chase-Lev): the atomic-index pool is already perfectly balanced for these independent
  tasks; stealing is only worth its complexity if a heterogeneous population shows tail imbalance.
- **NUMA**: thread pinning + first-touch local arenas, deferred until profiling shows cross-socket falloff
  (the per-task working set is tens of KB and fits L1/L2; scaling is compute-bound).
- **GPU backend** (Phase 6): persistent kernels.

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
NDVM_THREADS=16 ndvm/build/ndvm_par prog.scm binds              # population throughput
NDVM_PAR_DUMP=1 NDVM_THREADS=8 ndvm/build/ndvm_par prog.scm binds   # per-task hex (determinism)
pytest ndvm/tests/test_parallel_determinism.py                  # parallel == serial, byte-identical
```
