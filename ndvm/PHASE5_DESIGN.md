# NDVM Phase 5 — Multicore Scheduler: LOCKED Implementation Spec

Status: LOCKED. Implement directly. Validation runs only on the `eight` partition (gcc/12.1.0); the Mac is edit-only.

## 1. Chosen model + justification

**Chosen: D3 (work-stealing-capable pool, thread-local-everything), shipped in the D3 staging order — atomic-index scheduler as the production default, Chase-Lev work-stealing built behind it only if the heterogeneity benchmark justifies the complexity.** The parallelism axis is **candidate-level (outer)**: a population of independent `(program, bindings)` Tasks, each a complete deterministic serial NDVM forward+backward on its own thread-local `Interp`. The already-built B-lane axis (inner) is untouched and composes orthogonally.

Justification: the recon establishes that 100% of mutable engine state is per-`Interp` instance state, so two distinct `Interp`s share nothing. D3's "each worker parses its own AST" makes the two per-Datum cache races (`dkind/dival/dfval`, `vhops/vslot`) **physically impossible rather than merely synchronized** — the cleanest possible answer to the correctness gate, with **zero changes to the hot `eval`/`lookup_var`/`decode` path** and therefore verbatim inheritance of every Phase-4 byte-identical guarantee and single-core perf number. D2's shared-frozen-AST saves W AST copies but buys a freeze flag gating two write sites, an owner/worker lifetime split, a prewarm-completeness obligation coupled to the eval-invariant-lexical-address assumption, and a single hot read-only AST that does not replicate cleanly across NUMA sockets. That memory saving is not worth the new invariants for this workload: parse+expand+decode is amortized across thousands of `begin_forward` reuses per warm worker (the Phase-4 reuse win, now sharded W-way), and the heavy candidates (Kalman ~0.2 ms/eval) dwarf their one-time parse. We take D3's shared-nothing isolation and keep the option (D2's per-NUMA-node read-only replication) available later precisely because shared-nothing makes replication trivial if profiling ever demands it.

## 2. Shared-state race table and the exact fix for each

| # | State | Location | Class | Fix (LOCKED) |
|---|-------|----------|-------|--------------|
| H1 | Boundary Interp cache (`static std::vector<...> cache`) — racing `push_back`/`erase`, can hand the same `Interp&` to two threads | `ndvm/python/ndvm_ext.cpp:21` `cached_interp()` | process-global mutable | Move the cache out of `ndvm_ext.cpp` into a **`thread_local` per-worker cache owned by `WorkerCtx`** in `parallel.cpp`. The free-function `static` cache is **deleted**. Each worker owns its small LRU `map<(src,B)->unique_ptr<Interp>>`, CAP 16. Transitively gives each worker its own parsed AST → resolves H2 for free. This is the single most important change. |
| H2 | Per-Datum decode cache `dkind/dival/dfval` (write-once) and inline-var cache `vhops/vslot` (re-derivable, re-written on slow path) | `ndvm/src/sexpr.hpp:24-34`; written in `Interp::decode()` and `Interp::lookup_var()` (`interp.cpp:71`) | shared-mutable-race **iff AST shared** | **Eliminated, not synchronized.** Each worker's `Interp::load(src)` builds its **own** value-typed `program_hold_` Datum tree (`Datum` is value-typed, deep-copied by `std::vector`). All five mutable fields are written/read by exactly one thread on nodes that thread alone owns. No atomics, no per-node locks, no decode-before-fanout barrier. **No change to `interp.hpp`/`eval`/`lookup_var`/`decode`.** |
| H3 | `g_gensym` file-static counter, `++g_gensym` inside `expand_macros` (reached from `load()`) | `ndvm/src/sexpr.cpp:118-120` | process-global mutable (front-end) | Change to **`static thread_local long g_gensym = 0;`**. Each thread's macro expansion gets its own deterministic sequence; gensym names are internal to one thread's self-contained AST and never cross a thread boundary, so per-thread determinism is preserved. (Note: the counter is not reset per program today, so names already depend on call order within a thread — unchanged semantics, now isolated per worker.) |
| H4 | `InterpError` thrown out of `eval`/`select_merge` (structural divergence, eval-step budget, comparison arity) | engine-wide | exception crossing thread boundary | Each worker **catches `InterpError` and `std::exception` at the task boundary** and marshals the message into `Result{ok=false, err=...}`; **no exception ever unwinds through the pool / `std::thread`.** `ActiveGuard` RAII already restores `active_full_/active_lanes_` on intra-thread unwind, so per-thread exception safety is intact. One bad candidate sets its own slot and does not abort the batch. |
| — | `errno` in `parse_long` | `interp.cpp:210` | per-thread by C standard | No action. |
| — | `group_A`/`group_B`/`sf_opcode`/`head_sym`'s `static const std::string empty`; `materialize_*`/`parse_*` statics | `interp.cpp:256/266/294`, `sexpr.hpp:44` | read-only-after-init | No action — C++11 magic-static thread-safe init; never written after init; all are pure functions. |
| — | All per-Interp engine state (`primal_/adj_/vadj_/pairs_/closures_/vecs_/frames_/args_pool_/actset_pool_/tape_/scalar_params_/sym_ids_/sym_names_/program_hold_/loaded_src_/counters`) | `interp.hpp` Interp class | per-Interp-safe | No action beyond **never sharing one `Interp` across threads.** `begin_forward()`/`reset_state()` clears per-forward state and keeps the warm program/decode; tape+backward are per-Interp. |
| — | Result sink `results[task_id]` | new `parallel.cpp` | disjoint writes | Pre-size `results` to `T` before spawning; each worker writes only its own task's slot. NaN-sentinel init + post-run assert that no slot remains sentinel (catches double-run/skip). |
| — | Co-search population reduction (sum/argmin of fitnesses) | above the engine (driver) | order-dependent aggregate | **Out of engine scope but called out:** any population-level float reduction must run in **fixed task-id order** (sort then reduce, or deterministic tree reduction). The Phase-5 gate covers per-task outputs only. |

The crux is H1+H2: making the boundary cache `thread_local` is what gives each worker its own AST, which is what makes the per-Datum cache fields single-owner. H3 is the one true process-global mutable in the engine and is easy to miss because it lives in the front-end.

## 3. Data-structure + new-file changes

**New files**

- `ndvm/src/parallel.hpp` / `parallel.cpp` — the `Task`/`Result` structs, `evaluate_batch(...)`, and the **`thread_local` per-worker `Interp` LRU cache** (the relocated, race-free replacement for `cached_interp`). Holds the one shared evaluator implementation `eval_one(Interp&, const Task&, Result&)` factored out of the current `eval_and_grad_batched` body so serial and parallel share exactly one code path.
- `ndvm/src/pool.hpp` / `pool.cpp` — the thread pool. **Stage 5a:** a trivial scheduler = persistent `W` threads + a shared `std::atomic<size_t> next` that idle workers `fetch_add` a small chunk from. **Stage 5b (optional):** a Chase-Lev lock-free work-stealing deque per worker (atomic top/bottom, circular array, LIFO own-pop / FIFO steal, randomized victim, atomic done-counter). Persistent threads created once (singleton), reused across `evaluate_batch` calls so warm per-worker `Interp`s survive optimizer epochs.
- `ndvm/tools/ndvm_par.cpp` — threaded CLI driver for the gates: reads a population manifest `{src-file, param-vector, B}`; `NDVM_THREADS=N`; dumps `results[task_id]` as **hex bitcast** (`%08x` float / `%016llx` double) sorted by task_id.

**Task / Result API (LOCKED)**

```cpp
struct Task {
  size_t index;                              // stable result slot -> determinism
  const std::string* src;                    // borrowed; lives for the whole batch
  uint32_t B;
  std::vector<std::string> snames;
  std::vector<float>       svals_flat;       // [P*B], param-major (matches eval_and_grad_batched)
  std::vector<std::string> mnames;
  std::vector<uint32_t>    mrows, mcols;
  std::vector<std::vector<float>> mdata;     // shared matrices, broadcast per lane
  bool want_grad;
};
struct Result {
  std::vector<double> outs;                  // [B]
  std::vector<double> grads_flat;            // [P*B] per-lane d(out_b)/d(param_i)
  bool ok = true; std::string err;           // InterpError captured per task, never thrown across threads
};
std::vector<Result> evaluate_batch(const std::vector<Task>& tasks, int nthreads = 0); // 0 => NDVM_THREADS or hw_concurrency
```

**Thread-local Interp lifecycle (per worker):** pick/insert thread-local `Interp` for `(src,B)` → `Interp::load(src)` (parse-cache hit after first task with that src on this thread) → `begin_forward()` → `bind_scalar_batched`/`bind_matrix` → `set_taping(want_grad)` → `run` → `backward`/`grad_lane` if `want_grad` → write `results[task.index]`. The whole forward+tape+backward pipeline is thread-private.

**Boundary integration** — `ndvm/python/ndvm_ext.cpp`:
- **Delete** the `static cached_interp`. Add `evaluate_batch_py(list-of-task-tuples, nthreads)` that builds `Task`s, **releases the GIL (`py::gil_scoped_release`) around the entire native section**, calls `evaluate_batch`, then re-acquires only to build the returned `list-of-(outs, grads_flat)`. Workers touch no `py::object`. Keep `eval_and_grad` and `eval_and_grad_batched` unchanged for the single-candidate / B-only paths.
- `ndvm/python/ndvm_autograd.py` — `NDVMFunction` gains a **population path** that packs the whole population into ONE `evaluate_batch_py` call. (A driver that still loops one candidate per Python call stays GIL-serialized and sees no speedup — the win requires the caller to batch.)

**Build (CMake threading)** — `ndvm/CMakeLists.txt`:
- `find_package(Threads REQUIRED)`; `target_link_libraries(ndvm PUBLIC Threads::Threads)`.
- `pool.cpp`/`parallel.cpp` are picked up by the existing `file(GLOB src/*.cpp)`.
- Add a `NDVM_TSAN` option → `-fsanitize=thread -fno-omit-frame-pointer` (separate binary; TSan and ASan are mutually exclusive).
- Wire the currently-stubbed `NDVM_ENABLE_PYTHON` so `evaluate_batch_py` builds; add a native pool test target `tools/ndvm_par.cpp`.
- `ndvm/python/setup.py` adds `pool.cpp`/`parallel.cpp` and `-pthread` (`use_ninja=False`; cluster has no ninja). No external deps (Chase-Lev is hand-written `std::atomic` only).

`interp.hpp`/`interp.cpp` internals are **unchanged** except `g_gensym` → `thread_local` in `sexpr.cpp`. Optionally add a comment/`static_assert` documenting the invariant that two `Interp`s never share `Datum`s.

## 4. Determinism guarantee (parallel == serial, byte for byte, per task)

The gate is **exact IEEE-754 bit equality**, not a tolerance. It holds structurally, independent of thread count and schedule:

1. **Tasks are independent and pure.** A task's `Result` depends only on its own `(src, B, params, matrices)`; no task writes state another task reads. There is **no cross-task floating-point reduction** anywhere in the engine, so there is no thread-count-dependent reassociation.
2. **Within a task, the arithmetic is the unchanged Phase-4 serial walk** on a private `Interp`: same float32 ops in the same order, same fixed arena/allocation order, same deterministic tape replay, same per-lane backward seeding. No intra-task parallelism. The B-lane reductions stay inside one worker, unchanged.
3. **Result placement is by `task.index`** into a pre-sized vector — never append-on-completion — so output order is independent of which worker ran which task or finish order.
4. **Per-worker `Interp` reuse is result-neutral:** `begin_forward`/`reset_state` restores ctor-equivalent per-forward state before each task; carryover is impossible. The decode/inline-var caches are write-once classifications / validated re-derivable hints that Phase-4 already proved byte-identical to the uncached scan (`NDVM_NO_INLINE`), and they are now per-worker, so their fill order cannot leak across tasks.
5. **No non-FP nondeterminism:** the eval-step cap is **step-based (`eval_steps_`/`max_eval_steps_`), not wall-clock**; grads are written by **name lookup (`gm.find`), not `unordered_map` iteration order**; symbol interning is per-Interp and deterministic given the same program; no `rand`/`time`/`getenv`/global RNG exists.

Therefore `evaluate_batch(tasks) == [serial eval of each task]` bit-for-bit for any `W` and any steal interleaving. Anything else is a bug (shared-state corruption, a race, or an accidental cross-task reduction) — which is exactly what the gate catches.

## 5. Staging — with a regression gate after every stage

Every stage's hard precondition: **all existing single-core suites stay byte-identical** — `compare_equivalence` (33 fwd / 82 grad), `test_batched` (27), `test_divergent` (21), `test_reuse` (30), boundary suites (56). No stage advances until its gate passes on a compute node.

- **S1 — Plumbing, no threads.** Factor `eval_one()` out of `eval_and_grad_batched`; prove byte-identical (all suites). Relocate the cache: delete `static cached_interp`, add the `thread_local` `WorkerCtx` cache used single-threaded. Change `g_gensym` → `thread_local`. Land CMake `Threads` + setup.py `-pthread`. *Gate: pure-refactor, all existing suites byte-identical.*
- **S2 — `evaluate_batch` at W=1.** Implement `Task`/`Result` + the atomic-index scheduler; run the whole population through it with one worker. *Gate: byte-identical to today's serial loop (isolates pool mechanics from concurrency).*
- **S3 — True multicore + the CORE GATE.** Run `W ∈ {1,2,4,8,16,36}` on `eight` (gcc/12.1.0). *Gates: (a) byte-identical (memcmp of hex outs+grads) for every W vs serial; (b) ThreadSanitizer-clean on the contention stress driver + boundary; (c) near-linear scaling on the Kalman headline.* This stage is the project's correctness gate (byte-identical + TSan-clean + near-linear, on a compute node).
- **S4 — (optional) Work-stealing.** Swap the atomic index for the Chase-Lev pool only if the heterogeneous benchmark (Kalman ~0.2 ms interleaved with tiny ~0.0005 ms candidates) shows static/atomic-index tail imbalance. *Gate: re-run S3 (a)+(b) unchanged — stealing must not alter any per-task result or the ordered `Result` vector.* **Decision rule:** keep the atomic index as the production default and DROP stealing unless measured cache-warmth from contiguous same-src LIFO runs beats the atomic index's perfect balance.
- **S5 — Torch boundary + co-search integration.** Wire `evaluate_batch_py` into `NDVMFunction`'s population path and the OpenEvolve inner-loop population evaluator; run the full boundary suite + a real co-search end-to-end. *Gate: parallel == serial == PyTorch DMCI oracle (`test_autograd_boundary`, `test_autograd_batched`).*
- **S6 — (deferred) NUMA.** Pinning + first-touch behind `NDVM_PIN`/`NDVM_NUMA`, then per-node read-only program replication, only if S3 profiling shows cross-socket falloff. Not on the critical path.

## 6. Validation plan

**Builds (compute node, `module load gcc/12.1.0`):**
- Serial reference: `g++ -std=c++17 -O3 -Isrc src/*.cpp tools/ndvm_run.cpp -o /tmp/ndvm_run_gpp`
- Parallel release: `g++ -std=c++17 -O3 -Isrc src/*.cpp tools/ndvm_par.cpp -o /tmp/ndvm_par -pthread`
- Parallel TSan: `g++ -std=c++17 -O1 -g -fsanitize=thread -fno-omit-frame-pointer -Isrc src/*.cpp tools/ndvm_par.cpp -o /tmp/ndvm_par_tsan -pthread`
- Keep the existing ASan/UBSan single-thread build for serial correctness (mutually exclusive with TSan).
- Torch ext: `.venv/bin/python python/setup.py build_ext --inplace` (`use_ninja=False`).

**Gate 1 — Determinism (`tests/test_parallel_determinism.py`).** Run `ndvm_par` at `NDVM_THREADS ∈ {1,2,4,8,16,36}`; emit one record per task `{src-hash, param-hash, B, hex bitcast of every out lane + grad lane}`, sorted by task-id. Diff each thread-count's dump against `THREADS=1` (exact string equality). Repeat-run stability: `THREADS=16` run **20×**, all identical to each other and to `THREADS=1`. Compare decimal-free hex only (`%08x`/`%016llx`) to expose sub-ULP differences. PASS = every diff empty.

**Gate 2 — Race-freedom (TSan, `tests/test_parallel_stress.py` → `/tmp/ndvm_par_tsan`).** `TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1 history_size=7 exitcode=66"`. Stress harness that maximizes contention: `THREADS = 2× cores` (oversubscribe to force mid-eval preemption); a SMALL set of 4 distinct programs (factorial, logistic, recursive, 80-step Kalman) replicated to ~8192 tasks so many threads hit the same src key at once; a **cold-start burst** (empty caches, all threads first-touch the same key simultaneously); a **cache-thrash** variant (more distinct programs than CAP=16 so eviction fires while others read); Kalman+divergent included so matrix-LU/tape/lane-mask/pools run concurrently. Run 10×. Must cover the relocated cache hit+miss paths, the result sink (disjoint slots), and the atomic work-queue. PASS = exit 0 every run. Pre-flight: `grep -rnE '\bstatic\b' src/ tools/` and confirm no non-const static is written during eval (only `parse_long`/`group_A`/`group_B`/`sf_opcode`/materialize helpers remain, all pure; `g_gensym` now `thread_local`).

**Gate 3 — Cross-check (parallel == serial == oracle).** Diff the `THREADS=1` dump against the existing PyTorch-DMCI-oracle boundary tests (`test_autograd_boundary.py`, `test_autograd_batched.py`) run through the threaded path. Plus all existing serial suites at their locked counts.

**Scaling (`tests/test_parallel_scaling.py`, its own srun for clean PMU).** Sweep `N ∈ {1,2,4,8,16,24,32,36}` + one oversubscribed `N=72` plateau point. Pin (`OMP_PROC_BIND=close OMP_PLACES=cores` or pthread affinity). Metric: candidate-evals/sec; report `S(N)`, `E(N)`. **Headline = 80-step 2×2 Kalman NLL** (16384 tasks, distinct seeds; per-task working set tens of KB → fits L1/L2). Secondary: factorial/recursive (parse-bound end) and a divergent-control program (lane masks under contention). **Bandwidth-artifact controls:** `perf stat -e LLC-loads,LLC-load-misses,mem_load_retired.l3_miss` — LLC-miss must stay flat as N grows (compute-bound); add a deliberately bandwidth-heavy large-B (e.g. B=4096) contrast curve that flattens early to prove we distinguish a bandwidth limit from an engine limit; 64-byte-align/pad the result sink (HITM ≈ 0, no false sharing); discard the first warm-up pass; `lscpu` for socket count, and if 2 sockets add a single-socket curve + `numactl --localalloc`.

**HPC recipe** — one `validate_phase5.sh` inside ONE `srun` on `eight` (because `/tmp` is node-local, build AND test in the same allocation): `srun --partition=eight --time=30:00 --cpus-per-task=36 --mem=16G bash ndvm/validate_phase5.sh`. Use `PY=/mnt/ceph/sheneman/src/nncompile-ndvm/.venv/bin/python` by absolute path (no conda, no source-activate — the repo `.venv` works only on compute nodes; the login node silently falls back to conda base with no torch). `compare_equivalence.py` needs `--ndvm-run PATH`; `test_batched`/`test_divergent` honor `NDVM_RUN`.

**PASS criteria (all must hold):** Gate 1 — every thread-count hex dump identical to `THREADS=1`, 20× `THREADS=16` identical. Gate 2 — TSan exit 0 on all stress variants, 10 repeats. Gate 3 — threaded boundary matches the oracle; serial suites still 33/82, 27, 21, 30, 56. Scaling — `E(N) ≥ ~0.85` to physical cores on Kalman with flat LLC-miss evidence, large-B contrast flattening early.

**Deliverable:** `ndvm/PHASE5.md` reporting the three gates + the scaling table/curves + perf-counter evidence, documenting the `thread_local`-cache relocation + `thread_local g_gensym` as the gated Phase-5 mechanism.

## 7. Residual risks + what the adversarial implementation review must hammer

- **The relocated cache (H1) under cold-start + thrash.** The review must confirm the `static cached_interp` is fully deleted (no second copy lingering in `ndvm_ext.cpp`) and that the `thread_local` `WorkerCtx` cache cannot be reached by two threads. Hammer the cold-start burst (all threads first-touch the same key) and the >CAP eviction-while-read path under TSan.
- **Memory model on the work-queue / done-counter (and Chase-Lev if built).** Atomic-index `fetch_add` must hand out **disjoint** tasks (off-by-one → double-write a slot or skip a task). NaN-sentinel + post-run assert every slot written. If S4 lands, the Chase-Lev deque is genuinely tricky — ABA, top/bottom memory ordering, the last-element steal/pop race — and is the highest-risk code in the phase; the review must scrutinize the memory orderings and run the steal-heavy heterogeneous load under TSan many times.
- **Accidental cross-task sharing.** Assert the invariant that every `Interp` (and its `primal_/adj_/tape_/pools/heap`) is touched by exactly one thread for one task's duration. Any future "optimization" that shares a tape/arena/adjoint/payload table across threads silently breaks byte-identical determinism. TSan + a code-review assert that no `Interp` crosses the work-queue boundary.
- **Exception safety across threads (H4).** Verify no `InterpError`/`std::exception` can unwind through `std::thread`/the pool — every worker task body is wrapped, errors marshaled into `Result`. Confirm `ActiveGuard` restores active-lane state on the intra-thread unwind path before the catch.
- **GIL discipline.** Confirm workers touch no `py::object`; the GIL is released for the whole native section and re-acquired only to build the return tuple. A stray Python touch on a worker thread is a crash/deadlock.
- **Determinism of `g_gensym` as `thread_local`.** Confirm gensym names never cross a thread boundary (each thread's AST is self-contained) and that the byte-identical gate still passes — the counter is not reset per program, so names depend on per-thread call order; the gate is the proof this is fine because each task's AST is independent.
- **Aggregation above the engine.** Out of engine scope but a real foot-gun: the co-search must reduce population fitnesses in **fixed task-id order**. Flag any completion-order float reduction in the driver.
- **NUMA / bandwidth.** On a 2-socket `eight` node the single hot read of the (per-worker, already-replicated) inputs is negligible, but the review must confirm the scaling curve is compute-bound via flat LLC-miss, not a bandwidth fluke, and that per-worker arenas first-touch locally when pinned. NUMA replication stays deferred until measured falloff.
