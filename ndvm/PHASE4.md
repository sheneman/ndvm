# NDVM Phase 4: structural caches (decoded-form + parse + inline-lookup caches)

Status: **done** -- decoded-form cache (10.1), cross-call parse caching, the inline variable-lookup cache
(10.2), a `let*` env-flattening, and allocation pooling (frame + args). Pure speed: results are
byte-identical to Phase 3b across every suite. CPU, single core, float32.

End-to-end, the 80-step Kalman matrix rollout (the Phase-0 latency blocker) goes from the Phase-3b baseline
**0.638 ms/eval to 0.201 ms/eval reused (~3.2x)** / 0.351 ms re-parsed each call (~1.8x), and a co-search
re-evaluating a small candidate program drops ~30x (e.g. recursive 0.017 -> 0.00054 ms reused).

## Defining-invariant check

This is **memoized decoding of runtime data, not per-program compilation** (design 10.1). The object
program stays runtime S-expression data; we only cache, per AST node, the *syntactic classification* the
interpreter would otherwise recompute on every visit. The evaluator is unchanged; the program is never
residualized into its own graph.

## What was built

`decode()` (`src/interp.cpp`) classifies each AST node once into a `DKind`:
- atoms -> a pre-parsed literal (`DK_LIT_INT`/`DK_LIT_FLOAT`/`DK_LIT_TRUE`/`DK_LIT_FALSE`, value cached) or
  a variable (`DK_VAR`, interned symbol id cached), or the empty list (`DK_NIL`);
- list forms -> a special-form opcode (`DK_SF` + an `SForm`), a group_A primitive (`DK_PRIM_A`), or a
  general application (`DK_APP`).

`eval()` dispatches on the cached `DKind` (a switch) instead of, on every visit, re-running
`strtol`/`strtod` on each atom, re-interning each symbol, walking the special-form string chain
(`sf == "if"`, `== "let"`, ...), and linearly scanning the ~30 group_A primitive names. The cache lives in
three `mutable` fields on `Datum` (`dkind`/`dival`/`dfval`), filled lazily on first eval.

**Why it is safe (byte-identical).** The interpreter's dispatch is purely *syntactic*: special forms and
group_A primitives are matched by name *before* any variable binding is consulted (so e.g. `if` and `+`
are never shadowable as operators), and an atom's literal-vs-variable status is fixed by its spelling. So
a node's decode is context-independent and stable for the AST's lifetime; `decode()` reproduces exactly
the classification `atom_value` + the special-form / `group_A` checks performed. AST nodes are value-typed
and never shared across lexical sites, so a cached decode is never read in a context it was not computed
for. (Single-threaded; a Phase-5 multicore walk must make these cache writes atomic or thread-local.)

## Validation

Byte-identical under **clang + ASan + UBSan** and **g++ 12.1.0 on an HPC compute node**, across every
suite: B=1 equivalence 33/33 forward + 82/82 gradients; batched self-consistency 27/27; divergent
lane-decomposition + raises 21/21; the B=1 PyTorch boundary 28/28 and the batched boundary 28/28 (the
cache is on `eval`, which the boundary drives).

## Speedup (the point)

`NDVM_BENCH` (full `run()` per iteration, B=1), before -> after the decode cache:

| program | baseline ms/eval | cached ms/eval | speedup |
|---|--:|--:|--:|
| `kalman2d_T80` (80-step matrix rollout) | 0.638 | 0.425 | ~1.5x |
| `logistic_map_loop` (16 iters) | 0.0299 | 0.0249 | ~1.2x |
| `factorial_letrec`, `recursive_define` (tiny) | ~0.017 | ~0.019 | flat |

The win scales with how often a node is re-evaluated in one walk: the 80-step Kalman loop body decodes
once and reuses the cache 79x, so the flagship matrix rollout (the Phase-0 latency blocker) drops ~1.5x.
The tiny programs are flat because the bench re-parses the program each iteration and their eval is too
shallow to amortize -- which the cross-call parse cache fixes.

## Cross-call parse caching (done)

The co-search evaluates the same candidate program many times (parameter restarts / optimizer steps), but
the PyTorch boundary previously created a fresh `Interp` and re-parsed + re-macro-expanded the source on
*every* call. Now an `Interp` can **reuse** its parsed + decoded program across forwards:

- `run()` caches the parsed+expanded program by source (`load()` skips parse/expand when the source is
  unchanged), so the decoded-form cache stays warm.
- `begin_forward()` / `reset_state()` clear the per-forward state (numeric arena, environment, heap, tape,
  active set) back to the ctor's initial state while KEEPING the program, its decoded-form cache, and the
  symbol table (so cached `DK_VAR` symbol ids stay valid). A forward is then `begin_forward()` -> re-bind
  params -> `run()`.
- The boundary (`python/ndvm_ext.cpp`) caches one `Interp` per `(source, B)` (bounded, evict-oldest) and
  reuses it, so parse + macro-expand + decode are paid once per program, not per eval. Single-threaded
  (matches the decode cache); a Phase-5 multicore boundary needs per-thread caches.

**Correctness:** a reused `Interp` is byte-identical to a fresh one -- forward outputs AND gradients, for
scalar / recursive / matrix / batched / divergent programs (`tests/test_reuse.py`, 30 cases comparing
`NDVM_REUSE` output to a fresh run bit for bit). All 56 PyTorch boundary tests pass through the reuse path
(including the 15-step Adam descent and the batched self-consistency).

**Speedup** (`NDVM_BENCH_REUSE`, parse paid once, vs `NDVM_BENCH`, re-parse per eval, ms/eval):

| program | re-parse each eval | reuse (parse once) | speedup |
|---|--:|--:|--:|
| `kalman2d_T80` | 0.430 | 0.277 | ~1.6x |
| `logistic_map_loop` | 0.0296 | 0.0051 | ~5.8x |
| `factorial_letrec` | 0.0304 | 0.0018 | ~16x |
| `recursive_define` | 0.0239 | 0.0012 | ~20x |

Combined with the decoded-form cache, the Kalman flagship goes 0.638 -> 0.277 ms/eval (~2.3x), and a
co-search re-evaluating a small candidate program drops ~15-20x.

## Inline variable-lookup cache + let* flattening (done)

The design's "symbol dispatch cache" (10.2): each `DK_VAR` node caches its variable's lexical address
(parent-frame hops, slot) so a lookup jumps to the binding instead of scanning frames. A node's lexical
address is invariant across all its evaluations (frames of the same lexical role are rebuilt structurally
identically; closures capture lexical envs at constant depth; the global env is fixed before the main
eval), so the cached address is reused, guarded by a `binds[slot] == symbol` check that slow-paths +
re-caches on any mismatch. The cache is only ever a validated hint, so it can never change a result.

**Correctness:** cache-ON is byte-identical to the pure scan (`NDVM_NO_INLINE`) on adversarial shadowing /
scoping programs -- nested shadows, an inner binding whose RHS references the outer same-named variable,
sibling scopes, a free variable accumulated across recursion -- and stable across reuse forwards
(`tests/test_inline_cache.py`, 12 cases), on top of the oracle-equivalence suite.

**Honest measurement:** the inline cache itself is **~neutral** in wall-clock (within noise). Caching the
lexical address still walks `hops` parent frames (which cannot be cached -- frames are fresh each eval), and
the per-frame bind scan it removes was already cheap. The real lookup cost is the env *depth*. So the
actual lever was **flattening `let*`**: NDVM's `let` already binds sequentially in ONE frame (each RHS sees
the prior binds), so `let*` now lowers to a single multi-binding `let` instead of nested single-binding
lets -- result-identical (same sequential semantics, same expansion content, the last RHS left raw to match
the oracle) but one frame instead of N. That cut the Kalman's ~6.7 hops/lookup sharply and gave ~7% on the
flagship, speeding up both the cache and the scan. The inline cache is kept (correct, free, completes 10.2,
and useful once the env is flat); `NDVM_NO_INLINE` toggles it for ablation.

## Allocation pooling (done)

The flat lookup ablation pointed at allocation as the remaining cost: each application heap-allocates an
args `vector`, each call/let a `Frame` (whose binds vector allocates), and each numeric result a payload.
Two pools (no semantic change, byte-identical, ASan+UBSan clean under clang and g++):

- **Frame pool**: `frame_top_` tracks the live frame count; `frames_` retains its high-water `Frame`
  objects so their binds-vector capacity persists. `reset_state()` lowers `frame_top_` to 0 instead of
  freeing, so a reused `Interp` (the co-search loop) re-binds into already-allocated frames. ~6% on the
  Kalman.
- **Args pool**: a stack of reusable arg vectors (`args_pool_` / `args_top_`); an application claims one
  (clearing it, keeping capacity) instead of heap-allocating. The fill loop evaluates each argument into a
  temporary *before* indexing `args_pool_[ai]`, because `eval()` can recurse and grow the pool -- an inline
  `args_pool_[ai].push_back(eval(...))` would hold a stale reference across the realloc (the Phase-2
  eval-order trap; the index discipline is verified ASan/UBSan-clean under g++). ~13.5% on the Kalman.

The payload arena needs no pool: `reset_state()` uses `assign`/`clear` which keep capacity, so on a reused
`Interp` the arena does not reallocate after the first forward. Together the pools take the Kalman reused
eval 0.245 -> 0.201 ms (~18%), and roughly HALVE the small co-search programs (they were allocation-bound):
factorial 0.0012 -> 0.00063, recursive 0.00097 -> 0.00054, logistic 0.0052 -> 0.0025 ms/eval.

## Not done (next)

- **Phase 5**: multi-core scheduler (then the decode/var caches, the frame/args pools, and the boundary
  Interp cache all need to be thread-local / per-thread). The single-core perf levers are now largely
  spent; the structural/numeric split + caches + pools have cut the flagship ~3.2x from the Phase-3b
  baseline, and the small co-search inner loop ~30x.

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
# all results byte-identical to Phase 3b:
python3 ndvm/tests/compare_equivalence.py        # 33 fwd / 82 grad
pytest ndvm/tests/test_batched.py ndvm/tests/test_divergent.py
NDVM_BENCH=5000 ndvm/build/ndvm_run prog.scm binds   # avg ms/eval
```
