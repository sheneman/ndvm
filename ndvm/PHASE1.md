# NDVM Phase 1: native CPU forward runtime (forward-equivalent to the DMCI oracle)

Status: **complete**. A native C++ evaluator runs object programs as runtime data and reproduces the
PyTorch DMCI oracle's forward outputs across 21 programs spanning every Phase-0 regime plus the float32
parity edge cases. No autograd yet (the reverse-mode tape is Phase 2). CPU, single core, float32.
The implementation was hardened by a multi-agent adversarial review (semantics, linalg parity, memory
safety, AD readiness); confirmed findings were fixed (see "Hardening" below).

## What was built

- **Front-end** (`src/sexpr.{hpp,cpp}`): S-expression reader + macro expander, faithful ports of
  `neural_compiler/parser/scheme_parser.py` (tokenize/parse) and `neural_compiler/dmci.py`
  (`expand_macros`, `_expand_loop`/`_rewrite_recur`/`_expand_let_star`/`_begin_wrap`). `loop`/`recur`,
  `let*`, `when`/`unless`, `vec`/`mat`, `cond`, and function-`define` lower to the core forms the
  evaluator implements, exactly as the oracle lowers them.
- **VM** (`src/interp.{hpp,cpp}`): the structural/numeric split. Discrete structure (tags, heap
  addresses, interned symbols, closures) is scalar native data; numbers live in a dense payload table
  (primal only in Phase 1). Write-once arena heap (pairs/closures/vectors), environment frames,
  `eval`/`apply` with lazy `if`, sequential `let`, `letrec`, `cond`, `begin`, `define`, closures, and a
  proper **tail-call trampoline** (the eval loop iterates in place on tail positions, so `loop`/`recur`,
  which expands to a self-passing closure, runs in constant stack).
- **Primitives** (`src/interp.cpp`, `src/interp_linalg.cpp`): arithmetic, comparison, logic, math, and
  list/pair ops, plus Strategy-B vectors/matrices (`vec`/`mat`/`ref`/`dot`/`matvec`/`matmul`/`transpose`/
  `trace`/`det`/`logdet`/`inv`/`eye`/`scale`/`zeros`/`ones`/elementwise `+ - * /`). Linear algebra uses
  partial-pivot LU in float32.
- **CLI + harness** (`tools/ndvm_run.cpp`, `tests/`): `ndvm_run` evaluates a program + bindings and (with
  `NDVM_BENCH=N`) times forward evaluation; `tests/oracle_refs.py` generates oracle references on HPC;
  `tests/compare_equivalence.py` + `tests/test_equivalence.py` are the equivalence gate.

## Forward equivalence (the deliverable)

**21/21 programs match the PyTorch DMCI oracle** (torch 2.12.0, float32) within tolerance; `pytest
tests/test_equivalence.py` is green. Coverage: scalar arithmetic and division, transcendentals
(`exp`/`cos`/`log`/`pow`), closures and higher-order application, `letrec` recursion (`factorial`=120
exact), `cond`/`begin`/nested-`let`/list ops, the 16-step logistic `loop`/`recur`, the 80-step 2x2
Kalman-filter NLL matrix rollout, function-`define` recursion, `cross`/`normalize`, and the float32 edge
cases. Most scalar and structural programs match to **0** or ~1e-10; the Kalman rollout matches to **abs
2.3e-7** (float32 LU vs torch/LAPACK); `sqrt(-1)` and `log(-1)` reproduce the `1e-8` input clamp exactly;
`1/0` reproduces `+inf`. (A 22nd program, function-`define` with a free var inside the body, is excluded
because the *oracle's* free-var detection cannot compile it; NDVM evaluates it correctly.)

### Float32 parity rules honored (from the semantics spec)

float32 everywhere (no runtime integer type); truthiness is `payload != 0` for all tags; `<=` is
`not(>)` and `>=` is `not(<)` (NaN-correct); `sqrt`/`log` clamp input to `>=1e-8`; variadic `+`/`*` fold
from identity left-to-right; `modulo`=fmod, `remainder`=floor-mod; `det`/`inv`/`logdet` via LU; env
lookup returns numeric `0` when unbound; `let` is sequential; `eq?` is per-tag.

## A bug worth recording

AddressSanitizer caught an out-of-bounds read on the Kalman path: `prim_compare`/`prim_math`
dereferenced `args[1]`/`args[0]` before confirming the operator and its arity, so unary ops routed
through the dispatch chain (`eye`, `det`, `exp`, ...) read past the argument vector. The scalar programs
had only "passed" on benign uninitialized memory. Fixed by reading arguments only inside confirmed
op branches. Lesson: keep ASan/UBSan in the validation loop (the one place C++ is weaker than the
alternatives), which already paid for itself.

## Forward speed (honest framing)

`ndvm_run` forward time (parse + macro-expand + eval per run, single core) vs the DMCI Phase-0 baseline:

| program | NDVM (ms) | DMCI baseline (ms) |
|---|--:|--:|
| scalar `(+ (* a x) b)` | 0.014 | 4.37 |
| damped oscillator | 0.014 | 14.61 |
| logistic loop (16) | 0.029 | 229.08 |
| Kalman NLL (T=80) | 0.67 | 10246.71 |

These ratios (about 300x to 15000x) are large but **conflate two different wins** and must not be read
as the design's "2-5x sequential" projection. NDVM here runs the object program *directly*: it is a
native differentiable interpreter, so it does not need the meta-circular tower DMCI uses to obtain
program-as-data and gradients. So the gap is (native vs Python) AND (direct vs meta-circular tower). The
honest like-for-like number (native runtime executing `compiler.scm` meta-circularly vs the Python
meta-circular backend) is the deferred metacircular-path measurement and is the right number to quote as
an interpreter speedup. We report the table above as an end-to-end use-case speedup, with that caveat.

## Hardening (adversarial review outcomes)

A 19-agent review (4 review dimensions, then per-finding verification) produced 21 findings, 10
confirmed. Fixed in this phase: `cross`/`normalize` were dispatched but unimplemented (fell through to
0) and are now implemented (`normalize` clamps the norm to `1e-8` like the oracle); `ref` now
bounds-checks its index (a negative/large float index previously truncated to a huge `uint32_t` and read
out of bounds, an ASan-confirmed UB); `prim_compare` guards argument count; and `and`/`or` were removed
because `bootstrap/compiler.scm` does not implement them, so NDVM must not be more permissive than the
oracle. The remaining confirmed item is documented below.

## Known limitations

- **Structural-value truthiness.** NDVM treats every pair/closure/vector/symbol as truthy (standard
  Scheme). The oracle instead reads `payload[0]` (a heap address / interned id), so a structural value
  whose address/id is `0` is falsy there. This is an artifact of the oracle's float-payload encoding,
  is reachable only by using a raw structural value directly as an `if`/`cond` test (non-idiomatic; not
  in the validated set; idiomatic tests go through `=`/`<`/`null?`/`pair?` which return BOOLEAN), and is
  not cheaply matchable because NDVM numbers its heaps differently from the oracle. Documented, not
  reproduced.
- Single-core, float32, forward-only. No batching, no GPU, no autograd yet.

## Phase-2 contract changes (from the AD-readiness review, to do before/with Phase 2)

These are design notes, not Phase-1 bugs. To add reverse-mode AD without a rewrite:
- Replace the flat `primal_` float buffer with a `Payload{shape, primal_offset, adjoint_offset,
  tape_birth}` table; `Val.pid` becomes a payload id. This carries shape + adjoint storage per numeric
  value.
- Add a define-by-run tape (`TapeOp{opcode, in-payload-ids, ...}`) recorded by each numeric primitive
  only; structural ops emit nothing. Add a `reverse_pass()` that replays it into the adjoint buffer.
- Store boolean results as a `0.0/1.0` float payload (not in `aux`) so gradients can flow through
  comparison results as trace-constants.
- Extend `VecCell` with a batch axis (`[B, rows*cols]`) for the Phase-3 batch-native payloads.

## Build and run

```bash
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release && cmake --build ndvm/build -j
ndvm/build/ndvm_run <program.scm> [bindings]        # bindings: "scalar name v" / "matrix name r c v..."
python3 ndvm/tests/oracle_refs.py                   # on an HPC compute node (needs torch)
pytest ndvm/tests/test_equivalence.py               # 19/19 forward-equivalence vs the oracle
```

## Not done in Phase 1 (next)

- **Phase 2:** native reverse-mode tape (adjoint buffer + define-by-run tape over numeric primitives
  only) and the PyTorch autograd boundary; gradient equivalence vs the oracle + finite differences.
- **Phase 3:** batch-native payloads (the co-search throughput multiplier).
- The like-for-like metacircular forward-speed measurement (run `compiler.scm` natively).
- GPU (Phase 6) and MLIR/Enzyme (Phase 7).
