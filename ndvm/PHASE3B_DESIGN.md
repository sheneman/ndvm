# NDVM Phase 3b — LOCKED Implementation Spec (lane masks / divergent control flow)

Status: **LOCKED.** Supersedes `ndvm/PHASE3B_DESIGN.md` (the candidate). Implement directly against this document. Strategy S1 is confirmed with all blocking-issue fixes folded in. S2 and S3 are rejected (rationale in §1).

---

## 1. Chosen strategy + justification

**Chosen: S1 — active-lane-set gating + a single `Op::SELECT` merge node**, with two corrections that make it implementable (cond lowering to nested-if; SELECT restricted to lane-mergeable values only). The structural walk stays scalar and shared; the arena stays uniformly `B_`-strided; the only thing that narrows under divergence is *which lanes each kernel computes* and *which lanes each tape node replays*.

S1 is correct because every value that escapes a divergent branch does so through exactly one `Op::SELECT`, whose `then_lanes`/`else_lanes` are a **total, disjoint partition of the parent active set**. Consequently (a) no full-active VJP ever reads a branch-only (possibly NaN) inactive-lane primal; (b) the two branch sub-recursions live over disjoint lane universes forever, so no lane's adjoint is double-counted; (c) a shared leaf read in both buckets receives each lane's adjoint from exactly the one bucket that reached it. The reviewer could construct no `0*NaN` or wrong-per-lane counterexample for non-recursive, recursive/convergence-loop, nested, or aliased divergence.

- **vs. S2 (dense compaction / true bucketing):** rejected. S2's only edge is cache-dense work-shrinking inside deep divergent loops, which are NOT the flagship workloads (Kalman, battery, DMCI metacircular runs are uniform-batch, where S2 collapses to S1's identity fast path and never fires). It pays *all* of S1's per-kernel edits and then adds a per-payload width table threaded through the hottest arena accessors (`alloc_payload`, `primal_at`/`adj_at`, `grad_lane`), two coexisting payload width-classes every kernel must classify, an origin-map pool, mixed-width adjoint buffers, and a scatter/gather merge that remaps global lane indices. It breaks the single most load-bearing Phase-3 invariant (uniform `B_`-stride) at the riskiest place. Keep S2's compaction in reserve as a *surgical* optimization if a future profiler shows S1's active-index striding dominates a real divergent-loop hotspot.
- **vs. S3 (oracle-literal freeze-recur-inputs):** rejected as a primary design. S3 is not an alternative to S1 — it is S1's branch core *plus* a loop-specialization layer (it still needs S1 split-and-select for non-loop and inner divergent ifs). It computes every terminated lane's full body every iteration and appends tape nodes for them (tape grows as the *sum* of per-lane iteration counts, not the max), requires the freeze threaded as a per-param stop-gradient SELECT on every loop param (a silent footgun absent from the oracle's fresh-graph re-eval), and re-introduces a loop concept NDVM deliberately erased via a fragile expansion-time marker. Its one genuine advantage (not gating the B-looped kernels) costs work-optimality and structural NaN-safety. **We cite the oracle's `_eval_loop_batched` freeze (`engine.py:1382`) only as the loop-case *semantic anchor* that S1 provably reproduces by lane-decomposition.**

---

## 2. Blocking issues and their resolution

### B1 — SELECT cannot merge per-lane-divergent structural (PAIR/CLOSURE/VEC) results; tag-equality guard is insufficient and untested.

**Root cause (confirmed in source):** a `Val` is `{tag, aux, pid}` (`interp.hpp:24-28`). For `PAIR`/`CLOSURE`/`VEC`, `aux` is a **scalar heap address shared across all B lanes** (`cons` at `interp.hpp:81`, `mk_vec` at `:82`, `make_closure` at `:166`). Only `pid->primal_` and `VecCell.data` are `B_`-strided. So a single `Val` physically cannot carry pair-cell #5 on lanes {0,2} and pair-cell #8 on lanes {1,3}. `v_then.tag == v_else.tag == PAIR` with different `aux` passes the candidate guard yet is unmergeable; it would silently mis-merge (all lanes get one branch's heap address).

**Resolution — strengthen the SELECT guard to FULL structural identity, raise otherwise:**

`Op::SELECT` is defined ONLY for lane-mergeable payloads:
- **Scalars** (`T::INT`/`T::FLOAT`/`T::BOOLEAN`, i.e. `pid`-backed): always mergeable (the `B_`-strided primal is the per-lane carrier).
- **VecCells** (`T::VEC`): mergeable **iff** `v_then.tag == v_else.tag == T::VEC` **AND** `ndim`, `rows`, `cols` are all equal between the two branch results. (Shape must match for the slab route to be well-defined; see §4 and note N-VEC.)

For **PAIR / CLOSURE**, mergeability requires `v_then.aux == v_else.aux` (identical heap cell — the only case where a single scalar `aux` is correct for all lanes). Any divergence (`tag` mismatch, `VEC` shape mismatch, or PAIR/CLOSURE with differing `aux`) raises:

```
InterpError("per-lane-divergent structural value unsupported: a divergent branch
             returns different heap structure / shape across lanes (tag/aux/shape mismatch)")
```

This matches the oracle, which raises at `B>1` for structurally divergent programs, and the §7 boundary "do not batch structurally different programs."

**Mandatory new test (this is the central missing test):** a divergent branch that returns a per-lane PAIR must FIRE the raise rather than silently mis-merge. Minimal form `(if (= t 1) (cons 1 2) (cons 3 4))` batched `t=[1,0]`; loop form `(let ((f (lambda (self acc n) (if (= n 0) acc (self self (cons n acc) (- n 1)))))) (f f (quote ()) k))` batched `k=[2,4]`. Both must raise the structural-divergence error. See §6 test D5.

### B2 — `cond` per-lane divergence has no evaluation mechanism; the clause-loop cannot split, and "native multi-way split" is undefined.

**Root cause (confirmed):** `cond` is kept as a native multi-clause form; the trampoline (`interp.cpp:212-221`) evaluates clause tests sequentially with a single `e = &clause.list[1]` retarget and `break`. A single `Datum*` continuation cannot carry "remaining clauses restricted to the lanes whose test failed." The candidate listed `cond` as an S1a deliverable while leaving its mechanism an open question — a contradiction.

**Resolution — COMMIT to lowering `cond` to right-nested `if`; delete the native multi-way alternative.**

`(cond (t1 e1) (t2 e2) ... (else en))` is treated as
`(if t1 e1 (if t2 e2 (... en)))`, and a `cond` with no `else` terminates the nest with `(boolean #f)` (matching the current `interp.cpp:220` `return boolean(false)`). Implementation choice (pick one, document in code): synthesize the nested-if **in the `cond` handler** by recursing under shrinking active sets (no AST rewrite needed; the handler evaluates `t1` over the active set, splits into `t1!=0` and `t1==0` lanes, evaluates `e1` under the true-lanes and the *tail cond* under the false-lanes, and SELECTs), OR lower in macro-expansion. Either way `cond` reuses **exactly** the if-split + SELECT path — one adjoint-routing op, per-lane test laziness preserved (the `t2` test is only evaluated over `t1`'s false-lanes), and the open question is closed. **"Native multi-way split" is deleted from scope.**

Uniform/all-active `cond` keeps the existing fast trampoline path byte-identical (the split only engages when a clause test is mixed over the active set).

---

## 3. Final data-structure changes (`interp.hpp`)

### 3.1 Active set as engine state (dynamic scope)
Add to `Interp` private state:
```cpp
// Active lane set (dynamic scope). active_full_ == true => all B_ lanes active (fast path).
bool active_full_ = true;
std::vector<uint32_t> active_lanes_;   // sorted lane indices; valid ONLY when !active_full_
```
- `active_full_` is the **fast-path flag**: when true, kernels run the literal `0..B_` loop and `active_lanes_` is ignored.
- Representation: **sorted index list** (not bitset). For B in the hundreds this is cheap and the FULL sentinel covers the common case.

### 3.2 Active-set pool (for the tape)
```cpp
// Pooled, deduped active-lane index lists. id 0 reserved as FULL sentinel.
static constexpr uint32_t ACTSET_FULL = 0;
std::vector<std::vector<uint32_t>> actset_pool_;   // actset_pool_[0] = {} meaning FULL
uint32_t intern_actset(const std::vector<uint32_t>& lanes); // returns FULL when lanes.size()==B_
```
`intern_actset` returns `ACTSET_FULL` whenever the set is all-B (so uniform tapes are byte-identical), else dedups against the pool (a whole bucket shares one id).

### 3.3 `TNode` — a NEW field for the active set; do NOT overload `aux`
`aux` is already load-bearing (`DET`/`LOGDET` cache the inverse-slab VecCell id; `REF` caches the gather index — `interp_tape.cpp:135,138,150`). Add a separate field:
```cpp
struct TNode {
  Op op; Ref out; std::vector<Ref> ins;
  uint32_t aux = NONE;          // unchanged: DET/LOGDET inv-slab id, REF index, SELECT then_lanes id
  uint32_t actset = ACTSET_FULL;// NEW: active-set id at record time (FULL => 0..B_ replay)
};
```
`rec()`/`rec_v()` snapshot `actset = active_full_ ? ACTSET_FULL : intern_actset(active_lanes_)` **at record time** (the same way they snapshot `ref_of(out)`), so nested splits never alias.

### 3.4 `Op::SELECT`
```cpp
enum class Op : uint8_t { ... existing ..., SELECT };
```
- SELECT records **two** disjoint index sets: its own active set = `then_lanes ∪ else_lanes` lives in `TNode.actset` (the routing universe / self-gate); `then_lanes` is interned and stored in `TNode.aux` (SELECT does not use `aux` for a slab, so it is free). `else_lanes = actset \ then_lanes`.
- **Critical:** SELECT backward gates `v_else` accumulation to `(parent active set MINUS then_lanes)`, NOT `0..B_ MINUS then_lanes`. A lane that already terminated at an OUTER select is not in this inner select's `actset` and its adjoint here must stay 0 (verified by trace: nested divergence is correct only if `else = parent-active complement`).

### 3.5 `truthy` signature
`truthy()` is currently `const` and returns a plain `bool` (`interp.hpp:134`, `interp.cpp:60`). It must (a) reduce over the active set only, and (b) expose a per-lane classifier for the split. Replace with:
```cpp
// reduces over the active set: ALL active nonzero -> THEN; ALL active zero -> ELSE; else MIXED.
enum class Branch { THEN, ELSE, MIXED };
Branch classify(const Val& v) const;          // const; reads active set + B-strided payload
// fills then_lanes/else_lanes from the active set by per-lane test (used only on MIXED)
void split_lanes(const Val& v, std::vector<uint32_t>& then_lanes,
                                std::vector<uint32_t>& else_lanes) const;
```
The legacy `bool truthy(const Val&)` is removed; structural/NIL truthiness folds into `classify` (NIL -> ELSE; PAIR/CLOSURE/VEC/SYMBOL -> THEN, lane-uniform). NaN partitions cleanly (`NaN!=0` true -> THEN, `NaN==0` false -> ELSE; no gap).

### 3.6 RAII active-set guard (exception safety — mandatory)
```cpp
struct ActiveGuard {                  // restores active_full_ + active_lanes_ in dtor
  Interp* I; bool saved_full; std::vector<uint32_t> saved_lanes;
  explicit ActiveGuard(Interp* i): I(i), saved_full(i->active_full_), saved_lanes(i->active_lanes_) {}
  void set(std::vector<uint32_t> lanes); // sets active_lanes_, active_full_ = (lanes.size()==B_)
  ~ActiveGuard(){ I->active_full_ = saved_full; I->active_lanes_ = std::move(saved_lanes); }
};
```
All branch recursion sets the active set through `ActiveGuard`, so a thrown structural-divergence error (B1) or any deeper `InterpError` restores the engine's active set before propagating. Manual post-recurse restore is **forbidden** (a throw would leak a shrunken set into engine state and corrupt the next form's `classify`).

---

## 4. Final algorithms

### 4.1 `classify` (replaces `truthy`)
Reduce over the active set only:
```
for each lane b in active set (active_full_ => 0..B_):
    t_b = primal_at(v.pid)[b] != 0       // for numeric/bool; NIL=>false; structural=>true (uniform)
if all t_b true  -> THEN
if all t_b false -> ELSE
else             -> MIXED
```
B=1 and uniform batches are all-active-agree -> never MIXED -> existing fast path. The any_t/any_f reduction is order-independent, so a slow-path active-index iteration yields the identical decision (byte-identical gate safe).

### 4.2 `if` handler (`interp.cpp:171`)
```
Val t = eval(cond, en);
switch (classify(t)) {
  case THEN: e = &then_branch; continue;     // TCO preserved, byte-identical fast path
  case ELSE: e = &else_branch; continue;     // TCO preserved
  case MIXED:
     split_lanes(t, then_lanes, else_lanes); // both non-empty => each a strict subset
     Val v_then, v_else;
     { ActiveGuard g(this); g.set(then_lanes); v_then = eval(then_branch, en); }
     { ActiveGuard g(this); g.set(else_lanes); v_else = eval(else_branch, en); }
     return select_merge(then_lanes, v_then, else_lanes, v_else);  // returns full-active Val, breaks TCO
}
```
The divergent path **returns** (to feed SELECT) — it is correctly a non-TCO point. Uniform `if` keeps `e = branch; continue` and stays trampolined. Each split strictly shrinks the active set by >=1, so nested splits along any root-to-leaf path are <= B-1 deep, independent of iteration count.

### 4.3 `cond` handler
Lowered to nested-if per B2. Uniform path: existing sequential clause trampoline (byte-identical). Mixed clause-0 test: evaluate `e0` under THEN-lanes, recurse the **tail cond** (clauses 1..else) under ELSE-lanes via the same MIXED machinery, then SELECT. No-clause-matches residual returns `boolean(false)` for its lanes.

### 4.4 `select_merge` / `Op::SELECT` forward
```
require lane-mergeability (B1): tag-equal; VEC => ndim/rows/cols equal; PAIR/CLOSURE => aux equal;
        else raise "per-lane-divergent structural value unsupported".
scalar (pid-backed):
   alloc out payload; for b in then_lanes: out[b] = v_then[b]; for b in else_lanes: out[b] = v_else[b]
   (inactive-to-parent lanes, if any nested level, are left stale — never read)
VEC: alloc out VecCell same shape; for b in then_lanes copy v_then's b-slab; for b in else_lanes copy v_else's b-slab
PAIR/CLOSURE (aux equal): return that Val unchanged (no SELECT node needed; structurally identical)
record TNode{ op=SELECT, out, ins={v_then, v_else}, aux=intern_actset(then_lanes),
              actset=intern_actset(then_lanes ∪ else_lanes) }   // only for scalar/VEC
```

### 4.5 `Op::SELECT` backward (in `dispatch_adjoint`)
```
then = actset_pool_[n.aux]; universe = actset_pool_[n.actset]; else = universe \ then
scalar: for b in then: A(v_then,b) += A(out,b);   for b in else: A(v_else,b) += A(out,b)
VEC   : for b in then: VA(v_then,b)[k] += VA(out,b)[k];  for b in else: VA(v_else,b)[k] += VA(out,b)[k]
```
`else` is the **parent-active complement**, not the full-B complement (§3.4). Reverse-replay order `[then-ops][else-ops][SELECT]` is order-robust because then/else write disjoint lanes; SELECT recorded last is replayed first, routing the seed to each branch's source before its subgraph runs. Shared-leaf accumulation is additive-correct because disjoint buckets touch disjoint lanes of the same `B_`-strided `adj_` slot.

### 4.6 Kernel-gating helper (the hard fast-path requirement)
Every forward kernel and every VJP gates at **loop granularity**, never per-element inside the loop:
```cpp
#define FOR_ACTIVE(b)  \
  if (active_full_) for (uint32_t b = 0; b < B_; ++b)  /* existing Phase-3 loop verbatim */ \
  else for (uint32_t b : active_lanes_)
```
- The FULL branch is **textually the existing `for (b=0;b<B_;++b)` body**, preserving reduction order and byte-identity for uniform/B=1.
- Backward uses the node's recorded set: `bool full = (n.actset==ACTSET_FULL); const auto& lanes = full ? {} : actset_pool_[n.actset];` then the same dual loop. `FULL` short-circuits to the existing `0..B_` replay.
- A naive `for(b) if(active[b])` form is **forbidden** (it would preserve numerics but perturb the fast path and reduction-order argument).
- Every VJP case in `dispatch_adjoint` (~40 cases, `interp_tape.cpp:46-154`) MUST adopt the node-active loop. If even one stays `0..B_` it reads an inactive lane's stale/Inf primal and poisons a shared leaf via `+=` (the `0*NaN` trap). This is an implementation obligation across all cases, not a property assertable once.

### 4.7 `backward()` seed invariant
`backward()` (`interp_tape.cpp:30`) seeds all B lanes = 1. This is correct **iff the top-level returned Val is full-active** (every split is re-merged by a SELECT at its own scope before escaping). Add an assertion at `backward()` entry: the output node's active set == FULL (the proposed lowering guarantees it; the assert catches any future op that returns under a reduced set).

---

## 5. Staging (regression gate after EACH stage: B=1 33/33 + 82/82, uniform 27/27, byte-identical, clang AND g++)

- **S0 — plumbing, no behavior change.** Add `active_full_`/`active_lanes_`, `actset_pool_`/`intern_actset`, `ActiveGuard` (RAII), `TNode.actset` (default FULL), `Op::SELECT` enum slot, the `FOR_ACTIVE` helper wired into all forward kernels and all VJPs (FULL branch = existing loops verbatim). `classify`/`split_lanes` added; `if`/`cond` still take THEN/ELSE only (MIXED still raises the existing error). **Gate: byte-identical regression — no tape/primal change for uniform/B=1.**
- **S1a — non-recursive scalar divergence.** `if`/`cond` split on MIXED; `cond` lowered to nested-if (B2); scalar `Op::SELECT` fwd+bwd; B1 structural guard wired (scalar passes, structural raises). **Gate: divergent scalar-branch lane-decomposition test (D1) + structural-divergence raise test (D5).**
- **S1b — recursive / convergence-loop divergence.** The self-passing-closure loop shape under per-lane termination. No new mechanism (reuses S1a split+select+gating); validates termination, the <=B-1 split-depth bound, and active-set replay. **Gate: per-lane convergence-loop lane-decomposition (D2), the NaN/Inf-in-dead-lane stressor (D3), and the nested >2-termination-level test (D4).**
- **S1c — VecCell / matrix divergence.** SELECT over VecCell slabs with the shape check (ndim/rows/cols); matrix VJPs gated. **Gate: matrix-divergent lane-decomposition (D6) + VEC-shape-mismatch raise test.**

---

## 6. Validation gate

**Lane-decomposition method (the proof, since no oracle batches divergence):** for each divergent program and a B-vector of per-lane parameter sets, run (i) one **batched** NDVM run at width B, and (ii) **B independent B=1** NDVM runs (one per lane's params). Assert, for every lane b: forward output and **per-lane gradient** (`grad_lane(pid,b)` for batched == `grad_scalar` of run b) agree to float32 tolerance. Each B=1 run already matches the PyTorch oracle (Phase 1/2), so batched-divergent == oracle by transitivity. Run under **clang AND g++** (g++ caught a Phase-2 eval-order UB clang/ASan missed — validate on the deployment compiler).

**Anchoring note (resolves the masked-path ambiguity):** the chain anchors **solely on the heap-backed tagged path at B=1**. Drop the framing that `_eval_loop_batched` (masked path) is the "reference semantics" — it disagrees with the B=1 tagged path on non-terminating lanes (masked returns stale `result`, `engine.py:1460`; tagged B=1 **raises** after max_iter, `:218`). Cite the masked freeze only as the loop-case intuition.

**Bounded-iteration guard (mandatory):** NDVM `eval` has no max-iter bound (only `eval_steps_`), while the oracle bounds at 10000 (`engine.py:43,218`). Batched divergence trampolines until the LAST lane terminates (= max over lanes), which can exceed a per-lane B=1 count. Add a per-loop iteration cap mirroring the oracle's 10000; a runaway lane must fail loudly (`InterpError`), not hang the batch. Lane-decomposition equivalence then requires every lane to terminate within the bound the B=1 reference uses.

**Coverage assertions (debug build):** (a) per-node active-set coverage — each lane is touched only by nodes on its own branch path; (b) poison inactive payload slots with NaN under a debug flag and confirm clean output (catches a missed kernel/VJP gate); (c) the NaN/Inf-in-dead-lane stressor must exercise **every recorded op type** under divergence, not just arithmetic.

**Specific divergent test programs:**

| ID | Shape | Program (sketch) | Batch | Asserts |
|----|-------|------------------|-------|---------|
| D1 | per-lane scalar branch | `(* x (if (> x 0) x 1.0))` | x=[2,-3] | fwd+grad per lane == 2×(B=1) |
| D2 | convergence loop | Newton/fixed-point `(if (< (abs (- x root)) eps) x (recur step))` | x0 lanes converge at different iters | fwd+grad per lane; depth == B-1 not iter count |
| D3 | NaN/Inf-in-dead-lane stressor | loop whose terminated-lane recur would `1/(x-root)`→Inf | one lane diverges late | clean grads (no `0*NaN`); freeze-class equivalent to oracle |
| D4 | nested >2 termination levels | B=3 convergence loop, iters=[1,2,3] | forces two nested SELECTs | per-lane grad confirms actset dedup/compose across nested SELECTs |
| **D5** | **structural divergence (MUST RAISE)** | `(if (= t 1) (cons 1 2) (cons 3 4))` and the list-accumulator `(f f '() k)` k=[2,4] | t=[1,0] / k=[2,4] | raises "per-lane-divergent structural value unsupported" (NOT silent mis-merge) |
| D6 | matrix divergent | `(if (> s 0) (matvec A x) (matvec B x))` | s=[1,-1] | VEC SELECT fwd+grad per lane; shape check |
| D6′ | VEC shape mismatch (MUST RAISE) | branches return vectors of different length | mixed test | raises structural-divergence error |

Confirm previously-raising MIXED cases now run and match; confirm structural divergence still raises a clear error.

---

## 7. Residual risks + adversarial implementation-review focus

1. **Per-VJP gating completeness (HIGHEST).** All ~40 cases in `dispatch_adjoint` must use the node-active loop. A single forgotten `0..B_` reads a dead lane's stale/Inf primal and poisons a shared leaf via `+=`. Review focus: enumerate every case; the NaN-poison stressor (D3) must hit *every* recorded op type, and the debug NaN-poisoning of inactive slots must produce clean output.
2. **SELECT else-set = parent-active complement, not full-B complement.** Nested divergence is correct only if `else_lanes = actset \ then_lanes` (§3.4). Review focus: a nested D4 trace confirming an outer-terminated lane gets zero adjoint at the inner SELECT.
3. **Structural-divergence guard strictness (B1).** Must check tag AND (VEC: ndim/rows/cols) AND (PAIR/CLOSURE: identical aux), and raise — never silently merge. Review focus: D5/D6′ must raise, not mis-merge. This is the central previously-untested failure mode.
4. **Fast-path byte-identity.** `FOR_ACTIVE` FULL branch must be the existing loop verbatim (loop-granularity dispatch). Review focus: confirm no per-element predicate crept into any hot loop (variadic fold, MUL product, VSUM, MATMUL/MATVEC accumulation, LU); reduction order unchanged; `TNode.actset` FULL short-circuits replay.
5. **Exception-safe restore.** All branch recursion through `ActiveGuard` (RAII). Review focus: a catch-and-continue caller after a thrown structural-divergence error sees an uncorrupted active set.
6. **`cond` lowering correctness + laziness.** Per-lane test laziness: clause-k+1's test evaluated only over clause-k's false-lanes. Review focus: a 3-clause `cond` over 3 disjoint lane groups produces a chain of SELECTs partitioning the active set exactly; no clause evaluated for lanes it does not own.
7. **C++ stack depth (robustness, not correctness).** Depth is bounded by B-1 splits, but a singleton surviving lane in a deep loop recurses native `eval` per iteration after its last split. For large B (low thousands) or very deep single-lane loops this approaches the stack budget. Bound is fine at the B=256 target (~256 frames << 8MB). Optional guardrail: tail-recursive-loop-shape detection to re-trampoline the surviving-lane recursion (O(1) instead of O(B) frames). Not required for S1 correctness; revisit only if a workload overflows.
8. **`classify`/`split_lanes` signature refactor.** Pure refactor (truthy was `const bool`); confirm the B=1/uniform byte-identical claim holds against the new signature.
