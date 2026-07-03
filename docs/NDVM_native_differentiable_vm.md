# NDVM future-work design (recorded 2026-06-19)

Status: FUTURE WORK, targeting a separate paper. Not part of the current ICLR submission
(`paper-iclr/`). This is the concrete design for the "interpreter speedup" track: a native,
batching-first, reverse-mode differentiable VM for the compiled DMCI evaluator. It is the
performance-engineering alternative to Futamura specialization (it keeps "program as data";
it does NOT residualize each candidate into its own graph). It is also the de-risk gate for a
future turbulence-ROM flagship (which the 2026-06-18 assessment found compute-blocked by the
current ~250 ms/step interpreter). Grounding cost model: the DMCI perf autopsy found forward time
is ~61% tagged-value boxing, ~25% graph-walking, with raw arithmetic + autograd ~1%.

---

# NDVM: A Native Differentiable Virtual Machine for DMCI

## 1. Objective

Build a hyper-efficient Native Differentiable Virtual Machine, NDVM, for executing the compiled DMCI Scheme evaluator with native performance while preserving the defining DMCI property:

```text
compile evaluator once
supply arbitrary object programs as data
differentiate through the evaluator to continuous parameters
avoid per-program recompilation or hand-written gradients
```

NDVM is not a Futamura-specializing compiler and does not residualize each candidate program into a unique computation graph. It is a high-performance execution engine for the already-compiled evaluator. Its purpose is to replace the current Python/PyTorch eager runtime with a native, batching-first, reverse-mode differentiable VM that makes tags, heap addresses, environments, and closures cheap while keeping numeric payloads differentiable and efficiently batched.

The target is not merely "faster Python." The target is a new runtime representation:

```text
structural data: native scalar values
numeric data: dense differentiable payload buffers
control flow: exact realized trace
AD: compact native reverse-mode tape
batching: first-class execution dimension
```

The expected result is a system that preserves DMCI's "compile once, differentiate everywhere" semantics while reducing single-evaluation overhead by roughly one order of magnitude and improving batched population fitting by several-fold to tens-fold over the current PyTorch backend.

---

## 2. Core design invariants

NDVM must preserve the following invariants.

### 2.1 Interpreter-level differentiability

The differentiable object remains the compiled evaluator, not each object program. A Scheme object program remains symbolic data consumed by the evaluator.

Allowed:

```text
compile Scheme evaluator to NDVM bytecode
execute arbitrary S-expression programs as runtime data
cache structural lookups and decoded forms
batch repeated runs of the same object program
```

Not allowed as a default execution model:

```text
compile each object program to a residual differentiable graph
require retracing per candidate
generate program-specific native code for each AlphaEvolve candidate
```

Optional hot-path specialization may exist later as a cache-tier optimization, but NDVM's baseline semantics must work without it.

### 2.2 Exact trace semantics

NDVM differentiates the realized execution trace. Branches, tags, symbols, heap addresses, and dispatch are discrete. Numeric payloads are differentiable. Gradients are correct on trace-constant regions and inherit the source program's nondifferentiability at branch boundaries.

NDVM must therefore support:

```text
lazy conditionals
data-dependent evaluator dispatch
variable-length loops
trampolined tail calls
runtime heap allocation
closures
association-list or frame-based environments
```

A purely static XLA-style graph is insufficient for the general evaluator unless it is wrapped in masking or scan transformations. NDVM should instead implement define-by-run reverse mode natively.

### 2.3 Structural/numeric split

Runtime values must separate structural identity from differentiable payloads.

Current PyTorch-style tagged tensor representation:

```text
Value = [one-hot tag | payload]
```

is useful for proof and backend uniformity, but inefficient as an execution representation. NDVM should instead use scalar tags and dense payload buffers.

---

## 3. Architecture overview

```text
Scheme evaluator source
        ↓
DMCI compiler
        ↓
ComputeGraph for evaluator
        ↓
NDVM lowering
        ↓
Evaluator bytecode / native graph
        ↓
NDVM runtime
        ↓
forward result + native AD tape
        ↓
reverse pass
        ↓
gradients with respect to bound numeric parameters
```

Object programs enter as runtime S-expression data:

```text
object program P(theta)
        ↓
compact symbolic heap / object heap
        ↓
compiled evaluator running on NDVM
        ↓
loss
        ↓
reverse-mode gradients to theta
```

The evaluator may be compiled once to NDVM bytecode or native code. The object program is not compiled into a unique graph by default.

---

## 4. Runtime value representation

### 4.1 Native tagged values

Use a compact scalar value representation:

```cpp
enum class Tag : uint8_t {
    NIL,
    BOOL,
    INT,
    FLOAT,
    CHAR,
    SYMBOL,
    PAIR,
    STRING,
    CLOSURE,
    VECTOR
};

struct Value {
    Tag tag;
    uint32_t aux;
    uint32_t payload_id;
};
```

Interpretation:

```text
tag        type discriminator
aux        symbol id, heap address, closure id, vector id, immediate int, etc.
payload_id index into differentiable payload table, if numeric
```

For non-numeric values, `payload_id = INVALID`.

This eliminates tensor allocation for symbolic values, tag one-hot construction, tag slicing, and tensor concatenation.

### 4.2 Differentiable payload table

Numeric payloads live in dense storage:

```cpp
struct Payload {
    DType dtype;
    Shape shape;
    uint32_t primal_offset;
    uint32_t adjoint_offset;
    uint32_t tape_birth;
};
```

Payload memory is structure-of-arrays:

```text
float primal_buffer[]
float adjoint_buffer[]
```

For batched execution:

```text
payload primal shape = [B] or [B, ...]
payload adjoint shape = [B] or [B, ...]
```

A scalar structural `Value` can point to a batched numeric payload. Thus tags and heap addresses remain scalar while numeric data carries the batch dimension.

This is the critical representation change.

Instead of:

```text
B × 14 values per runtime object
```

NDVM stores:

```text
one scalar tag/address object
B numeric payload elements only when needed
```

### 4.3 Immediate values

Small integers, booleans, nil, symbols, and characters should be immediate where possible:

```cpp
Value make_bool(bool b);
Value make_symbol(SymbolId s);
Value make_int(int32_t i);
```

Only differentiable numeric values receive payload-table entries.

### 4.4 Tag discipline

Tags are never differentiated. Tag checks are native integer comparisons. Tag errors produce VM exceptions or guarded slow-path behavior.

This preserves the DMCI theoretical separation:

```text
structure is discrete
payloads are differentiable
```

---

## 5. Heap and environment representation

### 5.1 Immutable arena heap

Because the DMCI subset excludes mutation, use a write-once arena heap.

```cpp
struct PairCell {
    Value car;
    Value cdr;
};

struct ClosureCell {
    uint32_t code_ptr;
    Value env;
};

struct VectorCell {
    uint32_t start;
    uint32_t length;
};

struct Heap {
    Arena<PairCell> pairs;
    Arena<ClosureCell> closures;
    Arena<VectorCell> vectors;
    Arena<Value> vector_elements;
};
```

`cons` appends a `PairCell` and returns `Value{PAIR, addr, INVALID}`. `car` and `cdr` perform direct arena loads.

This avoids PyTorch in-place mutation hazards entirely. Autograd is handled by NDVM's native tape, not by versioned tensor buffers.

### 5.2 Environment frames

The self-hosted evaluator may express environments as association lists. NDVM should support that representation faithfully, but internally add optional fast paths.

Baseline:

```text
environment = Scheme list of symbol/value pairs
lookup = interpreter-defined association search
```

Optimized runtime caches:

```cpp
struct EnvShape {
    uint64_t shape_id;
    uint32_t parent_shape_id;
    SymbolId symbols[];
};

struct EnvCacheEntry {
    SymbolId symbol;
    uint64_t shape_id;
    uint32_t depth;
    uint32_t offset;
};
```

Lookup remains semantically association-list lookup, but repeated lookups can be accelerated by inline caches.

### 5.3 Symbol interning

All symbols are interned to `uint32_t`.

```text
'+       → primitive id 3
'if      → special form id 4
'lambda  → special form id 5
'alpha   → symbol id 211
```

No runtime string comparisons occur in hot paths.

---

## 6. Bytecode and execution engine

### 6.1 Lowering ComputeGraph to NDVM bytecode

The compiled evaluator ComputeGraph should lower to a compact bytecode or direct-threaded instruction stream.

Example instruction classes:

```text
LOAD_CONST
LOAD_INPUT
MAKE_VALUE
GET_TAG
ASSERT_TAG
CONS
CAR
CDR
MAKE_CLOSURE
ENV_LOOKUP
CALL
TAIL_CALL
RETURN
IF
PHI
ADD_F
MUL_F
DIV_F
EXP_F
LOG_F
SQRT_F
LT_F
EQ_TAG
EQ_SYMBOL
TAPE_MARK
```

The object program is data in the heap. The evaluator bytecode is fixed.

### 6.2 Direct-threaded interpreter

Initial implementation should use a switch dispatch or computed-goto/direct-threaded interpreter.

```cpp
while (running) {
    switch (*pc++) {
        case OP_ADD_F: ...
        case OP_CONS: ...
        case OP_CAR: ...
    }
}
```

Later, evaluator bytecode can be AOT-compiled to C++ or MLIR once. This still compiles only the evaluator, not object programs.

### 6.3 Evaluator graph unrolling

Unroll the evaluator ComputeGraph, not the object program.

Valid:

```text
ComputeGraph(evaluator) → fixed NDVM instruction stream
```

Avoid:

```text
object program P → residual graph P*
```

The first preserves DMCI. The second reintroduces per-program compilation.

### 6.4 Superinstructions

Fuse common bytecode sequences into superinstructions:

```text
UNWRAP_FLOAT_ADD_REBOX
UNWRAP_FLOAT_MUL_REBOX
CAR_CDR_SYMBOL_EQ
ENV_LOOKUP_FLOAT
PRIMITIVE_NUMERIC_APPLY
CONS2
LIST_REF_2
TAG_TEST_BRANCH
```

These eliminate dispatch overhead and reduce tape emission overhead.

Example:

```cpp
Value op_add_float(Value a, Value b) {
    PayloadId pa = require_float(a);
    PayloadId pb = require_float(b);
    PayloadId out = alloc_payload_like(pa, pb);
    vector_add(payload[out], payload[pa], payload[pb]);
    tape.push({AD_ADD, out, pa, pb});
    return Value{Tag::FLOAT, 0, out};
}
```

---

## 7. Native reverse-mode AD

### 7.1 Define-by-run native tape

NDVM should implement PyTorch-like define-by-run reverse mode, but without PyTorch tensor-object overhead.

Forward execution appends compact tape records for differentiable primitives only.

```cpp
enum class ADOpc : uint16_t {
    ADD,
    SUB,
    MUL,
    DIV,
    NEG,
    EXP,
    LOG,
    SQRT,
    POW,
    SIN,
    COS,
    MIN_REALIZED,
    MAX_REALIZED,
    SELECT,
    DOT,
    MATMUL,
    REDUCE_SUM,
    COPY
};

struct TapeNode {
    ADOpc op;
    PayloadId out;
    PayloadId in1;
    PayloadId in2;
    uint32_t aux;
    MaskId mask;
};
```

Non-differentiable structural operations do not emit tape nodes:

```text
symbol lookup
tag check
pair allocation
car/cdr
closure construction
environment lookup
branch dispatch
```

They affect control but not gradient propagation except by choosing the realized trace.

### 7.2 Backward pass

Backward replay:

```cpp
for node in reverse(tape):
    dispatch_adjoint(node)
```

Examples:

```text
ADD:
    adj[in1] += adj[out]
    adj[in2] += adj[out]

MUL:
    adj[in1] += adj[out] * primal[in2]
    adj[in2] += adj[out] * primal[in1]

EXP:
    adj[in1] += adj[out] * primal[out]
```

Use vectorized loops over batch dimensions.

### 7.3 Adjoint bytecode

For performance, represent backward operations as adjoint bytecode rather than generic autograd objects.

Forward primitive emits a backward instruction:

```text
F_MUL out, a, b
    emits B_MUL out, a, b
```

Backward bytecode can be fused, vectorized, and replayed efficiently.

### 7.4 Tape memory management

Use arena allocation for tape nodes.

```cpp
struct TapeArena {
    TapeNode* nodes;
    uint32_t size;
    uint32_t capacity;
};
```

For OpenEvolve workloads:

```text
one tape per candidate × restart batch
thread-local tape arenas
reuse arenas across optimizer epochs
```

For GPU:

```text
forward persistent kernel writes tape to global memory
backward kernel replays tape in reverse
```

### 7.5 Checkpointing

Long recursive programs may generate large tapes. Support checkpointing:

```text
store selected primal checkpoints
recompute forward segments during backward
reduce memory at cost of extra compute
```

This is critical for large-memory GPU workloads.

---

## 8. Control flow, lazy branches, and trace semantics

### 8.1 Lazy conditionals

NDVM must preserve lazy branch evaluation.

```text
IF cond then_expr else_expr
    evaluate cond
    evaluate only selected branch
```

Eager evaluation is invalid for general recursive DMCI programs because it can expand untaken recursive branches.

### 8.2 Trace-constant gradients

Branch predicates are discrete routing decisions. A parameter used only in a branch predicate receives no gradient through that predicate. A parameter used in the selected differentiable branch receives ordinary gradients.

NDVM should expose branch boundary diagnostics:

```text
parameter appears only in predicate
branch flipped during optimization
batch lanes diverged
trace instability score
```

These diagnostics are useful for debugging co-search candidates.

### 8.3 Tail calls and trampoline

Implement tail calls as a VM primitive, not as Python recursion.

```cpp
struct Frame {
    uint32_t return_pc;
    Value env;
    uint32_t stack_base;
};

TAIL_CALL:
    replace current frame
    jump to callee
```

Proper tail recursion should be stack-stable.

---

## 9. Batching model

### 9.1 Batch-native payloads

Batching is a first-class NDVM feature. Structural values are scalar; payloads are batched.

```text
Value tag: FLOAT
Value payload_id: 17
Payload[17].primal: float[B]
Payload[17].adjoint: float[B]
```

This allows one evaluator walk to operate over many parameter vectors, restarts, cells, or input points.

### 9.2 Batch axes

Support named axes:

```text
B_data       input points
B_restart    optimizer restarts
B_cell       per-cell fits
B_population population members
```

Internally flatten compatible axes:

```text
B = B_data × B_restart × B_cell
```

Do not batch across structurally different object programs by default. Batch within one candidate program or within trace-compatible groups.

### 9.3 Lane masks

For data-dependent numeric branches, support lane masks.

```cpp
struct Mask {
    uint64_t* bits;
    uint32_t nlanes;
};
```

If all lanes agree, take one branch.

If lanes diverge:

```text
then_mask = active & cond
else_mask = active & ~cond
execute then branch under then_mask
execute else branch under else_mask
merge
```

This preserves exact semantics while allowing partial batching.

### 9.4 Trace bucketing

For severe divergence, bucket lanes by trace signature.

```text
trace_hash = hash(branch decisions, primitive dispatches)
```

Execution strategy:

```text
run lanes sharing trace_hash as dense sub-batch
avoid mask fragmentation
```

This borrows from GPU ray tracing, SIMT divergence handling, and vectorized database execution.

### 9.5 Population batching

For OpenEvolve-style co-search, the most important batching pattern is:

```text
one candidate program
many parameter initializations
many cells
many input sequences
```

NDVM should make this cheap:

```text
structural program heap loaded once
evaluator walk performed once per batch
numeric payload arrays carry all restarts/cells
native tape records vectorized operations
```

---

## 10. Structural caching without program compilation

NDVM should aggressively cache structure while preserving the "program as data" model.

### 10.1 Decoded S-expression cache

For a fixed object program, cache decoded form metadata:

```cpp
struct DecodedForm {
    HeapAddr addr;
    FormKind kind;
    SymbolId operator_symbol;
    PrimitiveId primitive;
    HeapAddr operands;
};
```

This avoids repeated `car/cdr` chains and symbol comparisons.

This is not per-program graph compilation. It is memoized decoding of runtime data.

### 10.2 Inline caches

Use VM-style inline caches:

```text
symbol lookup cache
primitive dispatch cache
environment slot cache
closure application cache
special-form cache
```

Example:

```cpp
if (env.shape_id == cached_shape && symbol == cached_symbol) {
    return env.slots[cached_offset];
} else {
    return slow_env_lookup(symbol, env);
}
```

### 10.3 Branch prediction

Maintain branch-history metadata:

```text
branch site id
last outcome
stability counter
taken ratio
lane divergence rate
```

For stable branches, execute predicted branch first and verify guard. On guard failure, discard speculative result and execute correct branch.

This is optional and should be added only after the baseline VM is correct.

---

## 11. CPU execution strategy

### 11.1 CPU-first baseline

The first production NDVM should be CPU-first. The workload is branchy, heap-heavy, and dominated by small operations. Current GPU execution suffers when many tiny operations become separate kernel launches. A native CPU runtime can remove Python and tensor-object overhead while exploiting SIMD over batch payloads.

### 11.2 SIMD vectorization

Use vectorized kernels over payload buffers:

```text
AVX2
AVX-512
ARM NEON/SVE where appropriate
```

Arithmetic primitives operate over contiguous `float[B]` buffers.

```cpp
for i in simd_range(B):
    out[i] = a[i] * b[i] + c[i];
```

### 11.3 Multi-core parallelism

Use two levels of CPU parallelism:

```text
outer: candidate programs / islands / structures
inner: batch lanes / restarts / cells
```

A scheduler should assign:

```text
one candidate fit per worker
thread-local heap and tape
shared immutable program data
shared symbol table
```

For a 64-core node, the ideal pattern is:

```text
64 candidate fits in parallel
each candidate uses SIMD batching over restarts/cells/data
```

### 11.4 NUMA awareness

For large population-batched workloads:

```text
pin worker threads
allocate tape/payload arenas on local NUMA node
avoid cross-socket payload writes
replicate read-only program heaps if necessary
```

---

## 12. GPU execution strategy

### 12.1 GPU only where it wins

Do not implement GPU as one kernel per VM primitive. That reproduces the PyTorch eager failure mode.

A useful GPU NDVM requires persistent kernels:

```text
one launch per rollout or large segment
many VM steps inside the kernel
```

### 12.2 Persistent GPU interpreter kernel

Possible mapping:

```text
one CUDA block = one candidate program or trace bucket
threads/warps = batch lanes
shared memory = hot structural metadata
global memory = payload buffers and tape
```

Forward:

```text
persistent kernel executes evaluator loop
numeric primitives operate across lanes
tape nodes written to global memory
```

Backward:

```text
reverse kernel replays tape
adjoints accumulated into payload buffers
```

### 12.3 SIMT divergence

Use lane masks for branches.

```text
warp ballot for branch condition
if all lanes agree: execute one branch
else: execute both branch paths under masks or split trace bucket
```

Trace bucketing should be used for high-divergence workloads.

### 12.4 GPU tape representation

GPU tape nodes must be compact and coalesced.

```cpp
struct GpuTapeNode {
    uint16_t op;
    uint16_t flags;
    uint32_t out;
    uint32_t in1;
    uint32_t in2;
    uint32_t aux;
    uint32_t mask_id;
};
```

Tape storage should use append offsets allocated per block or per trace bucket to avoid global atomics in hot loops.

### 12.5 GPU suitability

GPU backend is appropriate for:

```text
large batch sizes
population-scale fitting
matrix/vector-heavy primitives
long rollouts with repeated numeric kernels
many cells/restarts per candidate
```

GPU backend is poor for:

```text
single scalar evaluation
highly branch-divergent traces
short programs
control-heavy symbolic manipulation
```

The CPU backend should remain the primary universal runtime.

---

## 13. Optional MLIR/Enzyme path

NDVM has two plausible AD implementation routes.

### 13.1 Hand-written native tape

Pros:

```text
full control
predictable performance
works naturally with VM semantics
easy to preserve realized-trace AD
```

Cons:

```text
must implement adjoints manually
primitive coverage burden
```

### 13.2 LLVM/MLIR + Enzyme

Compile the evaluator VM or numeric primitive layer to LLVM/MLIR and apply Enzyme-style AD.

Possible architecture:

```text
NDVM numeric kernels
        ↓
LLVM/MLIR
        ↓
Enzyme
        ↓
generated adjoint kernels
```

Do not rely on Enzyme to differentiate arbitrary interpreter control initially. Use it first for numeric kernels and fused primitive regions.

Long-term possibility:

```text
native evaluator implementation
        ↓
LLVM
        ↓
Enzyme differentiates evaluator execution
```

This is scientifically interesting but higher-risk. The lower-risk route is manual tape plus optional Enzyme-generated adjoints for dense numeric kernels.

---

## 14. API boundary

Expose NDVM as a Python extension while keeping execution native.

### 14.1 Python API

```python
vm = ndvm.compile_evaluator(compute_graph)
program = ndvm.load_program(sexpr)
params = ndvm.make_parameters(...)
result = vm.forward(program, params, inputs)
loss = loss_fn(result, target)
grads = vm.backward(loss_grad)
```

### 14.2 PyTorch integration

Provide a custom autograd function:

```python
class NDVMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, program_handle, params, inputs):
        y, tape_handle = ndvm.forward(program_handle, params, inputs)
        ctx.tape_handle = tape_handle
        return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_params, grad_inputs = ndvm.backward(ctx.tape_handle, grad_y)
        return None, grad_params, grad_inputs
```

This allows external optimizers and experiment code to remain PyTorch-compatible while the interpreter runtime is native.

### 14.3 Serialization

Support cached artifacts:

```text
compiled evaluator bytecode
symbol table
primitive table
object program heap
decoded-form cache
profiling metadata
```

---

## 15. Correctness and validation

### 15.1 Forward equivalence

Validate NDVM against current PyTorch DMCI:

```text
same program
same inputs
same parameters
same output within tolerance
```

Coverage:

```text
arithmetic
closures
recursion
letrec
pairs/lists
higher-order functions
tail calls
program-generated data structures
```

### 15.2 Gradient equivalence

Validate against:

```text
current DMCI PyTorch backend
direct compilation where available
finite differences on small examples
```

Metrics:

```text
relative gradient error
cosine similarity
loss trajectory identity
optimizer convergence epoch
branch trace agreement
```

### 15.3 Trace diagnostics

Record:

```text
branch decisions
tag dispatch sequence
heap allocation count
tape length
payload allocation count
mask divergence
cache hit rates
```

These diagnostics are essential for debugging branch-boundary and batching behavior.

### 15.4 Determinism

NDVM should support deterministic execution modes:

```text
fixed allocation order
stable symbol interning
deterministic reduction order
reproducible tape replay
```

---

## 16. Performance model

Let current sequential DMCI time be:

```text
T_current = T_box + T_python + T_graphwalk + T_heap + T_dispatch + T_arith + T_autograd
```

Empirically, most time is representation and graph walking.

NDVM targets:

```text
T_box       → near zero
T_python    → zero in hot path
T_graphwalk → native bytecode / AOT evaluator
T_heap      → arena loads/stores
T_dispatch  → direct-threaded VM + superinstructions + inline caches
T_arith     → SIMD/GPU numeric kernels
T_autograd  → compact native tape
```

Expected sequential improvement:

```text
current overhead over direct compile: ~14x
NDVM target overhead: ~1.5x to 3x
speedup over current sequential DMCI: ~5x to 11x
```

Expected batched improvement over current batched PyTorch DMCI:

```text
scalar closed-form programs:        1.5x to 4x
recursive/loop-heavy programs:      3x to 10x
training/backward-heavy workloads:  4x to 15x
OpenEvolve inner fitting:           5x to 30x
matrix-heavy GPU workloads:         10x to 100x possible
```

NDVM should not promise to beat specialized direct compilation or JAX/XLA on simple closed-form equations. Its advantage is:

```text
coverage + no per-program compilation + exact gradients + high batched throughput
```

---

## 17. Prior-work positioning

NDVM draws from several traditions but occupies a distinct point.

### 17.1 Enzyme / LLVM AD

Relevant because it demonstrates high-performance native AD over optimized IR. Difference: Enzyme is compiler-centric and differentiates known programs; NDVM differentiates execution of a runtime interpreter.

### 17.2 PyTorch / JAX

Relevant because they provide dynamic and staged differentiable execution. Difference: PyTorch eager imposes tensor-object and tiny-op overhead; JAX/XLA requires traceable static control and struggles with the dynamic evaluator.

### 17.3 Lisp/Scheme VMs

Relevant for tagged values, closures, environments, tail calls, inline caches, generational heaps, and JIT techniques. Difference: traditional VMs are not reverse-mode differentiable over numeric payloads.

### 17.4 JVM / HotSpot / LuaJIT

Relevant for inline caches, speculative optimization, tracing, branch prediction, and tiered execution. Difference: NDVM's primary tier must remain interpreter-level and differentiable without object-program compilation.

### 17.5 Differentiable interpreters

Relevant conceptually, but most prior differentiable interpreters either make control soft/differentiable or target restricted synthesis languages. NDVM keeps symbolic control discrete and differentiates only numeric payloads through exact realized traces.

NDVM's novelty is the combination:

```text
high-performance symbolic VM
+
native reverse-mode AD
+
runtime object programs as data
+
compile-once differentiable evaluator
+
batch-native population execution
```

---

## 18. Implementation plan

### Phase 0: Profiling contract

Before implementation, formalize the current cost model.

Collect:

```text
per-op counts
wrap/unwrap counts
heap ops
branch counts
tape node counts
payload allocation counts
batch scaling curves
CPU/GPU kernel launch counts
```

This defines the baseline.

### Phase 1: Native forward runtime

Implement:

```text
Value representation
symbol table
immutable heap arena
environment representation
NDVM bytecode
direct-threaded evaluator execution
primitive dispatch
lazy conditionals
tail-call trampoline
```

No AD yet.

Deliverable:

```text
forward-equivalent native evaluator
```

Expected speedup:

```text
2x to 5x forward scalar
```

### Phase 2: Native reverse-mode tape

Implement:

```text
payload buffers
adjoint buffers
tape nodes
backward bytecode
primitive adjoints
gradient extraction
PyTorch custom autograd wrapper
```

Deliverable:

```text
gradient-equivalent native evaluator
```

Expected speedup:

```text
4x to 10x fwd+bwd scalar
```

### Phase 3: Batch-native execution

Implement:

```text
batched payloads
SIMD loops
batch axes
lane masks
masked branch execution
population batching
```

Deliverable:

```text
single evaluator walk over large parameter/data batches
```

Expected speedup:

```text
2x to 10x over current batched DMCI
```

### Phase 4: Structural caches

Implement:

```text
decoded S-expression cache
symbol dispatch cache
environment lookup inline cache
primitive inline cache
trace hash collection
cache invalidation rules
```

Deliverable:

```text
high-throughput repeated evaluation of fixed object programs
```

Expected speedup:

```text
1.5x to 5x on repeated fitting workloads
```

### Phase 5: Multi-core scheduler

Implement:

```text
thread-local heaps/tapes
candidate-level parallelism
work stealing
NUMA-aware arenas
batch partitioning
```

Deliverable:

```text
OpenEvolve-scale CPU runtime
```

Expected speedup:

```text
near-linear over candidates until memory bandwidth or batch size limits
```

### Phase 6: GPU backend

Implement only after CPU VM is mature.

Components:

```text
persistent forward kernel
GPU tape
backward replay kernel
lane masks
trace bucketing
dense numeric kernels
```

Deliverable:

```text
GPU acceleration for large batched or matrix-heavy NDVM workloads
```

Expected speedup:

```text
significant only for large-batch or numerically heavy workloads
```

### Phase 7: Optional MLIR/Enzyme integration

Investigate:

```text
MLIR lowering for numeric kernels
Enzyme-generated adjoints for fused regions
AOT compilation of evaluator bytecode
```

This is an advanced optimization, not a baseline dependency.

---

## 19. Engineering risks

### 19.1 AD correctness bugs

Native AD must be exhaustively tested. Every primitive needs a correct adjoint. Branch masks must correctly gate adjoint accumulation.

Mitigation:

```text
property tests
finite differences
comparison to PyTorch DMCI
comparison to direct compilation
trace replay checks
```

### 19.2 Tape memory blowup

Recursive programs and long rollouts can generate large tapes.

Mitigation:

```text
checkpointing
tape compression
adjoint superinstructions
segment recomputation
```

### 19.3 GPU underperformance

GPU backend can easily lose to CPU if it launches too many kernels or suffers divergence.

Mitigation:

```text
persistent kernels only
large-batch threshold
CPU fallback
trace bucketing
matrix-heavy primitive fusion
```

### 19.4 Semantic drift

Structural caches can accidentally become program-specific compilation.

Mitigation:

```text
define allowed cache semantics clearly
cache decoded data and lookup paths only
do not replace evaluator semantics with residual program graphs
maintain interpreter-trace equivalence tests
```

### 19.5 Complexity creep

A full VM, AD engine, CPU scheduler, and GPU backend is a large project.

Mitigation:

```text
CPU-first
manual tape first
GPU later
MLIR/Enzyme later
keep PyTorch backend as reference oracle
```

---

## 20. Success criteria

Minimum viable NDVM:

```text
executes compiled evaluator natively
supports core DMCI Scheme subset
matches PyTorch DMCI forward outputs
matches gradients to numerical tolerance
supports batching over payloads
exposes PyTorch-compatible autograd interface
achieves >5x sequential speedup
achieves >2x speedup over current batched backend on recursive workloads
```

Ambitious target:

```text
sequential DMCI within 1.5x to 3x of direct compilation
batched recursive workloads within 2x to 4x of specialized vectorized implementations
OpenEvolve inner-loop fitting 10x to 30x faster than current batched PyTorch DMCI
GPU backend accelerates large-batch/matrix-heavy workloads by 10x+ over CPU NDVM
```

Research-grade success:

```text
NDVM establishes a new systems point:
a high-performance differentiable symbolic VM where arbitrary runtime programs inherit gradients from one compiled interpreter.
```

---

## 21. Summary

NDVM should be built around one principle:

```text
make the interpreter fast without compiling away the interpreted program
```

The runtime should replace tensorized tags with native scalar tags, PyTorch tensor-object bookkeeping with dense payload buffers, Python graph walking with native bytecode, PyTorch autograd objects with compact adjoint bytecode, and naive batching with structural/numeric split batching. CPU should be the primary execution engine because the workload is branchy and symbolic; GPU should be used only through persistent kernels and dense numeric payload operations.

This preserves the DMCI thesis while making the implementation credible as a high-performance runtime system. The end state is not a neural compiler for every candidate program. It is a native differentiable VM for a compiled meta-circular evaluator: a differentiable Lisp/Scheme runtime specialized for scientific program-and-parameter co-search.
