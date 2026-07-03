# NDVM Phase-6 GPU Backend — LOCKED Plan

## 1. Verdict: DESIGN-ONLY (defer-with-design), gated POC trigger pre-committed

**DESIGN-ONLY now. No CUDA build this cycle.**

The value assessment is unambiguous: GPU NDVM beats the 64-core CPU scheduler in exactly one regime — large-D (D≥32), long matrix-heavy rollouts (T≥200), with population×B in the thousands of trace-compatible lanes (turbulence-ROM class), where O(D³) linalg per step finally gives the 4090's ~82 FP32 TFLOP/s and ~1 TB/s something to amortize against. In the co-search common case the CPU wins decisively: tiny scalar/recursive candidates and the 2×2 Kalman are branchy symbolic walks with <1% arithmetic intensity (Phase-0: ~85% boxing+walk), and 64 cores already deliver 66k Kalman evals/s (near-linear, 15.8×/98.9% at 16 cores) and tens of millions of tiny evals/s. The one winning regime is real but **not currently load-bearing**: every flagship actually pursued (battery, ENSO Kalman-LIM, flu) is low-D and smooth, and the sole high-D consumer (turbulence ROM) was independently deferred to next-cycle (2026-06-18: narrow novelty, non-smoothability unproven). A full persistent-kernel CUDA backend therefore has **no committed consumer today**. Building it now would be speculative work against a doubly-gated application, exactly what doc 12.1/12.5 and risk 19.3 warn against. We lock the design, pin an explicit GO trigger, and keep the Phase-5 CPU scheduler as primary.

## 2. Chosen GPU architecture (locked design, for when the trigger fires)

**Architecture: D2 — persistent batch-lane interpreter (one block = one candidate, threads = B lanes).** Not D1 (thread-per-candidate; targets the tiny-scalar regime the CPU already owns — wrong bet). Not D3 (per-op offload; Amdahl-bounded to ~1% since the interpreter stays on CPU — cannot accelerate the dominant walk). D2 is the only design whose win condition (large B × large D × large population) matches the only regime where GPU beats CPU.

- **Flat program representation (program-as-data, invariant-preserving).** Host-side one-time lowering pass (Phase-6a, `ndvm/src/gpu/flatten.{hpp,cpp}`) linearizes the Phase-4 decoded `Datum` tree into a pointer-free, GPU-uploadable struct-of-arrays node image:
  - `kind[]` (uint8 DKind), `op[]` (uint8: SForm opcode **or** a pre-resolved group_A/group_B primitive opcode — no strings), `fval[]` (float literal), `ivar[]` (int32: interned sym id + resolved `vhops<<16|vslot` lexical address), CSR children: `child_begin[]`/`child_count[]` into a flat `children[]` index array.
  - Closures become `{param-id range, body node id, captured-env id}`. Symbol ids already interned; primitive names resolved to small-int opcodes at flatten time so the kernel never touches a string. Quote/cons-building (`SF_QUOTE` → `materialize`) is **excluded** from the GPU path (heap pairs) — raise→CPU fallback.
  - This is doc 10.1/19.4-sanctioned decoded-data caching: the SAME classification the CPU computes, serialized. The kernel dispatches on `kind[ip]`/`op[ip]` in a switch identical to `eval()`'s — swap the node array, interpret a different program, **no recompile, no per-program codegen.** A new candidate is a new node array uploaded, not a new kernel.

- **Thread/block mapping.** Block = candidate; `blockDim = next-multiple-of-32(B)` capped at 1024 (B>1024 strides). A designated leader thread drives the structural walk **once per block** (eval for-loop: node cursor, frame push/pop, branch classify, tape append) reading lane-uniform structural state from **shared memory** (env/frame stack, args-pool, closure table, cursor, lane mask). Numeric primitives fan out across block threads (thread t = lane t): the `for b in 0..B` loops in `interp_linalg.cpp`/`interp_tape.cpp` become `b = threadIdx`. B-strided payload arena (`primal_`, `adj_`), VecCell data, and the per-block tape live in **global memory**. A grid of G blocks runs G candidates; population is trace-bucketed by identical decoded image at the host (scheduling, not residualization).

- **Divergence.** (1) Intra-block per-lane MIXED branches (Phase-3b) → warp-ballot: leader computes the per-lane boolean, `__ballot_sync` yields then/else masks, both subtrees are walked once by the leader (control uniform by construction), masked lanes no-op their numeric writes, `Op::SELECT` merges. `intern_actset` id → `mask_id` in the tape node. SIMT divergence is confined to the masked numeric kernels (the cheap place). (2) Inter-candidate structural divergence → host trace-bucketing so blocks in a launch share a node image. The honest cost — idle lane-threads during the serial structural walk — is the central risk and is what the POC measures (§6).

- **Forward vs +grad.** Forward kernel: leader-driven eval switch + `__syncthreads`-gated per-lane primitives; tape appended via a per-block shared-mem cursor (**no global atomics in the hot loop**; each block owns its tape region). Tape = fixed-arity `GpuTapeNode {op, flags, out, in1, in2, aux, mask_id}`; variadic +/−/*/÷ lowered to binary-fold nodes; >2 ins → out-of-line operand table. Backward kernel: flat reverse linear scan replaying the tape (port `dispatch_adjoint`), per-lane VJP gated by `mask_id` (preserves the no-`0*NaN` guarantee); matrix-element VJPs use thread-per-output-element to stay race-free. Reverse is **more** GPU-friendly than forward (no recursion, no allocation, no walk divergence). det/inv hazard: closed-form 2×2/3×3 (no pivot, no allocation, uniform) for the flagship; warp-cooperative LU only if D grows; larger-D batched LU stays a separate decision. PyTorch `autograd.Function` boundary unchanged at the API level — dispatches to CUDA above the (B×D) threshold, CPU below. Forward-only mode (no tape) is a launch flag for fitness-only ranking generations.

## 3. The GO trigger (pre-committed, explicit)

The design moves to a **SCOPED POC** (§4) — never a blind full build — the moment ALL of these hold:
1. A **committed** high-D, long-rollout, matrix-heavy flagship exists in the validated suite (turbulence-ROM class): **D ≥ 32, T ≥ 200, population×B in the thousands** of trace-compatible lanes.
2. A **measured** CPU wall is the actual blocker (hours/candidate-fit on the Phase-5 64-core scheduler), documented on the sheneman partition.
3. Non-smoothability / fit feasibility for that flagship is resolved (the 2026-06-18 turbulence open question), so we are not accelerating a fit that does not converge.

Until all three fire, this is DESIGN-ONLY.

## 4. The POC (scoped; only runs after the trigger) — smallest deliverable + a real measured number

**One vertical slice that proves or kills the persistent-kernel model with a measured 4090-vs-64-core number.**

- **Scope:** the D2 forward path only (backward stubbed for the first measurement, then enabled), for **one** program image hard-supplied as a **device node array still interpreted as data** (not codegen'd). Program subset: leader-driven structural walk + per-lane `matvec`/`matmul` + closed-form small-matrix `det`/`inv` (no quote/cons, no heap lists, no general LU). Per-block tape append exercised even while backward is stubbed.
- **Representative workload:** the D×D Kalman/SSM rollout with **D swept {2, 8, 32, 64}**, **B swept {32, 256, 1024}**, population **G = 1024…16384 blocks**, T≈80–200 steps. This deliberately spans from the current 2×2 (expected loss) to the high-D regime (expected win) so the crossover contour is measured, not assumed.
- **The number:** candidate-evals/sec, GPU vs the Phase-5 `ndvm_par` 64-core baseline (66k/s at 2×2 Kalman) on the **same node's CPU**, plus a forward+grad number once backward is enabled. Plus Nsight: achieved occupancy and the **structural-walk vs numeric time split** inside the kernel (quantifies the idle-lane diagnosis).
- **Hypothesis to confirm/refute:** GPU loses at D=2 for all B (walk-dominated); wins only once D ≥ ~16–32 AND B ≥ ~256. **Kill criterion:** if even D=64, B=1024 does not clear **≥ 2× the 64-core node on forward+grad**, the persistent-batch-lane design is not worth the full build for then-current workloads, and Phase 6 stays deferred. Plot the (B, D) contour where GPU crosses the CPU line — that contour is the whole decision.

POC effort if triggered: ~2–3 weeks for the slice (host flattener for one image + device forward kernel + per-lane matvec/matmul + closed-form det/inv + benchmark harness), before any commitment to the full backend (§2 full build is "Large").

## 5. Validation plan

- **Correctness, tolerance-based GPU-vs-CPU (NOT bit-exact):** GPU FMA/reduction order differs from CPU, so the GPU gate is tolerance-based (the one honest relaxation vs the CPU determinism gate, which IS byte-identical to the oracle). Forward outputs and gradients must match the CPU NDVM (itself byte-identical to the PyTorch-DMCI oracle) within float32 tolerance across the existing **33 forward / 82 grad / 27 batched / 21 divergent** suites, plus a dedicated GPU-vs-CPU cross-check on identical programs/params (Kalman dNLL/dq,dr included).
- **Throughput:** the §4 evals/sec sweep + Nsight occupancy and walk/numeric split.
- **Reference oracle:** PyTorch-DMCI + CPU NDVM remain the oracle for every number; the GPU never becomes its own ground truth.
- **Environment (sheneman partition):** `srun -p sheneman --gres=gpu:1`, `module load cuda/12.8` for `nvcc`, `sm_89`, single RTX 4090 (cc 8.9), CUDA 12.8, fp32 throughout. Repo `.venv` (uv) works only on compute nodes — torch/DMCI via `sbatch`/`srun`, never the login node (silent conda-base fallback, no torch). Validate on the deployment compiler (g++ caught a Phase-2 eval-order UB clang/ASan missed).

## 6. Explicitly deferred (and why)

- **The full D2 CUDA backend** — no committed consumer; doubly-gated behind an application (turbulence ROM) that is itself deferred. Build only after the §3 trigger and a passing §4 POC.
- **D1 (thread-per-candidate)** — targets the tiny-scalar regime where the CPU already does ~100M/s; back-of-envelope shows GPU underwhelms (~5M/s) for ultra-tiny work. Not the bet. Kept on paper only.
- **D3 (per-op CPU→GPU offload)** — Amdahl-bounded to ~1.01× max (interpreter stays on CPU, numeric slice ~1%); catastrophic on 2×2 Kalman (kernel launch ~5–10µs vs ~0.15µs arithmetic). Revisit only as the conservative numeric *tier* of a future large-D flagship, never as "the" backend.
- **General-purpose GPU interpreter** — explicitly not built; quote/cons-building, dynamic heap arenas, growable env frames, string dispatch, and per-lane-divergent LU are GPU-hostile and raise→CPU fallback by design.
- **Batched cuBLAS/cuSOLVER LU for large D, Phase-3b lane-mask micro-optimizations, batched torch boundary** — sequenced after the POC validates the core model.
- **Enzyme/MLIR autodiff (Phase 7)** — out of scope; adjoints stay hand-written and validated kernel-by-kernel against the existing host VJPs.

## 7. The honest risk it underperforms CPU, and how the POC measures it fairly

The NDVM eval loop is a textbook GPU-hostile workload: deep recursion, per-step heap/frame/arg allocation, pointer/index chasing, a giant branchy switch, and tape appends — and in D2 ALL of that runs on a **single leader thread per block while B−1 lane-threads idle and other warps park at `__syncthreads()`**. For the current realistic flagships the GPU very likely **loses**: the 2×2 Kalman's numeric fan-out is ~a dozen FLOPs so the serial structural walk (80 steps × dozens of frames) dominates, and a ~2 GHz SM with no branch prediction is slower per-candidate than a 5 GHz core with warm L1; tiny scalar candidates have nothing to parallelize across lanes and pay pure launch+divergence overhead. The GPU can only win back ground via sheer block count, and only when B and D are both large enough that O(D³) numerics dwarf the fixed walk.

The POC measures this **fairly and adversarially**: (a) it sweeps D from 2 (expected loss) through 64 (expected win) and B from 32 to 1024, so the loss regime is included by construction, not hidden; (b) it benchmarks against the strongest CPU bar — the actual Phase-5 64-core `ndvm_par` at 66k Kalman evals/s on the same node, not a strawman single core; (c) it uses Nsight to quantify the idle-lane diagnosis directly (occupancy + walk-vs-numeric time split), so a loss is explained, not just observed; (d) it enforces a hard kill criterion (≥2× at D=64/B=1024 forward+grad or the design stays deferred). The fair, decisive outcome we accept in advance: if a flattened-program persistent kernel cannot clear 2× with divergence under control at the population sizes a real co-search uses, the locked conclusion is **"GPU is a niche accelerator for a not-yet-committed dense-numeric flagship; the 64-core CPU stays primary"** — precisely what doc 12.1/12.5/19.3 predict.
