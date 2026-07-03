# NDVM Phase 6: GPU backend -- scoped POC (forward numeric ceiling)

Status: **scoped POC done; full backend still deferred.** The locked plan (`PHASE6_DESIGN.md`) called for
a scoped proof-of-concept to measure the GPU-vs-CPU crossover before any full-backend commitment, with a
hard kill criterion. This is that POC's first measurement: the **forward numeric ceiling** of the
persistent-kernel design, measured on an RTX 4090 against the Phase-5 64-thread CPU. The full D2
interpreter, the backward pass, and the production backend remain deferred (see "What this does and does
not show").

## What was built

`gpu/kalman_poc.cu`: a CUDA kernel that evaluates a **population of independent D-dimensional Kalman-filter
NLLs** -- the dense-numeric rollout that would run on the GPU in the locked D2 persistent-kernel design.
One block per candidate; the block's threads cooperate on the D x D linear algebra (`F P F^T` is a real
O(D^3) matmul, the GPU-favorable work); the 2x2 innovation covariance is a closed-form inverse + log-det
(no general LU); candidates differ in their fitted noise parameters `(q, r)`, with shared dynamics `F`,
observation map `H`, and observation sequence (the Kalman MLE flagship: fit `q, r` to one trajectory across
many restarts). A multithreaded CPU reference runs the **identical** computation; correctness is the GPU
result matching the CPU within float32 tolerance.

## Result (RTX 4090, sm_89, vs a 64-thread CPU on the same node, T=80 steps)

Candidate-evals/sec, population G=16384, swept over state dimension D -- all correct (max relative error
$\sim$3e-7, zero mismatches):

| D | GPU evals/s | CPU evals/s | GPU speedup |
|---|--:|--:|--:|
| 2 | 6.81M | 3.55M | 1.9x |
| 8 | 5.83M | 0.98M | 5.9x |
| 16 | 1.84M | 0.167M | **11.0x** |
| 32 | 0.109M | 0.021M | 5.2x |
| 64 | 0.0135M | 0.0024M | 5.5x |

Population sweep (D=16): GPU wins from G=256 (19.2x) and saturates near 2.0M evals/s by G=4096; the CPU
scales with G. The GPU beats the 64-thread CPU at **every** D and population tested, and **clears the
$\geq$2x kill-criterion bar at all D, including D=64 (5.5x)**. The speedup peaks at D=16 (the D x D work
maps one-element-per-thread to a 256-thread block) and falls at D=32/64 as the per-block shared-memory
footprint (3 D x D matrices; ~49 KB at D=64) caps SM occupancy.

## What this does and does not show (the honest reading)

This is the **numeric ceiling**, not the product. It measures a specialized dense-numeric Kalman kernel
(no interpreter dispatch, no tape, forward only) against a specialized CPU Kalman -- the **upper bound** for
a GPU NDVM. It is a real, positive signal: the 4090's dense-numeric throughput beats 64 CPU cores by
2--19x on this workload, correct to float32, so a GPU backend is **not dead on arrival**.

It deliberately does **not** yet answer the design's actual question, which is why the full backend stays
deferred:

- **Interpreter overhead is excluded.** The locked D2 design runs the *branchy structural walk* on the GPU
  (a leader thread per block while B-1 lanes idle), which the feasibility study argued erodes the win at
  small D. This POC's specialized kernel has no such walk -- every thread does numeric work. The true
  interpreted performance sits below this ceiling by an amount only the full D2 POC can measure. So the
  ceiling winning at small D does **not** contradict the study's prediction that the *interpreted* version
  loses there.
- **Forward only.** The $\geq$2x kill criterion is on forward+grad. The backward Kalman (reverse-mode tape
  replay) roughly doubles the work on both sides; the crossover is expected to be similar, but it is
  unmeasured here.
- **The CPU bar here is the specialized Kalman** (3.6M evals/s at D=2), ~55x faster than the *interpreted*
  NDVM Kalman (the Phase-5 66k evals/s). Comparing the GPU specialized kernel to the CPU *interpreter*
  would be apples-to-oranges; this POC fairly compares specialized-to-specialized (the numeric ceiling on
  both sides).
- **Float32 precision is delicate at large D.** A first run with an under-damped `F` (spectral radius near
  1) made the covariance blow up and the GPU's fused multiply-adds diverge from the CPU's (5.7% error at
  D=64). A well-conditioned `F` (coupling scaled by 1/D) fixed it (3e-7). A production large-D backend
  would need careful conditioning or fp64 for the covariance.

## Verdict (refines, does not overturn, the deferral)

The numeric crossover is favorable, so the GPU design is viable in principle and the architecture is sound.
But the full GO still waits on two things from `PHASE6_DESIGN.md`: (1) the full D2 POC -- the *interpreted*
forward+grad kernel -- confirming the win survives the structural-walk overhead and the gradient; and (2) a
committed high-D long-rollout flagship to consume it (the current flagships are low-D, where even this
ceiling's absolute throughput, and the CPU's near-linear scaling, make the CPU the pragmatic choice). The
CPU multicore stays primary; this POC measured the one number the plan asked for first, and it passed.

## Build and run

```bash
module load cuda/12.8
nvcc -O3 -arch=sm_89 -Xcompiler -pthread ndvm/gpu/kalman_poc.cu -o /tmp/kalman_poc
NDVM_D=16 NDVM_G=16384 NDVM_T=80 NDVM_CPU_THREADS=64 /tmp/kalman_poc   # one (D, population) point
```
