# NDVM Phase 0 baseline cost model (locked)

> **Provenance note (updated).** This document records the *original* Phase-0 locking run on node
> `n104`. The manuscript's reported Phase-0 numbers are the later **n128 re-run** committed at
> `results/baseline_n128.json` (plus `_r2`, `_r3`; median of three), and the `REPRODUCE.md` artifact guide is
> the authoritative artifact guide. The cost *structure* (batch-independence, boxing+walk dominance,
> arithmetic ~1%, ~257x fwd:bwd) is identical on both nodes; only absolute times differ because n104 is
> a different machine. The n104 numbers below are kept as the historical locking record and are labeled
> as such; do not read them as the paper's reported values.

The reference point every NDVM speedup is measured against: the **current PyTorch DMCI backend**,
profiled on an HPC compute node before any native code exists. Source: `profile_dmci_baseline.py`
(run via `slurm_baseline.sh`). Raw artifacts (paper numbers): `results/baseline_n128.json`,
`results/baseline_n128_r2.json`, `results/baseline_n128_r3.json`. The original n104 locking run is
preserved at `results/phase0_5161442.out`.

**Environment (original n104 locking run).** Node `n104` (gpu-8 partition, CPU run), Python 3.11.10,
torch 2.12.0+cu130, single process, 8 threads. `iters=30`, batches `{1,8,64,256,1024}`, `--decompose`. Job 5161442, 12m38s,
peak RSS 480 MB. Run in an isolated checkout (`/mnt/ceph/sheneman/src/nncompile-ndvm`, copied venv);
the ICLR manuscript job on `n113` was untouched.

> Note on the stack: the original perf autopsy (`experiments/exp_a/results/profile_decomposition.txt`)
> was Python 3.8 / torch 1.13 and reported 5.76 ms forward for a scalar program. This baseline is on
> Python 3.11 / torch 2.12 and is slightly faster (4.37 ms for the same `(+ (* a x) b)` shape), but the
> **cost structure is identical**. The bucket fractions below reproduce the autopsy within ~1 point.

## Per-program wall-clock (B=1)

| program | regime | fwd (ms) | bwd (ms) | rollout | ms / rollout-step |
|---|---|--:|--:|--:|--:|
| `scalar_mul_add` `(+ (* alpha x) beta)` | scalar | 4.37 | 0.25 | 1 | — |
| `michaelis_menten` `(/ (* Vmax S) (+ Km S))` | scalar | 6.32 | 0.29 | 1 | — |
| `damped_oscillator` `(* A (* (exp ..) (cos ..)))` | transcendental | 14.61 | 0.53 | 1 | — |
| `logistic_map_loop` (16-step loop/recur) | recursive scalar | 229.08 | 3.07 | 16 | 14.3 |
| `kalman2d_T80` (80-step Kalman NLL, matrix ops) | matrix rollout | **10246.71** | 39.79 | 80 | **128.1** |

Forward dominates everything: the fwd:bwd ratio is ~17x for the scalar program and **~257x** for the
Kalman rollout. **Backward is not a target.** A single 80-step 2D Kalman *forward* takes **10.2 seconds**
through the interpreter; this is the cost that blocks the turbulence-ROM / Kalman-MLE flagships.

## Headline finding: cost is overhead-bound and batch-independent

Forward time is essentially flat from B=1 to B=1024 (a 1024x payload):

| program | B=1 | B=8 | B=64 | B=256 | B=1024 | Δ(1→1024) |
|---|--:|--:|--:|--:|--:|--:|
| `scalar_mul_add` | 4.37 | 4.44 | 4.45 | 4.47 | 4.52 | **+3.4%** |
| `michaelis_menten` | 6.32 | 6.38 | 6.38 | 6.42 | 6.48 | +2.5% |
| `damped_oscillator` | 14.61 | 14.73 | 14.78 | 14.78 | 14.92 | +2.1% |
| `logistic_map_loop` | 229.08 | 229.51 | 229.98 | 229.92 | 229.91 | **+0.4%** |

The marginal cost of a batch lane is ~0.15 µs (just the dense payload arithmetic); the multi-millisecond
bulk is the per-walk interpreter overhead, paid once regardless of B. **The interpreter walk, not the
math, is the cost.** This is the direct empirical case for NDVM's structural/numeric split and for
population batching (B_restart x B_cell x B_data) as the throughput multiplier: one evaluator walk should
amortize over thousands of parameter vectors at near-zero marginal cost.

## Forward cost decomposition (regime-invariant)

Per-bucket share of profiled forward time (cProfile `tottime`; bucket rules mirror the exp_a autopsy):

| bucket | scalar | michaelis | damped | logistic | **kalman** |
|---|--:|--:|--:|--:|--:|
| **Tagged-value wrap/unwrap (boxing)** | 61.0% | 61.7% | 66.3% | 62.2% | **64.9%** |
| Evaluator graph-walking | 25.0% | 24.1% | 20.6% | 23.8% | 21.5% |
| Heap (cons/car/cdr) | 7.1% | 7.0% | 5.5% | 6.8% | 6.5% |
| Dispatch (tagged_if/eq?/select) | 2.9% | 3.0% | 4.2% | 4.0% | 4.7% |
| Tagged arithmetic wrappers | 3.0% | 3.3% | 2.7% | 2.7% | 2.3% |
| Raw arithmetic / linalg | 1.0% | 0.9% | 0.7% | 0.6% | 0.1% |
| PyTorch autograd (forward) | ~0% | ~0% | ~0% | ~0% | ~0% |

The structure barely moves across a 2300x range of program cost (4 ms scalar -> 10 s matrix rollout):

- **Boxing is ~61-66% of forward in every regime.** It is the #1 NDVM target, exactly as the design says.
- **Boxing + graph-walking = ~85-87%** of forward, everywhere. NDVM's scalar-tag + dense-payload
  representation plus a native direct-threaded evaluator attack this ~85% directly.
- Raw arithmetic (including 2x2 `inv`/`det`/`matmul` in Kalman) is **<1%**. The interpreter is not
  FLOP-bound; it is representation-bound.

### The boxing tax, in call counts

cProfile call counts per single forward make the tax concrete:

| program | boxing calls / fwd | graph-walk calls | arithmetic calls |
|---|--:|--:|--:|
| `scalar_mul_add` | 828 | 27 | 8 |
| `logistic_map_loop` (16 steps) | 43,214 | 719 | 258 |
| `kalman2d_T80` (80 steps) | **1,994,674** | 35,095 | 3,442 |

Folding an 80-step 2D Kalman filter constructs/destructs **~2 million** `[14]`-element tagged-value
tensors. Each is a `torch.cat`/allocation of a 10-wide one-hot tag plus a 4-wide payload. That is the
10-second forward pass, and it is what the structural/numeric split eliminates: a scalar tag + an index
into a dense payload buffer, with no per-value tensor allocation.

## Implications for the NDVM phases

- **Phase 1 (native forward) + the Value/payload split target the ~85% (boxing + walk) directly.** Even
  before AD, a native scalar-tagged evaluator with dense payloads should remove most of the forward cost.
- **Phase 2 (native tape) is low-risk on the perf axis:** backward is <1% of total today, so the AD
  engine's job is correctness and not regressing, not speed.
- **Phase 3 (batch-native) is where the order-of-magnitude lives for co-search:** the batch curve shows
  the walk is already amortizable; a native runtime that batches the payload makes B_restart x B_cell
  fits nearly free per evaluator walk.
- **The flagship gate is forward latency on matrix rollouts** (128 ms/step, 10 s/forward here). That is
  the number NDVM must move for the Kalman-MLE / turbulence-ROM flagships to become compute-feasible.

## Reproduce

```bash
# On an HPC gpu-8 compute node (NOT the login node / Mac; torch lives in the venv).
cd /mnt/ceph/sheneman/src/nncompile-ndvm
sbatch ndvm/profiling/slurm_baseline.sh        # excludes n113 (ICLR) + bad nodes
# or directly:
.venv/bin/python ndvm/profiling/profile_dmci_baseline.py --iters 30 --batches 1 8 64 256 1024 --decompose
```
