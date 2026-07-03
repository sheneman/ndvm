# NDVM: the Native Differentiable Virtual Machine

**Differentiate the evaluator, not the program: exact reverse-mode gradients through a fixed
interpreter over programs kept as runtime data, batched across whole populations of parameter
vectors in a single evaluator walk.**

NDVM is a native (C++17) runtime representation for differentiable symbolic computation. It splits
every runtime value along the line that matters for differentiation: discrete structure (type tags,
symbols, heap addresses, environments, control) stays native scalar data, while numbers live in
dense batched payload buffers with a compact reverse-mode tape recorded along the realized
execution trace. Programs remain first-class runtime data walked by one compiled evaluator; nothing
is staged or compiled per program. One structural walk serves an entire population of parameter
vectors, so per-lane calibration cost falls by roughly 60x at a batch of 256, and independent
candidate evaluations scale near-linearly across CPU cores.

This is the artifact repository for the paper:

> **Differentiate the Evaluator, Not the Program: An Efficient Runtime Representation for
> Neuro-Symbolic Learning** (Lucas Sheneman, 2026; preprint forthcoming)

The tree behind the paper's reported numbers is frozen at the git tag **`v1.0`**
(`git checkout v1.0`). The repository contains the code, the committed reference results, and the
reproducibility documentation; the manuscript itself is distributed separately.

## The related systems

NDVM is the third system in a line of work on differentiating through program execution:

| System | Repository | Paper |
| --- | --- | --- |
| **Neural Compiler**: compile Scheme into differentiable PyTorch graphs | [sheneman/neural_compiler](https://github.com/sheneman/neural_compiler) | [arXiv:2605.22498](https://arxiv.org/abs/2605.22498) |
| **DMCI**: a compiled self-hosted Scheme interpreter that is differentiable end to end, so programs stay data | [sheneman/dmci](https://github.com/sheneman/dmci) | [arXiv:2606.09930](https://arxiv.org/abs/2606.09930) |
| **NDVM** (this repo): a native runtime representation that makes interpreter-level differentiation fast without compiling programs away | [sheneman/ndvm](https://github.com/sheneman/ndvm) | forthcoming |

DMCI showed that a differentiable interpreter preserves program-as-data; its locked cost model
showed that 85 to 90 percent of forward time goes to value boxing and evaluator walking rather
than arithmetic. NDVM is the runtime answer to that measurement. The eager PyTorch DMCI ships in
this repository (under `neural_compiler/`) because it is both the measured baseline and the
correctness oracle every NDVM gradient is validated against.

## Quick start (no Python required)

The one-command smoke test builds the native runtime with your system compiler and checks forward
values and a reverse-mode gradient against known answers, in a few seconds, with no Python or
PyTorch dependency:

```bash
bash ndvm/smoke_test.sh
```

## Full build

The C++ runtime builds with CMake in Release mode; the differentiable PyTorch op builds as a
CppExtension (no ninja required):

```bash
# native runtime + CLI drivers
cmake -S ndvm -B ndvm/build -DCMAKE_BUILD_TYPE=Release
cmake --build ndvm/build -j

# PyTorch autograd boundary (requires torch; see requirements.txt)
cd ndvm/python && python setup.py build_ext --inplace
```

Equivalence is validated under both g++ and clang++; g++ 12.1.0 is the deployment compiler behind
the paper's numbers. Python 3.11+, PyTorch >= 2.0, NumPy required for the harnesses; JAX/jaxlib and
sympy are optional (external staged baselines only); openai and python-dotenv are optional (LLM
candidate proposal only; see below).

## Repository layout

| Path | What it is |
| --- | --- |
| `ndvm/src`, `ndvm/include` | The native runtime: s-expression parser, direct-threaded evaluator, batch-native payload table, reverse-mode tape, lane masks, multicore scheduler |
| `ndvm/tools` | CLI drivers: `ndvm_run`, `ndvm_par` (multicore), `boxed_run` (native boxed-value baseline), `compiled_kalman` (hand-differentiated compiled ceiling) |
| `ndvm/python` | The PyTorch `autograd.Function` boundary that exposes NDVM as a differentiable tensor op |
| `ndvm/profiling` | Every profiling and baseline harness in the paper, plus `results/` with the committed reference outputs for each measured table and figure |
| `ndvm/tests` | Randomized differential tester (forward, gradient, finite-difference gates vs the oracle) plus `results/` with the frozen corpora and reports |
| `ndvm/gpu` | A specialized forward-only CUDA kernel estimating the dense-numeric ceiling (a bound, not a GPU NDVM) |
| `ndvm/PHASE*.md` | Development phase notes recording how the runtime was built and validated stage by stage |
| `neural_compiler/` | The eager PyTorch DMCI: the measured baseline and the correctness oracle (see [sheneman/dmci](https://github.com/sheneman/dmci) for its own system repo) |
| `bootstrap/compiler.scm` | The self-hosted Scheme evaluator source that DMCI compiles |
| `REPRODUCE.md` | The artifact guide: environment capture, figure/table to script map, run protocol, tolerances and seeds |

The layout is meaningful: the harnesses locate the oracle and the Scheme source relative to the
repository root, so keep `ndvm/`, `neural_compiler/`, and `bootstrap/` as siblings.

## Reproducing the paper

`REPRODUCE.md` maps every table and figure to the script that regenerates it, records the exact
measurement environment, and states the tolerances, seeds, and run counts behind every gate. A
committed reference result accompanies every measured table (under `ndvm/profiling/results/` and
`ndvm/tests/results/`). Deterministic quantities (allocation counts, differential-test gate
booleans) are bit-reproducible; wall-clock values reproduce within run-to-run variance on
comparable hardware.

Two site-specific notes:

- The `*.sbatch` files are verbatim records of the cluster runs behind the paper (Slurm partition
  and node paths included). They are documentation of the protocol; the underlying Python scripts
  run on any Linux machine directly.
- The co-search candidate programs were proposed by an LLM behind a university endpoint, then
  compile-validated and cached. The caches are committed
  (`ndvm/profiling/results/cosearch*_candidates_valid.jsonl`), so the timed co-search results
  replay without any LLM access. Regenerating candidates with your own OpenAI-compatible endpoint
  is optional (`cosearch_propose.py`, `cosearch_rec_propose.py`).

## What is measured (summary)

- Exact per-parameter reverse-mode gradient equivalence to the reference DMCI backend across the
  program suite, including an 80-step Kalman filter's noise-covariance gradients through the
  matrix-adjoint path; 600 randomized forward/gradient/finite-difference checks at seed 1234 and
  1500 more at seed 7.
- Batch amortization: per-lane cost falls about 60x from B=1 to B=256 in the native runtime
  (about 21x deployed through the PyTorch boundary).
- Near-linear multicore scaling for independent candidates (about 15x on 16 cores), byte-identical
  to serial execution.
- In fixed-budget co-search over LLM-proposed programs, NDVM reaches good held-out fits about 24x
  sooner in wall-clock on a symbolic-regression task, and about 340x on a recurrence-heavy task.

## Citing

Until the preprint appears, please cite this repository:

```bibtex
@misc{sheneman2026ndvm,
  author       = {Lucas Sheneman},
  title        = {{NDVM}: the Native Differentiable Virtual Machine},
  year         = {2026},
  howpublished = {\url{https://github.com/sheneman/ndvm}},
  note         = {Code and data artifact, tag v1.0}
}
```

## License

MIT (see `LICENSE`).
