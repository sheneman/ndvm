// NDVM Phase 5: candidate-level multicore scheduler. A population of independent (program, bindings)
// evaluations is run across a pool of worker threads, each with a thread-local Interp (so 100% of the
// engine's mutable state is per-thread and nothing is shared). Each task is a complete deterministic
// serial NDVM forward + reverse pass; results are placed by task index, so a parallel run is
// byte-identical to a serial one for any thread count. The B-lane (batched) axis is orthogonal and
// untouched: a task may itself be batched (Task::B > 1), and candidates are parallel across cores.
#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace ndvm {

// One independent candidate evaluation.
struct Task {
  std::string src;                            // program source (parsed once per worker, then reused)
  uint32_t B = 1;                             // inner batch width (per-lane parameter sets within the task)
  std::vector<std::string> snames;            // differentiable scalar parameter names
  std::vector<float> svals;                   // [snames.size() * B], param-major: svals[i*B + b]
  std::vector<std::string> mnames;            // matrix input names (shared across the task's B lanes)
  std::vector<uint32_t> mrows, mcols;         // matrix shapes
  std::vector<std::vector<float>> mdata;      // matrix data (row-major)
  bool want_grad = true;
};

struct Result {
  bool ok = false;                            // false if the task raised (err holds the message)
  std::string err;
  std::vector<double> outs;                   // [B] per-lane forward output
  std::vector<double> grads;                  // [snames.size() * B] per-lane d(out_b)/d(param_i lane b)
};

// Evaluate the population across `nthreads` workers (<=0 => NDVM_THREADS env, else hardware_concurrency).
// Deterministic: results[i] corresponds to tasks[i] and is bit-identical to a serial evaluation of tasks[i]
// regardless of thread count or scheduling. A task that raises sets its own Result{ok=false} and does not
// abort the batch.
std::vector<Result> evaluate_batch(const std::vector<Task>& tasks, int nthreads = 0);

}  // namespace ndvm
