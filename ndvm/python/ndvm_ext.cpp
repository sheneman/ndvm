// pybind11 glue exposing the NDVM native runtime to Python (the Phase-2 PyTorch autograd boundary).
// One entry point, eval_and_grad, runs a program forward and (optionally) the native reverse pass,
// returning the scalar output and d(output)/d(param) for each bound scalar -- everything NDVMFunction
// needs. Compiled via torch.utils.cpp_extension (which provides pybind11) alongside the ndvm src/*.cpp.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "interp.hpp"
#include "parallel.hpp"

namespace py = pybind11;
using namespace ndvm;

// Phase 4: a small cache of Interps keyed on (source, B) so a co-search that evaluates the SAME candidate
// program many times (parameter restarts / optimizer steps) pays parse + macro-expand + decode ONCE, not
// per call. Each call resets the per-forward state (begin_forward) and re-binds, so the warm parsed
// program + decoded-form cache + symbol table are reused. Single-threaded (matches the Phase-4 decode
// cache); a Phase-5 multicore boundary needs per-thread caches or a lock around the whole eval.
static Interp& cached_interp(const std::string& src, uint32_t B) {
  // thread_local (Phase 5): the single-candidate boundary path runs under the GIL, but a thread_local
  // cache removes the process-global-mutable hazard entirely (the parallel population path uses its own
  // per-worker pool in parallel.cpp; nothing process-global is shared across threads).
  static thread_local std::vector<std::pair<std::pair<std::string, uint32_t>, std::unique_ptr<Interp>>> cache;
  static const size_t CAP = 16;
  for (auto& e : cache) if (e.first.second == B && e.first.first == src) return *e.second;
  if (cache.size() >= CAP) cache.erase(cache.begin());   // evict oldest (simple bound)
  cache.emplace_back(std::make_pair(src, B), std::make_unique<Interp>(B));
  return *cache.back().second;
}

// Returns (output, grads) where grads[i] = d(output)/d(scalar_names[i]) (0 if want_grad is false or the
// output is non-scalar). Matrix inputs (e.g. as_matrix obs) are bound but not differentiated.
static py::tuple eval_and_grad(const std::string& src,
                               const std::vector<std::string>& snames,
                               const std::vector<double>& svals,
                               const std::vector<std::string>& mnames,
                               const std::vector<uint32_t>& mrows,
                               const std::vector<uint32_t>& mcols,
                               const std::vector<std::vector<float>>& mdata,
                               bool want_grad) {
  Interp I;
  for (size_t i = 0; i < snames.size(); ++i) I.bind_scalar(snames[i], static_cast<float>(svals[i]));
  for (size_t i = 0; i < mnames.size(); ++i) {
    std::vector<float> d = mdata[i];
    I.bind_matrix(mnames[i], mrows[i], mcols[i], std::move(d));
  }
  I.set_taping(want_grad);             // unconditional: taping state never carries over a reused Interp
  Val r = I.run(src);
  double out = I.is_num(r) ? static_cast<double>(I.num(r)) : 0.0;

  std::vector<double> grads(snames.size(), 0.0);
  if (want_grad && I.is_num(r)) {
    I.backward(r);
    std::unordered_map<std::string, float> gm;
    for (const auto& p : I.scalar_params()) gm[p.first] = I.grad_scalar(p.second);
    for (size_t i = 0; i < snames.size(); ++i) {
      auto it = gm.find(snames[i]);
      if (it != gm.end()) grads[i] = static_cast<double>(it->second);
    }
  }
  return py::make_tuple(out, grads);
}

// Allocation instrumentation (reviewer #6: substantiate "representation, not arithmetic" with allocation
// counts, not cProfile time). Runs ONE forward (no tape) and returns (value, payload_allocs, eval_steps):
// payload_allocs is the number of native dense-payload slots the runtime created for the whole forward,
// the NDVM analogue of the eager backend's per-value boxed-tensor allocations.
static py::tuple forward_alloc_count(const std::string& src,
                                     const std::vector<std::string>& snames,
                                     const std::vector<double>& svals,
                                     const std::vector<std::string>& mnames,
                                     const std::vector<uint32_t>& mrows,
                                     const std::vector<uint32_t>& mcols,
                                     const std::vector<std::vector<float>>& mdata) {
  Interp I;
  for (size_t i = 0; i < snames.size(); ++i) I.bind_scalar(snames[i], static_cast<float>(svals[i]));
  for (size_t i = 0; i < mnames.size(); ++i) {
    std::vector<float> d = mdata[i];
    I.bind_matrix(mnames[i], mrows[i], mcols[i], std::move(d));
  }
  I.set_taping(false);
  Val r = I.run(src);
  double out = I.is_num(r) ? static_cast<double>(I.num(r)) : 0.0;
  return py::make_tuple(out, static_cast<double>(I.payload_allocs()), static_cast<double>(I.eval_steps()));
}

// Batched (Phase 3b): fit B independent per-lane parameter vectors in ONE structural walk. svals_flat is
// [num_params * B], param-major (svals_flat[i*B + b] = param i, lane b). Matrices are shared across lanes
// (one bound matrix broadcast to all B). Returns (outs[B], grads_flat[num_params*B]) where grads_flat[i*B
// + b] = d(out_b)/d(param_i lane b) -- each lane's output depends only on its own lane's params, so this
// is the full per-lane gradient (NDVM seeds all B output adjoints to 1). Lanes that diverge structurally
// raise (per the Phase-3b control-flow rule), surfaced here as a Python exception.
static py::tuple eval_and_grad_batched(const std::string& src,
                                       const std::vector<std::string>& snames,
                                       const std::vector<double>& svals_flat,
                                       uint32_t B,
                                       const std::vector<std::string>& mnames,
                                       const std::vector<uint32_t>& mrows,
                                       const std::vector<uint32_t>& mcols,
                                       const std::vector<std::vector<float>>& mdata,
                                       bool want_grad) {
  if (B == 0) B = 1;
  const size_t P = snames.size();
  if (svals_flat.size() != P * B) throw std::runtime_error("eval_and_grad_batched: svals size != snames*B");
  Interp& I = cached_interp(src, B);   // reuse the parsed + decoded program across calls
  I.begin_forward();                   // fresh per-forward state; keeps the warm program + decode cache
  for (size_t i = 0; i < P; ++i) {
    std::vector<float> v(B);
    for (uint32_t b = 0; b < B; ++b) v[b] = static_cast<float>(svals_flat[i * B + b]);
    I.bind_scalar_batched(snames[i], v);
  }
  for (size_t i = 0; i < mnames.size(); ++i) {
    std::vector<float> d = mdata[i];
    I.bind_matrix(mnames[i], mrows[i], mcols[i], std::move(d));
  }
  I.set_taping(want_grad);             // unconditional: taping state never carries over a reused Interp
  Val r = I.run(src);
  const bool numeric = I.is_num(r);

  std::vector<double> outs(B, 0.0);
  if (numeric) for (uint32_t b = 0; b < B; ++b) outs[b] = static_cast<double>(I.num_lane(r, b));

  std::vector<double> grads_flat((size_t)P * B, 0.0);
  if (want_grad && numeric) {
    I.backward(r);
    std::unordered_map<std::string, uint32_t> pid;
    for (const auto& p : I.scalar_params()) pid[p.first] = p.second;
    for (size_t i = 0; i < P; ++i) {
      auto it = pid.find(snames[i]);
      if (it != pid.end()) for (uint32_t b = 0; b < B; ++b) grads_flat[i * B + b] = static_cast<double>(I.grad_lane(it->second, b));
    }
  }
  return py::make_tuple(outs, grads_flat);
}

// Phase 5: evaluate a POPULATION of independent candidate tasks across worker threads. Each task tuple is
// (src, snames, svals[P*B], B, mnames, mrows, mcols, mdata, want_grad). The whole native section runs with
// the GIL RELEASED (workers touch no Python objects); the result is byte-identical to evaluating each task
// serially, regardless of thread count. Returns a list of (ok, err, outs[B], grads[P*B]) in task order.
static py::list evaluate_batch_py(py::list tasks_py, int nthreads) {
  std::vector<Task> tasks;
  tasks.reserve(py::len(tasks_py));
  for (auto item : tasks_py) {
    py::tuple t = py::reinterpret_borrow<py::tuple>(item);
    Task task;
    task.src    = t[0].cast<std::string>();
    task.snames = t[1].cast<std::vector<std::string>>();
    task.svals  = t[2].cast<std::vector<float>>();
    task.B      = t[3].cast<uint32_t>();
    task.mnames = t[4].cast<std::vector<std::string>>();
    task.mrows  = t[5].cast<std::vector<uint32_t>>();
    task.mcols  = t[6].cast<std::vector<uint32_t>>();
    task.mdata  = t[7].cast<std::vector<std::vector<float>>>();
    task.want_grad = t[8].cast<bool>();
    tasks.push_back(std::move(task));
  }
  std::vector<Result> results;
  { py::gil_scoped_release rel; results = evaluate_batch(tasks, nthreads); }   // no Python touched here
  py::list out;
  for (auto& r : results) out.append(py::make_tuple(r.ok, r.err, r.outs, r.grads));
  return out;
}

PYBIND11_MODULE(ndvm_native, m) {
  m.doc() = "NDVM native runtime (forward + reverse-mode AD) Python boundary";
  m.def("forward_alloc_count", &forward_alloc_count,
        py::arg("src"), py::arg("snames"), py::arg("svals"),
        py::arg("mnames"), py::arg("mrows"), py::arg("mcols"), py::arg("mdata"),
        "Run one forward (no tape); return (value, payload_allocs, eval_steps).");
  m.def("eval_and_grad", &eval_and_grad,
        py::arg("src"), py::arg("snames"), py::arg("svals"),
        py::arg("mnames"), py::arg("mrows"), py::arg("mcols"), py::arg("mdata"),
        py::arg("want_grad") = true,
        "Run a Scheme program through the native interpreter; return (output, d(output)/d(param)...).");
  m.def("eval_and_grad_batched", &eval_and_grad_batched,
        py::arg("src"), py::arg("snames"), py::arg("svals"), py::arg("B"),
        py::arg("mnames"), py::arg("mrows"), py::arg("mcols"), py::arg("mdata"),
        py::arg("want_grad") = true,
        "Batched: one walk over B lanes; return (outs[B], per-lane d(out_b)/d(param_i lane b) flat[P*B]).");
  m.def("evaluate_batch", &evaluate_batch_py,
        py::arg("tasks"), py::arg("nthreads") = 0,
        "Phase 5: evaluate a population of independent (src, snames, svals, B, mnames, mrows, mcols, mdata, "
        "want_grad) tasks across worker threads (GIL released); return [(ok, err, outs, grads)] in order.");
}
