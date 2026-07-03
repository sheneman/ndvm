// NDVM Phase 5: candidate-level multicore scheduler implementation. See parallel.hpp.
#include "parallel.hpp"
#include "interp.hpp"
#include <atomic>
#include <thread>
#include <memory>
#include <unordered_map>
#include <system_error>
#include <algorithm>
#include <cstdlib>

namespace ndvm {

// Thread-local Interp cache (one per worker thread): each (src,B) gets its OWN Interp with its OWN parsed
// program, so the per-Datum decode/inline-var caches are written and read by exactly one thread on nodes
// that thread alone owns -- the AST-cache race is eliminated, not synchronized. Bounded LRU (evict
// oldest). This replaces the process-global static cache that the single-candidate boundary used.
static Interp& worker_interp(const std::string& src, uint32_t B) {
  static thread_local std::vector<std::pair<std::pair<std::string, uint32_t>, std::unique_ptr<Interp>>> cache;
  static const size_t CAP = 16;
  for (auto& e : cache) if (e.first.second == B && e.first.first == src) return *e.second;
  if (cache.size() >= CAP) cache.erase(cache.begin());
  cache.emplace_back(std::make_pair(src, B), std::make_unique<Interp>(B));
  return *cache.back().second;
}

// One task = a complete deterministic serial forward + reverse pass on a thread-local Interp. Any
// InterpError (structural divergence, eval-step budget) or std::exception is caught here and marshaled
// into the Result, so no exception ever unwinds through the worker thread or the pool.
static void eval_one(const Task& t, Result& r) {
  try {
    const uint32_t B = t.B == 0 ? 1 : t.B;
    const size_t P = t.snames.size();
    // Validate the task shape before indexing (operator[] is UB on a short vector; this keeps the raw API
    // as crash-proof as the Python wrapper, and the throw is marshaled into the Result by the catch).
    if (t.svals.size() != P * B) throw InterpError("task svals size mismatch (expected snames*B)");
    if (t.mnames.size() != t.mrows.size() || t.mnames.size() != t.mcols.size() || t.mnames.size() != t.mdata.size())
      throw InterpError("task matrix arrays length mismatch");
    Interp& I = worker_interp(t.src, B);
    I.begin_forward();                                  // fresh per-forward state; keeps the warm program
    for (size_t i = 0; i < P; ++i) {
      std::vector<float> v(B);
      for (uint32_t b = 0; b < B; ++b) v[b] = t.svals[i * B + b];
      I.bind_scalar_batched(t.snames[i], v);
    }
    for (size_t i = 0; i < t.mnames.size(); ++i) {
      std::vector<float> d = t.mdata[i];
      I.bind_matrix(t.mnames[i], t.mrows[i], t.mcols[i], std::move(d));
    }
    I.set_taping(t.want_grad);                          // unconditional: task N never depends on task N-1's taping
    Val res = I.run(t.src);
    const bool numeric = I.is_num(res);
    r.outs.assign(B, 0.0);
    if (numeric) for (uint32_t b = 0; b < B; ++b) r.outs[b] = static_cast<double>(I.num_lane(res, b));
    r.grads.assign(P * B, 0.0);
    if (t.want_grad && numeric) {
      I.backward(res);
      std::unordered_map<std::string, uint32_t> pid;
      for (const auto& p : I.scalar_params()) pid[p.first] = p.second;
      for (size_t i = 0; i < P; ++i) {
        auto it = pid.find(t.snames[i]);
        if (it != pid.end()) for (uint32_t b = 0; b < B; ++b) r.grads[i * B + b] = static_cast<double>(I.grad_lane(it->second, b));
      }
    }
    r.ok = true;
  } catch (const std::exception& e) {
    r.ok = false; r.err = e.what();
  } catch (...) {
    r.ok = false; r.err = "unknown error";
  }
}

std::vector<Result> evaluate_batch(const std::vector<Task>& tasks, int nthreads) {
  const size_t T = tasks.size();
  std::vector<Result> results(T);                       // pre-sized: each worker writes only its own slot
  if (T == 0) return results;

  int nt = nthreads;
  if (nt <= 0) {
    if (const char* e = std::getenv("NDVM_THREADS")) nt = std::atoi(e);
    if (nt <= 0) nt = static_cast<int>(std::thread::hardware_concurrency());
    if (nt <= 0) nt = 1;
  }
  if (static_cast<size_t>(nt) > T) nt = static_cast<int>(T);
  int hw = static_cast<int>(std::thread::hardware_concurrency()); if (hw <= 0) hw = 1;
  if (nt > 4 * hw) nt = 4 * hw;                            // sane ceiling: never try to spawn an absurd count

  std::atomic<size_t> next{0};
  auto worker = [&]() {
    for (;;) {
      size_t i = next.fetch_add(1, std::memory_order_relaxed);   // disjoint task indices, no shared writes
      if (i >= T) break;
      eval_one(tasks[i], results[i]);
    }
  };

  if (nt == 1) { worker(); return results; }              // serial path: identical work, no threads spawned
  std::vector<std::thread> pool;
  pool.reserve(nt);
  try {
    for (int w = 0; w < nt; ++w) pool.emplace_back(worker);
  } catch (const std::system_error&) {
    // OS refused another thread (RLIMIT_NPROC / EAGAIN). Degrade gracefully: the atomic-index pool
    // self-balances, so however many workers we got finish all tasks (deterministically, by index).
  }
  if (pool.empty()) { worker(); return results; }         // none spawned -> run on this thread
  for (auto& th : pool) th.join();
  return results;
}

}  // namespace ndvm
