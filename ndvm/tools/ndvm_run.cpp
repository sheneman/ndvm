// ndvm_run: minimal CLI driver for the Phase-1 native forward runtime, used by the equivalence
// harness. Reads a program file and an optional bindings file, evaluates, and prints the scalar
// result plus diagnostics. Bindings file lines:
//   scalar <name> <value>
//   matrix <name> <rows> <cols> <v0> <v1> ...   (row-major)
#include "interp.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <tuple>
#include <utility>

static std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: ndvm_run <program_file> [bindings_file]\n"); return 2; }
  uint32_t B = 1;
  if (const char* nb = std::getenv("NDVM_B")) { long v = std::strtol(nb, nullptr, 10); if (v > 0) B = (uint32_t)v; }
  ndvm::Interp I(B);
  if (const char* ms = std::getenv("NDVM_MAX_STEPS")) { long v = std::strtol(ms, nullptr, 10); if (v > 0) I.set_max_eval_steps((uint64_t)v); }
  if (std::getenv("NDVM_NO_INLINE")) I.set_inline_cache(false);   // ablation: disable the var-lookup cache
  try {
    if (argc >= 3) {
      std::ifstream bf(argv[2]);
      std::string line;
      while (std::getline(bf, line)) {
        std::istringstream ls(line);
        std::string kind; ls >> kind;
        if (kind == "scalar") { std::string name; double v; ls >> name >> v; I.bind_scalar(name, static_cast<float>(v)); }
        else if (kind == "scalarb") {  // one value per lane: scalarb <name> v0 v1 ... v(B-1)
          std::string name; ls >> name; std::vector<float> vals; double x; while (ls >> x) vals.push_back(static_cast<float>(x));
          I.bind_scalar_batched(name, vals);
        }
        else if (kind == "matrix") {
          std::string name; uint32_t rows, cols; ls >> name >> rows >> cols;
          std::vector<float> d; double x; while (ls >> x) d.push_back(static_cast<float>(x));
          I.bind_matrix(name, rows, cols, std::move(d));
        }
      }
    }
    std::string src = slurp(argv[1]);
    // Optional bench: NDVM_BENCH=<N> times N fresh forward evaluations and prints avg ms.
    if (const char* nb = std::getenv("NDVM_BENCH")) {
      long n = std::strtol(nb, nullptr, 10);
      // Read bindings once into a reusable spec is overkill; re-bind per run on a fresh interp.
      std::string bindfile = (argc >= 3) ? argv[2] : "";
      auto t0 = std::chrono::steady_clock::now();
      for (long it = 0; it < n; ++it) {
        ndvm::Interp J;
        if (!bindfile.empty()) {
          std::ifstream bf(bindfile); std::string line;
          while (std::getline(bf, line)) {
            std::istringstream ls(line); std::string kind; ls >> kind;
            if (kind == "scalar") { std::string nm; double v; ls >> nm >> v; J.bind_scalar(nm, (float)v); }
            else if (kind == "matrix") { std::string nm; uint32_t rr, cc; ls >> nm >> rr >> cc;
              std::vector<float> d; double x; while (ls >> x) d.push_back((float)x); J.bind_matrix(nm, rr, cc, std::move(d)); }
          }
        }
        volatile float sink = J.is_num(J.run(src)) ? 1.0f : 0.0f; (void)sink;
      }
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)n;
      std::printf("bench %.6f ms/eval over %ld runs\n", ms, n);
      return 0;
    }
    // Reuse bench: NDVM_BENCH_REUSE=<N> reuses ONE Interp across N forwards (parse + macro-expand +
    // decode paid ONCE; per forward = begin_forward + re-bind + eval). Measures the cross-call
    // parse-caching win -- the co-search inner loop that the PyTorch boundary now drives.
    if (const char* nb = std::getenv("NDVM_BENCH_REUSE")) {
      long n = std::strtol(nb, nullptr, 10);
      std::vector<std::pair<std::string, float>> sc;
      std::vector<std::tuple<std::string, uint32_t, uint32_t, std::vector<float>>> mt;
      if (argc >= 3) {
        std::ifstream bf(argv[2]); std::string line;
        while (std::getline(bf, line)) {
          std::istringstream ls(line); std::string kind; ls >> kind;
          if (kind == "scalar") { std::string nm; double v; ls >> nm >> v; sc.push_back({nm, (float)v}); }
          else if (kind == "matrix") { std::string nm; uint32_t rr, cc; ls >> nm >> rr >> cc;
            std::vector<float> d; double x; while (ls >> x) d.push_back((float)x); mt.push_back({nm, rr, cc, std::move(d)}); }
        }
      }
      ndvm::Interp J;
      auto t0 = std::chrono::steady_clock::now();
      for (long it = 0; it < n; ++it) {
        J.begin_forward();
        for (auto& s : sc) J.bind_scalar(s.first, s.second);
        for (auto& m : mt) J.bind_matrix(std::get<0>(m), std::get<1>(m), std::get<2>(m), std::get<3>(m));
        volatile float sink = J.is_num(J.run(src)) ? 1.0f : 0.0f; (void)sink;
      }
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)n;
      std::printf("bench_reuse %.6f ms/eval over %ld runs\n", ms, n);
      return 0;
    }
    bool grad_mode = std::getenv("NDVM_GRAD") != nullptr;
    if (grad_mode) I.set_taping(true);   // record the tape during the forward run
    ndvm::Val r;
    if (const char* rn = std::getenv("NDVM_REUSE")) {
      // Reuse this Interp across N forwards (begin_forward + re-bind + run). The printed output is the
      // LAST forward's; a fresh single run (NDVM_REUSE unset) must produce byte-identical output, which
      // validates reset_state (a reused Interp == a fresh one).
      long n = std::strtol(rn, nullptr, 10); if (n < 1) n = 1;
      std::vector<std::pair<std::string, std::vector<float>>> sc;   // name -> B-wide per-lane values
      std::vector<std::tuple<std::string, uint32_t, uint32_t, std::vector<float>>> mt;
      if (argc >= 3) {
        std::ifstream bf(argv[2]); std::string line;
        while (std::getline(bf, line)) {
          std::istringstream ls(line); std::string kind; ls >> kind;
          if (kind == "scalar") { std::string nm; double v; ls >> nm >> v; sc.push_back({nm, std::vector<float>(B, (float)v)}); }
          else if (kind == "scalarb") { std::string nm; ls >> nm; std::vector<float> vs; double x; while (ls >> x) vs.push_back((float)x); sc.push_back({nm, std::move(vs)}); }
          else if (kind == "matrix") { std::string nm; uint32_t rr, cc; ls >> nm >> rr >> cc; std::vector<float> d; double x; while (ls >> x) d.push_back((float)x); mt.push_back({nm, rr, cc, std::move(d)}); }
        }
      }
      for (long it = 0; it < n; ++it) {
        I.begin_forward();
        if (grad_mode) I.set_taping(true);
        for (auto& s : sc) I.bind_scalar_batched(s.first, s.second);
        for (auto& m : mt) I.bind_matrix(std::get<0>(m), std::get<1>(m), std::get<2>(m), std::get<3>(m));
        r = I.run(src);
      }
    } else {
      r = I.run(src);
    }
    bool numeric = I.is_num(r) || r.tag == ndvm::T::BOOLEAN;
    if (grad_mode && numeric) {           // reverse pass: per-lane d(result)/d(param) for each bound scalar
      I.backward(r);
      for (const auto& p : I.scalar_params()) {
        std::printf("grad %s", p.first.c_str());
        for (uint32_t b = 0; b < B; ++b) std::printf(" %.9g", static_cast<double>(I.grad_lane(p.second, b)));
        std::printf("\n");
      }
    }
    if (numeric) {  // per-lane outputs (one value per batch lane)
      std::printf("result");
      for (uint32_t b = 0; b < B; ++b) std::printf(" %.9g", static_cast<double>(I.num_lane(r, b)));
      std::printf("\n");
    }
    else if (r.tag == ndvm::T::VEC) {
      const ndvm::VecCell& v = I.vec(r);
      std::printf("vec %u %u %u", (unsigned)v.ndim, v.rows, v.cols);
      for (float x : v.data) std::printf(" %.9g", static_cast<double>(x));
      std::printf("\n");
    } else std::printf("result <non-numeric tag=%d>\n", (int)r.tag);
    std::fprintf(stderr, "diag eval_steps=%llu payload_allocs=%llu heap_pairs=%llu\n",
                 (unsigned long long)I.eval_steps(), (unsigned long long)I.payload_allocs(),
                 (unsigned long long)I.heap_pairs());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
