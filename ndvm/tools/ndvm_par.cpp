// ndvm_par: Phase-5 parallel driver. Builds a population of N distinct candidate tasks from one program
// (each task perturbs the bound scalars deterministically), runs them through evaluate_batch across
// NDVM_THREADS workers, and either dumps each task's result as raw hex bit patterns (NDVM_PAR_DUMP, for
// the determinism gate: the dump must be byte-identical across thread counts) or reports throughput
// (the scaling gate). NDVM_PAR_N tasks (default 1000); NDVM_B inner batch width (default 1).
#include "parallel.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <tuple>

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static unsigned long long bits(double d) { unsigned long long u; std::memcpy(&u, &d, sizeof(u)); return u; }

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: ndvm_par <program> [binds]\n"); return 2; }
  std::string src = slurp(argv[1]);
  uint32_t B = 1; if (const char* nb = std::getenv("NDVM_B")) { long v = std::strtol(nb, nullptr, 10); if (v > 0) B = (uint32_t)v; }
  long N = 1000;  if (const char* nn = std::getenv("NDVM_PAR_N")) { long v = std::strtol(nn, nullptr, 10); if (v > 0) N = v; }
  int threads = 0; if (const char* nt = std::getenv("NDVM_THREADS")) threads = (int)std::strtol(nt, nullptr, 10);
  bool dump = std::getenv("NDVM_PAR_DUMP") != nullptr;

  std::vector<std::pair<std::string, float>> sc;
  std::vector<std::tuple<std::string, uint32_t, uint32_t, std::vector<float>>> mt;
  if (argc >= 3) {
    std::ifstream bf(argv[2]); std::string line;
    while (std::getline(bf, line)) {
      std::istringstream ls(line); std::string k; ls >> k;
      if (k == "scalar") { std::string n; double v; ls >> n >> v; sc.push_back({n, (float)v}); }
      else if (k == "matrix") { std::string n; uint32_t r, c; ls >> n >> r >> c; std::vector<float> d; double x; while (ls >> x) d.push_back((float)x); mt.push_back({n, r, c, std::move(d)}); }
    }
  }

  // N distinct tasks: task k scales every bound scalar by (1 + 0.000123*k); lane b adds (1 + 0.01*b).
  std::vector<ndvm::Task> tasks((size_t)N);
  for (long k = 0; k < N; ++k) {
    ndvm::Task& t = tasks[(size_t)k];
    t.src = src; t.B = B; t.want_grad = true;
    for (auto& s : sc) {
      t.snames.push_back(s.first);
      for (uint32_t b = 0; b < B; ++b) t.svals.push_back(s.second * (1.0f + 0.000123f * (float)k) * (1.0f + 0.01f * (float)b));
    }
    for (auto& m : mt) { t.mnames.push_back(std::get<0>(m)); t.mrows.push_back(std::get<1>(m)); t.mcols.push_back(std::get<2>(m)); t.mdata.push_back(std::get<3>(m)); }
  }

  auto t0 = std::chrono::steady_clock::now();
  std::vector<ndvm::Result> res = ndvm::evaluate_batch(tasks, threads);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  if (dump) {
    for (long k = 0; k < N; ++k) {
      const ndvm::Result& r = res[(size_t)k];
      std::printf("t%ld ok=%d", k, r.ok ? 1 : 0);
      for (double o : r.outs) std::printf(" o%016llx", bits(o));
      for (double g : r.grads) std::printf(" g%016llx", bits(g));
      std::printf("\n");
    }
  } else {
    long okc = 0; for (auto& r : res) if (r.ok) ++okc;
    std::printf("par N=%ld threads=%d B=%u ok=%ld wall=%.3f ms throughput=%.1f tasks/s\n",
                N, threads, B, okc, ms, (double)N / (ms / 1000.0));
  }
  return 0;
}
