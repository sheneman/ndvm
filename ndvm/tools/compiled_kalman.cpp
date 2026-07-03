// compiled_kalman: the native compiled-differentiable ceiling for the 80-step 2x2 Kalman NLL. This is
// the C++ analog of the JAX/XLA-staged baseline: the program is COMPILED CODE, not interpreted data.
// The Kalman objective is hand-coded directly in C++ and differentiated by hand-coded forward-mode dual
// numbers over the two parameters (q, r) -- exact analytic gradients, the right method for few inputs.
// There is no interpreter, no parser, no program-as-data: this is the floor a search loop could only
// reach by giving up program-as-data and compiling each candidate, which is exactly the route NDVM
// declines. It exists to bound how far below the interpreter the native compiled ceiling sits.
//
// Validated: forward NLL must match the DMCI/NDVM oracle (831.22 at q=0.05, r=0.10, obs=seed-0 randn),
// and (dNLL/dq, dNLL/dr) must match NDVM's reverse-mode gradient to float tolerance.
//
// CLI: compiled_kalman <bindings_file>     (reads scalar q, scalar r, matrix obs T 2 ...)
//   COMPILED_BENCH=<N> COMPILED_MODE=fwd|grad   times N evals of forward-only or forward+gradient.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---- forward-mode dual scalar: value + derivative wrt q + derivative wrt r ----
struct Dual { float v, dq, dr; };
static inline Dual dc(float c) { return {c, 0.0f, 0.0f}; }
static inline Dual dadd(Dual a, Dual b) { return {a.v + b.v, a.dq + b.dq, a.dr + b.dr}; }
static inline Dual dsub(Dual a, Dual b) { return {a.v - b.v, a.dq - b.dq, a.dr - b.dr}; }
static inline Dual dneg(Dual a) { return {-a.v, -a.dq, -a.dr}; }
static inline Dual dmul(Dual a, Dual b) { return {a.v * b.v, a.dq * b.v + a.v * b.dq, a.dr * b.v + a.v * b.dr}; }
static inline Dual ddiv(Dual a, Dual b) { float iv = 1.0f / b.v; float iv2 = iv * iv;
  return {a.v * iv, (a.dq * b.v - a.v * b.dq) * iv2, (a.dr * b.v - a.v * b.dr) * iv2}; }
static inline Dual dlog(Dual a) { float iv = 1.0f / a.v; return {std::log(a.v), a.dq * iv, a.dr * iv}; }

// 2x2 matrix and 2-vector of duals (row-major).
struct M2 { Dual a[4]; };
struct V2 { Dual a[2]; };
static M2 mat_add(const M2& A, const M2& B) { M2 r; for (int i = 0; i < 4; ++i) r.a[i] = dadd(A.a[i], B.a[i]); return r; }
static M2 mat_sub(const M2& A, const M2& B) { M2 r; for (int i = 0; i < 4; ++i) r.a[i] = dsub(A.a[i], B.a[i]); return r; }
static V2 vec_sub(const V2& A, const V2& B) { V2 r; for (int i = 0; i < 2; ++i) r.a[i] = dsub(A.a[i], B.a[i]); return r; }
static V2 vec_add(const V2& A, const V2& B) { V2 r; for (int i = 0; i < 2; ++i) r.a[i] = dadd(A.a[i], B.a[i]); return r; }
static Dual det2(const M2& S) { return dsub(dmul(S.a[0], S.a[3]), dmul(S.a[1], S.a[2])); }
static M2 inv2(const M2& S) { Dual det = det2(S); Dual id = ddiv(dc(1.0f), det);
  M2 r; r.a[0] = dmul(S.a[3], id); r.a[1] = dmul(dneg(S.a[1]), id); r.a[2] = dmul(dneg(S.a[2]), id); r.a[3] = dmul(S.a[0], id); return r; }
static M2 mat_mul(const M2& A, const M2& B) { M2 r;
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) r.a[i * 2 + j] = dadd(dmul(A.a[i * 2], B.a[j]), dmul(A.a[i * 2 + 1], B.a[2 + j]));
  return r; }
static V2 mat_vec(const M2& A, const V2& v) { V2 r;
  for (int i = 0; i < 2; ++i) r.a[i] = dadd(dmul(A.a[i * 2], v.a[0]), dmul(A.a[i * 2 + 1], v.a[1])); return r; }
static Dual dot2(const V2& a, const V2& b) { return dadd(dmul(a.a[0], b.a[0]), dmul(a.a[1], b.a[1])); }

// Forward + exact gradient (dual): returns L with L.v = NLL, L.dq = dNLL/dq, L.dr = dNLL/dr.
static Dual kalman_fwdgrad(float q0, float r0, const float* obs, int T) {
  Dual q = {q0, 1.0f, 0.0f}, r = {r0, 0.0f, 1.0f};
  V2 x; x.a[0] = dc(0.0f); x.a[1] = dc(0.0f);
  M2 P; P.a[0] = dc(1.0f); P.a[1] = dc(0.0f); P.a[2] = dc(0.0f); P.a[3] = dc(1.0f);
  Dual L = dc(0.0f);
  for (int k = 0; k < T; ++k) {
    M2 Q; Q.a[0] = q; Q.a[1] = dc(0.0f); Q.a[2] = dc(0.0f); Q.a[3] = q;
    M2 R; R.a[0] = r; R.a[1] = dc(0.0f); R.a[2] = dc(0.0f); R.a[3] = r;
    M2 Ppred = mat_add(P, Q);
    V2 y; y.a[0] = dc(obs[k * 2]); y.a[1] = dc(obs[k * 2 + 1]);
    V2 e = vec_sub(y, x);
    M2 S = mat_add(Ppred, R);
    M2 Sinv = inv2(S);
    M2 Kg = mat_mul(Ppred, Sinv);
    x = vec_add(x, mat_vec(Kg, e));
    M2 I; I.a[0] = dc(1.0f); I.a[1] = dc(0.0f); I.a[2] = dc(0.0f); I.a[3] = dc(1.0f);
    P = mat_mul(mat_sub(I, Kg), Ppred);
    Dual nll = dadd(dlog(det2(S)), dot2(e, mat_vec(Sinv, e)));
    L = dadd(L, nll);
  }
  return L;
}

// Forward only (plain float; the pure native forward ceiling, no derivative bookkeeping).
static float kalman_fwd(float q, float r, const float* obs, int T) {
  float x0 = 0, x1 = 0, P00 = 1, P01 = 0, P10 = 0, P11 = 1, L = 0;
  for (int k = 0; k < T; ++k) {
    float Pp00 = P00 + q, Pp01 = P01, Pp10 = P10, Pp11 = P11 + q;
    float e0 = obs[k * 2] - x0, e1 = obs[k * 2 + 1] - x1;
    float S00 = Pp00 + r, S01 = Pp01, S10 = Pp10, S11 = Pp11 + r;
    float det = S00 * S11 - S01 * S10, id = 1.0f / det;
    float Si00 = S11 * id, Si01 = -S01 * id, Si10 = -S10 * id, Si11 = S00 * id;
    float Kg00 = Pp00 * Si00 + Pp01 * Si10, Kg01 = Pp00 * Si01 + Pp01 * Si11;
    float Kg10 = Pp10 * Si00 + Pp11 * Si10, Kg11 = Pp10 * Si01 + Pp11 * Si11;
    x0 = x0 + Kg00 * e0 + Kg01 * e1; x1 = x1 + Kg10 * e0 + Kg11 * e1;
    float A00 = 1 - Kg00, A01 = -Kg01, A10 = -Kg10, A11 = 1 - Kg11;
    float nP00 = A00 * Pp00 + A01 * Pp10, nP01 = A00 * Pp01 + A01 * Pp11;
    float nP10 = A10 * Pp00 + A11 * Pp10, nP11 = A10 * Pp01 + A11 * Pp11;
    P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    float Sie0 = Si00 * e0 + Si01 * e1, Sie1 = Si10 * e0 + Si11 * e1;
    L += std::log(det) + (e0 * Sie0 + e1 * Sie1);
  }
  return L;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: compiled_kalman <bindings_file>\n"); return 2; }
  float q = 0.05f, r = 0.10f; int T = 0; std::vector<float> obs;
  { std::ifstream bf(argv[1]); std::string line;
    while (std::getline(bf, line)) { std::istringstream ls(line); std::string kind; ls >> kind;
      if (kind == "scalar") { std::string nm; double v; ls >> nm >> v; if (nm == "q") q = (float)v; else if (nm == "r") r = (float)v; }
      else if (kind == "matrix") { std::string nm; uint32_t rr, cc; ls >> nm >> rr >> cc; T = (int)rr;
        double x; while (ls >> x) obs.push_back((float)x); } } }
  if (T == 0) { std::fprintf(stderr, "ERROR: no obs matrix in bindings\n"); return 1; }
  Dual L = kalman_fwdgrad(q, r, obs.data(), T);
  std::printf("NLL %.6f  dNLL/dq %.9g  dNLL/dr %.9g  (T=%d q=%g r=%g)\n", L.v, (double)L.dq, (double)L.dr, T, q, r);

  if (const char* nb = std::getenv("COMPILED_BENCH")) {
    long n = std::strtol(nb, nullptr, 10); if (n < 1) n = 1;
    const char* mode = std::getenv("COMPILED_MODE"); bool grad = mode && std::string(mode) == "grad";
    volatile float sink = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < n; ++i) { if (grad) sink += kalman_fwdgrad(q, r, obs.data(), T).v; else sink += kalman_fwd(q, r, obs.data(), T); }
    auto t1 = std::chrono::steady_clock::now(); (void)sink;
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)n;
    std::printf("bench %s %.6g ms/eval over %ld runs\n", grad ? "fwd+grad" : "fwd", ms, n);
  }
  return 0;
}
