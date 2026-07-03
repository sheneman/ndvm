// NDVM Phase 6 scoped POC (forward numeric ceiling). See PHASE6_DESIGN.md section 4.
//
// Measures the GPU-vs-CPU crossover for a POPULATION of independent D-dimensional Kalman-filter NLL
// evaluations -- the dense-numeric rollout that would run on the GPU in the locked D2 persistent-kernel
// design (one block = one candidate; threads cooperate on the D x D linear algebra). Candidates differ in
// their fitted noise parameters (q, r); the dynamics F, the observation map H, and the observation
// sequence are shared (the Kalman MLE flagship: fit q, r to one data trajectory across many restarts).
//
// This is the FORWARD NUMERIC CEILING: no interpreter dispatch, no tape, no gradient -- the best case for
// the GPU, and the necessary condition for the full D2 interpreter to beat CPU. If the GPU cannot clear the
// 64-core CPU here at large D, the full backend (which only adds the branchy structural walk on top) stays
// deferred (the PHASE6_DESIGN.md kill criterion). Correctness: the GPU result matches a CPU reference of the
// identical math within float32 tolerance.
//
// Setup: state dim D (swept), observation dim m = 2 (so the innovation covariance S is 2x2, closed-form
// inverse + log-det -- no general LU, per the plan). F is a fixed dense D x D (so F P F^T is a real O(D^3)
// matmul, the GPU-favorable work); H observes the first two state components; Q = q I_D, R = r I_2.
//
// Build (sheneman partition): module load cuda/12.8 && nvcc -O3 -arch=sm_89 -Xcompiler -fopenmp \
//   ndvm/gpu/kalman_poc.cu -o /tmp/kalman_poc
// Run: NDVM_D=32 NDVM_G=8192 NDVM_T=80 NDVM_CPU_THREADS=64 /tmp/kalman_poc

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)

static const int MAXD = 64;   // shared-memory budget bound (P, Pm, tmp each D*D floats)

// ---------------------------------------------------------------------------
// Shared Kalman math, written once as a sequential reference (CPU). The GPU kernel below performs the
// identical operations cooperatively across a block's threads; both must agree within float32 tolerance.
// F: [D*D] row-major; obs: [T*2]; returns the negative log-likelihood for params (q, r).
// ---------------------------------------------------------------------------
static float kalman_nll_cpu(int D, int T, const float* F, const float* obs, float q, float r) {
  float x[MAXD]; float P[MAXD * MAXD]; float Pm[MAXD * MAXD]; float FP[MAXD * MAXD];
  for (int i = 0; i < D; ++i) { x[i] = 0.0f; for (int j = 0; j < D; ++j) P[i * D + j] = (i == j) ? 1.0f : 0.0f; }
  float nll = 0.0f;
  const float LOG2PI = 1.8378770664093453f;
  for (int t = 0; t < T; ++t) {
    // predict: xm = F x ; FP = F P ; Pm = FP F^T + qI
    float xm[MAXD];
    for (int i = 0; i < D; ++i) { float s = 0.0f; for (int k = 0; k < D; ++k) s += F[i * D + k] * x[k]; xm[i] = s; }
    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) { float s = 0.0f; for (int k = 0; k < D; ++k) s += F[i * D + k] * P[k * D + j]; FP[i * D + j] = s; }
    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j) { float s = 0.0f; for (int k = 0; k < D; ++k) s += FP[i * D + k] * F[j * D + k]; Pm[i * D + j] = s + (i == j ? q : 0.0f); }
    // update (m = 2, H = first two rows): innovation, S (2x2), inverse, gain, state, covariance, nll
    float y0 = obs[t * 2 + 0] - xm[0], y1 = obs[t * 2 + 1] - xm[1];
    float S00 = Pm[0 * D + 0] + r, S01 = Pm[0 * D + 1], S10 = Pm[1 * D + 0], S11 = Pm[1 * D + 1] + r;
    float det = S00 * S11 - S01 * S10;
    float iS00 = S11 / det, iS01 = -S01 / det, iS10 = -S10 / det, iS11 = S00 / det;
    // K = Pm H^T S^-1  (D x 2): Pm H^T = first two columns of Pm
    float K[MAXD * 2];
    for (int i = 0; i < D; ++i) {
      float a = Pm[i * D + 0], b = Pm[i * D + 1];
      K[i * 2 + 0] = a * iS00 + b * iS10;
      K[i * 2 + 1] = a * iS01 + b * iS11;
    }
    for (int i = 0; i < D; ++i) x[i] = xm[i] + K[i * 2 + 0] * y0 + K[i * 2 + 1] * y1;
    // P = Pm - K (H Pm) ; H Pm = first two ROWS of Pm
    for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j)
      P[i * D + j] = Pm[i * D + j] - (K[i * 2 + 0] * Pm[0 * D + j] + K[i * 2 + 1] * Pm[1 * D + j]);
    float quad = y0 * (iS00 * y0 + iS01 * y1) + y1 * (iS10 * y0 + iS11 * y1);
    nll += 0.5f * (quad + logf(det) + 2.0f * LOG2PI);
  }
  return nll;
}

// ---------------------------------------------------------------------------
// GPU: one block per candidate. F, obs are shared (global/constant); x, P, Pm, FP live in shared memory.
// Threads cooperate on the D x D matmuls (each thread owns a strided set of output elements). The 2x2 S
// solve + the scalar NLL accumulation are done by thread 0 (tiny, serial -- the leader-thread cost the
// design's idle-lane diagnosis is about; the POC measures exactly this).
// ---------------------------------------------------------------------------
__global__ void kalman_nll_kernel(int D, int T, const float* __restrict__ F, const float* __restrict__ obs,
                                  const float* __restrict__ qs, const float* __restrict__ rs,
                                  float* __restrict__ out, int G) {
  int g = blockIdx.x; if (g >= G) return;
  extern __shared__ float sh[];
  float* x  = sh;                  // D
  float* P  = x + D;               // D*D
  float* Pm = P + D * D;           // D*D
  float* FP = Pm + D * D;          // D*D
  float* sc = FP + D * D;          // scratch: xm[D] + K[D*2] + nll[1] + S/iS[8]
  float* xm = sc;                  // D
  float* K  = xm + D;              // D*2
  float* red = K + D * 2;          // 1 (nll accumulator)
  float* Sh  = red + 1;            // 8: S00 S01 S10 S11 iS00 iS01 iS10 iS11
  const int tid = threadIdx.x, nt = blockDim.x;
  const float q = qs[g], r = rs[g];
  const float LOG2PI = 1.8378770664093453f;

  for (int i = tid; i < D; i += nt) { x[i] = 0.0f; }
  for (int e = tid; e < D * D; e += nt) { int i = e / D, j = e % D; P[e] = (i == j) ? 1.0f : 0.0f; }
  if (tid == 0) red[0] = 0.0f;
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    for (int i = tid; i < D; i += nt) { float s = 0.0f; for (int k = 0; k < D; ++k) s += F[i * D + k] * x[k]; xm[i] = s; }
    __syncthreads();
    for (int e = tid; e < D * D; e += nt) { int i = e / D, j = e % D; float s = 0.0f; for (int k = 0; k < D; ++k) s += F[i * D + k] * P[k * D + j]; FP[e] = s; }
    __syncthreads();
    for (int e = tid; e < D * D; e += nt) { int i = e / D, j = e % D; float s = 0.0f; for (int k = 0; k < D; ++k) s += FP[i * D + k] * F[j * D + k]; Pm[e] = s + (i == j ? q : 0.0f); }
    __syncthreads();
    if (tid == 0) {
      float S00 = Pm[0] + r, S01 = Pm[1], S10 = Pm[D], S11 = Pm[D + 1] + r;
      float det = S00 * S11 - S01 * S10;
      Sh[4] = S11 / det; Sh[5] = -S01 / det; Sh[6] = -S10 / det; Sh[7] = S00 / det; Sh[0] = det;
    }
    __syncthreads();
    float iS00 = Sh[4], iS01 = Sh[5], iS10 = Sh[6], iS11 = Sh[7];
    for (int i = tid; i < D; i += nt) {
      float a = Pm[i * D + 0], b = Pm[i * D + 1];
      K[i * 2 + 0] = a * iS00 + b * iS10;
      K[i * 2 + 1] = a * iS01 + b * iS11;
    }
    __syncthreads();
    float y0 = obs[t * 2 + 0] - xm[0], y1 = obs[t * 2 + 1] - xm[1];
    for (int i = tid; i < D; i += nt) x[i] = xm[i] + K[i * 2 + 0] * y0 + K[i * 2 + 1] * y1;
    for (int e = tid; e < D * D; e += nt) { int i = e / D, j = e % D;
      FP[e] = Pm[e] - (K[i * 2 + 0] * Pm[0 * D + j] + K[i * 2 + 1] * Pm[1 * D + j]); }   // FP reused for new P
    __syncthreads();
    for (int e = tid; e < D * D; e += nt) P[e] = FP[e];
    if (tid == 0) {
      float quad = y0 * (iS00 * y0 + iS01 * y1) + y1 * (iS10 * y0 + iS11 * y1);
      red[0] += 0.5f * (quad + logf(Sh[0]) + 2.0f * LOG2PI);
    }
    __syncthreads();
  }
  if (tid == 0) out[g] = red[0];
}

int main() {
  int D = getenv("NDVM_D") ? atoi(getenv("NDVM_D")) : 32;
  int G = getenv("NDVM_G") ? atoi(getenv("NDVM_G")) : 8192;
  int T = getenv("NDVM_T") ? atoi(getenv("NDVM_T")) : 80;
  int cpuThreads = getenv("NDVM_CPU_THREADS") ? atoi(getenv("NDVM_CPU_THREADS")) : (int)std::thread::hardware_concurrency();
  if (D < 2 || D > MAXD) { std::fprintf(stderr, "D in [2,%d]\n", MAXD); return 2; }

  // fixed dense F (mild coupling so F P F^T is a real D x D matmul), H = first 2 rows, synthetic obs,
  // candidate (q, r) sweeps. Deterministic (no RNG) so runs are reproducible.
  std::vector<float> F(D * D), obs(T * 2), qs(G), rs(G);
  // Stable dynamics at any D: diagonal 0.9, off-diagonal coupling scaled by 1/D so row sums stay bounded
  // (spectral radius < 1), keeping the filter well-conditioned so the float32 covariance does not blow up.
  for (int i = 0; i < D; ++i) for (int j = 0; j < D; ++j)
    F[i * D + j] = (i == j ? 0.9f : (0.3f / D) * cosf(0.7f * (i + 1) * (j + 1)));
  for (int t = 0; t < T; ++t) { obs[t * 2 + 0] = sinf(0.3f * t); obs[t * 2 + 1] = cosf(0.2f * t) * 0.5f; }
  for (int g = 0; g < G; ++g) { qs[g] = 0.05f * (1.0f + 0.0007f * g); rs[g] = 0.10f * (1.0f + 0.0005f * g); }

  // ---- GPU ----
  float *dF, *dObs, *dQ, *dR, *dOut;
  CK(cudaMalloc(&dF, F.size() * 4)); CK(cudaMalloc(&dObs, obs.size() * 4));
  CK(cudaMalloc(&dQ, G * 4)); CK(cudaMalloc(&dR, G * 4)); CK(cudaMalloc(&dOut, G * 4));
  CK(cudaMemcpy(dF, F.data(), F.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dObs, obs.data(), obs.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dQ, qs.data(), G * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dR, rs.data(), G * 4, cudaMemcpyHostToDevice));
  int tpb = 256;
  size_t shmem = (size_t)(D + 3 * D * D + D + 2 * D + 1 + 8) * sizeof(float);
  CK(cudaFuncSetAttribute(kalman_nll_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
  kalman_nll_kernel<<<G, tpb, shmem>>>(D, T, dF, dObs, dQ, dR, dOut, G);  // warm-up
  CK(cudaDeviceSynchronize());
  auto g0 = std::chrono::steady_clock::now();
  kalman_nll_kernel<<<G, tpb, shmem>>>(D, T, dF, dObs, dQ, dR, dOut, G);
  CK(cudaDeviceSynchronize());
  auto g1 = std::chrono::steady_clock::now();
  double gms = std::chrono::duration<double, std::milli>(g1 - g0).count();
  std::vector<float> gout(G); CK(cudaMemcpy(gout.data(), dOut, G * 4, cudaMemcpyDeviceToHost));

  // ---- CPU (multithreaded over candidates) ----
  std::vector<float> cout(G);
  auto c0 = std::chrono::steady_clock::now();
  {
    std::atomic<int> next{0};
    auto worker = [&]() { for (;;) { int g = next.fetch_add(1); if (g >= G) break; cout[g] = kalman_nll_cpu(D, T, F.data(), obs.data(), qs[g], rs[g]); } };
    std::vector<std::thread> pool;
    for (int w = 0; w < cpuThreads; ++w) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }
  auto c1 = std::chrono::steady_clock::now();
  double cms = std::chrono::duration<double, std::milli>(c1 - c0).count();

  // ---- correctness: GPU vs CPU within float32 tolerance ----
  double maxrel = 0.0; int bad = 0;
  for (int g = 0; g < G; ++g) {
    double a = gout[g], b = cout[g], rel = fabs(a - b) / (fabs(b) + 1e-6);
    if (rel > maxrel) maxrel = rel;
    if (rel > 2e-3) ++bad;
  }
  double gtps = G / (gms / 1000.0), ctps = G / (cms / 1000.0);
  std::printf("D=%d G=%d T=%d cpu_threads=%d | GPU %.3f ms (%.0f evals/s) | CPU %.3f ms (%.0f evals/s) | "
              "speedup %.2fx | max_rel_err %.2e bad=%d\n",
              D, G, T, cpuThreads, gms, gtps, cms, ctps, gtps / ctps, maxrel, bad);
  cudaFree(dF); cudaFree(dObs); cudaFree(dQ); cudaFree(dR); cudaFree(dOut);
  return bad == 0 ? 0 : 1;
}
