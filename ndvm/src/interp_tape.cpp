// NDVM Phase 2: native reverse-mode AD. Tape recording + backward replay.
// The forward records one TNode per differentiable numeric op (structural ops record nothing);
// backward sizes the adjoint buffers, seeds the scalar output adjoint to 1, and replays the tape in
// reverse, accumulating per-op VJPs into adj_ (scalars) and vadj_ (VecCells). VJP formulas match the
// torch autograd of the corresponding primitive (see neural_compiler/ops/primitives.py), incl. the
// 1e-8 clamp subgradients (grad 0 below the clamp) and the matrix adjoints for det/logdet/inv.
#include "interp.hpp"
#include <cmath>

namespace ndvm {

void Interp::rec(Op op, const Val& out, std::initializer_list<Val> ins, uint32_t aux) {
  if (!taping_) return;
  TNode n; n.op = op; n.out = ref_of(out); n.aux = aux;
  n.actset = active_full_ ? ACTSET_FULL : intern_actset(active_lanes_);   // lanes live when recorded
  for (const Val& v : ins) n.ins.push_back(ref_of(v));
  tape_.push_back(std::move(n));
}

void Interp::rec_v(Op op, const Val& out, const std::vector<Val>& ins, uint32_t aux) {
  if (!taping_) return;
  TNode n; n.op = op; n.out = ref_of(out); n.aux = aux;
  n.actset = active_full_ ? ACTSET_FULL : intern_actset(active_lanes_);   // lanes live when recorded
  for (const Val& v : ins) n.ins.push_back(ref_of(v));
  tape_.push_back(std::move(n));
}

void Interp::backward(const Val& output) {
  adj_.assign(primal_.size(), 0.0f);            // primal_ is B-strided; adj_ mirrors it
  vadj_.assign(vecs_.size(), {});
  for (size_t i = 0; i < vecs_.size(); ++i) vadj_[i].assign(vecs_[i].data.size(), 0.0f);
  // Seed d(output)/d(output)=1 on ALL B lanes (the native equivalent of loss=output.sum(); .backward()).
  if (output.tag == T::INT || output.tag == T::FLOAT) { float* a = adj_at(output.pid); for (uint32_t b = 0; b < B_; ++b) a[b] = 1.0f; }
  else if (output.tag == T::VEC) for (float& x : vadj_[output.aux]) x = 1.0f;
  for (size_t i = tape_.size(); i-- > 0;) dispatch_adjoint(tape_[i]);
}

void Interp::dispatch_adjoint(const TNode& n) {
  const std::vector<Ref>& in = n.ins;
  // per-lane scalar accessors (B-strided)
  auto P = [&](const Ref& r, uint32_t b) -> float { return primal_[(size_t)r.id * B_ + b]; };
  auto A = [&](const Ref& r, uint32_t b) -> float& { return adj_[(size_t)r.id * B_ + b]; };
  // per-lane vector/matrix slab accessors (lane b occupies VSZ(r) contiguous floats)
  auto VSZ = [&](const Ref& r) -> size_t { return (size_t)vecs_[r.id].rows * vecs_[r.id].cols; };
  auto VP = [&](const Ref& r, uint32_t b) -> const float* { return &vecs_[r.id].data[(size_t)b * VSZ(r)]; };
  auto VA = [&](const Ref& r, uint32_t b) -> float* { return &vadj_[r.id][(size_t)b * VSZ(r)]; };

  // Phase 3b: replay each VJP over the active lanes the node was recorded under. FULL (the uniform/B=1
  // case) loops 0..B_ exactly as Phase 3 -- the lambda inlines to the original counter loop, so values
  // are byte-identical. A reduced set replays only its lanes, so an inactive (terminated) lane's
  // stale/non-finite primal is never read and never accumulates into a shared leaf (no 0*NaN).
  const bool full = (n.actset == ACTSET_FULL);
  auto each = [&](auto&& f) {
    if (full) { for (uint32_t b = 0; b < B_; ++b) f(b); }
    else { for (uint32_t b : actset_pool_[n.actset]) f(b); }
  };

  switch (n.op) {
    // ---- scalar arithmetic (variadic), per lane ----
    case Op::ADD: each([&](uint32_t b){ float g = A(n.out, b); for (auto& r : in) A(r, b) += g; }); break;
    case Op::SUB: each([&](uint32_t b){ float g = A(n.out, b);
      if (in.size() == 1) { A(in[0], b) -= g; }
      else { A(in[0], b) += g; for (size_t i = 1; i < in.size(); ++i) A(in[i], b) -= g; } }); break;
    case Op::MUL: each([&](uint32_t b){ float g = A(n.out, b);
      for (size_t i = 0; i < in.size(); ++i) { float p = 1.0f;
        for (size_t j = 0; j < in.size(); ++j) if (j != i) p *= P(in[j], b);
        A(in[i], b) += g * p; } }); break;
    case Op::DIV: each([&](uint32_t b){ float g = A(n.out, b);
      if (in.size() == 1) { float a = P(in[0], b); A(in[0], b) -= g / (a * a); }
      else { float D = 1.0f; for (size_t i = 1; i < in.size(); ++i) D *= P(in[i], b);
        A(in[0], b) += g / D; float z = P(n.out, b);
        for (size_t i = 1; i < in.size(); ++i) A(in[i], b) -= g * z / P(in[i], b); } }); break;

    // ---- transcendental / clamped, per lane (clamp swallows grad below 1e-8) ----
    case Op::SIN: each([&](uint32_t b){ A(in[0], b) += A(n.out, b) * std::cos(P(in[0], b)); }); break;
    case Op::COS: each([&](uint32_t b){ A(in[0], b) -= A(n.out, b) * std::sin(P(in[0], b)); }); break;
    case Op::EXP: each([&](uint32_t b){ A(in[0], b) += A(n.out, b) * P(n.out, b); }); break;
    case Op::SQRT: each([&](uint32_t b){ if (P(in[0], b) > 1e-8f) A(in[0], b) += 0.5f * A(n.out, b) / P(n.out, b); }); break;
    case Op::LOG:  each([&](uint32_t b){ if (P(in[0], b) > 1e-8f) A(in[0], b) += A(n.out, b) / P(in[0], b); }); break;
    case Op::POW:  each([&](uint32_t b){ float a = P(in[0], b), e = P(in[1], b), z = P(n.out, b), g = A(n.out, b);
      A(in[0], b) += g * e * z / a; A(in[1], b) += g * z * std::log(a); }); break;
    case Op::ABS:  each([&](uint32_t b){ float a = P(in[0], b); A(in[0], b) += A(n.out, b) * ((a > 0) - (a < 0)); }); break;
    case Op::MINB: each([&](uint32_t b){ float a = P(in[0], b), c = P(in[1], b), g = A(n.out, b);
      if (a < c) A(in[0], b) += g; else if (a > c) A(in[1], b) += g; else { A(in[0], b) += g; A(in[1], b) += g; } }); break;
    case Op::MAXB: each([&](uint32_t b){ float a = P(in[0], b), c = P(in[1], b), g = A(n.out, b);
      if (a > c) A(in[0], b) += g; else if (a < c) A(in[1], b) += g; else { A(in[0], b) += g; A(in[1], b) += g; } }); break;
    case Op::MOD:  each([&](uint32_t b){ float g = A(n.out, b); A(in[0], b) += g; A(in[1], b) -= g * std::trunc(P(in[0], b) / P(in[1], b)); }); break;
    case Op::REM:  each([&](uint32_t b){ float g = A(n.out, b); A(in[0], b) += g; A(in[1], b) -= g * std::floor(P(in[0], b) / P(in[1], b)); }); break;

    // ---- elementwise vec/mat (per lane; sz = per-lane element count) ----
    case Op::EW_ADD: each([&](uint32_t b){ size_t sz = VSZ(n.out); const float* g = VA(n.out, b);
      for (auto& r : in) { float* d = VA(r, b); for (size_t k = 0; k < sz; ++k) d[k] += g[k]; } }); break;
    case Op::EW_SUB: each([&](uint32_t b){ size_t sz = VSZ(n.out); const float* g = VA(n.out, b);
      { float* d0 = VA(in[0], b); for (size_t k = 0; k < sz; ++k) d0[k] += g[k]; }
      for (size_t i = 1; i < in.size(); ++i) { float* d = VA(in[i], b); for (size_t k = 0; k < sz; ++k) d[k] -= g[k]; } }); break;
    case Op::EW_NEG: each([&](uint32_t b){ size_t sz = VSZ(n.out); const float* g = VA(n.out, b); float* d = VA(in[0], b);
      for (size_t k = 0; k < sz; ++k) d[k] -= g[k]; }); break;
    case Op::EW_MUL: each([&](uint32_t b){ size_t sz = VSZ(n.out); const float* g = VA(n.out, b);
      float* da = VA(in[0], b); float* db = VA(in[1], b); const float* pa = VP(in[0], b); const float* pb = VP(in[1], b);
      for (size_t k = 0; k < sz; ++k) { da[k] += g[k] * pb[k]; db[k] += g[k] * pa[k]; } }); break;
    case Op::EW_DIV: each([&](uint32_t b){ size_t sz = VSZ(n.out); const float* g = VA(n.out, b);
      float* da = VA(in[0], b); float* db = VA(in[1], b); const float* pa = VP(in[0], b); const float* pb = VP(in[1], b);
      for (size_t k = 0; k < sz; ++k) { da[k] += g[k] / pb[k]; db[k] -= g[k] * pa[k] / (pb[k] * pb[k]); } }); break;
    case Op::SCALE: { const Ref& s = in[0].is_vec ? in[1] : in[0]; const Ref& M = in[0].is_vec ? in[0] : in[1];
      each([&](uint32_t b){ size_t sz = VSZ(M); const float* g = VA(n.out, b); float* dM = VA(M, b); const float* pM = VP(M, b);
        float sv = P(s, b), ds = 0.0f; for (size_t k = 0; k < sz; ++k) { dM[k] += g[k] * sv; ds += g[k] * pM[k]; } A(s, b) += ds; }); } break;

    // ---- reductions / products (per lane; reduce over the feature dim only) ----
    case Op::DOT: each([&](uint32_t b){ size_t sz = VSZ(in[0]); float g = A(n.out, b);
      float* da = VA(in[0], b); float* db = VA(in[1], b); const float* pa = VP(in[0], b); const float* pb = VP(in[1], b);
      for (size_t k = 0; k < sz; ++k) { da[k] += g * pb[k]; db[k] += g * pa[k]; } }); break;
    case Op::VSUM: each([&](uint32_t b){ size_t sz = VSZ(in[0]); float g = A(n.out, b); float* d = VA(in[0], b);
      for (size_t k = 0; k < sz; ++k) d[k] += g; }); break;
    case Op::NORM: each([&](uint32_t b){ size_t sz = VSZ(in[0]); float g = A(n.out, b), nz = P(n.out, b);
      const float* pv = VP(in[0], b); float* d = VA(in[0], b); if (nz > 0) for (size_t k = 0; k < sz; ++k) d[k] += g * pv[k] / nz; }); break;
    case Op::NORMALIZE: each([&](uint32_t b){ size_t sz = VSZ(in[0]); const float* g = VA(n.out, b); const float* pv = VP(in[0], b); float* d = VA(in[0], b);
      float s2 = 0; for (size_t k = 0; k < sz; ++k) s2 += pv[k] * pv[k]; float nrm = std::sqrt(s2), nn = std::max(nrm, 1e-8f);
      if (nrm > 1e-8f) { float gv = 0; for (size_t k = 0; k < sz; ++k) gv += g[k] * pv[k];
        for (size_t k = 0; k < sz; ++k) d[k] += g[k] / nn - gv * pv[k] / (nn * nn * nn); }
      else { for (size_t k = 0; k < sz; ++k) d[k] += g[k] / nn; } }); break;
    case Op::CROSS: each([&](uint32_t b){ const float* g = VA(n.out, b); const float* pa = VP(in[0], b); const float* pb = VP(in[1], b);
      float* da = VA(in[0], b); float* db = VA(in[1], b);
      da[0] += g[2] * pb[1] - g[1] * pb[2]; da[1] += g[0] * pb[2] - g[2] * pb[0]; da[2] += g[1] * pb[0] - g[0] * pb[1];
      db[0] += g[1] * pa[2] - g[2] * pa[1]; db[1] += g[2] * pa[0] - g[0] * pa[2]; db[2] += g[0] * pa[1] - g[1] * pa[0]; }); break;

    // ---- matrix (per lane) ----
    case Op::MATMUL: { uint32_t m = vecs_[in[0].id].rows, k = vecs_[in[0].id].cols, nn = vecs_[in[1].id].cols;
      each([&](uint32_t b){ const float* g = VA(n.out, b); const float* pA = VP(in[0], b); const float* pB = VP(in[1], b); float* dA = VA(in[0], b); float* dB = VA(in[1], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t p = 0; p < k; ++p) { float acc = 0; for (uint32_t j = 0; j < nn; ++j) acc += g[i * nn + j] * pB[p * nn + j]; dA[i * k + p] += acc; }
        for (uint32_t p = 0; p < k; ++p) for (uint32_t j = 0; j < nn; ++j) { float acc = 0; for (uint32_t i = 0; i < m; ++i) acc += pA[i * k + p] * g[i * nn + j]; dB[p * nn + j] += acc; } }); } break;
    case Op::MATVEC: { uint32_t m = vecs_[in[0].id].rows, k = vecs_[in[0].id].cols;
      each([&](uint32_t b){ const float* g = VA(n.out, b); const float* pA = VP(in[0], b); const float* pv = VP(in[1], b); float* dA = VA(in[0], b); float* dv = VA(in[1], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t p = 0; p < k; ++p) dA[i * k + p] += g[i] * pv[p];
        for (uint32_t p = 0; p < k; ++p) { float acc = 0; for (uint32_t i = 0; i < m; ++i) acc += pA[i * k + p] * g[i]; dv[p] += acc; } }); } break;
    case Op::TRANSPOSE: { uint32_t m = vecs_[in[0].id].rows, nn = vecs_[in[0].id].cols;
      each([&](uint32_t b){ const float* g = VA(n.out, b); float* dA = VA(in[0], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < nn; ++j) dA[i * nn + j] += g[j * m + i]; }); } break;
    case Op::TRACE: { uint32_t m = vecs_[in[0].id].rows, cc = vecs_[in[0].id].cols, dmin = std::min(m, cc);
      each([&](uint32_t b){ float g = A(n.out, b); float* dA = VA(in[0], b); for (uint32_t i = 0; i < dmin; ++i) dA[i * cc + i] += g; }); } break;
    case Op::OUTER: { uint32_t na = vecs_[in[0].id].cols, nb = vecs_[in[1].id].cols;
      each([&](uint32_t b){ const float* g = VA(n.out, b); const float* pa = VP(in[0], b); const float* pb = VP(in[1], b); float* da = VA(in[0], b); float* db = VA(in[1], b);
        for (uint32_t i = 0; i < na; ++i) { float acc = 0; for (uint32_t j = 0; j < nb; ++j) acc += g[i * nb + j] * pb[j]; da[i] += acc; }
        for (uint32_t j = 0; j < nb; ++j) { float acc = 0; for (uint32_t i = 0; i < na; ++i) acc += g[i * nb + j] * pa[i]; db[j] += acc; } }); } break;

    // ---- LU-based (per lane): det/logdet read the cached inverse slab in aux; inv uses its output ----
    case Op::DET: { uint32_t m = vecs_[in[0].id].rows;
      each([&](uint32_t b){ float g = A(n.out, b), det = P(n.out, b); const float* inv = &vecs_[n.aux].data[(size_t)b * m * m]; float* dA = VA(in[0], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < m; ++j) dA[i * m + j] += g * det * inv[j * m + i]; }); } break;
    case Op::LOGDET: { uint32_t m = vecs_[in[0].id].rows;
      each([&](uint32_t b){ float g = A(n.out, b); const float* inv = &vecs_[n.aux].data[(size_t)b * m * m]; float* dA = VA(in[0], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < m; ++j) dA[i * m + j] += g * inv[j * m + i]; }); } break;
    case Op::INV: { uint32_t m = vecs_[n.out.id].rows;
      each([&](uint32_t b){ const float* Bm = VP(n.out, b); const float* g = VA(n.out, b); float* dA = VA(in[0], b);
        for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < m; ++j) { float acc = 0;
          for (uint32_t p = 0; p < m; ++p) for (uint32_t q = 0; q < m; ++q) acc += Bm[p * m + i] * g[p * m + q] * Bm[j * m + q];
          dA[i * m + j] -= acc; } }); } break;

    // ---- constructors / gather (per lane) ----
    case Op::VEC: each([&](uint32_t b){ const float* g = VA(n.out, b); for (size_t i = 0; i < in.size(); ++i) A(in[i], b) += g[i]; }); break;
    case Op::MAT: { uint32_t nn = vecs_[n.out.id].cols;
      each([&](uint32_t b){ const float* g = VA(n.out, b); for (size_t rr = 0; rr < in.size(); ++rr) { float* dr = VA(in[rr], b); for (uint32_t c = 0; c < nn; ++c) dr[c] += g[rr * nn + c]; } }); } break;
    case Op::REF: { uint32_t i = n.aux;
      each([&](uint32_t b){ float* ds = VA(in[0], b);
        if (!n.out.is_vec) { ds[i] += A(n.out, b); }
        else { uint32_t cols = vecs_[in[0].id].cols; const float* go = VA(n.out, b); for (uint32_t c = 0; c < cols; ++c) ds[i * cols + c] += go[c]; } }); } break;

    // ---- Phase 3b: per-lane merge at a divergent branch ----
    // out[b] came from in[0] (v_then) on then_lanes (= actset_pool_[aux]); from in[1] (v_else) on the
    // rest of this node's universe (= actset_pool_[actset] \ then_lanes; NOT the full-B complement, so
    // a lane terminated at an OUTER select gets zero adjoint here). Route the seed to the owning source.
    case Op::SELECT: {
      const std::vector<uint32_t>& then_l = actset_pool_[n.aux];
      std::vector<char> is_then((size_t)B_, 0); for (uint32_t b : then_l) is_then[b] = 1;
      if (n.out.is_vec) { size_t sz = VSZ(n.out);
        for (uint32_t b : then_l) { const float* g = VA(n.out, b); float* d = VA(in[0], b); for (size_t k = 0; k < sz; ++k) d[k] += g[k]; }
        each([&](uint32_t b){ if (!is_then[b]) { const float* g = VA(n.out, b); float* d = VA(in[1], b); for (size_t k = 0; k < sz; ++k) d[k] += g[k]; } });
      } else {
        for (uint32_t b : then_l) A(in[0], b) += A(n.out, b);
        each([&](uint32_t b){ if (!is_then[b]) A(in[1], b) += A(n.out, b); });
      }
    } break;
  }
}

}  // namespace ndvm
