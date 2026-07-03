// NDVM Strategy-B vector/matrix primitives (float32), batch-native (Phase 3).
// A VecCell's data is B*rows*cols, lane-leading: lane b occupies data[b*sz .. (b+1)*sz), sz=rows*cols,
// row-major within a lane. ndim/rows/cols are scalar (shape is uniform across lanes). Every kernel
// loops lanes; scalar-output ops accumulate a B-wide result via alloc_payload_batch. Inputs are read
// fully into the output buffer before any allocation, so no VecCell/primal pointer dangles across a
// push. det/inv/logdet run an independent partial-pivot LU per lane (bit-identical to the B=1 path).
#include "interp.hpp"
#include <cmath>
#include <vector>

namespace ndvm {

static void list_to_vals(const Interp&, Val lst, std::vector<Val>& out, const std::vector<Pair>* pairs) {
  while (lst.tag == T::PAIR) { out.push_back((*pairs)[lst.aux].car); lst = (*pairs)[lst.aux].cdr; }
}

struct LU { bool ok = true; float det = 0.0f; float logabsdet = 0.0f; std::vector<float> inv; };

static LU lu_invert(const float* A, uint32_t n) {  // A points at one n*n lane
  LU r;
  std::vector<float> a(A, A + (size_t)n * n);
  std::vector<float> inv(n * n, 0.0f);
  for (uint32_t i = 0; i < n; ++i) inv[i * n + i] = 1.0f;
  float det = 1.0f, logabs = 0.0f;
  for (uint32_t col = 0; col < n; ++col) {
    uint32_t piv = col; float best = std::fabs(a[col * n + col]);
    for (uint32_t row = col + 1; row < n; ++row) { float v = std::fabs(a[row * n + col]); if (v > best) { best = v; piv = row; } }
    if (piv != col) { for (uint32_t k = 0; k < n; ++k) { std::swap(a[col * n + k], a[piv * n + k]); std::swap(inv[col * n + k], inv[piv * n + k]); } det = -det; }
    float d = a[col * n + col];
    if (d == 0.0f) { r.ok = false; r.det = 0.0f; r.logabsdet = -INFINITY; r.inv = inv; return r; }
    det *= d; logabs += std::log(std::fabs(d));
    float invd = 1.0f / d;
    for (uint32_t k = 0; k < n; ++k) { a[col * n + k] *= invd; inv[col * n + k] *= invd; }
    for (uint32_t row = 0; row < n; ++row) { if (row == col) continue; float f = a[row * n + col]; if (f == 0.0f) continue;
      for (uint32_t k = 0; k < n; ++k) { a[row * n + k] -= f * a[col * n + k]; inv[row * n + k] -= f * inv[col * n + k]; } }
  }
  r.det = det; r.logabsdet = logabs; r.inv = std::move(inv);
  return r;
}

Val Interp::prim_linalg(const std::string& op, const std::vector<Val>& args, bool& handled) {
  handled = true;
  const uint32_t Bb = B_;
  auto is_scalar = [](const Val& v) { return v.tag == T::INT || v.tag == T::FLOAT; };
  // A structural size arg (eye/zeros/ones) sets a uniform VecCell shape, so it must agree across the
  // active batch (under divergence lane 0 may be inactive -> read the first active lane). A per-lane
  // size is structural divergence (shapes are not B-strided); raise rather than silently use one lane.
  auto uniform_size = [&](const Val& v, const char* nm) -> uint32_t {
    float s = num_lane(v, first_active());
    auto chk = [&](uint32_t b){ if (num_lane(v, b) != s) throw InterpError(std::string(nm) + ": size must be uniform across the active batch"); };
    if (active_full_) for (uint32_t b = 0; b < Bb; ++b) chk(b); else for (uint32_t b : active_lanes_) chk(b);
    return (uint32_t)s;
  };

  // ---- elementwise +/-/*// over vec/mat with scalar broadcast (pure compute; B-wide) ----
  auto ew = [&](const Val& A, const Val& B, char o) -> Val {
    auto ap = [&](float x, float y) { switch (o) { case '+': return x + y; case '-': return x - y; case '*': return x * y; default: return x / y; } };
    if (is_scalar(A) || is_scalar(B)) {
      const Val& sV = is_scalar(A) ? A : B; const Val& vV = is_scalar(A) ? B : A;
      uint8_t nd = vecs_[vV.aux].ndim; uint32_t rr = vecs_[vV.aux].rows, cc = vecs_[vV.aux].cols, sz = rr * cc;
      std::vector<float> d((size_t)Bb * sz);
      for (uint32_t b = 0; b < Bb; ++b) { float s = primal_at(sV.pid)[b]; const float* vd = &vecs_[vV.aux].data[(size_t)b * sz];
        for (uint32_t i = 0; i < sz; ++i) d[(size_t)b * sz + i] = is_scalar(A) ? ap(s, vd[i]) : ap(vd[i], s); }
      return mk_vec(nd, rr, cc, std::move(d));
    }
    uint8_t nd = vecs_[A.aux].ndim; uint32_t rr = vecs_[A.aux].rows, cc = vecs_[A.aux].cols, sz = rr * cc;
    std::vector<float> d((size_t)Bb * sz);
    for (uint32_t b = 0; b < Bb; ++b) { const float* x = &vecs_[A.aux].data[(size_t)b * sz]; const float* y = &vecs_[B.aux].data[(size_t)b * sz];
      for (uint32_t i = 0; i < sz; ++i) d[(size_t)b * sz + i] = ap(x[i], y[i]); }
    return mk_vec(nd, rr, cc, std::move(d));
  };
  if (op == "+" || op == "-" || op == "*" || op == "/") {
    const Op eop = op[0] == '+' ? Op::EW_ADD : op[0] == '-' ? Op::EW_SUB : op[0] == '*' ? Op::EW_MUL : Op::EW_DIV;
    if (op == "-" && args.size() == 1) {
      uint8_t nd = vecs_[args[0].aux].ndim; uint32_t rr = vecs_[args[0].aux].rows, cc = vecs_[args[0].aux].cols;
      std::vector<float> d(vecs_[args[0].aux].data.size());
      for (size_t i = 0; i < d.size(); ++i) d[i] = -vecs_[args[0].aux].data[i];
      Val r = mk_vec(nd, rr, cc, std::move(d)); rec(Op::EW_NEG, r, {args[0]}); return r;
    }
    Val acc = args[0];
    for (size_t i = 1; i < args.size(); ++i) { Val prev = acc; acc = ew(prev, args[i], op[0]); rec(eop, acc, {prev, args[i]}); }
    return acc;
  }

  // ---- constructors ----
  if (op == "vec") {  // (vec (list e...)) -- elements are B-wide scalar payloads -> [B,n]
    std::vector<Val> elems; list_to_vals(*this, args[0], elems, &pairs_);
    uint32_t n = (uint32_t)elems.size();
    std::vector<float> d((size_t)Bb * n);
    for (uint32_t b = 0; b < Bb; ++b) for (uint32_t i = 0; i < n; ++i) d[(size_t)b * n + i] = primal_at(elems[i].pid)[b];
    Val r = mk_vec(1, 1, n, std::move(d)); rec_v(Op::VEC, r, elems); return r;
  }
  if (op == "mat") {  // (mat (list row...)) -- rows are [B,n] vectors -> [B,m,n]
    std::vector<Val> rows; list_to_vals(*this, args[0], rows, &pairs_);
    uint32_t m = (uint32_t)rows.size();
    uint32_t n = m ? vecs_[rows[0].aux].cols : 0;
    for (auto& rv : rows)
      if (rv.tag != T::VEC || vecs_[rv.aux].ndim != 1 || vecs_[rv.aux].cols != n)
        throw InterpError("mat: rows must be vectors of equal length");
    std::vector<float> d((size_t)Bb * m * n);
    for (uint32_t b = 0; b < Bb; ++b) for (uint32_t rIdx = 0; rIdx < m; ++rIdx) {
      const float* rd = &vecs_[rows[rIdx].aux].data[(size_t)b * n];
      for (uint32_t c = 0; c < n; ++c) d[(size_t)b * m * n + rIdx * n + c] = rd[c];
    }
    Val r = mk_vec(2, m, n, std::move(d)); rec_v(Op::MAT, r, rows); return r;
  }
  if (op == "ref") {
    uint8_t nd = vecs_[args[0].aux].ndim; uint32_t rr = vecs_[args[0].aux].rows, cc = vecs_[args[0].aux].cols;
    // The index is structural (it selects which heap data is read), so it must be uniform across the
    // ACTIVE batch. Under divergence lane 0 may be inactive (its payload stale), so read the index from
    // the first active lane and check uniformity over the active lanes only. A per-lane-distinct index
    // is data-dependent indexing -- raise rather than silently gather (per-lane gather is future work).
    uint32_t fa = first_active();
    float fi = num_lane(args[1], fa); if (fi < 0.0f) throw InterpError("ref: negative index");
    auto chk = [&](uint32_t b){ if (num_lane(args[1], b) != fi) throw InterpError("ref: index must be uniform across the active batch"); };
    if (active_full_) for (uint32_t b = 0; b < Bb; ++b) chk(b); else for (uint32_t b : active_lanes_) chk(b);
    uint32_t i = (uint32_t)fi;
    if (nd == 1) { if (i >= cc) throw InterpError("ref: vector index out of range");
      std::vector<float> res(Bb); for (uint32_t b = 0; b < Bb; ++b) res[b] = vecs_[args[0].aux].data[(size_t)b * cc + i];
      Val r{T::FLOAT, 0, alloc_payload_batch(res)}; rec(Op::REF, r, {args[0]}, i); return r; }
    if (i >= rr) throw InterpError("ref: matrix row index out of range");
    std::vector<float> d((size_t)Bb * cc);
    for (uint32_t b = 0; b < Bb; ++b) { const float* row = &vecs_[args[0].aux].data[(size_t)b * rr * cc + (size_t)i * cc];
      for (uint32_t c = 0; c < cc; ++c) d[(size_t)b * cc + c] = row[c]; }
    Val r = mk_vec(1, 1, cc, std::move(d)); rec(Op::REF, r, {args[0]}, i); return r;
  }

  // ---- reductions / products (reduce over the feature dim only; output keeps B) ----
  if (op == "dot") { uint32_t n = vecs_[args[0].aux].cols; std::vector<float> res(Bb);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * n]; const float* c = &vecs_[args[1].aux].data[(size_t)b * n];
      float s = 0.0f; for (uint32_t i = 0; i < n; ++i) s += a[i] * c[i]; res[b] = s; }
    Val r{T::FLOAT, 0, alloc_payload_batch(res)}; rec(Op::DOT, r, {args[0], args[1]}); return r; }
  if (op == "vsum") { uint32_t n = vecs_[args[0].aux].cols; std::vector<float> res(Bb);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * n]; float s = 0.0f; for (uint32_t i = 0; i < n; ++i) s += a[i]; res[b] = s; }
    Val r{T::FLOAT, 0, alloc_payload_batch(res)}; rec(Op::VSUM, r, {args[0]}); return r; }
  if (op == "vlen") return mk_float((float)vecs_[args[0].aux].cols);  // structural, no grad
  if (op == "norm") { uint32_t n = vecs_[args[0].aux].cols; std::vector<float> res(Bb);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * n]; float s = 0.0f; for (uint32_t i = 0; i < n; ++i) s += a[i] * a[i]; res[b] = std::sqrt(s); }
    Val r{T::FLOAT, 0, alloc_payload_batch(res)}; rec(Op::NORM, r, {args[0]}); return r; }
  if (op == "normalize") { uint8_t nd = vecs_[args[0].aux].ndim; uint32_t rr = vecs_[args[0].aux].rows, cc = vecs_[args[0].aux].cols, sz = rr * cc;
    std::vector<float> d((size_t)Bb * sz);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * sz]; float s = 0.0f; for (uint32_t i = 0; i < sz; ++i) s += a[i] * a[i];
      float nrm = std::max(std::sqrt(s), 1e-8f); for (uint32_t i = 0; i < sz; ++i) d[(size_t)b * sz + i] = a[i] / nrm; }
    Val r = mk_vec(nd, rr, cc, std::move(d)); rec(Op::NORMALIZE, r, {args[0]}); return r; }
  if (op == "cross") { if (vecs_[args[0].aux].cols != 3 || vecs_[args[1].aux].cols != 3) throw InterpError("cross: expects 3-vectors");
    std::vector<float> d((size_t)Bb * 3);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * 3]; const float* c = &vecs_[args[1].aux].data[(size_t)b * 3]; float* o = &d[(size_t)b * 3];
      o[0] = a[1] * c[2] - a[2] * c[1]; o[1] = a[2] * c[0] - a[0] * c[2]; o[2] = a[0] * c[1] - a[1] * c[0]; }
    Val r = mk_vec(1, 1, 3, std::move(d)); rec(Op::CROSS, r, {args[0], args[1]}); return r; }
  if (op == "matvec") { uint32_t m = vecs_[args[0].aux].rows, k = vecs_[args[0].aux].cols;
    std::vector<float> d((size_t)Bb * m);
    for (uint32_t b = 0; b < Bb; ++b) { const float* A = &vecs_[args[0].aux].data[(size_t)b * m * k]; const float* v = &vecs_[args[1].aux].data[(size_t)b * k];
      for (uint32_t i = 0; i < m; ++i) { float s = 0.0f; for (uint32_t j = 0; j < k; ++j) s += A[i * k + j] * v[j]; d[(size_t)b * m + i] = s; } }
    Val r = mk_vec(1, 1, m, std::move(d)); rec(Op::MATVEC, r, {args[0], args[1]}); return r; }
  if (op == "matmul") { uint32_t m = vecs_[args[0].aux].rows, k = vecs_[args[0].aux].cols, n = vecs_[args[1].aux].cols;
    std::vector<float> d((size_t)Bb * m * n);
    for (uint32_t b = 0; b < Bb; ++b) { const float* A = &vecs_[args[0].aux].data[(size_t)b * m * k]; const float* B = &vecs_[args[1].aux].data[(size_t)b * k * n]; float* o = &d[(size_t)b * m * n];
      for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) { float s = 0.0f; for (uint32_t p = 0; p < k; ++p) s += A[i * k + p] * B[p * n + j]; o[i * n + j] = s; } }
    Val r = mk_vec(2, m, n, std::move(d)); rec(Op::MATMUL, r, {args[0], args[1]}); return r; }
  if (op == "transpose") { uint32_t m = vecs_[args[0].aux].rows, n = vecs_[args[0].aux].cols;
    std::vector<float> d((size_t)Bb * m * n);
    for (uint32_t b = 0; b < Bb; ++b) { const float* A = &vecs_[args[0].aux].data[(size_t)b * m * n]; float* o = &d[(size_t)b * m * n];
      for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) o[j * m + i] = A[i * n + j]; }
    Val r = mk_vec(2, n, m, std::move(d)); rec(Op::TRANSPOSE, r, {args[0]}); return r; }
  if (op == "trace") { uint32_t m = vecs_[args[0].aux].rows, n = vecs_[args[0].aux].cols, dmin = std::min(m, n); std::vector<float> res(Bb);
    for (uint32_t b = 0; b < Bb; ++b) { const float* A = &vecs_[args[0].aux].data[(size_t)b * m * n]; float s = 0.0f; for (uint32_t i = 0; i < dmin; ++i) s += A[i * n + i]; res[b] = s; }
    Val r{T::FLOAT, 0, alloc_payload_batch(res)}; rec(Op::TRACE, r, {args[0]}); return r; }
  if (op == "outer") { uint32_t na = vecs_[args[0].aux].cols, nb = vecs_[args[1].aux].cols;
    std::vector<float> d((size_t)Bb * na * nb);
    for (uint32_t b = 0; b < Bb; ++b) { const float* a = &vecs_[args[0].aux].data[(size_t)b * na]; const float* c = &vecs_[args[1].aux].data[(size_t)b * nb]; float* o = &d[(size_t)b * na * nb];
      for (uint32_t i = 0; i < na; ++i) for (uint32_t j = 0; j < nb; ++j) o[i * nb + j] = a[i] * c[j]; }
    Val r = mk_vec(2, na, nb, std::move(d)); rec(Op::OUTER, r, {args[0], args[1]}); return r; }

  // ---- linear algebra: independent partial-pivot LU per lane ----
  if (op == "det" || op == "logdet" || op == "inv") {
    uint32_t m = vecs_[args[0].aux].rows, cols = vecs_[args[0].aux].cols;
    if (m != cols) throw InterpError(op + ": requires a square matrix");
    if (op == "inv") { std::vector<float> d((size_t)Bb * m * m);
      for (uint32_t b = 0; b < Bb; ++b) { LU lu = lu_invert(&vecs_[args[0].aux].data[(size_t)b * m * m], m);
        for (uint32_t i = 0; i < m * m; ++i) d[(size_t)b * m * m + i] = lu.inv[i]; }
      Val out = mk_vec(2, m, m, std::move(d)); rec(Op::INV, out, {args[0]}); return out; }
    std::vector<float> res(Bb), invd; if (taping_) invd.resize((size_t)Bb * m * m);
    for (uint32_t b = 0; b < Bb; ++b) { LU lu = lu_invert(&vecs_[args[0].aux].data[(size_t)b * m * m], m);
      res[b] = (op == "det") ? lu.det : lu.logabsdet;
      if (taping_) for (uint32_t i = 0; i < m * m; ++i) invd[(size_t)b * m * m + i] = lu.inv[i]; }
    Val out{T::FLOAT, 0, alloc_payload_batch(res)};
    if (taping_) { Val invc = mk_vec(2, m, m, std::move(invd)); rec(op == "det" ? Op::DET : Op::LOGDET, out, {args[0]}, invc.aux); }
    return out;
  }

  // ---- constant constructors (same matrix for every lane) ----
  if (op == "eye") { uint32_t n = uniform_size(args[0], "eye"); std::vector<float> d((size_t)Bb * n * n, 0.0f);
    for (uint32_t b = 0; b < Bb; ++b) for (uint32_t i = 0; i < n; ++i) d[(size_t)b * n * n + i * n + i] = 1.0f;
    return mk_vec(2, n, n, std::move(d)); }
  if (op == "zeros") { uint32_t n = uniform_size(args[0], "zeros"); return mk_vec(1, 1, n, std::vector<float>((size_t)Bb * n, 0.0f)); }
  if (op == "ones")  { uint32_t n = uniform_size(args[0], "ones"); return mk_vec(1, 1, n, std::vector<float>((size_t)Bb * n, 1.0f)); }
  if (op == "scale") { Val r = ew(args[0], args[1], '*'); rec(Op::SCALE, r, {args[0], args[1]}); return r; }

  handled = false; return nil();
}

}  // namespace ndvm
