// boxed_run: a native C++ FORWARD interpreter for the DMCI core subset that keeps the BOXED tagged-value
// representation, i.e. WITHOUT NDVM's structural/numeric split. Every interpreter value (number, boolean,
// symbol, pair, closure, vector/matrix) is a separately allocated tagged Box, the native analog of the
// eager DMCI backend's per-value [14] tagged tensor, but with no Python and no torch. It reuses NDVM's
// exact reader and macro expander (ndvm::parse_top_level + ndvm::expand_macros), so it walks the identical
// program, and it mirrors NDVM's float32 forward semantics (truthiness payload!=0; <= is not(>); sqrt/log
// clamp to 1e-8; unbound/fallthrough -> 0; partial-pivot LU for det/inv). It is a tree-walking evaluator
// (no decoded-form or inline-lexical caches), so a measured NDVM-over-boxed gap reflects the structural/
// numeric split together with NDVM's structural-walk optimizations, not the representation in isolation.
//
// Purpose: isolate, in native code, how much of NDVM's speedup over the eager backend survives once the
// host language is no longer Python. The eager backend, this boxed C++ interpreter, the tuned-eager Python
// encoding, and NDVM form a 2x2 over {boxed, split} x {Python, native}.
//
// CLI mirrors ndvm_run: boxed_run <program_file> [bindings_file]; BOXED_BENCH=<N> times N fresh forwards.
#include "sexpr.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using ndvm::Datum;

namespace boxed {

enum class Tag : uint8_t { NIL, BOOL, INT, FLOAT, SYM, PAIR, CLOSURE, VEC };

struct Env;
// One boxed tagged value. Unused members stay empty; numeric/structural values alike are heap-boxed
// (allocated from the arena), which is exactly the per-value representation the split removes.
struct Box {
  Tag tag = Tag::NIL;
  float num = 0.0f;                 // BOOL/INT/FLOAT scalar payload
  uint32_t sym = 0;                 // SYM id
  uint8_t ndim = 1;                 // VEC: 1 = vector, 2 = matrix
  uint32_t rows = 1, cols = 0;      // VEC shape
  std::vector<float> data;          // VEC payload (row-major)
  Box* car = nullptr; Box* cdr = nullptr;             // PAIR
  std::vector<uint32_t> params; const Datum* body = nullptr; Env* env = nullptr;  // CLOSURE
};

struct Env { std::vector<std::pair<uint32_t, Box*>> binds; Env* parent = nullptr; };

struct BoxError : std::runtime_error { using std::runtime_error::runtime_error; };

static bool parse_long(const std::string& s, long& out) {
  if (s.empty()) return false;
  char* end = nullptr; errno = 0; long v = std::strtol(s.c_str(), &end, 10);
  if (end != s.c_str() + s.size() || errno != 0) return false;
  out = v; return true;
}
static bool parse_double(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr; double v = std::strtod(s.c_str(), &end);
  if (end != s.c_str() + s.size()) return false;
  out = v; return true;
}

// Partial-pivot LU (det + inverse), bit-for-bit the same routine NDVM uses (interp_linalg.cpp).
struct LU { bool ok = true; float det = 0.0f; std::vector<float> inv; };
static LU lu_invert(const float* A, uint32_t n) {
  LU r; std::vector<float> a(A, A + (size_t)n * n); std::vector<float> inv(n * n, 0.0f);
  for (uint32_t i = 0; i < n; ++i) inv[i * n + i] = 1.0f;
  float det = 1.0f;
  for (uint32_t col = 0; col < n; ++col) {
    uint32_t piv = col; float best = std::fabs(a[col * n + col]);
    for (uint32_t row = col + 1; row < n; ++row) { float v = std::fabs(a[row * n + col]); if (v > best) { best = v; piv = row; } }
    if (piv != col) { for (uint32_t k = 0; k < n; ++k) { std::swap(a[col * n + k], a[piv * n + k]); std::swap(inv[col * n + k], inv[piv * n + k]); } det = -det; }
    float d = a[col * n + col];
    if (d == 0.0f) { r.ok = false; r.det = 0.0f; r.inv = inv; return r; }
    det *= d; float invd = 1.0f / d;
    for (uint32_t k = 0; k < n; ++k) { a[col * n + k] *= invd; inv[col * n + k] *= invd; }
    for (uint32_t row = 0; row < n; ++row) { if (row == col) continue; float f = a[row * n + col]; if (f == 0.0f) continue;
      for (uint32_t k = 0; k < n; ++k) { a[row * n + k] -= f * a[col * n + k]; inv[row * n + k] -= f * inv[col * n + k]; } }
  }
  r.det = det; r.inv = std::move(inv); return r;
}

class Boxed {
 public:
  Boxed() { global_ = new_env(nullptr); }

  uint32_t intern(const std::string& s) {
    auto it = sym_ids_.find(s); if (it != sym_ids_.end()) return it->second;
    uint32_t id = (uint32_t)sym_names_.size(); sym_names_.push_back(s); sym_ids_.emplace(s, id); return id;
  }

  void bind_scalar(const std::string& name, float v) { def(global_, intern(name), mkfloat(v)); }
  void bind_matrix(const std::string& name, uint32_t rows, uint32_t cols, std::vector<float> d) {
    def(global_, intern(name), mkvec(2, rows, cols, std::move(d)));
  }

  Box* run(const std::string& src) {
    std::vector<Datum> forms = ndvm::parse_top_level(src);
    ndvm::reset_gensym();
    prog_.clear(); for (const Datum& f : forms) prog_.push_back(ndvm::expand_macros(f));
    // collect top-level defines, then eval the last form (mirrors NDVM run_loaded)
    size_t last = prog_.size() - 1;
    for (size_t i = 0; i < prog_.size(); ++i) {
      const Datum& f = prog_[i];
      if (f.is_atom || f.list.empty() || !f.list[0].is_sym("define")) continue;
      const Datum& tgt = f.list[1];
      if (tgt.is_list()) { std::vector<uint32_t> ps; for (size_t j = 1; j < tgt.list.size(); ++j) ps.push_back(intern(tgt.list[j].atom));
        def(global_, intern(tgt.list[0].atom), mkclosure(std::move(ps), &f.list[2], global_)); }
      else def(global_, intern(tgt.atom), eval(f.list[2], global_));
    }
    return eval(prog_[last], global_);
  }

  // diagnostics
  size_t box_allocs() const { return arena_.size(); }
  bool is_num(const Box* v) const { return v->tag == Tag::INT || v->tag == Tag::FLOAT || v->tag == Tag::BOOL; }

 private:
  std::deque<Box> arena_;            // arena: stable addresses, chunked allocation (same discipline as NDVM)
  std::deque<Env> envs_;
  std::unordered_map<std::string, uint32_t> sym_ids_;
  std::vector<std::string> sym_names_;
  std::vector<Datum> prog_;
  Env* global_ = nullptr;

  Box* alloc() { arena_.emplace_back(); return &arena_.back(); }   // EVERY value is a boxed object
  Env* new_env(Env* parent) { envs_.emplace_back(); Env* e = &envs_.back(); e->parent = parent; return e; }
  void def(Env* e, uint32_t sym, Box* v) { e->binds.push_back({sym, v}); }

  Box* mknil() { Box* b = alloc(); b->tag = Tag::NIL; return b; }
  Box* mkbool(bool v) { Box* b = alloc(); b->tag = Tag::BOOL; b->num = v ? 1.0f : 0.0f; return b; }
  Box* mkint(float v) { Box* b = alloc(); b->tag = Tag::INT; b->num = v; return b; }
  Box* mkfloat(float v) { Box* b = alloc(); b->tag = Tag::FLOAT; b->num = v; return b; }
  Box* mksym(uint32_t id) { Box* b = alloc(); b->tag = Tag::SYM; b->sym = id; return b; }
  Box* mkpair(Box* a, Box* d) { Box* b = alloc(); b->tag = Tag::PAIR; b->car = a; b->cdr = d; return b; }
  Box* mkvec(uint8_t nd, uint32_t r, uint32_t c, std::vector<float> d) {
    Box* b = alloc(); b->tag = Tag::VEC; b->ndim = nd; b->rows = r; b->cols = c; b->data = std::move(d); return b; }
  Box* mkclosure(std::vector<uint32_t> ps, const Datum* body, Env* env) {
    Box* b = alloc(); b->tag = Tag::CLOSURE; b->params = std::move(ps); b->body = body; b->env = env; return b; }

  Box* lookup(uint32_t sym, Env* env) {
    for (Env* e = env; e; e = e->parent)
      for (size_t i = e->binds.size(); i-- > 0;) if (e->binds[i].first == sym) return e->binds[i].second;
    return mkfloat(0.0f);   // unbound -> 0 (matches compiler.scm)
  }

  Box* atom_value(const Datum& d, Env* env) {
    const std::string& a = d.atom;
    if (a == "#t") return mkbool(true);
    if (a == "#f") return mkbool(false);
    long iv; if (parse_long(a, iv)) return mkint((float)iv);
    double dv; if (parse_double(a, dv)) return mkfloat((float)dv);
    return lookup(intern(a), env);
  }

  static bool group_A(const std::string& n) {
    static const char* names[] = {"+","-","*","/","=","<",">","<=",">=","not",
      "cons","car","cdr","null?","pair?","number?","boolean?","symbol?","eq?","list",
      "sin","cos","exp","sqrt","log","abs","pow","min","max","modulo","remainder", nullptr};
    for (int i = 0; names[i]; ++i) if (n == names[i]) return true; return false;
  }
  static bool group_B(const std::string& n) {
    static const char* names[] = {"vec","mat","ref","dot","cross","norm","normalize","vsum","vlen",
      "scale","matvec","matmul","transpose","trace","det","logdet","inv","outer","eye","zeros","ones", nullptr};
    for (int i = 0; names[i]; ++i) if (n == names[i]) return true; return false;
  }

  void list_to_vals(Box* lst, std::vector<Box*>& out) { while (lst->tag == Tag::PAIR) { out.push_back(lst->car); lst = lst->cdr; } }

  // truthiness: a numeric/bool box is true iff payload != 0; nil is false; other structure is true.
  bool truthy(Box* v) {
    switch (v->tag) { case Tag::INT: case Tag::FLOAT: case Tag::BOOL: return v->num != 0.0f;
      case Tag::NIL: return false; default: return true; }
  }

  Box* eval(const Datum& expr, Env* env) {
    const Datum* e = &expr; Env* en = env;
    for (;;) {
      if (e->is_atom) return atom_value(*e, en);
      if (e->list.empty()) return mknil();
      const Datum& head = e->list[0];
      if (head.is_atom) {
        const std::string& h = head.atom;
        if (h == "quote") return materialize(e->list[1]);
        if (h == "if") { Box* t = eval(e->list[1], en); e = truthy(t) ? &e->list[2] : &e->list[3]; continue; }
        if (h == "begin") { for (size_t i = 1; i + 1 < e->size(); ++i) eval(e->list[i], en); e = &e->list[e->size() - 1]; continue; }
        if (h == "let") { Env* nf = new_env(en);
          for (const Datum& b : e->list[1].list) { Box* v = eval(b.list[1], nf); def(nf, intern(b.list[0].atom), v); }
          en = nf; e = &e->list[2]; continue; }
        if (h == "letrec") { Env* nf = new_env(en);
          for (const Datum& b : e->list[1].list) { const Datum& rhs = b.list[1];
            if (!rhs.is_atom && !rhs.list.empty() && rhs.list[0].is_sym("lambda")) { std::vector<uint32_t> ps;
              for (const Datum& p : rhs.list[1].list) ps.push_back(intern(p.atom)); def(nf, intern(b.list[0].atom), mkclosure(std::move(ps), &rhs.list[2], nf)); }
            else def(nf, intern(b.list[0].atom), eval(rhs, nf)); }
          en = nf; e = &e->list[2]; continue; }
        if (h == "lambda") { std::vector<uint32_t> ps; for (const Datum& p : e->list[1].list) ps.push_back(intern(p.atom));
          return mkclosure(std::move(ps), &e->list[2], en); }
        if (h == "cond") { bool adv = false;
          for (size_t i = 1; i < e->size() && !adv; ++i) { const Datum& cl = e->list[i];
            if (cl.list[0].is_sym("else")) { e = &cl.list[1]; adv = true; break; }
            if (truthy(eval(cl.list[0], en))) { e = &cl.list[1]; adv = true; } }
          if (adv) continue; return mkbool(false); }
        if (h == "define") { const Datum& tgt = e->list[1];
          if (tgt.is_list()) { std::vector<uint32_t> ps; for (size_t i = 1; i < tgt.list.size(); ++i) ps.push_back(intern(tgt.list[i].atom));
            def(en, intern(tgt.list[0].atom), mkclosure(std::move(ps), &e->list[2], en)); }
          else def(en, intern(tgt.atom), eval(e->list[2], en));
          return mknil(); }
      }
      // application
      std::vector<Box*> args; args.reserve(e->size() - 1);
      for (size_t i = 1; i < e->size(); ++i) args.push_back(eval(e->list[i], en));
      if (head.is_atom && group_A(head.atom)) { bool h = false; Box* r = prim_A(head.atom, args, h); if (h) return r; }
      Box* func = eval(head, en);
      if (func->tag == Tag::CLOSURE) { Env* nf = new_env(func->env);
        for (size_t i = 0; i < func->params.size() && i < args.size(); ++i) def(nf, func->params[i], args[i]);
        en = nf; e = func->body; continue; }
      if (head.is_atom && group_B(head.atom)) { bool h = false; Box* r = prim_B(head.atom, args, h); if (h) return r; }
      return mkfloat(0.0f);   // fallthrough
    }
  }

  Box* materialize(const Datum& d) {
    if (d.is_atom) { if (d.atom == "#t") return mkbool(true); if (d.atom == "#f") return mkbool(false);
      long iv; double dv; if (parse_long(d.atom, iv)) return mkint((float)iv); if (parse_double(d.atom, dv)) return mkfloat((float)dv);
      return mksym(intern(d.atom)); }
    if (d.list.empty()) return mknil();
    Box* r = mknil(); for (size_t i = d.list.size(); i-- > 0;) r = mkpair(materialize(d.list[i]), r); return r;
  }

  // group_A: scalar arithmetic / comparison / math / list. Delegates vec/mat arithmetic to prim_B.
  Box* prim_A(const std::string& op, std::vector<Box*>& a, bool& h) {
    h = true;
    bool any_vec = false; for (Box* x : a) if (x->tag == Tag::VEC) any_vec = true;
    if ((op == "+" || op == "-" || op == "*" || op == "/") && any_vec) return prim_B(op, a, h);
    if (op == "+") { float s = 0.0f; for (Box* x : a) s += x->num; return mkfloat(s); }
    if (op == "*") { float s = 1.0f; for (Box* x : a) s *= x->num; return mkfloat(s); }
    if (op == "-") { if (a.size() == 1) return mkfloat(-a[0]->num); float s = a[0]->num; for (size_t i = 1; i < a.size(); ++i) s -= a[i]->num; return mkfloat(s); }
    if (op == "/") { if (a.size() == 1) return mkfloat(1.0f / a[0]->num); float s = a[0]->num; for (size_t i = 1; i < a.size(); ++i) s /= a[i]->num; return mkfloat(s); }
    if (op == "=" || op == "<" || op == ">" || op == "<=" || op == ">=") { float x = a[0]->num, y = a[1]->num;
      bool t = op == "=" ? x == y : op == "<" ? x < y : op == ">" ? x > y : op == "<=" ? !(x > y) : !(x < y); return mkbool(t); }
    if (op == "not") return mkbool(a[0]->num == 0.0f);
    if (op == "sin") return mkfloat(std::sin(a[0]->num));
    if (op == "cos") return mkfloat(std::cos(a[0]->num));
    if (op == "exp") return mkfloat(std::exp(a[0]->num));
    if (op == "sqrt") return mkfloat(std::sqrt(std::max(a[0]->num, 1e-8f)));
    if (op == "log") return mkfloat(std::log(std::max(a[0]->num, 1e-8f)));
    if (op == "abs") return mkfloat(std::fabs(a[0]->num));
    if (op == "pow") return mkfloat(std::pow(a[0]->num, a[1]->num));
    if (op == "min") return mkfloat(std::min(a[0]->num, a[1]->num));
    if (op == "max") return mkfloat(std::max(a[0]->num, a[1]->num));
    if (op == "modulo") return mkfloat(std::fmod(a[0]->num, a[1]->num));
    if (op == "remainder") { float x = a[0]->num, y = a[1]->num; return mkfloat(x - y * std::floor(x / y)); }
    if (op == "cons") return mkpair(a[0], a[1]);
    if (op == "car") return a[0]->car;
    if (op == "cdr") return a[0]->cdr;
    if (op == "null?") return mkbool(a[0]->tag == Tag::NIL);
    if (op == "pair?") return mkbool(a[0]->tag == Tag::PAIR);
    if (op == "number?") return mkbool(a[0]->tag == Tag::INT || a[0]->tag == Tag::FLOAT);
    if (op == "boolean?") return mkbool(a[0]->tag == Tag::BOOL);
    if (op == "symbol?") return mkbool(a[0]->tag == Tag::SYM);
    if (op == "eq?") { Box* x = a[0]; Box* y = a[1]; if (x->tag != y->tag) return mkbool(false);
      switch (x->tag) { case Tag::NIL: return mkbool(true); case Tag::SYM: return mkbool(x->sym == y->sym);
        case Tag::PAIR: case Tag::CLOSURE: case Tag::VEC: return mkbool(x == y);
        default: return mkbool(x->num == y->num); } }
    if (op == "list") { Box* r = mknil(); for (size_t i = a.size(); i-- > 0;) r = mkpair(a[i], r); return r; }
    h = false; return nullptr;
  }

  // group_B: vector/matrix primitives (B=1).
  Box* prim_B(const std::string& op, std::vector<Box*>& a, bool& h) {
    h = true;
    auto is_scalar = [](Box* v) { return v->tag == Tag::INT || v->tag == Tag::FLOAT; };
    auto ap = [](char o, float x, float y) { switch (o) { case '+': return x + y; case '-': return x - y; case '*': return x * y; default: return x / y; } };
    auto ew = [&](Box* A, Box* B, char o) -> Box* {
      if (is_scalar(A) || is_scalar(B)) { Box* s = is_scalar(A) ? A : B; Box* v = is_scalar(A) ? B : A;
        std::vector<float> d(v->data.size()); for (size_t i = 0; i < d.size(); ++i) d[i] = is_scalar(A) ? ap(o, s->num, v->data[i]) : ap(o, v->data[i], s->num);
        return mkvec(v->ndim, v->rows, v->cols, std::move(d)); }
      std::vector<float> d(A->data.size()); for (size_t i = 0; i < d.size(); ++i) d[i] = ap(o, A->data[i], B->data[i]);
      return mkvec(A->ndim, A->rows, A->cols, std::move(d));
    };
    if (op == "+" || op == "-" || op == "*" || op == "/") {
      if (op == "-" && a.size() == 1) { std::vector<float> d(a[0]->data.size()); for (size_t i = 0; i < d.size(); ++i) d[i] = -a[0]->data[i];
        return mkvec(a[0]->ndim, a[0]->rows, a[0]->cols, std::move(d)); }
      Box* acc = a[0]; for (size_t i = 1; i < a.size(); ++i) acc = ew(acc, a[i], op[0]); return acc;
    }
    if (op == "vec") { std::vector<Box*> el; list_to_vals(a[0], el); uint32_t n = (uint32_t)el.size();
      std::vector<float> d(n); for (uint32_t i = 0; i < n; ++i) d[i] = el[i]->num; return mkvec(1, 1, n, std::move(d)); }
    if (op == "mat") { std::vector<Box*> rows; list_to_vals(a[0], rows); uint32_t m = (uint32_t)rows.size();
      uint32_t n = m ? rows[0]->cols : 0; std::vector<float> d((size_t)m * n);
      for (uint32_t r = 0; r < m; ++r) for (uint32_t c = 0; c < n; ++c) d[r * n + c] = rows[r]->data[c]; return mkvec(2, m, n, std::move(d)); }
    if (op == "ref") { Box* M = a[0]; uint32_t i = (uint32_t)a[1]->num;
      if (M->ndim == 1) return mkfloat(M->data[i]);
      std::vector<float> d(M->cols); for (uint32_t c = 0; c < M->cols; ++c) d[c] = M->data[(size_t)i * M->cols + c]; return mkvec(1, 1, M->cols, std::move(d)); }
    if (op == "dot") { uint32_t n = a[0]->cols; float s = 0.0f; for (uint32_t i = 0; i < n; ++i) s += a[0]->data[i] * a[1]->data[i]; return mkfloat(s); }
    if (op == "vsum") { float s = 0.0f; for (float x : a[0]->data) s += x; return mkfloat(s); }
    if (op == "vlen") return mkfloat((float)a[0]->cols);
    if (op == "norm") { float s = 0.0f; for (float x : a[0]->data) s += x * x; return mkfloat(std::sqrt(s)); }
    if (op == "matvec") { uint32_t m = a[0]->rows, k = a[0]->cols; std::vector<float> d(m);
      for (uint32_t i = 0; i < m; ++i) { float s = 0.0f; for (uint32_t j = 0; j < k; ++j) s += a[0]->data[i * k + j] * a[1]->data[j]; d[i] = s; } return mkvec(1, 1, m, std::move(d)); }
    if (op == "matmul") { uint32_t m = a[0]->rows, k = a[0]->cols, n = a[1]->cols; std::vector<float> d((size_t)m * n);
      for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) { float s = 0.0f; for (uint32_t p = 0; p < k; ++p) s += a[0]->data[i * k + p] * a[1]->data[p * n + j]; d[i * n + j] = s; } return mkvec(2, m, n, std::move(d)); }
    if (op == "transpose") { uint32_t m = a[0]->rows, n = a[0]->cols; std::vector<float> d((size_t)m * n);
      for (uint32_t i = 0; i < m; ++i) for (uint32_t j = 0; j < n; ++j) d[j * m + i] = a[0]->data[i * n + j]; return mkvec(2, n, m, std::move(d)); }
    if (op == "trace") { uint32_t m = a[0]->rows, n = a[0]->cols, dm = std::min(m, n); float s = 0.0f; for (uint32_t i = 0; i < dm; ++i) s += a[0]->data[i * n + i]; return mkfloat(s); }
    if (op == "outer") { uint32_t na = a[0]->cols, nb = a[1]->cols; std::vector<float> d((size_t)na * nb);
      for (uint32_t i = 0; i < na; ++i) for (uint32_t j = 0; j < nb; ++j) d[i * nb + j] = a[0]->data[i] * a[1]->data[j]; return mkvec(2, na, nb, std::move(d)); }
    if (op == "det" || op == "logdet" || op == "inv") { uint32_t m = a[0]->rows;
      if (op == "inv") { LU lu = lu_invert(a[0]->data.data(), m); return mkvec(2, m, m, std::move(lu.inv)); }
      LU lu = lu_invert(a[0]->data.data(), m);
      if (op == "det") return mkfloat(lu.det);
      return mkfloat(lu.ok ? std::log(std::fabs(lu.det)) : -INFINITY); }
    if (op == "eye") { uint32_t n = (uint32_t)a[0]->num; std::vector<float> d((size_t)n * n, 0.0f); for (uint32_t i = 0; i < n; ++i) d[i * n + i] = 1.0f; return mkvec(2, n, n, std::move(d)); }
    if (op == "zeros") { uint32_t n = (uint32_t)a[0]->num; return mkvec(1, 1, n, std::vector<float>(n, 0.0f)); }
    if (op == "ones") { uint32_t n = (uint32_t)a[0]->num; return mkvec(1, 1, n, std::vector<float>(n, 1.0f)); }
    if (op == "scale") return ew(a[0], a[1], '*');
    h = false; return nullptr;
  }
};

}  // namespace boxed

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }

struct ScalarBind { std::string name; float val; };
struct MatrixBind { std::string name; uint32_t rows, cols; std::vector<float> data; };

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: boxed_run <program_file> [bindings_file]\n"); return 2; }
  std::vector<ScalarBind> sc; std::vector<MatrixBind> mt;
  if (argc >= 3) {
    std::ifstream bf(argv[2]); std::string line;
    while (std::getline(bf, line)) {
      std::istringstream ls(line); std::string kind; ls >> kind;
      if (kind == "scalar") { std::string nm; double v; ls >> nm >> v; sc.push_back({nm, (float)v}); }
      else if (kind == "matrix") { std::string nm; uint32_t rr, cc; ls >> nm >> rr >> cc;
        std::vector<float> d; double x; while (ls >> x) d.push_back((float)x); mt.push_back({nm, rr, cc, std::move(d)}); }
    }
  }
  std::string src = slurp(argv[1]);
  try {
    if (const char* nb = std::getenv("BOXED_BENCH")) {
      long n = std::strtol(nb, nullptr, 10); if (n < 1) n = 1;
      auto t0 = std::chrono::steady_clock::now();
      for (long it = 0; it < n; ++it) { boxed::Boxed I;
        for (auto& s : sc) I.bind_scalar(s.name, s.val);
        for (auto& m : mt) I.bind_matrix(m.name, m.rows, m.cols, m.data);
        volatile float sink = I.is_num(I.run(src)) ? 1.0f : 0.0f; (void)sink; }
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / (double)n;
      std::printf("bench %.6f ms/eval over %ld runs\n", ms, n);
      return 0;
    }
    boxed::Boxed I;
    for (auto& s : sc) I.bind_scalar(s.name, s.val);
    for (auto& m : mt) I.bind_matrix(m.name, m.rows, m.cols, m.data);
    boxed::Box* r = I.run(src);
    if (I.is_num(r)) std::printf("result %.9g\n", (double)r->num);
    else if (r->tag == boxed::Tag::VEC) { std::printf("vec %u %u %u", (unsigned)r->ndim, r->rows, r->cols); for (float x : r->data) std::printf(" %.9g", (double)x); std::printf("\n"); }
    else std::printf("result <non-numeric tag=%d>\n", (int)r->tag);
    std::fprintf(stderr, "diag box_allocs=%zu\n", I.box_allocs());
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
