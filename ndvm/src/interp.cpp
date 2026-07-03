// NDVM native forward runtime: evaluator core + scalar/list/math primitives.
// Forward semantics mirror bootstrap/compiler.scm under the DMCI float32-native rules captured in
// the Phase-1 semantics spec: all numbers are float32; truthiness is (payload != 0); <= and >= are
// not(>) and not(<); sqrt/log clamp input to >=1e-8; variadic +/* fold from identity left-to-right.
#include "interp.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace ndvm {

Interp::Interp(uint32_t B) : B_(B == 0 ? 1 : B) {
  for (uint32_t b = 0; b < B_; ++b) primal_.push_back(0.0f);  // payload id 0 = canonical 0.0 (B-wide)
  global_env_ = new_frame(NONE);
  actset_pool_.push_back({});                                 // id 0 == ACTSET_FULL sentinel (all lanes)
}

// Intern a sorted active-lane set for the tape. A set covering all B_ lanes collapses to the FULL
// sentinel (so uniform tapes stay byte-identical); otherwise dedup against the pool (a whole bucket of
// nodes recorded under one split shares a single id, so the pool stays small).
uint32_t Interp::intern_actset(const std::vector<uint32_t>& lanes) {
  if (lanes.size() == B_) return ACTSET_FULL;
  for (uint32_t i = 1; i < (uint32_t)actset_pool_.size(); ++i)
    if (actset_pool_[i] == lanes) return i;
  actset_pool_.push_back(lanes);
  return (uint32_t)(actset_pool_.size() - 1);
}

uint32_t Interp::intern(const std::string& s) {
  auto it = sym_ids_.find(s);
  if (it != sym_ids_.end()) return it->second;
  uint32_t id = sym_names_.size();
  sym_names_.push_back(s);
  sym_ids_.emplace(s, id);
  return id;
}

Val Interp::lookup(uint32_t symid, uint32_t env) const {
  uint32_t f = env;
  while (f != NONE) {
    const Frame& fr = frames_[f];
    for (size_t i = fr.binds.size(); i-- > 0;)
      if (fr.binds[i].first == symid) return fr.binds[i].second;
    f = fr.parent;
  }
  // env-lookup fallthrough: unbound -> bare numeric 0 (matches compiler.scm:31).
  return Val{T::FLOAT, 0, 0};  // payload id 0 is always 0.0 (seeded in ctor below)
}

// Variable reference via the node's inline lexical-address cache (Phase 4). On a hit, jump `vhops`
// parents to the binding frame and read slot `vslot`, validated by binds[vslot].first == symid. A miss
// (no cache, walked off the chain, slot out of range, or symbol mismatch) falls back to the scanning
// lookup AND re-caches the address it finds. Semantics are exactly lookup(node.dival, env): the cache is
// only ever a hint validated each use, so a wrong hint can never change the result (it slow-paths). An
// unbound variable returns 0 and is not cached.
Val Interp::lookup_var(const Datum& node, uint32_t env) const {
  const uint32_t symid = static_cast<uint32_t>(node.dival);
  if (use_inline_cache_ && node.vhops >= 0) {
    uint32_t f = env;
    for (int32_t i = 0; i < node.vhops && f != NONE; ++i) f = frames_[f].parent;
    if (f != NONE) {
      const Frame& fr = frames_[f];
      if (static_cast<uint32_t>(node.vslot) < fr.binds.size() && fr.binds[node.vslot].first == symid)
        return fr.binds[node.vslot].second;     // validated hit
    }
    // fall through: stale/invalid cache -> rescan + re-cache below
  }
  uint32_t f = env; int32_t hops = 0;
  while (f != NONE) {
    const Frame& fr = frames_[f];
    for (size_t i = fr.binds.size(); i-- > 0;)
      if (fr.binds[i].first == symid) {
        if (use_inline_cache_) { node.vhops = hops; node.vslot = static_cast<int32_t>(i); }   // re-cache
        return fr.binds[i].second;
      }
    f = fr.parent; ++hops;
  }
  return Val{T::FLOAT, 0, 0};                    // unbound -> 0 (not cached)
}

void Interp::bind_scalar(const std::string& name, float v) {
  Val val = mk_float(v);                          // broadcast v to all B lanes
  frame_define(global_env_, intern(name), val);
  scalar_params_.push_back({name, val.pid});      // leaf payload; read its adjoint after backward
}

void Interp::bind_scalar_batched(const std::string& name, const std::vector<float>& vals) {
  if (vals.size() != B_) throw InterpError("bind_scalar_batched: value count must equal batch width");
  Val val = Val{T::FLOAT, 0, alloc_payload_batch(vals)};   // one distinct value per lane
  frame_define(global_env_, intern(name), val);
  scalar_params_.push_back({name, val.pid});
}

void Interp::bind_matrix(const std::string& name, uint32_t rows, uint32_t cols, std::vector<float> data) {
  // Broadcast the same matrix to all B lanes (a shared input, e.g. obs, across the batch).
  std::vector<float> rep((size_t)B_ * data.size());
  for (uint32_t b = 0; b < B_; ++b) std::copy(data.begin(), data.end(), rep.begin() + (size_t)b * data.size());
  Val mv = mk_vec(2, rows, cols, std::move(rep));
  frame_define(global_env_, intern(name), mv);
}

Interp::Branch Interp::classify(const Val& v) const {
  // DMCI tagged_if over the ACTIVE lane set: a lane is true iff its numeric payload != 0. all active
  // nonzero -> THEN; all active zero -> ELSE; mixed -> MIXED (per-lane divergence). For B=1 and uniform
  // batches the active lanes always agree, so MIXED never fires (the Phase-3 fast path). Structural
  // values (pair/closure/vec/symbol) are truthy and lane-uniform; nil is false. NaN partitions cleanly
  // (NaN!=0 is true -> THEN side; no gap). Reduction is order-independent, so the active-index slow path
  // yields the identical decision to the 0..B_ fast path.
  switch (v.tag) {
    case T::INT: case T::FLOAT: case T::BOOLEAN: {
      const float* p = primal_at(v.pid);
      bool any_t = false, any_f = false;
      auto vote = [&](uint32_t b){ if (p[b] != 0.0f) any_t = true; else any_f = true; };
      if (active_full_) for (uint32_t b = 0; b < B_; ++b) vote(b);
      else for (uint32_t b : active_lanes_) vote(b);
      if (any_t && any_f) return Branch::MIXED;
      return any_t ? Branch::THEN : Branch::ELSE;
    }
    case T::NIL: return Branch::ELSE;
    default: return Branch::THEN;
  }
}

void Interp::split_lanes(const Val& v, std::vector<uint32_t>& then_lanes, std::vector<uint32_t>& else_lanes) const {
  // Partition the active set by the per-lane test (called only on a MIXED numeric/bool test). Both lists
  // come out sorted (we iterate the active set in order) and non-empty (MIXED => at least one each), as
  // intern_actset dedup and the SELECT backward routing assume.
  then_lanes.clear(); else_lanes.clear();
  const float* p = primal_at(v.pid);
  auto cls = [&](uint32_t b){ (p[b] != 0.0f ? then_lanes : else_lanes).push_back(b); };
  if (active_full_) for (uint32_t b = 0; b < B_; ++b) cls(b);
  else for (uint32_t b : active_lanes_) cls(b);
}

Val Interp::select_merge(const std::vector<uint32_t>& then_lanes, const Val& v_then,
                         const std::vector<uint32_t>& else_lanes, const Val& v_else) {
  // Merge two branch results into one Val over the parent active set: then_lanes take v_then's value,
  // else_lanes take v_else's. Defined only for lane-mergeable values. A per-lane-divergent STRUCTURE
  // (different heap cell or shape across lanes) cannot be carried by one scalar-aux Val, so raise --
  // matching the oracle's B>1 structural-divergence error (do not batch structurally different programs).
  auto is_num = [](const Val& v){ return v.tag == T::INT || v.tag == T::FLOAT; };
  auto record = [&](const Val& r){
    if (!taping_) return;
    TNode n; n.op = Op::SELECT; n.out = ref_of(r); n.ins = {ref_of(v_then), ref_of(v_else)};
    n.aux = intern_actset(then_lanes);                                       // then-lanes (subset)
    n.actset = active_full_ ? ACTSET_FULL : intern_actset(active_lanes_);    // universe = parent active set
    tape_.push_back(std::move(n));
  };
  // Scalar-mergeable: same tag (INT/FLOAT/BOOLEAN carried by the B-strided primal), OR two numbers (INT
  // and FLOAT are interchangeable -- both satisfy number? and nothing distinguishes them). A BOOLEAN
  // merges ONLY with a BOOLEAN: a boolean and a number have observably different tags (boolean?/number?),
  // so merging them would silently erase per-lane tag divergence -- that is structural divergence; raise.
  bool same_scalar = (v_then.tag == v_else.tag) &&
                     (v_then.tag == T::INT || v_then.tag == T::FLOAT || v_then.tag == T::BOOLEAN);
  if (same_scalar || (is_num(v_then) && is_num(v_else))) {
    uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);          // alloc first (only growth)
    const float* pt = primal_at(v_then.pid); const float* pe = primal_at(v_else.pid);
    for (uint32_t b : then_lanes) o[b] = pt[b];
    for (uint32_t b : else_lanes) o[b] = pe[b];
    Val r{ v_then.tag == v_else.tag ? v_then.tag : T::FLOAT, 0, pid };       // INT/FLOAT mix -> FLOAT
    record(r); return r;
  }
  if (v_then.tag != v_else.tag)
    throw InterpError("per-lane-divergent structural value unsupported: branches return different value "
                      "kinds across lanes (tag mismatch)");
  if (v_then.tag == T::VEC) {
    const VecCell& vt = vecs_[v_then.aux]; const VecCell& ve = vecs_[v_else.aux];
    if (vt.ndim != ve.ndim || vt.rows != ve.rows || vt.cols != ve.cols)
      throw InterpError("per-lane-divergent structural value unsupported: branches return vectors/"
                        "matrices of different shape across lanes");
    uint32_t sz = vt.rows * vt.cols;
    std::vector<float> d((size_t)B_ * sz);                                   // read both slabs before mk_vec
    for (uint32_t b : then_lanes) std::copy(&vt.data[(size_t)b * sz], &vt.data[(size_t)(b + 1) * sz], &d[(size_t)b * sz]);
    for (uint32_t b : else_lanes) std::copy(&ve.data[(size_t)b * sz], &ve.data[(size_t)(b + 1) * sz], &d[(size_t)b * sz]);
    Val r = mk_vec(vt.ndim, vt.rows, vt.cols, std::move(d));
    record(r); return r;
  }
  // PAIR/CLOSURE/SYMBOL/NIL: a single scalar aux is correct for all lanes only if the branches agree.
  if (v_then.aux != v_else.aux)
    throw InterpError("per-lane-divergent structural value unsupported: branches return different heap "
                      "structure across lanes (aux mismatch)");
  return v_then;
}

Val Interp::eval_cond_tail(const Datum& cond_expr, size_t start, uint32_t en) {
  // (cond (t1 e1) (t2 e2) ... (else en)) lowered to right-nested if over the active set. A MIXED clause
  // test splits the active set: then-lanes take this clause body, else-lanes fall through to the tail
  // (clauses start+1..), then merge. No matching clause for a lane yields #f (matches eval()'s cond).
  for (size_t i = start; i < cond_expr.size(); ++i) {
    const Datum& clause = cond_expr.list[i];
    if (clause.list[0].is_sym("else")) return eval(clause.list[1], en);
    Val tv = eval(clause.list[0], en);
    Branch br = classify(tv);
    if (br == Branch::THEN) return eval(clause.list[1], en);
    if (br == Branch::MIXED) {
      std::vector<uint32_t> then_lanes, else_lanes;
      split_lanes(tv, then_lanes, else_lanes);
      Val v_then, v_else;
      { ActiveGuard g(this); g.set(then_lanes); v_then = eval(clause.list[1], en); }
      { ActiveGuard g(this); g.set(else_lanes); v_else = eval_cond_tail(cond_expr, i + 1, en); }
      return select_merge(then_lanes, v_then, else_lanes, v_else);
    }
    // ELSE: this clause is false for all active lanes; fall through to the next clause.
  }
  return boolean(false);
}

// ---- atom -> value (numbers self-evaluate; symbols are variable references) ----
static bool parse_long(const std::string& s, long& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  errno = 0;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end != s.c_str() + s.size() || errno != 0) return false;
  out = v; return true;
}
static bool parse_double(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (end != s.c_str() + s.size()) return false;
  out = v; return true;
}

Val Interp::atom_value(const std::string& a, uint32_t env) {
  if (a == "#t") return boolean(true);
  if (a == "#f") return boolean(false);
  long iv;
  if (parse_long(a, iv)) return mk_int(static_cast<float>(iv));
  double dv;
  if (parse_double(a, dv)) return mk_float(static_cast<float>(dv));
  return lookup(intern(a), env);  // symbol -> variable reference
}

// quote: materialize a datum into runtime data (numbers/bools/symbols/pairs/nil).
static Val materialize_impl(Interp& I, const Datum& d);
static Val materialize_list(Interp& I, const std::vector<Datum>& xs, size_t i) {
  if (i >= xs.size()) return I.nil();
  return I.cons(materialize_impl(I, xs[i]), materialize_list(I, xs, i + 1));
}
static Val materialize_impl(Interp& I, const Datum& d) {
  if (d.is_atom) {
    if (d.atom == "#t") return I.boolean(true);
    if (d.atom == "#f") return I.boolean(false);
    long iv; double dv;
    if (parse_long(d.atom, iv)) return I.mk_int(static_cast<float>(iv));
    if (parse_double(d.atom, dv)) return I.mk_float(static_cast<float>(dv));
    return I.symbol(I.intern(d.atom));
  }
  if (d.list.empty()) return I.nil();
  return materialize_list(I, d.list, 0);
}

// ---- primitive name groups (compiler.scm dispatch order) ----
static bool group_A(const std::string& n) {
  // Operators the DMCI interpreter (bootstrap/compiler.scm eval-apply) dispatches by name. Note
  // `and`/`or` are intentionally absent: compiler.scm does not implement them, so a program using
  // them is rejected/falls through on the oracle; NDVM must not be more permissive.
  static const char* names[] = {"+","-","*","/","=","<",">","<=",">=","not",
    "cons","car","cdr","null?","pair?","number?","boolean?","symbol?","eq?","list",
    "sin","cos","exp","sqrt","log","abs","pow","min","max","modulo","remainder", nullptr};
  for (int i = 0; names[i]; ++i) if (n == names[i]) return true;
  return false;
}
static bool group_B(const std::string& n) {
  static const char* names[] = {"vec","mat","ref","dot","cross","norm","normalize","vsum","vlen",
    "scale","matvec","matmul","transpose","trace","det","logdet","inv","outer","eye","zeros","ones", nullptr};
  for (int i = 0; names[i]; ++i) if (n == names[i]) return true;
  return false;
}

Val Interp::apply_primitive(const std::string& op, const std::vector<Val>& args, bool& handled) {
  handled = true;
  Val r = prim_arith(op, args, handled); if (handled) return r;
  r = prim_compare(op, args, handled);  if (handled) return r;
  r = prim_math(op, args, handled);     if (handled) return r;
  r = prim_list(op, args, handled);     if (handled) return r;
  r = prim_linalg(op, args, handled);   if (handled) return r;
  handled = false; return nil();
}

// ---------------------------------------------------------------------------
// Phase 4: decoded-form cache. The interpreter dispatches purely syntactically (special forms and
// group_A primitives are matched by NAME before any binding is consulted), so each AST node's decode is
// context-independent. decode() classifies a node ONCE; eval then dispatches on the cached kind, skipping
// the repeated atom re-parse / symbol re-intern / special-form string chain / primitive name scan. Pure
// speed: results are byte-identical. The win scales with how often a node is re-evaluated in one walk
// (e.g. an 80-step loop body decodes once, reuses 79x).
namespace {
enum DKind : uint8_t { DK_NONE = 0, DK_LIT_INT, DK_LIT_FLOAT, DK_LIT_TRUE, DK_LIT_FALSE,
                       DK_VAR, DK_NIL, DK_SF, DK_PRIM_A, DK_APP };
enum SForm : int32_t { SF_QUOTE = 0, SF_IF, SF_BEGIN, SF_LET, SF_LETREC, SF_LAMBDA, SF_COND, SF_DEFINE };
static int sf_opcode(const std::string& s) {
  if (s == "if") return SF_IF;           if (s == "let") return SF_LET;
  if (s == "lambda") return SF_LAMBDA;   if (s == "cond") return SF_COND;
  if (s == "begin") return SF_BEGIN;     if (s == "letrec") return SF_LETREC;
  if (s == "define") return SF_DEFINE;   if (s == "quote") return SF_QUOTE;
  return -1;
}
}  // namespace

void Interp::decode(const Datum& d) {
  // Mirror the syntactic classification atom_value + the eval special-form/primitive dispatch perform,
  // but record it so it is done once per node. d.* cache fields are mutable (eval holds a const ref).
  if (d.is_atom) {
    const std::string& a = d.atom;
    if (a == "#t") { d.dkind = DK_LIT_TRUE; return; }
    if (a == "#f") { d.dkind = DK_LIT_FALSE; return; }
    long iv;   if (parse_long(a, iv))   { d.dkind = DK_LIT_INT;   d.dfval = static_cast<float>(iv); return; }
    double dv; if (parse_double(a, dv)) { d.dkind = DK_LIT_FLOAT; d.dfval = static_cast<float>(dv); return; }
    d.dkind = DK_VAR; d.dival = static_cast<int32_t>(intern(a)); return;   // variable: cache the symbol id
  }
  if (d.list.empty()) { d.dkind = DK_NIL; return; }
  const Datum& head = d.list[0];
  if (head.is_atom) {
    int op = sf_opcode(head.atom);
    if (op >= 0) { d.dkind = DK_SF; d.dival = op; return; }     // special form (always wins over binding)
    if (group_A(head.atom)) { d.dkind = DK_PRIM_A; return; }    // scalar/list/math primitive by name
  }
  d.dkind = DK_APP;   // general application: eval head (closure) or group_B primitive, decided at runtime
}

// ===========================================================================
// eval: trampolined, with proper tail-call optimization. Tail positions
// (if/cond branches, let/letrec/begin body, closure application) loop in place.
// Dispatch is on the node's decoded-form cache (filled lazily by decode()).
// ===========================================================================
Val Interp::eval(const Datum& expr, uint32_t env) {
  const Datum* e = &expr;
  uint32_t en = env;
  for (;;) {
    if (++eval_steps_ > max_eval_steps_)
      throw InterpError("eval step budget exceeded: a batched lane may not terminate (data-dependent "
                        "control flow that never reaches its base case on some lane)");
    if (e->dkind == DK_NONE) decode(*e);
    switch (e->dkind) {
      case DK_LIT_INT:   return mk_int(e->dfval);
      case DK_LIT_FLOAT: return mk_float(e->dfval);
      case DK_LIT_TRUE:  return boolean(true);
      case DK_LIT_FALSE: return boolean(false);
      case DK_VAR:       return lookup_var(*e, en);   // inline lexical-address cache (Phase 4)
      case DK_NIL:       return nil();

      case DK_SF:
        switch (static_cast<SForm>(e->dival)) {
          case SF_QUOTE: return materialize_impl(*this, e->list[1]);
          case SF_IF: {
            Val t = eval(e->list[1], en);
            Branch br = classify(t);
            if (br == Branch::THEN) { e = &e->list[2]; continue; }  // TCO into the taken branch (fast path)
            if (br == Branch::ELSE) { e = &e->list[3]; continue; }
            // MIXED: per-lane divergence (Phase 3b). Split the active set, evaluate each branch under its
            // lane subset (ActiveGuard restores the parent set on return OR on a thrown deeper error),
            // then merge per lane. This breaks TCO -- correct: the merge needs both results.
            std::vector<uint32_t> then_lanes, else_lanes;
            split_lanes(t, then_lanes, else_lanes);
            Val v_then, v_else;
            { ActiveGuard g(this); g.set(then_lanes); v_then = eval(e->list[2], en); }
            { ActiveGuard g(this); g.set(else_lanes); v_else = eval(e->list[3], en); }
            return select_merge(then_lanes, v_then, else_lanes, v_else);
          }
          case SF_BEGIN: {
            for (size_t i = 1; i + 1 < e->size(); ++i) eval(e->list[i], en);
            e = &e->list[e->size() - 1];
            continue;
          }
          case SF_LET: {
            uint32_t nf = new_frame(en);
            const std::vector<Datum>& binds = e->list[1].list;
            for (const Datum& b : binds) {
              Val v = eval(b.list[1], nf);          // sequential: sees prior binds in nf + parent
              frame_define(nf, intern(b.list[0].atom), v);
            }
            en = nf; e = &e->list[2];
            continue;
          }
          case SF_LETREC: {
            uint32_t nf = new_frame(en);
            const std::vector<Datum>& binds = e->list[1].list;
            for (const Datum& b : binds) {
              const Datum& rhs = b.list[1];
              if (!rhs.is_atom && !rhs.list.empty() && rhs.list[0].is_sym("lambda")) {
                std::vector<uint32_t> params;
                for (const Datum& p : rhs.list[1].list) params.push_back(intern(p.atom));
                frame_define(nf, intern(b.list[0].atom), make_closure(params, &rhs.list[2], nf));
              } else {
                frame_define(nf, intern(b.list[0].atom), eval(rhs, nf));
              }
            }
            en = nf; e = &e->list[2];
            continue;
          }
          case SF_LAMBDA: {
            std::vector<uint32_t> params;
            for (const Datum& p : e->list[1].list) params.push_back(intern(p.atom));
            return make_closure(params, &e->list[2], en);
          }
          case SF_COND: {
            bool advanced = false;
            for (size_t i = 1; i < e->size() && !advanced; ++i) {
              const Datum& clause = e->list[i];
              if (clause.list[0].is_sym("else")) { e = &clause.list[1]; advanced = true; break; }
              Val tv = eval(clause.list[0], en);
              Branch br = classify(tv);
              if (br == Branch::THEN) { e = &clause.list[1]; advanced = true; }  // uniform: trampoline (TCO)
              else if (br == Branch::MIXED) {                                    // per-lane divergence
                std::vector<uint32_t> then_lanes, else_lanes;
                split_lanes(tv, then_lanes, else_lanes);
                Val v_then, v_else;
                { ActiveGuard g(this); g.set(then_lanes); v_then = eval(clause.list[1], en); }
                { ActiveGuard g(this); g.set(else_lanes); v_else = eval_cond_tail(*e, i + 1, en); }
                return select_merge(then_lanes, v_then, else_lanes, v_else);
              }
              // ELSE: this clause is false for all active lanes; fall through to the next clause.
            }
            if (advanced) continue;
            return boolean(false);
          }
          case SF_DEFINE: {
            const Datum& tgt = e->list[1];
            if (tgt.is_list()) {  // (define (name params...) body) -> closure
              std::vector<uint32_t> params;
              for (size_t i = 1; i < tgt.list.size(); ++i) params.push_back(intern(tgt.list[i].atom));
              frame_define(en, intern(tgt.list[0].atom), make_closure(params, &e->list[2], en));
            } else {              // (define name value)
              frame_define(en, intern(tgt.atom), eval(e->list[2], en));
            }
            return nil();
          }
        }
        break;  // unreachable: every SForm case returns or continues

      case DK_PRIM_A:
      case DK_APP: {
        const Datum& head = e->list[0];
        const size_t ai = args_top_++;                 // claim a pooled arg vector (stack discipline)
        if (ai >= args_pool_.size()) args_pool_.emplace_back();
        args_pool_[ai].clear();                         // keeps the vector's capacity (no heap alloc)
        // Fill via a temp first: eval() may recurse and grow args_pool_, so re-index args_pool_[ai] only
        // AFTER each eval completes. An inline args_pool_[ai].push_back(eval(...)) would, under C++17's
        // "object expr sequenced before argument" rule, hold a stale reference across the realloc -- the
        // Phase-2 eval-order trap.
        for (size_t i = 1; i < e->size(); ++i) { Val v = eval(e->list[i], en); args_pool_[ai].push_back(v); }
        // 1) group_A scalar/list/math primitive dispatched by NAME (before closures). DK_APP skips this.
        if (e->dkind == DK_PRIM_A) {
          bool h = false;
          Val r = apply_primitive(head.atom, args_pool_[ai], h);   // prims never grow args_pool_
          if (h) { --args_top_; return r; }
        }
        // 2) evaluate operator; if a closure, tail-apply
        Val func = eval(head, en);                      // may grow args_pool_; index args_pool_[ai] after
        if (func.tag == T::CLOSURE) {
          const Closure& c = closures_[func.aux];
          uint32_t nf = new_frame(c.env);               // grows frames_, not args_pool_
          std::vector<Val>& av = args_pool_[ai];
          for (size_t i = 0; i < c.params.size() && i < av.size(); ++i)
            frame_define(nf, c.params[i], av[i]);
          --args_top_;
          en = nf; e = c.body;
          continue;  // TCO
        }
        // 3) vector/matrix op by name (after closures)
        if (head.is_atom && group_B(head.atom)) {
          bool h = false;
          Val r = apply_primitive(head.atom, args_pool_[ai], h);
          if (h) { --args_top_; return r; }
        }
        // 4) fallthrough: bare numeric 0 (matches compiler.scm:282)
        --args_top_;
        return mk_float(0.0f);
      }
    }
  }
}

// ===========================================================================
// Program entry: parse, macro-expand, collect top-level defines, eval last form.
// ===========================================================================
// Parse + macro-expand into program_hold_, but ONLY if the source differs from what is already loaded.
// Re-running the same program (the co-search inner loop) reuses the parsed AST and its warm decoded-form
// cache, so parse + expand + decode are paid once per program, not per eval. This is memoized parsing of
// runtime data; the object program is still S-expressions and the evaluator is unchanged.
void Interp::load(const std::string& src) {
  if (src == loaded_src_ && !program_hold_.empty()) return;   // cache hit: keep warm program + decode
  std::vector<Datum> forms = parse_top_level(src);
  reset_gensym();                        // per-program gensym sequence (schedule-independent under threads)
  program_hold_.clear();
  program_hold_.reserve(forms.size());
  for (const Datum& f : forms) program_hold_.push_back(expand_macros(f));
  loaded_src_ = src;
}

// Collect ALL top-level defines into the global env (function-define -> closure capturing the global env,
// so mutual + self recursion resolve), then evaluate the LAST form. Mirrors compiler.scm
// scheme-eval-program (collect-defines + build-defined-env + eval last-form). Assumes load() has run.
Val Interp::run_loaded() {
  tape_.clear();                         // fresh tape per forward (repeated forward+backward is safe)
  size_t last = program_hold_.size() - 1;
  for (size_t i = 0; i < program_hold_.size(); ++i) {
    const Datum& f = program_hold_[i];
    if (f.is_atom || f.list.empty() || !f.list[0].is_sym("define")) continue;
    const Datum& tgt = f.list[1];
    if (tgt.is_list()) {  // (define (name params...) body)
      std::vector<uint32_t> params;
      for (size_t j = 1; j < tgt.list.size(); ++j) params.push_back(intern(tgt.list[j].atom));
      frame_define(global_env_, intern(tgt.list[0].atom), make_closure(params, &f.list[2], global_env_));
    } else {              // (define name value)
      frame_define(global_env_, intern(tgt.atom), eval(f.list[2], global_env_));
    }
  }
  return eval(program_hold_[last], global_env_);
}

Val Interp::run(const std::string& program_src) { load(program_src); return run_loaded(); }

// Reset the per-forward state (numeric arena, environment, heap, tape, params, active set) to the ctor's
// initial state, while KEEPING the parsed program (program_hold_/loaded_src_), the decoded-form cache on
// those Datums, and the symbol table (so cached DK_VAR symbol ids stay valid). This is what makes Interp
// reuse across forwards correct AND fast: begin_forward() -> re-bind params -> run() reuses everything.
void Interp::reset_state() {
  primal_.assign(B_, 0.0f);              // payload id 0 = canonical 0.0 (B-wide), as in the ctor
  adj_.clear(); vadj_.clear();
  pairs_.clear(); closures_.clear(); vecs_.clear();
  frame_top_ = 0; global_env_ = new_frame(NONE);   // reuse the frame pool (keeps per-frame binds capacity)
  tape_.clear(); scalar_params_.clear();
  actset_pool_.clear(); actset_pool_.push_back({});   // id 0 = ACTSET_FULL sentinel
  active_full_ = true; active_lanes_.clear();
  args_top_ = 0;                                       // reuse the args pool (keeps each vector's capacity)
  taping_ = false;                                     // ctor-equivalent: the caller re-sets taping per forward
  payload_allocs_ = 0; eval_steps_ = 0;
}

// ---------------------------------------------------------------------------
// Scalar arithmetic (vec/mat operands delegate to prim_linalg elementwise paths).
// ---------------------------------------------------------------------------
Val Interp::prim_arith(const std::string& op, const std::vector<Val>& args, bool& handled) {
  handled = true;
  const bool any_vec = [&]{ for (auto& a : args) if (a.tag == T::VEC) return true; return false; }();
  if (any_vec) {  // elementwise on vectors/matrices -> handled in prim_linalg
    return prim_linalg(op, args, handled);
  }
  // Per-lane scalar fold (B-wide). Allocate the output payload once (the only primal_ growth here),
  // then fill all B lanes; operand pointers stay valid because no further growth occurs.
  auto ret = [&](Op eop, uint32_t pid){ Val r{T::FLOAT, 0, pid}; rec_v(eop, r, args); return r; };
  if (op == "+") { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) { float acc = 0.0f; for (auto& a : args) acc += primal_at(a.pid)[b]; o[b] = acc; }
    return ret(Op::ADD, pid); }
  if (op == "*") { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) { float acc = 1.0f; for (auto& a : args) acc *= primal_at(a.pid)[b]; o[b] = acc; }
    return ret(Op::MUL, pid); }
  if (op == "-") { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) { float acc;
      if (args.size() == 1) acc = -primal_at(args[0].pid)[b];
      else { acc = primal_at(args[0].pid)[b]; for (size_t i = 1; i < args.size(); ++i) acc -= primal_at(args[i].pid)[b]; }
      o[b] = acc; }
    return ret(Op::SUB, pid); }
  if (op == "/") { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) { float acc;
      if (args.size() == 1) acc = 1.0f / primal_at(args[0].pid)[b];
      else { acc = primal_at(args[0].pid)[b]; for (size_t i = 1; i < args.size(); ++i) acc /= primal_at(args[i].pid)[b]; }
      o[b] = acc; }
    return ret(Op::DIV, pid); }
  handled = false; return nil();
}

Val Interp::prim_compare(const std::string& op, const std::vector<Val>& args, bool& handled) {
  // Only read args inside a confirmed branch (op name + arity), so unary ops routed through the
  // apply_primitive chain never dereference a missing second argument.
  // Per-lane truth into a B-wide BOOLEAN payload; truthy() reduces it at the branch (BatchError parity).
  handled = true;
  if (op == "=" || op == "<" || op == ">" || op == "<=" || op == ">=") {
    if (args.size() < 2) throw InterpError("comparison '" + op + "' needs 2 arguments");
    uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) { float a = primal_at(args[0].pid)[b], c = primal_at(args[1].pid)[b];
      bool t = (op == "=") ? (a == c) : (op == "<") ? (a < c) : (op == ">") ? (a > c)
             : (op == "<=") ? !(a > c) : !(a < c);   // <= is not(>), >= is not(<) (NaN-correct)
      o[b] = t ? 1.0f : 0.0f; }
    return Val{T::BOOLEAN, 0, pid};
  }
  if (op == "not") { if (args.empty()) throw InterpError("'not' needs 1 argument");
    uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    for (uint32_t b = 0; b < B_; ++b) o[b] = (primal_at(args[0].pid)[b] == 0.0f) ? 1.0f : 0.0f;
    return Val{T::BOOLEAN, 0, pid};
  }
  handled = false; return nil();  // `and`/`or` are NOT oracle primitives (see group_A)
}

Val Interp::prim_math(const std::string& op, const std::vector<Val>& args, bool& handled) {
  // Per-lane (B-wide). un1/un2 alloc the output once then fill all lanes; clamp/semantics per lane
  // match the oracle (sqrt/log clamp input to 1e-8 per lane). Op decided once, outside the lane loop.
  handled = true;
  auto un1 = [&](Op eop, auto f) { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    const float* x = primal_at(args[0].pid); for (uint32_t b = 0; b < B_; ++b) o[b] = f(x[b]);
    Val r{T::FLOAT, 0, pid}; rec(eop, r, {args[0]}); return r; };
  auto un2 = [&](Op eop, auto f) { uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
    const float* x = primal_at(args[0].pid); const float* y = primal_at(args[1].pid);
    for (uint32_t b = 0; b < B_; ++b) o[b] = f(x[b], y[b]);
    Val r{T::FLOAT, 0, pid}; rec(eop, r, {args[0], args[1]}); return r; };
  if (op == "sin") return un1(Op::SIN, [](float v) { return std::sin(v); });
  if (op == "cos") return un1(Op::COS, [](float v) { return std::cos(v); });
  if (op == "exp") return un1(Op::EXP, [](float v) { return std::exp(v); });
  if (op == "sqrt") return un1(Op::SQRT, [](float v) { return std::sqrt(std::max(v, 1e-8f)); });
  if (op == "log")  return un1(Op::LOG,  [](float v) { return std::log(std::max(v, 1e-8f)); });
  if (op == "abs") return un1(Op::ABS, [](float v) { return std::fabs(v); });
  if (op == "pow") return un2(Op::POW, [](float a, float c) { return std::pow(a, c); });
  if (op == "min") return un2(Op::MINB, [](float a, float c) { return std::min(a, c); });
  if (op == "max") return un2(Op::MAXB, [](float a, float c) { return std::max(a, c); });
  if (op == "modulo")    return un2(Op::MOD, [](float a, float c) { return std::fmod(a, c); });
  if (op == "remainder") return un2(Op::REM, [](float a, float c) { return a - c * std::floor(a / c); });
  handled = false; return nil();
}

Val Interp::prim_list(const std::string& op, const std::vector<Val>& args, bool& handled) {
  handled = true;
  if (op == "cons") return cons(args[0], args[1]);
  if (op == "car")  return pairs_[args[0].aux].car;
  if (op == "cdr")  return pairs_[args[0].aux].cdr;
  if (op == "null?") return boolean(args[0].tag == T::NIL);
  if (op == "pair?") return boolean(args[0].tag == T::PAIR);
  if (op == "number?") return boolean(args[0].tag == T::INT || args[0].tag == T::FLOAT);
  if (op == "boolean?") return boolean(args[0].tag == T::BOOLEAN);
  if (op == "symbol?") return boolean(args[0].tag == T::SYMBOL);
  if (op == "eq?") {
    const Val &a = args[0], &b = args[1];
    if (a.tag != b.tag) return boolean(false);
    switch (a.tag) {
      case T::NIL: return boolean(true);
      case T::SYMBOL: return boolean(a.aux == b.aux);   // symbols are scalar/uniform across lanes
      case T::PAIR: return boolean(a.aux == b.aux);
      case T::BOOLEAN: case T::INT: case T::FLOAT: {     // per-lane numeric equality -> B-wide truth
        uint32_t pid = alloc_payload(0.0f); float* o = primal_at(pid);
        const float* pa = primal_at(a.pid); const float* pb = primal_at(b.pid);
        for (uint32_t k = 0; k < B_; ++k) o[k] = (pa[k] == pb[k]) ? 1.0f : 0.0f;
        return Val{T::BOOLEAN, 0, pid};
      }
      default: return boolean(false);
    }
  }
  if (op == "list") {  // build a nil-terminated cons chain
    Val r = nil();
    for (size_t i = args.size(); i-- > 0;) r = cons(args[i], r);
    return r;
  }
  handled = false; return nil();
}

}  // namespace ndvm
