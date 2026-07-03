// NDVM native forward runtime (Phase 1): a native Scheme evaluator for the DMCI core subset, with
// the structural/numeric split. Discrete structure (tags, heap addresses, interned symbols,
// closures) is scalar native data; numbers live in a dense payload table (primal only in Phase 1;
// the adjoint buffer + reverse-mode tape arrive in Phase 2). Object programs are runtime data: the
// evaluator is fixed and walks any program. Lazy if + proper tail-call optimization are built in.
//
// Forward semantics mirror bootstrap/compiler.scm (the metacircular evaluator) so that
// native-eval(macro-expand(PROG)) equals the DMCI oracle's evaluate(compile_dmci(PROG)) output.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include "sexpr.hpp"

namespace ndvm {

enum class T : uint8_t { NIL, BOOLEAN, INT, FLOAT, SYMBOL, PAIR, CLOSURE, VEC };

constexpr uint32_t NONE = 0xFFFFFFFFu;

// Scalar structural value: discrete identity in {tag, aux}; numeric values index the payload table.
struct Val {
  T tag = T::NIL;
  uint32_t aux = 0;       // BOOL: 0/1; SYMBOL: id; PAIR/CLOSURE/VEC: heap addr
  uint32_t pid = NONE;    // index into the primal payload table (INT/FLOAT only)
};

struct Pair { Val car, cdr; };
struct Closure { std::vector<uint32_t> params; const Datum* body; uint32_t env; };
// Dense vector/matrix payload (Strategy B), row-major, float32. ndim==1 -> vector of length `cols`
// (rows==1); ndim==2 -> matrix [rows, cols]. Mirrors DMCI VECTOR feature_ndim 1 vs 2.
struct VecCell { uint8_t ndim = 1; uint32_t rows = 1, cols = 0; std::vector<float> data; };

// Environment frame: small assoc of interned-symbol -> value, with a parent link.
struct Frame { std::vector<std::pair<uint32_t, Val>> binds; uint32_t parent = NONE; };

struct InterpError : std::runtime_error { using std::runtime_error::runtime_error; };

// ---- Phase 2: native reverse-mode tape -------------------------------------
// A tape node records one differentiable numeric op (structural ops emit nothing). Operands are
// referenced by payload identity: a scalar by its payload id, a vector/matrix by its VecCell id.
// Primals persist in the arena, so backward reads them by id; adjoints accumulate into parallel
// buffers (adj_ for scalars, vadj_ per VecCell). Variadic +/-/*// record all inputs in one node.
enum class Op : uint8_t {
  ADD, SUB, MUL, DIV,                                  // scalar (variadic)
  SIN, COS, EXP, SQRT, LOG, POW, ABS, MINB, MAXB, MOD, REM,
  EW_ADD, EW_SUB, EW_MUL, EW_DIV, EW_NEG, SCALE,       // elementwise vec/mat + scalar broadcast
  DOT, VSUM, NORM, NORMALIZE, CROSS,
  MATMUL, MATVEC, TRANSPOSE, TRACE, OUTER,
  DET, LOGDET, INV,
  VEC, MAT, REF,
  SELECT,                                              // Phase 3b: per-lane merge at a divergent branch
};

// Active-lane set id (Phase 3b). 0 is the FULL sentinel: all B_ lanes active (the uniform fast path).
constexpr uint32_t ACTSET_FULL = 0;

struct Ref { bool is_vec; uint32_t id; };              // scalar payload id, or VecCell id
// aux: DET/LOGDET cache the inverse-slab VecCell id; REF caches the gather index; SELECT caches the
// interned then_lanes id. actset: the active-lane set in effect when the node was recorded (FULL =>
// backward replays 0..B_, byte-identical to Phase 3; a reduced set gates the VJP to those lanes only).
struct TNode { Op op; Ref out; std::vector<Ref> ins; uint32_t aux = NONE; uint32_t actset = ACTSET_FULL; };

class Interp {
 public:
  // B is the batch width (lanes). Phase 3: numeric payloads carry B lanes; the structural walk runs
  // once and is shared by all lanes. B==1 is byte-identical to the Phase-2 scalar runtime.
  explicit Interp(uint32_t B = 1);
  uint32_t batch() const { return B_; }
  void set_max_eval_steps(uint64_t n) { max_eval_steps_ = n; }      // cap to halt a non-terminating lane
  void set_inline_cache(bool on) { use_inline_cache_ = on; }        // Phase 4: toggle the var-lookup cache (ablation)

  // Symbol interning.
  uint32_t intern(const std::string& s);
  const std::string& sym_name(uint32_t id) const { return sym_names_[id]; }

  // Value constructors. Numeric/boolean payloads are B-wide (broadcast a scalar to all lanes).
  Val nil() const { return Val{T::NIL, 0, NONE}; }
  Val boolean(bool b) { return Val{T::BOOLEAN, 0, alloc_payload(b ? 1.0f : 0.0f)}; }   // B-wide truth
  Val mk_int(float v) { return Val{T::INT, 0, alloc_payload(v)}; }
  Val mk_float(float v) { return Val{T::FLOAT, 0, alloc_payload(v)}; }
  Val symbol(uint32_t id) const { return Val{T::SYMBOL, id, NONE}; }
  float num(const Val& v) const { return primal_[(size_t)v.pid * B_]; }   // lane 0 (structural/size reads)
  float num_lane(const Val& v, uint32_t b) const { return primal_[(size_t)v.pid * B_ + b]; }
  bool is_num(const Val& v) const { return v.tag == T::INT || v.tag == T::FLOAT; }

  // Heap.
  Val cons(Val a, Val d) { uint32_t a2 = pairs_.size(); pairs_.push_back({a, d}); return Val{T::PAIR, a2, NONE}; }
  Val mk_vec(uint8_t ndim, uint32_t rows, uint32_t cols, std::vector<float> data) {
    uint32_t a = vecs_.size(); vecs_.push_back(VecCell{ndim, rows, cols, std::move(data)}); return Val{T::VEC, a, NONE}; }
  const VecCell& vec(const Val& v) const { return vecs_[v.aux]; }

  // Binding inputs (free variables of the program).
  void bind_scalar(const std::string& name, float v);            // broadcast one value to all B lanes
  void bind_scalar_batched(const std::string& name, const std::vector<float>& vals);  // one value per lane (len B)
  void bind_matrix(const std::string& name, uint32_t rows, uint32_t cols, std::vector<float> data);

  // Parse + macro-expand + evaluate one program form; returns its value (use num() to read a scalar).
  // run() caches the parsed+expanded program by source: re-running the SAME src skips parse/expand and
  // keeps the warm decoded-form cache, so only the forward eval re-runs.
  Val run(const std::string& program_src);
  Val eval(const Datum& expr, uint32_t env);

  // Phase 4 cross-call reuse: reset the per-forward state (arena, env, heap, tape) for a FRESH forward
  // while KEEPING the parsed program, decoded-form cache, and symbol table. Call this before re-binding
  // parameters and re-running the same Interp -- the boundary caches one Interp per program and reuses
  // it across the co-search inner loop, so parse + macro-expand + decode are paid once, not per eval.
  void begin_forward() { reset_state(); }

  // Diagnostics for the cost model / Phase-2 readiness.
  uint64_t payload_allocs() const { return payload_allocs_; }
  uint64_t heap_pairs() const { return pairs_.size(); }
  uint64_t eval_steps() const { return eval_steps_; }
  uint64_t tape_len() const { return tape_.size(); }

  // ---- Phase 2: reverse-mode AD ----
  void set_taping(bool on) { taping_ = on; }                       // enable BEFORE the forward run
  void backward(const Val& output);                                // seed d(output)=1 on all B lanes, replay
  float grad_lane(uint32_t pid, uint32_t b) const {
    size_t i = (size_t)pid * B_ + b; return i < adj_.size() ? adj_[i] : 0.0f; }
  float grad_scalar(uint32_t pid) const { return grad_lane(pid, 0); }
  const std::vector<std::pair<std::string, uint32_t>>& scalar_params() const { return scalar_params_; }

 private:
  // B-strided payload arena: payload pid owns primal_[pid*B_ .. +B_]. alloc broadcasts one value.
  uint32_t alloc_payload(float v) { uint32_t id = (uint32_t)(primal_.size() / B_); ++payload_allocs_;
    for (uint32_t b = 0; b < B_; ++b) primal_.push_back(v); return id; }
  uint32_t alloc_payload_batch(const std::vector<float>& vals) {  // one value per lane (len B_)
    uint32_t id = (uint32_t)(primal_.size() / B_); ++payload_allocs_;
    for (float x : vals) primal_.push_back(x); return id; }
  float* primal_at(uint32_t pid) { return &primal_[(size_t)pid * B_]; }
  const float* primal_at(uint32_t pid) const { return &primal_[(size_t)pid * B_]; }
  float* adj_at(uint32_t pid) { return &adj_[(size_t)pid * B_]; }
  // Phase 4 frame pool: reuse Frame objects (and their binds-vector capacity) across forwards. frame_top_
  // is the live frame count; frames_ keeps the high-water Frames so their binds allocations persist
  // (reset_state lowers frame_top_ to 0 without freeing). A reused frame clears its binds (keeps capacity).
  uint32_t new_frame(uint32_t parent) {
    if (frame_top_ < frames_.size()) { Frame& fr = frames_[frame_top_]; fr.binds.clear(); fr.parent = parent; return frame_top_++; }
    frames_.push_back(Frame{{}, parent}); return frame_top_++;
  }
  void frame_define(uint32_t f, uint32_t symid, Val v) { frames_[f].binds.push_back({symid, v}); }
  Val lookup(uint32_t symid, uint32_t env) const;
  Val lookup_var(const Datum& node, uint32_t env) const;   // Phase 4: lookup via the node's inline cache

  Val atom_value(const std::string& a, uint32_t env);
  void decode(const Datum& d);           // Phase 4: classify d once into its decoded-form cache (d.dkind)
  void load(const std::string& src);     // Phase 4: parse + macro-expand into program_hold_ (cached by src)
  Val run_loaded();                      // Phase 4: collect top-level defines + eval the last loaded form
  void reset_state();                    // Phase 4: clear per-forward state; keep program + decode + symbols
  Val apply_closure_tail(const Closure& c, const std::vector<Val>& args, const Datum*& expr, uint32_t& env);
  Val apply_primitive(const std::string& op, const std::vector<Val>& args, bool& handled);

  // primitive groups (interp_prims.cpp)
  Val prim_arith(const std::string& op, const std::vector<Val>& args, bool& handled);
  Val prim_compare(const std::string& op, const std::vector<Val>& args, bool& handled);
  Val prim_math(const std::string& op, const std::vector<Val>& args, bool& handled);
  Val prim_list(const std::string& op, const std::vector<Val>& args, bool& handled);
  Val prim_linalg(const std::string& op, const std::vector<Val>& args, bool& handled);

  // Branch decision over the ACTIVE lane set (Phase 3b). DMCI tagged_if: a lane is true iff its numeric
  // payload != 0. all active lanes true -> THEN; all false -> ELSE; mixed -> MIXED (per-lane divergence).
  // B=1 and uniform batches are always all-active-agree, so they never reach MIXED (Phase-3 fast path).
  enum class Branch : uint8_t { THEN, ELSE, MIXED };
  Branch classify(const Val& v) const;
  void split_lanes(const Val& v, std::vector<uint32_t>& then_lanes, std::vector<uint32_t>& else_lanes) const;
  // Merge two branch results per lane into one full-active Val (records Op::SELECT for scalars/VecCells;
  // raises on per-lane-divergent structure -- tag/shape/aux mismatch). then_lanes u else_lanes == active.
  Val select_merge(const std::vector<uint32_t>& then_lanes, const Val& v_then,
                   const std::vector<uint32_t>& else_lanes, const Val& v_else);
  // Evaluate a cond's clauses [start..] over the active set, returning a Val (the nested-if tail used by
  // the divergent cond path; per-lane test laziness preserved -- clause k+1's test is only evaluated
  // over the lanes that fell through clause k). The uniform cond path stays trampolined in eval().
  Val eval_cond_tail(const Datum& cond_expr, size_t start, uint32_t env);

  // Tape recording (no-op unless taping_); one node per differentiable numeric op.
  Ref ref_of(const Val& v) const { return v.tag == T::VEC ? Ref{true, v.aux} : Ref{false, v.pid}; }
  void rec(Op op, const Val& out, std::initializer_list<Val> ins, uint32_t aux = NONE);
  void rec_v(Op op, const Val& out, const std::vector<Val>& ins, uint32_t aux = NONE);
  void dispatch_adjoint(const TNode& n);

  // interned-symbol ids for hot special-form / primitive names (filled in ctor)
  std::unordered_map<std::string, uint32_t> sym_ids_;
  std::vector<std::string> sym_names_;

  uint32_t B_ = 1;                       // batch width (lanes); payloads are B_-strided
  std::vector<float> primal_;            // dense numeric payload table, B_-strided per payload
  std::vector<Pair> pairs_;
  std::vector<Closure> closures_;
  std::vector<VecCell> vecs_;
  std::vector<Frame> frames_;
  uint32_t frame_top_ = 0;               // live frame count (frames_ retains capacity as a pool)
  uint32_t global_env_ = NONE;
  // Phase 4 args pool: a stack of reusable arg vectors so an application does not heap-allocate its args
  // each call. args_top_ is the live depth; args_pool_ keeps the vectors (with capacity) as a free list.
  std::vector<std::vector<Val>> args_pool_;
  size_t args_top_ = 0;

  // ---- Phase 3b: active-lane set (dynamic scope) ----------------------------
  // active_full_ == true => all B_ lanes active (the uniform fast path; active_lanes_ ignored). A
  // divergent if/cond narrows the active set around its branch recursion via ActiveGuard. classify(),
  // split_lanes(), select_merge(), ref's index check, and rec() are the only consumers; the per-lane
  // numeric kernels keep looping 0..B_ (inactive-lane outputs are discarded by select forward and never
  // read by a gated backward), so forward stays byte-identical to Phase 3.
  bool active_full_ = true;
  std::vector<uint32_t> active_lanes_;                  // sorted active lane indices; valid iff !active_full_
  std::vector<std::vector<uint32_t>> actset_pool_;      // pooled lane-index lists for the tape; [0] = {} = FULL
  uint32_t intern_actset(const std::vector<uint32_t>& lanes);   // returns ACTSET_FULL when lanes covers all B_
  void set_active(std::vector<uint32_t> lanes) {        // active_full_ <=> the set is all B_ lanes
    active_full_ = (lanes.size() == B_); active_lanes_ = std::move(lanes); }
  uint32_t first_active() const { return active_full_ ? 0u : active_lanes_.front(); }  // representative lane
  // RAII save/restore of the active set: branch recursion sets it through this, so a thrown
  // structural-divergence (or any deeper) InterpError restores the engine's active set on unwind.
  struct ActiveGuard {
    Interp* I; bool saved_full; std::vector<uint32_t> saved_lanes;
    explicit ActiveGuard(Interp* i) : I(i), saved_full(i->active_full_), saved_lanes(i->active_lanes_) {}
    void set(std::vector<uint32_t> lanes) { I->set_active(std::move(lanes)); }
    ~ActiveGuard() { I->active_full_ = saved_full; I->active_lanes_ = std::move(saved_lanes); }
    ActiveGuard(const ActiveGuard&) = delete; ActiveGuard& operator=(const ActiveGuard&) = delete;
  };

  uint64_t payload_allocs_ = 0;
  uint64_t eval_steps_ = 0;
  // Phase 3b: batched divergence trampolines until the LAST lane terminates, so a single non-terminating
  // lane would hang the whole batch. Cap total eval steps (generously -- real programs are far under) so
  // a runaway lane raises rather than hangs. NDVM has no loop node, so this is the global analogue of
  // the oracle's per-loop max_iter (engine.py).
  uint64_t max_eval_steps_ = 200000000;
  bool use_inline_cache_ = true;         // Phase 4: read the per-node var-lookup cache (off = always scan)

  // Phase-2 tape + adjoint buffers (allocated lazily; adjoints sized at backward()).
  bool taping_ = false;
  std::vector<TNode> tape_;
  std::vector<float> adj_;                                       // scalar adjoints, parallel to primal_
  std::vector<std::vector<float>> vadj_;                         // per-VecCell adjoints, parallel to vecs_
  std::vector<std::pair<std::string, uint32_t>> scalar_params_;  // (name, payload id) for bound scalars

  std::vector<Datum> program_hold_;      // keeps parsed program ASTs alive (closures hold Datum*)
  std::string loaded_src_;               // source last parsed into program_hold_ (Phase 4 parse cache key)
 public:
  Val make_closure(std::vector<uint32_t> params, const Datum* body, uint32_t env) {
    uint32_t a = closures_.size(); closures_.push_back(Closure{std::move(params), body, env}); return Val{T::CLOSURE, a, NONE}; }
  const Closure& closure(const Val& v) const { return closures_[v.aux]; }
};

}  // namespace ndvm
