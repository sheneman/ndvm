// NDVM front-end: S-expression datum, reader, and macro expander.
// Mirrors neural_compiler/parser/scheme_parser.py (tokenize/_parse_sexpr) and
// neural_compiler/dmci.py (expand_macros + _expand_loop/_rewrite_recur/_expand_let_star/_begin_wrap),
// so the native runtime sees exactly the CORE forms the DMCI evaluator (bootstrap/compiler.scm) sees.
// A Datum is the parsed program AST as nested data: either an atom (token string) or a list.
#pragma once
#include <string>
#include <vector>
#include <stdexcept>

namespace ndvm {

struct Datum {
  bool is_atom = true;
  std::string atom;            // valid iff is_atom
  std::vector<Datum> list;     // valid iff !is_atom

  // Phase 4 decoded-form cache: memoized syntactic classification of this node (DKind, defined in
  // interp.cpp). The interpreter's dispatch is purely syntactic -- special forms and primitives are
  // matched by name BEFORE any variable binding -- so a node's decode is context-independent and stable
  // for the AST's lifetime; it is filled lazily on first eval and reused thereafter. mutable because
  // eval holds the AST by const ref. dkind 0 means "not decoded yet". Single-threaded (Phase 4); a
  // Phase-5 multicore walk must make these writes atomic or thread-local.
  mutable uint8_t dkind = 0;   // DKind
  mutable int32_t dival = 0;   // DK_VAR: interned symbol id; DK_SF: special-form opcode
  mutable float dfval = 0.0f;  // DK_LIT_INT / DK_LIT_FLOAT: the parsed numeric value
  // Phase 4 inline variable-lookup cache (DK_VAR only): the lexical address (parent-frame hops, slot) of
  // this variable's binding, so a lookup jumps straight to it instead of scanning env frames. A node's
  // lexical address is invariant across all its evaluations (frames of the same lexical role are rebuilt
  // structurally identically; closures capture lexical envs at constant depth; the global env is fixed
  // before the main eval), so the cached address is reused -- guarded by a binds[slot]==symbol check that
  // slow-paths + re-caches on any mismatch. vhops < 0 means "not yet resolved".
  mutable int32_t vhops = -1;
  mutable int32_t vslot = 0;

  static Datum mk_atom(std::string s) { Datum d; d.is_atom = true; d.atom = std::move(s); return d; }
  static Datum mk_list(std::vector<Datum> xs) { Datum d; d.is_atom = false; d.list = std::move(xs); return d; }

  bool is_list() const { return !is_atom; }
  bool is_sym(const char* s) const { return is_atom && atom == s; }
  size_t size() const { return list.size(); }
  // Convenience: head symbol of a list, or "" if not a non-empty list with atom head.
  const std::string& head_sym() const {
    static const std::string empty;
    if (is_atom || list.empty() || !list[0].is_atom) return empty;
    return list[0].atom;
  }
};

struct ParseError : std::runtime_error { using std::runtime_error::runtime_error; };

// Tokenize + parse one or more top-level forms from Scheme source.
std::vector<std::string> tokenize(const std::string& src);
std::vector<Datum> parse_top_level(const std::string& src);   // list of top-level form data
Datum parse_one(const std::string& src);                      // exactly one form

// Recursively lower sugar (let*/when/unless/loop/recur, vec/mat, multi-expr bodies, cond, define)
// to the core forms the interpreter implements. Quoted data is left untouched.
Datum expand_macros(const Datum& node);

// Reset the macro-expander's gensym counter (Phase 5: call once per program load so generated symbol
// names are per-program -- deterministic regardless of how many programs a worker thread parsed before).
void reset_gensym();

// Serialize a datum back to Scheme source (for debugging / metacircular quoting).
std::string datum_to_string(const Datum& d);

}  // namespace ndvm
