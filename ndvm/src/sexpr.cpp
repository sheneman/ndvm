// NDVM front-end implementation: reader + macro expander.
// Faithful port of neural_compiler/parser/scheme_parser.py (tokenize/_parse_sexpr) and
// neural_compiler/dmci.py (expand_macros and helpers). Kept bug-for-bug compatible where the
// oracle's expansion has quirks (e.g. let* base case leaves the final binding unexpanded), so the
// native runtime evaluates exactly the datum the oracle's evaluator would see.
#include "sexpr.hpp"
#include <cctype>
#include <cstdlib>

namespace ndvm {

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

static bool is_delim(char c) {
  switch (c) {
    case '(': case ')': case '[': case ']':
    case '\'': case '`': case ',': case ';':
    case ' ': case '\t': case '\n': case '\r':
      return true;
    default: return false;
  }
}

std::vector<std::string> tokenize(const std::string& s) {
  std::vector<std::string> toks;
  size_t i = 0, n = s.size();
  while (i < n) {
    char c = s[i];
    if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
    if (c == ';') { while (i < n && s[i] != '\n') ++i; continue; }
    if (c == '\'') { toks.push_back("'"); ++i; continue; }
    if (c == '`') { toks.push_back("`"); ++i; continue; }
    if (c == ',') {
      if (i + 1 < n && s[i + 1] == '@') { toks.push_back(",@"); i += 2; }
      else { toks.push_back(","); ++i; }
      continue;
    }
    if (c == '(' || c == ')' || c == '[' || c == ']') { toks.emplace_back(1, c); ++i; continue; }
    if (c == '#') {
      if (i + 1 < n && s[i + 1] == '\\') {
        if (i + 2 >= n) throw ParseError("Incomplete character literal");
        size_t j = i + 2;
        while (j < n && !is_delim(s[j])) ++j;
        toks.push_back(s.substr(i, j - i));
        i = j;
      } else if (i + 1 < n && (s[i + 1] == 't' || s[i + 1] == 'f')) {
        toks.push_back(s.substr(i, 2));
        i += 2;
      } else {
        throw ParseError("Unexpected '#'");
      }
      continue;
    }
    size_t j = i;
    while (j < n && !is_delim(s[j])) ++j;
    toks.push_back(s.substr(i, j - i));
    i = j;
  }
  return toks;
}

// ---------------------------------------------------------------------------
// Parser (tokens -> Datum)
// ---------------------------------------------------------------------------

static Datum parse_sexpr(const std::vector<std::string>& t, size_t& pos);

static Datum parse_list(const std::vector<std::string>& t, size_t& pos, const char* close) {
  std::vector<Datum> items;
  while (pos < t.size() && t[pos] != close) items.push_back(parse_sexpr(t, pos));
  if (pos >= t.size()) throw ParseError("Missing closing delimiter");
  ++pos;  // skip close
  return Datum::mk_list(std::move(items));
}

static Datum parse_sexpr(const std::vector<std::string>& t, size_t& pos) {
  if (pos >= t.size()) throw ParseError("Unexpected end of input");
  const std::string& tok = t[pos];
  if (tok == "(") { ++pos; return parse_list(t, pos, ")"); }
  if (tok == "[") {
    // [..] is a vec literal: ["vec", elems...]
    ++pos;
    Datum d = parse_list(t, pos, "]");
    d.list.insert(d.list.begin(), Datum::mk_atom("vec"));
    return d;
  }
  if (tok == "'") { ++pos; Datum q = parse_sexpr(t, pos);
    return Datum::mk_list({Datum::mk_atom("quote"), q}); }
  if (tok == ")" || tok == "]") throw ParseError("Unexpected '" + tok + "'");
  // Note: quasiquote/unquote are not used by the target programs; treat as atoms if seen.
  ++pos;
  return Datum::mk_atom(tok);
}

std::vector<Datum> parse_top_level(const std::string& src) {
  std::vector<std::string> t = tokenize(src);
  std::vector<Datum> forms;
  size_t pos = 0;
  while (pos < t.size()) forms.push_back(parse_sexpr(t, pos));
  return forms;
}

Datum parse_one(const std::string& src) {
  std::vector<std::string> t = tokenize(src);
  if (t.empty()) throw ParseError("Empty input");
  size_t pos = 0;
  Datum d = parse_sexpr(t, pos);
  if (pos < t.size()) throw ParseError("Unexpected trailing tokens");
  return d;
}

// ---------------------------------------------------------------------------
// Macro expander (port of dmci.py expand_macros + helpers)
// ---------------------------------------------------------------------------

// thread_local (Phase 5): each worker thread expands macros into its OWN AST, so the gensym counter must
// be per-thread. Gensym names are internal to one thread's self-contained program and never cross a thread
// boundary, so per-thread sequences keep results byte-identical to serial.
static thread_local long g_gensym = 0;
static Datum gensym(const char* prefix) {
  return Datum::mk_atom(std::string(prefix) + std::to_string(++g_gensym));
}
void reset_gensym() { g_gensym = 0; }

static Datum sym(const char* s) { return Datum::mk_atom(s); }
static Datum lst(std::vector<Datum> xs) { return Datum::mk_list(std::move(xs)); }

// A body (list of forms) as a SINGLE expression; multi-expr bodies are begin-wrapped.
static Datum begin_wrap(std::vector<Datum> forms) {
  if (forms.size() == 1) return forms[0];
  std::vector<Datum> out; out.push_back(sym("begin"));
  for (auto& f : forms) out.push_back(f);
  return lst(std::move(out));
}

static Datum expand_binding(const Datum& b) {
  if (b.is_list() && b.size() >= 2) return lst({b.list[0], expand_macros(b.list[1])});
  return b;
}

// (let* (b1 b2 ...) body...) -> a SINGLE multi-binding let. NDVM's `let` evaluates its bindings
// sequentially in ONE frame (each binding's RHS sees the prior bindings; see the SF_LET handler), which
// is exactly let* semantics -- so a single let is result-identical to the nested single-binding lets the
// oracle (dmci.py) builds, but with ONE frame instead of N. That cuts environment DEPTH, so variable
// lookups walk far fewer parent frames (the dominant lookup cost on deep programs like the 80-step
// Kalman). The expansion CONTENT is unchanged: every binding's RHS is expanded except the last, which is
// left raw to match dmci.py's base-case quirk (so the evaluated program is byte-for-byte equivalent).
static Datum expand_let_star(const std::vector<Datum>& binds, const std::vector<Datum>& body) {
  std::vector<Datum> ebinds;
  for (size_t i = 0; i < binds.size(); ++i)
    ebinds.push_back(i + 1 < binds.size() ? expand_binding(binds[i]) : binds[i]);  // last RHS left raw
  std::vector<Datum> ebody; for (auto& x : body) ebody.push_back(expand_macros(x));
  return lst({sym("let"), lst(std::move(ebinds)), begin_wrap(std::move(ebody))});
}

// Replace a tail-position (recur a...) with the self-application (selfsym selfsym a...).
static Datum rewrite_recur(const Datum& e, const Datum& selfsym, size_t arity, bool tail) {
  if (e.is_atom || e.list.empty()) return e;
  const std::string& head = e.head_sym();
  if (head == "quote") return e;
  if (head == "recur") {
    if (!tail) throw ParseError("recur must be in tail position inside a loop");
    if (e.size() - 1 != arity) throw ParseError("recur arity mismatch with loop variables");
    std::vector<Datum> out{selfsym, selfsym};
    for (size_t i = 1; i < e.size(); ++i) out.push_back(rewrite_recur(e.list[i], selfsym, arity, false));
    return lst(std::move(out));
  }
  if (head == "if") {
    std::vector<Datum> out{sym("if"), rewrite_recur(e.list[1], selfsym, arity, false)};
    if (e.size() > 2) out.push_back(rewrite_recur(e.list[2], selfsym, arity, tail));
    if (e.size() > 3) out.push_back(rewrite_recur(e.list[3], selfsym, arity, tail));
    return lst(std::move(out));
  }
  if (head == "begin") {
    std::vector<Datum> out{sym("begin")};
    size_t n = e.size() - 1;
    for (size_t i = 1; i < e.size(); ++i)
      out.push_back(rewrite_recur(e.list[i], selfsym, arity, tail && i == n));
    return lst(std::move(out));
  }
  if (head == "cond") {
    std::vector<Datum> out{sym("cond")};
    for (size_t ci = 1; ci < e.size(); ++ci) {
      const Datum& clause = e.list[ci];
      if (clause.is_list() && !clause.list.empty()) {
        const Datum& test = clause.list[0];
        Datum test2 = test.is_sym("else") ? test : rewrite_recur(test, selfsym, arity, false);
        std::vector<Datum> newc{test2};
        size_t nr = clause.size() - 1;
        for (size_t j = 1; j < clause.size(); ++j)
          newc.push_back(rewrite_recur(clause.list[j], selfsym, arity, tail && j == nr));
        out.push_back(lst(std::move(newc)));
      } else {
        out.push_back(clause);
      }
    }
    return lst(std::move(out));
  }
  if (head == "let" || head == "letrec") {
    const std::vector<Datum>& binds = e.size() > 1 ? e.list[1].list : std::vector<Datum>{};
    std::vector<Datum> nb;
    for (auto& b : binds) {
      if (b.is_list() && b.size() >= 2) nb.push_back(lst({b.list[0], rewrite_recur(b.list[1], selfsym, arity, false)}));
      else nb.push_back(b);
    }
    std::vector<Datum> out{sym(head.c_str()), lst(std::move(nb))};
    if (e.size() > 2) out.push_back(rewrite_recur(e.list[2], selfsym, arity, tail));
    return lst(std::move(out));
  }
  if (head == "lambda") {
    std::vector<Datum> out{sym("lambda"), e.size() > 1 ? e.list[1] : lst({})};
    if (e.size() > 2) out.push_back(rewrite_recur(e.list[2], selfsym, arity, false));
    return lst(std::move(out));
  }
  // generic application: operator + operands all non-tail
  std::vector<Datum> out;
  for (auto& x : e.list) out.push_back(rewrite_recur(x, selfsym, arity, false));
  return lst(std::move(out));
}

// (loop ((v init)...) body...) -> (let ((f (lambda (self v...) body[recur->(self self ...)]))) (f f init...))
static Datum expand_loop(const Datum& node) {
  const std::vector<Datum>& binds = node.size() > 1 ? node.list[1].list : std::vector<Datum>{};
  std::vector<Datum> names, inits;
  for (auto& b : binds) { names.push_back(b.list[0]); inits.push_back(expand_macros(b.list[1])); }
  Datum fname = gensym("__loop_");
  Datum selfsym = gensym("__self_");
  std::vector<Datum> body;
  for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
  Datum body_expr = begin_wrap(std::move(body));
  Datum rewritten = rewrite_recur(body_expr, selfsym, names.size(), true);
  std::vector<Datum> params{selfsym};
  for (auto& nm : names) params.push_back(nm);
  Datum lam = lst({sym("lambda"), lst(std::move(params)), rewritten});
  std::vector<Datum> call{fname, fname};
  for (auto& in : inits) call.push_back(in);
  return lst({sym("let"), lst({lst({fname, lam})}), lst(std::move(call))});
}

Datum expand_macros(const Datum& node) {
  if (node.is_atom || node.list.empty()) return node;
  const std::string& head = node.head_sym();
  if (head == "quote") return node;
  if (head == "let*") {
    std::vector<Datum> binds = node.size() > 1 ? node.list[1].list : std::vector<Datum>{};
    std::vector<Datum> body(node.list.begin() + (node.size() > 2 ? 2 : node.size()), node.list.end());
    return expand_let_star(binds, body);
  }
  if (head == "when") {
    std::vector<Datum> body; for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
    return lst({sym("if"), expand_macros(node.list[1]), begin_wrap(std::move(body)), sym("#f")});
  }
  if (head == "unless") {
    std::vector<Datum> body; for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
    return lst({sym("if"), expand_macros(node.list[1]), sym("#f"), begin_wrap(std::move(body))});
  }
  if (head == "loop") return expand_loop(node);
  if (head == "vec" || head == "mat") {
    std::vector<Datum> elems{sym("list")};
    for (size_t i = 1; i < node.size(); ++i) elems.push_back(expand_macros(node.list[i]));
    return lst({sym(head.c_str()), lst(std::move(elems))});
  }
  if (head == "let" || head == "letrec") {
    const std::vector<Datum>& binds = node.size() > 1 ? node.list[1].list : std::vector<Datum>{};
    std::vector<Datum> nb; for (auto& b : binds) nb.push_back(expand_binding(b));
    std::vector<Datum> body; for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
    return lst({sym(head.c_str()), lst(std::move(nb)), begin_wrap(std::move(body))});
  }
  if (head == "lambda") {
    std::vector<Datum> body; for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
    return lst({sym("lambda"), node.size() > 1 ? node.list[1] : lst({}), begin_wrap(std::move(body))});
  }
  if (head == "cond") {
    std::vector<Datum> out{sym("cond")};
    for (size_t ci = 1; ci < node.size(); ++ci) {
      const Datum& clause = node.list[ci];
      if (clause.is_list() && !clause.list.empty()) {
        const Datum& test = clause.list[0];
        Datum test2 = test.is_sym("else") ? test : expand_macros(test);
        std::vector<Datum> rest; for (size_t i = 1; i < clause.size(); ++i) rest.push_back(expand_macros(clause.list[i]));
        if (!rest.empty()) out.push_back(lst({test2, begin_wrap(std::move(rest))}));
        else out.push_back(lst({test2}));
      } else out.push_back(clause);
    }
    return lst(std::move(out));
  }
  if (head == "define") {
    const Datum& target = node.list[1];
    if (target.is_list()) {  // (define (f a..) body...)
      std::vector<Datum> body; for (size_t i = 2; i < node.size(); ++i) body.push_back(expand_macros(node.list[i]));
      return lst({sym("define"), target, begin_wrap(std::move(body))});
    }
    std::vector<Datum> out{sym("define"), target};
    for (size_t i = 2; i < node.size(); ++i) out.push_back(expand_macros(node.list[i]));
    return lst(std::move(out));
  }
  std::vector<Datum> out; for (auto& x : node.list) out.push_back(expand_macros(x));
  return lst(std::move(out));
}

std::string datum_to_string(const Datum& d) {
  if (d.is_atom) return d.atom;
  std::string s = "(";
  for (size_t i = 0; i < d.list.size(); ++i) { if (i) s += " "; s += datum_to_string(d.list[i]); }
  return s + ")";
}

}  // namespace ndvm
