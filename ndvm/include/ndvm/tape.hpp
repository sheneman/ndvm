// Native define-by-run reverse-mode tape. Design section 7.
// Forward execution appends a compact record for DIFFERENTIABLE primitives only; structural ops
// (symbol lookup, tag check, cons, car/cdr, dispatch) emit nothing. Backward replays in reverse.
// Adjoint accumulation is vectorized over the payload batch dimension.
#pragma once
#include <cstdint>
#include <vector>

namespace ndvm {

enum class ADOpc : uint16_t {
  ADD, SUB, MUL, DIV, NEG,
  EXP, LOG, SQRT, POW, SIN, COS,
  MIN_REALIZED, MAX_REALIZED,   // gradient flows to the realized argument only
  SELECT,                       // masked merge of divergent batch lanes
  DOT, MATMUL, REDUCE_SUM, COPY,
};

using PayloadId = uint32_t;
using MaskId = uint32_t;
constexpr MaskId kNoMask = 0xFFFFFFFFu;

struct TapeNode {
  ADOpc op;
  PayloadId out;
  PayloadId in1;
  PayloadId in2;
  uint32_t aux;     // op-specific (e.g., reduction axis, pow exponent id)
  MaskId mask;      // lane mask for data-dependent numeric branches (design section 9.3)
};

// Arena-allocated tape; thread-local per candidate-fit, reused across optimizer epochs.
struct Tape {
  std::vector<TapeNode> nodes;
  void push(const TapeNode& n) { nodes.push_back(n); }
  void clear() { nodes.clear(); }
  // backward(): replay nodes in reverse, dispatching per-op adjoint kernels. Lands in Phase 2.
};

}  // namespace ndvm
