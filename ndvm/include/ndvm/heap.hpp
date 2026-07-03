// Immutable write-once arena heap. Design section 5.1.
// The DMCI subset excludes mutation, so cons/closure/vector cells are append-only; car/cdr are
// direct arena loads. No PyTorch in-place-mutation hazards; AD is handled by the native tape, not
// by versioned tensor buffers.
#pragma once
#include <cstdint>
#include <vector>
#include "ndvm/value.hpp"

namespace ndvm {

struct PairCell {
  Value car;
  Value cdr;
};

struct ClosureCell {
  uint32_t code_ptr;   // entry pc in evaluator bytecode
  Value env;
};

struct VectorCell {
  uint32_t start;      // index into vector_elements
  uint32_t length;
};

template <class T>
using Arena = std::vector<T>;  // append-only; addresses are stable indices

struct Heap {
  Arena<PairCell> pairs;
  Arena<ClosureCell> closures;
  Arena<VectorCell> vectors;
  Arena<Value> vector_elements;

  // cons appends a PairCell and returns Value{PAIR, addr, kInvalidPayload}; car/cdr are arena loads.
  Value cons(Value a, Value d) {
    uint32_t addr = static_cast<uint32_t>(pairs.size());
    pairs.push_back(PairCell{a, d});
    return Value{Tag::PAIR, addr, kInvalidPayload};
  }
  const Value& car(Value p) const { return pairs[p.aux].car; }
  const Value& cdr(Value p) const { return pairs[p.aux].cdr; }
};

}  // namespace ndvm
