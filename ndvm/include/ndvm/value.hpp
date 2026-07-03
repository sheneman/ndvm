// NDVM runtime value: scalar structural identity, payload index for numeric data.
// Design section 4.1/4.4. Tags are NEVER differentiated; only numeric payloads are.
#pragma once
#include <cstdint>

namespace ndvm {

enum class Tag : uint8_t {
  NIL,
  BOOL,
  INT,
  FLOAT,
  CHAR,
  SYMBOL,
  PAIR,
  STRING,
  CLOSURE,
  VECTOR,
};

// Sentinel for non-numeric values (no entry in the differentiable payload table).
constexpr uint32_t kInvalidPayload = 0xFFFFFFFFu;

// A runtime value is a compact scalar triple. `aux` reuses the same 32 bits for a symbol id,
// heap address, closure id, vector id, or immediate int depending on `tag`. `payload_id` indexes
// the differentiable payload table iff the value is numeric (FLOAT), else kInvalidPayload.
struct Value {
  Tag tag;
  uint32_t aux;
  uint32_t payload_id;
};

// Immediate constructors for non-numeric values (no payload-table allocation).
inline Value make_nil()              { return Value{Tag::NIL,    0,                kInvalidPayload}; }
inline Value make_bool(bool b)       { return Value{Tag::BOOL,   uint32_t(b),      kInvalidPayload}; }
inline Value make_int(int32_t i)     { return Value{Tag::INT,    uint32_t(i),      kInvalidPayload}; }
inline Value make_symbol(uint32_t s) { return Value{Tag::SYMBOL, s,                kInvalidPayload}; }

// Numeric value referencing a (possibly batched) differentiable payload.
inline Value make_float(uint32_t payload_id) { return Value{Tag::FLOAT, 0, payload_id}; }

inline bool is_numeric(const Value& v) { return v.payload_id != kInvalidPayload; }

}  // namespace ndvm
