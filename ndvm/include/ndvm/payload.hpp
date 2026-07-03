// Differentiable numeric payload table. Design section 4.2.
// Structure-of-arrays: primal and adjoint live in dense buffers; a scalar Value points here.
// The batch dimension lives on the PAYLOAD, not on the structural value -- this is the critical
// representation change (one scalar tag/address object, B numeric elements only when needed).
#pragma once
#include <cstdint>
#include <vector>

namespace ndvm {

enum class DType : uint8_t { F32, F64 };

// Minimal shape descriptor: a leading batch axis B plus an optional inner shape.
struct Shape {
  uint32_t batch = 1;            // B (or product of flattened batch axes)
  uint32_t inner = 1;            // product of inner dims (1 for scalars)
};

struct Payload {
  DType dtype = DType::F32;
  Shape shape;
  uint32_t primal_offset = 0;    // offset into the primal buffer
  uint32_t adjoint_offset = 0;   // offset into the adjoint buffer
  uint32_t tape_birth = 0;       // tape position at allocation (for liveness/checkpointing)
};

// Dense SoA backing store. Phase 2 wires real allocation + batched kernels (design sections 7, 9, 11).
struct PayloadTable {
  std::vector<Payload> meta;
  std::vector<float> primal_buffer;
  std::vector<float> adjoint_buffer;
  // alloc_like / alloc(shape) etc. land in Phase 2.
};

}  // namespace ndvm
