// NDVM native runtime entry point (Phase 0 placeholder).
// The forward VM (Phase 1), reverse-mode tape (Phase 2), batching (Phase 3), and backends land here
// and in sibling translation units. See ../../docs/NDVM_native_differentiable_vm.md section 18.
#include "ndvm/value.hpp"
#include "ndvm/heap.hpp"
#include "ndvm/payload.hpp"
#include "ndvm/tape.hpp"

namespace ndvm {

const char* version() { return "ndvm 0.0.0-phase0-scaffold"; }

}  // namespace ndvm
