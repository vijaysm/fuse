#pragma once

#include <memory>
#include <string_view>

#include "fuse/backend/mesh_view.hpp"

namespace fuse {

/// Dependency-free structured backend: no MPI, no files, no third-party
/// headers. It exists so the whole stack is testable before MOAB arrives.
///
/// SOURCES
///   "generated:line:N"      the unit interval [0,1], N cells
///   "generated:box:NxMxP"   the unit cube [0,1]^3, N*M*P cells
/// Each count must be a plain positive decimal integer -- no sign, no
/// whitespace, no other base -- and the resulting entity counts must fit
/// `std::size_t`. Anything else is `E-MESH-SOURCE` or `E-MESH-SIZE`; nothing
/// is guessed.
///
/// ENTITIES. For a box the backend represents dimensions 3 (cells), 2 (the
/// BOUNDARY faces, i.e. the skin -- interior faces are not enumerated) and 0
/// (vertices). For a line, dimensions 1 (cells) and 0 (vertices). Cell
/// measures sum to 1 (the domain volume) and boundary-face measures sum to 6
/// (the unit cube's surface area).
///
/// ENTITY SETS, all on a box and all of dimension 2:
///   "", "all"                 the whole mesh at the requested dimension
///   "left" / "right"          the x = 0 / x = 1 face,  M*P entities each
///   "front" / "back"          the y = 0 / y = 1 face,  N*P entities each
///   "bottom" / "top"          the z = 0 / z = 1 face,  N*M entities each
///   "wall"                    the entire boundary, 2*(N*M + M*P + N*P)
/// Any other name is `E-MESH-SET`.
///
/// SERIAL SEMANTICS. Every entity is owned; there are no ghosts, so
/// `owned_count == entity_count` and `is_owned` is always true. Global ids are
/// the local indices. Callers must not build these facts into their own code:
/// they are properties of this backend, not of `MeshView` (see the contract
/// there).
class SerialMeshBackend final : public MeshBackend {
 public:
  SerialMeshBackend();
  ~SerialMeshBackend() override;

  /// Moving transfers the loaded meshes and leaves the source empty but
  /// usable: a moved-from backend holds no meshes and reports `E-MESH-MOVED`
  /// from `load` and `view` rather than dereferencing a null pimpl.
  SerialMeshBackend(SerialMeshBackend&&) noexcept;
  SerialMeshBackend& operator=(SerialMeshBackend&&) noexcept;

  Result<void> load(const MeshDecl& decl) override;
  [[nodiscard]] Result<MeshId> mesh_id(std::string_view name) const override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "serial"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse
