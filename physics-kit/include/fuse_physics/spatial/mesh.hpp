#pragma once

/// A native, serial, tier-0 computational mesh, adapted from the GENERIC
/// topology surface of KARMA's karma_mesh.h/.C (docs/dev/physics-kit-
/// disposition.md Appendix A, port-adapt, owner decision D-A).
///
/// WHY NATIVE, NOT MOAB-BACKED (the gap this header names per Appendix A's
/// own instruction: "no fuse::backend equivalent (MeshView is remap-
/// oriented, not FE-assembly-oriented)"). KARMA's KARMAMesh owns a
/// moab::Core, is constructed with an MPI_Comm, and most of its substance
/// (resolve_shared_entities, ParallelComm, ghost layers, GLOBAL_ID/
/// LOCAL_ID tags) is genuinely about a DISTRIBUTED, MOAB-backed mesh.
/// physics-kit is tier 0 (docs/dev/physics-kit.md: nothing beyond Kokkos
/// and fmt) and this track's own standing rules forbid MPI runs in this
/// sandbox, so a build/test of that distributed, MOAB-bound surface is not
/// something this wave can produce AND verify here -- not a disposition
/// decision, a verification-budget one, named as an open gap rather than
/// guessed at (see design_notes/open_questions for the explicit statement).
///
/// What IS ported, natively: the shape of query KARMAMesh exposes that does
/// not actually need MOAB or MPI underneath it -- element/vertex storage,
/// side-vertex lookup in CN order (via ReferenceTopology's own side
/// tables), neighbor-by-shared-side, boundary classification, structured-
/// box generation, bounding box and centroid, and KARMA's own private
/// global/local id numbering scheme (assign_ids). A FUSE_HAVE_MOAB-gated
/// translation unit that binds this same query surface onto a real MOAB
/// database -- so a mesh read from a .h5m/.vtk/.exo file reaches the same
/// FEBase code this wave's own mesh does -- is future work, named here
/// rather than attempted without MOAB/MPI to build or test it against.
///
/// WAVE 1 SCOPE: hexahedral (Hex8) structured-box elements only
/// (generate_box), matching this wave's ReferenceTopology/Basis coverage.
/// karma_mesh_extract_edge.C / karma_mesh_extract_face.C (mesh_utils.hpp)
/// are ported against this same native Mesh.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"

namespace fuse::physics::spatial {

using GlobalId = std::uint64_t;
using LocalId = std::uint64_t;
/// An opaque per-mesh element or vertex handle. 0 is reserved as "no
/// entity", matching KARMA's own moab::EntityHandle convention (its
/// reinit()s and constructors refuse a zero handle).
using EntityHandle = std::size_t;

struct SideId {
  EntityHandle element;
  unsigned side;

  [[nodiscard]] bool operator==(const SideId&) const = default;
};

/// A native hexahedral structured-grid mesh, read/write surface restricted
/// to what this wave's FE core and tests need. See this header's own
/// comment for what is, and is not, ported from KARMAMesh.
class Mesh {
 public:
  explicit Mesh(unsigned dimension) : dimension_(dimension) {}

  [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::size_t n_vertices() const noexcept { return vertices_.size(); }
  [[nodiscard]] std::size_t n_elements() const noexcept { return elements_.size(); }

  /// Generate a structured box of nx*ny*nz Hex8 elements spanning [lo,hi].
  void generate_box(unsigned nx, unsigned ny, unsigned nz, const Point& lo, const Point& hi);

  [[nodiscard]] unsigned n_sides(EntityHandle element) const { return fe::ReferenceTopology(fe::GeometryVariant::Hex8).side_count(); }
  /// Global vertex handles of one side, in the SAME CN order
  /// ReferenceTopology::side_nodes() returns for Hex8.
  [[nodiscard]] std::vector<EntityHandle> side_vertices(EntityHandle element, unsigned side) const;
  [[nodiscard]] std::vector<GlobalId> canonical_side_vertex_ids(EntityHandle element, unsigned side) const;
  [[nodiscard]] std::optional<EntityHandle> neighbor(EntityHandle element, unsigned side) const;
  [[nodiscard]] bool is_boundary_side(EntityHandle element, unsigned side) const { return !neighbor(element, side).has_value(); }
  [[nodiscard]] SideId side_id(EntityHandle element, unsigned side) const { return {element, side}; }

  /// KARMA numbers entities in its own private tags rather than relying on
  /// a file-provided GLOBAL_ID (karma_mesh.h's own comment: GLOBAL_ID
  /// belongs to the file and KARMA never writes it). This native mesh has
  /// no file-provided numbering to preserve, so assign_ids() simply assigns
  /// the handle itself as both global and local id -- the identity map is
  /// the CORRECT idempotent numbering here, not a simplification of a
  /// richer one, since every element/vertex in this mesh already has a
  /// dense, contiguous handle by construction (generate_box's own loop
  /// order).
  void assign_ids() noexcept {}
  [[nodiscard]] GlobalId global_id(EntityHandle entity) const noexcept { return static_cast<GlobalId>(entity); }
  [[nodiscard]] LocalId local_id(EntityHandle entity) const noexcept { return static_cast<LocalId>(entity); }

  /// Vertex coordinates of one element, in reference-node order; returns
  /// the count written (8 for Hex8).
  [[nodiscard]] std::size_t element_coordinates(EntityHandle element, std::span<Point> coordinates) const;
  [[nodiscard]] Point centroid(EntityHandle element) const;
  [[nodiscard]] std::pair<Point, Point> bounding_box() const;

  [[nodiscard]] const std::vector<Point>& vertex_positions() const noexcept { return vertices_; }
  [[nodiscard]] const std::array<EntityHandle, 8>& element_connectivity(EntityHandle element) const {
    return elements_.at(element - 1);
  }

 private:
  unsigned dimension_;
  std::vector<Point> vertices_;                       // handle i -> vertices_[i-1]
  std::vector<std::array<EntityHandle, 8>> elements_;  // handle i -> elements_[i-1], Hex8 connectivity
};

}  // namespace fuse::physics::spatial
