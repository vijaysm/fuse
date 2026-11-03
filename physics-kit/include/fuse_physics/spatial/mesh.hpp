#pragma once

/// A computational mesh, adapted from the GENERIC topology surface of
/// KARMA's karma_mesh.h/.C (port-adapted; see
/// docs/dev/physics-kit.md).
///
/// physics-kit reaches MOAB only through fuse::backend, as the layering
/// page says. This mesh binds to `fuse::MeshView` (mesh_view.hpp) -- the SAME abstract
/// interface `fuse::native::NativeMeshView` and `MoabMeshView` both
/// implement -- rather than to MOAB directly or to the native backend's own
/// concrete types. `from_mesh_view()` below reads ownership
/// (`is_owned`/`owned_count`), global ids (`global_id`), and connectivity
/// (`cell_vertices`) through that one interface, so a mesh built from
/// `fuse::native::NativeMeshView::create_from_host()` and a (future) mesh
/// built from a real MOAB file reach IDENTICAL physics-kit code. Partition-
/// aware queries (owned vs ghost vertex/cell, global ids) are part of the
/// public surface, so that physics can run in parallel over a real
/// communicator rather than on MPI_COMM_SELF alone.
///
/// The self-generated box constructors (`generate_box`, 3-D Hex8;
/// `generate_2d_box`, 2-D Quad4) mark every vertex and
/// cell owned with an identity global id, which is the CORRECT numbering
/// for a single-rank, self-generated mesh, not a simplification of a
/// richer one.
///
/// STILL NATIVE AT TIER 0. Building and testing an ACTUAL multi-rank
/// `fuse::native::NativeMeshView` or a MOAB-backed one needs MPI (MOAB
/// itself is fine without MPI, but this tree's own tier ladder requires
/// FUSE_ENABLE_MPI before FUSE_ENABLE_MOAB, and no MPI runs here). The
/// tier-0 verification of `from_mesh_view` therefore uses a HAND-BUILT,
/// two-"rank" `fuse::native::MeshArrays` fixture (via `NativeMeshBuilder`,
/// which needs no MPI call at all: `add_vertex`/`add_cell` take `owner`/
/// `is_owned` directly) -- see spatial_mesh_test.cpp's
/// FromMeshViewReadsOwnershipAndGlobalIds. The genuine multi-rank case (an
/// actual partitioned box built and exchanged across real MPI ranks) is
/// `test/parallel/physics_mesh_mpi_test.cpp`: compiled in this tree with
/// FUSE_ENABLE_MPI=ON, never executed in this sandbox.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"

// Forward-declared, never #included here: fuse::MeshView is fuse::backend's
// own public type (mesh_view.hpp), and the only thing this header's public
// API needs to name about it is a reference parameter in from_mesh_view()'s
// declaration, which a forward declaration covers. The #include itself
// lives in mesh.cpp, keeping this header's own third-party-header surface
// unchanged (tools/check-layering.py's FORBIDDEN_PUBLIC rule is about
// third-party headers specifically; fuse::MeshView is a fuse:: type, not a
// third party, so naming it here would be fine either way -- forward
// declaration is simply the lighter include).
namespace fuse {
class MeshView;
}

namespace fuse::physics::spatial {

using GlobalId = std::uint64_t;
using LocalId = std::uint64_t;
/// An opaque per-mesh element or vertex handle, 1-based (0 reserved as "no
/// entity", matching KARMA's own moab::EntityHandle convention).
using EntityHandle = std::size_t;

struct SideId {
  EntityHandle element;
  unsigned side;

  [[nodiscard]] bool operator==(const SideId&) const = default;
};

/// A mesh: vertex positions, variable-arity element connectivity (each
/// element carries its own fe::GeometryVariant, at Lagrange order 1 --
/// Edge2/Tri3/Quad4/Tet4/Pyramid5/Prism6/Hex8), ownership flags and global
/// ids. Self-generated (generate_box/generate_2d_box) or read from a
/// fuse::MeshView (from_mesh_view).
class Mesh {
 public:
  explicit Mesh(unsigned dimension) : dimension_(dimension) {}

  [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
  [[nodiscard]] std::size_t n_vertices() const noexcept { return vertices_.size(); }
  [[nodiscard]] std::size_t n_elements() const noexcept { return elements_.size(); }

  /// Generate a structured box of nx*ny*nz Hex8 elements spanning [lo,hi].
  /// Every vertex and element is owned, with an identity global id.
  void generate_box(unsigned nx, unsigned ny, unsigned nz, const Point& lo, const Point& hi);

  /// Generate a structured 2-D box of nx*ny Quad4 elements spanning
  /// [lo,hi] (z ignored). Gives
  /// `extract_boundary_edges` (mesh_utils.hpp, the karma_mesh_extract_edge.C
  /// port) a 2-D source mesh to extract a 1-D boundary from.
  void generate_2d_box(unsigned nx, unsigned ny, const Point& lo, const Point& hi);

  /// Build from an arbitrary fuse::MeshView's TOP-dimension entities
  /// (`view.dimension()`), reading ownership, global ids and connectivity
  /// through the view's own abstract interface (owned_count/is_owned/
  /// global_id/cell_vertices, mesh_view.hpp) -- works identically whether
  /// `view` is `fuse::native::NativeMeshView` or a MOAB-backed one, since
  /// both implement the same interface. Refuses (E-PHYSICS-MESH-CELL-SHAPE)
  /// if a cell's own CellShape has no fe::GeometryVariant equivalent at
  /// order 1 (Polygon, Polyhedron) -- named so the message says which shape
  /// was rejected, not just "unsupported".
  [[nodiscard]] static Mesh from_mesh_view(const MeshView& view);

  [[nodiscard]] fe::GeometryVariant element_kind(EntityHandle element) const {
    return element_kind_.at(element - 1);
  }
  [[nodiscard]] unsigned n_sides(EntityHandle element) const {
    return fe::ReferenceTopology(element_kind(element)).side_count();
  }
  /// Global vertex handles of one side, in the SAME CN order
  /// ReferenceTopology::side_nodes() returns for this element's own kind.
  [[nodiscard]] std::vector<EntityHandle> side_vertices(EntityHandle element, unsigned side) const;
  [[nodiscard]] std::vector<GlobalId> canonical_side_vertex_ids(EntityHandle element, unsigned side) const;
  /// Local vertex HANDLES of one EDGE (a 1-D sub-entity, distinct from a
  /// codimension-1 "side" on a 3-D cell -- fe::edge_vertices' own header
  /// comment), in `fe::edge_vertices`' own corner order. Lets
  /// `fe::DofMap` number a 3-D cell's interior edge dofs.
  [[nodiscard]] std::vector<EntityHandle> edge_vertices(EntityHandle element, unsigned edge) const;
  /// Sorted (order-independent) GLOBAL vertex ids of one edge, the exact
  /// identity-matching role `canonical_side_vertex_ids` plays for a side.
  [[nodiscard]] std::vector<GlobalId> canonical_edge_vertex_ids(EntityHandle element, unsigned edge) const;
  [[nodiscard]] std::optional<EntityHandle> neighbor(EntityHandle element, unsigned side) const;
  [[nodiscard]] bool is_boundary_side(EntityHandle element, unsigned side) const {
    return !neighbor(element, side).has_value();
  }
  [[nodiscard]] SideId side_id(EntityHandle element, unsigned side) const { return {element, side}; }

  /// KARMA numbers entities in its own private tags rather than relying on
  /// a file-provided GLOBAL_ID (karma_mesh.h's own comment). A self-
  /// generated mesh's identity numbering is assigned at construction
  /// already; assign_ids() stays a no-op for it.
  void assign_ids() noexcept {}
  /// Ambiguous on its own (vertex handle 5 and element handle 5 are
  /// different entities, and a self-generated mesh gives both the same
  /// identity id, which hid this): kept for call sites that
  /// only ever use it on a self-generated mesh, where vertex and element
  /// ids happen to coincide. A caller that can tell which space it means
  /// -- every from_mesh_view()-sourced mesh, where the two spaces
  /// genuinely differ -- uses vertex_global_id/element_global_id instead.
  [[nodiscard]] GlobalId global_id(EntityHandle entity) const noexcept { return global_id_of(entity); }
  [[nodiscard]] GlobalId vertex_global_id(EntityHandle vertex) const noexcept { return vertex_gid_.at(vertex - 1); }
  [[nodiscard]] GlobalId element_global_id(EntityHandle element) const noexcept { return element_gid_.at(element - 1); }
  [[nodiscard]] LocalId local_id(EntityHandle entity) const noexcept { return static_cast<LocalId>(entity); }

  // ---- partition-aware queries ------------------------------------------
  [[nodiscard]] bool is_vertex_owned(EntityHandle vertex) const noexcept { return vertex_owned_.at(vertex - 1); }
  [[nodiscard]] bool is_element_owned(EntityHandle element) const noexcept { return element_owned_.at(element - 1); }
  /// The RANK that owns `vertex` -- `is_vertex_owned(vertex) == (vertex_owner(vertex) ==
  /// <this rank's own number>)`. A shared sub-entity's owner is
  /// the MINIMUM over its corner vertices' own `vertex_owner`, computable on every
  /// sharing rank from this query alone. A self-generated mesh (generate_box/
  /// generate_2d_box) or one built through add_vertex with no explicit owner_rank: 0
  /// (this port's own single-rank convention, matching the serial MeshView backend's
  /// identical `owner` override, mesh_view.hpp's own header comment).
  [[nodiscard]] int vertex_owner(EntityHandle vertex) const noexcept { return vertex_owner_.at(vertex - 1); }
  [[nodiscard]] std::size_t owned_vertex_count() const noexcept;
  [[nodiscard]] std::size_t owned_element_count() const noexcept;

  /// Vertex coordinates of one element, in reference-node order; returns
  /// the count written.
  [[nodiscard]] std::size_t element_coordinates(EntityHandle element, std::span<Point> coordinates) const;
  [[nodiscard]] Point centroid(EntityHandle element) const;
  [[nodiscard]] std::pair<Point, Point> bounding_box() const;

  [[nodiscard]] const std::vector<Point>& vertex_positions() const noexcept { return vertices_; }
  [[nodiscard]] const std::vector<EntityHandle>& element_connectivity(EntityHandle element) const {
    return elements_.at(element - 1);
  }

  /// Appends one vertex (owned, identity global id unless overridden) and
  /// returns its handle. Exposed so mesh_utils.hpp's boundary-extraction
  /// functions can build a lower-dimensional Mesh directly, without going
  /// through a box generator.
  EntityHandle add_vertex(const Point& position, bool owned = true,
                           std::optional<GlobalId> gid = std::nullopt, int owner_rank = 0);
  EntityHandle add_element(fe::GeometryVariant kind, std::vector<EntityHandle> connectivity, bool owned = true,
                            std::optional<GlobalId> gid = std::nullopt);

 private:
  [[nodiscard]] GlobalId global_id_of(EntityHandle entity) const noexcept;

  unsigned dimension_;
  std::vector<Point> vertices_;
  std::vector<bool> vertex_owned_;
  std::vector<int> vertex_owner_;     // the rank that owns each vertex
  std::vector<GlobalId> vertex_gid_;  // identity (handle) unless from_mesh_view overrides

  std::vector<std::vector<EntityHandle>> elements_;
  std::vector<fe::GeometryVariant> element_kind_;
  std::vector<bool> element_owned_;
  std::vector<GlobalId> element_gid_;
};

}  // namespace fuse::physics::spatial
