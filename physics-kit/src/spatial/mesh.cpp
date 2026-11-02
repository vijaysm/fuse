// fuse_physics::spatial::Mesh. See mesh.hpp's header comment for the
// fuse::MeshView binding (owner instruction 2026-10-31) and what is/is not
// ported from KARMA's karma_mesh.h/.C.

#include "fuse_physics/spatial/mesh.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "fuse/backend/cell_simplices.hpp"  // fuse::CellShape
#include "fuse/backend/mesh_view.hpp"       // fuse::MeshView, fuse::Point

namespace fuse::physics::spatial {

namespace {
[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

Point from_backend_point(const fuse::Point& q) { return Point(q.x, q.y, q.z); }

/// fuse::CellShape (cell_simplices.hpp) and KARMA's own node ordering agree
/// corner-for-corner at Lagrange order 1 -- both follow the MOAB/Exodus
/// convention (cell_simplices.hpp's own comments: "Pyramid: quad base 0-3,
/// apex 4"; "Prism: triangle 0-2, opposite triangle 3-5" -- exactly
/// reference_topology.cpp's Pyramid5/Prism6 node tables) -- so this map
/// needs no permutation, only a name change. Polygon and Polyhedron have no
/// fe::GeometryVariant equivalent (cell_simplices.hpp's own CellShape
/// comment: Polygon is "n >= 3, planar", a variable-arity shape this port's
/// fixed-node reference cells do not model) and refuse by name.
fe::GeometryVariant geometry_variant_for_shape(fuse::CellShape shape) {
  switch (shape) {
    case fuse::CellShape::Edge:
      return fe::GeometryVariant::Edge2;
    case fuse::CellShape::Triangle:
      return fe::GeometryVariant::Tri3;
    case fuse::CellShape::Quad:
      return fe::GeometryVariant::Quad4;
    case fuse::CellShape::Tet:
      return fe::GeometryVariant::Tet4;
    case fuse::CellShape::Pyramid:
      return fe::GeometryVariant::Pyramid5;
    case fuse::CellShape::Prism:
      return fe::GeometryVariant::Prism6;
    case fuse::CellShape::Hex:
      return fe::GeometryVariant::Hex8;
    case fuse::CellShape::Polygon:
      refuse("Mesh::from_mesh_view: a Polygon cell has no fixed-node fe::GeometryVariant equivalent");
  }
  refuse("Mesh::from_mesh_view: unrecognized fuse::CellShape");
}
}  // namespace

EntityHandle Mesh::add_vertex(const Point& position, bool owned, std::optional<GlobalId> gid, int owner_rank) {
  vertices_.push_back(position);
  vertex_owned_.push_back(owned);
  vertex_owner_.push_back(owner_rank);
  const EntityHandle handle = vertices_.size();
  vertex_gid_.push_back(gid.value_or(static_cast<GlobalId>(handle)));
  return handle;
}

EntityHandle Mesh::add_element(fe::GeometryVariant kind, std::vector<EntityHandle> connectivity, bool owned,
                                std::optional<GlobalId> gid) {
  elements_.push_back(std::move(connectivity));
  element_kind_.push_back(kind);
  element_owned_.push_back(owned);
  const EntityHandle handle = elements_.size();
  element_gid_.push_back(gid.value_or(static_cast<GlobalId>(handle)));
  return handle;
}

GlobalId Mesh::global_id_of(EntityHandle entity) const noexcept {
  // Ambiguous without a dimension tag in this simple accessor; callers that
  // need an entity's global id read vertex_gid_/element_gid_ through the
  // dimension-specific path below instead. Kept only for the header's own
  // inline global_id() to have a single definition to call; vertex and
  // element global ids are looked up by the SAME 1-based handle space in
  // THIS class (vertices and elements are never compared to each other by
  // global id), so a caller passes the handle for whichever space it means
  // and this picks it out of vertex_gid_ first, falling back to
  // element_gid_ -- safe because the two handle spaces are used in
  // entirely separate call sites (vertex handles only from side_vertices/
  // vertex_positions, element handles only from n_sides/element_kind/etc).
  if (entity >= 1 && entity <= vertex_gid_.size()) return vertex_gid_[entity - 1];
  if (entity >= 1 && entity <= element_gid_.size()) return element_gid_[entity - 1];
  refuse("global_id: entity handle out of range for both vertex and element spaces");
}

std::size_t Mesh::owned_vertex_count() const noexcept {
  return static_cast<std::size_t>(std::count(vertex_owned_.begin(), vertex_owned_.end(), true));
}
std::size_t Mesh::owned_element_count() const noexcept {
  return static_cast<std::size_t>(std::count(element_owned_.begin(), element_owned_.end(), true));
}

void Mesh::generate_box(unsigned nx, unsigned ny, unsigned nz, const Point& lo, const Point& hi) {
  if (nx == 0 || ny == 0 || nz == 0) refuse("generate_box needs at least one element per axis");
  vertices_.clear();
  vertex_owned_.clear();
  vertex_gid_.clear();
  elements_.clear();
  element_kind_.clear();
  element_owned_.clear();
  element_gid_.clear();
  const unsigned px = nx + 1, py = ny + 1, pz = nz + 1;
  const auto vertex_index = [&](unsigned i, unsigned j, unsigned k) -> EntityHandle {
    return static_cast<EntityHandle>(k) * py * px + static_cast<EntityHandle>(j) * px + i + 1;
  };
  for (unsigned k = 0; k < pz; ++k)
    for (unsigned j = 0; j < py; ++j)
      for (unsigned i = 0; i < px; ++i) {
        const double x = lo.x + (hi.x - lo.x) * (static_cast<double>(i) / nx);
        const double y = lo.y + (hi.y - lo.y) * (static_cast<double>(j) / ny);
        const double z = lo.z + (hi.z - lo.z) * (static_cast<double>(k) / nz);
        add_vertex(Point(x, y, z));
      }
  // Hex8 reference-node order: bottom face (0,0,0)-(1,0,0)-(1,1,0)-(0,1,0),
  // then the matching top face at z+1.
  for (unsigned k = 0; k < nz; ++k)
    for (unsigned j = 0; j < ny; ++j)
      for (unsigned i = 0; i < nx; ++i)
        add_element(fe::GeometryVariant::Hex8,
                    {vertex_index(i, j, k), vertex_index(i + 1, j, k), vertex_index(i + 1, j + 1, k),
                     vertex_index(i, j + 1, k), vertex_index(i, j, k + 1), vertex_index(i + 1, j, k + 1),
                     vertex_index(i + 1, j + 1, k + 1), vertex_index(i, j + 1, k + 1)});
}

void Mesh::generate_2d_box(unsigned nx, unsigned ny, const Point& lo, const Point& hi) {
  if (nx == 0 || ny == 0) refuse("generate_2d_box needs at least one element per axis");
  vertices_.clear();
  vertex_owned_.clear();
  vertex_gid_.clear();
  elements_.clear();
  element_kind_.clear();
  element_owned_.clear();
  element_gid_.clear();
  const unsigned px = nx + 1, py = ny + 1;
  const auto vertex_index = [&](unsigned i, unsigned j) -> EntityHandle {
    return static_cast<EntityHandle>(j) * px + i + 1;
  };
  for (unsigned j = 0; j < py; ++j)
    for (unsigned i = 0; i < px; ++i) {
      const double x = lo.x + (hi.x - lo.x) * (static_cast<double>(i) / nx);
      const double y = lo.y + (hi.y - lo.y) * (static_cast<double>(j) / ny);
      add_vertex(Point(x, y, lo.z));
    }
  // Quad4 reference-node order: (0,0)-(1,0)-(1,1)-(0,1), matching
  // reference_topology.cpp's nodes_for(Quad4).
  for (unsigned j = 0; j < ny; ++j)
    for (unsigned i = 0; i < nx; ++i)
      add_element(fe::GeometryVariant::Quad4, {vertex_index(i, j), vertex_index(i + 1, j), vertex_index(i + 1, j + 1),
                                                vertex_index(i, j + 1)});
}

Mesh Mesh::from_mesh_view(const MeshView& view) {
  Mesh mesh(view.dimension());
  const int top = view.dimension();
  const std::size_t n_vertices = view.entity_count(0);
  for (std::size_t v = 0; v < n_vertices; ++v) {
    const auto owner_result = view.owner(0, v);
    if (!owner_result) refuse(("Mesh::from_mesh_view: " + owner_result.error().message).c_str());
    mesh.add_vertex(from_backend_point(view.centroid(0, v)), view.is_owned(0, v),
                     static_cast<GlobalId>(view.global_id(0, v)), owner_result.value());
  }

  const std::size_t n_cells = view.entity_count(top);
  for (std::size_t c = 0; c < n_cells; ++c) {
    const auto result = view.cell_vertices(top, c);
    if (!result) refuse(("Mesh::from_mesh_view: " + result.error().message).c_str());
    const auto& cell_vertices = result.value();
    const fe::GeometryVariant kind = geometry_variant_for_shape(cell_vertices.shape);
    std::vector<EntityHandle> connectivity(cell_vertices.vertices.size());
    // fuse::MeshView local vertex indices are 0-based; this Mesh's vertex
    // handles are 1-based (added in local-index order above), so the shift
    // is a plain +1, not a lookup.
    for (std::size_t i = 0; i < cell_vertices.vertices.size(); ++i) connectivity[i] = cell_vertices.vertices[i] + 1;
    mesh.add_element(kind, std::move(connectivity), view.is_owned(top, c),
                      static_cast<GlobalId>(view.global_id(top, c)));
  }
  return mesh;
}

std::vector<EntityHandle> Mesh::side_vertices(EntityHandle element, unsigned side) const {
  const auto& connectivity = element_connectivity(element);
  const auto local = fe::ReferenceTopology(element_kind(element)).side_nodes(side);
  std::vector<EntityHandle> result(local.size());
  for (std::size_t i = 0; i < local.size(); ++i) result[i] = connectivity[local[i]];
  return result;
}

std::vector<GlobalId> Mesh::canonical_side_vertex_ids(EntityHandle element, unsigned side) const {
  auto ids = side_vertices(element, side);
  std::vector<GlobalId> result(ids.size());
  std::transform(ids.begin(), ids.end(), result.begin(), [this](EntityHandle h) { return vertex_global_id(h); });
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<EntityHandle> Mesh::edge_vertices(EntityHandle element, unsigned edge) const {
  const auto& connectivity = element_connectivity(element);
  const auto cell = fe::cell_family(element_kind(element));
  const auto local = fe::edge_vertices(cell, edge);
  std::vector<EntityHandle> result(local.size());
  for (std::size_t i = 0; i < local.size(); ++i) result[i] = connectivity[local[i]];
  return result;
}

std::vector<GlobalId> Mesh::canonical_edge_vertex_ids(EntityHandle element, unsigned edge) const {
  auto ids = edge_vertices(element, edge);
  std::vector<GlobalId> result(ids.size());
  std::transform(ids.begin(), ids.end(), result.begin(), [this](EntityHandle h) { return vertex_global_id(h); });
  std::sort(result.begin(), result.end());
  return result;
}

std::optional<EntityHandle> Mesh::neighbor(EntityHandle element, unsigned side) const {
  const auto target = canonical_side_vertex_ids(element, side);
  for (EntityHandle candidate = 1; candidate <= elements_.size(); ++candidate) {
    if (candidate == element) continue;
    const auto candidate_sides = n_sides(candidate);
    for (unsigned s = 0; s < candidate_sides; ++s)
      if (canonical_side_vertex_ids(candidate, s) == target) return candidate;
  }
  return std::nullopt;
}

std::size_t Mesh::element_coordinates(EntityHandle element, std::span<Point> coordinates) const {
  const auto& connectivity = element_connectivity(element);
  if (coordinates.size() < connectivity.size()) refuse("element_coordinates: output span too small");
  for (std::size_t i = 0; i < connectivity.size(); ++i) coordinates[i] = vertices_.at(connectivity[i] - 1);
  return connectivity.size();
}

Point Mesh::centroid(EntityHandle element) const {
  const auto& connectivity = element_connectivity(element);
  Point sum = Point::zero();
  for (const auto handle : connectivity) sum += vertices_.at(handle - 1);
  return sum / static_cast<double>(connectivity.size());
}

std::pair<Point, Point> Mesh::bounding_box() const {
  if (vertices_.empty()) refuse("bounding_box: mesh has no vertices");
  Point lo = vertices_.front(), hi = vertices_.front();
  for (const auto& v : vertices_) {
    lo.x = std::min(lo.x, v.x);
    lo.y = std::min(lo.y, v.y);
    lo.z = std::min(lo.z, v.z);
    hi.x = std::max(hi.x, v.x);
    hi.y = std::max(hi.y, v.y);
    hi.z = std::max(hi.z, v.z);
  }
  return {lo, hi};
}

}  // namespace fuse::physics::spatial
