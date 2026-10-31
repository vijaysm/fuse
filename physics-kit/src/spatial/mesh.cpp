// fuse_physics::spatial::Mesh -- native structured Hex8 mesh. See
// mesh.hpp's header comment for what this does and does not port from
// KARMA's karma_mesh.h/.C.

#include "fuse_physics/spatial/mesh.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace fuse::physics::spatial {

namespace {
[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }
}  // namespace

void Mesh::generate_box(unsigned nx, unsigned ny, unsigned nz, const Point& lo, const Point& hi) {
  if (nx == 0 || ny == 0 || nz == 0) refuse("generate_box needs at least one element per axis");
  vertices_.clear();
  elements_.clear();
  const unsigned px = nx + 1, py = ny + 1, pz = nz + 1;
  vertices_.reserve(static_cast<std::size_t>(px) * py * pz);
  const auto vertex_index = [&](unsigned i, unsigned j, unsigned k) -> EntityHandle {
    return static_cast<EntityHandle>(k) * py * px + static_cast<EntityHandle>(j) * px + i + 1;
  };
  for (unsigned k = 0; k < pz; ++k)
    for (unsigned j = 0; j < py; ++j)
      for (unsigned i = 0; i < px; ++i) {
        const double x = lo.x + (hi.x - lo.x) * (static_cast<double>(i) / nx);
        const double y = lo.y + (hi.y - lo.y) * (static_cast<double>(j) / ny);
        const double z = lo.z + (hi.z - lo.z) * (static_cast<double>(k) / nz);
        vertices_.push_back(Point(x, y, z));
      }
  elements_.reserve(static_cast<std::size_t>(nx) * ny * nz);
  // Hex8 reference-node order (reference_topology.cpp's nodes_for): the
  // bottom face (0,0,0)-(1,0,0)-(1,1,0)-(0,1,0), then the matching top
  // face at z+1, each a right-handed loop seen from outside the cell.
  for (unsigned k = 0; k < nz; ++k)
    for (unsigned j = 0; j < ny; ++j)
      for (unsigned i = 0; i < nx; ++i) {
        elements_.push_back({vertex_index(i, j, k), vertex_index(i + 1, j, k), vertex_index(i + 1, j + 1, k),
                              vertex_index(i, j + 1, k), vertex_index(i, j, k + 1), vertex_index(i + 1, j, k + 1),
                              vertex_index(i + 1, j + 1, k + 1), vertex_index(i, j + 1, k + 1)});
      }
}

std::vector<EntityHandle> Mesh::side_vertices(EntityHandle element, unsigned side) const {
  const auto& connectivity = element_connectivity(element);
  const auto local = fe::ReferenceTopology(fe::GeometryVariant::Hex8).side_nodes(side);
  std::vector<EntityHandle> result(local.size());
  for (std::size_t i = 0; i < local.size(); ++i) result[i] = connectivity[local[i]];
  return result;
}

std::vector<GlobalId> Mesh::canonical_side_vertex_ids(EntityHandle element, unsigned side) const {
  auto ids = side_vertices(element, side);
  std::vector<GlobalId> result(ids.size());
  std::transform(ids.begin(), ids.end(), result.begin(), [this](EntityHandle h) { return global_id(h); });
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
