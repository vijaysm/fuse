// fuse_physics::spatial::extract_boundary_faces / extract_edge_at_coordinate
// -- adapted from KARMA's karma_mesh_extract_face.C / karma_mesh_extract_edge.C.
// See mesh_utils.hpp's header comment for scope and the adaptation note.

#include "fuse_physics/spatial/mesh_utils.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace fuse::physics::spatial {

std::vector<SideId> extract_boundary_faces(const Mesh& volume) {
  std::vector<SideId> result;
  for (EntityHandle element = 1; element <= volume.n_elements(); ++element) {
    const auto sides = volume.n_sides(element);
    for (unsigned side = 0; side < sides; ++side)
      if (volume.is_boundary_side(element, side)) result.push_back(volume.side_id(element, side));
  }
  return result;
}

Mesh extract_edge_at_coordinate(const Mesh& surface, unsigned direction, double value, double tolerance) {
  if (surface.dimension() != 2) throw std::invalid_argument("extract_edge_at_coordinate requires a 2-D source mesh");
  if (direction > 1) throw std::invalid_argument("extract_edge_at_coordinate: direction must be 0 (x) or 1 (y)");

  Mesh boundary(1);
  std::map<EntityHandle, EntityHandle> source_to_boundary_vertex;  // dedup by SOURCE vertex handle

  const auto on_line = [&](EntityHandle source_vertex) {
    return std::abs(surface.vertex_positions()[source_vertex - 1](direction) - value) < tolerance;
  };
  const auto boundary_vertex_for = [&](EntityHandle source_vertex) -> EntityHandle {
    const auto found = source_to_boundary_vertex.find(source_vertex);
    if (found != source_to_boundary_vertex.end()) return found->second;
    const EntityHandle added = boundary.add_vertex(surface.vertex_positions()[source_vertex - 1]);
    source_to_boundary_vertex.emplace(source_vertex, added);
    return added;
  };

  // Each interior edge is shared by two elements and would otherwise be
  // collected twice; dedupe by the SOURCE mesh's own (unordered) vertex
  // pair, matching karma_mesh_extract_edge.C's own node-id-based
  // deduplication (its node_map), generalized from vertices to edges.
  std::map<std::pair<EntityHandle, EntityHandle>, bool> seen_edges;
  for (EntityHandle element = 1; element <= surface.n_elements(); ++element) {
    const auto sides = surface.n_sides(element);
    for (unsigned side = 0; side < sides; ++side) {
      const auto vertices = surface.side_vertices(element, side);
      if (vertices.size() != 2) continue;  // a 2-D element's sides are always edges; defensive only
      if (!on_line(vertices[0]) || !on_line(vertices[1])) continue;
      const auto key = vertices[0] < vertices[1] ? std::pair{vertices[0], vertices[1]}
                                                  : std::pair{vertices[1], vertices[0]};
      if (!seen_edges.emplace(key, true).second) continue;
      boundary.add_element(fe::GeometryVariant::Edge2,
                            {boundary_vertex_for(vertices[0]), boundary_vertex_for(vertices[1])});
    }
  }
  return boundary;
}

}  // namespace fuse::physics::spatial
