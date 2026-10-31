// fuse_physics::spatial::extract_boundary_faces -- adapted from KARMA's
// karma_mesh_extract_face.C. See mesh_utils.hpp's header comment for scope.

#include "fuse_physics/spatial/mesh_utils.hpp"

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

}  // namespace fuse::physics::spatial
