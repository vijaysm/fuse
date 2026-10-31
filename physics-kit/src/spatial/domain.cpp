// fuse_physics::spatial::Domain -- adapted from KARMA's karma_domain.h/.C
// and enums/domain_shape.h. See domain.hpp's header comment for the I6 fix.

#include "fuse_physics/spatial/domain.hpp"

namespace fuse::physics::spatial {

void Domain::check_shape_supported(Shape shape) {
  // The I6 fix: the upstream assert's intended condition (see domain.hpp)
  // rejects BOTH unsupported shapes, not at most one of them.
  if (shape == Shape::Hexagon || shape == Shape::Unstructured)
    throw std::invalid_argument("Domain: only Edge/Square/Cube shapes are supported in this wave");
}

Domain::Domain(unsigned dimension, Shape shape, double length_x, double length_y, double length_z, bool symmetric)
    : dimension_(dimension),
      shape_(shape),
      length_x_(length_x),
      length_y_(length_y),
      length_z_(length_z),
      symmetric_x_(symmetric),
      symmetric_y_(symmetric),
      symmetric_z_(symmetric) {
  check_shape_supported(shape_);
}

Domain::Domain(unsigned dimension, Shape shape, double length_x, double length_y, double length_z,
               bool symmetric_x, bool symmetric_y, bool symmetric_z)
    : dimension_(dimension),
      shape_(shape),
      length_x_(length_x),
      length_y_(length_y),
      length_z_(length_z),
      symmetric_x_(symmetric_x),
      symmetric_y_(symmetric_y),
      symmetric_z_(symmetric_z) {
  check_shape_supported(shape_);
}

}  // namespace fuse::physics::spatial
