// fuse_physics::fe::GeometryMap -- adapted from KARMA's
// karma_geometry_map.h/.C. See geometry_map.hpp's header comment for scope.

#include "fuse_physics/fe/geometry_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

constexpr double scaled_tolerance = 256.0 * std::numeric_limits<double>::epsilon();

bool in_reference_cell(const ReferenceTopology& reference, const Point& point, double tolerance) {
  for (unsigned d = 0; d < reference.dimension(); ++d)
    if (point(d) < -tolerance || point(d) > 1 + tolerance) return false;
  if (reference.family() == CellFamily::Triangle || reference.family() == CellFamily::Tetrahedron) {
    const double sum = point(0) + point(1) + (reference.family() == CellFamily::Tetrahedron ? point(2) : 0.0);
    if (sum > 1 + tolerance) return false;
  }
  return true;
}

/// Outward reference-space co-normal for a side, used only to fix the sign
/// of the computed physical normal (the KARMA source's own
/// reference_outward_normal table, restricted to this wave's five cell
/// families).
Point reference_outward_normal(CellFamily family, unsigned side) {
  switch (family) {
    case CellFamily::Triangle:
      return std::array{Point(0, -1, 0), Point(1, 1, 0), Point(-1, 0, 0)}[side];
    case CellFamily::Quadrilateral:
      return std::array{Point(0, -1, 0), Point(1, 0, 0), Point(0, 1, 0), Point(-1, 0, 0)}[side];
    case CellFamily::Tetrahedron:
      return std::array{Point(0, -1, 0), Point(1, 1, 1), Point(-1, 0, 0), Point(0, 0, -1)}[side];
    case CellFamily::Hexahedron:
      return std::array{Point(0, -1, 0), Point(1, 0, 0),  Point(0, 1, 0),
                         Point(-1, 0, 0), Point(0, 0, -1), Point(0, 0, 1)}[side];
    case CellFamily::Edge:
      refuse("an edge side has no 3-D reference co-normal");
  }
  refuse("unsupported family for side normal");
}

}  // namespace

GeometryMap::GeometryMap(ReferenceTopology reference, std::span<const Point> nodes)
    : reference_(std::move(reference)) {
  reinit(nodes);
}

GeometryMap::GeometryMap(ReferenceTopology reference) : reference_(std::move(reference)) {}

void GeometryMap::reinit(std::span<const Point> nodes) {
  if (nodes.size() != reference_.node_count()) refuse("geometry node count does not match variant");
  std::copy(nodes.begin(), nodes.end(), nodes_.begin());
  node_count_ = nodes.size();
  update_affine_quad4_cache();
  validate();
}

void GeometryMap::update_affine_quad4_cache() {
  has_affine_quad4_cache_ = false;
  if (reference_.variant() != GeometryVariant::Quad4 ||
      !(nodes_[0] - nodes_[1] + nodes_[2] - nodes_[3] == Point::zero()))
    return;
  affine_quad4_origin_ = nodes_[0];
  affine_quad4_r_ = nodes_[1] - affine_quad4_origin_;
  affine_quad4_s_ = nodes_[3] - affine_quad4_origin_;
  affine_quad4_jacobian_ = Matrix3::zero();
  affine_quad4_jacobian_.set_col(0, affine_quad4_r_);
  affine_quad4_jacobian_.set_col(1, affine_quad4_s_);
  affine_quad4_metric_ = Matrix3::zero();
  affine_quad4_metric_(0, 0) = dot(affine_quad4_r_, affine_quad4_r_);
  affine_quad4_metric_(0, 1) = dot(affine_quad4_r_, affine_quad4_s_);
  affine_quad4_metric_(1, 0) = affine_quad4_metric_(0, 1);
  affine_quad4_metric_(1, 1) = dot(affine_quad4_s_, affine_quad4_s_);
  affine_quad4_measure_ = norm(cross(affine_quad4_r_, affine_quad4_s_));
  has_affine_quad4_cache_ = true;
}

Point GeometryMap::map(const Point& reference_point) const {
  if (has_affine_quad4_cache_)
    return affine_quad4_origin_ + reference_point(0) * affine_quad4_r_ + reference_point(1) * affine_quad4_s_;
  reference_.evaluate_values(reference_point, std::span<double>(values_.data(), node_count_));
  Point result = Point::zero();
  for (std::size_t i = 0; i < node_count_; ++i) result += values_[i] * nodes_[i];
  return result;
}

Matrix3 GeometryMap::jacobian(const Point& reference_point) const {
  if (has_affine_quad4_cache_) return affine_quad4_jacobian_;
  reference_.evaluate(reference_point, std::span<double>(values_.data(), node_count_),
                       std::span<Point>(derivatives_.data(), node_count_));
  Matrix3 result = Matrix3::zero();
  for (std::size_t i = 0; i < node_count_; ++i)
    for (unsigned d = 0; d < reference_.dimension(); ++d)
      result.set_col(d, result.col(d) + derivatives_[i](d) * nodes_[i]);
  return result;
}

Matrix3 GeometryMap::metric(const Point& reference_point) const {
  if (has_affine_quad4_cache_) return affine_quad4_metric_;
  const Matrix3 j = jacobian(reference_point);
  return matmul(transpose(j), j);
}

double GeometryMap::measure(const Point& reference_point) const {
  if (has_affine_quad4_cache_) return affine_quad4_measure_;
  const Matrix3 j = jacobian(reference_point);
  if (reference_.dimension() == 1) return norm(j.col(0));
  if (reference_.dimension() == 2) return norm(cross(j.col(0), j.col(1)));
  return block_determinant(j, 3);
}

InverseMapResult GeometryMap::try_inverse_map(const Point& physical_point, double reference_tolerance,
                                                double residual_tolerance) const noexcept {
  try {
    if (!physical_point.all_finite() || reference_tolerance < 0.0 || residual_tolerance < 0.0)
      return {InverseMapStatus::InvalidGeometry, Point::zero(), std::numeric_limits<double>::infinity()};
    Point lower = nodes_[0];
    Point upper = nodes_[0];
    for (std::size_t node = 1; node < node_count_; ++node) {
      lower.x = std::min(lower.x, nodes_[node].x);
      lower.y = std::min(lower.y, nodes_[node].y);
      lower.z = std::min(lower.z, nodes_[node].z);
      upper.x = std::max(upper.x, nodes_[node].x);
      upper.y = std::max(upper.y, nodes_[node].y);
      upper.z = std::max(upper.z, nodes_[node].z);
    }
    const double box_tolerance = std::max(reference_tolerance, residual_tolerance);
    for (unsigned d = 0; d < 3; ++d)
      if (physical_point(d) < lower(d) - box_tolerance || physical_point(d) > upper(d) + box_tolerance)
        return {InverseMapStatus::NotContained, Point::zero(), std::numeric_limits<double>::infinity()};
    const Point reference = inverse_map(physical_point);
    if (!in_reference_cell(reference_, reference, reference_tolerance))
      return {InverseMapStatus::NotContained, reference, norm(map(reference) - physical_point)};
    const double residual = norm(map(reference) - physical_point);
    if (!std::isfinite(residual) || residual > residual_tolerance)
      return {InverseMapStatus::NotContained, reference, residual};
    if (!std::isfinite(measure(reference)) || measure(reference) <= 0.0)
      return {InverseMapStatus::InvalidGeometry, reference, residual};
    return {InverseMapStatus::Located, reference, residual};
  } catch (...) {
    return {InverseMapStatus::InvalidGeometry, Point::zero(), std::numeric_limits<double>::infinity()};
  }
}

Point GeometryMap::inverse_map(const Point& physical_point) const {
  if (!physical_point.all_finite()) refuse("inverse-map point must be finite");
  double scale = 0.0;
  for (std::size_t i = 0; i < node_count_; ++i)
    for (std::size_t j = i + 1; j < node_count_; ++j) scale = std::max(scale, distance(nodes_[i], nodes_[j]));

  Point reference = reference_.center();
  constexpr int maximum_iterations = 40;
  const double tolerance = 2.e-12;
  for (int iteration = 0; iteration < maximum_iterations; ++iteration) {
    const Point residual = map(reference) - physical_point;
    if (norm(residual) <= tolerance * scale) return reference;
    const Matrix3 j = jacobian(reference);
    const unsigned dim = reference_.dimension();
    const Matrix3 metric_tensor = matmul(transpose(j), j);
    const double metric_scale = [&] {
      double m = 0.0;
      for (unsigned r = 0; r < dim; ++r)
        for (unsigned c = 0; c < dim; ++c) m = std::max(m, std::abs(metric_tensor(r, c)));
      return m;
    }();
    const double determinant = block_determinant(metric_tensor, dim);
    if (!std::isfinite(determinant) || metric_scale <= 0.0 ||
        determinant <= scaled_tolerance * std::pow(metric_scale, static_cast<int>(dim)))
      refuse("inverse map encountered a rank-deficient or ill-conditioned Jacobian");
    const Matrix3 inverse = block_inverse(metric_tensor, dim);
    const Point correction = matvec(inverse, matvec(transpose(j), residual));
    if (!correction.all_finite()) refuse("inverse map correction is nonfinite");

    double damping = 1.0;
    Point candidate = reference;
    bool accepted = false;
    while (damping >= 1.0 / 1024.0) {
      candidate = reference;
      for (unsigned d = 0; d < dim; ++d) candidate(d) -= damping * correction(d);
      if (in_reference_cell(reference_, candidate, 0.0) &&
          dot(map(candidate) - physical_point, map(candidate) - physical_point) <
              dot(residual, residual)) {
        accepted = true;
        break;
      }
      damping *= 0.5;
    }
    if (!accepted) refuse("inverse map line search failed");
    reference = candidate;
  }
  refuse("inverse map failed to converge");
}

SideGeometry GeometryMap::side(unsigned side_index, const Point& side_point) const {
  if (side_index >= reference_.side_count()) refuse("side index is out of range");
  return side_from_parent(side_index, reference_.side_reference_map(side_index, side_point),
                           reference_.side_reference_jacobian(side_index, side_point));
}

SideGeometry GeometryMap::side_from_parent(unsigned side_index, const Point& parent, const Matrix3& parent_jacobian) const {
  if (side_index >= reference_.side_count()) refuse("side index is out of range");
  const Matrix3 physical_jacobian = jacobian(parent);
  SideGeometry result;
  result.point = map(parent);
  result.tangents = matmul(physical_jacobian, parent_jacobian);
  const unsigned dim = reference_.dimension();
  if (dim == 2) {
    const Vector3 tangent = result.tangents.col(0);
    const Vector3 surface_normal = normalized(cross(physical_jacobian.col(0), physical_jacobian.col(1)));
    result.normal = cross(tangent, surface_normal);
    result.measure = norm(result.normal);
  } else if (dim == 3) {
    result.normal = cross(result.tangents.col(0), result.tangents.col(1));
    result.measure = norm(result.normal);
  } else {
    refuse("zero-dimensional edge endpoints have no unique spatial normal");
  }
  double side_scale = 0.0;
  for (std::size_t i = 0; i < node_count_; ++i)
    for (std::size_t j = i + 1; j < node_count_; ++j) side_scale = std::max(side_scale, distance(nodes_[i], nodes_[j]));
  const double side_limit = scaled_tolerance * std::pow(side_scale, static_cast<int>(dim - 1));
  if (!std::isfinite(result.measure) || side_scale <= 0.0 || result.measure <= side_limit)
    refuse("side mapping is degenerate");
  result.normal = result.normal / result.measure;

  const Matrix3 metric_tensor = matmul(transpose(physical_jacobian), physical_jacobian);
  const Matrix3 inverse = block_inverse(metric_tensor, dim);
  const Vector3 reference_covector = reference_outward_normal(reference_.family(), side_index);
  const Vector3 canonical = matvec(physical_jacobian, matvec(inverse, reference_covector));
  if (!canonical.all_finite() || norm(canonical) == 0.0) refuse("side orientation produced a nonfinite canonical co-normal");
  if (dot(result.normal, canonical) < 0.0) result.normal = -result.normal;
  return result;
}

void GeometryMap::validate() const {
  if (node_count_ != reference_.node_count()) refuse("geometry node count does not match variant");
  for (std::size_t i = 0; i < node_count_; ++i)
    if (!nodes_[i].all_finite()) refuse("geometry nodes must be finite");
  double scale = 0.0;
  for (std::size_t i = 0; i < node_count_; ++i)
    for (std::size_t j = i + 1; j < node_count_; ++j) scale = std::max(scale, distance(nodes_[i], nodes_[j]));
  if (!std::isfinite(scale) || scale <= 0.0) refuse("geometry has zero or nonfinite scale");

  if (reference_.variant() == GeometryVariant::Edge3) {
    // x'(r) = a*r+b. Minimize |x'(r)|^2 exactly on [0,1] (same closed form
    // the KARMA source derives inline for this one case).
    const Vector3 a = 4.0 * nodes_[0] + 4.0 * nodes_[1] - 8.0 * nodes_[2];
    const Vector3 b = -3.0 * nodes_[0] - nodes_[1] + 4.0 * nodes_[2];
    const double a2 = dot(a, a);
    const double stationary = a2 == 0.0 ? 0.0 : std::clamp(-dot(a, b) / a2, 0.0, 1.0);
    const double minimum_speed = std::min({norm(b), norm(a + b), norm(stationary * a + b)});
    if (!std::isfinite(minimum_speed) || minimum_speed <= scaled_tolerance * scale)
      refuse("EDGE3 geometry has zero or ill-conditioned derivative");
    return;
  }

  // Recursively sample each parameter cell at corners, edge/face midpoints
  // and center; refine through depth 3 where the metric varies sharply.
  // Not a formal global-injectivity certificate (same caveat KARMA's own
  // comment states).
  const double limit = scaled_tolerance * std::pow(scale, static_cast<int>(reference_.dimension()));
  Vector3 surface_orientation = Vector3::zero();
  Matrix3 center_jacobian = Matrix3::zero();
  if (reference_.dimension() == 2) {
    center_jacobian = jacobian(reference_.center());
    surface_orientation = cross(center_jacobian.col(0), center_jacobian.col(1));
    if (!surface_orientation.all_finite() || norm(surface_orientation) <= limit)
      refuse("surface orientation anchor is degenerate");
    surface_orientation = normalized(surface_orientation);
  }
  if (has_affine_quad4_cache_) {
    const auto bounds = symmetric_block_eigenvalue_bounds(affine_quad4_metric_, 2);
    const double value = dot(cross(center_jacobian.col(0), center_jacobian.col(1)), surface_orientation);
    if (!std::isfinite(value) || value <= limit || !std::isfinite(bounds.second) || !std::isfinite(bounds.first) ||
        bounds.first <= scaled_tolerance * scaled_tolerance * bounds.second)
      refuse("geometry is degenerate, inverted, or ill-conditioned at a validation point");
    return;
  }
  const auto checked_geometry = [&](const Point& point) -> std::pair<double, double> {
    if (!in_reference_cell(reference_, point, 0.0)) return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    const Matrix3 j = jacobian(point);
    const Matrix3 metric_tensor = matmul(transpose(j), j);
    const unsigned dim = reference_.dimension();
    const auto [smallest, largest] = symmetric_block_eigenvalue_bounds(metric_tensor, dim);
    double value = std::sqrt(block_determinant(metric_tensor, dim));
    if (dim == 2) value = dot(cross(j.col(0), j.col(1)), surface_orientation);
    else if (dim == 3) value = block_determinant(j, 3);
    if (!std::isfinite(value) || value <= limit || !std::isfinite(largest) || smallest <= scaled_tolerance * scaled_tolerance * largest)
      refuse("geometry is degenerate, inverted, or ill-conditioned at a validation point");
    return {value, std::sqrt(largest / smallest)};
  };
  constexpr unsigned initial_subdivisions = 8;
  constexpr unsigned maximum_depth = 3;
  const std::function<void(const Point&, const Point&, unsigned)> inspect = [&](const Point& low, const Point& high,
                                                                                 unsigned depth) {
    double minimum = std::numeric_limits<double>::max(), maximum = 0.0, maximum_condition = 1.0;
    unsigned valid_count = 0;
    const Point middle = .5 * (low + high);
    unsigned sample_count = 1;
    for (unsigned d = 0; d < reference_.dimension(); ++d) sample_count *= 3;
    for (unsigned code = 0; code < sample_count; ++code) {
      Point sample = low;
      unsigned digits = code;
      for (unsigned d = 0; d < reference_.dimension(); ++d) {
        const unsigned location = digits % 3;
        digits /= 3;
        sample(d) = location == 0 ? low(d) : location == 1 ? middle(d) : high(d);
      }
      const auto [value, condition] = checked_geometry(sample);
      if (std::isfinite(value)) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        maximum_condition = std::max(maximum_condition, condition);
        ++valid_count;
      }
    }
    if (depth == maximum_depth || valid_count == 0 || (maximum <= 4.0 * minimum && maximum_condition <= 16.0)) return;
    const unsigned combinations = 1u << reference_.dimension();
    for (unsigned mask = 0; mask < combinations; ++mask) {
      Point child_low = low, child_high = high;
      for (unsigned d = 0; d < reference_.dimension(); ++d) {
        if (mask & (1u << d)) child_low(d) = middle(d);
        else child_high(d) = middle(d);
      }
      inspect(child_low, child_high, depth + 1);
    }
  };
  for (unsigned k = 0; k < (reference_.dimension() == 3 ? initial_subdivisions : 1); ++k)
    for (unsigned j = 0; j < (reference_.dimension() >= 2 ? initial_subdivisions : 1); ++j)
      for (unsigned i = 0; i < initial_subdivisions; ++i) {
        Point low(i / 8.0, j / 8.0, k / 8.0);
        Point high((i + 1) / 8.0, reference_.dimension() >= 2 ? (j + 1) / 8.0 : 0.0,
                   reference_.dimension() == 3 ? (k + 1) / 8.0 : 0.0);
        inspect(low, high, 0);
      }
}

}  // namespace fuse::physics::fe
