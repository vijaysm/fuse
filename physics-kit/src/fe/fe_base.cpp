// fuse_physics::fe::FEBase -- adapted from KARMA's karma_fe_base.h/.C.
// See fe_base.hpp's header comment for the mesh/quadrature decoupling and
// this wave's scope.

#include "fuse_physics/fe/fe_base.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

CellFamily side_family(CellFamily parent) {
  switch (parent) {
    case CellFamily::Edge:
      refuse("an edge has no side cell family (its sides are vertices)");
    case CellFamily::Triangle:
    case CellFamily::Quadrilateral:
      return CellFamily::Edge;
    case CellFamily::Tetrahedron:
      return CellFamily::Triangle;
    case CellFamily::Hexahedron:
      return CellFamily::Quadrilateral;
  }
  refuse("unknown cell family");
}

constexpr double condition_tolerance = 256.0 * std::numeric_limits<double>::epsilon();

}  // namespace

FEBase::FEBase(const spatial::Mesh& mesh, GeometryVariant geometry_variant, Basis basis, int quadrature_degree,
               FEIntegrationDomain domain)
    : mesh_(mesh),
      geometry_variant_(geometry_variant),
      geometry_reference_(geometry_variant),
      basis_(std::move(basis)),
      geometry_(geometry_reference_),
      quadrature_(domain == FEIntegrationDomain::Volume ? geometry_reference_.family()
                                                          : side_family(geometry_reference_.family()),
                  quadrature_degree),
      domain_(domain),
      n_basis_(basis_.size()),
      capacity_(quadrature_.n_points()),
      active_qpoints_(quadrature_.n_points()),
      jxw_(capacity_),
      phi_(n_basis_, std::vector<double>(capacity_)),
      dphi_(n_basis_, std::vector<Vector3>(capacity_, Vector3::zero())),
      xyz_(capacity_, Point::zero()),
      normals_(capacity_, Vector3::zero()),
      geometry_nodes_scratch_(geometry_reference_.node_count()),
      reference_phi_(n_basis_),
      reference_gradient_(n_basis_),
      side_parent_points_(domain_ == FEIntegrationDomain::Face ? geometry_reference_.side_count() * capacity_ : 0),
      side_parent_jacobians_(side_parent_points_.size(), Matrix3::zero()),
      physical_point_scratch_(domain_ == FEIntegrationDomain::Face ? capacity_ : 0),
      reference_point_scratch_(physical_point_scratch_.size()) {
  if (cell_family(geometry_variant) != basis_.key().cell)
    refuse("geometry and field basis reference cells must match");

  invalidate();
  if (domain_ == FEIntegrationDomain::Face)
    for (unsigned side = 0; side < geometry_reference_.side_count(); ++side) {
      const auto side_points = quadrature_.points();
      for (std::size_t q = 0; q < side_points.size(); ++q) {
        const std::size_t index = side * capacity_ + q;
        if (geometry_reference_.dimension() == 1) {
          side_parent_points_[index] = Point(static_cast<double>(side), 0, 0);
          continue;
        }
        side_parent_points_[index] = geometry_reference_.side_reference_map(side, side_points[q]);
        side_parent_jacobians_[index] = geometry_reference_.side_reference_jacobian(side, side_points[q]);
      }
    }
}

void FEBase::set_active_size(std::size_t count) {
  if (count > capacity_) refuse("active quadrature count exceeds fixed FE capacity");
  active_qpoints_ = count;
  jxw_.resize(count);
  xyz_.resize(count);
  normals_.resize(count);
  for (auto& row : phi_) row.resize(count);
  for (auto& row : dphi_) row.resize(count);
}

void FEBase::invalidate() noexcept {
  valid_ = false;
  const double invalid = std::numeric_limits<double>::quiet_NaN();
  std::fill(jxw_.begin(), jxw_.end(), invalid);
  for (auto& row : phi_) std::fill(row.begin(), row.end(), invalid);
  for (auto& row : dphi_)
    for (auto& value : row) value = Vector3(invalid, invalid, invalid);
  for (auto& point : xyz_) point = Point(invalid, invalid, invalid);
  for (auto& normal : normals_) normal = Vector3(invalid, invalid, invalid);
}

void FEBase::evaluate_reference_points(const GeometryMap& geometry, std::span<const Point> points,
                                        std::span<const double> weights, bool compute_weights) {
  const unsigned reference_dimension = geometry.reference_dimension();
  for (std::size_t q = 0; q < points.size(); ++q) {
    const Point& natural = points[q];
    basis_.evaluate(natural, reference_phi_, reference_gradient_);

    const Matrix3 jacobian = geometry.jacobian(natural);
    const Matrix3 metric_tensor = geometry.metric(natural);
    const double scale = [&] {
      double m = 0.0;
      for (unsigned r = 0; r < reference_dimension; ++r)
        for (unsigned c = 0; c < reference_dimension; ++c) m = std::max(m, std::abs(metric_tensor(r, c)));
      return m;
    }();
    const double determinant = block_determinant(metric_tensor, reference_dimension);
    if (!std::isfinite(determinant) || scale <= 0.0 ||
        determinant <= condition_tolerance * std::pow(scale, static_cast<int>(reference_dimension)))
      refuse("finite-element Jacobian is rank-deficient or ill-conditioned");
    const Matrix3 metric_inverse = block_inverse(metric_tensor, reference_dimension);
    const Matrix3 gradient_transform = matmul(jacobian, metric_inverse);

    xyz_[q] = geometry.map(natural);
    for (std::size_t i = 0; i < n_basis_; ++i) {
      phi_[i][q] = reference_phi_[i];
      dphi_[i][q] = matvec(gradient_transform, reference_gradient_[i]);
    }
    if (compute_weights) jxw_[q] = weights[q] * geometry.measure(natural);
  }
}

void FEBase::reinit(spatial::EntityHandle element) {
  invalidate();
  try {
    if (domain_ != FEIntegrationDomain::Volume) refuse("volume reinit called on face finite element");
    const std::size_t nconn = mesh_.element_coordinates(element, geometry_nodes_scratch_);
    if (nconn != geometry_reference_.node_count()) refuse("element connectivity does not match bound geometry variant");
    geometry_.reinit(std::span<const Point>(geometry_nodes_scratch_.data(), nconn));
    set_active_size(quadrature_.n_points());
    evaluate_reference_points(geometry_, quadrature_.points(), quadrature_.weights(), true);
    valid_ = true;
  } catch (...) {
    invalidate();
    throw;
  }
}

void FEBase::reinit(spatial::EntityHandle element, unsigned side) {
  invalidate();
  try {
    if (domain_ != FEIntegrationDomain::Face) refuse("face reinit called on volume finite element");
    if (side >= geometry_reference_.side_count()) refuse("finite-element side index is out of range");
    const std::size_t nconn = mesh_.element_coordinates(element, geometry_nodes_scratch_);
    if (nconn != geometry_reference_.node_count()) refuse("element connectivity does not match bound geometry variant");
    geometry_.reinit(std::span<const Point>(geometry_nodes_scratch_.data(), nconn));

    const auto weights = quadrature_.weights();
    set_active_size(quadrature_.n_points());
    invalidate();
    for (std::size_t q = 0; q < active_qpoints_; ++q) {
      const std::size_t side_offset = side * capacity_ + q;
      if (geometry_reference_.dimension() == 1) {
        const Matrix3 jacobian = geometry_.jacobian(side_parent_points_[side_offset]);
        xyz_[q] = geometry_.map(side_parent_points_[side_offset]);
        normals_[q] = (side == 0 ? -1.0 : 1.0) * normalized(jacobian.col(0));
        jxw_[q] = weights[q];
      } else {
        const SideGeometry side_geometry =
            geometry_.side_from_parent(side, side_parent_points_[side_offset], side_parent_jacobians_[side_offset]);
        xyz_[q] = side_geometry.point;
        normals_[q] = side_geometry.normal;
        jxw_[q] = weights[q] * side_geometry.measure;
      }
    }
    std::vector<Point> parent_points(active_qpoints_);
    for (std::size_t q = 0; q < active_qpoints_; ++q) parent_points[q] = side_parent_points_[side * capacity_ + q];
    evaluate_reference_points(geometry_, parent_points, {}, false);
    valid_ = true;
  } catch (...) {
    invalidate();
    throw;
  }
}

void FEBase::reinit_at_physical_points(spatial::EntityHandle element, std::span<const Point> points) {
  if (domain_ != FEIntegrationDomain::Face) refuse("physical-point reinit requires face finite element");
  if (points.empty() || points.size() > capacity_) refuse("physical-point count must be nonzero and within fixed FE capacity");
  physical_point_scratch_.assign(points.begin(), points.end());
  set_active_size(points.size());
  invalidate();
  try {
    const std::size_t nconn = mesh_.element_coordinates(element, geometry_nodes_scratch_);
    if (nconn != geometry_reference_.node_count()) refuse("element connectivity does not match bound geometry variant");
    geometry_.reinit(std::span<const Point>(geometry_nodes_scratch_.data(), nconn));
    for (const auto& point : physical_point_scratch_)
      if (!point.all_finite()) refuse("physical point coordinates must be finite");
    reference_point_scratch_.resize(active_qpoints_);
    for (std::size_t q = 0; q < active_qpoints_; ++q) reference_point_scratch_[q] = geometry_.inverse_map(physical_point_scratch_[q]);
    evaluate_reference_points(geometry_, reference_point_scratch_, {}, false);
    valid_ = true;
  } catch (...) {
    invalidate();
    throw;
  }
}

}  // namespace fuse::physics::fe
