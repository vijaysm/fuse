// fuse_physics::fe::BoundaryConditionManager -- see boundary_condition.hpp's
// own header comment for how physics-kit's geometric boundary identity and
// single-rank scope diverge from KARMA's own MOAB side-set lookup.

#include "fuse_physics/fe/boundary_condition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

#include "fuse_physics/fe/geometry_map.hpp"
#include "fuse_physics/fe/reference_topology.hpp"

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

/// The one axis and side (lo = false means the MIN face, true the MAX
/// face) each BoundaryId names.
struct AxisFace {
  int axis;      // 0 = x, 1 = y, 2 = z
  bool at_max;   // false: position's own coordinate must equal lo's; true: must equal lo's plus the axis length
};

AxisFace axis_face_of(BoundaryId id) {
  switch (id) {
    case BoundaryId::XMin: return {0, false};
    case BoundaryId::XMax: return {0, true};
    case BoundaryId::YMin: return {1, false};
    case BoundaryId::YMax: return {1, true};
    case BoundaryId::ZMin: return {2, false};
    case BoundaryId::ZMax: return {2, true};
  }
  refuse("BoundaryConditionManager: unreachable BoundaryId");
}

double axis_coordinate(const Point& p, int axis) {
  switch (axis) {
    case 0: return p.x;
    case 1: return p.y;
    default: return p.z;
  }
}

bool boundary_requires_dimension(BoundaryId id, unsigned dimension) {
  const auto face = axis_face_of(id);
  return static_cast<unsigned>(face.axis) < dimension;
}

const char* boundary_id_name(BoundaryId id) {
  switch (id) {
    case BoundaryId::XMin: return "XMin";
    case BoundaryId::XMax: return "XMax";
    case BoundaryId::YMin: return "YMin";
    case BoundaryId::YMax: return "YMax";
    case BoundaryId::ZMin: return "ZMin";
    case BoundaryId::ZMax: return "ZMax";
  }
  return "?";
}

/// The lower corner of `mesh`'s own vertex bounding box: the
/// componentwise minimum over every vertex, the measured box origin a
/// mesh generated from a non-zero corner (generate_box/generate_2d_box
/// both take an explicit lo point) still needs.
Point measured_lower_corner(const spatial::Mesh& mesh) {
  const auto& positions = mesh.vertex_positions();
  if (positions.empty()) refuse("BoundaryConditionManager: mesh has no vertices to measure a bounding box from");
  Point lo = positions.front();
  for (const auto& p : positions) {
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
  }
  return lo;
}

}  // namespace

std::vector<BoundaryId> classify_position(const Point& position, const Point& lo, double length_x, double length_y,
                                           double length_z, unsigned dimension, double tolerance) {
  std::vector<BoundaryId> result;
  constexpr std::array<BoundaryId, 6> kAll = {BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin,
                                               BoundaryId::YMax, BoundaryId::ZMin, BoundaryId::ZMax};
  const std::array<double, 3> length = {length_x, length_y, length_z};
  for (const auto id : kAll) {
    if (!boundary_requires_dimension(id, dimension)) continue;
    const auto face = axis_face_of(id);
    const double coordinate = axis_coordinate(position, face.axis);
    const double lo_coordinate = axis_coordinate(lo, face.axis);
    const double target = face.at_max ? lo_coordinate + length[static_cast<std::size_t>(face.axis)] : lo_coordinate;
    if (std::fabs(coordinate - target) <= tolerance) result.push_back(id);
  }
  return result;
}

void dof_physical_positions(const spatial::Mesh& mesh, const DofMap& dof_map, unsigned variable, const Basis& basis,
                             std::vector<Point>& out) {
  out.assign(static_cast<std::size_t>(dof_map.n_dofs()), Point{});
  const auto& interpolation_points = basis.interpolation_points();
  std::vector<std::int64_t> indices;
  for (spatial::EntityHandle e = 1; e <= mesh.n_elements(); ++e) {
    dof_map.dof_indices(e, indices, variable);
    const auto& connectivity = mesh.element_connectivity(e);
    const auto& positions = mesh.vertex_positions();
    std::vector<Point> element_vertices;
    element_vertices.reserve(connectivity.size());
    for (const auto v : connectivity) element_vertices.push_back(positions.at(v - 1));
    const GeometryMap gmap(ReferenceTopology(mesh.element_kind(e)), element_vertices);
    for (std::size_t k = 0; k < indices.size() && k < interpolation_points.size(); ++k)
      out[static_cast<std::size_t>(indices[k])] = gmap.map(interpolation_points[k]);
  }
}

BoundaryConditionManager::BoundaryConditionManager(const spatial::Mesh& mesh, const spatial::Domain& domain)
    : mesh_(&mesh), domain_(&domain), lo_(measured_lower_corner(mesh)) {
  // The distance within which a coordinate counts as lying on a face. With
  // u the unit roundoff, lo and L the lower corner and length of one axis
  // and M = max(|lo|, |lo + L|): generate_box places the upper face at
  // fl(lo + fl(hi - lo)) and the face is looked for at fl(lo + L). When
  // hi is lo + L to within one rounding, |fl(hi - lo) - L| <= u (M + L),
  // and each of the two sums rounds by at most u M, so the two coordinates
  // differ by at most u (3 M + L) <= 4 u max(M, L) = 2 eps max(M, L). The
  // largest such value over the axes is used. It scales with the
  // coordinates, not only with the lengths: a box far from the origin
  // carries rounding in proportion to its distance from it.
  double scale = 0.0;
  const std::array<double, 3> lower = {lo_.x, lo_.y, lo_.z};
  const std::array<double, 3> length = {domain.length_x(), domain.length_y(), domain.length_z()};
  for (unsigned axis = 0; axis < domain.dimension(); ++axis)
    scale = std::max({scale, std::fabs(lower[axis]), std::fabs(lower[axis] + length[axis]), std::fabs(length[axis])});
  tolerance_ = 2.0 * std::numeric_limits<double>::epsilon() * scale;
}

void BoundaryConditionManager::set(BoundaryId boundary_id, BoundaryCondition condition) {
  if (!boundary_requires_dimension(boundary_id, domain_->dimension()))
    refuse(
        "BoundaryConditionManager::set: this boundary id names an axis beyond the domain's own "
        "dimension (E-PHYSICS-BC-UNKNOWN-BOUNDARY)");
  if (condition.kind == BoundaryConditionKind::Robin && !(condition.robin_coefficient > 0.0))
    refuse(
        "BoundaryConditionManager::set: a Robin condition's own coefficient must be strictly "
        "positive for the discrete operator to stay coercive (E-PHYSICS-BC-ROBIN-COEFFICIENT)");
  if (conditions_.find(boundary_id) != conditions_.end())
    refuse(
        "BoundaryConditionManager::set: this boundary id already carries a condition; two "
        "conditions on one boundary are not supported (E-PHYSICS-BC-DUPLICATE)");
  conditions_.emplace(boundary_id, std::move(condition));
}

void BoundaryConditionManager::resolve(const DofMap& dof_map, const Basis& basis, unsigned variable) {
  if (dof_map.row_lo_by_rank().size() > 1)
    refuse(
        "BoundaryConditionManager::resolve: this DofMap spans more than one rank; boundary "
        "resolution walks the whole mesh locally and is single-rank only "
        "(E-PHYSICS-BC-MPI-NOT-SUPPORTED)");
  if (conditions_.empty())
    refuse("BoundaryConditionManager::resolve: no boundary condition was ever registered (E-PHYSICS-BC-NO-CONDITIONS)");

  dirichlet_.clear();
  neumann_.clear();
  robin_.clear();

  std::vector<Point> dof_position;
  dof_physical_positions(*mesh_, dof_map, variable, basis, dof_position);

  const auto& positions = mesh_->vertex_positions();
  std::set<BoundaryId> matched_ids;
  std::set<std::int64_t> seen_dirichlet_dofs;
  std::vector<std::int64_t> indices;
  std::vector<unsigned> local_indices;

  for (spatial::EntityHandle e = 1; e <= mesh_->n_elements(); ++e) {
    const unsigned n_sides = mesh_->n_sides(e);
    for (unsigned side = 0; side < n_sides; ++side) {
      if (!mesh_->is_boundary_side(e, side)) continue;
      const auto side_vertices = mesh_->side_vertices(e, side);
      // Every corner of this side must classify to the SAME boundary id
      // for the side to belong to it -- an axis-aligned box mesh's own
      // face never straddles two box faces at once (classify_position's
      // own header comment), so intersecting every corner's own
      // candidate set leaves at most one id. Side identity is purely
      // geometric (the mesh's own corner vertices), independent of the
      // field basis's own order.
      std::vector<BoundaryId> common;
      for (std::size_t k = 0; k < side_vertices.size(); ++k) {
        const Point& p = positions.at(side_vertices[k] - 1);
        const auto matches = classify_position(p, lo_, domain_->length_x(), domain_->length_y(), domain_->length_z(),
                                                domain_->dimension(), tolerance_);
        if (k == 0) {
          common = matches;
        } else {
          std::vector<BoundaryId> next;
          for (const auto id : common)
            if (std::find(matches.begin(), matches.end(), id) != matches.end()) next.push_back(id);
          common = std::move(next);
        }
      }
      if (common.empty()) continue;  // an exterior side with no registered condition at all
      const auto found = conditions_.find(common.front());
      if (found == conditions_.end()) continue;
      matched_ids.insert(common.front());

      if (found->second.kind == BoundaryConditionKind::Dirichlet) {
        dof_map.dof_indices(e, indices, variable);
        dof_map.dofs_on_side(e, side, variable, basis, local_indices);
        for (const unsigned local_index : local_indices) {
          const std::int64_t dof = indices.at(local_index);
          if (!seen_dirichlet_dofs.insert(dof).second) continue;  // already resolved via an earlier side
          dirichlet_.push_back(ResolvedDirichlet{dof, dof_position.at(static_cast<std::size_t>(dof)), &found->second});
        }
      } else if (found->second.kind == BoundaryConditionKind::Neumann) {
        neumann_.push_back(ResolvedFace{e, side, &found->second});
      } else if (found->second.kind == BoundaryConditionKind::Robin) {
        robin_.push_back(ResolvedFace{e, side, &found->second});
      }
    }
  }

  for (const auto& [id, condition] : conditions_) {
    (void)condition;
    if (matched_ids.find(id) == matched_ids.end())
      refuse((std::string("BoundaryConditionManager::resolve: boundary ") + boundary_id_name(id) +
              " matched no side of this mesh (E-PHYSICS-BC-UNMATCHED)")
                 .c_str());
  }
}

}  // namespace fuse::physics::fe
