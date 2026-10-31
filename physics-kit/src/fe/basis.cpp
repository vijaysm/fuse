// fuse_physics::fe::Basis / BasisRegistry -- adapted from KARMA's
// karma_basis.h/.C. See basis.hpp's header comment for the wave's scope.
//
// NOT PORTED THIS WAVE: KARMA's OrientationTransform / apply_orientation /
// apply_orientation_inverse machinery, which lets a basis function's local
// coefficient be re-signed/permuted to agree with a shared edge or face's
// GLOBAL orientation when two elements see it from opposite local vertex
// order. Single-element evaluation (this wave's whole test surface --
// quadrature exactness, basis properties on one reference cell, geometry
// maps on one element) never exercises it; it matters only once elements
// are assembled into a global system, which is explicitly a later wave
// (numerics/linear, "WAVE 2" per the task). Named here and in
// design_notes/open_questions rather than silently dropped.

#include "fuse_physics/fe/basis.hpp"

#include <array>
#include <stdexcept>

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

bool valid_key(const BasisKey& key) noexcept {
  return key.family == FieldBasisFamily::LagrangeNodal || key.order <= maximum_basis_order;
}

GeometryVariant variant_for(CellFamily cell, unsigned order) {
  switch (cell) {
    case CellFamily::Edge:
      return order <= 1 ? GeometryVariant::Edge2 : GeometryVariant::Edge3;
    case CellFamily::Triangle:
      return order <= 1 ? GeometryVariant::Tri3 : GeometryVariant::Tri6;
    case CellFamily::Quadrilateral:
      return order <= 1 ? GeometryVariant::Quad4 : GeometryVariant::Quad9;
    case CellFamily::Tetrahedron:
      return order <= 1 ? GeometryVariant::Tet4 : GeometryVariant::Tet10;
    case CellFamily::Hexahedron:
      return order <= 1 ? GeometryVariant::Hex8 : GeometryVariant::Hex27;
  }
  refuse("unknown cell family");
}

std::string cell_name(CellFamily cell) {
  switch (cell) {
    case CellFamily::Edge:
      return "Edge";
    case CellFamily::Triangle:
      return "Triangle";
    case CellFamily::Quadrilateral:
      return "Quadrilateral";
    case CellFamily::Tetrahedron:
      return "Tetrahedron";
    case CellFamily::Hexahedron:
      return "Hexahedron";
  }
  return "Unknown";
}

std::string continuity_name(BasisContinuity continuity) {
  if (continuity == BasisContinuity::cG) return "cG";
  if (continuity == BasisContinuity::dG) return "dG";
  return "SE";
}

/// Lagrange nodal basis of order 0 (cell-local constant), 1 or 2, on one of
/// this wave's five cell families. Order 1/2 shape values and gradients are
/// delegated to ReferenceTopology's own analytic evaluation (the Lagrange
/// nodal basis on a simplex/tensor cell IS the reference shape-function
/// set by construction); order 0 is a single constant function.
class BuiltinBasis {
 public:
  explicit BuiltinBasis(BasisKey key)
      : key_(key), reference_(variant_for(key.cell, key.order == 0 ? 1 : key.order)) {
    if (key.order > 2) refuse("built-in basis order must be P0, P1, or P2");
  }

  [[nodiscard]] std::string name() const {
    return "LagrangeNodal " + cell_name(key_.cell) + " P" + std::to_string(key_.order) + " " +
           continuity_name(key_.continuity);
  }
  [[nodiscard]] BasisKey key() const noexcept { return key_; }
  [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept {
    return CoefficientRepresentation::Nodal;
  }
  [[nodiscard]] std::size_t size() const noexcept { return key_.order == 0 ? 1 : reference_.node_count(); }

  void evaluate_values(const Point& point, std::span<double> values) const {
    if (values.size() != size()) refuse("basis output storage must equal basis size");
    if (key_.order == 0) {
      values[0] = 1.0;
      return;
    }
    reference_.evaluate_values(point, values);
  }
  void evaluate(const Point& point, std::span<double> values, std::span<Vector3> gradients) const {
    if (values.size() != size() || gradients.size() != size())
      refuse("basis output storage must equal basis size");
    if (key_.order == 0) {
      values[0] = 1.0;
      gradients[0] = Vector3::zero();
      return;
    }
    reference_.evaluate(point, values, gradients);
  }
  [[nodiscard]] std::vector<Point> interpolation_points() const {
    if (key_.order == 0) return {ReferenceTopology(variant_for(key_.cell, 1)).center()};
    return ReferenceTopology(variant_for(key_.cell, key_.order)).reference_nodes();
  }
  [[nodiscard]] std::vector<DofAssociation> dof_associations() const {
    if (key_.continuity != BasisContinuity::cG) {
      std::vector<DofAssociation> result(size());
      for (std::size_t i = 0; i < result.size(); ++i) result[i] = {cell_dimension_of(key_.cell), 0, static_cast<unsigned>(i)};
      return result;
    }
    if (key_.order == 0) return {{cell_dimension_of(key_.cell), 0, 0}};
    const std::size_t vertex_count = ReferenceTopology(variant_for(key_.cell, 1)).node_count();
    std::vector<DofAssociation> result;
    for (std::size_t i = 0; i < vertex_count; ++i) result.push_back({0, static_cast<unsigned>(i), 0});
    if (key_.order == 1) return result;

    constexpr std::array<unsigned, 5> edge_count{1, 3, 4, 6, 12};  // Edge, Tri, Quad, Tet, Hex
    const unsigned edges = edge_count[static_cast<std::size_t>(key_.cell)];
    for (unsigned i = 0; i < edges; ++i) result.push_back({1, i, 0});
    const unsigned face_centers = key_.cell == CellFamily::Hexahedron ? 6 : 0;
    for (unsigned i = 0; i < face_centers; ++i) result.push_back({2, i, 0});
    if (result.size() < size()) result.push_back({cell_dimension_of(key_.cell), 0, 0});
    return result;
  }

 private:
  static unsigned cell_dimension_of(CellFamily cell) { return cell_dimension(cell); }

  BasisKey key_;
  ReferenceTopology reference_;
};

}  // namespace

BasisRegistry::BasisRegistry() {
  constexpr std::array cells{CellFamily::Edge, CellFamily::Triangle, CellFamily::Quadrilateral,
                              CellFamily::Tetrahedron, CellFamily::Hexahedron};
  constexpr std::array continuities{BasisContinuity::cG, BasisContinuity::dG, BasisContinuity::SE};
  for (const auto cell : cells)
    for (unsigned order = 0; order <= 2; ++order)
      for (const auto continuity : continuities) {
        // A globally continuous elementwise constant space is not a cG cell basis.
        if (order == 0 && continuity == BasisContinuity::cG) continue;
        const BasisKey key{cell, FieldBasisFamily::LagrangeNodal, order, continuity};
        factories_.emplace(key, [key] { return Basis(BuiltinBasis(key)); });
      }
}

void BasisRegistry::register_factory(const BasisKey& key, Factory factory, DuplicatePolicy policy) {
  if (!valid_key(key)) refuse("invalid basis key");
  if (!factory) refuse("basis factory must be callable");
  const auto found = factories_.find(key);
  if (found != factories_.end() && policy == DuplicatePolicy::Reject) refuse("basis key is already registered");
  factories_.insert_or_assign(key, std::move(factory));
}

bool BasisRegistry::supports(const BasisKey& key) const noexcept {
  return valid_key(key) && factories_.contains(key);
}

Basis BasisRegistry::create(const BasisKey& key) const {
  if (!valid_key(key)) refuse("invalid basis key");
  const auto found = factories_.find(key);
  if (found == factories_.end()) refuse("unknown basis key");
  Basis result = found->second();
  if (!(result.key() == key)) refuse("basis factory returned a mismatched key");
  return result;
}

}  // namespace fuse::physics::fe
