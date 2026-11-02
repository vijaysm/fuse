// fuse_physics::fe::Basis / BasisRegistry -- adapted from KARMA's
// karma_basis.h/.C. See basis.hpp's header comment for the scope.
//
// ORIENTATION: KARMA's
// OrientationTransform / apply_orientation / apply_orientation_inverse,
// letting a basis function's local DOF be re-signed/permuted to agree
// with a shared edge or face's GLOBAL orientation when two elements see
// it from opposite local vertex order. KARMA determines each sub-entity's
// own shape (to size orientation_count: 2 for an edge, 6 for a triangular
// face, 8 for a quadrilateral face) via moab::CN::SubEntityType;
// physics-kit has no MOAB dependency (docs/dev/layering.md), so
// subentity_shape() below reads the SAME fact off
// ReferenceTopology::side_topology(side).family() instead -- the two are
// the same classification (reference_topology.cpp's cell
// corner/side ordering agrees with MOAB/Exodus convention
// exactly), just read through a native accessor
// rather than a MOAB one.
//
// The built-in bases (order <= 2) place at most ONE DOF on any
// edge or face, so every generated OrientationTransform is PROVABLY the
// identity permutation with sign +1 (basis.hpp's own header comment) --
// ported with KARMA's identical general-purpose construction (it does
// not special-case the DOF count; a future higher-order basis gets a
// genuine permutation for free), with the IDENTITY result confirmed by
// fe_basis_test.cpp rather than assumed.

#include "fuse_physics/fe/basis.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

bool valid_key(const BasisKey& key) noexcept {
  // The order bound applies uniformly to every family. An earlier
  // `key.family == LagrangeNodal ||` clause accepted ANY key.order for a
  // LagrangeNodal key, regardless of maximum_basis_order, because the
  // family clause alone made the whole disjunction true.
  return key.order <= maximum_basis_order;
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
    case CellFamily::Prism:
      // The COMPLETE family (Prism18), not the Quad8-like serendipity
      // Prism15 -- same principle basis.hpp's header states for Quad9
      // over Quad8.
      return order <= 1 ? GeometryVariant::Prism6 : GeometryVariant::Prism18;
    case CellFamily::Pyramid:
      return order <= 1 ? GeometryVariant::Pyramid5 : GeometryVariant::Pyramid14;
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
    case CellFamily::Prism:
      return "Prism";
    case CellFamily::Pyramid:
      return "Pyramid";
  }
  return "Unknown";
}

/// The number of distinct local-vertex orderings a (entity_dimension,
/// local_entity_index) sub-entity of `cell` can be seen in by its two
/// (or more) incident elements -- 2 for any edge, 6 for a triangular
/// face, 8 for a quadrilateral face, 1 (no choice) for anything else
/// (cell-dimension entity_dimension itself: never shared, so never
/// re-oriented).
unsigned orientation_count(CellFamily cell, unsigned entity_dimension, unsigned local_entity_index) {
  if (entity_dimension == 1) return 2;  // every edge, in every cell family
  if (entity_dimension != 2) return 1;
  const auto side_family = ReferenceTopology(variant_for(cell, 1)).side_topology(local_entity_index).family();
  if (side_family == CellFamily::Triangle) return 6;
  if (side_family == CellFamily::Quadrilateral) return 8;
  return 1;
}

std::string continuity_name(BasisContinuity continuity) {
  if (continuity == BasisContinuity::cG) return "cG";
  if (continuity == BasisContinuity::dG) return "dG";
  return "SE";
}

/// Lagrange nodal basis of order 0 (cell-local constant), 1 or 2, on one of
/// the supported cell families. Order 1/2 shape values and gradients are
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

    // Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid.
    constexpr std::array<unsigned, 7> edge_count{1, 3, 4, 6, 12, 9, 8};
    const unsigned edges = edge_count[static_cast<std::size_t>(key_.cell)];
    for (unsigned i = 0; i < edges; ++i) result.push_back({1, i, 0});
    // Hexahedron's 6 quad faces and Prism's 3 quad SIDE faces (not its 2
    // triangular caps, which carry no P2 face dof -- Tri6 has none
    // either) each get a face-centre dof at this order.
    const unsigned face_centers = key_.cell == CellFamily::Hexahedron ? 6 : key_.cell == CellFamily::Prism ? 3 : 0;
    for (unsigned i = 0; i < face_centers; ++i) result.push_back({2, i, 0});
    // The pyramid's one extra P2 node (Pyramid14's 14th, at the base
    // centre p(0,0,0)) lies ON the base QUAD face -- side index 4 in
    // side_topology's ordering (4 triangular sides first, the
    // quad base last) -- so it is a genuine face dof, not a cell-
    // interior one: it must be covered by an orientation transform when
    // two pyramids share a base face, same as the hexahedron's and
    // prism's face centres.
    if (key_.cell == CellFamily::Pyramid) result.push_back({2, 4, 0});
    if (result.size() < size()) result.push_back({cell_dimension_of(key_.cell), 0, 0});
    return result;
  }

  /// KARMA's own construction (karma_basis.C's BuiltinBasis::
  /// orientation_transforms): for every (entity_dimension, sub-entity)
  /// this basis places a dof on, and every orientation code that
  /// sub-entity's shape can present, emit the IDENTITY permutation with
  /// sign +1 -- correct by construction whenever the sub-entity carries
  /// at most one dof (basis.hpp's header comment derives why that holds
  /// for every key the registry builds), ported unchanged rather
  /// than special-cased so a future higher-order basis only needs to
  /// override this method, not re-derive the surrounding bookkeeping.
  [[nodiscard]] std::vector<OrientationTransform> orientation_transforms() const {
    std::vector<OrientationTransform> result;
    const auto associations = dof_associations();
    const unsigned top_dimension = cell_dimension_of(key_.cell);
    for (unsigned entity_dimension = 1; entity_dimension < top_dimension; ++entity_dimension) {
      const unsigned n_entities = entity_dimension == 1 ? edge_count_of(key_.cell) : side_count_of(key_.cell);
      for (unsigned entity = 0; entity < n_entities; ++entity) {
        const auto dof_count = static_cast<std::size_t>(
            std::count_if(associations.begin(), associations.end(), [=](const DofAssociation& association) {
              return association.entity_dimension == entity_dimension && association.local_entity_index == entity;
            }));
        const unsigned codes = orientation_count(key_.cell, entity_dimension, entity);
        for (unsigned code = 0; code < codes; ++code) {
          std::vector<std::size_t> permutation(dof_count);
          for (std::size_t i = 0; i < dof_count; ++i) permutation[i] = i;
          result.push_back({entity_dimension, entity, code, std::move(permutation), std::vector<int>(dof_count, 1)});
        }
      }
    }
    return result;
  }

 private:
  static unsigned cell_dimension_of(CellFamily cell) { return cell_dimension(cell); }
  static unsigned edge_count_of(CellFamily cell) {
    constexpr std::array<unsigned, 7> edge_count{1, 3, 4, 6, 12, 9, 8};
    return edge_count[static_cast<std::size_t>(cell)];
  }
  static unsigned side_count_of(CellFamily cell) { return ReferenceTopology(variant_for(cell, 1)).side_count(); }

  BasisKey key_;
  ReferenceTopology reference_;
};

}  // namespace

BasisRegistry::BasisRegistry() {
  constexpr std::array cells{CellFamily::Edge,        CellFamily::Triangle, CellFamily::Quadrilateral,
                              CellFamily::Tetrahedron, CellFamily::Hexahedron, CellFamily::Prism,
                              CellFamily::Pyramid};
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

namespace {
const OrientationTransform& find_transform(const std::vector<OrientationTransform>& transforms,
                                            unsigned entity_dimension, unsigned local_entity_index,
                                            unsigned orientation_code) {
  const auto found = std::find_if(transforms.begin(), transforms.end(), [&](const OrientationTransform& transform) {
    return transform.entity_dimension == entity_dimension && transform.local_entity_index == local_entity_index &&
           transform.orientation_code == orientation_code;
  });
  if (found == transforms.end()) refuse("unknown orientation state for this basis's sub-entity");
  return *found;
}
}  // namespace

void Basis::apply_orientation(unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                               std::span<const double> local, std::span<double> canonical) const {
  const auto& transform = find_transform(orientation_transforms_, entity_dimension, local_entity_index, orientation_code);
  if (local.size() != transform.local_to_canonical.size() || canonical.size() != local.size())
    refuse("orientation coefficient storage must match the sub-entity's own dof count");
  for (std::size_t i = 0; i < local.size(); ++i)
    canonical[transform.local_to_canonical[i]] = static_cast<double>(transform.signs[i]) * local[i];
}

void Basis::apply_orientation_inverse(unsigned entity_dimension, unsigned local_entity_index,
                                       unsigned orientation_code, std::span<const double> canonical,
                                       std::span<double> local) const {
  const auto& transform = find_transform(orientation_transforms_, entity_dimension, local_entity_index, orientation_code);
  if (canonical.size() != transform.local_to_canonical.size() || local.size() != canonical.size())
    refuse("orientation coefficient storage must match the sub-entity's own dof count");
  for (std::size_t i = 0; i < canonical.size(); ++i)
    local[i] = static_cast<double>(transform.signs[i]) * canonical[transform.local_to_canonical[i]];
}

}  // namespace fuse::physics::fe
