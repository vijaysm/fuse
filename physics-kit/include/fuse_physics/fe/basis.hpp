#pragma once

/// Geometry-independent scalar finite-element bases, adapted from KARMA's
/// karma_basis.h/.C (docs/dev/physics-kit-disposition.md Appendix A,
/// port-adapt, owner decision D-A).
///
/// WAVE 1b SCOPE (reference_topology.hpp's own note applies here too):
/// built-in Lagrange nodal bases of order 0, 1 or 2 on all seven
/// CellFamily values this wave's ReferenceTopology implements (Edge,
/// Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid) --
/// order 2 meaning the complete tensor/simplex family (Quad9, Tet10,
/// Hex27, Prism18, Pyramid14), not a serendipity variant (Quad8, Prism15,
/// Pyramid13 -- reference_topology.hpp's own face-shape-only role for
/// those). FieldBasisFamily/BasisContinuity keep KARMA's full vocabulary
/// (a later wave's user-defined or hierarchical/Legendre bases plug into
/// the same registry) even though only LagrangeNodal has a built-in
/// factory so far.
///
/// ORIENTATION (owner instruction 2026-10-31, item 5): ported from
/// KARMA's karma_basis.h EdgeOrientation/TriangleOrientation/
/// QuadrilateralOrientation and OrientationTransform, ADDED to the Basis
/// surface below rather than kept out-of-scope as wave 1 left it. A basis
/// function pinned to a shared edge or face (DofAssociation with
/// entity_dimension >= 1) may be evaluated by two elements that see that
/// sub-entity in opposite local vertex order; apply_orientation/
/// apply_orientation_inverse re-signs/permutes that sub-entity's local
/// DOF slots into a GLOBAL canonical order so assembly (a later wave) can
/// add contributions from both elements into the same global unknown.
/// This wave's built-in bases (order <= 2) never place more than ONE DOF
/// on any shared edge or face (an edge midpoint, a face centre), so every
/// generated OrientationTransform is PROVABLY the identity permutation
/// with sign +1 for every orientation code: a permutation of a one- (or
/// zero-) element set has only one value. fe_basis_test.cpp's own
/// orientation tests hold the round-trip to EXACT (not approximate)
/// equality for this reason -- see that file's comment for where the
/// bound would stop being zero (an order >= 3 basis with two or more DOFs
/// on one edge, not in this wave's scope).

#include <array>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

enum class FieldBasisFamily { LagrangeNodal, Hierarchical, Legendre, SpectralElement, UserDefined };
enum class BasisContinuity { cG, dG, SE };
enum class CoefficientRepresentation { Nodal, Modal };

inline constexpr unsigned maximum_basis_order = 1024;

struct BasisKey {
  CellFamily cell;
  FieldBasisFamily family;
  unsigned order;
  BasisContinuity continuity;

  [[nodiscard]] bool operator==(const BasisKey&) const = default;
  [[nodiscard]] bool operator<(const BasisKey& rhs) const noexcept {
    if (cell != rhs.cell) return cell < rhs.cell;
    if (family != rhs.family) return family < rhs.family;
    if (order != rhs.order) return order < rhs.order;
    return continuity < rhs.continuity;
  }
};

/// Which reference sub-entity (by dimension and local index) a basis
/// function's degree of freedom is pinned to -- 0 for a vertex DOF, 1 for
/// an edge DOF, up to cell_dimension(cell) for an interior DOF.
struct DofAssociation {
  unsigned entity_dimension;
  unsigned local_entity_index;
  unsigned slot;

  [[nodiscard]] bool operator==(const DofAssociation&) const = default;
};

/// A shared edge's two possible local-to-canonical vertex orders.
enum class EdgeOrientation : unsigned { Direct, Reversed };
/// A shared triangular face's six possible local-to-canonical vertex
/// orders (3 rotations, each with or without a reflection).
enum class TriangleOrientation : unsigned { Rotate0, Rotate1, Rotate2, ReflectRotate0, ReflectRotate1, ReflectRotate2 };
/// A shared quadrilateral face's eight possible local-to-canonical vertex
/// orders (4 rotations, each with or without a reflection).
enum class QuadrilateralOrientation : unsigned {
  Rotate0,
  Rotate1,
  Rotate2,
  Rotate3,
  ReflectRotate0,
  ReflectRotate1,
  ReflectRotate2,
  ReflectRotate3
};

/// These permutations map local vertex indices to canonical indices.
/// RotateK maps i to (i + K) mod N; ReflectRotateK maps i to (K - i) mod N
/// for cyclic vertex order (KARMA's karma_basis.h, same comment, same
/// tables).
[[nodiscard]] constexpr std::array<unsigned, 2> local_to_canonical_vertices(EdgeOrientation orientation) {
  constexpr std::array<std::array<unsigned, 2>, 2> permutations{{{0, 1}, {1, 0}}};
  return permutations.at(static_cast<std::size_t>(orientation));
}
[[nodiscard]] constexpr std::array<unsigned, 3> local_to_canonical_vertices(TriangleOrientation orientation) {
  constexpr std::array<std::array<unsigned, 3>, 6> permutations{
      {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}}};
  return permutations.at(static_cast<std::size_t>(orientation));
}
[[nodiscard]] constexpr std::array<unsigned, 4> local_to_canonical_vertices(QuadrilateralOrientation orientation) {
  constexpr std::array<std::array<unsigned, 4>, 8> permutations{{{0, 1, 2, 3},
                                                                   {1, 2, 3, 0},
                                                                   {2, 3, 0, 1},
                                                                   {3, 0, 1, 2},
                                                                   {0, 3, 2, 1},
                                                                   {1, 0, 3, 2},
                                                                   {2, 1, 0, 3},
                                                                   {3, 2, 1, 0}}};
  return permutations.at(static_cast<std::size_t>(orientation));
}

/// How to re-sign/permute the DOF slots this basis places on one
/// sub-entity (entity_dimension, local_entity_index) when that sub-entity
/// is seen in the local vertex order named by orientation_code (an
/// EdgeOrientation/TriangleOrientation/QuadrilateralOrientation value,
/// stored as its underlying unsigned so OrientationTransform stays
/// shape-agnostic): canonical[local_to_canonical[i]] = signs[i] *
/// local[i].
struct OrientationTransform {
  unsigned entity_dimension;
  unsigned local_entity_index;
  unsigned orientation_code;
  std::vector<std::size_t> local_to_canonical;
  std::vector<int> signs;

  [[nodiscard]] bool operator==(const OrientationTransform&) const = default;
};

/// A type-erased scalar basis, ported from KARMA's own type-erasure
/// pattern (KARMABasis's Concept/Model, the same technique std::function
/// uses): any Basis-shaped type can be wrapped without the registry
/// depending on its concrete type.
class Basis {
 public:
  template <class Concrete>
  explicit Basis(Concrete basis) : self_(std::make_shared<Model<Concrete>>(std::move(basis))) {
    key_ = self_->key();
    size_ = self_->size();
    interpolation_points_ = self_->interpolation_points();
    dof_associations_ = self_->dof_associations();
    orientation_transforms_ = self_->orientation_transforms();
    if (interpolation_points_.size() != size_ || dof_associations_.size() != size_)
      throw std::invalid_argument("basis size, interpolation points, and associations must agree");
  }

  [[nodiscard]] std::string name() const { return self_->name(); }
  [[nodiscard]] BasisKey key() const noexcept { return key_; }
  [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept {
    return self_->coefficient_representation();
  }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  void evaluate_values(const Point& point, std::span<double> values) const { self_->evaluate_values(point, values); }
  void evaluate(const Point& point, std::span<double> values, std::span<Vector3> gradients) const {
    self_->evaluate(point, values, gradients);
  }
  [[nodiscard]] const std::vector<Point>& interpolation_points() const noexcept { return interpolation_points_; }
  [[nodiscard]] const std::vector<DofAssociation>& dof_associations() const noexcept { return dof_associations_; }
  [[nodiscard]] const std::vector<OrientationTransform>& orientation_transforms() const noexcept {
    return orientation_transforms_;
  }

  /// Re-signs/permutes `local`'s DOF slots on (entity_dimension,
  /// local_entity_index) into `canonical`'s global order, using the
  /// transform named by orientation_code (basis.cpp's own
  /// validate_metadata has already confirmed every (entity, code) this
  /// basis's sub-entities can present is covered). `canonical` and
  /// `local` must each have the sub-entity's own DOF count.
  void apply_orientation(unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                          std::span<const double> local, std::span<double> canonical) const;
  /// The inverse of apply_orientation: canonical order back to local order.
  void apply_orientation_inverse(unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                                  std::span<const double> canonical, std::span<double> local) const;

 private:
  struct Concept {
    virtual ~Concept() = default;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual BasisKey key() const noexcept = 0;
    [[nodiscard]] virtual CoefficientRepresentation coefficient_representation() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    virtual void evaluate_values(const Point&, std::span<double>) const = 0;
    virtual void evaluate(const Point&, std::span<double>, std::span<Vector3>) const = 0;
    [[nodiscard]] virtual std::vector<Point> interpolation_points() const = 0;
    [[nodiscard]] virtual std::vector<DofAssociation> dof_associations() const = 0;
    [[nodiscard]] virtual std::vector<OrientationTransform> orientation_transforms() const = 0;
  };
  template <class Concrete>
  struct Model final : Concept {
    explicit Model(Concrete value) : basis(std::move(value)) {}
    [[nodiscard]] std::string name() const override { return basis.name(); }
    [[nodiscard]] BasisKey key() const noexcept override { return basis.key(); }
    [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept override {
      return basis.coefficient_representation();
    }
    [[nodiscard]] std::size_t size() const noexcept override { return basis.size(); }
    void evaluate_values(const Point& point, std::span<double> values) const override {
      basis.evaluate_values(point, values);
    }
    void evaluate(const Point& point, std::span<double> values, std::span<Vector3> gradients) const override {
      basis.evaluate(point, values, gradients);
    }
    [[nodiscard]] std::vector<Point> interpolation_points() const override { return basis.interpolation_points(); }
    [[nodiscard]] std::vector<DofAssociation> dof_associations() const override { return basis.dof_associations(); }
    [[nodiscard]] std::vector<OrientationTransform> orientation_transforms() const override {
      return basis.orientation_transforms();
    }
    Concrete basis;
  };

  std::shared_ptr<const Concept> self_;
  BasisKey key_{};
  std::size_t size_ = 0;
  std::vector<Point> interpolation_points_;
  std::vector<DofAssociation> dof_associations_;
  std::vector<OrientationTransform> orientation_transforms_;
};

enum class DuplicatePolicy { Reject, Replace };

/// Builds Basis instances from a BasisKey, pre-populated with the built-in
/// Lagrange nodal factories (BuiltinBasis, basis.cpp) for every (cell,
/// order <= 2, continuity) this wave supports; register_factory() lets a
/// later wave add Hierarchical/Legendre/SpectralElement/UserDefined
/// factories without changing this class.
class BasisRegistry {
 public:
  using Factory = std::function<Basis()>;

  BasisRegistry();
  void register_factory(const BasisKey& key, Factory factory, DuplicatePolicy policy = DuplicatePolicy::Reject);
  [[nodiscard]] bool supports(const BasisKey& key) const noexcept;
  [[nodiscard]] Basis create(const BasisKey& key) const;

 private:
  std::map<BasisKey, Factory> factories_;
};

}  // namespace fuse::physics::fe
