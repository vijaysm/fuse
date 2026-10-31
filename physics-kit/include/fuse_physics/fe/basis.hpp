#pragma once

/// Geometry-independent scalar finite-element bases, adapted from KARMA's
/// karma_basis.h/.C (docs/dev/physics-kit-disposition.md Appendix A,
/// port-adapt, owner decision D-A).
///
/// WAVE 1 SCOPE (reference_topology.hpp's own note applies here too):
/// built-in Lagrange nodal bases of order 0, 1 or 2 on the five CellFamily
/// values this wave's ReferenceTopology implements (Edge, Triangle,
/// Quadrilateral, Tetrahedron, Hexahedron) -- order 2 meaning the complete
/// tensor/simplex family (Quad9, Tet10, Hex27), not the Quad8 serendipity
/// variant. FieldBasisFamily/BasisContinuity keep KARMA's full vocabulary
/// (a later wave's user-defined or hierarchical/Legendre bases plug into
/// the same registry) even though only LagrangeNodal has a built-in
/// factory so far.

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
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
    Concrete basis;
  };

  std::shared_ptr<const Concept> self_;
  BasisKey key_{};
  std::size_t size_ = 0;
  std::vector<Point> interpolation_points_;
  std::vector<DofAssociation> dof_associations_;
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
