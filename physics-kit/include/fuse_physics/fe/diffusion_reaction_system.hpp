#pragma once

/// fuse_physics::fe::DiffusionReactionSystem -- the physics-module
/// contract (one concrete class, not an abstract interface) for the
/// scalar diffusion-reaction equation
///
///   du/dt = div(k(x,u) grad(u)) - R(x,u) + f(x, t)   in the box domain,
///   u = g_D(x, t)                on every Dirichlet boundary,
///   k du/dn = g_N(x, t)          on every Neumann boundary,
///   k du/dn + c u = g_R(x, t)    on every Robin boundary (c > 0),
///
/// adapted from KARMA's include/physics/base/karma_physics_base.h (the
/// generic physics-module virtual interface) and include/physics/
/// diffusion-reaction/{diffusion_reaction_system.h,
/// diffusion_reaction_system_operator.h,
/// diffusion_reaction_mass_operator.h, diffusion_reaction_assembly.h}
/// (the concrete diffusion-reaction model built on it).
///
/// ONE CONCRETE CLASS, NOT AN ABSTRACT BASE. KARMAPhysicsBase is an
/// abstract interface with about 25 virtuals so that KARMA's own
/// time-stepping/solve driver can hold a `KARMAPhysicsBase*` to ANY
/// physics module; physics-kit's time/nonlinear layers are fuse::num's
/// `Integrator<Value>`/`fuse::num::solve`, which take a
/// `TemporalProblem<Value>` (a struct of std::function members) or a
/// `ResidualFn` directly -- NEITHER needs a physics module to inherit
/// from anything (physics-kit depends on fuse::num, never the reverse,
/// and owns no second Newton or time loop). `DiffusionReactionSystem`'s
/// public methods ARE the few KARMAPhysicsBase virtuals this one
/// physics module needs, bound into `TemporalProblem`/`ResidualFn`
/// through `as_temporal_problem()` rather than through inheritance. A
/// second physics module would be its own concrete class with the same
/// shape, not a sibling override of this one.
///
/// WHICH KARMAPhysicsBase VIRTUALS THIS REPLACES, AND WHICH IT DROPS
/// (docs/dev/physics-kit.md carries the same table):
///   form_initial_guess            -> project_initial_condition (nodal
///                                     interpolation of a u0(x) callable
///                                     -- a Lagrange basis's own dof IS
///                                     its own interpolation point, so
///                                     no separate L2 projection solve
///                                     is needed for this basis family).
///   compute_L2_error (KARMA's own declaration is commented out, no
///     live implementation exists upstream to port) -> l2_error,
///     implemented here using the quadrature-sum L2 error construction.
///   solve() (KARMA's own physics-owned solve loop) -> DROPPED. A
///     caller solves the steady problem via `fuse::num::solve(...)` and
///     the transient problem via any `fuse::num::Integrator` stepping
///     `as_temporal_problem()` directly.
///   find_boundary_conditions -> DROPPED; replaced wholesale by
///     `BoundaryConditionManager::set`/`resolve`.
///   get_send_list, get_n_nz, get_n_oz -> DROPPED: PETSc Mat
///     preallocation-era bookkeeping; `fuse::linalg::DistSparseMatrix::
///     from_triplets` infers its own sparsity.
///   init_coupled_physics, update_coupled_physics_solution -> DROPPED:
///     a `Component`/`ImplicitComponent` adapter wraps this class from
///     outside for coupling, rather than this class growing coupling-
///     aware virtuals.
///   interpolate_solution and its quadrature-point-matrix siblings ->
///     DROPPED: field transfer onto a SECOND mesh, out of scope for a
///     single-mesh physics module.
///   material_number, partition_number, n_materials, get_properties_base
///     -> DROPPED along with a material_id-keyed property map (this
///     port's self-generated meshes carry no material/block id to key
///     against, diffusion_reaction_properties.hpp's own header comment).
///   get_solution_with_name, add_system -> DROPPED: this class owns
///     exactly ONE scalar field; a caller needing several holds several
///     instances directly.
///   read, write -> DROPPED: no checkpoint format exists in physics-kit.
///   print_info, system_type, is_continuous, clear, reinit,
///   prepare_for_transient, step_preprocess, step_postprocess,
///   stage_preprocess, stage_postprocess -> REPLACED, overall, by this
///     class's own four-verb lifecycle: construct, initialize(), the
///     caller's own step through fuse::num, and re-initialize.
///
/// NO ASSEMBLED NEWTON JACOBIAN. `fuse::num::solve`'s own Newton-Krylov
/// is Jacobian-free (an internal matrix-free MFFD directional
/// derivative of the residual, newton_krylov.hpp's own header comment).
/// `assemble_approximate_jacobian` below exists ONLY to drive an
/// optional Jacobi/SSOR PRECONDITIONER for that same matrix-free solve
/// (`fuse::num::solve`'s own `RefreshPreconditionerFn`/
/// `PreconditionerApplyFn`, matching the pattern physics-kit/test/
/// steady_reaction_diffusion_mms_test.cpp's own
/// `assemble_steady_jacobian` already establishes); the Newton solve
/// itself never reads an assembled matrix.
///
/// DIRICHLET ROWS, ONE-SIDED, NOT `eliminate_dirichlet_rows`.
/// `AlgebraicSystem::eliminate_dirichlet_rows` performs SYMMETRIC (row
/// AND column) elimination for a PCG-compatible assembled linear solve.
/// Nothing in the STEADY solve runs PCG (matrix-free GMRES inside
/// `fuse::num::solve`, with no symmetry requirement on an assembled
/// system since there is none): `assemble_residual` and
/// `assemble_approximate_jacobian` both use the ONE-SIDED identity-row
/// convention `steady_reaction_diffusion_mms_test.cpp` already
/// establishes and measures (identity row in the residual, matched
/// row-for-row in the preconditioner's own Dirichlet rows -- that
/// file's own measurement: a mismatched preconditioner row roughly
/// doubles the GMRES count). `eliminate_dirichlet_rows`'s own
/// single-rank limitation is orthogonal to this class's own scope:
/// `DiffusionReactionSystem` is single-rank regardless, because
/// `BoundaryConditionManager::resolve` already refuses a multi-rank
/// `DofMap` by name.
///
/// THE SEMI-DISCRETE SYSTEM AND ITS TWO FORMS. With F the set of free
/// dofs, D the Dirichlet dofs and g the Dirichlet data, the Galerkin
/// semi-discretisation is
///   sum_j M_ij u'_j = f_i(t, u)   for i in F  (j over ALL dofs),
///   u'_i = g'_i(t)                for i in D,
/// where f = load - K(u) u - M R(u) is the negated spatial operator and M
/// the consistent mass matrix. Two integrator-facing forms are offered:
///
///  * The mass form, `as_temporal_problem()`: `TemporalProblem::rhs` is
///    `rhs()` (f on the free rows, g' on the Dirichlet rows) and
///    `TemporalProblem::mass` is `mass()` (M on the free rows, including
///    its Dirichlet columns; the identity on the Dirichlet rows). Nothing
///    is solved inside either function, so an implicit integrator that
///    differences its stage residual differences exact arithmetic. This is
///    the form for an integrator with mass-matrix support
///    (`fuse::num::DirkIntegrator`, implicit stages).
///
///  * The explicit form, `as_explicit_form_temporal_problem()`:
///    `TemporalProblem::rhs` is `explicit_rhs()`, which returns u' itself
///    by solving the system above, and no mass operator is set. The free
///    rows are solved with the Dirichlet columns moved to the right-hand
///    side, `sum_{j in F} M_ij u'_j = f_i - sum_{j in D} M_ij g'_j`, so
///    the matrix is symmetric positive definite. This is the form for an
///    integrator without mass-matrix support (the explicit family, and
///    `fuse::num::RosenbrockIntegrator`). The inner solve is carried to
///    the accuracy the arithmetic allows (diffusion_reaction_system.cpp),
///    because an integrator that differences `explicit_rhs` to form a
///    Jacobian action amplifies its rounding error by the reciprocal of
///    the differencing step.
///
/// TRANSIENT DIRICHLET ROWS ARE DIFFERENTIATED. A Dirichlet row of either
/// form states `u' = g'(t)` with `g'` read from
/// `BoundaryCondition::value_time_derivative`; with `u(0) = g(0)` from
/// `project_initial_condition` it integrates to `u(t) = g(t)` to the
/// integrator's own accuracy, for every tableau (an algebraic row
/// `0 = g(t) - u` is satisfied at the end of a step only by a stiffly
/// accurate method). A Dirichlet condition without `value_time_derivative`
/// is refused by `rhs` (E-PHYSICS-DRS-DIRICHLET-RATE-MISSING): a missing
/// rate cannot be told apart from boundary data that is constant in time,
/// and treating it as zero would hold time-dependent data at its initial
/// value without a diagnostic. `BoundaryCondition::steady_dirichlet`
/// builds the constant-in-time case.
///
/// CACHED MASS MATRICES. `initialize()` assembles the consistent mass
/// matrix M, and M with every Dirichlet row and column replaced by the
/// identity (with its Jacobi preconditioner), once; `mass()` and
/// `explicit_rhs()` apply them as sparse products. Both depend on the
/// mesh, the basis and the set of Dirichlet dofs only.

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse/num/integrator.hpp"
#include "fuse_physics/fe/algebraic_system.hpp"
#include "fuse_physics/fe/boundary_condition.hpp"
#include "fuse_physics/fe/diffusion_reaction_properties.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/geometry.hpp"
#include "fuse_physics/spatial/domain.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::fe {

/// (position, time) -> value -- the same two-argument shape
/// `BoundaryValue` uses, for the problem's own forcing term (a time-
/// dependent property of the PROBLEM being solved, kept apart from
/// `DiffusionReactionCoefficients`, a property of the MATERIAL --
/// diffusion_reaction_properties.hpp's own header comment).
using ForcingTerm = std::function<double(const Point&, double)>;

/// (position) -> value -- an initial condition, evaluated once at t=0
/// with no time argument.
using InitialCondition = std::function<double(const Point&)>;

class DiffusionReactionSystem {
 public:
  /// `mesh`/`domain`/`dof_map` outlive this object. `dof_map` must
  /// carry exactly one scalar variable (n_variables() == 1) -- refuses
  /// E-PHYSICS-DRS-NOT-SCALAR otherwise: a vector-valued or multi-field
  /// DiffusionReactionSystem is out of scope (gray_scott_model.h's own
  /// two-field u/v extension would be a separate class with this one's
  /// same shape, not built here).
  DiffusionReactionSystem(const spatial::Mesh& mesh, const spatial::Domain& domain, const DofMap& dof_map,
                           Basis basis, GeometryVariant geometry_variant, int quadrature_degree);

  void set_coefficients(DiffusionReactionCoefficients coefficients) { coefficients_ = std::move(coefficients); }
  [[nodiscard]] const DiffusionReactionCoefficients& coefficients() const noexcept { return coefficients_; }

  void set_forcing(ForcingTerm forcing) { forcing_ = std::move(forcing); }

  /// Registers one condition per boundary id (BoundaryConditionManager's
  /// own set() refusals -- E-PHYSICS-BC-UNKNOWN-BOUNDARY,
  /// E-PHYSICS-BC-DUPLICATE, E-PHYSICS-BC-ROBIN-COEFFICIENT -- apply
  /// unchanged). Must be called, for every boundary id the domain's own
  /// dimension has, before `initialize()`.
  void set_boundary_condition(BoundaryId boundary_id, BoundaryCondition condition);

  /// LIFECYCLE VERB 2: resolves every registered boundary condition
  /// against `dof_map` (BoundaryConditionManager::resolve, refusing
  /// E-PHYSICS-BC-NO-CONDITIONS, E-PHYSICS-BC-MPI-NOT-SUPPORTED, and
  /// E-PHYSICS-BC-UNMATCHED as that class's own header states) and
  /// builds the element->face lookup `assemble_residual`/`rhs` use for
  /// Neumann/Robin contributions. Safe to call again (LIFECYCLE VERB 4,
  /// re-initialize) after `set_boundary_condition`/`set_coefficients`/
  /// `set_forcing` changes: it fully rebuilds internal state from the
  /// CURRENT registrations rather than accumulating, so a changed
  /// boundary value or coefficient takes effect from the next
  /// `assemble_residual`/`rhs` call onward. Refuses
  /// E-PHYSICS-DRS-NOT-INITIALIZED if `assemble_residual`/`rhs`/`mass`/
  /// `project_initial_condition` is called before the FIRST
  /// `initialize()`.
  void initialize();

  [[nodiscard]] const BoundaryConditionManager& boundary_conditions() const { return boundary_; }
  [[nodiscard]] const DofMap& dof_map() const noexcept { return *dof_map_; }

  /// LIFECYCLE VERB 3 (one evaluation of it): the steady Newton
  /// residual, `fuse::num::solve`'s own `ResidualFn` contract
  /// (newton_krylov.hpp) -- `x`/`fx` sized `dof_map().n_dofs()`,
  /// single-rank. `t` is fixed at 0.0 for every boundary/forcing
  /// evaluation (a steady problem's own data has no time axis).
  [[nodiscard]] fuse::Result<void> assemble_residual(fuse::linalg::DistVector& x, fuse::linalg::DistVector& fx) const;

  /// The right-hand side of the mass form: `out_i = f_i(t, y)` on the free
  /// rows and `out_i = g'_i(t)` on the Dirichlet rows. Refuses
  /// E-PHYSICS-DRS-DIRICHLET-RATE-MISSING when a Dirichlet condition has
  /// no `value_time_derivative`.
  [[nodiscard]] fuse::Result<void> rhs(double t, std::span<const double> y, std::span<double> out) const;

  /// The mass operator of the mass form: `out_i = sum_j M_ij v_j` on the
  /// free rows (j over all dofs) and `out_i = v_i` on the Dirichlet rows.
  [[nodiscard]] fuse::Result<void> mass(std::span<const double> v, std::span<double> out) const;

  /// The right-hand side of the explicit form: `out = u'`, the solution of
  /// the semi-discrete system for the state `y` at time `t`.
  [[nodiscard]] fuse::Result<void> explicit_rhs(double t, std::span<const double> y, std::span<double> out) const;

  /// `M u' = f(t, u)`: `rhs` and `mass` bound to this object, which must
  /// outlive the returned problem. `dimension` is `dof_map().n_dofs()`;
  /// `jacobian_action`, `dfdt` and `dynamical_time_scales` are unset.
  [[nodiscard]] fuse::num::TemporalProblem<double> as_temporal_problem() const;

  /// `u' = explicit_rhs(t, u)`, with no mass operator: for an integrator
  /// without mass-matrix support. Same lifetime rule as above.
  [[nodiscard]] fuse::num::TemporalProblem<double> as_explicit_form_temporal_problem() const;

  /// Sizes `out` to `dof_map().n_dofs()` and sets `out[dof] <-
  /// u0(position of dof)` at every dof (the general interpolation-point
  /// position `dof_physical_positions` computes, valid for any basis
  /// order, not only a vertex dof). Every Dirichlet dof instead reads
  /// the boundary condition's own value at t=0, not `u0`: an initial
  /// condition that disagreed with the boundary data at t=0 would make
  /// `rhs`'s own Dirichlet row integrate away from the wrong starting
  /// point; reading the boundary data keeps the two consistent by
  /// construction. A transient problem wanting `u0 == u_exact(., 0)`
  /// passes `[&](const Point& p){ return u_exact(p, 0.0); }`.
  void project_initial_condition(const InitialCondition& u0, std::vector<double>& out) const;

  /// `sqrt(integral((u_h(x) - exact(x, t))^2) dV)`, exact nodal values
  /// `x` interpolated through the SAME basis `assemble_residual`/`rhs`
  /// use, quadrature-summed over every owned element.
  [[nodiscard]] double l2_error(const std::vector<double>& x, double t,
                                 const std::function<double(const Point&, double)>& exact) const;

  /// The matrix `K(k(x)) + M diag(R'(x)) + B`, with an identity row at
  /// every Dirichlet dof: K the diffusion form with the diffusion
  /// coefficient evaluated at the state `x`, M diag(R') the derivative of
  /// the nodal reaction term (`reaction_derivative` unset is read as 0),
  /// and B the boundary mass block of the Robin faces. For a diffusion
  /// coefficient that does not depend on the field this is the Jacobian of
  /// `assemble_residual` at `x`; for one that does, the derivative of the
  /// coefficient is not included (a Picard linearisation), which is why
  /// the name says approximate. Its use is as the matrix a Jacobi, SSOR
  /// or block-Jacobi preconditioner of the matrix-free Newton-Krylov solve
  /// is built from (`fuse::num::solve`, `RefreshPreconditionerFn` and
  /// `PreconditionerApplyFn`).
  [[nodiscard]] fuse::Result<linalg::DistSparseMatrix> assemble_approximate_jacobian(
      const fuse::linalg::DistVector& x) const;

 private:
  const spatial::Mesh* mesh_;
  const spatial::Domain* domain_;
  const DofMap* dof_map_;
  Basis basis_;
  GeometryVariant geometry_variant_;
  mutable FEBase fe_;            ///< mutable: FEBase::reinit mutates cached quadrature state, read-only from this class's own public surface
  mutable FEBase face_fe_;       ///< a second FEBase, reinit(element, side)-only, kept separate from fe_ so a volume and a face reinit never invalidate each other's cached state mid-loop
  AlgebraicSystem system_;       ///< owns this rank's Reducer/GhostExchanger; its own new_vector()/assemble_vector()/assemble_matrix() build and fill every DistVector/DistSparseMatrix this class returns
  DiffusionReactionCoefficients coefficients_;
  ForcingTerm forcing_ = [](const Point&, double) { return 0.0; };
  BoundaryConditionManager boundary_;
  bool initialized_ = false;
  /// element -> every resolved Neumann/Robin face ON that element
  /// (built once in initialize(), read by the per-element accumulation
  /// lambda -- most elements map to nothing, an interior element of a
  /// box mesh touching no boundary face at all).
  std::unordered_map<spatial::EntityHandle, std::vector<const ResolvedFace*>> faces_by_element_;
  /// dof -> the ResolvedDirichlet naming its own boundary value (built
  /// once in initialize(), read by assemble_residual/rhs/mass's own
  /// row-identity override without a per-call linear scan of
  /// boundary_.dirichlet_dofs()).
  std::unordered_map<std::int64_t, const ResolvedDirichlet*> dirichlet_by_dof_;
  /// dof -> physical position, built once in initialize() (a GeometryMap
  /// interpolation per element, dof_physical_positions()'s own cost) and
  /// read directly by every later residual/rhs/mass/Jacobian/initial-
  /// condition call. A per-call recomputation here would run once per
  /// MFFD directional-derivative evaluation inside the outer Newton-
  /// Krylov solve -- potentially thousands of times per solve -- turning
  /// an O(n_elements) setup cost into part of the per-call cost of every
  /// solve, on top of whatever FEBase::reinit itself costs per element
  /// (diffusion_reaction_steady_order_test.cpp's own comment on why a
  /// 3-D Hex8 ladder is not run measures THAT cost, GeometryMap::validate,
  /// as the dominant one for a non-affine 3-D element -- caching this
  /// member removes a second, smaller, genuinely redundant cost stacked
  /// on top of it, not that dominant one).
  std::vector<Point> dof_position_;
  /// The consistent mass matrix M (header comment, "CACHED MASS
  /// MATRICES").
  std::optional<linalg::DistSparseMatrix> mass_matrix_;
  /// M with every Dirichlet row and column replaced by the identity, and
  /// its Jacobi preconditioner: the symmetric positive definite system
  /// `explicit_rhs` solves.
  std::optional<linalg::DistSparseMatrix> eliminated_mass_matrix_;
  std::optional<linalg::JacobiPreconditioner> mass_preconditioner_;

  /// RAW spatial operator, Dirichlet rows NOT overridden: `out_i =
  /// sum_j K_ij(k) * y_j + sum_j M_ij * R(x_j, y_j) - load_i(forcing, t)`
  /// at every dof, interior and boundary alike (`+ neumann/robin face
  /// terms` folded into `load_i` at the dofs their own faces touch).
  /// `assemble_residual`/`rhs` each apply their own Dirichlet-row
  /// override afterward.
  [[nodiscard]] fuse::Result<fuse::linalg::DistVector> raw_spatial_operator(double t, const double* y) const;
};

}  // namespace fuse::physics::fe
