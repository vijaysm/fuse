#pragma once

/// Geometry-independent, physics-agnostic element operators: the mass
/// and stiffness (diffusion) bilinear forms every physics built on
/// fuse_physics::fe shares. Adapted from KARMA's own
/// include/physics/common/generic_mass_operator.h (port-adapted), the abstraction level
/// test/mass_assembly_test.C's own KARMA source actually exercises:
/// this, not karma_algebraic_system.h itself, is where a per-element
/// bilinear form belongs.
///
/// NOT A LITERAL PORT. KARMA's own element_mass_matrix(system, element,
/// degree) reads its FEBase/basis/quadrature objects out of a
/// KARMAAlgebraicSystem's own per-variable bookkeeping; this port takes
/// an already-reinit-able fe::FEBase directly (fe_base.hpp), so the SAME two functions below serve a caller that
/// built its FEBase from fe::AlgebraicSystem's own DofMap or from
/// neither (a standalone quadrature check, as fe_basis_test.cpp and this
/// file's own test do).
///
/// KARMA's include/numerics/karma_diffusion.h (a per-element stiffness
/// operator parallel to generic_mass_operator.h) does not exist in the
/// upstream/karma export this port builds from -- element_stiffness_
/// matrix below is the textbook FE bilinear form integral(grad(phi_i) .
/// grad(phi_j)) dV, derived and cited as such, the same
/// posture taken for KARMA gaps with no upstream file to port
/// from.

#include <functional>
#include <vector>

#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/geometry.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::fe {

/// `mass(i, j) = integral(phi_i * phi_j) dV` over `element`, row-major
/// `n x n` (`n = fe.get_phi().size()`, the basis size `fe` was built
/// with). `fe` must already be bound to the SAME mesh `element` lives
/// in; this function calls `fe.reinit(element)` itself. Exact for a
/// polynomial `phi_i * phi_j` of degree <= `fe`'s own quadrature degree
/// on an AFFINE element (constant Jacobian) -- the caller's own
/// responsibility to pick a high enough degree (the tests use 2 * basis order, exact for `phi_i *
/// phi_j` itself, on the affine box elements they build).
[[nodiscard]] std::vector<double> element_mass_matrix(FEBase& fe, spatial::EntityHandle element);

/// `stiffness(i, j) = integral(grad(phi_i) . grad(phi_j)) dV` over
/// `element`, row-major `n x n`. Same reinit and quadrature-degree
/// contract as `element_mass_matrix`; for a P1 (linear) basis the
/// gradient is CONSTANT per element, so `grad(phi_i) . grad(phi_j)` has
/// degree 0 and even a single-point rule integrates it exactly.
[[nodiscard]] std::vector<double> element_stiffness_matrix(FEBase& fe, spatial::EntityHandle element);

/// `load(i) = integral(phi_i * source(x)) dV` over `element`, the SAME
/// quadrature-sum construction `element_mass_matrix` uses (`fe.get_xyz()`
/// in place of a second basis function), generalised to an arbitrary
/// (possibly position-dependent) scalar `source`. No KARMA source ports
/// this -- `generic_mass_operator.h` has no load-vector method of its own
/// -- so this is the textbook consistent Galerkin load vector, cited as
/// such, the same posture taken for
/// `element_stiffness_matrix` (its own header comment states the
/// identical reasoning for a different missing KARMA file). Used by
/// the manufactured-solution forcing term:
/// evaluating `source` at each quadrature point's PHYSICAL coordinate
/// keeps the forcing term's own discretisation at the SAME quadrature
/// order as the bilinear forms it is added against, rather than the
/// coarser nodal (collocated) evaluation `reaction`/`reaction_derivative`
/// above use for the nonlinear term -- a deliberate difference, stated:
/// the forcing is DATA (known exactly, so quadrature-evaluating it adds
/// no new approximation question), while the reaction term is the
/// UNKNOWN solution itself, where KARMA's own nodal convention is kept
/// for the nonlinear term unchanged.
[[nodiscard]] std::vector<double> element_load_vector(FEBase& fe, spatial::EntityHandle element,
                                                       const std::function<double(const Point&)>& source);

/// A Crank-Nicolson transient reaction-diffusion model, adapted from
/// KARMA's own test/transient_reaction_diffusion_assembly_test.C
/// (KARMA's own model evaluates `R(state) =
/// alpha * state + beta * coupled + source`, AFFINE in `state`).
///
/// NOT a literal port of KARMA's own affine model. `R(state) = alpha *
/// state^2 + beta * coupled + source` -- QUADRATIC in `state` -- is
/// used here instead, because the finite-difference Jacobian check's tolerance is
/// DERIVED from the step size and the reaction term's second-
/// derivative bound: an affine reaction has R''(state) == 0 exactly,
/// which collapses that derivation to round-off alone (exercising
/// nothing about the truncation term). The quadratic
/// term keeps R''(state) == 2*alpha, a known, nonzero, EXACT constant
/// -- the derivation in element_transient_reaction_diffusion's own
/// header comment below is exact, not an estimate, because of this
/// choice.
struct TransientReactionDiffusionModel {
  double alpha;     ///< reaction curvature: R(u) = alpha*u^2 + beta*coupled + source
  double beta;
  double source;
  double coupled;   ///< a fixed externally-coupled value, held constant across the element
};

[[nodiscard]] inline double reaction(const TransientReactionDiffusionModel& model, double state) {
  return model.alpha * state * state + model.beta * model.coupled + model.source;
}
[[nodiscard]] inline double reaction_derivative(const TransientReactionDiffusionModel& model, double state) {
  return 2.0 * model.alpha * state;
}
/// R''(state) -- CONSTANT for this quadratic model, independent of `state`;
/// named so a caller (and this file's own Jacobian derivation) never
/// has to re-derive it from `reaction_derivative`'s own finite
/// difference.
[[nodiscard]] inline double reaction_second_derivative(const TransientReactionDiffusionModel& model) {
  return 2.0 * model.alpha;
}

/// `element`'s own Crank-Nicolson (theta = 1/2) residual and Jacobian
/// for `du/dt = -(reaction-diffusion operator)`, time step `h`, old
/// state `state_old` and new (trial) state `state`, both size `n` (the
/// basis `fe` was built with, matching `element_mass_matrix`'s own
/// contract):
///
///   residual_i = sum_j M_ij * (state_j - state_old_j) / h
///              + 0.5 * sum_j K_ij * (state_j + state_old_j)
///              - 0.5 * sum_j M_ij * (R(state_j) + R(state_old_j))
///
/// (M = element_mass_matrix, K = element_stiffness_matrix, R =
/// `reaction` above, evaluated NODALLY -- not via quadrature, KARMA's
/// own construction) and
///
///   jacobian_ik = d(residual_i)/d(state_k)
///               = M_ik / h + 0.5 * K_ik - 0.5 * M_ik * R'(state_k)
///
/// (R(state_j) depends on state_j alone, so only the k == j term of
/// the reaction sum survives the derivative -- EXACT, no truncation,
/// since this is the ANALYTIC Jacobian, not the finite-difference one
/// `jacobian_finite_difference_tolerance` below bounds).
struct ElementResidualJacobian {
  std::vector<double> residual;  ///< size n
  std::vector<double> jacobian;  ///< row-major n x n
  /// Sum of the absolute values of the 3n terms that form residual[i]
  /// (the quantity Higham's summation bound multiplies: the residual
  /// itself cancels, so |residual[i]| understates its round-off).
  std::vector<double> residual_abs_terms;  ///< size n
};
[[nodiscard]] ElementResidualJacobian element_transient_reaction_diffusion(
    FEBase& fe, spatial::EntityHandle element, const TransientReactionDiffusionModel& model, double h,
    const std::vector<double>& state, const std::vector<double>& state_old);

/// Derived step and error bound for checking one Jacobian entry J_ik by a
/// FORWARD difference of residual_i in state_k (the round-off term uses
/// the sum of absolute values of residual_i's terms, with no adjustable
/// "evaluation noise factor").
///
/// Three error terms, every input measured or exact:
///  (1) truncation: (h/2) |f''|, with f'' = d^2 residual_i / d state_k^2 =
///      -0.5 M_ik R''(state_k) EXACTLY (the reaction term is the only
///      nonlinearity in state_k; R'' is the model's known second derivative);
///  (2) evaluation round-off: residual_i is a sum of 3n products, each formed
///      from a mass or stiffness entry that is itself a quadrature sum of n_q
///      products; by Higham (2002, Thm 8.1 / Sec. 3.1) the computed value
///      differs from the exact one by at most delta = (3n + n_q + 4) eps S_i,
///      where S_i is the sum of the ABSOLUTE terms of the residual (what
///      cancels in residual_i does not cancel in its round-off). A forward
///      difference subtracts two such evaluations: 2 delta / h;
///  (3) step rounding: fl(state_k + h) - state_k differs from h by at most
///      eps |state_k|, so the quotient carries |J_ik| eps |state_k| / h.
/// Balancing (1) against (2): h* = 2 sqrt(delta / |f''|), at which the two
/// are equal and E(h*) = 2 sqrt(delta |f''|); term (3) is added at h*. When
/// f'' = 0 (an affine reaction) there is no truncation term and the step is
/// chosen so that (2) + (3) is at most eps^(1/2)-relative: h = sqrt(delta).
struct FiniteDifferenceCheck {
  double step;       ///< the balancing h*
  double tolerance;  ///< E(h*) + the step-rounding term at h*
};
[[nodiscard]] FiniteDifferenceCheck jacobian_finite_difference_tolerance(
    const TransientReactionDiffusionModel& model, double mass_ik, double residual_abs_terms,
    std::size_t n_dofs, std::size_t n_quadrature_points, double state_k, double jacobian_ik);

}  // namespace fuse::physics::fe
