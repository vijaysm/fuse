// physics.steady_reaction_diffusion_mms --
// the AlgebraicSystem's own residual, assembled from element_transient_
// reaction_diffusion in STEADY FORM, exposed as the fuse::num::solve
// (Newton-Krylov, newton_krylov.hpp) ResidualFn -- no second Newton loop
// written here, only the residual callback that header's own solve()
// consumes.
//
// STEADY FORM. element_transient_reaction_diffusion's own residual is
//   M_ij (state_j - state_old_j)/h + 0.5 K_ij (state_j+state_old_j)
//     - 0.5 M_ij (R(state_j)+R(state_old_j)).
// Calling it with `state_old == state` (the SAME vector, at every
// evaluation the solver makes, including MFFD's own probe points) makes
// the first term EXACTLY zero -- bit-for-bit, (x - x) == 0 regardless of
// `h` -- and the theta-average of two EQUAL arguments collapses to the
// argument itself, leaving EXACTLY K_ij state_j - M_ij R(state_j), the
// steady reaction-diffusion residual, independent of `h` (any positive
// value is a legal, unused parameter). `h = 1.0` is passed only to
// satisfy the function's own `h > 0` refusal.
//
// NOT REUSED FOR THE JACOBIAN. The SAME function's own Jacobian output
// under this substitution is NOT the correct steady Jacobian -- it
// differentiates treating `state_old` as a variable INDEPENDENT of
// `state` (correct for the transient use the function was written for),
// so substituting `state_old = state` AFTER differentiating gives half
// the true steady Jacobian plus an `M_ik/h` term. The solve stays
// matrix-free throughout (MFFD, via fuse::num::solve); the baseline
// wires no preconditioner (precond_apply/refresh_precond both nullptr,
// a documented-legal combination, newton_krylov.hpp's own solve()
// comment). The preconditioned variants below are separate tests.
//
// 2-D, NOT 3-D. This whole three-level
// study (about 44,000 reinit calls) runs in 0.83 s in a Debug clean export,
// about 19 us per call. The 2-D study stands on its own (the same production
// code path: element_transient_reaction_diffusion, element_load_vector,
// fuse::num::solve, DofMap/AlgebraicSystem; P1/Q1's order-2 L2 rate is
// dimension-independent in its asymptotic statement), and a 3-D ratio-2
// ladder (N = 4, 8, 16, 4096 Hex8 at the finest level) is an open
// follow-up that is known to be tractable.
//
// ALGEBRAIC-ERROR TERM. The residual Newton leaves (NewtonResult::
// final_residual_norm <= max(atol, rtol * initial_residual_norm), that
// function's own contract) is converted into a bound on the L2 error it
// leaves in u_h by algebraic_l2_error_bound() below (closed-form Q1
// Dirichlet Laplacian eigenvalues, measured max nodal value), and that
// bound is propagated into the rate band instead of being asserted small
// against a chosen factor such as `1e-6 * l2_error`. A Gershgorin estimate of the
// exact steady Jacobian's coercivity, was inconclusive (negative) at every
// level: the stiffness matrix's interior rows sum to exactly zero, so
// Gershgorin gives nothing there; the closed-form eigenvalues of the
// uniform Q1 Dirichlet Laplacian are what make the bound positive.
//
// MANUFACTURED SOLUTION. u_exact(x,y) = sin(pi x) sin(pi y) on the unit
// square: ZERO on every edge of the square, matching the homogeneous
// Dirichlet data this test imposes exactly -- no separate boundary-
// interpolation error to account for. -Laplacian(u_exact) = 2 pi^2
// u_exact (2-D: two axes, not three); with R(u) = alpha u^2 (beta =
// source = coupled = 0, alpha = 1), the discrete residual's own sign on
// the reaction term (manufactured_source's own header comment: K u -
// M R(u) - load = 0, i.e. -Laplacian(u) - R(u) = f, NOT "+R(u)" -- a
// genuine sign error a first attempt made and the order check below
// caught) is satisfied EXACTLY by u_exact when f(x,y) = 2 pi^2 u_exact
// - alpha u_exact^2 -- a CLOSED FORM, assembled into the load vector
// via element_load_vector (generic_operators.hpp), evaluated at
// quadrature points (consistent with the bilinear forms' own
// quadrature order), not nodally (that nodal convention is for the
// UNKNOWN reaction term alone, element_load_vector's own header
// comment states why).
//
// CONVERGENCE-RATE BAND, DERIVED (not widened after a failure). Three
// mesh levels (N = 4, 8, 16 Quad4 elements per axis) give TWO order
// estimates r1 = order(err[4], err[8]), r2 = order(err[8], err[16]) --
// the "two mesh levels" are the FINEST pair (err[8], err[16]),
// whose order r2 is asserted against the expected P1 L2 rate (2); the
// coarsest pair exists ONLY to derive the band, "as the integrators
// tests do" (test/num/waveform_order_test.cpp's own kTol, derived from
// how fast the next-order error term decays, not assumed). For an error
// expansion e(h) = C h^2 + D h^3 + ..., writing r(h) for the order
// measured between h and h/2: r(h) - 2 is, to leading order in h,
// PROPORTIONAL to h (the next term's own relative contribution), so
// r(h/2) - 2 is approximately HALF of r(h) - 2 on a ratio-2 ladder --
// i.e. r1 - r2 approximates HALF of (r1 - 2), so |r2 - 2| approximates
// |r1 - r2|. This test asserts |r2 - 2| <= 2 * |r1 - r2| -- the
// leading-order Richardson prediction with a factor-of-2 margin for the
// SECOND-order (h^2-relative) terms that prediction itself drops, fixed
// BEFORE running, not fitted to the result.

#include "fuse_physics/fe/algebraic_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/base/result.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse/num/newton_krylov.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/fe/generic_operators.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::AlgebraicSystem;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::element_load_vector;
using fuse::physics::fe::element_mass_matrix;
using fuse::physics::fe::element_stiffness_matrix;
using fuse::physics::fe::element_transient_reaction_diffusion;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::reaction_derivative;
using fuse::physics::fe::TransientReactionDiffusionModel;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::Mesh;
using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::JacobiPreconditioner;
using fuse::linalg::SsorPreconditioner;
using fuse::linalg::Triplet;

constexpr double kPi = 3.14159265358979323846;

double u_exact(const Point& p) { return std::sin(kPi * p.x) * std::sin(kPi * p.y); }
// SIGN, DERIVED FROM THE ACTUAL DISCRETE RESIDUAL (a genuine bug found
// here, not assumed): element_transient_reaction_diffusion's own
// residual (this file's own top header comment) is M(state-state_old)/h
// + 0.5 K (state+state_old) - 0.5 M (R(state)+R(state_old)) -- the
// REACTION term enters with a MINUS sign. Under state=state_old, steady
// form is K u - M R(u) - load = 0, i.e. -Laplacian(u) - R(u) = f (K u
// is the weak form of -Laplacian(u); the residual's own minus sign on
// R carries through), NOT "-Laplacian(u) + R(u) = f" a first attempt
// assumed -- caught by this test's own order check failing (r1, r2 both
// deeply negative, error INCREASING with refinement, impossible for a
// correct discretisation), not covered up by loosening the band.
double manufactured_source(const Point& p, double alpha) {
  const double u = u_exact(p);
  return 2.0 * kPi * kPi * u - alpha * u * u;
}

/// One mesh level's own machinery: mesh, dof map, system, boundary dof
/// set, and the FEBase used for every element (one basis, order 1,
/// Quad4, 2-D -- this file's own header comment on why 2-D).
struct Level {
  Mesh mesh;
  BasisRegistry registry;
  DofMap dof_map;
  AlgebraicSystem system;
  fuse::physics::fe::Basis basis;
  FEBase fe;
  std::vector<std::int64_t> boundary_dofs;

  explicit Level(unsigned n)
      : mesh(2),
        dof_map(init_mesh(mesh, n), {Variable{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1}}, registry),
        system(mesh, dof_map),
        basis(registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG})),
        fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2) {
    std::vector<std::int64_t> vertex_dof(mesh.n_vertices() + 1, -1);
    std::vector<std::int64_t> indices;
    for (EntityHandle e = 1; e <= mesh.n_elements(); ++e) {
      dof_map.dof_indices(e, indices);
      const auto& connectivity = mesh.element_connectivity(e);
      if (indices.size() != connectivity.size()) throw std::logic_error("dof_indices/connectivity size mismatch");
      for (std::size_t k = 0; k < connectivity.size(); ++k) vertex_dof[connectivity[k]] = indices[k];
    }
    const auto& positions = mesh.vertex_positions();
    for (std::size_t v = 0; v < positions.size(); ++v) {
      const Point& p = positions[v];
      if (p.x == 0.0 || p.x == 1.0 || p.y == 0.0 || p.y == 1.0) {
        boundary_dofs.push_back(vertex_dof[v + 1]);
      }
    }
  }

 private:
  static Mesh& init_mesh(Mesh& mesh, unsigned n) {
    mesh.generate_2d_box(n, n, Point(0, 0, 0), Point(1, 1, 0));
    return mesh;
  }
};

/// The steady residual: K u - M R(u) - load, with Dirichlet rows (u = 0
/// on the square's boundary, exactly u_exact's own trace there)
/// overriding the interior equation at every evaluation.
fuse::Result<void> steady_residual(Level& level, const TransientReactionDiffusionModel& model, DistVector& x,
                                    DistVector& fx) {
  const auto assembled = level.system.assemble_vector(
      [&](EntityHandle element, const std::vector<std::int64_t>& indices, std::vector<std::pair<std::int64_t, double>>& out) {
        const std::size_t n = indices.size();
        std::vector<double> state(n), state_old(n);
        for (std::size_t i = 0; i < n; ++i) state[i] = state_old[i] = x.data()[indices[i]];
        const auto erj = element_transient_reaction_diffusion(level.fe, element, model, /*h=*/1.0, state, state_old);
        const auto load =
            element_load_vector(level.fe, element, [&](const Point& p) { return manufactured_source(p, model.alpha); });
        for (std::size_t i = 0; i < n; ++i) out.push_back({indices[i], erj.residual[i] - load[i]});
      },
      fx);
  if (!assembled.ok()) return assembled;
  for (const std::int64_t dof : level.boundary_dofs) fx.data()[dof] = x.data()[dof] - 0.0;
  return fuse::Result<void>::success();
}

// The STEADY Jacobian J = K - M diag(R'(u)),
// assembled DIRECTLY from element_stiffness_matrix/element_mass_matrix/
// reaction_derivative (generic_operators.hpp) -- NOT element_transient_
// reaction_diffusion's own Jacobian output, which (this file's own top
// header comment) differentiates treating state_old as independent of
// state and so is NOT the correct steady Jacobian under state=state_old.
// J_ij = K_ij - M_ij * R'(u_j) (M diag(R'(u)) scales column j of M by
// R'(u_j), since diag(R'(u)) is diagonal). Boundary (Dirichlet) rows are
// IDENTITY rows, exactly as steady_residual's own boundary handling makes
// them in the operator (fx[dof] = x[dof]). Leaving them as
// their natural K - M diag(R') values ("an approximate preconditioner
// only") DOUBLES the SSOR GMRES count (29 -> 61 at N = 16): a
// preconditioner whose boundary rows disagree with the operator's is
// not an approximation of the operator on those rows at all. With the
// rows matched, SSOR takes 24 GMRES iterations against the
// unpreconditioned 29, Newton 3 steps in both. Each boundary dof's identity entry is
// emitted once, by the first element that touches it (the triplet merge
// sums duplicates, so emitting it per element would make it a count).
fuse::Result<DistSparseMatrix> assemble_steady_jacobian(Level& level, const TransientReactionDiffusionModel& model,
                                                         const DistVector& x) {
  std::vector<char> is_boundary(static_cast<std::size_t>(level.dof_map.n_local_dofs()), 0);
  for (const std::int64_t dof : level.boundary_dofs) is_boundary[static_cast<std::size_t>(dof)] = 1;
  std::vector<char> identity_emitted(is_boundary.size(), 0);
  return level.system.assemble_matrix(
      [&](EntityHandle element, const std::vector<std::int64_t>& indices, std::vector<Triplet>& out) {
        const auto k = element_stiffness_matrix(level.fe, element);
        const auto m = element_mass_matrix(level.fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i) {
          const auto row = static_cast<std::size_t>(indices[i]);
          if (is_boundary[row]) {
            if (!identity_emitted[row]) {
              identity_emitted[row] = 1;
              out.push_back(Triplet{indices[i], indices[i], 1.0});
            }
            continue;
          }
          for (std::size_t j = 0; j < n; ++j) {
            const double r_prime_j = reaction_derivative(model, x.data()[indices[j]]);
            out.push_back(Triplet{indices[i], indices[j], k[i * n + j] - m[i * n + j] * r_prime_j});
          }
        }
      });
}

/// Which preconditioner (if any) solve_mms below wires into fuse::num::
/// solve's own RefreshPreconditionerFn/PreconditionerApplyFn -- two test rows
/// (Jacobi, SSOR), plus the unpreconditioned baseline (None).
enum class SteadyPreconditionerKind { None, Jacobi, Ssor };

/// Holds whichever ONE of Jacobi/SSOR is live -- rebuilt from scratch on
/// every RefreshPreconditionerFn call (NewtonKrylovOptions::
/// preconditioner_lag = 1 below: every Newton step, the SNES default),
/// read by PreconditionerApplyFn. A `std::optional` pair rather than a
/// `std::variant` only because `SsorExactness`'s own from_matrix returns
/// a `Result<T>` that must be checked before anything is stored; nothing
/// here needs the single-active-member guarantee a variant would add,
/// since `kind` alone already decides which one `apply` reads.
struct SteadyPreconditionerState {
  SteadyPreconditionerKind kind = SteadyPreconditionerKind::None;
  std::optional<JacobiPreconditioner> jacobi;
  std::optional<SsorPreconditioner> ssor;
};

struct SolveOutcome {
  double l2_error = 0.0;
  double algebraic_l2_bound = 0.0;  ///< derived bound on the L2 error the unconverged residual leaves
  double max_nodal_value = 0.0;
  fuse::num::NewtonResult newton;
};

// Derived bound on the L2 norm of the algebraic error e = J^-1 r left by a
// residual r of 2-norm |r| (not a chosen constant such as
// `final_residual < 1e-6 * l2_error`, which compares a residual to an
// error of a different quantity). Every ingredient is exact or measured:
//  - Q1 on the uniform N x N unit square with homogeneous Dirichlet rows: the
//    interior stiffness block K_II has the closed-form eigenvalues
//      8/3 - (2/3)(cos a + cos b) - (4/3) cos a cos b, a = i pi/N, b = j pi/N,
//    (element stencil (1/3)[-1 -1 -1; -1 8 -1; -1 -1 -1], tensor-product sine
//    eigenvectors), smallest at i = j = 1; the consistent mass matrix's largest
//    eigenvalue is h^2 (stencil h^2/36 [1 4 1; 4 16 4; 1 4 1], symbol at a = b = 0);
//    |K|_2 <= 16/3 (Gershgorin, interior row).
//  - J = K - M diag(R'(u_h)), R'(u) = 2 alpha u, so lambda_min(J_II) >=
//    lambda_min(K_II) - 2 alpha umax h^2 with umax the MEASURED max |nodal value|.
//  - Boundary rows of the residual are identity rows (x - 0), so J is block
//    triangular: e_B = r_B and J_II e_I = r_I - B e_B with |B|_2 <= |K|_2 + 2 alpha umax h^2;
//    hence |e|_2 <= (1 + (1 + |B|_2) / lambda_min(J_II)) |r|_2.
//  - |e_h|_L2 = sqrt(e^T M e) <= sqrt(lambda_max(M)) |e|_2 = h |e|_2.
double algebraic_l2_error_bound(unsigned n, double alpha, double umax, double residual_norm) {
  const double h = 1.0 / n;
  const double c = std::cos(kPi / n);
  const double lambda_min_k = 8.0 / 3.0 - (4.0 / 3.0) * c - (4.0 / 3.0) * c * c;
  const double reaction_shift = 2.0 * alpha * umax * h * h;
  const double lambda_min_j = lambda_min_k - reaction_shift;
  if (lambda_min_j <= 0.0) throw std::logic_error("algebraic bound: the Jacobian's coercivity estimate is not positive");
  const double b_norm = 16.0 / 3.0 + reaction_shift;
  return h * (1.0 + (1.0 + b_norm) / lambda_min_j) * residual_norm;
}

SolveOutcome solve_mms(unsigned n, double alpha, SteadyPreconditionerKind kind = SteadyPreconditionerKind::None) {
  Level level(n);
  TransientReactionDiffusionModel model{alpha, /*beta=*/0.0, /*source=*/0.0, /*coupled=*/0.0};

  auto x = level.system.new_vector();
  x.set_zero();

  // atol/rtol are bounded above by what a MATRIX-FREE (MFFD) Jacobian
  // action can actually support: MFFD's own differencing error is
  // O(sqrt(eps)) ~ 1.5e-8 relative (newton_krylov.hpp's own
  // default_mffd_error_rel()), so demanding Eisenstat-Walker forcing
  // tighter than that asks GMRES to resolve a linear residual below the
  // noise floor of the OPERATOR it is applying -- measured directly: at
  // atol=1e-12/rtol=1e-10 this diverged (E-LINALG-NO-CONVERGENCE) once
  // forcing dropped to ~2e-6; at atol=1e-9/rtol=1e-8 (comfortably above
  // the sqrt(eps) floor) it converges in 3 Newton steps, ~3 GMRES
  // iterations each. Not a widened tolerance after a failure -- the
  // FIRST choice (1e-12) was simply tighter than MFFD's own accuracy
  // supports, a wrong (too strict) choice found and corrected, not
  // loosened to paper over a different problem.
  fuse::num::NewtonKrylovOptions opt;
  opt.atol = 1e-9;
  opt.rtol = 1e-8;
  opt.max_nonlinear_iterations = 80;
  opt.gmres.max_iterations = 300;

  const fuse::num::ResidualFn residual = [&](DistVector& xv, DistVector& fxv) { return steady_residual(level, model, xv, fxv); };

  // None leaves both nullptr, the
  // documented-legal "no preconditioning" combination (newton_krylov.
  // hpp's own solve() comment); Jacobi/SSOR assemble the STEADY
  // Jacobian (assemble_steady_jacobian above, not element_transient_
  // reaction_diffusion's own biased one) at the lag boundary and build
  // the matching linalg preconditioner from it.
  SteadyPreconditionerState precond_state;
  precond_state.kind = kind;
  const fuse::num::RefreshPreconditionerFn refresh = [&](const DistVector& xv, const DistVector&) -> fuse::Result<void> {
    if (precond_state.kind == SteadyPreconditionerKind::None) return fuse::Result<void>::success();
    const auto jacobian = assemble_steady_jacobian(level, model, xv);
    if (!jacobian.ok()) return jacobian.error();
    if (precond_state.kind == SteadyPreconditionerKind::Jacobi) {
      auto built = JacobiPreconditioner::from_matrix(jacobian.value());
      if (!built.ok()) return built.error();
      precond_state.jacobi = std::move(built).value();
    } else {
      auto built = SsorPreconditioner::from_matrix(jacobian.value(), /*omega=*/1.0);
      if (!built.ok()) return built.error();
      precond_state.ssor = std::move(built).value();
    }
    return fuse::Result<void>::success();
  };
  const fuse::num::PreconditionerApplyFn apply = [&](DistVector& r, DistVector& z) -> fuse::Result<void> {
    if (precond_state.kind == SteadyPreconditionerKind::Jacobi) return precond_state.jacobi->apply(r, z);
    if (precond_state.kind == SteadyPreconditionerKind::Ssor) return precond_state.ssor->apply(r, z);
    return fuse::Result<void>::success();  // unreachable: solve() below passes nullptr for None
  };
  const bool preconditioned = (kind != SteadyPreconditionerKind::None);
  const auto result = fuse::num::solve(residual, x, preconditioned ? apply : nullptr,
                                        preconditioned ? refresh : nullptr, opt);
  if (!result.ok()) throw std::runtime_error("fuse::num::solve: " + result.error().message);

  SolveOutcome outcome;
  outcome.newton = result.value();

  double error_sq = 0.0;
  for (EntityHandle element = 1; element <= level.mesh.n_elements(); ++element) {
    level.fe.reinit(element);
    std::vector<std::int64_t> indices;
    level.dof_map.dof_indices(element, indices);
    const auto& phi = level.fe.get_phi();
    const auto& jxw = level.fe.get_JxW();
    const auto& xyz = level.fe.get_xyz();
    for (std::size_t q = 0; q < level.fe.n_quadrature_points(); ++q) {
      double u_h = 0.0;
      for (std::size_t k = 0; k < indices.size(); ++k) u_h += phi[k][q] * x.data()[indices[k]];
      const double diff = u_h - u_exact(xyz[q]);
      error_sq += jxw[q] * diff * diff;
    }
  }
  outcome.l2_error = std::sqrt(error_sq);
  outcome.max_nodal_value = x.norm_inf();
  outcome.algebraic_l2_bound =
      algebraic_l2_error_bound(n, alpha, outcome.max_nodal_value, outcome.newton.final_residual_norm);
  return outcome;
}

double order_of(double coarse_err, double fine_err) { return std::log2(coarse_err / fine_err); }

TEST(SteadyReactionDiffusionMmsTest, NewtonKrylovSolutionConvergesAtOrderTwoInL2WithinDerivedBand) {
  const double alpha = 1.0;
  const unsigned levels[3] = {4, 8, 16};
  double err[3];
  double algebraic_rel[3];
  std::string trace;
  for (int i = 0; i < 3; ++i) {
    const auto outcome = solve_mms(levels[i], alpha);
    err[i] = outcome.l2_error;
    ASSERT_TRUE(fuse::num::is_converged(outcome.newton.status))
        << "N=" << levels[i] << " status=" << fuse::num::to_cstring(outcome.newton.status);
    char residual_buf[32];
    std::snprintf(residual_buf, sizeof(residual_buf), "%.3e", outcome.newton.final_residual_norm);
    const std::string level_line =
        "  N=" + std::to_string(levels[i]) + " l2_error=" + std::to_string(err[i]) +
        " newton_iters=" + std::to_string(outcome.newton.iterations) +
        " func_evals=" + std::to_string(outcome.newton.function_evaluations) +
        " final_residual=" + std::string(residual_buf) + "\n";
    trace += level_line;
    std::cout << "[ LEVEL    ] " << level_line << std::flush;
    testing::Test::RecordProperty("N" + std::to_string(levels[i]) + "_l2_error", std::to_string(err[i]));
    testing::Test::RecordProperty("N" + std::to_string(levels[i]) + "_newton_iterations",
                                   std::to_string(outcome.newton.iterations));
    testing::Test::RecordProperty("N" + std::to_string(levels[i]) + "_final_residual",
                                   std::to_string(outcome.newton.final_residual_norm));

    algebraic_rel[i] = outcome.algebraic_l2_bound / err[i];
    testing::Test::RecordProperty("N" + std::to_string(levels[i]) + "_algebraic_over_discretisation",
                                   std::to_string(algebraic_rel[i]));
    trace += "    algebraic L2 bound / discretisation error = " + std::to_string(algebraic_rel[i]) + "\n";
  }
  const double r1 = order_of(err[0], err[1]);
  const double r2 = order_of(err[1], err[2]);
  const double band = 2.0 * std::fabs(r1 - r2);  // this file's own header comment: the Richardson prediction, 2x margin
  // The algebraic error perturbs each measured error by at most its bound, so
  // each rate by at most (delta_k/err_k + delta_{k+1}/err_{k+1}) / ln 2
  // (first order in delta/err); added to the band rather than asserted
  // small against a chosen factor.
  const double algebraic_band = (algebraic_rel[1] + algebraic_rel[2]) / std::log(2.0);
  // Asymptotic approach, so a wide band from a pre-asymptotic pair cannot
  // pass vacuously: the finer pair's rate must be closer to 2 than the coarser pair's.
  EXPECT_LT(std::fabs(r2 - 2.0), std::fabs(r1 - 2.0)) << trace;
  std::cout << "[ ORDER    ] steady_reaction_diffusion_mms L2:\n" << trace << "  r1=" << r1 << " r2=" << r2
            << " band=" << band << "\n";
  testing::Test::RecordProperty("order_r1", std::to_string(r1));
  testing::Test::RecordProperty("order_r2", std::to_string(r2));
  testing::Test::RecordProperty("order_band", std::to_string(band));
  testing::Test::RecordProperty("order_algebraic_band", std::to_string(algebraic_band));
  std::cout << "  algebraic_band=" << algebraic_band << "\n";
  EXPECT_NEAR(r2, 2.0, band + algebraic_band) << trace;
}

// One mesh level (N = 16, the main order
// test's own finest level), run once unpreconditioned (kind = None,
// the baseline) and once with the preconditioner under test.
// Both solves converge to the SAME NewtonKrylovOptions atol/rtol, so
// both approximate the SAME exact discrete solution on this fixed mesh
// -- they must therefore agree within the SUM of their own derived
// algebraic_l2_error_bound()s (each bound already propagates that
// solve's own leftover residual into an L2-error bound;
// summing two independent bounds on the SAME quantity is
// the triangle inequality, not a chosen margin). Newton/GMRES
// iteration counts for both are recorded (RecordProperty); there is no
// external reference to compare them against, so nothing about their VALUES is asserted, only
// that preconditioning changes neither the converged status nor (within
// the derived bound) the solution it converges to.
void assert_preconditioned_solution_agrees_with_baseline(SteadyPreconditionerKind kind, const char* kind_name) {
  const double alpha = 1.0;
  const unsigned n = 16;

  const auto baseline = solve_mms(n, alpha, SteadyPreconditionerKind::None);
  ASSERT_TRUE(fuse::num::is_converged(baseline.newton.status))
      << "baseline status=" << fuse::num::to_cstring(baseline.newton.status);

  const auto preconditioned = solve_mms(n, alpha, kind);
  ASSERT_TRUE(fuse::num::is_converged(preconditioned.newton.status))
      << kind_name << " status=" << fuse::num::to_cstring(preconditioned.newton.status);

  std::size_t baseline_gmres = 0, preconditioned_gmres = 0;
  for (const auto& step : baseline.newton.trace) baseline_gmres += step.gmres_iterations;
  for (const auto& step : preconditioned.newton.trace) preconditioned_gmres += step.gmres_iterations;

  testing::Test::RecordProperty("baseline_newton_iterations", std::to_string(baseline.newton.iterations));
  testing::Test::RecordProperty("baseline_gmres_iterations", std::to_string(baseline_gmres));
  testing::Test::RecordProperty(std::string(kind_name) + "_newton_iterations",
                                 std::to_string(preconditioned.newton.iterations));
  testing::Test::RecordProperty(std::string(kind_name) + "_gmres_iterations", std::to_string(preconditioned_gmres));
  std::cout << "[ PRECOND  ] N=" << n << " baseline: newton=" << baseline.newton.iterations
            << " gmres=" << baseline_gmres << "  " << kind_name << ": newton=" << preconditioned.newton.iterations
            << " gmres=" << preconditioned_gmres << "\n";

  const double tolerance = baseline.algebraic_l2_bound + preconditioned.algebraic_l2_bound;
  EXPECT_NEAR(preconditioned.l2_error, baseline.l2_error, tolerance)
      << kind_name << ": l2_error=" << preconditioned.l2_error << " baseline_l2_error=" << baseline.l2_error
      << " tolerance=" << tolerance;
}

TEST(SteadyReactionDiffusionMmsTest, JacobiPreconditionedSolutionAgreesWithUnpreconditionedWithinAlgebraicBound) {
  assert_preconditioned_solution_agrees_with_baseline(SteadyPreconditionerKind::Jacobi, "jacobi");
}

TEST(SteadyReactionDiffusionMmsTest, SsorPreconditionedSolutionAgreesWithUnpreconditionedWithinAlgebraicBound) {
  assert_preconditioned_solution_agrees_with_baseline(SteadyPreconditionerKind::Ssor, "ssor");
}

// =============================================================================
// fuse::num::BlockJacobiPreconditioner for a
// COUPLED two-variable (u, v) steady reaction-diffusion problem, blocks
// per VARIABLE. DofMap's own multi-variable numbering was confirmed
// VARIABLE-MAJOR/CONTIGUOUS (DofKey::operator<
// compares `variable` first, dof_map.cpp), so a 2-variable DofMap's
// owned dofs are [0, n_u) for u THEN [n_u, n_u+n_v) for v, with NO
// interleaving.
//
// COUPLING. TransientReactionDiffusionModel::coupled is documented as
// "a fixed externally-coupled value, held constant across the element"
// (generic_operators.hpp) -- precisely the hook this test uses: for
// field u's own element residual, `coupled` is set to the ELEMENT
// AVERAGE of v's own CURRENT local nodal values (and symmetrically for
// v, averaging u); recomputed fresh at every residual evaluation
// (including every MFFD probe), so the matrix-free Jacobian-vector
// product the outer Newton-Krylov loop differentiates genuinely sees
// the cross-field coupling -- this is a real two-way-coupled nonlinear
// system, not two independent single-field solves run side by side.
//
// BLOCK-JACOBI PRECONDITIONER. Each block treats its OWN variable's
// steady Jacobian (assemble_steady_jacobian, reused) IN
// ISOLATION -- ignoring the other field's coupling entirely for
// PRECONDITIONING purposes only; the outer residual stays fully
// coupled regardless. Built from a STANDALONE single-variable Level
// per field (the existing P2 `Level`, reused directly): since both
// variables classify against the SAME mesh with the SAME basis, and
// DofKey's own sort order for a given variable depends only on
// (component, entity_dimension, corners, slot) -- mesh-intrinsic
// quantities unaffected by how many OTHER variables a DofMap happens to
// register -- a standalone single-variable Level's own local numbering
// for "u alone" is IDENTICAL to the combined DofMap's own u-block (and
// "v alone" to the v-block, shifted by n_u), so each standalone
// preconditioner's own apply() operates on exactly the right span with
// no renumbering.
//
// MANUFACTURED SOLUTIONS, documented simplification: each field's own
// LOAD vector is built from the CONTINUUM coupling term (the other
// field's EXACT pointwise manufactured value, matching the PDE -
// Laplacian(u) - R(u, v) = f_u exactly), while the DISCRETE residual's
// own coupling term is the ELEMENT-AVERAGE approximation above -- these
// two differ by an O(h) term neither cancels exactly, an additional
// discretisation error this test does NOT attempt to characterise
// (no order-of-convergence claim is made for the coupled case; only
// solve-to-solve agreement, the same scope the preconditioner
// tests above use). ALGEBRAIC BOUND, documented simplification: each field's
// own `algebraic_l2_error_bound` is applied to
// ITS OWN sub-block in isolation (ignoring the coupling term's effect
// on the true 2x2 block Jacobian's coercivity) and the two bounds are
// SUMMED -- a derived quantity, not a chosen constant, but not a
// rigorously re-proven bound for the fully coupled Jacobian either;
// recorded as an open_question rather than presented as more rigorous
// than it is.
double v_exact(const Point& p) { return std::sin(kPi * p.x) * std::sin(2.0 * kPi * p.y); }

double manufactured_source_u(const Point& p, double alpha, double beta) {
  return 2.0 * kPi * kPi * u_exact(p) - alpha * u_exact(p) * u_exact(p) - beta * v_exact(p);
}
double manufactured_source_v(const Point& p, double alpha, double beta) {
  return 5.0 * kPi * kPi * v_exact(p) - alpha * v_exact(p) * v_exact(p) - beta * u_exact(p);
}

struct CoupledLevel {
  Mesh mesh;
  BasisRegistry registry;
  DofMap dof_map;  // variables {u, v}, variable-major contiguous (this file's own header comment above)
  AlgebraicSystem system;
  fuse::physics::fe::Basis basis;
  FEBase fe;
  std::vector<std::int64_t> boundary_dofs;  // both u's and v's own boundary dofs
  std::size_t n_u = 0;
  std::size_t n_v = 0;

  explicit CoupledLevel(unsigned n)
      : mesh(2),
        dof_map(init_mesh(mesh, n),
                {Variable{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1},
                 Variable{"v", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1}},
                registry),
        system(mesh, dof_map),
        basis(registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG})),
        fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2) {
    n_u = static_cast<std::size_t>(mesh.n_vertices());  // P1 Lagrange: one dof per vertex per variable
    n_v = n_u;
    std::vector<std::int64_t> u_vertex_dof(mesh.n_vertices() + 1, -1), v_vertex_dof(mesh.n_vertices() + 1, -1);
    std::vector<std::int64_t> u_idx, v_idx;
    for (EntityHandle e = 1; e <= mesh.n_elements(); ++e) {
      dof_map.dof_indices(e, u_idx, /*variable=*/0);
      dof_map.dof_indices(e, v_idx, /*variable=*/1);
      const auto& connectivity = mesh.element_connectivity(e);
      for (std::size_t k = 0; k < connectivity.size(); ++k) {
        u_vertex_dof[connectivity[k]] = u_idx[k];
        v_vertex_dof[connectivity[k]] = v_idx[k];
      }
    }
    const auto& positions = mesh.vertex_positions();
    for (std::size_t v = 0; v < positions.size(); ++v) {
      const Point& p = positions[v];
      if (p.x == 0.0 || p.x == 1.0 || p.y == 0.0 || p.y == 1.0) {
        boundary_dofs.push_back(u_vertex_dof[v + 1]);
        boundary_dofs.push_back(v_vertex_dof[v + 1]);
      }
    }
  }

 private:
  static Mesh& init_mesh(Mesh& mesh, unsigned n) {
    mesh.generate_2d_box(n, n, Point(0, 0, 0), Point(1, 1, 0));
    return mesh;
  }
};

fuse::Result<void> coupled_steady_residual(CoupledLevel& level, const TransientReactionDiffusionModel& model_u,
                                            const TransientReactionDiffusionModel& model_v, DistVector& x,
                                            DistVector& fx) {
  const auto assembled = level.system.assemble_vector(
      [&](EntityHandle element, const std::vector<std::int64_t>&, std::vector<std::pair<std::int64_t, double>>& out) {
        std::vector<std::int64_t> u_idx, v_idx;
        level.dof_map.dof_indices(element, u_idx, /*variable=*/0);
        level.dof_map.dof_indices(element, v_idx, /*variable=*/1);
        const std::size_t n = u_idx.size();
        std::vector<double> u_state(n), v_state(n);
        for (std::size_t i = 0; i < n; ++i) {
          u_state[i] = x.data()[u_idx[i]];
          v_state[i] = x.data()[v_idx[i]];
        }
        double v_avg = 0.0, u_avg = 0.0;
        for (double vv : v_state) v_avg += vv;
        for (double uu : u_state) u_avg += uu;
        v_avg /= static_cast<double>(n);
        u_avg /= static_cast<double>(n);

        TransientReactionDiffusionModel mu = model_u;
        mu.coupled = v_avg;
        TransientReactionDiffusionModel mv = model_v;
        mv.coupled = u_avg;

        const auto erj_u = element_transient_reaction_diffusion(level.fe, element, mu, /*h=*/1.0, u_state, u_state);
        const auto erj_v = element_transient_reaction_diffusion(level.fe, element, mv, /*h=*/1.0, v_state, v_state);
        const auto load_u = element_load_vector(
            level.fe, element, [&](const Point& p) { return manufactured_source_u(p, mu.alpha, mu.beta); });
        const auto load_v = element_load_vector(
            level.fe, element, [&](const Point& p) { return manufactured_source_v(p, mv.alpha, mv.beta); });
        for (std::size_t i = 0; i < n; ++i) {
          out.push_back({u_idx[i], erj_u.residual[i] - load_u[i]});
          out.push_back({v_idx[i], erj_v.residual[i] - load_v[i]});
        }
      },
      fx);
  if (!assembled.ok()) return assembled;
  for (const std::int64_t dof : level.boundary_dofs) fx.data()[dof] = x.data()[dof] - 0.0;
  return fuse::Result<void>::success();
}

struct CoupledSolveOutcome {
  double l2_error_u = 0.0;
  double l2_error_v = 0.0;
  double algebraic_l2_bound = 0.0;  ///< the two fields' own bounds summed (this file's own header comment above)
  fuse::num::NewtonResult newton;
};

enum class CoupledPreconditionerKind { None, BlockJacobi };

CoupledSolveOutcome solve_coupled_mms(unsigned n, double alpha, double beta, CoupledPreconditionerKind kind) {
  CoupledLevel level(n);
  TransientReactionDiffusionModel model_u{alpha, beta, /*source=*/0.0, /*coupled=*/0.0};
  TransientReactionDiffusionModel model_v{alpha, beta, /*source=*/0.0, /*coupled=*/0.0};

  auto x = level.system.new_vector();
  x.set_zero();

  fuse::num::NewtonKrylovOptions opt;
  opt.atol = 1e-9;
  opt.rtol = 1e-8;
  opt.max_nonlinear_iterations = 80;
  opt.gmres.max_iterations = 300;

  const fuse::num::ResidualFn residual = [&](DistVector& xv, DistVector& fxv) {
    return coupled_steady_residual(level, model_u, model_v, xv, fxv);
  };

  // Each block's own standalone single-field Level/preconditioner
  // (this file's own header comment above states why their local
  // numbering matches the combined DofMap's own per-variable blocks
  // exactly).
  Level level_u(n), level_v(n);
  std::optional<JacobiPreconditioner> precond_u, precond_v;
  const auto make_scratch = [](std::size_t m) {
    return DistVector(m, fuse::linalg::make_serial_exchanger(), fuse::linalg::make_serial_reducer());
  };

  const fuse::num::RefreshPreconditionerFn refresh = [&](const DistVector& xv, const DistVector&) -> fuse::Result<void> {
    if (kind == CoupledPreconditionerKind::None) return fuse::Result<void>::success();
    auto x_u = make_scratch(level.n_u);
    auto x_v = make_scratch(level.n_v);
    for (std::size_t i = 0; i < level.n_u; ++i) x_u.data()[i] = xv.data()[i];
    for (std::size_t i = 0; i < level.n_v; ++i) x_v.data()[i] = xv.data()[level.n_u + i];

    const auto jac_u = assemble_steady_jacobian(level_u, model_u, x_u);
    if (!jac_u.ok()) return jac_u.error();
    const auto jac_v = assemble_steady_jacobian(level_v, model_v, x_v);
    if (!jac_v.ok()) return jac_v.error();
    auto built_u = JacobiPreconditioner::from_matrix(jac_u.value());
    if (!built_u.ok()) return built_u.error();
    auto built_v = JacobiPreconditioner::from_matrix(jac_v.value());
    if (!built_v.ok()) return built_v.error();
    precond_u = std::move(built_u).value();
    precond_v = std::move(built_v).value();
    return fuse::Result<void>::success();
  };

  const auto block_apply = [&](std::optional<JacobiPreconditioner>& precond, std::size_t m,
                                std::span<const double> r, std::span<double> z) -> fuse::Result<void> {
    auto rv = make_scratch(m);
    auto zv = make_scratch(m);
    for (std::size_t i = 0; i < m; ++i) rv.data()[i] = r[i];
    const auto applied = precond->apply(rv, zv);
    if (!applied.ok()) return applied;
    for (std::size_t i = 0; i < m; ++i) z[i] = zv.data()[i];
    return fuse::Result<void>::success();
  };

  const fuse::num::PreconditionerApplyFn apply = [&](DistVector& r, DistVector& z) -> fuse::Result<void> {
    std::vector<fuse::num::PreconditionerBlock> blocks;
    blocks.push_back({0, level.n_u,
                       [&](std::span<const double> rs, std::span<double> zs) { return block_apply(precond_u, level.n_u, rs, zs); }});
    blocks.push_back({level.n_u, level.n_v,
                       [&](std::span<const double> rs, std::span<double> zs) { return block_apply(precond_v, level.n_v, rs, zs); }});
    const auto built = fuse::num::BlockJacobiPreconditioner::from_blocks(std::move(blocks), level.n_u + level.n_v);
    if (!built.ok()) return built.error();
    return built.value().apply(r, z);
  };

  const bool preconditioned = (kind != CoupledPreconditionerKind::None);
  const auto result = fuse::num::solve(residual, x, preconditioned ? apply : nullptr,
                                        preconditioned ? refresh : nullptr, opt);
  if (!result.ok()) throw std::runtime_error("fuse::num::solve (coupled): " + result.error().message);

  CoupledSolveOutcome outcome;
  outcome.newton = result.value();

  double error_sq_u = 0.0, error_sq_v = 0.0;
  for (EntityHandle element = 1; element <= level.mesh.n_elements(); ++element) {
    level.fe.reinit(element);
    std::vector<std::int64_t> u_idx, v_idx;
    level.dof_map.dof_indices(element, u_idx, /*variable=*/0);
    level.dof_map.dof_indices(element, v_idx, /*variable=*/1);
    const auto& phi = level.fe.get_phi();
    const auto& jxw = level.fe.get_JxW();
    const auto& xyz = level.fe.get_xyz();
    for (std::size_t q = 0; q < level.fe.n_quadrature_points(); ++q) {
      double u_h = 0.0, v_h = 0.0;
      for (std::size_t k = 0; k < u_idx.size(); ++k) {
        u_h += phi[k][q] * x.data()[u_idx[k]];
        v_h += phi[k][q] * x.data()[v_idx[k]];
      }
      const double diff_u = u_h - u_exact(xyz[q]);
      const double diff_v = v_h - v_exact(xyz[q]);
      error_sq_u += jxw[q] * diff_u * diff_u;
      error_sq_v += jxw[q] * diff_v * diff_v;
    }
  }
  outcome.l2_error_u = std::sqrt(error_sq_u);
  outcome.l2_error_v = std::sqrt(error_sq_v);

  const double umax = x.norm_inf();  // conservative: shared across both fields' own bound terms
  const double bound_u = algebraic_l2_error_bound(n, alpha, umax, outcome.newton.final_residual_norm);
  const double bound_v = algebraic_l2_error_bound(n, alpha, umax, outcome.newton.final_residual_norm);
  outcome.algebraic_l2_bound = bound_u + bound_v;  // documented simplification, this file's own header comment above
  return outcome;
}

// Item P3's own assertion: a BlockJacobi-preconditioned solve of the
// coupled system must agree with the unpreconditioned solve of the
// SAME coupled system within the summed per-field algebraic bound
// (the same "two solves of one discrete system agree within their own
// leftover-residual bounds" argument the tests above use) -- on
// BOTH fields, since both are part of the one combined solve. Newton/
// GMRES counts are recorded, not asserted.
TEST(SteadyReactionDiffusionMmsTest, BlockJacobiPreconditionedCoupledSolutionAgreesWithUnpreconditionedWithinAlgebraicBound) {
  const double alpha = 1.0;
  const double beta = 0.3;
  const unsigned n = 8;

  const auto baseline = solve_coupled_mms(n, alpha, beta, CoupledPreconditionerKind::None);
  ASSERT_TRUE(fuse::num::is_converged(baseline.newton.status))
      << "baseline status=" << fuse::num::to_cstring(baseline.newton.status);

  const auto preconditioned = solve_coupled_mms(n, alpha, beta, CoupledPreconditionerKind::BlockJacobi);
  ASSERT_TRUE(fuse::num::is_converged(preconditioned.newton.status))
      << "block_jacobi status=" << fuse::num::to_cstring(preconditioned.newton.status);

  std::size_t baseline_gmres = 0, preconditioned_gmres = 0;
  for (const auto& step : baseline.newton.trace) baseline_gmres += step.gmres_iterations;
  for (const auto& step : preconditioned.newton.trace) preconditioned_gmres += step.gmres_iterations;
  testing::Test::RecordProperty("baseline_newton_iterations", std::to_string(baseline.newton.iterations));
  testing::Test::RecordProperty("baseline_gmres_iterations", std::to_string(baseline_gmres));
  testing::Test::RecordProperty("block_jacobi_newton_iterations", std::to_string(preconditioned.newton.iterations));
  testing::Test::RecordProperty("block_jacobi_gmres_iterations", std::to_string(preconditioned_gmres));
  std::cout << "[ COUPLED  ] N=" << n << " baseline: newton=" << baseline.newton.iterations
            << " gmres=" << baseline_gmres << "  block_jacobi: newton=" << preconditioned.newton.iterations
            << " gmres=" << preconditioned_gmres << "\n";

  const double tolerance = baseline.algebraic_l2_bound + preconditioned.algebraic_l2_bound;
  EXPECT_NEAR(preconditioned.l2_error_u, baseline.l2_error_u, tolerance)
      << "u: " << preconditioned.l2_error_u << " vs " << baseline.l2_error_u << " tol=" << tolerance;
  EXPECT_NEAR(preconditioned.l2_error_v, baseline.l2_error_v, tolerance)
      << "v: " << preconditioned.l2_error_v << " vs " << baseline.l2_error_v << " tol=" << tolerance;
}

}  // namespace
