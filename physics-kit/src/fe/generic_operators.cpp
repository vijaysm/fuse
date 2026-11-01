// fuse_physics::fe::element_mass_matrix / element_stiffness_matrix /
// element_transient_reaction_diffusion -- adapted from KARMA's
// include/physics/common/generic_mass_operator.h and
// test/transient_reaction_diffusion_assembly_test.C. See generic_
// operators.hpp's own header comment for what is and is not a literal
// port here.

#include "fuse_physics/fe/generic_operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fuse::physics::fe {

namespace {
[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }
}  // namespace

std::vector<double> element_mass_matrix(FEBase& fe, spatial::EntityHandle element) {
  fe.reinit(element);
  const auto& phi = fe.get_phi();
  const auto& jxw = fe.get_JxW();
  const std::size_t n = phi.size();
  std::vector<double> mass(n * n, 0.0);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    const double weight = jxw[q];
    for (std::size_t i = 0; i < n; ++i) {
      const double weighted_phi_i = weight * phi[i][q];
      for (std::size_t j = 0; j < n; ++j) mass[i * n + j] += weighted_phi_i * phi[j][q];
    }
  }
  return mass;
}

std::vector<double> element_stiffness_matrix(FEBase& fe, spatial::EntityHandle element) {
  fe.reinit(element);
  const auto& dphi = fe.get_dphi();
  const auto& jxw = fe.get_JxW();
  const std::size_t n = dphi.size();
  std::vector<double> stiffness(n * n, 0.0);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    const double weight = jxw[q];
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) stiffness[i * n + j] += weight * dot(dphi[i][q], dphi[j][q]);
    }
  }
  return stiffness;
}

std::vector<double> element_load_vector(FEBase& fe, spatial::EntityHandle element,
                                         const std::function<double(const Point&)>& source) {
  fe.reinit(element);
  const auto& phi = fe.get_phi();
  const auto& jxw = fe.get_JxW();
  const auto& xyz = fe.get_xyz();
  const std::size_t n = phi.size();
  std::vector<double> load(n, 0.0);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    const double weighted_source = jxw[q] * source(xyz[q]);
    for (std::size_t i = 0; i < n; ++i) load[i] += weighted_source * phi[i][q];
  }
  return load;
}


ElementResidualJacobian element_transient_reaction_diffusion(FEBase& fe, spatial::EntityHandle element,
                                                               const TransientReactionDiffusionModel& model,
                                                               double h, const std::vector<double>& state,
                                                               const std::vector<double>& state_old) {
  if (h <= 0.0) refuse("element_transient_reaction_diffusion: the time step must be positive");
  const auto mass = element_mass_matrix(fe, element);
  const auto stiffness = element_stiffness_matrix(fe, element);
  const std::size_t n = state.size();
  if (state_old.size() != n || mass.size() != n * n)
    refuse("element_transient_reaction_diffusion: state/state_old/basis size mismatch");

  ElementResidualJacobian result;
  result.residual.assign(n, 0.0);
  result.jacobian.assign(n * n, 0.0);
  result.residual_abs_terms.assign(n, 0.0);

  std::vector<double> reaction_new(n), reaction_old(n), reaction_derivative_new(n);
  for (std::size_t j = 0; j < n; ++j) {
    reaction_new[j] = reaction(model, state[j]);
    reaction_old[j] = reaction(model, state_old[j]);
    reaction_derivative_new[j] = reaction_derivative(model, state[j]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    double residual_i = 0.0, abs_terms = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      const double m_ij = mass[i * n + j];
      const double k_ij = stiffness[i * n + j];
      const double t1 = m_ij * (state[j] - state_old[j]) / h;
      const double t2 = 0.5 * k_ij * (state[j] + state_old[j]);
      const double t3 = 0.5 * m_ij * (reaction_new[j] + reaction_old[j]);
      residual_i += t1;
      residual_i += t2;
      residual_i -= t3;
      abs_terms += std::abs(t1) + std::abs(t2) + std::abs(t3);
    }
    result.residual[i] = residual_i;
    result.residual_abs_terms[i] = abs_terms;
    for (std::size_t k = 0; k < n; ++k) {
      const double m_ik = mass[i * n + k];
      const double k_ik = stiffness[i * n + k];
      result.jacobian[i * n + k] = m_ik / h + 0.5 * k_ik - 0.5 * m_ik * reaction_derivative_new[k];
    }
  }
  return result;
}

FiniteDifferenceCheck jacobian_finite_difference_tolerance(const TransientReactionDiffusionModel& model,
                                                             double mass_ik, double residual_abs_terms,
                                                             std::size_t n_dofs, std::size_t n_quadrature_points,
                                                             double state_k, double jacobian_ik) {
  // See generic_operators.hpp for the derivation and the numbering of the terms.
  const double eps = std::numeric_limits<double>::epsilon();
  const double ops = static_cast<double>(3 * n_dofs + n_quadrature_points + 4);
  const double delta = ops * eps * residual_abs_terms;  // (2) one evaluation's round-off
  const double second_derivative = std::abs(0.5 * mass_ik * reaction_second_derivative(model));  // (1) exact
  double step = 0.0, bound = 0.0;
  if (second_derivative > 0.0 && delta > 0.0) {
    step = 2.0 * std::sqrt(delta / second_derivative);
    bound = 2.0 * std::sqrt(delta * second_derivative);
  } else {
    // Affine reaction (or a zero residual row): no truncation term; the step
    // is taken as sqrt(delta) so that 2 delta / h = 2 sqrt(delta).
    step = std::sqrt(std::max(delta, eps));
    bound = 2.0 * delta / step;
  }
  bound += std::abs(jacobian_ik) * eps * std::abs(state_k) / step;  // (3) step rounding
  return {step, bound};
}

}  // namespace fuse::physics::fe
