// fuse_physics::fe::DiffusionReactionSystem. The header comment of
// diffusion_reaction_system.hpp states the semi-discrete system, its two
// integrator-facing forms and the treatment of Dirichlet rows.

#include "fuse_physics/fe/diffusion_reaction_system.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/gmres.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse_physics/fe/boundary_condition.hpp"
#include "fuse_physics/fe/generic_operators.hpp"

namespace fuse::physics::fe {

namespace {
[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }
}  // namespace

DiffusionReactionSystem::DiffusionReactionSystem(const spatial::Mesh& mesh, const spatial::Domain& domain,
                                                  const DofMap& dof_map, Basis basis, GeometryVariant geometry_variant,
                                                  int quadrature_degree)
    : mesh_(&mesh),
      domain_(&domain),
      dof_map_(&dof_map),
      basis_(basis),
      geometry_variant_(geometry_variant),
      fe_(mesh, geometry_variant, basis, quadrature_degree, FEIntegrationDomain::Volume),
      face_fe_(mesh, geometry_variant, basis, quadrature_degree, FEIntegrationDomain::Face),
      system_(mesh, dof_map),
      boundary_(mesh, domain) {
  if (dof_map.n_variables() != 1)
    refuse(
        "DiffusionReactionSystem: dof_map must carry exactly one scalar variable "
        "(E-PHYSICS-DRS-NOT-SCALAR)");
}

void DiffusionReactionSystem::set_boundary_condition(BoundaryId boundary_id, BoundaryCondition condition) {
  boundary_.set(boundary_id, std::move(condition));
}

void DiffusionReactionSystem::initialize() {
  boundary_.resolve(*dof_map_, basis_, 0);

  faces_by_element_.clear();
  for (const auto& face : boundary_.neumann_faces()) faces_by_element_[face.element].push_back(&face);
  for (const auto& face : boundary_.robin_faces()) faces_by_element_[face.element].push_back(&face);

  dirichlet_by_dof_.clear();
  for (const auto& d : boundary_.dirichlet_dofs()) dirichlet_by_dof_[d.dof] = &d;

  // Built ONCE here, not per residual call: this class's own header
  // comment on dof_position_ states why a per-call recomputation turns
  // an O(n_elements) setup cost into the dominant cost of an entire
  // Newton-Krylov solve.
  dof_physical_positions(*mesh_, *dof_map_, 0, basis_, dof_position_);

  // Two cached matrices (header comment, "CACHED MASS MATRICES"): the
  // consistent mass matrix M itself, and M with every Dirichlet row and
  // column replaced by the identity (symmetric positive definite, the
  // matrix the explicit form solves with).
  {
    auto full = system_.assemble_matrix(
        [&](spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
            std::vector<linalg::Triplet>& out) {
          const std::size_t n = indices.size();
          fe_.reinit(element);
          const auto mass_local = element_mass_matrix(fe_, element);
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
              out.push_back(linalg::Triplet{indices[i], indices[j], mass_local[i * n + j]});
        });
    if (!full.ok())
      refuse(("DiffusionReactionSystem::initialize: mass-matrix assembly failed: " + full.error().message).c_str());
    mass_matrix_ = std::move(full.value());

    std::vector<char> identity_emitted(static_cast<std::size_t>(dof_map_->n_dofs()), 0);
    auto eliminated = system_.assemble_matrix(
        [&](spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
            std::vector<linalg::Triplet>& out) {
          const std::size_t n = indices.size();
          fe_.reinit(element);
          const auto mass_local = element_mass_matrix(fe_, element);
          for (std::size_t i = 0; i < n; ++i) {
            const auto row = static_cast<std::size_t>(indices[i]);
            if (dirichlet_by_dof_.find(indices[i]) != dirichlet_by_dof_.end()) {
              if (!identity_emitted[row]) {
                identity_emitted[row] = 1;
                out.push_back(linalg::Triplet{indices[i], indices[i], 1.0});
              }
              continue;
            }
            for (std::size_t j = 0; j < n; ++j) {
              if (dirichlet_by_dof_.find(indices[j]) != dirichlet_by_dof_.end()) continue;
              out.push_back(linalg::Triplet{indices[i], indices[j], mass_local[i * n + j]});
            }
          }
        });
    if (!eliminated.ok())
      refuse(("DiffusionReactionSystem::initialize: mass-matrix assembly failed: " + eliminated.error().message)
                 .c_str());
    eliminated_mass_matrix_ = std::move(eliminated.value());
    auto precond = fuse::linalg::JacobiPreconditioner::from_matrix(*eliminated_mass_matrix_);
    if (!precond.ok())
      refuse(("DiffusionReactionSystem::initialize: mass-matrix preconditioner setup failed: " +
              precond.error().message)
                 .c_str());
    mass_preconditioner_ = std::move(precond.value());
  }

  initialized_ = true;
}

namespace {
/// `op(u)_i = sum_j K_ij(k) u_j + sum_j M_ij R(x_j, u_j) - load_i`, the
/// SHARED per-element computation `raw_spatial_operator` below folds
/// Neumann/Robin face terms into. Free function (not a member) so it
/// can be called from the `AlgebraicSystem::assemble_vector`
/// contribution lambda with no additional captures beyond what that
/// lambda already needs.
void accumulate_element_contribution(FEBase& fe, FEBase& face_fe, spatial::EntityHandle element,
                                      const std::vector<std::int64_t>& indices, const std::vector<Point>& dof_position,
                                      const std::vector<const ResolvedFace*>* faces,
                                      const DiffusionReactionCoefficients& coefficients, const ForcingTerm& forcing,
                                      double t, const double* y, std::vector<double>& local) {
  const std::size_t n = indices.size();
  std::vector<double> state(n);
  for (std::size_t i = 0; i < n; ++i) state[i] = y[indices[i]];
  local.assign(n, 0.0);

  fe.reinit(element);
  const auto& phi = fe.get_phi();
  const auto& dphi = fe.get_dphi();
  const auto& jxw = fe.get_JxW();
  const auto& xyz = fe.get_xyz();
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    double u_q = 0.0;
    for (std::size_t j = 0; j < n; ++j) u_q += phi[j][q] * state[j];
    const double k_q = coefficients.diffusion(xyz[q], u_q);
    const double source_q = forcing(xyz[q], t);
    const double w = jxw[q];
    for (std::size_t i = 0; i < n; ++i) {
      double grad_term = 0.0;
      for (std::size_t j = 0; j < n; ++j) grad_term += dot(dphi[i][q], dphi[j][q]) * state[j];
      local[i] += w * (k_q * grad_term - phi[i][q] * source_q);
    }
  }

  // Nodal reaction, scattered through the element mass matrix (the
  // established convention generic_operators.hpp's own
  // element_transient_reaction_diffusion uses), evaluated at each
  // dof's own interpolation-point physical position -- valid for any
  // basis order, not only a vertex dof.
  std::vector<double> reaction_nodal(n);
  for (std::size_t j = 0; j < n; ++j) reaction_nodal[j] = coefficients.reaction(dof_position[indices[j]], state[j]);
  for (std::size_t i = 0; i < n; ++i) {
    double m_r = 0.0;
    for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
      double row = 0.0;
      for (std::size_t j = 0; j < n; ++j) row += phi[j][q] * reaction_nodal[j];
      m_r += jxw[q] * phi[i][q] * row;
    }
    local[i] += m_r;
  }

  if (faces == nullptr) return;
  for (const ResolvedFace* face : *faces) {
    face_fe.reinit(element, face->side);
    const auto& fphi = face_fe.get_phi();
    const auto& fjxw = face_fe.get_JxW();
    const auto& fxyz = face_fe.get_xyz();
    const bool is_robin = face->condition->kind == BoundaryConditionKind::Robin;
    const double c = face->condition->robin_coefficient;
    for (std::size_t q = 0; q < face_fe.n_quadrature_points(); ++q) {
      const double g = face->condition->value(fxyz[q], t);
      const double w = fjxw[q];
      double u_face_q = 0.0;
      if (is_robin) {
        for (std::size_t j = 0; j < n; ++j) u_face_q += fphi[j][q] * state[j];
      }
      for (std::size_t i = 0; i < n; ++i) {
        local[i] -= w * fphi[i][q] * g;
        if (is_robin) local[i] += w * c * fphi[i][q] * u_face_q;
      }
    }
  }
}

}  // namespace

fuse::Result<fuse::linalg::DistVector> DiffusionReactionSystem::raw_spatial_operator(double t, const double* y) const {
  if (!initialized_)
    refuse("DiffusionReactionSystem: initialize() must run before assemble_residual/rhs (E-PHYSICS-DRS-NOT-INITIALIZED)");

  fuse::linalg::DistVector out = system_.new_vector();
  const auto assembled = system_.assemble_vector(
      [&](spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
          std::vector<std::pair<std::int64_t, double>>& contrib) {
        const auto found = faces_by_element_.find(element);
        const std::vector<const ResolvedFace*>* faces = (found != faces_by_element_.end()) ? &found->second : nullptr;
        std::vector<double> local;
        accumulate_element_contribution(fe_, face_fe_, element, indices, dof_position_, faces, coefficients_, forcing_,
                                         t, y, local);
        for (std::size_t i = 0; i < indices.size(); ++i) contrib.push_back({indices[i], local[i]});
      },
      out);
  if (!assembled.ok()) return assembled.error();
  return out;
}

fuse::Result<void> DiffusionReactionSystem::assemble_residual(fuse::linalg::DistVector& x,
                                                                fuse::linalg::DistVector& fx) const {
  auto op = raw_spatial_operator(0.0, x.data());
  if (!op.ok()) return op.error();
  for (std::size_t i = 0; i < fx.size(); ++i) fx.data()[i] = op.value().data()[i];
  for (const auto& [dof, resolved] : dirichlet_by_dof_)
    fx.data()[static_cast<std::size_t>(dof)] =
        x.data()[static_cast<std::size_t>(dof)] - resolved->condition->value(resolved->position, 0.0);
  return fuse::Result<void>::success();
}

fuse::Result<void> DiffusionReactionSystem::mass(std::span<const double> v, std::span<double> out) const {
  if (!initialized_)
    refuse("DiffusionReactionSystem: initialize() must run before mass() (E-PHYSICS-DRS-NOT-INITIALIZED)");
  fuse::linalg::DistVector in = system_.new_vector_for(*mass_matrix_);
  fuse::linalg::DistVector product = system_.new_vector_for(*mass_matrix_);
  for (std::size_t i = 0; i < v.size(); ++i) in.data()[i] = v[i];
  if (auto applied = mass_matrix_->apply(in, product); !applied.ok()) return applied.error();
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = product.data()[i];
  for (const auto& [dof, resolved] : dirichlet_by_dof_) {
    (void)resolved;
    out[static_cast<std::size_t>(dof)] = v[static_cast<std::size_t>(dof)];
  }
  return fuse::Result<void>::success();
}

fuse::Result<void> DiffusionReactionSystem::rhs(double t, std::span<const double> y, std::span<double> out) const {
  auto op = raw_spatial_operator(t, y.data());
  if (!op.ok()) return op.error();
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = -op.value().data()[i];
  for (const auto& [dof, resolved] : dirichlet_by_dof_) {
    const BoundaryValue& rate = resolved->condition->value_time_derivative;
    if (!rate)
      return fuse::Failure{"E-PHYSICS-DRS-DIRICHLET-RATE-MISSING",
                            "DiffusionReactionSystem::rhs: a Dirichlet condition has no value_time_derivative; a "
                            "transient solve needs d(value)/dt at every Dirichlet dof (a time-independent "
                            "boundary value supplies a function returning 0)"};
    out[static_cast<std::size_t>(dof)] = rate(resolved->position, t);
  }
  return fuse::Result<void>::success();
}

fuse::Result<void> DiffusionReactionSystem::explicit_rhs(double t, std::span<const double> y,
                                                          std::span<double> out) const {
  const std::size_t n = static_cast<std::size_t>(dof_map_->n_dofs());
  std::vector<double> f(n);
  if (auto r = rhs(t, y, std::span<double>(f)); !r.ok()) return r.error();

  // Free rows: sum_{j free} M_ij u'_j = f_i - sum_{j Dirichlet} M_ij g'_j.
  // The second term is the consistent mass matrix applied to the vector
  // that holds g' at the Dirichlet dofs and zero elsewhere.
  fuse::linalg::DistVector rate = system_.new_vector_for(*mass_matrix_);
  fuse::linalg::DistVector coupling = system_.new_vector_for(*mass_matrix_);
  rate.set_zero();
  for (const auto& [dof, resolved] : dirichlet_by_dof_) {
    (void)resolved;
    rate.data()[static_cast<std::size_t>(dof)] = f[static_cast<std::size_t>(dof)];
  }
  if (auto applied = mass_matrix_->apply(rate, coupling); !applied.ok()) return applied.error();

  fuse::linalg::DistVector b = system_.new_vector_for(*eliminated_mass_matrix_);
  for (std::size_t i = 0; i < n; ++i) b.data()[i] = f[i] - coupling.data()[i];
  for (const auto& [dof, resolved] : dirichlet_by_dof_) {
    (void)resolved;
    b.data()[static_cast<std::size_t>(dof)] = f[static_cast<std::size_t>(dof)];
  }
  fuse::linalg::DistVector solution = system_.new_vector_for(*eliminated_mass_matrix_);
  solution.set_zero();

  // Solved to the accuracy the arithmetic allows, with no tolerance of
  // this class's choosing: the requested relative residual is the unit
  // roundoff, and a solve that stops at its attainable accuracy
  // (fuse/linalg/gmres.hpp) is accepted. An integrator that differences
  // this function to form a Jacobian action divides its rounding error by
  // a step of order sqrt(unit roundoff), so any looser inner solve would
  // cap the accuracy of that action.
  fuse::linalg::GmresOptions opt;
  opt.rel_tol = std::numeric_limits<double>::epsilon();
  opt.return_at_attainable_accuracy = true;
  const auto solved = fuse::linalg::gmres(*eliminated_mass_matrix_, *mass_preconditioner_, solution, b, opt);
  if (!solved.ok()) return solved.error();
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = solution.data()[i];
  return fuse::Result<void>::success();
}

fuse::num::TemporalProblem<double> DiffusionReactionSystem::as_temporal_problem() const {
  fuse::num::TemporalProblem<double> problem;
  problem.dimension = static_cast<std::size_t>(dof_map_->n_dofs());
  problem.rhs = [this](double t, std::span<const double> y, std::span<double> out) { return this->rhs(t, y, out); };
  problem.mass = [this](std::span<const double> v, std::span<double> out) { return this->mass(v, out); };
  return problem;
}

fuse::num::TemporalProblem<double> DiffusionReactionSystem::as_explicit_form_temporal_problem() const {
  fuse::num::TemporalProblem<double> problem;
  problem.dimension = static_cast<std::size_t>(dof_map_->n_dofs());
  problem.rhs = [this](double t, std::span<const double> y, std::span<double> out) {
    return this->explicit_rhs(t, y, out);
  };
  return problem;
}

void DiffusionReactionSystem::project_initial_condition(const InitialCondition& u0, std::vector<double>& out) const {
  if (!initialized_)
    refuse(
        "DiffusionReactionSystem: initialize() must run before project_initial_condition "
        "(E-PHYSICS-DRS-NOT-INITIALIZED)");
  out.assign(static_cast<std::size_t>(dof_map_->n_dofs()), 0.0);
  for (std::size_t dof = 0; dof < dof_position_.size(); ++dof) out[dof] = u0(dof_position_[dof]);
  for (const auto& [dof, resolved] : dirichlet_by_dof_)
    out[static_cast<std::size_t>(dof)] = resolved->condition->value(resolved->position, 0.0);
}

double DiffusionReactionSystem::l2_error(const std::vector<double>& x, double t,
                                          const std::function<double(const Point&, double)>& exact) const {
  double total = 0.0;
  std::vector<std::int64_t> indices;
  for (spatial::EntityHandle e = 1; e <= mesh_->n_elements(); ++e) {
    if (!dof_map_->owns_element_for_assembly(e)) continue;
    dof_map_->dof_indices(e, indices);
    const std::size_t n = indices.size();
    fe_.reinit(e);
    const auto& phi = fe_.get_phi();
    const auto& jxw = fe_.get_JxW();
    const auto& xyz = fe_.get_xyz();
    for (std::size_t q = 0; q < fe_.n_quadrature_points(); ++q) {
      double u_q = 0.0;
      for (std::size_t j = 0; j < n; ++j) u_q += phi[j][q] * x[static_cast<std::size_t>(indices[j])];
      const double diff = u_q - exact(xyz[q], t);
      total += jxw[q] * diff * diff;
    }
  }
  return std::sqrt(total);
}

fuse::Result<fuse::linalg::DistSparseMatrix> DiffusionReactionSystem::assemble_approximate_jacobian(
    const fuse::linalg::DistVector& x) const {
  if (!initialized_)
    refuse(
        "DiffusionReactionSystem: initialize() must run before assemble_approximate_jacobian "
        "(E-PHYSICS-DRS-NOT-INITIALIZED)");
  std::vector<char> identity_emitted(static_cast<std::size_t>(dof_map_->n_dofs()), 0);
  return system_.assemble_matrix([&](spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
                                      std::vector<linalg::Triplet>& out) {
    const std::size_t n = indices.size();
    std::vector<double> state(n);
    for (std::size_t i = 0; i < n; ++i) state[i] = x.data()[indices[i]];

    fe_.reinit(element);
    const auto& phi = fe_.get_phi();
    const auto& dphi = fe_.get_dphi();
    const auto& jxw = fe_.get_JxW();
    const auto& xyz = fe_.get_xyz();

    std::vector<double> k(n * n, 0.0), m(n * n, 0.0);
    for (std::size_t q = 0; q < fe_.n_quadrature_points(); ++q) {
      double u_q = 0.0;
      for (std::size_t j = 0; j < n; ++j) u_q += phi[j][q] * state[j];
      const double k_q = coefficients_.diffusion(xyz[q], u_q);
      const double w = jxw[q];
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          k[i * n + j] += w * k_q * dot(dphi[i][q], dphi[j][q]);
          m[i * n + j] += w * phi[i][q] * phi[j][q];
        }
      }
    }

    // The Robin term of the residual, + c * integral(phi_i u) over each
    // Robin face of this element, differentiates to c * integral(phi_i
    // phi_j).
    std::vector<double> robin(n * n, 0.0);
    if (const auto found = faces_by_element_.find(element); found != faces_by_element_.end()) {
      for (const ResolvedFace* face : found->second) {
        if (face->condition->kind != BoundaryConditionKind::Robin) continue;
        face_fe_.reinit(element, face->side);
        const auto& fphi = face_fe_.get_phi();
        const auto& fjxw = face_fe_.get_JxW();
        const double c = face->condition->robin_coefficient;
        for (std::size_t q = 0; q < face_fe_.n_quadrature_points(); ++q)
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) robin[i * n + j] += fjxw[q] * c * fphi[i][q] * fphi[j][q];
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      const auto row = static_cast<std::size_t>(indices[i]);
      if (dirichlet_by_dof_.find(indices[i]) != dirichlet_by_dof_.end()) {
        if (!identity_emitted[row]) {
          identity_emitted[row] = 1;
          out.push_back(linalg::Triplet{indices[i], indices[i], 1.0});
        }
        continue;
      }
      // The residual adds the reaction as + sum_j M_ij R(x_j, u_j)
      // (accumulate_element_contribution), so its derivative with respect
      // to u_j is + M_ij R'(x_j, u_j).
      for (std::size_t j = 0; j < n; ++j) {
        const double r_prime_j = coefficients_.reaction_derivative
                                      ? coefficients_.reaction_derivative(dof_position_[indices[j]], state[j])
                                      : 0.0;
        out.push_back(
            linalg::Triplet{indices[i], indices[j], k[i * n + j] + m[i * n + j] * r_prime_j + robin[i * n + j]});
      }
    }
  });
}

}  // namespace fuse::physics::fe
