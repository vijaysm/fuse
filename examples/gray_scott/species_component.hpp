#pragma once

/// One Gray-Scott species (U or V) as a `fuse::Component`, driven by
/// `fuse::WindowCoupler` (include/fuse/runtime/window_coupler.hpp) exactly
/// the way `LinearOdeComponent` is in
/// test/runtime/window_coupler_test.cpp: `advance()` reads its partner's
/// value from a `required` buffer WindowCoupler writes into before the
/// call -- the partner's MEAN over the substep, per that header's own class
/// comment -- and writes its own new state into `provided`, which for this
/// component is the same backing array as its internal state (both species
/// live on the same structured grid, so "provided" is simply "the whole
/// field, freshly advanced").
///
/// One step ([t_begin, t_end)) is Crank-Nicolson in time:
///   (s_new - s_old)/dt = 1/2 [ D*Lap(s_new) + React(s_new, p) + Src(t_end)
///                            + D*Lap(s_old) + React(s_old, p) + Src(t_begin) ]
/// with `p` the (frozen, per this call) partner value and `Src` an optional
/// manufactured forcing (zero in a physical run; see gray_scott_mms_test.cpp
/// for the MMS use). The partner does not appear as a Newton unknown -- only
/// `s`'s own value does -- so the Jacobian needs only d(React)/d(self), never
/// d(React)/d(partner); `monolithic_reference.hpp`'s fully coupled solver is
/// the one that needs the cross term.
///
/// The resulting Newton system is solved with Eigen::SparseLU, re-factorized
/// every Newton iteration (the reaction term's diagonal contribution depends
/// on the current iterate; the diffusion part does not, but re-factoring is
/// simplest and correct, and is not the dominant cost at the grid sizes this
/// example uses). Newton stops when the residual's L2 norm is at most
/// `newton_tol` -- see gray-scott.md for how that is chosen relative to the
/// coupling tolerance the caller sets on `WindowCouplerConfig::convergence`.

#include <Eigen/SparseLU>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

#include "gray_scott_physics.hpp"

namespace gray_scott {

enum class Species { U, V };

/// Manufactured-solution forcing: S(x, y, t). Defaults to "no source" (a
/// physical run); the MMS tests supply a nonzero one.
using SourceFn = std::function<double(double, double, double)>;

template <Species S>
class GrayScottSpeciesComponent : public fuse::Component {
 public:
  GrayScottSpeciesComponent(std::string name, Grid grid, double diffusion,
                            Kinetics kinetics, std::vector<double> initial_state,
                            double newton_tol, int newton_max_iter = 50)
      : name_(std::move(name)),
        grid_(grid),
        diffusion_(diffusion),
        kinetics_(kinetics),
        lap_(periodic_laplacian(grid)),
        state_(std::move(initial_state)),
        required_(grid.size(), 0.0),
        saved_(grid.size(), 0.0),
        newton_tol_(newton_tol),
        newton_max_iter_(newton_max_iter) {
    if (state_.size() != grid_.size()) {
      FUSE_THROW(fuse::InvalidArgument, "{}",
                std::string("GrayScottSpeciesComponent: initial_state has ") +
                    std::to_string(state_.size()) + " entries, grid has " +
                    std::to_string(grid_.size()));
    }
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const fuse::CouplingGraph&) override {
    // WindowCoupler binds provided()/required() directly (see
    // fuse/runtime/window_coupler.hpp's WindowMember comment); this
    // component does not go through the FieldRegistry, exactly like
    // LinearOdeComponent in window_coupler_test.cpp.
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double t0 = ctx.t_begin.to_double();
    const double t1 = ctx.t_end.to_double();
    const std::size_t n2 = grid_.size();

    if (dt <= 0.0) {
      // Degenerate step (can occur at a straddle boundary with zero
      // remaining length); nothing to advance.
      return fuse::Result<void>::success();
    }

    const std::vector<double> old = state_;         // s^n, fixed for this call
    const std::vector<double>& p = required_;        // partner, fixed for this call

    // Precompute the per-cell source contribution (constant across Newton
    // iterations -- it depends on (x, y, t), never on the unknown).
    std::vector<double> src_sum(n2, 0.0);
    if (source_) {
      for (int i = 0; i < grid_.n; ++i) {
        const double x = grid_.x_center(i);
        for (int j = 0; j < grid_.n; ++j) {
          const double y = grid_.y_center(j);
          const std::size_t m = grid_.index(i, j);
          src_sum[m] = source_(x, y, t0) + source_(x, y, t1);
        }
      }
    }

    // Constant (iteration-independent) part of the Jacobian:
    // base = I - (dt/2 * D) * Laplacian. Diagonal entries of `lap_` are
    // already nonzero, so the identity contribution lands on existing
    // structural entries -- no insertion needed below.
    Eigen::SparseMatrix<double> base = (-0.5 * dt * diffusion_) * lap_;
    for (int m = 0; m < static_cast<int>(n2); ++m) base.coeffRef(m, m) += 1.0;
    base.makeCompressed();

    Eigen::VectorXd u_new = Eigen::Map<const Eigen::VectorXd>(old.data(), static_cast<int>(n2));
    Eigen::VectorXd u_old = u_new;
    Eigen::VectorXd lap_old = lap_ * u_old;

    Eigen::VectorXd residual(static_cast<int>(n2));
    bool converged = false;
    for (int iter = 0; iter < newton_max_iter_; ++iter) {
      Eigen::VectorXd lap_new = lap_ * u_new;
      for (std::size_t m = 0; m < n2; ++m) {
        const double react_new = reaction(u_new[static_cast<int>(m)], p[m]);
        const double react_old = reaction(u_old[static_cast<int>(m)], p[m]);
        residual[static_cast<int>(m)] =
            u_new[static_cast<int>(m)] - u_old[static_cast<int>(m)] -
            0.5 * dt *
                (diffusion_ * (lap_new[static_cast<int>(m)] + lap_old[static_cast<int>(m)]) +
                 react_new + react_old + src_sum[m]);
      }
      const double resid_norm = residual.norm();
      if (resid_norm <= newton_tol_) {
        converged = true;
        break;
      }

      Eigen::SparseMatrix<double> jac = base;
      for (int m = 0; m < static_cast<int>(n2); ++m) {
        const double d_self = d_reaction_d_self(u_new[m], p[static_cast<std::size_t>(m)]);
        jac.coeffRef(m, m) -= 0.5 * dt * d_self;
      }
      jac.makeCompressed();

      Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
      solver.compute(jac);
      if (solver.info() != Eigen::Success) {
        return fuse::Failure{"E-EXAMPLE-GS-NEWTON-FACTORIZE",
                             name_ + ": SparseLU factorization failed at Newton iteration " +
                                 std::to_string(iter)};
      }
      const Eigen::VectorXd delta = solver.solve(-residual);
      if (solver.info() != Eigen::Success) {
        return fuse::Failure{"E-EXAMPLE-GS-NEWTON-SOLVE",
                             name_ + ": SparseLU solve failed at Newton iteration " +
                                 std::to_string(iter)};
      }
      u_new += delta;
    }
    if (!converged) {
      return fuse::Failure{
          "E-EXAMPLE-GS-NEWTON-NOCONVERGE",
          name_ + ": Newton did not reach tolerance " + std::to_string(newton_tol_) +
              " within " + std::to_string(newton_max_iter_) + " iterations"};
    }

    double flux_step = 0.0;
    for (std::size_t m = 0; m < n2; ++m) {
      state_[m] = u_new[static_cast<int>(m)];
      flux_step += coupling_term(u_new[static_cast<int>(m)], p[m]) +
                   coupling_term(u_old[static_cast<int>(m)], p[m]);
    }
    coupling_flux_ += 0.5 * dt * flux_step;
    (void)t1;
    return fuse::Result<void>::success();
  }

  // The accumulated coupling flux is part of the checkpoint. Outside it, every
  // coupling sweep and every straddle replay added its flux again, so the
  // measure counted iterations rather than the committed trajectory's exchange
  // -- off by O(1) at 7:13, where the Fixed member replays.
  fuse::Result<void> save_state() override {
    saved_ = state_;
    saved_flux_ = coupling_flux_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    std::copy(saved_.begin(), saved_.end(), state_.begin());  // element-wise: no reallocation
    coupling_flux_ = saved_flux_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  void set_source(SourceFn fn) { source_ = std::move(fn); }

  /// Running integral of the COUPLING-only part of this species' reaction
  /// term (U: -u*v^2; V: +u*v^2 -- the autocatalytic exchange, excluding the
  /// F(1-u)/-(F+k)v feed/decay term, which is not an exchange with the
  /// partner). Accumulated with the same trapezoidal (CN) quadrature the
  /// Newton residual itself uses, so this is exactly the amount THIS
  /// species' own discrete equation attributes to the exchange over every
  /// advance() call since the last reset_coupling_flux() -- used by
  /// gray_scott_reference_test.cpp's mass-exchange mismatch measurement
  /// (plan step 3, "measure the mismatch between what U removes and what V
  /// gains").
  [[nodiscard]] double coupling_flux() const { return coupling_flux_; }
  void reset_coupling_flux() { coupling_flux_ = 0.0; }

  [[nodiscard]] double* provided_ptr() { return state_.data(); }
  [[nodiscard]] double* required_ptr() { return required_.data(); }
  [[nodiscard]] const std::vector<double>& state() const { return state_; }
  void set_state(const std::vector<double>& s) {
    if (s.size() != state_.size()) {
      FUSE_THROW(fuse::InvalidArgument, "{}", std::string("GrayScottSpeciesComponent::set_state: size mismatch"));
    }
    std::copy(s.begin(), s.end(), state_.begin());
  }
  void set_required(const std::vector<double>& p) {
    if (p.size() != required_.size()) {
      FUSE_THROW(fuse::InvalidArgument, "{}", std::string("GrayScottSpeciesComponent::set_required: size mismatch"));
    }
    std::copy(p.begin(), p.end(), required_.begin());
  }
  [[nodiscard]] const Grid& grid() const { return grid_; }

 private:
  [[nodiscard]] double reaction(double self, double partner) const noexcept {
    if constexpr (S == Species::U) {
      return kinetics_.reaction_u(self, partner);
    } else {
      return kinetics_.reaction_v(partner, self);
    }
  }
  [[nodiscard]] double d_reaction_d_self(double self, double partner) const noexcept {
    if constexpr (S == Species::U) {
      return kinetics_.d_reaction_u_du(self, partner);
    } else {
      return kinetics_.d_reaction_v_dv(partner, self);
    }
  }
  /// The autocatalytic exchange term only (no F(1-u)/-(F+k)v feed/decay):
  /// U removes -self*partner^2; V gains +partner*self^2.
  [[nodiscard]] double coupling_term(double self, double partner) const noexcept {
    if constexpr (S == Species::U) {
      return -self * partner * partner;
    } else {
      return partner * self * self;
    }
  }

  std::string name_;
  Grid        grid_;
  double      diffusion_;
  Kinetics    kinetics_;
  Eigen::SparseMatrix<double> lap_;
  std::vector<double> state_;
  std::vector<double> required_;
  std::vector<double> saved_;
  double newton_tol_;
  int    newton_max_iter_;
  SourceFn source_;
  double coupling_flux_ = 0.0;
  double saved_flux_ = 0.0;
};

using GrayScottU = GrayScottSpeciesComponent<Species::U>;
using GrayScottV = GrayScottSpeciesComponent<Species::V>;

}  // namespace gray_scott
