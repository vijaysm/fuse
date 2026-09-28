#pragma once

/// U and V as `fuse::Component`s, driven by `fuse::WindowCoupler`
/// (include/fuse/runtime/window_coupler.hpp), for the multirate Brusselator
/// example. See brusselator_physics.hpp's file header for the model and
/// the mesh/index conventions, and docs/user/brusselator.md
/// "Where the FUSE port necessarily differs from the Toy" for the exchange
/// design this class pair implements.
///
/// THE EXCHANGE, ON ONE SHARED (U) GRID (NUMERICS.md's design decision,
/// preserved exactly): `q = U^2 * P_VU(V) - A*U` is computed ONCE, by U,
/// never independently re-derived on V's own grid (which would generally
/// destroy the U^2*V cancellation the Toy's NUMERICS.md warns about).
///   * `UComponent::provided_` is `q`, on the U grid, LAGGED: computed from
///     U's OWN state at the START of its step (`state_` before this call)
///     and the partner mean it just received -- an explicit, backward-
///     Euler-LAGGED reaction, matching "backward Euler for diffusion and
///     the U decay" (the task's model statement says nothing about an
///     implicit reaction) while keeping U's own linear system LINEAR, at
///     the cost of resolving q at only U's own step resolution, not the
///     Toy's union-of-both-schedules knot resolution. See
///     docs/user/brusselator.md for the measured consequence.
///   * `VComponent::provided_` is `P_VU(V)` -- V's OWN NEW state (valid at
///     `ctx.t_end`, so it DOES satisfy the `Provides` contract on the
///     nose, unlike U's q) remapped onto the U grid with FUSE's
///     conservative supermesh weights (`fuse::apply_weights`), never a
///     hand-rolled sparse product.
/// Both members therefore use `WindowMember::length = N_u` (the U grid
/// size) for BOTH `provided` and `required` -- `V.required_` receives the
/// RAW (unremapped) q on the U grid, and `VComponent::advance` applies
/// `P_UV` to it internally before using it, exactly mirroring "V remaps
/// the received q from the U grid onto its own grid" (NUMERICS.md).
///
/// DIAGNOSTIC ACCUMULATORS AND THE WINDOW-COUPLER CHECKPOINT RULE. Every
/// value below that accumulates across `advance()` calls
/// (`external_total_`, `exchange_out_`/`exchange_in_`) is part of
/// `save_state()`/`restore_state()`. A prior track's Gray-Scott component
/// (examples/gray_scott/species_component.hpp's `coupling_flux_`) found
/// this the hard way: left outside the checkpoint, such a value is
/// incremented on every Picard sweep and every straddle replay, so it
/// measures ITERATION COUNT, not the committed trajectory's exchange --
/// see that header's own comment on `coupling_flux_`.

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/result.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

#include "brusselator_physics.hpp"

namespace brusselator {

/// Shared physical parameters, all defaulted to the Toy's own values
/// (README.md "Model and reference").
struct PhysicsConfig {
  double diffusion = 10.0;  ///< alpha
  double A = 3.4;
  double B = 1.0;
  bool   forcing = true;
};

/// U_t = a*Lap(U) + B - U + q + f. Backward Euler for diffusion and the
/// U decay; q and f enter explicitly (f exactly integrated over its
/// active duration within the step; q is the lagged reaction described
/// in the file header).
class UComponent final : public fuse::Component {
 public:
  UComponent(std::string name, const NativeGrid& grid, PhysicsConfig cfg, Vector initial_v_on_u)
      : name_(std::move(name)),
        n_(grid.size()),
        cfg_(cfg),
        laplacian_(periodic_laplacian(grid.nx(), grid.ny())),
        mask_(forcing_mask(grid.nx(), grid.ny())),
        mask_mean_(mask_.mean()),
        state_(initial_condition(/*is_u=*/true, grid.nx(), grid.ny())),
        required_(std::move(initial_v_on_u)),
        provided_(n_),
        q_prev_(n_),
        saved_state_(n_) {
    reaction_kokkos(state_, required_, &provided_);
    q_prev_ = provided_;
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry&, const fuse::CouplingGraph&) override {
    // WindowCoupler binds provided()/required() directly, exactly like
    // gray_scott's GrayScottSpeciesComponent -- see that header's own
    // comment on why this is a no-op here.
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double t0 = ctx.t_begin.to_double();
    const double t1 = ctx.t_end.to_double();
    if (dt <= 0.0) return fuse::Result<void>::success();

    // Explicit, LAGGED reaction driving THIS step's (linear) solve: q_prev_
    // is the sample this component reported at the end of the PREVIOUS
    // step (or the t=0 seed), i.e. known before this call -- see the file
    // header. This is what keeps U's own system linear (no per-cell
    // Newton), at the documented cost of resolving q at U's own step
    // resolution rather than the Toy's union-of-both-schedules knots.
    const double active = cfg_.forcing ? forcing_active_duration(t0, t1) : 0.0;
    Vector rhs_base = state_.array() + dt * cfg_.B;
    if (cfg_.forcing && active > 0.0) rhs_base += active * mask_;

    if (factored_step_ < 0.0 || std::abs(dt - factored_step_) > 32 * std::numeric_limits<double>::epsilon() * dt) {
      Eigen::SparseMatrix<double> matrix = (-dt * cfg_.diffusion) * laplacian_;
      for (int i = 0; i < n_; ++i) matrix.coeffRef(i, i) += 1.0 + dt;
      factor_.compute(matrix);
      if (factor_.info() != Eigen::Success) {
        return fuse::Failure{"E-BRUSS-U-FACTORIZE", name_ + ": diffusion factorization failed"};
      }
      factored_step_ = dt;
    }
    // U CONSUMES EXACTLY WHAT IT PROVIDES. WindowCoupler hands V the
    // substep mean of the piecewise-LINEAR waveform through U's consecutive
    // provided samples, so over this step V receives 0.5*dt*(q_prev + q_new).
    // A U step that consumed dt*q_prev (the start-of-step rate, which keeps
    // the step linear) left the Toy's discrete budget identity holding only
    // to first order: measured 1.686e-3 per window against the Toy's 1.2e-13
    // -- a real conservation defect of the scheme, the two species exchanging
    // different amounts of the same reaction, not a diagnostic artifact.
    // q_new depends on the state being solved for, so the step is a short
    // fixed point on it: solve, re-evaluate q_new, repeat. The contraction
    // factor is about 0.5*dt*|dq/dU| = 0.5*dt*(2UV + A), of order 1e-2 at the
    // default dt, so a handful of solves against the one LDLT factorisation
    // suffices. Robustness first: a step that does not contract to the
    // tolerance is refused, not accepted with a warning.
    Vector q_new = q_prev_;
    Vector next(n_);
    bool converged = false;
    for (int k = 0; k < kMaxReactionIterations; ++k) {
      const Vector rhs = rhs_base + (0.5 * dt) * (q_prev_ + q_new);
      next = factor_.solve(rhs);
      if (factor_.info() != Eigen::Success || !next.allFinite()) {
        return fuse::Failure{"E-BRUSS-U-SOLVE", name_ + ": diffusion solve failed or produced a non-finite state"};
      }
      Vector q_next(n_);
      reaction_kokkos(next, required_, &q_next);
      const double change = (q_next - q_new).lpNorm<Eigen::Infinity>();
      const double scale = 1.0 + q_next.lpNorm<Eigen::Infinity>();
      q_new = std::move(q_next);
      if (change <= kReactionTolerance * scale) { converged = true; break; }
    }
    if (!converged) {
      return fuse::Failure{"E-BRUSS-U-REACTION",
                           name_ + ": the implicit reaction fixed point did not contract to " +
                               std::to_string(kReactionTolerance) + " in " +
                               std::to_string(kMaxReactionIterations) + " solves over [" +
                               std::to_string(t0) + ", " + std::to_string(t1) + "]; reduce dt_u"};
    }
    state_ = std::move(next);

    external_total_ += dt * cfg_.B - dt * state_.mean() + active * mask_mean_;

    // The same trapezoid the step consumed and WindowCoupler delivers to V.
    exchange_out_ += 0.5 * dt * (q_prev_.mean() + q_new.mean());
    provided_ = q_new;
    q_prev_ = std::move(q_new);
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_state_ = state_;
    saved_external_total_ = external_total_;
    saved_exchange_out_ = exchange_out_;
    saved_q_prev_ = q_prev_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    std::copy(saved_state_.begin(), saved_state_.end(), state_.begin());
    external_total_ = saved_external_total_;
    exchange_out_ = saved_exchange_out_;
    q_prev_ = saved_q_prev_;
    return fuse::Result<void>::success();
  }
  /// Rollback: save_state()/restore_state() are real. Bitwise: a hand-written
  /// solver with no RNG and no unordered reduction (elementwise Kokkos
  /// kernels only), and example.*.omp_identical measures it bit for bit at
  /// one thread and at every core. Hidden state: YES -- advance() mutates
  /// state that is not bound through register_fields() (the clock, the
  /// history, the solver's internals), which the runtime's byte-for-byte
  /// rollback verification cannot see; save_state() is what carries it.
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true);
  }

  [[nodiscard]] double* provided_ptr() { return provided_.data(); }
  [[nodiscard]] double* required_ptr() { return required_.data(); }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] double external_total() const { return external_total_; }
  [[nodiscard]] double exchange_out() const { return exchange_out_; }

 private:
  /// Fixed-point tolerance on the reaction, relative to 1 + |q|_inf. The
  /// Picard tolerance across windows is 1e-11 by default; this must be
  /// tighter so the inner iteration is never what limits the outer one.
  static constexpr double kReactionTolerance = 1e-13;
  static constexpr int    kMaxReactionIterations = 50;

  std::string name_;
  int n_;
  PhysicsConfig cfg_;
  Sparse laplacian_;
  Vector mask_;
  double mask_mean_;
  Vector state_;
  Vector required_;   ///< P_VU(V) mean over this step, on the U grid
  Vector provided_;   ///< q, on the U grid, valid at ctx.t_end
  Vector q_prev_;     ///< the LAGGED q driving the current step's own RHS
  Vector saved_state_, saved_q_prev_;
  Eigen::SimplicialLDLT<Sparse> factor_;
  double factored_step_ = -1.0;
  double external_total_ = 0.0, saved_external_total_ = 0.0;
  double exchange_out_ = 0.0, saved_exchange_out_ = 0.0;
};

/// V_t = a*Lap(V) - q. Backward Euler for diffusion; the received q
/// (raw, on the U grid) is remapped onto V's own grid with `weights_uv`
/// before being subtracted.
class VComponent final : public fuse::Component {
 public:
  VComponent(std::string name, const NativeGrid& u_grid, const NativeGrid& v_grid,
            PhysicsConfig cfg, const fuse::RemapWeights* weights_uv,
            const fuse::RemapWeights* weights_vu)
      : name_(std::move(name)),
        u_grid_(&u_grid),
        v_grid_(&v_grid),
        n_u_(u_grid.size()),
        n_v_(v_grid.size()),
        cfg_(cfg),
        laplacian_(periodic_laplacian(v_grid.nx(), v_grid.ny())),
        state_(initial_condition(/*is_u=*/false, v_grid.nx(), v_grid.ny())),
        required_(Vector::Zero(n_u_)),
        provided_(n_u_),
        saved_state_(n_v_),
        weights_uv_(weights_uv),
        weights_vu_(weights_vu) {
    remap_state_to_u(&provided_);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry&, const fuse::CouplingGraph&) override {
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    if (dt <= 0.0) return fuse::Result<void>::success();

    // Remap the received (raw, on the U grid) q mean onto V's own grid --
    // NUMERICS.md's "V remaps the received q from the U grid onto its own
    // grid". P_UV is linear, so applying it to the WindowCoupler-supplied
    // MEAN commutes with the time integral it already represents
    // (NUMERICS.md "Spatial transfer and time integration ... commute").
    std::vector<double> q_native_u(static_cast<std::size_t>(n_u_));
    {
      Vector q_canonical_u = Eigen::Map<const Vector>(required_.data(), n_u_);
      u_grid_->to_native(q_canonical_u, &q_native_u);
    }
    std::vector<double> q_native_v(static_cast<std::size_t>(n_v_), 0.0);
    auto applied = fuse::apply_weights(*weights_uv_, q_native_u.data(), q_native_v.data(), 1);
    if (!applied.ok()) return applied.error();
    Vector q_on_v(n_v_);
    v_grid_->to_canonical(q_native_v, &q_on_v);

    Vector rhs = state_ - dt * q_on_v;
    if (factored_step_ < 0.0 || std::abs(dt - factored_step_) > 32 * std::numeric_limits<double>::epsilon() * dt) {
      Eigen::SparseMatrix<double> matrix = (-dt * cfg_.diffusion) * laplacian_;
      for (int i = 0; i < n_v_; ++i) matrix.coeffRef(i, i) += 1.0;
      factor_.compute(matrix);
      if (factor_.info() != Eigen::Success) {
        return fuse::Failure{"E-BRUSS-V-FACTORIZE", name_ + ": diffusion factorization failed"};
      }
      factored_step_ = dt;
    }
    Vector next = factor_.solve(rhs);
    if (factor_.info() != Eigen::Success || !next.allFinite()) {
      return fuse::Failure{"E-BRUSS-V-SOLVE", name_ + ": diffusion solve failed or produced a non-finite state"};
    }
    state_ = std::move(next);
    remap_state_to_u(&provided_);

    exchange_in_ += dt * q_on_v.mean();
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_state_ = state_;
    saved_exchange_in_ = exchange_in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    std::copy(saved_state_.begin(), saved_state_.end(), state_.begin());
    exchange_in_ = saved_exchange_in_;
    return fuse::Result<void>::success();
  }
  /// Rollback: save_state()/restore_state() are real. Bitwise: a hand-written
  /// solver with no RNG and no unordered reduction (elementwise Kokkos
  /// kernels only), and example.*.omp_identical measures it bit for bit at
  /// one thread and at every core. Hidden state: YES -- advance() mutates
  /// state that is not bound through register_fields() (the clock, the
  /// history, the solver's internals), which the runtime's byte-for-byte
  /// rollback verification cannot see; save_state() is what carries it.
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true);
  }

  [[nodiscard]] double* provided_ptr() { return provided_.data(); }
  [[nodiscard]] double* required_ptr() { return required_.data(); }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] double exchange_in() const { return exchange_in_; }

 private:
  /// provided_ = P_VU(state_): V's own new state remapped onto the U grid.
  void remap_state_to_u(Vector* out) {
    std::vector<double> v_native(static_cast<std::size_t>(n_v_));
    v_grid_->to_native(state_, &v_native);
    std::vector<double> u_native(static_cast<std::size_t>(n_u_), 0.0);
    // weights_vu_ is built with plan.src = V's region, plan.tgt = U's
    // region, so row = U (target) native index, col = V (source) native
    // index -- exactly apply_weights's dst/src convention below.
    (void)fuse::apply_weights(*weights_vu_, v_native.data(), u_native.data(), 1);
    u_grid_->to_canonical(u_native, out);
  }

  std::string name_;
  const NativeGrid* u_grid_;
  const NativeGrid* v_grid_;
  int n_u_, n_v_;
  PhysicsConfig cfg_;
  Sparse laplacian_;
  Vector state_;
  Vector required_;  ///< raw q mean over this step, on the U grid (unremapped)
  Vector provided_;  ///< P_VU(state_), on the U grid
  Vector saved_state_;
  Eigen::SimplicialLDLT<Sparse> factor_;
  double factored_step_ = -1.0;
  double exchange_in_ = 0.0, saved_exchange_in_ = 0.0;
  const fuse::RemapWeights* weights_uv_;  ///< U grid -> V grid
  const fuse::RemapWeights* weights_vu_;  ///< V grid -> U grid
};

}  // namespace brusselator
