#pragma once

/// Time windows track (Conformance, Unit 2): the ONE physics source
/// species_component.hpp's own UComponent/VComponent and the participant-
/// ABI shim (examples/brusselator/participants/shim.cpp) both call --
/// extracted from UComponent::advance()/VComponent::advance() (this
/// session) so the reaction-diffusion solve itself exists exactly once,
/// not duplicated between the in-process (fuse::Component) and the
/// externally-driven (fuse_begin_step()/fuse_end_step()) drivers.
///
/// `UCore` is the WHOLE of U's own physics -- U never did any remap work
/// itself (species_component.hpp's own file header: ALL remap, both
/// directions, was hand-done inside VComponent), so nothing further
/// needs splitting out. `VDiffusionCore` is deliberately NOT the whole of
/// V's own physics: it is only the "solve backward-Euler diffusion given
/// q already on THIS grid" step -- the lead's own ruling (Conformance
/// Unit 2) is that the participant graph IS Unit 1's own graph
/// (brusselator_graph.hpp's `add_window_coupling`) PLUS, and only plus,
/// `add_coupler()`/`remap_on(coupler)` on both transfers: both transfer
/// fields stay declared on `u_region` (species_component.hpp's own file
/// header, "WHY BOTH TRANSFER FIELDS LIVE ON u_region"), so the coupler-
/// routed Remap this compiles is the IDENTITY (confirmed this session,
/// `WindowRunner.WindowChannelRemapRoutedToCouplerMatchesIdentity
/// ClosedForm`, test/runtime/window_runner_test.cpp) and V still stands
/// in for the real remap by hand, exactly as `VComponent` (Unit 1,
/// `--driver window`/`--driver runner`) already does: `advance()` remaps
/// q FROM the U grid onto its own grid by hand (`fuse::apply_weights`,
/// `weights_uv_`) before calling this, and remaps its own new state BACK
/// onto the U grid afterward (`remap_state_to_u`). The externally-driven
/// shim (shim.cpp) needs the IDENTICAL by-hand remap glue for the same
/// reason -- its own `VParticipant` wrapper (below `VDiffusionCore`)
/// carries it, built from the SAME `NativeGrid`/`fuse::RemapWeights`
/// machinery `VComponent` uses, so `VDiffusionCore` itself stays exactly
/// what both drivers share: V's own diffusion solve, nothing else.

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "brusselator_physics.hpp"

namespace brusselator {

/// Shared physical parameters, all defaulted to the Toy's own values
/// (README.md "Model and reference"). Moved here (from
/// species_component.hpp, where it used to live) when UCore/VDiffusionCore
/// were factored out, since UCore's own constructor needs it and
/// species_core.hpp has no dependency on species_component.hpp (the
/// reverse is true: species_component.hpp includes this header).
struct PhysicsConfig {
  double diffusion = 10.0;  ///< alpha
  double A = 3.4;
  double B = 1.0;
  bool   forcing = true;
};

/// U_t = a*Lap(U) + B - U + q + f, extracted verbatim from
/// UComponent::advance() (species_component.hpp's own file header has
/// the full derivation and the measured conservation consequence of the
/// lagged-reaction choice). Pure math: no fuse::Component, no
/// FieldRegistry, no NativeGrid -- just `nx*ny` and the physical
/// parameters, so it is usable from a C-ABI shim with no FUSE mesh
/// machinery linked in at all.
class UCore {
 public:
  UCore(int nx, int ny, PhysicsConfig cfg, Vector initial_v_on_u)
      : n_(nx * ny),
        cfg_(cfg),
        laplacian_(periodic_laplacian(nx, ny)),
        mask_(forcing_mask(nx, ny)),
        mask_mean_(mask_.mean()),
        state_(initial_condition(/*is_u=*/true, nx, ny)),
        required_(std::move(initial_v_on_u)),
        provided_(n_),
        q_prev_(n_),
        saved_state_(n_),
        // Found by inspection this session (examples/brusselator/
        // participants work): `saved_q_prev_` was left at
        // Eigen::VectorXd's own default size (0), not `n_` like every
        // sibling saved_* buffer here -- a genuine latent bug (if
        // restore_state() ever ran before the first save_state(), the
        // resulting size-0 `q_prev_ = saved_q_prev_` assignment would
        // misbehave) and fixed on its own merits. NOTE: measured this
        // session (BrusselatorParticipants.
        // ExternallyDrivenBothMembersMatchInProcessRunner, 10 repeated
        // runs before and after this fix) that this fix did NOT change
        // the observed intermittent E-RT-NONFINITE failure rate (4/10
        // both times) -- that non-determinism has a DIFFERENT, still-
        // unidentified cause, reported separately to the lead rather
        // than claimed as resolved here.
        saved_q_prev_(n_) {
    reaction_kokkos(state_, required_, &provided_);
    q_prev_ = provided_;
  }

  /// `required`: P_VU(V) mean over [t_begin, t_end), on U's own grid,
  /// size n(). On success, `state()`/`provided()` are valid at t_end.
  /// `dt`: the EXACT step length, i.e. `fuse::StepContext::dt().to_double()`
  /// -- NOT `t_end - t_begin` recomputed from the two already-rounded
  /// doubles. Found by the lead (bit-identical check, Conformance Unit 2):
  /// the old UComponent::advance() this was extracted from called
  /// `ctx.dt().to_double()` (the EXACT rational t_end-t_begin, rounded
  /// ONCE) for every dt-scaled quantity here, while `t0`/`t1` (also each
  /// rounded once from their own Rationals) were used ONLY for
  /// forcing_active_duration(). Recomputing `dt` as a double subtraction
  /// of `t_end`/`t_begin` inside this function (this file's own original
  /// version) adds a THIRD, different rounding -- the two are not always
  /// bit-equal for a Rational step. VERIFIED after applying this fix to
  /// BOTH call sites (species_component.hpp's own UComponent::advance()
  /// and shim.cpp's own bruss_u_step()) and rebuilding: `diff` of the
  /// full stdout of both drivers at both --end 0.5 and --end 3, pre- vs
  /// post-refactor, is empty (exit 0) in all four cases -- the pre-
  /// refactor global_balance_error (2.407e-12 at --end 0.5, 1.266e-11 at
  /// --end 3) and this fixed version's are the SAME printed figures, not
  /// just close. Before this specific fix (species_core.hpp's own
  /// `dt = t_end - t_begin`, with both call sites otherwise already
  /// ported), the refactor's own global_balance_error was 1.842e-12 and
  /// 1.038e-11 respectively -- different from the pre-refactor figures,
  /// though `state()`/`final_mean_u` were already bit-identical either
  /// way (only round-off-scale diagnostics that sum many steps' worth of
  /// dt-scaled terms -- external_total_, exchange_out_ -- could even
  /// show the difference).
  [[nodiscard]] fuse::Result<void> step(double t_begin, double t_end, double dt,
                                        const Vector& required) {
    required_ = required;
    if (dt <= 0.0) return fuse::Result<void>::success();

    const double active = cfg_.forcing ? forcing_active_duration(t_begin, t_end) : 0.0;
    Vector rhs_base = state_.array() + dt * cfg_.B;
    if (cfg_.forcing && active > 0.0) rhs_base += active * mask_;

    if (factored_step_ < 0.0 ||
        std::abs(dt - factored_step_) > 32 * std::numeric_limits<double>::epsilon() * dt) {
      Eigen::SparseMatrix<double> matrix = (-dt * cfg_.diffusion) * laplacian_;
      for (int i = 0; i < n_; ++i) matrix.coeffRef(i, i) += 1.0 + dt;
      factor_.compute(matrix);
      if (factor_.info() != Eigen::Success) {
        return fuse::Failure{"E-BRUSS-U-FACTORIZE", "UCore: diffusion factorization failed"};
      }
      factored_step_ = dt;
    }
    Vector q_new = q_prev_;
    Vector next(n_);
    bool converged = false;
    for (int k = 0; k < kMaxReactionIterations; ++k) {
      const Vector rhs = rhs_base + (0.5 * dt) * (q_prev_ + q_new);
      next = factor_.solve(rhs);
      if (factor_.info() != Eigen::Success || !next.allFinite()) {
        return fuse::Failure{"E-BRUSS-U-SOLVE",
                             "UCore: diffusion solve failed or produced a non-finite state"};
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
                           "UCore: the implicit reaction fixed point did not contract to " +
                               std::to_string(kReactionTolerance) + " in " +
                               std::to_string(kMaxReactionIterations) + " solves over [" +
                               std::to_string(t_begin) + ", " + std::to_string(t_end) + "]"};
    }
    state_ = std::move(next);
    external_total_ += dt * cfg_.B - dt * state_.mean() + active * mask_mean_;
    exchange_out_ += 0.5 * dt * (q_prev_.mean() + q_new.mean());
    provided_ = q_new;
    q_prev_ = std::move(q_new);
    return fuse::Result<void>::success();
  }

  void save_state() {
    saved_state_ = state_;
    saved_external_total_ = external_total_;
    saved_exchange_out_ = exchange_out_;
    saved_q_prev_ = q_prev_;
  }
  void restore_state() {
    state_ = saved_state_;
    external_total_ = saved_external_total_;
    exchange_out_ = saved_exchange_out_;
    q_prev_ = saved_q_prev_;
  }

  [[nodiscard]] int n() const { return n_; }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] const Vector& provided() const { return provided_; }
  [[nodiscard]] double external_total() const { return external_total_; }
  [[nodiscard]] double exchange_out() const { return exchange_out_; }

 private:
  static constexpr double kReactionTolerance = 1e-13;
  static constexpr int    kMaxReactionIterations = 50;

  int n_;
  PhysicsConfig cfg_;
  Sparse laplacian_;
  Vector mask_;
  double mask_mean_;
  Vector state_;
  Vector required_, provided_, q_prev_;
  Vector saved_state_, saved_q_prev_;
  Eigen::SimplicialLDLT<Sparse> factor_;
  double factored_step_ = -1.0;
  double external_total_ = 0.0, saved_external_total_ = 0.0;
  double exchange_out_ = 0.0, saved_exchange_out_ = 0.0;
};

/// V_t = a*Lap(V) - q_on_v, extracted from the diffusion-solve portion of
/// VComponent::advance() only -- this class's own top comment (above)
/// explains why the remap half is deliberately NOT here.
class VDiffusionCore {
 public:
  VDiffusionCore(int nx, int ny, double diffusion, Vector initial_state)
      : n_(nx * ny), diffusion_(diffusion), laplacian_(periodic_laplacian(nx, ny)),
        state_(std::move(initial_state)), saved_state_(n_) {}

  /// `q_on_v`: the reaction rate mean over this step, ALREADY on this
  /// grid (size n()). On success, `state()` is valid at t_end = t_begin+dt.
  [[nodiscard]] fuse::Result<void> step(double dt, const Vector& q_on_v) {
    if (dt <= 0.0) return fuse::Result<void>::success();
    Vector rhs = state_ - dt * q_on_v;
    if (factored_step_ < 0.0 ||
        std::abs(dt - factored_step_) > 32 * std::numeric_limits<double>::epsilon() * dt) {
      Eigen::SparseMatrix<double> matrix = (-dt * diffusion_) * laplacian_;
      for (int i = 0; i < n_; ++i) matrix.coeffRef(i, i) += 1.0;
      factor_.compute(matrix);
      if (factor_.info() != Eigen::Success) {
        return fuse::Failure{"E-BRUSS-V-FACTORIZE", "VDiffusionCore: diffusion factorization failed"};
      }
      factored_step_ = dt;
    }
    Vector next = factor_.solve(rhs);
    if (factor_.info() != Eigen::Success || !next.allFinite()) {
      return fuse::Failure{"E-BRUSS-V-SOLVE",
                           "VDiffusionCore: diffusion solve failed or produced a non-finite state"};
    }
    state_ = std::move(next);
    exchange_in_ += dt * q_on_v.mean();
    return fuse::Result<void>::success();
  }

  void save_state() {
    saved_state_ = state_;
    saved_exchange_in_ = exchange_in_;
  }
  void restore_state() {
    state_ = saved_state_;
    exchange_in_ = saved_exchange_in_;
  }

  [[nodiscard]] int n() const { return n_; }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] double exchange_in() const { return exchange_in_; }

 private:
  int n_;
  double diffusion_;
  Sparse laplacian_;
  Vector state_, saved_state_;
  Eigen::SimplicialLDLT<Sparse> factor_;
  double factored_step_ = -1.0;
  double exchange_in_ = 0.0, saved_exchange_in_ = 0.0;
};

/// V as the participant-ABI shim's own driver sees it: `VDiffusionCore`
/// PLUS the by-hand remap glue `VComponent` (species_component.hpp) also
/// carries -- U-grid-shaped `required`/`provided`, remapped to/from V's
/// own grid with `fuse::apply_weights` before/after `VDiffusionCore::
/// step()`. Needed because the participant graph is Unit 1's own graph
/// (both transfer fields on `u_region`) plus only `add_coupler()`/
/// `remap_on()` (the lead's own ruling, Conformance Unit 2) -- so
/// `fuse_initialize()` hands this shim's caller U-grid-shaped data on
/// "v_in"/"v_out", exactly as it hands `VComponent` the same fields
/// in-process, and V still stands in for the real remap by hand either
/// way. Builds its OWN tiny backend+graph, once, purely to resolve
/// `fuse::build_weights`'s two `RemapPlan`s (exactly `examples/
/// brusselator/main.cpp`'s own "V (source) -> U (target)"/"U (source)
/// -> V (target)" pair) -- this class owns that machinery so neither
/// `species_component.hpp` nor a real participant process needs to
/// duplicate it.
class VParticipant {
 public:
  VParticipant(const VParticipant&) = delete;
  VParticipant& operator=(const VParticipant&) = delete;
  VParticipant(VParticipant&&) = default;
  VParticipant& operator=(VParticipant&&) = default;

  [[nodiscard]] static fuse::Result<VParticipant> make(int nx_u, int ny_u, int nx_v, int ny_v,
                                                        double diffusion) {
    fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
    auto u_grid_r = NativeGrid::make(backend, "u", nx_u, ny_u);
    if (!u_grid_r.ok()) return u_grid_r.error();
    auto v_grid_r = NativeGrid::make(backend, "v", nx_v, ny_v);
    if (!v_grid_r.ok()) return v_grid_r.error();

    fuse::CouplingGraph g;
    const fuse::MeshId u_mesh = g.add_mesh(
        {.name = "u", .source = "generated:quad:" + std::to_string(nx_u) + "x" +
                                 std::to_string(ny_u)});
    const fuse::MeshId v_mesh = g.add_mesh(
        {.name = "v", .source = "generated:quad:" + std::to_string(nx_v) + "x" +
                                 std::to_string(ny_v)});
    const fuse::RegionId u_region = g.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
    const fuse::RegionId v_region = g.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

    fuse::RemapPlan plan_vu;  // V (source) -> U (target)
    plan_vu.src = v_region;
    plan_vu.tgt = u_region;
    auto w_vu = fuse::build_weights(plan_vu, g, backend);
    if (!w_vu.ok()) return w_vu.error();
    fuse::RemapPlan plan_uv;  // U (source) -> V (target)
    plan_uv.src = u_region;
    plan_uv.tgt = v_region;
    auto w_uv = fuse::build_weights(plan_uv, g, backend);
    if (!w_uv.ok()) return w_uv.error();

    return VParticipant(std::move(backend), std::move(u_grid_r.value()),
                        std::move(v_grid_r.value()), diffusion, std::move(w_vu.value()),
                        std::move(w_uv.value()));
  }

  /// `required`: the raw q mean this step, on the U grid (unremapped),
  /// size `n_u()`. On success, `provided` (size `n_u()`) receives
  /// `P_VU(state())`, valid at t_begin+dt -- exactly `VComponent::
  /// advance()`'s own two-sided remap (species_component.hpp).
  [[nodiscard]] fuse::Result<void> step(double dt, const Vector& required, Vector* provided) {
    std::vector<double> q_native_u(static_cast<std::size_t>(n_u_));
    {
      Vector q_canonical_u = required;
      u_grid_.to_native(q_canonical_u, &q_native_u);
    }
    std::vector<double> q_native_v(static_cast<std::size_t>(n_v_), 0.0);
    auto applied = fuse::apply_weights(weights_uv_, q_native_u.data(), q_native_v.data(), 1);
    if (!applied.ok()) return applied.error();
    Vector q_on_v(n_v_);
    v_grid_.to_canonical(q_native_v, &q_on_v);

    const fuse::Result<void> r = core_->step(dt, q_on_v);
    if (!r.ok()) return r;
    remap_state_to_u(provided);
    return fuse::Result<void>::success();
  }

  void save_state() { core_->save_state(); }
  void restore_state() { core_->restore_state(); }

  [[nodiscard]] int n_u() const { return n_u_; }
  [[nodiscard]] const Vector& state() const { return core_->state(); }
  [[nodiscard]] double exchange_in() const { return core_->exchange_in(); }

 private:
  VParticipant(fuse::native::NativeMeshBackend backend, NativeGrid u_grid, NativeGrid v_grid,
              double diffusion, fuse::RemapWeights weights_vu, fuse::RemapWeights weights_uv)
      : backend_(std::move(backend)),
        u_grid_(std::move(u_grid)),
        v_grid_(std::move(v_grid)),
        n_u_(u_grid_.size()),
        n_v_(v_grid_.size()),
        // heap-allocated: VDiffusionCore holds an Eigen::SimplicialLDLT
        // factorization object, which Eigen makes neither copyable nor
        // movable (its own base class only implements move at a level
        // that still leaves the derived solver type immovable) -- a
        // direct VDiffusionCore member would make VParticipant itself
        // immovable, and bruss_v_create() (shim.cpp) move-constructs
        // the handle's VParticipant from VParticipant::make()'s return.
        core_(std::make_unique<VDiffusionCore>(
            v_grid_.nx(), v_grid_.ny(), diffusion,
            initial_condition(/*is_u=*/false, v_grid_.nx(), v_grid_.ny()))),
        weights_vu_(std::move(weights_vu)),
        weights_uv_(std::move(weights_uv)) {}

  /// provided = P_VU(state()): V's own new state remapped onto the U grid.
  void remap_state_to_u(Vector* out) {
    std::vector<double> v_native(static_cast<std::size_t>(n_v_));
    v_grid_.to_native(core_->state(), &v_native);
    std::vector<double> u_native(static_cast<std::size_t>(n_u_), 0.0);
    (void)fuse::apply_weights(weights_vu_, v_native.data(), u_native.data(), 1);
    u_grid_.to_canonical(u_native, out);
  }

  fuse::native::NativeMeshBackend backend_;
  NativeGrid u_grid_, v_grid_;
  int n_u_, n_v_;
  std::unique_ptr<VDiffusionCore> core_;
  fuse::RemapWeights weights_vu_;  ///< V grid -> U grid
  fuse::RemapWeights weights_uv_;  ///< U grid -> V grid
};

}  // namespace brusselator