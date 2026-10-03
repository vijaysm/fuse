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
#include "species_core.hpp"

namespace brusselator {

/// U_t = a*Lap(U) + B - U + q + f. Backward Euler for diffusion and the
/// U decay; q and f enter explicitly (f exactly integrated over its
/// active duration within the step; q is the lagged reaction described
/// in the file header). The solve itself is `UCore` (species_core.hpp,
/// Time windows track Unit 2) -- this class is now a thin fuse::Component
/// wrapper around it, keeping ONLY the storage FUSE itself must bind
/// through (`provided_`/`required_`, read/written by raw pointer by both
/// `fuse::WindowCoupler` and `fuse::Runner`'s own FieldRegistry).
class UComponent final : public fuse::Component {
 public:
  UComponent(std::string name, const NativeGrid& grid, PhysicsConfig cfg, Vector initial_v_on_u)
      : name_(std::move(name)),
        n_(grid.size()),
        core_(grid.nx(), grid.ny(), cfg, initial_v_on_u),
        required_(std::move(initial_v_on_u)),
        provided_(core_.provided()) {}

  [[nodiscard]] std::string name() const override { return name_; }

  /// `fuse::WindowCoupler` (examples/brusselator/main.cpp's `--driver
  /// window`) binds `provided_ptr()`/`required_ptr()` directly, never
  /// calling this -- so this body is exercised only by the Runner path
  /// (`--driver runner`, `fuse::Runner::initialize()`), which resolves a
  /// window-scoped member's storage through the `FieldRegistry` exactly as
  /// an ordinary single-clock component's `value_arc`/`feed_arc` is
  /// resolved (window_coupler.hpp's own class comment). Binds `u_out`
  /// (`Intent::Provides`, this component's `q`) to `provided_.data()` and
  /// `u_in` (`Intent::Requires`, the V data this component receives) to
  /// `required_.data()`, both `n_` doubles -- the SAME storage the
  /// WindowCoupler path reads/writes through `provided_ptr()`/
  /// `required_ptr()`, so the two drivers exercise one physics object
  /// rather than two.
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg, const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-BRUSS-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides ? provided_.data() : required_.data();
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1,
                                                       .count = static_cast<std::size_t>(n_)},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    // ctx.dt().to_double(): the EXACT rational step length, rounded ONCE --
    // NOT re-derived from t_end.to_double() - t_begin.to_double() inside
    // UCore::step() (species_core.hpp's own comment on `step()` has the
    // full account of why that extra rounding is not bit-identical to this).
    const fuse::Result<void> r = core_.step(
        ctx.t_begin.to_double(), ctx.t_end.to_double(), ctx.dt().to_double(), required_);
    if (!r.ok()) return r;
    provided_ = core_.provided();
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    core_.save_state();
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    core_.restore_state();
    // provided_ is Sample-0-restamped from the LIVE core_.provided() by
    // every caller that needs the window-start value (the Runner's own
    // window_start_value cache, src/runtime/iteration.cpp) -- restoring
    // it here too keeps a direct (non-window) reader of provided_ptr()
    // consistent with core_ immediately after a rollback, matching what
    // restore_state() already did before this refactor (`provided_` was
    // never itself snapshotted/restored directly either -- only read
    // fresh via the next advance()'s own write -- so this assignment
    // changes nothing observable, it just keeps the two in sync a step
    // sooner).
    provided_ = core_.provided();
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
  [[nodiscard]] const Vector& state() const { return core_.state(); }
  [[nodiscard]] double external_total() const { return core_.external_total(); }
  [[nodiscard]] double exchange_out() const { return core_.exchange_out(); }

 private:
  std::string name_;
  int n_;
  UCore core_;
  Vector required_;   ///< P_VU(V) mean over this step, on the U grid -- FUSE-bound storage
  Vector provided_;   ///< q, on the U grid, valid at ctx.t_end -- FUSE-bound storage
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
        core_(v_grid.nx(), v_grid.ny(), cfg.diffusion,
              initial_condition(/*is_u=*/false, v_grid.nx(), v_grid.ny())),
        required_(Vector::Zero(n_u_)),
        provided_(n_u_),
        weights_uv_(weights_uv),
        weights_vu_(weights_vu) {
    remap_state_to_u(&provided_);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  /// See `UComponent::register_fields()`'s comment: exercised only by the
  /// Runner path. Binds `v_out` (`Intent::Provides`, `P_VU(state_)` on the
  /// U grid) to `provided_.data()` and `v_in` (`Intent::Requires`, the raw
  /// q this component receives on the U grid) to `required_.data()`, both
  /// `n_u_` doubles -- both of this component's window-transfer fields
  /// live on `n_u_`, not `n_v_` (the file header's "both members use
  /// `WindowMember::length = N_u`").
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg, const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-BRUSS-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides ? provided_.data() : required_.data();
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1,
                                                       .count = static_cast<std::size_t>(n_u_)},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
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

    const fuse::Result<void> r = core_.step(dt, q_on_v);
    if (!r.ok()) return r;
    remap_state_to_u(&provided_);
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    core_.save_state();
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    core_.restore_state();
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
  [[nodiscard]] const Vector& state() const { return core_.state(); }
  [[nodiscard]] double exchange_in() const { return core_.exchange_in(); }

 private:
  /// provided_ = P_VU(state_): V's own new state remapped onto the U grid.
  void remap_state_to_u(Vector* out) {
    std::vector<double> v_native(static_cast<std::size_t>(n_v_));
    v_grid_->to_native(core_.state(), &v_native);
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
  VDiffusionCore core_;
  Vector required_;  ///< raw q mean over this step, on the U grid (unremapped) -- FUSE-bound
  Vector provided_;  ///< P_VU(state_), on the U grid -- FUSE-bound
  const fuse::RemapWeights* weights_uv_;  ///< U grid -> V grid
  const fuse::RemapWeights* weights_vu_;  ///< V grid -> U grid
};

}  // namespace brusselator
