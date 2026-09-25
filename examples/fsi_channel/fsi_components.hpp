#pragma once

/// The two `fuse::Component`s of the FSI channel: a beam and a fluid,
/// strongly coupled through FUSE's iteration region on the non-matching 1D
/// interface described in docs/user/fsi-channel.md.
///
/// ONE CLOCK. Both components declare the SAME `ComponentDecl::interval`
/// (the physical time step dt, exactly as a `Rational`), so every period is
/// one dt and there is no sub-cycling or waveform coupling in play here --
/// the iteration region inside a period is what makes the coupling
/// implicit, not the clock.
///
/// FIELDS (see docs/user/interface-transfer.md for `H`/`H^T D`):
///   beam provides "beam_velocity"  (nodal, ContinuousGLL order 1)
///   fluid requires "fluid_velocity_in" (FV) <- consistent H
///   fluid provides "fluid_force"   (FV, Extensive, a DENSITY)
///   beam requires "beam_load"      (nodal, ContinuousGLL order 1, Extensive)
///          <- consistent-transpose, H^T D
///
/// UNIT CONVENTION ON THE INTERFACE MESHES. FUSE's "generated:line:n" mesh
/// spans the NORMALISED interval [0, 1] regardless of the physical channel
/// length L -- H and H^T only ever see that normalised coordinate. So the
/// "fluid_force" density this component provides is not the wall traction
/// itself: it is scaled by L so that, once `consistent-transpose`
/// multiplies by the SOURCE region's own (normalised) cell measure
/// 1/Nx_fluid, the delivered nodal amount is the correctly-scaled physical
/// force in newtons (per unit channel depth): traction[i] * (L / Nx_fluid)
/// = traction[i] * dx_fluid, the physical force on that one wall cell.
/// `docs/user/fsi-channel.md` carries the same derivation for the record.
///
/// NODAL LOAD -> DISTRIBUTED LINE LOAD. `beam_load`, once delivered, is
/// already the physical force (impressed at a node from the fluid's H^T D)
/// in newtons -- a POINT load, not what `BeamSolver::step`/`solve_static`
/// consume (a DISTRIBUTED q(x_i), N/m). `BeamSolver::tributary_length(i)`
/// is exactly the conversion factor (docs/user/fsi-channel.md derives it):
/// q(x_i) = beam_load[i] / tributary_length(i).

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <fuse/runtime/component.hpp>

#include "beam_solver.hpp"
#include "stokes_solver.hpp"

namespace fsi {

/// Physical scale factor converting a wall traction [Pa] (force per unit
/// area, i.e. per unit length in this per-unit-depth 2D model) into the
/// "fluid_force" FUSE field's density convention -- see the file header.
[[nodiscard]] inline double traction_to_fuse_density(double traction, double L) {
  return traction * L;
}

class BeamComponent final : public fuse::Component {
 public:
  BeamComponent(std::string name, BeamParams p)
      : name_(std::move(name)), beam_(p) {
    const std::size_t n = beam_.n_nodes();
    velocity_out_.assign(n, 0.0);
    load_in_.assign(n, 0.0);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const std::string& fname = g.field(f).name;
      double* base = fname == "beam_velocity" ? velocity_out_.data()
                                              : load_in_.data();
      const std::size_t count = fname == "beam_velocity" ? velocity_out_.size()
                                                          : load_in_.size();
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1, .count = count},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const std::size_t n = beam_.n_nodes();
    std::vector<double> q(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      q[i] = load_in_[i] / beam_.tributary_length(i);
    }
    if (dt > 0.0) {
      // KNOWN_ISSUES.md D9's own resolution: `load_in_` (hence `q`) is now
      // the mid-step (t_{n+1/2}) load H^T D delivers from the fluid's own
      // mid-step traction (FluidComponent::advance's own comment) --
      // step_midpoint_load() consumes it AS a mid-step load directly
      // (BeamSolver::step_midpoint_load's own comment has the derivation),
      // rather than step()'s "force at t_{n+1}" convention, which this
      // load is NOT.
      beam_.step_midpoint_load(dt, q);
    }
    const std::vector<double>& v = beam_.velocity();
    std::copy(v.begin(), v.end(), velocity_out_.begin());
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_ = beam_.save();
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    beam_.restore(saved_);
    const std::vector<double>& v = beam_.velocity();
    std::copy(v.begin(), v.end(), velocity_out_.begin());
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] const BeamSolver& solver() const { return beam_; }
  [[nodiscard]] BeamSolver& solver() { return beam_; }
  [[nodiscard]] const std::vector<double>& velocity_out() const {
    return velocity_out_;
  }
  [[nodiscard]] const std::vector<double>& load_in() const { return load_in_; }

  /// Diagnostic-only counterpart to `FluidComponent::
  /// refresh_force_from_current_state()`: syncs the registered,
  /// DERIVED "beam_velocity" field (`velocity_out_`) to the solver's
  /// CURRENT `wdot`, for a caller that just bypassed `advance()` via a
  /// direct `solver().set_state(...)` call (see that method's own
  /// comment for why a direct solver seed leaves a derived, registered
  /// field stale otherwise). Used by `fsi_energy_diag_test.cpp` to build
  /// the coupled step map's affine columns from an arbitrary carried
  /// state, not from a real trajectory.
  void refresh_velocity_from_current_state() {
    const std::vector<double>& v = beam_.velocity();
    std::copy(v.begin(), v.end(), velocity_out_.begin());
  }

 private:
  std::string          name_;
  BeamSolver            beam_;
  std::vector<double>   velocity_out_;  // Provides: beam_velocity, n+1
  std::vector<double>   load_in_;       // Requires: beam_load, n+1
  BeamSolver::State      saved_;
};

class FluidComponent final : public fuse::Component {
 public:
  FluidComponent(std::string name, ChannelParams p,
                std::function<double(double)> inflow_profile)
      : name_(std::move(name)),
        fluid_(p),
        inflow_(std::move(inflow_profile)) {
    velocity_in_.assign(p.nx, 0.0);
    velocity_in_t0_.assign(p.nx, 0.0);
    force_out_.assign(p.nx, 0.0);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const std::string& fname = g.field(f).name;
      double* base = fname == "fluid_force" ? force_out_.data()
                                            : velocity_in_.data();
      const std::size_t count = fname == "fluid_force" ? force_out_.size()
                                                        : velocity_in_.size();
      auto r = reg.bind(f, fuse::FieldRef{base,
                                          fuse::Layout{.stride = 1, .count = count},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double t0 = ctx.t_begin.to_double();
    const double t1 = ctx.t_end.to_double();
    const double dt = t1 - t0;
    if (dt > 0.0) {
      // velocity_in_t0_ (fixed for every sweep of this period, captured by
      // save_state() at the period's TRUE start) is the wall velocity at
      // t0; velocity_in_ (whatever this sweep's coupling channel just
      // delivered) is the wall velocity at t1. See StokesSolver::step's
      // own header comment for why both time levels matter.
      fluid_.step(dt, t0, t1, velocity_in_t0_, velocity_in_, inflow_);
    }
    // wall_traction() reads the mid-step (t_{n+1/2}) traction -- see
    // KNOWN_ISSUES.md D9 ("Partitioned time-level consistency: the force
    // side, round 2" in docs/user/fsi-channel.md) for the full derivation
    // and, importantly, for what this change did and did NOT fix.
    // Crank-Nicolson's own trapezoidal terms determine the pressure ONLY
    // at t_{n+1/2} (a fully-implicit pressure GRADIENT, no p0 carried as
    // state -- StokesSolver::step's own comment has the full derivation),
    // so `StokesSolver::wall_traction()` genuinely IS the load at
    // t_{n+1/2}, delivered to `BeamComponent::step_midpoint_load()` as
    // exactly that, not mislabelled as "the load at t_{n+1}". An earlier
    // round instead made the pressure GRADIENT trapezoidal to get a
    // t1-accurate p1, carrying the previous step's p0 as literal state --
    // that state has a genuine |lambda|=1 parasitic component
    // (incompressibility only ever pins the p0/p1 AVERAGE), and this
    // change DOES remove it: fluid-alone's own step-map spectral radius
    // moves from EXACTLY 1 to 0.9946 under this fix, and the p-block of
    // the coupled step map's own input sensitivity drops to exactly zero
    // (test/verification/fsi_energy_diag_test.cpp, both measured
    // directly). What this change does NOT do is fix D9's headline
    // symptom: the coupled system's own step-map spectral radius is
    // UNCHANGED to 4 significant figures (1.115467 -> 1.115466), and an
    // ablation (zeroing the pressure term in the delivered traction)
    // gives the identical spectral radius before and after this change
    // (1.000512 either way) -- so the parasitic p0 mode this fixes is a
    // REAL defect, correctly removed, but it is demonstrably not the (or
    // not the only) mechanism sustaining D9's energy growth. That
    // mechanism remains open; see KNOWN_ISSUES.md D9's own "round 2"
    // section for the full evidence.
    const std::vector<double> traction = fluid_.wall_traction();
    const double L = fluid_.params().L;
    for (std::size_t i = 0; i < traction.size(); ++i) {
      force_out_[i] = traction_to_fuse_density(traction[i], L);
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_ = fluid_.save();
    // Captures the wall velocity at the TRUE start of this period, before
    // any sweep of the upcoming iteration region has overwritten
    // velocity_in_ with a new guess -- see advance()'s own comment.
    velocity_in_t0_ = velocity_in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    fluid_.restore(saved_);
    const std::vector<double> traction = fluid_.wall_traction();
    const double L = fluid_.params().L;
    for (std::size_t i = 0; i < traction.size(); ++i) {
      force_out_[i] = traction_to_fuse_density(traction[i], L);
    }
    return fuse::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] const StokesSolver& solver() const { return fluid_; }
  [[nodiscard]] StokesSolver& solver() { return fluid_; }

  /// Recomputes `force_out_` (the "fluid_force" registered field) from the
  /// CURRENT solver state, without advancing time. Needed after any DIRECT
  /// call to `solver().set_state(...)` (a test-only bypass of the normal
  /// advance()/register_fields() write path): set_state() has no way to
  /// know this component also owns a derived, registered field that must
  /// stay in sync, so a caller that seeds the solver directly must call
  /// this too -- otherwise the RUNTIME's own checkpoint machinery captures
  /// a STALE force_out_ (whatever advance() last wrote, possibly zero from
  /// construction) at save_state() time, while restore_state() recomputes
  /// force_out_ FRESH from the (now-seeded) solver state -- a genuine
  /// mismatch the checkpoint-consistency check correctly refuses.
  void refresh_force_from_current_state() {
    // wall_traction() is now a CACHED value (StokesSolver::step's own
    // comment), normally refreshed only inside step() -- a direct
    // set_state() bypass needs solver().recompute_traction_from_current_
    // state() first, or this would read whatever traction_mid_ happened
    // to hold before the bypass (stale, or the FlowState default zero).
    fluid_.recompute_traction_from_current_state();
    const std::vector<double> traction = fluid_.wall_traction();
    const double L = fluid_.params().L;
    for (std::size_t i = 0; i < traction.size(); ++i) {
      force_out_[i] = traction_to_fuse_density(traction[i], L);
    }
  }

  /// Diagnostic-only: directly overwrites the registered, REQUIRES
  /// "fluid_velocity_in" field. A caller building an arbitrary carried
  /// state (rather than a real trajectory point) needs this because
  /// `save_state()` treats `velocity_in_t0_` as part of what it
  /// checkpoints (a plain copy of whatever `velocity_in_` holds at the
  /// moment `run_period()` calls it, BEFORE this period's own
  /// `wall_velocity` channel delivery has run) -- see that comment.
  /// Normally `velocity_in_` is written only by the channel delivery
  /// instruction (H applied to the beam's `beam_velocity`), so it is
  /// otherwise unreachable from outside the graph machinery. Used by
  /// `fsi_energy_diag_test.cpp`, which needs `velocity_in_t0_` to be an
  /// independently perturbable entry of the coupled step map's carried
  /// state, exactly as `Component::save_state()`'s own contract implies.
  ///
  /// ELEMENT-WISE, not a vector assignment: `register_fields()` bound
  /// `velocity_in_.data()` (the ORIGINAL buffer address) into the
  /// runtime's FieldRegistry, which keeps that raw pointer for the whole
  /// Runner's lifetime. `velocity_in_ = std::move(v)` would free that
  /// buffer and give `velocity_in_` a NEW one at a different address,
  /// leaving the registry's copy of the OLD pointer dangling -- a
  /// heap-use-after-free the runtime's very next field access hits
  /// (caught directly, this round, via AddressSanitizer: a genuinely
  /// non-deterministic SIGTRAP/SIGBUS depending on heap layout, not a
  /// library defect at all). Every other in-place field mutator in this
  /// file (`refresh_force_from_current_state()`,
  /// `refresh_velocity_from_current_state()`) already uses `std::copy`
  /// into the existing storage for exactly this reason; this one hadn't,
  /// and is fixed to match.
  void set_velocity_in_for_test(const std::vector<double>& v) {
    if (v.size() != velocity_in_.size()) {
      throw std::invalid_argument(
          "FluidComponent::set_velocity_in_for_test: size mismatch");
    }
    std::copy(v.begin(), v.end(), velocity_in_.begin());
  }
  [[nodiscard]] const std::vector<double>& velocity_in() const {
    return velocity_in_;
  }
  [[nodiscard]] const std::vector<double>& force_out() const { return force_out_; }

 private:
  std::string                   name_;
  StokesSolver                   fluid_;
  std::function<double(double)>  inflow_;
  std::vector<double>            velocity_in_;  // Requires: fluid_velocity_in, nx
  std::vector<double>            velocity_in_t0_;  // wall velocity at t0 (this period's start)
  std::vector<double>            force_out_;    // Provides: fluid_force, nx
  FlowState                       saved_;
};

}  // namespace fsi
