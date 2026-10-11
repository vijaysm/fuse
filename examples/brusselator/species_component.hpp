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

#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/num/waveform.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/window_coupler.hpp"

#include "brusselator_physics.hpp"
#include "species_core.hpp"

namespace brusselator {

/// S7.2 WindowNewton (newton-krylov.md Sec. 9.5-9.6): a FNV-1a-style
/// combine of a fixed-size list of doubles' own bit patterns, into the
/// `std::uint64_t` `fuse::LinearizationToken::signature()` returns.
/// `UComponent`/`VComponent`'s own tokens never actually linearise AT a
/// point (Sec. 9.5: the preconditioner is a FIXED operator, independent
/// of the Newton iterate) -- `LinearizationToken::signature()`'s own
/// doc licenses exactly this: "for a component that only ever linearises
/// once, a fixed constant". Built from (dt, diffusion) per the lead's
/// own instruction, rather than literally returning a constant, so a
/// genuine change to either (a different `--dt-u`/`--dt-v`/`--alpha` run)
/// is reflected, even though nothing downstream of this track's own
/// wiring ever compares two signatures against each other today.
[[nodiscard]] inline std::uint64_t fnv1a_combine_doubles(
    std::initializer_list<double> values) {
  std::uint64_t h = 0xcbf29ce484222325ULL;
  for (const double v : values) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    h ^= bits;
    h *= 0x100000001b3ULL;
  }
  return h;
}

/// U_t = a*Lap(U) + B - U + q + f. Backward Euler for diffusion and the
/// U decay; q and f enter explicitly (f exactly integrated over its
/// active duration within the step; q is the lagged reaction described
/// in the file header). The solve itself is `UCore` (species_core.hpp,
/// Time windows track Unit 2) -- this class is now a thin fuse::Component
/// wrapper around it, keeping ONLY the storage FUSE itself must bind
/// through (`provided_`/`required_`, read/written by raw pointer by both
/// `fuse::WindowCoupler` and `fuse::Runner`'s own FieldRegistry).
class UComponent final : public fuse::Component,
                         public fuse::ImplicitComponent,
                         public fuse::LinearizationToken {
 public:
  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `own_interval`/
  /// `partner_interval`/`window_length` are new, TRAILING, DEFAULTED
  /// parameters -- every existing caller (examples/brusselator/main.cpp's
  /// `--driver window` construction, and every `test/verification/
  /// brusselator_*_test.cpp` that builds a `UComponent` directly without
  /// selecting `--accel newton`) compiles and behaves EXACTLY as before,
  /// since none of those call sites ever reach `evaluate_residual()`/
  /// `linearization()` (`main.cpp`'s own `--accel newton` path is the
  /// one caller that passes real values here). Defaults to `Rational{0}`
  /// -- an explicit, checkable "not set" value (`capabilities()` below
  /// reads it directly, never inferring "was this passed" by comparing
  /// against some other default a legitimate caller might also choose),
  /// since a real interval or window length is never zero or negative.
  UComponent(std::string name, const NativeGrid& grid, PhysicsConfig cfg, Vector initial_v_on_u,
            fuse::Rational own_interval = fuse::Rational{0},
            fuse::Rational partner_interval = fuse::Rational{0},
            fuse::Rational window_length = fuse::Rational{0})
      : name_(std::move(name)),
        n_(grid.size()),
        core_(grid, cfg, initial_v_on_u),
        required_(std::move(initial_v_on_u)),
        provided_(core_.provided()),
        diffusion_(cfg.diffusion),
        own_interval_(own_interval),
        partner_interval_(partner_interval),
        window_length_(window_length) {}

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
  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `kCapabilityResidual`/
  /// `kCapabilityLinearization` are set ONLY when the window metadata
  /// constructor parameters were actually passed (`own_interval_ >
  /// Rational{0}`) -- a `UComponent` built by a caller that never set
  /// them (every pre-existing test file, and the `--driver window` path)
  /// genuinely CANNOT answer `evaluate_residual()`/`linearization()`
  /// correctly (it has no window length or partner interval to
  /// reconstruct a substep grid from), and the checked-not-trusted
  /// pattern `kCapabilityRollback` already follows (component.hpp) exists
  /// precisely so a region's own setup-time check reads the BIT, not an
  /// optimistic assumption that implementing the method is enough.
  [[nodiscard]] std::uint64_t capabilities() const override {
    std::uint64_t caps = fuse::make_capabilities(
        /*rollback=*/true, fuse::DeterminismLevel::Bitwise, /*hidden_state=*/true);
    if (own_interval_ > fuse::Rational{0}) {
      caps |= fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
    }
    return caps;
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9): the window-member
  /// generalisation of `fuse::ImplicitComponent::evaluate_residual()` --
  /// see that method's own comment (component.hpp) for what "a frozen
  /// time level" means here: `ctx` spans the WHOLE window `[T0, T1)`;
  /// `interface_in` is this member's own stacked import over that window
  /// (the partner's own `(element, sample)` layout, `n_` elements by
  /// `partner_interval_`'s own capacity); `residual_out` is this
  /// member's own stacked EXPORT over the SAME window (`n_` elements by
  /// `own_interval_`'s own capacity) -- this component's own `G`, run by
  /// hand over its own substep grid, NEVER the genuine residual `x - G`
  /// (only the Runner assembles that, one level up). NOT called by this
  /// track's own Newton runtime wiring (`Runner::Impl::evaluate_window_
  /// region_residual()` reaches this member's own `G` through
  /// `advance_window()` directly, mirroring the ordinary seam's own
  /// `DirichletSlab::evaluate_residual()` precedent, examples/dn_heat/
  /// dn_component.hpp) -- this override exists to make `kCapabilityResidual`
  /// true for a genuine reason.
  ///
  /// PURITY. Runs every substep on a `core_.save_state()`/`restore_
  /// state()` bracket, never the live state `advance()` mutates --
  /// `test/runtime/implicit_component_test.cpp`'s own contract (component.
  /// hpp) is what this bracket exists to satisfy.
  fuse::Result<void> evaluate_residual(const fuse::StepContext& ctx,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    const std::size_t own_capacity = own_window_capacity();
    const std::size_t partner_capacity = fuse::window_sample_capacity(
        window_length_, partner_interval_, fuse::DtPolicy::Adjustable);
    const std::size_t expected_cols = static_cast<std::size_t>(n_) * partner_capacity;
    const std::size_t expected_rows = static_cast<std::size_t>(n_) * own_capacity;
    if (interface_in.size() != expected_cols || residual_out.size() != expected_rows) {
      return fuse::Failure{"E-BRUSS-COMP",
                           "UComponent::evaluate_residual: expected " +
                               std::to_string(expected_cols) + " in / " +
                               std::to_string(expected_rows) + " out, got " +
                               std::to_string(interface_in.size()) + "/" +
                               std::to_string(residual_out.size())};
    }

    const fuse::Rational T0 = ctx.t_begin;
    const fuse::Rational T1 = ctx.t_end;
    // The partner's own sample TIMES, from the graph alone -- the SAME
    // `fuse::window_sample_times()` `Runner::Impl::advance_window()`
    // itself uses for its own import waveform (`member_start = T0`
    // rather than `Rational{0}` with a separate `window_index`: the two
    // are equivalent whenever `T0` is itself the window's own absolute
    // start, which `ctx` always is for this call by construction).
    const std::vector<fuse::Rational> partner_times = fuse::window_sample_times(
        partner_interval_, fuse::DtPolicy::Adjustable, T0, window_length_, 0);

    std::vector<fuse::Waveform> partner_wf;
    partner_wf.reserve(static_cast<std::size_t>(n_));
    for (int e = 0; e < n_; ++e) {
      fuse::Waveform wf(1, static_cast<int>(partner_capacity), fuse::TemporalBasis::Linear, 1);
      wf.reset(T0, T1);
      for (std::size_t k = 0; k < partner_times.size(); ++k) {
        double v = interface_in[static_cast<std::size_t>(e) * partner_capacity + k];
        const fuse::Result<void> p = wf.push(partner_times[k], &v);
        if (!p.ok()) return p.error();
      }
      partner_wf.push_back(std::move(wf));
    }

    const auto planned =
        fuse::detail::plan_substeps(T0, own_interval_, T1, fuse::DtPolicy::Adjustable);
    const std::vector<fuse::detail::WindowSubstep>& steps = planned.first;

    core_.save_state();
    for (std::size_t e = 0; e < static_cast<std::size_t>(n_); ++e) {
      residual_out[e * own_capacity + 0] = core_.provided()[static_cast<int>(e)];
    }
    Vector required_substep(n_);
    for (std::size_t s_idx = 0; s_idx < steps.size(); ++s_idx) {
      const fuse::detail::WindowSubstep& sub = steps[s_idx];
      for (int e = 0; e < n_; ++e) {
        const fuse::Result<double> mv =
            fuse::detail::mean_over(partner_wf[static_cast<std::size_t>(e)], sub.t0, sub.t1);
        if (!mv.ok()) { core_.restore_state(); return mv.error(); }
        required_substep[e] = mv.value();
      }
      const fuse::Result<void> r = core_.step(
          sub.t0.to_double(), sub.t1.to_double(), (sub.t1 - sub.t0).to_double(),
          required_substep);
      if (!r.ok()) { core_.restore_state(); return r; }
      for (std::size_t e = 0; e < static_cast<std::size_t>(n_); ++e) {
        residual_out[e * own_capacity + (s_idx + 1)] = core_.provided()[static_cast<int>(e)];
      }
    }
    core_.restore_state();
    return fuse::Result<void>::success();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }
  [[nodiscard]] fuse::LinearizationToken* linearization() override { return this; }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): neither this
  /// component's own `evaluate_residual()`'s true Jacobian action NOR
  /// its transpose is ever computed -- this token exists only for
  /// `apply_preconditioner()` below (a FIXED, per-sample diffusion
  /// solve, never a function of a linearisation point); the Newton
  /// solver's own matrix-free action (`fuse::num::MffdJacobianAction`,
  /// over `F` directly) is what computes the real one, and this track's
  /// own wiring (`Runner::Impl::run_region_newton_window()`) never calls
  /// either of these two -- confirmed by `UComponentToken.
  /// ApplyJacobianIsNeverCalledDuringAWindowNewtonSolve` (test/
  /// verification/brusselator_species_core_decomposition_test.cpp).
  fuse::Result<void> apply_jacobian(std::span<const double>,
                                    std::span<double>) override {
    return fuse::Failure{"E-NEWTON-NO-JACOBIAN-ACTION",
                         "UComponent's own LinearizationToken exists only for its "
                         "apply_preconditioner(); the Newton solver's own matrix-free "
                         "Jacobian action computes the real one over F directly and "
                         "never calls this"};
  }
  fuse::Result<void> apply_jacobian_transpose(std::span<const double>,
                                              std::span<double>) override {
    return fuse::Failure{"E-NEWTON-NO-JACOBIAN-ACTION",
                         "UComponent's own LinearizationToken exists only for its "
                         "apply_preconditioner(); nothing in this track's own wiring "
                         "calls the transpose action either"};
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `z = M^-1 r`, PER
  /// SAMPLE of this component's own window-export block -- a block-
  /// DIAGONAL-IN-TIME approximation of the genuinely lower-triangular-
  /// in-time window Jacobian (Sec. 9.5's own derivation), each sample
  /// smoothed by `UCore::apply_diffusion_preconditioner()`, the SAME
  /// already-factored diffusion system `step()` itself solves every
  /// substep, applied here with `own_interval_`'s own length for EVERY
  /// sample (including sample 0, the window-start value, and the
  /// window's own final, possibly-SHORTER-than-`own_interval_` substep)
  /// -- a deliberate simplification, not an attempt at the exact
  /// per-substep `dt`: the preconditioner's own job is to cut GMRES
  /// iterations, not to reproduce the window map exactly, and a slightly
  /// mis-sized diagonal scale does that job approximately either way.
  fuse::Result<void> apply_preconditioner(std::span<const double> r,
                                          std::span<double> z) override {
    const std::size_t capacity = own_window_capacity();
    if (r.size() != rows() || z.size() != cols()) {
      return fuse::Failure{"E-BRUSS-COMP",
                           "UComponent::apply_preconditioner: expected " +
                               std::to_string(rows()) + " in / " +
                               std::to_string(cols()) + " out, got " +
                               std::to_string(r.size()) + "/" + std::to_string(z.size())};
    }
    Vector r_elem(n_);
    for (std::size_t s = 0; s < capacity; ++s) {
      for (int e = 0; e < n_; ++e) {
        r_elem[e] = r[static_cast<std::size_t>(e) * capacity + s];
      }
      const fuse::Result<Vector> solved =
          core_.apply_diffusion_preconditioner(own_interval_.to_double(), r_elem);
      if (!solved.ok()) return solved.error();
      for (int e = 0; e < n_; ++e) {
        z[static_cast<std::size_t>(e) * capacity + s] = solved.value()[e];
      }
    }
    return fuse::Result<void>::success();
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `rows() == cols()`,
  /// both this component's own EXPORT block length -- DELIBERATELY NOT
  /// `evaluate_residual()`'s own (import-length, export-length) pair
  /// (which genuinely differ here): `fuse::num::BlockJacobiPreconditioner::
  /// PreconditionerBlock` (`fuse/num/newton_krylov.hpp`) ties one
  /// `{offset, length}` pair to BOTH `r` and `z`, which only a SQUARE
  /// token can honour, and `apply_jacobian`'s own rows/cols meaning is
  /// moot here anyway -- it is refused, never called.
  [[nodiscard]] std::size_t rows() const override {
    return static_cast<std::size_t>(n_) * own_window_capacity();
  }
  [[nodiscard]] std::size_t cols() const override { return rows(); }
  [[nodiscard]] std::uint64_t signature() const override {
    return fnv1a_combine_doubles({own_interval_.to_double(), diffusion_});
  }

  [[nodiscard]] double* provided_ptr() { return provided_.data(); }
  [[nodiscard]] double* required_ptr() { return required_.data(); }
  [[nodiscard]] const Vector& state() const { return core_.state(); }
  [[nodiscard]] double external_total() const { return core_.external_total(); }
  [[nodiscard]] double exchange_out() const { return core_.exchange_out(); }

 private:
  [[nodiscard]] std::size_t own_window_capacity() const {
    return fuse::window_sample_capacity(window_length_, own_interval_,
                                        fuse::DtPolicy::Adjustable);
  }

  std::string name_;
  int n_;
  UCore core_;
  Vector required_;   ///< P_VU(V) mean over this step, on the U grid -- FUSE-bound storage
  Vector provided_;   ///< q, on the U grid, valid at ctx.t_end -- FUSE-bound storage
  double diffusion_;  ///< cfg.diffusion, captured for LinearizationToken::signature() only
  /// S7.2 WindowNewton: this member's own window metadata -- `Rational{0}`
  /// (unset) unless the constructor's own trailing parameters were given
  /// real values. See `capabilities()`'s own comment for why the two
  /// Newton capability bits are gated on `own_interval_ > Rational{0}`.
  fuse::Rational own_interval_;
  fuse::Rational partner_interval_;
  fuse::Rational window_length_;
};

/// V_t = a*Lap(V) - q. Backward Euler for diffusion; the received q
/// (raw, on the U grid) is remapped onto V's own grid with `weights_uv`
/// before being subtracted.
class VComponent final : public fuse::Component,
                         public fuse::ImplicitComponent,
                         public fuse::LinearizationToken {
 public:
  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `own_interval`/
  /// `partner_interval`/`window_length`, trailing and defaulted, mirror
  /// `UComponent`'s own identical addition -- see that constructor's own
  /// comment for why every existing caller is unaffected.
  VComponent(std::string name, const NativeGrid& u_grid, const NativeGrid& v_grid,
            PhysicsConfig cfg, const fuse::RemapWeights* weights_uv,
            const fuse::RemapWeights* weights_vu,
            fuse::Rational own_interval = fuse::Rational{0},
            fuse::Rational partner_interval = fuse::Rational{0},
            fuse::Rational window_length = fuse::Rational{0})
      : name_(std::move(name)),
        u_grid_(&u_grid),
        v_grid_(&v_grid),
        n_u_(u_grid.size()),
        n_v_(v_grid.size()),
        core_(v_grid, cfg.diffusion),
        required_(Vector::Zero(n_u_)),
        provided_(n_u_),
        weights_uv_(weights_uv),
        weights_vu_(weights_vu),
        diffusion_(cfg.diffusion),
        own_interval_(own_interval),
        partner_interval_(partner_interval),
        window_length_(window_length) {
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
  /// See `UComponent::capabilities()`'s own comment -- the identical
  /// gate, on the identical sentinel.
  [[nodiscard]] std::uint64_t capabilities() const override {
    std::uint64_t caps = fuse::make_capabilities(
        /*rollback=*/true, fuse::DeterminismLevel::Bitwise, /*hidden_state=*/true);
    if (own_interval_ > fuse::Rational{0}) {
      caps |= fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
    }
    return caps;
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9): `UComponent::evaluate_
  /// residual()`'s own account applies verbatim, with ONE addition this
  /// member's own `advance()` already has to make and this override must
  /// mirror: every substep's own import mean is remapped U grid -> V grid
  /// (`weights_uv_`) BEFORE `core_.step()`, and every substep's own
  /// output state is remapped V grid -> U grid (`remap_u_to_native_u()`
  /// below, the SAME direction `remap_state_to_u()` already uses) before
  /// being written into `residual_out` -- both of this member's own
  /// window-transfer fields live on the U grid (`n_u_`), never `n_v_`
  /// (this file's own header comment), including the ones this override
  /// reads/writes.
  fuse::Result<void> evaluate_residual(const fuse::StepContext& ctx,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    const std::size_t own_capacity = own_window_capacity();
    const std::size_t partner_capacity = fuse::window_sample_capacity(
        window_length_, partner_interval_, fuse::DtPolicy::Adjustable);
    const std::size_t expected_cols = static_cast<std::size_t>(n_u_) * partner_capacity;
    const std::size_t expected_rows = static_cast<std::size_t>(n_u_) * own_capacity;
    if (interface_in.size() != expected_cols || residual_out.size() != expected_rows) {
      return fuse::Failure{"E-BRUSS-COMP",
                           "VComponent::evaluate_residual: expected " +
                               std::to_string(expected_cols) + " in / " +
                               std::to_string(expected_rows) + " out, got " +
                               std::to_string(interface_in.size()) + "/" +
                               std::to_string(residual_out.size())};
    }

    const fuse::Rational T0 = ctx.t_begin;
    const fuse::Rational T1 = ctx.t_end;
    const std::vector<fuse::Rational> partner_times = fuse::window_sample_times(
        partner_interval_, fuse::DtPolicy::Adjustable, T0, window_length_, 0);

    std::vector<fuse::Waveform> partner_wf;  // on the U grid, one per U-grid element
    partner_wf.reserve(static_cast<std::size_t>(n_u_));
    for (int e = 0; e < n_u_; ++e) {
      fuse::Waveform wf(1, static_cast<int>(partner_capacity), fuse::TemporalBasis::Linear, 1);
      wf.reset(T0, T1);
      for (std::size_t k = 0; k < partner_times.size(); ++k) {
        double v = interface_in[static_cast<std::size_t>(e) * partner_capacity + k];
        const fuse::Result<void> p = wf.push(partner_times[k], &v);
        if (!p.ok()) return p.error();
      }
      partner_wf.push_back(std::move(wf));
    }

    const auto planned =
        fuse::detail::plan_substeps(T0, own_interval_, T1, fuse::DtPolicy::Adjustable);
    const std::vector<fuse::detail::WindowSubstep>& steps = planned.first;

    core_.save_state();
    Vector start_u(n_u_);
    remap_state_to_u(&start_u);
    for (std::size_t e = 0; e < static_cast<std::size_t>(n_u_); ++e) {
      residual_out[e * own_capacity + 0] = start_u[static_cast<int>(e)];
    }
    for (std::size_t s_idx = 0; s_idx < steps.size(); ++s_idx) {
      const fuse::detail::WindowSubstep& sub = steps[s_idx];
      Vector q_on_u(n_u_);
      for (int e = 0; e < n_u_; ++e) {
        const fuse::Result<double> mv =
            fuse::detail::mean_over(partner_wf[static_cast<std::size_t>(e)], sub.t0, sub.t1);
        if (!mv.ok()) { core_.restore_state(); return mv.error(); }
        q_on_u[e] = mv.value();
      }
      Vector q_on_v(n_v_);
      if (const fuse::Result<void> rm = remap_u_to_v(q_on_u, &q_on_v); !rm.ok()) {
        core_.restore_state();
        return rm;
      }
      const fuse::Result<void> r = core_.step((sub.t1 - sub.t0).to_double(), q_on_v);
      if (!r.ok()) { core_.restore_state(); return r; }
      Vector sample_u(n_u_);
      remap_state_to_u(&sample_u);
      for (std::size_t e = 0; e < static_cast<std::size_t>(n_u_); ++e) {
        residual_out[e * own_capacity + (s_idx + 1)] = sample_u[static_cast<int>(e)];
      }
    }
    core_.restore_state();
    return fuse::Result<void>::success();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }
  [[nodiscard]] fuse::LinearizationToken* linearization() override { return this; }

  /// See `UComponent::apply_jacobian()`'s own comment -- identical
  /// reasoning, this component's own token.
  fuse::Result<void> apply_jacobian(std::span<const double>,
                                    std::span<double>) override {
    return fuse::Failure{"E-NEWTON-NO-JACOBIAN-ACTION",
                         "VComponent's own LinearizationToken exists only for its "
                         "apply_preconditioner(); the Newton solver's own matrix-free "
                         "Jacobian action computes the real one over F directly and "
                         "never calls this"};
  }
  fuse::Result<void> apply_jacobian_transpose(std::span<const double>,
                                              std::span<double>) override {
    return fuse::Failure{"E-NEWTON-NO-JACOBIAN-ACTION",
                         "VComponent's own LinearizationToken exists only for its "
                         "apply_preconditioner(); nothing in this track's own wiring "
                         "calls the transpose action either"};
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `UComponent::apply_
  /// preconditioner()`'s own account, with the SAME U-grid/V-grid remap
  /// `evaluate_residual()` above needs: `r`/`z` are on the U grid (this
  /// member's own export block, like every one of its window-transfer
  /// fields); each sample is remapped to V's own grid, smoothed by
  /// `VDiffusionCore::apply_diffusion_preconditioner()` (the SAME
  /// already-factored system `step()` itself solves), and remapped back.
  fuse::Result<void> apply_preconditioner(std::span<const double> r,
                                          std::span<double> z) override {
    const std::size_t capacity = own_window_capacity();
    if (r.size() != rows() || z.size() != cols()) {
      return fuse::Failure{"E-BRUSS-COMP",
                           "VComponent::apply_preconditioner: expected " +
                               std::to_string(rows()) + " in / " +
                               std::to_string(cols()) + " out, got " +
                               std::to_string(r.size()) + "/" + std::to_string(z.size())};
    }
    Vector r_u(n_u_);
    for (std::size_t s = 0; s < capacity; ++s) {
      for (int e = 0; e < n_u_; ++e) r_u[e] = r[static_cast<std::size_t>(e) * capacity + s];
      Vector r_v(n_v_);
      if (const fuse::Result<void> rm = remap_u_to_v(r_u, &r_v); !rm.ok()) return rm;
      const fuse::Result<Vector> solved =
          core_.apply_diffusion_preconditioner(own_interval_.to_double(), r_v);
      if (!solved.ok()) return solved.error();
      Vector z_u(n_u_);
      remap_state_to_u_from(solved.value(), &z_u);
      for (int e = 0; e < n_u_; ++e) z[static_cast<std::size_t>(e) * capacity + s] = z_u[e];
    }
    return fuse::Result<void>::success();
  }

  /// See `UComponent::rows()`/`cols()`'s own comment -- identical
  /// reasoning, this component's own export block (`n_u_` elements, its
  /// own window capacity).
  [[nodiscard]] std::size_t rows() const override {
    return static_cast<std::size_t>(n_u_) * own_window_capacity();
  }
  [[nodiscard]] std::size_t cols() const override { return rows(); }
  [[nodiscard]] std::uint64_t signature() const override {
    return fnv1a_combine_doubles({own_interval_.to_double(), diffusion_});
  }

  [[nodiscard]] double* provided_ptr() { return provided_.data(); }
  [[nodiscard]] double* required_ptr() { return required_.data(); }
  [[nodiscard]] const Vector& state() const { return core_.state(); }
  [[nodiscard]] double exchange_in() const { return core_.exchange_in(); }

 private:
  /// provided_ = P_VU(state_): V's own new state remapped onto the U grid.
  void remap_state_to_u(Vector* out) { remap_state_to_u_from(core_.state(), out); }

  /// `remap_state_to_u()`'s own body, generalised to an ARBITRARY
  /// V-grid vector rather than only `core_.state()` -- `apply_
  /// preconditioner()` above needs this for a solved preconditioner
  /// sample that is never `core_.state()` itself.
  void remap_state_to_u_from(const Vector& on_v, Vector* out) {
    std::vector<double> v_native(static_cast<std::size_t>(n_v_));
    v_grid_->to_native(on_v, &v_native);
    std::vector<double> u_native(static_cast<std::size_t>(n_u_), 0.0);
    // weights_vu_ is built with plan.src = V's region, plan.tgt = U's
    // region, so row = U (target) native index, col = V (source) native
    // index -- exactly apply_weights's dst/src convention below.
    (void)fuse::apply_weights(*weights_vu_, v_native.data(), u_native.data(), 1);
    u_grid_->to_canonical(u_native, out);
  }

  /// The forward direction `advance()` already applies to `required_`
  /// before calling `core_.step()` -- factored out here so `evaluate_
  /// residual()`/`apply_preconditioner()` (both need it per substep, on
  /// a value that is never `required_` itself) can call it too.
  [[nodiscard]] fuse::Result<void> remap_u_to_v(const Vector& on_u, Vector* out_v) {
    std::vector<double> native_u(static_cast<std::size_t>(n_u_));
    u_grid_->to_native(on_u, &native_u);
    std::vector<double> native_v(static_cast<std::size_t>(n_v_), 0.0);
    const fuse::Result<void> applied =
        fuse::apply_weights(*weights_uv_, native_u.data(), native_v.data(), 1);
    if (!applied.ok()) return applied;
    v_grid_->to_canonical(native_v, out_v);
    return fuse::Result<void>::success();
  }

  [[nodiscard]] std::size_t own_window_capacity() const {
    return fuse::window_sample_capacity(window_length_, own_interval_,
                                        fuse::DtPolicy::Adjustable);
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
  double diffusion_;  ///< cfg.diffusion, captured for LinearizationToken::signature() only
  fuse::Rational own_interval_;      ///< V's own dt_v -- Rational{0} unless set at construction
  fuse::Rational partner_interval_;  ///< U's own dt_u
  fuse::Rational window_length_;
};

}  // namespace brusselator
