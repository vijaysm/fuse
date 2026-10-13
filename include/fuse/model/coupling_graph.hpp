#pragma once

#include <algorithm>
#include <limits>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/merge.hpp"
#include "fuse/model/component.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/region.hpp"
#include "fuse/model/iteration.hpp"

namespace fuse {

using IterationRegionIdx = RegionIdx;

/// Spec §5.3's authored iteration region: which channels iterate, when to
/// stop, and how to accelerate.
///
/// `convergence` HAS NO SAFE DEFAULT. An empty `ConvergenceCriterion` reports
/// satisfied (true is the identity of AND, see num/convergence.hpp), so a
/// region carrying one stops after a single sweep and reports that it
/// converged: an explicit scheme wearing an implicit scheme's report.
/// `verify_iteration_region()` refuses it with `E-ITER-NOCONVERGE`, and the
/// runner calls that from `initialize()`. Set it with `.until(...)`.
///
/// Both this and `acceleration` cross the descriptor (§3.2's narrow waist) and
/// the YAML round trip. Dropping either there would let a graph authored in
/// YAML, or handed across the C ABI, iterate with no convergence test at all;
/// see `graph/descriptor.hpp` and `src/graph/yaml_io.cpp`, which carry them,
/// and test/verification/iteration_test.cpp, which pins the round trip.
struct IterationRegionDecl {
  std::string            name;
  std::vector<ChannelId> channels;
  int                    max_iterations = 1;
  ConvergenceCriterion   convergence;
  /// Defaults to plain Picard. `AccelKind::None` is the honest default for
  /// "the author asked for no acceleration"; anything else would silently
  /// change the iterates.
  ///
  /// IQN-ILS is the recommended strong method. IQN-IMVJ is deliberately NOT
  /// recommended and NOT a default: measured against IQN-ILS on a nonlinear
  /// global contraction with a provably unique root it was 1.3-3.6x slower
  /// with erratic per-window iteration counts (40, 62, 54, 45, 36, 119, 52,
  /// 74, 39 against a flat 15 for IQN-ILS), the signature of a stale
  /// Jacobian, and on a multi-root map it landed in a *different* valid fixed
  /// point in 5 of 8 windows. It has neither a restart mode nor column
  /// filtering here (preCICE ships `imvj-restart-mode` for precisely this
  /// reason), so its retained inverse Jacobian is never refreshed.
  ///
  /// `AccelConfig::filter` (QR1/QR2/QR3) is implemented and active:
  /// `filter_qn_columns()` (src/num/accel_quasi_newton.cpp) is called from
  /// the accelerator, `fuse/model/iteration.hpp` documents each filter's
  /// measured effect on sweep count, and the freeze fixture pins
  /// `FilterKind::QR1` in the canonical bytes. BM-11 (review 2026-10-11):
  /// this comment used to say the filter was "a documented no-op until
  /// Task 7.4", which was true when this declaration was written but
  /// stayed uncorrected after Task 7.4 landed -- a reader tuning
  /// convergence off this comment alone would skip the one knob that can
  /// change sweep count by a large factor.
  AccelConfig            acceleration{.kind = AccelKind::None};

  /// Time windows track: the OPT-IN that will turn this region from today's
  /// single-clock loop into a coupling window (`fuse::CouplingWindowScope`,
  /// `fuse/model/iteration.hpp`). `std::nullopt` (the default) is every
  /// region declared before this member existed, and -- as of this
  /// declaration's own introduction -- every region there is: `schedule()`
  /// does not yet inspect this member. It still runs its existing
  /// contiguous-run check (`mark_regions`, src/schedule/scheduler.cpp)
  /// unconditionally and still refuses a multi-clock region with
  /// `E-SCHED-REGION`, whether or not this field is set. Setting it changes
  /// nothing today; only the descriptor (v7), the YAML round trip, and
  /// `fuse::validate_coupling_window()`'s standalone legality gate
  /// (`fuse/schedule/window.hpp`) can see it. Teaching `schedule()` to call
  /// that gate for a window-scoped region instead of the contiguous-run
  /// check is designed but NOT WIRED (see docs/dev/testing.md's "Five
  /// states" and this track's design notes) -- adding that call is a
  /// separate, not-yet-made change to scheduler.cpp.
  std::optional<CouplingWindowScope> window_scope;

  /// S7.1's Newton-Krylov seam. `true` requires every member of this
  /// region to set `kCapabilityResidual` (fuse/model/capabilities.hpp),
  /// i.e. to implement `fuse::ImplicitComponent`
  /// (fuse/runtime/component.hpp) -- checked at `initialize()` by
  /// `fuse::verify_newton_capability_support()` and refused with
  /// `E-NEWTON-CAPABILITY`, naming the component, when it does not.
  ///
  /// A FIELD ON THE DECLARATION, NOT A NEW `AccelKind` ENUMERATOR: the
  /// Newton solver itself (S7.2) is a separate, not-yet-landed track that
  /// will add its own `AccelKind` value and set this flag when that value
  /// is chosen; this field exists on its own so the capability refusal
  /// (S7.1) can be tested before any Newton code exists, per this track's
  /// own brief. Defaults to `false`: every region declared before this
  /// member existed asks for nothing new.
  bool requires_residual_capability = false;

  /// S7.1's Newton-Krylov seam. `true` additionally requires every member
  /// to set `kCapabilityLinearization`, i.e. to be prepared to return a
  /// usable `fuse::LinearizationToken` from `ImplicitComponent::
  /// linearization()` once it has been asked for a residual -- the
  /// physics-based block-Jacobi preconditioner (S7.2) needs this from
  /// every member it preconditions; a Newton loop driven purely by a
  /// matrix-free Jacobian action with no physics-based preconditioner does
  /// not. Independent of `requires_residual_capability` in the type
  /// system, though every region that sets this should also set that one
  /// -- a component cannot linearise a residual it was never asked to
  /// evaluate. Defaults to `false`.
  bool requires_linearization_capability = false;

  /// S7.2's own Newton-specific options (`fuse::NewtonConfig`,
  /// fuse/model/iteration.hpp): line search strategy, forcing-term
  /// choice, lag, GMRES restart/budget, tolerances. Read ONLY when
  /// `acceleration.kind == AccelKind::Newton`; present unconditionally
  /// (rather than `std::optional`) the same way `AccelConfig` itself is,
  /// so a region declared before this member existed carries SNES's own
  /// defaults with no special-casing anywhere that reads it.
  NewtonConfig newton_config{};
};

class CouplingGraph;

/// `g.iterate({...}).until(...).max_iterations(...)` from spec §5.3.
///
/// Re-resolves its declaration through the graph on every mutating call
/// rather than caching a reference, so it stays valid across an
/// `add_iteration_region()` that reallocates the underlying vector -- the same
/// rule `ChannelBuilder` follows and for the same reason.
class IterationBuilder {
 public:
  IterationBuilder(CouplingGraph& g, IterationRegionIdx id)
      : graph_(&g), id_(id) {}

  /// A whole criterion, e.g. `Converged::Relative(1e-6, "T_wall") &
  /// Converged::Absolute(1e-8, "flux")`.
  IterationBuilder& until(ConvergenceCriterion c);
  /// The single-measure spelling. `Converged::Relative(...)` returns a
  /// `ConvergenceMeasure`, and spec §5.3's example passes one directly.
  IterationBuilder& until(ConvergenceMeasure m);
  IterationBuilder& max_iterations(int n);
  /// Sets `acceleration` to `cfg`. S7.2's own addition: when
  /// `cfg.kind == AccelKind::Newton`, this ALSO sets
  /// `requires_residual_capability = true` (and, when
  /// `newton_config.use_block_jacobi_preconditioner` is already `true` at
  /// the time this is called, `requires_linearization_capability` too) --
  /// so choosing Newton through the generic `AccelConfig` spelling cannot
  /// silently skip the capability gate `verify_newton_capability_support()`
  /// enforces at `initialize()`. Selecting any OTHER kind leaves both
  /// capability flags exactly as they were (a region can ask for the
  /// residual capability without Newton, e.g. a future solver reusing the
  /// same seam; this method never clears a flag it did not itself set).
  IterationBuilder& accelerate(AccelConfig cfg);
  /// Sets `acceleration.initial_relaxation`. There is deliberately no second
  /// `initial_relaxation` field on the declaration: two spellings of one
  /// number is a silent disagreement waiting to happen.
  IterationBuilder& initial_relaxation(double omega);
  IterationBuilder& name(std::string n);

  /// S7.2: selects `AccelKind::Newton` with the given options in one call,
  /// and sets `requires_residual_capability = true` unconditionally plus
  /// `requires_linearization_capability = cfg.use_block_jacobi_preconditioner`
  /// -- the single-method spelling `accelerate({.kind = AccelKind::Newton})`
  /// would otherwise need two more calls (`.requires_residual()` and,
  /// conditionally, `.requires_linearization()`) to say the same thing.
  IterationBuilder& newton(NewtonConfig cfg = {});

  /// S7.1's own fluent setters, left out of that track's brief (its own
  /// comment on `IterationRegionDecl::requires_residual_capability`: "the
  /// Newton solver itself (S7.2) adds its own AccelKind value and sets
  /// these flags") -- added here, by the track that needed them, rather
  /// than guessed at by the track that did not. Exposed independently of
  /// `accelerate()`/`newton()` for a region that wants the residual seam
  /// without Newton itself (S7.1's own non-goal, but not this type's to
  /// forbid).
  IterationBuilder& requires_residual(bool v = true);
  IterationBuilder& requires_linearization(bool v = true);

  [[nodiscard]] IterationRegionIdx id() const noexcept { return id_; }

 private:
  [[nodiscard]] IterationRegionDecl& target();

  CouplingGraph*     graph_;
  IterationRegionIdx id_;
};

/// Spec §2 layer 1: the authored representation. Pure data — no execution,
/// no MPI, no MOAB. This is the only thing users write and the only thing
/// serialised.
class CouplingGraph {
 public:
  // ChannelBuilder re-resolves its target channel through channel_mut() on
  // every mutating call, rather than caching a Channel&, so it stays valid
  // across add_channel() calls that reallocate channels_.
  friend class ChannelBuilder;
  friend class IterationBuilder;

  MeshId add_mesh(MeshDecl d) {
    meshes_.push_back(std::move(d));
    return MeshId{static_cast<std::uint32_t>(meshes_.size() - 1)};
  }

  RegionId add_region(Region r) {
    regions_.push_back(std::move(r));
    return RegionId{static_cast<std::uint32_t>(regions_.size() - 1)};
  }

  ComponentId add_component(ComponentDecl c) {
    components_.push_back(std::move(c));
    return ComponentId{static_cast<std::uint32_t>(components_.size() - 1)};
  }

  /// A separately-placed coupler: no fields, no clock, `kind ==
  /// ComponentKind::Coupler`. Lives in the SAME `components_` array as every
  /// physics component -- see `ComponentKind`'s comment for why -- so it is
  /// an `add_component()` call with the fields a coupler cannot have left at
  /// their defaults, plus the one it must not be left at.
  ComponentId add_coupler(CouplerDecl d) {
    return add_component(ComponentDecl{.name = std::move(d.name),
                                       .ranks = std::move(d.ranks),
                                       .kind = ComponentKind::Coupler,
                                       .launch = std::move(d.launch)});
  }

  FieldId add_field(ComponentId owner, FieldSpec f) {
    FUSE_ASSERT(owner.valid() && owner.value() < components_.size(),
                "component id {} is out of range ({} components)", owner.value(), components_.size());
    fields_.push_back(std::move(f));
    const FieldId id{static_cast<std::uint32_t>(fields_.size() - 1)};
    field_owner_.push_back(owner);
    components_.at(owner.value()).fields.push_back(id);
    return id;
  }

  ChannelId add_channel(std::string name, FieldId source, FieldId target) {
    channels_.push_back(Channel{.name = std::move(name),
                                .source = source,
                                .target = target});
    return ChannelId{static_cast<std::uint32_t>(channels_.size() - 1)};
  }

  IterationRegionIdx add_iteration_region(IterationRegionDecl d) {
    iteration_regions_.push_back(std::move(d));
    return IterationRegionIdx{
        static_cast<std::uint32_t>(iteration_regions_.size() - 1)};
  }

  /// Spec §5.3: `g.iterate({fsi_channels}).until(...).max_iterations(30)`.
  ///
  /// The region is named `"iterate<k>"` unless `.name()` overrides it; the
  /// name appears in every diagnostic the region can raise, so it is part of
  /// the descriptor and the YAML.
  [[nodiscard]] IterationBuilder iterate(std::vector<ChannelId> channels) {
    IterationRegionDecl d;
    d.name = "iterate" + std::to_string(iteration_regions_.size());
    d.channels = std::move(channels);
    return IterationBuilder{*this, add_iteration_region(std::move(d))};
  }

  [[nodiscard]] ChannelBuilder channel(ChannelId id) {
    FUSE_ASSERT(id.valid() && id.value() < channels_.size(),
                "channel id {} is out of range ({} channels)", id.value(), channels_.size());
    return ChannelBuilder{*this, id};
  }

  [[nodiscard]] const MeshDecl& mesh(MeshId id) const {
    return meshes_.at(id.value());
  }
  [[nodiscard]] const Region& region(RegionId id) const {
    return regions_.at(id.value());
  }
  [[nodiscard]] const ComponentDecl& component(ComponentId id) const {
    return components_.at(id.value());
  }
  [[nodiscard]] const FieldSpec& field(FieldId id) const {
    return fields_.at(id.value());
  }
  /// Declare how several channels writing `target` are combined. Without
  /// this, a field with more than one writer is refused.
  CouplingGraph& merge(FieldId target, MergeOp op) {
    const std::uint32_t key = target.value();
    for (auto& entry : merges_) {
      if (entry.first == key) { entry.second.op = op; return *this; }
    }
    merges_.push_back({key, MergeSpec{op}});
    std::sort(merges_.begin(), merges_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return *this;
  }

  /// Null when the field has no declared merge.
  [[nodiscard]] const MergeSpec* merge_for(FieldId target) const {
    for (const auto& entry : merges_) {
      if (entry.first == target.value()) return &entry.second;
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t merge_count() const noexcept { return merges_.size(); }

  /// Sorted by field id, so serialization and diagnostics are reproducible.
  [[nodiscard]] const std::pair<std::uint32_t, MergeSpec>& merge_at(std::size_t i) const {
    return merges_.at(i);
  }

  [[nodiscard]] const Channel& channel_at(ChannelId id) const {
    return channels_.at(id.value());
  }
  [[nodiscard]] const IterationRegionDecl& iteration_region(
      IterationRegionIdx id) const {
    return iteration_regions_.at(id.value());
  }

  [[nodiscard]] std::size_t mesh_count() const noexcept { return meshes_.size(); }
  [[nodiscard]] std::size_t region_count() const noexcept { return regions_.size(); }
  [[nodiscard]] std::size_t component_count() const noexcept {
    return components_.size();
  }
  [[nodiscard]] std::size_t field_count() const noexcept { return fields_.size(); }
  [[nodiscard]] std::size_t channel_count() const noexcept {
    return channels_.size();
  }
  [[nodiscard]] std::size_t iteration_region_count() const noexcept {
    return iteration_regions_.size();
  }

  [[nodiscard]] ComponentId owner_of(FieldId f) const {
    return field_owner_.at(f.value());
  }

  [[nodiscard]] std::optional<ComponentId> find_component(
      std::string_view name) const {
    for (std::size_t i = 0; i < components_.size(); ++i) {
      if (components_[i].name == name) {
        return ComponentId{static_cast<std::uint32_t>(i)};
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<FieldId> find_field(ComponentId owner,
                                                  std::string_view name) const {
    for (const FieldId f : components_.at(owner.value()).fields) {
      if (fields_.at(f.value()).name == name) return f;
    }
    return std::nullopt;
  }

 private:
  // Bounds-checked mutable access for ChannelBuilder. Not exposed publicly:
  // external callers mutate only through channel()'s builder, which is the
  // one path required to survive reallocation.
  [[nodiscard]] Channel& channel_mut(ChannelId id) {
    return channels_.at(id.value());
  }
  [[nodiscard]] IterationRegionDecl& iteration_region_mut(
      IterationRegionIdx id) {
    return iteration_regions_.at(id.value());
  }

  std::vector<MeshDecl>            meshes_;
  std::vector<Region>              regions_;
  std::vector<ComponentDecl>       components_;
  std::vector<FieldSpec>           fields_;
  std::vector<ComponentId>         field_owner_;  // parallel to fields_
  std::vector<Channel>             channels_;
  std::vector<IterationRegionDecl> iteration_regions_;
  std::vector<std::pair<std::uint32_t, MergeSpec>> merges_;  // sorted by field id
};

/// A channel's lag AS COMPILED, not as authored: the single source of
/// truth for whether a field transfer is loop-carried, read by BOTH
/// `lower()` (src/ir/rules_channel.cpp step 4, which applies exactly this
/// shift) and `schedule()`'s `check_window_causality()`
/// (src/schedule/scheduler.cpp) -- one function, so the two cannot decide
/// this differently for the same channel. Model layer (not schedule),
/// since `ir` sits below `schedule` (docs/dev/layering.md) and must reach
/// this from `lower()` itself.
///
/// Outside any window-scoped iteration region, or inside one whose
/// `CouplingWindowScope::iteration` is `WaveformIteration::GaussSeidel`,
/// this is exactly `Channel::lag` -- today's behaviour, unchanged.
///
/// Inside a `WaveformIteration::Jacobi` window, every one of the region's
/// own channels gets the loop-carried shift `.lag(1)` gives, WHATEVER
/// `Channel::lag` the user wrote: Jacobi reads every partner's PREVIOUS
/// iterate for the whole sweep (`WindowCoupler::sweep_window()`'s own
/// scheme), so the group needs no `.lag()` at all to be schedulable, and
/// what the user did write is not consulted.
namespace detail {
// Shared by `effective_channel_lag()` and
// `channel_time_mismatch_is_authorized()` below: whether `ch` is a member
// of a `WaveformIteration::Jacobi` window-scoped region's own `channels`.
// One membership test, so the two functions cannot disagree about which
// channels Jacobi's auto-lag (and the mismatch it authorizes) covers.
[[nodiscard]] inline bool is_jacobi_window_channel(const CouplingGraph& g,
                                                   ChannelId ch) {
  if (!ch.valid() || ch.value() >= g.channel_count()) return false;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    if (decl.window_scope->iteration != WaveformIteration::Jacobi) continue;
    for (const ChannelId c : decl.channels) {
      if (c.value() == ch.value()) return true;
    }
  }
  return false;
}
}  // namespace detail

[[nodiscard]] inline int effective_channel_lag(const CouplingGraph& g,
                                               ChannelId ch) {
  if (!ch.valid() || ch.value() >= g.channel_count()) return 0;
  if (detail::is_jacobi_window_channel(g, ch)) return 1;
  return g.channel_at(ch).lag;
}

/// Whether `lower()`'s own type_check() (src/ir/lowering.cpp step 4) may
/// admit the time-level shift `effective_channel_lag()` puts on `ch`'s
/// delivered value, without the user's own `Channel::
/// time_level_mismatch_reason`. True when the user declared one
/// themselves via `.allow_time_level_mismatch(...)` (today's rule,
/// unchanged) -- OR when the shift comes from a `WaveformIteration::
/// Jacobi` window the COMPILER itself chose to apply, over the user's own
/// (possibly zero) `Channel::lag`: the same reasoning that makes
/// `effective_channel_lag()` override `Channel::lag` for such a channel
/// applies here too, and using the SAME membership test
/// (`detail::is_jacobi_window_channel`) keeps the two from disagreeing
/// about which channels this covers.
[[nodiscard]] inline bool channel_time_mismatch_is_authorized(
    const CouplingGraph& g, ChannelId ch) {
  if (!ch.valid() || ch.value() >= g.channel_count()) return false;
  if (!g.channel_at(ch).time_level_mismatch_reason.empty()) return true;
  return detail::is_jacobi_window_channel(g, ch);
}

/// The window's length: the named component's own interval, or the
/// explicit one. `E-SCHED-WINDOW-AMBIGUOUS` if both or neither are set;
/// `E-SCHED-WINDOW-COMPONENT` if `window_component` does not name a
/// component in `g`; `E-SCHED-WINDOW-INTERVAL` if the resulting length is
/// not positive.
///
/// Model layer (not schedule), for the SAME reason `window_sample_
/// capacity()` below is: `ir` sits below `schedule` (docs/dev/
/// layering.md) and `window_lowering.cpp` needs this exact resolution
/// (the window length in hand) to size a member's own compiled field,
/// the same value `validate_coupling_window()` (fuse/schedule/window.hpp)
/// resolves for its own checks -- one function, so lowering and schedule
/// cannot resolve two different window lengths for the same scope.
[[nodiscard]] inline Result<Rational> window_length(
    const CouplingGraph& g, const CouplingWindowScope& scope) {
  const bool has_component = scope.window_component.has_value();
  const bool has_explicit = scope.explicit_interval.has_value();
  if (has_component == has_explicit) {
    return Failure{"E-SCHED-WINDOW-AMBIGUOUS",
                 "CouplingWindowScope must set exactly one of "
                 "window_component / explicit_interval"};
  }
  if (has_explicit) {
    const Rational len = *scope.explicit_interval;
    if (!(len > Rational{0})) {
      return Failure{"E-SCHED-WINDOW-INTERVAL",
                   "CouplingWindowScope::explicit_interval (" +
                       len.to_string() + ") must be positive"};
    }
    return len;
  }
  const std::optional<ComponentId> found =
      g.find_component(*scope.window_component);
  if (!found.has_value()) {
    return Failure{"E-SCHED-WINDOW-COMPONENT",
                 "CouplingWindowScope::window_component '" +
                     *scope.window_component +
                     "' does not name a component in the graph"};
  }
  const Rational len = g.component(*found).interval;
  if (!(len > Rational{0})) {
    return Failure{"E-SCHED-WINDOW-INTERVAL",
                 "window-defining component '" + *scope.window_component +
                     "' has a non-positive interval (" + len.to_string() +
                     ")"};
  }
  return len;
}

/// The exact number of waveform samples one member's own outgoing
/// window-scoped field must carry to cover one coupling window: at most
/// `ceil(window_length / interval)` substep-END samples, plus the
/// window-START sample, plus one more for a `DtPolicy::Fixed` member's
/// possible straddling overshoot sample (`WindowSubstep::straddle`,
/// `fuse::detail::plan_substeps()`, `fuse/runtime/window_coupler.hpp`).
///
/// EXACT Rational arithmetic, never `std::ceil` on a `double`: this number
/// now fixes a COMPILED field width (`window_source_type()`,
/// `src/ir/window_lowering.cpp`) as well as `validate_coupling_window()`'s
/// own `E-SCHED-WINDOW-CAPACITY` refusal (`src/schedule/window.cpp`) --
/// the SAME function, called from both, so the two cannot disagree about
/// how wide a window-scoped field is. Model layer (not schedule or ir),
/// since `ir` sits below `schedule` (docs/dev/layering.md) and
/// `window_lowering.cpp` must reach this from `lower()` itself, the same
/// reason `effective_channel_lag()` lives here.
///
/// `interval` not positive: returns 1 (a harmless minimum). The caller's
/// own gate (`E-SCHED-WINDOW-SHORTEN`/`E-SCHED-WINDOW-INTERVAL`) is what
/// actually refuses that case; this function never fails, matching
/// `effective_channel_lag()`'s own "never fails, refusal is someone
/// else's job" shape.
///
/// BM-19 (review 2026-10-11): the division used to be the throwing `/`
/// operator, which can `FUSE_REFUSE` on an unrepresentable ratio --
/// escaping as a thrown `Refusal` the signature never promised, and
/// `abort_no_exceptions()` in a `FUSE_ENABLE_EXCEPTIONS=OFF` build, where
/// this "never fails" function would abort instead. `try_div` makes the
/// promise true: an unrepresentable ratio is treated the same as the
/// already-documented non-positive-`interval` case above, a harmless
/// minimum the caller's own gate is responsible for refusing instead.
[[nodiscard]] inline std::size_t window_sample_capacity(
    Rational window_length, Rational interval, DtPolicy policy) {
  if (!(interval > Rational{0})) return 1;
  // Review 2026-10-11 IS-6: `window_length / interval` (Rational::operator/)
  // FUSE_REFUSEs with E-BASE-RATIONAL-OVERFLOW on an unrepresentable ratio
  // (~2^63-magnitude window/interval values) -- an exception this function's
  // own contract ("never fails, refusal is someone else's job", matching
  // effective_channel_lag()'s shape) does not allow, and which used to
  // escape schedule() uncaught instead of surfacing as an E-SCHED-*
  // diagnostic. `try_div` cannot throw; on overflow, report the capacity as
  // unrepresentably large rather than the ratio itself -- the caller's own
  // `needed > waveform_capacity` gate (E-SCHED-WINDOW-CAPACITY,
  // src/schedule/window.cpp) already refuses anything this large, so no
  // caller ever observes the sentinel as a usable count.
  const std::optional<Rational> ratio = Rational::try_div(window_length, interval);
  if (!ratio.has_value()) return std::numeric_limits<std::size_t>::max();
  std::int64_t substeps = ratio->num() / ratio->den();
  if (ratio->num() % ratio->den() != 0) ++substeps;
  if (substeps < 0) substeps = 0;
  std::size_t needed = static_cast<std::size_t>(substeps) + 1;
  if (policy == DtPolicy::Fixed) needed += 1;
  return needed;
}

/// The exact sample TIMES (not just how many) a window-scoped member's own
/// committed trajectory carries within window `window_index` (0-based) of
/// length `window_length`, from the graph alone: this member's own
/// `interval`, `policy`, and the absolute time `member_start` its native
/// grid is anchored at (its own `t=0`, i.e. the initial condition's own
/// time -- always `Rational{0}` for every caller today, kept as a
/// parameter because a component free to start its clock elsewhere is
/// not otherwise excluded by anything in `fuse::Component`'s own
/// contract). No runtime state -- a straddling member's own carried-
/// forward clock -- is ever consulted, which is what lets a CONSUMER
/// compute the identical list a PROVIDER's own grid produces, without
/// reading the provider's `WindowMemberState` (`fuse::detail::
/// WindowMemberState`, `fuse/runtime/window_coupler.hpp`): `DtPolicy::
/// Fixed` never shortens a step to land on a window boundary (its own
/// docstring), so its Kth substep always ends at EXACTLY `member_start +
/// K * interval`, in every window that value falls in, whether or not an
/// earlier window's own straddle carried the member's real clock past
/// that window's own end -- the grid itself never moves, only which
/// window each of its points falls into does.
///
/// Returned, always: `T0 = member_start + window_index * window_length`,
/// then every native grid point (`DtPolicy::Fixed`) or restarted-at-T0
/// grid point (`DtPolicy::Adjustable`, `fuse::detail::plan_substeps()`'s
/// own contract) strictly inside `(T0, T1)`, then `T1 = T0 +
/// window_length` -- in ascending order, with no duplicates (`T0`/`T1`
/// are always present even when they coincide with a native grid point,
/// because that is where the window's own field transfer needs a
/// sample regardless of whether the member's own physics needed one
/// there too).
///
/// CALLER'S OWN RESPONSIBILITY -- reading this list against a member's
/// own compiled sample buffer. A member whose clock has never straddled
/// a window boundary (every window up to and including this one landed
/// exactly on a native grid point at its own T0) has one buffer entry
/// per entry of this list, in the same order: `window_sample_capacity()`
/// was sized for exactly that correspondence, and it is what today's
/// only caller (`Runner::Impl::advance_window()`, before this track's
/// own straddle replay lands) relies on. A member whose clock HAS
/// carried a straddle into this window (this window's own `T0` is not
/// itself one of the member's own native grid points -- checkable from
/// the graph alone, `(T0 - member_start)` not an exact multiple of
/// `interval`, with no need to ask the member) breaks that
/// correspondence: this list's own leading `T0` entry has no buffer
/// slot of its own (the member's real clock skipped past it, straight
/// to this list's SECOND entry, which is where its window-start
/// checkpoint value actually sits), and every remaining entry shifts
/// down by one buffer slot. `T0`'s own value is never lost -- it is
/// exactly what the consumer's own `WindowMemberState::history`
/// recorded when this member's straddle resolved at the end of the
/// PREVIOUS window (`Runner::Impl::advance_window()`'s own history-
/// update comment) -- but it must be read from there, never
/// reconstructed from this member's own compiled buffer, whichever
/// window instance is being read.
/// BM-19 (review 2026-10-11): every Rational operation below used to use
/// the throwing spellings (`+`, `-`, `*`, `/`), any of which can
/// `FUSE_REFUSE` on an unrepresentable intermediate -- an uncaught
/// `Refusal` escaping a function with no `Result` in its signature to
/// report that through, or an abort in a `FUSE_ENABLE_EXCEPTIONS=OFF`
/// build. `try_*` is used throughout instead; the four lambdas below set
/// `ok` on the first unrepresentable step and every later computation and
/// loop is gated on it, so the function always returns a well-formed (if,
/// on that one unrepresentable path, truncated after `T0`) list ending in
/// `T1` rather than throwing. No legitimate window -- every caller passes
/// a `window_length`/`interval` that already fits comfortably once
/// reduced -- can reach this path; it exists for the same reason
/// `window_sample_capacity()` just above now also uses `try_div`: a
/// corrupted or adversarially constructed graph must not be able to
/// crash this function.
[[nodiscard]] inline std::vector<Rational> window_sample_times(
    Rational interval, DtPolicy policy, Rational member_start,
    Rational window_length, std::int64_t window_index) {
  bool ok = true;
  const auto add = [&](Rational a, Rational b) {
    const auto r = Rational::try_add(a, b);
    if (!r) { ok = false; return a; }
    return *r;
  };
  const auto sub = [&](Rational a, Rational b) {
    const auto r = Rational::try_sub(a, b);
    if (!r) { ok = false; return a; }
    return *r;
  };
  const auto mul = [&](Rational a, Rational b) {
    const auto r = Rational::try_mul(a, b);
    if (!r) { ok = false; return a; }
    return *r;
  };
  const auto div = [&](Rational a, Rational b) {
    const auto r = Rational::try_div(a, b);
    if (!r) { ok = false; return a; }
    return *r;
  };

  const Rational T0 = add(member_start, mul(Rational(window_index), window_length));
  const Rational T1 = ok ? add(T0, window_length) : T0;
  std::vector<Rational> times;
  times.push_back(T0);
  if (ok && interval > Rational{0}) {
    if (policy == DtPolicy::Fixed) {
      // The member's own GLOBAL grid, anchored at `member_start`,
      // independent of `window_index`: the first grid point strictly
      // after `T0`, found by exact-Rational floor division (never a
      // `double`) rather than by stepping from `member_start` one
      // `interval` at a time -- the same number either way, but O(1)
      // instead of O(window_index).
      const Rational offset = div(sub(T0, member_start), interval);
      if (ok) {
        std::int64_t k = offset.num() / offset.den();  // floor: offset >= 0
                                                       // always (window_index
                                                       // >= 0), so truncating
                                                       // integer division IS
                                                       // floor division here.
        Rational t = add(member_start, mul(Rational(k), interval));
        while (ok && !(t > T0)) t = add(t, interval);
        while (ok && t < T1) {
          times.push_back(t);
          t = add(t, interval);
        }
      }
    } else {
      // Adjustable: the grid restarts at T0 every window
      // (`fuse::detail::plan_substeps()`'s own contract).
      Rational t = add(T0, interval);
      while (ok && t < T1) {
        times.push_back(t);
        t = add(t, interval);
      }
    }
  }
  times.push_back(T1);
  return times;
}

/// Whether `window_sample_times()`'s own leading `T0` entry, for THIS
/// window, has a buffer slot of its own in the member's compiled sample
/// buffer -- `0` if it does (every entry maps 1:1 to a buffer index, the
/// same index), `1` if a carry-in shifts every remaining entry down by
/// one buffer slot instead (entry 0 has no slot: read it from the
/// consumer's own history; entry `k >= 1` maps to buffer index `k - 1`).
/// See `window_sample_times()`'s own header comment for the reasoning.
///
/// Exact-Rational, graph alone (`T0`, `interval`, `policy`,
/// `member_start`), no runtime state: `DtPolicy::Adjustable` never
/// carries in (its own grid restarts at `T0` every window, so entry 0
/// -- `T0` itself -- is always that member's real value there); a
/// `DtPolicy::Fixed` member carries in exactly when this window's own
/// `T0` is not itself one of its own native grid points (`member_start +
/// k * interval` for some integer `k`) -- the member's real clock, by
/// the time this window's sweep 0 runs, already sits at the NEXT native
/// grid point past `T0`, carried there by an earlier window's own
/// straddle, so its buffer's own index 0 holds that later value, not the
/// one AT `T0`.
[[nodiscard]] inline int window_sample_shift(Rational interval,
                                             DtPolicy policy,
                                             Rational member_start,
                                             Rational T0) {
  if (policy != DtPolicy::Fixed) return 0;
  if (!(interval > Rational{0})) return 0;
  const Rational offset = (T0 - member_start) / interval;
  return (offset.num() % offset.den() != 0) ? 1 : 0;
}

inline Channel& ChannelBuilder::target() { return graph_->channel_mut(id_); }

inline IterationRegionDecl& IterationBuilder::target() {
  return graph_->iteration_region_mut(id_);
}

inline IterationBuilder& IterationBuilder::until(ConvergenceCriterion c) {
  target().convergence = std::move(c);
  return *this;
}

inline IterationBuilder& IterationBuilder::until(ConvergenceMeasure m) {
  ConvergenceCriterion c;
  c.add(std::move(m));
  return until(std::move(c));
}

inline IterationBuilder& IterationBuilder::max_iterations(int n) {
  target().max_iterations = n;
  return *this;
}

inline IterationBuilder& IterationBuilder::accelerate(AccelConfig cfg) {
  IterationRegionDecl& t = target();
  t.acceleration = cfg;
  if (cfg.kind == AccelKind::Newton) {
    t.requires_residual_capability = true;
    if (t.newton_config.use_block_jacobi_preconditioner) {
      t.requires_linearization_capability = true;
    }
  }
  return *this;
}

inline IterationBuilder& IterationBuilder::initial_relaxation(double omega) {
  target().acceleration.initial_relaxation = omega;
  return *this;
}

inline IterationBuilder& IterationBuilder::name(std::string n) {
  target().name = std::move(n);
  return *this;
}

inline IterationBuilder& IterationBuilder::newton(NewtonConfig cfg) {
  IterationRegionDecl& t = target();
  t.newton_config = cfg;
  t.acceleration.kind = AccelKind::Newton;
  t.requires_residual_capability = true;
  if (cfg.use_block_jacobi_preconditioner) {
    t.requires_linearization_capability = true;
  }
  return *this;
}

inline IterationBuilder& IterationBuilder::requires_residual(bool v) {
  target().requires_residual_capability = v;
  return *this;
}

inline IterationBuilder& IterationBuilder::requires_linearization(bool v) {
  target().requires_linearization_capability = v;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::accumulate(AccumOp op, Window w) {
  Channel& c = target();
  c.accumulate.op = op;
  c.accumulate.window = w;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::remap(RemapFlags flags) {
  Channel& c = target();
  c.remap.flags = c.remap.flags | flags;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::remap_method(std::string method) {
  target().remap.method_override = std::move(method);
  return *this;
}

inline ChannelBuilder& ChannelBuilder::remap_on(ComponentId c) {
  target().remap.remap_on = c;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::conservative() {
  return remap(RemapFlags::Conservative);
}

inline ChannelBuilder& ChannelBuilder::spacetime(TemporalBasis basis,
                                                  int degree, Window w) {
  Channel& c = target();
  c.spacetime.basis = basis;
  c.spacetime.degree = degree;
  c.spacetime.window = w;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::relax(double omega) {
  Channel& c = target();
  c.relax.enabled = true;
  c.relax.omega = omega;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::extrapolate(int order) {
  Channel& c = target();
  c.extrapolate.enabled = true;
  c.extrapolate.order = order;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::lag(int steps) {
  target().lag = steps;
  return *this;
}

inline ChannelBuilder& ChannelBuilder::allow_time_level_mismatch(
    std::string reason) {
  target().time_level_mismatch_reason = std::move(reason);
  return *this;
}

}  // namespace fuse
