#pragma once

#include <algorithm>

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
  /// `AccelConfig::filter` (QR1/QR2/QR3) is a documented no-op until Task
  /// 7.4; setting it changes nothing today.
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
  IterationBuilder& accelerate(AccelConfig cfg);
  /// Sets `acceleration.initial_relaxation`. There is deliberately no second
  /// `initial_relaxation` field on the declaration: two spellings of one
  /// number is a silent disagreement waiting to happen.
  IterationBuilder& initial_relaxation(double omega);
  IterationBuilder& name(std::string n);

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
                                       .kind = ComponentKind::Coupler});
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
  target().acceleration = cfg;
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
