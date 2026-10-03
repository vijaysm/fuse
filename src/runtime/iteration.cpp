#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "runner_impl.hpp"
#include "fuse/num/kernels.hpp"

/// Spec §5.3: executing an iteration region.
///
/// WHAT THE ITERATION IS OVER. Let `y` be the vector FUSE delivers into the
/// participants' `Requires` storage -- the `Sink` outputs, which by spec §4.2
/// *are* the components' own arrays. One sweep evaluates `G(y)`: every
/// checkpointed component is restored, re-advanced over the SAME interval with
/// `y` on its inputs, and the region body transports, remaps, constrains and
/// diagnoses its outputs. The region converges when `||G(y) - y||` satisfies
/// the criterion. `G` is exactly the coupled operator whose fixed point the
/// implicit scheme is defined by, so accelerating on `(y, G(y))` is
/// accelerating on the right map -- not on some coupler-internal intermediate
/// that happens to be handy.
///
/// THE FOUR THINGS THAT ARE SILENT WHEN WRONG, and where each is handled.
///
///  1. CHECKPOINTS MUST BE TAKEN BEFORE THE FIRST ADVANCE, not at the region's
///     enter instruction. By the time the interpreter reaches the region the
///     participants have already advanced once for this period -- the plan
///     emits the chain at the slab end, after the ticks STRICTLY EARLIER than
///     it (a chain instance is `BeforeTicks`, so the ticks at its own stamp
///     have not run). That still leaves every participant advanced when the
///     region is reached, because a slab end is strictly greater than the
///     tick that opens the slab; what makes it airtight is that
///     resolve_regions() refuses a member tick at or after `enter_time`
///     outright. A save_state() at
///     region entry would capture post-advance state; every restore would then
///     leave the component one advance ahead of where the sweep believes it
///     is, and the iteration would converge, monotonically and with a healthy
///     residual history, to the fixed point of a DIFFERENT operator.
///     checkpoint_regions() therefore runs from the top of step_period().
///
///  2. THE RUNTIME VERIFIES THE ROLLBACK RATHER THAN TRUSTING IT. spec §5.3
///     is explicit that FUSE "verifies at runtime" that the component saved
///     what the derived set said it must; that is the claim that makes the
///     derivation trustworthy instead of merely plausible, and it is what
///     preCICE's requiresWritingCheckpoint() does not do.
///     `supports_rollback()` is a promise, checked at setup
///     (E-ITER-NOROLLBACK). Here, after every restore_state(), the component's
///     registered `Provides` storage is compared byte for byte against the
///     bytes it held when save_state() was called (E-ITER-ROLLBACK). A
///     component that returns success from restore_state() without restoring
///     is otherwise indistinguishable from one that works.
///
///  3. NON-CONVERGENCE IS REPORTED. Exhausting max_iterations is a failure,
///     not a quiet acceptance: `RunStats::windows_not_converged` counts it and
///     `RunnerConfig::abort_on_nonconvergence` turns it into E-RT-NOCONVERGE.
///     THE STATE ON EXIT, converged or not, is the same and is well defined:
///     the coupler buffers hold the last computed `G(y)`, the components hold
///     the state they reached when advanced from `y`, and the `Sink`
///     instructions that follow the region in the stream deliver `G(y)`. On
///     convergence `||G(y) - y||` is within tolerance so the two agree; on
///     exhaustion they do not, which is precisely what the counter says.
///
///  4. NOTHING IN THE SWEEP LOOP ALLOCATES. Every buffer here -- the
///     checkpoints, the iterate vectors, the per-block norms, even
///     `IterationStats::residual_history`, whose capacity is reserved -- is
///     sized in resolve_regions(). Measured at 0 allocations over 213 sweeps
///     in 8 periods with `AccelKind::None`. True of the accelerators as well
///     since their scratch was hoisted into the accelerator objects
///     (src/num/accel_*.cpp): an accelerated region measures 0 allocations
///     over the same 8 periods, against 56 -- one per accelerator update --
///     before. TheAcceleratorAllocatesNothingPerUpdate keeps that at zero for
///     Aitken through the runner, and
///     Accelerator.SteadyStateUpdateAllocatesNothing for every kind directly.
///
/// WHAT IS REFUSED RATHER THAN GUESSED. resolve_regions() rejects, with
/// E-RT-REGION, every plan shape whose correct execution it cannot establish:
/// a region body that appears more than once in a period, a residual with no
/// `Sink` to deliver it through, a checkpointed component with a tick after
/// the region begins, and -- the important one -- any instruction OUTSIDE the
/// region that reads a value derived from a checkpointed component's output.
/// That last instruction would carry a pre-iteration value into work the
/// region never re-runs, which is the same class of silent staleness as (1).
namespace fuse {
namespace {

/// The first error in `d`, keeping its code so callers can test on it, with
/// the whole diagnostic set as the message.
[[nodiscard]] Failure first_error(const Diagnostics& d, const char* fallback) {
  for (const Diagnostic& item : d.items()) {
    if (item.severity == Severity::Error) {
      return Failure{item.code, d.to_string()};
    }
  }
  return Failure{fallback, d.to_string()};
}

}  // namespace

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::resolve_regions(ComponentSet& components,
                                           FieldRegistry& registry) {
  regions.clear();
  region_stats.clear();
  region_explain.clear();

  const std::size_t n_regions = plan.region_count();
  if (n_regions == 0) return Result<void>::success();

  // The three layers must agree on how many regions there are. `lower()` emits
  // one DAG region per declaration in order and fails the lowering otherwise
  // (E-IR-REGION-EMPTY), and `schedule()` copies the DAG's count into the plan,
  // so a mismatch means these three artifacts do not describe one another --
  // and the index that reads a criterion out of the graph would read the wrong
  // one.
  if (dag.region_count() != n_regions ||
      graph.iteration_region_count() != n_regions) {
    return Failure{"E-RT-REGION",
                 "the plan declares " + std::to_string(n_regions) +
                     " iteration region(s), the operator DAG " +
                     std::to_string(dag.region_count()) +
                     " and the coupling graph " +
                     std::to_string(graph.iteration_region_count()) +
                     "; they must be the same regions in the same order for a "
                     "region's convergence criterion to be read from the graph "
                     "declaration it came from"};
  }

  regions.resize(n_regions);
  region_stats.assign(n_regions, IterationStats{});
  region_explain.assign(n_regions, std::string{});

  const std::vector<Instruction>& stream = plan.instructions();

  for (std::size_t ri = 0; ri < n_regions; ++ri) {
    detail::RegionRuntime& R = regions[ri];
    const auto rid = RegionIdx{static_cast<std::uint32_t>(ri)};
    const OperatorDAG::IterationRegion& reg = dag.region_at(rid);
    const IterationRegionDecl& decl = graph.iteration_region(rid);

    R.index = ri;
    R.name = reg.name;
    R.max_iterations = reg.max_iterations;
    R.convergence = decl.convergence;

    // THE GATE (spec §5.3, Task 7.2). This is the only call site of
    // verify_iteration_region(), and it is why IterationRegionDecl carries a
    // ConvergenceCriterion at all: without the field there is nowhere for the
    // criterion to live, so the E-ITER-NOCONVERGE check could never run and a
    // region with no convergence measure would stop after one sweep and report
    // success. E-ITER-MAXITER and E-ITER-NOROLLBACK come from the same call.
    const Diagnostics gate = verify_iteration_region(graph, dag, reg,
                                                     decl.convergence,
                                                     components);
    if (gate.has_errors()) return first_error(gate, "E-ITER-REGION");

    const CheckpointSet cp = derive_checkpoint_set(graph, dag, reg);
    region_explain[ri] = cp.explanation;

    // --- the components the region re-advances, and the state FUSE can see --
    for (const ComponentId c : cp.components) {
      const std::string& cname = graph.component(c).name;
      std::size_t slot = components.size();
      for (std::size_t j = 0; j < components.size(); ++j) {
        if (components.at(j).name() == cname) { slot = j; break; }
      }
      if (slot == components.size()) {
        // verify_rollback_support() already rejects this; belt and braces,
        // because the index below is used unchecked in the sweep loop.
        return Failure{"E-ITER-NOROLLBACK",
                     "iteration region '" + R.name +
                         "' must checkpoint component '" + cname +
                         "', which is not registered"};
      }
      const std::size_t which = R.component_slots.size();
      R.component_slots.push_back(slot);
      for (std::uint32_t a = 0; a < slot_source_owner.size(); ++a) {
        if (slot_source_owner[a] != static_cast<std::int32_t>(c.value())) {
          continue;
        }
        R.verify_arcs.push_back(a);
        R.verify_save.emplace_back(slots[a].total(), 0.0);
        R.verify_component.push_back(which);
      }
    }

    // --- every contiguous pass over the body --------------------------------
    // Moved ahead of the participant computation below (it used to follow):
    // that computation now needs [R.enter, R.exit] to see every Transport
    // and Epoch INSIDE the region body, not only the checkpointed
    // components' own declared ranks.
    //
    // Time windows track: an ORDINARY region's own clock ticks once per
    // schedule period by construction, so mark_regions() (src/schedule/
    // scheduler.cpp) stamps exactly one (enter, exit) run for it and the
    // check below is unchanged from before this track. A WINDOW-scoped
    // region's own window may tick several times per period (a 7:13 ratio
    // ticks `period / window_length` times), so its body legitimately
    // appears as that many SEPARATE runs -- each is its own pass, with its
    // own checkpoint (run_region()'s own top-of-call save_state() for a
    // window region; see RegionRuntime::window_scoped's comment), not
    // several independent fixed-point problems sharing one.
    R.window_scoped = decl.window_scope.has_value();
    std::size_t pending_enter = stream.size();  // stream.size() == "none open"
    for (std::size_t i = 0; i < stream.size(); ++i) {
      if (stream[i].region_enter == static_cast<std::int32_t>(ri)) {
        if (pending_enter != stream.size()) {
          return Failure{"E-RT-REGION",
                       "iteration region '" + R.name + "' has two enter "
                       "boundaries with no exit between them, which "
                       "mark_regions() should never produce"};
        }
        pending_enter = i;
      }
      if (stream[i].region_exit == static_cast<std::int32_t>(ri)) {
        if (pending_enter == stream.size() || i < pending_enter) {
          return Failure{"E-RT-REGION",
                       "iteration region '" + R.name + "' has an exit "
                       "boundary with no matching enter before it"};
        }
        R.passes.emplace_back(pending_enter, i);
        pending_enter = stream.size();
      }
    }
    if (pending_enter != stream.size()) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name +
                       "' has an enter boundary with no matching exit"};
    }
    if (!R.window_scoped && R.passes.size() != 1) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name + "' appears as " +
                       std::to_string(R.passes.size()) +
                       " separate passes over the body in one period; "
                       "exactly one pass is required for a region with no "
                       "coupling window. A body that appears more than once "
                       "per period is several independent fixed-point "
                       "problems sharing one checkpoint, which this runner "
                       "will not guess at: split it into one region per "
                       "clock, or opt the region into a coupling window"};
    }
    if (R.passes.empty()) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name +
                       "' has no pass over its body in this period at all"};
    }
    R.enter = R.passes.front().first;
    R.exit = R.passes.front().second;

    // Time windows track (C3): resolve every window member's own static
    // config -- window_member_config's own comment (runner_impl.hpp) --
    // from the graph alone. Every pass of a window-scoped region repeats
    // the identical instruction shape (only `dst_step`/`time` differ), so
    // scanning the FIRST pass once, here, at setup, is enough; every
    // later pass's own WindowAdvance instructions carry the SAME
    // `origin_node`, per `finish_window_regions()`'s "at most one such
    // node per member per region" (src/ir/window_lowering.cpp).
    //
    // Keyed by a window member's own EXPORT arc id (`ins.out.value()`).
    // A channel's residual block's own `value_arc` (resolve_regions()'s
    // own residual-blocks loop, below) is NOT necessarily this id
    // directly: the lookup there walks back through any interposed
    // single-input, size-preserving node first (the Brusselator's own
    // graph puts exactly one `OpKind::Interpolate` resample between the
    // two, measured 2026-10-03 -- a shape-PRESERVING resample for an
    // identity transfer with no real fuse::Remap, brusselator_graph.
    // hpp's own "WHY BOTH TRANSFER FIELDS LIVE ON u_region" comment, not
    // a no-op alias, so the two arc ids genuinely differ).
    //
    // Value: {capacity, valid_substeps} -- `capacity` is this member's
    // own `WindowMemberConfig::capacity` (the per-element stride the
    // raw arc is laid out with: sample 0 is the window-START value,
    // samples 1..capacity-1 are the per-substep END values written by
    // `advance_window()`); `valid_substeps` is the POLICY-INDEPENDENT
    // count of real substep-end samples -- `window_sample_capacity
    // (window_length, interval, DtPolicy::Adjustable) - 1`, i.e.
    // `ceil(window_length / interval)` -- never `capacity - 1` directly,
    // since a `DtPolicy::Fixed` member's own `capacity` reserves one
    // MORE slot than it actually fills on a non-straddling window
    // instance (`window_sample_capacity()`'s own doc comment,
    // fuse/model/coupling_graph.hpp).
    //
    // The per-block residual computation below (this function's own
    // sweep-loop counterpart in `run_region()`) consults this to strip
    // the window-start sample before norming, matching the driver's own
    // named-measure residual (`WindowCoupler::sweep_window()`'s own
    // comment: "ONLY that channel's CONSUMER's own flat slice", sized
    // to EXACTLY `steps[consumer].size()`, never +1). MEASURED,
    // 2026-10-03 (this delegation's own report), against the
    // Brusselator's own window 20, sweep k=1 -- pairing each side's own
    // measure by its NUMERIC VALUE, not by channel label, since the two
    // sides name the same pair of quantities in SWAPPED order (the
    // Runner's own block for channel `u2v` holds U's own export, the
    // SAME quantity the driver's own "v_out"-named measure reads via
    // channel `v2u`'s consumer being U -- `WindowCoupler::sweep_window
    // ()`'s own "each measure is tested against the residual of the
    // field transfer it names, as the Runner does" comment describes
    // the FORMULA, not this index correspondence): before this fix, the
    // Runner's own u2v block (element length 256, capacity 11) summed
    // 2816 entries against the driver's own paired measure's 2560
    // (element length 256, 10 substeps) -- 256 too many, i.e. exactly
    // one extra entry per element, the window-start sample; the two
    // sides' own `ref` hexfloats disagreed from the third significant
    // hex digit on (`0x1.6be7e489b2a8cp+5` vs `0x1.5b510b3c3ad89p+5`).
    // After this fix, stripping that one entry, the two sides' own
    // `res`/`ref` are bit-for-bit identical at every sweep k>=1 of
    // window 20, and `BrusselatorRunner.{Jacobi,GaussSeidel}
    // MatchesDriver` (test/verification/brusselator_runner_test.cpp)
    // pass EXPECT_EQ over the full 150-window span with the real
    // stopping rule, both waveform iterations.
    std::unordered_map<std::uint32_t, std::pair<std::size_t, std::size_t>>
        window_export_meta;
    if (R.window_scoped) {
      const CouplingWindowScope& scope = *decl.window_scope;
      const Result<Rational> wlen = window_length(graph, scope);
      if (!wlen.ok()) return wlen.error();
      for (std::size_t i = R.enter; i <= R.exit; ++i) {
        const Instruction& ins = stream[i];
        if (ins.kind != OpKind::WindowAdvance) continue;
        const std::uint32_t node = ins.origin_node.value();
        if (window_member_config.count(node)) continue;

        const int rank_set = dag.arc(ins.out).type.rank_set;
        if (rank_set < 0) {
          return Failure{"E-RT-WINDOW-UNPLACED",
                       "iteration region '" + R.name + "': a WindowAdvance "
                       "instruction's own output arc has no owning "
                       "component (rank_set < 0)"};
        }
        const ComponentId member{static_cast<std::uint32_t>(rank_set)};
        const std::string& mname = graph.component(member).name;
        std::size_t slot = components.size();
        for (std::size_t j = 0; j < components.size(); ++j) {
          if (components.at(j).name() == mname) { slot = j; break; }
        }
        if (slot == components.size()) {
          return Failure{"E-RT-WINDOW-UNPLACED",
                       "iteration region '" + R.name + "': window member '" +
                           mname + "' is not a registered component"};
        }

        WindowMemberConfig cfg;
        cfg.component_slot = slot;
        cfg.interval = graph.component(member).interval;
        cfg.policy = scope.policy_for(mname).value_or(DtPolicy::Adjustable);
        cfg.order = scope.order;
        cfg.window_length = wlen.value();
        cfg.capacity =
            window_sample_capacity(cfg.window_length, cfg.interval, cfg.policy);
        cfg.window_import = ins.attrs.window_import;
        cfg.provider_component_slot = components.size();  // sentinel: none
        cfg.region_index = ri;

        // The channel THIS member sources (its own export) and the one
        // it targets (its own import), if any -- mirrors member_exported_
        // field()/member_imported_field() (src/ir/window_lowering.cpp),
        // duplicated rather than called for the same reason that file
        // duplicates window_region_members(): those two are file-local to
        // a different translation unit.
        FieldId exported_field, imported_field, provider_exported_field;
        for (const ChannelId cid : decl.channels) {
          const Channel& ch = graph.channel_at(cid);
          if (!exported_field.valid() &&
              graph.owner_of(ch.source).value() == member.value()) {
            exported_field = ch.source;
          }
          if (!imported_field.valid() &&
              graph.owner_of(ch.target).value() == member.value()) {
            imported_field = ch.target;
            provider_exported_field = ch.source;
            cfg.import_lagged = effective_channel_lag(graph, cid) != 0;
            const ComponentId provider = graph.owner_of(ch.source);
            const std::string& pname = graph.component(provider).name;
            for (std::size_t j = 0; j < components.size(); ++j) {
              if (components.at(j).name() == pname) {
                cfg.provider_component_slot = j;
                break;
              }
            }
            cfg.provider_interval = graph.component(provider).interval;
            cfg.provider_policy =
                scope.policy_for(graph.component(provider).name)
                    .value_or(DtPolicy::Adjustable);
            cfg.provider_capacity = window_sample_capacity(
                cfg.window_length, cfg.provider_interval, cfg.provider_policy);

            // Time windows track: resolve_pending_window_straddles() (this
            // function's own sibling, below) publishes a straddling
            // provider's real post-replay value straight into THIS
            // consumer's own `window_import` arc, bypassing Transport/
            // Remap entirely -- exact ONLY when the channel moves no
            // data at all: same ranks (nothing to send), same region
            // (nothing to remap). `DtPolicy::Fixed` with an interval
            // that does not divide the window is what can ever leave a
            // straddle PENDING in the first place (plan_substeps()'s own
            // contract; `DtPolicy::Adjustable` never straddles) -- a
            // channel that could never see a straddle has nothing this
            // publish ever touches, so it is not rejected here even if
            // it is itself remapped or cross-rank.
            const Rational win_over_interval =
                cfg.window_length / cfg.provider_interval;
            const bool provider_can_straddle =
                cfg.provider_policy == DtPolicy::Fixed &&
                win_over_interval.num() % win_over_interval.den() != 0;
            const bool same_ranks =
                graph.component(provider).ranks.same_as(
                    graph.component(member).ranks);
            const bool same_region =
                graph.field(ch.source).region == graph.field(ch.target).region;
            if (provider_can_straddle && (!same_ranks || !same_region)) {
              return Failure{
                  "E-RT-WINDOW-STRADDLE-REPUBLISH",
                  "iteration region '" + R.name + "': window member '" +
                      mname + "' imports from '" + pname +
                      "', a DtPolicy::Fixed provider whose own interval "
                      "does not divide the window (it can leave a "
                      "straddle pending), through a channel that is " +
                      std::string(!same_ranks ? "not on the same ranks"
                                               : "not on the same region") +
                      " -- resolve_pending_window_straddles()'s own "
                      "republish of a straddling provider's real value "
                      "writes directly into this consumer's own "
                      "window_import arc, which is only exact for a "
                      "same-rank, same-region (identity) transfer; a real "
                      "Transport or Remap between provider and consumer "
                      "would need to run for this consumer's own copy to "
                      "be correct, which this publish does not do"};
            }
          }
        }

        // This member's own NATIVE (single-sample) Provides/Requires
        // storage -- resolved directly from `registry`, bypassing the arc
        // system entirely (WindowMemberConfig::provided's own comment,
        // runner_impl.hpp).
        if (exported_field.valid()) {
          Result<FieldRef> pref = registry.get(exported_field);
          if (!pref.ok()) return pref.error();
          if (pref.value().layout().components != 1 ||
              pref.value().layout().stride != 1) {
            return Failure{"E-RT-WINDOW-LAYOUT",
                         "iteration region '" + R.name + "': window member '" +
                             mname + "' exports a field with more than one "
                             "component per element or a non-contiguous "
                             "layout, which advance_window() does not yet "
                             "support"};
          }
          cfg.provided = pref.value().data();
          cfg.length = pref.value().size();
        }
        if (imported_field.valid()) {
          Result<FieldRef> rref = registry.get(imported_field);
          if (!rref.ok()) return rref.error();
          if (rref.value().layout().components != 1 ||
              rref.value().layout().stride != 1) {
            return Failure{"E-RT-WINDOW-LAYOUT",
                         "iteration region '" + R.name + "': window member '" +
                             mname + "' imports a field with more than one "
                             "component per element or a non-contiguous "
                             "layout, which advance_window() does not yet "
                             "support"};
          }
          cfg.required = rref.value().data();
          if (cfg.length == 0) cfg.length = rref.value().size();
        }

        if (cfg.import_lagged && cfg.window_import.valid()) {
          R.window_lagged_import_arcs.push_back(cfg.window_import);
        }

        // Cold start (the user's ruling): before window 0 ever runs, a
        // consumer's import must read the provider's own registered
        // INITIAL value, held constant -- exactly what an ORDINARY
        // (non-window) lag(1) channel already gets for free, because its
        // Source arc is an ALIAS onto the component's own native storage
        // (expand.cpp's own "Source moves no data" comment): the live
        // value there IS the registered initial condition from the
        // moment register_fields() ran, with no separate buffer to
        // prime. Neither this member's own K-wide EXPORT arc (`ins.out`)
        // nor a consumer's K-wide IMPORT arc (`cfg.window_import`) is
        // such an alias (the C3 buffer-ownership fix, the `/window`-
        // labelled Sink, plan_buffers.cpp) -- both are genuinely
        // separate, coupler-owned buffers -- so BOTH are primed
        // explicitly here, once, at setup, to the SAME constant (this
        // member's own `cfg.provided` value for its own export; the
        // provider's own registered value, fetched via `registry.get()`
        // exactly as `cfg.provided`/`cfg.required` already are, for a
        // consumer's own import). Priming only the export and relying on
        // the field transfer (Transport, Remap, Epoch, Sink) to carry it
        // downstream is not enough for a LAGGED import specifically:
        // `run_region()`'s own shadow refresh (which a lagged read goes
        // through) snapshots the import arc's live content BEFORE this
        // sweep's own body -- including that very chain -- ever runs, so
        // on window 0's own sweep 0 the chain has not had a chance to
        // deliver anything yet. Priming the import arc directly closes
        // that gap; priming the export arc too keeps the chain's own
        // later, real deliveries (window 0 onward) consistent with what
        // was seeded, rather than overwriting it with un-primed zeros
        // the one time the chain DOES run inside sweep 0's own body (an
        // unlagged import, or a lagged one's later sweeps).
        if (cfg.provided != nullptr && ins.out.valid() &&
            ins.out.value() < slots.size()) {
          detail::Slot& export_slot = slots[ins.out.value()];
          if (export_slot.data != nullptr &&
              export_slot.total() == cfg.length * cfg.capacity) {
            for (std::size_t e = 0; e < cfg.length; ++e) {
              for (std::size_t s = 0; s < cfg.capacity; ++s) {
                export_slot.data[e * cfg.capacity + s] = cfg.provided[e];
              }
            }
          }
        }
        if (cfg.required != nullptr && cfg.window_import.valid() &&
            provider_exported_field.valid() &&
            cfg.window_import.value() < slots.size()) {
          const Result<FieldRef> provided_ref =
              registry.get(provider_exported_field);
          if (!provided_ref.ok()) return provided_ref.error();
          const double* provider_v0 = provided_ref.value().data();
          // The PROVIDER's own exported-field length -- NOT cfg.length,
          // which is THIS member's own exported (or, absent an export,
          // imported) length, read from a DIFFERENT field a few lines
          // above. The two agree whenever this is a same-resolution
          // transfer (the common case, and every case the existing
          // suite covered before Conformance Unit 2's own cross-
          // resolution U/V box graph), but nothing enforces that they
          // must: a real heap-buffer-overflow READ of `provider_v0[e]`
          // for e in [provider_length, cfg.length) resulted whenever
          // cfg.length (this member's own, larger) was used as the loop
          // bound to index `provider_v0` (the provider's own, smaller,
          // buffer) -- found by ASan on a two-member window with
          // mismatched field lengths (reported to the lead as Finding
          // 3(a); fixed here as Conformance Unit 2's own Defect 2).
          const std::size_t provider_length = provided_ref.value().size();
          // Bounds BOTH ends: `provider_v0[e]` is valid only for
          // e < provider_length; `import_slot.data`/`seed_st.history[e]`
          // below are sized to cfg.length (the check on import_slot.
          // total() just below still gates on cfg.length -- that is
          // THIS buffer's own real size, unchanged). When the two
          // differ, priming is necessarily partial: entries
          // [prime_count, cfg.length) are left at whatever import_slot/
          // history already held (zero-initialized storage, or an
          // empty history list) -- a correct REMAP between the two
          // resolutions is a real Transport/Remap, which this one-time
          // cold-start shortcut has never attempted (the comment above
          // this block already says as much for the lagged, same-rank
          // case; cross-resolution is the same limitation, not a new
          // one introduced by this fix).
          const std::size_t prime_count = std::min(cfg.length, provider_length);
          detail::Slot& import_slot = slots[cfg.window_import.value()];
          if (import_slot.data != nullptr &&
              import_slot.total() == cfg.length * cfg.provider_capacity) {
            for (std::size_t e = 0; e < prime_count; ++e) {
              for (std::size_t s = 0; s < cfg.provider_capacity; ++s) {
                import_slot.data[e * cfg.provider_capacity + s] =
                    provider_v0[e];
              }
            }
          }

          // Time windows track: this member's own straddle-tail
          // extrapolation (advance_window()'s own comment, the split-
          // advance callbacks) reads `window_member_state[node].history`
          // for points BEFORE the window it is currently splitting --
          // `detail::WindowMemberState::history`'s own struct comment
          // ("seeded with the initial condition, t=0"), mirrored here
          // exactly: one list per element, seeded with the SAME
          // provider value the import arc itself was just primed with,
          // at t=0, so the very first window that ever straddles
          // (however early) has at least one point to extrapolate from
          // instead of `fuse::extrapolate_boundary_value()`'s own
          // `FUSE_ASSERT(!history.empty())` firing. `seed_st.history`
          // itself is still sized to cfg.length (THIS member's own
          // element count, what every other reader of `history` -- e.g.
          // the straddle replay -- indexes by); only the SEEDING loop
          // is bounded by `prime_count`, so an element beyond the
          // provider's own length is left with an EMPTY history list,
          // same residual limitation as the import_slot priming above.
          detail::WindowMemberState& seed_st = window_member_state[node];
          seed_st.history.assign(cfg.length, {});
          for (std::size_t e = 0; e < prime_count; ++e) {
            seed_st.history[e].push_back({Rational{0}, provider_v0[e]});
          }
        }

        window_member_config[node] = cfg;
        R.window_member_nodes.push_back(NodeId{node});

        // This member's own export arc -- window_export_meta's own
        // top comment -- regardless of whether cfg.provided is null
        // (a consumer-only member exports nothing, so ins.out for its
        // own WindowAdvance instruction is never a channel's
        // value_arc and this entry is simply never looked up).
        if (ins.out.valid()) {
          const std::size_t valid_substeps =
              window_sample_capacity(cfg.window_length, cfg.interval,
                                     DtPolicy::Adjustable) -
              1;
          window_export_meta[ins.out.value()] = {cfg.capacity,
                                                  valid_substeps};
        }
      }

      // The D7 envelope (KNOWN_ISSUES.md; WindowCoupler::create()'s own
      // check, window_coupler.cpp): a member straddles a window end only
      // if it is Fixed and its own interval does not divide the window.
      // The replay (run_region()'s own straddle-replay step,
      // detail::advance_straddle_split()) handles ONE such member per
      // window -- two replaying against each other's own UNRESOLVED
      // trajectories at once is the case neither this Runner nor
      // WindowCoupler supports. Checked here, once per region at setup,
      // exactly as WindowCoupler checks it once at create() -- a
      // graph-level property of policy+interval+window_length, not
      // something that depends on any particular window instance ever
      // running.
      bool have_straddler = false;
      for (const NodeId node : R.window_member_nodes) {
        const WindowMemberConfig& cfg = window_member_config.at(node.value());
        if (cfg.policy != DtPolicy::Fixed) continue;
        if ((cfg.window_length / cfg.interval).is_integer()) continue;
        if (have_straddler) {
          return Failure{
              "E-RT-WINDOW-STRADDLE",
              "iteration region '" + R.name + "': more than one window "
              "member is Fixed with a step that does not divide the window "
              "(" + cfg.window_length.to_string() + "); both can straddle "
              "one window end, and only one straddling member per window is "
              "supported (KNOWN_ISSUES.md D7, mirrors WindowCoupler::"
              "create()'s own refusal). Make all but one of them Adjustable, "
              "or choose a window every Fixed member's own step divides"};
        }
        have_straddler = true;
      }
    }

    // R-10 (REVIEW.md): the convergence decision in run_region() used to be
    // computed entirely from this rank's own buffers, with no MPI_Allreduce
    // or equivalent before `all_satisfied()` was called -- a silent
    // hang-or-wrong-answer risk when the region's components sit on more
    // than one rank, because two ranks measuring different slices of a
    // not-yet-converged field could satisfy the tolerance on different
    // sweeps and desynchronise the region's own per-sweep exchanges. Fixed
    // by reducing every measure's residual and reference contribution
    // across the region's participants on every sweep (run_region(), via
    // reduce_region_residuals()); `R.participants`, computed below, is what
    // that reduction runs over.
    //
    // Checked against `config.base_communicator` rather than the graph's
    // declared `RankSet` cardinality alone -- a Tier 0 caller with no
    // communicator configured (e.g. verification.iteration) legitimately
    // declares a region's components on different nominal ranks while
    // running all of them, right here, on the one process that exists;
    // there is no second rank to reduce across, and `R.participants` stays
    // empty (run_region() takes that as "purely local", unchanged from
    // before this fix).
    if (config.base_communicator.has_value()) {
      std::vector<int> region_ranks;
      for (const ComponentId c : cp.components) {
        for (const int rk : graph.component(c).ranks) region_ranks.push_back(rk);
      }
      // The coupler-placement track: a channel inside this region that
      // routes through a separately-placed coupler (RemapSpec::remap_on)
      // makes the coupler's ranks participants of every Transport/Epoch
      // that channel's lowering emitted -- even though the coupler owns no
      // state and so is never named by `derive_checkpoint_set()`'s
      // `components` above. Missing them here would leave the coupler's
      // ranks executing the region's collectives (its own Transport/Epoch
      // pairs are inside [R.enter, R.exit] like any other) while sitting
      // outside `R.participants`' residual reduction -- not a wrong answer,
      // since the coupler carries no residual to contribute, but a rank
      // that took part in one collective and not the other is exactly the
      // hang class E-SCHED-EPOCH-PARTIAL exists to catch at compile time;
      // this is the same property, enforced at the ONE place a scheduler
      // check cannot reach because it is about which RUNTIME ranks reduce a
      // residual, not which ranks a single instruction's attrs name. Read
      // directly off the instruction stream -- the same source
      // `epoch_sequence()` (src/schedule/scheduler.cpp) reads -- rather than
      // re-derived from the graph, so the two cannot disagree about which
      // ranks a region's body actually touches.
      for (std::size_t i = R.enter; i <= R.exit; ++i) {
        const Instruction& ins = stream[i];
        if (ins.kind != OpKind::Transport && ins.kind != OpKind::Epoch) continue;
        for (const int rank_set : {ins.attrs.src_rank_set, ins.attrs.dst_rank_set}) {
          if (rank_set < 0 || static_cast<std::uint32_t>(rank_set) >=
                                  graph.component_count()) {
            continue;
          }
          for (const int rk : graph.component(ComponentId{
                   static_cast<std::uint32_t>(rank_set)}).ranks) {
            region_ranks.push_back(rk);
          }
        }
      }
      std::sort(region_ranks.begin(), region_ranks.end());
      region_ranks.erase(std::unique(region_ranks.begin(), region_ranks.end()),
                         region_ranks.end());
      if (region_ranks.size() > 1) {
        // The one case this fix does NOT cover: a dynamic accelerator
        // (Aitken and everything with more history than that) derives its
        // relaxation coefficient from an inner product over this rank's own
        // local iterate history (fuse::num, src/num/accel_quasi_newton.cpp
        // and accel_relaxation.cpp's Aitken branch -- outside this fixer's
        // owned paths, and `Accelerator::update()`'s signature has no hook
        // for a caller-supplied global reduction). Two ranks with different
        // local slices would compute different coefficients from that
        // inner product and take different accelerator paths even with the
        // residual itself correctly agreed, which is the same
        // desynchronisation risk R-10 opened with. `None` (plain Picard)
        // and `Constant` (a fixed, config-supplied omega, identical on
        // every rank because AccelConfig comes from the graph) need
        // nothing beyond the residual reduction below and are genuinely
        // supported.
        if (decl.acceleration.kind != AccelKind::None &&
            decl.acceleration.kind != AccelKind::Constant) {
          return Failure{
              "E-RT-REGION-MPI",
              "iteration region '" + R.name + "' checkpoints components "
              "placed on " + std::to_string(region_ranks.size()) +
              " different ranks and names accelerator '" +
              std::string(to_cstring(decl.acceleration.kind)) +
              "', whose dynamic coefficient is computed from this rank's "
              "own local iterate history with no cross-rank agreement "
              "(REVIEW.md R-10) and no hook to add one from here. Use "
              "AccelKind::None or AccelKind::Constant for a region spanning "
              "more than one rank, or place every component this region "
              "checkpoints on the same rank"};
        }
        R.participants = std::move(region_ranks);
      }
    }

    // --- the residual blocks, and the Sink each is delivered through --------
    std::size_t offset = 0;
    for (const ArcId res : reg.residuals) {
      if (!res.valid() || res.value() >= dag.arc_count()) {
        return Failure{"E-RT-REGION", "iteration region '" + R.name +
                                        "' names a residual arc outside the "
                                        "operator DAG"};
      }
      const NodeId prod = dag.arc(res).producer;
      if (!prod.valid() || dag.node(prod).inputs.empty()) {
        return Failure{"E-RT-REGION",
                     "the residual arc %" + std::to_string(res.value()) +
                         " of iteration region '" + R.name +
                         "' has no producer with an input, so there is no "
                         "coupling value to measure"};
      }
      detail::RegionBlock b;
      b.residual_arc = res.value();
      b.value_arc = dag.node(prod).inputs.front().value();
      if (b.value_arc >= slots.size()) {
        return Failure{"E-RT-REGION", "iteration region '" + R.name +
                                        "' measures an arc with no buffer"};
      }
      b.length = slots[b.value_arc].total();

      // window_export_meta's own top comment, above: resolve whether
      // this channel's value arc is a window member's own export, and
      // if so, this block's own residual computation (run_region())
      // strips the window-start sample before norming.
      //
      // `b.value_arc` itself is the Residual node's own input, NOT
      // necessarily the WindowAdvance instruction's own raw `ins.out`
      // directly: measured, 2026-10-03 (this delegation's own report),
      // the Brusselator's own graph interposes one `OpKind::Interpolate`
      // node (a SHAPE-PRESERVING resample of the window waveform, same
      // `length*capacity` total() in as out, for an identity transfer
      // with no real fuse::Remap -- brusselator_graph.hpp's own "WHY
      // BOTH TRANSFER FIELDS LIVE ON u_region" comment) between the raw
      // export and the Residual's own input. Walk back through a
      // bounded chain of such single-input, size-preserving nodes
      // (never more than `kMaxHops`, so a graph shape this was never
      // measured against fails CLOSED -- `b.window_export_capacity`
      // stays 0, this block's own residual reverts to the unstripped,
      // pre-this-fix computation, rather than silently matching the
      // wrong arc) until `window_export_meta` resolves or the chain
      // runs out.
      {
        constexpr int kMaxHops = 4;
        std::uint32_t probe = b.value_arc;
        for (int hop = 0; hop <= kMaxHops; ++hop) {
          const auto found = window_export_meta.find(probe);
          if (found != window_export_meta.end()) {
            b.window_export_capacity = found->second.first;
            b.window_export_valid_substeps = found->second.second;
            break;
          }
          if (probe >= slots.size()) break;
          const NodeId probe_prod = dag.arc(ArcId{probe}).producer;
          if (!probe_prod.valid()) break;
          const auto& pn = dag.node(probe_prod);
          if (pn.inputs.size() != 1) break;
          const std::uint32_t next = pn.inputs.front().value();
          if (next >= slots.size() ||
              slots[next].total() != slots[probe].total()) {
            break;
          }
          probe = next;
        }
      }

      // The delivery. Without a Sink the accelerated iterate never reaches a
      // component, so every sweep would re-advance from identical inputs and
      // the residual would sit still until max_iterations ran out.
      //
      // Counts DISTINCT origin nodes, not instruction occurrences: an
      // ordinary region's own delivering Sink can legitimately sit
      // OUTSIDE [R.enter, R.exit] (region_membership(), scheduler.cpp,
      // marks a region's own body/residual-producing nodes only, not
      // every node that feeds one), so restricting the scan range is
      // wrong; but a window-scoped region's window may tick several
      // times per period (R.passes' own comment), and each tick is its
      // OWN separate Instruction carrying the SAME origin_node
      // (finish_window_regions()'s "at most one such node per member per
      // region" contract, src/ir/window_lowering.cpp) -- scanning the
      // whole stream by INSTRUCTION would find one matching Sink per
      // window instance, all for the identical single Sink NODE, and
      // report "delivered by 7 Sink instructions" for a 91/13 = 7-window
      // period where exactly one Sink node genuinely does the delivering.
      // De-duplicating by node, rather than restricting the range, is
      // correct for both shapes at once.
      std::set<std::uint32_t> sink_nodes;
      for (std::size_t i = 0; i < stream.size(); ++i) {
        if (stream[i].kind != OpKind::Sink || stream[i].n_in == 0) continue;
        if (stream[i].in[0].value() != b.value_arc) continue;
        b.sink_instruction = i;
        b.feed_arc = stream[i].out.value();
        sink_nodes.insert(stream[i].origin_node.value());
      }
      const std::size_t sinks = sink_nodes.size();
      if (sinks != 1) {
        return Failure{"E-RT-REGION",
                     "iteration region '" + R.name + "': the value on arc %" +
                         std::to_string(b.value_arc) + " is delivered by " +
                         std::to_string(sinks) +
                         " Sink instructions; exactly one is required, because "
                         "the sweep re-delivers the accelerated iterate through "
                         "it. With none, every sweep re-advances from identical "
                         "inputs and the region cannot converge"};
      }
      if (b.feed_arc >= slots.size() ||
          slots[b.feed_arc].total() != b.length) {
        return Failure{"E-RT-REGION",
                     "iteration region '" + R.name + "': the coupling value (" +
                         std::to_string(b.length) +
                         " values) and the storage it is delivered into do not "
                         "have the same length"};
      }
      // R-10 (REVIEW.md): a rank that is a genuine participant of a
      // multi-rank region (R.participants.size() > 1) but owns no local
      // entries of THIS measure's coupling value is ordinary -- the same
      // "n == 0 is a success no-op" case Accelerator::update() documents --
      // not an error: reduce_region_residuals() contributes an undefined
      // (owned == 0, never a converged-looking 0) entry for it and the
      // region converges on the reduced, globally-owned residual. Outside
      // that case (no communicator, or every checkpointed component on one
      // rank), an empty coupling value is still what it always was: a
      // region that would report convergence after one sweep with nothing
      // ever measured, refused below.
      if (b.length == 0 && R.participants.size() <= 1) {
        return Failure{"E-RT-REGION",
                     "iteration region '" + R.name +
                         "' measures an empty coupling value, so its residual "
                         "norm is identically zero and the region would report "
                         "convergence after one sweep"};
      }

      const ChannelId origin = dag.node(prod).origin;
      if (origin.valid() && origin.value() < graph.channel_count()) {
        const Channel& ch = graph.channel_at(origin);
        b.channel_name = ch.name;
        if (ch.source.valid() && ch.source.value() < graph.field_count()) {
          b.source_field = graph.field(ch.source).name;
        }
        if (ch.target.valid() && ch.target.value() < graph.field_count()) {
          b.target_field = graph.field(ch.target).name;
        }
      }
      b.offset = offset;
      offset += b.length;
      R.blocks.push_back(std::move(b));
    }
    R.n = offset;
    if (R.blocks.empty()) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name +
                       "' has no residual, so there is nothing to test for "
                       "convergence and the loop could only ever run to "
                       "max_iterations"};
    }

    // --- the convergence measures against those blocks ----------------------
    //
    // Both rules -- exactly one target per measure, and every target measured
    // -- live in match_convergence_measures() (src/ir/checkpoint.cpp) rather
    // than here, because the C ABI's soma_initialize() has to enforce the
    // identical rules with no Runner and no DAG regions to hand. The only
    // thing that differs between the two call sites is where the candidate
    // names come from: the DAG's residual blocks here, the graph's channel
    // list there.
    std::vector<MeasureTarget> targets;
    targets.reserve(R.blocks.size());
    for (const detail::RegionBlock& b : R.blocks) {
      targets.push_back(MeasureTarget{b.channel_name, b.source_field,
                                      b.target_field});
    }
    Result<std::vector<std::size_t>> matched =
        match_convergence_measures(R.name, R.convergence, targets);
    if (!matched.ok()) return matched.error();
    R.measure_block = std::move(matched.value());

    // --- the ticks the sweep re-runs ---------------------------------------
    //
    // Time windows track: this ENTIRE block is skipped for a window-scoped
    // region. It re-derives the ordinary re-advance mechanism (a
    // component's own native-interval Tick, re-run once per sweep via
    // R.tick_indices) -- but a window member gets NO Tick at all
    // (build_ticks()'s own window_member_components() exclusion, src/
    // runtime/runner.cpp): its entire re-advance sequence, every substep,
    // is driven directly by its own OpKind::WindowAdvance instruction
    // (advance_window(), called from run_region()'s body loop below), not
    // by anything this block could find in `all_ticks`/`ticks` at all.
    // Applying this block's own "re-advances no component" refusal to a
    // window region would therefore misfire on every one of them.
    if (!R.window_scoped) {
      const Rational enter_time = stream[R.enter].time;
      // Validated over `all_ticks`, identical on every process, so every
      // process reaches the same verdict; only this process's own steps
      // (`ticks`) are then recorded for re-stepping.
      std::size_t member_steps = 0;
      for (std::size_t t = 0; t < all_ticks.size(); ++t) {
        const bool member =
            std::find(R.component_slots.begin(), R.component_slots.end(),
                      all_ticks[t].slot) != R.component_slots.end();
        if (!member) continue;
        ++member_steps;
        // `>=`, not `>`. A member tick at EXACTLY `enter_time` has not run when
        // the region body starts: the enter instruction is `BeforeTicks` (see
        // TickPhase), so the main cursor in step_period() has advanced only the
        // ticks strictly before it. Admitting such a tick here would put it in
        // `tick_indices`, so every sweep from k=1 would re-advance a component
        // that never advanced at iteration 0, and the main cursor would advance
        // it once more after the region exits -- a double advance with no
        // diagnostic.
        //
        // REACHABILITY (R12 follow-up, corrected -- this comment previously
        // claimed "no graph reaches this today ... which E-SCHED-REGION refuses
        // first"; that was false, disproved by construction rather than by
        // argument). A region whose body contains a space-time channel's
        // `Sample` node reaches it directly: `Sample` runs `AfterTicks` at the
        // SAME nominal time as the tick that feeds it (a single-source-step
        // history, the only shape E-SCHED-REGION's contiguity rule admits
        // inside a region, has nothing to wait for before sampling), so it
        // becomes the region's chronologically first body-member instruction
        // and pins `enter_time` to that tick's own time -- after which the
        // very tick that produced the sampled value fails `ticks[t].time <
        // enter_time`. This holds whether the space-time channel's source is
        // one of the region's own re-advanced components, or an external
        // component pulled into the same checkpoint set because its channel is
        // a body member (`derive_checkpoint_set()`, src/ir/checkpoint.cpp, does
        // not distinguish the two). See
        // `Iteration.ASpacetimeChannelInAnIterationRegionIsRefusedNotSilentlyWrong`
        // (test/verification/iteration_test.cpp) for both constructions,
        // measured. The guard still must not depend on reachability being rare
        // -- it refuses correctly here too -- but "no graph reaches this" is no
        // longer an accurate reason to leave it undertested.
        if (!(all_ticks[t].time < enter_time)) {
          return Failure{"E-RT-REGION",
                       "component '" + components.at(all_ticks[t].slot).name() +
                           "' is checkpointed by iteration region '" + R.name +
                           "' but has an advance at t=" +
                           all_ticks[t].time.to_string() +
                           ", at or after the region begins at t=" +
                           enter_time.to_string() +
                           ". That advance would run once, outside the loop, "
                           "from state the region had already iterated"};
        }
      }
      for (std::size_t t = 0; t < ticks.size(); ++t) {
        const bool member =
            std::find(R.component_slots.begin(), R.component_slots.end(),
                      ticks[t].slot) != R.component_slots.end();
        if (member && ticks[t].time < enter_time) R.tick_indices.push_back(t);
      }
      if (member_steps == 0) {
        return Failure{"E-RT-REGION",
                     "iteration region '" + R.name +
                         "' re-advances no component, so a sweep would recompute "
                         "the same values from the same inputs for ever"};
      }
    }  // if (!R.window_scoped)

    // --- the loop-carried coupler buffers ----------------------------------
    //
    // THEY ARE DELIBERATELY NOT RESTORED BETWEEN SWEEPS, and this is a
    // correction to a literal reading of src/ir/checkpoint.cpp's header ("a
    // value needs saving exactly when the body both reads and writes it across
    // the sweep boundary"). Restoring them is provably the wrong physics, by
    // the arithmetic of the only two node kinds the analysis names:
    //
    //   * Relax computes `out = w*chain + (1-w)*prev` (num::relax) and then
    //     sets `prev = out`. Reset `prev` to its period-start value x0 on every
    //     sweep and the composed map's fixed point satisfies
    //     `x* = w*chain(x*) + (1-w)*x0`, which is the coupling fixed point
    //     `x* = chain(x*)` only when w == 1 or x0 already was the answer. A
    //     relaxed region would converge, smoothly, to a number that depends on
    //     where the window started.
    //   * Accelerate's state IS the residual and iterate history. Resetting it
    //     every sweep is exactly `reset()` every sweep, i.e. no acceleration
    //     at all -- and checkpoint.cpp names it as carried precisely because it
    //     accumulates over sweeps.
    //
    // `carried` is what must SURVIVE a sweep, which is also how
    // schedule/buffers.hpp reads it (those buffers may not be recycled inside a
    // period) and how schedule/acyclicity.cpp reads it (they are what breaks
    // the dependence cycle). plan_buffers() already gives each of them its own
    // storage, so surviving costs the runner nothing.
    //
    // What the sweep loop DOES owe them is consistency with the iterate it
    // delivered: see the previous-iterate sync in run_region(). So that this
    // is a checked claim and not an omission, every carried arc must be one the
    // loop accounts for.
    for (const ArcId a : reg.carried) {
      if (!a.valid() || a.value() >= slots.size()) {
        return Failure{"E-RT-REGION",
                     "iteration region '" + R.name +
                         "' carries an arc with no buffer"};
      }
      R.carried_arcs.push_back(a.value());
      bool accounted = false;
      for (const detail::RegionBlock& b : R.blocks) {
        if (b.value_arc == a.value() || b.residual_arc == a.value()) {
          accounted = true;
          break;
        }
      }
      if (!accounted) {
        return Failure{
            "E-RT-REGION",
            "the derived checkpoint set of iteration region '" + R.name +
                "' names coupler buffer %" + std::to_string(a.value()) +
                " as loop-carried, but it is not the value or the residual of "
                "any of the region's channels, so the sweep loop has no way to "
                "keep its previous-iterate shadow consistent with the iterate "
                "it delivers. Refused rather than left to drift"};
      }
    }

    // --- nothing outside the region may read iterated state -----------------
    const Result<void> taint = check_region_isolation(R, cp);
    if (!taint.ok()) return taint;

    // --- the accelerator ----------------------------------------------------
    //
    // A WaveformIteration::GaussSeidel window's own UNLAGGED transfer is
    // resolved WITHIN the same sweep it is fed from (WindowMemberConfig::
    // import_lagged's own comment) -- it carries no residual of its own to
    // accelerate, exactly like WindowCoupler's `accel_members`/
    // `accel_member_offset`/`accel_total` restriction excludes it from the
    // accelerated vector there (window_coupler.hpp/.cpp). R.blocks below is
    // NOT YET restricted the same way: every block this region measures,
    // lagged or not, currently enters R.accel's vector undifferentiated.
    // Running a real accelerator over that undifferentiated vector would
    // accelerate a quantity the method does not define an iterate history
    // for, which is a silently-wrong answer, not a scoped approximation --
    // so a GaussSeidel window rejects any accelerator but `AccelKind::None`
    // until R.blocks carries the same restriction.
    if (R.window_scoped &&
        decl.window_scope->iteration == WaveformIteration::GaussSeidel &&
        decl.acceleration.kind != AccelKind::None) {
      return Failure{
          "E-RT-WINDOW-ACCEL",
          "iteration region '" + R.name + "' is a WaveformIteration::"
          "GaussSeidel coupling window naming accelerator '" +
          std::string(to_cstring(decl.acceleration.kind)) +
          "'; the Runner does not yet exclude the unlagged transfer's own "
          "residual block from the accelerated vector the way "
          "WindowCoupler's accel_members/accel_member_offset/accel_total "
          "restriction does, so accelerating GaussSeidel here would "
          "compute something other than the method. Use AccelKind::None "
          "for a GaussSeidel window until that restriction is wired in"};
    }

    Result<std::unique_ptr<Accelerator>> acc =
        make_accelerator(decl.acceleration);
    if (!acc.ok()) return acc.error();
    R.accel = std::move(acc.value());
    if (R.blocks.size() > 1) {
      // Per-field scaling (spec §5.3) needs to know where each field starts.
      std::vector<std::size_t> offsets;
      offsets.reserve(R.blocks.size());
      for (const detail::RegionBlock& b : R.blocks) offsets.push_back(b.offset);
      R.accel->set_field_partition(std::move(offsets));
    }

    // --- everything the sweep loop touches, sized once ----------------------
    R.x_old.assign(R.n, 0.0);
    R.x_new.assign(R.n, 0.0);
    R.x_out.assign(R.n, 0.0);
    R.block_residual.assign(R.blocks.size(), 0.0);
    R.block_value.assign(R.blocks.size(), 0.0);
    R.block_first.assign(R.blocks.size(), 0.0);
    R.residual_norms.assign(R.convergence.size(), 0.0);
    R.reference_norms.assign(R.convergence.size(), 0.0);
    region_stats[ri].residual_history.reserve(
        static_cast<std::size_t>(R.max_iterations));
  }
  return Result<void>::success();
}

/// A BACKSTOP, not the first line of defence. `schedule()`'s contiguity rule
/// (E-SCHED-REGION) already refuses every graph that can be authored with an
/// explicit channel hanging off an iterated component, because that channel's
/// instructions interleave with the region body -- measured, see
/// RejectsAnExplicitChannelHangingOffAnIteratedComponent. This catches the case
/// where they do not interleave, and any plan that did not come from
/// `schedule()`. It runs once, at setup, so the cost of keeping it is nothing
/// and the cost of not having it is a converged iterate that never reaches a
/// consumer.
Result<void> Runner::Impl::check_region_isolation(const detail::RegionRuntime& R,
                                                  const CheckpointSet& cp) {
  // Every arc reachable from a checkpointed component's own output. A value
  // downstream of one of these changes on every sweep, so an instruction that
  // reads it and is not re-run by the loop is holding a pre-iteration value.
  std::vector<bool> tainted(slots.size(), false);
  for (const ComponentId c : cp.components) {
    for (std::uint32_t a = 0; a < slot_source_owner.size(); ++a) {
      if (slot_source_owner[a] == static_cast<std::int32_t>(c.value())) {
        tainted[a] = true;
      }
    }
  }
  for (const NodeId n : dag.topological_order()) {
    const Node& node = dag.node(n);
    if (!node.output.valid() || node.output.value() >= tainted.size()) continue;
    for (const ArcId in : node.inputs) {
      if (in.valid() && in.value() < tainted.size() && tainted[in.value()]) {
        tainted[node.output.value()] = true;
        break;
      }
    }
  }

  const std::vector<Instruction>& stream = plan.instructions();
  // Time windows track: a window-scoped region can have more than one
  // pass in one period (RegionRuntime::passes' own comment) -- every one
  // of them is inside the loop, not just the first, so every one must be
  // excluded here.
  const auto inside_some_pass = [&](std::size_t i) {
    for (const auto& [pe, px] : R.passes) {
      if (i >= pe && i <= px) return true;
    }
    return false;
  };
  for (std::size_t i = 0; i < stream.size(); ++i) {
    if (inside_some_pass(i)) continue;
    bool is_delivery = false;
    for (const detail::RegionBlock& b : R.blocks) {
      if (b.sink_instruction == i) { is_delivery = true; break; }
    }
    if (is_delivery) continue;
    for (std::uint8_t k = 0; k < stream[i].n_in; ++k) {
      const std::uint32_t a = stream[i].in[k].value();
      if (a >= tainted.size() || !tainted[a]) continue;
      return Failure{
          "E-RT-REGION",
          "instruction " + std::to_string(i) + " (" +
              std::string{to_cstring(stream[i].kind)} + " reading arc %" +
              std::to_string(a) +
              ") lies outside iteration region '" + R.name +
              "' but reads a value produced by a component the region "
              "re-advances on every sweep. It would run once, on the "
              "pre-iteration value, and the converged iterate would never "
              "reach it -- a wrong answer with no symptom. Put the channel "
              "inside the region, or take the component out of it"};
    }
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// Checkpointing
// ---------------------------------------------------------------------------

Result<void> Runner::Impl::checkpoint_regions(ComponentSet& components) {
  for (detail::RegionRuntime& R : regions) {
    // Time windows track: a window-scoped region's own checkpoint happens
    // once per PASS (once per window instance), inside run_region()
    // itself, not once per period here -- see RegionRuntime::
    // window_scoped's own comment for why an ordinary region's
    // once-at-period-top timing would be wrong for a window member (it
    // has no ordinary tick to have advanced yet by period-top, but by the
    // SECOND window instance in the same period it does, and this
    // period-top call would capture stale, pre-that-window-instance
    // state).
    if (R.window_scoped) continue;
    for (const std::size_t slot : R.component_slots) {
      const Result<void> s = components.at(slot).save_state();
      if (!s.ok()) return s;
      // KNOWN_ISSUES.md D12: consumed by advance() the next time it is
      // about to suspend for this slot -- see runner_impl.hpp's own
      // comment on slot_just_checkpointed/slot_just_restored.
      if (slot < slot_just_checkpointed.size()) slot_just_checkpointed[slot] = 1;
    }
    // The bytes the verification below compares against. Taken here, right
    // after save_state(), so they are exactly what the component claims to
    // have captured.
    for (std::size_t v = 0; v < R.verify_arcs.size(); ++v) {
      const detail::Slot& s = slots[R.verify_arcs[v]];
      if (s.data != nullptr) {
        num::copy(s.data, R.verify_save[v].data(), R.verify_save[v].size());
      }
    }
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// The sweep loop
// ---------------------------------------------------------------------------

detail::Task<Result<void>> Runner::Impl::run_region(
    ComponentSet& components, detail::RegionRuntime& R, std::size_t enter,
    std::size_t exit) {
  // Re-used, per-sweep scratch for the per-block residual computation
  // below: a window-scoped block's own stripped (window-start sample
  // excluded) copy of x_new_block/x_old_block, built fresh every sweep
  // from whichever block in R.blocks is currently widest (RegionBlock::
  // window_export_capacity's own top comment, runner_impl.hpp).
  // std::vector::resize() is a no-op once big enough, so this never
  // reallocates past the first sweep that needs it.
  std::vector<double> window_residual_new, window_residual_old;
  IterationStats& st = region_stats[R.index];
  st.iterations = 0;
  st.converged = false;
  st.residual_history.clear();  // keeps the capacity reserved at setup

  const std::vector<Instruction>& stream = plan.instructions();

  R.accel->reset();

  // Time windows track: a window-scoped region's own checkpoint happens
  // HERE, once per PASS (once per window instance) -- checkpoint_regions()
  // (called once per PERIOD, from step_period()) explicitly skips a
  // window-scoped region, per RegionRuntime::window_scoped's own comment
  // (a window member has no ordinary tick to have advanced yet by
  // period-top, but by the SECOND window instance in the same period it
  // does, and a period-top checkpoint would capture stale, pre-that-
  // window-instance state). Mirrors checkpoint_regions()'s own body
  // exactly (save_state() over R.component_slots, then the verify_arcs
  // byte-capture used later by the rollback-verification check), just
  // moved to run once per CALL (= once per window instance) instead of
  // once per period.
  if (R.window_scoped) {
    // Resolve any straddle still pending from the PREVIOUS window
    // instance BEFORE taking THIS window's own checkpoint, below --
    // that checkpoint would otherwise overwrite the very state the
    // replay needs to restore_state() back to (resolve_pending_window_
    // straddles()'s own comment, runner_impl.hpp).
    const Result<void> replay_r = resolve_pending_window_straddles(
        components, R);
    if (!replay_r.ok()) co_return replay_r.error();

    for (const std::size_t slot : R.component_slots) {
      const Result<void> s = components.at(slot).save_state();
      if (!s.ok()) co_return s;
      // KNOWN_ISSUES.md D12: see the matching line in checkpoint_regions().
      if (slot < slot_just_checkpointed.size()) slot_just_checkpointed[slot] = 1;
    }
    for (std::size_t v = 0; v < R.verify_arcs.size(); ++v) {
      const detail::Slot& s = slots[R.verify_arcs[v]];
      if (s.data != nullptr) {
        num::copy(s.data, R.verify_save[v].data(), R.verify_save[v].size());
      }
    }
  }

  // y_0: what the components were ACTUALLY advanced with. Reading the Sink
  // outputs -- the components' own Requires arrays -- rather than recording
  // what we last wrote means the residual measures the real map, including
  // anything the delivery did on the way.
  for (const detail::RegionBlock& b : R.blocks) {
    num::copy(slots[b.feed_arc].data, R.x_old.data() + b.offset, b.length);
  }

  for (int k = 0; k < R.max_iterations; ++k) {
    // Time windows track: snapshot every LAGGED window_import arc BEFORE
    // this sweep's own body runs -- R.window_lagged_import_arcs' own
    // comment (runner_impl.hpp). advance_window() reads `previous[arc]`
    // for a lagged import instead of the live buffer, so it sees LAST
    // sweep's delivery (or, at k=0 of a new window instance, the
    // PREVIOUS window's final delivery -- nothing has touched this arc
    // since then) regardless of where the delivering Sink sits in the
    // stream relative to this member's own WindowAdvance instruction.
    for (const ArcId a : R.window_lagged_import_arcs) {
      const detail::Slot& s = slots[a.value()];
      std::vector<double>& shadow = previous[a.value()];
      if (shadow.size() != s.total()) shadow.assign(s.total(), 0.0);
      if (s.data != nullptr) num::copy(s.data, shadow.data(), s.total());
    }
    for (std::size_t i = enter; i <= exit; ++i) {
      // The same rule as the period loop (`is_collective`, fuse/ir/op.hpp):
      // after a local failure keep executing the collectives, so no partner is
      // left inside one waiting for this rank, and skip the local arithmetic.
      // An iteration region is where this matters most -- its body runs many
      // times, so a rank that drops out of the exchange desynchronises the
      // remaining iterations rather than just one step.
      if (step_failure.has_value() && !is_collective(stream[i].kind)) continue;
      // Time windows track: execute() has no ComponentSet& and cannot call
      // advance()/save_state()/restore_state() itself (see its own
      // OpKind::WindowAdvance case, src/runtime/runner.cpp) -- a
      // WindowAdvance instruction is dispatched to advance_window()
      // instead, here, the one place that both has `components` and is
      // guaranteed a WindowAdvance instruction only ever appears inside
      // some region's body.
      Result<void> e = Result<void>::success();
      if (stream[i].kind == OpKind::WindowAdvance) {
        e = co_await advance_window(components, stream[i], resolved[i], k);
      } else {
        e = execute(stream[i], resolved[i]);
      }
      if (!e.ok()) {
        if (!step_failure.has_value()) step_failure = e.error();
        if (stream[i].kind == OpKind::Epoch) co_return *step_failure;
        continue;
      }
      ++stats.instructions_executed;
    }

    // G(y), and the residual G(y) - y. Computed unconditionally, even after
    // a local failure earlier in this same sweep: R-10 (REVIEW.md)'s
    // cross-rank reduction just below is itself a collective every
    // participant must reach on every sweep -- mirroring reconcile_epoch(),
    // and for the same reason -- so the local half of it cannot be skipped
    // just because this rank already knows it failed. A rank whose local
    // step_failure is set contributes an undefined (never a
    // converged-looking zero) entry instead, below.
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      const double* g = slots[b.value_arc].data;
      double* x_new_block = R.x_new.data() + b.offset;
      const double* x_old_block = R.x_old.data() + b.offset;

      // KNOWN_ISSUES.md D11 (closing) / matrix cell A-SPACES-TRANSPORT: the
      // copy is elementwise -- trivially thread-count-independent, same
      // argument as every other elementwise kernel in
      // fuse/num/space_kernels.hpp. The two norms below are REDUCTIONS, and
      // are exactly the case that header's `deterministic_norm_squared`/
      // `deterministic_norm_squared_diff` exist for: a plain accumulator
      // (what this loop used to be) or a `Kokkos::parallel_reduce` both
      // associate depending on how the work is split, which a runtime whose
      // bit-for-bit reproducibility across thread counts is a requirement
      // cannot use. `reduction_scratch` is sized once, in initialize(), from
      // the largest block any region has; nothing here resizes it.
      num::copy<HostExecSpace>(num::ConstSpanOf<HostExecSpace>(g, b.length),
                               num::SpanOf<HostExecSpace>(x_new_block,
                                                          b.length));

      // Time windows track: a window-scoped value arc carries one
      // window-START sample (index 0 of every element's own
      // `window_export_capacity`-wide stride) ahead of its real
      // substep-end samples -- RegionBlock::window_export_capacity's
      // own top comment (runner_impl.hpp). That sample never changes
      // within one window instance (Runner::Impl::window_start_value's
      // own fix), so it never contributes to the RESIDUAL, but DOES
      // inflate the VALUE norm by a constant every sweep -- loosening
      // a `MeasureKind::Relative` tolerance and able to flip the
      // convergence decision onto a different sweep than the driver's
      // own named-measure residual, which is sized to EXACTLY the
      // member's own substep count and never includes it
      // (`WindowCoupler::sweep_window()`'s own comment). Measured,
      // 2026-10-03 (this delegation's own report): the Brusselator's
      // own window 20. Gathered into scratch, in the SAME element-
      // major order the driver's own flat vector uses, so the two
      // sides norm the identical set of entries -- never touching
      // `x_new_block`/`x_old_block` (== `R.x_new`/`R.x_old`)
      // themselves, which the accelerator and the rollback-
      // verification byte-capture both read over the FULL arc and
      // must keep seeing exactly what `advance_window()` wrote.
      std::size_t norm_length = b.length;
      const double* new_ptr = x_new_block;
      const double* old_ptr = x_old_block;
      if (b.window_export_capacity > 0) {
        const std::size_t capacity = b.window_export_capacity;
        const std::size_t n_sub = b.window_export_valid_substeps;
        const std::size_t elem_len = b.length / capacity;
        norm_length = elem_len * n_sub;
        window_residual_new.resize(norm_length);
        window_residual_old.resize(norm_length);
        for (std::size_t e = 0; e < elem_len; ++e) {
          std::copy(x_new_block + e * capacity + 1,
                    x_new_block + e * capacity + 1 + n_sub,
                    window_residual_new.begin() +
                        static_cast<std::ptrdiff_t>(e * n_sub));
          std::copy(x_old_block + e * capacity + 1,
                    x_old_block + e * capacity + 1 + n_sub,
                    window_residual_old.begin() +
                        static_cast<std::ptrdiff_t>(e * n_sub));
        }
        new_ptr = window_residual_new.data();
        old_ptr = window_residual_old.data();
      }

      const std::size_t n_chunks = num::deterministic_chunks(norm_length);
      num::SpanOf<HostExecSpace> scratch(reduction_scratch.data(), n_chunks);
      const double vs = num::deterministic_norm_squared<HostExecSpace>(
          num::ConstSpanOf<HostExecSpace>(new_ptr, norm_length), scratch);
      const double rs = num::deterministic_norm_squared_diff<HostExecSpace>(
          num::ConstSpanOf<HostExecSpace>(new_ptr, norm_length),
          num::ConstSpanOf<HostExecSpace>(old_ptr, norm_length), scratch);
      R.block_residual[bi] = std::sqrt(rs);
      R.block_value[bi] = std::sqrt(vs);
    }

    // R-10 (REVIEW.md): agree the residual across the region's participants
    // before testing convergence or letting the accelerator see it, so
    // every participant decides from the same globally-owned norm rather
    // than its own local slice. `R.participants.size() <= 1` (no
    // communicator, or every checkpointed component on one rank) skips
    // this entirely -- run_region() is then exactly what it always was.
    if (R.participants.size() > 1) {
      const std::size_t nb = R.blocks.size();
      std::vector<double> payload(1 + 4 * nb);
      payload[0] = step_failure.has_value() ? 0.0 : 1.0;
      for (std::size_t bi = 0; bi < nb; ++bi) {
        // "n == 0 is a success no-op. A rank that owns no interface
        // elements is ordinary in a parallel run, not an error"
        // (Accelerator::update()'s own doc comment) -- the same rule
        // applies to a rank contributing nothing to this measure. Zero is
        // never sent as a value: field_norm()'s convention (undefined, not
        // a converged zero) is `owned == 0`, so a locally-failed rank or a
        // rank with no local entries for this block sends owned == 0 and
        // reduce_region_residuals() (via num::combine()) excludes it.
        const std::size_t length = R.blocks[bi].length;
        const bool owned_here = !step_failure.has_value() && length > 0;
        const double owned = owned_here ? static_cast<double>(length) : 0.0;
        payload[1 + 4 * bi + 0] = owned_here ? R.block_residual[bi] : 0.0;
        payload[1 + 4 * bi + 1] = owned;
        payload[1 + 4 * bi + 2] = owned_here ? R.block_value[bi] : 0.0;
        payload[1 + 4 * bi + 3] = owned;
      }
      const Result<void> red = reduce_region_residuals(R, payload);
      if (!red.ok()) co_return red.error();
      if (payload[0] < 0.5) {
        // A participant's local failure carries through the reduction
        // exactly as reconcile_epoch() carries an Epoch's: this rank's own
        // sweep may have succeeded locally, but stopping here -- rather
        // than waiting for the failed rank at the next sweep's reduction --
        // is the point of reaching the collective at all.
        if (!step_failure.has_value()) {
          step_failure = Failure{
              "E-RT-REGION-PEER",
              "a rank taking part in iteration region '" + R.name +
                  "'s residual agreement failed; stopping here rather than "
                  "waiting for it at the next sweep. The failing rank "
                  "reports the cause."};
        }
      } else {
        for (std::size_t bi = 0; bi < nb; ++bi) {
          if (payload[1 + 4 * bi + 1] == 0.0) {
            // Not one participant owned any part of this block this sweep
            // -- a setup defect (the measure names a coupling value nobody
            // in the region's rank set ever provides), not a converged
            // zero, and not something the sweep loop can recover from.
            co_return Failure{
                "E-RT-REGION-UNMEASURED",
                "iteration region '" + R.name + "': no participant of the "
                "residual agreement owns any part of block " +
                    std::to_string(bi) + "'s coupling value this sweep"};
          }
          R.block_residual[bi] = payload[1 + 4 * bi + 0];
          R.block_value[bi] = payload[1 + 4 * bi + 2];
        }
      }
    }

    if (step_failure.has_value()) {
      // Leaving the region rather than iterating again: every agreement point
      // in the body -- and, now, this sweep's residual reduction -- has been
      // reached this pass, so the partners already know.
      const Failure f = *step_failure;
      step_failure.reset();
      co_return f;
    }
    ++st.iterations;
    ++stats.total_iterations;

    if (k == 0) {
      for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
        R.block_first[bi] = R.block_residual[bi];
      }
    }
    double total = 0.0;
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      total += R.block_residual[bi] * R.block_residual[bi];
    }
    st.residual_history.push_back(std::sqrt(total));

    for (std::size_t mi = 0; mi < R.measure_block.size(); ++mi) {
      const std::size_t bi = R.measure_block[mi];
      R.residual_norms[mi] = R.block_residual[bi];
      // convergence.hpp: Relative divides by the norm of the current value,
      // ResidualRelative by the first residual of the window. Same arithmetic,
      // different reference, and the enumerator is the only thing that says
      // which -- so it has to be honoured here, not approximated.
      R.reference_norms[mi] =
          R.convergence.measures()[mi].kind == MeasureKind::ResidualRelative
              ? R.block_first[bi]
              : R.block_value[bi];
    }
    // Throws only on a size mismatch, which resolve_regions() made impossible;
    // run_period()'s catch turns it into E-RT-RUN rather than letting it out of
    // a Result-returning call.
    if (R.convergence.all_satisfied(R.residual_norms, R.reference_norms)) {
      st.converged = true;
      break;
    }
    if (k + 1 == R.max_iterations) break;

    const Result<void> a = R.accel->update(R.x_old.data(), R.x_new.data(),
                                           R.x_out.data(), R.n, k);
    if (!a.ok()) {
      // E-ACCEL-DIVERGED and friends: x_out's contents are not part of the
      // contract after a failed update, so nothing here reads them.
      //
      // R-09 (REVIEW.md): record the failure and fall through to the
      // post-loop failure check below instead of returning straight out of
      // the sweep loop. A bare return here bypassed step_period()'s
      // failure-carries-forward mechanism once that caller stopped
      // early-returning on a failed region (the fix immediately above this
      // one, in runner.cpp): every remaining collective this rank owes its
      // partners -- in this region and in the rest of the period -- still
      // needs to be reached, and only step_period()'s continuation can do
      // that once this function returns.
      if (!step_failure.has_value()) step_failure = a.error();
      break;
    }

    // Roll the components back. The loop-carried coupler buffers are NOT rolled
    // back; see the long note in resolve_regions(). Restoring first and
    // delivering second is the order that matters here: restore_state() puts
    // the component's Requires arrays back to the checkpoint, so delivering
    // before it would be undone.
    for (const std::size_t slot : R.component_slots) {
      const Result<void> rs = components.at(slot).restore_state();
      if (!rs.ok()) co_return rs;
      // KNOWN_ISSUES.md D12: the re-advance below (R.tick_indices) is
      // this slot's first advance() after THIS restore_state(), so it is
      // the one that must report requires_restore -- and, since a
      // restore always supersedes whatever checkpoint flag an earlier
      // sweep of the SAME region entry left set, clear that one too.
      if (slot < slot_just_restored.size()) slot_just_restored[slot] = 1;
      if (slot < slot_just_checkpointed.size()) slot_just_checkpointed[slot] = 0;
    }

    // Spec §5.3's verification. `supports_rollback()` was a promise; this is
    // the check. Bytes, not a tolerance: a restore is a copy of what was saved,
    // so anything but an exact match means state the analysis named did not
    // come back -- and a NaN, which no tolerance comparison can see, does.
    //
    // R16: `R.verify_arcs` is REGISTERED storage only -- the `detail::Slot`s
    // the field registry knows about, one per arc the derived checkpoint set
    // (`CheckpointSet`, src/ir/lowering.hpp) said must survive a rejected
    // iterate. A component's own private member state (anything `advance()`
    // mutates that was never `bind()`-ed to an arc) is invisible to this
    // `memcmp` and to `stats.checkpoint_verifications`: nothing here can see
    // it, so nothing here can catch a `save_state()`/`restore_state()` pair
    // that gets the private part wrong while getting every registered field
    // byte-exact. "FUSE verifies rollback" is true of registered storage and
    // only registered storage; see the caveat on `Component::save_state()`
    // (include/fuse/runtime/component.hpp) and on
    // `RunStats::checkpoint_verifications` (include/fuse/runtime/runner.hpp)
    // for the same statement made where a component author is more likely to
    // read it first.
    for (std::size_t v = 0; v < R.verify_arcs.size(); ++v) {
      const detail::Slot& s = slots[R.verify_arcs[v]];
      const std::vector<double>& saved = R.verify_save[v];
      if (s.data == nullptr || saved.empty()) continue;
      ++stats.checkpoint_verifications;
      if (std::memcmp(s.data, saved.data(), saved.size() * sizeof(double)) ==
          0) {
        continue;
      }
      const std::size_t which = R.verify_component[v];
      co_return Failure{
          "E-ITER-ROLLBACK",
          "component '" +
              components.at(R.component_slots[which]).name() +
              "' is in the derived checkpoint set of iteration region '" +
              R.name +
              "', its supports_rollback() returned true, and its "
              "restore_state() returned success -- but the storage it "
              "registered for arc %" + std::to_string(R.verify_arcs[v]) +
              " does not hold the values it held when save_state() was called. "
              "save_state() must capture everything advance() writes, "
              "including the registered field arrays, and restore_state() must "
              "copy it back element-wise into that same storage. Sweep " +
              std::to_string(st.iterations) +
              " would otherwise start from state the previous sweep left "
              "behind, and the region would converge to the fixed point of a "
              "different operator"};
    }

    // Deliver the accelerated iterate, and bring every previous-iterate shadow
    // in the region into line with it.
    //
    // A shadow (`Impl::previous`) records "the value this operator produced
    // last time". After acceleration that is no longer the value the
    // components were fed, and the two operators that keep one both need the
    // fed value, not the raw one:
    //
    //   * Residual would otherwise report `G(y_{k+1}) - G(y_k)`, which is not
    //     the residual this loop converges on, so the arc anyone inspects
    //     would disagree with the convergence report.
    //   * Relax computes `w*chain + (1-w)*prev`. `prev` must be the iterate
    //     that was actually delivered, or the under-relaxation is taken about
    //     a point nothing was evaluated at.
    //
    // This is also what makes the carried buffers correct without restoring
    // them.
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      const double* fed = R.x_out.data() + b.offset;
      num::copy(fed, slots[b.value_arc].data, b.length);
      for (const std::uint32_t arc : {b.value_arc, b.residual_arc}) {
        std::vector<double>& shadow = previous[arc];
        if (shadow.size() == b.length) num::copy(fed, shadow.data(), b.length);
      }
      if (step_failure.has_value() &&
          !is_collective(stream[b.sink_instruction].kind)) {
        continue;
      }
      const Result<void> e = execute(stream[b.sink_instruction],
                                     resolved[b.sink_instruction]);
      if (!e.ok()) {
        if (!step_failure.has_value()) step_failure = e.error();
        continue;
      }
      ++stats.instructions_executed;
    }

    // Re-advance, over the SAME interval, with ctx.iteration = k + 1.
    for (const std::size_t t : R.tick_indices) {
      if (step_failure.has_value()) break;
      const Result<void> adv = co_await advance(components, ticks[t], k + 1);
      if (!adv.ok()) step_failure = adv.error();
    }
    R.x_old.swap(R.x_out);
  }

  // R-09 (REVIEW.md): the accelerator-divergence `break` above leaves this
  // sweep's rollback, delivery and re-advance unexecuted -- correctly, since
  // every collective this pass owed its partners was already reached inside
  // the body loop above (`for (std::size_t i = R.enter; ...)`), before the
  // residual and acceleration code runs at all. What must not happen is
  // falling into the non-convergence path below and reporting success (or a
  // stale E-RT-NOCONVERGE) instead of the failure that actually stopped the
  // sweep.
  if (step_failure.has_value()) {
    const Failure f = *step_failure;
    step_failure.reset();
    co_return f;
  }

  if (!st.converged) {
    ++stats.windows_not_converged;
    if (config.abort_on_nonconvergence) {
      co_return Failure{"E-RT-NOCONVERGE",
                   "iteration region '" + R.name + "' ran its full " +
                       std::to_string(R.max_iterations) +
                       " iterations in period " +
                       std::to_string(period_index) +
                       " without meeting its convergence criterion: " +
                       R.convergence.report(R.residual_norms,
                                            R.reference_norms) +
                       ". The values left behind are the last iterate, not a "
                       "converged solution"};
    }
  }

  // This region's own pass is over (converged, or run to max_iterations
  // with abort_on_nonconvergence == false, exactly the two cases that
  // reach here): every window-import node's own `st.history` -- the
  // NEXT window instance's own predictor input (advance_window()'s own
  // "use_predictor" comment) -- is refreshed here from the LIVE
  // `window_import` arc, never `previous[]`. advance_window() itself
  // updates this same entry once EVERY sweep, but from whichever of
  // `previous[]`/`slots[]` its own `cfg.import_lagged` reads for the
  // ORDINARY mean-query -- for a LAGGED (Jacobi-forced or GaussSeidel-
  // declared) import, that is deliberately the PROVIDER's PRECEDING
  // sweep's delivery, one sweep behind the provider's own truly final
  // output, and reading it here would carry that same one-sweep lag
  // into the seed the driver's own analogous record never has (its
  // `history` commits member i's OWN fresh advance() output once the
  // whole sweep loop exits, always unlagged from i's own perspective).
  // `slots[]` is the live buffer every provider's own WindowAdvance
  // instruction just wrote its final sweep's output into; reading it
  // NOW, after every node in this region has already run its own last
  // sweep (this loop's exit, above), is the one point in the pass where
  // that read cannot race a provider that has not run yet -- unlike
  // inside advance_window() itself, where dispatch order within the
  // SAME sweep leaves no such guarantee.
  if (R.window_scoped) {
    for (auto& [node, cfg] : window_member_config) {
      if (cfg.region_index != R.index || !cfg.window_import.valid() ||
          cfg.required == nullptr) {
        continue;
      }
      const auto dst_it = window_last_dst_step.find(node);
      if (dst_it == window_last_dst_step.end()) continue;
      const Rational this_T0 = Rational(dst_it->second) * cfg.window_length;
      const Rational this_T1 = this_T0 + cfg.window_length;
      // `window_sample_times()`/`window_sample_shift()` (fuse/model/
      // coupling_graph.hpp): the SAME graph-only reconstruction
      // advance_window() uses for its own import waveform (this
      // function's own top comment) -- the provider's own buffer
      // index that holds its value AT `this_T1` is the last entry of
      // that list, shifted down by one slot whenever an earlier
      // window's own straddle carried the provider's real clock past
      // `this_T0` before this window's own sweep 0 ever ran.
      const std::size_t sample_count =
          window_sample_times(cfg.provider_interval, cfg.provider_policy,
                              Rational{0}, cfg.window_length,
                              dst_it->second)
              .size();
      const int shift = window_sample_shift(cfg.provider_interval,
                                            cfg.provider_policy,
                                            Rational{0}, this_T0);
      const std::size_t last_sample =
          sample_count - 1 - static_cast<std::size_t>(shift);
      const detail::Slot& import_slot = slots[cfg.window_import.value()];
      if (import_slot.data == nullptr) continue;
      detail::WindowMemberState& st_import = window_member_state[node];
      for (std::size_t e = 0; e < cfg.length; ++e) {
        const double v_boundary =
            import_slot.data[e * cfg.provider_capacity + last_sample];
        if (st_import.history.size() <= e) st_import.history.resize(e + 1);
        if (st_import.history[e].empty() ||
            st_import.history[e].back().first != this_T1) {
          st_import.history[e].push_back({this_T1, v_boundary});
        } else {
          st_import.history[e].back().second = v_boundary;
        }
      }
    }
  }
  co_return Result<void>::success();
}

// Mirrors WindowCoupler::resolve_pending_straddles() (window_coupler.cpp)
// on the Runner -- see that function's own comment for WHY a replay is
// needed at all (not just "carry the clock"): a straddling member's own
// state, at the end of the window that just finished, was advanced past
// that window's own end using whichever sweep's import snapshot was live
// at the time -- for a LAGGED import (Jacobi-forced, or GaussSeidel's own
// lagged direction), that is ONE SWEEP BEHIND the window's own FINAL,
// now-authoritative delivery, by design (this function's own sibling,
// advance_window()'s top comment). This replay redoes every substep
// since the member's own last checkpoint with that now-authoritative
// data, so the state `run_region()`'s own NEXT checkpoint (immediately
// after this call) captures is the one the driver's own single-sweep-
// lag-free `history` always had.
Result<void> Runner::Impl::resolve_pending_window_straddles(
    ComponentSet& components, detail::RegionRuntime& R) {
  for (const NodeId node : R.window_member_nodes) {
    detail::WindowMemberState& st = window_member_state[node.value()];
    if (!st.straddle_pending) continue;
    const WindowMemberConfig& cfg = window_member_config.at(node.value());
    Component& comp = components.at(cfg.component_slot);
    // KNOWN_ISSUES.md D12: this function is NOT a coroutine (see its own
    // declaration comment, runner_impl.hpp) -- its replay below calls
    // comp.restore_state() and comp.advance(ctx) directly, with no
    // ComponentSet-tick for a suspend to attach to cleanly. Refuse rather
    // than call an externally driven component's (no-op) advance() here,
    // out of any order fuse_begin_step()/fuse_end_step() could make sense
    // of -- this replay runs BEFORE run_region() takes this window
    // instance's own checkpoint (this function's own caller), so an
    // external caller has no step boundary that corresponds to it at all.
    if (comp.is_externally_driven()) {
      return Failure{
          "E-RT-WINDOW-EXTERNAL-STRADDLE",
          "iteration region '" + R.name + "': component '" + comp.name() +
              "' is driven from outside the Runner (Component::"
              "is_externally_driven()) and straddled a window boundary; "
              "replaying its carried-over substeps has no corresponding "
              "fuse_begin_step()/fuse_end_step() for an external caller to "
              "resume through, so this combination is refused rather than "
              "silently advancing the component out of order"};
    }
    // `straddle_window_start`: the CHECKPOINT's own time (this member's
    // own clock when it was taken, possibly already past the window's
    // own nominal start if an EARLIER straddle carried it there --
    // WindowCoupler's own `WindowMemberState::straddle_window_start`
    // field comment, mirrored exactly). `nominal_T0`/`prev_T1` are the
    // window's own boundaries (`prev_T1 - window_length` recovers
    // `nominal_T0` exactly, since `prev_T1` is always an integer
    // multiple of `window_length`) -- what this member's own IMPORT is
    // scoped to, regardless of where ITS OWN checkpoint sat.
    const Rational checkpoint_clock = st.straddle_window_start;
    const Rational prev_T1 = st.straddle_window_end;
    const Rational nominal_T0 = prev_T1 - cfg.window_length;
    const Rational target_clock = st.clock;

    const Result<void> r0 = comp.restore_state();
    if (!r0.ok()) return r0;

    // This member's own import over the window that just finished,
    // [nominal_T0, prev_T1) -- built exactly as advance_window() itself
    // builds it (the SAME `window_sample_times()`/`window_sample_
    // shift()` correspondence), but reading the LIVE buffer directly:
    // this is never a pass's first sweep, so there is no predictor to
    // apply, and nothing overwrites `slots[cfg.window_import.value()]`
    // between the window that just finished and this call.
    std::vector<Waveform> import_wf;
    if (cfg.window_import.valid() && cfg.required != nullptr) {
      const double* arr = slots[cfg.window_import.value()].data;
      const Rational idx_q = nominal_T0 / cfg.window_length;
      const std::int64_t prev_window_index = idx_q.num() / idx_q.den();
      const std::vector<Rational> times = window_sample_times(
          cfg.provider_interval, cfg.provider_policy, Rational{0},
          cfg.window_length, prev_window_index);
      const int shift = window_sample_shift(
          cfg.provider_interval, cfg.provider_policy, Rational{0}, nominal_T0);
      import_wf.reserve(cfg.length);
      for (std::size_t e = 0; e < cfg.length; ++e) {
        Waveform wf(1, cfg.provider_capacity, TemporalBasis::Linear, 1);
        wf.reset(nominal_T0, prev_T1);
        const bool have_history =
            e < st.history.size() && !st.history[e].empty();
        for (std::size_t k = 0; k < times.size(); ++k) {
          const double v = (shift == 1 && k == 0)
              ? (have_history ? st.history[e].back().second
                              : arr[e * cfg.provider_capacity + 0])
              : arr[e * cfg.provider_capacity +
                   (k - static_cast<std::size_t>(shift))];
          Result<void> p = wf.push(times[k], &v);
          if (!p.ok()) return p.error();
        }
        import_wf.push_back(std::move(wf));
      }
    }

    fuse::WindowMember member_view;
    member_view.component = &comp;
    member_view.provided = cfg.provided;
    std::vector<double> scratch_required;
    member_view.required = cfg.required != nullptr
        ? cfg.required
        : (scratch_required.assign(cfg.length, 0.0), scratch_required.data());
    member_view.length = cfg.length;

    // Replays every substep since this member's own last checkpoint
    // (`checkpoint_clock`, the value that checkpoint held), up to
    // `target_clock` (wherever the straddle left `st.clock`) -- at most
    // ONE chunk straddles `prev_T1` (plan_substeps()'s own contract: an
    // overshoot is always < this member's own interval), split exactly
    // there via the SAME shared detail::advance_straddle_split() the
    // original (now superseded) substep loop used.
    Rational clock = checkpoint_clock;
    while (clock < target_clock) {
      const Rational end = (clock + cfg.interval < target_clock)
                                ? clock + cfg.interval
                                : target_clock;
      const bool this_chunk_straddles = (clock < prev_T1 && prev_T1 < end);
      if (this_chunk_straddles) {
        std::vector<double> out_scratch(cfg.length);
        const Result<void> split_r = detail::advance_straddle_split(
            member_view, clock, prev_T1, end, 0, 0,
            [&](std::size_t e, Rational a, Rational b) -> Result<double> {
              if (import_wf.empty()) return member_view.required[e];
              const Result<double> v = detail::mean_over(import_wf[e], a, b);
              return v;
            },
            [&](std::size_t e) -> Result<double> {
              if (import_wf.empty()) return member_view.required[e];
              double v = 0.0;
              const Result<void> r = import_wf[e].evaluate(prev_T1, &v);
              if (!r.ok()) return member_view.required[e];
              return v;
            },
            [&](std::size_t e, Rational target) -> double {
              if (e >= st.history.size() || st.history[e].empty()) {
                return member_view.required[e];
              }
              const double v = fuse::extrapolate_boundary_value(
                  st.history[e], target, cfg.order);
              return v;
            },
            out_scratch.data());
        if (!split_r.ok()) return split_r.error();
      } else {
        for (std::size_t e = 0; e < cfg.length; ++e) {
          if (!import_wf.empty()) {
            const Result<double> mv = detail::mean_over(import_wf[e], clock, end);
            if (mv.ok()) member_view.required[e] = mv.value();
          }
        }
        StepContext ctx;
        ctx.t_begin = clock;
        ctx.t_end = end;
        ctx.step = 0;
        ctx.iteration = 0;
        const Result<void> ar = comp.advance(ctx);
        if (!ar.ok()) return ar;
      }
      clock = end;
    }
    st.straddle_pending = false;

    // Publish the real post-replay value into every OTHER member's own
    // compiled IMPORT arc, slot 0 -- not into THIS member's own export
    // arc (`ins.out`), which is a different arc, connected to the
    // consumer's `window_import` only by the field transfer's
    // Transport/Remap/Sink chain, and that chain has not run for THIS
    // window instance yet. The driver's equivalent is
    // `build_delivered_member()` reading `members_[i].provided[e]`
    // directly (window_coupler.cpp): the value a LAGGED consumer's
    // predictor needs at `window_sample_times()` index
    // `k == window_sample_shift(...)` on the NEXT window's sweep 0.
    //
    // KNOWN LIMITATION. Writing straight into the consumer's import arc
    // bypasses the field transfer, so it is exact only for an identity
    // transfer between components on the same processor set. That is
    // ENFORCED, not assumed: resolve_regions() rejects every other
    // configuration of a provider that can cross a window end
    // (E-RT-WINDOW-STRADDLE-REPUBLISH), and the loop below fails rather
    // than skipping if it ever meets one. Sending this value through the
    // field transfer itself, so that a remap or a separate processor set
    // is handled, is open. `previous[]`'s once-per-sweep snapshot
    // (run_region()'s k=0 top) reads `slots[]` strictly AFTER this
    // function, so it picks up this write.
    for (auto& [other_node, other_cfg] : window_member_config) {
      // Not a consumer of THIS provider at all -- a legitimate skip, not
      // a failure: most entries in this map are not.
      if (!other_cfg.import_lagged || !other_cfg.window_import.valid()) {
        continue;
      }
      if (other_cfg.provider_component_slot != cfg.component_slot) continue;

      // Past this point, `other_cfg` IS a lagged consumer of this exact
      // provider, and the setup-time check above (`E-RT-WINDOW-STRADDLE-
      // REPUBLISH`) already refused any such channel that is not
      // same-rank/same-region when the provider can straddle -- so
      // nothing below should legitimately fail. If it does anyway (the
      // arc id out of range, the slot never allocated, or sized for a
      // layout this write does not expect), that is exactly the silent-
      // wrong-answer case the setup check exists to rule out IN
      // ADVANCE; report it loudly here too rather than skip it quietly,
      // in case setup's own graph-level check ever missed a path that
      // reaches this write some other way.
      if (other_cfg.window_import.value() >= slots.size()) {
        return Failure{"E-RT-WINDOW-STRADDLE-REPUBLISH",
                     "resolve_pending_window_straddles(): a lagged "
                     "consumer's own window_import arc id is out of "
                     "range; this should have been caught at setup"};
      }
      detail::Slot& import_slot = slots[other_cfg.window_import.value()];
      if (import_slot.data == nullptr) {
        return Failure{"E-RT-WINDOW-STRADDLE-REPUBLISH",
                     "resolve_pending_window_straddles(): a lagged "
                     "consumer's own window_import arc is not allocated "
                     "on this process -- publishing the straddling "
                     "provider's real value here would silently leave "
                     "that process with a stale predictor input"};
      }
      if (import_slot.total() != other_cfg.length * other_cfg.provider_capacity) {
        return Failure{"E-RT-WINDOW-STRADDLE-REPUBLISH",
                     "resolve_pending_window_straddles(): a lagged "
                     "consumer's own window_import arc is not sized the "
                     "way this write expects (length * provider_capacity)"};
      }
      for (std::size_t e = 0; e < other_cfg.length && e < cfg.length; ++e) {
        import_slot.data[e * other_cfg.provider_capacity + 0] = cfg.provided[e];
      }
    }
  }
  return Result<void>::success();
}

// Time windows track (C3): the Jacobi/GaussSeidel design question the
// former stub above named is resolved -- see fuse::WaveformIteration
// (fuse/model/iteration.hpp) and effective_channel_lag() (fuse/model/
// coupling_graph.hpp). What decides "same-sweep-fresh vs previous-sweep"
// for THIS function is entirely mechanical, below: an UNLAGGED import
// reads the live `window_import` arc directly (correct automatically,
// because the provider's own WindowAdvance instruction and field-
// transfer chain are positioned strictly earlier in this SAME sweep's
// instruction stream -- window_lowering.cpp's own ordering, checked at
// schedule time by window_import_is_lagged()'s causality rule); a
// LAGGED import reads `Impl::previous[window_import]`, snapshotted once
// per sweep, BEFORE this sweep's body runs, by run_region() itself
// (R.window_lagged_import_arcs' own comment) -- so a lagged read can
// never see this sweep's own fresh delivery, regardless of where the
// delivering Sink sits in the stream. Neither reads anything from a
// partner's OWN WindowMemberState: every sample TIME below comes from
// `window_sample_times()`/`plan_substeps()` applied to (window_length,
// this member's or its import's PROVIDER's own interval/policy, window
// index) -- the graph alone (WindowMemberConfig's own comment) -- and
// every sample VALUE comes from the compiled dataflow arc the ordinary
// channel rule already built, or (the one exception, and still never a
// partner's live state) this member's OWN `st.history`, which already
// holds what arrived through a PAST field transfer. There is no side
// channel.
//
// A `DtPolicy::Fixed` member's own last substep crossing the window end
// (a "straddle") is handled two ways depending on whose substep it is.
// THIS member's own straddle is split in place, below, via `detail::
// advance_straddle_split()`, giving a provisional value the member's own
// state carries into the NEXT window instance; `resolve_pending_window_
// straddles()` (this file, called from `run_region()`, BEFORE this
// function ever runs for that next instance) then REPLAYS it with that
// window's own now-final data, following the SAME choreography
// `WindowCoupler::resolve_pending_straddles()` uses -- NOT YET VERIFIED
// bit-exact against it (test/runtime/window_runner_straddle_test.cpp's
// own *FixedBCrossing tests still disagree at this writing; see that
// file's own comment on the open gap). An IMPORT PROVIDER's straddle is
// handled by
// `window_sample_times()`/`window_sample_shift()` (fuse/model/coupling_
// graph.hpp) reconstructing the provider's own grid and buffer-index
// correspondence from the graph alone -- verified against the compiled
// buffer a straddling provider actually writes, and against a real
// carried-forward clock over a full period, for a provider whose own
// interval is smaller than the window (the only envelope this track
// names; KNOWN_ISSUES.md D7 -- TWO Fixed members straddling the SAME
// window -- remains refused at setup, unconditionally, identically to
// `WindowCoupler::create()`'s own `E-RT-WINDOW-STRADDLE`).
std::int64_t Runner::Impl::window_absolute_instance(const WindowMemberConfig& cfg,
                                                    std::int64_t dst_step) const {
  const Rational ratio = plan.period() / cfg.window_length;
  // Guaranteed exact by schedule()'s own E-SCHED-PERIOD check
  // (src/schedule/expand.cpp's tick_count, called on every window-scoped
  // channel's overridden src_dt/dst_dt): a window length that did not
  // divide the compiled period evenly was already refused before this
  // Runner (or any Runner) was ever constructed.
  FUSE_ASSERT(ratio.is_integer(),
             "window_absolute_instance: period {} is not an exact multiple of "
             "window_length {} -- schedule() should have refused this",
             plan.period().to_string(), cfg.window_length.to_string());
  return period_index * ratio.num() + dst_step;
}

detail::Task<Result<void>> Runner::Impl::advance_window(
    ComponentSet& components, const Instruction& ins,
    const detail::Resolved& /*r*/, int iteration) {
  const std::uint32_t node = ins.origin_node.value();
  const auto cfg_it = window_member_config.find(node);
  if (cfg_it == window_member_config.end()) {
    co_return Failure{"E-RT-WINDOW-UNPLACED",
                 "WindowAdvance instruction's own node has no window "
                 "member config; resolve_regions() should have built one"};
  }
  const WindowMemberConfig& cfg = cfg_it->second;
  detail::WindowMemberState& st = window_member_state[node];
  Component& comp = components.at(cfg.component_slot);

  const std::int64_t absolute_instance =
      window_absolute_instance(cfg, ins.dst_step);
  const Rational T0 = Rational(absolute_instance) * cfg.window_length;
  const Rational T1 = T0 + cfg.window_length;

  const auto last_it = window_last_dst_step.find(node);
  // Compared as the ABSOLUTE window instance, not `ins.dst_step` alone --
  // `window_last_dst_step`'s own declaration (runner_impl.hpp) has the
  // full account of why a dst_step-only comparison is wrong on the
  // second and later `run_period()` call.
  const bool new_instance =
      last_it == window_last_dst_step.end() || last_it->second != absolute_instance;
  window_last_dst_step[node] = absolute_instance;

  // `run_region()`'s own k-loop always starts a pass at k=0 (its own
  // "for (int k = 0; k < R.max_iterations; ++k)" is entirely local to
  // ONE call, i.e. one pass); `new_instance` (dst_step-based) must agree.
  // A mismatch means the compiled stream is calling this in a shape
  // resolve_regions() never anticipated.
  const bool first_sweep_of_pass = (iteration == 0);
  if (first_sweep_of_pass != new_instance) {
    co_return Failure{"E-RT-WINDOW-SWEEP",
                 "WindowAdvance: iteration==0 (" +
                     std::to_string(first_sweep_of_pass) +
                     ") disagrees with the dst_step-based first-sweep "
                     "detection (" + std::to_string(new_instance) +
                     ") for the same call"};
  }

  Rational pass_start_clock;
  if (first_sweep_of_pass) {
    // `st.clock` already sits wherever the last substep of the previous
    // window instance left it (past T0 only if that substep straddled);
    // a member that has never straddled starts fresh, exactly at this
    // window's own nominal start. Compared against `T0` directly, never
    // against `st.straddle_pending` -- `resolve_pending_window_
    // straddles()` (called from `run_region()`, BEFORE this function
    // ever runs for this pass) already clears that flag once its own
    // replay resolves it, but `st.clock` itself is untouched by that
    // replay (the replay redoes the COMPONENT's state, never this
    // bookkeeping). MEASURED (this session): switching this ternary's
    // condition from `st.straddle_pending` to `st.clock > T0` fixed a
    // real bug -- the OLD condition, now always false by the time this
    // runs (the replay clears it first), was silently collapsing every
    // post-straddle window back onto `T0`, one whole pass_start_clock
    // wrong -- confirmed by instrumenting this call and
    // resolve_pending_window_straddles() temporarily and comparing the
    // printed T0/T1/clock sequence against the hand-derived (13,7)
    // schedule (instrumentation since removed). This
    // fixes THAT bug specifically; it does not, by itself, make the
    // wider `*FixedBCrossing` comparisons in test/runtime/window_
    // runner_straddle_test.cpp bit-exact against the driver -- they
    // still disagree at this writing; see that file's own comment on
    // the open gap. Cached in `window_pass_start_clock`
    // (its own comment, runner_impl.hpp) so EVERY sweep of this pass
    // plans from the SAME clock, not from wherever the PREVIOUS
    // sweep's own advance left `st.clock`.
    pass_start_clock = st.clock > T0 ? st.clock : T0;
    st.straddle_pending = false;
    window_pass_start_clock[node] = pass_start_clock;
  } else {
    pass_start_clock = window_pass_start_clock.at(node);
  }

  // This sweep's import trajectory, one Waveform per element, built from
  // whichever buffer (live or shadow) this import's lag status reads --
  // this function's own top comment. The LAG is about which SWEEP's
  // snapshot to read, not which window: `previous[]` holds THIS SAME
  // window's own previous-sweep delivery on every sweep but the pass's
  // first, where (there being no earlier sweep of this window yet) it
  // still holds the PREVIOUS window instance's own final delivery --
  // exactly the window-to-window seed Jacobi/GaussSeidel's lagged
  // direction needs at a window boundary.
  std::vector<Waveform>            import_wf;
  Rational import_t0{0}, import_t1{0};
  // fuse/model/iteration.hpp's own WaveformIteration::Jacobi docstring:
  // every field transfer in a Jacobi window is loop-carried whatever
  // Channel::lag the user wrote -- `cfg.import_lagged` alone (the
  // graph's own declared lag, binding only under GaussSeidel) is not
  // enough here; resolve_regions()'s matching fix to
  // R.window_lagged_import_arcs is what keeps `previous[]` correctly
  // populated for this read. `cfg.import_lagged` (below) already covers
  // the Jacobi case on its own: `effective_channel_lag()`
  // (fuse/model/coupling_graph.hpp) forces lag=1 for every channel in a
  // Jacobi window regardless of the graph's own `.lag()` declaration
  // (its own `is_jacobi_window_channel()` helper), so `cfg.import_lagged`
  // is already true here for BOTH directions in this file's graphs --
  // confirmed by instrumented probing this session, not assumed.
  if (cfg.window_import.valid() && cfg.required != nullptr) {
    // On the first sweep of a NEW window instance, a LAGGED import has
    // nothing genuine to read yet (the provider has not run THIS
    // window): the driver's own documented predictor
    // (window_coupler.hpp's `WindowCoupler::sweep_window()` comment,
    // "Seed y_fed ... from the driver's documented predictor:
    // extrapolate_boundary_value() ... at least second order") is what
    // must be reproduced here instead of a raw buffer read.
    // `cfg.import_lagged` alone already selects the lagged case (this
    // function's own top comment: forced true, both directions, under
    // Jacobi; declared true for GaussSeidel's lagged direction only).
    const bool use_predictor = cfg.import_lagged && first_sweep_of_pass;
    const double* arr = cfg.import_lagged
                            ? previous[cfg.window_import.value()].data()
                            : slots[cfg.window_import.value()].data;
    // `import_t0 = T0` unconditionally (no "back one window" shift):
    // the driver's own predictor, above, already needs no raw read at
    // all for the first-sweep-of-a-lagged-pass case, and every other
    // sweep already used T0.
    import_t0 = T0;
    import_t1 = import_t0 + cfg.window_length;

    // `window_sample_times()` (fuse/model/coupling_graph.hpp) names the
    // exact times this window's import carries, from the graph alone
    // (the provider's own interval/policy, the window's own length and
    // index) -- no runtime state, so a CONSUMER can compute it without
    // ever reading the provider's own `WindowMemberState`. Verified
    // against a real carried-forward `DtPolicy::Fixed` grid over a full
    // period (test/model/coupling_graph_test.cpp,
    // WindowSampleTimesAgreesWithTheRealCarriedForwardGrid) and, below,
    // against the actual compiled buffer this function's own provider
    // writes (test/runtime/window_runner_straddle_test.cpp).
    //
    // `window_sample_shift()` names how many of the provider's own
    // buffer slots this list's own LEADING (`T0`) entry has already
    // been carried past by an EARLIER window's own straddle -- also
    // graph alone (this window's own `T0`, checked for exact
    // divisibility by the provider's own interval). `shift == 0`: every
    // entry maps 1:1 to the SAME buffer index. `shift == 1`: entry 0
    // (`T0` itself) has no buffer slot of its own -- the provider's
    // real clock, by the time THIS window's sweep 0 runs, already sits
    // at the next native grid point past `T0` (an earlier straddle
    // carried it there), so its buffer's own index 0 holds THAT later
    // value, not the one at `T0` -- and every entry from 1 on maps to
    // buffer index `k - 1` instead of `k`.
    //
    // `T0`'s own value is never lost when `shift == 1`: it is exactly
    // what THIS consumer's own `st.history` already recorded when the
    // provider's straddle resolved at the end of the PREVIOUS window
    // (this function's own history-update code, below) -- read from
    // there, never reconstructed from the provider's own buffer.
    const std::vector<Rational> times = window_sample_times(
        cfg.provider_interval, cfg.provider_policy, Rational{0},
        cfg.window_length, absolute_instance);
    const int shift = window_sample_shift(
        cfg.provider_interval, cfg.provider_policy, Rational{0}, T0);

    import_wf.reserve(cfg.length);
    for (std::size_t e = 0; e < cfg.length; ++e) {
      Waveform wf(1, cfg.provider_capacity, TemporalBasis::Linear, 1);
      wf.reset(import_t0, import_t1);
      // `st.history` is THIS member's own record of what arrived
      // through the field transfer at every earlier window's own end
      // (resolve_regions()'s cold-start seeding comment, and this
      // function's own history update below) -- never a partner's
      // state, and always non-empty from setup onward (seeded with the
      // provider's registered initial value at t=0). What it holds IS
      // the provider's own committed trajectory at those times, sample
      // for sample, whether read directly (the driver's own `state_
      // [i].history`) or through the field transfer (here) -- the two
      // coincide exactly for an identity remap, which is what every
      // caller of this function uses today.
      const bool have_history = e < st.history.size() && !st.history[e].empty();
      for (std::size_t k = 0; k < times.size(); ++k) {
        double v;
        if (use_predictor) {
          // The driver's own predictor (window_coupler.hpp's
          // `WindowCoupler::sweep_window()` comment) never
          // extrapolates the window-START sample either: `times[0]`
          // is always `T0`, and build_delivered_member() uses the
          // provider's REAL current value there (`start_value[i][e]`,
          // captured by save_state() right before the sweep), never
          // `extrapolate_boundary_value()` -- reading THIS member's
          // own history exactly, not through a polynomial fit at one
          // of its own knots (not guaranteed bit-exact there), is what
          // keeps this sample identical to the driver's.
          //
          // `times[k]` for `k >= 1` is extrapolated UNLESS it is this
          // provider's own first entry with a buffer slot of its own
          // (`k == shift`, `window_sample_shift()`'s own return
          // value): the driver's `build_delivered_member()` inserts a
          // SECOND real (non-extrapolated) knot there too --
          // `start_value[i][e]`, the provider's own live state at its
          // own `start_clock[i]`, which is exactly `times[shift]` when
          // `shift == 1` (an earlier straddle already carried the
          // provider's own clock there). `resolve_pending_window_
          // straddles()` (this function's own sibling) publishes that
          // SAME real value into this provider's own export, buffer
          // index 0, the moment its straddle resolves -- before this
          // pass's sweep 0 ever starts -- so `arr[...+0]` already
          // holds it here. (An earlier version of this function
          // extrapolated `k == shift` too, on the mistaken claim that
          // doing so was "robust to shift ... since neither branch
          // reads the raw buffer" -- it is not: the driver's own
          // extra real knot has no extrapolated counterpart, and nothing
          // published it to the Runner's own buffer until the fix
          // above.) Every entry PAST that one (`k > shift`) still has
          // nothing real behind it (the provider has not run ITS OWN
          // substeps for this window yet), so extrapolation remains
          // the only option there.
          if (k == 0) {
            v = have_history ? st.history[e].back().second
                              : arr[e * cfg.provider_capacity + 0];
          } else if (shift == 1 && k == static_cast<std::size_t>(shift)) {
            v = arr[e * cfg.provider_capacity + 0];
          } else {
            v = have_history
                    ? fuse::extrapolate_boundary_value(st.history[e],
                                                        times[k], cfg.order)
                    : arr[e * cfg.provider_capacity + 0];
          }
        } else if (shift == 1 && k == 0) {
          // This entry's value has no buffer slot of its own: the
          // provider's straddle carried its real clock past T0 before
          // this window's own sweep 0 ever ran. The consumer's own
          // history already holds it exactly (recorded when that
          // straddle resolved at the end of the PREVIOUS window).
          v = have_history ? st.history[e].back().second
                            : arr[e * cfg.provider_capacity + 0];
        } else {
          v = arr[e * cfg.provider_capacity +
                 (k - static_cast<std::size_t>(shift))];
        }
        Result<void> p = wf.push(times[k], &v);
        if (!p.ok()) co_return p.error();
      }
      import_wf.push_back(std::move(wf));
    }
  }

  auto planned_self =
      detail::plan_substeps(pass_start_clock, cfg.interval, T1, cfg.policy);
  const std::vector<detail::WindowSubstep>& steps = planned_self.first;
  const Rational final_clock = planned_self.second;

  if (steps.size() + 1 > cfg.capacity) {
    co_return Failure{"E-RT-WINDOW-CAPACITY",
                 "window member needs " + std::to_string(steps.size() + 1) +
                     " samples this window, more than its own compiled "
                     "capacity (" + std::to_string(cfg.capacity) +
                     "); window_sample_capacity() should have sized the "
                     "field wide enough for this not to happen"};
  }

  double* out = cfg.provided != nullptr ? slots[ins.out.value()].data : nullptr;

  // Sample 0: the window-start value. NEVER `cfg.provided[e]` read live --
  // `Component::restore_state()` is only contractually required to
  // restore what `save_state()` captured (the component's own hidden
  // state); a `Provides` field that is itself a function of that hidden
  // state (e.g. a remap of it) is not guaranteed to be resynchronized by
  // restore_state() alone, and run_region()'s own generic rollback never
  // touches it directly. Reading `cfg.provided[e]` here on the second and
  // later sweeps of a pass would then mislabel whatever the PREVIOUS
  // (rejected) sweep's own last `advance()` call left behind as the
  // window-start value -- exactly the defect `WindowCoupler::
  // sweep_window()`'s own `start_value[i][e]` (window_coupler.cpp's own
  // comment) was fixed to remove on the driver side. Cached once, on
  // this pass's own first sweep, mirroring `window_pass_start_clock`'s
  // own pattern (this struct's own comment) for exactly the same reason.
  std::vector<double>& start_value = window_start_value[node];
  if (first_sweep_of_pass) {
    start_value.assign(cfg.provided, cfg.provided + cfg.length);
  }
  if (out != nullptr) {
    for (std::size_t e = 0; e < cfg.length; ++e) {
      out[e * cfg.capacity + 0] = start_value[e];
    }
  }

  // `import_t0`'s own shift (this function's own top comment): on the
  // FIRST sweep of a pass, a LAGGED import's waveform lives on the
  // PREVIOUS window's own absolute time axis (`[T0-window_length,
  // T1-window_length)`), while `steps[]` (THIS member's own substeps)
  // lives on THIS window's (`[T0, T1)`). Querying `steps[s]`'s own
  // `[t0, t1)` against `import_wf` directly, with no compensating
  // shift, intersects two ranges that are `window_length` apart and are
  // NEVER simultaneously within both members' own domains except at one
  // degenerate point -- every substep's own mean query silently no-ops
  // (`b > a` false), leaving `cfg.required[e]` at whatever it already
  // held. `query_shift` re-aligns the substep's own query onto the
  // SAME axis `import_wf` is built on (zero when there is no shift to
  // undo, i.e. every sweep but a lagged import's first).
  const Rational query_shift = import_t0 - T0;
  // A null `cfg.required` (a pure exporter -- no Requires field in this
  // region) gives `detail::advance_straddle_split()`'s own unconditional
  // `member.required[e] = ...` nowhere safe to write if this member's own
  // last substep straddles below; point it at throwaway scratch storage
  // instead of `nullptr` in that case. The component's own `advance()`
  // never reads a field it does not declare, so writing into scratch
  // rather than the real (absent) buffer changes nothing it can observe.
  std::vector<double> scratch_required;
  double* const required_for_split = cfg.required != nullptr
                                          ? cfg.required
                                          : (scratch_required.assign(
                                                 cfg.length, 0.0),
                                             scratch_required.data());
  for (std::size_t s = 0; s < steps.size(); ++s) {
    const detail::WindowSubstep& sub = steps[s];
    const bool is_last_straddle = (s + 1 == steps.size()) && sub.straddle;
    if (is_last_straddle) {
      // This member's OWN last substep crosses T1 (DtPolicy::Fixed, this
      // member's own interval not dividing the window). Split the
      // advance() exactly at T1 -- the SAME two-phase choreography
      // WindowCoupler::sweep_window()'s own inline handling uses, via
      // the SAME shared detail::advance_straddle_split() -- instead of
      // one call over the whole straddling chunk with a single blended
      // mean: the committed/exchanged sample (written to `out` below)
      // is the value AT T1, from the in-window mean over [sub.t0, T1)
      // (the SAME formula every other substep already uses, just
      // clamped to this window's own end); this member's OWN state then
      // continues past T1 to sub.t1 using a blended boundary (import_wf
      // evaluated exactly at T1, still inside its own known domain --
      // `boundary_at_split` below) and extrapolated tail
      // (`st.history`, this member's own record of what it received at
      // every PRIOR window's own end, seeded at setup -- resolve_
      // regions()'s own comment -- never a partner's state) so it stays
      // on its own native clock across the window boundary.
      fuse::WindowMember member_view;
      member_view.component = &comp;
      member_view.required = required_for_split;
      member_view.provided = cfg.provided;
      member_view.length = cfg.length;

      std::vector<double> v_at_t1(cfg.length);
      const Result<void> split_r = detail::advance_straddle_split(
          member_view, sub.t0, T1, sub.t1,
          static_cast<std::size_t>(absolute_instance), iteration,
          [&](std::size_t e, Rational a, Rational b) -> Result<double> {
            if (import_wf.empty()) return required_for_split[e];
            const Rational qa = a + query_shift;
            const Rational qb = b + query_shift;
            const Rational ca = qa < import_t0 ? import_t0 : qa;
            const Rational cb = qb > import_t1 ? import_t1 : qb;
            if (cb <= ca) return required_for_split[e];
            const Result<double> v = detail::mean_over(import_wf[e], ca, cb);
            return v;
          },
          [&](std::size_t e) -> Result<double> {
            if (import_wf.empty()) return required_for_split[e];
            double v = 0.0;
            const Result<void> r = import_wf[e].evaluate(import_t1, &v);
            if (!r.ok()) return required_for_split[e];
            return v;
          },
          [&](std::size_t e, Rational target) -> double {
            if (e >= st.history.size() || st.history[e].empty()) {
              return required_for_split[e];
            }
            const double v = fuse::extrapolate_boundary_value(
                st.history[e], target, cfg.order);
            return v;
          },
          v_at_t1.data());
      if (!split_r.ok()) co_return split_r.error();
      if (out != nullptr) {
        for (std::size_t e = 0; e < cfg.length; ++e) {
          out[e * cfg.capacity + (s + 1)] = v_at_t1[e];
        }
      }
      continue;
    }

    if (cfg.required != nullptr && !import_wf.empty()) {
      for (std::size_t e = 0; e < cfg.length; ++e) {
        const Rational q0 = sub.t0 + query_shift;
        const Rational q1 = sub.t1 + query_shift;
        const Rational a = q0 < import_t0 ? import_t0 : q0;
        const Rational b = q1 > import_t1 ? import_t1 : q1;
        if (b > a) {
          const Result<double> mv = detail::mean_over(import_wf[e], a, b);
          if (mv.ok()) cfg.required[e] = mv.value();
        }
      }
    }
    StepContext ctx;
    ctx.t_begin = sub.t0;
    ctx.t_end = sub.t1;
    // Matches `WindowCoupler::sweep_window()`'s own convention exactly
    // (`ctx.step = window_index_`, src/runtime/window_coupler.cpp): the
    // ABSOLUTE window instance count, never `ins.dst_step` alone (period-
    // relative; `window_absolute_instance()`'s own comment has the full
    // account).
    ctx.step = absolute_instance;
    ctx.iteration = iteration;
    // KNOWN_ISSUES.md D12: same rule as Runner::Impl::advance() (runner.cpp)
    // -- suspend here, before the call, for the one component override
    // that is_externally_driven(); every real (in-process) window member
    // takes the branch never, so the unconditional `comp.advance(ctx)`
    // immediately below runs at exactly the point and order it always has.
    if (comp.is_externally_driven()) {
      ExternalAdvanceRequest req;
      req.component = comp.name();
      req.ctx = ctx;
      if (cfg.component_slot < slot_just_checkpointed.size()) {
        req.requires_checkpoint = slot_just_checkpointed[cfg.component_slot] != 0;
        slot_just_checkpointed[cfg.component_slot] = 0;
      }
      if (cfg.component_slot < slot_just_restored.size()) {
        req.requires_restore = slot_just_restored[cfg.component_slot] != 0;
        slot_just_restored[cfg.component_slot] = 0;
      }
      pending_external_request = std::move(req);
      co_await detail::ExternalSuspend{&active_handle};
    }
    const Result<void> ar = comp.advance(ctx);
    if (!ar.ok()) co_return ar;
    if (out != nullptr) {
      for (std::size_t e = 0; e < cfg.length; ++e) {
        out[e * cfg.capacity + (s + 1)] = cfg.provided[e];
      }
    }
  }

  // Every slot beyond the valid count is never read (this function's own
  // top comment / WindowMemberConfig::capacity) but is zeroed rather than
  // left as whatever a PREVIOUS, possibly-wider window instance wrote --
  // a stale nonzero tail is silently-wrong-looking state for anyone
  // inspecting the raw buffer, even though nothing in this design reads it.
  if (out != nullptr) {
    for (std::size_t s2 = steps.size() + 1; s2 < cfg.capacity; ++s2) {
      for (std::size_t e = 0; e < cfg.length; ++e) out[e * cfg.capacity + s2] = 0.0;
    }
  }

  // This member's own record of what it received, for a LATER window's
  // own straddle-tail `extrapolate_tail` (this function's own top
  // comment, resolve_regions()'s seeding comment), is intentionally NOT
  // touched here, mid-sweep. (An earlier version of this function DID
  // push/refine `st.history[e]` right here, every sweep, from
  // `import_wf[e]` -- measured, this session, to be the cause of a
  // real divergence from the driver: THIS member's own self-straddle
  // split, a few lines above in THIS SAME call, reads `st.history[e]`
  // for ITS OWN `extrapolate_tail`, and a mid-pass refinement changes
  // what that read sees sweep over sweep, sweep 1 onward, within the
  // SAME window -- something the driver's own `state_[i].history` can
  // never do, because `WindowCoupler::run_window()` commits it exactly
  // ONCE, strictly after its own sweep loop converges (window_coupler.
  // cpp's "Commit" comment). `run_region()`'s own post-convergence
  // refresh, below this function in the file (its own comment: "every
  // window-import node's own `st.history` ... is refreshed here from
  // the LIVE `window_import` arc, never `previous[]`"), already writes
  // the SAME field, correctly, exactly once, after this region's own
  // k-loop exits -- making the per-sweep write here not just wrong but
  // redundant.)

  st.clock = final_clock;
  // A straddling last substep (DtPolicy::Fixed, this member's own
  // interval not dividing the window) leaves `st.clock` past T1: the
  // NEXT window instance's own `pass_start_clock` (this function's top)
  // must resume from there, not from that window's own T0, and its
  // straddle-tail (the split-advance above, run again for whichever of
  // ITS OWN substeps crosses ITS window's end) reads `st.history` for
  // points before ITS window -- already covers this member continuing
  // to straddle every window, not just once.
  if (!steps.empty() && steps.back().straddle) {
    st.straddle_pending = true;
    // `pass_start_clock`, NOT `T0`: WindowCoupler's own field comment
    // (`WindowMemberState::straddle_window_start`, window_coupler.hpp)
    // names this "checkpoint restore_state() returns to" -- the SAME
    // `start_clock[i]` (window_coupler.cpp's own sweep_window()) it
    // assigns there, captured at THIS window's own top, BEFORE any of
    // its own substeps ran. When this member already carried a straddle
    // INTO this window, that checkpoint sits at wherever the carry left
    // it, past `T0` -- not at `T0` itself. MEASURED (this session): the
    // `T0` version walked the WRONG number of chunks in `resolve_
    // pending_window_straddles()`'s own replay (an extra, sub-interval
    // leftover chunk at the tail, from clipping a walk that started too
    // early against `target_clock`) -- confirmed by instrumenting the
    // replay's own chunk loop temporarily and comparing its printed
    // trace against the hand-derived (13,7) schedule (instrumentation
    // since removed).
    st.straddle_window_start = pass_start_clock;
    st.straddle_window_end = T1;
  }

  co_return Result<void>::success();
}

}  // namespace fuse
