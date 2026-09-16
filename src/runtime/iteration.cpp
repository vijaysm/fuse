#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

Result<void> Runner::Impl::resolve_regions(ComponentSet& components) {
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

    // --- one contiguous pass over the body, and only one -------------------
    std::size_t enters = 0;
    std::size_t exits = 0;
    for (std::size_t i = 0; i < stream.size(); ++i) {
      if (stream[i].region_enter == static_cast<std::int32_t>(ri)) {
        R.enter = i;
        ++enters;
      }
      if (stream[i].region_exit == static_cast<std::int32_t>(ri)) {
        R.exit = i;
        ++exits;
      }
    }
    if (enters != 1 || exits != 1 || R.exit < R.enter) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name + "' appears as " +
                       std::to_string(enters) + " enter and " +
                       std::to_string(exits) +
                       " exit boundaries in one period; exactly one pass over "
                       "the body is required. A body that appears more than "
                       "once per period is several independent fixed-point "
                       "problems sharing one checkpoint, which this runner "
                       "will not guess at: split it into one region per clock"};
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

      // The delivery. Without a Sink the accelerated iterate never reaches a
      // component, so every sweep would re-advance from identical inputs and
      // the residual would sit still until max_iterations ran out.
      std::size_t sinks = 0;
      for (std::size_t i = 0; i < stream.size(); ++i) {
        if (stream[i].kind != OpKind::Sink || stream[i].n_in == 0) continue;
        if (stream[i].in[0].value() != b.value_arc) continue;
        b.sink_instruction = i;
        b.feed_arc = stream[i].out.value();
        ++sinks;
      }
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
      if (b.length == 0) {
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
    const Rational enter_time = stream[R.enter].time;
    for (std::size_t t = 0; t < ticks.size(); ++t) {
      const bool mine =
          std::find(R.component_slots.begin(), R.component_slots.end(),
                    ticks[t].slot) != R.component_slots.end();
      if (!mine) continue;
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
      if (!(ticks[t].time < enter_time)) {
        return Failure{"E-RT-REGION",
                     "component '" + components.at(ticks[t].slot).name() +
                         "' is checkpointed by iteration region '" + R.name +
                         "' but has an advance at t=" +
                         ticks[t].time.to_string() +
                         ", at or after the region begins at t=" +
                         enter_time.to_string() +
                         ". That advance would run once, outside the loop, "
                         "from state the region had already iterated"};
      }
      R.tick_indices.push_back(t);
    }
    if (R.tick_indices.empty()) {
      return Failure{"E-RT-REGION",
                   "iteration region '" + R.name +
                       "' re-advances no component, so a sweep would recompute "
                       "the same values from the same inputs for ever"};
    }

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
  for (std::size_t i = 0; i < stream.size(); ++i) {
    if (i >= R.enter && i <= R.exit) continue;
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
    for (const std::size_t slot : R.component_slots) {
      const Result<void> s = components.at(slot).save_state();
      if (!s.ok()) return s;
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

Result<void> Runner::Impl::run_region(ComponentSet& components,
                                      detail::RegionRuntime& R) {
  IterationStats& st = region_stats[R.index];
  st.iterations = 0;
  st.converged = false;
  st.residual_history.clear();  // keeps the capacity reserved at setup

  const std::vector<Instruction>& stream = plan.instructions();

  R.accel->reset();

  // y_0: what the components were ACTUALLY advanced with. Reading the Sink
  // outputs -- the components' own Requires arrays -- rather than recording
  // what we last wrote means the residual measures the real map, including
  // anything the delivery did on the way.
  for (const detail::RegionBlock& b : R.blocks) {
    num::copy(slots[b.feed_arc].data, R.x_old.data() + b.offset, b.length);
  }

  for (int k = 0; k < R.max_iterations; ++k) {
    for (std::size_t i = R.enter; i <= R.exit; ++i) {
      const Result<void> e = execute(stream[i], resolved[i]);
      if (!e.ok()) return e;
      ++stats.instructions_executed;
    }
    ++st.iterations;
    ++stats.total_iterations;

    // G(y), and the residual G(y) - y.
    double total = 0.0;
    for (std::size_t bi = 0; bi < R.blocks.size(); ++bi) {
      const detail::RegionBlock& b = R.blocks[bi];
      const double* g = slots[b.value_arc].data;
      double rs = 0.0;
      double vs = 0.0;
      for (std::size_t j = 0; j < b.length; ++j) {
        const double v = g[j];
        const double d = v - R.x_old[b.offset + j];
        R.x_new[b.offset + j] = v;
        rs += d * d;
        vs += v * v;
      }
      R.block_residual[bi] = std::sqrt(rs);
      R.block_value[bi] = std::sqrt(vs);
      if (k == 0) R.block_first[bi] = R.block_residual[bi];
      total += rs;
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
      return a.error();
    }

    // Roll the components back. The loop-carried coupler buffers are NOT rolled
    // back; see the long note in resolve_regions(). Restoring first and
    // delivering second is the order that matters here: restore_state() puts
    // the component's Requires arrays back to the checkpoint, so delivering
    // before it would be undone.
    for (const std::size_t slot : R.component_slots) {
      const Result<void> rs = components.at(slot).restore_state();
      if (!rs.ok()) return rs;
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
      return Failure{
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
      const Result<void> e = execute(stream[b.sink_instruction],
                                     resolved[b.sink_instruction]);
      if (!e.ok()) return e;
      ++stats.instructions_executed;
    }

    // Re-advance, over the SAME interval, with ctx.iteration = k + 1.
    for (const std::size_t t : R.tick_indices) {
      const Result<void> adv = advance(components, ticks[t], k + 1);
      if (!adv.ok()) return adv;
    }
    R.x_old.swap(R.x_out);
  }

  if (!st.converged) {
    ++stats.windows_not_converged;
    if (config.abort_on_nonconvergence) {
      return Failure{"E-RT-NOCONVERGE",
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
  return Result<void>::success();
}

}  // namespace fuse
