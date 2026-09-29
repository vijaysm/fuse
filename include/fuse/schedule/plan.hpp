#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/ir/dag.hpp"
#include "fuse/schedule/acyclicity.hpp"
#include "fuse/schedule/buffers.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/expand.hpp"

namespace fuse {

/// One entry of the flat instruction stream. Three inputs covers every
/// primitive in spec §5.1; Merge with more than three sources is lowered as a
/// left-fold of pairwise merges. A node that arrives here with more inputs
/// than the array holds is rejected by `schedule()` rather than truncated --
/// a truncated instruction reads the wrong operands and computes plausible
/// wrong numbers.
struct Instruction {
  OpKind                kind = OpKind::Source;
  ArcId                 out;
  std::array<ArcId, 3>  in{};
  std::uint8_t          n_in = 0;
  OpAttrs               attrs;
  Rational              time{0};
  /// Which step of the channel's source and target clocks this instance
  /// belongs to, copied verbatim from `TimedNode` (see expand.hpp for the
  /// exact convention). Both are always filled in.
  ///
  /// Carried into the instruction stream for Phase 6 Task 6.3: a space-time
  /// remap's temporal operator `T` is indexed by (target step, source step),
  /// so the interpreter has to know which pair each instance stands for. It is
  /// deliberately NOT printed by `ExecutionPlan::to_string()` -- that dump is
  /// byte-compared by `ExecutionPlan.DumpIsByteExact`, and the step indices are
  /// already recoverable from `time` and the component intervals.
  std::int64_t          src_step = 0;
  std::int64_t          dst_step = 0;
  /// Where this instruction sits relative to the component ticks that begin at
  /// `time`, copied verbatim from `TimedNode` (see expand.hpp). It completes
  /// the timestamp: `time` alone cannot say whether a value delivered at `t` is
  /// seen by the step that begins at `t`, and `Runner::Impl::step_period` reads
  /// this to decide.
  ///
  /// UNLIKE the step indices, this IS printed by `ExecutionPlan::to_string()`
  /// (as `after-ticks`, only on the instructions that carry it). It has to be:
  /// it is not recoverable from the other printed fields. It was, briefly,
  /// while `AfterTicks` meant exactly `Sample` -- but the causality contract in
  /// expand.hpp also puts a target-limited slab instance after the ticks, and
  /// which slabs those are depends on `src_step` and the channel's source
  /// interval, neither of which the dump shows. Omitting it would let two plans
  /// that run differently dump identically, which is the one thing a
  /// byte-compared artifact must not do.
  TickPhase             phase = TickPhase::BeforeTicks;
  /// Iteration-region index (spec §5.3), or -1. `region_enter` is set on the
  /// first instruction of one pass over a region's body and `region_exit` on
  /// the last, so the interpreter has explicit loop boundaries and never has
  /// to reconstruct them from node membership. `schedule()` refuses to build
  /// a plan whose regions cannot be expressed this way (see E-SCHED-REGION).
  std::int32_t          region_enter = -1;
  std::int32_t          region_exit = -1;
  NodeId                origin_node;
};

namespace detail {
/// Time windows track: the ordering proof for the subcycling node
/// (`OpKind::WindowAdvance`). Exposed here, in `detail`, rather than kept
/// anonymous in src/schedule/scheduler.cpp (where `schedule()` calls it),
/// so a test can exercise it directly against a hand-built instruction
/// stream -- the same reason `detail::RegionBlock` (runner_impl.hpp) is
/// visible outside its own translation unit. See the definition (its own
/// doc comment, src/schedule/scheduler.cpp) for the exact invariant and
/// why it exists: `OpAttrs::window_import` (fuse/ir/op.hpp) is deliberately
/// not a real `Node::inputs` edge, so `check_dependence_order()` cannot
/// see it, and this is what checks it instead.
[[nodiscard]] Result<void> check_window_causality(
    const OperatorDAG& dag, const std::vector<Instruction>& instructions);
}  // namespace detail

/// Spec §2 layer 3: immutable after construction. The hot loop interprets this
/// vector -- no virtual dispatch per DOF, no map lookup, no allocation.
///
/// The instruction order is a *dependence* order, not merely a time order:
/// `expand()` emits at three different time conventions (chain nodes at the
/// slab end, `Sample` at the source tick start, reductions at the target step
/// end), so ties at equal times are the normal case and "sorted by time" alone
/// proves nothing. `schedule()` orders by (time, `TickPhase`, topological rank
/// of the DAG node, NodeId) -- the rank, not the node id, is what breaks a tie
/// among instructions of one phase -- and then verifies against the SSA arcs
/// that every operand is written before it is read.
/// `CycleReport::emission_order_valid` is deliberately not used for this; see
/// `schedule()`.
class ExecutionPlan {
 public:
  ExecutionPlan() = default;
  ExecutionPlan(std::vector<Instruction> instructions, BufferPlan buffers,
                Rational period, bool lockstep, std::size_t region_count)
      : instructions_(std::move(instructions)),
        buffers_(std::move(buffers)),
        period_(period),
        lockstep_(lockstep),
        region_count_(region_count) {}

  [[nodiscard]] const std::vector<Instruction>& instructions() const noexcept {
    return instructions_;
  }
  [[nodiscard]] const BufferPlan& buffers() const noexcept { return buffers_; }
  [[nodiscard]] Rational period() const noexcept { return period_; }
  [[nodiscard]] bool lockstep() const noexcept { return lockstep_; }
  [[nodiscard]] std::size_t region_count() const noexcept { return region_count_; }

  /// The --fuse-dump-plan artifact (spec §2.3). Stable and deterministic: the
  /// stream is imbued with std::locale::classic(), so neither a grouping
  /// thousands separator nor a comma decimal point from the host
  /// application's global locale can reach a golden file.
  [[nodiscard]] std::string to_string() const;

 private:
  std::vector<Instruction> instructions_;
  BufferPlan               buffers_;
  Rational                 period_{1};
  bool                     lockstep_ = false;
  std::size_t              region_count_ = 0;
};

struct ScheduleResult {
  ExecutionPlan plan;
  Diagnostics   diags;
  [[nodiscard]] bool ok() const { return !diags.has_errors(); }
};

/// expand -> prove_acyclic -> plan_buffers -> emit.
///
/// Never throws: every downstream step that can throw (`plan_buffers` throws
/// std::invalid_argument / std::out_of_range / std::overflow_error by
/// contract, and `OperatorDAG`'s accessors are `.at()`-checked) is caught and
/// reported as a diagnostic instead. An init-time setup error belongs in
/// `diags`, which is the channel the caller already has to read.
///
/// **Nothing is returned in `plan` unless every gate passed.** `ok() == false`
/// always comes with a default-constructed, empty `ExecutionPlan`; there is no
/// half-built plan for a caller to mistake for a usable one.
///
/// Error codes:
///   E-SCHED-PERIOD / -EXPLODE / -EXPAND / -INTERVAL  forwarded from expand()
///   E-SCHED-CYCLE    spec §8.2's hard gate: the time-resolved schedule for
///                    one period has a dependence cycle no lag breaks
///   E-SCHED-ORDER    the emitted stream reads an operand no earlier
///                    instruction wrote. A scheduler defect, not a user error,
///                    but emitting the stream anyway would hand the runtime
///                    instructions that read what nobody has written yet
///
/// `CycleReport::emission_order_valid` is deliberately **not** consulted.
/// `prove_acyclic()` derives it from a dependence graph that over-approximates
/// on purpose (`join_all()` reports the extremal endpoints of a join, which
/// are not a real producer/consumer pair), and a sweep of 26,647 generated
/// graphs measured about 8% spurious violations on clean multirate schedules
/// and zero real ones. Failing on it would reject roughly one legal multirate
/// schedule in twelve -- exactly the configurations spec §8 exists to support.
/// Ordering is derived from the SSA dependence relation instead and verified
/// against it, which has no such failure mode.
///   E-SCHED-ARITY    an operator with more inputs than `Instruction` holds
///   E-SCHED-BUFFERS  plan_buffers rejected the DAG or the region sizes
///   E-SCHED-REGION   an iteration region's body does not appear in the
///                    instruction stream as contiguous whole-body passes, so
///                    it cannot be expressed as flat enter/exit boundaries
///   E-SCHED-PLAN     the expansion or the buffer plan does not describe `dag`
///
/// `mode` is forwarded to `expand()` and is a test seam: it exists so that a
/// lockstep graph can be built both with and without spec §8.2's fast path and
/// the two plans compared. `ExpandMode::Auto` is the production setting.
/// `ExecutionPlan::lockstep()` reports the property of the *graph* either way,
/// so the two dumps are comparable byte for byte.
/// The agreement points a rank reaches, in plan order.
///
/// Every rank compiles this same plan from the same graph (the descriptor is
/// byte-identical across ranks, which `model.descriptor_mpi` checks), so two
/// ranks can only disagree by executing a different subset of it. Comparing
/// these sequences is how the compiler proves they do not: if rank A expects
/// three agreement points and rank B two, the run deadlocks at the third, and
/// it deadlocks at run time on a machine, which is the worst place to find out.
[[nodiscard]] std::vector<std::string> epoch_sequence(const ExecutionPlan& plan,
                                                      const CouplingGraph& g,
                                                      int rank);

[[nodiscard]] ScheduleResult schedule(const CouplingGraph& g,
                                      const OperatorDAG& dag,
                                      const ClockSet& clocks,
                                      const RegionSizes& sizes,
                                      ExpandMode mode = ExpandMode::Auto);

}  // namespace fuse
