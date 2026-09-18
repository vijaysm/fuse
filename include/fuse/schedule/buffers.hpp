#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/ir/dag.hpp"
#include "fuse/schedule/expand.hpp"

namespace fuse {

/// Supplied by the mesh backend: how many entities does this region hold on
/// this rank? Phase 3 tests pass a constant; Phase 4 Task 4.2 wires the real
/// one via `make_region_sizes(graph, backend)`. Taking the callback now means
/// the signature does not change when the backend arrives.
///
/// It is only ever called with a valid `RegionId`; `plan_buffers` rejects a
/// DAG carrying an invalid one rather than passing 0xFFFFFFFF to a backend
/// that would index an array with it.
using RegionSizes = std::function<std::size_t(RegionId)>;

/// The storage one SSA arc needs for one period. Exactly one per DAG arc.
struct BufferDecl {
  ArcId       arc;
  /// Entities in `region` on this rank, from the `RegionSizes` callback.
  /// Filled for *every* arc, component-owned or not: it describes the region,
  /// not who allocates it. `is_component_owned` alone decides allocation.
  std::size_t elements = 0;
  int         components = 1;   // always >= 1; a smaller value is rejected
  RegionId    region;
  /// The arc aliases storage a component registered, so the coupler must not
  /// allocate it and must not count it in `total_coupler_elements`
  /// (spec §4.2: components own their data). True for exactly the `Source`
  /// and `Sink` node outputs, which are the DAG's field boundary.
  bool        is_component_owned = false;
  /// The buffer carries a value from one period into the next, so it must not
  /// be recycled for another arc within a period. See `plan_buffers`.
  bool        is_persistent = false;
  std::string label;
};

struct BufferPlan {
  std::vector<BufferDecl> buffers;
  /// Sum over non-component-owned arcs of `elements * components`. This is
  /// everything the coupler itself allocates, in elements.
  std::size_t             total_coupler_elements = 0;
};

/// Spec §2 layer 3: everything the hot loop needs, allocated once.
///
/// One `BufferDecl` per arc of `dag`, in arc order, so `plan.buffers[i]`
/// describes `ArcId{i}`.
///
/// `is_component_owned` (spec §4.2). The coupler never owns component data. A
/// `Source` output is the provider's registered array as it leaves the
/// component; a `Sink` output is the consumer's registered array the coupled
/// value is delivered into. Everything between them -- accumulators, waveform
/// buffers, transport staging, remap targets -- is a coupler intermediate.
/// Both mistakes here are silent: allocating a shadow buffer for a registered
/// field means the component never sees the coupled values, and treating an
/// intermediate as registered means writing through a pointer nobody bound.
///
/// `is_persistent`. A buffer whose value is produced in one period and read in
/// another cannot be reused inside a period. Two sources:
///   - `ValueKind::Waveform`, the accumulator a `Sample` fills;
///   - a negative `TimeLevel::offset`, i.e. an arc reached through `lag(k)`
///     (spec §4.3: `lag(k)` *is* an offset of -k). The lagged buffer is
///     written at the end of period p and read in period p+1; recycling it
///     makes the consumer read this period's value instead of last period's,
///     which turns an explicit lagged scheme into an implicit one with no
///     error anywhere -- only a divergence much later.
/// Over-marking only wastes memory, so a value that stays lagged through
/// several downstream ops marks all of them.
///
/// Persistence *across iterations* of an iteration region is deliberately not
/// covered here: that is `OperatorDAG::IterationRegion::carried`, the derived
/// checkpoint set (spec §5.3, Phase 7 Task 7.2).
///
/// `timed` is checked for consistency with `dag` and otherwise unused: the
/// per-period instance count only changes buffer sizes once waveform depth is
/// real (Phase 6).
///
/// Throws (this returns a plain value, not a `Result`, so the checks cannot be
/// asserts -- an assert compiles out under Release's -O3 -DNDEBUG, and
/// unsigned wraparound is *defined* behaviour that neither ASan nor UBSan
/// reports):
///   - `std::invalid_argument` if `sizes` is empty, if an arc's `components`
///     is < 1, or if an arc's region is invalid;
///   - `std::out_of_range` if a node's output arc id is out of range for
///     `dag`, or `timed` names a node `dag` does not have;
///   - `std::overflow_error` if `elements * components`, or the running total,
///     does not fit `std::size_t`.
/// Buffer planning that REPORTS. Every failure it can detect is a setup
/// mistake the caller turns into a diagnostic, so none of them is routed
/// through an exception -- which also means the behaviour is identical with
/// exceptions disabled.
[[nodiscard]] Result<BufferPlan> try_plan_buffers(const OperatorDAG& dag,
                                                  const TimedDag& timed,
                                                  const RegionSizes& sizes);

/// The throwing spelling, for direct callers that treat these as programming
/// errors.
[[nodiscard]] BufferPlan plan_buffers(const OperatorDAG& dag,
                                      const TimedDag& timed,
                                      const RegionSizes& sizes);

}  // namespace fuse
