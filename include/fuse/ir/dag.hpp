#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fuse/ir/op.hpp"
#include "fuse/base/error.hpp"

namespace fuse {

/// Spec Section 2 layer 2. Single assignment: every arc has exactly one producer,
/// enforced at add_node time rather than checked afterwards.
class OperatorDAG {
 public:
  /// Spec Section 5.3. `carried` is the derived checkpoint set (Phase 7 Task 7.2):
  /// the loop-carried coupler buffers whose sweep-k value depends on sweep
  /// k-1's (src/ir/checkpoint.cpp names a node's output as carried when its
  /// kind is `OpKind::Relax` or `OpKind::Accelerate` -- see
  /// `Runner::Impl::run_region`'s long comment in src/runtime/iteration.cpp
  /// for why restoring these between sweeps would be the wrong physics).
  ///
  /// R13: as of Phase 7, `carried` can in practice only ever be populated by
  /// `Relax` nodes. `OpKind::Accelerate` has no emitter anywhere in
  /// src/ir/rules_channel.cpp or src/ir/rules_derived.cpp (grepped, zero
  /// hits) and no lowering rule is planned to add one -- Task 7.3 implemented
  /// acceleration as a REGION-LEVEL construct instead (`num::Accelerator`,
  /// driven by `IterationRegionDecl::acceleration` and applied in
  /// `Runner::Impl::run_region` to the region's whole concatenated iterate
  /// vector, `R.x_old`/`R.x_new`/`R.x_out`), not as a per-channel IR node the
  /// way `Relax` is. `OpKind::Accelerate` is therefore not "reserved for a
  /// future lowering rule" so much as a spec-table artifact (spec Section 5.1 lists
  /// it alongside `Relax` among the primitives) that predates that design
  /// decision and was never removed; `checkpoint.cpp`'s `Accelerate` branch
  /// and `Runner::Impl::execute`'s `case OpKind::Accelerate` (runner.cpp,
  /// currently a plain passthrough copy, same as `Sink`) are both unreachable
  /// code kept in sync with an enumerator nothing emits, not evidence that
  /// something does.
  struct IterationRegion {
    std::string         name;
    std::vector<NodeId> body;
    std::vector<ArcId>  residuals;
    std::vector<ArcId>  carried;
    int                 max_iterations = 1;
  };

  ArcId add_arc(ValueType type, std::string name) {
    const ArcId id{static_cast<std::uint32_t>(arcs_.size())};
    arcs_.push_back(Arc{id, NodeId{}, std::move(type), std::move(name)});
    consumers_.emplace_back();
    return id;
  }

  // Bounds and the SSA invariant are enforced unconditionally, not via
  // assert(): they must hold in Release (-DNDEBUG) builds too, since a
  // second producer or an out-of-range ArcId is a heap-buffer-overflow
  // waiting to happen once asserts compile out. `.at()` covers the bounds
  // checks (mirrors CouplingGraph's accessors); the SSA check has no
  // container-provided equivalent, so it throws std::logic_error directly.
  NodeId add_node(OpKind kind, std::vector<ArcId> inputs, ArcId output,
                  OpAttrs attrs, ChannelId origin, std::string label) {
    const NodeId id{static_cast<std::uint32_t>(nodes_.size())};
    for (const ArcId in : inputs) {
      // .at() bounds-checks in every build configuration; assert() alone
      // would compile out under -DNDEBUG and leave an out-of-range index
      // to run straight into consumers_[in.value()].
      consumers_.at(in.value()).push_back(id);
    }
    if (output.valid()) {
      Arc& out_arc = arcs_.at(output.value());
      if (out_arc.producer.valid()) {
        FUSE_THROW(InvariantViolation, "{}",
            "OperatorDAG::add_node: SSA violation - arc " +
            std::to_string(output.value()) + " already has a producer (node " +
            std::to_string(out_arc.producer.value()) + "); cannot assign node " +
            std::to_string(id.value()) + " as a second producer");
      }
      out_arc.producer = id;
    }
    nodes_.push_back(Node{id, kind, std::move(inputs), output, std::move(attrs),
                          origin, std::move(label)});
    return id;
  }

  // Reference-invalidation hazard: the returned reference is into a
  // std::vector that add_arc()/add_node() can reallocate. It is invalidated
  // by any subsequent call to either. Callers must re-resolve by id rather
  // than hold the reference across a mutating call -- the same rule as
  // ChannelBuilder (see channel.hpp), which was redesigned for exactly this
  // reason.
  [[nodiscard]] const Node& node(NodeId id) const { return nodes_.at(id.value()); }
  // Reference-invalidation hazard: see node() above; invalidated by any
  // subsequent add_arc()/add_node().
  [[nodiscard]] const Arc&  arc(ArcId id) const { return arcs_.at(id.value()); }
  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
  [[nodiscard]] std::size_t arc_count() const noexcept { return arcs_.size(); }

  // Reference-invalidation hazard: see node() above; invalidated by any
  // subsequent add_arc()/add_node() (consumers_ is parallel to arcs_ and
  // also grows on add_arc(); the inner vector is appended to by add_node()).
  [[nodiscard]] const std::vector<NodeId>& consumers_of(ArcId id) const {
    return consumers_.at(id.value());
  }

  /// Kahn's algorithm with ties broken by ascending NodeId, so the order is
  /// reproducible run to run -- golden dumps depend on it.
  [[nodiscard]] std::vector<NodeId> topological_order() const {
    std::vector<int> indegree(nodes_.size(), 0);
    for (const auto& n : nodes_) {
      for (const ArcId in : n.inputs) {
        if (arcs_[in.value()].producer.valid()) {
          indegree[n.id.value()]++;
        }
      }
    }
    std::priority_queue<std::uint32_t, std::vector<std::uint32_t>,
                        std::greater<>> ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (indegree[i] == 0) ready.push(static_cast<std::uint32_t>(i));
    }
    std::vector<NodeId> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
      const std::uint32_t v = ready.top();
      ready.pop();
      order.push_back(NodeId{v});
      const ArcId out = nodes_[v].output;
      if (!out.valid()) continue;
      for (const NodeId c : consumers_[out.value()]) {
        if (--indegree[c.value()] == 0) ready.push(c.value());
      }
    }
    return order;
  }

  [[nodiscard]] bool is_acyclic() const {
    return topological_order().size() == nodes_.size();
  }

  RegionIdx add_region(IterationRegion r) {
    regions_.push_back(std::move(r));
    return RegionIdx{static_cast<std::uint32_t>(regions_.size() - 1)};
  }
  [[nodiscard]] const IterationRegion& region_at(RegionIdx i) const {
    return regions_.at(i.value());
  }
  [[nodiscard]] std::size_t region_count() const noexcept {
    return regions_.size();
  }

 private:
  std::vector<Node>                nodes_;
  std::vector<Arc>                 arcs_;
  std::vector<std::vector<NodeId>> consumers_;  // parallel to arcs_
  std::vector<IterationRegion>     regions_;
};

}  // namespace fuse
