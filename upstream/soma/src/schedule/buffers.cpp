#include "soma/schedule/buffers.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace soma {
namespace {

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

/// `a * b`, rejected rather than wrapped.
///
/// Checked in every build configuration, not via assert(): unsigned overflow
/// is defined to wrap, so neither UBSan nor ASan reports it, and the wrapped
/// product is a plausible-looking small allocation whose first write is a heap
/// overflow.
[[nodiscard]] std::size_t checked_mul(std::size_t a, std::size_t b,
                                      const char* what) {
  if (a != 0 && b > kSizeMax / a) {
    throw std::overflow_error(std::string{"plan_buffers: "} + what + ": " +
                              std::to_string(a) + " * " + std::to_string(b) +
                              " does not fit std::size_t");
  }
  return a * b;
}

[[nodiscard]] std::size_t checked_add(std::size_t a, std::size_t b,
                                      const char* what) {
  if (a > kSizeMax - b) {
    throw std::overflow_error(std::string{"plan_buffers: "} + what + ": " +
                              std::to_string(a) + " + " + std::to_string(b) +
                              " does not fit std::size_t");
  }
  return a + b;
}

}  // namespace

BufferPlan plan_buffers(const OperatorDAG& dag, const TimedDag& timed,
                        const RegionSizes& sizes) {
  if (!sizes) {
    throw std::invalid_argument(
        "plan_buffers: the RegionSizes callback is empty; a mesh backend (or, "
        "before Phase 4 Task 4.2, a stub) must supply one");
  }

  const std::size_t n_arcs = dag.arc_count();
  const std::size_t n_nodes = dag.node_count();

  // `timed` is otherwise unused (see the header): the one thing it can tell us
  // now is whether it was produced from this DAG at all. Pairing a TimedDag
  // with the wrong lowering would silently plan against the wrong arcs.
  for (const TimedNode& tn : timed.nodes) {
    if (!tn.node.valid() || tn.node.value() >= n_nodes) {
      throw std::out_of_range(
          "plan_buffers: TimedDag names node " +
          std::to_string(tn.node.value()) + " but the DAG has " +
          std::to_string(n_nodes) + "; the TimedDag is not this DAG's");
    }
  }

  // Arcs that alias component storage: Source outputs and Sink outputs
  // (spec §4.2). Everything else is a coupler intermediate.
  std::vector<char> component_owned(n_arcs, 0);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const Node& n = dag.node(NodeId{static_cast<std::uint32_t>(i)});
    if (n.kind != OpKind::Source && n.kind != OpKind::Sink) continue;
    if (!n.output.valid()) continue;
    if (n.output.value() >= n_arcs) {
      throw std::out_of_range("plan_buffers: node " + std::to_string(i) +
                              " outputs arc " +
                              std::to_string(n.output.value()) +
                              " but the DAG has " + std::to_string(n_arcs));
    }
    component_owned[n.output.value()] = 1;
  }

  BufferPlan plan;
  plan.buffers.reserve(n_arcs);

  for (std::size_t i = 0; i < n_arcs; ++i) {
    const ArcId aid{static_cast<std::uint32_t>(i)};
    const Arc&  a = dag.arc(aid);

    if (a.type.components < 1) {
      // A negative count cast to std::size_t is the sharpest case: -1 becomes
      // 2^64-1 and the product wraps back to something small and plausible.
      // validate() does not police this, so plan_buffers must.
      throw std::invalid_argument(
          "plan_buffers: arc " + std::to_string(i) + " (\"" + a.name +
          "\") declares " + std::to_string(a.type.components) +
          " components; a field must have at least 1");
    }
    if (!a.type.region.valid()) {
      throw std::invalid_argument("plan_buffers: arc " + std::to_string(i) +
                                  " (\"" + a.name +
                                  "\") has no region, so its size cannot be "
                                  "asked of the mesh backend");
    }

    BufferDecl b;
    b.arc = aid;
    b.region = a.type.region;
    b.components = a.type.components;
    b.is_component_owned = component_owned[i] != 0;

    // Waveform accumulators, and anything reached through lag(k) -- which is
    // exactly a negative TimeLevel offset (spec §4.3) -- carry a value across
    // the period boundary and must not be recycled inside one. Rational's
    // operator<=> compares in 128 bits and is noexcept, so this cannot throw.
    b.is_persistent = (a.type.kind == ValueKind::Waveform) ||
                      (a.type.time.offset < Rational{0});

    b.label = a.name.empty() ? ("%" + std::to_string(i)) : a.name;

    // Filled for component-owned arcs too: `elements` describes the region,
    // and Phase 4 needs it to check a registered pointer's Layout::count
    // against what the plan expects. Allocation is governed by the flag.
    b.elements = sizes(b.region);

    if (!b.is_component_owned) {
      const std::size_t need =
          checked_mul(b.elements, static_cast<std::size_t>(b.components),
                      "buffer size");
      plan.total_coupler_elements =
          checked_add(plan.total_coupler_elements, need, "total coupler size");
    }
    plan.buffers.push_back(std::move(b));
  }

  return plan;
}

}  // namespace soma
