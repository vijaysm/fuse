#include <optional>

#include "fuse/schedule/buffers.hpp"
#include "fuse/base/error.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fuse {

/// One code for every way buffer planning can fail; the message carries which.
inline constexpr const char* kBuffersCode = "E-SCHED-BUFFERS";
namespace {

constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

/// `a * b`, rejected rather than wrapped.
///
/// Checked in every build configuration, not via assert(): unsigned overflow
/// is defined to wrap, so neither UBSan nor ASan reports it, and the wrapped
/// product is a plausible-looking small allocation whose first write is a heap
/// overflow.
[[nodiscard]] std::optional<std::size_t> checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > kSizeMax / a) return std::nullopt;
  return a * b;
}

[[nodiscard]] std::optional<std::size_t> checked_add(std::size_t a, std::size_t b) {
  if (a > kSizeMax - b) return std::nullopt;
  return a + b;
}

}  // namespace

Result<BufferPlan> try_plan_buffers(const OperatorDAG& dag, const TimedDag& timed,
                                    const RegionSizes& sizes) {
  if (!sizes) {
    return Failure{kBuffersCode, "plan_buffers: the RegionSizes callback is empty; a mesh backend (or, "
        "before Phase 4 Task 4.2, a stub) must supply one"};
  }

  const std::size_t n_arcs = dag.arc_count();
  const std::size_t n_nodes = dag.node_count();

  // `timed` is otherwise unused (see the header): the one thing it can tell us
  // now is whether it was produced from this DAG at all. Pairing a TimedDag
  // with the wrong lowering would silently plan against the wrong arcs.
  for (const TimedNode& tn : timed.nodes) {
    if (!tn.node.valid() || tn.node.value() >= n_nodes) {
      return Failure{kBuffersCode, "plan_buffers: TimedDag names node " +
          std::to_string(tn.node.value()) + " but the DAG has " +
          std::to_string(n_nodes) + "; the TimedDag is not this DAG's"};
    }
  }

  // Arcs that alias component storage: Source outputs and Sink outputs
  // (spec §4.2). Everything else is a coupler intermediate.
  //
  // Time windows track (C3): a window-scoped IMPORT's own Sink is the one
  // exception. Its output is K samples wide (window_import_width(),
  // src/ir/window_lowering.cpp) -- the provider's whole flat waveform for
  // this window, delivered unchanged -- never the SINGLE value the
  // component's own native Requires storage holds; advance_window()
  // (src/runtime/iteration.cpp) reads this buffer directly and writes
  // its OWN per-substep mean into the component's native storage
  // separately. Aliasing this arc onto that native storage the way an
  // ordinary Sink's output is aliased would size-mismatch it (the
  // component registers 1 value, the plan wants K) and, if the sizes
  // were forced to agree, overwrite the K-wide delivery with a single
  // scalar mid-sweep. `emit_field_sink()` (src/ir/lowering.cpp) marks
  // exactly this Sink with the SAME "/window" label suffix the window
  // EXPORT side's own WindowAdvance output already uses for the
  // identical reason (that one is excluded above by node KIND, since
  // WindowAdvance is neither Source nor Sink; this one needs a label
  // check because it genuinely is a Sink).
  const auto is_window_arc = [](const std::string& label) {
    constexpr std::string_view kSuffix = "/window";
    return label.size() >= kSuffix.size() &&
           label.compare(label.size() - kSuffix.size(), kSuffix.size(),
                        kSuffix) == 0;
  };
  std::vector<char> component_owned(n_arcs, 0);
  for (std::size_t i = 0; i < n_nodes; ++i) {
    const Node& n = dag.node(NodeId{static_cast<std::uint32_t>(i)});
    if (n.kind != OpKind::Source && n.kind != OpKind::Sink) continue;
    if (!n.output.valid()) continue;
    if (n.output.value() >= n_arcs) {
      return Failure{kBuffersCode, "plan_buffers: node " + std::to_string(i) +
                              " outputs arc " +
                              std::to_string(n.output.value()) +
                              " but the DAG has " + std::to_string(n_arcs)};
    }
    if (n.kind == OpKind::Sink && is_window_arc(n.label)) continue;
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
      return Failure{kBuffersCode, "plan_buffers: arc " + std::to_string(i) + " (\"" + a.name +
          "\") declares " + std::to_string(a.type.components) +
          " components; a field must have at least 1"};
    }
    if (!a.type.region.valid()) {
      return Failure{kBuffersCode, "plan_buffers: arc " + std::to_string(i) +
                                  " (\"" + a.name +
                                  "\") has no region, so its size cannot be "
                                  "asked of the mesh backend"};
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
    b.elements = sizes(b.region, a.type.rank_set);

    if (!b.is_component_owned) {
      const auto need =
          checked_mul(b.elements, static_cast<std::size_t>(b.components));
      if (!need) {
        return Failure{kBuffersCode,
                       "plan_buffers: buffer size " + std::to_string(b.elements) +
                           " * " + std::to_string(b.components) +
                           " does not fit std::size_t"};
      }
      const auto total = checked_add(plan.total_coupler_elements, *need);
      if (!total) {
        return Failure{kBuffersCode,
                       "plan_buffers: total coupler size " +
                           std::to_string(plan.total_coupler_elements) + " + " +
                           std::to_string(*need) + " does not fit std::size_t"};
      }
      plan.total_coupler_elements = *total;
    }
    plan.buffers.push_back(std::move(b));
  }

  return plan;
}

/// The throwing spelling, unchanged for direct callers. `schedule()` uses the
/// reporting one above: it must turn a setup mistake into a diagnostic, and
/// routing that through an exception made the conversion depend on exceptions
/// being enabled -- with `FUSE_ENABLE_EXCEPTIONS=OFF` the refusal aborted
/// instead of unwinding and the process died inside
/// `ExecutionPlan.ReportsAnEmptyRegionSizesCallbackAsADiagnostic`.
BufferPlan plan_buffers(const OperatorDAG& dag, const TimedDag& timed,
                        const RegionSizes& sizes) {
  auto r = try_plan_buffers(dag, timed, sizes);
  if (!r.ok()) {
    FUSE_THROW(InvalidArgument, "{}", r.error().message);
  }
  return std::move(r).value();
}

}  // namespace fuse
