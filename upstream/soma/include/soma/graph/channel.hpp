#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "soma/base/ids.hpp"
#include "soma/graph/component.hpp"

namespace soma {

// Defined in coupling_graph.hpp. ChannelBuilder holds only a reference to
// it, so a forward declaration avoids a circular include; the mutating
// ChannelBuilder method bodies are defined out-of-line, after CouplingGraph
// is a complete type.
class CouplingGraph;

enum class RemapFlags : std::uint32_t {
  None         = 0,
  Conservative = 1u << 0,
  Monotone     = 1u << 1,
  HighOrder    = 1u << 2,
};

[[nodiscard]] constexpr RemapFlags operator|(RemapFlags a, RemapFlags b) noexcept {
  return static_cast<RemapFlags>(static_cast<std::uint32_t>(a) |
                                 static_cast<std::uint32_t>(b));
}
[[nodiscard]] constexpr RemapFlags operator&(RemapFlags a, RemapFlags b) noexcept {
  return static_cast<RemapFlags>(static_cast<std::uint32_t>(a) &
                                 static_cast<std::uint32_t>(b));
}
[[nodiscard]] constexpr bool has(RemapFlags set, RemapFlags bit) noexcept {
  return static_cast<std::uint32_t>(set & bit) != 0u;
}

struct AccumulateSpec {
  AccumOp op = AccumOp::Instant;
  Window  window{};
};

struct RemapSpec {
  RemapFlags  flags = RemapFlags::None;
  std::string method_override;  // empty = derive from disc pair + conservation
};

struct SpacetimeSpec {
  TemporalBasis basis = TemporalBasis::None;
  int           degree = 0;
  Window        window{};
};

struct RelaxSpec {
  bool   enabled = false;
  double omega = 1.0;
};

struct ExtrapolateSpec {
  bool enabled = false;
  int  order = 0;
};

/// One directed coupling edge with its temporal and spatial semantics.
/// Spec §5.2 lowers this into primitive operators.
struct Channel {
  std::string     name;
  FieldId         source;
  FieldId         target;
  AccumulateSpec  accumulate;
  RemapSpec       remap;
  SpacetimeSpec   spacetime;
  RelaxSpec       relax;
  ExtrapolateSpec extrapolate;
  int             lag = 0;  // spec §4.3: lag(k) is a TimeLevel offset of -k
  std::string     time_level_mismatch_reason;  // empty = strict typing enforced
};

/// Fluent mutator. Deliberately does not validate: spec Global Constraints
/// require all problems to be reported together by validate() (Task 1.5).
///
/// Holds `(CouplingGraph&, ChannelId)` rather than a `Channel&`: the graph's
/// backing `std::vector<Channel>` can reallocate on any subsequent
/// `add_channel()` call, which would leave a stored `Channel&` dangling.
/// Each mutating call re-resolves the channel through the graph, so a
/// builder kept alive (named, not just used as a fluent temporary) across
/// intervening `add_channel()` calls stays valid. Method bodies are defined
/// out-of-line in coupling_graph.hpp, once CouplingGraph is a complete type.
class ChannelBuilder {
 public:
  ChannelBuilder(CouplingGraph& g, ChannelId id) : graph_(&g), id_(id) {}

  ChannelBuilder& accumulate(AccumOp op, Window w = Window::coupling_period());
  ChannelBuilder& remap(RemapFlags flags);
  ChannelBuilder& remap_method(std::string method);
  ChannelBuilder& conservative();
  ChannelBuilder& spacetime(TemporalBasis basis, int degree,
                            Window w = Window::coupling_period());
  ChannelBuilder& relax(double omega);
  ChannelBuilder& extrapolate(int order);
  ChannelBuilder& lag(int steps);
  ChannelBuilder& allow_time_level_mismatch(std::string reason);

 private:
  [[nodiscard]] Channel& target();

  CouplingGraph* graph_;
  ChannelId      id_;
};

}  // namespace soma
