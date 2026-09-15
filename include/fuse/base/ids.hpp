#pragma once

#include <compare>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace fuse {

/// Strong-typed index. Distinct Tag types are distinct C++ types, so a
/// ComponentId cannot be passed where a ChannelId is expected.
template <class Tag>
class Id {
 public:
  static constexpr std::uint32_t kInvalid = 0xFFFFFFFFu;

  constexpr Id() = default;
  explicit constexpr Id(std::uint32_t v) : value_(v) {}

  [[nodiscard]] constexpr std::uint32_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalid; }

  friend constexpr bool operator==(const Id&, const Id&) = default;
  friend constexpr std::strong_ordering operator<=>(const Id&, const Id&) = default;

 private:
  std::uint32_t value_ = kInvalid;
};

struct ComponentTag {};
struct RegionTag {};
struct FieldTag {};
struct ChannelTag {};
struct MeshTag {};
struct NodeTag {};
struct ArcTag {};
struct IterRegionTag {};

using ComponentId = Id<ComponentTag>;
using RegionId    = Id<RegionTag>;
using FieldId     = Id<FieldTag>;
using ChannelId   = Id<ChannelTag>;
using MeshId      = Id<MeshTag>;
using NodeId      = Id<NodeTag>;       // operator node in the DAG
using ArcId       = Id<ArcTag>;        // SSA value
using RegionIdx   = Id<IterRegionTag>; // iteration region

enum class Discretization { FV, ContinuousGLL, DiscontinuousGLL, PointCloud };
enum class Conservation   { None, Extensive, Intensive };
enum class Intent         { Provides, Requires, Bidirectional };
enum class AccumOp        { Instant, Sum, Mean, Min, Max, Integral };
enum class TemporalBasis  { None, Constant, Linear, DG, Spline };

[[nodiscard]] constexpr const char* to_cstring(Discretization d) noexcept {
  switch (d) {
    case Discretization::FV:               return "FV";
    case Discretization::ContinuousGLL:    return "cGLL";
    case Discretization::DiscontinuousGLL: return "dGLL";
    case Discretization::PointCloud:       return "PointCloud";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(Conservation c) noexcept {
  switch (c) {
    case Conservation::None:      return "None";
    case Conservation::Extensive: return "Extensive";
    case Conservation::Intensive: return "Intensive";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(Intent i) noexcept {
  switch (i) {
    case Intent::Provides:      return "Provides";
    case Intent::Requires:      return "Requires";
    case Intent::Bidirectional: return "Bidirectional";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(AccumOp a) noexcept {
  switch (a) {
    case AccumOp::Instant:  return "Instant";
    case AccumOp::Sum:      return "Sum";
    case AccumOp::Mean:     return "Mean";
    case AccumOp::Min:      return "Min";
    case AccumOp::Max:      return "Max";
    case AccumOp::Integral: return "Integral";
  }
  return "Unknown";
}

[[nodiscard]] constexpr const char* to_cstring(TemporalBasis t) noexcept {
  switch (t) {
    case TemporalBasis::None:     return "None";
    case TemporalBasis::Constant: return "Constant";
    case TemporalBasis::Linear:   return "Linear";
    case TemporalBasis::DG:       return "DG";
    case TemporalBasis::Spline:   return "Spline";
  }
  return "Unknown";
}

}  // namespace fuse

template <class Tag>
struct std::hash<fuse::Id<Tag>> {
  std::size_t operator()(const fuse::Id<Tag>& id) const noexcept {
    return std::hash<std::uint32_t>{}(id.value());
  }
};
