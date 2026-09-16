#pragma once

/// How several channels writing one field are combined.
///
/// Until now this was simply refused, for a specific reason worth keeping in
/// view: the compiler used to bind one arc per target field, so a second
/// channel writing the same field silently replaced the first. A fuzz sweep of
/// the parent project found 232 graphs out of 19,181 that the acyclicity proof
/// certified as loop-free *because* the dropped writers removed the loop — the
/// plan was proven correct, and it was not the plan the user described.
///
/// The fix is not to make the compiler guess. Summing two heat fluxes and
/// averaging two temperatures are different physics, and nothing in the field
/// descriptions says which one the user means. So combining is legal only when
/// the user states the rule, and stating it is what this type is.

#include <cstdint>
#include <string>
#include <string_view>

#include "fuse/base/ids.hpp"

namespace fuse {

enum class MergeOp : std::uint8_t {
  /// Add the contributions. The right rule for an extensive quantity split
  /// across contributing components: two heat fluxes into one surface add.
  Sum = 0,
  /// Arithmetic mean. For an intensive quantity sampled by several sources —
  /// note that it is unweighted, so it is only meaningful when the sources
  /// cover comparable extents.
  Mean = 1,
  /// Take the first contributing channel in declaration order, ignore the
  /// rest. An override: a fine-scale source shadowing a background field.
  First = 2,
};

[[nodiscard]] constexpr std::string_view merge_op_name(MergeOp op) noexcept {
  switch (op) {
    case MergeOp::Sum:   return "Sum";
    case MergeOp::Mean:  return "Mean";
    case MergeOp::First: return "First";
  }
  return "Unknown";
}

/// Declared per target field. The channels are not listed: every channel that
/// targets the field participates, because a list would let a channel be
/// written, forgotten, and silently dropped — the failure this type exists to
/// remove. Order is channel declaration order, which is what makes `First`
/// well defined and the merge deterministic across ranks.
struct MergeSpec {
  MergeOp op = MergeOp::Sum;
};

}  // namespace fuse
