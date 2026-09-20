#pragma once

/// Rank-agreement rules as pure functions over already-gathered data.
///
/// The rules themselves involve no MPI: a collective produces an array, and
/// a rule reads it. Keeping them here makes them testable in Tier 0, where
/// most of this project's tests run, instead of only inside a two-rank job.
/// `runtime/epoch_schedule.hpp` is the same idea for the agreement schedule.
///
/// Written 2026-09-20 because the component-agreement rule read as though it
/// could not fail to fire, and a host run showed both ranks accepting a spec
/// it should have refused. A rule you can only exercise under `mpiexec` is a
/// rule you cannot debug.

#include <cstddef>
#include <vector>

namespace fuse::transport {

/// The outcome of checking that every participating rank called `build_plan()`
/// with the same component count.
struct ComponentAgreement {
  /// The first rank in either set, or -1 when no rank is in either set.
  int first_rank = -1;
  /// That rank's component count.
  int first_value = 0;
  /// The first participating rank that disagrees with it, or -1 when they all
  /// agree (which includes the no-participant case).
  int disagreeing_rank = -1;
  /// That rank's component count, meaningful only when `disagreeing_rank >= 0`.
  int disagreeing_value = 0;

  [[nodiscard]] bool agreed() const noexcept { return disagreeing_rank < 0; }
};

/// `all_components[r]` is rank r's `TransportSpec::components`, as gathered.
/// A rank counts as participating when it is in the source set, the
/// destination set, or both; a rank in neither sends and receives nothing and
/// its component count is not part of any count array, so it is not consulted.
///
/// The three vectors are indexed by rank and must be the same length. A short
/// `all_components` is treated as the end of the participant list rather than
/// read out of bounds.
[[nodiscard]] inline ComponentAgreement check_component_agreement(
    const std::vector<int>& all_components, const std::vector<char>& is_src,
    const std::vector<char>& is_dst) noexcept {
  ComponentAgreement out;
  const std::size_t n =
      std::min(all_components.size(), std::min(is_src.size(), is_dst.size()));
  for (std::size_t r = 0; r < n; ++r) {
    if (is_src[r] == 0 && is_dst[r] == 0) continue;
    if (out.first_rank < 0) {
      out.first_rank = static_cast<int>(r);
      out.first_value = all_components[r];
      continue;
    }
    if (all_components[r] != out.first_value) {
      out.disagreeing_rank = static_cast<int>(r);
      out.disagreeing_value = all_components[r];
      return out;
    }
  }
  return out;
}

}  // namespace fuse::transport
