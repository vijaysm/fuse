#pragma once

#include <cstdint>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

/// One interval of the intersected source/target time partitions.
/// src_step and dst_step say which step of each side the slab falls within;
/// a slab never straddles a step boundary on either side, by construction.
struct Slab {
  Rational     begin;
  Rational     end;
  std::int64_t src_step = 0;
  std::int64_t dst_step = 0;

  [[nodiscard]] Rational length() const { return end - begin; }
  friend bool operator==(const Slab&, const Slab&) = default;
};

struct SlabDecomposition {
  std::vector<Slab> slabs;
  Rational          period{1};
  /// True when the source and target partitions coincide -- equal intervals,
  /// both anchored at 0 -- so every slab is a whole step of both sides and
  /// nothing is sub-stepped. This is spec §8.2's lockstep fast path. Note it
  /// is *not* the same as `slabs.size() == 1`: equal intervals shorter than
  /// the period give several slabs and are still the lockstep case.
  bool              degenerate = false;
};

/// Spec §8.3: a runaway decomposition is a setup error, not a hang.
inline constexpr std::int64_t kMaxSlabsPerPeriod = 100'000;

/// Intersect the source and target time partitions of [0, period).
///
/// Errors (never exceptions -- every arithmetic step that could overflow is
/// range-checked in __int128 before it reaches Rational):
///   E-SCHED-INTERVAL  a non-positive source or target interval
///   E-SCHED-PERIOD    a non-positive period, a period that is not an integer
///                     multiple of one of the intervals, or a decomposition
///                     larger than kMaxSlabsPerPeriod slabs
[[nodiscard]] Result<SlabDecomposition> decompose(Rational src_interval,
                                                  Rational dst_interval,
                                                  Rational period);

/// Test seam: how many times has decompose() been called? Used to prove the
/// lockstep fast path (spec §8.2) does not touch the slab machinery.
[[nodiscard]] std::uint64_t decompose_call_count();
void reset_decompose_call_count();

}  // namespace fuse
