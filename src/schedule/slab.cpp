#include "fuse/schedule/slab.hpp"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

namespace fuse {
namespace {

using detail::Int128;

[[nodiscard]] Int128 abs128(Int128 v) { return v < 0 ? -v : v; }

[[nodiscard]] std::string i128_to_string(Int128 v) {
  if (v == 0) return "0";
  const bool neg = v < 0;
  Int128 m = abs128(v);
  std::string digits;
  while (m > 0) {
    digits.push_back(static_cast<char>('0' + static_cast<int>(m % 10)));
    m /= 10;
  }
  if (neg) digits.push_back('-');
  std::reverse(digits.begin(), digits.end());
  return digits;
}

/// i * interval, formed and range-checked in __int128.
///
/// Rational's own operator* throws std::logic_error when the reduced result
/// does not fit int64_t, and that can genuinely happen here for a legal
/// interval and period (e.g. interval 8999999999999999999/4 over a period of
/// 8999999999999999999: tick 3 has numerator 3*P, which is > 2^63). decompose()
/// returns a Result, so the condition has to be detected rather than thrown.
[[nodiscard]] Result<Rational> scaled_tick(std::int64_t i,
                                           const Rational& interval) {
  const Int128 n = static_cast<Int128>(i) * interval.num();
  const Int128 d = interval.den();
  const Int128 g = detail::gcd128(abs128(n), d);
  const Int128 rn = n / g;
  if (!detail::fits_i64(rn)) {
    return Failure{"E-SCHED-PERIOD",
                 "tick " + std::to_string(i) + " of interval " +
                     interval.to_string() + " is " + i128_to_string(rn) + "/" +
                     i128_to_string(d / g) +
                     ", whose numerator does not fit a 64-bit Rational"};
  }
  return Rational{static_cast<std::int64_t>(rn),
                  static_cast<std::int64_t>(d / g)};
}

/// Every tick of `interval` in [0, period], inclusive of both ends.
/// `interval` and `period` are already known to be strictly positive.
[[nodiscard]] Result<std::vector<Rational>> ticks_of(const Rational& interval,
                                                     const Rational& period) {
  // period / interval in __int128, so an unrepresentable ratio is reported
  // rather than thrown out of Rational::operator/.
  const Int128 num = static_cast<Int128>(period.num()) * interval.den();
  const Int128 den = static_cast<Int128>(period.den()) * interval.num();
  // den > 0: denominators are always positive and interval.num() > 0.
  if (num % den != 0) {
    // period is always an LCM of the intervals, so this cannot happen unless a
    // caller passes an unrelated period.
    return Failure{"E-SCHED-PERIOD",
                 "period " + period.to_string() +
                     " is not an integer multiple of interval " +
                     interval.to_string()};
  }
  const Int128 n = num / den;
  if (n > kMaxSlabsPerPeriod) {
    return Failure{"E-SCHED-PERIOD",
                 "interval " + interval.to_string() + " needs " +
                     i128_to_string(n) + " ticks per period " +
                     period.to_string() + ", exceeding the limit of " +
                     std::to_string(kMaxSlabsPerPeriod)};
  }

  const auto count = static_cast<std::int64_t>(n);
  std::vector<Rational> out;
  out.reserve(static_cast<std::size_t>(count) + 1);
  for (std::int64_t i = 0; i <= count; ++i) {
    Result<Rational> t = scaled_tick(i, interval);
    if (!t.ok()) return t.error();
    out.push_back(t.value());
  }
  return out;
}

/// Which step of `interval` contains the half-open slab starting at `t`?
/// t >= 0 and t <= period, and both tick counts are already bounded by
/// kMaxSlabsPerPeriod, so the quotient is a small non-negative rational and
/// the truncating division below is a floor.
[[nodiscard]] std::int64_t step_containing(const Rational& t,
                                           const Rational& interval) {
  const Rational q = t / interval;
  return q.num() / q.den();
}

/// Test seam (declared in slab.hpp): counts calls to decompose() so that the
/// lockstep fast path of spec Section 8.2 can be *proved* not to touch the slab
/// machinery rather than assumed not to. Atomic because two threads may build
/// two plans concurrently; relaxed ordering, since the counter is only ever
/// read after the build it measures has finished.
std::atomic<std::uint64_t>& decompose_counter() {
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}

}  // namespace

std::uint64_t decompose_call_count() {
  return decompose_counter().load(std::memory_order_relaxed);
}
void reset_decompose_call_count() {
  decompose_counter().store(0, std::memory_order_relaxed);
}

Result<SlabDecomposition> decompose(Rational src_interval,
                                    Rational dst_interval, Rational period) {
  decompose_counter().fetch_add(1, std::memory_order_relaxed);
  if (!(src_interval > Rational{0})) {
    return Failure{"E-SCHED-INTERVAL", "source interval must be positive, got " +
                                         src_interval.to_string()};
  }
  if (!(dst_interval > Rational{0})) {
    return Failure{"E-SCHED-INTERVAL", "target interval must be positive, got " +
                                         dst_interval.to_string()};
  }
  if (!(period > Rational{0})) {
    return Failure{"E-SCHED-PERIOD",
                 "period must be positive, got " + period.to_string()};
  }

  Result<std::vector<Rational>> src_ticks = ticks_of(src_interval, period);
  if (!src_ticks.ok()) return src_ticks.error();
  Result<std::vector<Rational>> dst_ticks = ticks_of(dst_interval, period);
  if (!dst_ticks.ok()) return dst_ticks.error();

  std::vector<Rational>& dst = dst_ticks.value();
  std::vector<Rational> all = std::move(src_ticks.value());
  all.insert(all.end(), dst.begin(), dst.end());
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());

  // Count slabs, not boundaries: n boundaries are n-1 slabs, and comparing
  // boundaries against the limit would reject a decomposition of exactly
  // kMaxSlabsPerPeriod slabs that ticks_of() has already accepted.
  const std::size_t slab_count = all.empty() ? 0u : all.size() - 1u;
  if (static_cast<std::int64_t>(slab_count) > kMaxSlabsPerPeriod) {
    return Failure{"E-SCHED-PERIOD",
                 "slab decomposition would need " +
                     std::to_string(slab_count) +
                     " slabs, exceeding the limit of " +
                     std::to_string(kMaxSlabsPerPeriod)};
  }

  SlabDecomposition out;
  out.period = period;
  out.slabs.reserve(slab_count);
  for (std::size_t i = 1; i < all.size(); ++i) {
    Slab s;
    s.begin = all[i - 1];
    s.end = all[i];
    s.src_step = step_containing(s.begin, src_interval);
    s.dst_step = step_containing(s.begin, dst_interval);
    out.slabs.push_back(s);
  }
  // The partitions coincide exactly when the intervals are equal; both are
  // anchored at 0, so equal intervals means every slab is a whole step of
  // both sides (spec Section 8.2's lockstep fast path).
  out.degenerate = (src_interval == dst_interval);
  return out;
}

}  // namespace fuse
