#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "fuse/partition/detail/collectives.hpp"
#include "fuse/partition/types.hpp"

/// Kokkos port of rcb_moab/ParallelMedian.{hpp,cpp}::weighted_median.
///
/// Same algorithm, same arithmetic, same iteration order as the source
/// (docs/dev/partitioning.md "Determinism"): per-iteration histogram
/// refinement over K uniform buckets, a fixed iteration cap, and bucket
/// bounds (not strict equality against split_value) as the tie region a
/// caller must classify against. What changed is WHERE each piece runs:
/// the per-element work (the local axis range, the per-bucket weight sum)
/// is a Kokkos kernel over whatever memory space `ExecutionSpace` names;
/// the Allreduce is `detail::allreduce_*`, so this compiles and runs with
/// `comm == kNullComm` in a base (no-MPI) build exactly as it does with a
/// real communicator in the MPI-enabled build configuration.
namespace fuse::partition {

struct MedianOptions {
  int num_buckets = kDefaultHistogramBuckets;
  int refine_iterations = kDefaultHistogramRefineIters;
  double frac_target = 0.5;
  double geom_tol_relative = 1e-9;
};

struct MedianResult {
  double split_value = 0.0;
  Weight weight_below = 0;
  Weight weight_at_or_below = 0;
  Weight weight_total = 0;
  int iterations_used = 0;
  /// Final histogram bucket bounds -- the tie region. Any element with
  /// `bucket_lo <= c <= bucket_hi` is within the median's resolution and
  /// must be classified as a tie, not by strict comparison against
  /// `split_value` (see rcb_moab's ParallelMedian.hpp for why: floating
  /// point can place `split_value` on either side of a shared coordinate
  /// value, which a strict `<`/`>` would then classify inconsistently).
  double bucket_lo = 0.0;
  double bucket_hi = 0.0;
};

namespace detail {

/// Local (this rank's) sum of weights(subset(k)) for k in [0, subset.extent(0)).
template <class ExecutionSpace>
Weight local_weight_sum(
    Kokkos::View<const Weight*, typename ExecutionSpace::memory_space> weights,
    Kokkos::View<const std::size_t*, typename ExecutionSpace::memory_space> subset) {
  const std::size_t n = subset.extent(0);
  Weight total = 0;
  Kokkos::parallel_reduce(
      "fuse::partition::local_weight_sum", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t k, Weight& acc) { acc += weights(subset(k)); }, total);
  return total;
}

/// Local [lo, hi] of centroids(subset(k), axis) for k in [0, subset.extent(0)).
/// Two separate reductions (min, then max) rather than one combined
/// reducer -- simpler to get right than a hand-rolled paired reducer, and
/// this library takes robustness over micro-optimisation throughout.
template <class ExecutionSpace>
void local_axis_range(
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    Kokkos::View<const std::size_t*, typename ExecutionSpace::memory_space> subset, int axis,
    double& lo_out, double& hi_out) {
  const std::size_t n = subset.extent(0);
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  Kokkos::parallel_reduce(
      "fuse::partition::local_axis_min", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t k, double& acc) {
        const double c = centroids(subset(k), axis);
        if (c < acc) acc = c;
      },
      Kokkos::Min<double>(lo));
  Kokkos::parallel_reduce(
      "fuse::partition::local_axis_max", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t k, double& acc) {
        const double c = centroids(subset(k), axis);
        if (c > acc) acc = c;
      },
      Kokkos::Max<double>(hi));
  lo_out = lo;
  hi_out = hi;
}

/// One refine iteration's histogram: hist(0) = weight strictly below
/// cur_lo; hist(1..K) = per-bucket weight in [cur_lo, cur_hi]. `hist` must
/// already be zeroed and sized K+1; accumulation is via Kokkos::atomic_add
/// since many subset elements can land in the same bucket.
///
/// BUCKET INDEX ARITHMETIC MUST BE IDENTICAL ON EVERY EXECUTION SPACE. The
/// expression below is `(c - cur_lo) * inv_dx`, computed as two SEPARATE
/// operations (a subtraction, then a multiplication) with no addition
/// combined into either. A fused multiply-add (common in device backends'
/// codegen, less common on a plain host build) can only change the result
/// of an `a*b+c` PATTERN; there is no such pattern here for the compiler to
/// fuse, so this expression is already immune to host/device FMA
/// divergence and needs no `#pragma STDC FP_CONTRACT OFF` -- the guarantee
/// comes from the expression's shape, not a compiler flag. The other place
/// this port could have introduced an FMA-prone `a + b*c` expression
/// (updating cur_lo/cur_hi from the chosen bucket) is deliberately kept on
/// the HOST, in `weighted_median` below, where exactly one code path is
/// ever compiled -- so no host/device arithmetic divergence is possible
/// there either. See docs/dev/partitioning.md "Determinism".
template <class ExecutionSpace>
void accumulate_histogram(
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    Kokkos::View<const Weight*, typename ExecutionSpace::memory_space> weights,
    Kokkos::View<const std::size_t*, typename ExecutionSpace::memory_space> subset, int axis,
    double cur_lo, double cur_hi, int K,
    Kokkos::View<Weight*, typename ExecutionSpace::memory_space> hist) {
  const std::size_t n = subset.extent(0);
  const double width = cur_hi - cur_lo;
  const double dx = width / static_cast<double>(K);
  const double inv_dx = 1.0 / dx;
  Kokkos::parallel_for(
      "fuse::partition::accumulate_histogram", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t k) {
        const std::size_t gi = subset(k);
        const double c = centroids(gi, axis);
        const Weight w = weights(gi);
        if (c < cur_lo) {
          Kokkos::atomic_add(&hist(0), w);
        } else if (c <= cur_hi) {
          const double delta = c - cur_lo;       // step 1: subtract
          const double scaled = delta * inv_dx;  // step 2: multiply -- no FMA pattern present
          int bucket = static_cast<int>(scaled);
          if (bucket < 0) bucket = 0;
          if (bucket >= K) bucket = K - 1;
          Kokkos::atomic_add(&hist(1 + bucket), w);
        }
      });
}

}  // namespace detail

/// `centroids`/`weights` are the FULL arrays; `subset` names the indices
/// (into those arrays) this recursion frame is splitting, and `axis` the
/// coordinate to split on. COLLECTIVE over `comm` (see
/// detail/collectives.hpp): every rank must call this in lockstep,
/// including one whose `subset` is empty.
template <class ExecutionSpace>
MedianResult weighted_median(
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    Kokkos::View<const Weight*, typename ExecutionSpace::memory_space> weights,
    Kokkos::View<const std::size_t*, typename ExecutionSpace::memory_space> subset, int axis,
    CommHandle comm, const MedianOptions& opts = {}) {
  MedianResult result;

  const Weight local_total = detail::local_weight_sum<ExecutionSpace>(weights, subset);
  Weight total = 0;
  detail::allreduce_sum_i64(&local_total, &total, 1, comm);
  result.weight_total = total;
  if (total <= 0) return result;  // nothing to split

  double local_lo = 0.0, local_hi = 0.0;
  detail::local_axis_range<ExecutionSpace>(centroids, subset, axis, local_lo, local_hi);
  double lo = 0.0, hi = 0.0;
  detail::allreduce_min_f64(&local_lo, &lo, 1, comm);
  detail::allreduce_max_f64(&local_hi, &hi, 1, comm);
  if (!std::isfinite(lo) || !std::isfinite(hi)) {
    lo = 0.0;
    hi = 0.0;
  }

  if (!(hi > lo)) {
    // Degenerate: every element (globally) shares one coordinate value (or
    // there is no input at all). The split value equals that coordinate;
    // every element becomes a tie, distributed by GID -- see the caller.
    result.split_value = lo;
    result.weight_below = 0;
    result.weight_at_or_below = total;
    result.bucket_lo = lo;
    result.bucket_hi = lo;
    return result;
  }

  const double initial_range = hi - lo;
  const Weight target =
      static_cast<Weight>(std::llround(opts.frac_target * static_cast<double>(total)));
  const int K = std::max(2, opts.num_buckets);

  Kokkos::View<Weight*, typename ExecutionSpace::memory_space> hist(
      "fuse::partition::median_hist", static_cast<std::size_t>(K + 1));
  std::vector<Weight> host_hist(static_cast<std::size_t>(K + 1), 0);
  std::vector<Weight> global_hist(static_cast<std::size_t>(K + 1), 0);

  double cur_lo = lo;
  double cur_hi = hi;
  Weight last_chosen_weight = std::numeric_limits<Weight>::max();
  int iter = 0;
  for (; iter < opts.refine_iterations; ++iter) {
    const double width = cur_hi - cur_lo;
    if (width <= opts.geom_tol_relative * initial_range) break;
    if (iter > 0 && last_chosen_weight <= 1) break;

    Kokkos::deep_copy(hist, Weight{0});
    detail::accumulate_histogram<ExecutionSpace>(centroids, weights, subset, axis, cur_lo, cur_hi,
                                                 K, hist);
    auto host_mirror = Kokkos::create_mirror_view(hist);
    Kokkos::deep_copy(host_mirror, hist);
    for (int s = 0; s < K + 1; ++s) host_hist[static_cast<std::size_t>(s)] = host_mirror(s);

    detail::allreduce_sum_i64(host_hist.data(), global_hist.data(), K + 1, comm);

    const Weight prefix_below_segment = global_hist[0];
    Weight running = prefix_below_segment;
    int chosen = K - 1;
    Weight running_before_chosen = running;
    for (int b = 0; b < K; ++b) {
      const Weight after = running + global_hist[static_cast<std::size_t>(1 + b)];
      if (after >= target) {
        chosen = b;
        running_before_chosen = running;
        break;
      }
      running = after;
    }
    last_chosen_weight = global_hist[static_cast<std::size_t>(1 + chosen)];
    const double dx = width / static_cast<double>(K);
    // Host-only scalar update -- see the header comment on FMA above for
    // why this is the one place `a + b*c` was allowed to stay.
    const double new_lo = cur_lo + static_cast<double>(chosen) * dx;
    const double new_hi = cur_lo + static_cast<double>(chosen + 1) * dx;
    cur_lo = new_lo;
    cur_hi = new_hi;

    result.split_value = 0.5 * (cur_lo + cur_hi);
    result.weight_below = running_before_chosen;
    result.weight_at_or_below =
        running_before_chosen + global_hist[static_cast<std::size_t>(1 + chosen)];
    result.bucket_lo = cur_lo;
    result.bucket_hi = cur_hi;
  }

  if (iter == 0) {
    result.split_value = 0.5 * (cur_lo + cur_hi);
    result.weight_below = 0;
    result.weight_at_or_below = total;
    result.bucket_lo = cur_lo;
    result.bucket_hi = cur_hi;
  }
  result.iterations_used = iter;
  return result;
}

}  // namespace fuse::partition
