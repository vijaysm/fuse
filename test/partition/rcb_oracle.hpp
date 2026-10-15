// test/partition/rcb_oracle.hpp -- serial reference for fuse::partition's RCB.
//
// TEST CODE, NOT LIBRARY CODE (docs/dev/testing.md, the-five-states: this is
// what "Verified" is checked against, not the thing being verified). Kept
// the way fuse/num/kernels.hpp's serial form is kept as the oracle for its
// Kokkos kernels.
//
// This is a line-for-line port of two files from the user's rcb_moab
// (/Users/mahadevan/Code/rcb_moab):
//   * src/RCBPartitioner.cpp   recursive_bisect_elements  (the recursion,
//     axis selection, tie classification and tie distribution)
//   * src/ParallelMedian.cpp   weighted_median            (the histogram-
//     refined parallel weighted median)
//
// Every arithmetic expression, iteration order, and tie-break rule below is
// copied verbatim, including `std::llround` and the exact order operations
// happen in -- bit-for-bit parity between this oracle and fuse::partition's
// Kokkos port depends on it (docs/dev/partitioning.md "Determinism").
//
// THE ONLY THING THAT DIFFERS is the communication substrate. rcb_moab runs
// one real MPI rank per process and calls MPI_Allreduce / MPI_Allgatherv.
// This oracle runs P *emulated* ranks in one process: the caller partitions
// a GLOBAL index space into P "owner" slices (RankSlice), and every
// collective in the original is replaced by an explicit reduction over the
// P slices' local values (`detail::reduce_sum` / `reduce_min` / `reduce_max`,
// and rank-ordered concatenation for the Allgatherv). Summing int64 weights
// this way is exactly what MPI_SUM does (int64 addition is associative --
// no reassociation hazard), and MIN/MAX over doubles is likewise
// order-independent, so the emulation is not an approximation: it computes
// the identical answer rcb_moab's real Allreduce would, for any P.
//
// This is what makes RcbBfBAcrossRanks::* testable without a working MPI
// launcher (this sandbox cannot start one -- see docs/dev/testing.md
// "Host-verified parallel results"): the SAME oracle, called with different
// emulated-P/owner partitions of the SAME global input, either agrees with
// itself (a property test, no MPI or Kokkos involved at all) or serves as
// the P=1 ground truth that a real MPI run's gathered input is checked
// against on the host.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace fuse::partition::testing {

using OracleWeight = std::int64_t;
using OraclePoint3 = std::array<double, 3>;

// -------------------------------------------------------------- median

struct OracleMedianOptions {
  int num_buckets = 128;
  int refine_iterations = 4;
  double frac_target = 0.5;
  double geom_tol_relative = 1e-9;
};

struct OracleMedianResult {
  double split_value = 0.0;
  OracleWeight weight_below = 0;
  OracleWeight weight_at_or_below = 0;
  OracleWeight weight_total = 0;
  int iterations_used = 0;
  // Final histogram bucket bounds -- the "tie region"; see rcb_moab's
  // MedianResult doc for why classification uses these, not strict
  // comparison against split_value.
  double bucket_lo = 0.0;
  double bucket_hi = 0.0;
};

namespace detail {

// Explicit stand-ins for MPI_Allreduce(..., MPI_SUM/MPI_MIN/MPI_MAX, comm)
// over P emulated ranks' local scalars, summed/compared rank 0..P-1 in
// order. int64 SUM is associative (bit-exact regardless of grouping);
// double MIN/MAX are likewise order-independent -- so this reduction order
// is not a choice that could disagree with a real Allreduce's answer.
inline OracleWeight reduce_sum(const std::vector<OracleWeight>& per_rank) {
  OracleWeight total = 0;
  for (OracleWeight w : per_rank) total += w;
  return total;
}

inline double reduce_min(const std::vector<double>& per_rank) {
  double m = std::numeric_limits<double>::infinity();
  for (double v : per_rank) {
    if (v < m) m = v;
  }
  return m;
}

inline double reduce_max(const std::vector<double>& per_rank) {
  double m = -std::numeric_limits<double>::infinity();
  for (double v : per_rank) {
    if (v > m) m = v;
  }
  return m;
}

}  // namespace detail

// Emulated-P port of rcb_moab/ParallelMedian.cpp::weighted_median.
// `coords_per_rank[r]` / `weights_per_rank[r]` are emulated rank r's LOCAL
// arrays (same length per rank, r in [0, P)).
inline OracleMedianResult weighted_median_emulated(
    const std::vector<std::vector<double>>& coords_per_rank,
    const std::vector<std::vector<OracleWeight>>& weights_per_rank,
    const OracleMedianOptions& opts = {}) {
  const std::size_t P = coords_per_rank.size();
  OracleMedianResult result;

  std::vector<OracleWeight> local_totals(P, 0);
  for (std::size_t r = 0; r < P; ++r) {
    for (OracleWeight w : weights_per_rank[r]) local_totals[r] += w;
  }
  const OracleWeight total = detail::reduce_sum(local_totals);
  result.weight_total = total;
  if (total <= 0) return result;  // nothing to split

  std::vector<double> local_lo(P, std::numeric_limits<double>::infinity());
  std::vector<double> local_hi(P, -std::numeric_limits<double>::infinity());
  for (std::size_t r = 0; r < P; ++r) {
    for (double c : coords_per_rank[r]) {
      if (c < local_lo[r]) local_lo[r] = c;
      if (c > local_hi[r]) local_hi[r] = c;
    }
  }
  double lo = detail::reduce_min(local_lo);
  double hi = detail::reduce_max(local_hi);
  if (!std::isfinite(lo) || !std::isfinite(hi)) {
    lo = 0.0;
    hi = 0.0;
  }

  if (!(hi > lo)) {
    // Degenerate: all coordinates identical (or no input at all). Every
    // cell becomes a tie, distributed by GID -- see the caller.
    result.split_value = lo;
    result.weight_below = 0;
    result.weight_at_or_below = total;
    result.bucket_lo = lo;
    result.bucket_hi = lo;
    return result;
  }

  const double initial_range = hi - lo;
  const OracleWeight target = static_cast<OracleWeight>(
      std::llround(opts.frac_target * static_cast<double>(total)));
  const int K = std::max(2, opts.num_buckets);

  double cur_lo = lo;
  double cur_hi = hi;
  OracleWeight last_chosen_weight = std::numeric_limits<OracleWeight>::max();
  int iter = 0;
  for (; iter < opts.refine_iterations; ++iter) {
    const double width = cur_hi - cur_lo;
    if (width <= opts.geom_tol_relative * initial_range) break;
    if (iter > 0 && last_chosen_weight <= 1) break;

    const double dx = width / static_cast<double>(K);
    const double inv_dx = 1.0 / dx;

    // local_combined[r] = [prefix_below, hist[0..K-1]] for emulated rank r,
    // exactly rcb_moab's per-rank `local_combined` buffer.
    std::vector<std::vector<OracleWeight>> local_combined(
        P, std::vector<OracleWeight>(static_cast<std::size_t>(K + 1), 0));
    for (std::size_t r = 0; r < P; ++r) {
      auto& buf = local_combined[r];
      const auto& coords = coords_per_rank[r];
      const auto& weights = weights_per_rank[r];
      for (std::size_t i = 0; i < coords.size(); ++i) {
        const double c = coords[i];
        const OracleWeight w = weights[i];
        if (c < cur_lo) {
          buf[0] += w;
        } else if (c <= cur_hi) {
          int bucket = static_cast<int>((c - cur_lo) * inv_dx);
          if (bucket < 0) bucket = 0;
          if (bucket >= K) bucket = K - 1;
          buf[static_cast<std::size_t>(1 + bucket)] += w;
        }
      }
    }

    std::vector<OracleWeight> global_combined(static_cast<std::size_t>(K + 1), 0);
    for (int slot = 0; slot < K + 1; ++slot) {
      std::vector<OracleWeight> per_rank(P, 0);
      for (std::size_t r = 0; r < P; ++r) {
        per_rank[r] = local_combined[r][static_cast<std::size_t>(slot)];
      }
      global_combined[static_cast<std::size_t>(slot)] = detail::reduce_sum(per_rank);
    }

    const OracleWeight prefix_below_segment = global_combined[0];
    OracleWeight running = prefix_below_segment;
    int chosen = K - 1;
    OracleWeight running_before_chosen = running;
    for (int b = 0; b < K; ++b) {
      const OracleWeight after =
          running + global_combined[static_cast<std::size_t>(1 + b)];
      if (after >= target) {
        chosen = b;
        running_before_chosen = running;
        break;
      }
      running = after;
    }
    last_chosen_weight = global_combined[static_cast<std::size_t>(1 + chosen)];
    const double new_lo = cur_lo + static_cast<double>(chosen) * dx;
    const double new_hi = cur_lo + static_cast<double>(chosen + 1) * dx;
    cur_lo = new_lo;
    cur_hi = new_hi;

    result.split_value = 0.5 * (cur_lo + cur_hi);
    result.weight_below = running_before_chosen;
    result.weight_at_or_below =
        running_before_chosen + global_combined[static_cast<std::size_t>(1 + chosen)];
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

// ----------------------------------------------------------- recursion

namespace detail {

// Port of rcb_moab/RCBPartitioner.cpp::recursive_bisect_elements. `indices`
// are GLOBAL indices still "in play"; `owner[i]` says which emulated rank
// holds global index i, so we re-derive each rank's local slice of
// `indices` at every level rather than carrying per-rank state down.
inline void recursive_bisect_oracle_impl(
    const std::vector<OraclePoint3>& centroids, const std::vector<OracleWeight>& weights,
    const std::vector<std::int64_t>& gids, const std::vector<int>& owner, int P,
    std::vector<int>& assignment_out, const std::vector<std::size_t>& indices, int part_lo,
    int part_hi, int depth, const OracleMedianOptions& opts) {
  // Base case only -- NEVER `indices.empty()`. See rcb_moab's CLAUDE.md
  // "Global collective discipline": every emulated rank must reach every
  // reduction below, including one contributing zero elements.
  if (part_hi - part_lo <= 1) {
    for (auto i : indices) assignment_out[i] = part_lo;
    return;
  }

  std::vector<std::vector<std::size_t>> local_idx(static_cast<std::size_t>(P));
  for (auto i : indices) local_idx[static_cast<std::size_t>(owner[i])].push_back(i);

  // Axis selection: longest current global extent, depth%3 fallback.
  std::array<std::vector<double>, 3> lo_per_rank, hi_per_rank;
  for (int a = 0; a < 3; ++a) {
    lo_per_rank[static_cast<std::size_t>(a)].assign(
        static_cast<std::size_t>(P), std::numeric_limits<double>::infinity());
    hi_per_rank[static_cast<std::size_t>(a)].assign(
        static_cast<std::size_t>(P), -std::numeric_limits<double>::infinity());
  }
  for (int r = 0; r < P; ++r) {
    for (auto i : local_idx[static_cast<std::size_t>(r)]) {
      for (int a = 0; a < 3; ++a) {
        const double c = centroids[i][static_cast<std::size_t>(a)];
        auto& lo_a = lo_per_rank[static_cast<std::size_t>(a)];
        auto& hi_a = hi_per_rank[static_cast<std::size_t>(a)];
        if (c < lo_a[static_cast<std::size_t>(r)]) lo_a[static_cast<std::size_t>(r)] = c;
        if (c > hi_a[static_cast<std::size_t>(r)]) hi_a[static_cast<std::size_t>(r)] = c;
      }
    }
  }
  std::array<double, 3> global_lo{}, global_hi{};
  for (int a = 0; a < 3; ++a) {
    global_lo[static_cast<std::size_t>(a)] = reduce_min(lo_per_rank[static_cast<std::size_t>(a)]);
    global_hi[static_cast<std::size_t>(a)] = reduce_max(hi_per_rank[static_cast<std::size_t>(a)]);
  }
  int axis = 0;
  double best_extent = global_hi[0] - global_lo[0];
  for (int a = 1; a < 3; ++a) {
    const double e = global_hi[static_cast<std::size_t>(a)] - global_lo[static_cast<std::size_t>(a)];
    if (e > best_extent) {
      best_extent = e;
      axis = a;
    }
  }
  if (!(best_extent > 0.0)) axis = depth % 3;

  const int part_mid_count = (part_hi - part_lo + 1) / 2;
  const double frac =
      static_cast<double>(part_mid_count) / static_cast<double>(part_hi - part_lo);

  std::vector<std::vector<double>> coords_per_rank(static_cast<std::size_t>(P));
  std::vector<std::vector<OracleWeight>> weights_per_rank(static_cast<std::size_t>(P));
  for (int r = 0; r < P; ++r) {
    auto& idx = local_idx[static_cast<std::size_t>(r)];
    coords_per_rank[static_cast<std::size_t>(r)].reserve(idx.size());
    weights_per_rank[static_cast<std::size_t>(r)].reserve(idx.size());
    for (auto i : idx) {
      coords_per_rank[static_cast<std::size_t>(r)].push_back(
          centroids[i][static_cast<std::size_t>(axis)]);
      weights_per_rank[static_cast<std::size_t>(r)].push_back(weights[i]);
    }
  }

  OracleMedianOptions mopts = opts;
  mopts.frac_target = frac;
  const OracleMedianResult med = weighted_median_emulated(coords_per_rank, weights_per_rank, mopts);

  std::vector<std::vector<std::size_t>> left_per_rank(static_cast<std::size_t>(P)),
      right_per_rank(static_cast<std::size_t>(P)), onsplit_per_rank(static_cast<std::size_t>(P));
  for (int r = 0; r < P; ++r) {
    for (auto i : local_idx[static_cast<std::size_t>(r)]) {
      const double c = centroids[i][static_cast<std::size_t>(axis)];
      if (c < med.bucket_lo) {
        left_per_rank[static_cast<std::size_t>(r)].push_back(i);
      } else if (c > med.bucket_hi) {
        right_per_rank[static_cast<std::size_t>(r)].push_back(i);
      } else {
        onsplit_per_rank[static_cast<std::size_t>(r)].push_back(i);
      }
    }
  }

  // rcb_moab sorts on_split by GID locally before building the per-rank
  // Allgatherv payload. PS-9 correction (2026-10-11 review): the comment
  // this replaced claimed the subsequent GLOBAL sort by GID made this local
  // pre-sort "provably not affect the final `goes_left` decision" -- true
  // only when every gid is unique. `rcb_partition` does NOT require unique
  // gids, and when two on-split entries share one, this rank's local order
  // between them decides which "gathered position" each receives, which
  // the global sort's weight walk can then split across the cutoff
  // differently depending on that order. `std::stable_sort` (not
  // `std::sort`) makes the order TOTAL -- gid, then this rank's own
  // position before sorting -- matching rcb.hpp's identical fix exactly,
  // which is what makes bit-parity against this oracle hold with duplicate
  // gids present, not only in their absence.
  std::vector<OracleWeight> tie_w_per_rank(static_cast<std::size_t>(P), 0);
  for (int r = 0; r < P; ++r) {
    std::stable_sort(onsplit_per_rank[static_cast<std::size_t>(r)].begin(),
                     onsplit_per_rank[static_cast<std::size_t>(r)].end(),
                     [&](std::size_t a, std::size_t b) { return gids[a] < gids[b]; });
    for (auto i : onsplit_per_rank[static_cast<std::size_t>(r)]) {
      tie_w_per_rank[static_cast<std::size_t>(r)] += weights[i];
    }
  }
  const OracleWeight global_tie_w = reduce_sum(tie_w_per_rank);
  const OracleWeight target_left_w =
      static_cast<OracleWeight>(std::llround(frac * static_cast<double>(med.weight_total)));
  const OracleWeight need_from_ties =
      target_left_w > med.weight_below ? target_left_w - med.weight_below : 0;

  if (need_from_ties > 0 && global_tie_w > 0) {
    std::vector<std::int64_t> all_gids;
    std::vector<OracleWeight> all_ws;
    std::vector<std::size_t> all_idx;
    for (int r = 0; r < P; ++r) {
      for (auto i : onsplit_per_rank[static_cast<std::size_t>(r)]) {
        all_gids.push_back(gids[i]);
        all_ws.push_back(weights[i]);
        all_idx.push_back(i);
      }
    }
    const int total = static_cast<int>(all_gids.size());
    std::vector<int> order(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) order[static_cast<std::size_t>(i)] = i;
    // `std::stable_sort`: total order (gid, then gathered position --
    // `order`'s own initial value) -- rcb.hpp's identical fix, for the
    // identical reason.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return all_gids[static_cast<std::size_t>(a)] < all_gids[static_cast<std::size_t>(b)];
    });

    OracleWeight cum = 0;
    std::vector<bool> goes_left(static_cast<std::size_t>(total), false);
    for (int idx : order) {
      const OracleWeight w = all_ws[static_cast<std::size_t>(idx)];
      if (cum + w <= need_from_ties) {
        goes_left[static_cast<std::size_t>(idx)] = true;
        cum += w;
      } else {
        break;
      }
    }
    for (int k = 0; k < total; ++k) {
      const std::size_t i = all_idx[static_cast<std::size_t>(k)];
      const int r = owner[i];
      if (goes_left[static_cast<std::size_t>(k)]) {
        left_per_rank[static_cast<std::size_t>(r)].push_back(i);
      } else {
        right_per_rank[static_cast<std::size_t>(r)].push_back(i);
      }
    }
  } else {
    for (int r = 0; r < P; ++r) {
      for (auto i : onsplit_per_rank[static_cast<std::size_t>(r)]) {
        right_per_rank[static_cast<std::size_t>(r)].push_back(i);
      }
    }
  }

  std::vector<std::size_t> left, right;
  for (int r = 0; r < P; ++r) {
    for (auto i : left_per_rank[static_cast<std::size_t>(r)]) left.push_back(i);
    for (auto i : right_per_rank[static_cast<std::size_t>(r)]) right.push_back(i);
  }

  const int part_mid = part_lo + part_mid_count;
  recursive_bisect_oracle_impl(centroids, weights, gids, owner, P, assignment_out, left, part_lo,
                              part_mid, depth + 1, opts);
  recursive_bisect_oracle_impl(centroids, weights, gids, owner, P, assignment_out, right, part_mid,
                              part_hi, depth + 1, opts);
}

}  // namespace detail

// Top-level oracle entry. `owner[i]` in [0, P) assigns each global element
// to an emulated rank; P is inferred as `max(owner) + 1`. `num_parts` may be
// anything >= 1, independent of P -- exactly rcb_moab's
// "--num-parts != comm_size" contract.
inline std::vector<int> recursive_bisect_oracle(const std::vector<OraclePoint3>& centroids,
                                                const std::vector<OracleWeight>& weights,
                                                const std::vector<std::int64_t>& gids,
                                                const std::vector<int>& owner, int num_parts,
                                                const OracleMedianOptions& opts = {}) {
  const std::size_t n = centroids.size();
  std::vector<int> assignment(n, 0);
  if (n == 0 || num_parts <= 0) return assignment;

  int P = 1;
  for (int o : owner) P = std::max(P, o + 1);

  std::vector<std::size_t> all_indices(n);
  for (std::size_t i = 0; i < n; ++i) all_indices[i] = i;

  detail::recursive_bisect_oracle_impl(centroids, weights, gids, owner, P, assignment, all_indices,
                                       0, num_parts, 0, opts);
  return assignment;
}

// Convenience owner-assignment: element i belongs to emulated rank
// `i * P / n` -- P contiguous, near-equal blocks by global index. Good
// enough for a fixture test that only needs "P is genuinely > 1", not a
// specific ownership pattern.
inline std::vector<int> owner_by_contiguous_blocks(std::size_t n, int P) {
  std::vector<int> owner(n, 0);
  if (P <= 1 || n == 0) return owner;
  for (std::size_t i = 0; i < n; ++i) {
    int r = static_cast<int>((i * static_cast<std::size_t>(P)) / n);
    if (r >= P) r = P - 1;
    owner[i] = r;
  }
  return owner;
}

}  // namespace fuse::partition::testing
