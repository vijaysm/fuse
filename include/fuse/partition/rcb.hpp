#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/partition/coord_transform.hpp"
#include "fuse/partition/detail/collectives.hpp"
#include "fuse/partition/median.hpp"
#include "fuse/partition/types.hpp"

/// fuse::partition::rcb_partition -- Kokkos port of rcb_moab's
/// RCBPartitioner::single_level_rcb + recursive_bisect_elements. A FAITHFUL
/// port: the same recursion, the same collective sequence per level (axis
/// extent, weighted median, tie weight, tie Allgatherv), the same base case
/// (`part_hi - part_lo <= 1`, checked before anything else -- never
/// `indices.empty()`), and the same tie-break rule (classify by the
/// median's bucket bounds, sort ties by global id, distribute by a global
/// weight-ordered walk). It is deliberately NOT restructured into
/// level-synchronous batching; see docs/dev/partitioning.md.
///
/// GENERALITY. `centroids` is three doubles per element and nothing else --
/// there is no cell-type or topological-dimension parameter anywhere in
/// this file. A 1D line, a 2D planar or manifold mesh (a sphere is one
/// case, via CoordTransform::Gnomonic), and a 3D volumetric mesh of any
/// cell type (hex, tet, prism, pyramid, polyhedral, or mixed) all reach
/// this function through the same call. Where a caller's data comes from a
/// `fuse::MeshView`, `mesh_adapter.hpp`'s `centroids_of` produces the
/// `centroids`/`global_ids` this function wants; per-element weights come
/// from the caller (they are not derived from the mesh view -- see
/// docs/dev/partitioning.md).
namespace fuse::partition {

struct RcbOptions {
  /// Number of partitions. 0 (default) means "use the communicator's size"
  /// (1 when `comm == kNullComm`) -- rcb_moab's `RCBOptions::num_parts`
  /// convention. May be set to any value >= 1, independent of the
  /// communicator size (rcb_moab's "--num-parts != comm_size" contract);
  /// this function only ever produces an ASSIGNMENT, so there is no
  /// migration step here that would need parts and ranks to correspond.
  ///
  /// PRECONDITION when `comm != kNullComm`: `num_parts` (as given here,
  /// before the 0 -> comm-size resolution), `median_buckets` and
  /// `median_refine_iters` below must be IDENTICAL on every rank of `comm`.
  /// They decide the SHAPE of `detail::recursive_bisect`'s recursion --
  /// how many levels, how many histogram Allreduces per level -- not just a
  /// value every collective below reduces away; two ranks with
  /// different-but-individually-valid options would otherwise execute a
  /// different number or kind of collectives at the very first level.
  /// `rcb_partition` agrees all three with an `MPI_Allreduce` (`MAX` and
  /// `MIN`, so disagreement is detectable) before touching
  /// `recursive_bisect` and refuses `E-PART-OPTS` on disagreement.
  int num_parts = 0;

  int median_buckets = kDefaultHistogramBuckets;
  int median_refine_iters = kDefaultHistogramRefineIters;

  /// Optional pre-bisection transform. Identity by default; see
  /// coord_transform.hpp.
  CoordTransform coord_transform = CoordTransform::Identity;
  GnomonicOptions gnomonic;

  /// See types.hpp. kNullComm (default) means the base build configuration: exactly one
  /// process, no MPI touched.
  CommHandle comm = kNullComm;
};

template <class ExecutionSpace>
struct RcbResult {
  using MemorySpace = typename ExecutionSpace::memory_space;

  /// Per-element assigned part id, in [0, num_parts_used); same order and
  /// length as the caller's `centroids`.
  Kokkos::View<int*, MemorySpace> part_id;
  int num_parts_used = 0;
};

namespace detail {

/// Uploads a host index list to `ExecutionSpace`'s memory space. Recursion
/// bookkeeping (which global indices are still "in play" at this frame)
/// stays on the host as a `std::vector` -- the recursion tree has depth
/// O(log2(num_parts)), and each frame's subset is usually much smaller than
/// the one above it, so this is cheap relative to the O(subset size)
/// device kernels (bounding box, histogram, classification) that follow.
template <class ExecutionSpace>
Kokkos::View<std::size_t*, typename ExecutionSpace::memory_space> upload_subset(
    const std::vector<std::size_t>& indices) {
  using MemorySpace = typename ExecutionSpace::memory_space;
  Kokkos::View<std::size_t*, MemorySpace> dev("fuse::partition::subset", indices.size());
  Kokkos::View<const std::size_t*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host_view(indices.empty() ? nullptr : indices.data(), indices.size());
  Kokkos::deep_copy(dev, host_view);
  return dev;
}

/// Classification kernel: tag(k) = 0 (left) / 1 (right) / 2 (tie, "on the
/// split") for each subset(k), comparing centroids(subset(k), axis) against
/// the median's bucket bounds. Pure comparisons -- no arithmetic an FMA
/// could perturb.
template <class ExecutionSpace>
void classify(Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
             Kokkos::View<const std::size_t*, typename ExecutionSpace::memory_space> subset,
             int axis, double bucket_lo, double bucket_hi,
             Kokkos::View<int*, typename ExecutionSpace::memory_space> tag) {
  const std::size_t n = subset.extent(0);
  Kokkos::parallel_for(
      "fuse::partition::classify", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t k) {
        const double c = centroids(subset(k), axis);
        tag(k) = (c < bucket_lo) ? 0 : (c > bucket_hi ? 1 : 2);
      });
}

/// The recursion. `indices` are GLOBAL indices still in play at this frame;
/// `weights_host`/`gids_host` are full-length host mirrors, read for the
/// tie bookkeeping below (small scalar work, deliberately host-side --
/// docs/dev/partitioning.md).
template <class ExecutionSpace>
void recursive_bisect(
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    Kokkos::View<const Weight*, typename ExecutionSpace::memory_space> weights,
    const std::vector<Weight>& weights_host, const std::vector<std::int64_t>& gids_host,
    Kokkos::View<int*, typename ExecutionSpace::memory_space> part_id,
    const std::vector<std::size_t>& indices, int part_lo, int part_hi, int depth,
    CommHandle comm, const RcbOptions& opts) {
  // Base case ONLY -- never `indices.empty()`. `part_lo`/`part_hi` are
  // identical on every rank at this depth (computed only from depth and
  // num_parts), so this is the one safe early exit; an empty-subset early
  // return here would desynchronise the collectives below on the very next
  // level whenever ranks disagree on which subtrees are empty. See
  // rcb_moab's CLAUDE.md "Global collective discipline" -- commit 3bbbbe6
  // there fixed exactly this deadlock once.
  if (part_hi - part_lo <= 1) {
    auto dev_indices = upload_subset<ExecutionSpace>(indices);
    Kokkos::parallel_for(
        "fuse::partition::assign_leaf", Kokkos::RangePolicy<ExecutionSpace>(0, dev_indices.extent(0)),
        KOKKOS_LAMBDA(const std::size_t k) { part_id(dev_indices(k)) = part_lo; });
    return;
  }

  auto dev_indices = upload_subset<ExecutionSpace>(indices);

  // Axis selection: longest current global extent; depth%3 fallback when
  // every axis is degenerate (all points coincide, or there is no input
  // anywhere).
  double local_lo[3], local_hi[3];
  for (int a = 0; a < 3; ++a) {
    local_axis_range<ExecutionSpace>(centroids, dev_indices, a, local_lo[a], local_hi[a]);
  }
  double global_lo[3], global_hi[3];
  allreduce_min_f64(local_lo, global_lo, 3, comm);
  allreduce_max_f64(local_hi, global_hi, 3, comm);

  int axis = 0;
  double best_extent = global_hi[0] - global_lo[0];
  for (int a = 1; a < 3; ++a) {
    const double e = global_hi[a] - global_lo[a];
    if (e > best_extent) {
      best_extent = e;
      axis = a;
    }
  }
  if (!(best_extent > 0.0)) axis = depth % 3;

  const int part_mid_count = (part_hi - part_lo + 1) / 2;
  const double frac = static_cast<double>(part_mid_count) / static_cast<double>(part_hi - part_lo);

  MedianOptions mopts;
  mopts.frac_target = frac;
  mopts.num_buckets = opts.median_buckets;
  mopts.refine_iterations = opts.median_refine_iters;
  const MedianResult med =
      weighted_median<ExecutionSpace>(centroids, weights, dev_indices, axis, comm, mopts);

  Kokkos::View<int*, typename ExecutionSpace::memory_space> tag("fuse::partition::tag",
                                                                dev_indices.extent(0));
  classify<ExecutionSpace>(centroids, dev_indices, axis, med.bucket_lo, med.bucket_hi, tag);
  auto tag_host = Kokkos::create_mirror_view(tag);
  Kokkos::deep_copy(tag_host, tag);

  std::vector<std::size_t> left, right, on_split;
  left.reserve(indices.size());
  right.reserve(indices.size());
  for (std::size_t k = 0; k < indices.size(); ++k) {
    const std::size_t gi = indices[k];
    const int t = tag_host(k);
    if (t == 0) {
      left.push_back(gi);
    } else if (t == 1) {
      right.push_back(gi);
    } else {
      on_split.push_back(gi);
    }
  }

  // Sort on-split entries by GID for deterministic placement (rcb_moab
  // RCBPartitioner.cpp:297). `rcb_partition` does not require unique
  // global ids (docs/dev/partitioning.md "Ties"), so a comparator keyed on
  // gid ALONE leaves the order among entries that share one gid
  // unspecified -- `std::stable_sort`, not `std::sort`, makes that order
  // TOTAL (gid, then this rank's own position before sorting) rather than
  // whatever an unstable sort's internal algorithm happens to produce.
  // This local order is read below as "this rank's k-th on-split entry
  // occupies gathered position offset_gids + k", so an unspecified local
  // order would make an unspecified GLOBAL order too.
  std::stable_sort(on_split.begin(), on_split.end(),
                   [&](std::size_t a, std::size_t b) { return gids_host[a] < gids_host[b]; });

  Weight local_tie_w = 0;
  for (auto i : on_split) local_tie_w += weights_host[i];
  Weight global_tie_w = 0;
  allreduce_sum_i64(&local_tie_w, &global_tie_w, 1, comm);

  const Weight target_left_w =
      static_cast<Weight>(std::llround(frac * static_cast<double>(med.weight_total)));
  const Weight need_from_ties =
      target_left_w > med.weight_below ? target_left_w - med.weight_below : 0;

  if (need_from_ties > 0 && global_tie_w > 0) {
    std::vector<std::int64_t> local_gids, local_ws;
    local_gids.reserve(on_split.size());
    local_ws.reserve(on_split.size());
    for (auto i : on_split) {
      local_gids.push_back(gids_host[i]);
      local_ws.push_back(weights_host[i]);
    }

    std::vector<std::int64_t> all_gids, all_ws;
    int offset_gids = 0, offset_ws = 0;
    allgatherv_i64(local_gids.data(), static_cast<int>(local_gids.size()), all_gids, offset_gids,
                  comm);
    allgatherv_i64(local_ws.data(), static_cast<int>(local_ws.size()), all_ws, offset_ws, comm);
    FUSE_ASSERT(offset_gids == offset_ws,
               "allgatherv_i64 returned different offsets ({} vs {}) for two calls with the "
               "same local_count -- comm handling is inconsistent",
               offset_gids, offset_ws);

    const int total = static_cast<int>(all_gids.size());
    std::vector<int> order(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i) order[static_cast<std::size_t>(i)] = i;
    // `std::stable_sort`: a total order (gid, then GATHERED position --
    // `order`'s own initial value, preserved for ties by stability) rather
    // than an `std::sort`-unstable one, so two entries that share a gid are
    // assigned `goes_left` reproducibly instead of by an unspecified
    // tie-break (see the local pre-sort's comment above, and
    // docs/dev/partitioning.md "Ties").
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      return all_gids[static_cast<std::size_t>(a)] < all_gids[static_cast<std::size_t>(b)];
    });

    Weight cum = 0;
    std::vector<bool> goes_left(static_cast<std::size_t>(total), false);
    for (int idx : order) {
      const Weight w = all_ws[static_cast<std::size_t>(idx)];
      if (cum + w <= need_from_ties) {
        goes_left[static_cast<std::size_t>(idx)] = true;
        cum += w;
      } else {
        break;
      }
    }

    for (std::size_t k = 0; k < on_split.size(); ++k) {
      const int gi_in_gathered = offset_gids + static_cast<int>(k);
      if (goes_left[static_cast<std::size_t>(gi_in_gathered)]) {
        left.push_back(on_split[k]);
      } else {
        right.push_back(on_split[k]);
      }
    }
  } else {
    for (auto i : on_split) right.push_back(i);
  }

  const int part_mid = part_lo + part_mid_count;
  recursive_bisect<ExecutionSpace>(centroids, weights, weights_host, gids_host, part_id, left,
                                   part_lo, part_mid, depth + 1, comm, opts);
  recursive_bisect<ExecutionSpace>(centroids, weights, weights_host, gids_host, part_id, right,
                                   part_mid, part_hi, depth + 1, comm, opts);
}

}  // namespace detail

/// Entry point. Rejects rather than guesses on a malformed call
/// (E-PART-SIZE-MISMATCH, E-PART-NUM-PARTS, E-PART-NO-MPI, E-PART-WEIGHT,
/// E-PART-OPTS); every other degenerate case -- all
/// points identical, an empty local input, zero total weight, a single
/// part -- is handled by the SAME code path as the general case, not a
/// special branch (see docs/dev/partitioning.md).
///
/// RANK AGREEMENT. When `comm != kNullComm`, every one of the rejections
/// above is a rank-local computation but NEVER a rank-local RETURN: this
/// rank's own verdict (and, for E-PART-OPTS, the actual `num_parts`/
/// `median_buckets`/`median_refine_iters` it was given) is agreed with
/// every other rank in `comm` via `MPI_Allreduce` BEFORE
/// `detail::recursive_bisect`'s first collective -- the same
/// agree-before-collective recipe `fuse::native::migrate` and
/// `centroids_of` use. A caller's precondition bug on one rank must never
/// let that rank refuse alone while its peers enter a collective it
/// skipped (docs/dev/rank-agreement.md).
template <class ExecutionSpace>
RcbResult<ExecutionSpace> rcb_partition(
    const RcbOptions& opts,
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    Kokkos::View<const Weight*, typename ExecutionSpace::memory_space> weights,
    Kokkos::View<const std::int64_t*, typename ExecutionSpace::memory_space> gids) {
  using MemorySpace = typename ExecutionSpace::memory_space;
  const std::size_t n = centroids.extent(0);

#if !FUSE_HAVE_MPI
  if (opts.comm != kNullComm) {
    FUSE_REFUSE("E-PART-NO-MPI",
               "a non-null communicator was supplied but this build has no MPI "
               "(FUSE_HAVE_MPI=0); rebuild with -DFUSE_ENABLE_MPI=ON");
  }
#endif

  // Every precondition below is a RANK-LOCAL computation, decided before
  // any collective -- see the header comment on rank agreement. None of
  // them is returned on its own when `comm != kNullComm`.
  const bool locally_size_ok = (weights.extent(0) == n && gids.extent(0) == n);
  const bool locally_num_parts_ok = (opts.num_parts >= 0);

  // Negative weights are meaningless for load balance (they are not
  // refused by weighted_median's own degenerate-total handling, which
  // treats a non-positive SUM as "nothing to split" and sorts every
  // element to the right by the sign of an arbitrary coordinate). Checked
  // over the execution space the weights actually live in, not assumed to
  // be host-resident.
  int locally_weights_nonneg = 1;
  if (weights.extent(0) > 0) {
    // `Kokkos::parallel_reduce` with an EMPTY policy range writes the
    // reducer's own IDENTITY into the result (for `Kokkos::Max<int>`, a
    // large negative sentinel, not 0) regardless of what `any_negative`
    // was initialized to -- an empty `weights` is vacuously all
    // non-negative, so the reduce must not run at all in that case.
    int any_negative = 0;
    Kokkos::parallel_reduce(
        "fuse::partition::check_weights_nonnegative",
        Kokkos::RangePolicy<ExecutionSpace>(0, weights.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i, int& acc) {
          // `acc` starts each thread's local accumulation at Kokkos::Max<int>'s
          // IDENTITY value, not 0 -- an element that never sees a negative
          // weight must still explicitly combine 0 into `acc`, or every
          // non-negative-only run reduces to the identity (a large negative
          // sentinel, != 0) and is misread as "found a negative weight".
          const int bad = (weights(i) < Weight{0}) ? 1 : 0;
          if (bad > acc) acc = bad;
        },
        Kokkos::Max<int>(any_negative));
    locally_weights_nonneg = (any_negative == 0) ? 1 : 0;
  }
  const bool locally_ok = locally_size_ok && locally_num_parts_ok && (locally_weights_nonneg != 0);

  if (opts.comm == kNullComm) {
    // Exactly one process: nothing to agree with, so the local verdict IS
    // the global one -- refuse directly, in the order below.
    if (!locally_size_ok) {
      FUSE_REFUSE("E-PART-SIZE-MISMATCH",
                 "centroids/weights/gids must have the same length; got {} / {} / {}", n,
                 weights.extent(0), gids.extent(0));
    }
    if (!locally_num_parts_ok) {
      FUSE_REFUSE("E-PART-NUM-PARTS",
                 "num_parts must be >= 0 (0 = use the communicator size); got {}",
                 opts.num_parts);
    }
    if (!locally_weights_nonneg) {
      FUSE_REFUSE("E-PART-WEIGHT",
                 "every element weight must be >= 0 (a negative weight is not a load-balance "
                 "metric; weighted_median would place the split at an arbitrary coordinate)");
    }
  } else {
    // Agree this rank's own verdict, AND the recursion's own
    // shape -- num_parts, median_buckets, median_refine_iters -- with
    // every other rank in `comm` BEFORE `recursive_bisect`'s first
    // collective. One Allreduce(MAX) and one Allreduce(MIN) over four
    // int64s: index 0 is the combined local verdict (1 = this rank is
    // happy to proceed), 1..3 are the three options a caller must supply
    // identically.
    std::array<std::int64_t, 4> local_vals = {
        locally_ok ? std::int64_t{1} : std::int64_t{0}, static_cast<std::int64_t>(opts.num_parts),
        static_cast<std::int64_t>(opts.median_buckets),
        static_cast<std::int64_t>(opts.median_refine_iters)};
    std::array<std::int64_t, 4> max_vals{}, min_vals{};
    detail::allreduce_max_i64(local_vals.data(), max_vals.data(), 4, opts.comm);
    detail::allreduce_min_i64(local_vals.data(), min_vals.data(), 4, opts.comm);

    if (min_vals[0] == 0) {
      // At least one rank's own verdict was 0. A rank whose OWN verdict
      // was 0 names its own specific reason; every other rank refuses the
      // generic E-PART-OPTS so none is left waiting on the collective it
      // would otherwise have entered.
      if (!locally_size_ok) {
        FUSE_REFUSE("E-PART-SIZE-MISMATCH",
                   "centroids/weights/gids must have the same length; got {} / {} / {}", n,
                   weights.extent(0), gids.extent(0));
      }
      if (!locally_num_parts_ok) {
        FUSE_REFUSE("E-PART-NUM-PARTS",
                   "num_parts must be >= 0 (0 = use the communicator size); got {}",
                   opts.num_parts);
      }
      if (!locally_weights_nonneg) {
        FUSE_REFUSE("E-PART-WEIGHT",
                   "every element weight must be >= 0 (a negative weight is not a "
                   "load-balance metric; weighted_median would place the split at an "
                   "arbitrary coordinate)");
      }
      FUSE_REFUSE("E-PART-OPTS",
                 "rank {} is proceeding, but at least one other rank in this communicator "
                 "refused its own precondition on entry to rcb_partition; every rank refuses "
                 "together so none hangs waiting on recursive_bisect's first collective",
                 detail::comm_rank(opts.comm));
    }
    if (max_vals[1] != min_vals[1] || max_vals[2] != min_vals[2] || max_vals[3] != min_vals[3]) {
      FUSE_REFUSE("E-PART-OPTS",
                 "ranks disagree about RcbOptions (num_parts {}..{}, median_buckets {}..{}, "
                 "median_refine_iters {}..{} across the communicator); every rank's recursion "
                 "must agree the exact collective sequence it will execute",
                 min_vals[1], max_vals[1], min_vals[2], max_vals[2], min_vals[3], max_vals[3]);
    }
  }

  const int comm_size_val = detail::comm_size(opts.comm);
  const int num_parts = (opts.num_parts > 0) ? opts.num_parts : comm_size_val;
  if (num_parts <= 0) {
    FUSE_REFUSE("E-PART-NUM-PARTS", "effective num_parts resolved to {} (communicator size {})",
               num_parts, comm_size_val);
  }

  RcbResult<ExecutionSpace> result;
  result.part_id = Kokkos::View<int*, MemorySpace>("fuse::partition::part_id", n);
  result.num_parts_used = num_parts;

  auto transformed = apply_coord_transform<ExecutionSpace>(centroids, opts.coord_transform,
                                                            opts.gnomonic);

  // Host mirrors of weights/gids, read throughout the recursion for tie
  // bookkeeping. One copy, made once, reused at every level.
  auto weights_mirror = Kokkos::create_mirror_view(weights);
  Kokkos::deep_copy(weights_mirror, weights);
  auto gids_mirror = Kokkos::create_mirror_view(gids);
  Kokkos::deep_copy(gids_mirror, gids);
  std::vector<Weight> weights_host(n);
  std::vector<std::int64_t> gids_host(n);
  for (std::size_t i = 0; i < n; ++i) {
    weights_host[i] = weights_mirror(i);
    gids_host[i] = gids_mirror(i);
  }

  std::vector<std::size_t> all_indices(n);
  for (std::size_t i = 0; i < n; ++i) all_indices[i] = i;

  detail::recursive_bisect<ExecutionSpace>(transformed, weights, weights_host, gids_host,
                                           result.part_id, all_indices, 0, num_parts, 0, opts.comm,
                                           opts);
  return result;
}

}  // namespace fuse::partition
