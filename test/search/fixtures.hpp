// test/search/fixtures.hpp -- random box/point generators and a brute-force
// reference, so every property test compares fuse::search's answer against
// an INDEPENDENT computation of the same fuse::search::contains/padded
// rules (geometry.hpp), not a re-invented notion of containment.

#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "fuse/search/geometry.hpp"
#include "fuse/search/types.hpp"

namespace fuse::search::testing {

/// `n` axis-aligned boxes of side `~1` on a jittered lattice in `dim`
/// dimensions (1, 2 or 3), embedded in R^3 with unused coordinates exactly
/// zero (Point's convention). Deterministic (seeded), so a failing case is
/// reproducible without recording the random state.
inline std::vector<Box> random_boxes(int dim, std::size_t n, std::uint64_t seed = 12345) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> center(0.0, 10.0);
  std::uniform_real_distribution<double> half_extent(0.2, 0.6);
  std::vector<Box> boxes;
  boxes.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Point lo{}, hi{};
    const double cx = center(rng), ex = half_extent(rng);
    lo.x = cx - ex;
    hi.x = cx + ex;
    if (dim >= 2) {
      const double cy = center(rng), ey = half_extent(rng);
      lo.y = cy - ey;
      hi.y = cy + ey;
    }
    if (dim >= 3) {
      const double cz = center(rng), ez = half_extent(rng);
      lo.z = cz - ez;
      hi.z = cz + ez;
    }
    boxes.push_back(Box{lo, hi});
  }
  return boxes;
}

/// `n` random query points spanning roughly the same range `random_boxes`
/// covers, so a meaningful fraction actually land inside a box.
inline std::vector<Point> random_points(int dim, std::size_t n, std::uint64_t seed = 54321) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> coord(-1.0, 11.0);
  std::vector<Point> pts;
  pts.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Point p{};
    p.x = coord(rng);
    if (dim >= 2) p.y = coord(rng);
    if (dim >= 3) p.z = coord(rng);
    pts.push_back(p);
  }
  return pts;
}

/// The independent reference `box_index_test.cpp` checks `BoxIndex` against:
/// every box index whose (already-padded, if the caller padded) box
/// contains `p`, via `geometry.hpp::contains` -- the same predicate
/// `BoxIndex` itself is built to answer, computed by linear scan instead of
/// a tree.
inline std::vector<std::size_t> brute_force_contains(const std::vector<Box>& boxes,
                                                      const Point& p) {
  std::vector<std::size_t> hits;
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (contains(boxes[i], p)) hits.push_back(i);
  }
  return hits;
}

/// Squared distance from `p` to the closest point of `b` (0 if `p` is
/// inside): the standard "distance to an AABB" formula, clamping each axis
/// to the box before taking the norm. Used ONLY by the test's own brute-force
/// k-nearest reference -- `fuse::search` deliberately exposes no distance
/// metric of its own (box_index.hpp's `query_nearest` doc comment).
inline double squared_distance_to_box(const Box& b, const Point& p) {
  auto axis = [](double v, double lo, double hi) {
    const double c = std::clamp(v, lo, hi);
    return (v - c) * (v - c);
  };
  return axis(p.x, b.lo.x, b.hi.x) + axis(p.y, b.lo.y, b.hi.y) + axis(p.z, b.lo.z, b.hi.z);
}

/// The `k` smallest-distance box indices to `p`, ties broken by index
/// (matching `QueryResult`'s own ascending-by-index sort). Do NOT compare a
/// `BoxIndex::query_nearest` answer against this by exact set equality when
/// boxes can overlap `p` (several boxes containing the same point all sit
/// at distance exactly zero) -- use `is_valid_knn` below instead, which is
/// the property a k-nearest answer actually owes when there is a genuine
/// tie at the k-th boundary. Kept for the (rarer, well-separated) fixture
/// where ties do not arise.
inline std::vector<std::size_t> brute_force_nearest(const std::vector<Box>& boxes, const Point& p,
                                                    int k) {
  std::vector<std::pair<double, std::size_t>> ranked;
  ranked.reserve(boxes.size());
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    ranked.push_back({squared_distance_to_box(boxes[i], p), i});
  }
  std::sort(ranked.begin(), ranked.end());
  const std::size_t take = std::min(static_cast<std::size_t>(k), ranked.size());
  std::vector<std::size_t> hits;
  hits.reserve(take);
  for (std::size_t i = 0; i < take; ++i) hits.push_back(ranked[i].second);
  std::sort(hits.begin(), hits.end());
  return hits;
}

/// True if `got` (a k-nearest answer's hits for one query point, as reported
/// by `BoxIndex`) is a VALID k-nearest set for `p` against `boxes`: every
/// box strictly closer than the k-th smallest distance is present, no box
/// farther than it is, and the count matches `min(k, boxes.size())`.
/// DELIBERATELY tolerant of which box wins a tie exactly AT the k-th
/// boundary distance -- with axis-aligned boxes that can overlap `p`
/// (several elements sharing a face or all containing `p` at once), several
/// boxes legitimately sit at the identical smallest distance (0, if `p` is
/// inside more than one), and no k-nearest algorithm -- ArborX's or a
/// from-scratch one -- owes a SPECIFIC one of them a slot over another; only
/// this structural property is owed.
inline bool is_valid_knn(const std::vector<Box>& boxes, const Point& p, int k,
                        const std::vector<std::size_t>& got) {
  std::vector<double> dist(boxes.size());
  for (std::size_t i = 0; i < boxes.size(); ++i) dist[i] = squared_distance_to_box(boxes[i], p);
  std::vector<double> sorted_dist = dist;
  std::sort(sorted_dist.begin(), sorted_dist.end());

  const std::size_t want = std::min(static_cast<std::size_t>(k), boxes.size());
  if (got.size() != want) return false;
  if (want == 0) return true;

  const double boundary = sorted_dist[want - 1];
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const bool included = std::find(got.begin(), got.end(), i) != got.end();
    if (dist[i] < boundary && !included) return false;  // a strictly closer box was left out
  }
  for (std::size_t idx : got) {
    if (dist[idx] > boundary) return false;  // a strictly farther box was included
  }
  return true;
}

}  // namespace fuse::search::testing
