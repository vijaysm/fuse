#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/search/types.hpp"

/// A host-space spatial index over axis-aligned boxes, backed by
/// `ArborX::BoundingVolumeHierarchy` (src/search/box_index.cpp is the only
/// translation unit that names an ArborX type -- this header, like
/// `fuse::linalg`'s facade, never does).
namespace fuse::search {

/// CSR-style batched query result: query `q`'s hits are
/// `indices[offsets[q] .. offsets[q + 1])`, always ASCENDING within a query
/// (BoxIndex's own numbering: position in the `boxes` vector `build()` was
/// given). Sorted for determinism -- ArborX's own traversal order is an
/// implementation detail nothing downstream should depend on -- not because
/// a hit's position in the tree carries any relevance ranking; there is
/// none, and `query_nearest`'s hits are sorted the same way, not by
/// distance, for exactly that reason.
struct QueryResult {
  std::vector<std::size_t> offsets;  ///< size == number of queries + 1
  std::vector<std::size_t> indices;  ///< size == offsets.back()
};

/// Host-only (Kokkos::DefaultHostExecutionSpace / Kokkos::HostSpace), the
/// same scoping `fuse::partition::centroids_of` uses for its own Kokkos
/// views: nothing here is templated on an execution space because nothing
/// in FUSE yet needs this index built or queried on a device.
class BoxIndex {
 public:
  BoxIndex();
  ~BoxIndex();
  BoxIndex(BoxIndex&&) noexcept;
  BoxIndex& operator=(BoxIndex&&) noexcept;
  BoxIndex(const BoxIndex&) = delete;
  BoxIndex& operator=(const BoxIndex&) = delete;

  /// Builds the index over `boxes`, taken by value (moved into internal
  /// Kokkos storage). `boxes` are indexed exactly as given: `boxes[k]`'s hits
  /// report position `k`. This function applies NO padding/tolerance of its
  /// own (geometry.hpp's `padded()` is a separate, explicit step a caller
  /// takes first if it wants the "point on a shared face finds both cells"
  /// property -- `element_boxes`, mesh_adapter.hpp, does this).
  ///
  /// DIMENSION IS READ FROM THE DATA, NOT A PARAMETER: every box is
  /// classified independently (`geometry.hpp::classify_dim`) from its own
  /// extents, and `dimension()` reports that dimension once every box
  /// agrees on it.
  ///
  /// Errors, never throws:
  ///   * `E-SEARCH-MIXED-DIM` -- `boxes` are not all the same dimension
  ///     (e.g. some carry a nonzero z-extent and others do not). An empty
  ///     `boxes` has nothing to disagree about and always succeeds, with
  ///     `dimension() == 0` and `empty() == true`.
  [[nodiscard]] static Result<BoxIndex> build(std::vector<Box> boxes);

  /// 0 (nothing was built), 1, 2 or 3, per `build()`'s classification.
  [[nodiscard]] int dimension() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  /// For every point in `points`, every box index `k` with
  /// `contains(boxes[k], points[q])` (geometry.hpp) -- so a caller that
  /// wants the shared-face tolerance must have padded `boxes` before
  /// `build()`; this function does not add any tolerance of its own. Never
  /// fails: an empty index or an empty `points` both return a valid,
  /// all-empty `QueryResult`.
  [[nodiscard]] Result<QueryResult> query_points(const std::vector<Point>& points) const;

  /// For every point in `points`, the `k` nearest boxes by distance to the
  /// box (0 if the point is inside it), `k` capped to `size()` when the
  /// index holds fewer boxes than that. Hits are NOT sorted by distance (see
  /// `QueryResult`'s comment) -- a caller that needs the nearest-first order
  /// must compute distances itself (`geometry.hpp` has none, deliberately:
  /// nothing here commits to a specific box-to-point distance metric beyond
  /// ArborX's own, which this facade does not expose).
  ///
  /// Errors, never throws: `E-SEARCH-K` if `k <= 0`.
  [[nodiscard]] Result<QueryResult> query_nearest(const std::vector<Point>& points, int k) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int dimension_ = 0;
};

}  // namespace fuse::search
