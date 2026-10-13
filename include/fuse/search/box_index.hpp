#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/base/result.hpp"
#include "fuse/search/types.hpp"

/// A spatial index over axis-aligned boxes, backed by
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

/// `query_points`/`query_nearest`'s device-resident counterpart: the same
/// CSR shape as `QueryResult`, but both arrays live in
/// `Kokkos::DefaultExecutionSpace::memory_space` so a caller that already
/// holds its query points on the device reads the hits back without a host
/// round trip. Populated only by the `Kokkos::View` overloads below, which
/// always build and query on `Kokkos::DefaultExecutionSpace` -- on a build
/// with no CUDA/HIP/SYCL backend that space IS host memory, so this is then
/// the same data `QueryResult` would hold, just in a Kokkos container.
struct QueryResultView {
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> offsets;
  Kokkos::View<std::size_t*, Kokkos::DefaultExecutionSpace::memory_space> indices;
};

/// `BoxIndex` builds and queries on `Kokkos::DefaultHostExecutionSpace` /
/// `Kokkos::HostSpace` by default -- byte-identical to this facade's
/// behaviour before `fuse::search::ExecutionSpace` existed -- or on
/// `Kokkos::DefaultExecutionSpace` and its `memory_space` when
/// `resolve_execution_space()` (types.hpp) resolves to `Device`, which
/// RunnerConfig's own `FUSE_EXECUTION_SPACE=device` convention selects, or a
/// caller names explicitly. `executes_on_device()`/`space_name()` report
/// which one a given index actually built on (not merely which one was
/// requested: `Device` degrades to `Host` on a build with no device
/// backend).
class BoxIndex {
 public:
  BoxIndex();
  ~BoxIndex();
  BoxIndex(BoxIndex&&) noexcept;
  BoxIndex& operator=(BoxIndex&&) noexcept;
  BoxIndex(const BoxIndex&) = delete;
  BoxIndex& operator=(const BoxIndex&) = delete;

  /// Builds the index over `boxes`, taken by value (moved into internal
  /// Kokkos storage, copied to the device additionally when `space` resolves
  /// to `Device` -- see the class comment). `boxes` are indexed exactly as
  /// given: `boxes[k]`'s hits report position `k`. This function applies NO
  /// padding/tolerance of its own (geometry.hpp's `padded()` is a separate,
  /// explicit step a caller takes first if it wants the "point on a shared
  /// face finds both cells" property -- `element_boxes`, mesh_adapter.hpp,
  /// does this).
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
  ///   * `E-SEARCH-SPACE` -- `space == ExecutionSpace::Default` and
  ///     `FUSE_EXECUTION_SPACE` names neither `host` nor `device`
  ///     (`resolve_execution_space()`, types.hpp).
  [[nodiscard]] static Result<BoxIndex> build(std::vector<Box> boxes,
                                              ExecutionSpace space = ExecutionSpace::Default);

  /// Device-resident entry point: `boxes` already lives in
  /// `Kokkos::DefaultExecutionSpace::memory_space` (a device-resident caller
  /// never copies through the host to reach this). Always builds on
  /// `Kokkos::DefaultExecutionSpace`; `executes_on_device()` reports whether
  /// that space is actually a device on this build. Same refusal as the
  /// `std::vector` overload for a mixed-dimension `boxes`.
  ///
  /// A DISTINCT NAME, not a `build` overload: `BoxIndex::build({})` (an
  /// empty braced-init-list, with no other argument to break the tie) is
  /// equally valid as an empty `std::vector<Box>` or a default-constructed
  /// `Kokkos::View`, so overloading on parameter type alone would make that
  /// call -- which this file's own `EmptyIndexIsValidNotARefusal` test
  /// already made before this entry point existed -- ambiguous and a
  /// compile error. A `std::vector<Box>` argument already in hand is never
  /// itself ambiguous against a `Kokkos::View` parameter (no converting
  /// constructor exists either way); only the bare braced-list call is.
  [[nodiscard]] static Result<BoxIndex> build_device_resident(
      Kokkos::View<const Box*, Kokkos::DefaultExecutionSpace::memory_space> boxes);

  /// 0 (nothing was built), 1, 2 or 3, per `build()`'s classification.
  [[nodiscard]] int dimension() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  /// Whether this index's tree lives on a device execution space (`true`
  /// only when `build()` resolved to `Device` AND this build has a real
  /// CUDA/HIP/SYCL backend -- `Device` requested on a host-only build
  /// degrades to `Host`, so this reports what actually happened, not what
  /// was asked for).
  [[nodiscard]] bool executes_on_device() const noexcept;

  /// `Kokkos::DefaultExecutionSpace::name()` if `executes_on_device()`,
  /// else `Kokkos::DefaultHostExecutionSpace::name()` -- e.g. `"Cuda"` or
  /// `"Serial"`/`"OpenMP"`.
  [[nodiscard]] const char* space_name() const noexcept;

  /// For every point in `points`, every box index `k` with
  /// `contains(boxes[k], points[q])` (geometry.hpp) -- so a caller that
  /// wants the shared-face tolerance must have padded `boxes` before
  /// `build()`; this function does not add any tolerance of its own. Never
  /// fails: an empty index or an empty `points` both return a valid,
  /// all-empty `QueryResult`.
  [[nodiscard]] Result<QueryResult> query_points(const std::vector<Point>& points) const;

  /// Device-resident counterpart of `query_points`: `points` and the
  /// returned hits never leave `Kokkos::DefaultExecutionSpace::memory_space`.
  [[nodiscard]] Result<QueryResultView> query_points_device_resident(
      Kokkos::View<const Point*, Kokkos::DefaultExecutionSpace::memory_space> points) const;

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

  /// Device-resident counterpart of `query_nearest`.
  [[nodiscard]] Result<QueryResultView> query_nearest_device_resident(
      Kokkos::View<const Point*, Kokkos::DefaultExecutionSpace::memory_space> points, int k) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int dimension_ = 0;
  bool device_ = false;
};

}  // namespace fuse::search
