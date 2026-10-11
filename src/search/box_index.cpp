// src/search/box_index.cpp -- the only translation unit in fuse::search that
// names an ArborX type (types.hpp's header comment: this layer is a facade
// over it, the same posture fuse::linalg takes over its dense solver).

#include "fuse/search/box_index.hpp"

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <utility>

#include <fmt/format.h>

#include "fuse/search/geometry.hpp"

namespace fuse::search {

namespace {

using Coordinate = double;
using ArborBox = ArborX::Box<3, Coordinate>;
using ArborPoint = ArborX::Point<3, Coordinate>;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
// PairValueIndex, not a bare Box: ArborX's own convenience (indices, offsets)
// overload only extracts an index automatically when the tree's Value type
// IS the index (a plain box/point tree deduces Value = the geometry itself,
// and DefaultCallback::operator() then tries to write that geometry into an
// integer output view, which does not compile -- found empirically against
// this exact ArborX 2.1 / Kokkos 5.1.0 pair while proving the dependency; see
// the ArborX track's dependency-resolution report). Attaching the index
// explicitly and extracting it with our own callback (ExtractIndex below)
// sidesteps that entirely, and works identically for spatial and nearest
// predicates.
using Value = ArborX::PairValueIndex<ArborBox, int>;
using Tree = ArborX::BoundingVolumeHierarchy<MemorySpace, Value>;

ArborBox to_arbor(const Box& b) {
  return ArborBox{ArborPoint{b.lo.x, b.lo.y, b.lo.z}, ArborPoint{b.hi.x, b.hi.y, b.hi.z}};
}

ArborPoint to_arbor(const Point& p) { return ArborPoint{p.x, p.y, p.z}; }

/// Writes `value.index` (not `value` itself -- see the `Value` alias comment
/// above) into the CRS output view. Identical for every predicate tag ArborX
/// dispatches on (spatial intersects, nearest): the tag only changes how
/// `_out` is invoked around this, never what this callback does.
struct ExtractIndex {
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const&, Value const& value,
                                  OutputFunctor const& out) const {
    out(static_cast<std::size_t>(value.index));
  }
};

/// Sorts each query's hit group ascending by index (QueryResult's doc:
/// determinism, not relevance) and copies out of Kokkos views into the
/// plain-vector CRS the public header promises.
QueryResult to_result(const Kokkos::View<std::size_t*, MemorySpace>& indices,
                      const Kokkos::View<int*, MemorySpace>& offsets) {
  QueryResult r;
  r.offsets.assign(offsets.data(), offsets.data() + offsets.extent(0));
  r.indices.assign(indices.data(), indices.data() + indices.extent(0));
  for (std::size_t q = 0; q + 1 < r.offsets.size(); ++q) {
    std::sort(r.indices.begin() + static_cast<std::ptrdiff_t>(r.offsets[q]),
              r.indices.begin() + static_cast<std::ptrdiff_t>(r.offsets[q + 1]));
  }
  return r;
}

}  // namespace

struct BoxIndex::Impl {
  Tree tree;
  ExecutionSpace space{};
};

BoxIndex::BoxIndex() = default;
BoxIndex::~BoxIndex() = default;
BoxIndex::BoxIndex(BoxIndex&&) noexcept = default;
BoxIndex& BoxIndex::operator=(BoxIndex&&) noexcept = default;

Result<BoxIndex> BoxIndex::build(std::vector<Box> boxes) {
  BoxIndex out;
  const std::size_t n = boxes.size();
  out.impl_ = std::make_unique<Impl>();
  if (n == 0) {
    out.dimension_ = 0;
    out.impl_->tree = Tree();  // BoundingVolumeHierarchy's own "build an empty tree" default
    return std::move(out);
  }

  const int dim0 = classify_dim(boxes[0]);
  for (std::size_t i = 1; i < n; ++i) {
    const int di = classify_dim(boxes[i]);
    if (di != dim0) {
      return Failure{"E-SEARCH-MIXED-DIM",
                     fmt::format("box 0 classifies as {}D but box {} classifies as {}D; "
                                 "a spatial index cannot mix dimensionalities in one build "
                                 "(fuse::search::Point's convention: unused coordinates must "
                                 "be exactly zero)",
                                 dim0, i, di)};
    }
  }

  Kokkos::View<ArborBox*, MemorySpace> arbor_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::boxes"), n);
  for (std::size_t i = 0; i < n; ++i) arbor_boxes(i) = to_arbor(boxes[i]);

  out.impl_->tree =
      Tree(out.impl_->space, ArborX::Experimental::attach_indices<int>(arbor_boxes));
  out.dimension_ = dim0;
  return std::move(out);
}

int BoxIndex::dimension() const noexcept { return dimension_; }
std::size_t BoxIndex::size() const noexcept { return impl_ ? impl_->tree.size() : 0; }
bool BoxIndex::empty() const noexcept { return size() == 0; }

Result<QueryResult> BoxIndex::query_points(const std::vector<Point>& points) const {
  QueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);
  if (empty() || points.empty()) return empty_result;

  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, MemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds"), points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    preds(i) = ArborX::intersects(to_arbor(points[i]));
  }

  Kokkos::View<std::size_t*, MemorySpace> indices("fuse::search::indices", 0);
  Kokkos::View<int*, MemorySpace> offsets("fuse::search::offsets", 0);
  impl_->tree.query(impl_->space, preds, ExtractIndex{}, indices, offsets);
  return to_result(indices, offsets);
}

Result<QueryResult> BoxIndex::query_nearest(const std::vector<Point>& points, int k) const {
  if (k <= 0) {
    return Failure{"E-SEARCH-K", fmt::format("query_nearest: k == {} must be positive", k)};
  }
  QueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);
  if (empty() || points.empty()) return empty_result;

  Kokkos::View<decltype(ArborX::nearest(ArborPoint{}, 1))*, MemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_preds"),
      points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    preds(i) = ArborX::nearest(to_arbor(points[i]), k);
  }

  Kokkos::View<std::size_t*, MemorySpace> indices("fuse::search::nearest_indices", 0);
  Kokkos::View<int*, MemorySpace> offsets("fuse::search::nearest_offsets", 0);
  impl_->tree.query(impl_->space, preds, ExtractIndex{}, indices, offsets);
  return to_result(indices, offsets);
}

}  // namespace fuse::search
