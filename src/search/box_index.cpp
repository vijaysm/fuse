// src/search/box_index.cpp -- one of the two translation units in
// fuse::search that name an ArborX type (the other is distributed.cpp,
// added after this comment was first written); search.md:5's "nothing
// outside src/search/*.cpp names an ArborX type" is the accurate claim
// (types.hpp's header comment: this layer is a facade over it, the same
// posture fuse::linalg takes over its dense solver).
//
// EXECUTION SPACE. BoxIndex builds and queries on either
// Kokkos::DefaultHostExecutionSpace/Kokkos::HostSpace (the facade's
// behaviour before fuse::search::ExecutionSpace existed, still the default)
// or Kokkos::DefaultExecutionSpace and its memory_space, selected at
// build() time (docs/dev/search.md's "Execution space" section). `Value`
// (ArborX::PairValueIndex<ArborBox, int>) names no space, so the SAME type
// indexes a host or a device tree; only the Tree's MemorySpace differs, so
// `Impl` carries one tree of each kind and `BoxIndex::device_` says which is
// live. On a build with no CUDA/HIP/SYCL backend,
// Kokkos::DefaultExecutionSpace IS Kokkos::DefaultHostExecutionSpace, so the
// "device" tree degenerates to the exact same type (and behaviour) as the
// host one -- `kHasDeviceBackend` below is what keeps `executes_on_device()`
// truthful in that case (`Device` was requested, but nothing resembling a
// device actually ran it).

#include "fuse/search/box_index.hpp"

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <fmt/format.h>

#include "fuse/search/geometry.hpp"

namespace fuse::search {

namespace {

using Coordinate = double;
using ArborBox = ArborX::Box<3, Coordinate>;
using ArborPoint = ArborX::Point<3, Coordinate>;
// PairValueIndex, not a bare Box: ArborX's own convenience (indices, offsets)
// overload only extracts an index automatically when the tree's Value type
// IS the index (a plain box/point tree deduces Value = the geometry itself,
// and DefaultCallback::operator() then tries to write that geometry into an
// integer output view, which does not compile -- found empirically against
// this exact ArborX 2.1 / Kokkos 5.1.0 pair while proving the dependency; see
// the ArborX track's dependency-resolution report). Attaching the index
// explicitly and extracting it with our own callback (ExtractIndex below)
// sidesteps that entirely, and works identically for spatial and nearest
// predicates, and for a host or a device tree -- PairValueIndex names no
// space, so one `Value` type serves both.
using Value = ArborX::PairValueIndex<ArborBox, int>;

using HostMemorySpace = Kokkos::HostSpace;
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostTree = ArborX::BoundingVolumeHierarchy<HostMemorySpace, Value>;

using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceTree = ArborX::BoundingVolumeHierarchy<DeviceMemorySpace, Value>;

/// True only when this build's default execution space is a real
/// CUDA/HIP/SYCL device -- i.e. `DeviceExecutionSpace` did not collapse to
/// `HostExecutionSpace`. `BoxIndex::executes_on_device()` reports
/// `device_ && kHasDeviceBackend`, so requesting `Device` on a host-only
/// build is truthfully reported as having run on the host.
inline constexpr bool kHasDeviceBackend =
    !std::is_same_v<DeviceExecutionSpace, HostExecutionSpace>;

KOKKOS_FUNCTION ArborBox to_arbor(const Box& b) {
  return ArborBox{ArborPoint{b.lo.x, b.lo.y, b.lo.z}, ArborPoint{b.hi.x, b.hi.y, b.hi.z}};
}

KOKKOS_FUNCTION ArborPoint to_arbor(const Point& p) { return ArborPoint{p.x, p.y, p.z}; }

/// Writes `value.index` (not `value` itself -- see the `Value` alias comment
/// above) into the CRS output view. Identical for every predicate tag ArborX
/// dispatches on (spatial intersects, nearest), and for a host or a device
/// tree: `Value` names no space.
struct ExtractIndex {
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const&, Value const& value,
                                  OutputFunctor const& out) const {
    out(static_cast<std::size_t>(value.index));
  }
};

/// Sorts each query's hit group ascending by index (QueryResult's doc:
/// determinism, not relevance) and copies out of HOST Kokkos views into the
/// plain-vector CRS the public header promises. Always given host-accessible
/// views -- a device build's result is deep_copy'd to a host mirror before
/// this runs (see `BoxIndex::query_points`/`query_nearest` below), so this
/// function itself never touches device memory.
QueryResult to_result(const Kokkos::View<std::size_t*, HostMemorySpace>& indices,
                      const Kokkos::View<int*, HostMemorySpace>& offsets) {
  QueryResult r;
  r.offsets.assign(offsets.data(), offsets.data() + offsets.extent(0));
  r.indices.assign(indices.data(), indices.data() + indices.extent(0));
  for (std::size_t q = 0; q + 1 < r.offsets.size(); ++q) {
    std::sort(r.indices.begin() + static_cast<std::ptrdiff_t>(r.offsets[q]),
              r.indices.begin() + static_cast<std::ptrdiff_t>(r.offsets[q + 1]));
  }
  return r;
}

/// Device-resident counterpart of `to_result`'s sort step: an in-place
/// insertion sort of each query's (typically short) hit segment, run as one
/// Kokkos thread per query directly on `indices` wherever it lives --
/// `Kokkos::DefaultExecutionSpace::memory_space` for the view-based API this
/// serves, so a device-resident caller's hits never round-trip through the
/// host to be sorted. O(k^2) per segment is the accepted cost of staying
/// fully device-resident for the typically-small hit counts this facade
/// returns; the std::vector-returning API above sorts on the host instead,
/// where std::sort's O(k log k) is available.
template <class Space>
void sort_segments_on_device(Kokkos::View<std::size_t*, typename Space::memory_space> indices,
                             Kokkos::View<int*, typename Space::memory_space> offsets,
                             std::size_t nqueries) {
  Kokkos::parallel_for(
      "fuse::search::sort_segments",
      Kokkos::RangePolicy<Space>(0, static_cast<std::int64_t>(nqueries)),
      KOKKOS_LAMBDA(std::int64_t q) {
        const auto begin = static_cast<std::size_t>(offsets(q));
        const auto end = static_cast<std::size_t>(offsets(q + 1));
        for (std::size_t i = begin + 1; i < end; ++i) {
          const std::size_t key = indices(i);
          std::size_t j = i;
          while (j > begin && indices(j - 1) > key) {
            indices(j) = indices(j - 1);
            --j;
          }
          indices(j) = key;
        }
      });
}

}  // namespace

struct BoxIndex::Impl {
  HostTree host_tree;
  HostExecutionSpace host_space{};
  DeviceTree device_tree;
  DeviceExecutionSpace device_space{};
};

BoxIndex::BoxIndex() = default;
BoxIndex::~BoxIndex() = default;
BoxIndex::BoxIndex(BoxIndex&&) noexcept = default;
BoxIndex& BoxIndex::operator=(BoxIndex&&) noexcept = default;

namespace {
/// Shared by both `build()` overloads: every box classified, all agreeing on
/// one dimension. `boxes[i]` is read through `at(i)`, which is either a
/// `std::vector<Box>` subscript or a host-mirror view lookup, so the SAME
/// check runs whether the caller handed over a `std::vector` or a device
/// view (the device view is mirrored to host first -- see the `Kokkos::View`
/// `build()` overload -- purely to run this validation; the tree itself is
/// still built from the device-resident data).
template <class At>
Result<int> classify_all(std::size_t n, At&& at) {
  if (n == 0) return 0;
  const int dim0 = classify_dim(at(0));
  for (std::size_t i = 1; i < n; ++i) {
    const int di = classify_dim(at(i));
    if (di != dim0) {
      return Failure{"E-SEARCH-MIXED-DIM",
                     fmt::format("box 0 classifies as {}D but box {} classifies as {}D; "
                                 "a spatial index cannot mix dimensionalities in one build "
                                 "(fuse::search::Point's convention: unused coordinates must "
                                 "be exactly zero)",
                                 dim0, i, di)};
    }
  }
  return dim0;
}

/// Shared by both `query_nearest` overloads, so the refusal lives exactly
/// once in this file (build-evidence/mutations/search.toml's
/// "search-k-refusal-disabled" anchors this single copy).
Result<void> check_k(int k) {
  if (k <= 0) {
    return Failure{"E-SEARCH-K", fmt::format("query_nearest: k == {} must be positive", k)};
  }
  return Result<void>::success();
}
}  // namespace

Result<BoxIndex> BoxIndex::build(std::vector<Box> boxes, ExecutionSpace space) {
  const Result<ExecutionSpace> resolved = resolve_execution_space(space);
  if (!resolved.ok()) return resolved.error();
  const bool use_device = resolved.value() == ExecutionSpace::Device && kHasDeviceBackend;

  BoxIndex out;
  const std::size_t n = boxes.size();
  out.impl_ = std::make_unique<Impl>();
  out.device_ = use_device;
  if (n == 0) {
    out.dimension_ = 0;
    return std::move(out);  // default-constructed trees are ArborX's own empty tree
  }

  const Result<int> dim = classify_all(n, [&](std::size_t i) -> const Box& { return boxes[i]; });
  if (!dim.ok()) return dim.error();

  Kokkos::View<ArborBox*, HostMemorySpace> host_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::boxes"), n);
  for (std::size_t i = 0; i < n; ++i) host_boxes(i) = to_arbor(boxes[i]);

  if (use_device) {
    Kokkos::View<ArborBox*, DeviceMemorySpace> device_boxes(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::boxes_device"), n);
    Kokkos::deep_copy(device_boxes, host_boxes);
    out.impl_->device_tree =
        DeviceTree(out.impl_->device_space, ArborX::Experimental::attach_indices<int>(device_boxes));
  } else {
    out.impl_->host_tree =
        HostTree(out.impl_->host_space, ArborX::Experimental::attach_indices<int>(host_boxes));
  }
  out.dimension_ = dim.value();
  return std::move(out);
}

Result<BoxIndex> BoxIndex::build_device_resident(
    Kokkos::View<const Box*, Kokkos::DefaultExecutionSpace::memory_space> boxes) {
  BoxIndex out;
  const std::size_t n = boxes.extent(0);
  out.impl_ = std::make_unique<Impl>();
  out.device_ = kHasDeviceBackend;
  if (n == 0) {
    out.dimension_ = 0;
    return std::move(out);
  }

  // Validation (classify_all) needs host-readable boxes; mirror once. The
  // TREE itself is still built directly from `boxes`, wherever it lives --
  // this mirror exists only so classify_dim's per-box check (host-only: it
  // returns a Result, which is not device-callable) has something to read.
  auto host_mirror = Kokkos::create_mirror_view(boxes);
  Kokkos::deep_copy(host_mirror, boxes);
  const Result<int> dim =
      classify_all(n, [&](std::size_t i) -> const Box& { return host_mirror(i); });
  if (!dim.ok()) return dim.error();

  Kokkos::View<ArborBox*, DeviceMemorySpace> arbor_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::boxes_device_resident"), n);
  Kokkos::parallel_for(
      "fuse::search::convert_boxes",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n)),
      KOKKOS_LAMBDA(std::int64_t i) { arbor_boxes(i) = to_arbor(boxes(i)); });
  out.impl_->device_tree =
      DeviceTree(out.impl_->device_space, ArborX::Experimental::attach_indices<int>(arbor_boxes));
  out.dimension_ = dim.value();
  return std::move(out);
}

int BoxIndex::dimension() const noexcept { return dimension_; }
std::size_t BoxIndex::size() const noexcept {
  if (!impl_) return 0;
  return device_ ? impl_->device_tree.size() : impl_->host_tree.size();
}
bool BoxIndex::empty() const noexcept { return size() == 0; }
bool BoxIndex::executes_on_device() const noexcept { return device_ && kHasDeviceBackend; }
const char* BoxIndex::space_name() const noexcept {
  return executes_on_device() ? DeviceExecutionSpace::name() : HostExecutionSpace::name();
}

Result<QueryResult> BoxIndex::query_points(const std::vector<Point>& points) const {
  QueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);
  if (empty() || points.empty()) return empty_result;

  if (device_) {
    Kokkos::View<ArborPoint*, HostMemorySpace> host_pts(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds_host"),
        points.size());
    for (std::size_t i = 0; i < points.size(); ++i) host_pts(i) = to_arbor(points[i]);
    Kokkos::View<ArborPoint*, DeviceMemorySpace> device_pts(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds_device"),
        points.size());
    Kokkos::deep_copy(device_pts, host_pts);

    Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, DeviceMemorySpace> preds(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds"),
        points.size());
    Kokkos::parallel_for(
        "fuse::search::build_point_preds",
        Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(points.size())),
        KOKKOS_LAMBDA(std::int64_t i) { preds(i) = ArborX::intersects(device_pts(i)); });

    Kokkos::View<std::size_t*, DeviceMemorySpace> indices("fuse::search::indices_device", 0);
    Kokkos::View<int*, DeviceMemorySpace> offsets("fuse::search::offsets_device", 0);
    impl_->device_tree.query(impl_->device_space, preds, ExtractIndex{}, indices, offsets);

    Kokkos::View<std::size_t*, HostMemorySpace> host_indices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::indices"),
        indices.extent(0));
    Kokkos::View<int*, HostMemorySpace> host_offsets(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::offsets"),
        offsets.extent(0));
    Kokkos::deep_copy(host_indices, indices);
    Kokkos::deep_copy(host_offsets, offsets);
    return to_result(host_indices, host_offsets);
  }

  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, HostMemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds"), points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    preds(i) = ArborX::intersects(to_arbor(points[i]));
  }

  Kokkos::View<std::size_t*, HostMemorySpace> indices("fuse::search::indices", 0);
  Kokkos::View<int*, HostMemorySpace> offsets("fuse::search::offsets", 0);
  impl_->host_tree.query(impl_->host_space, preds, ExtractIndex{}, indices, offsets);
  return to_result(indices, offsets);
}

Result<QueryResultView> BoxIndex::query_points_device_resident(
    Kokkos::View<const Point*, Kokkos::DefaultExecutionSpace::memory_space> points) const {
  const std::size_t n = points.extent(0);
  QueryResultView empty_result;
  empty_result.offsets =
      Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::offsets_empty", n + 1);
  empty_result.indices =
      Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::indices_empty", 0);
  if (empty() || n == 0) return empty_result;

  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, DeviceMemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::point_preds_device_resident"),
      n);
  Kokkos::parallel_for(
      "fuse::search::build_point_preds_device_resident",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n)),
      KOKKOS_LAMBDA(std::int64_t i) { preds(i) = ArborX::intersects(to_arbor(points(i))); });

  QueryResultView out;
  out.indices = Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::indices_device_resident", 0);
  Kokkos::View<int*, DeviceMemorySpace> offsets("fuse::search::offsets_device_resident", 0);
  impl_->device_tree.query(impl_->device_space, preds, ExtractIndex{}, out.indices, offsets);
  // ArborX's own query signature wants an `int` offsets view; the public
  // result type promises `std::size_t` throughout (QueryResultView's own
  // doc), so one more device-resident copy-and-widen before returning.
  Kokkos::View<std::size_t*, DeviceMemorySpace> offsets_sz(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::offsets_sz"),
      offsets.extent(0));
  Kokkos::parallel_for(
      "fuse::search::widen_offsets",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(offsets.extent(0))),
      KOKKOS_LAMBDA(std::int64_t i) { offsets_sz(i) = static_cast<std::size_t>(offsets(i)); });
  out.offsets = offsets_sz;
  sort_segments_on_device<DeviceExecutionSpace>(out.indices, offsets, n);
  return out;
}

Result<QueryResult> BoxIndex::query_nearest(const std::vector<Point>& points, int k) const {
  const Result<void> k_ok = check_k(k);
  if (!k_ok.ok()) return k_ok.error();
  QueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);
  if (empty() || points.empty()) return empty_result;

  // PS-7: cap explicitly rather than relying on ArborX::nearest's own
  // internal clamp -- box_index.hpp's doc promises "k capped to size()" as
  // THIS facade's contract, not a property borrowed from whatever ArborX
  // version happens to be linked. Applies to both the device and host
  // query_nearest paths below.
  const int k_capped = std::min(k, static_cast<int>(size()));

  if (device_) {
    Kokkos::View<ArborPoint*, HostMemorySpace> host_pts(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_pts_host"),
        points.size());
    for (std::size_t i = 0; i < points.size(); ++i) host_pts(i) = to_arbor(points[i]);
    Kokkos::View<ArborPoint*, DeviceMemorySpace> device_pts(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_pts_device"),
        points.size());
    Kokkos::deep_copy(device_pts, host_pts);

    Kokkos::View<decltype(ArborX::nearest(ArborPoint{}, 1))*, DeviceMemorySpace> preds(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_preds"),
        points.size());
    Kokkos::parallel_for(
        "fuse::search::build_nearest_preds",
        Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(points.size())),
        KOKKOS_LAMBDA(std::int64_t i) { preds(i) = ArborX::nearest(device_pts(i), k_capped); });

    Kokkos::View<std::size_t*, DeviceMemorySpace> indices("fuse::search::nearest_indices_device",
                                                          0);
    Kokkos::View<int*, DeviceMemorySpace> offsets("fuse::search::nearest_offsets_device", 0);
    impl_->device_tree.query(impl_->device_space, preds, ExtractIndex{}, indices, offsets);

    Kokkos::View<std::size_t*, HostMemorySpace> host_indices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_indices"),
        indices.extent(0));
    Kokkos::View<int*, HostMemorySpace> host_offsets(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_offsets"),
        offsets.extent(0));
    Kokkos::deep_copy(host_indices, indices);
    Kokkos::deep_copy(host_offsets, offsets);
    return to_result(host_indices, host_offsets);
  }

  Kokkos::View<decltype(ArborX::nearest(ArborPoint{}, 1))*, HostMemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_preds"),
      points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    preds(i) = ArborX::nearest(to_arbor(points[i]), k_capped);
  }

  Kokkos::View<std::size_t*, HostMemorySpace> indices("fuse::search::nearest_indices", 0);
  Kokkos::View<int*, HostMemorySpace> offsets("fuse::search::nearest_offsets", 0);
  impl_->host_tree.query(impl_->host_space, preds, ExtractIndex{}, indices, offsets);
  return to_result(indices, offsets);
}

Result<QueryResultView> BoxIndex::query_nearest_device_resident(
    Kokkos::View<const Point*, Kokkos::DefaultExecutionSpace::memory_space> points, int k) const {
  const Result<void> k_ok = check_k(k);
  if (!k_ok.ok()) return k_ok.error();
  const std::size_t n = points.extent(0);
  QueryResultView empty_result;
  empty_result.offsets =
      Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::nearest_offsets_empty", n + 1);
  empty_result.indices =
      Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::nearest_indices_empty", 0);
  if (empty() || n == 0) return empty_result;

  Kokkos::View<decltype(ArborX::nearest(ArborPoint{}, 1))*, DeviceMemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                        "fuse::search::nearest_preds_device_resident"),
      n);
  Kokkos::parallel_for(
      "fuse::search::build_nearest_preds_device_resident",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n)),
      KOKKOS_LAMBDA(std::int64_t i) { preds(i) = ArborX::nearest(to_arbor(points(i)), k); });

  QueryResultView out;
  out.indices =
      Kokkos::View<std::size_t*, DeviceMemorySpace>("fuse::search::nearest_indices_device_resident", 0);
  Kokkos::View<int*, DeviceMemorySpace> offsets("fuse::search::nearest_offsets_device_resident",
                                                0);
  impl_->device_tree.query(impl_->device_space, preds, ExtractIndex{}, out.indices, offsets);
  Kokkos::View<std::size_t*, DeviceMemorySpace> offsets_sz(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::nearest_offsets_sz"),
      offsets.extent(0));
  Kokkos::parallel_for(
      "fuse::search::widen_nearest_offsets",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(offsets.extent(0))),
      KOKKOS_LAMBDA(std::int64_t i) { offsets_sz(i) = static_cast<std::size_t>(offsets(i)); });
  out.offsets = offsets_sz;
  sort_segments_on_device<DeviceExecutionSpace>(out.indices, offsets, n);
  return out;
}

}  // namespace fuse::search
