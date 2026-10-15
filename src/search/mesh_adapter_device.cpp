// src/search/mesh_adapter_device.cpp -- element_boxes_device_resident
// (include/fuse/search/mesh_adapter.hpp): the device-resident counterpart
// of element_boxes, over a simplex CSR already resident in
// Kokkos::DefaultExecutionSpace::memory_space. Kept in its own translation
// unit, not header-only like element_boxes itself, because it is not a
// template (no MeshLike duck-typed mesh to instantiate against -- this
// file's own header comment) and its body is Kokkos kernels, not something
// a caller needs inlined.

#include "fuse/search/mesh_adapter.hpp"

#include <cfloat>
#include <cstdint>

#include <fmt/format.h>

namespace fuse::search {

namespace {

using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

// Device-callable counterparts of geometry.hpp's diagonal()/padded() --
// that header's own functions are plain `inline` (host-only; this
// codebase's convention, native_mesh.cpp's device_segment_length and
// friends are the precedent), so the same formulas are restated here for a
// device execution space rather than annotating geometry.hpp itself.
KOKKOS_FUNCTION double device_diagonal(const Box& b) {
  const double dx = b.hi.x - b.lo.x;
  const double dy = b.hi.y - b.lo.y;
  const double dz = b.hi.z - b.lo.z;
  return Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
}

KOKKOS_FUNCTION Box device_padded(const Box& b, double relative, double absolute_floor) {
  const double p = Kokkos::max(relative * device_diagonal(b), absolute_floor);
  Box out = b;
  if (b.hi.x != b.lo.x) {
    out.lo.x -= p;
    out.hi.x += p;
  }
  if (b.hi.y != b.lo.y) {
    out.lo.y -= p;
    out.hi.y += p;
  }
  if (b.hi.z != b.lo.z) {
    out.lo.z -= p;
    out.hi.z += p;
  }
  return out;
}

}  // namespace

Result<Kokkos::View<Box*, DeviceMemorySpace>> element_boxes_device_resident(
    Kokkos::View<const int*, DeviceMemorySpace> simplex_offsets,
    Kokkos::View<const double* [4][3], Kokkos::LayoutRight, DeviceMemorySpace> simplex_corners,
    int corners_per_simplex, Kokkos::View<const bool*, DeviceMemorySpace> owned,
    const Tolerance& tol) {
  const std::size_t n_entities =
      simplex_offsets.extent(0) == 0 ? 0 : simplex_offsets.extent(0) - 1;
  if (n_entities == 0) {
    return Kokkos::View<Box*, DeviceMemorySpace>(
        "fuse::search::element_boxes_device_resident_empty", 0);
  }
  const bool has_owned_filter = owned.extent(0) != 0;

  // E-SEARCH-NO-GEOMETRY: the same refusal element_boxes makes for an
  // owned entity with no simplex decomposition -- found here as the
  // smallest such entity index, via a Min reduction (n_entities itself is
  // the "not found" sentinel, one past every valid index).
  int first_bad = static_cast<int>(n_entities);
  Kokkos::parallel_reduce(
      "fuse::search::element_boxes_device_resident_check",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n_entities)),
      KOKKOS_LAMBDA(std::int64_t e, int& local_min) {
        const bool is_owned_e = !has_owned_filter || owned(e);
        if (!is_owned_e) return;
        const int scount = simplex_offsets(e + 1) - simplex_offsets(e);
        if (scount == 0 && static_cast<int>(e) < local_min) local_min = static_cast<int>(e);
      },
      Kokkos::Min<int>(first_bad));
  if (first_bad < static_cast<int>(n_entities)) {
    return Failure{"E-SEARCH-NO-GEOMETRY",
                  fmt::format("entity {} has no simplex decomposition; this device-resident "
                              "mesh carries no geometry for it",
                              first_bad)};
  }

  // Compact the owned entities' output positions with an exclusive scan --
  // the device-resident analogue of element_boxes's own sequential
  // out_owned_ids.push_back.
  Kokkos::View<int*, DeviceMemorySpace> compacted_index(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                        "fuse::search::element_boxes_compacted_index"),
      n_entities);
  int n_boxes = 0;
  Kokkos::parallel_scan(
      "fuse::search::element_boxes_device_resident_scan",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n_entities)),
      KOKKOS_LAMBDA(std::int64_t e, int& update, bool final) {
        const bool is_owned_e = !has_owned_filter || owned(e);
        if (final) compacted_index(e) = update;
        if (is_owned_e) update += 1;
      },
      n_boxes);

  Kokkos::View<Box*, DeviceMemorySpace> boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::element_boxes_device_resident"),
      static_cast<std::size_t>(n_boxes));
  const double relative = tol.relative;
  const double absolute_floor = tol.absolute_floor;
  Kokkos::parallel_for(
      "fuse::search::element_boxes_device_resident_fill",
      Kokkos::RangePolicy<DeviceExecutionSpace>(0, static_cast<std::int64_t>(n_entities)),
      KOKKOS_LAMBDA(std::int64_t e) {
        const bool is_owned_e = !has_owned_filter || owned(e);
        if (!is_owned_e) return;
        const int soff = simplex_offsets(e);
        const int scount = simplex_offsets(e + 1) - soff;
        double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
        double hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        for (int s = 0; s < scount; ++s) {
          const int row = soff + s;
          for (int k = 0; k < corners_per_simplex; ++k) {
            lo[0] = Kokkos::min(lo[0], simplex_corners(row, k, 0));
            lo[1] = Kokkos::min(lo[1], simplex_corners(row, k, 1));
            lo[2] = Kokkos::min(lo[2], simplex_corners(row, k, 2));
            hi[0] = Kokkos::max(hi[0], simplex_corners(row, k, 0));
            hi[1] = Kokkos::max(hi[1], simplex_corners(row, k, 1));
            hi[2] = Kokkos::max(hi[2], simplex_corners(row, k, 2));
          }
        }
        Box b;
        b.lo = Point{lo[0], lo[1], lo[2]};
        b.hi = Point{hi[0], hi[1], hi[2]};
        boxes(compacted_index(e)) = device_padded(b, relative, absolute_floor);
      });

  return boxes;
}

}  // namespace fuse::search
