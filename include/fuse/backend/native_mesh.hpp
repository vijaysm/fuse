#pragma once

/// FUSE's native distributed unstructured mesh: general cells (points,
/// segments, triangles, quads, polygons, tets, hexes, prisms, pyramids and
/// general polyhedra, mixed within one topological dimension) on Kokkos
/// views, implementing `fuse::MeshView`/`fuse::MeshBackend`
/// (mesh_view.hpp). See docs/dev/native-mesh.md for the full design;
/// this header's comments cover only what a caller needs.
///
/// TWO LAYERS.
///
///   * `MeshArrays<MemorySpace>` is plain data: vertex coordinates, cell
///     connectivity (a CSR of vertex indices), cell type, global ids,
///     owner ranks, and (for `CellType::Polyhedron` cells only) a
///     cell-to-face and face-to-vertex CSR. Every array lives in
///     `MemorySpace`, so a caller that wants everything on a GPU gets
///     that; `to_memory_space<Target>(arrays)` deep-copies to another
///     space, both directions.
///   * `NativeMeshView` is the (non-templated) `fuse::MeshView`. It is
///     built FROM a `MeshArrays<AnySpace>` by mirroring to the host once,
///     validating connectivity and computing the whole geometry contract
///     (simplex decomposition, measures, centroids) there -- the same
///     shape as `fuse::native::Mesh::finalize()` (native/mesh.hpp) and for
///     the same reason: `Result`/`Failure`/`fmt` are not device-callable,
///     and this is a one-time, per-rank, host-side cost (robustness over
///     efficiency -- docs/dev/native-mesh.md).
///
/// BUILDING. `NativeMeshBuilder` is a plain host-side (`std::vector`)
/// builder used by every source (generators, the file reader, migration's
/// unpack): `add_vertex`, `add_cell`, `add_polyhedron_cell`, then
/// `build()` -> `Result<MeshArrays<Kokkos::HostSpace>>`. It deduplicates a
/// polyhedron's faces by their vertex GLOBAL id set, recording a per-cell
/// orientation flag (`true` = this cell's face-vertex order matches the
/// face's canonical stored order; `false` = reversed) so that a face
/// shared by two cells is stored ONCE.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native/cell_topology.hpp"
#include "fuse/backend/native/geometry.hpp"
#include "fuse/base/result.hpp"

namespace fuse::native {

/// Plain connectivity + geometry INPUT data, portable across Kokkos memory
/// spaces. Never validated on its own -- `NativeMeshView::finalize` (via
/// the builder or a direct construction) is what checks it.
template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct MeshArrays {
  using MemorySpaceT = MemorySpace;

  int dim = 0;  ///< topological dimension of every cell (1, 2 or 3)

  // ---- vertices --------------------------------------------------------
  /// LayoutRight in EVERY memory space: a rank-2 View's default layout is
  /// the space's own (LayoutRight on the host, LayoutLeft on CUDA), and
  /// `Kokkos::deep_copy` cannot change memory space and layout in one call,
  /// so `to_memory_space<Device>()` on a host mesh threw (KNOWN_ISSUES.md
  /// D54). Pinning the layout keeps the host arrays as they always were
  /// and makes the device copy a plain contiguous memcpy.
  Kokkos::View<double* [3], Kokkos::LayoutRight, MemorySpace> vertex_coords;
  Kokkos::View<std::int64_t*, MemorySpace> vertex_gid;
  Kokkos::View<int*, MemorySpace> vertex_owner;
  Kokkos::View<bool*, MemorySpace> vertex_is_owned;

  // ---- cells: CSR of vertex indices ------------------------------------
  Kokkos::View<int*, MemorySpace> cell_vertex_offsets;  ///< size ncells + 1
  Kokkos::View<int*, MemorySpace> cell_vertex_indices;  ///< local vertex idx
  Kokkos::View<std::uint8_t*, MemorySpace> cell_type;   ///< CellType, as a byte
  Kokkos::View<std::int64_t*, MemorySpace> cell_gid;
  Kokkos::View<int*, MemorySpace> cell_owner;
  Kokkos::View<bool*, MemorySpace> cell_is_owned;

  // ---- polyhedron faces: cell-to-face CSR, then face-to-vertex CSR ----
  // Meaningless (both offsets equal) for a non-Polyhedron cell.
  Kokkos::View<int*, MemorySpace> cell_face_offsets;    ///< size ncells + 1
  Kokkos::View<int*, MemorySpace> cell_face_indices;    ///< index into the face list
  Kokkos::View<bool*, MemorySpace> cell_face_flip;      ///< true: reverse the stored face
  Kokkos::View<int*, MemorySpace> face_vertex_offsets;  ///< size nfaces + 1
  Kokkos::View<int*, MemorySpace> face_vertex_indices;  ///< local vertex idx, canonical order

  [[nodiscard]] std::size_t vertex_count() const noexcept { return vertex_coords.extent(0); }
  [[nodiscard]] std::size_t cell_count() const noexcept {
    return cell_vertex_offsets.extent(0) == 0 ? 0 : cell_vertex_offsets.extent(0) - 1;
  }
  [[nodiscard]] std::size_t face_count() const noexcept {
    return face_vertex_offsets.extent(0) == 0 ? 0 : face_vertex_offsets.extent(0) - 1;
  }
};

namespace detail {
template <class DstView, class SrcView>
DstView copy_view(const SrcView& src) {
  DstView dst(Kokkos::view_alloc(src.label(), Kokkos::WithoutInitializing), src.extent(0));
  Kokkos::deep_copy(dst, src);
  return dst;
}
}  // namespace detail

/// Deep-copies every array of `src` into a freshly allocated, same-shaped
/// `MeshArrays<Target>` -- unconditionally, even when `Target == Source`
/// (a same-space copy is then just cheap, not free; this does not alias
/// `src`'s memory the way `Kokkos::create_mirror_view` would). This is the
/// one hook that makes the mesh portable to a GPU build: every generator
/// and the file reader build on the host, and a caller that wants the
/// result device-resident calls this once.
template <class Target, class Source>
[[nodiscard]] MeshArrays<Target> to_memory_space(const MeshArrays<Source>& src) {
  MeshArrays<Target> dst;
  dst.dim = src.dim;
  dst.vertex_coords = detail::copy_view<Kokkos::View<double* [3], Kokkos::LayoutRight, Target>>(src.vertex_coords);
  dst.vertex_gid = detail::copy_view<Kokkos::View<std::int64_t*, Target>>(src.vertex_gid);
  dst.vertex_owner = detail::copy_view<Kokkos::View<int*, Target>>(src.vertex_owner);
  dst.vertex_is_owned = detail::copy_view<Kokkos::View<bool*, Target>>(src.vertex_is_owned);
  dst.cell_vertex_offsets =
      detail::copy_view<Kokkos::View<int*, Target>>(src.cell_vertex_offsets);
  dst.cell_vertex_indices =
      detail::copy_view<Kokkos::View<int*, Target>>(src.cell_vertex_indices);
  dst.cell_type = detail::copy_view<Kokkos::View<std::uint8_t*, Target>>(src.cell_type);
  dst.cell_gid = detail::copy_view<Kokkos::View<std::int64_t*, Target>>(src.cell_gid);
  dst.cell_owner = detail::copy_view<Kokkos::View<int*, Target>>(src.cell_owner);
  dst.cell_is_owned = detail::copy_view<Kokkos::View<bool*, Target>>(src.cell_is_owned);
  dst.cell_face_offsets = detail::copy_view<Kokkos::View<int*, Target>>(src.cell_face_offsets);
  dst.cell_face_indices = detail::copy_view<Kokkos::View<int*, Target>>(src.cell_face_indices);
  dst.cell_face_flip = detail::copy_view<Kokkos::View<bool*, Target>>(src.cell_face_flip);
  dst.face_vertex_offsets =
      detail::copy_view<Kokkos::View<int*, Target>>(src.face_vertex_offsets);
  dst.face_vertex_indices =
      detail::copy_view<Kokkos::View<int*, Target>>(src.face_vertex_indices);
  return dst;
}

/// Host-side builder. Never partial: `build()` either returns a fully
/// finalized `MeshArrays<Kokkos::HostSpace>` or a `Failure` naming exactly
/// what was wrong, and touches no Kokkos view until then.
class NativeMeshBuilder {
 public:
  explicit NativeMeshBuilder(int dim) : dim_(dim) {}

  /// Appends a vertex; returns its local index (0-based, in append order).
  int add_vertex(std::int64_t gid, double x, double y, double z, int owner, bool is_owned);

  /// Appends a non-polyhedron cell over local vertex indices already added
  /// (`add_vertex`'s return values), in `cell_topology.hpp`'s corner order
  /// for `type`. Returns the cell's local index.
  int add_cell(CellType type, const std::vector<int>& local_vertices, std::int64_t gid,
              int owner, bool is_owned);

  /// Appends a polyhedron cell from its faces, each given as LOCAL vertex
  /// indices in THIS CELL's own outward-CCW order (cell_topology.hpp's
  /// convention). Internally deduplicates by each face's vertex GLOBAL id
  /// set: a face already recorded (by another cell that shares it) is
  /// reused, with an orientation flag recording whether this cell's order
  /// matches or reverses the stored one. Returns the cell's local index.
  int add_polyhedron_cell(const std::vector<std::vector<int>>& faces, std::int64_t gid,
                          int owner, bool is_owned);

  /// Validates and finalizes. Errors, never throws:
  ///   E-MESH-DIM               a cell's topological dimension disagrees
  ///                            with the builder's declared `dim`
  ///   E-MESH-CONNECTIVITY      a cell/face vertex index is out of range
  ///   E-MESH-POLYGON-*         from a polygon/polyhedron-face decomposition
  ///   E-MESH-CELL-NOT-STAR-SHAPED   from a polyhedron cell decomposition
  ///   E-MESH-POLYGON-FACE-MISMATCH  a polyhedron face named the same
  ///                            vertex SET a previous cell already
  ///                            recorded, in an order that is neither a
  ///                            cyclic match nor a cyclic reversal of the
  ///                            stored one (NM-10: ambiguous, never
  ///                            silently reused)
  [[nodiscard]] Result<MeshArrays<Kokkos::HostSpace>> build() const;

 private:
  int dim_;
  std::vector<double> vx_, vy_, vz_;
  std::vector<std::int64_t> vgid_;
  std::vector<int> vowner_;
  std::vector<bool> vowned_;

  std::vector<CellType> ctype_;
  std::vector<std::vector<int>> cverts_;
  std::vector<std::int64_t> cgid_;
  std::vector<int> cowner_;
  std::vector<bool> cowned_;
  // Parallel to ctype_/cverts_: which face indices (into faces_) belong to
  // this cell, and this cell's orientation flag for each. Empty vectors for
  // a non-polyhedron cell.
  std::vector<std::vector<int>> cfaces_;
  std::vector<std::vector<bool>> cface_flip_;

  // The deduplicated face list: each face's local vertex indices in its
  // OWN canonical (first-recorded) order.
  std::vector<std::vector<int>> faces_;
  // gid-sorted-tuple -> index into faces_, used only during building.
  std::vector<std::pair<std::vector<std::int64_t>, int>> face_index_;
  // Set the first time `find_or_add_face` sees a vertex SET it already
  // recorded but in neither the stored cyclic order nor its reversal;
  // `build()` returns this (code, message) rather than ever reaching the
  // geometry pass with a face reused under an order that does not
  // actually match (NM-10).
  std::optional<std::pair<std::string, std::string>> face_mismatch_error_;

  int find_or_add_face(const std::vector<int>& local_vertices_outward, bool* flip);
};

/// A cell's faces as `OrientedFace`s (geometry.hpp), read from a finalized
/// `MeshArrays` on host and this cell's own outward-CCW convention.
/// Works for both a fixed 3D shape (`cell_topology.hpp`'s tables) and a
/// `Polyhedron` (the mesh's own face CSR, applying `cell_face_flip`).
[[nodiscard]] std::vector<OrientedFace> oriented_faces_of_cell(
    const MeshArrays<Kokkos::HostSpace>& arrays, std::size_t cell);

/// The concrete, host-only `fuse::MeshView`. Constructed once from a
/// `MeshArrays<AnySpace>` (mirrored to host internally) and `my_rank`,
/// after which every query is O(1) or O(simplex count) against arrays
/// computed at construction time -- never recomputed per query.
class NativeMeshView final : public MeshView {
 public:
  /// Errors are reported through this Result, not the constructor, which is
  /// why construction goes through this static factory. Mirrors `arrays`
  /// to the host (a no-op alias when `MemorySpace` already is host-
  /// accessible) and validates/decomposes there -- see the file header.
  template <class MemorySpace>
  [[nodiscard]] static Result<std::unique_ptr<NativeMeshView>> create(
      const MeshArrays<MemorySpace>& arrays, int my_rank) {
    const auto host = to_memory_space<Kokkos::HostSpace>(arrays);
    return create_from_host(host, my_rank);
  }

  /// The host-only worker every `create<MemorySpace>` delegates to.
  [[nodiscard]] static Result<std::unique_ptr<NativeMeshView>> create_from_host(
      const MeshArrays<Kokkos::HostSpace>& arrays, int my_rank);

  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int dim) const override;
  [[nodiscard]] std::size_t owned_count(int dim) const override;
  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override;
  [[nodiscard]] Point centroid(int dim, std::size_t i) const override;
  [[nodiscard]] double measure(int dim, std::size_t i) const override;
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override;
  [[nodiscard]] std::size_t simplex_count(int dim, std::size_t i) const override;
  void simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const override;
  [[nodiscard]] Result<CellVertices> cell_vertices(int dim, std::size_t i) const override;

  // ---- accessors used by migration/partition/IO, beyond MeshView -------
  [[nodiscard]] std::size_t cell_count() const noexcept { return cell_type_.size(); }
  [[nodiscard]] std::size_t vertex_count() const noexcept { return vertex_gid_.size(); }
  [[nodiscard]] CellType cell_type_of(std::size_t c) const { return cell_type_[c]; }
  [[nodiscard]] const std::vector<int>& cell_vertices(std::size_t c) const {
    return cell_vertices_[c];
  }
  [[nodiscard]] const std::vector<std::vector<int>>& cell_faces(std::size_t c) const {
    return cell_faces_[c];
  }
  [[nodiscard]] double vertex_x(std::size_t v, int k) const { return vertex_xyz_[v][k]; }
  [[nodiscard]] std::int64_t vertex_gid_of(std::size_t v) const { return vertex_gid_[v]; }
  [[nodiscard]] int vertex_owner_of(std::size_t v) const { return vertex_owner_[v]; }
  [[nodiscard]] bool vertex_owned_of(std::size_t v) const { return vertex_owned_[v]; }
  [[nodiscard]] int cell_owner_of(std::size_t c) const { return cell_owner_[c]; }
  [[nodiscard]] bool cell_owned_of(std::size_t c) const { return cell_owned_[c]; }
  [[nodiscard]] Point cell_centroid_point(std::size_t c) const { return centroid_[c]; }
  [[nodiscard]] double cell_measure_value(std::size_t c) const { return measure_[c]; }

  /// Default-constructs an empty, unfinalized view. Public only so
  /// `std::make_unique` can reach it from `create_from_host`; every other
  /// caller should go through `create`/`create_from_host`.
  NativeMeshView() = default;

  // ---- device-resident geometry (docs/dev/native-mesh.md, "Execution
  // space") -----------------------------------------------------------
  //
  // `measure(dim,i)`/`centroid(dim,i)` above keep serving the host arrays
  // this file has always computed -- `Result`/`Failure`/`fmt` validation
  // (canonical_fan_triangulation, canonical_cell_tetrahedra) is host-only by
  // this codebase's own standing convention, and that validation pass
  // already produces the host measure/centroid as an unavoidable byproduct,
  // so there is no separate "mirror rebuild" step to make lazy: the host
  // array IS the validation's own output, computed once, at construction.
  //
  // What is NEW is `device_measure_`/`device_centroid_` below: the SAME two
  // per-cell maps, INDEPENDENTLY recomputed by a Kokkos kernel dispatched on
  // Kokkos::DefaultExecutionSpace directly from this view's validated
  // simplex corners and vertex connectivity (`create_from_host`'s last
  // step), landing in `Kokkos::DefaultExecutionSpace::memory_space` -- real
  // device memory on a CUDA/HIP/SYCL build. A caller (or a verification
  // test) that wants the values where a device-resident kernel can read
  // them without a host round trip uses these; everything above the backend
  // is unaffected; it never sees this type.
 public:
  using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

  /// Per-cell measure, computed on the device (see above). Agrees with
  /// `measure(dim_, i)` to 1e-14 relative on a build with a real device
  /// backend (nvcc's FMA contraction, the same tolerance
  /// docs/dev/device-resident-arcs.md's reproducibility contract already
  /// accepts for RelaxFunctor/RemapMultiFunctor), bit-identically on a
  /// host-only build (Kokkos::DefaultExecutionSpace == the host space, and
  /// the kernel sums each cell's simplices in the same order the host loop
  /// does). Empty when `cell_count() == 0`.
  [[nodiscard]] const Kokkos::View<double*, DeviceMemSpace>& device_measure() const noexcept {
    return device_measure_;
  }

  /// Per-cell centroid (vertex average, matching `centroid(dim_, i)`'s own
  /// convention), computed on the device directly from `vertex_coords` and
  /// the cell-to-vertex CSR -- no simplex decomposition needed for this one,
  /// since a vertex average does not go through it even on the host (see
  /// `centroid(dim,i)`'s file comment above). Column 0/1/2 are x/y/z.
  [[nodiscard]] const Kokkos::View<double* [3], DeviceMemSpace>& device_centroid() const noexcept {
    return device_centroid_;
  }

 private:
  Kokkos::View<double*, DeviceMemSpace> device_measure_;
  Kokkos::View<double* [3], DeviceMemSpace> device_centroid_;

  void check(int dim, std::size_t i, const char* what) const;

  int dim_ = 0;
  int my_rank_ = 0;

  std::vector<std::array<double, 3>> vertex_xyz_;
  std::vector<std::int64_t> vertex_gid_;
  std::vector<int> vertex_owner_;
  std::vector<bool> vertex_owned_;

  std::vector<CellType> cell_type_;
  std::vector<std::vector<int>> cell_vertices_;
  std::vector<std::vector<std::vector<int>>> cell_faces_;  ///< [cell][face][local vtx]
  std::vector<std::int64_t> cell_gid_;
  std::vector<int> cell_owner_;
  std::vector<bool> cell_owned_;

  // Derived at construction: per-cell simplex decomposition (as vertex
  // triples/quadruples of PHYSICAL POINTS, since a simplex's corners are
  // not necessarily mesh vertices' local indices after decomposition would
  // need -- they always ARE mesh vertex positions here, but simplex_points
  // returns Points, so storing points directly avoids a second lookup).
  std::vector<std::vector<std::array<Point, 4>>> cell_simplices_;  ///< [cell][simplex]
  std::vector<Point> centroid_;
  std::vector<double> measure_;
};

}  // namespace fuse::native
