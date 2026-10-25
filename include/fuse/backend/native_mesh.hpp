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
  /// so `to_memory_space<Device>()` on a host mesh threw. Pinning the layout keeps the host arrays
  /// unchanged and makes the device copy a plain contiguous memcpy.
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
  /// Validates, never throws: `E-MESH-DIM` if `arrays.dim` is outside
  /// `[0, 3]`, `E-MESH-CONNECTIVITY` if a fixed-shape cell's corner count
  /// disagrees with its type or a corner names a vertex index out of
  /// range (NM-8: every in-tree producer already goes through
  /// `NativeMeshBuilder::build()`, which checks the same two things, so
  /// these guard only a caller that reaches this factory directly).
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
    ensure_host_mirror();
    return cell_vertices_[c];
  }
  [[nodiscard]] const std::vector<std::vector<int>>& cell_faces(std::size_t c) const {
    ensure_host_mirror();
    return cell_faces_[c];
  }
  [[nodiscard]] double vertex_x(std::size_t v, int k) const {
    ensure_host_mirror();
    return vertex_xyz_[v][k];
  }
  [[nodiscard]] std::int64_t vertex_gid_of(std::size_t v) const { return vertex_gid_[v]; }
  [[nodiscard]] int vertex_owner_of(std::size_t v) const { return vertex_owner_[v]; }
  [[nodiscard]] bool vertex_owned_of(std::size_t v) const { return vertex_owned_[v]; }
  [[nodiscard]] int cell_owner_of(std::size_t c) const { return cell_owner_[c]; }
  [[nodiscard]] bool cell_owned_of(std::size_t c) const { return cell_owned_[c]; }
  [[nodiscard]] Point cell_centroid_point(std::size_t c) const {
    ensure_host_mirror();
    return centroid_[c];
  }
  [[nodiscard]] double cell_measure_value(std::size_t c) const {
    ensure_host_mirror();
    return measure_[c];
  }

  /// Default-constructs an empty, unfinalized view. Public only so
  /// `std::make_unique` can reach it from `create_from_host`; every other
  /// caller should go through `create`/`create_from_host`.
  NativeMeshView() = default;

  // ---- device-resident geometry (docs/dev/native-mesh.md, "Execution
  // space") -----------------------------------------------------------
  //
  // DEVICE-PRIMARY. `device_arrays_` below -- a deep copy of the
  // constructor's input into `DeviceMemSpace` -- is this view's PRIMARY
  // storage: real device memory on a CUDA/HIP/SYCL build, the same space
  // `device_measure_`/`device_centroid_`/the device simplex CSR already
  // live in. Validation (connectivity ranges, `canonical_fan_triangulation`,
  // `canonical_cell_tetrahedra`) still runs exactly once, host-only, at
  // construction -- `Result`/`Failure`/`fmt` are not device-callable in this
  // codebase's own convention -- but its host-side scratch arrays are not
  // retained: `create_from_host` discards them once the device arrays below
  // are populated, so a `NativeMeshView` that nothing has queried through
  // the host-serving accessors holds no host mirror at all, only
  // `device_arrays_` and the device-resident derived geometry.
  //
  // ON-DEMAND HOST MIRRORS. `vertex_xyz_`/`cell_vertices_`/`cell_faces_`/
  // `cell_simplices_`/`centroid_`/`measure_` -- every host array
  // `measure(dim,i)`/`centroid(dim,i)`/`simplex_points`/`cell_vertices`/
  // `vertex_x` serve -- are built from `device_arrays_` (plus the device
  // measure/centroid/simplex CSR, already-validated so this step cannot
  // fail) the FIRST time any of those accessors is called, by
  // `ensure_host_mirror()`, and cached for the view's lifetime after that.
  // `host_mirrors_materialized()` is this count (0 before the first call,
  // 1 after), exposed for exactly this: a test that constructs a view and
  // checks no host array exists until something asks for one.
 public:
  using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

  /// This view's primary storage: the input `MeshArrays`, deep-copied to
  /// `DeviceMemSpace` at construction. Every on-demand host mirror below is
  /// rebuilt FROM this (and from the device-resident derived geometry), not
  /// from the constructor's original host argument, so a caller holding
  /// only this view can still reconstruct every host-serving accessor.
  [[nodiscard]] const MeshArrays<DeviceMemSpace>& device_arrays() const noexcept {
    return device_arrays_;
  }

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

  /// The validated per-cell simplex decomposition, flattened into a device
  /// CSR: cell `c`'s simplices are rows
  /// `device_simplex_offsets()(c) .. device_simplex_offsets()(c + 1)` of
  /// `device_simplex_corners()`, corner `k` (`k < dimension() + 1`) of
  /// simplex row `s` at `device_simplex_corners()(s, k, 0..2)` (x, y, z).
  /// This is the SAME CSR `device_measure()`'s kernel reduces -- previously
  /// built and discarded inside `create_from_host` (docs/dev/native-mesh.md
  /// used to say so explicitly); retained here so a caller building a
  /// spatial index (`fuse::search::element_boxes_device_resident`,
  /// `BoxIndex::build_device_resident`) never has to touch the host to get
  /// a native mesh's element boxes. Size `cell_count() + 1`; empty when
  /// `cell_count() == 0`.
  [[nodiscard]] const Kokkos::View<int*, DeviceMemSpace>& device_simplex_offsets() const noexcept {
    return device_simplex_offsets_;
  }

  /// See `device_simplex_offsets()`. Every row holds 4 corner slots
  /// regardless of `dimension()` -- a 1D/2D simplex's unused trailing
  /// corners are zero-filled, matching `cell_simplices_`'s own
  /// `std::array<Point, 4>` convention; a reader uses only the first
  /// `dimension() + 1` of them.
  [[nodiscard]] const Kokkos::View<double* [4][3], Kokkos::LayoutRight, DeviceMemSpace>&
  device_simplex_corners() const noexcept {
    return device_simplex_corners_;
  }

  /// 0 before any host-serving accessor (`measure`, `centroid`,
  /// `simplex_points`, `cell_vertices`, `vertex_x`, ...) has been called on
  /// this view, 1 after the first one materializes and caches the host
  /// mirror (see the class comment above). Exposed for tests only.
  [[nodiscard]] int host_mirrors_materialized() const noexcept {
    return host_mirror_materialized_ ? 1 : 0;
  }

 private:
  /// Builds `vertex_xyz_`/`cell_vertices_`/`cell_faces_`/`cell_simplices_`/
  /// `centroid_`/`measure_` from `device_arrays_` and the device-resident
  /// derived geometry, exactly once (a no-op after the first call). Never
  /// fails: the data it mirrors was already validated at construction.
  void ensure_host_mirror() const;

  Kokkos::View<double*, DeviceMemSpace> device_measure_;
  Kokkos::View<double* [3], DeviceMemSpace> device_centroid_;
  Kokkos::View<int*, DeviceMemSpace> device_simplex_offsets_;
  Kokkos::View<double* [4][3], Kokkos::LayoutRight, DeviceMemSpace> device_simplex_corners_;
  MeshArrays<DeviceMemSpace> device_arrays_;  ///< PRIMARY storage

  void check(int dim, std::size_t i, const char* what) const;

  int dim_ = 0;
  int my_rank_ = 0;

  // ---- cheap per-entity bookkeeping, eager -------------------------------
  // Plain scalars/flags copied once from the constructor's host argument --
  // no geometry, no fallible decomposition -- so these are not part of the
  // on-demand mirror above; they exist unconditionally from construction,
  // same as before this file's device-primary change.
  std::vector<std::int64_t> vertex_gid_;
  std::vector<int> vertex_owner_;
  std::vector<bool> vertex_owned_;

  std::vector<CellType> cell_type_;
  std::vector<std::int64_t> cell_gid_;
  std::vector<int> cell_owner_;
  std::vector<bool> cell_owned_;

  // ---- on-demand host mirror, lazy ---------------------------------------
  // `mutable`: materialized from inside `const` accessors, the first time
  // one of them is actually called (see `ensure_host_mirror()`).
  mutable bool host_mirror_materialized_ = false;
  mutable std::vector<std::array<double, 3>> vertex_xyz_;
  mutable std::vector<std::vector<int>> cell_vertices_;
  mutable std::vector<std::vector<std::vector<int>>> cell_faces_;  ///< [cell][face][local vtx]
  // [cell][simplex]; a simplex's corners are PHYSICAL POINTS (always 4
  // slots, trailing ones zero for a 1D/2D cell) -- see
  // `device_simplex_corners()`'s comment for why 4 always.
  mutable std::vector<std::vector<std::array<Point, 4>>> cell_simplices_;
  mutable std::vector<Point> centroid_;
  mutable std::vector<double> measure_;
};

}  // namespace fuse::native
