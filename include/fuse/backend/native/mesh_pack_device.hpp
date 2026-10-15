#pragma once

/// Device-capable pack for the native mesh's migration and halo wire
/// format (docs/dev/device-resident-arcs.md's Phase 2: "halo and migration
/// pack/unpack ... producing the device CSR directly"). `pack_for_migration`
/// (native_migration.cpp) and `pack_halo_cells` (native_halo.cpp) are now
/// wired directly to `pack_to`/`pack_fanout` below (Migration track,
/// 2026-10-15): both read `NativeMeshView::device_arrays()` -- plain CSR
/// data, valid in any Kokkos memory space -- instead of walking the VIEW's
/// own host-only, VALIDATED accessors (`cell_vertices`/`vertex_x`/
/// `cell_faces`, which materialize the lazy host mirror on first call;
/// docs/dev/native-mesh.md: `Result`/`Failure`/`fmt` are not
/// device-callable, so `NativeMeshView::create_from_host`'s VALIDATION
/// stays a one-time, host-side cost by design, but nothing past that
/// needs the mirror an old caller's `cell_vertices`/`vertex_x` calls used
/// to force). The owned-block numbering, every record's exact byte size,
/// and the byte-level write all run as Kokkos kernels on `Space`;
/// `exchange_migration_buffers`'s MPI alltoall is the GPU-aware exchange
/// track's own follow-up (this same track, see native_migration.hpp's
/// header comment on `mpi_is_gpu_aware`), not this header's concern.
///
/// `pack_to<kHaloRecord, Space>` is the named seam this is: a caller on
/// `Space == Kokkos::DefaultExecutionSpace::memory_space` gets
/// `Kokkos::View<std::byte*, Space>` buffers with no host round trip for
/// the per-cell data; a caller on `Space == Kokkos::HostSpace` gets the
/// same algorithm running on the host, byte-for-byte compatible with the
/// existing wire format (read_pod_at/append_pod_at below are the same
/// byte-at-a-time convention native_halo.cpp's own append_pod/read_pod
/// use, duplicated as KOKKOS_INLINE_FUNCTION so the identical loop
/// compiles for either Space's execution space -- not std::memcpy, which
/// device code can call under nvcc in practice but which is not the
/// portable-C++ byte loop the host side already established).
///
/// `kHaloRecord` selects which wire format: `false` is
/// `pack_for_migration`'s (gid, type, vertex closure, Polyhedron face
/// CSR); `true` is `pack_halo_cells`'s, the same record with two leading
/// fields (owner rank, owned-block index) a migrated cell never carries.
/// Both share one implementation here because they differ only in that
/// header, exactly as native_halo.cpp's own comment on `pack_one_halo_cell`
/// states.
///
/// NOT YET DONE: the receiving side's dedup-by-global-id merge
/// (`merge_halo_cells`'s cell dedup, `unpack_migrated_mesh`'s vertex
/// dedup) is not templated here. See this track's report
/// (report-weight-builders.md) for why and what the next increment is.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/backend/native/cell_topology.hpp"
#include "fuse/backend/native_mesh.hpp"

namespace fuse::native::device {

// ---- portable record I/O: byte-at-a-time, valid on host or device -------

template <class T>
KOKKOS_INLINE_FUNCTION void append_pod_at(std::byte* dst, std::size_t& pos, const T& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* p = reinterpret_cast<const std::byte*>(&v);
  for (std::size_t k = 0; k < sizeof(T); ++k) dst[pos + k] = p[k];
  pos += sizeof(T);
}

template <class T>
KOKKOS_INLINE_FUNCTION T read_pod_at(const std::byte* src, std::size_t& pos) {
  static_assert(std::is_trivially_copyable_v<T>);
  T v{};
  auto* out = reinterpret_cast<std::byte*>(&v);
  for (std::size_t k = 0; k < sizeof(T); ++k) out[k] = src[pos + k];
  pos += sizeof(T);
  return v;
}

/// Byte size of cell `c`'s record, computed from the CSR row widths alone
/// -- no cell needs a second pass to discover its own size, which is what
/// lets the per-destination offset scan in `pack_to` run as an ordinary
/// `Kokkos::parallel_scan` rather than needing a variable-length append.
template <bool kHaloRecord, class Space>
KOKKOS_INLINE_FUNCTION std::size_t record_size(const MeshArrays<Space>& mesh, int c) {
  std::size_t sz = sizeof(std::int64_t) + sizeof(std::uint8_t) + sizeof(std::int32_t);
  if constexpr (kHaloRecord) sz += sizeof(std::int32_t) + sizeof(std::int64_t);
  const int v0 = mesh.cell_vertex_offsets(c);
  const int v1 = mesh.cell_vertex_offsets(c + 1);
  sz += static_cast<std::size_t>(v1 - v0) * (sizeof(std::int64_t) + 3 * sizeof(double));
  if (static_cast<CellType>(mesh.cell_type(c)) == CellType::Polyhedron) {
    sz += sizeof(std::int32_t);
    const int f0 = mesh.cell_face_offsets(c);
    const int f1 = mesh.cell_face_offsets(c + 1);
    for (int fi = f0; fi < f1; ++fi) {
      const int face_idx = mesh.cell_face_indices(fi);
      const int fv0 = mesh.face_vertex_offsets(face_idx);
      const int fv1 = mesh.face_vertex_offsets(face_idx + 1);
      sz += sizeof(std::int32_t) + static_cast<std::size_t>(fv1 - fv0) * sizeof(std::int32_t);
    }
  }
  return sz;
}

/// Writes cell `c`'s record at `dst[pos0..)`. `owner_rank`/`owned_block_idx`
/// are read only when `kHaloRecord`. The Polyhedron face loop applies the
/// SAME flip rule `oriented_faces_of_cell` (native_mesh.cpp) does -- walk
/// the face's stored vertex list in reverse when this cell's own
/// orientation disagrees with the face's canonical (first-recorded)
/// direction -- then translates each face vertex's MESH-local index to
/// this cell's own closure position (0..n-1) by a linear scan over the
/// cell's own corner range: a device-friendly replacement for the host
/// packer's `std::unordered_map<int,int> local_to_closure`, correct
/// because a cell's own corner count is the polygon/polyhedron degree, a
/// handful of entries, never the mesh size.
template <bool kHaloRecord, class Space>
KOKKOS_INLINE_FUNCTION void write_record(const MeshArrays<Space>& mesh, int c, std::byte* dst,
                                         std::size_t pos0, std::int32_t owner_rank,
                                         std::int64_t owned_block_idx) {
  std::size_t pos = pos0;
  if constexpr (kHaloRecord) {
    append_pod_at<std::int32_t>(dst, pos, owner_rank);
    append_pod_at<std::int64_t>(dst, pos, owned_block_idx);
  }
  append_pod_at<std::int64_t>(dst, pos, mesh.cell_gid(c));
  const auto type_byte = mesh.cell_type(c);
  append_pod_at<std::uint8_t>(dst, pos, type_byte);
  const int v0 = mesh.cell_vertex_offsets(c);
  const int v1 = mesh.cell_vertex_offsets(c + 1);
  append_pod_at<std::int32_t>(dst, pos, v1 - v0);
  for (int k = v0; k < v1; ++k) {
    const int lv = mesh.cell_vertex_indices(k);
    append_pod_at<std::int64_t>(dst, pos, mesh.vertex_gid(lv));
    append_pod_at<double>(dst, pos, mesh.vertex_coords(lv, 0));
    append_pod_at<double>(dst, pos, mesh.vertex_coords(lv, 1));
    append_pod_at<double>(dst, pos, mesh.vertex_coords(lv, 2));
  }
  if (static_cast<CellType>(type_byte) == CellType::Polyhedron) {
    const int f0 = mesh.cell_face_offsets(c);
    const int f1 = mesh.cell_face_offsets(c + 1);
    append_pod_at<std::int32_t>(dst, pos, f1 - f0);
    for (int fi = f0; fi < f1; ++fi) {
      const int face_idx = mesh.cell_face_indices(fi);
      const bool flip = mesh.cell_face_flip(fi);
      const int fv0 = mesh.face_vertex_offsets(face_idx);
      const int fv1 = mesh.face_vertex_offsets(face_idx + 1);
      const int fvcount = fv1 - fv0;
      append_pod_at<std::int32_t>(dst, pos, fvcount);
      for (int k = 0; k < fvcount; ++k) {
        const int kk = flip ? (fvcount - 1 - k) : k;
        const int target_local_vertex = mesh.face_vertex_indices(fv0 + kk);
        int closure_pos = -1;
        for (int kv = v0; kv < v1; ++kv) {
          if (mesh.cell_vertex_indices(kv) == target_local_vertex) {
            closure_pos = kv - v0;
            break;
          }
        }
        append_pod_at<std::int32_t>(dst, pos, closure_pos);
      }
    }
  }
}

/// WIRE FORMAT v2 (Migration track, 2026-10-15; KNOWN_ISSUES D51's own
/// narrowing: "the wire format's variable-length, unprefixed records
/// make per-record boundary discovery... a serial walk... it would need
/// either a length-prefixed wire-format change"). Every buffer `pack_to`/
/// `pack_fanout` produce now opens with a FIXED-SIZE header a receiver
/// can read with one `parallel_for` indexed by record, never a
/// sequential walk of the variable-length payload that follows it:
///
///   [int64 record_count]
///   [record_count + 1 int64 byte offsets, ABSOLUTE from the start of
///    this buffer -- record r's bytes span [offsets[r], offsets[r+1]),
///    and offsets[record_count] is the buffer's own total size]
///   [payload: every record, back to back, unchanged from before this
///    header existed]
///
/// `alloc_buffer_with_header` below builds this from the SAME `contrib`/
/// `payload_offset` arrays `pack_to`/`pack_fanout` already compute for
/// the payload write step -- one more `parallel_scan` (contributing index
/// -> compacted 0..record_count-1 position) and two small `parallel_for`
/// writes, then returns the allocated buffer (header slots already
/// written) and that compacted `record_index`, which the caller's own
/// write step uses instead of `payload_offset` directly (offset by this
/// buffer's own `header_bytes = buf.extent(0) - total_payload`).
template <class Space>
[[nodiscard]] std::pair<Kokkos::View<std::byte*, Space>, Kokkos::View<int*, Space>>
alloc_buffer_with_header(const Kokkos::View<std::size_t*, Space>& contrib,
                         const Kokkos::View<std::size_t*, Space>& payload_offset,
                         std::size_t total_payload, int n) {
  using ExecSpace = typename Space::execution_space;

  Kokkos::View<int*, Space> record_index(
      Kokkos::view_alloc("fuse::native::device::header_record_index", Kokkos::WithoutInitializing),
      n);
  int n_records = 0;
  Kokkos::parallel_scan(
      "fuse::native::device::header_record_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, int& running, const bool final_pass) {
        const bool has_record = contrib(i) != 0;
        if (final_pass) record_index(i) = running;
        if (has_record) running += 1;
      },
      n_records);

  // Zero records: an entirely empty buffer, not a 16-byte header around no
  // payload -- preserves the pre-header wire contract a destination that
  // receives nothing already relied on (`pack_halo_cells`'s own "nothing is
  // ever sent to one's own rank" -> an empty buffer, native_halo_test.cpp)
  // and avoids shipping header bytes over MPI for an empty send.
  // `read_record_count_host`/`parse_buffers` (mesh_merge_device.hpp) treat
  // a zero-extent buffer as zero records without reading it.
  if (n_records == 0) {
    return {Kokkos::View<std::byte*, Space>(), record_index};
  }

  const std::size_t header_bytes =
      sizeof(std::int64_t) * static_cast<std::size_t>(2 + n_records);
  Kokkos::View<std::byte*, Space> buf(
      Kokkos::view_alloc("fuse::native::device::pack_buf", Kokkos::WithoutInitializing),
      header_bytes + total_payload);

  Kokkos::parallel_for(
      "fuse::native::device::header_write_count", Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t pos = 0;
        append_pod_at<std::int64_t>(buf.data(), pos, static_cast<std::int64_t>(n_records));
      });
  Kokkos::parallel_for(
      "fuse::native::device::header_write_offsets", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i) {
        if (contrib(i) == 0) return;
        std::size_t pos = sizeof(std::int64_t) * static_cast<std::size_t>(1 + record_index(i));
        append_pod_at<std::int64_t>(buf.data(), pos,
                                    static_cast<std::int64_t>(header_bytes + payload_offset(i)));
      });
  Kokkos::parallel_for(
      "fuse::native::device::header_write_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        std::size_t pos = sizeof(std::int64_t) * static_cast<std::size_t>(1 + n_records);
        append_pod_at<std::int64_t>(buf.data(), pos,
                                    static_cast<std::int64_t>(header_bytes + total_payload));
      });

  return {buf, record_index};
}

/// Packs every cell `c` with `cell_is_owned(c)` and
/// `dest_of_owned(owned_block_index(c)) == r` into a freshly allocated
/// `Kokkos::View<std::byte*, Space>` per destination rank `r`, entirely as
/// Kokkos kernels on `Space`. `dest_of_owned` is indexed by OWNED-BLOCK
/// position (ascending local cell index restricted to `cell_is_owned`),
/// matching `pack_for_migration`/`pack_halo_cells`'s own host convention
/// exactly, so a caller with a per-owned-cell destination array (an RCB
/// partition result, or `covering_set_destinations`'s flattened output)
/// hands it here unchanged. `owner_rank_of`/`owned_block_idx_of` (also
/// per-owned-cell) are read only when `kHaloRecord`; omit them for a
/// migration pack.
///
/// Cost: one `Kokkos::parallel_scan` for the owned-block numbering, one
/// `Kokkos::parallel_for` for per-cell record sizes, then for each of
/// `num_ranks` destinations one `parallel_for` + `parallel_scan` +
/// `parallel_reduce` + `parallel_for` -- O(num_ranks * cell_count())
/// total. Fine at this track's test scale (a handful of ranks, a few
/// hundred cells); a single bucketed (radix) assignment across all ranks
/// in one pass would remove the rank factor for a large communicator and
/// is future work, not needed yet.
template <bool kHaloRecord, class Space>
[[nodiscard]] std::vector<Kokkos::View<std::byte*, Space>> pack_to(
    const MeshArrays<Space>& mesh, const Kokkos::View<int*, Space>& dest_of_owned,
    int num_ranks, const Kokkos::View<std::int32_t*, Space>& owner_rank_of = {},
    const Kokkos::View<std::int64_t*, Space>& owned_block_idx_of = {}) {
  using ExecSpace = typename Space::execution_space;
  const int n = static_cast<int>(mesh.cell_count());
  const MeshArrays<Space> mesh_local = mesh;

  Kokkos::View<int*, Space> owned_idx("fuse::native::device::owned_idx", n);
  Kokkos::View<std::size_t*, Space> sizes("fuse::native::device::sizes", n);
  Kokkos::View<int*, Space> dest_full("fuse::native::device::dest_full", n);
  auto is_owned = mesh_local.cell_is_owned;

  Kokkos::parallel_scan(
      "fuse::native::device::owned_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int c, int& running, const bool final_pass) {
        const bool owned = is_owned(c);
        if (final_pass) owned_idx(c) = running;
        if (owned) running += 1;
      });

  Kokkos::parallel_for(
      "fuse::native::device::record_sizes_and_dest", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int c) {
        if (is_owned(c)) {
          sizes(c) = record_size<kHaloRecord>(mesh_local, c);
          dest_full(c) = dest_of_owned(owned_idx(c));
        } else {
          sizes(c) = 0;
          dest_full(c) = -1;
        }
      });

  std::vector<Kokkos::View<std::byte*, Space>> result(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    Kokkos::View<std::size_t*, Space> contrib("fuse::native::device::contrib", n);
    Kokkos::parallel_for(
        "fuse::native::device::contrib_fill", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int c) { contrib(c) = (dest_full(c) == r) ? sizes(c) : 0; });

    Kokkos::View<std::size_t*, Space> offset("fuse::native::device::offset", n);
    Kokkos::parallel_scan(
        "fuse::native::device::offset_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int c, std::size_t& running, const bool final_pass) {
          const std::size_t val = contrib(c);
          if (final_pass) offset(c) = running;
          running += val;
        });

    std::size_t total = 0;
    Kokkos::parallel_reduce(
        "fuse::native::device::total_reduce", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int c, std::size_t& acc) { acc += contrib(c); }, total);

    auto buf_and_index = alloc_buffer_with_header<Space>(contrib, offset, total, n);
    Kokkos::View<std::byte*, Space> buf = buf_and_index.first;
    const std::size_t header_bytes = buf.extent(0) - total;
    // The `if constexpr` dispatch is hoisted OUT of the lambda body, into two
    // separate kernel launches, rather than written as a branch inside one
    // `KOKKOS_LAMBDA` (which is what a first attempt did): nvcc's extended
    // `__host__ __device__` lambda rejects a variable's FIRST capture
    // occurring inside an `if constexpr` branch ("cannot first-capture
    // variable in constexpr-if context") -- measured on this track's own
    // device build, not a style preference. Each lambda below captures
    // exactly what it needs unconditionally, which both compiles under
    // nvcc and is the clearer reading anyway.
    if constexpr (kHaloRecord) {
      Kokkos::parallel_for(
          "fuse::native::device::write_records_halo", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const int c) {
            if (contrib(c) == 0) return;
            write_record<kHaloRecord>(mesh_local, c, buf.data(), header_bytes + offset(c),
                                      owner_rank_of(owned_idx(c)), owned_block_idx_of(owned_idx(c)));
          });
    } else {
      Kokkos::parallel_for(
          "fuse::native::device::write_records_migration", Kokkos::RangePolicy<ExecSpace>(0, n),
          KOKKOS_LAMBDA(const int c) {
            if (contrib(c) == 0) return;
            write_record<kHaloRecord>(mesh_local, c, buf.data(), header_bytes + offset(c), 0, 0);
          });
    }
    result[static_cast<std::size_t>(r)] = buf;
  }
  return result;
}

/// The many-to-many generalization `pack_to` cannot express: a migrated
/// cell has exactly ONE destination, but a halo copy's covering set
/// (`covering_set_destinations`, native_halo.hpp) may send the SAME owned
/// cell to SEVERAL destination ranks, so no `dest_of_owned(ob) == r`
/// equality test can name every destination of a cell at once. The domain
/// here is not the cell set but the flattened (cell, destination rank)
/// SEND list itself -- `cell_of_send(s)`/`dest_rank_of_send(s)` give send
/// index `s`'s local cell index and destination, with `n_sends` entries
/// total (>= the owned cell count when a cell is replicated to more than
/// one destination, <= it when some owned cells are sent nowhere).
/// `owner_rank_of_cell`/`owned_block_idx_of_cell` are indexed by LOCAL
/// CELL index directly (size `mesh.cell_count()`), not by send index or
/// owned-block position, since `pack_halo_cells`'s own owner rank is one
/// constant (`my_rank`) and `owned_block_index` is already a per-local-
/// cell lookup (native_halo.hpp's `owned_block_index`) -- the caller's
/// existing arrays need no re-indexing to become these views' host
/// mirrors. Read only when `kHaloRecord`; omit both for a hypothetical
/// fanout migration pack (not a case this track's two real callers need:
/// `pack_for_migration` itself still goes through `pack_to`, whose
/// one-destination-per-cell model is exactly what migration is).
///
/// Cost: one `parallel_for` over `n_sends` for record sizes, then per
/// destination rank the same `parallel_for` + `parallel_scan` +
/// `parallel_reduce` + `parallel_for` shape `pack_to` uses, over `n_sends`
/// instead of `cell_count()` -- `O(num_ranks * n_sends)`, matching
/// `pack_to`'s own cost note at this track's test scale.
template <bool kHaloRecord, class Space>
[[nodiscard]] std::vector<Kokkos::View<std::byte*, Space>> pack_fanout(
    const MeshArrays<Space>& mesh, const Kokkos::View<int*, Space>& cell_of_send,
    const Kokkos::View<int*, Space>& dest_rank_of_send, int num_ranks,
    const Kokkos::View<std::int32_t*, Space>& owner_rank_of_cell = {},
    const Kokkos::View<std::int64_t*, Space>& owned_block_idx_of_cell = {}) {
  using ExecSpace = typename Space::execution_space;
  const int n_sends = static_cast<int>(cell_of_send.extent(0));
  const MeshArrays<Space> mesh_local = mesh;

  Kokkos::View<std::size_t*, Space> sizes("fuse::native::device::fanout_sizes", n_sends);
  Kokkos::parallel_for(
      "fuse::native::device::fanout_record_sizes", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
      KOKKOS_LAMBDA(const int s) {
        sizes(s) = record_size<kHaloRecord>(mesh_local, cell_of_send(s));
      });

  std::vector<Kokkos::View<std::byte*, Space>> result(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    Kokkos::View<std::size_t*, Space> contrib("fuse::native::device::fanout_contrib", n_sends);
    Kokkos::parallel_for(
        "fuse::native::device::fanout_contrib_fill", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
        KOKKOS_LAMBDA(const int s) { contrib(s) = (dest_rank_of_send(s) == r) ? sizes(s) : 0; });

    Kokkos::View<std::size_t*, Space> offset("fuse::native::device::fanout_offset", n_sends);
    Kokkos::parallel_scan(
        "fuse::native::device::fanout_offset_scan", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
        KOKKOS_LAMBDA(const int s, std::size_t& running, const bool final_pass) {
          const std::size_t val = contrib(s);
          if (final_pass) offset(s) = running;
          running += val;
        });

    std::size_t total = 0;
    Kokkos::parallel_reduce(
        "fuse::native::device::fanout_total_reduce", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
        KOKKOS_LAMBDA(const int s, std::size_t& acc) { acc += contrib(s); }, total);

    auto buf_and_index = alloc_buffer_with_header<Space>(contrib, offset, total, n_sends);
    Kokkos::View<std::byte*, Space> buf = buf_and_index.first;
    const std::size_t header_bytes = buf.extent(0) - total;
    // Same nvcc first-capture-under-if-constexpr workaround `pack_to` above
    // documents: two launches, not one branching lambda.
    if constexpr (kHaloRecord) {
      Kokkos::parallel_for(
          "fuse::native::device::fanout_write_halo", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
          KOKKOS_LAMBDA(const int s) {
            if (contrib(s) == 0) return;
            const int c = cell_of_send(s);
            write_record<kHaloRecord>(mesh_local, c, buf.data(), header_bytes + offset(s),
                                      owner_rank_of_cell(c), owned_block_idx_of_cell(c));
          });
    } else {
      Kokkos::parallel_for(
          "fuse::native::device::fanout_write_migration", Kokkos::RangePolicy<ExecSpace>(0, n_sends),
          KOKKOS_LAMBDA(const int s) {
            if (contrib(s) == 0) return;
            write_record<kHaloRecord>(mesh_local, cell_of_send(s), buf.data(),
                                      header_bytes + offset(s), 0, 0);
          });
    }
    result[static_cast<std::size_t>(r)] = buf;
  }
  return result;
}

}  // namespace fuse::native::device
