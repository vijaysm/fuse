#pragma once

/// Device-resident dedup-by-global-id: the receive-side
/// MERGE decision `unpack_migrated_mesh`'s vertex dedup and
/// `merge_halo_cells`'s cell dedup both make -- "is this global id already
/// held, and if not, which ONE of possibly several incoming copies is the
/// canonical new record" -- as Kokkos kernels on `Space`, with no
/// `std::unordered_map`/`std::unordered_set` and no host round trip.
///
/// ALGORITHM (stated here, not assumed).
/// A sort-and-compact pass, not a sorted binary search: concatenate
/// `existing_gids` (already held, E entries) and `incoming_gids` (candidates
/// from the wire, N entries, possibly containing cross-sender duplicates of
/// each other AND of an existing entry) into one (E + N)-length key array,
/// `Kokkos::Experimental::sort_by_key` it ascending while carrying a signed
/// tag that encodes BOTH which side a key came from and its original
/// position, then ONE `Kokkos::parallel_scan` over the sorted order assigns
/// every DISTINCT key value exactly one ascending "slot" (a run boundary is
/// `keys(p) != keys(p - 1)`). A binary search would answer "is this gid
/// already held" against `existing_gids` alone in O(N log E) without
/// touching `incoming_gids`'s own cross-duplicates at all, which is cheaper
/// when E >> N (the common halo case: a few new ghost cells against a
/// large existing mesh) but does not by itself resolve two incoming copies
/// of the SAME new gid to one slot -- that still needs either a second
/// dedup pass over `incoming_gids` alone or the same sort this header does.
/// Sorting the concatenation once resolves both duplicate sources
/// (existing-vs-incoming AND incoming-vs-incoming) in a single pass, which
/// is why it is what this header implements; a binary-search variant is a
/// possible follow-on, not this one.
///
/// COMPLEXITY: `O((E + N) log(E + N))` for `sort_by_key` (the dominant
/// term; the official Kokkos sort backend, radix or bitonic depending on
/// `Space`), `O(E + N)` for the scan and the two scatter passes -- the same
/// complexity CLASS a host `std::sort` + "unique" pass over the same
/// concatenation would have, run on `Space` instead of serially.
///
/// SLOT NUMBERING IS ASCENDING BY GID VALUE, NOT BY APPEND ORDER. The Host
/// builders (`NativeMeshBuilder`, via `unpack_migrated_mesh`/
/// `merge_halo_cells`) keep every EXISTING vertex/cell at its own original
/// local index and append new ones in the order they are first encountered
/// in the incoming stream -- a convenience that lets existing cells'
/// connectivity be reused with no remapping at all. This header's
/// `merged_gids` is sorted by VALUE instead (the natural output of the
/// sort-and-compact pass above), so `slot_of_existing[i] != i` in general.
/// A caller that needs the Host builder's exact numbering adds one more
/// pass this header does not: keep `slot_of_existing[i] = i` unconditionally
/// (trivial), and renumber each canonical new entry by `E +` its own rank
/// among canonical entries in ASCENDING ORIGINAL INCOMING INDEX (a second
/// `parallel_scan` over `incoming_is_new`, not over the gid-sorted order).
/// Not implemented in this routine: it provides the dedup DECISION itself
/// (which gids are new, which incoming copy is canonical), the genuinely
/// algorithmic piece; the renumbering pass is a mechanical follow-on
/// sketched here, not a second algorithm.
///
/// WHAT THIS DOES NOT DO: the
/// wire format's records are VARIABLE length with no byte-length prefix, so
/// locating each record's start offset within a received buffer (the step
/// before this header's own gid array even exists) is only discoverable by
/// walking the buffer -- parallelizing THAT, or deciding to add a
/// length-prefix to the wire format so it CAN be parallelized, is a
/// separate, larger change `unpack_migrated_mesh`/`merge_halo_cells`
/// themselves would need, not attempted here. This header is the reusable
/// primitive their eventual device-resident rewrite would call once that
/// parsing step produces flat per-record gid arrays on `Space`.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "fuse/backend/native/mesh_pack_device.hpp"  // read_pod_at, append_pod_at, CellType

namespace fuse::native::device {

/// `dedup_by_gid`'s result. `merged_gids` is ascending by value, exactly
/// the distinct gids across `existing_gids` union `incoming_gids`, of size
/// `n_existing_unique + n_new` where `n_existing_unique` is however many of
/// `merged_gids`' entries also appear in `existing_gids` (every one of
/// them, by construction, if `existing_gids` itself carries no duplicate --
/// this header does not assume that and does not check it either, since
/// both real callers, `NativeMeshView`'s own existing vertex/cell list,
/// already guarantee it). `slot_of_existing`/`slot_of_incoming` are each
/// entry's position in `merged_gids`; a vertex merge uses `slot_of_incoming`
/// directly as the new mesh-local vertex index space is built, since every
/// duplicate copy of the same incoming gid already resolves to the SAME
/// slot. `incoming_is_new` additionally flags, for a cell merge (which
/// appends at most ONE physical record per new gid, not a value per slot),
/// exactly the canonical incoming record to materialize: the SMALLEST
/// original `incoming_gids` index sharing that slot, among slots no
/// existing entry claimed -- matching the Host path's own "first occurrence
/// in buffer order wins" semantics (`merge_halo_cells`/`unpack_migrated_mesh`
/// walk `received` in order and keep the first copy they see), not an
/// arbitrary one, so Device mode is free to pick either canonical answer
/// without disagreeing with what Host would have kept for the SAME input
/// order.
template <class Space>
struct DedupByGidResult {
  Kokkos::View<std::int64_t*, Space> merged_gids;
  Kokkos::View<int*, Space> slot_of_existing;
  Kokkos::View<int*, Space> slot_of_incoming;
  Kokkos::View<std::uint8_t*, Space> incoming_is_new;
  int n_existing = 0;
  int n_new = 0;
};

template <class Space>
[[nodiscard]] DedupByGidResult<Space> dedup_by_gid(
    const Kokkos::View<const std::int64_t*, Space>& existing_gids,
    const Kokkos::View<const std::int64_t*, Space>& incoming_gids) {
  using ExecSpace = typename Space::execution_space;
  const int E = static_cast<int>(existing_gids.extent(0));
  const int N = static_cast<int>(incoming_gids.extent(0));
  const int total = E + N;

  // `tags`: non-negative `i` for an existing entry at original index `i`;
  // negative `-(i + 1)` for an incoming entry at original index `i` -- one
  // signed 64-bit array distinguishes both the SIDE (sign) and the
  // ORIGINAL POSITION (magnitude) of every concatenated entry, so one
  // `sort_by_key` carries everything this header's later passes need.
  Kokkos::View<std::int64_t*, Space> keys(
      Kokkos::view_alloc("fuse::native::device::dedup_keys", Kokkos::WithoutInitializing), total);
  Kokkos::View<std::int64_t*, Space> tags(
      Kokkos::view_alloc("fuse::native::device::dedup_tags", Kokkos::WithoutInitializing), total);
  Kokkos::parallel_for(
      "fuse::native::device::dedup_fill", Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const int i) {
        if (i < E) {
          keys(i) = existing_gids(i);
          tags(i) = static_cast<std::int64_t>(i);
        } else {
          keys(i) = incoming_gids(i - E);
          tags(i) = -(static_cast<std::int64_t>(i - E) + 1);
        }
      });

  Kokkos::Experimental::sort_by_key(ExecSpace{}, keys, tags);

  Kokkos::View<int*, Space> slot_at(
      Kokkos::view_alloc("fuse::native::device::dedup_slot_at", Kokkos::WithoutInitializing),
      total);
  // INCLUSIVE in `running` (incremented BEFORE the write, not after): a run
  // of several equal keys must all read the SAME slot, which an exclusive
  // scan that increments after writing gets wrong for every member past
  // the first (`native_mesh_merge_device` covers it -- a duplicate run's
  // second-and-later members would land one slot ahead of its first,
  // merging the WRONG pair of distinct values into one slot). Incrementing first
  // makes `running` the 1-based count of DISTINCT keys seen through `p`
  // inclusive, so `running - 1` is the right 0-based slot for every member
  // of the run `p` belongs to, including its first.
  int n_unique = 0;
  Kokkos::parallel_scan(
      "fuse::native::device::dedup_scan", Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const int p, int& running, const bool final_pass) {
        const bool starts_run = (p == 0) || (keys(p) != keys(p - 1));
        if (starts_run) running += 1;
        if (final_pass) slot_at(p) = running - 1;
      },
      n_unique);

  Kokkos::View<std::int64_t*, Space> merged_gids(
      Kokkos::view_alloc("fuse::native::device::dedup_merged", Kokkos::WithoutInitializing),
      n_unique);
  Kokkos::View<std::uint8_t*, Space> slot_has_existing("fuse::native::device::dedup_has_existing",
                                                       n_unique);  // zero-initialized

  Kokkos::parallel_for(
      "fuse::native::device::dedup_slot_meta", Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const int p) {
        const int s = slot_at(p);
        merged_gids(s) = keys(p);  // every member of the run shares this value.
        if (tags(p) >= 0) slot_has_existing(s) = 1;  // benign race: every writer stores 1.
      });

  Kokkos::View<int*, Space> slot_of_existing(
      Kokkos::view_alloc("fuse::native::device::dedup_slot_existing", Kokkos::WithoutInitializing),
      E);
  Kokkos::View<int*, Space> slot_of_incoming(
      Kokkos::view_alloc("fuse::native::device::dedup_slot_incoming", Kokkos::WithoutInitializing),
      N);
  Kokkos::parallel_for(
      "fuse::native::device::dedup_scatter", Kokkos::RangePolicy<ExecSpace>(0, total),
      KOKKOS_LAMBDA(const int p) {
        const int s = slot_at(p);
        if (tags(p) >= 0) {
          slot_of_existing(static_cast<int>(tags(p))) = s;
        } else {
          slot_of_incoming(static_cast<int>(-tags(p) - 1)) = s;
        }
      });

  // Canonical incoming record per genuinely-new slot: the SMALLEST original
  // incoming index landing there, via atomic_min -- well-defined regardless
  // of scheduling order, and matching Host's "first in buffer order wins".
  Kokkos::View<int*, Space> canonical_incoming_of_slot(
      "fuse::native::device::dedup_canon", n_unique);
  Kokkos::deep_copy(canonical_incoming_of_slot, N);  // sentinel: N is out of [0, N) range.
  Kokkos::parallel_for(
      "fuse::native::device::dedup_canon_pass", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        const int s = slot_of_incoming(i);
        if (slot_has_existing(s) == 0) Kokkos::atomic_min(&canonical_incoming_of_slot(s), i);
      });

  Kokkos::View<std::uint8_t*, Space> incoming_is_new("fuse::native::device::dedup_is_new",
                                                      N);  // zero-initialized
  int n_new = 0;
  Kokkos::parallel_reduce(
      "fuse::native::device::dedup_flag_and_count", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, int& acc) {
        const int s = slot_of_incoming(i);
        const bool is_canon = slot_has_existing(s) == 0 && canonical_incoming_of_slot(s) == i;
        incoming_is_new(i) = is_canon ? std::uint8_t{1} : std::uint8_t{0};
        acc += is_canon ? 1 : 0;
      },
      n_new);

  DedupByGidResult<Space> out;
  out.merged_gids = merged_gids;
  out.slot_of_existing = slot_of_existing;
  out.slot_of_incoming = slot_of_incoming;
  out.incoming_is_new = incoming_is_new;
  out.n_existing = E;
  out.n_new = n_new;
  return out;
}

// =========================================================================
// The renumbering pass and the record-boundary parser the header comment
// above describes.
// =========================================================================

/// The append-order renumbering pass the "SLOT NUMBERING" paragraph above
/// describes but `dedup_by_gid` alone does not perform: `merged_gids` is
/// sorted by GID VALUE, but the Host builders keep every EXISTING entry at
/// its own original index and append new ones in FIRST-ENCOUNTER order --
/// this renumbers to match that convention exactly, so Device mode's final
/// vertex/cell numbering agrees with Host bit-for-bit, not merely up to a
/// permutation.
template <class Space>
struct RenumberedMerge {
  int n_existing = 0;
  int n_new = 0;
  /// `final_index_of_incoming(i)` is incoming entry `i`'s index in the
  /// FINAL (merged) numbering: `< n_existing` if `i`'s gid was already
  /// held (equal to that existing entry's own original index), else
  /// `n_existing + (that gid's rank among canonical new entries in
  /// ascending ORIGINAL INCOMING index, 0-based)`. Every duplicate copy
  /// of the same incoming gid maps to the SAME final index.
  Kokkos::View<int*, Space> final_index_of_incoming;
  /// Re-exposed from `dedup_by_gid` so a caller filtering "which records
  /// to actually keep" (a cell merge's own dedup) does not re-run it.
  Kokkos::View<std::uint8_t*, Space> incoming_is_new;
};

template <class Space>
[[nodiscard]] RenumberedMerge<Space> renumber_append_order(
    const Kokkos::View<const std::int64_t*, Space>& existing_gids,
    const Kokkos::View<const std::int64_t*, Space>& incoming_gids) {
  using ExecSpace = typename Space::execution_space;
  const DedupByGidResult<Space> dedup = dedup_by_gid<Space>(existing_gids, incoming_gids);
  const int E = dedup.n_existing;
  const int N = static_cast<int>(incoming_gids.extent(0));

  // new_rank(i): 1-based count of canonical (is_new) entries at position
  // <= i, in ORIGINAL INCOMING ORDER -- "this gid's rank among canonical
  // entries in ascending original incoming index."
  Kokkos::View<int*, Space> new_rank(
      Kokkos::view_alloc("fuse::native::device::renumber_new_rank", Kokkos::WithoutInitializing),
      N);
  int n_new = 0;
  auto incoming_is_new = dedup.incoming_is_new;
  Kokkos::parallel_scan(
      "fuse::native::device::renumber_new_rank_scan", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, int& running, const bool final_pass) {
        const bool is_new = incoming_is_new(i) != 0;
        if (is_new) running += 1;
        if (final_pass) new_rank(i) = running;
      },
      n_new);

  // final_index_of_slot(s): the FINAL merged index every member of slot s
  // maps to -- the existing entry's own original index for a slot an
  // existing gid claimed; `E + new_rank(i) - 1` for a genuinely new slot,
  // read off its own CANONICAL incoming member (never a non-canonical
  // duplicate: only `i` with `incoming_is_new(i)` writes).
  const int n_unique = static_cast<int>(dedup.merged_gids.extent(0));
  Kokkos::View<int*, Space> final_index_of_slot(
      Kokkos::view_alloc("fuse::native::device::renumber_final_of_slot",
                         Kokkos::WithoutInitializing),
      n_unique);
  auto slot_of_existing = dedup.slot_of_existing;
  auto slot_of_incoming = dedup.slot_of_incoming;
  Kokkos::parallel_for(
      "fuse::native::device::renumber_existing_scatter", Kokkos::RangePolicy<ExecSpace>(0, E),
      KOKKOS_LAMBDA(const int i) { final_index_of_slot(slot_of_existing(i)) = i; });
  Kokkos::parallel_for(
      "fuse::native::device::renumber_new_scatter", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        if (incoming_is_new(i) != 0) {
          final_index_of_slot(slot_of_incoming(i)) = E + new_rank(i) - 1;
        }
      });

  Kokkos::View<int*, Space> final_index_of_incoming(
      Kokkos::view_alloc("fuse::native::device::renumber_final_of_incoming",
                         Kokkos::WithoutInitializing),
      N);
  Kokkos::parallel_for(
      "fuse::native::device::renumber_gather", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i) {
        final_index_of_incoming(i) = final_index_of_slot(slot_of_incoming(i));
      });

  RenumberedMerge<Space> out;
  out.n_existing = E;
  out.n_new = n_new;
  out.final_index_of_incoming = final_index_of_incoming;
  out.incoming_is_new = incoming_is_new;
  return out;
}

// ---- device-parallel record-boundary parsing (wire format v2 header,
// mesh_pack_device.hpp's own "WIRE FORMAT v2" comment) -------------------

KOKKOS_INLINE_FUNCTION std::int64_t header_record_count(const std::byte* buf) {
  std::size_t pos = 0;
  return read_pod_at<std::int64_t>(buf, pos);
}

/// Record `r`'s ABSOLUTE byte start within `buf` (header included), read
/// directly from the header's own offsets array -- no walk of anything
/// before record `r`.
KOKKOS_INLINE_FUNCTION std::size_t header_record_start(const std::byte* buf, int r) {
  std::size_t pos = sizeof(std::int64_t) * static_cast<std::size_t>(1 + r);
  return static_cast<std::size_t>(read_pod_at<std::int64_t>(buf, pos));
}

/// One buffer's own record count, read via a tiny (8-byte) device-to-host
/// copy -- bookkeeping needed to size the flat output arrays below and to
/// bound the per-buffer kernel launches that follow, never the mesh
/// itself (the same class of host-side scalar `pack_to`'s own `num_ranks`
/// loop, or `exchange_migration_buffers`'s `send_sizes`, already reads).
template <class Space>
[[nodiscard]] int read_record_count_host(const Kokkos::View<std::byte*, Space>& buf) {
  if (buf.extent(0) == 0) return 0;
  using ExecSpace = typename Space::execution_space;
  Kokkos::View<std::int64_t, Space> count_d("fuse::native::device::parse_count_d");
  Kokkos::parallel_for(
      "fuse::native::device::parse_count_read", Kokkos::RangePolicy<ExecSpace>(0, 1),
      KOKKOS_LAMBDA(const int) { count_d() = header_record_count(buf.data()); });
  auto count_h = Kokkos::create_mirror_view(count_d);
  Kokkos::deep_copy(count_h, count_d);
  return static_cast<int>(count_h());
}

/// Every received buffer's records, flattened into one set of device
/// arrays -- the parsing step `pack_to`/`pack_fanout`'s own packing
/// mirrors in reverse, entirely on `Space`. `kHaloRecord` selects the
/// same wire variant `pack_to`/`pack_fanout`/`write_record` do.
///
/// Cost: `O(num_buffers)` small kernel launches (one per buffer, each
/// over that buffer's own record range only -- avoids needing an array
/// of raw pointers into several independently-allocated buffers for one
/// unified kernel, at the scale of a handful of
/// buffers) for the fixed-size fields, plus two scan+extract passes for
/// the variable-length vertex closure and (only when a record is a
/// `Polyhedron`) two more for its face section -- each inner loop over
/// ONE record's own small face/vertex count, single-threaded, the same
/// pattern `write_record`'s own face loop already uses on the pack side.
template <bool kHaloRecord, class Space>
struct ParsedBuffers {
  int n = 0;  ///< total records across every buffer, in buffer-then-record order
  Kokkos::View<std::int64_t*, Space> cell_gid;
  Kokkos::View<std::uint8_t*, Space> cell_type;
  Kokkos::View<int*, Space> n_verts;
  Kokkos::View<int*, Space> vertex_offset;  ///< size n + 1
  Kokkos::View<int*, Space> n_faces;        ///< size n, 0 for a non-Polyhedron record
  Kokkos::View<int*, Space> face_offset;    ///< size n + 1
  Kokkos::View<std::int32_t*, Space> owner_rank;       ///< size n, meaningful iff kHaloRecord
  Kokkos::View<std::int64_t*, Space> owned_block_idx;  ///< size n, meaningful iff kHaloRecord
  // Flat vertex-instance arrays, size vertex_offset(n).
  Kokkos::View<std::int64_t*, Space> vgid;
  Kokkos::View<double*, Space> vx, vy, vz;
  Kokkos::View<int*, Space> vertex_record;  ///< which record R this vertex instance came from
  // Flat per-face arrays, size face_offset(n).
  int total_faces = 0;
  Kokkos::View<int*, Space> face_vertex_count;   ///< size total_faces
  Kokkos::View<int*, Space> face_vertex_offset;  ///< size total_faces + 1
  // Flat closure-position array (RECORD-LOCAL vertex positions, 0..n_verts(record)-1),
  // size face_vertex_offset(total_faces).
  Kokkos::View<int*, Space> face_closure_pos;
  Kokkos::View<int*, Space> closure_vertex_base;  ///< that record's own vertex_offset(R)
};

template <bool kHaloRecord, class Space>
[[nodiscard]] ParsedBuffers<kHaloRecord, Space> parse_buffers(
    const std::vector<Kokkos::View<std::byte*, Space>>& received) {
  using ExecSpace = typename Space::execution_space;
  constexpr std::size_t prefix_bytes =
      (kHaloRecord ? (sizeof(std::int32_t) + sizeof(std::int64_t)) : std::size_t{0}) +
      sizeof(std::int64_t) + sizeof(std::uint8_t) + sizeof(std::int32_t);
  constexpr std::size_t vertex_stride = sizeof(std::int64_t) + 3 * sizeof(double);
  const int num_bufs = static_cast<int>(received.size());

  std::vector<int> n_per_buf(static_cast<std::size_t>(num_bufs), 0);
  std::vector<int> buf_record_start(static_cast<std::size_t>(num_bufs), 0);
  int n = 0;
  for (int b = 0; b < num_bufs; ++b) {
    buf_record_start[static_cast<std::size_t>(b)] = n;
    n_per_buf[static_cast<std::size_t>(b)] = read_record_count_host<Space>(received[static_cast<std::size_t>(b)]);
    n += n_per_buf[static_cast<std::size_t>(b)];
  }

  ParsedBuffers<kHaloRecord, Space> out;
  out.n = n;
  out.cell_gid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_cell_gid", Kokkos::WithoutInitializing), n);
  out.cell_type = Kokkos::View<std::uint8_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_cell_type", Kokkos::WithoutInitializing), n);
  out.n_verts = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_n_verts", Kokkos::WithoutInitializing), n);
  out.n_faces = Kokkos::View<int*, Space>("fuse::native::device::parse_n_faces", n);  // zero-init
  if constexpr (kHaloRecord) {
    out.owner_rank = Kokkos::View<std::int32_t*, Space>(
        Kokkos::view_alloc("fuse::native::device::parse_owner_rank", Kokkos::WithoutInitializing),
        n);
    out.owned_block_idx = Kokkos::View<std::int64_t*, Space>(
        Kokkos::view_alloc("fuse::native::device::parse_owned_idx", Kokkos::WithoutInitializing),
        n);
  }

  // Pass 1: fixed-prefix fields, plus (Polyhedron only) n_faces.
  for (int b = 0; b < num_bufs; ++b) {
    const int nb = n_per_buf[static_cast<std::size_t>(b)];
    if (nb == 0) continue;
    const Kokkos::View<std::byte*, Space> buf = received[static_cast<std::size_t>(b)];
    const int base = buf_record_start[static_cast<std::size_t>(b)];
    auto cell_gid = out.cell_gid;
    auto cell_type = out.cell_type;
    auto n_verts = out.n_verts;
    auto n_faces = out.n_faces;
    auto owner_rank = out.owner_rank;
    auto owned_block_idx = out.owned_block_idx;
    // Same nvcc first-capture-under-if-constexpr workaround `pack_to`/
    // `pack_fanout` (mesh_pack_device.hpp) document: two launches, not one
    // branching lambda, since `owner_rank`/`owned_block_idx` would otherwise
    // be first-captured only inside the `if constexpr (kHaloRecord)` arm.
    if constexpr (kHaloRecord) {
      Kokkos::parallel_for(
          "fuse::native::device::parse_prefix_halo", Kokkos::RangePolicy<ExecSpace>(0, nb),
          KOKKOS_LAMBDA(const int r) {
            const int R = base + r;
            std::size_t pos = header_record_start(buf.data(), r);
            owner_rank(R) = read_pod_at<std::int32_t>(buf.data(), pos);
            owned_block_idx(R) = read_pod_at<std::int64_t>(buf.data(), pos);
            cell_gid(R) = read_pod_at<std::int64_t>(buf.data(), pos);
            const auto type_byte = read_pod_at<std::uint8_t>(buf.data(), pos);
            cell_type(R) = type_byte;
            const auto nv = read_pod_at<std::int32_t>(buf.data(), pos);
            n_verts(R) = nv;
            if (static_cast<CellType>(type_byte) == CellType::Polyhedron) {
              pos += static_cast<std::size_t>(nv) * vertex_stride;
              n_faces(R) = read_pod_at<std::int32_t>(buf.data(), pos);
            }
          });
    } else {
      Kokkos::parallel_for(
          "fuse::native::device::parse_prefix_migration", Kokkos::RangePolicy<ExecSpace>(0, nb),
          KOKKOS_LAMBDA(const int r) {
            const int R = base + r;
            std::size_t pos = header_record_start(buf.data(), r);
            cell_gid(R) = read_pod_at<std::int64_t>(buf.data(), pos);
            const auto type_byte = read_pod_at<std::uint8_t>(buf.data(), pos);
            cell_type(R) = type_byte;
            const auto nv = read_pod_at<std::int32_t>(buf.data(), pos);
            n_verts(R) = nv;
            if (static_cast<CellType>(type_byte) == CellType::Polyhedron) {
              pos += static_cast<std::size_t>(nv) * vertex_stride;
              n_faces(R) = read_pod_at<std::int32_t>(buf.data(), pos);
            }
          });
    }
  }

  // vertex_offset scan.
  out.vertex_offset = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vertex_offset", Kokkos::WithoutInitializing),
      n + 1);
  int V = 0;
  {
    auto n_verts = out.n_verts;
    auto vertex_offset = out.vertex_offset;
    Kokkos::parallel_scan(
        "fuse::native::device::parse_vertex_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R, int& running, const bool final_pass) {
          if (final_pass) vertex_offset(R) = running;
          running += n_verts(R);
        },
        V);
    Kokkos::parallel_for(
        "fuse::native::device::parse_vertex_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { vertex_offset(n) = V; });
  }

  out.vgid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vgid", Kokkos::WithoutInitializing), V);
  out.vx = Kokkos::View<double*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vx", Kokkos::WithoutInitializing), V);
  out.vy = Kokkos::View<double*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vy", Kokkos::WithoutInitializing), V);
  out.vz = Kokkos::View<double*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vz", Kokkos::WithoutInitializing), V);
  out.vertex_record = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_vertex_record", Kokkos::WithoutInitializing),
      V);

  // Pass 2: vertex closure extraction.
  for (int b = 0; b < num_bufs; ++b) {
    const int nb = n_per_buf[static_cast<std::size_t>(b)];
    if (nb == 0) continue;
    const Kokkos::View<std::byte*, Space> buf = received[static_cast<std::size_t>(b)];
    const int base = buf_record_start[static_cast<std::size_t>(b)];
    auto n_verts = out.n_verts;
    auto vertex_offset = out.vertex_offset;
    auto vgid = out.vgid;
    auto vx = out.vx;
    auto vy = out.vy;
    auto vz = out.vz;
    auto vertex_record = out.vertex_record;
    Kokkos::parallel_for(
        "fuse::native::device::parse_vertices", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const int r) {
          const int R = base + r;
          std::size_t pos = header_record_start(buf.data(), r) + prefix_bytes;
          const int nv = n_verts(R);
          const int voff = vertex_offset(R);
          for (int k = 0; k < nv; ++k) {
            vgid(voff + k) = read_pod_at<std::int64_t>(buf.data(), pos);
            vx(voff + k) = read_pod_at<double>(buf.data(), pos);
            vy(voff + k) = read_pod_at<double>(buf.data(), pos);
            vz(voff + k) = read_pod_at<double>(buf.data(), pos);
            vertex_record(voff + k) = R;
          }
        });
  }

  // face_offset scan.
  out.face_offset = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_face_offset", Kokkos::WithoutInitializing),
      n + 1);
  int FC = 0;
  {
    auto n_faces = out.n_faces;
    auto face_offset = out.face_offset;
    Kokkos::parallel_scan(
        "fuse::native::device::parse_face_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R, int& running, const bool final_pass) {
          if (final_pass) face_offset(R) = running;
          running += n_faces(R);
        },
        FC);
    Kokkos::parallel_for(
        "fuse::native::device::parse_face_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { face_offset(n) = FC; });
  }
  out.total_faces = FC;

  out.face_vertex_count = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_fvc", Kokkos::WithoutInitializing), FC);

  // Pass 3: per-record sequential walk of its OWN (small) face section,
  // reading only each face's length and skipping its closure-position
  // values -- one thread per record, never across records.
  for (int b = 0; b < num_bufs; ++b) {
    const int nb = n_per_buf[static_cast<std::size_t>(b)];
    if (nb == 0) continue;
    const Kokkos::View<std::byte*, Space> buf = received[static_cast<std::size_t>(b)];
    const int base = buf_record_start[static_cast<std::size_t>(b)];
    auto n_verts = out.n_verts;
    auto n_faces = out.n_faces;
    auto face_offset = out.face_offset;
    auto face_vertex_count = out.face_vertex_count;
    Kokkos::parallel_for(
        "fuse::native::device::parse_face_counts", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const int r) {
          const int R = base + r;
          const int nf = n_faces(R);
          if (nf == 0) return;
          std::size_t pos = header_record_start(buf.data(), r) + prefix_bytes +
                            static_cast<std::size_t>(n_verts(R)) * vertex_stride +
                            sizeof(std::int32_t);  // skip the n_faces field itself
          const int foff = face_offset(R);
          for (int f = 0; f < nf; ++f) {
            const auto flen = read_pod_at<std::int32_t>(buf.data(), pos);
            face_vertex_count(foff + f) = flen;
            pos += static_cast<std::size_t>(flen) * sizeof(std::int32_t);
          }
        });
  }

  out.face_vertex_offset = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_fvo", Kokkos::WithoutInitializing), FC + 1);
  int CC = 0;
  {
    auto fvc = out.face_vertex_count;
    auto fvo = out.face_vertex_offset;
    Kokkos::parallel_scan(
        "fuse::native::device::parse_fvo_scan", Kokkos::RangePolicy<ExecSpace>(0, FC),
        KOKKOS_LAMBDA(const int F, int& running, const bool final_pass) {
          if (final_pass) fvo(F) = running;
          running += fvc(F);
        },
        CC);
    Kokkos::parallel_for(
        "fuse::native::device::parse_fvo_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { fvo(FC) = CC; });
  }

  out.face_closure_pos = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_closure_pos", Kokkos::WithoutInitializing),
      CC);
  out.closure_vertex_base = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::parse_closure_base", Kokkos::WithoutInitializing),
      CC);

  // Pass 4: same per-record walk as Pass 3, this time extracting the
  // closure-position VALUES themselves (and each one's owning record's
  // own vertex_offset, needed later to translate a record-local closure
  // position into a global incoming-vertex-instance index).
  for (int b = 0; b < num_bufs; ++b) {
    const int nb = n_per_buf[static_cast<std::size_t>(b)];
    if (nb == 0) continue;
    const Kokkos::View<std::byte*, Space> buf = received[static_cast<std::size_t>(b)];
    const int base = buf_record_start[static_cast<std::size_t>(b)];
    auto n_verts = out.n_verts;
    auto n_faces = out.n_faces;
    auto face_offset = out.face_offset;
    auto face_vertex_offset = out.face_vertex_offset;
    auto vertex_offset = out.vertex_offset;
    auto face_closure_pos = out.face_closure_pos;
    auto closure_vertex_base = out.closure_vertex_base;
    Kokkos::parallel_for(
        "fuse::native::device::parse_face_closures", Kokkos::RangePolicy<ExecSpace>(0, nb),
        KOKKOS_LAMBDA(const int r) {
          const int R = base + r;
          const int nf = n_faces(R);
          if (nf == 0) return;
          std::size_t pos = header_record_start(buf.data(), r) + prefix_bytes +
                            static_cast<std::size_t>(n_verts(R)) * vertex_stride +
                            sizeof(std::int32_t);
          const int foff = face_offset(R);
          const int voff = vertex_offset(R);
          for (int f = 0; f < nf; ++f) {
            const auto flen = read_pod_at<std::int32_t>(buf.data(), pos);
            const int cbase = face_vertex_offset(foff + f);
            for (int k = 0; k < flen; ++k) {
              face_closure_pos(cbase + k) = read_pod_at<std::int32_t>(buf.data(), pos);
              closure_vertex_base(cbase + k) = voff;
            }
          }
        });
  }

  return out;
}

// ---- the full device-resident unpack/merge, replacing the host round
// trip `migrate`/`build_native_remap_halo` used to require -----------------

/// Device-resident `unpack_migrated_mesh`: every received record becomes
/// a cell (migration has no cell dedup -- a cell migrates to exactly ONE
/// destination by construction, so no incoming gid repeats), vertices
/// deduplicated by global id with the SAME append-order numbering Host's
/// `NativeMeshBuilder` produces. Every vertex is tentatively
/// `owner = my_rank, is_owned = true` (unresolved -- `resolve_vertex_
/// ownership` settles it next, matching Host's own convention exactly).
///
/// POLYHEDRON LIMITATION (declared, not silently absorbed): a
/// `Polyhedron` cell's faces become NEW, PRIVATE face entries here, one
/// set per cell, remapped through the new vertex numbering by a gather --
/// never deduplicated against another cell's matching face the way
/// Host's `NativeMeshBuilder::find_or_add_face` does (sorted-key vertex
/// SET match plus cyclic-order comparison, with its own
/// `E-MESH-POLYGON-FACE-MISMATCH` refusal). Cell count, vertex count,
/// global ids, measure and centroid are unaffected (the geometry is the
/// same faces, just not memory-shared); `face_count()` is larger than
/// Host's for an input mesh containing Polyhedron cells. No row in the
/// existing migration/halo/repartition suite exercises a Polyhedron cell
/// (confirmed by reading every `generate_native_mesh` spec those tests
/// pass); see docs/dev/native-halo.md for the narrower test of this
/// limitation.
template <class Space>
[[nodiscard]] MeshArrays<Space> unpack_migrated_mesh_device(
    const std::vector<Kokkos::View<std::byte*, Space>>& received, int dim, int my_rank) {
  using ExecSpace = typename Space::execution_space;
  auto parsed = parse_buffers<false, Space>(received);
  const int n = parsed.n;
  const int V = static_cast<int>(parsed.vgid.extent(0));

  const Kokkos::View<const std::int64_t*, Space> existing_empty;  // extent 0
  const Kokkos::View<const std::int64_t*, Space> incoming_vgid = parsed.vgid;
  auto renum = renumber_append_order<Space>(existing_empty, incoming_vgid);
  const int n_new_verts = renum.n_new;

  MeshArrays<Space> out;
  out.dim = dim;

  out.vertex_coords = Kokkos::View<double* [3], Kokkos::LayoutRight, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_vc", Kokkos::WithoutInitializing),
      n_new_verts);
  out.vertex_gid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_vgid", Kokkos::WithoutInitializing),
      n_new_verts);
  out.vertex_owner = Kokkos::View<int*, Space>("fuse::native::device::unpack_vowner", n_new_verts);
  out.vertex_is_owned =
      Kokkos::View<bool*, Space>("fuse::native::device::unpack_vowned", n_new_verts);
  {
    auto final_index = renum.final_index_of_incoming;
    auto is_new = renum.incoming_is_new;
    auto vgid_in = parsed.vgid;
    auto vx = parsed.vx;
    auto vy = parsed.vy;
    auto vz = parsed.vz;
    auto vertex_coords = out.vertex_coords;
    auto vertex_gid = out.vertex_gid;
    auto vertex_owner = out.vertex_owner;
    auto vertex_is_owned = out.vertex_is_owned;
    Kokkos::parallel_for(
        "fuse::native::device::unpack_scatter_vertices", Kokkos::RangePolicy<ExecSpace>(0, V),
        KOKKOS_LAMBDA(const int i) {
          if (is_new(i) == 0) return;
          const int idx = final_index(i);
          vertex_gid(idx) = vgid_in(i);
          vertex_coords(idx, 0) = vx(i);
          vertex_coords(idx, 1) = vy(i);
          vertex_coords(idx, 2) = vz(i);
          vertex_owner(idx) = my_rank;
          vertex_is_owned(idx) = true;
        });
  }

  out.cell_gid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_cgid", Kokkos::WithoutInitializing), n);
  out.cell_type = Kokkos::View<std::uint8_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_ctype", Kokkos::WithoutInitializing), n);
  out.cell_owner = Kokkos::View<int*, Space>("fuse::native::device::unpack_cowner", n);
  out.cell_is_owned = Kokkos::View<bool*, Space>("fuse::native::device::unpack_cowned", n);
  {
    auto cell_gid = out.cell_gid;
    auto cell_type = out.cell_type;
    auto cell_owner = out.cell_owner;
    auto cell_is_owned = out.cell_is_owned;
    auto pcell_gid = parsed.cell_gid;
    auto pcell_type = parsed.cell_type;
    Kokkos::parallel_for(
        "fuse::native::device::unpack_cells", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R) {
          cell_gid(R) = pcell_gid(R);
          cell_type(R) = pcell_type(R);
          cell_owner(R) = my_rank;
          cell_is_owned(R) = true;
        });
  }

  out.cell_vertex_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_cvo", Kokkos::WithoutInitializing), n + 1);
  Kokkos::deep_copy(out.cell_vertex_offsets, parsed.vertex_offset);
  out.cell_vertex_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_cvi", Kokkos::WithoutInitializing), V);
  {
    auto final_index = renum.final_index_of_incoming;
    auto cell_vertex_indices = out.cell_vertex_indices;
    Kokkos::parallel_for(
        "fuse::native::device::unpack_remap_connectivity", Kokkos::RangePolicy<ExecSpace>(0, V),
        KOKKOS_LAMBDA(const int i) { cell_vertex_indices(i) = final_index(i); });
  }

  const int FC = parsed.total_faces;
  const int CC = static_cast<int>(parsed.face_closure_pos.extent(0));
  out.cell_face_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_cfo", Kokkos::WithoutInitializing), n + 1);
  Kokkos::deep_copy(out.cell_face_offsets, parsed.face_offset);
  out.cell_face_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_cfi", Kokkos::WithoutInitializing), FC);
  out.cell_face_flip = Kokkos::View<bool*, Space>("fuse::native::device::unpack_cff", FC);
  {
    auto cell_face_indices = out.cell_face_indices;
    Kokkos::parallel_for(
        "fuse::native::device::unpack_face_identity", Kokkos::RangePolicy<ExecSpace>(0, FC),
        KOKKOS_LAMBDA(const int f) { cell_face_indices(f) = f; });
  }
  out.face_vertex_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_fvo", Kokkos::WithoutInitializing), FC + 1);
  Kokkos::deep_copy(out.face_vertex_offsets, parsed.face_vertex_offset);
  out.face_vertex_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::unpack_fvi", Kokkos::WithoutInitializing), CC);
  {
    auto closure_pos = parsed.face_closure_pos;
    auto closure_base = parsed.closure_vertex_base;
    auto final_index = renum.final_index_of_incoming;
    auto face_vertex_indices = out.face_vertex_indices;
    Kokkos::parallel_for(
        "fuse::native::device::unpack_remap_faces", Kokkos::RangePolicy<ExecSpace>(0, CC),
        KOKKOS_LAMBDA(const int e) {
          const int incoming_instance = closure_base(e) + closure_pos(e);
          face_vertex_indices(e) = final_index(incoming_instance);
        });
  }

  return out;
}

/// Device-resident `merge_halo_cells`: `source`'s existing cells and
/// vertices preserved verbatim (same index, same connectivity -- no
/// existing vertex is ever renumbered), plus every record in `received`
/// whose cell global id is not already held, deduplicated by
/// `dedup_by_gid` the SAME "first occurrence in buffer/record order
/// wins" way Host's own sequential walk does. Vertices are gathered ONLY
/// from KEPT (genuinely new) records -- Host's own "only materialize
/// vertices for a cell we are actually about to keep" rule -- then
/// deduplicated against `source`'s existing vertices with the same
/// append-order renumbering `unpack_migrated_mesh_device` uses. A new
/// vertex's owner is the SENDING rank (the record's own `owner_rank`
/// field), `is_owned = false`, matching Host's convention; `*owner_rank`/
/// `*owned_block_idx` are the new cells' own `GhostPattern` inputs, in
/// append (kept) order.
///
/// Same POLYHEDRON LIMITATION as `unpack_migrated_mesh_device` (its own
/// comment): a NEW cell's faces are private, gathered through the new
/// vertex numbering, never deduplicated against another cell's face.
/// `source`'s own EXISTING faces are copied verbatim (unaffected, since
/// existing vertex numbering never changes).
template <class Space>
[[nodiscard]] MeshArrays<Space> merge_halo_cells_device(
    const MeshArrays<Space>& source, const std::vector<Kokkos::View<std::byte*, Space>>& received,
    int /*my_rank*/, Kokkos::View<std::int32_t*, Space>* owner_rank_out,
    Kokkos::View<std::int64_t*, Space>* owned_block_idx_out) {
  using ExecSpace = typename Space::execution_space;
  auto parsed = parse_buffers<true, Space>(received);
  const int n = parsed.n;
  const int E_cells = static_cast<int>(source.cell_count());
  const int E_verts = static_cast<int>(source.vertex_count());
  const int E_conn = static_cast<int>(source.cell_vertex_indices.extent(0));
  const int E_faces = static_cast<int>(source.face_count());
  const int E_fv = static_cast<int>(source.face_vertex_indices.extent(0));

  // 1. CELL dedup: already-held records are dropped entirely (idempotency
  //    -- Host's own "do not duplicate" rule), never contributing a cell
  //    OR a vertex.
  const Kokkos::View<const std::int64_t*, Space> existing_cell_gid = source.cell_gid;
  const Kokkos::View<const std::int64_t*, Space> incoming_cell_gid = parsed.cell_gid;
  auto cell_dedup = dedup_by_gid<Space>(existing_cell_gid, incoming_cell_gid);
  const int n_new_cells = cell_dedup.n_new;

  Kokkos::View<int*, Space> kept_index(
      Kokkos::view_alloc("fuse::native::device::merge_kept_index", Kokkos::WithoutInitializing),
      n);
  Kokkos::View<int*, Space> kept_record_of(
      Kokkos::view_alloc("fuse::native::device::merge_kept_record_of", Kokkos::WithoutInitializing),
      n_new_cells);
  {
    auto is_new = cell_dedup.incoming_is_new;
    int total_kept = 0;
    Kokkos::parallel_scan(
        "fuse::native::device::merge_kept_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R, int& running, const bool final_pass) {
          const bool keep = is_new(R) != 0;
          if (final_pass) kept_index(R) = running;
          if (keep) running += 1;
        },
        total_kept);
    Kokkos::parallel_for(
        "fuse::native::device::merge_kept_scatter", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R) {
          if (is_new(R) != 0) kept_record_of(kept_index(R)) = R;
        });
  }

  // 2. Filtered vertex gather: only KEPT records' vertices feed the pool.
  Kokkos::View<int*, Space> filtered_vertex_offset(
      Kokkos::view_alloc("fuse::native::device::merge_filt_voff", Kokkos::WithoutInitializing),
      n + 1);
  int V_filtered = 0;
  {
    auto is_new = cell_dedup.incoming_is_new;
    auto n_verts = parsed.n_verts;
    Kokkos::parallel_scan(
        "fuse::native::device::merge_filt_voff_scan", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R, int& running, const bool final_pass) {
          if (final_pass) filtered_vertex_offset(R) = running;
          if (is_new(R) != 0) running += n_verts(R);
        },
        V_filtered);
    Kokkos::parallel_for(
        "fuse::native::device::merge_filt_voff_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { filtered_vertex_offset(n) = V_filtered; });
  }

  Kokkos::View<std::int64_t*, Space> filtered_vgid(
      Kokkos::view_alloc("fuse::native::device::merge_filt_vgid", Kokkos::WithoutInitializing),
      V_filtered);
  Kokkos::View<double*, Space> filtered_vx(
      Kokkos::view_alloc("fuse::native::device::merge_filt_vx", Kokkos::WithoutInitializing),
      V_filtered);
  Kokkos::View<double*, Space> filtered_vy(
      Kokkos::view_alloc("fuse::native::device::merge_filt_vy", Kokkos::WithoutInitializing),
      V_filtered);
  Kokkos::View<double*, Space> filtered_vz(
      Kokkos::view_alloc("fuse::native::device::merge_filt_vz", Kokkos::WithoutInitializing),
      V_filtered);
  Kokkos::View<int*, Space> filtered_to_parsed(
      Kokkos::view_alloc("fuse::native::device::merge_filt_to_parsed",
                         Kokkos::WithoutInitializing),
      V_filtered);
  {
    auto is_new = cell_dedup.incoming_is_new;
    auto n_verts = parsed.n_verts;
    auto vertex_offset = parsed.vertex_offset;
    auto filt_off = filtered_vertex_offset;
    auto vgid = parsed.vgid;
    auto vx = parsed.vx;
    auto vy = parsed.vy;
    auto vz = parsed.vz;
    auto f_vgid = filtered_vgid;
    auto f_vx = filtered_vx;
    auto f_vy = filtered_vy;
    auto f_vz = filtered_vz;
    auto f2p = filtered_to_parsed;
    Kokkos::parallel_for(
        "fuse::native::device::merge_filter_vertices", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int R) {
          if (is_new(R) == 0) return;
          const int nv = n_verts(R);
          const int src0 = vertex_offset(R);
          const int dst0 = filt_off(R);
          for (int k = 0; k < nv; ++k) {
            f_vgid(dst0 + k) = vgid(src0 + k);
            f_vx(dst0 + k) = vx(src0 + k);
            f_vy(dst0 + k) = vy(src0 + k);
            f_vz(dst0 + k) = vz(src0 + k);
            f2p(dst0 + k) = src0 + k;
          }
        });
  }

  // 3. VERTEX dedup + append-order renumbering against EXISTING vertices.
  const Kokkos::View<const std::int64_t*, Space> existing_vertex_gid = source.vertex_gid;
  const Kokkos::View<const std::int64_t*, Space> filtered_vgid_const = filtered_vgid;
  auto vrenum = renumber_append_order<Space>(existing_vertex_gid, filtered_vgid_const);
  const int n_new_verts = vrenum.n_new;
  const int V_total = E_verts + n_new_verts;

  MeshArrays<Space> out;
  out.dim = source.dim;

  out.vertex_coords = Kokkos::View<double* [3], Kokkos::LayoutRight, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_vc", Kokkos::WithoutInitializing),
      V_total);
  out.vertex_gid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_vgid", Kokkos::WithoutInitializing),
      V_total);
  out.vertex_owner = Kokkos::View<int*, Space>("fuse::native::device::merge_vowner", V_total);
  out.vertex_is_owned = Kokkos::View<bool*, Space>("fuse::native::device::merge_vowned", V_total);

  Kokkos::deep_copy(Kokkos::subview(out.vertex_coords, Kokkos::make_pair(0, E_verts), Kokkos::ALL),
                    source.vertex_coords);
  Kokkos::deep_copy(Kokkos::subview(out.vertex_gid, Kokkos::make_pair(0, E_verts)),
                    source.vertex_gid);
  Kokkos::deep_copy(Kokkos::subview(out.vertex_owner, Kokkos::make_pair(0, E_verts)),
                    source.vertex_owner);
  Kokkos::deep_copy(Kokkos::subview(out.vertex_is_owned, Kokkos::make_pair(0, E_verts)),
                    source.vertex_is_owned);
  {
    auto final_index = vrenum.final_index_of_incoming;
    auto is_new = vrenum.incoming_is_new;
    auto f_vgid = filtered_vgid;
    auto f_vx = filtered_vx;
    auto f_vy = filtered_vy;
    auto f_vz = filtered_vz;
    auto f2p = filtered_to_parsed;
    auto vertex_record = parsed.vertex_record;
    auto owner_rank = parsed.owner_rank;
    auto vertex_coords = out.vertex_coords;
    auto vertex_gid = out.vertex_gid;
    auto vertex_owner = out.vertex_owner;
    auto vertex_is_owned = out.vertex_is_owned;
    Kokkos::parallel_for(
        "fuse::native::device::merge_scatter_vertices", Kokkos::RangePolicy<ExecSpace>(0, V_filtered),
        KOKKOS_LAMBDA(const int i) {
          if (is_new(i) == 0) return;
          const int idx = final_index(i);
          vertex_gid(idx) = f_vgid(i);
          vertex_coords(idx, 0) = f_vx(i);
          vertex_coords(idx, 1) = f_vy(i);
          vertex_coords(idx, 2) = f_vz(i);
          const int parsed_j = f2p(i);
          const int R = vertex_record(parsed_j);
          vertex_owner(idx) = owner_rank(R);
          vertex_is_owned(idx) = false;
        });
  }

  // 4. CELLS: existing verbatim, new ones appended in kept (append) order.
  const int n_cells_out = E_cells + n_new_cells;
  out.cell_gid = Kokkos::View<std::int64_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_cgid", Kokkos::WithoutInitializing),
      n_cells_out);
  out.cell_type = Kokkos::View<std::uint8_t*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_ctype", Kokkos::WithoutInitializing),
      n_cells_out);
  out.cell_owner = Kokkos::View<int*, Space>("fuse::native::device::merge_cowner", n_cells_out);
  out.cell_is_owned =
      Kokkos::View<bool*, Space>("fuse::native::device::merge_cowned", n_cells_out);

  Kokkos::deep_copy(Kokkos::subview(out.cell_gid, Kokkos::make_pair(0, E_cells)), source.cell_gid);
  Kokkos::deep_copy(Kokkos::subview(out.cell_type, Kokkos::make_pair(0, E_cells)),
                    source.cell_type);
  Kokkos::deep_copy(Kokkos::subview(out.cell_owner, Kokkos::make_pair(0, E_cells)),
                    source.cell_owner);
  Kokkos::deep_copy(Kokkos::subview(out.cell_is_owned, Kokkos::make_pair(0, E_cells)),
                    source.cell_is_owned);

  Kokkos::View<std::int32_t*, Space> owner_rank_new(
      Kokkos::view_alloc("fuse::native::device::merge_owner_new", Kokkos::WithoutInitializing),
      n_new_cells);
  Kokkos::View<std::int64_t*, Space> owned_idx_new(
      Kokkos::view_alloc("fuse::native::device::merge_owned_idx_new", Kokkos::WithoutInitializing),
      n_new_cells);
  {
    auto cell_gid = out.cell_gid;
    auto cell_type = out.cell_type;
    auto cell_owner = out.cell_owner;
    auto cell_is_owned = out.cell_is_owned;
    auto pcell_gid = parsed.cell_gid;
    auto pcell_type = parsed.cell_type;
    auto powner = parsed.owner_rank;
    auto pidx = parsed.owned_block_idx;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_cells", Kokkos::RangePolicy<ExecSpace>(0, n_new_cells),
        KOKKOS_LAMBDA(const int j) {
          const int R = kept_record_of(j);
          cell_gid(E_cells + j) = pcell_gid(R);
          cell_type(E_cells + j) = pcell_type(R);
          cell_owner(E_cells + j) = powner(R);
          cell_is_owned(E_cells + j) = false;
          owner_rank_new(j) = powner(R);
          owned_idx_new(j) = pidx(R);
        });
  }
  *owner_rank_out = owner_rank_new;
  *owned_block_idx_out = owned_idx_new;

  // cell_vertex_offsets/indices: existing verbatim, new cells' own CSR
  // built from their filtered vertex range, remapped through `vrenum`.
  out.cell_vertex_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_cvo", Kokkos::WithoutInitializing),
      n_cells_out + 1);
  // An EMPTY local mesh (E_cells == 0, the "rank hosts only the target" case `MeshArrays::
  // cell_count()`'s own `cell_vertex_offsets.extent(0) == 0 ? 0 : ...` branch documents)
  // reports its OWN `cell_vertex_offsets` at extent 0, NOT the canonical size-1 `{0}`
  // singleton a non-empty mesh's "ncells + 1" CSR invariant would suggest -- copying
  // `E_cells + 1 == 1` elements FROM a 0-extent source throws a Kokkos deep_copy shape
  // mismatch ("merge_cvo(1) (0)"), measured at exactly this line.
  // Skipped entirely when there is nothing existing to preserve: position 0 of `out`'s
  // own array (and every other position through `n_cells_out`) is still written
  // unconditionally below, either by the new-cells scan (when `n_new_cells > 0`) or by
  // the scan's own trailing single-element `parallel_for` (`cell_vertex_offsets(n_cells_
  // out) = E_conn + new_conn_total`, which also covers position 0 when `n_cells_out == 0`
  // too) -- no gap either way.
  if (E_cells > 0) {
    Kokkos::deep_copy(Kokkos::subview(out.cell_vertex_offsets, Kokkos::make_pair(0, E_cells + 1)),
                      source.cell_vertex_offsets);
  }
  Kokkos::View<int*, Space> new_cell_nverts(
      Kokkos::view_alloc("fuse::native::device::merge_new_nverts", Kokkos::WithoutInitializing),
      n_new_cells);
  {
    auto n_verts = parsed.n_verts;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_nverts_fill",
        Kokkos::RangePolicy<ExecSpace>(0, n_new_cells),
        KOKKOS_LAMBDA(const int j) { new_cell_nverts(j) = n_verts(kept_record_of(j)); });
  }
  int new_conn_total = 0;
  {
    auto cell_vertex_offsets = out.cell_vertex_offsets;
    Kokkos::parallel_scan(
        "fuse::native::device::merge_cvo_scan", Kokkos::RangePolicy<ExecSpace>(0, n_new_cells),
        KOKKOS_LAMBDA(const int j, int& running, const bool final_pass) {
          if (final_pass) cell_vertex_offsets(E_cells + j) = E_conn + running;
          running += new_cell_nverts(j);
        },
        new_conn_total);
    Kokkos::parallel_for(
        "fuse::native::device::merge_cvo_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { cell_vertex_offsets(n_cells_out) = E_conn + new_conn_total; });
  }

  out.cell_vertex_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_cvi", Kokkos::WithoutInitializing),
      E_conn + new_conn_total);
  Kokkos::deep_copy(Kokkos::subview(out.cell_vertex_indices, Kokkos::make_pair(0, E_conn)),
                    source.cell_vertex_indices);
  {
    auto final_index = vrenum.final_index_of_incoming;
    auto filt_off = filtered_vertex_offset;
    auto n_verts = parsed.n_verts;
    auto cell_vertex_offsets = out.cell_vertex_offsets;
    auto cell_vertex_indices = out.cell_vertex_indices;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_connectivity",
        Kokkos::RangePolicy<ExecSpace>(0, n_new_cells), KOKKOS_LAMBDA(const int j) {
          const int R = kept_record_of(j);
          const int nv = n_verts(R);
          const int dst0 = cell_vertex_offsets(E_cells + j);
          const int src0 = filt_off(R);
          for (int k = 0; k < nv; ++k) cell_vertex_indices(dst0 + k) = final_index(src0 + k);
        });
  }

  // Polyhedron faces: existing verbatim, new cells' own PRIVATE faces
  // (same limitation as unpack_migrated_mesh_device's own comment).
  Kokkos::View<int*, Space> new_cell_nfaces(
      Kokkos::view_alloc("fuse::native::device::merge_new_nfaces", Kokkos::WithoutInitializing),
      n_new_cells);
  {
    auto n_faces = parsed.n_faces;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_nfaces_fill",
        Kokkos::RangePolicy<ExecSpace>(0, n_new_cells),
        KOKKOS_LAMBDA(const int j) { new_cell_nfaces(j) = n_faces(kept_record_of(j)); });
  }
  out.cell_face_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_cfo", Kokkos::WithoutInitializing),
      n_cells_out + 1);
  // Same empty-local-mesh guard as `cell_vertex_offsets` above, same reason
  // (`cell_face_offsets` carries the identical "ncells + 1" CSR convention, `E_cells == 0`
  // means `source.cell_face_offsets` is 0-extent, not the 1-element `{0}` a copy of
  // `E_cells + 1` elements would need).
  if (E_cells > 0) {
    Kokkos::deep_copy(Kokkos::subview(out.cell_face_offsets, Kokkos::make_pair(0, E_cells + 1)),
                      source.cell_face_offsets);
  }
  int new_faces_total = 0;
  {
    auto cell_face_offsets = out.cell_face_offsets;
    Kokkos::parallel_scan(
        "fuse::native::device::merge_cfo_scan", Kokkos::RangePolicy<ExecSpace>(0, n_new_cells),
        KOKKOS_LAMBDA(const int j, int& running, const bool final_pass) {
          if (final_pass) cell_face_offsets(E_cells + j) = E_faces + running;
          running += new_cell_nfaces(j);
        },
        new_faces_total);
    Kokkos::parallel_for(
        "fuse::native::device::merge_cfo_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) { cell_face_offsets(n_cells_out) = E_faces + new_faces_total; });
  }

  const int FC_total = E_faces + new_faces_total;
  out.cell_face_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_cfi", Kokkos::WithoutInitializing),
      FC_total);
  Kokkos::deep_copy(Kokkos::subview(out.cell_face_indices, Kokkos::make_pair(0, E_faces)),
                    source.cell_face_indices);
  out.cell_face_flip = Kokkos::View<bool*, Space>("fuse::native::device::merge_cff", FC_total);
  Kokkos::deep_copy(Kokkos::subview(out.cell_face_flip, Kokkos::make_pair(0, E_faces)),
                    source.cell_face_flip);
  {
    auto cell_face_indices = out.cell_face_indices;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_face_identity",
        Kokkos::RangePolicy<ExecSpace>(0, new_faces_total),
        KOKKOS_LAMBDA(const int f) { cell_face_indices(E_faces + f) = E_faces + f; });
  }

  Kokkos::View<int*, Space> new_face_vertex_count(
      Kokkos::view_alloc("fuse::native::device::merge_new_fvc", Kokkos::WithoutInitializing),
      new_faces_total);
  {
    auto n_faces = parsed.n_faces;
    auto face_offset = parsed.face_offset;
    auto face_vertex_count = parsed.face_vertex_count;
    auto cell_face_offsets = out.cell_face_offsets;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_fvc_fill",
        Kokkos::RangePolicy<ExecSpace>(0, n_new_cells), KOKKOS_LAMBDA(const int j) {
          const int R = kept_record_of(j);
          const int nf = n_faces(R);
          const int parsed_face0 = face_offset(R);
          const int new_face0 = cell_face_offsets(E_cells + j) - E_faces;
          for (int f = 0; f < nf; ++f) {
            new_face_vertex_count(new_face0 + f) = face_vertex_count(parsed_face0 + f);
          }
        });
  }

  out.face_vertex_offsets = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_fvo", Kokkos::WithoutInitializing),
      FC_total + 1);
  // Same empty-local-mesh guard, same "ncells + 1"-style CSR convention, this time over
  // FACES rather than cells (`E_faces == 0` for any non-Polyhedron mesh regardless of
  // cell count, per `MeshArrays`'s own "meaningless ... for a non-Polyhedron cell"
  // comment -- so this guard is live on every quad/FV mesh the test fixtures use,
  // empty or not, not only the empty-local-mesh case `cell_vertex_offsets`/
  // `cell_face_offsets` above need it for).
  if (E_faces > 0) {
    Kokkos::deep_copy(Kokkos::subview(out.face_vertex_offsets, Kokkos::make_pair(0, E_faces + 1)),
                      source.face_vertex_offsets);
  }
  int new_closure_total = 0;
  {
    auto face_vertex_offsets = out.face_vertex_offsets;
    Kokkos::parallel_scan(
        "fuse::native::device::merge_fvo_scan", Kokkos::RangePolicy<ExecSpace>(0, new_faces_total),
        KOKKOS_LAMBDA(const int f, int& running, const bool final_pass) {
          if (final_pass) face_vertex_offsets(E_faces + f) = E_fv + running;
          running += new_face_vertex_count(f);
        },
        new_closure_total);
    Kokkos::parallel_for(
        "fuse::native::device::merge_fvo_end", Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
          face_vertex_offsets(E_faces + new_faces_total) = E_fv + new_closure_total;
        });
  }

  out.face_vertex_indices = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::native::device::merge_fvi", Kokkos::WithoutInitializing),
      E_fv + new_closure_total);
  Kokkos::deep_copy(Kokkos::subview(out.face_vertex_indices, Kokkos::make_pair(0, E_fv)),
                    source.face_vertex_indices);
  {
    auto n_faces = parsed.n_faces;
    auto face_offset = parsed.face_offset;
    auto p_face_vertex_offset = parsed.face_vertex_offset;
    auto face_closure_pos = parsed.face_closure_pos;
    auto filt_off = filtered_vertex_offset;
    auto final_index = vrenum.final_index_of_incoming;
    auto cell_face_offsets = out.cell_face_offsets;
    auto face_vertex_offsets = out.face_vertex_offsets;
    auto face_vertex_indices = out.face_vertex_indices;
    Kokkos::parallel_for(
        "fuse::native::device::merge_new_face_connectivity",
        Kokkos::RangePolicy<ExecSpace>(0, n_new_cells), KOKKOS_LAMBDA(const int j) {
          const int R = kept_record_of(j);
          const int nf = n_faces(R);
          const int parsed_face0 = face_offset(R);
          const int new_face0 = cell_face_offsets(E_cells + j) - E_faces;
          const int filt_vert0 = filt_off(R);
          for (int f = 0; f < nf; ++f) {
            const int pf = parsed_face0 + f;
            const int nfidx = new_face0 + f;
            const int cp0 = p_face_vertex_offset(pf);
            const int cnt = p_face_vertex_offset(pf + 1) - cp0;
            const int dst0 = face_vertex_offsets(E_faces + nfidx);
            for (int k = 0; k < cnt; ++k) {
              const int closure_val = face_closure_pos(cp0 + k);
              const int filtered_instance = filt_vert0 + closure_val;
              face_vertex_indices(dst0 + k) = final_index(filtered_instance);
            }
          }
        });
  }

  return out;
}

/// `NativeMeshBuilder::build()` (native_mesh.cpp) refuses `E-MESH-DIM` when
/// a cell's own topological dimension (`cell_dimension(cell_type)`,
/// cell_topology.hpp) disagrees with the mesh's declared `dim` -- a check
/// `unpack_migrated_mesh_device`/`merge_halo_cells_device` above do not
/// reproduce, since neither goes through `NativeMeshBuilder` at all. This
/// is the ONE validation callers of either (`migrate`, native_migration.cpp;
/// `build_native_remap_halo`, native_halo.cpp) run explicitly afterward, on
/// `Space`, before deciding whether the mesh they just unpacked/merged is
/// fit to hand to `NativeMeshView::create<Space>`. The one Host-side check
/// (`NativeMeshBuilder::build()`'s fixed-shape corner-count refusal,
/// `E-MESH-CONNECTIVITY`) is NOT reproduced, because no test here
/// exercises data that would need it.
template <class Space>
[[nodiscard]] bool all_cells_match_dimension(const MeshArrays<Space>& arrays) {
  using ExecSpace = typename Space::execution_space;
  const int n = static_cast<int>(arrays.cell_count());
  const int dim = arrays.dim;
  auto cell_type = arrays.cell_type;
  int violations = 0;
  Kokkos::parallel_reduce(
      "fuse::native::device::validate_cell_dimension", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int c, int& acc) {
        if (cell_dimension(static_cast<CellType>(cell_type(c))) != dim) acc += 1;
      },
      violations);
  return violations == 0;
}

}  // namespace fuse::native::device
