#pragma once

/// Device-resident dedup-by-global-id (KNOWN_ISSUES D51): the receive-side
/// MERGE decision `unpack_migrated_mesh`'s vertex dedup and
/// `merge_halo_cells`'s cell dedup both make -- "is this global id already
/// held, and if not, which ONE of possibly several incoming copies is the
/// canonical new record" -- as Kokkos kernels on `Space`, with no
/// `std::unordered_map`/`std::unordered_set` and no host round trip.
///
/// ALGORITHM (the task's own two named choices: stated here, not assumed).
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
/// Not implemented here: this track's remaining time went to proving the
/// dedup DECISION itself (which gids are new, which incoming copy is
/// canonical) is correct and reusable, which is the genuinely new
/// algorithmic piece KNOWN_ISSUES D51 asks for; the renumbering pass is a
/// mechanical follow-on sketched here, not a second algorithm.
///
/// WHAT THIS DOES NOT DO (yet, see KNOWN_ISSUES D51's own narrowing): the
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

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

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
  // the first (measured: a first draft used the exclusive-scan order and
  // `native_mesh_merge_device` caught it immediately -- a duplicate run's
  // second-and-later members landed one slot ahead of its first, merging
  // the WRONG pair of distinct values into one slot). Incrementing first
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

}  // namespace fuse::native::device
