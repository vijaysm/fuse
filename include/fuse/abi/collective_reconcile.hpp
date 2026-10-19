#pragma once

/// D61 (KNOWN_ISSUES.md): the participant ABI's own agreement rule for a
/// collective entry point (docs/dev/abi-design.md, "How the participant
/// ABI reconciles a collective refusal"). SPLIT FOR TESTABILITY, the same
/// reasoning include/fuse/transport/connect.hpp and include/fuse/runtime/
/// epoch_schedule.hpp already state for their own pure halves: the
/// arithmetic that decides WHICH status, and WHICH rank's message, wins is
/// a deterministic function of every rank's own local status, with no MPI
/// call in it, so a tier-0 test can fire every branch directly. Only
/// src/abi/participant.cpp's own MPI glue (the real MPI_Allreduce call)
/// needs a live communicator, and that glue calls this so the two paths
/// cannot disagree about what "agreed" means.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace fuse::abi_detail {

/// The agreed outcome of reconciling one collective ABI entry's local
/// status across a component's ranks. `status` is the fuse_status every
/// rank must now return (abi-design.md's own "every rank refuses with the
/// same code"); `origin_rank` is the lowest-indexed rank among those whose
/// own local status equals `status` -- the rank a non-originating rank's
/// rewritten message names.
struct ReconciledStatus {
  std::int32_t status = 0;  // FUSE_OK's own value; not included to avoid a
                            // dependency on types.h from this pure header.
  int origin_rank = 0;
};

/// Pure, MPI-free: the same rule MPI_MINLOC applies to an (fuse_status,
/// rank) pair over MPI_2INT -- the smallest VALUE wins, ties broken by the
/// smallest rank index -- computed here once so src/abi/participant.cpp's
/// real `MPI_Allreduce(..., MPI_2INT, MPI_MINLOC, ...)` and this header's
/// own tier-0 test (and the test-only stub reducer standing in for MPI,
/// test/abi/participant_test.cpp's AbiParticipantCollectiveReconciliation
/// cases) cannot drift apart on what the agreed answer is.
///
/// FUSE_OK is 0 (entries.def) and every FUSE_ERR_* code is negative, so the
/// plain integer minimum already propagates ANY refusal over FUSE_OK: a
/// component where every rank is fine agrees FUSE_OK; a component where at
/// least one rank refuses agrees that refusal's own fuse_status, even when
/// two different ranks refuse with two DIFFERENT codes (the more negative
/// one wins, deterministically, not by which rank happens to run first or
/// which rank's message a caller happens to read).
///
/// `local_statuses[r]` is rank r's own local status, 0-indexed in the same
/// rank numbering the communicator being reconciled over uses. Empty input
/// is defensive (a live reconciliation call always has at least this
/// rank's own entry) and returns {FUSE_OK's value (0), rank 0}.
[[nodiscard]] inline ReconciledStatus reconcile_local_statuses(
    const std::vector<std::int32_t>& local_statuses) noexcept {
  if (local_statuses.empty()) return ReconciledStatus{};
  ReconciledStatus out{local_statuses[0], 0};
  for (std::size_t r = 1; r < local_statuses.size(); ++r) {
    if (local_statuses[r] < out.status) {
      out.status = local_statuses[r];
      out.origin_rank = static_cast<int>(r);
    }
  }
  return out;
}

/// This rank's own colour and key for the round-4 per-component
/// MPI_Comm_split (src/abi/participant.cpp's own impl_initialize_local()),
/// decided here as pure arithmetic for the same "split for testability"
/// reason this header's other half already gives: the decision that was
/// WRONG in round 3 (every rank computing the SAME colour 0, landing every
/// component on one shared communicator -- KNOWN_ISSUES.md D61's own
/// round-4 paragraph) is exactly the part worth pinning directly, with no
/// MPI call in the way.
struct SplitColourKey {
  int colour = 0;
  int key = 0;
};

/// `component_value` is this rank's own `my_component`'s `ComponentId::
/// value()`, or `std::nullopt` when this rank hosts no component at all
/// (`fuse_ctx::my_component` unset at setup() time). `position_in_rankset`
/// is this rank's own position in that component's declared `RankSet`
/// (`RankSet::index_of`), or a negative value for the defensive case that
/// should never happen on the real setup()->initialize() path (a resolved
/// my_component whose own graph RankSet somehow does not name this rank).
///
/// Returns `std::nullopt` in either "no component" case above -- the
/// caller maps that to MPI_UNDEFINED, since this function has no MPI
/// dependency and is exercised directly at tier 0. Otherwise returns the
/// `{colour, key}` pair `MPI_Comm_split` partitions the base communicator
/// with: `colour` is `component_value` itself (DISTINCT per component --
/// never a uniform value two different components could both produce,
/// which is exactly the property round 3's `make_participant_set_reducer`
/// call did not have); `key` is `position_in_rankset` unchanged, ordering
/// ranks within the resulting sub-communicator by their own position in
/// the component's declared RankSet.
[[nodiscard]] inline std::optional<SplitColourKey> decide_split_colour_key(
    std::optional<std::uint32_t> component_value, int position_in_rankset) noexcept {
  if (!component_value.has_value() || position_in_rankset < 0) return std::nullopt;
  return SplitColourKey{static_cast<int>(*component_value), position_in_rankset};
}

}  // namespace fuse::abi_detail
