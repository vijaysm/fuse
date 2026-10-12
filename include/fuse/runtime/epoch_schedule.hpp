#pragma once

/// Who talks to whom when participants of a data move agree on their status.
///
/// Separated from the MPI calls that carry it out for one reason: a mistake
/// here is a **hang**, not a wrong number, and a hang can only be observed by
/// running a real multi-rank job. As arithmetic over `(participants, index)`
/// it is checkable without MPI at all -- `runtime.epoch_schedule` asserts that
/// every send has exactly one matching receive at every participant count, so
/// a mismatch fails a plain unit test instead of stalling a parallel leg.
///
/// The shape is a binomial reduce to participant 0 followed by a broadcast
/// back. Both phases depend only on the participant count and this rank's
/// position in the (sorted, identically derived) participant list, so every
/// participant computes the same pairings without communicating to agree on
/// them.

#include <array>
#include <cstddef>

namespace fuse {

/// One participant's part in the exchange, in the order it must be performed.
///
/// Fixed storage, no allocation: this is built once per agreement point per
/// step, and the runtime's standing property is that a running plan touches no
/// allocator (`RunnerTransport.TheTransportPathAllocatesNothingWhileRunning`).
/// A first version returned `std::vector`s and broke it -- eight allocations
/// per period on a two-component job.
struct EpochSchedule {
  /// At most one receive per doubling of the stride, so 31 bounds any
  /// participant count an `int` can express. Only the first
  /// `reduce_recv_count` entries are meaningful.
  std::array<int, 31> reduce_recv_from{};
  int reduce_recv_count = 0;

  /// Participant index this one sends its (reduced) status to, or -1 for
  /// participant 0, which holds the answer.
  int reduce_send_to = -1;

  /// Where the agreed answer comes from, or -1 for participant 0.
  int result_recv_from = -1;

  /// Participant 0 sends the agreed answer to every other participant. Held as
  /// a count rather than a list because it is simply `1 .. participants-1`,
  /// and materialising that list is the one part of this schedule whose size
  /// grows with the job.
  int result_send_count = 0;
};

/// `index` is a position in the participant list, not an MPI rank: the caller
/// maps positions to ranks. Out-of-range input yields an empty schedule rather
/// than an assertion, since a rank that does not take part simply does nothing.
///
/// RT-8 (review 2026-10-11): also empty for `participants` at or past
/// `kMaxStrideParticipants` (2^30) -- the stride walk below doubles `stride`
/// (and computes `2 * stride`) as a plain `int`; past this bound the next
/// doubling would be signed-integer-overflow UB, latent at any MPI job size
/// this library could plausibly be run at (this file's own "31 bounds any
/// participant count an int can express" comment already notes the
/// practical ceiling), but UB is UB whether or not a real run ever reaches
/// it. Clamping the DOMAIN here, rather than widening the stride walk's own
/// arithmetic to `unsigned`, keeps every line of that walk -- and the
/// mutations that anchor to them -- unchanged; a participant count this
/// large has no other meaning here anyway ("out of range" already returns
/// empty for `index`, by the same convention).
///
/// No allocation, so it is safe to call inside a running plan.
inline constexpr int kMaxStrideParticipants = 1 << 30;

[[nodiscard]] inline EpochSchedule epoch_schedule(int participants, int index) {
  EpochSchedule s;
  if (participants <= 1 || index < 0 || index >= participants ||
      participants >= kMaxStrideParticipants) {
    return s;
  }

  for (int stride = 1; stride < participants; stride *= 2) {
    if ((index % (2 * stride)) == 0) {
      const int from = index + stride;
      if (from < participants) {
        s.reduce_recv_from[static_cast<std::size_t>(s.reduce_recv_count++)] = from;
      }
    } else {
      // Hands its result up and takes no further part in the reduce: anything
      // it received has already been folded in.
      s.reduce_send_to = index - stride;
      break;
    }
  }

  if (index == 0) {
    s.result_send_count = participants - 1;
  } else {
    s.result_recv_from = 0;
  }
  return s;
}

/// RT-9 (review 2026-10-11): the pure comparison behind `E-RT-EPOCH-TAG`
/// (`resolve_epoch_participants()`, `src/runtime/runner.cpp`), factored
/// out so it is unit-testable without MPI at all -- rank-agreement.md's
/// own point about this refusal ("a refusal before the run is a better
/// failure than 'Invalid tag' raised inside a reduction") had no test
/// that made the comparison fire, so a regression that deleted or
/// inverted it would have passed the whole suite silently. `reported_
/// tag_ub` is whatever `MPI_Comm_get_attr(..., MPI_TAG_UB, ...)`
/// returned for the communicator in use; `max_used_tag` is `fuse::tags::
/// kMaxUsed` -- passed in rather than referenced here so this header
/// (already dependency-free apart from `<array>`/`<cstddef>`) gains no
/// include on `fuse/transport/tags.hpp`.
[[nodiscard]] inline bool tag_ub_too_low(int reported_tag_ub,
                                         int max_used_tag) {
  return reported_tag_ub < max_used_tag;
}

}  // namespace fuse
