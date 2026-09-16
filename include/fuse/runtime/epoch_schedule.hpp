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

#include <cstddef>
#include <vector>

namespace fuse {

/// One participant's part in the exchange, in the order it must be performed.
struct EpochSchedule {
  /// Participant indices this one receives a status from, during the reduce,
  /// in order.
  std::vector<int> reduce_recv_from;
  /// Participant index this one sends its (reduced) status to, or -1 for
  /// participant 0, which holds the answer.
  int reduce_send_to = -1;
  /// Where the agreed answer comes from, or -1 for participant 0.
  int result_recv_from = -1;
  /// Participants this one sends the agreed answer to; non-empty only for
  /// participant 0.
  std::vector<int> result_send_to;
};

/// `index` is a position in the participant list, not an MPI rank: the caller
/// maps positions to ranks. Out-of-range input yields an empty schedule rather
/// than an assertion, since a rank that does not take part simply does nothing.
[[nodiscard]] inline EpochSchedule epoch_schedule(int participants, int index) {
  EpochSchedule s;
  if (participants <= 1 || index < 0 || index >= participants) return s;

  for (int stride = 1; stride < participants; stride *= 2) {
    if ((index % (2 * stride)) == 0) {
      const int from = index + stride;
      if (from < participants) s.reduce_recv_from.push_back(from);
    } else {
      // Hands its result up and takes no further part in the reduce: anything
      // it received has already been folded in.
      s.reduce_send_to = index - stride;
      break;
    }
  }

  if (index == 0) {
    for (int k = 1; k < participants; ++k) s.result_send_to.push_back(k);
  } else {
    s.result_recv_from = 0;
  }
  return s;
}

}  // namespace fuse
