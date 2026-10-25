#pragma once

#include "fuse/transport/tags.hpp"

#include "fuse/base/config.hpp"  // defines the FUSE_HAVE_* macros read below

// The Crystal Router (Fox et al., *Solving Problems on Concurrent Processors*,
// 1988): deliver every message to the rank it names, in O(log P) rounds,
// without any rank ever learning the global communication pattern.
//
// WHY THIS FILE IS IN src/ AND NOT include/. It names `MPI_Comm` in its
// interface, and this project's standing rule is that `mpi.h` stays off a
// consumer's include path (`MpiTransport::set_communicator` takes an
// `MPI_Comm_c2f` handle for exactly that reason). `CrystalRouter` is an
// internal primitive with one caller inside the library -- `comm_graph.cpp` --
// so it is a private header rather than a public one with a handle-shaped
// interface it does not need. `test/parallel/comm_graph_test.cpp` reaches it
// by adding `src/` to its include path; it already includes `mpi.h` itself.
//
// WHY IT IS REIMPLEMENTED RATHER THAN TAKEN FROM MOAB. `moab::gs_data`
// (`moab/gs.hpp`) is a proven implementation of this same algorithm, and using
// it was the first instinct. It was rejected in design review: it would make
// the transport layer depend on the mesh-backend layer, breaking the "works
// for SerialMeshBackend too, no MOAB required" property that
// `mpi_transport.cpp` -- which also depends on no MOAB -- already has. See
// docs/history/soma/specs/2026-09-11-comm-graph-design.md Section 3.

#if FUSE_HAVE_MPI

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse::detail {

/// One message, addressed by rank of the router's communicator.
struct Envelope {
  int                       dest_rank = -1;
  std::vector<std::uint8_t> payload;
};

/// Routes `Envelope`s to the ranks they name.
///
/// COLLECTIVE. Every rank of `comm` must call `route()`, the same number of
/// times, in the same order -- including a rank with nothing to send and
/// nothing to receive, which calls it with an empty vector and gets an empty
/// vector back. This is the same rule `MpiTransport` states for
/// `MPI_Alltoallv`, and for the same reason.
///
/// COMMUNICATION CONTEXT. `route()` uses one tag (`kTag`) and only
/// point-to-point calls, each between a pair of ranks that exchange at most
/// one message per `route()`. MPI's non-overtaking guarantee therefore makes
/// consecutive `route()` calls on the same communicator safe without a
/// `MPI_Comm_dup`. It is NOT safe to interleave a `route()` with another
/// point-to-point exchange on the same communicator using `kTag`; `CommGraph`
/// hands it a freshly created union communicator, so nothing else is on it.
///
/// COMPLEXITY. `ceil(log2(P))` rounds, each a single pairwise exchange, so
/// each rank sends at most `ceil(log2(P))` messages regardless of how many
/// distinct ranks it addresses. No rank ever holds, or allgathers, the whole
/// pattern -- which is the property `MpiTransport::build_plan`'s
/// allgather-everything matching does not have.
///
/// THE PARTITION, and why the pairing is exact. Round `r` holds a contiguous
/// rank range `[lo, hi]` (initially `[0, P-1]`), splits it at `mid` with
/// `bl = ceil(n/2)` ranks below and `tl = floor(n/2)` above, and each side
/// hands the other everything addressed across the split. Because
/// `bl - tl` is 0 or 1, the pairing `lo+i <-> mid+1+i` covers every rank but
/// at most ONE: when `n` is odd, rank `mid` has no partner and sends to
/// `mid+1`, which knows to expect it from `n` alone. Every rank derives the
/// whole schedule from `(lo, hi, rank)`, so no round needs a count exchange,
/// a barrier, or an allgather.
class CrystalRouter {
 public:
  /// `comm` is borrowed, not duplicated, and must outlive the router.
  explicit CrystalRouter(MPI_Comm comm);

  CrystalRouter(const CrystalRouter&) = delete;
  CrystalRouter& operator=(const CrystalRouter&) = delete;
  CrystalRouter(CrystalRouter&&) = default;
  CrystalRouter& operator=(CrystalRouter&&) = default;

  /// COLLECTIVE. Returns every envelope routed TO this rank, in unspecified
  /// order. An envelope addressed to this rank is returned without leaving
  /// the process. `outgoing` is consumed.
  ///
  /// Errors, all three agreed collectively before any rank acts on them, so
  /// each is an identical return everywhere rather than one rank unwinding
  /// while its peers block:
  ///   * `E-CROUTER-RANK` -- a `dest_rank` outside `[0, P)`.
  ///   * `E-CROUTER-SIZE` -- this rank's outgoing envelopes total more than
  ///     the `INT_MAX` bytes an MPI count can express.
  ///   * `E-CROUTER-MPI` -- the agreement `MPI_Allreduce` itself failed.
  ///   * `E-CROUTER-COMM` -- called on a null communicator. Purely local.
  ///
  /// After the agreement there is no error return left: a routing round is
  /// between two point-to-point calls its peers are already committed to, so
  /// an MPI failure, a heap failure, or a round whose forwarded traffic grows
  /// past `INT_MAX` all `MPI_Abort` with a message. Same window, same answer,
  /// as `mpi_transport.cpp`.
  [[nodiscard]] Result<std::vector<Envelope>> route(
      std::vector<Envelope> outgoing);

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }

  /// Rounds the last `route()` performed. Diagnostic: the test that pins
  /// "O(log P), not O(P)" asserts on it.
  [[nodiscard]] int last_rounds() const noexcept { return last_rounds_; }

  /// The tag `route()` occupies on the communicator.
  // fuse/transport/tags.hpp holds every tag; see there for why.
  static constexpr int kTag = tags::kCrystalRouter;

 private:
  MPI_Comm comm_ = MPI_COMM_NULL;
  int      rank_ = -1;
  int      size_ = 0;
  int      last_rounds_ = 0;
};

}  // namespace fuse::detail

#endif  // FUSE_HAVE_MPI
