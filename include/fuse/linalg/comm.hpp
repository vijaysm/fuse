#pragma once

/// Communication, injected rather than included (docs/dev/layering.md:
/// `linalg` includes only `base`, so this header never names `mpi.h` --
/// the same isolation `fuse::partition::detail::collectives` and
/// `fuse::search::distributed` use for their own MPI symbols). The two
/// facades below are everything `DistVector`/`DistSparseMatrix` ask of
/// communication: a global reduction (`Reducer`) and a one-shot ghost
/// update along a pattern built once (`GhostExchanger`). A serial
/// implementation of both exists unconditionally for a one-rank build with
/// no MPI at all; the MPI implementations are compiled only when
/// `FUSE_ENABLE_MPI` adds `linalg/comm_mpi.cpp` to this library (this
/// file's own declarations of `make_mpi_reducer`/`make_mpi_exchanger` stay
/// inside `#if FUSE_HAVE_MPI` below for the same reason
/// `fuse/transport/comm_graph.hpp` does: the factories exist as symbols
/// only in a build that can define them).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/result.hpp"

namespace fuse::linalg {

/// True if MPI calls in THIS process may be handed CUDA/HIP device
/// pointers directly (a run-time probe, `MPIX_Query_cuda_support()` where
/// this MPI's own extension declares it, falling back to
/// `cmake/FuseMpiGpuAware.cmake`'s configure-time constant where it does
/// not; always false in a build with no MPI, and forced false by
/// `FUSE_TRANSPORT_HOST_STAGING=1` regardless of what either reports).
/// THE one probe `fuse::native::mpi_is_gpu_aware()`
/// (src/backend/native_migration.cpp) and `fuse::transport::
/// MpiTransport::is_gpu_aware()` (src/transport/mpi_transport.cpp) both
/// call now, in `fuse::linalg` because that layer sits below both
/// (tools/check-layering.py) and both already may depend on it --
/// Migration track, 2026-10-15; src/linalg/mpi_gpu_aware.cpp's own header
/// comment has the full history of why this used to be two copies.
[[nodiscard]] bool mpi_is_gpu_aware();

/// A global reduction over whatever communicator this instance was built
/// for. One rank in a build with no MPI, `comm size()` ranks under MPI --
/// the same "kNullComm needs no MPI symbol at all" convention
/// `fuse::partition::detail`'s collectives use, except here the choice is
/// baked into WHICH concrete `Reducer` a caller holds rather than threaded
/// through a `CommHandle` argument on every call, since `DistVector` and
/// `DistSparseMatrix` hold one for their whole lifetime.
class Reducer {
 public:
  virtual ~Reducer() = default;

  /// The sum of `local` across every participant.
  [[nodiscard]] virtual double sum(double local) const = 0;
  /// `inout[i]` becomes the sum of `inout[i]` across every participant,
  /// for `i` in `[0, n)`. The vectorised sibling of the scalar overload
  /// above, for a caller that has more than one scalar to reduce at once
  /// (e.g. several dot products in one block) and does not want one
  /// collective per scalar.
  virtual void sum(double* inout, std::size_t n) const = 0;
  /// The maximum of `local` across every participant.
  [[nodiscard]] virtual double max(double local) const = 0;

  /// This participant's position, `[0, size())`.
  [[nodiscard]] virtual int rank() const = 0;
  /// The number of participants, `>= 1`.
  [[nodiscard]] virtual int size() const = 0;

  /// `recv[r*n_each .. (r+1)*n_each)` becomes participant `r`'s own `send`,
  /// for every `r` in `[0, size())`, in RANK ORDER -- the one collective
  /// `fuse::linalg::SchurSlabSolver` needs that `sum`/`max` cannot answer:
  /// assembling a dense matrix or vector that every rank must hold in full
  /// (the Schur complement itself, and its reduced interface right-hand
  /// side), from each rank's own contiguous contribution, with no
  /// reduction of the values themselves. `send` carries `n_each` entries;
  /// `recv` must have room for `n_each * size()`. The "Containers" track's
  /// own two facades (comm.hpp's header comment) are UNCHANGED otherwise;
  /// this is the one addition the Schur-complement solver's own task
  /// authorised explicitly, reported here as instructed rather than left
  /// to be discovered as an undocumented diff.
  virtual void allgather(const double* send, std::size_t n_each, double* recv) const = 0;
};

/// One rank, no MPI symbol touched: `sum`/`max` are the identity, `rank()
/// == 0`, `size() == 1`. The reference every other `Reducer` must agree
/// with when there is exactly one participant.
[[nodiscard]] std::shared_ptr<const Reducer> make_serial_reducer();

/// Where a `DistVector`'s ghost entries come from, one entry per ghost, IN
/// GHOST ORDER (the order `DistVector::data()[owned_size() + g]` and
/// `GhostExchanger::update`'s `ghosts[g]` both use). `owner_rank[g]` is the
/// participant that OWNS the entry; `remote_local_index[g]` is that
/// participant's own local (owned-block) position for it -- never a global
/// id, the same convention `fuse::search::PointHit::index` documents for
/// an entirely different kind of pattern.
struct GhostPattern {
  std::vector<int> owner_rank;
  std::vector<std::int64_t> remote_local_index;
};

/// A one-shot ghost update along a pattern this instance was built from.
/// The pattern itself -- which ranks to talk to, how many entries each
/// way, the pack/unpack maps -- is derived ONCE, by the factory that built
/// this object (`make_serial_exchanger`/`make_mpi_exchanger`); `update()`
/// repeats only the data motion, which is why it takes no `GhostPattern`
/// argument of its own.
///
/// DEVIATION FROM THE TASK'S INTERFACE CONTRACT, reported per that task's
/// own instruction ("any deviation... must be stated"): the contract's
/// `GhostExchanger` has exactly one method, `update`, and its `DistVector`
/// constructor takes only `(owned_n, exchanger, reducer)` -- with nothing,
/// anywhere, telling a freshly constructed `DistVector` how many ghost
/// slots to allocate. `ghost_size()` is added here so the vector can ask
/// the one object that already knows (the exchanger was built from a
/// `GhostPattern`, which fixes the ghost count) rather than needing a
/// fourth constructor argument that would have to agree with the
/// exchanger's own pattern by construction-site convention alone.
class GhostExchanger {
 public:
  virtual ~GhostExchanger() = default;

  /// How many ghost entries this exchanger's pattern carries --
  /// `GhostPattern::owner_rank.size()` at the point this exchanger was
  /// built. `DistVector` sizes its ghost region from this, once, at
  /// construction.
  [[nodiscard]] virtual std::size_t ghost_size() const = 0;

  /// `ghosts[g]` becomes `owner_rank[g]`'s current value at
  /// `remote_local_index[g]`, for every `g` in `[0, ghost_n)`.
  /// `ghost_n` must equal `ghost_size()`; `owned_n` must equal the number
  /// of entries this participant owns (checked against what the pattern
  /// recorded about this participant when the pattern was built).
  ///
  /// COLLECTIVE under MPI: every participant named anywhere in ANY
  /// participant's pattern must call this when any one of them does --
  /// the "global collective discipline" `fuse::partition::detail`'s own
  /// header states for exactly this reason (docs/dev/partitioning.md).
  [[nodiscard]] virtual Result<void> update(const double* owned, std::size_t owned_n,
                                            double* ghosts, std::size_t ghost_n) const = 0;
};

/// Zero ghosts, update() a no-op: the "no ghosts at all" exchanger a
/// single-rank `DistVector` is built with. Not named in the task's
/// interface contract (only `make_serial_reducer` is) -- added because
/// requirement 1 ("serial reducer/exchanger, no ghosts") names both a
/// serial reducer AND a serial exchanger, and nothing else in the
/// contract supplies the latter.
[[nodiscard]] std::shared_ptr<const GhostExchanger> make_serial_exchanger();

#if FUSE_HAVE_MPI
/// `comm_f` is an `MPI_Comm_c2f` handle, exactly like
/// `fuse::search::CommHandle` -- this header stays free of `mpi.h` even
/// though the factory exists only in an MPI build; `src/linalg/comm_mpi.cpp`
/// is the one translation unit that resolves it back with `MPI_Comm_f2c`.
[[nodiscard]] std::shared_ptr<const Reducer> make_mpi_reducer(int comm_f);

/// A `Reducer` over EXACTLY one iteration region's own participant set --
/// `KNOWN_ISSUES.md` D31, `docs/dev/newton-krylov.md` Sec. 9.4's own
/// `ParticipantSetReducer` name. `make_mpi_reducer()` above reduces over
/// EVERY rank of the communicator it is handed; this factory is for the
/// case `RegionRuntime::participants` names -- a region's own rank set is
/// in general a PROPER SUBSET of `base_comm_f`'s ranks, and reducing over
/// the whole base communicator there would silently fold in ranks that
/// host no part of this region's own Newton vector at all.
///
/// COLLECTIVE OVER `base_comm_f`, called by EVERY rank of that
/// communicator, in the SAME fixed order as every other rank -- this is
/// an `MPI_Comm_split` underneath (see `src/linalg/comm_mpi.cpp`'s own
/// header comment for why a split, not the point-to-point addressing
/// `reduce_region_residuals()`/`gather_window_block_samples()` use, is
/// this factory's own contract), and `MPI_Comm_split` is collective over
/// `base_comm_f` regardless of which ranks end up IN the split -- a rank
/// outside `participants` still calls this function (with `MPI_UNDEFINED`
/// as its own colour underneath) so the other ranks' own split does not
/// wait on it forever.
///
/// `participants` must be the IDENTICAL list, in the IDENTICAL order, on
/// every rank of `base_comm_f` -- it is not re-validated for agreement
/// here (the same trust `resolve_regions()`'s own deterministic,
/// graph-derived construction of `RegionRuntime::participants` already
/// requires of every other collective built from it, `attach_comm_graphs()`
/// included). The SUB-communicator's own rank order follows `participants`'
/// own order (participant `i` becomes sub-rank `i`), the SAME order
/// `decide_window_convergence()`'s existing cross-rank reduction for
/// Picard already uses -- `ParticipantSetReducer::rank()` therefore reads
/// as "this rank's own position in `participants`", not its base rank
/// number.
///
/// Returns `nullptr` (success) for a rank NOT named in `participants` --
/// "a rank outside the participant set must not enter a reduction" is
/// enforced by giving such a rank no `Reducer` object to call at all,
/// rather than one whose `sum()`/`max()`/`allgather()` would need a
/// runtime guard against being invoked. A caller that already knows it
/// is not a participant (the SAME `std::find(part.begin(), part.end(),
/// me)` test `reduce_region_residuals()` uses) should not need the
/// returned pointer at all; one is still returned (as `nullptr`) rather
/// than this function being skippable, because the `MPI_Comm_split`
/// itself is NOT skippable.
///
/// Failure (`E-LINALG-PARTICIPANT-SPLIT`) means the split itself could
/// not be built (an out-of-range rank in `participants`, or
/// `MPI_Comm_split` itself returning an error code) -- this is the SAME
/// failure shape `MoabMeshBackend::set_placements()`'s own
/// `MPI_Comm_split` loop reports, reached identically on every rank of
/// `base_comm_f` so a caller can propagate it without any rank having
/// already returned early.
struct ParticipantSetReducerResult {
  /// `nullptr` on a rank not named in `participants` -- see the factory's
  /// own "a rank outside the participant set must not enter a reduction"
  /// paragraph above.
  std::shared_ptr<const Reducer> reducer;
  /// The split's own `MPI_Comm_c2f` handle, for a caller (`fuse::petsc::
  /// solve()`, D32) that needs the raw communicator itself rather than
  /// this comm-free `Reducer` facade -- bundled with `reducer` so the
  /// split is performed EXACTLY ONCE per call, rather than once for the
  /// `Reducer` and again for the handle (`MPI_Comm_split` builds a NEW
  /// communicator every call, even given identical colour/key, so a
  /// second call would not recover the first one's own handle at all).
  /// `0` (never a real handle) on a rank not named in `participants`,
  /// paired with `reducer == nullptr`.
  int comm_f = 0;
};

[[nodiscard]] Result<ParticipantSetReducerResult> make_participant_set_reducer(
    int base_comm_f, const std::vector<int>& participants);

/// Builds the exchange pattern ONCE: a preliminary exchange of request
/// counts and indices (so every participant learns which of its OWNED
/// entries the others want, and in what order), from which the counts,
/// displacements and pack/unpack maps `update()` needs are fixed for the
/// returned object's whole lifetime. `pattern` is THIS participant's own
/// ghost list; `owned_n` is how many entries it owns. See
/// `src/linalg/comm_mpi.cpp`'s header comment for the exact two-phase
/// (`MPI_Alltoall` for counts, `MPI_Alltoallv` for indices) derivation and
/// why a plain `MPI_Alltoallv` is enough for `update()` itself.
[[nodiscard]] std::shared_ptr<const GhostExchanger> make_mpi_exchanger(
    int comm_f, const GhostPattern& pattern, std::size_t owned_n);

/// The counts, displacements and pack/unpack maps `make_mpi_exchanger()`'s
/// own `update()` already derives once and reuses every call
/// (src/linalg/comm_mpi.cpp's header comment), exposed so a caller that
/// can pack and unpack on its OWN memory space -- a Kokkos kernel on a
/// device-resident ghost exchange (`fuse::backend`) or on a device-
/// resident nodal-remap exchange plan (`fuse::backend::
/// nodal_remap_distributed`) -- can drive the SAME `MPI_Alltoallv` over
/// buffers it already owns, instead of `update()`'s own host-side
/// pack/unpack loop. `GhostExchanger::update()`'s own signature and
/// contract (this file's header comment above) are UNCHANGED; this is an
/// ADDITIONAL facet the concrete MPI exchanger also implements, reached
/// with `std::dynamic_pointer_cast<const PackedExchanger>` on whatever
/// `make_mpi_exchanger()` returned (`nullptr` for the serial exchanger,
/// which carries no pattern to expose -- the same fact `ghost_size() ==
/// 0` already reports).
///
/// NAMING, so the two exchange methods below read against the same
/// mental picture `update()`'s own header comment draws: this
/// participant's REQUEST side is what it asks its peers for (the ghosts
/// it wants); its ANSWER side is what its peers ask IT for (which of its
/// own owned entries they want). `pack_index()`/`send_counts()`/
/// `send_displs()` describe the ANSWER side (`update()`'s own
/// `ans_index`/`ans_counts`/`ans_displs`, the data THIS rank sends);
/// `unpack_index()`/`recv_counts()`/`recv_displs()` describe the REQUEST
/// side (`update()`'s own `fill_index`/`req_counts`/`req_displs`, the
/// data THIS rank receives).
class PackedExchanger {
 public:
  virtual ~PackedExchanger() = default;

  /// This participant's own OWNED entries to pack and send, grouped by
  /// peer in send order: peer `p`'s slice is `pack_index()[send_displs()[p]
  /// .. send_displs()[p] + send_counts()[p])`, each entry a LOCAL index
  /// into the `owned` array `GhostExchanger::update()` itself would have
  /// taken.
  [[nodiscard]] virtual const std::vector<std::int64_t>& pack_index() const = 0;
  /// Where each received value lands, grouped by peer in receive order:
  /// peer `p`'s slice is `unpack_index()[recv_displs()[p] ..
  /// recv_displs()[p] + recv_counts()[p])`, each entry a GHOST-ORDER index
  /// into the `ghosts` array `GhostExchanger::update()` itself would have
  /// filled.
  [[nodiscard]] virtual const std::vector<std::int64_t>& unpack_index() const = 0;
  [[nodiscard]] virtual const std::vector<int>& send_counts() const = 0;
  [[nodiscard]] virtual const std::vector<int>& send_displs() const = 0;
  [[nodiscard]] virtual const std::vector<int>& recv_counts() const = 0;
  [[nodiscard]] virtual const std::vector<int>& recv_displs() const = 0;

  /// One `MPI_Alltoallv` over buffers the caller has ALREADY packed and
  /// will unpack itself: `send` holds `send_displs().back() +
  /// send_counts().back()` entries in ANSWER order (see `pack_index()`);
  /// `recv` receives the same total in REQUEST order (see
  /// `unpack_index()`) -- the identical data motion `update()`'s own
  /// single `MPI_Alltoallv` performs, minus that function's own host-side
  /// pack/unpack loops. `send`/`recv` may be device pointers under a
  /// GPU-aware MPI (`mpi_is_gpu_aware()`); this call does no staging of
  /// its own either way -- same collective discipline as
  /// `GhostExchanger::update()`.
  [[nodiscard]] virtual Result<void> exchange_packed(const double* send, double* recv) const = 0;

  /// The ADJOINT exchange `nodal_remap_distributed.cpp`'s H^T scatter-add
  /// needs: roles swapped relative to `exchange_packed()` above. `send`
  /// holds `recv_displs().back() + recv_counts().back()` entries in
  /// REQUEST order (what `exchange_packed()` would have received);
  /// `recv` receives `send_displs().back() + send_counts().back()`
  /// entries in ANSWER order (what `exchange_packed()` would have sent).
  /// Only the Alltoallv's own send/receive roles are swapped -- neither
  /// side is summed by this call; a caller receiving into the SAME
  /// destination index more than once (several requesters wanting the
  /// same owned entry) must accumulate itself, exactly as it would for
  /// any other scatter-add.
  [[nodiscard]] virtual Result<void> exchange_packed_adjoint(const double* send,
                                                              double* recv) const = 0;
};
#endif  // FUSE_HAVE_MPI

}  // namespace fuse::linalg
