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
#endif  // FUSE_HAVE_MPI

}  // namespace fuse::linalg
