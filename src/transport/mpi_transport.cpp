// Task 5.4: n->m redistribution between two disjoint rank sets (spec §4.1).
//
// THE SHAPE OF THIS FILE IS DICTATED BY TWO CONSTRAINTS THAT PULL APART.
//
// 1. Task 4.6's hot loop. `Runner::Impl::execute` calls `prepare()` and then
//    `move()` inside the coupling-cycle loop, and that loop is proved allocation-free
//    by replacing global operator new. So `prepare()` may not build anything:
//    no allocation, no rank-set analysis, no matching, no buffer sizing, no
//    communicator work. All of it lives in `build_plan()`, which is called
//    once before the run.
//
// 2. Deadlock. Every collective below is reached by every rank of the
//    communicator on every path, including error paths, and every error is
//    agreed before any rank acts on it. Concretely:
//
//      - Local validation sets a code; the code is MPI_Allreduce'd (MAX)
//        before anyone returns, so either all ranks return the error or none
//        does. A rank that fails validation still reaches that Allreduce.
//      - The matching itself is computed from allgathered data that is
//        identical on every rank, so "no source holds global id g" is
//        detected everywhere at once, not only on the rank that wanted g.
//        A second Allreduce after the matching catches anything that is not
//        deterministic (a bad_alloc on one rank).
//      - A rank in neither rank set still enters MPI_Alltoallv with zero
//        counts.
//      - `prepare()` performs no collective at all, so its error return
//        cannot desynchronise anything by itself; but because a caller that
//        stops there leaves its peers in `move()`, its only failure modes are
//        misuse (no plan, or a spec that is not the planned one), never data.
//
//    The one window that cannot be made both deadlock-free and correct is a
//    heap failure between two collectives: the rank that throws cannot reach
//    the next collective, and it cannot fake it either. That window is a
//    single allocation, and it is handled by MPI_Abort with a message --
//    a job that dies loudly, not one that hangs.

#include "fuse/base/config.hpp"  // FUSE_MPI_GPU_AWARE
#include "fuse/transport/transport.hpp"
#include "fuse/transport/comm_graph.hpp"
#include "fuse/transport/rank_agreement.hpp"

#include <fmt/format.h>

#include <mpi.h>
// Declares MPIX_Query_cuda_support()/MPIX_CUDA_AWARE_SUPPORT when this
// OpenMPI was built with CUDA-aware support -- the same header
// cmake/FuseMpiGpuAware.cmake's configure-time probe compiles against.
// Always present on OpenMPI (the extension macro inside it is what's
// conditional); #if defined(...) below, never an unconditional call.
#include <mpi-ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuse {
namespace {

/// A run-time probe, duplicated rather than shared with
/// `fuse::native::mpi_is_gpu_aware()` (src/backend/native_migration.cpp):
/// `transport` does not link `backend` (the two layers are related only by
/// the header-include direction tools/check-layering.py enforces, not by
/// the CMake library graph), and the same duplicate-rather-than-couple
/// choice `fuse::native::CommHandle` already makes for `fuse::partition`'s
/// own communicator type is the precedent (native_migration.hpp's own
/// comment). `MPIX_Query_cuda_support()` (OpenMPI's extension, declared by
/// <mpi-ext.h> exactly when `FUSE_MPI_GPU_AWARE`'s own configure-time
/// check -- cmake/FuseMpiGpuAware.cmake -- found the macro) reports whether
/// CUDA-aware buffers are actually enabled for THIS process; where the
/// extension is not declared at all, `FUSE_MPI_GPU_AWARE` is the only
/// signal there is. `FUSE_TRANSPORT_HOST_STAGING=1` forces this to false
/// regardless of either probe -- the one lever that lets the host-staged
/// fallback (this file's `MpiTransport::move`, the Runner's own
/// `OpKind::Transport` case) be exercised by a test even on a machine
/// whose MPI genuinely is GPU-aware.
bool transport_mpi_is_gpu_aware() {
  if (const char* force = std::getenv("FUSE_TRANSPORT_HOST_STAGING");
      force != nullptr && force[0] != '\0' && force[0] != '0') {
    return false;
  }
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  return MPIX_Query_cuda_support() != 0;
#else
  return FUSE_MPI_GPU_AWARE != 0;
#endif
}

constexpr std::size_t kIntMax =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "global ids travel as MPI_UINT64_T");

/// Codes the two agreement Allreduces carry. Ordered so that MPI_MAX picks a
/// deterministic winner when different ranks fail differently; the message is
/// this rank's own when it failed, and a pointer to the collective outcome
/// otherwise.
enum class Err : int {
  None = 0,
  Ranks = 1,
  Size = 2,
  Match = 3,
  Mpi = 4,
  Alloc = 5,
  // The send and receive halves of a peer-wise plan disagree. Distinct from
  // Match, which is a matching failure within one side's own data.
  PeerMatch = 6,
};

[[nodiscard]] const char* code_of(Err e) noexcept {
  switch (e) {
    case Err::None:  return "";
    case Err::Ranks: return "E-TRANSPORT-RANKS";
    case Err::Size:  return "E-TRANSPORT-SIZE";
    case Err::Match: return "E-TRANSPORT-MATCH";
    case Err::Mpi:   return "E-TRANSPORT-MPI";
    case Err::Alloc: return "E-TRANSPORT-ALLOC";
    case Err::PeerMatch: return "E-TRANSPORT-PEERMATCH";
  }
  return "E-TRANSPORT-MPI";
}

[[nodiscard]] std::string mpi_error_text(int rc) {
  char buf[MPI_MAX_ERROR_STRING];
  int len = 0;
  // MPI_Error_string can itself fail on a bogus code; do not recurse into it.
  if (MPI_Error_string(rc, buf, &len) != MPI_SUCCESS || len <= 0) {
    return "MPI error code " + std::to_string(rc);
  }
  return std::string(buf, static_cast<std::size_t>(len));
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MpiTransport::Impl {
  // A private duplicate of the caller's communicator: its own context (so two
  // field transfers' collectives can never be matched with each other) and its own
  // MPI_ERRORS_RETURN handler (so a failure is a Result, not an abort).
  MPI_Comm comm = MPI_COMM_NULL;
  int      rank = -1;
  int      size = 0;

  bool planned = false;

  // The planned spec's identity. prepare() compares against these; every
  // comparison is a scalar, so it neither allocates nor touches MPI.
  std::size_t p_elements = 0;
  int         p_components = 1;
  int         p_src_rank_set = -1;
  int         p_dst_rank_set = -1;
  std::size_t p_n_ids = 0;
  std::size_t p_n_src_ranks = 0;
  std::size_t p_n_dst_ranks = 0;
  std::size_t p_total = 0;  // elements * components: the `n` move() demands

  // The Alltoallv plan. Indexed by rank, in doubles.
  std::vector<int> sendcounts, sdispls, recvcounts, rdispls;
  // Slot k of the send buffer holds element `send_index[k]` of `src`; slot k
  // of the receive buffer belongs at element `recv_index[k]` of `dst`. Both
  // are in ascending peer-rank order, and within a peer in the DESTINATION's
  // id order -- which is what makes the sender's block and the receiver's
  // block agree element by element.
  std::vector<std::size_t> send_index, recv_index;
  std::vector<double>      send_buf, recv_buf;

  // Matches whose two ends are on THIS rank, which happens under partial
  // overlap. They never enter the Alltoallv -- its count arrays are indexed by
  // peer, and a self-entry would collide with a real peer's block -- so move()
  // copies them directly, source element to destination element.
  std::vector<std::pair<std::size_t, std::size_t>> local_copy;

  // Set when the plan came from build_plan_from_peers(). prepare() then
  // compares only the extents, because there is no TransportSpec id list to
  // compare against.
  bool p_from_peers = false;

  // transport_mpi_is_gpu_aware(), read once at construction so
  // is_gpu_aware() -- called from the Runner's hot loop, once per
  // Transport instruction per period -- is a field read, not a repeated
  // getenv()/extension call (docs/dev/device-resident-arcs.md Phase 3).
  const bool gpu_aware = transport_mpi_is_gpu_aware();

  ~Impl() {
    if (comm == MPI_COMM_NULL) return;
    int finalized = 0;
    // Freeing a communicator after MPI_Finalize is undefined; a static or
    // late-destroyed transport is a real possibility, so check.
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0) {
      MPI_Comm_free(&comm);
    }
    comm = MPI_COMM_NULL;
  }

  Impl() = default;
  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  /// MPI_Allreduce(MAX) of a local error code. Returns the agreed code.
  /// A failure of the Allreduce itself is reported as Err::Mpi, which every
  /// rank that saw the failure will also report -- and if only one rank saw
  /// it, the communicator is already unusable, which is not recoverable in
  /// any case.
  [[nodiscard]] Err agree(Err local, int* mpi_rc) noexcept {
    auto in = static_cast<int>(local);
    int  out = in;
    const int rc = MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MAX, comm);
    if (rc != MPI_SUCCESS) {
      *mpi_rc = rc;
      return Err::Mpi;
    }
    return static_cast<Err>(out);
  }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MpiTransport::MpiTransport() : impl_(std::make_unique<Impl>()) {}
MpiTransport::~MpiTransport() = default;
MpiTransport::MpiTransport(MpiTransport&&) noexcept = default;
MpiTransport& MpiTransport::operator=(MpiTransport&&) noexcept = default;

std::size_t MpiTransport::send_volume() const noexcept {
  return impl_ ? impl_->send_index.size() *
                     static_cast<std::size_t>(impl_->p_components)
               : 0;
}
std::size_t MpiTransport::recv_volume() const noexcept {
  return impl_ ? impl_->recv_index.size() *
                     static_cast<std::size_t>(impl_->p_components)
               : 0;
}

bool MpiTransport::is_gpu_aware() const noexcept {
  return impl_ != nullptr && impl_->gpu_aware;
}

// ---------------------------------------------------------------------------
// set_communicator
// ---------------------------------------------------------------------------

Result<void> MpiTransport::set_communicator(int fortran_comm_handle) {
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Failure{"E-TRANSPORT-COMM",
                 "set_communicator() on a moved-from MpiTransport"};
  }
  if (p->planned) {
    return Failure{"E-TRANSPORT-COMM",
                 "the communicator cannot be changed after build_plan(); the "
                 "displacement plan is expressed in that communicator's ranks"};
  }

  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return Failure{"E-TRANSPORT-MPIINIT",
                 "MPI is not initialised; MPI_Comm_f2c on a handle from an "
                 "uninitialised MPI is not meaningful"};
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized != 0) {
    return Failure{"E-TRANSPORT-MPIINIT",
                 "MPI has already been finalised"};
  }

  const MPI_Comm given =
      MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_comm_handle));
  if (given == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "Fortran communicator handle " +
                     std::to_string(fortran_comm_handle) +
                     " converts to MPI_COMM_NULL"};
  }

  // Release a previous duplicate first: MPI_Comm_free is collective, and so is
  // the MPI_Comm_dup below, so a second set_communicator is collective twice
  // over. Ordered free-then-dup so both are in the same order on every rank.
  if (p->comm != MPI_COMM_NULL) {
    const int rc = MPI_Comm_free(&p->comm);
    p->comm = MPI_COMM_NULL;
    if (rc != MPI_SUCCESS) {
      return Failure{"E-TRANSPORT-MPI",
                   "MPI_Comm_free on the previous duplicate failed: " +
                       mpi_error_text(rc)};
    }
  }

  MPI_Comm dup = MPI_COMM_NULL;
  int rc = MPI_Comm_dup(given, &dup);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-TRANSPORT-MPI",
                 "MPI_Comm_dup failed: " + mpi_error_text(rc)};
  }
  // From here on every MPI call on `dup` returns its status instead of
  // aborting the job, which is what makes "check every return code" possible.
  rc = MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN);
  if (rc != MPI_SUCCESS) {
    MPI_Comm_free(&dup);
    return Failure{"E-TRANSPORT-MPI",
                 "MPI_Comm_set_errhandler failed: " + mpi_error_text(rc)};
  }
  int rank = -1;
  int size = 0;
  rc = MPI_Comm_rank(dup, &rank);
  if (rc == MPI_SUCCESS) rc = MPI_Comm_size(dup, &size);
  if (rc != MPI_SUCCESS) {
    MPI_Comm_free(&dup);
    return Failure{"E-TRANSPORT-MPI",
                 "MPI_Comm_rank/size failed: " + mpi_error_text(rc)};
  }

  p->comm = dup;
  p->rank = rank;
  p->size = size;
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// build_plan
// ---------------------------------------------------------------------------

namespace {

/// The heap window between two collectives. A throw here cannot be turned into
/// an error return: this rank would skip the next collective and every peer
/// would block in it forever. Dying loudly is the only honest option.
[[noreturn]] void abort_job(MPI_Comm comm, const char* what) {
  // Between two collectives, where a throw would skip the next one and
  // hang every peer. No allocation, no formatting library, no locale.
  std::fprintf(stderr,  // NOLINT(fuse-raw-io)
               "FUSE MpiTransport: %s while building the redistribution plan; "
               "aborting the job because the surviving ranks would otherwise "
               "block in a collective this rank can no longer reach\n",
               what);
  std::fflush(stderr);
  MPI_Abort(comm, 70);
  std::abort();  // MPI_Abort is not marked noreturn
}

}  // namespace

Result<void> MpiTransport::build_plan(const TransportSpec& spec) {
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Failure{"E-TRANSPORT-COMM",
                 "build_plan() on a moved-from MpiTransport"};
  }
  if (p->comm == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "build_plan() before set_communicator(); there is no "
                 "communicator to redistribute over"};
  }

  const int  size = p->size;
  const int  rank = p->rank;
  const auto usize = static_cast<std::size_t>(size);

  p->planned = false;

  Err         err = Err::None;
  std::string detail;
  int         mpi_rc = MPI_SUCCESS;

  // -- Phase A: purely local validation, and every fixed-size allocation ----
  // Nothing here talks to MPI, and nothing here returns: a rank that fails
  // must still reach the Allreduce below.
  std::vector<char>     is_src, is_dst;
  std::vector<int>      counts, displs;
  std::vector<std::uint64_t> my_ids;

  const auto fail = [&](Err e, std::string d) {
    if (err == Err::None) {
      err = e;
      detail = std::move(d);
    }
  };

  try {
    is_src.assign(usize, 0);
    is_dst.assign(usize, 0);
    counts.assign(usize, 0);
    displs.assign(usize, 0);
    p->sendcounts.assign(usize, 0);
    p->sdispls.assign(usize, 0);
    p->recvcounts.assign(usize, 0);
    p->rdispls.assign(usize, 0);

    if (spec.components < 1) {
      fail(Err::Size, "components is " + std::to_string(spec.components) +
                          "; a field has at least one component");
    }
    if (spec.src_ranks.empty() || spec.dst_ranks.empty()) {
      fail(Err::Ranks,
           "the redistribution pattern needs a non-empty source rank set and "
           "a non-empty destination rank set; got " +
               std::to_string(spec.src_ranks.size()) + " and " +
               std::to_string(spec.dst_ranks.size()));
    }
    for (const int r : spec.src_ranks) {
      if (r < 0 || r >= size) {
        fail(Err::Ranks, "source rank " + std::to_string(r) +
                             " is outside [0, " + std::to_string(size) +
                             ") of the transport's communicator");
        continue;
      }
      if (is_src[static_cast<std::size_t>(r)] != 0) {
        fail(Err::Ranks,
             "rank " + std::to_string(r) + " appears twice in src_ranks");
      }
      is_src[static_cast<std::size_t>(r)] = 1;
    }
    for (const int r : spec.dst_ranks) {
      if (r < 0 || r >= size) {
        fail(Err::Ranks, "destination rank " + std::to_string(r) +
                             " is outside [0, " + std::to_string(size) +
                             ") of the transport's communicator");
        continue;
      }
      if (is_dst[static_cast<std::size_t>(r)] != 0) {
        fail(Err::Ranks,
             "rank " + std::to_string(r) + " appears twice in dst_ranks");
      }
      is_dst[static_cast<std::size_t>(r)] = 1;
    }
    for (std::size_t r = 0; r < usize; ++r) {
      if (is_src[r] != 0 && is_dst[r] != 0) {
        fail(Err::Ranks,
             "rank " + std::to_string(r) +
                 " is in both src_ranks and dst_ranks. TransportSpec carries "
                 "one global_ids list per rank, so a rank in both sets has no "
                 "way to state a source distribution and a different "
                 "destination distribution; use disjoint rank sets (spec "
                 "§4.1) and CopyTransport for the purely local part");
      }
    }

    if (spec.global_ids.size() != spec.elements) {
      fail(Err::Size,
           "global_ids holds " + std::to_string(spec.global_ids.size()) +
               " ids but elements is " + std::to_string(spec.elements) +
               "; the two describe the same list");
    }
    const bool in_a_set = err == Err::None &&
                          (is_src[static_cast<std::size_t>(rank)] != 0 ||
                           is_dst[static_cast<std::size_t>(rank)] != 0);
    if (err == Err::None && !in_a_set && spec.elements != 0) {
      fail(Err::Ranks,
           "rank " + std::to_string(rank) +
               " is in neither rank set yet declares " +
               std::to_string(spec.elements) +
               " elements, which nothing would ever send or receive");
    }

    // MPI counts are int. size_t arithmetic wraps in silence, so the product
    // is checked before it is formed, not after.
    const auto nc = static_cast<std::size_t>(std::max(spec.components, 1));
    if (spec.elements > kIntMax) {
      fail(Err::Size, "this rank holds " + std::to_string(spec.elements) +
                          " elements, more than the INT_MAX an MPI count can "
                          "express");
    } else if (nc != 0 && spec.elements > kIntMax / nc) {
      fail(Err::Size,
           "elements * components = " + std::to_string(spec.elements) + " * " +
               std::to_string(nc) +
               " exceeds INT_MAX, which no MPI count or displacement can "
               "express; split the field into several transports");
    }

    if (err == Err::None) {
      my_ids.resize(spec.elements);
      for (std::size_t i = 0; i < spec.elements; ++i) {
        my_ids[i] = static_cast<std::uint64_t>(spec.global_ids[i]);
      }
    }
  } catch (const std::exception& e) {
    // Before the first collective: no peer is waiting on this rank yet, but
    // they will be the moment they reach the Allreduce, so this is the same
    // trap as the mid-plan one.
    abort_job(p->comm, e.what());
  } catch (...) {
    abort_job(p->comm, "an unknown exception");
  }

  // -- Collective 1: agree on the local outcome ----------------------------
  {
    const Err agreed = p->agree(err, &mpi_rc);
    if (agreed != Err::None) {
      if (err != Err::None) return Failure{code_of(err), detail};
      if (agreed == Err::Mpi && mpi_rc != MPI_SUCCESS) {
        return Failure{"E-TRANSPORT-MPI",
                     "MPI_Allreduce failed while agreeing the plan: " +
                         mpi_error_text(mpi_rc)};
      }
      return Failure{code_of(agreed),
                   "another rank rejected the redistribution pattern (" +
                       std::string{code_of(agreed)} +
                       "); this rank's own spec was consistent"};
    }
  }

  // -- Collective 2: how many ids does each rank hold? ---------------------
  {
    auto      my_count = static_cast<int>(my_ids.size());
    const int rc = MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1,
                                 MPI_INT, p->comm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-TRANSPORT-MPI",
                   "MPI_Allgather of the id counts failed: " +
                       mpi_error_text(rc)};
    }
  }

  // -- Collective 2b: agree on components (R-30, REVIEW.md) ----------------
  // `sendcounts`/`recvcounts` below are each sized from THIS rank's own
  // `spec.components` (via `nc`); nothing about the id-matching above would
  // ever notice two ranks disagreeing on it, since ids carry no per-rank
  // component count. `build_plan_from_peers()`'s Phase B numbering check
  // catches this class of mismatch as a side effect of its own count
  // handshake; this, the older matched path, has none. A disagreement here
  // would size MPI_Alltoallv's count arrays inconsistently across ranks,
  // which is undefined behaviour per the MPI standard -- deferred through
  // the same `fail()`/`agree()` mechanism as every other outcome in this
  // function, so it is still reconciled at Collective 4 rather than left for
  // one rank to discover alone inside the actual data move.
  //
  // Review 2026-10-11 TR-10. This allocation sits strictly between
  // Collective 2 above and the Allgather just below, with nothing to catch
  // a bad_alloc here the way Phase A's own try/catch covers its own
  // allocations -- exactly the single window this file's own header
  // comment (point under "THE ONE WINDOW...") already names and answers by
  // aborting rather than letting an uncaught exception skip the
  // collectives every other rank is about to enter.
  std::vector<int> all_components;
  try {
    all_components.assign(usize, 0);
  } catch (const std::exception& e) {
    abort_job(p->comm, e.what());
  } catch (...) {
    abort_job(p->comm, "an unknown exception");
  }
  {
    const int my_components = spec.components;
    const int rc = MPI_Allgather(&my_components, 1, MPI_INT,
                                 all_components.data(), 1, MPI_INT, p->comm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-TRANSPORT-MPI",
                   "MPI_Allgather of components failed: " +
                       mpi_error_text(rc)};
    }
    // Opt-in trace, because this check has now twice not fired when reading
    // the code said it must (host runs 2026-09-20). Off unless
    // FUSE_TRANSPORT_DEBUG is set in the environment, so it costs a getenv
    // per build_plan() and nothing else.
    if (std::getenv("FUSE_TRANSPORT_DEBUG") != nullptr) {
      std::string line = "[build_plan rank " + std::to_string(rank) + "/" +
                         std::to_string(size) + "] err=" +
                         std::string{code_of(err)} + " components=[";
      for (std::size_t r = 0; r < usize; ++r) {
        line += (r ? "," : "") + std::to_string(all_components[r]);
      }
      line += "] participants=[";
      for (std::size_t r = 0; r < usize; ++r) {
        const bool part = is_src[r] != 0 || is_dst[r] != 0;
        line += (r ? "," : "") + std::string{part ? "1" : "0"};
      }
      line += "]\n";
      fmt::print(stderr, "{}", line);
    }

    // The rule itself is pure and lives in a header so the base build configuration can test it
    // (fuse/transport/rank_agreement.hpp); only the gather is MPI's. Computed
    // unconditionally so the trace below can report it even when an earlier
    // outcome already stands.
    const auto agreement =
        ::fuse::transport::check_component_agreement(all_components, is_src,
                                                     is_dst);
    if (std::getenv("FUSE_TRANSPORT_DEBUG") != nullptr) {
      // Sizes included deliberately: the rule clamps its loop to the shortest
      // of the three, so a short mask makes it agree vacuously and that is
      // indistinguishable from real agreement in any other output.
      fmt::print(stderr,
                 "[agreement rank {}] sizes comp={} src={} dst={} | "
                 "first=({},{}) disagree=({},{}) agreed={}\n",
                 rank, all_components.size(), is_src.size(), is_dst.size(),
                 agreement.first_rank, agreement.first_value,
                 agreement.disagreeing_rank, agreement.disagreeing_value,
                 agreement.agreed());
      std::fflush(stderr);
    }

    if (err == Err::None) {
      if (!agreement.agreed()) {
        // TR-10 (continued): the string concatenation below is itself an
        // allocation strictly between this Allgather and Collective 3; the
        // same reasoning as all_components above applies.
        try {
          fail(Err::PeerMatch,
               "rank " + std::to_string(agreement.disagreeing_rank) +
                   " called build_plan() with components = " +
                   std::to_string(agreement.disagreeing_value) + ", but rank " +
                   std::to_string(agreement.first_rank) +
                   " (also in the src or dst rank set) called it with " +
                   std::to_string(agreement.first_value) +
                   "; every rank taking part in this redistribution must agree "
                   "on the field's component count");
        } catch (const std::exception& e) {
          abort_job(p->comm, e.what());
        } catch (...) {
          abort_job(p->comm, "an unknown exception");
        }
      }
    }
  }

  // Derived identically on every rank from data every rank now has, so a
  // return here is a return everywhere.
  std::size_t total_ids = 0;
  for (std::size_t r = 0; r < usize; ++r) {
    displs[r] = static_cast<int>(total_ids);
    total_ids += static_cast<std::size_t>(counts[r]);
    if (total_ids > kIntMax) {
      return Failure{"E-TRANSPORT-SIZE",
                   "the union of the two rank sets holds more than INT_MAX "
                   "entities, which MPI_Allgatherv cannot address"};
    }
  }

  std::vector<std::uint64_t> all_ids;
  try {
    all_ids.resize(total_ids == 0 ? 1 : total_ids);
  } catch (const std::exception& e) {
    abort_job(p->comm, e.what());
  } catch (...) {
    abort_job(p->comm, "an unknown exception");
  }

  // -- Collective 3: the ids themselves ------------------------------------
  {
    const int rc = MPI_Allgatherv(
        my_ids.empty() ? nullptr : my_ids.data(),
        static_cast<int>(my_ids.size()), MPI_UINT64_T, all_ids.data(),
        counts.data(), displs.data(), MPI_UINT64_T, p->comm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-TRANSPORT-MPI",
                   "MPI_Allgatherv of the global ids failed: " +
                       mpi_error_text(rc)};
    }
  }

  // -- Phase D: the matching -----------------------------------------------
  // Deliberately performed in full on EVERY rank rather than only where it is
  // needed. The inputs are identical everywhere, so the outcome is identical
  // everywhere: "no source holds id g" cannot be a one-rank error return while
  // the others walk into MPI_Alltoallv.
  //
  // `err` IS NOT RESET HERE, and must not be. It is one monotonic outcome for
  // the whole function -- `fail()` keeps the first and ignores later ones --
  // and the loops below read it only to stop early. It used to be cleared at
  // this point, back when every outcome before it had already been agreed and
  // returned at Collective 1, so clearing it was harmless. Collective 2b (the
  // components agreement, R-30) then added an outcome recorded AFTER Collective
  // 1 and agreed only at Collective 4 -- on the far side of the clear. The
  // effect was that R-30's check ran, computed the right answer on every rank,
  // and had it erased two phases later: `build_plan()` returned success on
  // ranks that had just rejected. Found 2026-09-21 by tracing the outcome at
  // both ends (FUSE_TRANSPORT_DEBUG) after reading the path five times without
  // seeing it; `MpiTransport.BuildPlanRejectsDisagreeingComponentsAcrossRanks`
  // could not pass until this line went away.
  //
  // With the outcome kept, a rank that has already rejected skips the matching
  // below -- every rank does, since the outcome is derived from the gathered
  // array and is identical everywhere -- and Phase D holds no collective, so
  // there is nothing for a skipping rank to desynchronise.
  try {
    std::unordered_map<std::uint64_t, int> owner;
    owner.reserve(total_ids);
    for (std::size_t s = 0; s < usize && err == Err::None; ++s) {
      if (is_src[s] == 0) continue;
      const auto base = static_cast<std::size_t>(displs[s]);
      const auto n = static_cast<std::size_t>(counts[s]);
      for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t g = all_ids[base + i];
        const auto [it, inserted] = owner.emplace(g, static_cast<int>(s));
        if (!inserted) {
          // A ghost or a duplicate on the SOURCE side. Picking a winner would
          // make the value delivered depend on which copy happened to be
          // lower-numbered, and the two copies need not agree.
          fail(Err::Match,
               "global id " + std::to_string(g) +
                   " is held by source rank " + std::to_string(it->second) +
                   " and again by source rank " + std::to_string(s) +
                   "; pass owned entities only, or the value transported "
                   "would depend on which copy won");
          break;
        }
      }
    }

    if (err == Err::None) {
      for (std::size_t d = 0; d < usize && err == Err::None; ++d) {
        if (is_dst[d] == 0) continue;
        const auto base = static_cast<std::size_t>(displs[d]);
        const auto n = static_cast<std::size_t>(counts[d]);
        for (std::size_t i = 0; i < n; ++i) {
          if (owner.find(all_ids[base + i]) == owner.end()) {
            fail(Err::Match,
                 "destination rank " + std::to_string(d) + " wants global id " +
                     std::to_string(all_ids[base + i]) +
                     " but no source rank holds it");
            break;
          }
        }
      }
    }

    if (err == Err::None) {
      const auto nc = static_cast<std::size_t>(std::max(spec.components, 1));
      const auto urank = static_cast<std::size_t>(rank);

      // Where each of my own ids sits in my local array.
      std::unordered_map<std::uint64_t, std::size_t> mine;
      if (is_src[urank] != 0) {
        mine.reserve(my_ids.size());
        for (std::size_t i = 0; i < my_ids.size(); ++i) {
          if (!mine.emplace(my_ids[i], i).second) {
            fail(Err::Match, "rank " + std::to_string(rank) +
                                 " lists global id " +
                                 std::to_string(my_ids[i]) +
                                 " twice in its own global_ids");
            break;
          }
        }
      }

      p->send_index.clear();
      p->recv_index.clear();

      // Sends: peers in ascending rank order (MPI_Alltoallv indexes its count
      // arrays by rank), and within a peer in THAT peer's id order.
      if (err == Err::None && is_src[urank] != 0) {
        for (std::size_t d = 0; d < usize; ++d) {
          if (is_dst[d] == 0) continue;
          const std::size_t start = p->send_index.size();
          const auto base = static_cast<std::size_t>(displs[d]);
          const auto n = static_cast<std::size_t>(counts[d]);
          for (std::size_t i = 0; i < n; ++i) {
            const auto it = mine.find(all_ids[base + i]);
            if (it != mine.end() && owner.at(all_ids[base + i]) == rank) {
              p->send_index.push_back(it->second);
            }
          }
          p->sendcounts[d] =
              static_cast<int>((p->send_index.size() - start) * nc);
        }
      }

      // Receives: the mirror image. Peers ascending, and within a peer in MY
      // id order -- which is the same order the sender walked, because the
      // sender walked my id list too.
      if (err == Err::None && is_dst[urank] != 0) {
        for (std::size_t s = 0; s < usize; ++s) {
          if (is_src[s] == 0) continue;
          const std::size_t start = p->recv_index.size();
          for (std::size_t i = 0; i < my_ids.size(); ++i) {
            if (owner.at(my_ids[i]) == static_cast<int>(s)) {
              p->recv_index.push_back(i);
            }
          }
          p->recvcounts[s] =
              static_cast<int>((p->recv_index.size() - start) * nc);
        }
      }

      // Displacements, and the INT_MAX guard on the totals. Every count above
      // was bounded by elements*components, already checked; the sums are not.
      std::size_t send_total = 0;
      std::size_t recv_total = 0;
      for (std::size_t r = 0; r < usize && err == Err::None; ++r) {
        p->sdispls[r] = static_cast<int>(send_total);
        p->rdispls[r] = static_cast<int>(recv_total);
        send_total += static_cast<std::size_t>(p->sendcounts[r]);
        recv_total += static_cast<std::size_t>(p->recvcounts[r]);
        if (send_total > kIntMax || recv_total > kIntMax) {
          fail(Err::Size,
               "the Alltoallv displacement for rank " + std::to_string(r) +
                   " exceeds INT_MAX; split the field into several transports");
        }
      }

      if (err == Err::None) {
        // Sized once, here, and never again: this is what makes move()
        // allocation-free. The +1 keeps data() non-null for a rank that sends
        // or receives nothing, which several MPI implementations dislike even
        // with a zero count.
        p->send_buf.assign(send_total == 0 ? 1 : send_total, 0.0);
        p->recv_buf.assign(recv_total == 0 ? 1 : recv_total, 0.0);
      }
    }
  } catch (const std::bad_alloc&) {
    fail(Err::Alloc, "out of memory while building the redistribution plan");
  } catch (const std::exception& e) {
    fail(Err::Alloc, std::string{"exception while building the plan: "} +
                         e.what());
  } catch (...) {
    fail(Err::Alloc, "unknown exception while building the plan");
  }

  // -- Collective 4: agree again -------------------------------------------
  // The matching is deterministic, but a heap failure on one rank is not. This
  // is after the last collective the plan needs, so every rank reaches it.
  {
    const Err agreed = p->agree(err, &mpi_rc);
    if (std::getenv("FUSE_TRANSPORT_DEBUG") != nullptr) {
      // The pair that localises a lost outcome: `local` is what this rank
      // decided, `agreed` what the reduction returned. A rejection visible here
      // that still returns success can only be lost between the two.
      fmt::print(stderr, "[outcome rank {}] local={} agreed={}\n", rank,
                 static_cast<int>(err), static_cast<int>(agreed));
      std::fflush(stderr);
    }
    if (agreed != Err::None) {
      if (err != Err::None) return Failure{code_of(err), detail};
      if (agreed == Err::Mpi && mpi_rc != MPI_SUCCESS) {
        return Failure{"E-TRANSPORT-MPI",
                     "MPI_Allreduce failed while agreeing the plan: " +
                         mpi_error_text(mpi_rc)};
      }
      return Failure{code_of(agreed),
                   "another rank failed to build the redistribution plan (" +
                       std::string{code_of(agreed)} + ")"};
    }
  }

  p->p_elements = spec.elements;
  p->p_components = spec.components;
  p->p_src_rank_set = spec.src_rank_set;
  p->p_dst_rank_set = spec.dst_rank_set;
  p->p_n_ids = spec.global_ids.size();
  p->p_n_src_ranks = spec.src_ranks.size();
  p->p_n_dst_ranks = spec.dst_ranks.size();
  p->p_total =
      spec.elements * static_cast<std::size_t>(std::max(spec.components, 1));
  p->planned = true;
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// build_plan_from_peers -- the same plan, from a rank graph's peer-wise form
// ---------------------------------------------------------------------------
//
// No matching happens here. `send_plan()`/`recv_plan()` already hold exactly
// what the Alltoallv count arrays need, peer by peer in ascending rank, which
// is why closing partial overlap needed a transport path and not a matching
// redesign.

Result<void> MpiTransport::build_plan_from_peers(
    const std::vector<CommGraphPeer>& send,
    const std::vector<CommGraphPeer>& recv,
    const std::vector<LocalCopyPair>& local, std::size_t elements,
    int components) {
  Impl* p = impl_.get();
  if (p == nullptr || p->comm == MPI_COMM_NULL) {
    return Failure{"E-TRANSPORT-COMM",
                 "build_plan_from_peers() before set_communicator()"};
  }

  const int  size  = p->size;
  const auto usize = static_cast<std::size_t>(size);
  p->planned = false;

  Err         err = Err::None;
  std::string detail;
  int         mpi_rc = MPI_SUCCESS;
  const auto  fail = [&](Err e, std::string d) {
    if (err == Err::None) { err = e; detail = std::move(d); }
  };

  const auto        nc    = static_cast<std::size_t>(std::max(components, 1));
  const std::size_t total = elements * nc;

  // -- Phase A: local validation and every allocation ----------------------
  // Nothing here returns. A rank that fails must still reach the agreement
  // below, or it leaves its partners waiting in a collective.
  try {
    if (components <= 0) {
      fail(Err::Size,
           "components must be positive, got " + std::to_string(components));
    }

    // Peers ascending and in range. Ascending is not cosmetic: the count
    // arrays are indexed by rank and both sides walk their index vectors in
    // that order, so an out-of-order peer list pairs a sender's block with the
    // wrong receiver's block -- wrong values, no error.
    const auto check_peers = [&](const std::vector<CommGraphPeer>& peers,
                                 const char* which) {
      int previous = -1;
      for (const CommGraphPeer& peer : peers) {
        if (peer.base_rank < 0 || peer.base_rank >= size) {
          fail(Err::Ranks,
               std::string{which} + " plan names rank " +
                   std::to_string(peer.base_rank) +
                   ", outside this transport's communicator of size " +
                   std::to_string(size) +
                   "; set_communicator() must be given the communicator the "
                   "rank graph was built over");
          return;
        }
        if (peer.base_rank <= previous) {
          fail(Err::Match,
               std::string{which} + " plan is not in ascending rank order (" +
                   std::to_string(previous) + " then " +
                   std::to_string(peer.base_rank) + ")");
          return;
        }
        previous = peer.base_rank;
        for (const std::size_t idx : peer.local_indices) {
          if (idx >= elements) {
            fail(Err::Size,
                 std::string{which} + " plan for rank " +
                     std::to_string(peer.base_rank) + " names local element " +
                     std::to_string(idx) + ", but this rank holds " +
                     std::to_string(elements));
            return;
          }
        }
      }
    };
    check_peers(send, "send");
    check_peers(recv, "recv");

    p->sendcounts.assign(usize, 0);
    p->recvcounts.assign(usize, 0);
    p->sdispls.assign(usize, 0);
    p->rdispls.assign(usize, 0);
    p->send_index.clear();
    p->recv_index.clear();
    p->local_copy.clear();

    if (err == Err::None) {
      for (const CommGraphPeer& peer : send) {
        for (const std::size_t idx : peer.local_indices) {
          p->send_index.push_back(idx);
        }
        p->sendcounts[static_cast<std::size_t>(peer.base_rank)] =
            static_cast<int>(peer.local_indices.size() * nc);
      }
      for (const CommGraphPeer& peer : recv) {
        for (const std::size_t idx : peer.local_indices) {
          p->recv_index.push_back(idx);
        }
        p->recvcounts[static_cast<std::size_t>(peer.base_rank)] =
            static_cast<int>(peer.local_indices.size() * nc);
      }

      // Same-rank matches, range-checked here because an out-of-range index
      // would otherwise be a write past the field inside move().
      for (const LocalCopyPair& pair : local) {
        if (pair.source_local_index >= elements ||
            pair.dest_local_index >= elements) {
          fail(Err::Size,
               "a same-rank match names local elements " +
                   std::to_string(pair.source_local_index) + " -> " +
                   std::to_string(pair.dest_local_index) +
                   ", but this rank holds " + std::to_string(elements));
          break;
        }
        p->local_copy.emplace_back(pair.source_local_index,
                                   pair.dest_local_index);
      }
    }

    // Displacements, with build_plan()'s INT_MAX guard: the individual counts
    // are bounded by elements*components, the running sums are not.
    std::size_t send_total = 0;
    std::size_t recv_total = 0;
    for (std::size_t r = 0; r < usize && err == Err::None; ++r) {
      p->sdispls[r] = static_cast<int>(send_total);
      p->rdispls[r] = static_cast<int>(recv_total);
      send_total += static_cast<std::size_t>(p->sendcounts[r]);
      recv_total += static_cast<std::size_t>(p->recvcounts[r]);
      if (send_total > kIntMax || recv_total > kIntMax) {
        fail(Err::Size,
             "the Alltoallv displacement for rank " + std::to_string(r) +
                 " exceeds INT_MAX; split the field into several transports");
      }
    }
    if (err == Err::None) {
      // Sized once, here: this is what keeps move() allocation-free. The +1
      // keeps data() non-null for a rank that sends or receives nothing.
      p->send_buf.assign(send_total == 0 ? 1 : send_total, 0.0);
      p->recv_buf.assign(recv_total == 0 ? 1 : recv_total, 0.0);
    }
  } catch (const std::bad_alloc&) {
    fail(Err::Alloc, "out of memory while building the redistribution plan");
  } catch (const std::exception& e) {
    fail(Err::Alloc,
         std::string{"exception while building the plan: "} + e.what());
  } catch (...) {
    fail(Err::Alloc, "unknown exception while building the plan");
  }

  // -- Phase B: the numbering check, and the reason for this extra collective
  // Each rank tells every other how many doubles it intends to send it; each
  // then requires that to equal what it expects to receive. If
  // set_communicator() was given a different communicator than the rank graph
  // was built over, the two rank numberings differ and this disagrees. That is
  // the difference between an error at setup and a field delivered to the
  // wrong rank for the whole run.
  //
  // Issued on every rank, including one whose local Phase A already failed:
  // `p->sendcounts` was zero-filled by the unconditional `assign(usize, 0)`
  // above, so a failed rank truthfully reports "nothing to send" instead of
  // leaving its peers waiting inside this Alltoall while it goes straight to
  // the agreement below.
  {
    // Review 2026-10-11 TR-10. `expected`'s allocation sits strictly before
    // the Alltoall every rank now reaches unconditionally (TR-1 above); an
    // uncaught bad_alloc here would keep THIS rank from ever reaching that
    // collective while every peer already has, and the fail()-string
    // concatenation below sits strictly between that Alltoall and the final
    // agreement a few lines down -- both are the same "between two
    // collectives" window this file's own header comment already names and
    // answers by aborting rather than letting an uncaught exception skip a
    // collective a peer is already in.
    std::vector<int> expected;
    try {
      expected.assign(usize, 0);
    } catch (const std::exception& e) {
      abort_job(p->comm, e.what());
    } catch (...) {
      abort_job(p->comm, "an unknown exception");
    }
    mpi_rc = MPI_Alltoall(p->sendcounts.data(), 1, MPI_INT, expected.data(), 1,
                          MPI_INT, p->comm);
    if (err == Err::None) {
      if (mpi_rc != MPI_SUCCESS) {
        fail(Err::Mpi, "MPI_Alltoall failed while checking the peer plan: " +
                           mpi_error_text(mpi_rc));
      } else {
        try {
          for (std::size_t r = 0; r < usize; ++r) {
            if (expected[r] != p->recvcounts[r]) {
              fail(Err::PeerMatch,
                   "rank " + std::to_string(r) + " intends to send " +
                       std::to_string(expected[r]) +
                       " doubles to this rank, which expects " +
                       std::to_string(p->recvcounts[r]) +
                       ". The two halves of the plan disagree: either "
                       "set_communicator() was given a different communicator "
                       "than the rank graph was built over, or the two sides "
                       "came from different graphs");
              break;
            }
          }
        } catch (const std::exception& e) {
          abort_job(p->comm, e.what());
        } catch (...) {
          abort_job(p->comm, "an unknown exception");
        }
      }
    }
  }

  // -- Agree, so a failure is the same failure on every rank ---------------
  {
    const Err agreed = p->agree(err, &mpi_rc);
    if (agreed != Err::None) {
      if (err != Err::None) return Failure{code_of(err), detail};
      if (agreed == Err::Mpi && mpi_rc != MPI_SUCCESS) {
        return Failure{"E-TRANSPORT-MPI",
                     "MPI_Allreduce failed while agreeing the plan: " +
                         mpi_error_text(mpi_rc)};
      }
      return Failure{code_of(agreed),
                   "another rank failed to build the redistribution plan (" +
                       std::string{code_of(agreed)} + ")"};
    }
  }

  p->p_elements   = elements;
  p->p_components = components;
  p->p_total      = total;
  p->p_from_peers = true;
  // No spec identity to record: prepare() checks the extents only for a
  // peer-built plan, and the fields below are what it compares.
  p->p_src_rank_set = -1;
  p->p_dst_rank_set = -1;
  p->p_n_ids        = 0;
  p->p_n_src_ranks  = 0;
  p->p_n_dst_ranks  = 0;
  p->planned        = true;
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// prepare -- the hot loop's call
// ---------------------------------------------------------------------------
//
// Scalar comparisons only. No allocation, no MPI, no matching, no buffer
// sizing: `Runner::Impl::execute` calls this once per Transport instruction
// per coupling cycle, inside the loop Task 4.6 proved allocation-free.
Result<void> MpiTransport::prepare(const TransportSpec& spec) {
  const Impl* p = impl_.get();
  if (p == nullptr || !p->planned) {
    return Failure{"E-TRANSPORT-UNPLANNED",
                 "prepare() before a successful build_plan(); the "
                 "redistribution plan is collective and cannot be built here, "
                 "inside the hot loop"};
  }
  // A peer-built plan has no TransportSpec behind it, so only the extents are
  // comparable. They are also the only fields move() depends on: the rank-set
  // ids and id-list lengths exist to catch a re-spec of a MATCHED plan.
  const bool respec =
      p->p_from_peers
          ? (spec.elements != p->p_elements ||
             spec.components != p->p_components)
          : (spec.elements != p->p_elements ||
             spec.components != p->p_components ||
             spec.src_rank_set != p->p_src_rank_set ||
             spec.dst_rank_set != p->p_dst_rank_set ||
             spec.global_ids.size() != p->p_n_ids ||
             spec.src_ranks.size() != p->p_n_src_ranks ||
             spec.dst_ranks.size() != p->p_n_dst_ranks);
  if (respec) {
    return Failure{"E-TRANSPORT-RESPEC",
                 "prepare() was given a different TransportSpec than "
                 "build_plan() planned; rebuilding the plan is collective and "
                 "must not happen inside the loop"};
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------------
// move
// ---------------------------------------------------------------------------

Result<void> MpiTransport::move(const double* src, double* dst,
                                std::size_t n) {
  Impl* p = impl_.get();
  if (p == nullptr || !p->planned) {
    return Failure{"E-TRANSPORT-UNPLANNED",
                 "move() before a successful build_plan()"};
  }
  if (n != p->p_total) {
    return Failure{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) + " doubles, but the plan for "
                 "this rank covers exactly " + std::to_string(p->p_total) +
                     " (elements * components)"};
  }
  // A rank owning zero elements of a mesh entity set is ordinary in parallel, and
  // `std::vector<double>{}.data()` is permitted to be null. Null is therefore
  // accepted exactly when there is nothing to address.
  if (n != 0 && (src == nullptr || dst == nullptr)) {
    return Failure{"E-TRANSPORT-NULL",
                 "mpi transport given a null pointer for " +
                     std::to_string(n) + " doubles"};
  }

  const auto nc = static_cast<std::size_t>(std::max(p->p_components, 1));

  // Pack. Nothing is allocated: send_buf was sized by build_plan().
  double* sb = p->send_buf.data();
  for (std::size_t k = 0; k < p->send_index.size(); ++k) {
    std::memcpy(sb + k * nc, src + p->send_index[k] * nc, nc * sizeof(double));
  }

  const int rc = MPI_Alltoallv(sb, p->sendcounts.data(), p->sdispls.data(),
                               MPI_DOUBLE, p->recv_buf.data(),
                               p->recvcounts.data(), p->rdispls.data(),
                               MPI_DOUBLE, p->comm);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-TRANSPORT-MPI",
                 "MPI_Alltoallv failed: " + mpi_error_text(rc)};
  }

  // Unpack.
  const double* rb = p->recv_buf.data();
  for (std::size_t k = 0; k < p->recv_index.size(); ++k) {
    std::memcpy(dst + p->recv_index[k] * nc, rb + k * nc, nc * sizeof(double));
  }

  // Same-rank matches, under partial overlap. Done after the unpack so that a
  // destination element fed by both a peer and a local match ends up with the
  // local value -- which cannot happen, since a match has one source, but the
  // order is fixed deliberately rather than left to whichever loop ran last.
  // Empty for every disjoint plan, which is why the ordinary path pays nothing.
  for (const auto& [from, to] : p->local_copy) {
    std::memcpy(dst + to * nc, src + from * nc, nc * sizeof(double));
  }
  return Result<void>::success();
}

}  // namespace fuse
