// Task 5.4: n->m redistribution between two disjoint rank sets (spec §4.1).
//
// THE SHAPE OF THIS FILE IS DICTATED BY TWO CONSTRAINTS THAT PULL APART.
//
// 1. Task 4.6's hot loop. `Runner::Impl::execute` calls `prepare()` and then
//    `move()` inside the period loop, and that loop is proved allocation-free
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

#include "soma/transport/transport.hpp"

#include <mpi.h>

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

namespace soma {
namespace {

constexpr std::size_t kIntMax =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "global ids travel as MPI_UINT64_T");

/// Codes the two agreement Allreduces carry. Ordered so that MPI_MAX picks a
/// deterministic winner when different ranks fail differently; the message is
/// this rank's own when it failed, and a pointer to the collective verdict
/// otherwise.
enum class Err : int {
  None = 0,
  Ranks = 1,
  Size = 2,
  Match = 3,
  Mpi = 4,
  Alloc = 5,
};

[[nodiscard]] const char* code_of(Err e) noexcept {
  switch (e) {
    case Err::None:  return "";
    case Err::Ranks: return "E-TRANSPORT-RANKS";
    case Err::Size:  return "E-TRANSPORT-SIZE";
    case Err::Match: return "E-TRANSPORT-MATCH";
    case Err::Mpi:   return "E-TRANSPORT-MPI";
    case Err::Alloc: return "E-TRANSPORT-ALLOC";
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
  // channels' collectives can never be matched with each other) and its own
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

// ---------------------------------------------------------------------------
// set_communicator
// ---------------------------------------------------------------------------

Result<void> MpiTransport::set_communicator(int fortran_comm_handle) {
  Impl* p = impl_.get();
  if (p == nullptr) {
    return Error{"E-TRANSPORT-COMM",
                 "set_communicator() on a moved-from MpiTransport"};
  }
  if (p->planned) {
    return Error{"E-TRANSPORT-COMM",
                 "the communicator cannot be changed after build_plan(); the "
                 "displacement plan is expressed in that communicator's ranks"};
  }

  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return Error{"E-TRANSPORT-MPIINIT",
                 "MPI is not initialised; MPI_Comm_f2c on a handle from an "
                 "uninitialised MPI is not meaningful"};
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized != 0) {
    return Error{"E-TRANSPORT-MPIINIT",
                 "MPI has already been finalised"};
  }

  const MPI_Comm given =
      MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_comm_handle));
  if (given == MPI_COMM_NULL) {
    return Error{"E-TRANSPORT-COMM",
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
      return Error{"E-TRANSPORT-MPI",
                   "MPI_Comm_free on the previous duplicate failed: " +
                       mpi_error_text(rc)};
    }
  }

  MPI_Comm dup = MPI_COMM_NULL;
  int rc = MPI_Comm_dup(given, &dup);
  if (rc != MPI_SUCCESS) {
    return Error{"E-TRANSPORT-MPI",
                 "MPI_Comm_dup failed: " + mpi_error_text(rc)};
  }
  // From here on every MPI call on `dup` returns its status instead of
  // aborting the job, which is what makes "check every return code" possible.
  rc = MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN);
  if (rc != MPI_SUCCESS) {
    MPI_Comm_free(&dup);
    return Error{"E-TRANSPORT-MPI",
                 "MPI_Comm_set_errhandler failed: " + mpi_error_text(rc)};
  }
  int rank = -1;
  int size = 0;
  rc = MPI_Comm_rank(dup, &rank);
  if (rc == MPI_SUCCESS) rc = MPI_Comm_size(dup, &size);
  if (rc != MPI_SUCCESS) {
    MPI_Comm_free(&dup);
    return Error{"E-TRANSPORT-MPI",
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
  std::fprintf(stderr,
               "soma MpiTransport: %s while building the redistribution plan; "
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
    return Error{"E-TRANSPORT-COMM",
                 "build_plan() on a moved-from MpiTransport"};
  }
  if (p->comm == MPI_COMM_NULL) {
    return Error{"E-TRANSPORT-COMM",
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

  // -- Collective 1: agree on the local verdict ----------------------------
  {
    const Err agreed = p->agree(err, &mpi_rc);
    if (agreed != Err::None) {
      if (err != Err::None) return Error{code_of(err), detail};
      if (agreed == Err::Mpi && mpi_rc != MPI_SUCCESS) {
        return Error{"E-TRANSPORT-MPI",
                     "MPI_Allreduce failed while agreeing the plan: " +
                         mpi_error_text(mpi_rc)};
      }
      return Error{code_of(agreed),
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
      return Error{"E-TRANSPORT-MPI",
                   "MPI_Allgather of the id counts failed: " +
                       mpi_error_text(rc)};
    }
  }

  // Derived identically on every rank from data every rank now has, so a
  // return here is a return everywhere.
  std::size_t total_ids = 0;
  for (std::size_t r = 0; r < usize; ++r) {
    displs[r] = static_cast<int>(total_ids);
    total_ids += static_cast<std::size_t>(counts[r]);
    if (total_ids > kIntMax) {
      return Error{"E-TRANSPORT-SIZE",
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
      return Error{"E-TRANSPORT-MPI",
                   "MPI_Allgatherv of the global ids failed: " +
                       mpi_error_text(rc)};
    }
  }

  // -- Phase D: the matching -----------------------------------------------
  // Deliberately performed in full on EVERY rank rather than only where it is
  // needed. The inputs are identical everywhere, so the verdict is identical
  // everywhere: "no source holds id g" cannot be a one-rank error return while
  // the others walk into MPI_Alltoallv.
  err = Err::None;
  detail.clear();
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
    if (agreed != Err::None) {
      if (err != Err::None) return Error{code_of(err), detail};
      if (agreed == Err::Mpi && mpi_rc != MPI_SUCCESS) {
        return Error{"E-TRANSPORT-MPI",
                     "MPI_Allreduce failed while agreeing the plan: " +
                         mpi_error_text(mpi_rc)};
      }
      return Error{code_of(agreed),
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
// prepare -- the hot loop's call
// ---------------------------------------------------------------------------
//
// Scalar comparisons only. No allocation, no MPI, no matching, no buffer
// sizing: `Runner::Impl::execute` calls this once per Transport instruction
// per period, inside the loop Task 4.6 proved allocation-free.
Result<void> MpiTransport::prepare(const TransportSpec& spec) {
  const Impl* p = impl_.get();
  if (p == nullptr || !p->planned) {
    return Error{"E-TRANSPORT-UNPLANNED",
                 "prepare() before a successful build_plan(); the "
                 "redistribution plan is collective and cannot be built here, "
                 "inside the hot loop"};
  }
  if (spec.elements != p->p_elements || spec.components != p->p_components ||
      spec.src_rank_set != p->p_src_rank_set ||
      spec.dst_rank_set != p->p_dst_rank_set ||
      spec.global_ids.size() != p->p_n_ids ||
      spec.src_ranks.size() != p->p_n_src_ranks ||
      spec.dst_ranks.size() != p->p_n_dst_ranks) {
    return Error{"E-TRANSPORT-RESPEC",
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
    return Error{"E-TRANSPORT-UNPLANNED",
                 "move() before a successful build_plan()"};
  }
  if (n != p->p_total) {
    return Error{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) + " doubles, but the plan for "
                 "this rank covers exactly " + std::to_string(p->p_total) +
                     " (elements * components)"};
  }
  // A rank owning zero elements of a region is ordinary in parallel, and
  // `std::vector<double>{}.data()` is permitted to be null. Null is therefore
  // accepted exactly when there is nothing to address.
  if (n != 0 && (src == nullptr || dst == nullptr)) {
    return Error{"E-TRANSPORT-NULL",
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
    return Error{"E-TRANSPORT-MPI",
                 "MPI_Alltoallv failed: " + mpi_error_text(rc)};
  }

  // Unpack.
  const double* rb = p->recv_buf.data();
  for (std::size_t k = 0; k < p->recv_index.size(); ++k) {
    std::memcpy(dst + p->recv_index[k] * nc, rb + k * nc, nc * sizeof(double));
  }
  return Result<void>::success();
}

}  // namespace soma
