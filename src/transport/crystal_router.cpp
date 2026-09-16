// The Crystal Router. See crystal_router.hpp for the algorithm and for why
// this is a private header.
//
// DEADLOCK DISCIPLINE, the same one stated at the top of mpi_transport.cpp and
// applied here rather than reinvented:
//
//   - The only local validation is the destination-rank check, and it happens
//     BEFORE any point-to-point call and is agreed with an MPI_Allreduce(MAX)
//     over the whole communicator. Either every rank returns the error or none
//     does; a rank that fails still reaches that Allreduce.
//   - After that Allreduce every rank executes a schedule derived purely from
//     `(lo, hi, rank, size)`. It is therefore identical on every rank by
//     construction and cannot diverge on data.
//   - Each round is one MPI_Isend plus one or two Probe/Recv pairs. The Isend
//     is non-blocking, so no pair can block on each other.
//   - The two failure windows that cannot be agreed -- an MPI error and a heap
//     failure, both strictly between two point-to-point calls -- are handled
//     the way mpi_transport.cpp handles its equivalent: MPI_Abort with a
//     message. A job that dies loudly, not one that hangs.

#include "fuse/base/config.hpp"  // defines the FUSE_HAVE_* macros read below
#include "crystal_router.hpp"

#if FUSE_HAVE_MPI

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <string>
#include <utility>


namespace fuse::detail {
namespace {

constexpr std::size_t kIntMax =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

/// Wire format of one envelope: a 4-byte destination rank, an 8-byte payload
/// length, then the payload. Fixed-width and memcpy'd, so it does not depend
/// on the host's struct layout; the two sides of an exchange are the same
/// binary in one MPI job, so no endianness conversion is needed or attempted.
constexpr std::size_t kHeaderBytes = sizeof(std::int32_t) + sizeof(std::uint64_t);

[[noreturn]] void abort_job(MPI_Comm comm, const char* what) {
  // Between two collectives, where a throw would skip the next one and
  // hang every peer. No allocation, no formatting library, no locale.
  std::fprintf(stderr,  // NOLINT(fuse-raw-io)
               "FUSE: CrystalRouter cannot continue between two collectives "
               "(%s). Aborting rather than deadlocking the job.\n",
               what);
  std::fflush(stderr);
  MPI_Abort(comm, 1);
  std::abort();
}

[[nodiscard]] std::string mpi_error_text(int rc) {
  char buf[MPI_MAX_ERROR_STRING];
  int  len = 0;
  if (MPI_Error_string(rc, buf, &len) != MPI_SUCCESS || len <= 0) {
    return "MPI error code " + std::to_string(rc);
  }
  return std::string(buf, static_cast<std::size_t>(len));
}

void append_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
  const std::size_t at = b.size();
  b.resize(at + sizeof(v));
  std::memcpy(b.data() + at, &v, sizeof(v));
}

void append_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
  const std::size_t at = b.size();
  b.resize(at + sizeof(v));
  std::memcpy(b.data() + at, &v, sizeof(v));
}

}  // namespace

CrystalRouter::CrystalRouter(MPI_Comm comm) : comm_(comm) {
  if (comm_ != MPI_COMM_NULL) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }
}

Result<std::vector<Envelope>> CrystalRouter::route(
    std::vector<Envelope> outgoing) {
  last_rounds_ = 0;

  if (comm_ == MPI_COMM_NULL) {
    return Failure{"E-CROUTER-COMM",
                 "CrystalRouter::route() on a null communicator"};
  }

  // -- Local validation, agreed before anything is sent --------------------
  // 0 = ok, 1 = a destination is out of range, 2 = this rank's outgoing
  // traffic cannot be expressed as an MPI byte count. MAX picks a
  // deterministic winner if ranks fail differently.
  int         err = 0;
  std::string detail;
  std::size_t total_bytes = 0;
  for (const Envelope& e : outgoing) {
    if (e.dest_rank < 0 || e.dest_rank >= size_) {
      if (err == 0) {
        err = 1;
        detail = "envelope addressed to rank " + std::to_string(e.dest_rank) +
                 ", outside [0, " + std::to_string(size_) +
                 ") of the router's communicator";
      }
      continue;
    }
    // Overflow-safe: never form the sum that would wrap.
    if (e.payload.size() > kIntMax - kHeaderBytes ||
        total_bytes > kIntMax - kHeaderBytes - e.payload.size()) {
      if (err == 0) {
        err = 2;
        detail =
            "this rank's outgoing envelopes exceed the INT_MAX bytes an MPI "
            "count can express; split the routing into several calls";
      }
      total_bytes = kIntMax;  // saturate, do not wrap
      continue;
    }
    total_bytes += kHeaderBytes + e.payload.size();
  }

  {
    int agreed = err;
    const int rc = MPI_Allreduce(&err, &agreed, 1, MPI_INT, MPI_MAX, comm_);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CROUTER-MPI",
                   "MPI_Allreduce failed while agreeing the routing verdict: " +
                       mpi_error_text(rc)};
    }
    if (agreed != 0) {
      const char* code = agreed == 1 ? "E-CROUTER-RANK" : "E-CROUTER-SIZE";
      if (err != 0) return Failure{code, detail};
      return Failure{code,
                   "another rank rejected the routing request (" +
                       std::string{code} +
                       "); this rank's own envelopes were consistent"};
    }
  }

  // -- Routing -------------------------------------------------------------
  // INVARIANT, maintained by every round: every envelope still in `pending`
  // is addressed to a rank inside the current range [lo, hi], and this rank is
  // inside [lo, hi]. It holds initially because [0, P-1] contains every rank
  // and the validation above proved every destination is one. It is preserved
  // because a round ships out precisely the envelopes addressed across the
  // split. When the range collapses to a single rank it is therefore exactly
  // the statement "everything left is mine".
  std::vector<Envelope> pending = std::move(outgoing);

  int lo = 0;
  int hi = size_ - 1;

  std::vector<std::uint8_t> sendbuf;
  std::vector<std::uint8_t> recvbuf;
  std::vector<Envelope>     keep;

  while (lo < hi) {
    const int n = hi - lo + 1;
    const int bl = (n + 1) / 2;  // >= tl, always, so at most one rank unpaired
    const int mid = lo + bl - 1;
    const int tl = n - bl;
    const bool bottom = rank_ <= mid;
    const bool odd = (n % 2) != 0;

    // Who this rank talks to this round. Derived from (lo, hi, rank) alone --
    // no exchange, no barrier, and identical to what the peer derives.
    int send_to = -1;
    int recv_a = -1;
    int recv_b = -1;
    if (bottom) {
      const int i = rank_ - lo;
      if (i < tl) {
        send_to = mid + 1 + i;
        recv_a = send_to;
      } else {
        // i == tl, which happens only when n is odd, and only for rank `mid`.
        // Its messages go to the first rank above the split, which expects
        // them because it can see `n` is odd too. n odd and n > 1 implies
        // tl >= 1, so mid + 1 <= hi.
        send_to = mid + 1;
      }
    } else {
      const int i = rank_ - (mid + 1);
      send_to = lo + i;  // in [lo, lo + tl - 1], always a real bottom rank
      recv_a = send_to;
      if (odd && rank_ == mid + 1) recv_b = mid;
    }

    // Partition `pending` on which side of the split each envelope is bound
    // for. `keep` is reused across rounds so the loop does not reallocate per
    // round; `route()` itself is setup-time code, but the router is also the
    // inner loop of a directory build over a whole mesh.
    keep.clear();
    sendbuf.clear();
    std::size_t round_bytes = 0;
    for (Envelope& e : pending) {
      const bool across =
          bottom ? (e.dest_rank > mid) : (e.dest_rank <= mid);
      if (!across) {
        keep.push_back(std::move(e));
        continue;
      }
      const std::size_t need = kHeaderBytes + e.payload.size();
      if (round_bytes > kIntMax - need) {
        // Cannot be agreed: peers are already committed to this round's
        // exchange. See the file header.
        abort_job(comm_,
                  "a routing round's traffic grew beyond INT_MAX bytes");
      }
      round_bytes += need;
      try {
        append_u32(sendbuf, static_cast<std::uint32_t>(e.dest_rank));
        append_u64(sendbuf, static_cast<std::uint64_t>(e.payload.size()));
        sendbuf.insert(sendbuf.end(), e.payload.begin(), e.payload.end());
      } catch (const std::exception& ex) {
        abort_job(comm_, ex.what());
      } catch (...) {
        abort_job(comm_, "an unknown exception while packing a round");
      }
    }
    pending.swap(keep);

    MPI_Request req = MPI_REQUEST_NULL;
    if (send_to >= 0) {
      const int rc = MPI_Isend(sendbuf.empty() ? nullptr : sendbuf.data(),
                               static_cast<int>(sendbuf.size()), MPI_BYTE,
                               send_to, kTag, comm_, &req);
      if (rc != MPI_SUCCESS) {
        abort_job(comm_, ("MPI_Isend failed: " + mpi_error_text(rc)).c_str());
      }
    }

    for (const int src : {recv_a, recv_b}) {
      if (src < 0) continue;
      MPI_Status st{};
      int        rc = MPI_Probe(src, kTag, comm_, &st);
      if (rc != MPI_SUCCESS) {
        abort_job(comm_, ("MPI_Probe failed: " + mpi_error_text(rc)).c_str());
      }
      int nbytes = 0;
      rc = MPI_Get_count(&st, MPI_BYTE, &nbytes);
      if (rc != MPI_SUCCESS || nbytes < 0) {
        abort_job(comm_, "MPI_Get_count failed on a routed message");
      }
      try {
        recvbuf.resize(static_cast<std::size_t>(nbytes));
      } catch (const std::exception& ex) {
        abort_job(comm_, ex.what());
      } catch (...) {
        abort_job(comm_, "an unknown exception sizing a receive buffer");
      }
      rc = MPI_Recv(recvbuf.empty() ? nullptr : recvbuf.data(), nbytes,
                    MPI_BYTE, src, kTag, comm_, MPI_STATUS_IGNORE);
      if (rc != MPI_SUCCESS) {
        abort_job(comm_, ("MPI_Recv failed: " + mpi_error_text(rc)).c_str());
      }

      // Unpack. The bounds checks are defensive -- the sender is this same
      // code in the same job -- but a truncated buffer read out of bounds is
      // exactly the class of bug ASan is in the gating legs to catch, and a
      // check here costs nothing at setup time.
      std::size_t at = 0;
      const std::size_t have = recvbuf.size();
      try {
        while (at < have) {
          if (have - at < kHeaderBytes) {
            abort_job(comm_, "a routed message ended inside an envelope header");
          }
          std::uint32_t dest = 0;
          std::uint64_t len = 0;
          std::memcpy(&dest, recvbuf.data() + at, sizeof(dest));
          at += sizeof(dest);
          std::memcpy(&len, recvbuf.data() + at, sizeof(len));
          at += sizeof(len);
          if (len > have - at) {
            abort_job(comm_, "a routed envelope claims more bytes than arrived");
          }
          Envelope e;
          e.dest_rank = static_cast<int>(dest);
          e.payload.assign(recvbuf.data() + at,
                           recvbuf.data() + at + static_cast<std::size_t>(len));
          at += static_cast<std::size_t>(len);
          pending.push_back(std::move(e));
        }
      } catch (const std::exception& ex) {
        abort_job(comm_, ex.what());
      } catch (...) {
        abort_job(comm_, "an unknown exception unpacking a routed message");
      }
    }

    if (req != MPI_REQUEST_NULL) {
      const int rc = MPI_Wait(&req, MPI_STATUS_IGNORE);
      if (rc != MPI_SUCCESS) {
        abort_job(comm_, ("MPI_Wait failed: " + mpi_error_text(rc)).c_str());
      }
    }

    if (bottom) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
    ++last_rounds_;
  }

  return pending;
}

}  // namespace fuse::detail

#endif  // FUSE_HAVE_MPI
