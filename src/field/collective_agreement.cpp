// See include/fuse/field/collective_agreement.hpp for the contract.
//
// Collective sequence (every rank, in this order, no early return between
// them except on an MPI failure):
//   1. MPI_Allreduce(MPI_2INT, MPI_MAXLOC) over {refused ? 1 : 0, rank}:
//      the agreed verdict and the LOWEST-numbered refusing rank (MPI_MAXLOC
//      breaks a tie in the value on the smallest index; MPI 4.1, 6.9.4).
//   2. MPI_Allreduce(MPI_INT, MPI_SUM) over {refused ? 1 : 0}: how many.
//   3. If the verdict is "refused": MPI_Bcast of a three-int header (code
//      length, message length, numeric status) from the origin, then
//      MPI_Bcast of the bytes (code followed by message). Both
//      broadcasts are reached by every rank or by none, because the verdict
//      they are conditioned on is the agreed one, not the local one.
// The explicit MPI_Comm_rank/size are read from the communicator handed in,
// never from a cached value, so a sub-communicator is handled correctly.

#include "fuse/field/collective_agreement.hpp"

// Explicit, not transitive: the feature macro FUSE_HAVE_MPI is read below.
#include "fuse/base/config.hpp"

#include <cstdint>
#include <string>
#include <vector>

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse {

std::string format_agreed_refusal(std::string_view phase, const AgreedRefusal& refusal) {
  std::string out;
  out.reserve(phase.size() + refusal.origin_code.size() + refusal.origin_message.size() + 96);
  out.append(phase);
  out.append(": rank ");
  out.append(std::to_string(refusal.origin_rank));
  out.append(" of ");
  out.append(std::to_string(refusal.communicator_size));
  out.append(" refused (");
  out.append(std::to_string(refusal.refusing_ranks));
  out.append(refusal.refusing_ranks == 1 ? " refusing rank" : " refusing ranks");
  out.append("; the lowest-numbered is reported) with ");
  out.append(refusal.origin_code);
  out.append(": ");
  out.append(refusal.origin_message);
  return out;
}

Failure make_agreed_failure(std::string_view agreed_code, std::string_view phase,
                            const AgreedRefusal& refusal) {
  Failure f;
  f.code = std::string(agreed_code);
  f.message = format_agreed_refusal(phase, refusal);
  f.origin_rank = refusal.origin_rank;
  f.origin_code = refusal.origin_code;
  f.origin_status = refusal.origin_status;
  return f;
}

Result<void> agree_outcome(int fortran_communicator, const Result<void>& local,
                           std::string_view agreed_code, std::string_view comm_failure_code,
                           std::string_view phase, std::int32_t local_status_number) noexcept {
  const auto comm_failure = [&](const char* what) {
    return Failure{std::string(comm_failure_code),
                   std::string(phase) + ": " + what + " failed while agreeing the outcome"};
  };
#if FUSE_HAVE_MPI
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_communicator));
  if (comm == MPI_COMM_NULL) return comm_failure("MPI_Comm_f2c (null communicator)");
  int rank = 0;
  int size = 0;
  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS || MPI_Comm_size(comm, &size) != MPI_SUCCESS) {
    return comm_failure("MPI_Comm_rank/MPI_Comm_size");
  }
  static_assert(sizeof(int) == sizeof(std::int32_t),
                "MPI_2INT packs two `int`s; the verdict below is int32_t -- the two must agree");
  struct {
    std::int32_t refused;
    int rank;
  } in{local.ok() ? 0 : 1, rank}, out{0, 0};
  if (MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, comm) != MPI_SUCCESS) {
    return comm_failure("MPI_Allreduce(MPI_MAXLOC)");
  }
  int refusing = 0;
  if (MPI_Allreduce(&in.refused, &refusing, 1, MPI_INT, MPI_SUM, comm) != MPI_SUCCESS) {
    return comm_failure("MPI_Allreduce(MPI_SUM)");
  }
  if (out.refused == 0) return Result<void>::success();

  // The agreed verdict is "refused": the origin broadcasts its own Failure.
  AgreedRefusal refusal;
  refusal.origin_rank = out.rank;
  refusal.refusing_ranks = refusing;
  refusal.communicator_size = size;
  std::int32_t header[3] = {0, 0, 0};  // code length, message length, numeric status
  std::string bytes;
  if (rank == out.rank) {
    // On the origin, `local` is a failure by construction of MPI_MAXLOC.
    const Failure& mine = local.error();
    header[0] = static_cast<std::int32_t>(mine.code.size());
    header[1] = static_cast<std::int32_t>(mine.message.size());
    header[2] = local_status_number;
    bytes = mine.code + mine.message;
  }
  if (MPI_Bcast(header, 3, MPI_INT, out.rank, comm) != MPI_SUCCESS) {
    return comm_failure("MPI_Bcast (header)");
  }
  const std::int32_t lengths[2] = {header[0], header[1]};
  refusal.origin_status = header[2];
  bytes.resize(static_cast<std::size_t>(lengths[0]) + static_cast<std::size_t>(lengths[1]));
  if (!bytes.empty() &&
      MPI_Bcast(bytes.data(), static_cast<int>(bytes.size()), MPI_CHAR, out.rank, comm) !=
          MPI_SUCCESS) {
    return comm_failure("MPI_Bcast (bytes)");
  }
  refusal.origin_code = bytes.substr(0, static_cast<std::size_t>(lengths[0]));
  refusal.origin_message = bytes.substr(static_cast<std::size_t>(lengths[0]));
  return make_agreed_failure(agreed_code, phase, refusal);
#else
  (void)fortran_communicator;
  (void)local;
  (void)local_status_number;
  (void)agreed_code;
  return Failure{std::string(comm_failure_code),
                 std::string(phase) +
                     ": a communicator was supplied but this build has no MPI "
                     "(FUSE_HAVE_MPI=OFF); there is no communicator to agree over"};
#endif
}

}  // namespace fuse
