/// Restart track round 2, item 5 -- see include/fuse/restart_io/
/// collective_check_mpi.hpp's own header comment for the design and why
/// an int (MPI_Comm_c2f handle), not MPI_Comm, crosses the public
/// boundary. This translation unit is the ONLY one in fuse-coupler-
/// restart-io compiled under FUSE_ENABLE_MPI specifically for a genuine
/// collective (src/CMakeLists.txt's own comment distinguishes it from
/// the mpi.h HDF5's own header forces on every consumer regardless).
#include "fuse/restart_io/collective_check_mpi.hpp"

#include <algorithm>
#include <cstring>

#include <mpi.h>

namespace fuse::restart_io {

namespace {

/// Serializes `ids` as a flat byte buffer: [uint32 name_len][name bytes]
/// [int64 id], repeated. No alignment padding -- MPI_Allgatherv moves
/// MPI_BYTE, which has none to respect, and every reader below walks the
/// buffer by explicit offsets, never by casting a pointer into a struct.
std::vector<std::uint8_t> serialize_ids(
    const std::vector<std::pair<std::string, std::int64_t>>& ids) {
  std::vector<std::uint8_t> buf;
  for (const auto& [name, id] : ids) {
    const std::uint32_t len = static_cast<std::uint32_t>(name.size());
    const std::size_t off = buf.size();
    buf.resize(off + sizeof(len) + len + sizeof(id));
    std::memcpy(buf.data() + off, &len, sizeof(len));
    std::memcpy(buf.data() + off + sizeof(len), name.data(), len);
    std::memcpy(buf.data() + off + sizeof(len) + len, &id, sizeof(id));
  }
  return buf;
}

std::vector<std::uint8_t> serialize_declared(
    const std::unordered_map<std::string, std::int64_t>& declared) {
  std::vector<std::pair<std::string, std::int64_t>> as_pairs(declared.begin(),
                                                              declared.end());
  return serialize_ids(as_pairs);
}

/// The inverse of serialize_ids(): walks `buf` end to end, appending
/// every (name, id) pair it finds to `out`. `buf` may be the
/// concatenation of several ranks' own buffers back to back (exactly
/// what MPI_Allgatherv produces) -- this makes no assumption about where
/// one rank's contribution ends and the next begins, because every pair
/// is self-delimiting (its own length prefix) and reading them back to
/// back across a rank boundary is indistinguishable from reading them
/// back to back within one rank's own buffer.
Result<void> deserialize_ids_into(
    const std::vector<std::uint8_t>& buf,
    std::vector<std::pair<std::string, std::int64_t>>& out) {
  std::size_t off = 0;
  while (off < buf.size()) {
    if (off + sizeof(std::uint32_t) > buf.size()) {
      return Failure{"E-RESTART-HDF5",
                     "collective_duplicate_and_coverage_check: a gathered "
                     "id buffer is truncated (length prefix past end) -- "
                     "this indicates a corrupted MPI_Allgatherv displacement, "
                     "not a user-facing restart file defect"};
    }
    std::uint32_t len = 0;
    std::memcpy(&len, buf.data() + off, sizeof(len));
    off += sizeof(len);
    if (off + len + sizeof(std::int64_t) > buf.size()) {
      return Failure{"E-RESTART-HDF5",
                     "collective_duplicate_and_coverage_check: a gathered "
                     "id buffer is truncated (name or id past end)"};
    }
    std::string name(reinterpret_cast<const char*>(buf.data() + off), len);
    off += len;
    std::int64_t id = 0;
    std::memcpy(&id, buf.data() + off, sizeof(id));
    off += sizeof(id);
    out.emplace_back(std::move(name), id);
  }
  return Result<void>::success();
}

/// One Allgatherv round trip: every rank's own `local` buffer becomes
/// `out`, the concatenation of every rank's buffer in rank order.
Result<void> allgatherv_bytes(MPI_Comm comm,
                              const std::vector<std::uint8_t>& local,
                              std::vector<std::uint8_t>& out) {
  int size = 0;
  MPI_Comm_size(comm, &size);
  const int local_count = static_cast<int>(local.size());
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  if (MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                    comm) != MPI_SUCCESS) {
    return Failure{"E-RESTART-HDF5",
                   "collective_duplicate_and_coverage_check: "
                   "MPI_Allgather (sizes) failed"};
  }
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int total = 0;
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }
  out.resize(static_cast<std::size_t>(total));
  if (MPI_Allgatherv(local.data(), local_count, MPI_BYTE, out.data(),
                     counts.data(), displs.data(), MPI_BYTE,
                     comm) != MPI_SUCCESS) {
    return Failure{"E-RESTART-HDF5",
                   "collective_duplicate_and_coverage_check: "
                   "MPI_Allgatherv (payload) failed"};
  }
  return Result<void>::success();
}

}  // namespace

Result<void> collective_duplicate_and_coverage_check(
    int comm_f, const std::vector<std::pair<std::string, std::int64_t>>& local_ids,
    const std::unordered_map<std::string, std::int64_t>& local_declared_sizes) {
  const MPI_Comm comm = MPI_Comm_f2c(comm_f);

  // Owner's own "never one collective per array" rule: exactly two
  // Allgatherv calls total, covering every field's ids and every
  // field's declared size together, not one call per field.
  std::vector<std::uint8_t> gathered_ids;
  Result<void> r =
      allgatherv_bytes(comm, serialize_ids(local_ids), gathered_ids);
  if (!r.ok()) return r;
  std::vector<std::uint8_t> gathered_declared;
  r = allgatherv_bytes(comm, serialize_declared(local_declared_sizes),
                       gathered_declared);
  if (!r.ok()) return r;

  // Past this point every rank holds the IDENTICAL two buffers -- the
  // rest of this function is a PURE, LOCAL computation over data that is
  // now the same everywhere, so every rank reaches the identical verdict
  // without any further communication (the "agree first" rule is
  // satisfied by the gather itself, not by a trailing reduction).
  std::vector<std::pair<std::string, std::int64_t>> combined;
  r = deserialize_ids_into(gathered_ids, combined);
  if (!r.ok()) return r;
  std::vector<std::pair<std::string, std::int64_t>> declared_pairs;
  r = deserialize_ids_into(gathered_declared, declared_pairs);
  if (!r.ok()) return r;

  std::sort(combined.begin(), combined.end());
  for (std::size_t i = 1; i < combined.size(); ++i) {
    if (combined[i] == combined[i - 1]) {
      return Failure{"E-RESTART-DUPLICATE-ID",
                     "collective_duplicate_and_coverage_check: duplicate "
                     "(field, id) = ('" +
                         combined[i].first + "', " +
                         std::to_string(combined[i].second) +
                         ") reported by more than one rank (or twice by "
                         "the same rank)"};
    }
  }

  std::unordered_map<std::string, std::int64_t> agreed_declared;
  for (const auto& [name, declared] : declared_pairs) {
    const auto it = agreed_declared.find(name);
    if (it == agreed_declared.end()) {
      agreed_declared.emplace(name, declared);
    } else if (it->second != declared) {
      return Failure{"E-RESTART-GLOBAL-SIZE-MISMATCH",
                     "collective_duplicate_and_coverage_check: field '" +
                         name + "' declared " + std::to_string(it->second) +
                         " by one caller and " + std::to_string(declared) +
                         " by another -- every caller of the same field "
                         "must declare the same global size"};
    }
  }

  std::unordered_map<std::string, std::int64_t> received;
  for (const auto& [name, id] : combined) {
    (void)id;
    ++received[name];
  }
  for (const auto& [name, declared] : agreed_declared) {
    const auto it = received.find(name);
    const std::int64_t got = (it == received.end()) ? 0 : it->second;
    if (got != declared) {
      return Failure{"E-RESTART-BAD-NUMBERING",
                     "collective_duplicate_and_coverage_check: field '" +
                         name + "' declared " + std::to_string(declared) +
                         " ids, the whole collective received " +
                         std::to_string(got)};
    }
  }
  return Result<void>::success();
}

}  // namespace fuse::restart_io
