// fuse_physics::fe::AlgebraicSystem -- adapted from KARMA's
// karma_algebraic_system.h/.C. See algebraic_system.hpp's own header
// comment for how this port's assembly sink diverges from KARMA's own
// PETSc Mat/Vec construction.

#include "fuse_physics/fe/algebraic_system.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

/// Which rank owns global row/dof `index`, read from `row_lo` (ascending,
/// one entry per rank, DofMap::row_lo_by_rank()'s own contract).
int owner_of(const std::vector<std::int64_t>& row_lo, std::int64_t index) {
  const auto it = std::upper_bound(row_lo.begin(), row_lo.end(), index);
  return static_cast<int>(std::distance(row_lo.begin(), it)) - 1;
}

/// Exchanges every rank's own OUT-OF-RANGE entries (failing `in_range`)
/// to their owners, returning this rank's own in-range entries UNIONED
/// with everyone else's entries `in_range` now accepts. ROUND 4
/// ITEM X replaces the round-1/round-2 padded-allgather transport
/// (every rank's out-of-range buffer broadcast to every other rank,
/// padded to the largest buffer anyone holds -- the scalability caveat
/// this function's own header comment used to carry, mirrored in
/// dof_map.cpp's identical note on its own ghost resolution) with a
/// genuine point-to-point exchange: each out-of-range entry is routed,
/// via the new `owner_of_fn`, ONLY to the one participant that owns it,
/// through `Reducer::alltoallv` (comm.hpp) -- one `MPI_Alltoall` for the
/// per-rank counts, one `MPI_Alltoallv` for the payload, no padding and
/// no sentinel record, since every participant now knows exactly how
/// many records every other one is sending it. `n_int_fields` of an
/// entry's INTEGER fields travel through the native `std::int64_t`
/// channel (own reasoning for keeping an
/// integer payload off `double`'s bit pattern applies unchanged here);
/// its one genuine `double` VALUE field travels through the `double`
/// channel. Single-rank (`reducer.size() == 1`) is still the identity
/// check: nothing is ever out of range when one rank owns every row,
/// matching every tier-0 test this wave exercises; a leftover
/// out-of-range entry there is a caller mistake, refused by name rather
/// than silently dropped.
///
/// MESSAGE VOLUME, before vs after (R participants, this rank sending
/// `k` out-of-range entries of `n_int_fields` int64 fields + 1 double
/// each): the allgather version moved `max_k * R` records on EVERY
/// participant regardless of how many entries any of them actually
/// wanted (`max_k` being the largest `k` any participant holds, so one
/// unbalanced rank inflates every other rank's exchange too); this
/// version moves exactly `k` records out and exactly as many in as this
/// rank's own ownership range actually receives, with no `max()`
/// collective needed to size a shared buffer at all.
template <class Entry, class InRange, class OwnerOf, class PackInts, class ValueOf, class Unpack>
std::vector<Entry> exchange_foreign_entries(const std::vector<Entry>& local, const linalg::Reducer& reducer,
                                             InRange in_range_fn, OwnerOf owner_of_fn, PackInts pack_ints,
                                             ValueOf value_of, Unpack unpack, std::size_t n_int_fields) {
  std::vector<Entry> in_range;
  std::vector<Entry> out_of_range;
  in_range.reserve(local.size());
  for (const auto& entry : local) {
    if (in_range_fn(entry))
      in_range.push_back(entry);
    else
      out_of_range.push_back(entry);
  }
  if (reducer.size() == 1) {
    if (!out_of_range.empty())
      refuse("AlgebraicSystem: an element contribution names a row/dof no participant owns (single-rank run)");
    return in_range;
  }

  const int n_ranks = reducer.size();
  // Bucket this rank's own out-of-range entries by destination, ranks
  // ascending -- the grouping both `Reducer::alltoallv` calls below
  // (int channel and value channel) must agree on, since the k-th
  // received value in one channel must be the k-th received record in
  // the other.
  std::vector<std::vector<const Entry*>> by_rank(static_cast<std::size_t>(n_ranks));
  for (const auto& entry : out_of_range) {
    const int dest = owner_of_fn(entry);
    FUSE_ASSERT(dest >= 0 && dest < n_ranks,
               "AlgebraicSystem::exchange_foreign_entries: owner_of_fn returned rank {} outside [0, {})", dest,
               n_ranks);
    by_rank[static_cast<std::size_t>(dest)].push_back(&entry);
  }

  std::vector<int> send_int_counts(static_cast<std::size_t>(n_ranks), 0);
  std::vector<int> send_value_counts(static_cast<std::size_t>(n_ranks), 0);
  for (int r = 0; r < n_ranks; ++r) {
    send_value_counts[static_cast<std::size_t>(r)] = static_cast<int>(by_rank[static_cast<std::size_t>(r)].size());
    send_int_counts[static_cast<std::size_t>(r)] =
        send_value_counts[static_cast<std::size_t>(r)] * static_cast<int>(n_int_fields);
  }

  std::vector<std::int64_t> send_ints;
  std::vector<double> send_values;
  send_ints.reserve(out_of_range.size() * n_int_fields);
  send_values.reserve(out_of_range.size());
  std::vector<std::int64_t> record(n_int_fields);
  for (int r = 0; r < n_ranks; ++r) {
    for (const Entry* entry : by_rank[static_cast<std::size_t>(r)]) {
      pack_ints(record.data(), *entry);
      for (std::size_t k = 0; k < n_int_fields; ++k) send_ints.push_back(record[k]);
      send_values.push_back(value_of(*entry));
    }
  }

  std::vector<int> recv_int_counts;
  std::vector<int> recv_value_counts;
  const auto recv_ints = reducer.alltoallv(send_ints, send_int_counts, recv_int_counts);
  const auto recv_values = reducer.alltoallv(send_values, send_value_counts, recv_value_counts);

  const std::size_t n_received = recv_values.size();
  FUSE_ASSERT(recv_ints.size() == n_received * n_int_fields,
             "AlgebraicSystem::exchange_foreign_entries: int channel ({} values) and value channel ({} "
             "entries) disagree on how many records were received -- the two alltoallv calls above must "
             "use proportional per-rank counts (n_int_fields x the entry count) to stay in lock-step",
             recv_ints.size(), n_received);

  std::vector<Entry> result = in_range;
  for (std::size_t slot = 0; slot < n_received; ++slot) {
    const std::int64_t* int_record = &recv_ints[slot * n_int_fields];
    const Entry entry = unpack(int_record, recv_values[slot]);
    // Defensive, not load-bearing: owner_of_fn routed this record here
    // because it believes THIS rank owns it, so in_range_fn should
    // always agree; kept as a plain filter (not an assert) only because
    // the allgather version this replaces had the identical filter for
    // the identical reason -- a caller whose owner_of_fn disagrees with
    // its own in_range_fn has a wiring bug, not a legitimate case for
    // the record to be silently dropped forever, but diagnosing that
    // mismatch is outside this function's own contract.
    if (in_range_fn(entry)) result.push_back(entry);
  }
  return result;
}

}  // namespace

void eliminate_dirichlet_rows(std::vector<linalg::Triplet>& triplets, std::int64_t global_n,
                               const std::vector<std::int64_t>& dirichlet_rows,
                               const std::vector<double>& values, std::vector<double>& rhs) {
  if (dirichlet_rows.size() != values.size())
    refuse("eliminate_dirichlet_rows: dirichlet_rows and values must have the same size");
  if (rhs.size() != static_cast<std::size_t>(global_n))
    refuse("eliminate_dirichlet_rows: rhs must have global_n entries");

  std::vector<bool> is_dirichlet(static_cast<std::size_t>(global_n), false);
  std::vector<double> value_of(static_cast<std::size_t>(global_n), 0.0);
  for (std::size_t i = 0; i < dirichlet_rows.size(); ++i) {
    const auto row = dirichlet_rows[i];
    if (row < 0 || row >= global_n) refuse("eliminate_dirichlet_rows: a dirichlet row is out of range");
    is_dirichlet[static_cast<std::size_t>(row)] = true;
    value_of[static_cast<std::size_t>(row)] = values[i];
  }

  std::vector<linalg::Triplet> kept;
  kept.reserve(triplets.size());
  for (const auto& t : triplets) {
    if (t.row < 0 || t.row >= global_n || t.col < 0 || t.col >= global_n)
      refuse("eliminate_dirichlet_rows: a triplet's row or column is out of range");
    if (is_dirichlet[static_cast<std::size_t>(t.row)]) continue;  // row replaced by the identity row below
    if (is_dirichlet[static_cast<std::size_t>(t.col)]) {
      // Symmetric elimination: this triplet's column is being removed
      // exactly as its own row would be, so its contribution moves to
      // the free row's rhs instead of staying in the matrix.
      rhs[static_cast<std::size_t>(t.row)] -= t.value * value_of[static_cast<std::size_t>(t.col)];
      continue;
    }
    kept.push_back(t);
  }
  for (std::size_t i = 0; i < dirichlet_rows.size(); ++i) {
    const auto row = dirichlet_rows[i];
    kept.push_back(linalg::Triplet{row, row, 1.0});
    rhs[static_cast<std::size_t>(row)] = values[i];
  }
  triplets = std::move(kept);
}

AlgebraicSystem::AlgebraicSystem(const spatial::Mesh& mesh, const DofMap& dof_map,
                                  std::optional<int> fortran_communicator)
    : mesh_(&mesh), dof_map_(&dof_map), fortran_communicator_(fortran_communicator) {
  if (!fortran_communicator.has_value()) {
    reducer_ = linalg::make_serial_reducer();
    exchanger_ = linalg::make_serial_exchanger();
    return;
  }
#if FUSE_HAVE_MPI
  reducer_ = linalg::make_mpi_reducer(*fortran_communicator);
  const auto& row_lo = dof_map_->row_lo_by_rank();
  linalg::GhostPattern pattern;
  pattern.owner_rank.reserve(dof_map_->ghost_indices().size());
  pattern.remote_local_index.reserve(dof_map_->ghost_indices().size());
  for (const auto ghost_global : dof_map_->ghost_indices()) {
    const int owner = owner_of(row_lo, ghost_global);
    pattern.owner_rank.push_back(owner);
    pattern.remote_local_index.push_back(ghost_global - row_lo.at(static_cast<std::size_t>(owner)));
  }
  exchanger_ = linalg::make_mpi_exchanger(*fortran_communicator, pattern,
                                           static_cast<std::size_t>(dof_map_->n_local_dofs()));
#else
  refuse("AlgebraicSystem: a communicator was given but this build has no MPI (FUSE_HAVE_MPI=0)");
#endif
}

linalg::DistVector AlgebraicSystem::new_vector() const {
  linalg::DistVector v(static_cast<std::size_t>(dof_map_->n_local_dofs()), exchanger_, reducer_);
  v.set_zero();
  return v;
}

linalg::DistVector AlgebraicSystem::new_vector_for(const linalg::DistSparseMatrix& matrix) const {
  const auto owned_n = static_cast<std::size_t>(dof_map_->n_local_dofs());
  std::shared_ptr<const linalg::GhostExchanger> exchanger;
  if (!fortran_communicator_.has_value()) {
    exchanger = linalg::make_serial_exchanger();
  } else {
#if FUSE_HAVE_MPI
    exchanger = linalg::make_mpi_exchanger(*fortran_communicator_, matrix.ghost_pattern(), owned_n);
#else
    refuse("AlgebraicSystem::new_vector_for: a communicator was given but this build has no MPI (FUSE_HAVE_MPI=0)");
#endif
  }
  linalg::DistVector v(owned_n, std::move(exchanger), reducer_);
  v.set_zero();
  return v;
}

Result<linalg::DistSparseMatrix> AlgebraicSystem::assemble_matrix(const ElementContribution& contribution) const {
  const std::int64_t row_lo = dof_map_->first_owned_dof();
  const std::int64_t row_hi = dof_map_->end_owned_dof();
  const std::int64_t global_n = dof_map_->n_dofs();
  std::vector<std::int64_t> row_lo_by_rank = dof_map_->row_lo_by_rank();
  if (row_lo_by_rank.empty()) row_lo_by_rank = {row_lo};  // serial convenience path: one participant

  std::vector<linalg::Triplet> local;
  std::vector<std::int64_t> indices;
  for (spatial::EntityHandle element = 1; element <= mesh_->n_elements(); ++element) {
    if (!dof_map_->owns_element_for_assembly(element)) continue;
    dof_map_->dof_indices(element, indices);
    contribution(element, indices, local);
  }

  const auto pack_ints = [](std::int64_t* record, const linalg::Triplet& t) {
    record[0] = t.row;
    record[1] = t.col;
  };
  const auto value_of = [](const linalg::Triplet& t) { return t.value; };
  const auto unpack = [](const std::int64_t* record, double value) { return linalg::Triplet{record[0], record[1], value}; };
  const auto in_range = [&](const linalg::Triplet& t) { return t.row >= row_lo && t.row < row_hi; };
  // Item X: a triplet's destination is whichever rank owns ITS ROW
  // (row_lo_by_rank's own ascending contract, the same `owner_of`
  // helper AlgebraicSystem's own constructor uses to build the ghost
  // pattern) -- not a broadcast, since the row's owner is derivable
  // locally from row_lo_by_rank alone.
  const auto owner_of_entry = [&](const linalg::Triplet& t) { return owner_of(row_lo_by_rank, t.row); };
  const auto resolved = exchange_foreign_entries<linalg::Triplet>(local, *reducer_, in_range, owner_of_entry,
                                                                    pack_ints, value_of, unpack, 2);

  linalg::RowOwnership owners;
  owners.row_lo = row_lo_by_rank;
  return linalg::DistSparseMatrix::from_triplets(row_lo, row_hi, global_n, resolved, reducer_, owners);
}

Result<void> AlgebraicSystem::assemble_vector(const ElementVectorContribution& contribution,
                                               linalg::DistVector& out) const {
  const std::int64_t row_lo = dof_map_->first_owned_dof();
  const std::int64_t row_hi = dof_map_->end_owned_dof();
  if (out.owned_size() != static_cast<std::size_t>(dof_map_->n_local_dofs()))
    refuse("AlgebraicSystem::assemble_vector: out does not match this system's own row distribution");

  using Entry = std::pair<std::int64_t, double>;
  std::vector<Entry> local;
  std::vector<std::int64_t> indices;
  for (spatial::EntityHandle element = 1; element <= mesh_->n_elements(); ++element) {
    if (!dof_map_->owns_element_for_assembly(element)) continue;
    dof_map_->dof_indices(element, indices);
    contribution(element, indices, local);
  }

  const auto pack_ints = [](std::int64_t* record, const Entry& e) { record[0] = e.first; };
  const auto value_of = [](const Entry& e) { return e.second; };
  const auto unpack = [](const std::int64_t* record, double value) -> Entry { return {record[0], value}; };
  const auto in_range = [&](const Entry& e) { return e.first >= row_lo && e.first < row_hi; };
  std::vector<std::int64_t> row_lo_by_rank = dof_map_->row_lo_by_rank();
  if (row_lo_by_rank.empty()) row_lo_by_rank = {row_lo};  // serial convenience path: one participant
  const auto owner_of_entry = [&](const Entry& e) { return owner_of(row_lo_by_rank, e.first); };
  const auto resolved =
      exchange_foreign_entries<Entry>(local, *reducer_, in_range, owner_of_entry, pack_ints, value_of, unpack, 1);

  out.set_zero();
  for (const auto& [global_index, value] : resolved) out.data()[global_index - row_lo] += value;
  return Result<void>::success();
}

}  // namespace fuse::physics::fe
