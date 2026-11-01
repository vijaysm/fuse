// fuse_physics::fe::AlgebraicSystem -- adapted from KARMA's
// karma_algebraic_system.h/.C. See algebraic_system.hpp's own header
// comment for how this port's assembly sink diverges from KARMA's own
// PETSc Mat/Vec construction.

#include "fuse_physics/fe/algebraic_system.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "fuse/base/config.hpp"

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
/// with everyone else's entries `in_range` now accepts. A STATED ROUND-1
/// SIMPLIFICATION (algebraic_system.hpp's own header comment on
/// assemble_matrix): every rank's own out-of-range buffer is broadcast
/// to every other rank via the same padded-allgather technique
/// fe::DofMap's own ghost resolution uses (that file's header comment
/// states the identical scalability caveat), rather than a point-to-
/// point exchange sized to what each rank actually needs. Single-rank
/// (`reducer.size() == 1`) is the identity check: nothing is ever out of
/// range when one rank owns every row, matching every tier-0 test this
/// wave exercises; a leftover out-of-range entry there is a caller
/// mistake, refused by name rather than silently dropped.
template <class Entry, class InRange, class Pack, class Unpack>
std::vector<Entry> exchange_foreign_entries(const std::vector<Entry>& local, const linalg::Reducer& reducer,
                                             InRange in_range_fn, Pack pack, Unpack unpack,
                                             std::size_t record_width) {
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

  constexpr std::uint64_t kSentinelMarker = ~std::uint64_t{0};
  const auto max_count =
      static_cast<std::size_t>(std::llround(reducer.max(static_cast<double>(out_of_range.size()))));
  std::vector<double> send(max_count * record_width, 0.0);
  for (std::size_t i = 0; i < max_count; ++i) send[i * record_width] = std::bit_cast<double>(kSentinelMarker);
  for (std::size_t i = 0; i < out_of_range.size(); ++i) pack(&send[i * record_width], out_of_range[i]);

  std::vector<double> recv(max_count * record_width * static_cast<std::size_t>(reducer.size()), 0.0);
  reducer.allgather(send.data(), max_count * record_width, recv.data());

  std::vector<Entry> result = in_range;
  const std::size_t n_slots = max_count * static_cast<std::size_t>(reducer.size());
  for (std::size_t slot = 0; slot < n_slots; ++slot) {
    const double* record = &recv[slot * record_width];
    if (std::bit_cast<std::uint64_t>(record[0]) == kSentinelMarker) continue;
    const Entry entry = unpack(record);
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
    : mesh_(&mesh), dof_map_(&dof_map) {
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

Result<linalg::DistSparseMatrix> AlgebraicSystem::assemble_matrix(const ElementContribution& contribution) const {
  const std::int64_t row_lo = dof_map_->first_owned_dof();
  const std::int64_t row_hi = dof_map_->end_owned_dof();
  const std::int64_t global_n = dof_map_->n_dofs();

  std::vector<linalg::Triplet> local;
  std::vector<std::int64_t> indices;
  for (spatial::EntityHandle element = 1; element <= mesh_->n_elements(); ++element) {
    if (!dof_map_->owns_element_for_assembly(element)) continue;
    dof_map_->dof_indices(element, indices);
    contribution(element, indices, local);
  }

  const auto pack = [](double* record, const linalg::Triplet& t) {
    record[0] = std::bit_cast<double>(static_cast<std::uint64_t>(t.row));
    record[1] = std::bit_cast<double>(static_cast<std::uint64_t>(t.col));
    record[2] = t.value;
  };
  const auto unpack = [](const double* record) {
    return linalg::Triplet{static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(record[0])),
                            static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(record[1])), record[2]};
  };
  const auto in_range = [&](const linalg::Triplet& t) { return t.row >= row_lo && t.row < row_hi; };
  const auto resolved = exchange_foreign_entries<linalg::Triplet>(local, *reducer_, in_range, pack, unpack, 3);

  linalg::RowOwnership owners;
  owners.row_lo = dof_map_->row_lo_by_rank();
  if (owners.row_lo.empty()) owners.row_lo = {row_lo};  // serial convenience path: one participant
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

  const auto pack = [](double* record, const Entry& e) {
    record[0] = std::bit_cast<double>(static_cast<std::uint64_t>(e.first));
    record[1] = e.second;
  };
  const auto unpack = [](const double* record) -> Entry {
    return {static_cast<std::int64_t>(std::bit_cast<std::uint64_t>(record[0])), record[1]};
  };
  const auto in_range = [&](const Entry& e) { return e.first >= row_lo && e.first < row_hi; };
  const auto resolved = exchange_foreign_entries<Entry>(local, *reducer_, in_range, pack, unpack, 2);

  out.set_zero();
  for (const auto& [global_index, value] : resolved) out.data()[global_index - row_lo] += value;
  return Result<void>::success();
}

}  // namespace fuse::physics::fe
