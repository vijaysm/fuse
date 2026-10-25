// fuse::linalg::SchurSlabSolver -- docs/dev/schur-slab-solver.md has the full
// derivation; this file's own comments point at the specific step each piece
// of code performs, not the derivation itself.
//
// Eigen only here, never in include/fuse/linalg/schur_slab.hpp (the layering
// gate: zero third-party types in a linalg public header). Requires
// FUSE_LINALG_BACKEND=eigen (the default, cmake/FuseDependencies.cmake) --
// the same assumption examples/gray_scott's and examples/brusselator's own
// direct links of Eigen3::Eigen already make (their own CMakeLists.txt
// comments say so explicitly); there is no FUSE_LINALG_BACKEND=builtin
// direct sparse/dense factorisation in this library to fall back to, so this
// translation unit is unconditional rather than gated the way
// src/linalg/comm_mpi.cpp is gated on FUSE_ENABLE_MPI (src/CMakeLists.txt's
// own comment on the added link line says the same).

#include "fuse/linalg/schur_slab.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <fmt/format.h>

#include "fuse/base/error.hpp"

namespace fuse::linalg {

namespace {

using SpMat = Eigen::SparseMatrix<double>;

/// EXACT duplicate of `examples/brusselator/brusselator_physics.hpp`'s own
/// `periodic_laplacian(nx, ny)` -- same loop order (`j` outer, `i` inner),
/// same per-entry formula, same `emplace_back` sequence (diagonal, `+i`,
/// `-i`, `+j`, `-j`) -- so that `factor_whole_grid` below assembles a
/// matrix `Eigen::SparseMatrix::setFromTriplets` cannot distinguish, bit
/// for bit, from the Toy's own reference at `nranks==1`
/// (`SchurSlabSolver.OneRankIsBitIdenticalToWholeGridSparseCholesky`,
/// test/linalg/schur_slab_test.cpp). Duplicated rather than included:
/// `src/linalg` may not depend on `examples/`.
SpMat periodic_laplacian_duplicate(int nx, int ny) {
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 5);
  const double xx = double(nx) * nx, yy = double(ny) * ny;
  const auto index = [nx](int i, int j) { return j * nx + i; };
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int row = index(i, j);
      entries.emplace_back(row, row, -2 * (xx + yy));
      entries.emplace_back(row, index((i + 1) % nx, j), xx);
      entries.emplace_back(row, index((i + nx - 1) % nx, j), xx);
      entries.emplace_back(row, index(i, (j + 1) % ny), yy);
      entries.emplace_back(row, index(i, (j + ny - 1) % ny), yy);
    }
  }
  SpMat result(nx * ny, nx * ny);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

/// Agrees, across every rank, whether `local_ok` holds EVERYWHERE -- ONE
/// `Reducer::allgather` of a single double per rank. Returns the (always
/// ascending) rank indices that reported `!local_ok`, empty when every
/// rank agrees; every rank reads the SAME `recv`, so every rank builds the
/// SAME list with no further communication.
///
/// This is the collective every refusal in this file that depends on
/// RANK-VARYING RUNTIME DATA (a factorisation's own numerical outcome,
/// which no rank can predict from another's inputs) must perform BEFORE
/// returning. The row-band-size refusal below needs no collective at all,
/// since `row_ownership` already gives every rank the full partition (see
/// `factor()`'s own comment at that check). A numerical factorisation
/// failure, by contrast, genuinely CANNOT be predicted from
/// `row_ownership` alone, so it is this function's own collective that
/// keeps it from stranding a peer.
std::vector<int> ranks_not_ok(const Reducer& reducer, bool local_ok) {
  const int nranks = reducer.size();
  const double send = local_ok ? 0.0 : 1.0;
  std::vector<double> recv(static_cast<std::size_t>(nranks));
  reducer.allgather(&send, 1, recv.data());
  std::vector<int> bad;
  for (int r = 0; r < nranks; ++r) {
    if (recv[static_cast<std::size_t>(r)] != 0.0) bad.push_back(r);
  }
  return bad;
}

std::string format_rank_list(const std::vector<int>& ranks) {
  std::string out;
  for (std::size_t k = 0; k < ranks.size(); ++k) {
    if (k > 0) out += ", ";
    out += std::to_string(ranks[k]);
  }
  return out;
}

}  // namespace

struct SchurSlabSolver::Impl {
  // nranks == 1: no interface rows at all (docs/dev/schur-slab-solver.md's
  // "what degenerates" section) -- the whole grid, factored and solved
  // exactly as DirectDiffusionSolver (examples/brusselator/species_core.hpp)
  // does, minus its physics-specific extra_diagonal, which this solver's
  // interface does not carry.
  bool whole_grid = false;
  Eigen::SimplicialLDLT<SpMat> whole_factor;

  // nranks >= 2:
  std::int64_t n_interior = 0;           // (local_rows - 2) * nx
  Eigen::SimplicialLDLT<SpMat> interior_factor;  // n_interior x n_interior, present iff n_interior > 0
  Eigen::MatrixXd a_ig;                  // n_interior x 2*nx
  Eigen::LDLT<Eigen::MatrixXd> schur_factor;  // g_total x g_total, IDENTICAL on every rank
  std::int64_t g_total = 0;              // 2*nx*nranks
  std::int64_t g_row_lo = 0;             // this rank's own row/col offset into the Schur system

  int nx = 0;
  std::int64_t local_rows = 0;
  std::shared_ptr<const Reducer> reducer;
};

SchurSlabSolver::SchurSlabSolver() = default;
SchurSlabSolver::SchurSlabSolver(SchurSlabSolver&&) noexcept = default;
SchurSlabSolver& SchurSlabSolver::operator=(SchurSlabSolver&&) noexcept = default;
SchurSlabSolver::~SchurSlabSolver() = default;

Result<SchurSlabSolver> SchurSlabSolver::factor(int nx, std::int64_t ny, std::int64_t j_lo,
                                                std::int64_t j_hi, double coefficient,
                                                const RowOwnership& row_ownership,
                                                std::shared_ptr<const Reducer> reducer) {
  FUSE_ASSERT(nx > 0, "SchurSlabSolver::factor: nx must be positive, got {}", nx);
  FUSE_ASSERT(ny > 0, "SchurSlabSolver::factor: ny must be positive, got {}", ny);
  FUSE_ASSERT(j_lo >= 0 && j_lo <= j_hi && j_hi <= ny,
             "SchurSlabSolver::factor: [j_lo, j_hi) = [{}, {}) is not a valid sub-range of "
             "[0, {})",
             j_lo, j_hi, ny);
  FUSE_ASSERT(reducer != nullptr, "SchurSlabSolver::factor built with a null Reducer");

  const int nranks = reducer->size();
  const int rank = reducer->rank();
  FUSE_ASSERT(static_cast<std::size_t>(nranks) == row_ownership.row_lo.size(),
             "SchurSlabSolver::factor: reducer reports {} participants but row_ownership names "
             "{} entries",
             nranks, row_ownership.row_lo.size());
  FUSE_ASSERT(row_ownership.row_lo[static_cast<std::size_t>(rank)] == j_lo,
             "SchurSlabSolver::factor: row_ownership's own j_lo for this rank ({}) disagrees "
             "with the j_lo argument ({})",
             row_ownership.row_lo[static_cast<std::size_t>(rank)], j_lo);
  const std::int64_t expected_j_hi =
      (rank + 1 < nranks) ? row_ownership.row_lo[static_cast<std::size_t>(rank + 1)] : ny;
  FUSE_ASSERT(expected_j_hi == j_hi,
             "SchurSlabSolver::factor: row_ownership implies j_hi={} for this rank, but j_hi={} "
             "was given",
             expected_j_hi, j_hi);

  const std::int64_t local_rows = j_hi - j_lo;

  auto impl = std::make_unique<Impl>();
  impl->nx = nx;
  impl->local_rows = local_rows;
  impl->reducer = reducer;

  // -------------------------------------------------------------------
  // nranks == 1: no row-band partition at all -- the whole grid, solved
  // exactly as the Toy's own DirectDiffusionSolver does.
  // -------------------------------------------------------------------
  if (nranks == 1) {
    FUSE_ASSERT(j_lo == 0 && j_hi == ny,
               "SchurSlabSolver::factor: a single participant must own the whole grid [0, {}), "
               "got [{}, {})",
               ny, j_lo, j_hi);
    SpMat laplacian = periodic_laplacian_duplicate(nx, static_cast<int>(ny));
    SpMat matrix = (-coefficient) * laplacian;
    const int n = static_cast<int>(matrix.rows());
    for (int i = 0; i < n; ++i) matrix.coeffRef(i, i) += 1.0;

    impl->whole_grid = true;
    impl->whole_factor.compute(matrix);
    if (impl->whole_factor.info() != Eigen::Success) {
      return Failure{"E-LINALG-SCHUR-SLAB-FACTORIZE",
                    "SchurSlabSolver::factor: whole-grid SimplicialLDLT factorization failed"};
    }
    SchurSlabSolver solver;
    solver.impl_ = std::move(impl);
    return std::move(solver);
  }

  // -------------------------------------------------------------------
  // nranks >= 2: EVERY rank's own row band needs 2 distinct rows (E4 in
  // docs/dev/schur-slab-solver.md) to have a first AND a last interface
  // row at all -- not just THIS rank's own.
  //
  // This must not check only `local_rows < 2` and return on THAT alone: a
  // collective path (the Schur-assembly `allgather` below) returning on a
  // condition computed from THIS rank's own data violates the standing rule
  // that a collective path must never do that. If rank 0's single-row band
  // refused and returned without reaching that allgather while every other
  // rank's band was fine and went on into it, those ranks would wait inside
  // a collective nobody else entered (MPICH reports "Message truncated" in
  // MPI_Allgather).
  //
  // The check needs NO new collective: `row_ownership` already gives EVERY
  // rank the full partition (every participant's own `row_lo`, the same
  // convention the FUSE_ASSERT block above already trusts for THIS rank's
  // own `j_lo`/`j_hi`), so every rank can determine -- by scanning ALL
  // `nranks` bands locally, with zero communication -- whether ANY band
  // anywhere is undersized, and reach the IDENTICAL conclusion with the
  // IDENTICAL message, before the one real collective this function
  // performs.
  std::vector<int> undersized_ranks;
  for (int r = 0; r < nranks; ++r) {
    const std::int64_t r_lo = row_ownership.row_lo[static_cast<std::size_t>(r)];
    const std::int64_t r_hi =
        (r + 1 < nranks) ? row_ownership.row_lo[static_cast<std::size_t>(r + 1)] : ny;
    if (r_hi - r_lo < 2) undersized_ranks.push_back(r);
  }
  if (!undersized_ranks.empty()) {
    std::string detail;
    for (std::size_t k = 0; k < undersized_ranks.size(); ++k) {
      const int r = undersized_ranks[k];
      const std::int64_t r_lo = row_ownership.row_lo[static_cast<std::size_t>(r)];
      const std::int64_t r_hi =
          (r + 1 < nranks) ? row_ownership.row_lo[static_cast<std::size_t>(r + 1)] : ny;
      if (k > 0) detail += "; ";
      detail += fmt::format("rank {} owns {} row(s) [{}, {})", r, r_hi - r_lo, r_lo, r_hi);
    }
    return Failure{
        "E-LINALG-SCHUR-SLAB",
        fmt::format("SchurSlabSolver::factor: {} of {} ranks' own row bands own fewer than 2 "
                   "rows ({}); a Schur-complement row band needs at least 2 rows so its first "
                   "and last row are 2 DISTINCT interface rows -- a 1-row (or empty) band would "
                   "have to use its one row as both, double-counting it. Decided identically on "
                   "every rank from row_ownership alone, before the one collective this function "
                   "performs, so no rank is left waiting inside it.",
                   undersized_ranks.size(), nranks, detail)};
  }

  const double xx = double(nx) * nx;
  const double yy = double(ny) * ny;
  const double diag_val = 1.0 + 2.0 * coefficient * (xx + yy);
  const double horiz_val = -coefficient * xx;
  const double vert_val = -coefficient * yy;

  const std::int64_t n_interior = (local_rows - 2) * nx;
  std::vector<Eigen::Triplet<double>> interior_triplets;
  Eigen::MatrixXd a_ig = Eigen::MatrixXd::Zero(n_interior, 2 * nx);
  Eigen::MatrixXd a_gg_own = Eigen::MatrixXd::Zero(2 * nx, 2 * nx);

  // Local row k in [0, local_rows): k==0 is the FIRST interface row, k ==
  // local_rows-1 the LAST; k in [1, local_rows-2] is interior. Classifying
  // (row k, col k) pairs this way and bucketing each generated entry
  // accordingly is what turns ONE pass over the row band's own 5-point
  // stencil into A_II, A_IG and A_GG_own directly -- no intermediate
  // "whole row-band matrix" is ever assembled (docs/dev/
  // schur-slab-solver.md's "assembly, one pass" section has the proof that
  // this bucketing is exhaustive and that the dropped "interface row,
  // interior column" direction is redundant with the kept one by symmetry).
  const auto is_interior = [&](std::int64_t k) { return k >= 1 && k <= local_rows - 2; };
  const auto is_interface = [&](std::int64_t k) { return k == 0 || k == local_rows - 1; };
  const auto interior_row = [&](std::int64_t k, int i) { return (k - 1) * nx + i; };
  const auto interface_col = [&](std::int64_t k, int i) { return (k == 0 ? 0 : nx) + i; };

  const auto add_entry = [&](std::int64_t rk, int ri, std::int64_t ck, int ci, double value) {
    if (is_interior(rk) && is_interior(ck)) {
      interior_triplets.emplace_back(static_cast<int>(interior_row(rk, ri)),
                                     static_cast<int>(interior_row(ck, ci)), value);
    } else if (is_interior(rk) && is_interface(ck)) {
      a_ig(interior_row(rk, ri), interface_col(ck, ci)) += value;
    } else if (is_interface(rk) && is_interface(ck)) {
      a_gg_own(interface_col(rk, ri), interface_col(ck, ci)) += value;
    }
    // is_interface(rk) && is_interior(ck): the TRANSPOSE of an
    // (interior, interface) entry generated from the neighbour's own
    // iteration (the loop below adds both (k, k+1) and (k+1, k)
    // explicitly) -- not stored again; every use below reads A_GI as
    // `a_ig.transpose()`.
  };

  for (std::int64_t k = 0; k < local_rows; ++k) {
    for (int i = 0; i < nx; ++i) {
      add_entry(k, i, k, i, diag_val);
      const int ip1 = (i + 1) % nx;
      const int im1 = (i - 1 + nx) % nx;
      add_entry(k, i, k, ip1, horiz_val);
      add_entry(k, i, k, im1, horiz_val);
      // Vertical neighbours local to this row band only -- k+1 is always
      // in [0, local_rows) unless k IS the last row, k-1 unless k IS the
      // first; the periodic wrap to the NEXT/PREVIOUS row band (owned by
      // a different rank) is handled separately below, never here.
      if (k + 1 <= local_rows - 1) add_entry(k, i, k + 1, i, vert_val);
      if (k - 1 >= 0) add_entry(k, i, k - 1, i, vert_val);
    }
  }

  impl->n_interior = n_interior;
  impl->a_ig = a_ig;

  // Unlike the row-band-size refusal above, a factorisation's own
  // numerical outcome genuinely CANNOT be predicted from row_ownership
  // alone -- it depends on this rank's own assembled VALUES, which no
  // peer can see. `local_ok` is therefore decided locally but NOT acted
  // on locally: every rank's own vote is agreed by `ranks_not_ok` (ONE
  // small collective) before anyone returns, so a factorisation that
  // fails on only SOME ranks' own interior matrix (never expected for
  // this operator's SPD structure, but checked rather than assumed) can
  // never strand the others inside the Schur-assembly allgather below --
  // the same stranding hazard the row-band-size refusal was found to have
  // on the host, fixed here the OTHER way the task's own diagnosis named
  // ("agreed first... then every rank refuses with the same code").
  bool local_ok = true;
  Eigen::MatrixXd s_own = a_gg_own;
  if (n_interior > 0) {
    SpMat interior(n_interior, n_interior);
    interior.setFromTriplets(interior_triplets.begin(), interior_triplets.end());
    impl->interior_factor.compute(interior);
    if (impl->interior_factor.info() != Eigen::Success) {
      local_ok = false;
    } else {
      // S_own = A_GG_own - A_GI * A_II^-1 * A_IG = A_GG_own - A_IG^T * (A_II^-1 A_IG).
      const Eigen::MatrixXd y = impl->interior_factor.solve(a_ig);
      if (impl->interior_factor.info() != Eigen::Success) {
        local_ok = false;
      } else {
        s_own = a_gg_own - a_ig.transpose() * y;
      }
    }
  }

  if (const auto failed = ranks_not_ok(*reducer, local_ok); !failed.empty()) {
    return Failure{"E-LINALG-SCHUR-SLAB-FACTORIZE",
                  fmt::format("SchurSlabSolver::factor: interior factorisation failed on "
                             "rank(s) {}; agreed across every rank before the Schur "
                             "complement's own allgather, so no rank is left waiting inside it",
                             format_rank_list(failed))};
  }

  // The periodic ring of row bands: predecessor and successor are fixed by
  // rank arithmetic alone (RowOwnership's own ascending, contiguous
  // convention, docs/dev/schur-slab-solver.md) -- no owner lookup needed.
  const int prev_rank = (rank - 1 + nranks) % nranks;
  const int next_rank = (rank + 1) % nranks;
  const std::int64_t g_total = 2LL * nx * nranks;
  const std::int64_t own_col_lo = 2LL * nx * rank;
  const std::int64_t prev_last_col_lo = 2LL * nx * prev_rank + nx;  // predecessor's LAST block
  const std::int64_t next_first_col_lo = 2LL * nx * next_rank;      // successor's FIRST block

  // This rank's own 2*nx ROWS of the g_total x g_total dense Schur
  // complement, zero everywhere except its own block (S_own) and the two
  // single-row-wide couplings to its periodic neighbours -- see
  // docs/dev/schur-slab-solver.md's "assembly by allgather" section for
  // why this concatenates, rank by rank, into the complete matrix with no
  // further bookkeeping.
  Eigen::MatrixXd row_block = Eigen::MatrixXd::Zero(2 * nx, g_total);
  row_block.block(0, own_col_lo, 2 * nx, 2 * nx) = s_own;
  for (int i = 0; i < nx; ++i) {
    row_block(i, prev_last_col_lo + i) += vert_val;       // first row <-> predecessor's last row
    row_block(nx + i, next_first_col_lo + i) += vert_val;  // last row <-> successor's first row
  }

  std::vector<double> send(static_cast<std::size_t>(2 * nx) * static_cast<std::size_t>(g_total));
  for (std::int64_t r = 0; r < 2 * nx; ++r) {
    for (std::int64_t c = 0; c < g_total; ++c) {
      send[static_cast<std::size_t>(r * g_total + c)] = row_block(r, c);
    }
  }
  std::vector<double> recv(send.size() * static_cast<std::size_t>(nranks));
  reducer->allgather(send.data(), send.size(), recv.data());

  Eigen::MatrixXd schur(g_total, g_total);
  for (std::int64_t r = 0; r < g_total; ++r) {
    for (std::int64_t c = 0; c < g_total; ++c) {
      schur(r, c) = recv[static_cast<std::size_t>(r * g_total + c)];
    }
  }

  // Each cross-rank coupling is written TWICE, independently, from both
  // sides -- the predecessor/successor rank that owns the OTHER row also
  // writes the transposed entry from its own row_block (the loop just
  // above, read from the OTHER rank's own call). Eigen::LDLT's default
  // (Lower) reads ONLY the lower triangle, so a bug that corrupts just the
  // upper-triangular copy of a coupling would otherwise never reach the
  // factorisation at all -- not a hypothetical: this is exactly what the
  // "wrap coupling dropped" mutation (build-evidence/mutations/
  // linalg.toml) was first hand-proved against the UNSYMMETRISED matrix
  // and found to NOT kill (the dropped upper-triangular copy was simply
  // never read). Averaging both triangles here turns ANY such asymmetry
  // (round-off aside, the two copies should already agree exactly, since
  // both are the same vert_val constant) into a detectable half-strength
  // coupling instead of an invisible one, with no cost to the correct case.
  schur = 0.5 * (schur + schur.transpose().eval());

  impl->schur_factor.compute(schur);
  if (impl->schur_factor.info() != Eigen::Success) {
    return Failure{"E-LINALG-SCHUR-SLAB-FACTORIZE",
                  "SchurSlabSolver::factor: dense LDLT factorization of the assembled Schur "
                  "complement failed"};
  }
  impl->g_total = g_total;
  impl->g_row_lo = own_col_lo;

  SchurSlabSolver solver;
  solver.impl_ = std::move(impl);
  return std::move(solver);
}

Result<void> SchurSlabSolver::solve(const DistVector& rhs_owned, DistVector& x_owned) const {
  FUSE_ASSERT(impl_ != nullptr, "SchurSlabSolver::solve called on a moved-from object");
  const std::int64_t n_total = impl_->local_rows * impl_->nx;
  FUSE_ASSERT(static_cast<std::int64_t>(rhs_owned.owned_size()) == n_total,
             "SchurSlabSolver::solve: rhs_owned owns {} entries, this row band owns {}",
             rhs_owned.owned_size(), n_total);
  FUSE_ASSERT(static_cast<std::int64_t>(x_owned.owned_size()) == n_total,
             "SchurSlabSolver::solve: x_owned owns {} entries, this row band owns {}",
             x_owned.owned_size(), n_total);

  const double* b = rhs_owned.data();
  double* x = x_owned.data();
  const int nx = impl_->nx;

  if (impl_->whole_grid) {
    const Eigen::VectorXd rhs = Eigen::Map<const Eigen::VectorXd>(b, n_total);
    const Eigen::VectorXd sol = impl_->whole_factor.solve(rhs);
    if (impl_->whole_factor.info() != Eigen::Success) {
      return Failure{"E-LINALG-SCHUR-SLAB-SOLVE",
                    "SchurSlabSolver::solve: whole-grid solve failed"};
    }
    std::copy(sol.data(), sol.data() + n_total, x);
    return Result<void>::success();
  }

  const std::int64_t local_rows = impl_->local_rows;
  const std::int64_t n_interior = impl_->n_interior;

  Eigen::VectorXd b_interior(n_interior);
  Eigen::VectorXd b_g_own(2 * nx);
  for (std::int64_t k = 0; k < local_rows; ++k) {
    for (int i = 0; i < nx; ++i) {
      const double value = b[k * nx + i];
      if (k == 0) {
        b_g_own(i) = value;
      } else if (k == local_rows - 1) {
        b_g_own(nx + i) = value;
      } else {
        b_interior((k - 1) * nx + i) = value;
      }
    }
  }

  // b_G' = b_G - A_GI * A_II^-1 * b_I = b_G - A_IG^T * (A_II^-1 b_I), fully
  // local to this rank -- A_GI only reaches this rank's OWN interior rows
  // (docs/dev/schur-slab-solver.md: an interior row's two neighbours are
  // both inside its own row band by definition).
  //
  // AGREED before the b_G allgather below, not returned on locally -- the
  // same stranding hazard factor()'s own interior-factorisation check has
  // (this file's own `ranks_not_ok` comment has the full account): a
  // solve that fails on only SOME ranks' own interior system must not
  // strand the others inside that collective.
  Eigen::VectorXd interior_y;
  bool local_ok = true;
  if (n_interior > 0) {
    interior_y = impl_->interior_factor.solve(b_interior);
    if (impl_->interior_factor.info() != Eigen::Success) local_ok = false;
  }
  if (const auto failed = ranks_not_ok(*impl_->reducer, local_ok); !failed.empty()) {
    return Failure{"E-LINALG-SCHUR-SLAB-SOLVE",
                  fmt::format("SchurSlabSolver::solve: interior solve for the reduced "
                             "interface rhs failed on rank(s) {}; agreed across every rank "
                             "before the reduced-rhs allgather, so no rank is left waiting "
                             "inside it",
                             format_rank_list(failed))};
  }
  const Eigen::VectorXd b_g_reduced_own =
      (n_interior > 0) ? Eigen::VectorXd(b_g_own - impl_->a_ig.transpose() * interior_y) : b_g_own;

  std::vector<double> send(static_cast<std::size_t>(2 * nx));
  std::copy(b_g_reduced_own.data(), b_g_reduced_own.data() + 2 * nx, send.begin());
  std::vector<double> recv(send.size() * static_cast<std::size_t>(impl_->reducer->size()));
  impl_->reducer->allgather(send.data(), send.size(), recv.data());

  const Eigen::VectorXd b_g_full = Eigen::Map<const Eigen::VectorXd>(recv.data(), impl_->g_total);
  const Eigen::VectorXd x_g_full = impl_->schur_factor.solve(b_g_full);
  if (impl_->schur_factor.info() != Eigen::Success) {
    return Failure{"E-LINALG-SCHUR-SLAB-SOLVE",
                  "SchurSlabSolver::solve: dense solve of the assembled Schur complement "
                  "failed"};
  }
  const Eigen::VectorXd x_g_own = x_g_full.segment(impl_->g_row_lo, 2 * nx);

  // x_I = A_II^-1 (b_I - A_IG x_G), fully local (A_IG only reaches this
  // rank's own two interface blocks, never a neighbour's).
  Eigen::VectorXd x_interior;
  if (n_interior > 0) {
    x_interior = impl_->interior_factor.solve(b_interior - impl_->a_ig * x_g_own);
    if (impl_->interior_factor.info() != Eigen::Success) {
      return Failure{"E-LINALG-SCHUR-SLAB-SOLVE",
                    "SchurSlabSolver::solve: interior back-substitution failed"};
    }
  }

  for (std::int64_t k = 0; k < local_rows; ++k) {
    for (int i = 0; i < nx; ++i) {
      if (k == 0) {
        x[k * nx + i] = x_g_own(i);
      } else if (k == local_rows - 1) {
        x[k * nx + i] = x_g_own(nx + i);
      } else {
        x[k * nx + i] = x_interior((k - 1) * nx + i);
      }
    }
  }
  return Result<void>::success();
}

}  // namespace fuse::linalg
