// fuse::linalg::SchurSlabSolver at tier 0. docs/dev/schur-slab-solver.md has
// the derivation; this file measures it the same way test/linalg/
// dist_vector_test.cpp and dist_sparse_matrix_test.cpp measure DistVector/
// DistSparseMatrix -- an in-process "fake" multi-partition split, here
// generalised past 2 partitions to 3 (requirement 2 needs both) via the
// `two_pass_collective` helper below, since `Reducer::allgather` (comm.hpp) is a genuine N-way collective and the
// existing FakePeerExchanger/TwoPartitionReducer pattern binds exactly 2
// peers. MPI cannot start in this sandbox (docs/dev/testing.md):
// test/parallel/schur_slab_mpi_test.cpp repeats the 2- and 3-rank
// properties over a real MPI_COMM_WORLD, built here, run on the host.
//
// The whole-grid reference every case checks against is
// `brusselator::periodic_laplacian` (examples/brusselator/
// brusselator_physics.hpp) itself, scaled and shifted exactly the way
// `DirectDiffusionSolver::factor` (examples/brusselator/species_core.hpp)
// does (minus its own physics-specific `extra_diagonal`, which
// `SchurSlabSolver`'s interface does not carry) -- the Toy's own in-process
// reference.

#include "fuse/linalg/schur_slab.hpp"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"

#include "brusselator_physics.hpp"

namespace {

using fuse::Result;
using fuse::linalg::DistVector;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::Reducer;
using fuse::linalg::RowOwnership;
using fuse::linalg::SchurSlabSolver;

// ---------------------------------------------------------------------
// The same contiguous row-band split src/backend/native_generators.cpp's
// own `rank_slice(n, rank, num_ranks)` computes (its own formula,
// reproduced here rather than linked: a `linalg` test pulling in
// `fuse::backend` for one arithmetic helper would be a heavier dependency
// than the three lines it saves). `ny` not divisible by `num_ranks`
// produces the extra-row-per-low-rank split `rank_slice` itself documents.
// ---------------------------------------------------------------------

std::pair<std::int64_t, std::int64_t> row_band(std::int64_t ny, int rank, int num_ranks) {
  if (num_ranks <= 1) return {0, ny};
  const std::int64_t base = ny / num_ranks;
  const std::int64_t extra = ny % num_ranks;
  const std::int64_t r = rank;
  const std::int64_t lo = (r < extra) ? r * (base + 1) : extra * (base + 1) + (r - extra) * base;
  const std::int64_t size = base + (r < extra ? 1 : 0);
  return {lo, lo + size};
}

RowOwnership row_ownership_for(std::int64_t ny, int num_ranks) {
  RowOwnership owners;
  owners.row_lo.resize(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) owners.row_lo[static_cast<std::size_t>(r)] = row_band(ny, r, num_ranks).first;
  return owners;
}

/// `A^-1 rhs` for the WHOLE grid, `A = I - coefficient * L_h` -- built
/// exactly the way `DirectDiffusionSolver::factor` builds it (scale the
/// whole sparse Laplacian by `-coefficient`, then add 1.0 to every
/// diagonal), from `brusselator::periodic_laplacian` itself: the Toy's own
/// reference, never reimplemented.
Eigen::VectorXd whole_grid_reference_solve(int nx, int ny, double coefficient,
                                           const Eigen::VectorXd& rhs) {
  brusselator::Sparse laplacian = brusselator::periodic_laplacian(nx, ny);
  Eigen::SparseMatrix<double> matrix = (-coefficient) * laplacian;
  const int n = nx * ny;
  for (int i = 0; i < n; ++i) matrix.coeffRef(i, i) += 1.0;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
  factor.compute(matrix);
  EXPECT_EQ(factor.info(), Eigen::Success);
  return factor.solve(rhs);
}

// ---------------------------------------------------------------------
// An N-way in-process "fake" allgather: every rank's own call records its
// `send` into a SHARED cache slot (by rank), then reads every slot back
// into `recv`. Sequential (no threads, docs/dev/testing.md: MPI cannot
// start here), so a single pass leaves later ranks' slots stale for an
// EARLIER rank's own call -- `two_pass_collective` below runs `make_call`
// for every rank ONCE to populate every slot (discarding that pass's own,
// necessarily-incomplete, `recv`), then again now that every slot holds
// the value this rank's own deterministic computation would produce
// either way, so the second pass's `recv` is the true concatenation on
// EVERY call. Valid because `SchurSlabSolver::factor`/`::solve` are pure
// functions of (this rank's own already-fixed nx/ny/j_lo/j_hi/coefficient,
// and for `solve`, this rank's own already-fixed rhs) -- nothing a
// warm-up pass's call could leave mutated that the real pass would see.
// ---------------------------------------------------------------------

// Keyed by `n_each`: SchurSlabSolver now performs TWO DISTINCT collectives
// per `factor()`/`solve()` call (the small, 1-double-per-rank AGREEMENT
// round `ranks_not_ok` -- src/linalg/schur_slab.cpp -- which keeps the
// row-band-size refusal from stranding a peer rank inside the allgather
// below it; then the much-larger Schur-/rhs-assembly round). A single shared cache vector
// would have one round's own slot overwritten by the other's differently-
// sized payload mid-call; keying by `n_each` (which differs by
// CONSTRUCTION between the agreement round and the assembly round, for
// any nx this file uses) keeps them in separate slots instead.
class RingAllgatherReducer : public Reducer {
 public:
  using Cache = std::map<std::size_t, std::vector<std::vector<double>>>;

  RingAllgatherReducer(int rank, int size, std::shared_ptr<Cache> caches)
      : rank_(rank), size_(size), caches_(std::move(caches)) {}

  // Not exercised by SchurSlabSolver (it uses only allgather/rank/size);
  // implemented as the one-participant identity so an accidental call
  // never silently combines across ranks.
  [[nodiscard]] double sum(double local) const override { return local; }
  void sum(double* inout, std::size_t n) const override { (void)inout; (void)n; }
  [[nodiscard]] double max(double local) const override { return local; }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return size_; }

  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    ++allgather_calls_;
    auto& cache = (*caches_)[n_each];
    if (cache.empty()) cache.resize(static_cast<std::size_t>(size_));
    cache[static_cast<std::size_t>(rank_)].assign(send, send + n_each);
    for (int r = 0; r < size_; ++r) {
      const auto& slot = cache[static_cast<std::size_t>(r)];
      if (slot.size() == n_each) {
        std::copy(slot.begin(), slot.end(), recv + static_cast<std::size_t>(r) * n_each);
      } else {
        std::fill(recv + static_cast<std::size_t>(r) * n_each,
                  recv + static_cast<std::size_t>(r + 1) * n_each, 0.0);  // warm-up pass only
      }
    }
  }

  /// How many times THIS rank's own `allgather()` has been called --
  /// SchurSlabSolverRefusal's own stranding regressions compare this
  /// across ranks directly: two ranks that disagree about whether a
  /// refusal fires must still agree on how many collectives each one
  /// entered, or one of them was left waiting in a REAL MPI run.
  [[nodiscard]] int allgather_call_count() const { return allgather_calls_; }

 private:
  int rank_;
  int size_;
  std::shared_ptr<Cache> caches_;
  mutable int allgather_calls_ = 0;
};

/// Runs `make_call(rank)` for every rank in `[0, nranks)` TWICE -- see the
/// class comment above for why the SECOND pass's return values are the
/// ones callers should keep.
template <class F>
auto two_pass_collective(int nranks, F&& make_call) -> std::vector<std::invoke_result_t<F, int>> {
  for (int r = 0; r < nranks; ++r) (void)make_call(r);  // warm-up: discard
  std::vector<std::invoke_result_t<F, int>> out;
  out.reserve(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) out.push_back(make_call(r));  // real
  return out;
}

std::vector<std::shared_ptr<RingAllgatherReducer>> make_ring_reducers(int nranks) {
  auto caches = std::make_shared<RingAllgatherReducer::Cache>();
  std::vector<std::shared_ptr<RingAllgatherReducer>> reducers;
  reducers.reserve(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) reducers.push_back(std::make_shared<RingAllgatherReducer>(r, nranks, caches));
  return reducers;
}

// ---------------------------------------------------------------------
// Requirement 1: nranks == 1 is BIT-IDENTICAL to the whole-grid reference.
// ---------------------------------------------------------------------

TEST(SchurSlabSolverOneRank, BitIdenticalToWholeGridSparseCholesky) {
  constexpr int nx = 8, ny = 8;
  constexpr double coefficient = 0.37;
  const RowOwnership owners = row_ownership_for(ny, 1);

  std::mt19937 rng(5u);
  std::uniform_real_distribution<double> u(-4.0, 4.0);
  Eigen::VectorXd rhs(nx * ny);
  for (int i = 0; i < nx * ny; ++i) rhs(i) = u(rng);

  auto solver = SchurSlabSolver::factor(nx, ny, 0, ny, coefficient, owners, make_serial_reducer());
  ASSERT_TRUE(solver.ok()) << solver.error().message;

  DistVector rhs_owned(nx * ny, make_serial_exchanger(), make_serial_reducer());
  std::copy(rhs.data(), rhs.data() + nx * ny, rhs_owned.data());
  DistVector x_owned(nx * ny, make_serial_exchanger(), make_serial_reducer());
  ASSERT_TRUE(solver.value().solve(rhs_owned, x_owned).ok());

  const Eigen::VectorXd expect = whole_grid_reference_solve(nx, ny, coefficient, rhs);
  for (int i = 0; i < nx * ny; ++i) {
    EXPECT_EQ(x_owned.data()[i], expect(i)) << "entry " << i;
  }
}

// ---------------------------------------------------------------------
// Requirement 2: 2 and 3 in-process partitions, including ny not
// divisible by nranks (17 over 3), agree with the whole-grid reference to
// a MEASURED bound (round-off at double precision) -- and a dedicated
// small grid (nx=4, ny=8, 2 ranks) where n_interior happens to equal 2*nx,
// used again by the "transposition" mutation row
// (build-evidence/mutations/linalg.toml) because that coincidence is what
// keeps the mutated A_IG/A_GI swap dimensionally valid instead of
// crashing on a shape mismatch (docs/dev/schur-slab-solver.md's own note
// on that mutation's design).
// ---------------------------------------------------------------------

/// Builds `nranks` SchurSlabSolver objects over the in-process ring
/// allgather, solves the SAME global rhs on every rank, and checks each
/// rank's own owned entries against the whole-grid reference to `tol`.
void check_multi_partition_solve(int nx, int ny, int nranks, const Eigen::VectorXd& rhs_global,
                                 double tol) {
  const RowOwnership owners = row_ownership_for(ny, nranks);
  const auto reducers = make_ring_reducers(nranks);

  const auto solvers = two_pass_collective(nranks, [&](int r) {
    const auto [lo, hi] = row_band(ny, r, nranks);
    return SchurSlabSolver::factor(nx, ny, lo, hi, /*coefficient=*/0.6, owners, reducers[static_cast<std::size_t>(r)]);
  });
  for (int r = 0; r < nranks; ++r) {
    ASSERT_TRUE(solvers[static_cast<std::size_t>(r)].ok())
        << "rank " << r << ": " << solvers[static_cast<std::size_t>(r)].error().message;
  }

  std::vector<DistVector> rhs_owned;
  rhs_owned.reserve(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) {
    const auto [lo, hi] = row_band(ny, r, nranks);
    const std::int64_t n = (hi - lo) * nx;
    rhs_owned.emplace_back(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
    std::copy(rhs_global.data() + lo * nx, rhs_global.data() + hi * nx, rhs_owned.back().data());
  }

  const auto x_owned = two_pass_collective(nranks, [&](int r) {
    const auto [lo, hi] = row_band(ny, r, nranks);
    DistVector x(static_cast<std::size_t>((hi - lo) * nx), make_serial_exchanger(), make_serial_reducer());
    const auto result = solvers[static_cast<std::size_t>(r)].value().solve(rhs_owned[static_cast<std::size_t>(r)], x);
    EXPECT_TRUE(result.ok()) << "rank " << r << ": " << (result.ok() ? "" : result.error().message);
    return x;
  });

  const Eigen::VectorXd expect = whole_grid_reference_solve(nx, ny, 0.6, rhs_global);
  for (int r = 0; r < nranks; ++r) {
    const auto [lo, hi] = row_band(ny, r, nranks);
    for (std::int64_t k = 0; k < hi - lo; ++k) {
      for (int i = 0; i < nx; ++i) {
        const double got = x_owned[static_cast<std::size_t>(r)].data()[k * nx + i];
        const double want = expect((lo + k) * nx + i);
        EXPECT_NEAR(got, want, tol) << "rank " << r << " local row " << k << " i " << i;
      }
    }
  }
}

// DECLARATION ORDER MATTERS for this one case, so it stays FIRST among the
// multi-partition tests: nx=4, ny=8 gives n_interior (== 8) == 2*nx
// (== 8), the ONE shape at which the "transposition" mutation
// (build-evidence/mutations/linalg.toml, A_IG swapped for
// A_IG.transpose() in the back-substitution step) stays dimensionally
// valid -- MEASURED to be a mathematical no-op there (A_IG is provably
// `vert_val * I_8` at local_rows==4, hence its own transpose, docs/dev/
// schur-slab-solver.md's note on that mutation's design has the proof), so
// this test's own PASS is expected under that mutation too, not evidence
// against it. GoogleTest runs tests in declaration order by default (no
// --gtest_shuffle here); the kill is the VERY NEXT case
// (TwoPartitions16x16, n_interior=96 != 2*nx=32), which the mutation
// reaches as a genuine Eigen dimension assertion that aborts the whole
// process -- declaring this case first is what makes that sequence (one
// unaffected pass, then a crash) the reproducible hand-proof record rather
// than an accident of file order.
TEST(SchurSlabSolverMultiPartition, FourByEightTwoPartitionsAgreesWithWholeGridReference) {
  constexpr int nx = 4, ny = 8;
  std::mt19937 rng(31u);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  Eigen::VectorXd rhs(nx * ny);
  for (int i = 0; i < nx * ny; ++i) rhs(i) = u(rng);

  // MEASURED: max absolute discrepancy 2.5e-16, max relative 1.3e-13
  // against entries as large as 9.3e-2.
  check_multi_partition_solve(nx, ny, /*nranks=*/2, rhs, /*tol=*/1e-12);
}

TEST(SchurSlabSolverMultiPartition, TwoPartitions16x16AgreesWithWholeGridReference) {
  constexpr int nx = 16, ny = 16;
  std::mt19937 rng(13u);
  std::uniform_real_distribution<double> u(-3.0, 3.0);
  Eigen::VectorXd rhs(nx * ny);
  for (int i = 0; i < nx * ny; ++i) rhs(i) = u(rng);

  // MEASURED (this exact case, built Debug, AppleClang 21/arm64): max
  // absolute entry-wise discrepancy 8.4e-16, max RELATIVE discrepancy
  // 6.5e-14 against entries as large as 7.6e-2 -- the Schur-complement path
  // and the whole-grid SparseCholesky reach the same answer through
  // completely different factorisations (one sparse interior factor per
  // rank plus a dense LDLT of a 64x64 assembled matrix, vs. one sparse
  // factor of the whole 256x256 matrix) and round-off at double precision,
  // not a discretisation difference, is the entire discrepancy. 1e-12
  // absolute is ~1000x the measured value -- comfortably clear of
  // round-off noise on a different machine, nowhere near the 1e-9+ scale a
  // genuine Schur-assembly bug (a dropped term, a wrong sign) would
  // produce -- the same margin dist_sparse_matrix_test.cpp's own 16x16
  // periodic-Laplacian case uses for the identical reason.
  check_multi_partition_solve(nx, ny, /*nranks=*/2, rhs, /*tol=*/1e-12);
}

TEST(SchurSlabSolverMultiPartition, ThreePartitionsNySeventeenNotDivisibleByThree) {
  constexpr int nx = 5, ny = 17;  // 17 % 3 != 0: row bands of 6, 6, 5 rows.
  std::mt19937 rng(29u);
  std::uniform_real_distribution<double> u(-3.0, 3.0);
  Eigen::VectorXd rhs(nx * ny);
  for (int i = 0; i < nx * ny; ++i) rhs(i) = u(rng);

  // MEASURED: max absolute discrepancy 4.7e-15, max relative 2.4e-14
  // against entries as large as 2.9e-1 -- same reasoning as the 16x16
  // case's own comment just above.
  check_multi_partition_solve(nx, ny, /*nranks=*/3, rhs, /*tol=*/1e-12);
}

// ---------------------------------------------------------------------
// Requirement 3: the periodic wrap. 2 partitions, rhs nonzero only in the
// LAST row of the LAST row band -- the only path from that forcing to the
// FIRST row band's own FIRST row is the periodic wrap between the last
// row band and the first. A mutation that drops the wrap coupling (the
// "wrap coupling dropped" mutation row) must fail this.
// ---------------------------------------------------------------------

TEST(SchurSlabSolverPeriodicWrap, ForcingOnlyOnLastRowReachesFirstRowOnlyThroughTheWrap) {
  constexpr int nx = 6, ny = 10;
  constexpr int nranks = 2;
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nx * ny);
  // Last row of the grid (global j = ny-1), forced at one column only --
  // owned by rank 1 (row band [5, 10)), its own LAST local row.
  rhs((ny - 1) * nx + 2) = 7.0;

  // MEASURED: max absolute discrepancy 1.5e-16, max relative 1.4e-15
  // against entries as large as 1.9e-1.
  check_multi_partition_solve(nx, ny, nranks, rhs, /*tol=*/1e-12);

  // The point of this case: rank 0's FIRST local row (global row 0) is
  // reachable from the forcing ONLY through the periodic wrap (row 0's
  // "previous" neighbour is global row ny-1, owned by rank 1) -- confirm
  // it is actually nonzero in the reference, so the check above is not
  // vacuously passing on an all-zero row.
  const Eigen::VectorXd expect = whole_grid_reference_solve(nx, ny, 0.6, rhs);
  bool any_nonzero_first_row = false;
  for (int i = 0; i < nx; ++i) {
    if (std::fabs(expect(i)) > 1e-8) any_nonzero_first_row = true;
  }
  EXPECT_TRUE(any_nonzero_first_row)
      << "the reference's own first row is all zero, which would make this test unable to "
        "distinguish a dropped wrap from a correct one";
}

// ---------------------------------------------------------------------
// Requirement 4: a row band of fewer than 2 rows refuses, E-LINALG-SCHUR-SLAB.
// ---------------------------------------------------------------------

TEST(SchurSlabSolverRefusal, OneRowBandRefusesWithoutFactorizing) {
  constexpr int nx = 4;
  constexpr std::int64_t ny = 10;
  // 2 participants, row_lo = {0, 9}: rank 1 owns exactly row 9 -- one row.
  // Calling factor() for rank 1 ALONE (never rank 0) and still getting the
  // refusal is the point: row_ownership already names every participant's
  // own band, so rank 1 needs no reply from rank 0 to know rank 1's own
  // band is undersized.
  RowOwnership owners{{0, 9}};
  auto caches = std::make_shared<RingAllgatherReducer::Cache>();
  auto result = SchurSlabSolver::factor(
      nx, ny, /*j_lo=*/9, /*j_hi=*/10, /*coefficient=*/0.5, owners,
      std::make_shared<RingAllgatherReducer>(1, 2, caches));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-SCHUR-SLAB");
}

TEST(SchurSlabSolverRefusal, EmptyRowBandAlsoRefuses) {
  constexpr int nx = 4;
  constexpr std::int64_t ny = 10;
  // 3 participants, row_lo = {0, 5, 10}: rank 2 would own [10, 10) -- empty.
  RowOwnership owners{{0, 5, 10}};
  auto caches = std::make_shared<RingAllgatherReducer::Cache>();
  auto result = SchurSlabSolver::factor(
      nx, ny, /*j_lo=*/10, /*j_hi=*/10, /*coefficient=*/0.5, owners,
      std::make_shared<RingAllgatherReducer>(2, 3, caches));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-SCHUR-SLAB");
}

// ---------------------------------------------------------------------
// HOST-FOUND REGRESSION: factor()'s row-band-size refusal
// must not be decided from THIS rank's own `local_rows` alone and return
// immediately: that is a collective path (the Schur-assembly allgather)
// returning on a condition computed from only ONE rank's own data, the
// standing rule's own forbidden pattern. On the host, rank 1's single-row
// band refused and returned without ever calling that allgather, while
// rank 0's own fine band sailed past the (per-rank) check and reached it
// ALONE -- MPICH's own "Message truncated" in MPI_Allgather is exactly
// one rank waiting inside a collective nobody else entered. The refusal is
// decided from `row_ownership` alone (every rank already
// holds the identical array), which needs NO collective at all, so every
// rank reaches the SAME conclusion with the SAME (zero) number of
// allgather calls either way -- checked directly below, not inferred.
// ---------------------------------------------------------------------

TEST(SchurSlabSolverRefusal, OneRowBandRefusesOnEveryRankWithoutStrandingAnyoneInACollective) {
  constexpr int nx = 4;
  constexpr std::int64_t ny = 9;  // rank 0: 8 rows; rank 1: 1 row.
  const RowOwnership owners{{0, 8}};
  const auto reducers = make_ring_reducers(2);

  auto result0 = SchurSlabSolver::factor(nx, ny, /*j_lo=*/0, /*j_hi=*/8, 0.5, owners, reducers[0]);
  auto result1 = SchurSlabSolver::factor(nx, ny, /*j_lo=*/8, /*j_hi=*/9, 0.5, owners, reducers[1]);

  ASSERT_FALSE(result0.ok()) << "rank 0's own band is fine, but EVERY rank must refuse together";
  EXPECT_EQ(result0.error().code, "E-LINALG-SCHUR-SLAB");
  ASSERT_FALSE(result1.ok());
  EXPECT_EQ(result1.error().code, "E-LINALG-SCHUR-SLAB");

  // Neither rank may be left waiting inside a collective the other never
  // reaches: both made the SAME number of allgather calls -- zero, since
  // this refusal needs no collective at all.
  EXPECT_EQ(reducers[0]->allgather_call_count(), reducers[1]->allgather_call_count());
  EXPECT_EQ(reducers[0]->allgather_call_count(), 0);
}

// ---------------------------------------------------------------------
// The OTHER half of the same host diagnosis: a factorisation's own
// numerical outcome CANNOT be predicted from row_ownership (unlike the
// row-band-size case above), so it genuinely needs the collective
// agreement `ranks_not_ok` (src/linalg/schur_slab.cpp) performs. Proven
// with a LEGITIMATE, non-physical-but-interface-valid input: nx=1
// collapses the horizontal stencil into the diagonal (periodic self-loop,
// `(i+-1) mod 1 == i`), giving an EFFECTIVE interior diagonal of
// `1 + 2*coefficient*ny^2` -- `coefficient = -1/(2*ny^2)` makes that
// EXACTLY zero, leaving an interior tridiagonal block with zero diagonal
// and 0.5 off-diagonal entries, a textbook singular pattern (hand-verified
// determinant 0, confirmed by Eigen's own `info() == NumericalIssue`).
// Nothing in this interface asserts the two ranks' own `coefficient`
// arguments agree, so handing rank 1 this pathological value while rank 0
// keeps an ordinary one is a legitimate way to make ONE rank's own
// interior factorisation fail while the other's succeeds, with no
// internal access at all.
//
// MEASURED, not merely algebraic: a coefficient derived to make the
// interior's own diagonal `1 + 2*coefficient*(xx+yy)` exactly zero only
// actually LANDS on exactly zero in double-precision arithmetic when
// `2*(xx+yy)` is itself an exact power of 2 (otherwise the division and
// the back-multiplication each round, leaving a ~1e-17 residual that
// Eigen's SimplicialLDLT silently accepts as Success -- tried first with
// nx=1 ny=10, confirmed NOT to trigger a failure, kept here as the reason
// nx=ny=8 was chosen instead: 8^2+8^2=128=2^7 exactly, so
// coefficient=-1/256 zeroes the diagonal bit-for-bit, and Eigen's own
// info() reports NumericalIssue for the resulting interior block,
// confirmed by running it before writing this assertion).
// ---------------------------------------------------------------------

TEST(SchurSlabSolverRefusal,
    InteriorFactorizationFailureOnOneRankIsAgreedWithoutStrandingAnyoneInACollective) {
  constexpr int nx = 8;
  constexpr std::int64_t ny = 8;  // both ranks own 4 rows -> n_interior = 2*nx = 16 each.
  const RowOwnership owners{{0, 4}};
  constexpr double singular_coefficient = -1.0 / (2.0 * (double(nx) * nx + double(ny) * ny));

  // Pre-seed the agreement round's own cache slots directly with the
  // FINAL vote each rank will cast (rank 0 ok, rank 1 not) -- both votes
  // are pure functions of that rank's OWN fixed inputs (never of its
  // peer's), so this is not "cheating": it sidesteps the two_pass_collective
  // helper's own sequential-warm-up ordering (which cannot, by itself,
  // settle a round whose OUTCOME determines whether a SECOND, differently-
  // sized collective is even attempted) without changing what either rank
  // computes.
  auto caches = std::make_shared<RingAllgatherReducer::Cache>();
  (*caches)[1] = {{0.0}, {1.0}};  // n_each == 1 is the agreement round's own key
  auto reducer0 = std::make_shared<RingAllgatherReducer>(0, 2, caches);
  auto reducer1 = std::make_shared<RingAllgatherReducer>(1, 2, caches);

  auto result0 = SchurSlabSolver::factor(nx, ny, 0, 4, /*coefficient=*/0.5, owners, reducer0);
  auto result1 = SchurSlabSolver::factor(nx, ny, 4, 8, singular_coefficient, owners, reducer1);

  ASSERT_FALSE(result0.ok()) << "rank 0's OWN interior factored fine, but it must still agree to fail";
  EXPECT_EQ(result0.error().code, "E-LINALG-SCHUR-SLAB-FACTORIZE");
  ASSERT_FALSE(result1.ok());
  EXPECT_EQ(result1.error().code, "E-LINALG-SCHUR-SLAB-FACTORIZE");

  // Both ranks made exactly ONE allgather call (the agreement round
  // itself); neither reached the much larger Schur-assembly round the
  // agreement exists to guard.
  EXPECT_EQ(reducer0->allgather_call_count(), reducer1->allgather_call_count());
  EXPECT_EQ(reducer0->allgather_call_count(), 1);
}

// ---------------------------------------------------------------------
// Reducer::allgather itself (comm.hpp):
// the serial identity, and the N-way in-process fake used by every test
// above.
// ---------------------------------------------------------------------

TEST(LinalgComm, SerialReducerAllgatherIsTheOneParticipantIdentity) {
  auto r = make_serial_reducer();
  const std::vector<double> send{1.0, -2.0, 3.5};
  std::vector<double> recv(3, 0.0);
  r->allgather(send.data(), send.size(), recv.data());
  EXPECT_EQ(recv, send);
}

TEST(LinalgComm, RingAllgatherReducerConcatenatesInRankOrder) {
  constexpr int nranks = 3;
  const auto reducers = make_ring_reducers(nranks);
  const std::vector<std::vector<double>> per_rank_send{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};

  const auto recvs = two_pass_collective(nranks, [&](int r) {
    std::vector<double> recv(2 * static_cast<std::size_t>(nranks), -1.0);
    reducers[static_cast<std::size_t>(r)]->allgather(per_rank_send[static_cast<std::size_t>(r)].data(), 2,
                                                     recv.data());
    return recv;
  });

  const std::vector<double> expect{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  for (int r = 0; r < nranks; ++r) {
    EXPECT_EQ(recvs[static_cast<std::size_t>(r)], expect) << "rank " << r;
  }
}

}  // namespace
