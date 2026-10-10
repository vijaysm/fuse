// Physics track (closing the Grid track's own solver seam): UCore/
// VDiffusionCore decomposed over a component's own ranks, solver
// selectable (direct|schur|pcg|gmres), verified against the single-rank
// (`num_ranks()==1`) reference -- the SAME `UCore`/`VDiffusionCore`
// classes, not a reimplementation. The in-process N-partition simulation
// below runs REAL OS THREADS, one per partition, sharing a hand-written
// barrier-based `fuse::linalg::Reducer`/`GhostExchanger` pair: unlike the
// Grid track's own tier-0 two-partition tests (test/verification/
// brusselator_grid_decomposition_test.cpp, test/linalg/schur_slab_test.cpp's
// own "two-pass" replay trick), `UCore::step()` is itself STATEFUL across
// collectives -- its own reaction fixed-point loop DECIDES how many
// solves to run from a collective's own (`Reducer::max`) return value, so
// a sequential "run every rank once, discard, run again now the cache is
// warm" replay would let an UNSYNCHRONIZED warm-up-pass value steer one
// rank's own loop down a DIFFERENT iteration count than the real pass
// reproduces -- a hazard distinct from, but in the same family as, this
// track's own standing lesson ("a collective path must never return on a
// locally computed condition"). Real blocking barriers (one `std::thread`
// per partition, `std::condition_variable`-gated) remove the hazard
// entirely: every participant's `sum`/`max`/`allgather`/ghost-`update`
// call blocks until every OTHER participant named in it has arrived,
// exactly as a real `MPI_Allreduce`/`MPI_Allgather`/`MPI_Alltoallv` does,
// so every rank's own control flow sees the TRUE combined value the
// moment it asks for one -- never a stale or partial one.

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <fuse/backend/native_backend.hpp>

#include "brusselator_physics.hpp"
#include "species_core.hpp"

namespace {

using brusselator::NativeGrid;
using brusselator::PhysicsConfig;
using brusselator::SolverKind;
using brusselator::UCore;
using brusselator::VDiffusionCore;
using brusselator::Vector;
using fuse::linalg::GhostExchanger;
using fuse::linalg::GhostPattern;
using fuse::linalg::Reducer;

// ---------------------------------------------------------------------
// Barrier primitives: an in-process stand-in for a REAL N-way MPI
// collective, blocking (condition-variable-gated) rather than replayed,
// for exactly the reason this file's own header comment states.
// ---------------------------------------------------------------------

/// A blocking N-way scalar reduction (sum or max, chosen by `combine_fn`
/// at the call site, not by this class -- every participant's own code
/// calls its reducer the same way at the same logical point, by
/// construction, so the operation never differs between participants at
/// a given round). One object serves EVERY round of the run (the
/// `generation` counter below distinguishes rounds); a participant that
/// calls this object again before the PREVIOUS round's last participant
/// has arrived is a caller bug this does not need to guard against (the
/// participants here are this file's own thread bodies, which never do).
class ScalarBarrier {
 public:
  explicit ScalarBarrier(int size) : size_(size), local_(static_cast<std::size_t>(size), 0.0) {}

  template <class Combine>
  double combine(int rank, double local, double identity, Combine&& combine_fn) {
    std::unique_lock<std::mutex> lk(mu_);
    const int gen = generation_;
    local_[static_cast<std::size_t>(rank)] = local;
    if (++arrived_ == size_) {
      double acc = identity;
      for (double v : local_) acc = combine_fn(acc, v);
      combined_ = acc;
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return generation_ != gen; });
    }
    return combined_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int size_;
  int arrived_ = 0;
  int generation_ = 0;
  std::vector<double> local_;
  double combined_ = 0.0;
};

/// A blocking N-way gather: every participant's own `n_each`-sized chunk
/// (the SAME size for every participant -- this file's own tests pick
/// grids whose row count divides evenly by the partition count for
/// exactly this reason, the same simplifying choice
/// `fuse::linalg::make_mpi_exchanger`'s own 2-rank tier-0 sibling,
/// `TwoRankMpiHaloExchanger` (test/parallel/brusselator_grid_mpi_test.cpp),
/// is not bound by only because MPI's own `MPI_Alltoallv` tolerates
/// uneven counts) becomes `recv[r*n_each .. (r+1)*n_each)` for every `r`,
/// in rank order -- `fuse::linalg::Reducer::allgather`'s own contract,
/// and `BarrierGhostExchanger`'s own underlying primitive below (a ghost
/// update is "gather everyone's current owned block, then pick out the
/// entries this rank's own pattern names").
class GatherBarrier {
 public:
  GatherBarrier(int size, std::size_t n_each)
      : size_(size), n_each_(n_each), local_(static_cast<std::size_t>(size) * n_each, 0.0) {}

  void combine(int rank, const double* send, double* recv) {
    std::unique_lock<std::mutex> lk(mu_);
    const int gen = generation_;
    std::copy(send, send + n_each_, local_.begin() + static_cast<std::ptrdiff_t>(
                                        static_cast<std::size_t>(rank) * n_each_));
    if (++arrived_ == size_) {
      combined_ = local_;
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return generation_ != gen; });
    }
    std::copy(combined_.begin(), combined_.end(), recv);
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int size_;
  std::size_t n_each_;
  int arrived_ = 0;
  int generation_ = 0;
  std::vector<double> local_, combined_;
};

/// `fuse::linalg::Reducer` over `ScalarBarrier`s (one for `sum`, one for
/// `max`) and a `GatherBarrier` (for `allgather`, keyed by `n_each` since
/// `SchurSlabSolver`'s own two collectives per `factor()`/`solve()` use
/// two different sizes -- the same reason test/linalg/schur_slab_test.cpp's
/// own `RingAllgatherReducer` keys its cache by `n_each`). `sum(double*,
/// n)` (the vectorised overload) is not exercised by anything this file
/// builds (neither `UCore`/`VDiffusionCore` nor the Krylov solvers call
/// it) -- implemented as `n` independent scalar reductions rather than
/// left unimplemented, since a correct (if unused) implementation costs
/// nothing here and documents the intended semantics for a future caller.
class BarrierReducer final : public Reducer {
 public:
  BarrierReducer(int rank, int size, std::shared_ptr<ScalarBarrier> sum_barrier,
                std::shared_ptr<ScalarBarrier> max_barrier,
                std::shared_ptr<std::mutex> gather_registry_mu,
                std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> gather_registry)
      : rank_(rank),
        size_(size),
        sum_barrier_(std::move(sum_barrier)),
        max_barrier_(std::move(max_barrier)),
        gather_registry_mu_(std::move(gather_registry_mu)),
        gather_registry_(std::move(gather_registry)) {}

  [[nodiscard]] double sum(double local) const override {
    return sum_barrier_->combine(rank_, local, 0.0, std::plus<double>());
  }
  void sum(double* inout, std::size_t n) const override {
    for (std::size_t k = 0; k < n; ++k) inout[k] = sum(inout[k]);
  }
  [[nodiscard]] double max(double local) const override {
    return max_barrier_->combine(rank_, local, -std::numeric_limits<double>::infinity(),
                                 [](double a, double b) { return std::max(a, b); });
  }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return size_; }
  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    std::shared_ptr<GatherBarrier> barrier;
    {
      std::lock_guard<std::mutex> lk(*gather_registry_mu_);
      auto& slot = (*gather_registry_)[n_each];
      if (!slot) slot = std::make_shared<GatherBarrier>(size_, n_each);
      barrier = slot;
    }
    barrier->combine(rank_, send, recv);
  }

 private:
  int rank_, size_;
  std::shared_ptr<ScalarBarrier> sum_barrier_, max_barrier_;
  std::shared_ptr<std::mutex> gather_registry_mu_;
  std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> gather_registry_;
};

/// `fuse::linalg::GhostExchanger` built from the SAME `GatherBarrier`
/// registry `BarrierReducer::allgather` uses (keyed by `owned_n`, which
/// -- unlike `allgather`'s own two distinct Schur sizes -- is constant
/// for a given `KrylovDiffusionSolver` instance's whole run, so one key
/// suffices): every participant submits its CURRENT owned block, and
/// once all have arrived this rank reads back the full concatenation and
/// picks out whichever entries `pattern`'s own `(owner_rank,
/// remote_local_index)` pairs name -- a ghost update IS a gather
/// followed by a local pick, nothing more (`GhostExchangerFactory`'s own
/// comment, species_core.hpp, says why this pattern must come from the
/// OPERATOR's own derived `ghost_pattern()`, not `NativeGrid`'s).
class BarrierGhostExchanger final : public GhostExchanger {
 public:
  BarrierGhostExchanger(GhostPattern pattern, int rank, std::size_t owned_n,
                        std::shared_ptr<std::mutex> registry_mu,
                        std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> registry,
                        int size)
      : pattern_(std::move(pattern)),
        rank_(rank),
        owned_n_(owned_n),
        registry_mu_(std::move(registry_mu)),
        registry_(std::move(registry)),
        size_(size) {}

  [[nodiscard]] std::size_t ghost_size() const override { return pattern_.owner_rank.size(); }

  [[nodiscard]] fuse::Result<void> update(const double* owned, std::size_t owned_n,
                                          double* ghosts, std::size_t ghost_n) const override {
    if (owned_n != owned_n_ || ghost_n != pattern_.owner_rank.size()) {
      return fuse::Failure{"E-TEST-HALO-SIZE", "BarrierGhostExchanger: size mismatch"};
    }
    std::shared_ptr<GatherBarrier> barrier;
    {
      std::lock_guard<std::mutex> lk(*registry_mu_);
      auto& slot = (*registry_)[owned_n_];
      if (!slot) slot = std::make_shared<GatherBarrier>(size_, owned_n_);
      barrier = slot;
    }
    std::vector<double> combined(owned_n_ * static_cast<std::size_t>(size_));
    barrier->combine(rank_, owned, combined.data());
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern_.owner_rank[g]);
      const auto local = static_cast<std::size_t>(pattern_.remote_local_index[g]);
      ghosts[g] = combined[owner * owned_n_ + local];
    }
    return fuse::Result<void>::success();
  }

 private:
  GhostPattern pattern_;
  int rank_;
  std::size_t owned_n_;
  std::shared_ptr<std::mutex> registry_mu_;
  std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> registry_;
  int size_;
};

/// Everything shared across one run's `num_ranks` partitions: one pair of
/// `ScalarBarrier`s per rank-group (every partition's `BarrierReducer`
/// shares the SAME pair, so a `sum`/`max` call on any partition finds
/// every peer at the matching round) and one gather registry (shared by
/// BOTH `allgather` -- Schur's own two collectives -- and ghost updates,
/// keyed by size so the two never collide).
struct BarrierGroup {
  explicit BarrierGroup(int size)
      : sum_barrier(std::make_shared<ScalarBarrier>(size)),
        max_barrier(std::make_shared<ScalarBarrier>(size)),
        gather_registry_mu(std::make_shared<std::mutex>()),
        gather_registry(std::make_shared<std::map<std::size_t, std::shared_ptr<GatherBarrier>>>()) {}

  std::shared_ptr<ScalarBarrier> sum_barrier, max_barrier;
  std::shared_ptr<std::mutex> gather_registry_mu;
  std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> gather_registry;

  [[nodiscard]] std::shared_ptr<BarrierReducer> make_reducer(int rank, int size) const {
    return std::make_shared<BarrierReducer>(rank, size, sum_barrier, max_barrier, gather_registry_mu,
                                            gather_registry);
  }
  [[nodiscard]] brusselator::GhostExchangerFactory make_exchanger_factory(int rank, int size) const {
    auto mu = gather_registry_mu;
    auto reg = gather_registry;
    return [rank, size, mu, reg](const GhostPattern& pattern, std::size_t owned_n) {
      return std::static_pointer_cast<const GhostExchanger>(
          std::make_shared<BarrierGhostExchanger>(pattern, rank, owned_n, mu, reg, size));
    };
  }
};

/// Builds `num_ranks` `NativeGrid`s, one per partition, sharing a
/// SEPARATE `fuse::native::NativeMeshBackend` per rank (the backend
/// itself is not thread-shared -- only the `BarrierGroup` is). `backends`
/// is returned alongside the grids since `NativeGrid` does not own its
/// backend (brusselator_physics.hpp's own class comment) and the caller
/// must keep it alive for the grids' own lifetime.
struct PartitionedGrid {
  std::vector<std::unique_ptr<fuse::native::NativeMeshBackend>> backends;
  std::vector<NativeGrid> grids;
};

[[nodiscard]] PartitionedGrid make_partitioned_grid(const std::string& name, int nx, int ny,
                                                    int num_ranks) {
  PartitionedGrid out;
  for (int r = 0; r < num_ranks; ++r) {
    out.backends.push_back(std::make_unique<fuse::native::NativeMeshBackend>(r, num_ranks));
    auto g = NativeGrid::make(*out.backends.back(), name, nx, ny);
    EXPECT_TRUE(g.ok()) << g.error().code << ": " << g.error().message;
    out.grids.push_back(std::move(g.value()));
  }
  return out;
}

/// `whole[g.index(i,j)]` restricted to `g`'s own owned band, in LOCAL
/// canonical order -- what a decomposed `UCore`/`VDiffusionCore`'s own
/// constructor (its `initial_v_on_u` argument, OWNED-sized per this
/// track's refactor) and `step()` (its `required` argument) both need, as
/// opposed to the whole-grid-sized vector a caller with no decomposition
/// of its own (V's remap, out of THIS track's scope) still hands over.
[[nodiscard]] Vector slice_owned(const Vector& whole, const NativeGrid& g) {
  Vector out(g.owned_size());
  for (int lj = 0; lj < g.owned_rows(); ++lj) {
    const int j = g.j_lo() + lj;
    for (int i = 0; i < g.nx(); ++i) out[g.local_index(i, j)] = whole[g.index(i, j)];
  }
  return out;
}

/// Runs one `UCore::step()` per partition CONCURRENTLY (one `std::thread`
/// each), `required` sliced to each partition's own owned band from a
/// U-grid-shaped (whole) vector -- the correct input shape per this
/// track's own scope (U is fully decomposed; V's remap, a different
/// track's work, still hands U-grid-shaped data whole). Returns the
/// reassembled whole-grid `provided()`/`state()` (owned bands
/// concatenated in row order, which IS canonical order since row bands
/// are contiguous).
struct StepResult {
  Vector state, provided;
};

/// Builds `num_ranks` `Core` instances CONCURRENTLY, one `std::thread`
/// each -- not a sequential loop on the calling thread. Found necessary
/// by this file's own debugging (`TwoPartitionSchurMatchesSingleRankUCore`
/// hung on its very first run): `UCore`'s own constructor issues a
/// collective itself (`mask_mean_(global_mean(mask_))`, a `Reducer::sum`
/// call over `num_ranks` participants), so building rank 0's `UCore` to
/// completion before STARTING rank 1's -- what a plain `for` loop over
/// `make_unique<UCore>(...)` does -- blocks rank 0 inside that very first
/// collective forever: rank 1 never arrives, because rank 1's own
/// constructor has not been called yet. Exactly the same reason every
/// STEP below also runs one thread per partition (`step_all_partitions`):
/// a decomposed component's ranks are SEPARATE PROCESSES in production,
/// concurrent by construction: this file's own in-process stand-in must
/// be concurrent too, everywhere a collective can fire, construction
/// included -- not only where this file's author expected one.
template <class Core>
[[nodiscard]] std::vector<std::unique_ptr<Core>> make_cores_concurrently(
    int num_ranks, const std::function<std::unique_ptr<Core>(int)>& make_one) {
  std::vector<std::unique_ptr<Core>> cores(static_cast<std::size_t>(num_ranks));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    threads.emplace_back([&, r] { cores[static_cast<std::size_t>(r)] = make_one(r); });
  }
  for (auto& th : threads) th.join();
  return cores;
}

[[nodiscard]] StepResult step_all_partitions(std::vector<std::unique_ptr<UCore>>& cores,
                                             const std::vector<NativeGrid>& grids, double t_begin,
                                             double t_end, double dt, const Vector& required_whole) {
  const int num_ranks = static_cast<int>(cores.size());
  std::vector<fuse::Result<void>> results(static_cast<std::size_t>(num_ranks),
                                          fuse::Result<void>::success());
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    threads.emplace_back([&, r] {
      const auto& g = grids[static_cast<std::size_t>(r)];
      Vector required_owned(g.owned_size());
      for (int lj = 0; lj < g.owned_rows(); ++lj) {
        const int j = g.j_lo() + lj;
        for (int i = 0; i < g.nx(); ++i) {
          required_owned[g.local_index(i, j)] = required_whole[g.index(i, j)];
        }
      }
      results[static_cast<std::size_t>(r)] =
          cores[static_cast<std::size_t>(r)]->step(t_begin, t_end, dt, required_owned);
    });
  }
  for (auto& th : threads) th.join();
  for (const auto& r : results) EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);

  const int nx = grids.front().nx(), ny = grids.front().ny();
  StepResult out;
  out.state = Vector(nx * ny);
  out.provided = Vector(nx * ny);
  for (int r = 0; r < num_ranks; ++r) {
    const auto& g = grids[static_cast<std::size_t>(r)];
    const auto& core = *cores[static_cast<std::size_t>(r)];
    for (int lj = 0; lj < g.owned_rows(); ++lj) {
      const int j = g.j_lo() + lj;
      for (int i = 0; i < g.nx(); ++i) {
        out.state[g.index(i, j)] = core.state()[g.local_index(i, j)];
        out.provided[g.index(i, j)] = core.provided()[g.local_index(i, j)];
      }
    }
  }
  return out;
}

constexpr int kNx = 8, kNy = 8;  // divides evenly by 2 and by 4 (not 3; the
                                 // 3-partition case below uses kNy3 instead)
constexpr int kNy3 = 12;         // divides evenly by 3

}  // namespace

// ---- solver selection: refusals -----------------------------------------

TEST(BrusselatorSpeciesCoreDecomposition, ParseSolverKindRefusesUnknownName) {
  const auto r = brusselator::parse_solver_kind("bogus");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-BRUSS-SOLVER-UNKNOWN");
  EXPECT_EQ(brusselator::parse_solver_kind("auto").value(), SolverKind::Auto);
  EXPECT_EQ(brusselator::parse_solver_kind("direct").value(), SolverKind::Direct);
  EXPECT_EQ(brusselator::parse_solver_kind("schur").value(), SolverKind::Schur);
  EXPECT_EQ(brusselator::parse_solver_kind("pcg").value(), SolverKind::Pcg);
  EXPECT_EQ(brusselator::parse_solver_kind("gmres").value(), SolverKind::Gmres);
}

TEST(BrusselatorSpeciesCoreDecomposition, DirectSolverRefusesWhenDecomposed) {
  auto partitioned = make_partitioned_grid("g", kNx, kNy, 2);
  EXPECT_THROW(
      {
        auto solver = brusselator::make_diffusion_solver(
            partitioned.grids[0], SolverKind::Direct, 1e-12, fuse::linalg::make_serial_reducer(),
            brusselator::serial_exchanger_factory());
        (void)solver;
      },
      fuse::InvalidArgument);
}

TEST(BrusselatorSpeciesCoreDecomposition, VParticipantRefusesDecomposedVRanks) {
  auto made = brusselator::VParticipant::make(kNx, kNy, kNx, kNy, /*diffusion=*/10.0, /*rank=*/0,
                                              /*num_ranks=*/2);
  ASSERT_FALSE(made.ok());
  EXPECT_EQ(made.error().code, "E-BRUSS-V-DECOMPOSED-REMAP");
}

// ---- UCore: N-partition Schur reproduces the single-rank reference ------

TEST(BrusselatorSpeciesCoreDecomposition, TwoPartitionSchurMatchesSingleRankUCore) {
  auto partitioned = make_partitioned_grid("g", kNx, kNy, 2);
  fuse::native::NativeMeshBackend whole_backend(0, 1);
  auto whole_grid_r = NativeGrid::make(whole_backend, "g", kNx, kNy);
  ASSERT_TRUE(whole_grid_r.ok());
  const NativeGrid& whole_grid = whole_grid_r.value();

  PhysicsConfig cfg;
  cfg.solver = SolverKind::Schur;
  Vector required0 = brusselator::initial_condition(/*is_u=*/false, kNx, kNy);  // V0 stand-in on U's grid

  PhysicsConfig cfg_direct;
  cfg_direct.solver = SolverKind::Direct;
  UCore reference(whole_grid, cfg_direct, required0);

  const BarrierGroup group(2);
  auto cores = make_cores_concurrently<UCore>(2, [&](int r) {
    const auto& g = partitioned.grids[static_cast<std::size_t>(r)];
    return std::make_unique<UCore>(g, cfg, slice_owned(required0, g),
                                   group.make_reducer(r, 2),
                                   group.make_exchanger_factory(r, 2));
  });

  const double dt = 0.002;
  double t = 0.0;
  double max_abs_diff = 0.0;
  for (int window = 0; window < 2; ++window) {
    for (int sub = 0; sub < 10; ++sub) {
      const auto ref = reference.step(t, t + dt, dt, required0);
      ASSERT_TRUE(ref.ok()) << ref.error().message;
      const auto decomposed = step_all_partitions(cores, partitioned.grids, t, t + dt, dt, required0);
      t += dt;
      for (int c = 0; c < kNx * kNy; ++c) {
        max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.state[c] - reference.state()[c]));
      }
    }
  }
  // MEASURED, not assumed: an exact (Schur) row-banded solve of the SAME
  // linear system the whole-grid SparseCholesky solves differs from it
  // only by floating-point summation ORDER (the Schur complement's own
  // dense LDLT visits terms in a different order than a single global
  // sparse factorization) -- round-off, not approximation error. Printed
  // below so a future run's own number is in the test log even if this
  // bound needs revisiting.
  std::fprintf(stderr, "[TwoPartitionSchurMatchesSingleRankUCore] max_abs_diff=%.3e\n", max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-12);
}

TEST(BrusselatorSpeciesCoreDecomposition, ThreePartitionSchurMatchesSingleRankUCore) {
  auto partitioned = make_partitioned_grid("g", kNx, kNy3, 3);
  fuse::native::NativeMeshBackend whole_backend(0, 1);
  auto whole_grid_r = NativeGrid::make(whole_backend, "g", kNx, kNy3);
  ASSERT_TRUE(whole_grid_r.ok());
  const NativeGrid& whole_grid = whole_grid_r.value();

  PhysicsConfig cfg;
  cfg.solver = SolverKind::Schur;
  Vector required0 = brusselator::initial_condition(/*is_u=*/false, kNx, kNy3);

  PhysicsConfig cfg_direct;
  cfg_direct.solver = SolverKind::Direct;
  UCore reference(whole_grid, cfg_direct, required0);

  const BarrierGroup group(3);
  auto cores = make_cores_concurrently<UCore>(3, [&](int r) {
    const auto& g = partitioned.grids[static_cast<std::size_t>(r)];
    return std::make_unique<UCore>(g, cfg, slice_owned(required0, g),
                                   group.make_reducer(r, 3),
                                   group.make_exchanger_factory(r, 3));
  });

  const double dt = 0.002;
  double t = 0.0;
  double max_abs_diff = 0.0;
  for (int sub = 0; sub < 10; ++sub) {
    const auto ref = reference.step(t, t + dt, dt, required0);
    ASSERT_TRUE(ref.ok()) << ref.error().message;
    const auto decomposed = step_all_partitions(cores, partitioned.grids, t, t + dt, dt, required0);
    t += dt;
    for (int c = 0; c < kNx * kNy3; ++c) {
      max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.state[c] - reference.state()[c]));
    }
  }
  std::fprintf(stderr, "[ThreePartitionSchurMatchesSingleRankUCore] max_abs_diff=%.3e\n", max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-12);
}

// ---- UCore: N-partition Krylov (Pcg/Gmres) within a DERIVED tolerance ---

namespace {

/// Runs the SAME two-partition comparison as
/// `TwoPartitionSchurMatchesSingleRankUCore` but with `kind` (`Pcg` or
/// `Gmres`) at `rel_tol`, returning the measured max|difference| against
/// the single-rank `Direct` reference over 2 windows (20 steps) at this
/// example's own default `--dt-u 0.002`. Used both by the derivation
/// sweep (`DeriveKrylovSolverTolerance`, below) and by the two solver-
/// specific regression tests that pin the result.
double two_partition_krylov_vs_reference(SolverKind kind, double rel_tol) {
  auto partitioned = make_partitioned_grid("g", kNx, kNy, 2);
  fuse::native::NativeMeshBackend whole_backend(0, 1);
  auto whole_grid_r = NativeGrid::make(whole_backend, "g", kNx, kNy);
  const NativeGrid& whole_grid = whole_grid_r.value();

  Vector required0 = brusselator::initial_condition(/*is_u=*/false, kNx, kNy);

  PhysicsConfig cfg_direct;
  cfg_direct.solver = SolverKind::Direct;
  UCore reference(whole_grid, cfg_direct, required0);

  PhysicsConfig cfg;
  cfg.solver = kind;
  cfg.solver_tolerance = rel_tol;

  const BarrierGroup group(2);
  auto cores = make_cores_concurrently<UCore>(2, [&](int r) {
    const auto& g = partitioned.grids[static_cast<std::size_t>(r)];
    return std::make_unique<UCore>(g, cfg, slice_owned(required0, g),
                                   group.make_reducer(r, 2),
                                   group.make_exchanger_factory(r, 2));
  });

  const double dt = 0.002;
  double t = 0.0;
  double max_abs_diff = 0.0;
  for (int window = 0; window < 2; ++window) {
    for (int sub = 0; sub < 10; ++sub) {
      const auto ref = reference.step(t, t + dt, dt, required0);
      EXPECT_TRUE(ref.ok()) << ref.error().message;
      const auto decomposed = step_all_partitions(cores, partitioned.grids, t, t + dt, dt, required0);
      t += dt;
      for (int c = 0; c < kNx * kNy; ++c) {
        max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.state[c] - reference.state()[c]));
      }
    }
  }
  return max_abs_diff;
}

}  // namespace

/// DERIVES `PhysicsConfig::solver_tolerance`: sweeps `rel_tol` from loose
/// to tight for BOTH `Pcg` and `Gmres` and prints each one's measured
/// max|difference| against the single-rank reference, so the value this
/// test settles on (and `PhysicsConfig::solver_tolerance`'s own default,
/// updated to match once this run's numbers are read back) is read off
/// real measurements, never guessed. The bound compared against is the
/// SAME one `TwoPartitionSchurMatchesSingleRankUCore` measures (an exact
/// solve's own round-off-scale disagreement with the whole-grid
/// reference, from floating-point summation order alone) -- a Krylov
/// solver tightened past that point is resolving its own iteration past
/// where an EXACT decomposed solve already agrees with the undecomposed
/// one, which is tolerance spent on nothing.
TEST(BrusselatorSpeciesCoreDecomposition, DeriveKrylovSolverTolerance) {
  for (const SolverKind kind : {SolverKind::Pcg, SolverKind::Gmres}) {
    const char* name = kind == SolverKind::Pcg ? "pcg" : "gmres";
    for (const double tol : {1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13}) {
      const double diff = two_partition_krylov_vs_reference(kind, tol);
      std::fprintf(stderr, "[DeriveKrylovSolverTolerance] solver=%s rel_tol=%.0e max_abs_diff=%.3e\n",
                  name, tol, diff);
    }
  }
}

// Thresholds below pin `PhysicsConfig::solver_tolerance`'s own derived
// default (`1e-13`, species_core.hpp's own comment on that field has the
// full sweep this session measured): at that tolerance, pcg measured
// 5.862e-14 and gmres 5.462e-14 against the single-rank Direct reference
// -- both within ~2.5x of the EXACT Schur solve's own 2.420e-14 floor
// (TwoPartitionSchurMatchesSingleRankUCore). 1e-12, roughly 20x the
// measured value and still two orders of magnitude under the 1e-11
// coupling-window tolerance these solvers sit underneath, is the
// regression bound: loose enough not to fail on ordinary rebuild-to-
// rebuild floating-point variation, tight enough that the first-order-
// worse defect this guards against (a dropped ghost term, a wrong
// Jacobi diagonal, a wrong assembled sign) -- each of which this track's
// own mutation rows measure at 1e2-1e4x this bound -- fails it clearly.

TEST(BrusselatorSpeciesCoreDecomposition, TwoPartitionPcgMatchesSingleRankWithinDerivedTolerance) {
  const double diff = two_partition_krylov_vs_reference(SolverKind::Pcg, 1e-13);
  std::fprintf(stderr, "[TwoPartitionPcgMatchesSingleRankWithinDerivedTolerance] max_abs_diff=%.3e\n",
              diff);
  EXPECT_LT(diff, 1e-12);
}

TEST(BrusselatorSpeciesCoreDecomposition, TwoPartitionGmresMatchesSingleRankWithinDerivedTolerance) {
  const double diff = two_partition_krylov_vs_reference(SolverKind::Gmres, 1e-13);
  std::fprintf(stderr, "[TwoPartitionGmresMatchesSingleRankWithinDerivedTolerance] max_abs_diff=%.3e\n",
              diff);
  EXPECT_LT(diff, 1e-12);
}

// ---- VDiffusionCore: N-partition Schur reproduces the single-rank -------
// ---- reference -----------------------------------------------------------

TEST(BrusselatorSpeciesCoreDecomposition, TwoPartitionVDiffusionCoreMatchesSingleRank) {
  auto partitioned = make_partitioned_grid("g", kNx, kNy, 2);
  fuse::native::NativeMeshBackend whole_backend(0, 1);
  auto whole_grid_r = NativeGrid::make(whole_backend, "g", kNx, kNy);
  ASSERT_TRUE(whole_grid_r.ok());
  const NativeGrid& whole_grid = whole_grid_r.value();

  VDiffusionCore reference(whole_grid, /*diffusion=*/10.0, SolverKind::Direct);

  const BarrierGroup group(2);
  auto cores = make_cores_concurrently<VDiffusionCore>(2, [&](int r) {
    return std::make_unique<VDiffusionCore>(
        partitioned.grids[static_cast<std::size_t>(r)], /*diffusion=*/10.0, SolverKind::Schur,
        1e-12, group.make_reducer(r, 2), group.make_exchanger_factory(r, 2));
  });

  const double dt = 0.003;
  Vector q_on_v = Vector::Zero(kNx * kNy);
  for (int c = 0; c < kNx * kNy; ++c) q_on_v[c] = 0.01 * static_cast<double>(c % 7);

  double max_abs_diff = 0.0;
  for (int sub = 0; sub < 10; ++sub) {
    const auto ref = reference.step(dt, q_on_v);
    ASSERT_TRUE(ref.ok()) << ref.error().message;

    std::vector<fuse::Result<void>> results(2, fuse::Result<void>::success());
    std::vector<std::thread> threads;
    for (int r = 0; r < 2; ++r) {
      threads.emplace_back([&, r] {
        const auto& g = partitioned.grids[static_cast<std::size_t>(r)];
        Vector q_owned(g.owned_size());
        for (int lj = 0; lj < g.owned_rows(); ++lj) {
          const int j = g.j_lo() + lj;
          for (int i = 0; i < g.nx(); ++i) q_owned[g.local_index(i, j)] = q_on_v[g.index(i, j)];
        }
        results[static_cast<std::size_t>(r)] = cores[static_cast<std::size_t>(r)]->step(dt, q_owned);
      });
    }
    for (auto& th : threads) th.join();
    for (const auto& r : results) ASSERT_TRUE(r.ok()) << r.error().message;

    for (int r = 0; r < 2; ++r) {
      const auto& g = partitioned.grids[static_cast<std::size_t>(r)];
      const auto& core = *cores[static_cast<std::size_t>(r)];
      for (int lj = 0; lj < g.owned_rows(); ++lj) {
        const int j = g.j_lo() + lj;
        for (int i = 0; i < g.nx(); ++i) {
          max_abs_diff = std::max(
              max_abs_diff,
              std::abs(core.state()[g.local_index(i, j)] - reference.state()[g.index(i, j)]));
        }
      }
    }
  }
  std::fprintf(stderr, "[TwoPartitionVDiffusionCoreMatchesSingleRank] max_abs_diff=%.3e\n", max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-13);
}
