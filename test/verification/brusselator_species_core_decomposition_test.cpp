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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include <fuse/backend/native_halo.hpp>
#include <fuse/backend/native_mesh.hpp>
#include <fuse/backend/remap_plan.hpp>

#include "brusselator_physics.hpp"
#include "species_core.hpp"

namespace {

using brusselator::NativeGrid;
using brusselator::PhysicsConfig;
using brusselator::SolverKind;
using brusselator::UCore;
using brusselator::VDiffusionCore;
using brusselator::VParticipant;
using brusselator::Vector;
using fuse::ComponentId;
using fuse::linalg::GhostExchanger;
using fuse::linalg::GhostPattern;
using fuse::linalg::Reducer;
using fuse::native::NativeMeshBackend;
using fuse::native::NativeMeshView;

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
/// `max`) and a `GatherBarrier` registry for `allgather` (keyed by
/// `n_each` since `SchurSlabSolver`'s own two collectives per `factor()`/
/// `solve()` use two different sizes -- the same reason test/linalg/
/// schur_slab_test.cpp's own `RingAllgatherReducer` keys its cache by
/// `n_each`) -- a SEPARATE registry from `BarrierGhostExchanger`'s own
/// (below), even though both are "a `GatherBarrier` keyed by a size":
/// sharing one registry between allgather's OWN collectives and a ghost
/// update's would let two semantically unrelated exchanges that happen to
/// ask for the SAME size rendezvous on the SAME `GatherBarrier` and read
/// back a mix of each other's data -- a real risk once `VParticipant`'s
/// own two remap-halo exchangers (species_core.hpp) and a Krylov solver's
/// own ghost pattern (both PLAUSIBLY V's `owned_size()`) can appear in the
/// SAME run. `sum(double*, n)` (the vectorised overload) is not exercised
/// by anything this file builds (neither `UCore`/`VDiffusionCore` nor the
/// Krylov solvers call it) -- implemented as `n` independent scalar
/// reductions rather than left unimplemented, since a correct (if unused)
/// implementation costs nothing here and documents the intended semantics
/// for a future caller.
class BarrierReducer final : public Reducer {
 public:
  BarrierReducer(int rank, int size, std::shared_ptr<ScalarBarrier> sum_barrier,
                std::shared_ptr<ScalarBarrier> max_barrier,
                std::shared_ptr<std::mutex> allgather_registry_mu,
                std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> allgather_registry)
      : rank_(rank),
        size_(size),
        sum_barrier_(std::move(sum_barrier)),
        max_barrier_(std::move(max_barrier)),
        allgather_registry_mu_(std::move(allgather_registry_mu)),
        allgather_registry_(std::move(allgather_registry)) {}

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
      std::lock_guard<std::mutex> lk(*allgather_registry_mu_);
      auto& slot = (*allgather_registry_)[n_each];
      if (!slot) slot = std::make_shared<GatherBarrier>(size_, n_each);
      barrier = slot;
    }
    barrier->combine(rank_, send, recv);
  }

 private:
  int rank_, size_;
  std::shared_ptr<ScalarBarrier> sum_barrier_, max_barrier_;
  std::shared_ptr<std::mutex> allgather_registry_mu_;
  std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> allgather_registry_;
};

/// `GatherBarrier`'s own `combine()` assumes every rank sends the SAME
/// `n_each` (an Allgather, not an Allgatherv) -- true of `BarrierReducer::
/// allgather`'s own callers (`SchurSlabSolver`'s boundary/trace exchange is
/// a FIXED size independent of how many rows a rank owns), but FALSE of a
/// remap halo's own owned-cell count once a partition's row split is
/// genuinely uneven (found this session: a 6/5/5-row U split sends 48, 40,
/// 40 doubles from its three ranks respectively -- `GatherBarrier`'s fixed-
/// stride buffer silently mis-assembles that, and since `owned_n` was ALSO
/// baked into the ghost registry's own key at the time, three DIFFERENT
/// uneven `owned_n` values classified into three SEPARATE single-rank
/// barriers, each waiting forever for the other two ranks that could never
/// arrive on ITS key -- a real deadlock, not a correctness defect with a
/// symptom; `ctest`'s own default per-test timeout is what would have
/// caught it on a clean run instead of hanging the whole build). This
/// class is the Allgatherv-shaped sibling: every rank's `send_n` can
/// differ; the combined buffer and a `size()+1` prefix-sum `offsets` array
/// (so `combined[offsets[owner] + local]` finds owner `owner`'s own
/// `local`'th value) come back to every caller once all `size()` ranks
/// have arrived -- the same single-mutex/condition-variable rendezvous
/// `GatherBarrier` already uses, just deferring the output buffer's own
/// layout until every rank's size is known.
class VarGatherBarrier {
 public:
  explicit VarGatherBarrier(int size)
      : size_(size), sizes_(static_cast<std::size_t>(size), 0),
        sends_(static_cast<std::size_t>(size), nullptr),
        offsets_(static_cast<std::size_t>(size) + 1, 0) {}

  void combine(int rank, const double* send, std::size_t send_n, std::vector<double>* combined_out,
              std::vector<std::size_t>* offsets_out) {
    std::unique_lock<std::mutex> lk(mu_);
    const int gen = generation_;
    sizes_[static_cast<std::size_t>(rank)] = send_n;
    sends_[static_cast<std::size_t>(rank)] = send;
    if (++arrived_ == size_) {
      offsets_[0] = 0;
      for (std::size_t r = 0; r < static_cast<std::size_t>(size_); ++r) {
        offsets_[r + 1] = offsets_[r] + sizes_[r];
      }
      combined_.assign(offsets_[static_cast<std::size_t>(size_)], 0.0);
      for (std::size_t r = 0; r < static_cast<std::size_t>(size_); ++r) {
        std::copy(sends_[r], sends_[r] + sizes_[r], combined_.begin() + static_cast<std::ptrdiff_t>(offsets_[r]));
      }
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return generation_ != gen; });
    }
    *combined_out = combined_;
    *offsets_out = offsets_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int size_;
  int arrived_ = 0;
  int generation_ = 0;
  std::vector<std::size_t> sizes_;
  std::vector<const double*> sends_;
  std::vector<std::size_t> offsets_;
  std::vector<double> combined_;
};

/// `fuse::linalg::GhostExchanger` built from a `VarGatherBarrier` registry
/// keyed by `call_index` ALONE, not `(call_index, owned_n)`: `call_index`
/// is this EXCHANGER FACTORY's own Nth invocation (`BarrierGroup::
/// make_exchanger_factory`'s own closure counter, below), which already
/// lines up across ranks because every rank runs the SAME production
/// code path in the SAME order (`VParticipant::make`'s own two remap
/// exchangers, THEN whatever `VDiffusionCore`'s own constructor builds,
/// if anything) -- the same "same call site, same round" reasoning
/// `BarrierReducer::allgather`'s own `n_each` key relies on. `owned_n`
/// was folded into this key in an earlier round of this same session, to
/// guard against two UNRELATED ghost exchanges in the SAME run
/// coincidentally sharing one `owned_n` (`VParticipant`'s own V->U remap
/// exchanger and a Krylov solver's own V-grid ghost pattern can both be
/// V's `owned_size()`) -- but `owned_n` is NOT invariant across ranks at
/// a GENUINELY uneven row split (a 6/5/5-row U partition's three ranks
/// own 48/40/40 cells respectively), so keying on it split one logical
/// 3-rank collective into THREE single-rank ones, each waiting forever
/// for peers that could never arrive on its own key -- the actual
/// deadlock `VarGatherBarrier`'s own class comment (above) measured.
/// `call_index` alone already disambiguates same-run same-size exchanges
/// (VParticipant's two calls get indices 0 and 1 on every rank, never
/// colliding with each other or with a third caller's index 2) without
/// ever assuming a rank-invariant size. A ghost update IS a gather
/// followed by a local pick, nothing more (`GhostExchangerFactory`'s own
/// comment, species_core.hpp, says why the PATTERN must always come from
/// the OPERATOR/remap that actually owns it, never borrowed from a
/// different one).
class BarrierGhostExchanger final : public GhostExchanger {
 public:
  using Key = int;  // call_index

  BarrierGhostExchanger(GhostPattern pattern, int rank, std::size_t owned_n, int call_index,
                        std::shared_ptr<std::mutex> registry_mu,
                        std::shared_ptr<std::map<Key, std::shared_ptr<VarGatherBarrier>>> registry,
                        int size)
      : pattern_(std::move(pattern)),
        rank_(rank),
        owned_n_(owned_n),
        call_index_(call_index),
        registry_mu_(std::move(registry_mu)),
        registry_(std::move(registry)),
        size_(size) {}

  [[nodiscard]] std::size_t ghost_size() const override { return pattern_.owner_rank.size(); }

  [[nodiscard]] fuse::Result<void> update(const double* owned, std::size_t owned_n,
                                          double* ghosts, std::size_t ghost_n) const override {
    if (owned_n != owned_n_ || ghost_n != pattern_.owner_rank.size()) {
      return fuse::Failure{"E-TEST-HALO-SIZE", "BarrierGhostExchanger: size mismatch"};
    }
    std::shared_ptr<VarGatherBarrier> barrier;
    const Key key = call_index_;
    {
      std::lock_guard<std::mutex> lk(*registry_mu_);
      auto& slot = (*registry_)[key];
      if (!slot) slot = std::make_shared<VarGatherBarrier>(size_);
      barrier = slot;
    }
    std::vector<double> combined;
    std::vector<std::size_t> offsets;
    barrier->combine(rank_, owned, owned_n_, &combined, &offsets);
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern_.owner_rank[g]);
      const auto local = static_cast<std::size_t>(pattern_.remote_local_index[g]);
      ghosts[g] = combined[offsets[owner] + local];
    }
    return fuse::Result<void>::success();
  }

 private:
  GhostPattern pattern_;
  int rank_;
  std::size_t owned_n_;
  int call_index_;
  std::shared_ptr<std::mutex> registry_mu_;
  std::shared_ptr<std::map<Key, std::shared_ptr<VarGatherBarrier>>> registry_;
  int size_;
};

/// Everything shared across one run's `num_ranks` partitions: one pair of
/// `ScalarBarrier`s per rank-group (every partition's `BarrierReducer`
/// shares the SAME pair, so a `sum`/`max` call on any partition finds
/// every peer at the matching round), one registry for `allgather`, and a
/// SEPARATE registry for ghost updates (`BarrierGhostExchanger`'s own
/// class comment explains why the two must not share one).
struct BarrierGroup {
  explicit BarrierGroup(int size)
      : sum_barrier(std::make_shared<ScalarBarrier>(size)),
        max_barrier(std::make_shared<ScalarBarrier>(size)),
        allgather_registry_mu(std::make_shared<std::mutex>()),
        allgather_registry(std::make_shared<std::map<std::size_t, std::shared_ptr<GatherBarrier>>>()),
        ghost_registry_mu(std::make_shared<std::mutex>()),
        ghost_registry(
            std::make_shared<std::map<BarrierGhostExchanger::Key, std::shared_ptr<VarGatherBarrier>>>()) {}

  std::shared_ptr<ScalarBarrier> sum_barrier, max_barrier;
  std::shared_ptr<std::mutex> allgather_registry_mu;
  std::shared_ptr<std::map<std::size_t, std::shared_ptr<GatherBarrier>>> allgather_registry;
  std::shared_ptr<std::mutex> ghost_registry_mu;
  std::shared_ptr<std::map<BarrierGhostExchanger::Key, std::shared_ptr<VarGatherBarrier>>> ghost_registry;

  [[nodiscard]] std::shared_ptr<BarrierReducer> make_reducer(int rank, int size) const {
    return std::make_shared<BarrierReducer>(rank, size, sum_barrier, max_barrier, allgather_registry_mu,
                                            allgather_registry);
  }
  /// Returns a FRESH factory (its own `call_index` counter, starting at
  /// 0) each call -- a caller that needs the SAME counter sequence to
  /// line up across ranks (every `VParticipant`/`VDiffusionCore`-building
  /// test below) calls this ONCE PER RANK, exactly as `make_reducer` is
  /// already called, so rank r's Nth call to the returned factory and
  /// rank s's Nth call share a `call_index`, regardless of r/s.
  [[nodiscard]] brusselator::GhostExchangerFactory make_exchanger_factory(int rank, int size) const {
    auto mu = ghost_registry_mu;
    auto reg = ghost_registry;
    auto call_index = std::make_shared<int>(0);
    return [rank, size, mu, reg, call_index](const GhostPattern& pattern, std::size_t owned_n) {
      const int idx = (*call_index)++;
      return std::static_pointer_cast<const GhostExchanger>(
          std::make_shared<BarrierGhostExchanger>(pattern, rank, owned_n, idx, mu, reg, size));
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

// `E-BRUSS-V-DECOMPOSED-REMAP` (the outright num_ranks()>1 refusal this
// test used to check) is GONE: the Halo track's own work (docs/dev/
// native-halo.md) closed exactly the seam that refusal was guarding.
// What remains true in THIS sandbox is narrower and still worth pinning:
// a backend that is actually asked to BUILD a halo over more than one
// rank (a non-null communicator, no halo pre-installed) hits
// `fuse::native::build_native_remap_halo`'s own `E-MESH-NO-MPI` --
// this build has no MPI at all, so there is no in-process path for the
// REAL collective (`test/parallel/native_halo_mpi_test.cpp` is exactly
// the test that exercises it, build-verified only here). This is the
// expected, honest failure mode now; the tests below exercise the
// DECOMPOSED path for real by pre-installing a hand-computed halo, the
// same `NativeMeshBackend::commit_remap_halo` seam `test/backend/
// native_halo_test.cpp`'s own `BackendFixture` uses, which is possible
// precisely BECAUSE `VParticipant::make` checks `has_remap_halo` and
// skips calling the real (MPI-only) builder for a triple that is
// already there.
TEST(BrusselatorSpeciesCoreDecomposition, VParticipantFailsCleanlyWithoutMPIWhenAskedToDecompose) {
  NativeMeshBackend backend(0, 2);
  backend.set_communicator(1);  // non-null, so communicator_size() reports 2
  auto made = brusselator::VParticipant::make(std::move(backend), kNx, kNy, kNx, kNy,
                                              /*diffusion=*/10.0);
  ASSERT_FALSE(made.ok());
  EXPECT_EQ(made.error().code, "E-MESH-NO-MPI");
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


// ---- VParticipant: N-partition two-sided remap reproduces the ----------
// ---- single-rank reference (Halo track) ---------------------------------
//
// `VParticipant::make_from_grids`'s own `fuse::build_remap_halo` call is
// skipped (`has_remap_halo` already true) whenever a caller pre-installs
// a halo -- exactly what this section does, computing it the SAME way
// `test/backend/native_halo_test.cpp`'s own `BackendFixture`/
// `run_halo_pipeline` do (the pure covering-set/pack/merge pipeline,
// never the real MPI-backed `fuse::native::build_native_remap_halo`,
// which has no in-process multi-rank path at all without MPI). A
// PLACEHOLDER, never-dereferenced communicator handle is still set on
// each backend (mirroring `BackendFixture`'s own comment) purely so
// `communicator_size()` reports `num_ranks` wherever anything downstream
// reads it, matching what a real run would report.

namespace {

using fuse::native::Box3;
using fuse::native::cell_box;
using fuse::native::covering_set_destinations;
using fuse::native::merge_halo_cells;
using fuse::native::owned_block_index;
using fuse::native::pack_halo_cells;

/// One simulated rank's independently-built (source, target) view pair --
/// `test/backend/native_halo_test.cpp`'s own `RankPieces`, generalised
/// over the two mesh specs so it serves EITHER remap direction (U->V or
/// V->U) this track's own `VParticipant` needs a halo for.
struct RankMeshPieces {
  std::unique_ptr<NativeMeshView> src, tgt;
};

[[nodiscard]] std::vector<RankMeshPieces> build_rank_meshes(const std::string& src_spec,
                                                            const std::string& tgt_spec,
                                                            int num_ranks) {
  std::vector<RankMeshPieces> ranks;
  ranks.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    RankMeshPieces p;
    auto sa = fuse::native::generate_native_mesh(src_spec, r, num_ranks);
    EXPECT_TRUE(sa.ok()) << sa.error().message;
    auto sv = NativeMeshView::create_from_host(sa.value(), r);
    EXPECT_TRUE(sv.ok()) << sv.error().message;
    p.src = std::move(sv.value());

    auto ta = fuse::native::generate_native_mesh(tgt_spec, r, num_ranks);
    EXPECT_TRUE(ta.ok()) << ta.error().message;
    auto tv = NativeMeshView::create_from_host(ta.value(), r);
    EXPECT_TRUE(tv.ok()) << tv.error().message;
    p.tgt = std::move(tv.value());
    ranks.push_back(std::move(p));
  }
  return ranks;
}

struct HaloRunResult {
  std::vector<std::unique_ptr<NativeMeshView>> new_src;
  std::vector<GhostPattern> pattern;
};

/// The covering-set -> pack -> (in-process "exchange", a plain array
/// index rather than a real `MPI_Alltoallv` -- this pipeline has no
/// control-flow-dependent collective the way `UCore`'s own reaction loop
/// does, so a sequential per-rank loop reproduces a real multi-rank run's
/// OBSERVABLE result exactly, unlike the thread-barrier machinery
/// `BarrierReducer`/`BarrierGhostExchanger` above exist for) -> merge
/// pipeline, PURE PORT of `test/backend/native_halo_test.cpp`'s own
/// `run_halo_pipeline` (that file owns the correctness tests for this
/// pipeline itself -- `NativeHaloQuadGrid.*` -- so this copy is not
/// re-proving it, only reusing it to set up a REAL halo for THIS track's
/// own `VParticipant` tests).
[[nodiscard]] HaloRunResult run_remap_halo_pipeline(std::vector<RankMeshPieces>& ranks,
                                                    int num_ranks) {
  std::vector<Box3> all_target_boxes;
  std::vector<int> owner_of_target_box;
  for (int r = 0; r < num_ranks; ++r) {
    const NativeMeshView& t = *ranks[static_cast<std::size_t>(r)].tgt;
    for (std::size_t c = 0; c < t.entity_count(2); ++c) {
      // OWNED cells only -- `test/backend/native_halo_test.cpp`'s own
      // `run_halo_pipeline` filters on `is_owned` here; matched exactly
      // (every cell `generate_native_mesh` emits for a fresh per-rank
      // view is already owned, so this is a no-op TODAY, but the
      // reference's own filter is not decorative -- keep this port
      // identical to it rather than relying on that invariant silently).
      if (t.is_owned(2, c)) {
        all_target_boxes.push_back(cell_box(t, c));
        owner_of_target_box.push_back(r);
      }
    }
  }

  std::vector<std::vector<std::vector<std::byte>>> send_from(static_cast<std::size_t>(num_ranks));
  std::vector<std::vector<std::size_t>> owned_idx_of(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    NativeMeshView& s = *ranks[static_cast<std::size_t>(r)].src;
    owned_idx_of[static_cast<std::size_t>(r)] = owned_block_index(s);
    std::vector<Box3> my_owned_boxes;
    std::vector<std::size_t> owned_local_index;
    for (std::size_t c = 0; c < s.entity_count(2); ++c) {
      if (s.is_owned(2, c)) {
        my_owned_boxes.push_back(cell_box(s, c));
        owned_local_index.push_back(c);
      }
    }
    const auto dest_for_owned =
        covering_set_destinations(my_owned_boxes, r, all_target_boxes, owner_of_target_box);
    std::vector<std::vector<std::size_t>> send_by_local_index(static_cast<std::size_t>(num_ranks));
    for (std::size_t i = 0; i < dest_for_owned.size(); ++i) {
      for (int d : dest_for_owned[i]) {
        send_by_local_index[static_cast<std::size_t>(d)].push_back(owned_local_index[i]);
      }
    }
    send_from[static_cast<std::size_t>(r)] =
        pack_halo_cells(s, send_by_local_index, owned_idx_of[static_cast<std::size_t>(r)], r, num_ranks);
  }

  HaloRunResult result;
  result.new_src.resize(static_cast<std::size_t>(num_ranks));
  result.pattern.resize(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    std::vector<std::vector<std::byte>> received;
    received.reserve(static_cast<std::size_t>(num_ranks));
    for (int s = 0; s < num_ranks; ++s) {
      received.push_back(send_from[static_cast<std::size_t>(s)][static_cast<std::size_t>(r)]);
    }
    std::vector<int> owner_rank;
    std::vector<std::int64_t> owner_owned_index;
    auto merged = merge_halo_cells(*ranks[static_cast<std::size_t>(r)].src, received, r, &owner_rank,
                                   &owner_owned_index);
    EXPECT_TRUE(merged.ok()) << "rank " << r << ": " << merged.error().message;
    auto new_view = NativeMeshView::create_from_host(merged.value(), r);
    EXPECT_TRUE(new_view.ok()) << "rank " << r << ": " << new_view.error().message;
    result.new_src[static_cast<std::size_t>(r)] = std::move(new_view.value());
    result.pattern[static_cast<std::size_t>(r)].owner_rank = owner_rank;
    result.pattern[static_cast<std::size_t>(r)].remote_local_index = owner_owned_index;
  }
  return result;
}

/// One rank's fully-prepared pieces: the backend (halo already installed
/// both directions, via the real `commit_remap_halo` seam) plus the TWO
/// `NativeGrid`s `NativeGrid::make` already validated as UNWIDENED --
/// `VParticipant::make_grids` MUST run before the halo is installed (its
/// own `E-BRUSS-MESH-COUNT` check requires an exact `nx*owned_rows`
/// `entity_count`, which a pre-installed halo would violate), so this
/// function's own call order below is not incidental.
struct DecomposedVRank {
  std::unique_ptr<NativeMeshBackend> backend;
  NativeGrid u_grid, v_grid;
};

/// Builds `num_ranks` ranks' worth of `(backend, u_grid, v_grid)`, each
/// backend's BOTH remap halos (U->V and V->U) pre-installed via the real
/// `commit_remap_halo` seam -- ready to hand to `VParticipant::
/// make_from_grids` one at a time (it will see `has_remap_halo()==true`
/// for both triples and skip its own internal `fuse::build_remap_halo`
/// call, which has no in-process path without real MPI).
[[nodiscard]] std::vector<DecomposedVRank> build_decomposed_v_ranks(int nx_u, int ny_u, int nx_v,
                                                                    int ny_v, int num_ranks) {
  const std::string u_spec = "quad:" + std::to_string(nx_u) + "x" + std::to_string(ny_u);
  const std::string v_spec = "quad:" + std::to_string(nx_v) + "x" + std::to_string(ny_v);

  auto uv_ranks = build_rank_meshes(u_spec, v_spec, num_ranks);  // src=u, tgt=v
  auto uv_run = run_remap_halo_pipeline(uv_ranks, num_ranks);
  auto vu_ranks = build_rank_meshes(v_spec, u_spec, num_ranks);  // src=v, tgt=u
  auto vu_run = run_remap_halo_pipeline(vu_ranks, num_ranks);

  std::vector<DecomposedVRank> out;
  out.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    auto backend = std::make_unique<NativeMeshBackend>(r, num_ranks);
    backend->set_communicator(1);  // placeholder, never dereferenced -- see this section's header

    // `make_grids` FIRST, on the unwidened backend -- see this function's
    // own comment above.
    auto grids = VParticipant::make_grids(*backend, nx_u, ny_u, nx_v, ny_v);
    EXPECT_TRUE(grids.ok()) << (grids.ok() ? "" : grids.error().message);
    NativeGrid u_grid = std::move(grids.value().first);
    NativeGrid v_grid = std::move(grids.value().second);
    const fuse::MeshId u_mid = u_grid.mesh_id();
    const fuse::MeshId v_mid = v_grid.mesh_id();

    auto committed_uv = backend->commit_remap_halo(u_mid, ComponentId{},
                                                   std::move(uv_run.new_src[static_cast<std::size_t>(r)]),
                                                   v_mid, 2, uv_run.pattern[static_cast<std::size_t>(r)]);
    EXPECT_TRUE(committed_uv.ok()) << committed_uv.error().message;
    auto committed_vu = backend->commit_remap_halo(v_mid, ComponentId{},
                                                   std::move(vu_run.new_src[static_cast<std::size_t>(r)]),
                                                   u_mid, 2, vu_run.pattern[static_cast<std::size_t>(r)]);
    EXPECT_TRUE(committed_vu.ok()) << committed_vu.error().message;

    out.push_back(DecomposedVRank{std::move(backend), std::move(u_grid), std::move(v_grid)});
  }
  return out;
}

/// Runs one `VParticipant::step()` per partition CONCURRENTLY (one
/// `std::thread` each -- `VDiffusionCore`'s own constructor/solve issue
/// collectives, the SAME reason `make_cores_concurrently`/
/// `step_all_partitions` above run threaded). `required_whole`: U-grid-
/// shaped (whole), sliced to each partition's own owned band -- mirrors
/// `step_all_partitions`'s own contract; a real run would hand a
/// genuinely decomposed U's OWN per-rank share, not a slice of a shared
/// whole, but the slicing itself is identical either way.
struct VStepResult {
  Vector provided;  // U-grid-shaped, reassembled from every partition's own band
  // `VDiffusionCore::exchange_in()` is ALREADY a globally-reduced
  // quantity (`dt * global_mean(q_on_v)`, global_mean's own `reducer_->
  // sum(...)/global_n_` an allreduce-based mean, not a per-rank partial),
  // so every partition reports the SAME value by construction -- checked
  // explicitly below (`max_disagreement`), not assumed, since a defect
  // that broke the reduction would otherwise be invisible to a check that
  // only ever reads rank 0's own copy.
  double exchange_in = 0.0;
  double exchange_in_max_disagreement = 0.0;
};

[[nodiscard]] VStepResult step_all_v_partitions(std::vector<std::unique_ptr<VParticipant>>& cores,
                                                const std::vector<NativeGrid>& u_grids, double dt,
                                                const Vector& required_whole) {
  const int num_ranks = static_cast<int>(cores.size());
  std::vector<fuse::Result<void>> results(static_cast<std::size_t>(num_ranks),
                                          fuse::Result<void>::success());
  std::vector<Vector> provided_per_rank(static_cast<std::size_t>(num_ranks));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    threads.emplace_back([&, r] {
      const auto& g = u_grids[static_cast<std::size_t>(r)];
      Vector required_owned = slice_owned(required_whole, g);
      Vector provided(g.owned_size());
      results[static_cast<std::size_t>(r)] = cores[static_cast<std::size_t>(r)]->step(
          dt, required_owned, &provided);
      provided_per_rank[static_cast<std::size_t>(r)] = std::move(provided);
    });
  }
  for (auto& th : threads) th.join();
  for (const auto& r : results) EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);

  const int nx = u_grids.front().nx(), ny = u_grids.front().ny();
  VStepResult out;
  out.provided = Vector(nx * ny);
  for (int r = 0; r < num_ranks; ++r) {
    const auto& g = u_grids[static_cast<std::size_t>(r)];
    for (int lj = 0; lj < g.owned_rows(); ++lj) {
      const int j = g.j_lo() + lj;
      for (int i = 0; i < g.nx(); ++i) {
        out.provided[g.index(i, j)] = provided_per_rank[static_cast<std::size_t>(r)][g.local_index(i, j)];
      }
    }
    const double this_rank = cores[static_cast<std::size_t>(r)]->exchange_in();
    if (r == 0) {
      out.exchange_in = this_rank;
    } else {
      out.exchange_in_max_disagreement =
          std::max(out.exchange_in_max_disagreement, std::abs(this_rank - out.exchange_in));
    }
  }
  return out;
}

constexpr int kNxUHalo = 8, kNyUHalo = 8;    // 8x8 U / 12x12 V -- same 2:3 cell-width ratio as the
constexpr int kNxVHalo = 12, kNyVHalo = 12;  // example's own 16x16/24x24, at a size the tier-0
                                             // halo pipeline above runs quickly. The row split
                                             // aligns exactly at y=0.5 for 2 ranks, so this case
                                             // adopts ZERO cells (mirrors native_halo_test.cpp's
                                             // own 2-rank fixture).
constexpr int kNyUHalo3 = 12, kNyVHalo3 = 18;  // 8x12 U / 12x18 V: divides evenly by 3 in
                                               // BOTH meshes (zero adopted cells, like the 2-rank
                                               // case above) -- the ALIGNED 3-way split.
// KNOWN_ISSUES D29 (CLOSED): the genuinely MISALIGNED 3-way split (16 U rows
// as 6/5/5 against 24 V rows as 8/8/8, so SOME U cell crosses a V partition
// boundary) that used to measurably disagree with the single-rank reference
// -- see VParticipant::make_from_grids's own comment for the measured cause
// and fix. `ThreePartitionVParticipantMatchesSingleRankMisaligned` below now
// drives exactly this split end to end.
constexpr int kNyUMis = 16, kNyVMis = 24;  // 6/5/5 vs 8/8/8 over 3 partitions

}  // namespace

TEST(BrusselatorSpeciesCoreDecomposition, TwoPartitionVParticipantMatchesSingleRank) {
  // Single-rank reference: a fresh, undecomposed backend, exactly what
  // every EXISTING in-process caller already builds.
  NativeMeshBackend ref_backend(0, 1);
  auto reference_r = VParticipant::make(std::move(ref_backend), kNxUHalo, kNyUHalo, kNxVHalo,
                                        kNyVHalo, /*diffusion=*/10.0, SolverKind::Direct);
  ASSERT_TRUE(reference_r.ok()) << reference_r.error().message;
  VParticipant reference = std::move(reference_r.value());

  auto u_partitioned = make_partitioned_grid("u", kNxUHalo, kNyUHalo, 2);
  auto ranks = build_decomposed_v_ranks(kNxUHalo, kNyUHalo, kNxVHalo, kNyVHalo, 2);

  const BarrierGroup group(2);
  auto cores = make_cores_concurrently<VParticipant>(2, [&](int r) {
    auto& rk = ranks[static_cast<std::size_t>(r)];
    auto made = VParticipant::make_from_grids(std::move(*rk.backend), std::move(rk.u_grid),
                                              std::move(rk.v_grid), /*diffusion=*/10.0,
                                              SolverKind::Schur, 1e-12, group.make_reducer(r, 2),
                                              group.make_exchanger_factory(r, 2));
    EXPECT_TRUE(made.ok()) << (made.ok() ? "" : made.error().message);
    return std::make_unique<VParticipant>(std::move(made.value()));
  });

  const double dt = 0.003;
  Vector required = brusselator::initial_condition(/*is_u=*/false, kNxUHalo, kNyUHalo);  // q stand-in, on U's grid

  double max_abs_diff = 0.0;
  double exchange_in_single = 0.0;
  for (int sub = 0; sub < 6; ++sub) {
    Vector ref_provided(reference.n_u());
    const auto ref = reference.step(dt, required, &ref_provided);
    ASSERT_TRUE(ref.ok()) << ref.error().message;
    exchange_in_single = reference.exchange_in();

    const auto decomposed = step_all_v_partitions(cores, u_partitioned.grids, dt, required);
    for (int c = 0; c < kNxUHalo * kNyUHalo; ++c) {
      max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.provided[c] - ref_provided[c]));
    }
    // Conservation: every partition's own globally-reduced exchange_in()
    // must agree with every OTHER partition's (a real defect in the
    // reduction would make them diverge, invisible to a check that only
    // ever read one partition's own copy) AND with the single-rank
    // reference's.
    EXPECT_LT(decomposed.exchange_in_max_disagreement, 1e-12) << "sub-step " << sub;
    EXPECT_NEAR(decomposed.exchange_in, exchange_in_single, 1e-9) << "sub-step " << sub;
  }
  std::fprintf(stderr, "[TwoPartitionVParticipantMatchesSingleRank] max_abs_diff=%.3e\n", max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-12);
}

TEST(BrusselatorSpeciesCoreDecomposition, ThreePartitionVParticipantMatchesSingleRank) {
  NativeMeshBackend ref_backend(0, 1);
  auto reference_r = VParticipant::make(std::move(ref_backend), kNxUHalo, kNyUHalo3, kNxVHalo,
                                        kNyVHalo3, /*diffusion=*/10.0, SolverKind::Direct);
  ASSERT_TRUE(reference_r.ok()) << reference_r.error().message;
  VParticipant reference = std::move(reference_r.value());

  auto u_partitioned = make_partitioned_grid("u", kNxUHalo, kNyUHalo3, 3);
  auto ranks = build_decomposed_v_ranks(kNxUHalo, kNyUHalo3, kNxVHalo, kNyVHalo3, 3);

  const BarrierGroup group(3);
  auto cores = make_cores_concurrently<VParticipant>(3, [&](int r) {
    auto& rk = ranks[static_cast<std::size_t>(r)];
    auto made = VParticipant::make_from_grids(std::move(*rk.backend), std::move(rk.u_grid),
                                              std::move(rk.v_grid), /*diffusion=*/10.0,
                                              SolverKind::Schur, 1e-12, group.make_reducer(r, 3),
                                              group.make_exchanger_factory(r, 3));
    EXPECT_TRUE(made.ok()) << (made.ok() ? "" : made.error().message);
    return std::make_unique<VParticipant>(std::move(made.value()));
  });

  const double dt = 0.003;
  Vector required = brusselator::initial_condition(/*is_u=*/false, kNxUHalo, kNyUHalo3);

  double max_abs_diff = 0.0;
  double exchange_in_single = 0.0;
  for (int sub = 0; sub < 6; ++sub) {
    Vector ref_provided(reference.n_u());
    const auto ref = reference.step(dt, required, &ref_provided);
    ASSERT_TRUE(ref.ok()) << ref.error().message;
    exchange_in_single = reference.exchange_in();

    const auto decomposed = step_all_v_partitions(cores, u_partitioned.grids, dt, required);
    for (int c = 0; c < kNxUHalo * kNyUHalo3; ++c) {
      max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.provided[c] - ref_provided[c]));
    }
    EXPECT_LT(decomposed.exchange_in_max_disagreement, 1e-12) << "sub-step " << sub;
    EXPECT_NEAR(decomposed.exchange_in, exchange_in_single, 1e-9) << "sub-step " << sub;
  }
  std::fprintf(stderr, "[ThreePartitionVParticipantMatchesSingleRank] max_abs_diff=%.3e\n", max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-11);
}

// KNOWN_ISSUES D29 (CLOSED): the genuinely MISALIGNED 3-way split (U's 16
// rows as 6/5/5 against V's 24 as 8/8/8 -- SOME U cell crosses a V
// partition boundary, adopting cells in BOTH remap halo directions,
// measured per-rank at make_from_grids time as rank 0 "0 U, 12 V", rank 1
// "8 U, 12 V", rank 2 "8 U, 0 V" under FUSE_BRUSS_DEBUG=1). This used to
// diverge from the single-rank reference by ~3.1e-4 in exchange_in from the
// first sub-step and later crash inside the diffusion solve (an Eigen
// Block.h bounds assertion) -- a heap buffer overflow in `step()`/
// `remap_state_to_u()`, not a halo-construction defect (the pipeline itself
// is independently proven correct at this split by
// test/backend/native_halo_test.cpp's own `ThreeRanksMisalignedAdoptsExactly
// WhatTheOracleComputes`/`ThreeRanksMisalignedCompleteWeightsAndLinearField
// Reproduction`). See VParticipant::make_from_grids's own comment for the
// measured cause and fix (size the `apply_weights` destination buffer from
// the weights' own `n_rows`, not the pre-halo owned count).
TEST(BrusselatorSpeciesCoreDecomposition, ThreePartitionVParticipantMatchesSingleRankMisaligned) {
  NativeMeshBackend ref_backend(0, 1);
  auto reference_r = VParticipant::make(std::move(ref_backend), kNxUHalo, kNyUMis, kNxVHalo,
                                        kNyVMis, /*diffusion=*/10.0, SolverKind::Direct);
  ASSERT_TRUE(reference_r.ok()) << reference_r.error().message;
  VParticipant reference = std::move(reference_r.value());

  auto u_partitioned = make_partitioned_grid("u", kNxUHalo, kNyUMis, 3);
  auto ranks = build_decomposed_v_ranks(kNxUHalo, kNyUMis, kNxVHalo, kNyVMis, 3);

  const BarrierGroup group(3);
  auto cores = make_cores_concurrently<VParticipant>(3, [&](int r) {
    auto& rk = ranks[static_cast<std::size_t>(r)];
    auto made = VParticipant::make_from_grids(std::move(*rk.backend), std::move(rk.u_grid),
                                              std::move(rk.v_grid), /*diffusion=*/10.0,
                                              SolverKind::Schur, 1e-12, group.make_reducer(r, 3),
                                              group.make_exchanger_factory(r, 3));
    EXPECT_TRUE(made.ok()) << (made.ok() ? "" : made.error().message);
    return std::make_unique<VParticipant>(std::move(made.value()));
  });

  const double dt = 0.003;
  Vector required = brusselator::initial_condition(/*is_u=*/false, kNxUHalo, kNyUMis);

  double max_abs_diff = 0.0;
  double exchange_in_single = 0.0;
  for (int sub = 0; sub < 6; ++sub) {
    Vector ref_provided(reference.n_u());
    const auto ref = reference.step(dt, required, &ref_provided);
    ASSERT_TRUE(ref.ok()) << ref.error().message;
    exchange_in_single = reference.exchange_in();

    const auto decomposed = step_all_v_partitions(cores, u_partitioned.grids, dt, required);
    for (int c = 0; c < kNxUHalo * kNyUMis; ++c) {
      max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.provided[c] - ref_provided[c]));
    }
    EXPECT_LT(decomposed.exchange_in_max_disagreement, 1e-9) << "sub-step " << sub;
    EXPECT_NEAR(decomposed.exchange_in, exchange_in_single, 1e-9) << "sub-step " << sub;
  }
  // Measured (this fixture, Schur, 6 sub-steps): 1.732e-14. 1e-11 keeps the
  // same two-orders-of-magnitude margin the ALIGNED 3-way test above uses
  // over ITS OWN measured 7.994e-15 (KNOWN_ISSUES D29's "Verified correct ON
  // AN ALIGNED split" note).
  std::fprintf(stderr,
               "[ThreePartitionVParticipantMatchesSingleRankMisaligned] max_abs_diff=%.3e\n",
               max_abs_diff);
  EXPECT_LT(max_abs_diff, 1e-11);
}

// Same misaligned split, the Krylov (Pcg) solver instead of Schur -- the
// OTHER solver kind KNOWN_ISSUES D29 named as independently reproducing the
// pre-fix defect (ruling out a single solver's own code being the cause),
// over 20 sub-steps rather than 6 to rule out any slow growth the original
// "enters once, from the first sub-step" measurement did not have room to
// see.
TEST(BrusselatorSpeciesCoreDecomposition, ThreePartitionVParticipantMatchesSingleRankMisalignedPcg) {
  NativeMeshBackend ref_backend(0, 1);
  auto reference_r = VParticipant::make(std::move(ref_backend), kNxUHalo, kNyUMis, kNxVHalo,
                                        kNyVMis, /*diffusion=*/10.0, SolverKind::Direct);
  ASSERT_TRUE(reference_r.ok()) << reference_r.error().message;
  VParticipant reference = std::move(reference_r.value());

  auto u_partitioned = make_partitioned_grid("u", kNxUHalo, kNyUMis, 3);
  auto ranks = build_decomposed_v_ranks(kNxUHalo, kNyUMis, kNxVHalo, kNyVMis, 3);

  const BarrierGroup group(3);
  auto cores = make_cores_concurrently<VParticipant>(3, [&](int r) {
    auto& rk = ranks[static_cast<std::size_t>(r)];
    auto made = VParticipant::make_from_grids(std::move(*rk.backend), std::move(rk.u_grid),
                                              std::move(rk.v_grid), /*diffusion=*/10.0,
                                              SolverKind::Pcg, 1e-13, group.make_reducer(r, 3),
                                              group.make_exchanger_factory(r, 3));
    EXPECT_TRUE(made.ok()) << (made.ok() ? "" : made.error().message);
    return std::make_unique<VParticipant>(std::move(made.value()));
  });

  const double dt = 0.003;
  Vector required = brusselator::initial_condition(/*is_u=*/false, kNxUHalo, kNyUMis);

  double max_abs_diff = 0.0;
  double max_exchange_in_diff = 0.0;
  for (int sub = 0; sub < 20; ++sub) {
    Vector ref_provided(reference.n_u());
    const auto ref = reference.step(dt, required, &ref_provided);
    ASSERT_TRUE(ref.ok()) << ref.error().message;
    const double exchange_in_single = reference.exchange_in();

    const auto decomposed = step_all_v_partitions(cores, u_partitioned.grids, dt, required);
    for (int c = 0; c < kNxUHalo * kNyUMis; ++c) {
      max_abs_diff = std::max(max_abs_diff, std::abs(decomposed.provided[c] - ref_provided[c]));
    }
    max_exchange_in_diff =
        std::max(max_exchange_in_diff, std::abs(decomposed.exchange_in - exchange_in_single));
    EXPECT_LT(decomposed.exchange_in_max_disagreement, 1e-9) << "sub-step " << sub;
  }
  // Measured (this fixture, Pcg, 20 sub-steps): max_abs_diff=5.262e-14,
  // max_exchange_in_diff=1.388e-17.
  std::fprintf(stderr,
               "[ThreePartitionVParticipantMatchesSingleRankMisalignedPcg] max_abs_diff=%.3e "
               "max_exchange_in_diff=%.3e\n",
               max_abs_diff, max_exchange_in_diff);
  EXPECT_LT(max_abs_diff, 1e-11);
  EXPECT_LT(max_exchange_in_diff, 1e-9);
}
