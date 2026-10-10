// src/linalg/comm_mpi.cpp -- the only translation unit in fuse::linalg that
// names mpi.h (comm.hpp's own header comment: the same isolation
// fuse::partition::detail::collectives and fuse::search::distributed use
// for their own MPI symbols). Only added to fuse-coupler-linalg's sources,
// and only linked against MPI::MPI_CXX, when FUSE_ENABLE_MPI (src/CMakeLists.txt).
//
// THE GHOST EXCHANGE PATTERN, BUILT ONCE. Exactly the two-phase shape
// src/backend/moab_backend.cpp's halo exchange already uses for a
// different entity set (its own header comment there, "Tell each owner
// what we want, and learn what we are asked for"), specialised to the case
// where GhostPattern already carries the owner's own local index for every
// ghost (no global-id lookup is needed to derive it, unlike MOAB's halo,
// which starts from global ids and has to resolve ownership from them):
//
//   1. Group this rank's own ghosts by owner into `req_counts`/`req_displs`
//      (ascending owner rank) and `fill_index` (fill_index[k] is which
//      GHOST SLOT the k-th grouped request answers -- the unpack map).
//      The VALUES sent in that request are `remote_local_index[g]` itself
//      (no translation: the pattern already names the owner's own local
//      position for the entry it wants).
//   2. One MPI_Alltoall of `req_counts` against every other rank's own
//      request counts gives `ans_counts`/`ans_displs`: how many requests
//      THIS rank must answer, and from whom.
//   3. One MPI_Alltoallv of the request VALUES (grouped by req_counts/
//      req_displs) into `ans_index` (sized by ans_counts/ans_displs) tells
//      this rank exactly which of ITS OWN owned entries every other rank
//      wants, in the order that rank will expect them back -- the pack map.
//
// Steps 1-3 run once, in make_mpi_exchanger. `update()` repeats only the
// data motion: pack `owned[ans_index[k]]` into a send buffer ordered by
// `ans_counts`/`ans_displs`, one MPI_Alltoallv into a receive buffer
// ordered by `req_counts`/`req_displs`, then unpack `ghosts[fill_index[k]]
// = recv[k]`. A single MPI_Alltoallv per update() call is what the task's
// own contract names explicitly; persistent point-to-point communication
// would save the (fixed, amortised-once) pattern's setup cost on every
// call, which Alltoallv already does not pay again -- the only remaining
// saving would be the Alltoallv call's own internal bookkeeping versus a
// matched Isend/Irecv set, and nothing measured here shows that bookkeeping
// is the bottleneck, so the simpler, already-proven-correct-elsewhere
// collective is kept.

// Explicit, not transitive: the feature macro is read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"

#include "fuse/linalg/comm.hpp"

#include "fuse/base/error.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::linalg {

#if FUSE_HAVE_MPI

namespace {

MPI_Comm resolve(int comm_f) { return MPI_Comm_f2c(comm_f); }

class MpiReducer : public Reducer {
 public:
  explicit MpiReducer(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }

  [[nodiscard]] double sum(double local) const override {
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return global;
  }

  void sum(double* inout, std::size_t n) const override {
    if (n == 0) return;
    MPI_Allreduce(MPI_IN_PLACE, inout, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, comm_);
  }

  [[nodiscard]] double max(double local) const override {
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return global;
  }

  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return size_; }

 private:
  MPI_Comm comm_;
  int rank_ = 0;
  int size_ = 1;
};

class MpiExchanger : public GhostExchanger {
 public:
  MpiExchanger(MPI_Comm comm, std::size_t owned_n, std::size_t ghost_n,
              std::vector<int> req_counts, std::vector<int> req_displs,
              std::vector<std::int64_t> fill_index, std::vector<int> ans_counts,
              std::vector<int> ans_displs, std::vector<std::int64_t> ans_index)
      : comm_(comm),
        owned_n_(owned_n),
        ghost_n_(ghost_n),
        req_counts_(std::move(req_counts)),
        req_displs_(std::move(req_displs)),
        fill_index_(std::move(fill_index)),
        ans_counts_(std::move(ans_counts)),
        ans_displs_(std::move(ans_displs)),
        ans_index_(std::move(ans_index)) {}

  [[nodiscard]] std::size_t ghost_size() const override { return ghost_n_; }

  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const override {
    FUSE_ASSERT(owned_n == owned_n_,
               "MpiExchanger::update called with owned_n={}, but this exchanger's pattern was "
               "built for owned_n={}; the vector and the exchanger disagree about how many "
               "entries this participant owns",
               owned_n, owned_n_);
    FUSE_ASSERT(ghost_n == ghost_n_,
               "MpiExchanger::update called with ghost_n={}, but this exchanger's pattern "
               "carries {} ghosts",
               ghost_n, ghost_n_);

    const int send_total = ans_displs_.back();
    std::vector<double> send_buf(static_cast<std::size_t>(send_total), 0.0);
    for (int k = 0; k < send_total; ++k) {
      const std::int64_t idx = ans_index_[static_cast<std::size_t>(k)];
      send_buf[static_cast<std::size_t>(k)] = owned[idx];
    }

    const int recv_total = req_displs_.back();
    std::vector<double> recv_buf(static_cast<std::size_t>(recv_total), 0.0);
    const int rc = MPI_Alltoallv(send_buf.data(), ans_counts_.data(), ans_displs_.data(),
                                 MPI_DOUBLE, recv_buf.data(), req_counts_.data(),
                                 req_displs_.data(), MPI_DOUBLE, comm_);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-LINALG-MPI-ALLTOALLV", "MPI_Alltoallv of ghost values failed"};
    }
    for (int k = 0; k < recv_total; ++k) {
      ghosts[fill_index_[static_cast<std::size_t>(k)]] = recv_buf[static_cast<std::size_t>(k)];
    }
    return Result<void>::success();
  }

 private:
  MPI_Comm comm_;
  std::size_t owned_n_;
  std::size_t ghost_n_;
  std::vector<int> req_counts_;
  std::vector<int> req_displs_;          // size + 1
  std::vector<std::int64_t> fill_index_;  // size == ghost_n_, grouped-by-owner order
  std::vector<int> ans_counts_;
  std::vector<int> ans_displs_;          // size + 1
  std::vector<std::int64_t> ans_index_;  // size == ans_displs_.back(); local owned indices to send
};

}  // namespace

std::shared_ptr<const Reducer> make_mpi_reducer(int comm_f) {
  return std::make_shared<const MpiReducer>(resolve(comm_f));
}

std::shared_ptr<const GhostExchanger> make_mpi_exchanger(int comm_f, const GhostPattern& pattern,
                                                          std::size_t owned_n) {
  MPI_Comm comm = resolve(comm_f);
  int size = 1;
  MPI_Comm_size(comm, &size);
  const std::size_t np = static_cast<std::size_t>(size);
  const std::size_t ghost_n = pattern.owner_rank.size();
  FUSE_ASSERT(pattern.remote_local_index.size() == ghost_n,
             "GhostPattern has {} owner entries but {} remote_local_index entries; one entry "
             "per ghost is required",
             ghost_n, pattern.remote_local_index.size());

  // Phase 1: group this rank's own ghosts by owner.
  std::vector<int> req_counts(np, 0);
  for (int owner : pattern.owner_rank) {
    FUSE_ASSERT(owner >= 0 && static_cast<std::size_t>(owner) < np,
               "GhostPattern names owner rank {}, outside [0, {})", owner, np);
    ++req_counts[static_cast<std::size_t>(owner)];
  }
  std::vector<int> req_displs(np + 1, 0);
  for (std::size_t r = 0; r < np; ++r) req_displs[r + 1] = req_displs[r] + req_counts[r];

  std::vector<std::int64_t> req_values(ghost_n, 0);
  std::vector<std::int64_t> fill_index(ghost_n, 0);
  {
    std::vector<int> cursor(req_displs.begin(), req_displs.end() - 1);
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern.owner_rank[g]);
      const int slot = cursor[owner]++;
      req_values[static_cast<std::size_t>(slot)] = pattern.remote_local_index[g];
      fill_index[static_cast<std::size_t>(slot)] = static_cast<std::int64_t>(g);
    }
  }

  // Phase 2: learn how many requests every other rank will send to US.
  std::vector<int> ans_counts(np, 0);
  MPI_Alltoall(req_counts.data(), 1, MPI_INT, ans_counts.data(), 1, MPI_INT, comm);
  std::vector<int> ans_displs(np + 1, 0);
  for (std::size_t r = 0; r < np; ++r) ans_displs[r + 1] = ans_displs[r] + ans_counts[r];

  // Phase 3: learn WHICH of our owned entries every other rank wants.
  std::vector<std::int64_t> ans_index(static_cast<std::size_t>(ans_displs.back()), 0);
  MPI_Alltoallv(req_values.data(), req_counts.data(), req_displs.data(), MPI_INT64_T,
               ans_index.data(), ans_counts.data(), ans_displs.data(), MPI_INT64_T, comm);
  for (std::int64_t idx : ans_index) {
    FUSE_ASSERT(idx >= 0 && static_cast<std::size_t>(idx) < owned_n,
               "a peer requested local owned index {}, outside [0, {}); the requester's "
               "RowOwnership and this rank's own owned_n disagree",
               idx, owned_n);
  }

  return std::make_shared<const MpiExchanger>(comm, owned_n, ghost_n, std::move(req_counts),
                                              std::move(req_displs), std::move(fill_index),
                                              std::move(ans_counts), std::move(ans_displs),
                                              std::move(ans_index));
}

#endif  // FUSE_HAVE_MPI

}  // namespace fuse::linalg
