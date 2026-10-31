// Real-MPI acceptance: a transfer
// map built by the REAL supermesh (conservative) builder on two non-matching
// meshes, persisted at N real ranks and reloaded at M real ranks (N != M,
// both >= 2), reproduces the directly-built map's own result; and the
// distribution check refuses a partition mismatch by name. The
// no-MPI twin of this scenario, with "ranks" modelled rather than
// launched (docs/dev/testing.md), is
// test/backend/transfer_map_real_builder_reload_test.cpp -- see that file's
// own header for the Higham recursive-summation bound both files assert,
// and for why a single-entry row is checked bit-identical rather than to a
// tolerance.
//
// Registered with fuse_add_mpi_test, RANKS 2 3; needs a working MPI
// launcher and a two-invocation mpirun sequence (write at 2 ranks, then
// read at 3).
//
// TWO-INVOCATION PROTOCOL, copied from
// test/parallel/brusselator_restart_mpi_test.cpp's own
// WriteAtMoreRanksRestoreAtFewerByGlobalId: N=2 and M=3 are both real rank
// counts, and no single mpiexec invocation can change its own world size
// mid-run, so writing at 2 and reading at 3 needs two separate process
// launches against the same path. `FUSE_TRANSFER_MAP_RELOAD_MPI_FIXED_PATH`
// names the shared path prefix; `FUSE_TRANSFER_MAP_RELOAD_MPI_ROLE` (`write`
// or `read`) tells this invocation which half of the sequence it is --
// exactly the brusselator file's own "odd-one-out" reasoning: "3 then 2" and
// "2 then 3" are both meaningful sequences and only the host's own script
// knows which it is running. Neither variable set: skip, naming why, rather
// than silently asserting nothing (ctest's own default invocation, with
// neither variable set, exercises this path at both registered rank counts).
#include "fuse/backend/supermesh.hpp"
#include "fuse/backend/transfer_map_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include "../support/partitioned_test_backend.hpp"

namespace {

/// The write side of the two-invocation sequence always runs at this many
/// ranks; the read side always runs at `kReadRanks`. Both are named here,
/// once, rather than read from an environment variable, so the two
/// invocations and the two constants in this file cannot
/// drift apart silently.
constexpr int kWriteRanks = 2;
constexpr int kReadRanks = 3;
static_assert(kWriteRanks != kReadRanks && kWriteRanks >= 2 && kReadRanks >= 2,
             "the acceptance criterion names N != M, both >= 2");

constexpr std::size_t kNSrc = 11;
constexpr std::size_t kNTgt = 7;  // coprime with kNSrc, as in the tier-0 twin

::GeometricLineMesh full_source() {
  return ::GeometricLineMesh(kNSrc, /*num_ranks_in_set=*/1, /*my_rank_in_set=*/0, 0.0, 1.0);
}
::GeometricLineMesh target_slice(int ranks, int rank) {
  return ::GeometricLineMesh(kNTgt, ranks, rank, 0.0, 1.0);
}

[[nodiscard]] double field_value(std::size_t gid) {
  return 2.3 + 0.37 * static_cast<double>(gid);
}
[[nodiscard]] std::vector<double> source_field() {
  std::vector<double> f(kNSrc);
  for (std::size_t i = 0; i < kNSrc; ++i) f[i] = field_value(i);
  return f;
}
[[nodiscard]] std::vector<double> apply_map(const fuse::RemapWeights& w,
                                            const std::vector<double>& f) {
  std::vector<double> g(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    g[static_cast<std::size_t>(w.row[k])] +=
        w.val[k] * f[static_cast<std::size_t>(w.col[k])];
  }
  return g;
}
/// `gamma_k` from Higham's recursive-summation bound (Theorem 4.2,
/// "Accuracy and Stability of Numerical Algorithms", 2nd ed.) -- see
/// transfer_map_real_builder_reload_test.cpp's header for the full
/// derivation this file reuses unchanged.
[[nodiscard]] double gamma(std::size_t k) {
  constexpr double kU = std::numeric_limits<double>::epsilon() / 2.0;
  const double ku = static_cast<double>(k) * kU;
  return ku / (1.0 - ku);
}

std::string shard_path(const std::string& fixed, int write_rank) {
  return fixed + "." + std::to_string(write_rank) + ".fmap";
}

}  // namespace

TEST(TransferMapReloadMpi, WriteAtTwoRanksReadAtThreeReproducesTheDirectBuild) {
  const char* fixed = std::getenv("FUSE_TRANSFER_MAP_RELOAD_MPI_FIXED_PATH");
  if (fixed == nullptr) {
    GTEST_SKIP() << "needs FUSE_TRANSFER_MAP_RELOAD_MPI_FIXED_PATH set and this "
                    "binary run once at "
                 << kWriteRanks << " ranks (role=write) and once at " << kReadRanks
                 << " ranks (role=read) against the same path";
  }
  const std::string path(fixed);
  const char* role = std::getenv("FUSE_TRANSFER_MAP_RELOAD_MPI_ROLE");
  ASSERT_NE(role, nullptr) << "FUSE_TRANSFER_MAP_RELOAD_MPI_ROLE must be 'write' "
                              "or 'read' alongside the fixed path";

  if (std::string(role) == "write") {
    ASSERT_EQ(fuse::testing::size(), kWriteRanks)
        << "the write half of this sequence must be launched at exactly "
        << kWriteRanks << " ranks";
    const int rank = fuse::testing::rank();
    const ::GeometricLineMesh src = full_source();
    const ::GeometricLineMesh tgt = target_slice(kWriteRanks, rank);
    const auto w = fuse::build_supermesh_weights(src, tgt, /*dim=*/1);
    ASSERT_TRUE(w.ok()) << "rank " << rank << ": " << w.error().message;
    const auto p = fuse::to_persisted(w.value(), src, tgt, /*src_dim=*/1, /*tgt_dim=*/1);
    ASSERT_TRUE(p.ok()) << "rank " << rank << ": " << p.error().message;
    ASSERT_TRUE(fuse::write_transfer_map(shard_path(path, rank), p.value()).ok());
    return;
  }

  ASSERT_EQ(std::string(role), "read");
  ASSERT_EQ(fuse::testing::size(), kReadRanks)
      << "the read half of this sequence must be launched at exactly " << kReadRanks
      << " ranks";
  const int rank = fuse::testing::rank();

  std::vector<fuse::PersistedTransferMap> shards;
  for (int w = 0; w < kWriteRanks; ++w) {
    const auto s = fuse::read_transfer_map(shard_path(path, w));
    ASSERT_TRUE(s.ok()) << "shard " << w << ": " << s.error().code << ": "
                        << s.error().message;
    shards.push_back(s.value());
  }
  const auto whole = fuse::merge_transfer_maps(shards);
  ASSERT_TRUE(whole.ok()) << whole.error().message;

  const std::vector<double> f = source_field();
  const ::GeometricLineMesh src = full_source();
  const ::GeometricLineMesh tgt = target_slice(kReadRanks, rank);

  const auto reloaded = fuse::localise(whole.value(), src, tgt, 1, 1);
  ASSERT_TRUE(reloaded.ok()) << "rank " << rank << ": " << reloaded.error().message;
  const auto direct = fuse::build_supermesh_weights(src, tgt, /*dim=*/1);
  ASSERT_TRUE(direct.ok()) << "rank " << rank << ": " << direct.error().message;
  ASSERT_EQ(reloaded.value().n_rows, direct.value().n_rows);
  ASSERT_EQ(reloaded.value().n_rows, tgt.entity_count(1));

  const std::vector<double> g_reload = apply_map(reloaded.value(), f);
  const std::vector<double> g_direct = apply_map(direct.value(), f);

  std::vector<std::size_t> nnz(direct.value().n_rows, 0);
  std::vector<double> s(direct.value().n_rows, 0.0);
  for (std::size_t k = 0; k < direct.value().val.size(); ++k) {
    const auto row = static_cast<std::size_t>(direct.value().row[k]);
    const auto col = static_cast<std::size_t>(direct.value().col[k]);
    nnz[row] += 1;
    s[row] += std::abs(direct.value().val[k] * f[col]);
  }
  std::vector<std::size_t> nnz_reload(reloaded.value().n_rows, 0);
  for (std::size_t k = 0; k < reloaded.value().val.size(); ++k) {
    nnz_reload[static_cast<std::size_t>(reloaded.value().row[k])] += 1;
  }

  for (std::size_t j = 0; j < g_direct.size(); ++j) {
    ASSERT_EQ(nnz_reload[j], nnz[j])
        << "rank " << rank << " row " << j << ": reload and the direct "
           "build disagree on the contributing source-cell count";
    if (nnz[j] <= 1) {
      EXPECT_EQ(g_reload[j], g_direct[j])
          << "rank " << rank << " row " << j << " has one contributing "
             "source cell; reload and the direct build must be bit-identical";
      continue;
    }
    const double bound_abs = 2.0 * gamma(nnz[j] - 1) * s[j];
    EXPECT_NEAR(g_reload[j], g_direct[j], bound_abs)
        << "rank " << rank << " row " << j << ": |g_direct-g_reload|="
        << std::abs(g_direct[j] - g_reload[j]) << " exceeds the derived "
        << "Higham recursive-summation bound " << bound_abs << " (n=" << nnz[j]
        << ") -- see transfer_map_real_builder_reload_test.cpp for the "
           "derivation";
  }
}
