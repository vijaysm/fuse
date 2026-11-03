// The persisted-map round trip proven in transfer_map_io_test.cpp uses SYNTHETIC
// triplets, not a map the real builder produced. This file closes that gap
// at tier 0 (no MPI): the weights come from fuse::build_supermesh_weights
// (src/backend/supermesh.cpp), the lowest-order conservative builder, on two
// genuinely non-matching 1D meshes.
//
// RANK COUNTS ARE MODELLED, NOT LAUNCHED, exactly as transfer_map_io_test.cpp
// already does for the synthetic case: a partition is nothing but a set of
// global ids, `localise()` cannot tell how many ranks exist, and the matrix
// format's own contract (docs/user/transfer-maps.md) is rank-count
// independent by construction. Write at N=2 "ranks", read at M=3 (N != M,
// both >= 2), matching the acceptance criterion's own wording. The
// MPI-launched twin of this scenario is
// test/parallel/transfer_map_reload_mpi_test.cpp; that file is compiled, not
// run, in this sandbox (MPI cannot initialise here), so THIS file is the one
// executable evidence for the claim until a host runs that one.
//
// WHY A TOLERANCE IS NEEDED AT ALL, AND WHERE ITS NUMBER COMES FROM.
// Two numbers are compared for each target row: "direct", built by calling
// build_supermesh_weights AGAIN at the read-side (M-rank) partition, and
// "reloaded", obtained by writing the N-rank weights to shards, merging, and
// calling localise() at the M-rank partition. Both describe the identical
// geometric problem (the same pair of cells, the same measure-intersection
// formula), and build_supermesh_weights is a pure function of the geometry
// of the pair it is asked about, not of what else shares the rank -- so the
// PER-ENTRY weight for one (target cell, source cell) pair is the same
// floating-point value whichever call produced it. What can differ is the
// ORDER a multi-entry row is SUMMED in when applied to a field:
// localise()'s own contract sums in ascending SOURCE GLOBAL id
// (transfer_map_io.hpp, "WHY THE FILE HOLDS GLOBAL IDS AND NOT THE MATRIX"),
// while the direct build's own row order is whatever its candidate search
// visited, which this file does not assume agrees.
//
// The bound on that difference is the recursive-summation error bound
// (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
// Theorem 4.2): a sum of n floating-point terms computed in ANY association
// order satisfies
//     | fl(sum_k x_k) - sum_k x_k | <= gamma_{n-1} * sum_k |x_k|,
//     gamma_k = k*u / (1 - k*u), u = unit roundoff = 2^-53 for double.
// Applying this once to each of the two differently-ordered sums and adding
// the two bounds (triangle inequality) gives, for row j with n_j nonzero
// entries and term magnitudes x_jk = w_jk * f_k,
//     | g_direct[j] - g_reload[j] | <= 2 * gamma_{n_j - 1} * S_j,
//     S_j = sum_k |w_jk * f_k|   (computed from the actual weights and field,
//                                  not assumed).
// n_j <= 1 makes gamma_{n_j-1} == gamma_0 == 0: a single-entry row has
// nothing to sum in either order, so the two values are expected BIT
// IDENTICAL, and the test asserts that with EXPECT_EQ rather than a
// tolerance that would hide a real defect.

#include "fuse/backend/supermesh.hpp"
#include "fuse/backend/transfer_map_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "../support/partitioned_test_backend.hpp"

namespace {


/// `n` source cells on `[0, 1]`, never partitioned: every reader and writer
/// in this file holds the WHOLE source, as a replicated source would -- the
/// only shape reachable today, and the one
/// `localise()`'s own `E-MAP-COVERAGE` check demands (a kept row's source
/// entity must be held by the rank applying it).
::GeometricLineMesh full_source(std::size_t n) {
  return GeometricLineMesh(n, /*num_ranks_in_set=*/1, /*my_rank_in_set=*/0, 0.0, 1.0);
}

/// This rank's own slice of the `n`-cell target mesh, `rank` of `ranks`.
::GeometricLineMesh target_slice(std::size_t n, int ranks, int rank) {
  return GeometricLineMesh(n, ranks, rank, 0.0, 1.0);
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

/// `gamma_k` from Higham's recursive-summation bound, Theorem 4.2.
[[nodiscard]] double gamma(std::size_t k) {
  constexpr double kU = std::numeric_limits<double>::epsilon() / 2.0;  // 2^-53
  const double ku = static_cast<double>(k) * kU;
  return ku / (1.0 - ku);
}

constexpr std::size_t kNSrc = 11;
constexpr std::size_t kNTgt = 7;  // coprime with kNSrc: no internal boundary
                                  // coincides except at the domain ends.

/// `f(global source id) = 2.3 + 0.37 * id` -- bounded away from zero over
/// every cell this test touches, deliberately not symmetric so a bug that
/// only shows up away from the mesh's own midpoint is not masked.
[[nodiscard]] double field_value(std::size_t gid) {
  return 2.3 + 0.37 * static_cast<double>(gid);
}

[[nodiscard]] std::vector<double> source_field() {
  std::vector<double> f(kNSrc);
  for (std::size_t i = 0; i < kNSrc; ++i) f[i] = field_value(i);
  return f;
}

}  // namespace

TEST(TransferMapRealBuilderReload, WriteAtTwoRanksReadAtThreeReproducesTheDirectBuild) {
  constexpr int kWriteRanks = 2;
  constexpr int kReadRanks = 3;
  static_assert(kWriteRanks != kReadRanks && kWriteRanks >= 2 && kReadRanks >= 2,
               "the acceptance criterion names N != M, both >= 2");

  // ---- write at kWriteRanks "ranks", each building its own local target
  // slice against the REAL supermesh builder -----------------------------
  std::vector<fuse::PersistedTransferMap> shards;
  for (int rank = 0; rank < kWriteRanks; ++rank) {
    const ::GeometricLineMesh src = full_source(kNSrc);
    const ::GeometricLineMesh tgt = target_slice(kNTgt, kWriteRanks, rank);
    const auto w = fuse::build_supermesh_weights(src, tgt, /*dim=*/1);
    ASSERT_TRUE(w.ok()) << "write rank " << rank << ": " << w.error().message;
    const auto p = fuse::to_persisted(w.value(), src, tgt, /*src_dim=*/1, /*tgt_dim=*/1);
    ASSERT_TRUE(p.ok()) << "write rank " << rank << ": " << p.error().message;
    shards.push_back(p.value());
  }
  const auto whole = fuse::merge_transfer_maps(shards);
  ASSERT_TRUE(whole.ok()) << whole.error().message;

  const std::vector<double> f = source_field();

  // ---- read at kReadRanks "ranks": reload vs. a fresh direct build -------
  double max_observed_rel = 0.0;
  double max_derived_rel = 0.0;
  std::size_t max_row_nnz = 0;
  double total_reload = 0.0, total_direct = 0.0;
  const double h_tgt = 1.0 / static_cast<double>(kNTgt);
  for (int rank = 0; rank < kReadRanks; ++rank) {
    const ::GeometricLineMesh src = full_source(kNSrc);
    const ::GeometricLineMesh tgt = target_slice(kNTgt, kReadRanks, rank);

    const auto reloaded = fuse::localise(whole.value(), src, tgt, 1, 1);
    ASSERT_TRUE(reloaded.ok()) << "read rank " << rank << ": " << reloaded.error().message;

    const auto direct = fuse::build_supermesh_weights(src, tgt, /*dim=*/1);
    ASSERT_TRUE(direct.ok()) << "read rank " << rank << ": " << direct.error().message;

    ASSERT_EQ(reloaded.value().n_rows, direct.value().n_rows);
    ASSERT_EQ(reloaded.value().n_rows, tgt.entity_count(1));

    const std::vector<double> g_reload = apply_map(reloaded.value(), f);
    const std::vector<double> g_direct = apply_map(direct.value(), f);

    // Per-row nnz count and |w_jk * f_k| sum, read from the DIRECT matrix
    // (the reloaded one must agree on nnz -- checked below -- and the
    // per-entry weight values are the same pure function of the pair's
    // geometry either way, so either matrix gives the same S_j).
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
          << "read rank " << rank << " row " << j
          << ": reload and the direct build disagree on how many source "
             "cells contribute -- a dropped or duplicated entry, not a "
             "rounding question";
      max_row_nnz = std::max(max_row_nnz, nnz[j]);
      if (nnz[j] <= 1) {
        // Nothing to sum in either order: the two values are the same
        // floating-point computation, bit for bit.
        EXPECT_EQ(g_reload[j], g_direct[j])
            << "read rank " << rank << " row " << j
            << " has a single contributing source cell; reload and the "
               "direct build must be bit-identical, not merely close";
        continue;
      }
      const double bound_abs = 2.0 * gamma(nnz[j] - 1) * s[j];
      const double bound_rel = bound_abs / std::abs(g_direct[j]);
      max_derived_rel = std::max(max_derived_rel, bound_rel);
      const double observed_rel =
          std::abs(g_reload[j] - g_direct[j]) / std::abs(g_direct[j]);
      max_observed_rel = std::max(max_observed_rel, observed_rel);
      EXPECT_NEAR(g_reload[j], g_direct[j], bound_abs)
          << "read rank " << rank << " row " << j << ": |g_direct-g_reload|="
          << std::abs(g_direct[j] - g_reload[j]) << " exceeds the derived "
          << "Higham recursive-summation bound " << bound_abs
          << " (n=" << nnz[j] << ")";
    }

    for (std::size_t j = 0; j < g_direct.size(); ++j) {
      total_reload += g_reload[j] * h_tgt;
      total_direct += g_direct[j] * h_tgt;
    }
  }

  // Conservation (sum_j g_j m^t_j = sum_i f_i m^s_i) holds independently of
  // summation order for EACH matrix on its own (it is the builder's own
  // audited identity, max_col_defect ~ 1e-14 per Supermesh.* in
  // supermesh_test.cpp); checked here across the FULL read-side rank set
  // for both the reloaded and the directly-built matrices, confirming the
  // per-row tolerance above did not let a genuine mass leak through by
  // accident of cancellation.
  const double h_src = 1.0 / static_cast<double>(kNSrc);
  double total_src = 0.0;
  for (const double v : f) total_src += v * h_src;
  EXPECT_NEAR(total_direct, total_src, 1e-13 * std::abs(total_src))
      << "the directly-built matrix itself must conserve mass";
  EXPECT_NEAR(total_reload, total_src, 1e-13 * std::abs(total_src))
      << "the reloaded matrix must conserve mass";

  // The acceptance criterion's own number: the derived bound
  // must be tighter than this, not merely asserted at it. If it were not,
  // the derivation above would be missing a term, not an excuse to use
  // 1e-13 as the assertion itself.
  EXPECT_LT(max_derived_rel, 1e-13)
      << "derived bound " << max_derived_rel << " (max row nnz "
      << max_row_nnz << ") does not comfortably clear the plan's own 1e-13 "
      << "acceptance number; something about this mesh pair needs revisiting";
  // The bound must not be vacuous either: a derivation that permits an
  // error orders of magnitude above anything observed would be a sign a
  // term was DROPPED in the other direction (too generous to ever fail).
  // kNSrc/kNTgt are small enough that gamma_{n-1} for the largest observed
  // row is itself a tiny multiple of u, so demand the derived bound stays
  // within three decades of machine epsilon.
  EXPECT_LT(max_derived_rel, 1e3 * std::numeric_limits<double>::epsilon());
}

TEST(TransferMapRealBuilderReload, ADistributionMismatchRefusesByName) {
  // Written for a 7-cell target at 2 "ranks". A reader built for a
  // DIFFERENT target resolution (9 cells, not 7) owns global ids the file
  // never covers -- exactly what `E-MAP-COVERAGE` exists to catch: applying
  // the reloaded matrix would otherwise leave those rows at zero, silently
  // indistinguishable from a legitimately zero transfer.
  std::vector<fuse::PersistedTransferMap> shards;
  for (int rank = 0; rank < 2; ++rank) {
    const ::GeometricLineMesh src = full_source(kNSrc);
    const ::GeometricLineMesh tgt = target_slice(kNTgt, 2, rank);
    const auto w = fuse::build_supermesh_weights(src, tgt, 1);
    ASSERT_TRUE(w.ok()) << w.error().message;
    const auto p = fuse::to_persisted(w.value(), src, tgt, 1, 1);
    ASSERT_TRUE(p.ok()) << p.error().message;
    shards.push_back(p.value());
  }
  const auto whole = fuse::merge_transfer_maps(shards);
  ASSERT_TRUE(whole.ok()) << whole.error().message;

  const ::GeometricLineMesh src = full_source(kNSrc);
  const ::GeometricLineMesh mismatched_tgt = target_slice(/*n=*/9, /*ranks=*/1, /*rank=*/0);
  const auto reloaded = fuse::localise(whole.value(), src, mismatched_tgt, 1, 1);
  ASSERT_FALSE(reloaded.ok())
      << "a reader declaring a different target resolution than the map "
         "was built for must be refused, not silently zero-filled";
  EXPECT_EQ(reloaded.error().code, "E-MAP-COVERAGE");
}
