// Review 2026-10-11 NL-1: `DiffusionSolver::factor(dt, diffusion,
// extra_diagonal)` takes `diffusion` as a per-call argument, but all three
// implementations (`DirectDiffusionSolver`, `SchurDiffusionSolver`,
// `KrylovDiffusionSolver`, examples/brusselator/species_core.hpp) used to
// key their factorization cache on `(dt, extra_diagonal)` alone. A call at
// an unchanged `dt`/`extra_diagonal` but a different `diffusion` then
// reused a stale factorization of the WRONG matrix -- silently wrong, not
// refused. This is the defect class `c2c7162` ("Key StokesSolver's LU
// cache on the pressure-gradient form as well as dt") already closed for
// `fsi::StokesSolver`'s own LU cache (test/verification/
// fsi_stokes_mms_test.cpp's `FsiStokesLuCache` suite); this file applies
// the same regression shape to all three `DiffusionSolver` kinds: factor
// at `diffusion_a`, solve; factor again at the SAME dt/extra_diagonal but
// `diffusion_b != diffusion_a`, solve; the second solve must differ from
// the first and must match a solver built FRESH at `diffusion_b` (no
// cache to go stale). All three solvers run at `num_ranks() == 1`, where
// `DirectDiffusionSolver` is applicable and `Schur`/`Krylov` reduce to the
// whole-grid case -- the cache-key defect is in the per-instance guard,
// not in any cross-rank behaviour, so single rank is enough to fire it.

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <fuse/backend/native_mesh.hpp>
#include <fuse/linalg/comm.hpp>

#include "brusselator_physics.hpp"
#include "species_core.hpp"

namespace {

using brusselator::DiffusionSolver;
using brusselator::make_diffusion_solver;
using brusselator::NativeGrid;
using brusselator::SolverKind;
using brusselator::Vector;

constexpr int kNx = 4;
constexpr int kNy = 4;
constexpr double kDt = 0.01;
constexpr double kExtraDiagonal = 0.0;
constexpr double kDiffusionA = 1.0;
constexpr double kDiffusionB = 5.0;

Vector probe_rhs() {
  Vector rhs(kNx * kNy);
  for (int k = 0; k < kNx * kNy; ++k) {
    rhs[k] = 1.0 + 0.25 * static_cast<double>(k);
  }
  return rhs;
}

// Builds a single-rank `NativeGrid` -- the same construction
// test/verification/brusselator_species_core_decomposition_test.cpp's own
// whole-grid reference uses. `NativeGrid::make` reads `backend` only to
// copy `rank()`/`num_ranks()`/the row band out (brusselator_physics.hpp's
// own `NativeGrid` class, "nothing here keeps a reference to it"), so the
// backend need not outlive the returned grid.
NativeGrid make_single_rank_grid() {
  fuse::native::NativeMeshBackend backend(0, 1);
  auto r = NativeGrid::make(backend, "g", kNx, kNy);
  EXPECT_TRUE(r.ok()) << r.error().message;
  return std::move(r.value());
}

std::unique_ptr<DiffusionSolver> build_solver(SolverKind kind, const NativeGrid& grid) {
  return make_diffusion_solver(grid, kind, /*rel_tol=*/1e-10, fuse::linalg::make_serial_reducer(),
                               brusselator::serial_exchanger_factory());
}

// Runs the cache-staleness regression for one `SolverKind`: two solves at
// the SAME dt/extra_diagonal but different `diffusion` through one
// long-lived solver must differ, and the second must match a solver built
// fresh at `diffusion_b` (nothing to go stale).
void check_refactors_on_changed_diffusion(SolverKind kind) {
  const NativeGrid grid = make_single_rank_grid();
  const Vector rhs = probe_rhs();

  auto solver = build_solver(kind, grid);
  ASSERT_TRUE(solver->factor(kDt, kDiffusionA, kExtraDiagonal).ok());
  auto first = solver->solve(rhs);
  ASSERT_TRUE(first.ok()) << first.error().message;

  ASSERT_TRUE(solver->factor(kDt, kDiffusionB, kExtraDiagonal).ok());
  auto second = solver->solve(rhs);
  ASSERT_TRUE(second.ok()) << second.error().message;

  auto fresh_solver = build_solver(kind, grid);
  ASSERT_TRUE(fresh_solver->factor(kDt, kDiffusionB, kExtraDiagonal).ok());
  auto fresh = fresh_solver->solve(rhs);
  ASSERT_TRUE(fresh.ok()) << fresh.error().message;

  ASSERT_EQ(second.value().size(), first.value().size());
  double changed = 0.0;
  for (int k = 0; k < first.value().size(); ++k) {
    changed = std::max(changed, std::abs(second.value()[k] - first.value()[k]));
  }
  // Before the fix: the second factor() call is a no-op (same dt/extra as
  // the first), so `second` is bit-identical to `first` even though
  // diffusion changed -- `changed` is exactly 0.0, which is the failure
  // this assertion catches.
  EXPECT_GT(changed, 1e-6) << "solve() at a changed diffusion reproduced the "
                              "stale factorization's result (cache key missing diffusion)";

  ASSERT_EQ(second.value().size(), fresh.value().size());
  double worst_vs_fresh = 0.0;
  for (int k = 0; k < fresh.value().size(); ++k) {
    worst_vs_fresh = std::max(worst_vs_fresh, std::abs(second.value()[k] - fresh.value()[k]));
  }
  EXPECT_LT(worst_vs_fresh, 1e-9)
      << "solve() after a diffusion change disagrees with a freshly built solver at the same "
         "diffusion, worst abs diff "
      << worst_vs_fresh;
}

TEST(BrusselatorDiffusionSolverCache, DirectSolverRefactorsOnChangedDiffusion) {
  check_refactors_on_changed_diffusion(SolverKind::Direct);
}

TEST(BrusselatorDiffusionSolverCache, SchurSolverRefactorsOnChangedDiffusion) {
  check_refactors_on_changed_diffusion(SolverKind::Schur);
}

TEST(BrusselatorDiffusionSolverCache, PcgSolverRefactorsOnChangedDiffusion) {
  check_refactors_on_changed_diffusion(SolverKind::Pcg);
}

TEST(BrusselatorDiffusionSolverCache, GmresSolverRefactorsOnChangedDiffusion) {
  check_refactors_on_changed_diffusion(SolverKind::Gmres);
}

}  // namespace
