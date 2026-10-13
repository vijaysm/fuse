// Review 2026-10-11 TD-1: docs/user/brusselator.md:401-402 claims three
// refusals "tested": `E-BRUSS-MESH-INDEX`, `E-BRUSS-U-REACTION`,
// `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED`. Only the third is fired anywhere
// in test/ (`DirectSolverRefusesWhenDecomposed`,
// test/verification/brusselator_species_core_decomposition_test.cpp).
// This file closes the `E-BRUSS-U-REACTION` gap: `UCore::step`'s implicit
// reaction fixed point (species_core.hpp:605-647) gives up and refuses
// after `kMaxReactionIterations` (50) solves that never contract to
// `kReactionTolerance`. `docs/dev/schur-slab-solver.md`'s sibling fix
// aside, `E-BRUSS-MESH-INDEX` is addressed separately: see this file's
// own trailing comment and docs/user/brusselator.md's updated wording
// for why no test here fires it.
//
// FORCING THE REACTION FIXED POINT PAST ITS CAP. On a single cell
// (nx=ny=1, so the periodic Laplacian's self-loops cancel to exactly
// zero -- `periodic_laplacian(1,1)` -- and the diffusion solve reduces to
// a plain scalar divide by `1+extra_diagonal`), the whole iteration
// collapses to the SCALAR recurrence
//   u_{k+1} = (u0 + dt*B + 0.5*dt*(q_k + q_{k-1})) / (1 + dt)
//   q_{k+1} = u_{k+1}^2 * v - 3.4 * u_{k+1}
// (species_core.hpp:605-641's exact formulas; `3.4` because
// `reaction_kokkos_span` (brusselator_physics.hpp:487) HARDCODES that
// coefficient rather than reading `PhysicsConfig::A` -- `cfg.A` has no
// effect on this recurrence at all, confirmed by first choosing
// parameters assuming `cfg.A` was live, finding the chosen case actually
// converged, and re-deriving against the real formula). `v = required`
// held fixed across the inner loop. At `dt=3.0`, `cfg.B=0.0`, `v=0.5`
// (forcing off), a standalone Python replay of this recurrence from
// `UCore`'s own real single-cell initial condition
// (`u0 = 22*(0.5*0.5)^1.5 = 2.75`) runs all 50 steps with `|q_{k+1}-q_k|`
// oscillating between roughly 16 and 24 (`1e-13 * (1+|q|)` is never
// remotely close) -- bounded (`|u|` never exceeds ~5.6, `|q|` never
// exceeds ~18.5), not a blow-up that would instead hit
// `E-BRUSS-U-SOLVE`'s non-finite check first.

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

#include <fuse/backend/native_mesh.hpp>

#include "brusselator_physics.hpp"
#include "species_core.hpp"

namespace {

using brusselator::NativeGrid;
using brusselator::PhysicsConfig;
using brusselator::SolverKind;
using brusselator::UCore;
using brusselator::Vector;

NativeGrid make_one_cell_grid() {
  fuse::native::NativeMeshBackend backend(0, 1);
  auto r = NativeGrid::make(backend, "g", /*nx=*/1, /*ny=*/1);
  EXPECT_TRUE(r.ok()) << r.error().message;
  return std::move(r.value());
}

TEST(BrusselatorRefusalCoverage, UReactionFixedPointRefusesPastFiftySolves) {
  const NativeGrid grid = make_one_cell_grid();

  PhysicsConfig cfg;
  cfg.solver = SolverKind::Direct;
  cfg.diffusion = 10.0;  // irrelevant: L_h is identically zero on one cell
  // cfg.A is left at its default: reaction_kokkos_span hardcodes 3.4 and
  // never reads PhysicsConfig::A (see the file header comment above).
  cfg.B = 0.0;
  cfg.forcing = false;  // isolate the reaction recurrence from the forcing term
  Vector required(1);
  required[0] = 0.5;

  UCore core(grid, cfg, required);
  const double dt = 3.0;
  const auto stepped = core.step(/*t_begin=*/0.0, /*t_end=*/dt, dt, required);

  ASSERT_FALSE(stepped.ok()) << "expected the reaction fixed point to exceed its 50-solve cap, "
                                 "got state() = "
                              << core.state()[0];
  EXPECT_EQ(stepped.error().code, "E-BRUSS-U-REACTION");
}

}  // namespace

// `E-BRUSS-MESH-INDEX` (brusselator_physics.hpp:356/367, NativeGrid::make):
// fires only when a loaded mesh's own cell centroids disagree with the
// row band/formula `NativeGrid::make` itself just used to build the
// "generated:quad:<nx>x<ny>" source it loads -- i.e. only on a
// corrupted or hand-crafted MeshView, never on the honest generator this
// grid always drives. `NativeMeshBackend`'s public surface (native_
// backend.hpp) has no way to substitute a different view for a name's
// DEFAULT (unplaced) slot: `load()` always (re)generates one matching
// its own `nx`/`ny` by construction; loading the same name twice makes
// `mesh_id()` refuse `E-MESH-NAME-AMBIGUOUS` before any centroid is read
// (verified by reading `NativeMeshBackend::load`/`mesh_id`,
// src/backend/native_backend.cpp); `set_placed_view`/`commit_remap_halo`
// write the PLACED slot a `ComponentId` reads, not the default slot
// `NativeGrid::make`'s own `backend.view(id)` call reads, and the latter
// is besides documented "nothing but the native halo machinery may call
// this" (native_backend.hpp). No test in this file drives this refusal;
// docs/user/brusselator.md's "State" section is corrected to record it
// as implemented, deliberately untested, with this reasoning.
