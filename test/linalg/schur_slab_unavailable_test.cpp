// test/linalg/schur_slab_unavailable_test.cpp 
//
// src/linalg/schur_slab_unavailable.cpp is the TU this library compiles
// instead of src/linalg/schur_slab.cpp whenever FUSE_LINALG_BACKEND is not
// "eigen" (src/CMakeLists.txt's own backend-selected target_sources call).
// Without this file, no test row would exercise that build at all: the default
// (eigen) configuration links schur_slab.cpp and never reaches this code
// path, and the builtin configuration compiled the unavailable variant with
// nothing asserting its refusal fires with the documented code and message. This row is gated the OPPOSITE way every
// Eigen-linked row in this file is (just above):
// registered only when FUSE_LINALG_BACKEND is NOT eigen, so it builds and
// runs in exactly the configuration that compiles schur_slab_unavailable.cpp,
// and is declared gated (not silently absent) in the eigen configuration
// where SchurSlabSolver::factor() would actually attempt a real
// factorisation instead of refusing immediately.
//
// solve()'s own refusal in that TU is NOT exercised here: SchurSlabSolver's
// constructor is private and factor() is the only public path to an
// instance, so a never-factored solver is unreachable from outside the
// class in ANY configuration (schur_slab_unavailable.cpp's own header
// comment makes the same observation) -- there is no way for a test built
// against the public API to obtain a SchurSlabSolver to call solve() on
// when factor() always fails. Recorded here rather than silently skipped.

#include "fuse/linalg/schur_slab.hpp"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"

namespace {

using fuse::linalg::make_serial_reducer;
using fuse::linalg::RowOwnership;
using fuse::linalg::SchurSlabSolver;

TEST(SchurSlabSolverUnavailable, FactorRefusesWithTheBuiltinBackendCode) {
  constexpr int nx = 4;
  constexpr std::int64_t ny = 10;
  RowOwnership owners{{0}};
  auto result = SchurSlabSolver::factor(nx, ny, /*j_lo=*/0, /*j_hi=*/ny, /*coefficient=*/0.5,
                                         owners, make_serial_reducer());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-SCHUR-BACKEND");
  EXPECT_EQ(result.error().message,
            "SchurSlabSolver needs the Eigen linear-algebra backend "
            "(FUSE_LINALG_BACKEND=eigen); this build selected 'builtin'");
}

// A row-band size that WOULD trip E-LINALG-SCHUR-SLAB under the eigen
// backend (schur_slab_test.cpp's own OneRowBandRefusesWithoutFactorizing)
// still reports E-LINALG-SCHUR-BACKEND here: the unavailable variant
// refuses before it ever looks at the row band, so the backend refusal
// takes priority over the row-band refusal rather than racing it.
TEST(SchurSlabSolverUnavailable, BackendRefusalPrecedesTheRowBandSizeCheck) {
  constexpr int nx = 4;
  constexpr std::int64_t ny = 10;
  RowOwnership owners{{0, 9}};
  auto result = SchurSlabSolver::factor(nx, ny, /*j_lo=*/9, /*j_hi=*/10, /*coefficient=*/0.5,
                                         owners, make_serial_reducer());
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-SCHUR-BACKEND");
}

}  // namespace
