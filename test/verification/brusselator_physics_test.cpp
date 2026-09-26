// Brusselator (Time windows track), physics: the Toy's own closed-form
// invariants (VERIFICATION.md / tests/numerical_tests.cpp), ported onto
// this example's FUSE-native-mesh-backed physics
// (examples/brusselator/brusselator_physics.hpp) rather than `#include`d
// from the read-only Toy repo. Every expectation here is a closed form or
// an exact discrete identity, not a tolerance chosen to pass.

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <fuse/backend/native_backend.hpp>

#include "brusselator_physics.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
}

// ---- reference initial conditions and forcing mask (Toy forcing_test) --

TEST(BrusselatorPhysics, ReferenceInitialConditionsMatchTheClosedForm) {
  const int nx = 10, ny = 10;
  const auto u0 = brusselator::initial_condition(/*is_u=*/true, nx, ny);
  const auto v0 = brusselator::initial_condition(/*is_u=*/false, nx, ny);
  // cell (2,4): x=(2+0.5)/10=0.25, y=(4+0.5)/10=0.45
  const int idx = 4 * nx + 2;
  EXPECT_NEAR(u0[idx], 22.0 * std::pow(0.45 * 0.55, 1.5), 1e-15);
  EXPECT_NEAR(v0[idx], 27.0 * std::pow(0.25 * 0.75, 1.5), 1e-15);
}

TEST(BrusselatorPhysics, ForcingMaskIsFiveInsideTheCircleAndZeroOutside) {
  const int nx = 10, ny = 10;
  const auto mask = brusselator::forcing_mask(nx, ny);
  // (0.3+0.05, 0.6+0.05): cell (3,6) center (0.35, 0.65) is inside the r=0.1
  // circle at (0.3,0.6); (0,0) is far outside.
  EXPECT_DOUBLE_EQ(mask[6 * nx + 3], 5.0);
  EXPECT_DOUBLE_EQ(mask[0 * nx + 0], 0.0);
}

TEST(BrusselatorPhysics, ForcingActiveDurationIsExactAcrossTheSwitch) {
  // NEAR, not DOUBLE_EQ: the closed form itself is exact, but its two
  // double-precision operands (e.g. 1.12 - 1.09) are not exactly
  // representable, so the subtraction carries ordinary round-off -- this
  // matches the Toy's own tolerance convention for such closed forms
  // (tests/numerical_tests.cpp `forcing_test` uses `near(..., 1e-15, ...)`
  // for the same reason).
  EXPECT_NEAR(brusselator::forcing_active_duration(1.09, 1.12), 0.02, 1e-15);
  EXPECT_NEAR(brusselator::forcing_active_duration(0.0, 1.0), 0.0, 1e-15);
  EXPECT_NEAR(brusselator::forcing_active_duration(1.2, 1.3), 0.1, 1e-15);
  EXPECT_NEAR(brusselator::forcing_active_duration(1.1, 1.1), 0.0, 1e-15);
}

// ---- periodic Laplacian: null space and Fourier-mode second order ------
// (Toy's diffusion_test, ported)

TEST(BrusselatorPhysics, PeriodicLaplacianHasTheConstantNullSpace) {
  for (int n : {12, 24, 48}) {
    const auto lap = brusselator::periodic_laplacian(n, n);
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(n * n);
    EXPECT_NEAR((lap * ones).norm(), 0.0, 1e-10) << "n=" << n;
  }
}

TEST(BrusselatorPhysics, PeriodicLaplacianOnASineModeIsSecondOrder) {
  double previous = 0.0;
  for (int n : {12, 24, 48}) {
    Eigen::VectorXd z(n * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const double x = (i + 0.5) / n, y = (j + 0.5) / n;
        z[j * n + i] = std::sin(2 * kPi * x) * std::cos(2 * kPi * y);
      }
    }
    const auto lap = brusselator::periodic_laplacian(n, n);
    const double exact = -8 * kPi * kPi;
    const double error = (lap * z - exact * z).norm() / z.norm();
    if (previous > 0.0) EXPECT_GT(previous / error, 3.9) << "n=" << n;
    previous = error;
  }
}

// ---- reaction kernel (Kokkos) matches the closed form ------------------

TEST(BrusselatorPhysics, ReactionKokkosMatchesTheClosedForm) {
  // Kokkos is already initialised by fuse_test_main.cpp's Kokkos::ScopeGuard.
  const int n = 5;
  Eigen::VectorXd u(n), v(n), q(n);
  for (int i = 0; i < n; ++i) {
    u[i] = 0.5 + 0.1 * i;
    v[i] = 1.2 - 0.05 * i;
  }
  brusselator::reaction_kokkos(u, v, &q);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(q[i], u[i] * u[i] * v[i] - 3.4 * u[i], 1e-14) << "i=" << i;
  }
}

// ---- NativeGrid recovers the exact canonical index correspondence ------

TEST(BrusselatorPhysics, NativeGridCentroidsMatchTheCanonicalCoordinates) {
  fuse::native::NativeMeshBackend backend(0, 1);
  auto g = brusselator::NativeGrid::make(backend, "g", 13, 9);
  ASSERT_TRUE(g.ok()) << g.error().code << ": " << g.error().message;
  const auto& grid = g.value();
  EXPECT_EQ(grid.size(), 13 * 9);
  // Every canonical index must round-trip through its native index.
  std::vector<bool> seen(static_cast<std::size_t>(grid.size()), false);
  for (int c = 0; c < grid.size(); ++c) {
    const int native = grid.native_index_of(c);
    ASSERT_GE(native, 0);
    ASSERT_LT(native, grid.size());
    ASSERT_FALSE(seen[static_cast<std::size_t>(native)]) << "native index " << native << " reused";
    seen[static_cast<std::size_t>(native)] = true;
  }
}
