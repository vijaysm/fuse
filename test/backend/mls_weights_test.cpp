// Moving-least-squares remap weights (`fuse/backend/mls_weights.hpp`) for
// point clouds. Covers option parsing, the mesh entity set/discretization
// resolution `build_mls_weights` shares no code with `intx_weights.cpp`'s
// own (a deliberate, separate, smaller copy -- see the header comment),
// exact reproduction of the declared polynomial degree in 1D/2D/3D and on
// scattered points on a sphere, convergence order under refinement, every
// named rejection, and a cross-check against ArborX 2.1's own
// `ArborX::Interpolation::MovingLeastSquares::interpolate()` -- the PUBLIC
// class, not the `Details::` internals `mls_weights.hpp`'s own header
// comment explains why this file does not build on.

#include "fuse/backend/mls_weights.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <ArborX_InterpMovingLeastSquares.hpp>
#include <Kokkos_Core.hpp>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace {

using fuse::CouplingGraph;
using fuse::Discretization;
using fuse::MeshDecl;
using fuse::MeshId;
using fuse::MeshView;
using fuse::MlsRadial;
using fuse::Point;
using fuse::Region;
using fuse::RegionId;
using fuse::RemapFlags;
using fuse::RemapPlan;
using fuse::RemapWeights;
using fuse::Result;

// ---------------------------------------------------------------------------
// Test doubles. `FlatPointView` reports a flat list of point positions at
// exactly one (dim, count) slot -- everything `build_mls_weights` reads
// (`entity_count`, `centroid`) and nothing it does not (`measure` throws,
// matching mesh_view.hpp's own "dim 0 has no geometric measure" contract;
// `simplex_count`/`simplex_points` are never called by this builder and are
// trivial stubs). One class covers PointCloud (dim 0), ContinuousGLL order 1
// (vertex, dim 0) and FV/DiscontinuousGLL order 0 (cell centroid, dim ==
// whatever `resolve_mls_side` decided) alike, because all three ask this
// class the same two questions.
// ---------------------------------------------------------------------------

class FlatPointView final : public MeshView {
 public:
  FlatPointView(int dim, std::vector<Point> pts) : dim_(dim), pts_(std::move(pts)) {}

  int dimension() const override { return dim_; }
  std::size_t entity_count(int dim) const override {
    return dim == dim_ ? pts_.size() : 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int dim, std::size_t i) const override {
    check(dim, i);
    return true;
  }
  Point centroid(int dim, std::size_t i) const override {
    check(dim, i);
    return pts_[i];
  }
  double measure(int, std::size_t) const override {
    throw std::out_of_range("FlatPointView::measure: this view defines no measure");
  }
  std::size_t global_id(int dim, std::size_t i) const override {
    check(dim, i);
    return i;
  }
  std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, Point*) const override {}

 private:
  void check(int dim, std::size_t i) const {
    if (dim != dim_ || i >= pts_.size()) {
      throw std::out_of_range("FlatPointView: index out of range");
    }
  }
  int dim_;
  std::vector<Point> pts_;
};

/// Holds any number of named `FlatPointView`s; `mesh_id` resolves by name,
/// exactly as `MeshBackend::mesh_id`'s own contract asks for -- decoupling
/// the backend's storage order from the graph's `add_mesh` order, the same
/// reason `mesh_view.hpp`'s own doc gives for going through this lookup
/// rather than a position.
class TestBackend final : public fuse::MeshBackend {
 public:
  MeshId add(std::string name, int dim, std::vector<Point> pts) {
    names_.push_back(std::move(name));
    views_.push_back(std::make_unique<FlatPointView>(dim, std::move(pts)));
    return MeshId{static_cast<std::uint32_t>(views_.size() - 1)};
  }

  Result<void> load(const MeshDecl&) override { return Result<void>::success(); }
  Result<MeshId> mesh_id(std::string_view n) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == n) return MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "TestBackend: no such mesh"};
  }
  Result<const MeshView*> view(MeshId id) const override {
    if (!id.valid() || id.value() >= views_.size()) {
      return fuse::Failure{"E-MESH-ID", "TestBackend: bad mesh id"};
    }
    return views_[id.value()].get();
  }
  Result<std::size_t> region_size(const Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  std::string name() const override { return "test-flat-points"; }

 private:
  std::vector<std::string> names_;
  std::vector<std::unique_ptr<MeshView>> views_;
};

/// A graph plus backend holding exactly two mesh entity sets, "src" and "tgt", each a
/// flat point set at dimension `dim` (0 for PointCloud/vertex, or the cell
/// dimension for an FV-style side). Both mesh entity sets default to `Discretization::
/// PointCloud` on the `RemapPlan` the test builds; a case that wants a
/// different `(disc, order)` sets `plan.src_disc`/`tgt_disc` itself.
struct Fixture {
  CouplingGraph graph;
  TestBackend   backend;
  RegionId      src;
  RegionId      tgt;

  Fixture(int dim, std::vector<Point> src_pts, std::vector<Point> tgt_pts) {
    backend.add("src", dim, std::move(src_pts));
    const MeshId src_mesh = graph.add_mesh(MeshDecl{.name = "src", .source = "test"});
    src = graph.add_region(Region{.name = "src_region",
                                  .mesh = src_mesh,
                                  .entity_set = "",
                                  .dimension = dim});

    backend.add("tgt", dim, std::move(tgt_pts));
    const MeshId tgt_mesh = graph.add_mesh(MeshDecl{.name = "tgt", .source = "test"});
    tgt = graph.add_region(Region{.name = "tgt_region",
                                  .mesh = tgt_mesh,
                                  .entity_set = "",
                                  .dimension = dim});
  }

  [[nodiscard]] RemapPlan plan(std::string method, int dim = 0,
                               Discretization disc = Discretization::PointCloud) const {
    RemapPlan p;
    p.src = src;
    p.tgt = tgt;
    p.src_disc = disc;
    p.tgt_disc = disc;
    (void)dim;
    p.method = std::move(method);
    return p;
  }
};

// ---------------------------------------------------------------------------
// Point-set generators.
// ---------------------------------------------------------------------------

std::vector<Point> grid_1d(int n, double lo, double hi) {
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double t = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
    pts.push_back(Point{lo + t * (hi - lo), 0.0, 0.0});
  }
  return pts;
}

std::vector<Point> grid_2d(int nx, int ny) {
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      const double x = (nx == 1) ? 0.0 : static_cast<double>(i) / (nx - 1);
      const double y = (ny == 1) ? 0.0 : static_cast<double>(j) / (ny - 1);
      pts.push_back(Point{x, y, 0.0});
    }
  }
  return pts;
}

std::vector<Point> grid_3d(int nx, int ny, int nz) {
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(nx) * ny * nz);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        const double x = (nx == 1) ? 0.0 : static_cast<double>(i) / (nx - 1);
        const double y = (ny == 1) ? 0.0 : static_cast<double>(j) / (ny - 1);
        const double z = (nz == 1) ? 0.0 : static_cast<double>(k) / (nz - 1);
        pts.push_back(Point{x, y, z});
      }
    }
  }
  return pts;
}

/// `n` points scattered on the unit sphere, deterministic (fixed seed): a
/// normalised Gaussian vector is uniform on the sphere.
std::vector<Point> sphere_points(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    double x = g(rng), y = g(rng), z = g(rng);
    const double r = std::sqrt(x * x + y * y + z * z);
    x /= r;
    y /= r;
    z /= r;
    pts.push_back(Point{x, y, z});
  }
  return pts;
}

/// `n` points scattered uniformly at random in `[lo, hi]^3`, deterministic
/// (fixed seed). A quadratic-or-higher MLS fit on a PERFECTLY REGULAR grid is
/// a classic degeneracy in the meshfree-methods literature: a symmetric
/// neighbour stencil can make the moment matrix exactly singular in specific
/// directions (measured here: a 6x6x6 grid, k = 16, degree 2 -- rank 9 of
/// the 10 needed, condition estimate 2.2e5, `E-MLS-RANK`), which is a
/// property of the SAMPLING, not a defect in this builder -- the rejection is
/// doing exactly its job. A scattered cloud is both the more representative
/// case for a point-cloud method and the one that avoids the symmetry.
std::vector<Point> random_cube_points(int n, double lo, double hi, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(lo, hi);
  std::vector<Point> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) pts.push_back(Point{u(rng), u(rng), u(rng)});
  return pts;
}

// ---------------------------------------------------------------------------
// An independent reference polynomial: total degree <= `degree` in `dim`
// variables, coefficients supplied by the caller in the SAME total-degree
// enumeration `fuse::mls_basis_size` counts (this file's own copy of the
// enumeration -- deliberately independent of `mls_weights.cpp`'s internal
// `polynomial_basis`, which this test never sees). Any complete total-degree
// basis spans the same polynomial space, so this enumeration need not match
// the library's internal one for the reproduction property to hold.
// ---------------------------------------------------------------------------

double eval_poly(int dim, int degree, const std::vector<double>& c, double x, double y,
                 double z) {
  std::size_t idx = 0;
  double sum = 0.0;
  for (int total = 0; total <= degree; ++total) {
    if (dim == 1) {
      sum += c[idx++] * std::pow(x, total);
    } else if (dim == 2) {
      for (int a = total; a >= 0; --a) {
        const int b = total - a;
        sum += c[idx++] * std::pow(x, a) * std::pow(y, b);
      }
    } else {
      for (int a = total; a >= 0; --a) {
        for (int b = total - a; b >= 0; --b) {
          const int cc = total - a - b;
          sum += c[idx++] * std::pow(x, a) * std::pow(y, b) * std::pow(z, cc);
        }
      }
    }
  }
  return sum;
}

std::vector<double> fixed_coeffs(std::size_t n) {
  std::vector<double> c(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Deterministic, not all-equal, not all-integer -- a constant-coefficient
    // vector would not exercise the cross terms.
    c[i] = 0.6 + 0.37 * static_cast<double>(i) - 0.11 * static_cast<double>(i * i % 5);
  }
  return c;
}

/// Builds `plan`'s weights, asserts success, applies them to `f` sampled on
/// the source cloud, and returns the reconstructed target values.
std::vector<double> reconstruct(const RemapPlan& plan, const CouplingGraph& graph,
                                const TestBackend& backend, const std::vector<Point>& src_pts,
                                double (*f)(const Point&)) {
  const auto w_r = fuse::build_mls_weights(plan, graph, backend);
  EXPECT_TRUE(w_r.ok()) << (w_r.ok() ? "" : w_r.error().message);
  if (!w_r.ok()) return {};
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size());
  for (std::size_t i = 0; i < src_pts.size(); ++i) src_vals[i] = f(src_pts[i]);
  std::vector<double> dst_vals(w.n_rows, 0.0);
  const auto applied = fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1);
  EXPECT_TRUE(applied.ok());
  return dst_vals;
}

}  // namespace

// ===========================================================================
// parse_mls_options
// ===========================================================================

TEST(MlsOptions, ValidStringParsesAllThreeKeysInAnyOrder) {
  const auto r = fuse::parse_mls_options("mls:degree=2,k=10,radial=wendland2");
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().degree, 2);
  EXPECT_EQ(r.value().k, 10);
  EXPECT_EQ(r.value().radial, MlsRadial::Wendland2);

  const auto r2 = fuse::parse_mls_options("mls:radial=wendland6,k=7,degree=0");
  ASSERT_TRUE(r2.ok());
  EXPECT_EQ(r2.value().degree, 0);
  EXPECT_EQ(r2.value().k, 7);
  EXPECT_EQ(r2.value().radial, MlsRadial::Wendland6);
}

TEST(MlsOptions, BareMlsWithNoOptionsIsRefused) {
  const auto r = fuse::parse_mls_options("mls");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, MissingDegreeIsRefused) {
  const auto r = fuse::parse_mls_options("mls:k=5,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, MissingKIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, MissingRadialIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,k=5");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, UnknownRadialNameIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,k=5,radial=gaussian");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, UnknownKeyIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,k=5,radial=wendland0,extra=1");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, DuplicateKeyIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,degree=3,k=5,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, NonPositiveKIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=2,k=0,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, NonIntegerDegreeIsRefused) {
  const auto r = fuse::parse_mls_options("mls:degree=two,k=5,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, MethodNotStartingWithMlsIsRefused) {
  const auto r = fuse::parse_mls_options("tempest:degree=2,k=5,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsOptions, MlsWithNoColonSuffixIsRefused) {
  const auto r = fuse::parse_mls_options("mlsfoo:degree=2,k=5,radial=wendland0");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

// ===========================================================================
// mls_basis_size / mls_crbf
// ===========================================================================

TEST(MlsBasisSize, MatchesHandComputedBinomialCoefficients) {
  EXPECT_EQ(fuse::mls_basis_size(1, 0), 1u);
  EXPECT_EQ(fuse::mls_basis_size(1, 3), 4u);
  EXPECT_EQ(fuse::mls_basis_size(2, 0), 1u);
  EXPECT_EQ(fuse::mls_basis_size(2, 1), 3u);
  EXPECT_EQ(fuse::mls_basis_size(2, 2), 6u);
  EXPECT_EQ(fuse::mls_basis_size(3, 1), 4u);
  EXPECT_EQ(fuse::mls_basis_size(3, 2), 10u);
}

TEST(MlsCrbf, IsPositiveOnZeroToOneAndVanishesExactlyAtOne) {
  // ArborX's own tabulated Wendland polynomials are not all normalised to 1
  // at x == 0 (Wendland<4>'s constant term is 3, not 1) -- and this builder's
  // weight formula does not need them to be: any positive overall rescaling
  // of every phi_j for one target's fit cancels out of `phi_j * (c . basis_j)`
  // (`docs/user/point-cloud-remap.md`'s derivation), so the properties that
  // actually matter are positivity on [0, 1) and an exact zero at the
  // boundary -- not a specific value at zero.
  for (MlsRadial r : {MlsRadial::Wendland0, MlsRadial::Wendland2, MlsRadial::Wendland4,
                      MlsRadial::Wendland6}) {
    EXPECT_GT(fuse::mls_crbf(r, 0.0), 0.0);
    EXPECT_GT(fuse::mls_crbf(r, 0.5), 0.0);
    EXPECT_GT(fuse::mls_crbf(r, 0.99), 0.0);
    EXPECT_NEAR(fuse::mls_crbf(r, 1.0), 0.0, 1e-14);
  }
}

// ===========================================================================
// Exactness: MLS reproduces a polynomial of the declared degree exactly.
// ===========================================================================

TEST(MlsWeights, ExactForDegree2Polynomial1D) {
  const int dim = 1, degree = 2;
  auto src_pts = grid_1d(25, 0.0, 1.0);
  auto tgt_pts = grid_1d(11, 0.05, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=2,k=8,radial=wendland2");

  const auto coeffs = fixed_coeffs(fuse::mls_basis_size(dim, degree));
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = eval_poly(dim, degree, coeffs, src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    const double expect =
        eval_poly(dim, degree, coeffs, tgt_pts[i].x, tgt_pts[i].y, tgt_pts[i].z);
    EXPECT_NEAR(dst_vals[i], expect, 1e-9) << "target " << i;
  }
}

TEST(MlsWeights, ExactForDegree1Polynomial2D) {
  const int dim = 2, degree = 1;
  auto src_pts = grid_2d(9, 9);
  auto tgt_pts = grid_2d(5, 5);
  for (auto& p : tgt_pts) {
    p.x = 0.1 + 0.7 * p.x;
    p.y = 0.1 + 0.7 * p.y;
  }
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=1,k=8,radial=wendland2");

  const auto coeffs = fixed_coeffs(fuse::mls_basis_size(dim, degree));
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = eval_poly(dim, degree, coeffs, src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    const double expect =
        eval_poly(dim, degree, coeffs, tgt_pts[i].x, tgt_pts[i].y, tgt_pts[i].z);
    EXPECT_NEAR(dst_vals[i], expect, 1e-9) << "target " << i;
  }
}

TEST(MlsWeights, ExactForDegree2Polynomial3D) {
  const int dim = 3, degree = 2;
  // Scattered, not a regular grid (see random_cube_points's own comment): a
  // symmetric stencil is a classic degeneracy for an even-degree fit.
  auto src_pts = random_cube_points(120, 0.0, 1.0, /*seed=*/3);
  auto tgt_pts = random_cube_points(20, 0.15, 0.85, /*seed=*/4);
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=2,k=16,radial=wendland2");

  const auto coeffs = fixed_coeffs(fuse::mls_basis_size(dim, degree));
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = eval_poly(dim, degree, coeffs, src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    const double expect =
        eval_poly(dim, degree, coeffs, tgt_pts[i].x, tgt_pts[i].y, tgt_pts[i].z);
    EXPECT_NEAR(dst_vals[i], expect, 1e-8) << "target " << i;
  }
}

TEST(MlsWeights, ExactForDegree1PolynomialOnScatteredSpherePoints) {
  // The ambient dimension is 3 even though every point lies on a 2-manifold
  // (mls_weights.hpp's own "WHAT DEGREE AND K MEAN" section): the basis is
  // ordinary 3D monomials, restricted to the sphere by the data.
  const int dim = 3, degree = 1;
  auto src_pts = sphere_points(200, /*seed=*/1);
  auto tgt_pts = sphere_points(30, /*seed=*/2);
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=1,k=12,radial=wendland2");

  const auto coeffs = fixed_coeffs(fuse::mls_basis_size(dim, degree));
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = eval_poly(dim, degree, coeffs, src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    const double expect =
        eval_poly(dim, degree, coeffs, tgt_pts[i].x, tgt_pts[i].y, tgt_pts[i].z);
    EXPECT_NEAR(dst_vals[i], expect, 1e-8) << "target " << i;
  }
}

// ===========================================================================
// Convergence order: degree + 1 for a smooth (non-polynomial) field under
// refinement of a quasi-uniform cloud.
// ===========================================================================

TEST(MlsWeights, ConvergenceOrderIsDegreePlusOneOnRefinement) {
  const int degree = 1;  // order = degree + 1 = 2
  const auto smooth = [](double x, double y) { return std::sin(M_PI * x) * std::cos(M_PI * y); };

  const std::vector<Point> tgt_pts = {Point{0.5, 0.5, 0.0}, Point{0.3, 0.7, 0.0},
                                      Point{0.62, 0.18, 0.0}};
  std::vector<double> err;
  for (const int n : {9, 17, 33}) {
    auto src_pts = grid_2d(n, n);
    Fixture fx(0, src_pts, tgt_pts);
    const auto plan = fx.plan("mls:degree=1,k=10,radial=wendland2");
    const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
    ASSERT_TRUE(w_r.ok()) << w_r.error().message;
    const RemapWeights& w = w_r.value();

    std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
    for (std::size_t i = 0; i < src_pts.size(); ++i) {
      src_vals[i] = smooth(src_pts[i].x, src_pts[i].y);
    }
    ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());

    double max_err = 0.0;
    for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
      max_err = std::max(max_err, std::fabs(dst_vals[i] - smooth(tgt_pts[i].x, tgt_pts[i].y)));
    }
    err.push_back(max_err);
  }

  ASSERT_EQ(err.size(), 3u);
  const double order01 = std::log2(err[0] / err[1]);
  const double order12 = std::log2(err[1] / err[2]);
  // degree + 1 == 2, expected; a generous band around it (finite k, a fixed
  // rather than asymptotic h) rather than pinning the theoretical value.
  EXPECT_GT(order01, 1.3) << "err = " << err[0] << ", " << err[1] << ", " << err[2];
  EXPECT_GT(order12, 1.3) << "err = " << err[0] << ", " << err[1] << ", " << err[2];
}

// ===========================================================================
// Coincident points must not crash.
// ===========================================================================

TEST(MlsWeights, DuplicateSourcePointsDoNotCrashAndDegreeZeroStillReproducesAConstant) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  // Three exact duplicates of the same location, mixed in among the rest.
  src_pts.push_back(src_pts[3]);
  src_pts.push_back(src_pts[3]);
  src_pts.push_back(src_pts[3]);
  auto tgt_pts = grid_1d(5, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=0,k=5,radial=wendland2");

  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size(), 7.25), dst_vals(w.n_rows, 0.0);
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (double v : dst_vals) EXPECT_NEAR(v, 7.25, 1e-12);
}

TEST(MlsWeights, SourcePointExactlyAtTargetDoesNotDivideByZero) {
  auto src_pts = grid_1d(9, 0.0, 1.0);
  // The 5th source point sits exactly at the first target point.
  std::vector<Point> tgt_pts = {src_pts[4], Point{0.2, 0.0, 0.0}, Point{0.8, 0.0, 0.0}};
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=1,k=6,radial=wendland2");

  const auto coeffs = fixed_coeffs(fuse::mls_basis_size(1, 1));
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  std::vector<double> src_vals(src_pts.size()), dst_vals(w.n_rows, 0.0);
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = eval_poly(1, 1, coeffs, src_pts[i].x, 0.0, 0.0);
  }
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), dst_vals.data(), 1).ok());
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    EXPECT_TRUE(std::isfinite(dst_vals[i]));
    EXPECT_NEAR(dst_vals[i], eval_poly(1, 1, coeffs, tgt_pts[i].x, 0.0, 0.0), 1e-9);
  }
}

// ===========================================================================
// Rejections.
// ===========================================================================

TEST(MlsRefusals, MalformedMethodStringIsEMlsOptions) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  auto tgt_pts = grid_1d(3, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=2,k=5");  // radial missing
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-OPTIONS");
}

TEST(MlsRefusals, InvalidRegionIdIsEMlsRegion) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  auto tgt_pts = grid_1d(3, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=1,k=4,radial=wendland2");
  plan.src = RegionId{999};
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-REGION");
}

TEST(MlsRefusals, SubsetEntitySetIsEMlsRegion) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  auto tgt_pts = grid_1d(3, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  fx.graph.add_region(Region{
      .name = "subset", .mesh = MeshId{0}, .entity_set = "boundary", .dimension = 0});
  auto plan = fx.plan("mls:degree=1,k=4,radial=wendland2");
  plan.src = RegionId{2};  // the subset region just added
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-REGION");
}

TEST(MlsRefusals, UnsupportedDiscretizationIsEMlsDisc) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  auto tgt_pts = grid_1d(3, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=1,k=4,radial=wendland2");
  plan.src_disc = Discretization::ContinuousGLL;
  plan.src_order = 3;  // only order 1 has a point extraction
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-DISC");
}

TEST(MlsRefusals, KBelowBasisSizeIsEMlsNeighbors) {
  auto src_pts = grid_2d(9, 9);
  auto tgt_pts = grid_2d(3, 3);
  Fixture fx(0, src_pts, tgt_pts);
  // degree 2 in 2D needs 6 basis functions; k = 4 is short of that.
  auto plan = fx.plan("mls:degree=2,k=4,radial=wendland2");
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-NEIGHBORS");
}

TEST(MlsRefusals, TooFewSourcePointsIsEMlsNeighbors) {
  auto src_pts = grid_1d(3, 0.0, 1.0);  // only 3 source points
  auto tgt_pts = grid_1d(2, 0.2, 0.8);
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=1,k=5,radial=wendland2");  // k = 5 > 3 source points
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-NEIGHBORS");
}

TEST(MlsRefusals, AllCoincidentNeighboursIsRankDeficient) {
  // Every one of the k "neighbours" is the SAME point: the local moment
  // matrix for a degree-2 fit (6 basis functions) has rank exactly 1.
  const Point p{0.4, 0.4, 0.0};
  std::vector<Point> src_pts(8, p);
  // Pad with a few distinct, far-away points so the source side has more
  // than k points overall (E-MLS-NEIGHBORS must not fire instead).
  src_pts.push_back(Point{5.0, 5.0, 0.0});
  src_pts.push_back(Point{6.0, 5.0, 0.0});
  std::vector<Point> tgt_pts = {Point{0.4, 0.4, 0.0}};
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=2,k=8,radial=wendland2");
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-RANK");
  EXPECT_NE(r.error().message.find("Condition estimate"), std::string::npos) << r.error().message;
}

TEST(MlsRefusals, ConservativeFlagIsEMlsUnsupported) {
  auto src_pts = grid_1d(10, 0.0, 1.0);
  auto tgt_pts = grid_1d(3, 0.1, 0.9);
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=1,k=4,radial=wendland2");
  plan.flags = RemapFlags::Conservative;
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-UNSUPPORTED");
}

TEST(MlsRefusals, MultiRankNativeBackendIsEMlsUnsupported) {
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/2);
  ASSERT_TRUE(backend.load(MeshDecl{.name = "m", .source = "generated:line:8"}).ok());
  CouplingGraph graph;
  graph.add_mesh(MeshDecl{.name = "m", .source = "generated:line:8"});
  const RegionId region =
      graph.add_region(Region{.name = "r", .mesh = MeshId{0}, .entity_set = "", .dimension = 1});
  RemapPlan plan;
  plan.src = region;
  plan.tgt = region;
  plan.src_disc = Discretization::FV;
  plan.tgt_disc = Discretization::FV;
  plan.method = "mls:degree=0,k=2,radial=wendland0";
  const auto r = fuse::build_mls_weights(plan, graph, backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-UNSUPPORTED");
}

TEST(MlsRefusals, MonotoneFlagOnADegreeTwoFitThatIsNotBoundPreservingIsEMlsMonotone) {
  auto src_pts = grid_2d(9, 9);
  auto tgt_pts = grid_2d(4, 4);
  for (auto& p : tgt_pts) {
    p.x = 0.15 + 0.6 * p.x;
    p.y = 0.15 + 0.6 * p.y;
  }
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=2,k=10,radial=wendland2");
  plan.flags = RemapFlags::Monotone;
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-MONOTONE");
}

// ===========================================================================
// Cross-check against ArborX 2.1's own MovingLeastSquares::interpolate(),
// the PUBLIC class (this file never reaches into `Details::`).
// ===========================================================================

TEST(MlsWeights, MatchesArborXsOwnInterpolateOnTheSameCloud2D) {
  using ArborXPoint = ArborX::Point<2, double>;
  using MemorySpace = Kokkos::HostSpace;
  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  auto src_pts = grid_2d(9, 9);
  auto tgt_pts = grid_2d(5, 5);
  for (auto& p : tgt_pts) {
    p.x = 0.1 + 0.7 * p.x;
    p.y = 0.1 + 0.7 * p.y;
  }
  const int degree = 2, k = 10;
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=2,k=10,radial=wendland2");
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  const auto smooth = [](double x, double y) { return std::sin(1.3 * x) + 2.0 * x * y - y; };
  std::vector<double> src_vals(src_pts.size());
  for (std::size_t i = 0; i < src_pts.size(); ++i) src_vals[i] = smooth(src_pts[i].x, src_pts[i].y);
  std::vector<double> fuse_vals(w.n_rows, 0.0);
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), fuse_vals.data(), 1).ok());

  ExecutionSpace space{};
  Kokkos::View<ArborXPoint*, MemorySpace> asrc("asrc", src_pts.size());
  Kokkos::View<ArborXPoint*, MemorySpace> atgt("atgt", tgt_pts.size());
  Kokkos::View<double*, MemorySpace> avals("avals", src_pts.size());
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    asrc(i) = ArborXPoint{src_pts[i].x, src_pts[i].y};
    avals(i) = src_vals[i];
  }
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) atgt(i) = ArborXPoint{tgt_pts[i].x, tgt_pts[i].y};

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, asrc, atgt, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{}, k);
  Kokkos::View<double*, MemorySpace> arborx_vals("arborx_vals", tgt_pts.size());
  mls.interpolate(space, avals, arborx_vals);
  (void)degree;

  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    EXPECT_NEAR(fuse_vals[i], arborx_vals(i), 1e-9) << "target " << i;
  }
}

TEST(MlsWeights, MatchesArborXsOwnInterpolateOnTheSameCloud3D) {
  using ArborXPoint = ArborX::Point<3, double>;
  using MemorySpace = Kokkos::HostSpace;
  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  auto src_pts = grid_3d(5, 5, 5);
  auto tgt_pts = grid_3d(3, 3, 3);
  for (auto& p : tgt_pts) {
    p.x = 0.15 + 0.6 * p.x;
    p.y = 0.15 + 0.6 * p.y;
    p.z = 0.15 + 0.6 * p.z;
  }
  const int k = 12;
  Fixture fx(0, src_pts, tgt_pts);
  const auto plan = fx.plan("mls:degree=1,k=12,radial=wendland0");
  const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_TRUE(w_r.ok()) << w_r.error().message;
  const RemapWeights& w = w_r.value();

  const auto smooth = [](double x, double y, double z) { return 1.0 + 2.0 * x - 3.0 * y + z; };
  std::vector<double> src_vals(src_pts.size());
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    src_vals[i] = smooth(src_pts[i].x, src_pts[i].y, src_pts[i].z);
  }
  std::vector<double> fuse_vals(w.n_rows, 0.0);
  ASSERT_TRUE(fuse::apply_weights(w, src_vals.data(), fuse_vals.data(), 1).ok());

  ExecutionSpace space{};
  Kokkos::View<ArborXPoint*, MemorySpace> asrc("asrc", src_pts.size());
  Kokkos::View<ArborXPoint*, MemorySpace> atgt("atgt", tgt_pts.size());
  Kokkos::View<double*, MemorySpace> avals("avals", src_pts.size());
  for (std::size_t i = 0; i < src_pts.size(); ++i) {
    asrc(i) = ArborXPoint{src_pts[i].x, src_pts[i].y, src_pts[i].z};
    avals(i) = src_vals[i];
  }
  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    atgt(i) = ArborXPoint{tgt_pts[i].x, tgt_pts[i].y, tgt_pts[i].z};
  }

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, asrc, atgt, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{}, k);
  Kokkos::View<double*, MemorySpace> arborx_vals("arborx_vals", tgt_pts.size());
  mls.interpolate(space, avals, arborx_vals);

  for (std::size_t i = 0; i < tgt_pts.size(); ++i) {
    EXPECT_NEAR(fuse_vals[i], arborx_vals(i), 1e-8) << "target " << i;
  }
}

// ===========================================================================
// Device-resident arcs: build_mls_weights's own
// dispatch to fuse::device::solve_mls_on_space, read from the SAME
// FUSE_EXECUTION_SPACE convention RunnerConfig::resolved_execution_space()
// and fuse::search::resolve_execution_space() both use
// (mls_weights.cpp's own comment on its resolution point). These tests set
// and restore the variable directly, independent of whether this binary's
// OWN ctest row (`backend.mls_weights.device`, test/CMakeLists.txt) also
// sets it externally, so they run as a genuine Host/Device comparison
// either way.
// ===========================================================================

namespace {

/// RAII guard: sets FUSE_EXECUTION_SPACE for its lifetime, restores
/// whatever (if anything) was there before on destruction -- so a test
/// that forces "device" never leaks that choice into a test that runs
/// after it in the same binary.
class ExecutionSpaceEnvGuard {
 public:
  explicit ExecutionSpaceEnvGuard(const char* value) {
    const char* prev = std::getenv("FUSE_EXECUTION_SPACE");
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
    if (value == nullptr) {
      unsetenv("FUSE_EXECUTION_SPACE");
    } else {
      setenv("FUSE_EXECUTION_SPACE", value, /*overwrite=*/1);
    }
  }
  ~ExecutionSpaceEnvGuard() {
    if (had_prev_) {
      setenv("FUSE_EXECUTION_SPACE", prev_value_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv("FUSE_EXECUTION_SPACE");
    }
  }
  ExecutionSpaceEnvGuard(const ExecutionSpaceEnvGuard&) = delete;
  ExecutionSpaceEnvGuard& operator=(const ExecutionSpaceEnvGuard&) = delete;

 private:
  bool had_prev_ = false;
  std::string prev_value_;
};

}  // namespace

TEST(MlsWeightsDevice, DeviceMatchesHostStructureAndValuesWithinTolerance) {
  auto src_pts = grid_2d(11, 11);
  std::vector<Point> tgt_pts = {Point{0.23, 0.61, 0.0}, Point{0.5, 0.5, 0.0},
                                Point{0.81, 0.12, 0.0}, Point{0.05, 0.95, 0.0}};
  const RemapWeights host_w = [&] {
    ExecutionSpaceEnvGuard guard("host");
    Fixture fx(0, src_pts, tgt_pts);
    const auto plan = fx.plan("mls:degree=2,k=12,radial=wendland2");
    const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
    EXPECT_TRUE(w_r.ok()) << (w_r.ok() ? "" : w_r.error().message);
    return w_r.ok() ? w_r.value() : RemapWeights{};
  }();

  const RemapWeights device_w = [&] {
    ExecutionSpaceEnvGuard guard("device");
    Fixture fx(0, src_pts, tgt_pts);
    const auto plan = fx.plan("mls:degree=2,k=12,radial=wendland2");
    const auto w_r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
    EXPECT_TRUE(w_r.ok()) << (w_r.ok() ? "" : w_r.error().message);
    return w_r.ok() ? w_r.value() : RemapWeights{};
  }();

  // Same CSR structure: identical row/col triplets, in the same order --
  // both paths read the k-nearest result from the same deterministic,
  // ascending-sorted search.md convention.
  ASSERT_EQ(host_w.n_rows, device_w.n_rows);
  ASSERT_EQ(host_w.n_cols, device_w.n_cols);
  ASSERT_EQ(host_w.row.size(), device_w.row.size());
  ASSERT_EQ(host_w.row, device_w.row);
  ASSERT_EQ(host_w.col, device_w.col);
  for (std::size_t i = 0; i < host_w.val.size(); ++i) {
    const double rel = std::fabs(host_w.val[i]) > 1e-12
                            ? std::fabs(device_w.val[i] - host_w.val[i]) / std::fabs(host_w.val[i])
                            : std::fabs(device_w.val[i] - host_w.val[i]);
    EXPECT_LE(rel, 1e-12) << "triplet " << i << ": host " << host_w.val[i] << " vs device "
                          << device_w.val[i];
  }
  EXPECT_NEAR(host_w.max_row_defect, device_w.max_row_defect, 1e-12);
  EXPECT_NEAR(host_w.min_weight, device_w.min_weight, 1e-12);
  EXPECT_EQ(host_w.monotone, device_w.monotone);
}

TEST(MlsRefusalsDevice, AllCoincidentNeighboursIsRankDeficientUnderDeviceToo) {
  // Exactly the AllCoincidentNeighboursIsRankDeficient
  // fixture (Host-mode test above): 8 EXACT duplicates of one point, so
  // the moment matrix is singular to an exact-zero pivot -- the one case
  // the two different eliminations (Host's column-pivoted QR,
  // Device's partial-pivot Gaussian elimination) are both certain to
  // agree on, since neither needs a relative tolerance to see a literal
  // zero. build_mls_weights's Device path confirms the device solve's
  // coarser verdict against this exact Host computation before raising
  // E-MLS-RANK (mls_weights.cpp's own build_mls_weights_device comment),
  // so the message -- same target, same "Condition estimate" text -- is
  // the Host path's own, not a separately worded Device rejection.
  ExecutionSpaceEnvGuard guard("device");
  const Point p{0.4, 0.4, 0.0};
  std::vector<Point> src_pts(8, p);
  src_pts.push_back(Point{5.0, 5.0, 0.0});
  src_pts.push_back(Point{6.0, 5.0, 0.0});
  std::vector<Point> tgt_pts = {Point{0.4, 0.4, 0.0}};
  Fixture fx(0, src_pts, tgt_pts);
  auto plan = fx.plan("mls:degree=2,k=8,radial=wendland2");
  const auto r = fuse::build_mls_weights(plan, fx.graph, fx.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MLS-RANK");
  EXPECT_NE(r.error().message.find("target 0's"), std::string::npos) << r.error().message;
  EXPECT_NE(r.error().message.find("Condition estimate"), std::string::npos) << r.error().message;
}
