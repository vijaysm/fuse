// Task 5.3: the ORDER OF ACCURACY of the high-order remap, measured.
//
// WHY THIS FILE EXISTS SEPARATELY FROM remap_test.cpp.
//
// A high-order method cannot be verified by an error tolerance. A scheme that
// claims order 4 and is accidentally order 1 still produces a small error on a
// fine mesh, and passes any absolute bound anyone would write down. The only
// test that distinguishes them is a refinement sequence: build the same map at
// several resolutions, measure the error against an analytic field, and fit the
// slope of log(error) against log(h).
//
// WHAT IS COMPARED AGAINST WHAT.
//
// The map takes CELL AVERAGES to CELL AVERAGES. So the input is the exact
// spherical cell average of an analytic f on each source cell, and the answer
// it is graded against is the exact spherical cell average of the same f on
// each target cell. Feeding it point samples at the cell centroid instead would
// inject an O(h^2) error of its own and cap every measured rate at 2 -- which
// is exactly the sort of thing that makes an order-4 method look like an
// order-2 one and gets explained away.
//
// The "exact" cell average is a Duffy-transformed 10x10 Gauss-Legendre rule on
// each spherical triangle of a fan decomposition, written out longhand below so
// it shares no code with anything under src/. It is degree-19 exact on the
// reference triangle; the residual quadrature error is checked two ways -- the
// total area must recover 4*pi*R^2, and the whole table is recomputed with a
// 6-point rule and must agree to 7 digits (verified during development: the
// order-4 errors agreed to every digit printed).
//
// THE GEOMETRY IS THE SPHERE because the only high-order backend there is
// remaps spherical surfaces. The DRIVER is not: `convergence_study` below takes
// a mesh-writing callback, a field, and a quadrature over the cells, and would
// run unchanged on a planar or volumetric refinement family once a backend for
// one exists.

#include <gtest/gtest.h>

#if defined(SOMA_HAVE_MOAB) && defined(SOMA_HAVE_TEMPEST)

#include <soma/mesh/moab_backend.hpp>
#include <soma/remap/remap_plan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <unistd.h>

using soma::CouplingGraph;
using soma::MeshDecl;
using soma::MoabMeshBackend;
using soma::Region;
using soma::RemapFlags;
using soma::RemapPlan;

namespace {

using V3 = std::array<double, 3>;

// ===========================================================================
// A quadrature over spherical polygons, independent of everything under src/
// ===========================================================================

/// Gauss-Legendre nodes and weights on [0, 1], by Newton iteration on the
/// Legendre polynomial. Written here rather than pulled from a library so that
/// the reference values this test grades against depend on nothing the test is
/// grading.
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(static_cast<std::size_t>(n), 0.0);
  w.assign(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double t = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double dp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2 * j + 1) * t * p1 - j * p2) / (j + 1);
      }
      dp = n * (t * p0 - p1) / (t * t - 1.0);
      const double dt = -p0 / dp;
      t += dt;
      if (std::fabs(dt) < 1e-15) break;
    }
    x[static_cast<std::size_t>(i)] = 0.5 * (1.0 - t);
    w[static_cast<std::size_t>(i)] = 1.0 / ((1.0 - t * t) * dp * dp);
  }
}

struct TriQuad {
  std::vector<double> x, w;
  explicit TriQuad(int n) { gauss_legendre(n, x, w); }
};

double det3(const V3& a, const V3& b, const V3& c) {
  return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
         a[2] * (b[0] * c[1] - b[1] * c[0]);
}

/// Integral of `f` over the spherical triangle (a, b, c) of radius `r`, and its
/// area, by central projection of the flat triangle onto the sphere.
///
/// With x(u,v) = a + u(b-a) + v(c-a) and p = r x / |x|, the spherical area
/// element is |det(a,b,c)| / |x|^3 du dv (the plane's distance to the origin
/// times the flat area element, divided by |x|^3), times r^2. The Duffy
/// substitution u = s, v = t(1-s) maps the reference triangle onto the unit
/// square with Jacobian (1-s), so a tensor Gauss rule applies directly.
template <class F>
void spherical_triangle(const V3& a, const V3& b, const V3& c, double r,
                        const TriQuad& q, F f, double& integral, double& area) {
  integral = 0.0;
  area = 0.0;
  const double d = std::fabs(det3(a, b, c));
  const std::size_t n = q.x.size();
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double s = q.x[i], t = q.x[j];
      const double weight = q.w[i] * q.w[j] * (1.0 - s);
      const double u = s, v = t * (1.0 - s), w0 = 1.0 - u - v;
      const V3 x{w0 * a[0] + u * b[0] + v * c[0], w0 * a[1] + u * b[1] + v * c[1],
                 w0 * a[2] + u * b[2] + v * c[2]};
      const double len = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      const double jac = d / (len * len * len);
      const V3 p{r * x[0] / len, r * x[1] / len, r * x[2] / len};
      integral += weight * jac * f(p);
      area += weight * jac;
    }
  }
  // `d` ~ r^3 and `len^3` ~ r^3, so the Jacobian above is dimensionless and the
  // loop has accumulated the UNIT-sphere area. The physical area is r^2 times
  // that. (It cancels out of the cell average, but not out of the L2 weight.)
  integral *= r * r;
  area *= r * r;
}

// ===========================================================================
// The mesh family: equiangular cubed spheres, optionally rotated
// ===========================================================================

std::filesystem::path scratch_dir() {
  static const std::filesystem::path dir = [] {
    std::filesystem::path d =
        std::filesystem::temp_directory_path() /
        ("soma_remap_order_" + std::to_string(static_cast<long>(::getpid())));
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
  }();
  return dir;
}

/// Corner coordinates and connectivity, kept so the test can integrate over
/// exactly the cells it wrote -- MeshView reports a centroid and a measure but
/// not a cell's corners, and the reference cell averages need the corners.
struct Surface {
  std::vector<V3>               node;
  std::vector<std::vector<int>> face;
  std::string                   path;
};

/// An equiangular cubed sphere of resolution `n` at radius `r`, rotated by
/// `rot` radians about the z axis.
///
/// The rotation matters. Without it a coarse and a fine cubed sphere share
/// every panel boundary and are related by an exact scaling, and the resulting
/// error cancellations are a property of that symmetry rather than of the
/// method. With it the two grids are in general position.
Surface cubed_sphere(const std::string& name, int n, double r, double rot) {
  Surface s;
  std::map<std::tuple<long long, long long, long long>, int> weld;
  const double quantum = 1e-9 * r;
  const double cs = std::cos(rot), sn = std::sin(rot);

  auto add = [&](double x, double y, double z) {
    const double k = r / std::sqrt(x * x + y * y + z * z);
    x *= k;
    y *= k;
    z *= k;
    const double xr = cs * x - sn * y;
    const double yr = sn * x + cs * y;
    const auto key =
        std::make_tuple(std::llround(xr / quantum), std::llround(yr / quantum),
                        std::llround(z / quantum));
    const auto it = weld.find(key);
    if (it != weld.end()) return it->second;
    const int idx = static_cast<int>(s.node.size());
    s.node.push_back(V3{xr, yr, z});
    weld.emplace(key, idx);
    return idx;
  };
  const auto dir = [](int panel, double u, double v) -> V3 {
    switch (panel) {
      case 0: return {1.0, u, v};
      case 1: return {-u, 1.0, v};
      case 2: return {-1.0, -u, v};
      case 3: return {u, -1.0, v};
      case 4: return {-v, u, 1.0};
      default: return {v, u, -1.0};
    }
  };
  for (int panel = 0; panel < 6; ++panel) {
    std::vector<int> node(static_cast<std::size_t>(n + 1) * (n + 1));
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const double a = -M_PI / 4.0 + (M_PI / 2.0) * i / n;
        const double b = -M_PI / 4.0 + (M_PI / 2.0) * j / n;
        const V3 d = dir(panel, std::tan(a), std::tan(b));
        node[static_cast<std::size_t>(j) * (n + 1) + i] = add(d[0], d[1], d[2]);
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const auto at = [&](int p, int q) {
          return node[static_cast<std::size_t>(q) * (n + 1) + p];
        };
        s.face.push_back(
            {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)});
      }
    }
  }

  const std::filesystem::path p = scratch_dir() / (name + ".vtk");
  std::ofstream f(p);
  f.precision(17);
  f << "# vtk DataFile Version 3.0\nsoma remap order test\nASCII\n"
    << "DATASET UNSTRUCTURED_GRID\nPOINTS " << s.node.size() << " double\n";
  for (const V3& q : s.node) f << q[0] << " " << q[1] << " " << q[2] << "\n";
  f << "CELLS " << s.face.size() << " " << s.face.size() * 5 << "\n";
  for (const auto& c : s.face) {
    f << "4 " << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << "\n";
  }
  f << "CELL_TYPES " << s.face.size() << "\n";
  for (std::size_t i = 0; i < s.face.size(); ++i) f << "9\n";
  f.close();
  s.path = p.string();
  return s;
}

/// Exact cell averages and cell areas of `f` over `s`, by fanning each polygon
/// into spherical triangles.
template <class F>
void cell_averages(const Surface& s, double r, const TriQuad& q, F f,
                   std::vector<double>& avg, std::vector<double>& area) {
  avg.assign(s.face.size(), 0.0);
  area.assign(s.face.size(), 0.0);
  for (std::size_t c = 0; c < s.face.size(); ++c) {
    const auto& v = s.face[c];
    double total_i = 0.0, total_a = 0.0;
    for (std::size_t k = 1; k + 1 < v.size(); ++k) {
      double ii = 0.0, aa = 0.0;
      spherical_triangle(s.node[static_cast<std::size_t>(v[0])],
                         s.node[static_cast<std::size_t>(v[k])],
                         s.node[static_cast<std::size_t>(v[k + 1])], r, q, f, ii,
                         aa);
      total_i += ii;
      total_a += aa;
    }
    area[c] = total_a;
    avg[c] = total_i / total_a;
  }
}

// ===========================================================================
// The study
// ===========================================================================

struct Level {
  double h = 0.0;
  double error = 0.0;
  std::size_t src_cells = 0, tgt_cells = 0;
};

/// Least-squares slope of log(error) against log(h) -- the observed order.
/// A pairwise ratio between the two finest levels would be cheaper and much
/// noisier; every level should get a vote.
double fitted_order(const std::vector<Level>& levels) {
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  const double n = static_cast<double>(levels.size());
  for (const Level& l : levels) {
    const double x = std::log(l.h), y = std::log(l.error);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

constexpr double kRadius = 1.0;

/// Smooth, non-polynomial, and with no symmetry a cubed sphere shares: the
/// exponential breaks any parity in x, and the product of a sine and a cosine
/// at different wavenumbers breaks the panel symmetry. A low-degree polynomial
/// would be reproduced exactly by the higher-order fits and the study would
/// measure nothing.
double smooth_field(const V3& p) {
  return 2.0 + std::sin(2.0 * p[0]) * std::cos(3.0 * p[1]) * std::sin(p[2]) +
         0.3 * std::exp(0.5 * p[0]);
}

/// Build the map at `order` on a refinement sequence and return the table.
/// `quad_points` is exposed so the caller can show the answer does not depend
/// on the reference quadrature.
std::vector<Level> convergence_study(int order, int quad_points,
                                     bool print) {
  const TriQuad quad(quad_points);
  std::vector<Level> levels;

  for (const int ns : {6, 9, 12, 18, 24}) {
    const int nt = ns * 2 / 3;
    const std::string tag = std::to_string(ns) + "_" + std::to_string(nt);
    const Surface src = cubed_sphere("src" + tag, ns, kRadius, 0.0);
    const Surface tgt = cubed_sphere("tgt" + tag, nt, kRadius, 0.37);

    CouplingGraph graph;
    const auto ma = graph.add_mesh(MeshDecl{.name = "a", .source = src.path});
    const auto mb = graph.add_mesh(MeshDecl{.name = "b", .source = tgt.path});
    const auto ra =
        graph.add_region(Region{.name = "ra", .mesh = ma, .dimension = 2});
    const auto rb =
        graph.add_region(Region{.name = "rb", .mesh = mb, .dimension = 2});
    MoabMeshBackend backend;
    EXPECT_TRUE(backend.load(graph.mesh(ma)).ok());
    EXPECT_TRUE(backend.load(graph.mesh(mb)).ok());

    const RemapPlan plan{.src = ra,
                         .tgt = rb,
                         .src_order = order,
                         .flags = RemapFlags::Conservative};
    const auto r = build_weights(plan, graph, backend);
    EXPECT_TRUE(r.ok()) << "order " << order << " at ns=" << ns << ": "
                        << r.error().code << ": " << r.error().message;
    if (!r.ok()) return {};
    const soma::RemapWeights& w = r.value();
    EXPECT_EQ(w.n_cols, src.face.size());
    EXPECT_EQ(w.n_rows, tgt.face.size());
    if (w.n_cols != src.face.size() || w.n_rows != tgt.face.size()) return {};

    // The matrix is indexed by MeshView local index; the reference averages are
    // indexed by the order the cells were WRITTEN. A permutation between the
    // two would show up as a first-order-looking error and be blamed on the
    // method, so pin the correspondence: MeshView::centroid of a cell is the
    // mean of its corners, and that is checked cell by cell here.
    const soma::MeshView& tv = *backend.view(backend.mesh_id("b").value()).value();
    for (std::size_t j = 0; j < tgt.face.size(); ++j) {
      V3 c{0.0, 0.0, 0.0};
      for (const int v : tgt.face[j]) {
        for (int k = 0; k < 3; ++k) c[static_cast<std::size_t>(k)] +=
            tgt.node[static_cast<std::size_t>(v)][static_cast<std::size_t>(k)];
      }
      const double inv = 1.0 / static_cast<double>(tgt.face[j].size());
      const auto p = tv.centroid(2, j);
      EXPECT_NEAR(p.x, c[0] * inv, 1e-12) << "cell order differs at " << j;
      EXPECT_NEAR(p.y, c[1] * inv, 1e-12) << "cell order differs at " << j;
      EXPECT_NEAR(p.z, c[2] * inv, 1e-12) << "cell order differs at " << j;
    }

    std::vector<double> fs, as, ft, at;
    cell_averages(src, kRadius, quad, smooth_field, fs, as);
    cell_averages(tgt, kRadius, quad, smooth_field, ft, at);

    // The quadrature is graded too: the cell areas must sum to the sphere.
    double total = 0.0;
    for (const double a : at) total += a;
    // 1e-9 absolute on 4*pi, i.e. 1e-10 relative. The 10-point rule reaches
    // 2e-13 and the 6-point rule 1.5e-10, so this catches a broken quadrature
    // (a wrong Jacobian is wrong in the third digit, not the eleventh) without
    // pinning the rule's own truncation error.
    EXPECT_NEAR(total, 4.0 * M_PI * kRadius * kRadius, 1e-9)
        << "the reference quadrature does not recover the sphere's area";

    std::vector<double> g(w.n_rows, 0.0);
    EXPECT_TRUE(apply_weights(w, fs.data(), g.data(), 1).ok());

    double e2 = 0.0;
    for (std::size_t j = 0; j < w.n_rows; ++j) {
      const double d = g[j] - ft[j];
      e2 += d * d * at[j];
    }
    Level l;
    l.h = std::sqrt(4.0 * M_PI / static_cast<double>(tgt.face.size()));
    l.error = std::sqrt(e2 / total);
    l.src_cells = src.face.size();
    l.tgt_cells = tgt.face.size();
    levels.push_back(l);
  }

  if (print) {
    std::printf(
        "\n[order %d, %s builder] cubed sphere 3n/2 -> n, target rotated 0.37 "
        "rad, reference quadrature %dx%d Gauss-Legendre\n",
        order, order > 1 ? "TempestRemap" : "supermesh", quad_points,
        quad_points);
    std::printf("%8s %8s %12s %14s %10s\n", "src", "tgt", "h", "L2 error",
                "rate");
    for (std::size_t i = 0; i < levels.size(); ++i) {
      double rate = 0.0;
      if (i > 0) {
        rate = std::log(levels[i - 1].error / levels[i].error) /
               std::log(levels[i - 1].h / levels[i].h);
      }
      std::printf("%8zu %8zu %12.6f %14.6e %10s", levels[i].src_cells,
                  levels[i].tgt_cells, levels[i].h, levels[i].error,
                  i == 0 ? "--" : "");
      if (i > 0) std::printf("%10.3f", rate);
      std::printf("\n");
    }
    std::printf("[order %d] least-squares fit over all %zu levels: %.4f\n",
                order, levels.size(), fitted_order(levels));
  }
  return levels;
}

}  // namespace

// ===========================================================================
// The assertions
// ===========================================================================

TEST(RemapOrder, ObservedOrderMatchesTheClaimedOrder) {
  // The band. Lower edge `p - 0.25`: this is the assertion that bites. A map
  // that had silently dropped to first order would measure ~1.5 whatever `p`
  // said, and fails it for every p >= 2. Upper edge `p + 0.9`: the measured
  // rates run about half an order ABOVE nominal on this family (1.87, 2.54,
  // 3.42, 4.52 as measured), which is ordinary super-convergence for a smooth field
  // on a quasi-uniform refinement -- but it is bounded, and a "rate" of 6 for a
  // claimed 2 would mean the error is not what this test thinks it is.
  // NOTE ON THE ORDER-1 ROW. `method` is left empty, so the plan goes to the
  // builder a user would actually get: at order 1 that is Task 5.2's supermesh
  // builder, and only at order 2 and above is it TempestRemap. The two are
  // different first-order maps (TempestRemap's own order-1 map fits at 1.51 on
  // this family, the supermesh one at 1.87), and the row is labelled with the
  // builder that produced it.
  std::vector<double> fits;
  std::vector<std::vector<Level>> tables;
  for (const int order : {1, 2, 3, 4}) {
    const std::vector<Level> levels = convergence_study(order, 10, true);
    ASSERT_EQ(levels.size(), 5u) << "order " << order;
    const double fit = fitted_order(levels);
    EXPECT_GT(fit, order - 0.25) << "order " << order << " observed " << fit;
    EXPECT_LT(fit, order + 0.9) << "order " << order << " observed " << fit;
    fits.push_back(fit);
    tables.push_back(levels);
  }

  // Strictly increasing accuracy AND a strictly increasing rate. Either alone
  // could be produced by an accident; both together cannot be produced by a
  // map that is secretly the same matrix four times.
  for (std::size_t i = 1; i < fits.size(); ++i) {
    EXPECT_GT(fits[i], fits[i - 1] + 0.5)
        << "order " << (i + 1) << " does not converge faster than order " << i;
    for (std::size_t l = 0; l < tables[i].size(); ++l) {
      EXPECT_LT(tables[i][l].error, tables[i - 1][l].error)
          << "order " << (i + 1) << " is not more accurate than order " << i
          << " at level " << l;
    }
  }
  // And the finest order-4 error is four decades below the finest order-1 one,
  // which is the practical statement the order claim amounts to.
  EXPECT_LT(tables[3].back().error, 1e-3 * tables[0].back().error);
}

TEST(RemapOrder, TheAnswerDoesNotDependOnTheReferenceQuadrature) {
  // If the reference cell averages were the accuracy bottleneck, refining the
  // quadrature would move the table. It must not: a 6-point and a 12-point
  // Gauss rule have to give the same errors to many digits, or the "order"
  // above is the order of this file's quadrature and not of the remap.
  const std::vector<Level> coarse = convergence_study(4, 6, false);
  const std::vector<Level> fine = convergence_study(4, 12, false);
  ASSERT_EQ(coarse.size(), fine.size());
  ASSERT_FALSE(coarse.empty());
  for (std::size_t i = 0; i < coarse.size(); ++i) {
    EXPECT_NEAR(coarse[i].error, fine[i].error, 1e-6 * fine[i].error)
        << "level " << i;
  }
}

#else

TEST(RemapOrder, SkippedWithoutTheHighOrderBackend) {
  // Orders above 1 come from TempestRemap; with no TempestRemap there is
  // nothing whose order could be measured. Skipped rather than deleted so the
  // absence is visible in the test list.
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB, or without TempestRemap";
}

#endif  // SOMA_HAVE_MOAB && SOMA_HAVE_TEMPEST
