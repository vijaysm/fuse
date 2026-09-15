#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MOAB

#include <soma/mesh/moab_backend.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/remap/remap_plan.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <unistd.h>  // getpid, for a per-process scratch directory

using soma::CouplingGraph;
using soma::MeshDecl;
using soma::MoabMeshBackend;
using soma::Region;
using soma::RemapCache;
using soma::RemapFlags;
using soma::RemapPlan;
using soma::RemapWeights;

namespace {

// ===========================================================================
// Mesh fixtures
// ===========================================================================

struct MeshPair {
  CouplingGraph   graph;
  MoabMeshBackend backend;
  soma::RegionId  a, b;
  // Kept for the tests that need the raw views.
  soma::MeshId    ma, mb;
};

/// Two meshes, loaded in graph order so the backend's ids line up with the
/// graph's (build_weights resolves by name anyway; this just keeps the fixture
/// honest).
MeshPair pair_of(const std::string& sa, const std::string& sb, int dim) {
  MeshPair s;
  s.ma = s.graph.add_mesh(MeshDecl{.name = "a", .source = sa});
  s.mb = s.graph.add_mesh(MeshDecl{.name = "b", .source = sb});
  s.a = s.graph.add_region(
      Region{.name = "ra", .mesh = s.ma, .dimension = dim});
  s.b = s.graph.add_region(
      Region{.name = "rb", .mesh = s.mb, .dimension = dim});
  EXPECT_TRUE(s.backend.load(s.graph.mesh(s.ma)).ok());
  EXPECT_TRUE(s.backend.load(s.graph.mesh(s.mb)).ok());
  return s;
}

MeshPair non_matching(const std::string& a = "generated:box:4x4x4",
                   const std::string& b = "generated:box:5x5x5") {
  return pair_of(a, b, 3);
}

const soma::MeshView& view_of(const MeshPair& s, soma::RegionId r) {
  const auto id = s.backend.mesh_id(s.graph.mesh(s.graph.region(r).mesh).name);
  EXPECT_TRUE(id.ok());
  const auto v = s.backend.view(id.value());
  EXPECT_TRUE(v.ok());
  return *v.value();
}

RemapPlan conservative(const MeshPair& s) {
  return RemapPlan{
      .src = s.a, .tgt = s.b, .flags = RemapFlags::Conservative};
}

// ===========================================================================
// Mesh files written by the test
// ===========================================================================
//
// MOAB's "generated:" sources are always the unit cube or the unit interval,
// so no pair of them can ever fail to cover the same domain -- the partial
// overlap that this task has to have a policy for is unreachable through them,
// and so is any unstructured surface. Rather than check a binary mesh into the
// repository, the tests below write small ASCII legacy-VTK files into a
// per-process temporary directory and load those. Everything the tests assert
// is therefore reproducible from source with no data dependency.

std::filesystem::path scratch_dir() {
  static const std::filesystem::path dir = [] {
    std::filesystem::path d = std::filesystem::temp_directory_path() /
                              ("soma_remap_test_" +
                               std::to_string(static_cast<long>(::getpid())));
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
  }();
  return dir;
}

struct VtkMesh {
  std::vector<std::array<double, 3>> points;
  std::vector<std::vector<int>>      cells;
  int                                vtk_type = 12;  // 3 line, 9 quad, 12 hex
};

std::string write_vtk(const std::string& name, const VtkMesh& m) {
  const std::filesystem::path p = scratch_dir() / (name + ".vtk");
  std::ofstream f(p);
  f << "# vtk DataFile Version 3.0\nsoma remap test\nASCII\n"
    << "DATASET UNSTRUCTURED_GRID\nPOINTS " << m.points.size() << " double\n";
  f.precision(17);
  for (const auto& q : m.points) {
    f << q[0] << " " << q[1] << " " << q[2] << "\n";
  }
  std::size_t total = 0;
  for (const auto& c : m.cells) total += c.size() + 1;
  f << "CELLS " << m.cells.size() << " " << total << "\n";
  for (const auto& c : m.cells) {
    f << c.size();
    for (const int v : c) f << " " << v;
    f << "\n";
  }
  f << "CELL_TYPES " << m.cells.size() << "\n";
  for (std::size_t i = 0; i < m.cells.size(); ++i) f << m.vtk_type << "\n";
  f.close();
  EXPECT_TRUE(std::filesystem::exists(p));
  return p.string();
}

/// An axis-aligned hex grid over [x0,x1] x [y0,y1] x [z0,z1].
std::string write_boxes(const std::string& name, int nx, int ny, int nz,
                        double x0, double x1, double y0, double y1, double z0,
                        double z1) {
  VtkMesh m;
  m.vtk_type = 12;
  auto id = [&](int i, int j, int k) {
    return (k * (ny + 1) + j) * (nx + 1) + i;
  };
  for (int k = 0; k <= nz; ++k) {
    for (int j = 0; j <= ny; ++j) {
      for (int i = 0; i <= nx; ++i) {
        m.points.push_back({x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny,
                            z0 + (z1 - z0) * k / nz});
      }
    }
  }
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        m.cells.push_back({id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k),
                           id(i, j + 1, k), id(i, j, k + 1),
                           id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1),
                           id(i, j + 1, k + 1)});
      }
    }
  }
  return write_vtk(name, m);
}

/// A quad mesh tiling [0,1]^2 in the z = 0 plane whose interior vertices are
/// displaced, so the cells are genuinely unstructured convex quadrilaterals
/// and no axis-aligned shortcut applies. Deterministic: a fixed LCG, not
/// <random>, so the mesh is byte-identical on every platform.
std::string write_skewed_plane(const std::string& name, int n, double amp,
                               std::uint32_t seed, bool clockwise = false) {
  VtkMesh m;
  m.vtk_type = 9;
  std::uint32_t state = seed;
  auto rnd = [&] {
    state = state * 1664525u + 1013904223u;
    return static_cast<double>(state >> 8) / static_cast<double>(1u << 24) -
           0.5;
  };
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      double x = static_cast<double>(i) / n;
      double y = static_cast<double>(j) / n;
      const double dx = rnd();
      const double dy = rnd();
      if (i > 0 && i < n && j > 0 && j < n) {
        x += amp * dx / n;
        y += amp * dy / n;
      }
      m.points.push_back({x, y, 0.0});
    }
  }
  auto id = [&](int i, int j) { return j * (n + 1) + i; };
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      std::vector<int> c{id(i, j), id(i + 1, j), id(i + 1, j + 1),
                         id(i, j + 1)};
      // A clockwise mesh is a perfectly ordinary mesh -- MeshView::measure is
      // unsigned -- but MOAB's in-plane intersection needs the other winding,
      // so one of the two meshes below is written this way on purpose.
      if (clockwise) std::reverse(c.begin(), c.end());
      m.cells.push_back(std::move(c));
    }
  }
  return write_vtk(name, m);
}

/// An equiangular cubed sphere of resolution `n` at radius `r`: 6n^2 quads and
/// 6n^2 + 2 vertices once the six panels are welded. The weld is by rounded
/// coordinate, and the vertex count is asserted by the test that uses it, so a
/// failed weld cannot pass unnoticed.
std::string write_cubed_sphere(const std::string& name, int n, double r) {
  VtkMesh m;
  m.vtk_type = 9;
  std::map<std::tuple<long long, long long, long long>, int> weld;
  const double quantum = 1e-9 * r;

  auto add = [&](double x, double y, double z) {
    const double s = r / std::sqrt(x * x + y * y + z * z);
    x *= s;
    y *= s;
    z *= s;
    const auto key = std::make_tuple(std::llround(x / quantum),
                                     std::llround(y / quantum),
                                     std::llround(z / quantum));
    const auto it = weld.find(key);
    if (it != weld.end()) return it->second;
    const int idx = static_cast<int>(m.points.size());
    m.points.push_back({x, y, z});
    weld.emplace(key, idx);
    return idx;
  };

  // Panel k maps (a, b) on the unit face to a direction in R^3.
  const auto dir = [](int panel, double u, double v) -> std::array<double, 3> {
    switch (panel) {
      case 0: return {1.0, u, v};    // +x
      case 1: return {-u, 1.0, v};   // +y
      case 2: return {-1.0, -u, v};  // -x
      case 3: return {u, -1.0, v};   // -y
      case 4: return {-v, u, 1.0};   // +z
      default: return {v, u, -1.0};  // -z
    }
  };

  for (int panel = 0; panel < 6; ++panel) {
    std::vector<int> node(static_cast<std::size_t>(n + 1) * (n + 1));
    for (int j = 0; j <= n; ++j) {
      for (int i = 0; i <= n; ++i) {
        const double a = -M_PI / 4.0 + (M_PI / 2.0) * i / n;
        const double b = -M_PI / 4.0 + (M_PI / 2.0) * j / n;
        const auto d = dir(panel, std::tan(a), std::tan(b));
        node[static_cast<std::size_t>(j) * (n + 1) + i] = add(d[0], d[1], d[2]);
      }
    }
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const auto at = [&](int p, int q) {
          return node[static_cast<std::size_t>(q) * (n + 1) + p];
        };
        m.cells.push_back(
            {at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)});
      }
    }
  }
  return write_vtk(name, m);
}

/// A latitude-longitude mesh on the sphere of radius `r`: `nlon` x `nlat`
/// quadrilaterals, degenerating to triangles in the two polar rows.
///
/// Written in a mixed-topology VTK file (cell types 5 and 9), which also
/// exercises the polygon extractor's variable corner count -- everything else
/// in this file is quads or hexes.
///
/// The reason it exists: with few latitude bands, the polar triangles have a
/// neighbourhood that is nearly degenerate in the gnomonic coordinates
/// TempestRemap fits in, and its order-4 least-squares fit exceeds the built-in
/// conditioning threshold there. That is the ONLY way this project has found to
/// reach the silent-degradation path with an order the library accepts.
std::string write_lat_lon(const std::string& name, int nlon, int nlat,
                          double r) {
  const std::filesystem::path p = scratch_dir() / (name + ".vtk");
  std::vector<std::array<double, 3>> pts;
  std::vector<std::vector<int>> cells;
  std::vector<std::vector<int>> ring(static_cast<std::size_t>(nlat) + 1);

  for (int j = 0; j <= nlat; ++j) {
    const double lat = -M_PI / 2.0 + M_PI * j / nlat;
    if (j == 0 || j == nlat) {
      const int id = static_cast<int>(pts.size());
      pts.push_back({0.0, 0.0, j == 0 ? -r : r});
      ring[static_cast<std::size_t>(j)].assign(nlon, id);
      continue;
    }
    for (int i = 0; i < nlon; ++i) {
      const double lon = 2.0 * M_PI * i / nlon;
      pts.push_back({r * std::cos(lat) * std::cos(lon),
                     r * std::cos(lat) * std::sin(lon), r * std::sin(lat)});
      ring[static_cast<std::size_t>(j)].push_back(
          static_cast<int>(pts.size()) - 1);
    }
  }
  for (int j = 0; j < nlat; ++j) {
    for (int i = 0; i < nlon; ++i) {
      const int i2 = (i + 1) % nlon;
      const auto& lo = ring[static_cast<std::size_t>(j)];
      const auto& hi = ring[static_cast<std::size_t>(j) + 1];
      if (j == 0) {
        cells.push_back({lo[0], hi[static_cast<std::size_t>(i2)],
                         hi[static_cast<std::size_t>(i)]});
      } else if (j == nlat - 1) {
        cells.push_back({lo[static_cast<std::size_t>(i)],
                         lo[static_cast<std::size_t>(i2)], hi[0]});
      } else {
        cells.push_back({lo[static_cast<std::size_t>(i)],
                         lo[static_cast<std::size_t>(i2)],
                         hi[static_cast<std::size_t>(i2)],
                         hi[static_cast<std::size_t>(i)]});
      }
    }
  }

  std::ofstream f(p);
  f.precision(17);
  f << "# vtk DataFile Version 3.0\nsoma remap test\nASCII\n"
    << "DATASET UNSTRUCTURED_GRID\nPOINTS " << pts.size() << " double\n";
  for (const auto& q : pts) f << q[0] << " " << q[1] << " " << q[2] << "\n";
  std::size_t total = 0;
  for (const auto& c : cells) total += c.size() + 1;
  f << "CELLS " << cells.size() << " " << total << "\n";
  for (const auto& c : cells) {
    f << c.size();
    for (const int v : c) f << " " << v;
    f << "\n";
  }
  f << "CELL_TYPES " << cells.size() << "\n";
  for (const auto& c : cells) f << (c.size() == 3 ? 5 : 9) << "\n";
  f.close();
  EXPECT_TRUE(std::filesystem::exists(p));
  return p.string();
}

// ===========================================================================
// Small analysis helpers, written out longhand so they share no code with the
// implementation.
// ===========================================================================

double integral(const soma::MeshView& v, int dim,
                const std::vector<double>& f) {
  double total = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) total += f[i] * v.measure(dim, i);
  return total;
}

/// `sum_i W_ji` for each target row -- the ROW property, constant preservation.
std::vector<double> row_sums(const RemapWeights& w) {
  std::vector<double> s(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    s[static_cast<std::size_t>(w.row[k])] += w.val[k];
  }
  return s;
}

/// `sum_j W_ji * m^t_j` for each source column -- the COLUMN property,
/// conservation. Note the target measure inside the sum: the unweighted column
/// sum is a different, meaningless quantity, and confusing the two is the
/// mistake this pair of helpers exists to make impossible.
std::vector<double> measure_weighted_column_sums(const RemapWeights& w,
                                                 const soma::MeshView& tgt,
                                                 int dim) {
  std::vector<double> s(w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    s[static_cast<std::size_t>(w.col[k])] +=
        w.val[k] * tgt.measure(dim, static_cast<std::size_t>(w.row[k]));
  }
  return s;
}

std::vector<double> unweighted_column_sums(const RemapWeights& w) {
  std::vector<double> s(w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    s[static_cast<std::size_t>(w.col[k])] += w.val[k];
  }
  return s;
}

double sphere_radius(const soma::MeshView& v) {
  double lo = 1e300, hi = 0.0;
  for (std::size_t i = 0; i < v.entity_count(0); ++i) {
    const auto p = v.centroid(0, i);
    const double r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    lo = std::min(lo, r);
    hi = std::max(hi, r);
  }
  return 0.5 * (lo + hi);
}

double total_measure(const soma::MeshView& v, int dim) {
  double t = 0.0;
  for (std::size_t i = 0; i < v.entity_count(dim); ++i) {
    t += v.measure(dim, i);
  }
  return t;
}

/// For each target row, the smallest and largest source value it draws on.
/// A bound-preserving map keeps `g[j]` inside `[lo[j], hi[j]]`; that is a
/// strictly stronger statement than staying inside the global source range,
/// and Task 5.3's monotonicity tests check both.
struct LocalBounds {
  std::vector<double> lo, hi;
};

LocalBounds local_bounds(const RemapWeights& w,
                         const std::vector<double>& src) {
  LocalBounds b;
  b.lo.assign(w.n_rows, 1e300);
  b.hi.assign(w.n_rows, -1e300);
  for (std::size_t k = 0; k < w.nnz(); ++k) {
    const std::size_t j = static_cast<std::size_t>(w.row[k]);
    const double v = src[static_cast<std::size_t>(w.col[k])];
    b.lo[j] = std::min(b.lo[j], v);
    b.hi[j] = std::max(b.hi[j], v);
  }
  return b;
}

/// Worst amount by which `dst` leaves its row's source range, and how many
/// rows do. Zero and zero for a bound-preserving map.
double worst_local_violation(const RemapWeights& w,
                             const std::vector<double>& src,
                             const std::vector<double>& dst,
                             std::size_t* count = nullptr) {
  const LocalBounds b = local_bounds(w, src);
  double worst = 0.0;
  std::size_t n = 0;
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    if (b.lo[j] > b.hi[j]) continue;  // empty row
    const double d = std::max(b.lo[j] - dst[j], dst[j] - b.hi[j]);
    if (d > 1e-12) {
      worst = std::max(worst, d);
      ++n;
    }
  }
  if (count != nullptr) *count = n;
  return worst;
}

RemapPlan high_order(const MeshPair& s, int order) {
  return RemapPlan{.src = s.a,
                   .tgt = s.b,
                   .src_order = order,
                   .flags = RemapFlags::Conservative};
}

}  // namespace

// ===========================================================================
// Shape, determinism, and the plain matvec
// ===========================================================================

TEST(Remap, BuildsWeightsBetweenNonMatchingMeshes) {
  MeshPair s = non_matching();
  const auto w = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
  EXPECT_EQ(w.value().n_rows, 125u);  // target cells
  EXPECT_EQ(w.value().n_cols, 64u);   // source cells
  EXPECT_FALSE(w.value().val.empty());
  // 4x4x4 against 5x5x5 on [0,1]^3: every source cell meets exactly two target
  // cells per axis except where the faces coincide, and the count is a fixed
  // property of the pair, so pin it rather than just "not empty".
  EXPECT_EQ(w.value().nnz(), 8u * 8u * 8u);
}

TEST(Remap, TripletsAreSortedDeterministically) {
  MeshPair s = non_matching();
  const auto a = build_weights(conservative(s), s.graph, s.backend);
  const auto b = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(a.value().row, b.value().row);
  EXPECT_EQ(a.value().col, b.value().col);
  EXPECT_EQ(a.value().val, b.value().val);  // bitwise, not near
  for (std::size_t k = 1; k < a.value().val.size(); ++k) {
    const bool ordered =
        a.value().row[k - 1] < a.value().row[k] ||
        (a.value().row[k - 1] == a.value().row[k] &&
         a.value().col[k - 1] < a.value().col[k]);
    EXPECT_TRUE(ordered) << "triplet " << k << " is out of (row, col) order";
  }
}

TEST(Remap, WeightsAreNonNegativeForFirstOrder) {
  MeshPair s = non_matching();
  const auto w = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(w.ok());
  for (const double v : w.value().val) EXPECT_GE(v, 0.0);
}

TEST(Remap, HandlesVectorComponents) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const RemapWeights& w = r.value();

  std::vector<double> src(w.n_cols * 3);
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    src[3 * i + 0] = 1.0;
    src[3 * i + 1] = 2.0;
    src[3 * i + 2] = 3.0;
  }
  std::vector<double> dst(w.n_rows * 3, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 3).ok());
  for (std::size_t i = 0; i < w.n_rows; ++i) {
    EXPECT_NEAR(dst[3 * i + 0], 1.0, 1e-14);
    EXPECT_NEAR(dst[3 * i + 1], 2.0, 1e-14);
    EXPECT_NEAR(dst[3 * i + 2], 3.0, 1e-14);
  }
}

TEST(Remap, IdenticalMeshesGiveTheIdentityMap) {
  MeshPair s = non_matching("generated:box:3x3x3", "generated:box:3x3x3");
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const RemapWeights& w = r.value();
  ASSERT_EQ(w.n_rows, w.n_cols);

  std::vector<double> src(w.n_cols);
  std::iota(src.begin(), src.end(), 1.0);
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  for (std::size_t i = 0; i < src.size(); ++i) {
    EXPECT_NEAR(dst[i], src[i], 1e-12);
  }
}

TEST(Remap, SameRegionOnBothSidesIsExactlyTheIdentity) {
  MeshPair s = non_matching("generated:box:3x3x3", "generated:box:4x4x4");
  const RemapPlan plan{
      .src = s.a, .tgt = s.a, .flags = RemapFlags::Conservative};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const RemapWeights& w = r.value();
  EXPECT_EQ(w.n_rows, 27u);
  EXPECT_EQ(w.nnz(), 27u);
  for (std::size_t k = 0; k < w.nnz(); ++k) {
    EXPECT_EQ(w.row[k], w.col[k]);
    EXPECT_EQ(w.val[k], 1.0);  // exactly, not nearly
  }
}

TEST(Remap, WorksInOneDimension) {
  MeshPair s = pair_of("generated:line:7", "generated:line:11", 1);
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  EXPECT_EQ(w.n_cols, 7u);
  EXPECT_EQ(w.n_rows, 11u);
  EXPECT_LT(w.max_row_defect, 1e-14);
  EXPECT_LT(w.max_col_defect, 1e-14);
}

// ===========================================================================
// The ROW property: sum_i W_ji == 1, i.e. constants survive.
// ===========================================================================

TEST(RemapRow, SumsAreUnity) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  for (const double rs : row_sums(r.value())) EXPECT_NEAR(rs, 1.0, 1e-14);
  EXPECT_LT(r.value().max_row_defect, 1e-14);
}

TEST(RemapRow, ConstantFieldTransfersExactly) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const RemapWeights& w = r.value();
  const std::vector<double> src(w.n_cols, 3.5);
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  for (const double v : dst) EXPECT_NEAR(v, 3.5, 1e-13);
}

// ===========================================================================
// The COLUMN property: sum_j W_ji m^t_j == m^s_i, i.e. the integral is kept.
// ===========================================================================

TEST(RemapColumn, MeasureWeightedSumsRecoverTheSourceMeasure) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);
  const auto cs = measure_weighted_column_sums(r.value(), tv, 3);
  for (std::size_t i = 0; i < cs.size(); ++i) {
    EXPECT_NEAR(cs[i], sv.measure(3, i), 1e-14 * sv.measure(3, i))
        << "col " << i;
  }
  EXPECT_LT(r.value().max_col_defect, 1e-14);
}

TEST(RemapColumn, PreservesTheGlobalIntegralOfAVaryingField) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const RemapWeights& w = r.value();
  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);

  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto c = sv.centroid(3, i);
    src[i] = 1.0 + c.x + 2.0 * c.y - 0.5 * c.z * c.z;  // smooth, not constant
  }
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());

  const double before = integral(sv, 3, src);
  const double after = integral(tv, 3, dst);
  EXPECT_NEAR(after, before, 1e-13 * std::fabs(before));
}

// The test that catches the row/column mix-up the Task 6.2 brief made. If the
// two properties were the same statement, this would be redundant; it is not.
TEST(RemapColumn, TheUnweightedColumnSumIsNotUnity) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const auto cs = unweighted_column_sums(r.value());
  const double worst = *std::max_element(
      cs.begin(), cs.end(),
      [](double a, double b) {
        return std::fabs(a - 1.0) < std::fabs(b - 1.0);
      });
  // 4^3 source cells of volume 1/64 spread over 5^3 target cells of volume
  // 1/125, so an unweighted column sums to about 125/64 = 1.953, nowhere near
  // one. Stating it numerically pins the orientation of the matrix: swap row
  // for col in the builder and this becomes 1.
  EXPECT_NEAR(worst, 125.0 / 64.0, 1e-12);
}

// ===========================================================================
// apply_weights: the preconditions that would be undefined behaviour in
// Release if they were asserts.
// ===========================================================================

TEST(RemapApply, RejectsBadArguments) {
  RemapWeights w;
  w.n_rows = 2;
  w.n_cols = 2;
  w.row = {0, 1};
  w.col = {0, 1};
  w.val = {1.0, 1.0};
  std::vector<double> a(2, 1.0), b(2, 0.0);

  EXPECT_EQ(apply_weights(w, a.data(), b.data(), 0).error().code,
            "E-REMAP-ARG");
  EXPECT_EQ(apply_weights(w, a.data(), b.data(), -1).error().code,
            "E-REMAP-ARG");
  EXPECT_EQ(apply_weights(w, nullptr, b.data(), 1).error().code, "E-REMAP-ARG");
  EXPECT_EQ(apply_weights(w, a.data(), nullptr, 1).error().code, "E-REMAP-ARG");
  EXPECT_TRUE(apply_weights(w, a.data(), b.data(), 1).ok());
}

TEST(RemapApply, RejectsOutOfRangeTriplets) {
  RemapWeights w;
  w.n_rows = 2;
  w.n_cols = 2;
  w.row = {0, 5};  // 5 is outside [0, 2)
  w.col = {0, 1};
  w.val = {1.0, 1.0};
  std::vector<double> a(2, 1.0), b(2, 0.0);
  const auto r = apply_weights(w, a.data(), b.data(), 1);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-COO");

  RemapWeights ragged = w;
  ragged.row = {0};
  EXPECT_EQ(apply_weights(ragged, a.data(), b.data(), 1).error().code,
            "E-REMAP-COO");
}

// ===========================================================================
// Partial overlap
// ===========================================================================

TEST(RemapPartial, TargetWiderThanSourceKeepsEveryGramOfMass) {
  // Source tiles [0,1]^3; target tiles [0,2] x [0,1]^2. Every source cell is
  // covered, so the map is exactly conservative; half the target is not, so
  // constants do NOT survive there -- and that is the documented behaviour.
  const std::string a = write_boxes("pa_src", 4, 4, 4, 0, 1, 0, 1, 0, 1);
  const std::string b = write_boxes("pa_tgt", 8, 4, 4, 0, 2, 0, 1, 0, 1);
  MeshPair s = pair_of(a, b, 3);

  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();

  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-14);
  EXPECT_NEAR(w.min_tgt_coverage, 0.0, 1e-14);  // the far half is untouched
  EXPECT_LT(w.max_col_defect, 1e-14);           // still conservative
  EXPECT_NEAR(w.max_row_defect, 1.0, 1e-12);    // and not constant-preserving

  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);
  const std::vector<double> src(w.n_cols, 2.0);
  std::vector<double> dst(w.n_rows, -1.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());

  int inside = 0, outside = 0;
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    if (tv.centroid(3, j).x < 1.0) {
      EXPECT_NEAR(dst[j], 2.0, 1e-13);
      ++inside;
    } else {
      EXPECT_EQ(dst[j], 0.0) << "an uncovered target cell must not invent a "
                                "value by extrapolation";
      ++outside;
    }
  }
  EXPECT_EQ(inside, 64);
  EXPECT_EQ(outside, 64);
  EXPECT_NEAR(integral(tv, 3, dst), integral(sv, 3, src),
              1e-13 * integral(sv, 3, src));
}

TEST(RemapPartial, SourceWiderThanTargetIsRefusedWhenConservativeIsAsked) {
  const std::string a = write_boxes("pb_src", 8, 4, 4, 0, 2, 0, 1, 0, 1);
  const std::string b = write_boxes("pb_tgt", 4, 4, 4, 0, 1, 0, 1, 0, 1);
  MeshPair s = pair_of(a, b, 3);
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_FALSE(r.ok()) << "half the source mass would be dropped silently";
  EXPECT_EQ(r.error().code, "E-REMAP-COVERAGE");
  EXPECT_NE(r.error().message.find("Conservative"), std::string::npos);
}

TEST(RemapPartial, SourceWiderThanTargetTruncatesWhenConservationIsNotAsked) {
  const std::string a = write_boxes("pc_src", 8, 4, 4, 0, 2, 0, 1, 0, 1);
  const std::string b = write_boxes("pc_tgt", 4, 4, 4, 0, 1, 0, 1, 0, 1);
  MeshPair s = pair_of(a, b, 3);
  const RemapPlan plan{.src = s.a, .tgt = s.b, .flags = RemapFlags::None};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();

  EXPECT_NEAR(w.min_src_coverage, 0.0, 1e-14);  // the far half sees nothing
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-14);
  EXPECT_LT(w.max_row_defect, 1e-14);         // constants still survive
  EXPECT_NEAR(w.max_col_defect, 1.0, 1e-12);  // but half the mass is dropped

  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);
  const std::vector<double> src(w.n_cols, 2.0);
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  // Exactly half the source domain is covered, so exactly half the integral
  // arrives -- truncated, not redistributed.
  EXPECT_NEAR(integral(tv, 3, dst), 0.5 * integral(sv, 3, src),
              1e-13 * integral(sv, 3, src));
}

TEST(RemapPartial, DisjointDomainsProduceAnEmptyMap) {
  const std::string a = write_boxes("pd_src", 3, 3, 3, 0, 1, 0, 1, 0, 1);
  const std::string b = write_boxes("pd_tgt", 3, 3, 3, 5, 6, 0, 1, 0, 1);
  MeshPair s = pair_of(a, b, 3);
  const RemapPlan plan{.src = s.a, .tgt = s.b, .flags = RemapFlags::None};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value().nnz(), 0u);
  EXPECT_EQ(r.value().min_src_coverage, 0.0);
  EXPECT_EQ(r.value().min_tgt_coverage, 0.0);
  // And with the flag, the same pair is a hard error rather than a silent
  // matrix of zeros.
  EXPECT_EQ(build_weights(conservative(s), s.graph, s.backend).error().code,
            "E-REMAP-COVERAGE");
}

// ===========================================================================
// Unstructured surface in a plane -- moab::Intx2MeshInPlane
// ===========================================================================

TEST(RemapPlane, ConservesOnSkewedUnstructuredQuads) {
  const std::string a = write_skewed_plane("plane_a", 4, 0.6, 12345u);
  // Deliberately wound the other way; see write_skewed_plane.
  const std::string b = write_skewed_plane("plane_b", 5, 0.6, 98765u, true);
  MeshPair s = pair_of(a, b, 2);
  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);
  ASSERT_EQ(sv.entity_count(2), 16u);
  ASSERT_EQ(tv.entity_count(2), 25u);
  // Both tile the same unit square, so the impossibility argument in the
  // header does not bite: the two totals agree and BOTH properties can hold.
  ASSERT_NEAR(total_measure(sv, 2), 1.0, 1e-14);
  ASSERT_NEAR(total_measure(tv, 2), 1.0, 1e-14);

  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  std::printf("[plane] nnz=%zu row defect=%.3e col defect=%.3e cov=%.15f\n",
              w.nnz(), w.max_row_defect, w.max_col_defect, w.min_src_coverage);
  EXPECT_LT(w.max_row_defect, 1e-12);
  EXPECT_LT(w.max_col_defect, 1e-12);

  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto c = sv.centroid(2, i);
    src[i] = 0.3 + c.x - 2.0 * c.y;
  }
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  EXPECT_NEAR(integral(tv, 2, dst), integral(sv, 2, src), 1e-12);
}

// ===========================================================================
// The real thing: MOAB's cubed sphere -- moab::Intx2MeshOnSphere
// ===========================================================================

#ifdef SOMA_MOAB_TEST_MESH

TEST(RemapSphere, IdentityOnMoabsCubedSphere) {
  // The same file loaded twice: two independent MOAB meshes with identical
  // geometry, so the supermesh machinery runs for real and must come back with
  // the identity. Warped, non-planar quadrilaterals throughout.
  MeshPair s = pair_of(SOMA_MOAB_TEST_MESH, SOMA_MOAB_TEST_MESH, 2);
  const soma::MeshView& sv = view_of(s, s.a);
  ASSERT_EQ(sv.entity_count(2), 3456u);

  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  std::printf(
      "[sphere identity] nnz=%zu row defect=%.3e col defect=%.3e "
      "src cov=%.15f tgt cov=%.15f\n",
      w.nnz(), w.max_row_defect, w.max_col_defect, w.min_src_coverage,
      w.min_tgt_coverage);
  EXPECT_LT(w.max_row_defect, 1e-12);
  EXPECT_LT(w.max_col_defect, 1e-12);
  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-12);
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-12);

  const std::vector<double> src(w.n_cols, 7.25);
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  for (const double v : dst) EXPECT_NEAR(v, 7.25, 1e-11);

  // A second build on the same pair must give the same matrix. MOAB's
  // intersection creates edges, tags and vertices in the instance as it goes;
  // if any of that leaked into what the next build sees -- an entity count
  // that shifted, a parent tag left over -- this is where it shows up.
  const auto again = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(again.ok()) << again.error().code << ": "
                          << again.error().message;
  EXPECT_EQ(again.value().row, w.row);
  EXPECT_EQ(again.value().col, w.col);
  EXPECT_EQ(again.value().val, w.val);
}

TEST(RemapSphere, ConservesOnANonMatchingCubedSpherePair) {
  // spherecube.h5m is a 6x24^2 cubed sphere at radius 7996.4553620. The
  // partner is an equiangular 6x16^2 cubed sphere on the SAME sphere,
  // generated here so the test needs no second data file.
  MoabMeshBackend probe;
  ASSERT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  const double radius = sphere_radius(*probe.view(soma::MeshId{0}).value());
  ASSERT_NEAR(radius, 7996.4553620, 1e-6);

  const std::string partner = write_cubed_sphere("cs16", 16, radius);
  MeshPair s = pair_of(SOMA_MOAB_TEST_MESH, partner, 2);
  const soma::MeshView& sv = view_of(s, s.a);
  const soma::MeshView& tv = view_of(s, s.b);
  ASSERT_EQ(sv.entity_count(2), 3456u);
  ASSERT_EQ(tv.entity_count(2), 1536u);
  // 6n^2 + 2 vertices proves the six panels were welded along their seams; an
  // unwelded mesh has 6(n+1)^2 = 1734 and is full of cracks.
  ASSERT_EQ(tv.entity_count(0), 6u * 16u * 16u + 2u);

  const double ms = total_measure(sv, 2);
  const double mt = total_measure(tv, 2);
  std::printf(
      "[sphere 24->16] total measure src=%.10f tgt=%.10f  mismatch=%.6e\n", ms,
      mt, std::fabs(ms - mt) / ms);

  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  std::printf(
      "[sphere 24->16] nnz=%zu row defect=%.6e col defect=%.6e "
      "src cov=%.15f tgt cov=%.15f\n",
      w.nnz(), w.max_row_defect, w.max_col_defect, w.min_src_coverage,
      w.min_tgt_coverage);

  // CONSERVATION holds to machine precision on a real, non-planar,
  // non-matching pair.
  EXPECT_LT(w.max_col_defect, 1e-13);
  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-12);
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-12);

  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto c = sv.centroid(2, i);
    src[i] = 2.0 + c.x / radius + 3.0 * c.y / radius;
  }
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  const double before = integral(sv, 2, src);
  const double after = integral(tv, 2, dst);
  std::printf("[sphere 24->16] integral before=%.15e after=%.15e rel=%.3e\n",
              before, after, std::fabs(after - before) / std::fabs(before));
  EXPECT_NEAR(after, before, 1e-13 * std::fabs(before));

  // CONSTANT PRESERVATION cannot also hold: the two meshes are different
  // polyhedral approximations of one sphere and their total measures differ.
  // The row defect is that mismatch, not a bug -- assert the identity so a
  // future change that "fixes" one of the two silently breaks the other.
  const double mismatch = std::fabs(ms - mt) / mt;
  EXPECT_GT(w.max_row_defect, 0.5 * mismatch);
  EXPECT_LT(w.max_row_defect, 2.0 * mismatch);
  EXPECT_GT(w.max_row_defect, 1e-4) << "the pair should be a real test case";
}

// ===========================================================================
// Task 5.3: high order, and monotonicity
// ===========================================================================
//
// Everything below runs on the cubed-sphere pair, because the high-order
// backend is a spherical remapper. The ORDER of accuracy is not measured here
// -- an error bound on one mesh proves nothing about order, and a second-order
// scheme that is accidentally first-order passes any loose bound. It is
// measured by refinement in test/unit/remap_order_test.cpp.

namespace {

/// The pair used by the tests below: MOAB's 6x24^2 spherecube against an
/// equiangular 6x16^2 partner on the same sphere.
struct SpherePair {
  std::unique_ptr<MeshPair> mesh;
  double radius = 0.0;
};

SpherePair sphere_pair() {
  MoabMeshBackend probe;
  EXPECT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  SpherePair out;
  out.radius = sphere_radius(*probe.view(soma::MeshId{0}).value());
  const std::string partner = write_cubed_sphere("cs16ho", 16, out.radius);
  out.mesh = std::make_unique<MeshPair>(
      pair_of(SOMA_MOAB_TEST_MESH, partner, 2));
  return out;
}

/// A step across a latitude circle. Sampled at the cell centroid, so every
/// source value is exactly 0 or 1 and the bounds question needs no quadrature.
/// A SMOOTH field would let a limiter that does nothing pass, which is why
/// this exists.
std::vector<double> step_field(const soma::MeshView& v, double radius) {
  std::vector<double> f(v.entity_count(2));
  for (std::size_t i = 0; i < f.size(); ++i) {
    f[i] = v.centroid(2, i).z > 0.3 * radius ? 1.0 : 0.0;
  }
  return f;
}

}  // namespace

#ifdef SOMA_HAVE_TEMPEST

TEST(RemapHighOrder, BuildsAtOrdersTwoThreeAndFour) {
  const SpherePair p = sphere_pair();
  for (const int order : {2, 3, 4}) {
    const auto r = build_weights(high_order(*p.mesh, order), p.mesh->graph,
                                 p.mesh->backend);
    ASSERT_TRUE(r.ok()) << "order " << order << ": " << r.error().code << ": "
                        << r.error().message;
    const RemapWeights& w = r.value();
    std::printf(
        "[high order %d] nnz=%zu row defect=%.3e col defect(MeshView)=%.3e "
        "col defect(native)=%.3e min weight=%.6f\n",
        order, w.nnz(), w.max_row_defect, w.max_col_defect,
        w.max_col_defect_native, w.min_weight);
    EXPECT_EQ(w.n_rows, 1536u);
    EXPECT_EQ(w.n_cols, 3456u);
    EXPECT_EQ(w.order, order);
    EXPECT_FALSE(w.monotone);

    // A wider stencil than the first-order map is a necessary consequence of a
    // higher-order reconstruction. Not sufficient -- the order test is the one
    // that decides -- but it is a cheap, load-bearing sanity bound: an
    // order-N request that silently produced the order-1 matrix would have the
    // order-1 nnz, and E-REMAP-DEGRADED would not have fired.
    EXPECT_GT(w.nnz(), 6000u);

    // Negative weights are how the order is bought, and they are why the
    // high-order map is NOT bound preserving. Asserted so that a change which
    // quietly made the weights non-negative -- i.e. dropped to first order --
    // is caught here as well as by the order test.
    EXPECT_LT(w.min_weight, -1e-3) << "order " << order;

    // Row sums are unity to rounding: TempestRemap enforces consistency. This
    // is where the FIRST-order supermesh map on the same pair sits at 1e-3.
    EXPECT_LT(w.max_row_defect, 1e-12);
    // And conservation holds in TempestRemap's own spherical areas...
    EXPECT_LT(w.max_col_defect_native, 1e-12);
  }
}

TEST(RemapHighOrder, ConservationLivesInTheBackendsMeasureNotMeshViews) {
  // The one number in this task most likely to be misread. The map IS
  // conservative -- in TempestRemap's exact spherical-polygon areas. Measured
  // against MeshView::measure, which integrates the bilinear patch through the
  // cell corners and is therefore short of the spherical area by O(h^2), the
  // same matrix shows a ~1e-3 defect. Both numbers are reported, and the
  // identity between them is pinned here so that neither can be "fixed" into
  // agreement without this test noticing.
  const SpherePair p = sphere_pair();
  const soma::MeshView& sv = view_of(*p.mesh, p.mesh->a);
  const soma::MeshView& tv = view_of(*p.mesh, p.mesh->b);
  const double ms = total_measure(sv, 2);
  const double mt = total_measure(tv, 2);
  const double mismatch = std::fabs(ms - mt) / mt;

  const auto r =
      build_weights(high_order(*p.mesh, 2), p.mesh->graph, p.mesh->backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  std::printf(
      "[measure conventions] measure mismatch=%.6e  col defect(MeshView)=%.6e "
      " col defect(native)=%.6e  row defect=%.3e\n",
      mismatch, w.max_col_defect, w.max_col_defect_native, w.max_row_defect);

  EXPECT_GT(mismatch, 1e-4) << "the pair should be a real test case";
  EXPECT_LT(w.max_col_defect_native, 1e-12);
  // The MeshView-measure defect is the measure mismatch, not noise.
  EXPECT_GT(w.max_col_defect, 0.2 * mismatch);
  EXPECT_LT(w.max_col_defect, 5.0 * mismatch);
}

TEST(RemapMonotone, HighOrderOvershootsAStepAndTheLimitedMapDoesNot) {
  // THE monotonicity test. A step field, because a smooth one is passed by a
  // limiter that does nothing; and the unlimited run is shown VIOLATING the
  // bound on the same field, because a bound test that cannot fail proves
  // nothing.
  const SpherePair p = sphere_pair();
  const soma::MeshView& sv = view_of(*p.mesh, p.mesh->a);
  const std::vector<double> src = step_field(sv, p.radius);
  const double smin = *std::min_element(src.begin(), src.end());
  const double smax = *std::max_element(src.begin(), src.end());
  ASSERT_DOUBLE_EQ(smin, 0.0);
  ASSERT_DOUBLE_EQ(smax, 1.0);

  // --- the unlimited high-order map: overshoots -----------------------------
  double unlimited_violation = 0.0;
  for (const int order : {2, 3, 4}) {
    const auto r = build_weights(high_order(*p.mesh, order), p.mesh->graph,
                                 p.mesh->backend);
    ASSERT_TRUE(r.ok()) << r.error().message;
    const RemapWeights& w = r.value();
    std::vector<double> dst(w.n_rows, 0.0);
    ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
    const double lo = *std::min_element(dst.begin(), dst.end());
    const double hi = *std::max_element(dst.begin(), dst.end());
    std::size_t bad = 0;
    const double local = worst_local_violation(w, src, dst, &bad);
    std::printf(
        "[monotone] order %d UNLIMITED: target range [%.6f, %.6f] "
        "global violation=%.3e  local violation=%.3e on %zu of %zu cells\n",
        order, lo, hi, std::max(smin - lo, hi - smax), local, bad, w.n_rows);
    // The bound is genuinely broken. Both directions overshoot -- asserted at
    // machine scale, because the sign of the excursion is the qualitative
    // fact -- and the WORST excursion is far beyond any tolerance that could
    // be mistaken for rounding. (The two directions are not symmetric: at
    // order 2 the measured undershoot is 6.9e-2 and the overshoot 3.0e-4, so
    // demanding 1e-3 of each separately would be pinning noise.)
    EXPECT_LT(lo, smin - 1e-12) << "order " << order;
    EXPECT_GT(hi, smax + 1e-12) << "order " << order;
    EXPECT_GT(std::max(smin - lo, hi - smax), 1e-2) << "order " << order;
    EXPECT_GT(bad, 20u) << "order " << order;
    unlimited_violation =
        std::max(unlimited_violation, std::max(smin - lo, hi - smax));
  }
  EXPECT_GT(unlimited_violation, 1e-2);

  // --- the limited map: does not -------------------------------------------
  RemapPlan limited{.src = p.mesh->a,
                    .tgt = p.mesh->b,
                    .src_order = 1,
                    .flags = RemapFlags::Conservative | RemapFlags::Monotone};
  limited.method = "tempest";
  const auto r =
      build_weights(limited, p.mesh->graph, p.mesh->backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  EXPECT_TRUE(w.monotone);
  EXPECT_GE(w.min_weight, 0.0);
  EXPECT_LT(w.max_row_defect, 1e-12);

  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  const double lo = *std::min_element(dst.begin(), dst.end());
  const double hi = *std::max_element(dst.begin(), dst.end());
  std::size_t bad = 0;
  const double local = worst_local_violation(w, src, dst, &bad);
  std::printf(
      "[monotone] order 1 LIMITED  : target range [%.6f, %.6f] "
      "global violation=%.3e  local violation=%.3e on %zu of %zu cells\n",
      lo, hi, std::max(smin - lo, hi - smax), local, bad, w.n_rows);
  EXPECT_GE(lo, smin - 1e-13);
  EXPECT_LE(hi, smax + 1e-13);
  EXPECT_EQ(bad, 0u);
  EXPECT_LT(local, 1e-12);
}

TEST(RemapHighOrder, SilentDegradationToFirstOrderIsCaughtAndRefused) {
  // TempestRemap's failure mode that costs the most: when its least-squares fit
  // is too ill-conditioned on a cell it builds a FIRST-order row there, prints
  // "WARNING: Poor conditioning in fit matrix (...); dropping to 1st order" and
  // returns SUCCESS. A caller who asked for order 4 gets a map that is order 4
  // in some places and order 1 in others, with no error and no way to tell.
  //
  // A 32 x 4 latitude-longitude source against a 6x6^2 cubed sphere reaches it:
  // at order 4 the 64 polar triangles degrade. Orders 1 to 3 on the same pair
  // do not, which is what makes this a test of the DETECTION and not of the
  // mesh being generally unusable.
  MoabMeshBackend probe;
  ASSERT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  const double radius = sphere_radius(*probe.view(soma::MeshId{0}).value());

  const std::string rll = write_lat_lon("rll32x4", 32, 4, radius);
  const std::string cs = write_cubed_sphere("cs6deg", 6, radius);
  MeshPair s = pair_of(rll, cs, 2);
  ASSERT_EQ(view_of(s, s.a).entity_count(2), 128u);
  ASSERT_EQ(view_of(s, s.b).entity_count(2), 216u);

  for (const int order : {1, 2, 3}) {
    RemapPlan plan = high_order(s, order);
    plan.method = "tempest";  // order 1 would otherwise use the supermesh
    const auto r = build_weights(plan, s.graph, s.backend);
    EXPECT_TRUE(r.ok()) << "order " << order << " should build: "
                        << r.error().code << ": " << r.error().message;
    if (r.ok()) EXPECT_EQ(r.value().order, order);
  }

  // ASIDE, worth recording: the FIRST-ORDER supermesh builder cannot do this
  // pair at all. MOAB's Intx2MeshOnSphere fails with "can't find nodes" on the
  // degenerate polar triangles. That is a Task 5.2 limitation, not a Task 5.3
  // one, and it fails loudly (E-REMAP-INTX) rather than quietly, so it is
  // pinned here rather than fixed -- if a future MOAB starts coping, this
  // assertion is the notification.
  const auto supermesh = build_weights(high_order(s, 1), s.graph, s.backend);
  if (!supermesh.ok()) {
    EXPECT_EQ(supermesh.error().code, "E-REMAP-INTX");
  }

  const auto r = build_weights(high_order(s, 4), s.graph, s.backend);
  ASSERT_FALSE(r.ok()) << "order 4 on this pair degrades inside TempestRemap "
                          "and must not come back as a successful order-4 map";
  EXPECT_EQ(r.error().code, "E-REMAP-DEGRADED");
  // TempestRemap's own words are relayed, not paraphrased.
  EXPECT_NE(r.error().message.find("Poor conditioning"), std::string::npos)
      << r.error().message;
  std::printf("[degradation] %s\n", r.error().message.c_str());
}

TEST(RemapHighOrder, LatLonPolarTrianglesDefeatFirstOrderButHighOrderSucceeds) {
  // R9, closing the "strictly weaker" claim from an assertion into a pinned
  // regression test. The neighbouring test above already builds this exact
  // pair (a 32x4 latitude-longitude source against a 6x6^2 cubed-sphere
  // target) and, as an ASIDE, checks -- only IF the first-order supermesh
  // build happens to fail -- that it fails with E-REMAP-INTX. That is a
  // conditional assertion: if MOAB's Intx2MeshOnSphere ever silently started
  // succeeding on the degenerate polar triangles, that test would pass
  // without noticing. This test makes both halves of the claim
  // UNCONDITIONAL: the first-order path MUST fail on this pair, and the
  // high-order path MUST succeed on the very same pair.
  MoabMeshBackend probe;
  ASSERT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  const double radius = sphere_radius(*probe.view(soma::MeshId{0}).value());

  const std::string rll = write_lat_lon("rll32x4b", 32, 4, radius);
  const std::string cs = write_cubed_sphere("cs6degb", 6, radius);
  MeshPair s = pair_of(rll, cs, 2);
  ASSERT_EQ(view_of(s, s.a).entity_count(2), 128u);
  ASSERT_EQ(view_of(s, s.b).entity_count(2), 216u);

  // --- the first-order supermesh path: MUST fail ----------------------------
  const auto first_order = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_FALSE(first_order.ok())
      << "the first-order supermesh builder was expected to fail on "
         "degenerate polar triangles, per Task 5.2's documented limitation; "
         "if this now succeeds, moab::Intx2MeshOnSphere has improved and the "
         "'strictly weaker' claim in docs/HANDOFF.md's R9 needs revisiting";
  EXPECT_EQ(first_order.error().code, "E-REMAP-INTX");
  std::printf("[R9 poles] first-order FAILS as documented: %s: %s\n",
             first_order.error().code.c_str(),
             first_order.error().message.c_str());

  // --- the high-order path, orders 1-3: MUST succeed on the SAME pair ------
  for (const int order : {1, 2, 3}) {
    RemapPlan plan = high_order(s, order);
    plan.method = "tempest";  // order 1 would otherwise route to the (failing)
                             // supermesh builder
    const auto hi = build_weights(plan, s.graph, s.backend);
    ASSERT_TRUE(hi.ok()) << "order " << order << " should build on the pair "
                        << "the first-order path cannot handle: "
                        << hi.error().code << ": " << hi.error().message;
    EXPECT_EQ(hi.value().order, order);
    std::printf(
        "[R9 poles] high order %d SUCCEEDS where first-order fails: "
        "nnz=%zu row defect=%.3e col defect(native)=%.3e\n",
        order, hi.value().nnz(), hi.value().max_row_defect,
        hi.value().max_col_defect_native);
  }
}

TEST(RemapHighOrder,
    ProductionScaleFvAndCubedSphereValidateCoverageTolAndOrderFourGeneralizes) {
  // R9: `kMaxOrder = 4` (src/remap/tempest_weights.cpp) and
  // `kHighOrderCoverageTol = 1e-6` (src/remap/intx_weights.cpp) were both
  // measured on small, coarse meshes -- a 6x8^2 against 6x6^2 cubed sphere for
  // the order cap, a 32x4 lat-lon against a 6x6^2 cubed sphere for the
  // coverage tolerance. Neither had been checked against anything close to
  // production scale. This test builds a pair modeled on REAL climate-model
  // grids that were never part of this suite's fixtures:
  //
  //   * a 144 x 96 latitude-longitude grid -- the class of grid CESM/E3SM
  //     call "f19" (roughly 1.9 x 2.5 degree finite-volume), 13,824 cells,
  //     with the same degenerate polar-triangle rows as the small 32x4
  //     fixture but at ~72x the resolution;
  //   * a 6x30^2 equiangular cubed sphere -- the "ne30" resolution E3SM/CESM
  //     actually run at production scale, 5,400 cells.
  //
  // This is exactly the FV<->cubed-sphere regridding pair a real atmosphere
  // model uses TempestRemap for. Neither resolution appears anywhere else in
  // this test suite (16, 24 and 32x4/6x6^2 are all taken).
  //
  // MEASURED HERE, AND THE OUTCOME WAS NOT WHAT WAS FIRST ASSUMED. Orders 1-3
  // build cleanly with the source-coverage defect at ~1e-15 -- three orders
  // tighter than the 1.7e-8 measured on the small 32x4 fixture, and nine
  // tighter than `kHighOrderCoverageTol` itself, so the tolerance is amply
  // validated at production scale. Order 4, however, hits
  // `E-REMAP-DEGRADED`: TempestRemap's own conditioning check trips on this
  // real, fine 144x96 lat-lon source, exactly as it does on the small 32x4
  // one -- so the order-4 degradation on a lat-lon (pole-triangle) source is
  // NOT an artifact of coarseness, it is a structural property of the
  // gnomonic-coordinate fit near a lat-lon mesh's polar rows, independent of
  // resolution. `RemapHighOrder.BuildsAtOrdersTwoThreeAndFour` (above, in
  // this same file) proves the other half: a CUBED-SPHERE-only pair (no
  // lat-lon source) builds cleanly at order 4 with no warning at all. So
  // `kMaxOrder = 4` stands as a correct upper bound on what TempestRemap can
  // ever be ASKED for, and its safety continues to rest on the runtime
  // degradation detection working correctly (which this test additionally
  // confirms, at production scale) rather than on order 4 always
  // succeeding. See docs/HANDOFF.md R35 for the full writeup of this finding
  // and its practical consequence for callers with a lat-lon source.
  MoabMeshBackend probe;
  ASSERT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  // A real Earth radius (meters), deliberately NOT the MOAB fixture's
  // arbitrary radius -- this pair does not depend on SOMA_MOAB_TEST_MESH's
  // geometry at all, only on `probe` loading successfully to prove MOAB
  // itself is usable in this configuration.
  const double radius = 6371220.0;

  const std::string rll = write_lat_lon("f19_144x96", 144, 96, radius);
  const std::string cs = write_cubed_sphere("ne30", 30, radius);
  MeshPair s = pair_of(rll, cs, 2);
  ASSERT_EQ(view_of(s, s.a).entity_count(2), 144u * 96u);
  ASSERT_EQ(view_of(s, s.b).entity_count(2), 6u * 30u * 30u);

  // Orders 1-3 build cleanly; `kHighOrderCoverageTol` is validated with real
  // margin at production scale (measured defect ~1e-15, tolerance 1e-6).
  for (const int order : {1, 2, 3}) {
    RemapPlan plan = high_order(s, order);
    plan.method = "tempest";
    const auto r = build_weights(plan, s.graph, s.backend);
    ASSERT_TRUE(r.ok()) << "order " << order << " on the production-scale "
                        << "f19/ne30 pair: " << r.error().code << ": "
                        << r.error().message;
    const RemapWeights& w = r.value();
    EXPECT_EQ(w.order, order);
    std::printf(
        "[R9 production f19/ne30] order %d: nnz=%zu row defect=%.3e "
        "col defect(native)=%.3e src cov=%.15f tgt cov=%.15f min "
        "weight=%.6f\n",
        order, w.nnz(), w.max_row_defect, w.max_col_defect_native,
        w.min_src_coverage, w.min_tgt_coverage, w.min_weight);

    // kHighOrderCoverageTol validation: both coverage ratios must sit well
    // within the tolerance the library enforces, with real margin, not by
    // construction -- `build_weights` would have refused already
    // (E-REMAP-INTX / E-REMAP-COVERAGE) if either exceeded 1e-6, so a bare
    // `.ok()` already proves the tolerance held; these EXPECTs additionally
    // report and pin HOW MUCH margin there was, which is the evidence trail
    // R9 asks for.
    EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-6);
    EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-6);
  }

  // Order 4: MEASURED to degrade on this real, fine lat-lon source, caught
  // (loudly and correctly) by the existing detection -- the finding this
  // test exists to pin. A regression here (order 4 suddenly "succeeding"
  // silently, or failing with a different code) is exactly the failure mode
  // R9 is worried about, so it is asserted precisely rather than loosely.
  RemapPlan order4 = high_order(s, 4);
  order4.method = "tempest";
  const auto degraded = build_weights(order4, s.graph, s.backend);
  ASSERT_FALSE(degraded.ok())
      << "order 4 was expected to degrade on this fine lat-lon source, "
         "exactly as it does on the small 32x4 fixture -- if this now "
         "succeeds, TempestRemap's conditioning behavior has changed and "
         "docs/HANDOFF.md R35 needs revisiting";
  EXPECT_EQ(degraded.error().code, "E-REMAP-DEGRADED");
  // SOMA's OWN boilerplate sentence is always present regardless of what the
  // relayed tail happens to contain -- this is the part of the message a
  // caller can rely on.
  EXPECT_NE(degraded.error().message.find("silently built a lower-order map"),
           std::string::npos)
      << degraded.error().message;
  // TempestRemap's OWN "Poor conditioning" line is NOT reliably present at
  // this scale -- a second, distinct finding recorded in docs/HANDOFF.md R35.
  // `tail(out.log, 6)` (src/remap/tempest_weights.cpp) relays only the last 6
  // lines of the captured Announce stream, and on this 13,824-cell mesh those
  // 6 lines are entirely per-1000-element progress markers
  // ("..Element N/13824") printed AFTER the conditioning warning, which pushes
  // it out of the window. On the small 128-cell fixture the same warning
  // lands inside the last 6 lines by coincidence, not by any scale-invariant
  // guarantee. Recorded as fact, not asserted either way, because pinning a
  // brittle string-search result on log-formatting-dependent line counts
  // would be exactly the kind of goldenmaster failure Task 5.3's own tests
  // warn against elsewhere in this file.
  const bool tail_contains_warning =
      degraded.error().message.find("Poor conditioning") != std::string::npos;
  std::printf(
      "[R9 production f19/ne30] order 4 degrades (expected), E-REMAP-DEGRADED "
      "fires correctly; relayed tail contains \"Poor conditioning\": %s\n",
      tail_contains_warning ? "yes" : "NO (drowned out by progress lines -- "
                                      "see docs/HANDOFF.md R35)");

  // Order 5 must still be refused outright -- the cap is enforced, not just
  // usually not hit. Unrelated to the degradation above: this is refused
  // before TempestRemap is even called, purely on `request.order > kMaxOrder`.
  RemapPlan too_high = high_order(s, 5);
  too_high.method = "tempest";
  const auto refused = build_weights(too_high, s.graph, s.backend);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.error().code, "E-REMAP-UNSUPPORTED");
}

TEST(RemapHighOrder, ConcurrentBuildsAreSerializedNotCorrupted) {
  // R10. `AnnounceCapture` (src/remap/tempest_weights.cpp) redirects
  // TempestRemap's process-wide `Announce` `FILE*` for the duration of one
  // build. Two threads building concurrently must not corrupt each other's
  // captured diagnostics -- and because printing is TempestRemap's ONLY
  // failure channel (see this file's own header comment, and
  // tempest_weights.cpp's), a corrupted capture can turn a REAL build
  // failure into a false success, which is worse than a crash.
  //
  // Two threads, two INDEPENDENT MeshPairs (each with its own moab::Core) so
  // this test is about the Announce race specifically, not about
  // moab::Core's own thread safety, which is a different question this task
  // does not address. One thread repeatedly runs a build KNOWN to degrade
  // (order 4 on the 32x4 lat-lon / 6x6^2 cubed-sphere pair -- pinned
  // single-threaded above by SilentDegradationToFirstOrderIsCaughtAndRefused)
  // and must reliably observe E-REMAP-DEGRADED with "Poor conditioning" in
  // the relayed text; the other repeatedly runs a build KNOWN to succeed
  // cleanly (order 1, forced through the tempest path, same geometry) and
  // must reliably observe success. If the two threads' captures cross wires,
  // the most likely symptoms are: the degrading thread's failure silently
  // disappearing (its capture stolen or emptied so the "Poor conditioning"
  // check finds nothing and the build is wrongly reported as clean order-4),
  // or the clean thread spuriously failing (reading text left behind by the
  // other thread's capture). Both are checked, every iteration, in a loop --
  // a timing race does not reproduce on every run.
  MoabMeshBackend probe;
  ASSERT_TRUE(
      probe.load(MeshDecl{.name = "p", .source = SOMA_MOAB_TEST_MESH}).ok());
  const double radius = sphere_radius(*probe.view(soma::MeshId{0}).value());

  constexpr int kIterations = 30;
  int clean_failures = 0;
  int degraded_failures = 0;
  for (int iter = 0; iter < kIterations; ++iter) {
    const std::string tagA = "conc_deg" + std::to_string(iter);
    const std::string tagB = "conc_ok" + std::to_string(iter);
    MeshPair pairDeg =
        pair_of(write_lat_lon("rll_" + tagA, 32, 4, radius),
               write_cubed_sphere("cs_" + tagA, 6, radius), 2);
    MeshPair pairOk =
        pair_of(write_lat_lon("rll_" + tagB, 32, 4, radius),
               write_cubed_sphere("cs_" + tagB, 6, radius), 2);

    soma::Result<RemapWeights> resultDeg = soma::Error{"pending", "not yet run"};
    soma::Result<RemapWeights> resultOk = soma::Error{"pending", "not yet run"};

    std::thread degrading([&] {
      RemapPlan plan = high_order(pairDeg, 4);
      plan.method = "tempest";
      resultDeg = build_weights(plan, pairDeg.graph, pairDeg.backend);
    });
    std::thread clean([&] {
      RemapPlan plan = high_order(pairOk, 1);
      plan.method = "tempest";
      resultOk = build_weights(plan, pairOk.graph, pairOk.backend);
    });
    degrading.join();
    clean.join();

    // (a) no crash/hang: reaching here at all, for every iteration, proves
    // it; a CTest TIMEOUT is the backstop if it ever wedges instead.
    // (b) each thread's captured diagnostic is complete and attributed to
    // its OWN build, not the other's:
    if (resultDeg.ok()) {
      ++clean_failures;  // the degrading build was corrupted into "success"
    } else {
      EXPECT_EQ(resultDeg.error().code, "E-REMAP-DEGRADED")
          << "iteration " << iter << ": " << resultDeg.error().message;
      EXPECT_NE(resultDeg.error().message.find("Poor conditioning"),
               std::string::npos)
          << "iteration " << iter
          << ": degrading thread's captured text is missing its own "
             "diagnostic -- " << resultDeg.error().message;
    }
    if (!resultOk.ok()) {
      ++degraded_failures;  // the clean build was corrupted into "failure"
      ADD_FAILURE() << "iteration " << iter
                    << ": the order-1 build, which builds cleanly "
                       "single-threaded, failed under concurrency: "
                    << resultOk.error().code << ": "
                    << resultOk.error().message;
    } else {
      EXPECT_EQ(resultOk.value().order, 1) << "iteration " << iter;
    }
  }
  std::printf(
      "[R10 concurrency] %d iterations: %d degrading-build corruptions, %d "
      "clean-build corruptions\n",
      kIterations, clean_failures, degraded_failures);
  EXPECT_EQ(clean_failures, 0);
  EXPECT_EQ(degraded_failures, 0);
}

#else  // SOMA_HAVE_TEMPEST

TEST(RemapHighOrder, RefusedOutrightWhenTheBackendIsAbsent) {
  // The other half of the configuration matrix: MOAB present, TempestRemap not.
  // The high-order request must fail with a code that says WHICH dependency is
  // missing, and the first-order supermesh map must be unaffected.
  const SpherePair p = sphere_pair();
  const auto r =
      build_weights(high_order(*p.mesh, 2), p.mesh->graph, p.mesh->backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-NOTEMPEST");

  RemapPlan mono{.src = p.mesh->a,
                 .tgt = p.mesh->b,
                 .src_order = 1,
                 .flags = RemapFlags::Conservative | RemapFlags::Monotone};
  mono.method = "tempest";
  EXPECT_EQ(build_weights(mono, p.mesh->graph, p.mesh->backend).error().code,
            "E-REMAP-NOTEMPEST");

  const auto first_order =
      build_weights(conservative(*p.mesh), p.mesh->graph, p.mesh->backend);
  EXPECT_TRUE(first_order.ok()) << first_order.error().message;
}

#endif  // SOMA_HAVE_TEMPEST

TEST(RemapMonotone, TheSupermeshMapBetweenCubedSpheresIsNotCertified) {
  // Same pair, same first order, but the supermesh builder -- whose rows sum to
  // 1 - 1e-3 because the two meshes do not report the same total
  // MeshView::measure. Its weights are all positive, so it looks monotone if
  // you only check the sign; it is not, and the certification says so instead
  // of setting the flag.
  const SpherePair p = sphere_pair();
  RemapPlan plan{.src = p.mesh->a,
                 .tgt = p.mesh->b,
                 .flags = RemapFlags::Conservative | RemapFlags::Monotone};
  plan.method = "intx";
  const auto r = build_weights(plan, p.mesh->graph, p.mesh->backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-MONOTONE");
  std::printf("[monotone] supermesh on the sphere pair: %s\n",
              r.error().message.c_str());

  // Without the flag the same plan succeeds and reports monotone == false.
  RemapPlan unflagged = plan;
  unflagged.flags = RemapFlags::Conservative;
  const auto ok = build_weights(unflagged, p.mesh->graph, p.mesh->backend);
  ASSERT_TRUE(ok.ok()) << ok.error().message;
  EXPECT_FALSE(ok.value().monotone);
  EXPECT_GE(ok.value().min_weight, 0.0);
  EXPECT_GT(ok.value().max_row_defect, 1e-4);
}

#endif  // SOMA_MOAB_TEST_MESH

TEST(RemapMonotone, AxisAlignedFirstOrderMapIsBoundPreserving) {
  // No sphere, no TempestRemap: the first-order supermesh map between two
  // axis-aligned box meshes IS bound preserving, because there the two meshes
  // do report the same total measure. Task 5.2 refused RemapFlags::Monotone
  // here; refusing it was inaccurate, so 5.3 certifies it instead.
  MeshPair s = non_matching();
  RemapPlan plan{.src = s.a,
                 .tgt = s.b,
                 .flags = RemapFlags::Conservative | RemapFlags::Monotone};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const RemapWeights& w = r.value();
  EXPECT_TRUE(w.monotone);
  EXPECT_EQ(w.order, 1);
  EXPECT_GE(w.min_weight, 0.0);
  EXPECT_LT(w.max_row_defect, 1e-12);

  // And it really does preserve bounds on a field that would overshoot: a
  // checkerboard of 0 and 1 on the 4x4x4 source.
  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<double>((i / 1 + i / 4 + i / 16) % 2);
  }
  std::vector<double> dst(w.n_rows, 0.0);
  ASSERT_TRUE(apply_weights(w, src.data(), dst.data(), 1).ok());
  std::size_t bad = 0;
  EXPECT_LT(worst_local_violation(w, src, dst, &bad), 1e-12);
  EXPECT_EQ(bad, 0u);
}

// ===========================================================================
// Refusals
// ===========================================================================

TEST(RemapRefuses, HighOrderOnAVolumeMesh) {
  // Task 5.2 refused this pair for HighOrder, for order > 0 and for Monotone
  // alike, with one message about TempestRemap. Task 5.3 splits it: the two
  // high-order requests are still refused, but now because THE GEOMETRY is not
  // one the high-order backend handles (these are hexahedra), and the Monotone
  // request is now honoured -- see RemapMonotone.AxisAlignedFirstOrderMapIs
  // BoundPreserving below.
  MeshPair s = non_matching();
  const RemapPlan flagged{.src = s.a,
                          .tgt = s.b,
                          .flags = RemapFlags::Conservative |
                                   RemapFlags::HighOrder};
  const auto r = build_weights(flagged, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
  // The message must name the axis that is missing. "High order needs
  // TempestRemap" would be a lie now that TempestRemap is linked in.
  EXPECT_NE(r.error().message.find("dimension 3"), std::string::npos)
      << r.error().message;

  const RemapPlan high{.src = s.a, .tgt = s.b, .src_order = 2};
  const auto r2 = build_weights(high, s.graph, s.backend);
  ASSERT_FALSE(r2.ok());
  EXPECT_EQ(r2.error().code, "E-REMAP-UNSUPPORTED");

  // Naming the backend explicitly does not change the answer.
  RemapPlan named = conservative(s);
  named.method = "tempest";
  EXPECT_EQ(build_weights(named, s.graph, s.backend).error().code,
            "E-REMAP-UNSUPPORTED");
}

TEST(RemapRefuses, MonotoneTogetherWithAnOrderAboveOne) {
  // The Godunov-type barrier, stated as a refusal rather than resolved by
  // quietly returning the first-order map -- which is exactly what
  // TempestRemap itself does when fMonotone is set.
  MeshPair s = non_matching();
  for (const int order : {2, 3, 4}) {
    RemapPlan plan{.src = s.a,
                   .tgt = s.b,
                   .src_order = order,
                   .flags = RemapFlags::Conservative | RemapFlags::Monotone};
    const auto r = build_weights(plan, s.graph, s.backend);
    ASSERT_FALSE(r.ok()) << "order " << order;
    EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
    EXPECT_NE(r.error().message.find("Bound preservation"), std::string::npos)
        << r.error().message;
  }
}

TEST(RemapRefuses, OrdersTheBackendWillNotDeliver) {
  MeshPair s = non_matching();
  // Above the cap, refused before the geometry is even looked at.
  for (const int order : {5, 6, 12}) {
    const RemapPlan plan{.src = s.a, .tgt = s.b, .src_order = order};
    const auto r = build_weights(plan, s.graph, s.backend);
    ASSERT_FALSE(r.ok()) << "order " << order;
    EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
  }
  // A negative order is not "unspecified".
  const RemapPlan neg{.src = s.a, .tgt = s.b, .src_order = -1};
  EXPECT_EQ(build_weights(neg, s.graph, s.backend).error().code,
            "E-REMAP-UNSUPPORTED");
}

TEST(RemapRefuses, ContradictoryOrderRequests) {
  MeshPair s = non_matching();
  // Two different orders: one of them would have to be ignored.
  const RemapPlan split{
      .src = s.a, .tgt = s.b, .src_order = 2, .tgt_order = 3};
  const auto r = build_weights(split, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
  EXPECT_NE(r.error().message.find("target order"), std::string::npos);

  // HighOrder with an explicit order 1 says two opposite things.
  const RemapPlan one{.src = s.a,
                      .tgt = s.b,
                      .src_order = 1,
                      .flags = RemapFlags::HighOrder};
  EXPECT_EQ(build_weights(one, s.graph, s.backend).error().code,
            "E-REMAP-UNSUPPORTED");

  // The first-order builder, named explicitly, will not accept an order.
  RemapPlan intx{.src = s.a, .tgt = s.b, .src_order = 2};
  intx.method = "intx";
  EXPECT_EQ(build_weights(intx, s.graph, s.backend).error().code,
            "E-REMAP-UNSUPPORTED");
}

TEST(RemapRefuses, UnknownMethodNamesTheKnownOnes) {
  MeshPair s = non_matching();
  RemapPlan plan = conservative(s);
  plan.method = "bilinear";
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
  EXPECT_NE(r.error().message.find("'tempest'"), std::string::npos);
  EXPECT_NE(r.error().message.find("'intx'"), std::string::npos);
}

TEST(RemapRefuses, NonFiniteVolumeDiscretizations) {
  MeshPair s = non_matching();
  RemapPlan plan = conservative(s);
  plan.tgt_disc = soma::Discretization::ContinuousGLL;
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
}

TEST(RemapRefuses, SubsetRegions) {
  MeshPair s = non_matching();
  const auto rid = s.graph.add_region(Region{
      .name = "skin", .mesh = s.ma, .entity_set = "bottom", .dimension = 2});
  const RemapPlan plan{.src = rid, .tgt = s.b};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-REGION");
}

TEST(RemapRefuses, MismatchedDimensions) {
  MeshPair s = non_matching();
  const auto rid = s.graph.add_region(
      Region{.name = "faces", .mesh = s.ma, .dimension = 2});
  const RemapPlan plan{.src = rid, .tgt = s.b};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-DIM");
}

TEST(RemapRefuses, InvalidRegionIds) {
  MeshPair s = non_matching();
  const RemapPlan plan{.src = soma::RegionId{99}, .tgt = s.b};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-REGION");
}

TEST(RemapRefuses, ABackendThatIsNotMoab) {
  CouplingGraph g;
  const auto m =
      g.add_mesh(MeshDecl{.name = "a", .source = "generated:box:2x2x2"});
  const auto rid = g.add_region(Region{.name = "r", .mesh = m, .dimension = 3});
  soma::SerialMeshBackend serial;
  ASSERT_TRUE(serial.load(g.mesh(m)).ok());
  const RemapPlan plan{.src = rid, .tgt = rid};
  const auto r = build_weights(plan, g, serial);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-BACKEND");
}

TEST(RemapRefuses, SurfacesThatAreNeitherSphereNorPlane) {
  // A tilted plane: planar, but not z = const, so the x-y projection MOAB's
  // in-plane intersection works in is not the surface. Refused rather than
  // quietly remapped in a projection that is not the domain.
  VtkMesh m;
  m.vtk_type = 9;
  for (int j = 0; j <= 2; ++j) {
    for (int i = 0; i <= 2; ++i) {
      const double x = i * 0.5, y = j * 0.5;
      m.points.push_back({x, y, 0.75 * x});
    }
  }
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const int id = j * 3 + i;
      m.cells.push_back({id, id + 1, id + 4, id + 3});
    }
  }
  const std::string tilted = write_vtk("tilted", m);
  MeshPair s = pair_of(tilted, tilted, 2);
  const RemapPlan plan{.src = s.a, .tgt = s.b};
  const auto r = build_weights(plan, s.graph, s.backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED");
}

// ===========================================================================
// The cache
// ===========================================================================

TEST(RemapCacheTest, ReusesWeightsForIdenticalPlans) {
  MeshPair s = non_matching();
  const RemapPlan plan = conservative(s);
  RemapCache cache;
  const auto first = cache.get_or_build(plan, s.graph, s.backend);
  ASSERT_TRUE(first.ok()) << first.error().message;
  const auto second = cache.get_or_build(plan, s.graph, s.backend);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.value(), second.value());  // same object, not a rebuild
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_EQ(cache.misses(), 1u);
  EXPECT_EQ(cache.hits(), 1u);
}

TEST(RemapCacheTest, DistinguishesDifferentPlans) {
  MeshPair s = non_matching();
  RemapCache cache;
  ASSERT_TRUE(cache.get_or_build(conservative(s), s.graph, s.backend).ok());
  ASSERT_TRUE(cache
                  .get_or_build(RemapPlan{.src = s.b,
                                          .tgt = s.a,
                                          .flags = RemapFlags::Conservative},
                                s.graph, s.backend)
                  .ok());
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_EQ(cache.misses(), 2u);
  EXPECT_EQ(cache.hits(), 0u);
}

TEST(RemapCacheTest, PointersSurviveRehashing) {
  MeshPair s = non_matching("generated:box:2x2x2", "generated:box:3x3x3");
  RemapCache cache;
  const auto first = cache.get_or_build(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(first.ok());
  const RemapWeights* p = first.value();
  const std::vector<double> snapshot = p->val;
  // Enough distinct plans to force the unordered_map to rehash several times.
  for (int k = 1; k < 64; ++k) {
    RemapPlan plan = conservative(s);
    plan.method = "conservative";
    plan.temporal_degree = k;
    ASSERT_TRUE(cache.get_or_build(plan, s.graph, s.backend).ok());
  }
  EXPECT_EQ(p->val, snapshot);
  EXPECT_EQ(cache.get_or_build(conservative(s), s.graph, s.backend).value(), p);
}

TEST(RemapCacheTest, FailedBuildsAreNotCached) {
  MeshPair s = non_matching();
  RemapCache cache;
  const RemapPlan bad{.src = soma::RegionId{99}, .tgt = s.b};
  EXPECT_FALSE(cache.get_or_build(bad, s.graph, s.backend).ok());
  EXPECT_FALSE(cache.get_or_build(bad, s.graph, s.backend).ok());
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_EQ(cache.misses(), 2u);
  EXPECT_EQ(cache.hits(), 0u);
}

TEST(RemapPlanKey, CoversEveryField) {
  const RemapPlan base{.src = soma::RegionId{1}, .tgt = soma::RegionId{2}};
  std::vector<RemapPlan> variants;
  variants.push_back(base);
  {
    RemapPlan p = base; p.src = soma::RegionId{3}; variants.push_back(p);
  }
  {
    RemapPlan p = base; p.tgt = soma::RegionId{3}; variants.push_back(p);
  }
  {
    RemapPlan p = base; p.src_disc = soma::Discretization::PointCloud;
    variants.push_back(p);
  }
  {
    RemapPlan p = base; p.tgt_disc = soma::Discretization::PointCloud;
    variants.push_back(p);
  }
  { RemapPlan p = base; p.src_order = 1; variants.push_back(p); }
  { RemapPlan p = base; p.tgt_order = 1; variants.push_back(p); }
  {
    RemapPlan p = base; p.flags = RemapFlags::Conservative;
    variants.push_back(p);
  }
  { RemapPlan p = base; p.method = "x"; variants.push_back(p); }
  {
    RemapPlan p = base; p.temporal = soma::TemporalBasis::DG;
    variants.push_back(p);
  }
  { RemapPlan p = base; p.temporal_degree = 2; variants.push_back(p); }

  for (std::size_t i = 0; i < variants.size(); ++i) {
    for (std::size_t j = i + 1; j < variants.size(); ++j) {
      EXPECT_NE(variants[i].key(), variants[j].key()) << i << " vs " << j;
      EXPECT_FALSE(variants[i] == variants[j]) << i << " vs " << j;
    }
  }
  // A free-form method must not be able to forge another plan's key.
  RemapPlan forged = base;
  forged.method = "x/None0";
  RemapPlan honest = base;
  honest.method = "x";
  honest.temporal = soma::TemporalBasis::None;
  EXPECT_NE(forged.key(), honest.key());
}

TEST(RemapWeightsView, MatchesTheSpacetimeAdapter) {
  MeshPair s = non_matching();
  const auto r = build_weights(conservative(s), s.graph, s.backend);
  ASSERT_TRUE(r.ok());
  const soma::SpatialWeightsView v = spatial_view(r.value());
  EXPECT_EQ(v.row, r.value().row.data());
  EXPECT_EQ(v.col, r.value().col.data());
  EXPECT_EQ(v.val, r.value().val.data());
  EXPECT_EQ(v.nnz, r.value().nnz());
  EXPECT_EQ(v.n_rows, r.value().n_rows);
  EXPECT_EQ(v.n_cols, r.value().n_cols);
}

#else

TEST(Remap, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB";
}

#endif  // SOMA_HAVE_MOAB
