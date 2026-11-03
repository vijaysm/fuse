// Brusselator weights: the one place this port CAN
// be exact against the Toy (/Users/mahadevan/Code/AI/Toy). Both codes
// remap between two axis-aligned uniform rectangular grids over the same
// unit square, so the true overlap area between ANY source cell and ANY
// target cell is the EXACT PRODUCT of a 1D x-overlap length and a 1D
// y-overlap length -- exactly how the Toy's own `bruss::Transfer`
// constructs its weights (src/brusselator.cpp `overlaps()` + the
// triplet loop in `Transfer::Transfer`). This file reimplements that
// exact, integer-arithmetic `overlaps()` as an ORACLE (not a call into
// the read-only Toy repo) and compares FUSE's `fuse::build_weights` --
// FUSE's native-mesh-backed conservative supermesh path -- against it
// entry by entry, in both directions, at the example's default
// resolution (U 16x16, V 24x24).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/model/coupling_graph.hpp>

#include "brusselator_physics.hpp"

namespace {

/// The Toy's own `overlaps()` (src/brusselator.cpp), reproduced here as an
/// independent oracle: 1D interval overlaps between `source` and `target`
/// equal-width periodic cells over [0,1], using integer cross-products (not
/// floating-point spacing) to decide coincident edges exactly.
struct Overlap1D {
  int source, target;
  double length;
};
[[nodiscard]] std::vector<Overlap1D> overlaps(int source, int target) {
  std::vector<Overlap1D> result;
  int i = 0, j = 0;
  while (i < source && j < target) {
    const double lo = std::max(double(i) / source, double(j) / target);
    const double hi = std::min(double(i + 1) / source, double(j + 1) / target);
    if (hi > lo) result.push_back({i, j, hi - lo});
    const auto a = static_cast<long long>(i + 1) * target;
    const auto b = static_cast<long long>(j + 1) * source;
    if (a <= b) ++i;
    if (b <= a) ++j;
  }
  return result;
}

/// Dense oracle weight `W[target_canonical][source_canonical]` for a
/// `src_nx x src_ny` source grid onto a `tgt_nx x tgt_ny` target grid,
/// `W[T,S] = area(T ^ S) / area(T)` -- the Toy's own convention
/// (NUMERICS.md "Spatial transfer"), built as the product of independent
/// x/y 1D overlaps exactly as `Transfer::Transfer` does
/// (`x.length*y.length*target.size()`, since `area(T)=1/target.size()`
/// on a uniform grid).
[[nodiscard]] std::map<std::pair<int, int>, double> oracle_weights(int src_nx, int src_ny,
                                                                   int tgt_nx, int tgt_ny) {
  const auto xs = overlaps(src_nx, tgt_nx);
  const auto ys = overlaps(src_ny, tgt_ny);
  std::map<std::pair<int, int>, double> w;
  const double tgt_size = static_cast<double>(tgt_nx) * tgt_ny;
  for (const auto& y : ys) {
    for (const auto& x : xs) {
      const int t = y.target * tgt_nx + x.target;
      const int s = y.source * src_nx + x.source;
      w[{t, s}] += x.length * y.length * tgt_size;
    }
  }
  return w;
}

struct MeshPair {
  fuse::native::NativeMeshBackend backend{0, 1};
  brusselator::NativeGrid u, v;
};

[[nodiscard]] MeshPair make_pair(int nx_u, int ny_u, int nx_v, int ny_v) {
  fuse::native::NativeMeshBackend backend(0, 1);
  auto u = brusselator::NativeGrid::make(backend, "u", nx_u, ny_u);
  auto v = brusselator::NativeGrid::make(backend, "v", nx_v, ny_v);
  if (!u.ok() || !v.ok()) {
    ADD_FAILURE() << "NativeGrid::make failed: "
                  << (u.ok() ? "" : u.error().message) << (v.ok() ? "" : v.error().message);
  }
  return MeshPair{std::move(backend), std::move(u.value()), std::move(v.value())};
}

/// Compares FUSE's `fuse::build_weights` result (native-mesh-index
/// triplets) against `oracle_weights(src_nx, src_ny, tgt_nx, tgt_ny)`,
/// entry by entry, after mapping FUSE's native row/col indices back to
/// canonical (j*nx+i) through `NativeGrid::native_index_of`'s inverse.
/// Returns the maximum absolute entry difference (an oracle entry never
/// listed by FUSE, or vice versa, counts as a difference against 0).
[[nodiscard]] double max_weight_difference(const fuse::RemapWeights& w,
                                           const brusselator::NativeGrid& src_grid,
                                           const brusselator::NativeGrid& tgt_grid,
                                           const std::map<std::pair<int, int>, double>& oracle) {
  std::vector<int> native_to_canon_src(static_cast<std::size_t>(src_grid.size()));
  for (int c = 0; c < src_grid.size(); ++c) native_to_canon_src[static_cast<std::size_t>(src_grid.native_index_of(c))] = c;
  std::vector<int> native_to_canon_tgt(static_cast<std::size_t>(tgt_grid.size()));
  for (int c = 0; c < tgt_grid.size(); ++c) native_to_canon_tgt[static_cast<std::size_t>(tgt_grid.native_index_of(c))] = c;

  std::map<std::pair<int, int>, double> fuse_w;
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    const int t = native_to_canon_tgt[static_cast<std::size_t>(w.row[k])];
    const int s = native_to_canon_src[static_cast<std::size_t>(w.col[k])];
    fuse_w[{t, s}] += w.val[k];
  }
  double max_diff = 0.0;
  std::map<std::pair<int, int>, double> all = oracle;
  for (const auto& [key, val] : fuse_w) all[key] += 0.0;  // ensure key present
  for (const auto& [key, oracle_val] : all) {
    const double f = fuse_w.count(key) ? fuse_w.at(key) : 0.0;
    const double o = oracle.count(key) ? oracle.at(key) : 0.0;
    max_diff = std::max(max_diff, std::abs(f - o));
  }
  for (const auto& [key, f] : fuse_w) {
    const double o = oracle.count(key) ? oracle.at(key) : 0.0;
    max_diff = std::max(max_diff, std::abs(f - o));
  }
  return max_diff;
}

[[nodiscard]] fuse::Result<fuse::RemapWeights> weights_between(fuse::MeshBackend& backend,
                                                               fuse::MeshId src_mesh,
                                                               fuse::MeshId tgt_mesh,
                                                               fuse::CouplingGraph* graph) {
  const fuse::RegionId src_r = graph->add_region({.name = "src", .mesh = src_mesh, .dimension = 2});
  const fuse::RegionId tgt_r = graph->add_region({.name = "tgt", .mesh = tgt_mesh, .dimension = 2});
  fuse::RemapPlan plan;
  plan.src = src_r;
  plan.tgt = tgt_r;
  return fuse::build_weights(plan, *graph, backend);
}

}  // namespace

// ---- default resolution, both directions -----------------------------

TEST(BrusselatorWeights, VToUMatchesTheSeparableOracleAtTheDefaultResolution) {
  MeshPair p = make_pair(16, 16, 24, 24);
  fuse::CouplingGraph graph;
  const fuse::MeshId um = graph.add_mesh({.name = "u", .source = "generated:quad:16x16"});
  const fuse::MeshId vm = graph.add_mesh({.name = "v", .source = "generated:quad:24x24"});
  auto w = weights_between(p.backend, vm, um, &graph);  // V (source) -> U (target)
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  const auto oracle = oracle_weights(24, 24, 16, 16);
  const double max_diff = max_weight_difference(w.value(), p.v, p.u, oracle);
  std::printf("BrusselatorWeights V->U: max|FUSE-oracle|=%.3e (nnz=%zu) max_row_defect=%.3e max_col_defect=%.3e\n",
             max_diff, w.value().nnz(), w.value().max_row_defect, w.value().max_col_defect);
  EXPECT_LE(max_diff, 5e-13) << "measured max|FUSE - oracle| = " << max_diff;

  EXPECT_LE(w.value().max_row_defect, 5e-13);
  EXPECT_LE(w.value().max_col_defect, 5e-13);
}

TEST(BrusselatorWeights, UToVMatchesTheSeparableOracleAtTheDefaultResolution) {
  MeshPair p = make_pair(16, 16, 24, 24);
  fuse::CouplingGraph graph;
  const fuse::MeshId um = graph.add_mesh({.name = "u", .source = "generated:quad:16x16"});
  const fuse::MeshId vm = graph.add_mesh({.name = "v", .source = "generated:quad:24x24"});
  auto w = weights_between(p.backend, um, vm, &graph);  // U (source) -> V (target)
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  const auto oracle = oracle_weights(16, 16, 24, 24);
  const double max_diff = max_weight_difference(w.value(), p.u, p.v, oracle);
  std::printf("BrusselatorWeights U->V: max|FUSE-oracle|=%.3e (nnz=%zu) max_row_defect=%.3e max_col_defect=%.3e\n",
             max_diff, w.value().nnz(), w.value().max_row_defect, w.value().max_col_defect);
  EXPECT_LE(max_diff, 5e-13) << "measured max|FUSE - oracle| = " << max_diff;

  EXPECT_LE(w.value().max_row_defect, 5e-13);
  EXPECT_LE(w.value().max_col_defect, 5e-13);
}

// ---- the Toy's own invariants (NUMERICS.md), on the FUSE weights -------

TEST(BrusselatorWeights, RowSumsToOneAndColumnConservesMeasureBothDirections) {
  MeshPair p = make_pair(17, 13, 11, 19);  // the Toy's own "nonmatching" pair
  fuse::CouplingGraph graph;
  const fuse::MeshId um = graph.add_mesh({.name = "u", .source = "generated:quad:17x13"});
  const fuse::MeshId vm = graph.add_mesh({.name = "v", .source = "generated:quad:11x19"});

  for (const bool v_to_u : {true, false}) {
    fuse::CouplingGraph g2;
    const fuse::MeshId a = g2.add_mesh({.name = "u", .source = "generated:quad:17x13"});
    const fuse::MeshId b = g2.add_mesh({.name = "v", .source = "generated:quad:11x19"});
    auto w = v_to_u ? weights_between(p.backend, b, a, &g2) : weights_between(p.backend, a, b, &g2);
    ASSERT_TRUE(w.ok()) << w.error().message;
    EXPECT_NEAR(w.value().max_row_defect, 0.0, 5e-13) << "P*1 = 1 (constant preservation), v_to_u=" << v_to_u;
    EXPECT_NEAR(w.value().max_col_defect, 0.0, 5e-13) << "area_t^T*P = area_s^T (conservation), v_to_u=" << v_to_u;
    EXPECT_NEAR(w.value().src_measure_total, 1.0, 1e-12);
    EXPECT_NEAR(w.value().tgt_measure_total, 1.0, 1e-12);
  }
  (void)um;
  (void)vm;
}

TEST(BrusselatorWeights, ApplyingToAConstantFieldReproducesTheConstant) {
  MeshPair p = make_pair(16, 16, 24, 24);
  fuse::CouplingGraph graph;
  const fuse::MeshId um = graph.add_mesh({.name = "u", .source = "generated:quad:16x16"});
  const fuse::MeshId vm = graph.add_mesh({.name = "v", .source = "generated:quad:24x24"});
  auto w = weights_between(p.backend, vm, um, &graph);
  ASSERT_TRUE(w.ok());
  std::vector<double> ones(static_cast<std::size_t>(p.v.size()), 1.0);
  std::vector<double> out(static_cast<std::size_t>(p.u.size()), 0.0);
  ASSERT_TRUE(fuse::apply_weights(w.value(), ones.data(), out.data(), 1).ok());
  for (double x : out) EXPECT_NEAR(x, 1.0, 5e-13);
}
