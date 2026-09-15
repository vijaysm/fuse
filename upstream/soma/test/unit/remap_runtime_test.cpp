// Task 5.5 -- the conservative remap, through the runner.
//
// Task 5.2 built real FV<->FV supermesh weights (src/remap/intx_weights.cpp)
// and connected them to nothing: every caller of `build_weights`,
// `RemapCache` and `apply_weights` lived in test/unit/remap_test.cpp, and
// `OpKind::Remap` in the interpreter was `num::copy`. This file is the
// evidence that it is no longer so.
//
// WHAT A TEST HERE HAS TO DO, given that the thing being replaced is an
// identity copy:
//
//   (a) IT MUST USE A NON-COINCIDENT PAIR. On two identical meshes the real
//       remap and the identity agree exactly, so such a test cannot fail when
//       the wiring is removed and is worth nothing. Every numeric expectation
//       below is on a pair where the two differ.
//   (b) IT MUST ASSERT ON COUNTS, NOT ONLY ON MAGNITUDES.
//       `RunStats::worst_conservation_error` reads exactly 0 both when a
//       conservative remap was checked and was perfect and when no check ran
//       at all, so a bound on it proves nothing by itself. Every conservation
//       assertion below is paired with `conservation_checks`, and every
//       "a real remap happened" assertion with `remap_weight_builds`.
//   (c) IT MUST NAME WHICH PROPERTY IT IS CHECKING. Row sums unity is CONSTANT
//       PRESERVATION; MEASURE-WEIGHTED column sums is CONSERVATION. The two
//       are not simultaneously attainable unless the two meshes report the
//       same total measure (see soma/remap/remap_plan.hpp). Every mesh pair
//       here tiles the same box exactly, so both hold and both are checked --
//       separately, and labelled.
//
// The pairs are axis-aligned boxes on purpose: `build_weights` intersects
// those exactly, in a measure that agrees with `MeshView::measure` to
// rounding, so the expected numbers below are exact rather than tolerances
// pulled from a run. It also keeps MOAB's advancing front -- and the ~15000
// lines it prints -- out of the log.
//
// IN WHICH MEASURE. The runtime's `Constrain` operator integrates in
// `MeshView::measure` and nothing else -- that is the only measure the runtime
// can see, and it is the one a component's own fluxes are expressed in. On the
// box pairs used here `MeshView::measure` is exact and additive over the
// supermesh, so the two meshes report the same total and the tolerances below
// are at rounding. That does NOT generalise: on two cubed spheres the same
// check would report ~1e-3, because two polyhedral surfaces inscribed in one
// sphere have different total areas (soma/remap/remap_plan.hpp). A weight
// matrix conservative in its BUILDER's measure is not conservative in
// `MeshView`'s when the two disagree, and the number the runtime reports is
// the honest one for the quantity the components exchange.

#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MOAB

#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/moab_backend.hpp>
#include <soma/runtime/runner.hpp>
#include <soma/schedule/plan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <unistd.h>  // getpid, for a per-process scratch directory

using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::MeshDecl;
using soma::MeshView;
using soma::MoabMeshBackend;
using soma::Rational;
using soma::Region;
using soma::RegionId;
using soma::Runner;
using soma::RunnerConfig;
using soma::StepContext;
namespace u = soma::units;

// ---------------------------------------------------------------------------
// Allocation counting, as in runner_test.cpp. Spec §2 layer 3 requires the
// interpreted loop to allocate nothing, and a supermesh build is the largest
// allocation in the library, so this is the measurement that says the weights
// are built at setup and not per step.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_allocations = 0;
bool        g_counting = false;

class AllocationCounter {
 public:
  AllocationCounter() {
    g_allocations = 0;
    g_counting = true;
  }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;
  [[nodiscard]] std::size_t count() const { return g_allocations; }
};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_allocations;
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_counting) ++g_allocations;
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}

namespace {

// ===========================================================================
// A hex mesh written to disk, so the cell ORDER can be chosen
// ===========================================================================
//
// "generated:box:NxNxN" always enumerates its cells the same way, so a
// permuted partner cannot be asked for. The permuted partner is the whole
// point of PermutedPairIsNotAnIdentityCopy below, so that one mesh is written
// out as ASCII legacy VTK -- reproducible from source, no data file in the
// repository, exactly as test/unit/remap_test.cpp does it.

std::filesystem::path scratch_dir() {
  static const std::filesystem::path dir = [] {
    std::filesystem::path d = std::filesystem::temp_directory_path() /
                              ("soma_remap_runtime_test_" +
                               std::to_string(static_cast<long>(::getpid())));
    std::error_code ec;
    std::filesystem::create_directories(d, ec);
    return d;
  }();
  return dir;
}

/// An axis-aligned `n x n x n` hex grid over `[lo,hi]^3`. `reverse` emits the
/// cells last-first, which changes nothing about the geometry and everything
/// about which local index names which cell.
std::string write_box(const std::string& name, int n, double lo, double hi,
                      bool reverse) {
  const std::filesystem::path p = scratch_dir() / (name + ".vtk");
  std::ofstream f(p);
  f.precision(17);
  const int s = n + 1;
  auto vid = [&](int i, int j, int k) { return (k * s + j) * s + i; };
  f << "# vtk DataFile Version 3.0\nsoma remap runtime test\nASCII\n"
    << "DATASET UNSTRUCTURED_GRID\nPOINTS " << (s * s * s) << " double\n";
  for (int k = 0; k < s; ++k) {
    for (int j = 0; j < s; ++j) {
      for (int i = 0; i < s; ++i) {
        f << lo + (hi - lo) * i / n << " " << lo + (hi - lo) * j / n << " "
          << lo + (hi - lo) * k / n << "\n";
      }
    }
  }
  std::vector<std::array<int, 8>> cells;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        cells.push_back({vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                         vid(i, j + 1, k), vid(i, j, k + 1),
                         vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1),
                         vid(i, j + 1, k + 1)});
      }
    }
  }
  if (reverse) std::reverse(cells.begin(), cells.end());
  f << "CELLS " << cells.size() << " " << (cells.size() * 9) << "\n";
  for (const auto& c : cells) {
    f << 8;
    for (const int v : c) f << " " << v;
    f << "\n";
  }
  f << "CELL_TYPES " << cells.size() << "\n";
  for (std::size_t i = 0; i < cells.size(); ++i) f << 12 << "\n";
  f.close();
  EXPECT_TRUE(std::filesystem::exists(p));
  return p.string();
}

// ===========================================================================
// Components
// ===========================================================================

/// Writes a fixed, spatially varying profile into its registered array on
/// every advance. Fixed in time on purpose: this file is about the SPATIAL
/// operator, and a time-varying source would let a temporal bug masquerade as
/// a remap bug.
class Producer final : public soma::Component {
 public:
  explicit Producer(std::vector<double> values)
      : values_(std::move(values)), data_(values_.size(), 0.0) {}

  [[nodiscard]] std::string name() const override { return "producer"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext&) override {
    ++steps;
    // std::copy, NOT `data_ = values_`. The Runner resolved a raw pointer into
    // this array at setup and the FieldRef contract pins it for the whole run
    // (soma/runtime/component.hpp); a copy-assignment is not required to reuse
    // the buffer, and a swap certainly would not.
    std::copy(values_.begin(), values_.end(), data_.begin());
    return soma::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  [[nodiscard]] const double* address() const { return data_.data(); }
  int steps = 0;

 private:
  std::vector<double> values_;
  std::vector<double> data_;
};

class Consumer final : public soma::Component {
 public:
  explicit Consumer(std::size_t n) : data_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return "consumer"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext&) override {
    ++steps;
    return soma::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  int steps = 0;

 private:
  std::vector<double> data_;
};

// ===========================================================================
// Harness
// ===========================================================================

struct Opts {
  std::string        src_mesh = "generated:box:4x4x4";
  std::string        dst_mesh = "generated:box:3x3x3";
  soma::Conservation conservation = soma::Conservation::None;
  Rational           src_dt{1};
  Rational           dst_dt{1};
  soma::AccumOp      accum = soma::AccumOp::Instant;
};

struct Harness {
  CouplingGraph       graph;
  MoabMeshBackend     backend;
  soma::OperatorDAG   dag;
  soma::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;
  RegionId            r1, r2;
  std::size_t         n_src = 0, n_dst = 0;

  [[nodiscard]] Producer* producer() {
    return dynamic_cast<Producer*>(components.find("producer"));
  }
  [[nodiscard]] Consumer* consumer() {
    return dynamic_cast<Consumer*>(components.find("consumer"));
  }
  [[nodiscard]] const MeshView& view(RegionId r) const {
    const auto id = backend.mesh_id(graph.mesh(graph.region(r).mesh).name);
    EXPECT_TRUE(id.ok());
    const auto v = backend.view(id.value());
    EXPECT_TRUE(v.ok());
    return *v.value();
  }
  /// `f(centroid)` on every cell of the region, in local index order.
  template <class F>
  [[nodiscard]] std::vector<double> sample(RegionId r, F f) const {
    const MeshView& v = view(r);
    const int dim = graph.region(r).dimension;
    std::vector<double> out(v.entity_count(dim));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = f(v.centroid(dim, i));
    return out;
  }
  /// `sum_i x_i * measure_i` -- the integral a conservative map preserves.
  [[nodiscard]] double integral(RegionId r,
                                const std::vector<double>& x) const {
    const MeshView& v = view(r);
    const int dim = graph.region(r).dimension;
    EXPECT_EQ(x.size(), v.entity_count(dim));
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) s += x[i] * v.measure(dim, i);
    return s;
  }
};

/// Builds the graph, loads the meshes, lowers and schedules. The producer's
/// values are supplied by the caller *after* the mesh is loaded, so a test can
/// derive them from the source mesh's own geometry; `make` therefore takes the
/// profile as a callable on the source centroid.
template <class F>
std::unique_ptr<Harness> make(const Opts& o, F profile) {
  auto h = std::make_unique<Harness>();
  CouplingGraph& g = h->graph;

  const auto m1 = g.add_mesh(MeshDecl{.name = "m1", .source = o.src_mesh});
  const auto m2 = g.add_mesh(MeshDecl{.name = "m2", .source = o.dst_mesh});
  h->r1 = g.add_region(Region{.name = "r1", .mesh = m1, .dimension = 3});
  h->r2 = g.add_region(Region{.name = "r2", .mesh = m2, .dimension = 3});

  const auto p = g.add_component(soma::ComponentDecl{
      .name = "producer", .ranks = {.first = 0, .count = 1},
      .interval = o.src_dt});
  const auto c = g.add_component(soma::ComponentDecl{
      .name = "consumer", .ranks = {.first = 1, .count = 1},
      .interval = o.dst_dt});
  const auto s = g.add_field(
      p, soma::FieldSpec{.name = "out", .region = h->r1, .units = u::kKelvin,
                         .conservation = o.conservation,
                         .intent = soma::Intent::Provides});
  const auto t = g.add_field(
      c, soma::FieldSpec{.name = "in", .region = h->r2, .units = u::kKelvin,
                         .conservation = o.conservation,
                         .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (o.accum != soma::AccumOp::Instant) g.channel(ch).accumulate(o.accum);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto sz1 = h->backend.region_size(g.region(h->r1));
  const auto sz2 = h->backend.region_size(g.region(h->r2));
  EXPECT_TRUE(sz1.ok());
  EXPECT_TRUE(sz2.ok());
  h->n_src = sz1.value();
  h->n_dst = sz2.value();

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched =
      schedule(g, h->dag, clocks, soma::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  h->components.add(
      std::make_unique<Producer>(h->sample(h->r1, profile)));
  h->components.add(std::make_unique<Consumer>(h->n_dst));
  return h;
}

void bind_all(Harness& h) {
  for (auto& c : h.components) {
    ASSERT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

/// For each target cell, the source cell whose centroid coincides with it, or
/// -1. Used to say what an IDENTITY copy would have delivered versus what the
/// correct permutation delivers, without hard-coding either mesh's ordering.
std::vector<int> centroid_match(const Harness& h) {
  const MeshView& a = h.view(h.r1);
  const MeshView& b = h.view(h.r2);
  std::vector<int> out(b.entity_count(3), -1);
  for (std::size_t j = 0; j < out.size(); ++j) {
    const soma::Point q = b.centroid(3, j);
    for (std::size_t i = 0; i < a.entity_count(3); ++i) {
      const soma::Point p = a.centroid(3, i);
      if (std::abs(p.x - q.x) < 1e-12 && std::abs(p.y - q.y) < 1e-12 &&
          std::abs(p.z - q.z) < 1e-12) {
        out[j] = static_cast<int>(i);
        break;
      }
    }
  }
  return out;
}

}  // namespace

// ===========================================================================
// 1. The remap is not the identity
// ===========================================================================

// THE TEST THAT WOULD HAVE CAUGHT THE OLD BEHAVIOUR.
//
// Two 2x2x2 hex meshes of the same unit cube, differing only in the order the
// cells are enumerated. They have the same cell COUNT (8), the same cell
// MEASURE (0.125 everywhere) and the same total measure, so every guard that
// works from sizes accepts them -- and the element-wise copy the interpreter
// used to perform hands each cell its neighbour's value, silently. The correct
// conservative map between them is the exact permutation matrix, and this test
// pins it value by value.
TEST(RemapRuntime, PermutedPairIsNotAnIdentityCopy) {
  Opts o;
  o.src_mesh = "generated:box:2x2x2";
  o.dst_mesh = write_box("perm2", 2, 0.0, 1.0, /*reverse=*/true);
  // A profile that separates every cell from every other: no two of the eight
  // values coincide, so a permutation error cannot hide behind a symmetry.
  auto h = make(o, [](soma::Point p) {
    return 1.0 + p.x + 10.0 * p.y + 100.0 * p.z;
  });
  ASSERT_EQ(h->n_src, 8u);
  ASSERT_EQ(h->n_dst, 8u);

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  // Counts, not magnitudes: one matrix was built, and it has exactly one
  // non-zero per row because the supermesh of two identical tilings is the
  // tiling itself.
  EXPECT_EQ(runner.stats().remap_weight_builds, 1)
      << "the two meshes are not coincident, so real weights are required";
  EXPECT_EQ(runner.stats().remap_weight_nonzeros, 8)
      << "an exact permutation of 8 cells has 8 non-zeros";

  ASSERT_TRUE(runner.run_period(h->components).ok());

  const std::vector<int>    match = centroid_match(*h);
  const std::vector<double>& src = h->producer()->data();
  const std::vector<double>& got = h->consumer()->data();
  ASSERT_EQ(got.size(), 8u);

  int differs_from_identity = 0;
  for (std::size_t j = 0; j < got.size(); ++j) {
    ASSERT_GE(match[j], 0) << "target cell " << j << " has no source twin";
    // Volume-average over one coincident cell IS that cell's value, exactly.
    EXPECT_NEAR(got[j], src[static_cast<std::size_t>(match[j])], 1e-12)
        << "target cell " << j << " must carry the value of the source cell "
        << "that occupies the same place, not the one at the same index";
    if (std::abs(got[j] - src[j]) > 1e-12) ++differs_from_identity;
  }
  // The point of the whole test: if the interpreter were still copying, every
  // one of these would have matched src[j] instead.
  EXPECT_GT(differs_from_identity, 0)
      << "the permutation is trivial, so this test cannot distinguish a real "
         "remap from a copy and must be rebuilt on a different mesh pair";
}

// ===========================================================================
// 2. Conservation, end to end, through the runner
// ===========================================================================

// WHICH PROPERTY. This asserts CONSERVATION: the measure-weighted sum, i.e.
// `sum_j g_j m^t_j == sum_i f_i m^s_i`, which is the COLUMN property of the
// weight matrix. It is checked twice over -- once by the runtime's own
// `Constrain` operator, whose verdict is `RunStats`, and once here from the
// two components' arrays and the two meshes' measures, because the runtime
// check and the test would otherwise share every assumption they could both
// get wrong.
TEST(RemapRuntime, ConservesTheMeasureWeightedIntegralThroughTheRunner) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";   // 64 cells
  o.dst_mesh = "generated:box:3x3x3";   // 27 cells, non-nesting
  o.conservation = soma::Conservation::Extensive;
  auto h = make(o, [](soma::Point p) {
    return 2.0 + p.x * p.x + 3.0 * p.y - p.z;  // varies in all three axes
  });
  ASSERT_EQ(h->n_src, 64u);
  ASSERT_EQ(h->n_dst, 27u);

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().remap_weight_builds, 1);
  EXPECT_GT(runner.stats().remap_weight_nonzeros, 64)
      << "4x4x4 and 3x3x3 do not nest, so cells must genuinely split";

  ASSERT_TRUE(runner.run(h->components, 3).ok());

  // (i) The runtime's own verdict. The COUNT first: `worst_conservation_error`
  // is 0 both for "checked and perfect" and for "never ran", so the bound
  // below is meaningless without it.
  EXPECT_EQ(runner.stats().conservation_checks, 3);
  EXPECT_EQ(runner.stats().conservation_violations, 0);
  EXPECT_LT(runner.stats().worst_conservation_error, 1e-13);

  // (ii) The same statement, recomputed here.
  const double before = h->integral(h->r1, h->producer()->data());
  const double after = h->integral(h->r2, h->consumer()->data());
  EXPECT_GT(std::abs(before), 1.0) << "a null field would make this vacuous";
  EXPECT_NEAR(after, before, 1e-13 * std::abs(before));

  // (iii) And it is not a copy dressed up: the target values are cell
  // averages over a coarser mesh, so at least one differs from anything on
  // the source.
  const std::vector<double>& got = h->consumer()->data();
  const std::vector<double>& src = h->producer()->data();
  bool any_new = false;
  for (const double gv : got) {
    bool found = false;
    for (const double sv : src) {
      if (std::abs(gv - sv) < 1e-14) found = true;
    }
    if (!found) any_new = true;
  }
  EXPECT_TRUE(any_new)
      << "every target value coincides with a source value, which a genuine "
         "coarsening average should not produce";
}

// WHICH PROPERTY. This asserts CONSTANT PRESERVATION: the plain, unweighted
// ROW sum being one. It is a separate test from the one above because the two
// properties are separate, and a matrix that is wrong one way still looks
// right on the other's test -- a uniform field hides a column error, and a
// measure-weighted sum hides a row error. Both hold here only because the two
// meshes tile the same cube and therefore report the same total measure; on a
// pair where they do not (two cubed spheres, say) this test would have to
// carry that mismatch as its tolerance.
TEST(RemapRuntime, PreservesAConstantThroughTheRunner) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";
  o.dst_mesh = "generated:box:3x3x3";
  auto h = make(o, [](soma::Point) { return 7.25; });

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().remap_weight_builds, 1);
  ASSERT_TRUE(runner.run_period(h->components).ok());

  for (const double v : h->consumer()->data()) {
    EXPECT_NEAR(v, 7.25, 1e-13)
        << "row sums must be one, or a constant field is diluted";
  }
}

// ===========================================================================
// 3. A remap that genuinely loses mass, and the check that says so
// ===========================================================================

// The two tests below replace runner_test.cpp's
// ConservationCheckIntegratesAgainstTheRegionsMeasures and
// AbortsOnAConservationViolationWhenAsked, which drove the conservation check
// with a box-wall-to-line identity copy. That pair is now refused at setup,
// so the check needs a real transfer that really is non-conservative: a
// TRUNCATING remap, where the target mesh covers only part of the source
// domain and the mass outside it is dropped rather than redistributed
// (soma/remap/remap_plan.hpp, "partial overlap: truncate, do not
// extrapolate"). The channel does not set RemapFlags::Conservative, so the
// builder returns the truncating map instead of E-REMAP-COVERAGE.
//
// The arithmetic is exact and hand-computable. Source: 4x4x4 over the unit
// cube, total measure 1. Target: 2x2x2 over [0, 1/2]^3, total measure 1/8, and
// the two cell sizes are both 1/4, so each target cell coincides with exactly
// one source cell and is fully covered. With f = 1 + x, whose cell average is
// exactly 1 + x_centroid because it is linear:
//
//     before = integral over [0,1]^3 of (1+x)     = 1.5
//     after  = integral over [0,1/2]^3 of (1+x)   = 0.125 * 1.25 = 0.15625
//     relative error = (1.5 - 0.15625) / 1.5      = 0.8958333...
//
// The profile VARIES on purpose. A constant source would make this test pass
// against an element-wise copy as well -- the eight copied values would be the
// same constant -- and it would then be measuring nothing about the remap.
TEST(RemapRuntime, ReportsTheMassATruncatingRemapDrops) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";          // [0,1]^3
  o.dst_mesh = write_box("half", 2, 0.0, 0.5, /*reverse=*/false);
  o.conservation = soma::Conservation::Extensive;
  auto h = make(o, [](soma::Point p) { return 1.0 + p.x; });

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().remap_weight_builds, 1);
  EXPECT_EQ(runner.stats().remap_weight_nonzeros, 8)
      << "each of the 8 target cells is exactly one source cell";
  ASSERT_TRUE(runner.run_period(h->components).ok());

  EXPECT_EQ(runner.stats().conservation_checks, 1);
  EXPECT_EQ(runner.stats().conservation_violations, 1);
  EXPECT_NEAR(runner.stats().worst_conservation_error, 1.34375 / 1.5, 1e-12);

  // The delivered field is NOT diluted: every target cell gets its own source
  // cell's value in full. The shortfall is entirely source mass that had
  // nowhere to go, which is the difference between the two coverage figures,
  // and saying so here keeps the number above from reading as "broken remap".
  const std::vector<int>     match = centroid_match(*h);
  const std::vector<double>& src = h->producer()->data();
  const std::vector<double>& got = h->consumer()->data();
  int differs_from_identity = 0;
  for (std::size_t j = 0; j < got.size(); ++j) {
    ASSERT_GE(match[j], 0);
    EXPECT_NEAR(got[j], src[static_cast<std::size_t>(match[j])], 1e-13);
    if (std::abs(got[j] - src[j]) > 1e-13) ++differs_from_identity;
  }
  EXPECT_GT(differs_from_identity, 0)
      << "an element-wise copy would satisfy every other assertion here";
  EXPECT_NEAR(h->integral(h->r2, got), 0.15625, 1e-13);
  EXPECT_NEAR(h->integral(h->r1, src), 1.5, 1e-13);
}

TEST(RemapRuntime, AbortsOnAConservationViolationWhenAsked) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";
  o.dst_mesh = write_box("half_abort", 2, 0.0, 0.5, /*reverse=*/false);
  o.conservation = soma::Conservation::Extensive;
  auto h = make(o, [](soma::Point) { return 4.0; });

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  RunnerConfig cfg;
  cfg.abort_on_violation = true;
  ASSERT_TRUE(runner.initialize(h->components, h->registry, cfg).ok());
  const auto r = runner.run_period(h->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-CONSERVATION");
}

// And the flag the builder offers for exactly this situation is honoured
// through the runner: ask for a conservative map over a pair that cannot
// support one and setup fails, rather than delivering the truncating map
// under the conservative name.
TEST(RemapRuntime, RefusesATruncatingMapWhenTheChannelAsksForConservation) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";
  o.dst_mesh = write_box("half_flag", 2, 0.0, 0.5, /*reverse=*/false);
  auto h = make(o, [](soma::Point) { return 4.0; });
  // Re-lowering with the flag set: the harness builds the graph, so the flag
  // has to be applied to the already-built one and the plan rebuilt.
  h->graph.channel(soma::ChannelId{0}).remap(soma::RemapFlags::Conservative);
  const auto lowered = lower(h->graph);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const soma::ClockSet clocks{h->graph};
  auto sched = schedule(h->graph, h->dag, clocks,
                        soma::make_region_sizes(h->graph, h->backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-COVERAGE");
}

// ===========================================================================
// 4. The weights are built at setup, never in the loop
// ===========================================================================

TEST(RemapRuntime, AllocatesNothingWhileRunningARealRemap) {
  Opts o;
  o.src_mesh = "generated:box:4x4x4";
  o.dst_mesh = "generated:box:3x3x3";
  o.conservation = soma::Conservation::Extensive;
  o.accum = soma::AccumOp::Mean;
  o.src_dt = Rational{1, 4};  // subcycled: Sample, Accumulate, Remap, Constrain
  auto h = make(o, [](soma::Point p) { return 1.0 + p.x + p.y + p.z; });

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_EQ(runner.stats().remap_weight_builds, 1)
      << "a run with no real remap in it would make this test vacuous";

  // One warm-up period outside the counter, so anything allocated lazily
  // exactly once is still caught by the counted periods below.
  ASSERT_TRUE(runner.run_period(h->components).ok());

  std::size_t allocations = 0;
  bool        ok = false;
  {
    const AllocationCounter counter;
    ok = runner.run(h->components, 8).ok();
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  EXPECT_EQ(allocations, 0u)
      << "run() allocated " << allocations
      << " times; a supermesh build in the step loop would show up here as "
         "thousands";
  // Still one matrix after nine periods: nothing rebuilt it.
  EXPECT_EQ(runner.stats().remap_weight_builds, 1);
}

// The registered arrays must not move: the Runner resolved raw pointers into
// them at setup (soma/runtime/component.hpp). The producer reassigns its
// vector every advance, which is safe only because the size never changes.
TEST(RemapRuntime, RegisteredAddressesDoNotMove) {
  Opts o;
  o.src_mesh = "generated:box:2x2x2";
  o.dst_mesh = write_box("perm_addr", 2, 0.0, 1.0, /*reverse=*/true);
  auto h = make(o, [](soma::Point p) { return 1.0 + p.x + 10.0 * p.y; });

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const double* before = h->producer()->address();
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 4).ok());
  EXPECT_EQ(h->producer()->address(), before);
}

#else  // SOMA_HAVE_MOAB

TEST(RemapRuntime, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without MOAB; the conservative remap has no "
                  "dependency-free supermesh, so the runtime falls back to "
                  "the proved-coincident identity and refuses everything "
                  "else (test/unit/runner_test.cpp covers that path)";
}

#endif  // SOMA_HAVE_MOAB
