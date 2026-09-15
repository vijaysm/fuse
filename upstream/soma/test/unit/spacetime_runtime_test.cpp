// Task 6.3 -- Channel::spacetime, end to end.
//
// Spec §7.3's `M = T (x) W` was built in Tasks 6.1/6.2 (src/remap/spacetime.cpp)
// and connected to nothing: `Channel::spacetime` crossed the descriptor and the
// YAML and then vanished, and the runtime's `OpKind::Interpolate` was an
// identity copy marked "Phase 6 replaces this". This file pins the wiring.
//
// WHAT A TEST HERE HAS TO DO. A space-time remap that is silently still a copy
// produces plausible numbers, and for `TemporalBasis::Constant` in the
// *refining* direction (one source step feeding several target steps) the two
// genuinely agree, so a test built on that configuration proves nothing. Every
// numeric expectation below is therefore
//
//   (a) in the COARSENING direction, where the projection is an average over
//       several source steps and the copy is the last one, so they differ; and
//   (b) hand-computed from the definition of the conservative projection, not
//       read back out of `build_temporal_projection`, so a wrong operator
//       cannot agree with the expectation by construction; and
//   (c) different for different degrees on the SAME schedule, so a silent
//       downgrade to piecewise-constant is a failure and not a rounding
//       difference.
//
// The 3-source-steps-to-2-target-steps schedule below is deliberately
// non-nesting. When the two partitions nest, every slab is a whole source step
// and the degree-p reconstruction integrates to the same cell mean whatever p
// is -- Constant and DG(1) produce the SAME matrix and the test cannot tell a
// downgrade from the real thing. 1/3 against 1/2 has two partial slabs and does
// not have that problem.

#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/remap/spacetime.hpp>
#include <soma/runtime/runner.hpp>
#include <soma/schedule/plan.hpp>
#include <soma/schedule/slab.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::OpKind;
using soma::Rational;
using soma::Runner;
using soma::SerialMeshBackend;
using soma::StepContext;
using soma::TemporalBasis;
namespace u = soma::units;

// ---------------------------------------------------------------------------
// Allocation counting, as in runner_test.cpp: spec §2 layer 3 requires the
// interpreted loop to allocate nothing, and a space-time channel adds a
// history buffer and a temporal operator to it. Measured, not asserted by
// inspection.
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

constexpr std::size_t kCells = 4;

/// Writes `step` into every cell, where `step` counts advance() calls from 1.
/// The value therefore changes on every source step, which is the whole point:
/// a temporal projection and a copy of the latest value can only be told apart
/// on a signal that varies in time.
class Ramp final : public soma::Component {
 public:
  [[nodiscard]] std::string name() const override { return "src"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("src");
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
    for (double& v : data_) v = static_cast<double>(steps);
    return soma::Result<void>::success();
  }

  int steps = 0;

 private:
  std::vector<double> data_ = std::vector<double>(kCells, 0.0);
};

class Sink final : public soma::Component {
 public:
  [[nodiscard]] std::string name() const override { return "dst"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("dst");
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
    seen.push_back(data_[0]);
    return soma::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

  /// data_[0] as it stood at the start of each advance(). Reserved up front so
  /// a push_back cannot show up as the Runner's allocation.
  std::vector<double> seen = [] {
    std::vector<double> v;
    v.reserve(64);
    return v;
  }();

 private:
  std::vector<double> data_ = std::vector<double>(kCells, 0.0);
};

struct Opts {
  Rational      src_dt{1, 3};
  Rational      dst_dt{1, 2};
  TemporalBasis basis = TemporalBasis::DG;
  int           degree = 1;
  bool          enable_spacetime = true;
  soma::AccumOp accum = soma::AccumOp::Instant;
  soma::Window  window = soma::Window::coupling_period();
};

struct Harness {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  soma::OperatorDAG   dag;
  soma::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;

  [[nodiscard]] soma::ArcId arc_of(std::string_view label) const {
    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const soma::Node& n = dag.node(soma::NodeId{i});
      if (n.label == label) return n.output;
    }
    return soma::ArcId{};
  }
  [[nodiscard]] const soma::Node* node_of(std::string_view label) const {
    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const soma::Node& n = dag.node(soma::NodeId{i});
      if (n.label == label) return &n;
    }
    return nullptr;
  }
  [[nodiscard]] Sink* sink() {
    return dynamic_cast<Sink*>(components.find("dst"));
  }
};

/// Both regions are the same 4-cell line, so the spatial half of the operator
/// is the identity and any difference the tests see is temporal.
std::unique_ptr<Harness> build(Opts o) {
  auto h = std::make_unique<Harness>();
  CouplingGraph& g = h->graph;

  const std::string src = "generated:line:" + std::to_string(kCells);
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1", .source = src});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2", .source = src});
  const auto r1 =
      g.add_region(soma::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(soma::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto pc = g.add_component(soma::ComponentDecl{
      .name = "src", .ranks = {.first = 0, .count = 1}, .interval = o.src_dt});
  const auto cc = g.add_component(soma::ComponentDecl{
      .name = "dst", .ranks = {.first = 1, .count = 1}, .interval = o.dst_dt});
  const auto sf = g.add_field(
      pc, soma::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                          .intent = soma::Intent::Provides});
  const auto tf = g.add_field(
      cc, soma::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                          .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("link", sf, tf);
  if (o.enable_spacetime) {
    g.channel(ch).spacetime(o.basis, o.degree, o.window);
  }
  if (o.accum != soma::AccumOp::Instant) g.channel(ch).accumulate(o.accum);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched =
      schedule(g, h->dag, clocks, soma::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  h->components.add(std::make_unique<Ramp>());
  h->components.add(std::make_unique<Sink>());
  return h;
}

void bind_all(Harness& h) {
  for (auto& c : h.components) {
    ASSERT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

/// Lower + schedule + initialize `o` and return the first error code, or ""
/// when the whole setup succeeded. Deliberately spans all three layers: which
/// one rejects a configuration is an implementation detail, that it is
/// rejected before it computes a number is not.
std::string setup_error(Opts o) {
  CouplingGraph g;
  SerialMeshBackend backend;

  const std::string src = "generated:line:" + std::to_string(kCells);
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1", .source = src});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2", .source = src});
  const auto r1 =
      g.add_region(soma::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(soma::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto pc = g.add_component(soma::ComponentDecl{
      .name = "src", .ranks = {.first = 0, .count = 1}, .interval = o.src_dt});
  const auto cc = g.add_component(soma::ComponentDecl{
      .name = "dst", .ranks = {.first = 1, .count = 1}, .interval = o.dst_dt});
  const auto sf = g.add_field(
      pc, soma::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                          .intent = soma::Intent::Provides});
  const auto tf = g.add_field(
      cc, soma::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                          .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("link", sf, tf);
  if (o.enable_spacetime) g.channel(ch).spacetime(o.basis, o.degree, o.window);
  if (o.accum != soma::AccumOp::Instant) g.channel(ch).accumulate(o.accum);

  EXPECT_TRUE(backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  if (!lowered.ok()) return lowered.diags.items().front().code;

  const soma::ClockSet clocks{g};
  auto sched =
      schedule(g, lowered.dag, clocks, soma::make_region_sizes(g, backend));
  if (!sched.ok()) return sched.diags.items().front().code;

  ComponentSet components;
  components.add(std::make_unique<Ramp>());
  components.add(std::make_unique<Sink>());
  FieldRegistry registry;
  for (auto& c : components) {
    EXPECT_TRUE(c.register_fields(registry, g).ok());
  }
  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto r = runner.initialize(components, registry);
  return r.ok() ? std::string{} : r.error().code;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Lowering
// ---------------------------------------------------------------------------

TEST(SpacetimeLowering, EmitsASampleAndAnInterpolateCarryingTheBasis) {
  auto h = build(Opts{.basis = TemporalBasis::DG, .degree = 1});

  const soma::Node* sample = h->node_of("link/spacetime_sample");
  const soma::Node* proj = h->node_of("link/spacetime");
  ASSERT_NE(sample, nullptr) << "lowering emitted no space-time Sample";
  ASSERT_NE(proj, nullptr) << "lowering emitted no space-time Interpolate";

  EXPECT_EQ(sample->kind, OpKind::Sample);
  EXPECT_EQ(proj->kind, OpKind::Interpolate);

  // The Sample's value is a Waveform -- a window of samples, not an instant --
  // which is what makes plan_buffers mark it persistent across the period.
  EXPECT_EQ(h->dag.arc(sample->output).type.kind, soma::ValueKind::Waveform);
  EXPECT_EQ(h->dag.arc(proj->output).type.kind, soma::ValueKind::Instant);

  // The projection reads the sample, and carries the basis and degree the
  // channel asked for. Without the attributes the runtime cannot build `T`.
  ASSERT_EQ(proj->inputs.size(), 1u);
  EXPECT_EQ(proj->inputs[0], sample->output);
  EXPECT_EQ(proj->attrs.basis, TemporalBasis::DG);
  EXPECT_EQ(proj->attrs.degree, 1);
}

TEST(SpacetimeLowering, EmitsNeitherNodeWithoutASpacetimeSpec) {
  auto h = build(Opts{.enable_spacetime = false});
  EXPECT_EQ(h->node_of("link/spacetime_sample"), nullptr);
  EXPECT_EQ(h->node_of("link/spacetime"), nullptr);
  for (std::uint32_t i = 0; i < h->dag.node_count(); ++i) {
    EXPECT_NE(h->dag.node(soma::NodeId{i}).kind, OpKind::Interpolate);
  }
}

/// The two halves run on two different clocks, and that is not cosmetic: the
/// Sample must see every source step (three of them here) while the projection
/// closes each target step (two). A schedule that put both on one clock could
/// not supply the operator's operands.
TEST(SpacetimeLowering, ScheduleRunsSampleOnSourceStepsAndProjectionOnTarget) {
  auto h = build(Opts{});  // 1/3 against 1/2, so 3 source and 2 target steps

  std::vector<std::int64_t> sampled;
  std::vector<std::int64_t> projected;
  std::vector<Rational>     sample_times;
  std::vector<Rational>     project_times;
  for (const auto& ins : h->plan.instructions()) {
    if (ins.kind == OpKind::Sample) {
      sampled.push_back(ins.src_step);
      sample_times.push_back(ins.time);
    }
    if (ins.kind == OpKind::Interpolate) {
      projected.push_back(ins.dst_step);
      project_times.push_back(ins.time);
    }
  }
  EXPECT_EQ(sampled, (std::vector<std::int64_t>{0, 1, 2}));
  EXPECT_EQ(projected, (std::vector<std::int64_t>{0, 1}));
  EXPECT_EQ(sample_times, (std::vector<Rational>{Rational(0), Rational(1, 3),
                                                 Rational(2, 3)}));
  EXPECT_EQ(project_times,
            (std::vector<Rational>{Rational(1, 2), Rational(1)}));
}

// ---------------------------------------------------------------------------
// 2. The runtime really applies T, and not a copy
// ---------------------------------------------------------------------------
//
// Schedule: source interval 1/3, target interval 1/2, common period 1. The
// source produces 1, 2, 3 over steps 0, 1, 2 of the first period.
//
// The supermesh of the two partitions is
//     [0, 1/3) j=0 i=0     [1/3, 1/2) j=1 i=0
//     [1/2, 2/3) j=1 i=1   [2/3, 1)   j=2 i=1
//
// DEGREE 0 (piecewise constant in time). T_ij is the fraction of target step i
// covered by source step j:
//     row 0 = (2/3, 1/3, 0)        row 1 = (0, 1/3, 2/3)
// so target step 0 receives 2/3*1 + 1/3*2 = 4/3 and step 1 receives
// 1/3*2 + 2/3*3 = 8/3.
//
// DEGREE 1 (piecewise linear in time). Inside source step j the reconstruction
// is the unique line whose means over steps {j-1, j} (the stencil, clamped into
// the window) are the given coefficients; each slab then contributes the exact
// mean of that line over the slab, scaled by src_dt/dst_dt = 2/3. Working the
// two partial slabs through by hand:
//     row 0 = (3/4, 1/4, 0)        row 1 = (-1/12, 5/12, 2/3)
// so target step 0 receives 3/4*1 + 1/4*2 = 5/4 and step 1 receives
// -1/12*1 + 5/12*2 + 2/3*3 = 11/4.
//
// Both rows sum to one in both cases -- constant preservation, spec §7.3's
// property 1 -- which is the arithmetic check on the hand computation.
//
// A COPY of the source arc would deliver 2 at t = 1/2 (source step 1 is the
// latest one advanced) and 3 at t = 1. All six numbers are distinct.
//
// WHAT IS OBSERVABLE. `Runner::buffer()` reads the projection's arc, which by
// the end of the period holds the LAST target step, so the runtime assertions
// below are on row 1 -- the row with a partial slab, a whole slab, all three
// columns and a negative weight, i.e. the more discriminating of the two. Row 0
// cannot be read through the public surface mid-period; it is pinned against
// the same hand numbers one level down, directly on the matrix, in
// `TheMatrixMatchesTheHandComputation`.

namespace {

/// The stencil-start formula stated in the header comment
/// (`include/soma/remap/spacetime.hpp`, "STENCILS AT THE ENDS OF THE
/// WINDOW"), reimplemented HERE from the prose rather than copied from
/// `src/remap/spacetime.cpp`: symmetric for even `p`, one step towards the
/// past for odd `p`, clamped into `[0, n_src - 1 - p]`. Used only by
/// `StencilPlacementMatchesTheDocumentedFormulaAcrossASweep` below as an
/// independent oracle for where the stencil OUGHT to sit.
[[nodiscard]] std::int64_t documented_stencil_start(std::int64_t j, int p,
                                                    std::int64_t n_src) {
  std::int64_t       start = j - (p + 1) / 2;
  const std::int64_t last  = n_src - 1 - p;
  if (start < 0) start = 0;
  if (start > last) start = last;
  return start;
}

/// Run one period and return what the projection wrote at the LAST target
/// step of it.
double project_last_target_step(Harness& h) {
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);
  EXPECT_TRUE(runner.initialize(h.components, h.registry).ok());

  const auto r = runner.run_period(h.components);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);

  const auto buf = runner.buffer(h.arc_of("link/spacetime"));
  EXPECT_TRUE(buf.ok());
  return buf.ok() ? buf.value()[0] : 0.0;
}

}  // namespace

/// The two hand-computed rows, checked against the operator the runtime
/// actually builds for this schedule. This is what lets the runtime tests
/// above assert one number and still claim the whole matrix is right, and it
/// is a real check rather than a tautology: the numbers on the right were
/// derived from the definition of the conservative projection with pen and
/// paper, not read out of the library.
TEST(SpacetimeMatrix, TheMatrixMatchesTheHandComputation) {
  const auto slabs =
      soma::decompose(Rational{1, 3}, Rational{1, 2}, Rational{1});
  ASSERT_TRUE(slabs.ok()) << slabs.error().message;

  const auto p0 = soma::build_temporal_projection(
      slabs.value(), TemporalBasis::Constant, 0, Rational{1, 3},
      Rational{1, 2});
  ASSERT_TRUE(p0.ok()) << p0.error().message;
  EXPECT_NEAR(p0.value().coefficient(0, 0), 2.0 / 3.0, 1e-14);
  EXPECT_NEAR(p0.value().coefficient(0, 1), 1.0 / 3.0, 1e-14);
  EXPECT_NEAR(p0.value().coefficient(0, 2), 0.0, 1e-14);
  EXPECT_NEAR(p0.value().coefficient(1, 0), 0.0, 1e-14);
  EXPECT_NEAR(p0.value().coefficient(1, 1), 1.0 / 3.0, 1e-14);
  EXPECT_NEAR(p0.value().coefficient(1, 2), 2.0 / 3.0, 1e-14);

  const auto p1 = soma::build_temporal_projection(
      slabs.value(), TemporalBasis::DG, 1, Rational{1, 3}, Rational{1, 2});
  ASSERT_TRUE(p1.ok()) << p1.error().message;
  EXPECT_NEAR(p1.value().coefficient(0, 0), 3.0 / 4.0, 1e-14);
  EXPECT_NEAR(p1.value().coefficient(0, 1), 1.0 / 4.0, 1e-14);
  EXPECT_NEAR(p1.value().coefficient(0, 2), 0.0, 1e-14);
  EXPECT_NEAR(p1.value().coefficient(1, 0), -1.0 / 12.0, 1e-14);
  EXPECT_NEAR(p1.value().coefficient(1, 1), 5.0 / 12.0, 1e-14);
  EXPECT_NEAR(p1.value().coefficient(1, 2), 2.0 / 3.0, 1e-14);
}

// ---------------------------------------------------------------------------
// Degrees 2, 3, 4: entry-level golden checks closing a verification gap.
// ---------------------------------------------------------------------------
//
// THE GAP. `TheMatrixMatchesTheHandComputation` above pins entries at degree 0
// and 1 only. Every conservation property in spacetime_conservation_test.cpp
// -- row sums are unity, column sums carry dt_src -- is a STRUCTURAL
// consequence of using ANY internally-consistent degree-p moment-matching
// reconstruction over ANY p+1-wide stencil containing step j, not a
// consequence of using the PARTICULAR stencil position
// `include/soma/remap/spacetime.hpp` documents ("symmetric for even p, one
// step towards the past for odd p"). Moving the stencil -- e.g. mistyping
// `stencil_start`'s `start = j - (p + 1) / 2` as `start = j - p` -- therefore
// changes the matrix at degrees 2, 3 and 4 without tripping a single
// conservation test. Verified directly: applying that exact one-line edit to
// `stencil_start` in src/remap/spacetime.cpp and rebuilding leaves every test
// in spacetime_conservation_test.cpp green while the three tests below fail
// (see the task report for the before/after `ctest` output). Reverted before
// this file was committed.
//
// DERIVATION METHOD, shared by the three tests below. Source step j's
// reconstruction is the unique polynomial of degree p whose average over each
// of the p+1 stencil cells (`stencil_start(j, p, n_src) .. + p`) equals the
// given cell means. Working in `s = t/dt_src - j`, so step j itself is the
// cell [-1/2, 1/2], the moment of s^r over a unit cell centred at integer
// offset d is, exactly (integrate s^r from d-1/2 to d+1/2; odd cross terms
// cancel):
//
//     M_0(d) = 1
//     M_1(d) = d
//     M_2(d) = d^2 + 1/12
//     M_3(d) = d^3 + d/4
//     M_4(d) = d^4 + d^2/2 + 1/80
//
// The stencil weights w_k solve sum_k w_k * M_r(d_k) = mom_r for r = 0..p,
// where mom_r is the same moment taken over the SLAB's sub-interval instead
// of a whole cell and d_k is stencil cell k's offset from j. The matrix entry
// is `w_scale * w_k` at column `start + k`, w_scale = dt_src/dt_tgt -- exactly
// what `build_impl` computes (src/remap/spacetime.cpp); the derivations below
// solve the same system by hand and were cross-checked against an
// independent exact-rational script implementing this same definition from
// scratch, never against the library.
//
// LEMMA (used below without rederiving it each time): when a slab IS an
// entire stencil cell -- offset d = 0, the slab is the whole of step j and j
// is in its own stencil -- mom equals M(0) exactly, so w is the standard
// basis vector picking out that cell: weight 1 on column j, 0 elsewhere. This
// just says the reconstruction reproduces the cell mean it was built from; it
// holds at every degree.

/// DEGREE 2, schedule 5 source steps -> 3 target steps over period 1
/// (src_dt = 1/5, dst_dt = 1/3). Target step 1 covers real time [1/3, 2/3)
/// and, in fifteenths of the period, is fed by three slabs:
///     [5,6) j=1 (start=0)   [6,9) j=2, WHOLE cell (start=1)   [9,10) j=3 (start=2)
/// -- source steps 1, 2 AND 3, which is exactly the run the mutation
/// relocates: stencil_start(2,2,5) is 1 correctly and 0 mutated;
/// stencil_start(3,2,5) is 2 correctly and 1 mutated.
///
/// SLAB j=1, [5,6)/15: stencil {0,1,2}, offsets d=-1,0,1. In j=1's own
/// coordinate the slab is s in [1/6, 1/2] (u_lo = (5/15)/(1/5) - 1 - 1/2 =
/// 5/3 - 3/2 = 1/6), so mom = (1/3, 1/9, 13/324). Solving
///     w0 + w1 + w2 = 1/3
///     -w0 + w2 = 1/9                          (M_1 at d=-1,0,1)
///     (13/12)w0 + (1/12)w1 + (13/12)w2 = 13/324
/// gives w0=-4/81, w1=26/81, w2=5/81 on columns 0,1,2 (sum = 27/81 = 1/3,
/// matching mom0).
///
/// SLAB j=2, [6,9)/15: the WHOLE of step 2 -- weight 1 on column 2, 0
/// elsewhere, by the lemma.
///
/// SLAB j=3, [9,10)/15: stencil {2,3,4}, offsets d=-1,0,1 again. The slab is s
/// in [-1/2,-1/6] in j=3's own coordinate -- the mirror image of j=1's slab
/// about the cell centre -- and the same system with mom=(1/3,-1/9,13/324)
/// gives the mirrored weights w0=5/81, w1=26/81, w2=-4/81 on columns 2,3,4.
///
/// w_scale = dt_src/dt_tgt = (1/5)/(1/3) = 3/5. Summing the three slabs:
///     col0: (3/5)(-4/81)                          = -4/135
///     col1: (3/5)(26/81)                          = 26/135
///     col2: (3/5)(5/81) + (3/5)(1) + (3/5)(5/81)  = 91/135
///     col3: (3/5)(26/81)                          = 26/135
///     col4: (3/5)(-4/81)                          = -4/135
/// Row sum = (-4+26+91+26-4)/135 = 135/135 = 1 -- the arithmetic self-check.
TEST(SpacetimeMatrix, DegreeTwoMatchesTheHandComputation) {
  const auto slabs =
      soma::decompose(Rational{1, 5}, Rational{1, 3}, Rational{1});
  ASSERT_TRUE(slabs.ok()) << slabs.error().message;
  const auto p2 = soma::build_temporal_projection(
      slabs.value(), TemporalBasis::DG, 2, Rational{1, 5}, Rational{1, 3});
  ASSERT_TRUE(p2.ok()) << p2.error().message;
  EXPECT_NEAR(p2.value().coefficient(1, 0), -4.0 / 135.0, 1e-13);
  EXPECT_NEAR(p2.value().coefficient(1, 1), 26.0 / 135.0, 1e-13);
  EXPECT_NEAR(p2.value().coefficient(1, 2), 91.0 / 135.0, 1e-13);
  EXPECT_NEAR(p2.value().coefficient(1, 3), 26.0 / 135.0, 1e-13);
  EXPECT_NEAR(p2.value().coefficient(1, 4), -4.0 / 135.0, 1e-13);
}

/// DEGREE 3, schedule 7 source steps -> 4 target steps over period 1
/// (src_dt = 1/7, dst_dt = 1/4). Target step 2 covers real time [1/2, 3/4)
/// and, in twenty-eighths of the period, is fed by
///     [14,16) j=3 (start=1)  [16,20) j=4, WHOLE cell (start=2)  [20,21) j=5 (start=3)
/// -- source steps 3, 4 and 5; stencil_start(3,3,7) is 1 correctly and 0
/// mutated, stencil_start(5,3,7) is 3 correctly and 2 mutated.
///
/// Every unclamped degree-3 stencil has offsets d=-2,-1,0,1 (start = j-2), so
/// the same 4x4 system recurs for every slab and only the moments change.
/// Using M_1(d)=d and M_3(d)=d^3+d/4 at d=-2,-1,0,1 and eliminating w2 via the
/// row-0 equation collapses the system to closed form:
///     w0 = (5*mom1 - 4*mom3) / 24
///     w3 = (mom1 + mom2 - mom0/12)/2 - w0
///     w1 = (mom2 - mom0/12) - 4*w0 - w3
///     w2 = mom0 - w0 - w1 - w3
///
/// SLAB j=3, [14,16)/28 = s in [0, 1/2] in step 3's own coordinate:
/// mom=(1/2, 1/8, 1/24, 1/64), giving w=(3/128,-17/128,73/128,5/128) on
/// columns 1,2,3,4.
///
/// SLAB j=4, [16,20)/28: the WHOLE of step 4 -- weight 1 on column 4, 0
/// elsewhere, by the lemma.
///
/// SLAB j=5, [20,21)/28 = s in [-1/2,-1/4] in step 5's own coordinate:
/// mom=(1/4,-3/32,7/192,-15/1024), giving w=(-35/2048,217/2048,375/2048,
/// -45/2048) on columns 3,4,5,6.
///
/// w_scale = dt_src/dt_tgt = (1/7)/(1/4) = 4/7. Summing, over a common
/// denominator of 3584:
///     col1: (4/7)(3/128)                               = 48/3584
///     col2: (4/7)(-17/128)                              = -272/3584
///     col3: (4/7)(73/128) + (4/7)(-35/2048)             = 1133/3584
///     col4: (4/7)(5/128) + (4/7)(1) + (4/7)(217/2048)   = 2345/3584 = 335/512
///     col5: (4/7)(375/2048)                             = 375/3584
///     col6: (4/7)(-45/2048)                             = -45/3584
/// Row sum: (48-272+1133+2345+375-45)/3584 = 3584/3584 = 1.
TEST(SpacetimeMatrix, DegreeThreeMatchesTheHandComputation) {
  const auto slabs =
      soma::decompose(Rational{1, 7}, Rational{1, 4}, Rational{1});
  ASSERT_TRUE(slabs.ok()) << slabs.error().message;
  const auto p3 = soma::build_temporal_projection(
      slabs.value(), TemporalBasis::DG, 3, Rational{1, 7}, Rational{1, 4});
  ASSERT_TRUE(p3.ok()) << p3.error().message;
  EXPECT_NEAR(p3.value().coefficient(2, 1), 3.0 / 224.0, 1e-13);
  EXPECT_NEAR(p3.value().coefficient(2, 2), -17.0 / 224.0, 1e-13);
  EXPECT_NEAR(p3.value().coefficient(2, 3), 1133.0 / 3584.0, 1e-13);
  EXPECT_NEAR(p3.value().coefficient(2, 4), 335.0 / 512.0, 1e-13);
  EXPECT_NEAR(p3.value().coefficient(2, 5), 375.0 / 3584.0, 1e-13);
  EXPECT_NEAR(p3.value().coefficient(2, 6), -45.0 / 3584.0, 1e-13);
}

/// DEGREE 4, schedule 9 source steps -> 6 target steps over period 1
/// (src_dt = 1/9, dst_dt = 1/6). Target step 2 covers real time [1/3, 1/2)
/// and, in ninths of the period, is fed by [3,4) j=3 (WHOLE cell) and
/// [4,4.5) j=4 (start=2) -- stencil_start(4,4,9) is 2 correctly and 0
/// mutated.
///
/// The unclamped degree-4 stencil is symmetric, offsets d=-2,-1,0,1,2 (start
/// = j-2). Exploit the symmetry: S1=w_{-2}+w_2, D1=w_2-w_{-2},
/// S2=w_{-1}+w_1, D2=w_1-w_{-1}. M_0, M_2, M_4 are even in d and M_1, M_3 are
/// odd, so the r=0,2,4 equations involve only (S1,S2,w0) and the r=1,3
/// equations only (D1,D2): two decoupled, smaller systems.
///
/// SLAB j=4, [4,4.5)/9 = s in [-1/2, 0] in step 4's own coordinate:
/// mom=(1/2,-1/8,1/24,-1/64,1/160).
///   Antisymmetric: 2*D1+D2=-1/8, (17/2)*D1+(5/4)*D2=-1/64
///     => D1=3/128, D2=-11/64.
///   Symmetric: S1+S2+w0=1/2, 49*S1+13*S2+w0=1/2, 1441*S1+121*S2+w0=1/2
///     (the last two are the r=2, r=4 equations cleared of their common
///     denominators 12 and 80) => S1=0, S2=0, w0=1/2.
///   So w_{-2}=(S1-D1)/2=-3/256, w_2=(S1+D1)/2=3/256,
///      w_{-1}=(S2-D2)/2=11/128, w_1=(S2+D2)/2=-11/128, w0=1/2, on columns
///      2,3,4,5,6.
///
/// SLAB j=3, [3,4)/9: the WHOLE of step 3 -- weight 1 on column 3, 0
/// elsewhere, by the lemma.
///
/// w_scale = dt_src/dt_tgt = (1/9)/(1/6) = 2/3. Summing:
///     col2: (2/3)(-3/256)            = -1/128
///     col3: (2/3)(11/128) + (2/3)(1) = 139/192
///     col4: (2/3)(1/2)               = 1/3
///     col5: (2/3)(-11/128)           = -11/192
///     col6: (2/3)(3/256)             = 1/128
/// Row sum, over /384: (-3+278+128-22+3)/384 = 384/384 = 1.
TEST(SpacetimeMatrix, DegreeFourMatchesTheHandComputation) {
  const auto slabs =
      soma::decompose(Rational{1, 9}, Rational{1, 6}, Rational{1});
  ASSERT_TRUE(slabs.ok()) << slabs.error().message;
  const auto p4 = soma::build_temporal_projection(
      slabs.value(), TemporalBasis::DG, 4, Rational{1, 9}, Rational{1, 6});
  ASSERT_TRUE(p4.ok()) << p4.error().message;
  EXPECT_NEAR(p4.value().coefficient(2, 2), -1.0 / 128.0, 1e-13);
  EXPECT_NEAR(p4.value().coefficient(2, 3), 139.0 / 192.0, 1e-13);
  EXPECT_NEAR(p4.value().coefficient(2, 4), 1.0 / 3.0, 1e-13);
  EXPECT_NEAR(p4.value().coefficient(2, 5), -11.0 / 192.0, 1e-13);
  EXPECT_NEAR(p4.value().coefficient(2, 6), 1.0 / 128.0, 1e-13);
}

/// STENCIL PLACEMENT, swept -- complementary to the three hand-derived
/// matrices above, which each pin one schedule. For a NESTING refining
/// schedule (n_tgt = 2 * n_src: every source step splits into exactly two
/// target steps), a single source step feeds each target row and neither
/// half is the whole cell, so the union of the nonzero source columns across
/// BOTH halves of source step j is exactly the documented stencil [start,
/// start+p], where `start` is `documented_stencil_start` above -- recomputed
/// from the header's prose, not read out of the library.
///
/// Why the UNION of both halves, and not either half alone: a moment-matching
/// weight can be exactly zero for a special sub-interval (degree 1's
/// middle-third split always zeroes one endpoint, for instance), so a single
/// half's own span is not by itself a safe invariant. The union is: verified
/// by exhaustive rational search over p=1..4, n_src=p+1..13, every source
/// step (344 combinations total), 0 had a union narrower than the full
/// stencil. A wrong `start` -- the reviewer's mutation, or a
/// mirrored/future-leaning stencil for odd p -- moves the whole union and is
/// caught here regardless of which hand-picked schedule the entry-level tests
/// above happen to use.
TEST(SpacetimeMatrix, StencilPlacementMatchesTheDocumentedFormulaAcrossASweep) {
  int checked = 0;
  for (int p = 1; p <= soma::kMaxTemporalDegree; ++p) {
    for (std::int64_t n_src = p + 1; n_src <= 13; ++n_src) {
      const std::int64_t n_tgt = 2 * n_src;  // nesting: each source step -> two target steps
      const auto         slabs =
          soma::decompose(Rational{1, n_src}, Rational{1, n_tgt}, Rational{1});
      ASSERT_TRUE(slabs.ok()) << slabs.error().message;
      const auto proj = soma::build_temporal_projection(
          slabs.value(), TemporalBasis::DG, p, Rational{1, n_src},
          Rational{1, n_tgt});
      ASSERT_TRUE(proj.ok()) << proj.error().message;

      for (std::int64_t j = 0; j < n_src; ++j) {
        const std::int64_t expected_start = documented_stencil_start(j, p, n_src);
        std::int64_t        observed_lo = -1, observed_hi = -1;
        for (std::int64_t sub = 0; sub < 2; ++sub) {
          const std::int64_t i = 2 * j + sub;  // the two target rows fed by step j
          for (std::int64_t c = 0; c < n_src; ++c) {
            if (proj.value().coefficient(static_cast<std::size_t>(i),
                                         static_cast<std::size_t>(c)) != 0.0) {
              if (observed_lo < 0 || c < observed_lo) observed_lo = c;
              if (c > observed_hi) observed_hi = c;
            }
          }
        }
        ASSERT_GE(observed_lo, 0) << "p=" << p << " n_src=" << n_src
                                  << " j=" << j << ": row is entirely zero";
        EXPECT_EQ(observed_lo, expected_start)
            << "p=" << p << " n_src=" << n_src << " j=" << j
            << ": stencil starts in the wrong place";
        EXPECT_EQ(observed_hi, expected_start + p)
            << "p=" << p << " n_src=" << n_src << " j=" << j
            << ": stencil has the wrong width or end";
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 300);
  std::cout << "[ STENCIL ] placement verified on " << checked
            << " (degree, source-step-count, source step) combinations\n";
}

TEST(SpacetimeRuntime, DegreeZeroProjectsTheStepMeansAndNotTheLatestValue) {
  auto h = build(Opts{.basis = TemporalBasis::Constant, .degree = 0});
  const double got = project_last_target_step(*h);

  EXPECT_NEAR(got, 8.0 / 3.0, 1e-12);

  // The placeholder this task replaced copied the source arc, which at t = 1
  // holds source step 2's value, 3. Spelled out so that a regression to it is
  // a named failure rather than a mystery number.
  EXPECT_GT(std::abs(got - 3.0), 0.3) << "target step 1 got the copy";
}

TEST(SpacetimeRuntime, DegreeOneIsADifferentOperatorFromDegreeZero) {
  auto h = build(Opts{.basis = TemporalBasis::DG, .degree = 1});
  const double got = project_last_target_step(*h);

  EXPECT_NEAR(got, 11.0 / 4.0, 1e-12);

  // Neither the copy nor the degree-0 answer. The second is the assertion that
  // fails if the runtime silently downgrades a basis it cannot honour to
  // piecewise constant.
  EXPECT_GT(std::abs(got - 3.0), 0.2) << "got the copy";
  EXPECT_GT(std::abs(got - 8.0 / 3.0), 1e-3) << "got the degree-0 operator";
}

/// `Linear` and `DG(1)` are documented to build the same matrix
/// (spacetime.hpp), and that has to survive the wiring: the runtime must not
/// pick a different code path for the two spellings.
TEST(SpacetimeRuntime, LinearAndDgOneAgree) {
  auto a = build(Opts{.basis = TemporalBasis::Linear, .degree = 1});
  auto b = build(Opts{.basis = TemporalBasis::DG, .degree = 1});
  EXPECT_EQ(project_last_target_step(*a), project_last_target_step(*b));
}

/// Constant preservation, spec §7.3 property 1, measured through the whole
/// wiring rather than on the matrix: a source that never changes must arrive
/// unchanged. This is the check that a mis-scaled operator (every weight
/// multiplied by src_dt, say) fails while the two tests above might not, since
/// they compare against hand numbers on one schedule only.
TEST(SpacetimeRuntime, AConstantInTimeSourceArrivesUnchanged) {
  // Run three periods; from the second onwards the ramp has moved on, so
  // instead hold the source fixed by reading the projection of a period in
  // which every source step carried the same value. The simplest such source
  // is the second half of a two-step run with equal values, so use a component
  // that stops changing: assert on the row sums the runtime actually applied
  // by feeding a constant through a fresh harness whose Ramp is frozen.
  class Frozen final : public soma::Component {
   public:
    [[nodiscard]] std::string name() const override { return "src"; }
    soma::Result<void> register_fields(FieldRegistry& reg,
                                       const CouplingGraph& g) override {
      const auto cid = g.find_component("src");
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
      return soma::Result<void>::success();
    }

   private:
    std::vector<double> data_ = std::vector<double>(kCells, 7.25);
  };

  for (const int degree : {0, 1}) {
    auto h = build(Opts{.basis = degree == 0 ? TemporalBasis::Constant
                                             : TemporalBasis::DG,
                        .degree = degree});
    // Swap the ramp for a constant source.
    h->components = ComponentSet{};
    h->components.add(std::make_unique<Frozen>());
    h->components.add(std::make_unique<Sink>());
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    bind_all(*h);
    ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
    ASSERT_TRUE(runner.run_period(h->components).ok());
    const auto buf = runner.buffer(h->arc_of("link/spacetime"));
    ASSERT_TRUE(buf.ok());
    for (std::size_t i = 0; i < kCells; ++i) {
      EXPECT_NEAR(buf.value()[i], 7.25, 1e-13) << "degree " << degree;
    }
  }
}

/// The delivered value reaches the consumer's registered array, not just a
/// coupler intermediate: the whole point of wiring is that something downstream
/// changes.
TEST(SpacetimeRuntime, TheProjectionReachesTheConsumersArray) {
  auto h = build(Opts{.basis = TemporalBasis::Constant, .degree = 0});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  for (std::size_t i = 0; i < kCells; ++i) {
    EXPECT_NEAR(h->sink()->data()[i], 8.0 / 3.0, 1e-12);
  }
}

TEST(SpacetimeRuntime, AllocatesNothingWhileRunning) {
  auto h = build(Opts{.basis = TemporalBasis::DG, .degree = 1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());  // warm any lazy state

  AllocationCounter counter;
  for (int i = 0; i < 4; ++i) {
    const auto r = runner.run_period(h->components);
    if (!r.ok()) {
      g_counting = false;
      FAIL() << r.error().message;
    }
  }
  EXPECT_EQ(counter.count(), 0u);
}

// ---------------------------------------------------------------------------
// 3. What is refused, and why
// ---------------------------------------------------------------------------

/// Spec §7.3: a conservative Spline projection is a globally coupled
/// reconstruction that spacetime.cpp does not implement. It must be an error,
/// not a quiet substitution of the local cubic.
TEST(SpacetimeRefusal, SplineIsRejected) {
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Spline, .degree = 3}),
            "E-ST-BASIS");
}

TEST(SpacetimeRefusal, ADegreeTheBasisDoesNotAdmitIsRejected) {
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Constant, .degree = 1}),
            "E-ST-DEGREE");
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Linear, .degree = 2}),
            "E-ST-DEGREE");
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::DG, .degree = 5}),
            "E-ST-DEGREE");
}

/// The stencil of a degree-2 reconstruction reaches one source step FORWARD
/// (see `stencil_start` in src/remap/spacetime.cpp), so writing a target step
/// would need source data the source component has not produced when that
/// target step ends. Refused rather than served from the previous period's
/// history, which would be a one-window-lagged operator wearing this one's
/// name.
TEST(SpacetimeRefusal, ANonCausalStencilIsRejectedRatherThanLagged) {
  // 3 source steps into 2 target steps: the degree-2 stencil is the whole
  // window, so closing target step 0 (which ends inside source step 1) needs
  // source step 2.
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::DG, .degree = 2}),
            "E-RT-SPACETIME");
  // Degree 1 in the REFINING direction, 2 source steps into 3 target steps:
  // the line through source steps {0, 1} is not available until step 1 is
  // done, but target step 0 closes at t = 1/3, inside source step 0.
  EXPECT_EQ(setup_error(Opts{.src_dt = Rational{1, 2},
                             .dst_dt = Rational{1, 3},
                             .basis = TemporalBasis::DG,
                             .degree = 1}),
            "E-RT-SPACETIME");
}

/// R11 (Gate 2 remediation follow-up). The test above proves the runtime
/// rejects schedules that are non-causal BY CONSTRUCTION -- degree/direction
/// combinations where even a correctly-placed stencil has to reach forward.
/// It does not, by itself, prove that this runtime check (as opposed to the
/// golden hand-computed matrices in `TheMatrixMatchesTheHandComputation` and
/// friends, added for the same Gate 2 finding) is what would catch a
/// PLACEMENT bug in `stencil_start` -- one that builds a structurally
/// plausible, still-causal-looking matrix whose nonzero columns are simply in
/// the wrong place. Gate 2's own mutation (`start = j - p` instead of
/// `start = j - (p + 1) / 2`, see the comment above
/// `TheMatrixMatchesTheHandComputation`) shifts the stencil FURTHER into the
/// past, so it makes the causality check's job easier and cannot be relied on
/// to exercise it; that mutation is pinned by the golden matrices alone, as
/// documented there.
///
/// This test pins a DIFFERENT, forward-leaning placement bug against THIS
/// file's runtime path specifically: the true "mirror" of
/// `stencil_start`'s documented odd-degree tie-break ("one step towards the
/// past for odd p"). Swapping that tie-break -- using
/// `start = j - (p + 1) / 2 + (p % 2)` instead of `start = j - (p + 1) / 2`,
/// which leaves even degrees untouched (they have no tie to break) and moves
/// every ODD-degree stencil one step into the FUTURE instead -- still
/// produces a matrix satisfying both structural conservation identities
/// (partition of unity, integral conservation), so it is invisible to any
/// test built on those, and it still passes every golden hand-computation
/// test at even degrees. It was applied, by hand, directly to `stencil_start`
/// in src/remap/spacetime.cpp and the build rerun as part of this task's
/// verification (not committed; reverted immediately after). Measured
/// effect, `spacetime_runtime_test` under that mutation:
///
///   * `SpacetimeRefusal.*` (this suite, all six OTHER cases): all still
///     pass. None of the existing setup_error() schedules distinguishes the
///     two tie-breaks -- exactly the gap R11 flags.
///   * `SpacetimeRuntime.DegreeOneIsADifferentOperatorFromDegreeZero` --
///     which uses the schedule pinned below -- FAILED, and not on its numeric
///     assertion: `runner.initialize()` itself returned
///     `E-RT-SPACETIME: the DG(1) temporal projection needs source step 2 to
///     write target step 0, but only source steps up to 1 have been produced
///     when that target step ends`, from the exact CAUSALITY block this file
///     is about (src/runtime/runner.cpp, `check_spacetime_schedule`).
///
/// That is the missing proof: for THIS schedule, a mirrored placement is
/// forward-reaching, and it is the runtime check above -- not a golden
/// matrix, which this file's degree-1 tests do not even build directly --
/// that would refuse it. The assertion below is on the CORRECT, checked-in
/// code and is therefore an acceptance, not a rejection; the rejection under
/// the mutation is the evidence, recorded here so a future silent regression
/// of the tie-break direction has a numbered schedule to be caught on rather
/// than relying on this comment being read.
TEST(SpacetimeRefusal, TheDefaultCoarseningScheduleIsAcceptedAndWouldTripOnAMirroredLean) {
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::DG, .degree = 1}), "");
}

/// The causality rule is on the operator's actual non-zeros, not a blanket cap
/// on the degree, and this is where the difference shows. With NESTING
/// partitions every slab is a whole source step, so the reconstruction
/// integrates to the cell mean and the forward stencil entries come out
/// exactly zero and are dropped in canonicalisation. DG(2) over 6 source steps
/// into 2 target steps is therefore accepted -- and is the same matrix DG(0)
/// would build, which is not a downgrade but the correct answer for a partition
/// on which the degree cannot matter.
TEST(SpacetimeRefusal, ANestingPartitionAdmitsAHigherDegree) {
  EXPECT_EQ(setup_error(Opts{.src_dt = Rational{1, 6},
                             .dst_dt = Rational{1, 2},
                             .basis = TemporalBasis::DG,
                             .degree = 2}),
            "");
}

/// Degree 0 has no stencil, so it is causal in either direction, including the
/// refining one where it happens to agree with a copy. Kept as the boundary of
/// the rule above.
TEST(SpacetimeRefusal, DegreeZeroIsAcceptedInBothDirections) {
  EXPECT_EQ(setup_error(Opts{.src_dt = Rational{1, 2},
                             .dst_dt = Rational{1, 3},
                             .basis = TemporalBasis::Constant,
                             .degree = 0}),
            "");
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Constant, .degree = 0}),
            "");
}

/// Accumulation and a space-time projection are two temporal operators on the
/// same data. Letting one win silently is the exact failure mode spec §7.3
/// warns about, so both spelled together is refused.
TEST(SpacetimeRefusal, AccumulationCombinedWithSpacetimeIsRejected) {
  // Refused by expand(): each spelling needs its own Sample on the source
  // clock and a channel cannot have two. Without the explicit check it comes
  // out as E-SCHED-ORDER, whose message blames the scheduler.
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Constant,
                             .degree = 0,
                             .accum = soma::AccumOp::Mean}),
            "E-SCHED-SPACETIME");
}

/// Only a window equal to the plan's common period is implemented. A shorter
/// one is a different operator, and widening it silently would project over
/// data the user did not ask to include.
TEST(SpacetimeRefusal, AWindowOtherThanTheCommonPeriodIsRejected) {
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Constant,
                             .degree = 0,
                             .window = soma::Window::explicit_length(
                                 Rational{2, 5})}),
            "E-RT-SPACETIME");
  // Spelling the period a different way is accepted, because it IS the period.
  EXPECT_EQ(setup_error(Opts{.basis = TemporalBasis::Constant,
                             .degree = 0,
                             .window =
                                 soma::Window::explicit_length(Rational{1})}),
            "");
}
