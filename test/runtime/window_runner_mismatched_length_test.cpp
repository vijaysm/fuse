// Mismatched window-member field lengths: a window member's cold-start
// import priming (Runner::Impl::resolve_regions(), src/runtime/iteration.cpp)
// must bound the loop reading the PROVIDER's own exported-field buffer
// (`provider_v0`, resolved separately from `provider_exported_field`) by the
// provider's own field length, not by THIS member's own exported-field
// length (`cfg.length`). Using the wrong bound is a heap-buffer-overflow
// READ whenever the provider's own field is SMALLER, which ASan reports on
// a cross-resolution two-member window. Same-resolution pairs never expose
// it, so it is invisible to the ordinary (non-ASan) suite unless a window
// test combines mismatched field lengths with a real cross-mesh remap.
//
// This graph: A exports 4 values (mesh "a", generated:quad:2x2), B
// exports 9 (mesh "b", generated:quad:3x3). B's own import comes from A
// (the smaller provider), which is the overflow direction; A's own import
// comes from B (the larger provider), which is always safe and is
// exercised here for completeness.

#include <gtest/gtest.h>
#include <fuse/backend/native_backend.hpp>

#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using fuse::ComponentDecl;
using fuse::ComponentSet;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::CouplingWindowScope;
using fuse::DtPolicy;
using fuse::FieldRef;
using fuse::FieldRegistry;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::IterationRegionDecl;
using fuse::Layout;
using fuse::MemorySpace;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RankSet;
using fuse::Runner;
using fuse::WaveformIteration;

namespace u = fuse::units;

namespace {

/// `n`-element analogue of window_runner_test.cpp's own scalar
/// `LinearBlock`: `out[i] = a + k*in[i]`, elementwise, real
/// save_state()/restore_state() (needed: a window member's own
/// checkpoint/rollback is exercised every sweep, not optional).
class VectorBlock final : public fuse::Component {
 public:
  VectorBlock(std::string name, int n, double a, double k)
      : name_(std::move(name)), out_(n, 0.0), in_(n, 0.0), a_(a), k_(k) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      double* p = is_out ? out_.data() : in_.data();
      auto r = reg.bind(f, FieldRef{p, Layout{.stride = 1, .count = out_.size()},
                                    MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& /*ctx*/) override {
    ++advance_count;
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = a_ + k_ * in_[i];
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    out_ = saved_out_;
    in_ = saved_in_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }
  [[nodiscard]] const std::vector<double>& out() const { return out_; }

  int advance_count = 0;

 private:
  std::string name_;
  std::vector<double> out_, in_, saved_out_, saved_in_;
  double a_, k_;
};

CouplingGraph make_mismatched_length_window_graph() {
  CouplingGraph g;
  // 2x2 (4 cells) and 3x3 (9 cells) -- mutually non-nested
  // ("A exports 4 values, B exports 9").
  const auto ma = g.add_mesh(MeshDecl{.name = "a", .source = "generated:quad:2x2"});
  const auto mb = g.add_mesh(MeshDecl{.name = "b", .source = "generated:quad:3x3"});
  const auto ra = g.add_region(Region{.name = "ra", .mesh = ma, .dimension = 2});
  const auto rb = g.add_region(Region{.name = "rb", .mesh = mb, .dimension = 2});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, FieldSpec{.name = "a_out", .region = ra,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{.name = "b_in", .region = rb,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});
  const auto b_out = g.add_field(b, FieldSpec{.name = "b_out", .region = rb,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto a_in = g.add_field(a, FieldSpec{.name = "a_in", .region = ra,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});

  // a_out (4) -> b_in (9): B's own cfg.length (9, its own export) reads
  // the PROVIDER's export (A's a_out, 4) during cold-start priming --
  // the overflow direction. b_out (9) -> a_in (4): the reverse, always
  // safe (cfg.length 4 < provider's 9), exercised for completeness.
  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep, not a real-time lag");

  CouplingWindowScope scope;
  scope.window_component = "A";
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Adjustable}};
  scope.iteration = WaveformIteration::GaussSeidel;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-9, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-9, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

}  // namespace

// The regression test itself: this must not crash (the overflow, when
// present, reads past a small heap allocation -- whether that crashes,
// corrupts an unrelated value, or silently "succeeds" depends on heap
// layout the ordinary build does not control) and initialize()/one
// period must both succeed cleanly on a REAL cross-mesh remap (native
// backend, not the identity-only SerialMeshBackend). See
// MismatchedFieldLengthsWindow.ASanFindsNoOverflow (built separately in
// a sanitizer tree, not part of the ordinary suite) for the actual
// memory-safety oracle this one cannot be on its own.
TEST(WindowRunnerMismatchedLength, InitializeAndOnePeriodSucceedWithoutOverflow) {
  const CouplingGraph g = make_mismatched_length_window_graph();

  const auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0}), g).ok());
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{1}), g).ok());

  const fuse::ClockSet clocks{g};
  const auto sched =
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto comp_a = std::make_unique<VectorBlock>("A", 4, /*a=*/1.0, /*k=*/0.5);
  auto comp_b = std::make_unique<VectorBlock>("B", 9, /*a=*/0.0, /*k=*/1.0);
  VectorBlock* pa = comp_a.get();
  VectorBlock* pb = comp_b.get();
  ComponentSet components;
  components.add(std::move(comp_a));
  components.add(std::move(comp_b));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  // The crash this test exists to catch would have happened inside this
  // one call (resolve_regions(), called from initialize()) -- reaching
  // this ASSERT at all, under any build, is already most of the proof;
  // ASan (the separate sanitizer-tree run) is what proves it did so
  // without ever reading past the end of A's own 4-element buffer.
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const auto ran = runner.run_period(components);
  ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message;
  EXPECT_GT(pa->advance_count, 0);
  EXPECT_GT(pb->advance_count, 0);
  for (const double v : pa->out()) EXPECT_TRUE(std::isfinite(v));
  for (const double v : pb->out()) EXPECT_TRUE(std::isfinite(v));
}
