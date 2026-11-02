// A field of forward-mode dual numbers through the Runner and a real remap
// (plans/2026-10-22-universal-number-types.md, stage U2): the gradient that
// arrives equals the remap of the gradient, bitwise, because a field of dual
// numbers reaches every operation as a field with value and gradient as
// components. Also the two refusals that keep that honest: a minimum or maximum
// accumulation of a field with a gradient, and a bound field whose type
// the runtime cannot move.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/num/dual.hpp>
#include <fuse/runtime/runner.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::SerialMeshBackend;
using fuse::StepContext;
namespace u = fuse::units;

/// A Storage value type the runtime has no conversion for: one byte, named so
/// that the refusal can be recognised.
struct Foreign {
  std::uint8_t v;
};

template <>
struct fuse::value_traits<Foreign> {
  static constexpr std::string_view name = "myapp:byte";
  static constexpr fuse::FieldValueKind kind = fuse::FieldValueKind::Storage;
  using scalar_type = Foreign;
  static constexpr int scalar_count = 1;
  static constexpr int gradient_size = 0;
  static double to_binary64(Foreign f) noexcept { return f.v; }
  static Foreign from_binary64(double x) noexcept { return Foreign{static_cast<std::uint8_t>(x)}; }
};

namespace {

using D1 = fuse::num::dual<double, 1>;

/// How a component's array is registered. Dual and Flat2 are the same memory
/// (value then gradient, per cell) registered two ways; Plain is one scalar per
/// cell, and which of the two scalars it carries is chosen by the test.
enum class Bind { Dual, Flat2, Plain, Foreign };

int scalars_per_cell(Bind b) { return (b == Bind::Dual || b == Bind::Flat2) ? 2 : 1; }

/// The value the producer writes for scalar `which` of cell `i` at `step`. The
/// two scalars differ in shape, so a gradient that was the value, or a value that
/// was the gradient, cannot pass unnoticed.
double pattern(int which, std::size_t i, int step) {
  const double x = static_cast<double>(i);
  return which == 0 ? 2.0 * step + x + 0.125 * x * x
                   : 0.5 * x - 3.0 + 0.25 * step + 0.01 * x * x * x;
}

class DualProducer final : public fuse::Component {
 public:
  DualProducer(std::size_t cells, Bind bind, int plain_scalar)
      : cells_(cells),
        bind_(bind),
        plain_scalar_(plain_scalar),
        raw_(cells * static_cast<std::size_t>(scalars_per_cell(bind)), 0.0),
        foreign_(bind == Bind::Foreign ? cells : 0) {}

  [[nodiscard]] std::string name() const override { return "producer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      fuse::Result<void> r = fuse::Result<void>::success();
      switch (bind_) {
        case Bind::Dual:
          r = reg.bind(f, fuse::BasicFieldRef<D1>(
                              reinterpret_cast<D1*>(raw_.data()),
                              fuse::Layout{1, 1, cells_, 1},
                              fuse::MemorySpace::Host));
          break;
        case Bind::Flat2:
          r = reg.bind(f, fuse::FieldRef(raw_.data(),
                                         fuse::Layout{2, 1, cells_, 2},
                                         fuse::MemorySpace::Host));
          break;
        case Bind::Plain:
          r = reg.bind(f, fuse::FieldRef(raw_.data(),
                                         fuse::Layout{1, 0, cells_, 1},
                                         fuse::MemorySpace::Host));
          break;
        case Bind::Foreign:
          r = reg.bind(f, fuse::BasicFieldRef<Foreign>(
                              foreign_.data(), fuse::Layout{1, 0, cells_, 1},
                              fuse::MemorySpace::Host));
          break;
      }
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    const int s = scalars_per_cell(bind_);
    for (std::size_t i = 0; i < cells_; ++i) {
      for (int l = 0; l < s; ++l) {
        const int which = bind_ == Bind::Plain ? plain_scalar_ : l;
        if (bind_ == Bind::Foreign) {
          foreign_[i] = Foreign{static_cast<std::uint8_t>(i)};
        } else {
          raw_[i * static_cast<std::size_t>(s) + static_cast<std::size_t>(l)] =
              pattern(which, i, steps_);
        }
      }
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& raw() const { return raw_; }

 private:
  std::size_t cells_;
  Bind bind_;
  int plain_scalar_;
  int steps_ = 0;
  std::vector<double> raw_;
  std::vector<Foreign> foreign_;
};

class DualConsumer final : public fuse::Component {
 public:
  DualConsumer(std::size_t cells, Bind bind)
      : cells_(cells),
        bind_(bind),
        raw_(cells * static_cast<std::size_t>(scalars_per_cell(bind)), 0.0) {}

  [[nodiscard]] std::string name() const override { return "consumer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      fuse::Result<void> r = fuse::Result<void>::success();
      if (bind_ == Bind::Dual) {
        r = reg.bind(f, fuse::BasicFieldRef<D1>(
                            reinterpret_cast<D1*>(raw_.data()),
                            fuse::Layout{1, 1, cells_, 1},
                            fuse::MemorySpace::Host));
      } else if (bind_ == Bind::Flat2) {
        r = reg.bind(f, fuse::FieldRef(raw_.data(),
                                       fuse::Layout{2, 1, cells_, 2},
                                       fuse::MemorySpace::Host));
      } else {
        r = reg.bind(f, fuse::FieldRef(raw_.data(),
                                       fuse::Layout{1, 0, cells_, 1},
                                       fuse::MemorySpace::Host));
      }
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& raw() const { return raw_; }

 private:
  std::size_t cells_;
  Bind bind_;
  std::vector<double> raw_;
};

struct Rig {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  fuse::OperatorDAG   dag;
  fuse::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;
  DualProducer*       producer = nullptr;
  DualConsumer*       consumer = nullptr;
};

/// Producer on `src_cells` cells, consumer on `dst_cells`, joined by one
/// channel and the backend's remap between two line meshes of different
/// resolution, so the remap is not the identity.
std::unique_ptr<Rig> build(Bind producer_bind, Bind consumer_bind, int plain_scalar,
                           fuse::AccumOp accum = fuse::AccumOp::Instant,
                           std::size_t src_cells = 8, std::size_t dst_cells = 5) {
  auto h = std::make_unique<Rig>();
  auto& g = h->graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{
      .name = "m1", .source = "generated:line:" + std::to_string(src_cells)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{
      .name = "m2", .source = "generated:line:" + std::to_string(dst_cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "producer", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  // The graph counts scalars per cell: a value and its gradient are two.
  const int comps = scalars_per_cell(producer_bind);
  // An Integral channel delivers [source] x [time]: the target is declared so.
  const bool integral = accum == fuse::AccumOp::Integral;
  const auto s = g.add_field(
      p, fuse::FieldSpec{.name = "out", .region = r1, .components = comps,
                         .units = integral ? u::kWatt : u::kKelvin, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(
      c, fuse::FieldSpec{.name = "in", .region = r2, .components = comps,
                         .units = integral ? u::kJoule : u::kKelvin, .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(accum);
  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, h->dag, clocks,
                        fuse::make_region_sizes(g, h->dag, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  auto prod = std::make_unique<DualProducer>(src_cells, producer_bind, plain_scalar);
  auto cons = std::make_unique<DualConsumer>(dst_cells, consumer_bind);
  h->producer = prod.get();
  h->consumer = cons.get();
  h->components.add(std::move(prod));
  h->components.add(std::move(cons));
  return h;
}

fuse::Result<void> bind_and_initialize(Rig& h, Runner& runner) {
  for (auto& c : h.components) {
    auto r = c.register_fields(h.registry, h.graph);
    if (!r.ok()) return r;
  }
  return runner.initialize(h.components, h.registry);
}

std::vector<double> run_one_period(Bind pb, Bind cb, int plain_scalar) {
  auto h = build(pb, cb, plain_scalar);
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto init = bind_and_initialize(*h, runner);
  EXPECT_TRUE(init.ok()) << (init.ok() ? "" : init.error().message);
  EXPECT_TRUE(runner.run_period(h->components).ok());
  return h->consumer->raw();
}

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

TEST(DualFieldRunner, TheGradientThatArrivesIsTheRemapOfTheGradient) {
  const std::size_t dst = 5;
  const auto dual = run_one_period(Bind::Dual, Bind::Dual, 0);
  const auto value_only = run_one_period(Bind::Plain, Bind::Plain, 0);
  const auto gradient_only = run_one_period(Bind::Plain, Bind::Plain, 1);
  ASSERT_EQ(dual.size(), 2 * dst);
  ASSERT_EQ(value_only.size(), dst);
  ASSERT_EQ(gradient_only.size(), dst);

  bool value_moved = false, gradient_moved = false;
  for (std::size_t j = 0; j < dst; ++j) {
    EXPECT_EQ(bits(dual[2 * j]), bits(value_only[j])) << "value, cell " << j;
    EXPECT_EQ(bits(dual[2 * j + 1]), bits(gradient_only[j])) << "gradient, cell " << j;
    value_moved = value_moved || value_only[j] != 0.0;
    gradient_moved = gradient_moved || gradient_only[j] != 0.0;
  }
  // Not vacuous: both scalars arrived, and they are different fields.
  EXPECT_TRUE(value_moved);
  EXPECT_TRUE(gradient_moved);
  EXPECT_NE(bits(dual[0]), bits(dual[1]));
}

TEST(DualFieldRunner, ATypedDualFieldAndAPlainTwoComponentFieldMoveIdentically) {
  const auto dual = run_one_period(Bind::Dual, Bind::Dual, 0);
  const auto flat = run_one_period(Bind::Flat2, Bind::Flat2, 0);
  const auto mixed = run_one_period(Bind::Dual, Bind::Flat2, 0);
  ASSERT_EQ(dual.size(), flat.size());
  ASSERT_EQ(dual.size(), mixed.size());
  for (std::size_t k = 0; k < dual.size(); ++k) {
    EXPECT_EQ(bits(dual[k]), bits(flat[k])) << k;
    EXPECT_EQ(bits(dual[k]), bits(mixed[k])) << k;
  }
}

TEST(DualFieldRunner, MinimumAndMaximumAreRefusedOnlyForAFieldWithAGradient) {
  for (const auto op : {fuse::AccumOp::Min, fuse::AccumOp::Max}) {
    auto h = build(Bind::Dual, Bind::Dual, 0, op);
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    const auto r = bind_and_initialize(*h, runner);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-RT-GRADIENT-MINMAX");
    EXPECT_NE(r.error().message.find("link"), std::string::npos);
    EXPECT_NE(r.error().message.find("carries a gradient"), std::string::npos);
  }
  // The linear accumulations are fine on the same field...
  for (const auto op : {fuse::AccumOp::Sum, fuse::AccumOp::Mean,
                        fuse::AccumOp::Integral}) {
    auto h = build(Bind::Dual, Bind::Dual, 0, op);
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    const auto r = bind_and_initialize(*h, runner);
    EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  }
  // ...and a minimum of an ordinary one-scalar field is not touched.
  auto plain = build(Bind::Plain, Bind::Plain, 0, fuse::AccumOp::Min);
  Runner runner{plain->graph, plain->dag, plain->plan, plain->backend};
  const auto r = bind_and_initialize(*plain, runner);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

TEST(DualFieldRunner, ABoundFieldOfATypeTheRuntimeCannotMoveIsNotCalledUnbound) {
  auto h = build(Bind::Foreign, Bind::Plain, 0);
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-FIELD-TYPE");
  EXPECT_NE(r.error().message.find("myapp:byte"), std::string::npos);
}

}  // namespace
