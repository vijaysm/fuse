#include <gtest/gtest.h>
#include <soma/runtime/component.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using soma::Component;
using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::StepContext;

namespace {

class Toy final : public Component {
 public:
  explicit Toy(std::string n, std::size_t size = 4)
      : name_(std::move(n)), data_(size, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return soma::Error{"E-TEST", "component not in graph"};
    }
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext& ctx) override {
    ++advance_count;
    last = ctx;
    for (double& d : data_) d += 1.0;
    return soma::Result<void>::success();
  }

  soma::Result<void> save_state() override {
    saved_ = data_;
    ++save_count;
    return soma::Result<void>::success();
  }

  soma::Result<void> restore_state() override {
    if (saved_.size() != data_.size()) {
      return soma::Error{"E-TEST-CKPT", "restore_state before save_state"};
    }
    // Element-wise, deliberately not `data_ = saved_`: vector copy assignment
    // is permitted to reallocate, and data_.data() is the pointer every bound
    // FieldRef holds. Restoring a checkpoint must not move the storage.
    std::copy(saved_.begin(), saved_.end(), data_.begin());
    ++restore_count;
    return soma::Result<void>::success();
  }

  [[nodiscard]] bool supports_rollback() const override { return true; }

  int         advance_count = 0;
  int         save_count = 0;
  int         restore_count = 0;
  StepContext last{};

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  [[nodiscard]] const double* raw() const { return data_.data(); }

 private:
  std::string         name_;
  std::vector<double> data_;
  std::vector<double> saved_;
};

/// A component that overrides only what the interface makes pure virtual.
class Bare final : public Component {
 public:
  explicit Bare(std::string n = "bare") : name_(std::move(n)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  soma::Result<void> register_fields(FieldRegistry&,
                                     const CouplingGraph&) override {
    return soma::Result<void>::success();
  }
  soma::Result<void> advance(const StepContext&) override {
    return soma::Result<void>::success();
  }

 private:
  std::string name_;
};

/// One component named "alpha" owning one scalar field "f".
CouplingGraph make_graph(const std::string& component_name = "alpha") {
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto cid = g.add_component(soma::ComponentDecl{.name = component_name});
  g.add_field(cid, soma::FieldSpec{.name = "f", .region = r,
                                   .intent = soma::Intent::Provides});
  return g;
}

}  // namespace

// ---------------------------------------------------------------- StepContext

TEST(StepContextTest, DefaultIsAnEmptyIntervalAtTimeZero) {
  const StepContext ctx{};
  EXPECT_EQ(ctx.t_begin, Rational(0));
  EXPECT_EQ(ctx.t_end, Rational(0));
  EXPECT_EQ(ctx.step, 0);
  EXPECT_EQ(ctx.iteration, 0);
  EXPECT_EQ(ctx.dt(), Rational(0));
  EXPECT_TRUE(ctx.valid());
}

TEST(StepContextTest, DtIsExactForSubcyclingRatios) {
  const StepContext ctx{.t_begin = Rational{1, 3}, .t_end = Rational{1, 2}};
  EXPECT_EQ(ctx.dt(), Rational(1, 6));
}

TEST(StepContextTest, ValidRejectsInvertedIntervalsAndNegativeCounters) {
  EXPECT_FALSE((StepContext{.t_begin = Rational{1}, .t_end = Rational{0}})
                   .valid());
  EXPECT_FALSE((StepContext{.step = -1}).valid());
  EXPECT_FALSE((StepContext{.iteration = -1}).valid());
  // A zero-length step is legal: an iteration region can re-enter a component
  // on the same interval, and a component may be asked for t_begin == t_end.
  EXPECT_TRUE((StepContext{.t_begin = Rational{2}, .t_end = Rational{2}})
                  .valid());
}

// ------------------------------------------------------------------ Component

TEST(ComponentIface, AdvanceReceivesTheStepContext) {
  Toy t{"a"};
  const StepContext ctx{.t_begin = Rational{0}, .t_end = Rational{1, 2},
                        .step = 3, .iteration = 1};
  ASSERT_TRUE(t.advance(ctx).ok());
  EXPECT_EQ(t.advance_count, 1);
  EXPECT_EQ(t.last.t_end, Rational(1, 2));
  EXPECT_EQ(t.last.step, 3);
  EXPECT_EQ(t.last.iteration, 1);
}

TEST(ComponentIface, SaveAndRestoreRoundTrip) {
  Toy t{"a"};
  ASSERT_TRUE(t.advance(StepContext{}).ok());  // data becomes 1.0
  ASSERT_TRUE(t.save_state().ok());
  ASSERT_TRUE(t.advance(StepContext{}).ok());  // data becomes 2.0
  EXPECT_DOUBLE_EQ(t.data()[0], 2.0);
  ASSERT_TRUE(t.restore_state().ok());
  EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
  EXPECT_EQ(t.save_count, 1);
  EXPECT_EQ(t.restore_count, 1);
}

// A component that does not override save/restore must say so.
TEST(ComponentIface, RollbackDefaultsToUnsupported) {
  const Bare b;
  EXPECT_FALSE(b.supports_rollback());
  EXPECT_TRUE(Toy{"a"}.supports_rollback());
}

// The base-class defaults must be callable and must succeed, so that a driver
// can invoke them unconditionally on a component outside an iteration region.
TEST(ComponentIface, DefaultSaveAndRestoreAreSuccessfulNoOps) {
  Bare b;
  EXPECT_TRUE(b.save_state().ok());
  EXPECT_TRUE(b.restore_state().ok());
}

TEST(ComponentIface, IsDestroyedThroughTheBasePointer) {
  // The virtual destructor is what makes ComponentSet's unique_ptr<Component>
  // storage legal; without it every derived member leaks.
  struct Tracker final : Component {
    explicit Tracker(int& counter) : counter_(&counter) {}
    ~Tracker() override { ++*counter_; }
    [[nodiscard]] std::string name() const override { return "tracker"; }
    soma::Result<void> register_fields(FieldRegistry&,
                                       const CouplingGraph&) override {
      return soma::Result<void>::success();
    }
    soma::Result<void> advance(const StepContext&) override {
      return soma::Result<void>::success();
    }
    int* counter_;
  };
  int destroyed = 0;
  {
    const std::unique_ptr<Component> c = std::make_unique<Tracker>(destroyed);
  }
  EXPECT_EQ(destroyed, 1);
}

// --------------------------------------------------------------- ComponentSet

TEST(ComponentSet, AddsAndFindsByName) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  cs.add(std::make_unique<Toy>("beta"));
  EXPECT_EQ(cs.size(), 2u);
  ASSERT_NE(cs.find("beta"), nullptr);
  EXPECT_EQ(cs.find("beta")->name(), "beta");
  EXPECT_EQ(cs.find("missing"), nullptr);
}

TEST(ComponentSet, FindIsAvailableOnAConstSet) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  const ComponentSet& c = cs;
  ASSERT_NE(c.find("alpha"), nullptr);
  EXPECT_EQ(c.find("alpha")->name(), "alpha");
  EXPECT_EQ(c.find("beta"), nullptr);
}

TEST(ComponentSet, StartsEmpty) {
  const ComponentSet cs;
  EXPECT_EQ(cs.size(), 0u);
  EXPECT_TRUE(cs.empty());
  EXPECT_EQ(cs.begin(), cs.end());
}

TEST(ComponentSet, RejectsANullComponent) {
  ComponentSet cs;
  EXPECT_THROW(cs.add(nullptr), std::invalid_argument);
  EXPECT_EQ(cs.size(), 0u);
}

// An unnamed component can never be matched against a ComponentDecl, so it is
// refused at the point of insertion rather than silently never scheduled.
TEST(ComponentSet, RejectsAnEmptyName) {
  ComponentSet cs;
  EXPECT_THROW(cs.add(std::make_unique<Bare>("")), std::invalid_argument);
  EXPECT_EQ(cs.size(), 0u);
}

// find() would return the first of two same-named components and the other
// would never advance. That is wrong physics, not a lookup nuisance.
TEST(ComponentSet, RejectsADuplicateName) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  EXPECT_THROW(cs.add(std::make_unique<Toy>("alpha")), std::invalid_argument);
  EXPECT_EQ(cs.size(), 1u);
  ASSERT_NE(cs.find("alpha"), nullptr);
}

// Checked in every build configuration, Release included.
TEST(ComponentSet, AtIsBoundsChecked) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  EXPECT_EQ(cs.at(0).name(), "alpha");
  EXPECT_THROW((void)cs.at(1), std::out_of_range);
  const ComponentSet& c = cs;
  EXPECT_EQ(c.at(0).name(), "alpha");
  EXPECT_THROW((void)c.at(1), std::out_of_range);
}

TEST(ComponentSet, IteratesYieldingComponentReferences) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  cs.add(std::make_unique<Toy>("beta"));

  std::vector<std::string> seen;
  for (Component& c : cs) {
    ASSERT_TRUE(c.advance(StepContext{}).ok());
    seen.push_back(c.name());
  }
  EXPECT_EQ(seen, (std::vector<std::string>{"alpha", "beta"}));

  std::vector<std::string> seen_const;
  const ComponentSet& cref = cs;
  for (const Component& c : cref) seen_const.push_back(c.name());
  EXPECT_EQ(seen_const, (std::vector<std::string>{"alpha", "beta"}));

  // Insertion order is the iteration order and matches at().
  EXPECT_EQ(cs.at(1).name(), "beta");
}

TEST(ComponentSet, IsMovableSoItCanBeReturnedFromASetupFunction) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  Toy* raw = dynamic_cast<Toy*>(cs.find("alpha"));
  ASSERT_NE(raw, nullptr);

  const ComponentSet moved = std::move(cs);
  EXPECT_EQ(moved.size(), 1u);
  ASSERT_NE(moved.find("alpha"), nullptr);
  // Moving the set must not move the components: every FieldRef bound during
  // register_fields() points into component-owned storage.
  EXPECT_EQ(moved.find("alpha"), raw);
}

// ------------------------------------------------------- registration wiring

TEST(ComponentSet, RegistersFieldsAgainstTheGraph) {
  const CouplingGraph g = make_graph();

  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  FieldRegistry reg;
  ASSERT_TRUE(cs.at(0).register_fields(reg, g).ok());
  EXPECT_EQ(reg.bound_count(), 1u);
}

TEST(ComponentSet, RegisterFieldsErrorPropagatesWhenTheGraphDoesNotKnowIt) {
  const CouplingGraph g = make_graph("alpha");

  ComponentSet cs;
  cs.add(std::make_unique<Toy>("gamma"));
  FieldRegistry reg;
  const auto r = cs.at(0).register_fields(reg, g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TEST");
  EXPECT_EQ(reg.bound_count(), 0u);
}

// The end-to-end ownership contract: the registry holds a non-owning view into
// component storage, so a checkpoint round trip must be visible through the
// registry without any rebinding.
TEST(ComponentSet, RegistryObservesComponentStateAcrossACheckpointRoundTrip) {
  const CouplingGraph g = make_graph();

  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  FieldRegistry reg;
  ASSERT_TRUE(cs.at(0).register_fields(reg, g).ok());

  Component& c = cs.at(0);
  ASSERT_TRUE(c.supports_rollback());
  ASSERT_TRUE(c.advance(StepContext{.t_begin = Rational{0},
                                    .t_end = Rational{1}, .step = 0}).ok());
  ASSERT_TRUE(c.save_state().ok());
  ASSERT_TRUE(c.advance(StepContext{.t_begin = Rational{1},
                                    .t_end = Rational{2}, .step = 1}).ok());

  const auto view = reg.get(soma::FieldId{0});
  ASSERT_TRUE(view.ok());
  EXPECT_DOUBLE_EQ(view.value().at(0), 2.0);

  ASSERT_TRUE(c.restore_state().ok());
  // Same FieldRef, no rebind: the restore wrote through the same storage.
  EXPECT_DOUBLE_EQ(view.value().at(0), 1.0);
  EXPECT_EQ(view.value().data(), dynamic_cast<Toy&>(c).raw());
}
