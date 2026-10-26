// Fields of double-double and quad-double values through the Runner and a real
// remap (plans/2026-10-22-universal-number-types.md, stage U4). Such a field is
// carried as limbs and every operator that adds or multiplies values does it in
// the field's type, so what arrives is the remap of the data to the round-off of
// that type, not to binary64's. The remap weights are recovered from the Runner
// itself (a unit vector in, a column of the matrix out), and the exact result
// is computed with expansions that share no code with fuse::num.
//
// Also: the conservation check evaluates its integrals in the field's type (a
// dyadic remap, whose weights are exact, then shows a defect at the type's
// round-off, where the same data in binary64 shows binary64's), the closed list
// of types, and the refusals that keep the rest honest.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/num/arithmetic_kernels.hpp>
#include <fuse/runtime/runner.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../num/exact_expansion.hpp"

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::SerialMeshBackend;
using fuse::StepContext;
using fuse::num::dd;
using fuse::num::qd;
using fuse::test::Expansion;
namespace u = fuse::units;

/// An Arithmetic type the runtime has no operators for: three binary64 limbs.
struct Trio {
  double v[3];
};

template <>
struct fuse::value_traits<Trio> {
  static constexpr std::string_view name = "myapp:trio";
  static constexpr fuse::FieldValueKind kind = fuse::FieldValueKind::Arithmetic;
  using scalar_type = double;
  static constexpr int scalar_count = 3;
  static constexpr int gradient_size = 0;
  static double to_binary64(const Trio& t) noexcept { return t.v[0]; }
  static Trio from_binary64(double x) noexcept { return Trio{{x, 0.0, 0.0}}; }
};

namespace {

/// How a component's array is registered: as values of dd or qd (`Typed`, by the
/// number of limbs), or as that many binary64 components per cell (`Flat`).
enum class Bind { Typed, Flat, Trio };

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

fuse::Result<void> bind_field(FieldRegistry& reg, fuse::FieldId f, std::vector<double>& raw,
                              std::vector<Trio>& trio, std::size_t cells, int limbs, Bind bind) {
  switch (bind) {
    case Bind::Typed:
      if (limbs == 2) {
        return reg.bind(f, fuse::BasicFieldRef<dd>(reinterpret_cast<dd*>(raw.data()),
                                                   fuse::Layout{1, 1, cells, 1}, fuse::MemorySpace::Host));
      }
      return reg.bind(f, fuse::BasicFieldRef<qd>(reinterpret_cast<qd*>(raw.data()),
                                                 fuse::Layout{1, 1, cells, 1}, fuse::MemorySpace::Host));
    case Bind::Trio:
      return reg.bind(f, fuse::BasicFieldRef<Trio>(trio.data(), fuse::Layout{1, 1, cells, 1},
                                                   fuse::MemorySpace::Host));
    case Bind::Flat:
      break;
  }
  const int l = limbs;
  return reg.bind(f, fuse::FieldRef(raw.data(), fuse::Layout{static_cast<std::size_t>(l), 1, cells, l},
                                    fuse::MemorySpace::Host));
}

/// What the producer writes at `step`: `cells * limbs` binary64 values.
using Pattern = std::function<std::vector<double>(int step)>;

class Producer final : public fuse::Component {
 public:
  Producer(std::string name, std::size_t cells, int limbs, Bind bind, Pattern pattern)
      : name_(std::move(name)), cells_(cells), limbs_(limbs), bind_(bind), pattern_(std::move(pattern)),
        raw_(cells * static_cast<std::size_t>(limbs), 0.0), trio_(bind == Bind::Trio ? cells : 0) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = bind_field(reg, f, raw_, trio_, cells_, limbs_, bind_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    const std::vector<double> v = pattern_(steps_);
    for (std::size_t k = 0; k < raw_.size() && k < v.size(); ++k) raw_[k] = v[k];
    return fuse::Result<void>::success();
  }

 private:
  std::string name_;
  std::size_t cells_;
  int limbs_;
  Bind bind_;
  Pattern pattern_;
  int steps_ = 0;
  std::vector<double> raw_;
  std::vector<Trio> trio_;
};

class Consumer final : public fuse::Component {
 public:
  Consumer(std::size_t cells, int limbs, Bind bind)
      : cells_(cells), limbs_(limbs), bind_(bind), raw_(cells * static_cast<std::size_t>(limbs), 0.0),
        trio_(bind == Bind::Trio ? cells : 0) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = bind_field(reg, f, raw_, trio_, cells_, limbs_, bind_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
  [[nodiscard]] const std::vector<double>& raw() const { return raw_; }

 private:
  std::size_t cells_;
  int limbs_;
  Bind bind_;
  std::vector<double> raw_;
  std::vector<Trio> trio_;
};

struct Rig {
  CouplingGraph graph;
  SerialMeshBackend backend;
  fuse::OperatorDAG dag;
  fuse::ExecutionPlan plan;
  ComponentSet components;
  FieldRegistry registry;
  Producer* producer = nullptr;
  Producer* producer2 = nullptr;
  Consumer* consumer = nullptr;
};

struct Opts {
  int limbs = 2;                    // limbs per value on both sides (the graph's components)
  Bind producer_bind = Bind::Typed;
  Bind consumer_bind = Bind::Typed;
  std::size_t src_cells = 8;
  std::size_t dst_cells = 5;
  fuse::AccumOp accum = fuse::AccumOp::Instant;
  Rational src_dt{1};
  Rational dst_dt{1};               // the consumer's interval: the period the samples are taken over
  fuse::Units src_units = u::kKelvin;
  fuse::Units dst_units = u::kKelvin;  // an Integral channel needs the target in [source] x [time]
  fuse::Conservation conservation = fuse::Conservation::None;
  int declared_components = 0;      // 0: the same as limbs
  bool iteration_region = false;    // the channel is iterated to a (relative) convergence measure
  /// A second producer into the same target field, merged with the first: of dd
  /// values, or of binary64 values (two components per cell).
  enum class Second { None, Typed, Flat } second = Second::None;
  fuse::MergeOp merge = fuse::MergeOp::Sum;
};

std::unique_ptr<Rig> build(const Opts& o, Pattern pattern, Pattern pattern2 = {}) {
  auto h = std::make_unique<Rig>();
  auto& g = h->graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:" + std::to_string(o.src_cells)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:" + std::to_string(o.dst_cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{.name = "producer", .ranks = fuse::RankSet{0, 1}, .interval = o.src_dt});
  const auto c = g.add_component(fuse::ComponentDecl{.name = "consumer", .ranks = fuse::RankSet{1, 1}, .interval = o.dst_dt});
  // The graph counts binary64 values per cell: the limbs of the cell's value.
  const int comps = o.declared_components > 0 ? o.declared_components : o.limbs;
  const auto s = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = comps,
                                                .units = o.src_units, .conservation = o.conservation,
                                                .intent = fuse::Intent::Provides});
  const auto t = g.add_field(c, fuse::FieldSpec{.name = "in", .region = r2, .components = comps,
                                                .units = o.dst_units, .conservation = o.conservation,
                                                .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (o.accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(o.accum);
  if (o.second != Opts::Second::None) {
    const auto p2 = g.add_component(fuse::ComponentDecl{.name = "producer2", .ranks = fuse::RankSet{0, 1}, .interval = o.src_dt});
    const auto s2 = g.add_field(p2, fuse::FieldSpec{.name = "out", .region = r1, .components = comps,
                                                    .units = o.src_units, .conservation = o.conservation,
                                                    .intent = fuse::Intent::Provides});
    g.add_channel("link2", s2, t);
    g.merge(t, o.merge);
  }
  if (o.iteration_region) {
    fuse::ConvergenceCriterion crit;
    crit.add(fuse::Converged::Relative(1e-6, "link"));
    g.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {ch}, .max_iterations = 12, .convergence = crit});
  }
  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->dag, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;
  auto prod = std::make_unique<Producer>("producer", o.src_cells, o.limbs, o.producer_bind, std::move(pattern));
  auto cons = std::make_unique<Consumer>(o.dst_cells, o.limbs, o.consumer_bind);
  h->producer = prod.get();
  h->consumer = cons.get();
  h->components.add(std::move(prod));
  if (o.second != Opts::Second::None) {
    auto prod2 = std::make_unique<Producer>("producer2", o.src_cells, o.limbs,
                                            o.second == Opts::Second::Typed ? Bind::Typed : Bind::Flat,
                                            std::move(pattern2));
    h->producer2 = prod2.get();
    h->components.add(std::move(prod2));
  }
  h->components.add(std::move(cons));
  return h;
}

fuse::Result<void> bind_and_initialize(Rig& h, Runner& runner, const fuse::RunnerConfig& cfg = {}) {
  for (auto& c : h.components) {
    auto r = c.register_fields(h.registry, h.graph);
    if (!r.ok()) return r;
  }
  return runner.initialize(h.components, h.registry, cfg);
}

struct Outcome {
  std::vector<double> consumer;
  fuse::RunStats stats;
};

Outcome run_one_period(const Opts& o, Pattern pattern, fuse::RunnerConfig cfg = {}, Pattern pattern2 = {}) {
  auto h = build(o, std::move(pattern), std::move(pattern2));
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto init = bind_and_initialize(*h, runner, cfg);
  EXPECT_TRUE(init.ok()) << (init.ok() ? "" : init.error().message);
  const auto rp = runner.run_period(h->components);
  EXPECT_TRUE(rp.ok()) << (rp.ok() ? "" : rp.error().message);
  return {h->consumer->raw(), runner.stats()};
}

// ---- values and the exact arithmetic
template <class A>
std::vector<double> limbs_of(const A& a);
template <>
std::vector<double> limbs_of<dd>(const dd& a) { return {a.hi, a.lo}; }
template <>
std::vector<double> limbs_of<qd>(const qd& a) { return {a[0], a[1], a[2], a[3]}; }

template <class A>
A random_value(std::mt19937_64& rng) {
  std::uniform_real_distribution<double> unit(0.5, 1.0), sym(-1.0, 1.0);
  double l[4];
  l[0] = std::ldexp(unit(rng), static_cast<int>(rng() % 5) - 2);
  if (rng() & 1u) l[0] = -l[0];
  for (int i = 1; i < 4; ++i) l[i] = sym(rng) * std::ldexp(std::fabs(l[i - 1]), -54);
  return fuse::num::detail::Limbs<A>::load(l);
}

template <class A>
Expansion exact_of(const A& a) {
  Expansion e;
  const auto l = limbs_of(a);
  for (std::size_t i = l.size(); i-- > 0;) e = fuse::test::grow(e, l[i]);
  return e;
}

Expansion exact_of_limbs(const double* p, int limbs) {
  Expansion e;
  for (int i = limbs; i-- > 0;) e = fuse::test::grow(e, p[i]);
  return e;
}

/// The columns of the remap matrix as the Runner applies it: the plain
/// binary64 transfer of each unit vector.
std::vector<std::vector<double>> remap_columns(std::size_t src, std::size_t dst) {
  std::vector<std::vector<double>> w;
  for (std::size_t j = 0; j < src; ++j) {
    Opts o;
    o.limbs = 1;
    o.producer_bind = o.consumer_bind = Bind::Flat;
    o.src_cells = src;
    o.dst_cells = dst;
    const auto out = run_one_period(o, [&](int) {
      std::vector<double> e(src, 0.0);
      e[j] = 1.0;
      return e;
    });
    w.push_back(out.consumer);
  }
  return w;
}

/// The data of one period for `A`: `cells` random values, as limbs.
template <class A>
std::vector<A> data_of(std::uint64_t seed, std::size_t cells) {
  std::mt19937_64 rng(seed);
  std::vector<A> v;
  for (std::size_t i = 0; i < cells; ++i) v.push_back(random_value<A>(rng));
  return v;
}

template <class A>
std::vector<double> flat(const std::vector<A>& v) {
  std::vector<double> out;
  for (const A& a : v) for (const double x : limbs_of(a)) out.push_back(x);
  return out;
}

template <class A>
void expect_remap_is_exact_to_round_off(std::size_t src, std::size_t dst) {
  constexpr int L = fuse::num::limb_count<A>;
  const auto w = remap_columns(src, dst);
  const auto x = data_of<A>(11, src);
  Opts o;
  o.limbs = L;
  o.src_cells = src;
  o.dst_cells = dst;
  const Outcome got = run_one_period(o, [&](int) { return flat(x); });
  ASSERT_EQ(got.consumer.size(), dst * static_cast<std::size_t>(L));

  const double u = L == 2 ? 0x1p-104 : 0x1p-208;
  double worst = 0.0;
  bool nontrivial = false;
  for (std::size_t i = 0; i < dst; ++i) {
    Expansion exact;
    double mass = 0.0;
    for (std::size_t j = 0; j < src; ++j) {
      const double wij = w[j][i];
      exact = fuse::test::add(exact, fuse::test::mul(Expansion{wij}, exact_of(x[j])));
      mass += std::fabs(wij * limbs_of(x[j])[0]);
      nontrivial = nontrivial || (wij != 0.0 && wij != 1.0);
    }
    const Expansion have = exact_of_limbs(&got.consumer[i * static_cast<std::size_t>(L)], L);
    const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, have)));
    worst = std::fmax(worst, err / mass);
    EXPECT_LE(err, static_cast<double>(src + 1) * u * mass * 4) << "cell " << i;
  }
  EXPECT_TRUE(nontrivial) << "the remap was meant not to be a permutation";
  // and binary64 would not have been: the same transfer of the leading limbs alone is
  // many orders of magnitude away from the exact result of the full values
  Opts p;
  p.limbs = 1;
  p.producer_bind = p.consumer_bind = Bind::Flat;
  p.src_cells = src;
  p.dst_cells = dst;
  const Outcome plain = run_one_period(p, [&](int) {
    std::vector<double> lead;
    for (const A& a : x) lead.push_back(limbs_of(a)[0]);
    return lead;
  });
  double plain_worst = 0.0;
  for (std::size_t i = 0; i < dst; ++i) {
    Expansion exact;
    double mass = 0.0;
    for (std::size_t j = 0; j < src; ++j) {
      exact = fuse::test::add(exact, fuse::test::mul(Expansion{w[j][i]}, exact_of(x[j])));
      mass += std::fabs(w[j][i] * limbs_of(x[j])[0]);
    }
    const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, Expansion{plain.consumer[i]})));
    plain_worst = std::fmax(plain_worst, err / mass);
  }
  EXPECT_GT(plain_worst, 1e-18);
  ::testing::Test::RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
}

TEST(ArithmeticFieldRunner, ADoubleDoubleFieldIsRemappedToDoubleDoubleRoundOff) {
  expect_remap_is_exact_to_round_off<dd>(8, 5);
  expect_remap_is_exact_to_round_off<dd>(7, 3);
}

TEST(ArithmeticFieldRunner, AQuadDoubleFieldIsRemappedToQuadDoubleRoundOff) {
  expect_remap_is_exact_to_round_off<qd>(8, 5);
}

TEST(ArithmeticFieldRunner, TheBoundFieldsAreReportedWithTheirTypeAndLimbs) {
  const auto x = data_of<dd>(3, 8);
  Opts o;
  const Outcome got = run_one_period(o, [&](int) { return flat(x); });
  ASSERT_EQ(got.stats.arithmetic_fields.size(), 2u);
  for (const auto& f : got.stats.arithmetic_fields) {
    EXPECT_EQ(f.type, "dd");
    EXPECT_EQ(f.limbs, 2);
    EXPECT_EQ(f.unit_roundoff, 0x1p-104);
  }
  EXPECT_EQ(got.stats.arithmetic_fields[0].field, "producer.out");
  EXPECT_EQ(got.stats.arithmetic_fields[1].field, "consumer.in");
  // a binary64 run reports none
  Opts p;
  p.limbs = 1;
  p.producer_bind = p.consumer_bind = Bind::Flat;
  const Outcome plain = run_one_period(p, [&](int) { return std::vector<double>(8, 1.0); });
  EXPECT_TRUE(plain.stats.arithmetic_fields.empty());
}

TEST(ArithmeticFieldRunner, TheIdentityTransferMovesTheLimbsUnchanged) {
  const auto x = data_of<dd>(4, 6);
  Opts o;
  o.src_cells = o.dst_cells = 6;
  const Outcome got = run_one_period(o, [&](int) { return flat(x); });
  const auto want = flat(x);
  ASSERT_EQ(got.consumer.size(), want.size());
  for (std::size_t k = 0; k < want.size(); ++k) EXPECT_EQ(bits(got.consumer[k]), bits(want[k])) << k;
}

/// The mean over three sub-steps of a dd field: the sum of the samples in dd, one
/// division by three (a divisor that is not a power of two, so that scaling every
/// limb by a rounded 1/3 would not do), then the remap. Against the exact sum
/// transferred with the exact weights: three times the result is that.
TEST(ArithmeticFieldRunner, AMeanOverSubStepsIsAccumulatedInDoubleDouble) {
  const std::size_t src = 8, dst = 5;
  const auto w = remap_columns(src, dst);
  std::vector<std::vector<dd>> samples;
  for (int s = 0; s < 3; ++s) samples.push_back(data_of<dd>(100 + static_cast<std::uint64_t>(s), src));
  Opts o;
  o.src_cells = src;
  o.dst_cells = dst;
  o.accum = fuse::AccumOp::Mean;
  o.src_dt = Rational{1, 3};
  const Outcome got = run_one_period(o, [&](int step) { return flat(samples[static_cast<std::size_t>(step - 1)]); });
  for (std::size_t i = 0; i < dst; ++i) {
    Expansion exact;  // the remap of the sum of the samples
    double mass = 0.0;
    for (std::size_t j = 0; j < src; ++j) {
      Expansion sum;
      for (int s = 0; s < 3; ++s) sum = fuse::test::add(sum, exact_of(samples[static_cast<std::size_t>(s)][j]));
      exact = fuse::test::add(exact, fuse::test::mul(Expansion{w[j][i]}, sum));
      mass += std::fabs(w[j][i] * samples[0][j].hi);
    }
    const Expansion three_times = fuse::test::scale(exact_of_limbs(&got.consumer[i * 2], 2), 3.0);
    const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, three_times)));
    EXPECT_LE(err, 3 * 40 * 0x1p-104 * mass) << "cell " << i;
  }
}

/// The integral over sub-steps of a dd field: the sum of the samples in dd, times the
/// interval between samples, which lowering records as a ratio of integers (here 1/3,
/// and 2/5 over a period of 2). The ratio is applied in dd as a multiplication and a
/// division, so the result keeps dd precision: the remap of the exact sum times the
/// ratio, checked by multiplying the delivered values back by the denominator.
TEST(ArithmeticFieldRunner, AnIntegralOverSubStepsIsScaledByTheIntervalInDoubleDouble) {
  struct Case {
    int steps;
    Rational dt;
    Rational period;
  };
  for (const Case k : {Case{3, Rational{1, 3}, Rational{1}}, Case{5, Rational{2, 5}, Rational{2}}}) {
    const std::size_t src = 8, dst = 5;
    const auto w = remap_columns(src, dst);
    std::vector<std::vector<dd>> samples;
    for (int s = 0; s < k.steps; ++s) {
      samples.push_back(data_of<dd>(300 + static_cast<std::uint64_t>(s), src));
    }
    Opts o;
    o.src_cells = src;
    o.dst_cells = dst;
    o.accum = fuse::AccumOp::Integral;
    o.src_dt = k.dt;
    o.dst_dt = k.period;
    o.src_units = u::kWatt;
    o.dst_units = u::kJoule;
    const Outcome got =
        run_one_period(o, [&](int step) { return flat(samples[static_cast<std::size_t>(step - 1)]); });
    for (std::size_t i = 0; i < dst; ++i) {
      Expansion exact;  // the remap of the sum of the samples
      double mass = 0.0;
      for (std::size_t j = 0; j < src; ++j) {
        Expansion sum;
        for (int s = 0; s < k.steps; ++s) {
          sum = fuse::test::add(sum, exact_of(samples[static_cast<std::size_t>(s)][j]));
        }
        exact = fuse::test::add(exact, fuse::test::mul(Expansion{w[j][i]}, sum));
        mass += std::fabs(w[j][i] * samples[0][j].hi) * k.steps;
      }
      // delivered * den == remap(sum) * num
      const Expansion lhs = fuse::test::scale(exact_of_limbs(&got.consumer[i * 2], 2), static_cast<double>(k.dt.den()));
      const Expansion rhs = fuse::test::scale(exact, static_cast<double>(k.dt.num()));
      const double err = std::fabs(fuse::test::approx(fuse::test::sub(rhs, lhs)));
      EXPECT_LE(err, 40.0 * 0x1p-104 * mass * static_cast<double>(k.dt.den())) << "cell " << i << ", dt " << k.dt.num() << "/" << k.dt.den();
    }
  }
}

/// An interval whose numerator or denominator is beyond 2^53 is not a ratio of binary64
/// integers, so the integral of a dd field over such sub-steps is refused with the
/// interval named, and not scaled by a rounded ratio.
TEST(ArithmeticFieldRunner, AnIntegralOverAnIntervalBeyondBinary64IntegersIsRefused) {
  const std::int64_t big = (std::int64_t{1} << 53) + 1;
  // the numerator alone, the denominator alone, and both, each over a period of two sub-steps
  for (const Rational dt : {Rational{big, 2}, Rational{2, big}, Rational{big, big + 2}}) {
    const auto x = data_of<dd>(10, 8);
    Opts o;
    o.accum = fuse::AccumOp::Integral;
    o.src_dt = dt;
    o.dst_dt = Rational{2} * dt;
    o.src_units = u::kWatt;
    o.dst_units = u::kJoule;
    auto h = build(o, [&](int) { return flat(x); });
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    const auto init = bind_and_initialize(*h, runner);
    ASSERT_TRUE(init.ok()) << (init.ok() ? "" : init.error().message);
    const auto rp = runner.run_period(h->components);
    ASSERT_FALSE(rp.ok()) << dt.num() << "/" << dt.den();
    EXPECT_EQ(rp.error().code, "E-RT-ARITHMETIC-OP") << rp.error().message;
    EXPECT_NE(rp.error().message.find(std::to_string(dt.num()) + "/" + std::to_string(dt.den())),
              std::string::npos)
        << rp.error().message;
    EXPECT_NE(rp.error().message.find("Accumulate"), std::string::npos) << rp.error().message;
  }
}

/// A remap whose weights are exactly representable (eight cells onto four: each
/// target cell is the mean of two, an intensive field), so that the only error
/// left in the transfer of a dd field is that of the arithmetic.
TEST(ArithmeticFieldRunner, TheConservationDefectOfADyadicRemapIsAtTheTypesRoundOff) {
  const std::size_t src = 8, dst = 4;
  const auto x = data_of<dd>(21, src);
  Opts o;
  o.src_cells = src;
  o.dst_cells = dst;
  o.conservation = fuse::Conservation::Intensive;
  const Outcome got = run_one_period(o, [&](int) { return flat(x); });
  EXPECT_EQ(got.stats.conservation_checks, 1);
  EXPECT_EQ(got.stats.conservation_violations, 0);
  // the weights are what the geometry says, so the check can be made exactly: the
  // integral over the source cells (measure 1/8) against the target cells (1/4)
  Expansion before, after;
  for (std::size_t j = 0; j < src; ++j) before = fuse::test::add(before, fuse::test::scale(exact_of(x[j]), 0.125));
  for (std::size_t i = 0; i < dst; ++i) {
    after = fuse::test::add(after, fuse::test::scale(exact_of_limbs(&got.consumer[i * 2], 2), 0.25));
  }
  const double defect = std::fabs(fuse::test::approx(fuse::test::sub(after, before))) /
                        std::fabs(fuse::test::approx(before));
  ::testing::Test::RecordProperty("defect_log10", std::to_string(defect > 0.0 ? std::log10(defect) : -999.0));
  EXPECT_LT(defect, 1e-30);
  EXPECT_LT(got.stats.worst_conservation_error, 1e-30);
  // The reported value is the defect measured in dd: it agrees with the exact
  // one to the round-off of the sums, not to binary64's.
  EXPECT_LT(std::fabs(got.stats.worst_conservation_error - defect), 1e-30);
}

TEST(ArithmeticFieldRunner, ADoubleDoubleAccumulationReportsTheDefectOfTheStoredBinary64Values) {
  const std::size_t src = 8, dst = 4;
  // binary64 data of full precision: the mean of two cells is rounded when it is stored
  const auto xd = data_of<dd>(22, src);
  std::vector<double> x;
  for (const dd& a : xd) x.push_back(a.hi);
  Opts o;
  o.limbs = 1;
  o.producer_bind = o.consumer_bind = Bind::Flat;
  o.src_cells = src;
  o.dst_cells = dst;
  o.conservation = fuse::Conservation::Intensive;
  fuse::RunnerConfig cfg;
  cfg.accumulation_type = fuse::AccumulationType::DoubleDouble;
  const Outcome got = run_one_period(o, [&](int) { return x; }, cfg);
  Expansion before, after;
  for (std::size_t j = 0; j < src; ++j) before = fuse::test::add(before, fuse::test::scale(Expansion{x[j]}, 0.125));
  for (std::size_t i = 0; i < dst; ++i) after = fuse::test::add(after, fuse::test::scale(Expansion{got.consumer[i]}, 0.25));
  const double defect = std::fabs(fuse::test::approx(fuse::test::sub(after, before))) /
                        std::fabs(fuse::test::approx(before));
  // the stored values are binary64, so the defect is binary64's, and it is
  // reported exactly (the sums that measure it are double-double)
  ::testing::Test::RecordProperty("defect_log10", std::to_string(defect > 0.0 ? std::log10(defect) : -999.0));
  EXPECT_GT(defect, 0.0);
  EXPECT_LT(defect, 1e-15);
  EXPECT_LT(std::fabs(got.stats.worst_conservation_error - defect), 1e-28);
  // the row sums were accumulated in dd and rounded once: each target value is the
  // correctly rounded mean of its two source values
  for (std::size_t i = 0; i < dst; ++i) {
    const Expansion exact = fuse::test::scale(fuse::test::add(Expansion{x[2 * i]}, Expansion{x[2 * i + 1]}), 0.5);
    const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, Expansion{got.consumer[i]})));
    const double ulp = std::nextafter(std::fabs(got.consumer[i]), INFINITY) - std::fabs(got.consumer[i]);
    EXPECT_LE(err, 0.5 * ulp * (1.0 + 1e-15)) << "cell " << i;
  }
}

/// A defect far below binary64's round-off of the total: the exact integrals are
/// (1 + 2^-60) / 8 and 1 / 8, so the stored values lose 2^-60 of the total, which
/// no binary64 sum of the integrals could show (both round to 1/8).
TEST(ArithmeticFieldRunner, ADoubleDoubleAccumulationReportsADefectBelowTheRoundOffOfTheTotal) {
  const std::size_t src = 8, dst = 4;
  std::vector<double> x(src, 0.0);
  x[0] = 1.0;
  x[1] = 0x1p-60;
  Opts o;
  o.limbs = 1;
  o.producer_bind = o.consumer_bind = Bind::Flat;
  o.src_cells = src;
  o.dst_cells = dst;
  o.conservation = fuse::Conservation::Intensive;
  fuse::RunnerConfig cfg;
  cfg.accumulation_type = fuse::AccumulationType::DoubleDouble;
  const Outcome got = run_one_period(o, [&](int) { return x; }, cfg);
  Expansion before, after;
  for (std::size_t j = 0; j < src; ++j) before = fuse::test::add(before, fuse::test::scale(Expansion{x[j]}, 0.125));
  for (std::size_t i = 0; i < dst; ++i) after = fuse::test::add(after, fuse::test::scale(Expansion{got.consumer[i]}, 0.25));
  const double defect = std::fabs(fuse::test::approx(fuse::test::sub(after, before))) /
                        std::fabs(fuse::test::approx(before));
  EXPECT_GT(defect, 0.5 * 0x1p-60);   // the stored values do lose 2^-60 of the total...
  EXPECT_LT(defect, 2.0 * 0x1p-60);
  // ...and the check reports it, to the accuracy of its own sums (which are exact here)
  EXPECT_NEAR(got.stats.worst_conservation_error, defect, 1e-6 * defect);
}

TEST(ArithmeticFieldRunner, ABinary64FieldIsUnchangedByTheOptionWhenItIsLeftAtItsDefault) {
  const std::size_t src = 8, dst = 5;
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> x;
  for (std::size_t i = 0; i < src; ++i) x.push_back(d(rng));
  Opts o;
  o.limbs = 1;
  o.producer_bind = o.consumer_bind = Bind::Flat;
  o.src_cells = src;
  o.dst_cells = dst;
  const Outcome a = run_one_period(o, [&](int) { return x; });
  fuse::RunnerConfig cfg;
  cfg.accumulation_type = fuse::AccumulationType::DoubleDouble;
  const Outcome b = run_one_period(o, [&](int) { return x; }, cfg);
  ASSERT_EQ(a.consumer.size(), b.consumer.size());
  // the double-double accumulation is one rounding of the exact row sum; binary64's
  // accumulates a rounding per term, so the two differ by a few ulps at most
  bool differ = false;
  for (std::size_t i = 0; i < a.consumer.size(); ++i) {
    EXPECT_NEAR(a.consumer[i], b.consumer[i], 8 * 0x1p-52 * std::fabs(a.consumer[i]) + 1e-300);
    differ = differ || bits(a.consumer[i]) != bits(b.consumer[i]);
  }
  (void)differ;  // equal or not depends on the data; both are allowed
}

/// Two producers of dd values into one field, merged by a sum or a mean in dd.
TEST(ArithmeticFieldRunner, AMergeOfTwoDoubleDoubleFieldsIsSummedInDoubleDouble) {
  const std::size_t src = 8, dst = 5;
  const auto w = remap_columns(src, dst);
  const auto x1 = data_of<dd>(31, src), x2 = data_of<dd>(32, src);
  for (const auto merge : {fuse::MergeOp::Sum, fuse::MergeOp::Mean}) {
    Opts o;
    o.src_cells = src;
    o.dst_cells = dst;
    o.second = Opts::Second::Typed;
    o.merge = merge;
    const Outcome got = run_one_period(o, [&](int) { return flat(x1); }, {}, [&](int) { return flat(x2); });
    for (std::size_t i = 0; i < dst; ++i) {
      Expansion exact;
      double mass = 0.0;
      for (std::size_t j = 0; j < src; ++j) {
        const Expansion both = fuse::test::add(exact_of(x1[j]), exact_of(x2[j]));
        exact = fuse::test::add(exact, fuse::test::mul(Expansion{w[j][i]}, both));
        mass += std::fabs(w[j][i] * (x1[j].hi + x2[j].hi));
      }
      if (merge == fuse::MergeOp::Mean) exact = fuse::test::scale(exact, 0.5);
      const Expansion have = exact_of_limbs(&got.consumer[i * 2], 2);
      const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, have)));
      EXPECT_LE(err, 2 * static_cast<double>(src + 2) * 0x1p-104 * (mass + 1e-300) * 4)
          << (merge == fuse::MergeOp::Sum ? "sum" : "mean") << " cell " << i;
    }
  }
}

// ---- refusals
TEST(ArithmeticFieldRunner, AMergeOfDoubleDoubleAndBinary64ValuesIsRefusedByName) {
  const auto x = data_of<dd>(33, 8);
  Opts o;
  o.second = Opts::Second::Flat;
  auto h = build(o, [&](int) { return flat(x); }, [&](int) { return flat(x); });
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-ARITHMETIC-MIX") << r.error().message;
  EXPECT_NE(r.error().message.find("binary64"), std::string::npos) << r.error().message;
}

TEST(ArithmeticFieldRunner, DeliveryIntoABinary64FieldIsRefusedByName) {
  const auto x = data_of<dd>(6, 8);
  Opts o;
  o.consumer_bind = Bind::Flat;  // the consumer's array is two binary64 components per cell
  auto h = build(o, [&](int) { return flat(x); });
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-ARITHMETIC-MIX");
  EXPECT_NE(r.error().message.find("dd"), std::string::npos);
}

TEST(ArithmeticFieldRunner, ComponentsThatDoNotCountLimbsAreASizeError) {
  const auto x = data_of<dd>(7, 8);
  Opts o;
  o.declared_components = 1;  // the graph says one value per cell, the field holds two limbs
  auto h = build(o, [&](int) { return flat(x); });
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-SIZE");
}

TEST(ArithmeticFieldRunner, AnArithmeticTypeOutsideTheClosedListIsRefusedByName) {
  Opts o;
  o.limbs = 3;
  o.producer_bind = Bind::Trio;
  o.consumer_bind = Bind::Trio;
  auto h = build(o, [&](int) { return std::vector<double>(); });
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-FIELD-TYPE");
  EXPECT_NE(r.error().message.find("myapp:trio"), std::string::npos);
  EXPECT_NE(r.error().message.find("dd and qd"), std::string::npos);
}

TEST(ArithmeticFieldRunner, AnIterationRegionOverAnArithmeticFieldIsRefusedByName) {
  const auto x = data_of<dd>(9, 8);
  Opts o;
  o.iteration_region = true;
  auto h = build(o, [&](int) { return flat(x); });
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = bind_and_initialize(*h, runner);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-ARITHMETIC-OP") << r.error().message;
  EXPECT_NE(r.error().message.find("Residual"), std::string::npos) << r.error().message;
  EXPECT_NE(r.error().message.find("iteration region"), std::string::npos) << r.error().message;
}

TEST(ArithmeticFieldRunner, MinimumAndMaximumOfAnArithmeticFieldAreAccepted) {
  for (const auto op : {fuse::AccumOp::Min, fuse::AccumOp::Max, fuse::AccumOp::Sum, fuse::AccumOp::Integral}) {
    const auto x = data_of<dd>(8, 8);
    Opts o;
    o.accum = op;
    if (op == fuse::AccumOp::Integral) {  // delivers [source] x [time]
      o.src_units = u::kWatt;
      o.dst_units = u::kJoule;
    }
    auto h = build(o, [&](int) { return flat(x); });
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    const auto r = bind_and_initialize(*h, runner);
    EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  }
}

TEST(ArithmeticFieldRunner, TheScalarViewRefusesALimbFieldAndTheLimbViewAcceptsIt) {
  std::vector<double> raw(8 * 2, 0.0);
  std::vector<Trio> none;
  FieldRegistry reg;
  const fuse::FieldId id{0};
  ASSERT_TRUE(bind_field(reg, id, raw, none, 8, 2, Bind::Typed).ok());
  const auto scalars = reg.get_scalars(id);
  ASSERT_FALSE(scalars.ok());
  EXPECT_EQ(scalars.error().code, "E-FIELD-VALUE-TYPE");
  const auto limbs = reg.get_limbs(id);
  ASSERT_TRUE(limbs.ok()) << limbs.error().message;
  EXPECT_EQ(limbs.value().data(), raw.data());
  EXPECT_EQ(limbs.value().layout().components, 2);
  EXPECT_EQ(limbs.value().layout().count, 8u);
  // and a binary64 field is not a limb field
  std::vector<double> plain(8, 0.0);
  FieldRegistry reg2;
  ASSERT_TRUE(reg2.bind(id, fuse::FieldRef(plain.data(), fuse::Layout{1, 0, 8, 1}, fuse::MemorySpace::Host)).ok());
  EXPECT_FALSE(reg2.get_limbs(id).ok());
  EXPECT_EQ(reg2.get_limbs(id).error().code, "E-FIELD-VALUE-TYPE");
}

}  // namespace
