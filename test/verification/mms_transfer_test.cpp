// Spec Section 10 level 4: the first coupled run measured against an
// analytic answer.
//
// WHAT IS AND IS NOT BEING VERIFIED HERE
//
// `Remap` is an identity copy (src/runtime/runner.cpp), `Interpolate`
// and `Accelerate` are copies, and `Extrapolate` is order 0. So there is no
// spatial transfer error to measure and no point building a case whose answer
// depends on one: the number that came out would be the missing remap, not the
// coupler. Every configuration below therefore puts the two components on
// coincident line meshes, which is exactly where the identity remap is the
// correct operator.
//
// What CAN be wrong here is the TIME behaviour of the
// exchange: the window reduction the plan performs over a subcycled producer,
// and the one-coupling-cycle lag with which the result reaches the consumer
// (`Runner::Impl::step_period`'s tick convention). That has a measurable order,
// and `MmsOrder` below measures it.
//
// THE MANUFACTURED SOLUTION
//
//   source    S(x, t) = sin(pi x) exp(-lambda t)          -- evaluated exactly
//   responder dR/dt   = kappa (S(x, t) - R(x, t)),  R(x,0) = 0
//   exact     R(x, t) = sin(pi x) * kappa/(kappa - lambda)
//                                 * (exp(-lambda t) - exp(-kappa t))
//
// The source evaluates S in closed form and the responder integrates its own
// ODE in closed form over each step given a constant input, so NEITHER
// COMPONENT CONTRIBUTES A DISCRETISATION ERROR. Every bit of the measured
// error is the coupler's: the window mean the plan forms of S, and the one
// coupling cycle by which that mean is late. Both are O(H), so the study
// below expects and measures first order in the coupling cycle H. That is the
// honest order of an explicit, lagged, unaccelerated exchange; Phases 6 and 7
// are what raise it.
//
// `MmsOrder.BreakingTheExchangeDestroysTheResult` runs the identical study with
// the delivered field ignored -- the "is this tolerance loose enough to pass
// with the coupling switched off?" control -- and asserts that it fails the
// same gate by a wide margin.

#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
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

namespace {

constexpr double kPi = 3.14159265358979323846;

/// Cell centre of cell `i` of an `n`-cell uniform mesh on [0, 1]. The serial
/// backend's "generated:line:n" uses exactly this cell layout.
[[nodiscard]] double cell_x(std::size_t i, std::size_t n) {
  return (static_cast<double>(i) + 0.5) / static_cast<double>(n);
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Writes f(x, t) into its field; the coupled partner receives it.
///
/// Evaluated in closed form at `ctx.t_end`, so the source carries no
/// discretisation error of its own and cannot pollute the order study.
class AnalyticSource final : public fuse::Component {
 public:
  AnalyticSource(std::size_t n, double amplitude, double lambda)
      : data_(n, 0.0), amplitude_(amplitude), lambda_(lambda) {}

  [[nodiscard]] std::string name() const override { return "source"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("source");
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component 'source' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    t_ = ctx.t_end.to_double();
    for (std::size_t i = 0; i < data_.size(); ++i) {
      data_[i] = value(cell_x(i, data_.size()), t_);
    }
    return fuse::Result<void>::success();
  }

  /// S(x, t). `lambda_ == 0` gives the brief's linear-in-time ramp
  /// amplitude * sin(pi x) * t; otherwise the decaying exponential the order
  /// study drives the responder with.
  [[nodiscard]] double value(double x, double t) const {
    const double shape = amplitude_ * std::sin(kPi * x);
    return lambda_ == 0.0 ? shape * t : shape * std::exp(-lambda_ * t);
  }

  [[nodiscard]] double time() const { return t_; }

 private:
  std::vector<double> data_;
  double              amplitude_;
  double              lambda_;
  double              t_ = 0.0;
};

/// Holds whatever the field transfer delivers, and nothing else.
class Receiver final : public fuse::Component {
 public:
  explicit Receiver(std::size_t n) : data_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return "receiver"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("receiver");
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component 'receiver' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
};

/// dR/dt = kappa (S_in - R), integrated in CLOSED FORM over each step.
///
/// Over `[t0, t0 + dt)` the coupler holds `S_in` fixed, and for a constant
/// input the ODE has the exact solution
///     R(t0 + dt) = c + (R(t0) - c) exp(-kappa dt).
/// Using that rather than a time-stepper is the whole point: it leaves the
/// coupler's lag and window reduction as the only error in the run.
///
/// `ignore_input_` is the control. It replaces the delivered field with a
/// frozen S(x, 0), i.e. exactly what a run whose exchange never happened would
/// integrate against, so the study can be shown to reject it.
class RelaxResponder final : public fuse::Component {
 public:
  RelaxResponder(std::size_t n, double kappa, bool ignore_input)
      : r_(n, 0.0),
        in_(n, 0.0),
        // Sized here, never resized: `advance` indexes it whenever
        // ignore_input_ is set, and a caller that forgot freeze() must read
        // zeros rather than run off the end.
        frozen_(n, 0.0),
        kappa_(kappa),
        ignore_input_(ignore_input) {}

  [[nodiscard]] std::string name() const override { return "responder"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("responder");
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component 'responder' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      // in_, not r_: r_ is this component's own state. Binding the state array
      // to a Requires field would let the coupler overwrite the solution with
      // the boundary data every coupling cycle.
      auto r = reg.bind(f, fuse::FieldRef{in_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = in_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double decay = std::exp(-kappa_ * dt);
    for (std::size_t i = 0; i < r_.size(); ++i) {
      const double c = ignore_input_ ? frozen_[i] : in_[i];
      r_[i] = c + (r_[i] - c) * decay;
    }
    t_ = ctx.t_end.to_double();
    return fuse::Result<void>::success();
  }

  /// The control's stand-in for the coupled field. Copied element-wise into
  /// the array sized at construction; a short `v` is a caller error.
  void freeze(const std::vector<double>& v) {
    if (v.size() != frozen_.size()) {
      throw std::invalid_argument("freeze(): size mismatch");
    }
    std::copy(v.begin(), v.end(), frozen_.begin());
  }

  [[nodiscard]] const std::vector<double>& state() const { return r_; }
  [[nodiscard]] double time() const { return t_; }

 private:
  std::vector<double> r_;
  std::vector<double> in_;
  std::vector<double> frozen_;
  double              kappa_;
  bool                ignore_input_;
  double              t_ = 0.0;
};

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

struct Fixture {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  fuse::OperatorDAG   dag;
  fuse::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;
  AnalyticSource*     source = nullptr;
  Receiver*           receiver = nullptr;
  RelaxResponder*     responder = nullptr;
};

struct GraphOpts {
  std::size_t        cells = 16;
  Rational           src_dt{1};
  Rational           dst_dt{1};
  fuse::AccumOp      accum = fuse::AccumOp::Instant;
  std::string        consumer = "receiver";
  fuse::Conservation conservation = fuse::Conservation::None;
};

/// graph -> lower -> schedule, on two coincident `cells`-cell line meshes.
void build_plan(Fixture& f, const GraphOpts& o) {
  auto& g = f.graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{
      .name = "m1", .source = "generated:line:" + std::to_string(o.cells)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{
      .name = "m2", .source = "generated:line:" + std::to_string(o.cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1,
                                            .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2,
                                            .dimension = 1});
  const auto cs = g.add_component(fuse::ComponentDecl{
      .name = "source", .ranks = {0, 1},
      .interval = o.src_dt});
  const auto cr = g.add_component(fuse::ComponentDecl{
      .name = o.consumer, .ranks = {1, 1},
      .interval = o.dst_dt});
  const auto fs = g.add_field(cs, fuse::FieldSpec{
      .name = "T", .region = r1, .units = u::kKelvin,
      .conservation = o.conservation,
      .intent = fuse::Intent::Provides});
  const auto fr = g.add_field(cr, fuse::FieldSpec{
      .name = "T_bc", .region = r2, .units = u::kKelvin,
      .conservation = o.conservation,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("T_link", fs, fr);
  if (o.accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(o.accum);

  ASSERT_TRUE(f.backend.load(g.mesh(m1)).ok());
  ASSERT_TRUE(f.backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  f.dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, f.dag, clocks, fuse::make_region_sizes(g, f.dag, f.backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  f.plan = sched.plan;
}

/// The transfer fixture: an analytic ramp into a passive receiver.
std::unique_ptr<Fixture> build(std::size_t n, Rational src_dt, Rational dst_dt,
                               fuse::AccumOp accum,
                               fuse::Conservation cons =
                                   fuse::Conservation::None) {
  auto f = std::make_unique<Fixture>();
  build_plan(*f, GraphOpts{.cells = n,
                           .src_dt = src_dt,
                           .dst_dt = dst_dt,
                           .accum = accum,
                           .consumer = "receiver",
                           .conservation = cons});
  if (::testing::Test::HasFatalFailure()) return f;

  auto src = std::make_unique<AnalyticSource>(n, 1.0, 0.0);  // ramp: shape * t
  auto rcv = std::make_unique<Receiver>(n);
  f->source = src.get();
  f->receiver = rcv.get();
  f->components.add(std::move(src));
  f->components.add(std::move(rcv));
  return f;
}

void run(Fixture& f, std::int64_t periods) {
  Runner runner{f.graph, f.dag, f.plan, f.backend};
  for (auto& c : f.components) {
    ASSERT_TRUE(c.register_fields(f.registry, f.graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f.components, f.registry).ok());
  ASSERT_TRUE(runner.run(f.components, periods).ok());
}

}  // namespace

// ---------------------------------------------------------------------------
// Transfer correctness
// ---------------------------------------------------------------------------

// A field must transfer bit-exactly across the identity remap: no drift, no
// scaling, no permutation. The field carries a SPATIAL PROFILE rather than one
// constant on purpose -- a broadcast of element 0, a reversal, or a copy of
// only the first cell all satisfy "a constant transfers exactly".
TEST(MmsTransfer, ConstantFieldTransfersExactly) {
  auto f = build(16, Rational{1}, Rational{1}, fuse::AccumOp::Instant);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  run(*f, 1);
  for (std::size_t i = 0; i < f->receiver->data().size(); ++i) {
    const double x = cell_x(i, 16);
    EXPECT_DOUBLE_EQ(f->receiver->data()[i], std::sin(kPi * x));
  }
}

// Mean accumulation of a linear-in-time signal over [0,1] with 4 subcycling steps
// gives the average of t = 1/4, 1/2, 3/4, 1, i.e. 5/8.
TEST(MmsTransfer, MeanOfALinearSignalMatchesTheAnalyticAverage) {
  auto f = build(16, Rational{1, 4}, Rational{1}, fuse::AccumOp::Mean);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  run(*f, 1);
  const double x0 = cell_x(0, 16);
  const double expected = std::sin(kPi * x0) * 0.625;
  EXPECT_NEAR(f->receiver->data()[0], expected, 1e-13);
  // The whole profile, not only cell 0: a reduction applied to one element and
  // broadcast would pass the line above.
  for (std::size_t i = 0; i < f->receiver->data().size(); ++i) {
    EXPECT_NEAR(f->receiver->data()[i],
                std::sin(kPi * cell_x(i, 16)) * 0.625, 1e-13);
  }
}

TEST(MmsTransfer, RepeatedRunsAreBitIdentical) {
  auto a = build(16, Rational{1, 4}, Rational{1}, fuse::AccumOp::Mean);
  auto b = build(16, Rational{1, 4}, Rational{1}, fuse::AccumOp::Mean);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  run(*a, 5);
  run(*b, 5);
  ASSERT_EQ(a->receiver->data().size(), b->receiver->data().size());
  // Two runs that both delivered nothing are also bit-identical, so pin that
  // something arrived before comparing.
  ASSERT_NE(a->receiver->data()[0], 0.0);
  for (std::size_t i = 0; i < a->receiver->data().size(); ++i) {
    EXPECT_DOUBLE_EQ(a->receiver->data()[i], b->receiver->data()[i]);
  }
}

// Conservation::Extensive is what makes compilation into the dataflow graph emit the Constrain operator
// (src/ir/rules_derived.cpp). Without it the DAG holds no Constrain at all,
// the runner performs no check, and "violations == 0" is true of a build with
// the conservation machinery deleted -- so the count of checks actually
// PERFORMED is asserted here as well.
TEST(MmsTransfer, NoConservationViolationsOnAnIdentityRemap) {
  auto f = build(16, Rational{1}, Rational{1}, fuse::AccumOp::Instant,
                 fuse::Conservation::Extensive);
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c.register_fields(f->registry, f->graph).ok());
  }
  fuse::RunnerConfig cfg;
  cfg.abort_on_violation = true;
  ASSERT_TRUE(runner.initialize(f->components, f->registry, cfg).ok());
  ASSERT_TRUE(runner.run(f->components, 10).ok());
  EXPECT_EQ(runner.stats().conservation_checks, 10)
      << "one weighted-integral check per period, or the assertion below is "
         "vacuous";
  EXPECT_EQ(runner.stats().conservation_violations, 0);
  EXPECT_LT(runner.stats().worst_conservation_error, 1e-15);
}

// ---------------------------------------------------------------------------
// The manufactured-solution order study
// ---------------------------------------------------------------------------

namespace {

constexpr double kLambda = 1.0;   // decay rate of the manufactured source
constexpr double kKappa  = 4.0;   // responder relaxation rate
constexpr std::size_t kMmsCells = 32;
constexpr int         kSubsteps = 4;  // source ticks per coupling period

/// R(x, t) for the coupled problem above. kappa != lambda by construction.
[[nodiscard]] double exact_response(double x, double t) {
  return std::sin(kPi * x) * (kKappa / (kKappa - kLambda)) *
         (std::exp(-kLambda * t) - std::exp(-kKappa * t));
}

/// Discrete L2 norm on the uniform cell-centred mesh.
[[nodiscard]] double l2_error(const std::vector<double>& got, double t) {
  const std::size_t n = got.size();
  const double      dx = 1.0 / static_cast<double>(n);
  double            sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double e = got[i] - exact_response(cell_x(i, n), t);
    sum += e * e * dx;
  }
  return std::sqrt(sum);
}

/// One MMS run to t = 1 with coupling cycle H = 1/`periods`.
///
/// Returns the L2 error of the responder's state against `exact_response`.
/// A negative return means the run itself failed; the caller asserts on it.
double mms_error(std::int64_t periods, bool break_exchange) {
  const Rational dst_dt{1, periods};
  const Rational src_dt{1, periods * kSubsteps};

  Fixture f;
  build_plan(f, GraphOpts{.cells = kMmsCells,
                          .src_dt = src_dt,
                          .dst_dt = dst_dt,
                          .accum = fuse::AccumOp::Mean,
                          .consumer = "responder"});
  if (::testing::Test::HasFatalFailure()) return -1.0;
  EXPECT_EQ(f.plan.period(), dst_dt)
      << "the coupling period must be the responder's interval";

  auto src = std::make_unique<AnalyticSource>(kMmsCells, 1.0, kLambda);
  auto rsp = std::make_unique<RelaxResponder>(kMmsCells, kKappa,
                                              break_exchange);
  if (break_exchange) {
    std::vector<double> frozen(kMmsCells, 0.0);
    for (std::size_t i = 0; i < kMmsCells; ++i) {
      frozen[i] = src->value(cell_x(i, kMmsCells), 0.0);
    }
    rsp->freeze(frozen);
  }
  f.responder = rsp.get();
  f.components.add(std::move(src));
  f.components.add(std::move(rsp));

  Runner runner{f.graph, f.dag, f.plan, f.backend};
  for (auto& c : f.components) {
    EXPECT_TRUE(c.register_fields(f.registry, f.graph).ok());
  }
  const auto init = runner.initialize(f.components, f.registry);
  EXPECT_TRUE(init.ok()) << init.error().message;
  if (!init.ok()) return -1.0;
  const auto ran = runner.run(f.components, periods);
  EXPECT_TRUE(ran.ok()) << ran.error().message;
  if (!ran.ok()) return -1.0;

  EXPECT_DOUBLE_EQ(f.responder->time(), 1.0);
  return l2_error(f.responder->state(), 1.0);
}

[[nodiscard]] double order_of(double coarse, double fine) {
  return std::log2(coarse / fine);
}

const std::int64_t kLadder[] = {8, 16, 32, 64, 128};
constexpr std::size_t kRungs = sizeof(kLadder) / sizeof(kLadder[0]);

}  // namespace

// The measurement this whole file exists for: refine the coupling cycle and
// watch the error fall at the rate an explicit, one-coupling-cycle-lagged exchange has.
TEST(MmsOrder, ErrorFallsAtFirstOrderInTheCouplingPeriod) {
  double err[kRungs] = {};
  std::string table =
      "     H        L2 error        order\n";
  for (std::size_t i = 0; i < kRungs; ++i) {
    err[i] = mms_error(kLadder[i], /*break_exchange=*/false);
    ASSERT_GT(err[i], 0.0) << "H = 1/" << kLadder[i]
                           << ": an exactly zero error means the study is "
                              "measuring nothing";
    table += "  1/" + std::to_string(kLadder[i]) + "\t" +
             std::to_string(err[i]);
    if (i > 0) table += "\t" + std::to_string(order_of(err[i - 1], err[i]));
    table += "\n";
  }
  const double observed = order_of(err[kRungs - 2], err[kRungs - 1]);
  std::cout << "[ ORDER    ] coupled response, expected 1, observed "
            << observed << "\n" << table;
  ::testing::Test::RecordProperty("mms_order", std::to_string(observed));
  ::testing::Test::RecordProperty("mms_finest_error",
                                  std::to_string(err[kRungs - 1]));

  // 0.1 is more than eight times tighter than the 1.0 a lost order would give,
  // and comfortably looser than the ~0.01 the ladder actually wanders by.
  EXPECT_NEAR(observed, 1.0, 0.1) << table;
  // Absolute gate as well: an order-1 slope on a curve that is uniformly wrong
  // by a constant factor would still pass the line above.
  EXPECT_LT(err[kRungs - 1], 3.0e-3) << table;
}

// THE CONTROL. Identical study, except the responder integrates against a
// frozen S(x, 0) instead of the delivered field -- what the run degenerates to
// if the exchange never happens. If the gates above can be met this way they
// prove nothing about the coupling, so both are asserted to fail here.
TEST(MmsOrder, BreakingTheExchangeDestroysTheResult) {
  double coupled[kRungs] = {};
  double broken[kRungs] = {};
  std::string table = "     H     coupled L2      broken L2      ratio\n";
  for (std::size_t i = 0; i < kRungs; ++i) {
    coupled[i] = mms_error(kLadder[i], /*break_exchange=*/false);
    broken[i] = mms_error(kLadder[i], /*break_exchange=*/true);
    ASSERT_GT(coupled[i], 0.0);
    ASSERT_GT(broken[i], 0.0);
    table += "  1/" + std::to_string(kLadder[i]) + "\t" +
             std::to_string(coupled[i]) + "\t" + std::to_string(broken[i]) +
             "\t" + std::to_string(broken[i] / coupled[i]) + "\n";
  }
  const double broken_order = order_of(broken[kRungs - 2], broken[kRungs - 1]);
  std::cout << "[ CONTROL  ] exchange disabled: order " << broken_order
            << ", finest error " << broken[kRungs - 1] << " vs coupled "
            << coupled[kRungs - 1] << "\n" << table;
  ::testing::Test::RecordProperty("broken_finest_error",
                                  std::to_string(broken[kRungs - 1]));
  ::testing::Test::RecordProperty("broken_over_coupled",
                                  std::to_string(broken[kRungs - 1] /
                                                 coupled[kRungs - 1]));

  // Gate 1 of ErrorFallsAtFirstOrderInTheCouplingPeriod: the broken run does
  // not converge at all, so it cannot be mistaken for first order.
  EXPECT_GT(std::abs(broken_order - 1.0), 0.1)
      << "the disabled exchange converges at first order too, so the order "
         "gate is not evidence of coupling\n" << table;
  // Gate 2: the absolute tolerance. The broken run must miss it by a mile.
  EXPECT_GT(broken[kRungs - 1], 3.0e-3) << table;
  EXPECT_GT(broken[kRungs - 1] / coupled[kRungs - 1], 20.0)
      << "the coupled and uncoupled answers are too close for the study to "
         "distinguish them\n" << table;
}
