// ABI-EXACT-TIME, ABI-EXCEPTION-CONTAINMENT (build-evidence/abi-matrix.toml),
// plus an end-to-end lifecycle smoke test that checks an actual transferred
// VALUE, not just status codes -- the thing the status-code tests alone
// cannot prove about context.hpp's deferred-exchange scheme.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/capabilities.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "graph_fixture.hpp"
#include "test_support.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// A real fuse::Component the test, not the ABI, registers storage for and
/// drives -- used only to get `advance()` onto the ordinary FUSE call stack
/// so AThrowingComponentBecomesRuntimeErrorWithTheMessage has somewhere for
/// an exception to actually originate from.
class ThrowingComponent final : public fuse::Component {
 public:
  explicit ThrowingComponent(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  // Deliberately NOT wrapped in anything of fuse::Runner's own (Runner
  // never calls register_fields(); src/abi/participant.cpp's
  // impl_initialize() does, directly, with no try/catch of its own -- this
  // is the ABI's OWN exception boundary, not a second instance of
  // fuse::Runner::run_period()'s already-tested one). Both this and
  // advance() below throw, so the exception-containment claim is checked
  // at both of this ABI's two call sites for a test-injected component.
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                      const fuse::CouplingGraph&) override {
    throw std::runtime_error("boom: ThrowingComponent always throws");
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    throw std::runtime_error("boom: ThrowingComponent always throws");
  }

 private:
  std::string name_;
};

/// The harmless partner for ThrowingComponent -- real, in-process, but does
/// nothing -- so fuse_run() sees a graph with NO externally-driven
/// component and actually attempts to drive it (entries.def: `run` refuses
/// "when any component is driven from outside").
class TrivialComponent final : public fuse::Component {
 public:
  explicit TrivialComponent(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                      const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const auto r = reg.bind(
          f, fuse::FieldRef{storage_.data(),
                            fuse::Layout{.stride = 1, .count = storage_.size()},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }

 private:
  std::string name_;
  std::vector<double> storage_ = std::vector<double>(4, 0.0);
};

/// KNOWN_ISSUES.md D12, test (3): a hand-written implicit-coupling
/// participant, used as a real, in-process fuse::Component for the
/// REFERENCE run below (constructed directly against fuse::Runner,
/// bypassing the ABI entirely). `ImplicitCouplingConvergesLikeInProcess`'s
/// own hand-written driving loop performs the IDENTICAL arithmetic
/// (`provides[i] = offset + scale * requires[i]`) through
/// fuse_begin_step()/fuse_end_step() instead, so the two runs are bit-for-
/// bit comparable. Real save_state()/restore_state() over its own
/// Provides output (same technique as abi_detail::ExternalStepComponent's),
/// so it may legally sit inside an iteration region.
class AffineComponent final : public fuse::Component {
 public:
  AffineComponent(std::string name, std::string provides_field,
                  std::string requires_field, double scale, double offset,
                  std::size_t n)
      : name_(std::move(name)),
        provides_field_(std::move(provides_field)),
        requires_field_(std::move(requires_field)),
        scale_(scale),
        offset_(offset),
        provides_(n, 0.0),
        requires_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                      const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const std::string& fname = g.field(f).name;
      double* data = nullptr;
      std::size_t n = 0;
      if (fname == provides_field_) {
        data = provides_.data();
        n = provides_.size();
      } else if (fname == requires_field_) {
        data = requires_.data();
        n = requires_.size();
      } else {
        continue;
      }
      const auto r = reg.bind(
          f, fuse::FieldRef{data, fuse::Layout{.stride = 1, .count = n},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (std::size_t i = 0; i < provides_.size(); ++i) {
      provides_[i] = offset_ + scale_ * requires_[i];
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_ = provides_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    if (saved_.size() != provides_.size()) {
      return fuse::Failure{"E-TEST-RESTORE",
                           "AffineComponent: restore_state() called with no "
                           "matching save_state()"};
    }
    provides_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true,
                                   fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/false);
  }

 private:
  std::string name_, provides_field_, requires_field_;
  double scale_, offset_;
  std::vector<double> provides_, requires_;
  std::vector<double> saved_;
};

fuse_ctx* CreateAndLoad(const fuse_participant_abi_v1* t,
                        const std::vector<std::byte>& bytes) {
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(co.ctx, &src, nullptr), FUSE_OK);
  return co.ctx;
}

}  // namespace

// AbiParticipant.StepBoundsAreTheSchedulesRationals (ABI-EXACT-TIME): a
// component on a 1/3 interval ticks three times per common period of 1;
// begin_step's reported bounds for each tick must be the EXACT rationals
// 0/1, 1/3, 2/3 -- 3/3, which a double cannot represent losslessly for the
// middle tick (1/3 has no finite binary expansion), so comparing the
// numerator/denominator fields directly (never to_double()) is the point.
TEST(AbiParticipant, StepBoundsAreTheSchedulesRationals) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1, 3});
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {0, 0, 0, 0};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = fref.handle;
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  // Rational{k, 3} / Rational{k + 1, 3} ARE the expected bounds, in lowest
  // terms -- which, per fuse::Rational's own normalisation (rational.hpp:
  // "den_ == 1 when num_ == 0"), is 0/1 at k=0 and 1/1 (not 3/3) at k=3.
  // Comparing against Rational's own reduction, rather than hand-picked
  // numerator/denominator literals, is what this test is actually checking:
  // fuse_exact_time crossing the boundary with no information lost to a
  // double, not a specific unreduced spelling.
  // MEASURED (this pass, via a standalone probe replaying this exact
  // graph): "upstream"'s tick 0 and "downstream"'s only tick both fall
  // at t=0 (downstream's interval is 1, spanning the whole period), tied
  // and broken by declaration order -- upstream first, per
  // build_two_component_lag1_graph()'s own add_component() calls -- so
  // the Runner's instruction stream expects upstream's tick 0, THEN
  // downstream's one tick, THEN upstream's ticks 1 and 2, in that exact
  // order. Asking for upstream's tick 1 right after tick 0 is refused
  // E-ABI-STEP-ORDER (naming "downstream") -- KNOWN_ISSUES.md D12's fix
  // makes the Runner's own schedule the one source of truth for call
  // order, where the deferred scheme this file's history replaced had
  // none.
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";

  for (int k = 0; k < 3; ++k) {
    const fuse::Rational expected_begin{k, 3};
    const fuse::Rational expected_end{k + 1, 3};
    fuse_step_report rep{.struct_size = sizeof(rep)};
    ASSERT_EQ(t->begin_step(ctx, &req, &rep), FUSE_OK) << "tick " << k;
    EXPECT_EQ(rep.t_begin.numerator, expected_begin.num()) << "tick " << k;
    EXPECT_EQ(rep.t_begin.denominator, expected_begin.den()) << "tick " << k;
    EXPECT_EQ(rep.t_end.numerator, expected_end.num()) << "tick " << k;
    EXPECT_EQ(rep.t_end.denominator, expected_end.den()) << "tick " << k;
    EXPECT_EQ(rep.step, k) << "tick " << k;
    fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
    ASSERT_EQ(t->end_step(ctx, &req, &end_rep), FUSE_OK) << "tick " << k;

    if (k == 0) {
      fuse_step_report down_begin{.struct_size = sizeof(down_begin)};
      ASSERT_EQ(t->begin_step(ctx, &down_req, &down_begin), FUSE_OK);
      EXPECT_EQ(down_begin.t_begin.numerator, 0);
      EXPECT_EQ(down_begin.t_begin.denominator, 1);
      EXPECT_EQ(down_begin.t_end.numerator, 1);
      EXPECT_EQ(down_begin.t_end.denominator, 1);
      fuse_step_report down_end{.struct_size = sizeof(down_end)};
      ASSERT_EQ(t->end_step(ctx, &down_req, &down_end), FUSE_OK);
    }
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiParticipant.AThrowingComponentBecomesRuntimeErrorWithTheMessage
// (ABI-EXCEPTION-CONTAINMENT).
TEST(AbiParticipant, AThrowingComponentBecomesRuntimeErrorWithTheMessage) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "upstream", std::make_unique<ThrowingComponent>("upstream")));
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "downstream", std::make_unique<TrivialComponent>("downstream")));

  // ThrowingComponent throws from register_fields() -- called directly by
  // impl_initialize() (src/abi/participant.cpp), never by fuse::Runner --
  // so this is where the ABI's OWN exception boundary, not
  // fuse::Runner::run_period()'s already independently-tested one, is
  // exercised.
  const fuse_status rc = t->initialize(ctx, nullptr, nullptr);
  EXPECT_EQ(rc, FUSE_ERR_RUNTIME);

  fuse_error err{.struct_size = sizeof(err)};
  char buf[256] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  const std::string message(buf);
  EXPECT_NE(message.find("boom"), std::string::npos) << "last_error: " << message;

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// End-to-end: the exported value really crosses the boundary, not just the
// status codes that say it should have.
TEST(AbiParticipant, LifecycleEndToEndTransfersEachPeriodsValue) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {7.0, 7.0, 7.0, 7.0};
  double down_storage[4] = {-1, -1, -1, -1};

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;

  fname.component = "upstream";
  fname.field = "out";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";

  // Period 0: upstream writes 7.0. MEASURED (this pass): for this graph
  // -- both components interval 1 (one tick each, spanning the WHOLE
  // period) -- upstream's export is not VALID until upstream's own tick
  // reaches its t_end, which is the period's own end; the Transport/
  // Remap that delivers it is therefore scheduled AFTER both ticks (both
  // at the period's t_begin), never between them. down_storage is still
  // stale right after begin_step(downstream) -- confirmed below -- and
  // only becomes current once end_step(downstream) resumes PAST
  // downstream's own tick to that Transport/Remap. This is unchanged
  // from before KNOWN_ISSUES.md D12's fix for exactly this graph shape
  // (context.hpp's "exact for exactly one class of graph" claim); the
  // fix's own observable difference is in ImportIsCurrentExactlyWhenThe
  // ScheduleMakesItAvailable and the subcycled/implicit-coupling tests
  // below, where the exchange point is NOT simply "the period's own end".
  fuse_step_report rep{.struct_size = sizeof(rep)};
  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  up_storage[0] = 7.0;
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);
  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  EXPECT_DOUBLE_EQ(down_storage[0], 7.0) << "period 0's own export is visible by end_step";

  // Period 1: downstream sees period 1's own 9.0, for the same reason.
  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  up_storage[0] = 9.0;
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);
  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  EXPECT_DOUBLE_EQ(down_storage[0], 9.0) << "period 1's own export is visible by end_step";

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// KNOWN_ISSUES.md D12, test (1) (lag-0 transfer): the Runner suspends
// EXACTLY at the scheduled tick, neither early nor late. MEASURED (this
// pass, via a standalone probe replaying this exact graph): for two
// components of interval 1 (one tick each, spanning the whole period),
// the channel's Remap is scheduled strictly AFTER both ticks -- upstream's
// export is not valid until upstream's own tick reaches its t_end, which
// IS the period's end -- so downstream.in is correctly still stale right
// after begin_step(downstream) (nothing has suspended too early and
// skipped ahead to a not-yet-reached Remap), and becomes current the
// instant end_step(downstream) resumes the Runner past downstream's own
// tick to that Remap -- the same point LifecycleEndToEndTransfersEach
// PeriodsValue checks. Read between a DIFFERENT pair of ticks instead
// (SubcycledMeanAccumulation, ImplicitCouplingConvergesLikeInProcess,
// below) to see the fix change an outcome: for THIS graph shape, the
// deferred, end-of-period scheme this file's own history replaced
// delivered the identical value at the identical point, because "every
// externally driven component has reported in" and "the schedule's own
// Transport/Remap position" coincide exactly when there are only two,
// same-interval, non-subcycled components.
TEST(AbiParticipant, SuspensionHappensExactlyAtTheScheduledTickNeitherEarlyNorLate) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {-1, -1, -1, -1};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;

  fname.component = "upstream";
  fname.field = "out";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";

  fuse_step_report rep{.struct_size = sizeof(rep)};
  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  for (int i = 0; i < 4; ++i) up_storage[i] = 42.0;
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);

  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(down_storage[i], -1.0)
        << "downstream.in[" << i << "] must NOT be touched yet: the Remap "
           "feeding it is scheduled strictly after downstream's own tick "
           "for this graph, so a suspend here that already ran it would "
           "be running ahead of the schedule";
  }
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(down_storage[i], 42.0)
        << "downstream.in[" << i << "] is what upstream produced THIS "
           "period, current the instant end_step resumes past it";
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// KNOWN_ISSUES.md D12, test (4): begin_step() on a component the Runner's
// own instruction stream does not expect next is refused, naming the
// component it DOES expect -- and, crucially, consumes nothing, so a
// subsequent begin_step() for the correctly-named component still
// succeeds.
TEST(AbiParticipant, BeginStepOutOfTurnIsRefusedNamingTheExpectedComponent) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {0, 0, 0, 0};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;

  fname.component = "upstream";
  fname.field = "out";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  // "upstream" ticks before "downstream" in this graph's own schedule (its
  // export feeds downstream's import, un-lagged); asking for "downstream"
  // first is out of turn.
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_ERR_STATE);

  fuse_error err{.struct_size = sizeof(err)};
  char buf[256] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  const std::string message(buf);
  EXPECT_NE(message.find("upstream"), std::string::npos)
      << "the refusal must name the component the stream expects next: "
      << message;

  // Nothing was consumed: the correctly-named component still succeeds,
  // and the period proceeds exactly as it would have.
  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  up_storage[0] = 5.0;
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);
  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  EXPECT_DOUBLE_EQ(down_storage[0], 5.0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// KNOWN_ISSUES.md D12, test (2): "upstream" subcycles 4x per common period
// under AccumOp::Mean; "downstream" must see the MEAN of the four substep
// values FUSE actually ran, not the last one -- the class of transfer the
// deferred, end-of-period scheme could never produce (D12's own finding).
TEST(AbiParticipant, SubcycledMeanAccumulation) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_subcycled_mean_bytes(fuse::Rational{1, 4});
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {-1, -1, -1, -1};
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;

  fname.component = "upstream";
  fname.field = "out";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = up_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  // Four substeps: 2, 4, 6, 8 -- mean 5, not the last value 8. The same
  // sequence test/runtime/runner_test.cpp's own Accumulate.Mean tests use.
  //
  // MEASURED (this pass, via a standalone probe replaying this exact
  // graph): "downstream" (interval 1, one tick spanning the whole
  // period) ticks at the period's own t_begin, same as "upstream"'s
  // FIRST substep -- tie-broken by declaration order, so the Runner's
  // own instruction stream expects upstream's substep 0, THEN
  // downstream's one tick, THEN upstream's substeps 1-3, in that order;
  // asking for upstream's substep 1 right after its own substep 0 is
  // refused E-ABI-STEP-ORDER (naming "downstream"). This is the mirror
  // image of SuspensionHappensExactlyAtTheScheduledTickNeitherEarlyNor
  // Late's own finding: downstream's own tick running before any of
  // upstream's later substeps is correct, not a defect -- the Mean
  // Accumulate/Interpolate chain that delivers 5.0 cannot complete
  // before upstream's LAST substep (3) has run, so down_storage is
  // still -1 right after downstream's own end_step too, and only
  // becomes 5.0 once upstream's substep 3 resumes the Runner past the
  // whole chain.
  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  const double values[4] = {2.0, 4.0, 6.0, 8.0};
  fuse_step_report rep{.struct_size = sizeof(rep)};

  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK) << "substep 0";
  for (int i = 0; i < 4; ++i) up_storage[i] = values[0];
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK) << "substep 0";

  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(down_storage[i], -1.0)
        << "downstream.in[" << i << "] must still be untouched: the Mean "
           "needs upstream's substep 3, three substeps away";
  }

  for (int k = 1; k < 4; ++k) {
    ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK) << "substep " << k;
    for (int i = 0; i < 4; ++i) up_storage[i] = values[k];
    ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK) << "substep " << k;
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(down_storage[i], 5.0)
        << "downstream.in[" << i << "] must be the MEAN of 2,4,6,8, not "
           "the last substep's value, current the instant upstream's own "
           "last substep resumes the Runner past the Accumulate/"
           "Interpolate chain that delivers it";
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// KNOWN_ISSUES.md D12, test (3): an implicit coupling group ("alpha"/
// "beta" two-way coupled inside an iteration region) driven ENTIRELY
// through fuse_begin_step()/fuse_end_step(), compared bit-for-bit against
// a REFERENCE run of the identical graph driven directly through
// fuse::Runner in-process. Both sides run the SAME affine update
// (AffineComponent, above) so the two trajectories are byte-for-byte
// comparable rather than merely "both converge".
TEST(AbiParticipant, ImplicitCouplingConvergesLikeInProcess) {
  constexpr int kCells = 4;
  // MEASURED (this pass): the affine/relaxation coefficients below give a
  // 2x2 Jacobi iteration matrix with complex eigenvalues of magnitude
  // sqrt(0.35) =~ 0.592 (trace 0.3, det 0.35) -- convergent, but reaching
  // a 1e-9 relative tolerance needs ~39 sweeps, not the 30 a first draft
  // of this test used.
  constexpr int kMaxIterations = 60;

  // --- Reference: fuse::Runner driven directly, bypassing the ABI. ---
  const fuse::CouplingGraph g =
      fuse::abi_test::build_implicit_coupling_graph(kCells, kMaxIterations);
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{i})).ok());
  }
  const fuse::LoweringResult lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const fuse::ScheduleResult sched =
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  fuse::ComponentSet ref_components;
  ref_components.add(std::make_unique<AffineComponent>(
      "alpha", "a_out", "a_in", /*scale=*/-1.0, /*offset=*/10.0, kCells));
  ref_components.add(std::make_unique<AffineComponent>(
      "beta", "b_out", "b_in", /*scale=*/0.5, /*offset=*/0.0, kCells));
  fuse::FieldRegistry ref_registry;
  for (auto& comp : ref_components) {
    ASSERT_TRUE(comp.register_fields(ref_registry, g).ok());
  }
  fuse::Runner ref_runner{g, lowered.dag, sched.plan, backend};
  {
    const fuse::Result<void> init_r = ref_runner.initialize(ref_components, ref_registry);
    ASSERT_TRUE(init_r.ok()) << init_r.error().code << ": " << init_r.error().message;
  }
  ASSERT_TRUE(ref_runner.run_period(ref_components).ok());
  ASSERT_EQ(ref_runner.iteration_stats().size(), 1u);
  const int ref_iterations = ref_runner.iteration_stats()[0].iterations;
  ASSERT_GT(ref_iterations, 0);
  ASSERT_TRUE(ref_runner.iteration_stats()[0].converged)
      << "reference run must converge within " << kMaxIterations
      << " sweeps for this comparison to mean anything";

  double ref_a_out[kCells], ref_a_in[kCells], ref_b_out[kCells], ref_b_in[kCells];
  {
    const auto a_f = g.find_component("alpha");
    ASSERT_TRUE(a_f.has_value());
    for (int i = 0; i < kCells; ++i) {
      ref_a_out[i] = ref_registry.get(g.component(*a_f).fields[0]).value().at(
          static_cast<std::size_t>(i));
      ref_a_in[i] = ref_registry.get(g.component(*a_f).fields[1]).value().at(
          static_cast<std::size_t>(i));
    }
    const auto b_f = g.find_component("beta");
    ASSERT_TRUE(b_f.has_value());
    for (int i = 0; i < kCells; ++i) {
      ref_b_in[i] = ref_registry.get(g.component(*b_f).fields[0]).value().at(
          static_cast<std::size_t>(i));
      ref_b_out[i] = ref_registry.get(g.component(*b_f).fields[1]).value().at(
          static_cast<std::size_t>(i));
    }
  }

  // --- Same graph, driven entirely through the ABI. ---
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes =
      fuse::abi_test::build_implicit_coupling_bytes(kCells, kMaxIterations);
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double a_out[kCells] = {0, 0, 0, 0};
  double a_in[kCells] = {0, 0, 0, 0};
  double b_out[kCells] = {0, 0, 0, 0};
  double b_in[kCells] = {0, 0, 0, 0};

  auto register_one = [&](const char* component, const char* field, double* data) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = kCells;
    fview.owned_count = kCells;
    fview.stride = 1;
    fview.data = data;
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("alpha", "a_out", a_out);
  register_one("alpha", "a_in", a_in);
  register_one("beta", "b_out", b_out);
  register_one("beta", "b_in", b_in);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request a_req{.struct_size = sizeof(a_req)};
  a_req.component = "alpha";
  fuse_step_request b_req{.struct_size = sizeof(b_req)};
  b_req.component = "beta";

  for (int sweep = 0; sweep < ref_iterations; ++sweep) {
    fuse_step_report a_rep{.struct_size = sizeof(a_rep)};
    ASSERT_EQ(t->begin_step(ctx, &a_req, &a_rep), FUSE_OK) << "sweep " << sweep;
    // KNOWN_ISSUES.md D12: requires_checkpoint fires exactly on the first
    // sweep (right after checkpoint_regions() saved this component),
    // requires_restore on every sweep after it (right after
    // run_region()'s own restore loop).
    if (sweep == 0) {
      EXPECT_EQ(a_rep.requires_checkpoint, 1) << "sweep " << sweep;
      EXPECT_EQ(a_rep.requires_restore, 0) << "sweep " << sweep;
    } else {
      EXPECT_EQ(a_rep.requires_checkpoint, 0) << "sweep " << sweep;
      EXPECT_EQ(a_rep.requires_restore, 1) << "sweep " << sweep;
    }
    EXPECT_EQ(a_rep.iteration, sweep) << "sweep " << sweep;
    for (int i = 0; i < kCells; ++i) a_out[i] = 10.0 - a_in[i];
    ASSERT_EQ(t->end_step(ctx, &a_req, &a_rep), FUSE_OK) << "sweep " << sweep;

    fuse_step_report b_rep{.struct_size = sizeof(b_rep)};
    ASSERT_EQ(t->begin_step(ctx, &b_req, &b_rep), FUSE_OK) << "sweep " << sweep;
    if (sweep == 0) {
      EXPECT_EQ(b_rep.requires_checkpoint, 1) << "sweep " << sweep;
      EXPECT_EQ(b_rep.requires_restore, 0) << "sweep " << sweep;
    } else {
      EXPECT_EQ(b_rep.requires_checkpoint, 0) << "sweep " << sweep;
      EXPECT_EQ(b_rep.requires_restore, 1) << "sweep " << sweep;
    }
    for (int i = 0; i < kCells; ++i) b_out[i] = 0.5 * b_in[i];
    {
      const fuse_status rc = t->end_step(ctx, &b_req, &b_rep);
      if (rc != FUSE_OK) {
        fuse_error err{.struct_size = sizeof(err)};
        char buf[512] = {0};
        err.buffer = buf;
        err.capacity = sizeof(buf);
        t->last_error(ctx, nullptr, &err);
        ASSERT_EQ(rc, FUSE_OK) << "sweep " << sweep << ": " << buf;
      }
    }
  }

  fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);
  EXPECT_EQ(ongoing.period, 1)
      << "exactly ref_iterations sweeps must complete the one period";

  for (int i = 0; i < kCells; ++i) {
    EXPECT_DOUBLE_EQ(a_out[i], ref_a_out[i]) << "a_out[" << i << "]";
    EXPECT_DOUBLE_EQ(a_in[i], ref_a_in[i]) << "a_in[" << i << "]";
    EXPECT_DOUBLE_EQ(b_out[i], ref_b_out[i]) << "b_out[" << i << "]";
    EXPECT_DOUBLE_EQ(b_in[i], ref_b_in[i]) << "b_in[" << i << "]";
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
