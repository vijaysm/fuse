// ABI-EXACT-TIME, ABI-EXCEPTION-CONTAINMENT (build-evidence/abi-matrix.toml),
// plus an end-to-end lifecycle smoke test that checks an actual transferred
// VALUE, not just status codes -- the thing the status-code tests alone
// cannot prove about context.hpp's deferred-exchange scheme.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/capabilities.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
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

// docs/dev/restart.md round 2, item 4's participant-side request: a C
// consumer test of request_restart_point() through the REAL plain-C
// table, proving the flag reaches fuse::Runner::take_restart_point_
// request() (checked via the fuse::abi_test::runner_of() seam, since
// CONSUMING that flag is the embedding host's own job, never something
// the plain C ABI itself exposes a read-back entry for -- that method's
// own doc comment, fuse/runtime/runner.hpp).
TEST(AbiParticipant, RequestRestartPointSetsTheRunnersFlagIdempotently) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1});
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  // Before initialize(): a no-op, not an error -- no Runner exists yet
  // to hold the flag (entries.def's own doc comment on this entry).
  ASSERT_EQ(t->request_restart_point(ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_EQ(fuse::abi_test::runner_of(ctx), nullptr);

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

  fuse::Runner* runner = fuse::abi_test::runner_of(ctx);
  ASSERT_NE(runner, nullptr);
  EXPECT_FALSE(runner->take_restart_point_request())
      << "no request has been made yet";

  ASSERT_EQ(t->request_restart_point(ctx, nullptr, nullptr), FUSE_OK);
  // Idempotent: two requests before the host polls still leave exactly
  // ONE pending request, consumed by the FIRST take_restart_point_
  // request() call -- the second call must find nothing left pending.
  ASSERT_EQ(t->request_restart_point(ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_TRUE(runner->take_restart_point_request());
  EXPECT_FALSE(runner->take_restart_point_request())
      << "take_restart_point_request() must CONSUME the flag, not peek";

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

// A buffer too small for the message gets the prefix that fits, NUL-terminated,
// truncated = 1, and the full length -- the contract c-abi.md's example relies
// on. The first implementation copied NOTHING on truncation, so a caller
// printing the buffer read uninitialised bytes.
TEST(AbiParticipant, LastErrorCopiesThePrefixThatFitsWhenTruncated) {
  const auto* t = Table();
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "upstream", std::make_unique<ThrowingComponent>("upstream")));
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "downstream", std::make_unique<TrivialComponent>("downstream")));
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_ERR_RUNTIME);

  fuse_error full{.struct_size = sizeof(full)};
  char big[1024];
  std::memset(big, 'X', sizeof(big));
  full.buffer = big;
  full.capacity = sizeof(big);
  ASSERT_EQ(t->last_error(ctx, nullptr, &full), FUSE_OK);
  ASSERT_EQ(full.truncated, 0);
  const std::string whole(big);
  ASSERT_GT(whole.size(), 8u);
  EXPECT_EQ(full.length, whole.size());

  fuse_error part{.struct_size = sizeof(part)};
  char small[8];
  std::memset(small, 'X', sizeof(small));
  part.buffer = small;
  part.capacity = sizeof(small);
  ASSERT_EQ(t->last_error(ctx, nullptr, &part), FUSE_OK);
  EXPECT_EQ(part.truncated, 1);
  EXPECT_EQ(part.length, whole.size()) << "length is the full message's, not the copied prefix's";
  EXPECT_EQ(small[7], '\0') << "the truncated copy must be NUL-terminated";
  EXPECT_EQ(std::string(small), whole.substr(0, 7));

  fuse_error none{.struct_size = sizeof(none)};
  char one[1] = {'X'};
  none.buffer = one;
  none.capacity = 0;
  ASSERT_EQ(t->last_error(ctx, nullptr, &none), FUSE_OK);
  EXPECT_EQ(none.truncated, 1);
  EXPECT_EQ(one[0], 'X') << "capacity 0 must write nothing";

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// types.h documents fuse_error::status as "out: the fuse_status of the
// failed call"; Review 2026-10-11 AB-1 found impl_last_error() wrote
// FUSE_OK unconditionally instead. A fresh context (nothing has failed
// yet) must report FUSE_OK, and the context's own failing call's status
// -- not a fixed constant -- after a failure.
TEST(AbiParticipant, LastErrorReportsTheFailingCallsStatus) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_error before{.struct_size = sizeof(before)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &before), FUSE_OK);
  EXPECT_EQ(before.status, FUSE_OK)
      << "nothing has failed on this context yet";

  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "upstream", std::make_unique<ThrowingComponent>("upstream")));
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "downstream", std::make_unique<TrivialComponent>("downstream")));
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_ERR_RUNTIME);

  fuse_error after{.struct_size = sizeof(after)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &after), FUSE_OK);
  EXPECT_EQ(after.status, FUSE_ERR_RUNTIME)
      << "must report the failing initialize() call's own status, not "
         "FUSE_OK";

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
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
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

// Time windows track (Conformance, Unit 2): the FIRST test driving a
// WINDOW-scoped member (fuse::CouplingWindowScope, not merely an ordinary
// iteration region) through fuse_begin_step()/fuse_end_step() -- read
// from src/abi/participant.cpp and src/runtime/iteration.cpp before
// writing this: Runner::Impl::advance_window()'s own externally-driven
// branch (its own "KNOWN_ISSUES.md D12" comment, right before the
// `if (comp.is_externally_driven())` block) ALREADY computes
// `cfg.required[e]` as the substep MEAN before suspending -- the exact
// semantics docs/user/c-abi.md's own "What imports are current actually
// means" section documents for a window member -- and ALREADY writes
// `cfg.provided[e]` into the window's own K-wide export buffer on
// resume, through the SAME `co_await detail::ExternalSuspend{}` protocol
// `Runner::Impl::advance()` (runner.cpp) uses for an ordinary tick. The
// mechanism was not missing; it was simply never driven by a test. This
// one proves it, against the in-process fuse::Runner as the oracle, the
// same technique ImplicitCouplingConvergesLikeInProcess above uses for
// an ordinary region.
TEST(AbiParticipant, ExternallyDrivenWindowMemberMatchesInProcessRunner) {
  constexpr int kCells = 4;
  constexpr int kMaxIterations = 60;  // ImplicitCouplingConvergesLikeInProcess's own figure

  // --- Reference: fuse::Runner driven directly, bypassing the ABI. ---
  const fuse::CouplingGraph g =
      fuse::abi_test::build_implicit_window_coupling_graph(kCells, kMaxIterations);
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{i})).ok());
  }
  const fuse::LoweringResult lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const fuse::ScheduleResult sched =
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
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

  // --- Same graph, driven entirely through the ABI, both members
  // externally driven -- the Unit 2 shape (U and V both externally
  // driven) in miniature. ---
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes =
      fuse::abi_test::build_implicit_window_coupling_bytes(kCells, kMaxIterations);
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
    // advance_window()'s own per-window checkpoint (run_region()'s own
    // "once per PASS" comment) coincides with the period boundary for
    // this fixture's own single-instance-per-period window -- same
    // timing as the ordinary region's own checkpoint_regions(), so the
    // same requires_checkpoint/requires_restore pattern applies.
    if (sweep == 0) {
      EXPECT_EQ(a_rep.requires_checkpoint, 1) << "sweep " << sweep;
      EXPECT_EQ(a_rep.requires_restore, 0) << "sweep " << sweep;
    } else {
      EXPECT_EQ(a_rep.requires_checkpoint, 0) << "sweep " << sweep;
      EXPECT_EQ(a_rep.requires_restore, 1) << "sweep " << sweep;
    }
    EXPECT_EQ(a_rep.iteration, sweep) << "sweep " << sweep;
    // The window's own [t_begin, t_end) for this single-substep,
    // window_length=1 fixture is the whole window, [0, 1) -- proves
    // advance_window()'s own `ctx.t_begin = sub.t0; ctx.t_end = sub.t1;`
    // (set before the externally-driven suspend) reaches the ABI
    // caller unchanged, exactly as the generic ctx-to-report mapping
    // in fuse_begin_step() (src/abi/participant.cpp) promises for ANY
    // advance() call site, window or ordinary.
    EXPECT_EQ(a_rep.t_begin.numerator, 0) << "sweep " << sweep;
    EXPECT_EQ(a_rep.t_begin.denominator, 1) << "sweep " << sweep;
    EXPECT_EQ(a_rep.t_end.numerator, 1) << "sweep " << sweep;
    EXPECT_EQ(a_rep.t_end.denominator, 1) << "sweep " << sweep;
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

// AbiParticipant.EndTimeBoundedRunRefusesPastItsOwnDeclaredEnd (S6 phase 3,
// Conformance, Unit 3a): fuse_setup()'s own `end_time` bounds
// prime_next_request()'s own rollover (src/abi/participant.cpp) -- a
// begin_step() that would start a period at or past the declared end
// leaves c.next_request empty (success, not an error) instead of rolling
// over, so this call reports step_opened=0, next_component=-1 (nothing
// left, not a mismatch -- fuse_step_report::step_opened's own comment,
// entries.def), and fuse_is_coupling_ongoing() reports ongoing=0. This is
// the ABI-level unit test for the SAME mechanism brusselator_lowfive_
// harness_test.cpp's own loop relies on (that test's own `kPeriods`
// bound), isolated here from that test's LowFive/Fortran machinery, using
// ExternallyDrivenWindowMemberMatchesInProcessRunner's own fixture
// (window_length=1, one window per period) immediately above.
TEST(AbiParticipant, EndTimeBoundedRunRefusesPastItsOwnDeclaredEnd) {
  constexpr int kCells = 4;
  constexpr int kMaxIterations = 60;
  constexpr int kPeriods = 2;

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes =
      fuse::abi_test::build_implicit_window_coupling_bytes(kCells, kMaxIterations);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  {
    // window_length=1 for this fixture (ExternallyDrivenWindowMember
    // MatchesInProcessRunner's own comment on a_rep.t_begin/t_end, just
    // above), so kPeriods periods end at exactly kPeriods.
    fuse_setup_options opts{.struct_size = sizeof(opts)};
    opts.has_end_time = 1;
    opts.end_time = fuse_exact_time{kPeriods, 1};
    ASSERT_EQ(t->setup(ctx, &opts, nullptr), FUSE_OK);
  }

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

  // Drive by fuse_is_coupling_ongoing()'s own next_component, the SAME
  // shape brusselator_lowfive_harness_test.cpp's own loop uses, rather
  // than a hardcoded sweep count -- this fixture's own window converges
  // in a graph-dependent number of sweeps per period. A safety cap
  // (not kPeriods itself) guards against an infinite loop if end_time
  // ever stopped bounding this correctly; it is never meant to be tight.
  constexpr int kSafetyCap = 1000;
  int step_count = 0;
  for (; step_count < kSafetyCap; ++step_count) {
    fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK)
        << "step " << step_count;
    if (!ongoing.ongoing) break;
    const bool is_a = (ongoing.next_component == 0);
    fuse_step_request& req = is_a ? a_req : b_req;
    fuse_step_report brep{.struct_size = sizeof(brep)};
    ASSERT_EQ(t->begin_step(ctx, &req, &brep), FUSE_OK) << "step " << step_count;
    if (!brep.step_opened) continue;
    if (is_a) {
      for (int i = 0; i < kCells; ++i) a_out[i] = 10.0 - a_in[i];
    } else {
      for (int i = 0; i < kCells; ++i) b_out[i] = 0.5 * b_in[i];
    }
    fuse_step_report erep{.struct_size = sizeof(erep)};
    ASSERT_EQ(t->end_step(ctx, &req, &erep), FUSE_OK) << "step " << step_count;
  }
  ASSERT_LT(step_count, kSafetyCap) << "run never reached its own declared end_time";

  // The run is over: a FURTHER begin_step() call must succeed (nothing
  // left is not an error) with step_opened=0 and next_component=-1 --
  // and fuse_is_coupling_ongoing() must agree.
  fuse_step_report past_end_rep{.struct_size = sizeof(past_end_rep)};
  ASSERT_EQ(t->begin_step(ctx, &a_req, &past_end_rep), FUSE_OK);
  EXPECT_EQ(past_end_rep.step_opened, 0);
  EXPECT_EQ(past_end_rep.next_component, -1);

  fuse_ongoing ongoing_after{.struct_size = sizeof(ongoing_after)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing_after), FUSE_OK);
  EXPECT_EQ(ongoing_after.ongoing, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent (S6
// phase 3, Conformance, Unit 3a): case (ii) of the lead's own ruling --
// a caller's begin_step() resumes past a period boundary (rolled_over
// true in prime_next_request()'s own terms, src/abi/participant.cpp)
// and the NEW period's own first-due component is NOT the one this call
// named. This is the mismatch branch `treat_as_mismatch` takes for a
// genuinely different reason than an ordinary (non-rollover) out-of-turn
// mistake (BeginStepOutOfTurnIsRefusedNamingTheExpectedComponent, this
// file) -- not an error here, since the caller could not have known from
// outside which component a still-converging window's own next period
// would start on.
//
// Built, not found by chance: a THIRD externally driven component,
// "gamma", ticking once per period OUTSIDE the window region entirely
// (an ordinary, non-window channel feeds "alpha" a value from it, never
// read meaningfully). `Runner::components_due_at_period_start()`
// (runner.hpp's own doc) answers "gamma" for this graph, a fixed
// property of the plan -- gamma's own ordinary tick always precedes the
// window region, by `build_ticks()`'s own "time-0 ordinary ticks beat
// any window region" rule (runner.cpp's own comment on that function).
// But EVERY conditional "next_component" hint resolved while a delivery
// proxy is cached (`Runner::Impl::next_real_advance_after()`'s own wrap-
// to-region's-own-entry rule, once the forward scan from the delivery
// instruction finds nothing before the region's exit) names "alpha",
// the window's own fixed region-entry, REGARDLESS of whether the sweep
// that follows is another iteration or the one that converges and ends
// the period. So the two are declared to disagree on purpose: region-
// entry (alpha, what every hint during the window says) is NOT period-
// start (gamma, what actually happens once the window converges) --
// exactly the lead's own phrasing.
//
// The LowFive-bound channel is "beta"'s own b_out -> "alpha"'s a_in
// (the BACK channel), not the forward one, and "beta" is declared on a
// 1/2 interval against "alpha"'s 1 (genuinely multirate: two beta
// substeps per alpha substep, per sweep) -- both load-bearing. MEASURED
// (this test's own temporary instruction-stream dump, since removed):
// a channel's own ExternalDelivery instruction is placed right after
// its PRODUCER's own WindowAdvance (op.hpp's own ordering-edge
// comment), so binding the FORWARD channel (alpha's a_out, produced
// first in the sweep) lets the very next real begin_step() call absorb
// the delivery proxy silently, with the window converging in ONE sweep
// and no rollover ever reached at all -- this test's own first draft.
// Binding the BACK channel instead puts the delivery right after
// "beta"'s own (last) WindowAdvance, immediately before the sweep's own
// convergence check; being on a shorter interval than "alpha" is what
// forces the window to take several sweeps to converge (a 1:1 window
// with this same update rule converges after exactly one), long enough
// for the mismatch to be the thing this test is actually checking
// rather than an accident of how fast the iteration happens to settle.
//
// Driving by the hint throughout (never guessing): the call that
// follows "next_component = alpha" after the LAST sweep's delivery
// proxy finds the window already converged, rolls `c.periods_completed`
// forward, and discovers period 1's real first request names "gamma",
// not "alpha" -- `step_opened = 0`, `next_component` = gamma's
// ComponentId, `period` = 1, and NO step opens on alpha, beta, or gamma.
// Stepping the named component ("gamma") next succeeds normally and the
// SAME still-unconsumed request runs -- confirmed by reading "a_out"/
// "b_out" unchanged by the mismatched call itself (no physics ran twice).
TEST(AbiParticipant, MismatchedRolloverNamesTheRealPeriodStartComponent) {
  using fuse::ComponentDecl;
  using fuse::ComponentKind;
  using fuse::ConvergenceCriterion;
  using fuse::CouplingGraph;
  using fuse::CouplingWindowScope;
  using fuse::DtPolicy;
  using fuse::FieldSpec;
  using fuse::Intent;
  using fuse::IterationRegionDecl;
  using fuse::LaunchMode;
  using fuse::LaunchSpec;
  using fuse::MeshDecl;
  using fuse::Rational;
  using fuse::RankSet;
  using fuse::Region;
  using fuse::WaveformIteration;

  CouplingGraph g;
  const auto ma = g.add_mesh(MeshDecl{.name = "ma", .source = "generated:line:4"});
  const auto mb = g.add_mesh(MeshDecl{.name = "mb", .source = "generated:line:4"});
  const auto mg = g.add_mesh(MeshDecl{.name = "mg", .source = "generated:line:4"});
  const auto ra = g.add_region(Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto rg = g.add_region(Region{.name = "rg", .mesh = mg, .dimension = 1});

  const auto gamma = g.add_component(
      ComponentDecl{.name = "gamma", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto a = g.add_component(
      ComponentDecl{.name = "alpha", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(
      ComponentDecl{.name = "beta", .ranks = RankSet{0, 1}, .interval = Rational{1, 2}});

  const auto g_out = g.add_field(
      gamma, FieldSpec{.name = "g_out", .region = rg, .components = 1,
                       .units = fuse::units::kKelvin, .intent = Intent::Provides});
  const auto a_out = g.add_field(
      a, FieldSpec{.name = "a_out", .region = ra, .components = 1,
                  .units = fuse::units::kKelvin, .intent = Intent::Provides});
  const auto b_in = g.add_field(
      b, FieldSpec{.name = "b_in", .region = rb, .components = 1,
                  .units = fuse::units::kKelvin, .intent = Intent::Requires});
  const auto b_out = g.add_field(
      b, FieldSpec{.name = "b_out", .region = rb, .components = 1,
                  .units = fuse::units::kHeatFluxDensity, .intent = Intent::Provides});
  const auto a_in = g.add_field(
      a, FieldSpec{.name = "a_in", .region = ra, .components = 1,
                  .units = fuse::units::kHeatFluxDensity, .intent = Intent::Requires});
  const auto g_in = g.add_field(
      a, FieldSpec{.name = "g_in", .region = ra, .components = 1,
                  .units = fuse::units::kKelvin, .intent = Intent::Requires});
  (void)g.add_channel("g2a", g_out, g_in);  // ordinary; outside the window region

  // LowFive-bound on the BACK channel (beta's own b_out): see this
  // test's own header comment above for why this, and beta's own 1/2
  // interval, are both load-bearing.
  const_cast<FieldSpec&>(g.field(b_out)).hdf5_path = "scratch_case_ii.h5/x";
  (void)g.add_component(ComponentDecl{.name = "lowfive", .ranks = RankSet{0, 1},
                                      .interval = Rational{1},
                                      .kind = ComponentKind::Coupler,
                                      .launch = LaunchSpec{.mode = LaunchMode::LowFive}});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the window's own sweep, not a real-time lag");
  g.channel(back).relax(0.7);

  CouplingWindowScope scope;
  scope.explicit_interval = Rational{1};
  scope.order = 2;
  scope.policies = {{"alpha", DtPolicy::Adjustable}, {"beta", DtPolicy::Adjustable}};
  scope.iteration = WaveformIteration::Jacobi;

  ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = 60,
      .convergence = convergence, .window_scope = scope});

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::serialize(g);
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double gbuf[4] = {0, 0, 0, 0}, gabuf[4] = {0, 0, 0, 0};
  double abuf[4] = {0, 0, 0, 0}, a2buf[4] = {0, 0, 0, 0};
  double bbuf[4] = {0, 0, 0, 0}, b2buf[4] = {0, 0, 0, 0};
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
    fview.count = 4;
    fview.owned_count = 4;
    fview.stride = 1;
    fview.data = data;
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("gamma", "g_out", gbuf);
  register_one("alpha", "g_in", gabuf);
  register_one("alpha", "a_out", abuf);
  register_one("alpha", "a_in", a2buf);
  register_one("beta", "b_in", bbuf);
  register_one("beta", "b_out", b2buf);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  // Confirmed property of the plan, read with no side effect at all:
  // period-start is gamma (ComponentId 0), not alpha (ComponentId 1) --
  // the SAME claim this test's own comment makes, checked directly.
  fuse_ongoing ongoing0{.struct_size = sizeof(ongoing0)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing0), FUSE_OK);
  ASSERT_EQ(ongoing0.next_component, 0) << "expected gamma to be period-start";

  fuse_step_request g_req{.struct_size = sizeof(g_req)};
  g_req.component = "gamma";
  fuse_step_request a_req{.struct_size = sizeof(a_req)};
  a_req.component = "alpha";
  fuse_step_request b_req{.struct_size = sizeof(b_req)};
  b_req.component = "beta";

  // Period 0's own gamma tick: the very first call ever made, so there
  // is nothing to roll over yet -- step_opened must be 1.
  fuse_step_report g_begin{.struct_size = sizeof(g_begin)};
  ASSERT_EQ(t->begin_step(ctx, &g_req, &g_begin), FUSE_OK);
  ASSERT_EQ(g_begin.step_opened, 1);
  gbuf[0] = 1.0;
  fuse_step_report g_end{.struct_size = sizeof(g_end)};
  ASSERT_EQ(t->end_step(ctx, &g_req, &g_end), FUSE_OK);

  // Drive the window via the hint reported by the PRECEDING call
  // (g_end's, then each begin_step()/end_step()'s own next_component)
  // -- never guess; the conditional hint during the window may
  // legitimately differ sweep to sweep, and "wrong" is exactly what
  // this test is checking for at the eventual rollover. A safety cap
  // bounds the loop well above max_iterations=60.
  bool saw_mismatch = false;
  std::int32_t mismatch_next_component = -3;
  std::int32_t mismatch_period = -3;
  double a_out_before_mismatch = 0.0, b_out_before_mismatch = 0.0;
  std::int32_t next_hint = g_end.next_component;
  for (int k = 0; k < 400 && !saw_mismatch; ++k) {
    if (next_hint != 1 && next_hint != 2) {
      // -1 ("nothing cached", fuse_end_step()'s own direct period-
      // complete handling) -- read a fresh hint with no side effect,
      // the SAME free read this test already checked once above.
      fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
      ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK) << "k=" << k;
      next_hint = ongoing.next_component;
    }
    const bool want_alpha = (next_hint == 1);
    ASSERT_TRUE(want_alpha || next_hint == 2) << "k=" << k << " hint=" << next_hint;
    fuse_step_request& req = want_alpha ? a_req : b_req;
    // The application's own job for a LowFive-bound channel (docs/
    // user/lowfive.md): FUSE performs no transport for b2a at all, so
    // the "H5Dwrite(beta) + H5Dread(alpha)" this stands in for is this
    // one copy, done unconditionally before every alpha substep --
    // idempotent if called when nothing new is due, and the only way
    // alpha's own "a_in" ever reflects beta's latest "b_out" here.
    if (want_alpha) a2buf[0] = b2buf[0];
    fuse_step_report brep{.struct_size = sizeof(brep)};
    const fuse_status brc = t->begin_step(ctx, &req, &brep);
    ASSERT_EQ(brc, FUSE_OK) << "k=" << k;
    if (brep.step_opened == 0 && brep.delivery_pending == 1) {
      // S6 phase 3 (Conformance, Unit 3b), item 2 of this round: an
      // ORDINARY mid-sweep delivery stop, not the period-ending
      // mismatch this test is looking for -- every ExternalDelivery
      // proxy is now its own externally visible report (never silently
      // drained), so this fires on every sweep along the way, not only
      // the last one. Re-request per next_component and keep going.
      next_hint = brep.next_component;
      continue;
    }
    if (brep.step_opened == 0) {
      saw_mismatch = true;
      mismatch_next_component = brep.next_component;
      mismatch_period = brep.period;
      a_out_before_mismatch = abuf[0];
      b_out_before_mismatch = b2buf[0];
      break;
    }
    if (want_alpha) {
      abuf[0] = 10.0 - a2buf[0];
    } else {
      b2buf[0] = 0.5 * bbuf[0];
    }
    fuse_step_report erep{.struct_size = sizeof(erep)};
    ASSERT_EQ(t->end_step(ctx, &req, &erep), FUSE_OK) << "k=" << k;
    next_hint = erep.next_component;
  }
  ASSERT_TRUE(saw_mismatch) << "the window never rolled over within the safety cap";

  // Case (ii) itself: the mismatch names gamma (period-start), not
  // alpha (what every hint during the window said, and what this call
  // was actually asked for) -- and it is period 1, not period 0.
  EXPECT_EQ(mismatch_next_component, 0) << "expected gamma's ComponentId";
  EXPECT_EQ(mismatch_period, 1);

  // No physics ran twice on the mismatched call: alpha's/beta's own
  // exported values are exactly what the LAST successful sweep left.
  EXPECT_EQ(abuf[0], a_out_before_mismatch);
  EXPECT_EQ(b2buf[0], b_out_before_mismatch);

  // The pending request survived the mismatch untouched: stepping the
  // NAMED component (gamma) next succeeds normally.
  fuse_step_report g_begin2{.struct_size = sizeof(g_begin2)};
  ASSERT_EQ(t->begin_step(ctx, &g_req, &g_begin2), FUSE_OK);
  EXPECT_EQ(g_begin2.step_opened, 1);
  EXPECT_EQ(g_begin2.period, 1);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}


TEST(AbiParticipant, OldCallerStructSizeNeverTouchesNextComponent) {
  constexpr std::size_t kOldStepReportSize =
      offsetof(fuse_step_report, t_end) + sizeof(fuse_exact_time);
  constexpr std::size_t kOldOngoingSize =
      offsetof(fuse_ongoing, now) + sizeof(fuse_exact_time);
  static_assert(kOldStepReportSize < sizeof(fuse_step_report),
               "next_component must have grown fuse_step_report (rule 3)");
  static_assert(kOldOngoingSize < sizeof(fuse_ongoing),
               "next_component must have grown fuse_ongoing (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
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

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";

  // An old caller's fuse_step_report*, heap-exact: struct_size names the
  // pre-growth layout, and there is no byte beyond it for this entry
  // point to write into.
  auto old_rep_buf = std::make_unique<std::byte[]>(kOldStepReportSize);
  auto* old_rep = reinterpret_cast<fuse_step_report*>(old_rep_buf.get());
  old_rep->struct_size = static_cast<std::uint32_t>(kOldStepReportSize);
  ASSERT_EQ(t->begin_step(ctx, &req, old_rep), FUSE_OK);
  // struct_size itself is always safely rewritten (offset 0, within any
  // valid caller's allocation -- see struct_size_covers()'s own comment);
  // the CURRENT, grown size is what's reported, same as every other
  // argument struct already does for a from-this-library's-own-table
  // caller.
  EXPECT_EQ(old_rep->struct_size, sizeof(fuse_step_report));
  EXPECT_EQ(old_rep->step, 0);

  auto old_end_buf = std::make_unique<std::byte[]>(kOldStepReportSize);
  auto* old_end = reinterpret_cast<fuse_step_report*>(old_end_buf.get());
  old_end->struct_size = static_cast<std::uint32_t>(kOldStepReportSize);
  ASSERT_EQ(t->end_step(ctx, &req, old_end), FUSE_OK);
  EXPECT_EQ(old_end->struct_size, sizeof(fuse_step_report));

  // Same proof for fuse_ongoing.
  auto old_ongoing_buf = std::make_unique<std::byte[]>(kOldOngoingSize);
  auto* old_ongoing = reinterpret_cast<fuse_ongoing*>(old_ongoing_buf.get());
  old_ongoing->struct_size = static_cast<std::uint32_t>(kOldOngoingSize);
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, old_ongoing), FUSE_OK);
  EXPECT_EQ(old_ongoing->struct_size, sizeof(fuse_ongoing));

  // A NEW caller, by contrast, DOES see next_component -- -2 from
  // begin_step() (undetermined until end_step()), and a resolved index or
  // -1 from is_coupling_ongoing() once something is cached.
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report new_begin{.struct_size = sizeof(new_begin)};
  ASSERT_EQ(t->begin_step(ctx, &down_req, &new_begin), FUSE_OK);
  EXPECT_EQ(new_begin.next_component, -2);
  fuse_step_report new_end{.struct_size = sizeof(new_end)};
  ASSERT_EQ(t->end_step(ctx, &down_req, &new_end), FUSE_OK);
  // The period is now complete (both components took their one tick):
  // nothing left to resolve, so -1, not a real index.
  EXPECT_EQ(new_end.next_component, -1);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiParticipant.IsCouplingOngoingReportsThePlanFirstComponentNotTheFirst
// DeclaredOne (Defect 3's own refinement). "alpha" is declared FIRST
// (ComponentId 0) and "beta" SECOND (ComponentId 1), same as every other
// test built on build_implicit_window_coupling_graph() -- but this
// fixture's own `reorder_channels_so_beta_is_first_in_plan_order` flag
// swaps the window region's own `decl.channels` list, so beta's
// WindowAdvance node is created FIRST during lowering despite being
// declared second (members_in_first_seen_order(), src/ir/
// window_lowering.cpp, walks `decl.channels` in order, not declaration
// order -- see that fixture's own comment for the full mechanism). If
// fuse_is_coupling_ongoing() answered from declaration order (the bug
// this test exists to catch), it would report alpha (0); reading the
// compiled plan instead reports beta (1).
TEST(AbiParticipant, IsCouplingOngoingReportsThePlanFirstComponentNotTheFirstDeclaredOne) {
  constexpr int kCells = 4;
  constexpr int kMaxIterations = 60;

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_implicit_window_coupling_bytes(
      kCells, kMaxIterations, fuse::DtPolicy::Adjustable, fuse::DtPolicy::Adjustable,
      /*reorder_channels_so_beta_is_first_in_plan_order=*/true);
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

  // Nothing has been stepped yet -- this is exactly the period-start
  // case Runner::components_due_at_period_start() exists for.
  fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);

  // fuse_ctx is opaque here (test_support.hpp only forward-declares it);
  // build the identical graph object again, the same way the reference
  // side of every other test in this file does, purely to read off
  // alpha's/beta's own ComponentId values.
  const fuse::CouplingGraph g = fuse::abi_test::build_implicit_window_coupling_graph(
      kCells, kMaxIterations, fuse::DtPolicy::Adjustable, fuse::DtPolicy::Adjustable,
      /*reorder_channels_so_beta_is_first_in_plan_order=*/true);
  const auto alpha_id = g.find_component("alpha");
  const auto beta_id = g.find_component("beta");
  ASSERT_TRUE(alpha_id.has_value());
  ASSERT_TRUE(beta_id.has_value());
  ASSERT_EQ(alpha_id->value(), 0u) << "alpha must be declared first for this test to mean anything";
  ASSERT_EQ(beta_id->value(), 1u) << "beta must be declared second for this test to mean anything";

  EXPECT_EQ(ongoing.next_component, static_cast<std::int32_t>(beta_id->value()))
      << "fuse_is_coupling_ongoing() must report the PLAN's own first externally "
         "driven component (beta, declared SECOND but first in plan order via "
         "this fixture's reordered decl.channels), not declaration order "
         "(which would wrongly report alpha, ComponentId 0)";

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

namespace {

// Unit 2b (D19), step (d) -- RANKS WITHOUT A COMPONENT. The SAME shape as
// fuse::abi_test::build_implicit_window_coupling_graph() ("alpha"/"beta",
// both window-scoped, both externally driven), plus a THIRD,
// ComponentKind::Coupler placement -- a pure placement, on its own rank,
// unconnected to any channel (a coupler's own role is routing a Remap
// through `remap_on()`, which this test does not need: only its own
// RankSet, disjoint from alpha's/beta's, matters here) -- on a DIFFERENT
// rank from alpha/beta, so `fuse::abi_test::force_this_process()` can put
// the caller on either side of the hosting line this step draws.
[[nodiscard]] fuse::CouplingGraph build_window_graph_with_coupler_rank(
    int cells, int max_iterations, fuse::RankSet alpha_beta_ranks,
    fuse::RankSet coupler_ranks) {
  fuse::CouplingGraph g;
  const auto ma = g.add_mesh(
      fuse::MeshDecl{.name = "ma", .source = "generated:line:" + std::to_string(cells)});
  const auto mb = g.add_mesh(
      fuse::MeshDecl{.name = "mb", .source = "generated:line:" + std::to_string(cells)});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "alpha", .ranks = alpha_beta_ranks, .interval = fuse::Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "beta", .ranks = alpha_beta_ranks, .interval = fuse::Rational{1}});
  g.add_coupler(fuse::CouplerDecl{.name = "coupler", .ranks = coupler_ranks});

  const auto a_out = g.add_field(
      a, fuse::FieldSpec{.name = "a_out", .region = ra, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(
      b, fuse::FieldSpec{.name = "b_in", .region = rb, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(
      b, fuse::FieldSpec{.name = "b_out", .region = rb, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(
      a, fuse::FieldSpec{.name = "a_in", .region = ra, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the window's own sweep, not a real-time lag");
  g.channel(back).relax(0.7);

  fuse::CouplingWindowScope scope;
  scope.explicit_interval = fuse::Rational{1};
  scope.order = 2;
  scope.policies = {{"alpha", fuse::DtPolicy::Adjustable},
                    {"beta", fuse::DtPolicy::Adjustable}};
  scope.iteration = fuse::WaveformIteration::Jacobi;

  fuse::ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = max_iterations,
      .convergence = convergence, .window_scope = scope});
  return g;
}

void register_alpha_beta_fields(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                                int cells, double* a_out, double* a_in,
                                double* b_out, double* b_in) {
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
    fview.count = static_cast<std::uint64_t>(cells);
    fview.owned_count = static_cast<std::uint64_t>(cells);
    fview.stride = 1;
    fview.data = data;
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("alpha", "a_out", a_out);
  register_one("alpha", "a_in", a_in);
  register_one("beta", "b_out", b_out);
  register_one("beta", "b_in", b_in);
}

}  // namespace

TEST(AbiParticipant, RunAcceptedOnAProcessHostingOnlyTheCoupler) {
  constexpr int kCells = 4;
  constexpr int kMaxIterations = 30;
  const fuse::CouplingGraph g = build_window_graph_with_coupler_rank(
      kCells, kMaxIterations, /*alpha_beta_ranks=*/fuse::RankSet{0, 1},
      /*coupler_ranks=*/fuse::RankSet{5, 1});
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // Rank 5 hosts neither "alpha" (RankSet{0,1}) nor "beta" (same) -- only
  // the coupler, a placement with no advance() to drive at all.
  ASSERT_TRUE(fuse::abi_test::force_this_process(ctx, 5));

  double a_out[kCells] = {0, 0, 0, 0};
  double a_in[kCells] = {0, 0, 0, 0};
  double b_out[kCells] = {0, 0, 0, 0};
  double b_in[kCells] = {0, 0, 0, 0};
  register_alpha_beta_fields(t, ctx, kCells, a_out, a_in, b_out, b_in);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_run_request req{.struct_size = sizeof(req)};
  req.periods = 1;
  const fuse_status rc = t->run(ctx, &req, nullptr);
  if (rc != FUSE_OK) {
    fuse_error err{.struct_size = sizeof(err)};
    char buf[2048] = {0};
    err.buffer = buf;
    err.capacity = sizeof(buf);
    t->last_error(ctx, nullptr, &err);
    ADD_FAILURE() << "fuse_run unexpectedly refused on the coupler-only "
                  << "rank (" << (err.code ? err.code : "?") << "): " << buf;
  }
  EXPECT_EQ(rc, FUSE_OK);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiParticipant, RunStillRefusedOnAProcessHostingAnExternalComponent) {
  constexpr int kCells = 4;
  constexpr int kMaxIterations = 30;
  const fuse::CouplingGraph g = build_window_graph_with_coupler_rank(
      kCells, kMaxIterations, /*alpha_beta_ranks=*/fuse::RankSet{0, 1},
      /*coupler_ranks=*/fuse::RankSet{5, 1});
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // Rank 0 hosts BOTH "alpha" and "beta" (RankSet{0,1}) -- two externally
  // driven components, same as every OTHER test in this file that never
  // calls force_this_process() at all (this_process stays unset there,
  // which this rule treats identically to "every component hosted").
  ASSERT_TRUE(fuse::abi_test::force_this_process(ctx, 0));

  double a_out[kCells] = {0, 0, 0, 0};
  double a_in[kCells] = {0, 0, 0, 0};
  double b_out[kCells] = {0, 0, 0, 0};
  double b_in[kCells] = {0, 0, 0, 0};
  register_alpha_beta_fields(t, ctx, kCells, a_out, a_in, b_out, b_in);

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_run_request req{.struct_size = sizeof(req)};
  req.periods = 1;
  const fuse_status rc = t->run(ctx, &req, nullptr);
  EXPECT_EQ(rc, FUSE_ERR_UNSUPPORTED);
  fuse_error err{.struct_size = sizeof(err)};
  char buf[2048] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, "E-ABI-RUN-EXTERNAL");

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiParticipant.DeliveryPendingIsSetExactlyOncePerLowFiveSuspension (S6
// phase 3, Conformance, Unit 3b): `delivery_pending` (entries.def's own
// comment on the field) must read 1 on the report of the begin_step() call
// that returned because the Runner suspended at a LowFive-bound channel's
// own delivery proxy, and 0 on every other report -- never left to an
// implicit zero-init rely-by-accident.
//
// Same shape as MismatchedRolloverNamesTheRealPeriodStartComponent above,
// minus "gamma": a two-member (alpha/beta) window, beta on a 1/2 interval
// against alpha's 1 (multirate, several sweeps per period), the BACK
// channel (beta's b_out -> alpha's a_in) LowFive-bound. Without a third,
// outside-the-window component, period-start names alpha itself -- the
// SAME component every mid-window hint already named -- so every
// rollover this graph produces is a MATCHED one. Unit 3a's own fix folded
// matched and unmatched rollovers into the same `step_opened = 0` branch
// (entries.def's comment on that field), so this is still a clean probe
// of `delivery_pending` specifically, not a repeat of the mismatch case
// above.
//
// Driven for THREE periods. Prediction, from reading participant.cpp's
// four report-construction sites (this round's own work): delivery_pending
// is 1 on EXACTLY the one report per period that crosses the rollover
// (`treat_as_mismatch`, reached only via `rolled_over`), 0 on every
// ordinary substep's begin_step()/end_step() report in between, and 0 on
// the final "run is over" report once all three periods are driven.
// AbiParticipant.DeliveryFieldNamesTheRightChannelOnEachSuspension (S6
// phase 3, Conformance, Unit 3b), item 2 of this round, SUPERSEDING the
// single-channel version this test name replaces: with BOTH channels of
// a two-member window LowFive-bound (a_out forward, b_out back -- the
// REAL Brusselator shape, examples/brusselator/brusselator_graph.hpp's
// own build_lowfive_plan()), and deliver_external() now suspending for
// the EXPORTER as well as the importer (this round's own item 2(a)),
// every is_delivery_due suspension is its own report (never silently
// drained), and `delivery_field` must name EXACTLY the channel whose
// own ExternalDelivery proxy produced it -- a_out's own handle or
// b_out's own handle, never anything else, and never -1 while
// delivery_pending==1.
TEST(AbiParticipant, DeliveryFieldNamesTheRightChannelOnEachSuspension) {
  using fuse::ComponentDecl;
  using fuse::ComponentKind;
  using fuse::ConvergenceCriterion;
  using fuse::CouplingGraph;
  using fuse::CouplingWindowScope;
  using fuse::DtPolicy;
  using fuse::FieldSpec;
  using fuse::Intent;
  using fuse::IterationRegionDecl;
  using fuse::LaunchMode;
  using fuse::LaunchSpec;
  using fuse::MeshDecl;
  using fuse::Rational;
  using fuse::RankSet;
  using fuse::Region;
  using fuse::WaveformIteration;

  CouplingGraph g;
  const auto ma = g.add_mesh(MeshDecl{.name = "ma", .source = "generated:line:4"});
  const auto mb = g.add_mesh(MeshDecl{.name = "mb", .source = "generated:line:4"});
  const auto ra = g.add_region(Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(Region{.name = "rb", .mesh = mb, .dimension = 1});

  const auto a = g.add_component(
      ComponentDecl{.name = "alpha", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(
      ComponentDecl{.name = "beta", .ranks = RankSet{0, 1}, .interval = Rational{1, 2}});

  const auto a_out = g.add_field(
      a, FieldSpec{.name = "a_out", .region = ra, .components = 1,
                  .units = fuse::units::kKelvin, .intent = Intent::Provides});
  const auto b_in = g.add_field(
      b, FieldSpec{.name = "b_in", .region = rb, .components = 1,
                  .units = fuse::units::kKelvin, .intent = Intent::Requires});
  const auto b_out = g.add_field(
      b, FieldSpec{.name = "b_out", .region = rb, .components = 1,
                  .units = fuse::units::kHeatFluxDensity, .intent = Intent::Provides});
  const auto a_in = g.add_field(
      a, FieldSpec{.name = "a_in", .region = ra, .components = 1,
                  .units = fuse::units::kHeatFluxDensity, .intent = Intent::Requires});

  // BOTH directions LowFive-bound, the real Brusselator shape -- not
  // just the back channel, as this test's own prior (single-channel)
  // version used.
  const_cast<FieldSpec&>(g.field(a_out)).hdf5_path = "scratch_delivery_field_a.h5/x";
  const_cast<FieldSpec&>(g.field(b_out)).hdf5_path = "scratch_delivery_field_b.h5/x";
  (void)g.add_component(ComponentDecl{.name = "lowfive", .ranks = RankSet{0, 1},
                                      .interval = Rational{1},
                                      .kind = ComponentKind::Coupler,
                                      .launch = LaunchSpec{.mode = LaunchMode::LowFive}});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the window's own sweep, not a real-time lag");
  g.channel(back).relax(0.7);

  CouplingWindowScope scope;
  scope.explicit_interval = Rational{1};
  scope.order = 2;
  scope.policies = {{"alpha", DtPolicy::Adjustable}, {"beta", DtPolicy::Adjustable}};
  scope.iteration = WaveformIteration::Jacobi;

  ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = 60,
      .convergence = convergence, .window_scope = scope});

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::serialize(g);
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double abuf[4] = {0, 0, 0, 0}, a2buf[4] = {0, 0, 0, 0};
  double bbuf[4] = {0, 0, 0, 0}, b2buf[4] = {0, 0, 0, 0};
  std::int32_t a_out_handle = -1, b_out_handle = -1;
  auto register_one = [&](const char* component, const char* field, double* data) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    if (std::string(field) == "a_out") a_out_handle = fref.handle;
    if (std::string(field) == "b_out") b_out_handle = fref.handle;
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = 4;
    fview.owned_count = 4;
    fview.stride = 1;
    fview.data = data;
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("alpha", "a_out", abuf);
  register_one("alpha", "a_in", a2buf);
  register_one("beta", "b_in", bbuf);
  register_one("beta", "b_out", b2buf);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  ASSERT_GE(a_out_handle, 0);
  ASSERT_GE(b_out_handle, 0);
  ASSERT_NE(a_out_handle, b_out_handle);

  fuse_step_request a_req{.struct_size = sizeof(a_req)};
  a_req.component = "alpha";
  fuse_step_request b_req{.struct_size = sizeof(b_req)};
  b_req.component = "beta";

  // Period-start is alpha (ComponentId 0) from the very first call -- no
  // outside-the-window component to name instead, unlike the mismatch
  // test above.
  fuse_ongoing ongoing0{.struct_size = sizeof(ongoing0)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing0), FUSE_OK);
  ASSERT_EQ(ongoing0.next_component, 0) << "expected alpha to be period-start";

  constexpr int kPeriodsToDrive = 2;
  int last_period = 0;
  int delivery_pending_ones = 0;
  int saw_a_out = 0, saw_b_out = 0;
  std::int32_t next_hint = 0;  // alpha, the only period-start component here.
  for (int k = 0; k < 4000 && last_period < kPeriodsToDrive; ++k) {
    if (next_hint != 0 && next_hint != 1) {
      fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
      ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK) << "k=" << k;
      next_hint = ongoing.next_component;
    }
    const bool want_alpha = (next_hint == 0);
    ASSERT_TRUE(want_alpha || next_hint == 1) << "k=" << k << " hint=" << next_hint;
    fuse_step_request& req = want_alpha ? a_req : b_req;
    fuse_step_report brep{.struct_size = sizeof(brep)};
    ASSERT_EQ(t->begin_step(ctx, &req, &brep), FUSE_OK) << "k=" << k;
    last_period = brep.period;
    if (brep.delivery_pending == 1) {
      ++delivery_pending_ones;
      EXPECT_EQ(brep.step_opened, 0)
          << "k=" << k << ": delivery_pending=1 must coincide with step_opened=0";
      EXPECT_TRUE(brep.delivery_field == a_out_handle ||
                 brep.delivery_field == b_out_handle)
          << "k=" << k << ": delivery_field=" << brep.delivery_field
          << " names neither a_out (" << a_out_handle << ") nor b_out ("
          << b_out_handle << ")";
      // docs/user/c-abi.md's own protocol: on delivery_pending==1 naming
      // F, write F if producing it, read F if consuming it.
      if (brep.delivery_field == a_out_handle) {
        ++saw_a_out;
        a2buf[0] = abuf[0];  // consumer's own stand-in for H5Dread(a_out)
      } else if (brep.delivery_field == b_out_handle) {
        ++saw_b_out;
        a2buf[0] = b2buf[0];  // consumer's own stand-in for H5Dread(b_out)
      }
      next_hint = brep.next_component;
      continue;  // No step opened; re-request per `brep.next_component`.
    }
    EXPECT_EQ(brep.delivery_field, -1)
        << "k=" << k << ": delivery_field must be -1 whenever delivery_pending is 0";
    if (brep.step_opened == 0) {
      // A legitimate, non-delivery period rollover (this graph's own
      // window converging cleanly, the SAME event
      // MismatchedRolloverNamesTheRealPeriodStartComponent's own header
      // comment documents for a plain, non-LowFive case) -- unlike this
      // test's own EARLIER (single-channel, single-proxy-per-period)
      // version, a rollover here is no longer guaranteed to coincide
      // with the LAST delivery, now that every delivery is its own
      // separate, earlier stop (this round's own item 2 correction).
      // Re-request per next_component, same as a delivery stop.
      next_hint = brep.next_component;
      continue;
    }
    if (want_alpha) {
      abuf[0] = 10.0 - a2buf[0];
    } else {
      b2buf[0] = 0.5 * bbuf[0];
    }
    fuse_step_report erep{.struct_size = sizeof(erep)};
    ASSERT_EQ(t->end_step(ctx, &req, &erep), FUSE_OK) << "k=" << k;
    EXPECT_EQ(erep.delivery_pending, 0)
        << "k=" << k << ": fuse_end_step() must always report 0";
    EXPECT_EQ(erep.delivery_field, -1)
        << "k=" << k << ": fuse_end_step() must always report delivery_field=-1";
    next_hint = erep.next_component;
  }

  // The headline claim: BOTH channels' own field get named at least
  // once -- never silently dropped, never confused with each other.
  EXPECT_GT(saw_a_out, 0) << "a_out's own delivery was never reported";
  EXPECT_GT(saw_b_out, 0) << "b_out's own delivery was never reported";
  EXPECT_GT(delivery_pending_ones, 0);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiParticipant.DeliveryPendingIsAlwaysZeroOnAnOrdinaryGraph (S6 phase 3,
// Conformance, Unit 3b): the ordinary (non-window, non-LowFive) path never
// rolls over at all (step_opened's own comment, entries.def) -- so every
// begin_step()/end_step() report's delivery_pending must read 0, driven
// across several periods, never left to chance.
TEST(AbiParticipant, DeliveryPendingIsAlwaysZeroOnAnOrdinaryGraph) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double up_storage[4] = {1, 2, 3, 4};
  double down_storage[4] = {0, 0, 0, 0};
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
    fview.count = 4;
    fview.owned_count = 4;
    fview.stride = 1;
    fview.data = data;
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("upstream", "out", up_storage);
  register_one("downstream", "in", down_storage);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";

  int reports_checked = 0;
  for (int period = 0; period < 5; ++period) {
    for (fuse_step_request* req : {&up_req, &down_req}) {
      fuse_step_report brep{.struct_size = sizeof(brep)};
      ASSERT_EQ(t->begin_step(ctx, req, &brep), FUSE_OK)
          << "period=" << period << " component=" << req->component;
      EXPECT_EQ(brep.step_opened, 1);
      EXPECT_EQ(brep.delivery_pending, 0)
          << "period=" << period << " component=" << req->component
          << ": the ordinary path never rolls over, delivery_pending must"
             " stay 0";
      EXPECT_EQ(brep.delivery_field, -1)
          << "period=" << period << " component=" << req->component;
      ++reports_checked;
      fuse_step_report erep{.struct_size = sizeof(erep)};
      ASSERT_EQ(t->end_step(ctx, req, &erep), FUSE_OK)
          << "period=" << period << " component=" << req->component;
      EXPECT_EQ(erep.delivery_pending, 0)
          << "period=" << period << " component=" << req->component;
      EXPECT_EQ(erep.delivery_field, -1)
          << "period=" << period << " component=" << req->component;
      ++reports_checked;
    }
  }
  EXPECT_EQ(reports_checked, 20);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// -------------------------------------------- D61 collective reconciliation
//
// KNOWN_ISSUES.md D61: begin_step()/end_step()/setup()/initialize()/run()
// must reconcile a collective refusal across the component's ranks
// (docs/dev/abi-design.md, "How the participant ABI reconciles a
// collective refusal") rather than returning on this rank's own local
// verdict alone. These tests exercise the REAL glue
// (src/abi/participant.cpp's reconcile_collective_status(), reached
// through the public begin_step()/end_step() entries exactly as a real
// caller would) through fuse::abi_test::force_fake_collective_peers()
// -- the stub reducer that stands in for a live MPI_Allreduce(MPI_2INT,
// MPI_MINLOC, ...) at tier 0 (no MPI runtime). The pure arithmetic
// itself (which status, which rank) is tested independently and more
// exhaustively in collective_reconcile_test.cpp; these tests confirm
// the WIRING -- that begin_step()/end_step() actually call through to
// it, at the right gate (force_communicator_set()), with the right
// rank's message surviving.

TEST(AbiParticipantCollectiveReconciliation,
     ThisRankSStateRefusalIsOverriddenWhenAFakePeerAlsoRefusedWithADifferentCode) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  // begin_step() before initialize() is a LOCAL, deterministic refusal
  // (E-ABI-STEP-STATE, FUSE_ERR_STATE == -3) on every rank that reaches
  // it -- not the collective-agreement scenario this entry targets, but
  // a convenient, always-reproducible local refusal to reconcile a FAKE
  // peer's own DIFFERENT code against.
  ASSERT_TRUE(fuse::abi_test::force_communicator_set(ctx, 1));
  // self_rank=0, one fake peer reporting FUSE_ERR_INVALID (-1). -3 < -1,
  // so this rank's OWN refusal (FUSE_ERR_STATE) is the agreed one, and
  // this rank is the origin: its message must be UNCHANGED (still
  // naming the real local reason), not rewritten to name a peer.
  ASSERT_TRUE(fuse::abi_test::force_fake_collective_peers(ctx, /*self_rank=*/0,
                                                           {FUSE_ERR_INVALID}));

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  const fuse_status rc = t->begin_step(ctx, &req, &rep);
  EXPECT_EQ(rc, FUSE_ERR_STATE);

  fuse_error err{.struct_size = sizeof(err)};
  char buf[256] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  const std::string message(buf);
  // This rank originated the agreed status: its own message, naming the
  // real local reason (begin_step() before initialize()), must survive
  // -- NOT the generic "rank N refused" rewrite a non-originating rank
  // gets (checked by the next test).
  EXPECT_EQ(message.find("rank "), std::string::npos)
      << "the originating rank's own message must not be rewritten: " << message;

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiParticipantCollectiveReconciliation,
     ThisRankSSuccessIsOverriddenWhenAFakePeerRefused) {
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
  ASSERT_TRUE(fuse::abi_test::force_communicator_set(ctx, 1));

  // "upstream" ticks first in this graph's schedule, so this call
  // succeeds LOCALLY (FUSE_OK) -- but a fake peer (rank 1 of 2) reports
  // E-ABI-STEP-ORDER's own FUSE_ERR_STATE. The agreed status must be
  // that peer's refusal (the only non-FUSE_OK candidate), and this
  // rank's OWN report must be REWRITTEN to carry it, naming rank 1 --
  // exactly the hazard D61 exists to close: a rank that locally "passed"
  // must not proceed alone while a peer refused.
  ASSERT_TRUE(fuse::abi_test::force_fake_collective_peers(ctx, /*self_rank=*/0,
                                                           {FUSE_ERR_STATE}));

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  const fuse_status rc = t->begin_step(ctx, &req, &rep);
  EXPECT_EQ(rc, FUSE_ERR_STATE)
      << "a rank that locally succeeded must still adopt a peer's refusal";

  fuse_error err{.struct_size = sizeof(err)};
  char buf[256] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  const std::string message(buf);
  EXPECT_NE(message.find("rank 1"), std::string::npos)
      << "the rewritten message must name the refusing rank: " << message;

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiParticipantCollectiveReconciliation,
     NoCommunicatorConfiguredMeansNoReconciliationEvenWithFakePeersSet) {
  // Single-process posture (force_communicator_set() never called):
  // reconcile_collective_status() must treat this exactly like
  // reconcile_epoch() treats an unconfigured base_communicator -- a
  // no-op -- even if a test (by mistake, or deliberately here) left fake
  // peers configured. This is the gate every other test in this section
  // relies on NOT firing by accident on the suite's many other,
  // single-process AbiParticipant cases.
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  ASSERT_TRUE(fuse::abi_test::force_fake_collective_peers(ctx, /*self_rank=*/0,
                                                           {FUSE_OK}));

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "downstream";  // out of turn: a real, local E-ABI-STEP-STATE
  fuse_step_report rep{.struct_size = sizeof(rep)};
  const fuse_status rc = t->begin_step(ctx, &req, &rep);
  EXPECT_EQ(rc, FUSE_ERR_STATE)
      << "with no communicator configured, the local verdict must stand "
         "unreconciled, exactly as before D61";

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

