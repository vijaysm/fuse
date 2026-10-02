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
#include "fuse/base/rational.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

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
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
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
  }

  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report down_begin{.struct_size = sizeof(down_begin)};
  ASSERT_EQ(t->begin_step(ctx, &down_req, &down_begin), FUSE_OK);
  EXPECT_EQ(down_begin.t_begin.numerator, 0);
  EXPECT_EQ(down_begin.t_begin.denominator, 1);
  EXPECT_EQ(down_begin.t_end.numerator, 1);
  EXPECT_EQ(down_begin.t_end.denominator, 1);
  fuse_step_report down_end{.struct_size = sizeof(down_end)};
  ASSERT_EQ(t->end_step(ctx, &down_req, &down_end), FUSE_OK);

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

  // Period 0: upstream writes 7.0, read AFTER end_step(downstream), not
  // between begin_step()/end_step(): see context.hpp's header comment --
  // the real exchange for a period runs inside the LAST externally-driven
  // component's own end_step() call, which is downstream's here. The
  // channel is not lagged (measured in graph_fixture.hpp's own header
  // comment: lag changes typing, not timing), so downstream sees THIS
  // SAME period's export once both components have stepped.
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
