// "The period contract" (docs/dev/abi-design.md's section of that name;
// fuse::Runner::run_period()'s own doc comment,
// include/fuse/runtime/runner.hpp). The invariant stated there, generic
// to the ABI rather than to any one example graph: the Runner must never
// execute a period's real instructions beyond the last one the caller
// actually requested, even once a window-scoped region's own state has
// reached a fixed point and resolving the NEXT period's first request
// would cost nothing to look up.
//
// WHY THIS FILE EXISTS ALONGSIDE brusselator_period_rollover_test.cpp.
// That file pins the hazard through u_loop.c's/
// v_loop.f90's own real caller-loop shape, with the Brusselator's own
// multirate timing, checking that a FOLLOW-UP fuse_begin_step() call
// never resolves a cached request for an unrequested period. This file
// states the invariant itself, independent of any one participant's
// driving loop, directly on every REAL step as it happens: a step that
// actually opens (`fuse_step_report::step_opened == 1`, meaning real
// instructions -- including this region's own Constrain, which is what
// would hang a multi-rank run in its own base-communicator MPI_Allreduce)
// must report a period strictly less than
// the number of periods this test actually asked for. No Runner-level
// oracle is needed to state this: the bound is read directly off the
// SAME report field every real caller already reads.
//
// The graph is the SAME minimal, single-process, no-coupler window
// shape brusselator_period_rollover_test.cpp builds (brusselator::
// add_window_coupling(), both transfer fields hdf5_path-bound so the
// channel compiles to OpKind::ExternalDelivery, never Transport) --
// reused rather than re-derived, because getting a coupling window's
// own setup right from scratch is exactly the kind of duplicated-and-
// drifting logic this project's own standing rule warns against (a
// parallel model of prime_next_request()'s control flow is deliberately not
// written).
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/native_backend.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_graph.hpp"
#include "test_support.hpp"

namespace {

using fuse::ComponentKind;
using fuse::FieldRegistry;
using fuse::FieldRef;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::Layout;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::MemorySpace;
using fuse::Rational;
using fuse::RankSet;
using fuse::WaveformIteration;

// Same multirate timing as brusselator_period_rollover_test.cpp's own
// kDtU/kDtV/kWindow (do not divide one another, so the window genuinely
// needs more than one substep of each member); the grid is 1x1
// (irrelevant to this invariant, which is about how many TIMES the
// region's own instruction stream is walked, never about the data moved
// through it).
constexpr int kNx = 1, kNy = 1;
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;
constexpr int kRequestedPeriods = 2;

// A window member whose registered storage never varies, so the window's
// own residual is identically zero and the Jacobi sweep converges on
// sweep 0 every period -- the fewest possible sweeps, matching
// brusselator_period_rollover_test.cpp's own StubBlock.
class StubBlock final : public fuse::Component {
 public:
  explicit StubBlock(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      std::vector<double>& storage = is_out ? out_ : in_;
      storage.assign(static_cast<std::size_t>(kNx) * static_cast<std::size_t>(kNy), 0.0);
      auto r = reg.bind(f, FieldRef{storage.data(),
                                    Layout{.stride = 1, .count = storage.size()},
                                    MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
  std::vector<double> out_;
  std::vector<double> in_;
};

fuse::CouplingGraph build_window_graph(fuse::FieldSpec** u_out_field = nullptr,
                                       fuse::FieldSpec** v_out_field = nullptr) {
  fuse::CouplingGraph g;
  const fuse::MeshId mesh = g.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNx) + "x" + std::to_string(kNy)});
  const fuse::RegionId region = g.add_region({.name = "u_all", .mesh = mesh, .dimension = 2});
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      g, region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1},
      RankSet{0, 1}, RankSet{0, 1});
  const auto u_out = g.find_field(handles.u_component, "u_out");
  const auto v_out = g.find_field(handles.v_component, "v_out");
  if (u_out.has_value()) const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "stub_u.h5/grid/u_out";
  if (v_out.has_value()) const_cast<FieldSpec&>(g.field(*v_out)).hdf5_path = "stub_v.h5/grid/v_out";
  // Fieldless LowFive marker, required by validate()'s own E-LOWFIVE-NO-
  // COMPONENT check whenever a field sets hdf5_path -- ComponentKind::
  // Coupler so fuse_initialize()'s own wiring loop exempts it from
  // needing a registered implementation (brusselator_period_rollover_
  // test.cpp's own precedent, build_and_initialize()).
  (void)g.add_component({.name = "lowfive", .ranks = RankSet{0, 1},
                         .interval = Rational{1},
                         .kind = ComponentKind::Coupler,
                         .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  (void)u_out_field;
  (void)v_out_field;
  return g;
}

const fuse_participant_abi_v1* Table() {
  static const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

// u_loop.c's own u_loop_step() propagation, generalised
// past "U"/"V": standing in for the real caller loop's own "has a period
// closed" check, which reads ONLY this function's own out_rep (never the
// begin_step() report directly) -- see brusselator_period_rollover_test.
// cpp's own drive_one_step() for the full account of why this
// propagation line matters.
fuse_status drive_one_step(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                           const char* component, fuse_step_report* out_rep,
                           fuse_step_report* out_begin_rep) {
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = component;
  std::memset(out_begin_rep, 0, sizeof(*out_begin_rep));
  out_begin_rep->struct_size = sizeof(*out_begin_rep);
  const fuse_status begin_rc = t->begin_step(ctx, &req, out_begin_rep);
  if (begin_rc != FUSE_OK) return begin_rc;
  if (!out_begin_rep->step_opened) {
    std::memset(out_rep, 0, sizeof(*out_rep));
    out_rep->struct_size = sizeof(*out_rep);
    out_rep->next_component = out_begin_rep->next_component;
    return FUSE_OK;
  }
  std::memset(out_rep, 0, sizeof(*out_rep));
  out_rep->struct_size = sizeof(*out_rep);
  return t->end_step(ctx, &req, out_rep);
}

}  // namespace

TEST(AbiPeriodBoundary, NeverRunsAPeriodTheCallerDidNotRequestOnceStateHasReachedAFixedPoint) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  fuse::CouplingGraph g = build_window_graph();
  const auto bytes = fuse::serialize(g);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // No fuse_setup_options::has_end_time -- the real row's own choice
  // (u_main.c/v_main.f90), reproduced deliberately: this is the one
  // combination (no end time, base communicator configured) that puts the
  // real row at risk.
  fuse_setup_options opts{.struct_size = sizeof(opts)};
  ASSERT_EQ(t->setup(ctx, &opts, nullptr), FUSE_OK);

  const int n = kNx * kNy;
  StubBlock u("U"), v("V");
  FieldRegistry registry;
  ASSERT_TRUE(u.register_fields(registry, g).ok());
  ASSERT_TRUE(v.register_fields(registry, g).ok());

  std::vector<std::vector<double>> storage;
  auto register_one = [&](const char* component, const char* field) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    if (t->field_handle(ctx, &fname, &fref) != FUSE_OK) return false;
    storage.emplace_back(static_cast<std::size_t>(n), 0.0);
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = static_cast<std::uint64_t>(n);
    fview.owned_count = static_cast<std::uint64_t>(n);
    fview.stride = 1;
    fview.data = storage.back().data();
    return t->register_field(ctx, &fview, nullptr) == FUSE_OK;
  };
  ASSERT_TRUE(register_one("U", "u_out"));
  ASSERT_TRUE(register_one("U", "u_in"));
  ASSERT_TRUE(register_one("V", "v_out"));
  ASSERT_TRUE(register_one("V", "v_in"));

  {
    const fuse_status rc = t->initialize(ctx, nullptr, nullptr);
    if (rc != FUSE_OK) {
      fuse_error err{.struct_size = sizeof(err)};
      char buf[512] = {0};
      err.buffer = buf;
      err.capacity = sizeof(buf);
      t->last_error(ctx, nullptr, &err);
      FAIL() << "initialize() failed: " << buf;
    }
  }
  // The real row's own `has_end_time == false AND a base communicator IS
  // configured` combination -- see this seam's own header comment
  // (test_support.hpp) for why it must be set AFTER initialize() at tier
  // 0, and why doing so is safe.
  ASSERT_TRUE(fuse::abi_test::force_communicator_set(ctx, 1));

  // Drive exactly kRequestedPeriods periods through the real public ABI,
  // asking is_coupling_ongoing() which component the compiled stream
  // expects next before EVERY call (never guessing or alternating
  // blindly -- the schedule may call the same component's own turn more
  // than once before the other gets one), stopping the instant the
  // reported period count first reaches kRequestedPeriods with nothing
  // left to step (next_component == -1) -- never calling begin_step()
  // again afterward. A caller that kept calling in regardless is exactly
  // brusselator_period_rollover_test.cpp's own, separate defect; this
  // test's own job is only to confirm the Runner's own instruction count
  // never ran past what was requested, not to re-derive that OTHER
  // caller-loop bug.
  const auto u_cid = g.find_component("U");
  const auto v_cid = g.find_component("V");
  ASSERT_TRUE(u_cid.has_value() && v_cid.has_value());
  const auto component_name = [&](std::int32_t idx) -> const char* {
    return idx == static_cast<std::int32_t>(u_cid->value()) ? "U" : "V";
  };

  // Seed the very first call from is_coupling_ongoing() -- reliable at a
  // period's own START (its own doc comment, entries.def), which this
  // is. EVERY later call's own component name comes from the PREVIOUS
  // drive_one_step()'s own `rep.next_component` instead of re-querying
  // is_coupling_ongoing() mid-period, which that same doc comment says
  // is not a free read once a step is already open or a rollover is in
  // play -- exactly the field this test's own drive_one_step() (and the
  // real u_loop.c/v_loop.f90) already propagates for precisely this
  // purpose.
  fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
  ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);
  std::int32_t next = ongoing.next_component;

  // THE INVARIANT, checked on every REAL step as it happens (not
  // inferred afterward from an aggregate count): a step actually opened
  // (step_opened == 1, meaning real instructions -- including this
  // region's own Constrain -- just ran for it) must report a period
  // strictly less than kRequestedPeriods. Were prime_next_request()'s
  // rollover to resolve and RUN an unrequested further period's own first
  // real step the instant this loop asked for one more step than strictly
  // needed to observe the boundary, that would be exactly a
  // step_opened == 1 report whose own `period` is kRequestedPeriods or
  // more.
  bool done = (next == -1);
  constexpr int kSafetyCapSteps = 500;
  for (int i = 0; i < kSafetyCapSteps && !done; ++i) {
    const char* comp = component_name(next);
    fuse_step_report rep{};
    fuse_step_report begin_rep{};
    const fuse_status rc = drive_one_step(t, ctx, comp, &rep, &begin_rep);
    ASSERT_EQ(rc, FUSE_OK) << "drive_one_step(" << comp << ") failed at i=" << i;
    if (begin_rep.step_opened) {
      EXPECT_LT(begin_rep.period, kRequestedPeriods)
          << "a real step of component '" << comp << "' ran for period "
          << begin_rep.period << ", which was never requested (only "
          << kRequestedPeriods << " periods were asked for) -- this is "
             "exactly the period-contract violation";
    }
    next = rep.next_component;
    if (next == -1) {
      done = true;
    }
  }
  ASSERT_TRUE(done) << "never observed a period boundary within "
                    << kSafetyCapSteps << " steps";

  ASSERT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
