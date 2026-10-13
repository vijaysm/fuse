// Review 2026-10-11 AB-9: by I4 ("a refusal that no test makes fire does
// not count as implemented"), ~20 participant-ABI refusal codes had no
// firing test anywhere in the tree, view_field() had no test at all in
// test/abi/, and neither fuse_coupler_abi()'s own version refusal nor a
// double fuse_destroy() was exercised. One small, mostly single-assertion
// test per code, grouped by entry point, in the same style
// test/abi/participant_test.cpp's own lifecycle tests use.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/runtime/component.hpp"

#include "graph_fixture.hpp"
#include "test_support.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// Binds storage and does nothing else -- exactly participant_test.cpp's
/// own TrivialComponent, duplicated here (private to this TU, like every
/// other test-only Component in test/abi/*.cpp) so a component can be
/// driven fully in-process, leaving no externally-driven component behind
/// for fuse_run()'s own E-ABI-RUN-EXTERNAL check to object to.
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

/// A graph where BOTH components are driven in-process (TrivialComponent),
/// so fuse_run() never refuses E-ABI-RUN-EXTERNAL -- the fixture
/// RunPeriodsBelowMinusOneIsRefused/RunNoEndTimeIsRefused need to reach
/// their own checks, further down impl_run(), instead.
fuse_ctx* CreateLoadedAndFullyInProcess(const fuse_participant_abi_v1* t) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  EXPECT_TRUE(fuse::abi_test::inject_component(
      ctx, "upstream", std::make_unique<TrivialComponent>("upstream")));
  EXPECT_TRUE(fuse::abi_test::inject_component(
      ctx, "downstream", std::make_unique<TrivialComponent>("downstream")));
  return ctx;
}

void ExpectLastErrorCode(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                         const char* expected_code) {
  fuse_error err{.struct_size = sizeof(err)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, expected_code);
}

}  // namespace

// ---- fuse_load_graph --------------------------------------------------

TEST(AbiRefusalSweep, LoadGraphOnAnAlreadyLoadedContextIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-LOAD-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, LoadGraphRefusesAnAmbiguousSource) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;

  fuse_graph_source src{.struct_size = sizeof(src)};  // neither yaml_path nor bytes set
  EXPECT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-LOAD-SOURCE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_save_graph ---------------------------------------------------

TEST(AbiRefusalSweep, SaveGraphBeforeAnyGraphIsLoadedIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;

  fuse_graph_sink sink{.struct_size = sizeof(sink)};
  EXPECT_EQ(t->save_graph(ctx, nullptr, &sink), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-SAVE-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_setup_options --------------------------------------------------

TEST(AbiRefusalSweep, SetupRefusesAnMpmdComponentWithNoCommunicator) {
  // Review 2026-10-11 AB-4: my_component's own handling lived entirely
  // inside `if (in->communicator != 0)`, so a caller that named an
  // Mpmd/Connect component but left communicator == 0 fell through
  // silently -- the run proceeded single-process on a graph that
  // declares a multi-executable launch, with no refusal at all.
  fuse::CouplingGraph g;
  fuse::ComponentDecl decl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}};
  decl.launch.mode = fuse::LaunchMode::Mpmd;
  decl.launch.appnum = 0;
  g.add_component(decl);
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = 0;
  opts.has_end_time = 0;
  opts.my_component = "upstream";
  EXPECT_EQ(t->setup(ctx, &opts, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-SETUP-COMPONENT");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, SetupRefusesANonZeroReservedField) {
  // Review 2026-10-11 AB-11: entries.def documents fuse_setup_options::
  // reserved as "must be 0"; nothing checked it until this round's fix.
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = 0;
  opts.has_end_time = 0;
  opts.reserved = 1;
  EXPECT_EQ(t->setup(ctx, &opts, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-SETUP-RESERVED");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_initialize / fuse_is_coupling_ongoing -------------------------

TEST(AbiRefusalSweep, InitializeBeforeAnyGraphIsLoadedIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;

  EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-INIT-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, IsCouplingOngoingBeforeInitializeIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_ongoing out{.struct_size = sizeof(out)};
  EXPECT_EQ(t->is_coupling_ongoing(ctx, nullptr, &out), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-ONGOING-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_field_handle ---------------------------------------------------

TEST(AbiRefusalSweep, FieldHandleBeforeAnyGraphIsLoadedIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-FIELD-HANDLE-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, FieldHandleNullNameIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = nullptr;
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-FIELD-HANDLE-NULL");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, FieldHandleUnknownComponentIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "not-a-real-component";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_ERR_NOTFOUND);
  ExpectLastErrorCode(t, ctx, "E-ABI-FIELD-HANDLE-COMPONENT");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, FieldHandleUnknownFieldIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "not-a-real-field";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_ERR_NOTFOUND);
  ExpectLastErrorCode(t, ctx, "E-ABI-FIELD-HANDLE-FIELD");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_register_field ------------------------------------------------

TEST(AbiRefusalSweep, RegisterFieldBeforeAnyGraphIsLoadedIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;

  double storage[4] = {0, 0, 0, 0};
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = 0;
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = 1;
  fview.data = storage;
  EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-REGISTER-STATE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

class AbiRegisterFieldRefusal : public testing::Test {
 protected:
  void SetUp() override {
    t = Table();
    ASSERT_NE(t, nullptr);
    const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
    ctx = CreateAndLoad(t, bytes);
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = "upstream";
    fname.field = "out";
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    fview = fuse_field_view{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = 4;
    fview.owned_count = 4;
    fview.stride = 1;
    fview.data = storage;
  }
  void TearDown() override { EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK); }

  const fuse_participant_abi_v1* t = nullptr;
  fuse_ctx* ctx = nullptr;
  double storage[4] = {1, 2, 3, 4};
  fuse_field_view fview{};
};

TEST_F(AbiRegisterFieldRefusal, NonF64ElementIsRefused) {
  fview.element_type = FUSE_ELEMENT_I64;
  EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_ERR_UNSUPPORTED);
  ExpectLastErrorCode(t, ctx, "E-ABI-REGISTER-ELEMENT");
}

TEST_F(AbiRegisterFieldRefusal, NonHostMemspaceIsRefused) {
  fview.memspace = FUSE_MEMSPACE_DEVICE;
  EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_ERR_UNSUPPORTED);
  ExpectLastErrorCode(t, ctx, "E-ABI-REGISTER-MEMSPACE");
}

TEST_F(AbiRegisterFieldRefusal, GhostedOwnershipIsRefused) {
  fview.ownership = FUSE_OWNERSHIP_GHOSTED;
  EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_ERR_UNSUPPORTED);
  ExpectLastErrorCode(t, ctx, "E-ABI-REGISTER-OWNERSHIP");
}

TEST_F(AbiRegisterFieldRefusal, OwnedCountMismatchIsRefused) {
  fview.owned_count = fview.count - 1;
  EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-REGISTER-OWNED-COUNT");
}

// ---- fuse_view_field: no test exercised this entry AT ALL (AB-9) -------

TEST(AbiRefusalSweep, ViewFieldRefusesAHandleOutsideTheLoadedGraph) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;  // no graph loaded at all: field_count() == 0

  fuse_field_ref in{.struct_size = sizeof(in)};
  in.handle = 0;
  fuse_field_view out{.struct_size = sizeof(out)};
  EXPECT_EQ(t->view_field(ctx, &in, &out), FUSE_ERR_NOTFOUND);
  ExpectLastErrorCode(t, ctx, "E-ABI-VIEW-FIELD-HANDLE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, ViewFieldReturnsTheRegisteredStorage) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);

  double storage[4] = {1, 2, 3, 4};
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
  fview.data = storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);

  fuse_field_ref in{.struct_size = sizeof(in)};
  in.handle = fref.handle;
  fuse_field_view out{.struct_size = sizeof(out)};
  ASSERT_EQ(t->view_field(ctx, &in, &out), FUSE_OK);
  EXPECT_EQ(out.data, static_cast<void*>(storage))
      << "before initialize(), view_field must hand back the exact "
         "buffer register_field() was given";
  EXPECT_EQ(out.count, 4u);
  EXPECT_EQ(out.components, 1);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_initialize: E-ABI-DELIVERY-INPROCESS --------------------------

TEST(AbiRefusalSweep, LowFiveBoundChannelWithAnInProcessImporterIsRefused) {
  // A minimal graph with one LowFive-bound channel (source field's own
  // hdf5_path set, entries.def/field_spec.hpp) whose importer
  // ("downstream") is injected in-process: ExternalDelivery's own
  // suspension (src/runtime/iteration.cpp) can only resume through
  // fuse_begin_step()/fuse_end_step(), which an in-process component
  // never calls, so this must be refused at fuse_initialize() rather
  // than left to hang.
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(
      fuse::MeshDecl{.name = "m1", .source = "generated:line:4"});
  const auto m2 = g.add_mesh(
      fuse::MeshDecl{.name = "m2", .source = "generated:line:4"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto down = g.add_component(fuse::ComponentDecl{
      .name = "downstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  // validate() (field_spec.hpp's own comment) requires at least one
  // component to declare LaunchMode::LowFive before any field may set
  // hdf5_path -- a bare placement (E-LOWFIVE-COMPONENT-KIND: kind=Coupler,
  // no fields), never one of the two physics components themselves.
  g.add_coupler(fuse::CouplerDecl{
      .name = "lowfive_marker", .ranks = fuse::RankSet{0, 1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  const auto out_f = g.add_field(
      up, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                          .units = fuse::units::kKelvin,
                          .conservation = fuse::Conservation::None,
                          .intent = fuse::Intent::Provides,
                          .hdf5_path = "refusal_sweep_test.h5/out"});
  const auto in_f = g.add_field(
      down, fuse::FieldSpec{.name = "in", .region = r2, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", out_f, in_f);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "downstream", std::make_unique<TrivialComponent>("downstream")));

  EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-DELIVERY-INPROCESS");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_begin_step / fuse_end_step ------------------------------------

class AbiStepRefusal : public testing::Test {
 protected:
  void SetUp() override {
    t = Table();
    ASSERT_NE(t, nullptr);
    const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
    ctx = CreateAndLoad(t, bytes);
  }
  void RegisterBothFields() {
    auto register_one = [&](const char* comp, const char* field, double* data) {
      fuse_field_name fname{.struct_size = sizeof(fname)};
      fname.component = comp;
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
  }
  void TearDown() override { EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK); }

  const fuse_participant_abi_v1* t = nullptr;
  fuse_ctx* ctx = nullptr;
  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {0, 0, 0, 0};
};

TEST_F(AbiStepRefusal, BeginStepNullComponentIsRefused) {
  RegisterBothFields();
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = nullptr;
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, &req, &rep), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-NULL");
}

TEST_F(AbiStepRefusal, BeginStepUnknownComponentIsRefused) {
  RegisterBothFields();
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "not-a-real-component";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, &req, &rep), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-COMPONENT");
}

TEST_F(AbiStepRefusal, BeginStepOnAnInProcessComponentIsRefused) {
  ASSERT_TRUE(fuse::abi_test::inject_component(
      ctx, "upstream", std::make_unique<TrivialComponent>("upstream")));
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "downstream";
  fname.field = "in";
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
  fview.data = down_storage;
  ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";  // driven in-process now, not externally
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, &req, &rep), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-IN-PROCESS");
}

TEST_F(AbiStepRefusal, BeginStepTwiceOnTheSameComponentIsRefused) {
  RegisterBothFields();
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  ASSERT_EQ(t->begin_step(ctx, &req, &rep), FUSE_OK);
  EXPECT_EQ(t->begin_step(ctx, &req, &rep), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-ALREADY-OPEN");
}

TEST_F(AbiStepRefusal, BeginStepOnADifferentComponentWhileOneIsOpenIsRefused) {
  RegisterBothFields();
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_report up_rep{.struct_size = sizeof(up_rep)};
  ASSERT_EQ(t->begin_step(ctx, &up_req, &up_rep), FUSE_OK);  // "upstream" now open

  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report down_rep{.struct_size = sizeof(down_rep)};
  EXPECT_EQ(t->begin_step(ctx, &down_req, &down_rep), FUSE_ERR_STATE)
      << "KNOWN_ISSUES.md D12: at most one externally driven component "
         "may be open at a time";
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-ALREADY-OPEN");
}

TEST_F(AbiStepRefusal, EndStepWithNoMatchingBeginStepIsRefused) {
  RegisterBothFields();
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->end_step(ctx, &req, &rep), FUSE_ERR_STATE);
  ExpectLastErrorCode(t, ctx, "E-ABI-STEP-NOT-OPEN");
}

// ---- fuse_run ------------------------------------------------------------

TEST(AbiRefusalSweep, RunPeriodsBelowMinusOneIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = CreateLoadedAndFullyInProcess(t);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_run_request req{.struct_size = sizeof(req)};
  req.periods = -2;
  EXPECT_EQ(t->run(ctx, &req, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-RUN-PERIODS");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiRefusalSweep, RunUntilEndTimeWithNoEndTimeSetIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = CreateLoadedAndFullyInProcess(t);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_run_request req{.struct_size = sizeof(req)};
  req.periods = -1;  // "until end_time" -- but fuse_setup_options() never ran
  EXPECT_EQ(t->run(ctx, &req, nullptr), FUSE_ERR_INVALID);
  ExpectLastErrorCode(t, ctx, "E-ABI-RUN-NO-END-TIME");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// ---- fuse_destroy: double destroy --------------------------------------

TEST(AbiRefusalSweep, DestroyingAnAlreadyDestroyedHandleIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  ASSERT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  // `ctx` is now a stale handle (the table entry is gone; with_ctx()
  // checks liveness before ever dereferencing it, so this is safe to
  // call, not a real use-after-free).
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_ERR_INVALID);
}

// ---- fuse_coupler_abi: the version refusal itself ------------------------

TEST(AbiRefusalSweep, CouplerAbiRefusesAnUnsupportedVersion) {
  EXPECT_EQ(fuse_coupler_abi(FUSE_ABI_VERSION + 1), nullptr);
  EXPECT_EQ(fuse_coupler_abi(0), nullptr);
  EXPECT_NE(fuse_coupler_abi(FUSE_ABI_VERSION), nullptr);
}
