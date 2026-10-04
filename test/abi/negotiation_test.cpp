// ABI-TABLE-NEGOTIATION, ABI-FAULT-INJECTION (build-evidence/abi-matrix.toml).
//
// Both rows are pure struct-layout / table-substitution exercises: nothing
// here needs the library's internals, only the one exported symbol and the
// public headers -- exactly what abi-design.md point 6 ("a test substitutes
// its own table to inject a fault at one entry point, with no linker tricks
// and no rebuild") promises.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <vector>

#include "fuse/abi/participant.h"

#include "graph_fixture.hpp"

namespace {

/// A hand-built "v2" table: the EXACT member-for-member prefix of
/// fuse_participant_abi_v1 (via decltype, so it cannot silently drift from
/// the real layout) plus one appended entry, exactly the shape
/// abi-design.md rule 1 says a real v2 would have.
struct FakeV2Table {
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  decltype(fuse_participant_abi_v1::version) version;
  decltype(fuse_participant_abi_v1::capabilities) capabilities;
  decltype(fuse_participant_abi_v1::create) create;
  decltype(fuse_participant_abi_v1::destroy) destroy;
  decltype(fuse_participant_abi_v1::last_error) last_error;
  decltype(fuse_participant_abi_v1::load_graph) load_graph;
  decltype(fuse_participant_abi_v1::save_graph) save_graph;
  decltype(fuse_participant_abi_v1::setup) setup;
  decltype(fuse_participant_abi_v1::field_handle) field_handle;
  decltype(fuse_participant_abi_v1::register_field) register_field;
  decltype(fuse_participant_abi_v1::initialize) initialize;
  decltype(fuse_participant_abi_v1::is_coupling_ongoing) is_coupling_ongoing;
  decltype(fuse_participant_abi_v1::begin_step) begin_step;
  decltype(fuse_participant_abi_v1::end_step) end_step;
  decltype(fuse_participant_abi_v1::run) run;
  // Unit 3 (S6 phase 3, Conformance), appended 2026-10-04 to entries.def
  // LAST (the table's own "appended to, never reordered" rule) -- this
  // fixture's own top comment requires matching the real member order
  // exactly, or the static_assert below (sizeof(FakeV2Table) > sizeof
  // (fuse_participant_abi_v1)) would start failing the moment the real
  // table grows to the same entry count this fixture always enumerated
  // by hand.
  decltype(fuse_participant_abi_v1::view_field) view_field;
  // The appended v2 member. Never read by a v1 caller (struct_size says it
  // isn't there) and never read by this test's "v2 caller" EITHER, when the
  // table it was handed turns out to be a real v1 one -- see
  // V2CallerAgainstV1Table, which checks struct_size before it would
  // dereference this.
  fuse_status (*extra_entry)(fuse_ctx*, const fuse_null*, fuse_null*);
};

static_assert(offsetof(FakeV2Table, version) == offsetof(fuse_participant_abi_v1, version),
             "FakeV2Table's prefix must byte-match fuse_participant_abi_v1");
static_assert(sizeof(FakeV2Table) > sizeof(fuse_participant_abi_v1));

fuse_status StubExtraEntry(fuse_ctx*, const fuse_null*, fuse_null*) {
  return FUSE_ERR_UNSUPPORTED;  // never actually reached by either test
}

fuse_status StubBeginStepRefuses(fuse_ctx*, const fuse_step_request*,
                                 fuse_step_report*) {
  return FUSE_ERR_RUNTIME;
}

fuse_ctx* CreateAndLoad(const fuse_participant_abi_v1* t) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(co.ctx, &src, nullptr), FUSE_OK);
  return co.ctx;
}

/// Drives the two-component lag1 graph's first period through `t` far
/// enough to prove `t` works end to end: both fields registered,
/// initialize() succeeded, one full begin_step/end_step round on each
/// component.
void DriveOnePeriod(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                    double* up_storage, double* down_storage) {
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
  fuse_step_report rep{.struct_size = sizeof(rep)};
  ASSERT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);

  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  ASSERT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  ASSERT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
}

}  // namespace

// AbiNegotiation.V1CallerAgainstV2Table: a v1 caller, using nothing but the
// fuse_participant_abi_v1 type, drives the graph through a hand-built v2
// table (one appended entry) without ever knowing it is not v1.
TEST(AbiNegotiation, V1CallerAgainstV2Table) {
  const auto* real = fuse_coupler_abi(FUSE_ABI_VERSION);
  ASSERT_NE(real, nullptr);

  FakeV2Table v2{};
  std::memcpy(&v2, real, sizeof(fuse_participant_abi_v1));
  v2.struct_size = sizeof(FakeV2Table);
  v2.abi_version = 2;
  v2.extra_entry = StubExtraEntry;

  const auto* as_v1 = reinterpret_cast<const fuse_participant_abi_v1*>(&v2);
  fuse_ctx* ctx = CreateAndLoad(as_v1);
  double up_storage[4] = {1, 1, 1, 1};
  double down_storage[4] = {0, 0, 0, 0};
  DriveOnePeriod(as_v1, ctx, up_storage, down_storage);
  EXPECT_EQ(as_v1->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiNegotiation.V2CallerAgainstV1Table: a v2 caller sees struct_size <
// sizeof(its own, larger table), and must fall back to the v1 subset
// without ever dereferencing the appended member -- which, on a REAL v1
// table, is memory this test does not own and must not read.
TEST(AbiNegotiation, V2CallerAgainstV1Table) {
  const auto* real = fuse_coupler_abi(FUSE_ABI_VERSION);
  ASSERT_NE(real, nullptr);
  // The table's own struct_size, as the library's X-macro-generated
  // initialiser writes it (src/abi/participant.cpp): exactly
  // sizeof(fuse_participant_abi_v1), never 0 and never any other value --
  // the negotiation this whole test exercises has nothing to negotiate
  // against if the writer gets this field wrong.
  EXPECT_EQ(real->struct_size, sizeof(fuse_participant_abi_v1));
  const auto* as_v2 = reinterpret_cast<const FakeV2Table*>(real);

  ASSERT_LT(as_v2->struct_size, sizeof(FakeV2Table))
      << "a real v1 table's struct_size must be sizeof(fuse_participant_abi_v1), "
         "which is smaller than this test's v2 table -- otherwise the check "
         "below would wrongly call extra_entry on memory that is not there";

  // Negotiation: only the v1 subset is used. The whole two-component graph
  // runs correctly through it.
  const auto* as_v1 = reinterpret_cast<const fuse_participant_abi_v1*>(as_v2);
  fuse_ctx* ctx = CreateAndLoad(as_v1);
  double up_storage[4] = {2, 2, 2, 2};
  double down_storage[4] = {0, 0, 0, 0};
  DriveOnePeriod(as_v1, ctx, up_storage, down_storage);
  EXPECT_EQ(as_v1->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiNegotiation.ASubstitutedEntryIsCalledInsteadOfTheLibrarys
// (ABI-FAULT-INJECTION): copy the real table by value, replace one entry,
// and confirm the substitute -- not the library's own implementation -- is
// what runs.
TEST(AbiNegotiation, ASubstitutedEntryIsCalledInsteadOfTheLibrarys) {
  const auto* real = fuse_coupler_abi(FUSE_ABI_VERSION);
  ASSERT_NE(real, nullptr);

  fuse_participant_abi_v1 faulty = *real;
  faulty.begin_step = StubBeginStepRefuses;

  fuse_ctx* ctx = CreateAndLoad(&faulty);
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
  double up_storage[4] = {0, 0, 0, 0};
  double down_storage[4] = {0, 0, 0, 0};
  fname.component = "upstream";
  fname.field = "out";
  ASSERT_EQ(faulty.field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = up_storage;
  ASSERT_EQ(faulty.register_field(ctx, &fview, nullptr), FUSE_OK);
  fname.component = "downstream";
  fname.field = "in";
  ASSERT_EQ(faulty.field_handle(ctx, &fname, &fref), FUSE_OK);
  fview.handle = fref.handle;
  fview.data = down_storage;
  ASSERT_EQ(faulty.register_field(ctx, &fview, nullptr), FUSE_OK);
  ASSERT_EQ(faulty.initialize(ctx, nullptr, nullptr), FUSE_OK);

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "upstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  // The REAL library's begin_step would succeed here (it is the same
  // sequence DriveOnePeriod already exercises above); the substitute always
  // refuses, and that refusal is what this test is checking for.
  EXPECT_EQ(faulty.begin_step(ctx, &req, &rep), FUSE_ERR_RUNTIME);

  EXPECT_EQ(faulty.destroy(ctx, nullptr, nullptr), FUSE_OK);
}
