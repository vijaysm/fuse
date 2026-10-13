// ABI-SHORT-STRUCT, ABI-LONG-STRUCT (build-evidence/abi-matrix.toml).
//
// Both rows are about `struct_size` as a FIELD VALUE, not about how much
// memory the caller actually allocated: a real fuse_setup_options/fuse_capability_set
// object is always fully, safely constructed here, and only its
// `struct_size` member is overwritten to the value under test. That is
// exactly rule 2's contract -- the callee decides from the number the
// caller wrote, never from the caller's real allocation, which this test
// has no way to inspect anyway.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "graph_fixture.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  const auto* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

/// A loaded (but not yet initialized) context: enough state for
/// fuse_setup_options()'s struct_size behaviour, nothing more.
fuse_ctx* LoadedContext(const fuse_participant_abi_v1* t) {
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  const std::vector<std::byte> bytes = fuse::serialize(fuse::CouplingGraph{});
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(co.ctx, &src, nullptr), FUSE_OK);
  return co.ctx;
}

}  // namespace

TEST(AbiStructSize, ShortStructIsRefused) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // The OUT direction: fuse_capability_set, through an entry (ctx may be
  // NULL) that needs no live context at all.
  for (const std::uint32_t bad : {0u, 1u, 2u, 3u}) {
    fuse_capability_set caps{.struct_size = sizeof(fuse_capability_set)};
    caps.struct_size = bad;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION)
        << "struct_size = " << bad;
  }
  {
    fuse_capability_set caps{};
    caps.struct_size = FUSE_ABI_MAX_STRUCT_SIZE + 1;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION);
  }
  {
    fuse_capability_set caps{};
    caps.struct_size = 0xFFFFFFFFu;
    EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_ERR_VERSION);
  }

  // The IN direction: fuse_setup_options, on a real loaded context.
  fuse_ctx* ctx = LoadedContext(t);
  fuse_setup_options setup{.struct_size = sizeof(fuse_setup_options)};
  setup.communicator = 0;
  setup.has_end_time = 0;
  setup.struct_size  = 2;  // shorter than any frozen layout (>= 4)
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_ERR_VERSION);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiStructSize, LongerStructFromANewerCallerIsAccepted) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // OUT direction: a caller whose header declares fuse_capability_set with 64
  // extra trailing bytes this library's layout does not have must still get
  // a correct answer in the bytes this library DOES know about.
  fuse_capability_set caps{.struct_size = sizeof(fuse_capability_set) + 64};
  EXPECT_EQ(t->capabilities(nullptr, nullptr, &caps), FUSE_OK);
  EXPECT_NE(caps.flags, 0u) << "FUSE_CAP_YAML is always set";

  // IN direction: fuse_setup_options, same story.
  fuse_ctx* ctx = LoadedContext(t);
  fuse_setup_options setup{.struct_size = sizeof(fuse_setup_options) + 64};
  setup.communicator = 0;
  setup.has_end_time = 0;
  EXPECT_EQ(t->setup(ctx, &setup, nullptr), FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiStructSize.OldCallerStructSizeNeverReadsMyComponent (fuse_setup_options'
// own growth, examples/brusselator/participants: Conformance Unit 2).
// Heap-allocates EXACTLY the pre-growth byte count -- the same
// offsetof-derived expression src/abi/participant.cpp's own
// FirstFrozenStructSize<fuse_setup_options> specialization uses -- so a
// read past it is a genuine heap-buffer-overflow, not a benign read of
// slack the allocator happened to round up to. An old caller's buffer
// really is exactly this size; my_component did not exist when it was
// compiled, so fuse_setup() must never read the bytes where it would live.
TEST(AbiStructSize, OldCallerStructSizeNeverReadsMyComponent) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_setup_options, end_time) + sizeof(fuse_exact_time);
  static_assert(kOldSize < sizeof(fuse_setup_options),
               "my_component must have grown fuse_setup_options (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_ctx* ctx = LoadedContext(t);

  auto buf = std::make_unique<std::byte[]>(kOldSize);
  auto* old_opts = reinterpret_cast<fuse_setup_options*>(buf.get());
  old_opts->struct_size  = static_cast<std::uint32_t>(kOldSize);
  old_opts->communicator = 0;
  old_opts->has_end_time = 0;
  old_opts->reserved     = 0;
  old_opts->end_time     = fuse_exact_time{.numerator = 0, .denominator = 1};
  EXPECT_EQ(t->setup(ctx, old_opts, nullptr), FUSE_OK);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// AbiStructSize.OldCallerFuseStepRequestIsAccepted (Review 2026-10-11 AB-2):
// FirstFrozenStructSize<fuse_step_request> was missing, so check_struct_size
// refused struct_size=16 (the layout shipped before t_begin/t_end/step/
// iteration/requires_checkpoint/requires_restore/reserved were appended in
// 612b153, per c-abi.md's growth table "16 -> 72") with FUSE_ERR_VERSION,
// even though begin_step()/end_step() never read any of the grown members
// (entries.def documents them provider-direction-only). Heap-allocates
// EXACTLY the pre-growth byte count, mirroring
// OldCallerStructSizeNeverReadsMyComponent above, so a read past it is a
// genuine heap-buffer-overflow.
TEST(AbiStructSize, OldCallerFuseStepRequestIsAccepted) {
  constexpr std::size_t kOldSize =
      offsetof(fuse_step_request, component) + sizeof(const char*);
  static_assert(kOldSize < sizeof(fuse_step_request),
               "the provider-direction fields must have grown "
               "fuse_step_request (rule 3)");

  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  double up_storage[4] = {1, 2, 3, 4};
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

  auto buf = std::make_unique<std::byte[]>(kOldSize);
  auto* old_req = reinterpret_cast<fuse_step_request*>(buf.get());
  old_req->struct_size = static_cast<std::uint32_t>(kOldSize);
  old_req->component   = "upstream";

  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, old_req, &rep), FUSE_OK);
  EXPECT_EQ(rep.step, 0);

  old_req->component = "upstream";  // end_step's own struct_size check
  fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
  EXPECT_EQ(t->end_step(ctx, old_req, &end_rep), FUSE_OK);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
