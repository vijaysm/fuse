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
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/coupling_graph.hpp"

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
