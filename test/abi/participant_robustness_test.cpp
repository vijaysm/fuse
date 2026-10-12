// Ported from upstream/soma/test/unit/capi_test.c (docs/dev/soma-import.md,
// capi_test.c row): soma's flat C ABI had an explicit suite for its own
// defensive contract -- version is non-empty, a double destroy is reported
// rather than crashing, and the thread-local error slot is readable with no
// live context. include/fuse/abi/participant.h documents the SAME three
// contracts for fuse_coupler_abi() ("a double destroy is FUSE_ERR_INVALID,
// not a crash"; "ctx NULL asks the thread-local error of a failed create"),
// but every existing ABI test calls `last_error` on a live, non-NULL context
// (grepped across test/abi and test/verification) and none calls `destroy`
// twice or reads `version` into an otherwise-unused buffer. This file closes
// that specific gap: the defensive contract documented on the table itself,
// exercised directly, as distinct from the scheduling contract
// test/abi/participant_test.cpp already covers in depth.

#include "fuse/abi/participant.h"

#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

}  // namespace

TEST(AbiParticipantRobustness, VersionStringIsNonEmpty) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // "ctx may be NULL" (participant.h) -- the version query needs no live
  // context at all, matching soma's own Smoke.VersionIsNonEmpty.
  fuse_version_info info{.struct_size = sizeof(info)};
  ASSERT_EQ(t->version(nullptr, nullptr, &info), FUSE_OK);
  ASSERT_NE(info.string, nullptr);
  EXPECT_FALSE(std::string_view{info.string}.empty());
}

TEST(AbiParticipantRobustness, DoubleDestroyIsReportedNotCrashed) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  ASSERT_NE(co.ctx, nullptr);

  EXPECT_EQ(t->destroy(co.ctx, nullptr, nullptr), FUSE_OK);
  // participant.h's own comment: "a double destroy is FUSE_ERR_INVALID, not
  // a crash". The second call reuses the same (now-stale) handle on purpose.
  EXPECT_EQ(t->destroy(co.ctx, nullptr, nullptr), FUSE_ERR_INVALID);
}

TEST(AbiParticipantRobustness, ThreadLocalErrorReadableWithNoLiveContext) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  // participant.h: "ctx must be NULL" for create(). A non-NULL ctx is
  // exactly the malformed call that must fail AND populate the thread-local
  // slot -- there is no live context yet for a per-context slot to use.
  // `fuse_ctx` is an opaque (incomplete) type across this public header, so
  // the "non-NULL" value under test is a sentinel address, never
  // dereferenced: create()'s own check is `ctx != nullptr`, read before
  // anything would dereference it.
  fuse_ctx* const bogus = reinterpret_cast<fuse_ctx*>(std::uintptr_t{1});
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(bogus, nullptr, &co), FUSE_ERR_INVALID);

  // participant.h: "ctx NULL asks the thread-local error of a failed
  // create". No test anywhere else in this tree calls last_error with a
  // NULL ctx (grepped across test/abi, test/verification).
  fuse_error err{.struct_size = sizeof(err)};
  char buf[256] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  ASSERT_EQ(t->last_error(nullptr, nullptr, &err), FUSE_OK);
  const std::string_view message(buf);
  EXPECT_NE(message.find("must be NULL"), std::string_view::npos)
      << "last_error: " << message;
}
