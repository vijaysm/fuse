// ABI-GOLDEN-HEADERS, ABI-GOLDEN-SIZES, ABI-GROWTH (build-evidence/abi-matrix.toml).
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "fuse/abi/participant.h"
#include "fuse/abi/provider.h"

#ifndef FUSE_PYTHON3_EXECUTABLE
#define FUSE_PYTHON3_EXECUTABLE "python3"
#endif
#ifndef FUSE_SOURCE_DIR
#define FUSE_SOURCE_DIR "."
#endif

namespace {

/// Round `offset + size` up to the next multiple of `align`: the layout a
/// trivially-appended member would occupy if there were room for it without
/// moving anything else.
[[nodiscard]] constexpr std::size_t round_up(std::size_t v,
                                             std::size_t align) noexcept {
  return ((v + align - 1) / align) * align;
}

}  // namespace

// AbiGolden.HeadersMatchTheEntryList: types.h and participant.h are exactly
// what tools/gen-abi.py would regenerate from entries.def. Fails on any
// hand edit to either header, or a .def change nobody regenerated for.
TEST(AbiGolden, HeadersMatchTheEntryList) {
  const std::string cmd = std::string(FUSE_PYTHON3_EXECUTABLE) + " " +
                          FUSE_SOURCE_DIR + "/tools/gen-abi.py --check";
  const int rc = std::system(cmd.c_str());
  ASSERT_NE(rc, -1) << "could not spawn: " << cmd;
#if defined(_WIN32)
  const int exit_code = rc;
#else
  const int exit_code = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
#endif
  EXPECT_EQ(exit_code, 0)
      << "tools/gen-abi.py --check reported stale generated headers; run "
         "`python3 tools/gen-abi.py --write` and commit the result ("
      << cmd << ")";
}

// AbiGolden.StructSizesMatchTheTable: every argument struct's sizeof is the
// committed value in struct_sizes.golden, and sizeof(void*) == 8 (rule 3's
// "a member fits in trailing padding" hazard is only meaningful on a 64-bit
// pointer ABI).
TEST(AbiGolden, StructSizesMatchTheTable) {
  const std::unordered_map<std::string, std::size_t> measured = {
      {"fuse_null", sizeof(fuse_null)},
      {"fuse_version_info", sizeof(fuse_version_info)},
      {"fuse_capability_set", sizeof(fuse_capability_set)},
      {"fuse_context_out", sizeof(fuse_context_out)},
      {"fuse_error", sizeof(fuse_error)},
      {"fuse_graph_source", sizeof(fuse_graph_source)},
      {"fuse_graph_sink", sizeof(fuse_graph_sink)},
      {"fuse_setup_options", sizeof(fuse_setup_options)},
      {"fuse_field_name", sizeof(fuse_field_name)},
      {"fuse_field_ref", sizeof(fuse_field_ref)},
      {"fuse_field_view", sizeof(fuse_field_view)},
      {"fuse_step_request", sizeof(fuse_step_request)},
      {"fuse_step_report", sizeof(fuse_step_report)},
      {"fuse_ongoing", sizeof(fuse_ongoing)},
      {"fuse_run_request", sizeof(fuse_run_request)},
      {"fuse_base_communicator_info", sizeof(fuse_base_communicator_info)},
      {"fuse_participant_abi_v1", sizeof(fuse_participant_abi_v1)},
      {"fuse_field_spec", sizeof(fuse_field_spec)},
      {"fuse_field_table", sizeof(fuse_field_table)},
      {"fuse_provider_instance", sizeof(fuse_provider_instance)},
      {"fuse_provider_abi_v1", sizeof(fuse_provider_abi_v1)},
      {"fuse_restart_blob_sink", sizeof(fuse_restart_blob_sink)},
      {"fuse_restart_blob_source", sizeof(fuse_restart_blob_source)},
  };

  std::ifstream golden(std::string(FUSE_TEST_GOLDEN_DIR) + "/../abi/struct_sizes.golden");
  ASSERT_TRUE(golden.is_open()) << "could not open test/abi/struct_sizes.golden";
  std::string name;
  std::size_t expected = 0;
  int lines = 0;
  while (golden >> name >> expected) {
    ++lines;
    if (name == "sizeof(void*)") {
      EXPECT_EQ(sizeof(void*), expected) << "sizeof(void*)";
      continue;
    }
    const auto it = measured.find(name);
    ASSERT_NE(it, measured.end())
        << "struct_sizes.golden names '" << name
        << "', which this test does not measure -- add it to `measured` "
           "above";
    EXPECT_EQ(it->second, expected) << name;
  }
  EXPECT_EQ(lines, static_cast<int>(measured.size() + 1))
      << "struct_sizes.golden has a different number of lines than this "
         "test expects (measured structs + sizeof(void*))";
}

// AbiGolden.NoStructHasTrailingPaddingALaterMemberCouldHide: for every
// struct, sizeof(struct) must equal offsetof(last member) + sizeof(last
// member) rounded up to the struct's own alignment -- i.e. there is no
// trailing padding a later 4-byte append could silently vanish into without
// changing struct_size (abi-design.md rule 3; fuse_version_info was the one
// case this caught and carries its `reserved` member because of it).
#define FUSE_NO_TRAILING_PADDING(Type, last)                                 \
  EXPECT_EQ(sizeof(Type),                                                    \
            round_up(offsetof(Type, last) + sizeof(((Type*)nullptr)->last),  \
                    alignof(Type)))                                          \
      << #Type                                                               \
      << ": trailing padding exists after `" << #last                       \
      << "` -- a later appended member could land in it without changing "  \
         "struct_size (rule 3)"

TEST(AbiGolden, NoStructHasTrailingPaddingALaterMemberCouldHide) {
  FUSE_NO_TRAILING_PADDING(fuse_null, struct_size);
  FUSE_NO_TRAILING_PADDING(fuse_version_info, string);
  FUSE_NO_TRAILING_PADDING(fuse_capability_set, flags);
  FUSE_NO_TRAILING_PADDING(fuse_context_out, ctx);
  FUSE_NO_TRAILING_PADDING(fuse_error, code);
  FUSE_NO_TRAILING_PADDING(fuse_graph_source, byte_count);
  FUSE_NO_TRAILING_PADDING(fuse_graph_sink, written);
  FUSE_NO_TRAILING_PADDING(fuse_setup_options, my_component);
  FUSE_NO_TRAILING_PADDING(fuse_field_name, field);
  FUSE_NO_TRAILING_PADDING(fuse_field_ref, conservation);
  FUSE_NO_TRAILING_PADDING(fuse_field_view, window_samples);
  FUSE_NO_TRAILING_PADDING(fuse_step_request, reserved);
  FUSE_NO_TRAILING_PADDING(fuse_step_report, delivery_field);
  FUSE_NO_TRAILING_PADDING(fuse_ongoing, next_component);
  FUSE_NO_TRAILING_PADDING(fuse_run_request, periods);
  FUSE_NO_TRAILING_PADDING(fuse_base_communicator_info, reserved);
  FUSE_NO_TRAILING_PADDING(fuse_participant_abi_v1, request_restart_point);
  FUSE_NO_TRAILING_PADDING(fuse_field_spec, reserved);
  FUSE_NO_TRAILING_PADDING(fuse_field_table, written);
  FUSE_NO_TRAILING_PADDING(fuse_provider_instance, self);
  FUSE_NO_TRAILING_PADDING(fuse_provider_abi_v1, load_restart_blob);
}
#undef FUSE_NO_TRAILING_PADDING
