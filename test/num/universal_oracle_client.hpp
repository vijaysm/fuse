// The client side of test/num/universal_oracle.cpp, shared by the tests that compare
// fuse::num types with stillwater-sc/universal (dd_universal_test.cpp,
// posit_universal_test.cpp): a binary file of records goes to the oracle program and a
// file of results comes back. The oracle is a separate host-compiled program because
// Universal's headers do not compile under nvcc; see universal_oracle.cpp for the
// record layout and the operation codes.
#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#ifndef FUSE_UNIVERSAL_ORACLE
#error "FUSE_UNIVERSAL_ORACLE must name the built universal_oracle program"
#endif

namespace fuse_test::universal {

struct Record {
  std::uint32_t limbs;
  std::uint32_t op;
  double a[4];
  double b[4];
};
struct Result {
  double r[4];
};

/// Runs Universal on the records through the oracle program.
inline std::vector<Result> oracle_results(const std::vector<Record>& records) {
  namespace fs = std::filesystem;
  static int counter = 0;
  const std::string stem = (fs::temp_directory_path() / ("fuse_universal_oracle_" + std::to_string(::getpid()) + "_" +
                                                         std::to_string(counter++))).string();
  const std::string in = stem + ".in", out = stem + ".out";
  {
    std::FILE* f = std::fopen(in.c_str(), "wb");
    EXPECT_NE(f, nullptr);
    if (f == nullptr) return {};
    std::fwrite(records.data(), sizeof(Record), records.size(), f);
    std::fclose(f);
  }
  const std::string cmd = std::string("\"") + FUSE_UNIVERSAL_ORACLE + "\" \"" + in + "\" \"" + out + "\"";
  const int rc = std::system(cmd.c_str());
  EXPECT_EQ(rc, 0) << cmd;
  std::vector<Result> results(records.size());
  if (std::FILE* f = std::fopen(out.c_str(), "rb")) {
    const std::size_t got = std::fread(results.data(), sizeof(Result), results.size(), f);
    EXPECT_EQ(got, results.size());
    std::fclose(f);
  } else {
    ADD_FAILURE() << "the oracle wrote no output: " << cmd;
  }
  std::remove(in.c_str());
  std::remove(out.c_str());
  return results;
}

}  // namespace fuse_test::universal
