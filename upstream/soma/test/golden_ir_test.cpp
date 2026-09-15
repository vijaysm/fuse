#include <gtest/gtest.h>
#include <soma/graph/yaml_io.hpp>
#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in{path};
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out{path};
  out << text;
}

bool updating() {
  const char* v = std::getenv("SOMA_UPDATE_GOLDEN");
  return v != nullptr && std::string{v} == "1";
}

/// Lower test/data/<name>.yaml and compare with test/golden/<name>.ir.
void check_golden(const std::string& name) {
  const std::string yaml_path =
      std::string{SOMA_TEST_DATA_DIR} + "/" + name + ".yaml";
  const std::string golden_path =
      std::string{SOMA_TEST_GOLDEN_DIR} + "/" + name + ".ir";

  const auto graph = soma::load_yaml_file(yaml_path);
  ASSERT_TRUE(graph.ok()) << graph.error().message;

  const auto result = lower(graph.value());
  ASSERT_TRUE(result.ok()) << result.diags.to_string();

  const std::string actual = dump_text(result.dag);

  if (updating()) {
    write_file(golden_path, actual);
    GTEST_SKIP() << "golden file updated: " << golden_path;
  }

  const std::string expected = read_file(golden_path);
  ASSERT_FALSE(expected.empty())
      << "missing golden file " << golden_path
      << "\nrun with SOMA_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual, expected)
      << "IR changed. If intentional, run:\n"
      << "  SOMA_UPDATE_GOLDEN=1 ctest --test-dir build -R GoldenIr\n"
      << "and review the diff.";
}

}  // namespace

TEST(GoldenIr, PairExplicit)             { check_golden("pair_explicit"); }
TEST(GoldenIr, PairAccumulateRemapRelax) { check_golden("pair_accumulate_remap_relax"); }
TEST(GoldenIr, PairIterated)             { check_golden("pair_iterated"); }

// Spec §2.1: the architecture's central claim is that operator order is
// explicit and therefore checkable. If these two dumps were ever equal, the
// IR would not be recording what it promises to record.
TEST(GoldenIr, OperatorOrderIsObservable) {
  const auto a = soma::load_yaml_file(
      std::string{SOMA_TEST_DATA_DIR} + "/pair_accumulate_remap_relax.yaml");
  ASSERT_TRUE(a.ok()) << a.error().message;
  const auto b = soma::load_yaml_file(
      std::string{SOMA_TEST_DATA_DIR} + "/pair_explicit.yaml");
  ASSERT_TRUE(b.ok()) << b.error().message;

  const std::string da = dump_text(lower(a.value()).dag);
  const std::string db = dump_text(lower(b.value()).dag);
  EXPECT_NE(da, db);
}
