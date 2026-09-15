#include <gtest/gtest.h>
#include <soma/ir/dump.hpp>

#include <clocale>
#include <cstring>
#include <locale>

using soma::ArcId;
using soma::ChannelId;
using soma::OperatorDAG;
using soma::OpAttrs;
using soma::OpKind;
using soma::RegionId;
using soma::ValueType;
namespace u = soma::units;

namespace {

OperatorDAG small_dag() {
  OperatorDAG d;
  ValueType t{.region = RegionId{0}, .units = u::kHeatFluxDensity,
              .conservation = soma::Conservation::Extensive, .rank_set = 0};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "fluid.heat_flux");
  d.add_node(OpKind::Accumulate, {a0}, a1,
             OpAttrs{.accum = soma::AccumOp::Mean}, ChannelId{0},
             "flux/accumulate");
  return d;
}

}  // namespace

TEST(IrDump, StartsWithVersionBanner) {
  const std::string s = dump_text(small_dag());
  EXPECT_EQ(s.substr(0, 12), "# soma-ir v1");
}

TEST(IrDump, EmitsOneLinePerNodeInTopologicalOrder) {
  const std::string s = dump_text(small_dag());
  const auto src = s.find("Source");
  const auto acc = s.find("Accumulate");
  ASSERT_NE(src, std::string::npos);
  ASSERT_NE(acc, std::string::npos);
  EXPECT_LT(src, acc);
}

TEST(IrDump, ShowsArcIdsAndTypes) {
  const std::string s = dump_text(small_dag());
  EXPECT_NE(s.find("%0"), std::string::npos);
  EXPECT_NE(s.find("%1"), std::string::npos);
  EXPECT_NE(s.find("Extensive"), std::string::npos);
  EXPECT_NE(s.find("ranks=0"), std::string::npos);
}

TEST(IrDump, ShowsNonDefaultAttributesOnly) {
  const std::string s = dump_text(small_dag());
  EXPECT_NE(s.find("{accum=Mean}"), std::string::npos);
  // The Source node has all-default attrs, so no brace group on its line.
  const auto line_end = s.find('\n', s.find("Source"));
  const std::string src_line = s.substr(s.find("Source"), line_end - s.find("Source"));
  EXPECT_EQ(src_line.find('{'), std::string::npos);
}

TEST(IrDump, IsByteStableAcrossCalls) {
  const OperatorDAG d = small_dag();
  EXPECT_EQ(dump_text(d), dump_text(d));
}

TEST(IrDump, EmitsRegions) {
  OperatorDAG d = small_dag();
  d.add_region(OperatorDAG::IterationRegion{
      .name = "outer", .body = {soma::NodeId{1}}, .residuals = {ArcId{1}},
      .carried = {ArcId{0}}, .max_iterations = 30});
  const std::string s = dump_text(d);
  EXPECT_NE(s.find("region 0 \"outer\""), std::string::npos);
  EXPECT_NE(s.find("max_iter=30"), std::string::npos);
}

TEST(IrDump, GraphvizIsWellFormed) {
  const std::string s = dump_graphviz(small_dag());
  EXPECT_EQ(s.substr(0, 7), "digraph");
  EXPECT_NE(s.find("->"), std::string::npos);
  EXPECT_EQ(s.back(), '\n');
  EXPECT_NE(s.find('}'), std::string::npos);
}

// Task 2.8 diffs dump_text() byte-for-byte against checked-in golden files.
// std::to_string(double) formats through the global C locale, so under a
// comma-decimal locale (e.g. de_DE) a non-default omega would silently
// change the golden bytes on any machine/CI image with that locale active.
// This pins that the dump is locale-independent regardless of the ambient
// global locale. CI images vary in which locales are installed, so this
// skips (rather than fails) when de_DE isn't available.
TEST(IrDump, OmegaIsFormattedLocaleIndependently) {
  const char* kGermanLocale = "de_DE.UTF-8";
  const std::string previous = std::setlocale(LC_ALL, nullptr);

  if (!std::setlocale(LC_ALL, kGermanLocale)) {
    GTEST_SKIP() << kGermanLocale << " not available on this system";
  }

  OperatorDAG d = small_dag();
  // Overwrite the Accumulate node's attrs with a non-default omega by
  // rebuilding the dag with a Relax node carrying omega=0.7.
  OperatorDAG relax_dag;
  const ValueType t{.region = RegionId{0}};
  const ArcId a0 = relax_dag.add_arc(t, "v0");
  const ArcId a1 = relax_dag.add_arc(t, "v1");
  relax_dag.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "src");
  relax_dag.add_node(OpKind::Relax, {a0}, a1, OpAttrs{.omega = 0.7},
                      ChannelId{}, "relax");

  const std::string s = dump_text(relax_dag);

  std::setlocale(LC_ALL, previous.c_str());

  EXPECT_NE(s.find("0.7"), std::string::npos)
      << "expected locale-independent '0.7', got: " << s;
  EXPECT_EQ(s.find("0,7"), std::string::npos)
      << "dump leaked the German comma-decimal locale: " << s;
}
