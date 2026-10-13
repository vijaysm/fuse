#include <string>
#include <gtest/gtest.h>
#include <fuse/ir/dump.hpp>

#include <clocale>
#include <cstring>
#include <locale>

using fuse::ArcId;
using fuse::ChannelId;
using fuse::OperatorDAG;
using fuse::OpAttrs;
using fuse::OpKind;
using fuse::RegionId;
using fuse::ValueType;
namespace u = fuse::units;

namespace {

OperatorDAG small_dag() {
  OperatorDAG d;
  ValueType t{.region = RegionId{0}, .units = u::kHeatFluxDensity,
              .conservation = fuse::Conservation::Extensive, .rank_set = 0};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "fluid.heat_flux");
  d.add_node(OpKind::Accumulate, {a0}, a1,
             OpAttrs{.accum = fuse::AccumOp::Mean}, ChannelId{0},
             "flux/accumulate");
  return d;
}

}  // namespace

TEST(IrDump, StartsWithVersionBanner) {
  const std::string s = dump_text(small_dag());
  EXPECT_EQ(s.substr(0, 12), "# fuse-ir v1");
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
      .name = "outer", .body = {fuse::NodeId{1}}, .residuals = {ArcId{1}},
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
  // Review 2026-10-11 IS-11: legacy upstream residue in a public output
  // artifact, same class R-46 fixed for the LowFive probe dataset name
  // (src/transport/lowfive_transport.cpp) -- a different, still-open
  // instance of that class (TR-13) is not this file's concern.
  EXPECT_NE(s.find("digraph fuse_ir"), std::string::npos);
  EXPECT_EQ(s.find("soma"), std::string::npos);
}

// ---- locale independence ------------------------------------------------

namespace {

// A locale that groups thousands and uses a comma decimal point, built from a
// facet rather than a system locale name so the test does not depend on which
// locales happen to be generated on the machine (`de_DE.UTF-8` is absent on
// most CI images, and a test that silently skips proves nothing).
struct CommaDecimalGrouping : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

class GlobalLocale {
 public:
  explicit GlobalLocale(const std::locale& l) : previous_(std::locale::global(l)) {}
  ~GlobalLocale() { std::locale::global(previous_); }
  GlobalLocale(const GlobalLocale&) = delete;
  GlobalLocale& operator=(const GlobalLocale&) = delete;

 private:
  std::locale previous_;
};

}  // namespace

// The IR dump is a byte-compared golden artifact, so it must not depend on the
// host application's locale. Without the classic-locale pin, an integer prints
// as `16.384` and a double as `0,7` under the locale below -- and the process
// locale is the HOST's choice, not ours: a coupler linked into an application
// that called setlocale() would produce different golden bytes on the same data.
TEST(IrDump, TextIsIdenticalUnderAGroupingLocale) {
  // A non-default omega (a double) and element counts large enough to be
  // grouped are what the two hazards act on.
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "src");
  d.add_node(OpKind::Relax, {a0}, a1, OpAttrs{.omega = 0.7}, ChannelId{}, "relax");

  const std::string reference = dump_text(d);
  EXPECT_NE(reference.find("0.7"), std::string::npos) << reference;

  const GlobalLocale scoped{std::locale(std::locale::classic(), new CommaDecimalGrouping)};
  EXPECT_EQ(dump_text(d), reference)
      << "the dump changed under a grouping locale, so some value formats through "
         "the global locale the HOST application chose";
  // No comma check here: the dump's own attribute separator is ", ", so a
  // comma in the text is normal. Byte-equality against the reference is the
  // property that matters, and it is the one a locale change would break.
  EXPECT_NE(dump_text(d).find("0.7"), std::string::npos)
      << "the decimal point itself came from the locale";
}
