// The authoring-chain rules the charter makes binding ),
// tested here because they are invisible in an implementation that gets them
// wrong: a builder that threw mid-chain, or returned a status nobody could
// store, would still pass every other test in this directory.
#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/validate.hpp"

namespace {

using namespace fuse;

std::vector<std::string> codes(const Diagnostics& d) {
  std::vector<std::string> out;
  for (const auto& item : d.items()) out.push_back(item.code);
  return out;
}

bool has(const std::vector<std::string>& v, std::string_view code) {
  return std::find(v.begin(), v.end(), code) != v.end();
}

// Authoring calls return the builder, never a status, and an unhonourable
// instruction is RECORDED -- so a chain containing a contradiction still builds,
// and validate() then reports that contradiction alongside every unrelated
// defect. First-failure reporting is the behaviour being excluded.
TEST(Builder, AContradictoryChainStillBuildsAndValidateReportsEveryDefect) {
  CouplingGraph g;
  const auto a = g.add_component({.name = "a", .ranks = {0, 1}, .interval = Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = {1, 1}, .interval = Rational{1}});

  Units watt_per_m2{.length = 0, .mass = 1, .time = -3};
  const auto q = g.add_field(a, {.name = "q", .units = watt_per_m2, .intent = Intent::Provides});
  const auto t = g.add_field(b, {.name = "T",
                                 .components = 3,
                                 .units = Units{.temperature = 1},
                                 .intent = Intent::Requires});

  // Three independent defects in one chain: incompatible units, mismatched
  // component counts, and a self-coupling.
  const auto bad = g.add_channel("bad", q, t);
  g.channel(bad).lag(-1);                       // a negative lag is not a lag
  const auto loop = g.add_channel("loop", q, q);  // source is its own target
  (void)loop;

  const auto d = validate(g);
  const auto found = codes(d);
  EXPECT_TRUE(has(found, "E-UNITS")) << d.to_string();
  EXPECT_TRUE(has(found, "E-SHAPE")) << d.to_string();
  EXPECT_TRUE(has(found, "E-SELF")) << d.to_string();
  EXPECT_GE(d.size(), 3u) << "validate reported fewer defects than the graph has";
}

// The same rules from the other side: the fluent and non-fluent spellings of the
// same graph are the same graph, which the canonical bytes are the definition of.
TEST(Builder, FluentAndNonFluentSpellingsProduceIdenticalBytes) {
  CouplingGraph fluent;
  {
    const auto a = fluent.add_component({.name = "a", .ranks = {0, 2}, .interval = Rational{1, 2}});
    const auto b = fluent.add_component({.name = "b", .ranks = {2, 2}, .interval = Rational{1, 2}});
    const auto s = fluent.add_field(a, {.name = "q", .intent = Intent::Provides});
    const auto t = fluent.add_field(b, {.name = "q", .intent = Intent::Requires});
    const auto ch = fluent.add_channel("q", s, t);
    fluent.channel(ch).accumulate(AccumOp::Mean).relax(0.5).lag(2);
  }

  CouplingGraph direct;
  {
    const auto a = direct.add_component({.name = "a", .ranks = {0, 2}, .interval = Rational{1, 2}});
    const auto b = direct.add_component({.name = "b", .ranks = {2, 2}, .interval = Rational{1, 2}});
    const auto s = direct.add_field(a, {.name = "q", .intent = Intent::Provides});
    const auto t = direct.add_field(b, {.name = "q", .intent = Intent::Requires});
    const auto ch = direct.add_channel("q", s, t);
    // The same three settings, one statement each rather than one chain.
    auto builder = direct.channel(ch);
    builder.accumulate(AccumOp::Mean);
    builder.relax(0.5);
    builder.lag(2);
  }

  EXPECT_EQ(serialize(fluent), serialize(direct));
}

// An authoring call returns the builder, so a chain is a chain; nothing in the
// authoring surface hands back a Result for the caller to drop on the floor.
// (Result lives at the execution boundary: validate, run, initialize.)
TEST(Builder, AuthoringCallsReturnTheBuilder) {
  CouplingGraph g;
  const auto a = g.add_component({.name = "a", .ranks = {0, 1}, .interval = Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = {1, 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, {.name = "q", .intent = Intent::Provides});
  const auto t = g.add_field(b, {.name = "q", .intent = Intent::Requires});
  auto builder = g.channel(g.add_channel("q", s, t));

  static_assert(std::is_same_v<decltype(builder.accumulate(AccumOp::Sum)), ChannelBuilder&>);
  static_assert(std::is_same_v<decltype(builder.relax(0.5)), ChannelBuilder&>);
  static_assert(std::is_same_v<decltype(builder.lag(1)), ChannelBuilder&>);
  SUCCEED();
}

}  // namespace
