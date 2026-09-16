// A randomized sweep over generated graphs, checked against an independently
// computed answer.
//
// WHY A SECOND ANSWER. The loop detector works on the expanded instance graph:
// every (component, step) pair, every dataflow arc, the black-box rule that a
// component's outputs depend on all its inputs, and the rules for which edges
// an iteration region may cut. That is a lot of machinery, and a bug in it is
// invisible in the direction that matters -- reporting "no loop" for a graph
// that has one lets an unsolvable setup run and produce plausible numbers.
//
// So this suite restricts itself to graphs where the answer can be obtained a
// completely different way. Every component runs at the same interval, so a
// dependence within one time level exists exactly when a channel with no lag
// connects two components; a lagged channel depends on an earlier step and can
// never close a loop at one level. The answer is then whether the directed
// graph over COMPONENTS induced by the unlagged channels has a cycle -- a
// dozen lines of depth-first search over at most eight nodes, sharing no code
// with the detector.
//
// The restriction is the price of an independent answer, and it is stated
// plainly rather than hidden: multirate graphs, iteration regions and
// space-time windows are not generated here. They are covered by the named
// cases in acyclicity_test.cpp, where the expected answer is worked out by
// hand.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/schedule/acyclicity.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/expand.hpp>

namespace {

using namespace fuse;

struct Generated {
  CouplingGraph graph;
  /// The independently computed answer: true when the unlagged channels close
  /// a loop over the components.
  bool has_same_level_loop = false;
  std::string description;
};

/// Depth-first cycle search over an explicit adjacency list. Deliberately the
/// most ordinary algorithm available, on the smallest possible graph.
bool has_cycle(const std::vector<std::vector<int>>& adj) {
  const int n = static_cast<int>(adj.size());
  std::vector<int> state(n, 0);  // 0 unvisited, 1 on stack, 2 done
  std::vector<int> stack;
  for (int start = 0; start < n; ++start) {
    if (state[start] != 0) continue;
    stack.push_back(start);
    std::vector<std::size_t> next(n, 0);
    while (!stack.empty()) {
      const int v = stack.back();
      if (state[v] == 0) state[v] = 1;
      if (next[v] < adj[v].size()) {
        const int w = adj[v][next[v]++];
        if (state[w] == 1) return true;
        if (state[w] == 0) stack.push_back(w);
      } else {
        state[v] = 2;
        stack.pop_back();
      }
    }
  }
  return false;
}

Generated generate(std::mt19937& rng) {
  Generated out;
  CouplingGraph& g = out.graph;

  const int n_comp = 2 + static_cast<int>(rng() % 4);          // 2..5
  const int n_chan = 1 + static_cast<int>(rng() % 6);          // 1..6

  const auto mesh = g.add_mesh({.name = "m"});
  const auto region = g.add_region({.name = "r", .mesh = mesh});

  std::vector<ComponentId> comps;
  for (int i = 0; i < n_comp; ++i) {
    comps.push_back(g.add_component({.name = "c" + std::to_string(i),
                                     .ranks = {i, 1},
                                     .interval = Rational{1}}));
  }

  std::vector<std::vector<int>> adj(n_comp);
  int field_seq = 0;
  // Targets already written, per component, so a later channel can be aimed
  // at one of them. That is the case this sweep most needs to cover: several
  // channels writing one field used to have all but the last silently
  // dropped, which deleted a dependence and could remove a loop.
  std::vector<std::vector<FieldId>> written(n_comp);

  for (int k = 0; k < n_chan; ++k) {
    const int a = static_cast<int>(rng() % static_cast<unsigned>(n_comp));
    int b = static_cast<int>(rng() % static_cast<unsigned>(n_comp));
    if (b == a) b = (a + 1) % n_comp;  // a self-channel is a separate check
    const int lag = (rng() % 3 == 0) ? 1 : 0;

    const auto src = g.add_field(comps[a], {.name = "f" + std::to_string(field_seq++),
                                            .region = region,
                                            .intent = Intent::Provides});
    FieldId dst;
    if (!written[b].empty() && rng() % 3 == 0) {
      dst = written[b][rng() % written[b].size()];
      g.merge(dst, static_cast<MergeOp>(rng() % 3));
    } else {
      dst = g.add_field(comps[b], {.name = "f" + std::to_string(field_seq++),
                                   .region = region,
                                   .intent = Intent::Requires});
      written[b].push_back(dst);
    }
    auto ch = g.channel(g.add_channel("ch" + std::to_string(k), src, dst));
    if (lag > 0) {
      // A lagged channel delivers the previous step's value, so the target's
      // time level genuinely differs and the compiler requires the user to say
      // that is intended. Saying it here is part of generating a well-formed
      // graph, not a way around the check.
      ch.lag(lag).allow_time_level_mismatch("generated lagged channel");
    }

    out.description += (lag > 0 ? "c" : "C") + std::to_string(a) + "->" +
                       std::to_string(b) + "." + std::to_string(dst.value()) + " ";
    if (lag == 0) adj[a].push_back(b);
  }

  out.has_same_level_loop = has_cycle(adj);
  return out;
}

CycleReport run_detector(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  EXPECT_TRUE(td.ok()) << td.error().message;
  return prove_acyclic(g, lowered.dag, td.value());
}

/// 2000 by default so the suite stays a few seconds. FUSE_SWEEP_GRAPHS raises
/// it for a deliberate long run; the seed is fixed either way, so a failure is
/// reproducible from the reported index alone.
std::size_t sweep_size() {
  if (const char* env = std::getenv("FUSE_SWEEP_GRAPHS")) {
    const long n = std::strtol(env, nullptr, 10);
    if (n > 0) return static_cast<std::size_t>(n);
  }
  return 2000;
}

TEST(AcyclicitySweep, VerdictAgreesWithAnIndependentlyComputedAnswer) {
  std::mt19937 rng(20260916u);
  std::size_t with_loop = 0, without = 0, false_negatives = 0, false_positives = 0;

  const std::size_t n = sweep_size();
  for (std::size_t i = 0; i < n; ++i) {
    const Generated gen = generate(rng);

    // Skip graphs the authoring gate refuses: their meaning is undefined, so
    // there is nothing for the two answers to agree about. This is also where
    // undeclared multiple writers go.
    if (validate(gen.graph).has_errors()) continue;

    const CycleReport report = run_detector(gen.graph);
    if (gen.has_same_level_loop) {
      ++with_loop;
      if (report.acyclic) {
        ++false_negatives;
        ADD_FAILURE() << "graph " << i << " (" << gen.description
                      << ") has a same-level loop and was certified acyclic; "
                         "this is the direction that lets an unsolvable setup run";
      }
    } else {
      ++without;
      if (!report.acyclic) {
        ++false_positives;
        ADD_FAILURE() << "graph " << i << " (" << gen.description
                      << ") has no same-level loop and was rejected:\n"
                      << report.explanation;
      }
    }
  }

  // The sweep is only evidence if it generated both answers.
  EXPECT_GT(with_loop, n / 50) << "the generator produced almost no looping graphs";
  EXPECT_GT(without, n / 50) << "the generator produced almost no acyclic graphs";
  EXPECT_EQ(false_negatives, 0u);
  EXPECT_EQ(false_positives, 0u);
  std::printf("[ sweep    ] %zu graphs: %zu with a loop, %zu without\n",
              with_loop + without, with_loop, without);
}

}  // namespace
