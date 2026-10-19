#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"
#include "fuse/model/iteration.hpp"

// Spec Section 5.3's derived checkpoint set (Task 7.2).
//
// THE RULE, stated so it can be checked. Re-executing an iteration region's
// body must produce the same answer it would have produced had the previous
// sweep never run. A value therefore needs saving exactly when the body both
// reads and writes it across the sweep boundary:
//
//   1. LOOP-CARRIED. Produced inside the region and read inside the region at
//      a previous iterate. `Relax(x@[k-1], v, omega)` is the canonical case:
//      its output buffer at sweep k is a function of its output buffer at
//      sweep k-1, so that buffer is state, not a transient. So is an
//      `Accelerate` node, whose next iterate is a function of the residual and
//      iterate history it accumulated over previous sweeps.
//   2. LIVE-IN (produced outside, read inside). No checkpoint: re-executing
//      the body recomputes from it, and the body never writes it.
//   3. LIVE-OUT (produced inside, read outside). No checkpoint: a rejected
//      iterate never escapes the region, so nothing outside ever observes the
//      intermediate value.
//   4. COMPONENT STATE. A component whose Source output feeds the region, or
//      which the region delivers into, is re-advanced over the *same* time
//      interval on every sweep. Its internal state moves and must be
//      restorable; FUSE cannot see that state, so it can only name the
//      component and verify (verify_rollback_support) that it says it can roll
//      back.
//   5. PURE. Everything else in the body -- Sample, Accumulate, Transport,
//      Remap, Constrain, Diagnose -- is a function of live-in values and this
//      sweep's own upstream results. Recomputed, never saved. This is the
//      whole point of deriving the set instead of asking for it.
//
// WHY IT CANNOT UNDER-APPROXIMATE, and where it deliberately over-approximates.
// Checkpointing too much costs a copy. Checkpointing too little is silent
// wrong physics: sweep k+1 starts from state sweep k mutated, so the iteration
// converges to the fixed point of a different operator and the residual
// history looks perfectly healthy while it does. Every judgement call below
// therefore errs towards saving more, and each one says so:
//
//   * (rule 1) An input arc *typed* at `iterate < 0` is taken as carried
//     whether or not its producer is inside the body. No lowering rule sets
//     `iterate` yet, so today this fires never; when one does, a value from a
//     previous iterate produced outside the region would be saved needlessly
//     rather than missed.
//   * (rule 1) `Relax`'s previous-iterate operand is implicit -- lowering
//     emits a single-input node (rules_channel.cpp step 6) and there is no arc
//     to find. The node kind is therefore taken as sufficient evidence, with
//     no test on omega: omega == 1 is plain Picard, which still reads the
//     previous iterate to form the update.
//   * (rule 4) A component is included when it merely *receives* from the
//     region, not only when it feeds it. A pure downstream consumer arguably
//     need only be advanced once, after convergence; but a runner that
//     re-advances it every sweep and never restores it is wrong, and the cost
//     of being wrong here is not a slow run.
//   * The whole analysis is per-region and takes no account of nesting. A
//     value carried by an inner region is carried by the outer one too, and
//     since both sets are derived independently and unioned by the runner,
//     that direction is safe.

namespace fuse {
namespace {

/// Sorted, deduplicated membership test over `region.body`. A linear scan per
/// query would be quadratic in the body size, and a body is as large as the
/// lowered channels it covers.
[[nodiscard]] std::vector<std::uint32_t> body_index(
    const OperatorDAG& dag, const OperatorDAG::IterationRegion& region) {
  std::vector<std::uint32_t> body;
  body.reserve(region.body.size());
  for (const NodeId n : region.body) {
    // A body naming a node outside the DAG is malformed input, not something
    // to index with: skip it rather than reach past the end. It cannot make
    // the set too small in any case a caller can construct, because
    // build_iteration_regions() only ever pushes ids it just enumerated.
    if (n.valid() && n.value() < dag.node_count()) body.push_back(n.value());
  }
  std::sort(body.begin(), body.end());
  body.erase(std::unique(body.begin(), body.end()), body.end());
  return body;
}

void sort_unique(std::vector<std::uint32_t>& v) {
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
}

}  // namespace

CheckpointSet derive_checkpoint_set(const CouplingGraph& g,
                                    const OperatorDAG& dag,
                                    const OperatorDAG::IterationRegion& region) {
  CheckpointSet out;
  const std::vector<std::uint32_t> body = body_index(dag, region);

  std::vector<std::uint32_t> arcs;   // coupler_arcs, as raw indices
  std::vector<std::uint32_t> comps;  // components, as raw indices

  std::size_t relax_carried = 0;
  std::size_t accel_carried = 0;
  std::size_t typed_carried = 0;

  for (const std::uint32_t ni : body) {
    const Node& n = dag.node(NodeId{ni});

    for (const ArcId in : n.inputs) {
      if (!in.valid() || in.value() >= dag.arc_count()) continue;
      const Arc& a = dag.arc(in);

      // Rule 1, explicit form: the value is typed as belonging to a previous
      // iterate. Errs towards saving more -- no test that the producer is
      // inside the body (see the file header).
      if (a.type.time.iterate < 0) {
        arcs.push_back(in.value());
        ++typed_carried;
      }

      // Rule 4, producer side: an arc the region reads that a component's
      // Source emitted. `ValueType::rank_set` is the owning component index
      // (op.hpp), and lowering types a Source arc with source_type(), so it is
      // the owner of the provided field.
      const NodeId prod = a.producer;
      if (prod.valid() && prod.value() < dag.node_count() &&
          dag.node(prod).kind == OpKind::Source && a.type.rank_set >= 0 &&
          static_cast<std::size_t>(a.type.rank_set) < g.component_count()) {
        comps.push_back(static_cast<std::uint32_t>(a.type.rank_set));
      }
    }

    // Rule 1, structural form -- the one that actually fires today. Lowering
    // represents Relax's previous-iterate operand implicitly, so there is no
    // arc typed at iterate = -1 to match on; the node kind is the evidence.
    if ((n.kind == OpKind::Relax || n.kind == OpKind::Accelerate) &&
        n.output.valid() && n.output.value() < dag.arc_count()) {
      arcs.push_back(n.output.value());
      if (n.kind == OpKind::Relax) {
        ++relax_carried;
      } else {
        ++accel_carried;
      }
    }

    // Rule 4, both sides, via the channel this node was lowered from. The
    // producer-side scan above misses the component a channel *delivers* to,
    // because build_iteration_regions() runs before lower() creates the Sink
    // nodes (lowering.cpp steps 2 and 3), so the body contains no Sink and the
    // delivery is invisible in the DAG at this point.
    //
    // D73 ROUND 2: excludes `OpKind::BuildRemapHalo` explicitly, the same way
    // `build_iteration_regions()` (src/ir/regions.cpp) excludes it from a
    // region's own `body` collection. `BuildRemapHalo` carries the
    // qualifying channel's own `ChannelId` as its `origin` (so `execute_
    // remap_halo_instructions()` and `Runner::Impl::resolve_remap_halo()`
    // can find the channel it is for), but it is a ONE-TIME setup
    // instruction with no field value at all (`OpKind::BuildRemapHalo`'s own
    // doc comment, op.hpp) -- not a per-iteration body member whose
    // component this checkpoint set should name. Without this exclusion, a
    // caller that ever handed this function a `region.body` containing a
    // `BuildRemapHalo` node (this file's own `derive_checkpoint_set` does
    // not itself enforce round 1's exclusion -- it trusts the caller's own
    // `body`) would silently attribute that channel's components to the
    // checkpoint set regardless of whether the real body actually reads or
    // writes anything belonging to them.
    if (n.kind != OpKind::BuildRemapHalo &&
        n.origin.valid() && n.origin.value() < g.channel_count()) {
      const Channel& ch = g.channel_at(n.origin);
      if (ch.source.valid() && ch.source.value() < g.field_count()) {
        comps.push_back(g.owner_of(ch.source).value());
      }
      if (ch.target.valid() && ch.target.value() < g.field_count()) {
        comps.push_back(g.owner_of(ch.target).value());
      }
    }
  }

  sort_unique(arcs);
  sort_unique(comps);

  out.coupler_arcs.reserve(arcs.size());
  for (const std::uint32_t a : arcs) out.coupler_arcs.push_back(ArcId{a});
  out.components.reserve(comps.size());
  for (const std::uint32_t c : comps) out.components.push_back(ComponentId{c});

  // The explanation is not decoration: the user never wrote this set down, so
  // the only way they can disagree with it is if it is printed.
  std::string why = "iteration region '" + region.name + "': " +
                    std::to_string(out.coupler_arcs.size()) +
                    " coupler buffer(s) and " +
                    std::to_string(out.components.size()) +
                    " component(s) must survive a rejected iterate.";
  if (out.coupler_arcs.empty()) {
    why +=
        "\n  No coupler buffer is loop-carried: every value in the body is a "
        "pure function of values live-in to the region, so a re-swept iterate "
        "recomputes them. Nothing coupler-side is saved.";
  } else {
    why += "\n  Loop-carried coupler buffers (their value at sweep k reads "
           "sweep k-1, so they are state and not transients):";
    for (const ArcId a : out.coupler_arcs) {
      why += "\n    %" + std::to_string(a.value());
      if (a.value() < dag.arc_count()) {
        const Arc& arc = dag.arc(a);
        why += " \"" + arc.name + "\" : " + arc.type.to_string();
        if (arc.producer.valid() && arc.producer.value() < dag.node_count()) {
          why += " from ";
          why += to_cstring(dag.node(arc.producer).kind);
        }
      }
    }
    if (relax_carried != 0) {
      why += "\n    (" + std::to_string(relax_carried) +
             " from Relax: the relaxed value at iterate k is formed from the "
             "value at k-1, whatever omega is.)";
    }
    if (accel_carried != 0) {
      why += "\n    (" + std::to_string(accel_carried) +
             " from Accelerate: the next iterate is a function of the "
             "residual history accumulated over previous sweeps.)";
    }
    if (typed_carried != 0) {
      why += "\n    (" + std::to_string(typed_carried) +
             " read at a previous iterate, from the arc's own TimeLevel.)";
    }
  }
  if (out.components.empty()) {
    why += "\n  No component is re-advanced inside the region.";
  } else {
    why +=
        "\n  Components re-advanced over the same interval on every sweep, so "
        "their internal state must be restorable "
        "(save_state/restore_state):";
    for (const ComponentId c : out.components) {
      why += "\n    ";
      why += (c.value() < g.component_count()) ? g.component(c).name
                                               : ("component#" +
                                                  std::to_string(c.value()));
    }
  }
  why +=
      "\n  Everything else in the body is recomputed, not saved. Where this "
      "analysis had a choice it saved more: too much costs a copy, too little "
      "is a fixed point of the wrong operator.";
  out.explanation = std::move(why);

  return out;
}

// `verify_rollback_support(graph, ComponentSet)` lived here upstream. It asks the
// RUNTIME what each component can do (`Component::supports_rollback`), so in this
// library it is a layering inversion: `ir` may not depend on `runtime`. It moved
// to src/runtime/checkpoint_support.cpp, which S4 builds; the derivation above
// stays here because it is a pure function of the graph.
//
// `verify_iteration_region(..., ComponentSet)` moved with it, because it calls
// that function and likewise takes the registry.
//
// The split is not cosmetic. Deriving WHICH values must be carried is a property
// of the coupling graph and is provable at compile time; asking whether the
// components present can actually carry them is a property of the run. Conflating
// them is why the upstream rule could not be checked without a registry, and it
// is what made an `ir` source include a `runtime` header.

Diagnostics verify_iteration_declaration(
    const std::string& region_name, int max_iterations,
    const ConvergenceCriterion& convergence) {
  Diagnostics diags;
  // Task 7.1's hazard. all_satisfied() on an empty criterion returns true --
  // correctly, since true is the identity of AND and there is nothing to
  // refuse. For a region that means the first sweep is declared converged: an
  // explicit scheme wearing an implicit scheme's report. convergence.hpp
  // exposes empty() precisely so this can be refused here.
  if (convergence.empty()) {
    diags.add(Severity::Error, "E-ITER-NOCONVERGE",
              "iteration region '" + region_name +
                  "' has no convergence measure. An empty ConvergenceCriterion "
                  "is satisfied by definition (true is the identity of AND), "
                  "so the region would stop after one sweep and report that it "
                  "converged. Give it at least one measure, e.g. "
                  ".until(Converged::Relative(1e-6, \"field\")); if a single "
                  "sweep is what you want, do not declare an iteration region",
              "iteration region '" + region_name + "'");
  }

  if (max_iterations < 1) {
    diags.add(Severity::Error, "E-ITER-MAXITER",
              "iteration region '" + region_name + "' has max_iterations = " +
                  std::to_string(max_iterations) +
                  ", so its body would never run",
              "iteration region '" + region_name + "'");
  }
  return diags;
}

Result<std::vector<std::size_t>> match_convergence_measures(
    const std::string& region_name, const ConvergenceCriterion& convergence,
    const std::vector<MeasureTarget>& targets) {
  const std::vector<ConvergenceMeasure>& measures = convergence.measures();

  // --- rule 1: each measure names exactly one of those targets --------------
  std::vector<std::size_t> measure_target(measures.size(), 0);
  std::vector<bool>        covered(targets.size(), false);
  for (std::size_t mi = 0; mi < measures.size(); ++mi) {
    const std::string& want = measures[mi].field;
    std::size_t        hits = 0;
    for (std::size_t bi = 0; bi < targets.size(); ++bi) {
      const MeasureTarget& t = targets[bi];
      if (want != t.channel && want != t.source_field &&
          want != t.target_field) {
        continue;
      }
      measure_target[mi] = bi;
      covered[bi] = true;
      ++hits;
    }
    if (hits != 1) {
      std::string known;
      for (const MeasureTarget& t : targets) {
        if (!known.empty()) known += ", ";
        known += "channel '" + t.channel + "' (" + t.source_field + " -> " +
                 t.target_field + ")";
      }
      return Failure{"E-ITER-MEASURE",
                   "the convergence measure on '" + want +
                       "' matches " + std::to_string(hits) +
                       " of the residuals of iteration region '" + region_name +
                       "'; exactly one is required. A measure that matches "
                       "nothing would never be tested against anything, and "
                       "the region would run to max_iterations or, worse, "
                       "stop on the measures that do match. Name a channel "
                       "or one of its fields: " + known};
    }
  }

  // --- rule 2: EVERY residual is measured, not just one of them -------------
  //
  // MEASURED, not argued: on a two-channel pair, with the single measure the
  // obvious reading of spec Section 5.3's example suggests, IQN-ILS drove the measured
  // channel's residual to EXACTLY zero on sweep 3 while the unmeasured one
  // still stood at 0.8, and the region reported convergence at a_out = 2.25
  // against a true fixed point of 2.5 -- a 10% error, converged, with a
  // monotone residual history. A quasi-Newton step is a least-squares fit over
  // the whole concatenated vector and is perfectly capable of zeroing one block
  // at the expense of another; only Picard happens to bring them down together.
  // A partially measured region is therefore an iteration whose stopping rule
  // is not the coupling condition.
  for (std::size_t bi = 0; bi < covered.size(); ++bi) {
    if (covered[bi]) continue;
    return Failure{
        "E-ITER-MEASURE",
        "iteration region '" + region_name + "' exchanges channel '" +
            targets[bi].channel + "' (" + targets[bi].source_field + " -> " +
            targets[bi].target_field +
            "), but no convergence measure names it, so the region can stop "
            "while that channel is still far from its fixed point. An "
            "accelerator will do exactly that: a least-squares step over the "
            "concatenated residual can zero the measured block and leave the "
            "unmeasured one alone. Add a measure naming '" +
            targets[bi].source_field + "', '" + targets[bi].target_field +
            "' or '" + targets[bi].channel +
            "', or take the channel out of the region"};
  }

  return measure_target;
}

}  // namespace fuse
