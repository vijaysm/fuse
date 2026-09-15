#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/runtime/component.hpp"

/// Split out of the IR's checkpoint rule during the merge (see the note in
/// src/ir/checkpoint.cpp): deriving the checkpoint set is a property of the
/// graph and belongs to `ir`, while asking whether the components actually
/// present can carry it is a property of the run and belongs here.

namespace fuse {

Diagnostics verify_rollback_support(const CouplingGraph& g,
                                    const CheckpointSet& set,
                                    const ComponentSet& components) {
  Diagnostics diags;
  for (const ComponentId c : set.components) {
    if (!c.valid() || c.value() >= g.component_count()) {
      diags.add(Severity::Error, "E-ITER-NOROLLBACK",
                "the derived checkpoint set names component #" +
                    std::to_string(c.value()) +
                    ", which is not in the coupling graph, so its rollback "
                    "support cannot be verified");
      continue;
    }
    const std::string& name = g.component(c).name;
    const Component* impl = components.find(name);
    if (impl == nullptr) {
      // Not "assume it is fine": an unverified claim is refused. The whole
      // point of deriving the set is that the runtime checks it.
      diags.add(Severity::Error, "E-ITER-NOROLLBACK",
                "component '" + name +
                    "' must be checkpointed to iterate this region, but no "
                    "component with that name is registered, so it cannot be "
                    "asked whether it supports rollback");
      continue;
    }
    if (!impl->supports_rollback()) {
      diags.add(
          Severity::Error, "E-ITER-NOROLLBACK",
          "component '" + name +
              "' is re-advanced over the same interval on every sweep of this "
              "iteration region, so it must implement save_state() and "
              "restore_state(), but supports_rollback() returns false. "
              "Without a rollback each sweep starts from the state the "
              "previous sweep left behind, so the iteration converges to the "
              "fixed point of the wrong operator -- a wrong answer with no "
              "symptom, not a crash",
          "component '" + name + "'");
    }
  }
  return diags;
}

Diagnostics verify_iteration_region(const CouplingGraph& g,
                                    const OperatorDAG& dag,
                                    const OperatorDAG::IterationRegion& region,
                                    const ConvergenceCriterion& convergence,
                                    const ComponentSet& components) {
  Diagnostics diags = verify_iteration_declaration(
      region.name, region.max_iterations, convergence);

  diags.merge(
      verify_rollback_support(g, derive_checkpoint_set(g, dag, region),
                              components));
  return diags;
}

}  // namespace fuse
