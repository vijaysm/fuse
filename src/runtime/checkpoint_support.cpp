#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/multiscale.hpp"
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

    // KNOWN_ISSUES.md B3. `capabilities()` can now say more than whether a
    // component rolls back; the determinism bit is the one this region's own
    // mechanism can act on directly, so it is checked here rather than left
    // decorative. This is exactly the refusal
    // src/model/multiscale.cpp::review_surrogate_contract() predicts for a
    // `SurrogateContract` ("It is an error to use [a nondeterministic
    // surrogate] inside an iteration region, which the compiler checks when
    // it knows the graph") -- multiscale.cpp cannot check it (it has no
    // graph and no ComponentSet; see that file's header), so it is checked
    // here, the one place that has both. A component that says nothing about
    // determinism has declared NOTHING (capabilities.hpp: raw 0 is undeclared,
    // not `Bitwise`); that is a warning, W-ITER-DETERMINISM-UNDECLARED, so the
    // gap is visible in the diagnostics without refusing every component
    // written before the bitmask existed. An explicit `Nondeterministic` is
    // refused.
    const std::optional<DeterminismLevel> level = impl->determinism_level();
    if (!level.has_value()) {
      diags.add(
          Severity::Warning, "W-ITER-DETERMINISM-UNDECLARED",
          "component '" + name +
              "' is re-advanced inside an iteration region but its "
              "capabilities() declares no determinism level; the region's "
              "convergence test assumes the component is at least "
              "run-to-run reproducible. Declare one with "
              "fuse::make_capabilities()",
          "component '" + name + "'");
    }
    if (level == DeterminismLevel::Nondeterministic) {
      diags.add(
          Severity::Error, "E-ITER-NONDETERMINISTIC",
          "component '" + name +
              "' declares capabilities() with DeterminismLevel::Nondeterministic "
              "and is re-advanced inside iteration region's sweep loop, whose "
              "convergence test compares successive iterates. Noise in a "
              "component's own output is indistinguishable, in that "
              "comparison, from a residual that has not yet converged: the "
              "region may report convergence at an iterate that was never "
              "actually a fixed point, or may fail to converge at all for a "
              "reason that has nothing to do with the coupling. A "
              "nondeterministic component may be used in an explicit "
              "(non-iterated) coupling, just not inside this region",
          "component '" + name + "'");
    }
  }
  return diags;
}

// S7.1's Newton-Krylov seam. Mirrors verify_rollback_support() immediately
// above: same checkpoint set, same "an unverifiable claim is refused, not
// assumed" rule for a component the set names that is not registered at
// all (already reported by verify_rollback_support() over the same set;
// this function adds nothing further for that case so the two diagnostics
// are not duplicated).
Diagnostics verify_newton_capability_support(const CouplingGraph& g,
                                             const CheckpointSet& set,
                                             const ComponentSet& components,
                                             bool requires_residual,
                                             bool requires_linearization) {
  Diagnostics diags;
  if (!requires_residual && !requires_linearization) {
    // No Newton solver selected for this region (IterationRegionDecl's own
    // default): nothing to check, exactly like a region that declares no
    // accelerator incurring no extra rollback check beyond the existing one.
    return diags;
  }
  for (const ComponentId c : set.components) {
    if (!c.valid() || c.value() >= g.component_count()) continue;  // reported above
    const std::string& name = g.component(c).name;
    const Component* impl = components.find(name);
    if (impl == nullptr) continue;  // reported above

    const std::uint64_t caps = impl->capabilities();
    if (requires_residual && (caps & kCapabilityResidual) == 0) {
      diags.add(
          Severity::Error, "E-NEWTON-CAPABILITY",
          "component '" + name +
              "' is a member of a Newton-solved iteration region, so the "
              "coupler must be able to evaluate its own contribution to the "
              "coupled residual at a frozen time level without advancing "
              "its state, but capabilities() does not set "
              "kCapabilityResidual. Implement fuse::ImplicitComponent:: "
              "evaluate_residual(), override Component::implicit() to "
              "return it, and set kCapabilityResidual in capabilities()",
          "component '" + name + "'");
      continue;  // no residual capability: asking about linearization too
                 // would just repeat the same missing-interface defect.
    }
    if (requires_linearization && (caps & kCapabilityLinearization) == 0) {
      diags.add(
          Severity::Error, "E-NEWTON-CAPABILITY",
          "component '" + name +
              "' is a member of a Newton-solved iteration region whose "
              "preconditioner needs this component's own linearisation, but "
              "capabilities() does not set kCapabilityLinearization. "
              "Return a usable fuse::LinearizationToken from "
              "ImplicitComponent::linearization() once evaluate_residual() "
              "has been called, and set kCapabilityLinearization in "
              "capabilities()",
          "component '" + name + "'");
    }
  }
  return diags;
}

Diagnostics verify_iteration_region(const CouplingGraph& g,
                                    const OperatorDAG& dag,
                                    const OperatorDAG::IterationRegion& region,
                                    const ConvergenceCriterion& convergence,
                                    const ComponentSet& components,
                                    bool requires_residual_capability,
                                    bool requires_linearization_capability) {
  Diagnostics diags = verify_iteration_declaration(
      region.name, region.max_iterations, convergence);

  const CheckpointSet set = derive_checkpoint_set(g, dag, region);
  diags.merge(verify_rollback_support(g, set, components));
  diags.merge(verify_newton_capability_support(
      g, set, components, requires_residual_capability,
      requires_linearization_capability));
  return diags;
}

}  // namespace fuse
