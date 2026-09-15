#include "fuse/model/multiscale.hpp"

#include <fmt/format.h>

namespace fuse {

namespace {

/// One message shape for every refusal, so a user who hits one knows what the
/// state of the feature is and what to do instead, rather than reading "not
/// implemented" and guessing whether it is coming.
Failure not_implemented(std::string_view code, std::string_view what, std::string_view name) {
  return Failure{std::string(code),
                 fmt::format("{} is declared but not implemented ('{}'). The contract is fixed so "
                             "the descriptor does not change when it lands; until then a graph "
                             "using it is refused rather than run as if the feature were inert.",
                             what, name)};
}

}  // namespace

Result<void> declare_instance_set(const InstanceSetSpec& spec) {
  return not_implemented("E-MULTISCALE-NOT-IMPLEMENTED", "instance sets", spec.name);
}

Result<void> declare_closure(const ClosureSpec& spec) {
  return not_implemented("E-MULTISCALE-NOT-IMPLEMENTED", "closures", spec.name);
}

Result<void> check_overlap(const OverlapSpec& spec) {
  // Refuses even when the coverage would pass. Accounting for coverage on the
  // paths that happen to be correct, and not on the rest, is how a conservation
  // error becomes invisible.
  return not_implemented("E-MULTISCALE-NOT-IMPLEMENTED", "overlap accounting",
                         fmt::format("{} instances summing to {}", spec.coverage.size(),
                                     spec.total().to_string()));
}

Result<void> declare_sample_pool(const SamplePoolSpec& spec) {
  return not_implemented("E-MULTISCALE-NOT-IMPLEMENTED", "sample pools", spec.name);
}

Result<void> substitute_surrogate(ComponentId component, const SurrogateContract& contract) {
  // The contract is reviewed first, so a user gets BOTH answers at once: what is
  // wrong with the contract, and that substitution is not available yet. Giving
  // only the second would send them away to fix nothing.
  const Diagnostics review = review_surrogate_contract(contract);
  if (review.has_errors()) {
    return Failure{"E-SURROGATE-CONTRACT",
                   fmt::format("the contract for component {} is not substitutable: {}",
                               component.value(), review.to_string())};
  }
  return not_implemented("E-SURROGATE-NOT-IMPLEMENTED", "surrogate substitution", contract.name);
}

Diagnostics review_surrogate_contract(const SurrogateContract& contract) {
  Diagnostics d;
  const std::string ctx = fmt::format("surrogate '{}'", contract.name);

  if (contract.name.empty()) {
    d.add(Severity::Error, "E-SURROGATE-UNNAMED", "a surrogate contract must be named", ctx);
  }
  if (contract.applicability.empty()) {
    d.add(Severity::Error, "E-SURROGATE-NO-APPLICABILITY",
          "the contract states no applicability region, so there is nothing against which an "
          "input can be judged in or out of distribution",
          ctx);
  }
  if (contract.ood_detector.empty()) {
    d.add(Severity::Error, "E-SURROGATE-NO-OOD",
          "no out-of-distribution detector is named; a surrogate that cannot report when it is "
          "extrapolating is not substitutable for a component that would have converged or failed",
          ctx);
  }
  if (contract.determinism == DeterminismLevel::Nondeterministic) {
    // Not an error by itself: an explicit-coupling graph can use one. It is an
    // error to use it inside an iteration region, which the compiler checks when
    // it knows the graph -- so this is the warning, and that will be the refusal.
    d.add(Severity::Warning, "W-SURROGATE-NONDETERMINISTIC",
          "a nondeterministic surrogate cannot be used inside an iteration region: the "
          "convergence test compares successive iterates, and noise in the model is "
          "indistinguishable from a residual",
          ctx);
  }
  if (contract.carries_hidden_state) {
    d.add(Severity::Warning, "W-SURROGATE-HIDDEN-STATE",
          "the surrogate carries hidden state, so restoring its inputs does not restore it; an "
          "iteration region or a rollback needs an explicit checkpoint hook from the model",
          ctx);
  }
  return d;
}

}  // namespace fuse
