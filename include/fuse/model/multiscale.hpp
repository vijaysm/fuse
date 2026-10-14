#pragma once

/// Multiscale and surrogate contracts — **declared, and rejecting.**
///
/// Neither parent implements these. They are declared now, at the stage that
/// freezes the descriptor, for one reason: the narrow waist is about to become a
/// binary dependency, and a type added to it later is an ABI break. Declaring the
/// shape now costs nothing; discovering it after the freeze costs a version.
///
/// **Every entry point here refuses.** Not "returns empty", not "works for the
/// simple case" — a refusal with a code, which a test asserts fires. That is the
/// distinction the testing page calls out: a deliberate refusal counts as verified
/// only when a test makes the refusal happen. The alternative — a stub that
/// silently succeeds — would let a graph declare a micro-macro coupling, validate
/// clean, and run as if the closure had been applied.
///
/// What each type is FOR, so the eventual implementation is not a guess:
///
///  * `InstanceSet` — N independent instances of a component (representative
///    volume elements, subgrid columns, particles) sharing one description. The
///    macro scale sees an aggregate; each instance runs its own time loop.
///  * `Closure` — the reduction from instance results to the macro quantity, and
///    the distribution of macro state back down. The direction matters: a closure
///    that is not the adjoint of its own distribution silently breaks
///    conservation across scales.
///  * `Overlap` — how much of the macro cell each instance accounts for.
///    Coverage that does not sum to one is the multiscale analogue of a
///    non-conservative remap, so it is accounted for, not assumed.
///  * `SamplePool` — the instance results available for reuse, which is what makes
///    an emulator possible: it is trained on, and validated against, this pool.
///  * `SurrogateContract` — what a learned model must state before it may stand in
///    for a component: where it is applicable, how out-of-distribution input is
///    detected, whether it carries hidden state (and therefore whether a rollback
///    is even meaningful), how deterministic it is bit-for-bit, and what kind of
///    derivative it can supply. A surrogate without these is not substitutable;
///    it is a guess with a confident interface.

#include <cstdint>
#include <string>
#include <vector>

#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/semantics.hpp"

namespace fuse {

// ---------------------------------------------------------------- multiscale

/// N instances of one component description.
struct InstanceSetSpec {
  std::string name;
  ComponentId component{};
  /// Instance count is STATIC. A dynamic count makes the compiled plan
  /// data-dependent, which the plan's whole point is to avoid.
  std::uint32_t count = 0;
};

enum class ClosureKind : std::uint8_t {
  /// Macro value is the coverage-weighted mean of the instances.
  WeightedMean = 0,
  /// Macro value is the sum (extensive quantities).
  Sum = 1,
  /// Caller-supplied reduction.
  Custom = 2,
};

struct ClosureSpec {
  std::string name;
  ClosureKind kind = ClosureKind::WeightedMean;
  FieldId instance_field{};
  FieldId macro_field{};
};

/// Per-instance share of the macro cell. Exact rationals, so a coverage sum can
/// be CHECKED rather than compared against a tolerance.
struct OverlapSpec {
  std::vector<Rational> coverage;
  /// BM-19 (review 2026-10-11): the accumulation used the throwing `+`
  /// operator, which can `FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW", ...)` on
  /// an unrepresentable running sum -- the same refusal happens now, via
  /// `try_add`, but explicitly and by name at this call site instead of
  /// three stack frames down inside `Rational::operator+=`, so a reader
  /// (or a debugger backtrace) sees which accumulation overflowed rather
  /// than just that an addition somewhere did. `check_overlap()`
  /// (src/model/multiscale.cpp), today's only real consumer, refuses
  /// unconditionally regardless of the value (multiscale is declared and
  /// rejecting, not yet implemented), so this only ever affects the text
  /// of that refusal's message.
  [[nodiscard]] Rational total() const {
    Rational sum{0};
    for (const auto& c : coverage) {
      const auto next = Rational::try_add(sum, c);
      if (!next) {
        FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                   "OverlapSpec::total: coverage sum overflow");
      }
      sum = *next;
    }
    return sum;
  }
};

struct SamplePoolSpec {
  std::string name;
  /// Zero means "keep everything", which is a memory decision, not a default.
  std::uint64_t capacity = 0;
};

// ---------------------------------------------------------------- surrogate

enum class DeterminismLevel : std::uint8_t {
  /// Bit-identical for identical input, on identical hardware AND across the
  /// rank counts the coupler may use. Anything less cannot be used inside a
  /// fixed-point iteration whose convergence test compares iterates.
  Bitwise = 0,
  /// Reproducible run to run on one configuration, not across configurations.
  RunToRun = 1,
  /// Stochastic by construction.
  Nondeterministic = 2,
};

enum class DerivativeKind : std::uint8_t {
  None = 0,
  /// Directional derivative (JVP) available.
  Forward = 1,
  /// Adjoint (VJP) available.
  Reverse = 2,
  Both = 3,
};

struct SurrogateContract {
  std::string name;
  /// The region of input space where the model claims validity. Empty means the
  /// model makes no claim, which is a reason to refuse it, not to trust it.
  std::vector<FieldSemantics> applicability;
  /// How out-of-distribution input is detected at run time. Empty means it is
  /// not detected.
  std::string ood_detector;
  /// A model with hidden state cannot be rolled back by restoring its inputs,
  /// so an iteration region containing it needs its own checkpoint hook.
  bool carries_hidden_state = false;
  DeterminismLevel determinism = DeterminismLevel::Nondeterministic;
  DerivativeKind derivative = DerivativeKind::None;
};

// ------------------------------------------------------- the rejecting surface

/// Registering an instance set. Refuses: `E-MULTISCALE-NOT-IMPLEMENTED`.
[[nodiscard]] Result<void> declare_instance_set(const InstanceSetSpec& spec);

/// Registering a closure. Refuses: `E-MULTISCALE-NOT-IMPLEMENTED`.
[[nodiscard]] Result<void> declare_closure(const ClosureSpec& spec);

/// Checking that coverage sums to one. Refuses: `E-MULTISCALE-NOT-IMPLEMENTED`
/// — and note that it refuses even for coverage that WOULD pass, because a
/// check that only sometimes runs is worse than one that never does.
[[nodiscard]] Result<void> check_overlap(const OverlapSpec& spec);

/// Declaring a sample pool. Refuses: `E-MULTISCALE-NOT-IMPLEMENTED`.
[[nodiscard]] Result<void> declare_sample_pool(const SamplePoolSpec& spec);

/// Substituting a surrogate for a component. Refuses:
/// `E-SURROGATE-NOT-IMPLEMENTED`.
[[nodiscard]] Result<void> substitute_surrogate(ComponentId component,
                                                const SurrogateContract& contract);

/// Whether a contract states enough to be substitutable at all. This one does
/// NOT refuse: it is pure declaration checking, it is useful today for a user
/// writing a contract, and the eventual implementation will call it first.
[[nodiscard]] Diagnostics review_surrogate_contract(const SurrogateContract& contract);

}  // namespace fuse
