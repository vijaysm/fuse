#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <type_traits>

#include "fuse/base/error.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/num/butcher_tableau.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 1
/// (docs/dev/integrators-design.md Section 4.1/4.3/4.4).
///
/// Three things live here: the IntegratorValue concept and TemporalProblem<Value>
/// adapter (a type-erased, std::function-based "Problem" in the sense of the owner's
/// decision (6), 2026-10-27: "type-erased Problem"), the Integrator<Value> interface
/// every family (explicit now, DIRK/Rosenbrock in round 2) implements, and
/// SubStepClock, the Rational time accumulator the owner's decision (3) asked for.
///
/// numtypes (D-E, the parallel "higgs" branch track) has not landed in this checkout:
/// grepped, include/fuse/model/value_traits.hpp and include/fuse/num/dual.hpp do not
/// exist. IntegratorValue below is therefore a LOCAL concept, written from the task's
/// own quoted handoff notes (value_traits<T>::scalar_type, the CoupledValue concept),
/// not from reading a header that is not here -- it is deliberately satisfied by
/// `double` today and by `dual<S, N>` once that type lands, without this header naming
/// either `fuse::CoupledValue` or `fuse::dual` (neither exists to name). When numtypes
/// lands, this concept is replaced by (or forwards to) the real one; nothing downstream
/// of `IntegratorValue` should need to change, by construction, since every requirement
/// below is read from the SAME quoted shape fuse::CoupledValue will have.
namespace fuse::num {

/// Refusal codes shared by every `Integrator<Value>` family (Round 2: moved here from
/// `explicit_integrator.hpp`, which defined them first, so `DirkIntegrator`/
/// `RosenbrockIntegrator` can refuse the SAME named conditions without a duplicate
/// symbol definition -- `kMassUnsupportedCode` stays local to `explicit_integrator.hpp`
/// since only the explicit family actually refuses on a mass matrix; DIRK and
/// Rosenbrock both honour `TemporalProblem::mass` instead (task item 2/3)).
inline constexpr const char* kNoRhsCode = "E-INTEGRATOR-NO-RHS";
inline constexpr const char* kWrongFamilyCode = "E-INTEGRATOR-WRONG-FAMILY";
inline constexpr const char* kDimensionMismatchCode = "E-INTEGRATOR-DIMENSION-MISMATCH";

/// Primary template: T IS its own scalar type (the case for `double`/`float` -- the
/// only case this checkout can exercise). Specialised for `dual<S, N>` once
/// fuse/num/dual.hpp lands, mirroring `value_traits<T>::scalar_type`'s own convention
/// (the handoff notes quoted in this task: "value_traits<T>::scalar_type" for
/// `dual<S,N>` is `S`).
template <class T>
struct integrator_scalar {
  using type = T;
};

template <class T>
using integrator_scalar_t = typename integrator_scalar<T>::type;

/// A state-element type an Integrator can step. `double` satisfies this today; a future
/// `dual<S, N>` (numtypes) satisfies it once `integrator_scalar<dual<S,N>>` is
/// specialised to `S` -- this concept does not need to change for that, since every
/// requirement below is phrased in terms of `integrator_scalar_t<T>`, never `T` alone,
/// for exactly the quantities (scaling by a step size or a Butcher coefficient) that a
/// dual number's tangent components propagate through automatically.
///
/// Section 4.3 of the design doc: the step size and the embedded error norm are ALWAYS
/// `integrator_scalar_t<Value>` (an `S`, never a `Value` itself, never a dual) -- "a
/// step size with a nonzero tangent would mean the schedule point moves as a function
/// of the sensitivity parameter", out of this track's scope. That decision is why this
/// concept requires scaling a `T` by an `integrator_scalar_t<T>` rather than by a `T`.
template <class T>
concept IntegratorValue = requires(T a, T b, integrator_scalar_t<T> s) {
  requires std::is_floating_point_v<integrator_scalar_t<T>>;
  requires std::is_default_constructible_v<T>;
  requires std::is_copy_constructible_v<T>;
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { s * a } -> std::convertible_to<T>;
  { a * s } -> std::convertible_to<T>;
};

/// A type-erased right-hand side and its optional extensions (owner decision 6:
/// "type-erased Problem" -- a caller builds one of these by assigning `std::function`s
/// rather than this header needing a structural concept over the caller's own class).
/// Mirrors `KARMAInterface`'s own problem-facing surface (Stage A design Section 3,
/// `include/interface/karma_interface.h:70-177`), minus the preprocessing/postprocessing
/// hooks (`step_preprocess` etc.) that belong to the coupling layer above `fuse::num`
/// (the layering constraint, Stage A design Section 3.1: `num` may not depend on
/// `field`/`runtime`), not to the integrator itself.
///
/// `rhs` is required; `mass`/`jacobian_action`/`dfdt` are round-2 hooks (DIRK's/
/// Rosenbrock's own needs, Stage A design Section 3/4.1) declared now so round 2 adds
/// call sites, not a new type. Round 1's explicit family (explicit_integrator.hpp)
/// calls `rhs` only and refuses if a caller's problem sets `mass` (Section 4.1: an
/// explicit method with a nontrivial mass matrix needs a linear solve per stage, which
/// is round-2 scope, not an explicit stage loop silently ignoring M).
template <IntegratorValue Value>
struct TemporalProblem {
  using Scalar = integrator_scalar_t<Value>;

  /// out <- f(t, y). Required; default-constructed (empty) means "no problem set",
  /// which every Integrator refuses with E-INTEGRATOR-NO-RHS before taking a step.
  std::function<Result<void>(Scalar t, std::span<const Value> y, std::span<Value> out)> rhs;

  /// out <- M * v. Absent (round 1) means M = I; present is round-2 scope.
  std::function<Result<void>(std::span<const Value> v, std::span<Value> out)> mass;

  /// out <- J(t, y) * v. Round-2 scope (DIRK's Newton solve, Rosenbrock's linear one).
  std::function<Result<void>(Scalar t, std::span<const Value> y, std::span<const Value> v,
                             std::span<Value> out)> jacobian_action;

  /// out <- d/dt f(t, y). Round-2 scope (Rosenbrock's own non-autonomous correction,
  /// `RosenbrockIntegrator::dfdt`, upstream/karma/include/numerics/temporal/
  /// rosenbrock_integrator.h:41).
  std::function<Result<void>(Scalar t, std::span<const Value> y, std::span<Value> out)> dfdt;

  std::size_t dimension = 0;

  [[nodiscard]] bool has_rhs() const noexcept { return static_cast<bool>(rhs); }
  [[nodiscard]] bool has_mass() const noexcept { return static_cast<bool>(mass); }
  [[nodiscard]] bool has_jacobian_action() const noexcept { return static_cast<bool>(jacobian_action); }
  [[nodiscard]] bool has_dfdt() const noexcept { return static_cast<bool>(dfdt); }
};

/// One accepted-or-rejected step's own record. `error_norm`/`h_used`/`h_next` are
/// `Scalar` (Section 4.3: primal-only), never `Value` -- a `dual<S,N>` state still
/// reports a plain-`S` error norm and step size, by the same decision.
template <IntegratorValue Value>
struct StepRecord {
  using Scalar = integrator_scalar_t<Value>;
  Scalar t_new = Scalar{};
  Scalar h_used = Scalar{};
  Scalar h_next = Scalar{};
  std::optional<Scalar> error_norm;
  bool accepted = true;
  std::size_t function_evaluations = 0;
};

/// The interface every integrator family implements (Section 4.1 of the design doc).
/// Round 1 has exactly one concrete subtype, `ExplicitIntegrator<Value>`
/// (explicit_integrator.hpp); round 2 adds a DIRK one (built on `fuse::num::solve()`)
/// and a Rosenbrock one (built on `gmres`/`MatrixFreeOperator`, Stage A design Section
/// 3.2) beside it, both satisfying this SAME interface so a caller holding an
/// `Integrator<Value>&` never needs to know which family it got.
template <IntegratorValue Value>
class Integrator {
 public:
  using Scalar = integrator_scalar_t<Value>;

  virtual ~Integrator() = default;

  /// Attempts one step of (up to) size `h` from `t`, mutating `state` in place only if
  /// the step is ACCEPTED (an embedded method's rejected step must leave `state`
  /// untouched -- `ExplicitIntegrator::step`'s own contract, tested by
  /// explicit_integrator_test.cpp's own rejection case). Returns a `Failure` for a
  /// named refusal (E-INTEGRATOR-*, embedded_controller.hpp and this header), never for
  /// an ordinary rejection (a rejection is reported as `StepRecord::accepted == false`,
  /// not a `Result` failure -- it is the expected outcome of an adaptive method doing
  /// its job, not an error).
  [[nodiscard]] virtual Result<StepRecord<Value>> step(
      TemporalProblem<Value>& problem, Scalar t, Scalar h, std::span<Value> state) = 0;

  [[nodiscard]] virtual const ButcherTableau& tableau() const noexcept = 0;
};

/// A functor reducing n already rtol/atol-scaled error components
/// (Hairer's own convention, e(i) = err(i) / (atol + rtol*|y(i)|), dopri45_integrator.C:
/// 212-217) to the ONE scalar an EmbeddedController::propose() call decides on. Task
/// item 4: "a norm functor the caller supplies (DistVector::norm2 for distributed
/// state, plain for serial) so agreement across ranks is by construction" -- this
/// header defines only the serial one; a distributed caller supplies its own closure
/// over `fuse::linalg::DistVector::norm2()` without this header ever naming
/// `DistVector` (the same "named, not included" discipline
/// `fuse::num::newton_krylov.hpp`'s own `PreconditionerBlock::apply` uses for
/// `fuse::LinearizationToken`), which is also why round 1 needs no MPI code at all
/// (task's own instruction): the reduction, when there is one to do, is entirely the
/// CALLER's closure, never this header's.
using ScaledNormFn = std::function<double(std::span<const double> scaled_error)>;

/// `sqrt(sum(e_i^2) / n)` -- DOPRI45Integrator's own formula
/// (dopri45_integrator.C:240-245, `err = sqrt(err/n)`), computed locally (no reduction):
/// the default for a state that is not distributed across ranks at all.
[[nodiscard]] inline double serial_rms_norm(std::span<const double> scaled_error) {
  if (scaled_error.empty()) return 0.0;
  double sum_sq = 0.0;
  for (double v : scaled_error) sum_sq += v * v;
  return std::sqrt(sum_sq / static_cast<double>(scaled_error.size()));
}

// ===========================================================================
// SubStepClock -- owner decision (3), 2026-10-27: "a Rational time accumulator with a
// recorded per-window quantum, last step clipped exactly".
// ===========================================================================

/// Section 4.4/6.4 of the Stage A design named the problem this solves: an adaptive
/// sub-stepper's own `h` is proposed in floating point (EmbeddedController::propose,
/// embedded_controller.hpp, returns a `double`), while the coupling window and
/// `Waveform::push()` (fuse/num/waveform.hpp:216) both want an EXACT `Rational` sample
/// time (fuse/base/rational.hpp) -- converting an ACCUMULATED floating time back to an
/// exact Rational after the fact is not well-defined, because the double it would
/// convert is already lossy (the sum of many previous floating `h`'s).
///
/// SubStepClock sidesteps the conversion instead of performing it after the fact: the
/// window `[window_begin, window_end)` is divided, ONCE, at construction, into
/// `quantum_count` EQUAL exact quanta (`Rational` division of two already-exact
/// values is exact by construction, fuse/base/rational.hpp's own `operator/`), and
/// every sub-step advances by an INTEGER number of those quanta -- never by a floating
/// `h` rounded to a `Rational` after accumulating. `round_to_quantum()` is the ONE
/// place a floating proposal becomes an integer; `now()` is always
/// `window_begin + elapsed_quanta * quantum`, which is exact for every `elapsed_quanta`
/// by the same reasoning Rational's own arithmetic guarantees for any finite sum of
/// exact terms.
class SubStepClock {
 public:
  /// DEFAULT QUANTUM COUNT, DERIVED: 2^20 (1,048,576). A relative sub-step resolution
  /// of 2^-20 ~= 9.5e-7 of the window length is roughly THREE decades finer than the
  /// coarsest step an order-convergence study's own decade ever proposes (Stage A
  /// design Section 5.1's own verification plan measures the observed order over "a
  /// decade of h above the round-off floor", and a tier-0 test problem's own coarsest h
  /// in that decade is never finer than ~1e-2 to 1e-3 relative to the window -- MEASURED
  /// in explicit_integrator_test.cpp's own convergence rows, printed alongside the
  /// derived order tolerance). The SAME choice keeps `quantum_count * window.den()`
  /// comfortably inside int64 after reduction for any window denominator up to about
  /// 2^42 (`Rational`'s own 128-bit arithmetic intermediate, fuse/base/rational.hpp's
  /// own `detail::Int128`, absorbs the pre-reduction product regardless; the reduced
  /// RESULT is what must fit int64), far above every window denominator an actual
  /// schedule in this project has produced (`docs/user/brusselator.md`'s own window is
  /// 1/50; every other worked example's own denominator is a comparably small integer).
  static constexpr std::int64_t kDefaultQuantumCount = std::int64_t{1} << 20;

  SubStepClock(Rational window_begin, Rational window_end,
               std::int64_t quantum_count = kDefaultQuantumCount)
      : begin_(window_begin),
        end_(window_end),
        quantum_count_(quantum_count),
        quantum_((window_end - window_begin) / Rational(quantum_count)),
        elapsed_quanta_(0) {
    if (quantum_count <= 0) {
      FUSE_THROW(InvalidArgument,
                "SubStepClock: quantum_count must be positive, got {}", quantum_count);
    }
    if (window_end < window_begin) {
      FUSE_THROW(InvalidArgument,
                "SubStepClock: window_end ({}) precedes window_begin ({})",
                window_end.to_string(), window_begin.to_string());
    }
  }

  [[nodiscard]] Rational now() const {
    return begin_ + quantum_ * Rational(elapsed_quanta_);
  }
  [[nodiscard]] Rational window_begin() const noexcept { return begin_; }
  [[nodiscard]] Rational window_end() const noexcept { return end_; }
  [[nodiscard]] Rational quantum() const noexcept { return quantum_; }
  [[nodiscard]] std::int64_t quantum_count() const noexcept { return quantum_count_; }
  [[nodiscard]] std::int64_t elapsed_quanta() const noexcept { return elapsed_quanta_; }

  /// Exact (Rational comparison, no tolerance): the owner's own "last step clipped
  /// exactly" standard, satisfied by equality, not by "close enough".
  [[nodiscard]] bool at_window_end() const { return now() == end_; }

  /// `(window_end - now()) / quantum`, as an exact integer. Always exact in practice
  /// (the window and the quantum share the same construction, so their ratio reduces
  /// to a whole number at any point the clock can be in); FUSE_ASSERT catches it if a
  /// future change ever broke that invariant rather than silently narrowing a
  /// fractional Rational to its floor.
  [[nodiscard]] std::int64_t remaining_quanta() const {
    // A zero-length window (begin == end) makes quantum_ itself the Rational VALUE
    // zero (0 divided by quantum_count at construction); dividing BY zero is refused
    // by Rational::operator/= (fuse/base/rational.hpp:195-198) regardless of the
    // numerator, so this case is handled directly: a zero-length window trivially has
    // zero quanta remaining, always, by definition, with nothing to divide.
    if (quantum_.is_zero()) return 0;
    const Rational left = end_ - now();
    const Rational steps = left / quantum_;
    FUSE_ASSERT(steps.is_integer(),
               "SubStepClock::remaining_quanta: {} is not a whole number of quanta "
               "(window/quantum invariant broken)", steps.to_string());
    return steps.num();
  }

  /// Rounds a controller-proposed floating step to the nearest whole number of quanta
  /// (never truncation alone, which would bias every step short and could stall
  /// progress at a fine quantum), floored at one quantum (a proposal that rounds to
  /// zero would never advance time -- a defensive minimum, not a substitute for
  /// EmbeddedController's own E-INTEGRATOR-STEP-TOO-SMALL refusal, which fires before a
  /// proposal this small would ever reach here), and CLIPPED to the quanta remaining in
  /// the window -- the owner's "last step clipped exactly": a caller that always
  /// advances by this function's own return value can never leave the window, by
  /// construction, not by a separate post-hoc check.
  [[nodiscard]] std::int64_t round_to_quantum(double h_proposed) const {
    const double q = quantum_.to_double();
    std::int64_t n = (q > 0.0) ? static_cast<std::int64_t>(std::llround(h_proposed / q)) : 1;
    if (n < 1) n = 1;
    const std::int64_t remaining = remaining_quanta();
    if (n > remaining) n = remaining;
    return n;
  }

  /// The ONE mutator. `n` must be a positive count of quanta not exceeding
  /// `remaining_quanta()` -- always true of `round_to_quantum`'s own return value, so a
  /// caller that routes every proposed step through it first cannot violate this.
  void advance(std::int64_t n) {
    FUSE_ASSERT(n > 0, "SubStepClock::advance: n must be positive, got {}", n);
    FUSE_ASSERT(n <= remaining_quanta(),
               "SubStepClock::advance: {} quanta exceeds the {} remaining in the window",
               n, remaining_quanta());
    elapsed_quanta_ += n;
  }

 private:
  Rational begin_;
  Rational end_;
  std::int64_t quantum_count_;
  Rational quantum_;
  std::int64_t elapsed_quanta_;
};

}  // namespace fuse::num
