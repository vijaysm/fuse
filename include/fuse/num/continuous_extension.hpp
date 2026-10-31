#pragma once

#include <cstddef>
#include <vector>

#include "fuse/num/integrator.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 2, task item 5: dense output.
///
/// DEVIATION, stated up front rather than left for a reader to discover: the task
/// asks for "DOPRI5's free fourth-order interpolant (Hairer, Norsett and Wanner I,
/// Section II.6, the b_i(theta) polynomials, cited)". This round's own search for
/// that polynomial's exact coefficients found one machine-readable secondary source
/// (a Boost.odeint header reproducing Dormand & Prince's own continuous extension),
/// but the extracted text was garbled/incomplete (the auxiliary polynomials X1..X7 and
/// the combining constants A, B, C, D came back with missing operators and unclear
/// grouping) -- entering a quartic interpolant from an incomplete transcription would
/// risk shipping a SILENTLY wrong formula (a wrong dense-output interpolant "fails
/// silently, giving large interior errors on a step whose endpoints are perfect",
/// exactly the failure mode the task's own report contract exists to prevent). This
/// header instead implements the CUBIC HERMITE interpolant -- match value AND
/// derivative at both endpoints of the accepted step, a standard, unambiguous
/// construction (the same one MATLAB's own `ode45` dense output and many other
/// production solvers use when a method-specific higher-order interpolant is not
/// available) -- and `continuous_extension_test.cpp` MEASURES its actual order rather
/// than asserting the task's own "fourth order" label. The genuine DOP5(4)-specific
/// quartic interpolant and the stiffly accurate ESDIRK pairs' own published b(theta)
/// (Kennedy & Carpenter give them) are both left as a Round 3 open question, not
/// silently assumed equivalent to what ships here.
///
/// `fuse::Waveform` consumption (design only, task item 5's own instruction -- "design
/// only, no coupling code this round"): a coupling window's own temporal interpolant
/// needs `evaluate(theta)` for theta in [0,1] inside an ACCEPTED step, which is
/// exactly `ContinuousExtension<Value>::evaluate()`'s own signature below;
/// `fuse::num::Waveform::push()` (fuse/num/waveform.hpp) already accepts one exact
/// `Rational` sample per call, so a caller wanting a DENSE (not just end-of-window)
/// sample would call `push()` at a `Rational` theta computed from `SubStepClock`'s own
/// `now()`/`quantum()` (both already exact), evaluating THIS header's own
/// `evaluate(static_cast<Scalar>(theta))` to get the `Value` to push -- no new type is
/// needed on either side, which is why this round adds no `Waveform`-facing code.
namespace fuse::num {

/// The cubic Hermite interpolant over one accepted step `[t_n, t_n+h]`: matches
/// `y(t_n)=y0`, `y'(t_n)=f0`, `y(t_n+h)=y1`, `y'(t_n+h)=f1` exactly at `theta=0,1`.
/// `f0`/`f1` are the SAME right-hand-side evaluations an integrator already computed
/// for its own first and last stage (DOPRI45's own FSAL structure, Round 1's own
/// correction to `is_FSAL`, means `f1` costs no NEW evaluation for that family) --
/// this header does not recompute them, a caller supplies what it already has.
template <IntegratorValue Value>
class CubicHermiteExtension {
 public:
  using Scalar = integrator_scalar_t<Value>;

  CubicHermiteExtension(Scalar h, std::vector<Value> y0, std::vector<Value> f0,
                        std::vector<Value> y1, std::vector<Value> f1)
      : h_(h), y0_(std::move(y0)), f0_(std::move(f0)), y1_(std::move(y1)), f1_(std::move(f1)) {}

  /// `theta` in `[0, 1]`; `theta=0` returns `y0` exactly, `theta=1` returns `y1`
  /// exactly (both to round-off, not merely "close": the Hermite basis functions
  /// satisfy `h00(0)=1,h00(1)=0,h10(0)=h10(1)=0,h01(0)=0,h01(1)=1,h11(0)=h11(1)=0`
  /// exactly in exact arithmetic).
  [[nodiscard]] std::vector<Value> evaluate(Scalar theta) const {
    const Scalar t2 = theta * theta;
    const Scalar t3 = t2 * theta;
    const Scalar h00 = Scalar{2} * t3 - Scalar{3} * t2 + Scalar{1};
    const Scalar h10 = t3 - Scalar{2} * t2 + theta;
    const Scalar h01 = Scalar{-2} * t3 + Scalar{3} * t2;
    const Scalar h11 = t3 - t2;
    std::vector<Value> out(y0_.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = h00 * y0_[i] + (h_ * h10) * f0_[i] + h01 * y1_[i] + (h_ * h11) * f1_[i];
    }
    return out;
  }

 private:
  Scalar h_;
  std::vector<Value> y0_;
  std::vector<Value> f0_;
  std::vector<Value> y1_;
  std::vector<Value> f1_;
};

}  // namespace fuse::num
