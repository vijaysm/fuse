#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <variant>
#include <vector>

#include "fuse/num/integrator.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 3 (task item 1): dense output.
///
/// Round 2 shipped only the cubic Hermite interpolant (a stated deviation from DOPRI5's
/// own published quartic, after this track declined to transcribe it from a garbled
/// secondary source). Round 3 has the primary sources in hand: Hairer's own `dopri5.f`
/// (the author's reference FORTRAN, github.com/pv/dopri5_interface) for DOPRI5's own
/// quartic continuous extension, and the Kennedy & Carpenter NASA TM (artifact
/// 6dd464e5-d3ed-411c-8e99-ccd5ca95087d, Appendix D) for the new ESDIRK pairs' own
/// dense-output rows -- both reproduced below, neither re-derived.
///
/// Three concrete interpolants now exist behind one reporting wrapper,
/// `ContinuousExtension<Value>`, which tells a caller WHICH one it used (task's own
/// instruction) rather than silently picking one:
///   - `Dopri5QuarticExtension`: DOPRI5's own 4th-order "free" interpolant (Shampine
///     1986, cited explicitly in `dopri5.f`'s own comment -- the published choice, not
///     this round's own derivation), needing all 7 of DOPRI45's own stage derivatives.
///   - `PolynomialDenseOutput`: the generic `b_i(theta) = sum_j b*_ij * theta^j` form
///     Kennedy & Carpenter publish for their own ESDIRK pairs (eq. 54 of the TM's main
///     text), driven by a tableau's own `dense_output` table (butcher_tableau.hpp).
///   - `CubicHermiteExtension`: Round 2's own fallback, unchanged, used when neither of
///     the above has data for the tableau at hand (STILL the only option for every
///     classical/SDIRK/Rosenbrock tableau this project ships, none of which publish a
///     b*_ij dense-output table of their own).
namespace fuse::num {

/// Which concrete interpolant a `ContinuousExtension` ended up using -- the task's own
/// "reports WHICH it used" instruction, read literally: a caller inspects this rather
/// than needing to know in advance which family a tableau belongs to.
enum class DenseOutputKind { CubicHermite, Dopri5Quartic, PolynomialBStar };

/// Unchanged from Round 2 except for the one-line addition of `kind()`, so existing
/// Round 2 callers (none outside this file and its own test) keep compiling.
template <IntegratorValue Value>
class CubicHermiteExtension {
 public:
  using Scalar = integrator_scalar_t<Value>;

  CubicHermiteExtension(Scalar h, std::vector<Value> y0, std::vector<Value> f0,
                        std::vector<Value> y1, std::vector<Value> f1)
      : h_(h), y0_(std::move(y0)), f0_(std::move(f0)), y1_(std::move(y1)), f1_(std::move(f1)) {}

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

  [[nodiscard]] static constexpr DenseOutputKind kind() noexcept {
    return DenseOutputKind::CubicHermite;
  }

 private:
  Scalar h_;
  std::vector<Value> y0_;
  std::vector<Value> f0_;
  std::vector<Value> y1_;
  std::vector<Value> f1_;
};

/// DOPRI5's own quartic dense output, EXACTLY as `dopri5.f`'s own `CONTD5` function and
/// `CDOPRI` subroutine compute it (Hairer & Wanner's reference implementation) -- not
/// algebraically simplified or re-derived by this round, to avoid introducing a
/// transcription slip of its own on top of the primary source. `k` holds DOPRI45's own
/// seven stage derivatives for the step (`k[0]` = f(t0,y0), `k[6]` = the FSAL stage
/// f(t0+h, y1); DOPRI45's own Round 1 correction to `is_FSAL` is exactly what makes
/// `k[6]` free). `y0`, `y1` are the step's own endpoints.
template <IntegratorValue Value>
class Dopri5QuarticExtension {
 public:
  using Scalar = integrator_scalar_t<Value>;

  Dopri5QuarticExtension(Scalar h, std::vector<Value> y0, std::vector<Value> y1,
                         std::array<std::vector<Value>, 7> k)
      : h_(h), y0_(std::move(y0)), y1_(std::move(y1)), k_(std::move(k)) {}

  [[nodiscard]] std::vector<Value> evaluate(Scalar theta) const {
    // Shampine (1986)'s own constants, `dopri5.f`'s own `CDOPRI` subroutine, byte-for-byte.
    constexpr Scalar d1 = Scalar(-12715105075.0) / Scalar(11282082432.0);
    constexpr Scalar d3 = Scalar(87487479700.0) / Scalar(32700410799.0);
    constexpr Scalar d4 = Scalar(-10690763975.0) / Scalar(1880347072.0);
    constexpr Scalar d5 = Scalar(701980252875.0) / Scalar(199316789632.0);
    constexpr Scalar d6 = Scalar(-1453857185.0) / Scalar(822651844.0);
    constexpr Scalar d7 = Scalar(69997945.0) / Scalar(29380423.0);
    const Scalar theta1 = Scalar{1} - theta;
    std::vector<Value> out(y0_.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
      const Value ydiff = y1_[i] - y0_[i];
      const Value bspl = h_ * k_[0][i] - ydiff;
      // `CONTD5`'s own `CONT(3*ND+J)`: literally `-h*k7 + ydiff - bspl`, not the
      // algebraically-equal `2*ydiff - h*(k1+k7) - bspl` -- coded as the source writes
      // it so a slip in the hand-simplified form cannot be introduced here.
      const Value c3 = Value{-h_ * k_[6][i]} + ydiff - bspl;
      const Value c4 = h_ * (d1 * k_[0][i] + d3 * k_[2][i] + d4 * k_[3][i] + d5 * k_[4][i] +
                             d6 * k_[5][i] + d7 * k_[6][i]);
      // `CONTD5`'s own nested (Horner) form: y0 + theta*(ydiff + theta1*(bspl +
      // theta*(c3 + theta1*c4))).
      out[i] = y0_[i] + theta * (ydiff + theta1 * (bspl + theta * (c3 + theta1 * c4)));
    }
    return out;
  }

  [[nodiscard]] static constexpr DenseOutputKind kind() noexcept {
    return DenseOutputKind::Dopri5Quartic;
  }

 private:
  Scalar h_;
  std::vector<Value> y0_;
  std::vector<Value> y1_;
  std::array<std::vector<Value>, 7> k_;
};

/// The generic Kennedy & Carpenter dense-output form (the TM's own eq. 54,
/// butcher_tableau.hpp's own `ButcherTableau::dense_output` doc comment):
/// `b_i(theta) = sum_{j=1}^{p*} b*_ij * theta^j`, driven entirely by a tableau's own
/// `dense_output` table -- no tableau-specific code here, so adding a FOURTH pair's own
/// dense output later (Round 4's deferred ESDIRK/Kvaerno pairs) needs no new class.
/// `k` holds the RAW stage derivatives `f(t_i, y_i)` (the Butcher-tableau convention
/// every family in this project already uses for `k`, NOT an h-scaled stage increment).
template <IntegratorValue Value>
class PolynomialDenseOutput {
 public:
  using Scalar = integrator_scalar_t<Value>;

  PolynomialDenseOutput(Scalar h, std::vector<Value> y0, std::vector<std::vector<Value>> k,
                        std::vector<std::vector<double>> b_star)
      : h_(h), y0_(std::move(y0)), k_(std::move(k)), b_star_(std::move(b_star)) {}

  [[nodiscard]] std::vector<Value> evaluate(Scalar theta) const {
    std::vector<Value> out = y0_;
    const std::size_t n = y0_.size();
    const std::size_t stages = k_.size();
    for (std::size_t i = 0; i < stages; ++i) {
      Scalar bi_theta{0};
      Scalar theta_pow = theta;
      for (double coeff : b_star_[i]) {
        bi_theta += static_cast<Scalar>(coeff) * theta_pow;
        theta_pow *= theta;
      }
      for (std::size_t m = 0; m < n; ++m) {
        out[m] = out[m] + (h_ * bi_theta) * k_[i][m];
      }
    }
    return out;
  }

  [[nodiscard]] static constexpr DenseOutputKind kind() noexcept {
    return DenseOutputKind::PolynomialBStar;
  }

 private:
  Scalar h_;
  std::vector<Value> y0_;
  std::vector<std::vector<Value>> k_;
  std::vector<std::vector<double>> b_star_;
};

/// The reporting wrapper (task item 1's own "ContinuousExtension reports WHICH it
/// used" instruction): one type a caller holds regardless of which concrete
/// interpolant backs a given step, with `kind()` naming it.
template <IntegratorValue Value>
class ContinuousExtension {
 public:
  using Scalar = integrator_scalar_t<Value>;
  using Variant = std::variant<CubicHermiteExtension<Value>, Dopri5QuarticExtension<Value>,
                               PolynomialDenseOutput<Value>>;

  explicit ContinuousExtension(Variant v) : v_(std::move(v)) {}

  [[nodiscard]] std::vector<Value> evaluate(Scalar theta) const {
    return std::visit([theta](const auto& ext) { return ext.evaluate(theta); }, v_);
  }

  [[nodiscard]] DenseOutputKind kind() const noexcept {
    return std::visit([](const auto& ext) { return ext.kind(); }, v_);
  }

 private:
  Variant v_;
};

}  // namespace fuse::num
