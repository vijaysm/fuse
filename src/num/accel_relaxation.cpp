#include "fuse/num/accelerator.hpp"
#include "fuse/base/error.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// Spec Section 5.1 `Accelerate`, Section 5.3 implicit coupling groups: the members of the family
/// that need at most the previous residual -- None, Constant, Aitken and
/// Steffensen -- plus the shared validation and the factory.
///
/// Accumulation order: every reduction below runs strictly sequentially in
/// ascending index order into one scalar, matching the convention documented
/// at the top of fuse/num/kernels.hpp. The results are bit-reproducible for a
/// given buffer and not invariant under reassociation.
namespace fuse {

namespace detail {

std::string format_accel_double(double v) {
  // Not std::to_string(double) and not a default-constructed stream: both
  // format through the global locale, so "0.5" becomes "0,5" under de_DE and
  // any byte-for-byte comparison of the message breaks. Same reasoning as
  // detail::format_convergence_double in num/convergence.hpp.
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << v;
  return os.str();
}

Result<void> check_update_args(const double* x_old, const double* x_new,
                               const double* x_out, std::size_t n,
                               int iteration) {
  if (iteration < 0) {
    return Failure{"E-ACCEL-ITERATION",
                 "iteration index must be non-negative, got " +
                     std::to_string(iteration)};
  }
  if (n == 0) return Result<void>::success();
  if (x_old == nullptr || x_new == nullptr || x_out == nullptr) {
    return Failure{"E-ACCEL-NULL",
                 "accelerator update given a null pointer for a length-" +
                     std::to_string(n) + " vector"};
  }
  // Every kind records x_old and x_new into its history *after* filling
  // x_out, so writing the output over an input silently stores the wrong
  // secant pair -- a wrong answer, not a crash, and one that survives ASan and
  // UBSan because the aliased writes are perfectly well defined.
  //
  // Partial overlap is therefore rejected as well as exact aliasing, by a
  // range comparison. `<` between pointers into unrelated objects is
  // unspecified, so the comparison goes through std::less<const double*>,
  // which [comparisons]/[defns.order] require to be a strict total order over
  // *all* pointer values of the type and to agree with `<` wherever `<` is
  // itself defined -- in particular within one array. Two ranges that really
  // do overlap have their pointers in a single array object, so `<` is defined
  // for them and std::less must agree with it: the test has no false
  // negatives. A false positive would need an implementation whose total order
  // interleaves two disjoint arrays; rejecting such a call is safe, whereas
  // accepting an overlapping one is a silent wrong answer.
  const std::less<const double*> before;
  const auto overlaps = [&before, n](const double* a, const double* b) {
    return before(a, b + n) && before(b, a + n);
  };
  if (overlaps(x_out, x_old) || overlaps(x_out, x_new)) {
    return Failure{"E-ACCEL-ALIAS",
                 "accelerator update requires x_out not to overlap x_old or "
                 "x_new, not even partially"};
  }
  // Rejected up front rather than propagated: a NaN entering the secant
  // history poisons every later solve, and an inf residual would otherwise
  // divide out to a plausible-looking finite omega.
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x_old[i])) {
      return Failure{"E-ACCEL-NONFINITE",
                   "x_old[" + std::to_string(i) + "] is not finite"};
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x_new[i])) {
      return Failure{"E-ACCEL-NONFINITE",
                   "x_new[" + std::to_string(i) + "] is not finite"};
    }
  }
  return Result<void>::success();
}

/// x_out = x_old + w * (x_new - x_old). The fallback every method in the
/// family degrades to when it has no usable history.
///
/// The increment form is the default because it is the only one that preserves
/// a fixed point bit for bit: x_new == x_old gives a difference of exactly 0,
/// so x_out == x_old to the last bit for any w. The algebraically equal convex
/// form (1 - w) * x_old + w * x_new does not -- (1 - w) * x + w * x is not x in
/// floating point -- and swapping to it unconditionally would move every
/// converged iterate by a rounding error.
///
/// It has one failure mode the convex form does not: x_new - x_old can
/// overflow while the blend itself is perfectly representable, e.g. w = 0.5,
/// x_old = -1e308, x_new = +1e308, whose exact blend is 0 but whose difference
/// is +inf. Both inputs are validated finite before this is reached, so the
/// difference being non-finite is precisely that overflow and nothing else;
/// only then is the convex form used, and only for the components affected.
void relax_into(double w, const double* x_old, const double* x_new,
                double* x_out, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const double d = x_new[i] - x_old[i];
    x_out[i] = std::isfinite(d) ? x_old[i] + w * d
                                : (1.0 - w) * x_old[i] + w * x_new[i];
  }
}

Result<void> finish_or_fallback(const char* who, double w, const double* x_old,
                                const double* x_new, double* x_out,
                                std::size_t n) {
  std::size_t bad = n;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(x_out[i])) {
      bad = i;
      break;
    }
  }
  if (bad == n) return Result<void>::success();

  // The step just written is unusable. Replace it wholesale -- not
  // component-wise, which would leave an incoherent mixture of two different
  // steps -- by the under-relaxed fallback, so that a caller that logs the
  // failure and presses on has a finite iterate to press on from.
  relax_into(w, x_old, x_new, x_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (std::isfinite(x_out[i])) continue;
    // Even the fallback overflowed: |w| or the residual is large enough that
    // no relaxed blend of these two finite endpoints is representable. There
    // is no meaningful under-relaxed step left, so the iterate is rejected
    // outright and x_out is restored to x_old, which check_update_args has
    // already validated finite. The postcondition "x_out is finite" survives;
    // the message says that this is what happened.
    for (std::size_t j = 0; j < n; ++j) x_out[j] = x_old[j];
    return Failure{"E-ACCEL-DIVERGED",
                 std::string{who} + " produced a non-finite iterate at index " +
                     std::to_string(bad) +
                     "; the under-relaxed fallback overflowed too, so x_out "
                     "holds x_old unchanged"};
  }
  return Failure{"E-ACCEL-DIVERGED",
               std::string{who} + " produced a non-finite iterate at index " +
                   std::to_string(bad) +
                   "; x_out holds the under-relaxed fallback"};
}

/// Per-block reciprocal-2-norm weights for the least-squares row scaling.
/// `offsets` is the validated field partition. An empty partition is *not* one
/// block: it means no scaling at all, so every weight comes back as exactly 1
/// and no reciprocal norm is formed. An explicit single block (`{0}`) does form
/// the scalar 1/||r||; it leaves the least-squares minimiser unchanged, but it
/// is not the same computation.
///
/// A block gets weight 1 whenever its reciprocal norm is not a usable positive
/// finite number. Two cases reach that guard and both are ordinary rather than
/// pathological: a block that has converged far enough for its sum of squares
/// to underflow to exactly 0, and a block whose sum of squares overflowed to
/// inf. Without the guard those become 1/0 == inf and 1/inf == 0, and an inf
/// weight multiplied by a zero row is a NaN in the least-squares matrix.
///
/// The `isfinite(inv)` arm is defensive only: `norm` is a square root, so the
/// smallest positive value it can take is about 2.2e-162 and 1/norm cannot
/// overflow. It is kept because the cost is one comparison and the failure it
/// guards against is silent.
///
/// Note that this guard is not the last line of defence. A non-finite entry
/// that did reach the least-squares matrix would be caught by
/// qn_least_squares(), which rejects a non-finite column norm and reports
/// rank 0, so the accelerator would degrade to relaxation rather than emit a
/// NaN. The guard exists so the degradation is a documented weight of 1 for
/// that block, not the loss of every block's fit at once.
void field_weights(const std::vector<std::size_t>& offsets, const double* r,
                   std::size_t n, std::vector<double>& out) {
  out.assign(n, 1.0);
  if (offsets.empty()) return;
  for (std::size_t b = 0; b < offsets.size(); ++b) {
    const std::size_t lo = offsets[b];
    const std::size_t hi = (b + 1 < offsets.size()) ? offsets[b + 1] : n;
    if (lo >= n) break;
    const std::size_t end = hi < n ? hi : n;
    double s = 0.0;
    for (std::size_t i = lo; i < end; ++i) s += r[i] * r[i];
    const double norm = std::sqrt(s);
    const double inv  = 1.0 / norm;
    const double w = (norm > 0.0 && std::isfinite(norm) && inv > 0.0 &&
                      std::isfinite(inv))
                         ? inv
                         : 1.0;
    for (std::size_t i = lo; i < end; ++i) out[i] = w;
  }
}

/// Defined in accel_quasi_newton.cpp. Declared here rather than in the public
/// header so the least-squares machinery stays an implementation detail; the
/// two translation units are the only participants.
std::unique_ptr<Accelerator> make_quasi_newton(const AccelConfig& cfg);

std::size_t total_rows(std::span<const RestartRowBlock> blocks) {
  std::size_t n = 0;
  for (const RestartRowBlock& b : blocks) {
    n += b.ids.size() * (b.window_capacity > 0 ? b.window_capacity : 1);
  }
  return n;
}

Result<void> put_field_per_block(RestartWriter& writer,
                                 const std::string& name_prefix,
                                 std::span<const RestartRowBlock> blocks,
                                 std::span<const double> flat,
                                 int components) {
  std::size_t off = 0;
  for (const RestartRowBlock& block : blocks) {
    const std::size_t n_elem = block.ids.size();
    const std::string qualified = name_prefix + "/" + block.field_name;
    if (block.window_capacity == 0) {
      // Ordinary case (EXACT-KEY scheme): one row per id, contiguous in
      // `flat`.
      const Result<void> r = writer.put_field(
          qualified, block.ids,
          flat.subspan(off * static_cast<std::size_t>(components),
                       n_elem * static_cast<std::size_t>(components)),
          components, static_cast<std::int64_t>(n_elem));
      if (!r.ok()) return r;
      off += n_elem;
      continue;
    }
    // Window-scoped case: this
    // block's own slice of `flat` is `n_elem * window_capacity` rows,
    // element-major sample-minor (index(e, s) = e*window_capacity + s).
    // Demux into `window_capacity` separate, per-sample, per-element-id
    // sub-datasets -- a strided gather per sample, since the per-sample
    // rows are NOT contiguous in `flat` (they are contiguous per ELEMENT,
    // not per sample).
    const std::size_t cap = block.window_capacity;
    std::vector<double> sample_vals(n_elem * static_cast<std::size_t>(components));
    for (std::size_t s = 0; s < cap; ++s) {
      for (std::size_t e = 0; e < n_elem; ++e) {
        const std::size_t src_row = off + e * cap + s;
        for (int c = 0; c < components; ++c) {
          sample_vals[e * static_cast<std::size_t>(components) +
                      static_cast<std::size_t>(c)] =
              flat[src_row * static_cast<std::size_t>(components) +
                   static_cast<std::size_t>(c)];
        }
      }
      const std::string sample_qualified =
          qualified + "/sample" + std::to_string(s);
      const Result<void> r = writer.put_field(
          sample_qualified, block.ids, std::span<const double>(sample_vals),
          components, static_cast<std::int64_t>(n_elem));
      if (!r.ok()) return r;
    }
    off += n_elem * cap;
  }
  return Result<void>::success();
}

Result<void> get_field_per_block(RestartReader& reader,
                                 const std::string& name_prefix,
                                 std::span<const RestartRowBlock> blocks,
                                 std::span<double> flat, int components) {
  std::size_t off = 0;
  for (const RestartRowBlock& block : blocks) {
    const std::size_t n_elem = block.ids.size();
    const std::string qualified = name_prefix + "/" + block.field_name;
    if (block.window_capacity == 0) {
      const Result<fuse::ReadConversionReport> r = reader.get_field(
          qualified, block.ids,
          flat.subspan(off * static_cast<std::size_t>(components),
                       n_elem * static_cast<std::size_t>(components)),
          components);
      if (!r.ok()) return r.error();
      off += n_elem;
      continue;
    }
    const std::size_t cap = block.window_capacity;
    std::vector<double> sample_vals(n_elem * static_cast<std::size_t>(components));
    for (std::size_t s = 0; s < cap; ++s) {
      const std::string sample_qualified =
          qualified + "/sample" + std::to_string(s);
      const Result<fuse::ReadConversionReport> r = reader.get_field(
          sample_qualified, block.ids, std::span<double>(sample_vals),
          components);
      if (!r.ok()) return r.error();
      for (std::size_t e = 0; e < n_elem; ++e) {
        const std::size_t dst_row = off + e * cap + s;
        for (int c = 0; c < components; ++c) {
          flat[dst_row * static_cast<std::size_t>(components) +
               static_cast<std::size_t>(c)] =
              sample_vals[e * static_cast<std::size_t>(components) +
                         static_cast<std::size_t>(c)];
        }
      }
    }
    off += n_elem * cap;
  }
  return Result<void>::success();
}

}  // namespace detail

const char* to_cstring(AccelKind k) noexcept {
  switch (k) {
    case AccelKind::None:       return "None";
    case AccelKind::Constant:   return "Constant";
    case AccelKind::Aitken:     return "Aitken";
    case AccelKind::Secant:     return "Secant";
    case AccelKind::Steffensen: return "Steffensen";
    case AccelKind::IQN_ILS:    return "IQN-ILS";
    case AccelKind::IQN_IMVJ:   return "IQN-IMVJ";
  }
  return "?";
}

const char* to_cstring(FilterKind f) noexcept {
  switch (f) {
    case FilterKind::None: return "None";
    case FilterKind::QR1:  return "QR1";
    case FilterKind::QR2:  return "QR2";
    case FilterKind::QR3:  return "QR3";
  }
  return "?";
}

void Accelerator::set_field_partition(std::vector<std::size_t> offsets) {
  if (!offsets.empty()) {
    if (offsets.front() != 0) {
      FUSE_THROW(InvalidArgument, "{}", 
          "Accelerator::set_field_partition: the first offset must be 0, got " +
          std::to_string(offsets.front()));
    }
    // Non-decreasing, NOT strictly increasing. Two equal offsets are an
    // empty block: under a real partition a rank can own none of one field
    // of a mesh entity set and some of another, which is as ordinary here as it is in
    // the transport (MpiTransport.RankOwningZeroElementsIsOrdinary). It costs
    // field_weights() nothing -- its inner loops simply do not execute and
    // the block's weight is never read. A *decrease* remains a programmer
    // error: the blocks would overlap.
    // With 4 entities on 5 ranks, some ranks hold none, and rejecting that
    // would throw out of initialize() in resolve_regions() on one rank while
    // the others waited in a collective -- the rejection would not merely be
    // wrong, it would hang the job.
    for (std::size_t i = 1; i < offsets.size(); ++i) {
      if (offsets[i] < offsets[i - 1]) {
        FUSE_THROW(InvalidArgument, "{}", 
            "Accelerator::set_field_partition: offsets must not decrease, "
            "got " +
            std::to_string(offsets[i - 1]) + " then " +
            std::to_string(offsets[i]) + " at index " + std::to_string(i));
      }
    }
  }
  field_offsets_ = std::move(offsets);
}

namespace {

/// Picard: take the new value unchanged.
class NoneAccel final : public Accelerator {
 public:
  void reset() override {}
  [[nodiscard]] std::string name() const override { return "None"; }
  [[nodiscard]] std::size_t columns() const override { return 0; }

  /// Plain Picard carries no state at all between
  /// calls (x_out = x_new, nothing retained) -- trivially, correctly
  /// restartable by doing nothing, which is exactly what the base class's
  /// no-op export/import already do. Overridden here only so the Runner's
  /// restart_state_implemented() check sees `true` rather than the base
  /// class's conservative `false`.
  [[nodiscard]] bool restart_state_implemented() const noexcept override {
    return true;
  }

  /// The base class's export_restart_state() default IS a success no-op
  /// (correct here), but its import_restart_state() default is an
  /// UNCONDITIONAL Failure (by design -- Accelerator::import_restart_state()'s
  /// own doc comment), meant to be unreachable once
  /// restart_state_implemented() reports true. Declaring true above without
  /// overriding THIS method would mean any real Runner-driven restore through
  /// a None-accelerated region hit that unconditional Failure. Genuinely
  /// nothing to restore (no learned state at all), so this is a real no-op,
  /// not a workaround.
  Result<void> import_restart_state(RestartReader&, std::string_view,
                                    std::span<const RestartRowBlock>, bool,
                                    bool& did_reset) override {
    did_reset = false;
    return Result<void>::success();
  }

  Result<void> update(const double* x_old, const double* x_new, double* x_out,
                      std::size_t n, int iteration) override {
    const Result<void> args =
        detail::check_update_args(x_old, x_new, x_out, n, iteration);
    if (!args.ok()) return args;
    for (std::size_t i = 0; i < n; ++i) x_out[i] = x_new[i];
    return Result<void>::success();
  }
};

class ConstantAccel final : public Accelerator {
 public:
  explicit ConstantAccel(double omega) : omega_(omega) {}

  void reset() override {}
  [[nodiscard]] std::string name() const override { return "Constant"; }
  [[nodiscard]] std::size_t columns() const override { return 0; }

  /// `omega_` is a FIXED run option supplied at
  /// construction (AccelConfig::omega, restored identically by the
  /// graph/config the owning run re-reads, not by this interface) --
  /// never learned history, so there is genuinely nothing for restart to
  /// carry. Overridden only so the Runner sees `true` rather than the
  /// base class's conservative `false`.
  [[nodiscard]] bool restart_state_implemented() const noexcept override {
    return true;
  }

  /// Same latent bug, same fix, as NoneAccel above: omega_ is a fixed
  /// run option, never learned history, so there is genuinely nothing
  /// to restore; the base class's import_restart_state() default must
  /// not be relied on (it unconditionally fails by design).
  Result<void> import_restart_state(RestartReader&, std::string_view,
                                    std::span<const RestartRowBlock>, bool,
                                    bool& did_reset) override {
    did_reset = false;
    return Result<void>::success();
  }

  Result<void> update(const double* x_old, const double* x_new, double* x_out,
                      std::size_t n, int iteration) override {
    const Result<void> args =
        detail::check_update_args(x_old, x_new, x_out, n, iteration);
    if (!args.ok()) return args;
    detail::relax_into(omega_, x_old, x_new, x_out, n);
    // The blend at omega_ *is* this kind's under-relaxed fallback, so if it
    // overflowed there is nothing weaker to retry and finish_or_fallback
    // lands on x_old. Passing omega_ keeps the postcondition uniform across
    // the family rather than special-casing the one stateless kind.
    return detail::finish_or_fallback("Constant", omega_, x_old, x_new, x_out,
                                      n);
  }

 private:
  double omega_;
};

/// Aitken under-relaxation: one dynamic omega for the whole vector,
///
///   omega_k = -omega_{k-1} * <r_{k-1}, r_k - r_{k-1}> / ||r_k - r_{k-1}||^2
///   x_{k+1} = x_k + omega_k * r_k,       r_k = x_new - x_old
///
/// which is the standard vector generalisation (Irons & Tuck; the form preCICE
/// implements). The recursion needs the previous omega, so the fallback path
/// must also record the omega it used.
///
/// Near-zero denominator policy: the ratio is formed only when
/// ||r_k - r_{k-1}||^2 is strictly positive and finite AND the resulting omega
/// is finite and non-zero. Otherwise the step degrades to
/// `initial_relaxation`. Two failure modes are covered: a stalled iteration
/// (Delta r == 0 exactly, denominator 0) and an orthogonal residual change
/// (numerator 0, omega 0), which would freeze the iterate and then make every
/// later denominator 0 as well.
///
/// Storage: the current residual is a member, not a local, so a steady-state
/// update() performs no heap allocation -- an accelerated implicit coupling group
/// sits inside the sweep loop, which spec Section 2 layer 3 requires to be allocation
/// free. `r_` and `r_prev_` ping-pong by swap(), so the two buffers are
/// allocated once each, in the first two calls of the first window, and then
/// reused for the life of the object. Emptiness is therefore no longer a
/// usable "no history" signal -- a swapped-out buffer still holds n doubles --
/// so `have_prev_` says so explicitly.
class AitkenAccel final : public Accelerator {
 public:
  explicit AitkenAccel(double initial_relaxation)
      : omega0_(initial_relaxation), omega_(initial_relaxation) {}

  void reset() override {
    have_prev_ = false;
    omega_     = omega0_;
  }

  [[nodiscard]] std::string name() const override { return "Aitken"; }
  [[nodiscard]] std::size_t columns() const override { return 0; }

  Result<void> update(const double* x_old, const double* x_new, double* x_out,
                      std::size_t n, int iteration) override {
    const Result<void> args =
        detail::check_update_args(x_old, x_new, x_out, n, iteration);
    if (!args.ok()) return args;
    if (n == 0) return Result<void>::success();

    // iteration 0 is a window boundary, and a change of vector length
    // invalidates the stored residual outright.
    if (iteration == 0 || n != n_) reset();
    n_ = n;
    // Grow-only. resize() never releases capacity, so this is a no-op from the
    // second window on, and a shrinking n reuses the larger buffer rather than
    // trading it for a smaller one. Every element is written below before it
    // is read, so the value-initialisation resize() performs on a grown buffer
    // is never observable.
    r_.resize(n);
    r_prev_.resize(n);

    for (std::size_t i = 0; i < n; ++i) r_[i] = x_new[i] - x_old[i];

    // The omega recursion must reject a non-positive omega: an unguarded
    // formula accepts any finite, nonzero omega, including a negative one.
    // Measured in a 25-window coupling run: at window 3 the second omega is
    // negative (-0.031, then -0.18), the residual then stops decreasing
    // (frozen at 0.0162 from sweep 4 on, 59 sweeps burned) and omega drifts
    // toward zero. WHY the secant denominator collapses there is INFERRED, not
    // measured: it is not at rounding level (denominator 3e-7 against ||r||^2
    // of 2.6e-4). A convergent real relaxation factor for a STATIONARY scalar
    // iteration with spectrum lambda must lie in (0, 2(1 - Re lambda) /
    // |1 - lambda|^2) (rationale for the RULE, not a proof for this dynamic,
    // nonlinear scheme): a non-positive omega is never in that interval, so
    // rejecting one outright is sound. Three candidate fallbacks for a
    // rejected candidate were measured on that configuration, over all 25
    // windows including the one (window 9) the other two do not fix: keep the
    // last accepted positive omega (fixes windows 3, 6; not 9), the
    // residual-minimising scalar -<r_prev,dr>/||dr||^2 for the same two
    // residuals, without the recursive omega_ factor (same result: fixes 3, 6;
    // not 9), and restart to omega0_ (fixes all three, 3/6/9, and leaves every
    // other window's sweep count changed by at most a few sweeps). Restart to
    // omega0_ is implemented below; it disturbs neither of the two
    // bit-identical restart round-trip tests the other candidates were also
    // checked against (test/verification/brusselator_restart_test.cpp
    // SeparateLayoutJacobiAitken, GrowingLayoutJacobiAitken) -- unlike a
    // noise-threshold-triggered restart variant, which fixes window 9 too but
    // breaks both of those tests by making the restart decision sensitive to
    // ordinary round-off differences between two code paths meant to agree
    // only bit-for-bit, not to a tolerance; this rule takes no such threshold
    // and needs none.
    double w = omega0_;
    if (have_prev_) {
      double num = 0.0;
      double den = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double d = r_[i] - r_prev_[i];
        num += r_prev_[i] * d;
        den += d * d;
      }
      if (den > 0.0 && std::isfinite(den) && std::isfinite(num)) {
        const double candidate = -omega_ * num / den;
        if (std::isfinite(candidate) && candidate > 0.0) {
          w = candidate;
        } else {
          // Reject a non-positive (or non-finite) candidate: restart to
          // omega0_ rather than keep using a value no convergent
          // stationary relaxation could ever equal.
          w = omega0_;
        }
      }
    }
    omega_ = w;

    detail::relax_into(w, x_old, x_new, x_out, n);
    const Result<void> fin =
        detail::finish_or_fallback("Aitken", omega0_, x_old, x_new, x_out, n);
    if (!fin.ok()) {
      // Drop the history that produced the blow-up, as documented on
      // Accelerator::update: reset() restores omega_ to omega0_ as well, so
      // the next call cannot re-derive the same divergent factor.
      reset();
      return fin;
    }
    // Swap rather than copy: the two buffers exchange roles, so neither is
    // ever reallocated and the next call's resize() has nothing to do.
    r_.swap(r_prev_);
    have_prev_ = true;
    return Result<void>::success();
  }

  /// `r_prev_`/`omega_`/`have_prev_` are exactly
  /// the state the NEXT update() call reads (`r_` itself is recomputed
  /// fresh from x_new-x_old every call, so it carries nothing forward
  /// and is not restarted). `have_prev_ == false` (no sweep has
  /// completed since the last reset()) writes nothing -- a reader finding
  /// no fields for this name_prefix leaves `have_prev_` at its
  /// freshly-constructed `false`, which is the bit-identical same-rank
  /// state. EXACT-KEY fix: `r_prev_` is written per BLOCK
  /// (detail::put_field_per_block), one dataset per owning field name,
  /// never one dataset for the whole region keyed by a combined id.
  [[nodiscard]] bool restart_state_implemented() const noexcept override {
    return true;
  }

  Result<void> export_restart_state(
      RestartWriter& writer, std::string_view name_prefix,
      std::span<const RestartRowBlock> blocks) const override {
    if (!have_prev_) return Result<void>::success();
    if (detail::total_rows(blocks) != n_) {
      return Failure{"E-RESTART-BAD-NUMBERING",
                     "AitkenAccel::export_restart_state: " +
                         std::to_string(detail::total_rows(blocks)) +
                         " row global ids for an accelerator of length " +
                         std::to_string(n_)};
    }
    const std::string omega_name = std::string(name_prefix) + "/aitken/omega";
    const std::string r_prefix = std::string(name_prefix) + "/aitken/r_prev";
    Result<void> r = writer.put_scalar(omega_name, omega_);
    if (!r.ok()) return r;
    return detail::put_field_per_block(writer, r_prefix, blocks, r_prev_, 1);
  }

  Result<void> import_restart_state(RestartReader& reader,
                                    std::string_view name_prefix,
                                    std::span<const RestartRowBlock> blocks,
                                    bool allow_reset_on_mismatch,
                                    bool& did_reset) override {
    did_reset = false;
    n_ = detail::total_rows(blocks);
    r_.resize(n_);
    r_prev_.resize(n_);
    const std::string omega_name = std::string(name_prefix) + "/aitken/omega";
    const std::string r_prefix = std::string(name_prefix) + "/aitken/r_prev";
    double omega_read = 0.0;
    const Result<fuse::ReadConversionReport> om =
        reader.get_scalar(omega_name, omega_read);
    if (!om.ok()) {
      // E-RESTART-MISSING-SCALAR specifically: export_restart_state()
      // wrote nothing at all for this name_prefix because have_prev_ was
      // false when the point was written (no sweep had yet completed
      // since the last reset()) -- the bit-identical same-rank state,
      // not a refusal.
      if (om.error().code == "E-RESTART-MISSING-SCALAR") {
        have_prev_ = false;
        omega_ = omega0_;
        return Result<void>::success();
      }
      return om.error();
    }
    const Result<void> rr =
        detail::get_field_per_block(reader, r_prefix, blocks, r_prev_, 1);
    if (!rr.ok()) {
      // The scalar WAS written (history existed), but this row's own
      // global id is not among the ones the file recorded -- owner
      // decision 4's layout-change case.
      if (!allow_reset_on_mismatch) return rr;
      did_reset = true;
      reset();
      return Result<void>::success();
    }
    omega_ = omega_read;
    have_prev_ = true;
    return Result<void>::success();
  }

 private:
  double              omega0_;
  double              omega_;
  std::size_t         n_ = 0;
  std::vector<double> r_;
  std::vector<double> r_prev_;
  bool                have_prev_ = false;
};

/// Steffensen, i.e. Aitken's Delta^2 applied independently to each component:
/// a diagonal secant. Per component,
///
///   x_out[i] = x_old[i] - r[i] * dx[i] / dr[i]
///
/// with dx = x_old - x_old_prev and dr = r - r_prev. This is exactly the
/// one-column least squares of the quasi-Newton members restricted to a single
/// component, so no Picard remainder term survives.
///
/// Near-zero denominator policy, per component: the quotient is formed only
/// when |dr[i]| is above the rounding level of the residuals it was formed
/// from, |dr[i]| > kDenomRel * (|r[i]| + |r_prev[i]|), and when the resulting
/// factor is finite. A component that fails either test degrades to
/// `initial_relaxation`; the others still take their secant step.
///
/// Storage: as AitkenAccel, and for the same reason -- the residual buffers
/// are members that ping-pong by swap(), and `have_prev_` rather than a
/// cleared buffer records whether there is history.
class SteffensenAccel final : public Accelerator {
 public:
  explicit SteffensenAccel(double initial_relaxation)
      : omega0_(initial_relaxation) {}

  void reset() override { have_prev_ = false; }

  [[nodiscard]] std::string name() const override { return "Steffensen"; }
  [[nodiscard]] std::size_t columns() const override { return 0; }

  Result<void> update(const double* x_old, const double* x_new, double* x_out,
                      std::size_t n, int iteration) override {
    const Result<void> args =
        detail::check_update_args(x_old, x_new, x_out, n, iteration);
    if (!args.ok()) return args;
    if (n == 0) return Result<void>::success();

    if (iteration == 0 || n != n_) reset();
    n_ = n;
    // Grow-only; see the same three lines in AitkenAccel::update().
    r_.resize(n);
    r_prev_.resize(n);
    x_old_prev_.resize(n);

    for (std::size_t i = 0; i < n; ++i) r_[i] = x_new[i] - x_old[i];

    if (!have_prev_) {
      detail::relax_into(omega0_, x_old, x_new, x_out, n);
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const double dr = r_[i] - r_prev_[i];
        const double dx = x_old[i] - x_old_prev_[i];
        const double scale = std::abs(r_[i]) + std::abs(r_prev_[i]);
        double step = omega0_ * r_[i];
        if (std::abs(dr) > kDenomRel * scale && dr != 0.0) {
          const double candidate = -r_[i] * dx / dr;
          if (std::isfinite(candidate)) step = candidate;
        }
        x_out[i] = x_old[i] + step;
      }
    }

    const Result<void> fin = detail::finish_or_fallback(
        "Steffensen", omega0_, x_old, x_new, x_out, n);
    if (!fin.ok()) {
      reset();
      return fin;
    }
    r_.swap(r_prev_);
    for (std::size_t i = 0; i < n; ++i) x_old_prev_[i] = x_old[i];
    have_prev_ = true;
    return Result<void>::success();
  }

  /// `r_prev_`/`x_old_prev_`/`have_prev_` are the
  /// state the NEXT update() call reads (`omega0_` is a fixed run
  /// option, `r_` is recomputed fresh every call). EXACT-KEY fix: both
  /// arrays are written per BLOCK (detail::put_field_per_block), one
  /// dataset per owning field name.
  [[nodiscard]] bool restart_state_implemented() const noexcept override {
    return true;
  }

  Result<void> export_restart_state(
      RestartWriter& writer, std::string_view name_prefix,
      std::span<const RestartRowBlock> blocks) const override {
    if (!have_prev_) return Result<void>::success();
    if (detail::total_rows(blocks) != n_) {
      return Failure{"E-RESTART-BAD-NUMBERING",
                     "SteffensenAccel::export_restart_state: " +
                         std::to_string(detail::total_rows(blocks)) +
                         " row global ids for an accelerator of length " +
                         std::to_string(n_)};
    }
    const std::string r_prefix = std::string(name_prefix) + "/steffensen/r_prev";
    const std::string x_prefix =
        std::string(name_prefix) + "/steffensen/x_old_prev";
    Result<void> r = detail::put_field_per_block(writer, r_prefix, blocks,
                                                  r_prev_, 1);
    if (!r.ok()) return r;
    return detail::put_field_per_block(writer, x_prefix, blocks, x_old_prev_,
                                       1);
  }

  Result<void> import_restart_state(RestartReader& reader,
                                    std::string_view name_prefix,
                                    std::span<const RestartRowBlock> blocks,
                                    bool allow_reset_on_mismatch,
                                    bool& did_reset) override {
    did_reset = false;
    n_ = detail::total_rows(blocks);
    r_.resize(n_);
    r_prev_.resize(n_);
    x_old_prev_.resize(n_);
    const std::string r_prefix = std::string(name_prefix) + "/steffensen/r_prev";
    const std::string x_prefix =
        std::string(name_prefix) + "/steffensen/x_old_prev";
    const Result<void> rr =
        detail::get_field_per_block(reader, r_prefix, blocks, r_prev_, 1);
    if (!rr.ok()) {
      if (rr.error().code == "E-RESTART-MISSING-FIELD") {
        have_prev_ = false;
        return Result<void>::success();
      }
      if (!allow_reset_on_mismatch) return rr;
      did_reset = true;
      reset();
      return Result<void>::success();
    }
    const Result<void> xr =
        detail::get_field_per_block(reader, x_prefix, blocks, x_old_prev_, 1);
    if (!xr.ok()) {
      if (!allow_reset_on_mismatch) return xr;
      did_reset = true;
      reset();
      return Result<void>::success();
    }
    have_prev_ = true;
    return Result<void>::success();
  }

 private:
  /// A change in the residual smaller than this multiple of the residual
  /// magnitude is indistinguishable from rounding noise in the residual.
  static constexpr double kDenomRel = 1e-12;

  double              omega0_;
  std::size_t         n_ = 0;
  std::vector<double> r_;
  std::vector<double> r_prev_;
  std::vector<double> x_old_prev_;
  bool                have_prev_ = false;
};

Result<void> validate(const AccelConfig& cfg) {
  if (!std::isfinite(cfg.initial_relaxation) || cfg.initial_relaxation == 0.0) {
    return Failure{"E-ACCEL-CONFIG",
                 "initial_relaxation must be finite and non-zero, got " +
                     detail::format_accel_double(cfg.initial_relaxation)};
  }
  if (!std::isfinite(cfg.omega) || cfg.omega == 0.0) {
    return Failure{"E-ACCEL-CONFIG", "omega must be finite and non-zero, got " +
                                       detail::format_accel_double(cfg.omega)};
  }
  if (cfg.reuse < 1 || cfg.reuse > Accelerator::kMaxReuse) {
    return Failure{"E-ACCEL-CONFIG",
                 "reuse must be in [1, " +
                     std::to_string(Accelerator::kMaxReuse) + "], got " +
                     std::to_string(cfg.reuse)};
  }
  if (!std::isfinite(cfg.limit) || cfg.limit < 0.0 || cfg.limit >= 1.0) {
    return Failure{"E-ACCEL-CONFIG", "limit must be finite and in [0, 1), got " +
                                       detail::format_accel_double(cfg.limit)};
  }
  return Result<void>::success();
}

}  // namespace

Result<std::unique_ptr<Accelerator>> make_accelerator(const AccelConfig& cfg) {
  const Result<void> v = validate(cfg);
  if (!v.ok()) return v.error();

  switch (cfg.kind) {
    case AccelKind::None:
      return std::unique_ptr<Accelerator>{std::make_unique<NoneAccel>()};
    case AccelKind::Constant:
      return std::unique_ptr<Accelerator>{
          std::make_unique<ConstantAccel>(cfg.omega)};
    case AccelKind::Aitken:
      return std::unique_ptr<Accelerator>{
          std::make_unique<AitkenAccel>(cfg.initial_relaxation)};
    case AccelKind::Steffensen:
      return std::unique_ptr<Accelerator>{
          std::make_unique<SteffensenAccel>(cfg.initial_relaxation)};
    case AccelKind::Secant:
    case AccelKind::IQN_ILS:
    case AccelKind::IQN_IMVJ:
      return detail::make_quasi_newton(cfg);
  }
  return Failure{"E-ACCEL-KIND", "unknown accelerator kind " +
                                   std::to_string(static_cast<int>(cfg.kind))};
}

}  // namespace fuse
