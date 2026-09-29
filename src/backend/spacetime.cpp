#include "fuse/backend/spacetime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fuse {
namespace {

using detail::Int128;

[[nodiscard]] std::string i128_to_string(Int128 v) {
  if (v == 0) return "0";
  const bool  neg = v < 0;
  Int128      m   = neg ? -v : v;
  std::string digits;
  while (m > 0) {
    digits.push_back(static_cast<char>('0' + static_cast<int>(m % 10)));
    m /= 10;
  }
  if (neg) digits.push_back('-');
  std::reverse(digits.begin(), digits.end());
  return digits;
}

/// How many whole `interval`s fit in `period`. Formed in __int128 so an
/// unrepresentable ratio is *reported* rather than thrown out of
/// `Rational::operator/` -- this function is on a `Result` path.
[[nodiscard]] Result<std::int64_t> steps_in(const Rational& period,
                                            const Rational& interval,
                                            const char*     what) {
  if (!(interval > Rational{0})) {
    return Failure{"E-ST-INTERVAL", std::string(what) + " interval must be " +
                                      "positive, got " + interval.to_string()};
  }
  const Int128 num = static_cast<Int128>(period.num()) * interval.den();
  const Int128 den = static_cast<Int128>(period.den()) * interval.num();
  // den > 0: Rational keeps denominators positive and interval.num() > 0.
  if (num % den != 0) {
    return Failure{"E-ST-INTERVAL",
                 "period " + period.to_string() + " is not a whole multiple " +
                     "of the " + what + " interval " + interval.to_string()};
  }
  const Int128 n = num / den;
  if (n <= 0 || n > kMaxSlabsPerPeriod) {
    return Failure{"E-ST-INTERVAL",
                 std::string(what) + " interval " + interval.to_string() +
                     " gives " + i128_to_string(n) + " steps per period " +
                     period.to_string() + ", outside [1, " +
                     std::to_string(kMaxSlabsPerPeriod) + "]"};
  }
  return static_cast<std::int64_t>(n);
}

/// A COO triplet before canonicalisation.
struct Triplet {
  std::int32_t row;
  std::int32_t col;
  double       val;
};

/// Dense LU with partial pivoting for the tiny (<= 5x5) reconstruction
/// systems. LAPACK would be a core dependency for a 5x5 solve; the same
/// reasoning `num::fit_polynomial` gives.
struct Lu {
  std::vector<double>      a;    ///< m*m, factors in place
  std::vector<std::size_t> piv;  ///< row index actually used at each step
  std::size_t              m = 0;
};

[[nodiscard]] bool lu_factor(Lu& lu) {
  const std::size_t m = lu.m;
  lu.piv.resize(m);
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t best_row = k;
    double      best     = std::fabs(lu.a[k * m + k]);
    for (std::size_t r = k + 1; r < m; ++r) {
      const double v = std::fabs(lu.a[r * m + k]);
      if (v > best) {
        best     = v;
        best_row = r;
      }
    }
    if (!(best > 0.0)) return false;
    lu.piv[k] = best_row;
    if (best_row != k) {
      for (std::size_t c = 0; c < m; ++c) {
        std::swap(lu.a[k * m + c], lu.a[best_row * m + c]);
      }
    }
    for (std::size_t r = k + 1; r < m; ++r) {
      const double fac = lu.a[r * m + k] / lu.a[k * m + k];
      lu.a[r * m + k]  = fac;
      if (fac == 0.0) continue;
      for (std::size_t c = k + 1; c < m; ++c) {
        lu.a[r * m + c] -= fac * lu.a[k * m + c];
      }
    }
  }
  return true;
}

void lu_solve(const Lu& lu, std::vector<double>& b) {
  const std::size_t m = lu.m;
  // ALL the row interchanges are applied to b first, and only then is the
  // forward substitution run. Interleaving them (swap at step k, then
  // eliminate with column k) is wrong here and is a genuinely subtle trap:
  // `lu_factor` swaps WHOLE rows, so a pivot at step k > 0 also permutes the
  // multipliers already stored in columns 0..k-1, and an interleaved solve
  // would then use post-swap multipliers against pre-swap eliminations. It
  // happens to give the right answer for m <= 2 (there is no later swap to
  // move an earlier column), which is why degrees 0 and 1 looked correct
  // while 2, 3 and 4 lost both row-sum unity and their order.
  for (std::size_t k = 0; k < m; ++k) {
    if (lu.piv[k] != k) std::swap(b[k], b[lu.piv[k]]);
  }
  for (std::size_t k = 0; k < m; ++k) {
    for (std::size_t r = k + 1; r < m; ++r) b[r] -= lu.a[r * m + k] * b[k];
  }
  for (std::size_t i = m; i-- > 0;) {
    double s = b[i];
    for (std::size_t j = i + 1; j < m; ++j) s -= lu.a[i * m + j] * b[j];
    b[i] = s / lu.a[i * m + i];
  }
}

/// integral of s^r over [lo, hi], for r = 0 .. deg. Used both for the
/// stencil's cell-average constraints and for a common-refinement interval's moment vector.
void moments(double lo, double hi, int deg, std::vector<double>& out) {
  double plo = lo;
  double phi = hi;
  for (int r = 0; r <= deg; ++r) {
    out[static_cast<std::size_t>(r)] =
        (phi - plo) / static_cast<double>(r + 1);
    plo *= lo;
    phi *= hi;
  }
}

/// First step of the `p+1`-wide reconstruction stencil for source step `j`.
/// Symmetric for even `p`, one step towards the past for odd `p`, clamped
/// into the window. See the header comment.
[[nodiscard]] std::int64_t stencil_start(std::int64_t j, int p,
                                         std::int64_t n_src) {
  std::int64_t start = j - (p + 1) / 2;
  const std::int64_t last = n_src - 1 - p;  // >= 0: p <= n_src - 1
  if (start < 0) start = 0;
  if (start > last) start = last;
  return start;
}

[[nodiscard]] Result<TemporalProjection> build_impl(
    const SlabDecomposition& slabs, TemporalBasis basis, int degree,
    Rational src_interval, Rational dst_interval) {
  // ---- basis and degree ---------------------------------------------------
  switch (basis) {
    case TemporalBasis::None:
      return Failure{"E-ST-BASIS",
                   "build_temporal_projection: TemporalBasis::None is not a "
                   "temporal representation; a space-time projection needs a "
                   "basis"};
    case TemporalBasis::Spline:
      return Failure{"E-ST-BASIS",
                   "build_temporal_projection: a conservative Spline "
                   "projection is a globally coupled reconstruction and is "
                   "not implemented; use DG(3) for a degree-3 local "
                   "reconstruction rather than getting one silently under the "
                   "name Spline"};
    case TemporalBasis::Constant:
      if (degree != 0) {
        return Failure{"E-ST-DEGREE",
                     "build_temporal_projection: Constant basis requires "
                     "degree 0, got " +
                         std::to_string(degree)};
      }
      break;
    case TemporalBasis::Linear:
      if (degree != 1) {
        return Failure{"E-ST-DEGREE",
                     "build_temporal_projection: Linear basis requires "
                     "degree 1, got " +
                         std::to_string(degree)};
      }
      break;
    case TemporalBasis::DG:
      if (degree < 0 || degree > kMaxTemporalDegree) {
        return Failure{"E-ST-DEGREE",
                     "build_temporal_projection: DG degree must be in [0, " +
                         std::to_string(kMaxTemporalDegree) + "], got " +
                         std::to_string(degree)};
      }
      break;
  }

  // ---- step counts --------------------------------------------------------
  const Rational period = slabs.period;
  if (!(period > Rational{0})) {
    return Failure{"E-ST-INTERVAL",
                 "build_temporal_projection: period must be positive, got " +
                     period.to_string()};
  }
  Result<std::int64_t> n_src_r = steps_in(period, src_interval, "source");
  if (!n_src_r.ok()) return n_src_r.error();
  Result<std::int64_t> n_tgt_r = steps_in(period, dst_interval, "target");
  if (!n_tgt_r.ok()) return n_tgt_r.error();
  const std::int64_t n_src = n_src_r.value();
  const std::int64_t n_tgt = n_tgt_r.value();

  // ---- the decomposition really is the intersection of THESE partitions ---
  //
  // Checked rather than assumed: a SlabDecomposition built from different
  // intervals would still produce unit row sums (they are normalised by
  // dst_interval), so the failure would be silent and only visible as a wrong
  // answer.
  if (slabs.slabs.empty()) {
    return Failure{"E-ST-SLABS",
                 "build_temporal_projection: empty common-refinement-interval decomposition"};
  }
  {
    Rational cursor{0};
    for (std::size_t k = 0; k < slabs.slabs.size(); ++k) {
      const Slab& s   = slabs.slabs[k];
      const auto  ctx = " (common-refinement interval " + std::to_string(k) + " [" +
                       s.begin.to_string() + ", " + s.end.to_string() + "])";
      if (s.begin != cursor) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: common-refinement intervals do not tile the "
                     "period contiguously; expected a common-refinement interval starting at " +
                         cursor.to_string() + ctx};
      }
      if (!(s.end > s.begin)) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: non-positive common-refinement-interval length" +
                         ctx};
      }
      if (s.src_step < 0 || s.src_step >= n_src) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: src_step " +
                         std::to_string(s.src_step) + " outside [0, " +
                         std::to_string(n_src) + ")" + ctx};
      }
      if (s.dst_step < 0 || s.dst_step >= n_tgt) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: dst_step " +
                         std::to_string(s.dst_step) + " outside [0, " +
                         std::to_string(n_tgt) + ")" + ctx};
      }
      // The common-refinement interval must lie inside the step it claims, on BOTH sides. Exact
      // rational comparison; this is what catches a decomposition built from
      // other intervals.
      const Rational qb_src = s.begin / src_interval;
      const Rational qe_src = s.end / src_interval;
      if (qb_src < Rational{s.src_step} ||
          qe_src > Rational{s.src_step + 1}) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: common-refinement interval is not inside source "
                     "step " +
                         std::to_string(s.src_step) + " of interval " +
                         src_interval.to_string() + ctx};
      }
      const Rational qb_dst = s.begin / dst_interval;
      const Rational qe_dst = s.end / dst_interval;
      if (qb_dst < Rational{s.dst_step} ||
          qe_dst > Rational{s.dst_step + 1}) {
        return Failure{"E-ST-SLABS",
                     "build_temporal_projection: common-refinement interval is not inside target "
                     "step " +
                         std::to_string(s.dst_step) + " of interval " +
                         dst_interval.to_string() + ctx};
      }
      cursor = s.end;
    }
    if (cursor != period) {
      return Failure{"E-ST-SLABS",
                   "build_temporal_projection: common-refinement intervals end at " +
                       cursor.to_string() + ", not at the period " +
                       period.to_string()};
    }
  }

  // ---- assemble -----------------------------------------------------------
  const int p = static_cast<int>(
      std::min<std::int64_t>(degree, n_src - 1));  // clamped; see header
  const std::size_t m = static_cast<std::size_t>(p) + 1;

  std::vector<Triplet> trips;
  trips.reserve(slabs.slabs.size() * m);

  if (p == 0) {
    // Exactly the supermesh overlap fraction, formed in exact rational
    // arithmetic and narrowed once. Going through the scaled quadrature below
    // would give the same number to within a rounding or two, but this is the
    // case the row-sum tests measure at 1e-14 and there is no reason to spend
    // the accuracy.
    for (const Slab& s : slabs.slabs) {
      const double w = ((s.end - s.begin) / dst_interval).to_double();
      trips.push_back(Triplet{static_cast<std::int32_t>(s.dst_step),
                              static_cast<std::int32_t>(s.src_step), w});
    }
  } else {
    // A^T for the stencil depends only on the offset `start - j`, which takes
    // at most p+1 values, so factor each once instead of once per source step.
    const std::size_t n_off = static_cast<std::size_t>(p) + 1;  // offsets -p..0
    std::vector<Lu>   lus(n_off);
    std::vector<char> have(n_off, 0);
    std::vector<double> col(m);

    const double w_scale = (src_interval / dst_interval).to_double();
    std::vector<double> mom(m);
    std::vector<double> rhs(m);

    for (const Slab& s : slabs.slabs) {
      const std::int64_t j      = s.src_step;
      const std::int64_t start  = stencil_start(j, p, n_src);
      const std::int64_t offset = start - j;  // in [-p, 0]
      const std::size_t  oi     = static_cast<std::size_t>(-offset);
      if (!have[oi]) {
        Lu& lu = lus[oi];
        lu.m   = m;
        lu.a.assign(m * m, 0.0);
        // Row r, column k of A^T is A_{k r} = integral of s^r over the k-th
        // stencil cell, in units of the source interval, measured from the
        // centre of step j.
        for (std::size_t k = 0; k < m; ++k) {
          const double d = static_cast<double>(offset + static_cast<std::int64_t>(k));
          moments(d - 0.5, d + 0.5, p, col);
          for (std::size_t r = 0; r < m; ++r) lu.a[r * m + k] = col[r];
        }
        if (!lu_factor(lu)) {
          return Failure{"E-ST-SINGULAR",
                       "build_temporal_projection: singular degree-" +
                           std::to_string(p) +
                           " reconstruction system at source step " +
                           std::to_string(j)};
        }
        have[oi] = 1;
      }

      // Common-refinement-interval endpoints in units of the source interval, measured from the
      // centre of step j. The subtraction of the integer step index happens in
      // EXACT rational arithmetic, so `u` stays in [-1/2, 1/2] and carries no
      // cancellation even at step index 100000.
      const double u_lo =
          (s.begin / src_interval - Rational{j}).to_double() - 0.5;
      const double u_hi =
          (s.end / src_interval - Rational{j}).to_double() - 0.5;

      moments(u_lo, u_hi, p, mom);
      rhs = mom;
      lu_solve(lus[oi], rhs);  // rhs <- A^-T m : the stencil weights

      for (std::size_t k = 0; k < m; ++k) {
        trips.push_back(
            Triplet{static_cast<std::int32_t>(s.dst_step),
                    static_cast<std::int32_t>(start + static_cast<std::int64_t>(k)),
                    w_scale * rhs[k]});
      }
    }
  }

  // ---- canonicalise: stable sort by (row, col), merge, drop exact zeros ---
  //
  // stable_sort, not sort: equal (row, col) keys keep their insertion order, so
  // the merge below adds them in a fixed order and two builds from equal
  // inputs are bitwise equal.
  std::stable_sort(trips.begin(), trips.end(),
                   [](const Triplet& a, const Triplet& b) {
                     if (a.row != b.row) return a.row < b.row;
                     return a.col < b.col;
                   });

  TemporalProjection out;
  out.n_src_steps      = static_cast<std::size_t>(n_src);
  out.n_tgt_steps      = static_cast<std::size_t>(n_tgt);
  out.basis            = basis;
  out.degree           = degree;
  out.effective_degree = p;
  out.src_interval     = src_interval;
  out.dst_interval     = dst_interval;
  out.row.reserve(trips.size());
  out.col.reserve(trips.size());
  out.val.reserve(trips.size());

  for (std::size_t k = 0; k < trips.size();) {
    std::size_t e   = k;
    double      sum = 0.0;
    while (e < trips.size() && trips[e].row == trips[k].row &&
           trips[e].col == trips[k].col) {
      sum += trips[e].val;
      ++e;
    }
    if (sum != 0.0) {
      out.row.push_back(trips[k].row);
      out.col.push_back(trips[k].col);
      out.val.push_back(sum);
    }
    k = e;
  }
  return out;
}

}  // namespace

double TemporalProjection::coefficient(std::size_t tgt_step,
                                       std::size_t src_step) const noexcept {
  if (tgt_step > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      src_step > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return 0.0;
  }
  const auto r = static_cast<std::int32_t>(tgt_step);
  const auto c = static_cast<std::int32_t>(src_step);
  std::size_t lo = 0;
  std::size_t hi = val.size();
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (row[mid] < r || (row[mid] == r && col[mid] < c)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < val.size() && row[lo] == r && col[lo] == c) return val[lo];
  return 0.0;
}

Result<TemporalProjection> build_temporal_projection(
    const SlabDecomposition& slabs, TemporalBasis basis, int degree,
    Rational src_interval, Rational dst_interval) {
  // `Rational`'s operators throw std::logic_error when an exact result does
  // not fit int64 after reduction. Every such operation above is on legal,
  // in-range data for any decomposition `decompose()` actually produces, but
  // this function returns a `Result` and must therefore be no-throw by
  // contract, so the residual possibility is converted rather than escaping.
  // std::invalid_argument and std::out_of_range both derive from
  // std::logic_error and are caught here too.
  try {
    return build_impl(slabs, basis, degree, src_interval, dst_interval);
  } catch (const std::logic_error& e) {
    return Failure{"E-ST-OVERFLOW",
                 std::string("build_temporal_projection: exact time "
                             "arithmetic failed: ") +
                     e.what()};
  }
}

Result<void> apply_spacetime(const SpatialWeightsView& spatial,
                             const TemporalProjection& temporal,
                             const double* src, double* dst,
                             std::size_t n_src_space, int components,
                             std::size_t src_step, std::size_t tgt_step) {
  if (components <= 0) {
    return Failure{"E-ST-SHAPE", "apply_spacetime: components must be positive, "
                               "got " +
                                   std::to_string(components)};
  }
  if (n_src_space != spatial.n_cols) {
    return Failure{"E-ST-SHAPE",
                 "apply_spacetime: n_src_space " +
                     std::to_string(n_src_space) +
                     " disagrees with the spatial operator's column count " +
                     std::to_string(spatial.n_cols)};
  }
  if (src_step >= temporal.n_src_steps) {
    return Failure{"E-ST-STEP", "apply_spacetime: src_step " +
                                  std::to_string(src_step) + " >= " +
                                  std::to_string(temporal.n_src_steps)};
  }
  if (tgt_step >= temporal.n_tgt_steps) {
    return Failure{"E-ST-STEP", "apply_spacetime: tgt_step " +
                                  std::to_string(tgt_step) + " >= " +
                                  std::to_string(temporal.n_tgt_steps)};
  }
  if (spatial.nnz > 0 && (spatial.row == nullptr || spatial.col == nullptr ||
                          spatial.val == nullptr)) {
    return Failure{"E-ST-NULL",
                 "apply_spacetime: null spatial triplet array with nnz > 0"};
  }
  if (n_src_space > 0 && src == nullptr) {
    return Failure{"E-ST-NULL", "apply_spacetime: null src buffer"};
  }
  if (spatial.n_rows > 0 && dst == nullptr) {
    return Failure{"E-ST-NULL", "apply_spacetime: null dst buffer"};
  }

  const double w = temporal.coefficient(tgt_step, src_step);

  // REVIEW.md R-33: `w == 0.0` used to return here, before the triplet-index
  // validation loop below ran at all -- so a malformed `spatial` array (an
  // out-of-range row/column index) was only ever caught on a call whose
  // temporal coefficient happened to be nonzero. The validation now runs on
  // EVERY call regardless of `w`; only the accumulation itself -- which a
  // zero coefficient would zero out anyway -- is skipped per triplet.
  const auto nc = static_cast<std::size_t>(components);
  for (std::size_t k = 0; k < spatial.nnz; ++k) {
    const std::int32_t r = spatial.row[k];
    const std::int32_t c = spatial.col[k];
    if (r < 0 || static_cast<std::size_t>(r) >= spatial.n_rows) {
      return Failure{"E-ST-SHAPE", "apply_spacetime: spatial row index " +
                                     std::to_string(r) + " outside [0, " +
                                     std::to_string(spatial.n_rows) + ")"};
    }
    if (c < 0 || static_cast<std::size_t>(c) >= spatial.n_cols) {
      return Failure{"E-ST-SHAPE", "apply_spacetime: spatial column index " +
                                     std::to_string(c) + " outside [0, " +
                                     std::to_string(spatial.n_cols) + ")"};
    }
    if (w == 0.0) continue;  // structurally absent: validated, not applied
    const double        a  = w * spatial.val[k];
    const double* const in = src + static_cast<std::size_t>(c) * nc;
    double* const       ou = dst + static_cast<std::size_t>(r) * nc;
    for (std::size_t e = 0; e < nc; ++e) ou[e] += a * in[e];
  }
  return Result<void>::success();
}

}  // namespace fuse
