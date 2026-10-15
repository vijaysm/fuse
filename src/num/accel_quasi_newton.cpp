#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/linalg/least_squares.hpp"
#include "fuse/num/accelerator.hpp"

/// Spec Section 5.1 `Accelerate` and Section 5.3: the least-squares members of the
/// accelerator family -- Secant, IQN-ILS and IQN-IMVJ. They differ only in how
/// many columns they keep and whether anything survives a window boundary, so
/// there is one class.
///
/// ## The method
///
/// With r_k = x_new_k - x_old_k the fixed-point residual and R(x) = G(x) - x
/// the function whose root is wanted, a quasi-Newton step is
///
///     x_{k+1} = x_k - Jhat^{-1} r_k
///
/// where Jhat^{-1} is built from the secant pairs collected in this window:
/// V holds the residual differences Delta r_i and Wx the iterate differences
/// Delta x_i, and the secant condition is Jhat^{-1} Delta r_i = Delta x_i.
/// Writing alpha for the least-squares coefficients of -r_k in the columns of
/// V,
///
///     alpha = argmin || V a + r_k ||,      rho = r_k + V alpha
///     x_{k+1} = x_k + Wx alpha - Jprev^{-1} rho
///
/// rho is the part of the residual that the collected columns cannot explain.
/// IQN-ILS is exactly the case Jprev^{-1} = -I, which adds rho back as a full
/// Picard step; substituting Wx = Wn - V (Wn being the differences of x_new)
/// collapses the expression to the familiar
///
///     x_{k+1} = x_new_k + Wn alpha
///
/// IQN-IMVJ replaces -I with the multi-vector inverse Jacobian accumulated
/// over previous windows,
///
///     Jhat_m^{-1} = Jhat_{m-1}^{-1} + (Wx_m - Jhat_{m-1}^{-1} V_m) V_m^+
///
/// which is stored as a chain of low-rank blocks (Z_m, V_m) rather than as a
/// dense n x n matrix; see JacobianBlock.
///
/// ## Why a column-pivoted Householder QR
///
/// The columns of V are differences of nearly equal residual vectors, so V is
/// ill-conditioned by construction and becomes numerically rank deficient as
/// soon as the iteration starts to converge -- that is the normal case here,
/// not an edge case. Two consequences drive the choice of factorisation:
///
///   * Normal equations (V^T V a = -V^T r) square the condition number, so a
///     V with cond 1e8 -- routine after a few iterations -- leaves no correct
///     digits in a double solve. Householder QR's accuracy depends on cond(V),
///     not cond(V)^2, and the factorisation is unconditionally backward
///     stable. QnLeastSquares.QrBeatsNormalEquationsOnNearDependentColumns
///     measures the difference on a deliberately near-dependent pair.
///   * Plain Householder QR is stable but not rank revealing: it happily
///     returns a huge alpha from a vanishing R diagonal. Column pivoting makes
///     |R_00| >= |R_11| >= ... so a numerical rank can be read off, and the
///     solve is truncated there with the dropped coefficients set to zero.
///     The truncation threshold is max(m, k) * DBL_EPSILON relative to the
///     leading pivot -- a rounding-level test, deliberately far below the
///     Task 7.4 filter's `limit`, so it only removes columns that carry no
///     information at all.
namespace fuse {
namespace detail {

namespace {

/// Column-major (i, j) -> flat index into an m x k matrix.
inline std::size_t at(std::size_t m, std::size_t i, std::size_t j) noexcept {
  return j * m + i;
}

// The factorisation itself now lives in fuse::linalg (one dense
// least-squares solve behind one declaration, with interchangeable backends).
// It moved rather than being wrapped: the accelerator needs the
// scratch-supplied variant to stay allocation-free while running, and two
// copies of a Householder QR -- one for the hot loop and one behind the facade
// -- would be two places for the rank truncation to drift.

/// One past window's contribution to the IMVJ inverse Jacobian:
///     Jhat_m^{-1} v = Jhat_{m-1}^{-1} v + Z_m (V_m^+ v)
/// with Z_m = Wx_m - Jhat_{m-1}^{-1} V_m precomputed when the window closed.
/// Storing the blocks instead of a dense n x n matrix is what keeps IMVJ's
/// memory O(n * reuse) per window rather than O(n^2).
struct JacobianBlock {
  std::vector<double> z;     ///< n x cols, column major
  std::vector<double> v;     ///< n x cols, column major
  std::size_t         cols = 0;
};

/// Re-orthogonalisation threshold of the Daniel-Gragg-Kaufman-Stewart test: a
/// Gram-Schmidt pass that shrinks a vector by more than this factor has lost
/// digits to cancellation and is repeated. 1/sqrt(2) is the classical choice;
/// preCICE's QRFactorization uses the nearby 0.7 (its `_theta` is 1/0.7). The
/// exact value is not a semantic choice -- it only decides how often a second
/// pass runs, and two passes are enough at either setting.
constexpr double kDgksTheta = 0.7071067811865475;

/// The working buffers of one filter_qn_columns() call, owned by the
/// accelerator for the same reason LsScratch is: the filter runs once per
/// iteration, inside the sweep loop, and must not reach the allocator there.
/// `basis` and `work` belong to qn_remainder_norms(), which the filter calls
/// once per removal sweep. As with LsScratch, every element is written before
/// it is read on every call.
struct FilterScratch {
  std::vector<std::size_t> order;
  std::vector<double>      rho0;
  std::vector<double>      rho1;
  std::vector<char>        keep;
  std::vector<double>      basis;  ///< n x qc orthonormal Q, column major
  std::vector<double>      work;   ///< the column being orthogonalised
};

/// Diagonal of the (unpivoted) triangular factor of D V, taken with the columns
/// in the order given by `order` and with `d` the per-row weights.
///
/// `rho0[j]` is the 2-norm of column `order[j]`; `rho1[j]` is the 2-norm of the
/// part of that column orthogonal to every column processed before it, i.e.
/// |R_jj| of the QR factorisation of the reordered D V. Modified Gram-Schmidt
/// with DGKS re-orthogonalisation: MGS alone loses orthogonality like
/// cond(V) * eps, which on the deliberately near-dependent V of a quasi-Newton
/// window is exactly the regime the filter has to judge, so the second pass is
/// not optional here.
///
/// D is the same row scaling the least-squares solve applies, and it has to be
/// here for the same reason it is there: without it the notion of "nearly
/// dependent" is read off a matrix whose rows are in whatever physical units
/// the fields happen to use, so which columns survive would depend on the units
/// -- see PerFieldScalingMakesTheFitIndependentOfFieldUnits, which fails
/// outright if the filter judges the unscaled V.
///
/// A column whose remainder is exactly zero (or non-finite) does not extend the
/// basis; there is no direction to normalise, and any filter with a positive
/// limit drops such a column anyway.
void qn_remainder_norms(const std::vector<double>& v, std::size_t n,
                        const std::vector<double>& d,
                        const std::vector<std::size_t>& order,
                        std::vector<double>& rho0, std::vector<double>& rho1,
                        FilterScratch& scratch) {
  const std::size_t k = order.size();
  rho0.assign(k, 0.0);
  rho1.assign(k, 0.0);
  if (n == 0 || k == 0) return;

  // n x qc, column major: the orthonormal basis so far. clear() keeps the
  // capacity a previous call reserved, so the resize() in the loop below grows
  // into memory that is already there.
  std::vector<double>& q = scratch.basis;
  q.clear();
  q.reserve(n * k);
  std::size_t          qc = 0;
  std::vector<double>& u  = scratch.work;
  u.resize(n);

  for (std::size_t j = 0; j < k; ++j) {
    const std::size_t c = order[j];
    double            s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      u[i] = d[i] * v[at(n, i, c)];
      s += u[i] * u[i];
    }
    double rho = std::sqrt(s);
    rho0[j]    = rho;

    // At most four passes, as in preCICE: two is the textbook "twice is
    // enough", the rest are a bound on the loop rather than an expectation.
    for (int pass = 0; pass < 4 && rho > 0.0 && std::isfinite(rho); ++pass) {
      const double before = rho;
      for (std::size_t t = 0; t < qc; ++t) {
        double dot = 0.0;
        for (std::size_t i = 0; i < n; ++i) dot += q[at(n, i, t)] * u[i];
        for (std::size_t i = 0; i < n; ++i) u[i] -= dot * q[at(n, i, t)];
      }
      double s2 = 0.0;
      for (std::size_t i = 0; i < n; ++i) s2 += u[i] * u[i];
      rho = std::sqrt(s2);
      if (qc == 0 || rho > kDgksTheta * before) break;
    }
    rho1[j] = std::isfinite(rho) ? rho : 0.0;

    // A subnormal remainder would make the reciprocal overflow and poison the
    // basis for every later column, so it is treated as no direction at all.
    const double inv = rho1[j] > 0.0 ? 1.0 / rho1[j] : 0.0;
    if (rho1[j] > 0.0 && std::isfinite(inv)) {
      q.resize(n * (qc + 1));
      for (std::size_t i = 0; i < n; ++i) q[at(n, i, qc)] = u[i] * inv;
      ++qc;
    }
  }
}

/// SEAM FOR TASK 7.4 -- QR column filtering. Implemented.
///
/// Called once per iteration, after the new column pair has been appended and
/// before the least-squares solve, with the authority to drop any column from
/// both `v` and `w` (they must stay index aligned) and to lower `cols`.
/// `columns()` reports whatever survives, so a filter is visible to callers
/// and to tests.
///
/// ## What the filters are
///
/// These are the established preCICE IQN-ILS filters. Sources, checked against
/// the code and not paraphrased from memory:
///
///   * preCICE `develop`, src/acceleration/impl/QRFactorization.cpp,
///     `applyFilter` / `resetFilter` / `insertColumn`, and the filter constants
///     in src/acceleration/Acceleration.hpp;
///   * Davis, Schulte & Uekermann, "Enhancing Quasi-Newton Acceleration for
///     Fluid-Structure Interaction", Math. Comput. Appl. 27(3):40 (2022),
///     doi:10.3390/mca27030040, sec. 2.3.1 and Algorithm 3, which is where QR3
///     is defined;
///   * Haelterman et al., "Improving the performance of the partitioned QN-ILS
///     procedure for fluid-structure interaction problems: Filtering",
///     Computers & Structures 171 (2016) 9-17, from which Davis et al. take
///     QR1 and QR2 as the "old" and "new" QR filters.
///
/// All of them read the same quantity -- |R_jj|, the length of the part of
/// column j that the columns before it do not already explain -- and differ in
/// what they compare it against:
///
///   QR1   drop column j if |R_jj| < limit * ||R||_F.
///         One threshold shared by every column, set by the size of the whole
///         matrix (and ||R||_F == ||V||_F, Q being orthonormal). This is
///         preCICE's QR1FILTER. Its QR1FILTER_ABS sibling uses the bare
///         `limit` with no normalisation at all, which makes the outcome
///         depend on the physical units of the residual; it has no FilterKind
///         here.
///   QR2   drop column j if |R_jj| < limit * ||v_j||.
///         A per-column threshold: the test is on the *fraction* of the column
///         that is new, so it is invariant under rescaling any single column
///         and under rescaling V as a whole. preCICE's QR2FILTER, and the
///         default here.
///   QR3   the same column set as QR2, by construction -- see below.
///
/// QR1 and QR2 are genuinely different, and the difference is not a matter of
/// tuning: QR2 drops a large column that is nearly parallel to an existing one,
/// which QR1 keeps; QR1 drops a small but perfectly independent column, which
/// QR2 keeps. Because QR1's threshold grows with the whole matrix it wants a
/// much smaller limit than QR2 -- preCICE suggests 1e-5..1e-6 for QR1 against
/// 1e-2..1e-3 for QR2.
///
/// ## Why QR3 coincides with QR2 here
///
/// QR3 (Davis et al. 2022, Algorithm 3; preCICE QR3FILTER, and preCICE's
/// default since v3.2.0) is a *cost* optimisation, not a different criterion.
/// preCICE maintains the QR factorisation incrementally across iterations with
/// Givens rotations, and the en-bloc rebuild QR2 needs is expensive. QR3 runs
/// QR2's own test, |R_jj| < limit * ||v_j||, over the columns of that already
/// updated factorisation; if no column trips it, nothing is rebuilt and nothing
/// is dropped, and if any column trips it, preCICE falls through to the full
/// QR2 rebuild, which then decides what actually goes.
///
/// This implementation holds no factorisation between calls -- it rebuilds from
/// the stored V every iteration, because the row scaling D is recomputed from
/// the current residual every iteration anyway. The detector therefore reads
/// exactly the R that QR2 would build, so "trips the detector" and "QR2 drops
/// something" are the same event and the retained column set is identical.
/// FilterKind::QR3 is accordingly implemented as QR2 rather than faked as a
/// no-op or invented as a third criterion; QnColumnFilter.Qr3MatchesQr2 pins
/// the equality.
///
/// ## Order, and what can be dropped
///
/// Columns are examined newest first, matching preCICE, where `_matrixV` is
/// built with appendFront so column 0 is the most recent secant pair and
/// `resetFilter` re-inserts from there ("latest information is never filtered
/// out"). The newest column therefore never has to compete against a younger
/// one, and for QR2 -- whose threshold for the first column examined is
/// limit * ||v|| < ||v|| for any limit in [0, 1) -- it can never be dropped.
/// QR2 thus cannot empty a non-empty history. QR1 can, because its threshold is
/// not tied to the column being judged; `cols == 0` is a legal outcome and
/// simply means the step degrades to relaxation, exactly as at a window
/// boundary. (preCICE additionally guards QR1 with `if (_cols > 1)`; that guard
/// is redundant for limit < 1, since a lone column is its own ||R||_F.)
///
/// Removals are applied and the factorisation redone until a sweep drops
/// nothing, because deleting a column raises |R_jj| of every later one, and
/// QR1's ||R||_F shrinks. preCICE's `while (linearDependence)` loop does the
/// same. `cols` is at most kMaxReuse, so this is bounded work.
///
/// `limit == 0` disables the drop test, matching preCICE's
/// `bool applyFilter = (singularityLimit > 0.0)`, and keeps the [0, 1)
/// validation in make_accelerator meaningful at its lower end.
///
/// This is *not* what keeps the solve well posed: qn_least_squares() truncates
/// at the numerical rank unconditionally, whatever this function does. A filter
/// can only make the retained set smaller than what that truncation accepts.
///
/// `d` holds the per-row weights the least-squares solve will apply, so the
/// filter judges exactly the matrix that will be factorised and not a differently
/// scaled one.
void filter_qn_columns(FilterKind filter, double limit, std::size_t n,
                       const std::vector<double>& d, std::vector<double>& v,
                       std::vector<double>& w, std::size_t& cols,
                       FilterScratch& scratch) {
  if (filter == FilterKind::None) return;
  if (n == 0 || cols == 0) return;
  if (!(limit > 0.0) || !std::isfinite(limit)) return;
  // QR3's criterion is QR2's; see above for why they cannot part company in an
  // implementation that rebuilds the factorisation every call.
  const bool relative = (filter != FilterKind::QR1);

  std::vector<std::size_t>& order = scratch.order;
  std::vector<double>&      rho0  = scratch.rho0;
  std::vector<double>&      rho1  = scratch.rho1;
  std::vector<char>&        keep  = scratch.keep;

  for (std::size_t sweep = 0, guard = cols; sweep < guard && cols > 0;
       ++sweep) {
    order.resize(cols);
    for (std::size_t j = 0; j < cols; ++j) order[j] = cols - 1 - j;
    qn_remainder_norms(v, n, d, order, rho0, rho1, scratch);

    // QR1's threshold is the Frobenius norm of the current scaled V,
    // recomputed after every removal because the matrix it describes has
    // changed. It is the sum of the rho0 already in hand.
    double absolute = 0.0;
    if (!relative) {
      double f2 = 0.0;
      for (const double e : rho0) f2 += e * e;
      absolute = limit * std::sqrt(f2);
      if (!std::isfinite(absolute)) return;
    }

    keep.assign(cols, 1);
    std::size_t survivors = 0;
    for (std::size_t j = 0; j < cols; ++j) {
      const double thr = relative ? limit * rho0[j] : absolute;
      // A column of length 0 cannot reach any positive threshold and carries
      // no direction; it is dropped whatever the comparison says.
      const bool drop = !(rho0[j] > 0.0) || rho1[j] < thr;
      if (drop) {
        keep[order[j]] = 0;
      } else {
        ++survivors;
      }
    }
    if (survivors == cols) return;

    // Compact in place, oldest first, keeping v and w index aligned.
    std::size_t out = 0;
    for (std::size_t j = 0; j < cols; ++j) {
      if (!keep[j]) continue;
      if (out != j) {
        for (std::size_t i = 0; i < n; ++i) {
          v[at(n, i, out)] = v[at(n, i, j)];
          w[at(n, i, out)] = w[at(n, i, j)];
        }
      }
      ++out;
    }
    v.resize(n * out);
    w.resize(n * out);
    cols = out;
  }
}

/// Storage: every buffer this class needs is a member, so a steady-state
/// update() performs no heap allocation (a window boundary that folds a new
/// block in is nearly but not quite the same claim -- see
/// fold_window_into_jacobian()). An accelerated
/// implicit coupling group sits inside the sweep loop, which spec Section 2 layer 3 requires
/// to be allocation free, and the scratch of a least-squares solve over
/// k columns of length n is far too big to leave to the allocator once per
/// sweep. Nothing about the arithmetic changed with the buffers: every scratch
/// element is written before it is read on every call.
class QuasiNewtonAccel final : public Accelerator {
 public:
  QuasiNewtonAccel(AccelKind kind, double initial_relaxation,
                   std::size_t max_columns, FilterKind filter, double limit,
                   bool scale_fields, bool retain_across_reset)
      : kind_(kind),
        omega0_(initial_relaxation),
        max_columns_(max_columns),
        filter_(filter),
        limit_(limit),
        scale_fields_(scale_fields),
        retain_(retain_across_reset) {
    // The chain never grows past its cap, and fold_window_into_jacobian()
    // appends before it trims, so cap + 1 is the high-water mark; recycling
    // the evicted block does the rest. This is the only time blocks_ itself
    // reaches the allocator.
    blocks_.reserve(jacobian_cap() + 1);
  }

  void reset() override {
    if (retain_ && cols_ > 0 && n_ > 0) fold_window_into_jacobian();
    clear_window();
  }

  [[nodiscard]] std::string name() const override {
    return to_cstring(kind_);
  }

  [[nodiscard]] std::size_t columns() const override { return cols_; }

  [[nodiscard]] std::size_t retained_blocks() const noexcept override {
    return blocks_.size();
  }

  Result<void> update(const double* x_old, const double* x_new, double* x_out,
                      std::size_t n, int iteration) override {
    const Result<void> args =
        check_update_args(x_old, x_new, x_out, n, iteration);
    if (!args.ok()) return args;
    if (n == 0) return Result<void>::success();

    // A change of vector length invalidates every stored column and, for
    // IMVJ, the whole retained Jacobian: the operators no longer act on the
    // same space.
    if (n_ != n) {
      clear_window();
      blocks_.clear();
      chain_reprieves_ = 0;
      n_ = n;
    }
    // iteration 0 is a window boundary even if the caller did not call
    // reset(); the semantics must agree so that neither spelling silently
    // carries a stale column into a new window.
    if (iteration == 0) reset();

    // Grow-only, like every other buffer here: resize() never releases
    // capacity, so this is a no-op from the second call on.
    r_.resize(n);
    for (std::size_t i = 0; i < n; ++i) r_[i] = x_new[i] - x_old[i];

    if (have_prev_) append_column(x_new, r_);

    // Row scaling: the solve below minimises || D (V alpha + r) || with D the
    // per-field reciprocal-norm weights, so a field with a large magnitude
    // cannot dominate the fit. V and r are kept unscaled; only the solve --
    // and the filter, which has to judge the same matrix -- sees D.
    std::vector<double>& d = d_;
    if (scale_fields_) {
      field_weights(field_offsets_, r_.data(), n, d);
    } else {
      d.assign(n, 1.0);
    }
    filter_qn_columns(filter_, limit_, n_, d, v_, w_, cols_, filter_scratch_);

    const Result<void> step = accelerated_step(x_old, x_new, x_out, r_, d);

    // Recorded whatever the step did, so the next difference is taken against
    // what the caller actually evaluated.
    x_new_prev_.assign(x_new, x_new + n);
    // Swap rather than move-assign: both buffers keep their storage and swap
    // roles, so neither is ever reallocated. r_ is fully overwritten at the
    // top of the next call before anything reads it.
    r_.swap(r_prev_);
    have_prev_  = true;
    return step;
  }

 private:
  /// The retained chain's depth, from the policy documented on
  /// Accelerator::kMaxJacobianBlocks.
  [[nodiscard]] std::size_t jacobian_cap() const noexcept {
    return std::min<std::size_t>(max_columns_,
                                 static_cast<std::size_t>(kMaxJacobianBlocks));
  }

  /// Drops the per-window secant history. The buffers keep their capacity --
  /// clear() does not release it -- so the next window refills them without
  /// touching the allocator; `cols_` and `have_prev_`, not emptiness, are what
  /// say there is no history.
  void clear_window() {
    v_.clear();
    w_.clear();
    cols_ = 0;
    have_prev_ = false;
  }

  /// Appends Delta r to V and Delta x_new to W, dropping the oldest pair once
  /// the history depth would exceed `max_columns_`.
  ///
  /// An exactly zero residual difference is not appended: it carries no
  /// secant information, and admitting it would spend a history slot on a
  /// column the rank truncation is guaranteed to discard.
  void append_column(const double* x_new, const std::vector<double>& r) {
    const std::size_t n = n_;
    double dr2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = r[i] - r_prev_[i];
      dr2 += d * d;
    }
    if (!(dr2 > 0.0) || !std::isfinite(dr2)) return;

    if (cols_ == max_columns_) {
      // Drop the oldest pair. Erasing the leading column keeps the storage a
      // single flat buffer with no retained pointers into it; with
      // max_columns_ <= kMaxReuse this moves at most 31 columns.
      v_.erase(v_.begin(), v_.begin() + static_cast<std::ptrdiff_t>(n));
      w_.erase(w_.begin(), w_.begin() + static_cast<std::ptrdiff_t>(n));
      --cols_;
    }
    // Both vectors grow here, so nothing may hold a pointer into them across
    // this point; every access below indexes afresh.
    v_.reserve((cols_ + 1) * n);
    w_.reserve((cols_ + 1) * n);
    for (std::size_t i = 0; i < n; ++i) v_.push_back(r[i] - r_prev_[i]);
    for (std::size_t i = 0; i < n; ++i) {
      w_.push_back(x_new[i] - x_new_prev_[i]);
    }
    ++cols_;
  }

  /// y = Jprev^{-1} * src, with Jprev^{-1} = -I when no window has been
  /// folded in yet. That base case is what makes IQN-ILS the special case of
  /// IQN-IMVJ with an empty chain.
  ///
  /// Not const: the scratch it solves through belongs to the accelerator, for
  /// the no-allocation reason given on the class. Copy-assigning into `a` and
  /// `b` reuses their storage whenever it is large enough, which after the
  /// first pass over the chain it always is.
  void apply_prev_jacobian(const std::vector<double>& src,
                           std::vector<double>& y) {
    const std::size_t n = n_;
    y.resize(n);
    for (std::size_t i = 0; i < n; ++i) y[i] = -src[i];
    if (blocks_.empty()) return;

    std::vector<double>& a    = jac_a_;
    std::vector<double>& b    = jac_b_;
    std::vector<double>& beta = jac_beta_;
    for (const JacobianBlock& blk : blocks_) {
      a = blk.v;
      b = src;
      const std::size_t rank =
          linalg::least_squares_into(a, n, blk.cols, b, beta, ls_scratch_);
      if (rank == 0) continue;
      for (std::size_t j = 0; j < blk.cols; ++j) {
        const double c = beta[j];
        if (c == 0.0) continue;
        for (std::size_t i = 0; i < n; ++i) y[i] += blk.z[at(n, i, j)] * c;
      }
    }
  }

  /// Window boundary for IMVJ: freeze the current window's secant pairs into
  /// a new low-rank block of the retained inverse Jacobian.
  ///
  /// The new block is built, appended and only then is the chain trimmed --
  /// that order matters, because apply_prev_jacobian() below runs over the
  /// chain as it stands and would produce a different Z if the oldest block
  /// had already gone. What the trim evicts is moved into `spare_` rather than
  /// destroyed, and the next fold builds into `spare_`, so once the chain is
  /// at its cap a window boundary reuses that storage instead of trading an
  /// identically sized free for an allocation. NOT allocation-free in general,
  /// and this is a window boundary rather than the sweep loop so it does not
  /// have to be: a window that closes with more columns than the block being
  /// evicted grows `blk.v` and `blk.z` once each. It settles at zero when the
  /// window sizes do. The retained set, its order and its contents are bit for
  /// bit what the plain append-then-trim spelling produced.
  void fold_window_into_jacobian() {
    const std::size_t n   = n_;
    const std::size_t cap = jacobian_cap();

    JacobianBlock& blk = spare_;
    blk.cols = cols_;
    blk.v    = v_;                    // reuses the recycled block's storage
    blk.z.assign(n * cols_, 0.0);

    std::vector<double>& col = fold_col_;
    std::vector<double>& jv  = fold_jv_;
    col.resize(n);
    jv.resize(n);
    for (std::size_t j = 0; j < cols_; ++j) {
      for (std::size_t i = 0; i < n; ++i) col[i] = v_[at(n, i, j)];
      apply_prev_jacobian(col, jv);
      // Wx = Wn - V, so Wx[:, j] = w_[:, j] - v_[:, j].
      for (std::size_t i = 0; i < n; ++i) {
        blk.z[at(n, i, j)] = (w_[at(n, i, j)] - v_[at(n, i, j)]) - jv[i];
      }
    }
    blocks_.push_back(std::move(blk));  // spare_ is now moved-from, i.e. empty
    while (blocks_.size() > cap) {
      spare_ = std::move(blocks_.front());
      blocks_.erase(blocks_.begin());
    }
    // A different chain, so it is judged from the start again; see
    // kMaxChainReprieves.
    chain_reprieves_ = 0;
  }

  /// `d` holds the per-row least-squares weights computed by the caller (see
  /// update()); it is passed in rather than recomputed so that the filter and
  /// the solve provably see the same scaling.
  Result<void> accelerated_step(const double* x_old, const double* x_new,
                                double* x_out, const std::vector<double>& r,
                                const std::vector<double>& d) {
    const std::size_t n = n_;

    // No usable information at all: no columns this window and no retained
    // Jacobian. Every method in the family degrades to relaxation here, which
    // is the whole of the iteration-0 story for Secant and IQN-ILS.
    if (cols_ == 0 && blocks_.empty()) {
      relax_into(omega0_, x_old, x_new, x_out, n);
      // to_cstring, not name(): name() returns exactly this by value, and
      // building a std::string on the success path to hold a string literal
      // is a per-call cost that happens to be invisible only because every
      // kind name is short enough to fit in the small-string buffer.
      return finish_or_fallback(to_cstring(kind_), omega0_, x_old, x_new, x_out,
                                n);
    }

    std::vector<double>& alpha = alpha_;
    alpha.assign(cols_, 0.0);
    if (cols_ > 0) {
      std::vector<double>& a = ls_a_;
      a.resize(n * cols_);
      for (std::size_t j = 0; j < cols_; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
          a[at(n, i, j)] = d[i] * v_[at(n, i, j)];
        }
      }
      std::vector<double>& b = ls_b_;
      b.resize(n);
      for (std::size_t i = 0; i < n; ++i) b[i] = -d[i] * r[i];
      // A rank below cols_ is the ordinary outcome once the iteration nears
      // convergence, and needs no special case: the coefficients of the
      // dropped columns come back exactly 0, so the step simply uses the
      // directions the surviving columns do explain and leaves the rest to
      // the Jprev term below.
      static_cast<void>(
          linalg::least_squares_into(a, n, cols_, b, alpha, ls_scratch_));
    }

    // rho = r + V alpha, the part of the residual the columns cannot explain.
    std::vector<double>& rho = rho_;
    rho = r;
    for (std::size_t j = 0; j < cols_; ++j) {
      const double c = alpha[j];
      if (c == 0.0) continue;
      for (std::size_t i = 0; i < n; ++i) rho[i] += v_[at(n, i, j)] * c;
    }

    std::vector<double>& jrho = jrho_;
    apply_prev_jacobian(rho, jrho);

    // x_out = x_old + Wx alpha - Jprev^{-1} rho, with Wx = Wn - V.
    for (std::size_t i = 0; i < n; ++i) x_out[i] = x_old[i] - jrho[i];
    for (std::size_t j = 0; j < cols_; ++j) {
      const double c = alpha[j];
      if (c == 0.0) continue;
      for (std::size_t i = 0; i < n; ++i) {
        x_out[i] += (w_[at(n, i, j)] - v_[at(n, i, j)]) * c;
      }
    }

    // Overflow in the correction, not in the inputs. Leave the caller a finite
    // iterate but still fail: a non-finite step must never be reported as
    // progress. The columns that produced the blow-up are dropped, so a caller
    // that presses on with the fallback iterate rebuilds the history from
    // scratch instead of repeating the step.
    const Result<void> fin =
        finish_or_fallback(to_cstring(kind_), omega0_, x_old, x_new, x_out, n);
    if (!fin.ok()) {
      // Whether IMVJ's retained inverse Jacobian goes too has to be DECIDED,
      // not assumed either way, and the decision must be made here, before the
      // columns are dropped, because it is read off this step.
      //
      // Discarding the chain on any divergence would be wrong: the retained
      // Jacobian is the whole of IMVJ over IQN-ILS, most blow-ups come from
      // the current window's least squares, and throwing away every past
      // window because of one bad extrapolation turns IMVJ into IQN-ILS with
      // extra steps. Keeping a chain that just mapped a finite vector to a
      // non-finite one is equally wrong: the next call applies the same
      // operator and reproduces the same overflow -- the caller sees an
      // accelerator that can never recover.
      //
      // Two rules, and the second is what bounds the damage:
      //
      //   1. The chain goes if it is what made the difference: if the step
      //      this same window would have taken WITHOUT it -- the plain
      //      IQN-ILS step x_new + Wn alpha, which is this expression with
      //      Jprev^{-1} = -I -- is finite. A computation, not a threshold.
      //
      //   2. The chain goes anyway once it has been present for more than
      //      kMaxChainReprieves divergences it was not blamed for.
      //
      // Rule 2 is not belt and braces, it is the whole guarantee. Rule 1's
      // probe answers "would this step have been finite without the chain",
      // and a NO does not mean the chain is healthy -- the window's own
      // Wn alpha can overflow while the chain is poisoned too, and then the
      // probe exonerates a guilty chain. An earlier version of this code
      // argued that could not persist, because the diverged path clears the
      // window's columns and a probe at cols_ == 0 is x_new itself and
      // therefore finite. THAT ARGUMENT IS FALSE and was disproved by
      // execution: `have_prev_` survives the diverged return by design (see
      // below), so the very next call regenerates a column through
      // append_column() before the probe runs, cols_ is >= 1 again, and a
      // caller alternating two window-overflowing pairs holds a poisoned chain
      // for an unbounded run of failed calls --  11 in the test that found it.
      // ImvjDropsAPoisonedChainEvenWhenEveryProbeExoneratesIt is that
      // sequence, and rule 2 is what stops it: any one chain is present for at
      // most kMaxChainReprieves + 1 divergences, whatever the probe says and
      // whatever the caller does in between, because nothing but a fold --
      // which replaces the chain with a different one -- resets the count.
      bool drop_chain = false;
      if (!blocks_.empty()) {
        drop_chain = ils_step_is_finite(x_new) ||
                     ++chain_reprieves_ > kMaxChainReprieves;
      }
      v_.clear();
      w_.clear();
      cols_ = 0;
      if (drop_chain) {
        blocks_.clear();
        chain_reprieves_ = 0;
      }
    }
    return fin;
  }

  /// The step IQN-ILS would have taken from this window's columns and
  /// coefficients: x_new + Wn alpha, accumulated in the same order as the
  /// x_out loop above so that it really is the same arithmetic with the one
  /// term removed. Reads `alpha_`, so it is only meaningful immediately after
  /// accelerated_step() has filled it.
  ///
  /// TRUE means the chain is what diverged. FALSE means only that the window
  /// alone was enough to overflow -- NOT that the chain is sound; see rule 2
  /// above for what handles the case where both are.
  [[nodiscard]] bool ils_step_is_finite(const double* x_new) {
    const std::size_t n = n_;
    probe_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probe_[i] = x_new[i];
    for (std::size_t j = 0; j < cols_; ++j) {
      const double c = alpha_[j];
      if (c == 0.0) continue;
      for (std::size_t i = 0; i < n; ++i) probe_[i] += w_[at(n, i, j)] * c;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (!std::isfinite(probe_[i])) return false;
    }
    return true;
  }

  /// How many divergences a retained inverse Jacobian may be present for, and
  /// exonerated at, before it is dropped regardless of what the probe says.
  /// One: the first divergence in a window is very often the window's own
  /// least squares and costs the chain nothing, but a chain that is still
  /// there for a second is not worth the risk of an unbounded run of failures.
  /// The count is per chain -- folding a window in makes it a different
  /// operator and starts it again -- so this is not a budget that erodes over
  /// a long healthy run.
  static constexpr int kMaxChainReprieves = 1;

  AccelKind   kind_;
  double      omega0_;
  std::size_t max_columns_;
  FilterKind  filter_;
  double      limit_;
  bool        scale_fields_;
  bool        retain_;

  std::size_t         n_    = 0;
  std::size_t         cols_ = 0;
  std::vector<double> v_;  ///< n x cols_, column major, oldest column first
  std::vector<double> w_;  ///< n x cols_, column major, oldest column first
  std::vector<double> x_new_prev_;
  std::vector<double> r_prev_;
  bool                have_prev_ = false;
  std::vector<JacobianBlock> blocks_;
  /// Divergences the current `blocks_` has been present for and not blamed
  /// for. Reset when the chain changes or goes; see kMaxChainReprieves.
  int chain_reprieves_ = 0;
  /// Storage of the block the chain last evicted, kept so the next window
  /// boundary can refill it instead of allocating; see
  /// fold_window_into_jacobian(). Empty until the cap first binds.
  JacobianBlock spare_;

  // Scratch. None of it carries information between calls -- every element is
  // written before it is read -- and all of it exists so that update() does
  // not allocate. See the note on the class.
  std::vector<double> r_;      ///< this call's residual, x_new - x_old
  std::vector<double> d_;      ///< per-row least-squares weights
  std::vector<double> alpha_;  ///< least-squares coefficients of -r in V
  std::vector<double> ls_a_;   ///< the scaled V handed to the solve
  std::vector<double> ls_b_;   ///< the scaled -r handed to the solve
  std::vector<double> rho_;    ///< r + V alpha
  std::vector<double> jrho_;   ///< Jprev^{-1} rho
  std::vector<double> probe_;  ///< the IQN-ILS step, for the divergence probe
  std::vector<double> jac_a_;      ///< apply_prev_jacobian's copy of blk.v
  std::vector<double> jac_b_;      ///< apply_prev_jacobian's copy of src
  std::vector<double> jac_beta_;   ///< V_m^+ src
  std::vector<double> fold_col_;   ///< one column of V at a window boundary
  std::vector<double> fold_jv_;    ///< Jprev^{-1} of that column
  linalg::LsScratch   ls_scratch_;
  FilterScratch       filter_scratch_;
};

}  // namespace

std::unique_ptr<Accelerator> make_quasi_newton(const AccelConfig& cfg) {
  // Secant is the rank-1 member: the same least squares over a single column
  // pair, with nothing retained across windows.
  const bool        secant = (cfg.kind == AccelKind::Secant);
  const std::size_t cols =
      secant ? 1u : static_cast<std::size_t>(cfg.reuse);
  const bool retain = (cfg.kind == AccelKind::IQN_IMVJ);
  return std::make_unique<QuasiNewtonAccel>(cfg.kind, cfg.initial_relaxation,
                                            cols, cfg.filter, cfg.limit,
                                            cfg.scale_fields, retain);
}

}  // namespace detail
}  // namespace fuse
