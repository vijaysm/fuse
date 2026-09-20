// Explicit, not transitive: the feature macros are read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"

#include "fuse/linalg/least_squares.hpp"

#include <cstddef>
#include <vector>

#include "fuse/base/error.hpp"

#if FUSE_LINALG_EIGEN
#include <Eigen/Dense>
#endif

namespace fuse::linalg {

namespace {

/// Column-major indexing, matching how the accelerator stores its difference
/// vectors: one column per retained iteration, contiguous down the rows.
inline std::size_t at(std::size_t m, std::size_t i, std::size_t j) noexcept {
  return j * m + i;
}

void check_shape(const std::vector<double>& a, std::size_t m, std::size_t k,
                 const std::vector<double>& b, std::vector<double>& out) {
  // A caller mistake, not a numerical outcome, so it throws rather than
  // returning a rank the caller would have to interpret.
  if (m < k) {
    FUSE_THROW(InvalidArgument,
               "least_squares: {} rows for {} columns; the system must not be "
               "wider than it is tall",
               m, k);
  }
  if (a.size() < m * k) {
    FUSE_THROW(InvalidArgument,
               "least_squares: matrix holds {} entries, {}x{} needs {}",
               a.size(), m, k, m * k);
  }
  if (b.size() < m) {
    FUSE_THROW(InvalidArgument,
               "least_squares: right-hand side holds {} entries, {} rows need {}",
               b.size(), m, m);
  }
  out.assign(k, 0.0);
}

}  // namespace

Backend configured_backend() noexcept {
  // Reports what RUNS, not what the option asked for. A backend that reported
  // itself while another did the work would make an iteration-count report
  // unusable, which is the one thing this accessor exists for.
#if FUSE_LINALG_EIGEN
  return Backend::Eigen;
#else
  return Backend::Builtin;
#endif
}

const char* backend_name(Backend b) noexcept {
  switch (b) {
    case Backend::Builtin: return "builtin";
    case Backend::Eigen:   return "eigen";
  }
  return "unknown";
}

/// The built-in factorisation with the scratch supplied by the caller: this is
/// the variant the accelerator's inner loop calls, so a steady-state solve
/// touches no allocator.
std::size_t least_squares_into(std::vector<double>& a, std::size_t m,
                                  std::size_t k, std::vector<double>& b,
                                  std::vector<double>& out, LsScratch& work) {
  out.assign(k, 0.0);
  if (m == 0 || k == 0) return 0;

  std::vector<std::size_t>& perm = work.perm;
  std::vector<double>&      colnorm2 = work.colnorm2;
  perm.resize(k);
  colnorm2.resize(k);
  for (std::size_t j = 0; j < k; ++j) {
    perm[j] = j;
    double s = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      const double v = a[at(m, i, j)];
      s += v * v;
    }
    colnorm2[j] = s;
  }

  const std::size_t steps = std::min(m, k);
  const double      eps   = std::numeric_limits<double>::epsilon();
  const double      tol_rel =
      static_cast<double>(std::max(m, k)) * eps;

  double      pivot0 = 0.0;
  std::size_t rank   = 0;

  for (std::size_t j = 0; j < steps; ++j) {
    std::size_t p = j;
    for (std::size_t t = j + 1; t < k; ++t) {
      if (colnorm2[t] > colnorm2[p]) p = t;
    }
    const double best = colnorm2[p] > 0.0 ? std::sqrt(colnorm2[p]) : 0.0;
    if (j == 0) pivot0 = best;
    // Rank truncation. Everything from here on is indistinguishable from the
    // rounding of the columns already accepted.
    if (!(best > tol_rel * pivot0) || !(best > 0.0)) break;

    if (p != j) {
      for (std::size_t i = 0; i < m; ++i) {
        std::swap(a[at(m, i, j)], a[at(m, i, p)]);
      }
      std::swap(perm[j], perm[p]);
      std::swap(colnorm2[j], colnorm2[p]);
    }

    // Householder reflector on a[j:m, j].
    double norm_x = 0.0;
    for (std::size_t i = j; i < m; ++i) {
      const double v = a[at(m, i, j)];
      norm_x += v * v;
    }
    norm_x = std::sqrt(norm_x);
    if (!(norm_x > 0.0) || !std::isfinite(norm_x)) break;

    const double head  = a[at(m, j, j)];
    const double alpha = (head >= 0.0) ? -norm_x : norm_x;

    std::vector<double>& v = work.reflector;
    v.resize(m - j);
    for (std::size_t i = j; i < m; ++i) v[i - j] = a[at(m, i, j)];
    v[0] -= alpha;
    double vtv = 0.0;
    for (double e : v) vtv += e * e;
    if (!(vtv > 0.0) || !std::isfinite(vtv)) {
      // Already upper triangular in this column; nothing to reflect.
      a[at(m, j, j)] = head;
      ++rank;
      for (std::size_t t = j + 1; t < k; ++t) {
        const double rj = a[at(m, j, t)];
        colnorm2[t] -= rj * rj;
        if (colnorm2[t] < 0.0) colnorm2[t] = 0.0;
      }
      continue;
    }

    for (std::size_t t = j; t < k; ++t) {
      double dot = 0.0;
      for (std::size_t i = j; i < m; ++i) dot += v[i - j] * a[at(m, i, t)];
      const double f = 2.0 * dot / vtv;
      for (std::size_t i = j; i < m; ++i) a[at(m, i, t)] -= f * v[i - j];
    }
    {
      double dot = 0.0;
      for (std::size_t i = j; i < m; ++i) dot += v[i - j] * b[i];
      const double f = 2.0 * dot / vtv;
      for (std::size_t i = j; i < m; ++i) b[i] -= f * v[i - j];
    }
    a[at(m, j, j)] = alpha;
    ++rank;

    // Downdate the trailing column norms; recompute when cancellation has
    // eaten the estimate, which is what makes pivoting reliable rather than
    // merely cheap.
    for (std::size_t t = j + 1; t < k; ++t) {
      const double rj = a[at(m, j, t)];
      const double updated = colnorm2[t] - rj * rj;
      if (updated <= 0.125 * colnorm2[t] || updated < 0.0) {
        double s = 0.0;
        for (std::size_t i = j + 1; i < m; ++i) {
          const double e = a[at(m, i, t)];
          s += e * e;
        }
        colnorm2[t] = s;
      } else {
        colnorm2[t] = updated;
      }
    }
  }

  if (rank == 0) return 0;

  // Back substitution on the leading rank x rank triangle.
  std::vector<double>& z = work.z;
  z.resize(rank);
  for (std::size_t ii = rank; ii-- > 0;) {
    double s = b[ii];
    for (std::size_t t = ii + 1; t < rank; ++t) s -= a[at(m, ii, t)] * z[t];
    const double d = a[at(m, ii, ii)];
    if (!(std::abs(d) > 0.0) || !std::isfinite(d)) return 0;
    z[ii] = s / d;
    if (!std::isfinite(z[ii])) return 0;
  }
  for (std::size_t ii = 0; ii < rank; ++ii) out[perm[ii]] = z[ii];
  return rank;
}

std::size_t least_squares_builtin(std::vector<double>& a, std::size_t m,
                                  std::size_t k, std::vector<double>& b,
                                  std::vector<double>& out) {
  check_shape(a, m, k, b, out);
  if (k == 0) return 0;
  LsScratch work;
  return least_squares_into(a, m, k, b, out, work);
}

#if FUSE_LINALG_EIGEN
namespace {

/// Eigen's column-pivoted Householder QR, with the rank decided by Eigen's own
/// threshold rather than a second one of ours: two thresholds on the same
/// factorisation would disagree at the boundary, and the boundary is where the
/// filtered columns live.
std::size_t least_squares_eigen(std::vector<double>& a, std::size_t m,
                                std::size_t k, std::vector<double>& b,
                                std::vector<double>& out) {
  check_shape(a, m, k, b, out);
  if (k == 0) return 0;
  using Map = Eigen::Map<Eigen::MatrixXd>;
  Map A(a.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
  Eigen::Map<Eigen::VectorXd> rhs(b.data(), static_cast<Eigen::Index>(m));
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
  const auto rank = static_cast<std::size_t>(qr.rank());
  const Eigen::VectorXd sol = qr.solve(rhs);
  for (std::size_t j = 0; j < k; ++j) {
    out[j] = sol(static_cast<Eigen::Index>(j));
  }
  // Eigen's solve() writes a coefficient for every column, including ones its
  // rank decision dropped. Zeroing those keeps the facade's contract: a zero
  // means "filtered", not "small".
  if (rank < k) {
    const Eigen::VectorXi& perm = qr.colsPermutation().indices();
    for (std::size_t j = rank; j < k; ++j) {
      out[static_cast<std::size_t>(perm(static_cast<Eigen::Index>(j)))] = 0.0;
    }
  }
  return rank;
}

}  // namespace
#endif

std::size_t least_squares(std::vector<double>& a, std::size_t m, std::size_t k,
                          std::vector<double>& b, std::vector<double>& out) {
#if FUSE_LINALG_EIGEN
  return least_squares_eigen(a, m, k, b, out);
#else
  return least_squares_builtin(a, m, k, b, out);
#endif
}

}  // namespace fuse::linalg
