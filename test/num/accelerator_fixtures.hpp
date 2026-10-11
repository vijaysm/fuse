#pragma once

/// The ladder/diagonal linear fixed-point problems shared between
/// accelerator_test.cpp and tools/src/write_accelerator_baseline.cpp, plus the
/// per-toolchain baseline file format both read/write. Pulled out of the test
/// file so the test and the tool that regenerates its pinned values can never
/// silently drift apart by each having their own copy of this math.
///
/// Why a baseline file at all: the iteration counts measured on these
/// problems are pinned exactly (not just bounded), and that pin is only valid
/// for a given floating-point codegen. Routing the same host-only arithmetic
/// through nvcc's host pass (FUSE_TOOLCHAIN_ID's "-nvcc_wrapper" suffix, see
/// fuse/base/config.hpp) measurably changes two of them. Rather than re-pin
/// one "true" set of literals that drifts again on the next
/// machine/compiler/CUDA-toolkit combination, each toolchain gets its own row
/// in test/golden/accelerator.golden, generated once by a human running
/// fuse_write_accelerator_baseline (tools/src/write_accelerator_baseline.cpp)
/// -- never by this test, and never in CI: a baseline a test can rewrite is
/// not a baseline.

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fuse/num/accelerator.hpp>

namespace {

/// A general affine fixed-point map G(x) = p + A (x - p).
struct LinearMap {
  std::size_t         n = 0;
  std::vector<double> a;  ///< n x n, row major
  std::vector<double> p;

  std::vector<double> apply(const std::vector<double>& x) const {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      double s = p[i];
      for (std::size_t j = 0; j < n; ++j) s += a[i * n + j] * (x[j] - p[j]);
      out[i] = s;
    }
    return out;
  }
};

/// A = H D H with H = I - 2 v v^T / (v^T v) a Householder reflector, so A is
/// dense and symmetric with exactly the eigenvalues D. `v` is deliberately not
/// parallel to `p`: with v == p the reflector maps the initial error onto its
/// own negative and the "dense" problem degenerates into a mirror image of the
/// diagonal one.
inline LinearMap dense_map(const std::vector<double>& d, const std::vector<double>& p,
                           const std::vector<double>& v) {
  const std::size_t n = d.size();
  double vtv = 0.0;
  for (double e : v) vtv += e * e;
  std::vector<double> h(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      h[i * n + j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vtv;
    }
  }
  std::vector<double> hd(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) hd[i * n + j] = h[i * n + j] * d[j];
  }
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += hd[i * n + k] * h[k * n + j];
      a[i * n + j] = s;
    }
  }
  return LinearMap{n, std::move(a), p};
}

inline LinearMap diagonal_map(const std::vector<double>& d,
                              const std::vector<double>& p) {
  const std::size_t n = d.size();
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) a[i * n + i] = d[i];
  return LinearMap{n, std::move(a), p};
}

/// Iteration count on a LinearMap; -1 if it did not converge, -2 if update()
/// reported an error. Starts from x = 0 every time, so a second call on the
/// same accelerator is a second time window over the same problem.
inline int iterate_linear(fuse::Accelerator& accel, const LinearMap& g, double tol,
                          int max_iters) {
  std::vector<double> x(g.n, 0.0);
  for (int k = 0; k < max_iters; ++k) {
    const std::vector<double> gx = g.apply(x);
    double residual = 0.0;
    for (std::size_t i = 0; i < g.n; ++i) {
      const double d = gx[i] - x[i];
      residual += d * d;
    }
    if (std::sqrt(residual) < tol) return k;

    std::vector<double> next(g.n);
    const auto r = accel.update(x.data(), gx.data(), next.data(), g.n, k);
    if (!r.ok()) return -2;
    x = std::move(next);
  }
  return -1;
}

/// The one problem the ladder is measured on. Eigenvalues 0.95 (slow), 0.9,
/// 0.8, 0.5 and -0.7 (oscillating), in a basis where none of them is aligned
/// with a coordinate axis.
inline LinearMap ladder_problem() {
  return dense_map({0.95, 0.9, 0.8, 0.5, -0.7}, {1.0, 2.0, 3.0, 4.0, 5.0},
                   {1.0, -2.0, 0.5, 3.0, -1.5});
}

/// No gtest dependency here (this header is also included by
/// tools/src/write_accelerator_baseline.cpp, a plain executable): every
/// config passed through this file is known-valid, so a construction failure
/// means the fixture itself is broken, not the thing under test.
inline std::unique_ptr<fuse::Accelerator> must_make(const fuse::AccelConfig& cfg) {
  auto made = fuse::make_accelerator(cfg);
  if (!made.ok()) {
    std::fprintf(stderr, "must_make: make_accelerator failed: %s\n",
                 made.error().message.c_str());
    std::abort();
  }
  return std::move(made.value());
}

inline constexpr double kTol      = 1e-10;
inline constexpr int    kMaxIters = 5000;

/// The pinned values LadderIsMonotoneOnADenseLinearProblem and
/// SteffensenIsExactOnADiagonalProblemAndBasisDependent measure, for one
/// toolchain.
struct AcceleratorBaseline {
  int n_picard = 0, n_constant = 0, n_secant = 0, n_aitken = 0;
  int n_ils = 0, n_imvj = 0, n_diag = 0, n_dense = 0;
};

/// Reads `path`, which is "<toolchain_id> <8 ints>" data lines (fields in the
/// same order as AcceleratorBaseline's members above) with '#'-prefixed and
/// blank lines ignored. Returns true and fills `out` iff a line's first field
/// equals `toolchain_id` exactly.
inline bool read_accelerator_baseline(const std::string& path, const std::string& toolchain_id,
                                      AcceleratorBaseline* out) {
  std::ifstream in(path);
  if (!in.is_open()) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::string id;
    AcceleratorBaseline b;
    if (!(row >> id >> b.n_picard >> b.n_constant >> b.n_secant >> b.n_aitken >>
          b.n_ils >> b.n_imvj >> b.n_diag >> b.n_dense)) {
      continue;
    }
    if (id == toolchain_id) {
      *out = b;
      return true;
    }
  }
  return false;
}

/// Rewrites `path`: replaces the data line for `toolchain_id` if one exists,
/// else appends a new one. Comments, blank lines, and every other toolchain's
/// row are preserved verbatim and in place. Creates `path` if it does not
/// exist yet (a fresh header comment is written first).
inline void upsert_accelerator_baseline(const std::string& path, const std::string& toolchain_id,
                                        const AcceleratorBaseline& row) {
  std::vector<std::string> lines;
  {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
  }
  if (lines.empty()) {
    lines.push_back(
        "# toolchain_id n_picard n_constant n_secant n_aitken n_ils n_imvj n_diag n_dense");
  }
  std::ostringstream new_row;
  new_row << toolchain_id << ' ' << row.n_picard << ' ' << row.n_constant << ' '
          << row.n_secant << ' ' << row.n_aitken << ' ' << row.n_ils << ' '
          << row.n_imvj << ' ' << row.n_diag << ' ' << row.n_dense;

  bool replaced = false;
  for (std::string& line : lines) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream existing(line);
    std::string id;
    existing >> id;
    if (id == toolchain_id) {
      line     = new_row.str();
      replaced = true;
      break;
    }
  }
  if (!replaced) lines.push_back(new_row.str());

  std::ofstream out(path, std::ios::trunc);
  for (const std::string& line : lines) out << line << '\n';
}

}  // namespace
