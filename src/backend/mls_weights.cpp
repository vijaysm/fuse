#include "fuse/backend/mls_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include <Kokkos_Core.hpp>

#include "fuse/backend/mls_weights_device.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/linalg/least_squares.hpp"
#include "fuse/search/box_index.hpp"
#include "fuse/search/geometry.hpp"
#include "fuse/search/types.hpp"

namespace fuse {

namespace {

// ---------------------------------------------------------------------------
// Small numeric helpers. None of these throw; all take finite, already-
// validated inputs (degree/dim checked by the caller before either is used).
// ---------------------------------------------------------------------------

[[nodiscard]] double ipow(double base, int exp) noexcept {
  double r = 1.0;
  for (int i = 0; i < exp; ++i) r *= base;
  return r;
}

/// Total-degree monomial basis of `x, y, z` up to `degree`, in `dim`
/// variables. Ordering is internal to this file -- nothing compares it
/// against ArborX's own (`docs/user/point-cloud-remap.md`'s cross-check
/// compares INTERPOLATED VALUES, not raw basis vectors) -- and only needs to
/// be a consistent, complete total-degree basis, which it is: for `dim == 2`
/// each outer step `total` contributes the `total + 1` monomials `x^a y^b`
/// with `a + b == total`; `dim == 3` nests the same rule one level deeper.
[[nodiscard]] std::vector<double> polynomial_basis(int dim, int degree, double x, double y,
                                                    double z) {
  std::vector<double> out;
  out.reserve(mls_basis_size(dim, degree));
  for (int total = 0; total <= degree; ++total) {
    if (dim == 1) {
      out.push_back(ipow(x, total));
    } else if (dim == 2) {
      for (int a = total; a >= 0; --a) {
        const int b = total - a;
        out.push_back(ipow(x, a) * ipow(y, b));
      }
    } else {  // dim == 3
      for (int a = total; a >= 0; --a) {
        for (int b = total - a; b >= 0; --b) {
          const int c = total - a - b;
          out.push_back(ipow(x, a) * ipow(y, b) * ipow(z, c));
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Mesh entity set resolution. Deliberately a SEPARATE, smaller copy of
// `intx_weights.cpp`'s anonymous-namespace `supermesh_side` rather than a
// shared call: that helper is not exported past its own translation unit
// (mls_weights.cpp does not own intx_weights.cpp), and this version accepts
// `Discretization`/order rather than assuming FV, which is the whole reason
// this builder exists. `which` is "source"/"target", for the message only.
// ---------------------------------------------------------------------------

struct MlsSide {
  const MeshView* view = nullptr;
  int             dim = 0;
};

[[nodiscard]] Result<MlsSide> resolve_mls_side(const CouplingGraph& graph,
                                               const MeshBackend& backend, RegionId id,
                                               ComponentId placement, Discretization disc,
                                               int order, const char* which) {
  if (!id.valid() || id.value() >= graph.region_count()) {
    return Failure{"E-MLS-REGION",
                   fmt::format("{} region id {} is not one of the graph's {} regions", which,
                               id.value(), graph.region_count())};
  }
  const Region& r = graph.region(id);
  if (!(r.entity_set.empty() || r.entity_set == "all")) {
    return Failure{
        "E-MLS-REGION",
        fmt::format("{} region '{}' names entity set '{}'. MLS weights index MeshView "
                    "local indices over the whole mesh, so a sub-set region would give a "
                    "matrix wrong by a permutation",
                    which, r.name, r.entity_set)};
  }
  if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
    return Failure{"E-MLS-REGION",
                   fmt::format("{} region '{}' names no valid mesh", which, r.name)};
  }
  const auto mid = backend.mesh_id(graph.mesh(r.mesh).name);
  if (!mid.ok()) return mid.error();
  const auto view_r = backend.placed_view(mid.value(), placement);
  if (!view_r.ok()) return view_r.error();

  MlsSide s;
  s.view = view_r.value();
  if (disc == Discretization::PointCloud) {
    s.dim = 0;
  } else if (disc == Discretization::ContinuousGLL) {
    if (order != 1) {
      return Failure{
          "E-MLS-DISC",
          fmt::format("{} side declares ContinuousGLL order {}; only order 1 "
                      "(vertex-nodal, mesh_view.hpp's field_extent convention) has a "
                      "point extraction for MLS",
                      which, order)};
    }
    s.dim = 0;
  } else if (disc == Discretization::FV ||
            (disc == Discretization::DiscontinuousGLL && order == 0)) {
    s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();
  } else {
    return Failure{
        "E-MLS-DISC",
        fmt::format("{} side declares {} order {}, which has no point extraction for "
                    "MLS (supported: PointCloud, ContinuousGLL order 1, FV/"
                    "DiscontinuousGLL order 0)",
                    which, to_cstring(disc), order)};
  }
  return s;
}

/// Every local entity of `view` at `dim`, as a `fuse::search::Point`. Serial
/// only (this file's own "SERIAL ONLY" section): no ownership filtering, the
/// same convention `remap_plan.hpp`'s INDEX SPACE note documents for every
/// other builder ("row and col are MeshView LOCAL INDICES").
[[nodiscard]] std::vector<search::Point> gather_points(const MeshView& view, int dim) {
  const std::size_t n = view.entity_count(dim);
  std::vector<search::Point> pts;
  pts.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Point p = view.centroid(dim, i);
    pts.push_back(search::Point{p.x, p.y, p.z});
  }
  return pts;
}

/// The ambient dimension (1, 2 or 3) of the union of two point sets, via
/// `fuse::search::classify_dim` on their joint bounding box -- the same rule
/// `Point`'s own contract states (an unused axis is EXACTLY zero), so a
/// "point cloud on a sphere" classifies as 3, not 2: this builder fits in
/// the ambient space, never on a manifold (this header's own "WHAT DEGREE
/// AND K MEAN" section). At least one of the two sets must be non-empty.
[[nodiscard]] int classify_point_dim(const std::vector<search::Point>& a,
                                     const std::vector<search::Point>& b) {
  search::Point lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::max()};
  search::Point hi{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
                   std::numeric_limits<double>::lowest()};
  const auto fold = [&](const search::Point& p) {
    lo.x = std::min(lo.x, p.x);
    lo.y = std::min(lo.y, p.y);
    lo.z = std::min(lo.z, p.z);
    hi.x = std::max(hi.x, p.x);
    hi.y = std::max(hi.y, p.y);
    hi.z = std::max(hi.z, p.z);
  };
  for (const auto& p : a) fold(p);
  for (const auto& p : b) fold(p);
  return search::classify_dim(search::Box{lo, hi});
}

// ---------------------------------------------------------------------------
// One target's local moment-matrix assembly and solve, factored out of the
// serial per-target loop below so the SAME code -- not a second hand-written
// copy that could silently drift from it -- backs two call sites: the Host
// loop's own per-target iteration, and the Device path's recovery of
// `E-MLS-RANK`'s full diagnostic (target index, rank, condition estimate)
// for whichever target `fuse::device::solve_mls_on_space`'s coarser
// pass/fail status flagged (mls_weights_device.hpp's own "WHAT THIS DOES
// NOT DO" section). Pulling this out changes no arithmetic and no
// evaluation order from the Host path's pre-existing loop body, so Host
// mode's result is unaffected (bitwise).
// ---------------------------------------------------------------------------

struct MlsTargetFit {
  std::vector<std::vector<double>> basis_rows;  ///< k entries, each p_size long.
  std::vector<double>              phi;         ///< k entries.
  std::vector<double>              coeff;       ///< p_size entries.
};

[[nodiscard]] Result<MlsTargetFit> mls_fit_one_target(
    int dim, int degree, MlsRadial radial, std::size_t p_size, std::size_t t,
    const std::vector<search::Point>& src_pts, const search::Point& tp,
    const std::vector<std::size_t>& nbr_src_indices) {
  const std::size_t k = nbr_src_indices.size();
  std::vector<double> dists(k);
  double max_dist = 0.0;
  for (std::size_t local = 0; local < k; ++local) {
    const search::Point& sp = src_pts[nbr_src_indices[local]];
    const double dx = sp.x - tp.x;
    const double dy = sp.y - tp.y;
    const double dz = sp.z - tp.z;
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    dists[local] = d;
    max_dist = std::max(max_dist, d);
  }
  // Same 1.1x-past-the-farthest-neighbour, epsilon-floored radius the Host
  // loop uses (see build_mls_weights's own comment for why).
  const double radius = 1.1 * std::max(max_dist, std::numeric_limits<double>::epsilon());

  const auto at = [p_size](std::size_t i, std::size_t j) { return j * p_size + i; };
  std::vector<double> moment(p_size * p_size, 0.0);
  std::vector<double> phi(k);
  std::vector<std::vector<double>> basis_rows(k);
  for (std::size_t local = 0; local < k; ++local) {
    const search::Point& sp = src_pts[nbr_src_indices[local]];
    phi[local] = mls_crbf(radial, dists[local] / radius);
    basis_rows[local] = polynomial_basis(dim, degree, sp.x - tp.x, sp.y - tp.y, sp.z - tp.z);
    const std::vector<double>& b = basis_rows[local];
    for (std::size_t a = 0; a < p_size; ++a) {
      for (std::size_t c = 0; c < p_size; ++c) {
        moment[at(a, c)] += phi[local] * b[a] * b[c];
      }
    }
  }

  std::vector<double> rhs(p_size, 0.0);
  rhs[0] = 1.0;
  std::vector<double> coeff(p_size);
  std::vector<double> a_copy = moment;  // least_squares_builtin mutates in place.
  const std::size_t rank =
      fuse::linalg::least_squares_builtin(a_copy, p_size, p_size, rhs, coeff);

  if (rank < p_size) {
    const double diag0 = std::fabs(a_copy[at(0, 0)]);
    const double diag_last = rank >= 1 ? std::fabs(a_copy[at(rank - 1, rank - 1)]) : 0.0;
    const std::string cond_str = (rank >= 1 && diag_last > 0.0)
                                     ? fmt::format("{:.3e}", diag0 / diag_last)
                                     : std::string("singular (no accepted pivot)");
    return Failure{
        "E-MLS-RANK",
        fmt::format("target {}'s local degree-{} fit in {}D is rank-deficient: rank {} "
                    "of {} basis functions (its {} neighbours are too coincident, "
                    "collinear or coplanar for this degree). Condition estimate "
                    "(largest/smallest accepted pivot magnitude): {}",
                    t, degree, dim, rank, p_size, k, cond_str)};
  }

  MlsTargetFit fit;
  fit.basis_rows = std::move(basis_rows);
  fit.phi = std::move(phi);
  fit.coeff = std::move(coeff);
  return fit;
}

// ---------------------------------------------------------------------------
// Device path: the k-nearest search already ran on
// `fuse::search::resolve_execution_space()`'s resolved space by the time
// this is called (`BoxIndex::build`'s own `space` argument, threaded
// through from `build_mls_weights` below) -- this function performs the
// per-target moment-matrix assembly and solve on the SAME space, via
// `fuse::device::solve_mls_on_space`, then deep-copies the result back to
// host `RemapWeights` triplets. `knn` is the shared k-nearest result both
// the Host and Device paths resolve from (identical either way: search.md's
// own determinism guarantee -- "sorted ascending by index" -- holds
// regardless of which space actually ran the query), so a target's column
// order is identical between the two paths.
//
// `solve_mls_on_space`'s own elimination (an absolute pivot floor) is NOT
// the Host path's column-pivoted, rank-revealing QR (mls_weights_device.hpp's
// own header comment), so a target it flags as failing is CONFIRMED, not
// merely trusted: this function re-derives that target's full diagnostic by
// calling `mls_fit_one_target` -- the Host path's own code -- with the
// SAME neighbour set the device solve used. Two outcomes follow: the Host
// code agrees (a genuine rank deficiency) and `E-MLS-RANK` is returned
// naming that exact target and its Host-computed condition estimate, same
// as if the Host path itself had reached it; or the Host code disagrees
// (the device solve's coarser floor was over-eager) and that one target's
// weights are patched in from the Host-computed fit instead of the device's
// (meaningless, early-returned) elimination state, so an isolated
// disagreement never ships a wrong number silently. Scanned in ascending
// target order, matching the Host loop's own "stop at the first failing
// target" sequencing: if more than one target disagrees, the first one
// (by index) is what both paths would have named.
// ---------------------------------------------------------------------------

[[nodiscard]] Result<RemapWeights> build_mls_weights_device(
    int dim, const MlsOptions& opt, std::size_t p_size, const std::vector<search::Point>& src_pts,
    const std::vector<search::Point>& tgt_pts, const search::QueryResult& knn) {
  using Space = Kokkos::DefaultExecutionSpace::memory_space;
  const std::size_t n_cols = src_pts.size();
  const std::size_t n_rows = tgt_pts.size();
  const std::size_t k = static_cast<std::size_t>(opt.k);

  // Flat (n_rows * k) neighbour-index table, shared by the device solve and
  // by this function's own host-side diagnostic recovery -- built once from
  // `knn`, exactly the per-target slice the Host loop reads from the same
  // CSR.
  std::vector<std::int32_t> nbr_flat(n_rows * k);
  for (std::size_t t = 0; t < n_rows; ++t) {
    const std::size_t begin = knn.offsets[t];
    const std::size_t end = knn.offsets[t + 1];
    if (end - begin != k) {
      return Failure{"E-MLS-NEIGHBORS",
                     fmt::format("target {} got {} neighbour(s), fewer than the declared "
                                 "k={}",
                                 t, end - begin, opt.k)};
    }
    for (std::size_t j = begin; j < end; ++j) {
      nbr_flat[t * k + (j - begin)] = static_cast<std::int32_t>(knn.indices[j]);
    }
  }

  // A defect class to avoid (a `double* [3]` rank-2 view's
  // PER-SPACE default layout -- LayoutRight on Host, LayoutLeft on Cuda --
  // makes a plain `Kokkos::View<double* [3], Kokkos::HostSpace>` mirror
  // layout-INCOMPATIBLE with a same-shape `Kokkos::View<double* [3], Space>`
  // device view; `Kokkos::deep_copy` cannot bridge a memory-space change
  // and a layout change in the same call), hit here by this header's own
  // new views rather than `native_mesh.hpp`'s fixed ones: `Space` is left
  // implicit on these views (matching `solve_mls_on_space`'s own
  // unqualified `Kokkos::View<double* [3], Space>` parameter type exactly,
  // so the call below is not a second type to keep in sync), so the fix is
  // `Kokkos::create_mirror_view`, which derives the mirror's layout FROM
  // the device view it mirrors instead of assuming Host's own default.
  Kokkos::View<double* [3], Space> src_d(
      Kokkos::view_alloc("fuse::mls_device_src_d", Kokkos::WithoutInitializing), n_cols);
  auto src_h = Kokkos::create_mirror_view(src_d);
  for (std::size_t i = 0; i < n_cols; ++i) {
    src_h(i, 0) = src_pts[i].x;
    src_h(i, 1) = src_pts[i].y;
    src_h(i, 2) = src_pts[i].z;
  }
  Kokkos::deep_copy(src_d, src_h);

  Kokkos::View<double* [3], Space> tgt_d(
      Kokkos::view_alloc("fuse::mls_device_tgt_d", Kokkos::WithoutInitializing), n_rows);
  auto tgt_h = Kokkos::create_mirror_view(tgt_d);
  for (std::size_t i = 0; i < n_rows; ++i) {
    tgt_h(i, 0) = tgt_pts[i].x;
    tgt_h(i, 1) = tgt_pts[i].y;
    tgt_h(i, 2) = tgt_pts[i].z;
  }
  Kokkos::deep_copy(tgt_d, tgt_h);

  Kokkos::View<int*, Kokkos::HostSpace> nbr_h(
      Kokkos::view_alloc("fuse::mls_device_nbr_h", Kokkos::WithoutInitializing), n_rows * k);
  for (std::size_t i = 0; i < nbr_flat.size(); ++i) nbr_h(i) = nbr_flat[i];
  Kokkos::View<int*, Space> nbr_d(
      Kokkos::view_alloc("fuse::mls_device_nbr_d", Kokkos::WithoutInitializing), n_rows * k);
  Kokkos::deep_copy(nbr_d, nbr_h);

  const fuse::device::MlsDeviceResult<Space> device_result =
      fuse::device::solve_mls_on_space<Space>(dim, opt.degree, opt.k, opt.radial, src_d, tgt_d,
                                              nbr_d);

  auto row_h = Kokkos::create_mirror_view(device_result.row);
  auto col_h = Kokkos::create_mirror_view(device_result.col);
  auto val_h = Kokkos::create_mirror_view(device_result.val);
  auto status_h = Kokkos::create_mirror_view(device_result.status);
  Kokkos::deep_copy(row_h, device_result.row);
  Kokkos::deep_copy(col_h, device_result.col);
  Kokkos::deep_copy(val_h, device_result.val);
  Kokkos::deep_copy(status_h, device_result.status);

  RemapWeights out;
  out.n_rows = n_rows;
  out.n_cols = n_cols;
  out.order = opt.degree + 1;
  out.max_col_defect = std::numeric_limits<double>::infinity();
  out.max_col_defect_native = std::numeric_limits<double>::infinity();
  out.row.reserve(n_rows * k);
  out.col.reserve(n_rows * k);
  out.val.reserve(n_rows * k);

  double max_row_defect = 0.0;
  double min_weight = std::numeric_limits<double>::infinity();

  for (std::size_t t = 0; t < n_rows; ++t) {
    if (status_h(static_cast<int>(t)) != 0) {
      // Confirm (or refute) the device's coarser verdict with the Host
      // path's own rank-revealing solve, on the SAME neighbour set.
      std::vector<std::size_t> nbr_idx(k);
      for (std::size_t local = 0; local < k; ++local) {
        nbr_idx[local] = static_cast<std::size_t>(nbr_flat[t * k + local]);
      }
      auto fit_r = mls_fit_one_target(dim, opt.degree, opt.radial, p_size, t, src_pts,
                                      tgt_pts[t], nbr_idx);
      if (!fit_r.ok()) return fit_r.error();  // confirmed: same E-MLS-RANK the Host path would raise.
      // Not confirmed: patch this one row from the Host-computed fit rather
      // than the device's early-returned (meaningless) elimination state.
      const MlsTargetFit& fit = fit_r.value();
      double row_sum = 0.0;
      for (std::size_t local = 0; local < k; ++local) {
        double dotc = 0.0;
        for (std::size_t a = 0; a < p_size; ++a) dotc += fit.coeff[a] * fit.basis_rows[local][a];
        const double w = fit.phi[local] * dotc;
        out.row.push_back(static_cast<std::int32_t>(t));
        out.col.push_back(static_cast<std::int32_t>(nbr_idx[local]));
        out.val.push_back(w);
        row_sum += w;
        min_weight = std::min(min_weight, w);
      }
      max_row_defect = std::max(max_row_defect, std::fabs(row_sum - 1.0));
      continue;
    }

    double row_sum = 0.0;
    for (std::size_t local = 0; local < k; ++local) {
      const std::size_t idx = t * k + local;
      out.row.push_back(static_cast<std::int32_t>(row_h(idx)));
      out.col.push_back(static_cast<std::int32_t>(col_h(idx)));
      out.val.push_back(val_h(idx));
      row_sum += val_h(idx);
      min_weight = std::min(min_weight, val_h(idx));
    }
    max_row_defect = std::max(max_row_defect, std::fabs(row_sum - 1.0));
  }

  out.max_row_defect = max_row_defect;
  out.min_weight = out.val.empty() ? 0.0 : min_weight;
  out.monotone = (out.min_weight >= -1e-12) && (max_row_defect <= 1e-12);
  return out;
}

}  // namespace

const char* to_cstring(MlsRadial r) noexcept {
  switch (r) {
    case MlsRadial::Wendland0: return "wendland0";
    case MlsRadial::Wendland2: return "wendland2";
    case MlsRadial::Wendland4: return "wendland4";
    case MlsRadial::Wendland6: return "wendland6";
  }
  return "unknown";
}

double mls_crbf(MlsRadial r, double x) noexcept {
  // Same clamp ArborX's own CRBF::evaluate applies: CRBF(-a) == CRBF(a) ==
  // CRBF(|a|), so the argument is folded into [0, 1] before the closed form
  // runs, and a caller (this file's own `build_mls_weights`) need not
  // pre-clamp.
  const double t = std::min(std::fabs(x), 1.0);
  const double q = 1.0 - t;
  switch (r) {
    case MlsRadial::Wendland0:
      return q * q;
    case MlsRadial::Wendland2:
      return std::pow(q, 4) * (4.0 * t + 1.0);
    case MlsRadial::Wendland4:
      return std::pow(q, 6) * (35.0 * t * t + 18.0 * t + 3.0);
    case MlsRadial::Wendland6:
      return std::pow(q, 8) * (32.0 * t * t * t + 25.0 * t * t + 8.0 * t + 1.0);
  }
  return 0.0;
}

std::size_t mls_basis_size(int dim, int degree) noexcept {
  if (dim < 1 || dim > 3 || degree < 0) return 0;
  // C(dim + degree, dim) = prod_{i=1..dim}(degree + i) / dim! -- the product
  // is always an exact multiple of dim! (it IS the binomial coefficient
  // times dim!), so a single division at the end is exact.
  std::size_t num = 1;
  std::size_t den = 1;
  for (int i = 1; i <= dim; ++i) {
    num *= static_cast<std::size_t>(degree + i);
    den *= static_cast<std::size_t>(i);
  }
  return num / den;
}

Result<MlsOptions> parse_mls_options(std::string_view method) {
  constexpr std::string_view kPrefix = "mls";
  if (method.size() < kPrefix.size() || method.substr(0, kPrefix.size()) != kPrefix) {
    return Failure{"E-MLS-OPTIONS",
                   fmt::format("parse_mls_options: method '{}' does not start with 'mls'",
                               method)};
  }
  std::string_view rest = method.substr(kPrefix.size());
  if (rest.empty()) {
    return Failure{"E-MLS-OPTIONS",
                   "method 'mls' declares no options; degree=, k= and radial= are all "
                   "required and never defaulted, e.g. "
                   "'mls:degree=2,k=12,radial=wendland2'"};
  }
  if (rest.front() != ':') {
    return Failure{"E-MLS-OPTIONS",
                   fmt::format("method '{}' is not 'mls' followed by "
                              "':degree=...,k=...,radial=...'",
                              method)};
  }
  rest.remove_prefix(1);
  if (rest.empty()) {
    return Failure{"E-MLS-OPTIONS",
                   fmt::format("method '{}' has an empty option list after ':'", method)};
  }

  MlsOptions opt;
  bool have_degree = false;
  bool have_k = false;
  bool have_radial = false;

  std::size_t pos = 0;
  while (pos <= rest.size()) {
    const std::size_t comma = rest.find(',', pos);
    const std::string_view token =
        (comma == std::string_view::npos) ? rest.substr(pos) : rest.substr(pos, comma - pos);
    if (token.empty()) {
      return Failure{"E-MLS-OPTIONS",
                     fmt::format("method '{}' has an empty option between commas", method)};
    }
    const std::size_t eq = token.find('=');
    if (eq == std::string_view::npos || eq == 0 || eq + 1 == token.size()) {
      return Failure{"E-MLS-OPTIONS",
                     fmt::format("method '{}' has option '{}' without a 'key=value' shape",
                                 method, token)};
    }
    const std::string_view key = token.substr(0, eq);
    const std::string_view val = token.substr(eq + 1);

    // Manual integer parse: `std::from_chars` on `string_view` is available,
    // but a plain digit scan keeps the failure mode (a non-digit anywhere)
    // obvious without pulling in <charconv>'s error-code plumbing for two
    // one-off integers.
    const auto parse_uint = [](std::string_view v) -> Result<long long> {
      if (v.empty()) return Failure{"E-MLS-OPTIONS", "empty integer"};
      for (const char c : v) {
        if (c < '0' || c > '9') return Failure{"E-MLS-OPTIONS", "not an integer"};
      }
      long long value = 0;
      for (const char c : v) value = value * 10 + (c - '0');
      return value;
    };

    if (key == "degree") {
      if (have_degree) {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' declares 'degree' more than once", method)};
      }
      const auto n = parse_uint(val);
      if (!n.ok()) {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' has a non-integer degree '{}'", method, val)};
      }
      opt.degree = static_cast<int>(n.value());
      have_degree = true;
    } else if (key == "k") {
      if (have_k) {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' declares 'k' more than once", method)};
      }
      const auto n = parse_uint(val);
      if (!n.ok() || n.value() < 1) {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' has a non-positive or non-integer k '{}'",
                                   method, val)};
      }
      opt.k = static_cast<int>(n.value());
      have_k = true;
    } else if (key == "radial") {
      if (have_radial) {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' declares 'radial' more than once", method)};
      }
      if (val == "wendland0") {
        opt.radial = MlsRadial::Wendland0;
      } else if (val == "wendland2") {
        opt.radial = MlsRadial::Wendland2;
      } else if (val == "wendland4") {
        opt.radial = MlsRadial::Wendland4;
      } else if (val == "wendland6") {
        opt.radial = MlsRadial::Wendland6;
      } else {
        return Failure{"E-MLS-OPTIONS",
                       fmt::format("method '{}' names unknown radial weight '{}'; supported: "
                                   "wendland0, wendland2, wendland4, wendland6",
                                   method, val)};
      }
      have_radial = true;
    } else {
      return Failure{"E-MLS-OPTIONS",
                     fmt::format("method '{}' names unknown option '{}'", method, key)};
    }

    if (comma == std::string_view::npos) break;
    pos = comma + 1;
  }

  if (!have_degree || !have_k || !have_radial) {
    std::string missing;
    if (!have_degree) missing += "degree ";
    if (!have_k) missing += "k ";
    if (!have_radial) missing += "radial ";
    return Failure{"E-MLS-OPTIONS",
                   fmt::format("method '{}' is missing required option(s): {}-- degree, k "
                               "and radial are never defaulted",
                               method, missing)};
  }
  return opt;
}

Result<RemapWeights> build_mls_weights(const RemapPlan& plan, const CouplingGraph& graph,
                                       const MeshBackend& backend) {
  const auto options_r = parse_mls_options(plan.method);
  if (!options_r.ok()) return options_r.error();
  const MlsOptions& opt = options_r.value();

  if (has(plan.flags, RemapFlags::Conservative)) {
    return Failure{
        "E-MLS-UNSUPPORTED",
        "method 'mls' was asked for with RemapFlags::Conservative: moving least squares "
        "fits a local smooth polynomial and makes no conservation claim -- a point cloud "
        "need not even define MeshView::measure. Drop RemapFlags::Conservative, or use a "
        "method that builds a mass-conserving map (the supermesh or TempestRemap "
        "builders)."};
  }

  if (const auto* native = dynamic_cast<const native::NativeMeshBackend*>(&backend);
      native != nullptr && native->num_ranks() > 1) {
    return Failure{"E-MLS-UNSUPPORTED",
                   fmt::format("method 'mls' was asked for on a {}-rank backend: this "
                               "builder searches neighbours only among the entities this "
                               "rank's MeshView reports, with no halo, so a target near a "
                               "partition boundary could silently miss its true nearest "
                               "source points. Distributed MLS is a follow-on "
                               "(docs/user/point-cloud-remap.md); run this method at one "
                               "rank",
                               native->num_ranks())};
  }

  const auto src_r =
      resolve_mls_side(graph, backend, plan.src, plan.placement, plan.src_disc, plan.src_order,
                       "source");
  if (!src_r.ok()) return src_r.error();
  const auto tgt_r =
      resolve_mls_side(graph, backend, plan.tgt, plan.placement, plan.tgt_disc, plan.tgt_order,
                       "target");
  if (!tgt_r.ok()) return tgt_r.error();
  const MlsSide& src = src_r.value();
  const MlsSide& tgt = tgt_r.value();

  const std::vector<search::Point> src_pts = gather_points(*src.view, src.dim);
  const std::vector<search::Point> tgt_pts = gather_points(*tgt.view, tgt.dim);
  const std::size_t n_cols = src_pts.size();
  const std::size_t n_rows = tgt_pts.size();

  RemapWeights out;
  out.n_rows = n_rows;
  out.n_cols = n_cols;
  out.order = opt.degree + 1;
  // No conservation claim at all (this header's own "WHAT THE AUDIT FIELDS
  // MEAN" section): +inf reads as failing WeightsOperator::claims()'s
  // `<= kClaimTolerance` gate, rather than the struct default of 0.0, which
  // would silently PASS it.
  out.max_col_defect = std::numeric_limits<double>::infinity();
  out.max_col_defect_native = std::numeric_limits<double>::infinity();

  if (n_rows == 0) {
    // Vacuously monotone (no row to violate the bound) and exact (no row to
    // fail the reproduction property) -- the same "empty is a legitimate
    // result" posture `build_remap_halo` documents for zero adopted cells.
    out.monotone = true;
    return out;
  }

  // `classify_dim` (fuse::search::geometry.hpp) returns 1, 2 or 3 by
  // contract for any non-empty box, never anything else, so there is no
  // "unsupported dimension" branch to check here (and none to test: this
  // file names no rejection it cannot make fire).
  const int dim = classify_point_dim(src_pts, tgt_pts);

  const std::size_t basis_size = mls_basis_size(dim, opt.degree);
  if (static_cast<std::size_t>(opt.k) < basis_size) {
    return Failure{
        "E-MLS-NEIGHBORS",
        fmt::format("method 'mls' declares k={} neighbours, but a degree-{} polynomial in "
                    "{}D needs at least {} ({} monomials of total degree <= {}) to be "
                    "determined",
                    opt.k, opt.degree, dim, basis_size, basis_size, opt.degree)};
  }
  if (n_cols < static_cast<std::size_t>(opt.k)) {
    return Failure{"E-MLS-NEIGHBORS",
                   fmt::format("method 'mls' declares k={} neighbours, but the source side "
                               "has only {} point(s)",
                               opt.k, n_cols)};
  }

  // Which Kokkos execution
  // space the k-nearest search and the per-target solve run on. No explicit
  // argument on `RemapPlan` or `MlsOptions` for this -- `RemapPlan::key()`'s
  // own contract is that every field of the struct appears in the cache key
  // (remap_plan.hpp), and this builder's dispatch is reachable directly in
  // a unit test with no Runner and no RemapPlan field to set, exactly the
  // shape `fuse::search::resolve_execution_space()` (types.hpp) already
  // exists for: the SAME `FUSE_EXECUTION_SPACE` environment variable
  // `RunnerConfig::resolved_execution_space()` reads, or an explicit
  // `fuse::search::ExecutionSpace` argument, with no dependency on
  // `fuse::runtime` (this builder already depends on `fuse::search` for the
  // k-nearest search itself, so reusing its resolver adds no new edge).
  // This is the SAME convention `test/CMakeLists.txt`'s `fuse_add_device_twin`
  // exercises for every other device-mode row: re-running a binary with
  // `FUSE_EXECUTION_SPACE=device` in its environment, trusting the function
  // under test to read it. A Runner that sets `RunnerConfig::execution_space`
  // explicitly in code (bypassing the environment variable) is NOT seen by
  // this resolution -- a known gap: closing it needs either exporting the
  // Runner's own resolved space into the environment before a plan is
  // built, or threading an explicit space through `RemapPlan` (touching
  // its cache key).
  const auto space_r = search::resolve_execution_space();
  if (!space_r.ok()) return space_r.error();
  const search::ExecutionSpace space = space_r.value();

  // One degenerate (zero-size) box per source point: BoxIndex is a box
  // index, and a point IS the degenerate box `{p, p}`. No `padded()` step --
  // that tolerance exists so a point sitting on a shared cell FACE is found
  // in both adjacent boxes, which has no meaning for a bare point cloud.
  std::vector<search::Box> boxes;
  boxes.reserve(n_cols);
  for (const search::Point& p : src_pts) boxes.push_back(search::Box{p, p});
  auto index_r = search::BoxIndex::build(std::move(boxes), space);
  if (!index_r.ok()) return index_r.error();
  const search::BoxIndex& index = index_r.value();

  // The k-nearest search itself: `query_nearest`'s std::vector overload
  // builds and queries on `index`'s own resolved space (search.md, "Inputs
  // and outputs") -- a device-resolved index runs the broad-phase box
  // traversal and the k-nearest distance comparisons as Kokkos device
  // kernels, copying only the thin host<->device boundary at this call,
  // the same boundary every vector-based caller of this facade crosses.
  auto knn_r = index.query_nearest(tgt_pts, opt.k);
  if (!knn_r.ok()) return knn_r.error();
  const search::QueryResult& knn = knn_r.value();

  const std::size_t p_size = basis_size;

  if (space == search::ExecutionSpace::Device) {
    auto dev_r = build_mls_weights_device(dim, opt, p_size, src_pts, tgt_pts, knn);
    if (!dev_r.ok()) return dev_r.error();
    out = dev_r.value();
    if (has(plan.flags, RemapFlags::Monotone) && !out.monotone) {
      return Failure{
          "E-MLS-MONOTONE",
          fmt::format("RemapFlags::Monotone was asked for method 'mls' degree={}, but the "
                      "assembled matrix is not bound preserving: min weight {:.3e}, max row "
                      "defect {:.3e}. Bound preservation is generic for degree 0 (a "
                      "normalised weighted average) and is not guaranteed for degree >= 1 -- "
                      "drop RemapFlags::Monotone, or declare degree=0",
                      opt.degree, out.min_weight, out.max_row_defect)};
    }
    return out;
  }

  // ---- Host path ------------------------------------------------------------
  out.row.reserve(n_rows * static_cast<std::size_t>(opt.k));
  out.col.reserve(n_rows * static_cast<std::size_t>(opt.k));
  out.val.reserve(n_rows * static_cast<std::size_t>(opt.k));

  double max_row_defect = 0.0;
  double min_weight = std::numeric_limits<double>::infinity();

  for (std::size_t t = 0; t < n_rows; ++t) {
    const std::size_t begin = knn.offsets[t];
    const std::size_t end = knn.offsets[t + 1];
    if (end - begin != static_cast<std::size_t>(opt.k)) {
      // Cannot happen once n_cols >= k has been checked above (query_nearest
      // caps k only to `size()`, and size() == n_cols here) -- checked
      // anyway rather than assumed, per this file's own "coincident points
      // must not crash" requirement: a defensive, named rejection is cheaper
      // than a silently short row.
      return Failure{"E-MLS-NEIGHBORS",
                     fmt::format("target {} got {} neighbour(s), fewer than the declared "
                                 "k={}",
                                 t, end - begin, opt.k)};
    }

    std::vector<std::size_t> nbr_idx(static_cast<std::size_t>(opt.k));
    for (std::size_t j = begin; j < end; ++j) nbr_idx[j - begin] = knn.indices[j];

    auto fit_r = mls_fit_one_target(dim, opt.degree, opt.radial, p_size, t, src_pts, tgt_pts[t],
                                    nbr_idx);
    if (!fit_r.ok()) return fit_r.error();
    const MlsTargetFit& fit = fit_r.value();

    double row_sum = 0.0;
    for (std::size_t local = 0; local < nbr_idx.size(); ++local) {
      double dotc = 0.0;
      for (std::size_t a = 0; a < p_size; ++a) dotc += fit.coeff[a] * fit.basis_rows[local][a];
      const double w = fit.phi[local] * dotc;
      out.row.push_back(static_cast<std::int32_t>(t));
      out.col.push_back(static_cast<std::int32_t>(nbr_idx[local]));
      out.val.push_back(w);
      row_sum += w;
      min_weight = std::min(min_weight, w);
    }
    max_row_defect = std::max(max_row_defect, std::fabs(row_sum - 1.0));
  }

  out.max_row_defect = max_row_defect;
  out.min_weight = out.val.empty() ? 0.0 : min_weight;
  out.monotone = (out.min_weight >= -1e-12) && (max_row_defect <= 1e-12);

  if (has(plan.flags, RemapFlags::Monotone) && !out.monotone) {
    return Failure{
        "E-MLS-MONOTONE",
        fmt::format("RemapFlags::Monotone was asked for method 'mls' degree={}, but the "
                    "assembled matrix is not bound preserving: min weight {:.3e}, max row "
                    "defect {:.3e}. Bound preservation is generic for degree 0 (a "
                    "normalised weighted average) and is not guaranteed for degree >= 1 -- "
                    "drop RemapFlags::Monotone, or declare degree=0",
                    opt.degree, out.min_weight, max_row_defect)};
  }

  return out;
}

}  // namespace fuse
