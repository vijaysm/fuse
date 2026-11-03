// The derivative gate -- include/fuse/linalg/derivative_gate.hpp's own
// file header has the full account of what each of the three checks below
// verifies and why its bound is derived rather than chosen. This file is the
// ctest row every operator/residual case in the project runs through:
// `linalg.derivative_gate`.
//
// WHY ONE FILE FOR THREE DIFFERENT OPERATOR FAMILIES. The adjoint check
// (requirement 1) is run over every operator family this project builds with
// an apply and a transpose (or a self-adjoint apply doing duty for both):
// `fuse::linalg::DistSparseMatrix` at one and two in-process partitions
// (dist_sparse_matrix_test.cpp's own FakePeerExchanger technique, duplicated
// here rather than shared -- see that file's own header comment for why),
// `fuse::WeightsOperator`/`fuse::ComposedOperator` over hand-built matrices
// and over REAL supermesh/consistent-interpolation/moving-least-squares
// weights (operator_conformance_test.cpp's own technique: a `SerialMeshBackend`
// loaded with "generated:line:n" meshes, and mls_weights_test.cpp's own
// `FlatPointView`/`TestBackend`/`Fixture`, both duplicated here for the same
// reason), and `fuse::linalg::SchurSlabSolver` at one rank and at two
// in-process partitions (schur_slab_test.cpp's own `RingAllgatherReducer`/
// `two_pass_collective` technique, duplicated here too). Every one of these
// reduces, through its own small adapter below, to the SAME
// `fuse::linalg::HostApplyFn` shape derivative_gate.hpp works at.
//
// "nodal weights" (the fourth backend case):
// `fuse::build_consistent_interp_weights` (consistent_interp.hpp) builds `H`,
// whose own file header says plainly "H interpolates a field whose DOFs are
// the SOURCE mesh's VERTICES" -- i.e. nodal DOFs. The distributed nodal-remap
// primitives in `nodal_remap_distributed.hpp` (`apply_distributed_H`/
// `apply_distributed_HT`) are a DIFFERENT code path: they operate directly
// over MPI on distributed arrays, never producing a `RemapWeights` or a
// `fuse::LinearOperator` this gate's apply/apply_transpose shape applies to.
// This file's "consistent" case covers the nodal-DOF interpolation property;
// the MPI-distributed primitive is not separately adapted here.
//
// WHAT THE OTHER TWO CHECKS USE. The finite-difference convergence check
// (requirement 2) needs a residual and a claimed Jacobian action, not an
// operator with an apply_transpose, so it is run against three hand-written
// (x, v) -> (F, Jv) pairs below: a cubic map with its own exact derivative
// (and, deliberately, a WRONG one, to prove the check can fail), a
// Brusselator species reaction-diffusion residual built directly from
// brusselator_physics.hpp's own `periodic_laplacian` and the reaction term
// `u^2 v - A u`, and a backward-Euler-shaped AFFINE residual
// (same `periodic_laplacian`, no reaction term) whose remainder is
// mathematically zero at every h -- the "dn_heat-shaped" case,
// built directly rather than through examples/dn_heat.
//
// The assembled-vs-matrix-free check (requirement 3) compares a
// `DistSparseMatrix` assembled from the Brusselator diffusion operator's own
// triplet structure (species_core.hpp's `KrylovDiffusionSolver::factor`)
// against TWO independent references: `brusselator::periodic_laplacian`
// itself (an `Eigen::SparseMatrix` product, one rank) and a hand-written
// matrix-free periodic 5-point stencil kernel implementing the identical
// coefficients directly on arrays. NOTE: brusselator_physics.hpp's only
// Kokkos kernel (`reaction_kokkos_span`) computes the pointwise REACTION
// term, not diffusion -- there is no existing Kokkos diffusion stencil in
// the codebase for this check to call, so the "matrix-free kernel" here is a
// plain hand-written function implementing the same stencil arithmetic
// `periodic_laplacian`/`KrylovDiffusionSolver::factor` both already use
// (a deviation from using the real kernel).

#include "fuse/linalg/derivative_gate.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/consistent_interp.hpp"
#include "fuse/backend/linear_operator.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/mls_weights.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/backend/supermesh.hpp"
#include "fuse/backend/weights_operator.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/schur_slab.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "brusselator_physics.hpp"

namespace {

using fuse::Failure;
using fuse::LinearOperator;
using fuse::Result;
using fuse::linalg::adjoint_gate;
using fuse::linalg::AdjointGateResult;
using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::estimate_operator_norm;
using fuse::linalg::GhostExchanger;
using fuse::linalg::GhostPattern;
using fuse::linalg::HostApplyFn;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::matrix_free_agreement;
using fuse::linalg::Reducer;
using fuse::linalg::RowOwnership;
using fuse::linalg::SchurSlabSolver;
using fuse::linalg::taylor_remainder_order;
using fuse::linalg::Triplet;

// ===========================================================================
// Adapters: every operator family this gate runs against, reduced to
// fuse::linalg::HostApplyFn.
// ===========================================================================

// ---- fuse::LinearOperator (backend/linear_operator.hpp), span-shaped ------

HostApplyFn span_apply(std::shared_ptr<const LinearOperator> op) {
  return [op](std::span<const double> x, std::span<double> y) -> Result<void> {
    return op->apply(x, y, 1);
  };
}
HostApplyFn span_apply_transpose(std::shared_ptr<const LinearOperator> op) {
  return [op](std::span<const double> y, std::span<double> x) -> Result<void> {
    return op->apply_transpose(y, x, 1);
  };
}

// ---- DistSparseMatrix, one in-process partition ---------------------------

HostApplyFn serial_matrix_apply(std::shared_ptr<DistSparseMatrix> m, std::size_t n) {
  return [m, n](std::span<const double> x, std::span<double> y) -> Result<void> {
    DistVector dx(n, make_serial_exchanger(), make_serial_reducer());
    std::copy(x.begin(), x.end(), dx.data());
    DistVector dy(n, make_serial_exchanger(), make_serial_reducer());
    auto r = m->apply(dx, dy);
    if (!r.ok()) return r;
    std::copy(dy.data(), dy.data() + n, y.begin());
    return Result<void>::success();
  };
}

// ---- DistSparseMatrix, two in-process partitions ---------------------------
// Same device dist_sparse_matrix_test.cpp uses, duplicated rather than
// shared through a new header.

class FakePeerExchanger : public GhostExchanger {
 public:
  explicit FakePeerExchanger(GhostPattern pattern) : pattern_(std::move(pattern)) {}
  void bind_peers(std::array<const DistVector*, 2> peers) { peers_ = peers; }
  [[nodiscard]] std::size_t ghost_size() const override { return pattern_.owner_rank.size(); }
  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const override {
    (void)owned;
    (void)owned_n;
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern_.owner_rank[g]);
      ghosts[g] = peers_[owner]->data()[pattern_.remote_local_index[g]];
    }
    return Result<void>::success();
  }

 private:
  GhostPattern pattern_;
  std::array<const DistVector*, 2> peers_{nullptr, nullptr};
};

std::vector<Triplet> rows_in(const std::vector<Triplet>& t, std::int64_t lo, std::int64_t hi) {
  std::vector<Triplet> out;
  for (const Triplet& e : t) {
    if (e.row >= lo && e.row < hi) out.push_back(e);
  }
  return out;
}

struct TwoPartitionMats {
  DistSparseMatrix m0, m1;
  std::shared_ptr<FakePeerExchanger> exch0, exch1;
  std::int64_t split = 0, global_n = 0;
};

std::shared_ptr<TwoPartitionMats> build_two_partition(const std::vector<Triplet>& triplets,
                                                       std::int64_t global_n,
                                                       std::int64_t split) {
  const RowOwnership owners{{0, split}};
  auto m0 =
      DistSparseMatrix::from_triplets(0, split, global_n, rows_in(triplets, 0, split),
                                      make_serial_reducer(), owners);
  auto m1 = DistSparseMatrix::from_triplets(split, global_n, global_n,
                                            rows_in(triplets, split, global_n),
                                            make_serial_reducer(), owners);
  if (!m0.ok() || !m1.ok()) return nullptr;
  // DistSparseMatrix's default constructor is private (from_triplets is the
  // only public way to get one), so TwoPartitionMats -- an aggregate -- is
  // built in ONE braced aggregate-init call rather than default-constructed
  // and assigned into. Braces, not make_shared's parentheses: nvcc's host
  // pass does not implement parenthesized aggregate init (C++20 P0960), the
  // same gap MoabMeshBackend's MeshLoader construction works around.
  DistSparseMatrix m0v = std::move(m0).value();
  DistSparseMatrix m1v = std::move(m1).value();
  auto exch0 = std::make_shared<FakePeerExchanger>(m0v.ghost_pattern());
  auto exch1 = std::make_shared<FakePeerExchanger>(m1v.ghost_pattern());
  return std::shared_ptr<TwoPartitionMats>(new TwoPartitionMats{
      std::move(m0v), std::move(m1v), std::move(exch0), std::move(exch1), split, global_n});
}

HostApplyFn two_partition_apply(std::shared_ptr<TwoPartitionMats> tp) {
  return [tp](std::span<const double> x, std::span<double> y) -> Result<void> {
    const auto n0 = static_cast<std::size_t>(tp->split);
    const auto n1 = static_cast<std::size_t>(tp->global_n - tp->split);
    DistVector x0(n0, tp->exch0, make_serial_reducer());
    DistVector x1(n1, tp->exch1, make_serial_reducer());
    tp->exch0->bind_peers({nullptr, &x1});
    tp->exch1->bind_peers({&x0, nullptr});
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n0), x0.data());
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(n0), x.end(), x1.data());
    DistVector y0(n0, make_serial_exchanger(), make_serial_reducer());
    DistVector y1(n1, make_serial_exchanger(), make_serial_reducer());
    if (Result<void> r = tp->m0.apply(x0, y0); !r.ok()) return r;
    if (Result<void> r = tp->m1.apply(x1, y1); !r.ok()) return r;
    std::copy(y0.data(), y0.data() + n0, y.begin());
    std::copy(y1.data(), y1.data() + n1, y.begin() + static_cast<std::ptrdiff_t>(n0));
    return Result<void>::success();
  };
}

/// `A` built from `triplets` (global, canonical row/col numbering) at one
/// partition; `A^T` built the same way from the row/col-swapped triplets --
/// NOT `A`'s own `apply` reused in reverse (that is exactly the bug the
/// adjoint identity exists to catch), a genuinely separate assembly that
/// exercises `from_triplets`' column-split/ghost-pattern derivation on the
/// transposed structure.
std::vector<Triplet> transpose_of(const std::vector<Triplet>& t) {
  std::vector<Triplet> out;
  out.reserve(t.size());
  for (const Triplet& e : t) out.push_back({e.col, e.row, e.value});
  return out;
}

// ---- SchurSlabSolver (its own `solve()` as the operator A^-1) -------------
// `A = I - coefficient * L_h`, L_h the periodic 5-point Laplacian -- SYMMETRIC,
// so A^-1 is symmetric too: `solve()` stands in for both apply and
// apply_transpose. one-rank and two-in-process-partition variants, the
// latter via schur_slab_test.cpp's own RingAllgatherReducer/
// two_pass_collective technique (duplicated rather than shared through a
// new header).

std::pair<std::int64_t, std::int64_t> row_band(std::int64_t ny, int rank, int num_ranks) {
  if (num_ranks <= 1) return {0, ny};
  const std::int64_t base = ny / num_ranks;
  const std::int64_t extra = ny % num_ranks;
  const std::int64_t r = rank;
  const std::int64_t lo = (r < extra) ? r * (base + 1) : extra * (base + 1) + (r - extra) * base;
  const std::int64_t size = base + (r < extra ? 1 : 0);
  return {lo, lo + size};
}

RowOwnership schur_row_ownership(std::int64_t ny, int num_ranks) {
  RowOwnership owners;
  owners.row_lo.resize(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    owners.row_lo[static_cast<std::size_t>(r)] = row_band(ny, r, num_ranks).first;
  }
  return owners;
}

class RingAllgatherReducer : public Reducer {
 public:
  using Cache = std::map<std::size_t, std::vector<std::vector<double>>>;
  RingAllgatherReducer(int rank, int size, std::shared_ptr<Cache> caches)
      : rank_(rank), size_(size), caches_(std::move(caches)) {}
  [[nodiscard]] double sum(double local) const override { return local; }
  void sum(double* inout, std::size_t n) const override { (void)inout; (void)n; }
  [[nodiscard]] double max(double local) const override { return local; }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return size_; }
  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    auto& cache = (*caches_)[n_each];
    if (cache.empty()) cache.resize(static_cast<std::size_t>(size_));
    cache[static_cast<std::size_t>(rank_)].assign(send, send + n_each);
    for (int r = 0; r < size_; ++r) {
      const auto& slot = cache[static_cast<std::size_t>(r)];
      if (slot.size() == n_each) {
        std::copy(slot.begin(), slot.end(), recv + static_cast<std::size_t>(r) * n_each);
      } else {
        std::fill(recv + static_cast<std::size_t>(r) * n_each,
                  recv + static_cast<std::size_t>(r + 1) * n_each, 0.0);
      }
    }
  }

 private:
  int rank_;
  int size_;
  std::shared_ptr<Cache> caches_;
};

template <class F>
auto two_pass_collective(int nranks, F&& make_call) -> std::vector<std::invoke_result_t<F, int>> {
  for (int r = 0; r < nranks; ++r) (void)make_call(r);
  std::vector<std::invoke_result_t<F, int>> out;
  out.reserve(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) out.push_back(make_call(r));
  return out;
}

std::vector<std::shared_ptr<RingAllgatherReducer>> make_ring_reducers(int nranks) {
  auto caches = std::make_shared<RingAllgatherReducer::Cache>();
  std::vector<std::shared_ptr<RingAllgatherReducer>> reducers;
  reducers.reserve(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) {
    reducers.push_back(std::make_shared<RingAllgatherReducer>(r, nranks, caches));
  }
  return reducers;
}

struct SchurOperator {
  int nx = 0;
  std::int64_t ny = 0;
  int nranks = 0;
  double coefficient = 0.0;
  RowOwnership owners;
  std::vector<std::shared_ptr<RingAllgatherReducer>> reducers;
  std::vector<SchurSlabSolver> solvers;
};

std::shared_ptr<SchurOperator> build_schur_operator(int nx, std::int64_t ny, int nranks,
                                                    double coefficient) {
  auto out = std::make_shared<SchurOperator>();
  out->nx = nx;
  out->ny = ny;
  out->nranks = nranks;
  out->coefficient = coefficient;
  out->owners = schur_row_ownership(ny, nranks);
  out->reducers = make_ring_reducers(nranks);
  auto built = two_pass_collective(nranks, [&](int r) {
    const auto [lo, hi] = row_band(ny, r, nranks);
    return SchurSlabSolver::factor(nx, ny, lo, hi, coefficient, out->owners,
                                   out->reducers[static_cast<std::size_t>(r)]);
  });
  out->solvers.reserve(static_cast<std::size_t>(nranks));
  for (auto& s : built) {
    if (!s.ok()) return nullptr;
    out->solvers.push_back(std::move(s).value());
  }
  return out;
}

HostApplyFn schur_apply(std::shared_ptr<SchurOperator> op) {
  return [op](std::span<const double> x, std::span<double> y) -> Result<void> {
    const std::int64_t ny = op->ny;
    const int nranks = op->nranks;
    const int nx = op->nx;
    std::vector<DistVector> rhs;
    rhs.reserve(static_cast<std::size_t>(nranks));
    for (int r = 0; r < nranks; ++r) {
      const auto [lo, hi] = row_band(ny, r, nranks);
      const auto n = static_cast<std::size_t>((hi - lo) * nx);
      rhs.emplace_back(n, make_serial_exchanger(), make_serial_reducer());
      std::copy(x.begin() + lo * nx, x.begin() + hi * nx, rhs.back().data());
    }
    bool any_failed = false;
    Failure first_failure;
    auto solved = two_pass_collective(nranks, [&](int r) {
      const auto [lo, hi] = row_band(ny, r, nranks);
      DistVector xr(static_cast<std::size_t>((hi - lo) * nx), make_serial_exchanger(),
                   make_serial_reducer());
      auto res = op->solvers[static_cast<std::size_t>(r)].solve(rhs[static_cast<std::size_t>(r)], xr);
      if (!res.ok() && !any_failed) {
        any_failed = true;
        first_failure = res.error();
      }
      return xr;
    });
    if (any_failed) return first_failure;
    for (int r = 0; r < nranks; ++r) {
      const auto [lo, hi] = row_band(ny, r, nranks);
      std::copy(solved[static_cast<std::size_t>(r)].data(),
                solved[static_cast<std::size_t>(r)].data() + (hi - lo) * nx,
                y.begin() + lo * nx);
    }
    return Result<void>::success();
  };
}

// ---- fuse::WeightsOperator over REAL RemapWeights builders ----------------
// Three sources: a real consistent-interpolation H (SerialMeshBackend,
// "generated:line:n" meshes -- operator_conformance_test.cpp's own
// technique), a real supermesh map (supermesh_test.cpp's own `pair_of`
// technique), and a real moving-least-squares map (mls_weights_test.cpp's
// own `FlatPointView`/`TestBackend`/`Fixture` technique, trimmed to the one
// 1D case this file needs). All three are duplicated rather than shared
// through a new header, consistently with every other adapter in this file.

struct LineMeshPair {
  fuse::SerialMeshBackend backend;
  const fuse::MeshView* src = nullptr;
  const fuse::MeshView* tgt = nullptr;
};

std::unique_ptr<LineMeshPair> line_mesh_pair(const std::string& s, const std::string& t) {
  auto p = std::make_unique<LineMeshPair>();
  if (!p->backend.load({.name = "s", .source = s}).ok()) return nullptr;
  if (!p->backend.load({.name = "t", .source = t}).ok()) return nullptr;
  const auto sid = p->backend.mesh_id("s");
  const auto tid = p->backend.mesh_id("t");
  if (!sid.ok() || !tid.ok()) return nullptr;
  const auto sv = p->backend.view(sid.value());
  const auto tv = p->backend.view(tid.value());
  if (!sv.ok() || !tv.ok()) return nullptr;
  p->src = sv.value();
  p->tgt = tv.value();
  return p;
}

class FlatPointView final : public fuse::MeshView {
 public:
  FlatPointView(int dim, std::vector<fuse::Point> pts) : dim_(dim), pts_(std::move(pts)) {}
  int dimension() const override { return dim_; }
  std::size_t entity_count(int dim) const override { return dim == dim_ ? pts_.size() : 0; }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int dim, std::size_t i) const override {
    check(dim, i);
    return true;
  }
  fuse::Point centroid(int dim, std::size_t i) const override {
    check(dim, i);
    return pts_[i];
  }
  double measure(int, std::size_t) const override {
    throw std::out_of_range("FlatPointView::measure: this view defines no measure");
  }
  std::size_t global_id(int dim, std::size_t i) const override {
    check(dim, i);
    return i;
  }
  std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, fuse::Point*) const override {}

 private:
  void check(int dim, std::size_t i) const {
    if (dim != dim_ || i >= pts_.size()) throw std::out_of_range("FlatPointView: out of range");
  }
  int dim_;
  std::vector<fuse::Point> pts_;
};

class MlsTestBackend final : public fuse::MeshBackend {
 public:
  fuse::MeshId add(std::string name, int dim, std::vector<fuse::Point> pts) {
    names_.push_back(std::move(name));
    views_.push_back(std::make_unique<FlatPointView>(dim, std::move(pts)));
    return fuse::MeshId{static_cast<std::uint32_t>(views_.size() - 1)};
  }
  Result<void> load(const fuse::MeshDecl&) override { return Result<void>::success(); }
  Result<fuse::MeshId> mesh_id(std::string_view n) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == n) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return Failure{"E-MESH-NAME", "MlsTestBackend: no such mesh"};
  }
  Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= views_.size()) {
      return Failure{"E-MESH-ID", "MlsTestBackend: bad mesh id"};
    }
    return views_[id.value()].get();
  }
  Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  std::string name() const override { return "test-flat-points"; }

 private:
  std::vector<std::string> names_;
  std::vector<std::unique_ptr<fuse::MeshView>> views_;
};

std::vector<fuse::Point> grid_1d(int n, double lo, double hi) {
  std::vector<fuse::Point> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double t = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
    pts.push_back(fuse::Point{lo + t * (hi - lo), 0.0, 0.0});
  }
  return pts;
}

/// One real MLS RemapWeights: degree-1, 1D, 12 source points over [0,1], 6
/// target points over [0.1, 0.9]. `std::nullopt` on any construction
/// failure, so a caller can skip the case rather than crash (the same
/// posture operator_conformance_test.cpp's own `operators()` uses for its
/// consistent-interpolation case).
std::optional<fuse::RemapWeights> real_mls_weights() {
  MlsTestBackend backend;
  fuse::CouplingGraph graph;
  const auto src_pts = grid_1d(12, 0.0, 1.0);
  const auto tgt_pts = grid_1d(6, 0.1, 0.9);
  backend.add("src", 0, src_pts);
  const auto src_mesh = graph.add_mesh(fuse::MeshDecl{.name = "src", .source = "test"});
  const auto src_region = graph.add_region(
      fuse::Region{.name = "src_region", .mesh = src_mesh, .entity_set = "", .dimension = 0});
  backend.add("tgt", 0, tgt_pts);
  const auto tgt_mesh = graph.add_mesh(fuse::MeshDecl{.name = "tgt", .source = "test"});
  const auto tgt_region = graph.add_region(
      fuse::Region{.name = "tgt_region", .mesh = tgt_mesh, .entity_set = "", .dimension = 0});

  fuse::RemapPlan plan;
  plan.src = src_region;
  plan.tgt = tgt_region;
  plan.src_disc = fuse::Discretization::PointCloud;
  plan.tgt_disc = fuse::Discretization::PointCloud;
  plan.method = "mls:degree=1,k=6,radial=wendland2";

  auto w = fuse::build_mls_weights(plan, graph, backend);
  if (!w.ok()) return std::nullopt;
  return std::move(w).value();
}

// ===========================================================================
// The adjoint-gate case list.
// ===========================================================================

struct AdjointCase {
  std::string name;
  std::size_t rows = 0, cols = 0;
  HostApplyFn apply;
  HostApplyFn apply_transpose;
};

/// Keeps the RemapWeights/operator objects an AdjointCase's closures capture
/// by shared_ptr alive for the whole test binary -- `WeightsOperator` does
/// not copy its weights (weights_operator.hpp's own header comment), so the
/// RemapWeights must outlive every operator built over it.
struct AdjointFixtures {
  std::vector<std::shared_ptr<fuse::RemapWeights>> weights;
  std::vector<std::shared_ptr<const LinearOperator>> ops;
};

fuse::RemapWeights hand_built_rectangular() {
  fuse::RemapWeights w;
  w.n_rows = 4;
  w.n_cols = 3;
  const int rows[] = {0, 0, 1, 2, 2, 3};
  const int cols[] = {0, 1, 1, 0, 2, 2};
  const double val[] = {0.25, 0.75, 1.0, 0.4, 0.6, 1.0};
  for (int k = 0; k < 6; ++k) {
    w.row.push_back(rows[k]);
    w.col.push_back(cols[k]);
    w.val.push_back(val[k]);
  }
  w.max_row_defect = 0.0;
  w.max_col_defect = 0.35;
  w.monotone = true;
  w.order = 1;
  return w;
}

std::vector<AdjointCase> adjoint_cases(AdjointFixtures& fixtures) {
  std::vector<AdjointCase> cases;

  // -- DistSparseMatrix, one partition: periodic 16x16 Laplacian (symmetric)
  {
    constexpr int nx = 16, ny = 16;
    const auto t = brusselator::periodic_laplacian(nx, ny);
    std::vector<Triplet> tl;
    for (int col = 0; col < t.outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(t, col); it; ++it) {
        tl.push_back({it.row(), it.col(), it.value()});
      }
    }
    const std::int64_t n = nx * ny;
    const RowOwnership owners{{0, n}};
    auto m = DistSparseMatrix::from_triplets(0, n, n, tl, make_serial_reducer(), owners);
    if (m.ok()) {
      auto mp = std::make_shared<DistSparseMatrix>(std::move(m).value());
      cases.push_back({"DistSparseMatrix(periodic-laplacian-16x16, 1 partition, symmetric)",
                       static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                       serial_matrix_apply(mp, static_cast<std::size_t>(n)),
                       serial_matrix_apply(mp, static_cast<std::size_t>(n))});
    }
  }

  // -- DistSparseMatrix, two in-process partitions: same periodic Laplacian
  {
    constexpr int nx = 16, ny = 16;
    const auto t = brusselator::periodic_laplacian(nx, ny);
    std::vector<Triplet> tl;
    for (int col = 0; col < t.outerSize(); ++col) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(t, col); it; ++it) {
        tl.push_back({it.row(), it.col(), it.value()});
      }
    }
    const std::int64_t n = nx * ny;
    auto tp = build_two_partition(tl, n, n / 2);
    if (tp) {
      cases.push_back({"DistSparseMatrix(periodic-laplacian-16x16, 2 partitions, symmetric)",
                       static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                       two_partition_apply(tp), two_partition_apply(tp)});
    }
  }

  // -- DistSparseMatrix, two in-process partitions, random NONSYMMETRIC
  // (load-bearing: a transpose bug is invisible on a symmetric operator --
  // operator_conformance_test.cpp's own `rectangular()` comment states the
  // identical reasoning).
  {
    constexpr std::int64_t n = 50;
    std::mt19937 rng(41u);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    std::uniform_int_distribution<std::int64_t> col_pick(0, n - 1);
    std::vector<Triplet> tl;
    for (std::int64_t r = 0; r < n; ++r) {
      std::vector<std::int64_t> cols;
      while (cols.size() < 5) {
        const std::int64_t c = col_pick(rng);
        if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
      }
      for (std::int64_t c : cols) tl.push_back({r, c, u(rng)});
    }
    auto tp = build_two_partition(tl, n, n / 2);
    auto tpt = build_two_partition(transpose_of(tl), n, n / 2);
    if (tp && tpt) {
      cases.push_back({"DistSparseMatrix(random-nonsymmetric-50x50, 2 partitions)",
                       static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                       two_partition_apply(tp), two_partition_apply(tpt)});
    }
  }

  // -- WeightsOperator, hand-built rectangular (non-symmetric, non-square)
  {
    auto w = std::make_shared<fuse::RemapWeights>(hand_built_rectangular());
    fixtures.weights.push_back(w);
    auto op = std::make_shared<fuse::WeightsOperator>(*w);
    fixtures.ops.push_back(op);
    cases.push_back({"WeightsOperator(hand-built-4x3)", op->rows(), op->cols(), span_apply(op),
                     span_apply_transpose(op)});
  }

  // -- WeightsOperator, real consistent-interpolation H (5-node beam, 8-cell
  // fluid, generated:line meshes -- consistent_interp.hpp)
  {
    auto p = line_mesh_pair("generated:line:5", "generated:line:8");
    if (p) {
      auto h = fuse::build_consistent_interp_weights(*p->src, *p->tgt, 1);
      if (h.ok()) {
        auto w = std::make_shared<fuse::RemapWeights>(std::move(h).value());
        fixtures.weights.push_back(w);
        auto op = std::make_shared<fuse::WeightsOperator>(*w);
        fixtures.ops.push_back(op);
        cases.push_back({"WeightsOperator(consistent-interp-H, 5-node-beam/8-cell-fluid)",
                         op->rows(), op->cols(), span_apply(op), span_apply_transpose(op)});
      }
    }
  }

  // -- WeightsOperator, real supermesh map (generated:line:8 -> :5)
  {
    auto p = line_mesh_pair("generated:line:8", "generated:line:5");
    if (p) {
      auto s = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
      if (s.ok()) {
        auto w = std::make_shared<fuse::RemapWeights>(std::move(s).value());
        fixtures.weights.push_back(w);
        auto op = std::make_shared<fuse::WeightsOperator>(*w);
        fixtures.ops.push_back(op);
        cases.push_back({"WeightsOperator(supermesh, line8->line5)", op->rows(), op->cols(),
                         span_apply(op), span_apply_transpose(op)});
      }
    }
  }

  // -- WeightsOperator, real moving-least-squares map (1D, degree 1)
  {
    auto mls = real_mls_weights();
    if (mls) {
      auto w = std::make_shared<fuse::RemapWeights>(std::move(*mls));
      fixtures.weights.push_back(w);
      auto op = std::make_shared<fuse::WeightsOperator>(*w);
      fixtures.ops.push_back(op);
      cases.push_back({"WeightsOperator(mls-degree1-1d)", op->rows(), op->cols(), span_apply(op),
                       span_apply_transpose(op)});
    }
  }

  // -- ComposedOperator: the hand-built 4x3 operator (fixtures.ops[0], the
  // first operator pushed above) composed with an independently-built 3x4.
  // The composition is itself a LinearOperator, held to the same bound.
  {
    fuse::RemapWeights w2;
    w2.n_rows = 3;
    w2.n_cols = 4;
    const int rows[] = {0, 1, 1, 2, 2};
    const int cols[] = {0, 1, 2, 2, 3};
    const double val[] = {1.0, 0.5, 0.5, 0.3, 0.7};
    for (int k = 0; k < 5; ++k) {
      w2.row.push_back(rows[k]);
      w2.col.push_back(cols[k]);
      w2.val.push_back(val[k]);
    }
    w2.max_row_defect = 0.0;
    w2.max_col_defect = 0.5;
    w2.monotone = true;
    auto w2p = std::make_shared<fuse::RemapWeights>(w2);
    fixtures.weights.push_back(w2p);
    auto op2 = std::make_shared<fuse::WeightsOperator>(*w2p);
    fixtures.ops.push_back(op2);
    if (!fixtures.ops.empty()) {
      auto composed = fuse::compose(fixtures.ops[0] /* hand-built-4x3 */, op2);
      if (composed.ok()) {
        std::shared_ptr<const LinearOperator> c{std::move(composed).value()};
        fixtures.ops.push_back(c);
        cases.push_back({"ComposedOperator(hand-built-4x3 o hand-built-3x4)", c->rows(),
                         c->cols(), span_apply(c), span_apply_transpose(c)});
      }
    }
  }

  // -- SchurSlabSolver, one rank: A = I - 0.37*L_h, 8x8 grid
  {
    auto op = build_schur_operator(8, 8, 1, 0.37);
    if (op) {
      const std::size_t n = 64;
      cases.push_back({"SchurSlabSolver^-1(8x8, 1 rank, symmetric)", n, n, schur_apply(op),
                       schur_apply(op)});
    }
  }

  // -- SchurSlabSolver, two in-process partitions: A = I - 0.6*L_h, 4x8 grid
  {
    auto op = build_schur_operator(4, 8, 2, 0.6);
    if (op) {
      const std::size_t n = 32;
      cases.push_back({"SchurSlabSolver^-1(4x8, 2 partitions, symmetric)", n, n, schur_apply(op),
                       schur_apply(op)});
    }
  }

  return cases;
}

}  // namespace

// ===========================================================================
// 1. Adjoint identity.
// ===========================================================================

TEST(LinalgDerivativeGate, AdjointIdentityHoldsForEveryOperator) {
  AdjointFixtures fixtures;
  const auto cases = adjoint_cases(fixtures);
  ASSERT_GE(cases.size(), 9u) << "fewer operator cases constructed than expected -- one of the "
                                 "fixture builders failed silently";

  double worst_ratio = 0.0;
  std::string worst_name;
  for (const AdjointCase& c : cases) {
    const auto norm_r = estimate_operator_norm(c.rows, c.cols, c.apply, c.apply_transpose);
    ASSERT_TRUE(norm_r.ok()) << c.name << ": " << norm_r.error().message;
    const double norm_est = norm_r.value();
    ASSERT_GT(norm_est, 0.0) << c.name << ": a zero-norm operator is not exercised by this gate";

    for (unsigned seed : {1001u, 2002u}) {
      std::mt19937 rng(seed);
      std::uniform_real_distribution<double> u(-1.0, 1.0);
      std::vector<double> x(c.cols), y(c.rows);
      for (double& e : x) e = u(rng);
      for (double& e : y) e = u(rng);

      const auto r = adjoint_gate(c.rows, c.cols, c.apply, c.apply_transpose, x, y, norm_est,
                                  fuse::linalg::kAdjointGateConstant, c.name);
      ASSERT_TRUE(r.ok()) << c.name << ": " << r.error().message;
      EXPECT_TRUE(r.value().passed) << c.name << ": defect " << r.value().defect
                                    << " exceeds bound " << r.value().bound << " (norm_est "
                                    << norm_est << ")";
      const double ratio = r.value().bound > 0.0
                               ? r.value().defect / (r.value().bound / fuse::linalg::kAdjointGateConstant)
                               : 0.0;
      if (ratio > worst_ratio) {
        worst_ratio = ratio;
        worst_name = c.name;
      }
    }
  }
  // MEASURED, this run: the worst observed ratio of defect to
  // (eps * norm_est * ||x|| * ||y||) across every operator case above, and
  // which case produced it -- the number kAdjointGateConstant
  // (derivative_gate.hpp) must stay comfortably above.
  std::fprintf(stderr,
               "[derivative-gate] max observed adjoint ratio = %.6g (operator: %s), bound "
               "constant = %.1f\n",
               worst_ratio, worst_name.c_str(), fuse::linalg::kAdjointGateConstant);
  EXPECT_LT(worst_ratio, fuse::linalg::kAdjointGateConstant)
      << "the recorded constant no longer has headroom over the measured ratio";
}

TEST(LinalgDerivativeGate, ADeliberatelyWrongAdjointIsDetected) {
  // Control: the random-nonsymmetric DistSparseMatrix case, with its own
  // (correct) apply_transpose swapped for its own (forward) apply -- exactly
  // the "apply_transpose swapped for apply" mutation build-evidence/
  // mutations/linalg.toml names, but exercised directly here first so the
  // test methodology's own sensitivity is demonstrated, not merely assumed.
  // Shape-compatible (the matrix is square) but WRONG, since this matrix is
  // not symmetric by construction.
  constexpr std::int64_t n = 50;
  std::mt19937 rng(41u);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::uniform_int_distribution<std::int64_t> col_pick(0, n - 1);
  std::vector<Triplet> tl;
  for (std::int64_t r = 0; r < n; ++r) {
    std::vector<std::int64_t> cols;
    while (cols.size() < 5) {
      const std::int64_t c = col_pick(rng);
      if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
    }
    for (std::int64_t c : cols) tl.push_back({r, c, u(rng)});
  }
  auto tp = build_two_partition(tl, n, n / 2);
  ASSERT_TRUE(tp);
  const HostApplyFn apply = two_partition_apply(tp);
  // The mutation: apply_transpose "swapped for apply" -- literally the same
  // closure used on both sides.
  const HostApplyFn wrong_transpose = apply;

  const auto norm_r = estimate_operator_norm(static_cast<std::size_t>(n),
                                             static_cast<std::size_t>(n), apply, apply);
  ASSERT_TRUE(norm_r.ok());
  std::mt19937 rng2(7u);
  std::uniform_real_distribution<double> u2(-1.0, 1.0);
  std::vector<double> x(n), y(n);
  for (double& e : x) e = u2(rng2);
  for (double& e : y) e = u2(rng2);
  const auto r = adjoint_gate(static_cast<std::size_t>(n), static_cast<std::size_t>(n), apply,
                              wrong_transpose, x, y, norm_r.value());
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.value().passed)
      << "a mismatched (non-transpose) apply_transpose still satisfied the adjoint bound, so "
         "the check is not testing what it claims";
  EXPECT_GT(r.value().defect, 100.0 * r.value().bound)
      << "the wrong-transpose defect should be far above the bound, not marginally above it";
}

TEST(LinalgDerivativeGate, NormEstimateMatchesClosedFormForASmallSymmetricLaplacian) {
  // Validates estimate_operator_norm itself (not merely "plausible"): for a
  // small SYMMETRIC periodic Laplacian, ||A||_2 is exactly the largest
  // |eigenvalue|, computed independently by a dense eigensolve.
  constexpr int nx = 4, ny = 4;
  const auto t = brusselator::periodic_laplacian(nx, ny);
  const Eigen::MatrixXd dense = Eigen::MatrixXd(t);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(dense);
  ASSERT_EQ(es.info(), Eigen::Success);
  const double expect = es.eigenvalues().cwiseAbs().maxCoeff();

  std::vector<Triplet> tl;
  for (int col = 0; col < t.outerSize(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(t, col); it; ++it) {
      tl.push_back({it.row(), it.col(), it.value()});
    }
  }
  const std::int64_t n = nx * ny;
  const RowOwnership owners{{0, n}};
  auto m = DistSparseMatrix::from_triplets(0, n, n, tl, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok());
  auto mp = std::make_shared<DistSparseMatrix>(std::move(m).value());
  const HostApplyFn apply = serial_matrix_apply(mp, static_cast<std::size_t>(n));

  const auto got = estimate_operator_norm(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                          apply, apply, /*max_iterations=*/30);
  ASSERT_TRUE(got.ok());
  EXPECT_NEAR(got.value(), expect, 1e-6 * expect)
      << "power-iteration estimate " << got.value() << " vs. closed-form " << expect;
}

// ===========================================================================
// 2. Finite-difference derivative convergence.
// ===========================================================================

TEST(TaylorRemainderOrder, CubicMapWithAnalyticJacobianMeasuresOrderTwo) {
  // F(x)_i = x_i^3, J(x)v_i = 3 x_i^2 v_i -- exact.
  const auto F = [](const std::vector<double>& x) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * x[i] * x[i];
    return out;
  };
  const auto J = [](const std::vector<double>& x, const std::vector<double>& v) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = 3.0 * x[i] * x[i] * v[i];
    return out;
  };
  std::mt19937 rng(3u);
  std::uniform_real_distribution<double> ux(0.5, 2.0), uv(-1.0, 1.0);
  std::vector<double> x(12), v(12);
  for (double& e : x) e = ux(rng);
  for (double& e : v) e = uv(rng);

  const auto r = taylor_remainder_order(F, J, x, v, /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere) << "the correct Jacobian never reached a decreasing trend";
  EXPECT_GT(r.order, 1.8) << "observed order " << r.order << ", floor " << r.floor;
  EXPECT_LT(r.order, 2.3) << "observed order " << r.order << " suspiciously high";
}

TEST(TaylorRemainderOrder, WrongJacobianMeasuresOrderOneNotTwo) {
  // The test's own test: a Jacobian with the wrong coefficient (2x^2 instead
  // of the correct 3x^2) must be CAUGHT, not pass at order 2.
  const auto F = [](const std::vector<double>& x) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] * x[i] * x[i];
    return out;
  };
  const auto J_wrong = [](const std::vector<double>& x, const std::vector<double>& v) {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = 2.0 * x[i] * x[i] * v[i];
    return out;
  };
  std::mt19937 rng(3u);
  std::uniform_real_distribution<double> ux(0.5, 2.0), uv(-1.0, 1.0);
  std::vector<double> x(12), v(12);
  for (double& e : x) e = ux(rng);
  for (double& e : v) e = uv(rng);

  const auto r = taylor_remainder_order(F, J_wrong, x, v, /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere);
  EXPECT_GT(r.order, 0.7) << "observed order " << r.order;
  EXPECT_LT(r.order, 1.3) << "observed order " << r.order
                         << " -- a wrong Jacobian measured as order 2 would mean this check "
                            "cannot tell a correct linearisation from an incorrect one";
}

/// `D=10`, `A=3.4` -- brusselator::PhysicsConfig's own defaults
/// (species_core.hpp). `r(u) = u - u_old - dt*(D*Lap*u + u^2*v - A*u)` at a
/// frozen `v`; `J(u)v_dir = v_dir - dt*(D*Lap*v_dir + (2*u*v - A)*v_dir)`,
/// written by hand from the reaction term's own derivative.
std::pair<std::function<std::vector<double>(const std::vector<double>&)>,
         std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)>>
brusselator_species_residual(const Eigen::SparseMatrix<double>& lap, const std::vector<double>& u_old,
                             const std::vector<double>& v_frozen, double dt) {
  constexpr double D = 10.0, A = 3.4;
  const std::size_t n = u_old.size();
  auto F = [lap, u_old, v_frozen, dt, n](const std::vector<double>& u) {
    Eigen::VectorXd uvec = Eigen::Map<const Eigen::VectorXd>(u.data(), static_cast<int>(n));
    Eigen::VectorXd lap_u = lap * uvec;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double f = u[i] * u[i] * v_frozen[i] - A * u[i];
      out[i] = u[i] - u_old[i] - dt * (D * lap_u[static_cast<int>(i)] + f);
    }
    return out;
  };
  auto J = [lap, v_frozen, dt, n](const std::vector<double>& u, const std::vector<double>& v_dir) {
    Eigen::VectorXd vvec = Eigen::Map<const Eigen::VectorXd>(v_dir.data(), static_cast<int>(n));
    Eigen::VectorXd lap_v = lap * vvec;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      const double df_du = 2.0 * u[i] * v_frozen[i] - A;
      out[i] = v_dir[i] - dt * (D * lap_v[static_cast<int>(i)] + df_du * v_dir[i]);
    }
    return out;
  };
  return {F, J};
}

TEST(TaylorRemainderOrder, BrusselatorSpeciesResidualMeasuresOrderTwo) {
  constexpr int nx = 4, ny = 4;
  const auto lap = brusselator::periodic_laplacian(nx, ny);
  const std::size_t n = static_cast<std::size_t>(nx * ny);
  std::mt19937 rng(17u);
  std::uniform_real_distribution<double> uu(0.5, 3.0), uv(0.1, 2.0), udir(-1.0, 1.0);
  std::vector<double> u_old(n), v_frozen(n), u0(n), v(n);
  for (double& e : u_old) e = uu(rng);
  for (double& e : v_frozen) e = uv(rng);
  for (double& e : u0) e = uu(rng);
  for (double& e : v) e = udir(rng);
  const double dt = 1e-3;

  auto [F, J] = brusselator_species_residual(lap, u_old, v_frozen, dt);
  const auto r = taylor_remainder_order(F, J, u0, v, /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere);
  EXPECT_GT(r.order, 1.8) << "observed order " << r.order << ", floor " << r.floor;
  EXPECT_LT(r.order, 2.3) << "observed order " << r.order;
}

TEST(TaylorRemainderOrder, BackwardEulerAffineResidualIsAtTheFloorForEveryH) {
  // The "dn_heat-shaped" case: r(u) = (I - dt*D*Lap) u - rhs is AFFINE in u
  // (no reaction term at all), built directly from the same
  // periodic_laplacian the Brusselator diffusion operator uses, so its
  // Taylor remainder is mathematically zero at every h -- a correct
  // Jacobian action here must report `at_floor_everywhere`, never a fitted
  // order (there is nothing to fit: the remainder never decreases because
  // it starts at round-off).
  constexpr int nx = 4, ny = 4;
  const auto lap = brusselator::periodic_laplacian(nx, ny);
  const std::size_t n = static_cast<std::size_t>(nx * ny);
  constexpr double D = 10.0, dt = 1e-3;
  std::mt19937 rng(23u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> rhs(n), u0(n), v(n);
  for (double& e : rhs) e = u(rng);
  for (double& e : u0) e = u(rng);
  for (double& e : v) e = u(rng);

  auto F = [&lap, rhs, n](const std::vector<double>& x) {
    Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<int>(n));
    Eigen::VectorXd lap_x = lap * xv;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = x[i] - dt * D * lap_x[static_cast<int>(i)] - rhs[i];
    }
    return out;
  };
  auto J = [&lap, n](const std::vector<double>&, const std::vector<double>& v_dir) {
    Eigen::VectorXd vv = Eigen::Map<const Eigen::VectorXd>(v_dir.data(), static_cast<int>(n));
    Eigen::VectorXd lap_v = lap * vv;
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = v_dir[i] - dt * D * lap_v[static_cast<int>(i)];
    return out;
  };

  const auto r = taylor_remainder_order(F, J, u0, v, /*h0=*/1e-1, /*ndecades=*/9);
  EXPECT_TRUE(r.at_floor_everywhere)
      << "an affine residual's exact Jacobian measured a decreasing (non-floor) trend -- fitted "
         "order "
      << r.order;
  for (double rem : r.remainder) {
    EXPECT_LT(rem, 1e-8) << "an affine residual's remainder should be at round-off for every h";
  }
}

// ===========================================================================
// 3. Assembled-vs-matrix-free agreement.
// ===========================================================================

/// The Brusselator diffusion operator's own stencil arithmetic
/// (brusselator_physics.hpp's `periodic_laplacian`, species_core.hpp's
/// `KrylovDiffusionSolver::factor`), written directly on arrays rather than
/// assembled -- see this file's own header comment for why this is a plain
/// function and not a Kokkos kernel (there is no existing Kokkos diffusion
/// stencil in the codebase to call instead).
void periodic_laplacian_stencil_apply(int nx, int ny, const double* x, double* y) {
  const double xx = static_cast<double>(nx) * nx, yy = static_cast<double>(ny) * ny;
  const auto index = [nx](int i, int j) { return j * nx + i; };
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int row = index(i, j);
      y[row] = -2.0 * (xx + yy) * x[row] + xx * x[index((i + 1) % nx, j)] +
              xx * x[index((i + nx - 1) % nx, j)] + yy * x[index(i, (j + 1) % ny)] +
              yy * x[index(i, (j + ny - 1) % ny)];
    }
  }
}

TEST(LinalgDerivativeGate, AssembledDiffusionMatrixAgreesWithEigenProductOnOneRank) {
  constexpr int nx = 16, ny = 16;
  const auto lap = brusselator::periodic_laplacian(nx, ny);
  std::vector<Triplet> tl;
  for (int col = 0; col < lap.outerSize(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(lap, col); it; ++it) {
      tl.push_back({it.row(), it.col(), it.value()});
    }
  }
  const std::int64_t n = nx * ny;
  const RowOwnership owners{{0, n}};
  auto m = DistSparseMatrix::from_triplets(0, n, n, tl, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok());
  auto mp = std::make_shared<DistSparseMatrix>(std::move(m).value());
  const HostApplyFn dist_apply = serial_matrix_apply(mp, static_cast<std::size_t>(n));
  const HostApplyFn eigen_apply = [&lap, n](std::span<const double> x, std::span<double> y) {
    Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<int>(n));
    Eigen::VectorXd yv = lap * xv;
    std::copy(yv.data(), yv.data() + n, y.begin());
    return Result<void>::success();
  };

  const auto norm_r = estimate_operator_norm(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                             dist_apply, dist_apply);
  ASSERT_TRUE(norm_r.ok());
  const auto agree = matrix_free_agreement(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                           dist_apply, eigen_apply, /*n_vectors=*/20,
                                           norm_r.value());
  ASSERT_TRUE(agree.ok());
  EXPECT_TRUE(agree.value().passed) << "max defect " << agree.value().max_defect << " vs bound "
                                    << agree.value().bound;
}

TEST(LinalgDerivativeGate, AssembledDiffusionMatrixAgreesWithHandWrittenStencilApply) {
  constexpr int nx = 16, ny = 16;
  const auto lap = brusselator::periodic_laplacian(nx, ny);
  std::vector<Triplet> tl;
  for (int col = 0; col < lap.outerSize(); ++col) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(lap, col); it; ++it) {
      tl.push_back({it.row(), it.col(), it.value()});
    }
  }
  const std::int64_t n = nx * ny;
  const RowOwnership owners{{0, n}};
  auto m = DistSparseMatrix::from_triplets(0, n, n, tl, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok());
  auto mp = std::make_shared<DistSparseMatrix>(std::move(m).value());
  const HostApplyFn dist_apply = serial_matrix_apply(mp, static_cast<std::size_t>(n));
  const HostApplyFn stencil_apply = [n](std::span<const double> x, std::span<double> y) {
    periodic_laplacian_stencil_apply(nx, ny, x.data(), y.data());
    return Result<void>::success();
  };

  const auto norm_r = estimate_operator_norm(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                             dist_apply, dist_apply);
  ASSERT_TRUE(norm_r.ok());
  const auto agree = matrix_free_agreement(static_cast<std::size_t>(n), static_cast<std::size_t>(n),
                                           dist_apply, stencil_apply, /*n_vectors=*/20,
                                           norm_r.value());
  ASSERT_TRUE(agree.ok());
  EXPECT_TRUE(agree.value().passed) << "max defect " << agree.value().max_defect << " vs bound "
                                    << agree.value().bound;
  std::fprintf(stderr, "[derivative-gate] matrix-free agreement max_defect=%.6g bound=%.6g\n",
              agree.value().max_defect, agree.value().bound);
}
