#pragma once

/// Time windows track (Conformance, Unit 2): the ONE physics source
/// species_component.hpp's own UComponent/VComponent and the participant-
/// ABI shim (examples/brusselator/participants/shim.cpp) both call --
/// extracted from UComponent::advance()/VComponent::advance() (this
/// session) so the reaction-diffusion solve itself exists exactly once,
/// not duplicated between the in-process (fuse::Component) and the
/// externally-driven (fuse_begin_step()/fuse_end_step()) drivers.
///
/// `UCore` is the WHOLE of U's own physics -- U never did any remap work
/// itself (species_component.hpp's own file header: ALL remap, both
/// directions, was hand-done inside VComponent), so nothing further
/// needs splitting out. `VDiffusionCore` is deliberately NOT the whole of
/// V's own physics: it is only the "solve backward-Euler diffusion given
/// q already on THIS grid" step -- the lead's own ruling (Conformance
/// Unit 2) is that the participant graph IS Unit 1's own graph
/// (brusselator_graph.hpp's `add_window_coupling`) PLUS, and only plus,
/// `add_coupler()`/`remap_on(coupler)` on both transfers: both transfer
/// fields stay declared on `u_region` (species_component.hpp's own file
/// header, "WHY BOTH TRANSFER FIELDS LIVE ON u_region"), so the coupler-
/// routed Remap this compiles is the IDENTITY (confirmed this session,
/// `WindowRunner.WindowChannelRemapRoutedToCouplerMatchesIdentity
/// ClosedForm`, test/runtime/window_runner_test.cpp) and V still stands
/// in for the real remap by hand, exactly as `VComponent` (Unit 1,
/// `--driver window`/`--driver runner`) already does: `advance()` remaps
/// q FROM the U grid onto its own grid by hand (`fuse::apply_weights`,
/// `weights_uv_`) before calling this, and remaps its own new state BACK
/// onto the U grid afterward (`remap_state_to_u`). The externally-driven
/// shim (shim.cpp) needs the IDENTICAL by-hand remap glue for the same
/// reason -- its own `VParticipant` wrapper (below `VDiffusionCore`)
/// carries it, built from the SAME `NativeGrid`/`fuse::RemapWeights`
/// machinery `VComponent` uses, so `VDiffusionCore` itself stays exactly
/// what both drivers share: V's own diffusion solve, nothing else.

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/gmres.hpp"
#include "fuse/linalg/pcg.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse/linalg/schur_slab.hpp"

#include "brusselator_physics.hpp"

namespace brusselator {

/// Physics track (closing the Grid track's own solver seam): which
/// `DiffusionSolver` implementation (below) `UCore`/`VDiffusionCore`
/// build. `Auto` (the default) resolves to `Direct` at `num_ranks() == 1`
/// (every goldened single-rank caller: the in-process example, the 3-rank
/// MPMD row) and to `Schur` at `num_ranks() > 1` -- an exact direct solve
/// stays the default wherever one is affordable, consistent with this
/// project's standing rule that robustness and accuracy outweigh
/// efficiency. `Pcg`/`Gmres` are selectable explicitly (a participant's
/// own `--solver` flag) at any rank count, including 1 (where they are
/// exercised against `DirectDiffusionSolver`'s own answer, not only
/// against each other).
enum class SolverKind { Auto, Direct, Schur, Pcg, Gmres };

/// Parses a participant's `--solver` flag (or any other string-sourced
/// solver selection) into a `SolverKind`. Refuses `E-BRUSS-SOLVER-UNKNOWN`
/// for anything else -- a caller typo in a solver name must be a named,
/// caught refusal, not a silent fall-through to the default (test fires:
/// `BrusselatorSpeciesCoreDecomposition.ParseSolverKindRefusesUnknownName`).
[[nodiscard]] inline fuse::Result<SolverKind> parse_solver_kind(const std::string& s) {
  if (s == "auto") return SolverKind::Auto;
  if (s == "direct") return SolverKind::Direct;
  if (s == "schur") return SolverKind::Schur;
  if (s == "pcg") return SolverKind::Pcg;
  if (s == "gmres") return SolverKind::Gmres;
  return fuse::Failure{"E-BRUSS-SOLVER-UNKNOWN",
                       "parse_solver_kind: unknown diffusion solver '" + s +
                           "' (want one of auto|direct|schur|pcg|gmres)"};
}

/// Builds the `fuse::linalg::GhostExchanger` a `KrylovDiffusionSolver`
/// (below) needs for its assembled operator's own `DistVector`s, from
/// THAT operator's own derived `GhostPattern` (`DistSparseMatrix::
/// ghost_pattern()`, fixed by `from_triplets` from the triplets and row
/// ownership actually given it -- NOT necessarily in the same ghost ORDER
/// `NativeGrid::ghost_pattern()` uses for its own below-row-then-above-row
/// convention, so a caller must build the exchanger from the OPERATOR's
/// pattern, never the grid's, once the operator exists; see
/// docs/dev/linalg-distributed.md's "the UNIQUE set of such columns,
/// sorted ascending by global column" for why the two need not agree).
/// `fuse::linalg::make_serial_exchanger()` at `num_ranks() == 1` (the
/// operator's own ghost pattern is then empty: every off-diagonal column
/// of a periodic 1-rank Laplacian is still this rank's own); the
/// participants build one from `fuse::linalg::make_mpi_exchanger`
/// instead; the tier-0 two/three-partition verification test builds an
/// in-process stand-in from the SAME pattern.
using GhostExchangerFactory = std::function<std::shared_ptr<const fuse::linalg::GhostExchanger>(
    const fuse::linalg::GhostPattern&, std::size_t)>;

[[nodiscard]] inline GhostExchangerFactory serial_exchanger_factory() {
  return [](const fuse::linalg::GhostPattern&, std::size_t) {
    return fuse::linalg::make_serial_exchanger();
  };
}

/// Shared physical parameters, all defaulted to the Toy's own values
/// (README.md "Model and reference"). Moved here (from
/// species_component.hpp, where it used to live) when UCore/VDiffusionCore
/// were factored out, since UCore's own constructor needs it and
/// species_core.hpp has no dependency on species_component.hpp (the
/// reverse is true: species_component.hpp includes this header).
struct PhysicsConfig {
  double diffusion = 10.0;  ///< alpha
  double A = 3.4;
  double B = 1.0;
  bool   forcing = true;
  /// Physics track: which `DiffusionSolver` `UCore` builds (`VDiffusionCore`
  /// takes its own, separate `SolverKind` argument -- V's diffusion solve
  /// is independent of U's, species_core.hpp's own file header explains
  /// why `VDiffusionCore` is a distinct class rather than a second
  /// `UCore` instance).
  SolverKind solver = SolverKind::Auto;
  /// Relative residual tolerance `Pcg`/`Gmres` stop at -- DERIVED:
  /// `BrusselatorSpeciesCoreDecomposition.DeriveKrylovSolverTolerance`
  /// (test/verification/brusselator_species_core_decomposition_test.cpp)
  /// sweeps `rel_tol` from `1e-8` to `1e-13` for both `Pcg` and `Gmres`
  /// and measures each one's max|difference| against the single-rank
  /// `Direct` reference over a 2-partition, 2-window run (`--dt-u 0.002`,
  /// 20 steps) at this example's own defaults. MEASURED (this session,
  /// this build): `1e-11` -> 3.3e-12 (pcg) / 1.5e-12 (gmres); `1e-12` ->
  /// 1.4e-13 / 1.4e-13; `1e-13` -> 5.9e-14 / 5.5e-14 -- the SAME order of
  /// magnitude `TwoPartitionSchurMatchesSingleRankUCore` measures for an
  /// EXACT row-banded solve against the same reference (2.4e-14, pure
  /// floating-point summation-order disagreement, no iteration involved
  /// at all). `1e-13` is the tightest of the swept values whose own cost
  /// (a handful of extra PCG/GMRES iterations) is still negligible at
  /// this example's grid sizes, and it is the value that keeps a Krylov
  /// solve's own disagreement closest to that exact-solve floor rather
  /// than merely "small" -- this project's standing rule (robustness and
  /// accuracy over efficiency) favours it over a looser value that would
  /// also have passed. `Schur` needs no such tolerance (an exact solve,
  /// bounded only by the same round-off `Direct` already carries).
  double solver_tolerance = 1e-13;
};

/// The one thing `UCore`/`VDiffusionCore` need from a linear solve: factor
/// `(1 + extra_diagonal)*I - dt*diffusion*Laplacian` once per distinct `dt`
/// (re-using the cached factorisation under the SAME tolerance test both
/// classes already used before this seam existed) and solve it for a
/// right-hand side that changes every step. `DirectDiffusionSolver`
/// below, `Eigen::SimplicialLDLT` over the WHOLE nx*ny grid, is valid
/// only at `num_ranks() == 1`. `SchurDiffusionSolver` (exact, built on
/// `fuse::linalg::SchurSlabSolver`) and `KrylovDiffusionSolver` (`Pcg`/
/// `Gmres`, built on `fuse::linalg::DistSparseMatrix`/`DistVector`) are
/// the `num_ranks() > 1` implementations of this SAME interface the Grid
/// track's own seam was left for -- `UCore`/`VDiffusionCore` call
/// through this interface and do not change per solver kind; only
/// `make_diffusion_solver` (below) picks which concrete type to build.
class DiffusionSolver {
 public:
  virtual ~DiffusionSolver() = default;

  /// (Re)factors the matrix if `dt` differs from the last factored step by
  /// more than the caller's own re-factorisation tolerance; a no-op
  /// otherwise -- exactly the gated re-factorisation `UCore::step`/
  /// `VDiffusionCore::step` ran inline before this seam existed.
  /// `extra_diagonal` is `UCore`'s own `+dt` term (`VDiffusionCore` passes
  /// 0, its own unchanged diagonal).
  [[nodiscard]] virtual fuse::Result<void> factor(double dt, double diffusion,
                                                   double extra_diagonal) = 0;
  [[nodiscard]] virtual fuse::Result<Vector> solve(const Vector& rhs) = 0;
};

/// `Eigen::SimplicialLDLT` over the whole nx*ny periodic Laplacian --
/// `DiffusionSolver`'s one implementation today. The factor/solve bodies
/// are copied verbatim from `UCore::step`/`VDiffusionCore::step`'s own
/// inline code (this refactor's only change is WHERE that code lives),
/// so a caller at `num_ranks() == 1` sees byte-identical arithmetic.
class DirectDiffusionSolver final : public DiffusionSolver {
 public:
  explicit DirectDiffusionSolver(Sparse laplacian) : laplacian_(std::move(laplacian)) {}

  [[nodiscard]] fuse::Result<void> factor(double dt, double diffusion,
                                          double extra_diagonal) override {
    if (factored_step_ < 0.0 ||
        std::abs(dt - factored_step_) > 32 * std::numeric_limits<double>::epsilon() * dt) {
      Eigen::SparseMatrix<double> matrix = (-dt * diffusion) * laplacian_;
      const int n = static_cast<int>(laplacian_.rows());
      for (int i = 0; i < n; ++i) matrix.coeffRef(i, i) += 1.0 + extra_diagonal;
      factor_.compute(matrix);
      if (factor_.info() != Eigen::Success) {
        return fuse::Failure{"E-BRUSS-DIFFUSION-FACTORIZE",
                             "DirectDiffusionSolver: factorization failed"};
      }
      factored_step_ = dt;
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<Vector> solve(const Vector& rhs) override {
    Vector next = factor_.solve(rhs);
    if (factor_.info() != Eigen::Success || !next.allFinite()) {
      return fuse::Failure{"E-BRUSS-DIFFUSION-SOLVE",
                           "DirectDiffusionSolver: solve failed or produced a non-finite state"};
    }
    return next;
  }

 private:
  Sparse laplacian_;
  Eigen::SimplicialLDLT<Sparse> factor_;
  double factored_step_ = -1.0;
};

/// `fuse::linalg::SchurSlabSolver` over this rank's own row band -- an
/// EXACT direct solve of `(1 + extra_diagonal)*I - dt*diffusion*L_h`,
/// valid at any `num_ranks() >= 1`. `SchurSlabSolver::factor` only knows
/// the unit-coefficient form `A = I - coefficient*L_h`; the rescaling
/// below (`scale = 1 + extra_diagonal`, `coefficient = dt*diffusion /
/// scale`) is exact, not an approximation: `A = scale*I - dt*diffusion*
/// L_h = scale*(I - coefficient*L_h)`, so solving `A x = rhs` is exactly
/// `(I - coefficient*L_h) x = rhs/scale` for the SAME `x` -- no further
/// rescaling of the solution is needed. At `extra_diagonal == 0` (every
/// `VDiffusionCore` call) `scale == 1` and this is a no-op identity.
class SchurDiffusionSolver final : public DiffusionSolver {
 public:
  SchurDiffusionSolver(int nx, std::int64_t ny, std::int64_t j_lo, std::int64_t j_hi,
                       fuse::linalg::RowOwnership row_ownership,
                       std::shared_ptr<const fuse::linalg::Reducer> reducer)
      : nx_(nx),
        ny_(ny),
        j_lo_(j_lo),
        j_hi_(j_hi),
        row_ownership_(std::move(row_ownership)),
        reducer_(std::move(reducer)),
        owned_n_(static_cast<std::size_t>((j_hi_ - j_lo_) * nx_)),
        // Schur's own `solve()` needs no ghost communication at all (its
        // one allgather is internal to `factor()`/`solve()`, never
        // exposed through `DistVector::update_ghosts()` -- schur_slab.hpp's
        // own class comment) -- the zero-ghost serial exchanger is the
        // correct choice regardless of `reducer_`'s own rank count, not a
        // single-rank special case.
        exchanger_(fuse::linalg::make_serial_exchanger()) {}

  [[nodiscard]] fuse::Result<void> factor(double dt, double diffusion,
                                          double extra_diagonal) override {
    if (!factored_ ||
        std::abs(dt - factored_dt_) > 32 * std::numeric_limits<double>::epsilon() * std::max(dt, 1.0) ||
        extra_diagonal != factored_extra_diagonal_) {
      const double scale = 1.0 + extra_diagonal;
      const double coefficient = dt * diffusion / scale;
      auto solved = fuse::linalg::SchurSlabSolver::factor(nx_, ny_, j_lo_, j_hi_, coefficient,
                                                           row_ownership_, reducer_);
      if (!solved.ok()) {
        return fuse::Failure{"E-BRUSS-DIFFUSION-FACTORIZE",
                             "SchurDiffusionSolver: " + solved.error().code + ": " +
                                 solved.error().message};
      }
      solver_ = std::make_unique<fuse::linalg::SchurSlabSolver>(std::move(solved.value()));
      scale_ = scale;
      factored_dt_ = dt;
      factored_extra_diagonal_ = extra_diagonal;
      factored_ = true;
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<Vector> solve(const Vector& rhs) override {
    fuse::linalg::DistVector rhs_dv(owned_n_, exchanger_, reducer_);
    for (std::size_t k = 0; k < owned_n_; ++k) {
      rhs_dv.data()[k] = rhs[static_cast<int>(k)] / scale_;
    }
    fuse::linalg::DistVector x_dv(owned_n_, exchanger_, reducer_);
    x_dv.set_zero();
    const fuse::Result<void> solved = solver_->solve(rhs_dv, x_dv);
    if (!solved.ok()) {
      return fuse::Failure{"E-BRUSS-DIFFUSION-SOLVE",
                           "SchurDiffusionSolver: " + solved.error().code + ": " +
                               solved.error().message};
    }
    Vector out(static_cast<int>(owned_n_));
    for (std::size_t k = 0; k < owned_n_; ++k) out[static_cast<int>(k)] = x_dv.data()[k];
    if (!out.allFinite()) {
      return fuse::Failure{"E-BRUSS-DIFFUSION-SOLVE",
                           "SchurDiffusionSolver: solve produced a non-finite state"};
    }
    return out;
  }

 private:
  int nx_;
  std::int64_t ny_, j_lo_, j_hi_;
  fuse::linalg::RowOwnership row_ownership_;
  std::shared_ptr<const fuse::linalg::Reducer> reducer_;
  std::size_t owned_n_;
  std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_;
  std::unique_ptr<fuse::linalg::SchurSlabSolver> solver_;
  double scale_ = 1.0;
  bool factored_ = false;
  double factored_dt_ = -1.0;
  double factored_extra_diagonal_ = 0.0;
};

/// `fuse::linalg::pcg`/`gmres` over a `fuse::linalg::DistSparseMatrix`
/// assembly of `(1 + extra_diagonal)*I - dt*diffusion*L_h`, row-banded
/// exactly as `SchurDiffusionSolver` above, with a `JacobiPreconditioner`
/// (`DistSparseMatrix::diagonal()`, no communication) -- the native,
/// dependency-free Krylov alternative this track's deliverables name.
/// Global triplets use the SAME canonical index `j*nx+i` and the SAME
/// stencil coefficients `periodic_laplacian` builds (brusselator_physics.hpp),
/// restricted to this rank's own row band `[j_lo, j_hi)`; the off-process
/// columns (periodic neighbours across a row-band boundary) become the
/// matrix's own `ghost_pattern()`, NOT `NativeGrid::ghost_pattern()` --
/// see `GhostExchangerFactory`'s own comment for why a caller must build
/// its exchanger from the former.
class KrylovDiffusionSolver final : public DiffusionSolver {
 public:
  enum class Algorithm { Pcg, Gmres };

  KrylovDiffusionSolver(Algorithm algo, int nx, std::int64_t ny, std::int64_t j_lo,
                       std::int64_t j_hi, fuse::linalg::RowOwnership row_ownership,
                       std::shared_ptr<const fuse::linalg::Reducer> reducer,
                       GhostExchangerFactory make_exchanger, double rel_tol)
      : algo_(algo),
        nx_(nx),
        ny_(ny),
        j_lo_(j_lo),
        j_hi_(j_hi),
        row_ownership_(std::move(row_ownership)),
        reducer_(std::move(reducer)),
        make_exchanger_(std::move(make_exchanger)),
        owned_n_(static_cast<std::size_t>((j_hi_ - j_lo_) * nx_)),
        rel_tol_(rel_tol) {}

  [[nodiscard]] fuse::Result<void> factor(double dt, double diffusion,
                                          double extra_diagonal) override {
    if (!factored_ ||
        std::abs(dt - factored_dt_) > 32 * std::numeric_limits<double>::epsilon() * std::max(dt, 1.0) ||
        extra_diagonal != factored_extra_diagonal_) {
      const double scale = 1.0 + extra_diagonal;
      const double xx = static_cast<double>(nx_) * nx_;
      const double yy = static_cast<double>(ny_) * ny_;
      const std::int64_t global_n = static_cast<std::int64_t>(nx_) * ny_;
      const auto index = [this](std::int64_t i, std::int64_t j) { return j * nx_ + i; };
      std::vector<fuse::linalg::Triplet> triplets;
      triplets.reserve(owned_n_ * 5);
      for (std::int64_t j = j_lo_; j < j_hi_; ++j) {
        for (std::int64_t i = 0; i < nx_; ++i) {
          const std::int64_t row = index(i, j);
          triplets.push_back({row, row, scale + dt * diffusion * 2.0 * (xx + yy)});
          triplets.push_back({row, index((i + 1) % nx_, j), -dt * diffusion * xx});
          triplets.push_back({row, index((i + nx_ - 1) % nx_, j), -dt * diffusion * xx});
          triplets.push_back({row, index(i, (j + 1) % ny_), -dt * diffusion * yy});
          triplets.push_back({row, index(i, (j + ny_ - 1) % ny_), -dt * diffusion * yy});
        }
      }
      auto built = fuse::linalg::DistSparseMatrix::from_triplets(
          j_lo_ * nx_, j_hi_ * nx_, global_n, triplets, reducer_, row_ownership_);
      if (!built.ok()) {
        return fuse::Failure{"E-BRUSS-DIFFUSION-FACTORIZE",
                             "KrylovDiffusionSolver: " + built.error().code + ": " +
                                 built.error().message};
      }
      matrix_ = std::make_unique<fuse::linalg::DistSparseMatrix>(std::move(built.value()));
      exchanger_ = make_exchanger_(matrix_->ghost_pattern(), owned_n_);
      auto jac = fuse::linalg::JacobiPreconditioner::from_matrix(*matrix_);
      if (!jac.ok()) {
        return fuse::Failure{"E-BRUSS-DIFFUSION-FACTORIZE",
                             "KrylovDiffusionSolver: " + jac.error().code + ": " +
                                 jac.error().message};
      }
      precond_ = std::make_unique<fuse::linalg::JacobiPreconditioner>(std::move(jac.value()));
      factored_dt_ = dt;
      factored_extra_diagonal_ = extra_diagonal;
      factored_ = true;
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<Vector> solve(const Vector& rhs) override {
    fuse::linalg::DistVector b(owned_n_, exchanger_, reducer_);
    for (std::size_t k = 0; k < owned_n_; ++k) b.data()[k] = rhs[static_cast<int>(k)];
    fuse::linalg::DistVector x(owned_n_, exchanger_, reducer_);
    x.set_zero();

    fuse::Result<fuse::linalg::SolverReport> rep{fuse::linalg::SolverReport{}};
    if (algo_ == Algorithm::Pcg) {
      fuse::linalg::PcgOptions opt;
      opt.rel_tol = rel_tol_;
      rep = fuse::linalg::pcg(*matrix_, *precond_, x, b, opt);
    } else {
      fuse::linalg::GmresOptions opt;
      opt.rel_tol = rel_tol_;
      rep = fuse::linalg::gmres(*matrix_, *precond_, x, b, opt);
    }
    if (!rep.ok()) {
      return fuse::Failure{"E-BRUSS-DIFFUSION-SOLVE",
                           "KrylovDiffusionSolver: " + rep.error().code + ": " +
                               rep.error().message};
    }
    Vector out(static_cast<int>(owned_n_));
    for (std::size_t k = 0; k < owned_n_; ++k) out[static_cast<int>(k)] = x.data()[k];
    if (!out.allFinite()) {
      return fuse::Failure{"E-BRUSS-DIFFUSION-SOLVE",
                           "KrylovDiffusionSolver: solve produced a non-finite state"};
    }
    return out;
  }

 private:
  Algorithm algo_;
  int nx_;
  std::int64_t ny_, j_lo_, j_hi_;
  fuse::linalg::RowOwnership row_ownership_;
  std::shared_ptr<const fuse::linalg::Reducer> reducer_;
  GhostExchangerFactory make_exchanger_;
  std::size_t owned_n_;
  double rel_tol_;
  std::unique_ptr<fuse::linalg::DistSparseMatrix> matrix_;
  std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_;
  std::unique_ptr<fuse::linalg::JacobiPreconditioner> precond_;
  bool factored_ = false;
  double factored_dt_ = -1.0;
  double factored_extra_diagonal_ = 0.0;
};

/// Picks and builds the `DiffusionSolver` implementation `kind` names,
/// resolving `SolverKind::Auto` by `num_ranks()` (this enum's own doc
/// comment). `grid` is read only at construction (row band, ownership,
/// `nx`/`ny`); nothing here keeps a reference to it.
///
/// Throws `fuse::InvalidArgument` (a caller-configuration mistake, not a
/// data-dependent failure -- the same split `reaction_kokkos`'s own size
/// check draws) for `SolverKind::Direct` requested explicitly at
/// `num_ranks() > 1`: `DirectDiffusionSolver` is a whole-grid factorisation
/// that cannot be correct over a row band alone, so this is refused at
/// construction rather than left to fail confusingly deep inside a solve
/// (test fires: `BrusselatorSpeciesCoreDecomposition.
/// DirectSolverRefusesWhenDecomposed`).
[[nodiscard]] inline std::unique_ptr<DiffusionSolver> make_diffusion_solver(
    const NativeGrid& grid, SolverKind kind, double rel_tol,
    std::shared_ptr<const fuse::linalg::Reducer> reducer,
    const GhostExchangerFactory& make_exchanger) {
  SolverKind resolved = kind;
  if (resolved == SolverKind::Auto) {
    resolved = grid.num_ranks() == 1 ? SolverKind::Direct : SolverKind::Schur;
  }
  switch (resolved) {
    case SolverKind::Direct:
      if (grid.num_ranks() != 1) {
        FUSE_THROW(fuse::InvalidArgument, "{}",
                  std::string("make_diffusion_solver: SolverKind::Direct requires num_ranks()==1, "
                              "got ") +
                      std::to_string(grid.num_ranks()) +
                      " (E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED)");
      }
      return std::make_unique<DirectDiffusionSolver>(periodic_laplacian(grid.nx(), grid.ny()));
    case SolverKind::Schur:
      // SchurSlabSolver reinterprets fuse::linalg::RowOwnership over ny()
      // GRID rows, not nx()*ny() cells -- row_ownership_by_grid_row(),
      // not row_ownership() (that one is for DistSparseMatrix, below).
      return std::make_unique<SchurDiffusionSolver>(
          grid.nx(), grid.ny(), grid.j_lo(), grid.j_hi(), grid.row_ownership_by_grid_row(), reducer);
    case SolverKind::Pcg:
      return std::make_unique<KrylovDiffusionSolver>(
          KrylovDiffusionSolver::Algorithm::Pcg, grid.nx(), grid.ny(), grid.j_lo(), grid.j_hi(),
          grid.row_ownership(), reducer, make_exchanger, rel_tol);
    case SolverKind::Gmres:
      return std::make_unique<KrylovDiffusionSolver>(
          KrylovDiffusionSolver::Algorithm::Gmres, grid.nx(), grid.ny(), grid.j_lo(), grid.j_hi(),
          grid.row_ownership(), reducer, make_exchanger, rel_tol);
    case SolverKind::Auto:
      break;  // unreachable: resolved above
  }
  FUSE_THROW(fuse::InvalidArgument, "{}", std::string("make_diffusion_solver: unresolved SolverKind"));
}

/// U_t = a*Lap(U) + B - U + q + f, extracted verbatim from
/// UComponent::advance() (species_component.hpp's own file header has
/// the full derivation and the measured conservation consequence of the
/// lagged-reaction choice). Pure math: no fuse::Component, no
/// FieldRegistry, no NativeGrid -- just `nx*ny` and the physical
/// parameters, so it is usable from a C-ABI shim with no FUSE mesh
/// machinery linked in at all.
class UCore {
 public:
  /// `grid`: U's own `NativeGrid` (possibly decomposed over
  /// `grid.num_ranks()` ranks -- every vector below is sized
  /// `grid.owned_size()`, THIS rank's own row band, not `grid.size()`;
  /// at `num_ranks() == 1` the two are equal and every computation below
  /// is byte-identical to before this constructor took a grid rather than
  /// a bare `nx, ny` pair). `reducer`/`make_exchanger` default to the
  /// one-rank, zero-ghost pair (`fuse::linalg::make_serial_reducer()`/
  /// `serial_exchanger_factory()`) -- the only choice that is ever valid
  /// at `num_ranks() == 1` (every in-process caller: `UComponent`, the
  /// 3-rank MPMD row's own single-rank "U"); a decomposed participant
  /// (examples/brusselator/participants) supplies real ones.
  UCore(const NativeGrid& grid, PhysicsConfig cfg, Vector initial_v_on_u,
        std::shared_ptr<const fuse::linalg::Reducer> reducer = nullptr,
        GhostExchangerFactory make_exchanger = nullptr)
      : n_(grid.owned_size()),
        global_n_(grid.size()),
        cfg_(cfg),
        reducer_(reducer ? std::move(reducer) : fuse::linalg::make_serial_reducer()),
        solver_(make_diffusion_solver(grid, cfg.solver, cfg.solver_tolerance, reducer_,
                                      make_exchanger ? make_exchanger : serial_exchanger_factory())),
        mask_(forcing_mask_owned(grid.nx(), grid.ny(), grid.j_lo(), grid.owned_rows())),
        mask_mean_(global_mean(mask_)),
        state_(initial_condition_owned(/*is_u=*/true, grid.nx(), grid.ny(), grid.j_lo(),
                                       grid.owned_rows())),
        required_(std::move(initial_v_on_u)),
        provided_(n_),
        q_prev_(n_),
        saved_state_(n_),
        // Found by inspection this session (examples/brusselator/
        // participants work): `saved_q_prev_` was left at
        // Eigen::VectorXd's own default size (0), not `n_` like every
        // sibling saved_* buffer here -- a genuine latent bug (if
        // restore_state() ever ran before the first save_state(), the
        // resulting size-0 `q_prev_ = saved_q_prev_` assignment would
        // misbehave) and fixed on its own merits. NOTE: measured this
        // session (BrusselatorParticipants.
        // ExternallyDrivenBothMembersMatchInProcessRunner, 10 repeated
        // runs before and after this fix) that this fix did NOT change
        // the observed intermittent E-RT-NONFINITE failure rate (4/10
        // both times) -- that non-determinism has a DIFFERENT, still-
        // unidentified cause, reported separately to the lead rather
        // than claimed as resolved here.
        saved_q_prev_(n_) {
    reaction_kokkos(state_, required_, &provided_);
    q_prev_ = provided_;
  }

  /// `required`: P_VU(V) mean over [t_begin, t_end), on U's own grid,
  /// size n(). On success, `state()`/`provided()` are valid at t_end.
  /// `dt`: the EXACT step length, i.e. `fuse::StepContext::dt().to_double()`
  /// -- NOT `t_end - t_begin` recomputed from the two already-rounded
  /// doubles. Found by the lead (bit-identical check, Conformance Unit 2):
  /// the old UComponent::advance() this was extracted from called
  /// `ctx.dt().to_double()` (the EXACT rational t_end-t_begin, rounded
  /// ONCE) for every dt-scaled quantity here, while `t0`/`t1` (also each
  /// rounded once from their own Rationals) were used ONLY for
  /// forcing_active_duration(). Recomputing `dt` as a double subtraction
  /// of `t_end`/`t_begin` inside this function (this file's own original
  /// version) adds a THIRD, different rounding -- the two are not always
  /// bit-equal for a Rational step. VERIFIED after applying this fix to
  /// BOTH call sites (species_component.hpp's own UComponent::advance()
  /// and shim.cpp's own bruss_u_step()) and rebuilding: `diff` of the
  /// full stdout of both drivers at both --end 0.5 and --end 3, pre- vs
  /// post-refactor, is empty (exit 0) in all four cases -- the pre-
  /// refactor global_balance_error (2.407e-12 at --end 0.5, 1.266e-11 at
  /// --end 3) and this fixed version's are the SAME printed figures, not
  /// just close. Before this specific fix (species_core.hpp's own
  /// `dt = t_end - t_begin`, with both call sites otherwise already
  /// ported), the refactor's own global_balance_error was 1.842e-12 and
  /// 1.038e-11 respectively -- different from the pre-refactor figures,
  /// though `state()`/`final_mean_u` were already bit-identical either
  /// way (only round-off-scale diagnostics that sum many steps' worth of
  /// dt-scaled terms -- external_total_, exchange_out_ -- could even
  /// show the difference).
  [[nodiscard]] fuse::Result<void> step(double t_begin, double t_end, double dt,
                                        const Vector& required) {
    required_ = required;
    if (dt <= 0.0) return fuse::Result<void>::success();

    const double active = cfg_.forcing ? forcing_active_duration(t_begin, t_end) : 0.0;
    Vector rhs_base = state_.array() + dt * cfg_.B;
    if (cfg_.forcing && active > 0.0) rhs_base += active * mask_;

    if (const auto factored = solver_->factor(dt, cfg_.diffusion, /*extra_diagonal=*/dt);
        !factored.ok()) {
      return fuse::Failure{"E-BRUSS-U-FACTORIZE", "UCore: diffusion factorization failed"};
    }
    Vector q_new = q_prev_;
    Vector next(n_);
    bool converged = false;
    for (int k = 0; k < kMaxReactionIterations; ++k) {
      const Vector rhs = rhs_base + (0.5 * dt) * (q_prev_ + q_new);
      auto solved = solver_->solve(rhs);
      if (!solved.ok()) {
        return fuse::Failure{"E-BRUSS-U-SOLVE",
                             "UCore: diffusion solve failed or produced a non-finite state"};
      }
      next = std::move(solved.value());
      Vector q_next(n_);
      reaction_kokkos(next, required_, &q_next);
      // GLOBAL, not this rank's own partial: a decomposed solve() (Schur,
      // Pcg, Gmres) is COLLECTIVE, so every rank must take the SAME
      // branch below and therefore call solve() the SAME number of times
      // -- a per-rank LOCAL `change`/`scale` could let one rank's own
      // owned band converge a step early and stop calling solve() while
      // its peers keep going, stranding them inside the next collective
      // (the same hazard this track's own standing lesson names for a
      // refusal; this is the fixed-point loop's own instance of it). Two
      // `Reducer::max` calls over the SAME `fuse::linalg::Reducer` every
      // rank's `solve()` already shares make every rank's own `change`/
      // `scale` IDENTICAL, so the `break` below fires on the same `k`
      // everywhere. At `num_ranks()==1` (`make_serial_reducer()`),
      // `max(local)` returns `local` unchanged, so this is byte-identical
      // to the local `lpNorm<Eigen::Infinity>()` this replaced.
      const double change = reducer_->max((q_next - q_new).lpNorm<Eigen::Infinity>());
      const double scale = reducer_->max(1.0 + q_next.lpNorm<Eigen::Infinity>());
      q_new = std::move(q_next);
      if (change <= kReactionTolerance * scale) { converged = true; break; }
    }
    if (!converged) {
      return fuse::Failure{"E-BRUSS-U-REACTION",
                           "UCore: the implicit reaction fixed point did not contract to " +
                               std::to_string(kReactionTolerance) + " in " +
                               std::to_string(kMaxReactionIterations) + " solves over [" +
                               std::to_string(t_begin) + ", " + std::to_string(t_end) + "]"};
    }
    state_ = std::move(next);
    external_total_ += dt * cfg_.B - dt * global_mean(state_) + active * mask_mean_;
    exchange_out_ += 0.5 * dt * (global_mean(q_prev_) + global_mean(q_new));
    provided_ = q_new;
    q_prev_ = std::move(q_new);
    return fuse::Result<void>::success();
  }

  void save_state() {
    saved_state_ = state_;
    saved_external_total_ = external_total_;
    saved_exchange_out_ = exchange_out_;
    saved_q_prev_ = q_prev_;
  }
  void restore_state() {
    state_ = saved_state_;
    external_total_ = saved_external_total_;
    exchange_out_ = saved_exchange_out_;
    q_prev_ = saved_q_prev_;
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `z = M^-1 r` for this
  /// species' own block-Jacobi preconditioner, `M` the SAME `(1 +
  /// extra_diagonal)*I - dt*diffusion*Laplacian` system `step()` itself
  /// factors and solves every substep -- `dt`/`extra_diagonal` convention
  /// IDENTICAL to `step()`'s own call (`extra_diagonal = dt`, this
  /// class's own reaction term's `+dt` diagonal contribution). Calling
  /// `factor()` here is deliberately NOT skipped as "already factored by
  /// the last `step()` call": `factor()` is itself idempotent at an
  /// unchanged `dt` (`DiffusionSolver::factor()`'s own doc, species_core.
  /// hpp above -- "a no-op otherwise"), so re-asserting it costs nothing
  /// when a prior `step()` already factored this exact `dt`, and makes
  /// this method callable correctly even before any `step()` ever ran
  /// (a Newton solve's own preconditioner-refresh call, if one were ever
  /// wired at a `preconditioner_lag` boundary preceding the first
  /// substep). No state (`state_`/`q_prev_`/the accumulators) is read or
  /// written -- `solver_` is the ONLY member this touches, matching the
  /// per-sample, block-diagonal-in-time approximation newton-krylov.md
  /// Sec. 9.5 derives (the TRUE, lower-triangular-in-time sensitivity is
  /// never assembled; this is honestly a smoother, not an inverse).
  [[nodiscard]] fuse::Result<Vector> apply_diffusion_preconditioner(double dt, const Vector& r) {
    if (const auto factored = solver_->factor(dt, cfg_.diffusion,
                                              /*extra_diagonal=*/dt);
        !factored.ok()) {
      return fuse::Failure{"E-BRUSS-U-PRECOND-FACTORIZE",
                           "UCore::apply_diffusion_preconditioner: factorization failed"};
    }
    auto solved = solver_->solve(r);
    if (!solved.ok()) {
      return fuse::Failure{"E-BRUSS-U-PRECOND-SOLVE",
                           "UCore::apply_diffusion_preconditioner: solve failed or "
                           "produced a non-finite result"};
    }
    return solved;
  }

  /// This rank's own OWNED count (`grid.owned_size()` at construction) --
  /// equals the grid's GLOBAL cell count at `num_ranks()==1`, every
  /// existing (undecomposed) caller's own reading of `n()`. `required`/
  /// `provided`/`state` are all this size, not the global one; see
  /// `global_mean()` for how a GLOBAL quantity (every diagnostic
  /// accumulator) is computed from them regardless.
  [[nodiscard]] int n() const { return n_; }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] const Vector& provided() const { return provided_; }
  [[nodiscard]] double external_total() const { return external_total_; }
  [[nodiscard]] double exchange_out() const { return exchange_out_; }
  /// The mean of the WHOLE species over every rank's owned cells: one
  /// reduction, so it is a collective every rank of the component must
  /// call. On one rank it is `state().mean()` bit for bit. A participant's
  /// own loop over its owned slab gives a per-rank partial mean, which the
  /// first 5-rank host run (2026-10-10) would have logged as if global.
  [[nodiscard]] double state_mean() const { return global_mean(state_); }

 private:
  static constexpr double kReactionTolerance = 1e-13;
  static constexpr int    kMaxReactionIterations = 50;

  /// This rank's own owned partial sum of `owned`, reduced across every
  /// rank of `reducer_` and divided by the GLOBAL cell count `global_n_`
  /// -- never by `owned.size()`, which would silently report each rank's
  /// own share as if it were the whole mean (the same requirement
  /// `SlabField::owned_mean`, brusselator_physics.hpp, states for the
  /// explicit-quantity kernels). At `num_ranks()==1`, `global_n_==n_` and
  /// `reducer_->sum(local)==local`, so this equals `owned.mean()` bit for
  /// bit (`Eigen::VectorXd::mean()` is itself `sum()/size()`).
  [[nodiscard]] double global_mean(const Vector& owned) const {
    return reducer_->sum(owned.sum()) / static_cast<double>(global_n_);
  }

  int n_;
  int global_n_;
  PhysicsConfig cfg_;
  std::shared_ptr<const fuse::linalg::Reducer> reducer_;
  std::unique_ptr<DiffusionSolver> solver_;
  Vector mask_;
  double mask_mean_;
  Vector state_;
  Vector required_, provided_, q_prev_;
  Vector saved_state_, saved_q_prev_;
  double external_total_ = 0.0, saved_external_total_ = 0.0;
  double exchange_out_ = 0.0, saved_exchange_out_ = 0.0;
};

/// V_t = a*Lap(V) - q_on_v, extracted from the diffusion-solve portion of
/// VComponent::advance() only -- this class's own top comment (above)
/// explains why the remap half is deliberately NOT here.
class VDiffusionCore {
 public:
  /// `grid`: V's own `NativeGrid` (possibly decomposed, mirroring
  /// `UCore`'s own constructor comment above). `solver`/`solver_tolerance`
  /// select the `DiffusionSolver` the same way `PhysicsConfig::solver`/
  /// `solver_tolerance` do for `UCore` -- a SEPARATE choice from U's own
  /// (species_core.hpp's own file header explains why V's diffusion
  /// solve is independent of U's).
  VDiffusionCore(const NativeGrid& grid, double diffusion, SolverKind solver = SolverKind::Auto,
                double solver_tolerance = 1e-13,
                std::shared_ptr<const fuse::linalg::Reducer> reducer = nullptr,
                GhostExchangerFactory make_exchanger = nullptr)
      : n_(grid.owned_size()),
        global_n_(grid.size()),
        diffusion_(diffusion),
        reducer_(reducer ? std::move(reducer) : fuse::linalg::make_serial_reducer()),
        solver_(make_diffusion_solver(grid, solver, solver_tolerance, reducer_,
                                      make_exchanger ? make_exchanger : serial_exchanger_factory())),
        state_(initial_condition_owned(/*is_u=*/false, grid.nx(), grid.ny(), grid.j_lo(),
                                       grid.owned_rows())),
        saved_state_(n_) {}

  /// `q_on_v`: the reaction rate mean over this step, ALREADY on this
  /// grid (size n()). On success, `state()` is valid at t_end = t_begin+dt.
  [[nodiscard]] fuse::Result<void> step(double dt, const Vector& q_on_v) {
    if (dt <= 0.0) return fuse::Result<void>::success();
    Vector rhs = state_ - dt * q_on_v;
    if (const auto factored = solver_->factor(dt, diffusion_, /*extra_diagonal=*/0.0);
        !factored.ok()) {
      return fuse::Failure{"E-BRUSS-V-FACTORIZE", "VDiffusionCore: diffusion factorization failed"};
    }
    auto solved = solver_->solve(rhs);
    if (!solved.ok()) {
      return fuse::Failure{"E-BRUSS-V-SOLVE",
                           "VDiffusionCore: diffusion solve failed or produced a non-finite state"};
    }
    state_ = std::move(solved.value());
    exchange_in_ += dt * global_mean(q_on_v);
    return fuse::Result<void>::success();
  }

  void save_state() {
    saved_state_ = state_;
    saved_exchange_in_ = exchange_in_;
  }
  void restore_state() {
    state_ = saved_state_;
    exchange_in_ = saved_exchange_in_;
  }

  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): `UCore::apply_
  /// diffusion_preconditioner()`'s own comment, mirrored exactly --
  /// `extra_diagonal = 0.0` is `VDiffusionCore::step()`'s own unchanged
  /// convention (this class's own file header: "a SEPARATE choice from
  /// U's own"), not an omission.
  [[nodiscard]] fuse::Result<Vector> apply_diffusion_preconditioner(double dt, const Vector& r) {
    if (const auto factored = solver_->factor(dt, diffusion_, /*extra_diagonal=*/0.0);
        !factored.ok()) {
      return fuse::Failure{"E-BRUSS-V-PRECOND-FACTORIZE",
                           "VDiffusionCore::apply_diffusion_preconditioner: factorization "
                           "failed"};
    }
    auto solved = solver_->solve(r);
    if (!solved.ok()) {
      return fuse::Failure{"E-BRUSS-V-PRECOND-SOLVE",
                           "VDiffusionCore::apply_diffusion_preconditioner: solve failed or "
                           "produced a non-finite result"};
    }
    return solved;
  }

  [[nodiscard]] int n() const { return n_; }
  [[nodiscard]] const Vector& state() const { return state_; }
  [[nodiscard]] double exchange_in() const { return exchange_in_; }
  /// See `UCore::state_mean`: the whole species' mean, a collective.
  [[nodiscard]] double state_mean() const { return global_mean(state_); }

 private:
  /// See `UCore::global_mean`'s own comment -- identical reasoning, a
  /// separate copy because `VDiffusionCore` has no dependency on `UCore`.
  [[nodiscard]] double global_mean(const Vector& owned) const {
    return reducer_->sum(owned.sum()) / static_cast<double>(global_n_);
  }

  int n_;
  int global_n_;
  double diffusion_;
  std::shared_ptr<const fuse::linalg::Reducer> reducer_;
  std::unique_ptr<DiffusionSolver> solver_;
  Vector state_, saved_state_;
  double exchange_in_ = 0.0, saved_exchange_in_ = 0.0;
};

/// V as the participant-ABI shim's own driver sees it: `VDiffusionCore`
/// PLUS the by-hand remap glue `VComponent` (species_component.hpp) also
/// carries -- U-grid-shaped `required`/`provided`, remapped to/from V's
/// own grid with `fuse::apply_weights` before/after `VDiffusionCore::
/// step()`. Needed because the participant graph is Unit 1's own graph
/// (both transfer fields on `u_region`) plus only `add_coupler()`/
/// `remap_on()` (the lead's own ruling, Conformance Unit 2) -- so
/// `fuse_initialize()` hands this shim's caller U-grid-shaped data on
/// "v_in"/"v_out", exactly as it hands `VComponent` the same fields
/// in-process, and V still stands in for the real remap by hand either
/// way. Builds its OWN tiny backend+graph, once, purely to resolve
/// `fuse::build_weights`'s two `RemapPlan`s (exactly `examples/
/// brusselator/main.cpp`'s own "V (source) -> U (target)"/"U (source)
/// -> V (target)" pair) -- this class owns that machinery so neither
/// `species_component.hpp` nor a real participant process needs to
/// duplicate it.
///
/// DECOMPOSED (Halo track): both remaps now route through FUSE's own
/// distributed weights -- `fuse::build_remap_halo` widens each direction's
/// SOURCE mesh with the off-rank cells its own target boundary crosses
/// (docs/dev/native-halo.md), `fuse::build_weights` then intersects only
/// the cells THIS rank holds (owned + adopted), and an injected
/// `GhostExchanger` fills the adopted cells' field VALUES before each
/// `apply_weights` call. TWO DIFFERENT ghost sets exist on two DIFFERENT
/// views here, and must never be confused: `NativeGrid::ghost_pattern()`
/// (brusselator_physics.hpp) is the GRID's own one-row periodic stencil
/// halo, over LOCAL CANONICAL order, that a diffusion solve's own
/// `GhostExchangerFactory` use (species_core.hpp's own comment on that
/// factory); `remap_halo_pattern()` (below) is the REMAP's own adopted-
/// cell halo, over NATIVE MESH local-index order, built fresh per
/// direction from `NativeMeshBackend::remap_halo_pattern`. Neither this
/// class nor `NativeGrid` ever builds one from the other's pattern.
class VParticipant {
 public:
  VParticipant(const VParticipant&) = delete;
  VParticipant& operator=(const VParticipant&) = delete;
  VParticipant(VParticipant&&) = default;
  VParticipant& operator=(VParticipant&&) = default;

  /// `backend`: THIS component's own `fuse::native::NativeMeshBackend`,
  /// constructed AND (whenever `backend.num_ranks() > 1`)
  /// `set_communicator()`-ed by the CALLER -- injected by value, like
  /// `reducer`/`make_exchanger` below, rather than built internally,
  /// specifically so a tier-0 test can pre-install a halo by hand
  /// (`NativeMeshBackend::commit_remap_halo`, the SAME production seam
  /// `test/backend/native_halo_test.cpp`'s own `BackendFixture` uses) and
  /// have this factory SKIP calling the real, MPI-backed
  /// `fuse::build_remap_halo` for a triple that is already built (the
  /// `has_remap_halo` checks below): `fuse::native::build_native_remap_
  /// halo` has NO in-process multi-rank path at all in a build without
  /// MPI (`E-MESH-NO-MPI`) -- the same reason `test/parallel/native_halo_
  /// mpi_test.cpp` is build-verified only here, never run.
  ///
  /// `rank`/`num_ranks` are no longer separate parameters: both are
  /// `backend.rank()`/`backend.num_ranks()`, which `NativeGrid::make`
  /// already reads off the backend it is handed, so repeating them here
  /// could only ever disagree with the backend's own truth, never add
  /// information.
  ///
  /// Split into `make_grids` (just the two `NativeGrid::make` calls) and
  /// `make_from_grids` (graph/halo/weights/exchangers/construct) --
  /// `NativeGrid::make` itself REQUIRES an UNWIDENED view (its own
  /// `E-BRUSS-MESH-COUNT` check: `entity_count(2)` must equal exactly
  /// `nx*owned_rows`, before any halo ever adopts a cell), so a caller
  /// that wants to pre-install a halo by hand (this class's own `make`
  /// doc comment above) must do so AFTER `make_grids` and BEFORE
  /// `make_from_grids` -- a window this one-shot `make` below does not
  /// expose, by design: it always builds BOTH grids first, then the
  /// halo/weights, in that fixed order, exactly like a production caller.
  [[nodiscard]] static fuse::Result<std::pair<NativeGrid, NativeGrid>> make_grids(
      fuse::native::NativeMeshBackend& backend, int nx_u, int ny_u, int nx_v, int ny_v) {
    auto u_grid_r = NativeGrid::make(backend, "u", nx_u, ny_u);
    if (!u_grid_r.ok()) return u_grid_r.error();
    auto v_grid_r = NativeGrid::make(backend, "v", nx_v, ny_v);
    if (!v_grid_r.ok()) return v_grid_r.error();
    return std::make_pair(std::move(u_grid_r.value()), std::move(v_grid_r.value()));
  }

  [[nodiscard]] static fuse::Result<VParticipant> make_from_grids(
      fuse::native::NativeMeshBackend backend, NativeGrid u_grid, NativeGrid v_grid, double diffusion,
      SolverKind solver = SolverKind::Auto, double solver_tolerance = 1e-13,
      std::shared_ptr<const fuse::linalg::Reducer> reducer = nullptr,
      GhostExchangerFactory make_exchanger = nullptr) {
    const fuse::MeshId u_mid = u_grid.mesh_id();
    const fuse::MeshId v_mid = v_grid.mesh_id();
    const int nx_u = u_grid.nx(), ny_u = u_grid.ny(), nx_v = v_grid.nx(), ny_v = v_grid.ny();

    fuse::CouplingGraph g;
    const fuse::MeshId g_u_mesh = g.add_mesh(
        {.name = "u", .source = "generated:quad:" + std::to_string(nx_u) + "x" +
                                 std::to_string(ny_u)});
    const fuse::MeshId g_v_mesh = g.add_mesh(
        {.name = "v", .source = "generated:quad:" + std::to_string(nx_v) + "x" +
                                 std::to_string(ny_v)});
    const fuse::RegionId u_region = g.add_region({.name = "u_all", .mesh = g_u_mesh, .dimension = 2});
    const fuse::RegionId v_region = g.add_region({.name = "v_all", .mesh = g_v_mesh, .dimension = 2});

    fuse::RemapPlan plan_vu;  // V (source) -> U (target)
    plan_vu.src = v_region;
    plan_vu.tgt = u_region;
    fuse::RemapPlan plan_uv;  // U (source) -> V (target)
    plan_uv.src = u_region;
    plan_uv.tgt = v_region;

    // The remap halo, BEFORE anything sizes a buffer from entity_count
    // (remap_plan.hpp's own "ORDER MATTERS" discipline) -- for BOTH
    // directions, since U's own mesh is this participant's SOURCE for one
    // direction and TARGET for the other, and likewise for V's. Skipped
    // per direction when a caller already installed that triple's own
    // halo -- see `make`'s own comment above.
    if (!backend.has_remap_halo(u_mid, v_mid, 2)) {
      auto adopted_uv = fuse::build_remap_halo(plan_uv, g, backend);
      if (!adopted_uv.ok()) return adopted_uv.error();
    }
    if (!backend.has_remap_halo(v_mid, u_mid, 2)) {
      auto adopted_vu = fuse::build_remap_halo(plan_vu, g, backend);
      if (!adopted_vu.ok()) return adopted_vu.error();
    }

    auto w_vu = fuse::build_weights(plan_vu, g, backend);
    if (!w_vu.ok()) return w_vu.error();
    auto w_uv = fuse::build_weights(plan_uv, g, backend);
    if (!w_uv.ok()) return w_uv.error();

    // The remap halo's OWN exchanger pair, one per direction (DIFFERENT
    // patterns/owned counts, so two DIFFERENT exchanger instances, never
    // shared) -- NOT the grid's own `ghost_pattern()`/stencil halo (this
    // class's own file comment above explains why the two must stay
    // separate).
    GhostExchangerFactory exchanger_factory =
        make_exchanger ? make_exchanger : serial_exchanger_factory();
    auto pattern_uv = backend.remap_halo_pattern(u_mid, v_mid, 2);
    if (!pattern_uv.ok()) return pattern_uv.error();
    auto exchanger_uv =
        exchanger_factory(pattern_uv.value(), static_cast<std::size_t>(u_grid.owned_size()));
    if (!exchanger_uv) {
      return fuse::Failure{"E-BRUSS-V-REMAP-EXCHANGER",
                           "VParticipant::make_from_grids: make_exchanger returned null building "
                           "the U->V remap halo's own exchanger (num_ranks() > 1 without "
                           "FUSE_ENABLE_MPI?)"};
    }
    auto pattern_vu = backend.remap_halo_pattern(v_mid, u_mid, 2);
    if (!pattern_vu.ok()) return pattern_vu.error();
    // KNOWN_ISSUES D29 (CLOSED): a decomposition whose U and V bands do not
    // cover the same y-range (e.g. 16 rows as 6/5/5 against 24 as 8/8/8)
    // ADOPTS cells in one or both directions' remap halo. This USED to
    // measurably disagree with the single-rank reference: `step()` and
    // `remap_state_to_u()` sized their `apply_weights` destination buffers
    // from this rank's OWNED count (`n_v_`/`n_u_`, cached before any halo
    // widening) rather than from the weights' own `n_rows` (the TARGET
    // mesh's CURRENT, post-halo local entity count -- larger than owned
    // whenever that same mesh was ALSO widened as the SOURCE of the OTHER
    // transfer direction on this backend). `apply_weights` unconditionally
    // zeros and writes `dst[0, n_rows)`, so the undersized buffer overflowed
    // -- measured as a 3.1e-4 `exchange_in` disagreement from the first
    // sub-step (reading whatever heap memory followed the allocation) and,
    // on later steps, an Eigen `Block.h` bounds assertion once the
    // corrupted heap was read back inside the diffusion solve. Fixed by
    // sizing `q_native_v`/`u_native` from `weights_uv_.n_rows`/
    // `weights_vu_.n_rows` instead (see `step()`/`remap_state_to_u()`'s own
    // comments) -- no refusal is needed any more; this rank's own
    // owned/adopted counts are traced below (`FUSE_BRUSS_DEBUG`) purely as
    // an optional diagnostic, with no decision resting on them.
    if (std::getenv("FUSE_BRUSS_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[D29 rank=%d] make_from_grids: owned_u=%d owned_v=%d adopted_u(uv)=%zu "
                   "adopted_v(vu)=%zu\n",
                   backend.rank(), u_grid.owned_size(), v_grid.owned_size(),
                   pattern_uv.value().owner_rank.size(), pattern_vu.value().owner_rank.size());
    }
    auto exchanger_vu =
        exchanger_factory(pattern_vu.value(), static_cast<std::size_t>(v_grid.owned_size()));
    if (!exchanger_vu) {
      return fuse::Failure{"E-BRUSS-V-REMAP-EXCHANGER",
                           "VParticipant::make_from_grids: make_exchanger returned null building "
                           "the V->U remap halo's own exchanger (num_ranks() > 1 without "
                           "FUSE_ENABLE_MPI?)"};
    }

    return VParticipant(std::move(backend), std::move(u_grid), std::move(v_grid), diffusion, solver,
                        solver_tolerance, std::move(w_vu.value()), std::move(w_uv.value()),
                        std::move(exchanger_uv), std::move(exchanger_vu),
                        reducer ? std::move(reducer) : fuse::linalg::make_serial_reducer(),
                        exchanger_factory);
  }

  [[nodiscard]] static fuse::Result<VParticipant> make(
      fuse::native::NativeMeshBackend backend, int nx_u, int ny_u, int nx_v, int ny_v,
      double diffusion, SolverKind solver = SolverKind::Auto, double solver_tolerance = 1e-13,
      std::shared_ptr<const fuse::linalg::Reducer> reducer = nullptr,
      GhostExchangerFactory make_exchanger = nullptr) {
    auto grids = make_grids(backend, nx_u, ny_u, nx_v, ny_v);
    if (!grids.ok()) return grids.error();
    return make_from_grids(std::move(backend), std::move(grids.value().first),
                           std::move(grids.value().second), diffusion, solver, solver_tolerance,
                           std::move(reducer), std::move(make_exchanger));
  }

  /// `required`: the raw q mean this step, on U's own grid (unremapped,
  /// OWNED-sized: this rank's own `n_u()` band of it), ALREADY what the
  /// caller's own decomposition of U hands over -- a decomposed V never
  /// sees more of U's grid than its own remap needs. On success,
  /// `provided` (size `n_u()`) receives `P_VU(state())`, valid at
  /// t_begin+dt -- exactly `VComponent::advance()`'s own two-sided remap
  /// (species_component.hpp).
  [[nodiscard]] fuse::Result<void> step(double dt, const Vector& required, Vector* provided) {
    const bool trace = std::getenv("FUSE_BRUSS_DEBUG") != nullptr;
    std::vector<double> q_native_owned_u;
    u_grid_.to_native(required, &q_native_owned_u);
    std::vector<double> q_native_u(q_native_owned_u.size() + exchanger_uv_->ghost_size());
    std::copy(q_native_owned_u.begin(), q_native_owned_u.end(), q_native_u.begin());
    auto updated = exchanger_uv_->update(q_native_owned_u.data(), q_native_owned_u.size(),
                                         q_native_u.data() + q_native_owned_u.size(),
                                         exchanger_uv_->ghost_size());
    if (!updated.ok()) return updated.error();

    if (trace) {
      std::fprintf(stderr,
                   "[D29 rank=%d] step: owned_u=%zu ghost_uv=%zu q_native_u.size=%zu "
                   "weights_uv(n_rows=%zu,n_cols=%zu) n_v_=%d\n",
                   backend_.rank(), q_native_owned_u.size(), exchanger_uv_->ghost_size(),
                   q_native_u.size(), weights_uv_.n_rows, weights_uv_.n_cols, n_v_);
      for (std::size_t g = 0; g < std::min<std::size_t>(exchanger_uv_->ghost_size(), 4); ++g) {
        std::fprintf(stderr, "[D29 rank=%d]   ghost_uv[%zu] = %.17g\n", backend_.rank(), g,
                     q_native_u[q_native_owned_u.size() + g]);
      }
    }

    // `weights_uv_.n_rows` is V's CURRENT (post-halo) local entity count --
    // `owned_size(V)` plus whatever V was ALSO widened by as the SOURCE of
    // the OTHER direction's halo (V->U), built on this same backend before
    // this one (KNOWN_ISSUES D29): V's own role as a target here and as a
    // source for remap_state_to_u() are independent widenings of the SAME
    // mesh, and `RemapWeights::n_rows` (remap_plan.hpp: "target entities
    // (local)") always means the FULL local count, owned plus ghost,
    // matching `MeshView::entity_count`'s own "owned plus ghost" contract --
    // never the PRE-halo owned-only count `n_v_` cached at `make_grids()`
    // time. Sizing this buffer from `n_v_` let `apply_weights` (which
    // unconditionally zeros and writes `dst[0, n_rows)`) write past the end
    // of a too-small std::vector whenever V had been widened -- a heap
    // buffer overflow, not merely a wrong value: measured as a 3.1e-4
    // disagreement in exchange_in from the very first sub-step (reading
    // whatever followed the undersized allocation) and, on later steps, an
    // Eigen Block.h bounds assertion once the corrupted heap was read back
    // by the diffusion solve. The extra (ghost) rows this allocates are
    // never read afterward: `to_canonical` only ever addresses native
    // indices below `owned_size(V)`, via `native_of_canonical_`, which is
    // built from the UNWIDENED view at `make_grids()` time and therefore
    // never points past it.
    std::vector<double> q_native_v(weights_uv_.n_rows, 0.0);
    auto applied = fuse::apply_weights(weights_uv_, q_native_u.data(), q_native_v.data(), 1);
    if (!applied.ok()) return applied.error();
    Vector q_on_v(n_v_);
    v_grid_.to_canonical(q_native_v, &q_on_v);

    if (trace) {
      std::fprintf(stderr, "[D29 rank=%d] step: q_on_v.sum=%.17g q_on_v.size=%d\n", backend_.rank(),
                   q_on_v.sum(), n_v_);
    }

    const fuse::Result<void> r = core_->step(dt, q_on_v);
    if (!r.ok()) return r;
    if (trace) {
      std::fprintf(stderr, "[D29 rank=%d] step: exchange_in=%.17g\n", backend_.rank(),
                   core_->exchange_in());
    }
    return remap_state_to_u(provided);
  }

  void save_state() { core_->save_state(); }
  void restore_state() { core_->restore_state(); }

  [[nodiscard]] int n_u() const { return n_u_; }
  [[nodiscard]] const Vector& state() const { return core_->state(); }
  [[nodiscard]] double exchange_in() const { return core_->exchange_in(); }
  [[nodiscard]] double state_mean() const { return core_->state_mean(); }
  /// `P_VU(state())` on U's own grid (this rank's own owned band, size
  /// `n_u()`) -- what `step()` hands back as `provided`, computed from
  /// the CURRENT state. Right after construction that is the initial
  /// P_VU(V0): the value `VComponent::provided_` holds from ITS
  /// constructor (species_component.hpp, `remap_state_to_u(&provided_)`),
  /// which an externally driven V must copy into its registered export
  /// before fuse_initialize() for the first window's t=0 sample to be
  /// V's and not calloc's (shim.h, bruss_v_provided). `fuse::Result`, not
  /// a bare `Vector`, now that the remap it performs (`apply_weights`
  /// over an exchanger-filled, possibly-decomposed source buffer) has
  /// real failure modes a single-rank hand-apply never did.
  [[nodiscard]] fuse::Result<Vector> provided() {
    Vector out(n_u_);
    auto r = remap_state_to_u(&out);
    if (!r.ok()) return r.error();
    return out;
  }

 private:
  VParticipant(fuse::native::NativeMeshBackend backend, NativeGrid u_grid, NativeGrid v_grid,
              double diffusion, SolverKind solver, double solver_tolerance,
              fuse::RemapWeights weights_vu, fuse::RemapWeights weights_uv,
              std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_uv,
              std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_vu,
              std::shared_ptr<const fuse::linalg::Reducer> reducer,
              GhostExchangerFactory make_exchanger)
      : backend_(std::move(backend)),
        u_grid_(std::move(u_grid)),
        v_grid_(std::move(v_grid)),
        n_u_(u_grid_.owned_size()),
        n_v_(v_grid_.owned_size()),
        // heap-allocated: originally because VDiffusionCore held an
        // Eigen::SimplicialLDLT factorization object directly, which Eigen
        // makes neither copyable nor movable, and a direct VDiffusionCore
        // member would therefore have made VParticipant itself immovable.
        // The Grid track's DiffusionSolver seam (above) moved that
        // factorization behind a std::unique_ptr<DiffusionSolver>, so
        // VDiffusionCore is movable on its own merits now too -- kept
        // heap-allocated anyway: bruss_v_create() (shim.cpp) move-
        // constructs the handle's VParticipant from VParticipant::make()'s
        // return, and a second, independent reason to keep the indirection
        // was not worth removing for its own sake in this change.
        core_(std::make_unique<VDiffusionCore>(v_grid_, diffusion, solver, solver_tolerance, reducer,
                                               make_exchanger)),
        weights_vu_(std::move(weights_vu)),
        weights_uv_(std::move(weights_uv)),
        exchanger_uv_(std::move(exchanger_uv)),
        exchanger_vu_(std::move(exchanger_vu)) {}

  /// provided = P_VU(state()): V's own new state (this rank's own owned
  /// band) remapped onto U's own grid (this rank's own owned band).
  [[nodiscard]] fuse::Result<void> remap_state_to_u(Vector* out) {
    std::vector<double> v_native_owned;
    v_grid_.to_native(core_->state(), &v_native_owned);
    std::vector<double> v_native(v_native_owned.size() + exchanger_vu_->ghost_size());
    std::copy(v_native_owned.begin(), v_native_owned.end(), v_native.begin());
    auto updated = exchanger_vu_->update(v_native_owned.data(), v_native_owned.size(),
                                         v_native.data() + v_native_owned.size(),
                                         exchanger_vu_->ghost_size());
    if (!updated.ok()) return updated.error();

    // See `step()`'s own comment on `q_native_v` above for why this is
    // `weights_vu_.n_rows` (U's CURRENT, post-halo local entity count) and
    // not the PRE-halo owned-only `n_u_`: U is widened as the SOURCE of the
    // U->V halo (`exchanger_uv_`'s own pattern) independently of its role as
    // the TARGET here, and the same overflow this fixes there applies here
    // whenever that widening is nonzero.
    std::vector<double> u_native(weights_vu_.n_rows, 0.0);
    auto applied = fuse::apply_weights(weights_vu_, v_native.data(), u_native.data(), 1);
    if (!applied.ok()) return applied.error();
    u_grid_.to_canonical(u_native, out);
    return fuse::Result<void>::success();
  }

  fuse::native::NativeMeshBackend backend_;
  NativeGrid u_grid_, v_grid_;
  int n_u_, n_v_;
  std::unique_ptr<VDiffusionCore> core_;
  fuse::RemapWeights weights_vu_;  ///< V grid -> U grid
  fuse::RemapWeights weights_uv_;  ///< U grid -> V grid
  std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_uv_;  ///< fills U's adopted cells
  std::shared_ptr<const fuse::linalg::GhostExchanger> exchanger_vu_;  ///< fills V's adopted cells
};

}  // namespace brusselator