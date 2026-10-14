#pragma once

/// What the Krylov solvers (pcg.hpp, gmres.hpp) need from "a matrix": one
/// method, `apply(x, y)` computing `y <- A x`, taking `DistVector&`
/// rather than `const DistVector&` for `x` because `DistSparseMatrix::apply`
/// (dist_sparse_matrix.hpp) calls `x.update_ghosts()` first -- COLLECTIVE
/// under MPI, so `apply` on the matrix stays `const` while the vector it
/// reads is not. `LinearOperator` below is a structural (duck-typed)
/// concept over exactly that shape, not a base class: `DistSparseMatrix`
/// satisfies it without naming it, the same way `fuse::search`'s own
/// mesh adapter concept (docs/dev/search.md) lets a type below `backend`
/// be consumed from a layer that cannot include it.
///
/// `MatrixFreeOperator` is the other side of the same concept: a Jacobian-
/// free Newton-Krylov step needs `y <- J(u) v` without ever assembling `J`,
/// so it wraps a caller-supplied callable instead of a `DistSparseMatrix`.
/// Both are accepted by `pcg`/`gmres` through the same template parameter,
/// with no runtime dispatch and no common base class.

#include <concepts>
#include <cstddef>
#include <functional>
#include <utility>

#include "fuse/base/result.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace fuse::linalg {

/// One iterative-solve run's outcome: how many steps it took, the relative
/// residual norm the stopping test was decided from (ALWAYS the true
/// residual recomputed by a fresh `apply`, never the cheaper per-step
/// recurrence a solver also keeps -- pcg.hpp's and gmres.hpp's own header
/// comments say why that distinction matters), and whether it actually
/// converged. Returned alongside the solution rather than folded into
/// `Result`'s success value alone, so a caller that logs convergence
/// history does not have to re-derive the residual norm itself.
struct SolverReport {
  std::size_t iterations = 0;
  double residual_norm = 0.0;
  bool converged = false;
};

/// Any `T` for which `op.apply(x, y)` computes `y <- A x` and reports
/// failure (a communication failure inside a ghost update, the one outcome
/// `DistSparseMatrix::apply` returns a `Failure` for) through `Result<void>`
/// rather than a thrown exception -- the same execution-boundary contract
/// `fuse::Result`'s own header comment describes.
///
/// UNIFORM-FAILURE PRECONDITION (Review 2026-10-11 NL-6). `pcg`/`gmres`
/// return on the FIRST `!apply(...).ok()` with no cross-rank
/// reconciliation of that decision -- unlike `DistSparseMatrix::apply`'s
/// own refusals (I2-audited: every one is either a rank-identical
/// function of locally held data, agreed via `ranks_not_ok` before any
/// collective it guards, or has no later collective in the call), an
/// implementation of this concept whose `apply` can refuse on SOME ranks
/// of a call and succeed on OTHERS makes the solver's failing rank skip
/// every later `dot`/`norm2` collective its peers still enter --
/// rank-agreement.md's own hang class. `MatrixFreeOperator` below wraps
/// an arbitrary caller callable (a residual function's action), which
/// CAN refuse on one rank alone; every operator this library builds
/// itself only fails on an MPI error (process-fatal in practice), so
/// this is latent, not exercised, today. A caller supplying a
/// `MatrixFreeOperator` whose wrapped callable can fail data-dependently
/// must make that failure decision identically on every rank before
/// returning it (the same discipline `schur_slab.cpp`'s own `factor()`
/// uses for its own pre-collective refusals) -- this concept does not,
/// and structurally cannot, enforce that itself.
template <class T>
concept LinearOperator = requires(const T& op, DistVector& x, DistVector& y) {
  { op.apply(x, y) } -> std::same_as<Result<void>>;
};

/// The Jacobian-free hook: wraps a callable with the same `apply(x, y)`
/// shape `DistSparseMatrix` has, for a caller that computes `y <- J(u) v`
/// from a residual function's action rather than an assembled matrix (a
/// finite-difference or analytically linearised directional derivative).
/// Satisfies `LinearOperator` structurally, with no inheritance -- `pcg`
/// and `gmres` take either one through the same template parameter.
class MatrixFreeOperator {
 public:
  using ApplyFn = std::function<Result<void>(DistVector&, DistVector&)>;

  explicit MatrixFreeOperator(ApplyFn fn) : fn_(std::move(fn)) {}

  [[nodiscard]] Result<void> apply(DistVector& x, DistVector& y) const { return fn_(x, y); }

 private:
  ApplyFn fn_;
};

}  // namespace fuse::linalg
