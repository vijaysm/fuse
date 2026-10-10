// src/linalg/schur_slab_unavailable.cpp -- SchurSlabSolver when
// FUSE_LINALG_BACKEND is not "eigen".
//
// The solver's factorisations (SimplicialLDLT on each band's interior,
// dense LDLT on the assembled Schur complement; docs/dev/schur-slab-solver.md)
// have no built-in implementation, so a build without Eigen keeps the class
// and refuses at factor(): the caller learns by a named code that this
// configuration cannot run a decomposed direct solve, instead of fuse::linalg
// failing to build at all. solve() on a never-factored solver is the
// constructor-private class's only other state, and is unreachable.
#include "fuse/linalg/schur_slab.hpp"

#include "fuse/base/error.hpp"

namespace fuse::linalg {

struct SchurSlabSolver::Impl {};

SchurSlabSolver::SchurSlabSolver() = default;
SchurSlabSolver::SchurSlabSolver(SchurSlabSolver&&) noexcept = default;
SchurSlabSolver& SchurSlabSolver::operator=(SchurSlabSolver&&) noexcept = default;
SchurSlabSolver::~SchurSlabSolver() = default;

Result<SchurSlabSolver> SchurSlabSolver::factor(int, std::int64_t, std::int64_t, std::int64_t,
                                                double, const RowOwnership&,
                                                std::shared_ptr<const Reducer>) {
  return Failure{"E-LINALG-SCHUR-BACKEND",
                 "SchurSlabSolver needs the Eigen linear-algebra backend "
                 "(FUSE_LINALG_BACKEND=eigen); this build selected 'builtin'"};
}

Result<void> SchurSlabSolver::solve(const DistVector&, DistVector&) const {
  return Failure{"E-LINALG-SCHUR-BACKEND",
                 "SchurSlabSolver was never factored: this build has no Eigen backend"};
}

}  // namespace fuse::linalg
