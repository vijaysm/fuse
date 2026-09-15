#ifndef KARMA_COUPLING_PETSC_FIELD_STORAGE_H
#define KARMA_COUPLING_PETSC_FIELD_STORAGE_H

#include "karma_field_transaction.h"

#include <petscvec.h>

#include <span>
#include <vector>

/** A `FieldStorage` backed by a PETSc `Vec`, restricted to entries the
 * calling rank owns.
 *
 * Deliberately a separate header from `karma_field_transaction.h`: that
 * header was just made free of PETSc (see `karma_field_identity.h`) so a
 * language adapter binding the compliance harness does not inherit a PETSc
 * dependency, and adding a PETSc-typed `FieldStorage` implementation
 * directly there would undo that. Only code that already means to talk to
 * PETSc includes this header.
 *
 * Why owned entries only, rather than the global-index scatter
 * `commit_petsc_values()` used to perform through `VecSetValues` and
 * assembly: that pattern lets several ranks address the same off-process
 * entry with no defined order between them, which is the same last-writer
 * hazard already flagged in SOMA's lowering registry. A `FieldStorage`'s
 * `publish()` is specified as a rank-local operation - stage in private
 * memory, then make it visible - and a write with remote side effects does
 * not fit that contract no matter how it is spelled. `commit_petsc_values()`
 * had exactly one caller in the whole tree, its own single-rank test, and is
 * not covered by the compatibility oracle, so narrowing it costs nothing
 * observable and removes a nondeterminism class outright. A rank that
 * addresses an entry it does not own now gets a named refusal instead of an
 * unordered race.
 */
namespace KARMA
{
class PetscVectorStorage final : public FieldStorage
{
  public:
    /** `global_indices` must already be resolved (e.g. via
     * `KARMADofMap::global_index()`); this class performs no key lookup, only
     * the ownership check. Construction never throws - an index this rank
     * does not own is recorded, not rejected here, so `publish()` and
     * `read_into()` report it through the same `OutcomeReason` every other
     * storage failure uses, keeping one refusal path rather than two. */
    PetscVectorStorage(Vec vector, std::span<const PetscInt> global_indices);

    [[nodiscard]] bool all_owned() const noexcept { return _all_owned; }

    [[nodiscard]] std::size_t value_count() const override { return _indices.size(); }
    OutcomeReason read_into(std::span<Real> destination) const override;
    OutcomeReason publish(std::span<const Real> values) override;

  private:
    Vec _vector;
    std::vector<PetscInt> _indices;
    bool _all_owned = false;
};
} // namespace KARMA

#endif
