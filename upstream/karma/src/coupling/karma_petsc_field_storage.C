#include "karma_petsc_field_storage.h"

#include "petsc_raii.h"

#include <cmath>

namespace KARMA
{
PetscVectorStorage::PetscVectorStorage(Vec vector, std::span<const PetscInt> global_indices)
    : _vector(vector), _indices(global_indices.begin(), global_indices.end())
{
    PetscInt low = 0;
    PetscInt high = 0;
    // A failed range query leaves _all_owned false, same as an out-of-range
    // index: either way publish() and read_into() must refuse.
    if (VecGetOwnershipRange(_vector, &low, &high) != PETSC_SUCCESS) return;
    _all_owned = true;
    for (const auto index : _indices)
        if (index < low || index >= high)
        {
            _all_owned = false;
            break;
        }
}

OutcomeReason PetscVectorStorage::read_into(std::span<Real> destination) const
{
    if (!_all_owned) return OutcomeReason::InvalidArgument;
    if (destination.size() != _indices.size()) return OutcomeReason::InvalidArgument;
    if (_indices.empty()) return OutcomeReason::Success;

    std::vector<PetscScalar> values(_indices.size());
    // Every index is locally owned (checked at construction), so this reads
    // with no off-process communication.
    if (VecGetValues(_vector, static_cast<PetscInt>(_indices.size()), _indices.data(), values.data()) !=
        PETSC_SUCCESS)
        return OutcomeReason::StorageFailure;
    for (std::size_t entry = 0; entry != values.size(); ++entry) destination[entry] = static_cast<Real>(values[entry]);
    return OutcomeReason::Success;
}

OutcomeReason PetscVectorStorage::publish(std::span<const Real> values)
{
    if (!_all_owned) return OutcomeReason::InvalidArgument;
    if (values.size() != _indices.size()) return OutcomeReason::InvalidArgument;
    for (const auto value : values)
        if (!std::isfinite(value)) return OutcomeReason::NonfiniteValue;
    if (_indices.empty()) return OutcomeReason::Success;

    std::vector<PetscScalar> staged(values.size());
    for (std::size_t entry = 0; entry != values.size(); ++entry) staged[entry] = static_cast<PetscScalar>(values[entry]);

    // Stage into a duplicate and only copy back on success, so a caller never
    // observes a partially published commit - the same all-or-nothing shape
    // commit_petsc_values() always had, just now expressed once instead of at
    // every call site that touches a PETSc-backed field.
    PetscOwner<Vec, VecDestroy> staged_vector;
    if (VecDuplicate(_vector, staged_vector.put()) != PETSC_SUCCESS ||
        VecCopy(_vector, staged_vector.get()) != PETSC_SUCCESS)
        return OutcomeReason::StorageFailure;
    if (VecSetValues(staged_vector.get(), static_cast<PetscInt>(_indices.size()), _indices.data(), staged.data(),
                     INSERT_VALUES) != PETSC_SUCCESS)
        return OutcomeReason::StorageFailure;
    if (VecAssemblyBegin(staged_vector.get()) != PETSC_SUCCESS || VecAssemblyEnd(staged_vector.get()) != PETSC_SUCCESS)
        return OutcomeReason::StorageFailure;
    if (VecCopy(staged_vector.get(), _vector) != PETSC_SUCCESS) return OutcomeReason::StorageFailure;
    return OutcomeReason::Success;
}
} // namespace KARMA
