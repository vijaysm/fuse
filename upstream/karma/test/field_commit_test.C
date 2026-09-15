#include "karma_field.h"
#include "karma_petsc_field_storage.h"

#include <petsc.h>

#include <iostream>
#include <limits>
#include <vector>

namespace
{
/** PetscVectorStorage's own boundary check, exercised directly rather than
 * through commit_petsc_values: a single-rank run owns every registered
 * DofKey, so there is no way to make global_index() hand back a key this
 * rank does not own, and that is exactly why the storage class checks for
 * itself instead of trusting a caller who may run with more ranks than this
 * test does. A fixture that must fail the check, per the owned-entry precondition. */
void refuses_an_unowned_index(Vec vector)
{
    PetscInt low = 0;
    PetscInt high = 0;
    if (VecGetOwnershipRange(vector, &low, &high) != PETSC_SUCCESS)
        throw std::runtime_error("could not read the ownership range for the refusal case");

    // One index this rank owns, one it provably does not (one past the
    // owned range - valid for some rank in a larger communicator, never for
    // this one).
    const std::vector<PetscInt> mixed{low, high};
    KARMA::PetscVectorStorage storage(vector, mixed);
    if (storage.all_owned())
        throw std::runtime_error("PetscVectorStorage accepted an index outside its ownership range");

    const std::vector<KARMA::Real> values{1.0, 2.0};
    if (storage.publish(values) != KARMA::OutcomeReason::InvalidArgument)
        throw std::runtime_error("publish() did not refuse an unowned index");

    std::vector<KARMA::Real> destination(2, 0.0);
    if (storage.read_into(destination) != KARMA::OutcomeReason::InvalidArgument)
        throw std::runtime_error("read_into() did not refuse an unowned index");

    // The owned-only counterpart must still work: refusing an unowned index
    // must not make every construction unusable.
    const std::vector<PetscInt> owned_only{low};
    KARMA::PetscVectorStorage owned_storage(vector, owned_only);
    if (!owned_storage.all_owned()) throw std::runtime_error("an entirely owned index set was refused");
}

/** PetscVectorStorage::publish()'s own finiteness check, exercised directly.
 *
 * This needs its own case because both routes that normally reach publish()
 * screen nonfinite values first - commit_petsc_values() throws on one before
 * building a transaction at all, and FieldTransaction::stage() refuses one
 * before it can ever be committed. That makes the check inside publish()
 * unreachable through either of them, and an earlier mutation run confirmed
 * it: deleting the check killed no test. It is not dead code, though -
 * PetscVectorStorage is a public FieldStorage a caller may drive directly,
 * and this is the layer that owns the PETSc vector, so it is the last place
 * a NaN can be stopped before it lands in one. Defense that nothing
 * exercises is indistinguishable from defense that does not work.
 */
void refuses_a_nonfinite_publish(Vec vector)
{
    PetscInt low = 0;
    PetscInt high = 0;
    if (VecGetOwnershipRange(vector, &low, &high) != PETSC_SUCCESS)
        throw std::runtime_error("could not read the ownership range for the nonfinite case");

    const std::vector<PetscInt> owned{low};
    KARMA::PetscVectorStorage storage(vector, owned);
    if (!storage.all_owned()) throw std::runtime_error("the owned index set for the nonfinite case was refused");

    PetscScalar before = 0.0;
    if (VecGetValues(vector, 1, &low, &before) != PETSC_SUCCESS)
        throw std::runtime_error("could not read the pre-publish value");

    for (const KARMA::Real bad : {std::numeric_limits<KARMA::Real>::quiet_NaN(),
                                  std::numeric_limits<KARMA::Real>::infinity(),
                                  -std::numeric_limits<KARMA::Real>::infinity()})
    {
        const std::vector<KARMA::Real> values{bad};
        if (storage.publish(values) != KARMA::OutcomeReason::NonfiniteValue)
            throw std::runtime_error("publish() did not refuse a nonfinite value");
    }

    // A refused publish must leave the vector exactly as it was: reporting
    // the refusal but having already written is the worse of the two
    // failures, because the caller has no reason to look.
    PetscScalar after = 0.0;
    if (VecGetValues(vector, 1, &low, &after) != PETSC_SUCCESS)
        throw std::runtime_error("could not read the post-publish value");
    if (PetscRealPart(after) != PetscRealPart(before))
        throw std::runtime_error("a refused nonfinite publish still modified the vector");

    // And a finite publish through the same storage must still succeed, so
    // the refusal is specific to the value rather than jamming the object.
    const std::vector<KARMA::Real> good{42.0};
    if (storage.publish(good) != KARMA::OutcomeReason::Success)
        throw std::runtime_error("a finite publish was refused after a nonfinite one");
    PetscScalar published = 0.0;
    if (VecGetValues(vector, 1, &low, &published) != PETSC_SUCCESS || PetscRealPart(published) != 42.0)
        throw std::runtime_error("the finite publish did not reach the vector");
}
} // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int result = 0;
    try
    {
        KARMA::KARMAMesh mesh(1, PETSC_COMM_WORLD);
        mesh.generate_box(1, 0, 0);
        KARMA::KARMASystem system(mesh, "commit");
        const KARMA::VariableSpec specification{"u", KARMA::FieldBasisFamily::LagrangeNodal, 1,
                                                KARMA::BasisContinuity::cG, 1};
        system.add_variable(specification);
        system.init();
        if (VecSet(system.solution(), 1.0) != PETSC_SUCCESS)
            throw std::runtime_error("failed to initialize commit test vector");
        if (VecAssemblyBegin(system.solution()) != PETSC_SUCCESS || VecAssemblyEnd(system.solution()) != PETSC_SUCCESS)
            throw std::runtime_error("failed to assemble commit test vector");
        KARMA::KARMAField field(KARMA::PetscFieldBinding{
            &system, system.solution(), 0, specification, KARMA::FieldScalarRepresentation::PetscScalar});
        const auto owned = system.dof_map().owned_key_indices();
        if (owned.empty()) throw std::runtime_error("commit test has no owned DoFs");
        const std::pair<KARMA::DofKey, KARMA::Real> update{owned.front().first, 7.0};
        field.commit_petsc_values(std::span<const std::pair<KARMA::DofKey, KARMA::Real>>(&update, 1));
        PetscScalar value = 0.0;
        if (VecGetValues(system.solution(), 1, &owned.front().second, &value) != PETSC_SUCCESS ||
            PetscRealPart(value) != 7.0)
            throw std::runtime_error("transactional PETSc commit did not publish the staged value");

        refuses_an_unowned_index(system.solution());
        refuses_a_nonfinite_publish(system.solution());
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    PetscFinalize();
    return result;
}
