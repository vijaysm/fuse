#include "karma_fem_field_evaluator.h"
#include "karma_fem_transfer.h"
#include "karma_distributed_point_locator.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <cstring>
#include <map>
#include <vector>

namespace
{
void check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(operation);
}

std::map<std::uint64_t, double> run(MPI_Comm communicator)
{
    // This baseline uses replicated self-contained fixtures. Partitioned H5M
    // serial-reference comparison is a separate fixture TODO, not implied here.
    KARMA::KARMAMesh source_mesh(2, MPI_COMM_SELF);
    KARMA::KARMAMesh target_mesh(2, MPI_COMM_SELF);
    source_mesh.generate_box(2, 2, 0);
    target_mesh.generate_box(2, 2, 0);
    KARMA::KARMASystem source(source_mesh, "source");
    const KARMA::VariableSpec spec{"u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 1};
    source.add_variable(spec);
    source.init();
    for (const auto vertex : source_mesh.vertices())
    {
        double coordinates[3]{};
        source_mesh.moab()->get_coords(&vertex, 1, coordinates);
        const PetscInt index = source.dof_map().vertex_dof(vertex, 0);
        check(VecSetValue(source.solution(), index, coordinates[0] + 2.0 * coordinates[1], INSERT_VALUES), "source value");
    }
    check(VecAssemblyBegin(source.solution()), "source assembly begin");
    check(VecAssemblyEnd(source.solution()), "source assembly end");
    KARMA::KARMAField field(KARMA::PetscFieldBinding{&source, source.solution(), 0, spec,
                                                      KARMA::FieldScalarRepresentation::PetscScalar});
    KARMA::KARMAFEMFieldEvaluator evaluator(field);
    KARMA::KARMADistributedPointLocator locator(source_mesh);
    locator.setup();
    KARMA::KARMABasisRegistry registry;
    const auto basis = registry.create({KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::LagrangeNodal,
                                        0, KARMA::BasisContinuity::dG});
    std::vector<moab::EntityHandle> elements(target_mesh.elements().begin(), target_mesh.elements().end());
    const auto projected = KARMA::KARMAFEMTransfer::project_elements(target_mesh, elements, basis, 4, locator,
                                                                       evaluator, 1, true);
    std::map<std::uint64_t, double> result;
    for (std::size_t index = 0; index < elements.size(); ++index)
        result[static_cast<std::uint64_t>(target_mesh.global_id(elements[index]))] = projected[index].coefficients.front();
    (void)communicator;
    return result;
}
}

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int rank = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    int result = 0;
    try
    {
        const auto reference = run(PETSC_COMM_WORLD);
        const int count = static_cast<int>(reference.size());
        int broadcast_count = count;
        MPI_Bcast(&broadcast_count, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        std::vector<std::uint64_t> keys(static_cast<std::size_t>(broadcast_count));
        std::vector<double> values(static_cast<std::size_t>(broadcast_count));
        if (rank == 0)
        {
            int index = 0;
            for (const auto& [key, value] : reference)
            {
                keys[static_cast<std::size_t>(index)] = key;
                values[static_cast<std::size_t>(index)] = value;
                ++index;
            }
        }
        MPI_Bcast(keys.data(), broadcast_count, MPI_UINT64_T, 0, PETSC_COMM_WORLD);
        MPI_Bcast(values.data(), broadcast_count, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
        bool exact = reference.size() == static_cast<std::size_t>(broadcast_count);
        for (int index = 0; exact && index < broadcast_count; ++index)
        {
            const auto found = reference.find(keys[static_cast<std::size_t>(index)]);
            exact = found != reference.end() && std::memcmp(&found->second, &values[static_cast<std::size_t>(index)],
                                                             sizeof(double)) == 0;
        }
        int local = exact ? 1 : 0;
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD);
        if (rank == 0 && !global) std::cerr << "stable-key serial reference comparison failed\n";
        result = global ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    PetscFinalize();
    return result;
}
