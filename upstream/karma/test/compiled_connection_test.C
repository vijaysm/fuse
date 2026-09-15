#include "karma_compiled_fem_connection.h"
#include "karma_distributed_point_locator.h"
#include "karma_fem_field_evaluator.h"
#include "karma_fem_transfer.h"
#include "karma_mesh.h"
#include "karma_system.h"

#include <petsc.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(operation);
}

KARMA::VariableSpec scalar_cg_spec()
{
    return {"u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 1};
}

void set_source_values(KARMA::KARMASystem& system, double offset, double x_scale, double y_scale)
{
    check(VecSet(system.solution(), 0.0), "zero source values");
    for (const auto vertex : system.mesh().vertices())
    {
        double coordinates[3]{};
        require(system.mesh().moab()->get_coords(&vertex, 1, coordinates) == moab::MB_SUCCESS,
                "read source coordinates");
        const PetscInt index = system.dof_map().vertex_dof(vertex, 0, 0);
        const PetscScalar value = offset + x_scale * coordinates[0] + y_scale * coordinates[1];
        check(VecSetValue(system.solution(), index, value, INSERT_VALUES), "set source value");
    }
    check(VecAssemblyBegin(system.solution()), "assemble source values begin");
    check(VecAssemblyEnd(system.solution()), "assemble source values end");
}

void require_same_vector(Vec actual, Vec expected)
{
    Vec difference = nullptr;
    check(VecDuplicate(actual, &difference), "duplicate comparison vector");
    check(VecCopy(actual, difference), "copy comparison vector");
    check(VecAXPY(difference, -1.0, expected), "subtract comparison vector");
    PetscReal norm = 0.0;
    check(VecNorm(difference, NORM_INFINITY, &norm), "norm comparison vector");
    check(VecDestroy(&difference), "destroy comparison vector");
    require(std::isfinite(norm) && norm < 1.0e-11, "compiled projection differs from one-shot projection");
}
}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        KARMA::KARMAMesh source_mesh(2, PETSC_COMM_WORLD);
        KARMA::KARMAMesh target_mesh(2, PETSC_COMM_WORLD);
        source_mesh.generate_box(2, 3, 0);
        target_mesh.generate_box(3, 2, 0);

        const auto specification = scalar_cg_spec();
        KARMA::KARMASystem source_system(source_mesh, "compiled-connection-source");
        source_system.add_variable(specification);
        source_system.init();
        KARMA::KARMASystem target_system(target_mesh, "compiled-connection-target");
        target_system.add_variable(specification);
        target_system.init();

        set_source_values(source_system, 1.0, 0.25, -0.5);
        KARMA::KARMAField source_field(KARMA::PetscFieldBinding{
            &source_system, source_system.solution(), 0, specification,
            KARMA::FieldScalarRepresentation::PetscScalar});
        KARMA::CompiledFEMConnection connection(source_field, target_system, 0, 5);
        connection.setup();
        require(connection.setup_count() == 1, "compiled connection setup count is not one");
        require(connection.located_point_count() > 0, "compiled connection did not cache point locations");
        const void* located_points = connection.located_points_handle();

        connection.apply(source_field);
        const std::array<const void*, 5> handles{
            connection.mass_handle(), connection.rhs_handle(), connection.solution_handle(),
            connection.residual_handle(), connection.ksp_handle()};

        const std::array<std::array<double, 3>, 3> fields{{
            {{2.0, -0.75, 0.5}}, {{-1.0, 1.25, 0.125}}, {{0.5, -0.25, -1.5}}}};
        for (const auto& coefficients : fields)
        {
            set_source_values(source_system, coefficients[0], coefficients[1], coefficients[2]);
            connection.apply(source_field);
            require(connection.setup_count() == 1, "compiled connection repeated setup during apply");
            require(located_points == connection.located_points_handle(),
                    "compiled connection replaced cached point locations during apply");
            require(handles[0] == connection.mass_handle() && handles[1] == connection.rhs_handle() &&
                        handles[2] == connection.solution_handle() && handles[3] == connection.residual_handle() &&
                        handles[4] == connection.ksp_handle(),
                    "compiled connection replaced a persistent PETSc object");

            KARMA::KARMADistributedPointLocator locator(source_mesh);
            locator.setup();
            KARMA::KARMAFEMFieldEvaluator evaluator(source_field);
            Vec reference = target_system.create_vector();
            KARMA::KARMAFEMTransfer::project_cg(target_system, 0, 5, locator, evaluator, reference);
            require_same_vector(connection.staging(), reference);
            check(VecDestroy(&reference), "destroy one-shot projection");
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
