#include "karma_fem_field_evaluator.h"
#include "karma_fem_transfer.h"
#include "karma_distributed_point_locator.h"
#include "karma_system.h"

#include <petscvec.h>

#include <cmath>
#include <array>
#include <iostream>
#include <vector>

namespace
{
struct PetscSession
{
    PetscSession(int& argc, char**& argv) { PetscInitialize(&argc, &argv, nullptr, nullptr); }
    ~PetscSession() { PetscFinalize(); }
};

void check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(operation);
}

double exact_cell_average(double x0, double x1, double y0, double y1)
{
    const double area = (x1 - x0) * (y1 - y0);
    const double x_integral = (x1 * x1 * x1 - x0 * x0 * x0) / 3.0 * (y1 - y0);
    const double y_integral = (y1 * y1 * y1 - y0 * y0 * y0) / 3.0 * (x1 - x0);
    return (x_integral + y_integral) / area;
}

double run_case(unsigned cells, MPI_Comm communicator)
{
    // Keep each rank's generated fixture self-contained; the MPI reduction below
    // verifies identical convergence while distributed partition fixtures cover routing separately.
    KARMA::KARMAMesh source_mesh(2, MPI_COMM_SELF);
    KARMA::KARMAMesh target_mesh(2, MPI_COMM_SELF);
    source_mesh.generate_box(cells, cells, 0, KARMA::Point(0.0, 0.0, 0.0), KARMA::Point(1.0, 1.0, 0.0));
    target_mesh.generate_box(cells, cells, 0, KARMA::Point(0.0, 0.0, 0.0), KARMA::Point(1.0, 1.0, 0.0));

    KARMA::KARMASystem source_system(source_mesh, "quadratic_source");
    const KARMA::VariableSpec source_spec{"u", KARMA::FieldBasisFamily::LagrangeNodal, 1,
                                          KARMA::BasisContinuity::cG, 1};
    source_system.add_variable(source_spec);
    source_system.init();
    for (const auto vertex : source_mesh.vertices())
    {
        double coordinates[3]{};
        if (source_mesh.moab()->get_coords(&vertex, 1, coordinates) != moab::MB_SUCCESS)
            throw std::runtime_error("failed to read source vertex coordinates");
        const PetscInt index = source_system.dof_map().vertex_dof(vertex, 0, 0);
        const PetscScalar value = coordinates[0] * coordinates[0] + coordinates[1] * coordinates[1];
        check(VecSetValue(source_system.solution(), index, value, INSERT_VALUES), "VecSetValue(source)");
    }
    check(VecAssemblyBegin(source_system.solution()), "VecAssemblyBegin(source)");
    check(VecAssemblyEnd(source_system.solution()), "VecAssemblyEnd(source)");

    KARMA::PetscFieldBinding source_binding{&source_system, source_system.solution(), 0, source_spec,
                                             KARMA::FieldScalarRepresentation::PetscScalar};
    const KARMA::KARMAField source_field(source_binding);
    const KARMA::KARMAFEMFieldEvaluator evaluator(source_field);
    KARMA::KARMAMoabPointLocator locator(source_mesh);
    locator.setup();
    KARMA::KARMABasisRegistry target_registry;
    const auto target_basis = target_registry.create(
        {KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::LagrangeNodal, 0,
         KARMA::BasisContinuity::dG});

    double local_squared_error = 0.0;
    double local_count = 0.0;
    for (const auto element : target_mesh.owned_elements())
    {
        const auto projection = KARMA::KARMAFEMTransfer::project_element(
            target_mesh, element, target_basis, 4, locator, evaluator, 1, true);
        const auto center = target_mesh.centroid(element);
        const double h = 1.0 / static_cast<double>(cells);
        const double expected = exact_cell_average(center(0) - 0.5 * h, center(0) + 0.5 * h,
                                                   center(1) - 0.5 * h, center(1) + 0.5 * h);
        const double error = projection.coefficients.front() - expected;
        local_squared_error += h * h * error * error;
        local_count += 1.0;
    }
    double global_squared_error = 0.0;
    double global_count = 0.0;
    MPI_Allreduce(&local_squared_error, &global_squared_error, 1, MPI_DOUBLE, MPI_SUM, communicator);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_DOUBLE, MPI_SUM, communicator);
    if (global_count == 0.0) throw std::runtime_error("no target elements were projected");
    return std::sqrt(global_squared_error);
}
}

int main(int argc, char** argv)
{
    PetscSession petsc(argc, argv);
    MPI_Comm communicator = PETSC_COMM_WORLD;
    try
    {
        const double coarse = run_case(1, communicator);
        const double medium = run_case(2, communicator);
        const double fine = run_case(4, communicator);
        const double finest = run_case(8, communicator);
        const double rate_one = std::log(coarse / medium) / std::log(2.0);
        const double rate_two = std::log(medium / fine) / std::log(2.0);
        const double rate_three = std::log(fine / finest) / std::log(2.0);
        int rank = 0;
        MPI_Comm_rank(communicator, &rank);
        if (rank == 0)
            std::cout << "FV projection errors: " << coarse << ' ' << medium << ' ' << fine << ' ' << finest
                      << ", rates: " << rate_one << ' ' << rate_two << ' ' << rate_three << '\n';
        return rate_one > 1.5 && rate_two > 1.5 && rate_three > 1.5 ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
