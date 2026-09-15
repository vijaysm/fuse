#include "generic_mass_operator.h"
#include "karma_distributed_point_locator.h"
#include "karma_fe_base.h"
#include "karma_fem_field_evaluator.h"
#include "karma_fem_transfer.h"
#include "karma_geometry_map.h"
#include "karma_moab_io.h"
#include "karma_system.h"
#include "karma_transfer_diagnostics.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using FieldFunction = std::function<Real(const KARMA::Point&)>;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void petsc_check(PetscErrorCode error, const char* operation)
{
    if (error != PETSC_SUCCESS) throw std::runtime_error(operation);
}

KARMA::VariableSpec scalar_cg_spec()
{
    return {"u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 1};
}

void interpolate(KARMA::KARMASystem& system, const FieldFunction& field)
{
    petsc_check(VecSet(system.solution(), 0.0), "zero interpolation vector");
    for (const auto vertex : system.mesh().vertices())
    {
        double coordinates[3]{};
        require(system.mesh().moab()->get_coords(&vertex, 1, coordinates) == moab::MB_SUCCESS,
                "read interpolation coordinates");
        const PetscInt dof = system.dof_map().vertex_dof(vertex, 0, 0);
        petsc_check(VecSetValue(system.solution(), dof, field(KARMA::Point(coordinates)), INSERT_VALUES),
                    "set interpolation value");
    }
    petsc_check(VecAssemblyBegin(system.solution()), "begin interpolation assembly");
    petsc_check(VecAssemblyEnd(system.solution()), "end interpolation assembly");
}

Real maximum_nodal_error(const KARMA::KARMASystem& system, Vec values, const FieldFunction& exact)
{
    petsc_check(VecGhostUpdateBegin(values, INSERT_VALUES, SCATTER_FORWARD), "begin projected ghost update");
    petsc_check(VecGhostUpdateEnd(values, INSERT_VALUES, SCATTER_FORWARD), "end projected ghost update");
    Vec local_form = nullptr;
    petsc_check(VecGhostGetLocalForm(values, &local_form), "get projected local form");
    const PetscScalar* local_values = nullptr;
    petsc_check(VecGetArrayRead(local_form, &local_values), "read projected local form");
    const auto& local_to_global = system.dof_map().local_to_global_indices();
    Real local_error = 0.0;
    for (const auto vertex : system.mesh().vertices())
    {
        double coordinates[3]{};
        require(system.mesh().moab()->get_coords(&vertex, 1, coordinates) == moab::MB_SUCCESS,
                "read error coordinates");
        const PetscInt dof = system.dof_map().vertex_dof(vertex, 0, 0);
        const auto found = std::find(local_to_global.begin(), local_to_global.end(), dof);
        require(found != local_to_global.end(), "projected vertex DoF is not locally visible");
        const auto local = static_cast<std::size_t>(std::distance(local_to_global.begin(), found));
        local_error = std::max(local_error,
                               std::abs(PetscRealPart(local_values[local]) - exact(KARMA::Point(coordinates))));
    }
    petsc_check(VecRestoreArrayRead(local_form, &local_values), "restore projected local form array");
    petsc_check(VecGhostRestoreLocalForm(values, &local_form), "restore projected local form");
    Real global_error = 0.0;
    petsc_check(MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, system.communicator()),
                "reduce nodal error");
    return global_error;
}

KARMA::PetscOwner<Vec, VecDestroy> project(KARMA::KARMASystem& source, KARMA::KARMASystem& target,
                                             const FieldFunction& field)
{
    interpolate(source, field);
    const auto specification = scalar_cg_spec();
    const KARMA::KARMAField source_field(KARMA::PetscFieldBinding{
        &source, source.solution(), 0, specification, KARMA::FieldScalarRepresentation::PetscScalar});
    const KARMA::KARMAFEMFieldEvaluator evaluator(source_field);
    KARMA::KARMADistributedPointLocator locator(source.mesh());
    locator.setup();
    KARMA::PetscOwner<Vec, VecDestroy> projected;
    *projected.put() = target.create_vector();
    KARMA::KARMAFEMTransfer::project_cg(target, 0, 5, locator, evaluator, projected.get());
    return projected;
}

Real project_and_measure(KARMA::KARMASystem& source, KARMA::KARMASystem& target,
                         const FieldFunction& field)
{
    const auto projected = project(source, target, field);
    return maximum_nodal_error(target, projected.get(), field);
}

Real global_integral(const KARMA::KARMASystem& system, Vec field)
{
    KARMA::PetscOwner<Mat, MatDestroy> mass;
    *mass.put() = system.create_matrix();
    KARMA::assemble_mass_matrix(system, mass.get(), nullptr, 0, 5);
    KARMA::PetscOwner<Vec, VecDestroy> ones;
    *ones.put() = system.create_vector();
    petsc_check(VecSet(ones.get(), 1.0), "set integral test function");
    PetscScalar integral = 0.0;
    petsc_check(MatMult(mass.get(), field, ones.get()), "apply consistent mass for integral");
    petsc_check(VecSum(ones.get(), &integral), "sum consistent-mass integral");
    return PetscRealPart(integral);
}

Real l2_difference(const KARMA::KARMASystem& system, Vec left, Vec right)
{
    KARMA::PetscOwner<Vec, VecDestroy> difference;
    *difference.put() = system.create_vector();
    petsc_check(VecWAXPY(difference.get(), -1.0, right, left), "form field difference");
    KARMA::PetscOwner<Mat, MatDestroy> mass;
    *mass.put() = system.create_matrix();
    KARMA::assemble_mass_matrix(system, mass.get(), nullptr, 0, 5);
    KARMA::PetscOwner<Vec, VecDestroy> weighted;
    *weighted.put() = system.create_vector();
    petsc_check(MatMult(mass.get(), difference.get(), weighted.get()), "weight field difference");
    PetscScalar squared = 0.0;
    petsc_check(VecDot(difference.get(), weighted.get(), &squared), "compute field difference norm");
    return std::sqrt(std::max<Real>(0.0, PetscRealPart(squared)));
}

Real l2_error(const KARMA::KARMASystem& system, Vec values, const FieldFunction& exact)
{
    petsc_check(VecGhostUpdateBegin(values, INSERT_VALUES, SCATTER_FORWARD), "begin L2 ghost update");
    petsc_check(VecGhostUpdateEnd(values, INSERT_VALUES, SCATTER_FORWARD), "end L2 ghost update");
    Vec local_form = nullptr;
    petsc_check(VecGhostGetLocalForm(values, &local_form), "get L2 local form");
    const PetscScalar* local_values = nullptr;
    petsc_check(VecGetArrayRead(local_form, &local_values), "read L2 local form");
    Real local_squared_error = 0.0;
    for (const auto element : system.dof_map().assembly_elements())
    {
        const KARMA::KARMAGeometryMap geometry(system.mesh(), element);
        const auto& specification = system.variable(0);
        auto basis = system.basis_registry().create(
            {KARMA::reference_cell(geometry.variant()), specification.family, specification.order,
             specification.continuity});
        KARMA::KARMAFEBase fe(system.mesh(), geometry.variant(), std::move(basis), 5);
        fe.reinit(element);
        std::vector<PetscInt> local_dofs;
        system.dof_map().local_dof_indices(element, local_dofs);
        for (std::size_t point = 0; point < fe.n_quadrature_points(); ++point)
        {
            Real numerical = 0.0;
            for (std::size_t basis_index = 0; basis_index < local_dofs.size(); ++basis_index)
                numerical += PetscRealPart(local_values[local_dofs[basis_index]]) * fe.get_phi()[basis_index][point];
            const Real difference = numerical - exact(fe.get_xyz()[point]);
            local_squared_error += fe.get_JxW()[point] * difference * difference;
        }
    }
    petsc_check(VecRestoreArrayRead(local_form, &local_values), "restore L2 local form array");
    petsc_check(VecGhostRestoreLocalForm(values, &local_form), "restore L2 local form");
    Real global_squared_error = 0.0;
    petsc_check(MPI_Allreduce(&local_squared_error, &global_squared_error, 1, MPI_DOUBLE, MPI_SUM,
                              system.communicator()),
                "reduce L2 error");
    return std::sqrt(global_squared_error);
}

std::unique_ptr<KARMA::KARMASystem> make_system(KARMA::KARMAMesh& mesh, const std::string& name)
{
    auto system = std::make_unique<KARMA::KARMASystem>(mesh, name);
    system->add_variable(scalar_cg_spec());
    system->init();
    return system;
}

void load_or_generate_meshes(KARMA::KARMAMesh& mesh_a, KARMA::KARMAMesh& mesh_b,
                             const char* source_file, const char* target_file)
{
    if (source_file == nullptr)
    {
        mesh_a.generate_box(1, 2, 0);
        mesh_b.generate_box(2, 1, 0);
        return;
    }
    constexpr const char* options =
        "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
    KARMA::KARMAMoabIO(mesh_a).read_mesh(source_file, options);
    KARMA::KARMAMoabIO(mesh_b).read_mesh(target_file, options);
    mesh_a.update();
    mesh_b.update();
}

void verify_exact_reproduction(MPI_Comm communicator, const char* source_file = nullptr,
                               const char* target_file = nullptr)
{
    KARMA::KARMAMesh mesh_a(2, communicator);
    KARMA::KARMAMesh mesh_b(2, communicator);
    load_or_generate_meshes(mesh_a, mesh_b, source_file, target_file);
    const auto specification = scalar_cg_spec();
    KARMA::KARMASystem system_a(mesh_a, "cg-accuracy-a");
    KARMA::KARMASystem system_b(mesh_b, "cg-accuracy-b");
    system_a.add_variable(specification);
    system_b.add_variable(specification);
    system_a.init();
    system_b.init();

    const FieldFunction constant = [](const KARMA::Point&) { return 2.75; };
    const FieldFunction affine = [](const KARMA::Point& point) {
        return -0.625 + 1.75 * point(0) - 0.875 * point(1);
    };
    constexpr Real tolerance = 2.0e-11;
    require(project_and_measure(system_a, system_b, constant) < tolerance,
            "constant reproduction failed for A-to-B nonmatching cG projection");
    require(project_and_measure(system_b, system_a, constant) < tolerance,
            "constant reproduction failed for B-to-A nonmatching cG projection");
    require(project_and_measure(system_a, system_b, affine) < tolerance,
            "affine reproduction failed for A-to-B nonmatching cG projection");
    require(project_and_measure(system_b, system_a, affine) < tolerance,
            "affine reproduction failed for B-to-A nonmatching cG projection");
}

void verify_global_integral_conservation(MPI_Comm communicator)
{
    KARMA::KARMAMesh source_mesh(2, communicator);
    KARMA::KARMAMesh target_mesh(2, communicator);
    source_mesh.generate_box(2, 3, 0);
    target_mesh.generate_box(3, 2, 0);
    auto source = make_system(source_mesh, "cg-integral-source");
    auto target = make_system(target_mesh, "cg-integral-target");
    const FieldFunction affine = [](const KARMA::Point& point) {
        return 1.25 - 0.5 * point(0) + 2.0 * point(1);
    };
    const auto projected = project(*source, *target, affine);
    const auto report = KARMA::make_transfer_integral_report(
        {global_integral(*source, source->solution())}, {global_integral(*target, projected.get())},
        {global_integral(*target, projected.get())}, {0.0});
    require(report.relative_full_error.front() < 2.0e-11,
            "global consistent-mass integral was not conserved by cG projection");
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    if (rank == 0)
        std::cout << "global integral relative error (no local-conservation claim): "
                  << report.relative_full_error.front() << '\n';
}

struct QuadraticErrors
{
    Real source_interpolation = 0.0;
    Real transfer = 0.0;
    Real target_representation = 0.0;
    Real transfer_plus_target_representation = 0.0;
};

QuadraticErrors quadratic_errors(unsigned source_cells, unsigned target_cells, MPI_Comm communicator)
{
    KARMA::KARMAMesh source_mesh(2, communicator);
    KARMA::KARMAMesh target_mesh(2, communicator);
    source_mesh.generate_box(source_cells, source_cells + 1, 0);
    target_mesh.generate_box(target_cells + 1, target_cells, 0);
    auto source = make_system(source_mesh, "cg-quadratic-source");
    auto target = make_system(target_mesh, "cg-quadratic-target");
    const FieldFunction quadratic = [](const KARMA::Point& point) {
        return point(0) * point(0) + 0.5 * point(0) * point(1) + 2.0 * point(1) * point(1);
    };
    interpolate(*source, quadratic);
    const auto transferred = project(*source, *target, quadratic);
    interpolate(*target, quadratic);
    QuadraticErrors errors;
    errors.source_interpolation = l2_error(*source, source->solution(), quadratic);
    errors.target_representation = l2_error(*target, target->solution(), quadratic);
    errors.transfer = l2_difference(*target, transferred.get(), target->solution());
    errors.transfer_plus_target_representation = l2_error(*target, transferred.get(), quadratic);
    return errors;
}

bool residual_is_accepted(Mat mass, Vec solution, Vec rhs, Vec residual)
{
    return KARMA::transfer_residual_within_tolerance(mass, solution, rhs, residual, 1.0e-11);
}

void verify_residual_mutations(MPI_Comm communicator)
{
    KARMA::KARMAMesh mesh(2, communicator);
    mesh.generate_box(2, 1, 0);
    auto system = make_system(mesh, "cg-residual-mutations");
    KARMA::PetscOwner<Mat, MatDestroy> mass;
    *mass.put() = system->create_matrix();
    KARMA::assemble_mass_matrix(*system, mass.get(), nullptr, 0, 5);
    KARMA::PetscOwner<Vec, VecDestroy> exact;
    KARMA::PetscOwner<Vec, VecDestroy> rhs;
    KARMA::PetscOwner<Vec, VecDestroy> residual;
    *exact.put() = system->create_vector();
    *rhs.put() = system->create_vector();
    *residual.put() = system->create_vector();
    const FieldFunction field = [](const KARMA::Point& point) {
        return 0.75 + point(0) - 0.5 * point(1);
    };
    interpolate(*system, field);
    petsc_check(VecCopy(system->solution(), exact.get()), "copy exact mutation solution");
    petsc_check(MatMult(mass.get(), exact.get(), rhs.get()), "form exact mutation rhs");
    require(residual_is_accepted(mass.get(), exact.get(), rhs.get(), residual.get()),
            "exact mass solve residual was rejected");

    KARMA::PetscOwner<Mat, MatDestroy> wrong_weights;
    petsc_check(MatDuplicate(mass.get(), MAT_COPY_VALUES, wrong_weights.put()), "copy wrong-weight mass");
    petsc_check(MatScale(wrong_weights.get(), 0.5), "apply wrong quadrature weight scale");
    require(!residual_is_accepted(wrong_weights.get(), exact.get(), rhs.get(), residual.get()),
            "wrong mass weights escaped the scale-aware residual contract");

    KARMA::PetscOwner<Vec, VecDestroy> omitted_rhs;
    *omitted_rhs.put() = system->create_vector();
    petsc_check(VecCopy(rhs.get(), omitted_rhs.get()), "copy omitted-term rhs");
    PetscInt first = 0;
    PetscInt last = 0;
    petsc_check(VecGetOwnershipRange(omitted_rhs.get(), &first, &last), "read omitted-term ownership");
    if (first < last)
    {
        PetscScalar value = 0.0;
        petsc_check(VecGetValues(omitted_rhs.get(), 1, &first, &value), "read omitted rhs entry");
        value *= 0.5;
        petsc_check(VecSetValue(omitted_rhs.get(), first, value, INSERT_VALUES), "omit rhs contribution");
    }
    petsc_check(VecAssemblyBegin(omitted_rhs.get()), "begin omitted rhs assembly");
    petsc_check(VecAssemblyEnd(omitted_rhs.get()), "end omitted rhs assembly");
    require(!residual_is_accepted(mass.get(), exact.get(), omitted_rhs.get(), residual.get()),
            "omitted rhs term escaped the scale-aware residual contract");

    KARMA::PetscOwner<Vec, VecDestroy> loose_solution;
    *loose_solution.put() = system->create_vector();
    petsc_check(VecCopy(exact.get(), loose_solution.get()), "seed loose KSP solution");
    petsc_check(VecScale(loose_solution.get(), 0.75), "scale loose KSP initial guess");
    KARMA::PetscOwner<KSP, KSPDestroy> loose_ksp;
    petsc_check(KSPCreate(communicator, loose_ksp.put()), "create loose KSP");
    petsc_check(KSPSetOperators(loose_ksp.get(), mass.get(), mass.get()), "set loose KSP operators");
    petsc_check(KSPSetType(loose_ksp.get(), KSPCG), "set loose KSP type");
    petsc_check(KSPSetTolerances(loose_ksp.get(), 0.5, PETSC_DEFAULT, PETSC_DEFAULT, 1000),
                "set loose KSP tolerance");
    petsc_check(KSPSetInitialGuessNonzero(loose_ksp.get(), PETSC_TRUE), "set loose KSP initial guess policy");
    petsc_check(KSPSolve(loose_ksp.get(), rhs.get(), loose_solution.get()), "run loose KSP solve");
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    petsc_check(KSPGetConvergedReason(loose_ksp.get(), &reason), "read loose KSP reason");
    require(reason > 0, "loose KSP did not report convergence for mutation test");
    require(!residual_is_accepted(mass.get(), loose_solution.get(), rhs.get(), residual.get()),
            "loose KSP convergence escaped the independent residual contract");
}

void verify_independent_quadratic_refinement(MPI_Comm communicator)
{
    const auto source_coarse = quadratic_errors(1, 3, communicator);
    const auto source_fine = quadratic_errors(2, 3, communicator);
    const auto target_coarse = quadratic_errors(3, 1, communicator);
    const auto target_fine = quadratic_errors(3, 2, communicator);
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    if (rank == 0)
    {
        const auto print = [](const char* label, const QuadraticErrors& errors) {
            std::cout << label << ": source interpolation=" << errors.source_interpolation
                      << ", transfer=" << errors.transfer
                      << ", target representation=" << errors.target_representation
                      << ", transfer+target representation=" << errors.transfer_plus_target_representation << '\n';
        };
        print("quadratic source-coarse", source_coarse);
        print("quadratic source-fine", source_fine);
        print("quadratic target-coarse", target_coarse);
        print("quadratic target-fine", target_fine);
    }
    require(source_fine.source_interpolation < 0.35 * source_coarse.source_interpolation,
            "source interpolation error did not decrease independently under source refinement");
    require(source_fine.transfer < 0.45 * source_coarse.transfer,
            "transfer error did not decrease independently under source refinement");
    require(target_fine.target_representation < 0.35 * target_coarse.target_representation,
            "target representation error did not decrease independently under target refinement");
    require(target_fine.transfer_plus_target_representation < target_coarse.transfer_plus_target_representation,
            "combined transfer and target representation error did not decrease under target refinement");
}
}  // namespace

int main(int argc, char** argv)
{
    if (PetscInitialize(&argc, &argv, nullptr, nullptr) != PETSC_SUCCESS) return 1;
    int status = 0;
    try
    {
        if (argc == 4 && std::string(argv[1]) == "--partitioned")
            verify_exact_reproduction(PETSC_COMM_WORLD, argv[2], argv[3]);
        else
        {
            require(argc == 1, "invalid cG transfer accuracy arguments");
            verify_exact_reproduction(PETSC_COMM_WORLD);
            verify_global_integral_conservation(PETSC_COMM_WORLD);
            verify_independent_quadratic_refinement(PETSC_COMM_WORLD);
            verify_residual_mutations(PETSC_COMM_WORLD);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "cG transfer accuracy failure: " << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
