#include "karma_collective_outcome.h"
#include "karma_fem_transfer.h"

#include "generic_mass_operator.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_quadrature.h"
#include "karma_transfer_diagnostics.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"

#include <Eigen/LU>
#include <petscksp.h>

#include <cmath>
#include <limits>
#include <string>

namespace KARMA
{
namespace
{
/** Reconcile a rank-local failure across `communicator` and throw the agreed
 * message, or return when every rank succeeded.
 *
 * The reduction rule lives in `reconcile_failure_message()`, which this file
 * used to carry its own copy of.
 * This file's previous copy reported an MPI failure as "MPI error while reconciling FEM transfer failure";
 * the shared primitive reports it uniformly instead.
 */
void reconcile_failure(MPI_Comm communicator, bool failed, const std::string& local_message)
{
    const auto message = reconcile_failure_message(communicator, failed, local_message);
    if (!message.empty()) KARMA_THROW(InvalidArgument(message));
}
}

FEMTransferResult KARMAFEMTransfer::sample(const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source,
                                            std::span<const Point> target_points, unsigned components,
                                            const FillPolicy* fill)
{
    if (components == 0 || components != source.components())
        KARMA_THROW(InvalidArgument("FEM sampling component count differs from source"));
    if (fill != nullptr) fill->validate(components);
    const auto located = locator.locate(target_points);
    FEMTransferResult result;
    result.values.resize(target_points.size() * components, 0.0);
    result.coverage.resize(target_points.size());
    const auto evaluated = source.evaluate_batch(located);
    for (std::size_t point = 0; point < target_points.size(); ++point)
    {
        result.coverage[point].target = {FieldEntryKey::Kind::MoabGlobalId, point + 1, 0, {}};
        if (located[point].status == PointLocationStatus::Located)
        {
            const auto& values = evaluated[point];
            for (unsigned component = 0; component < components; ++component)
                result.values[point * components + component] = values[component];
            result.coverage[point].classification = CoverageClass::Full;
        }
        else if (located[point].status == PointLocationStatus::Uncovered)
        {
            if (fill == nullptr) KARMA_THROW(InvalidArgument("uncovered target point requires an explicit fill policy"));
            for (unsigned component = 0; component < components; ++component)
                result.values[point * components + component] = fill->values[component];
            result.coverage[point].classification = CoverageClass::Uncovered;
        }
        else
            KARMA_THROW(InvalidArgument(located[point].diagnostic.empty() ? "point location failed"
                                                                          : located[point].diagnostic));
    }
    return result;
}

FEMElementProjectionResult KARMAFEMTransfer::project_element(const KARMAMesh& target_mesh,
                                                              moab::EntityHandle target_element,
                                                              const KARMABasis& target_basis, int quadrature_degree,
                                                              const KARMAPointLocator& locator,
                                                              const KARMAFEMFieldEvaluator& source,
                                                              unsigned components, bool reject_partial)
{
    if (components == 0 || components != source.components())
        KARMA_THROW(InvalidArgument("FEM projection component count differs from source"));
    const GeometryVariant geometry_variant = [&] {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        if (target_mesh.moab()->get_connectivity(target_element, connectivity, count) != moab::MB_SUCCESS)
            KARMA_THROW(InvalidArgument("failed to read target element connectivity"));
        return KARMAReferenceTopology::from_moab(target_mesh.moab()->type_from_handle(target_element),
                                                  static_cast<std::size_t>(count));
    }();
    KARMAFEBase target_fe(target_mesh, geometry_variant, target_basis, quadrature_degree);
    target_fe.reinit(target_element);
    const std::size_t n_basis = target_basis.size();
    const std::size_t n_points = target_fe.n_quadrature_points();
    const auto located = locator.locate(std::span<const Point>(target_fe.get_xyz().data(), n_points));
    // evaluate_batch() is collective for remote source packets. Element-local
    // projection is currently safe only when all ranks enter the same element
    // sequence; partitioned projection orchestration remains a higher-level TODO.
    const auto& source_binding = std::get<PetscFieldBinding>(source.field().binding());
    if (source_binding.system->mesh().n_ranks() > 1)
        for (const auto& point : located)
            if (point.status == PointLocationStatus::Located && point.source_handle == 0)
                KARMA_THROW(InvalidArgument("partitioned element projection requires batched transfer orchestration"));
    RealDenseMatrix mass(n_basis, n_basis);
    std::vector<RealDenseVector> rhs;
    rhs.reserve(components);
    for (unsigned component = 0; component < components; ++component) rhs.emplace_back(n_basis);
    std::size_t covered = 0;
    const auto evaluated = source.evaluate_batch(located);
    for (std::size_t point = 0; point < n_points; ++point)
    {
        if (located[point].status != PointLocationStatus::Located)
        {
            if (located[point].status != PointLocationStatus::Uncovered)
                KARMA_THROW(InvalidArgument("target quadrature point location failed"));
            continue;
        }
        ++covered;
        const auto& values = evaluated[point];
        for (std::size_t i = 0; i < n_basis; ++i)
        {
            for (std::size_t j = 0; j < n_basis; ++j)
                mass(i, j) += target_fe.get_JxW()[point] * target_fe.get_phi()[i][point] * target_fe.get_phi()[j][point];
            for (unsigned component = 0; component < components; ++component)
                rhs[component](i) += target_fe.get_JxW()[point] * target_fe.get_phi()[i][point] * values[component];
        }
    }
    const Real fraction = n_points == 0 ? 0.0 : static_cast<Real>(covered) / static_cast<Real>(n_points);
    if (reject_partial && covered != n_points)
        KARMA_THROW(InvalidArgument("target element has partial or uncovered source coverage"));
    if (covered == 0) return {std::vector<Real>(components * n_basis, 0.0), CoverageClass::Uncovered, 0.0};
    if (covered != n_points) return {std::vector<Real>(components * n_basis, 0.0), CoverageClass::Partial, fraction};
    if (!mass.native().allFinite()) KARMA_THROW(InvalidArgument("target mass matrix is nonfinite"));
    Eigen::FullPivLU<RealDenseMatrix::Storage> factorization(mass.native());
    if (!factorization.isInvertible()) KARMA_THROW(InvalidArgument("target local mass matrix is singular"));
    FEMElementProjectionResult result;
    result.coefficients.resize(components * n_basis);
    for (unsigned component = 0; component < components; ++component)
    {
        const auto solution = factorization.solve(rhs[component].native());
        if (!solution.allFinite()) KARMA_THROW(InvalidArgument("target local projection is nonfinite"));
        for (std::size_t basis = 0; basis < n_basis; ++basis)
            result.coefficients[component * n_basis + basis] = solution(static_cast<Eigen::Index>(basis));
    }
    result.coverage = CoverageClass::Full;
    result.covered_measure_fraction = 1.0;
    return result;
}

std::vector<FEMElementProjectionResult> KARMAFEMTransfer::project_elements(
    const KARMAMesh& target_mesh, std::span<const moab::EntityHandle> target_elements,
    const KARMABasis& target_basis, int quadrature_degree, const KARMAPointLocator& locator,
    const KARMAFEMFieldEvaluator& source, unsigned components, bool reject_partial)
{
    if (components == 0 || components != source.components())
        KARMA_THROW(InvalidArgument("FEM projection component count differs from source"));
    struct ElementData
    {
        std::unique_ptr<KARMAFEBase> fe;
        std::size_t point_offset = 0;
    };
    std::vector<ElementData> elements;
    std::vector<Point> points;
    elements.reserve(target_elements.size());
    for (const auto element : target_elements)
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        if (target_mesh.moab()->get_connectivity(element, connectivity, count) != moab::MB_SUCCESS)
            KARMA_THROW(InvalidArgument("failed to read target element connectivity"));
        const auto geometry = KARMAReferenceTopology::from_moab(
            target_mesh.moab()->type_from_handle(element), static_cast<std::size_t>(count));
        auto fe = std::make_unique<KARMAFEBase>(target_mesh, geometry, target_basis, quadrature_degree);
        fe->reinit(element);
        ElementData data{std::move(fe), points.size()};
        points.insert(points.end(), data.fe->get_xyz().begin(),
                      data.fe->get_xyz().begin() + static_cast<std::ptrdiff_t>(data.fe->n_quadrature_points()));
        elements.push_back(std::move(data));
    }
    const auto located = locator.locate(points);
    const auto evaluated = source.evaluate_batch(located);
    std::vector<FEMElementProjectionResult> results;
    results.reserve(elements.size());
    for (const auto& element : elements)
    {
        const auto& fe = *element.fe;
        const std::size_t n_basis = target_basis.size();
        const std::size_t n_points = fe.n_quadrature_points();
        RealDenseMatrix mass(n_basis, n_basis);
        std::vector<RealDenseVector> rhs;
        for (unsigned component = 0; component < components; ++component) rhs.emplace_back(n_basis);
        std::size_t covered = 0;
        for (std::size_t point = 0; point < n_points; ++point)
        {
            const auto& location = located[element.point_offset + point];
            if (location.status != PointLocationStatus::Located)
            {
                if (location.status != PointLocationStatus::Uncovered)
                    KARMA_THROW(InvalidArgument("target quadrature point location failed"));
                continue;
            }
            ++covered;
            const auto& values = evaluated[element.point_offset + point];
            for (std::size_t i = 0; i < n_basis; ++i)
            {
                for (std::size_t j = 0; j < n_basis; ++j)
                    mass(i, j) += fe.get_JxW()[point] * fe.get_phi()[i][point] * fe.get_phi()[j][point];
                for (unsigned component = 0; component < components; ++component)
                    rhs[component](i) += fe.get_JxW()[point] * fe.get_phi()[i][point] * values[component];
            }
        }
        const Real fraction = n_points == 0 ? 0.0 : static_cast<Real>(covered) / static_cast<Real>(n_points);
        if (reject_partial && covered != n_points)
            KARMA_THROW(InvalidArgument("target element has partial or uncovered source coverage"));
        if (covered != n_points)
        {
            results.push_back({std::vector<Real>(components * n_basis, 0.0),
                               covered == 0 ? CoverageClass::Uncovered : CoverageClass::Partial, fraction});
            continue;
        }
        Eigen::FullPivLU<RealDenseMatrix::Storage> factorization(mass.native());
        if (!factorization.isInvertible()) KARMA_THROW(InvalidArgument("target local mass matrix is singular"));
        FEMElementProjectionResult result;
        result.coefficients.resize(components * n_basis);
        for (unsigned component = 0; component < components; ++component)
        {
            const auto solution = factorization.solve(rhs[component].native());
            if (!solution.allFinite()) KARMA_THROW(InvalidArgument("target local projection is nonfinite"));
            for (std::size_t basis = 0; basis < n_basis; ++basis)
                result.coefficients[component * n_basis + basis] = solution(static_cast<Eigen::Index>(basis));
        }
        results.push_back(std::move(result));
    }
    return results;
}

void KARMAFEMTransfer::project_cg(const KARMASystem& target_system, unsigned target_variable, int quadrature_degree,
                                   const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source, Vec target,
                                   Real residual_tolerance)
{
    target_system.validate_vector_layout(target, "cG transfer target");
    const VariableSpec* specification_pointer = nullptr;
    bool validation_failed = false;
    std::string validation_error;
    try
    {
        if (target_variable >= target_system.n_variables())
            KARMA_THROW(InvalidArgument("cG transfer target variable is out of range"));
        specification_pointer = &target_system.variable(target_variable);
        if (specification_pointer->continuity != BasisContinuity::cG)
            KARMA_THROW(InvalidArgument("global projection requires a cG target variable"));
        if (specification_pointer->components != 1)
            KARMA_THROW(InvalidArgument("global projection does not support multicomponent target variables"));
        if (!std::isfinite(residual_tolerance) || residual_tolerance <= 0.0)
            KARMA_THROW(InvalidArgument("cG transfer residual tolerance must be finite and positive"));
        int communicator_relation = MPI_UNEQUAL;
        if (MPI_Comm_compare(target_system.communicator(), source.field().communicator(), &communicator_relation) != MPI_SUCCESS ||
            (communicator_relation != MPI_IDENT && communicator_relation != MPI_CONGRUENT))
            KARMA_THROW(InvalidArgument("cG transfer source and target communicators are not congruent"));
    }
    catch (const std::exception& error) { validation_failed = true; validation_error = error.what(); }
    catch (...) { validation_failed = true; validation_error = "unknown cG transfer validation failure"; }
    reconcile_failure(target_system.communicator(), validation_failed, validation_error);
    const auto& specification = *specification_pointer;
    PetscOwner<Vec, VecDestroy> rhs;
    *rhs.put() = target_system.create_vector();
    if (VecSet(rhs.get(), 0.0) != PETSC_SUCCESS) KARMA_THROW(InvalidArgument("failed to initialize cG transfer rhs"));
    const auto& mesh = target_system.mesh();
    struct ElementData
    {
        moab::EntityHandle handle = 0;
        std::unique_ptr<KARMAFEBase> fe;
        std::size_t point_offset = 0;
    };
    std::vector<ElementData> elements;
    std::vector<Point> points;
    bool preparation_failed = false;
    std::string preparation_error;
    try
    {
        elements.reserve(target_system.dof_map().assembly_elements().size());
        for (const auto element : target_system.dof_map().assembly_elements())
        {
            const auto geometry = [&] {
                const moab::EntityHandle* connectivity = nullptr;
                int count = 0;
                if (mesh.moab()->get_connectivity(element, connectivity, count) != moab::MB_SUCCESS)
                    KARMA_THROW(InvalidArgument("failed to read cG target connectivity"));
                return KARMAReferenceTopology::from_moab(mesh.moab()->type_from_handle(element),
                                                          static_cast<std::size_t>(count));
            }();
            const auto basis = target_system.basis_registry().create(
                {reference_cell(geometry), specification.family, specification.order, specification.continuity});
            auto fe = std::make_unique<KARMAFEBase>(mesh, geometry, basis, quadrature_degree);
            fe->reinit(element);
            ElementData data{element, std::move(fe), points.size()};
            points.insert(points.end(), data.fe->get_xyz().begin(),
                          data.fe->get_xyz().begin() + static_cast<std::ptrdiff_t>(data.fe->n_quadrature_points()));
            elements.push_back(std::move(data));
        }
    }
    catch (const std::exception& error) { preparation_failed = true; preparation_error = error.what(); }
    catch (...) { preparation_failed = true; preparation_error = "unknown cG FE preparation failure"; }
    reconcile_failure(target_system.communicator(), preparation_failed, preparation_error);

    std::vector<LocatedPoint> located;
    bool location_failed = false;
    std::string location_error;
    try { located = locator.locate(points); }
    catch (const std::exception& error) { location_failed = true; location_error = error.what(); }
    catch (...) { location_failed = true; location_error = "unknown cG point location failure"; }
    reconcile_failure(target_system.communicator(), location_failed, location_error);
    const auto evaluated = source.evaluate_batch(located);
    bool assembly_failed = false;
    std::string assembly_error;
    try
    {
    for (const auto& element : elements)
    {
        const auto& fe = *element.fe;
        std::vector<PetscInt> indices;
        target_system.dof_map().dof_indices(element.handle, indices, target_variable);
        RealDenseVector local_rhs(indices.size());
        for (std::size_t point = 0; point < fe.n_quadrature_points(); ++point)
        {
            const auto index = element.point_offset + point;
            if (located[index].status != PointLocationStatus::Located)
                KARMA_THROW(InvalidArgument("cG target quadrature point is uncovered"));
            const auto& values = evaluated[index];
            for (std::size_t i = 0; i < indices.size(); ++i)
                local_rhs(i) += fe.get_JxW()[point] * fe.get_phi()[i][point] * values[0];
        }
        scatter(rhs.get(), local_rhs, indices, ADD_VALUES);
    }
    }
    catch (const std::exception& error) { assembly_failed = true; assembly_error = error.what(); }
    catch (...) { assembly_failed = true; assembly_error = "unknown cG local assembly failure"; }
    reconcile_failure(target_system.communicator(), assembly_failed, assembly_error);
    if (VecAssemblyBegin(rhs.get()) != PETSC_SUCCESS || VecAssemblyEnd(rhs.get()) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to assemble cG transfer rhs"));
    PetscOwner<Mat, MatDestroy> mass;
    *mass.put() = target_system.create_matrix();
    assemble_mass_matrix(target_system, mass.get(), nullptr, 0, quadrature_degree);
    PetscOwner<Vec, VecDestroy> solution;
    *solution.put() = target_system.create_vector();
    PetscOwner<KSP, KSPDestroy> solver;
    if (KSPCreate(target_system.communicator(), solver.put()) != PETSC_SUCCESS ||
        KSPSetOperators(solver.get(), mass.get(), mass.get()) != PETSC_SUCCESS ||
        KSPSetType(solver.get(), KSPCG) != PETSC_SUCCESS ||
        KSPSetTolerances(solver.get(), 1.e-13, PETSC_DEFAULT, PETSC_DEFAULT, 1000) != PETSC_SUCCESS ||
        KSPSetOptionsPrefix(solver.get(), "karma_cg_transfer_") != PETSC_SUCCESS ||
        KSPSetFromOptions(solver.get()) != PETSC_SUCCESS ||
        KSPSolve(solver.get(), rhs.get(), solution.get()) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("PETSc error in cG transfer solve"));
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    if (KSPGetConvergedReason(solver.get(), &reason) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to query cG transfer convergence"));
    if (reason <= 0) KARMA_THROW(InvalidArgument("cG transfer solve did not converge"));
    PetscOwner<Vec, VecDestroy> residual;
    *residual.put() = target_system.create_vector();
    if (!transfer_residual_within_tolerance(mass.get(), solution.get(), rhs.get(), residual.get(),
                                            residual_tolerance))
        KARMA_THROW(InvalidArgument("cG transfer residual exceeds the configured scale-aware tolerance"));
    if (VecCopy(solution.get(), target) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to publish cG transfer solution"));
}
}  // namespace KARMA
