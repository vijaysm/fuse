#include "karma_collective_outcome.h"
#include "karma_compiled_fem_connection.h"

#include "generic_mass_operator.h"
#include "karma_distributed_point_locator.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_fem_field_evaluator.h"
#include "karma_reference_topology.h"
#include "karma_system.h"
#include "karma_transfer_diagnostics.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"

#include <cmath>
#include <chrono>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace KARMA
{
namespace
{
/** Reconcile a rank-local failure across `communicator` and throw the agreed
 * message, or return when every rank succeeded.
 *
 * The reduction rule lives in `reconcile_failure_message()`, which this file
 * used to carry its own copy of.
 * This file's previous copy reported an MPI failure as "MPI error while reconciling compiled FEM connection failure";
 * the shared primitive reports it uniformly instead.
 */
void reconcile_failure(MPI_Comm communicator, bool failed, const std::string& local_message)
{
    const auto message = reconcile_failure_message(communicator, failed, local_message);
    if (!message.empty()) KARMA_THROW(InvalidArgument(message));
}

void petsc_check(PetscErrorCode error, const char* message)
{
    if (error != PETSC_SUCCESS) KARMA_THROW(InvalidArgument(message));
}

const PetscFieldBinding& petsc_binding(const KARMAField& field)
{
    if (!std::holds_alternative<PetscFieldBinding>(field.binding()))
        KARMA_THROW(InvalidArgument("compiled FEM connection requires a PETSc source field"));
    return std::get<PetscFieldBinding>(field.binding());
}
}  // namespace

class CompiledFEMConnection::Impl
{
  public:
    struct ElementData
    {
        moab::EntityHandle handle = 0;
        std::unique_ptr<KARMAFEBase> fe;
        std::size_t point_offset = 0;
    };

    explicit Impl(const KARMAMesh& source_mesh) : locator(source_mesh) {}

    KARMADistributedPointLocator locator;
    std::vector<ElementData> elements;
    std::vector<Point> points;
    std::vector<LocatedPoint> located;
    PetscOwner<Mat, MatDestroy> mass;
    PetscOwner<Vec, VecDestroy> rhs;
    PetscOwner<Vec, VecDestroy> solution;
    PetscOwner<Vec, VecDestroy> residual;
    PetscOwner<Vec, VecDestroy> staging;
    PetscOwner<KSP, KSPDestroy> ksp;
};

CompiledFEMConnection::CompiledFEMConnection(KARMAField source, const KARMASystem& target_system,
                                             unsigned target_variable, int quadrature_degree)
    : source_(std::move(source)), target_system_(&target_system), target_variable_(target_variable),
      quadrature_degree_(quadrature_degree)
{
}

CompiledFEMConnection::~CompiledFEMConnection() = default;
CompiledFEMConnection::CompiledFEMConnection(CompiledFEMConnection&&) noexcept = default;
CompiledFEMConnection& CompiledFEMConnection::operator=(CompiledFEMConnection&&) noexcept = default;

void CompiledFEMConnection::setup()
{
    const auto started = std::chrono::steady_clock::now();
    const auto& source_binding = petsc_binding(source_);
    const MPI_Comm communicator = target_system_->communicator();
    bool validation_failed = false;
    std::string validation_error;
    const VariableSpec* specification_pointer = nullptr;
    try
    {
        if (source_binding.system == nullptr || source_binding.vector == nullptr)
            KARMA_THROW(InvalidArgument("compiled FEM connection source binding is incomplete"));
        source_binding.system->validate_vector_layout(source_binding.vector, "compiled FEM source");
        if (target_variable_ >= target_system_->n_variables())
            KARMA_THROW(InvalidArgument("compiled FEM target variable is out of range"));
        specification_pointer = &target_system_->variable(target_variable_);
        if (specification_pointer->continuity != BasisContinuity::cG)
            KARMA_THROW(InvalidArgument("compiled FEM connection requires a cG target variable"));
        if (specification_pointer->components != 1 || source_.components() != 1)
            KARMA_THROW(InvalidArgument("compiled FEM connection supports scalar fields only"));
        if (quadrature_degree_ <= 0)
            KARMA_THROW(InvalidArgument("compiled FEM connection quadrature degree must be positive"));
        int relation = MPI_UNEQUAL;
        if (MPI_Comm_compare(communicator, source_.communicator(), &relation) != MPI_SUCCESS ||
            (relation != MPI_IDENT && relation != MPI_CONGRUENT))
            KARMA_THROW(InvalidArgument("compiled FEM source and target communicators are not congruent"));
    }
    catch (const std::exception& error) { validation_failed = true; validation_error = error.what(); }
    catch (...) { validation_failed = true; validation_error = "unknown compiled FEM validation failure"; }
    reconcile_failure(communicator, validation_failed, validation_error);

    auto compiled = std::make_unique<Impl>(source_binding.system->mesh());
    bool preparation_failed = false;
    std::string preparation_error;
    try
    {
        compiled->locator.setup();
        const auto& mesh = target_system_->mesh();
        compiled->elements.reserve(target_system_->dof_map().assembly_elements().size());
        for (const auto element : target_system_->dof_map().assembly_elements())
        {
            const moab::EntityHandle* connectivity = nullptr;
            int count = 0;
            if (mesh.moab()->get_connectivity(element, connectivity, count) != moab::MB_SUCCESS)
                KARMA_THROW(InvalidArgument("failed to read compiled cG target connectivity"));
            const auto geometry = KARMAReferenceTopology::from_moab(
                mesh.moab()->type_from_handle(element), static_cast<std::size_t>(count));
            const auto basis = target_system_->basis_registry().create(
                {reference_cell(geometry), specification_pointer->family, specification_pointer->order,
                 specification_pointer->continuity});
            auto fe = std::make_unique<KARMAFEBase>(mesh, geometry, basis, quadrature_degree_);
            fe->reinit(element);
            Impl::ElementData data{element, std::move(fe), compiled->points.size()};
            compiled->points.insert(
                compiled->points.end(), data.fe->get_xyz().begin(),
                data.fe->get_xyz().begin() + static_cast<std::ptrdiff_t>(data.fe->n_quadrature_points()));
            compiled->elements.push_back(std::move(data));
        }
    }
    catch (const std::exception& error) { preparation_failed = true; preparation_error = error.what(); }
    catch (...) { preparation_failed = true; preparation_error = "unknown compiled cG FE preparation failure"; }
    reconcile_failure(communicator, preparation_failed, preparation_error);

    bool location_failed = false;
    std::string location_error;
    try
    {
        compiled->located = compiled->locator.locate(compiled->points);
        for (const auto& point : compiled->located)
            if (point.status != PointLocationStatus::Located)
                KARMA_THROW(InvalidArgument("compiled cG target quadrature point is uncovered"));
    }
    catch (const std::exception& error) { location_failed = true; location_error = error.what(); }
    catch (...) { location_failed = true; location_error = "unknown compiled cG point location failure"; }
    reconcile_failure(communicator, location_failed, location_error);

    *compiled->rhs.put() = target_system_->create_vector();
    *compiled->solution.put() = target_system_->create_vector();
    *compiled->residual.put() = target_system_->create_vector();
    *compiled->staging.put() = target_system_->create_vector();
    *compiled->mass.put() = target_system_->create_matrix();
    assemble_mass_matrix(*target_system_, compiled->mass.get(), nullptr, 0, quadrature_degree_);
    petsc_check(KSPCreate(communicator, compiled->ksp.put()), "failed to create compiled cG transfer KSP");
    petsc_check(KSPSetOperators(compiled->ksp.get(), compiled->mass.get(), compiled->mass.get()),
                "failed to set compiled cG transfer operators");
    petsc_check(KSPSetType(compiled->ksp.get(), KSPCG), "failed to set compiled cG transfer KSP type");
    petsc_check(KSPSetTolerances(compiled->ksp.get(), 1.e-13, PETSC_DEFAULT, PETSC_DEFAULT, 1000),
                 "failed to set compiled cG transfer tolerances");
    petsc_check(KSPSetOptionsPrefix(compiled->ksp.get(), "karma_cg_transfer_"),
                 "failed to prefix compiled cG transfer KSP options");
    petsc_check(KSPSetFromOptions(compiled->ksp.get()), "failed to apply compiled cG transfer KSP options");

    petsc_check(KSPSetUp(compiled->ksp.get()), "failed to set up compiled cG transfer KSP");

    impl_ = std::move(compiled);
    ++setup_count_;
    diagnostics_.covered_points = impl_->located.size();
    diagnostics_.total_points = impl_->located.size();
    diagnostics_.setup_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void CompiledFEMConnection::apply(const KARMAField& source)
{
    const auto started = std::chrono::steady_clock::now();
    if (!impl_) KARMA_THROW(InvalidArgument("compiled FEM connection must be set up before apply"));
    const MPI_Comm communicator = target_system_->communicator();
    bool validation_failed = false;
    std::string validation_error;
    try
    {
        const auto& expected = petsc_binding(source_);
        const auto& actual = petsc_binding(source);
        const bool specification_changed =
            actual.specification.name != expected.specification.name ||
            actual.specification.family != expected.specification.family ||
            actual.specification.order != expected.specification.order ||
            actual.specification.continuity != expected.specification.continuity ||
            actual.specification.components != expected.specification.components;
        if (actual.system != expected.system || actual.variable != expected.variable || specification_changed)
            KARMA_THROW(InvalidArgument("compiled FEM source system or layout changed; call setup again"));
        actual.system->validate_vector_layout(actual.vector, "compiled FEM source apply");
    }
    catch (const std::exception& error) { validation_failed = true; validation_error = error.what(); }
    catch (...) { validation_failed = true; validation_error = "unknown compiled FEM apply validation failure"; }
    reconcile_failure(communicator, validation_failed, validation_error);

    KARMAFEMFieldEvaluator evaluator(source);
    const auto evaluated = evaluator.evaluate_batch(impl_->located);
    petsc_check(VecSet(impl_->rhs.get(), 0.0), "failed to zero compiled cG transfer rhs");
    bool assembly_failed = false;
    std::string assembly_error;
    try
    {
        for (const auto& element : impl_->elements)
        {
            const auto& fe = *element.fe;
            std::vector<PetscInt> indices;
            target_system_->dof_map().dof_indices(element.handle, indices, target_variable_);
            RealDenseVector local_rhs(indices.size());
            for (std::size_t point = 0; point < fe.n_quadrature_points(); ++point)
            {
                const auto index = element.point_offset + point;
                if (impl_->located[index].status != PointLocationStatus::Located)
                    KARMA_THROW(InvalidArgument("compiled cG target quadrature point is uncovered"));
                for (std::size_t basis = 0; basis < indices.size(); ++basis)
                    local_rhs(basis) += fe.get_JxW()[point] * fe.get_phi()[basis][point] * evaluated[index][0];
            }
            scatter(impl_->rhs.get(), local_rhs, indices, ADD_VALUES);
        }
    }
    catch (const std::exception& error) { assembly_failed = true; assembly_error = error.what(); }
    catch (...) { assembly_failed = true; assembly_error = "unknown compiled cG local assembly failure"; }
    reconcile_failure(communicator, assembly_failed, assembly_error);

    petsc_check(VecAssemblyBegin(impl_->rhs.get()), "failed to begin compiled cG rhs assembly");
    petsc_check(VecAssemblyEnd(impl_->rhs.get()), "failed to end compiled cG rhs assembly");
    petsc_check(KSPSolve(impl_->ksp.get(), impl_->rhs.get(), impl_->solution.get()),
                "failed to solve compiled cG transfer");
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    petsc_check(KSPGetConvergedReason(impl_->ksp.get(), &reason),
                "failed to query compiled cG transfer convergence");
    if (reason <= 0) KARMA_THROW(InvalidArgument("compiled cG transfer solve did not converge"));
    const auto residual = transfer_residual_report(impl_->mass.get(), impl_->solution.get(), impl_->rhs.get(),
                                                   impl_->residual.get());
    diagnostics_.solver_reason = reason;
    diagnostics_.residual_norm = residual.residual_norm;
    diagnostics_.residual_scale = residual.scale;
    diagnostics_.residual_tolerance = residual.tolerance;
    if (!residual.within_tolerance)
        KARMA_THROW(InvalidArgument("compiled cG transfer residual exceeds the configured scale-aware tolerance"));

    petsc_check(VecCopy(impl_->solution.get(), impl_->staging.get()),
                "failed to publish compiled cG transfer solution");
    diagnostics_.apply_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

Vec CompiledFEMConnection::staging() const noexcept { return impl_ ? impl_->staging.get() : nullptr; }
const void* CompiledFEMConnection::mass_handle() const noexcept { return impl_ ? impl_->mass.get() : nullptr; }
const void* CompiledFEMConnection::rhs_handle() const noexcept { return impl_ ? impl_->rhs.get() : nullptr; }
const void* CompiledFEMConnection::solution_handle() const noexcept { return impl_ ? impl_->solution.get() : nullptr; }
const void* CompiledFEMConnection::residual_handle() const noexcept { return impl_ ? impl_->residual.get() : nullptr; }
const void* CompiledFEMConnection::ksp_handle() const noexcept { return impl_ ? impl_->ksp.get() : nullptr; }
std::size_t CompiledFEMConnection::located_point_count() const noexcept
{
    return impl_ ? impl_->located.size() : 0;
}
const void* CompiledFEMConnection::located_points_handle() const noexcept
{
    return impl_ && !impl_->located.empty() ? impl_->located.data() : nullptr;
}
}  // namespace KARMA
