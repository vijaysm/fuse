#include "karma_transient_reaction_diffusion_adapter.h"

#include "karma_exceptions.h"
#include "karma_mesh.h"
#include "karma_moab_io.h"
#include "petsc_raii.h"
#include "../physics/common/impls/collective_signature.h"

#include <cmath>
#include <exception>
#include <utility>

namespace KARMA
{
namespace
{
FieldSpec output_spec(const std::string& id)
{
    return {id + ".state", "1", 1, FieldRole::Prognostic};
}

FieldSpec input_spec(const std::string& id)
{
    return {id + ".coupled", "1", 1, FieldRole::Input};
}
}

TransientReactionDiffusionNativeAdapter::TransientReactionDiffusionNativeAdapter(
    std::string id, KARMAMesh& mesh, GrayScottSpecies species, GrayScottParameters parameters, Real initial_state)
    : id_(std::move(id)), initial_state_(initial_state),
      specification_{id_, {{"state", PortDirection::Output, false, output_spec(id_)},
                           {"coupled", PortDirection::Input, true, input_spec(id_)}}, true, true},
      component_(mesh, make_model(species, GrayScottModel(parameters)))
{
    if (id_.empty() || !std::isfinite(initial_state_))
        KARMA_THROW(InvalidArgument("transient native adapter requires a nonempty id and finite initial state"));
    validate_component_spec(specification_);
    communicator_ = mesh.communicator();
}

TransientReactionDiffusionModel TransientReactionDiffusionNativeAdapter::make_model(
    GrayScottSpecies species, GrayScottModel model)
{
    TransientReactionDiffusionModel result;
    result.diffusion = model.diffusion(species);
    result.reaction = [model, species](const Point& point, Real state, Real coupled, Real time) {
        const auto reaction = species == GrayScottSpecies::U ? model.reaction(state, coupled).u : model.reaction(coupled, state).v;
        const auto& parameters = model.parameters();
        const auto& source = species == GrayScottSpecies::U ? parameters.source_u : parameters.source_v;
        return reaction + (source ? source(point, time) : 0.);
    };
    result.state_derivative = [model, species](const Point&, Real state, Real coupled, Real) {
        const auto jacobian = model.reaction_jacobian(species == GrayScottSpecies::U ? state : coupled,
                                                      species == GrayScottSpecies::U ? coupled : state);
        return species == GrayScottSpecies::U ? jacobian[0][0] : jacobian[1][1];
    };
    return result;
}

void TransientReactionDiffusionNativeAdapter::initialize()
{
    if (initialized_) return;
    component_.init();
    PetscOwner<Vec, VecDestroy> zero_input;
    *zero_input.put() = component_.system().create_vector();
    VecSet(zero_input.get(), 0);
    component_.bind_input(zero_input.get());
    VecSet(component_.solution(), initial_state_);
    if (VecCopy(component_.solution(), component_.trial_solution()) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to initialize transient trial state"));
    initialized_ = true;
}

void TransientReactionDiffusionNativeAdapter::require_port(const PortId& port, PortDirection direction) const
{
    if ((direction == PortDirection::Output && port != "state") ||
        (direction == PortDirection::Input && port != "coupled"))
        KARMA_THROW(InvalidArgument("unknown transient native adapter port: " + port));
}

KARMAField TransientReactionDiffusionNativeAdapter::field(Vec vector)
{
    return KARMAField(PetscFieldBinding{&component_.system(), vector, 0, component_.system().variable(0)});
}

KARMAField TransientReactionDiffusionNativeAdapter::input_field(PortId port)
{
    require_port(port, PortDirection::Input);
    return field(component_.input());
}

KARMAField TransientReactionDiffusionNativeAdapter::output_field(PortId port, FieldVisibility visibility)
{
    require_port(port, PortDirection::Output);
    return field(visibility == FieldVisibility::Committed ? component_.solution() : component_.trial_solution());
}

void TransientReactionDiffusionNativeAdapter::checkpoint() { component_.checkpoint(); }

void TransientReactionDiffusionNativeAdapter::bind_trial_input(PortId port, Vec input)
{
    require_port(port, PortDirection::Input);
    component_.bind_input(input);
}

void TransientReactionDiffusionNativeAdapter::advance(const TimeInterval& interval)
{
    component_.advance(interval.begin, interval.end);
}

void TransientReactionDiffusionNativeAdapter::commit() { component_.commit(); }

void TransientReactionDiffusionNativeAdapter::release_checkpoint() noexcept
{
    component_.release_checkpoint();
}

bool TransientReactionDiffusionNativeAdapter::restore() noexcept { return component_.restore(); }

Mat TransientReactionDiffusionNativeAdapter::mass_matrix(PortId port)
{
    require_port(port, PortDirection::Output);
    return component_.mass();
}

KARMASystem& TransientReactionDiffusionNativeAdapter::checkpoint_system()
{
    return component_.system();
}

Real TransientReactionDiffusionNativeAdapter::committed_time() const noexcept
{
    return component_.committed_time();
}

Vec TransientReactionDiffusionNativeAdapter::committed_solution() const
{
    return component_.solution();
}

Mat TransientReactionDiffusionNativeAdapter::mass() const { return component_.mass(); }

const TransientSolveDiagnostics& TransientReactionDiffusionNativeAdapter::diagnostics() const noexcept
{
    return component_.diagnostics();
}

void TransientReactionDiffusionNativeAdapter::set_initial_condition(
    const std::function<Real(const Point&)>& value)
{
    const bool invalid_request = !initialized_ || !value;
    detail::collective_signature_error(
        communicator_, invalid_request ? "initial condition requires an initialized adapter and callback" : "");
    auto& system = component_.system();
    const auto& vertices = system.mesh().vertices();
    std::vector<Real> coordinates;
    std::vector<PetscInt> indices;
    std::vector<PetscScalar> values;
    std::string local_error;
    try
    {
        system.mesh().vertex_coordinates(vertices, coordinates);
        indices.reserve(vertices.size());
        values.reserve(vertices.size());
        std::size_t ordinal = 0;
        for (const auto vertex : vertices)
        {
            const Point point{coordinates[3 * ordinal], coordinates[3 * ordinal + 1],
                              coordinates[3 * ordinal + 2]};
            const PetscScalar initial = value(point);
            if (!std::isfinite(initial))
                KARMA_THROW(InvalidArgument("initial-condition callback returned a nonfinite value"));
            indices.push_back(system.dof_map().vertex_dof(vertex, 0));
            values.push_back(initial);
            ++ordinal;
        }
    }
    catch (const std::exception& error) { local_error = error.what(); }
    catch (...) { local_error = "unknown initial-condition callback failure"; }
    detail::collective_signature_error(communicator_, local_error);

    PetscOwner<Vec, VecDestroy> staged;
    bool operation_failed = VecDuplicate(component_.solution(), staged.put()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to allocate transient initial condition" : "");
    operation_failed = VecSet(staged.get(), 0) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to clear transient initial condition" : "");
    operation_failed = VecSetValues(staged.get(), static_cast<PetscInt>(indices.size()), indices.data(), values.data(),
                                    INSERT_VALUES) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to stage transient initial condition values" : "");
    operation_failed = VecAssemblyBegin(staged.get()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to begin transient initial condition assembly" : "");
    operation_failed = VecAssemblyEnd(staged.get()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to end transient initial condition assembly" : "");

    PetscOwner<Vec, VecDestroy> old_committed;
    PetscOwner<Vec, VecDestroy> old_trial;
    operation_failed = VecDuplicate(component_.solution(), old_committed.put()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to allocate committed initial-state snapshot" : "");
    operation_failed = VecDuplicate(component_.trial_solution(), old_trial.put()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to allocate trial initial-state snapshot" : "");
    operation_failed = VecCopy(component_.solution(), old_committed.get()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to snapshot committed initial state" : "");
    operation_failed = VecCopy(component_.trial_solution(), old_trial.get()) != PETSC_SUCCESS;
    detail::collective_signature_error(communicator_, operation_failed ? "failed to snapshot trial initial state" : "");

    try
    {
        operation_failed = VecCopy(staged.get(), component_.solution()) != PETSC_SUCCESS;
        detail::collective_signature_error(
            communicator_, operation_failed ? "failed to publish transient committed initial condition" : "");
        operation_failed = VecCopy(staged.get(), component_.trial_solution()) != PETSC_SUCCESS;
        detail::collective_signature_error(
            communicator_, operation_failed ? "failed to publish transient trial initial condition" : "");
    }
    catch (...)
    {
        std::exception_ptr publication_failure = std::current_exception();
        std::string rollback_error;
        operation_failed = VecCopy(old_committed.get(), component_.solution()) != PETSC_SUCCESS;
        try
        {
            detail::collective_signature_error(
                communicator_, operation_failed ? "failed to roll back committed initial state" : "");
        }
        catch (const std::exception& error) { rollback_error = error.what(); }
        operation_failed = VecCopy(old_trial.get(), component_.trial_solution()) != PETSC_SUCCESS;
        try
        {
            detail::collective_signature_error(
                communicator_, operation_failed ? "failed to roll back trial initial state" : "");
        }
        catch (const std::exception& error)
        {
            if (!rollback_error.empty()) rollback_error += "; ";
            rollback_error += error.what();
        }
        if (!rollback_error.empty())
            KARMA_THROW(InvalidArgument("initial-condition publication failed and rollback failed: " + rollback_error));
        std::rethrow_exception(publication_failure);
    }
}

void TransientReactionDiffusionNativeAdapter::write_checkpoint(std::string_view filename)
{
    auto& system = component_.system();
    KARMAMoabIO(system.mesh()).write_checkpoint(filename, system, component_.solution());
}

void TransientReactionDiffusionNativeAdapter::restore_committed_solution(Vec staged, Real time)
{
    component_.restore_committed_solution(staged, time);
}
}  // namespace KARMA
