#include "karma_operator_split_graph.h"

#include "karma_collective_outcome.h"
#include "karma_compatibility.h"
#include "karma_semantics_adapter.h"

#include "karma_compiled_fem_connection.h"
#include "karma_exceptions.h"
#include "petsc_raii.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace KARMA
{
namespace
{
const ComponentInstanceSpec& find_component(const OperatorSplitGraphSpec& graph, const std::string& id)
{
    const auto found = std::find_if(graph.components.begin(), graph.components.end(),
                                    [&](const auto& component) { return component.id == id; });
    if (found == graph.components.end()) throw std::invalid_argument("connection references unknown component");
    return *found;
}

const PortSpec& find_port(const ComponentSpec& component, const PortId& id)
{
    const auto found = std::find_if(component.ports.begin(), component.ports.end(),
                                    [&](const auto& port) { return port.id == id; });
    if (found == component.ports.end()) throw std::invalid_argument("connection references unknown port");
    return *found;
}

void validate_connection(const OperatorSplitGraphSpec& graph, const ConnectionSpec& connection)
{
    const auto& source = find_component(graph, connection.source_component);
    const auto& target = find_component(graph, connection.target_component);
    const auto& source_port = find_port(source.component, connection.source_port);
    const auto& target_port = find_port(target.component, connection.target_port);
    if (source_port.direction != PortDirection::Output || target_port.direction != PortDirection::Input)
        throw std::invalid_argument("connections must point from output to input");
    // Whether the two fields AGREE is decided by validate_connection_semantics
    // below, not here. This used to compare unit strings for equality, which
    // is the defect the dimensional model exists to remove: "m/s" and
    // "m s^-1" are the same physical quantity and were refused as different,
    // while the comparison could say nothing about scale, offset, role or the
    // version window. Structural facts that are not about meaning - the
    // direction of the connection - stay here.
}

CompiledInstruction exchange(const ConnectionSpec& connection)
{
    return {InstructionKind::Exchange, {}, connection.source_component, connection.target_component, AdvanceInterval::None};
}

CompiledInstruction advance(const std::string& component, AdvanceInterval interval)
{
    return {InstructionKind::Advance, component, {}, {}, interval};
}

int component_index(const CompiledGraphMetadata& metadata, const std::string& id)
{
    for (int index = 0; index < 2; ++index)
        if (metadata.components[static_cast<std::size_t>(index)].id == id) return index;
    throw std::logic_error("compiled instruction references unknown component");
}

std::size_t connection_index(const CompiledGraphMetadata& metadata, const CompiledInstruction& instruction)
{
    for (std::size_t index = 0; index < metadata.connections.size(); ++index)
    {
        const auto& connection = metadata.connections[index];
        if (connection.source_component == instruction.source_component &&
            connection.target_component == instruction.target_component)
            return index;
    }
    throw std::logic_error("compiled exchange references unknown connection");
}

const ConnectionSpec& find_connection(const CompiledGraphMetadata& metadata, const CompiledInstruction& instruction)
{
    return metadata.connections[connection_index(metadata, instruction)];
}

TimeInterval interval_for(const CompiledInstruction& instruction, Real left, Real right)
{
    const Real midpoint = left + (right - left) / 2;
    switch (instruction.interval)
    {
    case AdvanceInterval::Full: return {left, right};
    case AdvanceInterval::HalfFirst: return {left, midpoint};
    case AdvanceInterval::HalfSecond: return {midpoint, right};
    case AdvanceInterval::None: break;
    }
    throw std::logic_error("advance instruction has no interval");
}

/** Reconcile a rank-local failure across `communicator` and throw an agreed
 * exception naming the instruction, or return if every rank succeeded.
 *
 * The reduction rule is NOT here. It was one of five copies of the rule
 * `reconcile_outcome()` implements, and the transaction layer's invariant says
 * exactly one exists in these sources; `reconcile_failure_message()` is it.
 * What remains here is this path's own context - which instruction - which no
 * reason code carries.
 */
void reconcile_or_throw(MPI_Comm communicator, bool failed, const std::string& local_message, std::size_t instruction)
{
    const auto message = reconcile_failure_message(communicator, failed, local_message);
    if (message.empty()) return;
    throw std::runtime_error("operator-split instruction " + std::to_string(instruction) + " failed on " + message);
}

const char* split_name(SplitScheme scheme)
{
    return scheme == SplitScheme::Lie ? "Lie" : "Strang";
}

const char* instruction_name(InstructionKind kind)
{
    switch (kind)
    {
    case InstructionKind::AcquireCheckpoint: return "checkpoint";
    case InstructionKind::Exchange: return "exchange";
    case InstructionKind::Advance: return "advance";
    case InstructionKind::Commit: return "commit";
    case InstructionKind::ReleaseCheckpoint: return "release";
    }
    return "unknown";
}

bool same_field(const FieldSpec& left, const FieldSpec& right)
{
    return left.id == right.id && left.units == right.units && left.components == right.components &&
           left.role == right.role && left.location == right.location && left.temporal == right.temporal;
}

bool same_port(const PortSpec& left, const PortSpec& right)
{
    return left.id == right.id && left.direction == right.direction && left.required == right.required &&
           same_field(left.field, right.field);
}

bool same_connection_endpoints(const ConnectionSpec& left, const ConnectionSpec& right)
{
    return std::tie(left.source_component, left.source_port, left.target_component, left.target_port) ==
           std::tie(right.source_component, right.source_port, right.target_component, right.target_port);
}

bool same_instruction(const CompiledInstruction& left, const CompiledInstruction& right)
{
    return left.kind == right.kind && left.component == right.component &&
           left.source_component == right.source_component && left.target_component == right.target_component &&
           left.interval == right.interval;
}

std::string canonical_description(const CompiledGraphMetadata& metadata, const std::vector<CompiledInstruction>& instructions)
{
    std::vector<std::string> components;
    for (const auto& instance : metadata.components)
    {
        std::vector<std::string> ports;
        for (const auto& port : instance.component.ports)
        {
            ports.push_back(port.id + ":" + std::to_string(static_cast<int>(port.direction)) + ":" +
                            std::to_string(port.required) + ":" + port.field.id + ":" + port.field.units + ":" +
                            std::to_string(port.field.components) + ":" + std::to_string(static_cast<int>(port.field.role)) +
                            ":" + std::to_string(static_cast<int>(port.field.location)) + ":" +
                            std::to_string(static_cast<int>(port.field.temporal)));
        }
        std::sort(ports.begin(), ports.end());
        std::ostringstream item;
        item << instance.id << ':' << instance.component.id << ':' << instance.component.supports_fractional_steps << ':'
             << instance.component.supports_checkpoint_restore << ":mesh=" << instance.mesh_id
             << ":layout=" << instance.layout_id << ":basis=" << instance.basis_id
             << ":solver=" << instance.solver_policy_id;
        for (const auto& port : ports) item << '|' << port;
        components.push_back(item.str());
    }
    std::sort(components.begin(), components.end());

    std::vector<std::string> connections;
    for (const auto& connection : metadata.connections)
        connections.push_back(connection.source_component + ":" + connection.source_port + ">" +
                              connection.target_component + ":" + connection.target_port + ":transfer=" +
                              connection.transfer_id + ":solver=" + connection.solver_policy_id);
    std::sort(connections.begin(), connections.end());

    std::ostringstream result;
    result << "graph=" << metadata.id << "|scheme=" << split_name(metadata.scheme);
    for (const auto& component : components) result << "|component=" << component;
    for (const auto& connection : connections) result << "|connection=" << connection;
    for (const auto& instruction : instructions)
        result << "|instruction=" << instruction_name(instruction.kind) << ':' << instruction.component << ':'
               << instruction.source_component << ':' << instruction.target_component << ':'
               << static_cast<int>(instruction.interval);
    return result.str();
}


}  // namespace

CompiledOperatorSplitGraph::CompiledOperatorSplitGraph(CompiledGraphMetadata metadata,
                                                       std::vector<CompiledInstruction> instructions)
    : metadata_(std::move(metadata)), instructions_(std::move(instructions))
{
}

std::string CompiledOperatorSplitGraph::signature() const
{
    return canonical_description(metadata_, instructions_);
}

CompiledGraphReport CompiledOperatorSplitGraph::report() const
{
    CompiledGraphReport result;
    result.signature = signature();
    result.scheme = metadata_.scheme;
    for (const auto& component : metadata_.components)
        result.components.push_back({component.id, component.component.ports, component.component.supports_fractional_steps,
                                     component.component.supports_checkpoint_restore, component.mesh_id, component.layout_id,
                                     component.basis_id, component.solver_policy_id,
                                     semantic_hash(serialize_component_semantics(
                                         component_semantics_from_spec(component.component)))});
    std::sort(result.components.begin(), result.components.end(),
              [](const auto& left, const auto& right) { return left.id < right.id; });
    for (const auto& connection : metadata_.connections)
        result.connections.push_back({connection, connection.transfer_id, connection.solver_policy_id,
                                      "not initialized", "unknown"});
    std::sort(result.connections.begin(), result.connections.end(), [](const auto& left, const auto& right) {
        return std::tie(left.connection.source_component, left.connection.source_port, left.connection.target_component,
                        left.connection.target_port) <
               std::tie(right.connection.source_component, right.connection.source_port, right.connection.target_component,
                        right.connection.target_port);
    });
    result.instructions = instructions_;
    result.capabilities = {"checkpoint rollback", "Lie/Strang operator splitting", "persistent scalar cG transfer"};
    result.deferred_features = {"general graph topology", "atomic durable graph-wide restart", "Anderson acceleration"};
    return result;
}

std::vector<std::string> CompiledOperatorSplitGraph::diff(const CompiledOperatorSplitGraph& other) const
{
    std::vector<std::string> differences;
    if (metadata_.id != other.metadata_.id) differences.push_back("component metadata differs");
    if (metadata_.scheme != other.metadata_.scheme || instructions_.size() != other.instructions_.size() ||
        !std::equal(instructions_.begin(), instructions_.end(), other.instructions_.begin(), same_instruction))
        differences.push_back("schedule differs");
    for (std::size_t index = 0; index < metadata_.components.size(); ++index)
    {
        const auto& left = metadata_.components[index];
        const auto& right = other.metadata_.components[index];
        if (left.id != right.id || left.component.id != right.component.id ||
            left.component.supports_fractional_steps != right.component.supports_fractional_steps ||
            left.component.supports_checkpoint_restore != right.component.supports_checkpoint_restore)
            differences.push_back("component metadata differs");
        if (left.mesh_id != right.mesh_id || left.layout_id != right.layout_id || left.basis_id != right.basis_id)
            differences.push_back("mesh/layout metadata differs");
        if (left.solver_policy_id != right.solver_policy_id) differences.push_back("solver policy differs");
        for (std::size_t port = 0; port < left.component.ports.size() && port < right.component.ports.size(); ++port)
        {
            const auto& left_port = left.component.ports[port];
            const auto& right_port = right.component.ports[port];
            if (left_port.id != right_port.id || left_port.direction != right_port.direction ||
                left_port.required != right_port.required)
                differences.push_back("port metadata differs");
            if (!same_field(left_port.field, right_port.field)) differences.push_back("field metadata differs");
        }
        if (left.component.ports.size() != right.component.ports.size()) differences.push_back("port metadata differs");
    }
    for (std::size_t index = 0; index < metadata_.connections.size(); ++index)
    {
        const auto& left = metadata_.connections[index];
        const auto& right = other.metadata_.connections[index];
        if (!same_connection_endpoints(left, right) || left.transfer_id != right.transfer_id)
            differences.push_back("transfer metadata differs");
        if (left.solver_policy_id != right.solver_policy_id) differences.push_back("solver policy differs");
    }
    std::sort(differences.begin(), differences.end());
    differences.erase(std::unique(differences.begin(), differences.end()), differences.end());
    return differences;
}

KARMACouplingScheduler::KARMACouplingScheduler(CompiledOperatorSplitGraph graph,
                                               std::array<KARMANativeComponent*, 2> components,
                                               SchedulerOptions options)
    : graph_(std::move(graph)), components_(components), options_(options)
{
    if (components_[0] == nullptr || components_[1] == nullptr || components_[0] == components_[1])
        throw std::invalid_argument("operator-split scheduler requires two distinct components");
    int initialized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized)
        throw std::invalid_argument("scheduler requires initialized MPI");
    const MPI_Comm communicator = components_[0]->communicator();
    bool metadata_failed = false;
    std::string metadata_message;
    try
    {
        const auto supplied = components_;
        for (std::size_t index = 0; index < components_.size(); ++index)
        {
            const auto& expected = graph_.metadata().components[index];
            const auto found = std::find_if(supplied.begin(), supplied.end(), [&](const auto* component) {
                return component->specification().id == expected.id;
            });
            if (found == supplied.end()) throw std::invalid_argument("runtime component ids do not match compiled graph");
            components_[index] = *found;
            const auto& actual = components_[index]->specification();
            if (!actual.supports_checkpoint_restore || actual.supports_fractional_steps !=
                    expected.component.supports_fractional_steps || actual.ports.size() != expected.component.ports.size())
                throw std::invalid_argument("runtime component metadata does not match compiled graph");
            for (const auto& expected_port : expected.component.ports)
            {
                const auto actual_port = std::find_if(actual.ports.begin(), actual.ports.end(),
                                                      [&](const auto& port) { return port.id == expected_port.id; });
                if (actual_port == actual.ports.end() || !same_port(*actual_port, expected_port))
                    throw std::invalid_argument("runtime component metadata does not match compiled graph");
            }
        }
    }
    catch (const std::exception& error) { metadata_failed = true; metadata_message = error.what(); }
    catch (...) { metadata_failed = true; metadata_message = "unknown runtime component metadata failure"; }
    reconcile_or_throw(communicator, metadata_failed, metadata_message, 0);
    int congruent = MPI_UNEQUAL;
    const bool communicator_failed =
        MPI_Comm_compare(components_[0]->communicator(), components_[1]->communicator(), &congruent) != MPI_SUCCESS ||
        (congruent != MPI_IDENT && congruent != MPI_CONGRUENT);
    reconcile_or_throw(communicator, communicator_failed,
                       communicator_failed ? "runtime component communicators are not congruent" : "", 0);
    std::array<bool, 2> initialization_failed{};
    std::array<std::string, 2> initialization_messages;
    for (std::size_t index = 0; index < components_.size(); ++index)
    {
        try { components_[index]->initialize(); }
        catch (const std::exception& error) { initialization_failed[index] = true; initialization_messages[index] = error.what(); }
        catch (...) { initialization_failed[index] = true; initialization_messages[index] = "unknown component initialization failure"; }
    }
    for (std::size_t index = 0; index < components_.size(); ++index)
        reconcile_or_throw(components_[index]->communicator(), initialization_failed[index], initialization_messages[index], 0);
    std::array<bool, 2> staging_failed{};
    std::array<std::string, 2> staging_messages;
    for (std::size_t index = 0; index < components_.size(); ++index)
    {
        try
        {
            const auto& connection = graph_.metadata().connections[index];
            const int target = component_index(graph_.metadata(), connection.target_component);
            if (options_.exchange_hook != nullptr && options_.staging_factory != nullptr)
            {
                const int source = component_index(graph_.metadata(), connection.source_component);
                staging_[index] = options_.staging_factory(connection, *components_[static_cast<std::size_t>(source)],
                                                            *components_[static_cast<std::size_t>(target)]);
            }
            else if (options_.exchange_hook != nullptr)
            {
                const auto target_field = components_[static_cast<std::size_t>(target)]->input_field(connection.target_port);
                const auto& binding = std::get<PetscFieldBinding>(target_field.binding());
                staging_[index] = binding.system->create_vector();
            }
            if (options_.exchange_hook != nullptr && staging_[index] == nullptr)
            {
                staging_failed[index] = true;
                staging_messages[index] = "failed to allocate edge staging vector";
            }
        }
        catch (const std::exception& error) { staging_failed[index] = true; staging_messages[index] = error.what(); }
        catch (...) { staging_failed[index] = true; staging_messages[index] = "unknown edge staging failure"; }
    }
    try
    {
        for (std::size_t index = 0; index < components_.size(); ++index)
            reconcile_or_throw(communicator, staging_failed[index], staging_messages[index], index);
    }
    catch (...)
    {
        if (options_.exchange_hook != nullptr)
            for (auto& vector : staging_)
                if (vector != nullptr) VecDestroy(&vector);
        throw;
    }
    state_ = SchedulerState::Ready;
}

KARMACouplingScheduler::KARMACouplingScheduler(CompiledOperatorSplitGraph graph,
                                               std::array<KARMANativeComponent*, 2> components,
                                               SchedulerPosition position, SchedulerOptions options)
    : KARMACouplingScheduler(std::move(graph), components, options)
{
    const bool invalid = !std::isfinite(position.time) || position.time < 0 ||
                         ((position.step == 0) != (position.time == 0));
    reconcile_or_throw(components_[0]->communicator(), invalid,
                       invalid ? "restart scheduler position is invalid" : "", 0);
    position_ = position;
}

KARMACouplingScheduler::~KARMACouplingScheduler()
{
    if (options_.exchange_hook != nullptr)
        for (auto& vector : staging_)
            if (vector != nullptr) VecDestroy(&vector);
}

std::array<std::size_t, 2> KARMACouplingScheduler::connection_setup_counts() const noexcept
{
    return {connections_[0] ? connections_[0]->setup_count() : 0,
            connections_[1] ? connections_[1]->setup_count() : 0};
}

std::array<FEMConnectionDiagnostics, 2> KARMACouplingScheduler::connection_diagnostics() const noexcept
{
    return {connections_[0] ? connections_[0]->diagnostics() : FEMConnectionDiagnostics{},
            connections_[1] ? connections_[1]->diagnostics() : FEMConnectionDiagnostics{}};
}

OperatorSplitRuntimeReport KARMACouplingScheduler::report() const
{
    OperatorSplitRuntimeReport result;
    result.state = state_;
    result.position = position_;
    result.schedule = graph_.instructions();
    for (std::size_t index = 0; index < graph_.metadata().connections.size(); ++index)
    {
        RuntimeConnectionReport connection;
        connection.connection = graph_.metadata().connections[index];
        if (connections_[index])
        {
            connection.setup_count = connections_[index]->setup_count();
            connection.setup = connection.setup_count == 0 ? "not initialized" : "initialized";
            connection.diagnostics = connections_[index]->diagnostics();
            connection.coverage = connection.diagnostics.total_points == 0
                                      ? "unknown"
                                      : std::to_string(connection.diagnostics.covered_points) + "/" +
                                            std::to_string(connection.diagnostics.total_points);
        }
        result.connections.push_back(std::move(connection));
    }
    return result;
}

void KARMACouplingScheduler::advance(Real time_left, Real time_right)
{
    if (state_ == SchedulerState::Failed)
        throw std::logic_error("scheduler is in a terminal failed state after a rollback failure");
    const bool invalid_interval = !std::isfinite(time_left) || !std::isfinite(time_right) || time_right <= time_left;
    const bool invalid_start = position_.step == 0 ? time_left != 0 : time_left != position_.time;
    reconcile_or_throw(components_[0]->communicator(), invalid_interval || invalid_start,
                       invalid_interval ? "operator-split macro interval must be positive and finite"
                                        : "operator-split macro interval does not start at scheduler time",
                       0);

    state_ = SchedulerState::Executing;
    const SchedulerPosition saved = position_;
    try
    {
        std::array<bool, 2> advanced{};
        for (std::size_t instruction_index = 0; instruction_index < graph_.instructions().size(); ++instruction_index)
        {
            const auto& instruction = graph_.instructions()[instruction_index];
            bool failed = false;
            std::string local_message;
            try
            {
                switch (instruction.kind)
                {
                case InstructionKind::AcquireCheckpoint:
                {
                    std::array<bool, 2> component_failed{};
                    std::array<std::string, 2> component_messages;
                     for (std::size_t component_index_value = 0; component_index_value < components_.size(); ++component_index_value)
                     {
                         try { components_[component_index_value]->checkpoint(); }
                         catch (const std::exception& error) { component_failed[component_index_value] = true; component_messages[component_index_value] = error.what(); }
                         catch (...) { component_failed[component_index_value] = true; component_messages[component_index_value] = "unknown checkpoint failure"; }
                         reconcile_or_throw(components_[component_index_value]->communicator(), component_failed[component_index_value],
                                            component_messages[component_index_value], instruction_index);
                     }
                    break;
                }
                case InstructionKind::Exchange:
                {
                    const int source = component_index(graph_.metadata(), instruction.source_component);
                    const int target = component_index(graph_.metadata(), instruction.target_component);
                    const auto& connection = find_connection(graph_.metadata(), instruction);
                    const auto visibility = advanced[static_cast<std::size_t>(source)] ?
                        FieldVisibility::TransactionLocal : FieldVisibility::Committed;
                    const auto edge = connection_index(graph_.metadata(), instruction);
                    if (options_.exchange_hook != nullptr)
                    {
                        options_.exchange_hook(connection, *components_[static_cast<std::size_t>(source)],
                                               *components_[static_cast<std::size_t>(target)], visibility,
                                               staging_[edge]);
                    }
                    else
                    {
                        const auto source_field = components_[static_cast<std::size_t>(source)]->output_field(
                            connection.source_port, visibility);
                        if (!connections_[edge])
                        {
                            const auto target_field = components_[static_cast<std::size_t>(target)]->input_field(
                                connection.target_port);
                            const auto& target_binding = std::get<PetscFieldBinding>(target_field.binding());
                            if (target_binding.system == nullptr)
                                throw std::invalid_argument("FEM exchange requires a PETSc target binding");
                            connections_[edge] = std::make_unique<CompiledFEMConnection>(
                                source_field, *target_binding.system, target_binding.variable, 5);
                            connections_[edge]->setup();
                        }
                        connections_[edge]->apply(source_field);
                        staging_[edge] = connections_[edge]->staging();
                    }
                    components_[static_cast<std::size_t>(target)]->bind_trial_input(connection.target_port,
                                                                                  staging_[edge]);
                    break;
                }
                case InstructionKind::Advance:
                {
                    const int component = component_index(graph_.metadata(), instruction.component);
                    components_[static_cast<std::size_t>(component)]->advance(
                        interval_for(instruction, time_left, time_right));
                    advanced[static_cast<std::size_t>(component)] = true;
                    break;
                }
                case InstructionKind::Commit:
                {
                    std::array<bool, 2> component_failed{};
                    std::array<std::string, 2> component_messages;
                     for (std::size_t component_index_value = 0; component_index_value < components_.size(); ++component_index_value)
                     {
                         component_failed[component_index_value] = !advanced[component_index_value];
                         component_messages[component_index_value] = component_failed[component_index_value] ?
                             "commit requires an advance for every component" : "";
                         if (!component_failed[component_index_value])
                         {
                             try { components_[component_index_value]->commit(); }
                             catch (const std::exception& error) { component_failed[component_index_value] = true; component_messages[component_index_value] = error.what(); }
                             catch (...) { component_failed[component_index_value] = true; component_messages[component_index_value] = "unknown commit failure"; }
                         }
                         reconcile_or_throw(components_[component_index_value]->communicator(), component_failed[component_index_value],
                                            component_messages[component_index_value], instruction_index);
                     }
                    break;
                }
                case InstructionKind::ReleaseCheckpoint:
                    for (auto* component : components_) component->release_checkpoint();
                    for (auto* component : components_)
                        reconcile_or_throw(component->communicator(), false, "", instruction_index);
                    break;
                }
            }
            catch (const std::exception& error)
            {
                failed = true;
                local_message = error.what();
            }
            catch (...)
            {
                failed = true;
                local_message = "unknown scheduler instruction failure";
            }
            reconcile_or_throw(components_[0]->communicator(), failed, local_message, instruction_index);
        }
        position_ = {time_right, saved.step + 1};
        state_ = SchedulerState::Ready;
    }
    catch (const std::exception& original)
    {
        const std::string original_message = original.what();
        std::array<bool, 2> restore_failed{};
        for (std::size_t index = 0; index < components_.size(); ++index)
            restore_failed[index] = !components_[index]->restore();
        try
        {
            for (std::size_t index = 0; index < components_.size(); ++index)
                reconcile_or_throw(components_[index]->communicator(), restore_failed[index],
                                   restore_failed[index] ? "component restore failed" : "", 0);
        }
        catch (const std::exception& rollback)
        {
            state_ = SchedulerState::Failed;
            throw std::runtime_error("operator-split step failed: " + original_message +
                                     "; additionally rollback restore failed and the scheduler is now unusable: " +
                                     rollback.what());
        }
        position_ = saved;
        state_ = SchedulerState::Ready;
        throw;
    }
    catch (...)
    {
        std::array<bool, 2> restore_failed{};
        for (std::size_t index = 0; index < components_.size(); ++index)
            restore_failed[index] = !components_[index]->restore();
        try
        {
            for (std::size_t index = 0; index < components_.size(); ++index)
                reconcile_or_throw(components_[index]->communicator(), restore_failed[index],
                                   restore_failed[index] ? "component restore failed" : "", 0);
        }
        catch (const std::exception& rollback)
        {
            state_ = SchedulerState::Failed;
            throw std::runtime_error("operator-split step failed: unknown scheduler failure; additionally rollback restore "
                                     "failed and the scheduler is now unusable: " + std::string(rollback.what()));
        }
        position_ = saved;
        state_ = SchedulerState::Ready;
        throw;
    }
}

FixedPointResult KARMACouplingScheduler::advance_fixed_point(Real time_left, Real time_right,
                                                              FixedPointOptions options)
{
    reconcile_or_throw(components_[0]->communicator(), state_ == SchedulerState::Failed,
                       state_ == SchedulerState::Failed
                           ? "scheduler is in a terminal failed state after a rollback failure"
                           : "", 0);
    const bool invalid_interval = !std::isfinite(time_left) || !std::isfinite(time_right) || time_right <= time_left;
    const bool invalid_start = position_.step == 0 ? time_left != 0 : time_left != position_.time;
    reconcile_or_throw(components_[0]->communicator(), invalid_interval || invalid_start,
                       invalid_interval ? "fixed-point macro interval must be positive and finite"
                                        : "fixed-point macro interval does not start at scheduler time", 0);
    reconcile_or_throw(components_[0]->communicator(), options_.exchange_hook != nullptr,
                       options_.exchange_hook != nullptr
                           ? "fixed-point coupling does not support scheduler exchange hooks"
                           : "", 0);

    struct IterateStorage
    {
        std::array<PetscOwner<Vec, VecDestroy>, 2> vectors;
        KARMAFixedPointIterator::Iterate fields;
    };
    IterateStorage initial;
    initial.fields.reserve(components_.size());
    for (std::size_t index = 0; index < components_.size(); ++index)
    {
        std::optional<PetscFieldBinding> binding;
        Mat mass = nullptr;
        bool failed = false;
        std::string message;
        try
        {
            const auto field = components_[index]->output_field("state", FieldVisibility::Committed);
            binding = std::get<PetscFieldBinding>(field.binding());
            mass = components_[index]->mass_matrix("state");
            if (mass == nullptr) throw std::invalid_argument("fixed-point coupling requires a component mass matrix");
        }
        catch (const std::exception& error) { failed = true; message = error.what(); }
        catch (...) { failed = true; message = "unknown fixed-point field/mass preflight failure"; }
        reconcile_or_throw(components_[index]->communicator(), failed, message, index);

        failed = VecDuplicate(binding->vector, initial.vectors[index].put()) != PETSC_SUCCESS;
        reconcile_or_throw(components_[index]->communicator(), failed,
                           failed ? "failed to allocate fixed-point initial iterate" : "", index);
        failed = VecCopy(binding->vector, initial.vectors[index].get()) != PETSC_SUCCESS;
        reconcile_or_throw(components_[index]->communicator(), failed,
                           failed ? "failed to copy fixed-point initial iterate" : "", index);
        initial.fields.push_back({graph_.metadata().components[index].id + ".state",
                                  initial.vectors[index].get(), mass, {}});
    }

    const SchedulerPosition saved = position_;
    state_ = SchedulerState::Executing;
    try
    {
    KARMAFixedPointIterator iterator(components_[0]->communicator(), options);
    const auto component_epoch = [&](std::size_t index, const auto& operation, const char* fallback) {
        bool failed = false;
        std::string message;
        try { operation(); }
        catch (const std::exception& error) { failed = true; message = error.what(); }
        catch (...) { failed = true; message = fallback; }
        reconcile_or_throw(components_[index]->communicator(), failed, message, index);
    };
    const auto result = iterator.solve(
        initial.fields,
        [&] {
            for (std::size_t index = 0; index < components_.size(); ++index)
                component_epoch(index, [&] { components_[index]->checkpoint(); },
                                "unknown fixed-point checkpoint failure");
        },
        [&] {
            bool restored = true;
            for (auto* component : components_) restored = component->restore() && restored;
            return restored;
        },
        [&](const auto& current, const auto& mapped) {
            for (std::size_t edge = 0; edge < graph_.metadata().connections.size(); ++edge)
            {
                const auto& connection = graph_.metadata().connections[edge];
                const int source = component_index(graph_.metadata(), connection.source_component);
                const int target = component_index(graph_.metadata(), connection.target_component);
                std::optional<PetscFieldBinding> component_source_binding;
                component_epoch(static_cast<std::size_t>(source), [&] {
                    const auto component_source_field = components_[static_cast<std::size_t>(source)]->output_field(
                        connection.source_port, FieldVisibility::Committed);
                    component_source_binding = std::get<PetscFieldBinding>(component_source_field.binding());
                }, "unknown fixed-point source binding failure");
                std::optional<KARMAField> source_field;
                component_epoch(static_cast<std::size_t>(source), [&] {
                    source_field.emplace(PetscFieldBinding{
                        component_source_binding->system, current[static_cast<std::size_t>(source)].values,
                        component_source_binding->variable, component_source_binding->specification,
                        component_source_binding->representation});
                }, "unknown fixed-point source field validation failure");
                if (!connections_[edge])
                {
                    std::optional<PetscFieldBinding> target_binding;
                    component_epoch(static_cast<std::size_t>(target), [&] {
                        const auto target_field = components_[static_cast<std::size_t>(target)]->input_field(
                            connection.target_port);
                        target_binding = std::get<PetscFieldBinding>(target_field.binding());
                    }, "unknown fixed-point target binding failure");
                    component_epoch(static_cast<std::size_t>(source), [&] {
                        connections_[edge] = std::make_unique<CompiledFEMConnection>(
                            *source_field, *target_binding->system, target_binding->variable, 5);
                    }, "unknown fixed-point connection construction failure");
                    component_epoch(static_cast<std::size_t>(source), [&] {
                        connections_[edge]->setup();
                    }, "unknown fixed-point connection setup failure");
                }
                component_epoch(static_cast<std::size_t>(source), [&] {
                    connections_[edge]->apply(*source_field);
                    staging_[edge] = connections_[edge]->staging();
                }, "unknown fixed-point connection apply failure");
                component_epoch(static_cast<std::size_t>(target), [&] {
                    components_[static_cast<std::size_t>(target)]->bind_trial_input(connection.target_port,
                                                                                   staging_[edge]);
                }, "unknown fixed-point input binding failure");
            }
            for (std::size_t index = 0; index < components_.size(); ++index)
            {
                component_epoch(index, [&] {
                    components_[index]->advance({time_left, time_right});
                }, "unknown fixed-point component advance failure");
                std::optional<PetscFieldBinding> mapped_binding;
                component_epoch(index, [&] {
                    const auto field = components_[index]->output_field("state", FieldVisibility::TransactionLocal);
                    mapped_binding = std::get<PetscFieldBinding>(field.binding());
                }, "unknown fixed-point mapped output lookup failure");
                component_epoch(index, [&] {
                    if (VecCopy(mapped_binding->vector, mapped[index].values) != PETSC_SUCCESS)
                        throw std::runtime_error("failed to collect fixed-point mapped iterate");
                }, "unknown fixed-point mapped iterate staging failure");
            }
        },
        [&](const auto& converged) {
            for (std::size_t index = 0; index < components_.size(); ++index)
            {
                std::optional<PetscFieldBinding> trial_binding;
                component_epoch(index, [&] {
                    const auto field = components_[index]->output_field("state", FieldVisibility::TransactionLocal);
                    trial_binding = std::get<PetscFieldBinding>(field.binding());
                }, "unknown converged fixed-point output lookup failure");
                component_epoch(index, [&] {
                    if (VecCopy(converged[index].values, trial_binding->vector) != PETSC_SUCCESS)
                        throw std::runtime_error("failed to stage converged fixed-point iterate");
                }, "unknown converged fixed-point staging failure");
            }
            for (std::size_t index = 0; index < components_.size(); ++index)
                component_epoch(index, [&] { components_[index]->commit(); },
                                "unknown fixed-point component commit failure");
            for (auto* component : components_) component->release_checkpoint();
        });
    if (result.status == FixedPointStatus::Converged)
    {
        position_ = {time_right, saved.step + 1};
        state_ = SchedulerState::Ready;
    }
    else
    {
        position_ = saved;
        state_ = result.status == FixedPointStatus::RollbackFailure ? SchedulerState::Failed : SchedulerState::Ready;
        if (result.status != FixedPointStatus::RollbackFailure)
            for (auto* component : components_) component->release_checkpoint();
    }
    return result;
    }
    catch (...)
    {
        position_ = saved;
        if (state_ == SchedulerState::Executing) state_ = SchedulerState::Ready;
        throw;
    }
}


/** Admit a connection only if the two ports agree about what they MEAN, not
 * merely about their names and shapes.
 *
 * This is the point at which the semantic model stops being a description the
 * library carries and starts being something the executing path consults. The
 * check adds a refusal and computes nothing: a connection that was already
 * meaningful compiles to the same instructions it did before, which is what
 * makes the behavioral oracle the right instrument for confirming it.
 *
 * A refusal here is a graph that should never have run. Two fields whose
 * units, shape, location or temporal interpretation disagree were previously
 * connected anyway, and the disagreement showed up as wrong numbers rather
 * than as an error.
 */
void validate_connection_semantics(const OperatorSplitGraphSpec& specification, const ConnectionSpec& connection)
{
    const auto component_by_id = [&specification](const std::string& id) -> const ComponentSpec& {
        for (const auto& instance : specification.components)
            if (instance.id == id) return instance.component;
        throw std::invalid_argument("connection names a component that is not in the graph: " + id);
    };

    const auto source = component_semantics_from_spec(component_by_id(connection.source_component));
    const auto target = component_semantics_from_spec(component_by_id(connection.target_component));
    for (const auto* component : {&source, &target})
    {
        const auto validation = validate_component_semantics(*component);
        if (!validation.ok())
            throw std::invalid_argument(
                "component semantics are invalid: " + component->id + ": diagnostic code " +
                std::to_string(static_cast<unsigned>(validation.diagnostics.front().code)) + " at " +
                validation.diagnostics.front().path);
    }

    const auto port_of = [](const ComponentSemantics& component, const PortId& id) -> const PortSemantics& {
        for (const auto& port : component.ports)
            if (port.id == id) return port;
        throw std::invalid_argument("connection names a port the component does not declare: " + id);
    };
    const auto& source_port = port_of(source, connection.source_port);
    const auto& target_port = port_of(target, connection.target_port);

    ConnectionRequest request;
    request.source = source_port.field;
    request.target = target_port.field;
    request.requirement = target_port.requirement;
    // A wired connection names both endpoints, so the wiring - not a name
    // match - is the authority on which field satisfies the requirement.
    // `PortSpec` carries no requirement of its own, so the adapter defaults
    // `requirement.field_id` to the port's OWN field id; leaving that in place
    // would refuse every connection whose two ports spell their field
    // differently, which is every connection in this graph. Everything else
    // the requirement carries - whether the input is mandatory, the accepted
    // version window, the capabilities it needs - still applies, and so does
    // every check about what the two fields MEAN.
    request.requirement.field_id = request.source.id;
    // The graph declares a projection between non-matching meshes, which is
    // an interpolation and is therefore itself the request for the spatial
    // transfer operator the checker will need.
    request.transfer.kind = TransferKind::Interpolation;
    request.transfer.algorithm = connection.transfer_id;
    request.explicitly_requested = operator_request_bit(OperatorKind::SpatialTransfer);

    const auto compatibility = check_connection(request);
    if (!compatibility.compiles())
    {
        const std::string reason = compatibility.diagnostics.empty()
                                       ? std::string("no diagnostic")
                                       : std::string(connection_reason_name(compatibility.diagnostics.front().reason));
        throw std::invalid_argument("connection " + connection.source_component + ":" + connection.source_port +
                                    " -> " + connection.target_component + ":" + connection.target_port +
                                    " is not semantically admissible: " + reason);
    }
}

CompiledOperatorSplitGraph compile_operator_split_graph(const OperatorSplitGraphSpec& input)
{
    OperatorSplitGraphSpec canonical = input;
    std::sort(canonical.components.begin(), canonical.components.end(),
              [](const auto& left, const auto& right) { return left.id < right.id; });
    for (auto& component : canonical.components)
        std::sort(component.component.ports.begin(), component.component.ports.end(),
                  [](const auto& left, const auto& right) { return left.id < right.id; });
    std::sort(canonical.connections.begin(), canonical.connections.end(), [](const auto& left, const auto& right) {
        return std::tie(left.source_component, left.source_port, left.target_component, left.target_port) <
               std::tie(right.source_component, right.source_port, right.target_component, right.target_port);
    });
    const OperatorSplitGraphSpec& specification = canonical;
    if (specification.id.empty()) throw std::invalid_argument("graph id is empty");
    if (specification.components[0].id.empty() || specification.components[1].id.empty() ||
        specification.components[0].id == specification.components[1].id)
        throw std::invalid_argument("graph component instance ids must be unique and nonempty");
    validate_component_spec(specification.components[0].component);
    validate_component_spec(specification.components[1].component);
    if (specification.components[0].component.id == specification.components[1].component.id ||
        specification.components[0].component.id != specification.components[0].id ||
        specification.components[1].component.id != specification.components[1].id)
        throw std::invalid_argument("component semantic and instance ids must match and be unique");
    if (!specification.components[0].component.supports_checkpoint_restore ||
        !specification.components[1].component.supports_checkpoint_restore)
        throw std::invalid_argument("operator-split components must support checkpoint restore");
    if (specification.scheme == SplitScheme::Strang &&
        (!specification.components[0].component.supports_fractional_steps ||
         !specification.components[1].component.supports_fractional_steps))
        throw std::invalid_argument("Strang requires fractional-step components");

    validate_connection(specification, specification.connections[0]);
    validate_connection(specification, specification.connections[1]);
    validate_connection_semantics(specification, specification.connections[0]);
    validate_connection_semantics(specification, specification.connections[1]);
    if (specification.connections[0].source_component == specification.connections[0].target_component ||
        specification.connections[1].source_component == specification.connections[1].target_component ||
        specification.connections[0].source_component != specification.connections[1].target_component ||
        specification.connections[0].target_component != specification.connections[1].source_component)
        throw std::invalid_argument("graph requires two reciprocal connections");

    const auto& first_connection = *std::find_if(
        specification.connections.begin(), specification.connections.end(), [&](const auto& connection) {
            return connection.source_component == specification.components[1].id &&
                   connection.target_component == specification.components[0].id;
        });
    const auto& second_connection = *std::find_if(
        specification.connections.begin(), specification.connections.end(), [&](const auto& connection) {
            return connection.source_component == specification.components[0].id &&
                   connection.target_component == specification.components[1].id;
        });
    std::vector<CompiledInstruction> instructions;
    instructions.push_back({InstructionKind::AcquireCheckpoint, {}, {}, {}, AdvanceInterval::None});
    instructions.push_back(exchange(first_connection));
    instructions.push_back(advance(specification.components[0].id,
                                   specification.scheme == SplitScheme::Lie ? AdvanceInterval::Full : AdvanceInterval::HalfFirst));
    instructions.push_back(exchange(second_connection));
    instructions.push_back(advance(specification.components[1].id, AdvanceInterval::Full));
    if (specification.scheme == SplitScheme::Strang)
    {
        instructions.push_back(exchange(first_connection));
        instructions.push_back(advance(specification.components[0].id, AdvanceInterval::HalfSecond));
    }
    instructions.push_back({InstructionKind::Commit, {}, {}, {}, AdvanceInterval::None});
    instructions.push_back({InstructionKind::ReleaseCheckpoint, {}, {}, {}, AdvanceInterval::None});
    return {CompiledGraphMetadata{specification.id, specification.components, specification.connections, specification.scheme},
            std::move(instructions)};
}

CompiledOperatorSplitGraph compile_operator_split_graph(const OperatorSplitGraphSpec& specification,
                                                        MPI_Comm communicator)
{
    std::optional<CompiledOperatorSplitGraph> compiled;
    bool failed = false;
    std::string message;
    try { compiled.emplace(compile_operator_split_graph(specification)); }
    catch (const std::exception& error) { failed = true; message = error.what(); }
    catch (...) { failed = true; message = "unknown graph compilation failure"; }
    reconcile_or_throw(communicator, failed, message, 0);
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);
    const std::string local_signature = compiled->signature();
    int reference_length = rank == 0 ? static_cast<int>(local_signature.size()) : 0;
    MPI_Bcast(&reference_length, 1, MPI_INT, 0, communicator);
    std::string reference(static_cast<std::size_t>(reference_length), '\0');
    if (rank == 0) reference = local_signature;
    if (reference_length != 0) MPI_Bcast(reference.data(), reference_length, MPI_CHAR, 0, communicator);
    const bool signature_failed = local_signature != reference;
    reconcile_or_throw(communicator, signature_failed, "compiled graph signature differs from rank 0", 0);
    return std::move(*compiled);
}
}  // namespace KARMA
