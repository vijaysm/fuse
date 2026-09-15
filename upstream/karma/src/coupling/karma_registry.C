#include "karma_registry.h"

#include <algorithm>

namespace KARMA
{
namespace
{
bool version_at_least(SemanticVersion value, SemanticVersion bound) noexcept
{
    return !(value < bound);
}

bool version_at_most(SemanticVersion value, SemanticVersion bound) noexcept
{
    // {0,0,0} is unbounded, not "nothing is acceptable".
    if (bound == SemanticVersion{0, 0, 0}) return true;
    return !(bound < value);
}
} // namespace

std::string_view registry_reason_name(RegistryReason reason) noexcept
{
    switch (reason)
    {
        case RegistryReason::Success: return "success";
        case RegistryReason::DuplicateImplementationId: return "duplicate implementation id";
        case RegistryReason::EmptyImplementationId: return "empty implementation id";
        case RegistryReason::MissingTable: return "missing component table";
        case RegistryReason::TableTooSmall: return "component table too small";
        case RegistryReason::Reentrant: return "registration during a visit";
        case RegistryReason::NotFound: return "not found";
        case RegistryReason::Ambiguous: return "ambiguous selection";
        case RegistryReason::VersionUnsatisfied: return "version unsatisfied";
        case RegistryReason::CapabilityUnsatisfied: return "capability unsatisfied";
        case RegistryReason::EmptyContractId: return "empty contract id";
        case RegistryReason::PortIncompatible: return "no compatible port";
    }
    return {};
}

RegistryReason ComponentRegistry::add(RegistryRecord record)
{
    // Checked first: the alternative is appending to a container something is
    // iterating, which is the defect this registry exists not to have.
    if (_visiting) return RegistryReason::Reentrant;

    if (record.implementation_id.empty()) return RegistryReason::EmptyImplementationId;
    if (record.contract_id.empty()) return RegistryReason::EmptyContractId;
    if (record.table == nullptr) return RegistryReason::MissingTable;
    if (record.table->struct_size < sizeof(karma_component_table)) return RegistryReason::TableTooSmall;

    const auto clash = std::find_if(_records.begin(), _records.end(), [&](const auto& existing) {
        return existing->implementation_id == record.implementation_id;
    });
    if (clash != _records.end()) return RegistryReason::DuplicateImplementationId;

    _records.push_back(std::make_unique<RegistryRecord>(std::move(record)));
    return RegistryReason::Success;
}

std::vector<const RegistryRecord*> ComponentRegistry::inventory() const
{
    std::vector<const RegistryRecord*> records;
    records.reserve(_records.size());
    for (const auto& record : _records) records.push_back(record.get());
    std::sort(records.begin(), records.end(),
              [](const RegistryRecord* left, const RegistryRecord* right) {
                  return left->implementation_id < right->implementation_id;
              });
    return records;
}

void ComponentRegistry::visit(const std::function<void(const RegistryRecord&)>& callback) const
{
    // The snapshot is taken before the guard is raised, and the guard is
    // released whatever the callback does, including throwing.
    const auto snapshot = inventory();
    _visiting = true;
    struct Release
    {
        bool& flag;
        ~Release() { flag = false; }
    } release{_visiting};
    for (const auto* record : snapshot) callback(*record);
}

SelectionResult ComponentRegistry::select(const SelectionRequest& request) const
{
    SelectionResult result;

    std::vector<const RegistryRecord*> candidates;
    // A named implementation is looked up by name and by name only. If it is
    // absent, that is the answer; another implementation of the same contract
    // is not an answer.
    for (const auto* record : inventory())
    {
        if (!request.implementation_id.empty() && record->implementation_id != request.implementation_id) continue;
        if (!request.contract_id.empty() && record->contract_id != request.contract_id) continue;
        candidates.push_back(record);
    }
    if (candidates.empty())
    {
        result.reason = RegistryReason::NotFound;
        return result;
    }

    // Filters are applied after identity so a refusal names the filter that
    // refused rather than reporting the record as absent.
    std::vector<const RegistryRecord*> versioned;
    for (const auto* record : candidates)
        if (version_at_least(record->implementation_version, request.minimum_version) &&
            version_at_most(record->implementation_version, request.maximum_version))
            versioned.push_back(record);
    if (versioned.empty())
    {
        result.reason = RegistryReason::VersionUnsatisfied;
        return result;
    }

    std::vector<const RegistryRecord*> capable;
    std::uint64_t missing = 0;
    for (const auto* record : versioned)
    {
        const auto absent = request.required_capabilities & ~record->provided_capabilities;
        if (absent == 0)
            capable.push_back(record);
        else
            missing |= absent;
    }
    if (capable.empty())
    {
        result.reason = RegistryReason::CapabilityUnsatisfied;
        result.missing_capabilities = missing;
        return result;
    }

    // The port check runs after capability filtering and before the
    // ambiguity policy: it can only narrow the candidate set, never widen
    // it, and an ambiguity among port-compatible candidates is still an
    // ambiguity.
    if (request.required_port_id)
    {
        std::vector<const RegistryRecord*> port_compatible;
        std::optional<ConnectionCompatibility> last_incompatibility;
        for (const auto* record : capable)
        {
            const auto port = std::find_if(record->contract.ports.begin(), record->contract.ports.end(),
                                           [&](const PortSemantics& candidate) {
                                               return candidate.id == *request.required_port_id;
                                           });
            if (port == record->contract.ports.end()) continue;

            ConnectionRequest connection;
            connection.source = *request.required_port_source;
            connection.target = port->field;
            connection.requirement = request.required_port_requirement.value_or(port->requirement);
            connection.allow_automatic_unit_scale = request.allow_automatic_unit_scale;
            connection.available_capabilities = record->provided_capabilities;
            const auto compatibility = check_connection(connection);
            if (compatibility.compiles())
                port_compatible.push_back(record);
            else if (!last_incompatibility)
                last_incompatibility = compatibility;
        }
        if (port_compatible.empty())
        {
            result.reason = RegistryReason::PortIncompatible;
            result.port_incompatibility = last_incompatibility;
            return result;
        }
        capable = std::move(port_compatible);
    }

    if (capable.size() == 1)
    {
        result.record = capable.front();
        result.reason = RegistryReason::Success;
        result.match_count = 1;
        return result;
    }

    result.match_count = capable.size();
    switch (request.policy)
    {
        case SelectionPolicy::RefuseAmbiguous:
            // Choosing here would apply a policy the caller never stated.
            result.reason = RegistryReason::Ambiguous;
            return result;
        case SelectionPolicy::HighestVersion:
            result.record = *std::max_element(capable.begin(), capable.end(),
                                              [](const RegistryRecord* left, const RegistryRecord* right) {
                                                  return left->implementation_version < right->implementation_version;
                                              });
            break;
        case SelectionPolicy::HighestPriority:
            result.record = *std::max_element(capable.begin(), capable.end(),
                                              [](const RegistryRecord* left, const RegistryRecord* right) {
                                                  return left->priority < right->priority;
                                              });
            break;
    }
    result.reason = RegistryReason::Success;
    return result;
}

ComponentRegistry& builtin_component_registry()
{
    static ComponentRegistry registry;
    return registry;
}
} // namespace KARMA
