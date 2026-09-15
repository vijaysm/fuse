// Static component registry: registration, filtering, deterministic order,
// refusal to substitute, and re-entrancy safety.
//
// Needs neither PETSc nor MPI: a registry is a table of descriptions.
#include "karma_registry.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

// A minimal table: the registry only checks that it is present and large
// enough, never calls it.
karma_component_table stub_table()
{
    karma_component_table table{};
    table.struct_size = static_cast<std::uint32_t>(sizeof(table));
    return table;
}

karma_component_table shared_stub = stub_table();

RegistryRecord record(const char* id, SemanticVersion version, const char* contract, std::uint64_t capabilities,
                      int priority = 0)
{
    RegistryRecord entry;
    entry.implementation_id = id;
    entry.implementation_version = version;
    entry.contract_id = contract;
    entry.table = &shared_stub;
    entry.configuration_schema_id = "karma.config.v1";
    entry.provided_capabilities = capabilities;
    entry.provenance = "static";
    entry.priority = priority;
    return entry;
}

SelectionRequest request(const char* contract)
{
    SelectionRequest selection;
    selection.contract_id = contract;
    return selection;
}

constexpr std::uint64_t host_arrays = KARMA_CAPABILITY_HOST_ARRAY_FIELDS;
constexpr std::uint64_t checkpoints = KARMA_CAPABILITY_CHECKPOINT_RESTORE;

void registration()
{
    ComponentRegistry registry;
    require(registry.size() == 0, "a fresh registry is not empty");
    require(registry.add(record("a", {1, 0, 0}, "heat", host_arrays)) == RegistryReason::Success,
            "registration failed");
    require(registry.size() == 1, "the record was not stored");

    // A duplicate identifier is refused, and the first record survives
    // unchanged: last-one-wins would silently replace an implementation.
    auto duplicate = record("a", {9, 9, 9}, "heat", 0);
    require(registry.add(duplicate) == RegistryReason::DuplicateImplementationId, "a duplicate id was accepted");
    require(registry.size() == 1, "a refused duplicate changed the registry");
    const auto found = registry.select(SelectionRequest{}.with_implementation("a"));
    require(found.reason == RegistryReason::Success && found.record != nullptr, "the original record is gone");
    require(found.record->implementation_version == SemanticVersion{1, 0, 0},
            "a refused duplicate overwrote the original version");

    // Structural rejections.
    require(registry.add(record("", {1, 0, 0}, "heat", 0)) == RegistryReason::EmptyImplementationId,
            "an empty identifier was accepted");
    auto no_table = record("b", {1, 0, 0}, "heat", 0);
    no_table.table = nullptr;
    require(registry.add(no_table) == RegistryReason::MissingTable, "a record without a table was accepted");
    auto small_table = record("c", {1, 0, 0}, "heat", 0);
    karma_component_table truncated{};
    truncated.struct_size = 8;
    small_table.table = &truncated;
    require(registry.add(small_table) == RegistryReason::TableTooSmall, "an undersized table was accepted");
    auto no_contract = record("d", {1, 0, 0}, "", 0);
    require(registry.add(no_contract) == RegistryReason::EmptyContractId, "a record without a contract was accepted");
}

void never_substitutes()
{
    ComponentRegistry registry;
    registry.add(record("solid.baseline", {1, 0, 0}, "mechanics", host_arrays));
    registry.add(record("solid.extended", {2, 0, 0}, "mechanics", host_arrays));

    // The point of the whole exercise: asking for an implementation that is
    // not registered is an error, even when another implementation of the
    // same contract is sitting right there. A substitution would change the
    // physics while reporting success.
    const auto missing = registry.select(SelectionRequest{}.with_implementation("solid.absent"));
    require(missing.reason == RegistryReason::NotFound, "an absent implementation was not reported missing");
    require(missing.record == nullptr, "a substitute was returned for an absent implementation");

    // An exact request that also fails a filter is refused for the filter,
    // not silently answered with a different implementation.
    auto demanding = SelectionRequest{}.with_implementation("solid.baseline");
    demanding.required_capabilities = checkpoints;
    const auto refused = registry.select(demanding);
    require(refused.reason == RegistryReason::CapabilityUnsatisfied, "the wrong reason was reported");
    require(refused.record == nullptr, "a substitute was returned for an unsatisfiable request");
}

void ambiguity_and_policies()
{
    ComponentRegistry registry;
    registry.add(record("heat.a", {1, 2, 0}, "heat", host_arrays, 5));
    registry.add(record("heat.b", {1, 5, 0}, "heat", host_arrays, 1));

    // Two implementations of one contract and no stated preference: refused.
    // Picking one would be a policy the caller never chose.
    const auto ambiguous = registry.select(request("heat"));
    require(ambiguous.reason == RegistryReason::Ambiguous, "an ambiguous selection was resolved silently");
    require(ambiguous.record == nullptr, "an ambiguous selection returned a record");

    auto by_version = request("heat");
    by_version.policy = SelectionPolicy::HighestVersion;
    const auto newest = registry.select(by_version);
    require(newest.reason == RegistryReason::Success && newest.record->implementation_id == "heat.b",
            "the highest version was not chosen");

    auto by_priority = request("heat");
    by_priority.policy = SelectionPolicy::HighestPriority;
    const auto preferred = registry.select(by_priority);
    require(preferred.reason == RegistryReason::Success && preferred.record->implementation_id == "heat.a",
            "the highest priority was not chosen");

    // One match needs no policy.
    auto single = request("heat");
    single.required_capabilities = host_arrays;
    single.minimum_version = {1, 4, 0};
    const auto only = registry.select(single);
    require(only.reason == RegistryReason::Success && only.record->implementation_id == "heat.b",
            "a single filtered match was not returned");
}

void version_and_capability_filtering()
{
    ComponentRegistry registry;
    registry.add(record("only", {1, 4, 2}, "heat", host_arrays | checkpoints));

    auto too_new = request("heat");
    too_new.minimum_version = {2, 0, 0};
    require(registry.select(too_new).reason == RegistryReason::VersionUnsatisfied, "a too-old version was accepted");

    auto too_old = request("heat");
    too_old.maximum_version = {1, 0, 0};
    require(registry.select(too_old).reason == RegistryReason::VersionUnsatisfied, "a too-new version was accepted");

    // An unset maximum is unbounded, not a refusal of everything. The
    // opposite reading would reject every implementation silently.
    auto unbounded = request("heat");
    unbounded.minimum_version = {1, 0, 0};
    unbounded.maximum_version = {0, 0, 0};
    require(registry.select(unbounded).reason == RegistryReason::Success, "an unset maximum was treated as a bound");

    auto within = request("heat");
    within.minimum_version = {1, 4, 0};
    within.maximum_version = {1, 5, 0};
    require(registry.select(within).reason == RegistryReason::Success, "a version inside the interval was refused");

    auto greedy = request("heat");
    greedy.required_capabilities = host_arrays | checkpoints | KARMA_CAPABILITY_CANCELLATION;
    const auto refused = registry.select(greedy);
    require(refused.reason == RegistryReason::CapabilityUnsatisfied, "a missing capability was granted");
    require(refused.missing_capabilities == KARMA_CAPABILITY_CANCELLATION,
            "the missing capability bits were not reported");
}

void deterministic_inventory()
{
    // The inventory must not depend on registration order, or two runs of the
    // same program with different static initialization order disagree about
    // what is installed.
    const std::vector<RegistryRecord> forward{record("c", {1, 0, 0}, "heat", 0),
                                              record("a", {1, 0, 0}, "heat", 0),
                                              record("b", {1, 0, 0}, "mechanics", 0)};
    ComponentRegistry first;
    for (const auto& one : forward) first.add(one);
    ComponentRegistry second;
    for (auto one = forward.rbegin(); one != forward.rend(); ++one) second.add(*one);

    const auto left = first.inventory();
    const auto right = second.inventory();
    require(left.size() == right.size(), "the inventories differ in size");
    for (std::size_t index = 0; index != left.size(); ++index)
        require(left[index]->implementation_id == right[index]->implementation_id,
                "the inventory depends on registration order");
    require(left.front()->implementation_id == "a", "the inventory is not sorted by identifier");
}

void reentrancy()
{
    // SOMA's LoweringRegistry::apply_first_match iterates `rules_` by
    // reference while calling each rule, and its register() both push_backs
    // and stable_sorts. A rule that registers during the callback therefore
    // invalidates the loop mid-iteration. This registry must not have that
    // shape: visiting snapshots the records, the records are stable in
    // memory, and a nested registration is refused rather than allowed to
    // corrupt the iteration.
    ComponentRegistry registry;
    registry.add(record("x", {1, 0, 0}, "heat", 0));
    registry.add(record("y", {1, 0, 0}, "heat", 0));

    std::vector<std::string> seen;
    RegistryReason nested = RegistryReason::Success;
    const auto* first_pointer = registry.inventory().front();
    registry.visit([&](const RegistryRecord& entry) {
        seen.push_back(entry.implementation_id);
        if (seen.size() == 1) nested = registry.add(record("z", {1, 0, 0}, "heat", 0));
    });
    require(nested == RegistryReason::Reentrant, "a nested registration was accepted during a visit");
    require(seen.size() == 2, "the visit did not see every record");
    require(seen[0] == "x" && seen[1] == "y", "the visit order changed under a nested registration");
    require(registry.inventory().front() == first_pointer, "record addresses are not stable");
    require(registry.size() == 2, "a refused nested registration changed the registry");

    // After the visit, registration works again: the guard is scoped, not
    // sticky.
    require(registry.add(record("z", {1, 0, 0}, "heat", 0)) == RegistryReason::Success,
            "the re-entrancy guard outlived the visit");
}

void reason_names()
{
    for (auto code = static_cast<std::uint32_t>(RegistryReason::Success);
         code <= static_cast<std::uint32_t>(RegistryReason::PortIncompatible); ++code)
        require(!registry_reason_name(static_cast<RegistryReason>(code)).empty(), "a reason code has no name");
    require(registry_reason_name(static_cast<RegistryReason>(9999)).empty(), "an unknown reason acquired a name");
}
}

int main()
{
    try
    {
        registration();
        never_substitutes();
        ambiguity_and_policies();
        version_and_capability_filtering();
        deterministic_inventory();
        reentrancy();
        reason_names();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
