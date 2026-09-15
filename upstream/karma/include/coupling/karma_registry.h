#ifndef KARMA_COUPLING_REGISTRY_H
#define KARMA_COUPLING_REGISTRY_H

#include "karma_compatibility.h"
#include "karma_component_abi.h"
#include "karma_semantic_version.h"
#include "karma_semantics.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/** The static component registry.
 *
 * Two rules give this file its shape.
 *
 * **It never substitutes.** A request naming an implementation that is not
 * registered is an error, even when another implementation of the same
 * contract is available; a request matching two implementations with no stated
 * preference is an error too. Answering either with a plausible alternative
 * would change the physics while reporting success, and the caller would have
 * no way to notice.
 *
 * **It is re-entrancy safe.** Records live behind stable pointers and `visit()`
 * iterates a snapshot, while a registration attempted from inside a visit is
 * refused. The shape to avoid is a container iterated by reference while a
 * callback may append to and re-sort it - `soma::ir::LoweringRegistry::
 * apply_first_match` iterates `std::vector<Entry> rules_` by reference while
 * calling each rule, and its `register` both `push_back`s and `stable_sort`s,
 * so a rule that registers during the callback invalidates the iteration.
 */
namespace KARMA
{
enum class RegistryReason : std::uint32_t
{
    Success = 0,
    /** Another record already claims this implementation identifier. The
     * existing record is kept: last-one-wins would silently replace an
     * implementation. */
    DuplicateImplementationId = 1,
    EmptyImplementationId = 2,
    MissingTable = 3,
    /** The component table is smaller than this build's ABI prefix. */
    TableTooSmall = 4,
    /** Registration was attempted from inside a visit. */
    Reentrant = 5,
    /** No record matches, or the named implementation is absent. */
    NotFound = 6,
    /** Several records match and the request stated no preference. */
    Ambiguous = 7,
    VersionUnsatisfied = 8,
    CapabilityUnsatisfied = 9,
    EmptyContractId = 10,
    /** A candidate exposes no port with the requested id, or exposes one
     * that `check_connection()` refuses to connect the caller's semantics
     * to. Distinct from `CapabilityUnsatisfied`: that is a bitmask the
     * record declares about itself, this is an actual connection-compiler
     * verdict on a specific port. */
    PortIncompatible = 11
};

[[nodiscard]] std::string_view registry_reason_name(RegistryReason reason) noexcept;

/** What is known about one installed implementation.
 *
 * `priority` exists only for a request that explicitly asks for it; nothing
 * consults it otherwise, because an implicit priority is a hidden policy.
 */
struct RegistryRecord
{
    std::string implementation_id;
    SemanticVersion implementation_version{};
    /** The semantic contract this implementation claims to satisfy. */
    std::string contract_id;
    /** The full semantic description, for connection checking. */
    ComponentSemantics contract;
    /** The C ABI factory table. Borrowed: the registry never owns or calls it. */
    const karma_component_table* table = nullptr;
    std::string configuration_schema_id;
    std::uint64_t provided_capabilities = 0;
    /** Where this record came from - static registration, a loaded object -
     * so an inventory can be explained rather than only listed. */
    std::string provenance;
    int priority = 0;
};

enum class SelectionPolicy : std::uint32_t
{
    /** Several matches is an error. The default, deliberately. */
    RefuseAmbiguous = 0,
    HighestVersion = 1,
    HighestPriority = 2
};

struct SelectionRequest
{
    /** Exact implementation identifier, or empty for any. */
    std::string implementation_id;
    /** Contract identifier, or empty for any. */
    std::string contract_id;
    SemanticVersion minimum_version{};
    /** `{0,0,0}` means unbounded. An unset upper bound must not read as a
     * refusal of everything. */
    SemanticVersion maximum_version{};
    std::uint64_t required_capabilities = 0;
    SelectionPolicy policy = SelectionPolicy::RefuseAmbiguous;

    /** When set, a candidate must expose a port with this id whose field
     * semantics `check_connection()` accepts a connection from
     * `required_port_source` into - i.e. "does this component have a port I
     * can actually wire my field into", not just "does its record claim the
     * right contract_id string". Absent by default, so a caller who never
     * sets this gets exactly the string-matching behavior the registry
     * always had. */
    std::optional<std::string> required_port_id;
    std::optional<FieldSemantics> required_port_source;
    std::optional<FieldRequirement> required_port_requirement;
    bool allow_automatic_unit_scale = true;

    SelectionRequest& with_implementation(std::string id)
    {
        implementation_id = std::move(id);
        return *this;
    }

    SelectionRequest& with_required_port(std::string port_id, FieldSemantics source,
                                         FieldRequirement requirement = {})
    {
        required_port_id = std::move(port_id);
        required_port_source = std::move(source);
        required_port_requirement = std::move(requirement);
        return *this;
    }
};

struct SelectionResult
{
    const RegistryRecord* record = nullptr;
    RegistryReason reason = RegistryReason::NotFound;
    /** Set when the reason is `CapabilityUnsatisfied`: which bits were
     * missing, so a caller can report them rather than guess. */
    std::uint64_t missing_capabilities = 0;
    /** Set when the reason is `Ambiguous`: how many records matched. */
    std::size_t match_count = 0;
    /** Set when the reason is `PortIncompatible` and at least one candidate
     * had a port with the right id: the connection checker's own verdict on
     * the best (first-checked) such candidate, so the caller can report why
     * rather than only that it failed. */
    std::optional<ConnectionCompatibility> port_incompatibility;
};

class ComponentRegistry
{
  public:
    ComponentRegistry() = default;
    ComponentRegistry(const ComponentRegistry&) = delete;
    ComponentRegistry& operator=(const ComponentRegistry&) = delete;

    [[nodiscard]] RegistryReason add(RegistryRecord record);

    [[nodiscard]] std::size_t size() const noexcept { return _records.size(); }

    /** Sorted by implementation identifier, so the inventory does not depend
     * on registration order - which static initialization order can change
     * between builds of the same program. Pointers stay valid for the
     * registry's lifetime. */
    [[nodiscard]] std::vector<const RegistryRecord*> inventory() const;

    /** Iterates a snapshot. A registration attempted from the callback is
     * refused, so neither the snapshot nor the callback's own reference can be
     * invalidated. */
    void visit(const std::function<void(const RegistryRecord&)>& callback) const;

    [[nodiscard]] SelectionResult select(const SelectionRequest& request) const;

  private:
    /** Indirect so a record's address survives every later registration. */
    std::vector<std::unique_ptr<RegistryRecord>> _records;
    mutable bool _visiting = false;
};

/** The process-wide registry that static registration populates. */
[[nodiscard]] ComponentRegistry& builtin_component_registry();
} // namespace KARMA

#endif
