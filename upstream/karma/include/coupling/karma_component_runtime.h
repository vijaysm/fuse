#ifndef KARMA_COUPLING_COMPONENT_RUNTIME_H
#define KARMA_COUPLING_COMPONENT_RUNTIME_H

#include "karma_collective_outcome.h"
#include "karma_registry.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

/** Host side of the component boundary.
 *
 * `ComponentInstance` is an RAII handle over a `karma_component_table`: it
 * calls the same C schema whether the implementation behind it is C++, a
 * dynamically loaded object, or a language adapter. There is deliberately no
 * second, faster path for native components - a native component that took a
 * shortcut would not be exercising the contract the others must satisfy, and
 * the contract is the thing under test.
 *
 * `NativeComponent` is the C++ interface an in-tree component implements;
 * `register_native_component()` wraps it in that same C table. The wrapping is
 * the only place native code is special.
 */
namespace KARMA
{
/** A C++ component. Implementations return reason codes and do not throw
 * across the boundary; the adapter contains anything that escapes anyway. */
class NativeComponent
{
  public:
    virtual ~NativeComponent() = default;

    virtual OutcomeReason initialize(Rational start) = 0;
    virtual OutcomeReason import_field(const karma_field_table& table) = 0;
    virtual OutcomeReason export_field(const karma_field_table& table) const = 0;
    virtual OutcomeReason advance(Rational begin, Rational end) = 0;
    virtual OutcomeReason commit() = 0;
    virtual OutcomeReason checkpoint(std::uint64_t& token) = 0;
    virtual OutcomeReason restore(std::uint64_t token) = 0;
    virtual OutcomeReason release_checkpoint(std::uint64_t token) = 0;

    /** What the component itself claims, so a registry record that overstates
     * it can be caught rather than believed. */
    virtual std::uint64_t provided_capabilities() const = 0;
    virtual SemanticVersion implementation_version() const = 0;
};

using NativeComponentFactory =
    std::function<std::unique_ptr<NativeComponent>(std::span<const std::byte> configuration)>;

/** Registers a C++ component and points its record at the shared native
 * adapter table. The factory is keyed by implementation identifier, which is
 * what the ABI `create` call carries. */
[[nodiscard]] RegistryReason register_native_component(ComponentRegistry& registry, RegistryRecord record,
                                                       NativeComponentFactory factory);

/** The one table every native component is driven through. */
[[nodiscard]] const karma_component_table& native_component_table();

struct ComponentInstanceCreateResult;

/** An RAII component handle. Move-only: two handles to one instance would
 * destroy it twice. */
class ComponentInstance
{
  public:
    ComponentInstance() = default;
    ComponentInstance(const ComponentInstance&) = delete;
    ComponentInstance& operator=(const ComponentInstance&) = delete;
    ComponentInstance(ComponentInstance&& other) noexcept;
    ComponentInstance& operator=(ComponentInstance&& other) noexcept;
    ~ComponentInstance();

    /** Declared before the class and defined after it: it holds a
     * ComponentInstance by value, which is incomplete inside its own class. */
    using CreateResult = ComponentInstanceCreateResult;

    /** On failure the returned handle is empty and nothing was allocated: a
     * half-constructed component must not be reachable. */
    [[nodiscard]] static CreateResult create(const RegistryRecord& record,
                                             std::span<const std::byte> configuration,
                                             std::uint64_t granted_capabilities,
                                             karma_communicator communicator = KARMA_COMMUNICATOR_NONE);

    [[nodiscard]] bool valid() const noexcept { return _instance != nullptr; }

    [[nodiscard]] karma_status bind_host();
    [[nodiscard]] karma_status initialize(Rational start);
    [[nodiscard]] karma_status metadata(karma_component_metadata& out) const;
    [[nodiscard]] karma_status import_fields(std::span<const karma_field_table> tables);
    [[nodiscard]] karma_status export_fields(std::span<const karma_field_table> tables) const;
    [[nodiscard]] karma_status advance(Rational begin, Rational end);
    [[nodiscard]] karma_status commit();
    [[nodiscard]] karma_status checkpoint(std::uint64_t& token);
    [[nodiscard]] karma_status restore(std::uint64_t token);
    [[nodiscard]] karma_status release_checkpoint(std::uint64_t token);
    [[nodiscard]] karma_status diagnostics(karma_diagnostic& out) const;
    [[nodiscard]] karma_status poll_cancellation(bool& cancelled) const;

    /** Advances and reconciles the local status across `communicator`, so one
     * rank cannot fail while its peers enter the next collective. */
    [[nodiscard]] ReconciledOutcome advance_collective(MPI_Comm communicator, Rational begin, Rational end);

  private:
    const karma_component_table* _table = nullptr;
    void* _instance = nullptr;
};

struct ComponentInstanceCreateResult
{
    ComponentInstance instance;
    karma_status status = KARMA_STATUS_OK;
};

/** Maps an ABI status onto the shared outcome vocabulary, so a component
 * failure and a transaction failure reconcile through one primitive. */
[[nodiscard]] OutcomeReason outcome_of_status(karma_status status) noexcept;

/** The inverse of `outcome_of_status`. Declared beside it, and tested with
 * it, because the pair is only useful if it round-trips: a status that
 * crosses into a collective reconciliation and back must still name the
 * failure that happened. */
[[nodiscard]] karma_status status_of_outcome(OutcomeReason reason) noexcept;

/** Inputs the compliance harness needs in order to exercise a component
 * without knowing what it computes. */
struct ComplianceFixture
{
    std::vector<std::byte> configuration;
    std::string field_id = "state";
    std::size_t element_count = 4;
    Rational start{0, 1};
    Rational step_end{1, 1};
    std::uint64_t granted_capabilities = 0;
};

struct ComplianceReport
{
    std::size_t checks = 0;
    std::vector<std::string> failures;
    [[nodiscard]] bool ok() const noexcept { return failures.empty(); }
};

/** Checks the properties every component must have, whatever it computes:
 * lifecycle ordering, that a record does not overstate its component,
 * field round-tripping, determinism, checkpoint/restore/release, buffer
 * lifetime, and diagnostics after a deliberate error.
 *
 * Reusable on purpose: the Fortran and Python adapters must pass this same
 * function, not a re-derived approximation of it.
 */
[[nodiscard]] ComplianceReport check_component_compliance(const RegistryRecord& record,
                                                          const ComplianceFixture& fixture);
} // namespace KARMA

#endif
