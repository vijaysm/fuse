// End-to-end: a native component driven exclusively through the versioned
// runtime contract, plus the reusable compliance harness itself.
//
// Needs neither PETSc nor MPI, which is the point of the reference components:
// the contract can be exercised anywhere the library builds.
#include "karma_reference_components.h"

#include <cstring>
#include <memory>
#include <span>
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

ComplianceFixture fixture(std::size_t count, double rate, std::uint64_t capabilities)
{
    ComplianceFixture value;
    ReferenceDecayConfiguration configuration;
    configuration.element_count = count;
    configuration.rate = rate;
    value.configuration = encode_reference_decay_configuration(configuration);
    value.element_count = count;
    value.granted_capabilities = capabilities;
    value.start = Rational{0, 1};
    value.step_end = Rational{1, 1};
    return value;
}

constexpr std::uint64_t everything = KARMA_CAPABILITY_HOST_ARRAY_FIELDS |
                                     KARMA_CAPABILITY_CHECKPOINT_RESTORE |
                                     KARMA_CAPABILITY_FRACTIONAL_STEPS;

const RegistryRecord& select(const ComponentRegistry& registry, const char* id)
{
    const auto found = registry.select(SelectionRequest{}.with_implementation(id));
    require(found.reason == RegistryReason::Success && found.record != nullptr, "the reference component is absent");
    return *found.record;
}

void static_registration()
{
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    require(registry.size() == 2, "both reference components should be registered");

    // Registering twice reports the duplicate rather than replacing anything.
    require(register_reference_components(registry) == RegistryReason::DuplicateImplementationId,
            "a second registration was accepted");
    require(registry.size() == 2, "a refused re-registration changed the registry");

    // Both satisfy one contract, so an unqualified request is ambiguous - and
    // asking for a third implementation is not answered with either of them.
    SelectionRequest by_contract;
    by_contract.contract_id = "karma.contract.scalar-decay";
    require(registry.select(by_contract).reason == RegistryReason::Ambiguous,
            "two implementations of one contract resolved silently");
    require(registry.select(SelectionRequest{}.with_implementation("karma.reference.decay.absent")).reason ==
                RegistryReason::NotFound,
            "an absent implementation was substituted");
}

void compliance_harness()
{
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");

    for (const char* id : {"karma.reference.decay.baseline", "karma.reference.decay.extended"})
    {
        const auto report = check_component_compliance(select(registry, id), fixture(4, 0.25, everything));
        if (!report.ok())
        {
            std::cerr << id << " failed " << report.failures.size() << " of " << report.checks << " checks:\n";
            for (const auto& failure : report.failures) std::cerr << "  - " << failure << '\n';
            throw std::runtime_error("a reference component is not compliant");
        }
        require(report.checks > 20, "the harness ran suspiciously few checks");
    }
}

void harness_detects_a_lying_record()
{
    // The harness must catch a record that overstates its component, because
    // every selection believes the record rather than asking the component.
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    auto overstated = select(registry, "karma.reference.decay.baseline");
    overstated.provided_capabilities |= KARMA_CAPABILITY_CANCELLATION;
    const auto report = check_component_compliance(overstated, fixture(4, 0.25, everything));
    require(!report.ok(), "an overstated record passed compliance");
    bool named = false;
    for (const auto& failure : report.failures)
        if (failure.find("claims capabilities") != std::string::npos) named = true;
    require(named, "the overstated capability was not the reported failure");

    auto wrong_version = select(registry, "karma.reference.decay.baseline");
    wrong_version.implementation_version = SemanticVersion{7, 7, 7};
    const auto second = check_component_compliance(wrong_version, fixture(4, 0.25, everything));
    require(!second.ok(), "a record disagreeing about the version passed compliance");
}

void end_to_end_through_the_contract()
{
    // The complete lifecycle path: create, bind, initialize, import, advance, commit,
    // export - every call through the C table, none through a C++ shortcut.
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    const auto& record = select(registry, "karma.reference.decay.baseline");

    ReferenceDecayConfiguration configuration;
    configuration.element_count = 3;
    configuration.rate = 0.0; // take the implementation's default
    const auto bytes = encode_reference_decay_configuration(configuration);

    auto created = ComponentInstance::create(record, bytes, everything);
    require(created.status == KARMA_STATUS_OK && created.instance.valid(), "create failed");
    auto& instance = created.instance;

    require(instance.bind_host() == KARMA_STATUS_OK, "bind_host failed");
    require(instance.initialize(Rational{0, 1}) == KARMA_STATUS_OK, "initialize failed");

    std::vector<double> values{1.0, 2.0, 4.0};
    karma_field_table table{};
    table.struct_size = static_cast<std::uint32_t>(sizeof(table));
    table.field_id = "state";
    table.element_type = KARMA_ELEMENT_F64;
    table.access = KARMA_ACCESS_READ_WRITE;
    table.update = KARMA_UPDATE_REPLACE;
    table.rank = 1;
    table.components = 1;
    table.extents[0] = 3;
    table.byte_strides[0] = sizeof(double);
    table.data = values.data();
    table.element_count = 3;
    require(instance.import_fields({&table, 1}) == KARMA_STATUS_OK, "import failed");

    // A whole step with rate 1: every value scales by the same factor, so the
    // ratios are preserved exactly even though the factor itself is inexact.
    require(instance.advance(Rational{0, 1}, Rational{1, 1}) == KARMA_STATUS_OK, "advance failed");
    require(instance.commit() == KARMA_STATUS_OK, "commit failed");

    std::vector<double> exported(3, 0.0);
    table.data = exported.data();
    require(instance.export_fields({&table, 1}) == KARMA_STATUS_OK, "export failed");
    require(exported[0] > 0.0 && exported[0] < 1.0, "the state did not decay");
    require(exported[1] == 2.0 * exported[0] && exported[2] == 4.0 * exported[0],
            "a uniform decay did not preserve the ratios exactly");

    // A fractional step is refused unless it was granted, and the check lives
    // in the runtime rather than in each component.
    auto restricted = ComponentInstance::create(record, bytes, KARMA_CAPABILITY_HOST_ARRAY_FIELDS);
    require(restricted.status == KARMA_STATUS_OK, "restricted create failed");
    require(restricted.instance.initialize(Rational{0, 1}) == KARMA_STATUS_OK, "restricted initialize failed");
    require(restricted.instance.advance(Rational{0, 1}, Rational{1, 2}) == KARMA_STATUS_UNSUPPORTED_CAPABILITY,
            "an ungranted fractional step was taken");
    require(restricted.instance.advance(Rational{0, 1}, Rational{1, 1}) == KARMA_STATUS_OK,
            "a whole step was refused");
    std::uint64_t token = 0;
    require(restricted.instance.checkpoint(token) == KARMA_STATUS_UNSUPPORTED_CAPABILITY,
            "checkpointing was allowed without the capability");
}

void failed_construction_leaves_nothing()
{
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    const auto& record = select(registry, "karma.reference.decay.baseline");

    // A configuration of the wrong size, a zero element count, and a negative
    // rate are each refused, and each leaves an invalid handle rather than a
    // half-built component.
    const std::vector<std::byte> short_bytes(4, std::byte{0});
    auto truncated = ComponentInstance::create(record, short_bytes, everything);
    require(truncated.status != KARMA_STATUS_OK, "a short configuration was accepted");
    require(!truncated.instance.valid(), "a failed create produced a usable handle");

    ReferenceDecayConfiguration empty;
    empty.element_count = 0;
    empty.rate = 1.0;
    auto zero = ComponentInstance::create(record, encode_reference_decay_configuration(empty), everything);
    require(zero.status != KARMA_STATUS_OK, "a zero element count was accepted");
    require(!zero.instance.valid(), "a failed create produced a usable handle");

    ReferenceDecayConfiguration negative;
    negative.element_count = 4;
    negative.rate = -1.0;
    auto bad_rate = ComponentInstance::create(record, encode_reference_decay_configuration(negative), everything);
    require(bad_rate.status != KARMA_STATUS_OK, "a negative rate was accepted");
    require(!bad_rate.instance.valid(), "a failed create produced a usable handle");

    // The handle is move-only, and moving from it leaves the source empty so
    // the instance is destroyed exactly once.
    ReferenceDecayConfiguration good;
    good.element_count = 2;
    good.rate = 1.0;
    auto created = ComponentInstance::create(record, encode_reference_decay_configuration(good), everything);
    require(created.instance.valid(), "create failed");
    ComponentInstance moved = std::move(created.instance);
    require(moved.valid(), "the moved-to handle is empty");
    require(!created.instance.valid(), "the moved-from handle still owns the instance");

    // Move ASSIGNMENT too, and onto a handle that already owns an instance:
    // the overwritten instance must be destroyed once and the source must be
    // emptied, or the same instance is destroyed twice at scope exit.
    auto target = ComponentInstance::create(record, encode_reference_decay_configuration(good), everything);
    require(target.instance.valid(), "the assignment target was not created");
    target.instance = std::move(moved);
    require(target.instance.valid(), "the assigned-to handle is empty");
    require(!moved.valid(), "the moved-from handle still owns the instance after assignment");

    // Self-assignment must not destroy the instance it is about to keep.
    ComponentInstance& alias = target.instance;
    target.instance = std::move(alias);
    require(target.instance.valid(), "self-assignment destroyed the instance");
}

void a_lying_create_is_not_trusted()
{
    // A badly written plugin may return OK and leave the out parameter null.
    // The host must not build a handle around nothing and call it valid.
    static karma_component_table liar = [] {
        auto table = native_component_table();
        table.create = [](const karma_component_config*, void** instance) {
            if (instance != nullptr) *instance = nullptr;
            return KARMA_STATUS_OK;
        };
        return table;
    }();

    RegistryRecord record;
    record.implementation_id = "karma.test.liar";
    record.contract_id = "karma.contract.scalar-decay";
    record.table = &liar;
    ComponentRegistry registry;
    require(registry.add(record) == RegistryReason::Success, "the stub record was refused");

    auto created = ComponentInstance::create(*registry.select(SelectionRequest{}.with_implementation("karma.test.liar")).record,
                                             {}, everything);
    require(!created.instance.valid(), "a create that returned no instance produced a usable handle");
    require(created.status == KARMA_STATUS_INTERNAL_ERROR,
            "a create that returned OK with no instance was reported as success");
}

/** A component that keeps the caller's buffer instead of copying it. It exists
 * to prove the harness's lifetime check is live: a harness whose checks can be
 * deleted without any test noticing is not evidence of anything. */
class RetainingComponent final : public NativeComponent
{
  public:
    explicit RetainingComponent(std::size_t count) : _count(count) {}

    OutcomeReason initialize(Rational) override { return OutcomeReason::Success; }
    OutcomeReason import_field(const karma_field_table& table) override
    {
        if (table.element_count != _count) return OutcomeReason::InvalidArgument;
        _borrowed = static_cast<const double*>(table.data); // the defect, deliberately
        return OutcomeReason::Success;
    }
    OutcomeReason export_field(const karma_field_table& table) const override
    {
        if (_borrowed == nullptr || table.element_count != _count) return OutcomeReason::InvalidArgument;
        auto* out = static_cast<double*>(table.data);
        for (std::size_t index = 0; index != _count; ++index) out[index] = _borrowed[index];
        return OutcomeReason::Success;
    }
    OutcomeReason advance(Rational, Rational) override { return OutcomeReason::Success; }
    OutcomeReason commit() override { return OutcomeReason::Success; }
    OutcomeReason checkpoint(std::uint64_t& token) override
    {
        token = 1;
        return OutcomeReason::Success;
    }
    OutcomeReason restore(std::uint64_t token) override
    {
        return token == 1 ? OutcomeReason::Success : OutcomeReason::InvalidArgument;
    }
    OutcomeReason release_checkpoint(std::uint64_t token) override
    {
        return token == 1 ? OutcomeReason::Success : OutcomeReason::InvalidArgument;
    }
    std::uint64_t provided_capabilities() const override
    {
        return KARMA_CAPABILITY_HOST_ARRAY_FIELDS | KARMA_CAPABILITY_CHECKPOINT_RESTORE |
               KARMA_CAPABILITY_FRACTIONAL_STEPS;
    }
    SemanticVersion implementation_version() const override { return {1, 0, 0}; }

  private:
    std::size_t _count;
    const double* _borrowed = nullptr;
};

void harness_detects_a_retained_buffer()
{
    ComponentRegistry registry;
    RegistryRecord record;
    record.implementation_id = "karma.test.retaining";
    record.implementation_version = SemanticVersion{1, 0, 0};
    record.contract_id = "karma.contract.scalar-decay";
    record.provided_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS | KARMA_CAPABILITY_CHECKPOINT_RESTORE |
                                   KARMA_CAPABILITY_FRACTIONAL_STEPS;
    require(register_native_component(registry, record,
                                      [](std::span<const std::byte> configuration) -> std::unique_ptr<NativeComponent> {
                                          if (configuration.size() != sizeof(ReferenceDecayConfiguration))
                                              return nullptr;
                                          ReferenceDecayConfiguration decoded{};
                                          std::memcpy(&decoded, configuration.data(), sizeof(decoded));
                                          return std::make_unique<RetainingComponent>(
                                              static_cast<std::size_t>(decoded.element_count));
                                      }) == RegistryReason::Success,
            "the retaining component was not registered");

    const auto report = check_component_compliance(select(registry, "karma.test.retaining"),
                                                   fixture(4, 1.0, everything));
    require(!report.ok(), "a component that retains the caller's buffer passed compliance");
    bool named = false;
    for (const auto& failure : report.failures)
        if (failure.find("retained the caller's buffer") != std::string::npos) named = true;
    require(named, "the retained buffer was not the reported failure");
}

FieldSemantics compatible_source_for_state()
{
    // Matches the "state" port declared in decay_contract(): dimensionless
    // scalar, Prognostic role, no version floor - an exact match, so
    // check_connection() should compile it with no operator at all.
    FieldSemantics source;
    source.id = "state";
    source.units = UnitDimensions{};
    source.shape = SemanticShape::Scalar;
    source.location = SemanticLocation::Unspecified;
    source.temporal = SemanticTemporal::InstantaneousCommitted;
    source.role = SemanticRole::Prognostic;
    source.ownership = SemanticOwnership::Owned;
    source.access = SemanticAccess::ReadWrite;
    source.update = SemanticUpdate::Replace;
    return source;
}

void registry_selects_by_port_compatibility()
{
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");

    // A field the "state" port genuinely accepts: selection by port must
    // succeed exactly as selection by implementation id does.
    auto request = SelectionRequest{}.with_implementation("karma.reference.decay.baseline");
    request.with_required_port("state", compatible_source_for_state());
    const auto found = registry.select(request);
    require(found.reason == RegistryReason::Success && found.record != nullptr,
            "a compatible port request was refused");

    // A port id that does not exist on the candidate must be refused with
    // PortIncompatible, not silently ignored back to a plain identity match.
    auto missing_port = SelectionRequest{}.with_implementation("karma.reference.decay.baseline");
    missing_port.with_required_port("no-such-port", compatible_source_for_state());
    const auto missing = registry.select(missing_port);
    require(missing.reason == RegistryReason::PortIncompatible, "a nonexistent port id was accepted");
    require(!missing.port_incompatibility.has_value(),
            "a nonexistent port id produced an incompatibility verdict from a check that never ran");

    // A field whose units cannot connect to the port's dimensionless field
    // without an operator that was never requested: PortIncompatible, and
    // the checker's own verdict travels with the refusal.
    auto mismatched = compatible_source_for_state();
    mismatched.units.length = 1; // now has a length dimension; the port is dimensionless
    auto incompatible_request = SelectionRequest{}.with_implementation("karma.reference.decay.baseline");
    incompatible_request.with_required_port("state", mismatched);
    const auto incompatible = registry.select(incompatible_request);
    require(incompatible.reason == RegistryReason::PortIncompatible,
            "a dimensionally incompatible field was accepted for the state port");
    require(incompatible.port_incompatibility.has_value(),
            "an actual connection-checker refusal was not reported");
    require(!incompatible.port_incompatibility->compiles(), "the reported incompatibility actually compiles");

    // The port filter narrows an existing candidate SET; it must not widen
    // one. Both reference components declare the same "state" port, so an
    // unqualified contract-id request stays Ambiguous whether or not a port
    // filter is layered on top of it.
    SelectionRequest by_contract;
    by_contract.contract_id = "karma.contract.scalar-decay";
    by_contract.with_required_port("state", compatible_source_for_state());
    require(registry.select(by_contract).reason == RegistryReason::Ambiguous,
            "a port filter resolved an ambiguous contract-id request instead of narrowing it");

    // An explicit requirement override must actually be used in place of the
    // port's own declared requirement, not just accepted and ignored. The
    // port's own requirement has no version floor; overriding it with one the
    // component's declared version cannot satisfy must turn a request that
    // would otherwise succeed into a refusal.
    FieldRequirement stricter;
    stricter.field_id = "state";
    stricter.required = true;
    stricter.minimum_version = SemanticVersion{99, 0, 0};
    auto overridden = SelectionRequest{}.with_implementation("karma.reference.decay.baseline");
    overridden.with_required_port("state", compatible_source_for_state(), stricter);
    require(registry.select(overridden).reason == RegistryReason::PortIncompatible,
            "an explicit requirement override was accepted without being applied");

    // And the reverse: the plain (no-override) request for the same port and
    // field must still succeed, so the override is doing the work rather than
    // the port itself having grown a floor.
    require(found.reason == RegistryReason::Success, "the un-overridden request regressed");
}

void compliance_harness_validates_the_records_own_semantics()
{
    // check_component_compliance() must catch a record whose OWN contract
    // does not validate under validate_component_semantics() - a distinct
    // failure from the "record overstates the component" case already
    // covered, since this one is wrong independent of the component at all.
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    auto invalid = select(registry, "karma.reference.decay.baseline");
    invalid.contract.ports.push_back(invalid.contract.ports.front()); // duplicate port id
    const auto report = check_component_compliance(invalid, fixture(4, 0.25, everything));
    require(!report.ok(), "a record with a duplicate port id passed compliance");
    bool named = false;
    for (const auto& failure : report.failures)
        if (failure.find("does not validate") != std::string::npos) named = true;
    require(named, "the invalid contract was not the reported failure");
}

void unregistered_identifier_is_refused()
{
    // Reaching the native table with an identifier no factory claims must be
    // an error, not a fallback to whichever factory happens to be first.
    ComponentRegistry registry;
    require(register_reference_components(registry) == RegistryReason::Success, "static registration failed");
    auto forged = select(registry, "karma.reference.decay.baseline");
    forged.implementation_id = "karma.reference.decay.never-registered";
    ReferenceDecayConfiguration configuration;
    configuration.element_count = 2;
    configuration.rate = 1.0;
    auto created = ComponentInstance::create(forged, encode_reference_decay_configuration(configuration), everything);
    require(created.status == KARMA_STATUS_UNKNOWN_FIELD, "an unregistered identifier was constructed");
    require(!created.instance.valid(), "an unregistered identifier produced a handle");
}
}

// Every ABI status must survive a trip through the collective vocabulary and
// back. This is the property that was broken: four statuses collapsed onto
// InvalidArgument and eight more onto Internal, so a reconciled verdict could
// name a failure that no rank had.
void every_status_round_trips()
{
    const karma_status all[] = {
        KARMA_STATUS_OK, KARMA_STATUS_INVALID_ARGUMENT, KARMA_STATUS_TRUNCATED_PREFIX,
        KARMA_STATUS_MALFORMED_SIZE, KARMA_STATUS_INCOMPATIBLE_MAJOR,
        KARMA_STATUS_UNSUPPORTED_CAPABILITY, KARMA_STATUS_TABLE_TOO_SMALL,
        KARMA_STATUS_UNKNOWN_FIELD, KARMA_STATUS_MALFORMED_FIELD_TABLE,
        KARMA_STATUS_NONFINITE_VALUE, KARMA_STATUS_INCOMPLETE_UPDATE,
        KARMA_STATUS_STORAGE_FAILURE, KARMA_STATUS_UNKNOWN_CHECKPOINT,
        KARMA_STATUS_NOT_INITIALIZED, KARMA_STATUS_CANCELLED, KARMA_STATUS_INTERNAL_ERROR,
    };
    for (const auto status : all)
        require(status_of_outcome(outcome_of_status(status)) == status,
                "an ABI status did not survive the collective vocabulary");

    // Distinctness is the point: no two statuses may share a reason, or the
    // round trip above would hold while reconciliation still flattened them.
    for (const auto left : all)
        for (const auto right : all)
            if (left != right)
                require(outcome_of_status(left) != outcome_of_status(right),
                        "two distinct ABI statuses map to the same reason");

    // An unrecognized status must not be mistaken for success.
    require(outcome_of_status(static_cast<karma_status>(9999)) != OutcomeReason::Success,
            "an unrecognized status reconciled as success");
}

// R5: the ABI built-in and the registered decay component have the same
// lifecycle and the same "state" field, and nothing kept them in step.
//
// They are deliberately NOT merged. They compute different things on purpose:
// the registered one is an analytic exponential decay, while the built-in is
// an accumulator whose reason for existing is to exercise the ABI's own
// mechanics - the not-initialized guard, capability gating, the cancellation
// path and the throwing fixture. Replacing it with the adapter-wrapped
// component would leave nothing implementing the raw C table by hand, so a
// bug in the adapter would be invisible to the ABI test and the harness
// alike, both of them travelling through the adapter.
//
// What R5 actually warns about is SILENT divergence of the shared shape. So
// both are held to one specification instead: the same compliance harness,
// which is the thing that was built to be reusable. Any lifecycle difference
// between the two now fails here rather than going unnoticed.
void the_abi_builtin_meets_the_same_specification()
{
    karma_plugin_query_result query{};
    query.struct_size = static_cast<std::uint32_t>(sizeof(query));
    require(karma_plugin_query(&query) == KARMA_STATUS_OK, "the plugin query failed");
    require(query.component_table != nullptr, "the plugin query returned no table");

    RegistryRecord record;
    record.implementation_id = query.implementation_id;
    const auto abi = query.implementation_version;
    record.implementation_version = SemanticVersion{abi.major, abi.minor, abi.patch};
    record.contract_id = "karma.reference.state";
    record.contract = reference_state_contract(record.implementation_id, record.implementation_version);
    record.table = query.component_table;
    record.provided_capabilities = query.provided_capabilities;
    record.provenance = "abi plugin query";

    // The built-in's configuration is a bare little-endian element count.
    ComplianceFixture fixture;
    fixture.field_id = "state";
    fixture.element_count = 4;
    fixture.granted_capabilities = query.provided_capabilities;
    fixture.configuration.resize(sizeof(std::uint64_t));
    const std::uint64_t count = fixture.element_count;
    for (std::size_t index = 0; index != sizeof(count); ++index)
        fixture.configuration[index] = static_cast<std::byte>((count >> (8 * index)) & 0xffU);

    const auto report = check_component_compliance(record, fixture);
    if (!report.ok())
    {
        std::string message = "the ABI built-in fails the shared component specification:";
        for (const auto& failure : report.failures) message += "\n  - " + failure;
        throw std::runtime_error(message);
    }
    require(report.checks > 0, "the harness ran no checks against the ABI built-in");
}

int main()
{
    try
    {
        every_status_round_trips();
        the_abi_builtin_meets_the_same_specification();
        static_registration();
        compliance_harness();
        harness_detects_a_lying_record();
        end_to_end_through_the_contract();
        failed_construction_leaves_nothing();
        unregistered_identifier_is_refused();
        a_lying_create_is_not_trusted();
        harness_detects_a_retained_buffer();
        registry_selects_by_port_compatibility();
        compliance_harness_validates_the_records_own_semantics();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
