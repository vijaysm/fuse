#include "karma_reference_components.h"

#include "karma_field_transaction.h"

#include <cmath>
#include <cstring>
#include <map>

namespace KARMA
{
namespace
{
constexpr const char* contract_id = "karma.contract.scalar-decay";

class ReferenceDecayComponent final : public NativeComponent
{
  public:
    ReferenceDecayComponent(std::size_t count, double rate, SemanticVersion version, std::uint64_t capabilities)
        : _values(count, Real{0}), _rate(rate), _version(version), _capabilities(capabilities)
    {
    }

    OutcomeReason initialize(Rational start) override
    {
        _position = start;
        return OutcomeReason::Success;
    }

    OutcomeReason import_field(const karma_field_table& table) override
    {
        if (std::strcmp(table.field_id, "state") != 0) return OutcomeReason::InvalidArgument;
        if (table.element_type != KARMA_ELEMENT_F64) return OutcomeReason::InvalidArgument;
        if (table.element_count != _values.size()) return OutcomeReason::InvalidArgument;

        // Published through the transaction, so a rejected import - a
        // nonfinite value, say - leaves the state exactly as it was rather
        // than partly overwritten.
        HostArrayStorage storage(_values);
        const auto mode = table.update == KARMA_UPDATE_ACCUMULATE
                              ? UpdateMode::Accumulate
                              : (table.update == KARMA_UPDATE_SPARSE_REPLACE ? UpdateMode::SparseReplace
                                                                               : UpdateMode::Replace);
        FieldTransaction transaction(storage, {1, _values.size(), 0}, mode);
        const auto* base = static_cast<const unsigned char*>(table.data);
        for (std::size_t entry = 0; entry != _values.size(); ++entry)
        {
            double value = 0;
            std::memcpy(&value, base + static_cast<std::size_t>(table.byte_strides[0]) * entry, sizeof(value));
            const auto staged = transaction.stage(entry, 0, value);
            if (staged != OutcomeReason::Success) return staged;
        }
        const auto committed = transaction.commit();
        return committed.ok() ? OutcomeReason::Success : committed.reason;
    }

    OutcomeReason export_field(const karma_field_table& table) const override
    {
        if (std::strcmp(table.field_id, "state") != 0) return OutcomeReason::InvalidArgument;
        if (table.element_count != _values.size()) return OutcomeReason::InvalidArgument;
        auto* base = static_cast<unsigned char*>(table.data);
        for (std::size_t entry = 0; entry != _values.size(); ++entry)
        {
            const double value = static_cast<double>(_values[entry]);
            std::memcpy(base + static_cast<std::size_t>(table.byte_strides[0]) * entry, &value, sizeof(value));
        }
        return OutcomeReason::Success;
    }

    OutcomeReason advance(Rational begin, Rational end) override
    {
        const auto length = rational_subtract(end, begin);
        if (!length.value) return OutcomeReason::InvalidArgument;
        // Analytic, so a step is one multiplication and two runs of the same
        // step are bitwise identical rather than merely close.
        const double factor = std::exp(-_rate * rational_to_double(*length.value));
        if (!std::isfinite(factor)) return OutcomeReason::NonfiniteValue;
        for (auto& value : _values) value *= factor;
        _position = end;
        return OutcomeReason::Success;
    }

    OutcomeReason commit() override { return OutcomeReason::Success; }

    OutcomeReason checkpoint(std::uint64_t& token) override
    {
        const auto assigned = _next_token++;
        _checkpoints[assigned] = _values;
        token = assigned;
        return OutcomeReason::Success;
    }

    OutcomeReason restore(std::uint64_t token) override
    {
        const auto found = _checkpoints.find(token);
        if (found == _checkpoints.end()) return OutcomeReason::InvalidArgument;
        _values = found->second;
        return OutcomeReason::Success;
    }

    OutcomeReason release_checkpoint(std::uint64_t token) override
    {
        return _checkpoints.erase(token) == 0 ? OutcomeReason::InvalidArgument : OutcomeReason::Success;
    }

    std::uint64_t provided_capabilities() const override { return _capabilities; }
    SemanticVersion implementation_version() const override { return _version; }

  private:
    std::vector<Real> _values;
    std::map<std::uint64_t, std::vector<Real>> _checkpoints;
    std::uint64_t _next_token = 1;
    double _rate;
    Rational _position{0, 1};
    SemanticVersion _version;
    std::uint64_t _capabilities;
};

constexpr std::uint64_t reference_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS |
                                                KARMA_CAPABILITY_CHECKPOINT_RESTORE |
                                                KARMA_CAPABILITY_FRACTIONAL_STEPS;

NativeComponentFactory decay_factory(SemanticVersion version, double default_rate)
{
    return [version, default_rate](std::span<const std::byte> configuration) -> std::unique_ptr<NativeComponent> {
        // A configuration of the wrong size is refused rather than padded:
        // reading a short record would take the rate from uninitialized bytes.
        if (configuration.size() != sizeof(ReferenceDecayConfiguration)) return nullptr;
        ReferenceDecayConfiguration decoded{};
        std::memcpy(&decoded, configuration.data(), sizeof(decoded));
        if (decoded.element_count == 0 || decoded.element_count > (UINT64_C(1) << 24)) return nullptr;
        if (!std::isfinite(decoded.rate) || decoded.rate < 0.0) return nullptr;
        const double rate = decoded.rate == 0.0 ? default_rate : decoded.rate;
        return std::make_unique<ReferenceDecayComponent>(static_cast<std::size_t>(decoded.element_count), rate,
                                                         version, reference_capabilities);
    };
}


RegistryRecord decay_record(const char* id, SemanticVersion version, int priority)
{
    RegistryRecord record;
    record.implementation_id = id;
    record.implementation_version = version;
    record.contract_id = contract_id;
    record.contract = reference_state_contract(id, version);
    record.configuration_schema_id = "karma.reference.decay.config.v1";
    record.provided_capabilities = reference_capabilities;
    record.provenance = "static registration in karma_reference_components.C";
    record.priority = priority;
    return record;
}
} // namespace

ComponentSemantics reference_state_contract(const std::string& id, SemanticVersion version)
{
    // A real port declaration, not a placeholder: this is what the
    // compliance harness now validates with validate_component_semantics(),
    // and what ComponentRegistry::select() checks a caller's field against
    // when it asks for this port by id.
    FieldSemantics state;
    state.id = "state";
    state.version = version;
    state.units = UnitDimensions{}; // dimensionless: a bare decaying scalar
    state.shape = SemanticShape::Scalar;
    state.components = 1;
    state.location = SemanticLocation::Unspecified;
    state.temporal = SemanticTemporal::InstantaneousCommitted;
    state.role = SemanticRole::Prognostic;
    state.ownership = SemanticOwnership::Owned;
    state.access = SemanticAccess::ReadWrite;
    state.update = SemanticUpdate::Replace;

    PortSemantics port;
    port.id = "state";
    port.direction = PortSemanticDirection::Input;
    port.field = state;
    port.requirement.field_id = "state";
    port.requirement.required = true;

    ComponentSemantics component;
    component.id = id;
    component.version = version;
    component.ports.push_back(port);
    component.capabilities = reference_capabilities;
    component.supports_fractional_steps = true;
    component.supports_checkpoint_restore = true;
    component.requires_communicator = false;
    return component;
}

std::vector<std::byte> encode_reference_decay_configuration(const ReferenceDecayConfiguration& value)
{
    std::vector<std::byte> bytes(sizeof(ReferenceDecayConfiguration));
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

RegistryReason register_reference_components(ComponentRegistry& registry)
{
    const auto first = register_native_component(registry, decay_record("karma.reference.decay.baseline", {1, 0, 0}, 10),
                                                 decay_factory({1, 0, 0}, 1.0));
    if (first != RegistryReason::Success) return first;
    return register_native_component(registry, decay_record("karma.reference.decay.extended", {2, 1, 0}, 1),
                                     decay_factory({2, 1, 0}, 0.5));
}
} // namespace KARMA
