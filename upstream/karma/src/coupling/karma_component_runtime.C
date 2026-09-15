#include "karma_component_runtime.h"

#include "karma_abi_records.h"
#include "karma_field_entry_encoding.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace KARMA
{
namespace
{
/** The native adapter's per-instance state. */
struct NativeHolder
{
    std::unique_ptr<NativeComponent> component;
    bool initialized = false;
    std::uint64_t granted_capabilities = 0;
    karma_status last_status = KARMA_STATUS_OK;
    std::string last_message;
};

/** Factories keyed by implementation identifier, which is what the ABI
 * `create` call carries. Guarded because static registration can happen from
 * several translation units' initializers. */
std::map<std::string, NativeComponentFactory>& factories()
{
    static std::map<std::string, NativeComponentFactory> table;
    return table;
}

std::mutex& factory_mutex()
{
    static std::mutex guard;
    return guard;
}

NativeHolder* holder(void* instance) noexcept { return static_cast<NativeHolder*>(instance); }


karma_status record_native_error(void* instance, const char* what) noexcept
{
    if (instance != nullptr)
    {
        auto* state = holder(instance);
        state->last_status = KARMA_STATUS_INTERNAL_ERROR;
        try
        {
            state->last_message = what != nullptr ? what : "unknown exception";
        }
        catch (...)
        {
        }
    }
    return KARMA_STATUS_INTERNAL_ERROR;
}

karma_status note(void* instance, karma_status status, const char* message) noexcept
{
    if (instance != nullptr)
    {
        auto* state = holder(instance);
        state->last_status = status;
        try
        {
            state->last_message = message != nullptr ? message : "";
        }
        catch (...)
        {
        }
    }
    return status;
}

/* The same containment rule as the ABI's own implementation: nothing may
 * unwind through a C, Fortran, or Python frame, so every entry point is
 * wrapped. A native component's virtual call is exactly where a C++ exception
 * would otherwise escape. */
#define KARMA_NATIVE_GUARD_BEGIN                                                                                       \
    try                                                                                                                \
    {
#define KARMA_NATIVE_GUARD_END(instance)                                                                               \
    }                                                                                                                  \
    catch (const std::exception& error)                                                                                 \
    {                                                                                                                  \
        return record_native_error(instance, error.what());                                                            \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        return record_native_error(instance, "unknown exception");                                                     \
    }

karma_status native_create(const karma_component_config* config, void** instance)
{
    if (config == nullptr || instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    *instance = nullptr;
    if (config->struct_size < sizeof(karma_component_config)) return KARMA_STATUS_TRUNCATED_PREFIX;
    if (config->implementation_id == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;

    NativeComponentFactory factory;
    {
        const std::lock_guard<std::mutex> lock(factory_mutex());
        const auto found = factories().find(config->implementation_id);
        // An unregistered identifier is refused, never answered with some
        // other factory.
        if (found == factories().end()) return KARMA_STATUS_UNKNOWN_FIELD;
        factory = found->second;
    }

    KARMA_NATIVE_GUARD_BEGIN
    const auto* bytes = static_cast<const std::byte*>(config->configuration);
    auto component = factory(std::span<const std::byte>(bytes, static_cast<std::size_t>(config->configuration_size)));
    // A factory that declines leaves nothing behind: no holder is allocated,
    // so there is no half-built instance for a caller to find.
    if (!component) return KARMA_STATUS_INVALID_ARGUMENT;

    auto state = std::make_unique<NativeHolder>();
    state->component = std::move(component);
    state->granted_capabilities = config->granted_capabilities;
    *instance = state.release();
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(nullptr)
}

karma_status native_bind_host(void* instance, const karma_host_table* host)
{
    if (instance == nullptr || host == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    if (host->struct_size < sizeof(karma_host_table)) return KARMA_STATUS_TRUNCATED_PREFIX;
    return KARMA_STATUS_OK;
}

karma_status native_initialize(void* instance, karma_exact_time start)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    if (start.denominator == 0) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    KARMA_NATIVE_GUARD_BEGIN
    const auto position = make_rational(start.numerator, start.denominator);
    if (!position.value) return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "start time is not representable");
    const auto reason = state->component->initialize(*position.value);
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "initialize refused");
    state->initialized = true;
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_metadata(void* instance, karma_component_metadata* metadata)
{
    if (instance == nullptr || metadata == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    if (metadata->struct_size < sizeof(karma_component_metadata)) return KARMA_STATUS_TRUNCATED_PREFIX;
    auto* state = holder(instance);
    KARMA_NATIVE_GUARD_BEGIN
    const auto version = state->component->implementation_version();
    metadata->implementation_version.major = version.major;
    metadata->implementation_version.minor = version.minor;
    metadata->implementation_version.patch = version.patch;
    const auto provided = state->component->provided_capabilities();
    metadata->provided_capabilities = provided;
    metadata->field_count = 1;
    metadata->supports_fractional_steps = (provided & KARMA_CAPABILITY_FRACTIONAL_STEPS) != 0 ? 1 : 0;
    metadata->supports_checkpoint_restore = (provided & KARMA_CAPABILITY_CHECKPOINT_RESTORE) != 0 ? 1 : 0;
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_import_fields(void* instance, const karma_field_table* tables, std::uint32_t count)
{
    if (instance == nullptr || (count != 0 && tables == nullptr)) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    if (!state->initialized) return note(instance, KARMA_STATUS_NOT_INITIALIZED, "import before initialize");
    KARMA_NATIVE_GUARD_BEGIN
    for (std::uint32_t index = 0; index != count; ++index)
    {
        const auto valid = validate_abi_field_table(&tables[index]);
        if (valid != KARMA_STATUS_OK) return note(instance, valid, "invalid import table");
        const auto reason = state->component->import_field(tables[index]);
        if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "import refused");
    }
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_export_fields(void* instance, const karma_field_table* tables, std::uint32_t count)
{
    if (instance == nullptr || (count != 0 && tables == nullptr)) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    if (!state->initialized) return note(instance, KARMA_STATUS_NOT_INITIALIZED, "export before initialize");
    KARMA_NATIVE_GUARD_BEGIN
    for (std::uint32_t index = 0; index != count; ++index)
    {
        const auto valid = validate_abi_field_table(&tables[index]);
        if (valid != KARMA_STATUS_OK) return note(instance, valid, "invalid export table");
        const auto reason = state->component->export_field(tables[index]);
        if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "export refused");
    }
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_advance(void* instance, karma_exact_interval interval)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    if (!state->initialized) return note(instance, KARMA_STATUS_NOT_INITIALIZED, "advance before initialize");
    if (interval.begin.denominator == 0 || interval.end.denominator == 0)
        return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "interval denominator is zero");
    KARMA_NATIVE_GUARD_BEGIN
    const auto begin = make_rational(interval.begin.numerator, interval.begin.denominator);
    const auto end = make_rational(interval.end.numerator, interval.end.denominator);
    if (!begin.value || !end.value)
        return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "interval is not representable");
    if (rational_compare(*end.value, *begin.value) == std::strong_ordering::less)
        return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "interval ends before it begins");
    const auto length = rational_subtract(*end.value, *begin.value);
    if (!length.value) return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "interval length overflows");
    // The capability gate lives here rather than in each component: a
    // fractional step is a property of the request, not of the physics.
    if (length.value->denominator != 1 &&
        (state->granted_capabilities & KARMA_CAPABILITY_FRACTIONAL_STEPS) == 0)
        return note(instance, KARMA_STATUS_UNSUPPORTED_CAPABILITY, "fractional steps were not granted");
    const auto reason = state->component->advance(*begin.value, *end.value);
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "advance refused");
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_commit(void* instance)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    if (!state->initialized) return note(instance, KARMA_STATUS_NOT_INITIALIZED, "commit before initialize");
    KARMA_NATIVE_GUARD_BEGIN
    const auto reason = state->component->commit();
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "commit refused");
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_checkpoint(void* instance, std::uint64_t* token)
{
    if (instance == nullptr || token == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    if ((state->granted_capabilities & KARMA_CAPABILITY_CHECKPOINT_RESTORE) == 0)
        return note(instance, KARMA_STATUS_UNSUPPORTED_CAPABILITY, "checkpointing was not granted");
    KARMA_NATIVE_GUARD_BEGIN
    const auto reason = state->component->checkpoint(*token);
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "checkpoint refused");
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_restore(void* instance, std::uint64_t token)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    KARMA_NATIVE_GUARD_BEGIN
    const auto reason = state->component->restore(token);
    if (reason == OutcomeReason::InvalidArgument)
        return note(instance, KARMA_STATUS_UNKNOWN_CHECKPOINT, "no such checkpoint");
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "restore refused");
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_release_checkpoint(void* instance, std::uint64_t token)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    KARMA_NATIVE_GUARD_BEGIN
    const auto reason = state->component->release_checkpoint(token);
    if (reason == OutcomeReason::InvalidArgument)
        return note(instance, KARMA_STATUS_UNKNOWN_CHECKPOINT, "no such checkpoint");
    if (reason != OutcomeReason::Success) return note(instance, status_of_outcome(reason), "release refused");
    return KARMA_STATUS_OK;
    KARMA_NATIVE_GUARD_END(instance)
}

karma_status native_diagnostics(void* instance, karma_diagnostic* diagnostic)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* state = holder(instance);
    return set_abi_diagnostic(diagnostic, state->last_status, state->last_message.c_str());
}

karma_status native_poll_cancellation(void* instance, std::int32_t* cancelled)
{
    if (instance == nullptr || cancelled == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    *cancelled = 0;
    return KARMA_STATUS_OK;
}

void native_destroy(void* instance) { delete holder(instance); }

karma_status native_advance_seconds(void* instance, double begin, double end)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto exact_begin = rational_from_dyadic_bounded(begin, 32);
    const auto exact_end = rational_from_dyadic_bounded(end, 32);
    if (!exact_begin.value || !exact_end.value)
        return note(instance, KARMA_STATUS_INVALID_ARGUMENT, "a seconds value is not exactly representable");
    karma_exact_interval interval;
    interval.begin.numerator = exact_begin.value->numerator;
    interval.begin.denominator = exact_begin.value->denominator;
    interval.end.numerator = exact_end.value->numerator;
    interval.end.denominator = exact_end.value->denominator;
    return native_advance(instance, interval);
}

karma_field_table host_array_table(const std::string& field_id, double* data, std::size_t count, std::int32_t access,
                                      std::int32_t update)
{
    karma_field_table table{};
    table.struct_size = static_cast<std::uint32_t>(sizeof(table));
    table.field_id = field_id.c_str();
    table.element_type = KARMA_ELEMENT_F64;
    table.ownership = KARMA_OWNERSHIP_OWNED;
    table.access = access;
    table.update = update;
    table.rank = 1;
    table.components = 1;
    table.extents[0] = static_cast<std::int64_t>(count);
    table.byte_strides[0] = static_cast<std::int64_t>(sizeof(double));
    table.data = data;
    table.element_count = count;
    return table;
}
} // namespace

const karma_component_table& native_component_table()
{
    static const karma_component_table table = {static_cast<std::uint32_t>(sizeof(karma_component_table)),
                                                   native_create,
                                                   native_bind_host,
                                                   native_initialize,
                                                   native_metadata,
                                                   native_import_fields,
                                                   native_export_fields,
                                                   native_advance,
                                                   native_commit,
                                                   native_checkpoint,
                                                   native_restore,
                                                   native_release_checkpoint,
                                                   native_diagnostics,
                                                   native_poll_cancellation,
                                                   native_destroy,
                                                   native_advance_seconds};
    return table;
}

RegistryReason register_native_component(ComponentRegistry& registry, RegistryRecord record,
                                         NativeComponentFactory factory)
{
    if (!factory) return RegistryReason::MissingTable;
    record.table = &native_component_table();
    const auto id = record.implementation_id;
    // The registry validates first, so a rejected record leaves no orphan
    // factory behind under an identifier nothing can select.
    const auto reason = registry.add(std::move(record));
    if (reason != RegistryReason::Success) return reason;
    const std::lock_guard<std::mutex> lock(factory_mutex());
    factories()[id] = std::move(factory);
    return RegistryReason::Success;
}

/** Every reason that corresponds to an ABI status maps to exactly that
 * status. The four reasons with no ABI counterpart - a duplicate staged
 * entry, an unknown staged entry, a failed local validation and a failed
 * rollback - are KARMA-side transaction outcomes that a component never
 * reports across the boundary; each maps to the closest status and is
 * listed explicitly rather than swept into a default, so adding a reason
 * forces a decision here instead of silently becoming an internal error. */
karma_status status_of_outcome(OutcomeReason reason) noexcept
{
    switch (reason)
    {
        case OutcomeReason::Success: return KARMA_STATUS_OK;
        case OutcomeReason::InvalidArgument: return KARMA_STATUS_INVALID_ARGUMENT;
        case OutcomeReason::TruncatedPrefix: return KARMA_STATUS_TRUNCATED_PREFIX;
        case OutcomeReason::MalformedSize: return KARMA_STATUS_MALFORMED_SIZE;
        case OutcomeReason::IncompatibleMajor: return KARMA_STATUS_INCOMPATIBLE_MAJOR;
        case OutcomeReason::UnsupportedCapability: return KARMA_STATUS_UNSUPPORTED_CAPABILITY;
        case OutcomeReason::TableTooSmall: return KARMA_STATUS_TABLE_TOO_SMALL;
        case OutcomeReason::UnknownField: return KARMA_STATUS_UNKNOWN_FIELD;
        case OutcomeReason::MalformedFieldTable: return KARMA_STATUS_MALFORMED_FIELD_TABLE;
        case OutcomeReason::NonfiniteValue: return KARMA_STATUS_NONFINITE_VALUE;
        case OutcomeReason::IncompleteUpdate: return KARMA_STATUS_INCOMPLETE_UPDATE;
        case OutcomeReason::StorageFailure: return KARMA_STATUS_STORAGE_FAILURE;
        case OutcomeReason::UnknownCheckpoint: return KARMA_STATUS_UNKNOWN_CHECKPOINT;
        case OutcomeReason::NotInitialized: return KARMA_STATUS_NOT_INITIALIZED;
        case OutcomeReason::Cancelled: return KARMA_STATUS_CANCELLED;
        case OutcomeReason::Internal: return KARMA_STATUS_INTERNAL_ERROR;

        // KARMA-side only; no ABI status names these exactly.
        case OutcomeReason::DuplicateEntry: return KARMA_STATUS_MALFORMED_FIELD_TABLE;
        case OutcomeReason::UnknownEntry: return KARMA_STATUS_UNKNOWN_FIELD;
        case OutcomeReason::ValidationFailed: return KARMA_STATUS_INVALID_ARGUMENT;
        case OutcomeReason::RollbackFailed: return KARMA_STATUS_STORAGE_FAILURE;
    }
    return KARMA_STATUS_INTERNAL_ERROR;
}

/** The inverse, and total: every ABI status has its own reason, so a status
 * crossing into a collective reconciliation keeps its meaning. It used to
 * collapse four statuses onto InvalidArgument and, through a default arm,
 * eight more onto Internal - so two ranks failing for genuinely different
 * reasons agreed on a verdict that named neither. */
OutcomeReason outcome_of_status(karma_status status) noexcept
{
    switch (status)
    {
        case KARMA_STATUS_OK: return OutcomeReason::Success;
        case KARMA_STATUS_INVALID_ARGUMENT: return OutcomeReason::InvalidArgument;
        case KARMA_STATUS_TRUNCATED_PREFIX: return OutcomeReason::TruncatedPrefix;
        case KARMA_STATUS_MALFORMED_SIZE: return OutcomeReason::MalformedSize;
        case KARMA_STATUS_INCOMPATIBLE_MAJOR: return OutcomeReason::IncompatibleMajor;
        case KARMA_STATUS_UNSUPPORTED_CAPABILITY: return OutcomeReason::UnsupportedCapability;
        case KARMA_STATUS_TABLE_TOO_SMALL: return OutcomeReason::TableTooSmall;
        case KARMA_STATUS_UNKNOWN_FIELD: return OutcomeReason::UnknownField;
        case KARMA_STATUS_MALFORMED_FIELD_TABLE: return OutcomeReason::MalformedFieldTable;
        case KARMA_STATUS_NONFINITE_VALUE: return OutcomeReason::NonfiniteValue;
        case KARMA_STATUS_INCOMPLETE_UPDATE: return OutcomeReason::IncompleteUpdate;
        case KARMA_STATUS_STORAGE_FAILURE: return OutcomeReason::StorageFailure;
        case KARMA_STATUS_UNKNOWN_CHECKPOINT: return OutcomeReason::UnknownCheckpoint;
        case KARMA_STATUS_NOT_INITIALIZED: return OutcomeReason::NotInitialized;
        case KARMA_STATUS_CANCELLED: return OutcomeReason::Cancelled;
        case KARMA_STATUS_INTERNAL_ERROR: return OutcomeReason::Internal;
        default: break;
    }
    // An unrecognized status is a real failure, not a success.
    return OutcomeReason::Internal;
}

ComponentInstance::ComponentInstance(ComponentInstance&& other) noexcept
    : _table(other._table), _instance(other._instance)
{
    other._table = nullptr;
    other._instance = nullptr;
}

ComponentInstance& ComponentInstance::operator=(ComponentInstance&& other) noexcept
{
    if (this != &other)
    {
        if (_instance != nullptr && _table != nullptr) _table->destroy(_instance);
        _table = other._table;
        _instance = other._instance;
        other._table = nullptr;
        other._instance = nullptr;
    }
    return *this;
}

ComponentInstance::~ComponentInstance()
{
    if (_instance != nullptr && _table != nullptr) _table->destroy(_instance);
}

ComponentInstanceCreateResult ComponentInstance::create(const RegistryRecord& record,
                                                          std::span<const std::byte> configuration,
                                                          std::uint64_t granted_capabilities,
                                                          karma_communicator communicator)
{
    CreateResult result;
    if (record.table == nullptr || record.table->struct_size < sizeof(karma_component_table))
    {
        result.status = KARMA_STATUS_INVALID_ARGUMENT;
        return result;
    }

    karma_component_config config{};
    config.struct_size = static_cast<std::uint32_t>(sizeof(config));
    config.implementation_id = record.implementation_id.c_str();
    config.configuration = configuration.empty() ? nullptr : configuration.data();
    config.configuration_size = configuration.size();
    config.communicator = communicator;
    config.granted_capabilities = granted_capabilities;

    void* instance = nullptr;
    result.status = record.table->create(&config, &instance);
    if (result.status != KARMA_STATUS_OK || instance == nullptr)
    {
        // Nothing is retained on failure, so the handle stays empty and its
        // destructor has nothing to destroy.
        result.status = result.status == KARMA_STATUS_OK ? KARMA_STATUS_INTERNAL_ERROR : result.status;
        return result;
    }
    result.instance._table = record.table;
    result.instance._instance = instance;
    return result;
}

karma_status ComponentInstance::bind_host()
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    static karma_host_table host = [] {
        karma_host_table table{};
        table.struct_size = static_cast<std::uint32_t>(sizeof(table));
        table.host_context = nullptr;
        table.report = [](void*, const karma_diagnostic*) { return KARMA_STATUS_OK; };
        table.communicator_size = [](void*, karma_communicator, std::int32_t* size) {
            if (size == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
            *size = 1;
            return KARMA_STATUS_OK;
        };
        table.communicator_rank = [](void*, karma_communicator, std::int32_t* rank) {
            if (rank == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
            *rank = 0;
            return KARMA_STATUS_OK;
        };
        return table;
    }();
    return _table->bind_host(_instance, &host);
}

karma_status ComponentInstance::initialize(Rational start)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    karma_exact_time time;
    time.numerator = start.numerator;
    time.denominator = start.denominator;
    return _table->initialize(_instance, time);
}

karma_status ComponentInstance::metadata(karma_component_metadata& out) const
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    out.struct_size = static_cast<std::uint32_t>(sizeof(out));
    return _table->metadata(_instance, &out);
}

karma_status ComponentInstance::import_fields(std::span<const karma_field_table> tables)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->import_fields(_instance, tables.data(), static_cast<std::uint32_t>(tables.size()));
}

karma_status ComponentInstance::export_fields(std::span<const karma_field_table> tables) const
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->export_fields(_instance, tables.data(), static_cast<std::uint32_t>(tables.size()));
}

karma_status ComponentInstance::advance(Rational begin, Rational end)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    karma_exact_interval interval;
    interval.begin.numerator = begin.numerator;
    interval.begin.denominator = begin.denominator;
    interval.end.numerator = end.numerator;
    interval.end.denominator = end.denominator;
    return _table->advance(_instance, interval);
}

karma_status ComponentInstance::commit()
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->commit(_instance);
}

karma_status ComponentInstance::checkpoint(std::uint64_t& token)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->checkpoint(_instance, &token);
}

karma_status ComponentInstance::restore(std::uint64_t token)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->restore(_instance, token);
}

karma_status ComponentInstance::release_checkpoint(std::uint64_t token)
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    return _table->release_checkpoint(_instance, token);
}

karma_status ComponentInstance::diagnostics(karma_diagnostic& out) const
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    out.struct_size = static_cast<std::uint32_t>(sizeof(out));
    return _table->diagnostics(_instance, &out);
}

karma_status ComponentInstance::poll_cancellation(bool& cancelled) const
{
    if (!valid()) return KARMA_STATUS_INVALID_ARGUMENT;
    std::int32_t flag = 0;
    const auto status = _table->poll_cancellation(_instance, &flag);
    cancelled = flag != 0;
    return status;
}

ReconciledOutcome ComponentInstance::advance_collective(MPI_Comm communicator, Rational begin, Rational end)
{
    // Every rank advances locally, then one agreed verdict is formed before
    // anyone proceeds. A rank that returned early would leave its peers in the
    // next collective with no partner.
    const auto status = advance(begin, end);
    LocalOutcome local;
    local.reason = outcome_of_status(status);
    return reconcile_outcome(communicator, local);
}

ComplianceReport check_component_compliance(const RegistryRecord& record, const ComplianceFixture& fixture)
{
    ComplianceReport report;
    const auto check = [&](bool condition, const char* what) {
        ++report.checks;
        if (!condition) report.failures.emplace_back(what);
    };

    const std::size_t count = fixture.element_count;
    std::vector<double> seed(count);
    for (std::size_t index = 0; index != count; ++index) seed[index] = 1.0 + static_cast<double>(index);

    // Wrong independent of what the component does: a record whose own
    // contract does not validate is a defect in the record, and nothing
    // downstream should have to discover that by watching the component
    // misbehave.
    const auto contract_validation = validate_component_semantics(record.contract);
    check(contract_validation.ok(),
          "the registry record's contract does not validate under validate_component_semantics");

    // The canonical field-entry encoding gets its first production caller
    // here, composing it with the compliance harness rather than only its
    // own test. This is deliberately NOT one of the checks the harness
    // reports: round-trip correctness is a property of the encoding
    // functions, not of the component under test - karma_field_entry_
    // encoding_test.C already mutation-tests it - so no fixture component
    // could ever make it fail without the encoding module itself being
    // broken. A failure here throws instead of being recorded as a
    // compliance failure, because it means this library has a defect, not
    // that the component does.
    FieldEntryKey sample_key;
    sample_key.kind = FieldEntryKey::Kind::MoabGlobalId;
    sample_key.global_id = 1;
    sample_key.component = 0;
    const auto encoded = encode_field_entry_key(sample_key);
    const auto decoded = decode_field_entry_key(encoded);
    if (!decoded.key.has_value() || *decoded.key != sample_key)
        throw std::logic_error("the canonical field-entry encoding did not round-trip inside the compliance harness");

    auto first = ComponentInstance::create(record, fixture.configuration, fixture.granted_capabilities);
    check(first.status == KARMA_STATUS_OK && first.instance.valid(), "create failed");
    if (!first.instance.valid()) return report;

    // Ordering: work before initialize is refused, not undefined.
    auto import_table = host_array_table(fixture.field_id, seed.data(), count, KARMA_ACCESS_READ_WRITE,
                                         KARMA_UPDATE_REPLACE);
    check(first.instance.import_fields({&import_table, 1}) == KARMA_STATUS_NOT_INITIALIZED,
          "import before initialize was allowed");
    check(first.instance.advance(fixture.start, fixture.step_end) == KARMA_STATUS_NOT_INITIALIZED,
          "advance before initialize was allowed");

    check(first.instance.bind_host() == KARMA_STATUS_OK, "bind_host failed");
    check(first.instance.initialize(fixture.start) == KARMA_STATUS_OK, "initialize failed");

    // The record must not overstate its component: a record claiming a
    // capability the component does not provide would be believed by every
    // selection.
    karma_component_metadata metadata{};
    check(first.instance.metadata(metadata) == KARMA_STATUS_OK, "metadata failed");
    check((record.provided_capabilities & ~metadata.provided_capabilities) == 0,
          "the registry record claims capabilities the component does not provide");
    check(metadata.implementation_version.major == record.implementation_version.major &&
              metadata.implementation_version.minor == record.implementation_version.minor &&
              metadata.implementation_version.patch == record.implementation_version.patch,
          "the registry record and the component disagree about the version");

    check(first.instance.import_fields({&import_table, 1}) == KARMA_STATUS_OK, "import failed");

    // Lifetime: the table is valid for the call, so the component must have
    // copied. Mutating the caller's buffer must not change what it holds.
    const double original = seed[0];
    seed[0] = original + 1000.0;
    std::vector<double> exported(count, 0.0);
    auto export_table = host_array_table(fixture.field_id, exported.data(), count, KARMA_ACCESS_READ_WRITE,
                                         KARMA_UPDATE_REPLACE);
    check(first.instance.export_fields({&export_table, 1}) == KARMA_STATUS_OK, "export failed");
    check(exported[0] == original, "the component retained the caller's buffer instead of copying");
    seed[0] = original;

    // Checkpoint, advance, restore: the restored state must be the state that
    // was checkpointed, byte for byte.
    if ((fixture.granted_capabilities & KARMA_CAPABILITY_CHECKPOINT_RESTORE) != 0)
    {
        std::uint64_t token = 0;
        check(first.instance.checkpoint(token) == KARMA_STATUS_OK, "checkpoint failed");
        check(first.instance.advance(fixture.start, fixture.step_end) == KARMA_STATUS_OK, "advance failed");
        check(first.instance.commit() == KARMA_STATUS_OK, "commit failed");
        std::vector<double> advanced(count, 0.0);
        auto advanced_table = host_array_table(fixture.field_id, advanced.data(), count, KARMA_ACCESS_READ_WRITE,
                                               KARMA_UPDATE_REPLACE);
        check(first.instance.export_fields({&advanced_table, 1}) == KARMA_STATUS_OK, "export after advance failed");
        check(first.instance.restore(token) == KARMA_STATUS_OK, "restore failed");
        std::vector<double> restored(count, 0.0);
        auto restored_table = host_array_table(fixture.field_id, restored.data(), count, KARMA_ACCESS_READ_WRITE,
                                               KARMA_UPDATE_REPLACE);
        check(first.instance.export_fields({&restored_table, 1}) == KARMA_STATUS_OK, "export after restore failed");
        check(std::memcmp(restored.data(), exported.data(), count * sizeof(double)) == 0,
              "restore did not reproduce the checkpointed state exactly");
        check(first.instance.release_checkpoint(token) == KARMA_STATUS_OK, "release failed");
        check(first.instance.restore(token) == KARMA_STATUS_UNKNOWN_CHECKPOINT,
              "a released checkpoint was still restorable");
        check(first.instance.release_checkpoint(token) == KARMA_STATUS_UNKNOWN_CHECKPOINT,
              "releasing twice was accepted");

        // A deliberate error must leave a diagnostic behind.
        karma_diagnostic diagnostic{};
        check(first.instance.diagnostics(diagnostic) == KARMA_STATUS_OK, "diagnostics failed");
        check(diagnostic.status == KARMA_STATUS_UNKNOWN_CHECKPOINT, "the diagnostic lost the last error");
    }

    // Determinism: a second instance, same inputs, same steps, identical
    // bytes. Not approximately identical - a coupled run that is only
    // approximately reproducible cannot be restarted.
    auto second = ComponentInstance::create(record, fixture.configuration, fixture.granted_capabilities);
    check(second.status == KARMA_STATUS_OK && second.instance.valid(), "the second create failed");
    if (second.instance.valid())
    {
        check(second.instance.initialize(fixture.start) == KARMA_STATUS_OK, "the second initialize failed");
        auto seed_table = host_array_table(fixture.field_id, seed.data(), count, KARMA_ACCESS_READ_WRITE,
                                           KARMA_UPDATE_REPLACE);
        check(second.instance.import_fields({&seed_table, 1}) == KARMA_STATUS_OK, "the second import failed");
        check(second.instance.advance(fixture.start, fixture.step_end) == KARMA_STATUS_OK,
              "the second advance failed");
        check(second.instance.commit() == KARMA_STATUS_OK, "the second commit failed");

        std::vector<double> replayed(count, 0.0);
        auto replay_table = host_array_table(fixture.field_id, replayed.data(), count, KARMA_ACCESS_READ_WRITE,
                                             KARMA_UPDATE_REPLACE);
        check(second.instance.export_fields({&replay_table, 1}) == KARMA_STATUS_OK, "the second export failed");

        auto third = ComponentInstance::create(record, fixture.configuration, fixture.granted_capabilities);
        if (third.instance.valid())
        {
            check(third.instance.initialize(fixture.start) == KARMA_STATUS_OK, "the third initialize failed");
            auto again = host_array_table(fixture.field_id, seed.data(), count, KARMA_ACCESS_READ_WRITE,
                                          KARMA_UPDATE_REPLACE);
            check(third.instance.import_fields({&again, 1}) == KARMA_STATUS_OK, "the third import failed");
            check(third.instance.advance(fixture.start, fixture.step_end) == KARMA_STATUS_OK,
                  "the third advance failed");
            check(third.instance.commit() == KARMA_STATUS_OK, "the third commit failed");
            std::vector<double> twice(count, 0.0);
            auto twice_table = host_array_table(fixture.field_id, twice.data(), count, KARMA_ACCESS_READ_WRITE,
                                                KARMA_UPDATE_REPLACE);
            check(third.instance.export_fields({&twice_table, 1}) == KARMA_STATUS_OK, "the third export failed");
            check(std::memcmp(twice.data(), replayed.data(), count * sizeof(double)) == 0,
                  "two identical runs produced different bytes");
        }
    }

    // An unknown field is refused rather than guessed at.
    std::vector<double> spare(count, 0.0);
    auto wrong_field = host_array_table("karma.compliance.no-such-field", spare.data(), count,
                                        KARMA_ACCESS_READ_WRITE, KARMA_UPDATE_REPLACE);
    check(first.instance.import_fields({&wrong_field, 1}) != KARMA_STATUS_OK, "an unknown field was imported");

    return report;
}
} // namespace KARMA
