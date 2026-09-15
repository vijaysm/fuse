// Adversarial negotiation and boundary behavior: what the ABI does when it is
// handed something wrong, which is the only interesting half of a stable ABI.
#include "karma_abi.h"
#include "karma_component_abi.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

karma_abi_request good_request()
{
    karma_abi_request request{};
    request.struct_size = static_cast<std::uint32_t>(sizeof(request));
    request.abi_major = 1;
    request.abi_minor = 0;
    request.required_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS;
    request.host_table_size = static_cast<std::uint32_t>(sizeof(karma_host_table));
    return request;
}

karma_abi_agreement empty_agreement()
{
    karma_abi_agreement agreement{};
    agreement.struct_size = static_cast<std::uint32_t>(sizeof(agreement));
    return agreement;
}

void null_arguments()
{
    auto agreement = empty_agreement();
    auto request = good_request();
    require(karma_negotiate_abi(nullptr, &agreement) == KARMA_STATUS_INVALID_ARGUMENT, "null request accepted");
    require(karma_negotiate_abi(&request, nullptr) == KARMA_STATUS_INVALID_ARGUMENT, "null agreement accepted");
    require(karma_validate_field_table(nullptr) == KARMA_STATUS_INVALID_ARGUMENT, "null table accepted");
    require(karma_plugin_query(nullptr) == KARMA_STATUS_INVALID_ARGUMENT, "null query accepted");
    require(karma_diagnostic_set(nullptr, KARMA_STATUS_OK, "x") == KARMA_STATUS_INVALID_ARGUMENT,
            "null diagnostic accepted");
}

void size_discipline()
{
    // A record smaller than this version can read is truncated: the caller is
    // older than the library.
    auto truncated = good_request();
    truncated.struct_size = static_cast<std::uint32_t>(sizeof(truncated) - 1);
    auto agreement = empty_agreement();
    require(karma_negotiate_abi(&truncated, &agreement) == KARMA_STATUS_TRUNCATED_PREFIX,
            "a truncated request was accepted");

    // A record larger than this version knows is a future tail and is read up
    // to the part we understand. This is the whole minor-extension mechanism.
    std::vector<unsigned char> oversized(sizeof(karma_abi_request) + 64, 0xAB);
    auto* future = reinterpret_cast<karma_abi_request*>(oversized.data());
    *future = good_request();
    future->struct_size = static_cast<std::uint32_t>(oversized.size());
    agreement = empty_agreement();
    require(karma_negotiate_abi(future, &agreement) == KARMA_STATUS_OK,
            "a request with a future tail was refused");

    // Zero and wild sizes are corrupted records, not very new ones, and are
    // reported differently so a caller can tell the two apart.
    auto zero = good_request();
    zero.struct_size = 0;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&zero, &agreement) == KARMA_STATUS_MALFORMED_SIZE, "a zero size was accepted");
    auto wild = good_request();
    wild.struct_size = KARMA_ABI_MAX_STRUCT_SIZE + 1;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&wild, &agreement) == KARMA_STATUS_MALFORMED_SIZE, "a wild size was accepted");

    // The out parameter is size-checked too: writing past a caller's smaller
    // record would corrupt its stack.
    auto request = good_request();
    auto small = empty_agreement();
    small.struct_size = static_cast<std::uint32_t>(sizeof(small) - 4);
    require(karma_negotiate_abi(&request, &small) == KARMA_STATUS_TRUNCATED_PREFIX,
            "a truncated agreement was written into");
}

void version_and_capabilities()
{
    auto wrong_major = good_request();
    wrong_major.abi_major = 2;
    auto agreement = empty_agreement();
    require(karma_negotiate_abi(&wrong_major, &agreement) == KARMA_STATUS_INCOMPATIBLE_MAJOR,
            "a foreign major version was accepted");
    // The refusal still reports the library's own major, so the caller learns
    // what it should have asked for instead of only that it was wrong.
    require(agreement.abi_major == 1, "a refusal did not report the supported major");

    // A newer caller talks down to the minor the library knows.
    auto newer = good_request();
    newer.abi_minor = 7;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&newer, &agreement) == KARMA_STATUS_OK, "a newer minor was refused");
    require(agreement.abi_minor == KARMA_ABI_MINOR, "the agreed minor is not the lower of the two");

    // An unknown *required* bit is a refusal, and the bits are named back.
    // Ignoring it would tell a participant it got a guarantee it did not.
    const std::uint64_t invented = UINT64_C(1) << 40;
    auto demanding = good_request();
    demanding.required_capabilities |= invented;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&demanding, &agreement) == KARMA_STATUS_UNSUPPORTED_CAPABILITY,
            "an unknown required capability was granted");
    require(agreement.unsupported_required == invented, "the unsupported bits were not reported");
    require((agreement.granted_capabilities & invented) == 0, "an unknown bit appeared as granted");

    // An unknown *optional* bit is simply not granted.
    auto hopeful = good_request();
    hopeful.optional_capabilities = invented;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&hopeful, &agreement) == KARMA_STATUS_OK, "an optional bit caused a refusal");
    require((agreement.granted_capabilities & invented) == 0, "an unknown optional bit was granted");

    // Size discovery rather than a bare refusal.
    auto small_table = good_request();
    small_table.host_table_size = 8;
    agreement = empty_agreement();
    require(karma_negotiate_abi(&small_table, &agreement) == KARMA_STATUS_TABLE_TOO_SMALL,
            "an undersized host table was accepted");
    require(agreement.required_host_table_size == sizeof(karma_host_table),
            "the required host table size was not reported");
    require(agreement.plugin_table_size == sizeof(karma_component_table),
            "the plugin table size was not reported");
}

void diagnostic_truncation()
{
    karma_diagnostic diagnostic{};
    diagnostic.struct_size = static_cast<std::uint32_t>(sizeof(diagnostic));
    const std::string enormous(KARMA_ABI_DIAGNOSTIC_CAPACITY * 3, 'x');
    require(karma_diagnostic_set(&diagnostic, KARMA_STATUS_STORAGE_FAILURE, enormous.c_str()) ==
                KARMA_STATUS_OK,
            "setting a long diagnostic failed");
    require(diagnostic.truncated == 1, "a truncated diagnostic did not say so");
    // Always NUL-terminated: a message that is silently cut and reads as
    // complete is worse than no message.
    require(std::strlen(diagnostic.message) == KARMA_ABI_DIAGNOSTIC_CAPACITY - 1,
            "the truncated message is the wrong length");
    require(diagnostic.status == KARMA_STATUS_STORAGE_FAILURE, "the status was lost");

    require(karma_diagnostic_set(&diagnostic, KARMA_STATUS_OK, "short") == KARMA_STATUS_OK, "set failed");
    require(diagnostic.truncated == 0, "a short diagnostic claimed truncation");
    require(std::string(diagnostic.message) == "short", "the message was not copied");

    // An unknown status names itself as unknown rather than as something
    // plausible.
    require(std::string(karma_status_name(9999)).empty(), "an unknown status acquired a name");
}

void malformed_field_tables()
{
    double values[4] = {1, 2, 3, 4};
    const auto base = [&] {
        karma_field_table table{};
        table.struct_size = static_cast<std::uint32_t>(sizeof(table));
        table.field_id = "state";
        table.element_type = KARMA_ELEMENT_F64;
        table.ownership = KARMA_OWNERSHIP_OWNED;
        table.access = KARMA_ACCESS_READ_WRITE;
        table.update = KARMA_UPDATE_REPLACE;
        table.rank = 1;
        table.components = 1;
        table.extents[0] = 4;
        table.byte_strides[0] = sizeof(double);
        table.data = values;
        table.element_count = 4;
        return table;
    };
    const auto valid = base();
    require(karma_validate_field_table(&valid) == KARMA_STATUS_OK, "a valid table was rejected");

    struct Case
    {
        const char* what;
        karma_field_table table;
    };
    std::vector<Case> cases;
    auto add = [&](const char* what, auto&& mutate) {
        auto table = base();
        mutate(table);
        cases.push_back({what, table});
    };
    add("no field identifier", [](auto& t) { t.field_id = nullptr; });
    add("unknown element type", [](auto& t) { t.element_type = 99; });
    add("zero rank", [](auto& t) { t.rank = 0; });
    add("rank beyond the maximum", [](auto& t) { t.rank = KARMA_ABI_FIELD_MAX_RANK + 1; });
    add("zero components", [](auto& t) { t.components = 0; });
    add("zero extent", [](auto& t) { t.extents[0] = 0; });
    add("negative extent", [](auto& t) { t.extents[0] = -4; });
    // A stride narrower than one element makes two entries overlap in memory.
    add("stride narrower than an element", [](auto& t) { t.byte_strides[0] = 4; });
    add("zero stride", [](auto& t) { t.byte_strides[0] = 0; });
    add("element count disagreeing with the extents", [](auto& t) { t.element_count = 5; });
    // Lifetime: a nonempty table with no memory behind it.
    add("null data with a nonzero count", [](auto& t) { t.data = nullptr; });
    add("bad ownership", [](auto& t) { t.ownership = 7; });
    add("bad access mode", [](auto& t) { t.access = 7; });
    add("bad update mode", [](auto& t) { t.update = 7; });
    add("key size without keys", [](auto& t) { t.key_bytes_size = 16; });
    add("keys without a record size", [](auto& t) {
        t.key_bytes = "0123456789";
        t.key_record_size = 0;
        t.key_bytes_size = 10;
    });
    add("key bytes disagreeing with the record size", [](auto& t) {
        t.key_bytes = "0123456789";
        t.key_record_size = 4;
        t.key_bytes_size = 10;
    });
    for (const auto& one : cases)
        require(karma_validate_field_table(&one.table) == KARMA_STATUS_MALFORMED_FIELD_TABLE, one.what);

    auto zero_size = base();
    zero_size.struct_size = 0;
    require(karma_validate_field_table(&zero_size) == KARMA_STATUS_MALFORMED_SIZE,
            "a zero-sized table was not malformed");

    // A correctly sized keyed table is accepted: the canonical encoding is
    // 274 bytes per entry.
    auto keyed = base();
    std::vector<unsigned char> keys(274 * 4, 0);
    keyed.key_bytes = keys.data();
    keyed.key_record_size = 274;
    keyed.key_bytes_size = keys.size();
    require(karma_validate_field_table(&keyed) == KARMA_STATUS_OK, "a valid keyed table was rejected");
}

const karma_component_table& reference_table(karma_plugin_query_result& query)
{
    query.struct_size = static_cast<std::uint32_t>(sizeof(query));
    require(karma_plugin_query(&query) == KARMA_STATUS_OK, "plugin query failed");
    return *query.component_table;
}

void* create(const karma_component_table& table, const char* id, std::uint64_t* count,
             std::uint64_t capabilities)
{
    karma_component_config config{};
    config.struct_size = static_cast<std::uint32_t>(sizeof(config));
    config.implementation_id = id;
    config.configuration = count;
    config.configuration_size = sizeof(*count);
    config.granted_capabilities = capabilities;
    void* instance = nullptr;
    require(table.create(&config, &instance) == KARMA_STATUS_OK, "create failed");
    return instance;
}

void exception_containment()
{
    karma_plugin_query_result query{};
    const auto& table = reference_table(query);
    std::uint64_t count = 2;

    // The fixture component throws from advance(). The guard must turn that
    // into a status: unwinding through a C, Fortran, or Python frame is
    // undefined, so an escaping exception is not a lesser bug than a wrong
    // answer.
    void* thrower = create(table, "karma.reference.throwing", &count, ~UINT64_C(0));
    karma_exact_time start{0, 1};
    require(table.initialize(thrower, start) == KARMA_STATUS_OK, "initialize failed");
    karma_exact_interval interval{{0, 1}, {1, 1}};
    require(table.advance(thrower, interval) == KARMA_STATUS_INTERNAL_ERROR,
            "a thrown exception did not become an internal error");

    // And the message survives, so the failure is diagnosable.
    karma_diagnostic diagnostic{};
    diagnostic.struct_size = static_cast<std::uint32_t>(sizeof(diagnostic));
    require(table.diagnostics(thrower, &diagnostic) == KARMA_STATUS_OK, "diagnostics failed");
    require(diagnostic.status == KARMA_STATUS_INTERNAL_ERROR, "the diagnostic lost the internal error");
    require(std::string(diagnostic.message).find("throw") != std::string::npos,
            "the exception message did not reach the diagnostic");
    table.destroy(thrower);

    // An unknown implementation is refused rather than substituted.
    karma_component_config unknown{};
    unknown.struct_size = static_cast<std::uint32_t>(sizeof(unknown));
    unknown.implementation_id = "karma.nonexistent";
    unknown.configuration = &count;
    unknown.configuration_size = sizeof(count);
    void* instance = nullptr;
    require(table.create(&unknown, &instance) != KARMA_STATUS_OK, "an unknown implementation was created");
    require(instance == nullptr, "a failed create left a dangling instance");
}

void lifecycle_ordering_and_lifetime()
{
    karma_plugin_query_result query{};
    const auto& table = reference_table(query);
    std::uint64_t count = 3;
    void* instance = create(table, "karma.reference", &count, ~UINT64_C(0));

    karma_field_table field{};
    field.struct_size = static_cast<std::uint32_t>(sizeof(field));
    field.field_id = "state";
    field.element_type = KARMA_ELEMENT_F64;
    field.access = KARMA_ACCESS_READ_WRITE;
    field.update = KARMA_UPDATE_REPLACE;
    field.rank = 1;
    field.components = 1;
    field.extents[0] = 3;
    field.byte_strides[0] = sizeof(double);
    field.element_count = 3;

    std::vector<double> caller{1.0, 2.0, 3.0};
    field.data = caller.data();
    // Import before initialize is an ordering error, not undefined behavior.
    require(table.import_fields(instance, &field, 1) == KARMA_STATUS_NOT_INITIALIZED,
            "import before initialize was allowed");

    karma_exact_time start{0, 1};
    require(table.initialize(instance, start) == KARMA_STATUS_OK, "initialize failed");
    require(table.import_fields(instance, &field, 1) == KARMA_STATUS_OK, "import failed");

    // Lifetime: the table is valid for the call only, so the component must
    // have copied. Mutating the caller's buffer afterwards must not change
    // what the component holds.
    caller[0] = 99.0;
    std::vector<double> exported(3, 0.0);
    field.data = exported.data();
    require(table.export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export failed");
    require(exported[0] == 1.0, "the component retained the caller's buffer instead of copying");

    // A nonfinite import is refused and leaves the state untouched.
    std::vector<double> poisoned{1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    field.data = poisoned.data();
    require(table.import_fields(instance, &field, 1) == KARMA_STATUS_NONFINITE_VALUE,
            "a nonfinite import was accepted");
    field.data = exported.data();
    require(table.export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export failed");
    require(exported[0] == 1.0 && exported[2] == 3.0, "a refused import mutated the state");

    // A wrong-sized table is refused rather than truncated.
    auto wrong = field;
    wrong.extents[0] = 2;
    wrong.element_count = 2;
    require(table.import_fields(instance, &wrong, 1) == KARMA_STATUS_MALFORMED_FIELD_TABLE,
            "a wrong-sized import was accepted");

    // The optional seconds shadow converts exactly or refuses; it never rounds.
    require(table.advance_seconds != nullptr, "the optional tail function is absent");
    require(table.advance_seconds(instance, 0.0, 0.5) == KARMA_STATUS_OK, "an exact seconds step was refused");
    require(table.advance_seconds(instance, 0.0, 0.1) == KARMA_STATUS_INVALID_ARGUMENT,
            "a seconds value needing a 55-bit denominator was accepted");

    table.destroy(instance);
}
}

int main()
{
    try
    {
        null_arguments();
        size_discipline();
        version_and_capabilities();
        diagnostic_truncation();
        malformed_field_tables();
        exception_containment();
        lifecycle_ordering_and_lifetime();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
