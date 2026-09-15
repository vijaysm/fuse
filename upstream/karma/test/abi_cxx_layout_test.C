// The same records seen from C++: standard layout, trivially copyable, and the
// same sizes the C compiler computes. If these two views ever disagree, the ABI
// is not an ABI.
// The stem differs from abi_c11_conformance_test.c on purpose: a .c and a .C
// file with the same stem are one file on a case-insensitive filesystem.
#include "karma_abi.h"
#include "karma_component_abi.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace
{
template <typename T> constexpr bool crosses_boundary()
{
    return std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>;
}
}

static_assert(crosses_boundary<karma_exact_time>());
static_assert(crosses_boundary<karma_exact_interval>());
static_assert(crosses_boundary<karma_version>());
static_assert(crosses_boundary<karma_diagnostic>());
static_assert(crosses_boundary<karma_abi_request>());
static_assert(crosses_boundary<karma_abi_agreement>());
static_assert(crosses_boundary<karma_field_table>());
static_assert(crosses_boundary<karma_component_config>());
static_assert(crosses_boundary<karma_component_metadata>());
static_assert(crosses_boundary<karma_host_table>());
static_assert(crosses_boundary<karma_component_table>());
static_assert(crosses_boundary<karma_plugin_query_result>());

// Fixed widths: nothing here may follow the platform's int.
static_assert(sizeof(karma_status) == 4);
static_assert(std::is_same_v<karma_status, std::int32_t>);
static_assert(sizeof(karma_exact_time) == 16);
static_assert(sizeof(karma_communicator) == 8);
static_assert(std::is_same_v<decltype(karma_exact_time{}.numerator), std::int64_t>);

// struct_size first, always: a reader that cannot find the size cannot decide
// whether it may read the rest.
static_assert(offsetof(karma_diagnostic, struct_size) == 0);
static_assert(offsetof(karma_abi_request, struct_size) == 0);
static_assert(offsetof(karma_abi_agreement, struct_size) == 0);
static_assert(offsetof(karma_field_table, struct_size) == 0);
static_assert(offsetof(karma_component_config, struct_size) == 0);
static_assert(offsetof(karma_component_metadata, struct_size) == 0);
static_assert(offsetof(karma_host_table, struct_size) == 0);
static_assert(offsetof(karma_component_table, struct_size) == 0);
static_assert(offsetof(karma_plugin_query_result, struct_size) == 0);

// The bounded diagnostic must stay bounded, or it is not usable without an
// allocator.
static_assert(sizeof(karma_diagnostic) < 1024);
static_assert(KARMA_ABI_DIAGNOSTIC_CAPACITY == 512);

int main()
{
    // Runtime confirmation that the C and C++ views agree on the sizes the
    // negotiation reports, which is what a language adapter allocates from.
    karma_abi_request request{};
    karma_abi_agreement agreement{};
    request.struct_size = static_cast<std::uint32_t>(sizeof(request));
    request.abi_major = 1;
    request.host_table_size = static_cast<std::uint32_t>(sizeof(karma_host_table));
    agreement.struct_size = static_cast<std::uint32_t>(sizeof(agreement));
    if (karma_negotiate_abi(&request, &agreement) != KARMA_STATUS_OK)
    {
        std::cerr << "negotiation failed\n";
        return 1;
    }
    if (agreement.required_host_table_size != sizeof(karma_host_table) ||
        agreement.plugin_table_size != sizeof(karma_component_table))
    {
        std::cerr << "the reported table sizes disagree with this header\n";
        return 1;
    }
    return 0;
}
