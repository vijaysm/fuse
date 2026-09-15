// The record layout a language adapter must reproduce.
//
// A Fortran or Python adapter cannot include a C header: it declares the
// records itself, in its own syntax, at byte offsets it has to hard-code. These
// assertions are those offsets. They are golden on purpose - if one changes,
// every adapter breaks, and the failure should be here rather than in a
// participant's memory.
//
// Sizes and offsets are for a 64-bit pointer and the platform's natural
// alignment; a future minor may only append.
#include "karma_abi.h"
#include "karma_component_abi.h"

#include <cstddef>
#include <cstdint>

static_assert(sizeof(void*) == 8, "this schema is for 64-bit pointers");

// --- karma_field_table: 144 bytes -------------------------------------
static_assert(sizeof(karma_field_table) == 144);
static_assert(offsetof(karma_field_table, struct_size) == 0);
static_assert(offsetof(karma_field_table, field_id) == 8);
static_assert(offsetof(karma_field_table, element_type) == 16);
static_assert(offsetof(karma_field_table, ownership) == 20);
static_assert(offsetof(karma_field_table, access) == 24);
static_assert(offsetof(karma_field_table, update) == 28);
static_assert(offsetof(karma_field_table, rank) == 32);
static_assert(offsetof(karma_field_table, components) == 36);
static_assert(offsetof(karma_field_table, extents) == 40);
static_assert(offsetof(karma_field_table, byte_strides) == 72);
static_assert(offsetof(karma_field_table, data) == 104);
static_assert(offsetof(karma_field_table, element_count) == 112);
static_assert(offsetof(karma_field_table, key_bytes) == 120);
static_assert(offsetof(karma_field_table, key_bytes_size) == 128);
static_assert(offsetof(karma_field_table, key_record_size) == 136);
static_assert(KARMA_ABI_FIELD_MAX_RANK == 4, "the extent and stride arrays are four wide");

// --- karma_component_table: 15 required entries plus one optional tail --
static_assert(sizeof(karma_component_table) == 128);
static_assert(offsetof(karma_component_table, struct_size) == 0);
static_assert(offsetof(karma_component_table, create) == 8);
static_assert(offsetof(karma_component_table, advance) == 56);
static_assert(offsetof(karma_component_table, destroy) == 112);
// The tail. An adapter detects it by struct_size rather than assuming it.
static_assert(offsetof(karma_component_table, advance_seconds) == 120);

// --- negotiation and configuration ---------------------------------------
static_assert(sizeof(karma_abi_request) == 40);
static_assert(sizeof(karma_abi_agreement) == 40);
static_assert(offsetof(karma_abi_agreement, unsupported_required) == 24);
static_assert(sizeof(karma_component_config) == 48);
static_assert(sizeof(karma_component_metadata) == 40);
static_assert(sizeof(karma_host_table) == 40);
static_assert(sizeof(karma_plugin_query_result) == 56);

// --- diagnostics: 12 bytes of header then the fixed buffer ---------------
static_assert(sizeof(karma_diagnostic) == 524);
static_assert(offsetof(karma_diagnostic, status) == 4);
static_assert(offsetof(karma_diagnostic, truncated) == 8);
static_assert(offsetof(karma_diagnostic, message) == 12);

// --- the values an adapter must mirror as its own constants --------------
static_assert(KARMA_STATUS_OK == 0);
static_assert(KARMA_STATUS_INTERNAL_ERROR == 15);
static_assert(KARMA_ELEMENT_F64 == 0);
static_assert(KARMA_UPDATE_REPLACE == 0);
static_assert(KARMA_UPDATE_ACCUMULATE == 2);
static_assert(KARMA_ACCESS_READ_ONLY == 0);
static_assert(KARMA_OWNERSHIP_OWNED == 0);
static_assert(KARMA_CAPABILITY_HOST_ARRAY_FIELDS == 1);
static_assert(KARMA_CAPABILITY_COMMUNICATOR == 32);
static_assert(KARMA_COMMUNICATOR_NONE == 0);

int main() { return 0; }
