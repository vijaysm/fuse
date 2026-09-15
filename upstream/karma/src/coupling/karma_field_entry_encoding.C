#include "karma_field_entry_encoding.h"

#include "karma_byte_writer.h"

#include <algorithm>
#include <array>

namespace KARMA
{
namespace
{
constexpr std::size_t length_prefix_width = 4;
constexpr std::size_t payload_width = field_entry_key_encoded_size - length_prefix_width;

constexpr std::uint8_t moab_discriminant = 0;
constexpr std::uint8_t dof_discriminant = 1;


void put(std::span<std::byte> buffer, std::size_t& offset, std::uint64_t value, unsigned width) noexcept
{
    for (unsigned index = 0; index != width; ++index)
        buffer[offset++] = static_cast<std::byte>((value >> (8 * index)) & 0xffU);
}

std::uint64_t get(std::span<const std::byte> bytes, std::size_t& offset, unsigned width) noexcept
{
    std::uint64_t value = 0;
    for (unsigned index = 0; index != width; ++index)
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned>(bytes[offset++])) << (8 * index);
    return value;
}
}

bool encode_field_entry_key(const FieldEntryKey& key, std::span<std::byte> buffer) noexcept
{
    if (buffer.size() < field_entry_key_encoded_size) return false;
    std::size_t offset = 0;
    put(buffer, offset, payload_width, 4);
    put(buffer, offset, field_entry_encoding_version, 1);
    put(buffer, offset, key.kind == FieldEntryKey::Kind::Dof ? dof_discriminant : moab_discriminant, 1);
    put(buffer, offset, key.global_id, 8);
    put(buffer, offset, key.component, 4);
    for (const auto word : key.dof.words) put(buffer, offset, word, 8);
    return offset == field_entry_key_encoded_size;
}

std::vector<std::byte> encode_field_entry_key(const FieldEntryKey& key)
{
    std::vector<std::byte> bytes(field_entry_key_encoded_size);
    // The fixed-size buffer cannot be too small; the result is checked to keep
    // the two overloads from drifting apart.
    const bool encoded = encode_field_entry_key(key, bytes);
    return encoded ? bytes : std::vector<std::byte>{};
}

FieldEntryDecodeResult decode_field_entry_key(std::span<const std::byte> bytes) noexcept
{
    if (bytes.size() < length_prefix_width) return {std::nullopt, FieldEntryDecodeError::Truncated};
    std::size_t offset = 0;
    const auto declared = get(bytes, offset, 4);
    // A declared length that disagrees with this version's fixed record is a
    // schema error, not a truncation, and is reported separately.
    if (declared != payload_width) return {std::nullopt, FieldEntryDecodeError::LengthMismatch};
    if (bytes.size() < field_entry_key_encoded_size) return {std::nullopt, FieldEntryDecodeError::Truncated};
    if (bytes.size() > field_entry_key_encoded_size) return {std::nullopt, FieldEntryDecodeError::TrailingBytes};

    if (get(bytes, offset, 1) != field_entry_encoding_version)
        return {std::nullopt, FieldEntryDecodeError::UnsupportedVersion};

    const auto discriminant = get(bytes, offset, 1);
    if (discriminant != moab_discriminant && discriminant != dof_discriminant)
        return {std::nullopt, FieldEntryDecodeError::UnknownKind};

    FieldEntryKey key;
    key.kind = discriminant == dof_discriminant ? FieldEntryKey::Kind::Dof : FieldEntryKey::Kind::MoabGlobalId;
    key.global_id = get(bytes, offset, 8);
    key.component = static_cast<unsigned>(get(bytes, offset, 4));
    for (auto& word : key.dof.words) word = get(bytes, offset, 8);
    return {key, FieldEntryDecodeError::None};
}

std::uint64_t field_entry_key_hash(const FieldEntryKey& key) noexcept
{
    std::array<std::byte, field_entry_key_encoded_size> buffer{};
    if (!encode_field_entry_key(key, buffer)) return 0;
    return bytes::fnv1a(buffer);
}
} // namespace KARMA
