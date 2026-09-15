#ifndef KARMA_COUPLING_FIELD_ENTRY_ENCODING_H
#define KARMA_COUPLING_FIELD_ENTRY_ENCODING_H

#include "karma_field_identity.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace KARMA
{
/** Canonical byte encoding of the existing `FieldEntryKey` identity.
 *
 * This does not introduce a new identity. `FieldEntryKey` already wraps both
 * supported meanings (MOAB global identities and `DofKey` algebraic entries);
 * what was missing was one endian-stable spelling of it that every new
 * contract can share. New subsystems - semantic-model hashing, C ABI field
 * tables, and graph checkpoint metadata - must use this encoding rather than
 * inventing another one.
 *
 * The existing transfer-map and safe-point checkpoint formats keep their own
 * schema versions and their own word layouts; migrating them onto this
 * encoding is a separate, separately approved change.
 *
 * Layout, little-endian throughout, fixed width:
 *
 * | offset | width | field |
 * | --- | --- | --- |
 * | 0 | 4 | payload length in bytes, excluding this prefix |
 * | 4 | 1 | encoding version |
 * | 5 | 1 | kind discriminant: 0 MoabGlobalId, 1 Dof |
 * | 6 | 8 | global id |
 * | 14 | 4 | component ordinal |
 * | 18 | 8 * DofKey::width | DoF words, variable-major, component-major, basis-minor |
 *
 * Both kinds encode every field. A MOAB key carries zeroed DoF words rather
 * than a shorter record, so the encoded size is a constant and a reader never
 * has to branch on kind to find the end of a record.
 */
inline constexpr std::uint8_t field_entry_encoding_version = 1;

inline constexpr std::size_t field_entry_key_encoded_size = 4 + 1 + 1 + 8 + 4 + 8 * DofKey::width;

enum class FieldEntryDecodeError : std::uint8_t
{
    None = 0,
    Truncated = 1,
    LengthMismatch = 2,
    UnsupportedVersion = 3,
    UnknownKind = 4,
    TrailingBytes = 5
};

struct FieldEntryDecodeResult
{
    std::optional<FieldEntryKey> key;
    FieldEntryDecodeError error = FieldEntryDecodeError::None;
};

[[nodiscard]] std::vector<std::byte> encode_field_entry_key(const FieldEntryKey& key);

/** Write into a caller-owned buffer of at least `field_entry_key_encoded_size`
 * bytes. Returns false and writes nothing when the buffer is too small, so an
 * allocation-free hot path can preallocate once during setup.
 */
[[nodiscard]] bool encode_field_entry_key(const FieldEntryKey& key, std::span<std::byte> buffer) noexcept;

/** Pure local decode with explicit reason codes; never throws. */
[[nodiscard]] FieldEntryDecodeResult decode_field_entry_key(std::span<const std::byte> bytes) noexcept;

/** FNV-1a over the canonical encoding. Stable across ranks and platforms, and
 * therefore usable as a distributed lookup key. Not a cryptographic digest.
 */
[[nodiscard]] std::uint64_t field_entry_key_hash(const FieldEntryKey& key) noexcept;
} // namespace KARMA

#endif
