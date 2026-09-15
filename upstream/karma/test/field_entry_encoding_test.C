// Serial encoding contract for the existing FieldEntryKey identity.
//
// This test deliberately never calls PetscInitialize: the encoding is pure
// byte manipulation over a plain-old-data key, and keeping it free of MPI
// makes it runnable wherever the library builds.
#include "karma_field_entry_encoding.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
// Pinned on first green run; a change here is a change of hash function.
// Independently derived: the encoding is reproduced outside this library and
// hashed with a published FNV-1a, rather than copied out of a passing run.
// It changed from 0xbdc955cfa93397c3 when the offset basis was corrected -
// the three coupling encoders had been carrying the published basis with its
// last digit dropped, which is not FNV-1a and which no external implementation
// would reproduce.
constexpr std::uint64_t expected_moab_hash = 0xe7f96370571789f1ULL;
}

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

FieldEntryKey moab_key(std::uint64_t global_id, unsigned component)
{
    FieldEntryKey key;
    key.kind = FieldEntryKey::Kind::MoabGlobalId;
    key.global_id = global_id;
    key.component = component;
    return key;
}

FieldEntryKey dof_key(std::uint64_t variable, std::uint64_t component, std::uint64_t basis)
{
    FieldEntryKey key;
    key.kind = FieldEntryKey::Kind::Dof;
    // Variable-major, component-major, basis-minor, matching KARMADofMap.
    key.dof[3] = variable;
    key.dof[5] = component;
    key.dof[6] = basis;
    return key;
}

void round_trip()
{
    std::vector<FieldEntryKey> keys{
        moab_key(0, 0),
        moab_key(7, 1),
        moab_key(std::numeric_limits<std::uint64_t>::max(), 4294967295U),
        dof_key(0, 0, 0),
        dof_key(1, 2, 11),
        dof_key(std::numeric_limits<std::uint64_t>::max(), 0, 0),
    };
    // A cG subentity key with every word populated exercises the full payload.
    FieldEntryKey saturated;
    saturated.kind = FieldEntryKey::Kind::Dof;
    saturated.global_id = 12345;
    saturated.component = 3;
    for (std::size_t index = 0; index != DofKey::width; ++index)
        saturated.dof[index] = 0x0123456789abcdefULL + index;
    keys.push_back(saturated);

    for (const auto& key : keys)
    {
        const auto bytes = encode_field_entry_key(key);
        require(bytes.size() == field_entry_key_encoded_size,
                "encoded field entry key has a nonconstant size");
        const auto decoded = decode_field_entry_key(bytes);
        require(decoded.error == FieldEntryDecodeError::None && decoded.key.has_value(),
                "round-trip decode rejected its own encoding");
        require(*decoded.key == key, "round-trip decode changed the key");
    }
}

void canonical_bytes()
{
    // Pinned little-endian layout. Changing these bytes is a schema change.
    const auto bytes = encode_field_entry_key(moab_key(0x0102030405060708ULL, 0x0a0b0c0dU));
    require(bytes.size() == field_entry_key_encoded_size, "canonical fixture size changed");
    const auto byte_at = [&](std::size_t index) { return static_cast<unsigned>(bytes[index]); };
    // payload length prefix, little-endian uint32
    const unsigned payload = field_entry_key_encoded_size - 4;
    require(byte_at(0) == (payload & 0xffU) && byte_at(1) == ((payload >> 8) & 0xffU) &&
            byte_at(2) == 0 && byte_at(3) == 0, "length prefix is not little-endian uint32");
    require(byte_at(4) == field_entry_encoding_version, "encoding version byte moved");
    require(byte_at(5) == 0, "MoabGlobalId kind discriminant changed");
    const std::array<unsigned, 8> global_id{0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
    for (std::size_t index = 0; index != global_id.size(); ++index)
        require(byte_at(6 + index) == global_id[index], "global id is not little-endian");
    const std::array<unsigned, 4> component{0x0d, 0x0c, 0x0b, 0x0a};
    for (std::size_t index = 0; index != component.size(); ++index)
        require(byte_at(14 + index) == component[index], "component is not little-endian");
    for (std::size_t index = 18; index != field_entry_key_encoded_size; ++index)
        require(byte_at(index) == 0, "unused DoF words are not zero filled");
    require(encode_field_entry_key(dof_key(0, 0, 0))[5] == std::byte{1},
            "Dof kind discriminant changed");
}

void distinct_encodings()
{
    // Kind is part of identity: a zero MOAB key and a zero DoF key differ.
    require(encode_field_entry_key(moab_key(0, 0)) != encode_field_entry_key(dof_key(0, 0, 0)),
            "kind is not encoded");
    require(encode_field_entry_key(moab_key(7, 0)) != encode_field_entry_key(moab_key(7, 1)),
            "component is not encoded");
    require(encode_field_entry_key(dof_key(1, 0, 0)) != encode_field_entry_key(dof_key(0, 1, 0)),
            "DoF word position is not encoded");
}

void malformed_input()
{
    const auto valid = encode_field_entry_key(dof_key(1, 2, 3));

    const auto reject = [](std::span<const std::byte> bytes, FieldEntryDecodeError expected, const char* message) {
        const auto decoded = decode_field_entry_key(bytes);
        require(!decoded.key.has_value() && decoded.error == expected, message);
    };

    reject({}, FieldEntryDecodeError::Truncated, "empty input accepted");
    reject(std::span(valid).first(3), FieldEntryDecodeError::Truncated, "short length prefix accepted");
    reject(std::span(valid).first(valid.size() - 1), FieldEntryDecodeError::Truncated,
           "truncated payload accepted");

    auto trailing = valid;
    trailing.push_back(std::byte{0});
    reject(trailing, FieldEntryDecodeError::TrailingBytes, "trailing bytes accepted");

    auto wrong_length = valid;
    wrong_length[0] = std::byte{7};
    reject(wrong_length, FieldEntryDecodeError::LengthMismatch, "declared length mismatch accepted");

    auto wrong_version = valid;
    wrong_version[4] = std::byte{field_entry_encoding_version + 1};
    reject(wrong_version, FieldEntryDecodeError::UnsupportedVersion, "unknown encoding version accepted");

    auto wrong_kind = valid;
    wrong_kind[5] = std::byte{2};
    reject(wrong_kind, FieldEntryDecodeError::UnknownKind, "unknown key kind accepted");
}

void hashing()
{
    require(field_entry_key_hash(dof_key(1, 2, 3)) == field_entry_key_hash(dof_key(1, 2, 3)),
            "hash is not deterministic");
    require(field_entry_key_hash(moab_key(7, 0)) != field_entry_key_hash(moab_key(7, 1)),
            "hash ignores the component");
    require(field_entry_key_hash(moab_key(0, 0)) != field_entry_key_hash(dof_key(0, 0, 0)),
            "hash ignores the kind");
    // Pinned so a change of hash function is visible rather than silent.
    require(field_entry_key_hash(moab_key(0x0102030405060708ULL, 0x0a0b0c0dU)) == expected_moab_hash,
            "field entry hash function changed");
}
}

int main()
{
    try
    {
        round_trip();
        canonical_bytes();
        distinct_encodings();
        malformed_input();
        hashing();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
