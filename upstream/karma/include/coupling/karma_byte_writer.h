#ifndef KARMA_COUPLING_BYTE_WRITER_H
#define KARMA_COUPLING_BYTE_WRITER_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

/** The canonical-byte rules, in one place.
 *
 * Every canonical encoding in the coupling layer - the field-entry key, the
 * semantic model, the operator chain - has to agree on how an integer, a
 * string and a rational become bytes, because the bytes are compared across
 * ranks and hashed into identities. They agreed by being written three times,
 * which is a different thing from agreeing by construction.
 *
 * **The length prefix is the rule that matters.** A key written immediately
 * before its value concatenates into exactly the same bytes as a different
 * key/value split of the same characters: `{"ab", "c"}` and `{"a", "bc"}` are
 * indistinguishable without it. Two records with different meaning would then
 * hash equal, and a compatibility check comparing hashes would admit a
 * connection it should refuse. `put_string` and `put_bytes` therefore always
 * write a 4-byte count first, even where the calling record happens to have
 * only one variable-length field today and no collision is constructible.
 * That case is a property of the record, not of the encoder, and it changes
 * the moment someone adds a second string.
 *
 * Numbers are little-endian and fixed width, so the bytes do not depend on
 * the host. Signed values are written as their two's-complement bit pattern
 * rather than with a sign character, so the width is the same for every value.
 *
 * Dependency-free on purpose: this header pulls in no KARMA type, so the
 * dependency-free encoders can use it without acquiring one.
 */
namespace KARMA::bytes
{
/** Little-endian, fixed width. Bytes above `width` are discarded, so the
 * caller picks a width that holds the value. */
inline void put_integer(std::vector<std::byte>& out, std::uint64_t value, std::size_t width)
{
    for (std::size_t index = 0; index != width; ++index)
        out.push_back(static_cast<std::byte>((value >> (8 * index)) & 0xffU));
}

/** Two's-complement bit pattern, so every value of a given type is the same
 * width and no sign character has to be parsed back out. */
inline void put_signed(std::vector<std::byte>& out, std::int64_t value, std::size_t width)
{
    put_integer(out, static_cast<std::uint64_t>(value), width);
}

/** Length-prefixed; see the header comment for why that is not optional. */
inline void put_bytes(std::vector<std::byte>& out, std::span<const std::byte> value)
{
    put_integer(out, value.size(), 4);
    out.insert(out.end(), value.begin(), value.end());
}

/** Length-prefixed; see the header comment for why that is not optional. */
inline void put_string(std::vector<std::byte>& out, std::string_view value)
{
    put_integer(out, value.size(), 4);
    for (const char character : value)
        out.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
}

/** Numerator then denominator, eight bytes each. Templated only so this
 * header stays free of the semantic model; any type with those two integer
 * members encodes identically. */
template <class RationalLike>
void put_rational(std::vector<std::byte>& out, const RationalLike& value)
{
    put_signed(out, value.numerator, 8);
    put_signed(out, value.denominator, 8);
}

/** FNV-1a, 64-bit. Not a cryptographic hash and not used as one: it turns a
 * canonical byte string into a comparable identity, and the canonical bytes
 * remain the authority whenever two identities collide.
 *
 * The offset basis is 14695981039346656037, the published FNV-1a value, and
 * `byte_writer_test` checks it against a published vector. The three coupling
 * encoders previously each carried 1469598103934665603 - the same digits with
 * the last one dropped in transcription - which is a perfectly serviceable
 * multiplicative hash but is NOT FNV-1a. That distinction was harmless while
 * the only consumer was this library comparing identities with itself, and
 * stops being harmless the moment a Fortran or Python adapter is expected to
 * compute the same identity for the same bytes: it would use a real FNV-1a,
 * get a different number, and the two would disagree about identity with
 * nothing to indicate why.
 *
 * The four pre-existing copies in the I/O and checkpoint sources keep the old
 * constant deliberately. Their hashes are WRITTEN INTO FILES, so correcting
 * them would invalidate every existing checkpoint; they carry their own schema
 * versions and are a separate migration. None of the three encoders here
 * persists its hash. */
inline std::uint64_t fnv1a(std::span<const std::byte> data) noexcept
{
    constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset_basis;
    for (const auto byte : data)
    {
        hash ^= std::to_integer<std::uint64_t>(byte);
        hash *= prime;
    }
    return hash;
}
} // namespace KARMA::bytes

#endif
