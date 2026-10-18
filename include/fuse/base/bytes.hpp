#pragma once

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
 * connection it should reject. `put_string` and `put_bytes` therefore always
 * write a 4-byte count first, even where the calling record happens to have
 * only one variable-length field today and no collision is constructible.
 * That case is a property of the record, not of the encoder, and it changes
 * the moment someone adds a second string.
 *
 * Numbers are little-endian and fixed width, so the bytes do not depend on
 * the host. Signed values are written as their two's-complement bit pattern
 * rather than with a sign character, so the width is the same for every value.
 * `get_integer`/`get_signed` are the decode side of `put_integer`/
 * `put_signed`: both operate on the VALUE through shifts, never on a value's
 * native memory layout, so neither direction depends on the host's own
 * endianness.
 *
 * Dependency-free on purpose: this header pulls in no FUSE type, so the
 * dependency-free encoders can use it without acquiring one.
 *
 * **Now used by the graph descriptor too (D62, descriptor kGraphVersion
 * 15).** Before kGraphVersion 15, the coupling-graph descriptor
 * (`fuse::serialize`/`fuse::deserialize`, fuse/model/descriptor.hpp -- "the
 * ABI narrow waist") did not route through this header at all:
 * `detail::Writer::pod`/`detail::Reader::pod` `memcpy`'d a value's native
 * in-memory representation directly, which was little-endian on every
 * machine this library has ever built for but was NOT the little-endian
 * encoding this header defines -- two canonical-byte encoders in the same
 * layer, with different endianness rules, invisible on every common-endian
 * host (BM-5, review 2026-10-11). A big-endian host or adapter would have
 * written a descriptor the golden files under test/golden/ (`waist-v*.bin`,
 * pinned to little-endian bytes only because every machine that has ever
 * produced them was little-endian), and every little-endian peer, would
 * have refused to read. kGraphVersion 15 (KNOWN_ISSUES.md D62, docs/dev/
 * abi-design.md's "The graph descriptor's byte order") routes `pod()`
 * through `put_integer()`/`get_integer()` (an unsigned POD),
 * `put_signed()`/`get_signed()` (a signed POD), or their bit pattern
 * through the same pair (a floating-point POD) -- the version bump absorbs
 * the resulting golden-file churn deliberately; every pre-15 golden file is
 * kept and refused by `kGraphVersion`'s own version check, same as every
 * earlier version bump.
 */
namespace fuse::bytes
{
/** Little-endian, fixed width. Bytes above `width` are discarded, so the
 * caller picks a width that holds the value. */
inline void put_integer(std::vector<std::byte>& out, std::uint64_t value, std::size_t width)
{
    for (std::size_t index = 0; index != width; ++index)
        out.push_back(static_cast<std::byte>((value >> (8 * index)) & 0xffU));
}

/** The decode side of `put_integer`: little-endian, fixed width, operating
 * on the VALUE through shifts rather than on a value's native memory
 * layout -- the same reason `put_integer`'s bytes do not depend on the
 * host's own endianness, decoding them does not either. `ok` is set to
 * false, and `pos` is left unmoved, if [pos, pos+width) runs past `buf`'s
 * end; the caller checks `ok` once after the record this value belongs to,
 * the same convention `fuse::detail::Reader` (fuse/model/descriptor.hpp)
 * already uses for every other truncation it detects. `pos` is advanced by
 * exactly `width` only on success, so a caller can retry nothing and must
 * instead treat `ok == false` as terminal for the whole decode, matching
 * `Reader::pod()`'s own existing contract. */
inline std::uint64_t get_integer(std::span<const std::byte> buf, std::size_t& pos,
                                 std::size_t width, bool& ok)
{
    if (pos + width > buf.size()) { ok = false; return 0; }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index != width; ++index)
        value |= static_cast<std::uint64_t>(buf[pos + index]) << (8 * index);
    pos += width;
    return value;
}

/** Two's-complement bit pattern, so every value of a given type is the same
 * width and no sign character has to be parsed back out. */
inline void put_signed(std::vector<std::byte>& out, std::int64_t value, std::size_t width)
{
    put_integer(out, static_cast<std::uint64_t>(value), width);
}

/** The decode side of `put_signed`: reads `width` bytes as `get_integer`
 * does, then sign-extends from the two's-complement bit at the TOP of that
 * width -- the exact inverse of `put_signed`'s own truncation to `width`
 * bytes of the 64-bit two's-complement pattern. `width == 8` needs no
 * extension (the value already occupies every bit `std::int64_t` has). */
inline std::int64_t get_signed(std::span<const std::byte> buf, std::size_t& pos,
                               std::size_t width, bool& ok)
{
    const std::uint64_t raw = get_integer(buf, pos, width, ok);
    if (!ok || width >= 8) return static_cast<std::int64_t>(raw);
    const std::uint64_t sign_bit = std::uint64_t{1} << (8 * width - 1);
    if ((raw & sign_bit) == 0) return static_cast<std::int64_t>(raw);
    const std::uint64_t sign_extended = raw | (~std::uint64_t{0} << (8 * width));
    return static_cast<std::int64_t>(sign_extended);
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

/** fuse::Rational spells its terms num()/den(); encode it as its ABI record so
 * the bytes are the same whichever side of the ABI produced them. Declared
 * here as a template on the accessor shape so this header still names no
 * FUSE type (it stays usable by the dependency-free encoders). */
template <class RationalType>
    requires requires(const RationalType& r) { r.num(); r.den(); }
void put_rational(std::vector<std::byte>& out, const RationalType& value)
{
    put_signed(out, value.num(), 8);
    put_signed(out, value.den(), 8);
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
} // namespace fuse::bytes

