#include "fuse/base/bytes.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace fuse;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> of(std::initializer_list<unsigned> values)
{
    std::vector<std::byte> out;
    for (const auto value : values) out.push_back(static_cast<std::byte>(value));
    return out;
}

void integers_are_little_endian_and_fixed_width()
{
    std::vector<std::byte> out;
    bytes::put_integer(out, 0x0102030405060708ULL, 8);
    require(out == of({0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01}),
            "put_integer is not little-endian");

    // Width is what the caller asked for, not what the value needs.
    std::vector<std::byte> narrow;
    bytes::put_integer(narrow, 0xffffULL, 1);
    require(narrow.size() == 1 && narrow[0] == std::byte{0xff}, "put_integer ignored its width");

    std::vector<std::byte> zero;
    bytes::put_integer(zero, 0, 4);
    require(zero.size() == 4, "a zero value still occupies its full width");
}

void signed_values_use_two_s_complement()
{
    std::vector<std::byte> negative, positive;
    bytes::put_signed(negative, -1, 8);
    bytes::put_signed(positive, 1, 8);
    require(negative.size() == positive.size(), "sign changed the encoded width");
    require(negative == of({0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}),
            "put_signed is not a two's-complement bit pattern");

    // The extremes must round-trip through the width without narrowing.
    std::vector<std::byte> lowest;
    bytes::put_signed(lowest, INT64_MIN, 8);
    require(lowest == of({0, 0, 0, 0, 0, 0, 0, 0x80}), "INT64_MIN did not encode exactly");
}

// The rule this file exists for.
void the_length_prefix_prevents_adjacent_collisions()
{
    std::vector<std::byte> left, right;
    bytes::put_string(left, "ab");
    bytes::put_string(left, "c");
    bytes::put_string(right, "a");
    bytes::put_string(right, "bc");
    require(left != right, "two different string splits encoded to the same bytes");

    // And the same for byte sequences, which share the rule.
    const auto first = of({1, 2});
    const auto second = of({3});
    const auto first_short = of({1});
    const auto second_long = of({2, 3});
    std::vector<std::byte> a, b;
    bytes::put_bytes(a, first);
    bytes::put_bytes(a, second);
    bytes::put_bytes(b, first_short);
    bytes::put_bytes(b, second_long);
    require(a != b, "two different byte splits encoded to the same bytes");

    // An empty string is still a value and must occupy its prefix, so that
    // {"", "a"} and {"a"} differ.
    std::vector<std::byte> empty_then, single;
    bytes::put_string(empty_then, "");
    bytes::put_string(empty_then, "a");
    bytes::put_string(single, "a");
    require(empty_then != single, "an empty string wrote nothing at all");

    // A string and a byte sequence with the same contents agree, because the
    // two encoders must not disagree about a shared rule.
    std::vector<std::byte> as_string, as_bytes;
    bytes::put_string(as_string, "hi");
    bytes::put_bytes(as_bytes, of({'h', 'i'}));
    require(as_string == as_bytes, "put_string and put_bytes disagree");
}

void rationals_encode_both_terms()
{
    struct Pair { std::int64_t numerator; std::int64_t denominator; };
    std::vector<std::byte> one_half, two_quarters;
    bytes::put_rational(one_half, Pair{1, 2});
    bytes::put_rational(two_quarters, Pair{2, 4});
    require(one_half.size() == 16, "a rational is not two 8-byte terms");
    // Unreduced values are NOT equal byte strings: reduction is the caller's
    // job, and the encoder must not quietly perform it.
    require(one_half != two_quarters, "the encoder reduced a rational behind the caller's back");

    std::vector<std::byte> negative_numerator, negative_denominator;
    bytes::put_rational(negative_numerator, Pair{-1, 2});
    bytes::put_rational(negative_denominator, Pair{1, -2});
    require(negative_numerator != negative_denominator, "sign position was lost");
}

void the_hash_depends_on_every_byte_and_on_order()
{
    const auto base = of({1, 2, 3, 4});
    const auto reordered = of({4, 3, 2, 1});
    const auto truncated = of({1, 2, 3});
    require(bytes::fnv1a(base) != bytes::fnv1a(reordered), "the hash ignores byte order");
    require(bytes::fnv1a(base) != bytes::fnv1a(truncated), "the hash ignores trailing bytes");
    require(bytes::fnv1a(base) == bytes::fnv1a(base), "the hash is not deterministic");

    // The published FNV-1a offset basis, which is what an empty input returns.
    require(bytes::fnv1a({}) == 14695981039346656037ULL, "the FNV-1a offset basis is wrong");

    // One published vector, so a transcribed constant cannot drift unnoticed.
    const auto a = of({'a'});
    require(bytes::fnv1a(a) == 0xaf63dc4c8601ec8cULL, "FNV-1a of \"a\" does not match the published value");
}
}

int main()
{
    try
    {
        integers_are_little_endian_and_fixed_width();
        signed_values_use_two_s_complement();
        the_length_prefix_prevents_adjacent_collisions();
        rationals_encode_both_terms();
        the_hash_depends_on_every_byte_and_on_order();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
