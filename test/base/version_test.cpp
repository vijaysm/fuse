#include <gtest/gtest.h>

#include "fuse/base/version.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace
{
using namespace fuse;

// Upstream's require() threw on the first failure; EXPECT_TRUE reports every
// failure in the function and the message travels with it unchanged.
void require(bool condition, const char* message) { EXPECT_TRUE(condition) << message; }

void parsing()
{
    for (const std::string_view input : {"0.0.0", "1.2.3", "4294967295.4294967295.4294967295"})
    {
        const auto parsed = parse_semantic_version(input);
        require(parsed.version.has_value() && parsed.error == VersionParseError::None, "valid version rejected");
        require(serialize_semantic_version(*parsed.version) == input, "version did not round-trip canonically");
    }
    for (const std::string_view input : {"", "1", "1.2", "1.2.3.4", "1..3", ".2.3", "1.2.",
                                         "-1.2.3", "+1.2.3", " 1.2.3", "1.2.3\n", "01.2.3",
                                         "1.02.3", "1.2.03", "1.2.3-rc1", "1.2.3+build", "a.2.3"})
    {
        const auto parsed = parse_semantic_version(input);
        require(!parsed.version && parsed.error == VersionParseError::Malformed, "malformed version accepted");
    }
    const std::string embedded_nul("1.2.3\0x", 7);
    require(!parse_semantic_version(embedded_nul).version, "embedded NUL accepted");
    for (const std::string_view input : {"4294967296.0.0", "0.4294967296.0", "0.0.4294967296",
                                         "999999999999999999999999999999.0.0"})
    {
        const auto parsed = parse_semantic_version(input);
        require(!parsed.version && parsed.error == VersionParseError::Overflow, "overflow accepted or misclassified");
    }
    // Deterministic full-width values exercise decimal boundaries without a random library dependency.
    std::uint32_t state = 1;
    for (int index = 0; index < 1000; ++index)
    {
        state = state * 1664525U + 1013904223U;
        const SemanticVersion version{state, state / 10U, state / 100U};
        require(parse_semantic_version(serialize_semantic_version(version)).version == version,
                "full-width version round-trip failed");
    }
    require(SemanticVersion{1, 99, 99} < SemanticVersion{2, 0, 0}, "major comparison is inverted");
    require(SemanticVersion{1, 2, 99} < SemanticVersion{1, 3, 0}, "minor comparison is inverted");
    require(SemanticVersion{1, 2, 3} < SemanticVersion{1, 2, 4}, "patch comparison is inverted");
}

void compatibility()
{
    constexpr std::array domains{VersionDomain::SemanticModel, VersionDomain::GraphSchema, VersionDomain::CAbi,
        VersionDomain::PluginConfiguration, VersionDomain::TransferMap, VersionDomain::ComponentState,
        VersionDomain::GraphCheckpoint};
    constexpr std::array<std::string_view, 7> names{"semantic-model", "graph-schema", "c-abi",
        "plugin-configuration", "transfer-map", "component-state", "graph-checkpoint"};
    for (std::size_t index = 0; index < domains.size(); ++index)
    {
        require(version_domain_name(domains[index]) == names[index], "version domain name changed");
        const VersionContract contract{domains[index], {1, 2, 3}, 1};
        const VersionReader reader{domains[index], {1, 2, 3}, 3, false};
        const auto exact = check_version_compatibility(contract, reader);
        require(exact.compatible() && exact.reason == CompatibilityReason::Exact &&
                exact.missing_capabilities == 0, "exact contract rejected");
        for (const auto other : domains)
            if (other != domains[index])
                require(check_version_compatibility(contract, {other, {1, 2, 3}, 3, false}).reason ==
                        CompatibilityReason::DomainMismatch, "independent version domains conflated");
    }
    const VersionReader reader{VersionDomain::CAbi, {1, 2, 3}, 3, false};
    const auto decide = [&](SemanticVersion version, std::uint64_t bits = 1) {
        return check_version_compatibility({VersionDomain::CAbi, version, bits}, reader);
    };
    require(decide({2, 0, 0}).reason == CompatibilityReason::MajorMismatch, "new major accepted");
    require(decide({0, 9, 0}).reason == CompatibilityReason::MajorMismatch, "old major accepted");
    require(decide({1, 1, 99}).reason == CompatibilityReason::OlderMinor && decide({1, 1, 99}).compatible(),
            "older minor prefix rejected");
    require(decide({1, 2, 2}).reason == CompatibilityReason::OlderPatch && decide({1, 2, 2}).compatible(),
            "older patch rejected");
    require(decide({1, 2, 4}).reason == CompatibilityReason::PatchTooNew && !decide({1, 2, 4}).compatible(),
            "required patch silently downgraded");
    require(decide({1, 3, 0}).reason == CompatibilityReason::NewerMinor && !decide({1, 3, 0}).compatible(),
            "newer minor accepted without explicit policy");
    auto tails = reader;
    tails.allow_optional_minor_tail = true;
    const auto future = check_version_compatibility({VersionDomain::CAbi, {1, 3, 0}, 1}, tails);
    require(future.compatible() && future.reason == CompatibilityReason::OptionalMinorTail,
            "explicit minor tail extension rejected");
    const auto missing = decide({1, 2, 3}, 0x8000000000000005ULL);
    require(!missing.compatible() && missing.reason == CompatibilityReason::MissingRequiredCapabilities &&
            missing.missing_capabilities == 0x8000000000000004ULL, "unknown required bits silently accepted");
    require(!check_version_compatibility({VersionDomain::CAbi, {1, 3, 0}, 4}, tails).compatible(),
            "minor tail policy bypassed required capabilities");
    const auto invalid = static_cast<VersionDomain>(255);
    require(!version_domain_name(invalid).has_value(), "invalid domain acquired a name");
    require(check_version_compatibility({invalid, {1, 2, 3}, 0}, reader).reason ==
            CompatibilityReason::InvalidDomain, "invalid contract domain accepted");
    require(check_version_compatibility({invalid, {1, 2, 3}, 0}, {invalid, {1, 2, 3}, 0, false}).reason ==
            CompatibilityReason::InvalidDomain, "matching invalid domains accepted");
}
}

TEST(SemanticVersion, Parsing) { parsing(); }
TEST(SemanticVersion, Compatibility) { compatibility(); }
