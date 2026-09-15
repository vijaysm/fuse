#include "karma_semantic_version.h"

#include <array>
#include <charconv>
#include <system_error>

namespace KARMA
{
VersionParseResult parse_semantic_version(std::string_view text) noexcept
{
    std::array<std::uint32_t, 3> words{};
    for (std::size_t index = 0; index < words.size(); ++index)
    {
        const auto separator = text.find('.');
        const bool last = index == words.size() - 1;
        if (last != (separator == std::string_view::npos)) return {{}, VersionParseError::Malformed};
        const auto word = text.substr(0, separator);
        if (word.empty() || (word.size() > 1 && word.front() == '0'))
            return {{}, VersionParseError::Malformed};
        for (const char digit : word)
            if (digit < '0' || digit > '9') return {{}, VersionParseError::Malformed};
        const auto converted = std::from_chars(word.data(), word.data() + word.size(), words[index]);
        if (converted.ec == std::errc::result_out_of_range) return {{}, VersionParseError::Overflow};
        if (converted.ec != std::errc{} || converted.ptr != word.data() + word.size())
            return {{}, VersionParseError::Malformed};
        if (!last) text.remove_prefix(separator + 1);
    }
    return {SemanticVersion{words[0], words[1], words[2]}, VersionParseError::None};
}

std::string serialize_semantic_version(SemanticVersion version)
{
    // Integer to_chars is independent of the process locale.
    std::string result;
    result.reserve(32);
    for (const auto word : {version.major, version.minor, version.patch})
    {
        if (!result.empty()) result.push_back('.');
        std::array<char, 10> digits{}; // UINT32_MAX has exactly ten decimal digits.
        const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), word);
        result.append(digits.data(), converted.ptr);
    }
    return result;
}

std::optional<std::string_view> version_domain_name(VersionDomain domain) noexcept
{
    switch (domain)
    {
    case VersionDomain::SemanticModel: return "semantic-model";
    case VersionDomain::GraphSchema: return "graph-schema";
    case VersionDomain::CAbi: return "c-abi";
    case VersionDomain::PluginConfiguration: return "plugin-configuration";
    case VersionDomain::TransferMap: return "transfer-map";
    case VersionDomain::ComponentState: return "component-state";
    case VersionDomain::GraphCheckpoint: return "graph-checkpoint";
    }
    return std::nullopt;
}

bool CompatibilityDecision::compatible() const noexcept
{
    switch (reason)
    {
    case CompatibilityReason::Exact:
    case CompatibilityReason::OlderPatch:
    case CompatibilityReason::OlderMinor:
    case CompatibilityReason::OptionalMinorTail: return true;
    default: return false;
    }
}

CompatibilityDecision check_version_compatibility(const VersionContract& contract, const VersionReader& reader) noexcept
{
    if (!version_domain_name(contract.domain) || !version_domain_name(reader.domain))
        return {CompatibilityReason::InvalidDomain, 0};
    if (contract.domain != reader.domain) return {CompatibilityReason::DomainMismatch, 0};
    if (contract.version.major != reader.version.major) return {CompatibilityReason::MajorMismatch, 0};
    const auto missing = contract.required_capabilities & ~reader.supported_capabilities;
    if (missing != 0) return {CompatibilityReason::MissingRequiredCapabilities, missing};
    if (contract.version.minor > reader.version.minor)
        return {reader.allow_optional_minor_tail ? CompatibilityReason::OptionalMinorTail :
                                                  CompatibilityReason::NewerMinor, 0};
    if (contract.version.minor < reader.version.minor) return {CompatibilityReason::OlderMinor, 0};
    if (contract.version.patch > reader.version.patch) return {CompatibilityReason::PatchTooNew, 0};
    if (contract.version.patch < reader.version.patch) return {CompatibilityReason::OlderPatch, 0};
    return {CompatibilityReason::Exact, 0};
}
} // namespace KARMA
