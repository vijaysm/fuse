#ifndef KARMA_COUPLING_SEMANTIC_VERSION_H
#define KARMA_COUPLING_SEMANTIC_VERSION_H

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace KARMA
{
/** Numeric contract version; prerelease/build suffixes are intentionally unsupported.
 * This evolving C++ API does not declare or freeze any C ABI or persisted schema.
 */
struct SemanticVersion
{
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    auto operator<=>(const SemanticVersion&) const = default;
};

enum class VersionParseError { None, Malformed, Overflow };

struct VersionParseResult
{
    std::optional<SemanticVersion> version;
    VersionParseError error = VersionParseError::None;
};

/** Accept exactly three unsigned decimal words, without leading zeros or whitespace. */
[[nodiscard]] VersionParseResult parse_semantic_version(std::string_view text) noexcept;
[[nodiscard]] std::string serialize_semantic_version(SemanticVersion version);

enum class VersionDomain : std::uint8_t
{
    SemanticModel = 0,
    GraphSchema = 1,
    CAbi = 2,
    PluginConfiguration = 3,
    TransferMap = 4,
    ComponentState = 5,
    GraphCheckpoint = 6
};

[[nodiscard]] std::optional<std::string_view> version_domain_name(VersionDomain domain) noexcept;

struct VersionContract
{
    VersionDomain domain = VersionDomain::SemanticModel;
    SemanticVersion version;
    std::uint64_t required_capabilities = 0;
};

/** Reader understands all older minors within its major (including major zero).
 * A contract's patch is a minimum reader patch within the same minor. A newer
 * minor may be consumed only when its owning format guarantees a usable prefix
 * and skippable optional tails. This check cannot validate table sizes or bytes.
 */
struct VersionReader
{
    VersionDomain domain = VersionDomain::SemanticModel;
    SemanticVersion version;
    std::uint64_t supported_capabilities = 0;
    bool allow_optional_minor_tail = false;
};

enum class CompatibilityReason : std::uint8_t
{
    Exact = 0,
    OlderPatch = 1,
    OlderMinor = 2,
    OptionalMinorTail = 3,
    InvalidDomain = 4,
    DomainMismatch = 5,
    MajorMismatch = 6,
    MissingRequiredCapabilities = 7,
    NewerMinor = 8,
    PatchTooNew = 9
};

struct CompatibilityDecision
{
    CompatibilityReason reason = CompatibilityReason::InvalidDomain;
    std::uint64_t missing_capabilities = 0;
    [[nodiscard]] bool compatible() const noexcept;
};

/** Pure local negotiation; callers reconcile decisions before collective work.
 * Unknown optional metadata is neither inspected nor consumed here; the owning
 * format must preserve it where allowed. No version number implies capability.
 */
[[nodiscard]] CompatibilityDecision check_version_compatibility(
    const VersionContract& contract, const VersionReader& reader) noexcept;
} // namespace KARMA

#endif
