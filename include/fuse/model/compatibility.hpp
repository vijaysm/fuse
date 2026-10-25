#pragma once


#include "fuse/model/semantics.hpp"
#include "fuse/model/transform.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/** Whether a connection between two semantic fields can be compiled, and into
 * what.
 *
 * Stages are evaluated in a fixed order and every stage runs, so one call
 * reports every problem rather than the first. The verdict names the worst
 * outcome, and `first_incompatible_stage` names the earliest stage that
 * refused, which is the one a user should fix first. Required capabilities are
 * collected even when an earlier stage already refused, because a caller
 * fixing units should not then discover a missing capability on the next
 * attempt.
 */
namespace fuse
{
enum class ConnectionStage : std::uint8_t
{
    Identity = 0,
    Unit = 1,
    Shape = 2,
    Location = 3,
    Temporal = 4,
    Conservation = 5,
    Capability = 6
};

[[nodiscard]] std::string_view connection_stage_name(ConnectionStage stage) noexcept;

enum class ConnectionVerdict : std::uint8_t
{
    /** Compiles to an empty chain: the fields already agree. */
    Exact = 0,
    /** Compiles, and no operator discards or invents information. */
    LosslessExplicit = 1,
    /** Compiles, and at least one operator loses information; every such
     * operator was explicitly requested. */
    LossyExplicit = 2,
    /** An operator would be needed and was not requested, or two operators
     * could satisfy the same gap and the choice is the caller's. */
    Ambiguous = 3,
    /** No operator chain can express the connection. */
    Unsupported = 4
};

/** Stable numeric reason codes. Values are never reused or renumbered: a
 * diagnostic that changes meaning between releases is worse than none.
 *
 * Named `ConnectionReason` rather than `CompatibilityReason` because
 * `fuse/base/version.hpp` already owns that name for version negotiation.
 * The two layers answer different questions - may this reader accept this
 * contract, versus may these two fields be connected - and collapsing them
 * into one name would force one of them to be renamed later. */
enum class ConnectionReason : std::uint16_t
{
    None = 0,
    EmptyFieldIdentifier = 1,
    RequirementVersionUnsatisfied = 2,
    RequirementFieldMismatch = 3,
    DimensionMismatch = 4,
    UnitScaleNotPermitted = 5,
    AffineConversionNotRequested = 6,
    UnitConversionOverflow = 7,
    /** At least one side's unit literal could not be interpreted, so the
     * two cannot be shown to agree. Not the same as a dimension mismatch:
     * the dimensions are unknown rather than known and different. */
    UnitsUndeclared = 30,
    /** The pure-scale conversion factor is negative -- a
     * sign flip, not "a change of spelling" -- and was not explicitly
     * requested. Distinct from UnitScaleNotPermitted so a caller can tell
     * "automatic scaling is off" from "this scale would flip a sign". */
    UnitScaleSignNotRequested = 31,
    ShapeChangeNotRequested = 8,
    ComponentReductionNotRequested = 9,
    ComponentReductionAmbiguous = 10,
    ComponentExpansionNotRequested = 11,
    LocationChangeRequiresTransfer = 12,
    ContinuityChangeRequiresTransfer = 13,
    TemporalInterpretationMismatch = 14,
    TemporalIntervalMismatch = 15,
    TemporalIntervalReversed = 16,
    ConservationNotProvidedByTransfer = 17,
    MissingRequiredCapabilities = 18
};

[[nodiscard]] std::string_view connection_reason_name(ConnectionReason reason) noexcept;

struct ConnectionDiagnostic
{
    ConnectionStage stage = ConnectionStage::Identity;
    ConnectionReason reason = ConnectionReason::None;
    /** Nonzero only for `MissingRequiredCapabilities`. */
    std::uint64_t missing_capabilities = 0;
    std::string detail;
};

struct ConnectionCompatibility
{
    ConnectionVerdict verdict = ConnectionVerdict::Exact;
    std::optional<ConnectionStage> first_incompatible_stage;
    std::vector<ConnectionDiagnostic> diagnostics;
    /** Union across stages, so one call reports the whole closure. */
    std::uint64_t missing_capabilities = 0;
    OperatorChain chain;

    [[nodiscard]] bool compiles() const noexcept
    {
        return verdict == ConnectionVerdict::Exact || verdict == ConnectionVerdict::LosslessExplicit ||
               verdict == ConnectionVerdict::LossyExplicit;
    }
};

struct ConnectionRequest
{
    FieldSemantics source;
    FieldSemantics target;
    /** The target port's requirement, which constrains the source. */
    FieldRequirement requirement;
    /** Exact rational validity intervals of the two values. */
    ExactInterval source_interval{{0, 1}, {0, 1}};
    ExactInterval target_interval{{0, 1}, {0, 1}};
    /** The spatial transfer the caller declared for this connection. */
    TransferSemantics transfer;
    /** What the connection must guarantee, independent of what the declared
     * transfer happens to provide. */
    ConservationGuarantee required_conservation = ConservationGuarantee::None;
    std::uint64_t available_capabilities = 0;
    /** Bitmask over `operator_request_bit()`. */
    std::uint64_t explicitly_requested = 0;
    /** Site policy: whether an exact zero-offset unit scale may be inserted
     * without an explicit request. */
    bool allow_automatic_unit_scale = true;
};

/** Total: evaluates every stage and never throws. */
[[nodiscard]] ConnectionCompatibility check_connection(const ConnectionRequest& request);
} // namespace fuse

