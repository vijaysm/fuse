#pragma once


#include "fuse/model/semantics.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** The ordered operator chain a compatible connection compiles to.
 *
 * A connection is either one explicit chain or a refusal. There is no runtime
 * fallback, because a fallback that silently substitutes an operator changes
 * the meaning of the coupled quantity while reporting success.
 *
 * Order is a semantic property, not a formatting choice. Unit and component
 * transforms act on values in the source's own space and must precede spatial
 * transfer; spatial transfer precedes temporal transfer, because projecting in
 * time a field that has not yet been mapped in space would project the wrong
 * samples; constraints act last, on the values that will actually be
 * published. `chain_is_ordered()` is therefore a correctness check.
 */
namespace fuse
{
/** Compiled-chain vocabulary. The transform-phase values are *defined as*
 * `TransformKind`'s, rather than repeated, so the declaration vocabulary a
 * caller writes and the chain vocabulary the compiler emits cannot drift apart;
 * the static assertions below fail if a `TransformKind` value is renumbered.
 * The remaining phases are numbered with gaps so a new operator can join a
 * phase without changing any other ordinal.
 */
enum class OperatorKind : std::uint8_t
{
    UnitScale = static_cast<std::uint8_t>(TransformKind::UnitScale),
    AffineUnit = static_cast<std::uint8_t>(TransformKind::AffineUnit),
    SignFlip = static_cast<std::uint8_t>(TransformKind::SignFlip),
    ComponentSelect = static_cast<std::uint8_t>(TransformKind::ComponentSelect),
    ComponentReorder = static_cast<std::uint8_t>(TransformKind::ComponentReorder),
    Aggregate = static_cast<std::uint8_t>(TransformKind::Aggregate),
    Expand = static_cast<std::uint8_t>(TransformKind::Expand),
    SpatialTransfer = 20,
    TemporalTransfer = 30,
    Clip = 40,
    ConservationEnforcement = 41
};

static_assert(static_cast<std::uint8_t>(OperatorKind::UnitScale) == 0);
static_assert(static_cast<std::uint8_t>(OperatorKind::Expand) == 6);
static_assert(static_cast<std::uint8_t>(TransformKind::Expand) < static_cast<std::uint8_t>(OperatorKind::SpatialTransfer),
              "a transform kind collided with the spatial phase");

/** Lifts a declared transform into the chain vocabulary. */
[[nodiscard]] OperatorKind operator_kind_from_transform(TransformKind kind) noexcept;

enum class OperatorPhase : std::uint8_t
{
    Transform = 0,
    Spatial = 1,
    Temporal = 2,
    Constraint = 3
};

[[nodiscard]] std::string_view operator_kind_name(OperatorKind kind) noexcept;
[[nodiscard]] OperatorPhase operator_phase(OperatorKind kind) noexcept;

/** True when the operator preserves the information in its input. Aggregation,
 * selection, clipping, and expansion do not: the first three discard values and
 * the last invents them. */
[[nodiscard]] bool operator_is_lossless(OperatorKind kind) noexcept;

/** Only exact zero-offset unit scaling may be inserted without being asked
 * for. Every other operator changes what the value means rather than how it is
 * spelled, so the caller must request it and own that decision. */
[[nodiscard]] bool operator_requires_explicit_request(OperatorKind kind) noexcept;

/** Bit position for expressing a set of requested operators. */
[[nodiscard]] std::uint64_t operator_request_bit(OperatorKind kind) noexcept;

struct OperatorStep
{
    OperatorKind kind = OperatorKind::UnitScale;
    /** False only for an automatically inserted exact unit scale. */
    bool explicitly_requested = false;
    /** For `UnitScale`: `v_t = scale * v_s`, `offset` unused (left at its
     * default).
     *
     * For `AffineUnit`: `v_t = scale * v_s +
     * offset`, where `scale = source.scale / target.scale` and `offset =
     * (source.offset - target.offset) / target.scale` -- the division by
     * `target.scale` applies to BOTH fields, not only to `scale`. Deriving
     * it: a source value `v_s` represents the physical quantity `q =
     * source.scale * v_s + source.offset` (SI base units); the target
     * representation of that same quantity is `v_t = (q - target.offset) /
     * target.scale`. Substituting gives exactly the formula above. The one
     * producer of this step (`check_units()`, src/model/compatibility.cpp)
     * applies it; there is no consumer yet (`ir_can_emit(AffineUnit)` is
     * false), so this comment is the step's only specification until one
     * lands. */
    Rational scale{1, 1};
    Rational offset{0, 1};
    std::vector<std::uint32_t> component_map;
    std::string algorithm;
    ConservationGuarantee conservation = ConservationGuarantee::None;
};

struct OperatorChain
{
    std::vector<OperatorStep> steps;

    [[nodiscard]] bool empty() const noexcept { return steps.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return steps.size(); }
};

/** Phases must be nondecreasing. Within the transform phase the relative order
 * of independent operators is not constrained, because unit scaling and
 * component selection commute. */
[[nodiscard]] bool chain_is_ordered(const OperatorChain& chain) noexcept;

/** True when every step that requires an explicit request carries one. */
[[nodiscard]] bool chain_requests_are_authorized(const OperatorChain& chain) noexcept;

/** True when no step discards or invents information. */
[[nodiscard]] bool chain_is_lossless(const OperatorChain& chain) noexcept;

[[nodiscard]] std::vector<std::byte> serialize_operator_chain(const OperatorChain& chain);
[[nodiscard]] std::uint64_t operator_chain_hash(const OperatorChain& chain);
} // namespace fuse

