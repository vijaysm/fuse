#ifndef KARMA_COUPLING_SEMANTICS_H
#define KARMA_COUPLING_SEMANTICS_H

#include "karma_semantic_version.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** Versioned semantic description of coupled fields, ports, and components.
 *
 * This header is deliberately dependency-free: no PETSc, MOAB, MPI, or KARMA
 * storage type appears in it. Semantic identity is what a quantity *means* -
 * its units, frame, shape, discrete location, temporal interpretation,
 * ownership, and access - and must not be defined by the container that
 * happens to hold it. Adapting the existing specification records into
 * this model is the job of `karma_semantics_adapter.h`, which is where the
 * storage-aware headers are allowed.
 *
 * Time is exact rational throughout. A `double` time cannot represent a
 * one-third subcycle, and cannot be compared for the exact rendezvous equality
 * that multirate schedules need, so no time value in this model, and no time
 * value in any contract built on it, is a `double`.
 */
namespace KARMA
{
// ---------------------------------------------------------------------------
// Exact rational arithmetic
// ---------------------------------------------------------------------------

enum class RationalError : std::uint8_t
{
    None = 0,
    ZeroDenominator = 1,
    Overflow = 2,
    NotFinite = 3,
    NotDyadic = 4
};

/** Always normalized: `gcd(|numerator|, denominator) == 1` and `denominator > 0`.
 *
 * Because construction normalizes, defaulted equality is value equality.
 * Ordering is deliberately not defaulted - a member-wise `<=>` would compare
 * numerators before denominators, which is not numeric order. Use
 * `rational_compare()`.
 */
struct Rational
{
    std::int64_t numerator = 0;
    std::int64_t denominator = 1;
    bool operator==(const Rational&) const = default;
};

struct RationalResult
{
    std::optional<Rational> value;
    RationalError error = RationalError::None;
};

[[nodiscard]] RationalResult make_rational(std::int64_t numerator, std::int64_t denominator) noexcept;
[[nodiscard]] RationalResult rational_add(Rational left, Rational right) noexcept;
[[nodiscard]] RationalResult rational_subtract(Rational left, Rational right) noexcept;
[[nodiscard]] RationalResult rational_multiply(Rational left, Rational right) noexcept;

/** Exact ordering; the cross products are formed in 128 bits and never overflow. */
[[nodiscard]] std::strong_ordering rational_compare(Rational left, Rational right) noexcept;

/** Exact conversion from a binary floating-point value.
 *
 * Every finite `double` is already a dyadic rational, so this is always
 * lossless and never rounds: `0.125` becomes `1/8`, and the `double` nearest
 * `0.1` becomes its true value `3602879701896397/36028797018963968`. Only a
 * nonfinite value, or one whose reduced numerator or denominator leaves `int64`
 * range, is rejected.
 *
 * Exactness alone is therefore not the interesting property. A value like the
 * `double` nearest `0.1` converts exactly and is still unusable as an exact
 * time, because its denominator is near 2^55 and the first addition of two such
 * values overflows. Use `rational_from_dyadic_bounded()` for any value that
 * will enter exact arithmetic, and declare step sizes as rationals rather than
 * converting them from doubles.
 */
[[nodiscard]] RationalResult rational_from_dyadic(double value) noexcept;

/** Exact conversion that additionally rejects a denominator wider than
 * `maximum_denominator_bits`, reporting `NotDyadic`.
 *
 * This is the guard for legacy double times: a scheduler position formed by
 * halving a clean dyadic step (1/8, 3/1024) passes, while a position carrying
 * the full 53-bit mantissa of a decimal literal is refused instead of being
 * admitted into arithmetic that will overflow.
 */
[[nodiscard]] RationalResult rational_from_dyadic_bounded(double value,
                                                          unsigned maximum_denominator_bits) noexcept;

/** Diagnostics and legacy interoperation only; never for comparison. */
[[nodiscard]] double rational_to_double(Rational value) noexcept;

[[nodiscard]] std::string serialize_rational(Rational value);

using ExactTime = Rational;

struct ExactInterval
{
    ExactTime begin;
    ExactTime end;
};

[[nodiscard]] bool exact_interval_ordered(const ExactInterval& interval) noexcept;
[[nodiscard]] RationalResult exact_interval_length(const ExactInterval& interval) noexcept;

// ---------------------------------------------------------------------------
// Semantic vocabulary
// ---------------------------------------------------------------------------

enum class SemanticShape : std::uint8_t { Scalar = 0, Vector = 1, Tensor2 = 2 };
enum class SemanticLocation : std::uint8_t
{
    Unspecified = 0,
    Vertex = 1,
    Element = 2,
    FiniteElementCoefficient = 3
};
enum class SemanticContinuity : std::uint8_t { Unspecified = 0, Discontinuous = 1, Continuous = 2 };
enum class SemanticTemporal : std::uint8_t
{
    InstantaneousCommitted = 0,
    InstantaneousTrial = 1,
    IntervalAverage = 2,
    IntervalIntegral = 3
};
enum class SemanticFrame : std::uint8_t { Unspecified = 0, Global = 1, ComponentLocal = 2 };
enum class SemanticOwnership : std::uint8_t { Owned = 0, Ghosted = 1, Replicated = 2 };
enum class SemanticAccess : std::uint8_t { ReadOnly = 0, ReadWrite = 1, WriteOnly = 2 };
enum class SemanticUpdate : std::uint8_t { Replace = 0, Accumulate = 1 };
enum class SemanticRole : std::uint8_t { Prognostic = 0, Diagnostic = 1, Input = 2, Parameter = 3 };
enum class PortSemanticDirection : std::uint8_t { Input = 0, Output = 1 };

/** SI base-dimension exponents with an exact affine conversion to base units.
 *
 * Dimensions and conversion are separate on purpose: two fields may share a
 * dimension and still need an explicit affine transform between them, and the
 * compatibility rules treat those two facts differently.
 */
struct UnitDimensions
{
    std::int8_t length = 0;
    std::int8_t mass = 0;
    std::int8_t time = 0;
    std::int8_t current = 0;
    std::int8_t temperature = 0;
    std::int8_t amount = 0;
    std::int8_t luminous = 0;
    Rational scale{1, 1};
    Rational offset{0, 1};
    bool operator==(const UnitDimensions&) const = default;
};

[[nodiscard]] bool unit_is_dimensionless(const UnitDimensions& units) noexcept;

/** Dimension agreement only; scale and offset are ignored. */
/** Parses a unit expression into SI dimensions, or fails.
 *
 * The grammar is deliberately small and total: a product or quotient of
 * symbols, each optionally raised to an integer power. `*` and a space both
 * mean multiplication, `/` divides the rest of nothing but the term that
 * follows it, and `^` takes a signed integer. `1` is the dimensionless unit,
 * so `1/s` parses. Recognized symbols are the seven SI base units and the
 * derived units below, which are exactly the ones whose dimensions are fixed
 * by definition.
 *
 * `kg/m^3`, `m/s`, `m s^-1`, `W/m^2/K` and `J` all parse; `m/s` and `m s^-1`
 * yield the same dimensions, which is the point - comparing unit STRINGS
 * called those two different quantities.
 *
 * **Metric prefixes are deliberately NOT parsed.** `km` is not metres, and
 * accepting it would mean carrying a scale factor whose spelling is ambiguous
 * (`m` is both metre and milli-). A prefixed unit therefore fails to parse
 * and is reported as undeclared, which is honest, rather than being silently
 * read as its base unit, which would be wrong by a factor of a thousand.
 *
 * Returns `std::nullopt` for anything it cannot parse - an unknown symbol, a
 * malformed exponent, an empty term. A caller must treat that as "dimensions
 * unknown" and NOT as dimensionless: see `unrecognized_units_key`.
 */
[[nodiscard]] std::optional<UnitDimensions> parse_unit_expression(std::string_view expression);

[[nodiscard]] bool unit_dimensions_match(const UnitDimensions& left, const UnitDimensions& right) noexcept;

/** Opaque metadata a producer does not understand and must not discard.
 *
 * Unknown *optional* metadata is preserved verbatim and participates in
 * identity. Unknown *required* capability bits are a different question and are
 * rejected by version negotiation, not here.
 */
struct MetadataEntry
{
    std::string key;
    std::vector<std::byte> value;
};

struct FieldSemantics
{
    std::string id;
    SemanticVersion version;
    UnitDimensions units;
    SemanticFrame frame = SemanticFrame::Unspecified;
    SemanticShape shape = SemanticShape::Scalar;
    std::uint32_t components = 1;
    SemanticLocation location = SemanticLocation::Unspecified;
    SemanticContinuity continuity = SemanticContinuity::Unspecified;
    std::uint32_t basis_degree = 0;
    SemanticTemporal temporal = SemanticTemporal::InstantaneousCommitted;
    SemanticRole role = SemanticRole::Input;
    SemanticOwnership ownership = SemanticOwnership::Owned;
    SemanticAccess access = SemanticAccess::ReadOnly;
    SemanticUpdate update = SemanticUpdate::Replace;
    std::vector<MetadataEntry> metadata;
};

struct FieldRequirement
{
    std::string field_id;
    bool required = false;
    SemanticVersion minimum_version;
    SemanticVersion maximum_version;
    std::uint64_t required_capabilities = 0;
    std::vector<MetadataEntry> metadata;
};

struct PortSemantics
{
    std::string id;
    PortSemanticDirection direction = PortSemanticDirection::Input;
    FieldSemantics field;
    FieldRequirement requirement;
    std::vector<MetadataEntry> metadata;
};

struct ComponentSemantics
{
    std::string id;
    SemanticVersion version;
    std::vector<PortSemantics> ports;
    std::uint64_t capabilities = 0;
    bool supports_fractional_steps = false;
    bool supports_checkpoint_restore = false;
    bool requires_communicator = true;
    std::vector<MetadataEntry> metadata;
};

enum class TransformKind : std::uint8_t
{
    UnitScale = 0,
    AffineUnit = 1,
    SignFlip = 2,
    ComponentSelect = 3,
    ComponentReorder = 4,
    Aggregate = 5,
    Expand = 6
};

struct TransformSemantics
{
    TransformKind kind = TransformKind::UnitScale;
    /** Only exact zero-offset unit scaling may be inserted automatically; every
     * other transform must be requested, because each changes the meaning of
     * the value rather than its spelling. */
    bool explicitly_requested = false;
    Rational scale{1, 1};
    Rational offset{0, 1};
    std::vector<std::uint32_t> component_map;
    std::vector<MetadataEntry> metadata;
};

enum class TransferKind : std::uint8_t { Copy = 0, Interpolation = 1, ConservativeRemap = 2 };
enum class ConservationGuarantee : std::uint8_t { None = 0, LocalConservative = 1, GlobalConservative = 2 };

struct TransferSemantics
{
    TransferKind kind = TransferKind::Copy;
    ConservationGuarantee conservation = ConservationGuarantee::None;
    std::string algorithm;
    std::vector<MetadataEntry> metadata;
};

using ScheduleNodeId = std::string;

struct ScheduleNodeSemantics
{
    ScheduleNodeId id;
    std::string kind;
    ExactInterval interval;
    std::vector<MetadataEntry> metadata;
};

// ---------------------------------------------------------------------------
// Canonical serialization and identity
// ---------------------------------------------------------------------------

/** Canonical byte form. Ports and metadata are sorted by identifier, every
 * string and byte sequence is length-prefixed so adjacent values cannot collide,
 * and every integer is fixed-width little-endian. Container order therefore
 * never changes identity, and no delimiter needs escaping.
 */
/** Metadata key under which an adapter records a unit literal it could not
 * interpret, holding the original bytes. A field carrying this is NOT
 * dimensionless - it is a field whose dimensions are unknown, and a
 * compatibility check must treat the two differently. */
inline constexpr std::string_view unrecognized_units_key = "karma.units.unrecognized";

[[nodiscard]] std::vector<std::byte> serialize_field_semantics(const FieldSemantics& field);
[[nodiscard]] std::vector<std::byte> serialize_port_semantics(const PortSemantics& port);
[[nodiscard]] std::vector<std::byte> serialize_component_semantics(const ComponentSemantics& component);
[[nodiscard]] std::vector<std::byte> serialize_transform_semantics(const TransformSemantics& transform);
[[nodiscard]] std::vector<std::byte> serialize_transfer_semantics(const TransferSemantics& transfer);
[[nodiscard]] std::vector<std::byte> serialize_schedule_node_semantics(const ScheduleNodeSemantics& node);

/** FNV-1a over a canonical serialization; the same function the stable
 * field-entry encoding uses, so hashes are comparable across contracts. */
[[nodiscard]] std::uint64_t semantic_hash(std::span<const std::byte> bytes) noexcept;

// ---------------------------------------------------------------------------
// Local structural validation
// ---------------------------------------------------------------------------

enum class SemanticDiagnosticCode : std::uint16_t
{
    EmptyIdentifier = 0,
    DuplicatePortId = 1,
    InvalidEnumValue = 2,
    NonfiniteScale = 3,
    ZeroComponents = 4,
    ShapeComponentMismatch = 5,
    RequirementVersionInverted = 6,
    RequirementFieldMismatch = 7,
    EmptyMetadataKey = 8,
    DuplicateMetadataKey = 9,
    IntervalReversed = 10
};

struct SemanticDiagnostic
{
    SemanticDiagnosticCode code = SemanticDiagnosticCode::EmptyIdentifier;
    /** Dotted path to the offending record, e.g. `solid.ports[coupled].field`. */
    std::string path;
};

struct SemanticValidation
{
    std::vector<SemanticDiagnostic> diagnostics;
    [[nodiscard]] bool ok() const noexcept { return diagnostics.empty(); }
};

/** Purely local and total: it reports every diagnostic it finds rather than the
 * first, so a caller can reconcile a complete local verdict across ranks in one
 * epoch instead of discovering failures one collective at a time.
 */
[[nodiscard]] SemanticValidation validate_field_semantics(const FieldSemantics& field, std::string_view path = "field");
[[nodiscard]] SemanticValidation validate_component_semantics(const ComponentSemantics& component);
} // namespace KARMA

#endif
