#ifndef KARMA_COUPLING_SEMANTICS_ADAPTER_H
#define KARMA_COUPLING_SEMANTICS_ADAPTER_H

#include "karma_component.h"
#include "karma_operator_split_graph.h"
#include "karma_semantics.h"

#include <optional>

/** Adaptation of the current two-component coupler's specification records into
 * the versioned semantic model.
 *
 * This is a separate header because `karma_component.h` reaches PETSc and MOAB
 * through `karma_field.h`, and `karma_semantics.h` must stay free of both. The
 * direction is deliberately one-way: the existing records are the behavioral
 * oracle and are not modified, so the semantic model is derived from them and
 * never the reverse.
 */
namespace KARMA
{
/** Free-text units in the existing specs are mapped only where the symbol is
 * one this adapter knows. An unrecognized literal yields zero dimensions plus a
 * `karma.units.unrecognized` metadata entry holding the original bytes, so a downstream
 * compatibility check can see that the units are undeclared rather than
 * silently treat the field as dimensionless.
 */
[[nodiscard]] FieldSemantics field_semantics_from_spec(const FieldSpec& specification, PortDirection direction);

[[nodiscard]] PortSemantics port_semantics_from_spec(const PortSpec& specification);

[[nodiscard]] ComponentSemantics component_semantics_from_spec(const ComponentSpec& specification);

/** Exact conversion of a scheduler position, bounded.
 *
 * A position formed by halving a clean dyadic step converts to a small exact
 * rational. A position carrying the full 53-bit mantissa of a decimal literal
 * converts exactly too, but its denominator is near 2^55 and the first addition
 * of two such values overflows, so it is refused with `NotDyadic` instead of
 * being admitted into exact arithmetic. The default bound of 32 denominator
 * bits admits every position the current scheduler forms from a dyadic step and
 * refuses the rest; the correct fix for a refused value is to declare the step
 * as a rational rather than to widen the bound.
 */
[[nodiscard]] RationalResult exact_time(Real position,
                                               unsigned maximum_denominator_bits = 32) noexcept;

struct ExactIntervalResult
{
    ExactInterval value;
    RationalError error = RationalError::None;
};

[[nodiscard]] ExactIntervalResult exact_interval(const TimeInterval& interval) noexcept;
} // namespace KARMA

#endif
