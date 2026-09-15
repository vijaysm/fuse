#ifndef KARMA_COUPLING_COMPONENT_H
#define KARMA_COUPLING_COMPONENT_H

#include "karma_field.h"

#include <string>
#include <vector>

namespace KARMA
{
using ComponentId = std::string;
using FieldId = std::string;
using PortId = std::string;
using GraphId = std::string;

enum class FieldRole
{
    Prognostic,
    Input
};

/** How a value is to be interpreted across a step.
 *
 * These mirror `SemanticTemporal` one-for-one. A specification that could only
 * say `InstantaneousCommitted` could not express a disagreement the semantic
 * model checks for, so a connection between an interval average and an
 * instantaneous value was inexpressible rather than refused.
 */
enum class TimeSemantics
{
    /** The committed value at an instant. */
    InstantaneousCommitted,
    /** A value at an instant within an unconverged iterate, not yet committed. */
    InstantaneousTrial,
    /** The mean over the step, which is not the value at either end of it. */
    IntervalAverage,
    /** The integral over the step, which carries the step length in its units. */
    IntervalIntegral
};

enum class PortDirection
{
    Input,
    Output
};

struct FieldSpec
{
    FieldId id;
    std::string units;
    unsigned components = 0;
    FieldRole role = FieldRole::Input;
    FieldLocation location = FieldLocation::FiniteElementCoefficient;
    TimeSemantics temporal = TimeSemantics::InstantaneousCommitted;
};

struct PortSpec
{
    PortId id;
    PortDirection direction = PortDirection::Input;
    bool required = false;
    FieldSpec field;
};

struct ComponentSpec
{
    ComponentId id;
    std::vector<PortSpec> ports;
    bool supports_fractional_steps = false;
    bool supports_checkpoint_restore = false;
};

void validate_component_spec(const ComponentSpec& specification);

}  // namespace KARMA

#endif
