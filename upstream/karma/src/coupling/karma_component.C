#include "karma_component.h"

#include <algorithm>
#include <stdexcept>

namespace KARMA
{
void validate_component_spec(const ComponentSpec& specification)
{
    if (specification.id.empty()) throw std::invalid_argument("component id is empty");
    if (specification.ports.size() != 2) throw std::invalid_argument("component requires two ports");

    bool output = false;
    bool input = false;
    for (std::size_t index = 0; index < specification.ports.size(); ++index)
    {
        const auto& port = specification.ports[index];
        if (port.id.empty() || port.field.id.empty() || port.field.units.empty() || port.field.components == 0)
            throw std::invalid_argument("component port has incomplete field metadata");
        if (port.field.components != 1 || port.field.location != FieldLocation::FiniteElementCoefficient ||
            port.field.temporal != TimeSemantics::InstantaneousCommitted)
            throw std::invalid_argument("component field metadata is unsupported");
        if (std::any_of(specification.ports.begin(), specification.ports.begin() + static_cast<std::ptrdiff_t>(index),
                        [&](const auto& previous) { return previous.id == port.id; }))
            throw std::invalid_argument("component port ids must be unique");
        if (port.direction == PortDirection::Output)
        {
            if (output || port.field.role != FieldRole::Prognostic) throw std::invalid_argument("invalid output port");
            output = true;
        }
        else
        {
            if (input || port.field.role != FieldRole::Input || !port.required)
                throw std::invalid_argument("invalid input port");
            input = true;
        }
    }
    if (!output || !input) throw std::invalid_argument("component requires one input and one output");
}
}  // namespace KARMA
