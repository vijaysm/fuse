#include "karma_field_coverage.h"

#include "karma_exceptions.h"

#include <cmath>

namespace KARMA
{
void FillPolicy::validate(unsigned components) const
{
    if (values.size() != components) KARMA_THROW(InvalidArgument("fill policy component count differs from field"));
    for (const Real value : values)
        if (!std::isfinite(value)) KARMA_THROW(InvalidArgument("fill policy contains a nonfinite value"));
}
}  // namespace KARMA
