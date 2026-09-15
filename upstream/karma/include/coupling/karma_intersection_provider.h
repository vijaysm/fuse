#ifndef KARMA_COUPLING_INTERSECTION_PROVIDER_H
#define KARMA_COUPLING_INTERSECTION_PROVIDER_H

#include "karma_field.h"

#include <vector>

namespace KARMA
{
struct IntersectionQuadraturePoint
{
    FieldEntryKey source_element;
    FieldEntryKey target_element;
    Real measure = 0.0;
    Point source_reference;
    Point target_reference;
};

class KARMAIntersectionProvider
{
  public:
    virtual ~KARMAIntersectionProvider() = default;
    [[nodiscard]] virtual std::vector< IntersectionQuadraturePoint > quadrature() const = 0;
};
}  // namespace KARMA

#endif
