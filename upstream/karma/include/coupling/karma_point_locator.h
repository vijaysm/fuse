#ifndef KARMA_COUPLING_POINT_LOCATOR_H
#define KARMA_COUPLING_POINT_LOCATOR_H

#include "karma_field.h"
#include "karma_geometry_map.h"

#include <cstddef>
#include <string>
#include <vector>

namespace KARMA
{
enum class PointLocationStatus
{
    Located,
    Uncovered,
    InvalidGeometry,
    BackendFailure
};

enum class BoundaryTracePolicy
{
    LowestStableElementId,
    RejectAmbiguous
};

struct PointLocationOptions
{
    Real reference_tolerance = 2.0e-12;
    Real residual_tolerance = 2.0e-12;
    BoundaryTracePolicy trace_policy = BoundaryTracePolicy::LowestStableElementId;
};

struct LocatedPoint
{
    PointLocationStatus status = PointLocationStatus::Uncovered;
    FieldEntryKey source_element;
    moab::EntityHandle source_handle = 0;
    Point physical_point = Point::Zero();
    Point reference_point = Point::Zero();
    Real residual = 0.0;
    int source_rank = -1;
    std::string diagnostic;
};

class KARMAPointLocator
{
  public:
    virtual ~KARMAPointLocator() = default;
    virtual void setup() = 0;
    [[nodiscard]] virtual std::vector< LocatedPoint > locate(std::span<const Point> points) const = 0;
};
}  // namespace KARMA

#endif
