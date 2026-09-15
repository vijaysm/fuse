#ifndef KARMA_COUPLING_FIELD_COVERAGE_H
#define KARMA_COUPLING_FIELD_COVERAGE_H

#include "karma_transfer_map.h"

#include <vector>

namespace KARMA
{
struct PointCoverage
{
    FieldEntryKey target;
    CoverageClass classification = CoverageClass::Uncovered;
};

struct ElementCoverage
{
    FieldEntryKey target_element;
    CoverageClass classification = CoverageClass::Uncovered;
    Real covered_measure_fraction = 0.0;
};

struct FillPolicy
{
    std::vector< Real > values;

    explicit FillPolicy(std::vector< Real > component_values) : values(std::move(component_values)) {}
    FillPolicy(Real scalar, unsigned components) : values(components, scalar) {}

    void validate(unsigned components) const;
};
}  // namespace KARMA

#endif
