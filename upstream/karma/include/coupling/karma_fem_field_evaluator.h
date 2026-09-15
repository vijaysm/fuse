#ifndef KARMA_COUPLING_FEM_FIELD_EVALUATOR_H
#define KARMA_COUPLING_FEM_FIELD_EVALUATOR_H

#include "karma_field.h"
#include "karma_point_locator.h"

#include <span>
#include <vector>

namespace KARMA
{
struct FEMFieldEvaluatorOptions
{
    BoundaryTracePolicy trace_policy = BoundaryTracePolicy::LowestStableElementId;
    bool reject_nonfinite = true;
};

class KARMAFEMFieldEvaluator
{
  public:
    KARMAFEMFieldEvaluator(const KARMAField& field, FEMFieldEvaluatorOptions options = {});

    [[nodiscard]] std::vector<Real> evaluate(const LocatedPoint& point) const;
    [[nodiscard]] std::vector<std::vector<Real>> evaluate_batch(std::span<const LocatedPoint> points) const;
    [[nodiscard]] unsigned components() const noexcept { return _field.components(); }
    [[nodiscard]] const KARMAField& field() const noexcept { return _field; }

  private:
    const KARMAField& _field;
    FEMFieldEvaluatorOptions _options;
};
}  // namespace KARMA

#endif
