#ifndef KARMA_COUPLING_FEM_TRANSFER_H
#define KARMA_COUPLING_FEM_TRANSFER_H

#include "karma_fem_field_evaluator.h"
#include "karma_field_coverage.h"
#include "karma_moab_point_locator.h"
#include "karma_transfer_map.h"
#include "karma_system.h"

#include <vector>

namespace KARMA
{
struct FEMTransferResult
{
    std::vector<Real> values;
    std::vector<PointCoverage> coverage;
};

struct FEMElementProjectionResult
{
    std::vector<Real> coefficients;
    CoverageClass coverage = CoverageClass::Full;
    Real covered_measure_fraction = 1.0;
};

class KARMAFEMTransfer
{
  public:
    static FEMTransferResult sample(const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source,
                                    std::span<const Point> target_points, unsigned components,
                                    const FillPolicy* fill = nullptr);

    static FEMElementProjectionResult project_element(const KARMAMesh& target_mesh, moab::EntityHandle target_element,
                                                       const KARMABasis& target_basis, int quadrature_degree,
                                                       const KARMAPointLocator& locator,
                                                       const KARMAFEMFieldEvaluator& source,
                                                       unsigned components,
                                                       bool reject_partial = true);

    static std::vector<FEMElementProjectionResult> project_elements(
        const KARMAMesh& target_mesh, std::span<const moab::EntityHandle> target_elements,
        const KARMABasis& target_basis, int quadrature_degree, const KARMAPointLocator& locator,
        const KARMAFEMFieldEvaluator& source, unsigned components, bool reject_partial = true);

    static void project_cg(const KARMASystem& target_system, unsigned target_variable, int quadrature_degree,
                           const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source, Vec target,
                           Real residual_tolerance = 1.0e-11);
};
}  // namespace KARMA

#endif
