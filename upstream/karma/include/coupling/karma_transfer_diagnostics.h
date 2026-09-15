#ifndef KARMA_COUPLING_TRANSFER_DIAGNOSTICS_H
#define KARMA_COUPLING_TRANSFER_DIAGNOSTICS_H

#include "karma_field_coverage.h"

#include <petscmat.h>
#include <petscvec.h>

#include <vector>

namespace KARMA
{
struct TransferIntegralReport
{
    std::vector< Real > source_integral;
    std::vector< Real > target_full_integral;
    std::vector< Real > target_covered_integral;
    std::vector< Real > fill_or_extension_integral;
    std::vector< Real > relative_full_error;
    std::vector< Real > relative_covered_error;
};

struct TransferResidualReport
{
    Real residual_norm = 0;
    Real scale = 0;
    Real tolerance = 0;
    bool within_tolerance = false;
};

[[nodiscard]] Real relative_integral_error(Real reference, Real value);
[[nodiscard]] TransferIntegralReport make_transfer_integral_report(
    std::vector<Real> source_integral, std::vector<Real> target_full_integral,
    std::vector<Real> target_covered_integral, std::vector<Real> fill_or_extension_integral);

/** Recomputes residual = mass * solution - rhs and tests
 * ||residual||_inf <= tolerance * max(||rhs||_inf, ||mass||_inf ||solution||_inf, epsilon).
 * All norms are collective and the caller supplies reusable residual storage. */
[[nodiscard]] bool transfer_residual_within_tolerance(Mat mass, Vec solution, Vec rhs, Vec residual,
                                                       Real tolerance = 1.0e-11);
[[nodiscard]] TransferResidualReport transfer_residual_report(
    Mat mass, Vec solution, Vec rhs, Vec residual, Real tolerance = 1.0e-11);
}  // namespace KARMA

#endif
