#include "karma_transfer_diagnostics.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace KARMA
{
Real relative_integral_error(Real reference, Real value)
{
    if (!std::isfinite(reference) || !std::isfinite(value))
        KARMA_THROW(InvalidArgument("integral diagnostic contains a nonfinite value"));
    if (reference == 0.0) return value == 0.0 ? 0.0 : std::numeric_limits<Real>::infinity();
    return std::abs(value - reference) / std::abs(reference);
}

TransferIntegralReport make_transfer_integral_report(std::vector<Real> source_integral,
                                                      std::vector<Real> target_full_integral,
                                                      std::vector<Real> target_covered_integral,
                                                      std::vector<Real> fill_or_extension_integral)
{
    if (source_integral.size() != target_full_integral.size() || source_integral.size() != target_covered_integral.size() ||
        source_integral.size() != fill_or_extension_integral.size())
        KARMA_THROW(InvalidArgument("integral diagnostic component counts differ"));
    TransferIntegralReport report{std::move(source_integral), std::move(target_full_integral),
                                  std::move(target_covered_integral), std::move(fill_or_extension_integral), {}, {}};
    report.relative_full_error.resize(report.source_integral.size());
    report.relative_covered_error.resize(report.source_integral.size());
    for (std::size_t component = 0; component < report.source_integral.size(); ++component)
    {
        report.relative_full_error[component] = relative_integral_error(report.source_integral[component],
                                                                          report.target_full_integral[component]);
        report.relative_covered_error[component] = relative_integral_error(report.source_integral[component],
                                                                            report.target_covered_integral[component]);
    }
    return report;
}

bool transfer_residual_within_tolerance(Mat mass, Vec solution, Vec rhs, Vec residual, Real tolerance)
{
    return transfer_residual_report(mass, solution, rhs, residual, tolerance).within_tolerance;
}

TransferResidualReport transfer_residual_report(Mat mass, Vec solution, Vec rhs, Vec residual, Real tolerance)
{
    if (mass == nullptr || solution == nullptr || rhs == nullptr || residual == nullptr)
        KARMA_THROW(InvalidArgument("transfer residual validation requires non-null PETSc objects"));
    if (!std::isfinite(tolerance) || tolerance <= 0.0)
        KARMA_THROW(InvalidArgument("transfer residual tolerance must be finite and positive"));
    if (MatMult(mass, solution, residual) != PETSC_SUCCESS || VecAXPY(residual, -1.0, rhs) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to compute transfer residual"));

    PetscReal residual_norm = 0.0;
    PetscReal rhs_norm = 0.0;
    PetscReal solution_norm = 0.0;
    PetscReal matrix_norm = 0.0;
    if (VecNorm(residual, NORM_INFINITY, &residual_norm) != PETSC_SUCCESS ||
        VecNorm(rhs, NORM_INFINITY, &rhs_norm) != PETSC_SUCCESS ||
        VecNorm(solution, NORM_INFINITY, &solution_norm) != PETSC_SUCCESS ||
        MatNorm(mass, NORM_INFINITY, &matrix_norm) != PETSC_SUCCESS)
        KARMA_THROW(InvalidArgument("failed to compute transfer residual scales"));
    if (!std::isfinite(residual_norm) || !std::isfinite(rhs_norm) || !std::isfinite(solution_norm) ||
        !std::isfinite(matrix_norm))
        return {residual_norm, 0.0, tolerance, false};

    // The backward-error scale covers both the assembled forcing and the
    // matrix-action magnitude. Epsilon supplies only a zero-problem floor and
    // therefore does not create a mesh- or units-dependent absolute tolerance.
    const PetscReal scale = std::max({rhs_norm, matrix_norm * solution_norm,
                                      std::numeric_limits<PetscReal>::epsilon()});
    return {residual_norm, scale, tolerance, residual_norm <= tolerance * scale};
}
}  // namespace KARMA
