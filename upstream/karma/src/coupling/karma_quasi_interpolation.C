#include "karma_quasi_interpolation.h"

#include "karma_exceptions.h"

namespace KARMA
{
NativeQuasiInterpolationCapabilities native_quasi_interpolation_capabilities(
    const NativeQuasiInterpolationOptions& options)
{
    if (options.scheme == TransferScheme::FiniteElement && options.finite_element.order >= 1 &&
        options.finite_element.order <= 2)
        return {true, true, true, {}};
    return {false, false, false, "native FEM quasi-interpolation supports only built-in P1/P2"};
}

FEMTransferResult native_quasi_interpolate(const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source,
                                           std::span<const Point> target_points,
                                           const NativeQuasiInterpolationOptions& options, const FillPolicy* fill)
{
    const auto capabilities = native_quasi_interpolation_capabilities(options);
    if (!capabilities.available) KARMA_THROW(InvalidArgument(capabilities.diagnostic));
    return KARMAFEMTransfer::sample(locator, source, target_points, source.components(), fill);
}
}  // namespace KARMA
