#ifndef KARMA_COUPLING_QUASI_INTERPOLATION_H
#define KARMA_COUPLING_QUASI_INTERPOLATION_H

#include "karma_fem_transfer.h"
#include "karma_field_transfer.h"

namespace KARMA
{
struct NativeQuasiInterpolationOptions
{
    TransferScheme scheme = TransferScheme::FiniteElement;
    FiniteElementTransferOptions finite_element;
};

struct NativeQuasiInterpolationCapabilities
{
    bool available = false;
    bool can_build_map = false;
    bool supports_parallel = false;
    std::string diagnostic;
};

[[nodiscard]] NativeQuasiInterpolationCapabilities native_quasi_interpolation_capabilities(
    const NativeQuasiInterpolationOptions& options);

[[nodiscard]] FEMTransferResult native_quasi_interpolate(
    const KARMAPointLocator& locator, const KARMAFEMFieldEvaluator& source,
    std::span<const Point> target_points, const NativeQuasiInterpolationOptions& options,
    const FillPolicy* fill = nullptr);
}  // namespace KARMA

#endif
