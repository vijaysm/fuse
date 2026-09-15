#include "karma_quasi_interpolation.h"

#include <iostream>

int main()
{
    const KARMA::NativeQuasiInterpolationOptions options{
        KARMA::TransferScheme::FiniteElement, KARMA::FiniteElementTransferOptions{1}};
    const auto capabilities = KARMA::native_quasi_interpolation_capabilities(options);
    if (!capabilities.available || !capabilities.can_build_map || !capabilities.supports_parallel)
    {
        std::cerr << "native FEM quasi-interpolation capability is invalid\n";
        return 1;
    }
    return 0;
}
