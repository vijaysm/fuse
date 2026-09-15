#include "karma_field_transfer.h"

#include <iostream>

int main()
{
    KARMA::KARMAFieldTransfer transfer(
        KARMA::TransferFamily::QuasiInterpolation,
        KARMA::TransferScheme::Shepard,
        KARMA::ShepardTransferOptions{-1.0, 0});
    if (transfer.capabilities().available || transfer.diagnostic().empty())
    {
        std::cerr << "invalid Shepard options were accepted\n";
        return 1;
    }

    const auto interpolation = KARMA::KARMAFieldTransfer::fem_interpolation_capabilities();
    if (!interpolation.available || !interpolation.can_build_map || interpolation.guarantees_conservation)
    {
        std::cerr << "FEM interpolation capability is invalid\n";
        return 1;
    }

    KARMA::KARMAFieldTransfer conservative(
        KARMA::TransferFamily::ConservativeRemap,
        KARMA::TransferScheme::ConservativeSupermesh,
        KARMA::ConservativeRemapOptions{});
    if (conservative.capabilities().available || conservative.capabilities().guarantees_conservation)
    {
        std::cerr << "unavailable conservative backend was reported as operational\n";
        return 1;
    }
    return 0;
}
