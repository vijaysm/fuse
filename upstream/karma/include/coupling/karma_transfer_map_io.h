#ifndef KARMA_COUPLING_TRANSFER_MAP_IO_H
#define KARMA_COUPLING_TRANSFER_MAP_IO_H

#include "karma_transfer_map.h"

#include <string_view>

namespace KARMA
{
struct TransferMapFileMetadata
{
    std::uint32_t schema_version = 1;
    std::uint32_t scalar_width = sizeof(Real);
    std::uint32_t components = 0;
    std::uint64_t source_signature = 0;
    std::uint64_t target_signature = 0;
};

void save_transfer_map(std::string_view filename, const KARMATransferMap& map,
                       TransferMapFileMetadata metadata = {});
[[nodiscard]] KARMATransferMap load_transfer_map(std::string_view filename,
                                                  TransferMapFileMetadata* metadata = nullptr);
}  // namespace KARMA

#endif
