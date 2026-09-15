#include "karma_conservative_transfer.h"

namespace KARMA
{
TransferCapabilities conservative_supermesh_capabilities()
{
    return {false, false, false, false, false,
            "ConservativeSupermesh requires TempestRemap or a validated volumetric remap backend"};
}
}  // namespace KARMA
