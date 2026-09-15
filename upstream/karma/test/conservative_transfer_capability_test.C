#include "karma_conservative_transfer.h"

#include <iostream>

int main()
{
    const auto capabilities = KARMA::conservative_supermesh_capabilities();
    if (capabilities.available || capabilities.guarantees_conservation || capabilities.unavailable_reason.empty())
    {
        std::cerr << "conservative capability boundary is invalid\n";
        return 1;
    }
    return 0;
}
