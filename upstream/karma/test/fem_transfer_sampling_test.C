#include "karma_fem_transfer.h"

#include <iostream>

int main()
{
    if (sizeof(KARMA::FEMTransferResult) == 0)
    {
        std::cerr << "invalid FEM transfer result\n";
        return 1;
    }
    return 0;
}
