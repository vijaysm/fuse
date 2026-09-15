#include "karma_fem_transfer.h"

#include <iostream>

int main()
{
    KARMA::FEMElementProjectionResult result;
    if (result.coverage != KARMA::CoverageClass::Full || result.covered_measure_fraction != 1.0)
    {
        std::cerr << "invalid default projection result\n";
        return 1;
    }
    return 0;
}
