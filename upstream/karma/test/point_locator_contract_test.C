#include "karma_point_locator.h"

#include <iostream>

int main()
{
    KARMA::LocatedPoint result;
    if (result.status != KARMA::PointLocationStatus::Uncovered || result.source_handle != 0)
    {
        std::cerr << "default location result is invalid\n";
        return 1;
    }
    KARMA::PointLocationOptions options;
    if (options.reference_tolerance < 0.0 || options.residual_tolerance < 0.0)
    {
        std::cerr << "default location tolerances are invalid\n";
        return 1;
    }
    return 0;
}
