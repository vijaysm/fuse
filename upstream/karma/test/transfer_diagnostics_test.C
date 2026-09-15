#include "karma_transfer_diagnostics.h"

#include <cmath>
#include <iostream>

int main()
{
    const auto report = KARMA::make_transfer_integral_report({0.0}, {0.0}, {1.0}, {0.0});
    if (report.relative_full_error[0] != 0.0 || !std::isinf(report.relative_covered_error[0]))
    {
        std::cerr << "zero-reference diagnostic semantics are invalid\n";
        return 1;
    }
    return 0;
}
