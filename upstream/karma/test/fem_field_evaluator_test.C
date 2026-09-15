#include "karma_basis.h"

#include <iostream>
#include <vector>

int main()
{
    KARMA::KARMABasisRegistry registry;
    const auto basis = registry.create({KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::LagrangeNodal, 2,
                                        KARMA::BasisContinuity::cG});
    std::vector<KARMA::Real> canonical{2.0};
    std::vector<KARMA::Real> local(1, 0.0);
    basis.apply_orientation_inverse(1, 0, 1, canonical, local);
    std::vector<KARMA::Real> recovered(1, 0.0);
    basis.apply_orientation(1, 0, 1, local, recovered);
    if (recovered != canonical)
    {
        std::cerr << "orientation inverse is inconsistent\n";
        return 1;
    }
    return 0;
}
