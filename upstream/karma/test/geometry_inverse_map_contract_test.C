#include "karma_geometry_map.h"

#include <array>
#include <iostream>

int main()
{
    const KARMA::KARMAReferenceTopology topology(
        KARMA::KARMAReferenceTopology::from_moab(moab::MBTRI, 3));
    const std::array<KARMA::Point, 3> nodes{
        KARMA::Point(0.0, 0.0, 0.0), KARMA::Point(1.0, 0.0, 0.0), KARMA::Point(0.0, 1.0, 0.0)};
    const KARMA::KARMAGeometryMap geometry(topology, nodes);
    const auto located = geometry.try_inverse_map(KARMA::Point(0.25, 0.25, 0.0), 1.0e-12, 1.0e-10);
    if (located.status != KARMA::InverseMapStatus::Located)
    {
        std::cerr << "interior point was not located\n";
        return 1;
    }
    const auto uncovered = geometry.try_inverse_map(KARMA::Point(1.2, 0.2, 0.0), 1.0e-12, 1.0e-10);
    if (uncovered.status != KARMA::InverseMapStatus::NotContained)
    {
        std::cerr << "exterior point was not classified as not contained\n";
        return 1;
    }
    return 0;
}
