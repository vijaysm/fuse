#include "karma_moab_point_locator.h"
#include "karma_mesh.h"

#include <iostream>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    KARMA::KARMAMesh mesh(2, MPI_COMM_SELF);
    mesh.generate_box(1, 1, 0, KARMA::Point(0.0, 0.0, 0.0), KARMA::Point(1.0, 1.0, 0.0));
    KARMA::KARMAMoabPointLocator locator(mesh);
    locator.setup();
    const std::vector<KARMA::Point> points{KARMA::Point(0.25, 0.25, 0.0), KARMA::Point(2.0, 2.0, 0.0)};
    const auto results = locator.locate(points);
    if (results.size() != points.size() || results[0].status != KARMA::PointLocationStatus::Located ||
        results[1].status != KARMA::PointLocationStatus::Uncovered)
    {
        std::cerr << "MOAB point location classification is incorrect\n";
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
