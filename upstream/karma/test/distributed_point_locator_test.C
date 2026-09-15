#include "karma_distributed_point_locator.h"

#include <mpi.h>

#include <iostream>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    KARMA::KARMAMesh mesh(2, MPI_COMM_WORLD);
    mesh.generate_box(1, 1, 0, KARMA::Point(0.0, 0.0, 0.0), KARMA::Point(1.0, 1.0, 0.0));
    KARMA::KARMADistributedPointLocator locator(mesh);
    locator.setup();
    const std::vector<KARMA::Point> points{KARMA::Point(0.25, 0.25, 0.0), KARMA::Point(2.0, 2.0, 0.0)};
    const auto results = locator.locate(points);
    const bool valid = results.size() == points.size() &&
                       results[0].status == KARMA::PointLocationStatus::Located &&
                       results[1].status == KARMA::PointLocationStatus::Uncovered;
    int local_valid = valid ? 1 : 0;
    int global_valid = 0;
    MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!global_valid) std::cerr << "distributed point location classification is incorrect\n";
    MPI_Finalize();
    return global_valid ? 0 : 1;
}
