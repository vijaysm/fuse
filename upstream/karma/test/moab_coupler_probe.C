#include <mpi.h>
#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>
#include <mbcoupler/Coupler.hpp>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    {
        moab::Core core;
        moab::Range elements;
        moab::ParallelComm pcomm(&core, MPI_COMM_WORLD);
        moab::Coupler coupler(&core, &pcomm, elements, 0, false);
        (void)coupler;
    }
    MPI_Finalize();
    return 0;
}
