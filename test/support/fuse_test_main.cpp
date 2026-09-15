#include "fuse_test_main.hpp"

namespace fuse::testing {
namespace {
int g_rank = 0;
int g_size = 1;
}  // namespace

int rank() { return g_rank; }
int size() { return g_size; }

bool requires_ranks(int exact) { return g_size == exact; }
bool requires_at_least_ranks(int minimum) { return g_size >= minimum; }

void MpiFailureReducer::OnTestEnd(const ::testing::TestInfo& info) {
#if FUSE_HAVE_MPI
  const int local_failed = info.result()->Failed() ? 1 : 0;
  int failing_ranks = 0;
  // Collective: every rank runs the same tests in the same order, so this matches.
  MPI_Allreduce(&local_failed, &failing_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (failing_ranks > 0 && local_failed == 0) {
    // Fail this rank too. Without this the suite's exit status depends on which rank
    // ctest happens to observe, and a partial failure can leave peers in a collective.
    ADD_FAILURE() << "failed on " << failing_ranks << " of " << g_size
                  << " rank(s); this rank's assertions passed. Reported here so the "
                     "test fails uniformly and no rank proceeds to the next collective.";
  }
#else
  (void)info;
#endif
}

}  // namespace fuse::testing

int main(int argc, char** argv) {
  // Order matters: MPI first (Kokkos may query the device count per rank), Kokkos
  // second, GoogleTest last so --gtest_* flags survive both initializers.
#if FUSE_HAVE_MPI
  int provided = 0;
  // THREAD_MULTIPLE because fmtlog's polling thread and a host-parallel region may
  // both be live while a transport progresses.
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &fuse::testing::g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &fuse::testing::g_size);
  if (provided < MPI_THREAD_MULTIPLE && fuse::testing::g_rank == 0) {
    std::fputs("[fuse] warning: MPI_THREAD_MULTIPLE unavailable; "
               "logging from host-parallel regions is degraded, tests still valid\n",
               stderr);
  }
#endif
  {
    Kokkos::ScopeGuard guard(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (fuse::testing::rank() != 0) {
      // Silence progress on non-zero ranks; failures still surface through the
      // reducer and through this rank's own non-zero exit status.
      delete listeners.Release(listeners.default_result_printer());
    }
    listeners.Append(new fuse::testing::MpiFailureReducer);

    const int status = RUN_ALL_TESTS();
#if FUSE_HAVE_MPI
    int any_failed = 0;
    int local = status;
    MPI_Allreduce(&local, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any_failed;
#else
    return status;
#endif
  }
}
