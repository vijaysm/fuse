// Explicit, not transitive: the feature macros are read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"

#include "fuse_test_main.hpp"

namespace fuse::testing {

int self_communicator_handle() {
#if FUSE_HAVE_MPI
  return static_cast<int>(MPI_Comm_c2f(MPI_COMM_SELF));
#else
  return 0;
#endif
}
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

    // Names the execution spaces this build enabled, once, on rank 0. It is
    // here rather than in a test because it is EVIDENCE FOR AN EXIT CRITERION
    // ("the kernels run on more than one execution space") and a reader of a
    // CI log should not have to infer it from the configure flags. A typed
    // test that silently instantiates once looks identical in a summary to
    // one that covered everything -- which is exactly what happened: the
    // space list omitted Threads, and every typed test in the project ran on
    // Serial alone until 2026-09-17.
    if (::fuse::testing::rank() == 0) {
      std::printf("[  spaces  ] enabled:");
#if defined(KOKKOS_ENABLE_CUDA)
      std::printf(" Cuda");
#endif
#if defined(KOKKOS_ENABLE_HIP)
      std::printf(" HIP");
#endif
#if defined(KOKKOS_ENABLE_SYCL)
      std::printf(" SYCL");
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
      std::printf(" OpenMP");
#endif
#if defined(KOKKOS_ENABLE_THREADS)
      std::printf(" Threads");
#endif
      std::printf(" Serial  (default %s)\n",
                  Kokkos::DefaultExecutionSpace::name());
    }
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
