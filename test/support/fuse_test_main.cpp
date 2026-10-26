// Explicit, not transitive: the feature macros are read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#include "fuse_test_main.hpp"

// std::accumulate, for the per-rank failure gather below. <string> and
// <vector> arrive with fuse_test_main.hpp, which the invariants gate reads
// as an explicit include of this file's own needs.
#include <numeric>

// Rank-to-device binding (docs/dev/device-resident-arcs.md Phase 3):
// <optional>/<string_view> for the glue below; <fmt/format.h> for its
// FUSE_DEVICE_DEBUG trace, the same convention every other device trace in
// this tree uses.
#include <optional>
#include <string_view>
#include <fmt/format.h>
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#include <impl/Kokkos_DeviceManagement.hpp>
#endif
#include "fuse/abi/device_binding.hpp"

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

// Rank-to-device binding glue (docs/dev/device-resident-arcs.md Phase 3):
// gathers the real inputs fuse::abi::resolve_device_id() needs. Duplicated
// from src/abi/participant.cpp's own private copy rather than shared --
// the two live in different libraries (fuse_test_main links fuse::base and
// GTest, not fuse::abi), and the logic is small enough that the
// duplication costs less than a new shared target would.
std::optional<int> explicit_device_id_override(int argc, char** argv) {
  if (const char* env = std::getenv("KOKKOS_DEVICE_ID");
      env != nullptr && env[0] != '\0') {
    return std::atoi(env);
  }
  constexpr std::string_view kPrefix = "--kokkos-device-id=";
  for (int i = 0; i < argc; ++i) {
    if (argv == nullptr || argv[i] == nullptr) continue;
    const std::string_view a{argv[i]};
    if (a.substr(0, kPrefix.size()) == kPrefix) {
      return std::atoi(std::string(a.substr(kPrefix.size())).c_str());
    }
  }
  return std::nullopt;
}

int local_rank_in_node() {
#if FUSE_HAVE_MPI
  int mpi_started = 0;
  if (MPI_Initialized(&mpi_started) != MPI_SUCCESS || mpi_started == 0) {
    return 0;
  }
  MPI_Comm local = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local) != MPI_SUCCESS) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(local, &rank);
  MPI_Comm_free(&local);
  return rank;
#else
  return 0;
#endif
}

int visible_device_count() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  return static_cast<int>(Kokkos::Impl::get_visible_devices().size());
#else
  return 0;
#endif
}

std::optional<int> resolve_device_id_for_this_process(int argc, char** argv) {
  const std::optional<int> override_id = explicit_device_id_override(argc, argv);
  const int count = visible_device_count();
  if (count <= 0 && !override_id.has_value()) return std::nullopt;
  const int local_rank = local_rank_in_node();
  const int id = fuse::abi::resolve_device_id(local_rank, count, override_id);
  if (std::getenv("FUSE_DEVICE_DEBUG") != nullptr) {
    fmt::print(stderr,
               "[device-bind] local_rank={} device_count={} override={} -> "
               "device_id={}\n",
               local_rank, count,
               override_id.has_value() ? std::to_string(*override_id) : "none",
               id);
  }
  return id;
}
}  // namespace

int rank() { return g_rank; }
int size() { return g_size; }

bool requires_ranks(int exact) { return g_size == exact; }
bool requires_at_least_ranks(int minimum) { return g_size >= minimum; }

void MpiFailureReducer::OnTestEnd(const ::testing::TestInfo& info) {
#if FUSE_HAVE_MPI
  const int local_failed = info.result()->Failed() ? 1 : 0;
  // Gathered rather than summed so the report can NAME the failing ranks. A
  // count alone is what a reader gets from ctest, and it is not enough: a
  // launcher commonly forwards only one rank's stdout, so the failing rank's
  // own assertion text can be absent from the log entirely and "failed on 1
  // of 3 rank(s)" is then the whole evidence (measured on the host,
  // 2026-10-08, parallel.window_runner_mpi.3). Which rank failed is the half
  // of the diagnosis that does survive this reduction, so it is carried.
  // Collective: every rank runs the same tests in the same order, so this matches.
  std::vector<int> per_rank(static_cast<std::size_t>(g_size), 0);
  MPI_Allgather(&local_failed, 1, MPI_INT, per_rank.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  const int failing_ranks =
      std::accumulate(per_rank.begin(), per_rank.end(), 0);
  if (failing_ranks > 0 && local_failed == 0) {
    std::string which;
    for (int r = 0; r < g_size; ++r) {
      if (per_rank[static_cast<std::size_t>(r)] == 0) continue;
      if (!which.empty()) which += ", ";
      which += std::to_string(r);
    }
    // Fail this rank too. Without this the suite's exit status depends on which rank
    // ctest happens to observe, and a partial failure can leave peers in a collective.
    ADD_FAILURE() << "failed on rank(s) " << which << " of " << g_size
                  << "; this rank (" << g_rank
                  << ") passed its own assertions. Reported here so the "
                     "test fails uniformly and no rank proceeds to the next "
                     "collective. For the failing rank's own output, rerun "
                     "with a per-rank capture: mpiexec -n "
                  << g_size << " -outfile-pattern=/tmp/<name>-%r.out <binary>";
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
  // THREAD_MULTIPLE requested defensively: nothing in this tree is known to
  // call MPI from more than one thread today (there is no process-wide
  // logger with its own polling thread -- that was fmtlog, fetched but
  // never wired in, removed 2026-10-12), but asking for the weakest level
  // this harness happens to need now would silently narrow what every test
  // registered under it may assume. MULTIPLE is the one request that can
  // only be loosened, never tightened, without re-auditing every test.
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &fuse::testing::g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &fuse::testing::g_size);
  if (provided < MPI_THREAD_MULTIPLE && fuse::testing::g_rank == 0) {
    std::fputs("[fuse] warning: MPI_THREAD_MULTIPLE unavailable; "
               "logging from host-parallel regions is degraded, tests still valid\n",
               stderr);
  }
#endif
  // Rank-to-device binding (docs/dev/device-resident-arcs.md Phase 3):
  // resolved from the REAL argv/env (so an explicit --kokkos-device-id=N
  // or KOKKOS_DEVICE_ID the caller already gave wins, unchanged), then
  // re-injected as the same flag so Kokkos::ScopeGuard's own argv parser
  // -- which strips every --kokkos-* flag it recognizes before
  // GoogleTest ever sees argv -- picks it up exactly as if the caller had
  // passed it. `augmented_argv`/`device_id_flag` must outlive the
  // ScopeGuard construction below, hence declared at this scope rather
  // than inside the `if`.
  std::vector<char*> augmented_argv;
  std::string        device_id_flag;
  if (const std::optional<int> dev =
          fuse::testing::resolve_device_id_for_this_process(argc, argv)) {
    device_id_flag = "--kokkos-device-id=" + std::to_string(*dev);
    augmented_argv.assign(argv, argv + argc);
    augmented_argv.push_back(device_id_flag.data());
    argv = augmented_argv.data();
    argc = static_cast<int>(augmented_argv.size());
  }
  // The diagnostics layer's own process-wide set-up (fuse/base/error.hpp):
  // the FUSE_ON_ERROR family is read here, once, so FUSE_ON_ERROR=debugger
  // (or =abort, =throw) on a failing test binary does what docs/dev/
  // debugging.md says, and every Error a test raises carries this rank.
  // Without the call the environment is parsed by nothing and ignored.
  // Under MPI g_rank is this process's MPI_COMM_WORLD rank; otherwise 0.
  fuse::initialize(fuse::testing::g_rank);
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
    if (fuse::testing::rank() != 0 && std::getenv("FUSE_TEST_PRINT_ALL_RANKS") == nullptr) {
      // Silence progress on non-zero ranks; failures still surface through the
      // reducer and through this rank's own non-zero exit status. The reducer
      // carries only the verdict, not the message, so a non-zero rank's own
      // failure text is lost unless FUSE_TEST_PRINT_ALL_RANKS is set (use it
      // with mpiexec's -outfile-pattern/-errfile-pattern to keep ranks apart).
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
