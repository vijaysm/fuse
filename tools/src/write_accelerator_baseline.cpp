// Run BY A HUMAN, deliberately, whenever a new machine/compiler/CUDA-toolkit
// combination needs its own pinned num.accelerator iteration counts, or when
// a legitimate change to the accelerator math changes them for every
// toolchain. Never run by a test, and never in CI: a baseline a test can
// rewrite is not a baseline.
//
//   fuse_write_accelerator_baseline test/golden/accelerator.golden
//
// Computes the same ladder/diagonal problems as accelerator_test.cpp (via the
// shared test/num/accelerator_fixtures.hpp) under THIS process's toolchain,
// and upserts the row for FUSE_TOOLCHAIN_ID (fuse/base/config.hpp) into the
// given file. Review the diff before committing -- this tool records what IS,
// not what SHOULD be. In particular, a measured n_dense of -1 means
// Steffensen did not converge within kMaxIters at all under this toolchain,
// which is worth a second look rather than reflexively accepting as the new
// expected value.
#include <cstdio>
#include <string>

#include <fuse/base/config.hpp>

#include "../../test/num/accelerator_fixtures.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: fuse_write_accelerator_baseline <path to accelerator.golden>\n");
    return 2;
  }
  const std::string path = argv[1];

  AcceleratorBaseline row{};
  {
    const LinearMap g = ladder_problem();
    auto picard   = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::None});
    auto constant = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::Constant, .omega = 1.0});
    auto secant   = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::Secant, .initial_relaxation = 0.5});
    auto aitken   = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::Aitken, .initial_relaxation = 0.5});
    auto ils      = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::IQN_ILS, .initial_relaxation = 0.5, .reuse = 8});
    auto imvj     = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
    row.n_picard   = iterate_linear(*picard, g, kTol, kMaxIters);
    row.n_constant = iterate_linear(*constant, g, kTol, kMaxIters);
    row.n_secant   = iterate_linear(*secant, g, kTol, kMaxIters);
    row.n_aitken   = iterate_linear(*aitken, g, kTol, kMaxIters);
    row.n_ils      = iterate_linear(*ils, g, kTol, kMaxIters);
    row.n_imvj     = iterate_linear(*imvj, g, kTol, kMaxIters);
  }
  {
    const std::vector<double> d{0.95, 0.9, 0.8, 0.5, -0.7};
    const std::vector<double> p{1.0, 2.0, 3.0, 4.0, 5.0};
    auto on_diagonal = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::Steffensen, .initial_relaxation = 0.5});
    row.n_diag = iterate_linear(*on_diagonal, diagonal_map(d, p), kTol, kMaxIters);
    auto on_dense = must_make(fuse::AccelConfig{.kind = fuse::AccelKind::Steffensen, .initial_relaxation = 0.5});
    row.n_dense = iterate_linear(*on_dense, ladder_problem(), kTol, kMaxIters);
  }

  std::fprintf(stderr,
               "FUSE_TOOLCHAIN_ID = %s\n"
               "n_picard=%d n_constant=%d n_secant=%d n_aitken=%d n_ils=%d n_imvj=%d "
               "n_diag=%d n_dense=%d\n",
               FUSE_TOOLCHAIN_ID, row.n_picard, row.n_constant, row.n_secant,
               row.n_aitken, row.n_ils, row.n_imvj, row.n_diag, row.n_dense);
  if (row.n_dense < 0 || row.n_picard < 0 || row.n_aitken < 0) {
    std::fprintf(stderr,
                 "warning: a negative count means that problem did not converge within "
                 "kMaxIters under this toolchain -- review before committing this row.\n");
  }

  upsert_accelerator_baseline(path, FUSE_TOOLCHAIN_ID, row);
  std::fprintf(stderr, "wrote toolchain '%s' row to %s\n", FUSE_TOOLCHAIN_ID, path.c_str());
  return 0;
}
