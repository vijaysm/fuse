// R3, half one: the remap path's parallel REFUSAL, exercised for real.
//
// WHY THIS FILE EXISTS. `build_weights` refuses a partitioned read with
// `E-REMAP-PARALLEL` (src/remap/intx_weights.cpp). Until this file, that branch
// had never executed: the MOAB test leg is built with MPI OFF, so
// `MoabMeshBackend::communicator_size()` is a compile-time `return 1` and only
// the non-firing side of the `if` was ever reached. An unexercised guard is not
// a guard -- it is a comment that happens to compile -- and this project has
// three recorded instances of a test that passed against the very bug it was
// named for. Everything below runs only in a MOAB + MPI configuration, under
// mpiexec, and is invisible to the three default legs.
//
// WHAT IS ESTABLISHED, and each is a separate failure mode:
//
//   1. ONE RANK IS NOT A REFUSAL. With a real communicator of size 1 the
//      weights must still build. A guard that fired on `communicator != none`
//      rather than on `communicator_size() > 1` would make every MPI job
//      refuse remap, and no serial test can see the difference.
//
//      The matrix is checked against the values `RemapSphere.IdentityOnMoabs-
//      CubedSphere` asserts for the SAME pair on the serial backend -- the
//      identity, one entry per cell, defects below 1e-12. Those are copied
//      constants, NOT a serial run performed here: this binary never
//      constructs a backend without a communicator, so if the serial
//      expectations ever change, that test moves and this one has to be
//      updated with it. Said plainly because "checked against the serial
//      answer" would overstate what happens.
//   2. MORE THAN ONE RANK IS A REFUSAL, ON EVERY RANK, with code
//      `E-REMAP-PARALLEL` and a message that names the rank count. Checked on
//      every rank because a guard that fired on rank 0 only would deadlock the
//      job at the next collective rather than fail it.
//   3. THE REFUSAL IS AHEAD OF THE BUILDER SPLIT. The same plan asked for
//      through the high-order builder (`method = "tempest"`) must be refused
//      with the SAME code. The guard lives in `resolve()`, which both builders
//      go through; moving it into either branch would leave the other silently
//      computing a partition-local answer. Nothing else in the suite pins that
//      placement.
//   4. THE READ REALLY WAS PARTITIONED. Asserted, not assumed: if MOAB had
//      quietly given every rank the whole mesh, every check above would pass
//      for the wrong reason and this file would be testing nothing.
//   5. A REFUSED BUILD IS NOT CACHED. `RemapCache` must count the attempt as a
//      miss and store nothing, so a caller that fixes the mesh and retries gets
//      a real attempt rather than the stale error.
//
// Not a GoogleTest binary, for the same reasons as moab_parallel_test.cpp: it
// must call MPI_Init before anything, and a failure on one rank has to bring
// the job down rather than leave the others waiting in a collective.

#include <soma/mesh/moab_backend.hpp>
#include <soma/remap/remap_plan.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_rank = 0;
int g_size = 1;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "[rank %d] FAIL: %s\n", g_rank, what.c_str());
    std::fflush(stderr);
    ++g_failures;
  }
}

void check_eq(long got, long want, const std::string& what) {
  check(got == want, what + ": got " + std::to_string(got) + ", want " +
                         std::to_string(want));
}

long reduce_sum(long v) {
  long out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

std::string describe(const soma::Error& e) { return e.code + ": " + e.message; }

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  if (argc < 2) {
    if (g_rank == 0) {
      std::fprintf(stderr, "usage: remap_parallel_test <mesh file>\n");
    }
    MPI_Finalize();
    return 2;
  }
  const std::string path = argv[1];

  // MOAB's spherecube.h5m, twice. Two independent file sets holding identical
  // geometry, so the intersection machinery runs for real and the right answer
  // at one rank is known exactly (the identity). The same file on both sides
  // also means the partitions align, which is deliberate: it makes the
  // refusal at more than one rank a statement about the COMMUNICATOR and not
  // about this particular pair of partitions happening to be awkward.
  soma::CouplingGraph graph;
  const soma::MeshId ms =
      graph.add_mesh(soma::MeshDecl{.name = "src", .source = path});
  const soma::MeshId mt =
      graph.add_mesh(soma::MeshDecl{.name = "tgt", .source = path});
  const soma::RegionId rs = graph.add_region(
      soma::Region{.name = "rsrc", .mesh = ms, .dimension = 2});
  const soma::RegionId rt = graph.add_region(
      soma::Region{.name = "rtgt", .mesh = mt, .dimension = 2});

  soma::MoabMeshBackend backend;
  {
    const auto r = backend.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    check(r.ok(), "set_communicator: " +
                      (r.ok() ? std::string{} : describe(r.error())));
    if (!r.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }
  for (const soma::MeshId id : {ms, mt}) {
    const auto r = backend.load(graph.mesh(id));
    check(r.ok(), "load: " + (r.ok() ? std::string{} : describe(r.error())));
    if (!r.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  // ---- 4. the read really was partitioned --------------------------------
  check_eq(backend.communicator_size(), g_size,
           "MoabMeshBackend::communicator_size() disagrees with MPI_Comm_size");

  const long kGlobalFaces = 3456;
  long local_faces = 0;
  {
    const auto id = backend.mesh_id("src");
    check(id.ok(), "mesh_id(src)");
    if (!id.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    const auto v = backend.view(id.value());
    check(v.ok(), "view(src)");
    if (!v.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    local_faces = static_cast<long>(v.value()->entity_count(2));
  }
  std::printf("[rank %d/%d] local source faces = %ld of %ld global\n", g_rank,
              g_size, local_faces, kGlobalFaces);
  std::fflush(stdout);
  if (g_size > 1) {
    check(local_faces < kGlobalFaces,
          "this rank holds all " + std::to_string(local_faces) +
              " faces, so the read was NOT partitioned and nothing below "
              "exercises the parallel path");
    check_eq(reduce_sum(local_faces), kGlobalFaces,
             "the ranks' face counts do not sum to the global count");
  } else {
    check_eq(local_faces, kGlobalFaces,
             "a one-rank communicator must still see the whole mesh");
  }

  soma::RemapPlan plan;
  plan.src = rs;
  plan.tgt = rt;
  plan.flags = soma::RemapFlags::Conservative;

  const auto w = soma::build_weights(plan, graph, backend);

  if (g_size == 1) {
    // ---- 1. one rank is not a refusal ------------------------------------
    check(w.ok(), "build_weights under a ONE-RANK communicator must succeed, "
                  "not refuse: " +
                      (w.ok() ? std::string{} : describe(w.error())));
    if (w.ok()) {
      const soma::RemapWeights& m = w.value();
      std::printf("[rank 0] np=1 nnz=%zu rows=%zu cols=%zu row defect=%.3e "
                  "col defect=%.3e src cov=%.15f\n",
                  m.nnz(), m.n_rows, m.n_cols, m.max_row_defect,
                  m.max_col_defect, m.min_src_coverage);
      check_eq(static_cast<long>(m.n_rows), kGlobalFaces, "n_rows");
      check_eq(static_cast<long>(m.n_cols), kGlobalFaces, "n_cols");
      // Identical geometry: exactly one overlap polygon per cell, and the
      // matrix is a permutation of ones. Anything else means the partitioned
      // read changed what the intersection saw.
      check_eq(static_cast<long>(m.nnz()), kGlobalFaces,
               "the identity map must have one entry per cell");
      check(m.max_row_defect < 1e-12,
            "row defect " + std::to_string(m.max_row_defect));
      check(m.max_col_defect < 1e-12,
            "col defect " + std::to_string(m.max_col_defect));
      check(std::fabs(m.min_src_coverage - 1.0) < 1e-12,
            "src coverage " + std::to_string(m.min_src_coverage));

      std::vector<double> src(m.n_cols, 7.25);
      std::vector<double> dst(m.n_rows, 0.0);
      const auto ap = soma::apply_weights(m, src.data(), dst.data(), 1);
      check(ap.ok(), "apply_weights");
      double worst = 0.0;
      for (const double v : dst) worst = std::max(worst, std::fabs(v - 7.25));
      check(worst < 1e-11,
            "a constant field did not survive the one-rank map (worst error " +
                std::to_string(worst) + ")");
    }
  } else {
    // ---- 2. more than one rank is a refusal, on every rank ---------------
    check(!w.ok(),
          "build_weights SUCCEEDED on a mesh partitioned across " +
              std::to_string(g_size) +
              " ranks. The supermesh here intersects only the cells this rank "
              "holds, so whatever it returned omits every overlap that "
              "crosses a partition boundary -- a plausible matrix with no "
              "diagnostic, which is exactly what E-REMAP-PARALLEL exists to "
              "prevent");
    if (!w.ok()) {
      const soma::Error& e = w.error();
      std::printf("[rank %d] np=%d refused with %s\n", g_rank, g_size,
                  describe(e).c_str());
      std::fflush(stdout);
      check(e.code == "E-REMAP-PARALLEL",
            "the refusal must be E-REMAP-PARALLEL, not " + describe(e));
      // The exact phrase, not just the digit: the message is long and a bare
      // std::to_string(g_size) would match a stray character in it.
      check(e.message.find(std::to_string(g_size) + " ranks") !=
                std::string::npos,
            "the message must name the rank count as '" +
                std::to_string(g_size) + " ranks': " + e.message);
    }

    // ---- 3. the refusal is ahead of the builder split --------------------
    // Same regions, the OTHER builder. `resolve()` runs before the plan
    // reaches either one, so both must be refused for the same reason. If
    // this leg reports E-REMAP-NOTEMPEST (or a geometry code) instead, the
    // guard has been moved into the first-order branch and the high-order
    // path is quietly computing a partition-local answer.
    soma::RemapPlan hi = plan;
    hi.method = "tempest";
    hi.src_order = 2;
    hi.tgt_order = 2;
    const auto wh = soma::build_weights(hi, graph, backend);
    check(!wh.ok(), "the high-order builder accepted a partitioned read");
    if (!wh.ok()) {
      check(wh.error().code == "E-REMAP-PARALLEL",
            "the high-order builder refused with " + describe(wh.error()) +
                " rather than E-REMAP-PARALLEL, so the guard no longer sits "
                "ahead of the builder split");
    }

    // ---- 5. a refused build is not cached --------------------------------
    soma::RemapCache cache;
    const auto c1 = cache.get_or_build(plan, graph, backend);
    check(!c1.ok(), "RemapCache::get_or_build swallowed the refusal");
    if (!c1.ok()) {
      check(c1.error().code == "E-REMAP-PARALLEL",
            "the cache changed the error code to " + describe(c1.error()));
    }
    const auto c2 = cache.get_or_build(plan, graph, backend);
    check(!c2.ok(), "the second attempt succeeded");
    check_eq(static_cast<long>(cache.size()), 0,
             "a failed build must not be cached");
    check_eq(static_cast<long>(cache.misses()), 2,
             "both attempts must count as real attempts");
    check_eq(static_cast<long>(cache.hits()), 0, "cache hits");
  }

  const long total_failures = reduce_sum(static_cast<long>(g_failures));
  if (g_rank == 0) {
    std::printf("%s: %ld failure(s) across %d rank(s)\n",
                total_failures == 0 ? "PASS" : "FAIL", total_failures, g_size);
    std::fflush(stdout);
  }
  MPI_Finalize();
  return total_failures == 0 ? 0 : 1;
}
