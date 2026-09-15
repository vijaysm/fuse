// The distributed half of Task 5.1, run under mpiexec -n 2.
//
// Everything serial can be checked in moab_mesh_test.cpp. What cannot is the
// one property `MeshView` was written to force: `entity_count` is a LOCAL
// count, ghosts exist, and a global quantity is a reduction over OWNED
// entities. A serial backend satisfies every assertion about ownership
// vacuously, so the only way to know the MOAB backend gets it right is to
// partition a real mesh across real ranks and watch the numbers disagree with
// the global ones.
//
// Not a GoogleTest binary: it must call MPI_Init before anything, and a
// failure on one rank has to bring the job down rather than leave the others
// waiting in a collective. Plain checks and MPI_Abort do that; GoogleTest's
// per-rank pass/fail reporting does not.

#include <soma/mesh/moab_backend.hpp>

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_rank = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "[rank %d] FAIL: %s\n", g_rank, what.c_str());
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

double reduce_sum(double v) {
  double out = 0.0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2) {
    if (g_rank == 0) {
      std::fprintf(stderr, "usage: moab_parallel_test <mesh file>\n");
    }
    MPI_Finalize();
    return 2;
  }
  const std::string path = argv[1];

  soma::MoabMeshBackend backend;
  {
    const auto r = backend.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    check(r.ok(), "set_communicator: " + (r.ok() ? std::string{} : r.error().code +
                                                       ": " + r.error().message));
    if (!r.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }
  {
    const auto r =
        backend.load(soma::MeshDecl{.name = "sphere", .source = path});
    check(r.ok(), "load: " + (r.ok() ? std::string{}
                                     : r.error().code + ": " + r.error().message));
    if (!r.ok()) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  const auto id = backend.mesh_id("sphere");
  check(id.ok(), "mesh_id");
  if (!id.ok()) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const auto vr = backend.view(id.value());
  check(vr.ok(), "view");
  if (!vr.ok()) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const soma::MeshView& v = *vr.value();

  // ---- the whole point: LOCAL counts, not global ones --------------------
  const long local_faces = static_cast<long>(v.entity_count(2));
  const long owned_faces = static_cast<long>(v.owned_count(2));
  const long local_verts = static_cast<long>(v.entity_count(0));
  const long owned_verts = static_cast<long>(v.owned_count(0));

  std::printf("[rank %d/%d] faces local=%ld owned=%ld | verts local=%ld "
              "owned=%ld\n",
              g_rank, size, local_faces, owned_faces, local_verts, owned_verts);

  const long kGlobalFaces = 3456;
  const long kGlobalVerts = 3458;

  if (size > 1) {
    check(local_faces < kGlobalFaces,
          "entity_count(2) reported the GLOBAL face count " +
              std::to_string(local_faces) +
              "; a partitioned read must report only this rank's faces");
    check(local_verts < kGlobalVerts, "entity_count(0) looks global");
  }
  check_eq(reduce_sum(owned_faces), kGlobalFaces,
           "owned faces summed over ranks");
  check_eq(reduce_sum(owned_verts), kGlobalVerts,
           "owned vertices summed over ranks");
  check(owned_faces <= local_faces, "owned_count(2) exceeds entity_count(2)");
  check(owned_verts <= local_verts, "owned_count(0) exceeds entity_count(0)");

  // Vertices on a partition boundary are visible to more than one rank, so
  // the sum of LOCAL counts overshoots. If it did not, nothing here would be
  // testing ghost handling at all.
  const long total_local_verts = reduce_sum(local_verts);
  if (size > 1) {
    check(total_local_verts > kGlobalVerts,
          "no vertex is shared between ranks, so this run does not exercise "
          "ownership at all (sum of local counts " +
              std::to_string(total_local_verts) + ")");
  }

  // ---- is_owned agrees with owned_count and with owned_entities ----------
  long counted = 0;
  for (long i = 0; i < local_verts; ++i) {
    if (v.is_owned(0, static_cast<std::size_t>(i))) ++counted;
  }
  check_eq(counted, owned_verts, "is_owned(0, .) disagrees with owned_count");

  const soma::Region all_faces{
      .name = "all", .mesh = id.value(), .entity_set = "", .dimension = 2};
  const auto owned = backend.owned_entities(all_faces);
  check(owned.ok(), "owned_entities");
  if (owned.ok()) {
    check_eq(static_cast<long>(owned.value().size()), owned_faces,
             "owned_entities size");
    for (std::size_t k = 1; k < owned.value().size(); ++k) {
      check(owned.value()[k - 1] < owned.value()[k],
            "owned_entities is not strictly increasing");
    }
  }

  // region_size is documented to be local and to include ghosts.
  const auto rs = backend.region_size(all_faces);
  check(rs.ok(), "region_size");
  if (rs.ok()) {
    check_eq(static_cast<long>(rs.value()), local_faces,
             "region_size must match entity_count, ghosts included");
  }

  // ---- partition_rank_of -------------------------------------------------
  {
    const soma::Region all_verts{
        .name = "v", .mesh = id.value(), .entity_set = "", .dimension = 0};
    long foreign = 0;
    for (long i = 0; i < local_verts; ++i) {
      const auto owner =
          backend.partition_rank_of(all_verts, static_cast<std::size_t>(i));
      check(owner.ok(), "partition_rank_of");
      if (!owner.ok()) break;
      check(owner.value() >= 0 && owner.value() < size,
            "partition_rank_of returned rank " +
                std::to_string(owner.value()) + " outside [0, " +
                std::to_string(size) + ")");
      const bool mine = v.is_owned(0, static_cast<std::size_t>(i));
      check(mine == (owner.value() == g_rank),
            "is_owned and partition_rank_of disagree for vertex " +
                std::to_string(i));
      if (!mine) ++foreign;
    }
    check_eq(local_verts - foreign, owned_verts,
             "owner ranks disagree with owned_count");
  }

  // ---- the conservation sum ----------------------------------------------
  // Summing over LOCAL faces would double count anything shared. Here the
  // faces happen not to be shared, but the vertices are, and the same loop
  // over dimension 0 is what a mass matrix would use.
  double owned_area = 0.0;
  for (long i = 0; i < local_faces; ++i) {
    if (v.is_owned(2, static_cast<std::size_t>(i))) {
      owned_area += v.measure(2, static_cast<std::size_t>(i));
    }
  }
  const double total_area = reduce_sum(owned_area);
  // The serial value, from moab_mesh_test.cpp. Reduction order differs, so
  // the tolerance is 1.2e-11 relative and not tighter.
  const double kSerialArea = 802651780.5417747;
  check(std::fabs(total_area - kSerialArea) < 1e-2,
        "owned area summed over ranks is " + std::to_string(total_area) +
            ", not the serial " + std::to_string(kSerialArea));
  if (g_rank == 0) {
    std::printf("[rank 0] total owned area = %.7f (serial %.7f)\n", total_area,
                kSerialArea);
  }

  // ---- global ids identify a shared vertex across ranks ------------------
  // An owned vertex and its ghost copies elsewhere share one id, which is the
  // only reason an id is usable for matching. Check the union of local ids
  // has exactly the global count, i.e. no rank invented its own numbering.
  {
    long local_max = 0;
    for (long i = 0; i < local_verts; ++i) {
      const long g = static_cast<long>(v.global_id(0, static_cast<std::size_t>(i)));
      check(g > 0, "a vertex global id is not positive");
      if (g > local_max) local_max = g;
    }
    long global_max = 0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_LONG, MPI_MAX,
                  MPI_COMM_WORLD);
    check_eq(global_max, kGlobalVerts,
             "the largest vertex global id across ranks");
  }

  long total_failures = reduce_sum(static_cast<long>(g_failures));
  if (g_rank == 0) {
    std::printf("%s: %ld failure(s) across %d rank(s)\n",
                total_failures == 0 ? "PASS" : "FAIL", total_failures, size);
  }
  MPI_Finalize();
  return total_failures == 0 ? 0 : 1;
}
