// R3, half two: conservative remap that is CORRECT on a partitioned read.
//
// The pair is deliberately MISALIGNED -- MOAB's 3456-cell cubed sphere with
// its own 8 partition sets against a 1536-cell partner with 4, written by
// test/tools/make_partitioned_sphere.cpp. Nothing about the two distributions
// lines up, so on more than one rank most of a rank's target cells are covered
// by source cells that live somewhere else. That is the case the whole halo
// exists for, and the case that a pair of identically-partitioned meshes
// cannot exercise (there, a partition-local intersection is accidentally
// right, which is why remap_parallel_test.cpp -- which uses one -- can prove
// the guard FIRES but cannot prove it PREVENTS anything).
//
// WHAT IS ESTABLISHED:
//
//   1. `build_remap_halo` widens the source mesh. `entity_count(2)` grows by
//      the adopted cells, the adopted cells are NOT owned, and the owned count
//      is unchanged -- so `region_size` widens with it and no reduction starts
//      double counting.
//   2. `exchange_remap_halo` puts the right values in the halo slots. Checked
//      against a function of the global id, so a slot filled from the wrong
//      owner is caught rather than averaged away.
//   3. `accumulate_remap_halo` is its exact reverse.
//   4. CONSERVATION IN PARALLEL. The global integral of a varying field
//      survives the remap to 1e-13 relative, at 1, 2 and 4 ranks, summed over
//      OWNED cells only. This is the deliverable.
//   5. THE HALO IS LOAD BEARING, and this is CI-enforced without mutating
//      anything: dropping the halo COLUMNS from the very matrix just verified
//      gives exactly the matrix a partition-local intersection would have
//      produced, and the same integral then loses two thirds of itself at two
//      ranks and seven eighths at four. Those are the numbers docs/HANDOFF.md
//      quotes; they are asserted here so they cannot rot.
//   6. Without the halo the build is refused, on the same misaligned pair.
//
// Not a GoogleTest binary, for the reasons moab_parallel_test.cpp gives.

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

double reduce_sum(double v) {
  double out = 0.0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return out;
}

std::string describe(const soma::Error& e) { return e.code + ": " + e.message; }

/// The field being transferred. A function of position, not a constant: a
/// constant survives a matrix that is wrong by a permutation, and survives
/// dropping a column whose neighbour picks up the slack.
double field_at(const soma::Point& p, double radius) {
  return 2.0 + p.x / radius + 3.0 * p.y / radius - 0.5 * p.z / radius;
}

/// The integral over the cells this rank OWNS. Ghost and halo cells are
/// another rank's copy; including them is the double count `is_owned` exists
/// to prevent.
double owned_integral(const soma::MeshView& v, int dim,
                      const std::vector<double>& f) {
  double s = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (v.is_owned(dim, i)) s += f[i] * v.measure(dim, i);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  if (argc < 3) {
    if (g_rank == 0) {
      std::fprintf(stderr,
                   "usage: remap_halo_parallel_test <source.h5m> "
                   "<partner.h5m>\n");
    }
    MPI_Finalize();
    return 2;
  }

  soma::CouplingGraph graph;
  const soma::MeshId ms =
      graph.add_mesh(soma::MeshDecl{.name = "src", .source = argv[1]});
  const soma::MeshId mt =
      graph.add_mesh(soma::MeshDecl{.name = "tgt", .source = argv[2]});
  const soma::RegionId rs = graph.add_region(
      soma::Region{.name = "rsrc", .mesh = ms, .dimension = 2});
  const soma::RegionId rt = graph.add_region(
      soma::Region{.name = "rtgt", .mesh = mt, .dimension = 2});

  soma::MoabMeshBackend backend;
  {
    const auto r = backend.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    check(r.ok(), "set_communicator");
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

  const auto sid = backend.mesh_id("src");
  const auto tid = backend.mesh_id("tgt");
  check(sid.ok() && tid.ok(), "mesh_id");
  if (!sid.ok() || !tid.ok()) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const soma::MeshView& sv = *backend.view(sid.value()).value();
  const soma::MeshView& tv = *backend.view(tid.value()).value();

  const long kGlobalSrc = 3456;
  const long kGlobalTgt = 1536;
  const std::size_t src_before = sv.entity_count(2);
  const std::size_t src_owned_before = sv.owned_count(2);
  check_eq(reduce_sum(static_cast<long>(src_owned_before)), kGlobalSrc,
           "owned source cells summed over ranks");
  check_eq(reduce_sum(static_cast<long>(tv.owned_count(2))), kGlobalTgt,
           "owned target cells summed over ranks");

  soma::RemapPlan plan;
  plan.src = rs;
  plan.tgt = rt;
  plan.method = "intx";
  plan.flags = soma::RemapFlags::Conservative;

  // ---- 6. without a halo, refused (on the misaligned pair) ---------------
  if (g_size > 1) {
    const auto w0 = soma::build_weights(plan, graph, backend);
    check(!w0.ok(), "build_weights accepted a misaligned partitioned pair "
                    "with no halo");
    if (!w0.ok()) {
      check(w0.error().code == "E-REMAP-PARALLEL",
            "expected E-REMAP-PARALLEL, got " + describe(w0.error()));
    }
  }

  // ---- 1. build the halo -------------------------------------------------
  const auto adopted = soma::build_remap_halo(plan, graph, backend);
  check(adopted.ok(), "build_remap_halo: " +
                          (adopted.ok() ? std::string{}
                                        : describe(adopted.error())));
  if (!adopted.ok()) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const std::size_t halo = adopted.value();
  std::printf("[rank %d/%d] source %zu owned -> %zu local (+%zu halo); "
              "target %zu\n",
              g_rank, g_size, src_owned_before, sv.entity_count(2), halo,
              tv.entity_count(2));
  std::fflush(stdout);

  check_eq(static_cast<long>(sv.entity_count(2)),
           static_cast<long>(src_before + halo),
           "entity_count(2) did not grow by exactly the adopted cells");
  check_eq(static_cast<long>(sv.owned_count(2)),
           static_cast<long>(src_owned_before),
           "adopting halo cells changed the OWNED count; a halo cell is "
           "another rank's and must never be owned here");
  check_eq(static_cast<long>(backend.remap_halo_size(sid.value(), 2)),
           static_cast<long>(halo), "remap_halo_size");
  check(backend.has_remap_halo(sid.value(), tid.value(), 2),
        "has_remap_halo is false right after building one");
  // `region_size` is what the buffer plan uses. If it did not widen with
  // entity_count, every buffer would be too small and the matrix would not fit.
  const soma::Region src_region{
      .name = "rsrc", .mesh = ms, .entity_set = "", .dimension = 2};
  const auto rsz = backend.region_size(src_region);
  check(rsz.ok() && rsz.value() == sv.entity_count(2),
        "region_size did not follow entity_count after the halo was built");
  if (g_size > 1) {
    check(halo > 0,
          "no cells were adopted on a deliberately misaligned pair, so this "
          "run does not exercise the halo at all");
    check(reduce_sum(static_cast<long>(sv.entity_count(2))) > kGlobalSrc,
          "the local counts still sum to the global one, so no cell is "
          "duplicated anywhere and there is no halo to test");
  }
  // Idempotent.
  const auto again = soma::build_remap_halo(plan, graph, backend);
  check(again.ok() && again.value() == halo,
        "build_remap_halo is not idempotent");
  check_eq(static_cast<long>(sv.entity_count(2)),
           static_cast<long>(src_before + halo),
           "a second build_remap_halo changed the local count");

  // ---- 2 and 3. the exchange moves the right values ----------------------
  {
    std::vector<double> b(sv.entity_count(2), 0.0);
    // A value only the OWNER can know, keyed on the global id, so a slot
    // filled from the wrong owner is a mismatch rather than a plausible number.
    for (std::size_t i = 0; i < b.size(); ++i) {
      b[i] = sv.is_owned(2, i)
                 ? 1000.0 + static_cast<double>(sv.global_id(2, i))
                 : -1.0;
    }
    const auto e = backend.exchange_remap_halo(sid.value(), 2, b.data(), 1);
    check(e.ok(), "exchange_remap_halo: " +
                      (e.ok() ? std::string{} : describe(e.error())));
    long wrong = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      const double want = 1000.0 + static_cast<double>(sv.global_id(2, i));
      if (std::fabs(b[i] - want) > 1e-9) ++wrong;
    }
    check_eq(wrong, 0, "halo slots not filled with their owner's value");

    // The reverse: one unit per copy, accumulated onto the owner. An owned
    // cell held as a halo copy by k other ranks must come back with k + 1.
    std::vector<double> c(sv.entity_count(2), 1.0);
    const auto ac = backend.accumulate_remap_halo(sid.value(), 2, c.data(), 1);
    check(ac.ok(), "accumulate_remap_halo: " +
                       (ac.ok() ? std::string{} : describe(ac.error())));
    double total = 0.0;
    long halo_nonzero = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
      if (sv.is_owned(2, i)) {
        total += c[i];
        check(c[i] >= 1.0, "an owned cell lost its own contribution");
      } else if (c[i] != 0.0) {
        ++halo_nonzero;
      }
    }
    check_eq(halo_nonzero, 0,
             "accumulate_remap_halo left values in the halo slots, so a "
             "second call would add them again");
    // Everything that existed anywhere is now on an owner, exactly once.
    check_eq(static_cast<long>(reduce_sum(total) + 0.5),
             reduce_sum(static_cast<long>(sv.entity_count(2))),
             "the accumulated total is not the number of local copies");
  }

  // ---- the high-order builder stays refused, halo or no halo -------------
  //
  // The halo makes the FIRST-ORDER supermesh correct in parallel; it does not
  // make the high-order backend correct, because the surface it is handed is
  // then a piece of the domain plus a fringe and its stencils near that fringe
  // are unmeasured. That refusal must survive the halo existing -- this is the
  // leg that would go green if someone decided the halo "fixed everything".
  if (g_size > 1) {
    soma::RemapPlan hi;
    hi.src = rs;
    hi.tgt = rt;
    hi.method = "tempest";
    hi.src_order = 2;
    hi.tgt_order = 2;
    const auto wh = soma::build_weights(hi, graph, backend);
    check(!wh.ok(), "the high-order builder accepted a partitioned read once "
                    "a halo existed");
    if (!wh.ok()) {
      check(wh.error().code == "E-REMAP-PARALLEL",
            "the high-order builder refused with " + describe(wh.error()) +
                " rather than E-REMAP-PARALLEL");
    }
  }

  // ---- 4. CONSERVATION IN PARALLEL ---------------------------------------
  const auto w = soma::build_weights(plan, graph, backend);
  check(w.ok(), "build_weights with a halo: " +
                    (w.ok() ? std::string{} : describe(w.error())));
  if (!w.ok()) {
    const long f = reduce_sum(static_cast<long>(g_failures));
    if (g_rank == 0) std::printf("FAIL: %ld failure(s)\n", f);
    MPI_Finalize();
    return 1;
  }
  const soma::RemapWeights& m = w.value();
  check_eq(static_cast<long>(m.n_cols), static_cast<long>(sv.entity_count(2)),
           "n_cols is not the source mesh's local entity count");
  check_eq(static_cast<long>(m.n_rows), static_cast<long>(tv.entity_count(2)),
           "n_rows is not the target mesh's local entity count");

  std::printf("[rank %d] nnz=%zu rows=%zu cols=%zu rowdef=%.3e coldef=%.3e "
              "srccov=%.14f tgtcov=%.14f\n",
              g_rank, m.nnz(), m.n_rows, m.n_cols, m.max_row_defect,
              m.max_col_defect, m.min_src_coverage, m.min_tgt_coverage);
  std::fflush(stdout);

  // The audit is reduced over the communicator, so every rank must report the
  // same numbers. A per-rank audit is how E-REMAP-COVERAGE fires on some ranks
  // and not others and the job hangs instead of failing.
  {
    double v[4] = {m.min_src_coverage, m.min_tgt_coverage, m.max_row_defect,
                   m.max_col_defect};
    double lo[4], hi[4];
    MPI_Allreduce(v, lo, 4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(v, hi, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    for (int k = 0; k < 4; ++k) {
      check(std::fabs(hi[k] - lo[k]) <= 0.0,
            "audit field " + std::to_string(k) +
                " differs between ranks (" + std::to_string(lo[k]) + " vs " +
                std::to_string(hi[k]) +
                "); a per-rank verdict deadlocks the job");
    }
  }
  check(m.min_src_coverage > 1.0 - 1e-9,
        "source coverage " + std::to_string(m.min_src_coverage) +
            " -- with a halo every source cell must be fully seen");
  check(m.max_col_defect < 1e-12,
        "column (conservation) defect " + std::to_string(m.max_col_defect));

  double radius = 0.0;
  for (std::size_t i = 0; i < sv.entity_count(2); ++i) {
    const soma::Point c = sv.centroid(2, i);
    radius = std::max(radius,
                      std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z));
  }
  {
    double r = radius;
    MPI_Allreduce(&r, &radius, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }

  std::vector<double> f(m.n_cols, 0.0);
  for (std::size_t i = 0; i < m.n_cols; ++i) {
    if (sv.is_owned(2, i)) f[i] = field_at(sv.centroid(2, i), radius);
  }
  // THE HALO SLOTS ARE FILLED HERE, and this is the step that makes the
  // matrix usable: without it the off-rank columns multiply whatever happened
  // to be in the buffer.
  {
    const auto e = backend.exchange_remap_halo(sid.value(), 2, f.data(), 1);
    check(e.ok(), "exchange before the matvec");
  }

  std::vector<double> g(m.n_rows, 0.0);
  check(soma::apply_weights(m, f.data(), g.data(), 1).ok(), "apply_weights");

  const double before = reduce_sum(owned_integral(sv, 2, f));
  const double after = reduce_sum(owned_integral(tv, 2, g));
  const double rel = std::fabs(after - before) / std::fabs(before);
  if (g_rank == 0) {
    std::printf("[rank 0] np=%d integral before=%.15e after=%.15e rel=%.3e\n",
                g_size, before, after, rel);
    std::fflush(stdout);
  }
  check(rel < 1e-13,
        "the global integral moved by " + std::to_string(rel) +
            " relative; a conservative remap on a partitioned read must "
            "preserve it");

  // ---- 5. the halo is load bearing, measured without mutating anything ---
  //
  // Dropping every column that names a cell this rank does not own turns the
  // verified matrix into EXACTLY the matrix a partition-local intersection
  // would have built -- same rows, same target cells, only the off-rank
  // sources missing. So the loss below is the loss the guard prevents, and it
  // is measured against the same run that just conserved.
  if (g_size > 1) {
    soma::RemapWeights local = m;
    std::size_t keep = 0;
    for (std::size_t k = 0; k < local.val.size(); ++k) {
      const auto c = static_cast<std::size_t>(local.col[k]);
      if (!sv.is_owned(2, c)) continue;
      local.row[keep] = local.row[k];
      local.col[keep] = local.col[k];
      local.val[keep] = local.val[k];
      ++keep;
    }
    local.row.resize(keep);
    local.col.resize(keep);
    local.val.resize(keep);

    std::vector<double> gl(local.n_rows, 0.0);
    check(soma::apply_weights(local, f.data(), gl.data(), 1).ok(),
          "apply_weights (halo columns dropped)");
    const double after_local = reduce_sum(owned_integral(tv, 2, gl));
    const double loss = (before - after_local) / before;
    if (g_rank == 0) {
      std::printf("[rank 0] np=%d WITHOUT the halo columns: after=%.15e, "
                  "%.4f%% of the quantity lost\n",
                  g_size, after_local, 100.0 * loss);
      std::fflush(stdout);
    }
    // The measured figures are 66.7% at two ranks and 87.5% at four on this
    // fixture. Asserted as bands, not to the digit: the exact value depends on
    // how MOAB assigns partition sets to ranks, and pinning it to 1e-6 would
    // make an unrelated MOAB change look like a conservation bug. The bands
    // are still nowhere near "conserved".
    const double floor_loss = (g_size >= 4) ? 0.80 : 0.55;
    check(loss > floor_loss,
          "dropping the off-rank columns lost only " +
              std::to_string(100.0 * loss) +
              "% of the quantity at " + std::to_string(g_size) +
              " ranks, so this fixture's two partitions are not actually "
              "misaligned and the halo is not being tested");
    check(loss < 0.999, "the local-only map transferred essentially nothing, "
                        "which suggests the fixture, not the halo, is wrong");
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
