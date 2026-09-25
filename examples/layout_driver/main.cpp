// PE-layout driver.
//
// Builds a two-component coupling graph -- comp1 provides a field, comp2
// requires it, one channel -- over CLI-specified INCLUSIVE rank ranges, and
// runs it through fuse::Runner with the MPI transport. It is the same recipe
// test/parallel/runner_transport_test.cpp's TwoComponentJob fixture exercises
// with Overlap::{Disjoint,Partial,Full}, driven here from the command line
// with an arbitrary pair of ranges instead.
//
//   mpiexec -n P fuse_example_layout_driver --comp1 a:b --comp2 c:d \
//       [--coupler e:f] [--entities N] [--steps K] [--check] [--timing out.csv]
//
// --coupler e:f (design C, plans/2026-09-23-coupler-placement-and-
// migration.md) routes the "link" channel's Remap through a separately
// placed coupler on ranks [e:f]: comp1 -> coupler -> comp2, with the
// coupler->comp2 hop elided whenever [e:f] is identical to comp2's own
// range. It always attaches RemapFlags::Conservative to the channel, since
// comp1 and comp2 share one region here and nothing else would make the
// coupler's remap_on do anything (validate()'s W-CHANNEL-REMAP-ON-UNUSED).
// Not supported together with --mesh (that mode never builds this
// field-transfer graph at all).
//
// The job's PE layout must cover the whole world exactly:
// min(a,b,c,d[,e,f]) == 0 and max(...) == P-1. A violation, a reversed or
// unparseable range, a rank outside [0, P-1], or --coupler with --mesh is a
// refusal on EVERY rank, never a hang -- argv is identical on every rank
// under mpiexec, so every validation decision up to the graph build is
// deterministic and reached before any collective call, and every
// collective this binary makes afterwards (Runner::initialize, run_period,
// the --check reduction) is followed by an agreement step before the
// process acts on the local half of the answer.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/native_migration.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/partition/repartition.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/transport/transport.hpp>

#include "partitioned_test_backend.hpp"

namespace {

using fuse::Rational;
using fuse::Runner;

// ---------------------------------------------------------------------------
// A minimal Component: one field, no physics. The driver writes the source
// side's storage directly (write_source_by_global_id, below) and reads the
// destination side's directly (--check); advance() has nothing to do.
// ---------------------------------------------------------------------------
class LayoutComponent final : public fuse::Component {
 public:
  LayoutComponent(std::string name, fuse::FieldId field, std::size_t count)
      : name_(std::move(name)), field_(field), storage_(count, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph&) override {
    fuse::Layout layout;
    layout.count = storage_.size();
    return registry.bind(
        field_, fuse::FieldRef{storage_.data(), layout, fuse::MemorySpace::Host});
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();  // the driver reads/writes storage
  }

  [[nodiscard]] double*       data() noexcept { return storage_.data(); }
  [[nodiscard]] const double* data() const noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t   size() const noexcept { return storage_.size(); }

 private:
  std::string         name_;
  fuse::FieldId       field_;
  std::vector<double> storage_;
};

// ---------------------------------------------------------------------------
// CLI parsing.
// ---------------------------------------------------------------------------

struct ParsedRange {
  int            lo = 0;
  int            hi = 0;
  fuse::RankSet  ranks;
};

/// "a:b", INCLUSIVE. `*error` is set (with a named E-LAYOUT-* code) and
/// nullopt returned for: no colon, a non-integer endpoint, a missing
/// endpoint ("a:" or ":b", the empty-range case), or lo > hi (reversed).
[[nodiscard]] std::optional<ParsedRange> parse_range(const std::string& text,
                                                       std::string* error) {
  const auto colon = text.find(':');
  if (colon == std::string::npos) {
    *error = "E-LAYOUT-PARSE: '" + text + "' is not of the form a:b";
    return std::nullopt;
  }
  const std::string lo_s = text.substr(0, colon);
  const std::string hi_s = text.substr(colon + 1);
  if (lo_s.empty() || hi_s.empty()) {
    *error = "E-LAYOUT-RANGE-EMPTY: '" + text + "' is missing an endpoint";
    return std::nullopt;
  }
  int lo = 0, hi = 0;
  try {
    std::size_t pos = 0;
    lo = std::stoi(lo_s, &pos);
    if (pos != lo_s.size()) throw std::invalid_argument(lo_s);
    pos = 0;
    hi = std::stoi(hi_s, &pos);
    if (pos != hi_s.size()) throw std::invalid_argument(hi_s);
  } catch (const std::exception&) {
    *error = "E-LAYOUT-PARSE: '" + text + "' has a non-integer endpoint";
    return std::nullopt;
  }
  if (lo > hi) {
    *error = "E-LAYOUT-RANGE-REVERSED: '" + text + "' has lo > hi (" +
             std::to_string(lo) + " > " + std::to_string(hi) + ")";
    return std::nullopt;
  }
  ParsedRange r;
  r.lo    = lo;
  r.hi    = hi;
  r.ranks = fuse::RankSet{lo, hi - lo + 1};
  return r;
}

struct Args {
  std::optional<ParsedRange> comp1;
  std::optional<ParsedRange> comp2;
  std::optional<ParsedRange> coupler;
  bool        have_coupler_flag = false;  // --coupler was given, even if unparseable
  std::size_t entities = 100;
  std::int64_t steps   = 1;
  bool        check    = false;
  std::string timing_path;
  bool        have_timing = false;
  std::string parse_error;  // non-empty means refuse before doing anything else

  // --mesh switches the driver into a SEPARATE mode (run_mesh_migration_mode,
  // below): instead of the synthetic PartitionedTestBackend field-transfer
  // demonstration above, comp1 and comp2 each get a real
  // fuse::native generated mesh, optionally migrated (--migrate) from
  // comp1's decomposition onto comp2's placement via RCB. --check in this
  // mode re-verifies MESH data transfer (global cell count, measure, and id
  // uniqueness conserved across the migration) -- not the field-transfer
  // channel the other mode checks, which this mode does not build at all.
  // See docs/dev/native-mesh.md "what is deferred": matching migrated
  // entities back to the coupling graph's field-transfer machinery
  // (arbitrary post-migration entity correspondence through MpiTransport)
  // is not implemented.
  std::optional<std::string> mesh_spec;
  bool        migrate = false;
};

[[nodiscard]] Args parse_args(int argc, char** argv) {
  Args args;
  std::optional<std::string> comp1_text, comp2_text, coupler_text;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        args.parse_error =
            "E-LAYOUT-PARSE: '" + std::string(flag) + "' needs a value";
        return std::nullopt;
      }
      return std::string(argv[++i]);
    };
    if (a == "--comp1") {
      auto v = next("--comp1");
      if (!v) return args;
      comp1_text = *v;
    } else if (a == "--comp2") {
      auto v = next("--comp2");
      if (!v) return args;
      comp2_text = *v;
    } else if (a == "--coupler") {
      args.have_coupler_flag = true;
      auto v = next("--coupler");
      if (!v) return args;
      coupler_text = *v;
    } else if (a == "--entities") {
      auto v = next("--entities");
      if (!v) return args;
      try {
        args.entities = static_cast<std::size_t>(std::stoll(*v));
      } catch (const std::exception&) {
        args.parse_error = "E-LAYOUT-PARSE: --entities '" + *v + "' is not an integer";
        return args;
      }
    } else if (a == "--steps") {
      auto v = next("--steps");
      if (!v) return args;
      try {
        args.steps = std::stoll(*v);
      } catch (const std::exception&) {
        args.parse_error = "E-LAYOUT-PARSE: --steps '" + *v + "' is not an integer";
        return args;
      }
    } else if (a == "--check") {
      args.check = true;
    } else if (a == "--mesh") {
      auto v = next("--mesh");
      if (!v) return args;
      args.mesh_spec = *v;
    } else if (a == "--migrate") {
      args.migrate = true;
    } else if (a == "--timing") {
      auto v = next("--timing");
      if (!v) return args;
      args.timing_path = *v;
      args.have_timing = true;
    } else {
      args.parse_error = "E-LAYOUT-PARSE: unrecognised argument '" + a + "'";
      return args;
    }
  }
  if (!comp1_text) {
    args.parse_error = "E-LAYOUT-PARSE: --comp1 a:b is required";
    return args;
  }
  if (!comp2_text) {
    args.parse_error = "E-LAYOUT-PARSE: --comp2 a:b is required";
    return args;
  }
  args.comp1 = parse_range(*comp1_text, &args.parse_error);
  if (!args.comp1) return args;
  args.comp2 = parse_range(*comp2_text, &args.parse_error);
  if (!args.comp2) return args;
  if (coupler_text) {
    args.coupler = parse_range(*coupler_text, &args.parse_error);
    if (!args.coupler) return args;
  }
  if (args.migrate && !args.mesh_spec) {
    args.parse_error = "E-LAYOUT-PARSE: --migrate needs --mesh <spec>";
    return args;
  }
  return args;
}

/// disjoint / partially overlapping / identical, spelled out for a rank-0
/// print rather than an enumerator: the driver reports exactly the comp1 vs
/// comp2 shape here regardless of --coupler, since that is the shape the
/// non-coupler tests already print and diff against.
[[nodiscard]] std::string_view shape_of(const fuse::RankSet& a,
                                        const fuse::RankSet& b) {
  if (a.same_as(b)) return "identical";
  if (!a.overlaps(b)) return "disjoint";
  return "partially overlapping";
}

/// One MPI_Allreduce(MAX) so every rank returns the SAME exit code -- the
/// discipline test/support/fuse_test_main.cpp uses to fail a job uniformly
/// instead of letting one rank return while its peers block in the next
/// collective. Safe to call unconditionally, including on the deterministic
/// argument-validation paths that are reached before any other collective:
/// argv is identical on every rank under mpiexec, so `local_status` already
/// agrees there, and the reduction costs one cheap round trip.
int agree_and_exit(int local_status) {
  int agreed = local_status;
  MPI_Allreduce(&local_status, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return agreed;
}

struct Stat {
  double min_s = 0.0, mean_s = 0.0, max_s = 0.0;
};

/// Collective: every rank must call this, even though only rank 0 uses the
/// result.
[[nodiscard]] Stat reduce_stat(double local_seconds) {
  int world = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  double mn = 0.0, mx = 0.0, sum = 0.0;
  MPI_Reduce(&local_seconds, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_seconds, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_seconds, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  Stat s;
  s.min_s  = mn;
  s.max_s  = mx;
  s.mean_s = sum / static_cast<double>(world);
  return s;
}

struct TimingRow {
  std::string phase;
  Stat        stat;
};

void write_timing_csv(const std::string& path, int world, const std::string& layout,
                     std::size_t entities, const std::vector<TimingRow>& rows,
                     bool* failed) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    std::cerr << "E-LAYOUT-TIMING: could not open '" << path << "' for writing\n";
    *failed = true;
    return;
  }
  out << "# fuse_example_layout_driver timing CSV\n"
      << "# schema: phase,ranks,layout,n,min_s,mean_s,max_s\n"
      << "# phase  : \"initialize\" (mesh load + Runner::initialize, includes\n"
      << "#          CommGraph build) or \"period_<k>\" for k = 1..steps\n"
      << "# ranks  : MPI world size this row was measured at\n"
      << "# layout : \"c1_<lo>-<hi>_c2_<lo>-<hi>\" (inclusive 0-based rank ranges)\n"
      << "# n      : total entities in the shared region (--entities)\n"
      << "# min_s/mean_s/max_s : wall-clock seconds for this phase, reduced with\n"
      << "#          MPI_MIN / (MPI_SUM / ranks) / MPI_MAX over every rank\n"
      << "phase,ranks,layout,n,min_s,mean_s,max_s\n";
  out.precision(9);
  for (const auto& row : rows) {
    out << row.phase << ',' << world << ',' << layout << ',' << entities << ','
        << row.stat.min_s << ',' << row.stat.mean_s << ',' << row.stat.max_s << '\n';
  }
  if (!out) {
    std::cerr << "E-LAYOUT-TIMING: write to '" << path << "' failed\n";
    *failed = true;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// --mesh / --migrate: a separate mode. See the Args::mesh_spec doc comment.
// ---------------------------------------------------------------------------

/// The topological dimension `fuse::native::generate_native_mesh("<kind>:...")`
/// produces, read from the kind alone (before any rank slicing happens) --
/// needed here to build a matching EMPTY mesh (`NativeMeshBuilder(dim)`) on
/// a rank that owns none of a placement, since `NativeMeshView` requires
/// every mesh, even an empty one, to declare its dimension up front.
[[nodiscard]] std::optional<int> infer_native_mesh_dim(const std::string& spec) {
  const auto colon = spec.find(':');
  const std::string kind = (colon == std::string::npos) ? spec : spec.substr(0, colon);
  if (kind == "line") return 1;
  if (kind == "quad" || kind == "tri" || kind == "cubedsphere") return 2;
  if (kind == "hexbox" || kind == "tetbox" || kind == "prismpyramidmix" ||
      kind == "polyhedral") {
    return 3;
  }
  return std::nullopt;
}

/// Builds this rank's slice of `spec` for a placement of `placement_size`
/// ranks, using LOCAL rank-within-the-placement numbering (so the mesh's
/// global ids stay independent of which world ranks the placement actually
/// occupies) -- or an empty mesh of the right dimension when this rank does
/// not belong to `ranks`.
[[nodiscard]] fuse::Result<std::unique_ptr<fuse::native::NativeMeshView>> build_placement_mesh(
    const std::string& spec, const fuse::RankSet& ranks, int world_rank, int dim) {
  if (ranks.contains(world_rank)) {
    const int local_rank = ranks.index_of(world_rank);
    auto arrays = fuse::native::generate_native_mesh(spec, local_rank,
                                                      static_cast<int>(ranks.size()));
    if (!arrays.ok()) return arrays.error();
    return fuse::native::NativeMeshView::create_from_host(arrays.value(), world_rank);
  }
  fuse::native::NativeMeshBuilder builder(dim);
  auto arrays = builder.build();
  if (!arrays.ok()) return arrays.error();
  return fuse::native::NativeMeshView::create_from_host(arrays.value(), world_rank);
}

/// The whole --mesh/--migrate mode: build comp1's placement mesh, build (or
/// migrate onto) comp2's, and under --check report the conservation
/// properties migration promises (global cell count, total measure, unique
/// ids) rather than the field-transfer channel the OTHER mode's --check
/// verifies -- see Args::mesh_spec's own comment on why the two are
/// separate claims.
int run_mesh_migration_mode(const Args& args, int rank, int world) {
  const auto dim = infer_native_mesh_dim(*args.mesh_spec);
  if (!dim) {
    if (rank == 0) {
      std::cerr << "E-LAYOUT-MESH: unrecognised --mesh kind in '" << *args.mesh_spec << "'\n";
    }
    return agree_and_exit(1);
  }

  auto comp1_view = build_placement_mesh(*args.mesh_spec, args.comp1->ranks, rank, *dim);
  if (!comp1_view.ok()) {
    if (rank == 0) std::cerr << "comp1 mesh failed: " << comp1_view.error().message << '\n';
    return agree_and_exit(1);
  }

  const fuse::partition::CommHandle world_comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  std::unique_ptr<fuse::native::NativeMeshView> comp2_view;

  if (args.migrate) {
    const auto& comp2_ranks = args.comp2->ranks;
    const int comp2_size = static_cast<int>(comp2_ranks.size());

    auto cd = fuse::partition::centroids_of(*comp1_view.value(), world_comm);
    if (!cd.ok()) {
      if (rank == 0) std::cerr << "centroids_of failed: " << cd.error().message << '\n';
      return agree_and_exit(1);
    }
    using Space = Kokkos::DefaultHostExecutionSpace;
    const std::size_t n = cd.value().centroids.extent(0);
    Kokkos::View<fuse::partition::Weight*, Space::memory_space> weights(
        Kokkos::view_alloc("layout_driver.weights", Kokkos::WithoutInitializing), n);
    Kokkos::deep_copy(weights, fuse::partition::Weight{1});

    fuse::partition::RcbOptions rcb_opts;
    rcb_opts.num_parts = comp2_size;  // RCB targets comp2's OWN rank count,
                                      // not the world's -- remapped below.
    rcb_opts.comm = world_comm;
    auto rcb_result = fuse::partition::rcb_partition<Space>(rcb_opts, cd.value().centroids,
                                                             weights, cd.value().global_ids);

    std::vector<int> dest(n);
    auto part_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rcb_result.part_id);
    for (std::size_t i = 0; i < n; ++i) {
      dest[i] = comp2_ranks.ranks()[static_cast<std::size_t>(part_host(i))];
    }

    auto migrated = fuse::native::migrate(*comp1_view.value(), dest, world_comm);
    if (!migrated.ok()) {
      if (rank == 0) std::cerr << "migrate failed: " << migrated.error().message << '\n';
      return agree_and_exit(1);
    }
    comp2_view = std::move(migrated.value());
  } else {
    auto v = build_placement_mesh(*args.mesh_spec, args.comp2->ranks, rank, *dim);
    if (!v.ok()) {
      if (rank == 0) std::cerr << "comp2 mesh failed: " << v.error().message << '\n';
      return agree_and_exit(1);
    }
    comp2_view = std::move(v.value());
  }

  int local_status = 0;
  if (args.check) {
    const unsigned long local_c1 = comp1_view.value()->owned_count(*dim);
    const unsigned long local_c2 = comp2_view->owned_count(*dim);
    double local_m1 = 0.0, local_m2 = 0.0;
    for (std::size_t c = 0; c < comp1_view.value()->cell_count(); ++c) {
      if (comp1_view.value()->is_owned(*dim, c)) local_m1 += comp1_view.value()->measure(*dim, c);
    }
    for (std::size_t c = 0; c < comp2_view->cell_count(); ++c) {
      if (comp2_view->is_owned(*dim, c)) local_m2 += comp2_view->measure(*dim, c);
    }
    unsigned long total_c1 = 0, total_c2 = 0;
    double total_m1 = 0.0, total_m2 = 0.0;
    MPI_Allreduce(&local_c1, &total_c1, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_c2, &total_c2, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_m1, &total_m1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_m2, &total_m2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "check: comp1 cells=" << total_c1 << " measure=" << total_m1
                << "; comp2 cells=" << total_c2 << " measure=" << total_m2 << '\n';
    }
    const bool cells_conserved = (args.migrate ? (total_c1 == total_c2) : true);
    const bool measure_ok = std::abs(total_m1 - total_m2) < 1e-9 * std::max(1.0, total_m1);
    if (!cells_conserved || (args.migrate && !measure_ok)) {
      if (rank == 0) std::cerr << "FAIL: mesh data transfer did not conserve cells/measure\n";
      local_status = 1;
    }
  }
  return agree_and_exit(local_status);
}

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  // Kokkos is initialised here and for the whole of main, before any argument
  // is parsed (the guard also strips --kokkos-* arguments). The native mesh,
  // the RCB input and its part ids are all Kokkos views, and allocating one
  // before initialize() aborts. The transport-only mode never allocates one,
  // which is how the first eleven layout tests passed while every --migrate
  // test failed. Declared after MPI_Init and in main's scope, so every view
  // built in a mode function is destroyed before the guard finalises Kokkos.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  const Args args = parse_args(argc, argv);
  if (!args.parse_error.empty()) {
    if (rank == 0) std::cerr << args.parse_error << '\n';
    return agree_and_exit(1);
  }

  // Every provided range, so the bounds check below (and later the coupler
  // refusal) sees the coupler's numbers exactly as it sees comp1/comp2's.
  std::vector<const ParsedRange*> all_ranges = {&*args.comp1, &*args.comp2};
  if (args.coupler) all_ranges.push_back(&*args.coupler);

  for (const ParsedRange* r : all_ranges) {
    if (r->lo < 0 || r->hi > world - 1) {
      if (rank == 0) {
        std::cerr << "E-LAYOUT-RANK-OUT-OF-RANGE: [" << r->lo << ':' << r->hi
                  << "] is not within [0:" << (world - 1) << "]\n";
      }
      return agree_and_exit(1);
    }
  }

  int global_min = all_ranges.front()->lo, global_max = all_ranges.front()->hi;
  for (const ParsedRange* r : all_ranges) {
    global_min = std::min(global_min, r->lo);
    global_max = std::max(global_max, r->hi);
  }
  if (global_min != 0 || global_max != world - 1) {
    if (rank == 0) {
      std::cerr << "E-LAYOUT-BOUNDS: the layout covers [" << global_min << ':'
                << global_max << "], not [0:" << (world - 1)
                << "] -- every rank of the job must belong to some placement\n";
    }
    return agree_and_exit(1);
  }

  if (rank == 0) {
    std::cout << "shape(comp1, comp2): "
              << shape_of(args.comp1->ranks, args.comp2->ranks) << '\n';
  }

  // --coupler is still refused in --mesh/--migrate mode: that mode never
  // builds the field-transfer coupling graph at all (see Args::mesh_spec's
  // comment), so there is no channel for a coupler placement to attach to.
  if (args.have_coupler_flag && args.mesh_spec) {
    if (rank == 0) {
      std::cerr << "E-LAYOUT-COUPLER: --coupler is not supported with --mesh\n";
    }
    return agree_and_exit(1);
  }

  if (args.mesh_spec) {
    return run_mesh_migration_mode(args, rank, world);
  }

  // ------------------------------------------------------------- the graph
  const int dim = 1;
  fuse::CouplingGraph graph;
  const auto mesh =
      graph.add_mesh(fuse::MeshDecl{.name = "shared", .source = "layout-driver"});
  const auto region =
      graph.add_region(fuse::Region{.name = "r", .mesh = mesh, .dimension = dim});

  const auto comp1_id = graph.add_component(fuse::ComponentDecl{
      .name = "comp1", .ranks = args.comp1->ranks, .interval = Rational{1}});
  const auto comp2_id = graph.add_component(fuse::ComponentDecl{
      .name = "comp2", .ranks = args.comp2->ranks, .interval = Rational{1}});
  const auto src_field = graph.add_field(
      comp1_id, fuse::FieldSpec{.name = "out", .region = region,
                                .units = fuse::units::kKelvin,
                                .intent = fuse::Intent::Provides});
  const auto dst_field = graph.add_field(
      comp2_id, fuse::FieldSpec{.name = "in", .region = region,
                                .units = fuse::units::kKelvin,
                                .intent = fuse::Intent::Requires});
  const auto link = graph.add_channel("link", src_field, dst_field);
  if (args.coupler) {
    // remap() with no flags: the two components' regions are the SAME
    // region (both fields share `region` above), so nothing about this
    // channel would otherwise emit a Remap at all -- could_ever_emit_remap
    // (fuse/model/remap_kind.hpp) would be false and remap_on would be
    // W-CHANNEL-REMAP-ON-UNUSED. Conservative asks for one unconditionally,
    // so --coupler always exercises the coupler-placement route it is
    // there to demonstrate, on every layout shape.
    const auto coupler_id = graph.add_coupler(
        fuse::CouplerDecl{.name = "coupler", .ranks = args.coupler->ranks});
    graph.channel(link).remap(fuse::RemapFlags::Conservative).remap_on(coupler_id);
  }

  PartitionedTestBackend backend(args.comp1->ranks, args.comp2->ranks,
                                 args.entities, dim);
  backend.set_placements({args.comp1->ranks, args.comp2->ranks});
  if (const auto r = backend.load(graph.mesh(mesh)); !r.ok()) {
    if (rank == 0) std::cerr << "mesh load failed: " << r.error().message << '\n';
    return agree_and_exit(1);
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) {
    if (rank == 0) std::cerr << lowered.diags.to_string();
    return agree_and_exit(1);
  }

  const fuse::ClockSet clocks{graph};
  const auto sched =
      schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, backend));
  if (!sched.ok()) {
    if (rank == 0) std::cerr << sched.diags.to_string();
    return agree_and_exit(1);
  }

  const auto size_of = [&](std::uint32_t cid) -> std::size_t {
    const auto n =
        backend.placed_region_size(graph.region(region), fuse::ComponentId{cid});
    return n.ok() ? n.value() : 0;
  };

  fuse::ComponentSet components;
  components.add(std::make_unique<LayoutComponent>("comp1", src_field, size_of(0)));
  components.add(std::make_unique<LayoutComponent>("comp2", dst_field, size_of(1)));

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (const auto r = c.register_fields(registry, graph); !r.ok()) {
      if (rank == 0) std::cerr << "register_fields failed: " << r.error().message << '\n';
      return agree_and_exit(1);
    }
  }

  Runner runner{graph, lowered.dag, sched.plan, backend};
  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-LAYOUT-TRANSPORT", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };

  std::vector<TimingRow> timing;
  const double t_init0 = MPI_Wtime();
  const auto init = runner.initialize(components, registry, cfg);
  const double t_init1 = MPI_Wtime();
  {
    const int local_failed = init.ok() ? 0 : 1;
    if (args.have_timing) timing.push_back({"initialize", reduce_stat(t_init1 - t_init0)});
    if (!init.ok() && rank == 0) {
      std::cerr << "initialize failed: " << init.error().code << ": "
                << init.error().message << '\n';
    }
    int agreed = local_failed;
    MPI_Allreduce(&local_failed, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (agreed != 0) return agree_and_exit(1);
  }

  // Source side: encode the global id in the value, exactly as
  // runner_transport_test.cpp's write_source_by_global_id does, so --check
  // can tell "delivered" apart from "coincidentally already correct".
  auto* comp1_component = dynamic_cast<LayoutComponent*>(components.find("comp1"));
  auto* comp2_component = dynamic_cast<LayoutComponent*>(components.find("comp2"));
  {
    const auto v = backend.placed_view(graph.region(region).mesh, fuse::ComponentId{0});
    if (v.ok()) {
      for (std::size_t i = 0; i < comp1_component->size(); ++i) {
        comp1_component->data()[i] = static_cast<double>(v.value()->global_id(dim, i));
      }
    }
  }

  for (std::int64_t k = 1; k <= args.steps; ++k) {
    const double t0 = MPI_Wtime();
    const auto run = runner.run_period(components);
    const double t1 = MPI_Wtime();
    const int local_failed = run.ok() ? 0 : 1;
    if (args.have_timing) {
      timing.push_back({"period_" + std::to_string(k), reduce_stat(t1 - t0)});
    }
    if (!run.ok() && rank == 0) {
      std::cerr << "run_period failed at step " << k << ": " << run.error().code
                << ": " << run.error().message << '\n';
    }
    int agreed = local_failed;
    MPI_Allreduce(&local_failed, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (agreed != 0) return agree_and_exit(1);
  }

  int local_status = 0;

  if (args.check) {
    // Mirrors expect_every_destination_entity_delivered
    // (runner_transport_test.cpp): every destination entity must hold its
    // own global id, and the destination's own decomposition must cover the
    // region exactly once.
    unsigned long local_ok = 0, local_n = 0;
    if (args.comp2->ranks.contains(rank)) {
      const auto v = backend.placed_view(graph.region(region).mesh, fuse::ComponentId{1});
      if (v.ok()) {
        local_n = comp2_component->size();
        for (std::size_t i = 0; i < comp2_component->size(); ++i) {
          if (comp2_component->data()[i] ==
              static_cast<double>(v.value()->global_id(dim, i))) {
            ++local_ok;
          }
        }
      }
    }
    unsigned long total_ok = 0, total_n = 0;
    MPI_Allreduce(&local_ok, &total_ok, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_n, &total_n, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "check: " << total_ok << "/" << total_n << " destination entities"
                   " delivered (job entities = " << args.entities << ")\n";
    }
    if (total_n != args.entities || total_ok != args.entities) {
      if (rank == 0) {
        std::cerr << "FAIL: delivered " << total_ok << "/" << total_n
                  << " but the job has " << args.entities << " entities\n";
      }
      local_status = 1;
    }
  }

  if (args.have_timing) {
    const std::string layout = "c1_" + std::to_string(args.comp1->lo) + "-" +
                               std::to_string(args.comp1->hi) + "_c2_" +
                               std::to_string(args.comp2->lo) + "-" +
                               std::to_string(args.comp2->hi);
    if (rank == 0) {
      bool failed = false;
      write_timing_csv(args.timing_path, world, layout, args.entities, timing, &failed);
      if (failed) local_status = 1;
    }
  }

  return agree_and_exit(local_status);
}
