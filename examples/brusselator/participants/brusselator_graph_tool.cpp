// Time windows track (Conformance, Unit 2): graph AUTHORING, separate
// from PARTICIPATING (the lead's own ruling on Q2). u_main.c and
// v_main.f90 never build a fuse::CouplingGraph themselves -- they call
// fuse_load_graph() on a YAML file and nothing else, exactly the way a
// real user hands a coupled job a graph someone else wrote. This tool
// is that "someone else": it calls brusselator_graph.hpp's own
// add_window_coupling() (the SAME builder test/verification/
// brusselator_participants_test.cpp uses for its in-process reference)
// with the ranks and LaunchSpecs a particular launch layout needs, then
// writes the result through fuse::to_yaml() (include/fuse/model/
// yaml_io.hpp) -- text YAML, not
// fuse::serialize()'s binary descriptor (this file's own "what
// fuse_save_graph produces" distinction matters here: a human-readable
// committed reference has to be the TEXT format, found this session
// while locating a format for the committed example).
//
// --launch mpmd: three app segments (U, V, coupler when --ranks-coupler
// is nonzero), each LaunchMode::Mpmd with appnum 0/1/2 in that order,
// ranks laid out as three CONSECUTIVE blocks in exactly that order
// (fuse::transport::validate_launch_layout()'s own requirement) --
// `mpiexec -n ranks-u ./u_main : -n ranks-v ./v_main : -n ranks-coupler
// ./coupler_main`.
//
// --launch accept: Q1's ruling (KNOWN_ISSUES.md D18) -- a processor set
// is ranks, not executables, and connect_pairwise() supports exactly
// two Connect-mode components, so the coupler cannot be its own third
// Connect-mode participant here. Its ranks instead live INSIDE the
// acceptor's (U's) own MPI_COMM_WORLD, exactly as before; what changed
// (KNOWN_ISSUES.md D36) is which ranks each of the three DECLARES.
// Through 2026-10-09, U's declared RankSet spanned its FULL local
// executable world (its own physics ranks plus the bundled coupler
// ranks), with the coupler's own RankSet the trailing, overlapping
// subset of it (`fuse::RankSet`'s own documented "overlap is legal").
// `validate_launch_layout()` accepted that because it skipped
// InProcess components (the coupler) entirely when checking that
// Connect/Mpmd blocks were consecutive -- but D24's fix (a placement is
// decomposed over its OWN declared RankSet, not the whole run) then
// reads that same declared count back on the next host run: U's own
// field was planned at 1/2 its real size on the host that first ran
// with ranks-coupler > 0, because U's declared RankSet said "2 ranks"
// while `u_main.c` decomposes its physics over `--ranks-u` (1 rank)
// alone. A component's RankSet must be exactly the ranks that HOST it,
// which for U no longer includes the coupler's bundled rank.
//
// So, since D36: U's declared RankSet is `{0, ranks_u}` -- its own
// physics ranks only; the coupler's is `{ranks_u, ranks_coupler}` --
// the GAP this leaves between U's own block and V's; V's is
// `{ranks_u + ranks_coupler, ranks_v}`, unchanged. `validate_launch_
// layout()`'s own D36 fix is the other half of this: a gap between two
// Connect/Mpmd blocks is now legal exactly when an in-process
// component's own RankSet covers it exactly, which is precisely what
// the coupler's new declaration does. `u_main.c` itself (not this
// tool) still decides, per rank, whether it is one of U's own physics
// ranks or one of the coupler's bundled placeholder ranks (its own
// --ranks-u argument) -- only the GRAPH's own account of which ranks
// belong to which declared component changed, not which physical ranks
// anything runs on.
//
// NOT YET CONFIRMED BY A HOST RUN beyond D36's own fix (MPI cannot
// start in this sandbox -- KNOWN_ISSUES.md C1): whether a rank inside
// the coupler's declared RankSet that registers none of U's fields (a
// bundled coupler placeholder rank) is tolerated by the rest of the
// participant ABI/Runner the way a component's own sparsely-populated
// RankSet already is elsewhere in the suite (examples/layout_driver's
// own uneven splits). Flagged in the Unit 2 report as the one open
// mechanical question beyond Q1-Q3, unchanged by D36.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>

#include "fuse/base/rational.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/model/yaml_io.hpp"

#include "brusselator_graph.hpp"
#include "brusselator_grid.h"

namespace {

// Mirrors test/verification/brusselator_participants_test.cpp's own
// kNxU/kNyU/kNxV/kNyV/kDtU/kDtV/kWindow/kTolerance/kMaxIterations, so a
// graph this tool writes and a graph that test builds in-process
// describe the identical physics problem -- the point of Unit 2's own
// conformance claim. Duplicated here rather than shared through a
// header because the two are different translation units with
// different link graphs (this tool links no gtest, the test links no
// CLI-argument parsing) and the values are few.
//
// S6 phase 3 (Conformance, Unit 3b), item 2(b): the REAL Brusselator
// example's own parameters (examples/brusselator/main.cpp's own Args
// defaults: --nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 --dt-u 0.002
// --dt-v 0.003 --window 0.02 --tolerance 1e-11 --max-iterations 60),
// not the small/degenerate placeholder this tool and the two
// conformance tests used through Unit 3a -- dt_u/dt_v/window are no
// longer all equal, so this is genuinely multi-substep-per-window now
// (10 U-substeps, ~6.67 -> 7 V-substeps per window; `delivery_pending`,
// item 2(a) of this same round, is what makes a caller no longer need
// to track that ratio itself). `reference_mpmd_3.yaml`/
// `reference_mpmd_3_lowfive.yaml` (this tool's own golden outputs) were
// regenerated against these values -- see this round's own report for
// the before/after `reproduces_reference` byte-identity measurement.
constexpr int kNxU = BRUSS_NX_U, kNyU = BRUSS_NY_U, kNxV = BRUSS_NX_V, kNyV = BRUSS_NY_V;  // brusselator_grid.h
const fuse::Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;

struct Args {
  int ranks_u = 1;
  int ranks_v = 1;
  int ranks_coupler = 0;
  std::string launch;  // "mpmd" | "accept"
  std::string out;
  std::string port_file = "brusselator_connect.port";
  // S6 phase 3 ("LowFive" track), Unit 3: "fuse" (the default, every run
  // before this track) moves u_out/v_out through FUSE's own ordinary
  // Transport; "lowfive" additionally sets FieldSpec::hdf5_path on both
  // and adds a LowFive-launched component so validate() accepts it (this
  // file's own `write_lowfive_declarations` below has the full account).
  std::string transport = "fuse";
  bool ok = false;
  std::string error;
};

[[nodiscard]] Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const char* flag) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        a.error = std::string(flag) + " needs a value";
        return std::nullopt;
      }
      return std::string(argv[++i]);
    };
    if (arg == "--ranks-u") {
      auto v = next("--ranks-u"); if (!v) return a;
      a.ranks_u = std::atoi(v->c_str());
    } else if (arg == "--ranks-v") {
      auto v = next("--ranks-v"); if (!v) return a;
      a.ranks_v = std::atoi(v->c_str());
    } else if (arg == "--ranks-coupler") {
      auto v = next("--ranks-coupler"); if (!v) return a;
      a.ranks_coupler = std::atoi(v->c_str());
    } else if (arg == "--launch") {
      auto v = next("--launch"); if (!v) return a;
      a.launch = *v;
    } else if (arg == "--out") {
      auto v = next("--out"); if (!v) return a;
      a.out = *v;
    } else if (arg == "--port-file") {
      auto v = next("--port-file"); if (!v) return a;
      a.port_file = *v;
    } else if (arg == "--transport") {
      auto v = next("--transport"); if (!v) return a;
      a.transport = *v;
    } else {
      a.error = "unrecognized argument: " + arg;
      return a;
    }
  }
  if (a.launch != "mpmd" && a.launch != "accept") {
    a.error = "--launch must be 'mpmd' or 'accept' (got '" + a.launch + "')";
    return a;
  }
  if (a.transport != "fuse" && a.transport != "lowfive") {
    a.error = "--transport must be 'fuse' or 'lowfive' (got '" + a.transport + "')";
    return a;
  }
  if (a.out.empty()) {
    a.error = "--out is required";
    return a;
  }
  if (a.ranks_u <= 0 || a.ranks_v <= 0 || a.ranks_coupler < 0) {
    a.error = "--ranks-u/--ranks-v must be positive and --ranks-coupler non-negative";
    return a;
  }
  a.ok = true;
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (!args.ok) {
    std::fprintf(stderr,
                 "usage: brusselator_graph_tool --ranks-u N --ranks-v N "
                 "[--ranks-coupler N] --launch mpmd|accept --out graph.yaml "
                 "[--port-file path]\nerror: %s\n",
                 args.error.c_str());
    return 1;
  }

  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  (void)graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  fuse::RankSet u_ranks, v_ranks;
  std::optional<fuse::RankSet> coupler_ranks;
  fuse::LaunchSpec u_launch, v_launch, coupler_launch;

  if (args.launch == "mpmd") {
    // Three consecutive blocks, in graph component order (U, V, then
    // coupler, matching add_window_coupling()'s own construction order)
    // -- exactly what validate_launch_layout() requires for three Mpmd
    // app segments.
    u_ranks = fuse::RankSet{0, args.ranks_u};
    v_ranks = fuse::RankSet{args.ranks_u, args.ranks_v};
    u_launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 0};
    v_launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 1};
    if (args.ranks_coupler > 0) {
      coupler_ranks = fuse::RankSet{args.ranks_u + args.ranks_v, args.ranks_coupler};
      coupler_launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 2};
    }
  } else {  // accept
    // KNOWN_ISSUES.md D36: U's declared range is exactly its own
    // physics ranks -- the ranks it actually hosts, which is what
    // NativeMeshBackend::register_default_placements() (D24) decomposes
    // a placement over. The coupler's bundled ranks still ride along
    // inside the acceptor's (U's) own local MPI_COMM_WORLD, physically
    // unchanged, but the GRAPH now declares them as the coupler's own
    // RankSet -- the gap this leaves between U's block and V's, which
    // validate_launch_layout()'s own D36 fix accepts because an
    // in-process component's RankSet covers it exactly. V's range is
    // unchanged: it still starts right after every rank U's executable
    // carries, physics and bundled coupler alike.
    u_ranks = fuse::RankSet{0, args.ranks_u};
    const int u_local_world = args.ranks_u + args.ranks_coupler;
    v_ranks = fuse::RankSet{u_local_world, args.ranks_v};
    u_launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect,
                               .is_acceptor = true,
                               .port_file = args.port_file};
    v_launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect,
                               .is_acceptor = false,
                               .port_file = args.port_file};
    if (args.ranks_coupler > 0) {
      coupler_ranks = fuse::RankSet{args.ranks_u, args.ranks_coupler};
      // coupler_launch left at its default (LaunchMode::InProcess):
      // its ranks ride along inside U's own local world; connect() is
      // never called on its behalf (KNOWN_ISSUES.md D18). Its declared
      // RankSet is unchanged by D36 -- it already named exactly the
      // bundled ranks; only U's own declaration (above) shrank to stop
      // including them.
    }
  }

  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, fuse::WaveformIteration::Jacobi, coupler_ranks,
      u_ranks, v_ranks, u_launch, v_launch, coupler_launch);
  (void)handles;

  // S6 phase 3 ("LowFive" track), Unit 3. A SEPARATE, fieldless component
  // declares LaunchMode::LowFive -- not U or V's own launch mode, which
  // stays whatever --launch chose (mpmd/accept): "where do my ranks come
  // from" (U/V's own launch story) and "does ANY component here need
  // LowFive interception set up" (this one) are different questions, the
  // same way a Coupler is a placement separate from the physics
  // components it serves (docs/dev/coupler-placement.md). Its own ranks
  // span every rank in use, so validate() accepts the graph regardless of
  // which rank ends up running fuse_setup() first -- fuse_setup()'s own
  // per-process dispatch (docs/user/lowfive.md) is what actually decides,
  // at run time, which processes set up the interception.
  if (args.transport == "lowfive") {
    const auto u_out = graph.find_field(handles.u_component, "u_out");
    const auto v_out = graph.find_field(handles.v_component, "v_out");
    if (!u_out || !v_out) {
      std::fprintf(stderr, "FAIL: could not find u_out/v_out fields\n");
      return 1;
    }
    const_cast<fuse::FieldSpec&>(graph.field(*u_out)).hdf5_path =
        "brusselator_u.h5/grid/u_out";
    const_cast<fuse::FieldSpec&>(graph.field(*v_out)).hdf5_path =
        "brusselator_v.h5/grid/v_out";
    const int total_ranks = args.ranks_u + args.ranks_v + args.ranks_coupler;
    // Unit 3 (S6 phase 3, Conformance): MEASURED (brusselator_lowfive_
    // harness_test.cpp's own first run that ever drove this fieldless
    // marker through a live Runner/ABI): with no explicit `.kind` -- the
    // ComponentDecl default, ComponentKind::Physics -- fuse_initialize()'s
    // own wiring loop (src/abi/participant.cpp) does not exempt it the
    // way it exempts a real ComponentKind::Coupler entry
    // (`if (decl.kind == ComponentKind::Coupler) continue;`), so it
    // becomes a THIRD externally driven component and any caller's own
    // begin_step() loop has no step call for it. brusselator_graph_tool_
    // lowfive_test.cpp never caught this because neither of its own
    // tests drives the generated graph through a live Runner/ABI at all
    // -- this component had the identical omission from the day it was
    // added until this fix, unobserved for exactly that reason.
    (void)graph.add_component(
        {.name = "lowfive", .ranks = fuse::RankSet{0, total_ranks},
         .interval = fuse::Rational{1},
         .kind = fuse::ComponentKind::Coupler,
         .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  }

  const auto yaml = fuse::to_yaml(graph);
  if (!yaml.ok()) {
    std::fprintf(stderr, "FAIL: to_yaml: %s: %s\n", yaml.error().code.c_str(),
                 yaml.error().message.c_str());
    return 1;
  }

  std::ofstream out(args.out, std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "FAIL: could not open '%s' for writing\n", args.out.c_str());
    return 1;
  }
  out << yaml.value();
  if (!out) {
    std::fprintf(stderr, "FAIL: could not write to '%s'\n", args.out.c_str());
    return 1;
  }
  return 0;
}
