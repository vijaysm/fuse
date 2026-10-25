#pragma once

// Shared by connect_a.cpp and connect_b.cpp. Both executables build the
// IDENTICAL two-component graph --
// component "a" (the producer) and component "b" (the consumer) -- declare
// their own LaunchSpec per the command line (mpmd segment, or connect
// acceptor/connector), call fuse::transport::connect(), and drive
// fuse::Runner over the resulting base communicator for a fixed number of
// periods, printing every value "b" receives in hexfloat so the host test
// driver (run_pair.sh, or a single MPMD mpiexec) can diff it against the
// in-process golden this same graph produces.
//
// WHY A SEPARATE BACKEND FROM test/support/partitioned_test_backend.hpp,
// RATHER THAN REUSING IT. That backend's per-rank decomposition is keyed off
// `fuse_test::this_rank()`, which reads MPI_COMM_WORLD directly -- correct
// for an ordinary single-mpiexec MPI test, where MPI_COMM_WORLD already IS
// the job's base communicator, but WRONG here for `Connect`-mode (separate
// launches): this executable's own MPI_COMM_WORLD contains only its own
// ranks, numbered from 0, which is not the merged base communicator
// connect() returns. ConnectTestBackend below is the same per-placement
// block-decomposition algorithm (block(), verbatim), parameterized by an
// explicit `me` the caller computes from the ACTUAL base communicator
// (MPI_Comm_rank on connect()'s result) instead of assuming it.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <fuse/ir/lowering.hpp>
#include <fuse/model/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/transport/connect.hpp>
#include <fuse/transport/transport.hpp>

#include "connect_partition.hpp"

namespace fuse_connect_test {

// ---------------------------------------------------------------------------
// ConnectTestBackend / ConnectTestMesh -- see this file's own header comment.
// ---------------------------------------------------------------------------

// Verbatim copy of test/support/partitioned_test_backend.hpp's own
// PartitionedTestMesh (see this file's header comment for why the backend
// around it is not also reused as-is): no geometry, every entity owned,
// zero simplices -- this view exists to exercise redistribution, not remap.
class ConnectTestMesh final : public fuse::MeshView {
 public:
  ConnectTestMesh(int dim, std::vector<std::size_t> gids)
      : dim_(dim), gid_(std::move(gids)) {}
  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == dim_ ? gid_.size() : 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override {
    return fuse::Point{};
  }
  [[nodiscard]] double measure(int, std::size_t) const override { return 1.0; }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim != dim_ || i >= gid_.size()) {
      throw std::out_of_range("ConnectTestMesh: index out of range");
    }
    return gid_[i];
  }
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override {
    return 0;
  }
  void simplex_points(int, std::size_t, std::size_t, fuse::Point*) const override {
    throw std::logic_error(
        "ConnectTestMesh::simplex_points: this view reports zero simplices, "
        "so nothing may ask it for one");
  }

 private:
  int dim_;
  std::vector<std::size_t> gid_;
};

class ConnectTestBackend final : public fuse::MeshBackend {
 public:
  // `me`: this rank's position in the communicator the graph's RankSets are
  // numbered over (the MERGED base communicator connect() returns), NOT
  // necessarily MPI_COMM_WORLD -- the whole reason this class exists rather
  // than reusing PartitionedTestBackend. `a`/`b`: the two components'
  // RankSets, in that numbering. `n_entities`: the producer's entity count,
  // redistributed onto `b`'s own decomposition by the Transport.
  ConnectTestBackend(int me, fuse::RankSet a, fuse::RankSet b,
                     std::size_t n_entities, int dim)
      : me_(me), a_(a), b_(b), n_(n_entities), dim_(dim) {}

  void set_placements(std::vector<fuse::RankSet> placements) {
    placements_ = std::move(placements);
  }

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    std::vector<std::size_t> gids;
    if (a_.contains(me_)) {
      gids = block(a_, me_);
    } else if (b_.contains(me_)) {
      gids = block(b_, me_);
    }
    meshes_.push_back(std::make_unique<ConnectTestMesh>(dim_, std::move(gids)));
    names_.push_back(decl.name);

    std::vector<std::unique_ptr<fuse::MeshView>> per;
    for (const fuse::RankSet& rs : placements_) {
      per.push_back(std::make_unique<ConnectTestMesh>(
          dim_, rs.contains(me_) ? block(rs, me_) : std::vector<std::size_t>{}));
    }
    placed_.push_back(std::move(per));
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId placement) const override {
    if (!placement.valid() || placements_.empty()) return view(id);
    if (!id.valid() || id.value() >= placed_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    if (placement.value() >= placed_[id.value()].size()) {
      return fuse::Failure{"E-MESH-PLACEMENT",
                           "no decomposition for component " +
                               std::to_string(placement.value())};
    }
    return placed_[id.value()][placement.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(
      const fuse::Region& r, fuse::ComponentId placement) const override {
    const auto v = placed_view(r.mesh, placement);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "no mesh named '" + std::string(name) + "'"};
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    return meshes_[id.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] std::string name() const override { return "connect-test"; }

 private:
  [[nodiscard]] std::vector<std::size_t> block(fuse::RankSet rs, int me) const {
    return partition_block(rs, me, n_);
  }

  int           me_;
  fuse::RankSet a_, b_;
  std::size_t   n_;
  int           dim_;
  std::vector<fuse::RankSet> placements_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::vector<std::unique_ptr<fuse::MeshView>>> placed_;
  std::vector<std::string> names_;
};

/// One field, `count` doubles. `is_producer` seeds the storage with this
/// rank's own global ids (ConnectTestBackend::placed_view) on `advance()`
/// so the field transfer has a known expected value; a consumer's advance()
/// does nothing (the Transport, not the component, moves the value in).
class ScalarComponent final : public fuse::Component {
 public:
  ScalarComponent(std::string name, fuse::FieldId field, std::size_t count,
                 bool is_producer, const fuse::MeshView* producer_view = nullptr,
                 int dim = 0)
      : name_(std::move(name)), field_(field), storage_(count, 0.0),
        is_producer_(is_producer), view_(producer_view), dim_(dim) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph&) override {
    fuse::Layout layout;
    layout.count = storage_.size();
    return registry.bind(field_, fuse::FieldRef{storage_.data(), layout,
                                                fuse::MemorySpace::Host});
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    if (is_producer_ && view_ != nullptr) {
      for (std::size_t i = 0; i < storage_.size(); ++i) {
        storage_[i] = static_cast<double>(view_->global_id(dim_, i));
      }
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return 0; }

  [[nodiscard]] const std::vector<double>& data() const { return storage_; }

 private:
  std::string          name_;
  fuse::FieldId        field_;
  std::vector<double>  storage_;
  bool                 is_producer_;
  const fuse::MeshView* view_;
  int                  dim_;
};

/// How THIS executable was told to launch, from the command line
/// (`--launch mpmd|accept|connect`). `accept`/`connect` are both
/// `fuse::LaunchMode::Connect`, differing only in `LaunchSpec::is_acceptor`.
enum class CliLaunch { Mpmd, Accept, Connect };

struct Args {
  CliLaunch launch = CliLaunch::Mpmd;
  int na = 0, nb = 0;              ///< declared rank counts of "a" and "b"
  std::string port_file;           ///< Connect mode only
  int periods = 3;
  double timeout_s = 30.0;
};

[[nodiscard]] inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
    if (arg == "--launch") {
      const auto v = next();
      if (v == "mpmd") a.launch = CliLaunch::Mpmd;
      else if (v == "accept") a.launch = CliLaunch::Accept;
      else if (v == "connect") a.launch = CliLaunch::Connect;
    } else if (arg == "--na") {
      a.na = std::atoi(next().c_str());
    } else if (arg == "--nb") {
      a.nb = std::atoi(next().c_str());
    } else if (arg == "--port-file") {
      a.port_file = next();
    } else if (arg == "--periods") {
      a.periods = std::atoi(next().c_str());
    } else if (arg == "--timeout") {
      a.timeout_s = std::atof(next().c_str());
    }
  }
  return a;
}

/// Builds the SAME two-component graph on every process regardless of which
/// component this executable plays (test/parallel/runner_transport_test.cpp's
/// StubComponent doc: build_ticks() requires a Component for every graph
/// component on every rank, not only the ranks that own it). `na`/`nb` are
/// rank counts; "a" occupies [0, na), "b" occupies [na, na+nb) of the base
/// communicator connect() will form -- the consecutive-block layout
/// fuse::transport::validate_launch_layout() requires.
[[nodiscard]] inline fuse::CouplingGraph build_graph(const Args& args,
                                                     fuse::ComponentId& out_a,
                                                     fuse::ComponentId& out_b,
                                                     fuse::RegionId& out_region) {
  fuse::CouplingGraph g;
  const auto mesh = g.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
  out_region = g.add_region(fuse::Region{.name = "r", .mesh = mesh, .dimension = 0});

  fuse::LaunchSpec launch_a, launch_b;
  if (args.launch == CliLaunch::Mpmd) {
    launch_a = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 0};
    launch_b = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 1};
  } else {
    // Exactly one side is the acceptor, and BOTH executables must say the
    // same one: each builds this graph from its own command line, and deriving
    // the acceptor from `args.launch == Accept` would be true on connect_a's
    // command line and false on connect_b's, so each side would make itself
    // the acceptor and both would wait in MPI_Comm_accept. Component "a" is the
    // acceptor by the graph, full stop; the --launch flag only names which
    // executable this is, and must agree with the side being played.
    launch_a = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect,
                                .is_acceptor = true,
                                .port_file = args.port_file};
    launch_b = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect,
                                .is_acceptor = false,
                                .port_file = args.port_file};
  }

  out_a = g.add_component(fuse::ComponentDecl{.name = "a",
                                              .ranks = fuse::RankSet{0, args.na},
                                              .interval = fuse::Rational{1},
                                              .launch = launch_a});
  out_b = g.add_component(fuse::ComponentDecl{.name = "b",
                                              .ranks = fuse::RankSet{args.na, args.nb},
                                              .interval = fuse::Rational{1},
                                              .launch = launch_b});
  const auto out_field = g.add_field(
      out_a, fuse::FieldSpec{.name = "out", .region = out_region,
                            .units = fuse::units::kKelvin,
                            .intent = fuse::Intent::Provides});
  const auto in_field = g.add_field(
      out_b, fuse::FieldSpec{.name = "in", .region = out_region,
                            .units = fuse::units::kKelvin,
                            .intent = fuse::Intent::Requires});
  g.add_channel("link", out_field, in_field);
  return g;
}

/// Drives one executable's half of the pair: builds the graph, calls
/// connect(), then runs `args.periods` coupling periods over the resulting
/// base communicator and prints every value component "b" holds, in
/// hexfloat, one line per local entity, prefixed by the MERGED rank (not
/// MPI_COMM_WORLD's) so the host test driver can match lines to the
/// in-process golden regardless of which launch mode produced them.
/// `my_name` is "a" or "b" -- which component THIS executable plays.
/// Returns the process exit code.
[[nodiscard]] inline int run_participant(int argc, char** argv,
                                         const std::string& my_name) {
  MPI_Init(&argc, &argv);
  const Args args = parse_args(argc, argv);

  fuse::ComponentId cid_a, cid_b;
  fuse::RegionId region;
  const fuse::CouplingGraph g = build_graph(args, cid_a, cid_b, region);
  const fuse::ComponentId my_id = (my_name == "a") ? cid_a : cid_b;

  fuse::transport::ConnectOptions copts;
  copts.my_component = my_id;
  copts.timeout_s = args.timeout_s;

  int world_comm_f = 0;
  {
    MPI_Fint f = MPI_Comm_c2f(MPI_COMM_WORLD);
    world_comm_f = static_cast<int>(f);
  }
  auto connected = fuse::transport::connect(g, world_comm_f, copts);
  if (!connected.ok()) {
    std::fprintf(stderr, "connect() refused: [%s] %s\n",
                connected.error().code.c_str(), connected.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const int base_comm_f = connected.value().base_comm_f;
  const MPI_Comm base_comm = MPI_Comm_f2c(static_cast<MPI_Fint>(base_comm_f));

  // Everything that USES the base communicator lives inside this scope, so
  // the backend, the Runner and the transport's own duplicate are all
  // destroyed before fuse::transport::disconnect() below -- which must
  // itself precede MPI_Finalize (connect.hpp's own doc on why that order is
  // load-bearing for a Connect-mode handle, and not merely tidy).
  {
  int merged_rank = 0;
  MPI_Comm_rank(base_comm, &merged_rank);

  const fuse::RankSet a_ranks = g.component(cid_a).ranks;
  const fuse::RankSet b_ranks = g.component(cid_b).ranks;
  constexpr int kDim = 0;

  ConnectTestBackend backend(merged_rank, a_ranks, b_ranks,
                             static_cast<std::size_t>(args.na), kDim);
  std::vector<fuse::RankSet> placements = {a_ranks, b_ranks};
  backend.set_placements(placements);
  if (auto r = backend.load(g.mesh(g.region(region).mesh)); !r.ok()) {
    std::fprintf(stderr, "backend.load() failed: [%s] %s\n",
                r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) {
    std::fprintf(stderr, "lower() failed:\n%s\n", lowered.diags.to_string().c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const fuse::ClockSet clocks{g};
  const auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) {
    std::fprintf(stderr, "schedule() failed:\n%s\n", sched.diags.to_string().c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  const auto a_view = backend.placed_view(g.region(region).mesh, cid_a);
  const auto a_count = backend.placed_region_size(g.region(region), cid_a);
  const auto b_count = backend.placed_region_size(g.region(region), cid_b);
  if (!a_view.ok() || !a_count.ok() || !b_count.ok()) {
    std::fprintf(stderr, "backend decomposition query failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  fuse::ComponentSet components;
  fuse::FieldRegistry registry;
  const fuse::ComponentDecl& decl_a = g.component(cid_a);
  const fuse::ComponentDecl& decl_b = g.component(cid_b);
  components.add(std::make_unique<ScalarComponent>(
      "a", decl_a.fields.at(0), a_count.value(), /*is_producer=*/true,
      a_view.value(), kDim));
  components.add(std::make_unique<ScalarComponent>(
      "b", decl_b.fields.at(0), b_count.value(), /*is_producer=*/false));
  for (auto& c : components) {
    if (!c.register_fields(registry, g).ok()) {
      std::fprintf(stderr, "register_fields() failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
  }

  fuse::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = base_comm_f;
  cfg.transport_setup = [base_comm_f](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(base_comm_f);
  };

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    std::fprintf(stderr, "Runner::initialize() failed: [%s] %s\n",
                r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (auto r = runner.run(components, args.periods); !r.ok()) {
    std::fprintf(stderr, "Runner::run() failed: [%s] %s\n",
                r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  auto* b = dynamic_cast<ScalarComponent*>(components.find("b"));
  for (std::size_t i = 0; i < b->data().size(); ++i) {
    std::printf("rank=%d i=%zu value=%a\n", merged_rank, i, b->data()[i]);
  }
  std::fflush(stdout);
  }  // end of the scope that uses the base communicator

  if (auto r = fuse::transport::disconnect(connected.value()); !r.ok()) {
    std::fprintf(stderr, "disconnect() failed: [%s] %s\n",
                r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
  return 0;
}

}  // namespace fuse_connect_test
