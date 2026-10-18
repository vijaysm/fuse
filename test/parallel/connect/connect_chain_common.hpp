#pragma once

// KNOWN_ISSUES.md D18, round 2 (the lead's own instruction): the N-way
// generalisation of connect_common.hpp's own two-executable pair, built
// separately from it (connect_a.cpp/connect_b.cpp and their six existing
// ctest rows are UNTOUCHED -- this file and connect_chain_{a,b,c,d}.cpp
// are new, independent executables) so the proven 2-component rows carry
// zero risk from this round's own new code.
//
// THE GRAPH. `count` components named "a","b","c","d" (only the first
// `count` declared), each ONE rank (na = 1 for every link -- see
// run_chain.sh's own header for why this keeps the golden comparison
// reusing golden-1+1.txt rather than a new golden file per chain length).
// Component i < count-1 exports field "outi" (Provides); component i > 0
// imports "ini" (Requires, fed by component i-1's "outi-1"). A middle
// component (0 < i < count-1) therefore has BOTH: ChainLinkComponent's
// own advance() copies its import storage into its export storage
// verbatim (an identity passthrough), so the value "a" writes (its own
// global id) reaches the LAST component in the chain unchanged, exactly
// as a single a->b hop already proves for two components -- this is the
// SAME claim, walked through more than one hop.
//
// THE LAUNCH. Every component is `LaunchMode::Connect`. Step 1 (between
// "a" and "b") is unchanged from connect_common.hpp's own two-component
// rule: "a" is the acceptor (port_ab), "b" is the connector. Each LATER
// component (i >= 2, i.e. "c" then "d") is the acceptor for ITS OWN step,
// publishing its own port file ("each component publishes its port name
// once", connect_chain_plan()'s own rule, include/fuse/transport/
// connect.hpp) -- the already-merged chain is always the connector at
// that step, which `fuse::transport::connect_chain()` (src/transport/
// connect.cpp) handles internally; this file only has to declare the
// LaunchSpecs connect_chain_plan() requires, not drive the merge itself.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

#include <fuse/base/units.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/transport/connect.hpp>
#include <fuse/transport/transport.hpp>

#include "connect_partition.hpp"

namespace fuse_connect_chain {

constexpr int kMaxChainComponents = 4;
constexpr const char* kComponentNames[kMaxChainComponents] = {"a", "b", "c", "d"};

// Verbatim reuse of connect_common.hpp's own ConnectTestMesh (see that
// file's header comment for why a from-scratch minimal MeshView exists
// at all rather than reusing test/support/partitioned_test_backend.hpp).
class ChainTestMesh final : public fuse::MeshView {
 public:
  ChainTestMesh(int dim, std::vector<std::size_t> gids) : dim_(dim), gid_(std::move(gids)) {}
  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == dim_ ? gid_.size() : 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override { return entity_count(dim); }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override { return fuse::Point{}; }
  [[nodiscard]] double measure(int, std::size_t) const override { return 1.0; }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim != dim_ || i >= gid_.size()) {
      throw std::out_of_range("ChainTestMesh: index out of range");
    }
    return gid_[i];
  }
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, fuse::Point*) const override {
    throw std::logic_error("ChainTestMesh: this view reports zero simplices");
  }

 private:
  int dim_;
  std::vector<std::size_t> gid_;
};

/// `me`: this rank's position in the MERGED base communicator connect()
/// returns (NOT MPI_COMM_WORLD -- connect_common.hpp's own
/// ConnectTestBackend states why). Every link's RankSet is {link_index,
/// 1}: one rank per component, in chain order, which is also why
/// partition_block() at every hop is the identity (`n/1` entities, all
/// of them, every time) and the chain's own output matches golden-1+1.txt
/// unchanged regardless of chain length.
class ChainTestBackend final : public fuse::MeshBackend {
 public:
  ChainTestBackend(int me, int count, std::size_t n_entities, int dim)
      : me_(me), count_(count), n_(n_entities), dim_(dim) {}

  void set_placements(std::vector<fuse::RankSet> placements) { placements_ = std::move(placements); }

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    std::vector<std::size_t> gids;
    for (int i = 0; i < count_; ++i) {
      const fuse::RankSet rs{i, 1};
      if (rs.contains(me_)) { gids = fuse_connect_test::partition_block(rs, me_, n_); break; }
    }
    meshes_.push_back(std::make_unique<ChainTestMesh>(dim_, std::move(gids)));
    names_.push_back(decl.name);
    std::vector<std::unique_ptr<fuse::MeshView>> per;
    for (const fuse::RankSet& rs : placements_) {
      per.push_back(std::make_unique<ChainTestMesh>(
          dim_, rs.contains(me_) ? fuse_connect_test::partition_block(rs, me_, n_) : std::vector<std::size_t>{}));
    }
    placed_.push_back(std::move(per));
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId placement) const override {
    if (!placement.valid() || placements_.empty()) return view(id);
    if (!id.valid() || id.value() >= placed_.size()) return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    if (placement.value() >= placed_[id.value()].size()) {
      return fuse::Failure{"E-MESH-PLACEMENT",
                           "no decomposition for component " + std::to_string(placement.value())};
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
    if (!id.valid() || id.value() >= meshes_.size()) return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    return meshes_[id.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] std::string name() const override { return "connect-chain-test"; }

 private:
  int me_, count_;
  std::size_t n_;
  int dim_;
  std::vector<fuse::RankSet> placements_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::vector<std::unique_ptr<fuse::MeshView>>> placed_;
  std::vector<std::string> names_;
};

/// One link of the chain. `has_in`/`has_out` select which of the two
/// roles below this instance plays (both, for a middle link):
///   - `is_producer_` (component 0 only): advance() writes its own
///     global id into its export storage (connect_common.hpp's own
///     ScalarComponent, same rule).
///   - otherwise, if it has BOTH an import and an export: advance()
///     copies import -> export verbatim (the passthrough this file's own
///     header comment describes).
///   - the LAST component has only an import; advance() does nothing,
///     exactly as connect_common.hpp's own consumer-only ScalarComponent.
class ChainLinkComponent final : public fuse::Component {
 public:
  ChainLinkComponent(std::string name, std::optional<fuse::FieldId> out_field,
                     std::optional<fuse::FieldId> in_field, std::size_t count,
                     bool is_producer, const fuse::MeshView* producer_view, int dim)
      : name_(std::move(name)), out_field_(out_field), in_field_(in_field),
        out_(out_field.has_value() ? count : 0), in_(in_field.has_value() ? count : 0),
        is_producer_(is_producer), view_(producer_view), dim_(dim) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry, const fuse::CouplingGraph&) override {
    if (out_field_.has_value()) {
      auto r = registry.bind(*out_field_, fuse::FieldRef{out_.data(),
                                                         fuse::Layout{.stride = 1, .count = out_.size()},
                                                         fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    if (in_field_.has_value()) {
      auto r = registry.bind(*in_field_, fuse::FieldRef{in_.data(),
                                                        fuse::Layout{.stride = 1, .count = in_.size()},
                                                        fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    if (is_producer_ && view_ != nullptr) {
      for (std::size_t i = 0; i < out_.size(); ++i) {
        out_[i] = static_cast<double>(view_->global_id(dim_, i));
      }
    } else if (out_field_.has_value() && in_field_.has_value()) {
      // Middle link: passthrough, identity copy.
      const std::size_t n = std::min(out_.size(), in_.size());
      for (std::size_t i = 0; i < n; ++i) out_[i] = in_[i];
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return 0; }

  [[nodiscard]] const std::vector<double>& held() const {
    return in_field_.has_value() ? in_ : out_;
  }

 private:
  std::string name_;
  std::optional<fuse::FieldId> out_field_;
  std::optional<fuse::FieldId> in_field_;
  std::vector<double> out_;
  std::vector<double> in_;
  bool is_producer_;
  const fuse::MeshView* view_;
  int dim_;
};

enum class CliLaunch { Connect };

struct Args {
  int count = 2;             ///< total chain length (2, 3 or 4)
  std::string port_ab;       ///< step 1's own port file (a/b)
  std::string port_c;        ///< step 2's own port file (c joins)
  std::string port_d;        ///< step 3's own port file (d joins)
  int periods = 1;
  double timeout_s = 60.0;
};

[[nodiscard]] inline Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
    if (arg == "--count") a.count = std::atoi(next().c_str());
    else if (arg == "--port-ab") a.port_ab = next();
    else if (arg == "--port-c") a.port_c = next();
    else if (arg == "--port-d") a.port_d = next();
    else if (arg == "--periods") a.periods = std::atoi(next().c_str());
    else if (arg == "--timeout") a.timeout_s = std::atof(next().c_str());
  }
  return a;
}

/// Builds the SAME `count`-component chain graph on every executable
/// (every rank needs a Component for every graph component, mirroring
/// connect_common.hpp's own build_graph() comment). `cids[i]` is
/// component i's ComponentId.
[[nodiscard]] inline fuse::CouplingGraph build_chain_graph(
    const Args& args, std::vector<fuse::ComponentId>& cids, fuse::RegionId& out_region) {
  fuse::CouplingGraph g;
  const auto mesh = g.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
  out_region = g.add_region(fuse::Region{.name = "r", .mesh = mesh, .dimension = 0});
  cids.assign(static_cast<std::size_t>(args.count), fuse::ComponentId{});

  for (int i = 0; i < args.count; ++i) {
    fuse::LaunchSpec launch;
    launch.mode = fuse::LaunchMode::Connect;
    if (i == 0) {
      launch.is_acceptor = true;
      launch.port_file = args.port_ab;
    } else if (i == 1) {
      launch.is_acceptor = false;
      launch.port_file = args.port_ab;
    } else if (i == 2) {
      launch.is_acceptor = true;
      launch.port_file = args.port_c;
    } else {  // i == 3
      launch.is_acceptor = true;
      launch.port_file = args.port_d;
    }
    cids[static_cast<std::size_t>(i)] = g.add_component(fuse::ComponentDecl{
        .name = kComponentNames[i], .ranks = fuse::RankSet{i, 1},
        .interval = fuse::Rational{1}, .launch = launch});
  }
  for (int i = 0; i + 1 < args.count; ++i) {
    const auto out_f = g.add_field(
        cids[static_cast<std::size_t>(i)],
        fuse::FieldSpec{.name = "out" + std::to_string(i), .region = out_region,
                       .units = fuse::units::kKelvin, .intent = fuse::Intent::Provides});
    const auto in_f = g.add_field(
        cids[static_cast<std::size_t>(i + 1)],
        fuse::FieldSpec{.name = "in" + std::to_string(i), .region = out_region,
                       .units = fuse::units::kKelvin, .intent = fuse::Intent::Requires});
    g.add_channel("link" + std::to_string(i), out_f, in_f);
  }
  return g;
}

/// Drives one executable's own link of the chain: builds the graph,
/// calls connect() (dispatches to fuse::transport::connect_chain() for
/// any count -- src/transport/connect.cpp), runs `args.periods` periods,
/// and -- ONLY on the LAST component -- prints every value it holds, in
/// the SAME "rank=... i=... value=..." format connect_common.hpp's own
/// run_participant() uses, so run_chain.sh's diff against golden-1+1.txt
/// is the identical comparison the 2-component row already makes.
/// `my_index` is this executable's own 0-based position in the chain.
[[nodiscard]] inline int run_chain_participant(int argc, char** argv, int my_index) {
  MPI_Init(&argc, &argv);
  const Args args = parse_args(argc, argv);

  std::vector<fuse::ComponentId> cids;
  fuse::RegionId region;
  const fuse::CouplingGraph g = build_chain_graph(args, cids, region);
  const fuse::ComponentId my_id = cids[static_cast<std::size_t>(my_index)];

  fuse::transport::ConnectOptions copts;
  copts.my_component = my_id;
  copts.timeout_s = args.timeout_s;

  int world_comm_f = 0;
  { MPI_Fint f = MPI_Comm_c2f(MPI_COMM_WORLD); world_comm_f = static_cast<int>(f); }
  auto connected = fuse::transport::connect(g, world_comm_f, copts);
  if (!connected.ok()) {
    std::fprintf(stderr, "connect() refused: [%s] %s\n",
                connected.error().code.c_str(), connected.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  const int base_comm_f = connected.value().base_comm_f;
  const MPI_Comm base_comm = MPI_Comm_f2c(static_cast<MPI_Fint>(base_comm_f));

  {
  int merged_rank = 0;
  MPI_Comm_rank(base_comm, &merged_rank);
  constexpr int kDim = 0;
  constexpr std::size_t kEntities = 4;

  ChainTestBackend backend(merged_rank, args.count, kEntities, kDim);
  std::vector<fuse::RankSet> placements;
  for (int i = 0; i < args.count; ++i) placements.push_back(fuse::RankSet{i, 1});
  backend.set_placements(placements);
  if (auto r = backend.load(g.mesh(g.region(region).mesh)); !r.ok()) {
    std::fprintf(stderr, "backend.load() failed: [%s] %s\n", r.error().code.c_str(), r.error().message.c_str());
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
  const auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) {
    std::fprintf(stderr, "schedule() failed:\n%s\n", sched.diags.to_string().c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  const auto a_view = backend.placed_view(g.region(region).mesh, cids[0]);
  if (!a_view.ok()) {
    std::fprintf(stderr, "backend decomposition query failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  fuse::ComponentSet components;
  fuse::FieldRegistry registry;
  std::vector<ChainLinkComponent*> links;
  for (int i = 0; i < args.count; ++i) {
    const fuse::ComponentDecl& decl = g.component(cids[static_cast<std::size_t>(i)]);
    std::optional<fuse::FieldId> out_f, in_f;
    for (const fuse::FieldId f : decl.fields) {
      if (g.field(f).intent == fuse::Intent::Provides) out_f = f;
      else in_f = f;
    }
    const auto count_r = backend.placed_region_size(g.region(region), cids[static_cast<std::size_t>(i)]);
    if (!count_r.ok()) {
      std::fprintf(stderr, "placed_region_size() failed for link %d\n", i);
      MPI_Abort(MPI_COMM_WORLD, 1);
      return 1;
    }
    auto link = std::make_unique<ChainLinkComponent>(
        kComponentNames[i], out_f, in_f, count_r.value(), /*is_producer=*/i == 0,
        i == 0 ? a_view.value() : nullptr, kDim);
    links.push_back(link.get());
    components.add(std::move(link));
  }
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
    std::fprintf(stderr, "Runner::initialize() failed: [%s] %s\n", r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (auto r = runner.run(components, args.periods); !r.ok()) {
    std::fprintf(stderr, "Runner::run() failed: [%s] %s\n", r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  if (my_index == args.count - 1) {
    const auto& values = links.back()->held();
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::printf("rank=%d i=%zu value=%a\n", merged_rank, i, values[i]);
    }
    std::fflush(stdout);
  }
  }  // end of the scope that uses the base communicator

  if (auto r = fuse::transport::disconnect(connected.value()); !r.ok()) {
    std::fprintf(stderr, "disconnect() failed: [%s] %s\n", r.error().code.c_str(), r.error().message.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
  return 0;
}

}  // namespace fuse_connect_chain
