// Scaled device benchmark (docs/dev/device-resident-arcs.md rollout step 6
// follow-up). Every row the step 6 gate measured is a few hundred to a few
// thousand cells, so kernel-launch latency and the remaining host stops
// dominate the ratio; this program exercises the SAME Runner machinery --
// a native structured-quad mesh pair, a Remap, an MpiTransport exchange, a
// Diagnose and a Sink -- at a problem size orders of magnitude larger, so
// the per-instruction overhead can be separated from genuine arithmetic
// cost. Not a new code path: the graph below is the same shape
// test/runtime/runner_device_test.cpp's weighted_remap_then_relax scenario
// and test/parallel/runner_transport_test.cpp's disjoint two-component job
// already exercise at a handful of cells; this file scales it up and adds
// the timing/RunStats report.
//
// THE GRAPH.
//
//   provider (N x N native quad cells)  --Remap-->  consumer ((N+1) x (N+1))
//
// provider's mesh is "generated:quad:NxN" and consumer's is
// "generated:quad:(N+1)x(N+1)", both over the unit square: the two grids
// cannot coincide (different cell counts at every index), so
// Runner::Impl::regions_coincide() returns false and the channel's Remap
// is a real conservative weighted matrix-vector product
// (RemapMultiFunctor), never the coincident-mesh identity copy -- the same
// property test/runtime/runner_device_test.cpp's own weighted_remap_then_
// relax scenario relies on ("8 cells -> 6 cells, so the regions cannot
// coincide"), scaled up here instead of hand-built at 8 cells. A Diagnose
// instruction follows automatically (fuse::insert_derived_ops: any channel
// that emitted a Remap gets one), then Sink. Both fields carry
// Conservation::None, so no Constrain is inserted -- this program's own
// chain is exactly Sample -> Remap -> Transport -> Diagnose -> Sink, no
// more and no fewer instructions.
//
// PLACEMENT AND TRANSPORT. At MPI world size >= 2, provider owns ranks
// [0, w/2) and consumer owns [w/2, w) -- disjoint, so a real MpiTransport
// exchange moves the field across ranks every period (RunnerConfig::
// transport = "mpi", a real base_communicator). At world size 1 both
// components are placed on the whole (one-rank) world, since a disjoint
// split would leave one side on an empty RankSet (rejected, E-TRANSPORT-
// RANKS; see test/parallel/runner_transport_test.cpp's own comment on
// exactly this).
//
// EXECUTION SPACE. RunnerConfig::execution_space follows the environment
// (FUSE_EXECUTION_SPACE) by default, the same convention
// fuse_add_device_twin()/DEVICE_TWIN ctest rows rely on to run one binary
// under both interpreters. --fields device additionally registers the two
// fields' storage in device memory (Kokkos::DefaultExecutionSpace::
// memory_space) and forces execution_space to Device explicitly (a
// device-registered field under Host execution is refused, E-RT-SPACE --
// see test/runtime/runner_device_test.cpp's
// DeviceRegisteredFieldIsRefusedUnderHostExecution). --fields host (the
// default) registers ordinary std::vector-backed host storage and leaves
// execution_space at Default, so the SAME command run once directly and
// once under FUSE_EXECUTION_SPACE=device compares the elementwise/
// weighted kernels' dispatch space with the fields staying host-resident
// -- the device twin's own convention.
//
// USAGE.
//
//   fuse_bench_scaled_device_benchmark [--n N] [--periods K]
//       [--fields host|device] [--verify] [--timing out.csv]
//
// --verify ignores --n (uses a small, fixed size) and --fields, and
// instead builds and runs the SAME graph twice in this one process --
// once under RunnerConfig::RunnerExecutionSpace::Host, once under Device
// (skipped, not silently passed, when this build has no device backend)
// -- comparing the consumer's delivered values to 1e-14 relative
// (test/runtime/runner_device_test.cpp's own tolerance for an FMA-
// contracted weighted-remap kernel, DeviceAndHostAgreeToRoundingOnRelax
// AndWeightedRemap). Exit status is non-zero on any mismatch, a Runner
// failure, or (compiled with a device backend) a verdict that never
// exercised the device path at all.
//
// Without --verify, exit status is non-zero only on a Runner failure;
// the printed report is the deliverable, not a pass/fail gate -- the
// correctness claim lives entirely in the --verify row.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <fuse/backend/native_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

namespace {

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;
using Space = RunnerConfig::RunnerExecutionSpace;

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
int world_size() {
  int w = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &w);
  return w;
}

// ---------------------------------------------------------------------------
// Components. Host-registered storage (std::vector) and device-registered
// storage (Kokkos::View in the default execution space's memory) each get
// their own small class, the same split
// test/runtime/runner_device_test.cpp's Producer/DeviceProducer and
// Consumer/DeviceConsumer draw, for the same reason: a FieldRef's memory
// space is a property of WHERE the pointer it carries lives, not something
// a single class can parameterise away without hiding that fact.
// ---------------------------------------------------------------------------

class HostProducer final : public fuse::Component {
 public:
  explicit HostProducer(std::size_t cells) : data_(cells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "provider"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("provider");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    // Varies per cell and per step: a stale buffer or a dropped step shows
    // up in the delivered values, the same discipline
    // runner_device_test.cpp's Producer::advance() documents.
    for (std::size_t i = 0; i < data_.size(); ++i) {
      data_[i] = static_cast<double>(steps_) + 0.125 * static_cast<double>(i % 97);
    }
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> data_;
  int                  steps_ = 0;
};

class HostConsumer final : public fuse::Component {
 public:
  explicit HostConsumer(std::size_t cells) : data_(cells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
};

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
using DeviceMem = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceVec = Kokkos::View<double*, DeviceMem>;

class DeviceProducer final : public fuse::Component {
 public:
  explicit DeviceProducer(std::size_t cells)
      : data_("provider.out", cells), host_(cells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "provider"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("provider");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.extent(0)},
                                          fuse::MemorySpace::Device});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    for (std::size_t i = 0; i < host_.size(); ++i) {
      host_[i] = static_cast<double>(steps_) + 0.125 * static_cast<double>(i % 97);
    }
    Kokkos::deep_copy(data_, Kokkos::View<const double*, Kokkos::HostSpace,
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                                 host_.data(), host_.size()));
    return fuse::Result<void>::success();
  }

 private:
  DeviceVec           data_;
  std::vector<double> host_;
  int                 steps_ = 0;
};

class DeviceConsumer final : public fuse::Component {
 public:
  explicit DeviceConsumer(std::size_t cells) : data_("consumer.in", cells) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.extent(0)},
                                          fuse::MemorySpace::Device});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::vector<double> values() const {
    std::vector<double> out(data_.extent(0), 0.0);
    Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), out.size()),
                      data_);
    return out;
  }

 private:
  DeviceVec data_;
};
#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP || KOKKOS_ENABLE_SYCL

// ---------------------------------------------------------------------------
// Graph / placement.
// ---------------------------------------------------------------------------

/// provider gets ranks [0, w/2), consumer [w/2, w) -- disjoint whenever
/// w >= 2 (the Overlap::Disjoint shape test/parallel/runner_transport_
/// test.cpp's placement_for() builds); at w == 1 a disjoint split leaves
/// one side on an empty RankSet, which build_plan rejects
/// (E-TRANSPORT-RANKS -- that file's own
/// ARunDeliversTheSourceValuesToTheDestinationRanks comment), so both
/// components take the whole one-rank world instead.
struct Placement {
  fuse::RankSet provider;
  fuse::RankSet consumer;
};
[[nodiscard]] Placement placement_for(int world) {
  if (world <= 1) return Placement{fuse::RankSet{0, 1}, fuse::RankSet{0, 1}};
  const int half = world / 2;
  return Placement{fuse::RankSet{0, half}, fuse::RankSet{half, world - half}};
}

/// Everything build_graph() hands back, by VALUE from the graph it just
/// built -- never a hardcoded index. CouplingGraph assigns every id in
/// add_*() call order, so provider/r_provider being added first would
/// make {0} the right guess today, but a guess is not a contract; the
/// rest of this file reads these fields instead of relying on that order.
struct GraphBundle {
  CouplingGraph     graph;
  fuse::ComponentId provider_id;
  fuse::ComponentId consumer_id;
  fuse::RegionId    r_provider;
  fuse::RegionId    r_consumer;
};

[[nodiscard]] GraphBundle build_graph(std::size_t n, const Placement& pl) {
  GraphBundle b;
  CouplingGraph& g = b.graph;
  const auto mesh_provider = g.add_mesh(fuse::MeshDecl{
      .name = "provider_mesh",
      .source = "generated:quad:" + std::to_string(n) + "x" + std::to_string(n)});
  const auto mesh_consumer = g.add_mesh(fuse::MeshDecl{
      .name = "consumer_mesh",
      .source = "generated:quad:" + std::to_string(n + 1) + "x" + std::to_string(n + 1)});
  b.r_provider = g.add_region(
      fuse::Region{.name = "r_provider", .mesh = mesh_provider, .dimension = 2});
  b.r_consumer = g.add_region(
      fuse::Region{.name = "r_consumer", .mesh = mesh_consumer, .dimension = 2});

  b.provider_id = g.add_component(fuse::ComponentDecl{
      .name = "provider", .ranks = pl.provider, .interval = Rational{1}});
  b.consumer_id = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = pl.consumer, .interval = Rational{1}});

  const auto out = g.add_field(
      b.provider_id, fuse::FieldSpec{.name = "out", .region = b.r_provider,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Provides});
  const auto in = g.add_field(
      b.consumer_id, fuse::FieldSpec{.name = "in", .region = b.r_consumer,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Requires});
  // No .remap()/.remap_method() call: the two meshes have different cell
  // counts at every index, so regions_coincide() is false and the default
  // method is the conservative weighted builder -- the same "no explicit
  // method" convention test/runtime/runner_device_test.cpp's
  // weighted_remap_then_relax() scenario relies on for its own 8 -> 6 cell
  // mismatch.
  g.add_channel("link", out, in);
  return b;
}

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) {
      return fuse::Failure{"E-BENCH-TRANSPORT", "expected an MpiTransport"};
    }
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

struct Harness {
  CouplingGraph          graph;
  fuse::native::NativeMeshBackend backend;
  fuse::OperatorDAG      dag;
  fuse::ExecutionPlan    plan;
  ComponentSet           components;
  FieldRegistry          registry;
  fuse::ComponentId      provider_id;
  fuse::ComponentId      consumer_id;
  fuse::RegionId         r_provider;
  fuse::RegionId         r_consumer;

  explicit Harness(int rank, int world) : backend(rank, world) {}
};

/// Builds the graph, backend, IR and plan, and the two components
/// (Host- or Device-registered per `device_fields`), but does not register
/// fields or initialize the Runner -- the caller decides whether to time
/// that separately (the scaled-run path does; --verify does not care).
[[nodiscard]] std::unique_ptr<Harness> build(std::size_t n, bool device_fields,
                                             fuse::Result<void>* err) {
  const int rank = world_rank();
  const int world = world_size();
  const Placement pl = placement_for(world);
  auto h = std::make_unique<Harness>(rank, world);
  GraphBundle b = build_graph(n, pl);
  h->graph = std::move(b.graph);
  h->provider_id = b.provider_id;
  h->consumer_id = b.consumer_id;
  h->r_provider = b.r_provider;
  h->r_consumer = b.r_consumer;

  for (std::uint32_t i = 0; i < h->graph.mesh_count(); ++i) {
    const auto r = h->backend.load(h->graph.mesh(fuse::MeshId{i}), h->graph);
    if (!r.ok()) { *err = r; return h; }
  }

  const auto lowered = lower(h->graph);
  if (!lowered.ok()) {
    *err = fuse::Failure{"E-BENCH-LOWER", lowered.diags.to_string()};
    return h;
  }
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{h->graph};
  const auto sched = schedule(h->graph, h->dag, clocks,
                              fuse::make_region_sizes(h->graph, h->backend));
  if (!sched.ok()) {
    *err = fuse::Failure{"E-BENCH-SCHEDULE", sched.diags.to_string()};
    return h;
  }
  h->plan = sched.plan;

  const auto size_of = [&](fuse::ComponentId cid, const fuse::RegionId& region) -> std::size_t {
    const auto sz = h->backend.placed_region_size(h->graph.region(region), cid);
    return sz.ok() ? sz.value() : 0;
  };
  const std::size_t provider_n = size_of(h->provider_id, h->r_provider);
  const std::size_t consumer_n = size_of(h->consumer_id, h->r_consumer);

  if (device_fields) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    h->components.add(std::make_unique<DeviceProducer>(provider_n));
    h->components.add(std::make_unique<DeviceConsumer>(consumer_n));
#else
    *err = fuse::Failure{"E-BENCH-DEVICE", "this build has no device backend"};
    return h;
#endif
  } else {
    h->components.add(std::make_unique<HostProducer>(provider_n));
    h->components.add(std::make_unique<HostConsumer>(consumer_n));
  }
  *err = fuse::Result<void>::success();
  return h;
}

/// std::vector<double> for either component flavour, host-side, for the
/// --verify comparison.
[[nodiscard]] std::vector<double> consumer_values(const Harness& h) {
  if (const auto* c = dynamic_cast<const HostConsumer*>(h.components.find("consumer"))) {
    return c->data();
  }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
  if (const auto* c = dynamic_cast<const DeviceConsumer*>(h.components.find("consumer"))) {
    return c->values();
  }
#endif
  return {};
}

// ---------------------------------------------------------------------------
// Timing. Collective MPI_MIN/MEAN/MAX of one rank's wall-clock seconds, the
// same reduction examples/layout_driver/main.cpp's own reduce_stat() uses.
// ---------------------------------------------------------------------------
struct Stat {
  double min_s = 0.0, mean_s = 0.0, max_s = 0.0;
};
[[nodiscard]] Stat reduce_stat(double local_seconds) {
  const int world = world_size();
  double mn = 0.0, mx = 0.0, sum = 0.0;
  MPI_Reduce(&local_seconds, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_seconds, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_seconds, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  Stat s;
  s.min_s  = mn;
  s.max_s  = mx;
  s.mean_s = sum / static_cast<double>(std::max(world, 1));
  return s;
}

/// /proc/loadavg's first field (1-minute load average), or -1.0 when this
/// host does not expose it (anything other than Linux).
[[nodiscard]] double one_minute_load() {
  std::ifstream in("/proc/loadavg");
  double load = -1.0;
  if (in) in >> load;
  return load;
}

// ---------------------------------------------------------------------------
// CLI.
// ---------------------------------------------------------------------------
struct Args {
  std::size_t  n = 1000;         // provider side length; provider has n*n cells
  std::int64_t periods = 5;
  std::string  fields = "host";  // "host" or "device"
  bool         verify = false;
  std::string  timing_path;
  bool         have_timing = false;
  std::string  parse_error;
};

[[nodiscard]] Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        args.parse_error = "E-BENCH-PARSE: '" + std::string(flag) + "' needs a value";
        return std::nullopt;
      }
      return std::string(argv[++i]);
    };
    if (a == "--n") {
      auto v = next("--n");
      if (!v) return args;
      try {
        args.n = static_cast<std::size_t>(std::stoull(*v));
      } catch (const std::exception&) {
        args.parse_error = "E-BENCH-PARSE: --n '" + *v + "' is not an integer";
        return args;
      }
    } else if (a == "--periods") {
      auto v = next("--periods");
      if (!v) return args;
      try {
        args.periods = std::stoll(*v);
      } catch (const std::exception&) {
        args.parse_error = "E-BENCH-PARSE: --periods '" + *v + "' is not an integer";
        return args;
      }
    } else if (a == "--fields") {
      auto v = next("--fields");
      if (!v) return args;
      if (*v != "host" && *v != "device") {
        args.parse_error = "E-BENCH-PARSE: --fields must be 'host' or 'device', got '" + *v + "'";
        return args;
      }
      args.fields = *v;
    } else if (a == "--verify") {
      args.verify = true;
    } else if (a == "--timing") {
      auto v = next("--timing");
      if (!v) return args;
      args.timing_path = *v;
      args.have_timing = true;
    } else {
      args.parse_error = "E-BENCH-PARSE: unrecognised argument '" + a + "'";
      return args;
    }
  }
  if (args.n < 1) {
    args.parse_error = "E-BENCH-PARSE: --n must be >= 1";
  }
  return args;
}

int agree_and_exit(int local_status) {
  int agreed = local_status;
  MPI_Allreduce(&local_status, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return agreed;
}

void print_run_stats(const fuse::RunStats& s, std::int64_t periods) {
  std::cout << "periods                  : " << s.periods << " (requested " << periods << ")\n"
            << "instructions_executed    : " << s.instructions_executed << "\n"
            << "remap_weight_builds       : " << s.remap_weight_builds << "\n"
            << "remap_weight_nonzeros     : " << s.remap_weight_nonzeros << "\n"
            << "device_dispatches         : " << s.device_dispatches << "\n"
            << "device_resident_arcs      : " << s.device_resident_arcs << "\n"
            << "device_boundary_copies    : " << s.device_boundary_copies << "\n"
            << "device_staged_copies      : " << s.device_staged_copies << "\n"
            << "device_transport_moves    : " << s.device_transport_moves << "\n";
}

// ---------------------------------------------------------------------------
// --verify: build and run the SAME small graph under Host and (if this
// build has one) Device, in this one process, and compare bitwise-ish.
// ---------------------------------------------------------------------------
int run_verify() {
  constexpr std::size_t kVerifyN = 6;        // 36 provider cells, 49 consumer
  constexpr std::int64_t kVerifyPeriods = 3;

  auto run_one = [&](Space space, bool device_fields, fuse::RunStats* out_stats,
                     std::vector<double>* out_values) -> fuse::Result<void> {
    fuse::Result<void> err;
    auto h = build(kVerifyN, device_fields, &err);
    if (!err.ok()) return err;
    for (auto& comp : h->components) {
      const auto r = comp.register_fields(h->registry, h->graph);
      if (!r.ok()) return r;
    }
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    RunnerConfig cfg;
    cfg.transport         = "mpi";
    cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
    cfg.transport_setup   = mpi_comm_setup();
    cfg.execution_space    = space;
    const auto init = runner.initialize(h->components, h->registry, cfg);
    if (!init.ok()) return init;
    for (std::int64_t k = 0; k < kVerifyPeriods; ++k) {
      const auto r = runner.run_period(h->components);
      if (!r.ok()) return r;
    }
    *out_stats = runner.stats();
    *out_values = consumer_values(*h);
    return fuse::Result<void>::success();
  };

  fuse::RunStats host_stats, dev_stats;
  std::vector<double> host_values, dev_values;

  const auto host_r = run_one(Space::Host, /*device_fields=*/false, &host_stats, &host_values);
  if (!host_r.ok()) {
    std::cerr << "FAIL: host verify run: " << host_r.error().code << ": "
              << host_r.error().message << "\n";
    return 1;
  }
  if (host_stats.remap_weight_builds <= 0) {
    std::cerr << "FAIL: host verify run built no real remap weights -- the "
                 "provider/consumer meshes coincided, so this is testing an "
                 "identity copy, not a weighted remap\n";
    return 1;
  }

  if constexpr (!kHasDeviceBackend) {
    std::cout << "verify: PASS (host only -- this build has no device backend)\n";
    return 0;
  }

  const auto dev_r = run_one(Space::Device, /*device_fields=*/false, &dev_stats, &dev_values);
  if (!dev_r.ok()) {
    std::cerr << "FAIL: device verify run: " << dev_r.error().code << ": "
              << dev_r.error().message << "\n";
    return 1;
  }
  if (dev_stats.device_dispatches <= 0) {
    std::cerr << "FAIL: device verify run reported zero device_dispatches -- "
                 "execution_space=Device was configured, but nothing ran on "
                 "the device\n";
    return 1;
  }
  if (host_values.size() != dev_values.size() || host_values.empty()) {
    std::cerr << "FAIL: host/device verify runs delivered different-sized "
                 "or empty consumer buffers (" << host_values.size() << " vs "
              << dev_values.size() << ")\n";
    return 1;
  }
  bool ok = true;
  for (std::size_t i = 0; i < host_values.size(); ++i) {
    const double a = host_values[i];
    const double b = dev_values[i];
    const double tol = 1e-14 * std::max(1.0, std::abs(a));
    if (std::abs(a - b) > tol) {
      std::cerr << "FAIL: consumer cell " << i << " host=" << a << " device=" << b
                << " (tolerance " << tol << ")\n";
      ok = false;
    }
  }
  if (!ok) return 1;
  std::cout << "verify: PASS (host and device agree to 1e-14 relative, "
            << host_values.size() << " consumer cells, remap_weight_builds="
            << host_stats.remap_weight_builds << " host / " << dev_stats.remap_weight_builds
            << " device)\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  const Args args = parse_args(argc, argv);
  if (!args.parse_error.empty()) {
    if (world_rank() == 0) std::cerr << args.parse_error << "\n";
    return agree_and_exit(1);
  }

  if (args.verify) {
    const int status = run_verify();
    return agree_and_exit(status);
  }

  const int rank = world_rank();
  const int world = world_size();
  const bool device_fields = (args.fields == "device");

  if (rank == 0) {
    std::cout << "world_size               : " << world << "\n"
              << "n (provider side length) : " << args.n << "\n"
              << "provider cells           : " << (args.n * args.n) << "\n"
              << "consumer cells           : " << ((args.n + 1) * (args.n + 1)) << "\n"
              << "fields                   : " << args.fields << "\n"
              << "FUSE_EXECUTION_SPACE     : "
              << (std::getenv("FUSE_EXECUTION_SPACE") ? std::getenv("FUSE_EXECUTION_SPACE")
                                                       : "(unset)")
              << "\n"
              << "load (1 min, before)     : " << one_minute_load() << "\n";
  }

  fuse::Result<void> build_err;
  const double t_build0 = MPI_Wtime();
  auto h = build(args.n, device_fields, &build_err);
  const double t_build1 = MPI_Wtime();
  if (!build_err.ok()) {
    if (rank == 0) {
      std::cerr << "build failed: " << build_err.error().code << ": "
                << build_err.error().message << "\n";
    }
    return agree_and_exit(1);
  }

  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    if (!r.ok()) {
      if (rank == 0) std::cerr << "register_fields failed: " << r.error().message << "\n";
      return agree_and_exit(1);
    }
  }

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  cfg.execution_space    = device_fields ? Space::Device : Space::Default;

  std::vector<std::pair<std::string, Stat>> timing;
  const double t_init0 = MPI_Wtime();
  const auto init = runner.initialize(h->components, h->registry, cfg);
  const double t_init1 = MPI_Wtime();
  timing.emplace_back("build", reduce_stat(t_build1 - t_build0));
  timing.emplace_back("initialize", reduce_stat(t_init1 - t_init0));
  {
    const int local_failed = init.ok() ? 0 : 1;
    int agreed = local_failed;
    MPI_Allreduce(&local_failed, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (agreed != 0) {
      if (rank == 0 && !init.ok()) {
        std::cerr << "initialize failed: " << init.error().code << ": "
                  << init.error().message << "\n";
      }
      return agree_and_exit(1);
    }
  }

  for (std::int64_t k = 1; k <= args.periods; ++k) {
    const double t0 = MPI_Wtime();
    const auto r = runner.run_period(h->components);
    const double t1 = MPI_Wtime();
    const int local_failed = r.ok() ? 0 : 1;
    timing.emplace_back("period_" + std::to_string(k), reduce_stat(t1 - t0));
    int agreed = local_failed;
    MPI_Allreduce(&local_failed, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (agreed != 0) {
      if (rank == 0 && !r.ok()) {
        std::cerr << "run_period failed at " << k << ": " << r.error().code << ": "
                  << r.error().message << "\n";
      }
      return agree_and_exit(1);
    }
  }

  if (rank == 0) {
    std::cout << "---- per-instruction wall time (s), MPI_MIN/MEAN/MAX over "
                 "ranks ----\n";
    std::cout.precision(6);
    for (const auto& [phase, stat] : timing) {
      std::cout << phase << " : min=" << stat.min_s << " mean=" << stat.mean_s
                << " max=" << stat.max_s << "\n";
    }
    std::cout << "---- RunStats (cumulative over " << args.periods << " periods) ----\n";
    print_run_stats(runner.stats(), args.periods);
    std::cout << "load (1 min, after)      : " << one_minute_load() << "\n";
  }

  if (args.have_timing && rank == 0) {
    std::ofstream out(args.timing_path, std::ios::trunc);
    if (!out) {
      std::cerr << "E-BENCH-TIMING: could not open '" << args.timing_path << "' for writing\n";
      return agree_and_exit(1);
    }
    out << "# fuse_bench_scaled_device_benchmark timing CSV\n"
        << "# schema: phase,ranks,n,fields,min_s,mean_s,max_s\n"
        << "phase,ranks,n,fields,min_s,mean_s,max_s\n";
    out.precision(9);
    for (const auto& [phase, stat] : timing) {
      out << phase << ',' << world << ',' << args.n << ',' << args.fields << ','
          << stat.min_s << ',' << stat.mean_s << ',' << stat.max_s << '\n';
    }
  }

  return agree_and_exit(0);
}
