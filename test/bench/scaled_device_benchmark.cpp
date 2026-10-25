// Scaled device benchmark (docs/dev/device-resident-arcs.md). Every row the
// rollout gate measured is a few hundred to a few
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
//   provider ((N+1)^2 native quad VERTICES, nodal) --Remap("consistent")-->
//   consumer (N^2 native quad CELLS, FV)
//
// ONE native structured quad mesh ("generated:quad:NxN"), two regions of
// different dimension and entity count: the provider's field is nodal
// (ContinuousGLL order 1, one value per vertex), the consumer's is plain
// FV (one value per cell). The two regions cannot coincide (different
// counts at every index), so Runner::Impl::regions_coincide() is false
// and the channel's Remap is a real interpolation, never the
// coincident-mesh identity copy -- built through the nodal-distributed
// path (fuse::build_nodal_dist_remap(), an ArborX-backed point-location
// search) rather than the general conservative weighted builder, whose
// host-only r3d/r2d clip this file measured as unusably slow at scale. A Diagnose instruction follows automatically
// (fuse::insert_derived_ops: any channel that emitted a Remap gets one),
// then Sink. Both fields carry Conservation::None, so no Constrain is
// inserted -- this program's own chain is Sample -> Remap -> Diagnose ->
// Sink; see build_graph()'s own closing comment for why a nodal "match-
// ing-orientation" Remap carries its cross-rank exchange INSIDE itself
// rather than beside it in a separate Transport instruction.
//
// PLACEMENT. At MPI world size >= 2, provider owns ranks [0, w/2) and
// consumer owns [w/2, w) -- disjoint, matching
// test/parallel/nodal_remap_runner_test.cpp's own DisjointRankSets shape,
// which proves this exact placement correct for a nodal "consistent"
// Remap at 1 through 5 real ranks. At world size 1 both components are
// placed on the whole (one-rank) world, since a disjoint split would
// leave one side on an empty RankSet (E-TRANSPORT-RANKS's own reasoning
// in test/parallel/runner_transport_test.cpp, though this file's nodal
// Remap fails a different, narrower way at an empty RankSet than a
// Transport does -- not reached here since world size 1 never splits).
//
// WHY NOT THE GENERAL WEIGHTED REMAP. An earlier version of this file
// used two differently-sized native meshes and the general (non-nodal)
// conservative builder instead -- a real Remap, and one with a genuine
// standalone Transport instruction beside it. Measured on this host:
// Runner::initialize()'s weight build took 13+ seconds at a mere 1e4
// cells and grew far faster than linearly with cell count (0.17s at 900
// cells, 0.98s at 2500, 13.4s at 10000), making the >= 1e6-cell target
// this file requires impractical within any reasonable
// TIMEOUT. Separately, and regardless of that cost, the general weighted
// builder's weight build at scale can produce a silently EMPTY conservative
// matrix (RunStats::remap_weight_nonzeros == 0) at rank counts above 1
// without an automatic remap-halo exchange across a disjoint placement.
// Hence this file's scaled row uses the nodal path instead.
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
  // ONE shared native quad mesh, two regions of different dimension: the
  // provider field lives on its (n+1)^2 VERTICES (dim 0, ContinuousGLL
  // order 1 -- a nodal field), the consumer field on its n^2 CELLS (dim
  // 2, plain FV). Different entity counts at every index, so
  // regions_coincide() is false and a real Remap is built -- but, more to
  // the point for a SCALED row, a nodal "consistent" Remap resolves
  // through fuse::build_nodal_dist_remap()'s own point-location search
  // (an ArborX-backed nearest/containment query, O(n log n)), not the
  // general conservative builder's broad-phase-plus-host-only-r3d-clip
  // path -- measured on this host to cost 13+ seconds of setup at a mere
  // 1e4 cells and growing far faster than linearly, which made it unusable at this file's target
  // size. The nodal path is also the one with its OWN cross-rank MPI
  // exchange (apply_nodal_dist_remap(), an MPI_Allgatherv under the
  // Remap instruction itself, docs/dev/device-resident-arcs.md's own
  // runtime-exchange section) -- see this function's own closing
  // comment on why that means no separate Transport instruction appears.
  // Independently generated (not one shared MeshDecl): two meshes of the
  // identical "generated:quad:NxN" spec, each its own NativeMeshBackend
  // slot. A shared single mesh with a vertex region and a cell region on
  // it hit a reproducible E-INTERP-NEWTON point-location refusal at a
  // multi-rank world size (global id 0, this file's own measurement log);
  // two independently generated meshes of the same spec do not.
  const auto mesh_provider = g.add_mesh(fuse::MeshDecl{
      .name = "provider_mesh",
      .source = "generated:quad:" + std::to_string(n) + "x" + std::to_string(n)});
  const auto mesh_consumer = g.add_mesh(fuse::MeshDecl{
      .name = "consumer_mesh",
      .source = "generated:quad:" + std::to_string(n) + "x" + std::to_string(n)});
  b.r_provider = g.add_region(
      fuse::Region{.name = "r_vertices", .mesh = mesh_provider, .dimension = 0});
  b.r_consumer = g.add_region(
      fuse::Region{.name = "r_cells", .mesh = mesh_consumer, .dimension = 2});

  b.provider_id = g.add_component(fuse::ComponentDecl{
      .name = "provider", .ranks = pl.provider, .interval = Rational{1}});
  b.consumer_id = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = pl.consumer, .interval = Rational{1}});

  const auto out = g.add_field(
      b.provider_id, fuse::FieldSpec{.name = "out", .region = b.r_provider,
                                     .disc = fuse::Discretization::ContinuousGLL,
                                     .order = 1, .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Provides});
  const auto in = g.add_field(
      b.consumer_id, fuse::FieldSpec{.name = "in", .region = b.r_consumer,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Requires});
  // "consistent": nodal SOURCE, matching this field transfer's own
  // orientation (a nodal TARGET would need "consistent-transpose"
  // instead) -- the one orientation test/runtime/runner_device_test.cpp's
  // own weighted_remap_then_relax() comment and
  // src/ir/rules_channel.cpp's resolve_remap() agree is routed through
  // the nodal-distributed path rather than refused
  // (E-RT-REMAP-NODAL-PARALLEL) when the component owning it spans more
  // than one rank, which `provider` here does at world size >= 2.
  const auto ch = g.add_channel("link", out, in);
  g.channel(ch).remap_method("consistent");
  // NOTE ON THE MISSING Transport INSTRUCTION. A nodal field whose
  // method matches its own orientation makes
  // fuse::remap_moves_placement() true, which routes this channel
  // through the Epoch-only nodal-distributed-remap path with NO separate
  // Transport node (test/parallel/nodal_remap_runner_test.cpp's own
  // header comment on DisjointRankSets explains the mechanism in full).
  // The cross-rank data movement still genuinely happens -- inside the
  // Remap instruction's own MPI_Allgatherv, not a Transport instruction
  // next to it -- so this file's instruction chain is Sample -> Remap ->
  // Diagnose -> Sink, not ... -> Remap -> Transport -> .... A SEPARATE,
  // non-nodal weighted channel that also carried a real Transport
  // instruction would need a remap-halo exchange, which
  // NativeMeshBackend lacks for this case; test/parallel/runner_transport_test.cpp's own
  // `parallel.runner_transport.{1,2,3}[.device]` ctest rows already
  // exercise a standalone Transport instruction (at a few cells, not
  // scaled) and are unaffected by anything in this file.
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
  const auto sched = schedule(h->graph, h->dag, clocks, fuse::make_region_sizes(h->graph, h->dag, h->backend));
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
  constexpr std::size_t kVerifyN = 6;        // 49 provider vertices, 36 consumer cells
  constexpr std::int64_t kVerifyPeriods = 3;

  auto run_one = [&](Space space, bool device_fields, fuse::RunStats* out_stats,
                     std::vector<double>* out_values) -> fuse::Result<void> {
    fuse::Result<void> err = fuse::Result<void>::success();
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
  // The nodal-distributed Remap keeps no RunStats counter of its own
  // (RunStats::remap_weight_builds only increments in the general
  // weighted branch, which a nodal "consistent" channel never reaches --
  // see build_graph()'s own comment on why this file uses the nodal path
  // instead). The check here is on the DELIVERED values instead: a
  // non-constant field reaching every one of kVerifyN*kVerifyN consumer
  // cells is not explainable by a coincident-region identity copy (the
  // provider/consumer regions here are different dimension and entity
  // count, so no identity path exists at all) or by a broadcast of one
  // scalar.
  if (host_values.size() != kVerifyN * kVerifyN) {
    std::cerr << "FAIL: host verify run delivered " << host_values.size()
              << " consumer cells, expected " << (kVerifyN * kVerifyN) << "\n";
    return 1;
  }
  if (std::all_of(host_values.begin(), host_values.end(),
                  [&](double v) { return v == host_values.front(); })) {
    std::cerr << "FAIL: host verify run delivered a constant field -- the nodal "
                 "Remap did not genuinely interpolate\n";
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
            << host_values.size() << " consumer cells, non-constant field)\n";
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
              << "provider vertices (nodal): " << ((args.n + 1) * (args.n + 1)) << "\n"
              << "consumer cells (FV)      : " << (args.n * args.n) << "\n"
              << "fields                   : " << args.fields << "\n"
              << "FUSE_EXECUTION_SPACE     : "
              << (std::getenv("FUSE_EXECUTION_SPACE") ? std::getenv("FUSE_EXECUTION_SPACE")
                                                       : "(unset)")
              << "\n"
              << "load (1 min, before)     : " << one_minute_load() << "\n";
  }

  fuse::Result<void> build_err = fuse::Result<void>::success();
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
