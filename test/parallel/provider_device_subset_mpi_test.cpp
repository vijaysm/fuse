// KNOWN_ISSUES.md D15 ("Cross check" track, owner instruction 2026-10-22:
// "For all judgement calls, cross verify on higgs"), Stage 3. Background:
// src/abi/provider_adapter.cpp's `unhosted_shell_` branch (register_fields())
// registers EVERY field as fuse::MemorySpace::Host, count 0, on a rank that
// does not host a Plugin component -- it never calls describe(), because the
// plugin is not loaded there. A hosting rank registers whatever memspace the
// plugin's own describe() reports, which for a device-resident field is
// FUSE_MEMSPACE_DEVICE. The lead's own concern (recorded here, not yet
// measured anywhere in this tree before this round): with
// FUSE_EXECUTION_SPACE=device, a non-hosting rank's Runner may see a
// Host-space, count-0 view for a field the hosting rank holds on the
// device, and either refuse with E-RT-SPACE or mis-count
// RunStats::device_boundary_copies.
//
// THE GRAPH, modeled on test/parallel/provider_subset_mpi_test.cpp's own
// "upstream"/"downstream" shape (same real base communicator, same
// fuse::abi_test::inject_component() seam for the native side):
//
//   "mixed" (test/abi/plugins/device_layout_provider.cpp, mounted as a
//   graph-declared LaunchMode::Plugin, RankSet{1, 1} -- rank 1 only) exports
//   "device_out" (Provides, kComponents components, FUSE_MEMSPACE_DEVICE
//   under a real device backend) through channel "link" to:
//
//   "consumer" (native, injected below, RankSet{0, 1} -- rank 0 only, the
//   SAME single-rank-host convention test/parallel/provider_subset_mpi_
//   test.cpp's own "downstream" uses) which Requires "in" (host memory).
//
// Only "device_out" is exercised -- the plugin's other two fields
// ("device_in", "echo_out") are left unbound, which test/abi/plugins/
// device_layout_provider.cpp's own p_bind()/p_advance() already tolerate
// (FieldSlot::bound stays false for an unregistered field; see that file's
// own D24/round5 comment on count==0).
//
// Both acceptable outcomes are handled explicitly by this file: if
// fuse_initialize() succeeds, this test keeps running and asserts the
// delivered value against an independently-known expected one, never a
// value read off a prior run and pasted in. If it fails, the fixture below
// reports the verbatim failure on every rank and FAILs cleanly rather than
// hanging or crashing -- this file does not attempt a fix either way.
//
// Compiled under FUSE_ENABLE_MPI only (test/CMakeLists.txt's own
// fuse_tests_gated_by(), the same convention provider_subset_mpi_test.cpp
// uses). SKIPPED BY NAME, at runtime, on a build with no device backend
// (test/runtime/runner_device_test.cpp's own kHasDeviceBackend/GTEST_SKIP
// convention, reused verbatim below) -- the file still compiles and the
// ctest row still exists on a host-only FUSE_ENABLE_MPI build, it just
// reports SKIPPED rather than being silently absent.
#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/units.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"

#include "test_support.hpp"

#ifndef FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN
#error "FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN (the built device_layout_provider shared object path) must be defined"
#endif

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

// Must match test/abi/plugins/device_layout_provider.cpp's own kComponents
// exactly -- restated here the same way test/abi/device_layout_test.cpp
// already restates it, since no ABI entry reports a plugin's own
// compile-time constant back to a caller.
constexpr int32_t kComponents = 3;

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

/// Identical to test/abi/device_layout_test.cpp's own ExecutionSpaceEnvGuard
/// (same FUSE_EXECUTION_SPACE env var, RunnerConfig::resolved_execution_
/// space(), src/runtime/runner.cpp) -- copied rather than shared, since
/// that class is local to its own translation unit and not exposed through
/// any header.
class ExecutionSpaceEnvGuard {
 public:
  explicit ExecutionSpaceEnvGuard(const char* value) {
    const char* prev = std::getenv("FUSE_EXECUTION_SPACE");
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
    if (value == nullptr) {
      unsetenv("FUSE_EXECUTION_SPACE");
    } else {
      setenv("FUSE_EXECUTION_SPACE", value, /*overwrite=*/1);
    }
  }
  ~ExecutionSpaceEnvGuard() {
    if (had_prev_) {
      setenv("FUSE_EXECUTION_SPACE", prev_value_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv("FUSE_EXECUTION_SPACE");
    }
  }
  ExecutionSpaceEnvGuard(const ExecutionSpaceEnvGuard&) = delete;
  ExecutionSpaceEnvGuard& operator=(const ExecutionSpaceEnvGuard&) = delete;

 private:
  bool        had_prev_ = false;
  std::string prev_value_;
};

/// test/parallel/provider_subset_mpi_test.cpp's own DownstreamSink, copied
/// rather than shared (that class is local to its own translation unit),
/// widened from one component to kComponents: a native, host-memory
/// component that records every value it receives for "in" -- fed by the
/// plugin's device-resident "device_out" through channel "link" -- so the
/// delivered value is independently observable without reaching into the
/// plugin's own C++ source. `hosted` (constructor argument, the caller's
/// own `graph.component(...).ranks.contains(rank)` check) decides whether
/// this rank registers the real kComponents-wide view or an empty one, the
/// same convention DownstreamSink's own header comment documents for D24.
class HostConsumer final : public fuse::Component {
 public:
  explicit HostConsumer(bool hosted) : hosted_(hosted) {}

  [[nodiscard]] std::string name() const override { return "consumer"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("consumer");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'consumer' component"};
    const std::size_t n = hosted_ ? 1 : 0;
    storage_.assign(n * static_cast<std::size_t>(kComponents), -1.0);
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      auto r = registry.bind(
          f, fuse::FieldRef{storage_.data(),
                             fuse::Layout{.stride = static_cast<std::size_t>(kComponents),
                                          .component_stride = 1,
                                          .count = n,
                                          .components = kComponents},
                             fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    if (storage_.empty()) {
      history.emplace_back(static_cast<std::size_t>(kComponents), -1.0);
    } else {
      history.push_back(storage_);
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return 0; }

  std::vector<std::vector<double>> history;

 private:
  bool                 hosted_;
  std::vector<double> storage_;
};

}  // namespace

// DISABLED_ (measured 2026-10-22, "Cross check" track, Stage 3, ssh:higgs,
// Open MPI 5.0.9 + Kokkos CUDA): fuse_initialize() fails before this test
// reaches the device-memspace question it was written to measure. Both
// ranks fail identically with E-RT-SIZE, "field 'mixed.device_out' is
// registered with 1x3 values but the plan sizes its region at 0x3"
// (fuse::Runner::Impl::resolve_buffers, src/runtime/runner.cpp:361).
// CAUSE (KNOWN_ISSUES.md D85): the setup call below passed
// MPI_Comm_c2f(MPI_COMM_WORLD) without has_communicator. MEASURED on higgs
// (lead, 2026-10-23): that handle is 0 under Open MPI 5.0.9 (0x44000000 under
// MPICH). READ in src/abi/participant.cpp: a handle of 0 without
// has_communicator=1 means "no communicator", so each rank took rank 0 of 1.
// INFERRED, not yet measured: a plugin at RankSet{1, 1} was then excluded from
// the region's sizing, giving the 0x3 above. The setup call now sets
// has_communicator = 1 (D86's field). NOT YET RUN on higgs: remove the
// DISABLED_ prefix only after that run shows the row reaches the
// device-memspace question; if it fails differently, D85's inference was
// incomplete and the new failure is the finding.
TEST(ProviderDeviceSubsetMpi,
    DISABLED_ADeviceResidentPluginFieldOnARankSubsetIsSeenCorrectlyByTheNonHostingRank) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";

  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_GE(world_size, 2) << "this row needs at least 2 ranks (RANKS 2)";

  const ExecutionSpaceEnvGuard env_guard("device");

  // "mixed" (the device_layout_provider plugin) is placed on rank 1 only --
  // RankSet{1, 1} -- per this round's own task. "consumer" (native,
  // injected below) is placed on rank 0 only, mirroring provider_subset_
  // mpi_test.cpp's own "downstream" convention exactly.
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:1"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:1"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto mixed = g.add_component(fuse::ComponentDecl{
      .name = "mixed", .ranks = fuse::RankSet{1, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN}});
  const auto consumer = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto out_f = g.add_field(
      mixed, fuse::FieldSpec{.name = "device_out", .region = r1, .components = kComponents,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Provides});
  // "mixed"'s own describe() (test/abi/plugins/device_layout_provider.cpp)
  // unconditionally reports THREE fields; ProviderAdapter::register_fields()
  // refuses outright ("describe() reports field 'X', which the graph does
  // not declare for this component") if the graph does not declare every
  // one of them, even a field this test never wires to a channel. Declared
  // here, left unconnected (W-NO-CONSUMER / no producer, warnings only, not
  // E-codes -- include/fuse/model/validate.hpp) -- this test exercises
  // "device_out" alone, per this round's own task.
  (void)g.add_field(
      mixed, fuse::FieldSpec{.name = "device_in", .region = r1, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  (void)g.add_field(
      mixed, fuse::FieldSpec{.name = "echo_out", .region = r1, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Provides});
  const auto in_f = g.add_field(
      consumer, fuse::FieldSpec{.name = "in", .region = r2, .components = kComponents,
                               .units = fuse::units::kKelvin,
                               .conservation = fuse::Conservation::None,
                               .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", out_f, in_f);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  const auto bytes = fuse::serialize(g);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // Real base communicator (every rank, identical input) -- same as
  // provider_subset_mpi_test.cpp's own setup() call.
  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
  // Open MPI's Fortran handle for MPI_COMM_WORLD is 0, which the ABI reads as
  // "none" unless has_communicator says the handle is deliberate (KNOWN_ISSUES.md D86, D85).
  opts.has_communicator = 1;
  const fuse_status setup_rc = t->setup(ctx, &opts, nullptr);
  std::fprintf(stdout, "XCHECK3 rank=%d world_size=%d setup_rc=%d\n", rank, world_size,
              setup_rc);
  std::fflush(stdout);
  ASSERT_EQ(setup_rc, FUSE_OK);

  auto consumer_sink = std::make_unique<HostConsumer>(/*hosted=*/rank == 0);
  HostConsumer* consumer_ptr = consumer_sink.get();
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "consumer", std::move(consumer_sink)));

  const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);
  fuse_error err{.struct_size = sizeof(err)};
  char buf[512] = {0};
  err.buffer = buf;
  err.capacity = sizeof(buf);
  t->last_error(ctx, nullptr, &err);
  std::fprintf(stdout, "XCHECK3 rank=%d init_rc=%d err.code=%s\n", rank, init_rc,
              buf[0] != '\0' ? buf : "<none>");
  std::fflush(stdout);

  if (init_rc != FUSE_OK) {
    // Per this round's own instruction: measure only, do not fix. Report
    // verbatim, on every rank, and stop here -- never a hang, never a
    // crash.
    std::fprintf(stderr, "rank %d: fuse_initialize failed: %s\n", rank, buf);
    std::fflush(stderr);
    FAIL() << "rank " << rank << ": fuse_initialize failed: " << buf;
    return;
  }

  // What memspace did THIS rank register for "device_out"? Via
  // fuse::abi_test::field_registry_of() -- the same reach-past-the-ABI seam
  // test/abi/conformance_test.cpp's own provider-parity checks already use
  // (KNOWN_ISSUES.md D15, Stage B-1, item 2) -- not the plugin's own
  // dlsym() accessor, which has no instance to read on rank 0 at all
  // (RankSet{1, 1} does not host "mixed" there, so register_fields() took
  // provider_adapter.cpp's own unhosted_shell_ branch, never dlopen()ing
  // the plugin).
  fuse::FieldRegistry* registry = fuse::abi_test::field_registry_of(ctx);
  ASSERT_NE(registry, nullptr);
  const fuse::Result<fuse::FieldRef> out_ref = registry->get(out_f);
  ASSERT_TRUE(out_ref.ok()) << out_ref.error().message;
  const fuse::FieldRef& out_view = out_ref.value();
  const char* memspace_name =
      out_view.space() == fuse::MemorySpace::Device ? "Device" : "Host";
  std::fprintf(stdout,
              "XCHECK3 rank=%d device_out_memspace=%s count=%zu components=%d\n", rank,
              memspace_name, out_view.layout().count, out_view.layout().components);
  std::fflush(stdout);

  const fuse_status run_rc = t->run(ctx, nullptr, nullptr);
  t->last_error(ctx, nullptr, &err);
  std::fprintf(stdout, "XCHECK3 rank=%d run_rc=%d err.code=%s\n", rank, run_rc,
              buf[0] != '\0' ? buf : "<none>");
  std::fflush(stdout);
  if (run_rc != FUSE_OK) {
    std::fprintf(stderr, "rank %d: fuse_run failed: %s\n", rank, buf);
    std::fflush(stderr);
    FAIL() << "rank " << rank << ": fuse_run failed: " << buf;
    return;
  }

  const fuse::RunStats stats = fuse::abi_test::run_stats(ctx);
  std::fprintf(stdout,
              "XCHECK3 rank=%d periods=%lld device_resident_arcs=%lld "
              "device_boundary_copies=%lld device_staged_copies=%lld\n",
              rank, static_cast<long long>(stats.periods),
              static_cast<long long>(stats.device_resident_arcs),
              static_cast<long long>(stats.device_boundary_copies),
              static_cast<long long>(stats.device_staged_copies));
  std::fflush(stdout);

  // Channel delivery in this tree's own convention lags by one period
  // (test/abi/device_layout_test.cpp's own kLayoutLagPeriods comment,
  // measured the same way there): period 0's read on "consumer" is the
  // pre-run cold default (-1.0 everywhere, this file's own sentinel, never
  // a value the plugin wrote). Only rank 0 (the hosting rank for
  // "consumer") has a non-empty `history` to assert against; rank 1 never
  // registers "consumer" and runs the SAME fuse_run() to completion with
  // nothing to check here.
  if (rank == 0) {
    ASSERT_FALSE(consumer_ptr->history.empty());
    const std::vector<double>& last = consumer_ptr->history.back();
    ASSERT_EQ(last.size(), static_cast<std::size_t>(kComponents));
    // device_layout_provider.cpp's own advance(): f(e, c) = 1000*(e+1) +
    // (c+1), entity e = 0 (one cell, "generated:line:1"), independently
    // derived from that file's own header comment, never a value read off
    // a prior run and pasted in.
    for (int c = 0; c < kComponents; ++c) {
      const double expected = 1000.0 * 1.0 + static_cast<double>(c + 1);
      std::fprintf(stdout, "XCHECK3 rank=0 component=%d delivered=%.17g expected=%.17g\n", c,
                  last[static_cast<std::size_t>(c)], expected);
    }
    std::fflush(stdout);
    for (int c = 0; c < kComponents; ++c) {
      const double expected = 1000.0 * 1.0 + static_cast<double>(c + 1);
      EXPECT_DOUBLE_EQ(last[static_cast<std::size_t>(c)], expected)
          << "component " << c;
    }
  }

  ASSERT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

#endif  // FUSE_HAVE_MPI
