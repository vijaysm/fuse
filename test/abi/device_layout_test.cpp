/// Field layout and host/device copies on a real CUDA/HIP/SYCL backend
/// (run on a CUDA host). The existing
/// MixedHostAndDeviceFieldsOnOneProviderConvergeThroughChannels row
/// (test/abi/conformance_test.cpp) is NOT extended here: that row's plugin
/// (test/abi/plugins/mixed_memspace_provider.c) originally dereferenced a
/// device pointer with an ordinary C store, which segfaulted on a real
/// device backend (exit code 139); it now writes through
/// mixed_memspace_device_write.cpp, the same Kokkos-kernel-launch
/// technique this file's own plugin uses. This file's own plugin
/// (test/abi/plugins/device_layout_provider.cpp) stays as its own,
/// separate test -- it covers 3-component layout and bidirectional
/// host/device copies that row does not.
///
/// THE GRAPH, one component "mixed" (device_layout_provider.cpp, mounted
/// as a graph-declared LaunchMode::Plugin, never dlopen()'d by this file
/// directly -- fuse_initialize() does that through
/// fuse::abi_detail::Registry::mount_dynamic()) between two native
/// components this file injects:
///
///   Source  --src_out (host)-->       device_in  (device, Requires)
///                                     "mixed"
///   device_out (device, Provides) --> Sink.layout_in (host)
///                                     "mixed"
///   echo_out (host, Provides)     --> Sink.echo_in  (host)
///
/// "device_out" has kComponents (> 1) components -- the layout question
/// (3a): advance() writes an injective f(entity, component) through
/// fuse_field_view's OWN reported stride/component_stride, and Sink
/// reads it back in host memory, so a Runner-side stride or
/// array-of-structures/structure-of-arrays mix-up changes what lands at
/// each (entity, component) and this file's own exact per-element
/// assertion catches it -- never a tolerance, since remap between two
/// geometrically-identical "generated:line:N" meshes is weight 1.0 per
/// cell, an exact identity map.
///
/// "src_out" -> "device_in" -> "echo_out" -> "echo_in" is the host ->
/// device -> host round trip (3b): a host-registered Provides field feeds
/// a device-resident Requires field on the plugin, which reads it back
/// with Kokkos::deep_copy (never a raw dereference) and echoes it,
/// unchanged, to a host Provides field -- so the test can compare what
/// the Sink actually received against what the Source actually sent,
/// through the Runner's own channel-delivery machinery, not an
/// out-of-band accessor.
///
/// A SECOND CMake target of the SAME plugin source
/// (FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN_HOST_REF,
/// test/CMakeLists.txt) reports "device_out"/"device_in" as
/// FUSE_MEMSPACE_HOST and runs its kernels on
/// Kokkos::DefaultHostExecutionSpace explicitly -- the all-host reference
/// run of the IDENTICAL arithmetic DeviceLayoutAndCopies.* compares the
/// real device run against, bit for bit, rather than a hand-recomputed
/// expected value.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/abi/provider.h"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"

#include "graph_fixture.hpp"
#include "test_support.hpp"

#ifndef FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN
#error "FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN (the built device_layout_provider shared object path) must be defined"
#endif
#ifndef FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN_HOST_REF
#error "FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN_HOST_REF must be defined"
#endif

namespace {

/// Must match test/abi/plugins/device_layout_provider.cpp's own
/// kComponents exactly -- there is no ABI entry that reports a plugin's
/// compile-time constant back to a caller, so this is restated here, the
/// same way every other fixture in this directory restates a plugin's
/// own formula (mixed_memspace_provider.c's header comment: "Physics:
/// identical to ... linear_provider.c's own single field").
constexpr int32_t kComponents = 3;
constexpr std::size_t kCells = 4;
// 5, not 3, so the "steady state" ranges below (after the channel-lag
// cold start) have more than one period to assert against.
constexpr int kPeriods = 5;

// Measured directly on this build (FUSE_DEVICE_DEBUG=1, on a CUDA host), not
// assumed: each channel hop in this graph delivers
// its exported value with a ONE-PERIOD lag (the same "lag-1" convention
// this tree's own graph_fixture.hpp names throughout,
// e.g. build_two_component_lag1_graph) -- a channel's own Remap/Diagnose
// instruction pair reads what the PREVIOUS period's Sample/mirror wrote,
// not the current period's. "device_out" -> "layout_in" is one hop, so
// period 0's Sink read is the pre-run cold default; periods 1..kPeriods-1
// hold the real value. "src_out" -> "device_in" -> (plugin echo) ->
// "echo_in" is TWO hops (one per channel), so periods 0 AND 1 are cold;
// periods 2..kPeriods-1 hold echo_in[period] == src_out's own value from
// period (period - 1).
constexpr int kLayoutLagPeriods = 1;
constexpr int kEchoLagPeriods = 2;

[[nodiscard]] const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// RAII guard: sets FUSE_EXECUTION_SPACE for its lifetime, restores
/// whatever (if anything) was there before -- the same convention
/// test/backend/mls_weights_test.cpp's own ScopedEnv and
/// test/runtime/runner_device_test.cpp use for the SAME environment
/// variable (RunnerConfig::resolved_execution_space(), src/runtime/
/// runner.cpp).
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
  bool had_prev_ = false;
  std::string prev_value_;
};

/// Same helper test/abi/conformance_test.cpp's own region_entity_count()
/// provides, duplicated here rather than shared across translation units
/// (that file keeps it anonymous-namespace local too) -- sizes a
/// component's own storage from the field's real mesh region instead of
/// hardcoding a count.
[[nodiscard]] fuse::Result<std::size_t> region_entity_count(const fuse::CouplingGraph& graph,
                                                             fuse::FieldId f,
                                                             fuse::SerialMeshBackend& backend) {
  for (std::uint32_t m = 0; m < graph.mesh_count(); ++m) {
    const fuse::Result<void> loaded = backend.load(graph.mesh(fuse::MeshId{m}));
    if (!loaded.ok()) return fuse::Failure{loaded.error()};
  }
  const fuse::Region& region = graph.region(graph.field(f).region);
  return backend.region_size(region);
}

/// "source": provides "src_out" (host, components = 1), one value per
/// entity, varying by both entity AND period -- g(e, step) = 7*(e+1) +
/// 0.5*step -- so a stale previous-period value reaching the Sink is
/// visibly wrong, not accidentally right.
class Source final : public fuse::Component {
 public:
  explicit Source(std::size_t cells) : storage_(cells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "source"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("source");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'source' component"};
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      auto r = registry.bind(
          f, fuse::FieldRef{storage_.data(),
                            fuse::Layout{.stride = 1, .component_stride = 1,
                                         .count = storage_.size(), .components = 1},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    ++steps_;
    for (std::size_t e = 0; e < storage_.size(); ++e) {
      storage_[e] = value_at(e, steps_);
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] static double value_at(std::size_t e, int step) {
    return 7.0 * static_cast<double>(e + 1) + 0.5 * static_cast<double>(step);
  }

 private:
  std::vector<double> storage_;
  int steps_ = 0;
};

/// "sink": requires "layout_in" (host, components = kComponents, from
/// "mixed".device_out) and "echo_in" (host, components = 1, from
/// "mixed".echo_out) -- by-name dispatch, the same convention
/// MixedMemspaceSink (test/abi/conformance_test.cpp) uses for its own two
/// independently-bound fields.
class Sink final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "sink"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("sink");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'sink' component"};
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      const fuse::FieldSpec& spec = graph.field(f);
      const fuse::Result<std::size_t> count = region_entity_count(graph, f, backend_);
      if (!count.ok()) return fuse::Result<void>{count.error()};
      if (spec.name == "layout_in") {
        layout_cells_ = count.value();
        layout_in_.assign(count.value() * static_cast<std::size_t>(spec.components), -1.0);
        auto r = registry.bind(
            f, fuse::FieldRef{layout_in_.data(),
                              fuse::Layout{.stride = static_cast<std::size_t>(spec.components),
                                           .component_stride = 1,
                                           .count = count.value(),
                                           .components = spec.components},
                              fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      } else if (spec.name == "echo_in") {
        echo_in_.assign(count.value(), -1.0);
        auto r = registry.bind(
            f, fuse::FieldRef{echo_in_.data(),
                              fuse::Layout{.stride = 1, .component_stride = 1,
                                           .count = count.value(), .components = 1},
                              fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      } else {
        continue;
      }
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    layout_history.push_back(layout_in_);
    echo_history.push_back(echo_in_);
    return fuse::Result<void>::success();
  }

  [[nodiscard]] double layout_at(std::size_t period, std::size_t e, std::size_t c) const {
    return layout_history.at(period).at(e * static_cast<std::size_t>(kComponents) + c);
  }

  std::vector<std::vector<double>> layout_history;
  std::vector<std::vector<double>> echo_history;

 private:
  fuse::SerialMeshBackend backend_;
  std::vector<double>     layout_in_;
  std::vector<double>     echo_in_;
  std::size_t             layout_cells_ = 0;
};

/// Builds the graph described in this file's own header comment.
/// `plugin_path` names either the real device plugin or its
/// FUSE_TEST_DEVICE_LAYOUT_FORCE_HOST-compiled host-reference twin --
/// the graph shape is identical either way (the plugin's own describe()
/// row decides FUSE_MEMSPACE_HOST vs FUSE_MEMSPACE_DEVICE for
/// "device_out"/"device_in", not this builder).
[[nodiscard]] fuse::CouplingGraph build_device_layout_graph(const char* plugin_path,
                                                            std::size_t cells) {
  fuse::CouplingGraph g;
  const auto m_src_up = g.add_mesh(
      fuse::MeshDecl{.name = "m_src_up", .source = "generated:line:" + std::to_string(cells)});
  const auto m_src_dn = g.add_mesh(
      fuse::MeshDecl{.name = "m_src_dn", .source = "generated:line:" + std::to_string(cells)});
  const auto m_layout_up = g.add_mesh(
      fuse::MeshDecl{.name = "m_layout_up", .source = "generated:line:" + std::to_string(cells)});
  const auto m_layout_dn = g.add_mesh(
      fuse::MeshDecl{.name = "m_layout_dn", .source = "generated:line:" + std::to_string(cells)});
  const auto m_echo_up = g.add_mesh(
      fuse::MeshDecl{.name = "m_echo_up", .source = "generated:line:" + std::to_string(cells)});
  const auto m_echo_dn = g.add_mesh(
      fuse::MeshDecl{.name = "m_echo_dn", .source = "generated:line:" + std::to_string(cells)});

  const auto r_src_up = g.add_region(fuse::Region{.name = "r_src_up", .mesh = m_src_up, .dimension = 1});
  const auto r_src_dn = g.add_region(fuse::Region{.name = "r_src_dn", .mesh = m_src_dn, .dimension = 1});
  const auto r_layout_up =
      g.add_region(fuse::Region{.name = "r_layout_up", .mesh = m_layout_up, .dimension = 1});
  const auto r_layout_dn =
      g.add_region(fuse::Region{.name = "r_layout_dn", .mesh = m_layout_dn, .dimension = 1});
  const auto r_echo_up = g.add_region(fuse::Region{.name = "r_echo_up", .mesh = m_echo_up, .dimension = 1});
  const auto r_echo_dn = g.add_region(fuse::Region{.name = "r_echo_dn", .mesh = m_echo_dn, .dimension = 1});

  const auto source = g.add_component(fuse::ComponentDecl{
      .name = "source", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto mixed = g.add_component(fuse::ComponentDecl{
      .name = "mixed", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin, .plugin_path = plugin_path}});
  const auto sink = g.add_component(fuse::ComponentDecl{
      .name = "sink", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

  const auto src_out = g.add_field(
      source, fuse::FieldSpec{.name = "src_out", .region = r_src_up, .components = 1,
                              .units = fuse::units::kKelvin,
                              .conservation = fuse::Conservation::None,
                              .intent = fuse::Intent::Provides});

  // Declaration order on "mixed" MUST match device_layout_provider.cpp's
  // own describe() row order (device_out, device_in, echo_out): bind()
  // dispatches by call-order counter, the same convention test/abi/
  // plugins/mixed_memspace_provider.c's own header comment documents and
  // relies on for its two fields.
  const auto device_out = g.add_field(
      mixed, fuse::FieldSpec{.name = "device_out", .region = r_layout_up, .components = kComponents,
                             .units = fuse::units::kKelvin,
                             .conservation = fuse::Conservation::None,
                             .intent = fuse::Intent::Provides});
  const auto device_in = g.add_field(
      mixed, fuse::FieldSpec{.name = "device_in", .region = r_src_dn, .components = 1,
                             .units = fuse::units::kKelvin,
                             .conservation = fuse::Conservation::None,
                             .intent = fuse::Intent::Requires});
  const auto echo_out = g.add_field(
      mixed, fuse::FieldSpec{.name = "echo_out", .region = r_echo_up, .components = 1,
                             .units = fuse::units::kKelvin,
                             .conservation = fuse::Conservation::None,
                             .intent = fuse::Intent::Provides});

  const auto layout_in = g.add_field(
      sink, fuse::FieldSpec{.name = "layout_in", .region = r_layout_dn, .components = kComponents,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto echo_in = g.add_field(
      sink, fuse::FieldSpec{.name = "echo_in", .region = r_echo_dn, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});

  const auto ch_src = g.add_channel("src_link", src_out, device_in);
  g.channel(ch_src).remap(fuse::RemapFlags::Conservative);
  const auto ch_layout = g.add_channel("layout_link", device_out, layout_in);
  g.channel(ch_layout).remap(fuse::RemapFlags::Conservative);
  const auto ch_echo = g.add_channel("echo_link", echo_out, echo_in);
  g.channel(ch_echo).remap(fuse::RemapFlags::Conservative);

  (void)r_src_up; (void)r_src_dn;  // named for readability; used through the fields above
  return g;
}

/// Sink's own recorded histories, COPIED OUT before fuse_destroy() frees
/// the Sink instance fuse_ctx owns -- an early version of this test kept
/// the raw Sink* past destroy() and read freed memory (a garbage
/// layout_history.size() and a std::bad_alloc downstream of it, both
/// measured directly before this struct existed); copying
/// the two histories here, inside run_device_layout_graph() while ctx is
/// still alive, is what makes the result safe to read after the function
/// returns.
struct RunOutcome {
  std::vector<std::vector<double>> layout_history;
  std::vector<std::vector<double>> echo_history;
  fuse::RunStats                   stats{};
  fuse_field_view                  device_out_view{sizeof(fuse_field_view)};
};

/// dlsym()s device_layout_provider.cpp's own test-only accessor on the
/// SAME shared object fuse_initialize() already mounted (dlopen() on an
/// already-loaded path returns a handle to the same mapping, refcounted
/// -- not a second load). MUST be called before fuse_destroy(): an
/// earlier version of this file called it from the TEST body, after
/// run_device_layout_graph() (and therefore fuse_destroy()) had already
/// returned, and read back an all-zero view -- fuse_destroy() dlcloses
/// every mounted plugin (src/abi/registry.cpp), dropping this .so's
/// refcount to zero between calls, so the "second" dlopen here was
/// actually a FRESH load whose g_last_instance had never been set.
[[nodiscard]] fuse_field_view read_last_device_out_view(const char* plugin_path) {
  void* handle = dlopen(plugin_path, RTLD_NOW);
  if (handle == nullptr) {
    ADD_FAILURE() << "dlopen(" << plugin_path << "): " << dlerror();
    return fuse_field_view{sizeof(fuse_field_view)};
  }
  using AccessorFn = fuse_field_view (*)();
  auto fn = reinterpret_cast<AccessorFn>(
      dlsym(handle, "fuse_test_device_layout_provider_last_device_out_view"));
  if (fn == nullptr) {
    ADD_FAILURE() << "dlsym(fuse_test_device_layout_provider_last_device_out_view): " << dlerror();
    dlclose(handle);
    return fuse_field_view{sizeof(fuse_field_view)};
  }
  const fuse_field_view v = fn();
  dlclose(handle);
  return v;
}

/// Runs the graph `periods` periods, under `execution_space` ("host",
/// "device", or nullptr for the process default), and returns the
/// Sink's own recorded histories (copied) plus RunStats.
[[nodiscard]] RunOutcome run_device_layout_graph(const char* plugin_path,
                                                 const char* execution_space, int periods) {
  // EXPECT_*/ADD_FAILURE(), never ASSERT_*/FAIL(): this function returns
  // RunOutcome, not void, and gtest's ASSERT_* macros expand to a bare
  // `return;` on failure, which does not compile against a non-void
  // return type -- each step below checks and returns RunOutcome{}
  // explicitly instead.
  const ExecutionSpaceEnvGuard env_guard(execution_space);
  const fuse::CouplingGraph g = build_device_layout_graph(plugin_path, kCells);
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  EXPECT_NE(t, nullptr);
  if (t == nullptr) return RunOutcome{};
  fuse_context_out co{sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  if (ctx == nullptr) return RunOutcome{};
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  EXPECT_TRUE(fuse::abi_test::inject_component(ctx, "source", std::make_unique<Source>(kCells)));
  auto sink_owned = std::make_unique<Sink>();
  Sink* sink_raw = sink_owned.get();
  EXPECT_TRUE(fuse::abi_test::inject_component(ctx, "sink", std::move(sink_owned)));

  const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);
  if (init_rc != FUSE_OK) {
    fuse_error err{sizeof(err)};
    char buf[512] = {0};
    err.buffer = buf;
    err.capacity = sizeof(buf);
    (void)t->last_error(ctx, nullptr, &err);
    ADD_FAILURE() << "fuse_initialize (" << (execution_space ? execution_space : "<default>")
                  << "): " << buf;
    (void)t->destroy(ctx, nullptr, nullptr);
    return RunOutcome{};
  }

  fuse_run_request rr{sizeof(rr)};
  rr.periods = periods;
  const fuse_status run_rc = t->run(ctx, &rr, nullptr);
  EXPECT_EQ(run_rc, FUSE_OK);
  if (run_rc != FUSE_OK) {
    (void)t->destroy(ctx, nullptr, nullptr);
    return RunOutcome{};
  }

  RunOutcome out;
  out.layout_history = sink_raw->layout_history;  // copy: sink_raw dies at destroy() below
  out.echo_history = sink_raw->echo_history;
  out.stats = fuse::abi_test::run_stats(ctx);
  // Read BEFORE destroy(): fuse_destroy() dlcloses every mounted plugin
  // (src/abi/registry.cpp), and this file's own dlopen() of the SAME
  // path afterward would otherwise be a fresh load with an empty
  // g_last_instance, not the already-loaded mapping this comment's own
  // header note assumes.
  out.device_out_view = read_last_device_out_view(plugin_path);

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// 3a. LAYOUT: fuse_field_view's reported stride/component_stride against
// what the Runner then reads, and the exact value at every (entity,
// component).
// ---------------------------------------------------------------------------
TEST(DeviceLayoutAndCopies, DeviceOutLayoutMatchesDerivedAoSAndEveryElementIsExact) {
#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
  GTEST_SKIP() << "no device backend on this build; see "
                  "MixedHostAndDeviceFieldsOnOneProviderConvergeThroughChannels and "
                  "the device-build run on the GPU host";
#else
  const RunOutcome outcome =
      run_device_layout_graph(FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN, "device", kPeriods);
  ASSERT_EQ(outcome.layout_history.size(), static_cast<std::size_t>(kPeriods));

  // fuse_field_view's OWN reported layout against ProviderAdapter::
  // register_fields()'s own allocation convention, derived (not read off
  // a run): array-of-structures, stride = components, component_stride =
  // 1, count = the region's entity count, components = kComponents.
  const fuse_field_view& view = outcome.device_out_view;
  EXPECT_EQ(view.memspace, FUSE_MEMSPACE_DEVICE);
  EXPECT_EQ(view.components, kComponents);
  EXPECT_EQ(view.count, kCells);
  EXPECT_EQ(view.stride, kComponents) << "expected array-of-structures: stride == components";
  EXPECT_EQ(view.component_stride, 1) << "expected array-of-structures: component_stride == 1";

  // Every (entity, component) exactly, every STEADY-STATE period -- f(e,
  // c) is constant across periods by construction, so a stale or rotated
  // delivery would show up as a mismatch even though the VALUE looks
  // plausible. Period 0 is this channel's own one-period cold start
  // (kLayoutLagPeriods, this file's own header comment) and is asserted
  // separately, below, against the Sink's OWN pre-run default rather
  // than f(e, c) -- asserting f(e, c) there would be the wrong
  // expectation, not a weakened one.
  for (int period = kLayoutLagPeriods; period < kPeriods; ++period) {
    const std::vector<double>& row = outcome.layout_history[static_cast<std::size_t>(period)];
    for (std::size_t e = 0; e < kCells; ++e) {
      for (int c = 0; c < kComponents; ++c) {
        const double expected = 1000.0 * static_cast<double>(e + 1) + static_cast<double>(c + 1);
        EXPECT_EQ(row.at(e * static_cast<std::size_t>(kComponents) + static_cast<std::size_t>(c)),
                  expected)
            << "period " << period << " entity " << e << " component " << c;
      }
    }
  }
  // The cold-start period itself: the Sink's own pre-run default
  // (-1.0, Sink::register_fields()'s own .assign(..., -1.0)), proving
  // the channel has not silently substituted a zero or a stale value
  // for "nothing delivered yet" -- FUSE's own "no silent fallback"
  // convention, named throughout this tree.
  {
    const std::vector<double>& cold = outcome.layout_history[0];
    for (double v : cold) EXPECT_EQ(v, -1.0) << "cold-start period should be the Sink's own default";
  }
#endif
}

// ---------------------------------------------------------------------------
// 3b. COPIES, both directions, bit for bit against an all-host reference
// run of the identical arithmetic -- and RunStats::device_boundary_copies
// against a count derived from the graph's own instruction list.
// ---------------------------------------------------------------------------
TEST(DeviceLayoutAndCopies, HostDeviceHostRoundTripMatchesHostReferenceExactly) {
#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
  GTEST_SKIP() << "no device backend on this build";
#else
  const RunOutcome host_outcome =
      run_device_layout_graph(FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN_HOST_REF, "host", kPeriods);
  const RunOutcome dev_outcome =
      run_device_layout_graph(FUSE_TEST_DEVICE_LAYOUT_PROVIDER_PLUGIN, "device", kPeriods);
  const fuse::RunStats& dev_stats = dev_outcome.stats;

  // Steady state only (kEchoLagPeriods, this file's own header comment):
  // echo_in[period] == src_out's own value from period (period - 1),
  // the ONE-PERIOD lag of the SECOND channel hop (echo_out -> echo_in)
  // on top of the first (src_out -> device_in).
  ASSERT_EQ(host_outcome.echo_history.size(), dev_outcome.echo_history.size());
  for (std::size_t period = static_cast<std::size_t>(kEchoLagPeriods);
       period < host_outcome.echo_history.size(); ++period) {
    ASSERT_EQ(host_outcome.echo_history[period].size(), dev_outcome.echo_history[period].size());
    for (std::size_t e = 0; e < host_outcome.echo_history[period].size(); ++e) {
      EXPECT_EQ(host_outcome.echo_history[period][e], dev_outcome.echo_history[period][e])
          << "echo mismatch, period " << period << " entity " << e;
      // Independently against the Source's own closed-form value, so a
      // bug shared by BOTH the host-reference and device plugins (e.g.
      // both reading the wrong period) cannot hide behind their mutual
      // agreement.
      EXPECT_EQ(host_outcome.echo_history[period][e],
                Source::value_at(e, static_cast<int>(period) - kEchoLagPeriods + 1))
          << "echo vs. closed-form source value, period " << period << " entity " << e;
    }
  }
  ASSERT_EQ(host_outcome.layout_history.size(), dev_outcome.layout_history.size());
  for (std::size_t period = static_cast<std::size_t>(kLayoutLagPeriods);
       period < host_outcome.layout_history.size(); ++period) {
    EXPECT_EQ(host_outcome.layout_history[period], dev_outcome.layout_history[period])
        << "device_out delivery mismatch, period " << period;
  }

  // RunStats::device_boundary_copies (docs/dev/device-resident-arcs.md,
  // "The boundary rule"), derived from this graph's own instruction
  // list -- measured with FUSE_DEVICE_DEBUG=1 on a CUDA host to confirm
  // which arcs actually cross, not assumed from the field declarations
  // alone. The trace's own "[device-place] arc"
  // lines, one per arc:
  //   arc %0  host mirrored  source.src_out        (component, Provides)
  //   arc %1  device         mixed.device_out       (component, Provides)
  //   arc %2  host mirrored  mixed.echo_out          (component, Provides)
  //   arc %9  device         mixed.device_in         (component, Requires)
  //   arc %10 host           sink.layout_in          (component, Requires)
  //   arc %11 host           sink.echo_in            (component, Requires)
  // "echo_out" is host-registered (FUSE_MEMSPACE_HOST, always, in both
  // CMake targets) but is STILL mirrored to device: once FUSE's own
  // execution_space is Device, the echo_link channel's own Remap/
  // Diagnose instructions run in device space regardless of what EITHER
  // of echo_out/echo_in's own memspace says, and rule 1 mirrors any
  // host-registered field such an instruction reads -- "both ends are
  // host" does not exempt the channel between them. Measured, not an
  // assumption this test's own first draft made (and got wrong: it
  // expected 2*kPeriods, not the 4*kPeriods actually observed). Per
  // period: two mirror-in crossings (src_out, echo_out) + two copy-out
  // crossings at a Sink (layout_in, echo_in) = 4.
  const std::int64_t expected_boundary_copies = 4 * kPeriods;
  EXPECT_EQ(dev_stats.device_boundary_copies, expected_boundary_copies)
      << "derived from the boundary rule plus the FUSE_DEVICE_DEBUG=1 arc trace: " << kPeriods
      << " periods x (2 mirrors [src_out, echo_out] + 2 sink copies [layout_in, echo_in]); "
         "see this test's own comment if the derivation needs a term";
#endif
}
