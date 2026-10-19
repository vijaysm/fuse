/// ABI-PROVIDER-CONFORMANCE and its negative rows (build-evidence/abi-matrix.toml,
/// S6 phase 4 "Adapter host" track): ONE graph -- the two-component lag-1
/// fixture, then the subcycled-mean fixture, both from graph_fixture.hpp --
/// run four ways: native C++ components, the same "upstream" component as a
/// C plugin built as a shared object in this test tree, the same component
/// written in Fortran as a plugin, and the same component through the
/// participant ABI (the test itself plays "upstream", driving
/// fuse_begin_step()/fuse_end_step() directly, exactly as
/// test/abi/participant_test.cpp's own lifecycle tests do). "downstream" is
/// always a native fuse::Component (DownstreamSink, below), injected the
/// same way in every kind, so every kind's run is checked against the
/// SAME observer.
///
/// PHYSICS. The one thing either kind implements: "upstream" writes
/// `out = t_end` (the exact rational cast to double) at every advance(),
/// where t_end is the window IEEE754 double division gives bit-identical
/// results across a C, a Fortran and a C++ translation unit on the same
/// hardware -- the property this harness needs and nothing more elaborate
/// buys (test/abi/plugins/linear_provider.{c,f90} implement the identical
/// formula). FUSE's own accumulation (AccumOp::Mean) does the averaging
/// across upstream's four substeps for the subcycled-mean fixture; this is
/// transparent to EVERY kind here exactly as it already is to a native
/// Component (AbiParticipant.SubcycledMeanAccumulation already pins that
/// for the participant direction).
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/abi/provider.h"
#include "fuse/base/rational.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"

#include "graph_fixture.hpp"
#include "provider_adapter.hpp"
#include "registry.hpp"
#include "test_support.hpp"

#ifndef FUSE_TEST_C_PROVIDER_PLUGIN
#error "FUSE_TEST_C_PROVIDER_PLUGIN (the built linear_provider shared object path) must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_VERSION
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_VERSION must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ADVANCE
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ADVANCE must be defined"
#endif
// review-2026-10-11 abi: TD-6/AB-6's own plugin variants.
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CREATE
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CREATE must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_BIND
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_BIND must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_WINDOW
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_WINDOW must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ELEMENT
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ELEMENT must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CAPABILITIES
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CAPABILITIES must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE_EXTRA
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE_EXTRA must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_SAVE
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_SAVE must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_BAD_RESTORE
#error "FUSE_TEST_C_PROVIDER_PLUGIN_BAD_RESTORE must be defined"
#endif
#ifndef FUSE_TEST_C_PROVIDER_PLUGIN_OLD_STEP_REPORT
#error "FUSE_TEST_C_PROVIDER_PLUGIN_OLD_STEP_REPORT must be defined"
#endif
#ifndef FUSE_TEST_FORTRAN_PROVIDER_PLUGIN
#error "FUSE_TEST_FORTRAN_PROVIDER_PLUGIN (the built linear_provider_f shared object path) must be defined"
#endif

namespace {

enum class Kind { Native, CPlugin, FortranPlugin, Participant };

[[nodiscard]] const char* KindName(Kind k) {
  switch (k) {
    case Kind::Native:        return "Native";
    case Kind::CPlugin:       return "CPlugin";
    case Kind::FortranPlugin: return "FortranPlugin";
    case Kind::Participant:   return "Participant";
  }
  return "Unknown";
}

[[nodiscard]] const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

// --------------------------------------------------------- native kind ----

/// The native C++ implementation of "upstream": out = t_end, exactly the
/// formula test/abi/plugins/linear_provider.{c,f90} implement, so the
/// Native run is a real member of the comparison, not a privileged
/// reference the other three are merely hoped to match.
/// Sizes its own storage from the field's real mesh region (the same thing
/// ProviderAdapter::register_fields() does generically) rather than
/// hardcoding a count -- the graph fixtures size their regions by their own
/// `cells` argument, and a component that assumed 1 would silently read or
/// write past a 1-element buffer the moment a fixture's default (4) is used.
[[nodiscard]] inline fuse::Result<std::size_t> region_entity_count(
    const fuse::CouplingGraph& graph, fuse::FieldId f, fuse::SerialMeshBackend& backend) {
  // SerialMeshBackend::region_size() indexes its own loaded-mesh vector
  // directly by MeshId, which requires every mesh 0..N-1 to have been
  // load()ed in that order first (the same reason
  // src/abi/provider_adapter.cpp loads every mesh in the graph up front,
  // not just the one field `f`'s own region happens to reference).
  for (std::uint32_t m = 0; m < graph.mesh_count(); ++m) {
    fuse::Result<void> loaded = backend.load(graph.mesh(fuse::MeshId{m}));
    if (!loaded.ok()) return fuse::Failure{loaded.error()};
  }
  const fuse::Region& region = graph.region(graph.field(f).region);
  return backend.region_size(region);
}

class UpstreamNative final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "upstream"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("upstream");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'upstream' component"};
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      const fuse::Result<std::size_t> count = region_entity_count(graph, f, backend_);
      if (!count.ok()) return fuse::Result<void>{count.error()};
      storage_.assign(count.value(), 0.0);
      auto r = registry.bind(
          f, fuse::FieldRef{storage_.data(), fuse::Layout{.stride = 1, .count = count.value()},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    for (double& v : storage_) v = ctx.t_end.to_double();
    return fuse::Result<void>::success();
  }

 private:
  fuse::SerialMeshBackend backend_;
  std::vector<double>     storage_;
};

/// Always the observer, in every kind: records the value "downstream.in"
/// holds at the end of every one of its own advance() calls (already
/// current by then -- Component::advance()'s own contract). Every entity
/// carries the identical value by construction (UpstreamNative and the two
/// plugins all write ONE scalar to every entity), so history[k] = the
/// field's entity 0 is a complete observation, not a sampled one.
class DownstreamSink final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "downstream"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("downstream");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'downstream' component"};
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      const fuse::Result<std::size_t> count = region_entity_count(graph, f, backend_);
      if (!count.ok()) return fuse::Result<void>{count.error()};
      storage_.assign(count.value(), -1.0);
      auto r = registry.bind(
          f, fuse::FieldRef{storage_.data(), fuse::Layout{.stride = 1, .count = count.value()},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    history.push_back(storage_.empty() ? -1.0 : storage_.front());
    return fuse::Result<void>::success();
  }

  std::vector<double> history;

 private:
  fuse::SerialMeshBackend backend_;
  std::vector<double>     storage_;
};

// ------------------------------------------------------ the run harness ----

struct RunOutcome {
  bool                 ok = true;
  std::string           error_code;
  std::vector<double>  history;
};

[[nodiscard]] RunOutcome run_graph(const std::vector<std::byte>& bytes, Kind kind, int periods,
                                   int upstream_ticks_per_period = 1, int cells = 4) {
  RunOutcome out;
  std::vector<double> up_storage(static_cast<std::size_t>(cells), 0.0);
  const auto* t = Table();
  if (t == nullptr) {
    out.ok = false;
    out.error_code = "E-TEST-NO-TABLE";
    return out;
  }

  fuse_context_out co{sizeof(co)};
  if (t->create(nullptr, nullptr, &co) != FUSE_OK) {
    out.ok = false;
    out.error_code = "E-TEST-CREATE";
    return out;
  }
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  if (t->load_graph(ctx, &src, nullptr) != FUSE_OK) {
    out.ok = false;
    out.error_code = "E-TEST-LOAD";
    (void)t->destroy(ctx, nullptr, nullptr);
    return out;
  }

  // "downstream" is ALWAYS native, injected the same way in every kind.
  auto sink = std::make_unique<DownstreamSink>();
  DownstreamSink* sink_ptr = sink.get();
  if (!fuse::abi_test::inject_component(ctx, "downstream", std::move(sink))) {
    out.ok = false;
    out.error_code = "E-TEST-INJECT-DOWNSTREAM";
    (void)t->destroy(ctx, nullptr, nullptr);
    return out;
  }

  fuse::abi_detail::Registry registry;
  bool upstream_injected = true;

  switch (kind) {
    case Kind::Native: {
      if (!fuse::abi_test::inject_component(ctx, "upstream", std::make_unique<UpstreamNative>())) {
        out.ok = false;
        out.error_code = "E-TEST-INJECT-UPSTREAM";
      }
      break;
    }
    case Kind::CPlugin:
    case Kind::FortranPlugin: {
      const std::string path = (kind == Kind::CPlugin) ? FUSE_TEST_C_PROVIDER_PLUGIN
                                                        : FUSE_TEST_FORTRAN_PROVIDER_PLUGIN;
      fuse::Result<void> mounted = registry.mount_dynamic("upstream", path);
      if (!mounted.ok()) {
        out.ok = false;
        out.error_code = mounted.error().code;
        break;
      }
      fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
      if (!extracted.ok()) {
        out.ok = false;
        out.error_code = extracted.error().code;
        break;
      }
      if (!fuse::abi_test::inject_component(ctx, "upstream", std::move(extracted).value())) {
        out.ok = false;
        out.error_code = "E-TEST-INJECT-UPSTREAM";
      }
      break;
    }
    case Kind::Participant: {
      upstream_injected = false;  // "upstream" stays the external placeholder
      break;
    }
  }
  if (!out.ok) {
    (void)t->destroy(ctx, nullptr, nullptr);
    return out;
  }

  if (!upstream_injected) {
    // The test itself plays "upstream": register its storage the way any
    // external participant would, before initialize().
    fuse_field_name fname{sizeof(fname)};
    fname.component = "upstream";
    fname.field = "out";
    fuse_field_ref fref{sizeof(fref)};
    if (t->field_handle(ctx, &fname, &fref) != FUSE_OK) {
      out.ok = false;
      out.error_code = "E-TEST-FIELD-HANDLE";
      (void)t->destroy(ctx, nullptr, nullptr);
      return out;
    }
    fuse_field_view fview{sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = up_storage.size();
    fview.owned_count = up_storage.size();
    fview.stride = 1;
    fview.data = up_storage.data();
    if (t->register_field(ctx, &fview, nullptr) != FUSE_OK) {
      out.ok = false;
      out.error_code = "E-TEST-REGISTER-FIELD";
      (void)t->destroy(ctx, nullptr, nullptr);
      return out;
    }
  }

  if (t->initialize(ctx, nullptr, nullptr) != FUSE_OK) {
    fuse_error err{sizeof(err)};
    (void)t->last_error(ctx, nullptr, &err);
    out.ok = false;
    out.error_code = err.code != nullptr ? err.code : "E-TEST-INITIALIZE";
    (void)t->destroy(ctx, nullptr, nullptr);
    return out;
  }

  if (upstream_injected) {
    fuse_run_request rr{sizeof(rr)};
    rr.periods = periods;
    if (t->run(ctx, &rr, nullptr) != FUSE_OK) {
      fuse_error err{sizeof(err)};
      (void)t->last_error(ctx, nullptr, &err);
      out.ok = false;
      out.error_code = err.code != nullptr ? err.code : "E-TEST-RUN";
      (void)t->destroy(ctx, nullptr, nullptr);
      return out;
    }
  } else {
    fuse_step_request req{sizeof(req)};
    req.component = "upstream";
    const int ticks = periods * upstream_ticks_per_period;
    for (int tick = 0; tick < ticks; ++tick) {
      fuse_step_report rep{sizeof(rep)};
      if (t->begin_step(ctx, &req, &rep) != FUSE_OK) {
        out.ok = false;
        out.error_code = "E-TEST-BEGIN-STEP";
        break;
      }
      // Write the SAME formula the native/plugin kinds implement, at the
      // EXACT window the schedule just reported -- never a double computed
      // any other way, so this kind is not privileged to a simpler answer.
      const fuse::Rational t_end = fuse::Rational::from_record(
          fuse::ExactTimeRecord{rep.t_end.numerator, rep.t_end.denominator}).value();
      for (double& v : up_storage) v = t_end.to_double();
      if (t->end_step(ctx, &req, &rep) != FUSE_OK) {
        out.ok = false;
        out.error_code = "E-TEST-END-STEP";
        break;
      }
    }
  }

  const std::vector<double> history = sink_ptr->history;
  (void)t->destroy(ctx, nullptr, nullptr);
  if (out.ok) out.history = history;
  return out;
}

constexpr Kind kAllKinds[] = {Kind::Native, Kind::CPlugin, Kind::FortranPlugin,
                              Kind::Participant};

}  // namespace

// ConformanceTest.LagOneGraphAllFourKindsMatchNative: the two-component
// lag-1 fixture, interval 1 on both sides (one tick per period), run all
// four ways; every kind's recorded "downstream.in" history must equal the
// Native run's, value for value.
TEST(ConformanceTest, LagOneGraphAllFourKindsMatchNative) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(
      /*upstream_interval=*/fuse::Rational{1}, /*cells=*/1);
  constexpr int kPeriods = 3;

  const RunOutcome native = run_graph(bytes, Kind::Native, kPeriods, /*ticks_per_period=*/1,
                                      /*cells=*/1);
  ASSERT_TRUE(native.ok) << native.error_code;
  ASSERT_EQ(native.history.size(), static_cast<std::size_t>(kPeriods));

  for (const Kind k : kAllKinds) {
    if (k == Kind::Native) continue;
    const RunOutcome r = run_graph(bytes, k, kPeriods, /*ticks_per_period=*/1, /*cells=*/1);
    ASSERT_TRUE(r.ok) << KindName(k) << ": " << r.error_code;
    ASSERT_EQ(r.history.size(), native.history.size()) << KindName(k);
    for (std::size_t i = 0; i < native.history.size(); ++i) {
      EXPECT_DOUBLE_EQ(r.history[i], native.history[i])
          << KindName(k) << " period " << i << " disagrees with the native run";
    }
  }
}

// ConformanceTest.SubcycledMeanGraphAllFourKindsMatchNative: upstream
// subcycles 4x per common period and exports under AccumOp::Mean
// (KNOWN_ISSUES.md D12's own fixture) -- "downstream" must see the MEAN of
// upstream's four substep values for every kind, not just the native one.
TEST(ConformanceTest, SubcycledMeanGraphAllFourKindsMatchNative) {
  const auto bytes = fuse::abi_test::build_subcycled_mean_bytes(
      /*upstream_interval=*/fuse::Rational{1, 4}, /*cells=*/1);
  constexpr int kPeriods = 2;
  constexpr int kTicksPerPeriod = 4;

  const RunOutcome native =
      run_graph(bytes, Kind::Native, kPeriods, kTicksPerPeriod, /*cells=*/1);
  ASSERT_TRUE(native.ok) << native.error_code;
  ASSERT_EQ(native.history.size(), static_cast<std::size_t>(kPeriods));

  for (const Kind k : kAllKinds) {
    if (k == Kind::Native) continue;
    const RunOutcome r = run_graph(bytes, k, kPeriods, kTicksPerPeriod, /*cells=*/1);
    ASSERT_TRUE(r.ok) << KindName(k) << ": " << r.error_code;
    ASSERT_EQ(r.history.size(), native.history.size()) << KindName(k);
    for (std::size_t i = 0; i < native.history.size(); ++i) {
      EXPECT_DOUBLE_EQ(r.history[i], native.history[i])
          << KindName(k) << " period " << i << " disagrees with the native run";
    }
  }
}

// ---- negative rows: a dlopen()'d plugin that misbehaves is refused, named,
// and never silently substituted (build-evidence/abi-matrix.toml). --------

TEST(ConformanceTest, PluginTableVersionMismatchIsRefused) {
  fuse::abi_detail::Registry registry;
  const fuse::Result<void> mounted =
      registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_VERSION);
  ASSERT_FALSE(mounted.ok());
  EXPECT_EQ(mounted.error().code, "E-ABI-VERSION");
}

TEST(ConformanceTest, PluginDescribeDisagreeingWithTheGraphIsRefused) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1},
                                                                    /*cells=*/1);
  const fuse_participant_abi_v1* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream",
                                               std::make_unique<DownstreamSink>()));

  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry
                  .mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE)
                  .ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "upstream", std::move(extracted).value()));

  const fuse_status rc = t->initialize(ctx, nullptr, nullptr);
  EXPECT_EQ(rc, FUSE_ERR_RUNTIME);
  fuse_error err{sizeof(err)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  // impl_initialize() wraps register_fields() errors as E-ABI-INIT-VALIDATE
  // only for graph validation; a component register_fields() failure
  // propagates its OWN code (E-ABI-DESCRIBE) through FUSE_ERR_RUNTIME's
  // last_error -- confirmed by the code field, not just the status.
  EXPECT_STREQ(err.code, "E-ABI-DESCRIBE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(ConformanceTest, PluginAdvanceErrorSurfacesAsComponentFailure) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1},
                                                                    /*cells=*/1);
  const fuse_participant_abi_v1* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream",
                                               std::make_unique<DownstreamSink>()));
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(
      registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ADVANCE).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "upstream", std::move(extracted).value()));

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);  // describe() is fine; only advance() misbehaves
  fuse_run_request rr{sizeof(rr)};
  rr.periods = 1;
  const fuse_status rc = t->run(ctx, &rr, nullptr);
  EXPECT_EQ(rc, FUSE_ERR_RUNTIME);
  fuse_error err{sizeof(err)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, "E-ABI-PROVIDER-ADVANCE");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// Review 2026-10-11 AB-3: a plugin whose advance() writes the pre-growth
// fuse_step_report struct_size (64 bytes, the layout that shipped before
// next_component/step_opened/period/delivery_pending/delivery_field were
// appended -- c-abi.md's own growth table) must still be ACCEPTED: this
// host only ever reads t_begin/t_end off a plugin's own report, both
// inside that original 64 bytes. Before this fix, provider_adapter.cpp's
// struct_size_ok<fuse_step_report> demanded >= sizeof(current) = 104 and
// refused this exact, backward-compatible plugin with E-ABI-VERSION.
TEST(ConformanceTest, OldPluginStepReportStructSizeIsAccepted) {
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1},
                                                                    /*cells=*/1);
  const fuse_participant_abi_v1* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream",
                                               std::make_unique<DownstreamSink>()));
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(
      registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_OLD_STEP_REPORT).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "upstream", std::move(extracted).value()));

  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_run_request rr{sizeof(rr)};
  rr.periods = 1;
  fuse_status rc = t->run(ctx, &rr, nullptr);
  if (rc != FUSE_OK) {
    fuse_error err{sizeof(err)};
    char buf[512] = {0};
    err.buffer = buf;
    err.capacity = sizeof(buf);
    t->last_error(ctx, nullptr, &err);
    ADD_FAILURE() << "fuse_run unexpectedly refused the old-struct-size plugin ("
                  << (err.code ? err.code : "?") << "): " << buf;
  }
  EXPECT_EQ(rc, FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// Review 2026-10-11 TD-6: a one-rank "mount a misbehaving plugin as
// upstream, then drive it through fuse_initialize()/fuse_run()" harness,
// parametrised over which single plugin entry misbehaves, firing each of
// the provider-adapter refusal codes TD-6 found untested by anything in
// the tree. Mirrors PluginAdvanceErrorSurfacesAsComponentFailure and
// PluginDescribeDisagreeingWithTheGraphIsRefused above; only the plugin
// path and the expected code/call differ per row.
struct ProviderAdapterRefusalCase {
  std::string test_name;
  const char* plugin_path;
  const char* expected_code;
  bool        fails_at_initialize;  // false => fails at run() instead
};

class ProviderAdapterRefusalTest
    : public testing::TestWithParam<ProviderAdapterRefusalCase> {};

TEST_P(ProviderAdapterRefusalTest, FiresTheExpectedCode) {
  const auto& p = GetParam();
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes(fuse::Rational{1},
                                                                    /*cells=*/1);
  const fuse_participant_abi_v1* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream",
                                               std::make_unique<DownstreamSink>()));
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", p.plugin_path).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "upstream", std::move(extracted).value()));

  fuse_error err{sizeof(err)};
  if (p.fails_at_initialize) {
    EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_ERR_RUNTIME);
  } else {
    ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
    fuse_run_request rr{sizeof(rr)};
    rr.periods = 1;
    EXPECT_EQ(t->run(ctx, &rr, nullptr), FUSE_ERR_RUNTIME);
  }
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, p.expected_code);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

INSTANTIATE_TEST_SUITE_P(
    TD6AndAB6, ProviderAdapterRefusalTest,
    testing::Values(
        ProviderAdapterRefusalCase{"CreateFails", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CREATE,
                                   "E-ABI-PROVIDER-CREATE", true},
        ProviderAdapterRefusalCase{"BindFails", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_BIND,
                                   "E-ABI-PROVIDER-BIND", true},
        ProviderAdapterRefusalCase{"NonF64Element", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_ELEMENT,
                                   "E-ABI-PROVIDER-ELEMENT", true},
        ProviderAdapterRefusalCase{"CapabilitiesFails",
                                   FUSE_TEST_C_PROVIDER_PLUGIN_BAD_CAPABILITIES,
                                   "E-ABI-PROVIDER-CAPABILITIES", true},
        ProviderAdapterRefusalCase{"DescribeSuperset",
                                   FUSE_TEST_C_PROVIDER_PLUGIN_BAD_DESCRIBE_EXTRA,
                                   "E-ABI-DESCRIBE", true},
        ProviderAdapterRefusalCase{"WindowEchoWrong", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_WINDOW,
                                   "E-ABI-PROVIDER-WINDOW", false}),
    [](const testing::TestParamInfo<ProviderAdapterRefusalCase>& info) {
      return info.param.test_name;
    });

// The five provider-adapter refusal codes above that register_fields()
// and advance() CANNOT reach through fuse_initialize()/fuse_run() --
// save_state()/restore_state() are only called mid-run from a checkpoint
// this fixture never takes, no-graph and no-table are mis-set-up states
// the participant ABI itself never produces -- are driven directly
// through the same fuse::Component interface fuse::Runner uses, exactly
// the test-only seam test/abi/test_support.hpp already opens for fuse_ctx.
TEST(ConformanceTest, PluginSaveStateErrorIsRefused) {
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_SAVE).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();

  fuse::FieldRegistry freg;
  const fuse::CouplingGraph graph =
      fuse::abi_test::build_two_component_lag1_graph(fuse::Rational{1}, /*cells=*/1);
  ASSERT_TRUE(comp->register_fields(freg, graph).ok());

  const fuse::Result<void> saved = comp->save_state();
  ASSERT_FALSE(saved.ok());
  EXPECT_EQ(saved.error().code, "E-ABI-PROVIDER-SAVE");
}

TEST(ConformanceTest, PluginRestoreStateErrorIsRefused) {
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN_BAD_RESTORE).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();

  fuse::FieldRegistry freg;
  const fuse::CouplingGraph graph =
      fuse::abi_test::build_two_component_lag1_graph(fuse::Rational{1}, /*cells=*/1);
  ASSERT_TRUE(comp->register_fields(freg, graph).ok());

  const fuse::Result<void> restored = comp->restore_state();
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-ABI-PROVIDER-RESTORE");
}

// docs/dev/restart.md round 2, item 4: "a C consumer test of the new
// entries at tier 0." Exercises fuse_provider_abi_v1::save_restart_blob/
// load_restart_blob end to end through the REAL stack this project's
// own C ABI defines -- a dlopen()'d C plugin (test/abi/plugins/
// linear_provider.c), ProviderAdapter's query-then-fill CALLER side
// (src/abi/provider_adapter.cpp), and fuse::Component's own interface
// (fuse/runtime/component.hpp) -- never calling the generated struct
// layout directly, which would only prove the header compiles.
TEST(ConformanceTest, ProviderSaveAndLoadRestartBlobRoundTrips) {
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();

  fuse::FieldRegistry freg;
  const fuse::CouplingGraph graph =
      fuse::abi_test::build_two_component_lag1_graph(fuse::Rational{1}, /*cells=*/1);
  ASSERT_TRUE(comp->register_fields(freg, graph).ok());

  // The plugin set FUSE_CAP_RESTART_FORM_B, so the capability bit this
  // C ABI entry is gated on must have made it all the way through.
  EXPECT_TRUE(comp->supports_restart_form_b());

  const fuse::Result<std::vector<std::byte>> saved = comp->save_restart_blob();
  ASSERT_TRUE(saved.ok()) << saved.error().code << ": " << saved.error().message;
  // The plugin's own fixed byte pattern (g_restart_blob, "LIN-RB-1"),
  // never a default-constructed empty blob.
  const std::vector<std::byte> expected = {
      std::byte{0x4c}, std::byte{0x49}, std::byte{0x4e}, std::byte{0x2d},
      std::byte{0x52}, std::byte{0x42}, std::byte{0x2d}, std::byte{0x31}};
  EXPECT_EQ(saved.value(), expected);
  EXPECT_EQ(comp->restart_blob_version(), 7u);

  // Round trip: load the SAME bytes back. The plugin's own
  // p_load_restart_blob() checks them against its own fixed pattern and
  // the version, so this proves the bytes and the version both crossed
  // the ABI boundary intact, not just that SOME call returned FUSE_OK.
  const fuse::Result<void> loaded =
      comp->load_restart_blob(saved.value(), comp->restart_blob_version());
  ASSERT_TRUE(loaded.ok()) << loaded.error().code << ": " << loaded.error().message;
}

TEST(ConformanceTest, ProviderAdapterRefusesWhenComponentNotInGraph) {
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();

  fuse::FieldRegistry freg;
  const fuse::CouplingGraph empty_graph;  // declares no "upstream" component at all
  const fuse::Result<void> rf = comp->register_fields(freg, empty_graph);
  ASSERT_FALSE(rf.ok());
  EXPECT_EQ(rf.error().code, "E-ABI-PROVIDER-GRAPH");
}

TEST(ConformanceTest, ProviderAdapterRefusesATableMissingARequiredEntry) {
  // No dlopen() needed: ProviderAdapter's own register_fields() refuses
  // before ever reading through table_ when it is null (the state a
  // plugin whose dlopen/version-negotiation failed earlier leaves it in).
  fuse::abi_detail::ProviderAdapter adapter("upstream", nullptr);
  fuse::FieldRegistry freg;
  const fuse::CouplingGraph graph =
      fuse::abi_test::build_two_component_lag1_graph(fuse::Rational{1}, /*cells=*/1);
  const fuse::Result<void> rf = adapter.register_fields(freg, graph);
  ASSERT_FALSE(rf.ok());
  EXPECT_EQ(rf.error().code, "E-ABI-PROVIDER-TABLE");
}

TEST(ConformanceTest, ProviderAdapterRegistersAnEmptyViewWhenThisRankDoesNotHostIt) {
  // KNOWN_ISSUES.md D15, round 3 (host measurement): a Plugin component
  // placed on a rank SUBSET registered its own field's FULL region size
  // (from this adapter's own private, rank-unaware fuse::SerialMeshBackend
  // -- this class's own header comment) on a NON-hosting rank, where the
  // real plan (built from the rank-aware fuse::native::NativeMeshBackend)
  // correctly expects an EMPTY view there -- "field 'upstream.out' is
  // registered with 1x1 values but the plan sizes its region at 0x1".
  // D24 (KNOWN_ISSUES.md, closed 2026-10-09): "a component that does not
  // live on this rank gets an empty view of the right dimension ...
  // because every rank resolves every placement ... while only hosting
  // ranks read one." This pins the fix: set_hosting_rank() to a rank
  // OUTSIDE the component's own declared RankSet ({0,1} here --
  // build_two_component_lag1_graph()'s own "upstream" placement,
  // test/abi/graph_fixture.hpp), and register_fields() must bind a
  // ZERO-count view instead of asking backend_ for the region's full
  // size at all -- never a refusal, matching every other non-hosting
  // rank's own silent "nothing to do here" posture in this codebase.
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();
  auto* adapter = dynamic_cast<fuse::abi_detail::ProviderAdapter*>(comp.get());
  ASSERT_NE(adapter, nullptr);
  adapter->set_hosting_rank(1);  // "upstream"'s own RankSet is {0, 1} -- rank 0 only

  fuse::FieldRegistry freg;
  const fuse::CouplingGraph graph =
      fuse::abi_test::build_two_component_lag1_graph(fuse::Rational{1}, /*cells=*/1);
  const fuse::Result<void> rf = comp->register_fields(freg, graph);
  ASSERT_TRUE(rf.ok()) << (rf.ok() ? "" : rf.error().message);

  const std::optional<fuse::FieldId> fid =
      graph.find_field(graph.find_component("upstream").value(), "out");
  ASSERT_TRUE(fid.has_value());
  const fuse::Result<fuse::FieldRef> bound = freg.get(*fid);
  ASSERT_TRUE(bound.ok()) << (bound.ok() ? "" : bound.error().message);
  EXPECT_EQ(bound.value().size(), 0u)
      << "a non-hosting rank must register an EMPTY view, not the region's full size";
}

TEST(ConformanceTest, ProviderAdapterSplitsAMultiRankComponentsRegionByPosition) {
  // KNOWN_ISSUES.md D15, round 5 (host measurement): round 3's fix above
  // (ProviderAdapterRegistersAnEmptyViewWhenThisRankDoesNotHostIt) only
  // covers RankSet MEMBERSHIP -- "hosted or not", a binary question that
  // happens to be the whole answer for a ONE-rank component (every
  // fixture before this round). provider_subset_mpi_test.cpp's own
  // 3-rank graph places "upstream" on RankSet{1, 2} (TWO ranks): the
  // real plan decomposes its one-cell region over THOSE two ranks by
  // POSITION (D24's own rule, applied by NativeMeshBackend to every
  // placement), giving rank 1 (position 0 of 2) the one cell and rank 2
  // (position 1 of 2) none -- but round 3's rule answers `hosted = true`
  // for BOTH (both are RankSet members) and registered the FULL region
  // size on rank 2 too, measured on the host as "field 'upstream.out' is
  // registered with 1x1 values but the plan sizes its region at 0x1".
  // Fixed by applying the SAME per-position split the plan's own backend
  // already applies (fuse::abi_detail::even_rank_share(), include/fuse/
  // abi/rank_share.hpp) whenever `hosting_rank_` is set and the
  // component's own RankSet has more than one member.
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();
  auto* adapter = dynamic_cast<fuse::abi_detail::ProviderAdapter*>(comp.get());
  ASSERT_NE(adapter, nullptr);

  fuse::CouplingGraph graph;
  const auto mesh = graph.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto region = graph.add_region(fuse::Region{.name = "r", .mesh = mesh, .dimension = 1});
  const auto up = graph.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{1, 2}, .interval = fuse::Rational{1}});
  (void)graph.add_field(up, fuse::FieldSpec{.name = "out", .region = region, .components = 1,
                                            .units = fuse::units::kKelvin,
                                            .conservation = fuse::Conservation::None,
                                            .intent = fuse::Intent::Provides});
  const std::optional<fuse::FieldId> fid =
      graph.find_field(graph.find_component("upstream").value(), "out");
  ASSERT_TRUE(fid.has_value());

  fuse::FieldRegistry freg_rank1;
  adapter->set_hosting_rank(1);  // position 0 of 2 -- owns the one cell
  ASSERT_TRUE(comp->register_fields(freg_rank1, graph).ok());
  const fuse::Result<fuse::FieldRef> bound1 = freg_rank1.get(*fid);
  ASSERT_TRUE(bound1.ok()) << (bound1.ok() ? "" : bound1.error().message);
  EXPECT_EQ(bound1.value().size(), 1u) << "rank 1 (position 0 of 2) owns the one cell";

  fuse::FieldRegistry freg_rank2;
  adapter->set_hosting_rank(2);  // position 1 of 2 -- a RankSet MEMBER, owns NONE of it
  ASSERT_TRUE(comp->register_fields(freg_rank2, graph).ok());
  const fuse::Result<fuse::FieldRef> bound2 = freg_rank2.get(*fid);
  ASSERT_TRUE(bound2.ok()) << (bound2.ok() ? "" : bound2.error().message);
  EXPECT_EQ(bound2.value().size(), 0u)
      << "rank 2 is a RankSet member (round 3's binary rule would have registered the full "
         "size here) but owns none of the one-cell region at this position -- the bug "
         "round 5 fixes";
}

TEST(ConformanceTest, ProviderAdapterRegisterFieldsExceptionBecomesAFailure) {
  // register_fields()'s own try/catch (E-ABI-PROVIDER-EXCEPTION) guards
  // against anything FUSE's own backend/graph calls can throw, not just
  // the plugin; a field whose region the graph never declared (a
  // malformed descriptor this test builds directly, bypassing
  // fuse::validate(), the same way a corrupted wire descriptor would
  // reach this code if validation were ever skipped) makes
  // graph.region(spec.region) throw std::out_of_range.
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_dynamic("upstream", FUSE_TEST_C_PROVIDER_PLUGIN).ok());
  fuse::Result<std::unique_ptr<fuse::Component>> extracted = registry.extract("upstream");
  ASSERT_TRUE(extracted.ok());
  std::unique_ptr<fuse::Component> comp = std::move(extracted).value();

  fuse::CouplingGraph graph;
  const auto up = graph.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  graph.add_field(up, fuse::FieldSpec{.name = "out",
                                      .region = fuse::RegionId{0},  // no region 0 exists
                                      .components = 1,
                                      .units = fuse::units::kKelvin,
                                      .conservation = fuse::Conservation::None,
                                      .intent = fuse::Intent::Provides});

  fuse::FieldRegistry freg;
  const fuse::Result<void> rf = comp->register_fields(freg, graph);
  ASSERT_FALSE(rf.ok());
  EXPECT_EQ(rf.error().code, "E-ABI-PROVIDER-EXCEPTION");
}

// ---- struct_size short/long on the provider table itself (no dlopen; a
// hand-built table, the same seam test/abi/struct_size_test.cpp uses for
// the participant table). ---------------------------------------------------

TEST(ConformanceTest, ProviderTableStructSizeShortIsRefused) {
  fuse_provider_abi_v1 bad{};
  bad.struct_size = sizeof(fuse_provider_abi_v1) - 1;
  bad.abi_version = FUSE_ABI_VERSION;
  fuse::abi_detail::Registry registry;
  const fuse::Result<void> mounted = registry.mount_table("upstream", &bad);
  ASSERT_FALSE(mounted.ok());
  EXPECT_EQ(mounted.error().code, "E-ABI-VERSION");
}

TEST(ConformanceTest, ProviderTableStructSizeLongFromANewerCallerIsAccepted) {
  fuse_provider_abi_v1 newer{};
  newer.struct_size = sizeof(fuse_provider_abi_v1) + 64;  // as if from a v1.1 header
  newer.abi_version = FUSE_ABI_VERSION;
  fuse::abi_detail::Registry registry;
  const fuse::Result<void> mounted = registry.mount_table("upstream", &newer);
  EXPECT_TRUE(mounted.ok()) << (mounted.ok() ? "" : mounted.error().message);
}

TEST(ConformanceTest, ProviderTableVersionMismatchIsRefused) {
  fuse_provider_abi_v1 wrong_version{};
  wrong_version.struct_size = sizeof(fuse_provider_abi_v1);
  wrong_version.abi_version = FUSE_ABI_VERSION + 1;  // "v2"
  fuse::abi_detail::Registry registry;
  const fuse::Result<void> mounted = registry.mount_table("upstream", &wrong_version);
  ASSERT_FALSE(mounted.ok());
  EXPECT_EQ(mounted.error().code, "E-ABI-VERSION");
}

TEST(ConformanceTest, RegistryRefusesADuplicateName) {
  fuse_provider_abi_v1 table{};
  table.struct_size = sizeof(fuse_provider_abi_v1);
  table.abi_version = FUSE_ABI_VERSION;
  fuse::abi_detail::Registry registry;
  ASSERT_TRUE(registry.mount_table("upstream", &table).ok());
  const fuse::Result<void> second = registry.mount_table("upstream", &table);
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.error().code, "E-ABI-REGISTRY-DUPLICATE");
}

// ---- the `plugin` LAUNCH KIND: a provider named IN THE GRAPH, mounted
// automatically at fuse_initialize() (src/abi/participant.cpp), never
// test-injected -- the end-to-end path test_support.hpp's seam otherwise
// bypasses entirely. ----------------------------------------------------

TEST(ConformanceTest, GraphDeclaredPluginIsAutoMountedAndRun) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:1"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:1"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = FUSE_TEST_C_PROVIDER_PLUGIN}});
  const auto down = g.add_component(fuse::ComponentDecl{
      .name = "downstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto out_f = g.add_field(
      up, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                          .units = fuse::units::kKelvin,
                          .conservation = fuse::Conservation::None,
                          .intent = fuse::Intent::Provides});
  const auto in_f = g.add_field(
      down, fuse::FieldSpec{.name = "in", .region = r2, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", out_f, in_f);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // "downstream" is the only component this test injects; "upstream" is
  // named `plugin` IN THE GRAPH -- nothing here tells the ABI about it.
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream",
                                               std::make_unique<DownstreamSink>()));

  const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);
  if (init_rc != FUSE_OK) {
    fuse_error err{sizeof(err)};
    (void)t->last_error(ctx, nullptr, &err);
    FAIL() << "fuse_initialize: " << (err.code != nullptr ? err.code : "?");
  }

  fuse_run_request rr{sizeof(rr)};
  rr.periods = 3;
  ASSERT_EQ(t->run(ctx, &rr, nullptr), FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(ConformanceTest, GraphDeclaredPluginWithNoFileIsRefusedAtInitialize) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:1"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = "/nonexistent/path/to/a/plugin.so"}});
  (void)g.add_field(up, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                                        .units = fuse::units::kKelvin,
                                        .intent = fuse::Intent::Provides});
  const auto bytes = fuse::serialize(g);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  const fuse_status rc = t->initialize(ctx, nullptr, nullptr);
  EXPECT_EQ(rc, FUSE_ERR_RUNTIME);
  fuse_error err{sizeof(err)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, "E-ABI-LOAD");
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(ConformanceTest, DlopenFailureIsRefusedWithDlerrorText) {
  fuse::abi_detail::Registry registry;
  const fuse::Result<void> mounted =
      registry.mount_dynamic("upstream", "/nonexistent/path/to/a/plugin.so");
  ASSERT_FALSE(mounted.ok());
  EXPECT_EQ(mounted.error().code, "E-ABI-LOAD");
  EXPECT_FALSE(mounted.error().message.empty());
}
