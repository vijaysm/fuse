// A plugin on a rank subset: the `LaunchMode::Plugin` component is placed
// on a STRICT SUBSET of a real multi-rank base communicator and this file
// confirms both halves: the hosting rank(s) actually call through the
// plugin, and the NON-hosting rank runs `fuse_run()` to completion without
// ever touching it. This is pure MPI and needs no device support. Its graph
// and driving sequence mirror test/abi/conformance_test.cpp's own
// GraphDeclaredPluginIsAutoMountedAndRun, which runs single-process.
//
// Compiled under FUSE_ENABLE_MPI only and needs a working MPI launcher.
// Registry::mount_dynamic() dlopen()s the plugin only on a hosting rank:
// the `dlopen(..., RTLD_NOLOAD)` check below pins that, for rank 0, never a
// member of "upstream"'s own RankSet in this fixture; the tier-0 twin with
// no MPI is test/abi/conformance_test.cpp's own
// ConformanceTest.NonHostingRankNeverDlopensThePluginOrCallsAnyOfItsEntries.
//
// A plugin-side refusal must be visible to every rank of the base
// communicator, not only a hosting one, per the project's standing "agree
// first, then every rank refuses with the same code" rule.
// AHostingRanksPluginMountFailureIsAgreedByEveryRankNotOnlyTheHostingOnes
// below places a deliberately BROKEN plugin (the BAD_VERSION variant)
// on every hosting rank and confirms EVERY rank -- hosting or not --
// agrees on the SAME failure, through one MPI_Allreduce(MPI_MINLOC)
// over the base communicator (src/abi/participant.cpp's reconcile_
// plugin_agreement()); the tier-0 twin with no MPI is test/abi/
// conformance_test.cpp's own NonHostingRankAdoptsAHostingPeersPlugin
// MountFailureThroughTheBaseCommunicatorReduction (faked peers) and
// HostingRanksOwnPluginMountFailureStillReachesTheBaseCommunicatorRed
// uction (a real local failure, white-box call-count check).
#include "fuse/base/config.hpp"

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/units.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/runtime/component.hpp"

#include "test_support.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// test/abi/conformance_test.cpp's own DownstreamSink, copied rather than
/// shared (that file's class is local to its own translation unit, not
/// exposed through any header): a native component that simply records
/// every value it receives, so a hosting rank's own effect through the
/// plugin is independently observable without reaching into the
/// plugin's own C source.
///
/// A component that does not live on this rank gets an empty view.
/// "downstream" is placed at RankSet {0, 1} (rank 0 only); on a 2-rank
/// world, rank 1 is NOT in it, so registering count=1 there would disagree
/// with the real plan's own 0-entry expectation (an E-RT-SETUP-PEER
/// failure). `hosted` (constructor argument, the
/// caller's own `graph.component(...).ranks.contains(rank)` check,
/// mirroring ProviderAdapter::set_hosting_rank()'s own convention)
/// decides whether this rank registers the real size or an empty view.
class DownstreamSink final : public fuse::Component {
 public:
  explicit DownstreamSink(bool hosted) : hosted_(hosted) {}

  [[nodiscard]] std::string name() const override { return "downstream"; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    const auto cid = graph.find_component("downstream");
    if (!cid.has_value()) return fuse::Failure{"E-TEST", "no 'downstream' component"};
    const std::size_t count = hosted_ ? 1 : 0;
    storage_.assign(count, -1.0);
    for (const fuse::FieldId f : graph.component(*cid).fields) {
      auto r = registry.bind(f, fuse::FieldRef{storage_.data(),
                                               fuse::Layout{.stride = 1, .count = storage_.size()},
                                               fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    history.push_back(storage_.empty() ? -1.0 : storage_.front());
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return 0; }

  std::vector<double> history;

 private:
  bool                 hosted_;
  std::vector<double> storage_;
};

}  // namespace

TEST(ProviderSubsetMpi, APluginOnARankSubsetRunsOnlyOnItsOwnHostingRanksWithNoHang) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_GE(world_size, 2) << "this row needs at least 2 ranks (RANKS 2 3)";

  // "upstream" (the plugin) is placed on every rank EXCEPT rank 0 --
  // {1} of a 2-rank world, {1,2} of a 3-rank world.
  // "downstream" (native, injected below) stays on rank 0 alone.
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:1"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:1"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{1, world_size - 1}, .interval = fuse::Rational{1},
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

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  // Real base communicator (every rank, identical input) -- what makes
  // this an actual multi-rank placement rather than N independent
  // single-process runs.
  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
  // Open MPI's Fortran handle for MPI_COMM_WORLD is 0, which the ABI reads as
  // "none" unless has_communicator says the handle is deliberate.
  opts.has_communicator = 1;
  ASSERT_EQ(t->setup(ctx, &opts, nullptr), FUSE_OK);

  // "downstream" is placed at RankSet{0, 1} -- rank 0 only (see `down`
  // above); every other rank must register an EMPTY view.
  auto sink = std::make_unique<DownstreamSink>(/*hosted=*/rank == 0);
  DownstreamSink* sink_ptr = sink.get();
  ASSERT_TRUE(fuse::abi_test::inject_component(ctx, "downstream", std::move(sink)));

  const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);
  if (init_rc != FUSE_OK) {
    fuse_error err{.struct_size = sizeof(err)};
    char buf[512] = {0};
    err.buffer = buf;
    err.capacity = sizeof(buf);
    t->last_error(ctx, nullptr, &err);
    // test/support/fuse_test_main.cpp's own main() deletes every NON-ZERO
    // rank's own GoogleTest result printer (so an MPI job's own log is not
    // N copies of the same test names) -- which ALSO silently swallows a
    // FAIL() message's own text on any rank but 0. The one rank whose own
    // fuse_initialize() genuinely failed is exactly the rank most likely
    // to be non-zero here ("upstream" is hosted on ranks 1..world_size-1,
    // never rank 0), and E-RT-SETUP-PEER's own message, printed on rank 0,
    // says only "see that rank's own error". Printed to stderr directly, bypassing
    // GoogleTest's own printer
    // entirely, so it reaches this rank's own captured log regardless of
    // which rank is 0.
    std::fprintf(stderr, "rank %d: fuse_initialize failed: %s\n", rank, buf);
    std::fflush(stderr);
    FAIL() << "rank " << rank << ": fuse_initialize failed: " << buf;
  }

  // "upstream"'s own RankSet is {1, world_size - 1} --
  // rank 0 is NEVER a member of it, on either a 2- or a 3-rank world,
  // so rank 0 is ALWAYS the non-hosting case for this fixture. Direct,
  // OS-level evidence that this rank's own ProviderAdapter never
  // dlopen()'d "upstream"'s shared object at all -- the tier-0
  // conformance test (test/abi/conformance_test.cpp's own
  // NonHostingRankNeverDlopensThePluginOrCallsAnyOfItsEntries) pins the
  // identical check with no MPI; this is its real-multi-process twin.
  if (rank == 0) {
    void* resident = dlopen(FUSE_TEST_C_PROVIDER_PLUGIN, RTLD_NOLOAD | RTLD_LAZY);
    EXPECT_EQ(resident, nullptr)
        << "rank 0 is never a member of \"upstream\"'s own RankSet and must "
           "never dlopen() its shared object";
  }

  // Every rank -- hosting the plugin or not -- calls fuse_run() with the
  // SAME request: the NON-hosting rank hosts nothing at all (neither
  // "upstream" nor "downstream", for every rank >= 1 beyond the plugin's
  // own RankSet on a 3-rank world there is none here, but rank 0 hosts
  // only "downstream", never "upstream") and must still execute its own
  // share of the period's collectives, per entries.def's own doc on
  // `run` (and the reconciliation, if this rank's call ever refused
  // locally while others did not).
  fuse_run_request rr{.struct_size = sizeof(rr)};
  rr.periods = 3;
  const fuse_status run_rc = t->run(ctx, &rr, nullptr);

  std::vector<std::int32_t> all_rc(static_cast<std::size_t>(world_size));
  const std::int32_t mine = run_rc;
  MPI_Allgather(&mine, 1, MPI_INT32_T, all_rc.data(), 1, MPI_INT32_T, MPI_COMM_WORLD);
  for (int r = 0; r < world_size; ++r) {
    EXPECT_EQ(all_rc[static_cast<std::size_t>(r)], FUSE_OK)
        << "rank " << r << " did not return FUSE_OK from fuse_run() "
           "(rank " << rank << " observing)";
  }

  if (rank == 0) {
    // Only rank 0 hosts "downstream"; its own history is the
    // independent evidence that values actually crossed from the
    // plugin, hosted entirely on OTHER ranks, to here.
    EXPECT_EQ(sink_ptr->history.size(), 3u);
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}


// The real multi-process twin of test/abi/
// conformance_test.cpp's own NonHostingRankAdoptsAHostingPeersPluginMount
// FailureThroughTheBaseCommunicatorReduction and HostingRanksOwnPlugin
// MountFailureStillReachesTheBaseCommunicatorReduction -- no fake peers
// here, a REAL MPI_Allreduce(MPI_MINLOC) over MPI_COMM_WORLD. "upstream"
// (the plugin) is placed on every rank EXCEPT rank 0, exactly as the
// happy-path row above, but its own plugin_path is the already-built
// BAD_VERSION variant (test/CMakeLists.txt), so EVERY hosting rank's own
// mount_dynamic() fails for real.
TEST(ProviderSubsetMpi,
    AHostingRanksPluginMountFailureIsAgreedByEveryRankNotOnlyTheHostingOnes) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  ASSERT_GE(world_size, 2) << "this row needs at least 2 ranks (RANKS 2 3)";

  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:1"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{1, world_size - 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = FUSE_TEST_C_PROVIDER_PLUGIN_BAD_VERSION}});
  const auto out_f = g.add_field(
      up, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                          .units = fuse::units::kKelvin,
                          .conservation = fuse::Conservation::None,
                          .intent = fuse::Intent::Provides});
  (void)out_f;
  const auto bytes = fuse::serialize(g);

  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
  // Open MPI's Fortran handle for MPI_COMM_WORLD is 0, which the ABI reads as
  // "none" unless has_communicator says the handle is deliberate.
  opts.has_communicator = 1;
  ASSERT_EQ(t->setup(ctx, &opts, nullptr), FUSE_OK);

  const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);

  // THE assertion this row exists for: rank 0 (never a member of
  // "upstream"'s own RankSet, so its own local plugin_mount_status is
  // trivially FUSE_OK) must NOT return FUSE_OK -- it must agree with
  // every hosting rank's own real E-ABI-VERSION failure, through the
  // base-communicator MPI_Allreduce(MPI_MINLOC) reconcile_plugin_
  // agreement() runs, never through collective_comm_f (which stays 0
  // for this plugin-only, my_component-unset graph).
  std::vector<std::int32_t> all_rc(static_cast<std::size_t>(world_size));
  const std::int32_t mine = init_rc;
  MPI_Allgather(&mine, 1, MPI_INT32_T, all_rc.data(), 1, MPI_INT32_T, MPI_COMM_WORLD);
  for (int r = 0; r < world_size; ++r) {
    EXPECT_EQ(all_rc[static_cast<std::size_t>(r)], FUSE_ERR_RUNTIME)
        << "rank " << r << " did not agree on FUSE_ERR_RUNTIME for the plugin-mount failure "
           "(rank " << rank << " observing)";
  }

  // Rank 0 is the non-originating rank here (its own local status was
  // FUSE_OK, not the agreed minimum), so its own message is rewritten --
  // E-ABI-PLUGIN-AGREEMENT-REFUSED, naming a hosting rank, not the raw
  // E-ABI-VERSION a hosting rank's own last_error() carries (checked only
  // on rank 0, where the distinction is unambiguous: every OTHER rank
  // here hosts "upstream" and is itself an origin candidate).
  if (rank == 0) {
    fuse_error err{.struct_size = sizeof(err)};
    char buf[512] = {0};
    err.buffer = buf;
    err.capacity = sizeof(buf);
    ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
    EXPECT_STREQ(err.code, "E-ABI-PLUGIN-AGREEMENT-REFUSED");
    // The peer's message carries the ORIGIN's own code and
    // text, not only the rank: the hosting rank refused with E-ABI-VERSION.
    const std::string message(buf);  // last_error() writes the message into the caller's buffer
    EXPECT_NE(message.find("E-ABI-VERSION"), std::string::npos) << message;
    EXPECT_NE(message.find("refusing rank"), std::string::npos) << message;
  }

  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
