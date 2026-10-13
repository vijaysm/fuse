// KNOWN_ISSUES.md D57 ("Collectives" track). Root cause, established by
// reading src/abi/participant.cpp's own Runner::Impl::resume_period()
// (src/runtime/runner.cpp) and its two callers: for a window-coupled,
// ExternalDelivery-bound coupling region driven by fuse_begin_step()/
// fuse_end_step() with NO end_time declared (fuse_setup_options::
// has_end_time left at its zero default -- examples/brusselator/
// participants/u_main.c's own `fuse_setup_options setup; memset(&setup,
// 0, sizeof(setup)); ...; t->setup(ctx, &setup, NULL);`, mirrored by
// v_main.f90), `prime_next_request()`'s own rollover (the branch that
// fires when its own first `resume_period()` call finds the current
// period already drained) has NOTHING bounding how many further periods
// it may silently start and run before answering fuse_begin_step()'s own
// caller -- it is gated ONLY by `c.has_end_time`, and with that false the
// gate never applies at all (`if (c.has_end_time && ...)`, both
// occurrences in the function, are no-ops). The SAME discovery, reached
// instead through `fuse_end_step()`'s own direct, unconditional
// `resume_period()` call, is reported immediately (next_component == -1,
// deferring the next period's own start to a LATER call) -- this entry's
// own fix makes `prime_next_request()`'s rollover behave identically
// instead of eagerly continuing.
//
// WHY THIS GRAPH SHAPE. examples/brusselator/participants/
// brusselator_graph_tool.cpp's own kDtU/kDtV/kWindow (genuinely
// multirate: 10 U-substeps, ~7 V-substeps per window) is reused here
// verbatim, via brusselator::add_window_coupling() -- the SAME builder
// the graph tool and every other Brusselator participants test use --
// with NO coupler component (coupler_ranks left at std::nullopt): both
// transfer fields' own `.hdf5_path` is set exactly as `--transport
// lowfive` does, so each channel compiles to OpKind::ExternalDelivery
// (src/ir/rules_channel.cpp), not Transport -- the shape
// test/verification/brusselator_lowfive_harness_test.cpp's own header
// comment documents as having a delivery proxy trailing every sweep,
// which is what makes fuse_end_step()'s own prefetch never discover a
// period's true end for this graph (that file's own comment, verified
// here too) and `prime_next_request()`'s rollover the ONLY place a
// period boundary is ever found.
//
// WHY NOT THE REAL THREE-RANK PARTICIPANTS GRAPH. test/verification/
// brusselator_external_delivery_size_test.cpp's own
// ReachesPastExternalDeliveryOnBothRanks already measured, and this file
// does not re-litigate, that `Runner::initialize()` on the REAL
// U{0,1}/coupler{1,1}/V{2,1} MPMD layout cannot get past
// `resolve_regions()`'s own `E-RT-REGION` ("an empty coupling value...")
// refusal at tier 0 at all: `RunnerConfig::force_this_process` carries no
// base_communicator, so `R.participants` stays empty on every rank
// (iteration.cpp, gated on `config.base_communicator.has_value()`), and a
// convergence measure naming the component not hosted here reports
// unmeasurable. The narrowest reachable shape that still exercises the
// SAME `resume_period()`/`prime_next_request()` machinery this entry's
// fix touches is the single-process, no-coupler graph below, driven
// through the REAL ABI (never a parallel re-implementation of
// prime_next_request()'s own logic) -- u_loop.c's own wrapper (`if
// (!out_begin_rep->step_opened) { zero *out_rep; return; }`) is copied
// here VERBATIM (StepComponent::drive(), below) rather than
// paraphrased, specifically so this test calls the exact same decision
// u_main.c's/v_main.f90's own loops make, not a model of it that could
// silently stop matching them.
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>

#include "fuse/abi/participant.h"
#include "fuse/backend/native_backend.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_graph.hpp"
#include "test_support.hpp"

namespace {

using fuse::ComponentKind;
using fuse::FieldRegistry;
using fuse::FieldRef;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::Layout;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::MemorySpace;
using fuse::Rational;
using fuse::RankSet;
using fuse::Result;
using fuse::WaveformIteration;

// Same multirate timing brusselator_graph_tool.cpp/brusselator_lowfive_
// harness_test.cpp both use (kDtU/kDtV do not divide one another, so the
// window genuinely needs more than one substep of each member) -- the
// grid is 1x1 (irrelevant to this defect, which is about how many TIMES
// the region's own instruction stream is walked, never about the data
// moved through it) so the whole test runs in well under a second.
constexpr int kNx = 1, kNy = 1;
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;

// window_hosting_test.cpp's/brusselator_external_delivery_size_test.cpp's
// own StubBlock, unchanged in spirit: a window member whose registered
// storage never varies, so the window's own residual is identically
// zero and the Jacobi sweep converges on sweep 0 every period -- the
// FEWEST possible sweeps, which is also the case most likely to let a
// single resume_period() call run clear through a sweep (or a whole
// period) in one go, exactly the condition prime_next_request()'s own
// header comment names as what exposes this defect.
class StubBlock final : public fuse::Component {
 public:
  explicit StubBlock(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      std::vector<double>& storage = is_out ? out_ : in_;
      storage.assign(static_cast<std::size_t>(kNx) * static_cast<std::size_t>(kNy), 0.0);
      auto r = reg.bind(f, FieldRef{storage.data(),
                                    Layout{.stride = 1, .count = storage.size()},
                                    MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
  std::vector<double> out_;
  std::vector<double> in_;
};

const fuse_participant_abi_v1* Table() {
  static const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

// u_loop.c's own u_loop_step(), copied verbatim in shape (the same
// begin_step()/physics-or-nothing/end_step() sequence, the SAME "no step
// opened means *out_rep stays zeroed, no end_step() call" rule) --
// generalised past "U" specifically since this test drives two stub
// components through ONE process (tier 0 has no MPI, so both of U's and
// V's own separate ranks collapse into one here; the real multi-rank
// case is exactly what ReachesPastExternalDeliveryOnBothRanks's own
// header comment already established tier 0 cannot reach for THIS
// graph's full three-rank shape). No physics call sits where u_loop.c's
// own bruss_u_step() does -- StubBlock::advance() is never invoked by
// this driving pattern at all, exactly as it never is for a real
// externally-driven component (the APPLICATION, not Component::advance(),
// is what moves a registered field's own contents between begin_step()
// and end_step()); StubBlock's fields simply never change, which is what
// makes the window converge on sweep 0 (see StubBlock's own comment).
fuse_status drive_one_step(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                           const char* component, fuse_step_report* out_rep,
                           fuse_step_report* out_begin_rep) {
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = component;
  std::memset(out_begin_rep, 0, sizeof(*out_begin_rep));
  out_begin_rep->struct_size = sizeof(*out_begin_rep);
  const fuse_status begin_rc = t->begin_step(ctx, &req, out_begin_rep);
  if (begin_rc != FUSE_OK) return begin_rc;
  if (!out_begin_rep->step_opened) {
    std::memset(out_rep, 0, sizeof(*out_rep));
    out_rep->struct_size = sizeof(*out_rep);
    return FUSE_OK;
  }
  std::memset(out_rep, 0, sizeof(*out_rep));
  out_rep->struct_size = sizeof(*out_rep);
  return t->end_step(ctx, &req, out_rep);
}

struct SetupResult {
  fuse_ctx* ctx = nullptr;
  std::int32_t u_idx = -1, v_idx = -1;
  std::unique_ptr<StubBlock> u, v;
};

// Builds the SAME graph shape brusselator_graph_tool.cpp's own
// --transport lowfive branch compiles (add_window_coupling(), then
// hdf5_path on both transfer fields), with NO coupler component -- the
// single-process shape brusselator_lowfive_harness_test.cpp's own
// build_lowfive_graph_and_weights() uses, minus the real species
// physics and remap weights this defect does not need. Returns a FRESH
// fuse_ctx with setup()/initialize()/register_field() already done;
// `opts.has_end_time` is left at ITS OWN zero default throughout --
// u_main.c's/v_main.f90's own choice, reproduced here deliberately, not
// an oversight this test works around.
SetupResult build_and_initialize(fuse::CouplingGraph& graph_storage) {
  SetupResult out;
  fuse::CouplingGraph& g = graph_storage;
  const fuse::MeshId mesh = g.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNx) + "x" + std::to_string(kNy)});
  const fuse::RegionId region = g.add_region({.name = "u_all", .mesh = mesh, .dimension = 2});

  // coupler_ranks = RankSet{0, 1} -- SAME single rank everything else
  // is on, not a separate placement: add_window_coupling()'s own code
  // only cares whether coupler_ranks.has_value() when deciding to add
  // `.remap(RemapFlags::Conservative).remap_on(coupler)` to BOTH
  // channels (brusselator_graph.hpp), which is what makes src/ir/
  // rules_channel.cpp emit a Remap (hence, per src/ir/rules_derived.cpp
  // 's own has_remap-gated condition, a Constrain) node at all -- with
  // no coupler, both transfer fields share one region and the channel
  // is a bare identity with NEITHER (brusselator_graph.hpp's own "WHY
  // BOTH TRANSFER FIELDS LIVE ON u_region" header comment), which is
  // what this entry's own first-draft version of this test measured
  // (fuse::RunStats::conservation_checks stayed 0 for every period
  // count, discriminating nothing). The real participants row DOES
  // declare a coupler (brusselator_graph_tool.cpp's own `--ranks-
  // coupler 1`), so this is the faithful choice, not a simplification
  // of convenience -- it is what makes a Constrain instruction (and,
  // on the real multi-rank graph, D57's own base-communicator
  // MPI_Allreduce) exist in this region's body at all.
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      g, region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1},
      RankSet{0, 1}, RankSet{0, 1});

  const auto u_out = g.find_field(handles.u_component, "u_out");
  const auto v_out = g.find_field(handles.v_component, "v_out");
  if (!u_out.has_value() || !v_out.has_value()) return out;
  const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "stub_u.h5/grid/u_out";
  const_cast<FieldSpec&>(g.field(*v_out)).hdf5_path = "stub_v.h5/grid/v_out";
  // Fieldless LowFive marker -- brusselator_graph_tool.cpp's own
  // write_lowfive_declarations()/brusselator_lowfive_harness_test.cpp's
  // own build_lowfive_graph_and_weights() precedent, ComponentKind::
  // Coupler so fuse_initialize()'s own wiring loop exempts it from
  // needing a registered implementation.
  (void)g.add_component({.name = "lowfive", .ranks = RankSet{0, 1},
                         .interval = Rational{1},
                         .kind = ComponentKind::Coupler,
                         .launch = LaunchSpec{.mode = LaunchMode::LowFive}});

  const auto* t = Table();
  if (t == nullptr) return out;
  const auto bytes = fuse::serialize(g);
  fuse_context_out co{.struct_size = sizeof(co)};
  if (t->create(nullptr, nullptr, &co) != FUSE_OK) return out;
  out.ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  if (t->load_graph(out.ctx, &src, nullptr) != FUSE_OK) return out;

  // The ONE line this entry's own header comment calls out: no
  // fuse_setup_options::has_end_time at all, matching u_main.c/
  // v_main.f90 exactly (both memset their own `setup` to zero and set
  // only communicator/my_component -- grep'd, not assumed, in this
  // round's own report).
  fuse_setup_options opts{.struct_size = sizeof(opts)};
  if (t->setup(out.ctx, &opts, nullptr) != FUSE_OK) return out;

  const int n = kNx * kNy;
  out.u = std::make_unique<StubBlock>("U");
  out.v = std::make_unique<StubBlock>("V");
  FieldRegistry registry;
  if (!out.u->register_fields(registry, g).ok()) return out;
  if (!out.v->register_fields(registry, g).ok()) return out;

  auto register_one = [&](const char* component, const char* field) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    if (t->field_handle(out.ctx, &fname, &fref) != FUSE_OK) return false;
    static std::vector<std::unique_ptr<std::vector<double>>> keep;
    keep.push_back(std::make_unique<std::vector<double>>(static_cast<std::size_t>(n), 0.0));
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = static_cast<uint64_t>(n);
    fview.owned_count = static_cast<uint64_t>(n);
    fview.stride = 1;
    fview.data = keep.back()->data();
    return t->register_field(out.ctx, &fview, nullptr) == FUSE_OK;
  };
  if (!register_one("U", "u_out")) return out;
  if (!register_one("U", "u_in")) return out;
  if (!register_one("V", "v_out")) return out;
  if (!register_one("V", "v_in")) return out;

  if (t->initialize(out.ctx, nullptr, nullptr) != FUSE_OK) {
    out.ctx = nullptr;
    return out;
  }

  // u_main.c/v_main.f90 both DO set a non-zero `communicator` (through
  // the real public ABI, which this single-process test cannot do at
  // tier 0: a non-zero communicator makes fuse_initialize() pick the
  // "mpi" transport, which has no registered backend on a no-MPI build
  // and fails initialize() outright, regardless of whether this graph's
  // own LowFive-bound channels ever need a real Transport exchange) --
  // the real row's own `has_end_time == false AND a base communicator IS
  // configured` combination, which is exactly what distinguishes it from
  // `AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent`'s
  // own single-process, no-communicator scenario (test/abi/
  // participant_test.cpp), where the pre-existing eager-continuation
  // behaviour this entry's fix must NOT disturb is pinned. Set here,
  // AFTER initialize() already succeeded with "copy" transport baked
  // into the live Runner (genuinely unaffected by this call) --
  // `prime_next_request()`'s own `c.communicator_set` read has no other
  // run-time consumer, so this reaches exactly the gate under test and
  // nothing upstream of it. See force_communicator_set()'s own header
  // comment (test_support.hpp) for why this ordering is both necessary
  // (setting it before initialize() fails outright at tier 0) and safe
  // (every MPI call site `config.base_communicator.has_value()` can
  // reach is compiled out under `#if FUSE_HAVE_MPI`, absent at tier 0).
  if (!fuse::abi_test::force_communicator_set(out.ctx, 1)) {
    out.ctx = nullptr;
    return out;
  }

  const auto u_cid = g.find_component("U");
  const auto v_cid = g.find_component("V");
  out.u_idx = static_cast<std::int32_t>(u_cid->value());
  out.v_idx = static_cast<std::int32_t>(v_cid->value());
  return out;
}

// The reference this entry's own claim is checked against: the IDENTICAL
// graph shape (same add_window_coupling() call, same hdf5_path fields,
// same StubBlock pair), run `periods` times through fuse::Runner::run()
// directly (src/runtime/runner.cpp's own `for (i = 0; i < periods; ++i)
// run_period();` loop) -- never through the ABI, never through
// prime_next_request() at all. This is the "how much work SHOULD running
// exactly `periods` periods take" oracle: fuse::RunStats::
// conservation_checks (one increment per OpKind::Constrain instruction
// `execute()` actually runs, src/runtime/runner.cpp) counts a quantity
// that is EXACTLY what KNOWN_ISSUES.md D57's own real host failure could
// not finish paying for a third, unrequested time -- the base-
// communicator MPI_Allreduce a Constrain instruction with `ci.reduce`
// set issues over EVERY rank of `config.base_communicator`, which this
// single-process reference has none of, so counting instructions rather
// than reproducing the hang itself is what makes this comparison run at
// tier 0 at all (this file's own header has the full account of why the
// real three-rank graph cannot reach even this much).
std::int64_t reference_conservation_checks(int periods) {
  fuse::CouplingGraph g;
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
  const fuse::MeshId mesh = g.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNx) + "x" + std::to_string(kNy)});
  const fuse::RegionId region = g.add_region({.name = "u_all", .mesh = mesh, .dimension = 2});
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      g, region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1},
      RankSet{0, 1}, RankSet{0, 1});
  const auto u_out = g.find_field(handles.u_component, "u_out");
  const auto v_out = g.find_field(handles.v_component, "v_out");
  if (!u_out.has_value() || !v_out.has_value()) return -1;
  const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "stub_u.h5/grid/u_out";
  const_cast<FieldSpec&>(g.field(*v_out)).hdf5_path = "stub_v.h5/grid/v_out";

  {
    // backend.load(decl, graph) -- NOT register_default_placements()
    // directly: the two-argument load() loads the mesh declaration
    // itself FIRST (what register_default_placements() needs already
    // registered, "no mesh with id 0" otherwise), then calls register_
    // default_placements() on it (this overload's own doc comment,
    // include/fuse/backend/native_backend.hpp) -- test/verification/
    // brusselator_external_delivery_size_test.cpp's own precedent.
    const auto r = backend.load(g.mesh(mesh), g);
    if (!r.ok()) { fprintf(stderr, "DIAG backend.load: %s: %s\n", r.error().code.c_str(), r.error().message.c_str()); return -1; }
  }
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { fprintf(stderr, "DIAG lower: %s\n", lowered.diags.to_string().c_str()); return -1; }
  const fuse::ClockSet clocks{g};
  const auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) { fprintf(stderr, "DIAG schedule: %s\n", sched.diags.to_string().c_str()); return -1; }

  fuse::ComponentSet components;
  components.add(std::make_unique<StubBlock>("U"));
  components.add(std::make_unique<StubBlock>("V"));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) { fprintf(stderr, "DIAG register_fields: %s: %s\n", r.error().code.c_str(), r.error().message.c_str()); return -1; }
  }

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  fuse::RunnerConfig cfg;
  {
    const auto r = runner.initialize(components, registry, cfg);
    if (!r.ok()) { fprintf(stderr, "DIAG initialize: %s: %s\n", r.error().code.c_str(), r.error().message.c_str()); return -1; }
  }
  {
    const auto r = runner.run(components, periods);
    if (!r.ok()) { fprintf(stderr, "DIAG run: %s: %s\n", r.error().code.c_str(), r.error().message.c_str()); return -1; }
  }
  return runner.stats().conservation_checks;
}

}  // namespace

// KNOWN_ISSUES.md D57. The literal claim: driving this graph's two
// externally-driven members through fuse_begin_step()/fuse_end_step(),
// with NO end_time declared, for however many ABI calls it takes to see
// the SAME period-boundary signal fuse_begin_step() ever reports twice
// in a row, completes EXACTLY two Runner-internal periods -- never
// three. Before this entry's own fix, `prime_next_request()`'s own
// rollover (src/abi/participant.cpp) answers the call that discovers
// period 1 (0-indexed) is already drained by silently starting AND
// RUNNING period 2's own first chunk in the SAME call, so the SECOND
// period-boundary signal this loop ever observes already reports
// c.periods_completed == 3 (one further period already under way, with
// no bound anywhere stopping it from continuing past that if this test
// kept calling), not 2. Mirrors examples/brusselator/participants/
// u_main.c's/v_main.f90's own period-counting loop exactly: NEITHER
// looks at a wall-clock/has_end_time bound anywhere in its own source
// (grep'd for this round's own report) -- BOTH rely entirely on seeing
// `step_opened == 0` with a FRESH, not-already-cached request (this
// test's own `saw_boundary`, mirroring `rep.next_component == -1`'s own
// role in u_main.c's loop, generalised past the single-component case
// that check alone covers -- see this test's own driving loop comment)
// exactly once per real period.
TEST(BrusselatorPeriodRollover, StopsAtTheRequestedPeriodCountWithNoEndTime) {
  fuse::CouplingGraph g;
  SetupResult setup = build_and_initialize(g);
  ASSERT_NE(setup.ctx, nullptr) << "graph build/setup/initialize failed";
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  constexpr int kRequestedPeriods = 2;
  // Generous but finite: the defect this test pins does not actually
  // diverge for THIS single-process, no-coupler graph (there is no
  // coupler rank to hang against) -- it just keeps completing further,
  // unrequested periods. A real multi-rank run hangs instead (this
  // round's own host trace, KNOWN_ISSUES.md D57); this cap is what
  // keeps THIS reproduction finite and gives a clear failure message
  // instead of a timeout if the defect is somehow worse than measured.
  constexpr int kSafetyCapSteps = 20000;

  std::int32_t next_hint = -1;
  int periods_seen_by_boundary = 0;
  std::int32_t last_reported_period = -1;
  int steps_taken = 0;
  for (; steps_taken < kSafetyCapSteps; ++steps_taken) {
    if (next_hint == -1) {
      fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
      ASSERT_EQ(t->is_coupling_ongoing(setup.ctx, nullptr, &ongoing), FUSE_OK);
      next_hint = ongoing.next_component;
    }
    const bool is_u = (next_hint == setup.u_idx);
    ASSERT_TRUE(is_u || next_hint == setup.v_idx)
        << "step " << steps_taken << ": next_component resolved to neither "
        << "U (" << setup.u_idx << ") nor V (" << setup.v_idx << "), got " << next_hint;
    fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
    fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
    const fuse_status rc = drive_one_step(
        t, setup.ctx, is_u ? "U" : "V", &end_rep, &begin_rep);
    if (rc != FUSE_OK) {
      char buf[2048] = {0};
      fuse_error err{.struct_size = sizeof(err)};
      err.buffer = buf;
      err.capacity = sizeof(buf);
      t->last_error(setup.ctx, nullptr, &err);
      ADD_FAILURE() << "step " << steps_taken << " (" << (is_u ? "U" : "V")
                    << "): rc=" << rc << " " << (err.code ? err.code : "?")
                    << ": " << buf;
    }
    ASSERT_EQ(rc, FUSE_OK) << "step " << steps_taken;

    if (!begin_rep.step_opened) {
      // Mirrors u_main.c's/v_main.f90's own "no step -- a delivery
      // notification, or the period just closed" branch: the ONLY
      // signal either loop ever consults here, this test's own
      // `begin_rep.period` (entries.def: "set on EVERY fuse_step_report
      // regardless of step_opened"), reported true for a genuine
      // period-boundary report, never for a plain delivery-pending one
      // (whose own `period` stays whatever c.periods_completed already
      // was, unchanged).
      if (!begin_rep.delivery_pending && begin_rep.period != last_reported_period) {
        ++periods_seen_by_boundary;
        last_reported_period = begin_rep.period;
        if (periods_seen_by_boundary == kRequestedPeriods) {
          // u_main.c's/v_main.f90's own outer loop would exit HERE --
          // this is the call whose own report says "period
          // kRequestedPeriods is done." fuse_step_report::period itself
          // is NOT the discriminating signal (see this test's own
          // header: a rolled-over call always reports step_opened == 0
          // and the CORRECT c.periods_completed regardless of whether
          // it also silently ran ahead) -- what is checked below,
          // outside this loop, is how much of the Runner's own work
          // (fuse::abi_test::run_stats(), src/abi/test_support.hpp) had
          // ALREADY happened by the time this exact call returned.
          ASSERT_EQ(begin_rep.period, kRequestedPeriods)
              << "fuse_begin_step() reported period " << begin_rep.period
              << " at the " << kRequestedPeriods << "-th period-boundary "
                 "signal (sanity check on this test's own loop)";
          fprintf(stderr, "DIAG stopping-call begin_rep.next_component=%d (== -1 means "
                  "genuinely nothing more without a NEW request; any other value means "
                  "period %d's own first chunk is ALREADY cached/run)\n",
                  begin_rep.next_component, kRequestedPeriods);
          // KNOWN_ISSUES.md D57's own real discriminator: impl_begin_step()
          // (src/abi/participant.cpp) reports next_component == -1 ONLY
          // through its "!c.next_request.has_value()" branch -- genuinely
          // nothing cached, nothing of any further period has run. Every
          // OTHER step_opened==0 case (delivery pending, or the "treat_as_
          // mismatch" rollover branch its own header comment documents)
          // names a REAL component, because `c.next_request` already holds
          // a REAL, already-resolved ExternalAdvanceRequest -- for a
          // rollover specifically, one `resume_period()` call already
          // walked INTO the new period's own stream to find it. A caller
          // that sees anything but -1 here has no way to tell "this is
          // period kRequestedPeriods continuing" from "period
          // kRequestedPeriods+1 already began" -- which is exactly D57's
          // own defect.
          EXPECT_EQ(begin_rep.next_component, -1)
              << "fuse_begin_step() named component " << begin_rep.next_component
              << " as due next, immediately after reporting period "
              << kRequestedPeriods << " complete -- with NO end_time "
                 "declared, that can only be period " << kRequestedPeriods
              << "+1's own already-cached first request (KNOWN_ISSUES.md "
                 "D57), not a legitimate continuation of period "
              << kRequestedPeriods << " (which this report just said was "
                 "already over)";
          break;
        }
      }
      // Trust the Runner's own answer, never guess: -1 here means
      // "re-resolve via fuse_is_coupling_ongoing() at the top of the
      // next iteration" (the SAME fallback this loop's own top already
      // has), not "try the other component" -- the compiled stream can
      // legitimately ask for the SAME component again (an ordinary tick
      // immediately followed by another), and guessing wrong is exactly
      // what produced this test's own first-draft E-ABI-STEP-ORDER
      // failure ("component 'V' asked out of turn... expects 'U' next").
      next_hint = begin_rep.next_component;
      continue;
    }
    next_hint = end_rep.next_component;
  }
  ASSERT_LT(steps_taken, kSafetyCapSteps)
      << "never observed " << kRequestedPeriods << " period-boundary signals "
         "within " << kSafetyCapSteps << " ABI calls -- see this test's own "
         "header for what that would mean";

  // Read BEFORE destroy() (which tears the Runner down). Reported, not
  // asserted: measured (both with and without this entry's own fix)
  // at 6 for this minimal, 1-cell StubBlock graph either way -- the
  // defect's own first visible effect (the EXPECT_EQ above, which DOES
  // discriminate) is reached by resume_period()'s eager SECOND call
  // finding period kRequestedPeriods+1's own FIRST suspend point,
  // which for this graph's own instruction order is an ordinary
  // WindowAdvance needing a fresh external substep -- BEFORE either
  // channel's own ExternalDelivery/Remap/Constrain chain for that
  // period ever runs. Continuing to drive past the stopping point
  // above (what a caller that does not check next_component == -1
  // would do next) is what would let conservation_checks itself
  // diverge, and on the real three-rank MPI graph is exactly the
  // continued, legitimate-looking delivery traffic KNOWN_ISSUES.md
  // D57's own measured host trace shows (U/V's own 122/121 deliveries)
  // before the run reaches that period's own Constrain reduction and
  // hangs. Kept here as a secondary, honestly-labelled data point, not
  // part of this test's own pass/fail claim.
  const fuse::RunStats abi_stats = fuse::abi_test::run_stats(setup.ctx);
  EXPECT_EQ(t->destroy(setup.ctx, nullptr, nullptr), FUSE_OK);
  const std::int64_t reference = reference_conservation_checks(kRequestedPeriods);
  fprintf(stderr,
          "measured (not asserted): abi conservation_checks=%lld, "
          "fuse::Runner::run(.., %d) reference=%lld, abi "
          "instructions_executed=%lld\n",
          (long long)abi_stats.conservation_checks, kRequestedPeriods,
          (long long)reference, (long long)abi_stats.instructions_executed);
}
