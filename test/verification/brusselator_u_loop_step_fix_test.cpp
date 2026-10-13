// KNOWN_ISSUES.md D57 ("Collectives" track), Round 2. Companion to
// brusselator_period_rollover_test.cpp's own TEST, which mirrors
// `u_loop_step()`'s/`v_loop_step()`'s zeroing logic VERBATIM in a
// standalone helper (`drive_one_step()`, that file) rather than calling
// the real `examples/brusselator/participants/u_loop.c`/`v_loop.f90`
// functions -- deliberately, to keep that file's own grid trivial and
// fast. A mirror can drift from the file it was copied from without
// anyone noticing (exactly the risk this track's own standing rule
// against "a model of it that could silently stop matching" warns
// about), so THIS file exists to close that gap: it drives the REAL
// `u_loop_step()` (u_loop.h) and the REAL `bruss_v_loop_step()`
// (v_loop.h, v_loop.f90's own `bind(C, name="bruss_v_loop_step")`
// export) directly, through real `bruss_u_handle`/`bruss_v_handle`
// physics objects (shim.h), so a mutation reverting u_loop.c's/
// v_loop.f90's own `out_rep->next_component = out_begin_rep->
// next_component;` line (the fix itself) is caught by rebuilding and
// running THIS test -- not just a model of the file that was never
// actually recompiled.
//
// Same graph shape as brusselator_period_rollover_test.cpp's own
// build_and_initialize() (LowFive-bound transfer fields, no end_time,
// communicator forced via the same test-only seam) and the same
// decisive assertion (a REAL step of an unrequested period ever
// running), but with a 1x1 grid driven by the REAL physics shim instead
// of StubBlock, so this is a genuine, if minimal, instance of the
// u_main.c/v_main.f90 shape -- not a parallel reimplementation of it.
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"

#include "brusselator_graph.hpp"
#include "shim.h"
#include "species_core.hpp"
#include "test_support.hpp"
#include "u_loop.h"
#include "v_loop.h"

namespace {

using fuse::ComponentKind;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::Rational;
using fuse::RankSet;
using fuse::WaveformIteration;

// Same genuinely multirate dt_u/dt_v/window every other file in this
// conformance suite uses; the grid itself (1x1) is irrelevant to this
// defect (see brusselator_period_rollover_test.cpp's own header on this
// same point) so this test runs in well under a second despite driving
// the REAL reaction-diffusion solve on every substep.
constexpr int kNx = 1, kNy = 1;
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;

const fuse_participant_abi_v1* Table() {
  static const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

struct SetupResult {
  fuse_ctx* ctx = nullptr;
  std::int32_t u_idx = -1, v_idx = -1;
  bruss_u_handle* uh = nullptr;
  bruss_v_handle* vh = nullptr;
  std::vector<double> u_in, u_out, v_in, v_out;
};

// Builds the SAME graph shape brusselator_period_rollover_test.cpp's own
// build_and_initialize() does (LowFive-bound channels, no coupler
// remap beyond the identity, has_end_time left at its zero default),
// but registers REAL bruss_u_handle/bruss_v_handle physics instead of
// StubBlock, exactly the substitution this file's own header explains.
SetupResult build_and_initialize(fuse::CouplingGraph& g) {
  SetupResult out;
  const fuse::MeshId mesh = g.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNx) + "x" + std::to_string(kNy)});
  const fuse::RegionId region = g.add_region({.name = "u_all", .mesh = mesh, .dimension = 2});

  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      g, region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1},
      RankSet{0, 1}, RankSet{0, 1});

  const auto u_out_f = g.find_field(handles.u_component, "u_out");
  const auto v_out_f = g.find_field(handles.v_component, "v_out");
  if (!u_out_f.has_value() || !v_out_f.has_value()) return out;
  const_cast<fuse::FieldSpec&>(g.field(*u_out_f)).hdf5_path = "stub_u.h5/grid/u_out";
  const_cast<fuse::FieldSpec&>(g.field(*v_out_f)).hdf5_path = "stub_v.h5/grid/v_out";
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

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  if (t->setup(out.ctx, &opts, nullptr) != FUSE_OK) return out;

  const int n = kNx * kNy;
  out.u_in.assign(static_cast<std::size_t>(n), 0.0);
  out.u_out.assign(static_cast<std::size_t>(n), 0.0);
  out.v_in.assign(static_cast<std::size_t>(n), 0.0);
  out.v_out.assign(static_cast<std::size_t>(n), 0.0);

  auto register_one = [&](const char* component, const char* field, double* data) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    if (t->field_handle(out.ctx, &fname, &fref) != FUSE_OK) return false;
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = static_cast<uint64_t>(n);
    fview.owned_count = static_cast<uint64_t>(n);
    fview.stride = 1;
    fview.data = data;
    return t->register_field(out.ctx, &fview, nullptr) == FUSE_OK;
  };
  if (!register_one("U", "u_out", out.u_out.data())) return out;
  if (!register_one("U", "u_in", out.u_in.data())) return out;
  if (!register_one("V", "v_out", out.v_out.data())) return out;
  if (!register_one("V", "v_in", out.v_in.data())) return out;

  if (t->initialize(out.ctx, nullptr, nullptr) != FUSE_OK) {
    out.ctx = nullptr;
    return out;
  }

  // See build_and_initialize()'s own comment in
  // brusselator_period_rollover_test.cpp for why this is both necessary
  // (a non-zero communicator through the real public ABI fails
  // initialize() outright at tier 0) and safe (every MPI call site it
  // reaches is compiled out under `#if FUSE_HAVE_MPI`).
  if (!fuse::abi_test::force_communicator_set(out.ctx, 1)) {
    out.ctx = nullptr;
    return out;
  }

  const brusselator::PhysicsConfig cfg;  // defaults: diffusion=10, A=3.4, B=1, forcing=true
  std::vector<double> v0_on_u(static_cast<std::size_t>(n), 0.0);
  out.uh = bruss_u_create(kNx, kNy, cfg.diffusion, cfg.A, cfg.B, cfg.forcing ? 1 : 0,
                          v0_on_u.data(), /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  if (out.uh == nullptr) { out.ctx = nullptr; return out; }
  out.vh = bruss_v_create(kNx, kNy, kNx, kNy, cfg.diffusion,
                          /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  if (out.vh == nullptr) { out.ctx = nullptr; return out; }

  const auto u_cid = g.find_component("U");
  const auto v_cid = g.find_component("V");
  out.u_idx = static_cast<std::int32_t>(u_cid->value());
  out.v_idx = static_cast<std::int32_t>(v_cid->value());
  return out;
}

}  // namespace

// KNOWN_ISSUES.md D57, Round 2. Drives the REAL u_loop_step()/
// bruss_v_loop_step() -- not a mirror -- through a fixed, 2-period outer
// bound exactly like u_main.c's/v_main.f90's own `for (p = 0; p <
// periods; ++p)`, with the only period-closed signal either real driver
// ever reads (`rep.next_component == -1`, u_loop.c's/v_main.f90's own
// `out_rep`/`end_rep`). Mutating away the fix this entry's own u_loop.c/
// v_loop.f90 change made (`out_rep->next_component = out_begin_rep->
// next_component;`, guarded by `!out_begin_rep->step_opened`) must make
// this FAIL the same way brusselator_period_rollover_test.cpp's own
// mirror-based test does -- a real step of period index kRequestedPeriods
// (one Runner::run(components, kRequestedPeriods) never requests) is
// reached because neither u_loop_step() nor bruss_v_loop_step() lets the
// caller observe the correctly-reported -1.
TEST(BrusselatorULoopStepFix, RealShimFunctionsNeverSeeThePeriodBoundaryThroughTheZeroedReport) {
  fuse::CouplingGraph g;
  SetupResult setup = build_and_initialize(g);
  ASSERT_NE(setup.ctx, nullptr) << "graph build/setup/initialize/physics-create failed";
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  constexpr int kRequestedPeriods = 2;  // the real host row's own --periods 2
  constexpr int kSafetyCapSteps = 500;  // see brusselator_period_rollover_test.cpp's own rationale

  int global_periods_done = 0;
  std::int32_t next_hint = -1;
  bool observed_unauthorized_period = false;
  std::int32_t unauthorized_period_value = -999;
  int steps_taken = 0;
  for (; steps_taken < kSafetyCapSteps && global_periods_done < kRequestedPeriods;
       ++steps_taken) {
    if (next_hint == -1) {
      fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
      ASSERT_EQ(t->is_coupling_ongoing(setup.ctx, nullptr, &ongoing), FUSE_OK)
          << "step " << steps_taken;
      next_hint = ongoing.next_component;
    }
    const bool is_u = (next_hint == setup.u_idx);
    ASSERT_TRUE(is_u || next_hint == setup.v_idx)
        << "step " << steps_taken << ": next_component resolved to neither "
        << "U (" << setup.u_idx << ") nor V (" << setup.v_idx << "), got " << next_hint;

    fuse_step_request req{.struct_size = sizeof(req)};
    req.component = is_u ? "U" : "V";
    fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
    fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
    const fuse_status rc = is_u
        ? u_loop_step(setup.ctx, setup.uh, &req, setup.u_in.data(), setup.u_out.data(),
                     &end_rep, &begin_rep)
        : bruss_v_loop_step(setup.ctx, setup.vh, &req, setup.v_in.data(), setup.v_out.data(),
                            &end_rep, &begin_rep);
    if (rc != FUSE_OK) {
      char buf[2048] = {0};
      fuse_error err{.struct_size = sizeof(err)};
      err.buffer = buf;
      err.capacity = sizeof(buf);
      t->last_error(setup.ctx, nullptr, &err);
      ADD_FAILURE() << "step " << steps_taken << " (" << (is_u ? "U" : "V")
                    << "): rc=" << rc << " " << (err.code ? err.code : "?")
                    << ": " << buf
                    << (is_u ? bruss_u_last_error(setup.uh) : bruss_v_last_error(setup.vh));
      break;
    }

    // The decisive, airtight signal -- same as
    // brusselator_period_rollover_test.cpp's own: a REAL step
    // (checkpoint/physics/end_step all ran, step_opened == 1) of period
    // index kRequestedPeriods or beyond.
    if (begin_rep.step_opened && begin_rep.period >= kRequestedPeriods) {
      observed_unauthorized_period = true;
      unauthorized_period_value = begin_rep.period;
      break;
    }

    if (!begin_rep.step_opened) {
      if (end_rep.next_component == -1) {
        ++global_periods_done;
      }
      next_hint = begin_rep.next_component;  // dispatch artifact only.
      continue;
    }

    if (end_rep.next_component == -1) {
      ++global_periods_done;
    }
    next_hint = end_rep.next_component;
  }

  fprintf(stderr,
          "DIAG (real shim) steps_taken=%d global_periods_done=%d "
          "observed_unauthorized_period=%d unauthorized_period_value=%d\n",
          steps_taken, global_periods_done, observed_unauthorized_period ? 1 : 0,
          unauthorized_period_value);

  EXPECT_FALSE(observed_unauthorized_period)
      << "a real step of period " << unauthorized_period_value
      << " ran (kRequestedPeriods=" << kRequestedPeriods
      << ") through the REAL u_loop_step()/bruss_v_loop_step() -- "
         "KNOWN_ISSUES.md D57";
  EXPECT_LT(steps_taken, kSafetyCapSteps)
      << "never reached " << kRequestedPeriods << " periods done (global_"
         "periods_done=" << global_periods_done << ") within " << kSafetyCapSteps
      << " ABI calls";

  EXPECT_EQ(t->destroy(setup.ctx, nullptr, nullptr), FUSE_OK);
  if (setup.uh != nullptr) bruss_u_destroy(setup.uh);
  if (setup.vh != nullptr) bruss_v_destroy(setup.vh);
}

// KNOWN_ISSUES.md D57, Round 2. v_loop.f90's own counterpart to the test
// above: TRACED (is_u/begin_rep.next_component/begin_rep.period printed
// at every step_opened == 0 call in the test above, both
// WaveformIteration::Jacobi and ::GaussSeidel, and with kDtU/kDtV
// swapped) that the call which ends up observing `c.next_request` empty
// at the genuine period-close -- the ONE value (-1) that makes a
// zeroing mutation observable -- is ALWAYS graph component 0 ("U" by
// name in this test's own graph), never component 1 ("V"): an artifact
// of this single-ctx simulation sharing ONE `c.next_request` cache for
// a schedule that otherwise has no reason to prefer one component over
// the other (the real row's two ranks each run their OWN independent
// prime_next_request(), so this tie never arises there -- see this
// test file's own header). Rather than leave v_loop.f90's real
// propagation line unexercised at the one value that matters, THIS test
// drives graph component 0 ("U") through the REAL `bruss_v_loop_step()`
// (v_loop.f90) instead of `u_loop_step()` -- legal because neither real
// shim function's own wrapper logic (fuse_begin_step()/physics-or-
// nothing/fuse_end_step(), u_loop.h's/v_loop.h's own doc comments) reads
// the component NAME at all; `req->component` (set to whatever this
// test's caller supplies) is what the ABI itself matches against the
// graph's own declared component, not anything the shim inspects -- and
// graph component 1 ("V") through `u_loop_step()`, crossed the other
// way, so the test overall still drives the real graph's own two
// members with the real shim functions, just with the file that
// happens to reach the schedule's structurally-terminal slot in THIS
// graph shape swapped to the one this entry needs to isolate.
TEST(BrusselatorULoopStepFix, RealVLoopFortranFunctionNeverSeesThePeriodBoundaryThroughTheZeroedReport) {
  fuse::CouplingGraph g;
  SetupResult setup = build_and_initialize(g);
  ASSERT_NE(setup.ctx, nullptr) << "graph build/setup/initialize failed";
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  constexpr int kRequestedPeriods = 2;
  constexpr int kSafetyCapSteps = 500;

  int global_periods_done = 0;
  std::int32_t next_hint = -1;
  bool observed_unauthorized_period = false;
  std::int32_t unauthorized_period_value = -999;
  int steps_taken = 0;
  for (; steps_taken < kSafetyCapSteps && global_periods_done < kRequestedPeriods;
       ++steps_taken) {
    if (next_hint == -1) {
      fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
      ASSERT_EQ(t->is_coupling_ongoing(setup.ctx, nullptr, &ongoing), FUSE_OK)
          << "step " << steps_taken;
      next_hint = ongoing.next_component;
    }
    // component 0 ("U" by graph declaration) is driven by the REAL
    // bruss_v_loop_step() (v_loop.f90); component 1 ("V") by the REAL
    // u_loop_step() (u_loop.c) -- crossed, this test's own header
    // comment explains why. Either real physics handle works as the
    // `vh`/`uh` argument regardless of which graph component name is
    // attached to the call (the shim's own physics never reads that
    // name), so this is still two real, independent reaction-diffusion
    // solves, just swapped relative to their usual graph role.
    const bool is_component0 = (next_hint == setup.u_idx);
    ASSERT_TRUE(is_component0 || next_hint == setup.v_idx)
        << "step " << steps_taken << ": next_component resolved to neither "
        << "U (" << setup.u_idx << ") nor V (" << setup.v_idx << "), got " << next_hint;

    fuse_step_request req{.struct_size = sizeof(req)};
    req.component = is_component0 ? "U" : "V";
    fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
    fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
    const fuse_status rc = is_component0
        ? bruss_v_loop_step(setup.ctx, setup.vh, &req, setup.u_in.data(), setup.u_out.data(),
                            &end_rep, &begin_rep)
        : u_loop_step(setup.ctx, setup.uh, &req, setup.v_in.data(), setup.v_out.data(),
                     &end_rep, &begin_rep);
    if (rc != FUSE_OK) {
      char buf[2048] = {0};
      fuse_error err{.struct_size = sizeof(err)};
      err.buffer = buf;
      err.capacity = sizeof(buf);
      t->last_error(setup.ctx, nullptr, &err);
      ADD_FAILURE() << "step " << steps_taken << " (component " << (is_component0 ? "U" : "V")
                    << ", driven by " << (is_component0 ? "bruss_v_loop_step" : "u_loop_step")
                    << "): rc=" << rc << " " << (err.code ? err.code : "?") << ": " << buf;
      break;
    }

    if (begin_rep.step_opened && begin_rep.period >= kRequestedPeriods) {
      observed_unauthorized_period = true;
      unauthorized_period_value = begin_rep.period;
      break;
    }

    if (!begin_rep.step_opened) {
      if (end_rep.next_component == -1) {
        ++global_periods_done;
      }
      next_hint = begin_rep.next_component;
      continue;
    }

    if (end_rep.next_component == -1) {
      ++global_periods_done;
    }
    next_hint = end_rep.next_component;
  }

  fprintf(stderr,
          "DIAG (crossed shim, component0=bruss_v_loop_step) steps_taken=%d "
          "global_periods_done=%d observed_unauthorized_period=%d "
          "unauthorized_period_value=%d\n",
          steps_taken, global_periods_done, observed_unauthorized_period ? 1 : 0,
          unauthorized_period_value);

  EXPECT_FALSE(observed_unauthorized_period)
      << "a real step of period " << unauthorized_period_value
      << " ran (kRequestedPeriods=" << kRequestedPeriods
      << ") -- component 0 (\"U\"), driven by the REAL bruss_v_loop_step() "
         "(v_loop.f90), reached the schedule's own structurally-terminal "
         "slot in this graph shape and failed to propagate the genuine "
         "close -- KNOWN_ISSUES.md D57";
  EXPECT_LT(steps_taken, kSafetyCapSteps)
      << "never reached " << kRequestedPeriods << " periods done (global_"
         "periods_done=" << global_periods_done << ") within " << kSafetyCapSteps
      << " ABI calls";

  EXPECT_EQ(t->destroy(setup.ctx, nullptr, nullptr), FUSE_OK);
  if (setup.uh != nullptr) bruss_u_destroy(setup.uh);
  if (setup.vh != nullptr) bruss_v_destroy(setup.vh);
}

