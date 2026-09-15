// Task 8.5, Demo B: a coupled run whose data exchange goes through the
// LowFive HDF5-VOL transport -- real interception, in memory, no file ever
// touching disk.
//
// THE HONEST FRAMING (docs/transport/lowfive_transport.hpp's own hazard
// notes, and the reason this demo is shaped the way it is): LowFive here is
// SERIAL, IN-PROCESS ONLY. There is no DistMetadataVOL, no MPI intercomm, and
// `TransportSpec::src_ranks`/`dst_ranks`/`global_ids` are recorded but never
// read by this backend. This is therefore not a distributed demo. What it
// demonstrates, and the only thing it claims, is: an existing HDF5-writing
// component's ordinary `H5Fcreate`/`H5Dcreate`/`H5Dwrite` calls are
// transparently intercepted and the data is delivered to the coupled partner
// with zero bytes ever written to disk.
//
// THE PHYSICS is exactly Demo A's Task 4.7 manufactured pair
// (examples/mms_pair/heat_component.hpp, reused verbatim, not reimplemented):
//
//   left   1D heat equation, explicit Euler, T_t = alpha T_xx on [0,1],
//          T(0,t) = T(1,t) = 0, T(x,0) = sin(pi x). Provides "T".
//   right  dR/dt = kappa (T - R), integrated in closed form. Requires "T_bc".
//
// It is reused rather than reinvented for two reasons: it already has a
// proven, decisive coupling-evidence gate (relax_err << uncoupled_error), and
// reusing it means any difference between this demo and mms_pair's is
// attributable to the TRANSPORT, which is the one thing Demo B is actually
// about. Demo A already proved the coupler's machinery is correct; Demo B's
// novel claim is that the same exchange can be carried through HDF5-VOL
// interception with nothing on disk, and that is what the checks below gate
// on -- not the physics, which is expected to come out identically whichever
// transport carries it (see the SABOTAGE C leg below, which proves exactly
// that and explains why it does not undermine the demo).
//
// WHAT THIS BINARY PROVES, IN ORDER:
//
//   1. The target directory is actually writable (a sentinel is created and
//      removed) -- otherwise "no file on disk" would be vacuously true of an
//      unwritable path and prove nothing.
//   2. A CONTROL: the identical HDF5 call sequence LowFiveTransport::move()
//      issues, but with H5P_DEFAULT (no interception), on the SAME path,
//      does create a real file. Without this, the absence checked in (4)
//      would be unfalsifiable.
//   3. The coupled run itself, through RunnerConfig::transport = "lowfive"
//      and the transport_setup hook this task adds to close the gap between
//      the Runner and a transport's backend-specific one-time setup
//      (LowFiveTransport::set_location + build_plan, exactly the
//      MpiTransport::set_communicator + build_plan gap Task 5.4 left as "the
//      next task's work").
//   4. After the run, the filesystem is checked directly: no file exists at
//      the path a plain HDF5 program would have created one at in (2).
//   5. Diagnostics: LowFiveTransport::intercepted_writes() is nonzero, which
//      is LowFive's OWN callback firing -- connector code executing, not just
//      a return code that plain file I/O would also produce.
//
// THEN THREE SABOTAGE LEGS, because a demo that merely runs proves nothing
// (Tasks 4.2, 4.7, 5.1 and the MPI unpack test all shipped that failure mode):
//
//   SABOTAGE A  RunnerConfig{.transport = "lowfive"} with NO transport_setup
//               -- exactly what every RunnerConfig looked like before this
//               task closed the gap. The registry hands back an unconfigured
//               LowFiveTransport; the run must fail loudly at the first
//               Transport instruction (E-TRANSPORT-UNPLANNED), not silently
//               degrade to a no-op or to disk I/O. This is the proof the gap
//               this task closes was real: without the hook, Demo B cannot
//               be built at all.
//   SABOTAGE B  transport_setup calls set_location() at a directory that
//               cannot exist -- the same "fails loudly rather than writing a
//               file" guarantee Task 8.4 built, re-exercised here inside this
//               demo's own gated run rather than only in a unit test.
//               Expected: E-TRANSPORT-LF-LOCATION.
//   SABOTAGE C  the SAME graph and SAME physics, coupled through plain
//               CopyTransport instead. The physics gate passes here too --
//               that is expected and is not a bug in the demo, because
//               CopyTransport really does deliver the same values. The point
//               of this leg is exactly to show that the physics gate is
//               NOT what distinguishes "went through HDF5-VOL interception"
//               from "didn't": only the filesystem evidence and
//               intercepted_writes() (both unavailable/zero for a
//               CopyTransport run, since it is a different concrete type)
//               tell the two apart.
//
// SINGLE-REGISTRATION SAFETY (LowFive's process-wide VOL singleton, hazard 3
// of this task's brief). This binary constructs LowFiveTransport more than
// once (the main run, and sabotage legs A/B), but see
// src/transport/lowfive_transport.cpp's `vol_singleton()`: the VOL connector
// registration itself is a function-local static, constructed at most once
// per process and never destroyed, regardless of how many LowFiveTransport
// objects come and go -- the hazard `lowfive_transport_test.cpp` already
// exercises via `TwoTransportsCannotClaimOneFileName` and
// `TheNameIsReleasedWhenTheTransportDies`. The only per-instance state is a
// claimed *file name*, released when that transport is destroyed, which is
// why sabotage A and B (which claim no name, or fail before claiming one)
// never collide with the main run's claim on `real_path`.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <hdf5.h>

#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>
#include <soma/transport/transport.hpp>

#include "heat_component.hpp"  // examples/mms_pair -- reused, not duplicated

namespace {

namespace fs = std::filesystem;

constexpr std::size_t  kCells     = 64;
constexpr double       kAlpha     = 0.01;
constexpr double       kKappa     = 2.0;
constexpr std::int64_t kPeriods   = 100;
constexpr std::int64_t kPeriodDen = 1000;
constexpr std::int64_t kSubsteps  = 4;

// Same gates mms_pair's Demo A already validated this physics against.
constexpr double kHeatGate               = 1.0e-4;
constexpr double kRelaxGate              = 3.0e-3;
constexpr double kCouplingEvidenceRatio  = 0.05;

int  g_failures = 0;
void check(bool cond, const std::string& what) {
  if (cond) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

soma::CouplingGraph build_graph(std::size_t cells) {
  soma::CouplingGraph g;
  const auto m_left = g.add_mesh(soma::MeshDecl{
      .name = "left_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto m_right = g.add_mesh(soma::MeshDecl{
      .name = "right_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto r_left = g.add_region(
      soma::Region{.name = "left", .mesh = m_left, .dimension = 1});
  const auto r_right = g.add_region(
      soma::Region{.name = "right", .mesh = m_right, .dimension = 1});

  // Declared in this order so ComponentId assignment (sequential, see
  // CouplingGraph::add_component) makes "left" id 0 and "right" id 1 --
  // exactly the src_rank_set/dst_rank_set the TransportSpec below must match
  // for LowFiveTransport::prepare() to accept it (E-TRANSPORT-RESPEC
  // otherwise). Disjoint single-rank sets, so rules_channel.cpp emits a real
  // Transport op rather than treating the pair as co-located.
  const auto left = g.add_component(soma::ComponentDecl{
      .name = "left", .ranks = {.first = 0, .count = 1},
      .interval = soma::Rational{1, kPeriodDen * kSubsteps}});
  const auto right = g.add_component(soma::ComponentDecl{
      .name = "right", .ranks = {.first = 1, .count = 1},
      .interval = soma::Rational{1, kPeriodDen}});

  const auto t_out = g.add_field(left, soma::FieldSpec{
      .name = "T", .region = r_left, .units = soma::units::kKelvin,
      .conservation = soma::Conservation::Extensive,
      .intent = soma::Intent::Provides});
  const auto t_in = g.add_field(right, soma::FieldSpec{
      .name = "T_bc", .region = r_right, .units = soma::units::kKelvin,
      .conservation = soma::Conservation::Extensive,
      .intent = soma::Intent::Requires});

  const auto ch = g.add_channel("T_link", t_out, t_in);
  g.channel(ch).accumulate(soma::AccumOp::Mean);
  return g;
}

[[nodiscard]] bool path_exists(const std::string& p) {
  std::error_code ec;
  return fs::exists(p, ec);
}

// --- evidence (1): the directory is actually writable -----------------------
[[nodiscard]] bool prove_directory_writable(const fs::path& dir) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  const fs::path sentinel = dir / "soma_demo_b_writable_probe";
  std::FILE*     f = std::fopen(sentinel.string().c_str(), "wb");
  if (f == nullptr) return false;
  std::fclose(f);
  const bool created = path_exists(sentinel.string());
  fs::remove(sentinel, ec);
  return created;
}

// --- evidence (2): the CONTROL. An "unmodified HDF5 component" -- it has
// never heard of LowFive -- issuing the identical call sequence
// LowFiveTransport::move() issues, but with H5P_DEFAULT. Without this, a real
// file failing to appear later would be equally consistent with "intercepted"
// and with "this path can never be written to".
[[nodiscard]] bool control_run_creates_a_real_file(const std::string& path,
                                                   const std::string& dataset,
                                                   std::size_t n) {
  if (path_exists(path)) {
    std::error_code ec;
    fs::remove(path, ec);
  }
  const std::vector<double> data(n, 1.0);
  const auto  dim = static_cast<hsize_t>(n);
  const hid_t space = H5Screate_simple(1, &dim, nullptr);
  if (space < 0) return false;
  const hid_t file =
      H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  bool ok = file >= 0;
  if (ok) {
    const hid_t dset =
        H5Dcreate2(file, dataset.c_str(), H5T_IEEE_F64LE, space, H5P_DEFAULT,
                   H5P_DEFAULT, H5P_DEFAULT);
    ok = dset >= 0 && H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, data.data()) >= 0;
    if (dset >= 0) H5Dclose(dset);
    H5Fclose(file);
  }
  H5Sclose(space);
  const bool appeared = path_exists(path);
  if (appeared) {
    std::error_code ec;
    fs::remove(path, ec);
  }
  return ok && appeared;
}

struct Outcome {
  bool         ok = false;
  std::string  error_code;
  std::string  error;
  double       heat_err = 0.0;
  double       relax_err = 0.0;
  double       uncoupled = 0.0;
  std::int64_t conservation_checks = 0;
  std::int64_t conservation_violations = 0;
  // Only meaningful when the transport actually was a LowFiveTransport --
  // zero for "copy" (sabotage C), which is the point of sabotage C.
  std::size_t  intercepted_writes = 0;
};

/// Builds the SAME graph and the SAME two components every time it is
/// called, wires the coupler with `transport_name` and `setup`, runs
/// `kPeriods`, and reports what happened -- including, if `setup` produced a
/// LowFiveTransport, its interception diagnostic, read while the Runner (and
/// therefore the transport) is still alive.
Outcome run_case(
    const std::string&                                       transport_name,
    std::function<soma::Result<void>(soma::Transport&)> setup) {
  Outcome o;
  soma::LowFiveTransport* lf_ptr = nullptr;

  soma::CouplingGraph graph = build_graph(kCells);
  soma::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(soma::MeshId{i}));
    if (!r.ok()) {
      o.error = "mesh load: " + r.error().message;
      o.error_code = r.error().code;
      return o;
    }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) {
    o.error = lowered.diags.to_string();
    return o;
  }

  const soma::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks,
                        soma::make_region_sizes(graph, backend));
  if (!sched.ok()) {
    o.error = sched.diags.to_string();
    return o;
  }

  soma::ComponentSet components;
  auto left = std::make_unique<mms::HeatComponent>("left", kCells, kAlpha);
  auto right = std::make_unique<mms::RelaxComponent>(
      "right", kCells, kKappa, mms::kPi * mms::kPi * kAlpha);
  mms::HeatComponent*  left_ptr = left.get();
  mms::RelaxComponent* right_ptr = right.get();
  components.add(std::move(left));
  components.add(std::move(right));

  soma::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) {
      o.error = "register_fields: " + r.error().message;
      o.error_code = r.error().code;
      return o;
    }
  }

  soma::RunnerConfig cfg;
  cfg.transport = transport_name;
  if (setup) {
    cfg.transport_setup = [&](soma::Transport& t) -> soma::Result<void> {
      lf_ptr = dynamic_cast<soma::LowFiveTransport*>(&t);
      return setup(t);
    };
  }

  soma::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    o.error = "initialize: " + r.error().message;
    o.error_code = r.error().code;
    return o;
  }
  if (const auto r = runner.run(components, kPeriods); !r.ok()) {
    o.error = "run: " + r.error().message;
    o.error_code = r.error().code;
    return o;
  }

  o.ok = true;
  o.heat_err = left_ptr->l2_error();
  o.relax_err = right_ptr->l2_error();
  o.uncoupled = right_ptr->uncoupled_error();
  o.conservation_checks = runner.stats().conservation_checks;
  o.conservation_violations = runner.stats().conservation_violations;
  // Read before `runner` (and the transport it owns) goes out of scope at the
  // end of this function -- lf_ptr would dangle after that.
  if (lf_ptr != nullptr) o.intercepted_writes = lf_ptr->intercepted_writes();
  return o;
}

}  // namespace

int main() {
  int status = 0;

  const fs::path    dir = fs::temp_directory_path() / "soma_demo_b_lowfive";
  const std::string real_path = (dir / "component_output.h5").string();
  const std::string dataset = "T_bc";

  std::printf(
      "Demo B -- coupled exchange through the LowFive HDF5-VOL transport\n"
      "  physics            : Task 4.7's manufactured heat/relax pair "
      "(examples/mms_pair), unmodified\n"
      "  target path         : %s (never created -- see evidence below)\n"
      "  dataset             : %s\n\n",
      real_path.c_str(), dataset.c_str());

  // ---------------------------------------------------------------------
  // Evidence (1): writability.
  // ---------------------------------------------------------------------
  const bool writable = prove_directory_writable(dir);
  check(writable, "the target directory " + dir.string() +
                      " is not writable; \"no file on disk\" would prove "
                      "nothing there");
  std::printf(
      "[evidence 1] %s is writable (sentinel file created and removed)\n",
      dir.c_str());

  // ---------------------------------------------------------------------
  // Evidence (2): the control.
  // ---------------------------------------------------------------------
  const bool control_ok =
      control_run_creates_a_real_file(real_path, dataset, kCells);
  check(control_ok,
        "control leg: an unmodified HDF5 program (H5P_DEFAULT) did not "
        "create a real file at " +
            real_path +
            "; every 'no file appeared' claim below would be unfalsifiable");
  std::printf(
      "[evidence 2] control: identical HDF5 calls with H5P_DEFAULT on the "
      "same path DID create a real file (now removed)\n\n");

  // ---------------------------------------------------------------------
  // The real run: coupled through LowFive.
  // ---------------------------------------------------------------------
  const soma::TransportSpec spec{
      .src_rank_set = 0, .dst_rank_set = 1, .elements = kCells,
      .components = 1};
  const Outcome main_run = run_case(
      "lowfive", [&](soma::Transport& t) -> soma::Result<void> {
        auto* lf = dynamic_cast<soma::LowFiveTransport*>(&t);
        if (lf == nullptr) {
          return soma::Error{
              "E-DEMO-WRONGTYPE",
              "the registry's \"lowfive\" transport is not a "
              "LowFiveTransport"};
        }
        if (auto r = lf->set_location(real_path, dataset); !r.ok()) return r;
        return lf->build_plan(spec);
      });

  check(main_run.ok, "the lowfive-transported run failed: " + main_run.error);

  // ---------------------------------------------------------------------
  // Evidence (4): post-run filesystem check.
  // ---------------------------------------------------------------------
  const bool absent_after_run = !path_exists(real_path);
  check(absent_after_run,
        "a real file was found at " + real_path +
            " after the coupled run; the LowFive VOL connector did not "
            "intercept the exchange and this channel degraded to disk I/O");
  std::printf(
      "[evidence 4] post-run filesystem check on %s: %s\n",
      real_path.c_str(), absent_after_run ? "absent, as claimed" : "PRESENT");

  // ---------------------------------------------------------------------
  // Evidence (5): the connector's own counter, not just a return code.
  // ---------------------------------------------------------------------
  check(main_run.ok && main_run.intercepted_writes > 0,
        "intercepted_writes() was " +
            std::to_string(main_run.intercepted_writes) +
            "; LowFive's own after_dataset_write callback never fired, so "
            "nothing proves the write went through the connector rather than "
            "around it");
  std::printf(
      "[evidence 5] LowFiveTransport::intercepted_writes() = %zu (must be > "
      "0: connector code executed, which nothing in the native HDF5 path "
      "can produce)\n\n",
      main_run.intercepted_writes);

  if (main_run.ok) {
    std::printf(
        "coupled physics (through LowFive):\n"
        "  L2 error left  (own disc.): %.6e\n"
        "  L2 error right (coupled)  : %.6e\n"
        "  L2 error right (no exch.) : %.6e\n"
        "  coupled / uncoupled       : %.6e\n"
        "  conservation checks/viol. : %lld / %lld\n\n",
        main_run.heat_err, main_run.relax_err, main_run.uncoupled,
        main_run.relax_err / main_run.uncoupled,
        static_cast<long long>(main_run.conservation_checks),
        static_cast<long long>(main_run.conservation_violations));

    check(main_run.conservation_checks > 0,
          "no conservation check ran, so the reported violation count is "
          "vacuous");
    check(main_run.conservation_violations == 0,
          std::to_string(main_run.conservation_violations) +
              " conservation violation(s) on coincident regions");
    check(main_run.heat_err > 0.0 && main_run.heat_err < kHeatGate,
          "left L2 error " + std::to_string(main_run.heat_err) +
              " outside (0, " + std::to_string(kHeatGate) + ")");
    check(main_run.relax_err < kRelaxGate,
          "right L2 error " + std::to_string(main_run.relax_err) +
              " >= " + std::to_string(kRelaxGate));
    check(main_run.relax_err < kCouplingEvidenceRatio * main_run.uncoupled,
          "the coupled error " + std::to_string(main_run.relax_err) +
              " is not decisively below the no-exchange error " +
              std::to_string(main_run.uncoupled) +
              "; the physics gate would pass with the exchange broken");
  }

  // =========================================================================
  // SABOTAGE. A run that merely completes proves nothing (Tasks 4.2, 4.7,
  // 5.1, the MPI unpack test). Three independent ways to break the ONE thing
  // this demo is actually about -- real interception, nothing on disk.
  // =========================================================================

  // --- SABOTAGE A: the exact gap this task closed --------------------------
  // No transport_setup at all: the registry hands back an unconfigured
  // LowFiveTransport, precisely what every RunnerConfig produced before this
  // task added the hook. It must fail loudly at the first Transport
  // instruction, not silently do nothing.
  {
    const Outcome sab_a = run_case("lowfive", {});
    check(!sab_a.ok,
          "SABOTAGE A: RunnerConfig{.transport=\"lowfive\"} with no "
          "transport_setup unexpectedly succeeded -- the runner-to-backend "
          "setup gap this task closes would be invisible if this passed");
    check(sab_a.error_code == "E-TRANSPORT-UNPLANNED",
          "SABOTAGE A: expected E-TRANSPORT-UNPLANNED, got '" +
              sab_a.error_code + "': " + sab_a.error);
    std::printf(
        "[sabotage A] lowfive with no transport_setup: initialize()+run() -> "
        "%s (%s)\n",
        sab_a.ok ? "OK -- WRONG" : "failed, as required", sab_a.error_code.c_str());
  }

  // --- SABOTAGE B: LowFive's own fail-loud guarantee, exercised here -------
  // set_location() at a directory that cannot exist must be refused before
  // any HDF5 call is made, exactly as lowfive_transport_test.cpp's
  // RejectsAnUnwritableLocation proves in isolation -- reproduced here inside
  // this demo's own gate, per the standing rule that the demo itself must
  // prove it, not only a unit test.
  {
    const Outcome sab_b = run_case(
        "lowfive", [](soma::Transport& t) -> soma::Result<void> {
          auto* lf = dynamic_cast<soma::LowFiveTransport*>(&t);
          if (lf == nullptr) {
            return soma::Error{"E-DEMO-WRONGTYPE", "not a LowFiveTransport"};
          }
          return lf->set_location(
              "/soma-demo-b-no-such-directory-4f2c1a/x.h5", "field");
        });
    check(!sab_b.ok,
          "SABOTAGE B: set_location() at a nonexistent directory "
          "unexpectedly succeeded");
    check(sab_b.error_code == "E-TRANSPORT-LF-LOCATION",
          "SABOTAGE B: expected E-TRANSPORT-LF-LOCATION, got '" +
              sab_b.error_code + "': " + sab_b.error);
    std::printf(
        "[sabotage B] set_location() at an unwritable path: initialize() -> "
        "%s (%s)\n",
        sab_b.ok ? "OK -- WRONG" : "failed, as required", sab_b.error_code.c_str());
  }

  // --- SABOTAGE C: same physics, plain CopyTransport ------------------------
  // The physics gate passes here too -- correctly, since CopyTransport
  // genuinely delivers the same values. That is not a hole in the demo: it is
  // the proof that the physics gate is NOT what distinguishes "went through
  // HDF5-VOL interception" from "didn't". Only the filesystem evidence above
  // and intercepted_writes() (necessarily 0 here -- there is no
  // LowFiveTransport to read it from) do that.
  {
    const Outcome sab_c = run_case("copy", {});
    check(sab_c.ok,
          "SABOTAGE C (copy transport) unexpectedly failed: " + sab_c.error);
    check(sab_c.intercepted_writes == 0,
          "SABOTAGE C: a copy-transport run reported nonzero "
          "intercepted_writes(), which should be impossible");
    std::printf(
        "[sabotage C] copy transport, same physics: relax_err = %.6e "
        "(lowfive run: %.6e) -- indistinguishable by physics alone; "
        "intercepted_writes() = %zu is the only thing that tells them apart\n",
        sab_c.relax_err, main_run.relax_err, sab_c.intercepted_writes);
    check(main_run.ok &&
              sab_c.ok
              // Both must actually solve the physics -- otherwise this leg
              // would be comparing a broken lowfive run against a broken
              // copy run and prove nothing.
              && sab_c.relax_err < kRelaxGate && main_run.relax_err < kRelaxGate,
          "SABOTAGE C: the comparison requires both runs to have actually "
          "converged to the physical answer");
  }

  std::printf("\n");
  if (g_failures != 0) {
    std::fprintf(stderr, "%d gate(s) failed\n", g_failures);
    status = 1;
  } else {
    std::printf("all gates passed\n");
  }
  return status;
}
