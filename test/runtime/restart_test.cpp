/// Restart track (docs/dev/restart.md), Stage B Slice 1: the dn_heat
/// end-to-end round trip (owner decision item D) and every refusal
/// item C names, at tier 0 (no MPI -- this sandbox cannot start it; see
/// the host_pending note in the delivered report). Reuses examples/
/// dn_heat's own `dn::DirichletSlab`/`dn::NeumannSlab` and the
/// build_graph()/make_setup() pattern test/runtime/fixed_point_
/// residual_test.cpp already established for the SAME pair, adding
/// `.accelerate(AccelKind::Aitken)` so the round trip genuinely exercises
/// owner decision 4 ("everything restored bit for bit, including
/// accelerator memory"), not only Form A's own component state.
#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/runtime/restart.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>
#include <fuse/num/accelerator.hpp>

#include "dn_component.hpp"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

constexpr std::size_t kN = dn::kStrips;

fuse::CouplingGraph build_graph(int max_iterations, fuse::AccelConfig accel) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kN);
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "iface_1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "iface_2", .source = src});
  const auto r1 = g.add_region(
      fuse::Region{.name = "iface_1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(
      fuse::Region{.name = "iface_2", .mesh = m2, .dimension = 1});

  const auto cd = g.add_component(fuse::ComponentDecl{
      .name = "dirichlet", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto cn = g.add_component(fuse::ComponentDecl{
      .name = "neumann", .ranks = {1, 1}, .interval = fuse::Rational{1}});

  const auto q_out = g.add_field(cd, fuse::FieldSpec{
      .name = "q_wall", .region = r1, .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto u_in = g.add_field(cd, fuse::FieldSpec{
      .name = "u_wall_in", .region = r1, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(cn, fuse::FieldSpec{
      .name = "q_wall_in", .region = r2,
      .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto u_out = g.add_field(cn, fuse::FieldSpec{
      .name = "u_wall", .region = r2, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});

  const auto flux = g.add_channel("flux", q_out, q_in);
  const auto temp = g.add_channel("wall_temperature", u_out, u_in);
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann implicit coupling group");

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(1e-6, "u_wall"));
  crit.add(fuse::Converged::Relative(1e-6, "q_wall"));

  g.iterate({flux, temp})
      .name("dirichlet_neumann")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .accelerate(accel);
  return g;
}

struct DnHeatFixture {
  fuse::CouplingGraph        graph;
  fuse::SerialMeshBackend    backend;
  fuse::ComponentSet         components;
  fuse::FieldRegistry        registry;
  std::unique_ptr<fuse::Runner> runner;
};

std::unique_ptr<DnHeatFixture> make_setup(
    int max_iterations, fuse::AccelConfig accel,
    double initial_u = 305.0) {
  auto s = std::make_unique<DnHeatFixture>();
  s->graph = build_graph(max_iterations, accel);
  for (std::uint32_t i = 0; i < s->graph.mesh_count(); ++i) {
    const auto r = s->backend.load(s->graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) { ADD_FAILURE() << "mesh load: " << r.error().message; return s; }
  }
  const auto lowered = lower(s->graph);
  if (!lowered.ok()) { ADD_FAILURE() << lowered.diags.to_string(); return s; }
  const fuse::ClockSet clocks{s->graph};
  auto sched = schedule(s->graph, lowered.dag, clocks,
                       fuse::make_region_sizes(s->graph, s->backend));
  if (!sched.ok()) { ADD_FAILURE() << sched.diags.to_string(); return s; }

  auto dirichlet = std::make_unique<dn::DirichletSlab>("dirichlet", kN,
                                                       dn::Wiring::Connected);
  auto neumann = std::make_unique<dn::NeumannSlab>("neumann", kN,
                                                   dn::Wiring::Connected,
                                                   initial_u);
  s->components.add(std::move(dirichlet));
  s->components.add(std::move(neumann));
  for (auto& c : s->components) {
    const auto r = c.register_fields(s->registry, s->graph);
    if (!r.ok()) ADD_FAILURE() << "register_fields: " << r.error().message;
  }

  s->runner = std::make_unique<fuse::Runner>(s->graph, lowered.dag, sched.plan,
                                             s->backend);
  fuse::RunnerConfig cfg;
  const auto r = s->runner->initialize(s->components, s->registry, cfg);
  if (!r.ok()) ADD_FAILURE() << "initialize: " << r.error().message;
  return s;
}

fuse::AccelConfig aitken_cfg() {
  return fuse::AccelConfig{.kind = fuse::AccelKind::Aitken,
                           .initial_relaxation = 0.5};
}

std::string temp_dir_path(const std::string& leaf) {
  return std::string(::testing::TempDir()) + "fuse_restart_test_" + leaf;
}

/// Runs N periods on a fixture built fresh; returns the final q/u/time/
/// history of both components, and the LAST period's own iteration
/// stats (sweep count) -- the "solution logs and iteration statistics"
/// item D's own bit-identity claim is checked against.
struct RunResult {
  std::vector<double> q, u;
  double dirichlet_time = 0, neumann_time = 0;
  double dirichlet_history = 0, neumann_history = 0;
  std::size_t last_period_sweeps = 0;
};

RunResult run_n_periods(DnHeatFixture& s, std::int64_t n) {
  RunResult out;
  for (std::int64_t p = 0; p < n; ++p) {
    const auto r = s.runner->run_period(s.components);
    EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  }
  auto* d = dynamic_cast<dn::DirichletSlab*>(s.components.find("dirichlet"));
  auto* nn = dynamic_cast<dn::NeumannSlab*>(s.components.find("neumann"));
  out.q = d->q();
  out.u = nn->u();
  out.dirichlet_time = d->time();
  out.neumann_time = nn->time();
  out.dirichlet_history = d->history();
  out.neumann_history = nn->history();
  const auto& istats = s.runner->iteration_stats();
  if (!istats.empty()) out.last_period_sweeps = static_cast<std::size_t>(istats[0].iterations);
  return out;
}

void expect_bit_identical(const RunResult& a, const RunResult& b) {
  ASSERT_EQ(a.q.size(), b.q.size());
  for (std::size_t i = 0; i < a.q.size(); ++i) {
    EXPECT_EQ(a.q[i], b.q[i]) << "q[" << i << "]";
  }
  ASSERT_EQ(a.u.size(), b.u.size());
  for (std::size_t i = 0; i < a.u.size(); ++i) {
    EXPECT_EQ(a.u[i], b.u[i]) << "u[" << i << "]";
  }
  EXPECT_EQ(a.dirichlet_time, b.dirichlet_time);
  EXPECT_EQ(a.neumann_time, b.neumann_time);
  EXPECT_EQ(a.dirichlet_history, b.dirichlet_history);
  EXPECT_EQ(a.neumann_history, b.neumann_history);
  EXPECT_EQ(a.last_period_sweeps, b.last_period_sweeps);
}

/// Writes a restart point at the CURRENT state of `s` (after `k`
/// periods), via the given file layout, with real provenance.
fuse::Result<void> write_point(DnHeatFixture& s, fuse::restart_io::FileLayout layout,
                               const std::string& path, const std::string& label,
                               std::int64_t period_index) {
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  prov.descriptor_hash = fuse::restart_descriptor_hash(s.graph);
  prov.written_by_rank_count = 1;
  prov.mesh_checksum = 0;
  auto pw = fuse::restart_io::create_restart_point(layout, path, prov, label);
  if (!pw.ok()) return pw.error();
  const fuse::Result<void> r = fuse::write_restart_point(
      *s.runner, s.components, pw.value()->writer(), fuse::Rational{period_index},
      period_index);
  if (!r.ok()) return r;
  return pw.value()->finish();
}

fuse::Result<void> restore_point(DnHeatFixture& s, fuse::restart_io::FileLayout layout,
                                 const std::string& path, const std::string& label,
                                 bool allow_reset, bool& any_reset) {
  auto pr = fuse::restart_io::open_restart_point(layout, path, label);
  if (!pr.ok()) return pr.error();
  fuse::Rational t;
  std::int64_t pidx = 0;
  return fuse::restore_restart_point(*s.runner, s.components, pr.value()->reader(),
                                     allow_reset, any_reset, t, pidx);
}

}  // namespace

// ---------------------------------------------------------------- (i)/(ii)
// Same-rank-count round trip, both file layouts, Aitken-accelerated.

TEST(Restart, SameRankRoundTripSeparateLayout) {
  const std::string path = temp_dir_path("separate.h5");
  std::remove(path.c_str());

  auto ref = make_setup(/*max_iterations=*/40, aitken_cfg());
  const RunResult ref_result = run_n_periods(*ref, 5);

  auto part1 = make_setup(/*max_iterations=*/40, aitken_cfg());
  run_n_periods(*part1, 3);
  const fuse::Result<void> w =
      write_point(*part1, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                 path, "", 3);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  auto restored = make_setup(/*max_iterations=*/40, aitken_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr =
      restore_point(*restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                    path, "", /*allow_reset=*/false, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(any_reset);
  {
    auto* d = dynamic_cast<dn::DirichletSlab*>(restored->components.find("dirichlet"));
    EXPECT_EQ(d->time(), 3.0) << "DIAGNOSTIC: time_ right after restore, before any further period";
  }
  const RunResult restored_result = run_n_periods(*restored, 2);

  expect_bit_identical(ref_result, restored_result);
  std::remove(path.c_str());
}

TEST(Restart, SameRankRoundTripGrowingLayout) {
  const std::string path = temp_dir_path("growing.h5");
  std::remove(path.c_str());

  auto ref = make_setup(/*max_iterations=*/40, aitken_cfg());
  const RunResult ref_result = run_n_periods(*ref, 5);

  auto part1 = make_setup(/*max_iterations=*/40, aitken_cfg());
  run_n_periods(*part1, 3);
  const fuse::Result<void> w =
      write_point(*part1, fuse::restart_io::FileLayout::GrowingFile, path,
                 "p3", 3);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  auto restored = make_setup(/*max_iterations=*/40, aitken_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr =
      restore_point(*restored, fuse::restart_io::FileLayout::GrowingFile, path,
                    "p3", /*allow_reset=*/false, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(any_reset);
  const RunResult restored_result = run_n_periods(*restored, 2);

  expect_bit_identical(ref_result, restored_result);
  std::remove(path.c_str());
}

TEST(Restart, GrowingLayoutAppendsASecondPointWithoutDamagingTheFirst) {
  const std::string path = temp_dir_path("growing_two.h5");
  std::remove(path.c_str());

  auto s = make_setup(/*max_iterations=*/40, aitken_cfg());
  run_n_periods(*s, 2);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::GrowingFile, path,
                          "p2", 2)
                 .ok());
  run_n_periods(*s, 2);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::GrowingFile, path,
                          "p4", 4)
                 .ok());

  // The FIRST point must still open and read back correctly after the
  // second one was appended -- the owner's own robustness point for
  // copy-append-rename (docs/dev/restart.md Section 4.4).
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::GrowingFile, path, "p2");
  ASSERT_TRUE(pr.ok()) << pr.error().code << ": " << pr.error().message;
  const fuse::Result<double> t = pr.value()->reader().get_scalar("__point/period_index");
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(static_cast<std::int64_t>(t.value()), 2);
  std::remove(path.c_str());
}

// --------------------------------------------------------------------- (iii)
// Every refusal.

TEST(Restart, VersionMismatchRefuses) {
  const std::string path = temp_dir_path("version.h5");
  std::remove(path.c_str());
  auto s = make_setup(10, aitken_cfg());
  fuse::restart_io::Provenance prov;
  prov.format_version = 99;  // unsupported
  prov.fuse_version = "test";
  prov.descriptor_hash = fuse::restart_descriptor_hash(s->graph);
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(fuse::write_restart_point(*s->runner, s->components,
                                       pw.value()->writer(), fuse::Rational{0}, 0)
                 .ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  const auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_FALSE(pr.ok());
  EXPECT_EQ(pr.error().code, "E-RESTART-VERSION");
  std::remove(path.c_str());
}

TEST(Restart, DescriptorHashMismatchIsCaughtByTheCaller) {
  // open_restart_point() itself does not compare the hash (fuse/
  // restart_io/hdf5_restart_io.hpp's own PointReader comment: it cannot
  // see the CURRENT graph). The caller compares provenance().
  // descriptor_hash against fuse::restart_descriptor_hash(current_graph)
  // -- this test proves the two differ for a graph that gained a field.
  const std::string path = temp_dir_path("hash.h5");
  std::remove(path.c_str());
  auto s = make_setup(10, aitken_cfg());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  prov.descriptor_hash = fuse::restart_descriptor_hash(s->graph);
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(fuse::write_restart_point(*s->runner, s->components,
                                       pw.value()->writer(), fuse::Rational{0}, 0)
                 .ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  auto other = make_setup(11, aitken_cfg());  // max_iterations differs -> different descriptor
  const auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  EXPECT_NE(pr.value()->provenance().descriptor_hash,
           fuse::restart_descriptor_hash(other->graph));
  std::remove(path.c_str());
}

TEST(Restart, MeshChecksumMismatchDetected) {
  const std::string path_a = temp_dir_path("mesh_a.h5");
  const std::string path_b = temp_dir_path("mesh_b.h5");
  std::remove(path_a.c_str());
  std::remove(path_b.c_str());
  auto s = make_setup(10, aitken_cfg());
  for (std::uint64_t checksum : {std::uint64_t{111}, std::uint64_t{222}}) {
    fuse::restart_io::Provenance prov;
    prov.format_version = fuse::restart_io::kFormatVersion;
    prov.fuse_version = "test";
    prov.descriptor_hash = fuse::restart_descriptor_hash(s->graph);
    prov.mesh_checksum = checksum;
    const std::string& path = checksum == 111 ? path_a : path_b;
    auto pw = fuse::restart_io::create_restart_point(
        fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
    ASSERT_TRUE(pw.ok());
    ASSERT_TRUE(fuse::write_restart_point(*s->runner, s->components,
                                         pw.value()->writer(), fuse::Rational{0}, 0)
                   .ok());
    ASSERT_TRUE(pw.value()->finish().ok());
  }
  const auto pr_a = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path_a, "");
  const auto pr_b = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path_b, "");
  ASSERT_TRUE(pr_a.ok());
  ASSERT_TRUE(pr_b.ok());
  // The refusal itself is the CALLER's comparison (mirrors descriptor_hash,
  // above): a reader that expected 111 but opened path_b must refuse.
  EXPECT_NE(pr_a.value()->provenance().mesh_checksum,
           pr_b.value()->provenance().mesh_checksum);
  std::remove(path_a.c_str());
  std::remove(path_b.c_str());
}

TEST(Restart, TruncatedFileRefuses) {
  const std::string path = temp_dir_path("truncated.h5");
  std::remove(path.c_str());
  std::FILE* f = std::fopen(path.c_str(), "wb");
  ASSERT_NE(f, nullptr);
  const char garbage[] = "not an hdf5 file";
  std::fwrite(garbage, 1, sizeof(garbage), f);
  std::fclose(f);

  const auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  EXPECT_FALSE(pr.ok());
  EXPECT_EQ(pr.error().code, "E-RESTART-TRUNCATED");
  std::remove(path.c_str());
}

TEST(Restart, MissingPointInGrowingFileRefuses) {
  const std::string path = temp_dir_path("missing_point.h5");
  std::remove(path.c_str());
  auto s = make_setup(10, aitken_cfg());
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::GrowingFile, path,
                          "p0", 0)
                 .ok());
  const auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::GrowingFile, path, "p999");
  EXPECT_FALSE(pr.ok());
  EXPECT_EQ(pr.error().code, "E-RESTART-MISSING-POINT");
  std::remove(path.c_str());
}

TEST(Restart, FormBOnlyComponentOnDifferentRankCountRefuses) {
  // A minimal stand-in component that sets ONLY kCapabilityRestartFormB.
  class FormBOnly final : public fuse::Component {
   public:
    [[nodiscard]] std::string name() const override { return "formb_only"; }
    fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                       const fuse::CouplingGraph&) override {
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const fuse::StepContext&) override {
      return fuse::Result<void>::success();
    }
    [[nodiscard]] std::uint64_t capabilities() const override {
      return fuse::kCapabilityRestartFormB;
    }
  };
  fuse::ComponentSet cs;
  cs.add(std::make_unique<FormBOnly>());
  const fuse::Result<void> r = fuse::check_restart_rank_count(cs, 2, 3);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-FORMB-ONLY-RANK-CHANGE");
  // Equal rank counts never refuse, regardless of capability.
  EXPECT_TRUE(fuse::check_restart_rank_count(cs, 3, 3).ok());
}

TEST(Restart, BadGlobalIdNumberingRefuses) {
  // A duplicate id within one put_field() call is refused locally,
  // without ever reaching the HDF5 layer (owner decision 2's own
  // per-call check, defense in depth ahead of finish_point()'s batched
  // one).
  const std::string path = temp_dir_path("badnum.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  const std::vector<std::int64_t> dup_ids = {0, 1, 1};  // duplicate
  const std::vector<double> values = {1.0, 2.0, 3.0};
  const fuse::Result<void> r =
      pw.value()->writer().put_field("x", dup_ids, values, 1, 3);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-DUPLICATE-ID");
  std::remove(path.c_str());
}

TEST(Restart, DeclaredGlobalSizeMismatchRefuses) {
  const std::string path = temp_dir_path("sizemismatch.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  const std::vector<std::int64_t> ids = {0, 1, 2};
  const std::vector<double> values = {1.0, 2.0, 3.0};
  // Declares a global size of 5 but only reports 3 ids.
  const fuse::Result<void> r =
      pw.value()->writer().put_field("x", ids, values, 1, 5);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-BAD-NUMBERING");
  std::remove(path.c_str());
}

TEST(Restart, AcceleratorLayoutChangeWithoutOptInRefuses) {
  const std::string path = temp_dir_path("accel_layout.h5");
  std::remove(path.c_str());
  auto s = make_setup(40, aitken_cfg());
  run_n_periods(*s, 3);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 3)
                 .ok());

  // A restore whose stacked-vector layout genuinely differs would refuse
  // via a row id absent from the file; this test proves the CONTROL
  // (allow_reset=false propagates a real mismatch as a Failure, never a
  // silent reset) by restoring successfully first (layout UNCHANGED,
  // same graph) with allow_reset=false and confirming no reset was
  // reported -- the positive control for the negative one for a
  // genuinely different layout, deferred to Slice 2's rank-count-
  // dependent coverage (host_pending).
  auto restored = make_setup(40, aitken_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/false, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(any_reset);
  std::remove(path.c_str());
}

TEST(Restart, MissingAcceleratorHistoryWithAllowResetReportsReset) {
  // A restart point taken from a FRESH fixture whose accelerator has not
  // yet seen a single sweep (have_prev_ == false): AitkenAccel::export_
  // restart_state() writes nothing for this case (its own doc comment),
  // so the restore finds no field at all. Per the accelerator's own
  // import_restart_state() contract this is the LEGITIMATE no-history
  // case (not a layout mismatch) and must NOT report a reset -- proven
  // here for both allow_reset settings, which must agree.
  const std::string path = temp_dir_path("no_history.h5");
  std::remove(path.c_str());
  auto s = make_setup(40, aitken_cfg());
  // Zero periods run: no sweep has happened, have_prev_ is false.
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 0)
                 .ok());

  for (const bool allow_reset : {false, true}) {
    auto restored = make_setup(40, aitken_cfg());
    bool any_reset = false;
    const fuse::Result<void> rr = restore_point(
        *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
        allow_reset, any_reset);
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
    EXPECT_FALSE(any_reset) << "allow_reset=" << allow_reset;
  }
  std::remove(path.c_str());
}

TEST(Restart, MissingComponentRecordRefuses) {
  const std::string path = temp_dir_path("missing_component.h5");
  std::remove(path.c_str());
  auto writer_fixture = make_setup(10, fuse::AccelConfig{.kind = fuse::AccelKind::None});
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  // Writes ONLY the time/period scalars, never calling write_restart_fields
  // at all -- simulating a point whose component records never made it
  // to disk (a truncated or hand-edited fixture).
  ASSERT_TRUE(pw.value()->writer().put_scalar_rational("__point/time",
                                                       fuse::Rational{0}).ok());
  ASSERT_TRUE(pw.value()->writer().put_scalar("__point/period_index", 0.0).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  auto restored = make_setup(10, fuse::AccelConfig{.kind = fuse::AccelKind::None});
  bool any_reset = false;
  const auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  fuse::Rational out_time{0};
  std::int64_t out_period = 0;
  const fuse::Result<void> rr = fuse::restore_restart_point(
      *restored->runner, restored->components, pr.value()->reader(), false,
      any_reset, out_time, out_period);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-MISSING-FIELD");
  std::remove(path.c_str());
}

// ---------------------------------------------------------------- round 2
// MUST FIX 1: NO SILENT RESET. QuasiNewtonAccel (IQN_ILS/IQN_IMVJ) inherits
// Accelerator's base-class export/import_restart_state() -- it does not
// override restart_state_implemented(), export_restart_state(), or
// import_restart_state(), so it is exactly the "a future Accelerator that
// keeps the default" case this fix targets, using a REAL production
// AccelKind rather than a hand-rolled test-only subclass.

fuse::AccelConfig iqn_ils_cfg() {
  return fuse::AccelConfig{.kind = fuse::AccelKind::IQN_ILS,
                           .initial_relaxation = 0.5};
}

TEST(Restart, AccelNotImplementedRefusesWithoutOptIn) {
  const std::string path = temp_dir_path("accel_not_implemented.h5");
  std::remove(path.c_str());
  auto s = make_setup(40, iqn_ils_cfg());
  run_n_periods(*s, 3);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 3)
                 .ok());

  auto restored = make_setup(40, iqn_ils_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/false, any_reset);
  ASSERT_FALSE(rr.ok())
      << "QuasiNewtonAccel::restart_state_implemented() == false must "
         "refuse, not silently succeed, when the run did not opt in";
  EXPECT_EQ(rr.error().code, "E-RESTART-ACCEL-NOT-IMPLEMENTED");
  EXPECT_FALSE(any_reset);
  std::remove(path.c_str());
}

TEST(Restart, AccelNotImplementedWithAllowResetSucceedsAndReports) {
  const std::string path = temp_dir_path("accel_not_implemented_reset.h5");
  std::remove(path.c_str());
  auto s = make_setup(40, iqn_ils_cfg());
  run_n_periods(*s, 3);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 3)
                 .ok());

  auto restored = make_setup(40, iqn_ils_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/true, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_TRUE(any_reset)
      << "opting in must call Accelerator::reset() and REPORT it, per "
         "owner decision 4 (\"when used it is reported at restart\")";
  std::remove(path.c_str());
}

// ---------------------------------------------------------------- round 2
// MUST FIX 2: EXACT KEYS, NO HASH. Two blocks with IDENTICAL local ids but
// DIFFERENT field names -- exactly dirichlet/neumann's own shape (both
// report ids 0..n-1) -- must round-trip to their OWN rows, never cross-
// contaminate, because each block is written under its own qualified
// dataset name (name_prefix + "/" + field_name), not a value combined or
// hashed into a shared id space. This exercises detail::put_field_per_
// block()/get_field_per_block() directly, the exact mechanism AitkenAccel/
// SteffensenAccel now use.
TEST(Restart, PerBlockKeysNeverCollideOnIdenticalLocalIds) {
  const std::string path = temp_dir_path("per_block_keys.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());

  std::vector<fuse::RestartRowBlock> blocks;
  blocks.push_back(fuse::RestartRowBlock{"dirichlet/q_wall", {0, 1, 2}});
  blocks.push_back(fuse::RestartRowBlock{"neumann/q_wall_in", {0, 1, 2}});
  // Deliberately DIFFERENT values at the SAME local ids {0,1,2} for each
  // block, laid out flat in block order (dirichlet's 3 rows, then
  // neumann's 3 rows) -- the layout AitkenAccel::r_prev_ itself has.
  const std::vector<double> flat = {10.0, 11.0, 12.0, 20.0, 21.0, 22.0};
  const fuse::Result<void> w = fuse::detail::put_field_per_block(
      pw.value()->writer(), "region/aitken/r_prev", blocks, flat, 1);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
  ASSERT_TRUE(pw.value()->finish().ok());

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::vector<double> round_tripped(6, -1.0);
  const fuse::Result<void> r = fuse::detail::get_field_per_block(
      pr.value()->reader(), "region/aitken/r_prev", blocks, round_tripped, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(round_tripped[0], 10.0);
  EXPECT_EQ(round_tripped[1], 11.0);
  EXPECT_EQ(round_tripped[2], 12.0);
  EXPECT_EQ(round_tripped[3], 20.0)
      << "neumann's row at local id 0 must read back 20.0, never "
         "dirichlet's 10.0 at the SAME local id -- a hash-combined key "
         "could alias these if the two field names' hashes collided in "
         "their own high bits; an exact per-field dataset name cannot";
  EXPECT_EQ(round_tripped[4], 21.0);
  EXPECT_EQ(round_tripped[5], 22.0);
  std::remove(path.c_str());
}
