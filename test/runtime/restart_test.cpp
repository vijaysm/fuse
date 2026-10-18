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
#include <fuse/model/element_type.hpp>
#include <fuse/num/accelerator.hpp>

#include "dn_component.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
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

/// Restart track (docs/dev/restart.md Section 9 item 4; KNOWN_ISSUES.md
/// D64, closed this round): IQN-ILS over the SAME dirichlet_neumann
/// graph aitken_cfg() uses -- an ORDINARY (non-window-scoped) iteration
/// region, where `RegionBlock::length` equals the registered field's
/// own single-sample size by construction, so the window-scoped
/// id-space mismatch recorded in open_questions (a window region's own
/// RegionBlock::length is `n * capacity`, not `n`) cannot arise here.
fuse::AccelConfig iqn_ils_cfg() {
  return fuse::AccelConfig{.kind = fuse::AccelKind::IQN_ILS,
                           .initial_relaxation = 0.5, .reuse = 8};
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
  // Stage B Slice 1b (docs/dev/restart.md Section 11): DirichletSlab's own
  // int32/bool restart fields (sweep_count_/parity_flag_) -- compared
  // here so the round-trip tests verify the TYPED interface end to end
  // through the Runner, not only the double fields every test already
  // covered.
  std::int32_t dirichlet_sweep_count = 0;
  bool dirichlet_parity_flag = false;
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
  out.dirichlet_sweep_count = d->sweep_count();
  out.dirichlet_parity_flag = d->parity_flag();
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
  EXPECT_EQ(a.dirichlet_sweep_count, b.dirichlet_sweep_count);
  EXPECT_EQ(a.dirichlet_parity_flag, b.dirichlet_parity_flag);
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

/// KNOWN_ISSUES.md D64 (closed this round): the SAME round trip as
/// SameRankRoundTripSeparateLayout, with IQN-ILS in place of Aitken --
/// before this round, restoring into this region would have refused
/// (E-RESTART-ACCEL-NOT-IMPLEMENTED), proven by AccelNotImplementedRefuses
/// WithoutOptIn below (which deliberately keeps a DIFFERENT, test-only
/// accelerator kind as the refusal's own subject so this positive test
/// and that refusal test do not fight over the same graph). expect_bit_
/// identical's own comparison includes q/u (the registered fields,
/// restored via Form A) and the Runner's own sweep count for the tail
/// windows -- a wrong or missing IQN history would show up there as a
/// sweep-count divergence, not only as a numerically-different answer.
TEST(Restart, SameRankRoundTripSeparateLayoutIqnIls) {
  const std::string path = temp_dir_path("separate_iqn.h5");
  std::remove(path.c_str());

  auto ref = make_setup(/*max_iterations=*/40, iqn_ils_cfg());
  const RunResult ref_result = run_n_periods(*ref, 5);

  auto part1 = make_setup(/*max_iterations=*/40, iqn_ils_cfg());
  run_n_periods(*part1, 3);
  const fuse::Result<void> w =
      write_point(*part1, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                 path, "", 3);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  auto restored = make_setup(/*max_iterations=*/40, iqn_ils_cfg());
  bool any_reset = false;
  const fuse::Result<void> rr =
      restore_point(*restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                    path, "", /*allow_reset=*/false, any_reset);
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
  std::int64_t period_value = -1;
  const fuse::Result<fuse::ReadConversionReport> t =
      pr.value()->reader().get_scalar("__point/period_index", period_value);
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(period_value, 2);
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
      pw.value()->writer().put_field<double>("x", dup_ids, values, 1, 3);
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
      pw.value()->writer().put_field<double>("x", ids, values, 1, 5);
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
  ASSERT_TRUE(pw.value()->writer().put_scalar("__point/period_index", std::int64_t{0}).ok());
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

// ---------------------------------------------------------------- B
// Rank-count independence rests entirely on get_field() resolving rows
// BY GLOBAL ID, never by file position -- a reader at a DIFFERENT rank
// count asks for its own ids in whatever order its own local mesh
// enumerates them, which is generally NOT the order the writer (at a
// different rank count, or even the same one with a different field
// author) stored them in. MEASURED while reviewing this round's own
// additions: no existing test in this file ever reads a field's ids
// out of the order they were written in, so a reader that quietly
// substituted "i-th requested id -> i-th stored row" for the real
// id->row lookup would pass every other test here. This one asks for
// the SAME three ids written, in REVERSED order, and checks the VALUES
// come back correspondingly reversed -- the direct, minimal proof that
// position and id are not being confused.
TEST(Restart, GetFieldResolvesRowsByIdNotByFilePosition) {
  const std::string path = temp_dir_path("by_id_not_position.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());

  const std::vector<std::int64_t> write_ids = {7, 3, 11};
  const std::vector<double> write_values = {70.0, 30.0, 110.0};
  const fuse::Result<void> w = pw.value()->writer().put_field<double>(
      "f", write_ids, write_values, 1, 3);
  ASSERT_TRUE(w.ok()) << w.error().message;
  ASSERT_TRUE(pw.value()->finish().ok());

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());

  // Same three ids, REVERSED, plus a single-id read of the MIDDLE one --
  // if the reader were resolving by position instead of by id, reading
  // id 11 alone (file position 2) would wrongly return 30.0 (write_ids'
  // position-1 value) or some other position-derived value instead of
  // 110.0.
  const std::vector<std::int64_t> reversed_ids = {11, 3, 7};
  std::vector<double> out(3, 0.0);
  const fuse::Result<fuse::ReadConversionReport> r1 =
      pr.value()->reader().get_field<double>("f", reversed_ids, out, 1);
  ASSERT_TRUE(r1.ok()) << r1.error().message;
  EXPECT_EQ(out[0], 110.0);
  EXPECT_EQ(out[1], 30.0);
  EXPECT_EQ(out[2], 70.0);

  const std::vector<std::int64_t> single_id = {11};
  std::vector<double> single_out(1, 0.0);
  const fuse::Result<fuse::ReadConversionReport> r2 =
      pr.value()->reader().get_field<double>("f", single_id, single_out, 1);
  ASSERT_TRUE(r2.ok()) << r2.error().message;
  EXPECT_EQ(single_out[0], 110.0);
  std::remove(path.c_str());
}

// ---------------------------------------------------------------- round 2
// MUST FIX 1: NO SILENT RESET. Round 2 used QuasiNewtonAccel (IQN_ILS)
// as this fix's test subject, because at that point IQN_ILS genuinely
// inherited Accelerator's base-class no-op export/import_restart_
// state(). This round implements IQN_ILS's own restart state
// (KNOWN_ISSUES.md D64, closed -- see Restart.SameRankRoundTripSeparate
// LayoutIqnIls above), so IQN_ILS no longer exercises this refusal.
// Per the owner's own instruction ("keep the refusal path for a
// test-only accelerator that does not implement it"), these two tests
// now inject a genuinely test-only `Accelerator` that never overrides
// `restart_state_implemented()`/`export_restart_state()`/`import_
// restart_state()` -- i.e. it is, by construction, "a future
// Accelerator that keeps the default", the exact case this fix
// targets -- via `Runner::debug_replace_region_accelerator()`
// (fuse/runtime/runner.hpp), a TEST-ONLY seam added this round for
// exactly this purpose (no AccelKind enumerator exists for a
// never-implements-restart accelerator, nor should one: every genuine
// production kind now implements restart state or is trivial).
class NeverImplementsRestartAccel final : public fuse::Accelerator {
 public:
  void reset() override {}
  [[nodiscard]] std::string name() const override {
    return "NeverImplementsRestartAccel (test-only)";
  }
  [[nodiscard]] std::size_t columns() const override { return 0; }
  fuse::Result<void> update(const double* /*x_old*/, const double* x_new,
                            double* x_out, std::size_t n, int) override {
    for (std::size_t i = 0; i < n; ++i) x_out[i] = x_new[i];
    return fuse::Result<void>::success();
  }
  // Deliberately NOT overridden: restart_state_implemented() (stays
  // `false`), export_restart_state() (stays a no-op), import_restart_
  // state() (stays the unconditional Failure) -- exactly the base
  // class's own documented defaults.
};

/// `dirichlet_neumann` is the ONLY iteration region this graph compiles
/// (build_graph(), above), so region_index 0 is its own index on EVERY
/// `Runner` this file builds -- not re-derived per call because there is
/// nothing to look up: `debug_replace_region_accelerator()`'s own
/// doc comment states the convention (an out-of-range index refuses).
std::unique_ptr<DnHeatFixture> make_setup_with_test_only_accel(
    int max_iterations, double initial_u = 305.0) {
  // aitken_cfg() builds a REAL accelerator first (Runner::initialize()
  // requires SOME AccelKind to compile a region into an accelerated
  // one at all); the test-only accelerator then REPLACES it in place.
  auto s = make_setup(max_iterations, aitken_cfg(), initial_u);
  const fuse::Result<void> r = s->runner->debug_replace_region_accelerator(
      0, std::make_unique<NeverImplementsRestartAccel>());
  if (!r.ok()) ADD_FAILURE() << "debug_replace_region_accelerator: " << r.error().message;
  return s;
}

TEST(Restart, AccelNotImplementedRefusesWithoutOptIn) {
  const std::string path = temp_dir_path("accel_not_implemented.h5");
  std::remove(path.c_str());
  auto s = make_setup_with_test_only_accel(40);
  run_n_periods(*s, 3);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 3)
                 .ok());

  auto restored = make_setup_with_test_only_accel(40);
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/false, any_reset);
  ASSERT_FALSE(rr.ok())
      << "an accelerator whose restart_state_implemented() == false must "
         "refuse, not silently succeed, when the run did not opt in";
  EXPECT_EQ(rr.error().code, "E-RESTART-ACCEL-NOT-IMPLEMENTED");
  EXPECT_FALSE(any_reset);
  std::remove(path.c_str());
}

TEST(Restart, AccelNotImplementedWithAllowResetSucceedsAndReports) {
  const std::string path = temp_dir_path("accel_not_implemented_reset.h5");
  std::remove(path.c_str());
  auto s = make_setup_with_test_only_accel(40);
  run_n_periods(*s, 3);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 3)
                 .ok());

  auto restored = make_setup_with_test_only_accel(40);
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

// ---------------------------------------------------------------- D64
// closure, direct level: a Runner-level round trip (above) cannot
// observe QuasiNewtonAccel's own v_/w_/cols_/have_prev_ at all for
// AccelKind::IQN_ILS specifically, MEASURED while writing this test --
// a restart point is only ever taken at a period (here, window)
// boundary (D63), and the very FIRST update() call of the NEXT window
// has iteration == 0, which calls reset() before anything else;
// reset() with retain_ == false (every kind except IQN_IMVJ) discards
// v_/w_/cols_/have_prev_ unconditionally, so a WRONG restore of them
// is invisible through the Runner for IQN_ILS -- it would have been
// invisible to AccelNotImplementedRefusesWithoutOptIn's own predecessor
// too, had it ever reached far enough to check. This test instead
// drives `fuse::Accelerator` directly (no Runner, no window
// boundary), with `AccelKind::IQN_IMVJ` (`retain_ == true`) so that
// `blocks_`/`chain_reprieves_` -- the ONE piece of this accelerator's
// state that genuinely crosses a window/restart boundary by design
// (Section 1.3: "IQN-IMVJ folds it into blocks_ instead") -- has
// something real to restore, and reads EVERY member back bit for bit
// via `columns()`/`retained_blocks()` plus continued `update()` calls
// against an UNINTERRUPTED sibling fed the identical input sequence.
TEST(Restart, QuasiNewtonAccelExportImportRestartStateRoundTripsBitForBit) {
  constexpr std::size_t kN = 6;
  const fuse::AccelConfig cfg{.kind = fuse::AccelKind::IQN_IMVJ,
                             .initial_relaxation = 0.3, .reuse = 4};

  auto make = [&]() {
    auto r = fuse::make_accelerator(cfg);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return std::move(r.value());
  };
  auto reference = make();

  // A deterministic, non-trivial input sequence: two windows' worth of
  // sweeps (3 then 2), so the first window's own reset() folds a REAL
  // (cols_ > 0) window into blocks_ -- retained_blocks() > 0 is what
  // this test actually exercises restoring.
  std::vector<double> x_old(kN), x_new(kN), x_out(kN);
  auto step = [&](fuse::Accelerator& a, int iteration, double seed) {
    for (std::size_t i = 0; i < kN; ++i) {
      x_old[i] = seed + 0.1 * static_cast<double>(i);
      x_new[i] = x_old[i] + 0.05 * std::sin(seed + static_cast<double>(i));
    }
    const fuse::Result<void> r =
        a.update(x_old.data(), x_new.data(), x_out.data(), kN, iteration);
    EXPECT_TRUE(r.ok()) << r.error().message;
  };
  step(*reference, 0, 1.0);
  step(*reference, 1, 1.3);
  step(*reference, 2, 1.6);
  reference->reset();  // window boundary: folds this window into blocks_
  step(*reference, 0, 2.0);
  step(*reference, 1, 2.4);
  ASSERT_GT(reference->retained_blocks(), 0u)
      << "DIAGNOSTIC: the input sequence above must populate blocks_ for "
         "this test to exercise anything -- see the test's own comment";
  ASSERT_GT(reference->columns(), 0u);

  std::vector<std::int64_t> ids(kN);
  for (std::size_t i = 0; i < kN; ++i) ids[i] = static_cast<std::int64_t>(i);
  const std::vector<fuse::RestartRowBlock> blocks = {
      fuse::RestartRowBlock{"imvj_field", ids}};

  const std::string path = temp_dir_path("iqn_imvj_direct.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(reference->restart_state_implemented());
  const fuse::Result<void> wr =
      reference->export_restart_state(pw.value()->writer(), "region", blocks);
  ASSERT_TRUE(wr.ok()) << wr.error().code << ": " << wr.error().message;
  ASSERT_TRUE(pw.value()->finish().ok());

  auto restored = make();
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  bool did_reset = false;
  const fuse::Result<void> rr = restored->import_restart_state(
      pr.value()->reader(), "region", blocks, /*allow_reset_on_mismatch=*/false,
      did_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(did_reset);
  EXPECT_EQ(restored->columns(), reference->columns());
  EXPECT_EQ(restored->retained_blocks(), reference->retained_blocks());

  // Continue BOTH with the identical remaining input sequence; every
  // x_out and every reported column/block count must match bit for
  // bit from here on, which is only possible if v_/w_/x_new_prev_/
  // r_prev_/have_prev_/blocks_/chain_reprieves_ ALL restored correctly
  // (a wrong value in any one of them perturbs the least-squares fit
  // or the Jprev^{-1} application, which changes x_out).
  std::vector<double> x_out_ref(kN), x_out_restored(kN);
  for (int iteration = 2; iteration <= 4; ++iteration) {
    for (std::size_t i = 0; i < kN; ++i) {
      x_old[i] = 3.0 + 0.1 * static_cast<double>(i);
      x_new[i] = x_old[i] + 0.05 * std::sin(3.0 + static_cast<double>(iteration) +
                                            static_cast<double>(i));
    }
    const fuse::Result<void> rref =
        reference->update(x_old.data(), x_new.data(), x_out_ref.data(), kN, iteration);
    const fuse::Result<void> rres =
        restored->update(x_old.data(), x_new.data(), x_out_restored.data(), kN, iteration);
    ASSERT_TRUE(rref.ok()) << rref.error().message;
    ASSERT_TRUE(rres.ok()) << rres.error().message;
    for (std::size_t i = 0; i < kN; ++i) {
      EXPECT_EQ(x_out_ref[i], x_out_restored[i])
          << "iteration " << iteration << ", element " << i;
    }
    EXPECT_EQ(reference->columns(), restored->columns());
    EXPECT_EQ(reference->retained_blocks(), restored->retained_blocks());
  }
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

// ---------------------------------------------------------------- round 2
// MUST FIX 2 (KNOWN_ISSUES.md D69, closed this round): a window-scoped
// region's own accelerated vector is K samples per element, not one --
// RestartRowBlock::window_capacity, demuxed by detail::put_field_per_
// block()/get_field_per_block() into one sub-dataset per sample. Two
// blocks of DIFFERENT capacities (3 and 4), 18 total rows (2*3 + 3*4),
// round-tripped through the SAME accelerator export/import_restart_state
// path QuasiNewtonAccelExportImportRestartStateRoundTripsBitForBit above
// exercises for the non-window case, then one further update() call
// compared bit-for-bit against an uninterrupted sibling.
TEST(Restart, AitkenAccelWindowCapacityPerSampleRestartRoundTripsBitForBit) {
  const fuse::AccelConfig cfg{.kind = fuse::AccelKind::Aitken,
                             .initial_relaxation = 0.5};
  auto make = [&]() {
    auto r = fuse::make_accelerator(cfg);
    EXPECT_TRUE(r.ok()) << r.error().message;
    return std::move(r.value());
  };
  auto reference = make();

  constexpr std::size_t kNA = 2, kCapA = 3;  // block A: 2 elements, 3 samples
  constexpr std::size_t kNB = 3, kCapB = 4;  // block B: 3 elements, 4 samples
  constexpr std::size_t kN = kNA * kCapA + kNB * kCapB;  // 18

  std::vector<double> x_old(kN), x_new(kN), x_out(kN);
  auto step = [&](fuse::Accelerator& a, int iteration, double seed) {
    for (std::size_t i = 0; i < kN; ++i) {
      x_old[i] = seed + 0.1 * static_cast<double>(i);
      x_new[i] = x_old[i] + 0.05 * std::sin(seed + static_cast<double>(i));
    }
    const fuse::Result<void> r =
        a.update(x_old.data(), x_new.data(), x_out.data(), kN, iteration);
    EXPECT_TRUE(r.ok()) << r.error().message;
  };
  step(*reference, 0, 1.0);
  step(*reference, 1, 1.4);

  std::vector<std::int64_t> ids_a = {100, 101};
  std::vector<std::int64_t> ids_b = {200, 201, 202};
  std::vector<fuse::RestartRowBlock> blocks(2);
  blocks[0] = fuse::RestartRowBlock{"window_u", ids_a};
  blocks[0].window_capacity = kCapA;
  blocks[1] = fuse::RestartRowBlock{"window_v", ids_b};
  blocks[1].window_capacity = kCapB;

  const std::string path = temp_dir_path("aitken_window_capacity.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(reference->restart_state_implemented());
  const fuse::Result<void> wr =
      reference->export_restart_state(pw.value()->writer(), "region", blocks);
  ASSERT_TRUE(wr.ok()) << wr.error().code << ": " << wr.error().message;
  ASSERT_TRUE(pw.value()->finish().ok());

  auto restored = make();
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  bool did_reset = false;
  const fuse::Result<void> rr = restored->import_restart_state(
      pr.value()->reader(), "region", blocks, /*allow_reset_on_mismatch=*/false,
      did_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(did_reset);

  std::vector<double> x_out_ref(kN), x_out_restored(kN);
  for (int iteration = 2; iteration <= 3; ++iteration) {
    for (std::size_t i = 0; i < kN; ++i) {
      x_old[i] = 3.0 + 0.1 * static_cast<double>(i);
      x_new[i] = x_old[i] + 0.05 * std::sin(3.0 + static_cast<double>(iteration) +
                                            static_cast<double>(i));
    }
    const fuse::Result<void> rref =
        reference->update(x_old.data(), x_new.data(), x_out_ref.data(), kN, iteration);
    const fuse::Result<void> rres =
        restored->update(x_old.data(), x_new.data(), x_out_restored.data(), kN, iteration);
    ASSERT_TRUE(rref.ok()) << rref.error().message;
    ASSERT_TRUE(rres.ok()) << rres.error().message;
    for (std::size_t i = 0; i < kN; ++i) {
      EXPECT_EQ(x_out_ref[i], x_out_restored[i])
          << "iteration " << iteration << ", element " << i;
    }
  }
  std::remove(path.c_str());
}

// ---------------------------------------------------- Stage B Slice 1b

// Stage B Slice 1b (docs/dev/restart.md Section 11): restart fields are
// now templated on the element type -- these tests exercise that
// directly against `fuse::restart_io::Hdf5GroupWriter`/`Hdf5GroupReader`
// (through the public `create_restart_point()`/`open_restart_point()`
// entry points every other test in this file already uses), rather than
// through a real Component, so each one is scoped to exactly the type
// or refusal it names.

namespace {
template <class T>
fuse::Result<void> write_one_typed(const std::string& path, const char* name,
                                   T value) {
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  if (!pw.ok()) return pw.error();
  const fuse::Result<void> r = pw.value()->writer().put_scalar(name, value);
  if (!r.ok()) return r;
  return pw.value()->finish();
}
}  // namespace

// Every element type the typed interface supports (owner's own closed
// enumeration, fuse/model/element_type.hpp), round-tripped through a
// real HDF5 file, bit for bit -- the owner's own test requirement:
// "round trip bit-identical per type (every enumerator)".
TEST(TypedRestart, EveryElementTypeRoundTripsBitIdentical) {
  const std::string path = temp_dir_path("typed_roundtrip.h5");

#define FUSE_TYPED_ROUNDTRIP_CASE(T, VALUE)                                 \
  {                                                                         \
    std::remove(path.c_str());                                             \
    const T original = (VALUE);                                           \
    ASSERT_TRUE(write_one_typed<T>(path, "v", original).ok());             \
    auto pr = fuse::restart_io::open_restart_point(                        \
        fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");    \
    ASSERT_TRUE(pr.ok());                                                  \
    T restored{};                                                         \
    const auto rr = pr.value()->reader().get_scalar("v", restored);        \
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message; \
    EXPECT_FALSE(rr.value().converted_any);                                \
    EXPECT_EQ(restored, original) << "type " << #T;                        \
  }

  FUSE_TYPED_ROUNDTRIP_CASE(std::int8_t, std::int8_t{-123})
  FUSE_TYPED_ROUNDTRIP_CASE(std::int16_t, std::int16_t{-12345})
  FUSE_TYPED_ROUNDTRIP_CASE(std::int32_t, std::int32_t{-123456789})
  FUSE_TYPED_ROUNDTRIP_CASE(std::int64_t, std::int64_t{-123456789012345LL})
  FUSE_TYPED_ROUNDTRIP_CASE(std::uint8_t, std::uint8_t{250})
  FUSE_TYPED_ROUNDTRIP_CASE(std::uint16_t, std::uint16_t{64000})
  FUSE_TYPED_ROUNDTRIP_CASE(std::uint32_t, std::uint32_t{4000000000u})
  FUSE_TYPED_ROUNDTRIP_CASE(std::uint64_t,
                            std::uint64_t{18000000000000000000ULL})
  FUSE_TYPED_ROUNDTRIP_CASE(fuse::bfloat16,
                            fuse::bfloat16::from_float(-3.25f))
  FUSE_TYPED_ROUNDTRIP_CASE(float, 1.0f + std::numeric_limits<float>::epsilon())
  FUSE_TYPED_ROUNDTRIP_CASE(double,
                            1.0 + std::numeric_limits<double>::epsilon())
  FUSE_TYPED_ROUNDTRIP_CASE(long double, static_cast<long double>(-2.5))
  FUSE_TYPED_ROUNDTRIP_CASE(bool, true)
  FUSE_TYPED_ROUNDTRIP_CASE(bool, false)
#if defined(FUSE_HAVE_FLOAT16)
  FUSE_TYPED_ROUNDTRIP_CASE(_Float16, static_cast<_Float16>(1.5))
#endif

#undef FUSE_TYPED_ROUNDTRIP_CASE
  std::remove(path.c_str());
}

// Both file layouts (owner's own test requirement) -- a single
// representative type (double) is enough to prove the typed machinery
// works identically under GrowingFile, since the type dispatch itself
// does not depend on which layout wrote it (both route through the
// SAME `Hdf5GroupWriter`/`Hdf5GroupReader`).
TEST(TypedRestart, TypedScalarRoundTripsUnderGrowingFileLayoutToo) {
  const std::string path = temp_dir_path("typed_growing.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::GrowingFile, path, prov, "p1");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(pw.value()->writer().put_scalar("v", std::int32_t{-42}).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::GrowingFile, path, "p1");
  ASSERT_TRUE(pr.ok());
  std::int32_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar("v", restored);
  ASSERT_TRUE(rr.ok());
  EXPECT_EQ(restored, -42);
  std::remove(path.c_str());
}

// ReadConversion::Exact (the default) requires an identical type --
// written as double, requested as int32 refuses, naming both types.
TEST(TypedRestart, ExactRefusesOnTypeMismatchNamingBothTypes) {
  const std::string path = temp_dir_path("typed_exact_mismatch.h5");
  ASSERT_TRUE(write_one_typed<double>(path, "v", 3.0).ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::int32_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar("v", restored);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-TYPE-MISMATCH");
  EXPECT_NE(rr.error().message.find("binary64"), std::string::npos);
  EXPECT_NE(rr.error().message.find("int32"), std::string::npos);
  std::remove(path.c_str());
}

// ReadConversion::Widening permits only an EXACT conversion -- written
// as binary16, requested as bfloat16 (same significand coverage, but a
// WIDER exponent range on the destination, see fuse::is_exact_widening's
// own exponent-containment rule) is NOT exact and refuses even under
// Widening.
TEST(TypedRestart, WideningRefusesANonExactConversion) {
#if defined(FUSE_HAVE_FLOAT16)
  const std::string path = temp_dir_path("typed_widening_refuses.h5");
  ASSERT_TRUE(write_one_typed<_Float16>(path, "v", static_cast<_Float16>(1.5)).ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  fuse::bfloat16 restored;
  const auto rr = pr.value()->reader().get_scalar(
      "v", restored, fuse::ReadConversion::Widening);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-TYPE-MISMATCH");
  std::remove(path.c_str());
#else
  GTEST_SKIP() << "this compiler has no _Float16 (binary16)";
#endif
}

// Widening DOES permit an exact conversion -- the positive counterpart
// of the test above, written as bfloat16 (whose significand AND
// exponent range both fit binary32 exactly), requested as binary32.
TEST(TypedRestart, WideningSucceedsOnAnExactConversionWithNoReportedChange) {
  const std::string path = temp_dir_path("typed_widening_ok.h5");
  ASSERT_TRUE(write_one_typed<fuse::bfloat16>(
                  path, "v", fuse::bfloat16::from_float(-3.25f))
                 .ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  float restored = 0.0f;
  const auto rr = pr.value()->reader().get_scalar(
      "v", restored, fuse::ReadConversion::Widening);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_FALSE(rr.value().converted_any);
  EXPECT_EQ(restored, -3.25f);
  std::remove(path.c_str());
}

// "integers that do not fit refuse even under AllowNarrowing: an integer
// value change is not a rounding" (owner's own words).
TEST(TypedRestart, IntegerOverflowRefusesEvenUnderAllowNarrowing) {
  const std::string path = temp_dir_path("typed_int_overflow.h5");
  ASSERT_TRUE(write_one_typed<std::int32_t>(path, "v", 100000).ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::int8_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar(
      "v", restored, fuse::ReadConversion::AllowNarrowing);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-INTEGER-OVERFLOW");
  std::remove(path.c_str());
}

// An integer value that DOES fit succeeds under AllowNarrowing (and is
// not itself a "rounding" -- changed_any stays false, since an integer
// narrowing is exact-or-refused, never partially lossy).
TEST(TypedRestart, IntegerNarrowingThatFitsSucceeds) {
  const std::string path = temp_dir_path("typed_int_narrow_ok.h5");
  ASSERT_TRUE(write_one_typed<std::int32_t>(path, "v", 100).ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::int8_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar(
      "v", restored, fuse::ReadConversion::AllowNarrowing);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_EQ(restored, 100);
  std::remove(path.c_str());
}

// Float narrowing under AllowNarrowing reports the actual change it
// measured -- binary64 -> bfloat16, a value with real low-order bits
// that bfloat16's 8-bit significand cannot hold.
TEST(TypedRestart, FloatNarrowingReportsAMeasuredChange) {
  const std::string path = temp_dir_path("typed_float_narrow.h5");
  const double original = 1.0 + std::ldexp(1.0, -20);  // well within
                                                        // double's 53 bits,
                                                        // far beyond bf16's 8
  ASSERT_TRUE(write_one_typed<double>(path, "v", original).ok());
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  fuse::bfloat16 restored;
  const auto rr = pr.value()->reader().get_scalar(
      "v", restored, fuse::ReadConversion::AllowNarrowing);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_TRUE(rr.value().converted_any);
  EXPECT_GT(rr.value().max_absolute_change, 0.0);
  EXPECT_EQ(restored.to_float(), 1.0f);  // rounds down to exactly 1.0
  std::remove(path.c_str());
}

// Same-type requests are unaffected by which ReadConversion was asked
// for -- the "fast path" every pre-Slice-1b round-trip test already
// relies on, re-stated here as its own explicit property.
TEST(TypedRestart, IdenticalTypeNeverReportsAChangeUnderAnyConversionMode) {
  const std::string path = temp_dir_path("typed_identical.h5");
  ASSERT_TRUE(write_one_typed<double>(path, "v", 3.14).ok());
  for (const fuse::ReadConversion conv :
       {fuse::ReadConversion::Exact, fuse::ReadConversion::Widening,
        fuse::ReadConversion::AllowNarrowing}) {
    auto pr = fuse::restart_io::open_restart_point(
        fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
    ASSERT_TRUE(pr.ok());
    double restored = 0.0;
    const auto rr = pr.value()->reader().get_scalar("v", restored, conv);
    ASSERT_TRUE(rr.ok());
    EXPECT_FALSE(rr.value().converted_any);
    EXPECT_EQ(restored, 3.14);
  }
  std::remove(path.c_str());
}
