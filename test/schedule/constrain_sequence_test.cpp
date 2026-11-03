// A pure, deterministic derivation from the same data the Runner uses:
// fuse::constrain_sequence() (include/fuse/schedule/plan.hpp), the
// sibling of the pre-existing fuse::epoch_sequence() for the OTHER real
// base-communicator collective (OpKind::Constrain's conservation
// MPI_Allreduce, src/runtime/runner.cpp).
//
// This ties the static model to the REAL three-rank Brusselator
// participants layout (U{0}, V{1}, coupler{2} -- examples/brusselator/
// participants/brusselator_graph_tool.cpp's own constants and rank
// sets, the exact shape of the real row), built through fuse::lower()/fuse::schedule() alone -- both
// are pure functions of fuse::CouplingGraph, so this reaches past the
// E-RT-REGION ceiling that blocks Runner::initialize() on this layout at
// tier 0 (test/verification/brusselator_external_delivery_size_test.
// cpp's own ReachesPastExternalDeliveryOnBothRanks).
#include <gtest/gtest.h>

#include <string>

#include "fuse/backend/native_backend.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_graph.hpp"

namespace {

using fuse::FieldSpec;
using fuse::RankSet;
using fuse::WaveformIteration;

constexpr int kNx = 1, kNy = 1;
const fuse::Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;

TEST(ConstrainSequence, IdenticalAcrossAllThreeRealRankPlacements) {
  fuse::CouplingGraph g;
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/3);
  const fuse::MeshId mesh = g.add_mesh(
      {.name = "u",
       .source = "generated:quad:" + std::to_string(kNx) + "x" + std::to_string(kNy)});
  const fuse::RegionId region =
      g.add_region({.name = "u_all", .mesh = mesh, .dimension = 2});

  // The real row's own layout: U on rank 0, V on rank 1, the conservative remap coupler
  // on rank 2, brusselator_graph_tool.cpp's own --ranks-u 1 --ranks-v 1
  // --ranks-coupler 1.
  const auto handles = brusselator::add_window_coupling(
      g, region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi,
      /*coupler_ranks=*/RankSet{2, 1}, /*u_ranks=*/RankSet{0, 1},
      /*v_ranks=*/RankSet{1, 1});

  const auto u_out = g.find_field(handles.u_component, "u_out");
  const auto v_out = g.find_field(handles.v_component, "v_out");
  ASSERT_TRUE(u_out.has_value());
  ASSERT_TRUE(v_out.has_value());
  // ExternalDelivery/LowFive-bound, exactly as `--transport lowfive`
  // compiles it (brusselator_graph_tool.cpp's own write_lowfive_
  // declarations()): this is the channel shape that forces
  // `ranks_differ = false` in src/ir/rules_channel.cpp's own
  // lower_channel(), suppressing Transport/Epoch for the hop but NOT
  // Constrain -- the whole reason this row's own Constrain instruction
  // has no Epoch anywhere near it.
  const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "stub_u.h5/grid/u_out";
  const_cast<FieldSpec&>(g.field(*v_out)).hdf5_path = "stub_v.h5/grid/v_out";

  ASSERT_TRUE(backend.load(g.mesh(mesh), g).ok());
  const auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sched =
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  // Sanity: this row's own graph really does compile a Constrain
  // instruction with no accompanying Transport/Epoch -- a test that passed vacuously, on a plan
  // with no Constrain instruction at all, would prove nothing.
  int n_constrain = 0, n_transport = 0, n_epoch = 0;
  for (const auto& ins : sched.plan.instructions()) {
    if (ins.kind == fuse::OpKind::Constrain) ++n_constrain;
    if (ins.kind == fuse::OpKind::Transport) ++n_transport;
    if (ins.kind == fuse::OpKind::Epoch) ++n_epoch;
  }
  ASSERT_GT(n_constrain, 0) << "no Constrain instruction compiled at all; "
                               "the comparison below would be vacuous";
  EXPECT_EQ(n_transport, 0) << "lowfive_bound should have suppressed Transport";
  EXPECT_EQ(n_epoch, 0) << "lowfive_bound should have suppressed the Transport-"
                           "guarding Epoch (Constrain has no Epoch of its own)";

  // The actual proof: all three real rank placements see the identical
  // Constrain sequence -- no static schedule divergence anywhere in
  // this row's own compiled plan.
  const auto seq_u = fuse::constrain_sequence(sched.plan, g, /*rank=*/0);
  const auto seq_v = fuse::constrain_sequence(sched.plan, g, /*rank=*/1);
  const auto seq_coupler = fuse::constrain_sequence(sched.plan, g, /*rank=*/2);
  ASSERT_FALSE(seq_u.empty());
  EXPECT_EQ(seq_u, seq_v);
  EXPECT_EQ(seq_u, seq_coupler);
}

}  // namespace
