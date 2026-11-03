// build_lowfive_bindings() needs no HDF5, no LowFive and no MPI -- it is pure
// CouplingGraph traversal, same
// tier as everything in test/transport/lowfive_test.cpp, just split into
// its own binary because it needs fuse/model/coupling_graph.hpp and that
// file's own tests deliberately do not (see this file's header comment,
// fuse/transport/lowfive_graph.hpp).

#include <fuse/model/coupling_graph.hpp>
#include <fuse/transport/lowfive_graph.hpp>

#include <gtest/gtest.h>

using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
namespace u = fuse::units;

namespace {

struct Fixture {
  CouplingGraph g;
  fuse::ComponentId producer, consumer;
  fuse::FieldId      out_field, in_field;

  Fixture() {
    const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
    const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                       .entity_set = "w", .dimension = 2});
    producer = g.add_component(ComponentDecl{.name = "producer",
                                             .ranks = fuse::RankSet{0, 1},
                                             .interval = Rational{1}});
    consumer = g.add_component(ComponentDecl{.name = "consumer",
                                             .ranks = fuse::RankSet{1, 1},
                                             .interval = Rational{1}});
    out_field = g.add_field(producer, FieldSpec{.name = "q", .region = r,
                                                .units = u::kHeatFluxDensity,
                                                .conservation = Conservation::Extensive,
                                                .intent = Intent::Provides});
    in_field = g.add_field(consumer, FieldSpec{.name = "q_in", .region = r,
                                               .units = u::kHeatFluxDensity,
                                               .conservation = Conservation::Extensive,
                                               .intent = Intent::Requires});
  }
};

}  // namespace

TEST(BuildLowFiveBindings, EmptyGraphProducesAnEmptyTable) {
  Fixture f;
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value().empty());
}

TEST(BuildLowFiveBindings, OneFieldWithHdf5PathBecomesOneQualifiedBinding) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.out_field)).hdf5_path =
      "producer_out.h5/grid/q";
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(r.value().size(), 1u);
  const auto* b = r.value().find_by_field("producer.q");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->hdf5_path, "producer_out.h5/grid/q");
}

// The qualification (component.field, not the bare field name) is what
// lets two different components use the same local field name without
// build_lowfive_bindings() mistaking that for a collision.
TEST(BuildLowFiveBindings, SameLocalFieldNameOnTwoComponentsDoesNotCollide) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.out_field)).name = "shared_name";
  const_cast<FieldSpec&>(f.g.field(f.out_field)).hdf5_path =
      "producer_out.h5/grid/a";
  const_cast<FieldSpec&>(f.g.field(f.in_field)).name = "shared_name";
  const_cast<FieldSpec&>(f.g.field(f.in_field)).hdf5_path =
      "consumer_in.h5/grid/b";
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value().size(), 2u);
  EXPECT_NE(r.value().find_by_field("producer.shared_name"), nullptr);
  EXPECT_NE(r.value().find_by_field("consumer.shared_name"), nullptr);
}

TEST(BuildLowFiveBindings, FieldsWithNoHdf5PathAreSkipped) {
  Fixture f;
  // Neither field sets hdf5_path -- the ordinary case for every graph
  // that predates hdf5_path, and still the common case for a channel that
  // moves its bytes through an ordinary Transport.
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value().empty());
}

TEST(BuildLowFiveBindings, TwoFieldsSharingARealPathIsRefused) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.out_field)).hdf5_path = "shared.h5/grid/q";
  const_cast<FieldSpec&>(f.g.field(f.in_field)).hdf5_path = "shared.h5/grid/q";
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-DUPPATH");
}

TEST(BuildLowFiveBindings, AMalformedHdf5PathIsRefused) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.out_field)).hdf5_path = "not_an_hdf5_path";
  const auto r = fuse::build_lowfive_bindings(f.g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
}
