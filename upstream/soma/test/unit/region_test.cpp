#include <gtest/gtest.h>
#include <soma/graph/field_spec.hpp>
#include <soma/graph/region.hpp>

namespace u = soma::units;
using soma::Conservation;
using soma::Discretization;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshId;
using soma::Region;
using soma::RegionId;

TEST(Region, DefaultsToWholeMesh) {
  const Region r{.name = "fluid_all", .mesh = MeshId{0}};
  EXPECT_TRUE(r.entity_set.empty());
  EXPECT_EQ(r.dimension, -1);
}

TEST(Region, NamedEntitySetIsASurface) {
  const Region r{.name = "wall", .mesh = MeshId{0},
                 .entity_set = "sideset_3", .dimension = 2};
  EXPECT_EQ(r.entity_set, "sideset_3");
  EXPECT_EQ(r.dimension, 2);
}

TEST(FieldSpec, DefaultsAreFvScalarProvidesNone) {
  const FieldSpec f{.name = "T", .region = RegionId{0}};
  EXPECT_EQ(f.disc, Discretization::FV);
  EXPECT_EQ(f.components, 1);
  EXPECT_EQ(f.conservation, Conservation::None);
  EXPECT_EQ(f.intent, Intent::Provides);
  EXPECT_TRUE(f.units.is_dimensionless());
}

TEST(FieldSpec, CarriesUnitsAndConservation) {
  const FieldSpec q{.name = "heat_flux",
                    .region = RegionId{1},
                    .units = u::kHeatFluxDensity,
                    .conservation = Conservation::Extensive,
                    .intent = Intent::Provides};
  EXPECT_EQ(q.units, u::kHeatFluxDensity);
  EXPECT_EQ(q.conservation, Conservation::Extensive);
}

TEST(FieldSpec, SameShapeComparesComponents) {
  const FieldSpec a{.name = "v", .region = RegionId{0}, .components = 3};
  const FieldSpec b{.name = "w", .region = RegionId{1}, .components = 3};
  const FieldSpec c{.name = "s", .region = RegionId{1}, .components = 1};
  EXPECT_TRUE(same_shape(a, b));
  EXPECT_FALSE(same_shape(a, c));
}
