// FUSE_ELEMENT_DD and FUSE_ELEMENT_QD, and the gradient_size of a registered field, through
// the plain-C participant table: a C
// or Fortran component that keeps double-double or quad-double arrays registers them as
// arrays of pairs or quadruples of binary64 values, and the coupler does its sums and remaps
// in that type, so that the low limbs arrive. The graph counts binary64 values per entity (a
// double-double value is two, a quad-double value four); the view counts values of its
// element type. A component with a forward-mode gradient laid out in its trailing
// components says so with gradient_size, and a minimum or maximum channel is refused.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"

namespace {

const fuse_participant_abi_v1* Table() { return fuse_coupler_abi(FUSE_ABI_VERSION); }

/// Two single-rank, lockstep components over matching four-cell meshes, "upstream"
/// providing "downstream" one field of `components` binary64 values per cell: the weights
/// of the transfer are exactly one, so what arrives is what was written.
std::vector<std::byte> GraphBytes(int components, std::optional<fuse::AccumOp> accum = std::nullopt) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:4"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:4"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto down = g.add_component(fuse::ComponentDecl{
      .name = "downstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto out_f = g.add_field(
      up, fuse::FieldSpec{.name = "out", .region = r1, .components = components,
                          .units = fuse::units::kKelvin,
                          .conservation = fuse::Conservation::None,
                          .intent = fuse::Intent::Provides});
  const auto in_f = g.add_field(
      down, fuse::FieldSpec{.name = "in", .region = r2, .components = components,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", out_f, in_f);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  if (accum.has_value()) g.channel(ch).accumulate(*accum);
  return fuse::serialize(g);
}

fuse_ctx* CreateAndLoad(const fuse_participant_abi_v1* t, const std::vector<std::byte>& bytes) {
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(co.ctx, &src, nullptr), FUSE_OK);
  return co.ctx;
}

fuse_status Register(const fuse_participant_abi_v1* t, fuse_ctx* ctx, const char* component,
                     const char* field, int element_type, int components, void* data,
                     int gradient_size = 0, std::uint32_t struct_size = sizeof(fuse_field_view)) {
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = component;
  fname.field = field;
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view fview{.struct_size = struct_size};
  fview.handle = fref.handle;
  fview.element_type = element_type;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = components;
  fview.count = 4;
  fview.owned_count = 4;
  fview.stride = components;
  fview.component_stride = 1;
  fview.data = data;
  fview.gradient_size = gradient_size;
  return t->register_field(ctx, &fview, nullptr);
}

std::string LastErrorCode(const fuse_participant_abi_v1* t, fuse_ctx* ctx) {
  fuse_error err{.struct_size = sizeof(err)};
  EXPECT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  return err.code != nullptr ? err.code : "";
}

fuse_status Period(const fuse_participant_abi_v1* t, fuse_ctx* ctx, const std::function<void()>& write) {
  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  fuse_status st = t->begin_step(ctx, &up_req, &rep);
  if (st != FUSE_OK) return st;
  write();
  st = t->end_step(ctx, &up_req, &rep);
  if (st != FUSE_OK) return st;
  st = t->begin_step(ctx, &down_req, &rep);
  if (st != FUSE_OK) return st;
  return t->end_step(ctx, &down_req, &rep);
}

/// A value of `limbs` binary64 limbs, each at most 2^-54 times the one before (so that the
/// limbs do not overlap and any pair or quadruple is a normalised value): not representable
/// in fewer, so a transfer that drops a limb is visible.
std::vector<double> Value(int cell, int limbs) {
  std::vector<double> v;
  double x = 1.0 + 0.173 * cell;
  for (int l = 0; l < limbs; ++l) {
    v.push_back(x);
    x = (((cell + l) % 2 == 0) ? 1.0 : -1.0) * x * std::ldexp(0.7 + 0.05 * l, -54);
  }
  return v;
}

/// One side of a transfer: its element type, the binary64 values it stores per cell, and
/// the components its view declares (values of its own element type).
struct Side {
  int element_type;
  int doubles_per_cell;
  int view_components;
};

/// One period of the two-component graph (`graph_components` binary64 values per cell);
/// `up` writes `Value(cell, up.doubles_per_cell)` and the downstream array is returned.
std::vector<double> Transfer(Side up_side, Side down_side, int graph_components, fuse_status* init_status,
                             std::string* init_code) {
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(graph_components));
  std::vector<double> up(static_cast<std::size_t>(4 * up_side.doubles_per_cell), 0.0);
  std::vector<double> down(static_cast<std::size_t>(4 * down_side.doubles_per_cell), 0.0);
  EXPECT_EQ(Register(t, ctx, "upstream", "out", up_side.element_type, up_side.view_components, up.data()),
            FUSE_OK);
  EXPECT_EQ(Register(t, ctx, "downstream", "in", down_side.element_type, down_side.view_components,
                     down.data()),
            FUSE_OK);
  *init_status = t->initialize(ctx, nullptr, nullptr);
  if (*init_status != FUSE_OK) {
    *init_code = LastErrorCode(t, ctx);
    EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
    return {};
  }
  EXPECT_EQ(Period(t, ctx, [&] {
              for (int c = 0; c < 4; ++c) {
                const auto v = Value(c, up_side.doubles_per_cell);
                for (int l = 0; l < up_side.doubles_per_cell; ++l) {
                  up[static_cast<std::size_t>(c * up_side.doubles_per_cell + l)] = v[static_cast<std::size_t>(l)];
                }
              }
            }),
            FUSE_OK);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  return down;
}

}  // namespace

TEST(AbiArithmeticElement, ADoubleDoubleArrayArrivesWithItsLowLimbs) {
  fuse_status st;
  std::string code;
  const auto got = Transfer({FUSE_ELEMENT_DD, 2, 1}, {FUSE_ELEMENT_DD, 2, 1}, 2, &st, &code);
  ASSERT_EQ(st, FUSE_OK) << code;
  ASSERT_EQ(got.size(), 8u);
  for (int c = 0; c < 4; ++c) {
    const auto want = Value(c, 2);
    EXPECT_EQ(got[static_cast<std::size_t>(2 * c)], want[0]) << "cell " << c;
    EXPECT_EQ(got[static_cast<std::size_t>(2 * c + 1)], want[1]) << "cell " << c << " (the low limb)";
    EXPECT_NE(want[1], 0.0);
  }
}

TEST(AbiArithmeticElement, AQuadDoubleArrayArrivesWithAllFourLimbs) {
  fuse_status st;
  std::string code;
  const auto got = Transfer({FUSE_ELEMENT_QD, 4, 1}, {FUSE_ELEMENT_QD, 4, 1}, 4, &st, &code);
  ASSERT_EQ(st, FUSE_OK) << code;
  ASSERT_EQ(got.size(), 16u);
  for (int c = 0; c < 4; ++c) {
    const auto want = Value(c, 4);
    for (int l = 0; l < 4; ++l) {
      EXPECT_EQ(got[static_cast<std::size_t>(4 * c + l)], want[static_cast<std::size_t>(l)])
          << "cell " << c << " limb " << l;
    }
  }
}

TEST(AbiArithmeticElement, SeveralValuesPerCellAreCountedInValuesNotLimbs) {
  // two double-double values per cell: the graph says 4 binary64 values, the view says 2
  fuse_status st;
  std::string code;
  const auto got = Transfer({FUSE_ELEMENT_DD, 4, 2}, {FUSE_ELEMENT_DD, 4, 2}, 4, &st, &code);
  ASSERT_EQ(st, FUSE_OK) << code;
  for (int c = 0; c < 4; ++c) {
    const auto want = Value(c, 4);
    for (int l = 0; l < 4; ++l) {
      EXPECT_EQ(got[static_cast<std::size_t>(4 * c + l)], want[static_cast<std::size_t>(l)])
          << "cell " << c << " limb " << l;
    }
  }
}

TEST(AbiArithmeticElement, ViewFieldReportsTheTypeAndCountsValues) {
  const auto* t = Table();
  for (const auto& [et, limbs] : {std::pair{FUSE_ELEMENT_DD, 2}, std::pair{FUSE_ELEMENT_QD, 4}}) {
    fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(limbs));
    std::vector<double> up(static_cast<std::size_t>(4 * limbs), 0.0);
    std::vector<double> down(static_cast<std::size_t>(4 * limbs), 0.0);
    ASSERT_EQ(Register(t, ctx, "upstream", "out", et, 1, up.data()), FUSE_OK);
    ASSERT_EQ(Register(t, ctx, "downstream", "in", et, 1, down.data()), FUSE_OK);
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = "upstream";
    fname.field = "out";
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    EXPECT_EQ(fref.components, limbs) << "the graph counts binary64 values";
    fuse_field_view view{.struct_size = sizeof(view)};
    ASSERT_EQ(t->view_field(ctx, &fref, &view), FUSE_OK);
    EXPECT_EQ(view.element_type, et);
    EXPECT_EQ(view.components, 1) << "the view counts values of its element type";
    EXPECT_EQ(view.count, 4u);
    EXPECT_EQ(view.data, static_cast<void*>(up.data()));
    EXPECT_EQ(view.gradient_size, 0);
    EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  }
}

TEST(AbiArithmeticElement, ComponentsThatDoNotCountTheLimbsAreRefusedAtInitialize) {
  // the graph has 2 binary64 values per cell; a view of 2 double-double values is 4
  fuse_status st;
  std::string code;
  const auto got = Transfer({FUSE_ELEMENT_DD, 2, 2}, {FUSE_ELEMENT_DD, 2, 2}, 2, &st, &code);
  (void)got;
  EXPECT_NE(st, FUSE_OK);
  EXPECT_EQ(code, "E-RT-SIZE");
}

TEST(AbiArithmeticElement, ADoubleDoubleFieldIsNotDeliveredIntoABinary64Field) {
  fuse_status st;
  std::string code;
  const auto got = Transfer({FUSE_ELEMENT_DD, 2, 1}, {FUSE_ELEMENT_F64, 2, 2}, 2, &st, &code);
  (void)got;
  EXPECT_NE(st, FUSE_OK) << "a double-double upstream into a binary64 downstream mixes the types";
  EXPECT_EQ(code, "E-RT-ARITHMETIC-MIX");
}

TEST(AbiGradient, AFieldWithAGradientIsRefusedOnAMinimumOrMaximumChannel) {
  const auto* t = Table();
  for (const auto op : {fuse::AccumOp::Min, fuse::AccumOp::Max}) {
    for (const int gradient : {2, 0}) {
      fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(3, op));
      std::vector<double> up(12, 0.0), down(12, 0.0);
      ASSERT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 3, up.data(), gradient), FUSE_OK);
      ASSERT_EQ(Register(t, ctx, "downstream", "in", FUSE_ELEMENT_F64, 3, down.data()), FUSE_OK);
      const fuse_status st = t->initialize(ctx, nullptr, nullptr);
      if (gradient > 0) {
        EXPECT_NE(st, FUSE_OK);
        EXPECT_EQ(LastErrorCode(t, ctx), "E-RT-GRADIENT-MINMAX");
      } else {
        EXPECT_EQ(st, FUSE_OK) << LastErrorCode(t, ctx);
      }
      EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
    }
  }
}

TEST(AbiGradient, ASumOrMeanChannelCarriesTheGradientAlong) {
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(3));
  std::vector<double> up(12, 0.0), down(12, 0.0);
  ASSERT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 3, up.data(), 2), FUSE_OK);
  ASSERT_EQ(Register(t, ctx, "downstream", "in", FUSE_ELEMENT_F64, 3, down.data(), 2), FUSE_OK);
  ASSERT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK) << LastErrorCode(t, ctx);
  ASSERT_EQ(Period(t, ctx, [&] {
              for (int c = 0; c < 4; ++c) {
                up[static_cast<std::size_t>(3 * c)] = 1.0 + c;
                up[static_cast<std::size_t>(3 * c + 1)] = 0.5 * c;
                up[static_cast<std::size_t>(3 * c + 2)] = -0.25 * c;
              }
            }),
            FUSE_OK);
  EXPECT_EQ(down, up) << "value and gradient arrive together";
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiGradient, AGradientSizeThatLeavesNoValueIsRefusedAtRegistration) {
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(3));
  std::vector<double> up(12, 0.0);
  for (const int bad : {3, 4, -1}) {
    EXPECT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 3, up.data(), bad), FUSE_ERR_INVALID)
        << bad;
    EXPECT_EQ(LastErrorCode(t, ctx), "E-ABI-REGISTER-GRADIENT");
  }
  EXPECT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 1, up.data(), 1), FUSE_ERR_INVALID);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiGradient, ViewFieldReportsTheRegisteredGradientSize) {
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(3));
  std::vector<double> up(12, 0.0), down(12, 0.0);
  ASSERT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 3, up.data(), 2), FUSE_OK);
  ASSERT_EQ(Register(t, ctx, "downstream", "in", FUSE_ELEMENT_F64, 3, down.data()), FUSE_OK);
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "upstream";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view view{.struct_size = sizeof(view)};
  ASSERT_EQ(t->view_field(ctx, &fref, &view), FUSE_OK);
  EXPECT_EQ(view.gradient_size, 2);
  EXPECT_EQ(view.components, 3);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiGradient, AViewFromBeforeTheMemberExistedRegistersAsAnOrdinaryField) {
  // a caller built against the header that ended at window_samples passes its smaller
  // struct_size; the bytes after it are not read, whatever they hold
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, GraphBytes(3, fuse::AccumOp::Min));
  std::vector<double> up(12, 0.0), down(12, 0.0);
  const std::uint32_t old_size = static_cast<std::uint32_t>(offsetof(fuse_field_view, gradient_size));
  ASSERT_LT(old_size, sizeof(fuse_field_view));
  EXPECT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_F64, 3, up.data(), 2, old_size), FUSE_OK);
  EXPECT_EQ(Register(t, ctx, "downstream", "in", FUSE_ELEMENT_F64, 3, down.data(), 0, old_size), FUSE_OK);
  EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK)
      << "the gradient_size the old caller never wrote must not have been read: "
      << LastErrorCode(t, ctx);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

TEST(AbiGradient, TheRegistryRefusesAGradientThatLeavesNoValueOrNoGradient) {
  std::vector<double> data(12, 0.0);
  fuse::FieldRegistry reg;
  const fuse::FieldId id{0};
  const auto unbound = reg.declare_gradient(id, 1);
  ASSERT_FALSE(unbound.ok());
  EXPECT_EQ(unbound.error().code, "E-FIELD-UNBOUND");
  ASSERT_TRUE(reg.bind(id, fuse::FieldRef(data.data(), fuse::Layout{3, 1, 4, 3}, fuse::MemorySpace::Host)).ok());
  for (const int bad : {0, 3, 4, -1}) {
    const auto r = reg.declare_gradient(id, bad);
    ASSERT_FALSE(r.ok()) << bad;
    EXPECT_EQ(r.error().code, "E-FIELD-GRADIENT") << bad;
  }
  EXPECT_FALSE(reg.has_gradient(id));
  ASSERT_TRUE(reg.declare_gradient(id, 2).ok());
  EXPECT_TRUE(reg.has_gradient(id));
  EXPECT_EQ(reg.get_any(id).value().gradient_size(), 2);
}
