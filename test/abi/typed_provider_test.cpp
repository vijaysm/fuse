// A provider plugin's fields in the element types the host allocates for it: binary64, the
// storage types (binary32, bfloat16, binary16, posit16, posit32) and the arithmetic types (double-double,
// quad-double), and a field's gradient_size (plans/2026-10-22-universal-number-types.md,
// stage U4). The plugin is an in-memory table, so that the adapter's own decisions are what
// the test sees: the type each field is bound in, the view the plugin is handed, and the
// refusals by name. A row counts values of its element type; the graph counts binary64
// values, so a double-double row of 1 value is a graph field of 2 components.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fuse/abi/provider.h"
#include "fuse/base/units.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/bfloat16.hpp"
#include "fuse/model/binary16.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/num/dd.hpp"
#include "fuse/num/posit.hpp"
#include "fuse/num/qd.hpp"

#include "provider_adapter.hpp"

namespace {

struct Row {
  const char* name;
  int graph_components;  // what the graph declares: binary64 values per cell
  int row_components;    // what describe() reports: values of the element type
  int element_type;
  int gradient_size = 0;
  int memspace = FUSE_MEMSPACE_HOST;
};

struct Seen {
  std::int32_t element_type = -1;
  std::int32_t components = 0;
  std::int32_t gradient_size = 0;
  std::int64_t stride = 0;
  std::uint64_t count = 0;
  void* data = nullptr;
};

std::vector<Row> g_rows;
std::vector<Seen> g_seen;
int g_instance = 0;
// the adapter owns the storage the fields are bound to, so it outlives each test body
std::unique_ptr<fuse::abi_detail::ProviderAdapter> g_adapter;

fuse_status t_create(fuse_provider_ctx*, const fuse_null*, fuse_provider_instance* out) {
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = reinterpret_cast<fuse_provider_ctx*>(&g_instance);
  return FUSE_OK;
}
fuse_status t_destroy(fuse_provider_ctx*, const fuse_null*, fuse_null*) { return FUSE_OK; }
fuse_status t_describe(fuse_provider_ctx*, const fuse_null*, fuse_field_table* out) {
  out->struct_size = sizeof(fuse_field_table);
  out->written = g_rows.size();
  if (out->buffer != nullptr && out->capacity >= g_rows.size()) {
    for (std::size_t i = 0; i < g_rows.size(); ++i) {
      fuse_field_spec& r = out->buffer[i];
      r.struct_size = sizeof(fuse_field_spec);
      r.name = g_rows[i].name;
      r.components = g_rows[i].row_components;
      r.intent = 0;  // Provides
      r.element_type = g_rows[i].element_type;
      r.memspace = g_rows[i].memspace;
      r.gradient_size = g_rows[i].gradient_size;
    }
  }
  return FUSE_OK;
}
fuse_status t_bind(fuse_provider_ctx*, const fuse_field_view* in, fuse_null*) {
  Seen s;
  s.element_type = in->element_type;
  s.components = in->components;
  s.gradient_size = in->gradient_size;
  s.stride = in->stride;
  s.count = in->count;
  s.data = in->data;
  g_seen.push_back(s);
  return FUSE_OK;
}
fuse_status t_advance(fuse_provider_ctx*, const fuse_step_request*, fuse_step_report*) { return FUSE_OK; }

const fuse_provider_abi_v1 kTable = {
    .struct_size = sizeof(fuse_provider_abi_v1),
    .abi_version = FUSE_ABI_VERSION,
    .create = t_create,
    .destroy = t_destroy,
    .describe = t_describe,
    .bind = t_bind,
    .advance = t_advance,
};

fuse::CouplingGraph MakeGraph(const std::vector<Row>& rows, int cells) {
  fuse::CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:" + std::to_string(cells)});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 1});
  const auto prov = g.add_component(fuse::ComponentDecl{
      .name = "prov", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  for (const Row& row : rows) {
    g.add_field(prov, fuse::FieldSpec{.name = row.name, .region = r, .components = row.graph_components,
                                      .units = fuse::units::kKelvin,
                                      .conservation = fuse::Conservation::None,
                                      .intent = fuse::Intent::Provides});
  }
  return g;
}

fuse::Result<void> Register(const std::vector<Row>& rows, fuse::FieldRegistry& registry,
                             fuse::CouplingGraph& graph_out, int cells = 4) {
  g_rows = rows;
  g_seen.clear();
  graph_out = MakeGraph(rows, cells);
  g_adapter = std::make_unique<fuse::abi_detail::ProviderAdapter>("prov", &kTable);
  return g_adapter->register_fields(registry, graph_out);
}

fuse::FieldId Id(const fuse::CouplingGraph& g, const char* name) {
  return g.find_field(g.find_component("prov").value(), name).value();
}

}  // namespace

TEST(AbiTypedProvider, EachRowIsBoundInItsOwnTypeAndTheViewCountsValues) {
  const std::vector<Row> rows = {
      {"plain", 1, 1, FUSE_ELEMENT_F64},   {"single", 1, 1, FUSE_ELEMENT_F32},
      {"brain", 1, 1, FUSE_ELEMENT_BF16},  {"half", 1, 1, FUSE_ELEMENT_F16},
      {"dd", 2, 1, FUSE_ELEMENT_DD},       {"qd", 4, 1, FUSE_ELEMENT_QD},
      {"dd3", 6, 3, FUSE_ELEMENT_DD},      {"grad", 3, 3, FUSE_ELEMENT_F64, 2},
      {"p16", 1, 1, FUSE_ELEMENT_POSIT16}, {"p32", 2, 2, FUSE_ELEMENT_POSIT32},
  };
  fuse::FieldRegistry registry;
  fuse::CouplingGraph graph;
  const auto r = Register(rows, registry, graph);
  ASSERT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  ASSERT_EQ(g_seen.size(), rows.size());

  EXPECT_TRUE(registry.get_any(Id(graph, "plain")).value().holds<double>());
  EXPECT_TRUE(registry.get_any(Id(graph, "single")).value().holds<float>());
  EXPECT_TRUE(registry.get_any(Id(graph, "brain")).value().holds<fuse::bfloat16>());
  EXPECT_TRUE(registry.get_any(Id(graph, "half")).value().holds<fuse::binary16>());
  EXPECT_TRUE(registry.get_any(Id(graph, "dd")).value().holds<fuse::num::dd>());
  EXPECT_TRUE(registry.get_any(Id(graph, "qd")).value().holds<fuse::num::qd>());
  EXPECT_TRUE(registry.get_any(Id(graph, "dd3")).value().holds<fuse::num::dd>());
  EXPECT_TRUE(registry.get_any(Id(graph, "p16")).value().holds<fuse::num::posit16>());
  EXPECT_TRUE(registry.get_any(Id(graph, "p32")).value().holds<fuse::num::posit32>());

  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto any = registry.get_any(Id(graph, rows[i].name));
    ASSERT_TRUE(any.ok());
    EXPECT_EQ(any.value().layout().count, 4u) << rows[i].name;
    EXPECT_EQ(any.value().layout().components, rows[i].row_components) << rows[i].name;
    EXPECT_EQ(g_seen[i].element_type, rows[i].element_type) << rows[i].name;
    EXPECT_EQ(g_seen[i].components, rows[i].row_components) << rows[i].name;
    EXPECT_EQ(g_seen[i].stride, rows[i].row_components) << rows[i].name;
    EXPECT_EQ(g_seen[i].count, 4u) << rows[i].name;
    EXPECT_EQ(g_seen[i].gradient_size, rows[i].gradient_size) << rows[i].name;
    EXPECT_EQ(g_seen[i].data, any.value().data()) << rows[i].name;
  }
  // the gradient the plugin declared is the one the Runner refuses minimum and maximum on
  EXPECT_TRUE(registry.has_gradient(Id(graph, "grad")));
  EXPECT_FALSE(registry.has_gradient(Id(graph, "plain")));
  EXPECT_FALSE(registry.has_gradient(Id(graph, "dd")));
}

TEST(AbiTypedProvider, TheStorageHoldsEveryElementOfEveryTypeAndNoMoreThanAWordOver) {
  // three cells, so that the byte counts are not multiples of the word: 6, 36, 48 and 192 bytes
  const std::vector<Row> rows = {{"half", 1, 1, FUSE_ELEMENT_F16},  {"single", 3, 3, FUSE_ELEMENT_F32},
                                 {"dd", 2, 1, FUSE_ELEMENT_DD},      {"qd", 8, 2, FUSE_ELEMENT_QD},
                                 {"p16", 1, 1, FUSE_ELEMENT_POSIT16}, {"p32", 3, 3, FUSE_ELEMENT_POSIT32}};
  const std::size_t element_size[] = {2, 4, 16, 32, 2, 4};
  fuse::FieldRegistry registry;
  fuse::CouplingGraph graph;
  ASSERT_TRUE(Register(rows, registry, graph, 3).ok());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t need = 3u * static_cast<std::size_t>(rows[i].row_components) * element_size[i];
    EXPECT_GE(g_adapter->host_storage_bytes(i), need) << rows[i].name;
    EXPECT_LT(g_adapter->host_storage_bytes(i), need + sizeof(double)) << rows[i].name;
  }
  // and the last element of each field can be written and read back
  static_cast<float*>(g_seen[1].data)[3 * 3 - 1] = 2.5f;
  EXPECT_EQ(static_cast<float*>(g_seen[1].data)[3 * 3 - 1], 2.5f);
  static_cast<fuse::num::dd*>(g_seen[2].data)[3 - 1] = fuse::num::dd{1.0, 0x1p-60};
  EXPECT_EQ(static_cast<fuse::num::dd*>(g_seen[2].data)[3 - 1].lo, 0x1p-60);
  static_cast<fuse::num::qd*>(g_seen[3].data)[3 * 2 - 1] = fuse::num::qd(1.0, 0x1p-60, 0x1p-120, 0x1p-180);
  EXPECT_EQ(static_cast<fuse::num::qd*>(g_seen[3].data)[3 * 2 - 1][3], 0x1p-180);
  static_cast<fuse::num::posit16*>(g_seen[4].data)[3 - 1] = fuse::num::posit16::from_double(3.0);
  EXPECT_EQ(static_cast<fuse::num::posit16*>(g_seen[4].data)[3 - 1].to_double(), 3.0);
  static_cast<fuse::num::posit32*>(g_seen[5].data)[3 * 3 - 1] = fuse::num::posit32::from_double(-0.75);
  EXPECT_EQ(static_cast<fuse::num::posit32*>(g_seen[5].data)[3 * 3 - 1].to_double(), -0.75);
}

TEST(AbiTypedProvider, ARowThatCountsBinary64ValuesForADoubleDoubleFieldDisagreesWithTheGraph) {
  fuse::FieldRegistry registry;
  fuse::CouplingGraph graph;
  const auto r = Register({{"dd", 2, 2, FUSE_ELEMENT_DD}}, registry, graph);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ABI-DESCRIBE");
  // and a graph count that is not a whole number of values
  const auto odd = Register({{"dd", 3, 1, FUSE_ELEMENT_DD}}, registry, graph);
  ASSERT_FALSE(odd.ok());
  EXPECT_EQ(odd.error().code, "E-ABI-DESCRIBE");
}

TEST(AbiTypedProvider, AnElementTypeOutsideTheListIsRefusedByName) {
  for (const int et : {static_cast<int>(FUSE_ELEMENT_I64), 99, -1}) {
    fuse::FieldRegistry registry;
    fuse::CouplingGraph graph;
    const auto r = Register({{"x", 1, 1, et}}, registry, graph);
    ASSERT_FALSE(r.ok()) << et;
    EXPECT_EQ(r.error().code, "E-ABI-PROVIDER-ELEMENT") << et;
  }
}

TEST(AbiTypedProvider, DeviceStorageIsForBinary64Only) {
  fuse::FieldRegistry registry;
  fuse::CouplingGraph graph;
  const auto r = Register({{"x", 1, 1, FUSE_ELEMENT_F32, 0, FUSE_MEMSPACE_DEVICE}}, registry, graph);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ABI-PROVIDER-ELEMENT");
  EXPECT_NE(r.error().message.find("binary64 only"), std::string::npos) << r.error().message;
}

TEST(AbiTypedProvider, AGradientThatLeavesNoValueIsRefusedByName) {
  for (const int bad : {3, 4, -1}) {
    fuse::FieldRegistry registry;
    fuse::CouplingGraph graph;
    const auto r = Register({{"g", 3, 3, FUSE_ELEMENT_F64, bad}}, registry, graph);
    ASSERT_FALSE(r.ok()) << bad;
    EXPECT_EQ(r.error().code, "E-ABI-PROVIDER-GRADIENT") << bad;
  }
}
