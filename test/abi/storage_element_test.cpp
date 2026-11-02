// FUSE_ELEMENT_F32, FUSE_ELEMENT_BF16, FUSE_ELEMENT_F16 and, from stage U5,
// FUSE_ELEMENT_POSIT16 and FUSE_ELEMENT_POSIT32 through the plain-C
// participant table: a
// C or Fortran component that keeps single, half or posit precision arrays registers
// them as they are. The coupler widens them to binary64 when the component has
// written them and rounds once into them on delivery, so the value a downstream
// component receives is the binary64 transfer of what the upstream wrote, rounded
// once to the downstream type. fuse_view_field reports the element type back.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/model/value_traits.hpp"
#include "fuse/num/posit.hpp"

#include "graph_fixture.hpp"
#include "test_support.hpp"

namespace {

const fuse_participant_abi_v1* Table() { return fuse_coupler_abi(FUSE_ABI_VERSION); }

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
                     const char* field, int element_type, void* data, int count = 4) {
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = component;
  fname.field = field;
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = fref.handle;
  fview.element_type = element_type;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = static_cast<std::uint64_t>(count);
  fview.owned_count = static_cast<std::uint64_t>(count);
  fview.stride = 1;
  fview.data = data;
  return t->register_field(ctx, &fview, nullptr);
}

// One period of the two-component graph: upstream writes `values` into its
// array through `write`, downstream's array is read back through `read`.
template <class Up, class Down, class Write, class Read>
std::vector<double> OnePeriod(int up_type, int down_type, const std::vector<double>& values,
                              Write write, Read read) {
  const auto* t = Table();
  const auto bytes = fuse::abi_test::build_two_component_lag1_bytes();
  fuse_ctx* ctx = CreateAndLoad(t, bytes);
  std::vector<Up> up(4);
  std::vector<Down> down(4);
  EXPECT_EQ(Register(t, ctx, "upstream", "out", up_type, up.data()), FUSE_OK);
  EXPECT_EQ(Register(t, ctx, "downstream", "in", down_type, down.data()), FUSE_OK);
  EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request up_req{.struct_size = sizeof(up_req)};
  up_req.component = "upstream";
  fuse_step_request down_req{.struct_size = sizeof(down_req)};
  down_req.component = "downstream";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_EQ(t->begin_step(ctx, &up_req, &rep), FUSE_OK);
  write(up, values);
  EXPECT_EQ(t->end_step(ctx, &up_req, &rep), FUSE_OK);
  EXPECT_EQ(t->begin_step(ctx, &down_req, &rep), FUSE_OK);
  EXPECT_EQ(t->end_step(ctx, &down_req, &rep), FUSE_OK);
  std::vector<double> out = read(down);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  return out;
}

const std::vector<double> kValues = {1.1, 2.7182818284590451, -3.3333333333333335, 1234.5678901234};

template <class T>
void WriteAs(std::vector<T>& v, const std::vector<double>& x) {
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = fuse::value_traits<T>::from_binary64(x[i]);
}
template <class T>
std::vector<double> ReadAs(const std::vector<T>& v) {
  std::vector<double> x;
  for (const T& e : v) x.push_back(fuse::value_traits<T>::to_binary64(e));
  return x;
}

template <class T>
double Through(double x) {
  return fuse::value_traits<T>::to_binary64(fuse::value_traits<T>::from_binary64(x));
}

}  // namespace

TEST(AbiStorageElement, AFloatUpstreamArrivesAsTheSameFloatInAFloatDownstream) {
  // an identity-like channel on matching meshes: the value in is the value out,
  // widened and rounded back to the same format -- unchanged
  const auto got = OnePeriod<float, float>(FUSE_ELEMENT_F32, FUSE_ELEMENT_F32, kValues,
                                           WriteAs<float>, ReadAs<float>);
  for (std::size_t i = 0; i < kValues.size(); ++i) {
    EXPECT_EQ(got[i], Through<float>(kValues[i])) << i;
  }
}

TEST(AbiStorageElement, FormatsMixAcrossTheBoundary) {
  // binary64 upstream into a binary16 downstream: one rounding, at delivery
  const auto into16 = OnePeriod<double, fuse::binary16>(FUSE_ELEMENT_F64, FUSE_ELEMENT_F16, kValues,
                                                         WriteAs<double>, ReadAs<fuse::binary16>);
  const auto exact = OnePeriod<double, double>(FUSE_ELEMENT_F64, FUSE_ELEMENT_F64, kValues,
                                               WriteAs<double>, ReadAs<double>);
  for (std::size_t i = 0; i < kValues.size(); ++i) {
    EXPECT_EQ(into16[i], Through<fuse::binary16>(exact[i])) << i;
  }
  // bfloat16 upstream into a binary64 downstream: widened exactly, no rounding
  const auto from_bf = OnePeriod<fuse::bfloat16, double>(FUSE_ELEMENT_BF16, FUSE_ELEMENT_F64, kValues,
                                                         WriteAs<fuse::bfloat16>, ReadAs<double>);
  for (std::size_t i = 0; i < kValues.size(); ++i) {
    EXPECT_EQ(from_bf[i], Through<fuse::bfloat16>(kValues[i])) << i;
  }
}

TEST(AbiStorageElement, APositUpstreamArrivesAsTheSamePositInAPositDownstream) {
  using fuse::num::posit16;
  using fuse::num::posit32;
  const auto got16 = OnePeriod<posit16, posit16>(FUSE_ELEMENT_POSIT16, FUSE_ELEMENT_POSIT16, kValues,
                                                 WriteAs<posit16>, ReadAs<posit16>);
  const auto got32 = OnePeriod<posit32, posit32>(FUSE_ELEMENT_POSIT32, FUSE_ELEMENT_POSIT32, kValues,
                                                 WriteAs<posit32>, ReadAs<posit32>);
  for (std::size_t i = 0; i < kValues.size(); ++i) {
    EXPECT_EQ(got16[i], Through<posit16>(kValues[i])) << i;
    EXPECT_EQ(got32[i], Through<posit32>(kValues[i])) << i;
    EXPECT_NE(got16[i], kValues[i]) << "the 16-bit posit is not the binary64 value";
  }
}

TEST(AbiStorageElement, PositsMixWithTheOtherFormatsAndAValueBeyondTheRangeIsClamped) {
  using fuse::num::posit16;
  using fuse::num::posit32;
  // binary64 into posit16, binary16 into posit32, posit32 into binary64: one rounding at most
  const auto into16 = OnePeriod<double, posit16>(FUSE_ELEMENT_F64, FUSE_ELEMENT_POSIT16, kValues,
                                                 WriteAs<double>, ReadAs<posit16>);
  const auto from32 = OnePeriod<posit32, double>(FUSE_ELEMENT_POSIT32, FUSE_ELEMENT_F64, kValues,
                                                 WriteAs<posit32>, ReadAs<double>);
  const auto half_to_32 = OnePeriod<fuse::binary16, posit32>(
      FUSE_ELEMENT_F16, FUSE_ELEMENT_POSIT32, kValues, WriteAs<fuse::binary16>, ReadAs<posit32>);
  for (std::size_t i = 0; i < kValues.size(); ++i) {
    EXPECT_EQ(into16[i], Through<posit16>(kValues[i])) << i;
    EXPECT_EQ(from32[i], Through<posit32>(kValues[i])) << i;
    EXPECT_EQ(half_to_32[i], Through<posit32>(Through<fuse::binary16>(kValues[i]))) << i;
  }
  // a value beyond the largest posit16 (2^56) is the largest posit16, not an infinity
  const std::vector<double> huge = {1e20, -1e20, 1e-10, -1e-30};
  const auto clamped = OnePeriod<double, posit16>(FUSE_ELEMENT_F64, FUSE_ELEMENT_POSIT16, huge,
                                                  WriteAs<double>, ReadAs<posit16>);
  EXPECT_EQ(clamped[0], posit16::maxpos().to_double());
  EXPECT_EQ(clamped[1], -posit16::maxpos().to_double());
  EXPECT_EQ(clamped[2], Through<posit16>(1e-10));  // within range: rounded, not clamped
  EXPECT_GT(clamped[2], posit16::minpos().to_double());
  // below the smallest positive posit16 (2^-56): the smallest, not zero
  EXPECT_EQ(clamped[3], -posit16::minpos().to_double());
}

TEST(AbiStorageElement, ViewFieldReportsTheElementTypeTheFieldWasRegisteredWith) {
  const auto* t = Table();
  for (const int et : {FUSE_ELEMENT_F64, FUSE_ELEMENT_F32, FUSE_ELEMENT_BF16, FUSE_ELEMENT_F16,
                       FUSE_ELEMENT_POSIT16, FUSE_ELEMENT_POSIT32}) {
    fuse_ctx* ctx = CreateAndLoad(t, fuse::abi_test::build_two_component_lag1_bytes());
    alignas(8) unsigned char up[4 * 8] = {};
    alignas(8) unsigned char down[4 * 8] = {};
    ASSERT_EQ(Register(t, ctx, "upstream", "out", et, up), FUSE_OK);
    ASSERT_EQ(Register(t, ctx, "downstream", "in", et, down), FUSE_OK);
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = "upstream";
    fname.field = "out";
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    fuse_field_view view{.struct_size = sizeof(view)};
    ASSERT_EQ(t->view_field(ctx, &fref, &view), FUSE_OK);
    EXPECT_EQ(view.element_type, et);
    EXPECT_EQ(view.data, static_cast<void*>(up));
    EXPECT_EQ(view.count, 4u);
    EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  }
}

TEST(AbiStorageElement, TheIntegerElementTypeIsStillRefusedByName) {
  const auto* t = Table();
  fuse_ctx* ctx = CreateAndLoad(t, fuse::abi_test::build_two_component_lag1_bytes());
  std::int64_t data[4] = {};
  EXPECT_EQ(Register(t, ctx, "upstream", "out", FUSE_ELEMENT_I64, data), FUSE_ERR_UNSUPPORTED);
  fuse_error err{.struct_size = sizeof(err)};
  ASSERT_EQ(t->last_error(ctx, nullptr, &err), FUSE_OK);
  EXPECT_STREQ(err.code, "E-ABI-REGISTER-ELEMENT");
  EXPECT_EQ(Register(t, ctx, "upstream", "out", 99, data), FUSE_ERR_UNSUPPORTED);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
