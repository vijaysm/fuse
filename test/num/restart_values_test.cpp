// Fields of any coupled value type in a restart file: double-double, quad-double and
// dual numbers stored as their scalars, the storage formats as themselves
// (plans/2026-10-22-universal-number-types.md, stage U4). The writer and reader here keep
// what they are given in memory, with the checks the HDF5 ones make on the type and the
// component count, so the test is of what put_values and get_values hand to the restart
// interface: the stored type, the components per row, the bytes.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "fuse/model/bfloat16.hpp"
#include "fuse/model/binary16.hpp"
#include "fuse/model/restart_values.hpp"
#include "fuse/num/dd.hpp"
#include "fuse/num/dual.hpp"
#include "fuse/num/qd.hpp"

namespace {

using fuse::num::dd;
using fuse::num::qd;

struct Stored {
  std::vector<std::int64_t> ids;
  fuse::ElementType type{};
  int components = 0;
  std::vector<std::byte> bytes;
};

class MemoryWriter final : public fuse::RestartWriter {
 public:
  std::map<std::string, Stored> fields;
  fuse::Result<void> put_field_bytes(std::string_view name, std::span<const std::int64_t> ids,
                                     fuse::ElementType type, std::span<const std::byte> values,
                                     int components, std::int64_t) override {
    if (values.size() != ids.size() * static_cast<std::size_t>(components) *
                             fuse::element_type_info(type).byte_size) {
      return fuse::Failure{"E-TEST-SIZE", "the bytes are not ids * components * element size"};
    }
    fields[std::string(name)] = {{ids.begin(), ids.end()}, type, components, {values.begin(), values.end()}};
    return fuse::Result<void>::success();
  }
  fuse::Result<void> put_scalar_bytes(std::string_view, fuse::ElementType, std::span<const std::byte>) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> put_scalar_rational(std::string_view, const fuse::Rational&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> put_blob(std::string_view, std::span<const std::byte>, std::uint32_t) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> finish_point() override { return fuse::Result<void>::success(); }
  fuse::Result<std::vector<std::int64_t>> recorded_ids(std::string_view name) const override {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) return fuse::Failure{"E-TEST-NOFIELD", "no such field"};
    return it->second.ids;
  }
};

/// Reads what a MemoryWriter holds; the type and the components must be what was stored, the
/// ReadConversion::Exact rule of the real reader.
class MemoryReader final : public fuse::RestartReader {
 public:
  explicit MemoryReader(const MemoryWriter& w) : w_(w) {}
  fuse::Result<fuse::ReadConversionReport> get_field_bytes(std::string_view name,
                                                           std::span<const std::int64_t> ids,
                                                           fuse::ElementType requested,
                                                           std::span<std::byte> out, int components,
                                                           fuse::ReadConversion) override {
    const auto it = w_.fields.find(std::string(name));
    if (it == w_.fields.end()) return fuse::Failure{"E-RESTART-MISSING-FIELD", "no such field"};
    const Stored& s = it->second;
    if (s.type != requested) return fuse::Failure{"E-RESTART-TYPE-MISMATCH", "stored type differs"};
    if (s.components != components) {
      return fuse::Failure{"E-RESTART-COMPONENTS-MISMATCH", "stored components differ"};
    }
    if (std::vector<std::int64_t>(ids.begin(), ids.end()) != s.ids || out.size() != s.bytes.size()) {
      return fuse::Failure{"E-RESTART-MISSING-ID", "ids or size differ"};
    }
    std::memcpy(out.data(), s.bytes.data(), out.size());
    return fuse::ReadConversionReport{};
  }
  fuse::Result<fuse::ReadConversionReport> get_scalar_bytes(std::string_view, fuse::ElementType,
                                                            std::span<std::byte>,
                                                            fuse::ReadConversion) override {
    return fuse::ReadConversionReport{};
  }
  fuse::Result<fuse::Rational> get_scalar_rational(std::string_view) override { return fuse::Rational{0}; }
  fuse::Result<std::vector<std::byte>> get_blob(std::string_view, std::uint32_t&) override {
    return std::vector<std::byte>{};
  }
  fuse::Result<bool> has_id(std::string_view, std::int64_t) override { return true; }

 private:
  const MemoryWriter& w_;
};

const std::vector<std::int64_t> kIds = {10, 11, 12};

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <class T>
std::vector<T> from_scalars(const std::vector<double>& s) {
  std::vector<T> v(s.size() * sizeof(double) / sizeof(T));
  std::memcpy(v.data(), s.data(), s.size() * sizeof(double));
  return v;
}

}  // namespace

TEST(RestartValues, ADoubleDoubleIsStoredAsItsTwoLimbsInBinary64) {
  const std::vector<dd> x = {dd(1.0, 0x1p-60), dd(-3.0, -0x1p-58), dd(0.1, 0x1p-57)};
  MemoryWriter w;
  ASSERT_TRUE(fuse::put_values<dd>(w, "q", kIds, x, 1, 3).ok());
  const Stored& s = w.fields.at("q");
  EXPECT_EQ(s.type, fuse::ElementType::Binary64);
  EXPECT_EQ(s.components, 2);
  ASSERT_EQ(s.bytes.size(), 3 * 2 * sizeof(double));
  const double hi1 = [&] { double d; std::memcpy(&d, s.bytes.data() + 2 * sizeof(double), sizeof d); return d; }();
  const double lo1 = [&] { double d; std::memcpy(&d, s.bytes.data() + 3 * sizeof(double), sizeof d); return d; }();
  EXPECT_EQ(hi1, -3.0);
  EXPECT_EQ(lo1, -0x1p-58) << "the low limb is stored, not dropped";

  MemoryReader r(w);
  std::vector<dd> back(3);
  ASSERT_TRUE(fuse::get_values<dd>(r, "q", kIds, back, 1).ok());
  EXPECT_TRUE(same_bits(back, x));
}

TEST(RestartValues, SeveralValuesPerRowMultiplyTheStoredComponents) {
  std::vector<qd> x;
  for (int i = 0; i < 6; ++i) x.push_back(qd(1.0 + i, 0x1p-55, 0x1p-110, -0x1p-165));
  MemoryWriter w;
  ASSERT_TRUE(fuse::put_values<qd>(w, "q", kIds, x, 2, 3).ok());
  EXPECT_EQ(w.fields.at("q").components, 8) << "two quad-double values of four limbs";
  EXPECT_EQ(w.fields.at("q").type, fuse::ElementType::Binary64);
  MemoryReader r(w);
  std::vector<qd> back(6);
  ASSERT_TRUE(fuse::get_values<qd>(r, "q", kIds, back, 2).ok());
  EXPECT_TRUE(same_bits(back, x));
  // the same components in values is what the reader has to ask for
  std::vector<qd> wrong(6);
  const auto bad = fuse::get_values<qd>(r, "q", kIds, wrong, 1);
  EXPECT_FALSE(bad.ok());
}

TEST(RestartValues, ADualNumberIsStoredAsItsValueAndGradient) {
  using D = fuse::num::dual<double, 2>;
  const std::vector<D> x = from_scalars<D>({1.5, 0.25, -0.5, 2.5, 0.125, 8.0, -3.0, 1.0, 0.0});
  ASSERT_EQ(x.size(), 3u);
  MemoryWriter w;
  ASSERT_TRUE(fuse::put_values<D>(w, "u", kIds, x, 1, 3).ok());
  EXPECT_EQ(w.fields.at("u").components, 3);
  EXPECT_EQ(w.fields.at("u").type, fuse::ElementType::Binary64);
  MemoryReader r(w);
  std::vector<D> back(3);
  ASSERT_TRUE(fuse::get_values<D>(r, "u", kIds, back, 1).ok());
  EXPECT_TRUE(same_bits(back, x));
}

TEST(RestartValues, TheStorageFormatsAreStoredAsThemselves) {
  const std::vector<float> f = {1.5f, -0.1f, 3.25f};
  const std::vector<fuse::bfloat16> b = {fuse::value_traits<fuse::bfloat16>::from_binary64(0.1),
                                         fuse::value_traits<fuse::bfloat16>::from_binary64(-7.0),
                                         fuse::value_traits<fuse::bfloat16>::from_binary64(1e10)};
  const std::vector<fuse::binary16> h = {fuse::value_traits<fuse::binary16>::from_binary64(0.1),
                                         fuse::value_traits<fuse::binary16>::from_binary64(-7.0),
                                         fuse::value_traits<fuse::binary16>::from_binary64(65504.0)};
  MemoryWriter w;
  ASSERT_TRUE(fuse::put_values<float>(w, "f", kIds, f, 1, 3).ok());
  ASSERT_TRUE(fuse::put_values<fuse::bfloat16>(w, "b", kIds, b, 1, 3).ok());
  ASSERT_TRUE(fuse::put_values<fuse::binary16>(w, "h", kIds, h, 1, 3).ok());
  EXPECT_EQ(w.fields.at("f").type, fuse::ElementType::Binary32);
  EXPECT_EQ(w.fields.at("b").type, fuse::ElementType::BFloat16);
  EXPECT_EQ(w.fields.at("h").type, fuse::ElementType::Binary16);
  for (const char* n : {"f", "b", "h"}) EXPECT_EQ(w.fields.at(n).components, 1) << n;
  MemoryReader r(w);
  std::vector<float> fb(3);
  std::vector<fuse::bfloat16> bb(3);
  std::vector<fuse::binary16> hb(3);
  ASSERT_TRUE(fuse::get_values<float>(r, "f", kIds, fb, 1).ok());
  ASSERT_TRUE(fuse::get_values<fuse::bfloat16>(r, "b", kIds, bb, 1).ok());
  ASSERT_TRUE(fuse::get_values<fuse::binary16>(r, "h", kIds, hb, 1).ok());
  EXPECT_TRUE(same_bits(fb, f));
  EXPECT_TRUE(same_bits(bb, b));
  EXPECT_TRUE(same_bits(hb, h));
}

TEST(RestartValues, ABinary64FileOfLimbsIsReadAsLimbsByAnyReader) {
  // a program that does not know the type reads the file it wrote as plain binary64
  const std::vector<dd> x = {dd(1.0, 0x1p-60), dd(2.0, 0x1p-59), dd(4.0, 0x1p-58)};
  MemoryWriter w;
  ASSERT_TRUE(fuse::put_values<dd>(w, "q", kIds, x, 1, 3).ok());
  MemoryReader r(w);
  std::vector<double> plain(6);
  ASSERT_TRUE(r.get_field<double>("q", kIds, plain, 2).ok());
  EXPECT_EQ(plain[0], 1.0);
  EXPECT_EQ(plain[1], 0x1p-60);
  EXPECT_EQ(plain[5], 0x1p-58);
}
