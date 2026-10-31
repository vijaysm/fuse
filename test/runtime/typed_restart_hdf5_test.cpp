/// (docs/dev/restart.md Section 11): the refusals that
/// can only be reached by corrupting a file -- `fuse::restart_io`'s own
/// typed-field metadata is INTERNALLY CONSISTENT by construction on
/// every write this project's own public interface can make, so testing
/// "the attribute and the dataset type disagree" or "a bool byte is
/// neither 0 nor 1" needs a file hand-edited below the public API, via
/// <hdf5.h> directly -- the ONE file in test/ that does, by the same
/// test-code latitude `tools/check-layering.py`'s own FORBIDDEN_PUBLIC
/// rule grants: that rule scans `include/fuse/**` only (never `test/`),
/// because the rule exists to keep a THIRD-PARTY TYPE out of this
/// project's own PUBLIC API surface, not to forbid a test from using a
/// library directly to construct a white-box fixture.
#include <hdf5.h>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

#include <fuse/model/element_type.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>

namespace {

std::string temp_path(const char* name) {
  return std::string("/tmp/") + name;
}

/// Opens `path` (already created by `fuse::restart_io::create_restart_
/// point()` and `finish()`ed) for raw, direct HDF5 edits -- the ONLY way
/// to construct a corrupted or foreign-platform fixture, since every
/// public write path keeps its own metadata self-consistent and matches
/// this build's own platform.
template <class F>
void edit_file_directly(const std::string& path, F&& fn) {
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  ASSERT_GE(file, 0) << "H5Fopen(" << path << ") failed";
  fn(file);
  H5Fclose(file);
}

void write_str_attr(hid_t loc, const char* name, const std::string& v) {
  const hid_t asp = H5Screate(H5S_SCALAR);
  const hid_t st = H5Tcopy(H5T_C_S1);
  H5Tset_size(st, v.size() + 1);
  const hid_t attr = H5Acreate2(loc, name, st, asp, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, st, v.c_str());
  H5Aclose(attr);
  H5Tclose(st);
  H5Sclose(asp);
}
void write_u32_attr(hid_t loc, const char* name, std::uint32_t v) {
  const hid_t asp = H5Screate(H5S_SCALAR);
  const hid_t attr =
      H5Acreate2(loc, name, H5T_NATIVE_UINT32, asp, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_UINT32, &v);
  H5Aclose(attr);
  H5Sclose(asp);
}
void write_u64_attr(hid_t loc, const char* name, std::uint64_t v) {
  const hid_t asp = H5Screate(H5S_SCALAR);
  const hid_t attr =
      H5Acreate2(loc, name, H5T_NATIVE_UINT64, asp, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attr, H5T_NATIVE_UINT64, &v);
  H5Aclose(attr);
  H5Sclose(asp);
}

}  // namespace

// "a reader that finds a byte other than 0/1 refuses by name"
// (docs/dev/restart.md Section 11, item 3).
TEST(TypedRestartHdf5, BoolByteNotZeroOrOneRefuses) {
  const std::string path = temp_path("typed_bool_corrupt.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(pw.value()->writer().put_scalar("v", true).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  edit_file_directly(path, [](hid_t file) {
    const hid_t d = H5Dopen2(file, "scalars/v", H5P_DEFAULT);
    ASSERT_GE(d, 0);
    const std::uint8_t corrupted = 2;
    ASSERT_GE(H5Dwrite(d, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       &corrupted),
              0);
    H5Dclose(d);
  });

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  bool restored = false;
  const auto rr = pr.value()->reader().get_scalar("v", restored);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-BAD-BOOL-BYTE");
  std::remove(path.c_str());
}

// "verify the attribute and the dataset type agree on read; disagreement
// refuses by name -- it means a corrupted or foreign file"
// (item 5) -- here, CORRUPTED: the attribute is changed but the
// real underlying HDF5 datatype is untouched, so the file's own two
// descriptions of itself disagree.
TEST(TypedRestartHdf5, AttributeDatasetTypeDisagreementRefuses) {
  const std::string path = temp_path("typed_attr_mismatch.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(pw.value()->writer().put_scalar("v", std::int32_t{7}).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  edit_file_directly(path, [](hid_t file) {
    const hid_t d = H5Dopen2(file, "scalars/v", H5P_DEFAULT);
    ASSERT_GE(d, 0);
    const hid_t a = H5Aopen(d, "byte_size", H5P_DEFAULT);
    ASSERT_GE(a, 0);
    const std::uint32_t wrong_size = 8;  // real dataset type is still 4 bytes
    ASSERT_GE(H5Awrite(a, H5T_NATIVE_UINT32, &wrong_size), 0);
    H5Aclose(a);
    H5Dclose(d);
  });

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::int32_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar("v", restored);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-TYPE-METADATA-MISMATCH");
  std::remove(path.c_str());
}

// An unknown element-type name (a newer file version using a kind this
// build has never heard of) refuses by name rather than guessing
// (`fuse::element_type_from_name()`'s own `std::nullopt` contract).
TEST(TypedRestartHdf5, UnknownElementTypeNameFromANewerFileVersionRefuses) {
  const std::string path = temp_path("typed_unknown_type.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  ASSERT_TRUE(pw.value()->writer().put_scalar("v", std::int32_t{7}).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  edit_file_directly(path, [](hid_t file) {
    const hid_t d = H5Dopen2(file, "scalars/v", H5P_DEFAULT);
    ASSERT_GE(d, 0);
    const hid_t a = H5Aopen(d, "element_type", H5P_DEFAULT);
    ASSERT_GE(a, 0);
    const hid_t t = H5Aget_type(a);
    const std::string future_name = "quad128";  // not a name this build knows
    char buf[16] = {};
    std::memcpy(buf, future_name.c_str(), future_name.size() + 1);
    ASSERT_GE(H5Awrite(a, t, buf), 0);
    H5Tclose(t);
    H5Aclose(a);
    H5Dclose(d);
  });

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  std::int32_t restored = 0;
  const auto rr = pr.value()->reader().get_scalar("v", restored);
  EXPECT_FALSE(rr.ok());
  EXPECT_EQ(rr.error().code, "E-RESTART-UNKNOWN-ELEMENT-TYPE");
  std::remove(path.c_str());
}

namespace {
/// Writes a hand-built x86-64-style 80-bit extended `long
/// double` dataset directly (docs/dev/restart.md Section 11, item 6) --
/// 64-bit EXPLICIT significand (`H5T_NORM_MSBSET`), 15-bit exponent,
/// bias 16383, 16-byte storage. THIS Mac cannot itself produce this
/// representation (its own `long double` is 53-bit, bit-identical to
/// `double`) -- bypasses `put_field`/`put_scalar` entirely.
/// `mantissa`/`biased_exponent`/`sign` name the ONE value this dataset holds.
void write_hand_built_x86_long_double(hid_t file, const char* path,
                                      std::uint64_t mantissa,
                                      std::uint16_t biased_exponent,
                                      bool sign) {
  hid_t t = H5Tcopy(H5T_IEEE_F64LE);
  ASSERT_GE(H5Tset_size(t, 16), 0);
  ASSERT_GE(H5Tset_precision(t, 80), 0);
  ASSERT_GE(H5Tset_fields(t, 79, 64, 15, 0, 64), 0);
  ASSERT_GE(H5Tset_ebias(t, 16383), 0);
  ASSERT_GE(H5Tset_norm(t, H5T_NORM_MSBSET), 0);
  ASSERT_GE(H5Tset_order(t, H5T_ORDER_LE), 0);

  unsigned char buf[16] = {};
  std::memcpy(buf, &mantissa, 8);
  const std::uint16_t tail = biased_exponent | (sign ? 0x8000u : 0u);
  std::memcpy(buf + 8, &tail, 2);

  const hid_t sp = H5Screate(H5S_SCALAR);
  const hid_t d = H5Dcreate2(file, path, t, sp, H5P_DEFAULT, H5P_DEFAULT,
                            H5P_DEFAULT);
  ASSERT_GE(d, 0);
  ASSERT_GE(H5Dwrite(d, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf), 0);

  write_str_attr(d, "element_type", "long double");
  write_u32_attr(d, "byte_size", 16);
  write_u32_attr(d, "significand_bits", 64);
  write_u32_attr(d, "exponent_bits", 15);
  write_u64_attr(d, "exponent_bias", 16383);
  write_str_attr(d, "byte_order", "little");
  write_u32_attr(d, "is_signed", 0);

  H5Dclose(d);
  H5Sclose(sp);
  H5Tclose(t);
}
}  // namespace

// The full end-to-end scenario (docs/dev/restart.md Section
// 11, item 6), through the REAL public RestartReader: Exact refuses,
// Widening refuses, AllowNarrowing reads with the reported maximum
// change equal to an independently computed value.
TEST(TypedRestartHdf5, HandBuiltX86LongDoubleExactAndWideningRefuseAllowNarrowingReportsDelta) {
  const std::string path = temp_path("typed_x86_longdouble.h5");
  std::remove(path.c_str());
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov, "");
  ASSERT_TRUE(pw.ok());
  // A real field has to exist first so finish_point()'s own checks pass
  // and the file is a legitimate restart point; the hand-built dataset
  // below is added directly afterward, under its own name.
  ASSERT_TRUE(pw.value()->writer().put_scalar("placeholder", std::int32_t{0}).ok());
  ASSERT_TRUE(pw.value()->finish().ok());

  // value = 1 + 2^-63, unbiased exponent 0 -> biased 16383.
  edit_file_directly(path, [](hid_t file) {
    write_hand_built_x86_long_double(file, "scalars/x86ld",
                                     0x8000000000000001ULL, 16383,
                                     /*sign=*/false);
  });

  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());

#if LDBL_MANT_DIG == 64
  // This build's own `long double` IS the x86 80-bit extended type the
  // dataset was hand-built as (x86-64 Linux: 64-bit significand, 15-bit
  // exponent, explicit leading bit) -- so the three conversion policies
  // all see the SAME type and every read is exact: the value 1 + 2^-63
  // is representable and comes back unchanged, and AllowNarrowing has
  // nothing to narrow (converted_any false, zero reported change). The
  // 53-bit branch below covers the arm64 macOS case.
  const long double expected = 1.0L + std::ldexp(1.0L, -63);
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar("x86ld", restored);
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
    EXPECT_EQ(restored, expected);
  }
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar(
        "x86ld", restored, fuse::ReadConversion::Widening);
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
    EXPECT_EQ(restored, expected);
  }
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar(
        "x86ld", restored, fuse::ReadConversion::AllowNarrowing);
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
    EXPECT_FALSE(rr.value().converted_any);
    EXPECT_EQ(rr.value().max_absolute_change, 0.0);
    EXPECT_EQ(restored, expected);
  }
#else
  // Exact: this build's own `long double` (53-bit) is not the SAME type
  // as the file's own hand-built 80-bit one, by every one of the width
  // checks `convert_one_element()` makes before it ever reaches
  // `ReadConversion` at all -- refuses.
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar("x86ld", restored);
    EXPECT_FALSE(rr.ok());
    EXPECT_EQ(rr.error().code, "E-RESTART-TYPE-MISMATCH");
  }
  // Widening: 64-bit significand does not fit this Mac's 53-bit one --
  // not an exact widening, refuses.
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar(
        "x86ld", restored, fuse::ReadConversion::Widening);
    EXPECT_FALSE(rr.ok());
    EXPECT_EQ(rr.error().code, "E-RESTART-TYPE-MISMATCH");
  }
  // AllowNarrowing: reads successfully, reporting the EXACT delta an
  // independent computation gives (ldexp(1.0, -63) -- the dropped low
  // 11 bits of the 64-bit significand, at unbiased exponent 0, computed
  // here via a DIFFERENT code path than fuse::detail::narrow_float()'s
  // own internal one).
  {
    long double restored = 0.0L;
    const auto rr = pr.value()->reader().get_scalar(
        "x86ld", restored, fuse::ReadConversion::AllowNarrowing);
    ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
    EXPECT_TRUE(rr.value().converted_any);
    const double independently_computed_delta = std::ldexp(1.0, -63);
    EXPECT_EQ(rr.value().max_absolute_change, independently_computed_delta);
    EXPECT_EQ(static_cast<double>(restored), 1.0);  // rounds down
  }
#endif
  std::remove(path.c_str());
}
