#include "fuse/restart_io/hdf5_restart_io.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fuse/model/element_type.hpp"
#include "fuse/model/float_bits.hpp"

namespace fuse::restart_io {

namespace {

/// RAII for an HDF5 identifier closed with `closer`. Every `hid_t` in
/// this file is wrapped in one of these -- a `Failure` return mid-function
/// must not leak an open group/dataset/dataspace, the same discipline
/// `fuse::Result` asks every OTHER boundary in this codebase to keep.
template <herr_t (*Closer)(hid_t)>
class Hid {
 public:
  Hid() = default;
  explicit Hid(hid_t id) : id_(id) {}
  Hid(const Hid&) = delete;
  Hid& operator=(const Hid&) = delete;
  Hid(Hid&& o) noexcept : id_(o.id_) { o.id_ = -1; }
  Hid& operator=(Hid&& o) noexcept {
    reset();
    id_ = o.id_;
    o.id_ = -1;
    return *this;
  }
  ~Hid() { reset(); }
  void reset() {
    if (id_ >= 0) Closer(id_);
    id_ = -1;
  }
  [[nodiscard]] hid_t get() const noexcept { return id_; }
  [[nodiscard]] bool valid() const noexcept { return id_ >= 0; }
  operator hid_t() const noexcept { return id_; }  // NOLINT: implicit

 private:
  hid_t id_ = -1;
};

using HFile = Hid<H5Fclose>;
using HGroup = Hid<H5Gclose>;
using HDataset = Hid<H5Dclose>;
using HDataspace = Hid<H5Sclose>;
using HAttr = Hid<H5Aclose>;
using HLcpl = Hid<H5Pclose>;
using HType = Hid<H5Tclose>;

/// A link-creation property list with intermediate-group creation on --
/// used for every `H5Dcreate2`/`H5Gcreate2` below, so a qualified name
/// like `"fields/dirichlet/u"` creates its own `fields/dirichlet/` group
/// on demand rather than requiring every caller to pre-create it.
[[nodiscard]] Result<HLcpl> make_lcpl() {
  HLcpl lcpl(H5Pcreate(H5P_LINK_CREATE));
  if (!lcpl.valid()) {
    return Failure{"E-RESTART-HDF5", "H5Pcreate(H5P_LINK_CREATE) failed"};
  }
  if (H5Pset_create_intermediate_group(lcpl, 1) < 0) {
    return Failure{"E-RESTART-HDF5",
                   "H5Pset_create_intermediate_group failed"};
  }
  return std::move(lcpl);  // nvcc does not implicitly move into Result<T>
}

[[nodiscard]] Result<bool> link_exists(hid_t loc, const std::string& path) {
  // H5Lexists on a multi-segment path needs every intermediate segment to
  // exist too, or it reports failure (negative) rather than "absent" --
  // walk the path component by component.
  std::string cur;
  std::size_t pos = 0;
  while (pos < path.size()) {
    const std::size_t slash = path.find('/', pos);
    const std::string seg = path.substr(pos, slash == std::string::npos
                                                  ? std::string::npos
                                                  : slash - pos);
    cur = cur.empty() ? seg : cur + "/" + seg;
    const htri_t e = H5Lexists(loc, cur.c_str(), H5P_DEFAULT);
    if (e < 0) return false;
    if (e == 0) return false;
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
  return true;
}

[[nodiscard]] Result<void> write_scalar_attr_u32(hid_t loc, const char* name,
                                                 std::uint32_t v) {
  HDataspace sp(H5Screate(H5S_SCALAR));
  HAttr a(H5Acreate2(loc, name, H5T_NATIVE_UINT32, sp, H5P_DEFAULT, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Acreate2 ") + name};
  if (H5Awrite(a, H5T_NATIVE_UINT32, &v) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Awrite ") + name};
  }
  return Result<void>::success();
}
[[nodiscard]] Result<void> write_scalar_attr_u64(hid_t loc, const char* name,
                                                 std::uint64_t v) {
  HDataspace sp(H5Screate(H5S_SCALAR));
  HAttr a(H5Acreate2(loc, name, H5T_NATIVE_UINT64, sp, H5P_DEFAULT, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Acreate2 ") + name};
  if (H5Awrite(a, H5T_NATIVE_UINT64, &v) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Awrite ") + name};
  }
  return Result<void>::success();
}
[[nodiscard]] Result<void> write_scalar_attr_str(hid_t loc, const char* name,
                                                 const std::string& v) {
  HDataspace sp(H5Screate(H5S_SCALAR));
  Hid<H5Tclose> t(H5Tcopy(H5T_C_S1));
  H5Tset_size(t, v.size() + 1);
  HAttr a(H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Acreate2 ") + name};
  if (H5Awrite(a, t, v.c_str()) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Awrite ") + name};
  }
  return Result<void>::success();
}

[[nodiscard]] Result<std::uint32_t> read_attr_u32(hid_t loc, const char* name) {
  if (H5Aexists(loc, name) <= 0) {
    return Failure{"E-RESTART-TRUNCATED",
                   std::string("missing attribute ") + name};
  }
  HAttr a(H5Aopen(loc, name, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Aopen ") + name};
  std::uint32_t v = 0;
  if (H5Aread(a, H5T_NATIVE_UINT32, &v) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Aread ") + name};
  }
  return v;
}
[[nodiscard]] Result<std::uint64_t> read_attr_u64(hid_t loc, const char* name) {
  if (H5Aexists(loc, name) <= 0) {
    return Failure{"E-RESTART-TRUNCATED",
                   std::string("missing attribute ") + name};
  }
  HAttr a(H5Aopen(loc, name, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Aopen ") + name};
  std::uint64_t v = 0;
  if (H5Aread(a, H5T_NATIVE_UINT64, &v) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Aread ") + name};
  }
  return v;
}
[[nodiscard]] Result<std::string> read_attr_str(hid_t loc, const char* name) {
  if (H5Aexists(loc, name) <= 0) {
    return Failure{"E-RESTART-TRUNCATED",
                   std::string("missing attribute ") + name};
  }
  HAttr a(H5Aopen(loc, name, H5P_DEFAULT));
  if (!a.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Aopen ") + name};
  Hid<H5Tclose> t(H5Aget_type(a));
  const std::size_t sz = H5Tget_size(t);
  std::vector<char> buf(sz + 1, '\0');
  if (H5Aread(a, t, buf.data()) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Aread ") + name};
  }
  return std::string(buf.data());
}

/// `fuse::restart_io`'s own typed-field backend (Stage B Slice 1b):
/// every value is written/read as an EXACT byte copy of its own native
/// representation, memory type identical to file type, for EVERY
/// `H5Dwrite`/`H5Dread` this backend ever issues -- see this file's own
/// top-of-file note on why: HDF5's own soft conversion cannot be
/// trusted uniformly across every float format this project supports
/// (measured: `H5T__conv_f_f()` refuses a `H5T_NORM_MSBSET` source/
/// destination with "normalization method not implemented yet"), so
/// this backend never asks HDF5 to convert anything -- all CROSS-TYPE
/// conversion is `fuse::detail::narrow_float()`/this file's own integer
/// conversion, never `H5Dread`'s own type-conversion machinery.
/// Predefined native types (`H5T_NATIVE_*`) are used for every INTEGER
/// width and `Bool`; every FLOAT type (`Binary16`/`BFloat16`/`Binary32`/
/// `Binary64`/`LongDouble`) goes through `make_derived_float_type()`
/// below -- ONE code path for all five, not a native-type shortcut for
/// the common case plus a separately-tested derived path for the
/// exotic one (see that function's own comment for why: a shortcut
/// using `H5T_NATIVE_LDOUBLE` cannot represent a FOREIGN file's `long
/// double`, e.g. x86-64 extended precision read on this Mac, docs/
/// dev/restart.md Section 11 item 6 -- measured to fail the identical
/// way the untrusted HDF5 soft-conversion above does).
/// Builds a user-defined HDF5 floating-point datatype matching `info`'s
/// OWN fields EXACTLY -- `significand_bits`/`exponent_bits`/
/// `exponent_bias`/`explicit_integer_bit`/`byte_size`, whatever they are.
/// Used for EVERY float `ElementType` (`Binary16`/`BFloat16`/`Binary32`/
/// `Binary64`/`LongDouble`), not only the two with no HDF5-predefined
/// equivalent -- `info` is NOT always this build's own canonical
/// metadata for its enumerator: `fuse::restart_io`'s own `read_and_
/// verify_type_metadata()` can hand this function an info built from a
/// FOREIGN file's own attributes (docs/dev/restart.md Section 11, item
/// 6 -- a hand-built x86-64-style 80-bit `long double`), which this
/// Mac's `H5T_NATIVE_LDOUBLE` cannot represent at all. ONE code path for
/// both cases, rather than a native-type shortcut for the common case
/// plus a separately-tested derived path for the exotic one, is what
/// closed a real, measured defect this round: `H5Dread()`/`H5Dwrite()`
/// against `H5T_NATIVE_LDOUBLE` as the MEMORY type while the FILE type
/// is a hand-built 80-bit MSBSET one asks HDF5 to CONVERT between them,
/// which fails the identical way `H5T__conv_f_f()` already does for any
/// other `H5T_NORM_MSBSET` pair ("normalization method not implemented
/// yet") -- this function makes the memory type and the file type
/// IDENTICAL in every case, so no conversion is ever requested.
///
/// PRECISION/ORDERING. HDF5's own `H5Tset_precision()` doc comment:
/// "When decreasing the precision of a floating point type, set the
/// locations and sizes of the sign, mantissa, and exponent fields
/// first" -- measured directly (docs/dev/restart.md Section 11) that
/// the CONVERSE is required for an INCREASE: growing size/precision
/// FIRST, then calling `H5Tset_fields()` once there is room. This
/// function therefore picks a base type (`H5T_IEEE_F32LE`, 32-bit
/// precision, if `info`'s own needed precision fits within it; `H5T_
/// IEEE_F64LE`, 64 bits, otherwise) and reorders size/precision/fields
/// in whichever direction it is really moving. `precision` itself is
/// `stored_mantissa_bits + exponent_bits + 1` (the sign bit) --
/// `stored_mantissa_bits` is `significand_bits` ITSELF for an explicit
/// leading bit, `significand_bits - 1` for an implicit one (the
/// implicit bit occupies no on-disk storage at all) -- NOT `significand_
/// bits + exponent_bits + 1`, which overcounts by exactly one stored
/// bit whenever the leading bit is implicit (every format here except
/// x86 extended): a overcounted `H5Tset_precision()` happened to be
/// silently corrected by `H5Tset_size()`'s own doc-stated auto-shrink
/// ("if the significant bits still hang over the edge, the number of
/// significant bits is decreased") for every DEcrease case this project
/// tested, masking the defect until the INCREASE case (x86 extended, no
/// shrink to recover into) surfaced it outright -- measured directly,
/// not a hypothetical.
[[nodiscard]] Result<HType> make_derived_float_type(const fuse::ElementTypeInfo& info) {
  const std::size_t stored_mantissa_bits =
      info.explicit_integer_bit ? info.significand_bits
                                : info.significand_bits - 1;
  const std::size_t precision = stored_mantissa_bits + info.exponent_bits + 1;
  const std::size_t mpos = 0;
  const std::size_t epos = stored_mantissa_bits;
  const std::size_t spos = stored_mantissa_bits + info.exponent_bits;
  const H5T_norm_t norm = info.explicit_integer_bit ? H5T_NORM_MSBSET
                                                     : H5T_NORM_IMPLIED;

  const bool use_f32_base = precision <= 32;
  HType t(H5Tcopy(use_f32_base ? H5T_IEEE_F32LE : H5T_IEEE_F64LE));
  if (!t.valid()) return Failure{"E-RESTART-HDF5", "H5Tcopy failed for " + info.name};
  const std::size_t base_precision = use_f32_base ? 32 : 64;

  auto set_fields = [&]() -> Result<void> {
    if (H5Tset_fields(t, spos, epos, info.exponent_bits, mpos,
                      stored_mantissa_bits) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Tset_fields failed for " + info.name};
    }
    if (H5Tset_ebias(t, static_cast<std::size_t>(info.exponent_bias)) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Tset_ebias failed for " + info.name};
    }
    if (H5Tset_norm(t, norm) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Tset_norm failed for " + info.name};
    }
    return Result<void>::success();
  };
  auto shrink_or_grow_size = [&]() -> Result<void> {
    if (H5Tset_precision(t, precision) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Tset_precision failed for " + info.name};
    }
    if (H5Tset_size(t, info.byte_size) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Tset_size failed for " + info.name};
    }
    return Result<void>::success();
  };

  if (precision <= base_precision) {
    // Decrease (or no change): fields first, while the base's own wider
    // precision still has room for every field position.
    Result<void> r = set_fields();
    if (!r.ok()) return r.error();
    r = shrink_or_grow_size();
    if (!r.ok()) return r.error();
  } else {
    // Increase: grow size/precision first, THEN fields -- the opposite
    // order, required (measured directly, see this function's own doc
    // comment).
    Result<void> r = shrink_or_grow_size();
    if (!r.ok()) return r.error();
    r = set_fields();
    if (!r.ok()) return r.error();
  }
  if (H5Tset_order(t, H5T_ORDER_LE) < 0) {
    return Failure{"E-RESTART-HDF5", "H5Tset_order failed for " + info.name};
  }
  return std::move(t);  // nvcc does not implicitly move into Result<T>
}

[[nodiscard]] Result<HType> make_hdf5_type_for(const fuse::ElementTypeInfo& info) {
  using fuse::ElementType;
  switch (info.element_type) {
    case ElementType::Int8:   return HType(H5Tcopy(H5T_NATIVE_INT8));
    case ElementType::Int16:  return HType(H5Tcopy(H5T_NATIVE_INT16));
    case ElementType::Int32:  return HType(H5Tcopy(H5T_NATIVE_INT32));
    case ElementType::Int64:  return HType(H5Tcopy(H5T_NATIVE_INT64));
    case ElementType::UInt8:  return HType(H5Tcopy(H5T_NATIVE_UINT8));
    case ElementType::UInt16: return HType(H5Tcopy(H5T_NATIVE_UINT16));
    case ElementType::UInt32: return HType(H5Tcopy(H5T_NATIVE_UINT32));
    case ElementType::UInt64: return HType(H5Tcopy(H5T_NATIVE_UINT64));
    case ElementType::Bool:   return HType(H5Tcopy(H5T_NATIVE_UINT8));
    case ElementType::Binary16:
    case ElementType::BFloat16:
    case ElementType::Binary32:
    case ElementType::Binary64:
    case ElementType::LongDouble:
      return make_derived_float_type(info);
  }
  return Failure{"E-RESTART-HDF5",
                 "make_hdf5_type_for: unhandled ElementType '" + info.name + "'"};
}

/// Writes every metadata fact `fuse::ElementTypeInfo` carries as its own
/// HDF5 attribute on `loc` (a dataset) -- docs/dev/restart.md Section 11,
/// item 5: "element-type name, byte size, for floating types significand
/// and exponent widths and exponent bias, byte order". `is_signed` is
/// meaningful only for an integer type; written as 0 for every other
/// kind (float, bool) so the attribute always exists and a reader never
/// needs a separate "does this attribute apply" branch.
[[nodiscard]] Result<void> write_type_metadata(hid_t loc,
                                               const fuse::ElementTypeInfo& info) {
  Result<void> r = write_scalar_attr_str(loc, "element_type", info.name);
  if (!r.ok()) return r;
  r = write_scalar_attr_u32(loc, "byte_size", static_cast<std::uint32_t>(info.byte_size));
  if (!r.ok()) return r;
  r = write_scalar_attr_u32(loc, "significand_bits",
                            static_cast<std::uint32_t>(info.significand_bits));
  if (!r.ok()) return r;
  r = write_scalar_attr_u32(loc, "exponent_bits",
                            static_cast<std::uint32_t>(info.exponent_bits));
  if (!r.ok()) return r;
  r = write_scalar_attr_u64(loc, "exponent_bias",
                            static_cast<std::uint64_t>(info.exponent_bias));
  if (!r.ok()) return r;
  r = write_scalar_attr_str(loc, "byte_order", "little");
  if (!r.ok()) return r;
  const std::uint32_t is_signed = info.is_signed_integer ? 1 : 0;
  return write_scalar_attr_u32(loc, "is_signed", is_signed);
}

/// Reads `loc`'s own `element_type`-family attributes AND independently
/// queries its ACTUAL HDF5 datatype (`H5Dget_type`), then checks the
/// two agree -- docs/dev/restart.md Section 11, item 5's own "the HDF5
/// dataset datatype itself must equal that type (verify the attribute
/// and the dataset type agree on read; disagreement refuses by name --
/// it means a corrupted or foreign file)".
///
/// RETURNS AN `ElementTypeInfo` BUILT FROM THE FILE'S OWN ATTRIBUTES,
/// NEVER `fuse::element_type_info()`'s global table for the matching
/// enumerator. This is deliberate, not an oversight: a restart point
/// written on a different platform (x86-64's 80-bit `long double`, read
/// on this Mac's 53-bit one, docs/dev/restart.md Section 11, item 6) is
/// internally self-consistent -- its OWN attributes agree with its OWN
/// actual HDF5 datatype -- but genuinely DIFFERS from what THIS build's
/// `element_type_info(ElementType::LongDouble)` says a `long double`
/// looks like. Comparing the attributes against the global table instead
/// of against the dataset's own real type would reject every such file
/// before `ReadConversion` is ever consulted, making `AllowNarrowing`
/// unable to do the ONE thing item 6 asks it to do. "Corrupted or
/// foreign" in the owner's own phrasing means a file whose OWN two
/// descriptions of itself (attribute vs. real HDF5 type) disagree --
/// not a file that disagrees with this reader's own platform.
///
/// An `element_type` NAME with no matching `ElementType` enumerator at
/// all (a newer file version using a kind this build has never heard of)
/// refuses by name (E-RESTART-UNKNOWN-ELEMENT-TYPE) rather than
/// guessing -- `fuse::ElementTypeInfo` needs SOME enumerator to tag the
/// returned info with, and there is no sense in which an unknown one
/// could serve that role.
[[nodiscard]] Result<fuse::ElementTypeInfo> read_and_verify_type_metadata(
    hid_t loc, const std::string& context) {
  const Result<std::string> name = read_attr_str(loc, "element_type");
  if (!name.ok()) return name.error();
  const auto maybe_type = fuse::element_type_from_name(name.value());
  if (!maybe_type.has_value()) {
    return Failure{"E-RESTART-UNKNOWN-ELEMENT-TYPE",
                   context + ": element_type attribute names '" + name.value() +
                       "', which this build does not recognise -- written by "
                       "a newer version of this format"};
  }

  const Result<std::uint32_t> byte_size = read_attr_u32(loc, "byte_size");
  if (!byte_size.ok()) return byte_size.error();
  const Result<std::uint32_t> sig_bits = read_attr_u32(loc, "significand_bits");
  if (!sig_bits.ok()) return sig_bits.error();
  const Result<std::uint32_t> exp_bits = read_attr_u32(loc, "exponent_bits");
  if (!exp_bits.ok()) return exp_bits.error();
  const Result<std::uint64_t> exp_bias = read_attr_u64(loc, "exponent_bias");
  if (!exp_bias.ok()) return exp_bias.error();
  const Result<std::uint32_t> is_signed_attr = read_attr_u32(loc, "is_signed");
  if (!is_signed_attr.ok()) return is_signed_attr.error();

  // Build the info THIS FILE describes, from its own attributes --
  // the canonical table entry for the same enumerator is consulted
  // nowhere in this function.
  fuse::ElementTypeInfo info;
  info.element_type = maybe_type.value();
  info.name = name.value();
  info.byte_size = byte_size.value();
  info.significand_bits = sig_bits.value();
  info.exponent_bits = exp_bits.value();
  info.exponent_bias = static_cast<std::int64_t>(exp_bias.value());
  info.is_bool = (maybe_type.value() == fuse::ElementType::Bool);
  info.is_float = (sig_bits.value() > 0 || exp_bits.value() > 0) && !info.is_bool;
  info.is_signed_integer = !info.is_float && !info.is_bool && is_signed_attr.value() != 0;
  info.is_unsigned_integer = !info.is_float && !info.is_bool && is_signed_attr.value() == 0;
  // explicit_integer_bit cannot be recovered from the attributes alone
  // (this project does not write one -- it is derived from H5Tget_norm()
  // below, where it matters: the cross-check against the dataset's own
  // actual HDF5 type).

  // Cross-check against the dataset's OWN actual HDF5 datatype, not only
  // the attributes -- an attribute is just more bytes in the same file; a
  // corruption (or a foreign WRITER that did not keep the two in step)
  // that changed one but not the other is exactly what this catches.
  Hid<H5Tclose> actual(H5Dget_type(loc));
  if (!actual.valid()) {
    return Failure{"E-RESTART-HDF5", context + ": H5Dget_type failed"};
  }
  const std::size_t actual_size = H5Tget_size(actual);
  if (actual_size != info.byte_size) {
    return Failure{"E-RESTART-TYPE-METADATA-MISMATCH",
                   context + ": dataset's own HDF5 type has size " +
                       std::to_string(actual_size) + ", attribute claims " +
                       std::to_string(info.byte_size) + " for '" +
                       name.value() + "'"};
  }
  const H5T_class_t cls = H5Tget_class(actual);
  if (info.is_float) {
    if (cls != H5T_FLOAT) {
      return Failure{"E-RESTART-TYPE-METADATA-MISMATCH",
                     context + ": attribute claims floating type '" +
                         name.value() + "' but the dataset's own HDF5 type "
                         "class is not H5T_FLOAT"};
    }
    size_t spos = 0, epos = 0, esize = 0, mpos = 0, msize = 0;
    if (H5Tget_fields(actual, &spos, &epos, &esize, &mpos, &msize) < 0) {
      return Failure{"E-RESTART-HDF5", context + ": H5Tget_fields failed"};
    }
    const H5T_norm_t norm = H5Tget_norm(actual);
    info.explicit_integer_bit = (norm == H5T_NORM_MSBSET);
    const std::size_t expected_stored_mantissa =
        info.explicit_integer_bit ? info.significand_bits
                                  : info.significand_bits - 1;
    const std::size_t actual_ebias = H5Tget_ebias(actual);
    if (esize != info.exponent_bits || msize != expected_stored_mantissa ||
        actual_ebias != static_cast<std::size_t>(info.exponent_bias)) {
      return Failure{"E-RESTART-TYPE-METADATA-MISMATCH",
                     context + ": dataset's own HDF5 float fields disagree "
                     "with its own element_type attribute ('" + name.value() + "')"};
    }
  } else if (!info.is_bool) {
    if (cls != H5T_INTEGER) {
      return Failure{"E-RESTART-TYPE-METADATA-MISMATCH",
                     context + ": attribute claims integer type '" +
                         name.value() + "' but the dataset's own HDF5 type "
                         "class is not H5T_INTEGER"};
    }
    const H5T_sign_t sign = H5Tget_sign(actual);
    const bool actual_signed = (sign == H5T_SGN_2);
    if (actual_signed != info.is_signed_integer) {
      return Failure{"E-RESTART-TYPE-METADATA-MISMATCH",
                     context + ": dataset's own HDF5 integer signedness "
                     "disagrees with its own element_type attribute ('" +
                         name.value() + "')"};
    }
  }
  return info;
}

/// `info.element_type == Bool`'s own extra invariant: every stored byte
/// must be exactly 0 or 1 (docs/dev/restart.md Section 11, item 3) --
/// checked unconditionally on read, independent of `ReadConversion`
/// (this is a corrupted-file detector, not a type-widening policy).
[[nodiscard]] Result<void> validate_bool_bytes(std::span<const std::uint8_t> bytes,
                                               const std::string& context) {
  for (const std::uint8_t b : bytes) {
    if (b != 0 && b != 1) {
      return Failure{"E-RESTART-BAD-BOOL-BYTE",
                     context + ": stored bool byte is " + std::to_string(b) +
                         ", neither 0 nor 1 -- the file is corrupted"};
    }
  }
  return Result<void>::success();
}

/// Decodes `raw` (exactly `info.byte_size` bytes, `info` an INTEGER or
/// `Bool` `ElementTypeInfo`) into a signed-or-unsigned 64-bit canonical
/// value -- the integer analogue of `fuse::detail::decode_float_bits()`.
struct DecodedInt {
  bool is_signed = false;
  std::int64_t s = 0;
  std::uint64_t u = 0;
};
[[nodiscard]] DecodedInt decode_int_bytes(std::span<const std::byte> raw,
                                          const fuse::ElementTypeInfo& info) {
  DecodedInt d;
  if (info.is_bool) {
    std::uint8_t v = 0;
    std::memcpy(&v, raw.data(), 1);
    d.is_signed = false;
    d.u = v;
    return d;
  }
  d.is_signed = info.is_signed_integer;
  if (d.is_signed) {
    switch (info.byte_size) {
      case 1: { std::int8_t v; std::memcpy(&v, raw.data(), 1); d.s = v; break; }
      case 2: { std::int16_t v; std::memcpy(&v, raw.data(), 2); d.s = v; break; }
      case 4: { std::int32_t v; std::memcpy(&v, raw.data(), 4); d.s = v; break; }
      default: { std::int64_t v; std::memcpy(&v, raw.data(), 8); d.s = v; break; }
    }
  } else {
    switch (info.byte_size) {
      case 1: { std::uint8_t v; std::memcpy(&v, raw.data(), 1); d.u = v; break; }
      case 2: { std::uint16_t v; std::memcpy(&v, raw.data(), 2); d.u = v; break; }
      case 4: { std::uint32_t v; std::memcpy(&v, raw.data(), 4); d.u = v; break; }
      default: { std::uint64_t v; std::memcpy(&v, raw.data(), 8); d.u = v; break; }
    }
  }
  return d;
}

/// Encodes `d` into `target`'s own native width, refusing
/// (E-RESTART-INTEGER-OVERFLOW) if the value does not fit -- "integers
/// that do not fit refuse even under AllowNarrowing: an integer value
/// change is not a rounding" (owner's own words, docs/dev/restart.md
/// Section 11) -- so this function takes no `ReadConversion` parameter at
/// all; the overflow check is unconditional. Signed -> unsigned is
/// refused unconditionally too (E-RESTART-SIGNED-TO-UNSIGNED), matching
/// `fuse::is_exact_widening()`'s own "never" rule -- `AllowNarrowing`
/// extends precision policy, not this safety rule.
[[nodiscard]] Result<void> encode_int_checked(const DecodedInt& d,
                                              const fuse::ElementTypeInfo& target,
                                              std::span<std::byte> out,
                                              const std::string& context) {
  if (d.is_signed && !target.is_signed_integer) {
    return Failure{"E-RESTART-SIGNED-TO-UNSIGNED",
                   context + ": refusing to convert a signed value to an "
                   "unsigned type '" + target.name + "'"};
  }
  if (target.is_signed_integer) {
    std::int64_t val;
    if (d.is_signed) {
      val = d.s;
    } else {
      if (d.u > static_cast<std::uint64_t>(INT64_MAX)) {
        return Failure{"E-RESTART-INTEGER-OVERFLOW",
                       context + ": value " + std::to_string(d.u) +
                           " does not fit in '" + target.name + "'"};
      }
      val = static_cast<std::int64_t>(d.u);
    }
    bool fits;
    switch (target.byte_size) {
      case 1: fits = (val >= -128 && val <= 127); break;
      case 2: fits = (val >= -32768 && val <= 32767); break;
      case 4: fits = (val >= -2147483648LL && val <= 2147483647LL); break;
      default: fits = true; break;
    }
    if (!fits) {
      return Failure{"E-RESTART-INTEGER-OVERFLOW",
                     context + ": value " + std::to_string(val) +
                         " does not fit in '" + target.name + "'"};
    }
    switch (target.byte_size) {
      case 1: { auto v = static_cast<std::int8_t>(val); std::memcpy(out.data(), &v, 1); break; }
      case 2: { auto v = static_cast<std::int16_t>(val); std::memcpy(out.data(), &v, 2); break; }
      case 4: { auto v = static_cast<std::int32_t>(val); std::memcpy(out.data(), &v, 4); break; }
      default: { std::memcpy(out.data(), &val, 8); break; }
    }
  } else {
    std::uint64_t val = d.u;  // d.is_signed is false here (checked above)
    bool fits;
    switch (target.byte_size) {
      case 1: fits = (val <= 255); break;
      case 2: fits = (val <= 65535); break;
      case 4: fits = (val <= 4294967295ULL); break;
      default: fits = true; break;
    }
    if (!fits) {
      return Failure{"E-RESTART-INTEGER-OVERFLOW",
                     context + ": value " + std::to_string(val) +
                         " does not fit in '" + target.name + "'"};
    }
    switch (target.byte_size) {
      case 1: { auto v = static_cast<std::uint8_t>(val); std::memcpy(out.data(), &v, 1); break; }
      case 2: { auto v = static_cast<std::uint16_t>(val); std::memcpy(out.data(), &v, 2); break; }
      case 4: { auto v = static_cast<std::uint32_t>(val); std::memcpy(out.data(), &v, 4); break; }
      default: { std::memcpy(out.data(), &val, 8); break; }
    }
  }
  return Result<void>::success();
}

/// Converts ONE element's worth of bytes from `stored` (`src`) to
/// `requested` (`dst`), per `conversion` -- the one dispatch every
/// `get_field_bytes()`/`get_scalar_bytes()` row goes through, whether
/// `stored == requested` (the fast, copy-only path every call takes
/// before Slice 1b existed) or not. Accumulates into `report` --
/// `changed`/`abs_change` are per-element; the caller aggregates
/// `max_absolute_change`/`changed_count`/`converted_any` across every
/// element a single call reads (docs/dev/restart.md Section 11's own
/// "measured over the values actually read", not merely declared from
/// the types).
[[nodiscard]] Result<void> convert_one_element(
    const fuse::ElementTypeInfo& from_info, std::span<const std::byte> src,
    fuse::ElementType requested, std::span<std::byte> dst,
    fuse::ReadConversion conversion, const std::string& context,
    bool& changed, double& abs_change) {
  changed = false;
  abs_change = 0.0;
  const fuse::ElementTypeInfo& to_info = fuse::element_type_info(requested);
  const bool identical_type =
      from_info.element_type == requested && from_info.byte_size == to_info.byte_size &&
      from_info.significand_bits == to_info.significand_bits &&
      from_info.exponent_bits == to_info.exponent_bits &&
      from_info.exponent_bias == to_info.exponent_bias;
  if (identical_type) {
    std::memcpy(dst.data(), src.data(), dst.size());
    return Result<void>::success();
  }
  if (conversion == fuse::ReadConversion::Exact) {
    return Failure{"E-RESTART-TYPE-MISMATCH",
                   context + ": stored as '" + from_info.name +
                       "', requested as '" + to_info.name +
                       "' -- ReadConversion::Exact requires an identical "
                       "type (pass ReadConversion::Widening or "
                       "AllowNarrowing to permit a conversion)"};
  }
  const bool is_widening_exact = fuse::is_exact_widening(from_info, to_info);

  if (conversion == fuse::ReadConversion::Widening && !is_widening_exact) {
    return Failure{"E-RESTART-TYPE-MISMATCH",
                   context + ": stored as '" + from_info.name +
                       "', requested as '" + to_info.name + "' is not an "
                       "exact widening -- ReadConversion::Widening does "
                       "not permit this conversion (pass AllowNarrowing "
                       "if a lossy conversion is acceptable)"};
  }

  if (from_info.is_float && to_info.is_float) {
    const fuse::detail::NarrowingResult nr =
        fuse::detail::narrow_float(src, from_info, to_info);
    fuse::detail::encode_float_bits(nr.decoded, to_info, dst);
    changed = nr.changed;
    abs_change = nr.absolute_change;
    return Result<void>::success();
  }

  if ((from_info.is_signed_integer || from_info.is_unsigned_integer ||
      from_info.is_bool) &&
      (to_info.is_signed_integer || to_info.is_unsigned_integer)) {
    const DecodedInt d = decode_int_bytes(src, from_info);
    // Integer -> integer is EXACT whenever is_exact_widening() said so
    // (Widening path, already checked above) and otherwise may still
    // fit under AllowNarrowing -- encode_int_checked() itself decides,
    // and an integer value change from narrowing is refused there, not
    // reported (owner's own rule: "an integer value change is not a
    // rounding").
    return encode_int_checked(d, to_info, dst, context);
  }

  if ((from_info.is_signed_integer || from_info.is_unsigned_integer ||
      from_info.is_bool) &&
      to_info.is_float) {
    // Only reached when is_exact_widening() already said this exact
    // (int,float) pair is covered -- never reached under AllowNarrowing
    // alone, because integer -> float is not in this project's own
    // AllowNarrowing scope (docs/dev/restart.md Section 11's own open
    // question on this gap). conversion must therefore be Widening here.
    if (conversion != fuse::ReadConversion::Widening) {
      return Failure{"E-RESTART-TYPE-MISMATCH",
                     context + ": integer '" + from_info.name +
                         "' -> float '" + to_info.name + "' is only "
                         "supported under ReadConversion::Widening, not "
                         "AllowNarrowing (no narrowing int->float "
                         "conversion is implemented)"};
    }
    const DecodedInt d = decode_int_bytes(src, from_info);
    fuse::detail::DecodedFloat fv;
    fv.sign = d.is_signed && d.s < 0;
    const std::uint64_t magnitude =
        d.is_signed ? static_cast<std::uint64_t>(fv.sign ? -d.s : d.s) : d.u;
    if (magnitude == 0) {
      fv.is_zero = true;
    } else {
      std::uint64_t m = magnitude;
      std::int64_t exp = 0;
      while (m > (std::uint64_t{1} << (to_info.significand_bits - 1))) {
        m >>= 1;
        ++exp;
      }
      while (m < (std::uint64_t{1} << (to_info.significand_bits - 1))) {
        m <<= 1;
        --exp;
      }
      fv.significand = m;
      fv.unbiased_exponent = exp;
    }
    fuse::detail::encode_float_bits(fv, to_info, dst);
    return Result<void>::success();
  }

  return Failure{"E-RESTART-TYPE-MISMATCH",
                 context + ": no conversion from '" + from_info.name +
                     "' to '" + to_info.name + "' is implemented"};
}

/// The write side of one restart point, scoped to the HDF5 group `group_`.
/// Every qualified name `put_field`/`put_scalar`/`put_scalar_rational`/
/// `put_blob` is given creates its own namespaced dataset path below this
/// group (`fields/<name>/...`, `scalars/<name>`, `scalars_rational/<name>`,
/// `blobs/<name>/...`), so a scalar and a field may share a bare name with
/// no collision.
class Hdf5GroupWriter final : public RestartWriter {
 public:
  explicit Hdf5GroupWriter(hid_t group) : group_(group) {}

  Result<void> put_field_bytes(std::string_view name,
                               std::span<const std::int64_t> ids,
                               fuse::ElementType type,
                               std::span<const std::byte> values,
                               int components,
                               std::int64_t declared_global_size) override {
    const std::string key(name);
    if (ids_by_name_.count(key) != 0) {
      return Failure{"E-RESTART-DUPLICATE-CALL",
                     "put_field('" + key + "') called more than once for "
                     "this restart point"};
    }
    const fuse::ElementTypeInfo& info = fuse::element_type_info(type);
    if (static_cast<std::int64_t>(ids.size()) * components *
            static_cast<std::int64_t>(info.byte_size) !=
        static_cast<std::int64_t>(values.size())) {
      return Failure{"E-RESTART-FIELD-SHAPE",
                     "put_field('" + key + "'): ids.size()*components*"
                     "byte_size != values.size()"};
    }
    {
      std::set<std::int64_t> uniq(ids.begin(), ids.end());
      if (uniq.size() != ids.size()) {
        return Failure{"E-RESTART-DUPLICATE-ID",
                       "put_field('" + key + "'): duplicate global id in "
                       "one call"};
      }
    }
    if (static_cast<std::int64_t>(ids.size()) != declared_global_size) {
      // Single-writer case (Slice 1, no MPI): this ONE call is the whole
      // graph's contribution, so "coverage of the declared global size"
      // reduces to "this call's own count equals it" -- the general,
      // multi-caller version of this check is finish_point()'s own job
      // (owner decision 2), re-run below for defense in depth even
      // though at a single rank it can only re-confirm this one.
      return Failure{"E-RESTART-BAD-NUMBERING",
                     "put_field('" + key + "'): declared_global_size=" +
                         std::to_string(declared_global_size) +
                         " but this call reported " +
                         std::to_string(ids.size()) + " ids"};
    }

    const std::string ids_path = "fields/" + key + "/ids";
    const std::string val_path = "fields/" + key + "/values";
    const Result<HLcpl> lcpl = make_lcpl();
    if (!lcpl.ok()) return lcpl.error();

    {
      const hsize_t dims[1] = {ids.size()};
      HDataspace sp(H5Screate_simple(1, dims, nullptr));
      HDataset d(H5Dcreate2(group_, ids_path.c_str(), H5T_NATIVE_INT64, sp,
                            lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
      if (!d.valid()) {
        return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + ids_path};
      }
      if (!ids.empty() &&
          H5Dwrite(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  ids.data()) < 0) {
        return Failure{"E-RESTART-HDF5", "H5Dwrite " + ids_path};
      }
    }
    {
      const Result<HType> h5type = make_hdf5_type_for(info);
      if (!h5type.ok()) return h5type.error();
      const hsize_t dims[2] = {ids.size(),
                               static_cast<hsize_t>(components)};
      HDataspace sp(H5Screate_simple(2, dims, nullptr));
      HDataset d(H5Dcreate2(group_, val_path.c_str(), h5type.value(), sp,
                            lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
      if (!d.valid()) {
        return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + val_path};
      }
      if (!values.empty() &&
          H5Dwrite(d, h5type.value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  values.data()) < 0) {
        return Failure{"E-RESTART-HDF5", "H5Dwrite " + val_path};
      }
      const Result<void> mr = write_type_metadata(d, info);
      if (!mr.ok()) return mr;
    }
    ids_by_name_[key] = std::vector<std::int64_t>(ids.begin(), ids.end());
    declared_size_by_name_[key] = declared_global_size;
    return Result<void>::success();
  }

  Result<void> put_scalar_bytes(std::string_view name, fuse::ElementType type,
                                std::span<const std::byte> value) override {
    const fuse::ElementTypeInfo& info = fuse::element_type_info(type);
    const std::string path = "scalars/" + std::string(name);
    const Result<HType> h5type = make_hdf5_type_for(info);
    if (!h5type.ok()) return h5type.error();
    const Result<HLcpl> lcpl = make_lcpl();
    if (!lcpl.ok()) return lcpl.error();
    HDataspace sp(H5Screate(H5S_SCALAR));
    HDataset d(H5Dcreate2(group_, path.c_str(), h5type.value(), sp,
                          lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + path};
    if (H5Dwrite(d, h5type.value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                value.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dwrite " + path};
    }
    return write_type_metadata(d, info);
  }

  Result<void> put_scalar_rational(std::string_view name,
                                   const Rational& value) override {
    const std::string path = "scalars_rational/" + std::string(name);
    const std::int64_t nd[2] = {value.num(), value.den()};
    const Result<HLcpl> lcpl = make_lcpl();
    if (!lcpl.ok()) return lcpl.error();
    const hsize_t dims[1] = {2};
    HDataspace sp(H5Screate_simple(1, dims, nullptr));
    HDataset d(H5Dcreate2(group_, path.c_str(), H5T_NATIVE_INT64, sp,
                          lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + path};
    if (H5Dwrite(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, nd) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dwrite " + path};
    }
    return Result<void>::success();
  }

  Result<void> put_blob(std::string_view name, std::span<const std::byte> bytes,
                        std::uint32_t version) override {
    const std::string path = "blobs/" + std::string(name) + "/bytes";
    const Result<HLcpl> lcpl = make_lcpl();
    if (!lcpl.ok()) return lcpl.error();
    const hsize_t dims[1] = {bytes.empty() ? 1 : bytes.size()};
    HDataspace sp(H5Screate_simple(1, dims, nullptr));
    HDataset d(H5Dcreate2(group_, path.c_str(), H5T_NATIVE_UINT8, sp,
                          lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + path};
    if (!bytes.empty() &&
        H5Dwrite(d, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                bytes.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dwrite " + path};
    }
    const Result<void> vr = write_scalar_attr_u32(d, "version", version);
    if (!vr.ok()) return vr;
    const Result<void> lr =
        write_scalar_attr_u64(d, "length", static_cast<std::uint64_t>(bytes.size()));
    if (!lr.ok()) return lr;
    return Result<void>::success();
  }

  Result<void> finish_point() override {
    // Owner decision 2's own batched check: ONE combined sort over every
    // field's own (name, id) pairs. At a single rank (Slice 1, no MPI)
    // every field was reported by exactly one caller (this object), so
    // this re-confirms what put_field() already checked rather than
    // discovering anything new -- Slice 2 extends this into a single
    // collective Allgatherv over the SAME combined buffer instead of
    // replacing it (docs/dev/restart.md Section 7(iv)).
    std::vector<std::pair<std::string, std::int64_t>> combined;
    for (const auto& [name, ids] : ids_by_name_) {
      for (const std::int64_t id : ids) combined.emplace_back(name, id);
    }
    std::sort(combined.begin(), combined.end());
    for (std::size_t i = 1; i < combined.size(); ++i) {
      if (combined[i] == combined[i - 1]) {
        return Failure{"E-RESTART-DUPLICATE-ID",
                       "finish_point(): duplicate (field, id) = ('" +
                           combined[i].first + "', " +
                           std::to_string(combined[i].second) + ")"};
      }
    }
    for (const auto& [name, declared] : declared_size_by_name_) {
      const auto it = ids_by_name_.find(name);
      const std::int64_t received =
          it == ids_by_name_.end() ? 0
                                   : static_cast<std::int64_t>(it->second.size());
      if (received != declared) {
        return Failure{"E-RESTART-BAD-NUMBERING",
                       "finish_point(): field '" + name + "' declared " +
                           std::to_string(declared) + " ids, received " +
                           std::to_string(received)};
      }
    }
    return Result<void>::success();
  }

  Result<std::vector<std::int64_t>> recorded_ids(
      std::string_view name) const override {
    const auto it = ids_by_name_.find(std::string(name));
    if (it == ids_by_name_.end()) {
      return Failure{"E-RESTART-MISSING-FIELD",
                     "recorded_ids: no put_field() call was made for '" +
                         std::string(name) + "'"};
    }
    return it->second;
  }

 private:
  hid_t group_;
  std::unordered_map<std::string, std::vector<std::int64_t>> ids_by_name_;
  std::unordered_map<std::string, std::int64_t> declared_size_by_name_;
};

/// The read side, mirroring `Hdf5GroupWriter`'s own namespacing.
class Hdf5GroupReader final : public RestartReader {
 public:
  explicit Hdf5GroupReader(hid_t group) : group_(group) {}

  Result<fuse::ReadConversionReport> get_field_bytes(
      std::string_view name, std::span<const std::int64_t> ids,
      fuse::ElementType requested_type, std::span<std::byte> out,
      int components, fuse::ReadConversion conversion) override {
    const std::string key(name);
    const std::string ids_path = "fields/" + key + "/ids";
    const std::string val_path = "fields/" + key + "/values";
    const Result<bool> exists = link_exists(group_, ids_path);
    if (!exists.ok() || !exists.value()) {
      return Failure{"E-RESTART-MISSING-FIELD",
                     "get_field: no field named '" + key + "' in this "
                     "restart point"};
    }
    HDataset idd(H5Dopen2(group_, ids_path.c_str(), H5P_DEFAULT));
    if (!idd.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + ids_path};
    HDataspace idsp(H5Dget_space(idd));
    const hssize_t n = H5Sget_simple_extent_npoints(idsp);
    std::vector<std::int64_t> file_ids(static_cast<std::size_t>(n));
    if (n > 0 && H5Dread(idd, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         file_ids.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dread " + ids_path};
    }
    HDataset vd(H5Dopen2(group_, val_path.c_str(), H5P_DEFAULT));
    if (!vd.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + val_path};

    const Result<fuse::ElementTypeInfo> stored_type_r =
        read_and_verify_type_metadata(vd, "get_field('" + key + "')");
    if (!stored_type_r.ok()) return stored_type_r.error();
    const fuse::ElementTypeInfo& stored_info = stored_type_r.value();
    const fuse::ElementTypeInfo& requested_info =
        fuse::element_type_info(requested_type);

    std::vector<std::byte> file_values(
        static_cast<std::size_t>(n) * static_cast<std::size_t>(components) *
        stored_info.byte_size);
    {
      const Result<HType> file_h5type = make_hdf5_type_for(stored_info);
      if (!file_h5type.ok()) return file_h5type.error();
      if (n > 0 &&
          H5Dread(vd, file_h5type.value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 file_values.data()) < 0) {
        return Failure{"E-RESTART-HDF5", "H5Dread " + val_path};
      }
    }
    if (stored_info.is_bool) {
      std::vector<std::uint8_t> as_u8(file_values.size());
      std::memcpy(as_u8.data(), file_values.data(), file_values.size());
      const Result<void> bv =
          validate_bool_bytes(as_u8, "get_field('" + key + "')");
      if (!bv.ok()) return bv.error();
    }

    std::unordered_map<std::int64_t, std::size_t> pos;
    pos.reserve(file_ids.size());
    for (std::size_t i = 0; i < file_ids.size(); ++i) pos[file_ids[i]] = i;
    if (static_cast<std::int64_t>(ids.size()) * components *
            static_cast<std::int64_t>(requested_info.byte_size) !=
        static_cast<std::int64_t>(out.size())) {
      return Failure{"E-RESTART-FIELD-SHAPE",
                     "get_field('" + key + "'): ids.size()*components*"
                     "byte_size != out.size()"};
    }

    fuse::ReadConversionReport report;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const auto it = pos.find(ids[i]);
      if (it == pos.end()) {
        return Failure{"E-RESTART-MISSING-ID",
                       "get_field('" + key + "'): id " +
                           std::to_string(ids[i]) +
                           " is not present in this restart point"};
      }
      for (int c = 0; c < components; ++c) {
        const std::size_t src_off =
            (it->second * static_cast<std::size_t>(components) +
             static_cast<std::size_t>(c)) *
            stored_info.byte_size;
        const std::size_t dst_off =
            (i * static_cast<std::size_t>(components) +
             static_cast<std::size_t>(c)) *
            requested_info.byte_size;
        bool changed = false;
        double abs_change = 0.0;
        const Result<void> cr = convert_one_element(
            stored_info,
            std::span<const std::byte>(file_values.data() + src_off,
                                       stored_info.byte_size),
            requested_type,
            std::span<std::byte>(out.data() + dst_off, requested_info.byte_size),
            conversion, "get_field('" + key + "')", changed, abs_change);
        if (!cr.ok()) return cr.error();
        if (changed) {
          report.converted_any = true;
          ++report.changed_count;
          report.max_absolute_change =
              std::max(report.max_absolute_change, abs_change);
        }
      }
    }
    return report;
  }

  Result<fuse::ReadConversionReport> get_scalar_bytes(
      std::string_view name, fuse::ElementType requested_type,
      std::span<std::byte> out, fuse::ReadConversion conversion) override {
    const std::string path = "scalars/" + std::string(name);
    const Result<bool> exists = link_exists(group_, path);
    if (!exists.ok() || !exists.value()) {
      return Failure{"E-RESTART-MISSING-SCALAR",
                     "get_scalar: no scalar named '" + std::string(name) +
                         "' in this restart point"};
    }
    HDataset d(H5Dopen2(group_, path.c_str(), H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + path};

    const Result<fuse::ElementTypeInfo> stored_type_r =
        read_and_verify_type_metadata(d, "get_scalar('" + std::string(name) + "')");
    if (!stored_type_r.ok()) return stored_type_r.error();
    const fuse::ElementTypeInfo& stored_info = stored_type_r.value();
    const fuse::ElementTypeInfo& requested_info =
        fuse::element_type_info(requested_type);

    std::vector<std::byte> file_value(stored_info.byte_size);
    const Result<HType> file_h5type = make_hdf5_type_for(stored_info);
    if (!file_h5type.ok()) return file_h5type.error();
    if (H5Dread(d, file_h5type.value(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
               file_value.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dread " + path};
    }
    if (stored_info.is_bool) {
      std::vector<std::uint8_t> as_u8(file_value.size());
      std::memcpy(as_u8.data(), file_value.data(), file_value.size());
      const Result<void> bv =
          validate_bool_bytes(as_u8, "get_scalar('" + std::string(name) + "')");
      if (!bv.ok()) return bv.error();
    }
    if (out.size() != requested_info.byte_size) {
      return Failure{"E-RESTART-FIELD-SHAPE",
                     "get_scalar('" + std::string(name) + "'): out size " +
                         std::to_string(out.size()) + " != requested type "
                         "byte size " + std::to_string(requested_info.byte_size)};
    }
    fuse::ReadConversionReport report;
    bool changed = false;
    double abs_change = 0.0;
    const Result<void> cr = convert_one_element(
        stored_info, file_value, requested_type, out, conversion,
        "get_scalar('" + std::string(name) + "')", changed, abs_change);
    if (!cr.ok()) return cr.error();
    if (changed) {
      report.converted_any = true;
      report.changed_count = 1;
      report.max_absolute_change = abs_change;
    }
    return report;
  }

  Result<Rational> get_scalar_rational(std::string_view name) override {
    const std::string path = "scalars_rational/" + std::string(name);
    const Result<bool> exists = link_exists(group_, path);
    if (!exists.ok() || !exists.value()) {
      return Failure{"E-RESTART-MISSING-SCALAR",
                     "get_scalar_rational: no scalar named '" +
                         std::string(name) + "' in this restart point"};
    }
    HDataset d(H5Dopen2(group_, path.c_str(), H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + path};
    std::int64_t nd[2] = {0, 1};
    if (H5Dread(d, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, nd) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dread " + path};
    }
    return Rational(nd[0], nd[1]);
  }

  Result<std::vector<std::byte>> get_blob(std::string_view name,
                                          std::uint32_t& version) override {
    const std::string path = "blobs/" + std::string(name) + "/bytes";
    const Result<bool> exists = link_exists(group_, path);
    if (!exists.ok() || !exists.value()) {
      return Failure{"E-RESTART-MISSING-BLOB",
                     "get_blob: no blob named '" + std::string(name) +
                         "' in this restart point"};
    }
    HDataset d(H5Dopen2(group_, path.c_str(), H5P_DEFAULT));
    if (!d.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + path};
    const Result<std::uint32_t> v = read_attr_u32(d, "version");
    if (!v.ok()) return v.error();
    version = v.value();
    const Result<std::uint64_t> len = read_attr_u64(d, "length");
    if (!len.ok()) return len.error();
    std::vector<std::byte> out(static_cast<std::size_t>(len.value()));
    if (!out.empty() &&
        H5Dread(d, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT,
               out.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dread " + path};
    }
    return out;
  }

  Result<bool> has_id(std::string_view name, std::int64_t id) override {
    const std::string ids_path = "fields/" + std::string(name) + "/ids";
    const Result<bool> exists = link_exists(group_, ids_path);
    if (!exists.ok() || !exists.value()) return false;
    HDataset idd(H5Dopen2(group_, ids_path.c_str(), H5P_DEFAULT));
    if (!idd.valid()) return Failure{"E-RESTART-HDF5", "H5Dopen2 " + ids_path};
    HDataspace idsp(H5Dget_space(idd));
    const hssize_t n = H5Sget_simple_extent_npoints(idsp);
    std::vector<std::int64_t> file_ids(static_cast<std::size_t>(n));
    if (n > 0 && H5Dread(idd, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                         file_ids.data()) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Dread " + ids_path};
    }
    return std::find(file_ids.begin(), file_ids.end(), id) != file_ids.end();
  }

 private:
  hid_t group_;
};

[[nodiscard]] std::string temp_name_for(const std::string& final_path) {
  return final_path + ".tmp";
}

/// `SeparateMeshAndPoints`: write directly to a fresh temp file, no prior
/// content to preserve. `GrowingFile`: owner decision 3's own
/// copy-append-rename -- copy whatever already exists at `path` (nothing,
/// for the first point) to the temp name, open THAT for read-write, add
/// the new group, close, then rename over `path`. A crash at any point
/// before the final rename leaves `path` exactly as it was; the rename
/// itself is the one atomic step (POSIX `rename()` within one
/// filesystem).
[[nodiscard]] Result<bool> file_exists(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::fclose(f);
  return true;
}

[[nodiscard]] Result<void> copy_file(const std::string& from, const std::string& to) {
  std::FILE* in = std::fopen(from.c_str(), "rb");
  if (in == nullptr) {
    return Failure{"E-RESTART-HDF5", "copy_file: cannot open " + from};
  }
  std::FILE* out = std::fopen(to.c_str(), "wb");
  if (out == nullptr) {
    std::fclose(in);
    return Failure{"E-RESTART-HDF5", "copy_file: cannot create " + to};
  }
  char buf[1 << 16];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) {
    if (std::fwrite(buf, 1, n, out) != n) {
      std::fclose(in);
      std::fclose(out);
      return Failure{"E-RESTART-HDF5", "copy_file: write failed to " + to};
    }
  }
  std::fclose(in);
  std::fclose(out);
  return Result<void>::success();
}

class PointWriterImpl final : public PointWriter {
 public:
  PointWriterImpl(HFile file, HGroup point_group, std::string temp_path,
                  std::string final_path, Hdf5GroupWriter inner)
      : file_(std::move(file)),
        point_group_(std::move(point_group)),
        temp_path_(std::move(temp_path)),
        final_path_(std::move(final_path)),
        inner_(std::move(inner)) {}

  RestartWriter& writer() override { return inner_; }

  Result<void> finish() override {
    const Result<void> fp = inner_.finish_point();
    if (!fp.ok()) return fp;
    point_group_.reset();
    file_.reset();
    if (std::rename(temp_path_.c_str(), final_path_.c_str()) != 0) {
      return Failure{"E-RESTART-HDF5",
                     "finish(): rename(" + temp_path_ + ", " + final_path_ +
                         ") failed"};
    }
    return Result<void>::success();
  }

 private:
  HFile            file_;
  HGroup           point_group_;
  std::string      temp_path_;
  std::string      final_path_;
  Hdf5GroupWriter  inner_;
};

}  // namespace

Result<std::unique_ptr<PointWriter>> create_restart_point(
    FileLayout layout, const std::string& path, const Provenance& provenance,
    const std::string& point_label) {
  const std::string temp_path = temp_name_for(path);
  bool reopening = false;
  if (layout == FileLayout::GrowingFile) {
    if (point_label.empty()) {
      return Failure{"E-RESTART-USAGE",
                     "create_restart_point(GrowingFile): point_label is "
                     "required"};
    }
    const Result<bool> exists = file_exists(path);
    if (!exists.ok()) return exists.error();
    if (exists.value()) {
      const Result<void> cp = copy_file(path, temp_path);
      if (!cp.ok()) return cp.error();
      reopening = true;
    }
  }

  HFile file;
  if (reopening) {
    file = HFile(H5Fopen(temp_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
  } else {
    file = HFile(H5Fcreate(temp_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                          H5P_DEFAULT));
  }
  if (!file.valid()) {
    return Failure{"E-RESTART-HDF5", "create_restart_point: cannot create/open " +
                                         temp_path};
  }

  // Provenance: written on the FILE's own root exactly once, for
  // SeparateMeshAndPoints every point file carries it fresh (cheap,
  // never contradicts itself since each point file is independent);
  // for GrowingFile it is written on the FIRST point only, and every
  // later point's own write checks (does not rewrite) it, per the
  // "mesh written once" rule applied to provenance generally -- a
  // SECOND write here would need to AGREE with the first, which a
  // mismatched call would silently paper over, so a re-open skips
  // writing it rather than risk two different values on the same file.
  if (!reopening) {
    Result<void> r = write_scalar_attr_u32(file, "format_version",
                                           provenance.format_version);
    if (!r.ok()) return r.error();
    r = write_scalar_attr_str(file, "fuse_version", provenance.fuse_version);
    if (!r.ok()) return r.error();
    // D62 (KNOWN_ISSUES.md): written as its own attribute, independent of
    // descriptor_hash, so open_restart_point() can refuse a descriptor-
    // version mismatch by name BEFORE ever comparing hashes -- see
    // Provenance::descriptor_version's own doc comment (fuse/restart_io/
    // hdf5_restart_io.hpp) for why the two checks are split this way.
    r = write_scalar_attr_u32(file, "descriptor_version", provenance.descriptor_version);
    if (!r.ok()) return r.error();
    r = write_scalar_attr_u64(file, "descriptor_hash", provenance.descriptor_hash);
    if (!r.ok()) return r.error();
    r = write_scalar_attr_u32(file, "written_by_rank_count",
                              provenance.written_by_rank_count);
    if (!r.ok()) return r.error();
    r = write_scalar_attr_u64(file, "mesh_checksum", provenance.mesh_checksum);
    if (!r.ok()) return r.error();
  }

  HGroup point_group;
  if (layout == FileLayout::SeparateMeshAndPoints) {
    point_group = HGroup(H5Gopen2(file, "/", H5P_DEFAULT));
  } else {
    const Result<HLcpl> lcpl = make_lcpl();
    if (!lcpl.ok()) return lcpl.error();
    const std::string group_path = "restart_points/" + point_label;
    point_group = HGroup(H5Gcreate2(file, group_path.c_str(), lcpl.value(),
                                    H5P_DEFAULT, H5P_DEFAULT));
  }
  if (!point_group.valid()) {
    return Failure{"E-RESTART-HDF5", "create_restart_point: cannot open/create "
                                         "the point's own group"};
  }

  Hdf5GroupWriter inner(point_group.get());
  return std::unique_ptr<PointWriter>(std::make_unique<PointWriterImpl>(
      std::move(file), std::move(point_group), temp_path, path, std::move(inner)));
}

namespace {
class PointReaderImpl final : public PointReader {
 public:
  PointReaderImpl(HFile file, HGroup group, Provenance prov, Hdf5GroupReader inner)
      : file_(std::move(file)),
        group_(std::move(group)),
        prov_(std::move(prov)),
        inner_(std::move(inner)) {}
  RestartReader& reader() override { return inner_; }
  const Provenance& provenance() const override { return prov_; }

 private:
  HFile            file_;
  HGroup           group_;
  Provenance       prov_;
  Hdf5GroupReader  inner_;
};
}  // namespace

Result<std::unique_ptr<PointReader>> open_restart_point(
    FileLayout layout, const std::string& path, const std::string& point_label) {
  const Result<bool> exists = file_exists(path);
  if (!exists.ok()) return exists.error();
  if (!exists.value()) {
    return Failure{"E-RESTART-TRUNCATED", "open_restart_point: " + path +
                                              " does not exist"};
  }
  HFile file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!file.valid()) {
    return Failure{"E-RESTART-TRUNCATED",
                   "open_restart_point: " + path +
                       " does not open as a valid HDF5 file"};
  }

  Provenance prov;
  Result<std::uint32_t> fv = read_attr_u32(file, "format_version");
  if (!fv.ok()) return fv.error();
  prov.format_version = fv.value();
  if (prov.format_version != kFormatVersion) {
    return Failure{"E-RESTART-VERSION",
                   "open_restart_point: format_version " +
                       std::to_string(prov.format_version) +
                       " is not the supported " +
                       std::to_string(kFormatVersion)};
  }
  Result<std::string> ver = read_attr_str(file, "fuse_version");
  if (!ver.ok()) return ver.error();
  prov.fuse_version = ver.value();
  // D62 (KNOWN_ISSUES.md): refused HERE, unconditionally, before
  // descriptor_hash is even read -- a descriptor-version mismatch means
  // the WIRE LAYOUT fuse::serialize()/deserialize() used to produce
  // descriptor_hash may itself disagree between the point's own writer
  // and this build, the same reason fuse::deserialize() refuses a
  // kGraphVersion mismatch before ever comparing content (fuse/model/
  // descriptor.hpp). Unlike descriptor_hash, no current graph is needed
  // to make this comparison -- fuse::kGraphVersion is a compile-time
  // constant of THIS build -- so this check belongs here, the same way
  // the pre-existing format_version check above does, rather than in the
  // caller-side fuse::check_restart_descriptor_hash() (fuse/runtime/
  // restart.hpp), which exists for the comparison that DOES need the
  // current graph.
  Result<std::uint32_t> dv = read_attr_u32(file, "descriptor_version");
  if (!dv.ok()) return dv.error();
  prov.descriptor_version = dv.value();
  if (prov.descriptor_version != fuse::kGraphVersion) {
    return Failure{"E-RESTART-DESCRIPTOR-VERSION",
                   "open_restart_point: this point's own descriptor_version " +
                       std::to_string(prov.descriptor_version) +
                       " is not the supported " +
                       std::to_string(fuse::kGraphVersion)};
  }
  Result<std::uint64_t> dh = read_attr_u64(file, "descriptor_hash");
  if (!dh.ok()) return dh.error();
  prov.descriptor_hash = dh.value();
  Result<std::uint32_t> rc = read_attr_u32(file, "written_by_rank_count");
  if (!rc.ok()) return rc.error();
  prov.written_by_rank_count = rc.value();
  Result<std::uint64_t> mc = read_attr_u64(file, "mesh_checksum");
  if (!mc.ok()) return mc.error();
  prov.mesh_checksum = mc.value();

  HGroup group;
  if (layout == FileLayout::SeparateMeshAndPoints) {
    group = HGroup(H5Gopen2(file, "/", H5P_DEFAULT));
  } else {
    if (point_label.empty()) {
      return Failure{"E-RESTART-USAGE",
                     "open_restart_point(GrowingFile): point_label is "
                     "required"};
    }
    const std::string group_path = "restart_points/" + point_label;
    if (H5Lexists(file, group_path.c_str(), H5P_DEFAULT) <= 0) {
      return Failure{"E-RESTART-MISSING-POINT",
                     "open_restart_point: no group '" + group_path +
                         "' in " + path};
    }
    group = HGroup(H5Gopen2(file, group_path.c_str(), H5P_DEFAULT));
  }
  if (!group.valid()) {
    return Failure{"E-RESTART-HDF5", "open_restart_point: cannot open the "
                                         "point's own group"};
  }

  Hdf5GroupReader inner(group.get());
  return std::unique_ptr<PointReader>(std::make_unique<PointReaderImpl>(
      std::move(file), std::move(group), prov, std::move(inner)));
}

Result<void> check_restart_descriptor_hash(const CouplingGraph& current_graph,
                                           const Provenance& provenance) {
  const std::uint64_t current_hash = restart_descriptor_hash(current_graph);
  if (provenance.descriptor_hash != current_hash) {
    return Failure{"E-RESTART-DESCRIPTOR-HASH",
                   "check_restart_descriptor_hash: this point's own "
                   "descriptor_hash " +
                       std::to_string(provenance.descriptor_hash) +
                       " does not match the current graph's " +
                       std::to_string(current_hash) +
                       " (same descriptor_version " +
                       std::to_string(provenance.descriptor_version) +
                       " -- the graph itself differs, not its wire format)"};
  }
  return Result<void>::success();
}

}  // namespace fuse::restart_io
