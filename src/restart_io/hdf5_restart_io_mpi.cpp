/// Restart track round 2, item 3 -- see include/fuse/restart_io/
/// hdf5_restart_io_mpi.hpp's own header comment for the design and this
/// round's own declared scope. This translation unit is compiled only
/// under FUSE_ENABLE_MPI, for the SAME reason collective_check_mpi.cpp
/// is a separate file from hdf5_restart_io.cpp (that file's own
/// src/CMakeLists.txt comment).
#include "fuse/restart_io/hdf5_restart_io_mpi.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fuse/model/element_type.hpp"
#include "fuse/restart_io/collective_check_mpi.hpp"

namespace fuse::restart_io {

namespace {

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
  /// Gives up ownership without closing, so a caller can close the
  /// handle itself and inspect the result (Hid::reset() discards it).
  [[nodiscard]] hid_t release() noexcept {
    const hid_t id = id_;
    id_ = -1;
    return id;
  }
  [[nodiscard]] bool valid() const noexcept { return id_ >= 0; }
  operator hid_t() const noexcept { return id_; }  // NOLINT: implicit

 private:
  hid_t id_ = -1;
};

using HFile = Hid<H5Fclose>;
using HGroup = Hid<H5Gclose>;
using HDataset = Hid<H5Dclose>;
using HDataspace = Hid<H5Sclose>;
using HProp = Hid<H5Pclose>;
using HAttr = Hid<H5Aclose>;
using HType = Hid<H5Tclose>;

/// This round's own declared scope (hdf5_restart_io_mpi.hpp's own header
/// comment): Binary64 and Int64 only. Returns the native HDF5 type and
/// element byte size, or a Failure naming the unsupported type.
[[nodiscard]] Result<std::pair<hid_t, std::size_t>> native_type_for(
    fuse::ElementType type) {
  switch (type) {
    case fuse::ElementType::Binary64:
      return std::pair<hid_t, std::size_t>{H5T_NATIVE_DOUBLE, sizeof(double)};
    case fuse::ElementType::Int64:
      return std::pair<hid_t, std::size_t>{H5T_NATIVE_INT64,
                                           sizeof(std::int64_t)};
    default:
      return Failure{"E-RESTART-TYPE-MISMATCH",
                     "collective restart writer supports only Binary64 and "
                     "Int64 this round (docs/dev/restart.md Section 4.3's "
                     "own declared scope) -- got element type " +
                         std::to_string(static_cast<int>(type))};
  }
}

[[nodiscard]] Result<HProp> make_lcpl() {
  HProp lcpl(H5Pcreate(H5P_LINK_CREATE));
  if (!lcpl.valid()) {
    return Failure{"E-RESTART-HDF5", "H5Pcreate(H5P_LINK_CREATE) failed"};
  }
  if (H5Pset_create_intermediate_group(lcpl, 1) < 0) {
    return Failure{"E-RESTART-HDF5", "H5Pset_create_intermediate_group failed"};
  }
  return std::move(lcpl);  // nvcc does not implicitly move into Result<T>
}


[[nodiscard]] Result<void> write_scalar_attr_str_collective(hid_t loc,
                                                             const char* name,
                                                             const std::string& value) {
  HType type(H5Tcopy(H5T_C_S1));
  if (!type.valid()) return Failure{"E-RESTART-HDF5", "H5Tcopy"};
  if (H5Tset_size(type, std::max<std::size_t>(value.size(), 1)) < 0) {
    return Failure{"E-RESTART-HDF5", "H5Tset_size"};
  }
  HDataspace space(H5Screate(H5S_SCALAR));
  if (!space.valid()) return Failure{"E-RESTART-HDF5", "H5Screate scalar"};
  HAttr attr(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT));
  if (!attr.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Acreate2 ") + name};
  if (H5Awrite(attr, type, value.c_str()) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Awrite ") + name};
  }
  return Result<void>::success();
}

template <class T>
[[nodiscard]] Result<void> write_scalar_attr_num_collective(hid_t loc,
                                                             const char* name,
                                                             T value,
                                                             hid_t native_type) {
  HDataspace space(H5Screate(H5S_SCALAR));
  if (!space.valid()) return Failure{"E-RESTART-HDF5", "H5Screate scalar"};
  HAttr attr(H5Acreate2(loc, name, native_type, space, H5P_DEFAULT, H5P_DEFAULT));
  if (!attr.valid()) return Failure{"E-RESTART-HDF5", std::string("H5Acreate2 ") + name};
  if (H5Awrite(attr, native_type, &value) < 0) {
    return Failure{"E-RESTART-HDF5", std::string("H5Awrite ") + name};
  }
  return Result<void>::success();
}

[[nodiscard]] Result<void> write_scalar_attr_u32_collective(hid_t loc, const char* name,
                                                             std::uint32_t value) {
  // H5T_NATIVE_UINT32/UINT64 are NOT compile-time constants (HDF5 resolves
  // them at H5open() time) -- `native_type` is a runtime parameter, never
  // a template non-type argument.
  return write_scalar_attr_num_collective<std::uint32_t>(loc, name, value,
                                                         H5T_NATIVE_UINT32);
}

[[nodiscard]] Result<void> write_scalar_attr_u64_collective(hid_t loc, const char* name,
                                                             std::uint64_t value) {
  return write_scalar_attr_num_collective<std::uint64_t>(loc, name, value,
                                                         H5T_NATIVE_UINT64);
}

/// The seven per-field type attributes the reader's own
/// read_and_verify_type_metadata() requires (hdf5_restart_io.cpp,
/// write_type_metadata). The collective writer was built against the
/// pre-typed reader and wrote none of them. Found by reading the reader
/// (read_and_verify_type_metadata opens "element_type" first), not by a
/// measurement: on the owner's machine the two earlier defects in this file
/// (KNOWN_ISSUES.md D75, items 1 and 2) refused every collectively written
/// file before the type check was reached, so this omission was never seen
/// as its own error. Attribute creation is collective in parallel HDF5;
/// every task calls this with the same values, as for the dataset itself.
[[nodiscard]] Result<void> write_type_metadata_collective(
    hid_t loc, const fuse::ElementTypeInfo& info) {
  Result<void> r = write_scalar_attr_str_collective(loc, "element_type", info.name);
  if (!r.ok()) return r;
  r = write_scalar_attr_u32_collective(loc, "byte_size",
                                       static_cast<std::uint32_t>(info.byte_size));
  if (!r.ok()) return r;
  r = write_scalar_attr_u32_collective(loc, "significand_bits",
                                       static_cast<std::uint32_t>(info.significand_bits));
  if (!r.ok()) return r;
  r = write_scalar_attr_u32_collective(loc, "exponent_bits",
                                       static_cast<std::uint32_t>(info.exponent_bits));
  if (!r.ok()) return r;
  r = write_scalar_attr_u64_collective(loc, "exponent_bias",
                                       static_cast<std::uint64_t>(info.exponent_bias));
  if (!r.ok()) return r;
  r = write_scalar_attr_str_collective(loc, "byte_order", "little");
  if (!r.ok()) return r;
  return write_scalar_attr_u32_collective(loc, "is_signed",
                                          info.is_signed_integer ? 1u : 0u);
}

/// One field's own collective write: `my_offset`'s own hyperslab of a
/// `declared_global_size`-long dataset, created collectively by every
/// task of `comm` (the SAME extent on every task, per this interface's
/// own contract), written collectively via `H5FD_MPIO_COLLECTIVE`.
/// `raw`'s own byte length is `local_count * components *
/// element_byte_size` -- the caller's own `put_field_bytes()` already
/// checked this against `ids.size()`.
[[nodiscard]] Result<void> write_one_collective_dataset(
    hid_t group, MPI_Comm comm, const std::string& path, hid_t native_type,
    std::int64_t declared_global_size, int components,
    std::int64_t my_offset, std::int64_t local_count, const void* raw,
    const fuse::ElementTypeInfo* type_info) {
  const Result<HProp> lcpl = make_lcpl();
  if (!lcpl.ok()) return lcpl.error();
  const hsize_t dims[2] = {static_cast<hsize_t>(declared_global_size),
                           static_cast<hsize_t>(components)};
  const int rank = components > 1 ? 2 : 1;
  HDataspace file_space(H5Screate_simple(rank, dims, nullptr));
  if (!file_space.valid()) {
    return Failure{"E-RESTART-HDF5", "H5Screate_simple (file) " + path};
  }
  // Collective: every task calls H5Dcreate2 with the IDENTICAL extent,
  // type and path -- HDF5's own metadata-operations-are-collective
  // requirement under the MPI-IO file driver.
  HDataset dset(H5Dcreate2(group, path.c_str(), native_type, file_space,
                          lcpl.value(), H5P_DEFAULT, H5P_DEFAULT));
  if (!dset.valid()) return Failure{"E-RESTART-HDF5", "H5Dcreate2 " + path};
  if (type_info != nullptr) {
    const Result<void> mr = write_type_metadata_collective(dset, *type_info);
    if (!mr.ok()) return mr;
  }

  HDataspace this_task_file_space(H5Dget_space(dset));
  if (!this_task_file_space.valid()) {
    return Failure{"E-RESTART-HDF5", "H5Dget_space " + path};
  }
  HDataspace mem_space;
  if (local_count > 0) {
    const hsize_t start[2] = {static_cast<hsize_t>(my_offset), 0};
    const hsize_t count[2] = {static_cast<hsize_t>(local_count),
                              static_cast<hsize_t>(components)};
    if (H5Sselect_hyperslab(this_task_file_space, H5S_SELECT_SET, start,
                            nullptr, count, nullptr) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Sselect_hyperslab " + path};
    }
    mem_space = HDataspace(H5Screate_simple(rank, count, nullptr));
  } else {
    // This task owns no rows of this field this point -- participates
    // in the SAME collective H5Dwrite call with a zero-size selection
    // (docs/dev/restart.md Section 4.3's own "replicated or absent
    // data" pattern, generalised to "absent FOR THIS TASK"), never
    // skips the call outright.
    if (H5Sselect_none(this_task_file_space) < 0) {
      return Failure{"E-RESTART-HDF5", "H5Sselect_none " + path};
    }
    mem_space = HDataspace(H5Screate(H5S_NULL));
  }
  if (!mem_space.valid()) {
    return Failure{"E-RESTART-HDF5", "H5Screate (mem) " + path};
  }
  HProp dxpl(H5Pcreate(H5P_DATASET_XFER));
  if (!dxpl.valid()) return Failure{"E-RESTART-HDF5", "H5Pcreate (dxpl) " + path};
  if (H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE) < 0) {
    return Failure{"E-RESTART-HDF5", "H5Pset_dxpl_mpio " + path};
  }
  if (H5Dwrite(dset, native_type, mem_space, this_task_file_space, dxpl,
              raw) < 0) {
    return Failure{"E-RESTART-HDF5", "H5Dwrite (collective) " + path};
  }
  (void)comm;
  return Result<void>::success();
}

/// The collective writer. Every task's own `put_field()` call supplies
/// ONLY its own disjoint rows; `declared_global_size` is the FIELD's
/// own total across every task (not this task's own local count, unlike
/// the single-task `Hdf5GroupWriter` -- see this file's own header
/// comment). Offsets are computed with `MPI_Exscan` the first time a
/// field name is seen by this task (every task calls `put_field()` for
/// the SAME set of field names, in principle at different local counts,
/// including zero).
class CollectivePointWriterImpl final : public PointWriter, public RestartWriter {
 public:
  CollectivePointWriterImpl(HFile file, HGroup group, MPI_Comm comm,
                            std::string temp_path, std::string final_path)
      : file_(std::move(file)),
        group_(std::move(group)),
        comm_(comm),
        temp_path_(std::move(temp_path)),
        final_path_(std::move(final_path)) {}

  RestartWriter& writer() override { return *this; }

  [[nodiscard]] static Result<void> scalar_unsupported(std::string_view name) {
    return Failure{"E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED",
                   "collective restart writer does not support scalars, "
                   "rationals, or blobs this round ('" + std::string(name) +
                   "') -- HDF5 metadata operations are collective-required "
                   "even for a value one task alone would host; see the "
                   "delivered report's own open_questions"};
  }

  Result<void> put_field_bytes(std::string_view name,
                               std::span<const std::int64_t> ids,
                               fuse::ElementType type,
                               std::span<const std::byte> values,
                               int components,
                               std::int64_t declared_global_size) override {
    const std::string key(name);
    if (seen_.count(key) != 0) {
      return Failure{"E-RESTART-DUPLICATE-CALL",
                     "put_field('" + key + "') called more than once (by "
                     "this task) for this restart point"};
    }
    seen_.insert(key);
    const Result<std::pair<hid_t, std::size_t>> nt = native_type_for(type);
    if (!nt.ok()) return nt.error();
    const auto [native_type, byte_size] = nt.value();
    const std::int64_t local_count = static_cast<std::int64_t>(ids.size());
    if (local_count * components * static_cast<std::int64_t>(byte_size) !=
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
                       "one task's own call"};
      }
    }
    std::int64_t my_offset = 0;
    if (MPI_Exscan(&local_count, &my_offset, 1, MPI_INT64_T, MPI_SUM, comm_) !=
        MPI_SUCCESS) {
      return Failure{"E-RESTART-HDF5", "MPI_Exscan for '" + key + "'"};
    }
    int rank_in_comm = 0;
    MPI_Comm_rank(comm_, &rank_in_comm);
    if (rank_in_comm == 0) my_offset = 0;  // MPI_Exscan leaves rank 0 undefined.

    const Result<void> wr = write_one_collective_dataset(
        group_, comm_, "fields/" + key + "/values", native_type,
        declared_global_size, components, my_offset, local_count,
        values.data(), &fuse::element_type_info(type));
    if (!wr.ok()) return wr;

    const Result<void> wi = write_one_collective_dataset(
        group_, comm_, "fields/" + key + "/ids", H5T_NATIVE_INT64,
        declared_global_size, 1, my_offset, local_count, ids.data(),
        nullptr);  // the ids dataset carries no type attributes, as in the serial writer
    if (!wi.ok()) return wi;

    for (std::int64_t id : ids) local_ids_.emplace_back(key, id);
    ids_by_name_[key] = std::vector<std::int64_t>(ids.begin(), ids.end());
    local_declared_sizes_[key] = declared_global_size;
    return Result<void>::success();
  }

  Result<void> put_scalar_bytes(std::string_view name, fuse::ElementType,
                                std::span<const std::byte>) override {
    return scalar_unsupported(name);
  }

  Result<void> put_scalar_rational(std::string_view name, const Rational&) override {
    return scalar_unsupported(name);
  }

  Result<void> put_blob(std::string_view name, std::span<const std::byte>,
                        std::uint32_t) override {
    return scalar_unsupported(name);
  }

  /// The LOCAL (this task's own) half of owner decision 2's batched
  /// check -- already folded into put_field_bytes() above (the
  /// per-call duplicate-id-within-one-call check) for this writer, so
  /// there is nothing further to do here; the GLOBAL, cross-task half
  /// is finish() below's own collective_duplicate_and_coverage_check()
  /// call, run once after every put_field() call for this point has
  /// returned, exactly matching RestartWriter::finish_point()'s own
  /// header contract.
  Result<void> finish_point() override { return Result<void>::success(); }

  Result<std::vector<std::int64_t>> recorded_ids(
      std::string_view name) const override {
    const auto it = ids_by_name_.find(std::string(name));
    if (it == ids_by_name_.end()) {
      return Failure{"E-RESTART-NO-RECORDED-IDS",
                     "recorded_ids('" + std::string(name) + "'): this "
                     "writer never recorded a put_field() call under that "
                     "name"};
    }
    return it->second;
  }

  Result<void> finish() override {
    const Result<void> check =
        collective_duplicate_and_coverage_check(MPI_Comm_c2f(comm_), local_ids_,
                                                local_declared_sizes_);
    if (!check.ok()) return check;

    // Collective rename-on-success (docs/dev/restart.md Section 4.4's
    // own replicated-write convention): every task agrees every task's
    // own put_field() calls succeeded BEFORE anyone renames, rank 0
    // alone performs the rename, then every task agrees the rename is
    // visible before returning.
    int local_ok = 1;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm_);
    if (!global_ok) {
      return Failure{"E-RESTART-HDF5",
                     "finish(): a peer task did not reach finish() "
                     "successfully"};
    }
    // Close the point group BEFORE the file, as the serial writer does.
    // With the MPI-IO driver the default close degree is semi: H5Fclose
    // fails while any object in the file is still open, Hid::reset()
    // discards that failure, and the file stays open with its metadata
    // unflushed when rank 0 renames it. Measured on the owner's machine
    // 2026-10-17: every collectively written file was refused on read
    // with E-RESTART-TRUNCATED ("does not open as a valid HDF5 file").
    // H5Fclose is collective; its result is checked so a failed close is
    // a refusal, never a renamed partial file.
    group_.reset();
    const hid_t file_id = file_.release();
    if (file_id >= 0 && H5Fclose(file_id) < 0) {
      return Failure{"E-RESTART-HDF5",
                     "finish(): collective H5Fclose of " + temp_path_ + " failed"};
    }
    int rank_in_comm = 0;
    MPI_Comm_rank(comm_, &rank_in_comm);
    int rename_rc = 0;
    if (rank_in_comm == 0) {
      rename_rc = std::rename(temp_path_.c_str(), final_path_.c_str());
    }
    MPI_Bcast(&rename_rc, 1, MPI_INT, 0, comm_);
    MPI_Barrier(comm_);
    if (rename_rc != 0) {
      return Failure{"E-RESTART-HDF5",
                     "finish(): rename " + temp_path_ + " -> " + final_path_ +
                         " failed on rank 0"};
    }
    return Result<void>::success();
  }

 private:
  HFile file_;
  HGroup group_;
  MPI_Comm comm_;
  std::string temp_path_;
  std::string final_path_;
  std::set<std::string> seen_;
  std::vector<std::pair<std::string, std::int64_t>> local_ids_;
  std::unordered_map<std::string, std::int64_t> local_declared_sizes_;
  std::unordered_map<std::string, std::vector<std::int64_t>> ids_by_name_;
};

}  // namespace

Result<std::unique_ptr<PointWriter>> create_restart_point_collective(
    FileLayout layout, const std::string& path, const Provenance& provenance,
    int comm_f, const std::string& point_label) {
  (void)point_label;
  if (layout != FileLayout::SeparateMeshAndPoints) {
    return Failure{"E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED",
                   "create_restart_point_collective: only "
                   "FileLayout::SeparateMeshAndPoints is supported this "
                   "round (docs/dev/restart.md Section 4.3's own declared "
                   "scope)"};
  }
  MPI_Comm comm = MPI_Comm_f2c(comm_f);
  int size = 0;
  MPI_Comm_size(comm, &size);

  const std::string temp_path = path + ".tmp";

  HProp fapl(H5Pcreate(H5P_FILE_ACCESS));
  if (!fapl.valid()) {
    return Failure{"E-RESTART-HDF5",
                   "create_restart_point_collective: H5Pcreate(fapl)"};
  }
  if (H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL) < 0) {
    return Failure{"E-RESTART-HDF5",
                   "create_restart_point_collective: H5Pset_fapl_mpio"};
  }
  HFile file(H5Fcreate(temp_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl));
  if (!file.valid()) {
    return Failure{"E-RESTART-HDF5",
                   "create_restart_point_collective: cannot create " + temp_path};
  }
  // The SAME five file attributes, under the SAME names, as the serial
  // writer (hdf5_restart_io.cpp, create_restart_point): the reader asks
  // for "format_version" and the rest by these exact names. The collective
  // writer first wrote "@"-prefixed names and a different version key, and
  // every file it produced was refused with E-RESTART-TRUNCATED "missing
  // attribute format_version" (owner's machine, 2026-10-17).
  Result<void> r = write_scalar_attr_u32_collective(file, "format_version",
                                                    provenance.format_version);
  if (!r.ok()) return r.error();
  r = write_scalar_attr_str_collective(file, "fuse_version", provenance.fuse_version);
  if (!r.ok()) return r.error();
  r = write_scalar_attr_u64_collective(file, "descriptor_hash", provenance.descriptor_hash);
  if (!r.ok()) return r.error();
  r = write_scalar_attr_u32_collective(file, "written_by_rank_count",
                                       static_cast<std::uint32_t>(size));
  if (!r.ok()) return r.error();
  r = write_scalar_attr_u64_collective(file, "mesh_checksum", provenance.mesh_checksum);
  if (!r.ok()) return r.error();

  HGroup point_group(H5Gopen2(file, "/", H5P_DEFAULT));
  if (!point_group.valid()) {
    return Failure{"E-RESTART-HDF5",
                   "create_restart_point_collective: cannot open the root "
                   "group of " + temp_path};
  }
  return std::unique_ptr<PointWriter>(std::make_unique<CollectivePointWriterImpl>(
      std::move(file), std::move(point_group), comm, temp_path, path));
}

}  // namespace fuse::restart_io
