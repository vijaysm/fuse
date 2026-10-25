#pragma once

/// The ONLY header
/// in this repository that may name an HDF5 type (`tools/check-layering.py`'s
/// FORBIDDEN_PUBLIC convention, the same one PETSc's own public headers
/// already follow -- `hid_t` never escapes past this file's own
/// `.cpp`: every type below is backend-agnostic, pimpl'd). Built ONLY
/// under FUSE_ENABLE_HDF5, against the SAME HDF5 target FUSE_ENABLE_
/// LOWFIVE resolves (cmake/FuseDependencies.cmake), using the DEFAULT
/// (non-VOL) property list throughout -- this writer never touches
/// LowFive's `H5VOLProperty`, so a restart file genuinely reaches disk
/// (docs/user/lowfive.md: LowFive's own interception is in-memory only).

#include <cstdint>
#include <memory>
#include <string>

#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/restart.hpp"

namespace fuse::restart_io {

/// Both file layouts (docs/dev/restart.md Section 4.4):
///   SeparateMeshAndPoints  `<run>.mesh.h5` once, `<run>.<period>.h5` per
///                          point, each carrying the mesh file's own
///                          checksum.
///   GrowingFile            one file, copy-append-renamed per point (chosen
///                          for robustness over in-place
///                          append: a crash mid-write can never damage an
///                          earlier point, at the cost of one file copy
///                          per point).
enum class FileLayout { SeparateMeshAndPoints, GrowingFile };

/// Bumped 1 -> 2 when `Provenance::
/// descriptor_version` was added below; is a restart-file schema change, by this
/// project's own append-only-with-a-version-bump convention (the same one
/// `fuse::kGraphVersion` itself follows) -- a point written under format
/// 1 has no `descriptor_version` attribute to read at all, so it is
/// refused by THIS check (unconditional, format_version != kFormatVersion)
/// rather than failing later with a generic HDF5 attribute-open
/// error that does not name what is actually missing.
inline constexpr std::uint32_t kFormatVersion = 2;

/// File-level metadata, written as real fixed-width HDF5 attributes
/// (never through `fuse::RestartWriter::put_scalar()`'s lossy `double`,
/// which cannot carry `descriptor_hash`'s full 64-bit range -- see
/// fuse/runtime/restart.hpp's own header comment on this exact point).
struct Provenance {
  std::uint32_t format_version = kFormatVersion;
  std::string   fuse_version;
  /// The coupling-graph descriptor's own
  /// `fuse::kGraphVersion` (fuse/model/descriptor.hpp) THIS POINT WAS
  /// WRITTEN UNDER, recorded independently of `descriptor_hash` below so
  /// `open_restart_point()` can distinguish "this point was written by a
  /// FUSE build with a DIFFERENT descriptor version" (refused here, by
  /// name, unconditionally -- a descriptor-version mismatch means the
  /// wire layout itself may disagree, the same reason `fuse::deserialize()`
  /// refuses a version mismatch before ever comparing content) from "this
  /// point was written by the SAME descriptor version but a DIFFERENT
  /// graph" (the caller's own job to catch by comparing `descriptor_hash`
  /// against `fuse::restart_descriptor_hash(current_graph)` --
  /// `fuse::check_restart_descriptor_hash()`, fuse/runtime/restart.hpp --
  /// since only the caller holds the current graph to hash). Defaults to
  /// THIS BUILD's own `fuse::kGraphVersion`, the only value a normal
  /// write path would ever want: a restart point does not choose a
  /// descriptor version, it inherits whatever the writing library build
  /// compiled in, exactly like `format_version` above inherits
  /// `kFormatVersion`.
  std::uint32_t descriptor_version = fuse::kGraphVersion;
  std::uint64_t descriptor_hash = 0;
  std::uint32_t written_by_rank_count = 1;
  /// 0 is the sentinel "no mesh was written for this point" -- Slice 1's
  /// own scope note (docs/dev/restart.md): none of this slice's test
  /// coverage (dn_heat) is mesh-backed, so the mesh-once/checksum
  /// machinery is EXERCISED structurally (a test fixture sets this to a
  /// synthetic nonzero value and corrupts it to make the mismatch
  /// refusal fire) without a real `fuse::MeshView` behind it; a real
  /// mesh writer is Slice 2's own deliverable, alongside the Brusselator
  /// example this slice's own design doc names as that slice's pilot.
  std::uint64_t mesh_checksum = 0;
};

/// The write side of one restart point. `finish()` performs the atomic
/// rename (`SeparateMeshAndPoints`) or copy-append-rename (`GrowingFile`,
/// docs/dev/restart.md Section 4.4) -- NOT `fuse::RestartWriter::
/// finish_point()`, which this object's own implementation already
/// calls internally as part of `finish()`. Calling `finish()` is what
/// makes the point durable; destroying a `PointWriter` without calling
/// it leaves only the temporary file (or nothing at all), by design --
/// the caller must not assume a point exists until `finish()` has
/// returned success.
class PointWriter {
 public:
  virtual ~PointWriter() = default;
  [[nodiscard]] virtual RestartWriter& writer() = 0;
  [[nodiscard]] virtual Result<void> finish() = 0;
};

/// Creates a new restart point for writing at `path`
/// (`SeparateMeshAndPoints`: `path` is the point file itself;
/// `GrowingFile`: `path` is the one growing file, and `point_label`
/// names the new `/restart_points/<point_label>/` group -- required,
/// never empty, for that layout only).
[[nodiscard]] Result<std::unique_ptr<PointWriter>> create_restart_point(
    FileLayout layout, const std::string& path, const Provenance& provenance,
    const std::string& point_label = {});

/// The read side. `provenance()` is valid immediately after a successful
/// `open_restart_point()` call -- the caller checks `format_version`/
/// `descriptor_hash`/`written_by_rank_count`/`mesh_checksum` against its
/// own expectations BEFORE calling `reader()` for anything
/// (docs/dev/restart.md Section 4.4's version-gate-first rule); this
/// class itself refuses nothing on their behalf, because what counts as
/// a mismatch for `written_by_rank_count` depends on which components
/// are Form-B-only (`fuse::check_restart_rank_count()`, fuse/runtime/
/// restart.hpp), a runtime-layer fact this file-level class cannot see.
class PointReader {
 public:
  virtual ~PointReader() = default;
  [[nodiscard]] virtual RestartReader& reader() = 0;
  [[nodiscard]] virtual const Provenance& provenance() const = 0;
};

/// Opens `path` for reading (`GrowingFile`: `point_label` selects which
/// `/restart_points/<point_label>/` group; empty reads the file's own
/// `/mesh`-adjacent latest-point marker -- Slice 1 writes exactly one
/// point per test, so `point_label` is required whenever more than one
/// exists). `Failure` (E-RESTART-TRUNCATED) if `path` does not open as a
/// valid HDF5 file at all, or is missing `@format_version`.
[[nodiscard]] Result<std::unique_ptr<PointReader>> open_restart_point(
    FileLayout layout, const std::string& path,
    const std::string& point_label = {});

/// The CONTENT half of the descriptor provenance check, with its own refusal code (E-RESTART-DESCRIPTOR-HASH),
/// distinguished from `open_restart_point()`'s own E-RESTART-DESCRIPTOR-
/// VERSION refusal above: that one fires on a DIFFERENT descriptor
/// VERSION and needs no graph (`fuse::kGraphVersion` is a compile-time
/// constant of this build, checked unconditionally by `open_restart_point()`
/// itself); this one fires on the SAME descriptor version but a DIFFERENT
/// graph, and needs `current_graph` to compute `fuse::restart_descriptor_
/// hash()` (fuse/model/restart.hpp) against. Free function, not a
/// `PointReader` method, for the SAME reason `PointReader`'s own doc
/// comment above gives for leaving every other provenance check to the
/// caller: this file-level reader cannot see the current graph. A
/// `provenance` that already came back from a successful
/// `open_restart_point()` is guaranteed to carry a `descriptor_version`
/// equal to `fuse::kGraphVersion` (that refusal fires first, inside
/// `open_restart_point()`, before a caller ever gets a `Provenance` to
/// pass here) -- so a mismatch this function reports is a genuine
/// content difference, a different graph under the SAME wire format,
/// never a version disagreement arriving mislabelled as a hash one.
[[nodiscard]] Result<void> check_restart_descriptor_hash(
    const CouplingGraph& current_graph, const Provenance& provenance);

}  // namespace fuse::restart_io
