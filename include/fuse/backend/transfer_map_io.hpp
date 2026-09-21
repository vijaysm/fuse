#pragma once

/// Persisted transfer maps — S5.4.
///
/// A remap matrix costs far more to build than to apply: a supermesh
/// intersection over two unstructured meshes dominates setup, and the result
/// is a pure function of the two meshes. Persisting it turns a repeated run,
/// a restart, or a rank-count change into a read.
///
/// ===========================================================================
/// WHY THE FILE HOLDS GLOBAL IDS AND NOT THE MATRIX
/// ===========================================================================
///
/// `RemapWeights::row` and `::col` are LOCAL entity indices — `n_rows` and
/// `n_cols` are documented as local, and `MeshView` deliberately reports
/// nothing global (see its header: "A rank sees a partition, never the whole
/// mesh"). Writing those indices out would produce a file that is only
/// readable by a run with the identical partition, which is the one case that
/// needs no file: if the partition is identical, so is the mesh, and the map
/// could just be rebuilt.
///
/// So each entry is stored as `(target global id, source global id, weight)`
/// via `MeshView::global_id`. That makes a shard written by any rank at any
/// rank count readable by any rank at any other rank count: a reader keeps
/// the rows whose target entity it owns and ignores the rest. Persisting at
/// two ranks and reloading at four is then not a special case — it is the
/// only thing the format can do.
///
/// Entries are sorted by (target gid, source gid). Sorted order is part of
/// the contract, not a convenience: floating-point summation is not
/// associative, so `apply()` over a permuted matrix returns different
/// rounding, and a map that reloads to a different answer than it was written
/// with would defeat the purpose. It is also what makes two shards mergeable
/// by a linear pass.
///
/// ===========================================================================
/// WHAT IS AND IS NOT VALIDATED
/// ===========================================================================
///
/// Validated, each with its own refusal:
///   - `E-MAP-MAGIC` / `E-MAP-VERSION` — not one of our files, or a format
///     this build does not read. The version is bumped whenever a field's
///     meaning changes, never reused.
///   - `E-MAP-TRUNCATED` — the entry count in the header disagrees with the
///     bytes present. A partial write from a killed job reads as truncated
///     rather than as a short map.
///   - `E-MAP-CHECKSUM` — the payload hash disagrees. This catches the
///     silent-corruption case that a length check cannot.
///   - `E-MAP-IDENTITY` — the map's own description (order, monotonicity,
///     the two topological dimensions) disagrees with what the caller asked
///     for. Two builders return the same triplet structure for different
///     kinds of object (`RemapWeights`'s own note), so the file says which.
///   - `E-MAP-COVERAGE` — **the distribution check.** A target entity this
///     rank owns has no row in what was read. That is what a mismatched
///     partition looks like from the reader's side: either the shards do not
///     cover this rank's share, or they were written for a different mesh.
///     Applying a matrix with a missing row would silently write zero into a
///     field the map was supposed to fill.
///
/// NOT validated, deliberately: whether the weights are still *numerically*
/// right for the meshes in hand. Nothing in a file can establish that — the
/// meshes could have moved under the same global ids. `RemapWeights` carries
/// its own conservation audit (`max_row_defect`, `max_col_defect`), those
/// fields are persisted, and a caller that cares re-checks them against the
/// meshes it actually loaded. The file reports what it was built from; it
/// does not vouch for it.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

/// What a map says about itself, and what a reader may require of it.
struct TransferMapIdentity {
  int  order = 1;         ///< `RemapWeights::order`
  bool monotone = false;  ///< `RemapWeights::monotone`
  int  src_dim = 0;       ///< topological dimension the source ids live on
  int  tgt_dim = 0;       ///< and the target's

  [[nodiscard]] bool operator==(const TransferMapIdentity&) const = default;
};

/// A map as it exists on disk: global ids, so rank-count independent.
struct PersistedTransferMap {
  TransferMapIdentity        identity;
  std::vector<std::uint64_t> tgt_gid;  ///< ascending, with `src_gid` within it
  std::vector<std::uint64_t> src_gid;
  std::vector<double>        val;

  /// Carried through from the builder's audit. Reported, never trusted --
  /// see the header note on what is not validated.
  double src_measure_total = 0.0;
  double tgt_measure_total = 0.0;
  double max_row_defect = 0.0;
  double max_col_defect = 0.0;

  [[nodiscard]] std::size_t size() const noexcept { return val.size(); }
};

/// Translate this rank's matrix into global ids, ready to write.
///
/// `src` and `tgt` must be the views the matrix was built from: the local
/// indices in `w` are indices into them. Errors: `E-MAP-SHAPE` if an index in
/// `w` is outside the view it should index.
[[nodiscard]] Result<PersistedTransferMap> to_persisted(
    const RemapWeights& w, const MeshView& src, const MeshView& tgt,
    int src_dim, int tgt_dim);

/// Serialise to bytes. Separate from the file write so the format can be
/// tested without a filesystem, and so a caller may put a map somewhere other
/// than a file.
[[nodiscard]] std::vector<std::byte> encode_transfer_map(
    const PersistedTransferMap& map);

/// Parse bytes written by `encode_transfer_map`. Errors: `E-MAP-MAGIC`,
/// `E-MAP-VERSION`, `E-MAP-TRUNCATED`, `E-MAP-CHECKSUM`.
[[nodiscard]] Result<PersistedTransferMap> decode_transfer_map(
    std::span<const std::byte> bytes);

/// Write / read one shard. `E-MAP-IO` covers every filesystem failure, with
/// the path and the reason in the message.
[[nodiscard]] Result<void> write_transfer_map(const std::string& path,
                                              const PersistedTransferMap& map);
[[nodiscard]] Result<PersistedTransferMap> read_transfer_map(
    const std::string& path);

/// Merge shards into one map, keeping the sorted order the format promises.
/// Duplicate (tgt, src) pairs are an error (`E-MAP-DUPLICATE`): two shards
/// claiming the same entry would double the weight, and silently summing them
/// is indistinguishable from a legitimately split entry.
[[nodiscard]] Result<PersistedTransferMap> merge_transfer_maps(
    std::span<const PersistedTransferMap> shards);

/// Turn a persisted map back into a matrix for the partition in hand.
///
/// Keeps exactly the rows whose target gid is owned by `tgt`, and maps both
/// ids back to local indices. Errors:
///   - `E-MAP-IDENTITY` if `required` is given and the map disagrees with it.
///   - `E-MAP-COVERAGE` if an owned target entity has no row, or a kept row
///     names a source gid this rank does not hold. The second is not a
///     partition the caller can apply: the source value is not here.
///
/// The returned matrix's `val` order follows the file, so `apply()` sums in
/// the order the map was written with and reproduces the writer's rounding.
[[nodiscard]] Result<RemapWeights> localise(
    const PersistedTransferMap& map, const MeshView& src, const MeshView& tgt,
    int src_dim, int tgt_dim,
    const TransferMapIdentity* required = nullptr);

}  // namespace fuse
