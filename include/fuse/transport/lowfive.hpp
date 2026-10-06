#pragma once

// Task 8.4 continuation (S6 phase 3, "LowFive" track): the dataset-name <->
// field mapping Demo B (examples/demo_b) needs, declared independently of
// `fuse::TransportSpec` and `fuse::FieldSpec`.
//
// WHY A SEPARATE HEADER, AND WHY NOT ON TransportSpec OR FieldSpec.
// `TransportSpec` (fuse/transport/transport.hpp) is shared by every
// transport backend (`CopyTransport`, `MpiTransport`, `LowFiveTransport`);
// adding a LowFive-only field there would be a field every OTHER backend
// carries and ignores, and it is also mid-edit by this phase's "MPI
// connect" track (see src/CMakeLists.txt's own track markers). `FieldSpec`
// (include/fuse/model/field_spec.hpp) is owned by the model-layer track, not
// this one. This header is therefore new and self-contained: it is read by
// examples/demo_b's own drivers only, and by test/transport/lowfive_test.cpp.
// Folding it into the coupling graph properly (so a YAML `channels:` entry
// can name an HDF5 dataset the way it names a field) is exactly the open
// question this track's report raises for the lead -- see KNOWN_ISSUES.md
// section D and docs/user/lowfive.md.
//
// The string format mirrors Wilkins's own data-centric port model (named in
// docs/plans/soma-phase-8.md's Task 8.4 interfaces): a "port" is an HDF5
// path pattern, `"<file>.h5/<dataset path>"`. A leading `*` in the file
// component is a wildcard (matches LowFive's own `set_memory("*", "*")`
// pattern, already used unconditionally in lowfive_transport.cpp); nothing
// else in the pattern is currently special.

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/component.hpp"

namespace fuse {

/// One field's HDF5 binding: which dataset, in which file, an unmodified
/// HDF5-writing/reading component's `field_name` corresponds to.
struct LowFiveFieldBinding {
  /// Matches a `FieldSpec::name` in the coupling graph (include/fuse/model/
  /// field_spec.hpp) -- not validated against one here, since this header
  /// does not depend on the model layer; `LowFiveFieldBindingTable::validate`
  /// checks only what is expressible from these strings alone.
  std::string field_name;
  /// `"<file>.h5/<dataset path>"`, e.g. `"producer_out.h5/grid/temperature"`.
  std::string hdf5_path;
};

/// Splits `hdf5_path` into the file component and the dataset-path
/// component, at the first `.h5/` (case-sensitive). Returns a Failure
/// (`E-LOWFIVE-PORT`) if that separator does not appear exactly once in a
/// position that leaves both sides non-empty.
///
/// Header-only (like fuse/base/result.hpp's own Diagnostics): this is pure
/// string parsing with no HDF5/LowFive dependency, deliberately kept out of
/// src/transport/lowfive_transport.cpp's FUSE_HAVE_LOWFIVE-guarded body so
/// it is testable in a build with LowFive disabled (test/transport/
/// lowfive_test.cpp does exactly that).
[[nodiscard]] inline Result<std::pair<std::string, std::string>>
split_hdf5_path(const std::string& hdf5_path) {
  const std::string sep = ".h5/";
  const auto        first = hdf5_path.find(sep);
  if (first == std::string::npos) {
    return Failure{"E-LOWFIVE-PORT",
                 "'" + hdf5_path +
                     "' has no '.h5/' separator; expected "
                     "\"<file>.h5/<dataset path>\""};
  }
  if (hdf5_path.find(sep, first + 1) != std::string::npos) {
    return Failure{"E-LOWFIVE-PORT",
                 "'" + hdf5_path + "' contains '.h5/' more than once"};
  }
  std::string file = hdf5_path.substr(0, first + 3);  // includes ".h5"
  std::string dataset = hdf5_path.substr(first + sep.size());
  if (file.size() <= 3) {  // nothing before ".h5"
    return Failure{"E-LOWFIVE-PORT",
                 "'" + hdf5_path + "' has an empty file name before '.h5/'"};
  }
  if (dataset.empty()) {
    return Failure{"E-LOWFIVE-PORT",
                 "'" + hdf5_path + "' has an empty dataset path after '.h5/'"};
  }
  return std::make_pair(std::move(file), std::move(dataset));
}

/// A table of bindings for one coupling, with the uniqueness FUSE can check
/// without the full coupling graph: no two bindings may name the same
/// field, and no two may name the same exact (non-wildcard) dataset path --
/// either would mean two different buffers silently landing on the same
/// HDF5 object, which upstream's `MetadataVOL`/`DistMetadataVOL` would not
/// detect either (docs/user/lowfive.md).
class LowFiveFieldBindingTable {
 public:
  /// Errors: `E-LOWFIVE-PORT` (malformed `hdf5_path`, from `split_hdf5_path`),
  /// `E-LOWFIVE-DUPFIELD` (field_name already bound),
  /// `E-LOWFIVE-DUPPATH` (hdf5_path already bound to a different field).
  Result<void> add(LowFiveFieldBinding binding) {
    if (binding.field_name.empty()) {
      return Failure{"E-LOWFIVE-PORT", "a binding's field_name is empty"};
    }
    auto split = split_hdf5_path(binding.hdf5_path);
    if (!split.ok()) {
      return split.error();
    }
    if (find_by_field(binding.field_name) != nullptr) {
      return Failure{"E-LOWFIVE-DUPFIELD",
                   "field '" + binding.field_name +
                       "' is already bound to an HDF5 path; a field has at "
                       "most one"};
    }
    // Wildcard file components ("*.h5/...", matching LowFive's own
    // set_memory("*", "*") pattern) are deliberately NOT checked for path
    // collisions against each other here -- two wildcard bindings on
    // different dataset paths are the ordinary multi-field case, and
    // detecting a genuine collision needs the actual file each resolves to
    // at runtime, which this header has no way to know.
    const bool is_wildcard = !binding.hdf5_path.empty() &&
                              binding.hdf5_path.front() == '*';
    if (!is_wildcard && find_by_path(binding.hdf5_path) != nullptr) {
      return Failure{"E-LOWFIVE-DUPPATH",
                   "HDF5 path '" + binding.hdf5_path +
                       "' is already bound to a different field; two fields "
                       "would silently overwrite each other's dataset"};
    }
    bindings_.push_back(std::move(binding));
    return Result<void>::success();
  }

  [[nodiscard]] bool empty() const noexcept { return bindings_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return bindings_.size(); }

  /// nullptr if `field_name` has no binding.
  [[nodiscard]] const LowFiveFieldBinding* find_by_field(
      const std::string& field_name) const noexcept {
    auto it = std::find_if(bindings_.begin(), bindings_.end(),
                           [&](const LowFiveFieldBinding& b) {
                             return b.field_name == field_name;
                           });
    return it == bindings_.end() ? nullptr : &*it;
  }
  /// nullptr if `hdf5_path` has no binding. Exact match only -- this does
  /// NOT interpret the `*` wildcard a hdf5_path's file component may carry;
  /// a wildcard binding only ever matches by field_name lookup.
  [[nodiscard]] const LowFiveFieldBinding* find_by_path(
      const std::string& hdf5_path) const noexcept {
    auto it = std::find_if(bindings_.begin(), bindings_.end(),
                           [&](const LowFiveFieldBinding& b) {
                             return b.hdf5_path == hdf5_path;
                           });
    return it == bindings_.end() ? nullptr : &*it;
  }

  [[nodiscard]] const std::vector<LowFiveFieldBinding>& bindings() const noexcept {
    return bindings_;
  }

 private:
  std::vector<LowFiveFieldBinding> bindings_;
};

/// Unit 3b (S6 phase 3, Conformance): the writer/reader intercommunicator
/// `DistMetadataVOL` needs for the distributed interception path
/// (docs/user/lowfive.md's own account of why the single-process
/// `MetadataVOL` cannot serve two separate OS processes). Derived from
/// `connect()`'s own base communicator (connect.hpp's own module comment:
/// "every rank of every component in the graph, numbered exactly as the
/// graph's RankSets say") -- NOT a second accept/connect handshake; U and
/// V are already merged into one communicator by the time `fuse_setup()`
/// returns, and this plan is exactly what `MPI_Comm_split()` +
/// `MPI_Intercomm_create()` need to carve a FRESH writer<->reader
/// intercommunicator back out of it, for LowFive's own exclusive use.
///
/// SPLIT FOR TESTABILITY, the same reasoning connect.hpp's own module
/// comment gives: this struct and `plan_lowfive_rendezvous()` below are
/// pure functions of two already-known `RankSet`s and a rank number --
/// no MPI call, so they run in the base build configuration this sandbox
/// can always start, instead of only inside a job MPI cannot launch here
/// (KNOWN_ISSUES.md C1). The actual `MPI_Comm_split`/`MPI_Intercomm_
/// create` calls this plan feeds are `#if FUSE_HAVE_MPI`-guarded,
/// alongside `connect()`'s own MPI-calling functions.
struct LowFiveRendezvousPlan {
  /// True if this rank is on the writer's side (the field whose
  /// `hdf5_path`-bound export this run's own H5Dwrite calls belong to);
  /// false on the reader's side. Which side's `RankSet` `this_rank`
  /// belongs to, computed once so a caller never has to ask twice.
  bool is_writer = false;
  /// `MPI_Comm_split()`'s own `color` argument for this rank: 0 for the
  /// writer's side, 1 for the reader's -- the two groups
  /// `MPI_Intercomm_create()` then joins.
  int split_color = 0;
  /// `MPI_Comm_split()`'s own `key` argument for this rank: its position
  /// within its OWN side's `RankSet` (`RankSet::index_of`), preserving
  /// the graph's own declared rank order inside the split communicator
  /// rather than leaving it to whatever order `MPI_Comm_split` happens
  /// to assign when ranks tie (every key here is already distinct, so
  /// there is no tie to break, but the order itself is the graph's own,
  /// not MPI's).
  int split_key = 0;
  /// `MPI_Intercomm_create()`'s own `local_leader` argument: the chosen
  /// leader's rank WITHIN the split-off local communicator. Always 0 --
  /// the leader is defined as the side's own lowest base-communicator
  /// rank (`RankSet::min_rank()`), which lands at index 0 after
  /// `MPI_Comm_split` (ranks are split preserving relative order via
  /// `split_key` above) -- written out explicitly rather than left for
  /// a caller to assume, since `MPI_Intercomm_create` takes it as a
  /// REQUIRED argument, not a default.
  int local_leader = 0;
  /// `MPI_Intercomm_create()`'s own `remote_leader` argument: the OTHER
  /// side's leader rank, but in `peer_comm`'s own numbering -- here, the
  /// base communicator's, since `peer_comm` is the base communicator
  /// itself (both sides already share it; no separate port/accept
  /// handshake is needed to find each other, unlike the ORIGINAL
  /// U<->V connect() this intercommunicator is carved back out of).
  /// Equals the other side's own `RankSet::min_rank()`.
  int remote_leader = 0;
  /// `MPI_Intercomm_create()`'s own `tag` argument: fixed, not derived
  /// from either RankSet, because both sides must agree on it without
  /// communicating first and the two sides' own rank-set arithmetic
  /// gives no reason to vary it run to run. Chosen away from 0 (easy to
  /// collide with a caller's own untagged traffic) and documented here
  /// as the one place that matters, since nothing else in this struct
  /// depends on it.
  int tag = 0x1073;
};

/// Computes `LowFiveRendezvousPlan` for `this_rank` from the writer's and
/// reader's own `RankSet`s (both already in the base communicator's
/// numbering, per `connect()`'s own contract). Errors (`Failure`, never a
/// throw):
///   `E-LOWFIVE-RENDEZVOUS-OVERLAP`  the two RankSets share a rank -- a
///     single process cannot be split into two different MPI_Comm_split
///     colors at once, and a channel's own writer and reader being the
///     SAME rank is the single-process case this function is not for
///     (MetadataVOL, not DistMetadataVOL -- docs/user/lowfive.md).
///   `E-LOWFIVE-RENDEZVOUS-UNPLACED`  `this_rank` is in neither RankSet --
///     a rank with no part in this channel has no intercommunicator to
///     join at all, and asking for one is a caller error, not a
///     legitimate "neither side" configuration (unlike an ordinary
///     coupler rank, which `fuse_run()` still drives through the
///     Runner's own collectives; this plan is LowFive-specific and a
///     bystander rank simply never calls into it).
[[nodiscard]] inline Result<LowFiveRendezvousPlan> plan_lowfive_rendezvous(
    const RankSet& writer_ranks, const RankSet& reader_ranks, int this_rank) {
  if (writer_ranks.overlaps(reader_ranks)) {
    return Failure{"E-LOWFIVE-RENDEZVOUS-OVERLAP",
                 "the writer's and reader's own RankSets share at least "
                 "one rank; a single process cannot join both sides of "
                 "this intercommunicator"};
  }
  const bool is_writer = writer_ranks.contains(this_rank);
  const bool is_reader = reader_ranks.contains(this_rank);
  if (!is_writer && !is_reader) {
    return Failure{"E-LOWFIVE-RENDEZVOUS-UNPLACED",
                 "rank " + std::to_string(this_rank) +
                     " is in neither the writer's nor the reader's own "
                     "RankSet; it has no part in this channel's own "
                     "rendezvous"};
  }
  const RankSet& own = is_writer ? writer_ranks : reader_ranks;
  const RankSet& other = is_writer ? reader_ranks : writer_ranks;
  LowFiveRendezvousPlan plan;
  plan.is_writer = is_writer;
  plan.split_color = is_writer ? 0 : 1;
  plan.split_key = own.index_of(this_rank);
  plan.local_leader = 0;
  plan.remote_leader = other.min_rank();
  return plan;
}

}  // namespace fuse
