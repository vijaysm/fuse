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

}  // namespace fuse
