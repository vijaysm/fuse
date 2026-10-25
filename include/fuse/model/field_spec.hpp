#pragma once

#include <optional>
#include <string>

#include "fuse/base/ids.hpp"
#include "fuse/base/units.hpp"

namespace fuse {

/// Spec Section 4.2. disc+order select the remap operator; units are checked at
/// lowering; conservation drives which verification runs.
struct FieldSpec {
  std::string    name;
  RegionId       region;
  Discretization disc = Discretization::FV;
  int            order = 0;       // polynomial order for GLL; ignored for FV
  int            components = 1;  // 1 scalar, 3 vector, 9 tensor
  Units          units{};
  Conservation   conservation = Conservation::None;
  Intent         intent = Intent::Provides;

  /// `"<file>.h5/<dataset path>"`
  /// (fuse::split_hdf5_path, fuse/transport/lowfive.hpp) -- this field
  /// moves between components not through FUSE's own Transport but by one
  /// side calling plain H5Dwrite and the other plain H5Dread at this path,
  /// with LowFive intercepting both calls in memory (docs/user/lowfive.md).
  /// unset (the default) for every field that moves through an ordinary
  /// Transport.
  /// `validate()` (fuse/model/validate.hpp) refuses a graph that sets this with no
  /// LowFive-launched component (`LaunchMode::LowFive`, fuse/model/
  /// launch.hpp) anywhere in it, and refuses two fields sharing the exact
  /// same (non-wildcard) path -- the same two rules
  /// `LowFiveFieldBindingTable::add` already enforces for a hand-built
  /// table; `fuse::build_lowfive_bindings()` (fuse/transport/
  /// lowfive_graph.hpp) builds one FROM a graph's own fields, so the two
  /// are never two sources of truth for the same rule.
  std::optional<std::string> hdf5_path;
};

[[nodiscard]] inline bool same_shape(const FieldSpec& a,
                                     const FieldSpec& b) noexcept {
  return a.components == b.components;
}

}  // namespace fuse
