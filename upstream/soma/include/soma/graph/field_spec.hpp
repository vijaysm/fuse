#pragma once

#include <string>

#include "soma/base/ids.hpp"
#include "soma/base/units.hpp"

namespace soma {

/// Spec §4.2. disc+order select the remap operator; units are checked at
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
};

[[nodiscard]] inline bool same_shape(const FieldSpec& a,
                                     const FieldSpec& b) noexcept {
  return a.components == b.components;
}

}  // namespace soma
