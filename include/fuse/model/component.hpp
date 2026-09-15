#pragma once

#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"

namespace fuse {

/// Contiguous range of world-communicator ranks owned by a component.
/// Disjoint rank sets are the normal case (spec §4.1); overlap is legal.
struct RankSet {
  int first = 0;
  int count = 1;

  [[nodiscard]] bool contains(int rank) const noexcept {
    return rank >= first && rank < first + count;
  }
  [[nodiscard]] bool overlaps(const RankSet& o) const noexcept {
    return first < o.first + o.count && o.first < first + count;
  }
  [[nodiscard]] int last() const noexcept { return first + count - 1; }
};

/// Spec §8.1: `interval` is the component's own coupling interval.
/// Independent intervals are the default; lockstep is all intervals equal.
struct ComponentDecl {
  std::string           name;
  RankSet               ranks;
  Rational              interval{1};
  std::vector<FieldId>  fields;
  // Deliberately NOT serialized by either the flat descriptor
  // (include/fuse/model/descriptor.hpp) or the YAML front-end
  // (src/graph/yaml_io.cpp). Nothing in the current graph API populates
  // this field -- there is no add_region_to_component()-style call -- so
  // Task 1.5-1.7 leaves it as a placeholder for whatever Phase 2 introduces
  // to populate it, at which point serialization support should be added
  // alongside the writer. Pinned by
  // Descriptor.RegionsFieldIsIntentionallyNotSerialized so a later
  // populate-without-serialize regression is caught rather than silently
  // passing through structurally_equal() (which is defined in terms of
  // serialize(), so it cannot see a field neither side writes).
  std::vector<RegionId> regions;
};

/// The time span an accumulation or space-time projection covers.
struct Window {
  enum class Kind { CouplingPeriod, FromComponent, Explicit };

  Kind        kind = Kind::CouplingPeriod;
  ComponentId component;   // valid only when kind == FromComponent
  Rational    length{0};   // valid only when kind == Explicit

  [[nodiscard]] static Window coupling_period() { return Window{}; }
  [[nodiscard]] static Window from(ComponentId c) {
    return Window{Kind::FromComponent, c, Rational{0}};
  }
  [[nodiscard]] static Window explicit_length(Rational len) {
    return Window{Kind::Explicit, ComponentId{}, len};
  }
};

}  // namespace fuse
