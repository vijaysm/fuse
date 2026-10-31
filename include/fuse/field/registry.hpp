#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/result.hpp"
#include "fuse/field/field_ref.hpp"

namespace fuse {

/// Maps FieldId to the component-owned storage bound at registration.
/// Resolution happens once at setup; the hot loop dereferences pointers.
///
/// The registry stores `FieldRef` *by value* and `get()` hands back a copy,
/// so a handle obtained from `get()` is never a reference into the registry's
/// own container and cannot be invalidated by a later `bind()`. The backing
/// container is node-based for the same reason.
///
/// Lifetime: the registry copies pointers, not data. See the FieldRef
/// lifetime contract -- the component that registered the storage must keep it
/// alive and at a fixed address for as long as the registry is in use.
class FieldRegistry {
 public:
  /// Binds `id` to component-owned storage. Rebinding an already-bound field
  /// is an error rather than a silent replacement: swapping a component's
  /// pointer mid-run is a class of bug worth rejecting.
  ///
  /// On any error the registry is left exactly as it was.
  Result<void> bind(FieldId id, FieldRef ref) {
    if (!id.valid()) {
      return Failure{"E-FIELD-ID",
                   "cannot bind storage to an invalid FieldId (the "
                   "default-constructed sentinel)"};
    }
    // A null pointer is rejected only when there is something to point AT.
    //
    // An unconditional rejection would make a component that legitimately
    // owns zero elements of a mesh entity set impossible to bind:
    // `std::vector<double>{}.data()` is null, and on a partitioned mesh a rank
    // owning none of a mesh entity set is ordinary, not an error (spec Section
    // 4.1, and the n->m redistribution is built around it). An empty view
    // addresses nothing -- `Layout::well_formed()` already says so explicitly
    // -- so there is no pointer for it to get wrong. Null with count = 4
    // (FieldRegistry.RejectsNullData) is still rejected.
    if (!ref.valid() && ref.layout().count != 0) {
      return Failure{"E-FIELD-NULL", "field " + std::to_string(id.value()) +
                                       " bound to a null pointer for " +
                                       std::to_string(ref.layout().count) +
                                       " elements"};
    }
    if (!ref.layout().well_formed()) {
      return Failure{"E-FIELD-LAYOUT",
                   "field " + std::to_string(id.value()) +
                       " has a layout whose element and component strides "
                       "alias or overflow: stride=" +
                       std::to_string(ref.layout().stride) +
                       ", component_stride=" +
                       std::to_string(ref.layout().component_stride) +
                       ", count=" + std::to_string(ref.layout().count) +
                       ", components=" +
                       std::to_string(ref.layout().components)};
    }
    if (refs_.contains(id)) {
      return Failure{"E-FIELD-REBIND", "field " + std::to_string(id.value()) +
                                         " is already bound"};
    }
    refs_.emplace(id, ref);
    return Result<void>::success();
  }

  [[nodiscard]] bool is_bound(FieldId id) const { return refs_.contains(id); }

  /// Returns a copy of the view. Deliberately by value: a reference into
  /// `refs_` would be a lifetime hazard for callers that hold it across a
  /// later bind().
  [[nodiscard]] Result<FieldRef> get(FieldId id) const {
    const auto it = refs_.find(id);
    if (it == refs_.end()) {
      return Failure{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    return it->second;
  }

  [[nodiscard]] std::size_t bound_count() const noexcept { return refs_.size(); }

  /// The subset of `required` that is not bound, in request order (duplicates
  /// preserved) so the caller's diagnostic reads in the order it asked.
  [[nodiscard]] std::vector<FieldId> unbound(
      const std::vector<FieldId>& required) const {
    std::vector<FieldId> missing;
    for (const FieldId f : required) {
      if (!is_bound(f)) missing.push_back(f);
    }
    return missing;
  }

 private:
  std::unordered_map<FieldId, FieldRef> refs_;
};

}  // namespace fuse
