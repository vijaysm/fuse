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
///
/// Value types: each field keeps the value type it was bound with
/// (fuse/model/value_traits.hpp). `get<T>()` returns a typed view only for that
/// type and refuses any other (`E-FIELD-VALUE-TYPE`); `get()` with no template
/// argument is `get<double>()`, which is what every caller that predates value
/// types asks for.
class FieldRegistry {
 public:
  /// Binds `id` to component-owned storage. Rebinding an already-bound field
  /// is an error rather than a silent replacement: swapping a component's
  /// pointer mid-run is a class of bug worth rejecting.
  ///
  /// On any error the registry is left exactly as it was.
  ///
  /// The non-template overload keeps `bind(id, {data, layout, space})` and
  /// other binary64 call sites that rely on an implicit conversion working.
  Result<void> bind(FieldId id, FieldRef ref) { return bind_typed(id, ref); }

  template <CoupledValue T>
  Result<void> bind(FieldId id, BasicFieldRef<T> ref) {
    return bind_typed(id, ref);
  }

  [[nodiscard]] bool is_bound(FieldId id) const { return refs_.contains(id); }

  /// Returns a copy of the view. Deliberately by value: a reference into
  /// `refs_` would be a lifetime hazard for callers that hold it across a
  /// later bind().
  template <CoupledValue T = double>
  [[nodiscard]] Result<BasicFieldRef<T>> get(FieldId id) const {
    const auto it = refs_.find(id);
    if (it == refs_.end()) {
      return Failure{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    auto typed = it->second.template as<T>();
    if (!typed) {
      return Failure{typed.error().code,
                     "field " + std::to_string(id.value()) + ": " +
                         typed.error().message};
    }
    return typed.value();
  }

  /// The field as scalars of type `S` (default binary64), `components *
  /// scalar_count` per element. A field that holds `S` is returned as it is; a
  /// field of arrays of scalars (a dual number with a gradient, say) is returned
  /// as its scalars, the components of each value following each other
  /// (`scalar_layout`). This is what the runtime reads, because every
  /// operation it applies to field values acts on each scalar separately (see
  /// value_traits.hpp for the one that does not and is refused). A field of any
  /// other type is refused with `E-FIELD-VALUE-TYPE`.
  template <CoupledValue S = double>
  [[nodiscard]] Result<BasicFieldRef<S>> get_scalars(FieldId id) const {
    const auto it = refs_.find(id);
    if (it == refs_.end()) {
      return Failure{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    auto scalars = it->second.template as_scalars<S>();
    if (!scalars) {
      return Failure{scalars.error().code,
                     "field " + std::to_string(id.value()) + ": " +
                         scalars.error().message};
    }
    return scalars.value();
  }

  /// The limbs of an Arithmetic field (double-double, quad-double), as scalars of
  /// type `S`, `components * scalar_count` per element. For the operators that
  /// are instantiated for the Arithmetic type from a closed list (stage U4);
  /// `get_scalars` refuses such a field, because an operator that acts on each
  /// scalar separately must not see its limbs as independent numbers. A field
  /// that is not an Arithmetic type over `S` is refused with `E-FIELD-VALUE-TYPE`.
  template <CoupledValue S = double>
  [[nodiscard]] Result<BasicFieldRef<S>> get_limbs(FieldId id) const {
    const auto it = refs_.find(id);
    if (it == refs_.end()) {
      return Failure{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    auto limbs = it->second.template as_limbs<S>();
    if (!limbs) {
      return Failure{limbs.error().code,
                     "field " + std::to_string(id.value()) + ": " +
                         limbs.error().message};
    }
    return limbs.value();
  }

  /// True when `id` is bound to a field whose values carry a gradient.
  [[nodiscard]] bool has_gradient(FieldId id) const {
    const auto it = refs_.find(id);
    return it != refs_.end() && it->second.gradient_size() > 0;
  }

  /// Declares that the last `n` components of each entity of the bound field
  /// `id` hold the gradient of the leading ones: what a participant that sees only
  /// arrays of binary64 (the C and Fortran interface) says in place of a dual-number
  /// type. Refused (`E-FIELD-GRADIENT`) unless `1 <= n < components`, so that at
  /// least one component is the value itself, and (`E-FIELD-UNBOUND`) for a field
  /// that is not bound.
  [[nodiscard]] Result<void> declare_gradient(FieldId id, int n) {
    const auto it = refs_.find(id);
    if (it == refs_.end()) {
      return Failure{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    const int components = it->second.layout().components;
    if (n < 1 || n >= components) {
      return Failure{"E-FIELD-GRADIENT",
                   "field " + std::to_string(id.value()) + ": a gradient of " +
                       std::to_string(n) + " components needs 1 <= n < components (" +
                       std::to_string(components) + ")"};
    }
    it->second.declare_gradient(n);
    return Result<void>::success();
  }

  /// The view with its value type erased, for code that dispatches on the
  /// value type itself.
  [[nodiscard]] Result<AnyFieldRef> get_any(FieldId id) const {
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
  template <CoupledValue T>
  Result<void> bind_typed(FieldId id, BasicFieldRef<T> ref) {
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
    refs_.emplace(id, AnyFieldRef(ref));
    return Result<void>::success();
  }

  std::unordered_map<FieldId, AnyFieldRef> refs_;
};

}  // namespace fuse
