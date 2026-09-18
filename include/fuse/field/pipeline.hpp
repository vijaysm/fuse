#pragma once

/// The lazy field pipeline: `slice`, `component` and `owned` chains that do no
/// work and allocate nothing until a terminal asks for a number.
///
/// This is the second of the two kinds of chaining the charter keeps apart
/// (usability rule U1). The other is the graph BUILDER, where every call
/// returns the builder and nothing is validated until `validate()`. Here every
/// non-terminal call returns a VIEW -- a `Kokkos::subview` of what it was
/// given -- and nothing is read until a terminal runs. No type exposes both
/// kinds of method, which a static gate in the tests asserts.
///
/// WHY THE VIEWS ARE LAZY, AND WHY THAT IS TESTED. A chain like
///
///     f.owned().slice({2, fuse::end}, fuse::all).component(1)
///
/// on a field of ten million elements must not copy ten million doubles three
/// times. Every stage is a strided view over the SAME memory, so the chain
/// costs a few integers. The problem with that claim is that an eager
/// implementation passes every functional test the lazy one does -- it just
/// runs slower and allocates. So laziness is pinned by an allocation-count
/// fence (`test/field/pipeline_test.cpp`), not by inspection.
///
/// WHY NON-TERMINALS REFUSE RATHER THAN RETURN A RESULT. The builder's rule
/// (U2) is that authoring calls never return a `Result`, so that one terminal
/// `validate()` can report every defect at once. The pipeline is the opposite
/// case: there is nothing to accumulate, the caller's arithmetic is simply
/// wrong, and a `Result` at every stage would make the chain unwritable. A
/// reversed or out-of-range extent therefore REFUSES -- `fuse::Refusal`
/// carrying the offending bounds -- which keeps the chain fluent and makes the
/// mistake loud. With `FUSE_ENABLE_EXCEPTIONS=OFF` a refusal reports and
/// aborts, as everywhere else.
///
/// LIFETIME (U6). Two spellings, ref-qualified:
///
///     lvalue field: `f.slice(...)`          -> a BORROWED view, unmanaged
///     rvalue field: `make_f().slice(...)`   -> an OWNING value
///
/// The rvalue case works because `Kokkos::subview` of a MANAGED view is itself
/// managed and shares the reference count, so the allocation outlives the
/// temporary without a copy. The dangling spelling -- taking a borrowed view
/// out of a temporary -- is `= delete`d, so it is a compile error rather than
/// a use-after-free (there is a compile-fail test for it).

#include <cstddef>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/field/extent.hpp"
#include "fuse/field/field_ref.hpp"

namespace fuse {

/// Resolves an `Extent` against a dimension of size `n`, refusing rather than
/// clamping. The bounds are in the message because "out of range" without
/// them sends the reader back to a debugger for something the library already
/// knows.
[[nodiscard]] inline std::size_t resolve_hi(const Extent& e, std::size_t n,
                                            const char* dim) {
  const std::size_t hi = e.hi_is_end ? n : e.hi;
  if (e.lo > hi) {
    FUSE_REFUSE("E-FIELD-EXTENT-REVERSED",
                "{} extent [{}, {}) is reversed; a slice is half-open and its "
                "lower bound must not exceed its upper", dim, e.lo, hi);
  }
  if (hi > n) {
    FUSE_REFUSE("E-FIELD-EXTENT-RANGE",
                "{} extent [{}, {}) reaches past the end of the {} available; "
                "the bound is not clamped because a clamp turns wrong "
                "arithmetic into a shorter loop that still produces numbers",
                dim, e.lo, hi, n);
  }
  return hi;
}

/// A borrowed, strided, two-dimensional view of a field: elements by
/// components. Unmanaged, so constructing one and taking subviews of it cannot
/// allocate -- which is the property the fence test pins.
template <class MemSpace = Kokkos::HostSpace>
class FieldSpan {
 public:
  using view_type =
      Kokkos::View<double**, Kokkos::LayoutStride, MemSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  explicit FieldSpan(view_type v) noexcept : v_(v) {}

  /// Wraps a registered field. `FieldRef`'s two strides map exactly onto a
  /// `LayoutStride`, which is why the pipeline needs no separate layout
  /// notion: interleaved and blocked storage are both just stride pairs.
  static FieldSpan from(const FieldRef& f) {
    if (!f.valid()) {
      FUSE_REFUSE("E-FIELD-NULL",
                  "the field is not bound to any storage, so there is nothing "
                  "to view");
    }
    const Layout& l = f.layout();
    Kokkos::LayoutStride ls(l.count, l.stride,
                            static_cast<std::size_t>(l.components),
                            l.effective_component_stride());
    return FieldSpan(view_type(const_cast<double*>(f.data()), ls));
  }

  // --- non-terminal stages: lazy, allocation-free, [[nodiscard]] -----------
  //
  // [[nodiscard]] because a discarded stage is always a bug: the call has no
  // effect whatsoever, so writing it and dropping it cannot have been intended.

  [[nodiscard]] FieldSpan slice(const Extent& elements,
                                const Extent& components) const {
    const std::size_t e_hi = resolve_hi(elements, v_.extent(0), "element");
    const std::size_t c_hi = resolve_hi(components, v_.extent(1), "component");
    return FieldSpan(Kokkos::subview(
        v_, Kokkos::make_pair(elements.lo, e_hi),
        Kokkos::make_pair(components.lo, c_hi)));
  }

  /// One component, keeping the element dimension. `component(1)` of a
  /// 3-vector field is its y values, still addressable as `(i, 0)`.
  [[nodiscard]] FieldSpan component(int c) const {
    return slice(all, Extent{c});
  }

  /// The leading `n` elements -- the owned ones, when a partitioned field
  /// stores owned before ghost. It does NOT ask the mesh which those are: the
  /// caller states the count, because a field view has no partition of its own
  /// and guessing one from the mesh would silently be wrong for a field
  /// registered over a different entity set.
  [[nodiscard]] FieldSpan owned(std::size_t n) const {
    return slice(Extent{0, n}, all);
  }

  // --- local terminals -----------------------------------------------------

  [[nodiscard]] std::size_t element_count() const noexcept {
    return v_.extent(0);
  }
  [[nodiscard]] std::size_t component_count() const noexcept {
    return v_.extent(1);
  }
  [[nodiscard]] const view_type& kokkos_view() const noexcept { return v_; }

  [[nodiscard]] double at(std::size_t i, std::size_t c = 0) const {
    if (i >= v_.extent(0) || c >= v_.extent(1)) {
      FUSE_REFUSE("E-FIELD-EXTENT-RANGE",
                  "({}, {}) is outside this view's {} x {}", i, c,
                  v_.extent(0), v_.extent(1));
    }
    return v_(i, c);
  }

  void fill(double value) const {
    for (std::size_t i = 0; i < v_.extent(0); ++i) {
      for (std::size_t c = 0; c < v_.extent(1); ++c) v_(i, c) = value;
    }
  }

  /// A LOCAL sum over this rank's part of the view. Named `local_sum` and not
  /// `sum` deliberately: `sum` over a distributed field is a collective, and a
  /// function that silently returns one rank's answer under that name is how a
  /// reduction becomes wrong without anyone noticing. The collective lives in
  /// `fuse/field/collective.hpp` and takes a communicator.
  [[nodiscard]] double local_sum() const {
    double s = 0.0;
    for (std::size_t i = 0; i < v_.extent(0); ++i) {
      for (std::size_t c = 0; c < v_.extent(1); ++c) s += v_(i, c);
    }
    return s;
  }

 private:
  view_type v_;
};

/// A field that owns its storage. The pipeline needs one so that the rvalue
/// lifetime rule has something to be about: a `FieldSpan` always borrows, and
/// borrowing from a temporary is the bug U6 exists to prevent.
template <class MemSpace = Kokkos::HostSpace>
class OwnedField {
 public:
  using view_type = Kokkos::View<double**, Kokkos::LayoutStride, MemSpace>;

  OwnedField(const std::string& label, std::size_t elements,
             std::size_t components)
      : v_(Kokkos::view_alloc(label, Kokkos::WithoutInitializing),
           Kokkos::LayoutStride(elements, components, components, 1)) {
    Kokkos::deep_copy(v_, 0.0);
  }
  explicit OwnedField(view_type v) noexcept : v_(v) {}

  /// BORROW (lvalue only). Deleted on an rvalue, which makes
  /// `make_field().span()` a compile error instead of a dangling view. This
  /// deletion is the whole of U6's enforcement and there is a compile-fail
  /// test for it.
  [[nodiscard]] FieldSpan<MemSpace> span() const& {
    return FieldSpan<MemSpace>(typename FieldSpan<MemSpace>::view_type(
        v_.data(), v_.layout()));
  }
  [[nodiscard]] FieldSpan<MemSpace> span() const&& = delete;

  /// Chaining on an rvalue yields an OWNING value. `Kokkos::subview` of a
  /// managed view is managed and shares the reference count, so the buffer
  /// outlives the temporary with no copy -- the count is incremented, nothing
  /// is allocated.
  [[nodiscard]] OwnedField slice(const Extent& elements,
                                 const Extent& components) && {
    const std::size_t e_hi = resolve_hi(elements, v_.extent(0), "element");
    const std::size_t c_hi = resolve_hi(components, v_.extent(1), "component");
    return OwnedField(Kokkos::subview(
        v_, Kokkos::make_pair(elements.lo, e_hi),
        Kokkos::make_pair(components.lo, c_hi)));
  }
  /// On an lvalue the same call borrows, because the field outlives the chain.
  [[nodiscard]] FieldSpan<MemSpace> slice(const Extent& elements,
                                          const Extent& components) const& {
    return span().slice(elements, components);
  }

  [[nodiscard]] std::size_t element_count() const noexcept {
    return v_.extent(0);
  }
  [[nodiscard]] std::size_t component_count() const noexcept {
    return v_.extent(1);
  }
  [[nodiscard]] const view_type& kokkos_view() const noexcept { return v_; }

 private:
  view_type v_;
};

}  // namespace fuse
