#pragma once

#include <cstddef>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/value_traits.hpp"
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace fuse {

/// Where the wrapped pointer lives. A registered device pointer stays on the
/// device; `Transport` decides whether host staging is required (spec 4.2).
enum class MemorySpace { Host, Device };

[[nodiscard]] constexpr const char* to_cstring(MemorySpace s) noexcept {
  switch (s) {
    case MemorySpace::Host:   return "Host";
    case MemorySpace::Device: return "Device";
  }
  return "Unknown";
}

namespace detail {

inline constexpr std::size_t kSizeMax = std::numeric_limits<std::size_t>::max();

/// Saturating multiply/add, used only by Layout::well_formed(). Saturation
/// rather than wraparound means an absurd layout is *rejected* instead of
/// silently folding back into a plausible-looking small offset.
[[nodiscard]] constexpr std::size_t sat_mul(std::size_t a,
                                            std::size_t b) noexcept {
  if (a == 0 || b == 0) return 0;
  return (a > kSizeMax / b) ? kSizeMax : a * b;
}

[[nodiscard]] constexpr std::size_t sat_add(std::size_t a,
                                            std::size_t b) noexcept {
  return (a > kSizeMax - b) ? kSizeMax : a + b;
}

}  // namespace detail

/// Element and component strides, in units of the field's value type `T`
/// (not bytes; for the binary64 `FieldRef`, in units of `double`).
///
///   - `stride`           step from element `i` to element `i + 1`.
///   - `component_stride` step from component `c` to component `c + 1`;
///                        `0` is shorthand for `1`, i.e. interleaved / AoS
///                        (component-fastest).
///   - `count`            number of elements addressable through the view.
///   - `components`       components per element; must be >= 1.
///
/// The two strides together cover AoS and SoA without a separate enum:
///
///   interleaved (x0 y0 z0 x1 y1 z1): stride = 3, component_stride = 1 (or 0)
///   blocked     (x0 x1 y0 y1 z0 z1): stride = 1, component_stride = 2
///
/// Note that `component_stride = 0` does *not* imply anything about `stride`:
/// an interleaved 3-vector still needs `stride >= 3`. `well_formed()` is what
/// catches the common mistake of leaving `stride` at its default of 1 while
/// setting `components = 3`.
struct Layout {
  std::size_t stride = 1;
  std::size_t component_stride = 0;
  std::size_t count = 0;
  int         components = 1;

  /// The component step actually used by the accessors: 0 means interleaved,
  /// which is a step of one value.
  [[nodiscard]] constexpr std::size_t effective_component_stride()
      const noexcept {
    return component_stride == 0 ? 1 : component_stride;
  }

  /// True when every valid `(element, component)` pair maps to a distinct,
  /// non-wrapping offset.
  ///
  /// This is a correctness property, not a stylistic one: an aliasing layout
  /// does not crash, it silently makes two logically distinct degrees of
  /// freedom the same value, which shows up much later as wrong physics.
  /// FieldRegistry::bind() therefore rejects a layout that fails this, in
  /// every build configuration.
  [[nodiscard]] constexpr bool well_formed() const noexcept {
    if (components < 1) return false;
    if (count == 0) return true;  // an empty view addresses nothing

    const std::size_t cs = effective_component_stride();
    const std::size_t nc = static_cast<std::size_t>(components);

    if (count > 1) {
      // Either each element owns a contiguous run of components (AoS) or
      // each component owns a contiguous run of elements (SoA). A zero
      // element stride collapses all elements onto one slot, hence the
      // explicit `stride >= 1` in the blocked case; the interleaved case
      // gets it for free from `cs >= 1`.
      // The necessary-and-sufficient bound for element i's component block
      // [i*stride, i*stride + (nc-1)*cs] to never reach element (i+1)'s block
      // at (i+1)*stride is stride > (nc-1)*cs, i.e. stride >= (nc-1)*cs + 1 --
      // not stride >= nc*cs, which is one full `cs` more conservative than
      // necessary and wrongly rejects a valid padded/strided layout (REVIEW.md
      // R-29). Symmetrically for the blocked case: cs > (count-1)*stride.
      const bool interleaved =
          stride > detail::sat_mul(nc - 1, cs);
      const bool blocked =
          stride >= 1 && cs > detail::sat_mul(count - 1, stride);
      if (!interleaved && !blocked) return false;
    }

    // The largest offset any accessor can produce must not have saturated.
    const std::size_t last =
        detail::sat_add(detail::sat_mul(count - 1, stride),
                        detail::sat_mul(nc - 1, cs));
    return last != detail::kSizeMax;
  }
};

/// Non-owning view over component-owned storage (spec 4.2), templated on the
/// field's value type `T` (fuse/model/value_traits.hpp). `FieldRef` is the
/// binary64 instance, and every interface that predates value types uses it.
///
/// LIFETIME CONTRACT. `FieldRef` never allocates, never copies the data, and
/// never takes ownership. It holds a raw `T*` supplied by the component
/// that owns the array, plus a layout describing how to walk it. Copying a
/// `FieldRef` copies the pointer, not the data, and every copy writes through
/// to the same owner storage.
///
/// The owner must keep the pointed-to storage alive, at a fixed address, for
/// as long as any `FieldRef` to it -- or any `FieldRegistry` holding one -- is
/// in use. In particular a component must not register `vec.data()` and then
/// grow `vec`: that reallocates and every registered view dangles. Register
/// storage whose address is stable for the run (the coupler resolves once at
/// setup and dereferences in the hot loop).
///
/// A default-constructed `FieldRef` is the "unbound" state: `valid()` is
/// false, `data()` is null, and `at()` throws.
template <CoupledValue T>
class BasicFieldRef {
 public:
  using value_type = T;

  BasicFieldRef() = default;
  BasicFieldRef(T* data, Layout layout, MemorySpace space)
      : data_(data), layout_(layout), space_(space) {}

  [[nodiscard]] T*       data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] const Layout& layout() const noexcept { return layout_; }
  [[nodiscard]] MemorySpace   space() const noexcept { return space_; }
  [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::size_t size() const noexcept { return layout_.count; }

  /// Unchecked index arithmetic, for hot loops that have already validated
  /// their bounds: `ref.data()[ref.offset(i, c)]`. Prefer `at()` anywhere
  /// the indices are not obviously in range.
  [[nodiscard]] std::size_t offset(std::size_t i, int c) const noexcept {
    return i * layout_.stride +
           static_cast<std::size_t>(c) * layout_.effective_component_stride();
  }

  /// Bounds-checked element access; throws `std::out_of_range` on an
  /// out-of-range element, an out-of-range or negative component, or an
  /// unbound view.
  ///
  /// Checked unconditionally rather than via assert(): under `-O3 -DNDEBUG`
  /// an assert compiles out and a bad index becomes a silent heap overflow.
  /// A negative `c` is the sharpest case -- `static_cast<std::size_t>(-1)`
  /// turns into an enormous positive offset.
  [[nodiscard]] T& at(std::size_t i, int c = 0) {
    return data_[checked_offset(i, c)];
  }
  [[nodiscard]] T at(std::size_t i, int c = 0) const {
    return data_[checked_offset(i, c)];
  }

 private:
  [[nodiscard]] std::size_t checked_offset(std::size_t i, int c) const {
    if (data_ == nullptr) {
      FUSE_THROW(InvalidArgument, "{}", "FieldRef::at: view is not bound to any storage");
    }
    if (i >= layout_.count) {
      FUSE_THROW(InvalidArgument, "{}", "FieldRef::at: element index " +
                              std::to_string(i) + " is outside [0, " +
                              std::to_string(layout_.count) + ")");
    }
    if (c < 0 || c >= layout_.components) {
      FUSE_THROW(InvalidArgument, "{}", "FieldRef::at: component " + std::to_string(c) +
                              " is outside [0, " +
                              std::to_string(layout_.components) + ")");
    }
    return offset(i, c);
  }

  T*          data_ = nullptr;
  Layout      layout_{};
  MemorySpace space_ = MemorySpace::Host;
};

/// The binary64 field view: every interface that predates value types.
using FieldRef = BasicFieldRef<double>;

// Spec 4.2, mechanically enforced: a view that owned anything could not be
// trivially copyable or trivially destructible.
static_assert(std::is_trivially_copyable_v<FieldRef>);
static_assert(std::is_trivially_destructible_v<FieldRef>);

/// A field view with its value type erased, so that one registry can hold
/// fields of different value types. It keeps the `FieldValueType` it was made from
/// and hands back a typed view only for that same type: a request for another
/// type is refused (`E-FIELD-VALUE-TYPE`) rather than reinterpreting the bytes,
/// and no conversion ever happens here. Converting a field's values is a
/// pipeline stage with a named rounding, never a side effect of reading it.
class AnyFieldRef {
 public:
  AnyFieldRef() = default;
  template <CoupledValue T>
  explicit AnyFieldRef(BasicFieldRef<T> ref) noexcept
      : data_(ref.data()),
        layout_(ref.layout()),
        space_(ref.space()),
        type_(field_value_type_of<T>()) {}

  [[nodiscard]] const FieldValueType& value_type() const noexcept { return type_; }
  [[nodiscard]] const Layout& layout() const noexcept { return layout_; }
  [[nodiscard]] MemorySpace space() const noexcept { return space_; }
  [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

  /// True when `as<T>()` would succeed.
  template <CoupledValue T>
  [[nodiscard]] bool holds() const noexcept {
    return type_ == field_value_type_of<T>();
  }

  template <CoupledValue T>
  [[nodiscard]] Result<BasicFieldRef<T>> as() const {
    if (!holds<T>()) {
      const FieldValueType want = field_value_type_of<T>();
      return Failure{"E-FIELD-VALUE-TYPE",
                     "the field holds " + std::string(type_.name) + " values (" +
                         std::to_string(type_.size) + " bytes), but " +
                         std::string(want.name) + " (" +
                         std::to_string(want.size) +
                         " bytes) was requested; a field's value type is fixed "
                         "when it is bound and reading it never converts"};
    }
    return BasicFieldRef<T>(static_cast<T*>(data_), layout_, space_);
  }

 private:
  void*       data_ = nullptr;
  Layout      layout_{};
  MemorySpace space_ = MemorySpace::Host;
  FieldValueType   type_ = field_value_type_of<double>();
};

static_assert(std::is_trivially_copyable_v<AnyFieldRef>);
static_assert(std::is_trivially_destructible_v<AnyFieldRef>);

}  // namespace fuse
