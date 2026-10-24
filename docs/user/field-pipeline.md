# Working with field data

This page is for anyone who reads or writes a registered field from their own
code: a component inspecting what the coupler delivered, a diagnostic that
sums part of an interface, a kernel that needs a Kokkos view of one component
of a vector field. The tools are `fuse::FieldSpan`, `fuse::OwnedField` and
`fuse::Extent` in `include/fuse/field/pipeline.hpp` and
`include/fuse/field/extent.hpp`.

Reading part of a field is a chain of stages that do no work:

```cpp
auto interface = fuse::FieldSpan<>::from(ref)
                     .owned(n_owned)                    // skip the ghosts
                     .slice({first, fuse::end}, fuse::all)
                     .component(1);                     // the y values
```

Every stage returns a **strided view over the same memory** (a
`Kokkos::subview`), so the chain above costs a few integers no matter how large
the field is. Only a terminal (`at`, `local_sum`, `fill`, `kokkos_view`) reads
or writes data.

## A complete example

Six elements of a 3-vector field, stored interleaved (`x0 y0 z0 x1 y1 z1 ...`),
of which the last two are ghosts. The same `FieldRef` a component passes to
`FieldRegistry::bind()` is what `FieldSpan::from()` takes.

```cpp
#include <fuse/field/field_ref.hpp>
#include <fuse/field/pipeline.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <vector>

int main() {
  Kokkos::ScopeGuard kokkos;

  std::vector<double> storage(6 * 3);
  for (std::size_t i = 0; i < 6; ++i)
    for (std::size_t c = 0; c < 3; ++c) storage[i * 3 + c] = 10.0 * i + c;

  const fuse::FieldRef ref{storage.data(),
                           fuse::Layout{.stride = 3, .count = 6, .components = 3},
                           fuse::MemorySpace::Host};

  // Bind the view. Nothing is copied: a FieldSpan is an unmanaged strided view.
  auto f = fuse::FieldSpan<>::from(ref);
  std::printf("shape: %zu elements x %zu components\n",
              f.element_count(), f.component_count());

  // Stages return views; terminals read values.
  const auto owned = f.owned(4);                              // elements 0..3
  const auto tail  = owned.slice({1, fuse::end}, fuse::all);  // elements 1..3 of those
  const auto y     = tail.component(1);                       // their y values
  std::printf("owned=%zu tail=%zu y: %zu x %zu\n", owned.element_count(),
              tail.element_count(), y.element_count(), y.component_count());
  std::printf("y(0)=%g y(2)=%g\n", y.at(0), y.at(2));         // element 1 and 3, component 1

  // One element, all components: a bare index is a one-element range.
  const auto e4 = f.slice(4, fuse::all);
  std::printf("element 4: %g %g %g\n", e4.at(0, 0), e4.at(0, 1), e4.at(0, 2));

  // Terminals that touch every entry of the view.
  std::printf("local_sum over owned z = %g\n", owned.component(2).local_sum());
  f.slice({4, fuse::end}, fuse::all).fill(0.0);               // zero the ghosts in place
  std::printf("storage[12..17] = %g %g %g %g %g %g\n", storage[12], storage[13],
              storage[14], storage[15], storage[16], storage[17]);

  // The underlying Kokkos view, for a kernel of your own.
  const auto& kv = y.kokkos_view();
  double s = 0.0;
  for (std::size_t i = 0; i < kv.extent(0); ++i) s += kv(i, 0);
  std::printf("sum of y over the tail = %g\n", s);

  // Owned storage: slice() on an rvalue owns, on an lvalue borrows.
  fuse::OwnedField<> field("demo", 8, 2);
  auto borrowed = field.slice({2, 5}, fuse::all);              // FieldSpan into `field`
  auto owning   = fuse::OwnedField<>("tmp", 8, 2).slice({2, 5}, fuse::all);  // OwnedField
  std::printf("borrowed %zu, owning %zu elements\n",
              borrowed.element_count(), owning.element_count());

  // A bad extent is refused, not clamped.
  try {
    (void)f.slice({5, 9}, fuse::all);
  } catch (const fuse::Error& e) {
    std::printf("refused: %s\n", e.what());
  }
  return 0;
}
```

Expected output (exact digits can vary by platform):

```
shape: 6 elements x 3 components
owned=4 tail=3 y: 3 x 1
y(0)=11 y(2)=31
element 4: 40 41 42
local_sum over owned z = 68
storage[12..17] = 0 0 0 0 0 0
sum of y over the tail = 63
borrowed 3, owning 3 elements
refused: [E-FIELD-EXTENT-RANGE] element extent [5, 9) reaches past the end of the 6 available; the bound is not clamped because a clamp turns wrong arithmetic into a shorter loop that still produces numbers
```

(The refusal also prints its message and a backtrace to standard error before
it throws; that is the on-error policy every `FUSE_REFUSE` runs.)

## The operations

### Binding

| Call | Result |
|---|---|
| `FieldSpan<>::from(const FieldRef&)` | an unmanaged `Kokkos::View<double**, LayoutStride>` over the registered storage, elements by components. Interleaved and blocked storage are both just the two strides of the `Layout`. Refuses `E-FIELD-NULL` for an unbound `FieldRef` whose `count` is not zero (a rank that owns zero entities legitimately binds a null pointer with `count == 0`). |
| `FieldSpan<MemSpace>` | the template parameter is the Kokkos memory space; the default is `Kokkos::HostSpace`. |

### Stages (lazy, allocation-free, `[[nodiscard]]`)

| Call | Keeps |
|---|---|
| `slice(Extent elements, Extent components)` | the given half-open ranges of both dimensions |
| `component(int c)` | one component, still addressable as `(i, 0)` |
| `owned(std::size_t n)` | the leading `n` elements. The caller states the count: a field view has no partition of its own, and guessing one from the mesh would be wrong for a field registered over a different entity set. |

A discarded stage is a bug (the call has no effect), which is why each is
`[[nodiscard]]`.

### Terminals

| Call | Does |
|---|---|
| `element_count()`, `component_count()` | the view's two extents |
| `at(i, c = 0)` | one value; refuses `E-FIELD-EXTENT-RANGE` outside the view |
| `fill(double)` | writes every entry of the view, in place in the owner's storage |
| `local_sum()` | the sum over every entry of **this rank's** view |
| `kokkos_view()` | the `Kokkos::View` itself, for your own kernel |

There is no `sum()`. A sum over a distributed field is a collective, and a
method named `sum` that returned one rank's partial answer would be wrong
without anyone noticing. For a global sum, reduce `local_sum()` yourself, for
example through `fuse::linalg::Reducer::sum()` (`include/fuse/linalg/comm.hpp`).

## Spelling a slice

The spelling is `std::mdspan`'s (`submdspan`), not a new one. `fuse::Extent`
is a half-open range `[lo, hi)`; `fuse::end` is a tag for "one past the last";
`fuse::all` is exactly `{0, fuse::end}`.

| You write | It means |
|---|---|
| `fuse::all` | the whole dimension |
| `{2, 5}` | elements 2, 3, 4 (half-open, like every other C++ range) |
| `{2, fuse::end}` | element 2 to the last |
| `7` | element 7 only (`Extent` converts from a bare index) |

`{lo, fuse::end}` resolves against the *current* extent, so it stays correct
after an earlier stage has already narrowed the dimension.

A reversed or out-of-range extent is **refused**, with the offending bounds in
the message: `E-FIELD-EXTENT-REVERSED` (`lo > hi`) or `E-FIELD-EXTENT-RANGE`
(`hi` past the end). It is not clamped: clamping turns a caller's wrong
arithmetic into a shorter loop that still produces numbers. The refusal is a
`fuse::Refusal` exception (a `fuse::Error`); with `FUSE_ENABLE_EXCEPTIONS=OFF`
it reports and aborts instead.

## Lifetime

`FieldSpan` always borrows. `OwnedField<MemSpace>(label, elements, components)`
owns a managed, zero-initialised buffer, and its methods are ref-qualified so
that the compiler tells the two spellings apart:

```cpp
fuse::OwnedField<> owned("f", n, 3);
auto a = owned.slice({1, fuse::end}, fuse::all);         // lvalue: a FieldSpan BORROWING from owned
auto b = make_field().slice({1, fuse::end}, fuse::all);  // rvalue: an OwnedField that OWNS the result
auto c = owned.span();                                   // a FieldSpan over the whole field
// make_field().span();                                  // does not compile: span() const&& is deleted
```

Chaining on a temporary yields an owning value: `Kokkos::subview` of a managed
view shares its reference count, so the buffer outlives the temporary with no
copy. Taking a *borrowed* view out of a temporary is the dangling shape, so
that overload is deleted.

The storage behind a `FieldRef` belongs to the component that registered it,
and must stay alive at a fixed address for as long as any view of it is in
use. In particular, never register `vec.data()` and then grow `vec`.

:::{note}
`FieldSpan` and `OwnedField` are the only types with pipeline methods, and the
graph-authoring types (`CouplingGraph`, `ChannelBuilder`) have none; the two
kinds of chain never mix. [](fluent-api.md) explains the distinction.
:::

## Portable kernels

The operations a coupled step spends its time in are templated on a Kokkos
execution space (`include/fuse/num/space_kernels.hpp`):

| Kernel | Runs |
|---|---|
| `accumulate` | once per subcycling step, on every coupled field |
| `gather` / `scatter` | packing and unpacking each data move |
| `remap_apply` | the sparse product that *is* the field transfer |
| `relax` | once per iteration sweep |
| `dot`, `weighted_norm_squared` | the convergence measure |

Each is checked against the plain-loop form in `fuse/num/kernels.hpp` on every
execution space the build enabled. Parameterisation is a precondition for
device performance, not evidence of it: what is verified is that each kernel
compiles for the space and returns the reference answer, with reductions
agreeing to accumulated rounding rather than bitwise, because a parallel
reduction associates differently.

## See also

- [](fluent-api.md) for the two kinds of chain and why they are kept apart
- [](combining-writers.md) for several sources writing one field
- [](component-interface.md) for `register_fields()` and the `FieldRef` contract
