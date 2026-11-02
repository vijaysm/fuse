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

## Value types

A field view is templated on the type of its values. `fuse::FieldRef` and
`fuse::FieldSpan<>` are the binary64 (`double`) views; any other value type is
named explicitly:

```cpp
fuse::BasicFieldRef<float> ref(data, layout, fuse::MemorySpace::Host);
registry.bind(id, ref);                       // the field remembers binary32
auto f = registry.get<float>(id);             // succeeds
auto g = registry.get(id);                    // get<double>(): refused
auto s = fuse::FieldSpan<float>::from(f.value()).component(0);
double total = s.local_sum();                 // accumulated in binary64
```

A field keeps the value type it was bound with. Reading it as another type is
refused with `E-FIELD-VALUE-TYPE`, naming both types; nothing converts on
read. `local_sum()` accumulates in binary64 for every value type, so a binary32
field of 2^24, 1 and 1 sums to 2^24 + 2, where a binary32 accumulator would
give 2^24.

A type can be a field value type when it is trivially copyable and has a
`fuse::value_traits<T>` specialization (`fuse/model/value_traits.hpp`). FUSE
provides binary64, binary32, `fuse::bfloat16`, `fuse::binary16` and the posits `fuse::num::posit16` and
`fuse::num::posit32`; you can add your own:

```cpp
struct Dual3 { double v[4]; };   // a value and a gradient of three entries

template <>
struct fuse::value_traits<Dual3> {
  static constexpr std::string_view name = "myapp:dual3";
  static constexpr fuse::FieldValueKind kind = fuse::FieldValueKind::ScalarArray;
  using scalar_type = double;            // what one scalar is
  static constexpr int scalar_count = 4;      // scalars in one value
  static constexpr int gradient_size = 3;  // of which the last 3 hold the gradient
  static double to_binary64(const Dual3& d) noexcept { return d.v[0]; }
  static Dual3 from_binary64(double x) noexcept { return {{x, 0, 0, 0}}; }
};
```

`kind` states how the type crosses the parts of FUSE that are not templates:

| Kind | Meaning | Examples |
|---|---|---|
| `ScalarArray` | an array of `scalar_count` scalars of `scalar_type`; every field operation acts on each scalar separately | binary64 (one scalar), complex numbers, forward-mode dual numbers |
| `Arithmetic` | `scalar_count` limbs (binary64 values) that are one number; the operators that add or multiply values do it in the type, from a closed list | double-double, quad-double |
| `Storage` | a compact format, widened exactly to binary64 and rounded once on return | binary32, bfloat16, binary16, posit16, posit32 |

`fuse::CoupledValue<T>` checks these requirements at compile time, including
that a `ScalarArray` type's scalars tile it exactly. A type without
`value_traits` does not compile as a field value type. The conversion
`value_traits<fuse::bfloat16>::from_binary64` (and the one for
`fuse::binary16`) is correctly rounded: to nearest, ties to even, in one step
from the binary64 value, because rounding to binary32 first and then to the
narrow format can move a value across a midpoint.

### Storage types: binary32, bfloat16 and binary16

A component may keep a field in single or half precision and still be coupled
to one that does not. Bind the array as it is (`BasicFieldRef<float>`,
`BasicFieldRef<fuse::bfloat16>`, `BasicFieldRef<fuse::binary16>`), or register it
through the C or Fortran interface with `FUSE_ELEMENT_F32`, `FUSE_ELEMENT_BF16`
or `FUSE_ELEMENT_F16`. The graph is unchanged: a field declares components, not
a precision.

What the coupler does with such a field:

- It reads the field into a binary64 image when component code has run, and the
  widening is exact. Remaps, accumulations, the conservation check and the
  transport all see binary64 values.
- It rounds into the field when it delivers a value the component requires, to
  nearest with ties to even, once. So the value a component receives is the
  binary64 result of the same coupling run with binary64 fields, rounded once
  to the receiving type; two coupled components of different precisions need
  nothing more.
- Remap weights, mesh coordinates and search stay binary64; they are not
  fields of a component.

`RunStats::storage_fields` lists every such field the run converts, with its
format and unit roundoff (2^-24 for binary32, 2^-11 for binary16, 2^-8 for
bfloat16): a value in the format's normal range changes by at most that
fraction when it is rounded. Set `RunnerConfig::measure_storage_rounding` to also
count, over every delivery, the finite values rounded, how many overflowed to
infinity or underflowed to zero, and the largest relative change actually made;
it costs one reduction per delivery (and a fence on a device), so it is off by
default.

The conversion runs where the field is. A host field is converted on the host;
a field registered in device memory is converted by a kernel on the device
(`fuse/num/storage_convert.hpp`), so the data does not cross to the host to be
rounded. The three formats are device-callable types
(`fuse/model/binary16.hpp`, `fuse/model/bfloat16.hpp`): conversions use integer
operations on the bit pattern, and `test/num/storage_test.cpp` compares the
device results with the host's over every 16-bit pattern and every midpoint.

What this does not cover:

- **The image costs 8 bytes a value** for each such field arc, in addition to the
  field. MPI and the transport move binary64, not the narrow format.
- **A field in a time window or an iteration region window** must still be
  binary64; the window code refuses a Storage field with `E-FIELD-VALUE-TYPE`.
- **Restart files.** A component stores a field of one of these formats with the
  restart writer and reads it back with the reader, as the format itself
  ([](restart.md#typed-fields)). The Runner does not write a component's arrays
  to a restart file, and no entry of the C and Fortran participant interface does
  either. A provider plugin's fields may have these formats, in host memory (see
  [](providers.md)).
- **Minimum and maximum accumulation** work: the minimum of the widened values
  is the widened minimum, and it is rounded once on delivery.
- A format other than these and the posits below is refused with `E-RT-FIELD-TYPE`,
  naming the type; adding one is a new instantiation of the kernels, not a
  registration.

### Storage types: posit16 and posit32

`fuse::num::posit16` and `fuse::num::posit32` (`fuse/num/posit.hpp`) are the 16-bit and
32-bit posits of the 2022 Posit Standard, `posit<16, 2>` and `posit<32, 2>` (two
exponent bits at both widths). They are Storage types like binary32 and bfloat16:
bind the array as it is (`BasicFieldRef<fuse::num::posit16>`), or register it through the
C or Fortran interface with `FUSE_ELEMENT_POSIT16` or `FUSE_ELEMENT_POSIT32` (an array of
the 16-bit or 32-bit patterns), and the coupler widens it exactly to binary64 and rounds
once into it, as described above. What differs from the IEEE formats:

- **Tapered precision.** A posit has the most fraction bits near one (11 for `posit16`,
  27 for `posit32`) and fewer as the magnitude grows or shrinks, so its relative rounding
  error is not one number. `RunStats::storage_fields` reports the unit roundoff at its
  best, 2^-12 and 2^-28, which is what a value near one sees.
- **Range.** `posit16` covers 2^-56 to 2^56 and `posit32` 2^-120 to 2^120.
- **Saturation.** A posit has no infinity and no signed zero, and rounding never
  produces zero from a nonzero value or overflows: a value past the largest posit becomes
  the largest posit, and a nonzero value below the smallest becomes the smallest. With
  `RunnerConfig::measure_storage_rounding` those are counted as overflow and underflow,
  like an infinity or a zero in the IEEE formats. Delivering into a posit field is
  therefore the one conversion whose result is never infinite.
- **NaR.** The single pattern `1000...0` (Not a Real) stands for every NaN and infinity: a
  binary64 NaN or infinity becomes NaR on delivery, and NaR widens to a quiet NaN. It is
  the most negative pattern in the order of the patterns, which is the order of the values.
- **Rounding.** To nearest, ties to the even pattern, once, from binary64, by the rule of
  the standard: the boundary between two neighbours is the value of the pattern one bit
  longer, so it is the midpoint of the two where they differ in a fraction bit and a
  different point where they differ in an exponent or regime bit. `posit_test.cpp` checks
  every boundary of `posit16`.

In component code `posit16` and `posit32` are device-callable types: `+ - * /` and
`sqrt` return the correctly rounded posit of the exact result (the same saturating rule),
the comparison operators are the integer order of the patterns, and `bits()` and
`from_bits()` give the pattern. `test/num/posit_test.cpp` checks every 16-bit pattern,
the boundaries between neighbours, and the arithmetic against exact values;
`test/num/posit_universal_test.cpp` compares the patterns with stillwater-sc/universal
(see "Testing against Universal" in the section on double-double and quad-double).

### Forward-mode gradients

`fuse::num::dual<S, N>` (`fuse/num/dual.hpp`) is a forward-mode automatic
differentiation number: a value and its gradient with respect to `N`
parameters, propagated by the rules of calculus through `+ - * /`, `sqrt`, `exp`, `log`,
`sin`, `cos`, `abs` and `pow` with a constant exponent. It runs inside a Kokkos
kernel on any execution space. In `value_traits` it is a `ScalarArray` of
`N + 1` scalars of type `S`, the first the value and the last `N` its gradient
(`gradient_size = N`).

Bind a field of them like any typed field. The graph counts scalars per entity,
so a one-component field of `dual<double, 3>` is declared with `components = 4`.

```cpp
std::vector<fuse::num::dual<double, 3>> t(n_cells);
registry.bind(id, fuse::BasicFieldRef<fuse::num::dual<double, 3>>(
                      t.data(), fuse::Layout{1, 1, n_cells, 1},
                      fuse::MemorySpace::Host));
auto scalars = registry.get_scalars<double>(id);   // 4 binary64 components
```

The runtime moves such a field as scalars (`FieldRegistry::get_scalars`). A remap
`y = W x` has weights that do not depend on the field, so the gradient of `y` is
`W` applied to the gradient of `x`. The tests check that bit for bit, both on the
remap kernel on every enabled execution space (`num.dual`) and through a
`Runner` and the backend's own remap (`runtime.dual_field_runner`). They also
check the Taylor remainder of the Brusselator species residual whose directional
derivative comes from dual numbers pushed through the remap kernel: order 2 for the
right gradient, order 1 when the gradient is left out of the remap.

What this does not cover:

- **Minimum and maximum accumulation** of a field with a gradient is refused
  at `Runner::initialize` with `E-RT-GRADIENT-MINMAX`: the gradient of a
  minimum is that of the sample picked, which comparing the gradient scalar by
  scalar does not give. Sum, mean and integral are linear and are accepted.
- **An affine unit conversion** (an offset, such as degrees Celsius to kelvin)
  would add the offset to the gradient as well, so the compiler refuses that step
  for every field (`E-IR-UNLOWERABLE`); a gradient is never offset.
- **The components must be interleaved** (`component_stride` 0 or 1). A blocked
  layout is refused with `E-FIELD-SCALAR-LAYOUT` rather than permuted.
- **A field must say it has a gradient.** Typed C++ registration does, through
  `value_traits`. A C or Fortran participant or a provider plugin says it with
  `gradient_size` in its field view or row
  ([](c-abi.md#fields-field_handle-and-register_field)), and the runtime then
  refuses a minimum or maximum accumulation of that field too. A gradient
  carried as extra components with `gradient_size` left at `0` moves correctly,
  but the runtime cannot tell it from values, so it does not refuse that
  accumulation. A field in an iteration region or a time window must be
  binary64, so it cannot be a field of dual numbers.
- **Restart files** store a field of dual numbers as its scalars with
  `fuse::put_values()` ([](restart.md#fields-of-double-double-quad-double-and-dual-values)).
- **Reverse mode** is a separate mechanism, below: a field with a gradient is
  forward mode, and reverse mode needs the component to supply its own adjoint.

### Reverse mode (adjoints)

Forward mode carries a gradient with each value. When there are many inputs and
few outputs, reverse mode is cheaper: it computes `G^T y`, the derivative of a
scalar of the final state with respect to every input, in one backward sweep.
FUSE supplies the adjoint of every coupling operator it applies (the transpose of
a remap, the factor of a unit conversion, the fan-out of a sum or a mean); the
component supplies the adjoint of its own `advance()` with whatever tool it uses.
No tape crosses the coupler: only arrays of binary64 adjoint variables move between a
component and FUSE, which is what lets the same design reach components written in
other languages (C and Fortran, see below) and running in other processes.

A component that can be reversed returns itself from `Component::adjoint()` and
implements `fuse::AdjointComponent` (`fuse/runtime/component.hpp`):

```cpp
class MyModel : public fuse::Component, public fuse::AdjointComponent {
  fuse::AdjointComponent* adjoint() override { return this; }
  // Bind an array of adjoint variables for each field, under the field's FieldId.
  fuse::Result<void> register_adjoint_fields(fuse::FieldRegistry&, const fuse::CouplingGraph&) override;
  // The adjoint of advance(ctx), for the same ctx.
  fuse::Result<void> advance_adjoint(const fuse::StepContext& ctx) override;
};
```

The run is a forward sweep followed by a backward one:

```cpp
runner.initialize(components, registry);
for (auto& c : components) c.adjoint()->register_adjoint_fields(adjoint_registry, graph);
runner.initialize_adjoint(components, adjoint_registry);
runner.run(components, n_periods);              // forward
// write the adjoint of the final state into the adjoint arrays (the seed)
for (int i = 0; i < n_periods; ++i) runner.run_period_adjoint(components);   // newest period first
// the adjoint arrays now hold the adjoint of the initial state
```

The component's side of the contract, for the adjoint array `Fbar` of a field F: for a
field it provides, `Fbar` enters `advance_adjoint` holding the adjoint of F as the
step left it (the coupler has added the adjoint of everything that read F later) and
must leave holding the adjoint of F before the step; for a field it requires, the
component adds the adjoint of each value it read and leaves the rest alone (the
coupler zeroes it when it reverses the delivery). The component keeps the forward
values its own adjoint needs; FUSE keeps none. The full contract, the rule for each
operator and the checks are in [](../dev/adjoint-schedule.md).

The adjoint array of a field is in the memory the field is in, host or device, and the
reverse sweep runs on the execution space the forward run used. Across ranks the
exchange of a channel (the MPI transport) is run backwards, and a rank that fails in
the middle of a sweep does not leave the ranks it exchanges with waiting: they are
told at the next agreement point and every rank returns an error. Supported:
binary64 fields, in one process or across ranks, remap, unit
conversion, sum, mean and integral accumulation, merge by first, sum or mean, and
lagged channels, and ordinary iteration regions on one rank (reversed as the fixed point
they converged to: the components they step have `advance_adjoint` called several times for the same
step, and an adjoint iteration that does not converge is an error, not a derivative). Refused by name
(`E-ADJ-*`) at `initialize_adjoint`: Newton and time-window regions, regions across ranks, minimum and
maximum accumulation, time histories (space-time channels),
a transport that has no adjoint (LowFive), distributed nodal remaps, fields of the dd, qd and
Storage types, an adjoint array in another memory space than its field, and the
operators with no adjoint (relaxation, residual). A component the C++ Runner steps that is driven from
outside (the participant ABI) is reversed with `Runner::resume_period_adjoint()`, which hands the caller
each of its steps in turn. From C and Fortran, a code that drives its own components registers their
adjoint arrays and reverses its steps between `fuse_begin_adjoint_step` and `fuse_end_adjoint_step`
([](c-abi.md#reverse-mode), [](fortran.md)), and a provider plugin implements `bind_adjoint` and
`advance_adjoint` ([](providers.md#adjoint-of-a-plugin)). The Python bindings (`pfuse`) have no adjoint
entry points.

### Double-double and quad-double

`fuse::num::dd` (`fuse/num/dd.hpp`) and `fuse::num::qd` (`fuse/num/qd.hpp`) are
double-double and quad-double numbers: unevaluated sums of two and of four
binary64 values, the limbs, most significant first, with about 106 and 212 bits of
significand over binary64's exponent range. They are plain structs of 16 and 32
bytes with `+ - * /`, `sqrt` and the comparisons, built from error-free
transformations (two-sum, and two-product with a fused multiply-add), so they
run inside a Kokkos kernel on any execution space and never allocate. The
algorithms are those of Hida, Li and Bailey (the QD library). On operands in the
normal range every result is the same bits as stillwater-sc/universal's `dd` and
`qd` (see the build option below); FUSE itself never includes Universal.

In `value_traits` they are `Arithmetic`: the limbs of one number, not independent
scalars. A remap, a sum or a division of them has to be done in the type, so the
runtime instantiates those operators for the type from a closed list
(`fuse/num/arithmetic_types.def`) and refuses anything not in it. Bind a field
like any typed field. The graph counts limbs per entity, so a one-value-per-cell
field of `dd` is declared with `components = 2` (`qd`: 4):

```cpp
std::vector<fuse::num::dd> t(n_cells);
registry.bind(id, fuse::BasicFieldRef<fuse::num::dd>(
                      t.data(), fuse::Layout{1, 1, n_cells, 1},
                      fuse::MemorySpace::Host));
auto limbs = registry.get_limbs<double>(id);   // 2 binary64 components per cell
```

`FieldRegistry::get_scalars` refuses such a field (`E-FIELD-VALUE-TYPE`): an
operator written for independent scalars, such as the remap of a field of dual
numbers, would otherwise treat the two limbs as two unrelated values.

**What the Runner does with them.** The operators that add or multiply values
work in the field's type: a remap (each row summed in `dd` or `qd` with the products of
the weights and the values formed exactly), accumulation by sum, mean, integral,
minimum and maximum (a mean is one division by the count in the type), a merge by
sum or mean, scaling, and the conservation check. The operators that only move
values (source, sink, transport, epoch, the identity transfer) move the limbs
unchanged. The remap weights stay binary64.

**The conservation check.** The two integrals are accumulated in the field's type,
added over the ranks in the type, and subtracted in the type, so the error the
check reports is that of the stored values and not the rounding error of the sums
that measure it. The per-rank integrals travel in one `MPI_Allgather` and are
added in rank order (`parallel.arithmetic_reduce_mpi`, at one to five ranks, checks the
bits against the ordered sum and the accuracy against the exact one), which gives the same bits on every rank and does not depend
on the shape of a reduction tree (a floating-point sum is not associative, so a
user-defined `MPI_Op` would). On a remap whose weights are exactly representable
(eight cells onto four, an intensive field) the defect of a `dd` field is 4e-32,
the round-off of the type; the same kind of data held in binary64 shows about 1e-17,
which is binary64's (`runtime.arithmetic_field_runner`).

**Accumulating a binary64 field in double-double.** `RunnerConfig::accumulation_type`
(default `AccumulationType::Binary64`) set to `DoubleDouble` makes the row sums of
the remap, and the integrals and their comparison in the conservation check, of a
binary64 field be accumulated in `dd` and rounded once: each remapped value is
the correctly rounded row sum, and the reported conservation error is the exact
defect of the stored binary64 values. It costs several times the arithmetic of a
binary64 sum, so it is off by default; it changes nothing for fields that are
already `dd` or `qd`.

**Accuracy.** The tests compare each operation with exact arithmetic (sums of
binary64 values with error-free transformations only, which share no code with
the types): `+ - * /` and `sqrt` of `dd` are within 2^-102 of the exact result, those of
`qd` within 2^-206 (the largest errors the tests observed: 2^-107 for the `dd`
sum, 2^-211 for `qd`). Through a
Runner and a real remap the transferred values are within 2^-108 (`dd`) and
2^-217 (`qd`) of the exact result for the weights the operator applies, where the
same transfer of the leading limbs alone, in binary64, is further than 1e-18.
The kernels give the same bits on every enabled execution space and match a
plain host loop in the documented summation order (`num.arithmetic_kernels`).

What this does not cover:

- **Iteration regions, relaxation and residuals** over a `dd` or `qd` field are
  refused with `E-RT-ARITHMETIC-OP`, as are the space-time projection and the
  distributed nodal remap: the residual norms and the accelerators are binary64.
- **Mixing types across a channel** (a `dd` field delivered into a binary64 field,
  or a merge of a `dd` and a binary64 field) is refused with `E-RT-ARITHMETIC-MIX`;
  the coupler does not convert between two fields' types.
- **An Arithmetic type not in the closed list** is refused with `E-RT-FIELD-TYPE`.
  Adding one is a line in `arithmetic_types.def`, its limb access in
  `fuse/num/arithmetic_kernels.hpp`, and a row in the kernel tests.
- **Results near the ends of the exponent range** (a product or a square of a limb
  that overflows or goes subnormal) are not specified, as in the QD library.
- **Restart files** store them with `fuse::put_values()` and `fuse::get_values()`
  ([](restart.md#fields-of-double-double-quad-double-and-dual-values)).
  `RunnerConfig::accumulation_type` is not reachable from the C and Fortran
  interface.
- **A field registered through the C or Fortran interface** is a `dd` or `qd` field
  only if it says so, with `FUSE_ELEMENT_DD` or `FUSE_ELEMENT_QD`
  ([](c-abi.md#fields-field_handle-and-register_field)). The same array registered
  as `FUSE_ELEMENT_F64` with twice or four times the components is a field of
  independent binary64 components to the runtime: it moves correctly through the
  operators that only move values, but a remap acts on each limb separately, in
  binary64.
- The weights of a remap are computed in binary64, so a `dd` field is remapped
  exactly for the weights as stored, not for the exact geometric overlaps.

**Testing against Universal.** The bit-for-bit comparison builds only with
`-DFUSE_ENABLE_UNIVERSAL=ON -DFUSE_UNIVERSAL_INCLUDE_DIR=<checkout>/include/sw`
(default off; Universal is a header-only checkout, never a dependency of the
library). Universal's headers do not compile under nvcc, so they are built into a
small separate program (`test/num/universal_oracle.cpp`) with a host compiler, and
the tests (`num.dd_universal` and `num.posit_universal`) run that program on files of
operands and compare what it returns. The compiler is `FUSE_UNIVERSAL_CXX`; it defaults to the project's
C++ compiler, which is right for a host build, and on a CUDA build (where that is
`nvcc_wrapper`) has to be set to `g++` or `clang++`. The double-double test compares
`+ - * /` and `sqrt` of both types on random operands, on operands of very different
magnitudes, on operands that cancel, and on the special values; the posit test compares
patterns pattern for pattern: the decoding of every 16-bit pattern and a sample of 32-bit
ones, the conversion from binary64 at the boundaries between neighbours and at random, and
the five operations.

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
