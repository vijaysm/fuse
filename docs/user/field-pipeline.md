# Working with field data

Reading part of a field is a chain of stages that do no work:

```cpp
auto interface = fuse::FieldSpan<>::from(ref)
                     .owned(n_owned)                    // skip the ghosts
                     .slice({first, fuse::end}, fuse::all)
                     .component(1);                     // the y values
```

Every stage returns a **strided view over the same memory** — a
`Kokkos::subview` — so the chain above costs a few integers no matter how large
the field is. That claim is pinned by a test, not by this page: a four-stage
chain on ten million elements runs inside an allocation counter and must record
**zero** allocations. An eager implementation would pass every other test here
and merely be slower, which is why the fence exists.

## Spelling a slice

The spelling is `std::mdspan`'s, not a new one:

| You write | It means |
|---|---|
| `fuse::all` | the whole dimension |
| `{2, 5}` | elements 2, 3, 4 — half-open, like every other C++ range |
| `{2, fuse::end}` | element 2 to the last |
| `7` | element 7 only |

`{lo, fuse::end}` resolves against the *current* extent, so it stays correct
after an earlier stage has already narrowed the dimension.

A reversed or out-of-range extent is **rejected**, with the offending bounds in
the message. It is not clamped: clamping turns a caller's wrong arithmetic into
a shorter loop that still produces numbers.

## Lifetime

Two spellings, and the compiler tells them apart:

```cpp
field.slice(...)                      // lvalue: BORROWS a view
make_field().slice(...)               // rvalue: OWNS the result
make_field().span()                   // does not compile
```

Chaining on a temporary yields an owning value — `Kokkos::subview` of a managed
view shares its reference count, so the buffer outlives the temporary with no
copy. Taking a *borrowed* view out of a temporary is the dangling shape, so
that overload is deleted; there is a compile-fail test for it, paired with a
control proving the permitted spelling still builds.

## Portable kernels

The operations a coupled step spends its time in are templated on a Kokkos
execution space:

| Kernel | Runs |
|---|---|
| `accumulate` | once per subcycling step, on every coupled field |
| `gather` / `scatter` | packing and unpacking each data move |
| `remap_apply` | the sparse product that *is* the field transfer |
| `relax` | once per iteration sweep |
| `dot`, `weighted_norm_squared` | the convergence measure |

Each is checked against the plain-loop form in `fuse/num/kernels.hpp` as an
oracle, on **every execution space the build enabled** — one test definition,
instantiated per space, so device coverage is a build option rather than a
second file that drifts.

:::{note}
**No device performance number is claimed.** Parameterisation is a
precondition for device performance, not evidence of it. What is verified is
that each kernel compiles for the space and returns the oracle's answer.
Reductions agree with the sequential sum *to rounding* rather than bitwise,
because a parallel reduction associates differently — the tolerance is stated
as accumulated rounding rather than tightened until it passes.
:::

## See also

- [](combining-writers.md) for several sources writing one field
