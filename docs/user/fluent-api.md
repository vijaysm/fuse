# The fluent API and its rules

:::{note}
**Planned.** Specified and scheduled; see the [roadmap](../dev/roadmap.md).
:::

FUSE uses method chaining as its primary idiom, in **two distinct forms** that never mix. Knowing
which one you are holding tells you what it costs and how it fails.

## Authoring chains build a description

```cpp
g.component("fluid").ranks({0, 2}).interval(1_r / 2);
```

These accumulate a description and commit nothing. They never return a status, so there is nothing to
check after each call — and nothing is silently dropped either. An instruction FUSE cannot honour is
*recorded* and reported by `validate()`.

(diagnostics-accumulate)=
## Diagnostics accumulate

A chained call has nowhere to put an error, so FUSE does not try. One terminal call reports
everything:

```cpp
if (auto r = g.validate(); !r) {
  fmt::print("{}", r.diagnostics());   // every failure, deterministically ordered
  return 1;
}
```

First-failure reporting is deliberately absent: a coupling graph with ten problems should tell you
about ten problems, not send you around the loop ten times. Status objects appear on `validate()`,
`run()`, `initialize()` and the C ABI entry points — the points where execution actually begins.

## Pipeline chains are lazy

Chains over field *data* return lightweight views and do no work:

```cpp
auto q = ctx.field("fluid", "heat_flux");   // owning handle

q.slice(1, fuse::all)                       // view. no work, no allocation
 .owned()                                   // view. drops halo entries
 .to_file(ctx, "q_wall.h5");                // terminal. this is where work happens
```

Non-terminal operations (`slice`, `owned`, `component`, `restrict`) are `[[nodiscard]]` views over
the underlying storage. Work happens only at a terminal (`to_file`, `copy_to`, `reduce`, `print`,
`commit`). A four-stage chain over a multi-gigabyte field therefore allocates **nothing** — an eager
implementation would allocate three intermediates, and on a large field that is the difference
between running and not.

## Slicing follows `mdspan`

There is no `2:end` in C++, so FUSE uses the spelling standard C++ is converging on rather than
inventing one:

```cpp
q.slice(1, fuse::all);          // whole extent      (cf. std::full_extent)
q.slice(1, {2, fuse::end});     // from 2 to the end (cf. std::strided_slice)
```

A reversed or out-of-range extent is rejected, and the diagnostic names the offending bounds.

## Terminals that communicate say so

`to_file`, `reduce` and `print` on a distributed field **are collective operations**, and their
signatures take the context to make that visible:

```cpp
auto total = q.owned().reduce(ctx, fuse::Sum{});
```

This matters because the tempting thing to write is a deadlock:

```cpp
if (rank == 0) {
  q.slice(1, fuse::all).to_file(ctx, "q.h5");   // rejected, with a diagnostic
}
```

Ranks 1 and above never enter the collective, so a naive implementation hangs at the *next* one,
somewhere unrelated. FUSE rejects the call instead and tells you why. `print` is rank-aware by
construction, so you do not have to remember to guard it.

## Chains cannot dangle

```cpp
auto x = make_field().slice(1, fuse::all);   // safe: owns its data
auto& f = ctx.field("fluid", "heat_flux");
auto y = f.slice(1, fuse::all);              // view into f, which outlives y
```

An rvalue chain returns an owning value; an lvalue chain returns a view. This is the bug class
`auto x = a + b` made famous in Eigen, and ref-qualified overloads remove it rather than documenting
it.
