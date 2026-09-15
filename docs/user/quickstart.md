# Quickstart

:::{note}
**Planned.** The interface below is specified and scheduled; the stage that builds it is listed on
the [roadmap](../dev/roadmap.md).
:::

A complete coupling of two components on non-matching meshes, running at different time steps:

```cpp
#include <fuse/fuse.hpp>

int main(int argc, char** argv) {
  fuse::Session session(argc, argv);        // owns Kokkos and MPI only if you do not

  fuse::Graph g;
  auto fluid = g.component("fluid").ranks({0, 2}).interval(1_r / 2);
  auto solid = g.component("solid").ranks({2, 2}).interval(1_r);
  g.channel(fluid["heat_flux"] >> solid["q_wall"]).conservative();

  if (auto r = g.validate(); !r) {          // every failure, not the first
    fmt::print("{}", r.diagnostics());
    return 1;
  }
  return fuse::run(g) ? 0 : 1;
}
```

Four things that example does not say out loud:

**Time is exact.** `1_r / 2` is a rational, not a `double`. Component intervals, coupling windows and
every instruction in the compiled plan use exact arithmetic, so a half-step schedule is still exactly
in step after ten million steps, and the scheduler can *prove* at initialisation that the coupling
cannot deadlock.

**Rank sets are explicit and may be disjoint.** `fluid` runs on ranks 0–1 and `solid` on ranks 2–3,
so the two components execute concurrently rather than taking turns on all four ranks.

**The transfer is conservative by request, and checked.** `.conservative()` selects a remap whose
integral of the transferred field matches the source to round-off, and the property is asserted by a
test rather than claimed.

**`validate()` reports everything.** A graph with five unrelated defects reports five diagnostics in
one call. See [](fluent-api.md#diagnostics-accumulate) for why.

## Seeing what the compiler did

```bash
./my_coupling --dry-run
```

prints the compiled plan: every instruction, its time level, and every operator FUSE inserted that
you did not write — unit conversions, time interpolation, remap, halo exchange. If a coupling behaves
unexpectedly, this output is the first place to look, because it is the thing that actually runs.
