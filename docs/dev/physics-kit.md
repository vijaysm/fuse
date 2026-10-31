# physics-kit (S8)

The optional finite-element component: KARMA's FEM core, physics modules (advection-reaction,
diffusion-reaction, Gray-Scott), FEM transfer and the 22-tableau Runge-Kutta/Rosenbrock integrator
family, ported into `fuse::` per [](physics-kit-disposition.md)'s disposition and
[](integrators-design.md)'s design for the temporal integrators. [](roadmap.md)'s S8 row tracks
where the port currently stands; at the skeleton stage (this page) no physics code exists yet --
only the directory, the build option, the two gates and one placeholder test described below.

## What it is, and what it is not

physics-kit is a **separate optional component**, not one of the `fuse::` layers
[](layering.md) enumerates. It depends on FUSE -- the `fuse::physics_kit` target links
`fuse::runtime`, `fuse::backend`, `fuse::num` and the `fuse::linalg` facade
(`physics-kit/CMakeLists.txt`) -- and FUSE never depends back on it: nothing under `include/fuse/`,
`src/`, `python/` or `examples/` may include a `fuse_physics/` header or link `fuse::physics_kit`.
That direction is what keeps FUSE adoptable by a code that already has its own discretisation
([](layering.md)'s own wording). The namespace is `fuse::physics`; headers live under
`physics-kit/include/fuse_physics/`.

## Tier and how to enable it

physics-kit is **tier 0**: it needs nothing beyond what `fuse::runtime`, `fuse::backend`,
`fuse::num` and `fuse::linalg` already need at tier 0 (Kokkos, fmt), so `FUSE_ENABLE_PHYSICS_KIT`
(`cmake/FuseDependencies.cmake`) defaults **ON** -- unlike `FUSE_ENABLE_PYTHON`/`FUSE_ENABLE_FORTRAN`,
which default OFF because they add a real third-party dependency physics-kit does not. PETSc is
**optional**, read through the existing `FUSE_HAVE_PETSC` define (`fuse/base/config.hpp`) with no
new `find_package(PETSc)` call of its own: a physics-kit source guards PETSc-dependent code with
`#if FUSE_HAVE_PETSC`, the same pattern `examples/dn_heat/main.cpp` and
`examples/petsc_reaction_diffusion/main.cpp` already use, and `physics-kit/CMakeLists.txt` links
`fuse::petsc` only when `FUSE_ENABLE_PETSC` is on.

```
cmake -S . -B build -DFUSE_ENABLE_PHYSICS_KIT=ON    # default; needs nothing beyond tier 0
cmake -S . -B build -DFUSE_ENABLE_PHYSICS_KIT=OFF   # skip the component entirely
```

## The gates

Two static checks keep the boundary and the naming honest, both run as part of `tools/check-
layering.py`'s and the new `tools/check-physics-kit-naming.py`'s CI job:

- **Layering** (`tools/check-layering.py`'s `check_physics_kit_boundary`): refuses a
  `#include ".../fuse_physics/..."` or a `fuse::physics_kit` link anywhere under `include/fuse/`,
  `src/`, `python/` or `examples/`. physics-kit/ itself is exempt -- it is the one place allowed to
  name any `fuse::` layer, by construction.
- **Naming** (`tools/check-physics-kit-naming.py`): refuses a `KARMA*` type prefix, a `karma_` file
  prefix or a `.C` extension anywhere under `physics-kit/` -- the three rules [](porting.md)'s
  naming map states for code arriving from `upstream/karma`, applied here because physics-kit is
  outside the four directories `tools/check-layering.py`'s own naming check already covers.

Both gates were proven to fire against a probe file (added, run, shown to produce a violation,
removed) before being accepted; the S8 skeleton track's `measured_results` has the verbatim output.

## Status

No physics code lives under `physics-kit/` yet. The skeleton (this track) is: the directory layout,
the `fuse::physics_kit` target with one placeholder translation unit, the `physics.skeleton` gtest
(asserts a version string and that `FUSE_HAVE_PETSC` reaches physics-kit code), the two gates above,
and this page. [](physics-kit-disposition.md) and [](integrators-design.md) are the read-first design
pages for what a later track ports here.

## Installation and the `physics_kit` component (S8.2, lead, 2026-10-31)

`fuse_physics_kit` ships in its own export set, `fuse-physics-kit-targets`, written beside the main
`fuse-coupler-targets` in `lib/cmake/fuse-coupler/` (the main set is closed inside `src/CMakeLists.txt`
before the root adds `physics-kit/`, so it cannot be joined). The package configuration includes the
second set only when the install was configured with `FUSE_ENABLE_PHYSICS_KIT`, and exposes it as a
`find_package` component:

```cmake
find_package(fuse-coupler CONFIG REQUIRED COMPONENTS physics_kit)
target_link_libraries(app PRIVATE fuse::physics_kit)
```

Against an install built without physics-kit, `COMPONENTS physics_kit` fails at configure time through
`check_required_components`, and `OPTIONAL_COMPONENTS physics_kit` reports
`fuse-coupler_physics_kit_FOUND` false; the target is never left undefined. The headers install under
`include/fuse_physics/`. Both directions are measured by `install.consumer_find_package`
(`test/install_consumer/`): with physics-kit on, a second consumer links the component and runs; with it
off, the consumer asks optionally and refuses if the component appears.

`tools/check-layering.py` applies the public-header third-party rule to `physics-kit/include/fuse_physics/`
as to any `fuse::` layer: none of PETSc, MPI, MOAB, HDF5, Eigen or Kokkos-Kernels may be named in a header a
consumer compiles, and of Kokkos only `Kokkos_Core.hpp`; a probe header naming `petscsys.h` and
`Kokkos_Random.hpp` produced two violation lines and was removed. `tools/check-invariants.py` has
scanned `physics-kit/` since the skeleton.

