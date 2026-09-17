# Third-party notices

FUSE itself is BSD 3-Clause (see `LICENSE`). The components below are obtained
by the build and carry their own terms, reproduced here as those terms require.

## r3d

Used as the geometric kernel for conservative remap: exact clipping of
simplices and analytic moments (`src/backend/supermesh.cpp`). Pinned in
`cmake/FuseDependencies.cmake`.

- Upstream: https://github.com/devonmpowell/r3d
- Author: Devon Powell

Its authors request that work using it cite:

> D. M. Powell and T. Abel, *An exact general remeshing scheme applied to
> physically conservative voxelization*, Journal of Computational Physics 297
> (2015) 340-356.
>
> LA-UR-15-26964.

The notice carried in its sources, reproduced as required:

> This program was prepared by Los Alamos National Security, LLC at Los Alamos National Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). All rights in the program are reserved by the DOE and Los Alamos National Security, LLC. Permission is granted to the public to copy and use this software without charge, provided that this Notice and any statement of authorship are reproduced on all copies. Neither the U.S. Government nor LANS makes any warranty, express or implied, or assumes any liability or responsibility for the use of this software. 

## yaml-cpp

Used by the configuration front-end (`src/model/yaml_io.cpp`), MIT licensed.

- Upstream: https://github.com/jbeder/yaml-cpp

## Kokkos, MOAB, PETSc, TempestRemap

Resolved from installed prefixes rather than built here, and each carries its
own licence at its own installation. Kokkos is Apache-2.0 with LLVM
exceptions; MOAB is LGPL-2.1-or-later; PETSc is BSD-2-Clause; TempestRemap is
BSD-3-Clause. Nothing from them is vendored into this repository.
