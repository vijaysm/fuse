# Parallel Runs and Checkpoints

## Parallel Mesh Modes

KARMA supports replicated SPMD meshes and externally partitioned MOAB meshes. For partitioned work, use H5M with partition metadata and MOAB parallel read options. Tests use `mbpart` to construct 2/3/4-rank fixtures.

KARMA currently consumes and validates partitions; its historical in-library partitioner is not active. Partitioned FE traversal expects positive, globally unique conventional MOAB `GLOBAL_ID` values and required ghost neighbors. Missing ghosts and inconsistent topology fail collectively instead of producing incomplete assembly.

Ranks with no owned elements or DoFs are valid and still participate in required collectives.

## Checkpoint API

`KARMAMoabIO` supports:

- `write_checkpoint()` for mesh, field descriptors, and a PETSc solution;
- `restore_solution()` into an existing compatible system;
- `load_checkpoint()` to construct a mesh and system with an optional custom basis registry.

Exact checkpoints require `.h5m` or `.mhdf`. The schema stores cG values on geometry vertices or in stable global-DoF records, and dG/SE values in element-packed records. Checksums, descriptor validation, exact key coverage, finite-value checks, and transactional restore protect against partial or incompatible data.

Publication stages a separate database, atomically replaces the destination, and synchronizes the parent directory. If final synchronization fails, the destination may already have been replaced and the exception reports that state.

Checkpoints can be restored with a different MPI rank count. The reader remaps stable identities rather than persisted MOAB handles and supports empty or zero-owner ranks.

## Compatibility Boundaries

A checkpoint must match the supported schema, scalar size, real representation, scalar kind, field descriptors, and basis compatibility identifiers. Current builds support double-real PETSc only. VTK and Gmsh are not exact restart formats.
