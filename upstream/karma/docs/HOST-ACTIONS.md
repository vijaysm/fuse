# Host verification

MPI and PETSc tests must run on a host where the toolchain and MOAB
partitioning utility are visible. The isolated checkout used for the current
verification is:

```bash
cd /Users/mahadevan/Code/AI/coupling/karma/.worktrees/usable-plugin-foundation
export PATH=/opt/homebrew/bin:/opt/mpich/bin:$PATH
```

## Release build and complete test matrix

```bash
cmake --preset release
cmake --build --preset release -j 12
ctest --preset release --output-on-failure
```

The current host run passes 425/425 tests in 187.51 seconds. The suite covers
serial and MPI mesh, finite-element, transfer, transaction, restart, physics,
ABI, registry, and coupling behavior. CTest creates isolated working
directories for tests that write scratch files and declares MPI process counts
to avoid oversubscription.

## Focused contract checks

```bash
ctest --test-dir build -R \
  'karma_(semantic_version|coupling_compatibility_oracle)' \
  --output-on-failure
```

This command runs the serial and two-rank compatibility oracle. The oracle
checks that the semantic admission and canonical metadata preserve the existing
coupler's graph, transfer-map, fixed-point, and restart invariants.

## No-exceptions configuration

```bash
cmake --preset no-exceptions
cmake --build --preset no-exceptions -j 12
ctest --test-dir build-no-exceptions -R \
  'karma_(semantic_version|coupling_compatibility_oracle)' \
  --output-on-failure
```

The no-exceptions build uses fail-fast refusal handling. Tests that intentionally
assert thrown refusals are therefore excluded from this focused command.

## Sanitizers

```bash
cmake --preset asan
cmake --build --preset asan -j 12
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --preset asan --output-on-failure
```

The current sanitizer contract check passes 3/3 tests. Leak detection is
disabled because the macOS host runtime does not support LeakSanitizer.

## Documentation

```bash
../../.venv-docs/bin/sphinx-build -E -W --keep-going \
  -b html docs docs/_build/html
```

The warning-clean build includes the developer and scientific guides, the
correspondence records, and the capability roadmap.

## Review notes

The active runtime supports in-process MPI components on one communicator. It
does not launch independent executables, load shared libraries dynamically, or
provide Fortran/Python adapters. Conservative remapping, general graph
partitioning, and zero-copy PETSc transaction binding remain separate work and
must not be inferred from the passing native tests.
