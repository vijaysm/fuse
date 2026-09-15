# Build and Test

## Build Configuration

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/opt/petsc/download-moab;/opt/eigen3'
cmake --build build -j4
```

The generated `karma_config.h` in the build tree must precede source include directories. Do not restore or prefer stale waf-generated configuration files.

Useful focused checks are:

```bash
ctest --test-dir build -N
ctest --test-dir build -R '^karma_smoke$' -V
ctest --test-dir build -R 'karma_(system|mass|diffusion|advection)' --output-on-failure
ctest --test-dir build -R 'karma_moab_io' --output-on-failure
```

## Test Registration

Active tests are plain C++ executables under `test/`:

```cmake
add_executable(karma_example_test example_test.C)
target_link_libraries(karma_example_test PRIVATE KARMA::karma)
add_test(NAME karma_example COMMAND karma_example_test)
```

MPI tests must use CMake's MPI launcher variables and finite timeouts. Asymmetric failure tests should use short timeouts. Partitioned tests should use CTest fixtures. MOAB I/O tests that share filenames must use `RESOURCE_LOCK karma_moab_io_files`.

## Scientific Verification

Test dimensions, units, signs, finite values, scale-sensitive tolerances, ownership, layouts, mappings, communicator compatibility, serial and multi-rank equivalence, solver convergence reasons, independent residuals, MMS rates, and critical mutations as applicable.

## Sanitizers

There is no canonical sanitizer preset yet. Use a separate build directory and compiler-supported flags, run the focused suite, and record the exact configure command.

## Documentation

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Read the Docs uses `.readthedocs.yaml` and the same pinned requirements. Keep new MyST pages in a Sphinx `{toctree}` and build with warnings as errors.
