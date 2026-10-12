:::{note}
**Historical document.** Preserved verbatim from `upstream/soma` (parent repository `coupe`,
commit `e52b981`, imported 2026-09-14 per `UPSTREAMS.lock`). Describes the design, build and
handoff state of the retired `soma::` library as it stood at that commit -- not FUSE's current
architecture. See [](../../dev/soma-import.md) for what from this design was ported, superseded,
or remains open.
:::

# Building SOMA

## Configurations

SOMA is tested in three configurations, and all three must pass before a
change is considered done:

- **Debug** — the default. `assert()` is active.
- **Release** — `-O3 -DNDEBUG`. `assert()` compiles out entirely.
- **ASan** — AddressSanitizer + UndefinedBehaviorSanitizer, with
  `-fno-sanitize-recover=all` so a diagnostic is a test failure rather than a
  log line.

These are not equivalent test runs. A correctness guard implemented only via
`assert()` (rather than a `.at()` bounds check or an explicit `throw`)
disappears in Release: an invariant violation that would abort in Debug
instead runs straight through as undefined behavior — silent data corruption,
a heap-buffer-overflow, or a division by zero on `int64_t`, depending on what
the guard was protecting. This has happened twice in this codebase already:
`OperatorDAG::add_node`'s SSA and bounds checks, and `Rational`'s overflow and
division-by-zero guards. Both were assert-only, both passed the full Debug
suite, and both were undetected until someone configured Release by hand.
Running both configurations on every change is how this class of bug gets
caught before it reaches someone who isn't running a debugger.

The ASan leg exists for a third category that neither of the other two can
see: undefined behavior whose symptom happens to be *correct output* on this
target. Phase 4's `extrapolate` kernel was planned as `history[i - stride]`
with unsigned `i` and `i < stride`, which forms `history + SIZE_MAX` —
pointer overflow. The address wrapped back to the right place, so the tests
printed `PASSED` in Debug and in Release. UBSan is the only configuration in
which that code fails. A test suite that is green in Debug and Release has
not established the absence of UB; it has established that the UB was lucky.

## Commands

A `CMakePresets.json` (upstream/soma/CMakePresets.json) at the repo root defines both
configurations. From a fresh clone:

```bash
# Debug — configures build/, asserts active
cmake --preset debug
cmake --build --preset debug -j
ctest --preset debug

# Release — configures build-release/, -O3 -DNDEBUG, asserts compiled out
cmake --preset release
cmake --build --preset release -j
ctest --preset release

# ASan+UBSan — configures build-asan/, diagnostics are fatal
cmake --preset asan
cmake --build --preset asan -j
ctest --preset asan
```

A fourth, optional preset covers the LowFive HDF5-VOL transport. It is not one
of the three that gate a change — it needs `/opt/lowfive-shared` and
`/opt/hdf5-shared` — but it must pass when that transport is touched:

```bash
# LowFive HDF5-VOL — configures build-lowfive/, SOMA_ENABLE_LOWFIVE=ON
cmake --preset lowfive
cmake --build --preset lowfive -j
ctest --preset lowfive
```

All three must report `100% tests passed` before a change is done. `ctest
--preset release` is the only thing in this repository's test story that
exercises the `-DNDEBUG` code path — there is no separate CI service config (this repo has
none) — so a change that only runs the Debug preset has not actually been
verified against Release.

Equivalent commands without presets (e.g. on a CMake older than 3.21, which
is what `CMakePresets.json` requires):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure

cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

## Verifying this catches something

To confirm the Release leg actually exercises a code path Debug does not,
check out the revision immediately before the commit titled `fix(soma):
enforce Rational invariants in release and add a release test leg` (i.e.
`git checkout <that commit>~1 -- include/soma/base/rational.hpp`, the version
where `gcd_rational`'s only overflow guard is
`assert(detail::fits_i64(g) && ...)`), then run:

```bash
cmake --build --preset release -j
ctest --preset release -R Rational
```

Expected: `Rational.GcdRationalAssertsOnUnrepresentableGcd` (that revision's
name for the test; the fix commit renames it to
`GcdRationalThrowsOnUnrepresentableGcd`) fails — the guard's `assert()`
compiled out under `-DNDEBUG`, so the unrepresentable gcd silently narrows
and wraps instead of the test's expected death. `ctest --preset debug -R
Rational` passes on the same revision, because `assert()` is still active
there. That gap between the two presets' results is the failure mode this
second leg exists to catch.

## Optional dependencies

Every optional backend is `OFF` by default (see
`cmake/SomaOptions.cmake` (upstream/soma/cmake/SomaOptions.cmake)), so the two
configurations above resolve no external packages at all. Turning one on adds
a discovery step that reports two different things:

- `SOMA_ENABLE_<X>` — what you asked for.
- `SOMA_HAVE_<X>` — what was actually found.

These are not the same, and conflating them is how a silently degraded
dependency gets mistaken for a working feature. A MOAB install can be present,
findable, and the right version while still lacking the optional backend a
feature needs; see the record below for a concrete instance on this machine.
`soma_print_summary()` prints both blocks at the end of configure.

### MOAB

```bash
cmake -B build-moab -DCMAKE_BUILD_TYPE=Debug \
      -DSOMA_ENABLE_MOAB=ON \
      -DCMAKE_PREFIX_PATH=/opt/moab-tempest
```

`cmake/FindSomaMOAB.cmake` (upstream/soma/cmake/FindSomaMOAB.cmake) locates MOAB via
`pkg-config`, then reads capabilities out of the located install's
`moab/MOABConfig.h` rather than assuming them from the version number. It sets
`SOMA_HAVE_MOAB`, `SOMA_HAVE_TEMPEST`, `SOMA_HAVE_IMOAB` and defines the
imported target `soma::moab`.

Two details in that file are deliberate and easy to undo by accident:

- It uses plain `set()`, not `set(... PARENT_SCOPE)`. The file is `include()`d,
  and `include()` does not open a new scope. From the top-level
  `CMakeLists.txt` there is no parent scope, so `PARENT_SCOPE` would discard
  every `SOMA_HAVE_*` and the caller's "MOAB not found" `FATAL_ERROR` would
  fire even on a successful find.
- It clears `INTERFACE_COMPILE_OPTIONS` on `PkgConfig::MOAB`. `MOAB.pc` puts
  the flags MOAB itself was compiled with into `Cflags`, and on this machine
  that list contains `-O2 -DNDEBUG`. Left in place, linking `soma::moab` would
  apply `-DNDEBUG` to SOMA's own translation units in a **Debug** build and
  compile out every `assert()` — the exact failure this document opens by
  warning about. Consumers need MOAB's headers and libraries, not its build
  flags.

## MOAB rebuild record (Phase 5, Task 5.0)

High-order and monotone remap (Task 5.3) go through MOAB's `TempestRemapper`.
Conservative FV↔FV remap only needs `Intx2Mesh` and is unaffected by anything
here.

### Before: the MOAB already on this machine lacks TempestRemap

`/opt/petsc/download-moab` is PETSc's MOAB, and another project (KARMA) links
against it, so it must not be touched. It is MOAB 5.6.0 — the version SOMA
wants — but built without the TempestRemap backend:

```console
$ PKG_CONFIG_PATH=/opt/petsc/download-moab/lib/pkgconfig pkg-config --modversion MOAB
5.6.0
$ grep -E "define MOAB_HAVE_(MPI|TEMPEST|NETCDF|HDF5|EIGEN|ZOLTAN)" \
       /opt/petsc/download-moab/include/moab/MOABConfig.h
#define MOAB_HAVE_HDF5 1
#define MOAB_HAVE_HDF5_H 1
#define MOAB_HAVE_HDF5_PARALLEL 1
#define MOAB_HAVE_MPI 1
$ nm -gU /opt/petsc/download-moab/lib/libMOAB.dylib | grep -c -i tempest
0
$ ls /opt/petsc/download-moab/include/moab/Remapping/
ls: /opt/petsc/download-moab/include/moab/Remapping/: No such file or directory
```

MPI and HDF5 yes; no `MOAB_HAVE_TEMPESTREMAP`, no NetCDF (which TempestRemap
requires), zero Tempest symbols, no `Remapping/` headers. Version match is not
capability match — this is why `FindSomaMOAB.cmake` reads `MOABConfig.h`.

### After: a separate prefix at `/opt/moab-tempest`

Built into a **new** prefix; nothing under `/opt/petsc`, `/opt/hdf5`, or
`/opt/mpich` was modified.

Source: the 5.6.0 release tarball
(`https://web.cels.anl.gov/projects/sigma/downloads/moab/moab-5.6.0.tar.gz`),
not a git clone — the tarball ships a pre-generated `configure`, which avoids
running `autoreconf` with the host's autoconf 2.73 against MOAB 5.6.0's older
macros. Configured out-of-source:

```bash
/tmp/soma-moab/moab-5.6.0/configure \
  --prefix=/opt/moab-tempest \
  --with-mpi=/opt/mpich \
  --with-hdf5=/opt/hdf5 \
  --with-netcdf=/opt/netcdf \
  --with-pnetcdf=/opt/pnetcdf \
  --with-eigen3=/opt/eigen3 \
  --download-tempestremap \
  --enable-shared --enable-static \
  --enable-optimize --enable-debug \
  --enable-mbtempest --enable-mbcoupler \
  CC=/opt/mpich/bin/mpicc CXX=/opt/mpich/bin/mpicxx \
  FC=/opt/mpich/bin/mpifort F77=/opt/mpich/bin/mpif77
make -j8 && make install
```

`--download-tempestremap` (the primary route) worked; the
`--with-tempestremap=<prefix>` fallback was not needed. It fetches
TempestRemap `master` from E3SM-Project, builds it, and installs it into the
same prefix (`/opt/moab-tempest/lib/libTempestRemap.dylib`). NetCDF is a hard
prerequisite for it and was supplied from the existing `/opt/netcdf` 4.9.2.

### Verification

```console
$ grep -E "define MOAB_HAVE_(MPI|TEMPEST|NETCDF|HDF5|EIGEN|ZOLTAN|PNETCDF)" \
       /opt/moab-tempest/include/moab/MOABConfig.h
#define MOAB_HAVE_EIGEN3 1
#define MOAB_HAVE_HDF5 1
#define MOAB_HAVE_HDF5_H 1
#define MOAB_HAVE_HDF5_PARALLEL 1
#define MOAB_HAVE_MPI 1
#define MOAB_HAVE_NETCDF 1
#define MOAB_HAVE_NETCDFPAR 1
#define MOAB_HAVE_NETCDF_H 1
#define MOAB_HAVE_NETCDF_PAR_H 1
#define MOAB_HAVE_PNETCDF 1
#define MOAB_HAVE_PNETCDF_H 1
#define MOAB_HAVE_TEMPESTREMAP 1
$ nm -gU /opt/moab-tempest/lib/libMOAB.dylib | grep -c -i tempest
62
$ ls /opt/moab-tempest/include/moab/Remapping/
Remapper.hpp  TempestOnlineMap.hpp  TempestRemapper.hpp
$ ls /opt/moab-tempest/include/moab/iMOAB.h
/opt/moab-tempest/include/moab/iMOAB.h
```

The macro is spelled `MOAB_HAVE_TEMPESTREMAP`, not `MOAB_HAVE_TEMPEST`.

**Count symbols in the shared library, and only in the shared library.** A
`.a`-only MOAB install makes `nm -gU <prefix>/lib/libMOAB.dylib` fail, and `nm`
echoes the missing path in its error — a path containing the word "tempest"
then makes `grep -c -i tempest` report `1` for a library that does not exist.
That false positive was observed here against another prefix. Confirm
`libMOAB.dylib` exists first; `--enable-shared` is what produces it.

Symbol counts prove linkage, not behaviour, so MOAB's own remap regression
tests were run from the build tree:

```console
$ ./test_remapping    # 6 assertions, CS/RLL overlap + area conservation
All 6 tests passed.
$ ./imoab_remapping   # SE(order 4) -> FV weight generation through iMOAB
 passed baseline test 1
```

`imoab_remapping` exercises the high-order path Task 5.3 depends on, end to
end, and checks the result against a stored baseline.

One known non-failure: `mbtempest --type 5` on a `GenerateCSMesh`/
`GenerateRLLMesh` exodus pair aborts with `Can't convert overlap TempestRemap
mesh to MOAB format` at `TempestRemapper.cpp:1557`. The pre-existing,
independently built `/opt/moab/tempest` install fails identically on the same
two input files, so this is a property of that mesh pair or of the invocation,
not of this build.

### Reproducing the "before" check against any prefix

```bash
P=/path/to/moab/prefix
PKG_CONFIG_PATH=$P/lib/pkgconfig pkg-config --modversion MOAB
grep -E "define MOAB_HAVE_(MPI|TEMPESTREMAP|NETCDF|HDF5)" $P/include/moab/MOABConfig.h
test -f $P/lib/libMOAB.dylib && nm -gU $P/lib/libMOAB.dylib | grep -c -i tempest
ls $P/include/moab/Remapping/

## LowFive and the shared-HDF5 requirement (Phase 8, Task 8.0)

LowFive is an HDF5 **VOL plugin**. HDF5 brings it in with `dlopen()`, and it
resolves the `H5*` symbols it was linked against out of a *shared* `libhdf5`
at load time. A static HDF5 therefore cannot back it — not as a
configuration inconvenience, but structurally.

### The failure, recorded verbatim

`/opt/hdf5` on this machine is parallel HDF5 1.14.1-2 built static-only:

```
$ ls /opt/hdf5/lib/
libhdf5.a  libhdf5.la  libhdf5.settings  libhdf5_hl.a  libhdf5_hl.la

$ grep -E "HDF5 Version|Parallel HDF5" /opt/hdf5/lib/libhdf5.settings
                   HDF5 Version: 1.14.1-2
                     Parallel HDF5: yes
```

There are no `.dylib` files. The prebuilt `/opt/lowfive` was nevertheless
linked against shared HDF5, so it cannot load:

```
$ otool -L /opt/lowfive/lib/liblowfive.dylib | grep -i hdf5
	/opt/hdf5/lib/libhdf5.310.dylib (compatibility version 312.0.0, current version 312.0.0)
	/opt/hdf5/lib/libhdf5_hl.310.dylib (compatibility version 311.0.0, current version 311.1.0)

$ cc /tmp/lftest.c -o /tmp/lftest && /tmp/lftest
FAIL: dlopen(/opt/lowfive/lib/liblowfive.dylib, 0x0002): Library not loaded: /opt/hdf5/lib/libhdf5.310.dylib
  Referenced from: <04A09401-3630-3B4C-947D-6995192EEBDA> /opt/lowfive/lib/liblowfive.dylib
  Reason: tried: '/opt/hdf5/lib/libhdf5.310.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/opt/hdf5/lib/libhdf5.310.dylib' (no such file), '/opt/hdf5/lib/libhdf5.310.dylib' (no such file), '/usr/local/lib/libhdf5.310.dylib' (no such file), '/usr/lib/libhdf5.310.dylib' (no such file, not in dyld cache)
```

Nothing in the SOMA build could have worked around this. The fix is a second
HDF5 install.

### `/opt/hdf5` must not be rebuilt in place

MOAB (`/opt/petsc/download-moab`) and PETSc (`/opt/petsc`) link the static
`/opt/hdf5`. Replacing it would relink both. The shared build goes into a
**new prefix**, `/opt/hdf5-shared`, and `/opt/hdf5` is left byte-for-byte
alone.

### Building the shared HDF5

The URL in the original task brief
(`https://support.hdfgroup.org/ftp/HDF5/releases/...`) is dead — it returns a
404 HTML page, not a tarball. Use the GitHub release asset:

```bash
curl -sSL -o /tmp/hdf5.tar.gz \
  https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_14_1-2/hdf5-1_14_1-2.tar.gz
tar xf /tmp/hdf5.tar.gz -C /tmp     # unpacks to /tmp/hdfsrc
```

Two things about that tarball are worth knowing before you copy this recipe:

1. **It has no generated `configure`.** The brief's `./configure --enable-shared
   ...` line cannot run against it. Bootstrapping with `autogen.sh` would need
   the exact autoconf/automake the 2023 release used (this machine has
   autoconf 2.73 / automake 1.18.1, both newer). The CMake build is HDF5's
   other first-class build system and is used here instead.
2. **Despite the file name, it contains 1.14.2-1 sources**, not 1.14.1-2
   (`H5_VERS_RELEASE 2`, `H5_VERS_SUBRELEASE "1"`). The git *tag* archive
   `archive/refs/tags/hdf5-1_14_1-2.tar.gz` does contain a true 1.14.1-2. This
   does not matter here: both carry `LT_VERS_INTERFACE = 311` and
   `LT_VERS_AGE = 1`, so both produce soname **310** (`libhdf5.310.dylib`) —
   the same ABI LowFive expects — and LowFive is rebuilt from source against
   it anyway.

```bash
cmake -B /tmp/hdf5-build -S /tmp/hdfsrc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/hdf5-shared \
  -DCMAKE_C_COMPILER=/opt/mpich/bin/mpicc \
  -DONLY_SHARED_LIBS=ON -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
  -DHDF5_ENABLE_PARALLEL=ON -DHDF5_BUILD_HL_LIB=ON -DHDF5_BUILD_TOOLS=ON \
  -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_JAVA=OFF -DBUILD_TESTING=OFF
cmake --build /tmp/hdf5-build -j8 && cmake --install /tmp/hdf5-build
```

Result — real shared libraries, parallel, no static archive:

```
$ grep -E "HDF5 Version|Parallel HDF5|Shared C Library|Static C Library" \
      /opt/hdf5-shared/lib/libhdf5.settings
                   HDF5 Version: 1.14.2.1
               Shared C Library: YES
               Static C Library: NO
                     Parallel HDF5: ON

$ file /opt/hdf5-shared/lib/libhdf5.310.1.0.dylib
... Mach-O 64-bit dynamically linked shared library arm64
```

Verified working with a collective parallel write on 3 ranks:

```
$ mpirun -n 3 ./ph5
rank0: wrote ph5.h5 with HDF5 1.14.2 on 3 ranks
```

### Rebuilding LowFive against it

`/opt/lowfive` cannot be reused: its `libhdf5` path is baked in as
`/opt/hdf5/lib/libhdf5.310.dylib`. Rebuild into `/opt/lowfive-shared`.

```bash
git clone --recursive https://github.com/diatomic/LowFive.git /tmp/lowfive-src
cmake -B /tmp/lowfive-build -S /tmp/lowfive-src -G Ninja \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/lowfive-shared \
  -DCMAKE_PREFIX_PATH=/opt/hdf5-shared \
  -DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON \
  -DCMAKE_C_COMPILER=/opt/mpich/bin/mpicc \
  -DCMAKE_CXX_COMPILER=/opt/mpich/bin/mpicxx \
  -Dlowfive_python=OFF -Dbuild_examples=OFF -DFMT_INSTALL=ON \
  -DCMAKE_INSTALL_RPATH=/opt/hdf5-shared/lib \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
cmake --build /tmp/lowfive-build -j8 && cmake --install /tmp/lowfive-build
```

Four of those flags are non-obvious and each cost a build iteration:

- **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`** — the vendored `ext/fmt` declares
  `cmake_minimum_required(VERSION 3.1...3.18)`. CMake 4.x rejects anything
  below 3.5 outright.
- **Do *not* pass `-DBUILD_SHARED_LIBS=ON`** (the brief suggests it). Upstream
  `src/CMakeLists.txt` links `lowfive-dist` against only `fmt::fmt` and `diy`,
  never HDF5. As a static archive that is fine; as a dylib it fails to link
  with ~50 undefined `_H5*` symbols. `liblowfive` itself is declared
  `add_library(lowfive SHARED ...)` unconditionally, so the plugin is shared
  either way. `BUILD_SHARED_LIBS=ON` would also make `fmt`/`spdlog` dylibs
  that `FMT_INSTALL=OFF` never installs, leaving the installed plugin with an
  unresolvable `@rpath/libfmt.8.dylib`.
- **`-DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON`** — `ext/diy/CMakeLists.txt` calls a
  bare `find_package(fmt)` *before* the top level adds the vendored
  `ext/fmt`. On a machine with conda it silently resolves
  `~/miniforge3/lib/cmake/fmt`, and the resulting plugin links **two different
  fmt ABIs at once** (`libfmt.8.dylib` from the submodule and
  `libfmt.12.dylib` from conda) — an ODR violation waiting to happen.
  Disabling the find leaves diy on the vendored fmt, which is what the top
  level's `-DDIY_SYSTEM_FMT` already intends.
- **`-DFMT_INSTALL=ON`** — LowFive's installed public header
  `lowfive/vol-metadata.hpp` does `#include <fmt/core.h>`, but LowFive does not
  install fmt's headers. Without this, *any* consumer of the install prefix
  fails to compile on the first include.

### Proof it now works

`dlopen` succeeds:

```
$ /tmp/lftest2
OK

$ otool -L /opt/lowfive-shared/lib/liblowfive.dylib | grep -i -E "hdf5|fmt"
	@rpath/libhdf5_hl.310.dylib (compatibility version 310.0.0, current version 310.0.1)
	@rpath/libhdf5.310.dylib (compatibility version 310.0.0, current version 310.1.0)

$ otool -l /opt/lowfive-shared/lib/liblowfive.dylib | grep -A2 LC_RPATH | grep path
         path /opt/hdf5-shared/lib (offset 12)
```

HDF5 loads it as a VOL connector. Baseline versus
`HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH`, same binary:

```
$ ./volcheck2
H5VLis_connector_registered_by_name("lowfive") = 0
H5VLis_connector_registered_by_value(510)        = 0
fapl VOL id = 648518346341351424

$ HDF5_PLUGIN_PATH=/opt/lowfive-shared/lib HDF5_VOL_CONNECTOR=lowfive ./volcheck2
H5VLis_connector_registered_by_name("lowfive") = 1
H5VLis_connector_registered_by_value(510)        = 1
fapl VOL id = 648518346341351425
```

And interception actually happens — a program that calls `H5Fcreate` /
`H5Dwrite` / `H5Dread` normally, with `MetadataVOL` set on the FAPL, round
trips its data with **no file created on disk**:

```
$ ./lfprog
read back through LowFive in-memory VOL: 7 8 9 10
file on disk? NO (intercepted, memory only)
clean exit of scope
```

That last check is the one that matters for Demo B, whose whole claim is that
two unmodified executables are coupled by intercepting their HDF5 I/O.

### Known issue: abort at exit

> **Amended in Task 8.4.** This was recorded as specific to the
> `HDF5_VOL_CONNECTOR` plugin-loading path. It is not. The **programmatic**
> path reproduces it too, whenever a native-VOL HDF5 call happens *before* the
> connector is registered. See
> [Abort at exit, revisited](#abort-at-exit-revisited-task-84) below for the
> reproducer and the fix that is now in the library.

When LowFive is loaded via `HDF5_VOL_CONNECTOR`, the process aborts *after*
`main` returns:

```
libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument
exit=134
```

All program output is produced first; the abort is in teardown. It is **not**
a `dlclose` problem — pinning the plugin with `RTLD_NODELETE` does not help.
HDF5's `atexit` handler terminates the connector after LowFive's C++ statics
(the spdlog logger and its mutex) have already been destroyed.

Two things avoid it:

- Call `H5close()` explicitly before returning from `main`, while the statics
  are still alive. Verified: `H5close() = 0`, `exit=0`.
- Use the programmatic path instead of the environment variable — construct
  `LowFive::MetadataVOL::create_MetadataVOL()` and apply it with
  `LowFive::H5VOLProperty`. `H5VOLProperty`'s destructor unregisters the
  connector at a well-defined point. The `./lfprog` run above exits 0.

Demo B should use the programmatic path.

### Using it from SOMA

`SOMA_ENABLE_LOWFIVE` is **OFF by default**, so the dependency-free core build
never looks for any of this. When it is ON,
`cmake/FindSomaLowFive.cmake` (upstream/soma/cmake/FindSomaLowFive.cmake) locates the
library and headers and then, rather than trusting that a file named
`liblowfive.dylib` is usable, actually `dlopen`s it at configure time. That is
the only check that catches the static-HDF5 case, since the extension of
LowFive's own library says nothing about how its HDF5 was built.

```bash
cmake -S . -B build -DSOMA_ENABLE_LOWFIVE=ON \
      -DCMAKE_PREFIX_PATH="/opt/lowfive-shared;/opt/hdf5-shared"
```

A missing or unusable LowFive is a warning, not an error: configuration
continues with `SOMA_HAVE_LOWFIVE OFF` and only Demo B is unavailable. The
configure summary reports the request and the probe result separately:

```
--   SOMA_ENABLE_LOWFIVE: ON
--   SOMA_HAVE_LOWFIVE: ON
```

Pointed at the old broken install, the same probe reproduces the failure at
configure time instead of at run time:

```
$ cmake -S . -B build -DSOMA_ENABLE_LOWFIVE=ON -DCMAKE_PREFIX_PATH=/opt/lowfive
CMake Warning at cmake/FindSomaLowFive.cmake:59 (message):
  LowFive: /opt/lowfive/lib/liblowfive.dylib exists but cannot be dlopened,
  so it cannot serve as an HDF5 VOL plugin.  This is usually a static or
  missing libhdf5.  See docs/BUILDING.md.

  dlopen(/opt/lowfive/lib/liblowfive.dylib, 0x0002): Library not loaded:
  /opt/mpich/lib/libmpicxx.12.dylib
--   SOMA_HAVE_LOWFIVE: OFF
```

(The reported missing library has drifted since Task 8.0 recorded it — the
static `/opt/lowfive` now fails on `libmpicxx.12.dylib` before it ever gets as
far as `libhdf5`. The verdict is unchanged: that prefix is not usable, and the
probe still catches it at configure time.)

---

## LowFive HDF5-VOL transport (Phase 8, Task 8.4)

### The build recipe

A `lowfive` preset is defined alongside `debug`/`release`/`asan`:

```bash
cmake --preset lowfive          # -> build-lowfive/
cmake --build --preset lowfive -j 6
ctest --preset lowfive -j 6
```

It is exactly

```bash
cmake -B build-lowfive -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_LOWFIVE=ON \
      -DCMAKE_PREFIX_PATH="/opt/lowfive-shared;/opt/hdf5-shared"
```

`/opt/lowfive-shared` and `/opt/hdf5-shared`, **not** `/opt/lowfive` and
`/opt/hdf5` — see the two sections above for why the static pair cannot work.

The sanitizer combination is also clean and is worth running when touching
this code, because HDF5 object ids are easy to leak and easy to double-close:

```bash
cmake -B build-lowfive-asan -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_LOWFIVE=ON \
      -DCMAKE_PREFIX_PATH="/opt/lowfive-shared;/opt/hdf5-shared" \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-lowfive-asan -j 6 && ctest --test-dir build-lowfive-asan -j 6
```

| configuration | tests |
|---|---|
| `debug`, `release`, `asan` (LowFive OFF) | 701 / 701 |
| `lowfive` | 720 / 720 |
| `lowfive` + ASan/UBSan | 720 / 720 |

The default legs were 696 before this task; the five extra are the
LowFive-off half of `test/unit/lowfive_transport_test.cpp`, which checks that
every entry point still links and reports `E-TRANSPORT-NOLOWFIVE`. The
LowFive-on legs replace those five with the twenty-four real ones.

### `<hdf5.h>` needs `<mpi.h>` even with no MPI in the build

`/opt/hdf5-shared` is **parallel** HDF5, and a parallel `H5public.h` does an
unconditional `#include <mpi.h>`. Every consumer of `<hdf5.h>` therefore needs
MPI's include directory, whether or not it makes an MPI call, links `libmpi`,
or has `SOMA_ENABLE_MPI=ON`. The first compile of the transport failed with
nothing but

```
/opt/hdf5-shared/include/H5public.h:65:10: fatal error: 'mpi.h' file not found
```

`FindHDF5` does not add it, so `cmake/FindSomaLowFive.cmake` does: when
`HDF5_IS_PARALLEL`, it runs `find_package(MPI COMPONENTS C)` and puts
`MPI_C_INCLUDE_DIRS` on the `soma::lowfive` interface. If no MPI is found the
module turns `SOMA_HAVE_LOWFIVE` back OFF with an explanation rather than
letting the build fail later with a confusing error.

The module also now *compiles* a probe TU that includes `<hdf5.h>` and
LowFive's headers, in addition to the pre-existing `dlopen` probe. The two
catch different things: `dlopen` proves the plugin loads, `try_compile` proves
a translation unit that uses it can be built.

### Abort at exit, revisited (Task 8.4)

The teardown abort documented above under the environment-variable path is
**not confined to it**. On the programmatic path it appears whenever a
native-VOL HDF5 call is made *before* the connector is registered. Reduced to
~30 lines: a plain `H5Fcreate`/`H5Dwrite`/`H5Fclose` with `H5P_DEFAULT`, then
`create_MetadataVOL()` + `H5VOLProperty` + one intercepted write.

```
$ ./probe6            # no fix
libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument
native leg done
vol leg done, returning from main
exit=134

$ ./probe6_fixed      # std::atexit(H5close) registered with the connector
native leg done
vol leg done, returning from main
atexit: H5close()=0
exit=0
```

In the test suite it showed up as exactly one `ctest` failure —
`LowFiveInterception.SamePathIsWrittenByPlainHdf5AndNotByTheTransport`,
reported as `Subprocess aborted` — that *passed* when the whole binary was run
in one process. That is the shape of bug that hides for a long time.

`src/transport/lowfive_transport.cpp` now registers `std::atexit(H5close)` at
the moment it registers the connector. Because `atexit` is LIFO and HDF5's own
handler is registered at `H5open` (earlier), ours runs first, terminating the
library while LowFive's spdlog statics are still alive — which is the remedy
this document already recommended, applied automatically.

### Upstream LowFive bug: `drop()` on an absent file name

`LowFive::MetadataVOL::drop(std::string)` does not test `files.find(name)`
against `files.end()`. Called with a name that is not in the in-memory map it
dereferences the end iterator, and the process dies:

```
stop reason = EXC_BAD_ACCESS (code=1, address=0x41)
frame #0: liblowfive.dylib`std::__1::__tree_is_left_child<...>
          at __tree:105:33
```

A two-call program reproduces it: `create_MetadataVOL()`, then
`drop("never-existed.h5")` — `exit=139`. It bites this transport because the
in-memory file is dropped after every `move()` and again in the destructor.
`MetadataVOL::files` is a public member, so the transport does the membership
test the library should be doing. **Not reported upstream**; the workaround is
local and commented at the call site.

### Using the transport

`SOMA_ENABLE_LOWFIVE` selects nothing about *whether the class exists* —
`soma::LowFiveTransport` is declared in `soma/transport/transport.hpp` and
compiled in every configuration, and `TransportRegistry::create("lowfive")`
succeeds everywhere. Without LowFive its setup calls return
`E-TRANSPORT-NOLOWFIVE` and name the option to turn on.

```cpp
soma::LowFiveTransport t;
t.set_location("/scratch/run/temperature.h5", "temperature");  // once
t.build_plan(spec);                                            // once
// hot loop:
t.prepare(spec);
t.move(src, dst, n);
```

`t.access_plist_handle()` returns the intercepting FAPL as a `long long` (it
is an HDF5 `hid_t`; it crosses as an integer so `hdf5.h` stays off consumers'
include paths, the same way `MpiTransport::set_communicator` takes
`MPI_Comm_c2f(comm)`). Handing it to an unmodified HDF5 program as the
`fapl_id` of its `H5Fcreate`/`H5Fopen` is what makes Demo B's "two unmodified
executables" claim work.

---

## Python bindings (Phase 8, Task 8.3)

`SOMA_ENABLE_PYTHON` is **OFF by default**, so the dependency-free core build
never looks for Python, pybind11 or NumPy, and the three presets above are
unaffected by anything in this section.

```bash
cmake -B build-py -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_PYTHON=ON
cmake --build build-py -j8
ctest --test-dir build-py --output-on-failure -R python_bindings_test
```

The module lands at `build-py/python/_soma.<abi>.so`; the pure-Python wrapper
that re-exports it lives in `python/soma/`. `ctest` puts both on `PYTHONPATH`
itself, so nothing has to be installed to run the suite.

### What is required, and what is only found

| | |
|---|---|
| Python | ≥ `SOMA_PYTHON_VERSION` (default 3.9), components `Interpreter` and `Development.Module` |
| NumPy | required, checked at configure time by importing it with the selected interpreter |
| pybind11 | `find_package` first, then a pinned `v2.13.6` FetchContent fallback |

`Development.Module`, not `Development`: an extension module does not embed the
interpreter and must not link `libpython` on macOS, where the loader supplies
those symbols from the host process.

The interpreter is **not** pinned to an exact minor version. The task brief
called for `find_package(Python 3.11 EXACT ...)` "to match the installed
LowFive extension"; on this machine that extension exists but is not usable.
`/opt/lowfive/lib/python3.11/site-packages/lowfive/_lowfive.cpython-311-darwin.so`
belongs to the static-HDF5 `/opt/lowfive` install this document records as
unusable as a VOL plugin, and it does not import even with that directory on
`PYTHONPATH`:

```
ImportError: dlopen(.../lowfive/_lowfive.cpython-311-darwin.so, 0x0002):
  Library not loaded: /opt/mpich/lib/libmpicxx.12.dylib
```

The install that *is* usable, `/opt/lowfive-shared`, was configured with
`-Dlowfive_python=OFF` and ships no Python module. `SOMA_PYTHON_VERSION` and
`SOMA_PYTHON_EXACT` restore the pin if Demo B ever gets an importable LowFive
extension built for a specific 3.x:

```bash
cmake -B build-py -DSOMA_ENABLE_PYTHON=ON \
      -DSOMA_PYTHON_VERSION=3.11 -DSOMA_PYTHON_EXACT=ON
```

`-DPython_EXECUTABLE=/path/to/python` selects a specific interpreter as usual;
`PYBIND11_FINDPYTHON` is forced ON so pybind11 uses that same one rather than
running its own search and possibly building against different headers.

### Position-independent code

Enabling the bindings sets `CMAKE_POSITION_INDEPENDENT_CODE ON` for the whole
project. Every static library that ends up inside a shared module has to be
PIC; on ELF targets a non-PIC `libsoma_core.a` fails to link into `_soma.so`
with `relocation R_X86_64_32S ... can not be used when making a shared object`.
It is scoped to `if(SOMA_ENABLE_PYTHON)` so the default build's objects are
unchanged.

### Running the Python tests under the sanitizer

The asan preset instruments `_soma.so`, but the interpreter that `dlopen()`s it
is not instrumented, and the ASan runtime has to be the first thing in the
process or its interceptors are never installed — the import then aborts with
*"Interceptors are not working. This may be because AddressSanitizer is loaded
too late"*. `python/CMakeLists.txt` therefore locates the runtime with
`${CMAKE_CXX_COMPILER} -print-file-name=` and preloads it into the test's
environment (`DYLD_INSERT_LIBRARIES` on macOS, `LD_PRELOAD` elsewhere), with
`ASAN_OPTIONS=detect_leaks=0` because CPython, NumPy and the module's own
module-level exception classes all hold deliberately-unfreed state for the life
of the process. Use-after-free and buffer-overflow detection — which is what
the NumPy binding contract needs — is unaffected. If the runtime cannot be
found, configuration emits a warning rather than silently registering a test
that will fail at import.

```bash
cmake -B build-asan-py -DCMAKE_BUILD_TYPE=Debug -DSOMA_ENABLE_PYTHON=ON \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan-py -j8 && ctest --test-dir build-asan-py --output-on-failure
```

### The NumPy buffer contract

SOMA never owns field storage (`soma_c.h`, "FIELD STORAGE"). An array bound
through `FieldRegistry.bind_numpy` or `Context.register_field` must be a
`numpy.ndarray`, dtype exactly `float64` in native byte order, C-contiguous,
writeable, and either 1-D `(count,)` or 2-D `(count, components)` — the 2-D
case is the interleaved AoS layout, `stride = components`,
`component_stride = 1`. A strided slice, a transpose, a wrong dtype, a
read-only array and a Python list are each refused with `soma.InvalidArgument`
(which is a `ValueError`), because every one of them would otherwise be a
silent copy or a read of the wrong doubles. The binding holds a strong
reference to the array for as long as the registry or context lives, which is
what makes a temporary expression safe and what makes NumPy refuse a later
in-place `ndarray.resize()` — a resize reallocates, and the pointer SOMA holds
would dangle.

### The shape-query gap, and why no new C entry point was added

The C ABI has no `soma_field_components()`, so a caller that holds only a field
handle cannot ask how many components the graph declares for it; the Fortran
shim's order test has to discover a mismatch by attempting the registration and
matching English text in the error message (`test/fortran/capi_test.f90`,
"registered with 3").

The Python bindings do **not** need that, and no function was added to the
frozen header. Every path that puts a graph into a `soma.Context` also leaves a
`soma::CouplingGraph` on the Python side — `load_graph()` takes either a
`CouplingGraph` or a descriptor it deserializes, and `load_yaml()` re-reads the
file with `soma::load_yaml_file` after the ABI has accepted it — so
`Context.field_components(handle)` answers from the graph, and
`register_field` compares the declared count with the array's shape and reports
both numbers *before* calling the ABI. Nothing in the Python front-end parses
an error string.

Adding `soma_field_components()` would still be worth doing for Fortran's sake.
It is a compatible addition (a new function with a new name), but the header's
own rule is that `SOMA_ABI_VERSION` is bumped when one is made, which touches
the Fortran module's version constant and every test that asserts the ABI
version. That belongs in a task that can update the Fortran shim to stop
parsing English at the same time, not in the Python bindings.
