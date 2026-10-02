# LowFive: coupling through intercepted HDF5

Spec §6 mode 4: two components exchange a field by one writing it with ordinary
`H5Fcreate`/`H5Dwrite` calls and the other reading it back with ordinary
`H5Fopen`/`H5Dread` calls — no FUSE-specific call on either side. LowFive, Argonne's
HDF5 Virtual Object Layer (VOL) connector, intercepts those calls in memory; no file
ever reaches disk. `fuse::LowFiveTransport` (`fuse/transport/transport.hpp`,
`src/transport/lowfive_transport.cpp`) is the seam.

This page has two parts: the single-process interception path, which is implemented
and measured end to end, and the cross-process (two-executable, "Demo B") path, which
compiles against LowFive's documented API but has never been run — MPI cannot start in
the sandbox this was written in. The difference in confidence between the two is the
most important thing on this page; it is restated at the start of each section.

## The two-HDF5-image hazard — closed, with a measured acceptance test

**Status: understood, reproduced, and configure-time refused. Fixing it on any given
machine is a host action (rebuilding LowFive) this checkout cannot perform.**

A binary that links both MOAB and LowFive can end up with **two independent copies**
of `libhdf5` mapped into one process — one through MOAB's own linkage, one through
LowFive's — if the two were built against different HDF5 installations. Neither the
linker nor the loader reports this as an error: both images load successfully, and
each has its own copy of HDF5's process-wide state (the property-list id table, the
VOL connector registry). A property-list id minted by one image means nothing to the
other. Concretely, LowFive's own interception machinery
(`fuse::LowFiveTransport::build_plan()`, mirrored in `tools/probe-hdf5-images.c`) calls
`H5Pcreate` to build a file-access property list and then `H5Pset_vol` to attach the
connector to it — and if those two calls resolve against different `libhdf5` images,
`H5Pset_vol` fails with HDF5's own `"not a file VOL ID"` diagnostic, measured directly
on this host (below), while the overall process still exits 0. **LowFive's
interception would silently not apply to MOAB's own HDF5 traffic** (MOAB's native
`.h5m` mesh I/O, for instance) in that configuration — it would simply degrade to
ordinary disk I/O, with no link error and no runtime error.

### Measured, 2026-10-02, on this host

Two LowFive installations exist in this lab's `/opt`:

| Prefix | HDF5 it links against | Relationship to MOAB's own HDF5 (`/opt/hdf5`) |
|---|---|---|
| `/opt/lowfive` | `/opt/hdf5/lib/libhdf5.310.dylib` (**absolute path**) | **Same path as MOAB's own linkage.** |
| `/opt/lowfive-shared` | `@rpath/libhdf5.310.dylib`, `LC_RPATH=/opt/hdf5-shared/lib` | **A different prefix** (`/opt/hdf5-shared`, a separately built "shared HDF5"). |

`tools/probe-hdf5-images.sh` links `tools/probe-hdf5-images.c` (one real call through
each library: `moab::Core`, then `LowFive::H5VOLProperty::apply()`) against MOAB
(`/opt/moab/cmake-all`) and each LowFive prefix in turn, runs the result under
`DYLD_PRINT_LIBRARIES=1`, and counts the distinct `libhdf5`/`libhdf5_hl` images that
actually load:

```
$ tools/probe-hdf5-images.sh /opt/lowfive
probe-hdf5-images.sh: libhdf5 images loaded (1):
  /opt/hdf5/lib/libhdf5.310.dylib
probe-hdf5-images.sh: libhdf5_hl images loaded (1):
  /opt/hdf5/lib/libhdf5_hl.310.dylib
probe-hdf5-images.sh: PASS -- exactly one libhdf5 and one libhdf5_hl image

$ tools/probe-hdf5-images.sh /opt/lowfive-shared
probe-hdf5-images.sh: libhdf5 images loaded (2):
  /opt/hdf5-shared/lib/libhdf5.310.1.0.dylib
  /opt/hdf5/lib/libhdf5.310.dylib
probe-hdf5-images.sh: libhdf5_hl images loaded (2):
  /opt/hdf5-shared/lib/libhdf5_hl.310.0.1.dylib
  /opt/hdf5/lib/libhdf5_hl.310.dylib
probe-hdf5-images.sh: FAIL -- the two-HDF5-image hazard is present
```

With the FAILing pairing, the probe's own `H5Pcreate`/`H5Pset_vol` sequence
(`tools/probe-hdf5-images.c`, the same sequence `LowFiveTransport::build_plan()`
runs) reproduces the silent-corruption symptom directly:

```
HDF5-DIAG: Error detected in HDF5 (1.14.2) thread 0:
  #000: ../../src/H5Pfapl.c line 6120 in H5Pset_vol(): not a file VOL ID
HDF5-DIAG: Error detected in HDF5 (1.14.2) thread 0:
  #000: ../../src/H5VL.c line 503 in H5VLunregister_connector(): not a VOL connector ID
probe: vol_id=648518346341351425 fapl_vol_matches=NO
```

— and the process still exits non-zero only because the probe itself checks for the
mismatch and reports it; nothing in HDF5 or the OS would have surfaced it otherwise.

**`/opt/lowfive` (the pairing that measures as safe) could not be run to full
completion by this probe either, but for an UNRELATED reason**: its own
`liblowfive.dylib` links against `/opt/mpich/lib/libmpicxx.12.dylib`, an MPICH ABI
version older than the one currently installed at `/opt/mpich`
(`libmpicxx.0.dylib`, MPICH 5.0 per `mpichversion`). The process aborts at load time
with `Library not loaded: .../libmpicxx.12.dylib` — but only AFTER the loader has
already mapped both `libhdf5`/`libhdf5_hl` (confirmed by inspecting the trace up to
that point), so the image-count verdict is unaffected. This is a second, independent
defect in this lab's `/opt/lowfive` install (a stale MPI linkage), not the hazard this
page is about; `tools/probe-hdf5-images.sh` reports it as a separate warning rather
than folding it into the PASS/FAIL verdict.

### Why a dynamic probe, not a static CMake check

An earlier draft of this fix considered comparing `MOAB_LIBRARIES`' HDF5 path against
`LOWFIVE_INCLUDE_DIR`'s sibling `lib/` directory as a configure-time string
comparison. That would have been wrong on this very host: `find_package(HDF5)`
resolved to `/opt/hdf5-shared` even when `-DCMAKE_PREFIX_PATH` named only `/opt/hdf5`,
because CMake's bundled `FindHDF5.cmake` module (CMake 4.3's
`share/cmake/Modules/FindHDF5.cmake`, line 678) tries a CONFIG-mode sub-search
(`find_package(HDF5 QUIET NO_MODULE ...)`) first, and CMake's CONFIG-mode search
globs `<prefix>/<name>*/` for every entry of `CMAKE_SYSTEM_PREFIX_PATH` — which
includes a bare `/opt` by default on this platform. `/opt/hdf5-shared` matches the
glob `hdf5*` under `/opt`, so it is a silent CONFIG-mode candidate **regardless of
what `CMAKE_PREFIX_PATH` was given**. (Forcing the legacy module path with
`-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON` avoids the glob and resolves to the intended
prefix; see the acceptance-test commands below.) A string comparison of *intended*
paths would have missed this entirely. The probe tests what actually got linked,
which is the only thing that can answer the question correctly.

### The fix, and why it is a host action

Rebuild LowFive against the **exact** HDF5 prefix MOAB uses, not a separately built
"shared HDF5" prefix — even though that separate prefix is, in isolation, a perfectly
good shared/VOL-pluggable HDF5. On this host, MOAB's own HDF5 (`/opt/hdf5`) already
*is* a shared build (`libhdf5.310.dylib` exists, confirmed by `otool -L
/opt/moab/cmake-all/lib/libMOAB.dylib`), so the existing `/opt/lowfive` install is
already built correctly against it — it is only its stale MPI linkage that needs a
rebuild:

```bash
git clone --recursive https://github.com/diatomic/LowFive.git /tmp/lowfive-src
cmake -B /tmp/lowfive-build -S /tmp/lowfive-src \
      -DCMAKE_INSTALL_PREFIX=/opt/lowfive-hdf5-matched \
      -DCMAKE_PREFIX_PATH="/opt/hdf5;/opt/diy" \
      -DCMAKE_CXX_COMPILER=/opt/mpich/bin/mpicxx \
      -DBUILD_SHARED_LIBS=ON -Dbuild_examples=OFF
cmake --build /tmp/lowfive-build -j8 && cmake --install /tmp/lowfive-build
```

`/opt` is not writable from the sandbox this task ran in, so this step, and verifying
it, are host actions. **Acceptance test**, in order:

```bash
tools/probe-hdf5-images.sh /opt/lowfive-hdf5-matched   # expect PASS, no warning this time
cmake -S . -B build-lf -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON -DFUSE_ENABLE_LOWFIVE=ON \
      -DCMAKE_PREFIX_PATH="...;/opt/moab/cmake-all;/opt/lowfive-hdf5-matched;/opt/hdf5" \
      -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON -DHDF5_ROOT=/opt/hdf5
```

`cmake/FuseDependencies.cmake` runs the probe itself as part of this configure
whenever `FUSE_ENABLE_LOWFIVE AND FUSE_ENABLE_MOAB`, using the SAME MOAB/HDF5/LowFive
this configure actually resolved (not a guess), and **refuses the configuration with
`FATAL_ERROR`** — not a warning — if it measures more than one image. Both outcomes
were exercised end to end against this checkout's real `cmake/FuseDependencies.cmake`
on 2026-10-02: `-DCMAKE_PREFIX_PATH=...;/opt/lowfive-shared` refuses with the message
above; `-DCMAKE_PREFIX_PATH=...;/opt/lowfive;/opt/hdf5` (plus
`-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON -DHDF5_ROOT=/opt/hdf5`, to route around the
glob surprise above) configures clean. `FUSE_ENABLE_LOWFIVE=ON` with
`FUSE_ENABLE_MOAB=OFF` skips the probe: with no MOAB linked there is nothing for
LowFive's HDF5 to collide with.

## Single-process interception — implemented, measured

`LowFiveTransport::set_location()` then `build_plan()` register LowFive's VOL
connector exactly once per process (a second live registration would unregister the
first out from under it — see the comment on `VolSingleton` in
`lowfive_transport.cpp`), build a file-access property list carrying it, and PROVE
interception is live before trusting it: a probe dataset is written and read back
through the connector, LowFive's own `after_dataset_write` callback must have fired,
and `stat()` on the path must fail (no real file). `move()` re-proves both on every
call, not just at setup, so a connector that stops intercepting mid-run fails loudly
(`E-TRANSPORT-LF-NOINTERCEPT`/`E-TRANSPORT-LF-ONDISK`) instead of quietly writing real
files. This path needs only `FUSE_ENABLE_LOWFIVE`, not MPI, and is exercised directly
in `test/transport/lowfive_test.cpp`.

## Distributed (cross-process) interception — Demo B

**Status: compiles against LowFive's public header contract
(`lowfive/vol-dist-metadata.hpp`); never executed. Confidence here is categorically
lower than the single-process path above** — there is no measurement to point to, only
a type-checked build. Treat every claim in this section as a design, not a result,
until a host with working MPI records an actual run.

Demo B's whole premise — two **unmodified** HDF5-writing/reading executables coupled
purely by interception — needs the data to cross an **actual OS process boundary**,
which the single-process path above cannot do: its in-memory file map
(`LowFive::MetadataVOL`) lives in one process's address space. Crossing processes
needs LowFive's `DistMetadataVOL`, which carries an MPI **intercommunicator** between
the producer's process group and the consumer's — joined the same way `MpiTransport`
joins any `MPI_Comm` to this codebase, crossing the boundary as a
`MPI_Comm_c2f()`/`MPI_Comm_f2c()` handle so this header never names `mpi.h`:

- `LowFiveTransport::set_intercomm(local_comm, intercomm)` — before
  `set_location()`, switches the process onto `DistMetadataVOL` instead of
  `MetadataVOL`. The connector choice is a process-wide, one-time decision, same as
  the single-process path's singleton.
- `serve_as_producer()` — called by the producer's driver after its own, unmodified
  `H5Fclose`; pushes the written data across the intercommunicator
  (`DistMetadataVOL::serve_all()`).
- `receive_as_consumer()` — called by the consumer's driver before its own,
  unmodified `H5Fopen`; receives the producer's file index
  (`DistMetadataVOL::broadcast_files()`).
- `move()` is refused (`E-TRANSPORT-LF-DISTRIBUTED`) once `set_intercomm()` has been
  called: writing and reading back in one call is meaningless once the two ends are
  two different processes.

**Forming the intercommunicator is explicitly NOT this transport's job** — it is
handed one, already formed, by `MPI_Comm_connect`/`MPI_Comm_accept` or an MPMD
launch's split. `examples/demo_b` performs its own minimal rendezvous for exactly this
reason, deliberately NOT reusing `src/transport/connect.cpp` (a different track's
file, out of this one's ownership for this phase) — see `KNOWN_ISSUES.md` section D
for the open question this raises for the lead about whether the two should merge.

### Demo B's own file split

`producer_hdf5.cpp` / `consumer_hdf5.cpp` are plain HDF5 — no FUSE header, no FUSE
call, checked by `test/verification/demo_b_test.cpp`'s own grep (if/when the lead
registers it) the same way upstream SOMA's phase-8 plan checks its own
`producer.cpp`/`consumer.cpp`. `producer_main.cpp` / `consumer_main.cpp` are the FUSE
side: they call `set_intercomm`/`set_location`/`build_plan`, then call into the
unmodified payload functions, then `serve_as_producer`/`receive_as_consumer`. Splitting
the files this way is what makes "the HDF5-writing code never mentions FUSE" checkable
by a grep over two specific files, rather than an informal claim about the whole demo.

### The dataset-name ↔ field mapping

`fuse/transport/lowfive.hpp`'s `LowFiveFieldBinding`/`LowFiveFieldBindingTable` map a
field name to an HDF5 port pattern (`"<file>.h5/<dataset path>"`, wildcards permitted
in the file component — Wilkins's own data-centric port model, named in
`docs/plans/soma-phase-8.md`'s Task 8.4). This is deliberately **not** attached to
`fuse::TransportSpec` (shared by every transport backend, and mid-edit by this
phase's "MPI connect" track) or `fuse::FieldSpec` (a different track's layer); it is
Demo B's own, local bookkeeping today. Folding it into the coupling graph properly —
so a YAML `channels:` entry can name an HDF5 dataset the way it names a field — is an
open question for the lead; this track's report names the exact field and suggests a
descriptor version.

## The `HDF5_VOL_CONNECTOR` abort (why the connector is attached programmatically)

A dangling reference existed in `src/transport/lowfive_transport.cpp` and
`fuse/transport/transport.hpp` pointing at a `docs/BUILDING.md` that this
repository (unlike its `upstream/soma` parent) does not have; this section is that
record, moved to where this checkout actually keeps it, and the three comments now
point here instead.

Loading the connector via the `HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH` environment
variables (rather than programmatically, via `LowFive::H5VOLProperty`) makes the
process abort *after* `main` returns:

```
libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument
exit=134
```

All program output is produced first; the abort is in teardown, and it is not a
`dlclose` problem (`RTLD_NODELETE` does not help). HDF5's `atexit` handler terminates
the connector after LowFive's own C++ statics (its spdlog logger and that logger's
mutex) have already been destroyed. The upstream parent project's own record
(`upstream/soma/docs/BUILDING.md`) reproduces the same failure even on the
programmatic path, whenever a native-VOL HDF5 call happens before the connector is
registered — the fix in both codebases is the same: register
`std::atexit(+[]{ H5close(); })` as part of constructing the VOL singleton, which
puts it ahead of HDF5's own atexit registration in the LIFO unwind order, so it runs
first. `VolSingleton`'s constructor (`lowfive_transport.cpp`) does exactly this.

## See also

- `KNOWN_ISSUES.md` section D for this hazard's entry and the Demo B execution gap.
- `tools/probe-hdf5-images.c`/`.sh` for the acceptance test in full.
