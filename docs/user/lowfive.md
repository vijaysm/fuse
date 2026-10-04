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
completion by this probe either, but for an UNRELATED reason, measured
2026-10-02**: its own `liblowfive.dylib` linked against
`/opt/mpich/lib/libmpicxx.12.dylib`, an MPICH ABI version older than the one
then installed at `/opt/mpich` (`libmpicxx.0.dylib`, MPICH 5.0 per
`mpichversion`). The process aborted at load time with `Library not loaded:
.../libmpicxx.12.dylib` — but only AFTER the loader had already mapped both
`libhdf5`/`libhdf5_hl` (confirmed by inspecting the trace up to that point),
so the image-count verdict was unaffected. This was a second, independent
defect in this lab's `/opt/lowfive` install (a stale MPI linkage), not the
hazard this page is about; `tools/probe-hdf5-images.sh` reported it as a
separate warning rather than folding it into the PASS/FAIL verdict.
**STALE as of 2026-10-04: `/opt/lowfive` has since been rebuilt and now links
the current `libmpicxx.0.dylib`/`libmpi.0.dylib`/`libpmpi.0.dylib` — this
specific defect is closed. It has been superseded by a DIFFERENT, still-open
one (both shared libraries' identical `@rpath/libfmt.8.dylib` dependency),
measured below under "A third `/opt/lowfive` install defect".**

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

### A third `/opt/lowfive` install defect, measured 2026-10-04: neither shared library can load on this host

The two-HDF5-image hazard above is not the only defect this lab's `/opt/lowfive`
install carries, and the stale-MPI-linkage paragraph in "The fix, and why it is a
host action" is now itself stale: `otool -L` on both shared libraries, run
2026-10-04, shows

```
$ otool -L /opt/lowfive/lib/liblowfive.dylib
	@rpath/liblowfive.dylib ...
	@rpath/libfmt.8.dylib (compatibility version 8.0.0, current version 8.1.1)
	...
	/opt/mpich/lib/libmpicxx.0.dylib ...
	/opt/mpich/lib/libmpi.0.dylib ...
	/opt/mpich/lib/libpmpi.0.dylib ...

$ otool -L /opt/lowfive/lib/liblowfive-dist.dylib
	@rpath/liblowfive-dist.dylib ...
	@rpath/liblowfive.dylib ...
	...
	@rpath/libfmt.8.dylib (compatibility version 8.0.0, current version 8.1.1)
	/opt/mpich/lib/libmpicxx.0.dylib ...
	/opt/mpich/lib/libmpi.0.dylib ...
	/opt/mpich/lib/libpmpi.0.dylib ...
```

**The MPI linkage is current now** — `libmpicxx.0.dylib`/`libmpi.0.dylib`/
`libpmpi.0.dylib`, not the stale `libmpicxx.12.dylib` this page recorded
2026-10-02: `/opt/lowfive` has been rebuilt since that paragraph was written,
and that specific defect is closed.

**Both libraries now carry an identical, unresolvable `@rpath/libfmt.8.dylib`
dependency.** This host has only fmt 12 (`/opt/homebrew/lib/libfmt.12.dylib`,
`libfmt.12.2.0.dylib`) — no `libfmt.8.dylib` anywhere under `/opt` or any
installed conda/homebrew prefix. The dyld error actually observed running
`fuse_transport_lowfive_test` names `liblowfive-dist.dylib` as the referencer
(load order — it is checked first):

```
dyld[99193]: Library not loaded: @rpath/libfmt.8.dylib
  Referenced from: <...> /opt/lowfive/lib/liblowfive-dist.dylib
  Reason: tried: '/opt/lowfive/lib/libfmt.8.dylib' (no such file), ...
```

— but the `otool -L` evidence above shows the identical dependency on the MAIN
`liblowfive.dylib` too, so this is not specific to the distributed archive. A
separate, EARLIER link-time failure (`fmt::v8::detail::...` symbols missing
for `liblowfive-dist.a`, the static archive, compiled against a real,
non-header-only fmt 8) is a different, already-isolated defect: forcing CMake
to treat `liblowfive-dist` as not-found
(`-DLOWFIVE_DIST_LIBRARY=LOWFIVE_DIST_LIBRARY-NOTFOUND`) removes that link
failure, confirming it was specific to the static archive — but the resulting
binary still fails to even start, at `dlopen`, on the identical
`libfmt.8.dylib` dependency the main shared library itself carries.

This means **no LowFive-linked binary can execute in this sandbox at all**, not
even the single-process path this page calls "implemented, measured" above —
that measurement predates this finding and was not re-confirmed against this
exact `/opt/lowfive` state. Relinking `/opt/lowfive`'s shared libraries against
an fmt this host actually has (or installing a compatible `libfmt.8.dylib`
for them to find) is a host action, like the hazard above; this sandbox has no
writable `/opt` and was declined permission to install a workaround via a new
package environment. Until that is fixed, every claim below the "Unit 3: a
LowFive-bound channel is an application-performed transfer" section is argued
from reading LowFive's own header contracts, not measured by running
anything.

## Unit 3 (S6 phase 3, Conformance): a LowFive-bound channel is an application-performed transfer

**Design decision, recorded before implementation, per the lead's own ruling
2026-10-04.** A channel whose field sets `hdf5_path` (`FieldSpec::hdf5_path`,
`fuse/model/field_spec.hpp`) is NOT moved by any `fuse::Transport`, including
`LowFiveTransport::move()`. The entire premise of putting LowFive in this
coupler is spec §6 mode 4's own contract: **"no FUSE-specific call on either
side"** — a code that already writes HDF5 couples with another that already
reads it, with LowFive intercepting in memory, and FUSE never touches the
bytes. `LowFiveTransport::move()` (its own internal `H5Fcreate`/`H5Dwrite`
then `H5Fopen`/`H5Dread` in one call, `src/transport/lowfive_transport.cpp`)
is the RIGHT tool for a channel FUSE itself must move end to end with no
unmodified external writer/reader — which is exactly what it already serves,
and remains unchanged — but it is the WRONG tool for this one, where the two
ends are ALREADY ordinary, unmodified HDF5 callers by design.

What FUSE does for a LowFive-bound channel instead: the schedule (when U must
have written before V may read), the waveform/window semantics (how many
samples the channel carries per window), Interpolate/Remap/Constrain if the
two sides' meshes differ, the convergence decision, and installing LowFive's
VOL connector with the right file/dataset bindings
(`fuse::build_lowfive_bindings()`, `fuse/transport/lowfive_graph.hpp`) so
neither side's H5Dwrite/H5Dread ever reaches a real disk. Everything else —
the actual bytes — is the two applications' own, ordinary HDF5 calls, exactly
as spec §6 mode 4 promises.

**How the VOL gets onto the application's own `H5Fcreate`/`H5Fopen` call,
with literal `H5P_DEFAULT`: HDF5's own default-VOL mechanism, THE design,
unmeasured until the host unblocks.** `LowFive::H5VOLProperty::apply(hid_t
list)` (`lowfive/H5VOLProperty.hpp`) is the only attachment primitive
LowFive's own header exposes, and it takes an EXPLICIT property list — an
application calling it itself would be a FUSE-specific call, contradicting
spec §6 mode 4's contract. The mechanism that keeps the contract literal is
HDF5's own: HDF5 reads the `HDF5_VOL_CONNECTOR` and `HDF5_PLUGIN_PATH`
environment variables at library initialisation and makes the named
connector the default for every `H5P_DEFAULT` file access thereafter — no
caller-side property-list call of any kind. LowFive ships no loadable plugin
of its own (`vol-metadata.hpp` exposes only the programmatic
`H5VOLProperty`/`MetadataVOL` path this page's single-process section
already uses), so **FUSE ships one**: a small shared library
(`src/transport/lowfive_plugin.cpp`, target `fuse-lowfive-vol`) exporting
the two symbols HDF5's plugin loader looks for,
`H5PLget_plugin_type`/`H5PLget_plugin_info`, which on load constructs
LowFive's `MetadataVOL` singleton with `set_memory`/`set_keep` (identical to
`VolSingleton`'s own constructor, `lowfive_transport.cpp:133-143`, including
its `std::atexit(+[]{ H5close(); })` teardown-ordering fix — the SAME
mutex-destruction-order hazard "The `HDF5_VOL_CONNECTOR` abort" section
below records applies here too, and the fix is the same one, just carried
by this new plugin's own constructor instead of `VolSingleton`'s) and the
field bindings, read from an environment variable `FUSE_LOWFIVE_BINDINGS`
naming the graph's own YAML path, through `fuse::build_lowfive_bindings()`.
The launcher — the ctest row, or a user's own job script — sets the two
HDF5 variables (and `FUSE_LOWFIVE_BINDINGS`); `uv_lowfive_main.c`'s own
`H5Fcreate`/`H5Dwrite`/`H5Fopen`/`H5Dread` calls are then genuinely ordinary,
with no FUSE call anywhere in them. **Unmeasured until the host's LowFive
loads at all** (the libfmt.8 blocker above) — this is recorded as the
design, not implemented yet; implementing a plugin that cannot be run here
would be writing code with no way to catch its own mistakes.

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

**Does not rule out `HDF5_VOL_CONNECTOR` for Unit 3's own plugin
(`fuse-lowfive-vol`, "Unit 3" section above).** What aborted here was loading
LowFive's OWN connector through the env-var mechanism with no fix in
place at all; the fix itself — `std::atexit(+[]{ H5close(); })`, registered
before any native-VOL HDF5 call — is a property of WHEN it runs, not of
WHICH loading path reached it. `fuse-lowfive-vol`'s own
`H5PLget_plugin_info` carries the identical fix in its own construction
path, so the mechanism this abort argues against programmatic-vs-plugin in
general is available to FUSE's plugin the same way it is to `VolSingleton`
today — unmeasured, like the rest of that design, until the host's LowFive
loads at all.

## See also

- `KNOWN_ISSUES.md` section D for this hazard's entry and the Demo B execution gap.
- `tools/probe-hdf5-images.c`/`.sh` for the acceptance test in full.
