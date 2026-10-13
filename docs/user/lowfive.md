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

**How the VOL gets onto the application's own `H5Fcreate`/`H5Fopen` call --
TWO mechanisms now exist, and the participants use the programmatic one.**
`LowFive::H5VOLProperty::apply(hid_t list)` (`lowfive/H5VOLProperty.hpp`) is
the only attachment primitive LowFive's own header exposes, and it takes an
EXPLICIT property list — an application calling it itself would be a
FUSE-specific call, contradicting spec §6 mode 4's contract. The mechanism
that keeps the contract literal is HDF5's own: HDF5 reads the
`HDF5_VOL_CONNECTOR` and `HDF5_PLUGIN_PATH` environment variables at
library initialisation and makes the named connector the default for every
`H5P_DEFAULT` file access thereafter — no caller-side property-list call of
any kind. LowFive ships no loadable plugin of its own (`vol-metadata.hpp`
exposes only the programmatic `H5VOLProperty`/`MetadataVOL` path this
page's single-process section already uses), so **FUSE ships one**: a small
shared library (`src/transport/lowfive_plugin_vol.cpp` +
`lowfive_plugin_graph.cpp`, target `fuse-lowfive-vol`) exporting the two
symbols HDF5's plugin loader looks for,
`H5PLget_plugin_type`/`H5PLget_plugin_info`. This plugin is UNCHANGED by the
"LowFive attach" track below and stays in the tree for a later track (it is
exercised by nothing in this build, now that the participants no longer use
it — see KNOWN_ISSUES.md D39).

**MEASURED, 2026-10-12 (`build-lowfive`, main `4fd400d`): the env-var route
is what the participants' own FIRST `H5Fcreate` null-dereferenced in.** With
KNOWN_ISSUES.md D38's own fix in place (the `ExternalDelivery` size
exemption), `verification.brusselator.participants.lowfive.{mpmd,accept}.3`
got past `fuse_initialize()` and both `brusselator_u_main`/
`brusselator_v_main` then crashed inside `LowFive::VOLBase::_file_create` on
their own first `bruss_lowfive_write()` call — `info_t::vol` (a
CLASS-STATIC member `create_MetadataVOL()`/`create_DistMetadataVOL()` set)
was still null at the point the env-string connector-name parser
(`str_to_info`) handed back a fresh `info_t`, for a reason KNOWN_ISSUES.md
D39 states but does not chase (whether `fuse-lowfive-vol`'s own
`ConfigureOnce()` ran before this call at all, or was cleared by
`VOLBase::_term` first). **The "LowFive attach" track's own fix sidesteps
this question rather than answering it**: `examples/brusselator/
participants/lowfive_attach.{h,cpp}` builds the SAME programmatic
`H5VOLProperty`/`MetadataVOL`-or-`DistMetadataVOL` construction
`VolSingleton`/`DistVolSingleton` already use in the measured single-process
path (`lowfive_transport.cpp`), and `bruss_lowfive_write`/`bruss_lowfive_
read` (`shim.{h,cpp}`) now take the resulting file-access property list as
an explicit `fapl_id` argument, passed to every `H5Fcreate`/`H5Fopen`
literally, in place of `H5P_DEFAULT` — never reaching HDF5's own
default-VOL resolution (the env-var route above) at all. `bruss_lowfive_
attach()` is called exactly once per process, right after `MPI_Init()`,
reading field bindings and writer/reader `RankSet`s from the SAME `--graph`
path the program's own command line already names (`fuse::
build_lowfive_bindings()`, the identical function `lowfive_plugin_graph.cpp`
calls) — no separate `FUSE_LOWFIVE_BINDINGS` environment variable for this
path. For the genuinely cross-process MPMD case (U and V are different
executables sharing one `MPI_COMM_WORLD`), the attach forms a
`LowFive::DistMetadataVOL` over an intercommunicator built with
`MPI_Intercomm_create()` from the caller's own ALREADY-FORMED physics-only
communicator (`u_main.c`'s own `u_comm` / `v_main.f90`'s own `v_comm`,
passed to `bruss_lowfive_attach()` as a new `physics_comm_f` argument) —
NOT a fresh `MPI_Comm_split(MPI_COMM_WORLD, ...)` of the attach's own, which
an earlier version of this track used and which KNOWN_ISSUES.md D39's own
"found while implementing" paragraph explains would hang (that split is
collective over the WHOLE of `MPI_COMM_WORLD`, including the coupler's own
rank, and the coupler never calls `bruss_lowfive_attach()` at all). The
rendezvous arithmetic itself (`fuse::plan_lowfive_rendezvous()`) is the SAME
the plugin's own `ConfigureDistributed()` already computes — hardened with a check that
comment names as missing (every HDF5-bound channel a rank takes part in
must agree on the SAME two-group partition; see KNOWN_ISSUES.md D39's own
account and `examples/brusselator/participants/lowfive_attach_decision.hpp`
for the full reasoning, tier-0 tested in `test/verification/
brusselator_lowfive_attach_test.cpp`). Accept mode refuses cleanly at
attach time (`E-LOWFIVE-ATTACH-ACCEPT`) rather than forming a
half-configured VOL — D14's own accept-mode intercommunicator question is
still open, and `verification.brusselator.participants.lowfive.accept.3`
stays red on purpose until it is decided.

The Brusselator's own `u_main.c`/`v_main.f90` (not a separate
LowFive-specific main, as an earlier plan for this section assumed before
Unit 3b was actually built — `examples/brusselator/participants/
{u_main.c,v_main.f90}` gained a `--transport lowfive` flag instead,
KNOWN_ISSUES.md D14's own Unit 3b account) call `H5Fcreate`/`H5Dwrite`/
`H5Fopen`/`H5Dread` through a small shim (`examples/brusselator/
participants/shim.{h,cpp}`'s own `bruss_lowfive_write`/`bruss_lowfive_read`
— plain HDF5, needed only because this host's own HDF5 ships no Fortran
module for `v_main.f90` to call directly) gated on
`fuse_step_report::delivery_pending`/`delivery_field` (`docs/user/
c-abi.md`'s own protocol), never on every substep. Internally, `shim.cpp`
also brackets each write's own `H5Fclose` with `bruss_lowfive_serve_
after_write()` and each read's own `H5Fopen` with `bruss_lowfive_receive_
before_read()` (no-ops in single-process mode) — mirroring Demo B's own
producer/consumer call order (below) for the Brusselator's bidirectional
exchange; whether that ordering is free of a mutual-wait deadlock between
the two processes is UNVERIFIED (no MPI in the sandbox either track was
written in) and is this track's own open item, not this page's.

**Host command lines, exactly as `ctest` runs them**
(`verification.brusselator.participants.lowfive.{mpmd,accept}.3`,
`examples/CMakeLists.txt`) -- world size 3 only (1+1+1). The three
environment variables below are still set in the test's own `ENVIRONMENT`
(an additive-only constraint on that shared file left them in place), but
are VESTIGIAL for both rows as of the "LowFive attach" track: every
`bruss_lowfive_write()`/`bruss_lowfive_read()` call now carries an explicit
`fapl_id` and refuses outright on a negative one, so neither row's own
`H5Fcreate`/`H5Fopen` ever reaches `HDF5_VOL_CONNECTOR`'s own default-VOL
resolution:

```
# vestigial for both rows now -- kept set, not load-bearing (see above)
export HDF5_VOL_CONNECTOR=lowfive
export HDF5_PLUGIN_PATH=<directory containing fuse-lowfive-vol>
export FUSE_LOWFIVE_BINDINGS=<scratch>/graph.yaml

# lowfive.mpmd.3 -- each program's own bruss_lowfive_attach() call (right
# after MPI_Init) reads graph.yaml from --graph below, not from the
# FUSE_LOWFIVE_BINDINGS variable above
mpiexec -n 1 brusselator_u_main --graph graph.yaml --log u.log --periods 2 --transport lowfive \
    : -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --transport lowfive \
    : -n 1 brusselator_coupler_main --graph graph.yaml

# lowfive.accept.3 -- bruss_lowfive_attach() refuses cleanly on both U and V
# (E-LOWFIVE-ATTACH-ACCEPT) the moment it reads the graph's own
# LaunchMode::Connect; stays red on purpose, KNOWN_ISSUES.md D39
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 --transport lowfive &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --transport lowfive &
wait

brusselator_check --u-log u.log --v-log v.log
```

**Unmeasured until the host's LowFive loads at all and runs these two rows**
(the libfmt.8 blocker above, KNOWN_ISSUES.md D14's own consolidated "what
remains" list, closed by a host action per D14/D38's own later
measurements) -- the tier-1+LowFive build is confirmed clean including the
"LowFive attach" track's own additions (this page's account above); whether
`lowfive.mpmd.3` now passes, and whether its own bidirectional serve/receive
choreography is free of the mutual-wait risk this page's account above
names, are the two questions the host run answers that this sandbox
cannot -- see KNOWN_ISSUES.md D39's own "what the host must verify"
paragraph for the exact command and what each outcome looks like.

**IR/runtime half, implemented and measured (no LowFive/HDF5 needed) —
2026-10-04.** `lower_channel()` (`src/ir/rules_channel.cpp`) emits a new
`OpKind::ExternalDelivery` node (`include/fuse/ir/op.hpp`) in place of
reading the channel's own producing arc, when the source field sets
`hdf5_path`. This severs the DATA edge FUSE would otherwise read the
field's own producer (its ordinary `Source`, or for a window-scoped
export its `WindowAdvance`) through — correct for a deployment where the
two sides may be different processes with no such edge at all — while
that producer keeps running unaffected (the component's own export
computation is not touched), and the application (an ordinary HDF5
write/read, or a test harness standing in for one) fills the new node's
own output instead. The DAG edge from that same producer is NOT severed,
though: `ExternalDelivery` carries it as an input of its own, same as
`Transport` would for an ordinary channel — an ORDERING edge only, never
read by `execute()` — so the scheduler places this node exactly where
`Transport` would stand (after the producer, before the channel's own
Remap/Diagnose/Sink), on every process. An earlier version of this node
had no input at all, and a LowFive-bound channel's own chain was
measured, this session, to get scheduled before the exporting member's
own `WindowAdvance` within the SAME sweep instead of after it — reading
last sweep's state, not the fresh one Transport's own ordering guarantees
for the ordinary case. The new node's arc is typed with the
CONSUMER's own rank_set, not the producer's (the application's own read
lands on the reader's own placement), which is also what makes the
Transport-before-Remap and coupler-second-hop logic further down in the
same function correctly emit neither a `Transport` nor an `Epoch` for
this channel's data hop — confirmed, not assumed: `ranks_differ` is
forced `false` for a LowFive-bound channel rather than left to a
rank-set comparison that would otherwise still fire one. Everything
downstream (Scale, Remap when the two meshes differ, Extrapolate, the
window-importing Sink) is unchanged, operating on the externally-delivered
arc exactly as it would on a Transport's.

Measured against two graphs, both green: `test/ir/lowering_channel_test.cpp`'s
`LowFiveBoundChannelEmitsExternalDeliveryNotTransport` (a synthetic,
non-window, two-rank-set pair) and
`test/verification/brusselator_graph_tool_lowfive_test.cpp`'s
`LowersWithExternalDeliveryAndNoTransport`, against the REAL tool-built,
window-scoped, genuinely cross-rank (`{0,1}`/`{1,1}`), genuinely
cross-mesh (3x3/4x4) Brusselator LowFive graph — two `ExternalDelivery`
nodes (one per LowFive-bound channel), zero `Transport`, zero `Epoch`.
Full tier-0 (166/166) and all 5 repo-health gates clean throughout.

**The ABI view of the window buffers, implemented and measured --
2026-10-04.** `fuse_field_view` gains an appended `window_samples`
member (`include/fuse/abi/entries.def`, by the growth rules --
`FirstFrozenStructSize<fuse_field_view>` records the pre-growth size, an
old caller's smaller `struct_size` never has it written). A new
participant-table entry, `view_field` (named to avoid colliding with the
`fuse_field_view` *type* once Fortran lower-cases both -- `field_view`
was the first name tried and `tools/gen-abi.py` caught the collision),
appended LAST per the table's own append-only rule (an entry inserted
mid-table would have shifted every later member's offset for an
existing caller -- caught and fixed before committing, by
`test/abi/negotiation_test.cpp`'s own `FakeV2Table` size check). Before
this change there was no FUSE-to-application view accessor at all --
only `register_field()` (application-to-FUSE). `Runner::field_window_
view(FieldId)` (next to `buffer()`, `include/fuse/runtime/runner.hpp`)
is the C++ accessor `impl_view_field` (`src/abi/participant.cpp`) calls:
two new FieldId-keyed maps (`window_export_field_arc`, `window_import_
field_arc`, populated in `resolve_regions()` alongside `window_member_
config` itself, from the SAME `exported_field`/`imported_field` that
function already resolves, not re-derived) answer "is this field part
of a coupling window" in O(1); a field in neither map gets `window_
samples = 0` and the ABI layer falls back to its own registered
storage, so the new entry answers uniformly for ANY field, window-scoped
or not. Layout measured against `Runner::Impl::resolve_regions()`'s own
priming loop (`data[e * K + s]`, entity-major then sample,
`src/runtime/iteration.cpp`) and reported the same way an ordinary
multi-component field already is: `components` widened to the field's
own declared width times `window_samples`, `count` unchanged (entities).
Full tier-0 (166/166) and all 5 repo-health gates clean.

**Unit 3a ("single process, the measured path"), the LowFive-free half —
DONE.** `test/verification/brusselator_lowfive_harness_test.cpp` drives
the real Brusselator species core (the SAME step functions `u_main.c`/
`v_main.f90` call, through the shim) over a genuinely multirate,
window-scoped, LowFive-bound graph, with `view_field()` + `memcpy()`
standing in for an application's H5Dwrite/H5Dread pair at every delivery
point. `BrusselatorLowFiveHarness.ExternalDeliveryMatchesInProcessRunner`
matches a reference `fuse::Runner` bit for bit — not just the final
state but every intermediate sweep, with identical iteration counts in
both waveform directions — pinned with no tolerance anywhere
(`EXPECT_EQ`, hexfloat on the first differing value were there one).

Reaching that bit-exact match took two real library fixes, both
permanent, not test-only workarounds (full account in `KNOWN_ISSUES.md`
D14's own "Unit 3a" paragraph):

1. An `ExternalDelivery` instruction is now a genuine suspension point
   (`Runner::Impl::deliver_external()`, `src/runtime/iteration.cpp`),
   not the no-op it started as — the Runner hands control back to the
   application between a LowFive-bound export and its import, exactly
   as it already does for a component's own `advance()`. The ABI's own
   `fuse_step_report::step_opened` field (appended for this) and the
   corrected `next_component` resolution (`Runner::Impl::next_real_
   advance_after()`, wrapping to a region's own entry when a forward
   scan from the delivery finds nothing before the region's exit) are
   what let a caller resume past it correctly; `docs/user/c-abi.md`'s
   protocol section documents the resulting loop shape.
2. `WindowMemberConfig::window_import_delivery_arc` (`src/runtime/
   runner_impl.hpp`) fixes a t=0 read in `advance_window()` that
   assumed the importing member's own chain had already run this
   sweep — true before the scheduling-order fix that moved a delivery
   instruction's own position to coincide with a sweep's own end, false
   after.

Covered, beyond the harness itself: a byte-exact golden-IR fixture
(`test/golden/pair_lowfive_bound.ir`, `GoldenIr.PairLowFiveBound`)
pinning the compiled `ExternalDelivery` shape independent of runtime
behavior; three dedicated `AbiParticipant` cases in `test/abi/
participant_test.cpp` for the rollover mechanics at the ABI boundary
(a matched rollover, an end-time-bounded run, and case (ii) — a
purpose-built graph where the window's own region-entry and the true
period-start component are different components by construction); and
six hand-proved mutations (`build-evidence/mutations/abi.toml` and
`build-evidence/mutations/transport-lowfive.toml`) confirming the
suspension, the cold-start delivery, and the `lowfive_bound` channel
classification are each load-bearing, not redundant with anything else.
Full tier-0 and all 5 repo-health gates clean throughout.

Separately, and reported as asked: whether a window member's own
application code can tell "this is my last substep of the current
sweep" from `fuse_step_request`/`fuse_step_report` alone -- yes, from
`t_end` (already populated for the participant direction, `src/abi/
participant.cpp:967-1018`) checked against the window boundary, using
the SAME window-length constant the participant's own `main()` is
already built with (identical to how every existing Brusselator
participant matches its CLI flags to the graph tool's); no appended
`int32_t` is needed for this piece.

**Still not done:** the participant mains' own `H5Dwrite`/`H5Dread`
wiring against REAL HDF5/LowFive (the distributed, cross-process half)
— blocked on the host-side libfmt/two-image issues this page's earlier
sections record, unrelated to anything Unit 3a's own library fixes
touch.

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
