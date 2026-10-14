# LowFive: coupling through intercepted HDF5

Spec §6 mode 4: two components exchange a field by one writing it with ordinary
`H5Fcreate`/`H5Dwrite` calls and the other reading it back with ordinary
`H5Fopen`/`H5Dread` calls — no FUSE-specific call on either side. LowFive, Argonne's
HDF5 Virtual Object Layer (VOL) connector, intercepts those calls in memory; no file
ever reaches disk. `fuse::LowFiveTransport` (`fuse/transport/transport.hpp`,
`src/transport/lowfive_transport.cpp`) is the seam.

This page has two parts: the single-process interception path, which is implemented
and measured end to end, and the cross-process (two-executable, "Demo B") path, which
is measured for one producer and one consumer process (ctest row `example.demo_b`,
2026-10-14; [Demo B](demo-b.md)). Older passages below that call the cross-process path
unexecuted are dated records of the state before that run.

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
brusselator_lowfive_attach_test.cpp`). Accept mode refused cleanly at
attach time (`E-LOWFIVE-ATTACH-ACCEPT`) rather than forming a
half-configured VOL through this round — D14's own accept-mode
intercommunicator question was still open, and
`verification.brusselator.participants.lowfive.accept.3` stayed red on
purpose until it was decided.

**Superseded, D14 Stage B (2026-10-13): the intercommunicator question is
decided, in-sandbox only.** The paragraph above describes
`bruss_lowfive_attach()` taking the caller's own already-formed
PHYSICS-ONLY communicator (`physics_comm_f`, U's `u_comm` / V's
`v_comm`) directly as the peer for `MPI_Intercomm_create()`. That
signature is gone: `bruss_lowfive_attach()` now takes a `bridge_comm_f`
(the merged base communicator `fuse_setup()` forms for a `Connect`-mode
component, now readable by any caller through the new, append-only ABI
entry `base_communicator()` / `fuse_base_communicator_info`,
`docs/user/c-abi.md`) and a `local_comm_f` the caller must obtain
separately, from the new `bruss_lowfive_join_local_split()` — a fresh,
collective `MPI_Comm_split` of that bridge, keyed by which of the
graph's own writer/reader `RankSet`s this rank belongs to (NOT a reuse
of `u_comm`/`v_comm`: the owner's own ruling was that reusing them
assumes, rather than derives, that their local ranks already match the
plan's expected numbering). Every rank that shares the bridge —
including the bundled coupler/placeholder rank that hosts no LowFive
channel at all — must call `bruss_lowfive_join_local_split()`
(a collective, never skippable), but only a rank that actually hosts a
qualifying channel goes on to call `bruss_lowfive_attach()` itself. With
the bridge exposed, `decide_attach_mode()` no longer refuses a
Connect-mode channel outright: it now refuses only a bridge that was
never formed (`E-LOWFIVE-ATTACH-ACCEPT-NO-BRIDGE`) or more than two
Connect-mode components (`E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY`,
D18), and otherwise proceeds exactly like an MPMD channel. Host-measured on the
owner's machine, 2026-10-13: `verification.brusselator.participants.lowfive.accept.3`
Passed on its first host run (1.26 s), with `brusselator_check --check: OK` and
`u.log`/`v.log` byte-identical to the non-LowFive accept row's; the numbers are in
KNOWN_ISSUES.md D14's "what remains" item 3. Only one configuration (one rank
each for U and V, plus the bundled coupler rank) has been run.

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
before_read()` (no-ops in single-process mode).

**Round 2 correction**, replacing what this paragraph said before (serve
right after `H5Fclose`, receive right before `H5Fopen`, mirroring Demo
B's own one-directional call order, with the mutual-wait risk left
"UNVERIFIED... this track's own open item"): the host MEASURED that risk
on 2026-10-13 — `verification.brusselator.participants.lowfive.mpmd.3`
hung the full 120 s, with both `brusselator_u_main` and `brusselator_v_
main` caught (macOS `sample`) inside `H5Fclose -> ... -> DistMetadataVOL::
file_close -> DistMetadataVOL::serve_all -> Index::serve`'s own
`MPI_Iprobe` loop, neither having read the other first. Reading LowFive's
own source (`/Users/mahadevan/Code/hdf5/LowFive`, rev `7f3a58b`) explains
why the "mirrors Demo B" framing above was already wrong the moment it
was written: `DistMetadataVOL::serve_on_close` defaults `true`, so
`file_close()` calls `serve_all()` itself, INSIDE `H5Fclose` — before
`bruss_lowfive_serve_after_write()`'s own round-1 body, placed AFTER
`H5Fclose` returns, ever ran a line.

**Round 2, SECOND correction** — the first fix this page described
right here (receive the peer's file FIRST, then flush a deferred serve,
BOTH inside the pre-open hook, before `H5Fopen`) was itself reviewed
against LowFive's own source and REJECTED before any host run: a
process blocked inside `serve_all()` for a file is released only when
the PEER completes its own open-AND-close of that file (`DistMetadataVOL
::file_close()`'s own `RemoteFile` branch calling `finish(0)`) — not by
the peer merely reaching a pre-open hook. Flushing there ran a first-
reader's own blocking serve before it had done anything that could
release the first-server's own pending one: the same circular wait one
write/read cycle later. The actual fix (`examples/brusselator/
participants/lowfive_attach.cpp`, KNOWN_ISSUES.md D39's own "Round 2
REVIEW" account has the full citations and the deadlock-freedom
argument, re-derived in terms of what the PEER observes): `bruss_
lowfive_attach()` still sets `serve_on_close = false`, and the pure,
role-ordered state machine (`lowfive_attach_decision.hpp`'s own
`ServeState`/`serve_state_step()`) now flushes a first-reader's own
deferred serve at a NEW event, fired AFTER this process's own `H5Fclose`
of the peer's file (`bruss_lowfive_after_read_close()`, called from
`shim.cpp`'s own `bruss_lowfive_read()`) — never before `H5Fopen`. The
pre-open hook (`bruss_lowfive_receive_before_read()`) does nothing
LowFive-level at all now: the real receive is `H5Fopen` itself, and
`LowFive::DistMetadataVOL::broadcast_files()` (which the first fix also
called there) turns out to operate entirely over the LOCAL communicator,
propagating an already-known file registry to a multi-rank component's
own sibling ranks, never to the peer — not reachable from any code path
inside `file_open()`/`file_close()` themselves, and removed. A fourth
hook, `bruss_lowfive_finalize()`, runs once at process exit and refuses
loudly if a deferred serve was ever left unflushed. This still
serializes the two transfers (never both sides serving in the same
window); the helper-thread follow-up that would let them overlap again
stays NOT started, for the same reason as before (`Index::serve()`'s own
barrier/ibarrier path and `Query::get_filenames()`'s own broadcast share
the SAME `local` communicator, and this build's own HDF5 is configured
`Threadsafety: no` with a plain `MPI_Init`). This version is checked
against an explicit-state simulation of the blocking semantics above
(`examples/brusselator/participants/lowfive_attach_model_check.hpp`),
not only against a written argument — that header's own top comment
states plainly what the simulation does and does not establish.

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

# lowfive.accept.3 -- D14 Stage B (2026-10-13, host-run: Passed, 1.26 s):
# base_communicator() exposes the merged bridge, and
# bruss_lowfive_join_local_split()/bruss_lowfive_attach() form the
# distributed attach on it. Both U's physics rank and V exchange through
# LowFive like the mpmd row above; U's bundled coupler/placeholder rank
# joins the collective split but never calls attach().
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 --transport lowfive &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --transport lowfive &
wait

brusselator_check --u-log u.log --v-log v.log
```

**Unmeasured on THIS round's sandbox (no MPI); PARTIALLY measured on the
host already.** The host ran `lowfive.mpmd.3` after round 1's own fix
(libfmt.8 blocker closed, D14/D38) and measured the mutual-wait risk this
page's account above once called "UNVERIFIED" directly: a 120 s hang,
both participants caught inside `DistMetadataVOL::serve_all`'s own
`Index::serve` probe loop. The FIRST round-2 fix (role-ordered serve,
flushing at the pre-open hook) never reached a host run at all -- caught
by review against LowFive's own source first and rejected (this page's
own "Round 2, SECOND correction" paragraph above has the account). The
SECOND fix (flush moved to a new post-close hook) is believed to close
the measured hang by construction (the deadlock-freedom argument in
KNOWN_ISSUES.md D39's own "Round 2 REVIEW" account, now also checked
against an explicit-state simulation of the blocking semantics --
`lowfive_attach_model_check.hpp`), but is itself UNEXECUTED on any
sandbox with a working MPI launcher -- whether `lowfive.mpmd.3` now
passes end to end, and whether `bruss_lowfive_finalize()` ever actually
fires its own unflushed-serve refusal on this row's own graph (it should
not, if I1/I2 hold as argued), are what the next host run answers -- see
KNOWN_ISSUES.md D39's own "what the host must verify" paragraph for the
exact command and what each outcome looks like.

**Round 4 correction** -- the SECOND fix above DID close the measured hang
(the host ran it: `lowfive.mpmd.3` no longer times out), but both
`brusselator_u_main` and `brusselator_v_main` now die in 1.2 s at their own
FIRST remote `H5Fopen`, with an uncaught `LowFive::MetadataError`,
`file_open(): no intercomm found for brusselator_{v,u}.h5`
(`DistMetadataVOL::file_open()`, `src/vol-dist-metadata.cpp:277` on this
host's own LowFive checkout). Two defects, found by reading LowFive's own
source rather than guessed from the exception text: nothing in
`lowfive_attach.cpp` (or, checked and reported but not touched, the
env-route plugin or `lowfive_transport.cpp`'s own `DistVolSingleton`) ever
called `LowFive::DistMetadataVOL::set_intercomm()` at all, so `file_open()`'s
own lookup had nothing to find; and `fuse::split_hdf5_path()`'s own
unslashed dataset-path output, passed straight to `vol.set_memory()`,
never matched `LowFive::Object::fullname()`'s own always-slashed form that
`dataset_create()` compares against, so `indexed_datasets` stayed `0` and
`serve_all()` would have returned without serving anything even once a
remote open succeeded. `bruss_lowfive_attach()`'s distributed branch now
calls a new pure function, `BuildIntercommRegistrations()`
(`lowfive_attach_decision.hpp`), once per binding, and `vol.set_intercomm()`
once per registration; `ApplyBindings()` normalizes the dataset pattern
through a second new pure function, `NormalizeDatasetPattern()`, before
ever calling `vol.set_memory()`, for both the single-process and
distributed branches alike. Neither fix touches `split_hdf5_path()` itself
(out of scope; the single-process path never reaches LowFive through this
unslashed string in the first place -- it uses `VolSingleton`'s own
blanket `"*"`/`"*"` wildcards). KNOWN_ISSUES.md D39's own "Round 4 update"
account has the full citations, the cross-check of the two other call
sites, and -- stated explicitly there, repeated here because it matters --
this fix has never been run on a host either: the row's own distributed
RPC path, per-iteration steady state, and data correctness remain
UNMEASURED by any round of this track so far.

## Diagnosing a hang

**Round 5** -- round 4's own fix held (no more `file_open(): no intercomm
found`), and the host's own NEXT run of `lowfive.mpmd.3` got PAST the first
remote open and into a NEW 120 s hang, measured with both `brusselator_u_main`
and `brusselator_v_main` inside `file_open()`'s own `get_file_hierarchy` RPC
call (`rpc::recv` -> `MPI_Probe`), neither inside `serve_all()` at all --
U (first-server, the lower-`min_rank()` component) should have been serving,
not also consuming. KNOWN_ISSUES.md D39's own "Round 5" account has the full
four-hypothesis analysis (H1: the cold-start write's own view guard was
false; H2: `serve_all()` returned at once because `indexed_datasets==0`;
H3: the serve hook returned an action other than `kServeNow`, or never ran;
H4: `serve_all()` served the wrong intercomm index, or `intercomms`/
`serve_indices` ended up empty) with LowFive-source citations for each --
this section covers only the DIAGNOSTIC TRACE this round adds, not that
analysis.

**Two independent, OFF-by-default env variables**, both read once per
process (a cached bool/one `getenv()` call each -- "keep the trace code
out of the hot path when disabled," the lead's own instruction):

- **`FUSE_LOWFIVE_ATTACH_TRACE=1`** (exactly the character `1`; anything
  else, including unset, is disabled -- `lowfive_attach_decision.hpp`'s own
  `ParseTraceEnabledEnv()`) turns on this track's OWN trace lines, all
  prefixed `TRACE(...)`, written to stderr:
  - One line at attach time (`bruss_lowfive_attach()`): world rank,
    component, role (`first-server`/`first-reader`), the `distributed`
    flag, the registered `(file, dataset_pattern, intercomm_index)` list
    (`vol.set_intercomm()`'s own arguments), the `(file, dataset)` memory
    patterns (`vol.set_memory()`'s own arguments), the `serve_on_close`
    value actually set, and (bonus, not in the brief's own enumerated
    list but free to read and directly diagnostic for H4) `intercomms.
    size()`/`files.size()`.
  - One line before and after `bruss_lowfive_write()`/`bruss_lowfive_
    read()` (`shim.cpp`): file, dataset, `nvals`, whether `H5Fcreate`/
    `H5Fopen` succeeded, and (write side only) an `H5Lexists()` check on
    the dataset path BEFORE `H5Fclose` plus the VOL's own in-memory file
    count AFTER it (`bruss_lowfive_attach_file_count()`, the best
    available proxy for "was the dataset attached at all" -- nothing
    public exposes a per-dataset `is_memory` flag from this translation
    unit's own installed LowFive headers).
  - One line when `u_lowfive_write_export`'s/`u_lowfive_read_import`'s
    (`u_main.c`) or `v_lowfive_write_export`'s/`v_lowfive_read_import`'s
    (`v_main.f90`, via a new `bind(C)` subroutine, `bruss_lowfive_
    trace_skip()` -- Fortran has no printf-equivalent of its own that
    reaches stderr with this exact shape) own `window_samples > 0 &&
    data` guard is FALSE, i.e. the write/read was SKIPPED -- settles H1
    directly.
  - One line before and after each of the two places this process ever
    calls `serve_all()` (`bruss_lowfive_serve_after_write()`'s own
    `kServeNow` case, `bruss_lowfive_after_read_close()`'s own
    `kFlushDeferred` case), with monotonic elapsed seconds -- a
    near-zero elapsed line is H2's own signature; a line that never
    gets its own matching "exit" line is the hang itself, caught live.
- **`FUSE_LOWFIVE_LOG=<level>`** (`trace`/`debug`/`info`, LowFive's own
  `spdlog`-based logger, `LowFive::create_logger()`) turns on LowFive's
  OWN internal trace lines -- `"enter serve_all, files.size = N"` and
  `Index`'s own logging settle H2/H4 directly, from inside the library
  rather than this track's own call sites around it. Called once, at the
  very top of `bruss_lowfive_attach()`, guarded against the one way
  `create_logger()` can throw (a second call in the same process --
  unreachable today, since `bruss_lowfive_attach()` already refuses a
  second call of its own, but guarded anyway per the lead's own
  instruction to handle it, not assume it away).

**Where the trace lines land.** `test/parallel/brusselator_participants/
run_participants.sh`'s own `mpmd` branch redirects each segment's stdout
AND stderr, per rank, to `$SCRATCH/u_stdout.<rank>.log`/`v_stdout.<rank>.
log`/`coupler_stdout.<rank>.log` (`sh -c '... exec "$@" >
"${f}.${PMI_RANK:-0}.log" 2>&1'`) -- exactly where the lead's own brief
said to look. Both env variables reach every one of the three programs:
`run_participants.sh`'s own three `mpiexec` invocations (one for `mpmd`,
two for `accept`) pass none of MPICH's own `-env`/`-genv`/`-envnone`
flags, so MPICH's hydra process manager uses its documented DEFAULT
behavior of propagating the LAUNCHING shell's own environment to every
spawned rank; CTest's own `ENVIRONMENT` test property (`examples/
CMakeLists.txt`, the three vestigial `HDF5_VOL_CONNECTOR`/
`HDF5_PLUGIN_PATH`/`FUSE_LOWFIVE_BINDINGS` lines this page's own earlier
section already names) only ADDS those three variables to the test
process's environment -- it does not clear or replace the rest of it
(CTest's own documented behavior) -- so exporting both variables in the
shell that invokes `ctest` is sufficient; nothing in `examples/
CMakeLists.txt`'s own `ENVIRONMENT` property needs to change.

**Host command, exactly as the lead's own brief asked for a form that
survives ctest/mpiexec:**

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=trace \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

KNOWN_ISSUES.md D39's own "Round 5" account has the full table of "trace
shows X => hypothesis Hn" this round's own report builds for the lead.

**Round 6 -- the trace answered it.** The host set both env variables and
ran the row to its 60 s timeout; the trace showed H1/H3/H4 RULED OUT (the
write ran, U took `kServeNow`, one intercomm is registered) and H2's own
SYMPTOM confirmed, but not round 4's own guessed mechanism (already
corrected by round 5): LowFive's own log printed `"enter serve_all,
files.size = 0"`, and `serve_all()` returned in 52 microseconds.

**Root cause, read directly from LowFive's source.** `LowFive::
MetadataVOL::keep` (`include/lowfive/vol-metadata.hpp`) is a PUBLIC bool,
default `false`; `MetadataVOL::file_close()` erases and deletes a file's
own metadata whenever `!keep`. `DistMetadataVOL::file_close()`'s own
LOCAL-file branch calls `serve_all()` on its own ONLY when `serve_on_close`
is true -- this attach's own round-2 fix set it `false` specifically so
NOTHING auto-serves inside `H5Fclose()` -- and then falls through to the
SAME `MetadataVOL::file_close()` above regardless. With `serve_on_close
=false` and `keep` left at its own default `false` (never set by any
round before this one), `H5Fclose()` neither serves internally NOR keeps
the file: it is erased and deleted before this process's own explicit
`bruss_lowfive_serve_after_write()` hook -- the ONLY server this protocol
ever gives a file -- ever runs a line. LowFive therefore only works with
`serve_on_close=true`, or with `serve_on_close=false` AND `keep=true`; this
attach shipped the first half of the second option without the second
half. KNOWN_ISSUES.md D39's own "Round 6" account has the full citations
(including why `MetadataVOL::drop()` and `clear_files()` needed a guard
and a "never call this for one file" warning respectively, and what
`MetadataVOL::file_create()`'s own `files.emplace()` does on a duplicate
key if a served file is never dropped).

**The fix: `decide_vol_settings()`, a file lifetime, and a guarded drop.**
`lowfive_attach_decision.hpp`'s own `decide_vol_settings(distributed)`
is now the ONE place that decides `serve_on_close`/`keep` together (both
`true`/`false` pairs this attach could otherwise set independently are
checked against the invariant `serve_on_close==false => keep==true` via
`VolSettingsSatisfyInvariant()`). `bruss_lowfive_attach()`'s distributed
branch applies both; its own exported file name(s) (`FindOwnExportedFiles
()`, derived from the SAME binding table/`this_name` the attach already
holds) are dropped, one at a time, right after each is actually SERVED --
at the first-server's own `kServeNow`, and at the first-reader's own
post-close `kFlushDeferred` -- never merely written, and never the peer's
own file. The file lifetime invariant: a file exists in `files` from its
writer's own close to the end of its serve, and is dropped before that
component's own NEXT `H5Fcreate` of the same name -- without the drop,
`file_create()`'s own `files.emplace()` is a no-op on that duplicate key,
and a later `serve_all()` would silently serve the PREVIOUS period's own
stale data, never the fresh write. The single-process branch is
UNCHANGED (its own existing `vol.set_keep(true)` already matches `decide_
vol_settings(false).keep`; it never serves anyone, so nothing is ever
dropped there either, and `serve_on_close` does not apply to `LowFive::
MetadataVOL` at all).

**The corrected account of what round 4's slash fix did and did not
break.** Round 5 already corrected the claim that an unslashed `set_
memory()` pattern starves `indexed_datasets` -- `MetadataVOL::file_
create()`/`dataset_create()` add to `files`/the object tree
UNCONDITIONALLY, regardless of pattern match (round 5's own account,
unchanged). The slash fix itself is STILL needed, for the reason `is_
memory` actually matters: `MetadataVOL::dataset_create()`'s own `is_
memory` flag (computed from the SAME pattern match, stored on the new
`Dataset` object) governs whether `H5Dwrite`/`H5Dread` on that dataset
use the in-memory path at all, or fall through toward a real, un-
intercepted file -- an unslashed pattern would have meant every
Brusselator dataset's own read/write silently missed interception, a
DIFFERENT defect from the indexing one round 4 originally (and wrongly)
blamed for the symptom it was chasing.

**`dataset_exists_before_close` was wrong, not the write.** Round 5's own
`H5Lexists()`-based check reported `0` even though the write that
preceded it succeeded -- confirmed this round to be an UNRELIABLE proxy,
not evidence the dataset was missing: `H5Lexists()` walks a path segment
by segment (`LowFive::MetadataVOL::link_specific()`'s own `H5VL_LINK_
EXISTS` case, `Object::locate()` -> `search()`), but `dataset_create()`,
called here with no prior `H5Gcreate` for an intermediate group, attaches
the dataset as ONE FLAT child literally named `"grid/u_out"` -- `search()`
looks for a child named exactly `"grid"` first, finds none, and reports
not-found, even though the dataset is genuinely there (`is_memory`'s own
check, which compares the WHOLE string via `Object::fullname()` rather
than walking it segment by segment, is unaffected -- confirmed by round
4's own fix actually working). REMOVED, replaced with `bruss_lowfive_
attach_has_real_file()` (`LowFive::MetadataVOL::has_real_file()`, PUBLIC,
FILE-level not dataset-level -- the best available proxy, since nothing
public exposes a per-dataset check at all).

**Host command, unchanged in form, now with a settled root cause behind
it:**

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=trace \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

KNOWN_ISSUES.md D39's own "Round 6" account has the full host_pending
paragraph: what each rank's trace must print on success, in order, and
for each plausible next failure what the trace/stack would look like and
which of this round's own claims it would contradict.

**Round 7 -- the serve now sees the file; the row fails fast instead of
hanging.** With round 6's fix landed, the host's own row now runs to
completion in under a second and FAILS with an abort, not a 60 s
timeout: V's own `H5Dget_space()` on a dataset it opened on the PEER's
file (via `H5Fopen`/`H5Dopen2`, i.e. a REMOTE dataset) throws `LowFive::
MetadataError: Unexpected no class in dataspace = -1`, from HDF5's own
diagnostic chain `H5Scopy(): not a dataspace` / `H5Iinc_ref(): invalid
ID` immediately before it.

**Root cause, read from source, not guessed.** This shim's own `bruss_
lowfive_write()` calls `H5Dcreate2(fid, "/grid/u_out", ...)` with NO
prior `H5Gcreate2(fid, "/grid", ...)`. Round 6 already found this
flattens the dataset into ONE child of the file/root literally named
`"grid/u_out"` (`LowFive::Object::search()`, src/metadata/object.hpp,
ALWAYS splits on the first `/` before attempting any match, confirmed
again this round by re-reading the same function) -- round 6 found this
breaks `H5Lexists()` ONLY cosmetically, since the write-side `is_memory`
check compares the whole string via `fullname()` and is unaffected. This
round finds the SAME flat name ALSO breaks `MetadataVOL::dataset_open()`
(src/vol-metadata/dataset.cpp) -- and this one is NOT cosmetic: since
`search("grid/u_out")` never finds a child named exactly `"grid"`, the
lookup falls through to `result->mdata_obj = parent->add_child(new
DummyDataset(name));` -- a placeholder class (src/metadata/dummy.hpp)
that is NOT a subclass of `Dataset` and whose own `space` member is
default-constructed (`id == 0`). `DistMetadataVOL::dataset_open()`'s own
promotion to an RPC-backed `RemoteDataset` (`dynamic_cast<Dataset*>
(mdata_obj) && RemoteObject::query(mdata_obj)`) never fires on a
`DummyDataset` (the dynamic_cast is correctly null between these two
unrelated classes) -- so the placeholder's own, never-populated `space`
is what `MetadataVOL::dataset_get()`'s GET_SPACE branch later reads:
`H5Scopy(0)` fails, returning LowFive's own documented sentinel `-1`;
the immediately following trace statement constructs `Dataspace(-1)` as
a LOGGER ARGUMENT (evaluated regardless of the active log level --
C++'s own eager-argument-evaluation rule, not a LowFive choice), whose
own ctor throws on `id != 0` with no valid HDF5 class -- the exact
measured message, down to the literal `-1`.

**Ranked against the alternatives, each read from source, not
assumed.** (a) `RemoteDataset`'s own ctor (src/metadata/remote.hpp) DOES
fetch `space` via a real RPC call named `"metadata"` -- this machinery
is real, and LowFive's own upstream `examples/prod-con-multidata/` demo
proves it works (producer: `H5Gcreate(file, "/group1", ...)` then
`H5Dcreate2(group, "grid", ...)`, a BARE name relative to the GROUP
handle; consumer: calls `H5Dget_space()` on the resulting remote dataset
successfully) -- just never reached in this row, since `make_remote_
dataset()` is never called at all. (b) a hid lifetime bug in
`deserialize()`/`make_remote_dataset()`: traced `LowFive::Hid`'s own
ref-counted `H5Iinc_ref`/`H5Idec_ref` discipline (src/metadata/hid.hpp)
through both functions and found it correctly balanced for the code
that WOULD run if that path were reached -- excluded as this crash's
cause, though unverified on a live host run, since that path has never
executed in this track. (c) a two-HDF5-image/version mismatch: excluded
-- the symptom (a correctly-formatted HDF5 diagnostic plus a C++
exception whose exact message and numeric value this round's own
account predicts from LowFive's own code) is what a logic bug on a
bogus id looks like, not an ABI/image mismatch, which this track's own
configure-time probe already guards against elsewhere and which this
round's own tier-1 configure re-confirmed PASS.

**The fix, in this shim, not in LowFive.** `bruss_lowfive_write()` now
calls `H5Gcreate2()` for the dataset's own parent group path before
`H5Dcreate2()`, derived by a new pure function, `SplitDatasetGroupPath()`
(`lowfive_attach_decision.hpp`). `LowFive::MetadataVOL::group_create()`
(src/vol-metadata/group.cpp) already walks its own unconsumed `search()`
remainder segment by segment, creating one `Group` per segment --
unlike `dataset_create()`, which never does -- so one `H5Gcreate2()`
call is sufficient even for a multi-segment parent path. `Object::
fullname()` (used by the write-side `is_memory` check round 4 fixed)
walks `parent` pointers and re-concatenates with `/`, so nesting the
dataset under a real `Group` named `"grid"` reproduces the IDENTICAL
`"/grid/u_out"` string that check already matches against -- no
regression to round 4's own fix.

**The width check (shim.cpp, the `H5Dget_space`/`H5Sget_simple_extent_
dims` pair guarding against a silently short/overrunning `H5Dread`) is
UNCHANGED in its own logic, and is NOT replaced.** `H5Dget_space` on a
remote dataset is a SUPPORTED, WORKING LowFive mechanism once the
dataset is properly nested under a real group (confirmed against
LowFive's own upstream example above) -- there is no upstream
limitation to work around here, so none of the fallback candidates the
lead asked evaluated are adopted: a "local view" does not apply (a
reader opening a peer's file has no local copy to query); a second VOL
path for `H5Sget_simple_extent_dims` does not help (the failure is
inside `H5Dget_space` itself, before dims extraction is even reached);
`H5Dget_storage_size()` is not merely redundant but UNSUPPORTED on a
memory-backed dataset at all in this LowFive revision (`MetadataVOL::
dataset_optional()`'s own warning: "not implemented in metadata yet");
an in-band size attribute or dataset would duplicate information a
WORKING `H5Dget_space` call already supplies more directly. Whether the
graph's own validation could make a runtime check unnecessary: `fuse::
validate()` (`include/fuse/model/validate.hpp`) only enforces a field's
own per-point COMPONENT count (scalar vs. vector width) between export
and import, never the TOTAL runtime array length -- that length is
`out_view.count * out_view.components` in `u_main.c`, with `count`
coming from each process's OWN domain decomposition, built
independently by each participant and never compared by the graph
itself. For THIS row (one rank per component, mirrored grids) the
counts happen to agree, but nothing in `validate()` GUARANTEES that for
a general decomposition -- the runtime check stays worth having, exactly
as it already was.

**Next calls in the read path, confirmed or excluded by source, not
exercised by any host run yet.** `H5Dread(did, H5T_NATIVE_DOUBLE, H5S_
ALL, H5S_ALL, H5P_DEFAULT, data)` -- CONFIRMED hazard: `DistMetadataVOL::
dataset_read()`'s own RemoteDataset branch (src/vol-dist-metadata.cpp)
carries an explicit, unresolved upstream comment, `// TODO: support
H5S_ALL, need to get file_space_id, mem_space_id with H5Dget_space`, and
passes whatever it is given straight into `Dataspace(file_space_id)`/
`Dataspace(mem_space_id)`; `H5S_ALL` is literally `0` in this HDF5 build
(`H5Spublic.h`), which hits `Dataspace`'s own early-return (`if (id ==
0) return;`) leaving `min`/`max` default-constructed/empty --
`RemoteDataset::query()`'s own bounding box would then be built from
those empty vectors, not the dataset's real extent. FIXED in this shim,
not in LowFive: `H5Dread` now reuses the SAME `space` handle the width
check already validated, for BOTH the file and memory dataspace
arguments, instead of `H5S_ALL` -- a standard HDF5 idiom for a full,
contiguous, correctly-sized read, and one that never exercises LowFive's
own cited TODO. `H5Dclose`/`H5Fclose` of the `RemoteFile`, and the
producer's own serve-loop return: CONFIRMED and REFINED (not changed)
from source -- `DistMetadataVOL::file_close()`'s own RemoteFile branch
does three things (an RPC `"file_close"` call, a COLLECTIVE `MPI_Barrier
(local)` across every rank of the consumer's own local communicator,
then -- only on the consumer's own root rank -- `finish(0)`);
`Index::serve()` (src/vol-dist/index.cpp) loops processing RPCs until
`finish` has been received on EVERY one of its own registered intercomm
servers, then returns. The model check's own "peer's open AND close
release the serve" wording already covers this; `lowfive_attach_model_
check.hpp`'s own "ROUND 7 CONFIRMATION" comment has the full citations,
appended without changing the two bullets it confirms. The WRITE side
of period 2 (create after the round-6 drop): the mechanics checked out
from source (the guarded drop runs synchronously, in the same call
stack, right after the serve that needs it, before any later create can
race it) but WHETHER the cross-component SCHEDULING (which periods's
write/read/drop interleave with which, enforced by `fuse::WindowCoupler`,
out of this track's own scope) actually reaches period 2 in the right
order has NEVER been exercised by any host run across all seven rounds
-- listed in `host_pending`, not claimed.

**Host command, unchanged in form, now past TWO fixed defects and
testing a third hazard it had never reached before:**

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

(`info`, not `trace`, per the lead's own instruction this round -- the
trace log is large; raise it back to `trace` only if `info` does not
separate the next failure's own hypotheses). KNOWN_ISSUES.md D39's own
"Round 7" account has the full host_pending paragraph.

**Round 8 -- the measured steady-state order is not "write,read every
period" on both sides; the fix is a policy correction plus a two-part
order check, not a dynamic exchange.** With round 7's fix landed, the
host's own row now round-trips a FULL period through LowFive for the
first time on both ranks (every `H5Dread rc=0`, correct extents,
`files_in_memory` back to 0 after each read-close) but still times out
at 120 s, one write into period 1: U (first-server) blocks in its own
`read(v#1)`, and V (first-reader) parks LAST, at the coupler's own
`MPI_Allreduce` sync, having just deferred its own `v#1` serve.
KNOWN_ISSUES.md D39's own "Round 8" account has the host's own full
trace, both ranks, verbatim.

**Root cause.** Round 2/3's own model assumed both sides run
write-then-read every period, with a first-reader's deferred serve
flushed after its own NEXT read's close. The measured orders differ by
role: U writes then reads every period, unchanged from its own cold
start; V writes then reads ONLY at cold start, then reads-then-writes
from period 1 onward. Under the old "first-reader always defers"
policy this is circular -- each side's own next step needs the other's
release first.

**What determines the steady-state order:** channel DECLARATION order
(`CouplingGraph::add_channel()` assigns each `ChannelId` strictly in
call order), combined with `Runner::Impl::deliver_external()`'s own
per-channel "exporter suspends before importer" rule and
`lower_channel()`'s own `ExternalDelivery` ordering-edge placement
(the same slot `Transport` would occupy) -- NOT component declaration
order, rank sets, the Jacobi iteration kind, or the acceleration kind
(the first three happen to coincide with channel order in this graph
but are not what drives it; the last two do not change the mechanism).
Not fully settled: whether `ExternalDelivery` fires once per coupling
window or once per Jacobi sweep -- the host's own measured counts (two
write-serve-read cycles across two periods) are consistent with
once-per-period, but this round did not trace the sweep-count machinery
far enough to prove it analytically. Full derivation, with every
citation, is in `lowfive_attach_decision.hpp`'s own "ROUND 8" header
comment and in KNOWN_ISSUES.md D39's own "Round 8" account.

**The fix:** `decide_serve_action(role, completed_any_read)` defers
ONLY a first-reader's cold-start write; every later write, either role,
serves immediately. `ServeState` gained `completed_any_read`.

**The order check, two parts, deliberately NOT a dynamic runtime
exchange.** (a) STATIC, at attach time: `CheckGraphOrderValidated()`
refuses (`E-LOWFIVE-ATTACH-ORDER-UNVALIDATED`, naming the first
differing property) unless the loaded graph's own channel-order
signature exactly matches the ONE measured, validated configuration --
this limits the fix to validated configurations only, stated plainly
here and in KNOWN_ISSUES.md, not silently assumed to generalize. (b)
LOCAL, per event: `CheckAndAdvanceOrder()` runs before every real
LowFive hook's own blocking call and refuses
(`E-LOWFIVE-ATTACH-ORDER-DEVIATION`, naming expected vs. observed event,
period, and slot) before that call, never after. (c) Nothing sound was
found for detecting the peer's own order at runtime instead: each side
only learns the peer's order from the OUTCOME of a serve/read exchange,
which is the exact mechanism already at risk of deadlock -- an exchange
performed after the first problematic write cannot prevent a mutual
serve already begun (the measured hang IS such a state), and the
exchange call itself could deadlock if one side is already blocked
inside `H5Fopen`. No sound pre-serve exchange is proposed; this is an
open question, not a NOT-NOW decision made for convenience.

**The model check:** event sequences (write/read/sync) are now DATA,
not inline control flow, with a third (coupler) actor added for the
collective sync. Four scenarios are checked against `RunSimulationRound8()`'s
own 3-actor search: the measured order under the new policy completes
for 1, 2, and 60 periods; the measured order under the OLD (round 2/3)
policy reproduces the measured hang (a regression test); both
"write-then-read every period" and both "read-then-write every period"
(the two strawman shapes round 2/3 implicitly assumed) each get stuck.
A soundness link (`static_check_accepts(order) == model_completes(order)`)
is asserted for every catalogued order -- true only for the measured
order, because the channel-order mechanism can only ever produce that
one shape; the two strawmen have no corresponding real graph at all.
KNOWN_ISSUES.md D39's own "Round 8" account has the full test inventory
and the three new mutation rows.

**Host command, unchanged in form, now past the first full period but
never past the first write into period 1:**

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

KNOWN_ISSUES.md D39's own "Round 8" account has the full host_pending
paragraph: the expected per-event trace line shape (event index, period
index, policy/reason string, order-check verdict) on both ranks, and
for each plausible next failure what the trace would look like and
which of this round's own claims it would contradict.

**Round 9 -- the order check and policy both hold on the host; a separate, measured exit-time crash is fixed for
all three processes; a naming correction.** The host confirms round 8's own fix directly:
`order-check-verdict=accepted` on both ranks, zero `E-LOWFIVE-ATTACH-ORDER-DEVIATION` refusals across 364 (U) and 365 (V)
per-rank order-check calls, and both coupling windows complete through LowFive (122 U writes/121 reads, 121 V
writes/122 reads, every `H5Dread rc=0`). The row still times out, but on a DIFFERENT track's own collective (the
coupler's own base-communicator `MPI_Allreduce`, never joined by U/V after the last delivery) -- not a defect in
anything this track owns.

**The coupler exit crash -- round 9's own root cause claim is RETRACTED by round 10's own measurement.**
`brusselator_coupler_main` aborts at process exit, AFTER `MPI_Finalize`, with `libc++abi: terminating due to
uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument`. Round 9 attributed this
to `examples/CMakeLists.txt`'s own ctest `ENVIRONMENT` property setting `HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH`
process-wide for the whole MPMD launch, and fixed it by having the coupler `unsetenv()` those three variables
before `MPI_Init`. Round 10's own host measurement shows the coupler's own trace line DID print (confirming the
`unsetenv()` ran) and the SAME abort still happened, byte-identical, on two different MOAB installs -- DISPROVING
environment inheritance as a complete explanation. Round 10's own reading excludes, with citations, every
FUSE-side path that could construct a LowFive VOL object on the coupler's own rank (`RunnerConfig::transport` is
always `"mpi"` for this graph, never `"lowfive"`, so `LowFiveTransport`/`VolSingleton`/`DistVolSingleton` are
never reached; `lowfive_attach.cpp` is not compiled into the coupler at all; the graph's own fieldless "lowfive"
marker component is skipped by the identical `ComponentKind::Coupler` exemption the real coupler component gets)
and confirms (but does not by itself explain) that `liblowfive`/`liblowfive-dist`/MOAB/TempestRemap are linked
into EVERY FUSE executable unconditionally, by build-time design, independent of role. The remaining, unexcluded
candidate -- a pre-`main()` static initialiser in a linked library, running before `unsetenv()` has a chance to
execute -- needs a host experiment to settle (an `lldb` session on the coupler's own MPMD segment, breakpoints on
`main`/`H5open`/`H5VLregister_connector`/`H5VLregister_connector_by_value`); KNOWN_ISSUES.md D39's own "Round 10"
account has the full exclusion list, citations, and the experiment design in full. The `unsetenv()` fix is KEPT
(still sound hygiene for a process with no deliberate use for the connector) but is no longer claimed to be THE
fix for this crash. U's and V's own fix (an explicit, synchronous `H5close()` in `bruss_lowfive_finalize()`,
before their own `MPI_Finalize()`) is UNAFFECTED by this retraction -- it addresses a different, confirmed
mechanism (their own deliberate attach via `lowfive_attach.cpp`), not the coupler's own mystery registration.

**Naming correction (no logic change).** The `period=N` trace field and two pure-function names (`ExpectedOrder
ForPeriod`, `OrderTracker::period_index`) were a misnomer: the unit they track is one write-then-read exchange,
not a coupling window -- the host measured 61 exchanges per window (122 U writes for a `--periods 2` run) ON
COMMIT 7aec9c8, BEFORE the owner's Kokkos+CUDA device-branch merge, not one. Renamed to `exchange=`/
`ExpectedOrderForExchange`/`exchange_index` throughout this track's own code, tests, and docs; "period" is
reserved for the Brusselator's own `--periods` CLI flag and the outer coupling-window loop -- this renaming
itself is unaffected by anything below. This also closed round 8's own open question using THAT figure:
`ExternalDelivery` fires once per Jacobi sweep, not once per window (~61 deliveries per window on 7aec9c8,
MEASURED, not derived). **Round 10 correction:** the SAME row, re-measured on HEAD 0dc8c70 (after both the
device-branch merge and the Collectives track's own fix for the row's own unrequested-period hang, which
completes the row for the first time), shows a DIFFERENT figure: 12 sweeps per period, 25 writes/25 reads per
rank for `--periods 2`, SYMMETRIC between U and V (a mid-fix measurement on b98777d, taken while the row still
hung on an unrequested third period, saw an asymmetric 26 U writes/25 reads -- superseded by this completed-row
figure, not an independent data point). The CAUSE of the change from 61 to 12 is still NOT fully established,
though the Collectives track's own D57 entry narrows it: running the real species physics gives
`ref_iterations = 12` on both components, refuting the earlier "61 is exactly `kMaxIterations + 1`" hypothesis
outright, and traces 61 to a commit predating that fix, confirming it as a genuine 7aec9c8 measurement. D57
also records, as an arithmetic observation only, not a finding, that 122 is also consistent with 12
sweeps/period over TEN periods rather than two -- not established, only consistent; this file does not claim a
cause for the 61-to-12 change. The exact relationship between either sweep count and the Collectives track's
own `max_iterations`/convergence machinery remains that track's own question in both cases.

**Host command, unchanged in form, now testing the exit-hygiene fix (never previously reached, since no process
in this row had ever gotten far enough to call its own exit() cleanly before):**

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

KNOWN_ISSUES.md D39's own "Round 9" account has the full host_pending paragraph: what each of the three
processes must print and do at exit, and for each plausible next failure what it would look like and which of
this round's own claims it would contradict.

**Round 10 -- round 9's own exit-crash fix is confirmed on the host to have RUN but not to have stopped the**
**crash; a read-first investigation excludes every FUSE-side registration path with citations; a host**
**experiment is designed, not run; this crash is now the ONLY thing between the row and a pass.** Same host,
same row, measured mid-round on HEAD b98777d (after the owner's Kokkos+CUDA device-branch merge;
`/opt/moab-tempest` was removed, MOAB now lives at `/opt/moab/cmake-all` with a shared `libhdf5.310` rather than
an embedded copy -- measured identical on both). The coupler's own trace line printed (confirming the
`unsetenv()` call ran), and the SAME abort still happened, byte-identical trace, on both MOAB installs. Three
FUSE-side mechanisms were excluded by reading, not by a host run: `RunnerConfig::transport` is assigned exactly
once in the tree and is always `"mpi"` for this graph, so `LowFiveTransport` is never constructed on any rank;
the graph's own fieldless "lowfive" marker component is skipped by the identical `ComponentKind::Coupler` check
the real coupler component is; `lowfive_attach.cpp` is not compiled into the coupler's own binary at all.
`liblowfive`/MOAB/TempestRemap ARE linked into every FUSE executable by build-time design (explaining
`otool -L`, not causation). No eager/static-constructor trigger was found in LowFive's own granted source. The
remaining, unexcluded candidate -- a pre-`main()` static initialiser inside MOAB's or TempestRemap's own
compiled binary, unreadable from this track -- needs a host `lldb` experiment to settle (breakpoints on
`main`/`H5open`/`H5VLregister_connector`/`H5VLregister_connector_by_value`, each outcome separating a distinct
candidate); KNOWN_ISSUES.md D39's own "Round 10" account has the full exclusion list, citations, and the
four-outcome experiment design in full. **LATER UPDATE, same round, HEAD now 0dc8c70:** the Collectives track's
own fix for the row's own unrequested-period hang landed, and the row NO LONGER HANGS -- it completes in
1.87 s, with U/V's own data path reproducing the transport-only twin exactly (byte-identical logs,
`brusselator_check --check: OK`) for the first time. This exit crash is now the ONLY remaining failure in the
row. U's and V's own exit codes were NOT observed in this measurement (mpiexec killed them after the coupler's
own BAD TERMINATION was reported); task B's own experiment is extended to capture them once the coupler's own
fix lands. No FUSE-code change was made this round: reading
did not establish a definite defect in FUSE's own code, only in a dependency this track cannot patch. Round
9's `unsetenv()` and U/V's `H5close()` calls are both KEPT, unchanged.

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
