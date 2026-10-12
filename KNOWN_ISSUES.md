# Known issues

Open items, split by **who has to act**. Closed entries are kept, struck
through, because the reason a thing was an issue is worth more than a shorter
file. Every entry names the file, the
function and the code, and quotes prior code verbatim where a revert is
involved — never a commit hash, because hashes do not survive a rebase.

The last section lists things that look like defects and are not, so nobody
spends time on them.

---

## A. Blocked on the repository owner

### A1. ~~There is no remote yet~~ — a remote exists and CI has run (2026-09-27); MPI legs still absent

**Update 2026-09-27.** `origin` is `github.com:vijaysm/fuse.git` and the
workflow has executed: run 36124149902 failed, and `d42caa6` fixed what it
found — exactly the class of defect this entry predicted (fetch-path flags,
a `CMAKE_DISABLE_FIND_PACKAGE_Kokkos` that broke ArborX's own REQUIRED
lookup, a missing `find_dependency(Eigen3)` in the installed config, the
`throws` label missing from the omp legs, Eigen's own tests leaking into
ctest). Whether the run after `d42caa6` went green is not recorded here; the
sandbox cannot see GitHub.

**What the five legs still do not cover:** none runs MPI, so every
`parallel.*`, `layout.*` and MOAB test remains a host-only outcome, and the
mutation rows marked `requires = "mpi"` are never assessed by CI. Adding an
MPI leg (MPICH from apt, `-DFUSE_ENABLE_MPI=ON`, `ctest -L mpi` at 2–4
ranks) is the one CI change that would move verification off your machine.
The text below is the original entry, kept for its reasoning.


**Your call, recorded 2026-09-17: the remote waits until all phases are
complete.** This is not an oversight and needs no action. What it costs is
worth writing down so it is not forgotten at the end:

- **The workflow file itself is unexercised.** `.github/workflows/ci.yml`
  defines five legs (`gates`, `docs`, `upstream-soma`, `tier0`,
  `tier0-no-exceptions`) and none has ever run, so its YAML wiring — job
  names, matrix, cache keys, action versions — is of unknown correctness. A
  workflow that has never run is not evidence of anything.
- **Mitigation:** the legs' *recipes* are run by hand here, so the configure
  lines and commands are exercised even though the wiring is not. Doing that for
  the OpenMP leg on 2026-09-17 found two real defects in the fetch path — see
  the note that follows this entry. When the remote appears, expect to debug the
  YAML and the Eigen fetch, not the rest of the build.

**What to do.** Nothing now. At the end, push and expect one round of workflow
fixes.

### Note to A1 and A5 — the OpenMP leg's recipe, run by hand 2026-09-17

Since the workflow cannot run, its OpenMP leg's configure line was executed
directly. **It found two defects, and would have failed on both.** That is the
whole argument for A1: a workflow that has never run is not evidence.

1. **Configure error on the fetch path.** `fuse_drop_missing_include_dirs()`
   called `set_target_properties` on Kokkos' targets. An *installed* Kokkos
   exports IMPORTED targets, which are settable; a Kokkos built from source
   exposes ALIASES, and `set_target_properties` on an alias is a hard CMake
   error. Every build ever done here used the installed Kokkos, so the fault
   was invisible. The function now skips aliases, which loses nothing: a target
   built in this build tree cannot have a stale include directory.

2. **`Config.KokkosHeadersMatchTheResolvedLibrary` failed.**
   `FUSE_KOKKOS_VERSION_EXPECTED` was the empty string on the fetch path,
   because the version was only recorded when `find_package` succeeded — so
   the test failed parsing `""` instead of comparing anything.
   `FetchContent_MakeAvailable()` is an `add_subdirectory()`, so the
   dependency's own `set(Kokkos_VERSION …)` is invisible to us and is not
   cached; it is now read out of the subproject's directory scope, which
   reports what was *built* rather than the tag we asked for.

**Measured after both fixes** — the same suite, two configurations:

| Configuration | Kokkos | Spaces reported | Result |
|---|---|---|---|
| installed `/opt/kokkos` | 4.6.0 | `Threads Serial` (default Threads) | 65/65 |
| fetched, pinned `4.4.01` | 4.4.1 | `OpenMP Serial` (default OpenMP) | 65/65 |
| installed `/opt/kokkos`, **after the 2026-09-17 upgrade** | 5.1.0 | `OpenMP Serial` (default OpenMP) | 65/65 |

The upgrade to 5.1.0 turned the installed device list from `THREADS;SERIAL` into
`OPENMP;SERIAL`, so OpenMP is now covered by the ordinary build and **Threads is
no longer enabled anywhere** — that space is currently unexercised, and a fetch
build with `-DKokkos_ENABLE_THREADS=ON` is what would restore it if wanted.

So the typed kernel tests are now verified on **all three host-parallel spaces**
— Threads, OpenMP and Serial — which is every one that exists without a device.

**One stated deviation.** The leg also fetches Eigen; this sandbox cannot reach
`gitlab.com`, so the run used `-DFUSE_LINALG_BACKEND=builtin` instead. The
OpenMP result above is unaffected — the facade's builtin factorisation is the
oracle the Eigen backend is tested against — but **the leg's Eigen fetch path
remains unexercised** and is one more thing to expect to debug when the remote
appears.

### ~~A2. Four parallel test legs cannot run in this sandbox~~ — CLOSED 2026-09-17

Still true of the sandbox, and no longer an open item: the host run covers it.
**855 tests, 0 failures**, including all 16 `parallel.*` entries
(`runner_transport`, `comm_graph`, `mpi_transport`, `epoch_reconcile` at
2/3/4/5 ranks), the 4 `example` entries, the 2 `compile_fail` entries and the
`field` and `num` additions. Both transport mutations were **killed** under
`FUSE_MUTATE_MPI=1`, so the peer-plan numbering check and the local-copy pass
are each load-bearing rather than merely present.

Re-run after any change to the transport, the runner or the rank-matching
graph — those are the three areas whose failure mode is a hang, which no
sandbox test can see:

```sh
cmake --build build-tier1 -j12 && (cd build-tier1 && ctest --output-on-failure)
FUSE_MUTATE_MPI=1 python3 tools/mutate.py --build-dir build-tier1 --area transport --write-log
```

### ~~A3. The Python and Fortran binding names are still proposals~~ — CLOSED 2026-09-17

Ruled: **`cfuse`** (C header and flat-symbol shim only), **`ffuse`** (Fortran
module), **`pfuse`** (Python package). `pip index versions pfuse` reports no
matching distribution, checked before the name landed. The collisions that
shaped the choice — and why a Julia binding must *not* be `jfuse` — are
recorded in `docs/dev/porting.md`, which now has no open decisions.

### A4. Which existing simulation code should be the first one we couple?

**Not needed until S6.** Listed here so it is not discovered at the last
minute, since the answer may involve getting a code built, which takes longer
than the coupling does.

**The plain version.** Everything coupled so far has been written by us, in
this repository, against our own interfaces. That proves the coupler works on
components that were designed for it. The claim we actually want to make is
stronger: *FUSE can couple a simulation code it has never met, without
modifying that code.* Demonstrating that needs one real program — someone
else's, already written, that we do not touch.

What such a program has to be, in ordinary terms:

- a **separate executable**, built by its own build system, not linked into our
  test suite;
- it **runs its own time loop** and calls FUSE when it wants to exchange, rather
  than being driven by us;
- it **hands us its existing arrays** — no rewriting its data structures;
- it can run **on its own set of processes**, alongside another code on
  different processes.

A Fortran or C++ heat, fluid, structural or transport solver you already have
is exactly the right shape. It does not need to be large; a small code that is
genuinely external teaches more here than a big one we adapt.

**Your answer, 2026-09-25: your own multirate Brusselator** (`~/Code/AI/Toy`,
written independently of FUSE; two species on non-nested periodic grids,
waveform-iterated windows, area-overlap conservative transfer, Eigen only).
Ported as `examples/brusselator` (`24a3a70`) onto the native quad mesh, the
supermesh remap and `WindowCoupler`, with the reaction on Kokkos —
`docs/user/brusselator.md` records what was kept verbatim, the one necessary
departure (where the shared reaction rate is sampled), and every measurement.
Weights match the Toy's separable overlap oracle to `1e-13`; the discrete
budget holds to round-off (`7.6e-13` per window over 575 windows, against a
first version at `1.7e-3` whose U consumed a different amount of the reaction
than V received — now a mutation row that must fail `--check`); the cross-code
difference at `--end 0.1` shrinks monotonically under 1x–8x refinement with
measured orders 0.73, 0.76, 0.86, and the two codes' Richardson limits agree
to `1.4e-6`.

Against the four properties above this was honest about what it showed: it
hands us its existing arrays (Eigen vectors, wrapped, not copied) — that one
held from the start. It was not a separate executable, did not run its own
time loop, and did not run on its own processes.

**ANSWERED, S6 phase 3 (Conformance): all four properties now demonstrated
on the SAME Brusselator, through the C ABI — `u_main.c`/`v_main.f90`.
One is measured in this sandbox; three build clean here and are host-only
(MPI cannot start in this sandbox, C1).**

1. **Separate executable, its own build system, not linked into our test
   suite.** `examples/brusselator/participants/{u_main.c,v_main.f90}`
   (`examples/CMakeLists.txt`'s own `brusselator_u_main`/`brusselator_v_main`
   targets) — genuinely separate programs, each with its own `main`, linking
   only `fuse::abi`/`fuse::abi-fortran` (never `fuse::runtime` or any other
   internal library), launched by
   `test/parallel/brusselator_participants/run_participants.sh` as
   independent `mpiexec` segments. **Built, host-only**: confirmed clean at
   tier 1 (`-DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_FORTRAN=ON`) this phase,
   including under `-DFUSE_ENABLE_LOWFIVE=ON` too; the executables
   themselves have not been RUN (ctest rows below).
2. **Runs its own time loop, calls FUSE when it wants to exchange.**
   `examples/brusselator/participants/{u_loop.c,v_loop.f90}` — the SAME
   loop-step functions `u_main.c`/`v_main.f90` call, structured exactly as an
   unmodified external code's own driver would be: `fuse_begin_step()`,
   advance local physics, `fuse_end_step()`, repeat, with no FUSE call in
   between. **Measured-tier-0**: `verification.brusselator_participants`
   and `verification.brusselator_lowfive_harness` drive these SAME loop
   functions directly, single-process, both waveform iterations, bit for
   bit against Unit 1's own in-process `Runner` across the example's full
   150-window span — the loop structure itself, independent of process
   placement, is proven correct, not merely compiled.
3. **Runs on its own set of processes, alongside another code on
   different processes.** `verification.brusselator.participants.
   {mpmd,accept}.{3,5}` (`examples/CMakeLists.txt`) — U, V, and the coupler
   as genuinely separate `mpiexec` segments (`mpmd`) or separate
   executables merging through `MPI_Comm_accept`/`MPI_Comm_connect`
   (`accept`, KNOWN_ISSUES.md D18's own routing). **Built, host-only**:
   all four executables build clean at tier 1 this phase; none of these
   four rows has been run (MPI cannot start here, C1).
4. **Hands us its existing arrays — and, as the fourth property's own
   strongest form, does not even need to hand FUSE the bytes, only tell it
   when to move them.** `--transport lowfive`:
   `verification.brusselator.participants.lowfive.{mpmd,accept}.3` — U and
   V exchange through a real `H5Dwrite`/`H5Dread` pair LowFive intercepts,
   FUSE itself touching no byte of the transferred field (`docs/user/
   lowfive.md`'s own contract). **Built, host-only, with an ADDITIONAL host
   dependency** beyond the plain rows above: the tier-1+LowFive build is
   confirmed clean this phase, but the two rows also need the host's own
   `/opt/lowfive` rebuilt against fmt 9+ before `fuse-lowfive-vol` can even
   `dlopen` (KNOWN_ISSUES.md D14's own consolidated "what remains" list
   names this first).

**A4 is answered — every property has a named demonstration — with one
honest qualification carried forward, not newly found here: all three
multi-process properties (2, 3, 4 in the ORIGINAL ordering; items 1, 3, 4
above) are confirmed only as far as this sandbox can confirm anything
needing real MPI — a clean BUILD, never a RUN. The single-process property
(own time loop) is the one with an actual bit-for-bit measurement behind
it. Closing A4 all the way to "measured" needs a host; the exact rows are
named above and, for the LowFive pair, carry their own further dependency,
tracked once, in D14, not repeated here.**

### A5. A device number waits on a GPU machine — your sequencing

**Your call, recorded 2026-09-17: get host threading fully working first, then
retry on a machine with a GPU.** That is the right order, and the host-threading
half is now measured rather than asserted: 65/65 on `Threads Serial` from the
installed Kokkos and 65/65 on `OpenMP Serial` from a fetched one, both recorded
in the note under A1. Those are all three host-parallel spaces.

Nothing in the code has to change for a GPU. The typed kernel tests instantiate
on whatever spaces the build enables, so on a CUDA- or HIP-enabled Kokkos they
pick up the device space automatically and the startup banner will name it. What
that run would add is a *measured* number; parameterisation is a precondition
for device performance, not evidence of it, and the headers say so.

**What to do.** Nothing until you have the machine. Then configure against a
GPU-enabled Kokkos and run the suite; the banner tells you whether the device
space was actually picked up.

---

## B. Mine, and already scheduled — please do not act on these

### ~~B1. The MOAB-enabled build configuration does not build: the MOAB backend is unported~~ — CLOSED 2026-09-18, superseded by C1

Stale and false as of the S5.2 port: `include/fuse/backend/moab_backend.hpp`
and `src/backend/moab_backend.cpp` both exist and are wired into
`fuse-coupler-backend`, so the compile error quoted below cannot reproduce at
HEAD (REVIEW.md R-23). See C1 below for the current status of the MOAB-enabled build configuration — it
builds; what is still open is a host run of its test suite.

<details><summary>Original text, kept for history</summary>

`-DFUSE_ENABLE_MOAB=ON` now configures and resolves MOAB 5.6.0 correctly, then
fails to compile:

```
src/backend/intx_weights.cpp:49:10: fatal error: 'fuse/backend/moab_backend.hpp' file not found
```

This is **not** a build defect. `upstream/soma/src/mesh/moab_backend.cpp` is
2,111 lines and has not been ported yet; porting it is the mesh-backend step of
S5. The conservative remap path that does not need MOAB works today and is the
differential oracle for this one when it lands.

</details>

### ~~B1a. The no-exceptions build aborts where the library catches its own rejection~~ — CLOSED 2026-09-17

Found by running the `tier0-no-exceptions` leg's recipe (11 of 57 aborting),
fixed in `9975cb3` and `555c62d`. The leg now passes **48/48**, and the normal
build stays at 65/65.

**What it was.** `FUSE_REFUSE` throws when exceptions are on and
reports-then-aborts when they are off, so any place the library threw and
caught *its own* rejection to build a value died before the catch ran.

**The fix, applied in two places.** Each operation now has ONE implementation
that reports, with the throwing spelling layered on top — so the two cannot
disagree about when something is an error, and the reporting path never
depends on exceptions:

- `Rational::try_add/try_sub/try_mul/try_div`, `try_gcd_rational`,
  `try_lcm_rational`. The four operators and `gcd_rational`/`lcm_rational` are
  now thin wrappers that refuse when those report nothing.
  `ClockSet::checked_common_period()` and the two unit-compatibility sites use
  the reporting forms and contain no `try`/`catch`.
- `try_plan_buffers()` returns `Result<BufferPlan>`; `plan_buffers()` is the
  throwing wrapper for direct callers; `schedule()` reads the diagnostic
  instead of catching it. Its old comment already said the throw belonged in
  diagnostics.

**A correction to what this entry first claimed.** It said the discriminator
was the *caught type* — that only `catch (const Error&)` sites were internal
control flow and everything catching `std::exception` was a containment
boundary. That was wrong, and running the leg proved it: `scheduler.cpp`
caught `std::exception` and was internal control flow, because **we** were the
thrower. The discriminator is whether the throw originates inside this library
on a path that must produce a value.

**The nine remaining targets were a missing label, as expected**, and are now
marked `THROWS` in `test/CMakeLists.txt`. Recorded there: the label is
target-granular, so excluding one also costs its non-throwing cases — prefer
per-case guards if the list grows.

### C0. Full implementation review, 2026-09-19 — 51 findings, fix plan in `REVIEW.md`

11 BLOCKER, 21 MAJOR. Read `REVIEW.md` §2 first: the fifty-one reduce to six causes, and §4 gives the
fix order with the gate additions that keep each one closed. Two corrections to THIS file followed
from it: B1 above was stale (the MOAB-enabled build configuration builds, see C1) — now closed, above. C1 below understated the
MOAB situation — the test file opened with `#ifdef FUSE_HAVE_MOAB` and no config include, so its 38
cases compiled to a skip stub in every configuration, host included (`REVIEW.md` R-06, not R-09 as
first written here; R-09 is the unrelated `step_period()` collective-agreement BLOCKER). `test/backend/moab_mesh_test.cpp` now includes `config.hpp` and tests `#if FUSE_HAVE_MOAB` — the
cases compile again, but C1's "needs one host run" still stands: nothing here can `MPI_Init`.

### ~~C1. The MOAB backend is built but unrun in the sandbox~~ — CLOSED: every MOAB row passes on the host, 2026-10-08

**Host run, 2026-10-08 (the user's machine, `build-all`, `ctest -L moab`):
12 of 12 rows pass**, `backend.moab_mesh` (47 cases) and `backend.remap_halo`
(8) in 0.57 s and 0.05 s, plus the ten multi-rank rows this entry's own
"needs one host run" was written before: `parallel.remap_halo_mpi.{1..5}`
and `parallel.moab_placed_view_mpi.{1..5}`, 0.06–0.59 s each. The seven cases
whose expectations this port moved from `std::out_of_range` to
`fuse::InvalidArgument` are inside that count and pass. Nothing about the
MOAB backend is unrun any more; what is still unrun is the mutation gate over
it (`FUSE_MUTATE_MPI=1 python3 tools/mutate.py --build-dir build-all --area
backend`), which is a question about the tests' own strength, not the code's.

The ten multi-rank rows were registered all along and `ctest -L moab` did not
select them: `fuse_add_mpi_test` had no `LABELS` option until `1ad9efc`, so
the selector reported 2 rows where the build had 12 — a label gap that read
as a coverage gap.

**Status before that run: needs one host run.** `S5.2` landed 2026-09-18: the MOAB-enabled build configuration configures,
compiles and links for the first time, and `test/backend/moab_mesh_test.cpp`
(47 cases as of 2026-10-08; 38 when this entry was written, ported from the
parent) is registered whenever MOAB is enabled.

None of it has executed here. The MOAB-enabled build configuration requires MPI by the build-configuration ladder, and this
sandbox cannot `MPI_Init` — every test in an MPI-enabled build dies at startup
in `MPI_Init_thread`, which is why the MOAB-enabled build configuration reports 81 of 87 failing here and why
that number says nothing about the code. The ladder is right to reject MOAB
without MPI, so there is no serial configuration to fall back on.

What a host run would establish: the 38 MOAB cases, and in particular the seven
whose expectations this port moved from `std::out_of_range` to
`fuse::InvalidArgument` when the bare throws became `FUSE_THROW`.

```sh
cmake --build build-tier2 -j12 && (cd build-tier2 && ctest --output-on-failure)
```

### B2. The 3D order study waits on a spatial index

`build_supermesh_weights` uses a brute-force bounding-box candidate test. It is
exact and has no tuning parameter, which is why it is the reference an index
must reproduce — but a five-level 3D convergence study is unaffordable with it
(the 5×5×5 → 7×7×7 pair alone is ~1.5 million clip calls). The 3D kernel is
pinned entrywise against the closed-form tensor product instead, which says
more about the geometry than a slope does.

### ~~B3. `supports_rollback()` is still a single bool~~ — CLOSED: `Component::capabilities()` is now a `std::uint64_t` bitmask

`include/fuse/runtime/component.hpp`, `class Component`. The ONE virtual is
now

```cpp
[[nodiscard]] virtual std::uint64_t capabilities() const { return 0; }
```

`supports_rollback()` (unchanged signature and meaning), plus two new
readers, `determinism_level()` and `carries_hidden_state()`, are non-virtual
convenience functions over it — a derived class states everything it knows
about itself in one call, and a stale
`bool supports_rollback() const override` from before this change no longer
overrides anything, which is a compile error
(`runtime.compile_fail.stale_rollback_override`, paired with the control
`runtime.compile_ok.capabilities_override`).

**The bit space is REUSED, not duplicated.** `fuse::ComponentSemantics::capabilities`
and `fuse::FieldRequirement::required_capabilities` (`include/fuse/model/semantics.hpp`)
already carried a `std::uint64_t` with no named bits at all — the model layer's
own tests set it with literal patterns like `0b1011` because there was nothing
else to set. `Component::capabilities()` now returns bits numbered in that
SAME space; the assignments live in the new `include/fuse/model/capabilities.hpp`
(bit 0 rollback, bits 1-2 a `fuse::DeterminismLevel` encoding, bit 3 hidden
state, the rest reserved), which both `component.hpp` (runtime) and
`semantics.hpp` (model, via a doc-comment pointer — no dependency edge is
needed there since the fields were already `std::uint64_t`) now reference.

**The runtime checks the new bits.** `verify_rollback_support()`
(`src/runtime/checkpoint_support.cpp`, reached from `Runner::initialize()` via
`verify_iteration_region()`) now also rejects `E-ITER-NONDETERMINISTIC` for a
component inside an implicit coupling group's checkpoint set whose
`determinism_level()` decodes to `DeterminismLevel::Nondeterministic` — the
exact rejection `src/model/multiscale.cpp`'s `review_surrogate_contract()`
already predicted in prose ("it is an error to use \[a nondeterministic
surrogate\] inside an implicit coupling group, which the compiler checks when it
knows the graph") but could not itself implement, having no graph and no
`ComponentSet`. Pinned by `Checkpoint.NondeterministicComponentInsideIterationRegionIsRejected`
(fires) and `Checkpoint.ExplicitlyBitwiseComponentInsideIterationRegionIsAccepted`
/ `Checkpoint.DefaultDeterminismInsideIterationRegionIsAccepted` (agree and
pass), all in `test/runtime/checkpoint_test.cpp`. Mutation rows
`b3-determinism-check-disabled` and `b3-rollback-bit-read-from-wrong-field`
(`build-evidence/mutations/ir-schedule.toml`) are hand-verified.

**Default.** `capabilities() == 0` (no override at all) decodes to {no
rollback, determinism **undeclared**, no hidden state}. The first version of
this change decoded the zero field as `Bitwise`, on the reasoning that it kept
every pre-bitmask override meaning what it meant before; rejected the same
day: `Bitwise` is the strongest claim in the enum ("bit-identical ... across
the rank counts the coupler may use") and a component that never mentioned
determinism has not made it — reading silence as that claim is exactly the
assumption this project's rules forbid. So the enumerators are stored at raw
value + 1, `decode_capability_determinism()` returns `std::nullopt` for zero,
and an implicit coupling group **warns** (`W-ITER-DETERMINISM-UNDECLARED`,
`Checkpoint.UndeclaredDeterminismInsideIterationRegionWarnsAndRuns`) rather
than rejecting — a rejection would reject every component written before the
bitmask. All nine example components now declare their real triple:
rollback, `Bitwise` (measured by `example.*.omp_identical`), and **hidden
state = true** — each keeps a time grid, a history or a solver's internals that
`register_fields()` never bound and the runtime's byte-for-byte rollback
check cannot see, which is precisely what their `save_state()` hooks carry.

**Measured, not asserted.** A full build at the base build configuration (this checkout, this session):
0 errors, 129/130 ctest pass (the 130th, `backend.native_generators.omp1`, is
a `./.tmp/` shared-path race between two ctest workers under `-j8`, unrelated
to this change — passes alone; `ctest -R backend.native_generators` 1/1
green). `runtime.checkpoint`, `runtime.component_iface` and `model.capabilities`
all pass in isolation. Both mutation rows above were actually applied to the
real files, rebuilt, run, and restored (`cmp` byte-identical against the
pre-edit copy) — the exact subtests each one flips are in the toml's own
`describes` field, not repeated here to avoid the two copies drifting.
`check-invariants.py` / `check-layering.py`: 0 violations each.
`mutate.py --check-anchors`: 178/178 valid. `check-parallel-matrix.py`: 3
pre-existing gaps (`A-PARTITIONED`, `A-SPACES-TRANSPORT`, `A-NODAL-HT-PARALLEL`
— none mention rollback, determinism or capabilities; unaffected by this
change). Sphinx: exit 0.

**Left open, found while closing this one:**
  - `test/parallel/runner_transport_test.cpp`'s `LinearRelaxComponent`
    (line ~528) still has the pre-B3 `bool supports_rollback() const override`
    and is now a compile error. It was out of scope for both trackers active
    on this file (one owned only this file's `StubComponent`, the other owns
    a different file); the fix is mechanical and identical to every site
    above: `[[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }`.
  - `docs/dev/abi-design.md` does not currently have a capabilities table (or
    any mention of `rollback`/`determinism`) to update — checked directly,
    not assumed absent.
  - `include/fuse/model/descriptor.hpp` does not serialize
    `Component::capabilities()` (it is a runtime, not a descriptor,
    property), so no descriptor version bump applies here.

### ~~B4. Coupling windows: implemented, type-checked, not built or run~~ — BUILT, TWO REAL BUGS FOUND AND FIXED, ALL 13 TESTS PASS

`include/fuse/schedule/window.hpp` + `src/schedule/window.cpp`
(`CouplingWindowScope`, `validate_coupling_window`) and
`include/fuse/runtime/window_coupler.hpp` + `src/runtime/window_coupler.cpp`
(`WindowCoupler`) — the multi-time-grid waveform-iteration mechanism, docs in
[](docs/dev/coupling-windows.md) and [](docs/user/coupling-windows.md).

**History.** This entry originally recorded that the mechanism had been
type-checked (`clang++ -fsyntax-only`) but never actually built, because the
disk shared by three concurrent tracks had under 1 GiB free. Once the lead
registered `schedule/window.cpp`/`runtime/window_coupler.cpp` in
`src/CMakeLists.txt` and the disk recovered, the real build surfaced 5 of 13
`runtime.window_coupler` failures the syntax-check and the Python
reimplementation had both missed. Root causes, found by comparing
`WindowCoupler`'s C++ against the validated Python model window-by-window on
the smallest failing case:

1. **`build_delivered()` read the live, mutating `provided[]` pointer for a
   member's window-start value.** Correct only for the very first call
   (right after `save_state()`, before any `advance()` this window); every
   later sweep's call read whatever the PREVIOUS sweep's `advance()` had
   just left there — the value at the end of the last subcycling step, silently
   mislabelled as the value at `start_clock[i]`. This corrupted every
   iterated `mean_over()` after sweep 0 for EVERY case, straddling or not
   (`ErrorConvergesNearOrderTwoAdjustableAdjustable`'s order collapsing to
   ~0 and `RatioOneMatchesTodaysSingleClockRegion`'s wrong converged value
   both trace to this). Fixed by capturing `start_value[i][e]` once, right
   after `save_state()`, and reading that captured copy instead of the live
   pointer inside `build_delivered()`.
2. **`build_delivered()`'s `Waveform::reset()` started at `start_clock[i]`,
   not at the window start `T0`.** When a `Fixed` member's time grid is already
   past `T0` (a resolved straddle from the previous window leaves it there
   by design), a consumer whose own subcycling step spans the full window --
   the window-definer's single subcycling step, in particular -- needs that
   member's value starting at `T0`, before the member's own current-window
   data begins. This crashed outright: `Waveform::integrate: [13, 26] is not
   inside the support [14, 26]` (`IncommensurateSevenThirteenAllPolicyPairs`,
   both order tests). Fixed by extending the reset to
   `min(start_clock[i], T0)` and, when `start_clock[i] > T0`, prepending a
   leading sample at `T0` using `state_[i].history[e].back().second` --
   the boundary value this member already committed at the end of the
   previous window, needing no re-derivation since `T1(prev) == T0(this
   window)` by construction.

Both bugs were absent from the Python cross-check because Python's
hand-rolled `wf_integrate`/`wf_evaluate` helpers silently clamp or drop
out-of-range contributions instead of rejecting -- the real
`fuse::Waveform::integrate()` correctly enforces its documented support
contract, which is what turned bug 2 into a loud crash instead of a silently
wrong number. Bug 1 doesn't depend on straddling at all and the Python
model never reproduced it because its own `build_delivered`-equivalent
re-derives each sweep's inputs from the SAME variables it also treats as
"the window-start value" without an analogous live/mutating pointer to
confuse them.

A third, separate defect was in the TEST, not `WindowCoupler`:
`RatioOneMatchesTodaysSingleClockRegion`'s hand-rolled reference (a) never
called `restore_state()` before re-advancing on each Gauss-Seidel sweep, so
it compounded across all 100 sweeps instead of resetting each one, and (b)
fed each component its partner's raw iterate rather than the MEAN of the
partner's window-start value and current iterate -- what a single
full-window subcycling step's `mean_over()` actually evaluates to for a two-knot
linear Waveform. Both fixed in the test fixture; `WindowCoupler`'s own code
needed no change for this one.

**Current state: all 13 `runtime.window_coupler` tests pass** (rebuilt and
run at the base build configuration: `cmake --build ./b0 --target fuse_runtime_window_coupler_test
fuse_schedule_window_scope_test fuse_schedule_multirate_region_test
fuse_verification_iteration_test fuse_runtime_runner_test && ctest`), along
with all 14 `schedule.window_scope`, all 3 `schedule.multirate_region`, all
32 `verification.iteration` and all 49 `runtime.runner` tests -- no
regression in any file this track does not own.

**Measured order, verbatim from this build** (refining the window length by
half four times at ratio 13:7; instrumented with a temporary `std::cerr`,
captured, then reverted -- these are not left in the test):
Adjustable/Adjustable errs `[0.262615, 0.0765105, 0.0118387, 0.00176083]`,
orders `[1.77922, 2.69214, 2.74918]`; Fixed/Fixed errs `[0.281933,
0.0614089, 0.00721211, 0.000942069]`, orders `[2.19883, 3.08995,
2.93652]`. Only the Adjustable/Adjustable series starts below 2 at the
coarsest step (1.78); Fixed/Fixed sits above 2 at every step measured in
THIS build. **This differs from the pre-fix Python cross-check's Fixed/Fixed
numbers** (`[1.94, 3.87, 3.00]`, quoted in an earlier revision of this
entry) -- both are comfortably inside the `EXPECT_GE(order, 1.7)` bound the
test actually asserts (unweakened), and the discrepancy is confined to the
`Fixed` policy's straddle-handling path (Adjustable/Adjustable matches the
Python model to 3+ significant figures on every number); it was not chased
further since no test failed and the bugs found and fixed above (see the
two numbered causes) fully explain the ORIGINAL failures. Recorded here as
an open observation, not a resolved discrepancy -- see the delegated task's
open_questions.

`WindowCoupler`'s own answer for the simplest case (m=1, window=1, 5
windows, Constant relaxation) now agrees with the independent Python
reimplementation to 8 significant figures (`0.051993874...`), where before
the fix it silently converged to a different, wrong value
(`0.018690611...`).

---

## D. Real limitations, documented and pinned

### ~~D1. An implicit coupling group's contiguity depends on declaration order~~ — FIXED 2026-09-21

Two independent field transfers on the same time grid, one inside an implicit coupling group.
The scheduler ordered an instant by topological rank alone, so the two chains
interleaved whenever their ranks alternated — and whether they did depended on
the order the user happened to declare their components in. An implicit coupling group cut in
half that way was rejected (`E-SCHED-REGION`) although an order existed that
kept it whole.

The interleaving instruction was usually the group's **own** sink: a
non-member that reads the group's converged output and must run after the
loop, which the rank tie-break sometimes placed between two members.

**Fixed** by `group_regions_within_instants()` (`src/schedule/scheduler.cpp`):
within each (time, phase) instant, emit greedily and prefer an open group's
ready members. Only ready instructions are ever emitted, so it cannot invert a
dependence — and with no group in an instant the result is the rank order
itself, which is why every golden IR file is byte-identical.

Grouping is **skipped** for an instant where a non-member reads an arc that a
member also reads and that no member produces — the same external value the
group re-advances. That is the hanging-field-transfer hazard
(`Iteration.RejectsAnExplicitChannelHangingOffAnIteratedComponent`): the
non-member would run once, on one sweep's iterate, so separating the interleave
would turn a rejection into a silently chosen processor set. Such an instant keeps
the order it had, and with it the outcome it produced before. The first attempt
at this fix did separate it, and that test is what caught it.

`ExecutionPlan.ARegionSchedulesWhicheverOrderItsNeighbourIsDeclaredIn` asserts
both spellings schedule; two mutations fail if either half of the rule is
removed. `E-SCHED-REGION`'s message no longer mentions declaration order.

---

### ~~D2. One CommGraph-backed Transport per run~~ — FIXED 2026-09-20

`Runner::Impl` holds a single `std::unique_ptr<Transport>`, so a plan with two
cross-rank field transfers is rejected at `initialize()`:

```
E-RT-TRANSPORT: this build wires one CommGraph-backed Transport per run, but
the plan has 2. One Transport instance per instruction is the next step.
```

Two independent cross-rank couplings is not an exotic graph — it is the
ordinary multiphysics case, and the brief this library exists to serve names
it. So this is the sharpest remaining limit on what can actually be coupled,
not a corner.

Surfaced 2026-09-20 by the host run of the MPI-enabled build configuration once two earlier faults in the same
test were cleared (`RunnerTransport.StillRunsASecondChannelAfterARegionFails
UnderMpi`, which needs exactly this). That test is the acceptance criterion and
is already written; it fails on this and nothing else.

**Fixed.** `Impl::graph_transports` is a vector of one Transport per
CommGraph-backed plan, built inside the per-plan loop of
`attach_comm_graphs()` — which is where it has to be, because
`peer_plan_needed`/`send_peers`/`recv_peers` describe the plan being walked
and are overwritten by the next iteration. Building after the loop, on the
last plan's peer lists, is what had limited the runner to one field transfer. Each
instance is created from the registry, passed through
`RunnerConfig::transport_setup`, then planned; every rank builds the same
number in the same order, which matters because `set_communicator()`
duplicates the communicator and `build_plan()` is collective. Instances are
NOT shared between plans: a Transport carries the plan it was built for and
`prepare()` rejects any other spec. Mutation
`one-transport-shared-across-every-plan` pins that. Verified in the sandbox
only as far as compiling in all three tiers; the acceptance test is
`RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi` and needs
a host run.

**The superseded plan was** — `transport` becomes a vector indexed by transport plan;
`attach_comm_graphs()` builds and wires one per plan rather than one per run
(`src/runtime/runner.cpp:1321` is the rejection, `:1467` the single-instance
`dynamic_cast` wiring, `:2097` and `:2124` the execution sites). The open
design question is whether two plans with an identical `TransportSpec` share
one instance or get one each.

---

### ~~D6. A component placed nowhere is legal, but a field transfer from it cannot initialise~~ — RULED AND FIXED 2026-09-22

`RankSet` said an empty processor set was legal; `MpiTransport::build_plan` rejected
one. **Ruling: the rejection is right** — transporting out of a component that
exists nowhere is a modelling error, and `RankSet`'s contract was the
incomplete half.

Fixed accordingly:

- `validate()` now rejects a field transfer whose source or target component is placed
  on no rank, with `E-PLACEMENT-EMPTY` naming the component and the field transfer.
  A model-layer check, so it fires for every transport and before any
  communicator exists — where the previous `E-TRANSPORT-RANKS` told the user
  about a rank set when the mistake was in the graph, and only under MPI.
- `RankSet`'s contract says an empty processor set is legal **only for a component
  with no field transfers**, and says why.
- `docs/user/placement.md` states the same rule from the user's side.
- Five tests pin both halves — each end rejected separately, both ends reported
  separately, and an empty processor set with no field transfer still accepted, because
  that is the supported way to switch a component off for one run. Two
  mutations (`empty-placement-channel-check-never-fires`,
  `empty-placement-check-fires-on-every-channel`) fail if either half goes.

The transport-level rejection stays exactly as it was: it is now a backstop
rather than the primary diagnosis.

---

### D5. Partial rank-set overlap through the Runner — FIXED AND HOST-VERIFIED 2026-09-23; MOAB view layer written 2026-09-28, awaiting the host

**MOAB per-component decomposition (`8cf2dfe`, `0e290ab`).** `MoabMeshBackend::set_placements(placements)`,
one `RankSet` per graph component (couplers included, read from `graph.component_count()`),
builds one `moab::Core` and one `moab::ParallelComm` per processor set on a sub-communicator split
from the base communicator over exactly that processor set's ranks: iMOAB's application-id model,
as approved. `placed_view`/`placed_region_size` answer from that decomposition, and outside the
processor set they return an **empty** view, never the default one. Processor-set count and ranges are
agreed over the base communicator before the first split, and every processor set's read outcome
is reconciled there too, so a rank outside a processor set is never left waiting. With no
processor sets declared every existing caller gets exactly today's answer. The caller declares
the processor sets, as with the native and test backends; the Runner does not.

**Not run anywhere yet.** No MOAB-linked binary can start in this sandbox (`moab::Core`'s
constructor calls `MPI_Init_thread`), so the 9 single-process cases in `moab_mesh_test.cpp`,
the 2–5 rank `parallel.moab_placed_view_mpi.*` tiling test and the 8 rows of
`build-evidence/mutations/moab-placement.toml` were compiled (the MOAB-enabled build configuration, 0 errors) and traced only.

**HOST-VERIFIED 2026-10-08.** The user's run: `parallel.moab_placed_view_mpi.{1..5}`
pass at 1, 2, 3, 4 and 5 ranks (0.06–0.33 s), alongside `backend.moab_mesh`'s
47 cases and the five `parallel.remap_halo_mpi` legs — 12 of 12 MOAB rows, no
failures. So one `moab::Core` and one `moab::ParallelComm` per processor set,
split from the base communicator, answering `placed_view`/`placed_region_size`
from that decomposition and returning an empty view outside the processor set,
is now measured at every rank count it registers, not inferred. What is still
unrun is the mutation gate over those tests:

```sh
FUSE_MUTATE_MPI=1 python3 tools/mutate.py --build-dir build-all --area backend
```

**What remains of D5:** per-placement remap *weights* built on two independently decomposed
MOAB processor sets and compared. The builders already resolve through `placed_view`, but nobody
has assembled and checked such a matrix.


**History.** First recorded as a coverage gap, then corrected to a capability
gap. The Runner filled both `CommGraphSide`s from one `backend.view(MeshId)`,
so each rank held one piece of each mesh, shared by every component on it,
and two partially overlapping processor sets could not both cover the mesh entity set.

**Ruling (2026-09-22): per-placement views in one backend**, MOAB's
application-id model.

**Done (`2a1e8c0`):**

- `MeshBackend::placed_view(MeshId, ComponentId)` and
  `placed_region_size(Region, ComponentId)`. They default to the
  processor-set-free answers, so every existing backend and every
  single-decomposition run behaves exactly as before; the base build configuration is unchanged at
  72/72. They are named rather than overloaded because an overload of `view`
  would be hidden in every derived backend. Contract: identical processor sets get
  identical decompositions, since the IR emits no Transport between them.
- `RegionSizes` takes the arc's processor set (`ValueType::rank_set`), so buffers
  are sized from their own component's decomposition.
- The Runner gives each CommGraph side its own component's view. The Remap
  identity check and the conservation measure use the arc's processor set, and
  the measure now resolves its mesh **by name** (it had passed the graph's
  `MeshId` straight to the backend, the positional-id hazard
  `make_region_sizes` exists to avoid).
- **A latent defect the harness exposed:** the hot loop ran the matched path's
  gather and scatter on a *peer* plan, where both are empty. A peer exchange
  sent zeros and delivered nothing through the Runner. No Runner test used
  partial overlap, which is the only shape that produces a peer plan, so it was
  never run. Fixed: the peer plan's extent is the larger of the rank's two
  blocks, the source block is staged whole, and only the destination indices
  the plan writes (precomputed, allocation-free) are copied back.
- ~~`E-RT-REMAP-PLACEMENT`: a Remap on a non-default decomposition rejects~~ —
  **closed 2026-09-25.** `RemapPlan::placement` carries the component whose
  decomposition the weights index; it is part of `RemapPlan::key()`, and every
  builder (supermesh, MOAB/Tempest, consistent, MLS) resolves its sides through
  `MeshBackend::placed_view(id, placement)`. Asserted where the matrix can be
  read: `Supermesh.WeightsAreBuiltOnThePlacementsOwnDecomposition` gives one
  processor set a target view numbered backwards and requires the row-reversed
  matrix, exactly. (Its first draft reversed BOTH sides — a mirror symmetry of
  a uniform line pair, matrix unchanged — and its own "the matrices coincide"
  guard caught that.) `Runner.AcceptsARemapOnANonDefaultPlacementDecomposition`
  replaces the rejection test.
- Three shape tests through one helper (partial at 3–5 ranks, full, disjoint);
  the test backend builds one decomposition per processor set.
- Mutations: `remap-ignores-the-placement-decomposition` (now the builder
  reading the default view, killed by the reversed-numbering test) KILLED here;
  `both-commgraph-sides-share-one-view`,
  `peer-plan-staged-through-the-matched-path` and
  `peer-extent-is-only-the-source-block` NOT-ASSESSED (need MPI).

**D3 note (2026-09-23):** D3's new `TwoIndependentCommGraphChannelsBothDeliver
InOnePeriod` is the first Runner test to check DATA delivered through TWO
distinct CommGraph-backed plans (`pi=0`/`pi=1`) in one coupling cycle, on the same
per-placement machinery this section built (`attach_comm_graphs()`,
`graph_transports`). Compiled only, not yet run (C1: this sandbox cannot
initialize MPI) — noted here so a future change to plan indexing re-runs it
alongside this section's own three shape tests once the host confirms it.

**Host run 1 (2026-09-23) found a regression I introduced**, now fixed: every
cross-processor-set Transport was rejected with `E-RT-SIZE` ("reads 12 values ... into
0"). The Runner held Transport to the rule that an instruction's input and output
buffers are the same size. That rule held only while both ends shared one view. With
each end in its own component's decomposition, a rank in one processor set and not the
other legitimately has 12 and 0. The base build configuration could not see it, because serial backends
keep one decomposition. Transport is now exempt from that rule, as Remap already was.
`check_transport_extents()` reconciles it after the CommGraph plans exist, with its
outcome agreed across ranks:
- a whole-arc copy must still have equal sizes;
- a matched plan's gather, scatter and local indices must fit their buffers;
- a peer plan's staged source must equal its input, and its written indices must
  fit its output.

The whole-arc-copy branch is tested by
`RunnerTransport.AWholeArcCopyBetweenDifferentDecompositionsIsRefused` (`6b07464`); the
serial harness cannot reach it, because its Transport feeds a Remap whose own size check
fires first. Mutation `transport-held-to-equal-extents` (MPI) pins the fix.

**Still open:**

**Host-verified (2026-09-23):** partial overlap delivers every entity at 3, 4 and 5
ranks, full overlap at every rank count, and all 18 `layout.*` tests pass, including the
3 `layout.migrate.*` ones.

1. **The MPI-gated mutations** for this fix (`both-commgraph-sides-share-one-view`,
   `peer-plan-staged-through-the-matched-path`, `peer-extent-is-only-the-source-block`,
   `transport-held-to-equal-extents`) have not yet been run on a host.
2. **`MoabMeshBackend` does not override the placed queries.** It still has one
   `ParallelComm` over one communicator, so with MOAB every component shares one
   decomposition. Partial overlap is therefore proved with the test backend
   only. The MOAB half is one `ParallelComm` per component communicator, with
   the mesh read once per processor set.
3. ~~**Per-placement remap weights**~~ — done 2026-09-25 (above). What remains
   of D5 is the MOAB half (item 2): `MoabMeshBackend::placed_view` still returns
   the default view for every processor set, so on MOAB the per-placement weights
   are the default weights until one `ParallelComm` per component exists.

---

### ~~D4. Two axes of the parallel matrix have no coverage~~ — CLOSED 2026-09-28, host-verified

**Host run 2026-09-28:** `ctest -R 'layout.coupler|parallel.runner_transport|parallel.remap_halo|backend.remap_halo'`
and `ctest -L omp` all pass on the user's machine. `A-PARTITIONED`'s parallel half therefore
has an outcome, not just a compile: `RemapHaloMpiSphere.BuildRemapHaloDeliversExactHaloValuesAndRoundTripsExactly`
passes at 2 ranks, printing `min_src_coverage=0 max_col_defect=1` for the ATM/OCN pair (rank 0,
local sums; both meshes span nearly the whole globe in their node coordinates, so the zero
coverage is not a regional-patch effect — the likeliest reading, a land-masked ocean grid, is
unconfirmed). `A-SPACES-TRANSPORT` closed with D11. The matrix's one remaining gap is D8's
`A-NODAL-HT-PARALLEL`, which is not this entry's.

`tools/check-parallel-matrix.py` (S5.5) checks that every cell of
`build-evidence/parallel-matrix.toml` names coverage that exists: the test is
really in the file, the file is really registered, the rejection is really in the
library *and* asserted somewhere, and a claimed rank count is really run. It
reported **33 claims verified, 0 broken, 2 gaps** when this entry was written.
Measured on `main` at `b76de49` (2026-09-28, after both closures below and
D11's): **70 claims verified, 0 broken, 1 gap, of 23 cells** — the one gap is
D8's `A-NODAL-HT-PARALLEL`, not one of this entry's two. (An intermediate
"66 verified, 2 gaps" figure that stood here briefly was the halo track's own
report, never re-run by the committer; struck for that reason.)

| Gap | What has no test |
|---|---|
| `A-SPACES-TRANSPORT` | the host-parallel space axis beyond the kernels: only `space_kernels_test.cpp` is typed over execution spaces, so transport and runtime run on one space |

~~`A-PARTITIONED`: the **parallel** half of an independently partitioned
source has no test~~ — **closed 2026-09-28.** The one-rank contract
(`A-PARTITIONED-SERIAL`) was already covered by
`test/backend/remap_halo_test.cpp` (2026-09-21); this closes what it left
open: `build_remap_halo`/`build_weights` under a REAL communicator of more
than one rank.

**What is now written**, all in `test/parallel/remap_halo_mpi_test.cpp`
(new; registered `fuse_add_mpi_test(parallel remap_halo_mpi ...)`, ranks
1–5, gated on `FUSE_ENABLE_MOAB AND FUSE_ENABLE_MPI` — ABSENT, not skipped,
when either is off):
- `RemapHaloMpi.ERemapParallelRefusesAboveOneRankAndNotAtOne` and
  `RemapHaloMpi.EMeshHaloGeometryRefusesTheBoxPairAboveOneRankAndNotAtOne` —
  the `E-REMAP-PARALLEL` and `E-MESH-HALO-GEOMETRY` boundaries
  `moab_backend.hpp`/`remap_plan.hpp` call load bearing ("`> 1` rejects;
  `== 1` must NOT"), checked on BOTH sides using a generated, non-spherical
  box pair — no mesh file needed, since both boundaries are structural
  (rank count, distinct file sets) and neither depends on real geometry.
- `RemapHaloMpiSphere.TheNonMatchingPairLoadsProjectsAndIsSphericalAtEvery
  RankCount` and `RemapHaloMpiSphere.BuildRemapHaloDeliversExactHaloValues
  AndRoundTripsExactly` — the REAL fixture, MOAB's own `wholeATM_T.h5m`
  (atmosphere) and `recMeshOcn.h5m` (ocean), both spherical and
  deliberately not matching (an identity pair cannot exercise a halo at
  all). These need `-DFUSE_MOAB_MESHFILES=<moab-source>/MeshFiles/unittest`
  and are not compiled at all without it, same as `A-PARTITIONED-SERIAL`'s
  own sphere cases.
  - **The fixture needed one more piece that nobody had built**: read
    directly off the files (`h5dump` on the raw node coordinates,
    2026-09-28), `wholeATM_T.h5m` sits at radius 1.0 and `recMeshOcn.h5m`
    at radius 6371229.0 (Earth radius in metres) — two different,
    individually well-formed, spheres. `intx_weights.cpp`'s own
    radius-agreement check (`kGeomTol = 1e-9`) would refuse that pairing
    outright with `E-REMAP-GEOMETRY`, correctly, since the library cannot
    know two different radii are meant to be the same sphere in different
    units. `MoabMeshBackend::project_onto_unit_sphere` (new;
    `include/fuse/backend/moab_backend.hpp` / `src/backend/moab_backend.cpp`)
    rescales one mesh's vertices by a single collectively-reconciled scalar
    factor onto a common unit sphere; the test calls it once per mesh,
    before `build_remap_halo` ever sees them.
  - Forward (`exchange_remap_halo`): every halo slot's value is checked
    EXACTLY against its own `global_id` — no gather needed for the
    reference, since every rank applies the same identity function to its
    own owned cells' global ids.
  - Reverse (`accumulate_remap_halo`): every local entity starts at exactly
    1.0; after the round trip, the Allreduce-summed total over every rank's
    owned entities is EXACTLY (726, the mesh's own known cell count,
    independent of rank count) + (the Allreduce-summed halo-instance count)
    — arithmetic on exact integer-valued doubles, not a tolerance. This is
    also the only place this cell can confirm the halo's send/receive
    bookkeeping agrees on every rank: a lost or duplicated exchange message
    breaks the identity outright.
  - A default (non-`Conservative`) `build_weights()` call with the halo in
    place is asserted to succeed and its audit numbers
    (`min_src_coverage`, `max_col_defect`) are printed for the host log, but
    NOT gated on a specific value: whether this real atmosphere/ocean pair's
    domains overlap enough for `RemapFlags::Conservative` to hold is a
    property of the files nobody has measured yet, and inventing a
    tolerance for a number never measured would violate this project's own
    "never widen a tolerance to make an assertion pass" rule.

**What compiled at the MOAB-enabled build configuration, in this sandbox**: configuring with
`-DFUSE_ENABLE_MOAB=ON -DFUSE_ENABLE_MPI=ON
-DFUSE_MOAB_MESHFILES=<moab-source>/MeshFiles/unittest` found both real
files (`fuse: sphere pair for the remap halo: .../wholeATM_T.h5m +
.../recMeshOcn.h5m`) and registered `test/parallel/remap_halo_mpi_test.cpp`
(the "tests N registered of N on disk" audit line covers it); both
`fuse_parallel_remap_halo_mpi_test` and the pre-existing
`fuse_backend_remap_halo_test` built with 0 errors.
`tools/check-parallel-matrix.py`, measured on `main` at `b76de49`: 70 claims
verified, 0 broken, 1 gap — `A-PARTITIONED` no longer among them (the 66/2
figure reported by the track that wrote this test was not re-run before it was
committed; the D4 heading above carries the measured number).
`tools/mutate.py --check-anchors` and `tools/check-invariants.py`/
`check-layering.py` are unaffected (0 violations).
**UNRUN here, correcting this section's own 2026-09-21 note that the
serial contract's outcome "comes from a host run"**: it does, and more
finally than stated then. This sandbox cannot initialise MPI even at ONE
process with no `mpiexec` at all (C1) — confirmed for THREE separate
binaries built against the same MPI-enabled MOAB (the pre-existing serial
`fuse_backend_remap_halo_test`, this new `fuse_parallel_remap_halo_mpi_test`,
and even MOAB's own `mbsize` CLI tool), every one aborting inside MPI's own
`MPI_Init` before any of them reach a single line of test code. The exact
host command, on a `build-all` configured with MOAB and MPI on:

```
ctest --test-dir build-all -R 'parallel.remap_halo' --output-on-failure
```

For the parallel half, the fixture problem is now fully solved and built,
not merely diagnosed: the note this replaces pointed at MOAB's own
`spherecube.h5m` (a single, self-matching sphere with no partner) and at
`GenerateCSMesh`/`GenerateRLLMesh` as a fallback for building one; the real
non-matching pair this section now uses needed no generation step, only the
radius projection described above.

A gap is deliberately not waivable: the gate counts them and fails, so the
number can only fall by writing the test. A **third** gap appearing, or any
`MATRIX-BROKEN` line, is a new defect rather than one of these.

**Also found by building this gate**, and fixed: rank count **1** was never
run (`fuse_add_mpi_test` defaulted to `2 3 4 5`). A single task is not a
trivial case of the others — every match has both ends on one rank, so the
same-rank shortcut carries the whole exchange and `MpiTransport` must see
nothing, which is row 7's requirement from the other side. The MPI-enabled build configuration went from 92
to 97 registered tests. **Unverified here:** MPI cannot initialise in this
sandbox, so a test that fails at 1 rank instead of skipping — i.e. one missing
its `FUSE_REQUIRE_AT_LEAST_RANKS` guard — will only show up on a host run.

---

### D3. ~~Four behaviours no test detects~~ — two runtime mutations survive on the host; DEFERRED 2026-10-02 in favour of the core implementation

**State on 2026-10-02 (host runs on `ba0892e` and `ca7f7a5`).** The hang the first host run of
`StillRunsASecondChannelAfterARegionFailsUnderMpi` hit was a real runtime defect, fixed in
`bb6f57f`: `reconcile_epoch()` returned `E-RT-EPOCH-PEER` on a healthy process while the failing
one walked on through the remaining collectives, so the failing one waited forever in the next
data redistribution. Since `3aae2ed` steps a component only on its own processes, a step failure
is process-local, which is what exposed it. A healthy process now records the partner's failure
and runs only the collective operations, as the implicit coupling group's residual agreement
already did; mutation row `healthy-rank-leaves-at-the-epoch` pins it (**KILLED** on the host).

With that fix, `-L mpi` on the host is green except for the items below, and the runtime
mutation run is **29 of 31 killed**. The two survivors are gaps in the *tests*, understood but
not yet closed:

1. **`region-failure-abandons-the-rest-of-the-period`.** The fixture's fluid fails on its very
   first call, at t=0. `linkB`'s t=1/2 synchronisation point then tells every process about it
   before the implicit coupling group begins, so under correct code and under the mutant alike all
   processes leave the period together; `cons_b->calls()` is the same either way. A kill needs
   the failure to arise *inside* the group, after the outside process's last agreement point.
2. **`coupler-region-participants-missing-its-ranks`.** `IterationRegionParticipantsIncludeADisjointCoupler`
   relies on `resolve_regions()`'s "would report convergence after one sweep" refusal, which only
   fires when the physics components alone sit on one process (2 processes). At 4 processes they
   span three and the refusal cannot fire. A kill needs an execution-level assertion: every
   process of the group, the coupler's included, takes the same number of sweeps (and more than
   one).

`ca7f7a5` attempted exactly those two changes (fluid failing on its second call; a period run with
an `MPI_Allreduce` of per-process sweep counts) and made `parallel.runner_transport` **fail at 2
processes and time out at 3, 4 and 5** on the host. It is reverted in `d384de7` (`64346e1` is
`ca7f7a5` under a misleading name: the first revert attempt failed on an invalid flag and the
following `--amend` only renamed it); the cause was not found before the decision to defer. The likely explanations, to check first when this is picked
up: (a) a step failure raised *inside* the group's sweep on only some processes reaches the
group's residual agreement on those processes while the others are already past it, or inside a
Transport that the failed processes now skip as non-collective — the same class of asymmetry
`bb6f57f` fixed at the Epoch, one level down; (b) the disjoint-coupler fixture's group may
converge in one sweep, so the `EXPECT_GT(hi, 1)` guard fails on every process rather than
exposing anything. Either way these are test-design questions on a fixture that has already
found four runtime defects; the runtime behaviours the two rows describe are not known to be
wrong.

Also still **NO-BASELINE** on the host while `runner_transport` was red: `both-commgraph-sides-share-one-view`,
`peer-plan-staged-through-the-matched-path`, `peer-extent-is-only-the-source-block`,
`transport-held-to-equal-extents`, `region-residual-skips-the-cross-rank-reduction` — all of
which were **KILLED** on the previous (green) host run of 2026-10-02 and need only a re-run on
the reverted tree to be confirmed again.

**Deferred** by the user's decision of 2026-10-02: the core implementation plan comes first;
example and runner-test hardening resumes after it.

The text below is the entry as it stood after the 2026-09-28 host run.

**Host mutation runs 2026-09-28 (`FUSE_MUTATE_MPI=1`, `build-all`):** `one-transport-shared-across-every-plan`
(transport area), `region-residual-skips-the-cross-rank-reduction` (runtime) and
`vertex-ownership-rule-picks-max-not-min-rank` (backend) are **KILLED** — the tests written for them
on 09-23 do detect them. `region-failure-abandons-the-rest-of-the-period` still **SURVIVED**. A trace
of `RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi` found no flaw in its kill
mechanism (`cons_b->calls() >= 2`), so the open question is empirical: run it alone, on the clean
tree and with the mutation hand-applied and the binary rebuilt, and report each rank's error code
and `cons_b->calls()`:

```sh
mpiexec -np 2 build-all/test/fuse_parallel_runner_transport_test --gtest_filter='*StillRunsASecondChannelAfterARegionFailsUnderMpi*'
```

The text below is the entry as it stood before the host run.


`FUSE_MUTATE_MPI=1 tools/mutate.py --area transport|runtime` exits 1 with three
survivors, plus a fourth in `native-mesh.toml` (vertex ownership, below).

| Survivor | Behaviour with no test that detects it |
|---|---|
| `one-transport-shared-across-every-plan` | D2's fix: one Transport per CommGraph-backed plan. Collapsing every plan onto instance 0 fails nothing. |
| `region-failure-abandons-the-rest-of-the-period` | R-09: a failing coupling cycle carries the failure forward instead of returning at the group. |
| `region-residual-skips-the-cross-rank-reduction` | R-10: the group's residual is agreed across processes rather than summed rank-locally. |

**D1 is fixed, and the fixture already reflects it — the old "declared first"
diagnosis in this section was stale.** Reading the checked-out
`test/parallel/runner_transport_test.cpp` directly: `with_independent_channel`
declares "linkB" AFTER `add_iteration_region` (see the class comment; D1
landed at `2de9247`/reordered again since). So the three survivors are not
explained by declaration order at all. Tracing each mutant's actual code path
instead:

- **`one-transport-shared-across-every-plan`.** `attach_comm_graphs()`
  (`src/runtime/runner.cpp`) still pushes one `TransportPlan` per Transport
  INSTRUCTION regardless of declaration order (`case OpKind::Transport:`,
  unconditional, no dedup by shape) — "link" and "linkB" already get distinct
  plan indices `pi=0`/`pi=1` in the existing fixture. The mutant nulls
  `graph_transports[1]`; executing "linkB"'s Transport at `pi=1` then hits the
  `tp == nullptr` guard and returns a clean `E-RT-TRANSPORT`. The OLD test
  never saw this: it ALSO forces "fluid" to fail inside the group, and
  `step_period()`'s failure branch only overwrites `step_failure`
  `if (!step_failure.has_value())` — the group's failure already claimed
  that slot, so "linkB"'s distinct error code was discarded, masked rather
  than absent. And `cons_b->calls()` never read the DELIVERED DATA, only
  whether the tick ran at all — which it does, from whatever the (possibly
  null-Transport, uninitialized) input buffer held.
- **`region-failure-abandons-the-rest-of-the-period` (R-09).** Producer_b/
  consumer_b's own tick sat at the SAME `t=0` the group occupies (interval
  `Rational{1}`, same as fluid/solid) — `group_regions_within_instants()`
  only reorders the INSTRUCTION stream inside one `(time, phase)` window, so
  a same-time observable's position relative to the group is a greedy
  implementation detail, not a property either the fix or the mutant is
  actually about. An "after" that depends on winning that tie-break is not a
  real "after".
- **`region-residual-skips-the-cross-rank-reduction` (R-10).**
  `RegionConvergesInTheSerialIterationCount` gives `A` the SAME `k` on every
  rank; because the map is entity-wise with no coupling between entities, a
  RELATIVE criterion's decay rate does not depend on how many entities are
  summed (1 rank-local, or the global 4) — local and agreed residuals reach
  the SAME outcome on this data regardless of how unequal the entity split
  is, so the mutant is invisible however many ranks run it.

**Done.** `test/parallel/runner_transport_test.cpp` (owned by this track):
- `TwoComponentJob` gained an `independent_interval` parameter (default
  `Rational{1}`, every existing caller unchanged) and `write_source_b_by_
  global_id()`/`count_destination_b_matches()`/`local_destination_b_entity_
  count()`, the "linkB" analogues of the existing "link" data-verification
  helpers.
- `StillRunsASecondChannelAfterARegionFailsUnderMpi` now puts producer_b/
  consumer_b on `Rational{1,2}` (two ticks per coupling cycle, at `t=0` and `t=1/2`)
  instead of sharing the group's `t=0`. `schedule()`'s own `(time, phase,
  rank)` sort key places the `t=1/2` tick strictly after everything at `t=0`
  — not a tie-break, an ordering guarantee — so `EXPECT_GE(cons_b->calls(),
  2)` is a genuine "after the group" check now.
- New `TwoIndependentCommGraphChannelsBothDeliverInOnePeriod`: nothing fails
  (no implicit coupling group at all), both "link" and "linkB" run, and both field transfers' data
  is checked by global id. This is the direct kill for the shared-transport
  mutant: no failure exists to mask a distinct error code, and a shared
  instance that happens to work for `pi=0` but corrupts `pi=1` is caught by
  checking `pi=1`'s own delivered data, not just whether the run failed.
- New `RegionResidualLocalConvergenceIsNotEnoughUnderMpi`: `LinearFixedPointJob`
  gained `kA`/`tol` parameters; rank 0 builds `A` with `kA = 0.001`, every
  other rank with `kA = 0.5` (same graph structure everywhere — coefficients
  are a runtime, not a graph, property). Simulated: rank 0's own relative
  residual clears `tol = 1e-6` by sweep 4, every other rank's is still
  `3.8e-3` at sweep 5 (`max_iterations = 5`). The fix must report
  `converged == false`, `iterations == 5` on EVERY rank (the agreed residual
  stays dominated by the slow ranks); the mutant lets rank 0 decide from its
  own local residual and report `converged == true` at a smaller count —
  caught by `EXPECT_FALSE`/`EXPECT_EQ` without needing a hang, though a hang
  on a later sweep (rank 0 stops participating once it thinks it's done) is
  an equally correct kill per the mutation's own `describes` text.

**Vertex ownership** (`vertex-ownership-rule-picks-max-not-min-rank`,
`build-evidence/mutations/native-mesh.toml`). New
`NativeMigrationMpi.VertexOwnerIsTheMinimumRankAmongHoldersAfterMigration`
(`test/backend/native_migration_mpi_test.cpp`): after a real
`fuse::partition::repartition`, gathers every rank's `(vertex gid, owner)`
pairs via the SAME Allgather-counts/Allgatherv-payload shape
`resolve_vertex_ownership` itself uses, independently recomputes the minimum
holder rank per gid in the TEST's own (unmutated) code, and checks both that
every holder of a shared vertex agrees on one owner and that the agreed
owner is that minimum. `RepartitionConservesGlobalCellCountAndMeasureAndUnique
Ids` never reads `vertex_owner`/`vertex_is_owned` at all, so it cannot see
this mutant regardless of rank count.

**Not reachable at the base build configuration.** `resolve_vertex_ownership`'s cross-rank
comparison lives inside `#if FUSE_HAVE_MPI`, which a build at the base build configuration does not
compile at all (dead-code-eliminated, not just runtime-skipped), and every
base-build-configuration caller of `migrate()` in `test/backend/native_migration_test.cpp`
passes `kNullComm`, which returns before that `#if` block regardless of
whether MPI is compiled in. No emulated assertion was added there: there is
no code path, compiled or not, for one to reach.

**Status: compiled here, awaiting the host's mutation run.** All four new/
changed assertions compile cleanly against `build-all`'s existing object
flags (checked by compiling each changed file to a throwaway `.o` with the
flags `compile_commands.json` recorded, given the sandbox's near-full disk —
see `local_verification`); none of them ran, and none of the three
`transport-runtime.toml` mutations or the `native-mesh.toml` one were
re-applied, since this sandbox cannot initialize MPI (C1) and mutate.py
rewrites sources in place (forbidden while two other tracks share this
checkout). `host_verification_needed` lists exactly what the host run must
confirm.

---

### D7. `WindowCoupler` supports at most one straddling member per window, and no Fixed step longer than the window — REJECTED 2026-09-24, support is design work

Found 2026-09-24 while testing an explicit window
(`test/runtime/window_coupler_test.cpp`) — an explicit window interval
shorter than every member's own step, so every member straddles at once,
which the delegated task's own design allows
(`docs/dev/coupling-windows.md`'s "The window is set by one named
component's step, or an explicit interval"). Two distinct gaps, not one:

1. **Simultaneous straddles corrupt each other's replay.**
   `resolve_pending_straddles()` (`src/runtime/window_coupler.cpp`)
   iterates members one at a time; each straddling member's replay reads
   the OTHER member's `last_window_waveform`/`history` for the partner
   data it needs. When BOTH members have `straddle_pending` in the same
   window, this throws `Waveform::integrate: [range] is not inside the
   support [range]` — measured directly, Fixed/Fixed throws within two
   windows (window=12/d, intervals 13/d and 7/d, every d tried); the
   mixed cases (one member Fixed, one Adjustable — only the Fixed one
   straddles) do not throw but show non-monotone or growing error with
   refinement, never order 2.
2. **A member whose interval exceeds the window by more than double
   must eventually skip a window outright** (real time, tracked by the
   window index, falls behind the member's own time grid by more than one
   window's worth), which nothing in the commit path
   (`sweep_window()`'s per-window `history[e].push_back(...)`,
   `build_delivered()`'s `Waveform::reset(start_clock[i], ...)`) handles:
   `real_traj[i][e]->reset()` is asked for a window with
   `window_begin > window_end` once `start_clock[i]` has raced past the
   current window's own end, and throws `InvalidArgument`. Measured with
   window=10/d, interval_A=13/d (excess 30%): the subcycling step grid alone
   (no Waveform involved) shows the first skip at window index 4,
   scale-invariant in d (a property of the 13:10 ratio, not the absolute
   step size) — confirmed by direct simulation, not the WindowCoupler
   build itself, since 1 already stops the run first at a much smaller
   window index for that same configuration.

**Now rejected, not run** (`WindowCoupler::create`). Running either case
returned a wrong answer without any diagnostic in the mixed configuration,
so both are rejections:
- `E-RT-WINDOW-LONGSTEP`: a Fixed member whose interval exceeds the window.
  This was the mixed case that "never reached order 2": the Fixed 13/d member
  against a 12/d window, which spans a window end it cannot replay.
- `E-RT-WINDOW-STRADDLE`: two Fixed members whose steps both fail to divide
  the window, so both can straddle one window end.
Tests `WindowCouplerRefusals.AFixedStepLongerThanTheWindowIsRefused` and
`TwoFixedMembersThatCanStraddleTogetherAreRefused` make each fire, and
`AFixedStepThatDividesTheWindowNeverStraddlesAndIsAccepted` shows that a Fixed
step dividing the window is not counted.

**What is verified inside the envelope.**
`WindowCouplerVerification.AWindowNeitherStepDividesConvergesAtOrderTwo`: a
12/d window with A Adjustable at 13/d (truncated to the window) and B at 7/d,
both Fixed and Adjustable, converges at order in [1.9, 2.1] at every one of
four phase-aligned refinements. It replaces the earlier
`BothMembersStraddleAnExplicitWindow`, which asserted only a finite result.

**What to do.** Both gaps need multi-window replay support:
`resolve_pending_straddles()` would need to resolve every member with a
pending straddle together (reading a CONSISTENT snapshot of each other's
data, not one already-replayed-this-call and one not), and the commit
path would need to represent "this member has no new data this window"
(skip the subcycling step AND the history commit, rather than committing
whatever `last_val` happened to hold) separately from "this member
straddled." Neither is a mutation-scale fix; both are new design work,
which is why this is filed here rather than attempted in the bug-fix
pass that found it (which fixed a different, unrelated defect — see
`docs/dev/coupling-windows.md`'s "Two bugs found in the straddle
replay" for that one).

---

### ~~D11. The coupler's runtime executes no Kokkos kernel — OpenMP coverage of it is vacuous, and that is a capability gap~~ — CLOSED 2026-09-28

Found 2026-09-25 while answering "test Kokkos+OpenMP". Kokkos 5.1.0 is built with
`OPENMP;SERIAL`, every test binary initialises OpenMP as the default execution
space, and both physics examples produce byte-identical output at 1 and 8
threads. That last fact was not reassurance: it held because `src/runtime/`
and `src/schedule/` called no `Kokkos::parallel_*` at all. The Runner's
weighted sums, residual norms, remap apply and accelerators were the serial
loops in `num/kernels.hpp`; the typed, space-portable kernels in
`num/space_kernels.hpp` were verified on every enabled space
(`SpaceKernels/*`) and called by nothing.

**Fix.** `src/runtime/runner.cpp`'s `execute()` and `src/runtime/
iteration.cpp`'s `run_region()` now call `space_kernels.hpp`'s kernels over
`Runner::Impl::HostExecSpace` (`Kokkos::DefaultHostExecutionSpace` — OpenMP
here, Serial on a Serial-only build), wrapping the SAME existing raw pointers
as unmanaged `SpanOf`/`ConstSpanOf` views at zero allocation cost:

  * **Elementwise** (`Scale`, `Merge`, `Residual`, `Relax`, `Interpolate`/
    `Sample`'s copies, `Sample`'s `accumulate_into`, `Accumulate`'s
    `accumulate_finalize`, `Extrapolate`'s order-0 copy) — thread-count-
    independent by construction: each output element's write touches no
    other, so no argument about associativity is needed. Two new additive
    kernels, `copy<Space>`/`scale<Space>`, cover what `kernels.hpp`'s
    pointer-based `copy` and a hand-written scale loop did; `Merge`'s Sum/
    Mean reuses the existing `accumulate<Space>` kernel once per input
    field transfer, in declaration order, which is the identical zero-then-
    accumulate-in-order the hand-written loop it replaces did.
  * **The remap apply** — `apply_weights`'s COO scan is now a CSR matvec
    (`space_kernels.hpp`'s `remap_apply`, a new multi-component overload for
    `components > 1`), with the CSR built ONCE per plan, in `resolve_remap()`
    (`build_remap_csr()`), never per coupling cycle. `RemapWeights::row`'s own header
    comment ("target entity, ascending") is NOT trusted as a universal
    invariant — `consistent_interp.hpp`'s `"consistent-transpose"` path
    transposes H's row-sorted triplets and does not re-sort the result, so
    `build_remap_csr()` STABLE-sorts by row itself rather than assuming;
    stability is what makes the CSR's per-row sum bit-identical to
    `apply_weights`' own ascending-k scan regardless of the input's original
    order (found by a real test failure at the base build configuration — `InterfaceTransfer.
    ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower` and two
    siblings — that a first cut of this fix, which DID assume `row` was
    always ascending, got wrong; see `include/fuse/backend/remap_plan.hpp`'s
    `RemapCsr`-adjacent comment in `runner_impl.hpp` for the argument).
  * **Reductions** — `iteration.cpp`'s block value/residual norms are the
    one place a plain `Kokkos::parallel_reduce` would have been WRONG: its
    own header comment says its summation order is the threads', which this
    runtime's bit-for-bit-across-thread-counts requirement cannot use. Three
    new kernels, `deterministic_sum`/`deterministic_dot`/
    `deterministic_norm_squared`(`_diff`), use a FIXED-CHUNK scheme instead:
    `Kokkos::parallel_for` over `ceil(n / 256)` chunks, each summing its own
    span sequentially into one scratch partial, then a plain host loop sums
    the partials in ascending chunk order — fixed by `n` alone, never by
    thread count. For `n <= 256` (every existing base-build-configuration implicit-coupling-group
    fixture) this is the identical computation as the single-accumulator sum
    it replaced, bit for bit; scratch is sized once, in `initialize()`, from
    the largest block any group has, and reused every sweep.

**A label-length trap, found and fixed the same way it would be found again.**
`Kokkos::parallel_for`/`parallel_reduce` copy their label into a
`std::string`; a label past this tree's libc++'s small-string-optimisation
capacity (measured: 22 characters) heap-allocates one PER CALL. The first cut
of the three `deterministic_*` kernels used self-describing 31-45 character
labels, and `runtime.iteration`'s two allocation-counted tests
(`Iteration.TheSweepLoopAllocatesNothing`/
`TheAcceleratorAllocatesNothingPerUpdate`) caught it immediately — 852 and 256
allocations where both expect 0. Fixed by shortening every label a
Runner-owned path reaches to stay under budget (`space_kernels.hpp`'s own
header comment records the number so a future kernel does not reintroduce
it).

**Verified, measured, this session:**

  * A full build at the base build configuration: 0 errors. A full `ctest` at the base build configuration: **136/136 (100%)**,
    including every `example.*.omp_identical` leg (`mms_pair`, `dn_heat`,
    `brusselator` — `fsi_channel` has never had one registered, unrelated to
    this closing) and `ctest -L omp` (29 legs, 100%). Getting the
    `omp_identical` legs to run at all also fixed a pre-existing
    `test/CMakeLists.txt` scoping bug: `FUSE_KOKKOS_HAS_OPENMP`/
    `_fuse_host_cores` were `set()` without `PARENT_SCOPE`, so
    `examples/CMakeLists.txt` — a sibling directory, not a child — never saw
    them and silently registered ZERO `omp_identical` tests for ANY example,
    including `brusselator`'s pre-existing, genuinely-Kokkos one. Every
    example whose `main()` calls into the Runner (`mms_pair`, `dn_heat`,
    `fsi_channel`, `gray_scott`) also gained its own `Kokkos::ScopeGuard`,
    matching `brusselator`'s and `layout_driver`'s: Kokkos `View`
    construction — even unmanaged — asserts `Kokkos::initialize()` has run,
    which none of those four needed before this closing.
  * A new typed test, `SpaceKernels.
    DeterministicReductionsMatchTheChunkedReferenceExactly`, compares against
    a plain-host reference implementing the identical chunked algorithm
    (adversarial 1e16/1.0/-1e16/1e-16-magnitude data) with `EXPECT_EQ` — no
    tolerance — at six sizes straddling the chunk boundary; registered with
    `fuse_add_omp_legs`. A new Runner-level test, `runtime.
    runner_determinism`, drives a real 2048-element implicit coupling group (large
    enough to span 8 fixed chunks) through `Relax`/`Residual` and pins the
    delivered field and the residual history to literals measured on this
    session's own passing run, also registered with `fuse_add_omp_legs`.
  * **Mutation-verified, by hand** (`tools/mutate.py` was not run against the
    tree; each mutation below was applied, built, run, and restored
    byte-for-byte via `diff`):
    - Replacing `deterministic_dot`'s fixed-chunk scheme with a plain
      `Kokkos::parallel_reduce` is killed by the new typed test on BOTH
      `Kokkos::OpenMP` and `Kokkos::Serial`, at every thread count — it is
      sensitive to any re-association, of which a thread-count difference is
      one case.
    - The same mutation applied to `deterministic_norm_squared_diff` is ALSO
      killed by `runtime.runner_determinism`, and demonstrates the actual
      defect this closing prevents: unmutated, `residual_history.back()`
      measures `1.8888945275990152` at `OMP_NUM_THREADS` 1, 8 and 12 alike
      (bit-identical); mutated, `OMP_NUM_THREADS=1` gives
      `1.8888945275990165` while `=12` happens to coincide with the
      unmutated literal — the mutation makes the answer DEPEND ON THREAD
      COUNT, which is exactly what this closing exists to prevent.
    - `runtime-scale-runs-serial-loop` (reverting `OpKind::Scale` to its old
      hand-written loop) is an HONEST NON-KILL: the full base-build-configuration suite still
      passed 100% with it applied. No test can distinguish a
      `Kokkos::parallel_for` over independent elements from a serial loop
      computing the identical products with identical rounding — there is no
      reduction to reassociate, so nothing about correctness, determinism,
      or the observable output differs, only which code path executed. The
      same is true of every other elementwise op this closing moved; Scale
      is the one instance named, not a special case among them. Recorded as
      a named gap (`build-evidence/mutations/transport-runtime.toml`), not a
      fabricated row: `tools/mutate.py`'s manifest loader requires
      `new`/`target`/`test` on every `[[mutation]]` row or aborts for the
      WHOLE tree, and there is no honest value to give those keys here.
  * `tools/check-parallel-matrix.py`: the `A-SPACES-TRANSPORT` gap is closed
    (70 claims verified, 0 broken, 1 gap remaining — `A-NODAL-HT-PARALLEL`,
    unrelated). `tools/mutate.py --check-anchors`: 179/179 valid (fixing one
    stale anchor this closing's own new `remap_apply` overload introduced by
    accidentally duplicating an existing mutation's anchor text verbatim).

**What is NOT claimed, and remains open under A5.** No device performance
number, and no device correctness number: this machine has no GPU. The
`deterministic_*` reductions' finalising sum reads `scratch` with plain host
code and are `static_assert`-guarded to a host-accessible memory space; a
device build needs a `deep_copy` of the partials to a host mirror first,
which nothing here implements. Moving the Runner onto a device execution
space, and the host/device mirrors a real coupled run would need at the
component boundary, is A5's own next step, not this closing's.

---

### ~~D12. `--coupler` on the processor-layout driver rejected unconditionally: routing through a coupler's own processor set was designed but not implemented~~ — CLOSED 2026-09-25

**Fix.** The coupler-placement track (design C,
`docs/dev/coupler-placement.md`): `ComponentKind::Coupler`, `add_coupler()`,
`RemapSpec::remap_on`. A field transfer's Remap now runs on any declared processor set
`C` (default: the target's own component) — value moves source -> C ->
target, remapped on `C`'s decomposition of both meshes, any hop between
identical processor sets elided exactly like every other hop. `C = target`
reproduces today's IR byte for byte (golden IR unchanged;
`CouplerLowering.CouplerEqualToTargetReproducesTodaysIRExactly`). Descriptor
bumped to v6 (a component's `kind` byte, a field transfer's `remap_on` word); YAML
gains a `couplers:` list and a field transfer's `remap_on:` key. New validation:
`E-COUPLER-FIELDS`, `E-CHANNEL-REMAP-ON`, `W-CHANNEL-REMAP-ON-UNUSED`.
`examples/layout_driver/main.cpp`'s `--coupler e:f` now builds and runs the
coupler-routed graph instead of rejecting; rejected only together with
`--mesh` (that mode builds no field-transfer graph at all).

**Verified at the base build configuration** (no MPI in this sandbox): every model/IR/schedule/
runtime unit test above passes unchanged (no regression when no coupler is
declared), 8 new `Validate.*` tests pin the validation rules, 7 new
`CouplerLowering.*`/`LoweringChannel.*` tests pin every lowering shape
(disjoint, remap-on-source, partial overlap, a coupler co-located with the
target under a distinct identity, the distributed-remap-with-coupler case,
and the "no possible remap" no-op case), and `Runner.
ACouplerNeedsNoComponentAndDeliversTheIdenticalValue` runs a serial (1
nominal-rank) end-to-end job and asserts the coupler-routed result equals
the direct result exactly (no tolerance) — see
`test/runtime/runner_test.cpp`.

**Compiled at the MPI-enabled build configuration, not run.** `fuse_example_layout_driver` builds cleanly
under `-DFUSE_ENABLE_MPI=ON` with the new code, and `layout.coupler.{3.
disjoint,3.equals-target,4.overlaps-comp1}` are registered
(`examples/CMakeLists.txt`) alongside the existing comp1/comp2 shapes.
`layout.refuse.coupler` (the old "unconditionally rejected" pin) is replaced
by `layout.refuse.coupler-with-mesh`.

**Host run 2026-09-27: all three `layout.coupler.*` rows FAILED**, each in
under a second, at `initialize` with `E-MESH-PLACEMENT`. The driver had told
its backend about two processor sets (`set_placements({comp1, comp2})`) and the
coupler was the third component, so `placed_view(mesh, coupler)` had no
decomposition to return. The library rejected correctly; the driver had not
kept the promise the coupler design makes ("a coupler is a component-array
entry like any other"). Fixed: the driver reads one processor set per graph
component, exactly as `runner_transport_test.cpp`'s harness does. That
harness is also why no gtest caught it — it already did the right thing, and
the base-build-configuration coupler tests use the serial backend, whose `placed_view` returns
one view for every processor set. Three gtests now run the coupler shapes on the
partitioned backend
(`RunnerTransport.{ADisjointCoupler,ACouplerOnTheTargetsRanks,ACouplerOverlappingTheSource}DeliversEveryEntity`,
matrix cell `A-COUPLER`). Both the rows and the gtests still need the host:

```sh
ctest --test-dir build-all -R 'layout.coupler|parallel.runner_transport' --output-on-failure
```

**Open, needing a host run to assess (`docs/dev/coupler-placement.md`'s own
section on it):** `resolve_remap()`'s MLS-serial and nodal-parallel-rejection
checks size themselves from the field transfer's declared source/target
components, not from the operator's actual processor set — for a coupler-routed
nodal remap where the coupler spans more than one rank while the field transfer's
declared endpoints are each single-rank, this could read the wrong rank
count. Not changed: the existing derivation was tuned through several rounds
of real host MPI debugging that this sandbox cannot reproduce or verify
against.

**Also open:** `src/runtime/iteration.cpp`'s implicit-coupling-group process
fix (a coupler-routed field transfer inside a group now unions the coupler's
ranks into `R.participants`) touches a `RunnerConfig::base_communicator`-
gated code path with no coverage at the base build configuration at all — genuinely needs MPI to
exercise.

---

### ~~D10. The conservation check is rank-local, so it rejects instead of checking in parallel~~ — FIXED 2026-09-25

**Fix (`54a27cc`).** `Constrain` is now collective (`fuse::is_collective`), so
a rank whose coupling cycle has already failed still reaches it instead of leaving its
partners inside a reduction; the two owned-share integrals are summed over the
base communicator in one `MPI_Allreduce` of three doubles — the third a flag
from any rank whose coupling cycle is already lost, so a real failure elsewhere is not
turned into a spurious conservation error on every partner. Totals, hence the
outcome, are identical on every rank with no further agreement. The rejection
`E-RT-CONSERVATION-MPI` and its fixture workaround are gone;
`Runner.AcceptsAConservationCheckOnAMultiRankTarget` pins the acceptance at
the base build configuration; `parallel.nodal_remap_runner` at 2–5 ranks exercises the reduction
against its own `MPI_Allreduce` of the same integrals (host outcome pending),
and mutations `conservation-check-not-reduced-across-ranks` (MPI) and
`constrain-classified-local` pin the two halves.

*Original entry follows.*


`Runner::Impl` checks a conservative field transfer by summing this rank's owned share
of the integral. With no cross-rank reduction that number is meaningless on a
multi-rank processor set — it would both false-fire on a correct cross-rank remap
and miss a real error — so the Runner rejects rather than report it
(`E-RT-CONSERVATION-MPI`, REVIEW.md N-transport-runtime-1). The rejection is
right; the missing capability is what matters.

**Found on the critical path 2026-09-25.** Every `parallel.nodal_remap_runner`
case at 2-5 ranks failed at `initialize` on this rejection, because its
`wall_force` field transfer is `.conservative()` and its target is placed on several
ranks. That fixture now sets `RunnerConfig::check_conservation = false` and
performs the same conservation statement itself with `MPI_Allreduce` — which is
exactly the check the Runner is declining to make.

**Why it is not merely cosmetic.** A conservative field transfer to a multi-rank target
is the normal case for FSI in parallel, and today every such job must either
disable the check or place the target on one rank. The interface load is the
quantity most worth checking.

**What it needs.** The reduction, over the target component's communicator,
which `Runner::Impl` already holds (it splits one per component from
`base_communicator`). The integral is a sum over owned entities, so the fix is
to reduce that sum and compare the totals on every rank, agreeing the outcome
the way `check_component_agreement` and `agree_on_local_failure` already do —
a rank-local comparison would reintroduce the disagreement class of defect
fixed in `27e55a1`. The rejection then applies only where no communicator is
available.

---

### D8. A Remap into a nodal field on more than one rank is rejected — PARTIALLY LIFTED (round 4): two real correctness defects found by a host MPI run are fixed; a real multi-rank host run is still the final outcome

`E-RT-REMAP-NODAL-PARALLEL` (`resolve_remap`, `src/runtime/runner.cpp`)
rejects any Remap whose target is a nodal field (ContinuousGLL order 1) on a
component placed on more than one rank, UNLESS the field transfer's method is
`"consistent"`/`"consistent-transpose"` (round 2, see below) — for every
other method (in particular the supermesh conservative default) it still
fires exactly as originally described here.
`InterfaceTransfer.ARemapIntoANodalFieldOnSeveralRanksViaASupermeshMethodStillRefuses`
makes it fire; `InterfaceTransfer.ARemapIntoANodalFieldOnTwoDeclaredRanksIsAcceptedAndMatchesOneRank`
is the SAME declared two-rank processor set, method `"consistent-transpose"`,
now accepted. Both conditions are read from the declared graph and
processor set, so every rank reaches the same outcome together.

**Why a rejection and not the owner sum that was first wired in.** A shared
vertex's rows on each holder are partial or already complete depending on the
decomposition the weights were built on. On a replicated source, which is
every case reachable today because MOAB gives every processor set the default
view (D5 item 2; the builders honour the processor set since 2026-09-25, but the
only backend with per-placement decompositions is the test one), every holder
computes the complete row. Summing
them onto the owner then multiplies the load by the holder count, silently. On
a genuinely partitioned source the rows are partial and the sum is right. No
test has run the partitioned case, and taking the owner's value for a pointwise
target is equally unverified there. The lead's first ruling assumed partial
rows without checking; the nodal-fields track found the replicated case.

**What stays.** Nodal Transport (the same field on both sides, with
lowest-rank vertex ownership), the shared-vertex primitives
`accumulate_shared_rows` and `copy_owner_shared_rows`, and their MPI tests in
`test/parallel/nodal_field_test.cpp`, whose outcomes come from a host run.
`reduce_nodal_remap_target()` stays compiled but is unreachable behind the
rejection: it is the path the rejection will open.

**What to do.** Build per-placement remap weights (D5 item 3). Then run a
genuinely partitioned, non-matching nodal Remap through the Runner at 1 to 5
ranks: global force conserved, H exact for a linear field, power balanced.
Lift the rejection only when that passes.

**Progress (2026-09-25, "Nodal remap" track).** `consistent_interp.hpp`'s
`build_consistent_interp_weights` is no longer 1D-only: a source of
topological dimension 2 or 3 (triangles, quads including a curved surface
like a cubed sphere, tets, hexes, and P1×P1 prisms) now dispatches to
`build_general_interp_weights` (`general_interp.hpp`), which locates each
target point's containing cell via `fuse::search`, inverts that cell's
reference map (a direct affine solve for a simplex, damped Gauss-Newton for
a genuinely multilinear quad/hex/prism), and reads off the weights —
verified serially (`test/backend/nodal_interp_test.cpp`, 22 cases: exact for
the shape's own affine/multilinear field on a single reference cell, order-2
convergence on non-matching real meshes for triangles and a cubed sphere,
H^T conservation and power balance, the lowest-source-global-id tie-break at
a shared face, and rejections for Pyramid/Polygon (`E-INTERP-SHAPE`, no
standard nodal basis), a missing `MeshView::cell_vertices` connectivity
answer, a degenerate cell, an out-of-domain point, an off-surface point, and
Newton non-convergence) — plus three of those rejection/tie-break rules pinned
by hand-applied, byte-restored mutations
(`build-evidence/mutations/nodal-remap.toml`).

**This directly answers D5 item 3's per-placement weights question for the
CONSISTENT methods specifically** (never for the supermesh conservative
methods, which since 2026-09-25 build on the processor set's view and are
otherwise unaffected): a
new, genuinely partitioned, non-replicated distributed primitive —
`build_distributed_interp_rows`/`apply_distributed_H`/`apply_distributed_HT`
(`nodal_remap_distributed.hpp`) — builds each evaluation entity's row ONLY on
the rank that owns it (an MPI_Allgatherv-replicated assembly of the source
mesh, then the same per-cell primitives the serial builder above already
validated; see that header's file comment for the explicit
robustness-over-efficiency trade this makes, and for why `H^T`'s cross-rank
sum is keyed by the CONTRIBUTING evaluation entity's global id rather than
by delivery order, which is what makes it bit-identical across rank counts
rather than merely correct at one). Compiled under `FUSE_ENABLE_MPI=ON`
(`test/parallel/nodal_remap_distributed_test.cpp`: disjoint, identical and
partially-overlapping rank sets, plus an out-of-domain rejection); UNRUN in
this sandbox (MPI cannot initialise here), so its outcome is a host action,
same as every other MPI-gated case in this file.

**Progress (2026-09-25, round 2, "Nodal remap" track): the Runner IS now
wired, and the null-comm path made it locally verifiable.** Three things
changed from round 1's "not wired, no local MPI outcome possible" state:

1. **Two-tier containment, general_interp.hpp.** Round 1's single loose
   tolerance (`kRefEps = 0.02`, one boolean "close enough" test feeding a
   lowest-source-global-id tie-break among every candidate that passed it)
   let a lower-global-id NEIGHBOUR's cell win a shared-boundary point by
   EXTRAPOLATING past its own edge, invisible to an affine-field test since
   an affine field cannot tell an interpolation from an extrapolation apart.
   Replaced by two tiers: `kRefEpsTight = 1e3 * kNewtonTol` (a genuine,
   rounding-scale containment; wins outright, tie-broken by lowest global id
   ONLY among tight matches) and `kRefEpsLoose = kSurfaceTol` (a single
   least-violating fallback, no tie-break, accepted only when it ALSO clears
   the same physically-derived bound the existing `kSurfaceTol` uses for the
   curved-surface faceting gap along the other axis). New regression test
   `GeneralInterp.PointDeepInsideBIsNotStolenByALowerGidNeighbourExtrapolating`
   (two triangles, a field with `v11 != v10+v01-v00` so an extrapolation is
   visibly wrong, not accidentally exact) fails under the old rule and passes
   under the new one; a new "tier (a) skipped" mutation
   (`build-evidence/mutations/nodal-remap.toml`) reproduces the old bug by
   widening tier (a)'s own threshold to `kRefEpsLoose`, confirmed KILLED by
   exactly that test. **Measured convergence orders, verbatim, after this
   fix** (`TriangleMeshOrderIsTwoUnderRefinement`,
   `SphereSurfaceOrderIsAtLeastTwoUnderRefinement`; both gated at `>= 1.8`,
   tightened from round 1's `1.7`/`1.5`): triangle mesh **1.955988**, cubed-
   sphere surface **1.914458** — both comfortably above the tightened gate,
   and both HIGHER than round 1's own measurements at the old (looser,
   tie-break-only) rule, direct evidence that the mis-assigned-cell bug was
   actually costing real accuracy, not merely a theoretical concern.
2. **A null-communicator (base build configuration) code path, nodal_remap_distributed.{hpp,cpp}.**
   The file's blanket `#if FUSE_HAVE_MPI` guard (round 1) is gone; only the
   one real MPI call site (`allgatherv_records`) branches on
   `comm_handle == fuse::search::kNullComm` (a local no-op copy, touching no
   MPI symbol) vs. a real communicator (unchanged, `FUSE_HAVE_MPI`-gated,
   rejects `E-RT-NODAL-DIST-NOMPI` otherwise) — the same convention
   `fuse::partition::detail::collectives.hpp` already established. The
   serial two-tier fix above was ALSO applied to this file's own
   (previously independently-tolerant) selection loop, since the same bug
   shape applies to a replicated-mesh candidate list exactly as it does to a
   local one. Proven by a new, PLAIN (`fuse_add_gtest`, no MPI linked)
   target, `backend.nodal_remap_distributed_nullcomm`, which actually RUNS
   in this sandbox (unlike `parallel.nodal_remap_distributed`, still
   compile-only here) — 3/3 passing, plus a new KILLED mutation
   (`build-evidence/mutations/nodal-remap-distributed.toml`) proving the
   `kNullComm` guard itself is load-bearing, not decoration.
3. **The Runner wiring itself, `src/runtime/runner.cpp`.** `resolve_remap`'s
   nodal-target check now branches on `Instruction::attrs.method`:
   `"consistent"`/`"consistent-transpose"` call the new
   `Runner::Impl::build_nodal_dist_remap` (builds one `DistributedInterpRows`
   at setup, via `comm_handle = *config.base_communicator` when set, else
   `fuse::search::kNullComm`) instead of rejecting; every other method still
   rejects `E-RT-REMAP-NODAL-PARALLEL` exactly as before (per-placement
   supermesh weights remain unimplemented, D5 item 3). `execute()`'s
   `OpKind::Remap` case calls the new `apply_nodal_dist_remap` per coupling cycle,
   allocation-free of the ROWS themselves (the `Allgatherv` buffers
   `apply_distributed_H`/`HT` build are a separate, per-call cost — see
   "Known scalability limitation" below). **This is a REAL Runner run, not a
   compile check**: `InterfaceTransfer.ARemapIntoANodalFieldOnTwoDeclaredRanksIsAcceptedAndMatchesOneRank`
   runs the exact field-transfer pair `ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower`
   already covers at one declared rank, a SECOND time at a DECLARED
   two-rank processor set (no real second process; `comm_handle` is
   `kNullComm` since `base_communicator` is unset, so this single process
   already owns everything, exactly the null-comm case item 2 proved), and
   checks the two-rank run reproduces the one-rank run's numbers exactly (H
   exact for a linear field, H^T conserves the total force, interface power
   balances) — not merely that it runs without throwing. Two new KILLED
   mutations (`build-evidence/mutations/nodal-remap-runner.toml`) pin the
   method-string dispatch and the transpose density-to-load scaling
   decision. The full base-build-configuration suite (101/101, including this new test and
   every previously-passing case) and the full MPI-enabled build configuration build
   both pass/compile with these changes; `check-invariants.py`/
   `check-layering.py`/`mutate.py --check-anchors` all report zero
   violations (160/160 anchors valid).

**`reduce_nodal_remap_target()`/`NodalReduceInfo`/`NodalReduceMode`/
`Impl::nodal_reductions` are DELETED, not merely left unreachable.** They
were genuinely dead code even before this round: `resolve_remap`'s own
`nodal_reduce_idx` local was set to a real index only inside a branch that
unconditionally returned a `Failure` first, so `nodal_reductions` was always
empty and `reduce_nodal_remap_target` was never reached by anything. Round 2
chose the Allgatherv-replicated design (item 3 above) over the design that
scaffolding anticipated ("every holder ends up with the owner's total,"
reusing `accumulate_shared_rows`/`copy_owner_shared_rows`); since a
decision was made, the unreachable alternative is removed rather than kept
as a second, unused implementation of the same feature. `accumulate_shared_rows`
and `copy_owner_shared_rows` (`fuse/transport/comm_graph.hpp`) themselves are
UNTOUCHED — they are general-purpose primitives with their own direct
callers in `test/parallel/nodal_field_test.cpp` (the "nodal fields" track's
Transport-side ownership tests), not exclusive to the deleted function.

**Known scalability limitation, disclosed not hidden: every
`build_nodal_dist_remap`/`apply_distributed_H`/`apply_distributed_HT` call
replicates the WHOLE source (or evaluation) mesh's geometry across every
rank via `MPI_Allgatherv`,** and every coupling cycle's `apply` re-gathers the
field VALUES the same way. This does not scale to a source mesh too large
to fit in one rank's memory, and is a documented, deliberate
robustness-over-efficiency trade (this project's own standing priority),
not an oversight — see `nodal_remap_distributed.hpp`'s file header for the
full reasoning. The planned replacement is a directory-based, point-to-point
query via `fuse::search::query_points_distributed` (already built,
currently unused by this path) for the ROW-LOCATION step specifically,
exchanging only what each pair of ranks actually needs instead of
broadcasting everything to everyone. **The current Allgatherv-based
implementation's results should be KEPT as the oracle the future scalable
implementation is tested against** — every exactness/conservation property
this round's tests check is a property of the MATH (H's rows sum to one;
H^T D's total is conserved), not of the Allgatherv mechanism, so a future
point-to-point implementation must reproduce these same numbers exactly,
not merely "also pass its own tests."

**Progress (2026-09-25, round 3): the missing Runner-level MPI test now
exists, compiled but unrun.** Round 2's own gap (b) below is closed:
`test/support/partitioned_test_backend.hpp` gained
`MultiMeshGeometricBackend`, a small composite that holds one
(`n_cells`, `RankSet`, `x_lo`, `x_hi`) `Spec` PER MESH NAME (registered
before `load()`), reusing `GeometricLineMesh` unchanged for each -- exactly
what closes the "`Runner` needs one `MeshBackend` but the two test backends
available each bake in a single mesh's shape" gap. `test/parallel/
nodal_remap_runner_test.cpp` (`NodalRemapRunnerMpi.{DisjointRankSets,
IdenticalRankSets, PartiallyOverlappingRankSets}`) drives a real beam/fluid
`"consistent"`/`"consistent-transpose"` field-transfer pair through
`lower -> schedule -> initialize -> run_period` on that backend, at 1-5
ranks, checking H's exactness pointwise, global force conservation and
power balance to `1e-13` relative, and -- the property this file's own name
promises -- each rank's `beam_load` entry against a ONE-RANK REFERENCE
computed IN THE SAME TEST (via `build_consistent_interp_weights` on an
unpartitioned `GeometricLineMesh`, not a second job launch), which is the
standard way this codebase already proves "bit-identical across rank
counts" without two separate compared launches (see
`nodal_remap_distributed_test.cpp`'s own header for the identical
argument). That reference was cross-checked in this sandbox, WITHOUT real
MPI, against `apply_distributed_HT` itself at `comm_handle ==
fuse::search::kNullComm`: the two agree to `1.4e-17` (round-off) at every
one of 8 beam vertices for `n_beam=7, n_fluid=11` -- strong evidence the
new test's own oracle is correct and consistent with the code path it
exercises, even though the test itself still cannot be RUN here (MPI
cannot initialise in this sandbox, same posture as every other MPI-gated
case).

**The two NOT-ASSESSED mutations from round 2 were reconsidered, not
mechanically retargeted.** Retargeting them at the new Runner test would
not have made them assessable: the "row built on every holder" row needs a
genuinely SHARED evaluation-side entity, and in both the primitive-level
test and the new Runner test the evaluation side is always the FLUID
mesh -- cell-based, never ownership-shared in `GeometricLineMesh`'s design
(only vertices, always the nodal side here, are ever shared). A real target
for that row needs a node-to-node `"consistent"` field transfer (both sides
nodal), not built this round since it is not the shape "H/H^T as the FSI
uses them" needs. The "transpose sums in rank order" row remains
undecidable by ANY single job launch at a fixed rank count, Runner-level or
not -- `build-evidence/mutations/nodal-remap-distributed.toml` now names
both the primitive-level and the new Runner-level tests as equally valid
surfaces for the two-launch (e.g. rank counts 2 and 4) comparison a host
would run to actually kill it.

**A genuine defect was found and precisely quantified: `apply_nodal_dist_remap`
does NOT allocate zero per coupling cycle.** `InterfaceTransfer.
ApplyNodalDistRemapHasAConstantMeasuredPerPeriodAllocationCount`
(interface_transfer_test.cpp) extends the SAME `AllocationCounter`
instrument `Runner.AllocatesNothingWhileRunning` (runner_test.cpp) uses --
not by editing that shared, heavily-reused harness (which has no
nodal-target-on-several-ranks case to extend), but with a second, focused
test next to it, since teaching the existing harness a new mesh/field/
field-transfer shape was a materially bigger and riskier change than one more
self-contained test. Measured, verbatim: **31 allocations per coupling cycle**,
constant across all 8 measured coupling cycles (no growth, no leak) -- NOT the
zero every other Remap kind achieves. The cause is not a bug in the Runner
wiring itself but the ALREADY-DISCLOSED cost this same D8 entry named in
round 2: `apply_distributed_H`/`apply_distributed_HT`
(`nodal_remap_distributed.hpp`) build fresh `Allgatherv` buffers, an
`unordered_map`, and (in the `H` branch) a copy of the local input slot on
EVERY call, by design, for a scalability trade this track has been explicit
about since round 1. This measurement turns that prior qualitative
disclosure into a concrete number future work can compare against: a
point-to-point replacement (see the "Known scalability limitation"
paragraph above) should be expected to reduce this count, and this test's
own assertion (a CONSTANT count across coupling cycles, not a zero one) will keep
catching a regression that makes it grow or leak even after that
replacement lands.

**Progress (2026-09-25, round 4): a real host MPI run found two genuine
correctness defects in round 2/3's own wiring, both fixed here, plus a
test-authoring bug in round 3's own new test.**

1. **Defect 1 — a rank-local rejection reachable inside a collective
   function (`nodal_remap_distributed.cpp`).** `build_distributed_interp_rows`
   returned `E-INTERP-DOMAIN`/`E-INTERP-NEWTON`/`E-INTERP-DEGENERATE`
   directly from inside its per-local-evaluation-entity loop, AFTER that
   same function's own `allgatherv_records` collective
   (`gather_global_source`). Since which entities are local differs per
   rank, one rank could hit the rejection while a DIFFERENT rank, whose own
   local points were all fine, returned success -- exactly the shape the
   host's own `parallel.nodal_remap_distributed.3/4/5` runs caught
   (`NodalRemapDistributed.TargetOutsideSourceDomainRefuses`:
   `rows_r.ok()` true where false was expected, at 3+ ranks; 1-2 ranks
   passed, masking it). **Fix:** a new helper,
   `agree_on_local_failure(comm_handle, local_failed, local_code_id,
   local_entity_gid, function_name)`, Allgathers a fixed 4-double-per-rank
   record (so a rank with ZERO local evaluation entities still contributes
   exactly one record and takes part in the agreement) and returns the
   SAME `Failure` -- same code, same message, naming the LOWEST failing
   rank and that entity's GLOBAL id -- on every rank when ANY rank failed,
   or success when none did. The per-entity loop now records-and-stops on
   its OWN first local failure instead of returning; the agreement call
   happens once, after the loop, unconditionally. The identical hazard,
   found independently (not named by the host report) by auditing every
   other post-collective early return in the same file, existed at
   `apply_distributed_H`'s own `E-RT-NODAL-DIST-OWNER` check (line ~411,
   after ITS OWN Allgatherv) and got the same fix; `apply_distributed_HT`
   was audited too and has no such early return (its post-Allgatherv code
   is pure deterministic sort/sum/copy) -- no fix needed there.
   **Verification available in this sandbox:** the null-communicator
   suite (`backend.nodal_remap_distributed_nullcomm`, 3/3) still passes,
   confirming the single-participant case is unchanged; a standalone
   diagnostic, compiled directly against this file's own
   `allgatherv_records`/`fail_code_str` (not a re-implementation) and fed
   hand-built 3- and 4-fake-rank record sets, confirmed the AGREEMENT
   DECISION logic itself picks the lowest failing rank's code/global id
   correctly and agrees "success" when none fail -- proving the rule
   without a real multi-process launch. **What still needs the host:** a
   genuine 3+-rank run of `parallel.nodal_remap_distributed` (see the exact
   command below) is the only way to observe every rank AGREEING, since
   `kNullComm`'s one participant has nothing to disagree with by
   construction -- confirmed empirically, not just argued: the new
   "domain outcome is not agreed" mutation
   (`build-evidence/mutations/nodal-remap-distributed.toml`) was applied
   and every base-build-configuration test, including the null-comm suite, still passed
   unchanged, exactly as expected for a mutation whose effect requires a
   SECOND rank to exist at all.

2. **Defect 2 — the nodal-SOURCE, cell-TARGET direction had no distributed
   path and no rejection (`runner.cpp`).** `resolve_remap`'s distributed
   trigger read only the REMAP TARGET's kind
   (`region_extent_kind(graph, tgt) == Vertex`), so a `"consistent"`
   field transfer whose SOURCE is nodal and target is cells -- exactly
   `wall_velocity` in the FSI field-transfer pair -- never took the distributed
   path regardless of either side's declared rank count, and fell through
   silently to the old rank-local `build_consistent_interp_weights`: on a
   rank owning zero local cells of the (cell) target this fails
   (`E-INTERP-NOGEOMETRY`, the host's own observed message on ALL FIVE
   `parallel.nodal_remap_runner.N`, every rank count); on a rank owning
   some cells it would have built weights against only that rank's own
   local view of the source, silently, with no diagnostic. **Fix:** the
   trigger now reads BOTH mesh entity sets' kind and BOTH components' declared rank
   count (`src_ranks`/`dst_ranks`), and fires whenever either side is
   nodal and either component is on more than one rank.
   `Runner::Impl::build_nodal_dist_remap`'s existing `!transpose` branch
   already builds exactly this orientation correctly (`nodal_region =
   src`, `eval_region = tgt`) -- the bug was purely in the trigger
   condition, not in that function. A method paired with the WRONG
   orientation (e.g. `"consistent-transpose"` with a nodal SOURCE) is
   rejected by name (`E-RT-REMAP-NODAL-PARALLEL`), as is a field transfer with
   BOTH sides nodal (`InterfaceTransfer.BothSidesNodalOnMultipleRanksRefuses`,
   new this round) -- neither `build_nodal_dist_remap` orientation has
   ever been run against a nodal-to-nodal field transfer, so it is rejected rather
   than silently picking one convention. A rank with zero local evaluation
   entities already contributed zero rows and still took part in every
   collective before this fix (`build_distributed_interp_rows`'s loop
   bound is `entity_count`, and `apply_distributed_H`/`HT` both call their
   own Allgatherv unconditionally regardless of row count) -- re-confirmed
   after Defect 1's agreement step was added, since that call is
   unconditional too. **Verified in this sandbox** (per the review's own
   instruction: this is exactly what a DECLARED multi-rank processor set at
   `comm_handle == kNullComm` can test without real MPI):
   `InterfaceTransfer.ARemapIntoANodalFieldOnTwoDeclaredRanksIsAcceptedAndMatchesOneRank`
   -- unmodified from round 2, since `beam_ranks{0, 2}` governs BOTH
   field transfers' nodal side already -- now ALSO exercises the newly-fixed
   nodal-source orientation through the SAME declared processor set (its own
   `fluid_in` closed-form check is H's correctness proof for that
   direction); `InterfaceTransfer.
   ApplyNodalDistRemapHasAConstantMeasuredPerPeriodAllocationCount`'s
   measured count rose from round 3's 31 to **50 allocations per coupling cycle**
   (still constant across coupling cycles), the expected increase now that BOTH
   `wall_velocity` (H) and `wall_force` (H^T D) route through the
   Allgatherv-based distributed builder instead of one. Full base-build-configuration ctest
   (101/101, ~210s), full MPI-enabled build configuration build, `check-invariants.py`/
   `check-layering.py` (0 violations) and `mutate.py --check-anchors`
   (164/164 valid) all reconfirmed after this fix.

3. **Defect 3, smaller — the new Runner test's own rank arithmetic
   (`nodal_remap_runner_test.cpp`).** `DisjointRankSets`'
   `fluid_ranks{(w+1)/2, w-(w+1)/2}` and `PartiallyOverlappingRankSets`'
   `beam_ranks{0, w-1}`/`fluid_ranks{1, w-1}` place a component on NO rank
   at all below a minimum world size -- the same two-sided-
   processor-set-on-one-rank shape commit 2383024 fixed elsewhere. Ad hoc
   `GTEST_SKIP` guards already caught this at the exact rank counts where
   it would otherwise misbehave; round 4 replaced them with
   `FUSE_REQUIRE_AT_LEAST_RANKS(2)`/`FUSE_REQUIRE_AT_LEAST_RANKS(3)`
   (`test/parallel/runner_transport_test.cpp`'s own convention), which
   states the requirement the same way every other multi-rank-shape test
   in this codebase does. `IdenticalRankSets`' own `RankSet{0,
   world_size()}` is never empty, so this cause does not explain the
   host's separately-reported rank-1 failure there; reproducing it needs a
   genuine MPI launch, which this sandbox cannot provide even at `-np 1`
   (`mpirun -np 1` fails in `HYDU_sock_listen`; running the built MPI test
   binary directly, without `mpirun`, aborts inside `MPI_Init`'s own
   debugger-attach teardown, `dbginit.c`'s `MPIR_Debugger_set_aborting`)
   -- confirmed by trying both, not assumed. That specific failure's
   actual message is a host action; see the command below.

**What is still open, honestly.** (a) No REAL multi-process (2+ actual
ranks) MPI host run has verified this Runner wiring or the round-4 fixes
above -- the null-comm (base build configuration) case IS a real run and IS the right
physics for a single participant, but it structurally cannot exercise
cross-rank AGREEMENT (nothing to agree ON with one voice) or a genuine
partition. The exact commands a host should run:
```
ctest --test-dir <mpi-build-dir> -R 'parallel\.nodal_remap_distributed' -V
ctest --test-dir <mpi-build-dir> -R 'parallel\.nodal_remap_runner' -V
```
naming, for `nodal_remap_runner`, `IdenticalRankSets`' actual rank-1
message specifically (defect 3 above). (b)
`InterfaceDomainPolicy::ClampToBoundary` remains unimplemented in the
distributed builder (unchanged from round 1). (c) The 50-allocations-per-coupling-cycle
measurement (round 4, was 31 in round 3) is a number to reduce, not (yet)
a fix -- no attempt was made to eliminate it, since doing so needs a
scratch-buffer-reusing redesign of `apply_distributed_H`/
`apply_distributed_HT`'s own, already-tested API, a change of a different
size and risk than measuring the existing behavior precisely. (d) The
"domain outcome is not agreed across ranks" mutation
(`build-evidence/mutations/nodal-remap-distributed.toml`) is NOT-ASSESSED
in this sandbox for the same single-participant reason as (a) -- a host
3+-rank run applying it on top of the fix is the only way to KILL it.
**Lift the rejection for the remaining methods only after (a) and D5 item 3
(per-placement supermesh weights) both land.**

**Host run 2026-09-25 (third): `parallel.nodal_remap_runner` still fails at 1-5 ranks.** The two
defects above are fixed and `parallel.nodal_remap_distributed` now passes, so this is a third,
separate cause, and it is NOT any of:
- the graph, the field transfers, or the two remap methods;
- `MultiMeshGeometricBackend`'s own views, geometry or global ids;
- the H and H^T D weights, or the transposed load;
- force conservation or interface power balance.

All of those are now covered by `verification.nodal_remap_runner_serial`, which drives the SAME
graph and the SAME backend as the MPI fixture at a single participant, with the MPI fixture's own
parameters (7/11 and 8/6) and its own serial reference, and passes to 1e-12. Since the failing
`.1` case is also a single participant, whatever remains differs only in the MPI-configured path:
`RunnerConfig::base_communicator` being set, and the `FUSE_HAVE_MPI` branches that follow from it
(`Runner::Impl`'s per-component `MPI_Comm_split`, the transport setup, and the epoch agreement).

`test/support/partitioned_test_backend.hpp` no longer needs MPI, which is what made that
bisection possible: its four `MPI_Comm_rank` calls became one `fuse_test::this_rank()` that
returns 0 without MPI, since at a single participant every declared processor set is the whole mesh.

**What this still needs from a host**, because MPI cannot initialise in the sandbox at all (not
even `-np 1`: `mpirun` fails in `HYDU_sock_listen`, and the binary run directly aborts inside
`MPI_Init`):

    ctest --preset <mpi-preset> -R 'parallel\.nodal_remap_runner\.1' --output-on-failure

One rank is enough, and the failing assertion's own message names the stage.


### ~~D9. The coupled channel-beam FSI gains energy over many periods~~ — FIXED 2026-09-25: the traction had the sign of the force on the fluid

**Root cause.** `StokesSolver::step()` delivered `-p + 2 mu dv/dy` as the wall
traction. That is Cauchy's `sigma . n` with the fluid's outward normal — the
force the *wall exerts on the fluid*. The force on the wall is its negative,
`+p - 2 mu dv/dy`; a pressurised channel pushes its wall outward. With the
wrong sign the fluid handed the beam a **negative added mass**, which is
exactly the observed signature: one real eigenvalue above 1 (1.115466 on the
diagnostic grid, energy ×1.228 per step), a pressure-plus-first-bending-mode
eigenvector, and insensitivity to every boundary detail four rounds of energy
budgets varied — because every one of those checks compared the delivered
vector with itself. Power balance, adjointness and the closed budgets were all
correct statements about a consistently wrong-signed force.

**Why no test saw it.** The steady-deflection test compared the beam against
the analytic solution *under whatever load the code produced*, with the same
sign built into its analytic load (`q1 = -dpdx`). Nothing stated the physics.

**Fix and evidence (`stokes_solver.hpp`, `fsi_coupled_test.cpp`,
`fsi_energy_diag_test.cpp`, `fsi_transient_test.cpp`).**
- Coupled step-map spectral radius: **1.115466 → 1.000000**; the |λ| = 1 mode
  is Newmark's acceleration passenger (`wddot_{n+1} = 2(v_{n+1}-v_n)/dt -
  wddot_n`, read by nothing), and carries no energy: E0 = 6.3e-21 at |x| = 1
  against an eigenvector residual of 7.7e-11, i.e. eig_quality².
  `CoupledStepMapRevealsTheGrowthMechanism` now asserts rho ≤ 1 + 1e-6.
- Long-run energy, 12 beam periods at dt = T1/40: geometric-mean per-step
  ratio **1.22796 → 0.998067**. `LongRunEnergyIsBounded` enabled with its
  original bound.
- Steady deflection order **1.743 → 1.923**, errors 1.48e-4, 3.97e-5, 1.03e-5.
- Two physics tests that fail on the old sign and cannot be satisfied by a
  self-consistent wrong sign: `APressurisedChannelPushesItsWallAwayFromTheFluid`
  and `AcceleratingTheWallAwayFromTheFluidIsOpposedByTheTraction`. Mutation
  `fsi-traction-is-the-force-on-the-fluid` reproduces the defect exactly.
- Transient self-convergence: enabled. Two things had been conflated in its
  old 0.82: the growth, and the fixture's **impulsive start** (fluid at rest,
  inflow switched on at t = 0 — not second-order data for an incompressible
  system; with the sign fixed alone the five-level regression over
  T1/40 … T1/640 still measures 0.87, and the three coarsest levels kept as
  `ImpulsiveStartIsMeasuredNotAsserted` give 0.15). From a smooth start — fluid at its
  exact Poiseuille state, beam at rest and displaced off equilibrium in its
  first-mode shape — at T1/160 … T1/2560 the order is **1.846** (pairwise
  1.849, 1.871, 1.809), inside the unchanged [1.8, 2.2] bound. It is flat at
  ~1.85 rather than trending to 2, and that gap is **not explained**; recorded
  as such. Levels T1/40 and T1/80 are pre-asymptotic (pairwise 0.29, 1.33).

The beam quadrature inconsistency round 4 found (`strain_energy()` missing a
`dx_beam` factor relative to `kinetic_energy()`) remains open and separate; it
never entered the state map.

**Host run 2026-09-25 15:57:** every FSI suite passes on the host — `fsi_transient` (254 s, order
1.846 inside the bound), `fsi_energy_diag` (115 s), `fsi_coupled`, `fsi_stokes_mms`, `fsi_beam`.

*Original entry, four rounds of diagnosis, follows.*


`examples/fsi_channel`: 2D unsteady Stokes (CN, MAC grid) and a clamped
Euler-Bernoulli beam (FD, Newmark average acceleration), strongly coupled
through consistent H and conservative Hᵀ D on a non-matching wall.

Every window converges tightly, so the converged solution is the monolithic
solution of the combined time discretisation. The growth therefore belongs to
that combined scheme, not to the coupling iteration. Each half alone is bounded:
- the beam alone, over 80 periods: Emax/E0 = 1.0000;
- the fluid alone under a prescribed periodic wall velocity, over 40 periods:
  bounded.

**Found and fixed on the way (earlier rounds):**
- the Crank-Nicolson pressure gradient was fully implicit against a trapezoidal
  viscous term, so the wall traction was p(t_{n+½}) labelled as p(t₁). That
  caused a sign-alternating growth of |ratio| 2.48 per coupling cycle. The pressure now
  converges at order 2.0013 against p(t₁), and the alternation is gone.
- the wall-velocity boundary condition used the t₁ value for both Crank-Nicolson
  levels.

**Residual, unresolved:** at mass ratio about 2.5 (h_s = 0.5, rho_f = 1,
H = 0.2) with IQN-ILS and dt = T₁/40, the total energy grows by a constant
1.22796 per step, the same to four figures across periods 3 to 12. The dt
self-convergence order is 0.822 (mid-span deflection 0.018725, 0.019143,
0.019390, 0.019533, 0.019608 m at 10 to 160 steps).
`FsiTransient.DISABLED_SelfConvergenceInDtIsSecondOrderInTheAsymptoticRegime`
keeps the unchanged bound.

**This round: the mechanism is now precisely located, via the lead's
linear-algebra method (`test/verification/fsi_energy_diag_test.cpp`).** One
converged coupled step is an exact affine map x_{n+1} = M x_n + b over the
full carried state (fluid u/v/p, the beam's w/wdot/wddot, and
`FluidComponent`'s own checkpointed `velocity_in_t0_`), because every
boundary datum here is time-independent and the coupling is a fixed linear
remap. Built column-by-column by perturbing each state entry (finite
difference is exact here, confirmed by two step sizes agreeing to
round-off):

- **rho(M) = 1.108145 on the production grid** (nx_beam=8, nx_fluid=10,
  ny_fluid=6) — rho² = 1.227985, matching the measured 1.22796 energy ratio
  to ~4 significant figures. **rho(M) = 1.115467-1.115908 on the smaller
  diagnostic grid** (nx_beam=6, nx_fluid=5, ny_fluid=3, the grid the
  COMMITTED `CoupledStepMapRevealsTheGrowthMechanism` test actually runs,
  for runtime) — rho² = 1.244-1.245, agreeing with 1.22796 only to ~2
  significant figures at this coarser resolution, though the same
  qualitative mechanism (a real, non-oscillatory eigenvalue dominated by
  the same pressure-and-beam-mode eigenvector) holds at both grids. Either
  way this is a genuine, real (not oscillatory) eigenvalue, confirming the
  growth is a scheme instability, not a coupling-iteration artefact
  (independently re-confirming what "every window converges tightly"
  already implied, this time by direct linear algebra rather than by
  absence of a symptom).
- **Beam alone and fluid alone are both bounded by the SAME method**:
  `FsiEnergyDiag.BeamAloneStepMapSpectralRadiusIsAtMostOne` measures
  rho=1.0000000060 (undamped Newmark, marginal as expected);
  `FsiEnergyDiag.FluidAloneStepMapSpectralRadiusIsAtMostOne` measures
  rho=1.0000000000 exactly, for a fixed (state-independent) wall velocity.
- **The dominant eigenvector is ~85-96% pressure**, ~4-15% the beam's own
  first bending mode (wddot), negligible everywhere else (beam w/wdot,
  fluid u/v, velocity_in_t0 all under 1%).
- **Its velocity field is exactly divergence-free everywhere except the
  outflow column**, where the local divergence is ~0.02-0.03 (vs 0 to
  round-off elsewhere) — a direct, measured footprint of the pressure-
  outlet's own documented trade-off (`cn_step`'s "PRESSURE GAUGE" comment,
  stokes_solver.hpp): replacing the outflow column's continuity equation
  with `p=0` at every row kills the discrete operator's second null
  direction (SVD-confirmed in an earlier round) but leaves a local mass
  source there with nothing damping it.
- **A global mass-conservation correction was added** (`StokesSolver::step`,
  a single uniform shift of the outflow column's own `u` so
  `Q_out = Q_in + Q_wall` exactly every step — the "transpiration volume
  source vs. pressure-outlet reference" check this task's own instructions
  named). It is real, harmless, and verified: all of `fsi_coupled_test.cpp`
  passes with IDENTICAL numbers (order 1.743205, power balance to 1e-12,
  same sweep counts), `fsi_stokes_mms_test.cpp` and `fsi_beam_test.cpp` are
  untouched (the fix lives in `StokesSolver::step`, never in `cn_step`
  itself, which the MMS suite calls directly). **It does NOT fix D9**:
  rho(M) moves by under 1% (1.115908 → 1.115467) and the long-run energy
  ratio is unchanged outside noise (1.22796 → 1.226473,
  `FsiEnergyDiag.DISABLED_LongRunEnergyGrowthApproachesBoundedAfterAFix`,
  left disabled with this number, not loosened).
- **Direct ablation isolates the channel, not the mechanism**: zeroing
  `wall_traction()`'s pressure term (viscous only) drops rho to 1.0005 —
  essentially marginal, a complex-conjugate pair, 81% beam-wddot / 19%
  pressure. Zeroing the viscous term instead (pressure only) leaves rho
  essentially unchanged (1.1157). So the pressure channel of the interface
  force carries the instability — but this is not itself a usable fix
  (pressure legitimately belongs in a wall traction).
- **Two further candidate fixes were tried and ruled out**: the order of
  `wall_traction()`'s wall-pressure extrapolation (2nd-order vs. nearest-
  cell) makes no measurable difference (rho 1.1159 → 1.1171); trapezoidally
  averaging the DELIVERED wall pressure across the t0/t1 step boundary
  (mirroring the momentum equation's own p0/p1 mixing) makes things
  dramatically WORSE, reopening the exact sign-alternating |lambda|≈3.27
  Nyquist instability the earlier round's trapezoidal-pressure-GRADIENT fix
  removed — confirming that fix is correct and this residual mode is a
  genuinely different mechanism, not a residue of the same one.

**Round 2: the leading hypothesis was tested directly, is confirmed correct
in its own right, and does NOT fix D9.** The lead's hypothesis: with the
trapezoidal pressure GRADIENT (the earlier round's fix, described above),
CN's momentum equation pins only p_mid = (p0+p1)/2; carrying p0 as literal
state makes p1 = 2 p_mid - p0 a free, |lambda|=1 parasitic direction, and
the wall traction delivered to the beam (built from p1) closes a feedback
loop through exactly that direction. Prescribed fix: make the fluid's
pressure gradient fully implicit again (no p0 carried — the solved
`p_field_` genuinely IS p_mid), deliver `wall_traction()` as that mid-step
value (viscous part trapezoidally averaged over t0/t1, matching CN's own
convention), and have the beam consume it AS a mid-step load directly
rather than averaging two endpoint forces a second time — a new
`BeamSolver::step_midpoint_load()` derived from averaging the beam's own
equation of motion at n and n+1 (M ddot{w}_avg + K w_avg = q_mid, exact
algebraic identity, confirmed to round-off after solving).

Implemented exactly as prescribed (`StokesSolver::step`'s `cn_step` call
reverted to `p0=nullptr`; `wall_traction()` now a cached mid-step accessor,
`traction_mid_`, carried through `save`/`restore`/`set_state`;
`BeamComponent::advance` now calls `step_midpoint_load`). Three independent
lines of evidence confirm the fix is CORRECTLY implemented, exactly as
derived:
- `FsiEnergyDiag.FluidAloneStepMapSpectralRadiusIsAtMostOne` moves from
  rho=1.0000000000 (exactly marginal — the parasitic mode) to
  rho=0.9946298894 (genuinely damped) — the targeted mode is gone.
- The 73×73 coupled step map's "fluid p" input column norm is EXACTLY zero
  (avg and max, measured via finite-difference perturbation of every
  carried-state entry) — perturbing p_field_ now provably has zero effect
  on the dynamics, exactly as expected once it is no longer carried state.
- `FsiBeam.MidpointLoadStepConvergesAtSecondOrderInDt` (new test, beam
  alone, sinusoidal mid-step forcing off-resonance) measures order=2.2174
  — the midpoint substitution does not cost the scheme's temporal order.

**But D9's headline symptom is unchanged.** Before/after this exact change:
- `FsiEnergyDiag.CoupledStepMapRevealsTheGrowthMechanism`'s own rho(M) on
  the committed diagnostic grid: **1.115467 → 1.115466** — unchanged to 4
  significant figures. The dominant eigenvector is still ~84% pressure,
  ~16% beam wddot, same as before.
- `FsiTransient.DISABLED_SelfConvergenceInDtIsSecondOrderInTheAsymptoticRegime`'s
  observed order: **0.822 → 0.8182** — unchanged outside noise. Still left
  disabled with this number, per the same "report, don't loosen" rule.
- `FsiEnergyDiag.DISABLED_LongRunEnergyGrowthApproachesBoundedAfterAFix`'s
  measured 480-step ratio: **1.226473, unchanged** from the earlier round's
  measurement (the mass-conservation fix's number, carried over — this
  round's fix moved it by less than the earlier round's own noise floor).
- A direct ablation (temporarily zeroing the pressure term in the
  delivered traction) gives the IDENTICAL rho=1.000512 before and after
  this round's fix — the viscous-only channel's own dynamics are
  unaffected by the p0-labelling change, exactly as expected, but the
  pressure-inclusive channel's gain is ALSO unaffected, which is the
  actual surprise: the parasitic p0 mode, while real, sits alongside a
  SEPARATE, comparable-magnitude, still-unidentified mechanism in the
  pressure channel that this fix does not touch.

**Conclusion: the hypothesis identified and fixed a real, independently
confirmed defect (a genuine |lambda|=1 parasitic state with a physical
provenance — the trapezoidal-pressure-gradient scheme genuinely does have
this problem), but it is demonstrably not the (or not the only) cause of
D9's coupled energy growth.** The two mechanisms happen to be similar in
magnitude (rho~1.11-1.12 either way) but are separable in the diagnostic:
removing one leaves the other's rho and the transient order untouched to
the precision measured. D9 remains OPEN.

**A genuine regression was found and fixed as part of this round's
change, unrelated to the above.** The earlier round's trapezoidal-
pressure-gradient scheme had, as an apparent side effect, made Aitken's
scalar relaxation converge (slowly, 422 sweeps) at mass_ratio=1, where it
previously did not converge at all. Reverting to the fully-implicit
pressure gradient (this round's fix) reverses that side effect: Aitken now
DIVERGES at mass_ratio=1 (residual reaches ~7.45e+111 within the 500-sweep
budget, confirmed non-finite with a longer budget), rather than merely
converging slowly. `FsiCoupled.AitkenNeedsFarMoreSweepsAtLowMassRatioThanAtHigh`
(which asserted convergence there) has been rewritten as
`FsiCoupled.AitkenDivergesAtLowMassRatioButConvergesAtHigh`, asserting
non-convergence and a growing residual at mass_ratio=1 (mirroring the
sibling `GaussSeidelDivergesWhileIqnIlsConvergesAtMassRatioOne`'s own
`residual_grew` check) while keeping the still-valid claim that Aitken
itself converges cleanly at mass_ratio=25, measured at 19 sweeps. IQN-ILS
is unaffected either way (19 sweeps at mass_ratio=1, both before and
after this round's fix) — the coupling accelerator that matters for this
task's own headline mass ratio still works.

**Not yet found (before round 3)**: the specific numerical inconsistency
that lets the pressure-dominated, outflow-vanishing mode gain energy once
coupled to the beam, when it is exactly marginal in the fluid alone.
Round 2 rules out the trapezoidal-pressure-gradient's p0-parasitic-state
mechanism specifically (confirmed real, confirmed fixed, confirmed NOT
the driver of the residual rho~1.115) — the remaining mechanism is still
open, and demonstrably lives in the pressure channel of the interface
force (the ablation above), not in how that pressure is time-labelled.

**Round 3: the lead's elimination argument, tested both empirically and
via a rigorous, validated discrete energy identity.** Hypothesis (A):
the traction delivered to the beam is not the discrete work-conjugate of
the wall velocity (wrong wall-pressure extrapolation, wrong viscous
coefficient). Hypothesis (B): the outflow's zero-curvature-ghost + p=0
treatment is not the physical "do-nothing" condition, so B_outflow is not
sign-definite. Derived directly from `cn_step`'s own discrete Laplacian
(a standard 3-point stencil in y for v, in x for u — no ghost
complication at either boundary this round cares about): the
energy-conjugate viscous traction at a Dirichlet-moving wall is a
ONE-SIDED difference with coefficient mu (NOT 2*mu, the textbook full
viscous stress) and the energy-conjugate pressure is the ADJACENT CELL's
own value exactly (no extrapolation) — confirmed by hand via the
standard discrete Green's identity for a 3-point Laplacian (proved for
the boundary case, N=2, and cross-checked numerically this round).

**Empirically, five candidate fixes were tried on the small diagnostic
grid, and NONE moved rho(M) meaningfully** (baseline 1.115466; range
measured across all five: 1.115367–1.117273): (A) pressure-only
(adjacent-cell p, no extrapolation, viscous unchanged) — rho=1.117151;
(A) viscous-only (coefficient mu, 1st-order one-sided, pressure
unchanged) — rho=1.115585; (A) combined (both) — rho=1.117273; (B)
do-nothing outflow (`-p+2*mu*du/dx=0` replacing the zero-curvature ghost
+ p=0, continuity restored at the outflow column, verified NOT to break
`FluidReconstructedPoiseuilleIsAFixedPointOfCnStep`,
`SteadyStateBeamDeflectionMatchesTheAnalyticLinearLoad`, or
`InterfacePowerBalancesToRoundoff`) — rho=1.115367; (A)+(B) combined —
rho=1.117173. Fluid-alone rho changed more substantially under each
variant (0.9946→0.9786 under do-nothing, for instance), confirming these
changes genuinely alter the fluid's own dynamics — the coupled system's
own dominant eigenvalue is simply insensitive to all of them.

**Why, confirmed by a rigorous, ROUND-OFF-VALIDATED discrete energy
identity** (`FsiEnergyDiag.EnergyIdentityResidualRevealsWhichHypothesisHolds`,
fluid alone, zero inflow, rest initial state, nonzero CN-distinct top
wall velocity — chosen so the LEFT/BOTTOM boundaries contribute nothing
and only the TOP wall and RIGHT outflow can inject or absorb energy).
Dotting every interior momentum row with its own CN-averaged velocity and
summing gives, exactly, `(E1_interior-E0_interior)/dt =
interior_viscous_power + pressure_div_power - boundary_bt_term`, where
`boundary_bt_term` is the continuity operator's own adjoint (B^T)
contribution from v_wall/u_outflow (boundary velocities referenced by
continuity but with no momentum row of their own — a term the naive
"momentum-row-only" identity misses entirely, and which this round's
diagnostic derived, added, and confirmed closes the identity to
`-7.3e-17`, round-off relative to E1~1.2e-2). `interior_viscous_power`
cleanly splits (proven, same discrete Green's identity) into bulk
dissipation plus the two LOCAL, boundary-attributable viscous flux terms
above (`W_wall_viscous_natural`, `B_out_viscous_natural`) — measured at
1.58e-5 and 2.13e-6 respectively, in this diagnostic's own units.
`pressure_div_power`, however, does **NOT** localise to either boundary:
it is `sum_cells(p1.div_avg)` over the WHOLE domain, dominated (measured:
-0.524, roughly 29,000x the local viscous terms combined, 1.79e-5) by
the fact that
`div_avg` (CN-averaging the rest initial condition against the solved
t1 state) is nonzero away from both boundaries too — a property of the
GLOBAL continuity constraint's own Lagrange multiplier, not a boundary
effect either hypothesis (A) or (B) can fix by construction. This is
exactly why the five empirical fixes above changed the LOCAL viscous
numbers but left the coupled rho untouched: the dominant term in this
discretization's own energy budget is not attributable to a wall-traction
formula or an outflow condition at all.

**Conclusion, per the lead's own fallback instruction.** Neither (A) nor
(B), correctly derived and empirically tested, is nonzero in a way that
explains D9's residual growth — the per-term bookkeeping shows why:
this MAC discretization's pressure channel is a global, not local,
quantity, so the lead's local A/B elimination argument does not cleanly
split for it the way it does for viscosity. D9 remains OPEN. The
remaining candidate, not yet tested: whether the mismatch is in the
GLOBAL pressure/continuity bookkeeping itself (e.g. the pressure gauge's
own treatment, or a genuine physical resonance between the beam's
stiffness and the fluid's pressure response that has nothing to do with
either boundary's own labelling) rather than in any single boundary
condition.

**Round 4: the full discrete energy budget on the coupled map's OWN
dominant eigenvector, evaluated through one real production coupling cycle, and
all three named candidates tested by measurement — none supplies the
growth.** Method (`FsiEnergyDiag.
FullBudgetOnTheUnstableModeItselfAttributesTheGrowth`, small diagnostic
grid, dt=T1/40): extract the dominant eigenvector `x` of the linear
coupled map `M` (rho=1.115466, real, positive — confirmed, not assumed:
imaginary part exactly 0), set `Umax=0` (the ONLY source of the map's
additive term `b`, since the beam has no forcing of its own — a new
`CoupledCase::Umax` field, default 1.0, added for this purpose and
unused by every existing caller), and run `x` through one ACTUAL
production coupling cycle (real `Runner`/IQN-ILS, not the finite-difference
linearization). Confirmed the output equals `lambda*x` to 8.8e-8
relative — genuinely validating `x` as an eigenvector of the real
nonlinear step, not just of its finite-difference approximation.

**Every term of the budget closes exactly**, both internally
(`gen.id.closure_residual` = 1.7e-17) and against the total
(`E1` vs `lambda²·E0`: 2.8e-10 relative; `dE_beam+dE_fluid` vs
`(lambda²-1)·E0`: exact to solver tolerance). The three candidates the
lead named were each tested directly, not assumed:
- **(a) the post-solve mass correction**: measured `correction_energy`
  (its own, isolated KE contribution, computed by independently
  re-solving `cn_step` with the SAME t0 state and converged wall
  velocities to get the RAW, uncorrected solution, then diffing against
  the actual — mass-corrected — production output at the outflow
  column) = **-4.572e-6, NEGATIVE**. The correction removes energy in
  this mode, not adds it — the opposite sign from what candidate (a)
  requires. RULED OUT by measurement, not argument.
- **(b) the outflow rows (restore continuity with a do-nothing traction
  condition)**: IMPLEMENTED directly in `cn_step`/`StokesSolver`
  (replacing the zero-curvature-extrapolation + `p=0`-column pair with a
  genuine `-p+2·mu·du/dx=0` momentum row for `u(nx,j)` paired with a
  real continuity row at `i=nx-1`), confirmed well-posed (`JacobiSVD`
  smallest singular value 7.25e-2, no null direction, on the nx=10,ny=6
  grid) and confirmed NOT to break `FluidReconstructedPoiseuilleIsA
  FixedPointOfCnStep` (needed a companion fix: the interior pin must
  stay INACTIVE under this scheme too, exactly as under the old one —
  the do-nothing rows already fix the gauge on their own; turning the
  pin on unconditionally, tried first, conflicted with the do-nothing
  rows' own implicit gauge and failed that test badly, worst_v up to
  0.95). With this candidate actually applied: **rho(M) = 1.115367,
  unchanged from baseline 1.115466 to within 0.01%**. RULED OUT by
  measurement; the change was reverted (`stokes_solver.hpp` confirmed
  byte-identical to the round-3 baseline via `cmp`) since it does not
  fix D9, per this task's standing "revert what doesn't work" rule.
- **(c) a pin row replacing continuity**: inapplicable by inspection —
  `pin` is unconditionally INACTIVE whenever `bcs.right.outflow=true`
  (`stokes_solver.hpp`'s own `if (!bcs.right.outflow && ...)`), which
  this example always sets. There is no pin row to remove.

**A genuinely large, but separately-explained, positive term was found
on the fluid side**: `ke_change_u_outflow` (the outflow column's OWN
kinetic-energy change, on the RAW/uncorrected solution) = +7.106e-6,
comparable to the entire fluid-side growth (+6.806e-6). This tracks
directly with a massive LOCAL divergence at the outflow column measured
on this eigenvector: `|div(u)|` there is 0.25-0.27 (vs 6.9e-14
everywhere else) — exactly the mechanism candidate (b) targeted. But
fixing that decoupling (candidate (b), above) moves rho(M) only from 1.115466 to 1.115367 (0.01%),
because `correction_energy` (-4.572e-6) and `boundary_bt_term`
(-4.525e-6 as measured, boundary_bt_term*dt = -4.525289976e-06; the SAME necessary B^T identity term round 3 already proved
load-bearing, not a bug) shift to compensate — the net effect on total
fluid growth is preserved under the change, just redistributed among
these three terms differently. **The fluid-side budget is internally
consistent and complete; no single named term, when corrected, changes
the total.**

**The beam side of the budget uncovered an independent, genuine defect,
unrelated to D9**: `BeamSolver::kinetic_energy()` (dx-weighted,
`0.5·rho_s·h_s·dx·Σwdot²`) and `strain_energy()` (unweighted,
`0.5·wᵀKw`) use INCONSISTENT quadratures relative to the pointwise-
collocation mass/stiffness `step_midpoint_load`'s own recurrence
actually advances (this file's own header: "M is the constant diagonal
rho_s·h_s with no dx factor"). Derived and confirmed to solver-tolerance
precision (mismatch/tolerance ratio 0.045, tolerance itself DERIVED from
the measured eigenvector quality, not chosen): the identity that
actually closes is `dE_beam = beam_wall_power + (1-dx_beam)·dPE`, NOT
the naive `dE_beam = beam_wall_power` (which fails by 1.47e-5, four
orders of magnitude over tolerance) — i.e. `strain_energy()` is missing
a `dx_beam = L/nx_beam` factor to be the SAME physical quadrature
`kinetic_energy()` already uses (`0.5·wᵀKw` = `PE_true/dx_beam`, by the
standard discrete integration-by-parts identity for a pointwise 4th-
difference stiffness operator). On the small grid here, `dx_beam=1/6`,
so this artifact term is **56% of the entire measured `(lambda²-1)·E0`
growth** — the single largest contributor found this round. **This does
NOT explain, and fixing it would NOT change, D9's measured per-coupling-cycle
ratio**: `kinetic_energy()`/`strain_energy()` are never called anywhere
in the state-map construction (`run_one_coupled_period`, `coupled_map`)
that produces `rho(M)`, and any quadratic energy functional evaluated on
a state dominated by a single real eigenvector grows at exactly
`rho²` per coupling cycle regardless of which quadratic form is used — verified
structurally (by inspecting every caller of `kinetic_energy()`/
`strain_energy()`), not merely argued. Left UNFIXED this round (owned
file, genuine bug, but explicitly not the term the lead asked this round
to fix — "fix the term that supplies growth, and NOTHING else" — and
this one does not supply `rho(M)`'s growth); flagged here for a future,
separate round.

**Interface adjointness (H vs HᵀD) was also checked and ruled out**: the
power the beam RECEIVES (`dt·Σload_in·wdot_avg`, beam's own nx_beam=6
grid) and the power the fluid DELIVERS (`W_wall_current`, the actual
production traction formula, fluid's own, different, nx_fluid=5 grid)
agree to 1.9e-12 absolute (vs a beam-side scale of 4.6e-6) — a
non-adjoint interpolation leak between the two different-resolution
grids was a plausible, un-named fourth candidate; measurement rules it
out too.

**Conclusion.** The energy budget is COMPLETE (every joule accounted
for, both the beam-side and fluid-side identities validated to
solver-tolerance closure) and NONE of the three named candidates, nor
the two additional ones checked this round (interface adjointness, the
beam quadrature defect), supplies `(lambda²-1)·E0`'s growth in a way
that, when corrected, changes `rho(M)`. The growth appears to be an
intrinsic property of the coupled discrete operator's own eigenstructure
— each individual energy-transfer term is present, correctly signed and
closed, and simply redistributes among terms differently under any of
the tested variants without changing the total. This is consistent with
the classical "added-mass instability" literature for partitioned FSI
(a segregated fluid/structure exchange can amplify even when each
sub-system is independently stable and the interface exchange is
individually energy-consistent, because of the SEQUENTIAL/staggered
order the exchange happens in within a step) — plausible at this task's
own mass_ratio≈2.5, but not yet confirmed as THE mechanism. D9 remains
OPEN. Per the lead's own instruction ("if no term supplies the growth,
report that the budget is incomplete... before changing code"): no
production code change is retained this round (the do-nothing outflow
candidate was implemented, measured, and reverted); the transient
self-convergence test and the long-run energy test both stay DISABLED
with their existing numbers, unchanged. The next round's search should
look at the STAGGERED, sequential nature of the partitioned exchange
itself (which quantity is evaluated at which time level, in which
order, within one IQN-ILS-converged step) rather than at any single
boundary condition or correction term — every term-level candidate this
task's own instructions anticipated has now been tested and excluded.

---

### ~~D13. The participant ABI's `begin_step`/`end_step` run the period's exchanges once, at the period's end — a one-period lag on every transfer, no time averages, no implicit coupling; the resumable execution loop is the fix~~ — CLOSED 2026-10-02

**Found 2026-10-02**, in the C ABI track's own report (`484d8ee`), and verified by reading
`src/abi/context.hpp`: every component the graph declares is wrapped in a component whose
`advance()` is a no-op, and `Runner::run_period()` is called exactly once per period, inside
whichever externally driven component's `end_step()` completes the period. Between a component's
`begin_step()` and `end_step()` the Runner has not run, so:

- an **import is not current when `begin_step()` returns**; it holds the previous period's
  exchange whatever the graph's `lag()` says. `entries.def` promises otherwise ("the component's
  imports are current on return"), and the tests work around it by reading an import only after
  their own `end_step()`;
- everything the schedule places **within** a period is lost: a subcycled component's `Sample` /
  `Accumulate` operations all run at the end and read the same final storage, so a time average
  is wrong; an implicit coupling group cannot iterate, and `fuse_step_report.requires_checkpoint`
  / `requires_restore` are always 0.

It is exact for exactly one class of graph: every field transfer lagged by one period, no
accumulation, no implicit coupling group — SOMA's bookkeeping-clock scheme, which is where the
mechanism came from.

**Fix.** Option 1 of the two choices this entry put to the user, "a resumable execution loop":
`fuse::Runner::Impl::advance()` (`src/runtime/runner.cpp`), `step_period()`, `run_region()` and
`advance_window()` (`src/runtime/iteration.cpp`) are now a hand-written C++20 coroutine
(`detail::Task<T>`, new file `src/runtime/resume.hpp`) that suspends exactly once, at an
externally driven component's own `advance()` (`Component::is_externally_driven()`, new virtual,
`include/fuse/runtime/component.hpp`), and resumes from there — never re-entering the loop from
its start. Every statement these four functions make and the order it makes them in is unchanged;
the only new control flow is the conditional suspend. The suspended coroutine's frame is reused
across periods through a size-keyed free list (`FrameArena`, `resume.hpp`), not the global
allocator, so the per-period hot loop stays allocation-free for an in-process run that never
suspends at all. The participant ABI (`src/abi/participant.cpp`, `src/abi/context.hpp`) now
drives this directly: `fuse_begin_step()` resumes the Runner and checks the step it stops at
against the caller's name (refusing an out-of-turn call with `E-ABI-STEP-ORDER`, naming the
component the schedule actually expects); `fuse_end_step()` resumes it again, past that component's
step. `abi_detail::ExternalStepComponent` (new file `src/abi/external_step_component.hpp`) stands
in for the externally driven component in the graph: a real, byte-exact `save_state()`/
`restore_state()` pair over its own `Intent::Provides` fields (needed to honestly claim
`supports_rollback()`, which entering an iteration region requires), and a no-op `advance()` since
the external caller already wrote those fields directly into the registered storage before
resuming. A window-scoped iteration region whose replayed member is externally driven is refused
with the new `E-RT-WINDOW-EXTERNAL-STRADDLE` rather than given suspend support — recorded as an
open item below, not silently assumed to work.

**Verified at the base build configuration** (no MPI in this sandbox), measured directly on this
host 2026-10-02. Baseline (unmodified `39d3239`, tier-0), at the start of this track's work:
153/153 tests pass. After this fix, same configuration, in a fresh run at the end: 155/155 tests
pass. This is a shared checkout other tracks (MPI connect/accept, LowFive) were editing for the
whole span this track ran, so the +2 is not cleanly this entry's own to account for: a
`ctest -N` listing comparison between the two build trees shows exactly one new ctest *target*,
`transport.lowfive`, from the concurrent LowFive track (unrelated to this entry) — the remaining
difference between that comparison (154 registered in the baseline tree vs. 153 actually run)
was not chased down further, since it predates this track's own changes either way. This track
itself added no new ctest target, only 4 new cases inside the pre-existing `abi.participant`
target.
`fuse_abi_participant_test` (`test/abi/participant_test.cpp`) grew from 3 to 7 `TEST()` cases: the
3 pre-existing ones pass unchanged, and 4 new ones each pin a piece of this fix specifically —
`SuspensionHappensExactlyAtTheScheduledTickNeitherEarlyNorLate` (the exact before/after of an
import's staleness around a suspend point — see its own long comment for why, on the specific
2-component point-to-point graph it uses, that is "still stale right after `begin_step()`,
current right after the partner's `end_step()`", not "current right after `begin_step()`": a
`lag(1)` channel between two same-interval components is, by the schedule's own construction,
ordered after both components' ticks, so no suspend placement changes when its import becomes
current — this is a property of that graph, not of the ABI, and is why
`test/abi/install_consumer/main.c` still reads its import after `end_step()`, unchanged);
`BeginStepOutOfTurnIsRefusedNamingTheExpectedComponent`; `SubcycledMeanAccumulation` (a
quarter-interval producer feeding a time-averaging `Requires` field on a whole-interval consumer —
the mean is correct mid-period, not just a copy of the period's last substep, which the old
scheme would have produced); `ImplicitCouplingConvergesLikeInProcess` (a 2-component affine FSI
graph driven entirely through the ABI, converging through 40 Jacobi sweeps with
`requires_checkpoint`/`requires_restore` reported correctly on every sweep, to the same
fixed point — bit-for-bit on every field — as the identical graph driven directly through
`fuse::Runner`, bypassing the ABI). `Runner.AllocatesNothingWhileRunning`
(`test/runtime/runner_test.cpp`) passes unchanged: the coroutine conversion adds no steady-state
allocation. Three mutations recorded in `build-evidence/mutations/abi.toml` (the step-order
refusal) and the new `build-evidence/mutations/runtime-resume.toml` (the suspend's placement
relative to the no-op `advance()` it guards — found EQUIVALENT, with the reasoning why; and
`requires_restore` never being set) confirm each names exactly the test it predicts, applied and
restored by hand, anchors re-verified with `tools/mutate.py --check-anchors`.

**Compiled at the MPI-enabled build configuration, not run.** `-DFUSE_ENABLE_MPI=ON` builds
cleanly against this host's real MPICH with every change above; MPI cannot start in this sandbox,
so the parallel test suite (`test/parallel/*`) was not run against this fix and is recorded as an
open item, not claimed here.

**Open items.** A window-scoped iteration region (`fuse::WindowCoupler`) with an externally
driven straddling member is refused (`E-RT-WINDOW-EXTERNAL-STRADDLE`), not given suspend support —
no test graph in this codebase currently combines window scoping with an externally driven
component, so this is a scope reduction recorded here rather than a verified behaviour. The MPI
build above is compiled, not run; `test/parallel/runner_transport_test.cpp`'s own allocation
check, and the full parallel suite, remain to be measured on a host that can start MPI.

### D14. LowFive: the two-HDF5-image hazard is closed by a measured configure-time probe; the distributed transport and Demo B are compiled, not run

**The two-HDF5-image hazard (S6 phase 3, "LowFive" track) — CLOSED with a measured,
configure-time acceptance test; the underlying fix is a host action.** A binary
linking both MOAB and LowFive can load two independent `libhdf5` images — MOAB's
dependency is an absolute path, LowFive's may resolve via its own `LC_RPATH` to a
different prefix — and when it does, LowFive's VOL interception silently does not
apply to MOAB's own HDF5 traffic (or vice versa): no link error, no runtime error,
data simply reaches disk. Measured directly on this host, 2026-10-02:
`/opt/lowfive` (built against MOAB's own `/opt/hdf5`) loads exactly one image of
each and is safe; `/opt/lowfive-shared` (built against a separately built
`/opt/hdf5-shared`) loads two of each and reproduces the exact silent-corruption
symptom (`H5Pset_vol(): "not a file VOL ID"`) while the process still exits 0. Full
record, including why a static CMake path comparison would have been wrong on this
very host (`find_package(HDF5)` resolves outside `CMAKE_PREFIX_PATH` via a glob over
`CMAKE_SYSTEM_PREFIX_PATH`'s default `/opt` entry): `docs/user/lowfive.md`.
`tools/probe-hdf5-images.{c,sh}` is the acceptance test — it links MOAB and a given
LowFive prefix, runs the result under a library-loader trace, and counts the
distinct `libhdf5`/`libhdf5_hl` images that actually load.
`cmake/FuseDependencies.cmake` runs it at configure time whenever
`FUSE_ENABLE_LOWFIVE AND FUSE_ENABLE_MOAB` and **refuses the configuration**
(`FATAL_ERROR`, not a warning) if it measures more than one image, naming the exact
rebuild line. Verified end to end against this checkout's real
`cmake/FuseDependencies.cmake`: the matched pairing configures clean, the mismatched
one refuses with the measured evidence in the error message. `/opt` is not writable
from the sandbox this track ran in, so rebuilding a LowFive to close a measured gap
(for instance `/opt/lowfive`'s own stale MPI linkage, a second, unrelated defect
recorded on the same page) is a host action, not something this entry can close
further.

**Demo B (two unmodified HDF5 executables, cross-process) — NOT executed; compiles
against LowFive's documented API only.** The single-process interception path
(`LowFiveTransport::set_location`/`build_plan`/`move`, unconditionally compiled, part
of this codebase before this phase) is measured end to end and is not in question.
Actually crossing an OS process boundary needs LowFive's `DistMetadataVOL` and an MPI
intercommunicator between the two executables — this phase adds
`LowFiveTransport::set_intercomm`/`serve_as_producer`/`receive_as_consumer`
(`src/transport/lowfive_transport.cpp`, declared in `fuse/transport/transport.hpp`)
against `DistMetadataVOL`'s public header contract, and `examples/demo_b`'s own
producer/consumer executables plus a minimal rendezvous. None of it has run: MPI
cannot start in the sandbox this was written in, and this exact combination (FUSE +
`DistMetadataVOL`) was never implemented in the `upstream/soma` parent project either
— its own `lowfive_transport.cpp` is single-process only, confirmed by inspection.
Compiles cleanly under `-DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON
-DFUSE_ENABLE_MOAB=ON` against this host's real MOAB/LowFive/HDF5 (measured
2026-10-02); the host run that would turn "compiles" into "works" is recorded as an
open item in this track's report, not claimed here.

**Unit 3 ("the same U/V pair over LowFive") — graph/model layer DONE and
tier-0 tested; the participant mains' own H5Dwrite/H5Dread wiring NOT
implemented, for a reason found while scoping it (below).**

DONE: `FieldSpec::hdf5_path` (`std::optional<std::string>`, `include/fuse/
model/field_spec.hpp`), descriptor v11 (`include/fuse/model/descriptor.hpp`,
golden `test/golden/waist-v11.bin`), full `to_yaml`/`from_yaml` round trip
(`hdf5_path:` under a field entry), and a fifth `LaunchMode` value,
`LowFive` (`include/fuse/model/launch.hpp`) — a component declares it to
say "I need LowFive interception set up," independently of that
component's OWN launch story (Mpmd/Connect/InProcess), the same way a
`Coupler` is a placement separate from the physics components it serves.
`validate()` (`fuse/model/validate.hpp`) refuses a field whose `hdf5_path`
is set with no `LaunchMode::LowFive` component anywhere in the graph
(`E-LOWFIVE-NO-COMPONENT`) and refuses two fields sharing the exact same
non-wildcard path (`E-LOWFIVE-DUPPATH`, mirroring `LowFiveFieldBindingTable
::add`'s own existing rule one layer down). `fuse::build_lowfive_bindings()`
(new file `include/fuse/transport/lowfive_graph.hpp`, transport layer since
it depends on `CouplingGraph`) builds a `LowFiveFieldBindingTable` FROM a
graph's own fields, qualified `"<component>.<field>"`; the standalone type
and its own hand-built-table constructor (`fuse/transport/lowfive.hpp`) are
untouched, exactly as asked. `brusselator_graph_tool`'s new `--transport
lowfive` flag sets `hdf5_path` on both `u_out` and `v_out` (the two
transfer/export fields) and adds a separate, fieldless `LaunchMode::LowFive`
component spanning every rank in use; a new committed reference
(`examples/brusselator/participants/reference_mpmd_3_lowfive.yaml`) pins its
output byte for byte (`verification.brusselator.graph_tool.reproduces_
reference_lowfive`), and `test/verification/brusselator_graph_tool_lowfive_
test.cpp` proves the graph it writes actually `validate()`s clean (plus the
negative control: the same fields without the LowFive component are
refused). Two new mutation rows, both hand-verified with a decisive,
exact-predicted kill: `hdf5-path-ignored-on-serialisation` (descriptor.hpp)
and `hdf5-path-duplicate-check-disabled` (validate.hpp) —
`build-evidence/mutations/base-model.toml`. All of this needs no LowFive,
no HDF5 and no MPI; it is exercised by the ordinary tier-0 suite (166/166
passing) and all 5 repo-health gates.

**Found while implementing, not before: adding `LaunchMode::LowFive` left
`fuse::transport::connect()`'s own exhaustive `switch (launch.mode)`
(`src/transport/connect.cpp`) silently missing the new case** — a real
gap, caught only because this session additionally configured and built a
genuine `-DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON` tree against this
host's `/opt/lowfive`+`/opt/hdf5` (a stronger check than this sandbox had
previously used for this track: compiler warnings, not just tier-0 green).
Fixed: `LowFive` now refuses with `E-CONNECT-SPEC` exactly like `InProcess`
and `Plugin` (nothing to connect; a LowFive-launched component's ranks come
from whatever other launch story applies, or none). No tier-0 test reaches
this branch — `connect()` itself requires `MPI_Initialized`, same as
`InProcess`/`Plugin`'s own identical, equally untested branches — so this
is a compile-time fix with no new regression test, consistent with those
two siblings' own existing coverage gap.

**NOT implemented: the participant mains' own `--transport lowfive` mode**
(U calling plain `H5Dwrite`, V plain `H5Dread`, FUSE's job limited to
registering the VOL connector and hand the caller a ready `fapl_id` —
`LowFiveTransport::access_plist_handle()`'s own documented purpose). Scoped
out, not merely postponed, for a concrete reason found while designing it:
LowFive's single-process interception (`MetadataVOL`) — the ONLY path this
codebase has ever measured end to end — needs the writer and the reader in
the SAME process's address space. The graph `--transport lowfive` builds
places U and V on DISJOINT ranks (mirroring every other Unit 2/2b fixture,
`--launch mpmd`), which needs the DISTRIBUTED path (`DistMetadataVOL`, an
MPI intercommunicator) — the same path D14's own Demo B section above
already records as compiled against LowFive's documented API but never
run, and whose rendezvous design (reuse `connect()`, or stay with Demo B's
own local one) is already an open question for the lead. Wiring U/V's
existing, large participant mains into either path — same-process
co-location (a materially different graph shape than every other
Brusselator fixture) or the unmeasured distributed one (a new
intercommunicator rendezvous plus a new small ABI surface to hand
`access_plist_handle()`'s value to pure C/Fortran callers, since neither
language may include `fuse/transport/transport.hpp` directly) — is real,
separable design and implementation work, not a mechanical extension of
what exists.

**Correction to this entry's own framing above, per the lead's ruling
2026-10-04: the Runner-Transport design this entry argued for was the wrong
frame.** A LowFive-bound channel's two ends are, BY DESIGN, ordinary,
unmodified HDF5 callers — spec §6 mode 4's own contract is "no FUSE-specific
call on either side" (`docs/user/lowfive.md` line 4). `LowFiveTransport::
move()`, and the Runner's per-plan Transport machinery this entry traced,
is the right tool for a channel FUSE itself must move end to end; it is the
wrong tool for this one. The corrected design (`docs/user/lowfive.md`'s new
"Unit 3... an application-performed transfer" section has the full account):
FUSE's job on such a channel is the schedule, the window semantics, any
Remap the two sides' meshes need, the convergence decision, and installing
LowFive's VOL bindings — never the bytes. No ABI growth (the earlier
`hdf5_fapl` proposal is withdrawn): an application needing a FUSE-provided
property list would itself be a FUSE-specific call, which the contract
above forbids.

**A new, measured blocker found while re-attempting this, 2026-10-04:
BOTH `liblowfive.dylib` and `liblowfive-dist.dylib` carry an identical,
unresolvable `@rpath/libfmt.8.dylib` dependency** (`otool -L` on both,
docs/user/lowfive.md's own subsection has the exact output) — not only the
separate distributed archive, as an earlier pass at this wrongly concluded
from the dyld error's own wording (it names `-dist` only because that is the
one dyld happens to check first; `otool` shows the dependency on the MAIN
shared library too). No `libfmt.8.dylib` exists anywhere under `/opt` or any
installed conda/homebrew prefix (9, 11 and 12 are present; not 8). Every
LowFive-linked binary therefore fails at `dlopen`, including the
single-process path this page elsewhere calls "implemented, measured" —
that measurement predates this finding and was not re-confirmed against
this exact install. Separately, `/opt/lowfive` HAS been rebuilt since this
page's own stale-MPI-linkage paragraph was written (`otool` now shows the
current `libmpicxx.0.dylib`, not `.12`) — that specific, earlier defect is
closed; this libfmt one is a different, still-open one. Full account,
including the forced-not-found CMake probe that isolated the libfmt
dependency from a separate, already-documented distributed-archive LINK
failure (missing `fmt::v8::...` symbols for the static `.a`, a different
problem from this dlopen-time one): `docs/user/lowfive.md`'s own subsection.
This sandbox has no writable `/opt` and was declined permission to install
a workaround via a new package environment; fixing `/opt/lowfive`'s own
shared-library dependency is a host action, like the hazard already on this
page. Until it is fixed,
step 5 of the lead's own 3a design ("if ordinary H5Dread in V cannot see
U's in-memory write within one process, stop and report the measured
behaviour") cannot be reached at all — there is nothing to measure yet,
only LowFive's own header contracts to read.

**Unit 3a ("single process, the measured path"), the LowFive-free half —
DONE, tier-0 tested, no HDF5/LowFive/MPI required.** This is the half that
needed no libfmt-blocked host dependency to reach: a LowFive-bound
channel's two ends are driven through the ordinary participant ABI
(`fuse_begin_step()`/`fuse_end_step()`), with the harness's own
`view_field()` + `memcpy()` standing in for the H5Dwrite/H5Dread pair an
application would use. Two real defects were found and fixed along the
way, both now permanent library behavior, not test-only workarounds:

1. `WindowMemberConfig::window_import_delivery_arc` (`src/runtime/
   runner_impl.hpp`): `advance_window()`'s own "t=0 correction" read
   `slots[cfg.window_import.value()]` assuming the importing member's
   own chain had already run this sweep — true under the old,
   non-interleaved instruction ordering, false once a scheduling-order
   fix (`OpKind::ExternalDelivery`'s own comment, `include/fuse/ir/
   op.hpp`) made a LowFive-bound delivery's own position coincide with
   the sweep's own end. Fixed by resolving the t=0 read through the SAME
   chain-root walk `view_field()` already used for a LowFive-bound
   export, stopping at `OpKind::ExternalDelivery` instead of assuming
   the chain had already executed.
2. `prime_next_request()` (`src/abi/participant.cpp`): a *matched*
   rollover (the caller asks for component `C`, resuming past a cached
   delivery proxy discovers the period has converged and rolled over,
   and the new period's own first-due component happens to BE `C`) used
   to fall through to the ordinary `step_opened = 1` path unconditionally
   — running the new period's first physics substep within the SAME
   call that performed the rollover, before any external caller could
   observe the period transition at all (`fuse_is_coupling_ongoing()`'s
   own `period` field lags one call behind by construction). Fixed by
   folding a matched rollover into the SAME `step_opened = 0` branch an
   *un*matched one already takes: `treat_as_mismatch = (req.component !=
   in->component) || rolled_over`, trading one extra, cheap begin_step()/
   end_step() round trip at every period boundary for a clean,
   physics-free observation point at the period's own true end, matched
   or not. `fuse_step_report::step_opened` (appended for exactly this)
   and `::period` (appended so a caller can tell a matched rollover from
   an ordinary first call) are the ABI surface this created; both are
   growth-only, byte-identical for an old caller (`abi-design.md` rule 2),
   and the ordinary non-window, non-LowFive path never rolls over, so it
   stays byte-identical to before this fix.

Covered by `test/verification/brusselator_lowfive_harness_test.cpp`
(`BrusselatorLowFiveHarness.ExternalDeliveryMatchesInProcessRunner`,
bit-for-bit against an in-process `fuse::Runner` reference, identical
sweep counts in both waveform iterations) and, at the ABI level directly,
by three new `AbiParticipant` cases in `test/abi/participant_test.cpp`:
`EndTimeBoundedRunRefusesPastItsOwnDeclaredEnd` (a matched rollover right
at a run's own declared end), and `MismatchedRolloverNamesTheRealPeriod
StartComponent` (case (ii): a purpose-built graph where the window's own
fixed region-entry and the true period-start component are DIFFERENT
components by construction, proving the mismatch names the real one, not
the hint). A dedicated golden-IR fixture, `test/data/pair_lowfive_bound.
yaml` / `test/golden/pair_lowfive_bound.ir` (`GoldenIr.PairLowFiveBound`),
pins the compiled shape (`ExternalDelivery`, one input — an ordering edge
to the source field's own producer, not a data dependency — never
`Transport`) byte-exact at the IR level, independent of the behavioral
harness. Six mutations recorded, all hand-proved: the rollover fix itself
(`abi-prime-next-request-rollover-removed`, `build-evidence/mutations/
abi.toml`) and five Unit-3-specific ones in `build-evidence/mutations/
transport-lowfive.toml` (the ExternalDelivery suspension removed, the
cold-start delivery omitted from the harness, `lowfive_bound` forced
false in `rules_channel.cpp`, plus the two pre-existing binding-table
checks). Full account of the investigation (the scheduling-order fix,
the per-sweep hexfloat tracing that isolated the two defects above, and
the case (i)/(ii) distinction) lives in this track's own session
history; `docs/user/lowfive.md`'s "Unit 3a" section and `docs/user/
c-abi.md`'s protocol section document the resulting behavior for a
caller.

**Unit 3b ("the distributed shape"): implemented, LowFive-free as far as
the sandbox allows — the real HDF5/LowFive path compiles against this
host's headers and nothing further; host-only, like every other
Conformance Unit 2/2b row this mirrors.**

1. `fuse::plan_lowfive_rendezvous(writer_ranks, reader_ranks, this_rank)`
   (`include/fuse/transport/lowfive.hpp`) — pure, MPI-free rank-set
   arithmetic (`MPI_Comm_split`'s own color/key and
   `MPI_Intercomm_create`'s own local/remote leader), mirroring
   `connect.hpp`'s own "split for testability" convention. `E-LOWFIVE-
   RENDEZVOUS-OVERLAP` (the two sets share a rank) and `E-LOWFIVE-
   RENDEZVOUS-UNPLACED` (`this_rank` in neither). Five new tests in
   `test/transport/lowfive_test.cpp`, including one deliberately
   non-contiguous `RankSet` proving `split_key` comes from `index_of`,
   not the raw rank number. Tier-0, no MPI needed to test the arithmetic
   itself.
2. `u_main.c`/`v_main.f90` gain `--transport lowfive`: after a sweep's
   last substep (`t_end` at a window boundary, checked by exact-rational
   cross-multiplication — no floating point), `view_field()` the K-wide
   export and `H5Dwrite` it to the field's own `hdf5_path`; before a
   sweep's own substeps, `view_field()` the LowFive-bound import's own
   `ExternalDelivery` arc and `H5Dread` into it. Both calls are PLAIN
   HDF5 (`H5P_DEFAULT`, no FUSE-specific call) through a new, tiny
   extern "C" shim (`examples/brusselator/participants/shim.{h,cpp}`:
   `bruss_lowfive_write`/`bruss_lowfive_read`/`bruss_lowfive_available`)
   rather than calling HDF5 directly from each language -- the C side
   could (u_main.c is already pure C), but the Fortran side cannot:
   this host's `/opt/hdf5` ships no Fortran module at all (checked), and
   this codebase's own established pattern for exactly this situation is
   shim.cpp, already linked into both mains for the physics core.
   `bruss_lowfive_available()` lets v_main.f90 refuse early with a clear
   message when `FUSE_HAVE_LOWFIVE` is 0 -- it has no C preprocessor to
   read that macro with directly, unlike u_main.c.
   Known scope limit, recorded rather than silently assumed away: neither
   main can know an UPCOMING step's own `t_begin` before calling
   `begin_step` for it, so the read happens before EVERY substep, not
   provably only a sweep's first -- correct for this track's own
   reference graph (`brusselator_graph_tool.cpp`'s `kDtU == kWindow`, the
   degenerate one-substep-per-window case) and merely redundant, not
   wrong, on a graph with more than one substep per sweep.
3. The VOL attachment itself is the `HDF5_VOL_CONNECTOR`/
   `HDF5_PLUGIN_PATH` environment-variable mechanism `docs/user/
   lowfive.md`'s own "Unit 3" section designed for the single-process
   case, extended here to the distributed one: a new target,
   `fuse-lowfive-vol` (a SHARED library HDF5's loader `dlopen()`s),
   exports the two required extern "C" symbols. On first call it reads
   `FUSE_LOWFIVE_BINDINGS` (a graph YAML path), applies every
   LowFive-bound field's own `hdf5_path` via `set_memory()`, and — if an
   MPI world of more than one rank and a LowFive-bound channel naming
   this rank are both present — builds the writer/reader
   intercommunicator itself via `plan_lowfive_rendezvous()` (item 1)
   plus `MPI_Comm_split`/`MPI_Intercomm_create`, then constructs
   `LowFive::DistMetadataVOL` instead of the single-process
   `MetadataVOL`. **Categorically less confidence than anything else in
   this entry**: this cannot be measured even once in this sandbox (the
   libfmt.8 dlopen blocker above still applies — LowFive cannot load at
   all here, so neither can a plugin built against it), it is scoped to
   MPMD launch only (ACCEPT mode's own merged base communicator is never
   exposed back to a caller through any ABI query, an open question for
   the lead rather than a silent guess), and the connector name this
   plugin's own ctest rows pass to `HDF5_VOL_CONNECTOR` ("lowfive") is
   this track's best reading of LowFive's own `VOLBase::name` member, not
   something this session found a way to confirm from headers alone.

   **A real, from-clean-configure build DOES catch real problems a
   build against an already-warm cache cannot — the lead's own rejection
   of this item's first pass, and the three-theory diagnosis that
   followed, are the record of that.** The first version (one
   translation unit, `lowfive_plugin.cpp`) built clean ONLY because of
   an accidental `target_link_libraries` call order; a fresh configure
   with `/opt/lowfive;/opt/hdf5` FIRST on `CMAKE_PREFIX_PATH` (the lead's
   own clean-export check) reproduced the exact failure the lead
   reported: `/opt/lowfive/include/fmt/format-inl.h` fails this host's
   C++20 consteval checking. The plugin is now two translation units
   with disjoint CMake dependencies — `lowfive_plugin_graph.cpp` (fuse
   model/base only, builds the field-binding list and the rendezvous
   plan) and `lowfive_plugin_vol.cpp` (LowFive/HDF5/MPI only, never a
   FUSE header beyond `base/config.hpp`), handing plain data across a
   third, dependency-free header — per the lead's own instruction, and a
   real fix: unlike the link-order version, no translation unit here can
   ever see both fmt-pulling header families at once, regardless of how
   `CMAKE_PREFIX_PATH` is ordered.

   That TU split alone was NOT the whole story, though — measured, not
   assumed, after the split still failed under the lead's own prefix
   ordering: `lowfive_plugin_vol.cpp`, which never includes a single
   FUSE header, failed with the IDENTICAL consteval error, proving the
   problem was never "two fmt versions in one translation unit" at all.
   **The actual root cause: `/opt/lowfive` ships its own bundled fmt
   8.1.1 WITH a discoverable CMake package config
   (`/opt/lowfive/lib/cmake/fmt/fmt-config.cmake`), and this project's
   own `fuse_require(fmt ...)` call had no minimum version — with
   `/opt/lowfive` first on `CMAKE_PREFIX_PATH`, `find_package(fmt)`
   simply accepted the first config it found, and `fmt::fmt` resolved to
   LowFive's bundled, broken 8.1.1 copy project-wide, not to homebrew's
   working 12.2.0.** Fixed at the source: `fuse_require(fmt VERSION 9
   ...)` (`cmake/FuseDependencies.cmake`) rejects 8.1.1 and finds
   homebrew's 12.2.0 instead, in either `CMAKE_PREFIX_PATH` ordering.
   Measured this session, from two FRESH configures (`/opt/lowfive;
   /opt/hdf5` first, and last, on `CMAKE_PREFIX_PATH`, exactly the two
   orderings the lead asked this to be proven against): both build every
   target clean, including `fuse-coupler-transport` (`lowfive_
   transport.cpp` and its four non-LowFive siblings, which the version-
   less `fuse_require(fmt ...)` call had ALSO been silently exposing to
   the same failure — this track's own earlier claim that "`lowfive_
   transport.cpp` compiles because it includes no FUSE fmt-using header"
   was wrong: it includes `fuse/transport/transport.hpp`, which includes
   `fuse/base/result.hpp` directly) and every shim.cpp-compiling target
   in both `examples/CMakeLists.txt` and `test/CMakeLists.txt`.
   `cmake/FuseDependencies.cmake`'s own `fuse_link_lowfive_with_fmt_fix()`
   function and the `BEFORE`-ordered include directory it still applies
   at several call sites are a now-redundant belt-and-suspenders left
   over from the two earlier, incomplete theories (recorded in that
   function's own comment rather than deleted) — correctness rests on
   the `VERSION 9` floor alone. **Lead's own ruling, accepting this
   item:** `fuse_link_lowfive_with_fmt_fix()` has been redundant since
   commit `f59b446` (the `VERSION 9` floor); left in place for now, a
   cleanup pass may remove it.
4. ctest rows `verification.brusselator.participants.lowfive.{mpmd,
   accept}.3` (`examples/CMakeLists.txt`), registered only when
   `FUSE_ENABLE_LOWFIVE`, `FUSE_ENABLE_MPI`, the Fortran target, and
   `fuse-lowfive-vol` all exist — world size 3 only (1+1+1), this track's
   own degenerate reference graph never needing more to prove the wiring.
   `HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH`/`FUSE_LOWFIVE_BINDINGS` are set
   in the test's own `ENVIRONMENT`, not inside `run_participants.sh`
   (extended with a new, optional trailing `[transport]` argument passed
   through to all three executables), so `ctest --show-only` shows a host
   operator exactly what each row depends on.
5. Tier-1+LowFive configure AND build (not run), each from a FRESH
   configure, verified TWICE this session with `CMAKE_PREFIX_PATH`
   reversed between runs (`/opt/lowfive;/opt/hdf5` first, then last;
   otherwise identical: `-DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_LOWFIVE=ON
   -DFUSE_ENABLE_FORTRAN=ON -DHDF5_ROOT=/opt/hdf5`) — every target in
   the full build (not a named subset) built clean both times, including
   `brusselator_u_main`, `brusselator_v_main`, `fuse-lowfive-vol`, and
   `fuse-coupler-transport`.

**Item 2 (this round): `delivery_pending`/`delivery_field` (ABI growth), the
real-parameter conformance tests, and the two findings they surfaced --
BOTH NOW FIXED.**
`fuse_step_report::delivery_pending` (entries.def, docs/user/c-abi.md's own
new section) is 1 on exactly the one report per LowFive-bound rollover,
confirmed both by a dedicated ABI-level test (`AbiParticipant.
DeliveryPendingIsAlwaysZeroOnAnOrdinaryGraph`) and by a hand-proved mutation
(`build-evidence/mutations/abi.toml`). `u_main.c`/`v_main.f90` were
rewritten to use it (cold-start write+read once, then gated on
`delivery_pending` only, replacing the prior "read before every substep"
simplification) and compile clean under the full tier-1+LowFive+MPI+Fortran
build. `brusselator_graph_tool.cpp`'s own constants were moved onto the real
example's own parameters (16x16/24x24, dt_u=1/500, dt_v=3/1000, window=1/50,
tolerance=1e-11); `reference_mpmd_3.yaml`/`reference_mpmd_3_lowfive.yaml`
were regenerated — `verification.brusselator.graph_tool.reproduces_reference
[_lowfive]` measured to FAIL against the old committed YAML, then PASS
against the regenerated one.

`test/verification/brusselator_participants_test.cpp` (no LowFive channel,
direct ABI access) was moved onto the same real parameters, driven by
`next_component` (not a fixed alternation — the real dt_u:dt_v:window ratio
is genuinely 10:7, not an integer multiple of either), parameterized on both
waveform iterations, run for `kPeriods=150` (`--end 3`). **Both Jacobi AND
GaussSeidel PASS** bit for bit against Unit 1's own in-process `Runner`
across all 150 windows (measured: U takes exactly 10 substeps/window, V
exactly 7, confirmed by counting, not assumed).

**Finding 1, FIXED: `channel_window()`'s own E-LAG validation gap.**
`fuse::detail::channel_window()` (`include/fuse/model/validate.hpp`): for a
channel with no `Explicit`/`FromComponent` accumulate window, it fell back
to `Window::Kind::CouplingPeriod`'s own answer, "the TARGET component's own
`.interval`" (U's, 1/500) — a notion of "the coupling window" that predates
`CouplingWindowScope` and had no path to that scope's own `explicit_interval`
(1/50) at all; GaussSeidel's own back channel ('v2u') tripped `E-LAG: lag
spans 3/1000 but the coupling window is 1/500`. Invisible through Unit 3a
(dt_u = window = 1/500 there, so the wrong and right answers coincided by
accident). **Fix**: `channel_window()` now resolves the channel's own
`IterationRegionDecl` first and, if it has a `CouplingWindowScope`, shares
`fuse::window_length()` (the SAME function the scheduler/`ClockSet` already
use) to compute the real window — falling through to the old switch-based
logic only for a channel OUTSIDE any window-scoped region. Two new tests
(`Validate.GaussSeidelWindowLagWithinTheRealWindowValidates`,
`Validate.RejectsLagExceedingTheWindowItself`) and a hand-proved mutation
(`build-evidence/mutations/base-model.toml`,
`model-channel-window-ignores-window-scope`) cover it.

**Finding 2, FIXED: `delivery_pending` needed `delivery_field`, and the
writer needed its own suspend too.** Two related gaps, both in
`fuse::detail::deliver_external()` (`src/runtime/iteration.cpp`):
(a) the ordering edge (`OpKind::ExternalDelivery`'s own comment, `op.hpp`)
puts the delivery instruction right after the EXPORTER's own last substep of
a sweep, but the function only ever suspended for the IMPORTER — a
distributed writer process (one that hosts the exporter but not the
importer) never learned when to write at all, a gap the single-process
harness this track's own earlier tests used had hidden (both ends were
always hosted together); and (b) `delivery_pending` alone cannot discriminate
WHICH of a graph's independently-paced LowFive-bound channels fired —
measured as a substantial (not ULP-scale) Jacobi divergence at window 1
once the harness was moved onto a real TWO-channel (`u_out`, `v_out`) graph
(harness=1.5913141996331861 vs ref=1.6009991770349754), traced to "deliver
everything whenever the bit is 1" copying a direction whose own producer had
not actually finished the sweep yet.

**Fix**: `deliver_external()` now suspends for the exporter (first, matching
real time order) AND the importer, independently checked; each suspend
carries a new `delivery_field` (`fuse_step_report`/`fuse_field_ref`'s own ABI
growth, entries.def, `sizeof(fuse_step_report)` 96 → 104 bytes) naming the
channel's own SOURCE field, regardless of which end the suspension is for.
`impl_begin_step()`'s old drain-while-loop (silently collapsing consecutive
delivery-due requests to the last one's info — the exact mechanism behind
the Jacobi regression above) was replaced with a single-stop branch: every
`is_delivery_due` request is its own externally visible report, consumed
exactly once. The protocol (docs/user/c-abi.md): "on a report naming field
F, write F if producing it or read F if consuming it, then
`fuse_begin_step()` again." A single-process caller driving more than one
externally-driven component (the harness; real distributed mains never face
this) must gate the action on `begin_rep.next_component` — the delivery's
own addressed component, which `resolve_cached_next_component()` now reports
directly for an `is_delivery_due` request instead of forward-scanning to
"the next real stepper" — not on which identity it happened to call
`fuse_begin_step()` with (`in->component` is never checked for a delivery-due
request, by design: either end may ask and get the same cached answer).

Tests: `AbiParticipant.DeliveryFieldNamesTheRightChannelOnEachSuspension`
(two-channel graph, asserts both channels' own fields appear and neither is
ever missed); `brusselator_participants_test` and
`brusselator_lowfive_harness_test`, both waveform iterations, all 150
windows, bit for bit, with matching sweep counts. Mutations (hand-proved,
`build-evidence/mutations/transport-lowfive.toml`):
`lowfive-external-delivery-suspension-removed` (importer's own suspend
removed → E-ABI-STEP-ORDER) and `lowfive-delivery-field-wrong-channel`
(source field swapped for target → the harness's own field-matching
assertion fires).

**What remains for this entry (S6 phase 3, Unit 4, consolidated 2026-10-07
— every open thread this entry's own prose raised above, gathered here
once; nothing below is repeated in D18 or D19, and the plain (non-LowFive)
participant rows those two entries track are NOT re-listed here):**

1. **Host action: rebuild `/opt/lowfive` (and `/opt/lowfive-dist`) against
   fmt 9 or newer.** Both ship a bundled fmt 8.1.1 and both carry an
   unresolvable `@rpath/libfmt.8.dylib` dependency (measured via `otool
   -L`) — no `libfmt.8.dylib` exists anywhere on this host, so every
   LowFive-linked binary fails at `dlopen`, including the single-process
   path. Blocks items 2-4 below entirely; nothing past this point can be
   attempted, only read from headers, until it is done.
2. **Host step: the connector-name check.** `HDF5_VOL_CONNECTOR=lowfive`
   is this track's own best reading of LowFive's `VOLBase::name` member —
   not something confirmed from headers alone, and not checkable until
   item 1 lets `fuse-lowfive-vol` actually load. First host attempt should
   confirm the name matches before trusting anything downstream of it.
3. **Open question for the lead: the accept-mode intercommunicator gap.**
   ACCEPT mode's own merged base communicator is never exposed back to a
   caller through any ABI query, so `fuse-lowfive-vol`'s own rendezvous
   logic (item 3 of the Unit 3b account above) has no way to reach it on
   that path — scoped to MPMD launch only until this is answered.
   Blocks `verification.brusselator.participants.lowfive.accept.3`
   specifically; `.mpmd.3` does not need it.
4. **ctest rows, host-only, blocked on item 1:**
   `verification.brusselator.participants.lowfive.mpmd.3`,
   `verification.brusselator.participants.lowfive.accept.3` (the second
   additionally blocked on item 3).
5. **Host run: Demo B** (`examples/demo_b`'s producer/consumer
   executables) — compiles against LowFive's documented
   `DistMetadataVOL` API only; never run. Needs item 1 and a real MPI
   intercommunicator between two genuinely separate processes.

### D15. The provider direction (S6 phase 4, "Adapter host" track) is implemented and conformance-tested on one process; device fields, mixed checkpoint/restore, and a plugin on a rank subset are not

**What is done, measured 2026-10-02.** `include/fuse/abi/provider.h` (generated from a second
`entries.def` entry list, `FUSE_ABI_PROVIDER_ENTRY`, sharing the participant direction's own
struct catalogue); `src/abi/provider_adapter.cpp` (wraps a `fuse_provider_abi_v1*` as a
`fuse::Component`); `src/abi/registry.cpp` (mounts a native component, an in-memory table, or a
`dlopen()`'d shared object); `fortran/ffuse_provider.f90` (the Fortran mirror, a plugin author
assembles the table itself rather than dereferencing one). `test/abi/conformance_test.cpp` runs
ONE graph (the two-component lag-1 fixture and the subcycled-mean fixture,
`test/abi/graph_fixture.hpp`) four ways — native C++, a dlopen()'d C plugin
(`test/abi/plugins/linear_provider.c`), a dlopen()'d Fortran plugin
(`test/abi/plugins/linear_provider.f90`), and the participant direction — with `EXPECT_EQ` on
every exchanged value against the native run; plus version/struct_size negotiation,
describe-mismatch, and advance-error refusal rows, each with a confirmed mutation
(`build-evidence/mutations/abi-provider.toml`). A `LaunchMode::Plugin` component named in the
graph (descriptor v10, `plugin_path`) is mounted automatically at `fuse_initialize()` — no
test-only injection needed — and driven in-process through `fuse_run()` exactly like a native
component; `ConformanceTest.GraphDeclaredPluginIsAutoMountedAndRun` is the end-to-end proof. A real
destruction-order bug was caught this way and fixed: `fuse_ctx::plugin_registry` (which
`dlclose()`s every mounted plugin's shared object) must be declared BEFORE `fuse_ctx::components`
(the `ComponentSet` holding the `ProviderAdapter`s that call through those same shared objects at
their own destruction) — C++ destroys members in reverse declaration order, and the wrong order
reproduced a SIGSEGV inside `ComponentSet`'s own destructor on every run, caught only because this
track's own conformance test exercises the graph-declared path end to end rather than only through
test-only injection (which bypasses `fuse_ctx::plugin_registry` entirely and therefore never hit
this ordering at all).

**What is not covered, named rather than silently absent:**
- **Device fields.** `register_field`'s own `FUSE_MEMSPACE_DEVICE` refusal
  (`E-ABI-REGISTER-MEMSPACE`) has a provider-direction counterpart nowhere in this track:
  `ProviderAdapter` only ever allocates host storage, and a plugin that wanted a device pointer
  would have no way to ask for one.
- **Checkpoint/restore through an iteration region mixing native and plugin components.**
  `ProviderAdapter::save_state()`/`restore_state()` forward to the plugin's own entries when
  present, but no test puts a plugin-backed component inside an `IterationRegionDecl` alongside a
  native one and checks the sweep converges identically to an all-native run — the conformance
  harness's two fixtures are both non-iterated.
- **A plugin launched on a subset of a component's ranks**, or the interaction between
  `LaunchMode::Plugin` and `LaunchMode::Mpmd`/`Connect` on different components of the same
  multi-rank graph, is untested: every fixture here runs single-process, and
  `fuse::transport::validate_launch_layout()` treats `Plugin` as needing no rank block at all
  (same as `InProcess`), which is correct for the single-process case this track verified but not
  separately confirmed for a `Plugin` component that itself spans more than one rank.
- **MPI.** Every row above is single-process; `ABI-COLLECTIVE-AGREEMENT`'s own host-run caveat
  applies identically to the provider direction and is not re-stated per row.

## C. Not defects — do not fix these

- **Five mutations report `NOT-ASSESSED` locally.** They need MPI, and a test
  that cannot start exits non-zero whether or not the mutation had an effect.
  Recording that as a kill would be evidence worse than none. Three are in
  `transport-mutations.md` (`peer-plan-skips-the-numbering-check`,
  `peer-plan-drops-the-local-copies`, and `build-plan-skips-the-components-agreement` —
  the last tagged R-30 in its own description, added while fixing that finding) and two
  are in `runtime-mutations.md` (`region-residual-skips-the-cross-rank-reduction`,
  tagged R-10, and `region-failure-abandons-the-rest-of-the-period`, tagged R-09;
  `cross-rank-region-check-fires-unconditionally` is assessable at the base build configuration and KILLED).
- **One mutation is declared equivalent**, with a written bound: `ir.scale`'s
  `unlowerable-step-silently-dropped` (`compiler-mutations.md`) — no graph the
  model can currently express reaches the defensive branch it mutates. The
  backend mutation once declared equivalent here — computing the conservation
  audit before rather than after the negligible-overlap floor — had that
  declaration withdrawn 2026-09-19 and is now an ordinary mutation, killed
  (`backend-mutations.md`).
- **The four `SkippedWithoutMoab` entries are the PARENT's, not FUSE's.** They
  live in `upstream/soma/test/unit/{moab_mesh,remap,remap_order,remap_runtime}_test.cpp`
  and are run by the imported-suite CI leg. FUSE's own suite does have a
  MOAB-dependent test now (`backend.moab_mesh`, 38 cases, registered whenever
  `FUSE_ENABLE_MOAB` is on) — what it lacks is a host run, per C1. Do not read
  those four parent skips as FUSE coverage either way.
- **`sphinx -b html -W` logs one `misc.highlighting_failure` warning in
  `docs/history/soma/plans/soma-phase-5.md`.** A verbatim plan-document CMake
  snippet (Step 5) uses a literal `...` as an elision placeholder, which is not
  valid CMake and trips Pygments' lexer (it recovers in relaxed mode; the HTML
  still renders). Fixing it means editing the historical text, which
  `docs/history/soma/` moved VERBATIM and which the repository owner directed
  (2026-10-12, soma-import track) be left untouched rather than edited to quiet
  the lexer. `python3 -m sphinx ... -W --keep-going` on this tree therefore
  exits 1 on this one warning alone; every other warning this track's docs
  move could have introduced (4 dead links in `BUILDING.md`, 1 in `HANDOFF.md`,
  1 broken anchor in `comm-graph-design.md`) was fixed by a minimal, disclosed
  markup-only touch instead of a suppression.
- **The serial loops in `fuse/num/kernels.hpp` are not dead code** now that
  `space_kernels.hpp` exists. They are the oracle the parallel kernels are
  tested against, and they are what runs where no execution space has been
  chosen.
- **`plans/` is not in the repository.** Task-level plans stay local by design,
  as in both parents; the durable record is `docs/dev/roadmap.md` and the rest
  of `docs/`.
- **A surface remap pair is rejected** (`E-REMAP-NONPLANAR`) rather than
  supported. The planar kernel would return the measure of the xy shadow, which
  is a plausible wrong answer. A surface pair needs a parameterisation.
- **The native mesh's `--migrate` layout-driver mode re-verifies MESH
  conservation, not FIELD transfer.** `docs/dev/native-mesh.md` "what is
  deferred": matching a migrated mesh's entities back to `MpiTransport`'s
  exchange plan, so a component's field DATA (not just its mesh) follows a
  repartition, is not implemented — closer in shape to a distributed remap
  operator than to migration itself. Do not read `layout.migrate.*`'s
  `--check` as evidence the field-transfer path handles a migrated mesh; it
  is not tested at all here.
- **The native mesh's `generated:cubedsphere:N` source has an unwelded seam.**
  Its six panels are independently numbered (`docs/dev/native-mesh.md`); a
  vertex on one panel's edge and the matching vertex on its neighbour's edge
  are distinct global ids, not merged into one. A caller needing a
  seam-welded sphere must do that merge itself.
- **`fuse::native::migrate`'s vertex-ownership rule (minimum rank holding a
  copy) is exercised only by an MPI test that has not run in this sandbox.**
  `resolve_vertex_ownership`'s Allgatherv-based agreement never executes at
  `comm == kNullComm` (every vertex trivially owned by rank 0 there), so
  the base build configuration cannot verify the cross-rank rule itself — only
  `native_migration_mpi_test.cpp` (2-5 ranks) does, and per C1 this sandbox
  cannot start `mpiexec`. The corresponding mutation
  (`vertex-ownership-rule-picks-max-not-min-rank`,
  `build-evidence/mutations/native-mesh.toml`) is `requires = "mpi"` for the
  same reason. D3 (below) adds
  `VertexOwnerIsTheMinimumRankAmongHoldersAfterMigration`, targeted at this
  exact comparison; it is compiled, not yet run, for the same C1 reason.

**Status after four investigation rounds (2026-09-25).** Every term of the discrete energy budget on
the unstable eigenmode has been measured, and each named candidate has been ruled out by
measurement, not by argument:
- the outflow mass correction removes energy, -4.571654e-06;
- a do-nothing outflow moves rho(M) by 0.01%;
- the pressure pin is inactive in this configuration;
- H and H^T D are adjoint to 1.9e-12;
- the conjugate (adjacent-cell) traction leaves rho(M) in 1.1154-1.1173.
The combined scheme still has rho(M) = 1.115466 on the diagnostic grid and 1.108145 on the
production grid (rho^2 = 1.227985 against the measured 1.22796).

**Next experiment, the one that discriminates.** Assemble the SAME discretisation as one monolithic
system (fluid saddle point plus beam, with H and H^T D as the coupling blocks) and take its step map.
- If the monolithic map also has rho > 1, the defect is in the coupled discretisation or the
  linearised transpiration model itself, independent of partitioning.
- If it has rho <= 1, the converged partitioned step is not computing the monolithic solution, and
  the gap between the two is the defect.
Also check whether any carried state is determined only through an average, as p_n was: Newmark's
a_n under the midpoint load is the candidate, since the eigenvector carries 14% of its mass in the
beam acceleration.

### D16. A window member's cold-start priming over-read a smaller provider buffer — FIXED 2026-10-03

Found by ASan on a two-member window-scoped region whose members export fields of different
lengths (Conformance Unit 2, the coupler-placement track's own cross-resolution U/V box graph).
`Runner::Impl::resolve_regions()` (`src/runtime/iteration.cpp`), setting up a window member's
cold-start import priming before window 0 ever runs, read `provider_v0[e]` — the PROVIDER's own
exported-field buffer — for `e` up to `cfg.length`, THIS member's own exported (or, absent an
export, imported) field length, resolved from a different field entirely a few lines earlier.
Whenever the provider's own field was smaller, this read past the end of its buffer: a real
heap-buffer-overflow READ, confirmed by ASan — `SUMMARY: AddressSanitizer: heap-buffer-overflow
iteration.cpp:534 in fuse::Runner::Impl::resolve_regions` on the original box-graph harness
(5/5 runs, identical stack), and reproduced independently on a fresh, minimal two-member window
(mesh "a" 4 cells, mesh "b" 9 cells) built for this fix.

**Fixed** by bounding both affected loops (the import-arc priming loop and the straddle-tail
history seeding loop right after it) to `std::min(cfg.length, provider_length)`, where
`provider_length` is the provider's own exported field's real size
(`provided_ref.value().size()`). When the two lengths differ, priming is necessarily partial —
entries `[prime_count, cfg.length)` are left at whatever `import_slot`/`history` already held
(zero-initialized storage, an empty history list) — a correct cross-resolution prime would need a
real Transport/Remap, which this one-time cold-start shortcut has never attempted even in the
same-resolution case (see the surrounding comment on why the identity-only shortcut exists at
all). Covered by `test/runtime/window_runner_mismatched_length_test.cpp`
(`WindowRunnerMismatchedLength.InitializeAndOnePeriodSucceedWithoutOverflow`) and
`build-evidence/mutations/window-prime-length.toml`'s own row — with the important caveat, recorded
in that row's own comment, that an ordinary (non-sanitized) build does NOT reliably fail on this
mutation (the over-read recovers whatever garbage sits past the small buffer without crashing);
AddressSanitizer is the real oracle for this specific defect, and was run by hand both with the
mutation applied (aborts, heap-buffer-overflow, pointing at the mutated line) and with it reverted
(clean, exit 0).

**Whether this explains the separately-measured same-resolution uniform-collapse finding: NO, not
confirmed.** A prior session found that remapping two IDENTICAL-resolution meshes sometimes
produced a uniform weight matrix instead of a true identity. This defect requires a length
MISMATCH to trigger at all (`provider_length < cfg.length`); a same-resolution pair never
satisfies that condition, so this fix cannot be the explanation, and a focused diagnostic this
session (`Supermesh.ScratchBoxNxMx1IdenticalPairDiagnostic`, test/backend/supermesh_test.cpp, added
then fully reverted) separately confirmed the generator and supermesh builder are correct on an
identical 3x3x1 pair (exact identity matrix, `min_src_coverage=min_tgt_coverage=1`). The
uniform-collapse finding's own root cause remains open.

### D17. `--accel aitken` converges through `--driver runner` but not through `--driver window`, on the identical `AccelConfig` — OPEN, not pursued (Conformance Unit 2's own scope needs no accelerator)

Measured on the Brusselator example (`examples/brusselator/main.cpp`) at its own default
parameters (`--dt-u 0.002 --dt-v 0.003 --window 0.02`, `A=3.4 B=1 alpha=10`):

- `--driver window --accel aitken --end 0.5 --check`: exits 1, no output directory created.
  stderr: `FAIL: run_window() at t=0.060000: coupling window [3/50, 2/25) ran its full 60
  iterations without meeting its convergence criterion` — windows 0–2 (t=0, 0.02, 0.04) converge;
  window 3 (t=0.06) does not, and never recovers even at `--max-iterations 500` (same window, same
  failure, confirmed on `--end 0.1`).
- `--driver runner --accel aitken --end 0.5 --check`: converges throughout —
  `max_iterations=31` (well under the 60 cap), `max_window_balance_error=2.227e-13`,
  `--check: OK`.

Both sides build the identical `AccelConfig{.kind = AccelKind::Aitken}` (default everything else)
from the same CLI flag — confirmed by reading both call sites in `main.cpp` directly (lines ~227
and ~326 as of this session). The asymmetry is real, not a CLI-parsing difference and not a
crash: a genuine convergence disagreement between the two drivers on IDENTICAL physics and an
identical accelerator configuration, isolated to the `--driver window` path specifically (the
direct `WindowCoupler` driver, not `fuse::Runner`).

**Not root-caused.** Aitken's own quasi-Newton relaxation numerics (`src/num/
accel_relaxation.cpp`) are a genuinely separate investigation from Conformance Unit 2's own scope,
and the Brusselator's own conformance case (both Unit 1 and Unit 2) uses no accelerator either
way — `AccelKind::None` throughout. Recorded here, per the lead's own instruction, because the
`--driver window` path is the REFERENCE driver for everything Aitken-accelerated elsewhere in the
suite, and this asymmetry should be understood before Aitken is treated as a conformance case
anywhere.


### ~~D20. `connect()` never released its base communicator, so an accept-mode run crashed in `MPI_Finalize`~~ — FIXED 2026-10-08, awaiting the host

**Measured (the user's host, 2026-10-08).** `parallel.connect.accept.1+1` failed with the acceptor
taking signal 11. Under `lldb` (run *under* `mpiexec` — a participant started without a launcher
dies in MPICH's own singleton init first, which cost one round trip) the crash is
`EXC_BAD_ACCESS` at `dbginit.c:515` in `MPII_CommL_forget`, reading `p->comm_next` through a
pointer of repeating `0xEF` — MPICH's freed-object fill. That function runs while MPI destroys a
communicator, so the crash is MPI tearing down a communicator list that already holds a destroyed
entry, not a fault in this library's own statements.

**Cause.** `MPI_Comm_accept`/`MPI_Comm_connect` leave the two separately launched programs
*connected* in the MPI sense, and `MPI_Finalize` is collective across connected programs. Nothing
in the library or its callers ever called `MPI_Comm_disconnect`: the only one in the tree was in
the LowFive demo. `connect()`'s MPMD path duplicates an ordinary communicator and owes no such
teardown, which is exactly why `parallel.connect.mpmd.*` passed throughout.

**Fix.** `ConnectedWorld` records which teardown its handle owes (`Free` for an MPMD duplicate,
`Disconnect` for a merged connection, `None` for a failed or already-released handle), and
`fuse::transport::disconnect()` performs it, idempotently, refusing with `E-CONNECT-FINALIZED`
if called after `MPI_Finalize`. Both callers release before finalizing: the connect test now
scopes everything that uses the communicator so the `Runner` and the transport's own duplicate
are destroyed first, and the participant ABI holds the handle in its context and releases it in
`fuse_destroy()` — taken out of the context before the delete and released after it, for the same
ordering reason.

**No tier-0 oracle.** `connect()` and `disconnect()` are both MPI-only, and the difference between
disconnecting and freeing is invisible inside one program: it is about whether a second program is
still connected. Tier 1 builds; the proof is the host's own `parallel.connect.accept.*`, and the
mutation `connect-mode-teardown-frees-instead-of-disconnecting`
(`build-evidence/mutations/transport-connect.toml`, `requires = "mpi"`) is marked host-only for
the same reason.

### ~~D24. Every placement got this rank's slice of the whole run~~ — CLOSED 2026-10-09

**What it was.** `NativeMeshBackend::register_default_placements()` generated every
component's placed view as `generate_native_mesh(spec, rank_, num_ranks_)` -- this
process's slice of a decomposition over the ENTIRE base communicator, the same answer
for every component. Its own comment said why that was thought safe: "every placement
this backend can ever generate is the SAME cheap, deterministic `(rank_, num_ranks_)`
view regardless of which component asks for it (tier-0: one process IS every
placement)". One process is indeed every placement. A component whose rank set is a
strict subset of the run is not, and a coupler on its own processor set is exactly
that.

**How it surfaced.** The first MPMD run with each component in its own executable
(host, 2026-10-09). U, alone on rank 0, received 96 cells of a 176-cell mesh; the
coupler, alone on rank 2, received the other 80. The two placements had no global ids
in common, so the exchange plan found no source for what the destination wanted:
`destination union rank 1 wants global id 178 ... but no source rank owns it`. The
message named neither the channel nor the sides until `src/runtime/runner.cpp` was
taught to append them, which is what made this diagnosable at all.

**Fix.** A placement is decomposed over its own `RankSet`, indexed by this process's
position within it, so a one-rank component holds its whole mesh wherever that rank
sits in the base communicator. A component that does not live on this rank gets an
empty view of the right dimension -- the contract `MoabMeshBackend::placed_view`
already answered outside a processor set -- rather than a failure, because every rank
resolves every placement (the exchange plan is collective) while only hosting ranks
read one. On one process every rank set is `{0}` at position 0 of 1, so no tier-0
answer changed; the suite did not move.

**Pinned by** `NativeBackend.APlacementIsDecomposedOverItsOwnRanksNotTheWholeRun` and
`NativeBackend.APlacementSpanningSeveralRanksStillSplitsAmongThem`, both of which need
no MPI -- the backend takes its position in the run as a constructor argument, so a
serial test can be rank 2 of 3. The mutation
`every-placement-gets-this-ranks-slice-of-the-whole-run` restores the old line and they
fail with exactly the arithmetic of the defect: 12 of 36 for the component alone on
rank 2, and 24 for one spanning ranks 1 and 2.

**Not the whole of multi-executable yet.** This clears the exchange plan's inputs; what
the MPMD rows do beyond it is unmeasured until they run again.

### D26. `FUSE_LINALG_BACKEND=builtin` has not configured since the examples linked Eigen directly

**Measured 2026-10-10 on a clean export of `9d9ab7a`** (before the Krylov and Schur
solvers landed): `cmake -DFUSE_LINALG_BACKEND=builtin` fails at configure with
`Target "fuse_verification_fsi_stokes_mms_test" links to: Eigen3::Eigen` (and the same
for `fsi_beam`, `fsi_coupled`, and every other test or example that names
`Eigen3::Eigen` directly -- 11 sites in `test/CMakeLists.txt`, 8 in
`examples/CMakeLists.txt`, each with a comment saying it relies on `eigen` being the
default). The option still selects the least-squares facade's implementation and is
reported by `fuse::linalg::configured_backend()`, but the configuration as a whole is
not buildable, and nothing refused it at the option.

**What the new solvers do about it.** `SchurSlabSolver` needs Eigen's `SimplicialLDLT`
and dense `LDLT`, so `src/linalg/schur_slab.cpp` is compiled only for the `eigen`
backend; for `builtin`, `schur_slab_unavailable.cpp` keeps the class and `factor()`
refuses with `E-LINALG-SCHUR-BACKEND`, so `fuse::linalg` itself still builds without
Eigen (measured: `--target fuse-coupler-linalg` builds clean under `builtin`). PCG and
GMRES are header-only and need nothing. Their tests compare against Eigen and link it
like the examples do.

**Open:** either `builtin` becomes a real configuration again (the Eigen-linked tests and
examples gated by `if(FUSE_LINALG_BACKEND STREQUAL "eigen")`, with a configure-time
message listing what is skipped) or the option is retired. Not decided here.

### D30. Two bit-for-bit tests fail by one ulp in a Release build — a decision on floating-point contraction — DECIDED AND CLOSED 2026-10-12

**Measured 2026-10-10 on a clean export of `cd26801`, `-DCMAKE_BUILD_TYPE=Release`, otherwise the
tier-0 configuration** (found by the example-documentation pass, which built the `tier0-release`
preset; reproduced by the lead):

- `DistVectorOneRank.ArithmeticMatchesAPlainLoopBitForBit` (`linalg.dist_vector`):
  `-2.159199103107206` vs `-2.1591991031072011` — one ulp.
- `SpaceKernels/0.DeterministicReductionsMatchTheChunkedReferenceExactly` and the OpenMP leg
  (`num.space_kernels`): `1.2895225459088601e+34` vs `1.2895225459088599e+34` and
  `1.3007153382180115e+34` vs `1.3007153382180113e+34` — one ulp each.

Every one of the 179 tier-0 rows passed in Debug, the configuration every commit that cycle was
verified in, and nothing in the build set a contraction policy (`-ffp-contract`, `-ffast-math` and
`-fp-model` appeared nowhere in `CMakeLists.txt`, `cmake/` or `src/CMakeLists.txt` — grepped).

**Correction of this entry's first account (2026-10-12).** It said that Apple clang contracts
`a*b + c` into a fused multiply-add at `-O2` and not at `-O0`. That is false. The compiler's default
is contraction *on* at every optimisation level, and `docs/dev/testing.md` ("Floating-point
contraction policy") has the disassembly. Debug agreed with the plain loop because at `-O0` neither
the library's loop nor the test's reference loop is vectorised or unrolled, so both are the same
sequence of fused instructions. At `-O2` the vectoriser splits the library's runtime-length
reduction into an always-unfused vector body plus a remainder that is fused by default, while the
test's reference loop, whose length is a compile-time constant, is fully unrolled and fused. One
limit is recorded there too: a minimal standalone reproduction of the same source did not show the
split, so the account comes from the real objects, not from an isolated case.

**Decision (the owner's, 2026-10-12): offer both, chosen per use.** `FUSE_FP_REPRODUCIBLE`
(`cmake/FuseFpPolicy.cmake`, default **ON**) applies `-ffp-contract=off` to every C, C++ and Fortran
target of this repository and to none of the vendored sources; **OFF** keeps the compiler's default
and is the `tier0-release-fast` preset. ON changes Debug's numbers as well as Release's, because the
flag reaches `-O0` too. Under OFF the three bit-identity assertions become the forward-error bound for
recursive summation, `2 γₙ Σ|xᵢ|` (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
secs. 3.1 and 4.2), with a worst observed ratio of 0.0060 (`DistVector::dot`, n = 37) and 0.0031
(`deterministic_norm_squared`, n = 256). A probe test, `base.fp_policy`, shows that the flag reached the
compiler, and it fails in both Debug and Release when the call that applies the flag is removed.

**What turning it ON moved.** Two tests outside the three named cases changed under ON:
`num.accelerator` no longer pins rounding-dependent iteration counts (its two cases are restated as
the properties they name), and `verification.fsi_energy_diag` now evaluates the energy budget on the
beam's own oscillatory mode, because the old "dominant" eigenvalue is Newmark's acceleration
passenger, λ = −1, whose energy is rounding noise. One sub-assertion of that test failed under OFF until D34 was
found and fixed (the test's own step rounding, not the contraction policy).

Verified 2026-10-12 from a clean export: Debug (policy ON), 179 of 179 quick rows and 16 of 16 slow
rows; Release ON and OFF, the 5 touched rows (OFF passes after D34's fix).

### D27. The native backend's remap halo has no `accumulate_remap_halo` counterpart

**Closed for the forward direction, 2026-10-10 ("Halo" track):** `fuse::native::NativeMeshBackend`
now builds a planar/volumetric remap halo (`build_native_remap_halo`,
`src/backend/native_halo.cpp`; docs/dev/native-halo.md) via the same `build_remap_halo` ->
`build_weights` -> exchange -> `apply_weights` sequence `remap_plan.hpp` documents for MOAB, and
`remap_halo_pattern()`'s `fuse::linalg::GhostPattern` plugs directly into
`fuse::linalg::make_mpi_exchanger` for the FORWARD fill (owner's field value -> halo copy) before
`apply_weights`.

**Open:** `MoabMeshBackend::accumulate_remap_halo` -- the REVERSE exchange (each halo slot's
contribution added back to its owner, then zeroed), built specifically for a conservation audit
that sums a source cell's weight contributions across every rank holding a copy of it -- has no
native counterpart. `test/backend/native_halo_test.cpp`'s own column-mass-by-global-id check
does this bookkeeping in the TEST, not through a library call; nothing in the Halo track's scope
needed a library-level reverse exchange, but a future native-backend conservation-audit tool would
want one.

### ~~D28. U's own species-core solver seam is closed; the mpmd.5 row that would first exercise it across a real two-rank U is still host-only~~ — CLOSED 2026-10-10 (host run, see D25's host paragraph)

**Closed this session (species-core decomposition track): the implicit diffusion SOLVE**, the one
piece D-series' own grid-decomposition entries left open (see "The solver seam" in
`docs/user/brusselator.md`), now decomposes with U's own grid. `UCore`/`VDiffusionCore`
(`examples/brusselator/species_core.hpp`) select a `DiffusionSolver` per `PhysicsConfig::solver`
(`SolverKind::Auto`/`Direct`/`Schur`/`Pcg`/`Gmres`, a new `--solver` flag on `u_main.c`/`v_main.f90`):
`SchurDiffusionSolver` wraps the Solvers track's `fuse::linalg::SchurSlabSolver`;
`KrylovDiffusionSolver` assembles the same operator as a `fuse::linalg::DistSparseMatrix` and solves
it with `fuse::linalg::pcg`/`gmres` under a `JacobiPreconditioner`. `u_main.c` now forms its own
physics-only sub-communicator (`MPI_Comm_split` on `is_physics_rank`), reads `u_rank`/`u_size` from
it instead of assuming `nx*ny` is the whole grid, and passes both plus the new `--solver` string and
an `MPI_Comm_c2f` handle through the participant ABI (`bruss_u_create`'s four new trailing
parameters). V's own CLI/ABI plumbing (`--ranks-u`/`--ranks-v`/`--solver` on `v_main.f90`,
`bruss_v_create`'s matching trailing parameters) is prepared identically, but `VParticipant::make`
still refuses outright at `num_ranks() > 1` (`E-BRUSS-V-DECOMPOSED-REMAP`) — V's two-sided remap
routed through FUSE's distributed weights is a separate ("Halo") track's own work, explicitly out of
scope here. **Superseded by D29**, which closes exactly that gap.

**Verified in the sandbox:** a new in-process N-partition harness,
`test/verification/brusselator_species_core_decomposition_test.cpp` (9 tests, all passing), measures
each decomposed solver against a `num_ranks() == 1` `DirectDiffusionSolver` reference — `Schur`
agrees to 2.4e-14/4.5e-14 (U, 2/3 partitions) and 2.7e-15 (V); `Pcg`/`Gmres` agree to 5.9e-14/5.5e-14
at the chosen `solver_tolerance = 1e-13` (swept from `1e-8` through `1e-13` to derive that default —
see "Species-core solver decomposition" in `docs/user/brusselator.md`). The full tier-0 suite passes
179/179 after the change, and `fuse_example_brusselator --end 0.1 --check` remains byte-identical to
the committed reference (`final_mean_u`/`final_mean_v` unchanged to all printed digits) — `Auto`
resolves to the original `Direct` path at `num_ranks() == 1`, so every existing single-rank caller is
unaffected. Three new mutation rows (`build-evidence/mutations/brusselator.toml`:
`brusselator-schur-coefficient-wrong-sign`, `brusselator-krylov-matrix-diagonal-drops-mass-term`,
`brusselator-krylov-matrix-drops-global-row-offset`) are each hand-proved against this harness; the
third does not fail cleanly — it HANGS (a stranded collective, the failure mode D21/D25 both name),
proved instead by a `timeout`-bounded rerun.

**Still host-only, same shape as D25's own gap.** `u_main.c`'s new `MPI_Comm_split` is collective
over the WHOLE shared `MPI_COMM_WORLD` an mpmd launch gives every segment (`u_main.c`, `v_main.f90`,
`coupler_main.c` all numbered into one world) — so `v_main.f90` and `coupler_main.c` each now carry a
MATCHING `MPI_Comm_split(..., MPI_UNDEFINED, ...)` call at the equivalent point in their own startup,
purely so U's own split does not strand them. (D29 replaces this one-color-plus-`MPI_UNDEFINED`
form with a single three-color split every one of the three programs now makes identically.)
This choreography across three separately-compiled
executables has never been run: this sandbox cannot start MPI at all (C1), and D25 itself recorded
that `mpmd.5` — the one row where U would actually span more than one rank — was never run even
before this change, because it exceeds the four-task rule. `mpmd.3`'s own U segment is one rank, so
it cannot exercise `MPI_Comm_split` with a non-trivial physics group at all. The next host run that
matters for this track is `mpmd.5` with `--solver schur` (or `pcg`/`gmres`) on U's own two ranks —
confirming both the split choreography above and the decomposed solver agreement this entry's own
tier-0 numbers can only approximate with an in-process thread-barrier stand-in for a real
`Reducer`/`GhostExchanger`.

### ~~D29. V's two-sided remap over an adopting halo measurably disagreed with the single-rank reference~~ — CLOSED 2026-10-10 (root-caused and fixed)

**Closed this session.** The guard added when this entry was opened — `VParticipant::make_from_grids`
refusing any decomposition whose remap halo adopted cells, in either direction, with
`E-BRUSS-V-REMAP-ADOPTION` — is REMOVED: `ThreePartitionVParticipantRefusesAnAdoptingHalo` and the
mutation row `brusselator-vparticipant-adopting-halo-not-refused` are both gone, superseded by the
tests and mutation row below.

**Root cause, measured by instrumenting (not guessed from reading).** `VParticipant::step()`'s own
`q_native_v` buffer (the `apply_weights(weights_uv_, ...)` destination for the U->V remap) and
`remap_state_to_u()`'s own `u_native` buffer (the `apply_weights(weights_vu_, ...)` destination for the
V->U remap) were each sized from this rank's OWNED cell count (`n_v_`/`n_u_`, cached at `make_grids()`
time, BEFORE either direction's halo is built). `RemapWeights::n_rows` (`remap_plan.hpp`: "target
entities (local)") is documented, correctly and intentionally, to mean the target mesh's CURRENT local
entity count -- owned plus ghost, the same "local = owned plus ghost" convention `MeshView::entity_count`
itself documents (`mesh_view.hpp`) -- and `fuse::build_supermesh_weights` (`src/backend/supermesh.cpp`)
honours exactly that: `w.n_rows = tgt.entity_count(dim)`. `apply_weights` (`remap_plan.hpp`)
unconditionally zeros and writes `dst[0, n_rows)`.

U and V are each the SOURCE of one halo direction and the TARGET of the other, on the SAME backend: V is
widened (gains adopted cells) as the source of the V->U halo, and that SAME V mesh is also the TARGET of
the U->V weights (`w_uv`), built AFTER both halos. Whenever V was ALSO widened (because this rank
adopted V cells for the V->U direction), `weights_uv_.n_rows` is `owned_size(V) + adopted_v`, not
`owned_size(V)` -- and `q_native_v`, sized from the stale pre-halo `n_v_`, was too small by exactly the
adopted count. `apply_weights`' unconditional zero-and-write loop then wrote past the end of that
`std::vector<double>` -- a heap buffer overflow, not merely a wrong value. The symmetric case (U widened
as the source of the U->V halo, inflating `weights_vu_.n_rows` beyond `n_u_`) overflows `u_native` in
`remap_state_to_u()` the same way. Measured on the instrumented scratch build (U 16 rows as 6/5/5, V 24
as 8/8/8, 3 ranks, `FUSE_BRUSS_DEBUG=1`): rank 0 traced `weights_uv(n_rows=108, n_cols=48)` against
`n_v_=96` (12 extra rows -- exactly its own measured `adopted_v(vu)=12`); ranks 1 and 2 traced an
analogous overflow of `u_native` from their own nonzero `adopted_u(uv)=8`. This explains both original
symptoms precisely: the overflow corrupts whatever heap memory follows the undersized vector, read back
as though it were real row values from the very FIRST `apply_weights` call (the measured 3.1e-4
`exchange_in` disagreement present from sub-step 0, never growing -- exactly "the error enters once");
and on later steps, the corrupted heap trips an `Eigen::Block.h` bounds assertion when the diffusion
solve (Schur or Pcg alike -- neither solver's own code is involved) reads memory the overflow had
clobbered.

Of the candidates this entry's own "Open" note listed when it was written, the fix matches candidate (b)
("a buffer sized from the PRE-halo entity_count") exactly, once sharpened: it is not that something was
cached from a view at `make_grids()` time in general (the grid's own `nx`/`ny`/`owned_rows`/
`native_of_canonical_` are all fine, precisely because they describe only the UNWIDENED owned block,
which widening never touches) -- it is specifically the two `apply_weights` DESTINATION buffers, each
sized from the wrong rank's wrong-direction owned count instead of from the weights matrix that actually
produced the data going into it.

**Fix** (`examples/brusselator/species_core.hpp`): `step()` now sizes `q_native_v` from
`weights_uv_.n_rows`; `remap_state_to_u()` now sizes `u_native` from `weights_vu_.n_rows`. Both are safe
to read back through `to_canonical()` unchanged: `to_canonical()` addresses only `native_of_canonical_`
entries, which are all built from the UNWIDENED view at `make_grids()` time and so never point past
`owned_size()`, regardless of how large the weights-sized buffer actually is. No refusal is needed any
more.

**Verified correct on the previously-failing MISALIGNED split** (U 16 rows as 6/5/5 against V 24 as
8/8/8, 3 ranks -- adopting rank 0 "0 U, 12 V", rank 1 "8 U, 12 V", rank 2 "8 U, 0 V" cells, traced under
`FUSE_BRUSS_DEBUG=1`): `test/verification/brusselator_species_core_decomposition_test.cpp`'s
`ThreePartitionVParticipantMatchesSingleRankMisaligned` (`SolverKind::Schur`, 6 sub-steps) measures
`max_abs_diff = 1.732e-14` against the single-rank reference, and
`ThreePartitionVParticipantMatchesSingleRankMisalignedPcg` (`SolverKind::Pcg`, 20 sub-steps, to rule out
any slow growth the original 6-sub-step run had no room to see) measures `max_abs_diff = 5.262e-14`,
`max_exchange_in_diff = 1.388e-17` -- both round-off, both solver kinds, no crash. The full tier-0 suite
(ctest, which registers one entry per test BINARY, not per `TEST()` case -- `ThreePartitionVParticipant
RefusesAnAdoptingHalo`'s removal and the two new cases' addition are both internal to the existing
`verification.brusselator_species_core_decomposition` binary) passes 179/179, unchanged from before this
fix. The halo-construction pipeline itself was never the defect -- `test/backend/native_halo_test.cpp`'s own
`ThreeRanksMisalignedAdoptsExactlyWhatTheOracleComputes`/
`ThreeRanksMisalignedCompleteWeightsAndLinearFieldReproduction` already proved it correct at this split,
one direction at a time; this entry's own bug only appears once BOTH directions' halos are built on the
SAME backend, each mesh widened as the source of one direction and read as the (now also widened) target
of the other -- a scenario those single-direction tests never construct.

**Mutation.** `brusselator-vparticipant-adopting-halo-not-refused` is retired (the refusal it killed no
longer exists). `brusselator-vparticipant-stale-adopted-ghost-not-exchanged`
(`build-evidence/mutations/brusselator.toml`) replaces it: skipping `exchanger_uv_->update()` in
`step()` (leaving the adopted/ghost slots at their zero-initialized default -- "stale" -- instead of the
real cross-rank values) is caught ONLY by the misaligned tests above (the aligned ones adopt zero cells,
so a skipped update is indistinguishable from a correct one there -- every ghost slot is empty either
way). Hand-proved 2026-10-10 on the unfiltered test binary: with the mutation applied,
`ThreePartitionVParticipantMatchesSingleRankMisaligned`'s `max_abs_diff` rises from `1.732e-14` to
`3.170e-03` and `...MisalignedPcg`'s from `5.262e-14` to `8.408e-03`, both far past the `1e-11` bound;
both FAIL cleanly. File restored and verified byte-identical with `cmp` before submission.

**The MOAB-backed builder is NOT a second instance (lead, measured 2026-10-10).** The
sub-agent flagged `w.n_rows = tgt.cells.size()` in `src/backend/intx_weights.cpp` (three
sites) as "the same row-count-inflation shape". It is the same CONVENTION, not a defect: every
builder -- supermesh (`w.n_rows = tgt.entity_count(dim)`), MOAB, MLS -- sizes the weights over
the target's whole local view, owned and non-owned alike, which is what `RemapWeights::n_rows`
is documented to mean. The defect here was a CALLER (this example) allocating the destination
from a count it had cached before the view was widened. The Runner, the library's own caller,
never does that: it checks `weights->n_rows != out.elements` and refuses with
`E-RT-REMAP-SIZE` (`src/runtime/runner.cpp`) rather than applying. Nothing to fix in the
builders.

**No longer host-only: `mpmd.5` passed on the host 2026-10-10 (D25's host paragraph has the
numbers).** The text below is the state before that run. The `mpmd.5` row (`--ranks-v 2`, the one row where V
spans more than one rank) could not be run in this sandbox (C1: MPI cannot start here). This entry's
own fix removed the one KNOWN correctness gap that stood between that row and a real host run -- the
`u_main.c`/`v_main.f90`/`coupler_main.c` three-color `MPI_Comm_split` choreography, and `v_main.f90`'s
cross-check of `v_rank`/`v_size` against `--ranks-v`, are unaffected by this change and remain
unexercised past a build check.
### ~~D25. A separately launched participant could not run a coupling window~~ — CLOSED 2026-10-09 (host-measured)

**Found by the first MPMD runs that reached `initialize()` (D24 cleared the exchange plan),
on the host, 2026-10-09.** Four defects in sequence, each exposed by the fix before it;
all four fixed, and `verification.brusselator.participants.mpmd.3` passes with
`max_window_balance_error`/`global_balance_error`/`max_window_exchange_error` all within
the check's thresholds, final means matching the single-process example to 2 ulp in
`mean_u` (1.5865861881738454 vs 1.5865861881738459, 4.4e-16) and 5 ulp in `mean_v`
(1.998222640303883 vs 1.9982226403038819, 1.1e-15).

1. **A window member's storage was resolved on every rank.** `resolve_regions()` read every
   member's native Provides/Requires storage from the registry on every rank; a separately
   launched participant binds only its own fields, so `U`'s rank refused `E-RT-UNBOUND` on `V`'s
   export. Unseen by `parallel.window_runner_mpi` because its `ComponentSet` holds every
   component on every rank. The binding is now gated on `cfg.hosted`; every reader of those
   pointers already tolerated null.
2. **…which made the initialize-time seed chain's selection per rank.** The chain that
   delivers each member's initial export to its consumer (Transport → Remap → Epoch) was
   skipped where `cfg.required == nullptr` — after fix 1, on every non-hosting rank. Measured
   with the macOS `sample` tool: all three programs in `MPI_Alltoallv`, U and V from this chain
   inside `Runner::initialize()`, the coupler from `run_region()`'s first real Transport — the
   collective mismatch the standing rule names. The chain's selection is now structural
   (`window_import` valid and in range), and the per-rank seeding that follows is guarded
   separately. The code's own comment had marked this collective safety "NOT YET INDEPENDENTLY
   VERIFIED"; this is the verification.
3. **Both participants carried a stale 3×3/4×4 grid**, from before `43fe58d` moved the graph
   tool to 16×16/24×24, so they registered 9-element buffers against 256-cell fields. Nothing
   refused it: the ordinary bind's `E-RT-SIZE` is bypassed by the window member's direct
   registry read, and the priming's own size guard quietly skipped. The run then produced a
   trajectory irreconcilable with the single-process example (first-window exchange 40% short).
   Now: the same read refuses `E-RT-SIZE` (`WindowRunner.RefusesAWindowMemberWhoseExportIsNotTheFieldsSize`,
   mutation `window-member-size-check-dropped`), and the grid and physics constants live in ONE
   header, `examples/brusselator/participants/brusselator_grid.h`, included by the graph tool
   and U and read by the Fortran V through `bruss_grid_dims()`.
4. **Neither participant put its initial export into the registered storage.** The Runner
   primes the first window's t=0 sample from whatever the export storage holds at
   `initialize()`; the in-process components hold their constructor's value there, the tier-0
   harness copies it in by hand, the standalone programs left `calloc`'s zero. U also seeded
   its initial rate from a zero V. Both now do what the harness does: `bruss_u_provided()` /
   `bruss_v_provided()` into the export before `initialize()`, and U reads a throwaway V's
   initial `P_VU(V0)` for its own initial rate.

**A startup-contract limitation worth naming, not deciding here:** item 4 means a participant
must know its partner's initial export before `fuse_initialize()`. A coupler-delivered initial
exchange (imports delivered before exports are frozen) would remove the need for U to
reconstruct V at all. Recorded; the user's call.

**Host run 2026-10-10 (main after `e76881c`): `verification.brusselator.participants.mpmd.5`
PASSES** -- U on two ranks, V on two, the coupler on one, five programs' worth of ranks under one
`mpiexec`, 1.37 s. Two participant defects stood in front of it, both found by the new size check
and the merged log: `v_main.f90` registered the whole U grid (256) for its two U-grid fields where
the plan sizes them per V rank (128) -- refused `E-RT-SIZE`, now `bruss_owned_size(nx_u, ny_u,
v_rank, v_size)` as `u_main.c` already did; and both participants logged a per-rank partial mean
with every rank writing the same `--log` path -- now the species' global mean through the core's
own reducer (`bruss_u_state_mean`/`bruss_v_state_mean`, a collective every rank calls) and rank 0
alone writes. Measured against the single-process example (`mean_u` 1.5865861881738459, `mean_v`
1.9982226403038819): the 3-rank row is now BIT-IDENTICAL in both means (0 ulp; the earlier 2 and 5
ulp were the participants' own sequential summation against the example's Eigen sum); the 5-rank
row differs by 8.9e-14 in `mean_u` and 9.4e-14 in `mean_v` (relative 5.6e-14 / 4.7e-14) -- the
decomposed Schur solve's own round-off, consistent with the 2-partition in-process bound of
4.5e-14 -- with `exchange_in`/`exchange_out` within 1e-14 of the 3-rank values and the
conservation check within its thresholds.

The accept rows remain blocked by D23.

### D23. Accept/connect cannot complete on MPICH's `ch3:nemesis` device when both programs share a node

**Status: not a defect in this library; it needs a different MPI build.** Every
`parallel.connect.accept.*` row, and the `verification.brusselator.participants.accept.*`
rows with it, aborts inside `MPI_Intercomm_merge`. Measured on the host 2026-10-09, with
a debugger on each side in turn -- the two stacks are identical frame for frame:

```
Assertion failed in file ../src/mpid/ch3/channels/nemesis/include/mpid_nem_inline.h
at line 325: vc_ch->is_local
  MPID_nem_mpich_sendv_header  <- the SHARED-MEMORY send path
  MPIDI_CH3_iSendv
  MPIDI_CH3_EagerContigIsend
  MPID_Isend
  MPIC_Sendrecv
  MPIR_Intercomm_merge_impl
  MPI_Intercomm_merge
  fuse::transport::connect_pairwise
```

MPICH takes its shared-memory send path for a virtual connection whose own `is_local`
flag says it is not shared-memory-reachable, which is exactly what a peer obtained
through `MPI_Comm_accept`/`MPI_Comm_connect` is: a process in a DIFFERENT job, with no
shared-memory segment in common, even when it happens to sit on the same node. Merging
a dynamically established intercommunicator is precisely what `MPI_Intercomm_merge` is
specified for, and `connect_pairwise` passes a different `high` on each side as it must,
so there is nothing here for this library to fix.

`MPIR_CVAR_NOLOCAL=1` does not help (measured): the shared-memory path is taken
unconditionally at that point rather than being chosen from the flag.

The MPMD path is unaffected and all `parallel.connect.mpmd.*` rows pass, because there
every peer belongs to ONE job launched by one `mpiexec` and is genuinely
shared-memory-reachable.

**Detected at configure time since 2026-10-09.** `tools/probe-mpi-dynamic.sh`
launches two jobs on this machine and performs the sequence `connect()`
performs; `cmake/FuseDependencies.cmake` sets `FUSE_ENABLE_CONNECT_ACCEPT` from
the verdict. On an MPI that cannot carry the mode the accept rows are no longer
registered, and a `connect`-mode component is refused at `connect()` with
`E-CONNECT-ACCEPT-UNAVAILABLE` naming MPMD as the way out, rather than aborting
inside MPI. The probe distinguishes "could not test" from "measured broken" and
leaves the mode ON for the former -- a login node where `mpiexec` needs an
allocation is the ordinary case, and disabling a working capability there would
be worse than a row that fails loudly.

**What would clear it**, in order of preference:

1. An MPICH built with `--with-device=ch4` -- the default since MPICH 4.0, with a
   different path for dynamic connections. The host's build is
   `--with-device=ch3 --enable-g=most --enable-fast=none`; the assertions that make
   this visible are compiled in by `--enable-g=most`, and a release build would take
   the shared-memory path on a connection that has none rather than abort.
2. Open MPI, whose dynamic process management does not share this limitation.
3. Two separate nodes, which removes the same-node misclassification. Not available
   on a laptop, and not a fix.

Until then the accept rows are expected to fail on this host, and the accept-mode code
path they cover is unverified by measurement. Everything the accept path shares with
MPMD -- the launch-spec round trip, the port file, `connect()`'s own refusals, the
teardown of D20 and the intercommunicator disconnect -- is covered by rows that do pass.

**Update 2026-10-12 — the MPI was reinstalled, the cause named above is gone, the rows have not run yet.** The
host's MPICH is now 5.0.0a1 with device `ch4:ofi` (`mpichversion`), and `tools/probe-mpi-dynamic.sh` exits 0:
accept/connect works between two same-node jobs. This entry stays open until `parallel.connect.accept.*` and
`verification.brusselator.participants.accept.3` have run on that MPI; the configure step could not show it,
because the verdict it had cached was stale (D35).

### ~~D22. A shared Kokkos aborts the process at exit~~ — CLOSED 2026-10-09

**What it was.** `Kokkos::initialize()` registers its own exit hook, and that hook ran
at a point where the libraries Kokkos itself sits on -- the OpenMP runtime above all --
had already been torn down. The process finished its work, printed every expected
value, and then died: `libc++abi: terminating due to uncaught exception of type
std::__1::system_error: mutex lock failed: Invalid argument`, locking a mutex in an
image that was already gone.

None of this was new code. It was invisible for as long as Kokkos was a STATIC library
linked inside `libfuse-abi`, destroyed along with the library that held it; it appeared
the day Kokkos was rebuilt as a shared library on this lab's machine (2026-10-09), when
it became an image of its own with its own place in the teardown order.
`abi.install_consumer_c` and `abi.install_consumer_fortran` both failed exactly this
way -- after printing `installed C consumer ok` and all four expected values -- and an
aborting MPMD participant printed the same message.

**Fix.** `ensure_kokkos_initialized()` registers a `std::atexit` hook of its own,
immediately after `Kokkos::initialize()`. `atexit` is LIFO, so this hook runs BEFORE
Kokkos's, while the runtime underneath is intact; Kokkos's own hook then finds itself
already finalized and does nothing. This is not finalizing from `fuse_destroy()`, which
that function's own comment rules out for good reason -- by the time an atexit hook
runs, no caller can create another context.

**Pinned by** the two install-consumer rows, which are a real oracle for it, and the
mutation `kokkos-is-left-to-its-own-exit-hook`.

**Note for a future reader.** Both failures of 2026-10-09 came from Kokkos changing
from static to shared, and only one of them was ours in this sense. The other -- two
copies of Kokkos's process-wide state, one in the participant executable and one in
`libfuse-abi.dylib`, with `fuse_create()` initialising one and the mesh `View` checked
against the other -- went away with the shared build and needed no source change. If a
future install is static again, that one returns: the symptom is `Kokkos ERROR: ... is
being constructed before initialize() has been called` from every participant that
links C++, while a pure-C one (the Brusselator's coupler) reaches the same point in
silence.

### ~~D21. A launched coupler's mode was lost in the file, hanging every MPMD run~~ — CLOSED 2026-10-09: the MPMD rows reach `connect()` and beyond on the host (D25)

**Measured (the user's host, 2026-10-08).** `verification.brusselator.participants.mpmd.{3,5}` timed
out at 120 s with `combined.log` zero bytes and no `u.log`/`v.log` at all — nothing had started.

**Cause.** `connect_mpmd()` agrees the MPMD layout with an `MPI_Allgather` of `MPI_APPNUM` over the
whole world, so every rank of that world must call `connect()`. The coupler's own program does call
`fuse_setup()` with `my_component = "coupler"`, and `brusselator_graph_tool.cpp` does build
`LaunchSpec{Mpmd, appnum = 2}` for it — but `CouplerDecl` had no `launch` field, `add_coupler()`
dropped it, and the writer emitted only `name`, `index` and `ranks` for a coupler. The mode
therefore never survived the file the three programs load, the coupler came back in-process, its
rank never called `connect()`, and U and V waited in the collective for a participant that was
never going to arrive. The example had noticed half of this and worked around it by building the
coupler's `ComponentDecl` by hand with `launch` set; that fixed the in-memory graph and not the
file, which is the only thing the other two programs see.

**Fix.** `CouplerDecl` carries a `LaunchSpec`, `add_coupler()` passes it through, and the YAML
writes and reads a coupler's `launch:` block with the same helpers the component entries use —
omitted entirely for an in-process coupler, so files written before this are byte-identical.
No descriptor version bump: the launch record was already part of every component record's bytes
(`descriptor.hpp`'s v9), and a coupler is a component-array entry.

**Pinned by** `YamlIo.ALaunchedCouplerKeepsItsLaunchModeAcrossTheRoundTrip` (mode, appnum and the
canonical bytes) and `YamlIo.AnInProcessCouplerWritesNoLaunchBlock`, with the mutation
`a-couplers-launch-mode-is-dropped-by-the-writer`. The hang itself is host-only.

### D18. `fuse::transport::connect()`'s `Connect`-mode path supports exactly two components; a third refuses `E-CONNECT-SPEC` — OPEN, not pursued (Conformance Unit 2 routes around it)

`connect_pairwise()` (`src/transport/connect.cpp`) reads, verbatim:

```
if (order.size() != 2) {
  return Failure{"E-CONNECT-SPEC",
                 "pairwise connect() supports exactly two connect-mode "
                 "components; the graph declares " + std::to_string(order.size())};
}
```

`order` is `connect_mode_components()`'s own fixed, graph-declared order of every component whose
`LaunchSpec::mode` is `Connect` (not `InProcess`, `Mpmd`, or `Plugin`). Three or more such
components refuse outright, before any MPI call happens — `LaunchMode::Mpmd` has no such limit
(launcher-style MPMD is one shared `MPI_COMM_WORLD` from the start; `MPI_APPNUM` checking scales to
any number of segments), but true separate-executable launches (`MPI_Open_port`/`MPI_Comm_accept`/
`MPI_Comm_connect`, then `MPI_Intercomm_merge`) are implemented as a single pairwise handshake,
fixed by `MPI_Intercomm_merge(intercomm, i_am_first ? 0 : 1, &merged)` into exactly the two-member
ordering `connect_pairwise()`'s own `order` establishes.

**Found this session** (Conformance Unit 2, Time Windows track) while scoping
`examples/brusselator/participants/{u_main.c,v_main.f90}`: the participant graph has THREE
components (U, V, and the coupler the Defect 1 fix added), so an `accept`-mode run with all three
as separate `Connect`-mode executables is not supported today.

**Routed around, not fixed, per the lead's own ruling**: a processor set is ranks, not
executables — the coupler's ranks are declared `LaunchMode::InProcess` and live inside the
*acceptor* executable's own `MPI_COMM_WORLD` (U's executable launched with its own ranks plus the
coupler's), so `accept` mode is two real `Connect`-mode executables (U+coupler, and V) merging
through the existing pairwise path, while `mpmd` mode is three ordinary app segments (no pairwise
limit to route around there at all). Both modes still exercise the SAME three-component graph; only
which executable's `MPI_COMM_WORLD` the coupler's ranks happen to sit inside differs.

**The generalisation, sketched but not implemented.** `connect_pairwise()`'s own numbering —
`connect_mode_components()`'s fixed graph order, with `MPI_Intercomm_merge`'s second argument
(`high` in the MPI standard's own terms) fixed by position in that order rather than by
acceptor/connector role — already IS the mechanism an N-way version would reuse, just run `N-1`
times instead of once: component 0's world and component 1's world merge first (exactly today's
path) into an intermediate communicator spanning `{0,1}`; that intermediate communicator then plays
the "local" side of a SECOND `MPI_Comm_accept`/`MPI_Comm_connect` against component 2's own world
(`MPI_Comm_accept`'s local-communicator argument is not required to be `MPI_COMM_WORLD` — any
communicator works, which is exactly what makes a chain of merges possible rather than only a
single pairwise one), producing a communicator spanning `{0,1,2}`; and so on, in
`connect_mode_components()`'s own fixed order, until every `Connect`-mode component has joined.
Which single rank of the growing intermediate communicator plays "acceptor" for the NEXT merge (and
where its port name is published) is the one real design choice a chain adds beyond today's single
pairwise case — reusing `LaunchSpec::is_acceptor`/`port_file` per ADJACENT PAIR in the chain, rather
than per component, is the natural reading of the field's existing meaning, but was not worked out
in enough detail here to commit to as a fix: Conformance Unit 2 needs no more than this routing to
make its own two cases match, and this is recorded instead of built so the next person extending
`connect()` itself does not have to re-derive this from scratch.

**Not pursued further this session.** `docs/user/multiple-executables.md` now points here from its
own `E-CONNECT-SPEC` row.

**Follow-up (same track, later in Unit 2): implemented, build-verified, not yet host-run.**
`examples/brusselator/participants/{brusselator_graph_tool,u_main.c,v_main.f90,coupler_main.c,
brusselator_check}` now exist. `brusselator_graph_tool --launch accept` declares U's own
`ComponentDecl::ranks` to span its FULL local executable world (its own physics ranks plus any
bundled coupler ranks), and the coupler's own `ranks` as the trailing, legally overlapping subset of
that range (`fuse::RankSet`'s own documented "overlap is legal") — `validate_launch_layout()` skips
`InProcess` components (the coupler) entirely when checking that `Connect`/`Mpmd` components occupy
consecutive blocks, so U's own declared range has to already include every rank the merge will
actually place before V's for the two checks to agree; confirmed (`validate()` and
`validate_launch_layout()` both return OK) against the generated graph for both `--launch mpmd` and
`--launch accept`, `--ranks-u 1 --ranks-v 1 --ranks-coupler 1`. `u_main.c`'s own `--ranks-u` tells it,
per rank of its own local world, whether to act as a real U-physics rank (register fields, drive
`u_loop_step`) or as a bundled coupler placeholder rank (register nothing, skip the step loop
entirely) — every rank still calls `fuse_setup()`/`fuse_initialize()` identically, since
`MPI_Comm_accept`/`MPI_Intercomm_merge` are collective over this executable's WHOLE local
`MPI_COMM_WORLD`, not just its "real" U ranks.

**Open mechanical question (Q4, not yet answered by the lead, beyond Q1–Q3): does a coupler
placeholder rank need to call `fuse_begin_step`/`fuse_end_step` at all?** This session's own
`u_main.c` has a bundled coupler placeholder rank skip the step loop ENTIRELY (no field registered,
no `begin_step`/`end_step` call for "U" from that rank) — reasoned by analogy with
`examples/layout_driver/main.cpp` (a true SPMD multi-component driver, where every rank calls
`Runner::run_period()` identically regardless of which component(s) it hosts, and a `Coupler`
component needs no `ComponentSet` entry of its own at all), but NOT verified against the Runner's own
internal remap collectives, which this sandbox cannot exercise (MPI cannot start here at all, C1
above). If a coupler placeholder rank's own step-loop participation turns out to matter, the fix is
local to `u_main.c`'s/`coupler_main.c`'s own loop structure, not to `connect()` or the graph model —
flagged here for the first host run to settle.

**Host rows registered, not run.** `test/parallel/brusselator_participants/run_participants.sh` (ctest
rows `verification.brusselator.participants.{mpmd,accept}.{3,5}`, `examples/CMakeLists.txt`) drives:

```
mpmd (3):    mpiexec -n 1 brusselator_u_main --graph g.yaml --log u.log --periods 2 \
                   : -n 1 brusselator_v_main --graph g.yaml --log v.log --periods 2 \
                   : -n 1 brusselator_coupler_main --graph g.yaml
mpmd (5):    ranks 2/2/1 in the same three-segment shape
accept (3):  mpiexec -n 2 brusselator_u_main --graph g.yaml --log u.log --ranks-u 1 --periods 2   (acceptor, backgrounded)
             mpiexec -n 1 brusselator_v_main --graph g.yaml --log v.log --periods 2               (connector)
accept (5):  U's executable at 3 ranks (--ranks-u 2), V's at 2
```
followed in every row by `brusselator_check --u-log u.log --v-log v.log`. All four executables and
`brusselator_graph_tool`/`brusselator_check` build cleanly at tier 1 (confirmed this session, both
trees' full `cmake --build`; the tier-0 harness's own `u_loop.c`/`v_loop.f90` sources additionally
BUILD cleanly into `fuse_verification_brusselator_{participants,check}_test` under
`FUSE_ENABLE_MPI=ON` too, but were only ever EXECUTED in the tier-0, FUSE_ENABLE_MPI=OFF tree, where
both pass — no binary built in the MPI-enabled tree was run, per this sandbox's own standing
constraint, C1 above).

**What remains for this entry (S6 phase 3, Unit 4, consolidated 2026-10-07):**

1. **Design work, sketched, not implemented, not scheduled:** the N-way
   generalisation of `connect_pairwise()` above — not a ctest row or a host
   step, so not repeated as one; Conformance Unit 2 needs no more than
   today's routing and is not blocked on it.
2. **Q4 (does a coupler placeholder rank need to call `fuse_begin_step`/
   `fuse_end_step` at all?) is settled by the SAME host run D19's own
   "what remains" list tracks — not a separate row, and not re-listed
   here.** The four ctest rows that answer it
   (`verification.brusselator.participants.{mpmd,accept}.{3,5}`) are D19's
   own to own; see that entry.

### ~~D19. The coupling-window path (`src/runtime/iteration.cpp`) has never run across more than one process~~ — CLOSED 2026-10-09: `parallel.window_runner_mpi.{2,3}` pass on the host (2026-10-08) and the three-executable Brusselator runs through it (D25)

**Found this session**, reviewing Conformance Unit 2's own host rows before they can be run: every
coupling-window test anywhere in this suite is tier 0 (one process). The window path has NO
analogue of the ordinary-tick path's own per-process hosting check (`runner.cpp`'s `hosted[]`,
`~1902`, `3aae2ed`; `grep -n 'this_process\|hosted\|ranks.contains' src/runtime/iteration.cpp`
returns nothing), and this is not merely a missing optimisation — on more than one process it is a
certain hang, for (at least) four separate reasons:

1. **No hosting check at all.** `advance_window()` would call EVERY window member's `advance()` on
   EVERY rank — the redundant physics the user rejected when windows were designed — and, through
   the ABI, would yield `ExternalSuspend` for a component on ranks where no process will ever call
   `begin_step()` for it.
2. **Priming reads the wrong address space.** `resolve_regions()`'s cold-start priming reads
   `registry.get(provider_exported_field)` and the provider's `v0` directly — a provider's field
   exists only in the provider's own address space; a rank that does not host the provider has
   nothing to read.
3. **The convergence decision is not agreed.** A window region's stopping decision is computed from
   LOCAL block norms; with members on different ranks, each rank sees only its own blocks, so
   different ranks could reach different stop/continue verdicts — the exact class of defect this
   project has fixed twice already (`E-RT-EPOCH-PEER`, region residual reductions): a collective
   path returning on a locally-computed condition.
4. **`coupler_main.c`'s and `u_main.c`'s own placeholder ranks never run the Runner at all** — they
   call `initialize()` then `destroy()`, with no `run()`/`begin_step`/`end_step` in between. The
   period's own Transports to and from the coupler's ranks, and the Remap routed through them, are
   executed BY THE RUNNER; a rank that never runs the Runner never executes them, so U's and V's own
   ranks would block forever in the first Transport that needs the coupler's participation. Unit
   2's own four host rows (`verification.brusselator.participants.{mpmd,accept}.{3,5}`) would hang
   on first contact with real MPI, not merely fail a numeric check.

**Scope: Unit 2b**, in five ordered steps, each with its own tier-0 test (the 20 existing window
tests plus the Brusselator as the bit-for-bit oracle for the single-process case):

- **(a) Hosting — DONE (`0eb576a`).** `this_process` (the Runner already carries it) threaded into
  `WindowMemberConfig`; a non-hosted member takes no substeps, no `advance()`/save/restore calls, no
  `ExternalSuspend` — its own export arc on that rank is whatever Transport delivers.
- **(b) Priming through the transfer — DONE (`2df7052`).** The import prime at `resolve_regions()`
  now comes through the field transfer (the channel's own compiled Transport/Remap/Epoch/Sink chain,
  re-run once from the provider's own export arc), never from `registry.get(provider_exported_field)`
  directly.
  **Correction to this entry's own original model, measured while building (b):** a window
  channel's own `Interpolate` node NEVER carries a real `TemporalBasis` — confirmed by dumping the
  actual compiled instruction stream for both a plain and a genuinely multirate (7:13) window graph:
  neither ever emits an `OpKind::Sample`, and `resolve_spacetime()` (`runner.cpp`) only builds a
  `SpacetimeInfo` for a node with `OpAttrs::basis != TemporalBasis::None`, which
  `src/ir/window_lowering.cpp` never sets for a window channel (it compiles the `lag(k)` no-op
  shape). So every window-channel `Interpolate` always takes `execute()`'s own `r.spacetime < 0`
  plain-copy branch; `advance_window()` hands over the complete K-wide waveform in one buffer every
  sweep, and there is no incremental, Sample-fed history for a window channel to depend on at all.
  **A reader building on this path does not need to seed `SpacetimeInfo::history` for a window
  channel — there is nothing there to seed.** (This entry originally assumed otherwise; measured and
  corrected 2026-10-03.)
- **(c) Agreed convergence — DONE.** The per-block residual/reference norms of a window region
  reduced across the region's participants (the union of its members' processor sets) before the
  stop/continue decision, in a fixed, deterministic order: `decide_window_convergence()`
  (`include/fuse/runtime/window_convergence.hpp`), a new pure function in the same no-MPI,
  tier-0-testable style as `rank_agreement.hpp`/`epoch_schedule.hpp`, fed by a new point-to-point
  all-to-all exchange (`Runner::Impl::gather_window_block_samples()`, new tag `kWindowConvergence`
  — never a true `MPI_Allgather`, since `R.participants` is a subset of `base`'s own ranks, matching
  `reduce_region_residuals()`'s own convention). A block whose value is not hosted here contributes
  no sample rather than an uninitialised one — `WindowMemberConfig::hosted` gates ownership only
  when `R.participants.size() > 1` (a real multi-rank scenario where a different rank genuinely
  supplies the value); with no second rank, a block stays owned, since disowning it there would make
  it unmeasurable by anyone. Six pure-function tests in `test/runtime/window_convergence_test.cpp`,
  including a direct instantiation of "a rank deciding from its local pairs only" (the lead's own
  suggested mutation). The cross-rank agreement itself — as opposed to the wiring that reaches
  `decide_window_convergence()` at all, which IS tier-0-killed, broadly and immediately — has no
  tier-0 oracle (every sandbox fixture has exactly one effective participant); its real proof is
  `parallel.window_runner_mpi.{2,3}` (step (e)), a host row.
  **CORRECTION, measured against the host run of `parallel.window_runner_mpi.{2,3}`
  (2026-10-08) — FIXED:** that host run FAILED (not merely unconfirmed): window 0, rank 0,
  `got.iterations_per_window[0] == 1` against a reference of 37 (windows 1-4: reference 35, 25, 15, 3
  vs `got` 1 every time), `got.x_b_per_window[0] == 0.2`, B's own unmodified initial value. Root
  cause: `window_export_meta` (`src/runtime/iteration.cpp`) stored the EXPORTING member's own
  `WindowMemberConfig::hosted` where `RegionBlock::window_hosted` (consulted by `run_region()`'s own
  `gate_by_hosted` above) needs whether the CONSUMER is hosted here — a window block's residual is
  measured through the compiled delivery chain (Transport/Epoch/Remap/Constrain/Diagnose), which runs
  on the consumer's own processor set, not the exporter's. On a disjoint-ranks channel this inverted
  ownership in both directions at once: the exporting rank wrongly claimed a block it never writes,
  and the consuming rank — the only rank that does — wrongly disclaimed it, so both sides summed a
  stale 0/0 residual/reference pair that reads as trivially converged. Fixed by a new
  `WindowMemberConfig::export_consumer_hosted` field (resolved from the channel's TARGET component's
  own processor set, the identical formula `hosted` itself uses), which `window_export_meta` now
  stores instead. Pinned by `test/runtime/window_hosting_test.cpp` (white-box over the public
  `Runner` API plus one test-only accessor, `Runner::test_only_window_block_hosted()`, since
  `RegionBlock::window_hosted`'s own gate is only ever reached with a real `base_communicator`, which
  a no-MPI build refuses outright) and a new pure-function case in `window_convergence_test.cpp`;
  mutation `window-runner-export-ownership-uses-the-exporter`
  (`build-evidence/mutations/window-runner.toml`) restores the exporter-side flag and is killed by
  exactly the two rank-specific `WindowHosting` tests. **Then taken one level further:** that
  field is per MEMBER, resolved against the first channel whose source is the member's export, so a
  member feeding two consumers on different processor sets would hand both blocks one answer. Each
  block now resolves its own ownership from its own channel's target (`resolve_regions()`, the
  `window_export_capacity > 0` branch), which is where the block's channel is already known;
  `WindowHosting.FanOutBlocksFollowTheirOwnConsumerNotTheFirstChannel` builds exactly that graph (one
  export, consumers on ranks 1 and 2) and the mutation
  `window-block-ownership-resolved-per-member-not-per-channel` kills it. Writing that fixture also
  measured a neighbouring refusal worth recording: a convergence measure naming the fanned-out SOURCE
  field is refused with `E-ITER-MEASURE` (it matches two residuals, and exactly one is required), so
  each consumer's own field is what such a region names. **Also found and fixed in the same host run:**
  `window_runner_mpi_test.cpp` compared a member's own `x()` on EVERY rank, but since `0eb576a` a
  member advances only on its own processor set — a rank that does not host a member never touches
  its real state, so the test was comparing a hosting rank's real trajectory against a non-hosting
  rank's untouched initial value. Each member's own state is now compared only on the ranks
  `RankSet::contains()` says host it; the agreed sweep count stays compared on every rank.
- **(d) Ranks without a component — DONE.** `fuse_run()` legal on a process hosting no
  externally-driven component: `Runner::hosts_component(ComponentId) const -> bool`, a new public
  method generalising `WindowMemberConfig::hosted`'s own rule, replaces `impl_run()`'s blanket
  `c.external_count > 0` graph-level refusal (`src/abi/participant.cpp`) with a per-component loop —
  explicitly skipping `ComponentKind::Coupler` entries, which never get `in_process` set at all and
  so would otherwise be wrongly refused on a rank hosting only a coupler. `coupler_main.c`'s and
  `u_main.c`'s own placeholder ranks now call `fuse_run()` for `--periods` periods instead of doing
  nothing between `initialize()`/`destroy()`. Tier-0-tested via a new test-only ABI seam,
  `fuse::abi_test::force_this_process()` (mirrors `inject_component()`'s own precedent: no entry in
  the frozen public ABI lets a C caller name a rank without a real communicator) —
  `AbiParticipant.RunAcceptedOnAProcessHostingOnlyTheCoupler` and
  `AbiParticipant.RunStillRefusedOnAProcessHostingAnExternalComponent`, both passing; the
  coupler-exclusion fix specifically validated by the former. `docs/user/c-abi.md` updated with a new
  "`fuse_run()` and ranks without a component of their own" section.
- **(e) `parallel.window_runner_mpi.{2,3}` — DONE (sandbox build only).** The 7:13 LCM-window fixture
  (`window_runner_multirate_test.cpp`'s own shape: A interval 13, B interval 7, window 91, so neither
  member ever straddles) as a REAL multi-process `Runner` run — A and B on genuinely disjoint ranks
  (0 and 1), a third, uncoupled component C added at 3 ranks (rank 2, no channels, proving its mere
  presence does not perturb the window) — compared, `EXPECT_EQ`, against the single-process `Runner`
  on every per-window final value AND every per-window iteration count, not merely completion.
  `test/parallel/window_runner_mpi_test.cpp`, registered via `fuse_add_mpi_test(parallel
  window_runner_mpi LIBS fuse::runtime RANKS 2 3)`. Build-verified only: `b1mpi` (tier 1,
  `FUSE_ENABLE_MPI=ON`) builds and links `fuse_parallel_window_runner_mpi_test` clean, and `ctest -N`
  there confirms the two rows exist (`parallel.window_runner_mpi.2`, `parallel.window_runner_mpi.3`);
  MPI cannot start in this sandbox at all (C1), so neither row has actually been RUN here. This is
  the host row that settles both (c)'s cross-rank agreement (no tier-0 oracle, as noted above) and
  (b)'s narrower registry-vs-chain comparison (`window-prime-length`/`window-runner` mutation notes,
  `build-evidence/mutations/`), neither of which a single sandboxed process can distinguish from its
  alternative.

**Fixed in the sandbox, awaiting the host.** Steps (a)-(e) above are all implemented and tier-0/
build verified (163/163 tier-0 tests pass; tier 1 — `FUSE_ENABLE_MPI=ON` — builds clean including the
new MPI test and the Brusselator's `coupler_main`/`u_main` placeholder-rank changes); every mutation
tried has a decisive sandbox-side kill except the two comparisons named above, which by their own
nature need a second real rank to even pose the question. None of this has been run under real MPI —
this sandbox cannot start it (C1) — so the fix is unconfirmed until a host executes:
- `verification.brusselator.participants.mpmd.{3,5}` (Unit 2, pre-existing)
- `verification.brusselator.participants.accept.{3,5}` (Unit 2, pre-existing)
- `parallel.window_runner_mpi.2` (step (e), new)
- `parallel.window_runner_mpi.3` (step (e), new)

### D31. Newton over a coupling window (S7.2 WindowNewton, `newton-krylov.md` Sec. 9) was single-rank only; the multi-rank `ParticipantSetReducer` is now built — BUILT 2026-10-12, host MPI run still required to close

`Runner::Impl::run_region_newton_window()`/`evaluate_window_region_residual()` (src/runtime/iteration.cpp)
implement Newton-Krylov over a WINDOW-scoped coupling region, reusing `check_region_supports_newton()`'s
own `R.participants.size() > 1` refusal (`E-NEWTON-UNSUPPORTED-REGION`) unchanged from the ordinary
(non-window) seam — a window-scoped Newton region with more than one cross-rank participant is refused
at setup exactly as it always was, never reaching the new code at all. Every norm and inner product
`fuse::num::solve()`'s own GMRES/line-search/forcing-term machinery computes over the window's own
Newton interface vector is already routed through a `fuse::linalg::Reducer` held by the region
(`RegionRuntime::newton_reducer`, `src/runtime/runner_impl.hpp`) rather than inlined as a local
sum-of-squares — but the reducer `resolve_regions()` (`src/runtime/iteration.cpp`) actually constructs
there is unconditionally `fuse::linalg::make_serial_reducer()`: a real cross-rank `ParticipantSetReducer`
(summing a local norm/inner product across exactly `R.participants`, the SAME rank set `decide_window_
convergence()`'s own existing cross-rank reduction for Picard's sweep loop already uses) is a comment on
`RegionRuntime::newton_reducer`'s own declaration documenting the FUTURE contract, not an implementation —
there is no code anywhere in this track that builds one, and no test exercises more than one rank.

**Why deferred rather than built now:** the task's own lead decision (2026-10-11, Stage B go-ahead)
scoped this track to single-rank only, explicitly asking for the Reducer SEAM (so a later multi-rank
step is a reducer swap, not a re-plumb of every call site) without the reducer's own cross-rank body.
Building `ParticipantSetReducer` honestly needs a real multi-rank host run to prove against — this
sandbox cannot start MPI (C1, same constraint D5/D8/D14/D23 already name) — so writing it now would be
exactly the same "implemented, never run under real MPI" category those issues already carry, for a
genuinely new reduction rather than a variant of one already host-verified elsewhere.

**What a host run must verify before this is closed:** a `ParticipantSetReducer` (new, `src/linalg/
comm_mpi.cpp` + `include/fuse/linalg/comm.hpp`, the files `newton-krylov.md` Sec. 9.4/9.7 already name
for this) that sums a local scalar across exactly one iteration region's own `R.participants` ranks (not
the full `base_communicator`, the same "this region's own rank subset, not the whole run" scoping
`decide_window_convergence()`'s existing reduction already gets right); `resolve_regions()` building one
for a window-scoped Newton region instead of `make_serial_reducer()` unconditionally; and a new host
test, `parallel.window_newton_mpi.{2,3}` (mirroring `parallel.window_runner_mpi.{2,3}`'s own shape, step
(e) above, on a window region with `AccelKind::Newton` instead of Picard) comparing every window
instance's final Newton iterate AND iteration/GMRES count against the single-process `Runner` on the
identical graph — the SAME cross-rank-agreement property `window_runner_mpi_test.cpp` already proves for
Picard, now for Newton's own reductions specifically.

**Built 2026-10-12 (Newton track), BUILD-VERIFIED ONLY — the sandbox this track built in cannot
complete a bare `MPI_Init()` (the same constraint this entry's own deferral named), so nothing below has
been RUN anywhere; the host must verify it.** `fuse::linalg::ParticipantSetReducer`
(`src/linalg/comm_mpi.cpp`, `include/fuse/linalg/comm.hpp`) is real: an `MPI_Comm_split` over exactly a
region's own `RegionRuntime::participants`, collective over the Runner's base communicator and called in
the same ascending-region-index order on every rank (mirroring `attach_comm_graphs()`'s own "every rank
calls every split" discipline); a rank outside the participant set gets `MPI_UNDEFINED` and no `Reducer`
object at all. The split is held, not point-to-point-addressed like `reduce_region_residuals()`/
`gather_window_block_samples()` — the task's own explicit choice, reasoned through in the factory's own
header comment — with its lifetime handled the way `MoabMeshBackend`'s own `OwnedComm` already handles an
identical one (freed in the Reducer's own destructor, guarded by `MPI_Finalized()`, never copyable).
`check_region_supports_newton()`'s blanket "more than one participant is refused" rule is gone, and
nothing narrower replaces it: `RegionRuntime::participants` is populated with more than the empty set
only when `RunnerConfig::base_communicator` has a value in the first place, so "more than one
participant" already implies a real communicator exists to build the Reducer from. The earlier
`E-RT-REGION-MPI` refusal (the one that blocks a dynamic accelerator on a multi-rank region) now also
exempts `AccelKind::Newton` alongside `None`/`Constant`. `RegionBlock::newton_offset`'s own window-Newton
vector-layout computation now excludes a block this rank does not host (`!b.window_hosted`) from the
SAME mechanism the GaussSeidel exclusion already used, making `RegionRuntime::newton_n` a per-rank LOCAL
total with zero ghosts (every cross-rank dependency the window's own body needs is already carried by its
compiled Transport instructions, proven correct for Picard's own multi-rank case). `run_region_newton()`/
`run_region_newton_window()` gained the identical "not a participant, nothing to do" early return
`reduce_region_residuals()` already has (`Runner::Impl::region_is_local_participant()`, new, kept in
`runner.cpp` so `iteration.cpp` still names no MPI type); `run_region_newton()`'s own ordinary (non-
window) seam now also reads `RegionRuntime::newton_reducer` instead of constructing a fresh serial reducer
every call, closing the one place that convention from Sec. 9.4 had not yet reached even in the
single-rank era. Full account: `docs/dev/newton-krylov.md` §9.10.

**Measured by hand (tier 0, no MPI, this sandbox):** mutating the `!b.window_hosted` exclusion to never
fire (`build-evidence/mutations/newton-parallel.toml`, row `window-newton-offset-not-gated-by-hosted`)
kills exactly the two new `window_newton_hosting_test.cpp` cases that name a genuine cross-rank exclusion
(2 FAILED of 3), leaves the no-`this_process`/no-communicator case passing (window_hosted already
defaults true there), and restores clean. `parallel.window_runner_mpi.{2,3}`, `verification.
brusselator_runner` and every other pre-existing tier-0 row this track touched were re-run after each
change and are unaffected (`newton_region_test`: 17/17; `window_fixed_point_residual_test`: 8/8;
`window_hosting_test`: 4/4; `brusselator_runner_test`'s own `NewtonConvergesInAPinnedIterationCount...`:
1/1).

**What a host run must still verify before this closes:** `parallel.window_newton_mpi.{2,3}`
(`test/parallel/window_newton_mpi_test.cpp`, built, never run) and `parallel.participant_reducer_mpi.
{2,3}` (`test/parallel/participant_reducer_mpi_test.cpp`, built, never run) — the real `MPI_Comm_split`/
`MPI_Allreduce`/`MPI_Allgather` correctness this sandbox cannot exercise at all. Every one of
`ParticipantSetReducer`'s own `sum`/`max`/`allgather` methods; whether a 2- or 3-rank window-Newton solve
over the Brusselator-shaped and the synthetic LinearOde-shaped fixtures actually reaches the SAME
outer-Newton and GMRES iteration counts the single-process `Runner` does, and whether the final member
states fall inside the Higham round-off bound the test derives and prints rather than asserting bitwise.
D32's own PETSc substrate lift (below) shares this same host-pending status under
`petsc.window_newton_mpi.{2,3}`.

### D32. The PETSc solver plugin is the second substrate behind the S7 seam — BUILT AND MEASURED 2026-10-11; the communicator remains

S7 was built on the library's own linear algebra first, by the user's decision of 2026-10-10: a
native Jacobian-free Newton-Krylov coupler (`fuse::num::solve()`, `docs/dev/newton-krylov.md`)
replicating the mechanisms that make PETSc SNES's `NEWTONLS` robust — the Walker–Pernice
finite-difference step, Eisenstat–Walker forcing, the cubic-backtracking, critical-point and damped
line searches, named convergence and divergence tests, a lagged block-Jacobi preconditioner — with
no PETSc dependency. PETSc remains the intended second substrate for problems the native solver
should not carry (very large or stiff coupled systems, preconditioners the library does not have),
and it plugs in behind the SAME seam: a `fuse::LinearizationToken` whose `apply_jacobian` is a
`MatShell`/`MatMFFD` and whose `apply_preconditioner` is a `PC`, or a `SNES` driving
`Runner::region_fixed_point_residual()`'s `F` directly. Nothing in the seam is PETSc-shaped and
nothing in the Newton solver knows which substrate a token comes from.

**State today.** `FUSE_ENABLE_PETSC` is a configure option (`cmake/FuseDependencies.cmake:121`,
tier 3, requires `FUSE_ENABLE_MOAB`) that locates PETSc through `pkg-config` and defines
`FUSE_HAVE_PETSC` in `fuse/base/config.hpp`, and nothing more: no source file under `src/` or
`include/` tests that macro, and no target links the imported library. Three PETSc installations are on the
development host, none of them matching this build: `/opt/petsc/chi-tech` (3.19, 64-bit indices,
HYPRE), `/opt/petsc/download-moab` (3.25, debug, MOAB, built against `/opt/mpich`), and
`/opt/petsc/debug-bare` (3.17, debug). **Decided 2026-10-11 by the owner: `/opt/petsc/download-moab`,
for PETSc AND MOAB** (the `all` preset's `MOAB_DIR` moved there from `/opt/moab/cmake-all`); the other
installs are old. Its MOAB has no TempestRemap, so the high-order spherical remap method refuses at run
time; measured that the registered row set is unchanged (299 rows either way, the only name differences
the OpenMP thread-count suffix). No Kokkos and 32-bit indices in that PETSc are accepted for now; a
Kokkos-enabled PETSc is a later rebuild on the owner's side, not a plugin concern.

**What a plugin track owes when it is opened.** The derivative gate (`linalg.derivative_gate`,
`docs/dev/derivative-gate.md`) on the PETSc-backed token: the adjoint identity on
`MatMult`/`MatMultTranspose`, the Taylor-remainder order on `MatMFFD`'s action, and agreement with
the native MFFD action to a measured bound. dn_heat and the Brusselator by the PETSc path against
the native results recorded in `docs/user/dn-heat.md` and `docs/user/brusselator.md` — same
graphs, same tolerances, iteration counts and final means side by side. A refusal when
`FUSE_ENABLE_PETSC` is off and a graph selects the PETSc substrate, with a test that makes it fire.

**Built 2026-10-11 (PetscToken track, S7.2b) and measured on the host the same day — all four
`petsc.*` rows pass in the MPI tree and in a tier-0 PETSc-only tree; `docs/dev/newton-krylov.md`
§10.3 carries the numbers. The sandbox the track built in cannot complete a bare `MPI_Init()`, so
the measurements are the host's.** `src/petsc/{init,token,solve}.cpp` and `include/fuse/petsc/{token,solve}.hpp` implement
exactly the plugin this entry asked for: `fuse::petsc::MffdToken` (a `fuse::LinearizationToken`
backed by `MATMFFD`/`MATMFFD_WP`, `apply_jacobian_transpose` refusing `E-PETSC-NO-TRANSPOSE`,
`apply_preconditioner` through a `PC` — `PCNONE` by default, `PCSHELL` composing the SAME
`fuse::num::BlockJacobiPreconditioner` the native solver uses for its own block-Jacobi, reused
rather than reimplemented), `fuse::petsc::make_assembled_token` (a real `MATAIJ`, for the adjoint
test the MFFD token itself has no transpose to run), and `fuse::petsc::solve()` (`SNESNEWTONLS`,
`MatCreateSNESMF`/`MATMFFD_WP`, `KSPGMRES` right-preconditioned, `SNESKSPSetUseEW` with the SAME
six Eisenstat-Walker constants `NewtonKrylovOptions` already uses, `SNESLINESEARCHBT`/`CP`/`NONE`
selectable, `SNESConvergedReason` mapped onto the native `NewtonStatus`). `tools/check-layering.py`
gained a `petsc` layer (topmost, `ALLOWED_PUBLIC_THIRD_PARTY` empty — no PETSc type anywhere in a
public header) and reports 0 violations; `cmake/FuseDependencies.cmake`'s old
`FUSE_ENABLE_PETSC requires FUSE_ENABLE_MOAB` gate is gone (the solver plugin needs neither MOAB
nor a mesh — that bundling was for a PETSc+MOAB physics kit this track is not). Clean tier-0
builds, BOTH with and without `FUSE_ENABLE_PETSC`, 0 compiler errors; `FUSE_ENABLE_PETSC=OFF`
registers exactly the pre-existing 187 tier-0 rows (100% pass, unchanged); `FUSE_ENABLE_PETSC=ON`
registers 191 (187 + `petsc.{init,token,derivative_gate,snes}`), and a tier-1 compile
(`-DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_FORTRAN=ON -DFUSE_ENABLE_PETSC=ON`) of every touched target
succeeds with 0 errors.

**Where the measurements came from.** The sandbox the track built in cannot complete a bare
`MPI_Init()` (MPICH's debugger-attach mutex assertion in `dbginit.c`, independent of PETSc — a
single-process constraint, more severe than C1's launcher one), so the track compiled and
registered everything and the lead ran it on the development host in two trees, `build-all`
(MPI on) and a tier-0 PETSc-only tree. Three defects the sandbox could not show were found and
fixed there: `ensure_initialized()` set `PETSC_COMM_WORLD` before `PetscInitialize` even with MPI
down, which made PETSc skip `MPI_Init` and fail with `PETSC_ERR_SUP` in every tier-0 process;
both tokens' `PC` had no operator attached, so `PCSetUp` refused "Missing A matrix" on every
`apply_preconditioner`; and the adjoint gate's control case used `x == y`, for which the identity
holds for any operator. A fourth was found by the corrected init test itself: `MPI_Initialized`
stays true after `MPI_Finalize`, so a second initialisation after a release walked into a
finalised MPICH; it now refuses `E-PETSC-MPI-FINALIZED`. All four `petsc.toml` rows are
hand-proved on the host's tier-0 tree (each kills exactly the case that guards it; the dropped
finalize check aborts the binary after its first case).

**What remains before this entry can close** (plan S7.2b phase 2): per-region substrate selection
(`NewtonConfig` substrate `native`/`petsc` through YAML and the descriptor, `E-NEWTON-PETSC-UNAVAILABLE`
in a build without PETSc with a test that fires it), dn_heat and the Brusselator by the PETSc path
recorded beside the native results, and the `forcing.eta_min` gap (`docs/dev/newton-krylov.md` §10.4:
SNES has no forcing floor, and the Brusselator window needed one on the native substrate).

**Built 2026-10-11 (PetscSelect track, plan S7.2b phase 2) -- the per-region selector itself, NOT
the `eta_min` gap, which stays open (§10.4/§11 both name it unchanged).** `NewtonConfig::substrate`
(`fuse::NewtonSubstrate`, `Native`/`Petsc`, fuse/model/iteration.hpp) selects WHICH registered
implementation a region's Newton solve dispatches to, through a small registry in `fuse::num`
(`register_newton_substrate()`/`newton_substrate()`, fuse/num/newton_krylov.hpp -- `fuse::num` sits
BELOW `fuse::runtime` in `tools/check-layering.py`'s own layer order, and `fuse::petsc` sits ABOVE
`fuse::runtime`, so the registry is the only place the Runner can dispatch to a substrate it is
layering-forbidden from naming directly). `Native` registers itself deterministically on first use
(a function-local static, never a static initialiser that might not link); `fuse::petsc::install()`
(new, fuse/petsc/solve.hpp) registers `Petsc`, never called by the library itself. `src/runtime/
iteration.cpp` checks the registry TWICE per Newton-selected region: eagerly in `resolve_regions()`
(so a graph naming an unavailable substrate refuses `E-NEWTON-PETSC-UNAVAILABLE` at
`Runner::initialize()`, before the first period -- `NewtonRegion.ARegionSelectingAnUnregisteredSubstrateIs
RefusedAtSetup`, test/runtime/newton_region_test.cpp, fires this TIER-0, WITHOUT PETSc), and again,
defensively, at the actual dispatch inside `run_region_newton()`/`run_region_newton_window()`. The
descriptor bumped to v13 (`test/golden/waist-v13.bin`, one more byte appended at the end of the v12
NewtonConfig block, following v12's own append-only placement exactly); YAML gained `newton_config.
substrate: native|petsc`. The communicator decision (docs/dev/newton-krylov.md §11.3): the registered
PETSc substrate always solves on `MPI_COMM_SELF`, never the Runner's own `base_communicator`, because
every Newton-selected region on EITHER substrate is already single-rank-only by `check_region_
supports_newton()` (D31's own restriction) -- nothing to reduce across ranks inside the solve itself
regardless of substrate; `SnesOptions::comm_f` stays available for a future multi-rank lift to revisit.

`examples/dn_heat/main.cpp --newton-petsc` and `examples/brusselator/main.cpp --substrate petsc` run
the IDENTICAL graphs the native rows do, by data (`NewtonConfig::substrate`) alone -- the default
invocation of either binary is byte-identical whether or not `FUSE_ENABLE_PETSC` was configured,
since both PETSc code paths are compiled in only under `FUSE_HAVE_PETSC` and run only when explicitly
asked for. Both compare the PETSc result against the native one to a bound DERIVED from the two
solvers' shared `rtol` and the problem's own structure (dn_heat's affine Jacobian; the Brusselator's
own `kTol`) plus the §10.3-measured 3.9e-8 relative inter-solver difference -- never a number chosen
to make the check pass; the exact derivation is in each file's own comment at the check. Registered
as `example.dn_heat.newton_petsc` and `petsc.brusselator_newton`, both gated by
`FUSE_ENABLE_PETSC`.

**Verified, tier-0 WITHOUT PETSc:** clean build, 0 errors, 187/187 tests pass (unaffected count and
pass rate -- this track added no new tier-0-unconditional ctest row), `/tmp/petscselect/b0`.
**Verified, tier-0 WITH PETSc:** clean build, 0 errors, 193 tests registered (187 + `petsc.{init,
token,derivative_gate,snes,brusselator_newton}` + `example.dn_heat.newton_petsc`); 187/193 PASS,
exactly the 6 PETSc-dependent rows FAIL by the same `MPI_Init()` constraint §10's own measurements
were taken around (SEGFAULT, not a logic failure) -- every pre-existing, PETSc-independent row,
INCLUDING `verification.brusselator_runner` itself, is unaffected, `/tmp/petscselect/bpetsc`. (An
earlier draft of the Brusselator PETSc-substrate case lived inside `test/verification/
brusselator_runner_test.cpp` itself; measured directly that its own crash took the WHOLE shared
binary down with it, failing every other case in that file too -- moved to its own dedicated binary,
`test/petsc/brusselator_newton_test.cpp`/`petsc.brusselator_newton`, the same isolation convention
every other `petsc.*` test already follows, and reconfirmed clean after the move.)
**Measured on the host 2026-10-11 and closed as far as phase 2 goes:** the dn_heat and Brusselator
PETSc-substrate tables (docs/user/dn-heat.md, brusselator.md) are filled from `build-all`; both
examples agree with the native path (dn_heat to 2.7e-10 K against Picard, the Brusselator to every
printed digit of the final means over 50 windows). The `forcing.eta_min` gap is **closed**: the
Brusselator did stall on PETSc without a floor, and `fuse::petsc::solve()` now applies
`NewtonConfig::eta_min` through a KSP convergence test (`docs/dev/newton-krylov.md` §10.4(1)), proved
by the `petsc-forcing-floor-not-applied` mutation. **Still open under D32:** the PETSc substrate
solves on `MPI_COMM_SELF` (correct while D31 restricts Newton regions to one participant; the
communicator plumbing is D31's). The 20× margin the agreement bounds once carried is gone: they are
built from each substrate's own terminal residuals (dn_heat: an exact bound, ‖J⁻¹‖∞ = 4/3 times the
sum over its 20 periods; the Brusselator: a calibrated reference error times one plus the residual
ratio), and both held on the owner's machine on 2026-10-12 (dn_heat: observed 3.572609e-10 K against
5.888415e-07 K, ratio 6.067e-04; the Brusselator: per element against 5.390239e-15, with the observed
maximum not yet printed — the test now prints it).

### D33. The omp legs and the `omp_identical` rows were registered with one thread whenever configure ran under `OMP_NUM_THREADS=1` — CLOSED 2026-10-12

`test/CMakeLists.txt` took the host core count from CMake's `ProcessorCount()`, and registered each
`*.omp<N>` leg and each `example.*.omp_identical` row with it. `ProcessorCount()` asks `sysctl -n
hw.ncpu` first and then `nproc`, and GNU `nproc` returns the value of `OMP_NUM_THREADS` when it is
set. Where `sysctl` answers (macOS), the variable has no effect. Where it does not, the count follows
the variable. Measured 2026-10-12 on this development sandbox, where `sysctl` fails with "Operation
not permitted" and the Homebrew `nproc` answers: configuring the committed head under
`OMP_NUM_THREADS=1` registered 13 omp legs and gave all five `omp_identical` rows `THREADS_HI=1`; with
the variable unset the same tree registers 26 legs and `THREADS_HI=12`. The five rows are the slowest
in the suite, and with `THREADS_HI=1` each compares one thread with one thread, passes, and can no
longer detect an order-dependent sum. The build tree on the owner's machine (`build-all`) was not
affected: it reads `THREADS_HI=12` in all five rows and has 26 legs. Any Linux host with GNU `nproc`
would be.

Fixed by `fuse_host_core_count()` in `cmake/FuseHostCores.cmake`, which removes the variable from the
environment for the duration of the call and restores it, and by a ctest row,
`build.host_core_count_ignores_omp_threads`, that runs `tools/check-host-cores.cmake`. The row puts a
failing `sysctl` and a GNU-style `nproc` first on `PATH`, so it reaches the variable-reading branch on
every machine; it passes on the fix and fails on both the removed line and the original bare
`ProcessorCount()`. With the fix, configuring under `OMP_NUM_THREADS=1` and with the variable unset
gives the same 194 rows and `THREADS_HI=12`. Configure now also states when the host has one core,
because the omp legs then compare one thread with one thread whatever the registration does.
Not covered: Windows (the check is skipped; `ProcessorCount()` reads `NUMBER_OF_PROCESSORS` there).

### D34. The FSI beam-side energy identity compared a quantity built with the literal step against a run that took the rounded one — CLOSED 2026-10-12

**Cause (measured).** `build_fsi_graph()` in `test/verification/fsi_energy_diag_test.cpp` does not hand
the coupling graph `c.dt`: it first rounds it through the file's own `rationalize_dt()`,
`Rational{llround(dt × 1e8), 1e8}`. For `c.dt = 3.51042838666104032e-03` that is 351043 / 1e8, so every
production step ran on `3.51043000000000015e-03`, a difference of −1.613339e-09 (relative 4.6e-7). The
beam-side identity then formed its predicted energy change as `c.dt × (wall power)`, with the literal
`c.dt`, against `dE_beam` produced by steps of the rounded one. The mismatch is therefore
`|c.dt − dt_delivered| × |beam_wall_power|`, which reproduces the observed floor to 7–8 significant
figures in both policies (predicted/actual 1.000000037 ON, 1.000000006 OFF). The floor was the same
absolute size on two different modes because that product is, not because anything scaled with the mode.

**Fixed.** The identity uses `dt_delivered = rationalize_dt(c.dt).to_double()`. The mismatch is now
4.065758e-20 (ON) and 6.776264e-21 (OFF), 2.8e-16 and 3.0e-17 of E0.

**Tolerance, derived rather than fitted.** The `eig_quality × |E0|` tolerance, and the widening of its
factor to 1.5 that was rejected after the assertion failed, are gone. The tolerance is
`c · u · (|dE_beam| + |dt·P| + |(1 − dx_beam)·dPE|)` with `u` the machine epsilon and `c` the largest
of three multipliers formed from each term's own bound (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., sec. 3.1: `|fl(Σ xᵢyᵢ) − Σ xᵢyᵢ| ≤ γₙ Σ|xᵢyᵢ|`; the quadratic form's uncancelled
sum, not its small value; and `2κu` for the outputs of the beam's direct solve, κ₂(Keff2) = 1.222628,
measured in the test). `c` is printed: 1.140213e4 (ON), 2.344491e4 (OFF). The tolerance is 1.0305e-16 and
1.9776e-16, so it is 2534.7 and 29183.6 times the observed mismatch. It is not tighter than the observed
mismatch in either policy, so no term was missing. Limits: the `2κu` term is the first-order bound for a
backward-stable solve and ignores the growth factor, and the nnz and κ come from a replica of the beam's
stiffness matrix built in the test, whose agreement with the solver's private matrix was checked in
scratch work that is not shipped.

**Verified 2026-10-12 from a clean export.** Debug with the default policy: 179 of 179 quick rows and
16 of 16 slow rows. Release ON and OFF: the 5 touched rows, including `verification.fsi_energy_diag`,
which failed under OFF before this fix (ratio 1.029). The mutation that restores the literal `c.dt`
fails the test with the 4.6e-7 relative step error exceeding the tolerance by factors of 1.058e4 (ON) and
5.319e3 (OFF); the `dx_beam`-coefficient
mutation is killed in both policies. Both restore byte-identically.

**How it was found, and two wrong turns.** This entry was first written as open, with the floor unexplained and
the test passing under ON only because its tolerance followed the eigenvector accuracy. Two hypotheses given during the search were wrong. The lead guessed the fluid
step's conditioning at about 1e7; it is 3.53e5, and its measured forward error is 2.2e-15. The track's
penultimate report then attributed the rounding to the core scheduler's tick resolution
(`kMaxPeriodTicks`); that was wrong, since the scheduler delivered exactly the rational the test gave
it, and the rounding is the test's own `rationalize_dt()`. Between those, a hidden-state leak across
rejected sweeps and the cached factorization were tested directly and excluded (replay with the cache
defeated reproduced production with a difference of 0.0; every member of the beam solver and component
round-tripped bitwise through all 16 restores).

**Still open, same class.**

- The fluid-side budget calls `measure_full_energy_identity_general(fc, c.dt, ...)` with the literal
  `c.dt` against a state that advanced on `dt_delivered`. The audit found it; it is not fixed, and the
  tolerance it is compared against has not been examined for whether it silently absorbs the 4.6e-7.
- The interface adjointness check (beam receives against fluid delivers) has a separate mismatch of
  about 4.75e-6 in both policies, attributed so far to a non-adjoint remap with `nx_beam = 6` against
  `nx_fluid = 5`. Not investigated.
- `examples/fsi_channel/main.cpp` rounds `dt` at 1e-6 (`rationalize_dt`) and afterwards uses the literal
  value for the step count and the CSV time column only. No dynamics computation sees it, so the effect is
  that the simulated end time and the time axis carry the rounding error: zero for the default
  `--dt 0.01`, about 1.22e-4 for a value that is not on the grid.

### D35. The accept/connect probe's verdict was cached once and never measured again — CLOSED 2026-10-12

`cmake/FuseDependencies.cmake` ran `tools/probe-mpi-dynamic.sh` only when `FUSE_MPI_DYNAMIC_PROBED` was not in
the cache, and cached whatever it returned, including "not tested", which records that nothing was measured.
Reinstalling the MPI therefore changed nothing: on the owner's machine, after the reinstall that replaced
`ch3:nemesis` with `ch4:ofi`, a reconfigure printed `accept/connect launch mode: ON (not tested)` without
probing, while running the probe by hand exited 0. A cached "measured broken" would have been worse:
`FUSE_ENABLE_CONNECT_ACCEPT`, an option cached at the first configure, would have stayed OFF with it, with
nothing left to measure again.

**Fix.** `cmake/FuseMpiDynamic.cmake` holds three functions of plain values. `fuse_mpi_fingerprint()` is the
content hash of the first MPI library `FindMPI` reports, of the launcher and of the compiler wrapper (a hash, so a
reinstall over the same prefix changes it); `fuse_mpi_dynamic_should_probe()` reuses a verdict only if it is a
measurement taken on the MPI configured now; `fuse_mpi_dynamic_option_value()` lets the option follow the probe
unless it was set by hand (a value that differs from what the probe itself set last time is kept; a cache from
before that record existed is read from its verdict). `tools/check-mpi-dynamic-cache.cmake`, registered as
`build.mpi_dynamic_cache_decisions`, runs 8 re-probe rows, 9 option rows and the fingerprint cases with no MPI. Four
mutation rows (`build-evidence/mutations/build-mpi-dynamic.toml`) are killed by it.

**Measured in a tier-1 tree in the sandbox, whose launcher cannot start a job (so every probe reports "could not
test"):** a second configure on the same MPI printed the re-probe line and probed again; a cached
`measured broken` taken on another MPI was re-probed and the option moved from OFF to ON with a message; and an
option set to OFF by hand while the probe default was ON stayed OFF through that configure and the next. Not
measured here: a re-probe that returns "measured working" or "measured broken", because the sandbox cannot launch;
the owner's reconfigure of `build-all` is where that shows.

### D36. Accept mode's bundled coupler rank was declared as part of U's own RankSet, so D24's fix mis-sized U's field

**Status: FIXED at the library and example levels; `verification.brusselator.participants.accept.3` (and
`.accept.5`, built but never run per D18's own ruling) are the host's rows to confirm.**

**What it was.** `examples/brusselator/participants/brusselator_graph_tool.cpp`'s accept branch declared U's
`RankSet` as its FULL local executable world -- its own physics rank(s) plus the coupler's bundled ones riding
along inside the same acceptor `MPI_COMM_WORLD` (D18's own routing around `connect_pairwise()`'s two-component
limit) -- with the coupler's own `RankSet` the trailing, legally overlapping subset of that range. Before D24,
every placement's decomposition came from `(rank_, num_ranks_)`, this process's position in the WHOLE run, so the
declared RankSet's own size was never consulted and the mismatch below could not arise. D24 fixed that: a
placement is now decomposed over its component's own declared `RankSet`, indexed by this process's position
within it -- correct for a component whose declaration already matches the ranks it actually runs on, and silently
wrong for one whose declaration includes ranks it does not (U's own bundled-coupler rank, after D18).

**How it surfaced.** Measured on the owner's host 2026-10-12, the first run where `parallel.connect.accept.*`
passed the accept handshake itself (MPICH 5.0.0a1, device `ch4:ofi`, after the reinstall D23's own update
records): `verification.brusselator.participants.accept.3` timed out at 120s.
`build-all/examples/brusselator-participants-scratch/accept-3/u_stdout.log` printed `FAIL(u_main): initialize:
status=-4 iteration region 'brusselator_window': window member 'U' registered 256 values for its export, but the
plan sizes that field at 128 on this rank` (`E-RT-SIZE`, `src/runtime/iteration.cpp`'s own window-member check,
`Runner::Impl::resolve_regions()`). `graph.yaml` for that row declared U's `RankSet` as `{first: 0, count: 2}` --
its own one physics rank (`--ranks-u 1`) plus the coupler's one bundled rank (`--ranks-coupler 1`) -- so D24's fix
decomposed U's 16x16=256-cell field over 2 ranks, giving 128 per rank, while `u_main.c` decomposes its physics
over `--ranks-u` (1 rank) alone and registers the whole 256. The 5-rank accept row (`--ranks-u 2 --ranks-coupler
1`) has the identical shape and would fail the same way; it needs 5 tasks and is the owner's to run, per D18's own
"build it but never run it" ruling, so it was never a candidate for this measurement.

**Fix.** A component's declared `RankSet` must be exactly the ranks that HOST it -- not every rank that happens to
share its executable. `brusselator_graph_tool.cpp`'s accept branch now declares U's own `RankSet` as `{0,
ranks_u}` (its own physics ranks only), the coupler's as `{ranks_u, ranks_coupler}` (the ranks U's own declaration
used to also claim), and V's as `{ranks_u + ranks_coupler, ranks_v}` (unchanged). This leaves a GAP -- the
coupler's own ranks -- between U's declared block and V's, where before D36 there was none (U's block ran all the
way to where V's began). `fuse::transport::validate_launch_layout()` (`include/fuse/transport/connect.hpp`,
`src/transport/connect.cpp`) is the function this gap would have been refused by (it requires every Connect/Mpmd
component's `RankSet` to be a consecutive block in graph order, with nothing said about what -- if anything --
may occupy the space between two of them), so it gained the other half of this fix: a gap between two consecutive
Connect/Mpmd blocks is now accepted exactly when it is covered, rank for rank, by in-process/plugin components
whose OWN ranks lie entirely inside that one gap (`check_gap_filled_by_in_process_components()`, same file).  Two
new refusals, both named with the ranks at fault: an UNCOVERED gap rank that no in-process component's `RankSet`
names, and a STRADDLING in-process component whose ranks reach across the boundary between a gap and a launched
block (or between two gaps) rather than sitting entirely inside one. The pre-D36 declaration (U's own range
spanning the coupler's, no gap at all) still validates unchanged -- it never reaches the new check, since there is
no gap to check.

**Not the whole of multi-executable yet.** `connect_pairwise()`'s own merge numbering (`MPI_Intercomm_merge`'s
high/low fixed by `connect_mode_components()`'s graph order, not by acceptor/connector role) is unaffected by this
fix: it depends on which components declare `LaunchMode::Connect` and their order in the graph, never on a
RankSet's own COUNT. Audited for the same class of mistake (deriving a world extent from a component's declared
RankSet count rather than its actual membership): `src/runtime/runner.cpp`'s MLS-remap seriality check
(`graph.component(owner).ranks.size() > 1U`, around line 638) reads a component's own RankSet size to decide
whether its placement is parallel -- unaffected by D36 in the sense that it is not wrong, and is in fact MORE
correct after this fix, since U's own RankSet size now correctly reflects how many ranks U itself is placed on,
rather than including a bundled rank that was never part of U's own placement; `src/model/yaml_io.cpp`'s
`{first, count}` writer (lines 443, 517) emits whatever `RankSet` it is given byte-faithfully and asserts nothing
about what the count means, so it is unaffected either way; `src/abi/participant.cpp` and `src/model/validate.hpp`
were grepped and found to derive nothing from a RankSet's own count (only emptiness, `E-PLACEMENT-EMPTY`, and this
process's own `(rank, num_ranks)` from the real communicator).

**A second defect, found by the same host run, in the same function, folded into this fix rather than left for
later: the `LowFive` skip-list gap.** First flagged here as unmeasured (this session's own first pass over
`validate_launch_layout()`), then MEASURED the same day on the owner's machine in a LowFive-enabled tree
(`build-lowfive`, independent of the sandbox's own `FUSE_HAVE_LOWFIVE`-less build): `verification.brusselator.
participants.lowfive.mpmd.3` failed in 1.7s. `coupler_stdout.2.log` printed `FAIL(coupler_main): setup: status=-4
component 'lowfive' starts at rank 0, but the launched components before it in the graph's own order occupy ranks
[0, 3); the launched components' rank sets must be consecutive blocks ...` against a `graph.yaml` declaring U
`{0,1}` mpmd appnum 0, V `{1,1}` mpmd appnum 1, coupler `{2,1}` mpmd appnum 2, and a fourth, fieldless `lowfive`
component (S6 phase 3, "LowFive" track, Unit 3's own marker, `brusselator_graph_tool.cpp`'s `--transport lowfive`
branch) with `ranks: {first: 0, count: 3}` and `launch: {mode: lowfive}`.

**Cause.** `validate_launch_layout()`'s skip-list for components needing no rank block of their own named
`InProcess` and `Plugin` only -- every OTHER mode, including `LowFive`, fell into the "launched, needs a
consecutive block" branch. Checked against the two other places that describe what a LowFive-launched component's
ranks mean: `connect()`'s own `case LaunchMode::LowFive` (`src/transport/connect.cpp`) gives it the IDENTICAL
refusal `connect()` gives `InProcess`/`Plugin` ("there is nothing to connect, and `my_world_comm_f` already is the
base communicator"), with an explicit comment that this is "the same refusal shape as InProcess/Plugin above";
`fuse::LaunchMode`'s own doc (`include/fuse/model/launch.hpp`) says of `LowFive` that "where do my ranks come
from" is unaffected by it -- "today, always in-process". The two agree with each other and disagree with
`validate_launch_layout()`'s own skip-list, which is the bug: nothing in either doc source supports treating a
LowFive marker as a block that needs consecutive placement, and no test caught it because none exercised
`validate_launch_layout()` against a LowFive-mode component at all (`test/verification/brusselator_graph_tool_
lowfive_test.cpp` checks `validate()`, a different function).

**Fix.** `LaunchMode::LowFive` joins the skip-list, with its OWN `continue` (not folded into the InProcess/Plugin
one) for a reason specific to it: unlike a real in-process coupler, a LowFive marker's `RankSet` is a cross-cutting
annotation over however many ranks the run actually has (today, the WHOLE run) rather than a placement confined to
one gap between two launched blocks, so it is deliberately NOT added to D36's own `fillers` list either -- doing
so would make every accept-mode row with a LowFive marker refuse as a STRADDLE (the marker's ranks touch the gap
AND both launched blocks on either side of it).

**Pinned by** three new `ValidateLaunchLayout` cases in `test/transport/connect_test.cpp`: the exact mpmd+lowfive
shape from the host's own `graph.yaml` now validates; the SAME marker alongside D36's own accept-mode gap (U's
block, the coupler's gap, V's block) also validates, confirming the marker is not treated as a straddling filler;
and a lowfive-only graph (no launched component at all) still gets the pre-existing "no component declares an
mpmd or connect launch" refusal, unchanged. A fourth mutation (`build-evidence/mutations/accept-layout.toml`'s
`accept-layout-lowfive-counted-as-launched`) restores the old, unskipped behaviour and is killed by all three.
This entry does not close D14: D14 is about the two-HDF5-image hazard and LowFive's own configure-time probe, which
this defect is independent of -- the lowfive ROW this fixes was never blocked by D14 on the owner's own
`build-lowfive` tree (that tree already has the fmt-12 rebuild D14's own Unit 3b account names), only by this
separate layout bug.

**Pinned by** `test/transport/connect_test.cpp`'s eleven new `ValidateLaunchLayout` cases (the D36 shape, the
pre-D36 shape unchanged, a Plugin-mode gap filler, a two-component split gap, an uncovered gap, a partly-covered
gap, the two straddle directions, and the three LowFive cases above), `test/verification/
brusselator_accept_layout_test.cpp`'s six new cases (U's and V's own planned placement sizes on every rank of the
3-rank accept-3 world, the graph tool's own output invariants, and that the real graph validates), and three new
committed references, `examples/brusselator/participants/reference_accept_{3,5}.yaml` plus
`reference_accept_3_lowfive.yaml` (the exact `--transport lowfive --launch accept` combination the LowFive
skip-list defect below was measured against), with their own
`verification.brusselator.graph_tool.reproduces_reference_accept_{3,5,3_lowfive}` byte-for-byte rows (accept mode
had no committed reference of any kind before this fix). Five mutations
(`build-evidence/mutations/accept-layout.toml`); a sixth, pre-existing mutation
(`build-evidence/mutations/transport-connect.toml`'s `connect-merge-order-not-enforced`) had its own anchor
updated to this fix's restructured loop and was re-verified against it.

**What the host must verify.** `verification.brusselator.participants.accept.3` (3 tasks across two `mpiexec`
invocations: U's executable at 2 ranks, V's at 1) should now reach past `fuse_initialize()` and run to completion
within the 120s timeout, printing matching `mean_u`/`mean_v` values to `examples/brusselator`'s own single-process
reference the way `mpmd.3` already does (D25's own measurement). `verification.brusselator.participants.accept.5`
(5 tasks, U's executable at 3 ranks, V's at 2) is built and this fix applies to it identically, but D18's own
ruling keeps it unrun without the owner's explicit say-so. This entry does not close D23: D23 is about whether
accept/connect can complete a handshake on this MPI at all, which the owner's own 2026-10-12 update already found
cleared, and the layout bug this entry fixes is downstream of that, found only once the handshake started
working. D23 stays open until the host's own `accept.*` rows are the ones to decide it, not this fix.
`verification.brusselator.participants.lowfive.mpmd.3` (the row that measured the LowFive skip-list defect,
3 tasks, on the owner's own `build-lowfive` tree) should now pass `fuse_setup()`'s own layout check; whether it
runs to completion and matches the non-LowFive reference is this entry's own second host-verification item,
independent of D14 and of D23 alike.

### D37. `pfuse` v1 (the pybind11 bindings) does not expose Newton, MPI, or either non-serial mesh backend to Python, and inherited one pre-existing naming bug in `dump_graphviz()` — OPEN, scope decision for v2

**What this track built.** `python/` (gated by `FUSE_ENABLE_PYTHON`, default OFF) binds `fuse::CouplingGraph`,
`fuse::Component`/`fuse::Runner` (via a `pfuse.Session` wrapper), `fuse::FieldRegistry`, the iteration/
accelerator/Newton-configuration model types, and the `fuse::Id`/`Rational`/`Units`/`Diagnostics` base types,
over `pybind11`. `docs/dev/python-surface.md` is the design; `docs/user/python.md` is the user-facing guide.

**Newton: native-only, by the owner's own ruling (2026-10-12), not a gap found late.** A Python-authored
`fuse::Component` subclass can be added to an ordinary (Picard/accelerated) iteration region and run through
`pfuse.Session`/`fuse::Runner` like any native component. It cannot be added to a Newton-solved region:
`fuse::Runner::initialize()`'s own `E-NEWTON-CAPABILITY` refusal (`src/runtime/checkpoint_support.cpp`) already
covers a component whose `capabilities()` does not claim `kCapabilityResidual`/`kCapabilityLinearization`, and
this track added no second refusal path for that case (`python/tests/test_newton.py`,
`TestNewtonCapabilityGate`). A Python component that CLAIMS either capability bit without `fuse::
ImplicitComponent`'s C++ hooks to back it (`residual()`/`jacobian_apply()`, neither exposed to Python) is
refused immediately when it is added to the graph, not deferred to `initialize()`
(`TestClaimingTheCapabilityIsRefusedImmediately`, same file) — `ForwardingComponent::compute_capabilities()`,
`python/src/component_bindings.cpp`. The existing refusal's own message says "override `Component::implicit()`
to return it," advice that is correct for a C++ author and impossible for a Python one, since pfuse does not
expose `Component::implicit()` at all; `test_newton.py` records the exact message
(`test_the_existing_refusal_message_names_cxx_overrides_a_python_reader_cannot_make`) for the owner to judge
whether a Python-specific wording belongs in that C++ message, which this track did not edit.

**MPI and the native/MOAB mesh backends are not reachable from Python at all in v1.** `python/CMakeLists.txt`
links only `fuse::model fuse::runtime fuse::backend fuse::ir fuse::schedule`; `fuse::SerialMeshBackend` is the
only backend `pfuse.CouplingGraph`'s mesh-loading path can resolve a `"generated:..."` source against, and
`fuse::RunnerConfig`'s default ("copy" transport, no `base_communicator`) is the only configuration `pfuse.
RunnerConfig` constructs. No MPI rank set beyond a single rank was run or is claimed to work.

**A genuine naming bug, found and reported, not fixed (out of this track's src/include scope).**
`include/fuse/ir/dump.hpp:126`, inside `dump_graphviz()`, still emits the literal header `"digraph soma_ir {"`
— a leftover from the soma → fuse rename that `dump_text()` (line 73, same header)
already carries correctly as `"# fuse-ir v1"`. `python/tests/test_graph.py` records this as an
`xfail(strict=True)` test citing both line numbers, so the suite documents the finding without silently
working around it and will fail loudly (an unexpected XPASS) the day someone fixes `dump_graphviz()` without
updating that test.

**Not pursued, no decision needed yet:** `ImplicitComponent` bindings (would lift the Newton restriction
above); a Python-side MPI story (`mpi4py`, communicator passing into `RunnerConfig`); native/MOAB backend
bindings. None of the three has a design in `docs/dev/python-surface.md` beyond the "what this does not
expose" list — a v2 scope decision, not a defect in v1.
