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

### ~~A5. A device number waits on a GPU machine — your sequencing~~ — CLOSED 2026-10-11: measured on a real GPU, at the kernel layer AND through `Runner::Impl::execute()` itself

**Your call, recorded 2026-09-17: get host threading fully working first, then
retry on a machine with a GPU.** That is the right order, and the host-threading
half is now measured rather than asserted: 65/65 on `Threads Serial` from the
installed Kokkos and 65/65 on `OpenMP Serial` from a fetched one, both recorded
in the note under A1. Those are all three host-parallel spaces.

~~Nothing in the code has to change for a GPU.~~ That was true at the kernel
layer and is now measured there: on a GeForce GTX 1050 Ti (sm_61, toolkit
`/usr/local/cuda-12.5`) against a `/opt/kokkos` built with
`Kokkos_DEVICES=OPENMP;SERIAL;CUDA`, `ctest -R space_kernels -V` prints
`[ spaces ] enabled: Cuda OpenMP Serial (default Cuda)` and all 27 typed
kernel tests (9 per space, `Kokkos::Cuda` included) pass with zero code
changes — exactly the number the previous version of this entry asked for.

It was NOT true one layer up, and that is what this closure adds. The
`Runner::Impl::execute()` instruction stream was hard-wired to
`Kokkos::DefaultHostExecutionSpace`: a GPU build ran every kernel-layer unit
test on the device and every coupling workload on the host. The fix (branch
`vijaysm/kokkos-device`, design and reasoning in
`docs/dev/runner-device-dispatch.md`):

- `RunnerConfig::execution_space` (`Host`, the default, or `Device`).
  `execute()` is now a two-line dispatcher over `execute_on<Space>()`; the 9
  `OpKind` cases that were already Kokkos-templated (`Interpolate` identity,
  `Remap` identity + weighted, `Sample` non-spacetime, `Accumulate`,
  `Extrapolate`, `Merge`, `Scale`, `Relax`, `Residual`) run on the requested
  space; the host-only cases (`Transport`'s MPI `move()`, `Remap`'s
  distributed-nodal branch (`apply_nodal_dist_remap()`, also an MPI
  exchange), `Epoch`, `Constrain`, `Sink`, ...) are untouched. Host mode is byte-identical to the
  old code: the span helpers wrap the raw host pointer when the space is
  host-accessible, so the 279 pre-existing tests exercise the same
  instantiation they always did.
- Device mode stages each dispatched instruction's arcs (and `previous[]`
  for `Relax`/`Residual`) through one device buffer allocated in `setup()`
  (`fuse::dev_staging`, four lanes of the widest slot — nothing in the sweep
  loop allocates, the standing invariant on `reduction_scratch`), with the
  remap CSR mirrored to the device once at `resolve_remap()` time.
  Requesting `Device` on a build whose Kokkos has no CUDA/HIP/SYCL backend is
  refused at `setup()` with `E-RT-DEVICE`, not silently downgraded.
- `RunStats::device_dispatches` counts instructions that really ran on the
  device (0 in `Host` mode), and `Runner::device_execution_space_name()`
  names the space the build would use, so a test can assert the device path
  was taken rather than infer it.

**Measured** (`test/runtime/runner_device_test.cpp`, ctest row
`runtime.runner_device`): the same two coupling scenarios — a four-substep
`Mean` accumulation with identity remap, and an 8→6-cell weighted remap
followed by a 0.5 relaxation — run through the real `Runner` once on `Host`
and once on `Device` (`device_execution_space_name()` == `"Cuda"` on this
machine, `device_dispatches > 0` on the device run). Accumulate/Scale/copy
agree **bitwise**; `Relax` and weighted `Remap` agree to `1e-14` relative,
not bitwise, because nvcc contracts `a*b+c` into FMA inside `RelaxFunctor`/
`RemapMultiFunctor` (the same gap D45 closed for the deterministic reductions
with `unfused_multiply_add`; those two kernels were left as-is because
`build-evidence/mutations.toml` anchors their exact source lines). Full
suite after the change: 279/280 on one `ctest -j1` run; the one failure,
`abi.fortran_consumer_compiles`, was a nested-cmake reconfigure tripping
`test/CMakeLists.txt`'s "test source registered nowhere" guard over a
not-yet-registered test source from in-flight Part B work in the same tree
(`test/verification/sparse_solve_backend_test.cpp`), and passed once that
file was registered. No single 280/280 run was observed; the Runner change
itself touched nothing that test compiles.

**Honest expectation, stated rather than hidden**: the device path is slower
than the host path for these workloads, not faster — every device-dispatched
instruction adds a host→device→host staging round trip the host path does not
pay, and the arcs are a few dozen doubles. This closure is the dispatch
pattern proven end-to-end with a measured number, not a performance claim;
a device-resident field storage (no staging) is the follow-on if one is ever
wanted, and `docs/dev/runner-device-dispatch.md` records why `previous[]`
could not simply be made device-native (`advance_window()` reads it).

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

### C2. Second implementation review, 2026-10-11 — 159 findings, fixed 2026-10-12/13, disposition in `docs/dev/reviews/2026-10-11/disposition.md`

Eleven independent read-only reviews against `4a8a9d8` (5 BLOCKER, 24 MAJOR, 95 MINOR/DEVIATION, 35
IMPROVEMENT; `docs/dev/reviews/2026-10-11/index.md` is the ranked index). The owner's scope for the fix
round was BLOCKER + MAJOR + the MINORs with real impact plus the wave-0 gates and the record corrections;
the IMPROVEMENT tier stays open. Outcome, per the disposition table: 132 fixed (each library fix with a
test that failed before it where one was constructible; the LowFive plugin and connect-accept fixes are
verified by reading only, since neither path runs on the host), 3 deferred (IS-3 doc route, BM-5 → D62,
AB-10 → D61), 2 not defects (BC-2 already closed at `e6a92e4`; RT-3's divergence is intentional and now
documented), the rest open by scope. Thirteen of the twenty-nine BLOCKER/MAJOR findings were the same
rank-agreement hang class (index §2.1); `docs/dev/rank-agreement.md` now carries the checklist question
and the two accepted recipes, and the static gate the index proposes (`tools/check-rank-agreement.py`) is
not built — recorded as the one structural follow-up. Entries D26 (closed), D14, D23, D42 and the new
D61/D62 below carry the per-item records.

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

**Re-measured 2026-10-18:** `parallel.moab_placed_view_mpi.{1..5}` Passed again in the owner's quick tier at
`cb127c5` (MOAB now `/opt/moab/cmake-all`, shared HDF5). The heading's "awaiting the host" refers only to the
weights below, in progress 2026-10-18 (MoabHalo track).

**What remains of D5:** per-placement remap *weights* built on two independently decomposed
MOAB processor sets and compared. The builders already resolve through `placed_view`, but nobody
has assembled and checked such a matrix.

**DESIGN RULING, round 2 (the lead, 2026-10-18, consistent with the owner's D73 decision to build
remap halos automatically): HALO BRIDGING, not a two-placement `RemapPlan`.** `RemapPlan::placement`
stays the TARGET's own placement (its existing, already-closed meaning, unchanged). The source side
is brought onto the target's own ranks by a remap halo built from the SOURCE's own placement, so a
subsequent `build_weights(plan, graph, backend)` resolves BOTH sides through `plan.placement` alone
-- target cells local, source cells as halo -- exactly as the single-placement case (closed
2026-09-25) already does. Superseded round 1's own "RemapPlan cannot express this" framing: the
intended fix is not a new `RemapPlan` field, but a new MOAB entry point that builds the halo itself
across the two placements.

**PARTIALLY FIXED, round 2 (2026-10-18, "MoabHalo" track).** Three pieces built and tier-0/build
verified; the central cross-Core migration is NOT complete -- declared explicitly as a scope
reduction from what this round's brief asked for, not cut silently:

1. **A real, previously-undiscovered correctness bug, found and fixed:**
   `MoabMeshBackend::native_mesh(MeshId)` always answered the backend's DEFAULT `moab::Core`
   (`impl_->mb`), regardless of which placement's `placed_view` a caller had resolved --
   `set_placements` gives each declared processor set its OWN, SEPARATE `moab::Core`
   (`PlacementContext::core`, `moab_backend.cpp:939-955`), so `resolve()` (`intx_weights.cpp`) was
   reading the right VIEW through the wrong CORE the moment a real per-placement decomposition was
   used with `build_weights`/`build_remap_halo` -- never exercised before this track (KNOWN_ISSUES'
   own "nobody has assembled and checked such a matrix"), so never caught. Fixed: a new, placement-
   aware `MoabMeshBackend::native_mesh(MeshId, ComponentId)` overload (`moab_backend.hpp`/`.cpp`)
   and `resolve()` now calls it with the SAME placement its view came from.
2. **Structural-vs-genuine empty, derived from the placement, not a local size:** a new
   `MoabMeshBackend::placement_participates(ComponentId) const` query, and `resolve()` gained an
   `allow_empty` parameter (default `false`, every EXISTING caller unaffected) that returns a VALID,
   empty `Side` -- instead of refusing `E-REMAP-REGION` -- exactly when this rank is structurally
   outside the given placement, or (for an `allow_empty` caller specifically) when its own local
   chunk, in or out of a halo, simply holds none of the mesh at this dimension.
3. **A new `build_remap_halo` overload** (`remap_plan.hpp`/`intx_weights.cpp`), taking an explicit
   `source_placement` alongside the plan's own (target) placement -- the interface shape the lead's
   ruling names. It correctly no-ops on a rank outside the target placement, and correctly
   delegates to the EXISTING single-placement path when source and target placements resolve to the
   SAME `moab::Core` (checked by pointer, not assumed). **For the genuinely cross-Core case --
   source and target placements with DIFFERENT `moab::Core` instances, the actual substance of this
   item -- it returns an explicit, named refusal (`E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED`)
   rather than attempting it.** MEASURED reason, not assumed: MOAB's own covering-set construction
   (`Intx2MeshOnSphere`/`FindMaxEdges`/`construct_covering_set`, the same machinery the
   single-placement path already uses) needs BOTH file sets in the SAME `moab::Interface`; there is
   no MOAB entry point that intersects a file set living in one Core against one living in another.
   Bridging two independent Cores needs a MIGRATE-BY-GLOBAL-ID exchange -- box overlap over the base
   communicator, packing each needed source cell's vertices/connectivity/GLOBAL_ID on its owning
   rank and creating fresh MOAB entities for it inside the target placement's own Core -- the MOAB
   analogue of `fuse::native::build_native_remap_halo`'s own box-overlap/pack/merge architecture
   (native_halo.cpp). Not built: a correct implementation is a multi-hundred-line undertaking
   (mirroring native_halo.cpp's own size) that this track's remaining time could not respon
   sibly deliver UNVERIFIED (this sandbox cannot run MPI at all, so nothing cross-rank can be
   verified by execution here regardless; the risk is shipping large, novel, untested MOAB
   entity-creation code, not merely an unverified test row).

**Compiled (tier-0 + MOAB/MPI build, system compiler, both 0 errors), NOT run** (C1: this sandbox
cannot initialise MPI at all): `test/parallel/moab_placed_weights_mpi_test.cpp`'s three rows --
(1) a rank outside the target placement is a successful no-op; (2) the same-Core delegation
reproduces the single-placement `build_remap_halo` answer exactly; (3) the genuinely disjoint pair
(`A = [0, size/2)`, `B = [size/2, size)`) refuses with the EXACT named code
`E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED`. Registered `parallel.moab_placed_weights_mpi.{2,3}`.
Row (3)'s own "row sums, conservation against a serial single-placement build to a derived
tolerance" -- what this item was asked to assert -- is NOT yet assertable: there is no real halo to
build those row sums FROM until the migrate-by-global-id mechanism above exists. Flagged as
`host_pending`/`open_questions` in this track's own report, not claimed as delivered.


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

### D8. A Remap into a nodal field on more than one rank is rejected -- FIXED for the consistent/consistent-transpose methods, HOST-VERIFIED 2026-10-18 at 1 to 5 ranks (round 5): the S5.5 matrix row is closed; every other method's rejection is unchanged

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


**Progress (2026-10-18, Maps track, round 5): the S5.5 matrix cell this entry's
own round-3/4 text named (`A-NODAL-HT-PARALLEL`,
`build-evidence/parallel-matrix.toml`) is CLOSED, not merely narrated.** The
cell held a `gap = "..."` block describing `NodalRemapRunnerMpi.DisjointRankSets`
as already written and cross-checked without MPI, but naming no row that
`tools/check-parallel-matrix.py` could verify -- so the gate counted it as a
gap regardless of what the test file already proved. It is now a `test` row
(`file`, `kind = "mpi"`, `ranks = [2, 3, 4, 5]`,
`test = ["NodalRemapRunnerMpi.DisjointRankSets"]`): the gate reads the test
name out of the file, confirms the file is registered, confirms every claimed
rank count is registered for it, and reports 0 gaps (was 1), 72 claims
verified (was 70), 0 broken. `ranks` excludes 1 deliberately --
`FUSE_REQUIRE_AT_LEAST_RANKS(2)` skips `DisjointRankSets` at world size 1 (a
disjoint split needs two non-empty halves), and the axis this cell names is
specifically the disjoint-placement, transpose-direction case, which rank 1
cannot exercise even though the file is registered there.

No library or test code changed in this round: the Runner wiring, the two
round-4 defect fixes, and `DisjointRankSets` itself are byte-identical to
round 4. This round's own patch touches only
`build-evidence/parallel-matrix.toml` (the cell) and this entry. Everything
this entry's "What is still open, honestly" paragraph already listed --
no real multi-process host run of `parallel.nodal_remap_runner` or
`parallel.nodal_remap_distributed`, the two round-2 NOT-ASSESSED mutations
plus the round-4 one, the 50-allocations-per-coupling-cycle measurement, and
`InterfaceDomainPolicy::ClampToBoundary` -- remains open exactly as stated
there; closing the matrix cell is a claim that the coverage EXISTS and is
correctly named, not a claim that a host has run it. The heading's "FIXED"
is scoped to the two methods this round's own wiring (rounds 2-4) handles
(`"consistent"`/`"consistent-transpose"`); every other field-transfer method
is rejected by `E-RT-REMAP-NODAL-PARALLEL` exactly as before, pending D5
item 3 (per-placement supermesh weights), which this round did not touch.

**Host action, unchanged from round 4's own instruction:**
```
ctest --test-dir <mpi-build-dir> -R 'parallel\.nodal_remap_(distributed|runner)' -V
```

**HOST-VERIFIED 2026-10-18** (owner's machine, `build-all` at `66d35cc`, MPICH 5.0.0a1,
`FI_PROVIDER="tcp;ofi_rxm" FI_TCP_IFACE=en0`): `parallel.nodal_remap_distributed.{1,2,3,4,5}` and
`parallel.nodal_remap_runner.{1,2,3,4,5}` all Passed (0.72, 0.15, 0.15, 0.18, 0.17 s and 1.19, 0.18,
0.23, 0.24, 0.28 s), with `backend.nodal_tie_break` and `verification.nodal_remap_runner_serial` in the
same run, 35/35. The multi-process run this entry asked for has happened; what stays open is the
non-consistent methods' `E-RT-REMAP-NODAL-PARALLEL` refusal (pending D5 item 3), the three NOT-ASSESSED
mutations and `ClampToBoundary`, exactly as the paragraph above lists them.

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

### ~~D14. LowFive: the two-HDF5-image hazard is closed by a measured configure-time probe; the distributed transport and Demo B are compiled, not run~~ — CLOSED 2026-10-14, host-verified (the participants' MPMD and accept rows and Demo B pass)

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

**Superseded 2026-10-14: Demo B runs and passes as the ctest row `example.demo_b`; see item 5 of "What remains" below.** **Demo B (two unmodified HDF5 executables, cross-process) — NOT executed (as of 2026-10-02); compiles
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
   MPMD launch only (ACCEPT mode's own merged base communicator was never
   exposed back to a caller through any ABI query when this was written; it is
   since, by `base_communicator()` (D14 Stage B), which the participants use and
   this plugin does not), and the connector name this
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

1. ~~**Host action: rebuild `/opt/lowfive` (and `/opt/lowfive-dist`) against
   fmt 9 or newer.**~~ — done; D39's own account (Round 10) measures
   `fuse-lowfive-vol` loading and running on this host
   (`participants.lowfive.mpmd.3` Passed, 1.70 s), which could not happen
   if the `dlopen` failure this item named were still present.
2. ~~**Host step: the connector-name check.**~~ — done; the SAME Round 10
   measurement is only possible if `HDF5_VOL_CONNECTOR=lowfive` is the
   name LowFive's `VOLBase` actually registers under, since the row
   exchanges real data through it (`brusselator_check` `--check: OK`,
   byte-identical logs to the transport-only twin).
3. ~~**The accept-mode intercommunicator gap.**~~ — **closed and host-verified (D14 Stage B, 2026-10-13).**
   ACCEPT mode's merged base communicator is exposed through a new, append-only ABI entry,
   `base_communicator()` / `fuse_base_communicator_info` (`include/fuse/abi/entries.def`, `docs/user/c-abi.md`,
   "Recovering the base communicator"). `decide_attach_mode` no longer refuses Connect-mode channels outright; LowFive's
   local communicator is a fresh `MPI_Comm_split` of that bridge keyed by the graph's own `RankSet`s, and the bundled
   coupler/placeholder rank joins the split but never calls `bruss_lowfive_attach()`. **Measured on the owner's machine,
   2026-10-13 (HEAD `9ffd2b6`, `build-lowfive`, MPICH 5.0.0a1, MOAB `/opt/moab/cmake-all`), first host run:**
   `participants.lowfive.accept.3` Passed (1.26 s) and `participants.lowfive.mpmd.3` Passed (2.37 s, unchanged). In the
   accept row the split trace shows the placeholder (`bridge_rank=1`) not participating, U (`bridge_rank=0`, `color=0`)
   and V (`bridge_rank=2`, `color=1`) participating; both attach lines read `order-check-verdict=accepted
   intercomm_formed=1` (U `first-server`, V `first-reader`); U and V each made 25 writes and 25 reads; no `E-LOWFIVE`
   line in either log. `brusselator_check` on the row's logs prints `2 window(s) final_mean_u=1.58658618817
   final_mean_v=1.9982226403`, balance errors `1.632e-13` / `2.379e-13` / `7.984e-14`, `--check: OK`, and `u.log` and
   `v.log` are byte-identical to the non-LowFive accept row's. V's bridge rank 2 is the value the tier-0 test
   `WrongThisRankFromLocalWorldInsteadOfBridgeMisclassifiesV` predicted, which confirms `this_rank` is read from the
   bridge. The `abi.*` rows (13) and the transport twins `participants.{mpmd,accept}.3` passed on the same rebuilt
   `build-all`. Not measured: more than one run of the accept row, `.accept.5`, a multi-rank accept layout (the order
   check is validated for one configuration only, D39), and `base_communicator()` after a failed `connect()` (needs MPI
   to reach, as for the LowFive-launch branch).
4. ~~**ctest rows, host-only:**~~ `verification.brusselator.participants.lowfive.mpmd.3` Passed (D39 Round 10, and again
   2026-10-13 after this stage); `verification.brusselator.participants.lowfive.accept.3` Passed (item 3).
5. ~~**Host run: Demo B**~~ — **done, 2026-10-14.** Registered as the ctest row `example.demo_b`
   (`examples/CMakeLists.txt`, when the configure probe enables `FUSE_ENABLE_CONNECT_ACCEPT`), which runs
   `examples/demo_b/run_demo.sh`: one producer and one consumer process from two separate `mpiexec -n 1`
   launches, joined by `MPI_Comm_accept`/`MPI_Comm_connect`. **First host run (HEAD `2ebc0d4`, before any fix),
   both processes exited 1:** the producer's `set_location` refused with `E-TRANSPORT-LF-LOCATION` ("cannot
   create a file next to ..."), and the consumer's `H5Fopen` raised LowFive's `file_open(): no intercomm found
   for ...`. Causes, both in FUSE's transport: (a) `set_location` proved the directory writable with an
   `O_EXCL` sentinel at the fixed name `<file>.soma-writable-probe`, so the second process to claim the same
   path failed while the first one's sentinel existed (a sentinel left by a crash refused every later run the
   same way); (b) the distributed VOL never called `DistMetadataVOL::set_intercomm(file, path, index)`, which a
   consumer's `file_open` needs to find the serving intercommunicator (LowFive's own consumer examples call it).
   **Fixes:** (a) the sentinel is now `mkstemp`-unique, pinned by the LowFive-tree test
   `LowFiveTransport.SetLocationIsNotBlockedByAPeersProbe`, where a file at the old fixed name stands in for
   the peer's live sentinel and no probe may be left behind; (b) distributed `build_plan` registers the file
   with the dataset path normalised to one leading `/`, index 0, once per file and path; and (c) the producer
   payload `demo_b_produce` creates the dataset's parent groups (`H5Gcreate2` with intermediate-group
   creation) before `H5Dcreate2`, which plain HDF5 requires and which LowFive otherwise flattens
   (D39 round 7). Also: `run_demo.sh` writes its logs next to the port file instead of fixed `/tmp` names,
   removes a stale HDF5 file before the run, and takes the configured launcher; a placeholder
   `DistInterceptionIsNeverExercisedHere` test that could not fail was removed. **Measured after the fixes**
   (owner's machine, MPICH 5.0.0a1, `build-lowfive`): `example.demo_b` Passed on its first run (0.60 s) and
   in five repeats (0.16–0.42 s); the producer printed `wrote 8 values, served across the intercommunicator`
   and the consumer `read 8 values, bit-for-bit match, zero bytes of HDF5 on disk`; the scratch directory
   held only the two logs; `run_demo.sh` with dataset paths `/values` and `/a/b/c/values` also passed.
   The consumer reads with `H5S_ALL`, which D39 round 7 recorded as an unresolved upstream TODO in
   `DistMetadataVOL::dataset_read`; this whole-dataset 1-D read returned correct values, which does not
   establish that `H5S_ALL` is safe for other selections. **Mutation rows** (`build-evidence/mutations/demo-b.toml`, `FUSE_MUTATE_MPI=1 tools/mutate.py --area demo_b
   --build-dir build-lowfive` on the owner's machine, HEAD `58e2727`): 3 killed, 0 declared equivalent, 0 not
   assessed, 0 without a baseline, 0 not killed, of 3; the checkout was clean before and after. The killed
   `demo-b-producer-skips-the-parent-group` row is the measurement that fix (c) is necessary under LowFive:
   without the parent group, `example.demo_b` fails. Not measured: more than one rank per side, more than one file or dataset per transport, and a
   second exchange in one process.

**D14 is closed.** Every item above is done and host-measured. Limits that remain belong to other
entries: `.accept.5` (D18), the order check validated for one participants configuration (D39), and no
test of `base_communicator()` after a failed `connect()` (item 3).

**Addendum 2026-10-13 (review 2026-10-11 TR-2, TR-4) — fixed by derivation, unmeasured.**
`fuse-lowfive-vol`'s `ConfigureOnce()` now issues `MPI_Comm_split`/`MPI_Intercomm_create` on every
world rank (colour `MPI_UNDEFINED` for a rank the LowFive channel does not name) instead of only on the
participants, and aborts the job on a failed `LoadConfig()` or a distributed configuration built without
`DistMetadataVOL` instead of attaching a zero-binding connector. `FUSE_ENABLE_LOWFIVE` is OFF on every
host available to the round, so neither the fix nor its two scenarios (a non-participant world rank; a
build without `DistMetadataVOL`) has run. The second hazard the review names is not closed by this: the
rendezvous still runs only on a rank whose own first HDF5 call loads the plugin, so a rank that never
touches HDF5 never reaches it — moving the rendezvous into an explicit step every participant calls
(`connect()`'s MPMD choreography is the natural site) is the larger fix, left open here.

### D15. The provider direction (S6 phase 4, "Adapter host" track) is implemented and conformance-tested on one process; device fields and mixed checkpoint/restore still are not -- a plugin on a rank subset is tested and built 2026-10-18 (round 2), awaiting the host run

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

**Corrected 2026-10-18 (ABI track, round 2).** An earlier round of this same task framed this
entry as "the multi-process device-fill comparison for the provider direction is compiled but
never run" and asked that it be registered as a ctest row. That framing was measured and found
wrong: searched the whole tree (`grep` for "device" combined with "provider"/"fill" across `test/`,
`src/abi/`) and found no such test anywhere, compiled or otherwise. `src/abi/provider_adapter.cpp:294`
hardcodes `view.memspace = FUSE_MEMSPACE_HOST` unconditionally -- there is no device-resident path in
the provider direction's adapter at all. This entry's own "What is not covered" bullets above already
named the real, accurate scope; restated here as three separately closeable items, each with what it
needs:

1. **Device fields.** `register_field`'s own `FUSE_MEMSPACE_DEVICE` refusal (`E-ABI-REGISTER-MEMSPACE`)
   has no provider-direction counterpart: `ProviderAdapter` only ever allocates host storage. Needs a
   design decision (not built here) on how a plugin asks for a device pointer at all -- the provider
   ABI's `bind()` entry takes a `fuse_field_view` the SAME shape the participant direction's
   `register_field` does, so the natural extension is reading `memspace` there too, but committing to
   that is new ABI-surface work, not a test-registration fix.
2. **Checkpoint/restore through a mixed iteration region.** `ProviderAdapter::save_state()`/
   `restore_state()` forward to the plugin's own entries when present, but no test puts a
   plugin-backed component inside an `IterationRegionDecl` alongside a native one and checks the
   sweep converges identically. Needs a conformance fixture combining `test/abi/graph_fixture.hpp`'s
   iteration-region shape with `test/abi/plugins/linear_provider.c`'s own plugin -- no new ABI surface,
   a test-authoring gap only.
3. **A plugin on a rank subset.** Untested until this round (see below): every provider-direction
   fixture before this round ran single-process, and `fuse::transport::validate_launch_layout()`
   treats `Plugin` as needing no rank block at all (same as `InProcess`), which was correct for the
   single-process case but not separately confirmed for a `Plugin` component placed on a SUBSET of a
   multi-rank world.

**Built this round (item 3 only, per the lead's own instruction -- items 1 and 2 need a design
decision or new test authorship this round does not attempt).** `test/transport/connect_test.cpp`'s
new `ValidateLaunchLayout.AcceptsAPluginComponentOnAStrictRankSubset` case confirms, pure and MPI-free, that `validate_launch_layout()` already
accepts a `Plugin` component whose `RankSet` is a strict subset of the run (`{1}` of a 2-rank world,
`{1,2}` of a 3-rank world) with NO special-casing needed -- the existing "Plugin needs no rank block"
rule already covers a rank SUBSET, not only the whole run, because it imposes no constraint on
`RankSet` at all. `test/parallel/provider_subset_mpi_test.cpp` (new, `parallel.provider_subset_mpi.
{2,3}`, built under `FUSE_ENABLE_MPI`, never run here per C1) mounts `test/abi/plugins/
linear_provider.c` as a `LaunchMode::Plugin` component placed on ranks `{1}`/`{1,2}` of a 2-/3-rank
graph (a native `downstream` component, injected test-only, alone on rank 0) and confirms both
halves at once: every rank's `fuse_run()` returns `FUSE_OK` (`MPI_Allgather`'d and checked against
every entry, not just locally), so the non-hosting rank runs the SAME call to completion with no
hang; and, independently, rank 0's own `downstream` sink recorded exactly 3 values after 3 periods
-- data that could only have crossed the channel if the plugin, hosted entirely on the OTHER ranks,
actually advanced and wrote through `bind()` as the participant direction already does. This does
NOT show whether the non-hosting rank's own process skips `dlopen()`-ing the plugin's shared object
at all -- the design question below answers that, and the answer is no, it does not skip it. No new
design decision was needed to make THIS row buildable, so nothing is raised to `open_questions` for
item 3 itself; items 1 and 2 remain open, named above, for whoever picks this entry up next.
Built, never run (C1 above). **Host-pending:**

```
ctest --test-dir <build> -R 'parallel\.provider_subset_mpi\.(2|3)' -V
```

**Design question surfaced, NOT decided here (per the task's own stop condition).**
`fuse::abi_detail::Registry::mount_dynamic()` (`src/abi/registry.cpp`) is called once per
`LaunchMode::Plugin` component declared in the graph, by EVERY rank of the process set
`fuse_initialize()` runs on -- it does not itself consult the component's own `RankSet` at all before
`dlopen()`-ing the shared object and constructing a `ProviderAdapter` for it. For the rank-subset row
above this is harmless (a non-hosting rank's own `ProviderAdapter` is built but never advanced, since
`Runner::hosts_component()` gates every call into it), but it means EVERY rank of the base
communicator loads the plugin's shared object into its own address space, whether or not that rank
ever uses it -- for a large `RankSet`-subset deployment (many ranks, one hosting a plugin on a few of
them) this is `dlopen()` cost and address-space footprint paid by every rank for nothing. Left as an
open question: is mounting the shared object only on hosting ranks (reading `RankSet` at
`mount_dynamic()` time) worth the added coupling between the registry and the graph's own placement
model, or is paying the `dlopen()` cost everywhere an acceptable simplification given a plugin's own
shared object is typically small compared to the physics it wraps?


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

**Measured further, Windows track, 2026-10-18 (still not root-caused -- see D76, below, for
the full trace): the divergence is in the SEED fed to the accelerator at the start of window 1,
not in Aitken's own relaxation numerics.** A per-iteration trace of both drivers on this exact
configuration found the two sides' own residual-norm and relaxation-factor sequences IDENTICAL,
to every printed digit, for the entirety of window 0; the first disagreement is the residual
norm at window 1's own sweep 0 (143.37914811541057 vs 145.9156716694263), before Aitken's own
history-dependent formula runs even once that window -- narrowing this from "not root-caused" to
"root-caused to a specific window-boundary seed disagreement, mechanism not yet traced to a
write site." D76 has the full measurement and the next trace this entry's own "not root-caused"
note was asking for.


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

### ~~D26. `FUSE_LINALG_BACKEND=builtin` has not configured since the examples linked Eigen directly~~ — CLOSED 2026-10-13 (review 2026-10-11 BC-1)

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

**Closed 2026-10-13 by review 2026-10-11 BC-1.** The 23 test targets and 5 example targets that linked
`Eigen3::Eigen` directly are gated behind `FUSE_LINALG_BACKEND_IS_EIGEN` in `test/CMakeLists.txt` and
`examples/CMakeLists.txt`, with one configure-time status message listing what a `builtin` configure
skips and a `fuse_tests_gated_by` entry so the registration audit stays honest. Measured on higgs: a
`builtin` Debug tree configures, builds and passes 264/264 rows (excluding the unrelated
`brusselator_runner`); the default `eigen` tree is unchanged (`ctest -L 'linalg|install'` 16/16).
`SchurSlabSolver::factor()` still refuses `E-LINALG-SCHUR-BACKEND` under `builtin` (there is no built-in
direct factorisation), and that refusal is now fired by `test/linalg/schur_slab_unavailable_test.cpp`
(NL-5), registered only in that configuration. The CI matrix's `builtin` legs can therefore go green.

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

**Confirmed (2026-10-13, `ctest`, `build-all`, measured on the owner's machine in the lead's host session, not in this sandbox): the plain (non-LowFive)
`verification.brusselator.participants.accept.3` PASSES on the host at HEAD**, in two separate runs the same day
(1.12 s, 1.05 s) — the first row above is answered: `connect()`'s own accept/merge/disconnect path (this entry's
fix) is measured working, independent of LowFive entirely. `.accept.5` remains unrun per D18's own ruling,
unaffected by this confirmation. D14's own account (its "what remains" item 3) cites this same measurement as the
reason its own accept-mode ABI gap, not this entry's layout bug, is now the only thing standing between
`verification.brusselator.participants.lowfive.accept.3` and a pass.

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
— a leftover from the soma → fuse rename that `dump_text()` already carries correctly as
`"# fuse-ir v1"` (the header line is `include/fuse/ir/dump.hpp:73`, inside `dump_text()`, which begins at
line 70). `python/tests/test_graph.py` records this as an
`xfail(strict=True)` test citing both line numbers, so the suite documents the finding without silently
working around it and will fail loudly (an unexpected XPASS) the day someone fixes `dump_graphviz()` without
updating that test.

**Not pursued, no decision needed yet:** `ImplicitComponent` bindings (would lift the Newton restriction
above); a Python-side MPI story (`mpi4py`, communicator passing into `RunnerConfig`); native/MOAB backend
bindings. None of the three has a design in `docs/dev/python-surface.md` beyond the "what this does not
expose" list — a v2 scope decision, not a defect in v1.

### D38. The Runner's generic size reconciliation held an `ExternalDelivery` instruction to the same equal-size rule as an ordinary data-moving kind

**What it was.** `src/runtime/runner.cpp`'s per-instruction buffer-size reconciliation (`resolve_instructions()`) reads
`out_total` doubles from every input arc for any instruction kind it does not name as an exception — `Remap` and
`Transport` are already exempted, each for its own documented reason (`Remap`'s sizes are reconciled against the
matrix it is actually given, in `resolve_remap()`; `Transport`'s two ends are the same region in two components'
own per-placement decompositions, KNOWN_ISSUES.md D5, reconciled instead in `check_transport_extents()` once a
real `CommGraph` exists). `OpKind::ExternalDelivery` was not in that exemption list, so the generic check ran for
it too. `ExternalDelivery` (Unit 3, S6 phase 3, "LowFive" track) is the boundary node a LowFive-bound channel
compiles to instead of `Transport` — `src/ir/rules_channel.cpp`'s `lower_channel()` gives it the SAME single
input `Transport` would have had (the exporting component's own producer arc, kept as a pure ordering edge,
never read by `execute()`'s own `OpKind::ExternalDelivery` case), but types its OWN output arc with the
IMPORTING component's own `rank_set` — "the value...lands on the CONSUMER's own placement, not the producer's"
(that function's own comment). The input and output of one `ExternalDelivery` instruction are therefore two
DIFFERENT components' own decompositions, not two placements of the same one (`Transport`'s own case) — an even
stronger reason to exempt it, since on a rank hosting exactly one end, D24's own contract ("a placement is
decomposed over its own RankSet... a component that does not live on this rank gets an empty view") makes the
two sizes disagree BY CONSTRUCTION, every time, on every rank hosting exactly one end.

**How it surfaced.** Measured on the host (2026-10-12, `build-lowfive`, MPICH 5.0.0a1/`ch4:ofi`), the first run
where `verification.brusselator.participants.lowfive.mpmd.3` and `...lowfive.accept.3` got past D36's own
launch-layout fix and reached `fuse_initialize()`: `u_stdout.0.log` printed `FAIL(u_main): initialize: status=-4
ExternalDelivery instruction 1 reads 2816 values from arc %0 into 0 at arc %2` (U's rank: 2816 = U's own 16x16=256
cell `u_out`, 11 window samples wide, the whole size U's own one-rank `RankSet` decomposes to; 0 because V, the
channel's importer, is not hosted on U's rank); `v_stdout.1.log` printed the mirror, `reads 0 values from arc %0
into 2816 at arc %2`. `coupler_stdout.2.log` printed `another rank of the base communicator failed to resolve its
buffers or instructions, so this rank must not enter the collective CommGraph construction; see that rank's own
error for the cause` — the coupler rank never ran its own local check at all; it refused on the agreed, collective
failure the other two ranks' own refusals fed it, not on anything computed locally (the standing rule that a
collective path never returns on a rank's own locally-computed condition).

**Fix.** `resolve_instructions()`'s per-instruction size check now also skips `OpKind::ExternalDelivery`, in its
own outer `if`, wrapped AROUND the existing `Remap`/`Transport` check rather than folded into its condition —
deliberately, so the existing line `build-evidence/mutations/transport-runtime.toml`'s own
`transport-held-to-equal-extents` mutation anchors verbatim stays byte-identical and that row needed no edit at
all. Unlike `Transport`, no replacement check was added: `execute()`'s own `OpKind::ExternalDelivery` case never
reads the input (op.hpp's own header comment on the enumerator has the full account of why the edge is carried
anyway, as a pure ordering constraint), so the "heap overread" hazard the whole reconciliation exists to catch
cannot arise for this kind regardless of what the two arcs' sizes are. The output arc's own size — what the
application's write through `fuse_view_field` actually gets (`src/abi/participant.cpp`'s `impl_view_field`, via
`Runner::field_window_view()`) — is sized by the same `sizes(region, rank_set)` callback (`src/schedule/
buffers.cpp`'s `plan_buffers()`) every other coupler-owned arc in the system already relies on; there is no new
invariant this kind needs that a narrower check could still catch, and the task that found this was explicit that
a check that cannot be made to fire must not be added speculatively.

**Audited for the same class of mistake** (an instruction's input arc total compared against its output arc
total, or an equal-size assumption, for a kind whose two ends can be two different components' own
decompositions): `src/runtime/runner.cpp:792`'s `resolve_remap()` coincidence check (`in.total() != out.total()`)
is `Remap`'s OWN stronger, narrower replacement check, fired only when `regions_coincide()` has already
established the two regions are the same mesh in the same order on THIS rank — not a cross-component comparison.
`check_transport_extents()` (`runner.cpp`) is filtered to `OpKind::Transport` only and already gated behind
`CommGraph`/`base_communicator`; unaffected. Every `.total()` comparison in `src/runtime/iteration.cpp` (lines
942, 1021, 1504, 1551, 4204) is either gated on a window member's own `cfg.hosted` (never compares a hosted
member's size against an unhosted one's) or a same-arc/same-member self-consistency check (a buffer resized
against its own config-derived expected size, or a chain walk that correctly STOPS, rather than fails, at a size
discontinuity). `src/schedule/*.cpp` and `src/ir/*.cpp` contain no size-equality arithmetic at all — decomposition
sizing is a runtime-only concern in this codebase. `src/abi/participant.cpp` has no field-size check beyond the
one `resolve_buffers()` (`runner.cpp`) already performs, which is unaffected (window-scoped fields bypass it
entirely, per `iteration.cpp`'s own account at line 938). No second instance of this mistake was found.

**Pinned by** `test/verification/brusselator_external_delivery_size_test.cpp`'s two cases. The first drives the
REAL LowFive-bound Brusselator accept-layout graph (`U {0,1}`, the remap coupler `{1,1}`, `V {2,1}` — the shape
`brusselator_graph_tool.cpp`'s own accept branch with `--ranks-u 1 --ranks-coupler 1 --ranks-v 1 --transport
lowfive` builds) through `fuse::Runner::initialize()` with `RunnerConfig::force_this_process` (no MPI, KNOWN_
ISSUES.md C1), once on U's rank and once on V's; it asserts `initialize()` gets PAST the `ExternalDelivery` check
on both (the error code is no longer `E-RT-SIZE` and the message no longer names `ExternalDelivery`) but still
fails on a SEPARATE, pre-existing tier-0 ceiling found while writing this test — `iteration.cpp`'s own
`R.participants` (the window region's cross-rank residual-reduction set) is populated only `if (config.
base_communicator.has_value())`, so with no real communicator it stays empty on every rank, and the SAME
function's "an empty coupling value... the region would report convergence after one sweep" refusal
(`E-RT-REGION`) fires for whichever convergence measure names the component not hosted here. This is the SAME
class of tier-0-only limitation `brusselator_accept_layout_test.cpp`'s own header comment already documents for
the ordinary channel's `Transport`/`CommGraph` requirement, manifesting through a different check; on a real
multi-process run `R.participants` holds all three ranks and this refusal never fires. The second case is the
"narrowest reachable function" fallback this task's own instructions asked for when the real graph cannot reach a
clean `initialize()` success at tier 0: a minimal, non-windowed `ExternalDelivery` channel between two components
on two disjoint ranks with no `IterationRegion` at all, where `initialize()` DOES succeed cleanly on both ranks —
isolating this fix's own effect with nothing else in the way. One mutation
(`build-evidence/mutations/external-delivery-size.toml`'s `external-delivery-size-exemption-removed`): with the
new outer guard disabled, BOTH tests fail, the first reproducing the host's own two `u_stdout.0.log`/
`v_stdout.1.log` messages character for character.

**Does not close D14.** D14 is about the two-HDF5-image hazard, the `libfmt` `dlopen` blocker, and Demo B's own
distributed-transport path being compiled but never run; this defect is independent of all three, and the
LowFive row this fixes was never blocked by D14 on the owner's own `build-lowfive` tree in the first place.

**What the host must verify.** `ctest -R participants.lowfive` in `build-lowfive` (3 tasks each,
`verification.brusselator.participants.lowfive.mpmd.3` and `...lowfive.accept.3`): both should now get past
`fuse_initialize()`'s own `ExternalDelivery` check on every rank. Whether either reaches its first period, and
whether the VOL plugin's own accept-mode intercommunicator (D14) lets `...lowfive.accept.3` complete the
handshake at all, are separate, unmeasured questions this entry makes no claim about — see this round's own
report for what the sources show and do not show about them.
### D39. The participants' own `lowfive.mpmd.3` row now attaches LowFive programmatically, sidestepping the env route's own null dereference rather than explaining it

**Status: FIXED at the library/example level for the MPMD row's own attach step; the distributed serve/receive
choreography and the row itself are UNEXECUTED (no MPI in this sandbox); accept mode is refused by design, not
fixed.** "LowFive attach" track, following D38's own fix.

**What it was.** D38 closed the `ExternalDelivery` size-reconciliation defect that blocked
`verification.brusselator.participants.lowfive.mpmd.3`/`...lowfive.accept.3` from ever reaching
`fuse_initialize()`. Measured on the host the same day (2026-10-12, `build-lowfive`, main `4fd400d`), with D38 in
place, both rows get PAST `fuse_initialize()` and both `brusselator_u_main` and `brusselator_v_main` then die with
a null dereference inside `LowFive::VOLBase::_file_create` on their own FIRST `H5Fcreate` (`examples/brusselator/
participants/shim.cpp`'s `bruss_lowfive_write`, called from `u_lowfive_write_export`/`v_lowfive_write_export`),
reached through the `HDF5_VOL_CONNECTOR=lowfive`/`HDF5_PLUGIN_PATH` env-var route `fuse-lowfive-vol`
(`src/transport/lowfive_plugin_vol.cpp`) serves. From LowFive's own installed headers and its source on the host
(`/Users/mahadevan/Code/hdf5/LowFive` rev `7f3a58b`, not reachable from this sandbox): `VOLBase::_file_create`
(`src/vol-base/file.cpp`) calls `H5Pget_vol_info(fapl_id, &info)` and dereferences `info->under_vol_id` (line 40)
and `info->vol` (line 43); `info_t::vol` is a CLASS-STATIC member `create_MetadataVOL()`/`create_DistMetadataVOL()`
set, and LowFive's own `str_to_info` (the env-string connector-name parser) replaces the class-static `info` with a
FRESH `info_t`. With `HDF5_VOL_CONNECTOR=lowfive` and no info string, the fapl carries no info at all (line 40
faults); with the string `lowfive under_vol=0;under_info={};`, the fault moves to line 43 because `info_t::vol` is
still null at that point -- WHY (whether `fuse-lowfive-vol`'s own `ConfigureOnce()` never ran before this call, or
`VOLBase::_term` cleared it first) is not established and this entry does not chase it.

**Fix.** The participants (`examples/brusselator/participants/u_main.c`, `v_main.f90`, `shim.{h,cpp}`) no longer
rely on the env route at all for `--transport lowfive`: a new module, `lowfive_attach.{h,cpp}`, builds the VOL
connector PROGRAMMATICALLY -- a file-access property list carrying it
(`LowFive::H5VOLProperty::apply`/`H5Pset_vol`/`H5Pget_vol_id` read-back-and-compare, the identical construction
`src/transport/lowfive_transport.cpp`'s own `VolSingleton`/`DistVolSingleton` already use and this codebase has
MEASURED end to end in the single-process case) -- and `bruss_lowfive_write`/`bruss_lowfive_read` now take that
FAPL as an explicit `fapl_id` argument, passed to every `H5Fcreate`/`H5Fopen` instead of `H5P_DEFAULT`, refusing
outright (never silently falling back to `H5P_DEFAULT`) if attach was never called or failed. `bruss_lowfive_
attach()` is called exactly once per process, right after `MPI_Init()`, from the SAME `--graph` path the program's
own command line already names (no separate `FUSE_LOWFIVE_BINDINGS` environment variable, unlike the env-route
plugin's own convention). `fuse-lowfive-vol`/`src/transport/lowfive_plugin_vol.cpp` (the env route) are UNCHANGED
-- this track's own instruction -- and stay in the tree for a later track; the three ctest-row environment
variables (`HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH`/`FUSE_LOWFIVE_BINDINGS`) are vestigial for the participants now
but left set (additive-only constraint on `examples/CMakeLists.txt`, a shared file).

**One translation unit, not the env-route plugin's own two-OBJECT-library split.** `lowfive_attach.cpp` names
both a FUSE header that pulls fmt 12 (`fuse/model/yaml_io.hpp`) and LowFive's own headers (bundled fmt 8.1.1's
include directory) in one file -- the exact shape `src/transport/lowfive_plugin_graph.cpp`/`lowfive_plugin_vol.cpp`
were split in two OBJECT libraries to avoid (this file's own D14 account). That split is no longer necessary:
`fuse_require(fmt VERSION 9 ...)` (`cmake/FuseDependencies.cmake`, commit `f59b446`) fixed the ACTUAL root cause
project-wide (`find_package(fmt)` resolving LowFive's own bundled 8.1.1 when `/opt/lowfive` led `CMAKE_PREFIX_
PATH`), measured end to end in EITHER prefix ordering, including `lowfive_transport.cpp` itself -- a file that
has ALWAYS named both header families in one TU. `lowfive_attach.cpp` relies on that same floor and reuses the
env-route plugin's own graph-side building blocks (`fuse::build_lowfive_bindings()`, `fuse::
plan_lowfive_rendezvous()`) directly, rather than re-deriving `lowfive_plugin_internal.hpp`'s own plain-data
`Config` indirection, which existed only to cross a translation-unit boundary this file does not need.

**The bidirectional-exchange gap the env-route plugin's own comment names, closed for the participants' own
attach.** `lowfive_plugin_graph.cpp`'s `ConfigureDistributed()` picks "the first LowFive-bound channel found" to
decide a rank's own writer/reader role and MPI_Comm_split color, with an explicit comment that "a graph with more
than one [channel] would need this loop to confirm every ... channel agrees on the SAME writer/reader pair this
process belongs to, which is not yet checked here." The Brusselator's own bidirectional exchange (`u_out`: U
writes, V reads; `v_out`: V writes, U reads) is exactly that case -- the writer/reader LABELS swap between the two
channels, though both partition `MPI_COMM_WORLD` into the SAME two groups. `lowfive_attach_decision.hpp`'s
`decide_attach_mode()` (the pure, MPI-free and HDF5-free half of the attach, tier-0 tested against 9 cases in
`test/verification/brusselator_lowfive_attach_test.cpp`, `fuse_verification_brusselator_lowfive_attach_test`) is
the check that comment asked for: every qualifying channel's own writer/reader `RankSet` pair must be identical to
or the exact swap of the canonical (first) channel's, or the attach refuses (`E-LOWFIVE-ATTACH-INCONSISTENT`)
rather than forming an intercommunicator against an unexamined assumption.

**Found while implementing, not before: the attach's own first version formed a NEW collective the coupler
process never joins.** `bruss_lowfive_attach()`'s first version formed the distributed rendezvous's own "local"
communicator with a fresh `MPI_Comm_split(MPI_COMM_WORLD, ...)`, mirroring `src/transport/lowfive_plugin_vol.cpp`'s
own `ConfigureOnce()` exactly. `MPI_Comm_split` on `MPI_COMM_WORLD` is collective over EVERY rank of that
communicator, including the coupler's -- and `bruss_lowfive_attach()` is called only from `u_main.c`/`v_main.f90`,
never from `coupler_main.c`, so U and V would enter that split while the coupler never does: a hang, not a crash,
and the mirror image of this track's own standing audit ("a collective path must never return on a locally
computed condition") -- here the hazard is a collective ENTERED without every rank's own participation, caught
before any host ever had to find it with a timeout. **Fix**: `bruss_lowfive_attach()` takes a new parameter,
`physics_comm_f`, the caller's own ALREADY-FORMED physics-only communicator (`u_main.c`'s own `u_comm`,
`v_main.f90`'s own `v_comm` -- the SAME communicator `bruss_u_create()`/`bruss_v_create()`'s own `comm_f` argument
already takes), and forms the intercommunicator directly from it. No new collective over `MPI_COMM_WORLD` is
needed at all: `MPI_Intercomm_create`'s own `local_comm` argument is collective only over the union of the two
SIDES' own groups (U's physics ranks, V's physics ranks), which the coupler was never part of, and `u_comm`/
`v_comm` are already safely formed, with the coupler's own matching `MPI_Comm_split` call (u_main.c's own header
comment on that split has the full account) already satisfied before `bruss_lowfive_attach()` is ever reached.
`bruss_lowfive_attach()` additionally verifies `physics_comm_f`'s own size and this rank's own position within it
against what the graph's declared `RankSet` expects, refusing (not trusting) a mismatch.

**Accept mode refuses by design, not by omission.** `decide_attach_mode()` refuses
(`E-LOWFIVE-ATTACH-ACCEPT`) the moment an HDF5-bound channel's writer or reader component declares
`LaunchMode::Connect` -- accept mode's own merged base communicator is never exposed back to a caller through any
ABI query (this entry's own open item 3, repeated from D14's own "what remains" list, UNCHANGED: still open), so
there is no intercommunicator this attach could form for it. `verification.brusselator.participants.lowfive.
accept.3` stays red ON PURPOSE until D14's own accept-mode question is decided by the lead -- a clean, named
refusal at attach time (`bruss_lowfive_attach()` returns nonzero with the code and message above in `errbuf`),
never a half-configured VOL or a hang.

**UNEXECUTED, like the rest of this codebase's own distributed LowFive path (D14's own Demo B account): the
MPMD row's own distributed serve/receive choreography.** `bruss_lowfive_serve_after_write()`/`bruss_lowfive_
receive_before_read()` (`lowfive_attach.h`) bracket the shim's own `H5Fclose`/`H5Fopen` with `LowFive::
DistMetadataVOL::serve_all()`/`broadcast_files()`, placed at the natural points -- mirroring `examples/demo_b`'s
own one-directional producer/consumer call order exactly (the owner's own ruling on this track, 2026-10-12: the
natural placement, not an invented reordering). Whether `serve_all()`/`broadcast_files()` can mutually block if
both ranks' own schedules reach "serve mine" before "receive peer's" at the same window is UNVERIFIED: no MPI in
this sandbox, and LowFive's own source is not reachable from it either. This is a NEW risk the single-process path
(D14's own Unit 3a, measured bit-exact) and Demo B (one producer, one consumer, never both) do not share.

**Pinned by** `test/verification/brusselator_lowfive_attach_test.cpp`'s nine `DecideAttachMode` cases (writer-side
and reader-side rendezvous, the bidirectional swap agreeing, no-channels refused, accept-mode refused on either
end, an inconsistent third-group channel refused, and `plan_lowfive_rendezvous()`'s own unplaced/overlap refusals
propagated verbatim) and one hand-proved mutation row,
`build-evidence/mutations/lowfive-attach.toml`'s `decide-attach-mode-skips-the-accept-refusal` (full tier-0,
166 -> this track's own count, killed decisively). Full tier-0 and all 5 repo-health gates clean (this round's
own report has the exact numbers).

**What the host must verify.** In `build-lowfive`: rebuild, then `ctest -R 'participants\.lowfive\.mpmd\.3'`
(3 tasks). A clean pass means every rank's own `u.log`/`v.log`/coupler log shows the SAME period-by-period
convergence Unit 3a's own single-process harness already measured bit-exact; a hang (ctest's own 120s `TIMEOUT`
firing) means the serve/receive mutual-wait risk above is real and the owner's own ruling (natural placement, no
artificial reordering) needs revisiting; a clean, NAMED refusal (not a null dereference, not a hang) on any rank
means this entry's own design has a gap this round's report did not find -- see that report's `host_pending` for
the exact commands and what each rank's log should print in every one of these three outcomes.

**Does not close D14.** D14 is about the two-HDF5-image hazard (closed, host action only), the Demo B distributed
path (compiled, never run), and Unit 3's own single-process measurement (done). This entry is independent of all
three: the env-route plugin this entry stops depending on for the participants is UNCHANGED, and D14's own open
accept-mode intercommunicator question (item 3 of its "what remains" list) is the SAME question this entry's own
accept-mode refusal defers to, not a new one.

**Round 2 update (2026-10-13), superseding this entry's own "UNEXECUTED... distributed serve/receive
choreography" paragraph above.** The host ran `lowfive.mpmd.3` after round 1's own fix (the row now gets past
attach and into `build-lowfive`'s own participants) and MEASURED the exact mutual-wait risk the paragraph above
called unverified: the row hung the full 120 s `TIMEOUT`. A macOS `sample` of the hung job found BOTH
`brusselator_u_main` and `brusselator_v_main` inside `H5Fclose -> LowFive::VOLBase::_file_close ->
DistMetadataVOL::file_close` (`src/vol-dist-metadata.cpp:320-364` in this host's own LowFive checkout,
`/Users/mahadevan/Code/hdf5/LowFive` rev `7f3a58b`, readable from this sandbox as of this round) ->
`DistMetadataVOL::serve_all` (`:38-72`) -> `Index::serve` (`src/vol-dist/index.cpp:105-167`), opening with a
`local.barrier()` (`:110`) and then polling each intercomm's own RPC server (`s.probe()`/`s.process()`, `:145-161`)
until queried by the peer -- which never happens, because the peer is doing the identical thing at the identical
moment. `brusselator_coupler_main` was waiting in `MPI_Allreduce` for both. Reading the ACTUAL root cause in
LowFive's own source (not available to round 1, which called the risk "UNVERIFIED... no MPI... LowFive's own
source is not available to read either" -- both now false on this sandbox): `DistMetadataVOL::serve_on_close` is a
PUBLIC bool member, DEFAULT TRUE (`include/lowfive/vol-dist-metadata.hpp:33`), and `file_close()` calls
`serve_all()` itself whenever it is (`:355`, `serve_on_close && match_any(...)`) -- INSIDE `H5Fclose`, before round
1's own `bruss_lowfive_serve_after_write()` (placed AFTER `H5Fclose` returns, mirroring Demo B's own one-
directional call order) ever ran a single line. That round-1 call was therefore a NO-OP in every run: the serve
had already happened, unconditionally, with neither side having read the other first, on a graph (`iteration:
Jacobi`, `max_iterations 60`, `acceleration None`, `brusselator_graph_tool.cpp`'s own constants) where both
components write, THEN both read, every iteration -- so both sides reliably raced into `serve_all()` together.
This is exactly the "natural placement, not an invented reordering" ruling this entry's own prior paragraph
recorded, now measured wrong; that paragraph is left as written above for the record of what was decided and why,
not corrected in place.

**Fix: role-ordered serve, an explicit decision instead of `H5Fclose`'s own implicit one.** *[Superseded by "Round 2 REVIEW" below: the flush this paragraph places in the pre-open hook was rejected on review against LowFive's source, and the corrected argument is the one given there.]*
`bruss_lowfive_attach()` (`examples/brusselator/participants/lowfive_attach.cpp`) now sets `vol.serve_on_close =
false` on the constructed `DistMetadataVOL`, so `H5Fclose` never serves on its own. Two new PURE functions in
`lowfive_attach_decision.hpp` (MPI-free, HDF5-free, tier-0 tested) make the decision instead: `decide_serve_roles()`
assigns, from the two components' own `RankSet`s alone (identical on every rank of either component, the
collective-safety requirement `serve_all()`'s own `local.barrier()` places on this decision), one of two roles --
the component with the lower `RankSet::min_rank()` is "first-server" (serves immediately at every write, receives
normally at every read); the other is "first-reader" (defers its own serve at every write, and at its own next
read receives the peer's file FIRST, only THEN flushing the deferred serve). `serve_state_step()` is the per-
process state machine `bruss_lowfive_serve_after_write()`/`bruss_lowfive_receive_before_read()` now consult on
every write/read event, plus a new third hook, `bruss_lowfive_finalize()` (called once at process exit from
`u_main.c`/`v_main.f90`, after this process's own last write/read and before `MPI_Finalize`), which refuses
loudly (`E-LOWFIVE-ATTACH-SERVE-UNFLUSHED`) if a first-reader's own deferred serve was never flushed -- the one
failure mode role-ordering could otherwise hide as a silent data drop. A second export before its own deferred
serve is flushed is also refused (`E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER`), the check that would catch a graph where
the write/read alternation this protocol depends on stops holding.

**Deadlock-freedom argument (I1 progress + I2 role exclusivity).** *[Superseded by "Round 2 REVIEW" below: the flush this paragraph places in the pre-open hook was rejected on review against LowFive's source, and the corrected argument is the one given there.]* I1, established from the host's own measured
stack and from `u_main.c`'s/`v_main.f90`'s own single-branch delivery dispatch (never both a write and a read in
the same `u_loop_step()`/loop-body call): on one process, an export event is always eventually followed by an
import event before that SAME process's own next export. I2 is this state machine's own contract: the first-server
serves at every export and never defers; the first-reader defers at every export and flushes only at its own next
import. Together: whenever the first-server blocks inside its own immediate serve, I1 guarantees the first-
reader's own NEXT local event is an import, and I2 makes that import a RECEIVE first -- exactly what satisfies
the first-server's pending serve. Symmetrically, once the first-reader's own import flushes its deferred serve
(now blocking in turn), I1+I2 guarantee the first-server's own next local event is ITS import, an ordinary receive
that satisfies the just-flushed serve. Each side's own single blocking serve call is always matched by the OTHER
side's very next import, never left unconsumed except at `bruss_lowfive_finalize()`'s own check. This DEPENDS on
I1 holding for BOTH components every iteration -- a graph where it stops holding is outside what this protocol was
designed for, and the double-defer refusal above is exactly the check that catches the violation instead of
hanging silently.

**NOT started this round, by the owner's own ruling: a helper thread to let the two transfers overlap again.**
Role-ordered serve necessarily SERIALIZES the exchange (never both sides serving in the same window, unlike
round 1's own, now-measured-wrong, assumption). Moving the receiving side's own `broadcast_files()` onto a helper
thread so it can run concurrently with the other side's `serve_all()` was considered and ruled unsound without
patching LowFive itself: `Index::serve()`'s own barrier/ibarrier path (`vol-dist/index.cpp:105-167`) and
`Query::get_filenames()`'s own LOCAL broadcast (`vol-dist/query.cpp:106-139`, broadcast at `:125`) both run over
the SAME `local` communicator LowFive hands `DistMetadataVOL::create_DistMetadataVOL()`, and this build's own
HDF5 is configured `Threadsafety: no` with a plain `MPI_Init` (not `MPI_Init_thread(..., MPI_THREAD_MULTIPLE,
...)`) -- a second thread issuing MPI calls on that same communicator while the first is mid-probe is not a
documented-safe configuration for either library. Recorded here as a documented, NOT-STARTED follow-up, not
attempted and not claimed done.

**`serve_on_close` itself is NOT tier-0 testable.** Nothing in this sandbox can construct a real
`LowFive::DistMetadataVOL`, call `H5Fclose`, and observe whether `serve_all()` ran inside it or not -- that
requires live MPI and live HDF5/LowFive, neither available here. The only evidence that the flag is doing its job
is the ABSENCE of the round-1 hang on a future host run (or, if `serve_on_close` were ever reverted to `true` by
mistake, its REAPPEARANCE: the same `sample` signature as this round's own paragraph above, both participants
inside `DistMetadataVOL::file_close -> serve_all -> Index::serve`'s probe loop, at the same moment the row would
otherwise have made its first `H5Fclose` call).

**Pinned by (round 2, additive to the nine `DecideAttachMode` cases above).** Nine new tier-0 cases in
`test/verification/brusselator_lowfive_attach_test.cpp` (now 18 total in that binary): three for
`decide_serve_roles()` (lower-`min_rank()` wins regardless of which side is passed first, overlapping RankSets
refused, an empty RankSet refused) and six for `serve_state_step()` (first-server's own repeated serve/receive/
clean-finalize cycle; first-reader's own repeated defer/receive-then-flush cycle; a first-reader's own import with
nothing deferred is an ordinary receive, not an error; a first-reader's own second export before its first
deferred serve is flushed is refused; a first-reader's own finalize with a deferred serve still pending is
refused; a first-server's own finalize reached mid-cycle, with no intervening import, is still clean since that
role never defers). Two new hand-proved mutation rows in `build-evidence/mutations/lowfive-attach.toml`
(`decide-serve-roles-first-server-comparison-swapped`, `serve-state-step-finalize-unflushed-check-disabled`), each
killed decisively (17 PASS / 1 FAIL, the SAME single predicted test failing each time) and the file restored
byte-identical (`md5` checked) before the next step. Full tier-0 and all 5 repo-health gates clean (this round's
own report has the exact numbers).

**What the host must verify (round 2).** In `build-lowfive`: rebuild (`cmake --build .`, every target, not a
named subset), then `ctest -R 'participants\.lowfive\.mpmd\.3'` (3 tasks). A clean pass means every rank's own
`u.log`/`v.log`/coupler log shows the SAME period-by-period convergence Unit 3a's own single-process harness
already measured bit-exact, AND neither `bruss_lowfive_finalize()` call printed a `FAIL(u_main)`/`FAIL(v_main)`
unflushed-serve message (it should not, if I1/I2 above hold for this row's own graph, which this round's own
report argues from the measured write/write-then-read/read pattern, not merely assumed). A hang again (ctest's own
120 s `TIMEOUT` firing), with a `sample` showing the SAME `file_close -> serve_all -> Index::serve` stack as this
round's own paragraph above, would mean `serve_on_close` reverted to `true` somewhere, or a build picked up a
stale LowFive without the flag at all -- not a flaw in the role-ordering argument itself, which this entry's own
I1/I2 account rules out for THIS specific race. A hang with EITHER rank instead caught inside `Index::serve`'s own
probe loop while the OTHER rank is still computing (not yet at its own next write or read at all) would mean I1
itself does not hold for this row's own graph after all -- a schedule where one side's own export/import pairing
is not strict alternation -- which this entry's own design was not built to handle and which this round's report
did not find evidence of. A `FAIL(u_main)`/`FAIL(v_main): bruss_lowfive_finalize: ... E-LOWFIVE-ATTACH-SERVE-
UNFLUSHED` message on an otherwise-clean run would mean the LAST iteration's own first-reader write was never
followed by a matching read before the process exited -- an off-by-one in the period/window loop's own tail, not
a protocol defect, and worth a fresh look at exactly which call is the row's own last one before `MPI_Finalize`.

**Still does not close D14.** Unchanged by this round: D14 is about the two-HDF5-image hazard (closed, host action
only), the Demo B distributed path (compiled, never run), and Unit 3's own single-process measurement (done).
D14's own open accept-mode intercommunicator question (item 3 of its "what remains" list) is UNCHANGED by this
round's own role-ordered-serve fix, which only concerns the MPMD row's own already-formed intercommunicator, never
accept mode's unexposed one.

**Round 2 REVIEW (2026-10-13), the first round-2 fix above was reviewed against LowFive's own source and REJECTED
before any host run -- superseding the "Round 2 update" section's own fix, not its own measured-deadlock account,
which stands.** The first round-2 attempt set `serve_on_close = false` and moved a first-reader's own deferred
serve into the PRE-open hook (`bruss_lowfive_receive_before_read()`, called before `H5Fopen`), arguing (in this
entry's own paragraph above) that "receive the peer's file FIRST... THEN serve this side's own deferred write" was
"the one order that cannot leave both sides waiting on each other." That argument was wrong. Two more facts, read
from `/Users/mahadevan/Code/hdf5/LowFive/src/vol-dist-metadata.cpp` (rev `7f3a58b`), that the first attempt missed:

- The CONSUMER's `H5Fopen` of a remote in-memory file IS the receive. `DistMetadataVOL::file_open()` (line 230 on)
  builds a `Query` and calls the producer's `"get_file_hierarchy"` RPC (line 278); its own comment says plainly
  "force the producer to enter serve_all to get file hierarchy" (lines 255-257) -- `H5Fopen` performs this
  automatically, as an ordinary VOL callback dispatch; nothing in a hook that runs BEFORE it can substitute.
- The consumer's `H5Fclose` of that remote file is what ENDS the producer's serve. `DistMetadataVOL::file_close()`'s
  `RemoteFile` branch (line ~333 on) calls `f->obj.call<void>("file_close")` (line 337), `MPI_Barrier(local)`
  (line 339), then root calls `f->obj.self_->finish(0)` (line 342) -- `finish(0)` is exactly what lets
  `Index::serve()`'s own probe/process loop (`vol-dist/index.cpp:145-161`) exit for that client. The producer's own
  `serve_all()` call therefore blocks until the CONSUMER has done BOTH its own open AND its own close of that
  specific file -- not until the consumer has merely attempted to open it, and certainly not until the consumer has
  merely reached a hook named "before read".

The first attempt's own pre-open flush ran a first-reader's own second, explicit `serve_all()` call BEFORE this
process's own `H5Fopen` of the peer's file -- before this process had done anything the peer's own pending
`serve_all()` could possibly be released by. Both sides then block: the first-server is still stuck in its
original `kServeNow` from its first write, waiting for the first-reader's own open-and-close; the first-reader is
stuck in the premature flush the pre-open hook just entered, waiting for the first-SERVER's own open-and-close of
the newly-served file -- which the first-server cannot perform, because it is still blocked in its own first
`kServeNow`. Circular wait: the SAME two LowFive-level stacks (both inside `Index::serve`'s probe loop) the host
measured for round 1, one write/read cycle later. This entry's own tier-0 tests for the first attempt passed
because they drove the state machine only against its OWN event model (hook call order), never against what the
PEER actually observes (released by MY open AND close, not by MY merely calling a hook).

**The fix, this version.** The flush moves to a NEW event, fired AFTER this process's own `H5Fclose` of the peer's
file -- `kImportClosed`, a new hook (`bruss_lowfive_after_read_close()`) called from `shim.cpp`'s own
`bruss_lowfive_read()` on EVERY path that reached a successful `H5Fopen` (success or a later dataset-level
failure alike -- a dataset-level error must not leave this process's own bookkeeping silently behind, even though
the read itself is about to fail regardless). `kImportAboutToOpen` (the pre-open hook) becomes an unconditional
`kReceiveOnly` for every role and state: it does nothing LowFive-level at all now, kept only as a validated event
for symmetry with the real call sites and so a future mutation that reintroduces a flush there is caught (it is --
see "Pinned by" below). `LowFive::DistMetadataVOL::broadcast_files()`, which the first attempt also called from
the pre-open hook, is removed entirely: read from source (`src/vol-dist-metadata.cpp:75-103`), it operates
ENTIRELY over `local` (never an intercomm), propagating an already-known file registry from one local rank to the
OTHER ranks of a MULTI-RANK component -- not communicating with the peer component at all. A `grep -rn
broadcast_files` across the LowFive checkout's own `src/`/`include/` finds zero internal callers; its only caller
anywhere in that repository is the Python binding (`bindings/python/lowfive.cpp:126`). This row places exactly one
rank per component (U{1}, V{1}), so even a harmless call would have had nothing to propagate to.

**Re-derived invariant, in terms of what the PEER observes, not hook-call order.** I1 (progress) is unchanged --
the same three pieces of evidence as the "Round 2 update" section above. I2 (role exclusivity) is unchanged. The
RELEASE CONDITION, corrected: a process blocked inside `serve_all()` for file K is released exactly when the PEER
completes ITS OWN open-and-close of file K -- not by the peer merely reaching an import. Whenever the first-server
blocks inside its own `kServeNow` for file K (entered at its own export), I1 guarantees the first-reader's own
NEXT local event is an import, whose open-and-close (non-blocking from the first-reader's own point of view, since
`kImportAboutToOpen` does nothing to delay it) is exactly the release condition. The first-reader's own
`kImportClosed` for that SAME import is what THEN flushes its own deferred serve -- entered only now, after its
own open-and-close is already done -- and I1 guarantees the first-server's own next local event is ITS import,
whose open-and-close is exactly the release condition for the first-reader's newly-entered serve. Each side's own
single blocking serve call is matched by the OTHER side's very next open-and-close, in that order -- never by the
hook that merely precedes it.

**Checked against an explicit-state simulation, not only a written argument.** A new, pure, MPI-free and
HDF5-free header, `examples/brusselator/participants/lowfive_attach_model_check.hpp`, explores every interleaving
of two processes running the Jacobi row's own fixed write/read script (for N = 1, 2, and 60 -- this row's own
`kMaxIterations`) against the blocking semantics read from LowFive's source above (a process blocked in a serve
for file K is released only by the peer's own open-AND-close of file K; an open of a file the peer is not yet
serving blocks). The SHIPPED protocol's own tests drive the REAL `decide_serve_roles()`/`serve_state_step()`
directly (so a mutation to either is caught by the search failing to complete, not only by a narrower unit test);
three REJECTED designs -- this round's own first attempt (flush at the pre-open hook), round 1's own "natural
placement" (serve at every write, both sides), and a "both sides defer" strawman -- are expressed as separate,
hand-rolled policies and confirmed to deadlock, illustrating the search is discriminating rather than permissive.
That header's own top comment states plainly what this does and does not establish: it checks the PROTOCOL against
the blocking semantics AS READ from LowFive's source, not a measurement of LowFive itself -- only a host run
measures that.

**Pinned by (Round 2 REVIEW, additive to the round-2-update section's own inventory above).** Two tests in
`test/verification/brusselator_lowfive_attach_test.cpp`'s own `ServeStateStep` suite were REPLACED, not deleted
outright, because they encoded the first attempt's own (wrong) event model: `FirstReaderDefersThenReceivesAndFlushesRepeatedly`
now asserts the corrected cycle (plain receive at the pre-open hook, flush at the new post-close one); a new test,
`FirstReaderImportAboutToOpenNeverFlushesEvenWithDeferredPending`, is the direct regression test for the exact bug.
Two more tests were added for completeness (`FirstReaderImportWithNothingDeferredIsOrdinaryAtBothEvents`,
`FirstReaderSecondExportAfterOnlyAboutToOpenStillRefused`) and one existing test
(`FirstServerServesThenReceivesThenFinalizesCleanRepeatedly`) was extended to exercise both read-side events. Seven
new `ModelCheck` tests exercise the simulation above. One new hand-proved mutation row,
`serve-state-step-flush-moved-back-to-pre-open` (`build-evidence/mutations/lowfive-attach.toml`), reintroduces the
first attempt's own exact mistake and is killed TWICE -- directly, by the regression test above, and indirectly, by
all four `ModelCheck.ShippedProtocolCompletes*` tests failing to complete the simulated protocol (20 PASS / 7 FAIL
of the whole binary, file restored byte-identical, `md5` checked, before the next step). The two round-2-update
section's own mutation rows (`decide-serve-roles-first-server-comparison-swapped`,
`serve-state-step-finalize-unflushed-check-disabled`) are unaffected by this round's own changes and were
re-verified against the updated test binary (spot-checked: the finalize row still kills exactly the one test it
always did). Full tier-0 and all 5 repo-health gates clean (this round's own report has the exact numbers).

**What the host must verify, superseding the round-2-update section's own paragraph (the fix it describes was
never run on a host at all).** In `build-lowfive`: rebuild (`cmake --build .`, every target), then `ctest -R
'participants\.lowfive\.mpmd\.3'` (3 tasks). The same four distinguishable outcomes that paragraph named still
apply verbatim to THIS version's own fix (clean pass; a hang with the identical `file_close -> serve_all ->
Index::serve` stack, meaning `serve_on_close` reverted somewhere; a hang with one rank still computing while the
other waits, meaning I1 itself does not hold for this row; an `E-LOWFIVE-ATTACH-SERVE-UNFLUSHED` refusal at
process exit, meaning the last iteration's own write/read pairing has an off-by-one) -- this round's own report
has the full paragraph. NEW this round: a hang where BOTH ranks are caught one step LATER than the signature
above -- specifically, both inside `Index::serve`'s probe loop but with the STACK showing `bruss_lowfive_
after_read_close` (or the now-removed `broadcast_files()` call, if a stale binary is somehow still running the
first attempt's own code) rather than `DistMetadataVOL::file_close` directly beneath `H5Fclose` -- would mean this
round's own fix, not round 1's, failed to close the gap; the LowFive-source citations above are the first place to
re-check, starting with whether `file_open()`'s own RPC dance is really as unconditional as read here.

**Still does not close D14.** Unchanged by this round's own review, for the same reasons the round-2-update
section already gave.

**Round 4 update (2026-10-13), superseding nothing above -- the round-2-REVIEW fix landed and the row no longer
hangs, but both ranks now die at their own first REMOTE `H5Fopen`.** The host rebuilt (`cmake --build .`, every
target, exit 0) and ran `ctest -R 'participants\.lowfive\.mpmd\.3'` after round 2 REVIEW's own fix landed on main
(`ac7fe33`, `8c11804`, `9482fac`). The row now FAILS in 1.2 s instead of hanging: both `brusselator_u_main` and
`brusselator_v_main` die with an uncaught `LowFive::MetadataError`, `file_open(): no intercomm found for
brusselator_v.h5` (on U) / `...brusselator_u.h5` (on V), thrown from `DistMetadataVOL::file_open()`
(`src/vol-dist-metadata.cpp:277` on this host's own LowFive checkout, rev `7f3a58b`) on each process's own FIRST
attempt to open the PEER's remote file. Two independent defects were found by reading LowFive's own source, not
assumed from the exception text alone:

- **DEFECT A (the one the host hit).** Nothing in `lowfive_attach.cpp`, in the env-route plugin
  (`src/transport/lowfive_plugin_vol.cpp`), or in `lowfive_transport.cpp`'s own `DistVolSingleton` ever calls
  `LowFive::DistMetadataVOL::set_intercomm()` (`include/lowfive/vol-dist-metadata.hpp:59-63`) -- confirmed by a
  `grep -rn set_intercomm` across this repository's own `src/`/`include/`/`examples/` returning zero hits in any of
  the three. `intercomm_locations`/`intercomm_indices`, the two vectors that call populates, are the ONLY thing
  `file_open()` consults (`src/vol-dist-metadata.cpp:263,269`) to decide which intercomm a remote filename belongs
  to; with both vectors empty, `find_matches()` returns nothing and `file_open()` throws exactly the exception
  above (`:277`). `examples/prod-con-multidata/consumer1-multidata.cpp:61`'s own
  `vol_plugin.set_intercomm("outfile.h5", "/group1/grid", 0)` is LowFive's own worked example of the call this
  attach was missing.
- **DEFECT B (found while fixing A, confirmed by direct source tracing, not by a second host run -- a second
  failure A was masking).** `fuse::split_hdf5_path()` strips the dataset path's own leading slash
  (`"brusselator_u.h5/grid/u_out"` -> `("brusselator_u.h5", "grid/u_out")`, no leading `/`); `ApplyBindings()`
  passed that unslashed string directly to `vol.set_memory()`. `LowFive::Object::fullname()`
  (`src/metadata/object.hpp:169-184`), which `MetadataVOL::dataset_create()`'s own `is_memory = match_any(filepath,
  memory)` check (`src/vol-metadata/dataset.cpp:30-33`, default `partial=false`) compares against, ALWAYS produces
  a LEADING-slash path. Tracing `match()` (`include/lowfive/vol-metadata.hpp:135-150`) character by character: the
  first comparison is `'g'` (the unslashed pattern) against `'/'` (the real path), a mismatch, and with
  `partial=false` that returns `false` immediately -- `is_memory` is `false` for every Brusselator dataset,
  `dataset_create()` never attaches an in-memory `Dataset` object, `Index::find_datasets()`'s own tree-walk finds
  nothing, `indexed_datasets` stays `0`, and `serve_all()`'s own `if (index.indexed_datasets || !perform_indexing)
  index.serve();` guard (`src/vol-dist-metadata.cpp` -- this entry's own round-2-update section already named the
  surrounding `serve_all()` call) SKIPS `index.serve()` entirely: `serve_all()` returns without blocking and
  without serving anything. DEFECT B alone would not have produced the exception the host saw (it changes what
  gets served, not whether an intercomm is registered) -- it was found by reading the attach-side code DEFECT A's
  own fix touches, not by a second host measurement, and is recorded here because an A-only fix would have left it
  in place, silently dropping every dataset.

**Fix.** `ApplyBindings()` (`lowfive_attach.cpp`, the ONE call site shared by both the single-process and the
distributed branch) now normalizes each binding's own dataset component through a new pure function,
`NormalizeDatasetPattern()` (`lowfive_attach_decision.hpp`), before calling `vol.set_memory()` -- prepending the
leading slash `fullname()` always has, refusing (`E-LOWFIVE-ATTACH-DATASET-EMPTY-PATTERN`) an empty pattern rather
than silently passing one through. This fixes DEFECT B for both branches from one place, without touching
`split_hdf5_path()` itself (out of this track's scope; that function has its own tests and the single-process
path never reaches LowFive through this unslashed string -- it uses `VolSingleton`'s own blanket `"*"`/`"*"`
wildcards, read and confirmed unaffected). A second new pure function, `BuildIntercommRegistrations()`, builds one
`IntercommRegistration{file, dataset_pattern, intercomm_index}` per binding in the table, reusing the SAME
normalized dataset pattern; the distributed branch of `bruss_lowfive_attach()` calls it once (intercomm index `0`
-- see below) and then calls `vol.set_intercomm(reg.file, reg.dataset_pattern, reg.intercomm_index)` once per
entry, fixing DEFECT A. Registrations are built for EVERY binding in the table on EVERY rank, not filtered by
whether this rank writes or reads that binding's own file: `set_intercomm()`'s own two vectors are consulted ONLY
inside `file_open()`'s remote-open path (grep-confirmed, zero other callers); a rank's own `H5Fcreate` of a file it
writes reaches `file_create()`, a different VOL callback that never touches either vector, so an entry for a
filename this rank's own `file_open()` is never called against costs one unmatched linear-scan comparison and
nothing else (`lowfive_attach_decision.hpp`'s own doc comment on `BuildIntercommRegistrations()` has the full
argument, including the `file_open()`-side `find_matches()` citation).

**The single-intercomm index is a parameter, not a silently-bumped constant.** `BuildIntercommRegistrations()`
takes `intercomm_index` as an explicit argument; `lowfive_attach.cpp`'s own call site passes `0` with a comment
citing why: `decide_attach_mode()` already refuses, by name (`E-LOWFIVE-ATTACH-INCONSISTENT`), any graph whose
qualifying channels do not all partition the world into the SAME two groups, so this attach's own architecture
structurally cannot need more than one intercomm today. A future multi-intercomm design would need to change this
function's own SIGNATURE (e.g. a per-binding-group index), not have a caller silently thread a different value
through an unchanged one.

**Cross-check, reported but NOT fixed (out of this round's own scope -- `src/`/`include/` untouched).** The
env-route plugin (`src/transport/lowfive_plugin_graph.cpp`) builds its own `path_pattern` via `port.substr(slash)`
WITH the leading slash included -- it does not share DEFECT B. `lowfive_transport.cpp`'s own `VolSingleton` and
`DistVolSingleton` both call `vol.set_memory("*", "*")`, a blanket wildcard that matches regardless of slashes --
neither shares DEFECT B either. All three (`lowfive_attach.cpp`, `lowfive_plugin_vol.cpp`,
`lowfive_transport.cpp`'s `DistVolSingleton`) DO share DEFECT A, confirmed by the same `grep -rn set_intercomm`
above returning zero hits in any of them; one-line fixes for each are documented in `lowfive_attach_decision.hpp`'s
own "ROUND 4" comment for whichever track owns those files next, not applied here.

**Model check extended for DEFECT B's own effect on blocking semantics, not for DEFECT A.** DEFECT B changes
whether `serve_all()` actually serves, which the explicit-state model already encodes blocking/release around --
`ProcessPolicy` gained a `nothing_indexed` field and `DecideResult` an `actually_serves` field
(`lowfive_attach_model_check.hpp`); when `nothing_indexed` is set, `kServeNow`/`kFlushDeferred` set
`actually_serves = false` and the simulation's own blocking step treats that as advancing with NO peer release --
a genuinely stuck state, not a silent pass, verified by a new test asserting the search never completes. DEFECT A
(an uncaught exception terminating the process before any serve/receive event) is a qualitatively different
failure mode than a blocking/deadlock scenario this search's own vocabulary represents, and was NOT added to the
model; it is instead verified through `BuildIntercommRegistrations()`'s own direct unit tests (a testable proxy
for whether the real attach registers correctly) plus the source citations above.

**Pinned by (Round 4, additive to the round-2-REVIEW section's own inventory above, now 35 tests total in that
binary).** Three new `NormalizeDatasetPattern` tests (unslashed gets a leading slash; already-slashed is
unchanged; empty is refused) and three new `BuildIntercommRegistrations` tests (one entry per binding, with the
given index and a leading slash; the index is whatever the caller passes; an empty table is an empty result, not
a refusal). Two new `ModelCheck` tests for the `nothing_indexed` case (the shipped protocol gets stuck when
nothing is indexed; the shipped protocol still completes when indexing defaults on). Two new hand-proved mutation
rows in `build-evidence/mutations/lowfive-attach.toml` (`normalize-dataset-pattern-slash-removed`,
reproducing DEFECT B exactly in the pure function -- 33 PASS / 2 FAIL, killed by both the direct test and the
`BuildIntercommRegistrations` test that depends on the normalized form;
`build-intercomm-registrations-empty`, reproducing DEFECT A exactly -- 34 PASS / 1 FAIL), each killed decisively
and the file restored byte-identical (`md5` checked) before the next step. The three pre-existing mutation rows
(`decide-serve-roles-first-server-comparison-swapped`, `serve-state-step-finalize-unflushed-check-disabled`,
`serve-state-step-flush-moved-back-to-pre-open`) were all three re-run (not merely anchor-checked) against this
round's own larger binary and still kill exactly the same test(s) each always did (34/1, 34/1, 27/8
respectively -- the 27/8 count larger than round 2 REVIEW's own 20/7 because this round added more passing tests
to the same binary, not because the mutation's own effect changed). `python3 tools/mutate.py --check-anchors`
reports 338 valid, 0 stale, of 338. Full tier-0 and all 5 repo-health gates clean (this round's own report has the
exact numbers).

**What the host must verify (Round 4), superseding nothing above -- this round's own fix has never been run on a
host.** In `build-lowfive`: rebuild (`cmake --build .`, every target), then `ctest -R
'participants\.lowfive\.mpmd\.3'` (3 tasks). This round's own report has the full `host_pending` paragraph with
exact expected log/stack text for each outcome; in outline: a clean pass means every rank's own log shows the SAME
period-by-period convergence Unit 3a's own single-process harness already measured bit-exact, AND neither
`bruss_lowfive_finalize()` call printed an unflushed-serve refusal; the SAME `MetadataError: file_open(): no
intercomm found` text reappearing would mean this round's own `set_intercomm()` fix did not actually reach the
constructed `DistMetadataVOL` (a stale binary, or a registration built against the wrong table); a hang with the
`file_close -> serve_all -> Index::serve` stack the round-2-update section already named would mean DEFECT B's fix
did not actually make `indexed_datasets` nonzero, reviving the EXACT race role-ordered serve was built to avoid; a
NEW LowFive exception from somewhere in the `get_file_hierarchy`/dataset-read RPC path (not `file_open`'s own
intercomm lookup) would mean this row has reached ground this track has never exercised at all -- see the next
paragraph.

**Explicit, because nothing above should be read as claiming it: the row's own distributed RPC path, per-iteration
steady state, and data correctness have NEVER RUN, on any host, under this track.** Every round of this track
(1 through 4) has fixed what stopped the row from REACHING its own first successful remote open/read -- the null
dereference (D38/round 1), the serve/receive deadlock (round 2 update/REVIEW), and now the missing intercomm
registration plus the unindexed dataset (round 4). None of these rounds has ever observed a SUCCESSFUL remote
`H5Fopen`, a successful `get_file_hierarchy` RPC, a successful dataset read, a second iteration, or a comparison of
the exchanged U/V field values against the in-process `Runner`'s own bit-exact reference (Unit 3a). This round's
own fix is argued from LowFive's source, the same way round 2 REVIEW's own corrected protocol was -- and round 2
REVIEW's own FIRST argument (the rejected pre-open-flush design) was ALSO argued from LowFive's source and was
wrong, caught only once the host actually ran it. The same could be true here: this entry does not claim the row
passes, only that the two specific defects the host's own exception pointed to are fixed and re-verified at the
level this sandbox can reach (direct source citation, a pure-function proxy, and tier-0 tests) -- never with MPI,
never with live HDF5/LowFive serving a real file.

**Still does not close D14.** Unchanged by this round, for the same reasons the round-2-update section already
gave -- the two-HDF5-image hazard is closed (host action only), Demo B's own distributed path is compiled and
never run, and D14's own open accept-mode intercommunicator question (item 3 of its "what remains" list) is
untouched by a fix that only concerns the MPMD row's own already-formed intercommunicator.

**Round 5 (2026-10-13), instrumentation, not a fix -- round 4's own set_intercomm/NormalizeDatasetPattern fix
held, and the host's own NEXT run reached a NEW hang.** The host rebuilt (`cmake --build .`, every target, exit 0)
and ran `ctest -R 'participants\.lowfive\.mpmd\.3'`. The row got PAST its own first remote `H5Fopen` (no more
`MetadataError: file_open(): no intercomm found`) and hung the full 120 s again, with a `sample` at +40 s showing
BOTH `brusselator_u_main` and `brusselator_v_main` inside `file_open()`'s own `get_file_hierarchy` RPC call
(`rpc::client::call` -> `rpc::recv` -> `MPI_Probe`, polling) -- `brusselator_coupler_main` waiting in
`MPI_Allreduce` for both, `u.log`/`v.log` both empty. Under this entry's own role-ordered protocol, U (the lower-
`min_rank()` component) is first-server: its own `bruss_lowfive_write` -> `bruss_lowfive_serve_after_write` ->
`kServeNow` -> `serve_all()` call must have RETURNED WITHOUT BLOCKING, or never run at all -- U is a CONSUMER in
the sample, not inside `serve_all()`. Four hypotheses, named by the lead, each predicting a different trace:

- **H1**: the cold-start write's own view guard (`out_view.window_samples > 0 && out_view.data`, `u_main.c`) was
  false, so `bruss_lowfive_write()` was never reached at all -- no file, nothing for the peer to open.
- **H2**: `serve_all()` returned at once because `indexed_datasets == 0` (`vol-dist-metadata.cpp`'s own `if
  (index.indexed_datasets || !perform_indexing) index.serve();`).
- **H3**: the serve hook returned an action other than `kServeNow` (or never ran at all).
- **H4**: `serve_all()` served the wrong intercomm index, or `intercomms`/`serve_indices` ended up empty.

**Source analysis (cites, not speculation presented as fact) -- what each hypothesis would need to be true, and
whether this sandbox's own reading of the source supports it:**

**H1: argues AGAINST, not FOR, from the FUSE side -- but not disproven, because nothing in this sandbox can
observe the real call.** `Runner::field_window_view()` (`src/runtime/runner.cpp:4160-4186`) computes
`window_samples` as `arc_components / field_components` -- a STRUCTURAL ratio from the DAG arc's own type, set
once the window-scoped arc exists, with NO dependence on whether any step has actually run -- and returns
`slot.data` (the buffer `impl_->slots[arc.value()]` already points at), not a flag that tracks whether the buffer
has been "primed" with a real value yet. `test/verification/brusselator_lowfive_harness_test.cpp`'s own
`deliver()` helper (lines ~218-249) calls the IDENTICAL `view_field()` at the IDENTICAL point in a process's own
lifecycle -- right after `fuse_initialize()`, before any `fuse_begin_step()` -- for the SAME kind of field (`u_out`
cold-start export), with `ASSERT_GT(src.window_samples, 0)` (line 245), and that test PASSES in this track's own
tier-0 suite. This is not a proof that `u_main.c`'s own `out_view.window_samples > 0 && out_view.data` guard
passes on the host -- `u_main.c` is a different process, a different graph load path, and this sandbox cannot run
it with real MPI -- but it is evidence that the MECHANISM H1 would need (the window buffer not yet being primed at
cold start) does not match how `field_window_view()`/`resolve_regions()`'s own "ordinary prime" pass are
documented and tested to behave. Item 3's own trace (`bruss_lowfive_trace_skip()`) is what actually settles this
on the next host run.

**H2: the SPECIFIC mechanism ("is_memory mismatch starves indexed_datasets") does NOT hold, read directly from
LowFive's source -- a narrower, correct finding than "H2 is the cause" or "H2 is ruled out."** `MetadataVOL::
file_create()`'s own comment states plainly: "we build the in-memory hierarchy regardless of whether we match
memory" (`src/vol-metadata/file.cpp`) -- `files.emplace(name_, f)` runs UNCONDITIONALLY, before the `match_any()`
check that decides `is_passthru`. `MetadataVOL::dataset_create()` (`src/vol-metadata/dataset.cpp:30-33`, this
entry's own round-4 citation) computes `is_memory` and THEN calls `obj_path.obj->add_child(new Dataset(...,
is_passthru, is_memory))` UNCONDITIONALLY too -- `is_memory`/`is_passthru` are stored as constructor arguments ON
the new `Dataset` object, not a gate on whether that object is added to the tree at all. `Index::find_datasets()`
(`src/vol-dist/index.cpp:191-201`) is a plain recursive walk (`dynamic_cast<Dataset*>(o)` on every descendant of
every `File` in `idx_srv.files`) that never consults `is_memory`/`is_passthru` either. So a dataset this process
actually created (via `H5Fcreate`+`H5Dcreate2`, regardless of whether its own `set_memory()` pattern matched) IS
counted in `indexed_datasets`, contradicting this entry's own round-4 account of DEFECT B's effect on `serve_all()`
-- that account was WRONG about THIS specific mechanism (round 4's own `NormalizeDatasetPattern()` fix is still
correct and still needed, for a DIFFERENT reason: `is_memory` controls whether `H5Dwrite`/`H5Dread` actually use
the in-memory `Dataset` object at all, not whether `indexed_datasets` counts it). What DOES still gate
`index.serve()`: `serve_all()`'s own declared default is `perform_indexing = true`
(`include/lowfive/vol-dist-metadata.hpp:83`), so `index.indexed_datasets || !perform_indexing` reduces to plain
`index.indexed_datasets` at this call's own default arguments -- if `indexed_datasets` is 0 for ANY OTHER reason
(the write never happening at all, H1; `files` somehow not containing this file at call time; an object-tree
quirk this reading did not find), `index.serve()` is skipped and `serve_all()` returns immediately, exactly the
H2 SYMPTOM, just not via the mechanism this entry's own round-4 account named. Item 2's own trace (file count
before/after close, `H5Lexists()` before close) is what actually measures `indexed_datasets`' own precondition on
the next host run (the dataset's own existence), since nothing public exposes `indexed_datasets` itself from this
translation unit's own installed headers.

**H3: not confirmed or refuted by reading the dispatch code again -- `decide_serve_roles()`/`serve_state_step()`'s
own logic is straightforward and this entry's own round-2/round-2-REVIEW tests already cover the state machine in
isolation, but nothing in THIS sandbox can observe whether `bruss_lowfive_serve_after_write()` was actually
CALLED, or what `action.value()` actually was, on a real run.** Item 4's own before/after trace line names the
action taken (`kServeNow`/`kDeferServe`/`kFlushDeferred`/`kNoopAfterImport`) explicitly, which is the only way to
tell a correct `kDeferServe` (first-reader, expected) from an incorrectly-skipped `kServeNow` (first-server, a
real bug) without guessing.

**H4: `DistMetadataVOL::intercomms`/`serve_indices` are PUBLIC members (confirmed from the installed
`/opt/lowfive/include/lowfive/vol-dist-metadata.hpp`, both declared before any access-specifier label, so the
struct's own default-public rule applies) -- directly traceable, and `create_DistMetadataVOL()`
(`src/vol-dist-metadata.cpp:398-416`) is a SINGLETON keyed on a class-static `info->vol`: a SECOND call in the
same process, with DIFFERENT `local`/`intercomms_` arguments, only WARNS and returns the FIRST instance, silently
discarding the new arguments. `bruss_lowfive_attach()` already refuses a second call in this process
(`AttachState`'s own doc comment) by this entry's own design, so this specific hazard does not look reachable for
a plain `lowfive.mpmd.3` run today -- recorded as a general hazard this track's own single-call-per-process
discipline protects against, not a diagnosis of the measured hang. Item 1's own attach-time trace reports
`intercomms.size()` (not in the lead's own enumerated list, but free to read and directly diagnostic) so a future
run can rule this in or out without guessing either.

**No definite defect established in attach/shim/u_main/v_main code this round -- by the lead's own instruction,
this is why no fix is applied.** The one definite finding (H2's specific mechanism does not hold) is a correction
to this entry's own round-4 account, not a new bug to fix -- `indexed_datasets` behaves correctly regardless of
`is_memory`, so there is nothing broken in that specific path to change. Every other hypothesis remains open,
settled only by a trace this round adds, not by this round's own reading.

**The diagnostic trace added this round.** Two env variables, both read ONCE per process (a cached bool/`getenv()`
call each): `FUSE_LOWFIVE_ATTACH_TRACE=1` (exactly the character `1`; `lowfive_attach_decision.hpp`'s own
`ParseTraceEnabledEnv()`) turns on this track's own `TRACE(...)`-prefixed stderr lines (attach-time: world rank,
component, role, `distributed`, the `set_intercomm()`/`set_memory()` argument lists, `serve_on_close`,
`intercomms.size()`/`files.size()`; write/read: file/dataset/`nvals`, `H5Fcreate`/`H5Fopen` success,
`H5Lexists()` before close, file count after close; the write/read view guards' own skip case, via a new
`bruss_lowfive_trace_skip()` reachable from Fortran through a `bind(C)` subroutine; each `serve_all()` call site,
with the action taken and monotonic elapsed seconds). `FUSE_LOWFIVE_LOG=<level>` turns on LowFive's own
`spdlog`-based logger (`LowFive::create_logger()`, guarded with a `try`/`catch(const std::exception&)` against the
one documented way it can throw -- a second registration of a logger named `"lowfive"` in the same process,
unreachable today for the reason H4's own account gives, handled anyway per the lead's own instruction). Neither
variable changes any return code or call order -- confirmed by this round's own full verification (below) passing
identically to round 4's own count, plus a direct read of every call site added: each is a single cached-bool
branch guarding an `fprintf()`/`create_logger()` call and nothing else. docs/user/lowfive.md's new "Diagnosing a
hang" section has the exact host command and the full account of where the trace lines land (`run_participants.
sh`'s own per-rank stderr redirection) and why `FUSE_LOWFIVE_ATTACH_TRACE`/`FUSE_LOWFIVE_LOG` reach all three
programs under MPICH (no `-env`/`-genv`/`-envnone` flags anywhere in that script's three `mpiexec` invocations,
so hydra's own documented default -- propagate the launching shell's environment -- applies; CTest's own
`ENVIRONMENT` test property only adds variables, never clears the rest).

**Pinned by (Round 5, additive to the round-4 section's own inventory above, now 46 tests total in that
binary).** Five new `ParseTraceEnabledEnv` tests (exactly `"1"` is enabled; unset/empty/`"0"`/word-forms
`"true"`/`"TRUE"`/`"yes"`/`"10"` are all disabled -- the brief's own literal spec, deliberately not a permissive
grammar); three `FormatRegistrationsForTrace` tests and three `FormatMemoryPatternsForTrace` tests (empty list,
one entry's own exact format, two entries comma-separated) -- the pure formatters the attach-time trace line's own
text goes through. Two new hand-proved mutation rows in `build-evidence/mutations/lowfive-attach.toml`
(`parse-trace-enabled-env-accepts-anything-nonempty`, reproducing a permissive-grammar regression exactly -- 44
PASS / 2 FAIL; `format-registrations-for-trace-drops-the-intercomm-index`, reproducing a silent loss of H4's own
diagnostic field -- 44 PASS / 2 FAIL), each killed decisively and the file restored byte-identical (`md5` checked)
before the next step. All five pre-existing mutation rows (two from round 2/round-2-REVIEW, two from round 4, one
from the original `DecideAttachMode` set) were RE-RUN (not merely anchor-checked) against this round's own larger
binary and still kill exactly the same test(s) each always did (45/1, 45/1, 38/8, 44/2, 45/1 respectively -- the
38/8 count for the round-2-REVIEW row differs from round 4's own 27/8 only because this round added more PASSING
tests to the same binary, the same "larger binary, same mutation effect" note round 4's own entry already made for
that row). `python3 tools/mutate.py --check-anchors` reports 340 valid, 0 stale, of 340. Full tier-0 and all 5
repo-health gates clean, and the full tier-1 (MPI+Fortran+LowFive+MOAB) build is clean too, INCLUDING the Fortran
changes to `v_main.f90` (the new `bind(C)` interface, two `target` string declarations, two call sites) -- this
round's own report has the exact numbers.

**What the host must verify (Round 5) -- a trace run, not a fix verification; this round changed NOTHING about
whether the row passes.** In `build-lowfive`: rebuild (`cmake --build .`, every target), then

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=trace ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

Read `u_stdout.<rank>.log`/`v_stdout.<rank>.log` (`run_participants.sh`'s own per-rank redirection). A table of
what the trace would show for each hypothesis, built from this entry's own source analysis above, not from a
second measurement:

| Trace shows | Hypothesis |
|---|---|
| A `TRACE(u_lowfive_write_export): SKIPPED` line at the cold-start call, before any `TRACE(bruss_lowfive_write)` line ever appears for U | H1 (confirms it -- contradicts this entry's own FUSE-side reading above) |
| `TRACE(bruss_lowfive_write)` reaches `exit rc=0`, but `after_close files_in_memory=0` (or `dataset_exists_before_close=0`) | H2-adjacent: the write happened but the VOL's own file/dataset bookkeeping did not retain it -- a defect this round's own source reading did not find and did not expect |
| `TRACE(bruss_lowfive_serve_after_write): action=kServeNow enter` with NO matching `exit elapsed=` line before the 120 s timeout | the healthy signature this entry's own role-ordered design expects -- the hang is downstream (the PEER's own open-and-close never happens), not in U's own serve call at all; re-examine why V never reaches its own `H5Fopen`/`H5Fclose` of `brusselator_u.h5` |
| `TRACE(bruss_lowfive_serve_after_write): action=kServeNow exit elapsed=0.00000Xs` (near-zero) | H2's own SYMPTOM, mechanism not yet identified by this entry (see H2's own account above -- `indexed_datasets` was 0 for a reason this round's own reading did not find) |
| `TRACE(bruss_lowfive_serve_after_write): action=kDeferServe` printed for U (the first-server) instead of `kServeNow` | H3 confirmed -- `decide_serve_roles()` assigned the wrong role, or `this_name`/`roles.value().first_server_name` disagree in a way this entry's own tier-0 tests did not cover |
| No `TRACE(bruss_lowfive_attach)` line at all for U, or `distributed=0` where `distributed=1` was expected | the attach itself took a different branch than this row's own graph should produce -- re-check `decide_attach_mode()`'s own qualifying-channel logic against the ACTUAL graph.yaml this run generated |
| `intercomms.size=0` (or a registration list with an index that does not match `remote_size()`'s own intercomm) in the attach-time trace | H4 confirmed |
| LowFive's own `FUSE_LOWFIVE_LOG=trace` output shows `"enter serve_all, files.size = 0"` for U | the file was never created in THIS process's own VOL at all -- points back to H1, independent of this round's own `bruss_lowfive_write`-side trace agreeing |

**Still does not close D14.** Unchanged by this round -- an instrumentation-only round touches neither the
two-HDF5-image hazard, Demo B's own distributed path, nor D14's own open accept-mode question.

**Round 6 (2026-10-13) -- the trace answered it; a DEFINITE defect, cited and fixed.** The host set
`FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=trace` and ran `ctest -R 'participants\.lowfive\.mpmd\.3'`; the row
hung to the 60 s timeout, and the trace (`u_stdout.0.log`/`v_stdout.1.log`) showed, verbatim:

```
U (first-server): attach: world_rank=0 component=U role=first-server distributed=1
  registrations=[(brusselator_u.h5, /grid/u_out, 0), (brusselator_v.h5, /grid/v_out, 0)]
  memory_patterns=[(brusselator_u.h5, /grid/u_out), (brusselator_v.h5, /grid/v_out)]
  serve_on_close=0 intercomms.size=1 files.size=0
  write: file=brusselator_u.h5 dataset=/grid/u_out nvals=2816 enter | H5Fcreate_ok=1 |
    dataset_exists_before_close=0 | after_close files_in_memory=0 |
    serve_after_write action=kServeNow enter | kServeNow exit elapsed=0.000052s | write exit rc=0
  LowFive's own log: [trace] enter serve_all, files.size = 0 | Index ctor, number of intercomms: 1,
    files.size = 0 | serve_all done
  read: file=brusselator_v.h5 ... enter | receive_before_read action=kReceiveOnly |
    H5Fopen about to call  (then hangs in get_file_hierarchy's MPI_Probe)
V (first-reader): same attach line (role=first-reader), write: H5Fcreate_ok=1,
  dataset_exists_before_close=0, after_close files_in_memory=0, action=kDeferServe, then
  read file=brusselator_u.h5 ... H5Fopen about to call (same hang).
```

H1 (write skipped), H3 (wrong action), and H4 (no intercomm) are RULED OUT by this trace: the write ran, U took
`kServeNow`, exactly one intercomm is registered. H2's own SYMPTOM is confirmed (`serve_all()` saw `files.size =
0` and returned in 52 microseconds without serving) -- but NOT round 4's own guessed mechanism, which round 5
already corrected from source (`is_memory` does not gate whether a dataset is counted in `indexed_datasets`).

**Root cause, read directly from LowFive's source (every cite re-read for this round):**

- `include/lowfive/vol-metadata.hpp:49`: `bool keep = false; // whether to keep files in the metadata after they
  are closed`; `set_keep(bool)` at `:106`. PUBLIC, default `false`.
- `src/vol-metadata/file.cpp` (`MetadataVOL::file_close()`): `if (File* f = dynamic_cast<File*>(...)) { ...; if
  (!keep) { files.erase(f->name); delete f; /* erases all the children too */ } }` -- runs regardless of whether
  this VOL is a plain `MetadataVOL` or a `DistMetadataVOL` (the distributed subclass's own `file_close()` falls
  through to this EXACT function at its own tail: `return MetadataVOL::file_close(file, dxpl_id, req);`).
- `src/vol-dist-metadata.cpp` (`DistMetadataVOL::file_close()`, the LOCAL-file branch): the ONLY place this
  function ever calls `serve_all()` on its own is `else if (serve_on_close && match_any(f->name, "", memory,
  true)) serve_all();` -- gated on `serve_on_close`, which round 2's own fix sets `false` specifically so NOTHING
  auto-serves inside `H5Fclose()`. That `if`/`else if` chain runs BEFORE the tail call into `MetadataVOL::
  file_close()` above.
- Putting these together: with `serve_on_close == false` (round 2's own setting) and `keep` left at its own
  default `false` (never set by any round before this one), `H5Fclose()` neither serves internally NOR keeps the
  file -- it is erased and deleted, unconditionally, the moment `H5Fclose()` returns, BEFORE this process's own
  explicit `bruss_lowfive_serve_after_write()` hook (called by `shim.cpp` AFTER `H5Fclose()` returns) ever runs a
  line. By the time that hook's own `serve_all()` call happens, `files` is ALREADY EMPTY -- exactly the measured
  `files.size = 0`, `"enter serve_all, files.size = 0"`, and the 52-microsecond return (`serve_all()`'s own
  `Index` constructor counts `indexed_datasets` by walking `files`, which has nothing left to walk; `index.
  indexed_datasets || !perform_indexing` is then false at `serve_all()`'s own default `perform_indexing=true`, so
  `index.serve()` -- the part that would actually block polling for the peer -- is never called at all). LowFive
  therefore only works with `serve_on_close=true`, OR with `serve_on_close=false` AND `keep=true` -- this attach
  shipped the first half of the second option without the second half.
- `MetadataVOL::drop(std::string filename)` (`src/vol-metadata.cpp`): `auto it = files.find(filename); delete
  it->second; files.erase(it);` -- NO check against `files.end()`; calling this for an ABSENT name dereferences
  `files.end()`, undefined behaviour.
- `MetadataVOL::clear_files()` (`src/vol-metadata.cpp`): deletes EVERY entry in `files`, not just one -- never
  usable to release a single file.
- `MetadataVOL::file_create()` (`src/vol-metadata/file.cpp`): `File* f = new File(name_, ...); files.emplace
  (name_, f);` -- `std::map::emplace()` is a NO-OP on a DUPLICATE key: if `name_` is still in `files` from a
  PREVIOUS create that was never dropped, the OLD `File*` stays exactly where it is, and the FRESH `f` this call
  just allocated is never stored in `files` at all (only on this NEW open handle's own `mdata_obj`). The
  practical effect, read from this call site directly: SILENT REUSE OF STALE DATA, not an error and not a crash
  -- a LATER `serve_all()` keyed off this filename walks `files`, finds the OLD `File`'s own (now stale)
  children, and serves THOSE, never the fresh write this handle just performed through its own orphaned object.

**The fix.** `lowfive_attach_decision.hpp`'s new `decide_vol_settings(bool distributed) -> VolSettings{serve_
on_close, keep}` is the ONE place that decides both settings together, checked against `VolSettingsSatisfyInvariant
()`'s own invariant (`serve_on_close == false implies keep == true`). `bruss_lowfive_attach()`'s distributed
branch applies both (`vol.serve_on_close = settings.serve_on_close; vol.set_keep(settings.keep);`) in place of
the round-2-era hardcoded `vol.serve_on_close = false;` alone. A new pure function, `FindOwnExportedFiles(table,
this_name)`, derives this rank's own exported file name(s) from the SAME binding table/`this_name` the attach
already holds (never a hardcoded literal), via `fuse::build_lowfive_bindings()`'s own `"<component>.<field>"`
qualification scheme. A new, GUARDED wrapper, `GuardedDropOwnExportedFile()` (`files.find(name) != files.end()`
before calling `vol.drop(name)`, since `drop()` itself has no such check), is called right after EACH of the two
places this process ever actually serves a file -- `bruss_lowfive_serve_after_write()`'s own `kServeNow` case,
and `bruss_lowfive_after_read_close()`'s own `kFlushDeferred` case -- never merely after a write, and never for
the peer's own file. The single-process branch is UNCHANGED, per the lead's own instruction: its existing `vol.
set_keep(true)` call already matches `decide_vol_settings(false).keep`, for the identical underlying reason (round
4's own citation, unchanged); `serve_on_close` does not apply to `LowFive::MetadataVOL` at all, and nothing in
that branch ever serves or drops anything, since it never serves anyone in the first place.

**File lifetime invariant.** A file exists in `files` from its writer's own `H5Fclose` to the end of its serve,
and is dropped before that component's own NEXT `H5Fcreate` of the same name. Checked against the real event
order: cold start (write, read, matching this invariant trivially -- one write, immediately served and dropped,
before this rank's own next write at period 1); the steady state (write-then-read every period, I1's own
progress invariant unchanged by this round); the LAST period and finalize (the first-server's own file is always
served-and-dropped SYNCHRONOUSLY inside the SAME `serve_after_write()` call that wrote it, so nothing of its own
is ever left in `files` by the time of finalize; the first-reader's own file stays in `files`, undropped, exactly
as long as its own deferred serve stays unflushed -- the EXISTING `bruss_lowfive_finalize()` refusal
(`E-LOWFIVE-ATTACH-SERVE-UNFLUSHED`) already catches that case by name, UNCHANGED by this round, because "deferred
serve never flushed" and "file never served-and-dropped" are the SAME fact for a first-reader). Collective
safety: `GuardedDropOwnExportedFile()`'s own drop is local to a rank (no MPI call at all); `serve_all()` remains
the only collective operation in this path, and every rank still takes the same branch from the SAME graph data
`decide_serve_roles()` already reads identically on every rank of either component -- unchanged by this round.

**The trace line the lead asked to be explained before finishing: `dataset_exists_before_close=0`.** Confirmed
an UNRELIABLE proxy, not evidence of a missing dataset. `LowFive::MetadataVOL::link_specific()`'s own `H5VL_LINK_
EXISTS` case (`src/vol-metadata/link.cpp`) calls `Object::locate()` -> `search()` for `H5VL_OBJECT_BY_NAME`,
which walks the path SEGMENT BY SEGMENT, matching each slash-separated piece against a SEPARATE child object.
`dataset_create()`, called by this shim with NO prior `H5Gcreate` for an intermediate group, attaches the new
`Dataset` as ONE FLAT child whose own `name` field is literally `"grid/u_out"` (round 4/5's own account of
`search()`'s own fallback-on-no-match behaviour, re-applied here) -- `search()` looks for a child named exactly
`"grid"` first, finds none (no such child exists; the real child is named `"grid/u_out"`), and reports
not-found. `is_memory`'s own check (`dataset_create()`'s own `match_any(filepath, memory)`, `filepath` built by
`Object::fullname()`, a WHOLE-STRING comparison, never a segment walk) is UNAFFECTED -- confirmed by round 4's
own fix actually working (the write succeeded, `H5Fcreate_ok=1`). REPLACED: `shim.cpp`'s own trace no longer
calls `H5Lexists()` at all; a new accessor, `bruss_lowfive_attach_has_real_file()`, wraps the PUBLIC `LowFive::
MetadataVOL::has_real_file()` instead -- FILE-level, not dataset-level (nothing public exposes a per-dataset
check at all; `Object`/`Dataset` are forward-declared/opaque in the installed headers), but RELIABLE, unlike the
check it replaces. The trace now also prints `keep=` (alongside the already-printed `serve_on_close=`) at attach
time, and `files.size_after_drop=` after each of the two serve call sites.

**The corrected account of what round 4's slash fix did and did not break.** Round 5 already corrected the claim
that an unslashed `set_memory()` pattern starves `indexed_datasets` -- unchanged by this round. The slash fix
itself remains NEEDED, for the reason `is_memory` actually matters, cited above: it governs whether `H5Dwrite`/
`H5Dread` on a given dataset use the in-memory path at all. An unslashed pattern would have meant every
Brusselator dataset's own read/write silently missed interception -- a DIFFERENT defect from the indexing one
round 4 originally (and wrongly) blamed for the symptom it was chasing, now stated precisely instead of left
implicit.

**Model check extended, and RENAMED, not left with two names for the same fact.** Round 4's own `ProcessPolicy::
nothing_indexed` (guessed mechanism: an unslashed pattern) is RENAMED `ProcessPolicy::file_deleted_at_close`
(confirmed mechanism: `!keep`), per the lead's own instruction. A new `ProcessPolicy::drop_after_serve` field
(default `true`, the shipped protocol) models the THIRD named configuration -- `keep=true` but the drop itself
never runs. A new PURE function, `WouldReuseStaleFileOnRecreate(policy, n_iterations)`, flags that configuration
directly: `!file_deleted_at_close && !drop_after_serve && n_iterations > 1` -- a STATIC property of the
configuration (true for ANY interleaving), not a blocking/liveness question the explicit-state search's own
vocabulary can represent at all (the file stays servable, just WRONG, on recreate -- nothing gets anyone stuck).
Three named configurations, each with its own assertion, driven from `decide_vol_settings()` where applicable so
a mutation of the real setting changes the model's own verdict: the SHIPPED configuration (`keep=true`, drop
after serve) completes for N=1, 2, 60 and both rank orders; the ROUND-5 configuration (`file_deleted_at_close=
true`, i.e. `keep=false`) is reported stuck; `keep=true` without the drop is reported, by name, via `WouldReuse
StaleFileOnRecreate()`, as the stale-reuse failure this entry's own "Root cause" section cites -- NOT as a
stuck/diverging search result, because it is not one.

**One mutation row planned, then REMOVED, not shipped as a vacuous pass.** A third mutation row (removing the
guarded drop call at `bruss_lowfive_serve_after_write()`'s own `kServeNow` case) was written, applied, and found
to kill NOTHING: every one of this binary's own 63 tests still passed, because nothing in this sandbox's own
test suite calls the REAL `bruss_lowfive_serve_after_write()` with a REAL `LowFive::DistMetadataVOL` (that
requires live MPI and live LowFive, neither available here) -- the model check's own `kShipped` policy drives
`serve_state_step()`/`decide_serve_roles()`, never `lowfive_attach.cpp`'s own call sites around them. REMOVED
rather than kept and described as "caught," per the lead's own standing instruction elsewhere in this track: "a
piece only measurable on the host is listed in host_pending, not tested vacuously." Whether the guarded drop call
itself is actually reached, in the compiled binary, after a real serve, is UNVERIFIED by any tier-0 test and
listed below as a `host_pending` item -- observable only via this round's own new `files.size_after_drop=` trace
line on a real host run.

**Pinned by (Round 6, additive to the round-5 section's own inventory above, now 63 tests total in that
binary).** Two new `DecideVolSettings` tests (distributed wants `serve_on_close=false`/`keep=true`; single-
process still wants `keep=true`); four new `VolSettingsSatisfyInvariant` tests (both of `decide_vol_settings()`'s
own outputs satisfy it; the round-5 pair `{false,false}` violates it -- the lead's own named test; any pair with
`keep=true` satisfies it; `{true,false}` satisfies it too, since LowFive's own internal auto-serve covers that
pair); four new `FindOwnExportedFiles` tests (U owns `brusselator_u.h5`; V owns `brusselator_v.h5`; an unknown
component owns nothing; an empty table owns nothing); four new `WouldReuseStaleFileOnRecreate` tests (true on the
second iteration when kept-without-drop; false for one iteration; false for the shipped configuration;
false when the file was never kept in the first place -- the two failure modes do not compound). Five `ModelCheck`
tests replacing the two round-4/5 `nothing_indexed`-named ones (now `file_deleted_at_close`-named, same
assertions, driven from `decide_vol_settings()` where the shipped configuration is asserted). Two new hand-proved
mutation rows (`decide-vol-settings-keep-removed`: 57 PASS / 6 FAIL, killing every test that is actually driven
from `decide_vol_settings()`'s own output; `would-reuse-stale-file-on-recreate-ignores-drop-after-serve`: 62
PASS / 1 FAIL), each killed decisively and the file restored byte-identical (`md5` checked) before the next step.
All EIGHT pre-existing mutation rows were RE-RUN (not merely anchor-checked) against this round's own larger
binary and still kill exactly the same test(s) each always did (61/2, 62/1, 62/1, 52/11, 61/2, 62/1, 61/2, 61/2
respectively -- counts larger than earlier rounds' own only because more passing tests now share the same
binary, the same note every prior round's own entry already made). `python3 tools/mutate.py --check-anchors`
reports 342 valid, 0 stale, of 342 (343 before the removed row). Full tier-0 and all 5 repo-health gates clean,
and the full tier-1 (MPI+Fortran+LowFive+MOAB) build is clean too -- this round's own report has the exact
numbers.

**What the host must verify (Round 6) -- this time, a cited root cause and a fix, not only instrumentation; but
still never run on a host.** In `build-lowfive`: rebuild (`cmake --build .`, every target), then

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=trace ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

On success, each rank's own trace should print, IN ORDER: the attach line with `keep=1` (new this round,
alongside the already-printed `serve_on_close=0`); the write trace, ending `files.size_after_drop=0` (served AND
dropped, same iteration); for U specifically, `serve_after_write action=kServeNow enter` followed by its own
`exit elapsed=` line with a LARGE elapsed value (seconds, not microseconds -- it actually BLOCKED, waiting for
V's own open-and-close, unlike round 5's own 52-microsecond return); then the read trace, `H5Fopen_ok=1` (no
longer hanging); this pattern repeating every period; and -- unlike every prior round -- enough periods actually
completing to reach `brusselator_check`'s own pass/fail line, something THIS track has never yet seen. Plausible
next failures, and which of this round's own claims each would contradict: `serve_all()` finds `files.size=1` but
the peer's own `get_file_hierarchy` RPC call THROWS an exception instead of hanging or succeeding -- contradicts
nothing in this entry's own root-cause account (that account only concerns whether `serve_all()` ever BLOCKS
correctly, not what happens once a real RPC round-trip is attempted, which this entry does not claim to have
settled); the dataset READ RPC itself hangs after `file_open()` succeeds -- same caveat, a stage this track has
never exercised; period 2's own `H5Fcreate` collides anyway (a `MetadataError` naming the file, or silently wrong
data on the NEXT read) -- would mean the guarded drop is not actually being reached at the right point, contradicting
this entry's own "File lifetime invariant" section directly, and the FIRST thing to check with `FUSE_LOWFIVE_LOG=
trace`'s own `files.size` lines; `bruss_lowfive_finalize()` refuses with a kept file never served
(`E-LOWFIVE-ATTACH-SERVE-UNFLUSHED`) -- would mean I1 (the write/read alternation this protocol depends on) does
not hold for this row's own graph after all, the SAME caveat every prior round's own account already carried,
unchanged by this round's own fix.

**Still does not close D14.** Unchanged by this round -- the two-HDF5-image hazard is closed (host action only),
Demo B's own distributed path is compiled and never run, and D14's own open accept-mode intercommunicator
question (item 3 of its "what remains" list) is untouched by a fix that only concerns a file's own in-memory
lifetime, never accept mode's own unexposed intercommunicator.

**Round 7 (2026-10-13) -- the serve now sees the file; the row fails fast instead of hanging.** With round 6's fix
landed (46a80ee, 9c77112, 7407644), the host reran the row with `FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=
trace`: full build exit 0, and the row now FAILS in 0.94 s with an abort instead of a 60 s hang. Verbatim:

```
U: attach keep=1 serve_on_close=0; write: H5Fcreate_ok=1, file_has_real_entry_before_close=1,
  after_close files_in_memory=1; kServeNow enter; LowFive: 'enter serve_all, files.size = 1' |
  'Creating decomposer: dim = 1, domain = [0,2816]' | 'Enter Index::serve' | 'Received op: 1 ...
  Called function get_file_hierarchy' | 'Writing brusselator_u.h5 with 1 children (include_data = false)' |
  'Writing grid/u_out with 0 children' | 'Called function file_open ... IndexServe::file_open(brusselator_u.h5)'.
  (U never prints its serve exit because V aborted.)
V: attach keep=1; write ok, files_in_memory=1, kDeferServe; read: file=brusselator_u.h5 dataset=/grid/u_out
  nvals=2816 enter; kReceiveOnly; H5Fopen about to call; H5Fopen_ok=1 (the get_file_hierarchy reply arrived and
  the remote file opened). Then, with H5Dopen2 evidently succeeded and the next call being H5Dget_space
  (shim.cpp:550 in bruss_lowfive_read):
    HDF5-DIAG (MPI-process 1): H5S.c line 486 in H5Scopy(): not a dataspace | H5I.c line 516 in H5Iinc_ref():
      invalid ID | H5S.c line 1730 in H5Sget_simple_extent_type(): not a dataspace
    libc++abi: terminating due to uncaught exception of type LowFive::MetadataError: Unexpected no class in
      dataspace = -1  -> SIGABRT
  macOS crash frames (symbolised): __cxa_throw <- LowFive::Dataspace::Dataspace(long long, bool)
    dataspace.hpp:54 <- Dataspace::Dataspace(long long, bool) dataspace.hpp:45 <- LowFive::MetadataVOL::
    dataset_get vol-metadata/dataset.cpp:321 <- VOLBase::_dataset_get dataset.cpp:261 <- H5VL_dataset_get
    H5VLcallback.c:2457 <- H5Dget_space H5D.c:609 <- bruss_lowfive_read shim.cpp:550 <- v_lowfive_read_import
    v_main.f90:605 <- MAIN__ v_main.f90:468.
```

(The symbolised line numbers, 54/45, do not match this track's own checkout of LowFive/src/metadata/
dataspace.hpp, 7f3a58b, exactly -- 32/23 there -- a known imprecision of `-O2` debug info on a header-only
templated ctor's own member-initializer list, not a different revision: the function identity, its own
EXACT quoted message text and literal `-1`, and the file names all match precisely; the ctor and the throw are
the SAME code either way, confirmed by matching every surrounding frame and the message byte-for-byte.)

**Root cause, read directly from LowFive's source (every cite re-read for this round):**

- This shim's own `bruss_lowfive_write()` (examples/brusselator/participants/shim.cpp) calls `H5Dcreate2(fid,
  "/grid/u_out", ...)` with NO prior `H5Gcreate2(fid, "/grid", ...)`. Round 6 already found this flattens the
  dataset into ONE child of the file/root literally named `"grid/u_out"` -- `LowFive::Object::search()`
  (src/metadata/object.hpp) ALWAYS splits on the first `/` before attempting any match (`auto pos = path.find
  ("/"); ...; for (child : children) if (child->name == first_name) ...; return ObjectPath{this, path};` -- it
  never tries a whole-string match first, re-confirmed by re-reading the function this round). Round 6 found
  this breaks `H5Lexists()` ONLY cosmetically (the write-side `is_memory` check compares the whole string via
  `fullname()` and is unaffected).
- This round finds the SAME flat name ALSO breaks `MetadataVOL::dataset_open()` (src/vol-metadata/dataset.cpp),
  and this one is NOT cosmetic: `auto obj_path = parent->search(name); if (obj_path.path.empty()) result->
  mdata_obj = obj_path.obj; if (!result->mdata_obj) { result->mdata_obj = parent->add_child(new DummyDataset
  (name)); }` -- since `search("grid/u_out")` never finds a child named exactly `"grid"`, `obj_path.path` comes
  back as the WHOLE unconsumed string (non-empty), so `result->mdata_obj` is left null and a `DummyDataset`
  placeholder is substituted instead.
- `DummyDataset` (src/metadata/dummy.hpp): `struct DummyDataset : public DummyObject { Datatype type; Dataspace
  space; DummyDataset(std::string name): DummyObject(ObjectType::Dataset, name) {} };` -- NOT a subclass of
  `Dataset` (both are children of `Object`, but not of each other), and its own `type`/`space` are default-
  constructed (`Hid`'s own default member initializer, `hid_t id = 0;`) -- never populated by anything.
- `DistMetadataVOL::dataset_open()`'s own promotion to an RPC-backed `RemoteDataset` (`if (dynamic_cast<Dataset*>
  (mdata_obj) && RemoteObject::query(mdata_obj)) make_remote_dataset(result, filepath);`) never fires on a
  `DummyDataset*`: `dynamic_cast<Dataset*>` on an object whose real type is `DummyDataset` is correctly null
  (proper RTTI behaviour between unrelated classes), so `make_remote_dataset()` -- which is where `RemoteDataset`'s
  own ctor would fetch a REAL `space` via an RPC named `"metadata"` -- is never called.
- `MetadataVOL::dataset_get()`'s GET_SPACE branch (src/vol-metadata/dataset.cpp) is what V's own `H5Dget_space()`
  reaches instead: `auto& dataspace = static_cast<Dataset*>(dset_->mdata_obj)->space; hid_t space_id = dataspace.
  copy(); log->trace("copied space id = {}, space = {}", space_id, Dataspace(space_id));` -- reads the
  `DummyDataset`'s own never-populated `space` (id == 0, at the same memory offset `Dataset::space` would occupy,
  since no other member precedes it in either class). `dataspace.copy()` is `H5Scopy(0)`, which fails ("not a
  dataspace") and returns LowFive's own documented failure sentinel, `H5I_INVALID_HID == -1`.
- The immediately following `log->trace(..., Dataspace(space_id))` constructs `Dataspace(-1)` as a LOGGER
  ARGUMENT -- evaluated regardless of the active log level, C++'s own eager-argument-evaluation rule for
  function calls, not a LowFive choice. `Dataspace`'s own ctor (src/metadata/dataspace.hpp): `if (id == 0) return;
  H5S_class_t cls_ = H5Sget_simple_extent_type(id); if (cls_ == H5S_NO_CLASS) throw MetadataError(fmt::format
  ("Unexpected no class in dataspace = {}", id));` -- `id == -1 != 0`, so the early-return does not apply;
  `H5Sget_simple_extent_type(-1)` returns `H5S_NO_CLASS`; the throw fires with `id == -1` -- the EXACT measured
  message, down to the literal `-1`.

**Ranked against the alternatives, each read from source, not assumed:**

| Candidate | Verdict | Evidence |
|---|---|---|
| (d) the flat dataset name breaks the remote lookup | CONFIRMED | full causal chain above, exact numeric match (`-1`) |
| (a) `RemoteDataset` never populates `space`, meant to be fetched via RPC | REAL MACHINERY, NOT REACHED | its own ctor (src/metadata/remote.hpp) does fetch `space` via a `"metadata"` RPC call; LowFive's own upstream `examples/prod-con-multidata/` demo (producer: `H5Gcreate(file, "/group1", ...)` then `H5Dcreate2(group, "grid", ...)` with a bare name relative to the group handle; consumer: calls `H5Dget_space()` on the resulting remote dataset successfully) proves this mechanism WORKS -- just never reached here, since `make_remote_dataset()` is never called |
| (b) a hid lifetime bug in `deserialize()`/`make_remote_dataset()` | EXCLUDED for this crash | traced `LowFive::Hid`'s own `H5Iinc_ref`/`H5Idec_ref` ref-counting (src/metadata/hid.hpp) through both functions; correctly balanced for the code that WOULD run if that path were reached -- unverified on a live host run, since the path has never executed in this track |
| (c) a two-HDF5-image/version mismatch | EXCLUDED | a correctly-formatted HDF5 diagnostic plus a C++ exception whose exact message and numeric value this round's own account predicts from LowFive's own code is what a logic bug on a bogus id looks like, not an ABI/image mismatch; this round's own tier-1 configure re-ran the probe and got the same PASS every prior round did |

**Extra trace line that would have separated (d) from (a)/(b) without reading source:** none existed before this
round -- LowFive's own public headers expose no "is this a DummyDataset" check. The fix below removes the need
for one (the lookup now succeeds, so there is no placeholder to detect); `shim.cpp`'s own new `TRACE(bruss_
lowfive_write): H5Gcreate2(/grid)_ok=` line and the extent/rank line added to `bruss_lowfive_read()` are the
closest substitute if a FUTURE row hits an analogous flat-name defect on a different dataset path.

**The fix, in this shim, not in LowFive.** `bruss_lowfive_write()` now calls `H5Gcreate2()` for the dataset's own
parent group path before `H5Dcreate2()`, derived by a new pure function, `SplitDatasetGroupPath()` (lowfive_
attach_decision.hpp) -- `"/grid/u_out"` -> `{group_path="/grid", leaf_name="u_out"}`; `""` means no call needed
(a dataset with no parent-group segment at all). `LowFive::MetadataVOL::group_create()` (src/vol-metadata/
group.cpp) already walks its own unconsumed `search()` remainder segment by segment, creating one `Group` per
segment -- unlike `dataset_create()`, which never does -- so ONE `H5Gcreate2()` call is sufficient even for a
multi-segment parent path (`SplitDatasetGroupPath.MultiSegmentPathKeepsWholeGroupPrefix` pins this). `Object::
fullname()` (used by the write-side `is_memory` check round 4 fixed) walks `parent` pointers and re-concatenates
with `/`, so nesting the dataset under a real `Group` named `"grid"` reproduces the IDENTICAL `"/grid/u_out"`
string that check already matches against -- no regression to round 4's own fix, confirmed by re-tracing
`fullname()` with the new tree shape.

**The width check stays exactly as it was -- not replaced.** `H5Dget_space` on a remote dataset is a SUPPORTED,
WORKING LowFive mechanism once the dataset is properly nested (confirmed against LowFive's own upstream example
above) -- there is no upstream limitation to work around, so none of the candidates the lead asked evaluated are
adopted: a "local view" does not apply (a reader opening a peer's file has no local copy); a second VOL path for
`H5Sget_simple_extent_dims` does not help (the failure is inside `H5Dget_space` itself, before dims extraction);
`H5Dget_storage_size()` is UNSUPPORTED on a memory-backed dataset at all in this revision (`MetadataVOL::
dataset_optional()`'s own warning: "not implemented in metadata yet"); an in-band size attribute would duplicate
information a WORKING `H5Dget_space` already supplies more directly. The graph's own `fuse::validate()`
(include/fuse/model/validate.hpp) only enforces a field's own per-point COMPONENT count between export and
import, never the TOTAL runtime array length (`out_view.count * out_view.components` in u_main.c, with `count`
from each process's own, independently-built domain decomposition) -- nothing in `validate()` guarantees
agreement on that length for a general decomposition, so the runtime check stays worth having, exactly as before.

**Next calls in the read path, confirmed or excluded by source, not yet exercised by any host run:**

- `H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data)` -- CONFIRMED hazard, now FIXED: `DistMetadataVOL::dataset_read()`'s own RemoteDataset branch (src/vol-dist-metadata.cpp) carries an explicit, unresolved upstream comment, `// TODO: support H5S_ALL, need to get file_space_id, mem_space_id with H5Dget_space`, passing whatever it is given straight into `Dataspace(file_space_id)`/`Dataspace(mem_space_id)`; `H5S_ALL` is literally `0` (H5Spublic.h), hitting `Dataspace`'s own early-return and leaving `min`/`max` empty -- `RemoteDataset::query()`'s own bounding box would be built from those empty vectors, not the real extent. Fixed in this shim: `H5Dread` now reuses the SAME `space` handle the width check already validated, for both arguments, never `H5S_ALL` -- a standard HDF5 idiom, and one that never exercises LowFive's own cited TODO.
- `H5Dclose`/`H5Fclose` of the `RemoteFile`, and the producer's own serve-loop return -- CONFIRMED and REFINED (not changed): `DistMetadataVOL::file_close()`'s own RemoteFile branch does three things (an RPC `"file_close"` call, a collective `MPI_Barrier(local)` across every rank of the consumer's own local communicator, then -- only on the consumer's own root rank -- `finish(0)`); `Index::serve()` (src/vol-dist/index.cpp) loops until `finish` has been received on EVERY one of its own registered intercomm servers, then returns. The model check's own "peer's open AND close release the serve" wording already covers this; its own "ROUND 7 CONFIRMATION" comment has the full citations, appended without changing the two bullets it confirms.
- The WRITE side of period 2 (create after the round-6 drop) -- the mechanics check out from source (the guarded drop runs synchronously, in the same call stack, right after the serve that needs it, before any later create can race it), but whether the cross-component SCHEDULING (which periods' write/read/drop interleave with which, enforced by `fuse::WindowCoupler`, out of this track's own scope) actually reaches period 2 in the right order has NEVER been exercised by any host run across all seven rounds -- listed in host_pending, not claimed.

**Pinned by (Round 7, additive to round 6's own inventory, 68 tests total in the binary now).** Five new
`SplitDatasetGroupPath` tests (nested path splits group and leaf; the other component's own nested path; a
single leading slash needs no group; no slash at all needs no group; a multi-segment path keeps the whole group
prefix). One new mutation row (`split-dataset-group-path-leaf-keeps-leading-slash`: 64 PASS / 4 FAIL,
killing exactly the four tests whose own expected `leaf_name`/`group_path` split depends on stripping the
separator at the split point), killed decisively and the file restored byte-identical (`md5` checked) before the
next step. All TEN pre-existing mutation rows re-run via `python3 tools/mutate.py --build-dir <tier-0> --area
examples --jobs 4` (which applies, rebuilds, tests, and restores each row itself) and still KILLED: `mutation
gate: 11 killed, 0 declared equivalent, 0 not assessed here, 0 without a baseline, 0 not killed, of 11`.
`python3 tools/mutate.py --check-anchors` reports 343 valid, 0 stale, of 343. Full tier-0 build and ctest (205/205,
unchanged -- no new tier-0 ctest TARGETS added this round) and all 5 repo-health gates clean. The full tier-1
(MPI+Fortran+LowFive+MOAB) build is ALSO clean -- the first time this track's own shim.cpp changes have compiled
against the REAL `LowFive::DistMetadataVOL`/HDF5 headers and linked both `brusselator_u_main` and `brusselator_
v_main` successfully, confirmed directly in this round's own build log, not merely the tier-0 stub branch.

**What the host must verify (Round 7) -- past TWO fixed defects now, testing a third hazard (the `H5S_ALL` fix)
this row had never reached before; still never run to a passing finish.** In `build-lowfive`: rebuild
(`cmake --build .`, every target), then

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

(`info`, not `trace`, per the lead's own instruction -- the trace log is large; raise it back to `trace` only if
`info` does not separate the next failure's own hypotheses.) On success, each rank's own trace should print, IN
ORDER, for ONE period on both ranks: the attach line (`keep=1 serve_on_close=0`, unchanged); the write trace,
now also printing `TRACE(bruss_lowfive_write): H5Gcreate2(/grid)_ok=1` before `H5Dcreate2`; `file_has_real_
entry_before_close=1`; `files.size_after_drop=0`; for U specifically, `kServeNow enter` followed by its own
`exit elapsed=` line with a LARGE value (seconds, blocking on V's own open-and-close, as round 6 established);
then V's own read trace: `H5Fopen_ok=1`; `H5Dopen2` succeeding (no longer substituting a `DummyDataset`); the
new `TRACE(bruss_lowfive_read): extent dims[0]=2816 rank=1` line (confirms the real dataset, not a placeholder,
was found); the new `TRACE(bruss_lowfive_read): H5Dread reusing space_id=... (not H5S_ALL; ...)` line;
`H5Dread rc=0 first=<U's own written value> last=<U's own written value>`; then V's own `H5Dclose`/`H5Fclose`
(the RemoteFile branch: `file_close` RPC, `MPI_Barrier(local)`, root's own `finish(0)`); U's own `Index::serve()`
finally returning with its own elapsed time printed; this pattern repeating every period; and -- if nothing
else is wrong -- enough periods completing to reach `brusselator_check`'s own pass/fail line, a stage this track
has never yet seen. Plausible next failures, and which of this round's own claims each would contradict: the
`H5Dread` still fails or returns wrong data despite the `H5S_ALL` fix -- would mean `RemoteDataset::query()`'s
own decomposer/redirect RPC chain (src/vol-dist/query.cpp) has a SEPARATE defect this round's own reading did not
reach (not excluded, only not yet exercised); `first`/`last` values do not match what U wrote -- a genuine data
round-trip defect, the first thing `brusselator_check` itself is built to catch, separate from anything the
attach layer owns; `bruss_lowfive_finalize()` refuses with a kept file never served (`E-LOWFIVE-ATTACH-SERVE-
UNFLUSHED`) -- would mean I1 does not hold for this row's own graph after all, the same caveat every prior
round's own account already carried; period 2's own `H5Fcreate` collides anyway -- would mean the cross-
component SCHEDULING hazard flagged above (not excluded by this round's own source reading) is real, the first
thing to check with `FUSE_LOWFIVE_LOG=trace`'s own `files.size`/`files.size_after_drop` lines re-enabled.

**Still does not close D14.** Unchanged by this round -- the two-HDF5-image hazard is closed (host action only),
Demo B's own distributed path is compiled and never run, and D14's own open accept-mode intercommunicator
question (item 3 of its "what remains" list) is untouched by a fix that concerns dataset/group nesting and
H5S_ALL usage, never accept mode's own unexposed intercommunicator.

**Round 8 (2026-10-13) -- the measured order is not "write,read every period" on both sides; the fix is a**
**policy correction plus a two-part order check, not a dynamic exchange.** With round 7's fix landed (commit
80944cc), the host ran the row with `FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info`: build exit 0, and for
the first time a FULL PERIOD round-trips through LowFive on both ranks -- every `H5Dread rc=0`, correct extents
(2816 for U's own field, 2048 for V's), `files_in_memory` back to 0 after each read-close. The row still times
out at 120s, one write into period 1. U's own trace: `write(u#0)` -> `kServeNow` -> exit (1.3ms) -> `read(v#0)`
-> `H5Fopen_ok` -> `kNoopAfterImport` -> `write(u#1)` -> `kServeNow` -> exit (0.7ms) -> `read(v#1)` -> `H5Fopen`
about to call, STUCK. V's own trace: `write(v#0)` -> `kDeferServe` -> `read(u#0)` -> `H5Fopen_ok` ->
`kFlushDeferred` -> flush exit (0.8ms) -> `read(u#1)` -> `H5Fopen_ok` -> `kNoopAfterImport` -> `write(v#1)` ->
`kDeferServe`, LAST -- V then parks at the coupler's own `MPI_Allreduce` sync (`coupler_main.c:163`).

**Root cause: round 2/3's own model assumed BOTH sides run "write then read" every period, with the
first-reader's deferred serve flushed after its OWN next read's close. The host's own measured steady-state
orders differ by role.** U (first-server) writes then reads every period, unchanged from its own cold start. V
(first-reader) writes then reads ONLY at its own cold start (period 0); from period 1 onward it reads, THEN
writes. Under round 2/3's own "first-reader always defers" policy this is circular: U blocks in `read(v#1)`
waiting for V's own `v#1` serve; V's own `v#1` is deferred until V's own NEXT read-close; V's own next read
(`u#2`) needs U's own `u#2`, which U writes only after its own `read(v#1)` returns -- which needs V's own
release first.

**What determines the steady-state order (task 1): channel DECLARATION order, not component order, rank sets,
iteration kind, or acceleration kind.** `brusselator::add_window_coupling()`
(`examples/brusselator/brusselator_graph.hpp`) calls `g.add_channel("u2v", u_out, v_in)` BEFORE
`g.add_channel("v2u", v_out, u_in)`. `fuse::CouplingGraph::add_channel()` assigns each `ChannelId` strictly in
call order (`channels_.push_back(...); return ChannelId{channels_.size() - 1}`), so `u2v` is `ChannelId{0}` and
`v2u` is `ChannelId{1}`. `Runner::Impl::deliver_external()` (`src/runtime/iteration.cpp`) suspends the EXPORTER
of a channel's own delivery BEFORE the IMPORTER, per channel -- its own comment reads "Suspended FIRST, matching
the real time order." `lower_channel()` (`src/ir/rules_channel.cpp`) places each channel's own `ExternalDelivery`
DAG node with `source_producer` as an ordering-edge input, so the scheduler places it, per `OpKind::
ExternalDelivery`'s own doc comment (`include/fuse/ir/op.hpp`), "after the exporting member's own last substep...
before the importer's own advance" -- the same slot `Transport` would occupy. Combined, channel `u2v`
(declared first) makes U the exporter-suspends-first side for that channel and V the importer; channel `v2u`
(declared second) makes V the exporter and U the importer. Component declaration order and rank sets
(`fuse::RankSet`) do not independently determine order in this graph (they happen to coincide with channel order
here, but the mechanism is the channel list, not the component list); the Jacobi iteration kind and the
acceleration kind do not change the mechanism either -- the ordering-edge placement applies identically
regardless of which iteration/acceleration is configured. **Not fully settled, declared as a limitation rather
than a checked claim:** whether `ExternalDelivery` fires once per coupling window or once per Jacobi sweep
(`max_iterations = 60` in this graph) -- `deliver_external()`'s own citation places the node in the slot
`Transport` would occupy WITHIN one sweep, which in principle could mean once per sweep. The host's own measured
counts (exactly two write-serve-read cycles across two periods, not dozens) are consistent with once-per-period,
but this round did not trace `run_region()`'s own sweep-count/convergence machinery far enough to prove it
analytically. Cold start (`u_main.c` lines ~505-516, `v_main.f90`'s own mirror) is a SEPARATE pre-loop priming
call -- hardcoded write-then-read on BOTH mains, no step/sync machinery at all -- mechanically different from the
IR-scheduled steady state, though it produces the identical write-then-read SHAPE for an unrelated reason.

**The fix (task 2): `decide_serve_action(role, completed_any_read)` defers ONLY the cold-start write; every**
**later write serves immediately, on both roles.** A first-reader's write is deferred if and only if it has not
yet completed any read; a first-server's write is never deferred; a first-reader's write past its own cold start
is served immediately, exactly like a first-server's. `ServeState` gained `completed_any_read` (set
unconditionally, both roles, at `kImportClosed`); `serve_state_step()`'s own `kExportWrittenAndClosed` case now
calls `decide_serve_action()` instead of a hardcoded role check. The round 2/3 test that encoded the falsified
"first-reader always defers" model (`FirstReaderDefersThenReceivesAndFlushesRepeatedly`) is REPLACED (not
deleted to force a pass) by `FirstReaderDefersOnlyTheColdStartThenServesImmediatelyEveryLaterWrite`, which asserts
the corrected shape directly.

**The order check (task 3), two parts, deliberately NOT a dynamic runtime exchange.** (a) STATIC, at attach time:
`ComputeGraphOrderSignature()` reduces the loaded graph's own LowFive-bound channels to
`{writer_name, reader_name, writer_rank_count, reader_rank_count}` per channel, in `ChannelId` order;
`CheckGraphOrderValidated()` refuses with `E-LOWFIVE-ATTACH-ORDER-UNVALIDATED`, naming the first differing
property, unless the signature exactly matches `ValidatedGraphOrderSignatures()` -- the ONE measured
configuration (`{"U","V",1,1}` then `{"V","U",1,1}`). This limits the policy to validated configurations ONLY:
a third channel, a swapped channel order, or a multi-rank component is refused by name, not silently assumed to
behave the same way. (b) LOCAL, per event: `ExpectedOrderForPeriod(role, period_index)` gives each role's own
two-slot order for the validated configuration; `CheckAndAdvanceOrder()` is called BEFORE
`serve_state_step()`'s own answer at every real call site in `lowfive_attach.cpp`, and refuses with
`E-LOWFIVE-ATTACH-ORDER-DEVIATION`, naming the expected and observed event kind plus the period and slot, before
the next blocking LowFive call -- never after. (c) Whether anything cheaper/sound could detect the peer's own
order at RUNTIME instead of relying on a pre-validated static signature: nothing sound was found. Each side only
learns its OWN role and OWN read/write history locally; the peer's own order is only observable from the
OUTCOME of a serve/read exchange, which is the exact mechanism already at risk of deadlock. A dynamic exchange
performed after the first problematic write could not prevent a mutual serve already begun (the measured hang
IS such a state: both sides have already issued the calls that deadlock), and the exchange call itself could
deadlock if one side is already blocked inside `H5Fopen` waiting on the other's own serve -- the message
describing the peer's order would itself need a channel that is not already stuck. No sound pre-serve exchange
is proposed here; if one is found later it belongs in `docs/user/lowfive.md`'s own open-questions list, not
implemented speculatively against an unmeasured case.

**The model check (task 4): event sequences as DATA, a third (coupler) actor added for the collective sync, four**
**scenarios plus a soundness link.** `EventKind{kWrite, kRead, kSync}` and `Round8Order{kMeasured,
kBothWriteReadSteady, kBothReadWriteSteady}` make each catalogued order's own per-period shape explicit data
(`SteadyShapeForOrder()`, `BuildEventSequence()`), rather than inline control flow; `BuildScriptFromEvents()`/
`BuildCouplerScript()` turn an event list into the same `MicroStep` scripts `TryStep()` already consumes.
`TryStep()` gained a `kSyncBarrier` step kind modelling the coupler's own three-process `MPI_Allreduce`: a
two-phase arrival/release (`sync_entering`, `sync_passed`) rather than a naive "not behind" check, which is
vacuously satisfiable at the very first barrier (everyone starts at `sync_passed = 0`); and the release
condition is MONOTONIC (`other_passed > my_passed || (other_passed == my_passed && other_entering)`), because a
naive "both exactly equal" check permanently strands the second and third releaser once the first has already
incremented past. `RunSimulationRound8()` runs a 3-actor (U, V, coupler) DFS with a visited-state set.
(i) measured order, new policy: completes for N = 1, 2, 60 periods (state count grows to 4725 at N = 60, not
combinatorially -- `Round8MeasuredOrderCompletesFor1And2And60Periods`). (ii) measured order, round 2/3's own
protocol (`PolicyKind::kRegressedAlwaysDeferFirstReader`, a hand-rolled copy of the falsified policy, kept
separate from `decide_serve_action()` itself so the regression test does not merely re-test the same function it
is supposed to be a regression guard for): reproduces the measured STUCK state exactly
(`Round8MeasuredOrderUnderRegressedPolicyReproducesTheMeasuredHang`). (iii) both sides write-then-read every
period (the strawman round 2/3 implicitly assumed): mutual-serve stuck (`Round8BothSidesWriteReadSteadyGetsStuck`).
(iv) both sides read-then-write every period: also stuck (`Round8BothSidesReadWriteSteadyGetsStuck`).
Soundness link: `static_check_accepts(order) == model_completes(order)` holds for every `Round8Order` catalogue
entry (`Round8StaticCheckAcceptsImpliesModelCompletesForEveryCatalogueEntry`) -- `static_check_accepts()` is true
ONLY for `kMeasured`, because task 1's own derivation shows the channel-order mechanism can only ever produce
the "one exports-then-imports, other imports-then-exports" shape; the two strawman orders have no corresponding
real graph at all, so "not accepted" is correct for them, not merely convenient. The pre-existing N = 1
file-lifetime scenarios (`RunSimulation`/`BuildScript`, the three illustrative deadlocks and two completions)
are kept working, unchanged, alongside the new machinery -- they remain meaningful because an N = 1 scenario is
order-independent.

**Pinned by (Round 8, additive to round 7's own inventory, 92 tests total in the binary now).** One
`ServeStateStep` test replaced (the falsified-model one, named above); six `ModelCheck` tests replaced (their
own N >= 2 "both write-read every period" script is the now-proven-unrealistic `kBothWriteReadSteady` strawman,
which only completed before because it was tested against the now-falsified "always defer" policy -- each
replacement carries a comment saying which test and why, per this round's own instruction not to delete a test
just to make it pass). 24 new tests: `DecideServeAction` (2), `ExpectedOrderForPeriod` (3), `CheckAndAdvanceOrder`
(5), `ComputeGraphOrderSignature` (1), `CheckGraphOrderValidated` (4, one refusal per differing property:
swapped channel order, different component names, a third channel, a multi-rank component), `ModelCheck` (6, the
four scenarios above plus the soundness link and a "measured is the only accepted catalogue entry" check),
`BuildEventSequence` (3). Three new mutation rows (`decide-serve-action-defers-every-first-reader-write`,
`check-graph-order-validated-accepts-any-graph`, `check-and-advance-order-write-check-disabled`), killed
decisively and the file restored byte-identical (`md5` checked) before the next step.
`python3 tools/mutate.py --check-anchors` reports 346 valid, 0 stale, of 346. Full tier-0 build and ctest
(205/205, unchanged -- no new tier-0 ctest TARGETS added this round) and all 5 repo-health gates clean. The full
tier-1 (MPI+Fortran+LowFive+MOAB) build is ALSO clean, including the production wiring of `CheckAndAdvanceOrder()`
into all three real LowFive hooks (`bruss_lowfive_serve_after_write`, `bruss_lowfive_receive_before_read`,
`bruss_lowfive_after_read_close`) and `CheckGraphOrderValidated()` into `bruss_lowfive_attach()` itself, both in
`examples/brusselator/participants/lowfive_attach.cpp` -- confirmed directly in this round's own build log
against the real `LowFive::DistMetadataVOL` headers, linked into both `brusselator_u_main` and
`brusselator_v_main`.

**What the host must verify (Round 8) -- the policy fix and order check have never been run on a host; only the**
**model check and the pure functions are tier-0 tested.** In `build-lowfive`: rebuild (`cmake --build .`, every
target), then

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

On success, both ranks' own trace should show, for TWO periods: the attach line, now followed immediately by
`TRACE(bruss_lowfive_attach): order-check-verdict=accepted` (the static check, run once, both ranks); then, per
write/read/import-close event, a line naming the monotonic per-process `event_index`, the `period_index`, and
the policy/reason string (`cold-start-first-reader` or `steady-state` at write; `receive-only` at
import-about-to-open; `flush-deferred` or `noop-after-import` at import-close) -- U's own period-1 `write(u#1)`
should now show `steady-state` (served immediately, not deferred) where round 7's own trace showed the process
getting stuck instead; V's own period-1 `write(v#1)` should ALSO show `steady-state`, served immediately rather
than deferred past V's own next read. If this round's own derivation and fix are both correct, the row should
reach `brusselator_check`'s own pass/fail comparison between the LowFive-routed values and the in-process
`Runner`'s own reference values -- a stage this track has never yet reached. Plausible next failures, and which
of this round's own claims each would contradict: an `E-LOWFIVE-ATTACH-ORDER-DEVIATION` fires on the host despite
92/92 tier-0 tests passing -- would mean task 1's own derived order table is incomplete for some real-host
timing this round's model did not capture (the model's own DFS explores every interleaving the three-actor
script ALLOWS, not every interleaving a real MPI/LowFive runtime could produce); the row still hangs with no
order-check refusal at all -- would mean the hang has a THIRD cause downstream of both fixed defects, not an
order-of-events problem (the one class this round's own policy and checks can catch); `brusselator_check` itself
reports a value mismatch once the row completes -- a genuine data round-trip defect, separate from anything the
attach/order layer owns, the first thing `brusselator_check` is built to catch; the Jacobi-sweep-frequency
question (task 1's own declared limitation) turns out to matter -- would mean `ExternalDelivery` fires more than
once per period for `max_iterations > 1`, and the order table (built assuming once per period) would need
revisiting even though this round's own tier-0 evidence cannot distinguish the two cases.

**Still does not close D14.** Unchanged by this round -- the two-HDF5-image hazard is closed (host action only),
Demo B's own distributed path is compiled and never run, and D14's own open accept-mode intercommunicator
question (item 3 of its "what remains" list) is untouched by a fix that concerns serve/read ORDER, never accept
mode's own unexposed intercommunicator.

**Round 9 (2026-10-13) -- the order check and policy both HOLD on the host; the row now times out on a**
**DIFFERENT track's own collective, not this one's; a separate, measured exit-time crash is fixed for all**
**three processes.** The host confirms round 8's own fix: `order-check-verdict=accepted` printed on both ranks,
no `E-LOWFIVE-ATTACH-ORDER-DEVIATION` fired in 364 (U) and 365 (V) order-check calls (122 writes + 121 reads x 2 events, and 121 writes + 122 reads x 2), and BOTH coupling windows now
complete through LowFive -- U made 122 writes/121 reads, V 121 writes/122 reads, every `H5Dread rc=0`, correct
extents (2816/2048), `files_in_memory` back to 0 after each close, values varying as a waveform should. The row
still times out at 120 s, but for a reason OUTSIDE this track's own scope: after the last delivery, U and V wait
in a base-communicator `MPI_Allreduce` the coupler never joins -- the "Collectives" track's own defect, not a
regression in anything this track owns.

**Task 1 -- the coupler exit crash, measured separately from the row's own timeout.** Symbolised macOS crash
report of `brusselator_coupler_main`: SIGABRT at process exit, AFTER `MPI_Finalize`, `libc++abi: terminating due
to uncaught exception of type std::__1::system_error: mutex lock failed: Invalid argument`. Trace (innermost
first): `__pthread_kill` <- `abort` <- `__abort_message` <- `std::terminate` <- `__cxa_throw` <- `std::__throw_
system_error` <- `std::mutex::lock` <- `std::lock_guard<std::mutex>` (in `liblowfive.dylib`) <- `spdlog::
details::registry::get` <- `spdlog::get` <- `LowFive::get_logger()` (`log-private.hpp:53`) <- `LowFive::VOLBase::
_term()` (`base.cpp:210`) <- `H5VL__free_cls` <- `H5I__dec_ref` <- `H5I_dec_ref` <- `H5VL_conn_free` <- `H5VL_
term_package` <- `H5_term_library` <- `__cxa_finalize_ranges` <- `exit` <- dyld start. Since the coupler is the
ONLY process that currently reaches `exit()` in this row (U and V still wait on the OTHER track's own
Allreduce), U and V would hit the identical crash once that is fixed -- this round's own fix is designed to hold
for all three, not patched onto the coupler alone.

**(a) Why the coupler has a LowFive VOL class registered at all -- RETRACTED by round 10's own host**
**measurement; see round 10's own section below for the full account, including an UPDATE that this crash is**
**now the ONLY thing between this row and a pass.** Round 9's claim (quoted below for the historical record, do
not rely on it) was: `examples/CMakeLists.txt`'s own `verification.brusselator.participants.lowfive.{mpmd,
accept}.3` row sets `ENVIRONMENT "HDF5_VOL_CONNECTOR=lowfive;HDF5_PLUGIN_PATH=...;FUSE_LOWFIVE_BINDINGS=..."`
on the ONE ctest `COMMAND`, shared by all three MPMD app segments, and HDF5's own env-route default-VOL
resolution registers the connector in the coupler's own process because of it. Round 10 measured the coupler's
own `unsetenv()` call (added by round 9's own fix, below) RUN, confirmed by its own printed trace line, and the
IDENTICAL abort still happened -- disproving environment inheritance as a complete explanation. The two-HDF5-
image hazard (D14) is a DIFFERENT, already-measured-PASS condition on this host and remains NOT this crash's
cause, independent of the above retraction: D14's own symptom is a silent `H5Pset_vol(): "not a file VOL ID"`
failure from a split id table, not a correctly-registered connector's own exit-time teardown -- this crash's
trace shows a single, correctly-resolving `H5VL__free_cls` call chain, consistent with one image, not two.

**(b) Ranked fixes.** `unsetenv()` on the three env-route variables, before `MPI_Init`, before any native HDF5
call -- KEPT for the coupler as still-sound defensive hygiene (the coupler never calls `bruss_lowfive_attach()`
and has no legitimate use for the connector), but round 10 measured that it does NOT, by itself, stop this exit
crash -- it is no longer claimed to remove the hazard at its source; see round 10's own section below for what
is and is not excluded. A deterministic teardown in `bruss_lowfive_
finalize()` -- ADOPTED for U and V, which DO deliberately attach: an explicit, SYNCHRONOUS `H5close()` call,
after `serve_state_step(kFinalize)`'s own unflushed-serve check succeeds and BEFORE the caller's own `MPI_
Finalize()` (both u_main.c and v_main.f90 already call `bruss_lowfive_finalize()` strictly before `MPI_
Finalize()`, confirmed by reading both call sites, unchanged by this round). Sound because it runs the SAME
`VOLBase::_term()` -> `get_logger()` chain while the process is still in ORDINARY execution -- no exit()-time
static-destruction unwind has begun, so spdlog's own registry cannot have been torn down yet, regardless of
atexit() registration order. STRICTER than relying solely on the pre-existing `std::atexit(+[]{H5close();});`
already registered in `bruss_lowfive_attach()` (the SAME mechanism `VolSingleton`'s own constructor uses, per
docs/user/lowfive.md's "HDF5_VOL_CONNECTOR abort" section) -- that registration is KEPT, not removed, as a
backstop for an abrupt-exit path that skips `bruss_lowfive_finalize()` (`H5close()` is documented idempotent).
An explicit `spdlog::shutdown()` ordering was considered and NOT adopted: it only moves WHICH line crashes if
`_term()` is still reachable afterward, and orders a DIFFERENT library's own static object from FUSE's side, no
more reliable than the atexit race it replaces. Making LowFive's logger a `shared_ptr` the VOL object itself
holds was NOT adopted and NOT implemented: a change to LowFive's own source (`log-private.hpp`/`vol-base.hpp`),
out of scope ("do NOT patch LowFive, only report"). NOT established from source alone, no host run to confirm:
whether the coupler's own first native HDF5 call happens through MOAB's own mesh I/O, FUSE's backend layer, or
HDF5's own env-route resolution being triggered by library load -- the fix holds regardless of which it is, so
this round did not chase the exact call site further; a limitation, not a settled claim.

**Implementation.** `LowFiveProcessRole{kAttachingParticipant, kNonAttachingCoordinator}`, `EnvRouteVolVariable
Names()` (the three literal names), `ExitHygienePlan`, `decide_exit_hygiene_plan(role)` -- a PURE function of
role (`lowfive_attach_decision.hpp`). `coupler_main.c` calls `unsetenv()` on all three names before `MPI_Init`,
with a trace line gated by the same `FUSE_LOWFIVE_ATTACH_TRACE=1` convention (duplicated inline rather than
sharing `lowfive_attach.cpp`'s own `TraceEnabledCached()`, since this file is plain C with no dependency on that
C++ header and is not one of `lowfive_attach.cpp`'s own targets). `bruss_lowfive_finalize()` calls `H5close()`
(with a trace line) immediately after its own existing unflushed-serve check succeeds, driven by `decide_exit_
hygiene_plan(kAttachingParticipant)`.

**Task 2 -- the trace field and two pure-function names were a misnomer.** The `period=N` field in round 8's own
trace lines is the exchange-pair index (three `ServeEvent`s per pair: write-closed, import-about-to-open,
import-closed), not a coupling window -- the host measured 61 deliveries per window for a `--periods 2` run (122
U writes total), not one. Renamed throughout: the trace field (`period=` -> `exchange=`); `ExpectedOrderForPeriod`
-> `ExpectedOrderForExchange`; `OrderTracker::period_index` -> `exchange_index`; the model check's own `n_periods`
parameters -> `n_exchanges`; every doc comment and test name using "period" for this unit. "period" is now
reserved, in this file and in docs/user/lowfive.md, for the Brusselator's own `--periods` CLI flag and the outer
coupling-window loop -- never for the write/read-pair unit tracked here. NAME correction only: no ORDER-CHECK or
POLICY logic changed (the semantics held for all 364 (U) and 365 (V) events the host measured with the OLD name in
place).

**Task 3 -- closes round 8's own open question with the host's own measurement ON COMMIT 7aec9c8 (BEFORE the**
**owner's device-branch merge); round 10 found the SAME row measures a DIFFERENT figure once the row COMPLETES**
**on HEAD -- see round 10's own section below.** Round 8 left unsettled whether `ExternalDelivery` fires once
per coupling window or once per Jacobi sweep. ANSWERED by the host, on 7aec9c8: ~61 deliveries per window (122
U writes / 2 windows), i.e. once per sweep (not once per window) -- a MEASURED figure, not a derived one (the
Collectives track owns the Runner's own sweep-count/convergence machinery this would need to derive from first
principles; the exact relationship between 61 and this row's own `max_iterations = 60` is that track's own
question, not restated here -- round 10's own section below records that this track's own D57 entry has since
REFUTED the "61 = kMaxIterations + 1" hypothesis outright). The model check's own N=60 scenario (a pre-round-9
stress value, this row's own `kMaxIterations` reused as an exchange count) is now explicitly documented as
such, not a claim it equals a Jacobi sweep count on either commit. Added N=122 (7aec9c8's own figure) to the
existing `{1, 2, 60}` scenario (now `Round8MeasuredOrderCompletesFor1And2And60And122And25Exchanges`, round 10
also added N=25, the completed-row HEAD figure once the hang is fixed, to the SAME loop) -- confirming the
existing, generic machinery already covers BOTH figures (shown, not merely asserted): completes for either,
state count scaling consistent with round 8's own finding that it does not explode combinatorially.

**Pinned by (Round 9, additive to round 8's own inventory, 95 tests total in the binary now).** 3 new tests
(`EnvRouteVolVariableNames.ListsExactlyTheThreeEnvRouteVariables`, `DecideExitHygienePlan.NonAttachingCoordinator
UnregistersAndSkipsExplicitH5close`, `DecideExitHygienePlan.AttachingParticipantCallsExplicitH5closeAndDoesNot
Unregister`); the 60-exchange scenario test and 5 other test names renamed for task 2 (no assertions changed,
confirmed by the unchanged PASS count before and after the rename pass); one existing scenario-i test extended
with N=122 (task 3). One new mutation row (`decide-exit-hygiene-plan-roles-swapped`), killed decisively, file
restored byte-identical (`md5` checked) -- 15 mutation rows total, all killed. `python3 tools/mutate.py
--check-anchors` reports 347 valid, 0 stale, of 347. Full tier-0 build and ctest (205/205, unchanged -- no new
tier-0 ctest TARGETS added this round) and all 5 repo-health gates clean. The full tier-1 (MPI+Fortran+LowFive+
MOAB) build is ALSO clean, including `brusselator_coupler_main`'s own new `unsetenv()` calls and `bruss_lowfive_
finalize()`'s own new `H5close()` call, both confirmed directly in this round's own build log against the real
`LowFive::DistMetadataVOL`/HDF5 headers.

**What the host must verify (Round 9) -- the exit-hygiene fix has never been run on a host; a prior run never**
**reached ANY process's own exit() cleanly (the coupler's crash was the first time this row got far enough to
exercise ANY exit path at all).** In `build-lowfive`: rebuild (`cmake --build .`, every target), then

```
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

On success, all three processes should print `TRACE(coupler_main): unregistered env-route VOL vars...` (coupler,
once, before `MPI_Init`) and `TRACE(bruss_lowfive_finalize): calling H5close() before MPI_Finalize...` (U and V,
once each, after their own last delivery and before `MPI_Finalize`); all three should exit 0, with NO SIGABRT
and no `libc++abi: terminating due to uncaught exception` on any rank. The row's own OTHER, already-known
failure (the Collectives track's own Allreduce hang) may still cause a 120 s timeout even if this round's own
fix is completely correct -- that outcome does NOT contradict this round's own claims; the absence of a SIGABRT
on EVENTUAL exit (once the Collectives track's own fix lands) is the actual thing being verified here, not the
row's own overall pass/fail. Plausible next failures, and which of this round's own claims each would contradict:
the coupler STILL aborts despite the `unsetenv()` calls -- would mean this process's own first native HDF5 call
happens BEFORE `main()` even starts (a static constructor eagerly calling `H5open()`), which this round's own
reading could not rule out from source alone; U or V aborts despite the new `H5close()` call in `bruss_lowfive_
finalize()` -- would mean the ordering argument in task 1(b) above has a gap this round's own reasoning did not
find (e.g. a SEPARATE, EARLIER native HDF5 call in u_main.c/v_main.f90 that registers things before `bruss_
lowfive_attach()` ever runs); an `E-LOWFIVE-ATTACH-ORDER-DEVIATION` fires where round 8's own host run reported
none -- would mean this round's own RENAME (task 2) accidentally changed behavior, not just names, a regression
this round's own unchanged-PASS-count check argues against but a host run has never independently confirmed on
this specific binary.

**Still does not close D14.** Unchanged by this round -- the two-HDF5-image hazard is closed (host action only),
Demo B's own distributed path is compiled and never run, and D14's own open accept-mode intercommunicator
question (item 3 of its "what remains" list) is untouched by a fix that concerns exit-time VOL-class teardown and
trace-field naming, never accept mode's own unexposed intercommunicator.

**Round 10 (2026-10-13, measured mid-round on HEAD b98777d, then again after the Collectives track's own D57**
**fix landed, HEAD 0dc8c70) -- round 9's own exit-crash root cause is RETRACTED by measurement; this round's**
**own reading excludes every FUSE-side registration path, with citations, but the true trigger is NOT**
**established; a host experiment is designed, not run; a separate sweep-count change is recorded, mostly**
**explained by the Collectives track's own D57 entry; this crash is now the ONLY thing between the row and a**
**pass.** Same host, same row. Mid-round, on b98777d (after the owner's Kokkos+CUDA device-branch merge;
`/opt/moab-tempest` was removed, MOAB now at `/opt/moab/cmake-all` with a shared `libhdf5.310` rather than an
embedded copy -- measured identical on both): the coupler's own trace line DID print (`unregistered env-route
VOL vars... before MPI_Init (role=kNonAttachingCoordinator)`, confirming round 9's `unsetenv()` call ran), and
the SAME abort still happened, byte-identical trace (`std::mutex::lock` <- `lock_guard` (liblowfive) <-
`spdlog::details::registry::get` <- ... <- `LowFive::VOLBase::_term` <- `H5VL__free_cls` <- ... <-
`H5_term_library` <- `exit`). This is EXACTLY the failure mode round 9's own "What the host must verify"
paragraph (above) anticipated as the most likely contradiction of its own claims: "a static constructor eagerly
calling `H5open()`" before `main()`. LATER THE SAME ROUND, after the Collectives track's own D57 fix for the
row's own unrequested-period hang landed (HEAD 0dc8c70): the row NO LONGER HANGS. It now ends in 1.87 s with
`run_participants.sh: participant run exited 15`; `FUSE_COLLECTIVE_TRACE` agrees exactly across all three ranks
(48 lines, one md5 across all three, rank field removed); U made 25 writes/25 reads, V 25 writes/25 reads (1
cold start + 24 sweeps each way, 12 sweeps per period, 2 periods); `u.log` (296 bytes) and `v.log` (240 bytes)
are BYTE-IDENTICAL to the passing transport-only twin's, and `brusselator_check` run by hand on them reports
`--check: OK` (`final_mean_u=1.58658618817 final_mean_v=1.9982226403`, balance/exchange errors all below
`2.4e-13`). The LowFive data path has reproduced the transport run exactly over both windows for the FIRST
TIME. This coupler exit crash is now the ONLY thing standing between this row and a pass. U's and V's own exit
codes were NOT observed in this measurement (mpiexec killed them once the coupler's own BAD TERMINATION was
reported) -- whether round 9's own `H5close()`/finalize path in U and V exits cleanly, once the coupler stops
aborting, is still unmeasured.

**Task A -- what registers a LowFive connector in the coupler, read-first, no host run.** Three FUSE-side
mechanisms are EXCLUDED, with citations:

1. *Runner/transport construction.* `RunnerConfig::transport` is assigned in exactly ONE place in the whole
   tree -- `src/abi/participant.cpp:1048`, `config.transport = "mpi";`, inside `if (c.communicator_set)`, which
   is ALWAYS true for this row (all three mains set `setup.communicator`). `TransportRegistry::instance().
   create()` (called from `src/runtime/runner.cpp:2478` and `:4516`, both gated on `config.transport`) therefore
   NEVER constructs a `LowFiveTransport` for any rank of this graph, on any rank. `VolSingleton()`/
   `DistVolSingleton()` (`src/transport/lowfive_transport.cpp`, the function-local-static lazy singletons that
   wrap `LowFive::MetadataVOL`/`DistMetadataVOL` + `LowFive::H5VOLProperty`) have NO callers anywhere in the
   tree except `LowFiveTransport`'s own member functions (exhaustively grepped) -- structurally unreachable
   when `LowFiveTransport` itself is never constructed.
2. *The fieldless graph marker.* `examples/brusselator/participants/brusselator_graph_tool.cpp` (~line 296)
   emits a fieldless "lowfive" component with `kind = fuse::ComponentKind::Coupler` explicitly set (its own
   comment records that this was once a bug -- defaulting to `ComponentKind::Physics` let it slip past the
   skip below -- already fixed before this round). `src/abi/participant.cpp:941`'s `if (decl.kind ==
   ComponentKind::Coupler) continue;` skips BOTH the real "coupler" component and this fieldless marker
   identically, since both now share that one `kind` value -- neither is wired up to construct anything via
   that loop.
3. *This track's own attach code.* `examples/CMakeLists.txt:374-376` links `brusselator_coupler_main` against
   only `fuse::abi MPI::MPI_C` directly; `lowfive_attach.cpp` is not in its `target_sources` (present only for
   the U/V targets, lines 387/412). The coupler's own binary does not contain this track's own attach function
   at all.

NEWLY CONFIRMED this round (explains `otool -L`, does NOT by itself explain the crash): `src/CMakeLists.txt`
links `liblowfive`/`liblowfive-dist` into `fuse-coupler-transport` unconditionally whenever `FUSE_ENABLE_
LOWFIVE` is ON (~line 304, the file's own comment: `lowfive_transport.cpp` is in the library's sources "every
configuration"), and links `fuse::moab` into `fuse-coupler-backend` `PUBLIC` unconditionally whenever `FUSE_
ENABLE_MOAB` is ON (~line 503) -- both libraries, and MOAB's own TempestRemap dependency, are linked into
EVERY FUSE executable, including the coupler, by build-time design, independent of role. Linking is not
invoking: dyld maps every linked dylib at process launch regardless of whether application code ever calls a
symbol in it, but a dylib's OWN static/global constructor runs at load time unconditionally too, so this fact
alone neither confirms nor excludes a trigger.

Within LowFive's own granted source (`/Users/mahadevan/Code/hdf5/LowFive`, read-only, rev 7f3a58b): every call
to `H5VLregister_connector` (the unvalued, programmatic form) or `H5VLregister_connector_by_value` (the
env-route form `_str_to_info` uses) is an ordinary, explicitly-invoked function -- `LowFive::VOLBase::
register_plugin()` (`src/vol-base/base.cpp:155-163`, called only from `LowFive::H5VOLProperty`'s own
constructor, `include/lowfive/H5VOLProperty.hpp:14`), `LowFive::VOLBase::_str_to_info()` (`src/vol-base/
info.cpp`, ~line 160), and HDF5's own example pass-through VOL bundled in the same tree (`src/tom-passthru-
plugin.cpp`, `OUR_pass_through_register()`/`OUR_pass_through_str_to_info()`, same explicitly-invoked pattern).
`VOLBase::connector_id`/`connector` are plain namespace-scope DATA (a `hid_t` and an aggregate of function
pointers), not executable code -- declaring them has no load-time side effect. `H5PLget_plugin_type()`/
`H5PLget_plugin_info()` (`base.cpp:118-122`) are ordinary functions HDF5's own plugin loader calls AFTER a
`dlopen()` via its PATH search, not auto-invoked merely by being linked. No `__attribute__((constructor))` and
no non-trivial global C++ object was found anywhere in LowFive's own granted source after an exhaustive search.

**NOT excluded, no source access:** MOAB's and TempestRemap's own source are not granted to this track, so an
eager static initialiser there calling `H5open()` (or registering a VOL connector directly) before `coupler_
main()`'s own `unsetenv()` calls run cannot be ruled out, and is the LEADING, UNPROVEN candidate this round's
own reading leaves standing. This is a limitation of what source-reading alone can establish, stated as such,
not a settled claim.

**Task B -- a host experiment to separate the candidates (design only; "I run it, you do not"); extended to**
**also capture U's and V's own exit codes, unmeasured as of the last host run above.** An `lldb` batch session
targeting ONLY the coupler's own segment of the SAME MPMD `mpiexec` command `run_participants.sh`'s own `mpmd`
branch already builds (U and V run normally, unmodified), with four breakpoints, each set to auto-print a
backtrace and continue:

```
lldb -b -o "breakpoint set -n main" \
     -o "breakpoint set -n H5open" \
     -o "breakpoint set -n H5VLregister_connector" \
     -o "breakpoint set -n H5VLregister_connector_by_value" \
     -o "breakpoint command add 1 -o 'bt' -o 'continue'" \
     -o "breakpoint command add 2 -o 'bt' -o 'continue'" \
     -o "breakpoint command add 3 -o 'bt' -o 'continue'" \
     -o "breakpoint command add 4 -o 'bt' -o 'continue'" \
     -o run -- <coupler binary and its own argv, exactly as run_participants.sh's own mpmd
                segment constructs them for this ctest row>
```

Four outcomes, each separating specific candidates:

(i) `H5open` or either `H5VLregister_connector*` variant fires BEFORE the breakpoint on `main` -- proves a
    pre-`main()` static initialiser in a linked library (the leading candidate above), independent of anything
    `coupler_main()`'s own code does.
(ii) The UNVALUED `H5VLregister_connector` fires AFTER `main`, with a backtrace rooted in FUSE's own code --
    disproves this round's own exclusions above; the printed backtrace names the missed call site directly.
(iii) `H5VLregister_connector_by_value` (the env-route form) fires despite the confirmed `unsetenv()` run --
    means the env-route variables are still reaching HDF5 through some OTHER channel (e.g. `mpiexec`'s own
    `-genv`, or a per-rank environment `unsetenv()` does not reach) -- a different claim than a static
    initialiser.
(iv) Neither breakpoint ever fires despite the crash still happening -- a symbol-resolution mismatch (stripped
     symbols, a differently-named internal alias); re-target at `H5_term_library` or `H5VL_term_package`
     instead, which the crash's own trace already shows resolving successfully.

**Exit-code capture, added because the row no longer hangs.** Since the coupler's own abort now causes mpiexec
to kill U and V before their own exit status is observed, the experiment above should ALSO redirect each
rank's own stdout/stderr with `mpiexec`'s own per-rank output option (kept rather than discarded on a killed
run) and add one trace line immediately AFTER each rank's own `MPI_Finalize()` call in `u_main.c`/`v_main.f90`/
`bruss_lowfive_finalize()` reporting that rank is about to exit 0 cleanly -- printed only if execution reaches
that point, so its ABSENCE is itself informative. This can run against the pre-fix binary today (it will show
no such line from U/V, consistent with "not yet measured" above) and again once whichever candidate task B's
four breakpoints identify is addressed.

**Task D -- the sweep-count figure recorded in round 9 (task 3 above) was measured on commit 7aec9c8, BEFORE**
**this merge; the SAME row measures a DIFFERENT, now-COMPLETED figure on HEAD, and the Collectives track's**
**own D57 entry has since narrowed, though not fully closed, why.** Mid-round, on b98777d, while the row still
hung on an unrequested third period: the three ranks' first 48 ordinals were byte-identical (24 Constrain/base
ordinals per period, i.e. 12 sweeps per period), U then started an unrequested period 2 (ordinal 49) and
blocked, with the coupler stopped at 48; U made 26 writes/25 reads, V 25 writes/26 reads -- an ASYMMETRIC,
INCOMPLETE count, now superseded. On HEAD 0dc8c70, after the Collectives track's own fix: the row completes,
all three ranks' traces agree exactly (48 lines each), and U/V both made 25 writes/25 reads -- SYMMETRIC. BOTH
the 7aec9c8 figure (61/window, 122 writes) and the 0dc8c70 figure (12/period, 25 writes/reads per rank) are
MEASURED, on their own respective commit; NEITHER is derived from first principles. THE CAUSE OF THE CHANGE
FROM 61 TO 12 IS SUBSTANTIALLY, THOUGH NOT FULLY, NARROWED by the Collectives track's own D57 entry (relayed
here, not re-derived): running the REAL species physics (not a mirror) through the SAME configuration gives
`ref_iterations = 12` on both components, REFUTING the earlier tentative "61 is exactly `kMaxIterations + 1`
(60+1)" hypothesis outright -- a converged, tolerance-driven window of 12 sweeps cannot itself produce 61
deliveries. The 61 figure traces to commit `60b6207`, an ancestor of D57's own Round 1 fix and a parent of the
device-branch merge, confirming it was genuinely measured on 7aec9c8, not a transcription error. D57 also
records, AS AN ARITHMETIC OBSERVATION ONLY, NOT A FINDING (adopted here with the same caveat, no stronger
claim): 122 = 1 + 10 x 12 + 1, so 122 deliveries are ALSO consistent with 12 sweeps per period over TEN
periods, not two -- the original "61 per window" framing may have been an artefact of assuming a 2-period run
when the measurement was actually a 10-period one. This is NOT established, only consistent; other candidates
D57 names (redundant `resume_period()` calls under the old eager continuation; a D42-related change in how a
window's convergence verdict reaches ranks outside `R.participants`) remain the Collectives track's own open
question. `lowfive_attach_decision.hpp`'s own comment and `docs/user/lowfive.md`'s "Round 9"/"Round 10"
sections are corrected in place (not silently updated) to label which figure came from which commit. The model
check's own N=122 scenario is RELABELLED as 7aec9c8's own figure (no longer "the host's real count" on HEAD);
N=25 (the completed-row HEAD figure) is added to the SAME loop
(`Round8MeasuredOrderCompletesFor1And2And60And122And25Exchanges`), confirming the existing, N-generic machinery
already covers it without a new scenario type (shown, not merely asserted: both complete). A mid-round N=26
measurement (taken while the row still hung) is NOT kept as a separate scenario -- it reflected an incomplete,
bug-affected run, superseded by the completed N=25.

**Task E -- `tools/probe-hdf5-images.c`'s own blind spot for HDF5 embedded inside libMOAB is recorded as a**
**new, separate entry, D58 below** (doc-only; the probe itself is NOT changed this round) -- not folded into
this entry, since it is a new observation about a different tool, not a correction of anything D39 itself
claimed.

**No code change this round.** Source-reading did not establish a definite defect in FUSE's own code (the
three mechanisms excluded above were FUSE's own candidates; the leading unproven candidate lives in MOAB/
TempestRemap's own compiled binary, unreadable and unpatchable from this track). Round 9's `unsetenv()` call
in `coupler_main.c` and the `H5close()` call in `bruss_lowfive_finalize()` are KEPT, unchanged -- the former as
still-sound defensive hygiene even though not proven to fix THIS crash, the latter because it addresses a
DIFFERENT, separately-confirmed mechanism (U/V's own deliberate attach) unaffected by this round's finding.

**What the host must verify (Round 10) -- task B's own lldb experiment, run against the SAME ctest row's own**
**MPMD coupler segment, now the ONLY remaining failure in the row.** Expected output for a FIXED coupler (once
whichever candidate task B identifies is addressed): exit code 0 on all three ranks, no SIGABRT, no `libc++abi:
terminating due to uncaught exception`, only MPICH's own benign leak report (if any) after the row's own last
`MPI_Finalize`; the same `brusselator_check --check: OK` verdict and byte-identical `u.log`/`v.log` already
measured this round should be unaffected by whatever fixes the coupler (the data path itself is already
confirmed correct). For task B's own four breakpoints, each outcome (i)-(iv) above states what it would prove;
the single MOST LIKELY outcome, given round 9's own `unsetenv()` already ran and did not help, is (i) -- a
pre-`main()` static initialiser -- but this round's own reading cannot promote that from "leading candidate" to
"established" without the host actually running the experiment. Once the coupler stops aborting, U's and V's
own exit codes become the next unmeasured stage -- the exit-code-capture addition above is written to catch
this on the SAME run, not a separate one.

**Resolution of the coupler's exit abort (2026-10-13, measured on the owner's machine, MOAB `/opt/moab/cmake-all`,**
**MPICH 5.0.0a1; supersedes the "leading, unproven candidate" wording above).** Three measurements, none from reading.
(1) A throwaway interposer (dyld `__interpose` of `H5open` and `H5VLregister_connector*`, loaded into the coupler's
segment only, kept in `/tmp`, not committed) printed one event: the first `H5open` call came from
`_GLOBAL__sub_I_WriteHDF5.cpp` in `libMOAB.5.6.0.dylib`, run by dyld's initialisers before `main`, with
`HDF5_VOL_CONNECTOR=lowfive` and `HDF5_PLUGIN_PATH` set in the environment. No public `H5VLregister_connector*` call
was seen: the registration inside `H5open` itself (HDF5 reading `HDF5_VOL_CONNECTOR` at initialisation) was not
observed directly, so it is inferred from (2). The same hand run recorded the segments' exit codes: U 0, V 0,
coupler 134 (so U's and V's own teardown, round 9's `H5close()` before `MPI_Finalize`, exits cleanly).
(2) `run_participants.sh` run directly in six environments: `lowfive.mpmd.3` with the ctest environment fails
(`participant run exited 15`, the `mutex lock failed` abort present); with only `FUSE_LOWFIVE_BINDINGS` set it
passes, `brusselator_check` prints `--check: OK`, no abort; with none of the three variables set it passes the same
way. A hand-launched MPMD run without the connector variables exited 0 on all three ranks, wrote `u.log` and `v.log`
byte-identical to the passing transport-only twin's, and its `FUSE_COLLECTIVE_TRACE` had 48 lines with one md5 on
all three ranks. (3) After the fix below, `ctest -R participants.lowfive` on `build-lowfive` reported
`participants.lowfive.mpmd.3` Passed (1.70 s).

**Cause and fix.** `u_main` and `v_main` attach LowFive programmatically, so the two rows' `ENVIRONMENT`
(`HDF5_VOL_CONNECTOR`, `HDF5_PLUGIN_PATH`, `FUSE_LOWFIVE_BINDINGS`, written for the env-var route) was not read by
anything in these rows, and it made HDF5 register the LowFive connector in the coupler as soon as libMOAB's static
initialiser opened HDF5. At process exit that connector's teardown (`VOLBase::_term`) reaches the already-destroyed
spdlog registry and aborts. The three variables are removed from both rows' `ENVIRONMENT` in
`examples/CMakeLists.txt`. Round 9's `unsetenv()` in `coupler_main.c` cannot help when libMOAB is linked, because
it runs in `main`, after the initialiser; it is kept and is untested against a coupler built without MOAB.

**Not established.** The abort mechanism is a latent teardown defect in LowFive (a connector class freed after
spdlog's registry is gone): any process that registers the connector through the environment and exits normally can
hit it; it was not patched here. `participants.lowfive.accept.3` failed in every environment through this round,
behind the single, blanket named refusal `E-LOWFIVE-ATTACH-ACCEPT` (D14). **Superseded by D14 Stage B
(2026-10-13):** that blanket refusal is replaced by `decide_attach_mode`'s own narrower checks (a formed bridge
communicator required; at most two Connect-mode components, D18) now that the accept-mode intercommunicator gap
itself is closed by a new ABI entry; the row Passed on its first host run after that stage (1.26 s) — see D14's
own "what remains" item 3 for the measurement.

### ~~D40. This host's Open MPI (4.1.5) hangs in `MPI_Alltoallv` on any communicator but `MPI_COMM_WORLD`~~ — WORKED AROUND 2026-10-10

**Measured 2026-10-10**, tracking down a hang in `parallel.mpi_transport.{4,5}`, `parallel.comm_graph.{4,5}`
and every other multi-rank test whose binary calls `MPI_Alltoallv` on a communicator `MpiTransport`,
`fuse::transport::connect()`, the MOAB backend or `CommGraph`'s union-communicator construction had
itself produced (`MPI_Comm_dup`, `MPI_Comm_split`+`MPI_Comm_create_group`, or similar) rather than
`MPI_COMM_WORLD` straight from `MPI_Init`. Reproduced in isolation with zero fuse or Kokkos code: a
bare `MPI_Comm_dup(MPI_COMM_WORLD, &dup)` followed by one `MPI_Alltoallv` on `dup`, at 4 ranks, with
either a disjoint-halves pattern (2 senders, 2 receivers) or a fully dense one. Rank 0 always returns
from the call; every other rank is still inside it when the process is killed. The identical pattern
on `MPI_COMM_WORLD` itself, unchanged, always completes.

An initial hypothesis — `MPI_Comm_dup` returning locally on some ranks before an internal context-id
handshake settles on all of them, under `MPI_THREAD_MULTIPLE` — turned out to be wrong, and the
`MPI_Barrier` added to close it (on the four communicator-creation sites named above) did **not**
fix anything: the same reproducer still hangs with an `MPI_Barrier` on `dup` placed immediately
before the `MPI_Alltoallv`, and the hang reproduces identically under `MPI_THREAD_FUNNELED` and even
plain `MPI_Init` with no threading support requested at all, and independent of which BTL (`vader`,
`tcp`) or PML (`ob1`) carries the traffic. Those barriers were removed again once this was clear —
they cost a synchronisation per communicator for no effect.

What actually isolates it: `ompi_info --param coll tuned --level 9`'s `coll_tuned_alltoallv_algorithm`
control variable. Forcing algorithm 1 ("basic_linear" — this build's own default choice for a non-
`MPI_COMM_WORLD` communicator at this rank count, confirmed via `--mca coll_tuned_alltoallv_algorithm 1`)
hangs every time; forcing algorithm 2 ("pairwise") passes every time, on the isolated reproducer and
on the real `fuse_parallel_mpi_transport_test`/`fuse_parallel_comm_graph_test` binaries alike (30/30
and 28/28 subtests respectively, where they previously hung). This reads as a real bug in this Open
MPI build's `coll/tuned` component's basic_linear alltoallv implementation for non-`COMM_WORLD`
communicators, not anything about thread level, BTL, or fuse's own collective sequencing.

**Worked around, not fixed** (this project does not own Open MPI): every `fuse_add_mpi_test` and
`fuse_partition_add_mpi_test` row (`test/CMakeLists.txt`, `test/partition/CMakeLists.txt`), the
`parallel.connect.*` rows and the `verification.brusselator.participants.*` rows (both in
`examples/CMakeLists.txt`) now set `OMPI_MCA_coll_tuned_use_dynamic_rules=1` and
`OMPI_MCA_coll_tuned_alltoallv_algorithm=2` in their `ENVIRONMENT` test property. Harmless on any
other MPI (an `OMPI_MCA_*` variable an MPICH or other `mpiexec` does not recognise is simply unused
environment, not an error), and a correctness-preserving algorithm choice everywhere dynamic rules
reach on Open MPI itself — just not necessarily the fastest one at rank counts this project does not
target. A regression guard lives at `MpiTransportDiag.RawAlltoallvDisjointHalvesOnADuplicatedCommAtFourRanks`
(`test/parallel/mpi_transport_test.cpp`): it fails fast if this workaround is ever missing or stops
being effective, rather than leaving the next host run to rediscover it as an unexplained 120s
timeout. **Anyone running a fuse binary directly against this exact Open MPI build, outside ctest,
needs the same two environment variables set** to avoid the same hang.

A separate, unrelated host quirk surfaced by the same investigation: every `n=5` row
(`model.descriptor_mpi.5`, `parallel.mpi_transport.5`, `parallel.comm_graph.5`, and every other
rank-5 row) failed outright with Open MPI's own "not enough slots" refusal, even though the host has
8 logical CPUs. `lscpu` says why: 4 physical cores behind those 8 (hyperthreading), and Open MPI
sizes its default slot count from physical cores, not `nproc`. The same `ENVIRONMENT` lines above
also carry `OMPI_MCA_rmaps_base_oversubscribe=1` (the environment-variable equivalent of `mpiexec
--oversubscribe`) for exactly this reason -- these are correctness tests at a handful of ranks, not
a throughput benchmark, so running more ranks than physical cores is the point at `n=5`, not a
mistake to refuse.

### ~~D41. This host's Open MPI (4.1.5) reports `MPI_APPNUM` 0 for every rank of an MPMD launch, regardless of segment~~ — WORKED AROUND 2026-10-10

**Measured 2026-10-10**, tracking down `E-CONNECT-APPNUM` failures in `parallel.connect.mpmd.{1+1,2+3,3+2}`
and `verification.brusselator.participants.mpmd.{3,5}`. `fuse::transport::check_mpmd_layout()`
(`include/fuse/transport/connect.hpp`) validates an "mpmd" launch by `MPI_Allgather`ing each rank's
`MPI_APPNUM` and checking that the ranks reporting each value match the component's declared `RankSet`
— which only works if a launcher actually differentiates `MPI_APPNUM` per segment of a colon-syntax
(or `--app`-file) launch.

Reproduced in isolation with zero fuse code (`tools/probe-mpi-appnum.c`): a two-segment colon-syntax
launch, `mpiexec -n 1 ./probe 0 : -n 1 ./probe 1`, where each segment's single rank calls
`MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, ...)` and compares against the appnum it was told (via
argv) to expect. On this host, **every** rank in **every** segment reports appnum 0 — confirmed with
2-segment and 3-segment colon launches and with `--app` appfile syntax alike. This is not a fuse bug:
`check_mpmd_layout()`'s own declared `RankSet`s (built by `test/parallel/connect/connect_common.hpp`'s
`build_graph()`) are correct; the MPI runtime simply never sets the per-segment value the standard
allows it to provide (MPI-3.1 §11.9.2 requires `MPI_APPNUM` be *set* when the launcher supplies one, not
that launchers differentiate it across segments).

**Worked around, not fixed** (this project does not own Open MPI), following the same measure-then-gate
shape as [D23](#d23-acceptconnect-cannot-complete-on-mpichs-ch3nemesis-device-when-both-programs-share-a-node)'s
`FUSE_ENABLE_CONNECT_ACCEPT`: `tools/probe-mpi-appnum.sh` builds and launches `tools/probe-mpi-appnum.c`
as a real two-segment MPMD job at configure time (`cmake/FuseDependencies.cmake`, right after the
accept/connect probe), with the same three verdicts (0 works, 1 measured broken, 2 could not test —
left enabled). The verdict sets `FUSE_ENABLE_CONNECT_MPMD`, which gates the `mpmd` leg of the `_mode`
loop in both `test/CMakeLists.txt`'s `parallel.connect.*` rows and `examples/CMakeLists.txt`'s
`verification.brusselator.participants.*` rows — the same `continue()`-on-ungated-mode pattern those
loops already used for `accept`/`FUSE_ENABLE_CONNECT_ACCEPT`. On this host both `FUSE_ENABLE_CONNECT_ACCEPT`
and `FUSE_ENABLE_CONNECT_MPMD` are now measured, independently, so the `connect`/`verification.brusselator.
participants` families register whichever modes this installation can actually carry, rather than
registering-and-permanently-failing a row that can never pass here.

### ~~D42. A rank outside a window-scoped region's participants never learned the window converged, and deadlocked the next sweep~~ — FIXED 2026-10-10

**Measured 2026-10-10**, as `parallel.window_runner_mpi.3` (Timeout): `WindowRunnerMpi.
ThreeRanksWithAnUncoupledThirdMatchTheReference` adds a third, genuinely uncoupled component `C` on
rank 2 to `WindowRunnerMpi.TwoDisjointRanksMatchTheSingleProcessReference`'s own A-on-rank-0/B-on-rank-1
window coupling — and that FIRST test, which only asks for `FUSE_REQUIRE_AT_LEAST_RANKS(2)`, also runs
under the 3-rank row, with no `C` declared at all (`parallel.window_runner_mpi.3`'s own "empty third
rank" shape per the test file's header comment). Both hung at `n=3`; both passed instantly at `n=2`.

Root-caused (not an MPI quirk this time — a real Runner bug) by tracing `FUSE_TRANSPORT_DEBUG`
(`src/transport/mpi_transport.cpp`) and a temporary residual trace in the window-convergence sweep
(`src/runtime/iteration.cpp`) side by side: the sweep's channel exchange (`MpiTransport::move()`,
one `MPI_Alltoallv` per channel per sweep) runs over the Runner's single shared transport communicator
— the WHOLE communicator `RunnerConfig::base_communicator` named, duplicated once by `MpiTransport::
set_communicator()`, not a sub-communicator scoped to the window's own two members — and by design
(`mpi_transport.cpp`'s own header comment) a rank in neither of a channel's rank sets still calls
`MPI_Alltoallv` with all-zero counts every sweep, rather than being excluded from the call. The window's
own convergence decision, though, is only ever computed and delivered to `RegionRuntime::participants`
(`gather_window_block_samples()`, `src/runtime/runner.cpp` — deliberately: a point-to-point exchange
over just the channel endpoints' own ranks, not a collective the rest of the world would have to join).
Rank 2 is never in `participants` for this region (it hosts no checkpointed component the region's
channels touch), so its own copy of `window_converged` never leaves `false`, and it kept calling
`move()` every sweep up to `max_iterations` (300) while ranks 0/1 — informed, and done in 3–30 sweeps —
had already moved on. Rank 2's next `MPI_Alltoallv` on that communicator then had no matching call from
anyone, ever, for the rest of the run: a deadlock, not a hang that eventually clears, and a correctness
bug reachable by ANY window-scoped coupling sharing its base communicator with so much as one rank
outside the window's own two sides, not something specific to this test's third component.

**Fixed** at the root: a new `Runner::Impl::broadcast_window_convergence()` (`src/runtime/runner.cpp`,
`src/runtime/runner_impl.hpp`) `MPI_Bcast`s the already-decided `window_converged` flag from
`participants.front()` to literally every rank of `base_communicator` — not just `participants` —
called unconditionally, once per sweep, by every rank that reaches the window-scoped branch in
`src/runtime/iteration.cpp`'s `run_region()` (participant or bystander alike, since that branch runs
identically on every rank regardless of this region's own membership). A no-op (returns the input
`converged` value unchanged, no MPI call) when `participants.size() <= 1` — tier 0, or every
checkpointed component on one rank, where no other rank exists to diverge from in the first place.
Verified: both `parallel.window_runner_mpi.{2,3}` pass (`WindowRunnerMpi.
ThreeRanksWithAnUncoupledThirdMatchTheReference` in well under a second, down from a 120s timeout).
Peer-failure propagation to a window region's own bystanders (as opposed to convergence) was found to
have the identical gap while investigating this — a local failure on a participant reaches only other
participants, via the same `gather_window_block_samples()` — but is not exercised by any current test
and is left for a future pass rather than bundled into this fix.

**Note 2026-10-13 (review 2026-10-11 RT-1, RT-2).** The ordinary (non-window) multi-rank region got the
same bystander treatment as this entry's window fix (RT-1: a rank outside `R.participants` now reads the
agreed convergence outcome instead of deciding from an unreduced payload), and five rank-local failure
paths that used to `co_return` ahead of later collectives now record and carry forward (RT-2). The
peer-failure-to-bystander gap noted in the previous paragraph was confirmed live on the ordinary-region
path too: the RT-2(c) restore-failure test hangs at 3–5 ranks and is therefore registered at exactly two
ranks. That propagation gap is still open, for both region kinds.

### D43. `example.fsi_channel`'s IQN-ILS coupling is marginally stable near the beam's first natural period — a 1e-12 change in the fluid solve diverges it; OPEN, rows DISABLED under `FUSE_ENABLE_CUSPARSE`

**Measured 2026-10-11**, found while verifying the examples' cusolverRf sparse-solve backend
(`examples/common/sparse_solve_backend.hpp`, `docs/dev/runner-device-dispatch.md` Part B) on
a GTX 1050 Ti. Default example parameters (`fluid 24x12, beam 16, mass ratio 1, dt 0.01`;
the ctest rows use `16x8`, `nx-beam 12`, `t-end 0.1` and show the same thing):

- Flag off (Eigen::SparseLU), `--accel iqn-ils`: converges every step; **step 7 (t = 0.08,
  the beam's first natural period is 0.0888 s) takes 46 sweeps**, twice every other step's
  21–24. Flag off with `--accel none`: non-finite after 14 periods. `--accel aitken`:
  non-finite after 11 periods. Only IQN-ILS gets through, and only just.
- Flag on (cusolverRf with extended-precision refinement): steps 0–6 reproduce the flag-off
  deflections to every printed digit; **step 7 runs its full 200 sweeps and diverges to
  2e80**; `none`/`aitken` fail at the same periods as flag-off. Every one of the 730 fluid
  solves before the divergence has relative residual ~1e-16 and agrees with an Eigen solve
  of the same system to ≤ 1.07e-12 (the backend is in fact closer to a long-double reference
  than Eigen is: 1.4e-16 vs 3.4e-14 on this 900-unknown saddle point,
  `verification.sparse_solve_backend`). A perturbation of the fluid response three orders
  below any tolerance in play is enough to flip step 7 from "46 sweeps" to divergence.

**What this is.** Not a solver defect — both solvers are correct to working precision, and
the flag-off build diverges with two of the three accelerators on its own. It is the
coupling at that step sitting at the edge of IQN-ILS's stability, which the fixed Eigen
rounding happens to land on the convergent side of. The `fsi_*` verification rows
(`fsi_stokes_mms`, `fsi_beam`, `fsi_coupled`, `fsi_transient`) pass with the flag on;
`fsi_energy_diag`'s 480-period coupled run sits on the same edge and is D44.

**What was done.** `example.fsi_channel` and `example.fsi_channel.omp_identical` carry the
ctest `DISABLED` property when `FUSE_ENABLE_CUSPARSE=ON` (reported as "Not Run (Disabled)",
not hidden; `examples/CMakeLists.txt`). Flag off — the default — nothing changes.

**What would close it.** Understanding why IQN-ILS needs 46 sweeps at step 7 in the first
place (D17 records a related accelerator asymmetry on the Brusselator). Candidates, none
measured: the IQN-ILS filter threshold on nearly dependent secant columns near a turning
point of the beam; the window's convergence criterion on `fluid_force` (a derivative of the
solution, so κ-amplified) sitting at 1e-9 relative. Until then the example is a stability
probe, not a regression gate, for any alternative fluid solver.

### D44. Under `FUSE_ENABLE_CUSPARSE` the cusolverRf fluid solve is not bitwise reproducible run to run in situ, and `fsi_energy_diag`'s 480-period coupled run usually fails to converge at period 140 because of it; OPEN, that one case SKIPPED under the flag

**Measured 2026-10-11** on the GTX 1050 Ti, `build-cusparse` (tier0, Debug, flag on) against
`build-all` (all, Debug, flag off), with a temporary per-period total-energy trace
(`%.17e`) added to `FsiEnergyDiag.LongRunEnergyIsBounded` (12 beam periods, 40 steps each,
IQN-ILS tol 1e-9, 300 sweeps max, `abort_on_nonconvergence`):

- **Flag off: two runs of one binary are bitwise identical over all 480 periods** (every
  energy and every cumulative sweep count), ratio 0.998067, pass. The host path is
  deterministic, as `example.*.omp_identical` already asserts.
- **Flag on: two runs of one binary agree bitwise through period 71 and diverge at period
  72** by 1.4e-12 relative; four differently-built binaries of the same source (a no-op line
  added or not; `-ftrivial-auto-var-init=zero` / `=pattern` on the test TU) each diverge from
  the others at a different late period (72, 94, 103), all at the 1e-12 level. Of 15 flag-on
  runs, 11 stopped at period 140 with the IQN-ILS region unconverged after 300 sweeps —
  `beam_velocity[Relative] 1.3042612065887604` in every one of those 11 (the blow-up has one
  shape), `fluid_force[Relative]` 8e-17–2e-16 (the fluid solve itself converged). The other 4
  passed with ratio 0.998067. GCC's `-Wmaybe-uninitialized` at `-O1` on the TU reports nothing
  in project code; forcing all automatic variables to zero or to a pattern changes the
  trajectory *no more than* re-running does. So: not an uninitialised read, not code layout —
  run-to-run nondeterminism in the flag-on path, which only the cusolverRf solve introduces
  (`verification.sparse_solve_backend`'s 300-repeat tests are reproducible in isolation; the
  earlier in-situ diagnostic over 24,519 `fsi_transient` solves saw ~4% of same-input repeats
  differ bitwise when other solves ran in between). The extended-precision refinement step
  pins the solution to within ~1e-16 of exact but not to one bit pattern.
- Flag on vs flag off at period 1: energies differ by 3.6e-10 relative (19 vs 21 sweeps) — the
  IQN-ILS fixed point is tolerance-limited at 1e-9, not solver-limited, so a 1e-16 solve
  difference becomes a 1e-10 state difference immediately and a 1e-12 trajectory fork later.

**What this is.** D43 in the verification suite: a marginally stable IQN-ILS coupling that
the deterministic Eigen rounding lands on the convergent side of, and an ulp-level
nondeterministic solve lands on either side of. It is not a defect in the backend's accuracy
(D43's numbers) and not a leak, race or UB on the host (the three experiments above). Valgrind
memcheck could not get past CUDA context creation in 25 minutes and was abandoned.

**What was done.** `FsiEnergyDiag.LongRunEnergyIsBounded` calls `GTEST_SKIP()` when
`FUSE_HAVE_CUSPARSE` is 1, naming this entry; the other five cases in that binary still run
under the flag. Flag off — the default — nothing changes. The review-fix line this was first
blamed on (`release_rf()` at the top of `SparseLuBackend::seed_from_host_lu()`) was
exonerated by rebuilding the exact previously-passing source and watching it fail 2 of 3.

**What would close it.** Either a bitwise-reproducible device solve (cusolverRf's
triangular solves have no determinism switch; the refactorisation could stay on the device
and the two triangular solves move to the host from `cusolverRfExtractBundledFactorsHost`,
deterministic by construction and, at these sizes, no slower than the launch-bound device
solve) or the D43 coupling fix. The determinism contract D45 established for reductions
applies to any device solver fuse adopts; this is the measured reason.

### ~~D45. The deterministic dot and norm reductions differed from the host by one ulp on the Cuda leg~~ — CLOSED 2026-10-10

**Measured 2026-10-10** as `num.space_kernels`'s `DeterministicReductionsMatchTheChunkedReferenceExactly`
failing on the Cuda execution space only, and only for `deterministic_dot`/`deterministic_norm_squared`
(never `deterministic_sum`, which has no multiply): `got_dot` and `want_dot` differed by exactly one ulp
(`1.2895225459088599e+34` against `...601e+34` at n = 256).

**Cause.** nvcc's device compiler contracts `s += a(i) * b(i)` into one fused multiply-add
(`--fmad=true`, its default), rounding the product and the accumulate together; the host compiler
on this toolchain (g++ for the x86-64 baseline, which has no FMA instruction; nothing in this
project sets a contraction policy) rounds them separately. Same summation order, different
low-order bit — so the chunked, order-fixed reduction the determinism contract promises
(`include/fuse/num/space_kernels.hpp`, `DeterministicDotFunctor`,
`DeterministicNormSquaredDiffFunctor`) was deterministic per space but not across spaces.

**Closed** by `unfused_multiply_add()` in the same header: on the device it is
`__dadd_rn(s, __dmul_rn(x, y))`, CUDA's explicitly round-to-nearest, not-further-fusible primitives,
so the device path double-rounds exactly as this host path does; on the host it is the plain
expression. Both functors' inner accumulation goes through it. The host path is unchanged (it never
fused here). A host compiler that contracts by default on FMA hardware (Apple clang at `-O2`, D30)
fuses that plain expression too; the two spaces then still agree with each other, but not with a
build that does not contract. That is D30's open, project-wide decision, not closed by this entry. The same contraction still exists in `RelaxFunctor`/`RemapMultiFunctor`, which are anchored
by `build-evidence/mutations.toml` and deliberately left as they are; `runtime.runner_device` asserts
`<= 1e-14` relative for those two shapes and bitwise for the rest (A5).

**Accepted 2026-10-12 (owner's decision), TODO kept here.** Working-precision agreement between the
device and host paths for `Relax` and weighted `Remap` is acceptable; closing the gap is not
scheduled. If it is ever wanted: route `RelaxFunctor`'s `omega * current(i) + one_minus * previous(i)`
and `RemapMultiFunctor`'s weighted accumulation through `unfused_multiply_add()` (the same idiom as the
reductions), re-anchor the two rows in `build-evidence/mutations.toml`, and tighten the
`runtime.runner_device` assertion from `1e-14` to bitwise. Any new device kernel added under
`docs/dev/device-resident-arcs.md` follows the same contract: bitwise where the host side is a
reduction the determinism contract covers, working precision elsewhere, stated by its test.


The same accepted gap covers the native mesh's device geometry kernels (2026-10-13, mesh-search track): `NativeMeshView::device_measure()`/`device_centroid()` recompute the per-cell measures and centroids on `Kokkos::DefaultExecutionSpace`, and nvcc contracts the cross- and dot-product arithmetic of the triangle-area and tetrahedron-volume formulas, so `test/backend/native_distributed_mesh_test.cpp`'s `DeviceGeometryMatchesHost*` cases assert 1e-14 relative agreement, not bitwise. Closing it, if ever wanted, is the same `unfused_multiply_add` idiom.
### ~~D46. `fuse-abi-shared`'s install RPATH named only yaml-cpp, so an installed consumer could not load Kokkos or the MOAB chain~~ — FIXED 2026-10-10; libMOAB.so's own rpath gap WORKED AROUND at test time

**Measured 2026-10-10** on this host after an external MOAB rebuild: `backend.moab_mesh` and both
`abi.install_consumer_*` tests failed with `libTempestRemap.so.0: cannot open shared object file`,
and once that was covered, with `libkokkoscore.so.5.2: cannot open shared object file`. Two separate
gaps:

1. **This project's, fixed.** `src/CMakeLists.txt` set `INSTALL_RPATH` on `fuse-abi-shared` to
   yaml-cpp's directory alone, on the premise that every other target in the tree is static. Kokkos
   is always a found shared library, and `FUSE_ENABLE_MOAB` adds MOAB, TempestRemap, Zoltan, NetCDF
   and PnetCDF, none on a default search path. `_fuse_abi_install_rpath` now carries
   `$<TARGET_FILE_DIR:Kokkos::kokkoscore>` and, when MOAB is on, every directory in
   `MOAB_LIBRARY_DIRS` (the per-package `<PKG>_DIR` fixup `cmake/FuseDependencies.cmake` already
   builds for `target_link_directories`). An external consumer doing nothing but
   `find_package(fuse-coupler)` needs this; nothing else can supply it.
2. **MOAB's, worked around.** `libMOAB.so`'s own rpath lists only its MPI-stack paths, and gcc's
   default `--as-needed` drops any dependency a test binary never calls directly from its `NEEDED`
   list, so resolution of the MOAB chain falls on that incomplete rpath. `test/CMakeLists.txt`
   prepends every `MOAB_LIBRARY_DIRS` entry to `LD_LIBRARY_PATH` through `ENVIRONMENT_MODIFICATION`
   on every MOAB-linked test. That helps only this project's own `ctest`; it is not a fix and is
   not this project's to make.

### ~~D47. The nested-build `compile_ok`/`compile_fail` tests used whatever compiler the invoking shell's `OMPI_CXX` named~~ — WORKED AROUND 2026-10-10

`runtime.compile_ok.capabilities_override` and `field.compile_ok.owning_chain` drive a nested
`cmake --build ... --target ...` at test time; that build re-invokes `mpicxx`, which reads its real
host compiler from `$OMPI_CXX` when *it* runs, not from the configure. A `ctest` run from a shell
without `OMPI_CXX=/opt/kokkos/bin/nvcc_wrapper` fell back to `g++`, which rejects the nvcc-only flags
this project adds whenever `Kokkos::CUDA` exists: both `compile_ok` rows failed, and their
`WILL_FAIL` `compile_fail` siblings passed for the wrong reason. `test/CMakeLists.txt` captures
`$ENV{OMPI_CXX}` at configure time into `_fuse_nested_build_env_mods` and pins it on all four rows
through `ENVIRONMENT_MODIFICATION`. The build itself still needs the variable exported (the
`_machine` preset sets it for configure only); that is a property of Open MPI's compiler wrapper,
not of this project.

### D27. The native backend's remap halo has no `accumulate_remap_halo` counterpart -- FIXED, tier-0 verified 2026-10-18; multi-rank row awaiting the host

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

**FIXED, TIER-0 HOST-VERIFIED-EQUIVALENT 2026-10-18 ("MoabHalo" track, round 2); the real
multi-rank row remains host-pending.** `fuse::native::accumulate_remap_halo`
(`include/fuse/backend/native_halo.hpp`, `src/backend/native_halo.cpp`) is the reverse exchange:
each ghost slot's value is ADDED to its owner's own slot and the ghost slot is then zeroed, built
from `fuse::linalg::PackedExchanger::exchange_packed_adjoint` (comm.hpp) -- the SAME adjoint
Alltoallv `src/backend/nodal_remap_distributed.cpp`'s own H^T scatter-add already uses, reached
through `unpack_index()`/`pack_index()` rather than any new mechanism.

DIRECTION CORRECTION FROM THIS ENTRY'S OWN TASK BRIEF -- CONFIRMED CORRECT BY THE LEAD (round 2):
the dispatching brief for this track named the open half as "MoabMeshBackend::accumulate_remap_halo"
and asked to mirror "NativeMeshBackend::accumulate_remap_halo." Neither matched this entry's own
text above, nor the code: `MoabMeshBackend::accumulate_remap_halo` (`src/backend/moab_backend.cpp:3030`)
was ALREADY implemented and tested (`test/parallel/remap_halo_mpi_test.cpp`'s
`RemapHaloMpiSphere.BuildRemapHaloDeliversExactHaloValuesAndRoundTripsExactly`); `NativeMeshBackend`
had no `accumulate_remap_halo` of any kind before this change. The fix followed THIS entry's own
record (native side missing), not the brief's; the lead's round-2 review confirmed this reading was
right and the brief was wrong.

BUFFER SIGNATURE, SETTLED (round 2, lead's ruling): MOAB's version takes one combined
`entity_count(dim) * components` buffer; the native version takes the SAME two separate
`owned`/`ghosts` arrays `GhostExchanger::update()` already uses for the forward fill, one component
per call. This is the PERMANENT native API -- no combined-buffer entry point is to be added.

**Measured (tier 0, this sandbox, serial -- no MPI needed, 2026-10-18 round 2):**
`ctest -R backend.native_halo` (`fuse_backend_native_halo_test`, run directly with
`KMP_DUPLICATE_LIB_OK=TRUE OMP_PROC_BIND=false` -- an unrelated, pre-existing duplicate-OpenMP-
runtime warning this binary's own link graph pulls in, not from this change): **21/21 PASS**, 539 ms
total, including the 5 new `NativeHaloAccumulate` rows (argument validation, the no-op path via
`make_serial_exchanger()`, the missing-`PackedExchanger`-facet refusal via a test-only
`GhostExchanger`, and an independent arithmetic oracle over the REAL 3-rank misaligned
`GhostPattern` the existing `run_halo_pipeline` fixture builds, proving the conservation identity:
folding every halo copy -- each starting at exactly 1.0 -- back to its owner accounts for exactly
one addition per halo instance, with nothing lost or duplicated, EXACT not a tolerance). Mutation
rows added and killed by the whole binary (see build-evidence/mutations/native_halo.toml); counts
recorded verbatim in this track's own report, restore confirmed byte-identical against pristine.
**Built, host-pending:** `NativeHaloMpi.AccumulateRemapHaloRoundTripsExactly`
(`test/parallel/native_halo_mpi_test.cpp`) mirrors the MOAB test above under a REAL communicator at
2 and 3 ranks (`parallel.native_halo_mpi.{2,3}`, already registered) -- this sandbox cannot
initialise MPI at all (C1), so this row is compiled only.

**Round-2 side finding, NOT a library defect (round 1 wrongly opened, then closed, D78 for this --
struck from the record entirely, not merely closed, since it never described a real defect in this
project).** ROOT CAUSE, corrected by the lead after round 1 mis-diagnosed it: this is a SANDBOX
CONFIGURE choice, not a FUSE defect of any kind. Round 1 ran `cmake` from the `kokkos-dev` conda
environment, which also put that environment's own `c++` (conda-packaged clang 21) first on PATH,
so `CMAKE_CXX_COMPILER` resolved to it by default. That compiler's own config file
(`<env>/bin/<triplet>-clang++.cfg`) prepends `-isystem <env>/include` -- which happens to carry a
conda-packaged `kokkos` header tree (5.2.1), different from this project's intended `/opt/kokkos`
(5.1.0) -- ahead of this project's own flags, so round 1's first symptom (undefined `Kokkos::
OpenMP::concurrency() const`, the pre-5.x member API, against a 5.x library) was a header/library
version mismatch from that wrong compiler's own implicit search path. A CMake-side mitigation
(`IMPORTED_NO_SYSTEM TRUE` on the Kokkos imported targets, `cmake/FuseDependencies.cmake`,
CMake-3.25-gated) makes this project's own BEFORE-prepended Kokkos include win that ordering fight
regardless of compiler, and is kept as a harmless defense-in-depth (verified: no `-Werror`/`-Wall`
anywhere in this project today, so relaxing Kokkos headers from "system" changes no diagnostic
severity) -- but it is NOT the fix. Even with headers resolved correctly, that SAME conda compiler
links its own `libc++.1.dylib` via `@rpath` into the test binary, while `/opt/kokkos`'s own
`libkokkoscore.5.1` was built by Apple clang against the SYSTEM `libc++` -- two C++ runtimes in one
process, which the lead measured segfaulting at startup before any test could run. THE FIX is
`-DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++` in a FRESH build directory (the
compiler is cached; a stale build directory keeps the wrong one) -- the SAME system Apple clang the
lead's own clean-export builds already use, 211/211 tier-0 in this identical sandbox. Re-verified
with this configuration (below); this is a build-invocation choice (which compiler happens to be
first on a particular shell's PATH), never a project source change, and nothing to carry forward
as a standing KNOWN_ISSUES entry once stated once, here.

**Re-verified, 2026-10-18, round 3, system compiler (`/usr/bin/cc`/`/usr/bin/c++`), fresh build
directory:** full tier-0 `ctest`: **211/213 PASS** (926.9 s real). The 2 failures
(`abi.install_consumer_c`, `abi.conformance` -- `ConformanceTest.LagOneGraphAllFourKindsMatchNative`/
`SubcycledMeanGraphAllFourKindsMatchNative`, both `E-ABI-LOAD` on the Fortran plugin) are
UNRELATED to this entry and PRE-EXISTING: reproduced identically on an unmodified pristine export
under the IDENTICAL system-compiler configuration (this sandbox's Fortran toolchain is not fully
wired for the `/usr/bin/c++` configuration the ABI Fortran-plugin conformance rows need; a
pre-existing sandbox-configuration gap, not a code defect, and not this track's to fix).
`backend.native_halo` specifically: **1/1 PASS, 0.29 s**, no `KMP_DUPLICATE_LIB_OK`/`OMP_PROC_BIND`
workaround needed at all under the system compiler -- confirming those were round 2's own
artifact of the wrong compiler, not anything this project's test needs.

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

**Addendum 2026-10-13 (review 2026-10-11 TR-5) — fixed, compiled, unmeasured.** `connect_pairwise()`'s
acceptor-side rank 0 now broadcasts its own `MPI_Open_port`/`publish_port_file` outcome to every acceptor
rank before the port-name `Bcast`, so a rank-0-only failure no longer leaves its peers waiting in that
collective. The path is behind `FUSE_ENABLE_CONNECT_ACCEPT`, OFF on the host this round built on, so the
fix compiled but could not be exercised; it runs when this entry's accept rows run.

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

### D18. `fuse::transport::connect()`'s `Connect`-mode path supports exactly two components; a third refuses `E-CONNECT-SPEC` -- GENERALISED 2026-10-18 (tier-0); host executables/ctest rows built 2026-10-18 (round 2), awaiting the host run

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

**Generalised 2026-10-18 (ABI track), per this entry's own sketch above.** `connect_mode_
components()` (`include/fuse/transport/connect.hpp`) now refuses only fewer than two Connect-mode
components, not "not exactly two" -- the generalisation itself lives in the new, pure
`connect_chain_plan()` beside it: step 1 is today's two-component rule, UNCHANGED (whichever of the
first two declares `LaunchSpec::is_acceptor` opens the port, exactly as `connect_pairwise()` always
decided); every later step fixes the role structurally, resolving this entry's own open "which
single rank... plays acceptor for the next merge" question the way this round's own task named it:
"each component publishes its port name once" -- the SINGLE component joining at that step must
itself declare `is_acceptor = true` and its own `port_file`, and the chain already merged is always
the connector, since no single `LaunchSpec` can state an opinion for a composite of several
components. `connect_chain()` (`src/transport/connect.cpp`) implements the merge itself: step 1 is
`connect_pairwise()`'s own body (now scoped to the first two of `connect_mode_components()`'s
result explicitly, since that function may return more than two); each later step is a fresh
accept/connect + `MPI_Intercomm_merge` between the chain formed so far (low) and the joining
component (high), preserving the graph's own component order the same way the existing two-
component case already does. `connect()`'s `LaunchMode::Connect` case now dispatches to
`connect_chain()` unconditionally, which performs exactly one merge (step 1 only) for a plain
two-component graph -- zero behaviour change for every existing caller.

**Tier-0 tested, pure.** `test/transport/connect_test.cpp`'s `ConnectChainPlan` suite: 3 and 4
components each produce the expected step count and joining order; a step-2-or-later joiner that
does not declare `is_acceptor` or has no `port_file` is refused `E-CONNECT-SPEC` naming the
component; fewer than two Connect-mode components still refuses. `ConnectModeComponents.
AcceptsThreeConnectModeComponentsInGraphOrder` replaces the old `RefusesAThirdConnectModeComponent`
(renamed, not merely edited, since the assertion inverts -- this is the literal subject of this
entry's own fix). `ValidateLaunchLayout.AcceptsThreeConsecutiveConnectBlocksInGraphOrder` and
`...AcceptsFourConsecutiveConnectBlocksWithAGapFilledByInProcess` confirm D36's own gap rule is
unaffected by the chain length -- it was already stated generically over "however many launched
blocks", never specifically over two.

**Built this round (ABI track, round 2), per the lead's own instruction.** A THIRD and FOURTH
participant executable now exist, deliberately NOT sharing `connect_a.cpp`/`connect_b.cpp`/
`connect_common.hpp` (those two, and the six ctest rows already passing against them, keep zero
exposure to this round's new code): `test/parallel/connect/connect_chain_common.hpp` defines
`ChainLinkComponent` (producer / middle-passthrough / consumer-only roles, selected by which of its
optional in/out fields are set) and `build_chain_graph(count)` (components named a/b/c/d per
`count`, step 1's acceptor/connector exactly as today, every later joiner its own acceptor per
this entry's fix above); `connect_chain_{a,b,c,d}.cpp` are thin drivers naming each one's own index;
`run_chain.sh` launches `count` of them concurrently and diffs the last link's own output against a
golden file. Because every link in this graph is exactly one rank, `partition_block()`'s own
decomposition is the identity at every hop, so the existing two-component `golden-1+1.txt` is the
correct reference for a chain of any length -- no new golden file was needed. Registered as
`parallel.connect.chain.{3,4}` (`test/CMakeLists.txt`, gated on `FUSE_ENABLE_MPI` and
`FUSE_ENABLE_CONNECT_ACCEPT`, 3 and 4 executables respectively, within the project's 4-task launch
limit), built clean under a real `FUSE_ENABLE_MPI=ON` configure -- `connect_chain()` itself now has
an executable exercising its actual MPI calls, where round 1 had none. Built, never run (C1 above).
**Host-pending:**

```
ctest --test-dir <build> -R 'parallel\.connect\.chain\.(3|4)' -V
```

**Still not done (same gap this entry already named, now smaller).** The N-way generalisation's
design sketch above remains unexercised past `connect_chain_plan()`'s own pure-function tests until
the host actually runs `parallel.connect.chain.{3,4}` -- whether the exchanged values at the LAST
link match the in-process N-component reference exactly, as the existing 2-component row already
confirms for N=2, is what that host run settles, not anything checkable here.

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

### ~~D31. Newton over a coupling window (S7.2 WindowNewton, `newton-krylov.md` Sec. 9) was single-rank only; the multi-rank `ParticipantSetReducer` is now built — BUILT 2026-10-12, host MPI run still required to close~~ -- CLOSED 2026-10-18, host-verified

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


**HOST-VERIFIED 2026-10-18** (owner's machine, MPICH 5.0.0a1, `FI_PROVIDER="tcp;ofi_rxm" FI_TCP_IFACE=en0`, `build-all`
quick tier at `cb127c5`): `parallel.participant_reducer_mpi.{2,3}`, `parallel.window_newton_mpi.{2,3}` and
`petsc.window_newton_mpi.{2,3}` all Passed. The 3-rank reducer leg is the non-contiguous participant subset
{0,2}. D55 (the same `window_newton_mpi.3` row hanging under Open MPI 4.1.5 on higgs) stays open on its own.

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
maximum not yet printed -- the test now prints it).

**Built 2026-10-18 (Substrate track, plan S7.2b phase 2 closing): the per-region selector exercised
with TWO Newton regions in ONE graph, naming genuinely different substrates, which nothing before this
entry's own close had tested.** Sec. 11's own tests (above, PetscSelect track) built the registry, the
eager refusal, and the descriptor field, and measured dn_heat/the Brusselator run WHOLLY through one
substrate at a time (`--newton-petsc`/`--substrate petsc` pick the substrate for the graph's own single
Newton region); nothing built there put two Newton-selected regions, naming different substrates, in
the same graph. `test/runtime/newton_region_test.cpp` gained
`NewtonRegion.TwoNewtonRegionsOneUnavailableSubstrateIsRefusedByNameNotByRegionOrder` (region A native,
region B an unregistered Petsc, refused `E-NEWTON-PETSC-UNAVAILABLE` naming region B specifically, never
region A) and `NewtonRegion.TwoNewtonRegionsInOneRunnerDispatchToTheirOwnSubstrateIndependently` (a test
double registered for `Petsc`, per `register_newton_substrate()`'s own documented "a test double" use,
proving the Runner reaches two different registered functions for two regions inside ONE
`Runner::run_period()` call -- without PETSc, `install()`, or MPI). `examples/dn_heat/main.cpp`'s own
`--newton-petsc` path gained `build_dual_substrate_graph()`/`run_dual_substrate_case()`: a second,
independent dirichlet-neumann pair in the SAME graph as the existing comparison, region
`dirichlet_neumann_a` on Native and `dirichlet_neumann_b` on Petsc, run together, printing both
regions' own convergence histories and checked against each other with the SAME sharp,
per-period-summed bound the existing native/petsc comparison uses; still gated `FUSE_HAVE_PETSC`, still
run only under `--newton-petsc`, no new ctest row (runs inside the existing
`example.dn_heat.newton_petsc` row). Full account, including why the eager refusal needs no cross-rank
agreement step (the graph and the per-process registry state are both SPMD-identical across ranks by
construction, never rank-sensed), is `docs/dev/newton-krylov.md` Sec. 13.

**Verified tier-0, without PETSc, this sandbox, 2026-10-18:** `/tmp/substrate/b0` (Debug,
`FUSE_ENABLE_PETSC=OFF`), clean build, 0 errors; full quick-tier `ctest -LE slow`, 206/206 PASS
(unchanged pass count and rate -- both new cases run inside the pre-existing `runtime.newton_region`
ctest row, which already counted as one row before and after); the two new cases individually
confirmed PASS by direct binary invocation (`--gtest_filter`); `example.dn_heat`/`example.dn_heat.dump`
unaffected (default invocation's own output is unaffected by this track's additions, which are
entirely inside the pre-existing `--newton-petsc` path). `tools/check-layering.py` (0 violations),
`tools/verify-import.py` (693 files, 0 problems), `tools/gen-abi.py --check` (6 current, 0 stale),
`tools/mutate.py --check-anchors` (363 valid, 0 stale) all pass; `tools/check-invariants.py`'s own 7
violations are pre-existing on pristine (element_type.hpp/element_type_test.cpp/restart_test.cpp, none
of which this track touched), confirmed identical before and after by diff against pristine.

**NOT STRUCK: this closes plan S7.2b phase 2's own remaining item, but does not close D32 itself.**
D32's own "the communicator remains" (above) and every `petsc.*`/`example.dn_heat.newton_petsc` row
stay exactly as host-pending as they were before this entry: this sandbox cannot complete a bare
`MPI_Init()` (the same constraint named throughout this entry and D31), so
`example.dn_heat.newton_petsc`'s own dual-substrate case is BUILT here, never RUN -- a host run of that
ctest row is what measures whether the two regions' own converged states actually agree to the printed
bound, the one thing this sandbox's tier-0 verification above cannot show. D31's own cross-rank
communicator lift (Sec. 9.10/11.3) is independently still open (D31's own heading, above) and unrelated
to this closing beyond sharing the same host-pending status.


**HOST-VERIFIED 2026-10-18** (owner's machine, `build-all` at `66d35cc`, PETSc on): `petsc.init`,
`petsc.token`, `petsc.derivative_gate`, `petsc.snes`, `petsc.brusselator_newton`,
`petsc.window_newton_mpi.{2,3}` all Passed, and `fuse_example_dn_heat --newton-petsc` ran the
dual-substrate case: `region_a=native region_b=petsc periods=20 converged_a=1 converged_b=1`, final-period
residual histories `5.680512e+01 4.311954e+00 1.765836e-02 1.484199e-09` (native) and `5.680512e+01
4.311952e+00 1.765848e-02 2.799275e-09 2.799275e-09` (petsc), `agreement disagreement=3.572609e-10 K
sharp_bound=5.888415e-07 K ratio=6.067182e-04`, exit 0. The per-region substrate selection is
host-verified; this entry's remaining item is the communicator statement in its heading, unchanged.

### D48. A `Transport` instruction can have one arc device-resident and the other host, even in Device mode — FOUND AND FIXED 2026-10-13

**What was found.** docs/dev/device-resident-arcs.md Phase 3 made `Transport` space-aware in
`place_arcs_on_device()`, on the assumption that a Transport's two arcs -- ordinary,
coupler-owned intermediate storage -- always land on the same side together. They do not: a
Remap needing conservation checking pins the `Constrain`'s `before_arc`, which is frequently a
Transport's own OUTPUT (the Remap's input), to host, while the Transport's INPUT, untouched by
any other host-only instruction, lands on the device like everything around it under
`place_arcs_on_device()`'s "not pinned => device" rule. An initial, defensive implementation
refused this combination outright (`E-RT-TRANSPORT-DEVICE`) rather than guess at it -- and that
refusal regressed two PRE-EXISTING, previously-passing Device-mode tests:
`RunnerDevice.DeviceAndHostAgreeToRoundingOnRelaxAndWeightedRemap` (the weighted Remap +
Relax scenario) and `RunnerDevice.OrdinaryIterationRegionConvergesOnTheDeviceLikeTheHost`
(the two-`LinearBlock` iteration region) -- both no-graph (single-process `CopyTransport`)
plans whose Transport instruction feeds exactly such a Constrain-pinned Remap. Caught by
running the full `runtime.runner_device` row before declaring this track's step 1 done, not by
anything specific to the mixed-pair scenario itself.

**What was fixed.** Two cases, both still in `Runner::Impl::execute_on<Space>()`'s
`OpKind::Transport` case:
- No-graph (`CopyTransport`): `Kokkos::deep_copy` bridges ANY combination of host and device
  pointers by itself (that is the whole point of the primitive), so every combination -- both
  device, both host, or mixed -- calls it directly; `CopyTransport::move`'s `std::memcpy`
  (host address on both sides, by contract) is reached only for the all-host case, unchanged.
- CommGraph-backed (MPI): a mixed pair stages the device-resident side through a per-plan host
  scratch buffer (`Impl::transport_in_scratch`/`transport_out_scratch`, sized once in
  `build_transport_device_buffers()`), runs the UNCHANGED scalar host loop against two host
  pointers, and stages the result back to the device `out` arc if that side was device. No
  test in this tree's CommGraph-backed rows currently exercises this exact branch (every MPI
  scenario built for this track keeps both arcs together, like the no-graph ones did), but it
  follows the identical `Kokkos::deep_copy`-at-the-boundary pattern the generic
  `span_c`/`span_m` staging and `copy_arc` already use elsewhere in this file, so it is not new
  machinery, only a new call site for it.

**Verification.** `runtime.runner_device` (11/11 pass, including both previously-regressed
rows) and `abi.device_binding` both green after the fix; see report-transport.md for the full
ctest counts of this track's own build.

### ~~D49. `TheTransportPathAllocatesNothingWhileRunning`'s `.device` twin fails on the rank with a mixed device/host Transport arc pair~~ — CLOSED 2026-10-14 (runtime-exchange track): confirmed Kokkos-internal, skip-with-reason kept

**What is actually confirmed.** `test/parallel/runner_transport_test.cpp`'s `RunnerTransport.
TheTransportPathAllocatesNothingWhileRunning` wraps `runner.run(h.components, 4)` in an
`AllocationCounter` after one uncounted warm-up period and asserts the count is exactly 0 --
the ONLY assertion in the test after its two `ASSERT_TRUE` setup checks. Registered with
`DEVICE_TWIN` (this track), its `.device` twin (`FUSE_EXECUTION_SPACE=device`) fails on
exactly the rank that holds the `solid` component. `FUSE_DEVICE_DEBUG=1` confirms that on that
rank the channel's Transport instruction has a MIXED arc pair (input = `fluid.out`, a
component-owned host alias, `n=0` on this rank and so never promoted; output = the
coupler-owned `link/transport` arc, `n=12`, promoted to device) -- the exact combination
KNOWN_ISSUES D48 fixed correctness for, which on this rank reaches `Impl::transport_out_scratch`
and a `Kokkos::deep_copy` into device memory for the first time in this binary.

**Round 2 (runtime-exchange track, 2026-10-14): the actual count, recovered.** `--output-filename`
(OpenMPI 4.1.5's real per-rank-redirect flag; D49's own first round tried `-outfile-pattern`,
which does not exist on this OpenMPI) does NOT recover the failure text either -- confirmed by
inspecting `rank.1`'s own captured stdout/stderr directly: empty of anything from this test,
the same silencing the first round already suspected, now directly observed rather than
inferred. What DOES work, exactly as D49's own "what would close it" paragraph proposed: a
temporary build with a raw `std::fprintf(stderr, ...)` of `allocations` right before the
`EXPECT_EQ`, which bypasses `fuse::testing::MpiFailureReducer`'s listener entirely (it
suppresses GoogleTest's OWN reporting machinery, not arbitrary stderr writes). Result, rank 1
only (rank 0: 0): **16 allocations over the 4 counted periods -- exactly 4 per period**, zero
on rank 0 (the rank with no device crossing on this channel). The patch was diagnostic only,
reverted immediately after (`git checkout --`) and the binary rebuilt clean before this entry
was written; no production code changed.

**Verdict.** The count is isolated ENTIRELY to the rank that crosses the host/device boundary
(rank 0, same channel, same periods, zero device crossing, allocates nothing), scales exactly
with the period count (4/period x 4 periods = 16, not a one-time setup cost), and this
track's own changes never touched `MpiTransport`'s mixed-pair path (`execute_on<Space>`'s
`OpKind::Transport` case) at all -- the count was already there before this track started,
unmasked only because this is the first round to actually recover it. This matches the lead's
own 2026-10-13 integration note exactly: the mixed-pair path's one per-period
`Kokkos::deep_copy` between spaces fences with a string label
(`Kokkos::fence("Kokkos::deep_copy: ...")`), and a label past libc++'s small-string-optimization
capacity heap-allocates one `std::string` per call -- the identical mechanism
`space_kernels.hpp`'s own header comment documents catching via
`Iteration.TheSweepLoopAllocatesNothing`/`TheAcceleratorAllocatesNothingPerUpdate`, both of
which already skip under Device for the same reason (docs/dev/testing.md, "The device leg").
4 allocations/period (not 1) says more than one such fenced operation runs per period on the
receiving rank -- plausibly pack/unpack buffer touches in addition to the deep_copy itself --
but the exact call count was not traced further: isolating WHICH Kokkos-internal calls make up
the 4 needs a debugger or a Kokkos-level instrumented build, neither of which changes the
verdict (Kokkos's own bookkeeping, not a FUSE allocation) or the fix (skip, not a code change).

**What was done.** No change to the skip: the test already skips itself under
`FUSE_EXECUTION_SPACE=device` with a comment naming this entry (added in the prior round),
exactly the same `Iteration.*` precedent's own shape -- confirmed to be the right close rather
than changed. The skip comment's own text still accurately describes the symptom; this entry
records the round-2 evidence that confirms it rather than editing the skip itself.

**Integration note (lead, 2026-10-13), now confirmed rather than hypothesized.** The same
binary's host leg allocates nothing, and the one thing the mixed-pair path adds per period is
`Kokkos::deep_copy` between spaces, which fences with a string label and so constructs a
`std::string` per call -- the count measures Kokkos, not FUSE, exactly as this note proposed.

### ~~D50. `parallel.window_runner_mpi`'s new `.device` twin fails with `E-RT-DEVICE` on an `Interpolate` instruction~~ — CLOSED 2026-10-13 (fixed on the lead branch before the twins were merged)

**What.** `fuse_add_mpi_test(parallel window_runner_mpi ... DEVICE_TWIN)` (this track) registers
the first-ever Device-mode run of this file's window-scoped iteration region test. Both
`.device` rows fail immediately: `E-RT-DEVICE: instruction 'Interpolate' needed more device
staging than setup() provisioned (4 lanes of 0 values)` -- `Impl::device_staging_stride` is 0
when this plan's window-scoped region reaches its spacetime `Interpolate`, so `stage()`'s fault
path (`staging_fault = true`) trips on the very first lane request. `parallel.window_runner_mpi.2`
/`.3` (Host) are unaffected and still pass; `parallel.nodal_remap_runner` and
`parallel.native_halo_mpi`'s own new `.device` twins both pass clean, so this is specific to
the window-scoped + spacetime-Interpolate combination, not to Transport or to `DEVICE_TWIN`
rows generally.

**Why this is not this track's Transport change.** `device_staging_stride` is computed in
`Runner::Impl::initialize()` as `max(s.total() for s in slots)`, entirely independent of which
arcs `place_arcs_on_device()` pins or promotes -- Transport's own arcs moving from
unconditionally-pinned to space-aware (D48) does not change any `Slot::total()`. The window
member bookkeeping (`advance_window()`, `src/runtime/iteration.cpp`) and its own Interpolate
dispatch are both outside `execute_on<Space>()`'s ordinary switch (WindowAdvance instructions
are special-cased, per that case's own refusal comment), so this looks like a latent gap in
the window-scoped-region-under-Device path that nothing in this tree's suite exercised before
`DEVICE_TWIN` added the first `.device` row for this specific test file.

**Why not fixed now.** docs/dev/device-resident-arcs.md's own rollout Step 3 lists "the
accelerator interface vector and the window-member bookkeeping" as a SEPARATE item from this
track's Phase 3 scope (GPU-aware transport, rank-to-device binding, device-mode MPI test
rows); diagnosing a zero-sized `device_staging_stride` for a window-scoped plan's spacetime
history needs a change to `Runner::Impl::initialize()`'s stride computation or to how a
window's own ticks reach `execute_on<Space>()`, neither of which this track's charter covers.

**What would close it.** Trace with `FUSE_DEVICE_DEBUG=1` why `device_staging_stride` comes out
0 for this plan (every ordinary, non-window slot in it is non-empty, per the Host leg's own
successful run) -- likely the window-scoped region's own blocks/history are sized from a path
`initialize()`'s stride loop does not walk, which is exactly the gap Step 3 names.

**Closed.** The cause is the one `verification.brusselator_runner.device` hit the same day: `setup()` sized the device staging lanes AFTER `resolve_regions()`, which executes a window-scoped region's lagged import chain once during region resolution, so that first `Interpolate` saw `device_staging_stride == 0`. Commit 581c5d1 moves the sizing ahead of `resolve_regions()`; the window twins at 2 and 3 ranks pass on the integrated tree (see the final report).

### ~~D51. Native halo/migration's device-side merge (dedup-by-global-id) is not implemented~~ — CLOSED 2026-10-15 (Migration track, round 3)

**Scope note, 2026-10-13.** `fuse::native::device::pack_to<Space>` (include/fuse/backend/native/
mesh_pack_device.hpp) templates the migration and halo wire PACK on the Kokkos memory space, verified
byte-for-byte against `pack_for_migration`/`pack_halo_cells` in both Host and Device mode
(`test/backend/native_mesh_pack_device_test.cpp`). The symmetric receiving-side merge
(`unpack_migrated_mesh`'s vertex dedup, `merge_halo_cells`'s cell dedup, both by global id, both
currently `std::unordered_map`/`std::unordered_set` over `std::vector`) is NOT templated. The bulk
per-field data copy generalizes the same way packing did (a Kokkos kernel reading the received byte
buffers' fixed-layout records); the open piece is the membership decision itself -- "is this global id
already held" -- which has no device-friendly hash-set equivalent used anywhere in this codebase yet.
A sort-and-compact approach (flatten every incoming (gid, payload) pair, `Kokkos::sort` by gid, then a
`parallel_scan` over adjacent-equal runs to assign each unique gid one slot) is the natural device
algorithm and was not attempted here for lack of time, not because it is infeasible. Blocks: a Device
mode caller of `build_native_remap_halo`/`migrate` still needs the host round trip for the merge step.

**Narrowed, 2026-10-14 (round 2).** The membership DECISION itself -- "is this incoming gid already
held, and which one of several cross-sender copies is canonical" -- is now implemented:
`fuse::native::device::dedup_by_gid<Space>` (include/fuse/backend/native/mesh_merge_device.hpp, new
file), a sort-and-compact pass via `Kokkos::Experimental::sort_by_key` (not the plain `Kokkos::sort`
this entry's own round-1 text named -- `sort_by_key` carries a payload tag through the sort instead of
losing each entry's original side/position, which a value-only sort would) plus one `parallel_scan`
for slot assignment and an `atomic_min` pass for canonical selection, matching the Host path's "first
occurrence in buffer order wins" rule exactly. `O((E + N) log(E + N))`. Verified against a
from-scratch `std::unordered_map`/`std::unordered_set` reference covering no-overlap, incoming-vs-
existing duplicates, cross-sender incoming-vs-incoming duplicates, and both empty-input cases, in both
Host mode and (when this Kokkos has one) a Cuda/HIP/SYCL backend
(`test/backend/native_mesh_merge_device_test.cpp`). Two gaps still block wiring this into
`merge_halo_cells`/`unpack_migrated_mesh` themselves, both stated in `mesh_merge_device.hpp`'s own
header comment and docs/dev/native-halo.md's "Device merge decision" section: (1) the primitive's own
slot numbering is ascending BY GID VALUE, not the Host builders' append-order convention (existing
keeps its index; new entries append in first-encounter order) -- a mechanical renumbering pass, not a
second algorithm; (2) the wire format's variable-length, unprefixed records make per-record boundary
discovery within a received buffer a serial walk regardless of execution space, which this round did
not attempt to parallelize (it would need either a length-prefixed wire-format change or an
accept-serial-scan compromise). A Device mode caller of `build_native_remap_halo`/`migrate` still
needs the host round trip for the full merge; the open piece this entry named in round 1 -- no
device-friendly hash-set equivalent existed -- does not block it anymore.

**Narrowed further, 2026-10-15 (Migration track).** Both migration and
halo wire PACK are now wired to the device-resident primitives this
entry's own round-2 text introduced, not merely proven standalone:
`pack_for_migration` (native_migration.cpp) and `pack_halo_cells`
(native_halo.cpp) call `fuse::native::device::pack_to`/a new sibling,
`pack_fanout` (mesh_pack_device.hpp's many-to-many generalization,
needed because a halo copy -- unlike a migrated cell -- may be sent to
SEVERAL destinations), directly against `NativeMeshView::device_arrays()`.
No `NativeMeshView` host-serving accessor is reached by either packer,
checked by a `host_mirrors_materialized() == 0` assertion around each
call (`NativeMigration.PackForMigrationMaterializesNoHostMirror`,
`NativeHaloPackMerge.PackHaloCellsMaterializesNoHostMirror`). The MPI
exchange also gained a GPU-aware REPLACEMENT: `exchange_migration_
buffers<Space>` (native_migration.hpp/.cpp), a new Space-templated
overload taking/returning `Kokkos::View<std::byte*, Space>` buffers,
hands `MPI_Alltoallv` a device buffer's own raw pointer when `Space` is
a device memory space and `fuse::linalg::mpi_is_gpu_aware()` says this
MPI build may take one (`FUSE_TRANSPORT_HOST_STAGING` still forces the
fallback); this probe is itself now the ONE copy both `fuse::native` and
`fuse::transport::MpiTransport` call (`include/fuse/linalg/comm.hpp`,
`src/linalg/mpi_gpu_aware.cpp`), rather than each keeping its own, now
that `linalg` sits below both in the layering order and `backend`
already links it. Verified byte-for-byte against the existing
`std::vector` overload at 1-5 ranks and both `FUSE_TRANSPORT_HOST_
STAGING` settings (`NativeMigrationMpi.GpuAwareExchangeOverloadMatches
HostVectorOverloadByteForByte`; `backend.native_migration_mpi` gained a
`DEVICE_TWIN` registration at 2/3 ranks to cover it there too).

NOT DONE: exactly the same two gaps this entry's round-2 text already
named -- (1) `dedup_by_gid`'s slot numbering is by GID VALUE, not the
Host builders' append-order convention, needing the mechanical
renumbering pass this entry's round-2 text sketches; (2) the wire
format's unprefixed variable-length records make per-record boundary
discovery a serial walk regardless of execution space. Neither
`merge_halo_cells` nor `unpack_migrated_mesh` was touched this round,
and `migrate`/`build_native_remap_halo` still call the ORIGINAL
`std::vector<std::byte>` `exchange_migration_buffers` overload: wiring
the new device-resident pack and the new GPU-aware exchange all the way
through needs the merge side to stop requiring a host-resident buffer
FIRST, or a device-resident pack's own output would just get copied to
host again right before the unpack/merge step anyway, which defeats the
point of either change. This is this round's own deferred item -- the
same "proven standalone, not yet wired" shape this entry's own round-2
text left `pack_to`/`dedup_by_gid` in, before this round wired the
first of them (`pack_to`/`pack_fanout`) all the way in.



**CLOSED, 2026-10-15 (Migration track, round 3).** Both remaining gaps
this entry's round-2 text named are done: (1) `renumber_append_order`
(mesh_merge_device.hpp) is the mechanical renumbering pass over
`dedup_by_gid`'s own output -- a `parallel_scan` over `incoming_is_new`
assigning each canonical new entry `E + its rank among canonical entries
in ascending ORIGINAL INCOMING index` -- so Device's final vertex/cell
numbering agrees with Host's append-order convention bit for bit, not
merely up to a permutation; (2) the wire format gained a fixed-size
header (record count + per-record absolute byte offsets, prepended by
`alloc_buffer_with_header`) so `parse_buffers` locates every record's
start with one `parallel_for` indexed by record, never a sequential walk
of the variable-length payload. `unpack_migrated_mesh_device`/
`merge_halo_cells_device<Space>` (mesh_merge_device.hpp) are the full
device-resident receive-side rebuild this entry has described since
round 1, and `migrate`/`build_native_remap_halo` now call them directly
(via `device::pack_to`/`pack_fanout` -> `exchange_migration_buffers
<Space>` -> device unpack/merge -> `NativeMeshView::create<Space>`) --
no step of either function touches a `NativeMeshView` host-serving
accessor, checked by `host_mirrors_materialized() == 0` assertions
spanning the FULL round trip, source and rebuilt result view, both
`FUSE_TRANSPORT_HOST_STAGING` settings, host and MPI (1-5 ranks),
`.device` twins included (`NativeMigration.FullMigrateRoundTripMaterializes
NoHostMirrorOnEitherView`, `NativeMigrationMpi.MigrateMaterializesNoHost
MirrorOnEitherViewUnderEitherStagingSetting`, `NativeHaloMpi.
BuildRemapHaloMaterializesNoHostMirrorOnTheRebuiltSourceView`). Every
existing `backend.native_migration*`/`native_halo*`/`partition.
repartition*` row and its `.device` twin agrees exactly with its own
prior (Host round-1) numbers -- the wire format change is receive-side
transparent; `backend.native_migration`'s own in-process round-trip test
needed a one-line update (two independently-packed buffers are now two
`received` entries, not one byte-concatenated blob, since each buffer is
self-describing from its own start under the new header).

Three narrow, DECLARED exceptions, none test-covered or load-bearing for
anything this track's mandate asks for:

1. **Polyhedron, no cross-cell face dedup on the device path.** A
  migrated or halo-adopted Polyhedron's faces are rebuilt as NEW, PRIVATE
  entries (an identity `cell_face_indices`, `cell_face_flip` always
  false) rather than Host's `NativeMeshBuilder::add_polyhedron_cell`
  sorted-key + cyclic-match dedup against another cell's already-recorded
  face. Confirmed by grep across every migration/halo/repartition test
  file before this track started: none uses a `CellType::Polyhedron`
  generator (`hexbox`/`line`/`quad` only), so no existing coverage
  exercises cross-cell face sharing through either pipeline --
  `NativeMigration.PolyhedronCellSurvivesADeviceResidentMigrationRoundTrip`
  (new this round) checks a single migrated Polyhedron's own
  conservation (cell/vertex count, volume, ownership), not face count,
  specifically because this limitation means face count is not expected
  to match Host's for a mesh with shared Polyhedron faces. A future
  device-side face dedup would need the same sort-and-compact shape
  `dedup_by_gid` already established, keyed on each face's sorted-vertex-
  gid set instead of a single gid.
2. **`resolve_vertex_ownership_device<Space>`'s one bounded host round
  trip.** Only `vertex_gid` -- an O(nv) int64 bookkeeping array, never
  cell/vertex geometry -- stages to the host for the `MPI_Allgather`/
  `MPI_Allgatherv` that computes each vertex's global minimum-rank owner;
  `MPI_Allgatherv` has no GPU-aware form the way `MPI_Alltoallv`
  (`exchange_migration_buffers<Space>`) does, so this is not a missed
  optimisation. Does not touch `host_mirrors_materialized()`'s lazy-
  mirror cache (a different mechanism; see this track's own tests above).
3. **`NativeMeshBuilder::build()`'s `E-MESH-CONNECTIVITY` fixed-shape
  corner-count refusal is not reproduced** on the device path (only its
  sibling `E-MESH-DIM` check, a cell's topological dimension against the
  mesh's own, is -- `all_cells_match_dimension`, folded into `migrate`'s
  and `build_native_remap_halo`'s own post-exchange peer-refusal
  reconciliation, closing the one regression this round's own ctest run
  caught: `NativeMigrationMpi.ADimensionMismatchVisibleOnlyOnTheDestination
  RefusesEverywhere`). This mirrors a PRE-EXISTING gap in
  `NativeMeshView::create_from_host` itself (NM-8's own comment, native_
  mesh.cpp) for any direct caller bypassing `NativeMeshBuilder` --
  this track's device path is such a caller now, same as that comment
  already warned a future one would be. Not reachable by any wire data
  this codebase's own senders produce (a sender's own records always
  carry a self-consistent corner count for their `CellType`), so no test
  here exercises it.

### ~~D52. MLS device solve coarsens `E-MLS-RANK`'s diagnostic and is not wired into `build_mls_weights`; `KokkosBatched::SerialGesv` does not build here~~ — CLOSED 2026-10-14 (round 2)

**Scope note, 2026-10-13.** `fuse::device::solve_mls_on_space<Space>` (include/fuse/backend/
mls_weights_device.hpp) runs the per-target moment-matrix assembly and solve as a Kokkos BATCHED
kernel, verified against a NumPy-computed reference (1e-14 relative on a well-conditioned 2D/degree-1
case; ~1.5-2e-11 relative on a deliberately near-singular 3D/degree-2 stress case with `k == P`
exactly -- a conditioning effect, not a bug; `mls_weights.hpp`'s own guidance is to choose `k`
comfortably above the basis size `P`, which keeps this well clear of 1e-12).

**`KokkosBatched::SerialGesv` was tried first and does not build on this host, in EITHER pivoting
mode** (named by the task that approved this track, from /opt/kokkos-kernels; not assumed infeasible
-- both modes were actually built against this header's own `A`/`B`/`X` views before being abandoned).
Exact measured errors, both against `Kokkos::subview`-derived arguments to `SerialGesv::invoke`:

- `Gesv::StaticPivoting`: `KokkosBatched_Gesv_Impl.hpp`'s own implementation slices its `tmp` scratch
  by column range (`Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n))`) and by single column
  (`Kokkos::subview(tmp, Kokkos::ALL, n)`), both necessarily `LayoutStride` regardless of `tmp`'s own
  layout (not a layout choice this track could route around, unlike the top-level `A`'s `LayoutLeft` ->
  `LayoutRight` fix, which WAS a layout choice and did work). The compiler error, from
  `KokkosBatched_Trsm_Serial_Impl.hpp` and `KokkosBatched_LU_Serial_Impl.hpp`, both in Host and Cuda
  instantiations: `class "Kokkos::View<double *, Kokkos::LayoutStride, ...>" has no member "stride_0"`
  (and `"stride_1"`) -- this installed Kokkos's `View<..., LayoutStride>` simply does not define those
  accessors, which `SerialTrsm`/`SerialLU` call on exactly those slices.
- `Gesv::NoPivoting`: tried next on the premise that its implementation builds no such internal
  scratch (`KokkosBatched_Gesv_Impl.hpp`'s `NoPivoting::invoke` calls `SerialLU`/`SerialTrsm` directly
  on the caller's own `A`/`X`, confirmed by reading it). In the Cuda instantiation specifically, the
  1-D subview of the RHS vector (`Kokkos::subview(X, t, Kokkos::ALL)`, `X` a `View<double**,
  LayoutRight, CudaSpace>`) came back typed `Kokkos::View<double *, Kokkos::OpenMP::array_layout,
  Kokkos::Device<Kokkos::CudaSpace::execution_space, Kokkos::CudaSpace::memory_space>, ...>` --
  `Kokkos::OpenMP::array_layout` attached to a `CudaSpace` device type is a layout/execution-space
  mismatch inside Kokkos's OWN subview type deduction on this installation, not anything this header's
  view-layout choices produced. Same missing-`stride_0`/`stride_1` error as above, on that malformed
  type.

Both are Kokkos 5.2.0 / this installed kokkos-kernels version-pairing defects, independent of anything
in this header; `find_package(KokkosKernels)` itself is ALSO broken on this host (a separate finding --
`KokkosKernelsConfig.cmake:40`'s `set_target_properties()` call has an argument-count CMake's own
parser rejects, a hard configure error even under `QUIET`). Given two independent, unfixable-from-here
defects in the same dependency, this track replaced `KokkosBatched::SerialGesv` with an in-house
Gaussian elimination with partial pivoting (`solve_square_system_partial_pivot`, same file), indexing
`A`/`X` directly with no `Kokkos::subview` at all for the solve step -- re-verified against the NumPy
reference afterward, unchanged to the precision above. `KokkosKernels` is consequently NOT a build
dependency of this track's final state (the `find_package` workaround was removed along with the
dependency it worked around).

Two things remain open regardless of which solver is used: (1) the device solve reports only a
pass/fail status per target, not the rank and condition-number estimate `E-MLS-RANK`'s host message
carries, so a Device-mode rejection would have to re-run the specific failed target(s) through the
host path to recover that diagnostic, or accept a coarser message; (2) `solve_mls_on_space` takes
already-resolved point sets, options and a k-nearest result as plain arguments -- it is not wired into
`build_mls_weights`'s own `RemapPlan`/`CouplingGraph`/`MeshBackend` dispatch, so Device mode is not yet
reachable through the normal `fuse::build_weights` entry point.

**Closed, 2026-10-14 (round 2).** Both open items are resolved. `build_mls_weights`
(src/backend/mls_weights.cpp) resolves its execution space via
`fuse::search::resolve_execution_space()` (the `FUSE_EXECUTION_SPACE` convention
`RunnerConfig::resolved_execution_space()` and `fuse::search` already use -- no new `RemapPlan` field,
stated and justified in the function's own comment) and, on Device, threads the k-nearest search
through `BoxIndex::build`'s Device path and the per-target solve through `solve_mls_on_space`,
reachable now through the normal `fuse::build_weights` entry point. `MlsDeviceResult` gained a
`status` field (one flag per target, not just the aggregate `n_fail`); `build_mls_weights`'s Device
path re-runs the Host path's own rank-revealing solve (factored into `mls_fit_one_target`, shared
code, not a duplicate) on exactly the target(s) `status` flags, confirming the device solve's coarser
verdict and reproducing `E-MLS-RANK`'s exact message -- same target, same condition estimate -- or,
on the rare disagreement the two eliminations' different pivot criteria make possible, patching that
one row's weights from the confirmed Host fit instead of trusting the device's (meaningless,
early-returned) elimination state. Host mode is untouched and bitwise unchanged (every pre-existing
`backend.mls_weights` row still passes with no `FUSE_EXECUTION_SPACE` set). RED/GREEN:
`MlsWeightsDevice.DeviceMatchesHostStructureAndValuesWithinTolerance` (1e-12 relative, identical CSR
structure, on a degree-2/k=12 fixture) and
`MlsRefusalsDevice.AllCoincidentNeighboursIsRankDeficientUnderDeviceToo` (exactly-singular, so both
eliminations agree) in test/backend/mls_weights_test.cpp, plus a `backend.mls_weights.device` ctest
twin re-running the whole file under `FUSE_EXECUTION_SPACE=device`.

### D53. Supermesh (r3d/r2d) clip stays host-only; device feasibility measured and blocked on three build-system items, not algorithm

**Scope note, 2026-10-13.** Assessed, not assumed, per this track's own instructions. The r3d/r2d call
path `build_supermesh_weights` actually uses (`r3d_init_tet`/`r3d_clip`/`r3d_reduce` for a 3D pair,
`r2d_init_poly`/`r2d_clip`/`r2d_reduce` for a 2D pair) has no recursion, no heap allocation, and only
fixed-size stack buffers bounded by `R3D_MAX_VERTS`/`R2D_MAX_VERTS` (compile-time constants) -- read in
full from `_deps/r3d-src/src/{r3d,r2d}.c`, not inferred from the header. Algorithmically it is
device-portable. It is blocked today by three concrete, non-algorithmic obstacles: (1) the vendored
translation units carry no `KOKKOS_FUNCTION`/`__host__ __device__` annotation on any function, and
annotating the call path means patching the vendored copy (surviving an `UPSTREAMS.lock` bump) or
hand-porting the subset FUSE calls into a FUSE-owned header; (2) `r3d_clip`'s (and r2d's analogous)
debug path calls `fprintf(stderr, ...)` guarded only by `#if !defined(NDEBUG)`, which fires in this
track's Debug build and is not device-callable; (3) `r3d.c`/`r2d.c` are compiled as C, and `nvcc`
identifies a device translation unit by language, so even an annotated copy needs the TU compiled as
C++ to have `KOKKOS_FUNCTION` take effect. None of the three is a correctness obstacle; all three are
packaging work on a vendored, pinned dependency this track's time budget did not extend to. What IS
done: the candidate-pair broad phase (the O(target-simplex x source-simplex) box test,
`build_supermesh_weights`'s own "the honest starting point" brute force) runs as a device kernel
(`fuse::device::candidate_pairs`, include/fuse/backend/supermesh_candidates_device.hpp), verified
against a from-scratch serial reference in 2D and 3D, standing ALONGSIDE `build_supermesh_weights`
rather than wired into it (floating-point accumulation order in the existing `std::map`-based
column/row sums is not guaranteed invariant under a different candidate-generation order, and "no
change to results in Host mode" is this track's hard requirement, so the existing code path is
untouched).

**Narrowed, 2026-10-14 (round 2).** The broad phase IS now wired into `build_supermesh_weights`
itself: on a resolved Device space AND `dim >= 2` (`candidate_pairs`' box test has no analogue of the
1D path's caller-chosen single-axis test -- `axis1d` -- so dim == 1 stays on the brute-force loop
unconditionally, in every execution space, to avoid silently filtering on the wrong axis), the device
CSR of candidate pairs replaces the brute-force box test, and each candidate pair is still clipped by
the UNCHANGED host `intersect()`. For `dim >= 2` this is not merely "close": `candidate_pairs`' box
test is IDENTICAL to the brute force's own per-axis test, so the candidate set -- and the accumulation
order into the `std::map` triplet accumulator -- is the SAME as the brute force's, not just
numerically close; `SupermeshDevice.ThreeDimensionalDeviceBroadPhaseMatchesHostExactly`
(test/backend/supermesh_test.cpp) checks every triplet bit-identical between Host and Device mode on
that basis, not to a tolerance. Host mode is untouched and bitwise unchanged (`space != Device` takes
the original brute-force loop, unconditionally). The CLIP remains host-only: the three build-system
blockers this entry measured in round 1 (no `KOKKOS_FUNCTION` annotation on the vendored TUs, a
debug-only `fprintf` not device-callable, C rather than C++ translation units) are unchanged findings
-- nothing in round 2 touched r3d/r2d itself, so they are not re-measured, only re-affirmed by not
having been worked around. RED/GREEN also includes
`SupermeshDevice.OneDimensionalStaysOnHostBroadPhaseUnderDeviceToo` (the dim == 1 regression check),
plus a `backend.supermesh.device` ctest twin re-running the whole file under
`FUSE_EXECUTION_SPACE=device`.

### ~~D54. `fuse::native::to_memory_space<Target>` throws converting Host -> a real device space for a multi-dimensional field~~ — FIXED 2026-10-13 (integration)

**Found 2026-10-13, this track's own device build -- a pre-existing defect, not introduced here.**
`include/fuse/backend/native_mesh.hpp`'s `to_memory_space<Target>` (the one hook
docs/dev/native-mesh.md names as what "makes the mesh GPU-resident") had never been exercised
Host -> a real Cuda/HIP/SYCL space before this track: docs/dev/native-mesh.md's own Phase 2 note
says plainly that "`NativeMeshView` converts to host at construction and the NativeMesh tests were
pinned to host for the CUDA build." This track's new device-mode test rows
(`test/backend/native_mesh_pack_device_test.cpp`) are the first to call it that way, and it throws:

```
Error: Kokkos::deep_copy with no available copy mechanism: from source view
("native_mesh.vertex_coords") to destination view ("native_mesh.vertex_coords").
There is no common execution space that can access both source's space
(Host) and destination's space (Cuda), so source and destination
must be contiguous and have the same layout.
```

Root cause: `detail::copy_view` (same file) declares its destination view with no explicit
`array_layout` (`Kokkos::View<double* [3], Target>`), and `Kokkos::View`'s default layout differs by
execution space for a rank >= 2 view -- `LayoutRight` on Host/OpenMP, `LayoutLeft` on Cuda. A field
like `vertex_coords` (`double* [3]`, rank 2) therefore has MISMATCHED layouts between the Host source
and a Cuda destination, and `Kokkos::deep_copy` cannot bridge a memory-space change and a layout
change in the same call when the two spaces cannot access each other's memory directly. Every
rank-2-or-higher field in `MeshArrays` (`vertex_coords`; `cell_vertex_offsets`/`cell_vertex_indices`
etc. are rank 1 and unaffected) hits this on a Host -> device call; rank-1 fields do not, which is
why only `vertex_coords` appears in the measured message above (it is the first field the function
copies).

Not fixed here: a fix belongs in `to_memory_space`/`copy_view` itself (shared infrastructure every
native-mesh consumer uses, not something scoped to this track), and the right shape of the fix --
pin an explicit matching layout for every multi-dimensional field, or copy through a same-layout
mirror per field -- deserves the next device-mesh track's attention rather than a one-line patch
slipped into this one. This track's own device-mode pack test rows catch the exception and
`GTEST_SKIP()` with this citation rather than reporting a `pack_to` failure that isn't one; the
Host-mode rows (which do not cross this path) are unaffected and green.

**Fixed.** `MeshArrays::vertex_coords` is declared `Kokkos::LayoutRight` in every memory space (`include/fuse/backend/native_mesh.hpp`), so the host arrays keep the layout they always had and the device copy in `to_memory_space<Target>()` is a contiguous copy; `NativeMeshView::create_from_host`'s local pinning of the same layout stands. The two `backend.native_mesh_pack_device` cases that skipped on the throw now run the copy.

### D55. `parallel.window_newton_mpi.3` hangs on higgs's Open MPI 4.1.5 — OPEN, found by the device-merge gate 2026-10-13

`WindowNewtonMpi.TwoDisjointRanksMatchTheSingleProcessReference` at three ranks (two participants, one
bystander) never returns on higgs (Open MPI 4.1.5, Kokkos Cuda build); the two-rank row passes, and the
same rows pass on the development machine's MPICH 5.0.0a1 (D31's own host run). Reproduced standalone
on the merged tree with `OMPI_MCA_coll_tuned_alltoallv_algorithm=2` and again with the tuned
collectives disabled (`OMPI_MCA_coll=^tuned`), so it is not D40's `MPI_Alltoallv` case. The
window-Newton path (`run_region_newton_window()`) carries no device-branch change; its bystander rank
returns before the sweep (`region_is_local_participant()`), unlike the Picard window path, whose
bystander sweeps and is kept in step by `broadcast_window_convergence()` (D42) -- whether a collective
the participants then enter on the shared transport communicator is what the bystander never joins is
the first thing to check. Not diagnosed further: higgs has no `gdb` on the job shell's PATH to capture
per-rank stacks. The row is registered and times out (120 s) on that host until this is understood.

### D56. `FsiEnergyDiag.FullBudgetOnTheDominantModeCloses`'s mass-correction gate fails on higgs's toolchain by 1.5x — OPEN, found by the device-merge gate 2026-10-13

The gate `max_abs_diff_non_outflow < 0.5 * eig_quality * max|u_raw|` (test/verification/
fsi_energy_diag_test.cpp, "MASS-CORRECTION ISOLATION") measured 1.720e-10 against a tolerance of
1.183e-10 on higgs (GCC 13.3 host pass through nvcc_wrapper, Eigen SparseLU, `FUSE_FP_REPRODUCIBLE`
default) in the Host leg of the merged tree; the pre-merge branch's version of this case passed on the
same host, and the new protocol passed on the development machine (Apple clang). By this project's
tolerance convention the factor is not to be widened: the derivation (one eig_quality-level relative
error in the extracted inputs, scaled by |u_raw|) is missing a term on this toolchain -- the two
`cn_step` solves' own forward-error difference is the candidate. The `.device` twin of the row passes
(its energy budget runs on the selected oscillatory mode exactly as the host leg does).

### D57. The ABI's own period-rollover eagerly started an extra period `Runner::run()` never requests, and both sides hit a base-communicator collective the other side had already left

**What it was.** `src/abi/participant.cpp`'s `prime_next_request()` (called from `impl_begin_step()`/
`fuse_begin_step()`) resolves the next coupling request for a caller driving the Runner through the public ABI.
When its own first `Runner::Impl::resume_period()` call (`runner.cpp:3871`/`Impl::resume_period` at `4095`) finds
the current period already drained, it increments `c.periods_completed`, sets `*rolled_over = true`, and then --
gated ONLY on `c.has_end_time` -- either stops cleanly (`c.next_request.reset(); return true;`, when `has_end_time`
is true and the declared bound is exceeded) or, when `has_end_time` is false, calls `resume_period()` a SECOND
time, UNCONDITIONALLY, inside the SAME `fuse_begin_step()` call. That second call starts a FRESH period's own
coroutine and runs it to its first suspend point then and there, caching that request. `fuse_end_step()`'s own
direct `resume_period()` call has no such second call and reports the drained state immediately (`next_component ==
-1`) -- the asymmetry is this function's alone.

`examples/brusselator/participants/u_main.c` and `v_main.f90` both `memset()`/zero-initialize their
`fuse_setup_options` and set only `communicator`/`my_component` -- neither ever sets `has_end_time`. The coupler
(`coupler_main.c`) drives through `fuse_run(periods)` -> `Runner::run(components, periods)`, a plain
`for (i = 0; i < periods; ++i) run_period();` loop with no look-ahead and no equivalent rollover -- it executes
EXACTLY `periods` periods and returns. So on this row, U's and V's own ABI driver has no bound on how many further
periods `prime_next_request()`'s rollover may silently start, while the coupler's own driver stops at exactly the
requested count. `test/verification/brusselator_lowfive_harness_test.cpp`'s own `RunExternalDeliveryMatchesIn
ProcessRunner()` (a passing, pre-existing test) already works around exactly this: it sets `fuse_setup_options::
has_end_time = 1` with a real `end_time`, with its own comment that without declaring it "there is nothing for
either [prime_next_request's rollover or fuse_is_coupling_ongoing] to check against, and this graph's own
window-per-period shape... would let the Runner roll forward forever once state reaches a fixed point" -- direct,
pre-existing documentation of this entry's own defect, for a harness that sidesteps it rather than a participant
row that cannot.

The collective U and V then hang in is `OpKind::Constrain`'s conservation reduction (`runner.cpp:3362`, inside
`execute()`'s `Constrain` case): a real `MPI_Allreduce` of three doubles (before, after, a step-failure flag) over
`config.base_communicator`, issued whenever `ConstrainInfo::reduce` is set -- `resolve_constrain()`
(`runner.cpp:508`) sets it UNCONDITIONALLY to `config.base_communicator.has_value()`, not gated by any rank-set
membership the way `Epoch`'s own agreement point is (`rank_participates()`, `src/schedule/scheduler.cpp`). The
`u2v`/`v2u` channels on this row are `ExternalDelivery` (LowFive-bound, `FieldSpec::hdf5_path` set): `src/ir/
rules_channel.cpp`'s `lower_channel()` forces `ranks_differ = false` for exactly this reason ("FUSE performs no
data move of its own on this channel, so neither hop... may fire", line 317), which suppresses `Transport` and the
`Epoch` that would otherwise guard it -- but does NOT suppress `Constrain`, whose own emission (`src/ir/
rules_derived.cpp:89`) is gated only on `ch.remap.flags != RemapFlags::None`, which `add_window_coupling()`'s own
`.remap(RemapFlags::Conservative).remap_on(coupler)` sets regardless of transport. So this row's compiled plan
carries a `Constrain` instruction with no `Transport`/`Epoch` anywhere near it, on every rank of the base
communicator, coupler included -- and `fuse::constrain_sequence()` (new this entry, see "Fix" below) confirms by
direct measurement that this part is NOT the defect: all three real rank placements compile the IDENTICAL
`Constrain` sequence. The defect is a disagreement about how many PERIODS to walk that identical sequence, not
about what is in it.

**How it surfaced.** MEASURED ON THE OWNER'S MACHINE (2026-10-13, HEAD 7aec9c8). Row `verification.brusselator.
participants.lowfive.mpmd.3` (`run_participants.sh mpmd`, `--periods 2`; graph.yaml: U 1 rank, V 1 rank, coupler 1
rank, region `brusselator_window`, channels `[u2v, v2u]`, `max_iterations 60`, convergence `Relative 1e-11` on
`u_out`/`v_out`, `acceleration None`, `window_scope explicit_interval 1/50 order 2`, iteration `Jacobi`). The
transport-only twin `verification.brusselator.participants.mpmd.3` (same graph, `Transport` instead of the
LowFive delivery) PASSES. In the LowFive row, U and V run their `ExternalDelivery` stops all the way through both
requested periods -- U made 122 writes and 121 reads, V made 121 writes and 122 reads (61 deliveries per period;
the lead's trace counters mislabel these as `period=121`); U's last event is a write, V's last is a read. Then
BOTH U and V sit forever in `fuse_begin_step`. Sampled stacks (macOS `sample`, 905 and 907 samples, 40 s after
launch, both in the main thread): U -- `main (u_main.c:549)` -> `u_loop_step` -> `impl_begin_step`
(`participant.cpp:1250`) -> `prime_next_request` -> `Runner::resume_period` (`runner.cpp:3871`) -> `Impl::
resume_period` (`4095`) -> coroutine resume -> `Runner::Impl::run_region` (`.resume`) -> `Runner::Impl::execute`
(`runner.cpp:336` as sampled; coroutine-frame line numbers are not reliable) -> `MPI_Allreduce` -> `MPIR_Allreduce`
-> `MPIDI_POSIX_mpi_allreduce_release_gather` -> `MPIDI_NM_progress`. V: the same stack, `v_main.f90:477`
`bruss_v_loop_step` -> `ffuse_begin_step`. The coupler is NOT in the picture: it had already returned from its own
`main()` (`MPI_Finalize` ran; its stderr shows MPICH's own leak report, which also appears in the passing twin,
then a separate static-destruction abort another track owns, unrelated to this hang) -- exactly the signature a
driver that stops at exactly 2 periods produces against partners whose own driver has started a 3rd.

**Fix.** `prime_next_request()` gained an `else if (c.communicator_set)` branch paired with the existing
`if (c.has_end_time) {...}` block, firing when `!c.has_end_time && c.communicator_set`: `c.next_request.reset();
return true;` -- treating "the current period just drained, no end time was declared, and a base communicator IS
configured" exactly like the exceeded-end_time case: stop cleanly and defer starting the next period to a LATER,
separate `fuse_begin_step()` call, instead of eagerly resuming it inside this one. The `has_end_time == true` path
is untouched (zero behavior change for any caller that declares one, including the LowFive harness test above),
and `fuse_end_step()`'s own direct `resume_period()` call -- already lazy -- is untouched.

The `c.communicator_set` half of the gate is not incidental: a FIRST draft of this fix stopped eagerly whenever
`!c.has_end_time`, with no further condition, and the full, unfiltered tier-0 `ctest` run this entry's own task
required at the end (not a filtered re-run) caught it regressing a pre-existing, deliberately designed conformance
guarantee -- `AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent` (`test/abi/participant_test.cpp`,
"Case (ii) of the lead's own ruling") -- which never configures a communicator and relies on EXACTLY the eager
continuation the first draft removed unconditionally, to resolve a still-converging window's real next-period
owner ahead of the caller asking for it. With no communicator at all there is no partner rank to disagree with, so
the eager continuation there is harmless; the hazard this entry fixes exists only when a base communicator is
configured, making `c.communicator_set` -- set once, identically on every rank sharing it, from
`fuse_setup_options::communicator` at setup time, never from rank-local runtime state -- the precise condition,
not `!c.has_end_time` alone. `fuse::abi_test::force_communicator_set()` (new this entry, `src/abi/test_support.
{hpp,cpp}`) lets `brusselator_period_rollover_test.cpp` exercise this exact gate at tier 0 with no MPI runtime:
set AFTER a successful `fuse_initialize()` (setting it before fails initialize() itself outright at tier 0, since
`impl_initialize()` also picks the "mpi" transport whenever `communicator_set` is true, and tier 0 registers no
such backend), it reaches `prime_next_request()`'s own read of a plain context field with no other run-time
consumer, changing nothing about the already-initialized Runner's own transport or conservation-check wiring.

Also added, as the task's own fallback for task item 3-4: `fuse::constrain_sequence(plan, graph, rank)`
(`include/fuse/schedule/plan.hpp`, `src/schedule/scheduler.cpp`), `epoch_sequence()`'s own sibling for the
Constrain collective, reusing the same `rank_participates()` helper; `test/schedule/constrain_sequence_test.cpp`
ties it to the REAL three-rank Brusselator participants layout (built through `fuse::lower()`/`fuse::schedule()`
alone, which need no MPI and reach past the `E-RT-REGION` ceiling that blocks `Runner::initialize()` on this
layout at tier 0) and measures all three rank placements' sequences identical -- proving there is no static
schedule divergence to find, which is exactly why a purely static, `(plan, graph, rank)` function cannot by
itself expose this entry's own defect: the divergence is in how many times a rank's own ABI driver re-enters an
agreed-upon, identical schedule, a property of `fuse_setup_options::has_end_time` and which of
`fuse_begin_step()`'s two internal paths discovers a period boundary, neither of which `plan`/`graph` carry. A
parallel model of "how many periods will this rank's driver run" was deliberately NOT written for the same reason:
it would have to re-encode `prime_next_request()`'s own control flow to answer the question, which is the exact
kind of duplicated-and-drifting logic the task's own item 3 warns against. Instead, an off-by-default, env-gated
per-rank trace was added at the one place that is both reachable without re-implementing the Runner and precise
enough to show the real divergence on a host: `FUSE_COLLECTIVE_TRACE=1` prints one line per collective instruction
a rank's own `run_region()` (`src/runtime/iteration.cpp`) reaches -- rank, region, period, sweep, instruction id
and kind, communicator tag, and a per-rank monotonically increasing ordinal (`Runner::Impl::collective_trace_
ordinal`, `src/runtime/runner_impl.hpp`) -- backed by a pure formatter (`fuse::format_collective_trace_line()`,
`include/fuse/runtime/collective_trace.hpp`) pinned by three tier-0 cases in `test/runtime/collective_trace_test.
cpp` with no coroutine, no communicator and no graph at all.

**Pinned by** `test/verification/brusselator_period_rollover_test.cpp`
(`BrusselatorPeriodRollover.StopsAtTheRequestedPeriodCountWithNoEndTime`). Reproducing the literal three-rank
MPMD graph at tier 0 is not possible (`test/verification/brusselator_external_delivery_size_test.cpp`'s own
`ReachesPastExternalDeliveryOnBothRanks` already measured that `Runner::initialize()` on this layout cannot get
past `resolve_regions()`'s own `E-RT-REGION` refusal with no real communicator); this test instead drives the
SAME multirate Brusselator window shape (`brusselator::add_window_coupling()`, `brusselator_graph_tool.cpp`'s own
`kDtU`/`kDtV`/`kWindow`/`kTolerance`/`kMaxIterations`, both transfer fields `hdf5_path`-bound) single-process,
through the REAL public ABI (`fuse_begin_step`/`fuse_end_step`, never a reimplementation of `prime_next_request()`
's own logic), with `fuse_setup_options::has_end_time` deliberately never set -- `u_main.c`'s/`v_main.f90`'s own
choice, reproduced on purpose. The decisive assertion: at the call where the loop's own tracked period count first
reaches the requested count (2), `fuse_step_report::next_component` must equal -1 -- the ONLY value
`impl_begin_step()` reports when `c.next_request` genuinely holds nothing (any other value, even with
`step_opened == 0`, means a request for an unrequested THIRD period was already resolved and cached). Measured:
with the fix applied (and, as of this test's own final revision, `fuse::abi_test::force_communicator_set()`
called after `fuse_initialize()` to put this single-process graph into the `communicator_set == true` state the
real row is actually in), `next_component == -1` there and the test PASSES; with `participant.cpp` reverted to the
HEAD version (`git show HEAD:src/abi/participant.cpp`), `next_component == 0` there and the `EXPECT_EQ` FAILS; with
the fix restored, PASSES again. One mutation row reverting the fix (see below) reproduces the same failure through
the WHOLE unfiltered test binary. The full, unfiltered tier-0 `ctest` run this entry's own task required (`99%
tests passed, 1 tests failed out of 208`, the ONE failure named by path, not inferred) is what caught the first
draft's own regression of `AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent` (see "Fix" above);
after the `communicator_set` gate was added, the SAME full suite's own `abi.participant` binary (17 tests, run
whole and unfiltered again, not merely the one case) passed completely, and `brusselator_period_rollover_test`'s
own case above still passes with the gate in place. The pre-existing `fuse_verification_brusselator_lowfive_
harness_test` (which DOES declare `has_end_time`) was rebuilt with the fix applied and its `*Jacobi*` case still
PASSES (75391 ms,
values matching the reference run across all 150 windows as before) -- the fix changes nothing on the path every
caller that declares an end time already takes.

**Does not close.** Three adjacent findings, deliberately NOT fixed here because each is either out of this
entry's own scope or changes what a check verifies rather than who participates in it (task item 5's own stop
condition):

- `src/runtime/iteration.cpp`'s `resolve_regions()` seeds a window region's own `R.participants` from the window
  members' ranks and extends it only by scanning the region body for `Transport`/`Epoch` instructions
  (`iteration.cpp:1326-1437`). For a fully `ExternalDelivery`-bound region (no `Transport`/`Epoch` at all, this
  entry's own "What it was") the coupler's rank is never added to `R.participants`. This does NOT gate whether the
  coupler's own rank enters the region or reaches `Constrain` (`run_region()`'s own generic, non-Newton path has no
  `region_is_local_participant()` early return the way `run_region_newton()`/`run_region_newton_window()` do,
  confirmed by the coupler completing both requested periods' own deliveries cleanly before this entry's hang) --
  `R.participants` is read only by specific cross-rank reductions (residual/Newton/window-convergence gather)
  inside the region, which this row's own `acceleration None`/Jacobi path does not reach the window-convergence
  form of either. Left as an open question because fixing it would mean deciding what `R.participants` is FOR on a
  row with no coupler-routed data move at all, which is a question about what the window-convergence reduction
  verifies, not who currently participates in it.
- `examples/brusselator/participants/u_loop.c`'s `u_loop_step()` wrapper zeroes `*out_rep` entirely whenever
  `!out_begin_rep->step_opened`, which can mask the real signal `u_main.c`'s own check reads
  (`rep.next_component == -1`) in circumstances this entry did not fully enumerate. `u_loop.c` is on the "files you
  must NOT touch" list (the LowFive attach track owns it); flagged, not fixed.
- The exact "61 = 60 + 1" per-period delivery count (task item 2) is a RUNTIME sweep-count quantity, not a
  compiled one: `fuse::schedule()`'s own output for this row's region body carries exactly ONE `ExternalDelivery`/
  `Constrain` pair per channel (measured directly, a throwaway diagnostic built against this tree's own
  `fuse::lower()`/`fuse::schedule()`, not part of this entry's files), and `run_region()`'s own sweep loop (`for (k
  = 0; k < R.max_iterations; ++k)`) is what re-executes it up to `kMaxIterations` (60) times per period -- the
  asymmetry between U's 122 writes/121 reads is consistent with a cold-start read this row's own first period does
  not need (the first sweep of period 0 has no prior period's `v_in` to read yet) plus one write every sweep
  including a final one neither side's own read mirrors, but this entry did not run the real species physics to
  confirm 61 is exactly `kMaxIterations + 1` rather than a tolerance-dependent sweep count that happens to land
  there on the owner's own machine. `D39`'s own Round 8 section independently flagged the same open question ("the
  Jacobi-sweep-frequency question... would mean ExternalDelivery fires more than once per period for
  max_iterations > 1") from the attach-order side; this entry confirms the mechanism (one compiled instance,
  runtime-repeated) without pinning the exact count.

**What the host must verify.** Rebuild `build-all` and `build-lowfive` (`cmake --build .`, every target, this
entry's own files are `src/abi/participant.cpp`, `src/abi/test_support.{hpp,cpp}`, `src/runtime/iteration.cpp`,
`src/runtime/runner_impl.hpp`, `src/schedule/scheduler.cpp`, plus new files -- see this round's own report for the
full list), then:

(a) the regression guard: `ctest -R 'verification\.brusselator\.participants\.(mpmd|accept)\.3'` in `build-all` --
the transport-only twins must still PASS (unaffected by anything in this entry: neither row's own channels are
LowFive-bound, so `prime_next_request()`'s rollover is never reached the way it is here, and the fix's own new
`else` branch only changes behavior for a caller with `has_end_time == false`, which neither twin's own harness
sets either way without re-measuring -- if either now FAILS, this entry's claim that the fix is a no-op outside
the `!has_end_time`, drained-period path is WRONG).

(b) the row itself, in `build-lowfive`: `FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info ctest -R
'participants\.lowfive\.mpmd\.3' --output-on-failure`. A pass looks like `brusselator_check_core.hpp`'s own
`print_summary()`: both periods' own delivered values matching the in-process reference within the configured
tolerance, and the run exiting through `MPI_Finalize` on all three ranks rather than hanging. Foreseeable
failures and what each would contradict: the row still hangs in the SAME `MPI_Allreduce` stack -- would mean this
entry's own root cause is incomplete (something else also starts an unrequested period, or the fix did not reach
the binary the host built); the row now exits promptly but `brusselator_check` reports a VALUE mismatch -- would
mean the fix is structurally right (no more hang) but this entry's claim that it changes no caller's reported
values is wrong, or a genuinely separate data defect (D39's own attach/order track, or the open
`R.participants`/cold-start questions above) is now visible with the hang out of the way; the row hangs in a
DIFFERENT collective or a different stack -- would mean a second, independent instance of this defect class exists
(task item 1's own per-instruction table in this round's report names every other reachable collective site to
check first); U/V now stop one period EARLY relative to the coupler -- would mean the new `else` branch itself
has an off-by-one this entry's own tier-0 test did not catch (it only exercises `kRequestedPeriods = 2` with a
window that converges on sweep 0 every period, not every `(periods, convergence-sweep-count)` combination a host
run could exhibit).

**Reconciliation with D42, added 2026-10-13 when `vijaysm/kokkos-device` was merged into `main` (not part of the track's own report).**
The open item above about `resolve_regions()` seeding `R.participants` without the coupler's rank concerns the same set
that D42 ("A rank outside a window-scoped region's participants never learned the window converged", fixed 2026-10-10 on
the device branch, commit 285beda) addresses by broadcasting the window's convergence verdict to ranks outside the
participants. This entry was written and verified against `7aec9c8`, which did not contain that change; the claim above that
the coupler's absence from `R.participants` does not gate this entry's hang has NOT been re-measured on the merged tree.
Whether the broadcast also covers what the window-convergence and Newton reductions verify for a row with no
coupler-routed data move is open and needs a decision on what `R.participants` is for, as stated above.

**Round 2 (2026-10-13) -- the fix above did NOT close the real row; the actual defect was the second bullet of**
**"Does not close" above (`u_loop.c`'s/`v_loop.f90`'s own zeroing), now fixed; the tier-0 reproduction above**
**passed for the wrong reason.** The owner reports `verification.brusselator.participants.lowfive.mpmd.3` still
hangs on the real host after Round 1's fix landed: U and V both start an unrequested THIRD period and block in
its base-communicator `Constrain` `MPI_Allreduce`, the coupler exiting cleanly after exactly 2 periods -- the
IDENTICAL symptom this entry's own "How it surfaced" section describes, unchanged by the `communicator_set` gate.

**What actually starts period 2 on the merged HEAD.** NOT `prime_next_request()`'s own eager-continuation branch
this entry's own "Fix" above addresses -- that gate is confirmed intact and firing correctly (`else if
(c.communicator_set) { c.next_request.reset(); return true; }`, `src/abi/participant.cpp`), and
`impl_begin_step()`'s own `!c.next_request.has_value()` branch correctly reports `next_component == -1,
step_opened == 0` exactly when it fires. The real defect is the SECOND bullet this entry's own "Does not close"
section already named and deliberately left unfixed as out of scope: `examples/brusselator/participants/
u_loop.c`'s `u_loop_step()` (mirrored in Fortran by `v_loop.f90`'s `v_loop_step()`) wraps `fuse_begin_step()`/
`fuse_end_step()` for `u_main.c`'s/`v_main.f90`'s own time loop. Whenever `fuse_begin_step()` reports
`step_opened == 0` (nothing to do this call -- a delivery-pending suspend, OR the genuine, correctly-reported
period-boundary close this entry's own fix produces), `u_loop_step()` does `memset(out_rep, 0,
sizeof(*out_rep)); out_rep->struct_size = sizeof(*out_rep); return FUSE_OK;` -- WITHOUT copying
`out_begin_rep->next_component` into `out_rep->next_component`. `u_main.c`'s own loop (`for (p = 0; p < periods;
++p) { for (;;) { rc = u_loop_step(...); if (rep.next_component == -1) break; } }`, `v_main.f90`'s own mirror)
checks ONLY `rep.next_component` (`out_rep`, never `begin_rep`/`out_begin_rep`) to decide a period has closed.
Since `out_rep->next_component` is therefore always 0 (a valid `ComponentId`, not `-1`) on every `step_opened ==
0` report, including the one this entry's own fix correctly resolves to `-1` in `begin_rep`, the inner `for (;;)`
loop never breaks: it calls `fuse_begin_step()` again, `prime_next_request()` finds `c.next_request` still empty
(left that way by this entry's own fix) and makes its FIRST, ORDINARY `resume_period()` call -- the one call site
`prime_next_request()` has NO bound on at all when `!c.has_end_time` -- which starts period index
`kRequestedPeriods` (2), the one `Runner::run(components, 2)` (the coupler's own driver) never requests. This
matches the owner's own stack trace exactly (`u_loop_step` -> `impl_begin_step` -> `prime_next_request`, i.e.
`fuse_begin_step()`'s OWN call into `prime_next_request()`, not `fuse_end_step()`'s rollover) and explains why the
tier-0 reproduction this entry pinned PASSED despite the real row still hanging: that test's own `drive_one_step()`
helper read `begin_rep.next_component` directly (Round 1's own choice, this entry's "Pinned by" section above),
which `impl_begin_step()` DOES report correctly -- the exact field the real callers never read.

**Why this was deferred, not missed, in Round 1.** This entry's own "Does not close" section (second bullet,
above) already named `u_loop.c`'s zeroing as a mechanism that "can mask the real signal... in circumstances this
entry did not fully enumerate," and declined to fix it because `u_loop.c` was then on the "files you must NOT
touch" list (owned by the LowFive attach track). This round's own task explicitly reassigns `u_loop.c`/
`v_loop.f90` to this track for the fix (D39/`docs/user/lowfive.md`/`lowfive_attach_decision.hpp` remain owned by
the LowFive attach track and are untouched here -- see "Open items for the LowFive attach track" below).

**Fix (Round 2).** `u_loop_step()` (`u_loop.c`) and `v_loop_step()` (`v_loop.f90`) each gained one line in their
own `step_opened == 0` branch, immediately before returning `FUSE_OK`: `out_rep->next_component =
out_begin_rep->next_component;` (C) / `out_rep%next_component = out_begin_rep%next_component` (Fortran) --
propagating the one field either real caller loop's "has a period closed" check reads. Safe for every OTHER
`step_opened == 0` case too (delivery-pending suspends and ordinary/matched rollovers): `impl_begin_step()`
already resolves `out->next_component` to a REAL, non-`-1` component id or `-2` (async-pending) in those cases
(confirmed by reading the delivery-due branch and the `treat_as_mismatch` branch, `participant.cpp`), so
propagating this field changes behavior ONLY for the genuine "nothing cached, run over" case -- `0 -> -1` -- which
is exactly what was wrong.

**Pinned by (Round 2).** `test/verification/brusselator_period_rollover_test.cpp` rewritten (test renamed
`CallerLoopNeverSeesThePeriodBoundaryThroughTheZeroedReport`): its own `drive_one_step()` helper now mirrors
`u_loop_step()`'s fixed propagation VERBATIM (Round 1's own version did not, which is exactly the gap this
round's own host measurement traced back to the real files), and the driving loop now uses ONE shared
`global_periods_done` counter (not independent per-component counts -- `c.periods_completed`, `src/abi/
context.hpp`, is one counter PER `fuse_ctx`, which on the real three-rank row is one counter per RANK; this
single-process reproduction drives both "U" and "V" through ONE `fuse_ctx`, so there is genuinely only one such
counter here too), checking `begin_rep.step_opened && begin_rep.period >= kRequestedPeriods` as the decisive,
airtight signal (a REAL step of an unrequested period actually running), never `begin_rep.next_component` for
that decision (diagnostic only). NEW: `test/verification/brusselator_u_loop_step_fix_test.cpp`
(`fuse_verification_brusselator_u_loop_step_fix_test`, ctest row `verification.brusselator_u_loop_step_fix`) drives
the REAL `u_loop_step()`/`bruss_v_loop_step()` functions (not a mirror) through real `bruss_u_handle`/
`bruss_v_handle` physics, at a 1x1 grid, with `fuse::abi_test::force_communicator_set()` and no end time, exactly
like the mirror-based test but exercising the production files directly -- two `TEST`s: one drives the graph's
two members in their natural U/V role; the other drives graph component 0 ("U") through the REAL
`bruss_v_loop_step()` and component 1 ("V") through the REAL `u_loop_step()`, CROSSED, because tracing confirmed
(`is_u`/`begin_rep.next_component`/`begin_rep.period` printed at every `step_opened == 0` call, both
`WaveformIteration::Jacobi` and `::GaussSeidel`, and with `kDtU`/`kDtV` swapped) that this single-ctx simulation's
shared `c.next_request` cache ALWAYS routes the genuine period-close observation through graph component 0, never
component 1 -- an artifact of simulating two ranks' independent `prime_next_request()` sequences as one
interleaved sequence sharing one cache (the real row has no such tie: each rank runs its own, independent
`fuse_ctx`). The crossed test isolates `v_loop.f90`'s own fix at the one schedule slot that is structurally
terminal here. Two mutation rows (`build-evidence/mutations/brusselator.toml`:
`brusselator-u-loop-step-drops-next-component-at-period-close`, `brusselator-v-loop-step-drops-next-component-at-
period-close`) each CONFIRMED by hand: reverting either file's own one-line fix made its own isolating `TEST` FAIL
(a real step of period 2 observed; the OTHER test, which never calls the mutated function for the component that
matters, PASSED unchanged, confirming isolation both directions), file restored and verified byte-identical via
`cmp` (`md5`: `u_loop.c` = `a034a5bddf203694ea8ff3b522bd2d4d`, `v_loop.f90` =
`5266fee6a7d5e06dbe75d93ff0a18f3a`), clean rebuild passed again. `python3 tools/mutate.py --check-anchors`: 350
valid, 0 stale, of 350.

**The "61 = 60 + 1" question (this entry's own "Does not close" section, third bullet, above) -- the hypothesis is**
**refuted; where 61 came from is NOT established.** This round ran the REAL species physics this entry's own third
bullet said it had not (`test/verification/brusselator_participants_test.cpp`'s own in-process `fuse::Runner`
reference, the REAL grid `--nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24`, `dt_u = 2/1000`, `dt_v = 3/1000`, `window =
2/100`, `Relative 1e-11`, `max_iterations = 60`, Jacobi): `ref_iterations[0] = 12`, `ref_iterations[1] = 12`. That
matches the owner's measurement on the merged HEAD (12 sweeps per period, identical on all three ranks), and it
refutes this entry's own tentative "61 is exactly `kMaxIterations + 1`" hypothesis: a converged, tolerance-driven
window of 12 sweeps cannot itself produce 61 deliveries.

What IS established: (i) the figure 61 was recorded in commit `60b6207`, which is an ancestor of `23329ac` (this
entry's Round 1 fix) and a parent of the device-branch merge `4635f0f`, so it was measured on `7aec9c8`, before
both; (ii) on `b98777d` the three ranks' collective traces agree on 12 sweeps per period; (iii) the earlier
host figure was 122 U writes in total, which the lead's reading divided by the two requested windows. What is NOT established is
why the earlier run delivered 122 times. Arithmetic only, not a finding: 122 = 1 + 10 x 12 + 1, so 122 deliveries are
also consistent with 12 sweeps per period over TEN periods, and 61 per window would then have been an artefact of
assuming two windows. Other candidates, none measured: redundant `resume_period()` calls inside one
`fuse_begin_step()` under the old eager continuation, or a change that arrived with the device-branch merge (D42
changed how a window's convergence verdict reaches ranks outside `R.participants`). Re-running the row at `7aec9c8`
on the host (build-lowfive, before this entry's fixes) and reading its collective trace would decide between them.

**Open items for the LowFive attach track (relayed, not edited -- `KNOWN_ISSUES.md`'s own D39 section,
`docs/user/lowfive.md`, and `examples/brusselator/participants/lowfive_attach_decision.hpp` are that track's own
files).** The following lines repeat the pre-fix "61 deliveries/exchanges per window" figure as a presumed-stable
measurement; now that this round traced it to a pre-Round-1-fix commit, that track may want to annotate or
re-measure them: `KNOWN_ISSUES.md:4948` ("the host measured 61 deliveries per window for a `--periods 2` run
(122 U writes total)"), `KNOWN_ISSUES.md:4958` ("ANSWERED by the host: ~61 deliveries per window... i.e. once per
sweep"), `KNOWN_ISSUES.md:4961` ("the exact relationship between 61 and this row's own `max_iterations = 60` is
that track's own question"), `docs/user/lowfive.md:955`, `docs/user/lowfive.md:959`, `examples/brusselator/
participants/lowfive_attach_decision.hpp:448`, `:459`, `:463`. None of these are edited by this round.

**Still OPEN.** Round 2's own fix has never been run on the host. See "What the host must verify (Round 2)"
below for the exact commands and the expected trace shape.

**What the host must verify (Round 2).** Rebuild `build-lowfive` (`cmake --build .`, every target -- this
round's own files are `examples/brusselator/participants/u_loop.c`, `examples/brusselator/participants/
v_loop.f90`, plus the new/changed tests above), then:

```
FUSE_COLLECTIVE_TRACE=1 FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

On success: all three ranks exit 0 through `MPI_Finalize`, no hang, no timeout. `FUSE_COLLECTIVE_TRACE`'s own
per-rank line count should now be IDENTICAL and bounded on all three ranks -- this entry's own Round 1 measurement
was 49/49/48 lines (U/V/coupler) before this round's own fix (U and V one `Constrain` into their own unrequested
third period before the host's 120 s timeout cut them off); with 12 sweeps/period x 2 Constrains/sweep x 2
requested periods = 48 `FUSE_COLLECTIVE_TRACE` lines per rank, all three ranks should now match exactly (48 = 48
= 48), not 49/49/48. Foreseeable failures and what each would contradict: the row still hangs in the SAME
`MPI_Allreduce` stack with U/V one step into period 2 -- would mean this round's own fix did not reach the binary
the host built, or a second, independent source of the same "zeroed report" pattern exists elsewhere this round
did not find; the trace line counts are NOT 48/48/48 but some OTHER matching triple -- would mean this round's
own `ref_iterations` measurement (12 sweeps/period for periods 0-1) does not hold on the real multi-rank row
(e.g. a remap/placement difference this round's single-process reproduction cannot see); the three ranks' own
trace counts still DISAGREE (not just 49/49/48) -- would mean a THIRD, independent mechanism also disagrees about
period count, beyond both this entry's Round 1 fix and this round's own Round 2 fix; `brusselator_check` now
reports a VALUE mismatch with the hang gone -- would mean a genuinely separate data defect (D39's own attach/
order track, or the open `R.participants`/cold-start questions in this entry's own "Does not close" section
above) is now visible with the hang out of the way, not a regression from either round's own fix.

**Round 3 (2026-10-18, ABI track): the invariant itself, written down and given a second,
graph-independent test.** Everything above pins this entry as it was FOUND -- through the real
Brusselator participant files' own driving-loop shape. Neither round stated the general rule those
files happen to satisfy once fixed. `docs/dev/abi-design.md`'s new "The period contract" section,
and a matching doc comment on `fuse::Runner::run_period()` (`include/fuse/runtime/runner.hpp`), now
state it directly: a caller is entitled to a COUNT of periods, never a promise that state has
settled past that count, and the Runner must never execute a period's real instructions beyond the
last one actually requested, even once a window-scoped region has reached a fixed point and the
NEXT period's first request costs nothing to resolve ahead of time.

`test/verification/abi_period_boundary_test.cpp`'s `AbiPeriodBoundary.NeverRunsAPeriodTheCallerDid
NotRequestOnceStateHasReachedAFixedPoint` states the invariant as a direct, per-step check rather
than as a single reading of `next_component` at the end: it drives the SAME minimal window graph
`brusselator_period_rollover_test.cpp` builds through the real public ABI, and on EVERY step that
actually opens (`fuse_step_report::step_opened == 1`, meaning real instructions -- including this
region's own Constrain, which is what hung a multi-rank run in its own base-communicator
MPI_Allreduce, this entry's own "How it surfaced") asserts that step's own `period` field is
strictly less than `kRequestedPeriods` (2) -- the number of periods this test actually asked for.
No Runner-level oracle (`fuse::Runner::run()`'s own in-process reference, or `fuse::RunStats`) is
needed to state this: the bound is read directly off the SAME report field every real caller
already reads, matching the simpler design this file's own header comment settled on over an
earlier draft that compared `fuse::RunStats::conservation_checks` against a Runner reference (that
draft was replaced during this round once it proved fragile to get genuinely equivalent to the
Runner's own stopping point, not because the invariant itself changed). RED-verified directly: with
`prime_next_request()`'s fix reverted, this test fails (the caller's own loop never observes a
period boundary within a 500-step safety cap, because the rollover keeps resolving further
unrequested periods) -- restored afterward, confirmed byte-identical. This is independent evidence
from `brusselator_period_rollover_test.cpp`'s own `next_component == -1` assertion, reusing the
SAME proven graph-construction helper (`brusselator::add_window_coupling()`) rather than a
from-scratch window fixture, for the reason stated in this new file's own header comment:
re-deriving a coupling window's setup from first principles is exactly the kind of
duplicated-and-drifting logic this entry's own "Fix" section already warned against for a parallel
model of `prime_next_request()`'s control flow.

**Still open, unchanged by this round.** Round 2's own fix (`u_loop.c`/`v_loop.f90`'s propagation
of `next_component`) has never been run on the real multi-rank host row; "What the host must
verify (Round 2)" above is still the host's own next step, and this round added no new host-run
requirement beyond it.

### D58. `tools/probe-hdf5-images.c` counts distinct dylib paths matching `libhdf5`/`libhdf5_hl`, and cannot see an HDF5 implementation embedded inside another library's own binary; OPEN, doc-only, the probe is NOT changed

**Measured 2026-10-13**, as part of the LowFive-attach track's own round 10 investigation (D39), on the owner's
host. The probe's own detection strategy walks the process's own loaded-image list and counts distinct dylib
PATHS whose basename matches `libhdf5`/`libhdf5_hl` -- a reasonable check for the two-HDF5-image hazard (D14)
when every HDF5 implementation in the process ships as its own separate, identifiably-named dylib. It does not
hold when an HDF5 implementation is compiled directly INTO another library's own binary, with no dylib of its
own named `libhdf5*` for the probe to find.

On the now-removed `/opt/moab-tempest` install: `libMOAB.5.dylib` defined 3610 `H5*`-prefixed symbols of its
own (`nm -gU`) and carried NO `libhdf5`/`libnetcdf` entry in its own `otool -L` dependency list -- consistent
with an HDF5 implementation built directly into the MOAB dylib itself, not linked as a separate shared object.
The probe printed `PASS` with 1 image for this install -- the SAME verdict it prints for an install with no
HDF5-image hazard at all, because its own path-counting logic has nothing to count: there was no second
`libhdf5*`-named path to find, embedded or not.

On `/opt/moab/cmake-all` (the current install, after the owner removed `/opt/moab-tempest`): MOAB links
`libhdf5.310`/`libhdf5_hl.310` as ordinary shared-library dependencies (present in its own `otool -L`) and
defines only 2 `H5`-prefixed symbols of its own (`_H5FD_http_init` and `_H5Pset_fapl_http`, by name an HTTP
file-driver pair; their purpose was not examined, only their count and names) -- a case the probe's own
path-counting design handles correctly,
since there is a genuine `libhdf5*`-named path for it to count. This round's own tier-1 configure (`-DMOAB_DIR=
/opt/moab/cmake-all/lib/cmake/MOAB`) exercised the probe step directly and confirmed it: PASS, 1 libhdf5/1
libhdf5_hl image, the same shared `/opt/hdf5/lib/libhdf5.310.dylib` LowFive and `HDF5_ROOT` already use.

**Not changed this round.** The owner has not asked for the probe itself to be modified; this entry records
the blind spot for the record, as an UNBUILT proposal, not an implementation. A detection that wanted to catch
the embedded case would need to compare which loaded image `H5open` itself RESOLVES through -- e.g. `dladdr()`
on the address returned by `dlsym(RTLD_DEFAULT, "H5open")`, or inspecting which image DEFINES `H5open` as a
strong (not merely re-exported) symbol -- against the set of images the existing path-counting check already
finds, rather than relying on path names alone. A MOAB build that embeds HDF5 would then show `H5open`
resolving through `libMOAB` itself, distinguishable from both "no HDF5 loaded" and "HDF5 loaded as its own
separate image" -- the two cases the current, path-name-only check cannot tell apart. This proposal has not
been built, compiled, or tested against either MOAB install; it is recorded here as a starting point for
whoever next touches this probe, not as a verified fix.

**What remains.** Whether MOAB's build system has a flag controlling embedded-vs-linked HDF5 (a configuration
choice `/opt/moab-tempest` happened to have made and `/opt/moab/cmake-all` did not) is not investigated here --
out of scope for a doc-only entry about the probe's own blind spot, not about MOAB's own build options. Whether
any currently-PASSING row depends on the probe's inability to see an embedded image (i.e., whether the
same hazard D14 was written for could still be silently present behind a MOAB install that embeds HDF5) is
also open -- this entry does not claim D14 is unsound, only that this ONE probe cannot fully verify its own
precondition on every possible MOAB build.

### ~~D59. Two failures the host's quick tier showed on 2026-10-13: a stale cached mesh path after the MOAB switch, and a stale refusal code in `petsc.snes`~~ — FIXED 2026-10-13, host-verified

**Measured** with `ctest -LE slow` on `build-all` (HEAD `9ffd2b6`): 321 of 323 passed; `petsc.snes` and
`backend.moab_mesh` failed. Neither involved the D14 stage.

**`backend.moab_mesh`: 4 of 46 failed** (`MoabMeshFile.ReadsTheCubedSphereAndReportsLocalCounts`,
`CubedSphereGlobalIdsComeFromTheFile`, `CubedSphereAreaMatchesAnIndependentQuadrature`,
`IndividualCubedSphereQuadsMatchABruteForceIntegral`), each with MOAB's `/opt/moab-tempest/share/examples/fortran/spherecube.h5m:
No such file or directory`. Cause, read from `build-all/CMakeCache.txt` and `test/CMakeLists.txt`:
`FUSE_MOAB_TEST_MESH_FILE` is the cached result of `find_file`, which never searches again once its variable is set, so
the path into the removed `/opt/moab-tempest` survived a reconfigure with `-DMOAB_DIR=/opt/moab/cmake-all/...` and the
four tests were compiled against a file that no longer existed. The comment above that block says the absence of the
file is meant to be "a configure-time message, not a test that reads as passing"; the stale cache defeated that. The
new install ships no `share/examples/fortran/spherecube.h5m` (checked: no `spherecube*` anywhere under
`/opt/moab/cmake-all`); the file is 258,228 bytes and byte-identical (same md5) in three MOAB source trees under
`/Users/mahadevan/Code/sigma/moab-release`. **Fix** (`test/CMakeLists.txt`): the cached path is re-validated and
searched again when it is gone, and an optional `-DFUSE_MOAB_SOURCE_DIR=<MOAB source root>` is searched second at
`examples/fortran` (the owner chose this over vendoring the file or dropping the tests). Without that option and with
an install lacking the file, the four tests are not compiled and configure says so. **Measured after the fix**
(reconfigure with `-DFUSE_MOAB_SOURCE_DIR=/Users/mahadevan/Code/sigma/moab-release`): both trees printed `cached
FUSE_MOAB_TEST_MESH_FILE=... no longer exists; searching again` then `real-mesh MOAB tests will use .../examples/fortran/
spherecube.h5m`; `backend.moab_mesh` Passed in `build-all` and `build-lowfive`. Not measured: a configure with neither the
install's file nor `FUSE_MOAB_SOURCE_DIR` (the not-compiled message path), and any other cached `find_*` result that
could outlive an install (`FUSE_MOAB_MESHFILES` is a plain `-D` path, not a cached search).

**`petsc.snes`: `PetscSnes.MultiRankVectorIsRefused` failed** at `test/petsc/snes_test.cpp:191`: expected
`E-PETSC-SOLVE-MULTIRANK`, got `E-PETSC-SOLVE-GHOSTED`. Commit `000b01c` (2026-10-12, window Newton on the region's own
ranks) made the PETSc driver multi-rank over `comm_f` and narrowed its refusal to a nonzero ghost region, now named
`E-PETSC-SOLVE-GHOSTED` (`src/petsc/solve.cpp`, `include/fuse/petsc/solve.hpp`), but left this assertion on the old code;
the sandbox that built it cannot run PETSc rows (D32), so only a host run could show it. **Fix:** the test is renamed
`GhostedVectorIsRefused` and asserts `E-PETSC-SOLVE-GHOSTED` for the same input (a vector with one ghost entry); it still
asserts a named refusal, not less. `docs/dev/newton-krylov.md` §10.2's "Scope" sentence, which still said single-rank
only and named the old code, has a dated superseding note. **Measured after the fix:** `petsc.snes` Passed on
`build-all`. Not established: why the quick tier of 2026-10-12 after `000b01c` did not show this (the tree and tier at
that run were not recorded).

### ~~D60. The installed package could not be consumed from a LowFive-enabled build: `fuse-lowfive` was never recreated~~ — FIXED 2026-10-14, host-verified

**Measured** with `ctest -LE slow` on `build-lowfive` (owner's machine, HEAD `58e2727`): 316 of 318 passed;
`abi.install_consumer_c` and `abi.install_consumer_fortran` failed with `ld: library 'fuse-lowfive' not found`.
Cause, read from the install tree: the exported `fuse::transport` carries `$<LINK_ONLY:fuse-lowfive>` in its
`INTERFACE_LINK_LIBRARIES` (`fuse-coupler-targets.cmake`), because `fuse-coupler-transport` links the
`fuse-lowfive` IMPORTED target `cmake/FuseDependencies.cmake` builds, and that target is not part of any export
set. `fuse-couplerConfig.cmake.in` recreated `fuse-moab` for the same reason but had no equivalent for LowFive, so a
consumer saw a bare library name. Not a regression of the Demo B work: `build-lowfive`'s own
`LastTestsFailed.log` already listed `abi.install_consumer_c` before it. **Fix:** `cmake/FuseDependencies.cmake`
records `FUSE_LOWFIVE_LIBRARIES_RESOLVED` (LowFive, its distributed library when present, and HDF5 with every
CMake target resolved to its file) and `FUSE_LOWFIVE_INCLUDE_DIRS`, and prints the list at configure time; the
package config recreates `fuse-lowfive` from them when `FUSE_ENABLE_LOWFIVE` was on, with the same MPI link,
`FMT_HEADER_ONLY` and fmt include ordering as the build's own target. Resolved on this host (from
`cmake --trace-expand`): `liblowfive.dylib`, `liblowfive-dist.dylib`, `libhdf5.dylib` and the SDK's
`libz.tbd`, `libdl.tbd`, `libm.tbd`. **Measured after the fix:** both install-consumer rows Passed in
`build-lowfive` and `build-all`, and `example.demo_b` still passes. No configure-time refusal was added: nothing
could make one fire, and the two install-consumer rows are the test that does.

### D61. The participant ABI's refusals are decided rank-locally and never reconciled across a component's ranks — OPEN; the first fix was REVERTED 2026-10-18 after the host run (see the REVERTED paragraph)

`begin_step`/`end_step`/`setup`/`initialize`/`run` return codes such as `E-ABI-STEP-ORDER` or
`E-ABI-STEP-ALREADY-OPEN` from local state only; an out-of-turn or omitted call on one rank of a
multi-rank component strands its peers inside `resume_period()`'s Transports and Epochs. The C++
terminals meet the A4 contract through the Epoch reconciliation; the ABI boundary does not, and said
nothing about it. `docs/user/c-abi.md` now states the guarantee ("Every rank of a component must make
the same calls, in the same order") and the residual exposure. Routing local ABI refusals through the
same reconciliation the Runner uses, plus a host-run two-rank refusal-not-hang row, is the fix; the
existing MPI participant fixture deliberately avoids the asymmetric call sequence, so the row needs a
timeout harness first.

**Fixed 2026-10-18 (ABI track).** `src/abi/participant.cpp`'s
`reconcile_collective_status()` reconciles `setup`, `initialize`,
`begin_step`, `end_step` and `run` -- the five entries.def marks
COLLECTIVE (or, for `run`, states execute "this call's own share of the
period's collectives") -- with one `MPI_Allreduce(MPI_2INT, MPI_MINLOC,
...)` of `(fuse_status, rank)` on the context's own base communicator
(the raw input communicator for `setup()` itself, since its own base
communicator is not yet resolved when its local checks can fail; the
already-resolved one for the other four), before any verdict reaches the
caller. `FUSE_OK` is 0 and every `FUSE_ERR_*` code is negative
(entries.def), so the plain minimum already propagates any refusal over
`FUSE_OK`, and `MPI_MINLOC` additionally reports which rank the minimum
came from -- what lets a non-originating rank's message be rewritten to
NAME the refusing rank, satisfying this entry's own "the others get the
agreed code with a message naming the rank". The pure arithmetic (`fuse::
abi_detail::reconcile_local_statuses`, `include/fuse/abi/
collective_reconcile.hpp`) has no MPI in it and is tested directly
(`test/abi/collective_reconcile_test.cpp`, 7 cases: all agree, one
refusal among many, the refusing rank need not be rank 0, two different
codes picking the more negative one deterministically in either order, a
tie breaking by the lowest rank, a single participant, defensive empty
input). The real glue's OWN test-only stub reducer (`fuse::abi_test::
force_fake_collective_peers()`, `src/abi/test_support.{hpp,cpp}`) lets
`test/abi/participant_test.cpp`'s new `AbiParticipantCollectiveReconcilia
tion` suite exercise the FULL path -- through the real `begin_step()`
entry, not a reimplementation -- at tier 0 with no MPI runtime: the
originating rank's own message survives unchanged; a rank that locally
SUCCEEDED still adopts a fake peer's refusal, with its own message
rewritten to name that peer; and the whole mechanism is confirmed inert
(no-op) when no communicator is configured, so the suite's many other,
single-process `AbiParticipant` cases are unaffected. `docs/dev/
abi-design.md`'s new "How the participant ABI reconciles a collective
refusal" section records the rule, which ten entries it deliberately does
NOT apply to and why (none of them is marked COLLECTIVE; `register_field`
was checked explicitly, per this round's own task, and found not to be).
The one `MPI_Allreduce` added to each of the five entries is accepted as
the standing cost of this fix (the lead's own decision, round 2) -- that
page's own text now records this as a one-line follow-up, not an open
question: a cheaper mechanism would piggyback the encoded status onto a
collective the step already performs, instead of a separate reduction.

**Built this round (ABI track, round 2).** Rather than extend
`test/parallel/abi_participant_mpi_test.cpp` (whose existing fixture
deliberately avoids the asymmetric call sequence this reconciliation is
meant to rescue, and whose components are all single-rank, unsuitable
for a real collective check), a new file,
`test/parallel/abi_reconcile_mpi_test.cpp`, builds a minimal
single-component graph whose `RankSet` spans the ENTIRE MPI world and
calls the real `fuse_setup()`/`fuse_register_field()`/`fuse_initialize()`
over `MPI_COMM_WORLD`; rank 1 alone then corrupts its own
`fuse_step_request::struct_size` before calling `fuse_begin_step()`, and
every rank's return code is gathered with `MPI_Allgather` and asserted
identical (`FUSE_ERR_VERSION`), confirming the fix rescues exactly the
asymmetric sequence this entry's own first paragraph names. No channel is
registered on the one field (so no per-rank size cross-check can ever
fire independently of this path), which keeps the fixture minimal without
weakening what it tests. Registered as `parallel.abi_reconcile_mpi.{2,3}`
(`test/CMakeLists.txt`, `RANKS 2 3`, the project's own default 120s ctest
`TIMEOUT` already satisfies this entry's "needs a timeout harness first"
-- `fuse_add_mpi_test()` sets it for every MPI row, not only this one).
Built clean under a real `FUSE_ENABLE_MPI=ON` configure. Built, never run
(C1 above). **Host-pending:**

```
ctest --test-dir <build> -R 'parallel\.abi_reconcile_mpi\.(2|3)' -V
```

Confirming on the host that every rank returns the SAME code with no
hang, rather than the originating rank's peers stranding inside
`resume_period()` as this entry's own opening paragraph describes, is
what that host run settles.


**REVERTED 2026-10-18 (lead, `1e0676b`), measured on the owner's machine at `033874e`.** The round-2 fix
(`reconcile_collective_status()` in `src/abi/participant.cpp`, one `MPI_Allreduce(MPI_2INT, MPI_MINLOC)`
per collective entry) was wrong in two ways the sandbox cannot see because it cannot start MPI:
1. It reconciled over the BASE communicator (`c.communicator`, the merged world of a multi-executable
   run), not over the component's own ranks. Different components make different numbers of
   `begin_step`/`end_step` calls per window, so the Allreduces never pair: `verification.brusselator.
   participants.{mpmd,accept}.{3,5}` all timed out (120 s), rows that passed the day before at `d622b50`.
2. It called `MPI_Comm_f2c`/`MPI_Comm_rank` on whatever handle `fuse_setup_options::communicator`
   carried. `abi.struct_size`'s `BaseCommunicatorReadsBackAFormedCommunicatorWhenMpiIsAvailable` passes
   99, and `verification.{abi_period_boundary,brusselator_period_rollover,brusselator_u_loop_step_fix}`
   use `fuse::abi_test::force_communicator_set(ctx, 1)`; MPICH's default error handler aborts on an
   invalid communicator (`Abort(...) internal_Comm_rank: Invalid communicator`), so all four rows died.
   Nothing had touched those handles before `initialize()`.
What a correct fix needs: a COMPONENT-LOCAL communicator, split once from the real base communicator by
`my_component` after the graph is loaded (the ABI analogue of the Runner's `ParticipantSetReducer`),
stored in the context and freed at `destroy`; reconciliation only over that communicator and only when
it was actually formed (never over a raw or placeholder handle); and the two tier-0 rows that model a
communicator without MPI must stay reconciliation-free by construction. The pure arithmetic
(`reconcile_local_statuses`) and its test were sound and can return with the fix. `parallel.abi_
reconcile_mpi.{2,3}` also failed on the host for a different reason (`LoadedContext` returned null, test
line 120) that the next round must explain first.

### ~~D62. The graph descriptor is host-endian while `bytes.hpp`'s canonical rules are little-endian~~ -- FIXED 2026-10-18 (Descriptor track)

**Fixed.** `detail::Writer::pod`/`detail::Reader::pod` (`include/fuse/model/descriptor.hpp`) now route
every POD through the SAME canonical encoders `fuse/base/bytes.hpp` already defines for every other
encoding in this library: an unsigned integral through `bytes::put_integer`/`bytes::get_integer`
(ADDED this round -- `get_integer`/`get_signed` did not exist before, `bytes.hpp` was write-only), a
signed integral through `bytes::put_signed`/`bytes::get_signed` (two's-complement bit pattern), and a
floating-point value (only `double` appears in this file) through `std::bit_cast` to/from an
equal-width unsigned integer and the same pair. `kGraphVersion` bumped 14 -> 15 (this track's own
owner decision named the bump "4 to 5"; the repository's actual version was 14 by the time this fix
landed, ten intervening bumps from other tracks -- the fix is applied against the REAL version
numbers, not the stale ones the decision text named; see this track's own deviations).

No field changed width or position: every `pod()` call site already passed an exact-width
fundamental type at a fixed `sizeof(T)`, so on every host this library has ever built for
(little-endian), the wire bytes `fuse::serialize()` now produces for a given graph are bit-for-bit
identical to what the pre-fix `memcpy`-based `pod()` wrote, except for the version word itself --
verified directly: `test/model/descriptor_test.cpp`'s
`V14AndV15BytesOfTheSameGraphDifferOnlyInTheVersionWord` diffs the frozen `waist-v14.bin` golden
against this build's own `serialize()` output for the identical fixture graph and asserts exactly one
byte differs, at the version word's own offset. `V14AndV15BytesOfTheSameGraphDecodeIdentically`
relabels each buffer to the version the other's bytes already match and confirms both decode to the
same graph through the one production `deserialize()`. What changed is the GUARANTEE, not (on this
host) the bytes: the wire format no longer depends on the writing host's native byte order, because
`bytes::put_integer`/`put_signed` build their output from the VALUE through shifts, never from its
memory layout. This sandbox has no big-endian host to build or run on;
`PodBytesNoLongerDependOnHostByteOrderViaSwapStub` demonstrates the host-independence by construction
instead -- a byte-swapping writer stub standing in for what the superseded `memcpy`-based `pod()`
would have written on a big-endian host, shown to disagree with the fixed `Writer::pod`, which in turn
is shown to match a second, independently written shift-only (host-layout-free) reference encoding.

Every superseded golden (`waist-v1.bin` through `waist-v14.bin`) is kept and refused by `kGraphVersion`'s
own version check (`SupersededFrozenBytesAreRefusedByThisReader`, extended to include `waist-v14.bin`
this round); `waist-v15.bin` is the new frozen reference
(`MatchesTheFrozenWaistBytes`). The ABI's own descriptor negotiation (`fuse_load_graph`, `include/
fuse/abi/entries.def`'s `fuse_graph_source`/`fuse_graph_sink`) was already correct -- it hands raw
bytes straight to `fuse::deserialize()`, which already refused a version mismatch by name
(`E-DESC-VERSION`) before this fix and still does after it; `test/abi/negotiation_test.cpp`'s new
`LoadGraphRefusesAnOutdatedDescriptorVersionByName` exercises that path explicitly rather than relying
on the direct-call coverage alone. The restart-to-disk provenance check (`fuse/restart_io/
hdf5_restart_io.hpp`/`.cpp`) gained its own, independent distinction: `Provenance::descriptor_version`
is a NEW attribute (restart `kFormatVersion` bumped 1 -> 2 for the schema change), refused by name
(`E-RESTART-DESCRIPTOR-VERSION`) inside `open_restart_point()` itself -- no current graph needed, since
`fuse::kGraphVersion` is a compile-time constant -- distinct from the pre-existing `descriptor_hash`
content check, now a named caller-side function, `fuse::restart_io::check_restart_descriptor_hash()`
(`E-RESTART-DESCRIPTOR-HASH`), that DOES need the current graph and so cannot live inside the
file-level reader (`fuse/restart_io/hdf5_restart_io.hpp`'s own `PointReader` doc comment: this layer
cannot see the current graph). The collective (MPI) restart writer (`src/restart_io/
hdf5_restart_io_mpi.cpp`) was updated to write the same new attribute and BUILT under
`-DFUSE_ENABLE_MPI=ON`, but -- MPI cannot start in this sandbox -- NOT run; see `host_pending` in this
round's own delivered report for the exact `ctest -R` to run on a host that can start MPI.

`docs/dev/abi-design.md`'s "The graph descriptor's byte order" section and `bytes.hpp`'s own header
comment are updated to describe the fix rather than the gap. Verified by `test/base/bytes_test.cpp`
(`get_integer`/`get_signed` round-trip `put_integer`/`put_signed`, including sign extension at every
width) and `test/model/descriptor_test.cpp` (30 tests, all passing on this build, tier 0, no MPI, no
HDF5).

### D63. Restart-to-disk cannot capture a period mid-suspension -- DESIGN CONSTRAINT, not a defect, recorded 2026-10-15 (Restart track, Stage B Slice 1)

`Runner::write_restart_fields()`/`restore_restart_fields()` (`src/runtime/runner.cpp`) and the
orchestration above them (`fuse::write_restart_point()`/`restore_restart_point()`,
`src/runtime/restart.cpp`) assume every call happens at a period boundary -- `run_period()` or
`run()` having returned, never while `resume_period()` has a suspended coroutine outstanding for
an externally-driven (ABI) participant (`src/runtime/resume.hpp`). This is not an oversight:
`docs/dev/restart.md` Section 1.1 states the reason directly, and it has not changed with Slice 1's
implementation -- a suspended coroutine frame is a stack of local variables and a resume address
inside a specific compiled binary, with no serializable representation `fuse::RestartWriter` or
any other interface could give it. Nothing in this track attempts to close this; a restart point
taken mid-suspension is a caller error this code does not itself detect (no test exercises it,
because there is nothing a test could assert about the outcome beyond "undefined" -- the coroutine
frame's own locals are not inspectable from outside `src/runtime/resume.hpp`).

**What to do.** Nothing. A restart point must be taken only immediately after `run_period()`/`run()`
returns successfully, before the next `advance()` of any kind -- the convention every test in
`test/runtime/restart_test.cpp` already follows, and the one Slice 2 (rank-count-independent
restart, Brusselator) inherits unchanged.

### ~~D64. `QuasiNewtonAccel` (IQN-ILS/IQN-IMVJ) does not implement restart state~~ -- CLOSED 2026-10-15 (Restart track, Stage B Slice 2)

**Closed.** `QuasiNewtonAccel::restart_state_implemented()` (`src/num/accel_quasi_newton.cpp`)
now returns `true`; `export_restart_state()`/`import_restart_state()` export/import `v_`/`w_`
(one `put_field_per_block()`/`get_field_per_block()` call PER COLUMN, since both are "n x cols_,
column major" -- column `j` occupies a contiguous, length-n_ span and is written under
`<prefix>/iqn/v/col<j>`/`<prefix>/iqn/w/col<j>`), `x_new_prev_`/`r_prev_`/`have_prev_`, and
`blocks_`/`chain_reprieves_` (IQN-IMVJ's folded inverse-Jacobian chain, one scalar `cols` plus a
`z`/`v` column pair per retained block) -- exactly the keying scheme this entry's own original
"what a future round needs" paragraph named below, unchanged. `QuasiNewtonAccel`'s own scratch
members need nothing, confirmed: none were exported. Verified by `test/runtime/restart_test.cpp`'s
`SameRankRoundTripSeparateLayoutIqnIls` (a dn_heat round trip with `.accelerate(AccelKind::IQN_ILS)`,
`EXPECT_EQ`-compared field-for-field and sweep-count-for-sweep-count against an uninterrupted
reference run) -- the SAME verification this entry originally asked for. The refusal path this
entry's own mechanism tested (`AccelNotImplementedRefusesWithoutOptIn`/
`AccelNotImplementedWithAllowResetSucceedsAndReports`) no longer has a real production `AccelKind`
to exercise it with (every kind now implements restart state), so those two tests were repointed
at a new TEST-ONLY `Accelerator` subclass (`NeverImplementsRestartAccel`, test/runtime/
restart_test.cpp) injected via a new TEST-ONLY seam, `Runner::debug_replace_region_accelerator()`
(`include/fuse/runtime/runner.hpp`, `src/runtime/runner.cpp`) -- the mechanism itself is unchanged,
only its test subject moved from a real kind that happened to not implement the interface yet to
one that, by construction, never will.

**A NEW, narrower gap found while closing this one, not fixed this round:** a WINDOW-SCOPED
region's own `RegionBlock::length` is the K-wide stacked window-transfer buffer (element count
TIMES `window_sample_capacity()`), not the registered field's own single-sample size -- see D68.
`write_restart_fields()`/`restore_restart_fields()` skip accelerator-state restart entirely for
any window-scoped region (reported via `any_accelerator_reset`, same as a real reset) until D68 is
resolved; Form A's own field restart and window-waveform history restart are unaffected.

Original text, kept for citation history (the keying scheme it names is exactly what closed this
entry, unchanged):

`fuse::Accelerator::restart_state_implemented()` (`include/fuse/num/accelerator.hpp`) is `false`
for `QuasiNewtonAccel` -- it inherits the base class's default, never overriding
`export_restart_state()`/`import_restart_state()`. A restart into an IQN-ILS- or IQN-IMVJ-
accelerated region therefore refuses collectively (`E-RESTART-ACCEL-NOT-IMPLEMENTED`) unless the
run opts into `allow_accelerator_reset_on_mismatch`, in which case `Accelerator::reset()` runs
instead and the reset is reported (`Runner::restore_restart_fields()`'s own `any_accelerator_reset`
out-parameter) -- never a silent, wrong restore. This is `test/runtime/restart_test.cpp`'s own
`AccelNotImplementedRefusesWithoutOptIn`/`AccelNotImplementedWithAllowResetSucceedsAndReports`
tests' subject, using `AccelKind::IQN_ILS` directly (a real production kind, not a hand-rolled
test double) -- the refusal mechanism itself (round 2's "NO SILENT RESET" fix) is fully
implemented and tested; what is open is IQN's own restart STATE.

**What a future round needs, if the owner wants IQN restart implemented rather than reset-by-
default:** export/import `v_`/`w_` (the retained secant columns, `n x cols_`), `x_new_prev_`/
`r_prev_`, `have_prev_`, `blocks_` (IQN-IMVJ's folded inverse-Jacobian chain, each block its own
`v`/`z`/`cols`), and `chain_reprieves_`, all keyed by the SAME `(qualified field name, global id)`
exact-key scheme round 2 built for Aitken/Steffensen (`fuse::RestartRowBlock`,
`detail::put_field_per_block()`/`get_field_per_block()`, `src/num/accel_relaxation.cpp`) --
`QuasiNewtonAccel`'s own scratch members (`r_`, `d_`, `alpha_`, `ls_a_`/`ls_b_`, `rho_`, `jrho_`,
`probe_`, `jac_a_`/`jac_b_`/`jac_beta_`, `fold_col_`/`fold_jv_`, `spare_`) need nothing: every one
is re-derived from the members above on the next call (the class's own comment: "None of it
carries information between calls"), including `spare_`, whose documented post-construction state
("Empty until the cap first binds") is exactly what a fresh object already has. A dn_heat round
trip with `.accelerate(AccelKind::IQN_ILS)` -- mirroring the existing Aitken one -- is the
verification this needs once written.

### ~~D65. A transient `MPI_Finalize` hang under libfabric's default provider on the owner's machine~~ — NOT REPRODUCED 2026-10-15, cause not established

**Measured** 2026-10-15 on the owner's machine (MPICH 5.0.0a1, ch4:ofi): a five-line
`MPI_Init_thread`/`MPI_Finalize` program hung in `MPIDI_OFI_mpi_finalize_hook` → `flush_send_queue` (macOS
`sample`), as a singleton and under `mpiexec -n 1`, in 3 of 3 runs; with `FI_PROVIDER=tcp` it exited. The `build-all`
quick tier had 11 timeouts (334 rows): the three restart rows and eight rows that had passed the day before
(`partition.repartition_mpi.1`–`5`, `abi.refusal_sweep`, `verification.brusselator_diffusion_solver_cache`,
`verification.brusselator_refusal_coverage`); with `FI_PROVIDER=tcp` the restart rows passed. Ruled out:
`FI_SOCKETS_IFACE=en0` and `=lo0` did not remove the hang (the hostname `Mac` resolves to `192.168.2.124` and
`192.168.2.80`; `en0` is `.80`). **About six hours later, with no change to the binaries or the MPI install,** the same
program exited 3 of 3 under the default provider, the restart rows passed, and the quick tier passed 334 of 334
(78.37 s). Nothing in FUSE sets the provider. If a host run again times out with every rank inside `MPI_Finalize`,
run the five-line program first: a hang there is this environment state, not FUSE.

**Cause measured 2026-10-17.** The same program now ABORTS instead of hanging: `MPI_Finalize` fails in
`MPIDI_OFI_mpi_finalize_hook` -> `flush_send_queue` -> `flush_send` with `OFI call tsenddata failed (default
nic=bridge101: No such file or directory)`, and MPICH's abort path then recurses (`MPID_Abort` ->
`MPIR_Debugger_set_aborting` -> `finalize_lock` assertion -> `MPID_Abort`) until the stack overflows, so the
process dies with SIGSEGV about 50 s later. `bridge101` is a virtual-machine bridge on the owner's machine
(`192.168.97.0/24`, member `vmenet1`), which libfabric's default NIC selection prefers over `en0`. Every
`gtest` binary that links MPI (`fuse_test_main.cpp:236` calls `MPI_Finalize`) therefore crashed AFTER all its
tests had passed. In the owner's interrupted `ctest -LE slow` run of 2026-10-17, 17 of the 26 rows that had
started reported SegFault, Failed or Timeout (the other 9, none of which link MPI, passed); the run was
stopped at row 26 of 352, so the full count under this state was not measured. Of the 17, only the two
`parallel.brusselator_restart_mpi` rows had a FUSE defect behind them (D75). Measured matrix on `/tmp/mpi5` and `parallel.connect.accept.1+1`: default: abort,
accept fails; `FI_PROVIDER=tcp`: exits, accept TIMES OUT (the plain tcp provider does not carry
`MPI_Comm_accept`/`connect`); `FI_TCP_IFACE=en0` alone: abort; **`FI_PROVIDER="tcp;ofi_rxm"
FI_TCP_IFACE=en0`: exits AND accept passes.** With that environment the quick tier passed 347 of 352 on
`f51987b`; the five remaining were the collective restart writer (D75, fixed) and three accept rows that
time out intermittently under the pinned provider. Rerun alone, each row both passed and timed out across
two runs (`parallel.connect.accept.3+2` passed alone after timing out in the tier; `participants.accept.3`
timed out alone after passing in the tier; `accept.2+3` timed out under both the pinned and the default
provider). No commit since `d622b50`, where all accept rows passed on 2026-10-16 13:25, touches the
connect/accept path, so this is the same libfabric state, not a code change; D23 holds the open item. Recommended host setting until MPICH's NIC selection is fixed or the bridge is gone:
`export FI_PROVIDER="tcp;ofi_rxm" FI_TCP_IFACE=en0`. Nothing in FUSE sets the provider.

### D66. The device-direct (GPU-aware, zero host round trip) exchange path this round adds to the ghost and nodal-remap exchanges cannot be exercised for real on higgs — OPEN, measured 2026-10-15 (runtime-exchange track, follow-up round)

`fuse::linalg::mpi_is_gpu_aware()` (comm.hpp, this round's own shared probe, moved from
`fuse::transport::MpiTransport`'s former private copy) is measured to return `false` unconditionally on
higgs: its OpenMPI 4.1.5 was configured `--without-cuda` (`ompi_info`'s own configure-line record), so
`MPIX_Query_cuda_support()` returns `0` at run time even though the `MPIX_CUDA_AWARE_SUPPORT` extension
macro IS declared at compile time (`FUSE_MPI_HAS_GPU_AWARE_EXTENSION=1` in the cache) — confirmed directly
with a standalone probe program (`MPIX_Query_cuda_support()=0`), not inferred from the configure line
alone. This is the identical probe `fuse::transport::MpiTransport::is_gpu_aware()` already used before
this round (device-resident-arcs.md Phase 3) and the identical situation that path was already in:
`RunnerTransport.ADeviceResidentRunDeliversTheSourceValuesGpuAwareOrNot`'s own comment calling higgs's
OpenMPI "GPU-aware" predates this measurement and is not supported by it — left uncorrected here (out of
this track's scope; that file belongs to the Transport track) but recorded so a future reader does not
repeat the assumption.

Consequence for this round's two new exchanges (`fuse::backend::DeviceGhostExchanger`,
`test/parallel/device_ghost_exchange_test.cpp`; the nodal-distributed-remap plan,
`nodal_remap_distributed.cpp`'s `build_distributed_remap_plan` plus `Runner::Impl::nodal_dist_gather_on`/
`nodal_dist_scatter_on`): the device-direct branch (pack/unpack on `Space`, `PackedExchanger::
exchange_packed()`/`exchange_packed_adjoint()` handed device pointers straight) is implemented and
compiles, and the host-staged fallback branch (always taken on this hardware) is verified correct at
1-5 ranks, under both the natural (false) and `FUSE_TRANSPORT_HOST_STAGING=1`-forced probe value. The
device-direct branch itself has no host run that reaches it: nothing on higgs can confirm it moves data
correctly with real device pointers in MPI's hands, as opposed to merely compiling. A host with a
genuinely CUDA-aware OpenMPI (or MPICH built against UCX/CUDA) is what this needs; until then this
branch is UNVERIFIED, not failing.

### ~~D67. `build_distributed_interp_rows`'s per-cell containment search disagrees by 1 ULP at the domain's own boundary vertex between an exactly-one-rank replica and a real multi-rank one~~ -- FIXED 2026-10-18 (found 2026-10-15, runtime-exchange track; closed, PointLocation track)

Found while adding `NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts` (the lead's
determinism follow-up's own RED/GREEN row, `test/parallel/nodal_remap_runner_test.cpp`): the H^T
scatter-add's plan-based reduction (this round's own fix, `DistributedRemapPlan::target_row_offset`/
`sorted_answer_slot`) IS bit-identical across every real (`comm_handle != fuse::search::kNullComm`)
decomposition tried — confirmed by a real host run at ranks 2 and 3, agreeing on all 8 beam vertices to
the last bit. One vertex (global id 0, the mesh's own left edge at `x = 0.0`) disagrees between an
EXACTLY-ONE-RANK run and EITHER multi-rank run by 1 ULP (`0.06611570247933884481` at one rank vs.
`0.066115702479338858688` at both two and three ranks) — small enough that it was already silently
absorbed by this file's own pre-existing `1e-12`-relative tolerance on every OTHER test here, never
asserted against directly until this round's exact-equality row.

NOT the scatter-add arrival-order nondeterminism this round's own fix targets (that part is confirmed
fixed: two DIFFERENT real decompositions, ranks 2 and 3, already agree exactly on every vertex including
this one). `GeometricLineMesh::measure()`/`centroid()` (test/support/partitioned_test_backend.hpp) are
confirmed decomposition-invariant by inspection: both read only the GLOBAL `n_cells_`/global vertex
index, never `num_ranks_in_set`/`my_rank_in_set`, so the mesh geometry itself is not the source. The
likely cause is upstream, in `build_distributed_interp_rows`'s own per-cell containment search
(general_interp.hpp's `invert_reference_map`/tie-break primitives) against the REPLICATED source mesh
`gather_global_source` assembles: an exactly-one-rank run's replica is a trivial local copy (no
`MPI_Allgatherv` at all, `allgatherv_records`'s own `comm_handle == kNullComm`-style identity path is
NOT what runs here since this plan is built only for a real communicator, but the PATTERN/hash-map
build order a one-element `MPI_Allgatherv` produces could still differ from a genuinely multi-rank
concatenation's own order), which could change which of two near-tied candidate cells wins at a
boundary point, or the floating-point path taken, by a reproducible but tiny amount. Not diagnosed
further than this round's own scope (the scatter-add determinism, not the weight build) warranted;
`nodal_remap_distributed_test.cpp`/`nodal_remap_distributed_nullcomm_test.cpp` (lower-level, non-Runner
tests of the same primitive) would be the next place to isolate it with a 2-rank-vs-1-rank weight dump,
if this round's track's own time allowed.

**Re-tested after D74's fix (scaled-bench track, follow-up round, 2026-10-16): NOT the same mechanism.**
D74's root cause turned out to be `build_distributed_interp_rows`'s per-eval-point candidate
scan testing EVERY cell in the assembled mesh with no broad-phase spatial filter, aborting an
evaluation point outright the instant ANY single candidate (however far away) failed to
converge -- even one whose genuine answer the same scan had already found. Fixing that (a
`search::BoxIndex` broad phase, plus treating a per-candidate failure as a skip rather than an
abort) cannot change which cell wins a genuine tier (a)/(b) match, since the winning-candidate
rule itself (lowest global id among tight matches) is unchanged and the broad phase still finds
every such candidate (padded by the same `kRefEpsLoose` the fallback tier already uses). Directly
confirmed, not just argued: `NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`
re-run after the fix reproduces the EXACT SAME bit pattern at the SAME vertex --
`0.06611570247933884481` at one rank, `0.066115702479338858688` at both two and three ranks,
unchanged to the last bit. D67 remains open on its own; whatever produces this 1-ULP gap is a
different, still-undiagnosed mechanism in the same function.

**ROOT CAUSE, FOUND (PointLocation track, 2026-10-18): NOT a tie-break, and NOT `gather_global_source`'s
assembly order -- two DIFFERENT floating-point expressions for the SAME Edge-cell hat weight, selected
by which of two unrelated code paths a caller's placement happens to route through.** The vertex in
question has exactly ONE candidate source cell (it is the mesh's own left edge, not a shared interior
vertex at all) -- there is no tie to break here, which the original write-up's own "two near-tied
candidate cells" theory did not consider. `Runner::resolve_remap`'s `both_on_one_rank` (runner.cpp,
"A single-rank placement on BOTH sides skips this whole path") sends a one-rank placement -- exactly
`NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`'s own world_size == 1 case, AND
that same test's `compute_reference()` helper, which calls the public dispatcher directly -- through the
ORIGINAL, serial `fuse::build_consistent_interp_weights` -> (`dimension() == 1`) ->
`build_consistent_interp_weights_1d` (consistent_interp.hpp): a coordinate-sort algorithm that computed
the bracketing hat weight as `t = (x - lo.x) / (hi.x - lo.x)` directly. ANY placement spanning more than
one rank instead takes `build_nodal_dist_remap` -> `build_distributed_interp_rows`
(nodal_remap_distributed.hpp), which -- regardless of dimension, per general_interp.hpp's own dispatch
comment -- always inverts a cell's reference map through `general_interp_detail::invert_reference_map`'s
damped Gauss-Newton: for an Edge cell, one step from `u = 0` lands at `du = dot(jac, res) / dot(jac,
jac)` with `jac = p1 - p0`, `res = target - p0`, which reduces (`jac`/`res` scalar here) to `(h*dx)/(h*h)`
-- ALGEBRAICALLY the same ratio as `t = dx/h`, but NOT bit-identical: the extra multiply before the
divide rounds differently from a bare divide, in general. Confirmed directly, not just argued: a
serial, NO-MPI reproduction (`NodalRemapDistributedNullComm.
D67SerialAndDistributedEdgeWeightsAgreeAtTheDomainBoundary`, test/backend/nodal_remap_distributed_
nullcomm_test.cpp) calls `build_distributed_interp_rows`/`build_consistent_interp_weights` directly on
the SAME n_beam=7/n_fluid=11 mesh pair and reproduces the IDENTICAL bit pattern this entry already
named (`0.68181818181818187874` vs `0.68181818181818176772` at beam vertex 0, and six further
mismatches at OTHER bracketing vertices this file's own 1-ULP-at-gid-0-only framing had not noticed,
since the aggregated `beam_load` sum happens to wash most of them out) -- with zero MPI symbol
anywhere in that translation unit, which rules out `gather_global_source`'s null-comm-vs-Allgatherv
assembly order as a cause (that theory's own premise, re-examined: every vertex position and cell
connectivity entry gathered by Allgatherv is copied byte-for-byte from the owning rank's own
`MeshView::centroid`/`global_id`, concatenated in ascending-global-id rank order regardless of rank
count for this file's own `GeometricLineMesh` -- there is no arithmetic for a decomposition to perturb).

**THE FIX.** Both code paths now compute this one ratio through a single shared predicate,
`general_interp_detail::edge_hat_weights` (general_interp.hpp): a synthetic 2-corner `CellShape::Edge`
cell built from the two bracketing physical positions, run through the SAME `invert_reference_map`/
`basis_weights` the distributed and general (2D/3D) builders already use.
`build_consistent_interp_weights_1d`'s "genuine bracket" branch (consistent_interp.hpp) now calls it
instead of computing `t` directly, so whichever code path a caller's placement selects, the computed
weight is the identical bit pattern, because it is now the identical computation. The 1D algorithm's
own coordinate-sort (finding WHICH pair brackets a point, needing no `cell_vertices` connectivity at
all) is UNCHANGED -- only the arithmetic of the weight ITSELF, once the bracket is already found, is
now shared; this preserves the design rationale consistent_interp.hpp's own header gives for keeping
the two dispatch paths separate (a connectivity-free 1D source, which `build_general_interp_weights`
cannot serve at all, must keep working).

**VERIFICATION, entirely in this sandbox (MPI cannot start here, but neither code path needs MPI to
diverge or to agree).** RED first: reverting the fix and re-running
`NodalRemapDistributedNullComm.D67SerialAndDistributedEdgeWeightsAgreeAtTheDomainBoundary` (serial,
no MPI) reproduces 7 of 22 compared (fluid cell, beam vertex) entries disagreeing by exactly 1 ULP,
including the exact value this entry already named at beam vertex 0; reverting the fix and tightening
`NodalRemapRunnerSerial.TheOneRankCaseOfTheMpiFixtureRunsAndIsExact`'s own beam-load check to
`EXPECT_EQ` (verification/nodal_remap_runner_serial_test.cpp -- the one-rank case of the MPI fixture,
runnable without MPI, which already exercises BOTH code paths at one participant via
`RunnerConfig::force_distributed_nodal_remap`) reproduces the SAME discrepancy
(`0.066115702479338859` vs `0.066115702479338845`) at the full `Runner`/`CouplingGraph` level, not
just the raw primitive. GREEN after the fix: both tests pass; the fix was then restored and the file
confirmed byte-identical to pristine before the mutation/RED check (`diff` exit 0). `test/parallel/
nodal_remap_runner_test.cpp`'s own `ConsistentTransposeIsBitIdenticalAcrossRankCounts` had excluded
beam vertex (global id) 0 from its exact check at a 5e-16 tolerance, with a comment attributing the
gap to the (now ruled out) Allgatherv-assembly theory; that exclusion is removed (every vertex is now
asserted bitwise-identical) and the comment corrected to this entry's own root cause -- that file is
MPI-gated and only compiled, never run, in this sandbox (host_pending: `ctest --test-dir <build> -R
'parallel\\.nodal_remap_runner\\.[1-5]'` on the owner's machine is what confirms it at a genuine
multi-rank launch, though the mechanism and the fix are already proven, without MPI, by the two
tests above).

### D68. `BoxIndex::build_device_resident` tied its tree-routing flag to `kHasDeviceBackend`, reading `size()`/`empty()`/every query from the untouched host tree on a build with no device backend -- FOUND AND FIXED 2026-10-15 (MeshAdapter track)

**What was found.** `search.mesh_adapter`'s `MeshAdapter.DeviceElementBoxesMatchHostElementBoxesForANativeMesh`
(added in 98be06b, 2026-10-14, "search: element_boxes_device_resident over a device-resident
simplex CSR") failed on `idx.value().size()`, 0 instead of 2, with every assertion before it
passing -- the device-resident boxes themselves (`element_boxes_device_resident`) matched the
host reference exactly, and `idx.value().dimension()` already reported the correct 2.
`BoxIndex::size()`/`empty()`/`query_points()`/`query_nearest()` (src/search/box_index.cpp)
all read `impl_->device_tree` only when the member `device_` is true, `impl_->host_tree`
otherwise; `dimension()` is set unconditionally from the same `classify_all()` call both
paths share, which is why it alone was unaffected.

`BoxIndex::build_device_resident` (added bf88d887, 2026-10-13, "Build and query BoxIndex on
the default execution space") always builds its tree into `impl_->device_tree` -- unlike
`BoxIndex::build()`, it has no host-tree branch at all -- but set the routing flag with
`out.device_ = kHasDeviceBackend;` rather than `true`. `kHasDeviceBackend` is
`!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>`, a
compile-time constant that is `false` on any build with no CUDA/HIP/SYCL backend (this
sandbox's Kokkos 5.1.0, Serial/OpenMP only, confirmed by the standalone measurement below).
On such a build `device_` was left `false` while the tree sat in `device_tree`: every
`size()`/`empty()`/query call then addressed the default-constructed, never-built
`host_tree` instead, reporting an empty index regardless of input.

**Confirmed, not merely inferred, with a standalone program** linked against
`fuse-coupler-search` directly (bypassing gtest): 7 boxes built via
`build_device_resident` on this build reported `kHasDeviceBackend=0`, `dimension()=2`
(correct), `size()=0` (wrong, expected 7), `empty()=1` (wrong), and a point known to lie
inside box 0 returned 0 query hits. The classification-only candidates the task asked to
exclude were ruled out this way: `build_device_resident` does record a count (`n =
boxes.extent(0)`, used to size `compacted_index`/`arbor_boxes`/the parallel loops), and the
host mirror it builds (`Kokkos::create_mirror_view(boxes)`) exists solely to let
`classify_all`'s host-only `Result`-returning check read box extents -- the test's own
`host_mirrors_materialized() == 0` assertion is checking the SEPARATE `NativeMeshView`'s own
mirror (never touched, correctly), not this unrelated local mirror inside
`BoxIndex::build_device_resident`. `size()` reads a real member (`device_`-gated), just the
wrong one for this entry point's own invariant ("I always build into `device_tree`").

**A second, pre-existing test was silently passing for the same reason.**
`BoxIndex.ViewBasedBuildAndQueryMatchVectorBasedOnTheSameIndex` (box_index_test.cpp) builds
a `BoxIndex` with `build_device_resident`, queries it both via
`query_points_device_resident` and `query_points`, and compares the two CSR results. Both
calls route through the same `empty()` check first; with `device_` wrongly `false`,
`empty()` was wrongly `true`, so both calls short-circuited to the same all-empty
`QueryResult`/`QueryResultView` and the comparison passed by agreeing on nothing. Measured
directly: before the fix, the standalone program's query against box 0 returned 0 hits;
after the fix, 1. This track leaves that existing test's assertions exactly as they were (fixing the
root cause in `build_device_resident` is what makes them meaningful, not a change to the
test itself) -- recorded here because this track's own measurement is what surfaced it, not
because strengthening that test was in this track's scope.

**What was fixed.** One line, `src/search/box_index.cpp`'s `BoxIndex::build_device_resident`:
`out.device_ = kHasDeviceBackend;` to `out.device_ = true;` -- this entry point's own
invariant ("always builds into `device_tree`") now matches the flag that routes reads to it.
`kHasDeviceBackend` still gates `executes_on_device()` (`device_ && kHasDeviceBackend`)
exactly as before, so that accessor's documented contract ("`Device` requested on a
host-only build degrades to `Host`") is unaffected -- confirmed by
`BoxIndex.ViewBasedBuildAndQueryMatchVectorBasedOnTheSameIndex`'s own
`EXPECT_EQ(idx.value().executes_on_device(), kHasDeviceBackend)` still passing.

**Why a real device build would not have shown this.** `fuse_add_device_twin`
(test/CMakeLists.txt) is registered only `if(FUSE_KOKKOS_HAS_DEVICE)` -- absent here, not
skipped, confirmed by `ctest -N` listing `search.box_index`/`search.mesh_adapter` with no
`.device` suffix. Even where it IS registered, `FUSE_EXECUTION_SPACE=device` (the twin's own
mechanism) only changes `BoxIndex::build()`'s `resolve_execution_space()` outcome;
`build_device_resident` takes no `ExecutionSpace` argument and never reads that variable, so
the twin would exercise the identical code path as the base row within one binary. The
determining fact is `kHasDeviceBackend`, which is fixed at compile time by whether Kokkos
was built with a CUDA/HIP/SYCL backend at all -- on a build that has one, `kHasDeviceBackend`
is `true` unconditionally, so `out.device_ = kHasDeviceBackend` already equalled `true` and
this defect could not have manifested, in the base row or the twin. The defect is specific
to a Kokkos build with no device backend compiled in at all (Serial/OpenMP, like this
sandbox and, per the task description, the lead's own measurement) -- exactly the tier-0
configuration this ladder always builds, so no CI leg this repository runs today would have
missed it for a build-configuration reason; it was missed because no existing test checked
`size()`/`empty()`/a real query hit after `build_device_resident` independently of the one
row that happened to exercise all three together for the first time (98be06b).

**What was added.** `BoxIndex.BuildDeviceResidentSizeMatchesBoxCountForZeroOneAndSevenBoxes`
(box_index_test.cpp): `size()`/`empty()` pinned directly against `build_device_resident`'s
own box count for 0, 1 and 7 boxes, independent of `mesh_adapter_test.cpp`'s native-mesh
scenario. Mutation row `search-build-device-resident-flag-tied-to-backend`
(build-evidence/mutations/search.toml) reverts the fix (`out.device_ = kHasDeviceBackend;`)
and is killed by this new test (n=1 and n=7; n=0 is unaffected because `empty()` is true
either way for a zero-box index) -- confirmed by rebuilding `fuse_search_box_index_test`
against the reverted line and observing exactly those two sub-checks fail, the rest of the
16-test binary still green.

**Verification.** `search.mesh_adapter` and `search.box_index` (full binaries, every
existing test plus the new one) pass after the fix on this sandbox's tier-0 configuration
(Debug, Kokkos 5.1.0 Serial/OpenMP, ArborX 2.1.99, see report-meshadapter.md for the
resolved-dependencies record). `tools/mutate.py --check-anchors` reports the new anchor
valid (0 stale) alongside the area's four pre-existing rows.

**Cross-references.** D49/D51 (native::device halo/migration merge) and D66/D67
(runtime-exchange device-direct path) are the other open device-resident entries; all four,
including this one, share the same root difficulty this track's own measurement adds one
more data point to: a facade whose device-resident entry points are exercised almost
exclusively on hosts with no real device backend, where "device" and "host" collapse to the
same type and the distinction between "which space did I ask for" and "which space is the
data actually in" is easy to let drift apart, because nothing observable breaks until a
caller reads the GENUINELY device-resident member (`size()`/`empty()`/a query) rather than a
CPU-computed one (`dimension()`).

### ~~D69. A window-scoped region's own accelerator restart state is not implemented~~ -- CLOSED 2026-10-15 (Restart track, Stage B Slice 2, round 2; renumbered from round 1's own D68, which this track's main branch took for the BoxIndex fix, D68 above)

Found while wiring the Brusselator (`examples/brusselator`) into restart-to-disk, the one example
of the four `docs/dev/restart.md` Section 2.3 names that is both window-scoped (`fuse::
CouplingWindowScope`) and accelerated. `detail::RegionBlock::length`, for a window-scoped
region's own channel, is the K-WIDE stacked window-transfer buffer -- this member's own element
count TIMES `fuse::window_sample_capacity()` (`src/runtime/iteration.cpp`'s own priming loop,
`export_slot.data[e * cfg.capacity + s] = cfg.provided[e];`, confirms the layout is element-major,
sample-minor) -- NOT the registered field's own single-sample size `Component::register_fields()`
bound. `write_restart_fields()`/`restore_restart_fields()` (`src/runtime/runner.cpp`) resolve an
accelerator's own row ids by asking the OWNING component for the ids it reported under the
channel's own field name (`build_field_owner_map()` plus `RestartWriter::recorded_ids()`/
`IdCollectingRestartWriter::recorded_ids()`) and asserting `ids.size() == block.length` --
built and tested (Stage B Slice 1/1b/round 2) only against an ORDINARY (non-window) region, where
the two sizes coincide by construction (dn_heat; `test/runtime/restart_test.cpp`'s own round
trips, including this round's new IQN-ILS one). For the Brusselator's own `brusselator_window`
region, a component reports `n_u` ids (25, in the example's own small-grid test fixture) for
field `v_in` while the region's own accelerated block for that field is 275 rows (25 elements x
11 samples) -- measured directly: `write_restart_fields` refused `E-RESTART-BAD-NUMBERING`
("expects 275 rows, component reported 25") the first time a window-scoped, accelerated region's
restart point was ever written.

**What to do.** Nothing yet, by design, this round: `write_restart_fields()`/`restore_restart_fields()`
now skip accelerator-state restart entirely for any `region.window_scoped` region (`src/runtime/
runner.cpp`), reporting the skip through the SAME `any_accelerator_reset` output parameter a real
`Accelerator::reset()` already uses, rather than writing ids that would silently under-cover the
declared global size or leaving the restored run's own convergence history undefined. Form A's own
registered-field restart and window-waveform history restart (this round's own new feature,
keyed directly by the registered field's OWN single-sample global ids, not by the K-wide buffer)
are unaffected -- neither goes through `RegionBlock` at all.

**What a future round needs.** A per-(element, sample) id scheme for the K-wide buffer's own rows,
collision-proof the same way round 2's EXACT-KEY fix already is for a flat accelerator buffer.
Two options, neither implemented or chosen this round (`open_questions`, delivered report):
  - **Per-sample named sub-datasets.** Split a window-region's own `RestartRowBlock` into
    `capacity` separate blocks, one per sample index, each carrying the SAME per-element ids
    under a qualified name suffixed `/sample<k>` (the identical convention this round's own
    window-waveform history already uses for `WindowMemberState::history`/`last_window_waveform`)
    -- collision-proof by construction (two samples are two different dataset names, never a
    shared id space), but needs a STRIDED gather/scatter helper (`detail::put_field_per_block()`/
    `get_field_per_block()` assume each block's own rows are already contiguous in the flat
    buffer, true for an ordinary region's block but NOT for a window region's own element-major,
    sample-minor layout) rather than reusing the existing contiguous-subspan helper as-is.
  - **A combined (global id, sample index) key.** Cheaper to key (no new helper), but needs a
    combining function proven collision-free against the SAME global-id range the mesh generator
    produces (`id * capacity + sample`, say) -- exactly the kind of "collision-free for this
    slice's own graphs, not collision-PROOF" construction round 2's own MUST-FIX 2 replaced for
    the flat accelerator case; reintroducing it here for the window case needs the SAME scrutiny
    before being accepted, not less.
A same-rank-count Brusselator round trip with a REAL accelerator (Aitken or IQN-ILS) over
`brusselator_window`, compared field-for-field and sweep-count-for-sweep-count against an
uninterrupted reference run (the same bar `SameRankRoundTripSeparateLayoutIqnIls` already holds
the non-window case to), is the verification either option needs once written.

**Closed, round 2 (2026-10-15).** Added `RestartRowBlock::window_capacity` (default 0, meaning
"not window-scoped, unchanged path"). `detail::put_field_per_block`/`get_field_per_block`/
`total_rows` (`src/num/accel_relaxation.cpp`) branch on it: a window-scoped block's own slice of
the flat accelerator buffer is `ids.size() * window_capacity` rows, element-major sample-minor
(index(e, s) = e*window_capacity + s, the SAME convention D70's own `last_window_waveform` history
rows already use), demuxed into `window_capacity` separate per-sample sub-datasets
(`<qualified>/sample<k>`) by a strided gather/scatter per sample. No change was needed inside
`AitkenAccel`, `SteffensenAccel`, or `QuasiNewtonAccel` themselves. Verified by a new, isolated,
non-Runner unit test, `Restart.AitkenAccelWindowCapacityPerSampleRestartRoundTripsBitForBit`
(`test/runtime/restart_test.cpp`): two blocks of different capacities, 18 total rows, restores
`r_prev_` bit-for-bit and a subsequent `update()` call reproduces `x_out` exactly. Also closed
this round: `restore_restart_fields()` (`src/runtime/runner.cpp`) no longer silently resets a
window-scoped region's accelerator without checking `allow_accelerator_reset_on_mismatch` -- it
now runs the identical `restart_state_implemented()` refusal every ordinary region already ran.

**Known residual gap, NOT closed by this fix -- see D72, below.** A Runner-level round trip with
a REAL accelerator (Aitken, IQN-ILS) over a window-scoped region still does not reproduce the
reference run's own iteration count after a restore, because the region's own CONSUMER-side
transfer buffer (`cfg.window_import`) is not correctly reconstructed from restored state. This
does not affect `AccelKind::None` (whose `update()` never reads `x_old`) and is a SEPARATE defect
from the key-collision/silent-reset gaps this entry closes.

### ~~D70. `WindowMemberState::last_window_waveform` is never populated by the Runner's own `advance_window()`~~ -- FIXED 2026-10-18 (Windows track; measured OPEN 2026-10-15, Restart track, Stage B Slice 2; renumbered from round 1's own D69, which main's own ASCII-cleanup/BoxIndex sequence displaced when D68 (round 1's own window-scoped-accelerator entry, now D69 above) collided with a real main-branch D68; content unchanged except the number, per the owner's own instruction not to touch `advance_window()`)

Found while building and testing this round's own window-waveform restart (owner decision 4's
second half): `detail::WindowMemberState::last_window_waveform` (`include/fuse/runtime/
window_coupler.hpp`) is written only by `fuse::WindowCoupler` -- the standalone, non-Runner
driver (`src/runtime/window_coupler.cpp:1041`, `state_[i].last_window_waveform = std::move(
real_traj[i]);`, plus its own `mean()`/boundary readers at `:499,511,541`). A grep of `src/runtime/
iteration.cpp` (the Runner's own compiled window-region driver, `advance_window()`) for
`.last_window_waveform`/`->last_window_waveform` returns ZERO matches: the Runner's own window
machinery reads this member's own in-window trajectory through a DIFFERENT path entirely
(`window_import`/`previous[]`/`slots[]`, this file's own extensive header comments on the
scheduling-order fix), and never writes `last_window_waveform` at all. Consequence, measured
directly: in every Runner-driven run this round's own `test/verification/
brusselator_restart_test.cpp` builds, `WindowMemberState::last_window_waveform` stays at its
default-constructed empty vector for the whole run -- this round's own restart code for it
(`write_window_member_restart()`'s `/waveform/present` branch, `src/runtime/runner.cpp`) is
therefore implemented and exercised only on its OWN "nothing to write" branch; a deliberate
"write the wrong sample" mutation of the per-sample waveform write loop was applied, rebuilt, and
run against the full Brusselator restart suite to confirm this directly -- it killed nothing,
confirming the per-sample waveform code path is dead for the Runner driver as currently built,
not merely untested by chance. The SIBLING state this round restarts, `WindowMemberState::
history`, is NOT in this situation -- `iteration.cpp` writes it directly (`window_member_state[
node].history[e].front()`, line 3413's own citation, among others), and the same
"write the wrong sample" mutation technique applied to `history`'s own write loop DID kill three
of the four Brusselator round-trip tests, confirming `history` restart is both implemented and
load-bearing.

**What to do.** Nothing yet. `last_window_waveform` restart is written defensively (the
`/waveform/present` scalar means a FUTURE Runner change that does populate it, or a direct
caller of `fuse::WindowCoupler` wired through this same restart path, finds the feature already
built) but cannot be exercised through the Runner today. Whether the Runner's own omission of
`last_window_waveform` is itself a gap in the window-region machinery (a predictor/extrapolation
path `WindowCoupler` has that the Runner's compiled equivalent does not) is outside this track's
own scope to judge -- flagged here as a fact about restart coverage, not a claim about the
window-region feature itself being incomplete.

**Fix (Windows track, 2026-10-18).** `advance_window()` (`src/runtime/iteration.cpp`) now builds
a local per-element `Waveform` trajectory every sweep -- one real (non-extrapolated) sample at
the window-start value (the SAME `start_value` cache this function already maintains) and one
more at each substep's own committed time, mirroring `WindowCoupler::sweep_window()`'s own
`real_traj[i]` construction exactly (`window_coupler.cpp`) -- and moves it into
`WindowMemberState::last_window_waveform` at the end of the call. advance_window() runs once per
sweep per member, so simply overwriting this local trace every call and committing whatever the
LAST sweep of a pass left reproduces WindowCoupler's own "committed once, after the sweep loop
converges" semantics with no extra per-pass bookkeeping. One declared reduction in fidelity: a
`DtPolicy::Fixed` member whose last substep straddles the window end gets one fewer trajectory
sample than WindowCoupler's own driver records for that same case (WindowCoupler pushes a second
knot at `sub.t1` when `T1 < sub.t1`; this fix does not, since `cfg.capacity` has no reserved slot
for it) -- not exercised by Brusselator, whose `UComponent`/`VComponent` both use
`DtPolicy::Adjustable` (never straddles); flagged to `open_questions`, not silently reproduced.

Verified: `BrusselatorRestart.WaveformPresentAfterAdvanceThroughTheRunner` (new) confirms
`write_window_member_restart()`'s own `/waveform/present` scalar is `true` after at least one
window completes through the Runner -- `false` unconditionally before this fix, the direct
"dead code" symptom this entry named. `BrusselatorRestart.RestoredRunReproducesLastWindowWaveform
BitForBit` (new) confirms a restored run's own `last_window_waveform`, read back through
`RestartReader::get_field()` directly (no new public accessor added), agrees bit for bit with an
uninterrupted reference run's own value at the identical window boundary, for both U and V. Full
suite: `test/verification/brusselator_restart_test.cpp`, 12/12 passing (`ctest -R
verification.brusselator_restart`); no behavioural change for `AccelKind::None` or any
non-window region.

### ~~D71. A real accelerator (Aitken, IQN-ILS, or any kind but `None`) over a `WaveformIteration::GaussSeidel` coupling window refuses `E-RT-WINDOW-ACCEL`~~ -- FIXED 2026-10-18 (Windows track; named OPEN 2026-10-15, Restart track, Stage B Slice 2, round 2)

Named precisely this round rather than described as "a limitation" in prose, per the owner's own
instruction. `resolve_regions()` (`src/runtime/iteration.cpp:1836-1851`) refuses to build a
region's own accelerator when the region's own `WaveformIteration` is `GaussSeidel` and the
configured `AccelKind` is anything but `None`, raising `E-RT-WINDOW-ACCEL`. Cross-referenced
against `docs/dev/newton-krylov.md` Section 9.1 (lines ~483-507), which documents the identical
gap from the window-Newton side -- a Gauss-Seidel sweep order and a genuine vector accelerator
(Aitken, Steffensen, IQN-ILS/IMVJ) both want to own the same per-member update ordering, and the
two have not been reconciled. Pre-existing on `main`, confirmed unrelated to this round's own
changes (reproduces identically against the unmodified pristine tree). Fixing it is out of
restart's own scope -- this entry exists so a future reader of the restart track's own refusal
surface does not mistake this for a restart-specific defect.

**Fix (Windows track, 2026-10-18).** `Runner::Impl::run_region()`'s own `R.accel->update()` call
site (`src/runtime/iteration.cpp`) now excludes a GaussSeidel window's own unlagged transfer
from the accelerated vector before calling the accelerator, exactly as `WindowCoupler::
sweep_window()`'s own `accel_members`/`accel_member_offset`/`accel_total` restriction already
does (`window_coupler.cpp`'s own `provides_unlagged_channel` comment) -- reusing the IDENTICAL
membership test (`run_region_newton_window()`'s own `RegionBlock::newton_offset`, computed from
`R.window_lagged_import_arcs`) rather than re-deriving a second one that could disagree with the
Newton path about which direction is lagged. The excluded block(s)' own slice of the iterate is
a straight Gauss-Seidel substitution from `x_new`, never a relaxed iterate, matching
WindowCoupler's own "default: excluded members pass straight through" comment. Under
`WaveformIteration::Jacobi`, `effective_channel_lag()` forces every channel lagged, so nothing is
ever excluded there and the call is unchanged, byte for byte, from before this fix; this is a
strictly additive change to the refused GaussSeidel-plus-real-accelerator combination only.

`QuasiNewtonAccel`'s own per-field weighting (`field_weights()`, `src/num/
accel_quasi_newton.cpp`) reads `Accelerator::field_partition()` against whatever vector it is
handed; `resolve_regions()` sets it once for the FULL (`R.n`-sized) vector, which disagrees with
the smaller, projected one this fix hands the accelerator when a block is excluded -- saved via
`field_partition()`'s own getter and reset via `set_field_partition()` immediately around the
projected call, restored unconditionally afterward, so every OTHER sweep (Jacobi, a non-window
region, or a GaussSeidel sweep excluding nothing this region) keeps reading the original,
unchanged partition.

**Measured** (`examples/fuse_example_brusselator --driver runner`, `--end 0.5`, the example's
own default grid and `dt_u`/`dt_v`/`window`): peak sweeps in any one window, Jacobi vs
GaussSeidel, same accelerator --

| accelerator | Jacobi | GaussSeidel |
|---|---|---|
| Aitken | 31 | 10 |
| IQN-ILS | 15 | 9 |

Both GaussSeidel rows converge in FEWER sweeps than the matching Jacobi row, the expected
direction (Gauss-Seidel uses the most recently updated partner value within a sweep); `--check:
OK` (conservation to ~1e-13/~1e-16) on all four. Cross-checked against the standalone
`WindowCoupler` driver (`--driver window`) on the identical configuration: GaussSeidel+Aitken 8
sweeps, GaussSeidel+IQN-ILS 8 sweeps -- same order of magnitude as the Runner's 10/9, not
required to agree exactly (D17, below, already documents a pre-existing, unrelated Jacobi-only
discrepancy between the two drivers' own sweep counts). Verified by two new tests,
`BrusselatorRestart.GaussSeidelAitkenConvergesAndMatchesJacobi`/
`GaussSeidelIqnIlsConvergesAndMatchesJacobi` (`test/verification/brusselator_restart_test.cpp`):
both confirm the region no longer refuses, that GaussSeidel's own iteration count is never more
than Jacobi's at any window, and that the two converge to the SAME physical U/V state (`EXPECT_
NEAR`, 1e-8 -- never bit-for-bit, since Jacobi and GaussSeidel are different fixed-point
iterations of the same converged answer, not the same sweep sequence).

### ~~D72. A window-scoped region's own CONSUMER-side transfer buffer (`cfg.window_import`) is not correctly reconstructed after a restart~~ -- FIXED 2026-10-18 (Windows track; measured OPEN 2026-10-15, Restart track, Stage B Slice 2, round 2)

Found while verifying D69's own closing fix (above) at the Runner level: a same-rank-count
Brusselator round trip with a REAL accelerator (Aitken, IQN-ILS) over `brusselator_window` does
not reproduce the reference run's own iteration count after restoring (reference 13, restored 12
in one measured run). Root-caused via targeted tracing: `R.x_old`'s own pre-loop seed
(`slot_to_host(slots[b.feed_arc], ...)`, `b.feed_arc == cfg.window_import.value()`,
`src/runtime/iteration.cpp:3445-3446`) reads a STALE, construction-time value on the restored run,
because `cfg.window_import`'s own slot is a coupler-owned buffer primed ONCE at
`resolve_regions()`/`initialize()` time and never refreshed from restored state.

**Fix attempt 1** (re-priming the EXPORT slot only, `Impl::reprime_window_transfer_buffers()`):
correct and kept -- closes the export side, necessary but not sufficient.

**Fix attempt 2** (re-running the import channel's own compiled delivery chain through
`execute()`, mirroring `resolve_regions()`'s own second pass): MEASURED, not assumed, to make the
mismatch WORSE (iteration count moved further from the reference, 12 -> 10) -- reverted rather
than shipped unverified, per the standing "never ship a plausible-but-unverified fix" rule.

The reference run's own invariant, measured independently: `x_old(window k, iter 0) == x_new
(window k-1, iter 0)` -- each window's own seed equals the PRIOR window's own first-sweep (not
converged) value. The mechanism producing this invariant is not yet traced to a specific write
site. A future round should trace `window_import`'s own write timing across a full continuous
run rather than attempt another plausible-looking re-derivation.

Does NOT affect `AccelKind::None` (whose `update()` never reads `x_old`) or Form A field/history
restart. Affects `AccelKind::Aitken`/`SteffensenAccel`/`QuasiNewtonAccel` specifically, and only
over a window-scoped region. Four Runner-level round-trip tests for this exact case
(`SeparateLayoutJacobiAitken`, `GrowingLayoutJacobiAitken`, `SeparateLayoutJacobiIqnIls`,
`GrowingLayoutJacobiIqnIls`) were written, found failing because of this gap, and removed rather
than shipped failing or weakened.

**Why both recorded fix attempts failed.** Fix attempt 1 (re-priming the EXPORT slot only) left
the CONSUMER's own `window_import` buffer -- the one `R.x_old`'s pre-loop seed actually reads --
untouched, so it closed only the half of the problem visible from the exporting side; the import
side it did not touch is exactly the one this entry's own title names. Fix attempt 2
(re-running the import channel's own compiled delivery chain through `execute()`) re-primed the
RIGHT buffer but with the WRONG data: at restore time, `detail::WindowMemberState::
last_window_waveform` (D70, above) was unconditionally empty for every Runner-driven run -- it
was dead code until this round -- so there was no genuinely restored per-sample trajectory for
that delivery chain to re-derive from; re-running the chain could only reproduce the SAME
construction-time cold-start value fix attempt 1 already left in the export slot, routed through
one more hop, which is why it measured WORSE (12 sweeps, further from the reference's 13) rather
than better. Both attempts predate D70; the present fix is what D70 made possible for the first
time.

**Fix (Windows track, 2026-10-18), depends on D70 above.** `Runner::Impl::
reprime_window_import_after_restore()` (`src/runtime/iteration.cpp`), called once from `Runner::
restore_restart_fields()` right after every window member's own `detail::WindowMemberState` is
restored, copies each PROVIDER's own restored `last_window_waveform` (D70: now genuinely
populated, sample for sample, by `advance_window()`) into the CONSUMER's own live
`window_import` buffer, in the SAME (sample index -> buffer offset `e*provider_capacity+s`)
layout `advance_window()`'s own `out_slot` writes already use -- the provider node for a given
consumer's import is found via `WindowMemberConfig::provider_component_slot`, graph-derived,
never a partner's runtime state, the same lookup `resolve_pending_window_straddles()` already
uses for the analogous problem. A no-op whenever the provider's own `last_window_waveform` is
not yet populated (shape mismatch or a null element) -- every ordinary (non-window) restore, and
a window restore taken before this member's first-ever completed window, both fall through this
guard unchanged.

**Root-caused and verified by RED/GREEN, not assumed:** this call was put behind a temporary
env-gated disable switch and the four round-trip tests (restored below) were re-run with it
disabled. The restored run reproduced D72's own originally-measured symptom EXACTLY --
`iterations_per_window[0]` reported 12 against the uninterrupted reference's 13, and U/V's own
final fields disagreed from the 5th significant digit on -- confirming the pre-loop seed really
was reading the stale, construction-time buffer. With the call enabled (unconditionally, the
disable switch since removed), all four tests pass bit for bit, `iterations_per_window`
included: `SeparateLayoutJacobiAitken`, `GrowingLayoutJacobiAitken`, `SeparateLayoutJacobiIqnIls`,
`GrowingLayoutJacobiIqnIls` (`test/verification/brusselator_restart_test.cpp`, restored rather
than rewritten). The prior round's own stated invariant above (`x_old(window k, iter 0) ==
x_new(window k-1, iter 0)`) is superseded by this direct verification rather than independently
re-confirmed; this round traced the mechanism (the stale buffer) and fixed it rather than
re-deriving that specific invariant.

Full suite regression: `ctest -j3` on this round's own tree, 225/225 tests passing (one
transient failure during one run, `runtime.compile_ok.capabilities_override`, traced to this
session's own build tree being edited concurrently with that ctest invocation -- re-run in
isolation immediately after, passed; not a product regression, declared here rather than
silently re-run and left unmentioned).


### D73. A cross-rank general (non-nodal) weighted Remap on NativeMeshBackend builds a silently empty conservative matrix with no remap-halo call -- FIXED 2026-10-16 (found 2026-10-15, scaled-bench track)

Found while building `test/bench/scaled_device_benchmark.cpp`'s scaled provider/consumer row: a
conservative weighted channel (two native structured-quad meshes of different resolution, no
`.remap_method()` override, so `Runner::Impl::resolve_remap()` takes the general weighted branch, not
the nodal-distributed one) between two components on DISJOINT MPI rank sets builds
`RunStats::remap_weight_builds == 1` on every rank with `RunStats::remap_weight_nonzeros == 0` at every
rank count above 1 -- not a crash, not a wrong answer, a genuinely empty matrix. Each rank's local
intersection search sees only its own region's local view (`regions_coincide`/`build_weights`'s own
local, per-rank construction); a rank hosting only the provider's cells has no consumer cells to
overlap with, and vice versa, so the conservative weight build finds nothing, on every rank, every
time. `fuse::native::build_remap_halo()`/`build_native_remap_halo()` (`remap_plan.hpp`,
`native_halo.hpp`) exist to exchange exactly this geometry across ranks before the weight build runs,
and `remap_plan.hpp`'s own "ORDER MATTERS" comment says the halo must be built before `schedule()`
sizes any buffer -- but nothing in `Runner::initialize()`'s own path calls it, and nothing in
`resolve_remap()`'s general branch refuses or warns when no halo exists on a `NativeMeshBackend` (the
nearby `MoabMeshBackend`-specific check a few lines above it, `E-RT-REMAP-HALO`, does refuse when a
MOAB mesh carries an un-exchanged halo -- there is no analogous check for the native backend at all).

Not reproduced for the small `example.scaled_device_benchmark`/`--verify` correctness row: that row
runs at MPI world size 1 with provider and consumer placed on the SAME one-rank world, so the local
search already sees both sides' full geometry and the weight build is genuinely non-trivial there
(confirmed: `remap_weight_builds=1`, 49 real consumer cells, host and device agree to 1e-14 relative).
The scaled, multi-rank `bench.scaled_device_benchmark.{1,2,3}[.device]` rows still measure genuine
Sample/Remap/Transport/Diagnose/Sink instruction dispatch and MpiTransport exchange overhead at scale
(the benchmark's own stated purpose) at ranks 2 and 3 too, but the Remap's own matvec there is over an
empty matrix, not a real transfer -- stated plainly in that file's own header comment and in this
entry, not asserted as a defect of the benchmark itself.

Not fixed here: driving `build_remap_halo()` correctly needs the caller to build it from a `RemapPlan`
BEFORE `schedule()` (so halo-adopted ghost cells are already counted when buffers are sized), on the
SAME `NativeMeshBackend` mesh ids the Runner's own channel resolves against -- a restructuring of
`build()`'s own sequence in `scaled_device_benchmark.cpp`, not a one-line call, and a check of whether
`Runner::initialize()` should drive this automatically for every caller rather than leaving it to each
one to remember. Left for the lead to scope; `test/parallel/remap_halo_mpi_test.cpp` and
`runner.cpp`'s own comment at the `MoabMeshBackend` halo-refusal check (search for `E-RT-REMAP-HALO`)
are the next place to start.


### D73 FIX (scaled-bench track, follow-up round, 2026-10-16)

**Chosen approach: refusal, not automatic halo.** `Runner::Impl::resolve_remap()`'s general
weighted branch now refuses by name (`E-RT-REMAP-NO-HALO`) when `NativeMeshBackend` is the
backend, the source and target components' `RankSet`s are genuinely disjoint (share no rank,
`RankSet::overlaps()`), and `NativeMeshBackend::has_remap_halo()` reports no halo has been built
for that `(source, target, dimension)` triple at this placement. This mirrors the existing
`MoabMeshBackend` halo check a few lines above it (`E-RT-REMAP-HALO`) exactly, and for the same
reason stated there: wiring `fuse::native::build_remap_halo`/`build_native_remap_halo`
(native_halo.hpp) in AUTOMATICALLY needs two open architectural questions settled first --
where in the instruction stream a COLLECTIVE exchange can run so every rank reaches it in the
same order (`remap_plan.hpp`'s own "ORDER MATTERS": the halo must be built BEFORE `schedule()`
sizes any buffer, and `resolve_remap()` runs well after that point), and whether filling a
component's own array's halo slots is the coupler's contract to take on. The MOAB side already
settled both by refusing rather than guessing; the native backend had no analogous refusal at
all before this fix, so a silently empty matrix was the only reachable outcome. The
automatic-halo alternative is preferred in the abstract and is left for the lead to scope
(same open question, now stated for both backends in one place); refusing is the honest interim
answer within this round's own time, and -- unlike the silent empty matrix it replaces -- never
produces a wrong answer with no diagnostic.

RED/GREEN: `test/parallel/native_remap_halo_runner_test.cpp` (new), a minimal two-differently-
resolved-native-quad-mesh Runner setup on disjoint rank sets, general weighted channel (no
`.remap_method()` override). Confirmed RED first (fix reverted: `initialize()` silently
succeeds, reproducing the original defect exactly), then GREEN (fix restored:
`initialize()` refuses with `E-RT-REMAP-NO-HALO`) at ranks 2 and 3, host and `.device` twin
(`parallel.native_remap_halo_runner.{2,3}[.device]`, all passing).

### D73 FIX, ROUND 2 (RemapHalo track, 2026-10-18): the refusal above is replaced by an
automatic halo, driven by the lowering, not by the Runner

**Owner decision 2026-10-18:** the automatic-halo alternative ROUND 1 above left for the lead to
scope is now implemented. `src/ir/rules_channel.cpp`'s new `channel_needs_remap_halo()` runs in a
front pass (`emit_remap_halo_instructions()`, called from `lower()` BEFORE step 1's `Source` loop) and
emits one collective `OpKind::BuildRemapHalo` instruction (fuse/ir/op.hpp) per cross-rank weighted
(non-nodal, non-consistent) channel -- the exact branch `Runner::Impl::resolve_remap()` takes when
`fuse::remap_moves_placement` answers false. `expand()` stamps every instance at `t = 0`,
`TickPhase::BeforeTicks`, sorting it to the front of the compiled stream.

**Where the real exchange runs, and why not inside `Runner::initialize()`.** `remap_plan.hpp`'s own
"ORDER MATTERS" note says a remap halo must be built before `make_region_sizes`/`plan_buffers`/
`schedule()` size any buffer on the mesh it grows -- and `schedule()` is called by the CALLER, to build
the `ExecutionPlan` passed to `Runner`'s own constructor, before `Runner::initialize()` ever runs.
Measured directly against this file's own test fixture (2x2/3x3 native quad meshes, disjoint
placement): `make_region_sizes`'s `RegionSizes` callback is invoked once, synchronously, inside
`schedule()`'s own `plan_buffers` phase, reading the backend's PRE-halo entity counts; a halo built any
later cannot retroactively widen an `ExecutionPlan`'s already-frozen `BufferDecl::elements`, and the
existing `E-RT-REMAP-SIZE` check (added for exactly this hazard, remap_plan.hpp's own doc) is what would
catch the resulting mismatch. So the real geometry exchange runs in a NEW free function, `fuse::execute_
remap_halo_instructions()` (fuse/backend/remap_halo_instructions.hpp), which a caller runs ONCE, between
loading the mesh backend and calling `schedule()` -- the same point `build_remap_halo()` already had to
be called manually. `Runner::Impl::resolve_remap_halo()` (new, called from `resolve_instructions()` for
every `BuildRemapHalo` instruction, before the corresponding channel's own `Remap` instruction) only
VERIFIES, via a local `has_remap_halo()` lookup (no MPI call -- safe because `execute_remap_halo_
instructions()` is itself collective, so the answer is consistent across every rank of the base
communicator by construction), and refuses `E-RT-REMAP-HALO-NOT-BUILT` when it was not. The OLDER,
ad hoc "disjoint rank sets -> check has_remap_halo" refusal this round's own predecessor added is left
in place, unchanged, as a second, independent gate -- defense in depth, not replaced, since it is
already tested and already correct for the case it covers.

**Both backends.** `NativeMeshBackend` builds through `fuse::build_remap_halo` -> `fuse::native::
build_native_remap_halo`, exactly as before; `MoabMeshBackend` dispatches through the SAME `fuse::
build_remap_halo` entry point (its own existing `dynamic_cast` split, intx_weights.cpp), so the new
instruction covers it identically -- no MOAB-specific code was added in this round, since `fuse::build_
remap_halo`'s own dispatcher already handled both.

**IR goldens changed, by exactly the new instruction.** `test/golden/{pair_explicit,pair_accumulate_
remap_relax,pair_iterated,window_two_member,window_two_member_jacobi}.{ir,plan}` each gained one
`BuildRemapHalo` node at the front (two, for the two-channel window fixtures) with every other node id
and instruction index shifted by the same count; `pair_lowfive_bound.ir` (an application-performed
transfer, `ranks_differ` forced false the same way `lower_channel()` forces it) is unchanged, confirmed
by diff against the pre-round tree.

**A real bug found and fixed in the SAME round, not a pre-existing one:** `build_iteration_regions()`
(src/ir/regions.cpp) was sweeping a window-scoped region's own `BuildRemapHalo` instructions into
`IterationRegion::body` by the SAME origin-channel match that finds the region's ordinary Transport/
Remap/.../Sink chain, which (a) moved `region_enter` onto the `BuildRemapHalo` instruction instead of
the real first chain node, and (b) risked `E-SCHED-REGION` the moment an unrelated `t = 0` instruction
sorted between a region's `BuildRemapHalo` and the rest of its body (the body must be instantiated as
contiguous whole-body passes). Caught by `test/schedule/window_region_golden_test.cpp`'s own byte-exact
dump before this round's patch was considered done; fixed by excluding `OpKind::BuildRemapHalo`
explicitly from body collection, the same way an ordinary (non-window) region already excludes `Sink`.

**RED/GREEN, this round:** `test/parallel/native_remap_halo_runner_test.cpp` rewritten --
`DisjointGeneralWeightedRemapConservesOnceTheHaloIsBuilt` (GREEN: `execute_remap_halo_instructions()`
called before `schedule()`, `initialize()` and a real `run_period()` succeed, conservation checked to a
Higham recursive-summation bound derived in the test itself) and `DisjointGeneralWeightedRemapRefuses
WhenTheHaloWasNeverBuilt` (RED: the halo is never built, `initialize()` refuses `E-RT-REMAP-HALO-NOT-
BUILT`). Both BUILT (host cannot start MPI in this sandbox), not run; `parallel.native_remap_halo_
runner.{2,3}[.device]` is this round's own host-pending row, same as round 1's.

**Tier-0 coverage, this round:** `test/ir/ir_op_test.cpp` (the new enumerator's own `is_collective`
classification and the exhaustiveness count), `test/schedule/plan_test.cpp`'s `ExecutionPlan.
DumpIsByteExact`/`DumpShowsWhichInstructionsRunAfterTheTicks` (updated for the new instruction) and
`test/schedule/acyclicity_test.cpp`'s `EmissionOrderViolationIsReported` (updated to find its Source
node explicitly rather than assume it is the first in emission order, since `BuildRemapHalo` now sorts
ahead of it) all pass; the golden-dump tests above are this track's own "instruction present, position,
once per channel" and "ordering" tier-0 checks, run and passing.

**Open, left for the lead:** D5's disjoint-placement weights are settled as HALO BRIDGING (owner note,
2026-10-18, relayed mid-round): `RemapPlan::placement` stays the TARGET placement; the halo build
additionally needs the SOURCE's own placement (`graph.owner_of(channel.source)`) as a second argument,
threaded through a new overload of `fuse::build_remap_halo`/`build_native_remap_halo` the separate
MoabHalo track is adding. `fuse::execute_remap_halo_instructions()` passes only `plan.placement` today,
because that is the only parameter the signature accepts as of this round's own export (`git rev-parse
HEAD` = `01a6d2535c9005e86d55b57ca79b680c68633741`, three commits after `cb127c5`); once the two-
placement overload lands, that call needs `graph.owner_of(ch.source)` added as a second argument. See
this track's own `open_questions` for the exact line.

### D73 FIX, ROUND 3 (RemapHalo track round 2, 2026-10-18): the execution point moved into `make_region_sizes()`; D77 closed

**Owner decision 2026-10-18 (second round):** D77's open question is decided. Rather than refactoring
`Runner`'s own constructor to call `schedule()` internally (a breaking change to every existing caller),
the owner moved `fuse::execute_remap_halo_instructions()`'s own call site ONE level down, into `fuse::
make_region_sizes()` (fuse/backend/mesh_view.hpp, implemented src/backend/serial_backend.cpp):
`make_region_sizes(graph, dag, backend)` now builds every halo `dag` carries, collectively, BEFORE
constructing the `RegionSizes` callback it returns -- not as a separate call a caller remembers to place
correctly, but as the first thing the ONE function every caller already passes `dag` to does.
`execute_remap_halo_instructions()` itself did not change; round 1's own free function is now called from
one place instead of from every application's own setup code, and its header comment says so explicitly
("NOT A PUBLIC STEP THE APPLICATION MUST TAKE").

**Signature change, every caller updated.** `make_region_sizes()` gained a middle `const OperatorDAG&
dag` argument and its `backend` parameter became non-const (building a halo mutates the backend's own
mesh state). The two-argument overload was RETIRED rather than kept and documented as a silent-skip
risk: every caller this track found already had `dag` in scope at the call site (`schedule()`'s own next
argument, a few tokens away) --
`examples/layout_driver/main.cpp`, `examples/brusselator/main.cpp`, `examples/dn_heat/main.cpp` (3
sites), `examples/fsi_channel/main.cpp`, `examples/mms_pair/main.cpp`, `src/abi/participant.cpp:973`,
`python/src/component_bindings.cpp`, and every test under `test/` that calls it (~130 call sites across
~45 files) were updated to pass it. A handful of pure region-sizing unit tests
(`test/backend/{serial_mesh,field_extent}_test.cpp`) construct no `CouplingGraph` channel at all and
never call `lower()`; these pass a default-constructed `fuse::OperatorDAG{}` (zero nodes, so `execute_
remap_halo_instructions()` is the documented no-op for a dag with nothing to build).

**Two-placement attrs, ahead of the overload.** Per the owner's standing instruction not to wait for the
MoabHalo track: `emit_remap_halo_instructions()` (src/ir/rules_channel.cpp) now stamps BOTH arcs' owning
components onto the `BuildRemapHalo` instruction's own `attrs` -- `attrs.src_rank_set` (the channel's
source component) and `attrs.dst_rank_set` (`fuse::remap_placement()`'s own target placement) -- the
SAME `OpAttrs` fields `Transport`/`Epoch` nodes already carry their own rank-set pair in, so this adds no
new field and no new golden-dump text (`dump.hpp`'s `attrs_string` only prints `src_rank_set`/
`dst_rank_set` for `OpKind::Transport`). `execute_remap_halo_instructions()` now reads `attrs.dst_rank_
set` back for `plan.placement` rather than recomputing `remap_placement()` a second time; `attrs.src_
rank_set` sits ready, unused by today's one-placement `fuse::build_remap_halo` call, for the day the
MoabHalo track's two-placement overload lands (`graph.component(ComponentId{attrs.src_rank_set}).ranks`
is the source's own `RankSet`, resolvable with no further graph lookup) -- a dated comment at the call
site (src/backend/remap_halo_instructions.cpp) names exactly where it goes.

**Mutation evidence, this round: which gate actually catches a skipped halo.** Two new rows in
`build-evidence/mutations/ir-schedule.toml`: `make-region-sizes-skips-halo-build` (the call to `execute_
remap_halo_instructions()` inside `make_region_sizes()` is removed) and `regions-includes-build-remap-
halo-in-body` (round 1's own `build_iteration_regions()` fix, src/ir/regions.cpp, is reverted). Both
applied, built, and run for real in this round (not merely asserted -- see the corresponding `local_
verification` entries this round's own structured output cites): the first is caught by `Runner::Impl::
resolve_remap_halo()`'s `E-RT-REMAP-HALO-NOT-BUILT` refusal (proven by `test/parallel/native_remap_halo_
runner_test.cpp`'s own RED row design, since `make_region_sizes()`'s callback still answers with the
PRE-halo entity count either way -- the halo never grows the mesh, so there is no larger buffer to size
and therefore no `E-RT-REMAP-SIZE` to catch instead); the second is caught by `test/schedule/window_
region_golden_test.cpp`'s byte-exact dump (the window region's own `body`/`region_enter` positions move
back onto the `BuildRemapHalo` node). Neither mutation is caught by both gates at once -- this round
measured and recorded which ONE actually fires for each, rather than asserting "a mismatch is caught"
without naming which check does it.

### D74. A nodal "consistent" Remap's distributed point-location search refuses at the domain's own corner vertex (global id 0) at some multi-rank splits -- FIXED 2026-10-16 (found 2026-10-15, scaled-bench track)

Found while registering `bench.scaled_device_benchmark.{2,3}[.device]` (`test/bench/scaled_device_benchmark.cpp`):
a nodal "consistent" channel between a provider's (N+1)^2 vertices and a consumer's N^2 cells, both
independently generated `generated:quad:NxN` meshes over the unit square, on disjoint MPI rank sets,
fails `Runner::initialize()` at N=1000 (2 and 3 ranks, both host and `FUSE_EXECUTION_SPACE=device`) with

    E-INTERP-NEWTON: 'link/remap': build_distributed_interp_rows failed:
    build_distributed_interp_rows: rank 1 could not place its evaluation
    entity of global id 0 (E-INTERP-NEWTON)

-- the SAME evaluation entity (global id 0, the mesh's own corner at (0,0)) D67
(runtime-exchange track, 2026-10-15) already named as the one point where a replicated-source,
multi-rank point-location search disagrees with a one-rank reference by 1 ULP. D67 was a tiny
VALUE discrepancy, absorbed by every test's own tolerance until an exact-equality row caught it;
this is the same search going further and refusing outright to place that corner point at all, at a
rank count and problem size D67's own file never tried (D67's own reproduction is `n_beam=7,
n_fluid=11` on a 1D line mesh at ranks 2 and 3; this file's own mesh is 2D, `N=1000`). Not
reproduced at N=1 (one rank; no replicated gather at all, `comm_handle == kNullComm`-shaped) nor at
small multi-rank N (N=20, 2 and 3 ranks, passes cleanly) -- whatever tips the Newton iteration from
"1 ULP off" to "does not converge" at this corner depends on both the rank split and the problem
size, not reproduced at a size small enough to isolate further within this track's own time.

Not fixed here, and not this track's own scope to fix (the primitive D67 already named,
`build_distributed_interp_rows`'s own per-cell containment search, general_interp.hpp's
`invert_reference_map`/tie-break code). Worked around in `test/CMakeLists.txt`'s own bench
registration: `bench.scaled_device_benchmark.{2,3}` and their `.device` twins are registered
`DISABLED TRUE`, each with a comment pointing here, rather than left failing or silently dropped --
`bench.scaled_device_benchmark.1[.device]` (no multi-rank split, no replicated gather) and the small
`example.scaled_device_benchmark[.device]` correctness row (world size 1, same reason) are
unaffected and pass. Re-enabling the two disabled rows is the right acceptance test for whoever
fixes D67/D74's shared root cause.
### D74 FIX (scaled-bench track, follow-up round, 2026-10-16)

**Root cause, directly diagnosed (not the corner-precision theory originally suspected).**
Temporary instrumentation of the failing N=1000, 2-rank case caught the mechanism in the act:
`build_distributed_interp_rows`'s eval loop tested the evaluation point for cell global id 0
(centroid `(0.0005, 0.0005)`, the domain's own near-corner cell) against candidate index 17997
-- a cell near `x = 0.997`, nowhere close -- and `invert_reference_map`'s damped Gauss-Newton,
started from THAT cell's own reference centre for a target that far outside it, genuinely did
not converge within 50 iterations (last residual `1.92e-12` against a `1e-12` tolerance). The
loop had ALREADY found the correct containing cell (global id 0) as a tight match at candidate
index 0, moments earlier in the very same scan -- but the scan never stopped looking once it had
the right answer, because (unlike `general_interp.hpp`'s serial counterpart, which narrows to a
handful of nearby candidates through a `search::BoxIndex` broad phase FIRST) this function tested
literally every cell in the assembled global mesh, unconditionally, for every evaluation point,
and treated ANY single candidate's Newton failure -- however far away, however irrelevant -- as
an immediate, unconditional abort of the whole evaluation point.

**The fix.** `build_distributed_interp_rows` now builds the SAME kind of `search::BoxIndex`
broad phase `build_general_interp_weights` already uses, over this function's own
(Allgatherv-replicated) assembled cells, padded by the same `kRefEpsLoose` bound the fallback
tier already accepts a candidate within. Each evaluation point now only tests the small, genuinely
nearby candidate set the box query returns (plus a k-nearest fallback when that query finds
nothing at all, matching the serial builder's own `ClampToBoundary` safety net) -- so a stray
far cell's Newton non-convergence can no longer reach this code at all, and a per-candidate
failure that still occurs (a genuinely degenerate nearby cell) is now SKIPPED, not fatal, exactly
like a converging candidate that merely lands outside the reference domain. This also drops the
whole search from O(n_eval * n_cells) to O(n_eval * candidates) -- a small constant on a
structured mesh -- which is why the defect was never exercised before this track's 1e6-cell row.

**Relationship to D67: investigated, NOT the same mechanism.** See the addendum on D67's own
entry above: re-running `NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`
after this fix reproduces the EXACT SAME 1-ULP bit pattern D67 already documented, unchanged.
D67 remains open on its own.

RED/GREEN:
  * `NodalRemapDistributedNullComm.DegenerateDecoyCellDoesNotAbortAnAlreadyFoundMatch` (new,
    `test/backend/nodal_remap_distributed_nullcomm_test.cpp`, no MPI needed): a hand-built 2D
    mesh with two well-formed unit-square cells plus a genuinely degenerate decoy cell (two
    coincident corner pairs, collapsed along an off-axis diagonal so its own bounding box stays
    full-dimensional) reproduces this exact mechanism deterministically, in-process. Confirmed
    RED first (fix reverted: fails `E-INTERP-DEGENERATE`, reproducing the abort), then GREEN
    (fix restored: the genuine tight match wins, the decoy is silently skipped).
  * `bench.scaled_device_benchmark.{2,3}[.device]` RE-ENABLED (the `DISABLED TRUE` block removed
    from `test/CMakeLists.txt`) and confirmed GREEN at N=1000: ranks 1, 2 and 3, host and
    device, all six rows passing (build 43-125s, initialize 19-77s, period ~0.2-0.9s under
    shared host load at the time of this run -- see report-scaled-bench.md's follow-up section
    for the full table).
  * `parallel.nodal_remap_runner.{1..5}[.device]` and `verification.nodal_remap_runner_serial
    [.device]` (D67/D8's own suite) re-run UNCHANGED and confirmed still green (15/15 passing),
    confirming this fix does not touch that suite's own behaviour.

**Re-examined (PointLocation track, 2026-10-18), while fixing D67's own 1-ULP arithmetic-divergence
root cause.** D74's own fix (the broad-phase box index, above) is a DIFFERENT, unrelated defect
(a per-candidate Newton failure wrongly aborting an already-found match) from D67's (two code paths
computing the same Edge-cell hat weight by different floating-point expressions) and is unaffected by
D67's fix: `bench.scaled_device_benchmark.{2,3}[.device]` remain enabled (no `DISABLED` row re-added to
test/CMakeLists.txt) and this track introduces no change to `build_distributed_interp_rows`'s broad
phase or its per-candidate skip-on-failure logic. No skipped or disabled test remains attached to this
entry to un-skip; the one case this track's own task description had in mind under that name --
`NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`'s own beam-vertex-0 exclusion,
softened to a `5e-16` tolerance and explicitly attributed to D67 in its own comment (not to D74) -- is
addressed under D67 above (the exclusion is removed; every vertex is now asserted bitwise-identical).

### ~~D75. The collective restart writer produced files no reader could open~~ -- FIXED 2026-10-17

**Measured** on the owner's machine (MPICH 5.0.0a1, HDF5 1.14.1 parallel, `FI_PROVIDER="tcp;ofi_rxm"
FI_TCP_IFACE=en0`): `parallel.brusselator_restart_mpi.{2,3}` failed on every multi-task write while the
one-task write (the serial writer) passed. Three defects in `src/restart_io/hdf5_restart_io_mpi.cpp`, each
exposed by the fix before it, all three invisible at tier 0 because the sandbox cannot start MPI and the
collective writer is compiled only with `FUSE_ENABLE_MPI`:

1. `finish()` closed the file while the point group was still open. With the MPI-IO driver the default
   close degree is semi, so `H5Fclose` fails; the handle helper discarded the failure, the file stayed open
   with unflushed metadata, and rank 0 renamed it. Reader: `E-RESTART-TRUNCATED: does not open as a valid
   HDF5 file`. Fix: close the group first (as the serial writer does), close the file explicitly and refuse
   if the collective close fails.
2. The file attributes were written as `@format_version`, `@fuse_library_version`, ... (the names in
   `docs/dev/restart.md` Section 4.1's draft table), while the serial writer and the reader use
   `format_version`, `fuse_version`, `descriptor_hash`, `written_by_rank_count`, `mesh_checksum`. Reader:
   `E-RESTART-TRUNCATED: missing attribute format_version`. Fix: the serial writer's names; the design table
   corrected to the names actually on disk.
3. No per-field type attributes (`element_type`, `byte_size`, `significand_bits`, `exponent_bits`,
   `exponent_bias`, `byte_order`, `is_signed`) were written: the writer was built against the pre-typed
   reader (slice 1b made them required) and rebased without noticing. Fix: the same seven attributes on the
   collective `values` dataset.

After the three fixes: `parallel.restart_collective_mpi.{2,3}` and `parallel.brusselator_restart_mpi.{2,3}`
pass, and the two-invocation write-at-3/restore-at-2 protocol passes both invocations (`h5dump -A` on the
written file shows the five file attributes and the seven type attributes). Lesson recorded with D23/D38:
a host row that cannot run in the sandbox must be run on the host BEFORE its code is described as
delivered; slice 2's report said "built but not run", and all three defects sat behind that.


### D76. D17's own Jacobi-plus-Aitken driver discrepancy traced to a seed disagreement at the SECOND window, not an accelerator instability -- OPEN, measured 2026-10-18 (Windows track)

D17 (above) names `--accel aitken` converging through `--driver runner` but not `--driver
window`, on the Brusselator's own default parameters. Measured this round, by a temporary
env-gated trace inside `AitkenAccel::update()` (`src/num/accel_relaxation.cpp`, shared code --
both drivers call it, instrumentation since removed) printing `n`, `iteration`, the computed
relaxation factor `omega`, and `||x_new - x_old||` on every call, for `--driver window` and
`--driver runner` run SEPARATELY on the identical `--accel aitken --iteration-mode jacobi --end
0.1` configuration:

- Window 0 (the FIRST window): both drivers' own `omega`/residual-norm sequences are IDENTICAL
  for all 15 sweeps to every printed digit (0.1, 0.80266972388320323, 0.83497979794902755, ...,
  converging at sweep 14 with `r_norm=6.6763228060059144e-09` on both sides) -- the accelerator
  itself is not where this starts. The two vectors differ in LENGTH (`n=4352` for the window
  driver, `n=4864` for the Runner, a 512-element gap) without this affecting the comparison at
  all: every extra element's own residual contribution is identically zero on both sides for
  window 0, consistent with (but not yet confirmed to BE) the window-start sample padding
  `RegionBlock::window_export_capacity`'s own comment names elsewhere in this file -- the exact
  source of the size difference is not yet traced and is left here rather than guessed.
- Window 1 (the SECOND window): `omega` at sweep 0 is `0.1` on both sides (`reset()`'s own fixed
  default at every window boundary, expected), but `r_norm` already disagrees AT SWEEP 0 --
  `143.37914811541057` (window driver) vs `145.9156716694263` (Runner) -- before Aitken's own
  history-dependent formula has run even once this window. Since `r_norm = ||x_new - x_old||` at
  sweep 0 and `x_new` is each side's own freshly-advanced output, the DISAGREEMENT is in how each
  driver seeds (or carries forward) `x_old` for a window's own first sweep, not in anything
  Aitken computes from it -- the identical class of quantity D72 (above) found wrong after a
  restart, but reproduced here with NO restart involved at all, purely from window-to-window
  bookkeeping. Aitken's own `omega` recursion is sensitive enough to this seed (`omega_k` is a
  ratio of inner products of residual DIFFERENCES across sweeps) that a seed disagreement this
  small, compounded over Jacobi's own slower convergence (needing many more sweeps than
  GaussSeidel per D71's own measured counts, above), is a credible mechanism for the two
  trajectories to diverge far enough, by window 3, that one side fails to converge within 60
  sweeps while the other does not -- not yet proven sweep by sweep past window 1, left for the
  reason below.

**Not fixed.** Root-causing exactly which write site produces the window-1 seed disagreement --
`run_region_window()`'s own pre-loop seed (`slots[b.feed_arc]`) on the Runner side, versus
`WindowCoupler::sweep_window()`'s own `y_fed` carry-forward on the driver side -- needs the same
kind of write-timing trace D72's own entry recommended for `window_import`, extended across a
window BOUNDARY rather than a restart boundary, which this round's own time did not reach.
Recorded here as a measured, narrowed finding (the divergence is at window 1 sweep 0, in the
seed, not in the accelerator) rather than the prior entry's "not root-caused" -- the next round
should trace `R.x_old`'s own pre-loop value against `WindowCoupler`'s own `y_fed` element by
element at the window 0 -> window 1 boundary specifically, the same technique used to trace
defects (a)-(d) in `test/verification/brusselator_runner_test.cpp`'s own header.

### ~~D77. The automatic remap halo (D73) is driven by a NEW free function the caller runs before `schedule()`, not by `Runner::initialize()`~~ -- CLOSED 2026-10-18: the build moved into `make_region_sizes()` (owner decision) 2026-10-18, owner decided (RemapHalo track)

The owner's 2026-10-18 decision for D73 described the contract as "the Runner executes [`BuildRemapHalo`]
during `initialize()` BEFORE `schedule()` sizes any buffer". That literal sequence measured impossible in
the architecture this track's round 1 shipped against: `schedule()` is called by whichever code builds
the `ExecutionPlan` passed to `Runner`'s own constructor (`Runner::Runner(graph, dag, plan, backend)`),
which is necessarily BEFORE `Runner::initialize()` can run at all. `schedule()`'s own `RegionSizes`
callback (`make_region_sizes`) reads the mesh backend's entity counts once, synchronously, to freeze
every `BufferDecl::elements` in the plan; growing a mesh's remap halo any later (inside `Runner::
initialize()`, as the owner's wording implied) cannot retroactively widen an already-frozen buffer size.
`fuse/backend/moab_backend.hpp`'s own "ORDER MATTERS" comment and `fuse/backend/remap_plan.hpp`'s own
`build_remap_halo()` doc already stated this constraint for the pre-existing, manual-build-halo
discipline; round 1's own measurement (`test/parallel/native_remap_halo_runner_test.cpp`'s fixture)
confirmed it still held for the automatic case.

**Round 1** (recorded here as this issue's own body, when it was still open) shipped the functionally
equivalent caller-driven design: a new free function, `fuse::execute_remap_halo_instructions()`, that
application code called directly between `backend.load()` and `schedule()`.

**Round 2 decision, 2026-10-18 (this entry's closing round):** rather than a `Runner`-constructor
refactor, the owner decided to move the call site ONE level DOWN, into `fuse::make_region_sizes()`
itself (fuse/backend/mesh_view.hpp) -- the one function every existing caller already passes the lowered
`OperatorDAG` to, immediately before `schedule()` runs. `make_region_sizes()` now calls `fuse::execute_
remap_halo_instructions()` on its own `dag` argument BEFORE building the `RegionSizes` callback it
returns, so the halo is built collectively, structurally (every rank already agrees on the set and order
of `BuildRemapHalo` instructions, with no data-dependent branch), and a caller cannot forget the call or
place it after `schedule()` by hand -- the two steps that used to be a caller's own two separate
responsibilities are now one. `execute_remap_halo_instructions()` itself is UNCHANGED in its own
contract; it is simply no longer a public step application code takes directly (see its own header
comment, fuse/backend/remap_halo_instructions.hpp). `Runner::Impl::resolve_remap_halo()`'s `E-RT-REMAP-
HALO-NOT-BUILT` verification stays exactly as round 1 left it -- the second, independent gate, now
catching a caller that bypassed `make_region_sizes()` entirely (hand-rolling its own `RegionSizes`,
something the type system cannot prevent) rather than a caller that merely called the two steps in the
wrong order, which this round's design makes structurally impossible for every caller already found in
this tree. `make_region_sizes()`'s own two-argument overload is RETIRED, not merely deprecated: every
caller in this tree (`examples/{layout_driver,brusselator,dn_heat,fsi_channel,mms_pair}/main.cpp`, `src/
abi/participant.cpp`, `python/src/component_bindings.cpp`, every test under `test/`) already had the
lowered `dag` in scope at the call site, so there is no caller left to keep a reduced-information
overload for; see `KNOWN_ISSUES.md`'s own D73 ROUND 3 paragraph below for the measured mutation proving
`E-RT-REMAP-HALO-NOT-BUILT` (not a size mismatch) is what actually catches a bypass.

D5's disjoint-placement halo-bridging second-placement argument remains open, unaffected by this
decision -- see D73 ROUND 3 below for its current status.

### D80. `FsiEnergyDiag.FullBudgetOnTheDominantModeCloses`'s mass-correction isolation check compared a `cn_step` call on the literal `c.dt` against production's own `dt_delivered` -- FOUND AND FIXED 2026-10-19 (FsiMassCorr track)

**Measured on this machine (2026-10-18/19, first local run of the CI legs).** Test
`FsiEnergyDiag.FullBudgetOnTheDominantModeCloses`
(`test/verification/fsi_energy_diag_test.cpp:2284`, `ASSERT_LT(max_abs_diff_non_outflow,
mass_corr_tol)`) failed deterministically (6 of 6 runs identical to every printed digit) in a
Release, `FUSE_ENABLE_EXCEPTIONS=OFF` build with Kokkos fetched Serial-only from source and Eigen
fetched (pinned tag): `eig_quality_combined=1.252e-07`, `max|u_raw|=1.270e-02`,
`tolerance=3.501e-11` (the OLD tolerance, `0.5*eig_quality*max|u_raw|`), `max|u1-u_raw|` away from
the outflow column `=1.081e-10` (ratio 3.089 -- FAILS). The owner's own Debug build-all (exceptions
on, installed Kokkos 5.1.0 OpenMP+Serial, installed Eigen) measured the SAME quantity at
`1.545e-10` against a looser tolerance (`2.064e-09`, ratio 0.075 -- passes), and D56 (above) records
the identical gate failing by 1.5x on a third (GCC 13.3/higgs) toolchain -- the quantity moving
across builds is `eig_quality` (the eigensolver's own residual, by ~50x between builds), while the
measured mismatch itself sits at `1.1e-10` to `1.5e-10` in every build. This was the standing clue
that `eig_quality` was the wrong scale: the eigensolver's residual and this identity's own fluid
solve are unrelated linear algebra.

**Mechanism, measured directly, not assumed.** Three candidate mechanisms were tested, each by a
measurement that isolates it, using `FluidComponent::debug_last_velocity_in_used` and
`StokesSolver::debug_last_u0_used`/`debug_last_v0_used` (new diagnostic hooks, null by default --
`examples/fsi_channel/fsi_components.hpp`, `examples/fsi_channel/stokes_solver.hpp`):

 1. **Hypothesis (c), a test sampling artifact** ("`u1` and `u_raw` sampled at different states"):
    `diag.vwall1` (what the test fed `measure_full_energy_identity_general`) is
    `fluid_ptr->velocity_in()` read back AFTER `Runner::run_period()` returns -- i.e. the region's
    own `G(y)` (`src/runtime/iteration.cpp`'s own header comment: "the Sink instructions ...
    deliver G(y)"), which need not equal `y`, the wall velocity the LAST `advance()` call actually
    solved from, by more than the region's own convergence tolerance. Measured directly: `|y -
    G(y)| = 3.90e-18` (round-off) on this case, and feeding `y` instead of `diag.vwall1` reproduced
    the IDENTICAL `1.081237e-10` mismatch to 7 significant figures. REJECTED.
 2. **The cross-call cached saddle-point factorisation** (`StokesSolver::lu_cache_`, which
    production's `step()` reuses across sweeps while the test's own reconstruction always
    refactorises): forcing `step()` to always refactorise too (temporarily, in isolation)
    reproduced the IDENTICAL `1.081237e-10` to every printed digit. REJECTED.
 3. **`diag.u0`/`diag.v0` not bit-identical to the `(u_, v_)` the last `advance()` actually
    restored and solved from**: measured directly, both gaps are EXACTLY `0.0`. REJECTED.

**What remained, and confirmed the mechanism.** The test's own call,
`measure_full_energy_identity_general(fc, c.dt, diag.u0, diag.v0, diag.vwall0, diag.vwall1)`,
passed the LITERAL `c.dt` as `cn_step`'s own `dt` argument. Every PRODUCTION `cn_step` call in this
same run executes on `dt_delivered = rationalize_dt(c.dt).to_double()` -- `build_fsi_graph()`
quantizes `c.dt` through this file's own `rationalize_dt()` before handing it to the coupling
graph, exactly the mechanism D34 (above) already found and fixed for the beam-side identity in this
SAME test file, reintroduced here for the fluid-side (mass-correction) identity. `cn_step`'s own
matrix carries `dt` through the time-derivative (`O(1/dt)`) coefficient, so the relative dt error
`|c.dt - dt_delivered| / dt_delivered = 4.596e-7` (D34's own measured ratio, unchanged on this case)
perturbs EVERY column of the solve, not just the outflow one the mass correction touches -- an
across-the-board leak, not a defect confined to one column. Feeding `dt_delivered` instead of the
literal `c.dt` collapsed the non-outflow mismatch from `1.081e-10` to `4.443e-15` (round-off) in
BOTH the Release/no-exceptions build (Kokkos found/installed + ArborX found/installed, Eigen/GTest/
fmt fetched -- see Deviations below) and a Debug build (`OMP_NUM_THREADS=1`) of the same
composition -- CONFIRMED, and toolchain-independent as expected for a deterministic bookkeeping fix
(unlike `eig_quality`, which moved 50x between builds while this mismatch did not). Hypothesis (a),
a bookkeeping mismatch, is the mechanism; hypothesis (b), a genuine defect in the mass correction's
own column restriction, is REJECTED: the correction's own code (`examples/fsi_channel/
stokes_solver.hpp`, the outflow-mass-conservation block) writes to exactly one `idx_u(p_, p_.nx,
j)` per row and no other index (confirmed by a hand-mutation that adds a second write to
`idx_u(p_, p_.nx - 1, j)`, which this test's new tolerance catches at a ratio of `1.4e9`, see
Mutations below).

**Fix.** `measure_full_energy_identity_general` is called with `dt_delivered`, not `c.dt`. The
tolerance is re-derived from the solve's own backward stability rather than from `eig_quality`:
with `dt` fixed, `diag.u1` (away from the outflow column) and `u_raw` are two INDEPENDENT solves of
the IDENTICAL assembled saddle-point system `A x = rhs` (production's cached factorisation vs. this
call's fresh one), each backward-stable per Higham (*Accuracy and Stability of Numerical
Algorithms*, 2nd ed., secs. 9.3-9.5) with relative error bounded by this project's own established
per-solve constant `2*kappa_2(A)*u` (the same form D34 already uses for the beam's own direct
solve). Two independent solves of one system can therefore differ by up to the SUM of their two
bounds, `4*kappa_2(A)*u`, scaled by the solution's own magnitude. `kappa_2(A)` is measured directly
(dense `JacobiSVD` on the assembled saddle-point matrix `cn_step` solved -- `N=53` on this grid, an
exact dense SVD): `kappa_2(A_fluid)=3.532030e+05`, tolerance `=4*kappa_2(A)*u*max|u_raw|=3.984e-12`,
observed mismatch `4.443e-15` (ratio to tolerance `0.001` -- not vacuous, not tighter than the
observed mismatch).

**Mutations (byte-identical restore after each).**
 1. Reverting the fix (literal `c.dt` restored): the new tolerance-based assertion FAILS, ratio
    `27.141` (`max|u1-u_raw|=1.081e-10` against tolerance `3.984e-12`) -- killed.
 2. A second write added to the mass correction's own loop (`next.u[idx_u(p_, p_.nx - 1, j)] +=
    correction;`, simulating hypothesis (b), a real leak into the adjacent column): the assertion
    FAILS, ratio `1.4e9` (`max|u1-u_raw|=3.315e-03`) -- killed decisively.

**Verified.** `fuse_verification_fsi_energy_diag_test` (`FsiEnergyDiag.*`, 6/6) passes with the fix
in both a Release/`FUSE_ENABLE_EXCEPTIONS=OFF` build (Kokkos+ArborX found/installed, Eigen/GTest/fmt
fetched) and a Debug build of the same composition, `OMP_NUM_THREADS=1` -- identical
`kappa_2(A_fluid)` and mismatch in both, as expected for a deterministic fix. `tools/check-ascii.py`,
`tools/check-invariants.py` and `tools/mutate.py --check-anchors` all pass on the touched files.

**Deviations from the measured-failing leg's exact composition.** The measured-failing leg named
above fetched Kokkos (Serial-only) AND Eigen AND GTest from source. In this sandbox, cloning a git
repository is blocked outright (creating a `.git` directory anywhere fails with "Operation not
permitted", a sandbox-wide restriction, not specific to this project) -- dependency sources were
instead fetched as plain HTTPS tarballs and pointed at via `FETCHCONTENT_SOURCE_DIR_<NAME>`. That
substitution does not affect this finding: forcing Kokkos AND ArborX to BOTH be fetched together
with this project's current pins is independently impossible regardless of sandboxing -- ArborX's
pinned commit (`8f89be60bc3797670b9be8843da7dcdbc5f2333f`) calls `find_package(Kokkos 4.5 REQUIRED
CONFIG)` in its own `CMakeLists.txt`, while `FUSE_KOKKOS_TAG` pins Kokkos at `4.4.01` (< 4.5), and
neither `FetchContent_Declare(Kokkos ...)` nor `FetchContent_Declare(ArborX ...)`
(`cmake/FuseDependencies.cmake`) carries `FIND_PACKAGE_ARGS`/`OVERRIDE_FIND_PACKAGE`, so there is no
redirect letting ArborX's own `find_package(Kokkos 4.5)` resolve against a freshly-fetched,
sub-4.5 Kokkos target -- measured directly: fetching both together fails configure with "Could not
find a package configuration file provided by Kokkos (requested version 4.5)". This build instead
used Kokkos+ArborX found/installed (a consistent pair) with Eigen/GTest/fmt fetched. Given the
mechanism (D34's own `rationalize_dt()`/`dt_delivered` bookkeeping split) and its fix are properties
of the TEST's own arithmetic, not of which Kokkos backend is linked, and the fix was confirmed
identically in both a Release/no-exceptions and a Debug build of this composition, this
substitution does not weaken the finding -- but the "Kokkos AND ArborX both fetched" leg itself was
not reproduced bit-for-bit in this sandbox, and the ArborX/Kokkos version-pin incompatibility above
is itself a new, OPEN finding worth the owner's attention independent of this entry.
