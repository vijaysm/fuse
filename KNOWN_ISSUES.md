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

### A1. There is no remote yet — deferred by decision, with one residual risk

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

**What to do — when S6 comes up, not now.** Either name a code, or say you
would rather we use a second separately-built FUSE executable as a stand-in. The
stand-in is a weaker result and would be reported as such, naming which of the
four properties above went untested — a program we wrote knows our conventions
even when it is built separately, so it cannot show that an unfamiliar code
works.

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

### ~~B1. Tier 2 does not build: the MOAB backend is unported~~ — CLOSED 2026-09-18, superseded by C1

Stale and false as of the S5.2 port: `include/fuse/backend/moab_backend.hpp`
and `src/backend/moab_backend.cpp` both exist and are wired into
`fuse-coupler-backend`, so the compile error quoted below cannot reproduce at
HEAD (REVIEW.md R-23). See C1 below for the current status of tier 2 — it
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

### ~~B1a. The no-exceptions build aborts where the library catches its own refusal~~ — CLOSED 2026-09-17

Found by running the `tier0-no-exceptions` leg's recipe (11 of 57 aborting),
fixed in `9975cb3` and `555c62d`. The leg now passes **48/48**, and the normal
build stays at 65/65.

**What it was.** `FUSE_REFUSE` throws when exceptions are on and
reports-then-aborts when they are off, so any place the library threw and
caught *its own* refusal to build a value died before the catch ran.

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
from it: B1 above was stale (tier 2 builds, see C1) — now closed, above. C1 below understated the
MOAB situation — the test file opened with `#ifdef FUSE_HAVE_MOAB` and no config include, so its 38
cases compiled to a skip stub in every configuration, host included (`REVIEW.md` R-06, not R-09 as
first written here; R-09 is the unrelated `step_period()` collective-agreement BLOCKER). `test/backend/moab_mesh_test.cpp` now includes `config.hpp` and tests `#if FUSE_HAVE_MOAB` — the
cases compile again, but C1's "needs one host run" still stands: nothing here can `MPI_Init`.

### C1. The MOAB backend is built but unrun in the sandbox

**Status: needs one host run.** `S5.2` landed 2026-09-18: tier 2 configures,
compiles and links for the first time, and `test/backend/moab_mesh_test.cpp`
(38 cases, ported from the parent) is registered whenever MOAB is enabled.

None of it has executed here. Tier 2 requires MPI by the tier ladder, and this
sandbox cannot `MPI_Init` — every test in an MPI-enabled build dies at startup
in `MPI_Init_thread`, which is why tier 2 reports 81 of 87 failing here and why
that number says nothing about the code. The ladder is right to refuse MOAB
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

### B3. `supports_rollback()` is still a single bool

`include/fuse/runtime/component.hpp`, `class Component`:

```cpp
[[nodiscard]] virtual bool supports_rollback() const
```

The model layer already carries `capabilities` / `required_capabilities` as
`std::uint64_t` (`include/fuse/model/semantics.hpp`). Replacing the bool with
the bitmask would let a component declare its determinism level and whether it
holds hidden state — which the declared `SurrogateContract` already asks about
and currently cannot be told. It changes the component interface, so it
sequences with the ABI work rather than being done in passing.

---

## D. Real limitations, documented and pinned

### ~~D1. A region's contiguity depends on declaration order~~ — FIXED 2026-09-21

Two independent channels on the same clock, one inside an iteration region.
The scheduler ordered an instant by topological rank alone, so the two chains
interleaved whenever their ranks alternated — and whether they did depended on
the order the user happened to declare their components in. A region cut in
half that way was refused (`E-SCHED-REGION`) although an order existed that
kept it whole.

The interleaving instruction was usually the region's **own** sink: a
non-member that reads the region's converged output and must run after the
loop, which the rank tie-break sometimes placed between two members.

**Fixed** by `group_regions_within_instants()` (`src/schedule/scheduler.cpp`):
within each (time, phase) instant, emit greedily and prefer an open region's
ready members. Only ready instructions are ever emitted, so it cannot invert a
dependence — and with no region in an instant the result is the rank order
itself, which is why every golden IR file is byte-identical.

Grouping is **skipped** for an instant where a non-member reads an arc that a
member also reads and that no member produces — the same external value the
region re-advances. That is the hanging-channel hazard
(`Iteration.RejectsAnExplicitChannelHangingOffAnIteratedComponent`): the
non-member would run once, on one sweep's iterate, so separating the interleave
would turn a refusal into a silently chosen placement. Such an instant keeps
the order it had, and with it the verdict it produced before. The first attempt
at this fix did separate it, and that test is what caught it.

`ExecutionPlan.ARegionSchedulesWhicheverOrderItsNeighbourIsDeclaredIn` asserts
both spellings schedule; two mutations fail if either half of the rule is
removed. `E-SCHED-REGION`'s message no longer mentions declaration order.

---

### ~~D2. One CommGraph-backed Transport per run~~ — FIXED 2026-09-20

`Runner::Impl` holds a single `std::unique_ptr<Transport>`, so a plan with two
cross-rank channels is refused at `initialize()`:

```
E-RT-TRANSPORT: this build wires one CommGraph-backed Transport per run, but
the plan has 2. One Transport instance per instruction is the next step.
```

Two independent cross-rank couplings is not an exotic graph — it is the
ordinary multiphysics case, and the brief this library exists to serve names
it. So this is the sharpest remaining limit on what can actually be coupled,
not a corner.

Surfaced 2026-09-20 by the host Tier 1 run once two earlier faults in the same
test were cleared (`RunnerTransport.StillRunsASecondChannelAfterARegionFails
UnderMpi`, which needs exactly this). That test is the acceptance criterion and
is already written; it fails on this and nothing else.

**Fixed.** `Impl::graph_transports` is a vector of one Transport per
CommGraph-backed plan, built inside the per-plan loop of
`attach_comm_graphs()` — which is where it has to be, because
`peer_plan_needed`/`send_peers`/`recv_peers` describe the plan being walked
and are overwritten by the next iteration. Building after the loop, on the
last plan's peer lists, is what had limited the runner to one channel. Each
instance is created from the registry, passed through
`RunnerConfig::transport_setup`, then planned; every rank builds the same
number in the same order, which matters because `set_communicator()`
duplicates the communicator and `build_plan()` is collective. Instances are
NOT shared between plans: a Transport carries the plan it was built for and
`prepare()` refuses any other spec. Mutation
`one-transport-shared-across-every-plan` pins that. Verified in the sandbox
only as far as compiling in all three tiers; the acceptance test is
`RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi` and needs
a host run.

**The superseded plan was** — `transport` becomes a vector indexed by transport plan;
`attach_comm_graphs()` builds and wires one per plan rather than one per run
(`src/runtime/runner.cpp:1321` is the refusal, `:1467` the single-instance
`dynamic_cast` wiring, `:2097` and `:2124` the execution sites). The open
design question is whether two plans with an identical `TransportSpec` share
one instance or get one each.

---

### D4. Two axes of the parallel matrix have no coverage — the gate is RED on purpose

`tools/check-parallel-matrix.py` (S5.5) checks that every cell of
`build-evidence/parallel-matrix.toml` names coverage that exists: the test is
really in the file, the file is really registered, the refusal is really in the
library *and* asserted somewhere, and a claimed rank count is really run. It
reports **33 claims verified, 0 broken, 2 gaps**, and exits 1 on the gaps.

| Gap | What has no test |
|---|---|
| `A-PARTITIONED` | an **independently partitioned source**: `build_remap_halo`, `exchange_remap_halo` and `E-RT-REMAP-HALO` have 34 references across `src/` and `include/` and **none under `test/`** |
| `A-SPACES-TRANSPORT` | the host-parallel space axis beyond the kernels: only `space_kernels_test.cpp` is typed over execution spaces, so transport and runtime run on one space |

`A-PARTITIONED` is the axis S5.5 records as appearing nowhere, and it is the
largest untested surface in the parallel path. A remap whose source is
partitioned independently of its target needs values the rank does not own;
`localise()` refuses that (`E-MAP-COVERAGE`, tested), which is correct but is
not coverage *of* the case — the halo exchange is what makes it work, and
nothing exercises it.

A gap is deliberately not waivable: the gate counts them and fails, so the
number can only fall by writing the test. A **third** gap appearing, or any
`MATRIX-BROKEN` line, is a new defect rather than one of these.

**Also found by building this gate**, and fixed: rank count **1** was never
run (`fuse_add_mpi_test` defaulted to `2 3 4 5`). A single task is not a
trivial case of the others — every match has both ends on one rank, so the
same-rank shortcut carries the whole exchange and `MpiTransport` must see
nothing, which is row 7's requirement from the other side. Tier 1 went from 92
to 97 registered tests. **Unverified here:** MPI cannot initialise in this
sandbox, so a test that fails at 1 rank instead of skipping — i.e. one missing
its `FUSE_REQUIRE_AT_LEAST_RANKS` guard — will only show up on a host run.

---

### D3. Three behaviours no test detects — the mutation gate is RED on purpose

`FUSE_MUTATE_MPI=1 tools/mutate.py --area transport|runtime` exits 1 with three
survivors. All 91 tier-1 tests pass; what is missing is coverage, and these
three are **known** — a fourth survivor is a new gap, not one of these.

| Survivor | Behaviour with no test that detects it |
|---|---|
| `one-transport-shared-across-every-plan` | D2's fix: one Transport per CommGraph-backed plan. Collapsing every plan onto instance 0 fails nothing. |
| `region-failure-abandons-the-rest-of-the-period` | R-09: a failing period carries the failure forward instead of returning at the region. |
| `region-residual-skips-the-cross-rank-reduction` | R-10: the region residual is agreed across participants rather than summed rank-locally. |

**One cause for all three.** In
`RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi` the
independent channel is declared *first* — the D1 workaround — and the iteration
region *last*. So every observable in that test happens before the region, and
"the rest of the period" after it is empty. R-09's mutant skips nothing; the
shared-Transport mutant still fails the run, and the first failure wins, so the
asserted code is unchanged; and no case in that binary has two cross-rank
channels both delivering. R-10's local-sum fallback reaches the same
convergence decision on this data.

Note what this means about the earlier claim that
`StillRunsASecondChannelAfterARegionFailsUnderMpi` was the acceptance test for
D2's capability: it is not, and the mutation is what established that.

**Fixing D1 comes first** (the user's ruling, 2026-09-21). The declaration-order
limitation is what forced the reordering that blinded the test, so removing it
is the root fix; the three tests are written against the fixture that follows.

---

## C. Not defects — do not fix these

- **Five mutations report `NOT-ASSESSED` locally.** They need MPI, and a test
  that cannot start exits non-zero whether or not the mutation had an effect.
  Recording that as a kill would be evidence worse than none. Three are in
  `transport-mutations.md` (`peer-plan-skips-the-numbering-check`,
  `peer-plan-drops-the-local-copies`, and `build-plan-skips-the-components-agreement` —
  the last tagged R-30 in its own description, added while fixing that finding) and two
  are in `runtime-mutations.md` (`region-residual-skips-the-cross-rank-reduction`,
  tagged R-10, and `region-failure-abandons-the-rest-of-the-period`, tagged R-09;
  `cross-rank-region-check-fires-unconditionally` is tier-0-assessable and KILLED).
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
- **A surface remap pair is refused** (`E-REMAP-NONPLANAR`) rather than
  supported. The planar kernel would return the measure of the xy shadow, which
  is a plausible wrong answer. A surface pair needs a parameterisation.
