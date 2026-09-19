<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are TR-<id>. -->

# Review: transport and runtime (FUSE, HEAD bdf4346)

Scope: `include/fuse/transport/**`, `src/transport/**`, `include/fuse/runtime/**`,
`src/runtime/**`, and `test/transport`, `test/runtime`, `test/parallel`. Read
against `docs/dev/invariants.md`, `docs/dev/layering.md`, `docs/dev/conventions.md`,
`docs/dev/testing.md`, `docs/dev/porting.md`, `docs/dev/rank-agreement.md`.

No source was edited, no build directory was created, no git state was changed.
Two gate scripts (`tools/check-layering.py`, `tools/check-invariants.py`) were run
read-only from the repo root as a cross-check against findings 4 and 5; nothing else
was executed. No individual translation unit was compiled (`-fsyntax-only`) — every
finding below is confirmed by direct reading of caller and callee, and, for findings
1–5, by an independent `grep`/`python3 -c` check of the exact claim (cited inline).

## 1. Coverage

| File | Lines | Read |
|---|---|---|
| `include/fuse/transport/transport.hpp` | 356 | full |
| `include/fuse/transport/comm_graph.hpp` | 329 | full |
| `include/fuse/transport/tags.hpp` | 43 | full |
| `src/transport/copy_transport.cpp` | 98 | full |
| `src/transport/crystal_router.hpp` | 128 | full |
| `src/transport/crystal_router.cpp` | 322 | full |
| `src/transport/comm_graph.cpp` | 1171 | full |
| `src/transport/mpi_transport.cpp` | 1007 | full |
| `src/transport/lowfive_transport.cpp` | 639 | full |
| `include/fuse/runtime/epoch_schedule.hpp` | 82 | full |
| `include/fuse/runtime/component.hpp` | 349 | full |
| `include/fuse/runtime/runner.hpp` | 174 | full |
| `src/runtime/checkpoint_support.cpp` | 71 | full |
| `src/runtime/runner_impl.hpp` | 424 | full |
| `src/runtime/runner.cpp` | 2265 | full |
| `src/runtime/iteration.cpp` | 773 | full |
| `include/fuse/ir/op.hpp` (out of scope, read to verify `is_collective`) | ~100 | the enum + the two functions only |
| `test/parallel/epoch_reconcile_test.cpp` | 513 | first 140 lines + targeted `grep` over the rest |
| `test/parallel/mpi_transport_test.cpp` | 1272 | ~350 lines read directly (setup, respec, peer-plan, hook tests) + targeted `grep` over the rest |
| `test/parallel/comm_graph_test.cpp` | 1515 | targeted `grep` only (harness/`main`, `FUSE_REQUIRE_RANKS` usage) |
| `test/parallel/runner_transport_test.cpp` | 648 | targeted `grep` only |
| `test/transport/transport_test.cpp`, `test/runtime/*.cpp` | ~2850 combined | not read line-by-line; used only to confirm no test exercises the two BLOCKER scenarios below (targeted `grep` for `IterationRegion`/`region_enter` combined with MPI, and for a components-mismatch test on `build_plan`) |

**Not fully read, and why:** `test/parallel/comm_graph_test.cpp` (1515 lines) and
`test/parallel/runner_transport_test.cpp` (648 lines) were sampled by targeted grep
rather than read end-to-end, given the size of the primary scope (transport +
runtime source is already ~5700 lines). This is a real coverage gap for the "tests
that pass vacuously" and "mutation evidence" angles specifically inside those two
files — see §4. Everything reported as a **finding** below is grounded in the
*source* files, which were read in full, not in an unread test file.

`include/fuse/ir/op.hpp` is outside the assigned scope but is the definition of
`is_collective()`, which `runner.cpp` and `iteration.cpp` both depend on for
deadlock-freedom; it was read narrowly (the `OpKind` enum and `is_collective`/
`to_cstring`) to confirm the classification the runtime relies on is complete.

## 2. Findings table

| id | severity | file:line | title |
|---|---|---|---|
| F1 | BLOCKER | `src/runtime/runner.cpp:2112-2118` | A failing iteration region makes `step_period()` abandon the rest of the period's instruction stream, including any later `Epoch`/`Transport` — a rank that should keep reconciling stops dead |
| F2 | BLOCKER | `src/runtime/iteration.cpp:600-639` | Iteration-region convergence is decided from a purely rank-local residual with no cross-rank agreement; ranks coupled through the region can take a different number of sweeps and desynchronise the region's own per-sweep `Transport`/`Epoch` calls |
| F3 | MAJOR | `src/transport/mpi_transport.cpp:355-358` vs `842-876` | `MpiTransport::build_plan()` has no cross-rank agreement on `TransportSpec::components`, unlike the peer-wise `build_plan_from_peers()`'s `E-TRANSPORT-PEERMATCH` check; a `components` mismatch across ranks is silently accepted and mis-sizes `MPI_Alltoallv`'s count arrays |
| F4 | MAJOR | `include/fuse/transport/transport.hpp:255`, `src/transport/lowfive_transport.cpp:66` | Residual `SOMA_ENABLE_LOWFIVE` (should be `FUSE_ENABLE_LOWFIVE`) survives in a doc comment and in the `E-TRANSPORT-NOLOWFIVE` user-facing error message, because `tools/check-layering.py`'s legacy-macro regex has a word-boundary blind spot on `-D<MACRO>` |
| F5 | MAJOR | `src/runtime/runner.cpp:1567` | `resolve_epoch_participants()`'s `MPI_TAG_UB` safety check queries `MPI_COMM_WORLD` instead of the configured `base_communicator`, defeating the check whenever the two differ |
| F6 | MINOR | `src/transport/lowfive_transport.cpp:447` | The interception probe dataset name `"__soma_lowfive_probe__"` is a second, purely cosmetic residual upstream name (not caught by any gate regex, since it is neither a macro nor a namespace) |
| F7 | DEVIATION | `test/parallel/*.cpp` | None of the four parallel test files in scope use the documented `FUSE_REQUIRE_RANKS`/`FUSE_REQUIRE_AT_LEAST_RANKS` harness macros (`testing.md`); they all hand-roll `if (size < N) GTEST_SKIP()` |

## 3. Findings in full

### F1 — BLOCKER — `src/runtime/runner.cpp:2112-2118` — region failure short-circuits the period instead of reconciling it

**What is wrong.** `step_period()`'s main walk over the instruction stream has two
different failure-handling paths, and they disagree. The ordinary path (instruction
execution, lines 2130-2140) does this on failure:

```cpp
if (!step_failure.has_value()) step_failure = r.error();
if (ins.kind == OpKind::Epoch) return *step_failure;
++i;
continue;
```

— it records the failure in `step_failure` and *keeps walking the stream*, so every
later `Epoch`/`Transport` this rank owes its partners is still reached (guarded by
`is_collective`, line 2130, and by the same rule inside `run_region`,
`iteration.cpp:581`). The region-entry path, immediately above it, does not:

```cpp
if (ins.region_enter >= 0) {
  detail::RegionRuntime& r = regions.at(...);
  const Result<void> rr = run_region(components, r);
  if (!rr.ok()) return rr;          // <-- returns straight out of step_period()
  i = r.exit + 1;
  continue;
}
```

`run_region()` returns a non-`ok()` `Result` for *every* failure that occurs while
executing the region body — a local instruction failure, a component `advance()`
refusal, or an accelerator divergence (`iteration.cpp:644-648`, which returns
`a.error()` directly and does not even go through `step_failure` once). In every one
of those cases `step_period()` executes `return rr;` immediately: it never sets its
own `step_failure`, never continues the `while (i < stream.size())` loop, and never
reaches the tail-tick flush (lines 2149-2156). Any `Epoch` or `Transport` instruction
stamped *after* the region in the same period's stream — for an entirely different
channel, between components that have nothing to do with the region — is never
executed by this rank. A partner rank waiting at that later `Epoch` blocks forever.

**How this was confirmed.** Read `run_region()` end to end
(`iteration.cpp:554-771`): every one of its early returns (`E-ITER-ROLLBACK` at 690,
`E-RT-NOCONVERGE` at 758, the accelerator's `a.error()` at 647, and the two
`step_failure`-reset returns at 594-595 and 763-765) is a plain `return`, and none of
them is routed through `step_period()`'s `step_failure` mechanism. Contrasted this
directly against the general-instruction path in the same function
(`runner.cpp:2130-2140`), which is the only other place `step_period()` decides what
to do about a failure, and confirmed it takes the opposite action (continue, not
return). `rank-agreement.md` states this exact mechanism is supposed to be symmetric
("A component that refuses its step carries forward the same way, both in the period
loop and inside an iteration region — where it matters most"); `roadmap.md`'s S4 row
claims as *done*, "a rank that fails mid-step reconciles at every remaining agreement
point." Grepped `test/parallel/epoch_reconcile_test.cpp` (the test that exists
specifically to pin this mechanism) for `IterationRegion`/`region_enter`: no match —
the test that would have caught this covers only the non-region case.

**Fix.** In `step_period()`, replace `if (!rr.ok()) return rr;` with the same pattern
used for ordinary instructions: set `step_failure` from `rr.error()` if not already
set, and continue the outer loop at `i = r.exit + 1` instead of returning. Symmetric
fix needed for the accelerator-divergence return inside `run_region()`
(`iteration.cpp:644-648`): it must set `step_failure` and fall through to the
existing post-loop failure-reporting code (line 755 onward) rather than returning
straight out of the sweep loop, or every partner still needs to be reached from
`step_period()`'s continuation after all.

**Test to add.** A parallel test that puts an iteration region between two
components on *different, disjoint* rank sets, has a channel with an unrelated third
component's `Transport`/`Epoch` scheduled later in the *same period*, fails the
region's component on one rank, and asserts (under the existing timeout harness,
`epoch_reconcile_test.cpp`'s pattern) that every rank returns rather than hangs.

### F2 — BLOCKER — `src/runtime/iteration.cpp:600-639` — iteration convergence is a per-rank local decision with no cross-rank agreement

**What is wrong.** The residual that decides whether an iteration region has
converged is computed entirely from this rank's own buffers:

```cpp
const double* g = slots[b.value_arc].data;   // this rank's own slot, line 604
...
R.block_residual[bi] = std::sqrt(rs);         // local only
...
if (R.convergence.all_satisfied(R.residual_norms, R.reference_norms)) {  // line 636
  st.converged = true;
  break;                                       // <-- exits the sweep loop, THIS RANK ONLY
}
```

There is no `MPI_Allreduce`, `MPI_Allgather`, or any other cross-rank exchange of the
residual, before or after this check. Confirmed by `grep -rn "MPI_" src/runtime/
iteration.cpp` (no hits) and `grep -rn "Allreduce" src/num/ include/fuse/num/` (no
hits) — the entire convergence machinery (`ConvergenceCriterion::all_satisfied`,
`include/fuse/model/iteration.hpp:171`) is pure per-rank arithmetic.

When a region's body includes a cross-rank `Transport`/`Epoch` (i.e. the two
components it couples are placed on different rank sets, which is exactly the
configuration `CommGraph`/`MpiTransport::build_plan_from_peers` exist for), a
rank-local residual is not a proxy for "the coupled system has converged" — it is a
measurement over whatever slice of the field this rank owns, and two ranks measuring
different slices of a not-yet-converged field will, in general, satisfy the
tolerance at different sweep counts `k`. The sweep loop (`iteration.cpp:573`,
`for (int k = 0; k < R.max_iterations; ++k)`) is per-rank: a rank whose local slice
happens to satisfy the criterion first `break`s out of the loop at line 638 and
returns from `run_region()` while its partner rank, still short of its own local
tolerance, executes another pass through `R.enter..R.exit` — including that pass's
own `Transport`/`Epoch` calls (line 582, inside the region body). The first rank is
no longer inside the region to answer; its partner's `Transport`/`Epoch` for the
extra sweep has no matching call anywhere. This is a hang (for `Epoch`'s
point-to-point pattern) or undefined `MPI_Alltoallv` behaviour (for `Transport`),
depending on which collective the divergent sweep reaches first.

**How this was confirmed.** Read the entire sweep loop (`iteration.cpp:554-753`) and
`ConvergenceCriterion::all_satisfied` (`include/fuse/model/iteration.hpp:171`, no
collective inside it — only local size checks and a fold over the passed-in
vectors). Grepped the whole `src/runtime/` and `src/num/` trees for any reduction
that could be feeding a globally-agreed residual into this call: none exists.
Grepped `test/parallel/*.cpp` for `IterationRegion`: no test combines an iteration
region with multi-rank placement at all — this configuration is untested, which is
consistent with it being broken rather than merely unverified, but the absence of a
test does not by itself prove the defect; the proof is the missing `MPI_*` call in
the convergence path itself.

**Fix.** Either (a) reduce the per-block residual/reference norms with an
`MPI_Allreduce` (sum-of-squares before the final `sqrt`, matching how a global L2
norm over a distributed field is normally computed) before calling `all_satisfied`,
so every rank in the region's rank set evaluates the identical criterion and
therefore takes the identical number of sweeps; or (b) if cross-rank iteration
regions are not yet an intended S5 capability (`roadmap.md` lists S5 — which is
"in progress" — rather than S4 as owning "parallel"), refuse the configuration at
compile time the same way `E-SCHED-MULTISCALE` refuses `Map`/`Reduce`/`Closure`:
detect at `resolve_regions()` that a checkpointed component's rank set differs from
another region member's, or that a `Transport` instruction lies inside
`[R.enter, R.exit]`, and return a new `E-RT-REGION-MPI` rather than silently
compiling and running it.

**Test to add.** Two components on disjoint rank pairs inside one iteration region,
with inputs constructed so that rank 0's local slice converges in fewer sweeps than
rank 1's (e.g. different per-rank tolerances via different local data magnitudes);
assert either a clean refusal at `initialize()` (if (b) is chosen) or identical
`IterationStats::iterations` on every rank together with no hang (if (a) is chosen).

### F3 — MAJOR — `src/transport/mpi_transport.cpp` — `build_plan()` has no cross-rank check that `components` agrees

**What is wrong.** `MpiTransport::build_plan()` validates `spec.components < 1`
purely locally (line 355-358) and never agrees the *value* of `components` across
ranks — only the matching of `global_ids` is allgathered and checked
(lines 468-513). `MpiTransport::build_plan_from_peers()`, added later for the
peer-wise/partial-overlap path, closes exactly this class of gap with an explicit
`MPI_Alltoall` "numbering check" (lines 842-876): each rank tells every peer how many
*doubles* (i.e. `local_indices.size() * components`) it intends to send, and each
peer requires that to equal what it computed for `recvcounts`; a disagreement is
`E-TRANSPORT-PEERMATCH`. That check catches a communicator-numbering mismatch, but
by construction it would equally catch a `components` mismatch between the two
sides, because `components` is baked into the very count being cross-checked.
`build_plan()`'s sender-side `sendcounts[d]` (line 600-602) and receiver-side
`recvcounts[s]` (line 617-619) are each computed from *this rank's own* `nc`
(`static_cast<std::size_t>(std::max(spec.components, 1))`, line 423, 565), with no
equivalent handshake. If two ranks call `build_plan()` with different
`TransportSpec::components` (a caller bug, not something the matched-id logic would
ever notice, since ids are per-entity and carry no per-rank component count), the
sender computes `sendcounts[d]` using its own `nc` while the receiver computes
`recvcounts[s]` using a *different* `nc`. `MPI_Alltoallv` requires
`sendcounts[j]` as declared by rank `i` to equal `recvcounts[i]` as declared by rank
`j`; violating that is undefined behaviour per the MPI standard, and in practice
manifests as data being written into the wrong offsets of `recv_buf` (silently wrong
answer) or, depending on the implementation, a hang.

**How this was confirmed.** Read `build_plan()` (lines 309-685) and
`build_plan_from_peers()` (696-907) side by side; the latter's Phase B comment
(842-847) states the exact rationale ("If set_communicator() was given a different
communicator than the rank graph was built over, the two rank numberings differ and
this disagrees") that would apply equally to a `components` mismatch, and the former
has no analogous phase. Confirmed via `grep -n "components" test/parallel/
mpi_transport_test.cpp` that every existing test either sets `components` before
`build_plan()` (never varying it per rank) or exercises the *unrelated* `prepare()`
respec check (`other.components = spec.components + 1`, line 688, which only
detects a change *within one rank* between `build_plan()` and `prepare()`, not a
cross-rank mismatch at `build_plan()` time). Confirmed via the Runner call site
(`runner.cpp:1372,1382,1394,1431,1433`) that `components` is *in practice* always
identical across ranks there, because it is copied from the compiled plan's
`out.components` (`runner.cpp:1126`), a value derived from the IR type system and
therefore invariant across ranks by construction — so this defect is not reachable
through `Runner`, only through direct use of the public `MpiTransport` API (which
`test/parallel/mpi_transport_test.cpp` exercises directly, unmediated by the
Runner's invariant).

**Fix.** Add a components-agreement collective to `build_plan()`, symmetric to
`build_plan_from_peers()`'s Phase B: e.g. fold `spec.components` into the existing
Collective 2 (`MPI_Allgather` of `my_count`, line 469-478) by also allgathering
`components`, or a single `MPI_Allreduce` of `(min components, max components)` and
refusing with a new or reused error code (`E-TRANSPORT-PEERMATCH` reads correctly
here too) when they disagree.

**Test to add.** A 2-rank test where rank 0 calls `build_plan()` with
`components = 1` and rank 1 with `components = 2` over an otherwise valid disjoint
pattern; assert a refusal (`E-TRANSPORT-PEERMATCH` or equivalent) rather than a
successful plan.

### F4 — MAJOR — residual `SOMA_ENABLE_LOWFIVE` survives, and the legacy-name gate has a blind spot that let it through

**What is wrong.** Two places tell a user or a reader to configure LowFive with a
CMake option that does not exist:

- `include/fuse/transport/transport.hpp:255` (doc comment on
  `LowFiveTransport::is_available()`): `` `-DSOMA_ENABLE_LOWFIVE=ON` ``
- `src/transport/lowfive_transport.cpp:66` (the actual `E-TRANSPORT-NOLOWFIVE`
  message returned to a caller from `set_location()`/`build_plan()`/`move()` on a
  build without LowFive): `"Reconfigure with -DSOMA_ENABLE_LOWFIVE=ON and ..."`

The real option, defined in `cmake/FuseDependencies.cmake:49` and consumed at
`src/CMakeLists.txt:13`, is `FUSE_ENABLE_LOWFIVE` — per `docs/dev/porting.md`'s
naming map ("`SOMA_*`, `KARMA_*` macros and CMake options → `FUSE_*`"), which
`SOMA_ENABLE_LOWFIVE` should have followed and did not. A user who hits
`E-TRANSPORT-NOLOWFIVE` and reconfigures exactly as told gets a CMake warning about
an unused variable, not LowFive.

This is exactly the class of defect `tools/check-layering.py`'s `LEGACY_MACRO`
check (`\b(KARMA|SOMA)_[A-Z_]+\b`, `check-layering.py:91`) exists to catch, and it
is why the gate's clean run (`check-layering: 0 violation(s)`, reproduced directly)
is misleading here: the regex requires a word boundary immediately before `SOMA`/
`KARMA`, and in both occurrences the macro name is written as a CMake `-D` flag —
`-DSOMA_ENABLE_LOWFIVE=ON` — where the character immediately preceding `S` is `D`,
a word character. `\bSOMA` does not match `...DSOMA...` because there is no
transition from non-word to word at that position. Verified directly:

```python
>>> re.compile(r'\b(KARMA|SOMA)_[A-Z_]+\b').search("-DSOMA_ENABLE_LOWFIVE=ON")
None
```

Any future legacy macro name written the conventional CMake way (`-D<MACRO>=...`)
in a comment or an error string will pass this gate silently, the same way these
two did.

**Fix.** Correct both strings to `FUSE_ENABLE_LOWFIVE`. Separately, harden
`LEGACY_MACRO` in `tools/check-layering.py` to also match a macro name immediately
preceded by `-D` (e.g. `r'(?:\b|-D)(KARMA|SOMA)_[A-Z_]+\b'`), so a `-D`-prefixed
legacy option name is caught the same way a bare one already is.

**Test to add.** None needed in the C++ suite; add a `tools/`-level test case (a
string fixture containing `"-DSOMA_ENABLE_X=ON"`) asserting `LEGACY_MACRO.search`
(or the corrected pattern) fires on it, so this exact blind spot cannot regress
silently a second time.

### F5 — MAJOR — the `MPI_TAG_UB` safety check queries the wrong communicator

**What is wrong.** `Runner::Impl::resolve_epoch_participants()` checks, once at
setup, that the highest tag this library uses (`tags::kMaxUsed`, i.e. `kEpoch =
0x5C14`) is within the implementation's `MPI_TAG_UB`, specifically to turn a
would-be "Invalid tag" failure inside a blocking `MPI_Recv`/`MPI_Send` (deep inside
`reconcile_epoch()`, which several ranks would already be waiting inside) into a
clean setup-time refusal (`E-RT-EPOCH-TAG`). The check reads:

```cpp
if (MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &found) == MPI_SUCCESS &&
    found != 0 && ub != nullptr && *static_cast<int*>(ub) < tags::kMaxUsed) {
```

— it queries `MPI_COMM_WORLD`, not the communicator that `reconcile_epoch()` and
`attach_comm_graphs()` actually use, which is `config.base_communicator`
(`MPI_Comm_f2c(*config.base_communicator)`, e.g. `runner.cpp:1619-1620`). Per the MPI
standard, `MPI_TAG_UB` is an attribute *of a communicator* and is not guaranteed
identical across every communicator in a process — it commonly *is* identical to
`MPI_COMM_WORLD`'s value for communicators derived from it by split/dup within one
MPI library, but nothing in the standard or in this code path establishes that,
and `RunnerConfig::base_communicator`'s own contract (`runner.hpp:29-31`) is stated
in general terms ("a communicator whose group contains every rank of every
component"), not "must be `MPI_COMM_WORLD`" — that equivalence is only true "under
spec §6 mode 1, which is the only integration mode v1 implements" (the same doc
comment), i.e. a documented *current usage pattern*, not an enforced precondition.
If a future or unusual caller hands `base_communicator` something other than
`MPI_COMM_WORLD` itself (an inter-communicator, a communicator merged from separate
MPI jobs, or one obtained through a library with a lower configured tag ceiling),
this check silently passes using `MPI_COMM_WORLD`'s (possibly higher) bound while
the communicator actually used by every `MPI_Send`/`MPI_Recv` in `reconcile_epoch()`
could have a lower one — reproducing precisely the "Invalid tag" failure inside a
reduction, on one rank, while its partners wait, that this check was written to
prevent.

**How this was confirmed.** Read `resolve_epoch_participants()`
(`runner.cpp:1556-1610`) and `reconcile_epoch()` (`runner.cpp:1612-1707`) together:
the former never converts `config.base_communicator` to an `MPI_Comm` before the
`MPI_Comm_get_attr` call; the latter always does (`runner.cpp:1619-1620`) and uses
that `base` for every `MPI_Recv`/`MPI_Send`. No other code path re-validates
`MPI_TAG_UB` against `base` anywhere in `src/runtime/` or `src/transport/`.

**Fix.** Convert `*config.base_communicator` to an `MPI_Comm` first (the same one
line used in `reconcile_epoch()`) and pass that to `MPI_Comm_get_attr` instead of
`MPI_COMM_WORLD`.

**Test to add.** Cannot be exercised without two communicators of different
`MPI_TAG_UB` in one job, which is not portably constructible in a unit test; note
in the fix's commit that this is verified by code inspection (matching `base` in
both functions) rather than by a runtime test, and consider a compile-time
`static_assert`-style comment pointing future editors at `reconcile_epoch()`'s use
of `base` so the two do not drift apart again.

### F6 — MINOR — a second, purely cosmetic residual upstream name

`src/transport/lowfive_transport.cpp:447` names its interception probe dataset
`"__soma_lowfive_probe__"`. This string never reaches a user (it exists only inside
one process's own HDF5 file, created and deleted within `build_plan()`), so it has
no functional consequence, but it is another leftover parent name that `docs/dev/
porting.md`'s renaming discipline would have caught had the string used `fuse`
instead. It is not caught by `check-layering.py` because none of `LEGACY_NAMESPACE`,
`LEGACY_TYPE_PREFIX`, or `LEGACY_MACRO` match a bare lowercase identifier that is
not a namespace qualifier or an upper-case macro. Rename to `"__fuse_lowfive_probe__"`
for consistency; no test needed.

### F7 — DEVIATION — parallel tests in scope do not use the documented rank-requirement macros

`docs/dev/testing.md` presents `FUSE_REQUIRE_RANKS(n)`/`FUSE_REQUIRE_AT_LEAST_RANKS(n)`
(defined in `test/support/fuse_test_main.hpp:50-58`, confirmed present and wired to
`GTEST_SKIP()` with a message naming the actual rank count) as *the* convention for
declaring a test's rank requirement, specifically so the requirement is "checkable
by grep" (`fuse_test_main.hpp:39-41`). None of the four files in scope
(`comm_graph_test.cpp`, `epoch_reconcile_test.cpp`, `mpi_transport_test.cpp`,
`runner_transport_test.cpp`) uses either macro — they hand-roll patterns like
`if (size < 2) GTEST_SKIP();` (`mpi_transport_test.cpp:641`) or
`if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";`
(same file, line 679). Functionally these are equivalent (an explicit skip, not a
silent pass — the thing `testing.md` actually requires), so this is a convention
deviation rather than a correctness defect, but it means the rank-count matrix for
these four files is not "checkable by grep" against the macro the way the rest of
the suite presumably is. No fix required for correctness; flagged so a maintainer
can decide whether to convert these call sites for consistency.

## 4. Not found

Checked for and did **not** find:

- No `#ifdef FUSE_HAVE_*`/`#ifdef FUSE_ENABLE_*` anywhere in the transport or
  runtime scope (`grep` across all 16 source/header files; every feature-macro read
  is `#if`, and every file that reads one also includes `fuse/base/config.hpp`
  directly).
- No `mpi.h`, `hdf5.h`, or LowFive header in a public header (`include/fuse/
  transport/*.hpp`, `include/fuse/runtime/*.hpp`) — I7 holds.
- No bare `throw std::...`, no bare `assert`, no `iostream`, no un-excepted
  `printf`/`fprintf` in library code in this scope; the three `std::fprintf` calls
  in the `abort_job()` panic paths (`comm_graph.cpp`, `crystal_router.cpp`,
  `mpi_transport.cpp`) all carry the `NOLINT(fuse-raw-io)` marker `tools/
  check-invariants.py` requires, and a direct run of both `check-layering.py` and
  `check-invariants.py` from the repo root reports 0 violations (the layering gate's
  clean run is the reason F4 exists as a *finding*, not a contradiction of it).
- No unmatched MPI tag: every tag used in the scanned files (`tags::kCrystalRouter`,
  `tags::kUnionCommCreate`, `tags::kEpoch`) is declared once in `tags.hpp` and used
  only where documented; `CrystalRouter::route()`'s single-tag point-to-point
  exchange, `CommGraph::build()`'s union-communicator creation
  (`tags::kUnionCommCreate`), and `reconcile_epoch()`'s reduction
  (`tags::kEpoch`) do not collide with each other or with `MPI_Alltoallv`/
  `MPI_Alltoall`/`MPI_Allreduce`/`MPI_Allgather`(v) calls, which are untagged
  collectives and cannot collide with a point-to-point tag.
- No leaked MPI resource on any *examined* path: `MpiTransport::Impl`'s destructor
  guards `MPI_Comm_free` with `MPI_Finalized()`; `CommGraph::build()`'s `finish`
  lambda frees the union communicator on every return path including every error
  path; `attach_comm_graphs()` frees every per-component split communicator
  unconditionally after the (possibly early-exited) plan loop, and every rank
  reaches that free identically because the splits themselves are unconditional.
- The binomial-tree reduce/broadcast in `epoch_schedule()` (`epoch_schedule.hpp`)
  was hand-traced for `participants` = 1, 2, 3 (odd), 4, and 5 and is internally
  consistent (every `reduce_recv_from` has exactly one matching `reduce_send_to`,
  every non-root receives from exactly participant 0). No off-by-one found.
- `CrystalRouter`'s odd/even split pairing (`crystal_router.cpp`, the `bottom`/`odd`
  branch) was hand-traced for `n` = 2, 3, 4, 5 ranks and is self-consistent (the one
  unpaired bottom rank when `n` is odd sends without expecting a same-round
  receive, and its target rank knows to expect the extra message because it can
  derive `n`'s parity itself).
- The byte-exact rollback verification (`iteration.cpp:679-706`, `std::memcmp`) is
  correctly bytewise (not tolerance-based) and correctly ordered *after*
  `restore_state()` and *before* delivery, matching `rank-agreement.md`'s "What is
  atomic" table.
- `Constrain`'s conservation check (`runner.cpp:1966-2001`) was **not** checked for
  the same rank-locality concern as F2 (whether a conservative-remap integral check
  needs a cross-rank reduction when the two regions are split across ranks) — this
  is a real open question this review did not have time to settle and is called out
  provisionally here rather than asserted either way; `test/verification/
  mms_transfer` measures order of convergence but that test was not confirmed to
  run at more than one rank.
- `test/parallel/comm_graph_test.cpp` (1515 lines) and `test/parallel/
  runner_transport_test.cpp` (648 lines) were not read end-to-end (see §1) — mutation
  evidence and vacuous-pass risk in those two files specifically is unassessed.

## 5. Suggested new tests / mutations

1. (F1) Iteration region + unrelated later channel, one rank's component in the
   region refuses its step; assert every rank returns (no hang) under the existing
   MPI timeout harness.
2. (F2) Cross-rank iteration region with a manufactured per-rank convergence skew;
   assert either a compile-time refusal or identical sweep counts on every rank.
3. (F3) `build_plan()` with disagreeing `components` across two ranks; assert a
   refusal, not a successful plan.
4. (F4) A `tools/`-level regex-fixture test pinning `LEGACY_MACRO` against a
   `-D<MACRO>=` string.
5. Mutation: flip `if (!rr.ok()) return rr;` (F1's exact line) to the two-line
   `step_failure = ...; continue;` fix, and confirm the new test in item 1 changes
   from hang/timeout to a clean multi-rank failure — this is the mutation that
   proves the fix and should be recorded under `build-evidence/` per I9.
6. Mutation: in `build_plan()`, inject a components-agreement `MPI_Allreduce` per
   the F3 fix, and confirm the new test in item 3 flips from an accepted plan to
   `E-TRANSPORT-PEERMATCH`.

## Summary

**7 findings: 2 BLOCKER, 3 MAJOR, 1 MINOR, 1 DEVIATION.** The two BLOCKERs are both
in the iteration-region machinery under multi-rank placement: a region failure
abandons the rest of the period's collective schedule instead of reconciling it
(F1), and region convergence is decided from a rank-local residual with no
cross-rank agreement at all, so coupled ranks can silently take different numbers
of sweeps and desynchronise the region's own per-sweep exchanges (F2) — both are
hang/wrong-answer risks in exactly the configuration `rank-agreement.md` and the S4
roadmap entry claim is already handled, and neither is exercised by any test in the
suite. The three MAJORs are a missing cross-rank handshake on `components` in the
older, matched `MpiTransport::build_plan()` that its newer peer-wise sibling already
has (F3); two leftover `SOMA_ENABLE_LOWFIVE` references — one of them a user-facing
error message telling readers to pass a CMake flag that does not exist — that
survived because the legacy-name gate's regex has a word-boundary blind spot on
`-D<MACRO>` syntax (F4); and an `MPI_TAG_UB` safety check that queries
`MPI_COMM_WORLD` instead of the runner's actual `base_communicator`, which is a
latent gap given the API does not force the two to be the same (F5). The MINOR (F6)
and DEVIATION (F7) are cosmetic and convention items with no correctness impact.
