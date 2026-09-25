# Testing

(the-five-states)=
## The five states

Every capability in FUSE is in exactly one of these, and the documentation says which. The parent
code bases conflated them, and several merge decisions turned entirely on the distinction.

Verified
: A test exists and passes.

Implemented, untested
: Code exists; nothing demonstrates it works.

Designed only
: A document describes it. **A design document is not evidence that the contract exists.**

Deliberate refusal
: FUSE declines the operation on purpose — and this counts as verified *only* if a test makes the
  refusal fire. An unexercised error path is not a feature.

Absent
: Not there.

## The harness

One GoogleTest main, `test/support/fuse_test_main`, for every tier. It exists because the two obvious
approaches are both wrong.

Plain GoogleTest under `mpiexec` lets rank 3 fail an assertion and return while ranks 0–2 proceed
into the next collective. The suite then **hangs**, and a hang is not a test result. So the harness
reduces the failure status across the communicator at the end of every test: a failure anywhere fails
everywhere. That is the same discipline the library applies to itself with `Epoch`
([](invariants.md) `I2`) — a good sign the abstraction is the right one.

Every rank printing its own progress makes the log unreadable, so only rank 0 keeps the default
printer; other ranks stay silent unless they are the rank that failed.

```cpp
FUSE_REQUIRE_RANKS(4);              // wrong size is a clear skip, never a silent pass
FUSE_REQUIRE_AT_LEAST_RANKS(2);
```

## Device coverage is a build option, not a second test file

```cpp
template <typename ExecutionSpace> class RemapTest : public ::testing::Test {};
FUSE_TYPED_TEST_ON_SPACES(RemapTest);
```

One test definition runs on every execution space the build enabled, so the device leg checks the
same assertions as the host leg instead of a parallel copy that drifts. The device leg is
environment-gated, and **no performance claim is made for hardware that has not been measured.**

### The thread-count leg (`ctest -L omp`)

A Kokkos built with OpenMP runs every test on OpenMP at whatever thread count the machine gives,
which is one point on the axis. `fuse_add_omp_legs(<test> <target>)` registers a test twice more,
at `OMP_NUM_THREADS=1` and at every core, so both extremes are measured on every build. It is
applied to the code that actually executes Kokkos kernels: the RCB partitioner, whose
`median.hpp` histogram sums weights with `Kokkos::atomic_add` (the one accumulation in the library
whose order is the threads'), the native mesh, and the typed numerical kernels.
`tools/compare-runs.cmake` additionally runs the two physics examples at both counts and requires
byte-identical output.

Read the label for what it is. `src/runtime/` and `src/schedule/` call no Kokkos kernel — the
Runner's sums, norms and remap apply are the serial loops in `num/kernels.hpp`, and
`num/space_kernels.hpp` is verified but uncalled — so the examples' output is identical across
thread counts *trivially*, and running the runner tests at two counts would compare a serial loop
with itself. That is a capability gap, `KNOWN_ISSUES.md` D11, and the comparison is in the tree so
that the day the runtime moves onto the space kernels it stops being trivial and starts being a test.

## CI legs

The rule is one leg per rung of the ladder, so no configuration can rot unnoticed.

**Running today** (`.github/workflows/ci.yml`): the layering, invariant and import-identity gates; a
documentation build with warnings as errors; **tier 0 as four legs** — Debug and Release across both
linear-algebra backends, with dependencies resolved from the runner's prefixes (`FUSE_ALLOW_FETCH=ON`
there, since a CI runner has no tuned install) — and a **no-exceptions leg** that
builds with `FUSE_ENABLE_EXCEPTIONS=OFF` and runs every test not labelled `throws`.

**Planned**: tier 1 at 2/3/4/5 ranks (S4), tier 2 and tier 3 (S5, S7), an ASan+UBSan leg (S1 exit),
LowFive (S6), and the mutation gate as a CI leg once its runtime is acceptable — today it is run before
a port is accepted and its log committed under `build-evidence/`.

The `throws` label is **declared** on a test target, not detected: a parent tried to find
refusal-asserting tests by grep, by helper name and by catch site, and both detectors were wrong in
opposite directions.

## Mutation evidence

A ported or new kernel is accepted only when at least one **named** mutation per public kernel is
killed by a test, recorded under `build-evidence/`. This is the rule that catches suites which pass
without being able to detect the defect they were written for. A port with zero killed mutants is not
accepted regardless of how green it looks.

## Numerical verification

Analytic solutions and method-of-manufactured-solutions cases assert **observed order of
convergence**, not just a tolerance: a transfer that is accidentally first-order still passes a loose
tolerance and fails an order gate. Conservation properties assert the integral to round-off. Iteration
counts for accelerated coupling are regression-tested exactly, because a silent change in count is how
an acceleration bug presents.

## Golden files

`test/golden/waist-v1.bin` is the frozen canonical descriptor: the bytes a
`fuse::testing::freeze_fixture()` graph serializes to, with every optional member at a non-default
value (a round trip over defaults proves nothing about a member — zero bytes written and zero read
agree). `Descriptor.MatchesTheFrozenWaistV1Bytes` compares against it.

**A golden file is never regenerated by the suite that checks it.** Regeneration is a deliberate human
act through a separate tool:

```bash
fuse_write_golden_descriptor test/golden/waist-v2.bin   # after bumping kGraphVersion
```

The previous file keeps its own name, so a reader can see what the bytes used to be. The tool and the
test share one fixture (`include/fuse/model/freeze_fixture.hpp`) so they cannot describe different
graphs — which is the one way a golden test passes while the frozen bytes mean nothing.

## The three-leg round-trip

`YamlIo.ThreeLegRoundTripIsStructurallyIdentical` authors a graph in C++, emits YAML, parses it back,
and compares the **canonical bytes** of the two. The comparison is against the bytes rather than
between the two graphs because two legs prove nothing about the third: a front-end that loses a member
symmetrically passes its own round-trip.

It earned that on its first run, catching a defect inherited from the parent: a unit with no symbol in
the table was emitted as a composed string (`kg*m^-1*s^-2`) that the parser could not read, so a graph
with a kilopascal field emitted cleanly and then failed to parse. An emitter whose own parser cannot
read its output is not a front-end. The fallback is now a structural map of the exponents plus the
exact scale and offset; a unit that does have a symbol still emits as that symbol, so ordinary YAML
stays readable.

## Cross-rank descriptor agreement

`model.descriptor_mpi.{2,3,4,5}` asserts that every rank serializes the same graph to the same bytes
— the last condition on the waist freeze, and one a single-process determinism test cannot catch (an
unordered container's iteration order, a pointer value reaching the buffer, a locale-dependent number
format). **Verified on the host, 2026-09-15: passed at 2, 3, 4 and 5 ranks.** The odd counts are
deliberate: a bug that only appears when ranks do not divide evenly is the usual kind.

It is a Tier 1 test and absent from a Tier 0 build, because there is no MPI to run it with and a
skipped test reads like a passing one in a summary line. This sandbox cannot launch MPI (`mpiexec`
fails in `HYDU_sock_create_and_listen_portstr`), so running it is a host action:

```bash
cmake -S . -B build-tier1 -DFUSE_ENABLE_MPI=ON \
      -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/kokkos-kernels;/opt/homebrew"
cmake --build build-tier1 -j && ctest --test-dir build-tier1 -R descriptor_mpi
```

## Host-verified parallel results

The sandbox this library is developed in cannot initialise MPI at all, so every
Tier 1 leg is compile-verified there and **run on the host**. Recorded results,
rather than claims:

| Date | Result |
|---|---|
| 2026-09-15 | `model.descriptor_mpi` at 2/3/4/5 ranks; full suite 796 tests, 4 MOAB skips |
| 2026-09-16 | full suite **843 tests, 0 failures**, 4 MOAB skips, including all 16 `parallel.*` legs: `runner_transport`, `comm_graph`, `mpi_transport`, `epoch_reconcile` at 2/3/4/5 ranks |
| 2026-09-16 | full suite **845 tests, 0 failures** after the peer-wise transport landed, and both `requires = "mpi"` mutations **killed** under `FUSE_MUTATE_MPI=1` — so the numbering check and the local-copy pass are each load-bearing, not merely present |
| 2026-09-17 | full suite **855 tests, 0 failures**, 36.2 s real; 4 skips, all of them the parent suite's MOAB-dependent cases. Adds the 4 `example`, 2 `compile_fail`, 1 `field` and 1 `num` (typed, per execution space) entries. Both `requires = "mpi"` transport mutations **killed** under `FUSE_MUTATE_MPI=1`. |
| 2026-09-17 | **Kokkos 5.1.0** (`OPENMP;SERIAL`) replaced 4.6.0 (`THREADS;SERIAL`) on the machine. Tier 0 65/65 and Tier 1 builds after two fixes: Apple clang needs a libomp hint, and `FindOpenMP` had selected GNU `libgomp` whose ABI lacks the `__kmpc_*` entry points Kokkos' objects call. Kernels now instantiate on `Kokkos::OpenMP` and `Serial`; **Threads is no longer enabled anywhere**, so that space is currently unexercised. |
| 2026-09-17 | full suite **855 tests, 0 failures**, 49.7 s real, on the upgraded **Kokkos 5.1.0** (`OPENMP;SERIAL`). Same 4 parent-suite MOAB skips. Confirms the two OpenMP resolution fixes on the host, not just in the sandbox. |

The 2026-09-16 run is what makes the rank-agreement mechanism verified rather
than asserted: `parallel.epoch_reconcile` fails one rank's component mid-step
and requires every other rank to return a failure instead of blocking.

## Convergence and iteration verification

`test/verification/` holds the two suites that say the coupler computes the
right answer rather than merely a consistent one:

- **`mms_transfer`** measures the convergence order of the coupled response
  against a manufactured solution and gates it at 1 ± 0.1, with an absolute
  error gate as well — an order-1 slope on a curve that is uniformly wrong by a
  constant factor would pass the slope alone. Observed: order 1.005, finest
  error 1.86e-3.
- **Its control** repeats the study with the exchange disabled, and asserts
  that both gates *fail*: order 3e-16 (no convergence) and a 197× larger
  error. Without the control, a run that never coupled could in principle
  satisfy the gates above.
- **`iteration`** covers 32 cases including six exact iteration-count
  regressions, so a change that alters how many sweeps a coupled solve takes
  fails a test rather than being absorbed as noise.

These are ports. Matching the parent's numbers exactly is the point: it is what
says the merged compiler, schedule and runtime reproduce known-good results.

## Mutations that cannot be assessed locally

`tools/mutate.py` reports `NOT-ASSESSED` for a mutation whose test needs a
working MPI launcher, rather than a verdict. The reason is narrow and
important: in an environment where the test cannot start at all, it exits
non-zero whether or not the mutation had any effect, so the runner would
record a **false kill** — evidence that is worse than none. Mark such a
mutation `requires = "mpi"` and assess it with `FUSE_MUTATE_MPI=1` on a host
that can launch it.

## A mutant that is undefined behaviour produces no evidence

`linalg-drops-rank-truncation` looped to `k` instead of `rank` over a vector
sized `rank`, so it read past the end. It was recorded as killed on one build
and survived on the next with no source change between them: the verdict was
whatever followed the allocation. A mutation has to be a *different program*,
not an invalid one — it now writes a small plausible weight to the dropped
columns, which is the defect the test guards against, and kills
deterministically. Same family as the existing rules: a mutant that fails to
compile is inconclusive, one whose test cannot run here is not assessed, and
one whose behaviour is undefined is neither.


## A counter must not be incremented inside a condition

```cpp
// WRONG. For fail_after_ == -1 the left operand is false, so `calls_++`
// is never evaluated: calls() stays 0 however often advance() ran.
if (fail_after_ >= 0 && calls_++ >= fail_after_) { ... }

// Right: count first, then decide.
const int n = calls_++;
if (fail_after_ >= 0 && n >= fail_after_) { ... }
```

This was real, and it was expensive. `runner_transport_test.cpp`'s stub had
the first form, and
`RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi` asserts
`cons_b->calls() >= 1` on a stub with no failure configured — so the
assertion could not pass whatever the runtime did. **Four separate real
defects were found and fixed against that one red test** before the counter
itself was read: an invalid communicator sentinel, an accelerator that
refused an empty field block, a fixture whose declaration order broke a
region's contiguity, and the whole-period advance blanket. All four were
genuine and all four needed fixing — but none of them was ever going to turn
that test green.

The lesson for a test instrument: a counter, a flag, or a log that a test
asserts on must be updated on **every** path, unconditionally, before any
branch that might skip it. An instrument that only works in the case it was
written for is worse than none, because a red test with a broken instrument
looks exactly like a red test with a broken subject.


## Never complete a name a pipe truncated

`grep ... | cut -c1-95` is a fine way to keep a survey readable and a bad
source for a value you are about to write down. A truncated test name looks
like a whole one, and finishing it from context produces a plausible name that
does not exist — `SpaceKernels.ReductionsAgreeWithTheSerialSumToRoundoff` for
the real `...ToRounding`, twice over, in
`build-evidence/parallel-matrix.toml`'s first draft.

The same applies to a name taken from prose. A spec section headed
"bidirectionality via `transpose()`" does not tell you the test is called
`TheReverseLegIsARelabellingAndCostsNoCollective`, and a comment describing a
requirement is not evidence that a test with any particular name meets it.

So: when a document, manifest, or commit message is about to state a symbol,
test name, error code or rank count, re-read it from the file at full width
first. Every gate in `tools/` exists because this class of error is invisible
to review — all seven wrong names in that draft read perfectly well.
