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
