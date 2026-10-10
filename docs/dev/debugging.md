# Errors, invariants and debugging

How FUSE fails, and how you find out why. The mechanism is in `include/fuse/base/error.hpp`; this
page is the design, the parts of it that exist, and the parts that are tracked so they are not lost.

## Three spellings, one mechanism

```cpp
FUSE_ASSERT(dt > 0_r, "non-positive step {}", dt);        // invariant: cannot happen
FUSE_THROW(InvalidArgument, "unknown component '{}'", n);  // a recoverable error of a named type
FUSE_REFUSE("E-RT-TRANSPORT-OVERLAP", "ranks {} overlap", r); // a deliberate refusal (I4)
```

Every one constructs a `fuse::Error` subclass carrying the code, the `fmt`-formatted message, the
source location, the MPI rank (once announced) and a `Backtrace` captured at the throw site; runs the
**on-error policy**; and then throws. In a build with `FUSE_ENABLE_EXCEPTIONS=OFF` the same policy
runs and the process aborts instead — the exception *types* and every call site are unconditional,
because a call site that reports an error names it whether or not the build will throw it.

`FUSE_ASSERT` is checked in **every** build. That is invariant `I6` with a mechanism behind it: an
`assert` compiles out under `NDEBUG`, so a Release build would carry the bug an `assert` was meant to
catch — which is precisely how an unqualified enumerator inside an `assert` survived in an upstream
tree for years ([](roadmap.md)). The test `Error.AssertIsCheckedInEveryBuild` runs on the Release CI
leg for this reason.

`FUSE_DEBUG_ASSERT(cond)` exists for device kernels and inner loops where a throw is impossible or
unaffordable. It aborts in Debug and vanishes in Release, and it is **never the only guard on an
invariant**: the same condition must be checked by a `FUSE_ASSERT` at the kernel's boundary.

## The exception vocabulary

Absorbed from the parent that had one, renamed, and reduced to what the library distinguishes:

| Type | Meaning | Upstream ancestor |
|---|---|---|
| `InvariantViolation` | a `FUSE_ASSERT` fired; the library's own logic is wrong | `LogicError`, `DynamicCastFailure` |
| `InvalidArgument` | the caller broke a documented precondition | `InvalidArgument` |
| `NullPointer` | a null where an object was required (is-an `InvalidArgument`) | `NullPointer` |
| `Refusal` | a deliberate refusal with an `E-…` code tests assert on | *(new — invariant I4)* |
| `NotImplemented` | a declared interface with nothing behind it | `NotImplemented` |
| `IoError` | file or stream access failed; the message names the path | `FileError` |

`throw` preserves the dynamic type, so `catch (const Refusal&)` works; every type is-an `Error`,
so a boundary can catch once. `Error::report()` is the multi-line human form; `code()`, `message()`,
`where()`, `rank()` and `backtrace()` are the structured one, so a `Diagnostic` can carry them
without parsing a string.

## The on-error hook

Modelled on PETSc's `-on_error_attach_debugger` / `-start_in_debugger`, because that is the workflow
people debugging parallel codes already know.

| Variable | Values | Effect |
|---|---|---|
| `FUSE_ON_ERROR` | `throw` · `backtrace` (default) · `debugger` · `abort` | what happens between constructing an error and throwing it |
| `FUSE_DEBUGGER` | `lldb` (default) · `gdb` | which debugger `debugger` attaches |
| `FUSE_DEBUGGER_TERMINAL` | a terminal program, or `none` | `xterm` by default; one window per rank, titled with rank and pid |
| `FUSE_DEBUGGER_RANKS` | `0,3` | only these ranks get a debugger; the rest print and throw |
| `FUSE_DEBUGGER_WAIT` | seconds | how long the process waits for the debugger to attach |
| `FUSE_BACKTRACE` | `0` · `1` | disable capture entirely (an `Error` then carries an empty trace) |

An unrecognised value is an `InvalidArgument`, not a silent default (I4). The same policy is
settable in code through `fuse::set_error_policy`. `fuse::initialize` (S4) reads the environment
once and announces the rank; until then errors carry rank −1.

```bash
FUSE_ON_ERROR=debugger FUSE_DEBUGGER_RANKS=2 mpiexec -n 4 ./my_coupling
```

opens one `xterm` running `lldb -p <pid>` on rank 2 the moment rank 2 throws, while the other three
ranks print their report and continue to the reconciliation `Epoch` — which is what stops the
remaining ranks from hanging in the next collective while you look at rank 2.

## Backtraces

`fuse::Backtrace` captures raw return addresses with `execinfo.h` (macOS, glibc, musl) and
symbolizes only when formatted, so capture allocates nothing (I5 holds on the error path) and the
cost lands where a human reads it. `std::stacktrace` is not implemented in libc++ as of clang 21;
when it is everywhere FUSE builds, `Backtrace` becomes a wrapper over it with the same interface.

`fuse::install_terminate_handler()` prints the report of an uncaught `fuse::Error` — or the `what()`
and a fresh backtrace for any other exception — before aborting. The test harness installs it;
`fuse::initialize` will.

## Status

| Capability | State | Evidence |
|---|---|---|
| `FUSE_ASSERT` checked under `NDEBUG` | **verified** | `Error.AssertIsCheckedInEveryBuild` on the Release leg |
| dynamic type preserved through the macros | **verified** | `Error.ThrowPreservesTheDynamicType` |
| backtrace captured at the throw site, disableable | **verified** | `Error.BacktraceIsCapturedAtTheThrowSite`, `…CanBeDisabledByPolicy` |
| policy from environment, refusing bad values | **verified** | `ErrorPolicy.EnvironmentSelectsEveryMode`, `…IsRefusedNotDefaulted` |
| debugger command and rank selection | **verified** | `ErrorPolicy.DebuggerCommandIsWhatWouldBeRun`, `…DebuggerRankSelection` |
| `abort` policy and `FUSE_DEBUG_ASSERT` abort | **verified** | death tests |
| terminate handler prints a report | **verified** | four death tests: fuse::Error, foreign exception, no active exception, idempotence. They call `std::terminate` from inside a catch, because GoogleTest intercepts an exception escaping a death-test statement; the runtime's own uncaught-throw path is the language's, not ours |
| `attach_debugger` spawns and waits | **implemented, untested** | needs a terminal and a debugger; CI has neither |
| exceptions-off build (`FUSE_ENABLE_EXCEPTIONS=OFF`) | designed; leg arrives with the no-exceptions CI leg | S1 exit |

## Tracked design items

Longer-dated, and listed here precisely so they are not lost. Each becomes a plan step when its
stage arrives.

1. **`start_in_debugger`** — PETSc's `-start_in_debugger`: `fuse::initialize` attaches a debugger to
   the selected ranks *before* anything runs, for bugs that never reach a throw. Same policy fields;
   trigger is initialization, not error. *S4, with `fuse::initialize`.*
2. **Rank-aware attach under MPI** — the message names hostname and pid so a window can be matched
   to a rank on a multi-node run; `FUSE_DEBUGGER_RANKS` accepts ranges (`0-3`) and `all`. *S4.*
3. **Collective error reporting** — when several ranks fail in the same `Epoch`, the reconciled
   report is printed once, by the lowest failing rank, with the set of failing ranks and one
   representative backtrace per distinct code, instead of N interleaved reports. *S3/S4, with
   `Epoch`.*
5. **`std::stacktrace` migration** — when libc++ ships it, replace the `execinfo.h` backend behind the
   same interface; keep the no-allocation-at-capture guarantee. *When the toolchain floor allows.*
6. **Signal handlers** — `SIGSEGV`/`SIGBUS`/`SIGFPE` print a backtrace and honour the debugger policy,
   as PETSc's signal handler does. Off by default: a library must not install signal handlers into a
   host code without being asked. *S4, opt-in through `fuse::initialize`.*
7. **Device-side diagnostics** — `FUSE_DEBUG_ASSERT` in a Kokkos kernel reports through
   `Kokkos::printf` and `Kokkos::abort`, not `execinfo.h`. *S4, with the first device kernel.*


## `FUSE_SCHEDULE_DEBUG` — what the scheduler did to an instant

Set it and `schedule()` prints, for each (time, phase) instant, the bounds,
whether the instant holds iteration-region members, whether it was judged
entangled, and every instruction in it with its region, the arcs it reads and
the arc it writes:

```
[instant 2..10] t=1 phase=0 members=true entangled=false
    [5] Remap "iterated/remap" region=0 reads %1:pa.out writes %5
    ...
    [10] Sink "pb.in" region=-1 reads %7:iterated/diagnose writes %10
```

The order shown is the one **before** region grouping, which is the point: it
is the layout the rule is about to act on.

This is what closed D1. The refusal said a region's body was not contiguous,
and five readings of the ordering code had not said why. The dump showed the
interleaving instruction was the region's *own* sink — a non-member reading
the region's converged output — and that made both halves of the fix obvious:
group the members, but not when a non-member reads an external value the
region re-advances. Reading a diagnostic beats reading the code that produces
it, and this session spent that lesson twice (see also
`FUSE_TRANSPORT_DEBUG` in `rank-agreement.md`).

## Opt-in traces outside `base/error.hpp`'s own mechanism

The on-error hook above is for something that already failed. Several places
in the tree additionally gate an ordinary diagnostic `fmt::print`/`fprintf`
behind a `std::getenv` check of their own — the same "one line per check, a
`getenv` otherwise" convention `FUSE_TRANSPORT_DEBUG` established — for a
quantity that is not wrong, just worth seeing without attaching a debugger:

`FUSE_CONSERVATION_DEBUG`
: `src/runtime/runner.cpp`'s conservation-integral check prints this
  process's own before/after weighted sums next to the totals the verdict
  reduces, plus how many weights are non-zero on each side — enough to see a
  wrong ownership or a wrong weight without instrumenting the check itself.

`FUSE_BRUSS_DEBUG`
: `examples/brusselator/species_core.hpp` prints each rank's owned/adopted
  cell counts at `make_from_grids()` and a step-level trace in `step()`. This
  is what localised D29 (a remap destination buffer sized from the wrong
  rank's pre-halo owned count, overflowing a `std::vector<double>`): tracing
  with `FUSE_BRUSS_DEBUG=1` at a 3-rank misaligned split showed
  `weights_uv(n_rows=108, ...)` against a cached `n_v_=96` — the 12-row gap
  being exactly that rank's own measured `adopted_v(vu)=12` — which turned
  "corrupted heap, wrong answer from sub-step 0" into a one-line diagnosis.
  No decision in the library rests on this trace; it is purely diagnostic,
  the same contract `FUSE_CONSERVATION_DEBUG` and `FUSE_TRANSPORT_DEBUG` keep.

Neither is a general mechanism the way `FUSE_ON_ERROR`/`base/error.hpp` is —
each is one `getenv` check at one site, added by the track that needed it.
A future similar trace should follow the same convention (opt-in, one line
per check, named for what it traces) rather than invent a new one.
