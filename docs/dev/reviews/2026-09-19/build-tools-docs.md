<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are BT-<id>. -->

# FUSE build/tools/docs review — HEAD bdf4346

Scope: `CMakeLists.txt`, `src/CMakeLists.txt`, `test/CMakeLists.txt`,
`examples/CMakeLists.txt`, `cmake/FuseDependencies.cmake`, `CMakePresets.json`
(uncommitted), `include/fuse/base/config.hpp.in`, `tools/*.py`,
`build-evidence/mutations.toml` + `*-mutations.md`, `.github/workflows/ci.yml`,
`UPSTREAMS.lock`, `KNOWN_ISSUES.md`, `docs/**`, `examples/**`. Read-only review;
no repository state was changed, no build directory was created under the
repository, and nothing was built.

## 1. Coverage

**Read in full, line by line:**
- `CMakeLists.txt` (70), `src/CMakeLists.txt` (232), `test/CMakeLists.txt` (200),
  `examples/CMakeLists.txt` (42), `cmake/FuseDependencies.cmake` (425),
  `CMakePresets.json` (84), `include/fuse/base/config.hpp.in` (23)
- `tools/check-layering.py` (202), `tools/check-invariants.py` (157),
  `tools/mutate.py` (157), `tools/sync-upstream.py` (116),
  `tools/verify-import.py` (158)
- `.github/workflows/ci.yml` (92), `UPSTREAMS.lock` (95), `KNOWN_ISSUES.md` (287)
- `build-evidence/mutations.toml` (729, all 70 entries) and all seven
  `build-evidence/*-mutations.md` logs
- `docs/dev/invariants.md`, `layering.md`, `conventions.md`, `testing.md`,
  `porting.md` (the five binding-rule pages named in the task), plus
  `docs/user/build.md` and `docs/dev/roadmap.md` (the two pages whose claims
  bear most directly on the build system)
- `examples/mms_pair/main.cpp` (211), `examples/dn_heat/main.cpp` (659) read for
  structure and macro usage; `examples/*/{heat,dn}_component.hpp` (267+397)
  scanned for feature-macro misuse (none found)
- `test/backend/moab_mesh_test.cpp` (755 lines) read in full after the
  preprocessor check below flagged it
- `include/fuse/runtime/epoch_schedule.hpp` (69) read in full to confirm a
  stale mutation anchor
- `src/backend/moab_backend.cpp` and `src/backend/intx_weights.cpp` read for
  their MOAB/MPI preprocessor guards (headers + first ~40 lines each, plus
  targeted greps for every `#if`/`#ifdef` in each file)

**Compiled/executed, in `/tmp`, nothing under the repository:**
- `tools/check-layering.py` and `tools/check-invariants.py` run directly
  against the working tree (both exit 0, see §4).
- `test/backend/moab_mesh_test.cpp` preprocessed with `clang++ -E`, once with
  `-DFUSE_HAVE_MOAB=1` on the command line and once with nothing defined
  (stub `gtest/gtest.h` supplied via `-include` in `/tmp`), to settle what the
  file's own `#ifdef` actually selects independent of any build (Finding 2).
- `tools/sync-upstream.py`'s `lock()` parser imported and run against the
  real `UPSTREAMS.lock` to confirm it returns empty `superseded`/`diverged`
  lists (Finding 4).
- `tools/mutate.py`'s exact anchor-matching rule (`text.count(old) == 1`)
  replayed in Python against all 70 entries in `build-evidence/mutations.toml`
  and the current source tree, to find every stale anchor (Finding 5).
- A synthetic header under `/tmp` fed to `check-layering.py`'s
  `check_legacy()` to confirm the comment-stripping gap (Finding 12).
- A systematic cross-reference of every `test/<layer>/*_test.cpp` file
  against every registration call in `test/CMakeLists.txt` (regex-driven,
  independently verified against the pre-existing `build-tier1/test/CTestTestfile.cmake`
  on disk) to find every test file that is never compiled (Finding 3).

**Not built:** no `cmake --build` or `ctest` was run; the project was not
configured in a new directory. `build-tier1/` and `build/tier0`, `build/tier2`
already existed on disk from prior sessions and were only *read* (their
generated `CTestTestfile.cmake` was used as independent evidence for Finding 3).

## 2. Findings table

| id | severity | file:line | title |
|---|---|---|---|
| F1 | BLOCKER | `.github/workflows/ci.yml:57` | `tier0` CI matrix's `kokkoskernels` leg cannot configure; `FUSE_LINALG_BACKEND` has no such value |
| F2 | BLOCKER | `test/backend/moab_mesh_test.cpp:3` | The entire MOAB backend test file compiles to a single skip stub, in every configuration, because its own `#ifdef` precedes any include that could define the macro |
| F3 | BLOCKER | `test/CMakeLists.txt:97-101` | `test/schedule/buffers_test.cpp` and `test/schedule/slab_test.cpp` (33 test cases) are never registered and have never been compiled or run |
| F4 | MAJOR | `tools/sync-upstream.py:39-46` | The lock-file parser cannot read the multi-line `superseded`/`diverged` arrays `UPSTREAMS.lock` actually uses; `superseded` silently comes back empty |
| F5 | MAJOR | `build-evidence/mutations.toml:455` | The `epoch-schedule-drops-the-answer` mutation anchor no longer exists in `epoch_schedule.hpp`; the committed "KILLED" verdict is unverifiable against the current implementation |
| F6 | MAJOR | `UPSTREAMS.lock:90-95` | The four MOAB paths superseded by the S5.2 port were recorded only as a comment, never added to `[soma].superseded`, contradicting `porting.md` step 6 |
| F7 | MAJOR | `KNOWN_ISSUES.md:165-176` | B1 ("Tier 2 does not build") is stale and false at HEAD, contradicted by C1 fifty lines below it in the same file/commit |
| F8 | MAJOR | `docs/user/build.md:9,29-37` | The build doc documents a `kokkoskernels` linear-algebra backend that does not exist, omits the real `builtin` backend, and claims Tier 0 already ships "both C ABIs" and "the Fortran shim" |
| F9 | MAJOR | *(absence, whole tree)* | No `install()`/`export()` anywhere; `conventions.md`'s "CMake package `fuse-coupler`" cannot currently be `find_package()`-d by anyone |
| F10 | MINOR | `src/backend/intx_weights.cpp:36` | `#if defined(FUSE_HAVE_MPI)` tests presence, not value, of a `#cmakedefine01` macro (latently masked by the tier ladder today) |
| F11 | MINOR | `CMakeLists.txt:65,68` | Stale status message: claims "stage S1" and lists only the `base` layer, four stages and eight layers after the fact |
| F12 | MINOR | `tools/check-layering.py:76-113` (`check_legacy`) | Legacy-name checks are not comment-aware, unlike the sibling gate `check-invariants.py`; a prose comment naming `KARMA_*`/`soma::` is flagged as a violation |
| F13 | MINOR | `docs/dev/layering.md` (diagram) | The canonical layer diagram omits `linalg` and `field`, which are real, separately-ranked layers in `tools/check-layering.py` |
| F14 | MINOR | `CMakePresets.json:11` | `binaryDir` introduces a third build-directory convention (`build/<preset>`) alongside the `build-tier1`/`build-fuse-s1` convention used throughout the docs and evidence logs |
| F15 | MINOR | `test/backend/moab_mesh_test.cpp:614` | `FUSE_MOAB_TEST_MESH` is referenced but defined nowhere in the build; that section is permanently dead code independent of F2 |
| F16 | MINOR | `cmake/FuseDependencies.cmake:99-101` | `FetchContent_Declare(KokkosKernels ...)` is dead: never `find_package`d or `FetchContent_MakeAvailable`d anywhere |
| F17 | IMPROVEMENT | `check-invariants.py:34` (`scan`) | The `FUSE_HAVE_*`/`#ifdef`-misuse gate scans only `include/` and `src/`, never `test/` or `examples/` — the exact blind spot that let F2 and F15 land uncaught |

## 3. Findings in full

### F1 — BLOCKER — `.github/workflows/ci.yml:57` — CI's `tier0` matrix cannot configure on the `kokkoskernels` leg

`ci.yml`'s `tier0` job matrix is:

```yaml
strategy:
  matrix:
    build: [Debug, Release]
    linalg: [eigen, kokkoskernels]
```

and the configure line passes `-DFUSE_LINALG_BACKEND=${{ matrix.linalg }}`. But
`cmake/FuseDependencies.cmake:62-76` declares:

```cmake
set(FUSE_LINALG_BACKEND "eigen" CACHE STRING "builtin | eigen")
set_property(CACHE FUSE_LINALG_BACKEND PROPERTY STRINGS builtin eigen)
if(NOT FUSE_LINALG_BACKEND MATCHES "^(builtin|eigen)$")
  message(FATAL_ERROR
    "FUSE_LINALG_BACKEND='${FUSE_LINALG_BACKEND}' has no implementation. ...
    A KokkosKernels backend is not available: it has no column-pivoted QR ...")
endif()
```

`kokkoskernels` matches neither `builtin` nor `eigen`, so this branch's
`FATAL_ERROR` fires at configure time, before `cmake --build` is ever invoked.
Two of the four legs `docs/dev/testing.md` calls "tier 0 as four legs: Debug
and Release across both linear-algebra backends" — `Debug`+`kokkoskernels` and
`Release`+`kokkoskernels` — cannot pass on any run, ever, deterministically.
Confirmed by reading the exact `MATCHES` guard and the CI matrix value side by
side; no build was needed to see this, the mismatch is textual and exact.

**Fix.** Change `ci.yml:57` to `linalg: [eigen, builtin]` (the two values
`FuseDependencies.cmake` actually implements). This is the same root cause as
F8 and F16 — a `kokkoskernels` linear-algebra backend was designed and then
abandoned in favour of `builtin`, and the cleanup missed this file, the user
doc (F8), and a dead `FetchContent_Declare` (F16).

**Test to add.** None needed beyond fixing the value — this is a CI
configuration bug, not a code gap. If it recurs, a one-line grep gate
("every `matrix.linalg` value in `ci.yml` must appear in
`FUSE_LINALG_BACKEND`'s `STRINGS` property") would catch a repeat.

### F2 — BLOCKER — `test/backend/moab_mesh_test.cpp:3` — the MOAB test suite silently compiles to one skip, in every build

The file opens:

```cpp
#include <gtest/gtest.h>

#ifdef FUSE_HAVE_MOAB

#include <fuse/backend/moab_backend.hpp>
#include <fuse/backend/serial_backend.hpp>
...
```

and closes (line 754):

```cpp
#else
TEST(MoabMesh, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without FUSE_ENABLE_MOAB";
}
#endif
```

`FUSE_HAVE_MOAB` is a `#cmakedefine01` macro (`include/fuse/base/config.hpp.in`):
it is *always* defined — to `0` or `1` — once `fuse/base/config.hpp` has been
included, and *undefined* before that. This file never includes
`fuse/base/config.hpp`, directly or (before line 3) transitively: the only
thing preceding the `#ifdef` is `<gtest/gtest.h>`. So at the point the
preprocessor evaluates line 3, `FUSE_HAVE_MOAB` has not been defined by
anything in this translation unit, and `#ifdef FUSE_HAVE_MOAB` is false —
**regardless of whether `FUSE_ENABLE_MOAB` was ON or OFF at configure time.**
The entire body between line 3 and the `#else` at line 754 — every one of the
38 ported test cases, every include of `moab_backend.hpp`, everything — is
compiled out unconditionally, and the file reduces to the single
`SkippedWithoutMoab` test.

Confirmed two ways: (1) reading the include order — no macro-defining header
precedes the `#ifdef`; (2) preprocessing the file directly with
`clang++ -E`, once with `-DFUSE_HAVE_MOAB=1` on the command line (the real
tests appear) and once with nothing defined and a stub `gtest/gtest.h`
supplied via `-include` (only `SkippedWithoutMoab` survives) — the second run
is what the actual build does, since no `-D` for this macro exists anywhere
in the CMake files.

This is exactly the anti-pattern this review was asked to hunt for
("`#ifdef` on `#cmakedefine01` macros") and it defeats the specific plan this
project has for validating it: `test/CMakeLists.txt:129-131` registers this
target only `if(FUSE_ENABLE_MOAB)`, and `KNOWN_ISSUES.md`'s `C1` entry (added
in the same HEAD commit) says a host run of tier 2 is what will "establish the
38 MOAB cases." It will not — it will report one passing skip, and a reader
who only looks at the ctest summary line will believe MOAB coverage exists.
Also note `moab_mesh_test.cpp:614`'s nested `#ifdef FUSE_MOAB_TEST_MESH` is
inside this dead branch and is separately never-defined by any build file
(F15) — a second, independent reason that block would never compile even
after F2 is fixed.

**Fix.** Move `#include "fuse/base/config.hpp"` to the top of the file,
before the `#ifdef`, and change `#ifdef FUSE_HAVE_MOAB` to
`#if FUSE_HAVE_MOAB` (matching the pattern already used correctly in
`src/backend/intx_weights.cpp:16` and the header comment at the top of
`src/backend/moab_backend.cpp:1-4`, which documents this exact class of bug
for a different macro).

**Test to add.** A `check-invariants.py`-style assertion is the right gate
(see F17): extend its scan to `test/`, which would have caught this. As a
standing regression test, add a CMake-time assertion in
`test/CMakeLists.txt` that fails configure if the MOAB test binary, once
built, reports fewer than N gtest cases via `--gtest_list_tests` — a
"registered but reduces to a stub" bug like this one is otherwise invisible
to `ctest` (a skip is not a failure).

### F3 — BLOCKER — `test/CMakeLists.txt:97-101` — `schedule.buffers` and `schedule.slab` are never built

`test/schedule/` contains nine `*_test.cpp` files. `test/CMakeLists.txt`'s
schedule section registers exactly seven of them:

```cmake
foreach(t acyclicity acyclicity_sweep expand plan lockstep epoch)
  fuse_add_gtest(schedule ${t})
endforeach()
fuse_add_gtest(schedule clock THROWS)
```

`test/schedule/buffers_test.cpp` (361 lines, 17 `TEST(...)` cases, including
several `EXPECT_THROW`) and `test/schedule/slab_test.cpp` (177 lines, 16
`TEST(...)` cases) are never named anywhere in `test/CMakeLists.txt`. Both
correspond to real, linked library sources — `src/CMakeLists.txt` builds
`schedule/buffers.cpp` and `schedule/slab.cpp` into `fuse-coupler-schedule` —
so this is not a case of dead test files for retired code; the code they test
is live and shipping.

Confirmed three ways: (1) a systematic cross-reference of every
`test/<layer>/*_test.cpp` file against every registration pattern in
`test/CMakeLists.txt` (`fuse_add_gtest`, the `foreach`+`${t}` forms, and the
`add_executable(fuse_parallel_...)` form) found exactly these two files
unmatched, and nothing else in the whole suite; (2) `git log` shows both
files have existed since at least commit `9975cb3`, well before HEAD,
so this is not a same-commit oversight about to be fixed; (3) the build
directory already on disk, `build-tier1/test/CTestTestfile.cmake`, lists
`schedule.clock`, `schedule.acyclicity`, `schedule.acyclicity_sweep`,
`schedule.expand`, `schedule.plan`, `schedule.lockstep`, `schedule.epoch` —
seven tests, no `buffers`, no `slab` — independent, on-disk proof that these
33 test cases have never executed in this repository. `build-evidence/mutations.toml`
has no entry with `test = "schedule.buffers"` or `test = "schedule.slab"`
either, so the mutation-evidence gate (I9) has never touched
`schedule/buffers.cpp` or `schedule/slab.cpp` — these two kernels have, in
effect, zero verified coverage today despite the appearance of a test suite.

**Fix.** Add `buffers` and `slab` to the `foreach` list at
`test/CMakeLists.txt:98` (neither is `THROWS`-only — `buffers_test.cpp` does
use `EXPECT_THROW` in places, so check per-case whether the target needs the
`THROWS` label the way `clock` has it, following the same audit the file's
own header comment describes doing for the nine `ir`/no-exceptions targets).

**Test to add.** Once registered, run the mutation gate against both files
(`tools/mutate.py --area schedule`) and add at least one named mutation per
public kernel in `buffers.cpp`/`slab.cpp` to `build-evidence/mutations.toml`,
per I9 — currently there are none.

### F4 — MAJOR — `tools/sync-upstream.py:39-46` — the lock parser cannot read `UPSTREAMS.lock`'s own array format

```python
lst = re.match(r"^\s*(diverged|superseded)\s*=\s*\[(.*)\]\s*$", line)
```

is matched line-by-line against `text.splitlines()`. `UPSTREAMS.lock` writes
`superseded` as a multi-line TOML-like array:

```
superseded = [
  "test/verification/mms_transfer_test.cpp",
  ...
  "test/golden/",
]
```

The opening line `superseded = [` has no closing `]` on it, so the regex
never matches on any line of this array; nothing is ever appended to
`out[cur]["superseded"]` beyond the empty list the parser initialises when it
sees the `[soma]`/`[karma]` header. Confirmed by importing `sync-upstream.py`
directly and calling `lock()` against the real file: it returns
`superseded=[]` for both parents, even though the file lists 20 (`soma`) and
11 (`karma`) superseded paths.

Because `classify()` (line 51-58) decides "superseded" only when every
touched path matches this (empty) list, the tool can never emit the
`superseded` verdict `porting.md` documents: *"a backport touching those
paths is classified `superseded` by `tools/sync-upstream.py` and ignored."*
Today it cannot be — a backport that touches a path FUSE has already
replaced and deleted the equivalent of would be classified `absorb` (since
`diverged` is also empty for the same reason whenever it is non-trivial) and
the tool would print nothing warning the human that the patch targets code
that no longer needs it. This is silent, not a crash, which is the worse
failure mode for a tool whose entire job is to flag exactly this case.

**Fix.** Replace the single-line regex with a small state machine that
accumulates lines from the `key = [` opening through the matching `]`,
splitting on `,` and stripping quotes — or, simpler, parse the file with the
real `tomllib` the same way `tools/mutate.py` already does for
`mutations.toml`, since `UPSTREAMS.lock`'s array syntax is valid TOML.

**Test to add.** A unit test for `sync-upstream.py::lock()` against a fixture
`UPSTREAMS.lock`-like file with a multi-line array, asserting the returned
list is non-empty and matches the fixture's paths — there is currently no
test at all for this script.

### F5 — MAJOR — `build-evidence/mutations.toml:455` — a mutation anchor no longer matches the code it claims to test

```toml
[[mutation]]
id = "epoch-schedule-drops-the-answer"
...
file = "include/fuse/runtime/epoch_schedule.hpp"
old = "    for (int k = 1; k < participants; ++k) s.result_send_to.push_back(k);"
new = "    for (int k = 1; k < participants; ++k) (void)k;"
target = "fuse_runtime_epoch_schedule_test"
test = "runtime.epoch_schedule"
```

`build-evidence/runtime-mutations.md` records this mutation as **KILLED** on
2026-09-16. `include/fuse/runtime/epoch_schedule.hpp` today implements a
binomial-tree reduce/broadcast (`for (int stride = 1; stride < participants;
stride *= 2) ...`), not the flat per-participant loop the anchor names — there
is no `result_send_to` field any more (it was replaced by
`result_send_count`, a count rather than a list, per the header's own
comment at line 44-47). `text.count(old)` against the current file is `0`.
Confirmed by running exactly the check `tools/mutate.py` performs
(`original.count(m["old"]) != 1`) against all 70 entries in
`mutations.toml`: this is the only one that fails, and it fails with `0`
occurrences, not a duplicate — the anchor text is simply gone.

The practical effect: `tools/mutate.py --area runtime` would report this
entry `INVALID` today (anchor occurs 0 times), which the tool treats as "not
killed" for its exit code — directly contradicting the committed log's
claim of `2 killed / 2 total` for the `runtime` area. Since the code was
substantially rewritten (flat loop → binomial tree) without updating the
evidence, there is currently **no verified mutation evidence** that the new
reduce/broadcast implementation would actually fail `runtime.epoch_schedule`
if participant 0 dropped the answer — precisely the rank-agreement defect
(I2) this mutation exists to guard against.

**Fix.** Rewrite the mutation to target the current implementation, e.g.
mutate `s.result_send_count = participants - 1;` (line 74) to
`s.result_send_count = 0;` (participant 0 sends to nobody — the same
"drops the answer" defect against the new code shape), then re-run
`tools/mutate.py --area runtime --write-log` to regenerate
`runtime-mutations.md` with a real, current verdict.

**Test to add.** None beyond the corrected mutation; the existing
`runtime.epoch_schedule` unit test is the right oracle, it just is not being
exercised by a mutation that still compiles against current source.

### F6 — MAJOR — `UPSTREAMS.lock:90-95` — MOAB's superseded paths were never added to the machine-readable list

`porting.md`'s per-area procedure, step 6: *"Record supersession. Add the
upstream paths to `superseded` in `UPSTREAMS.lock`. From that commit on, a
backport touching those paths is classified `superseded`..."* The S5.2 MOAB
port (`UPSTREAMS.lock:90-95`) instead records the four ported paths as a bare
comment trailing the `[karma]` section:

```
# Superseded 2026-09-18 by the MOAB backend port (S5.2):
#   soma/include/soma/mesh/moab_backend.hpp -> include/fuse/backend/moab_backend.hpp
#   soma/src/mesh/moab_backend.cpp          -> src/backend/moab_backend.cpp
#   soma/src/remap/tempest_weights.cpp      -> src/backend/tempest_weights.cpp
#   soma/test/unit/moab_mesh_test.cpp       -> test/backend/moab_mesh_test.cpp
```

These four lines are `#`-comments, not `superseded = [...]` array entries;
`[soma].superseded` (defined at `UPSTREAMS.lock:14-31`) still lists only the
20 paths from the S1/S2 ports and does not include
`include/soma/mesh/moab_backend.hpp`, `src/mesh/moab_backend.cpp`,
`src/remap/tempest_weights.cpp`, or `test/unit/moab_mesh_test.cpp`. Confirmed
by grepping the whole file for `moab`/`tempest`: the only hits are these five
comment lines. This compounds F4 — even a fixed parser would find nothing
here to read — but it is a distinct defect: the structured record itself was
never updated, so the omission is not merely a tooling gap, it is a process
step (`porting.md` step 6) that was skipped for this port.

**Fix.** Move these four paths into `[soma].superseded`'s array (formatted
the same way as the existing entries, each on its own line with a trailing
comment noting the S5.2 port), and keep the human-readable "Superseded
2026-09-18 by..." note as prose documentation above or below the array
entry, not as a substitute for it.

**Test to add.** Covered by the fixture test proposed for F4 — once
`sync-upstream.py` has a test asserting it reads real entries, a second
fixture case (a commit touching `soma/src/mesh/moab_backend.cpp` after these
four paths are properly added) would assert `classify()` returns
`"superseded"`.

### F7 — MAJOR — `KNOWN_ISSUES.md:165-176` — B1 is stale and contradicted by C1 in the same commit

`KNOWN_ISSUES.md`'s B1, unchanged text at HEAD:

> ### B1. Tier 2 does not build: the MOAB backend is unported
>
> `-DFUSE_ENABLE_MOAB=ON` now configures and resolves MOAB 5.6.0 correctly,
> then fails to compile:
> ```
> src/backend/intx_weights.cpp:49:10: fatal error: 'fuse/backend/moab_backend.hpp' file not found
> ```

This is false at HEAD: `include/fuse/backend/moab_backend.hpp` (15KB) and
`src/backend/moab_backend.cpp` (93KB) both exist and are wired into
`fuse-coupler-backend` (`src/CMakeLists.txt:172-183`, `if(FUSE_ENABLE_MOAB)
target_sources(... backend/moab_backend.cpp) ... target_link_libraries(...
PUBLIC fuse::moab)`). The quoted compiler error cannot reproduce — the file
it claims is missing is present. Fifty-two lines below, in the same file and
(per `git log`) the same commit `bdf4346` ("Record the decomposition rule and
the unrun MOAB suite"), **C1** correctly says: *"`S5.2` landed 2026-09-18:
tier 2 configures, compiles and links for the first time..."* — directly
contradicting B1's premise two sections up. `UPSTREAMS.lock`'s S5.2
supersession note (F6) independently confirms the same timeline: the MOAB
backend was ported on 2026-09-18, before this HEAD.

`KNOWN_ISSUES.md`'s own convention marks closed items with a struck-through
heading and a "CLOSED <date>" suffix (see A2, A3, B1a). B1 was never given
that treatment even though its content is now false, and its neighbour C1 —
added in the same edit — is effectively B1's replacement/successor entry but
was filed as a new "C" item rather than closing B1 and continuing the "B"
sequence, which is also why C1 sits between B1a and B2, out of the
otherwise-consistent A/B/C ordering the file's header explains ("split by who
has to act").

**Fix.** Strike through B1's heading, mark it "CLOSED 2026-09-18 — superseded
by C1", and leave a one-line pointer to C1 (matching the treatment already
given to A2/A3/B1a) rather than leaving a live, reproducible-looking compiler
error in the file for a build that now succeeds.

### F8 — MAJOR — `docs/user/build.md:9,29-37` — the build doc describes a backend that does not exist and capabilities that are not built

Three separate, confirmed mismatches in one page:

1. **Line 9**, the tier table: *"**0** (default) | C++20 compiler, Kokkos,
   and one of KokkosKernels or Eigen3 |"*. `cmake/FuseDependencies.cmake:62-76`
   accepts only `builtin` or `eigen` for `FUSE_LINALG_BACKEND`, and its own
   `FATAL_ERROR` text says explicitly: *"A KokkosKernels backend is not
   available: it has no column-pivoted QR with rank truncation in the form
   the quasi-Newton solve needs."* There is no path by which a user gets a
   KokkosKernels linalg backend; the doc's own worked example two lines
   below (`-DFUSE_LINALG_BACKEND=eigen`) doesn't even match the table cell
   that names it.

2. **Lines 29-37**, "Choosing the linear-algebra backend", documents exactly
   two backends, `eigen` and `kokkoskernels` ("Device-capable, and the right
   choice when you build with a GPU backend. Must match your Kokkos
   version."). Neither `FuseDependencies.cmake` nor `src/CMakeLists.txt`
   implements a KokkosKernels linalg option at all — the only
   `FetchContent_Declare(KokkosKernels ...)` in the tree (F16) is never
   consumed. The section never mentions `builtin`, which is the actual
   zero-dependency default the code documents internally as *"also the
   oracle the other is checked against, because the recorded iteration
   counts were produced with it"* (`FuseDependencies.cmake:55-58`) — the doc
   omits the one backend a user would reach for on a machine with no Eigen,
   and advertises one that will `FATAL_ERROR`.

3. **Line 9**, same table cell: Tier 0 "Unlocks... both C ABIs, the Fortran
   shim, YAML input." `include/fuse/abi/` contains only a `.clang-tidy` file
   — no headers, no ABI implementation of any kind exists at HEAD. YAML input
   is real (`src/model/yaml_io.cpp`, tested), but there is no C ABI and no
   Fortran shim: `FUSE_ENABLE_FORTRAN` is declared as a CMake option
   (`FuseDependencies.cmake:51`) and never consumed anywhere — turning it ON
   does nothing. `docs/dev/roadmap.md`'s own table (read as part of this
   review) correctly lists both ABIs and the Fortran shim as **S6**
   deliverables, not yet reached ("in progress: S5"); this page contradicts
   that page.

**Fix.** Rewrite the tier-0 row to say "`builtin` or Eigen3" and drop the
false ABI/Fortran claims (or mark them "planned, S6" to match `roadmap.md`);
rewrite the backend section to document `builtin` and `eigen`, dropping
`kokkoskernels` (or, if a KokkosKernels backend is still wanted, implement it
in `FuseDependencies.cmake` first and then document it — not the other way
round).

**Test to add.** The `docs` CI job (`sphinx-build -W --keep-going`) already
catches Sphinx warnings but has no way to catch a documentation claim that
disagrees with the CMake source of truth; none is proposed here beyond
fixing the prose, since this is a content defect rather than a mechanically
checkable one.

### F9 — MAJOR — no `install()`/`export()` in the CMake tree; `find_package(fuse-coupler)` cannot work

`conventions.md` states: *"Namespace `fuse::`. CMake package `fuse-coupler`.
... Libraries `libfuse-coupler-<layer>`."* Every one of the nine
`fuse-coupler-<layer>` targets is built with
`$<BUILD_INTERFACE:...>`/`$<INSTALL_INTERFACE:include>` generator
expressions on its include directories (`src/CMakeLists.txt:25-28` for
`base`, and the same pattern is implied — but not completed — for every other
layer), which is the setup step for an installable, exportable target. But
grepping the entire non-upstream CMake tree
(`CMakeLists.txt`, `src/CMakeLists.txt`, `test/CMakeLists.txt`,
`examples/CMakeLists.txt`, `cmake/FuseDependencies.cmake`) for `install(` or
`export(` returns zero matches. No target has an `EXPORT` set, no
`install(TARGETS ...)`, no generated `fuse-coupler-config.cmake`, no
`write_basic_package_version_file`. A downstream project cannot
`find_package(fuse-coupler)` today — the `$<INSTALL_INTERFACE:include>`
already written into `src/CMakeLists.txt:27` is dead code, since nothing
ever installs the headers to make that interface real.

This may simply reflect where the project is in its roadmap (no stage's exit
criterion mentions packaging), but `conventions.md` states the package name
as a present-tense fact rather than a planned one, and a reader has no way to
tell from the docs that this is not yet wired up.

**Fix.** Either add the missing `install(TARGETS ... EXPORT fuse-coupler-targets)`
/ `install(EXPORT ...)` / `configure_package_config_file` sequence (the
`BUILD_INTERFACE`/`INSTALL_INTERFACE` split is already there, so the
remaining work is the `install()` calls and one `Config.cmake.in`), or amend
`conventions.md` to say the CMake package is the *planned* name for when
install support lands, matching the calibrated language the rest of that
document uses elsewhere ("Enforced by...").

**Test to add.** A `cmake --install` smoke test in a future CI leg
(`install` job: configure, build, `cmake --install`, then configure a trivial
second project with `find_package(fuse-coupler)` against the install
prefix) — there is nothing to add until the install rules exist.

### F10 — MINOR — `src/backend/intx_weights.cpp:36` — `defined()` on a `#cmakedefine01` macro, masked only by the tier ladder

```cpp
#if defined(MOAB_HAVE_MPI) && defined(FUSE_HAVE_MPI)
#define FUSE_REMAP_PARALLEL 1
```

`FUSE_HAVE_MPI` is a `#cmakedefine01` macro — always defined to `0` or `1`
once `config.hpp` is included (which this file does, at line 18, before this
use). `defined(FUSE_HAVE_MPI)` is therefore always true once that include has
run, regardless of whether MPI is actually enabled; the correct test is
`#if FUSE_HAVE_MPI`, exactly as the sibling file `src/backend/moab_backend.cpp:24`
does it (`#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI`, no `defined()` on the
FUSE macro). Today this is harmless in practice: `FuseDependencies.cmake:79-84`
enforces `FUSE_ENABLE_MOAB` requires `FUSE_ENABLE_MPI`, and this file is only
ever compiled as part of the MOAB backend (`src/CMakeLists.txt`, inside
`if(FUSE_ENABLE_MOAB)`), so `FUSE_HAVE_MPI` is always `1` whenever this line
runs. But the guard's *correctness* is accidental — it depends on an
invariant enforced fifty files away — and if that tier-ladder rule is ever
relaxed, this line's behaviour would silently stop matching its neighbour's.

**Fix.** Drop `defined()`: `#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI`,
matching `moab_backend.cpp:24` exactly.

**Test to add.** None needed; `check-invariants.py`'s `FEATURE_MACRO` regex
(`\bFUSE_(HAVE_[A-Z0-9_]+|ENABLE_EXCEPTIONS)\b`) already matches
`FUSE_HAVE_MPI` here and confirms `config.hpp` is included before this line —
it just does not additionally check *how* the macro is tested
(`#if` vs. `#ifdef`/`defined()`) once presence is established. Extending the
gate to flag `defined(FUSE_HAVE_*)` and `#ifdef FUSE_HAVE_*` specifically
(as opposed to merely checking include order) would catch both this and F2
mechanically; see F17.

### F11 — MINOR — `CMakeLists.txt:65,68` — stale status banner

```cmake
message(STATUS "fuse ${PROJECT_VERSION} — stage S1")
...
message(STATUS "  fuse:: layers   : base")
```

The file's own leading comment says *"Each stage from S1 adds its own layer
directory here"* — but the banner was last touched at commit `3f17a33`
("Port the two parent examples and run them on their physics gates", S4) and
was not updated then, or through S5's `LinearOperator`/MOAB-backend landings
since. At HEAD the tree builds nine layers (`base`, `model`, `ir`,
`schedule`, `linalg`, `num`, `transport`, `backend`, `runtime`, per
`src/CMakeLists.txt`) and the project is mid-S5, not S1. Every configure run
prints a banner that understates the build by four stages and eight layers.

**Fix.** Either compute the layer list and stage from data already in the
build (e.g. a `FUSE_LAYERS` list set once in `src/CMakeLists.txt` and
referenced here) so it cannot drift again, or at minimum update the literal
string at each stage boundary as the comment already promises to do.

### F12 — MINOR — `tools/check-layering.py:76-113` — legacy-name checks are not comment-aware

`check_legacy()`'s `LEGACY_NAMESPACE`, `LEGACY_TYPE_PREFIX` and `LEGACY_MACRO`
regexes are applied to raw source lines with no comment stripping. Confirmed
with a synthetic header under `/tmp` (not touching the real tree):

```cpp
// This header replaces the old KARMA_ASSERT macro and the soma:: namespace
// convention; neither survives the port (see docs/dev/porting.md).
```

fed to `check_legacy()` reports two violations — `legacy namespace
'soma'/'KARMA'` and `legacy macro prefix` — on a line that is pure prose about
the rename, not a use of either. `tools/check-invariants.py`'s sibling
`FEATURE_MACRO`/`PRINTF` checks deliberately blank `//` comments first, with
the stated rationale (`check-invariants.py:97-101`) that *"a build-feature
macro NAMED IN PROSE is not a read of it... without this the gate fires on
any file that documents the macro, and the pressure becomes to reword the
documentation rather than fix the code."* The identical rationale applies
here and was not carried over to the sibling gate. This is currently latent —
no source file's comments trip it today, since `check-layering.py` exits 0 —
but it is exactly the kind of pressure `docs/dev/porting.md`'s own prose
would create for a header's rename changelog comment, and the "gates' blind
spots" this review was asked to characterize.

**Fix.** Reuse `check-invariants.py`'s `COMMENT_LINE` blanking (or factor it
into a shared helper both scripts import) before applying `LEGACY_NAMESPACE`,
`LEGACY_TYPE_PREFIX` and `LEGACY_MACRO` in `check_legacy()`.

### F13 — MINOR — `docs/dev/layering.md` diagram omits `linalg` and `field`

The canonical layer diagram (the fenced block near the top of the page) lists
nine rows — `interface`, `runtime`, `transport`, `backend`, `num`,
`schedule`, `ir`, `model`, `base` — and does not mention `linalg` or `field`
anywhere as their own row. Both are nonetheless real, separately-ranked
entries in `tools/check-layering.py`'s `LAYERS` list (`base`, `linalg`,
`model`, `ir`, `schedule`, `num`, `field`, `backend`, `transport`, `runtime`,
`abi` — eleven entries) and real target/directory boundaries
(`fuse::linalg`, `include/fuse/field`). A reader relying on the doc's diagram
alone — which is the stated enforcement mechanism, quoting the gate script
right below it — cannot tell where `linalg` or `field` sit in the dependency
order, or what third-party headers they may carry, without reading the gate
script's own comments (which do explain the positions, at
`check-layering.py:16-29`, but that explanation lives in the tool, not the
doc it backs).

**Fix.** Add `linalg` and `field` to the diagram, at minimum as a footnote
pointing at the comment already present in `check-layering.py:16-29`
explaining their positions (`linalg` beside `num`, not under it; `field`
included only by `backend` and `runtime`).

### F14 — MINOR — `CMakePresets.json:11` — a third, inconsistent build-directory convention

```json
"binaryDir": "${sourceDir}/build/${presetName}",
```

puts every preset's build tree under `build/<name>` (e.g. `build/tier1`).
Every other place in the tree that names a build directory uses a hyphenated,
top-level convention instead: `docs/dev/testing.md`'s host-verification
recipes use `build-tier1`/`build-tier2`, `KNOWN_ISSUES.md`'s C1 uses
`build-tier2`, and the mutation-evidence logs record runs against
`build-fuse-s1`. `.gitignore` already covers both shapes (`build/` and
`build-*/`), so nothing breaks, but a host operator following
`testing.md`'s literal `cmake -S . -B build-tier1 ...` command and later
running `cmake --build --preset tier1` is operating on two different,
independently-configured directories (`build-tier1/` vs. `build/tier1/`) that
happen to share a name only by coincidence of the tier number — the actual
build in `build-tier1/` on disk right now (confirmed present at the top of
this review) predates the preset file and was made by the manual recipe, not
the preset.

**Fix.** Either point `CMakePresets.json` at the hyphenated convention
(`"binaryDir": "${sourceDir}/build-${presetName}"`) to match every existing
document and log, or update `docs/dev/testing.md`/`KNOWN_ISSUES.md` to adopt
the preset's `build/<name>` layout once the presets are the recommended
entry point — pick one convention, not three.

### F15 — MINOR — `test/backend/moab_mesh_test.cpp:614` — `FUSE_MOAB_TEST_MESH` is never defined

```cpp
#ifdef FUSE_MOAB_TEST_MESH
```

guards the real-mesh-file quadrature tests (lines 614-752). No
`target_compile_definitions` anywhere in the tree defines
`FUSE_MOAB_TEST_MESH` — confirmed by grepping the whole repository (excluding
`build*/`) for the string; the only hits are this file's own `#ifdef`/uses.
This section is dead code independent of F2: even after F2's fix
(`#include config.hpp` + `#if FUSE_HAVE_MOAB`) makes the rest of the file
live, this nested block still never compiles, because nothing supplies the
macro that would name a real MOAB mesh fixture file.

**Fix.** Either add a CMake option/variable that points at a real cubed-sphere
mesh MOAB ships (the comment above line 614 says "the cubed sphere MOAB ships
in `share/examples`") and `target_compile_definitions` it into
`fuse_backend_moab_mesh_test`, or delete the dead block and its comment if no
such fixture path is actually planned.

### F16 — MINOR — `cmake/FuseDependencies.cmake:99-101` — dead `FetchContent_Declare(KokkosKernels ...)`

```cmake
FetchContent_Declare(KokkosKernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git GIT_TAG ${FUSE_KOKKOSKERNELS_TAG} GIT_SHALLOW ON)
```

is declared alongside `fmt`, `fmtlog`, `googletest`, `Kokkos`, `yaml-cpp`,
`r3d` and `Eigen3` — but unlike every one of those, it is never named in a
`fuse_require(...)` call, a `fuse_resolve_*` macro, or a bare
`FetchContent_MakeAvailable(KokkosKernels)` anywhere in
`fuse_resolve_dependencies()` or elsewhere in the file. `FUSE_KOKKOSKERNELS_TAG`
(line 32) is likewise set and never read again outside this one declaration.
This is the same abandoned-design remnant as F1 and F8: a KokkosKernels
linear-algebra backend was planned, the tag was pinned, the fetch was
declared — and then the project went with `builtin` instead, leaving this
declaration (and the doc, and the CI matrix value) uncleaned.

**Fix.** Delete the `FetchContent_Declare(KokkosKernels ...)` block and the
now-unused `FUSE_KOKKOSKERNELS_TAG` variable, unless a KokkosKernels backend
is still intended — in which case wire it into `fuse_resolve_dependencies()`
and `src/CMakeLists.txt`'s linalg target and update F1/F8 to match, rather
than leaving a fetch declaration nothing resolves.

### F17 — IMPROVEMENT — `tools/check-invariants.py:34` — the feature-macro gate never scans `test/` or `examples/`

```python
for area in ("include", "src"):
```

is the entire directory scope of `scan()`, which is where the
`#ifdef`/`defined()`-vs-`#if` misuse check on `FUSE_HAVE_*` macros lives. It
never runs against `test/` or `examples/`. This is precisely why F2 — a
textbook instance of the exact bug this checker exists to catch — sits
undetected in `test/backend/moab_mesh_test.cpp`: the file is invisible to the
one tool built to find it. By contrast, `check-layering.py`'s legacy-name
scan (`check_legacy`, F12) already covers `test/` and `examples/` explicitly,
with its own header comment explaining why it was extended there in
2026-09-16/17 after a stale upstream include was found hiding in exactly
those directories — the same lesson has not yet been applied to
`check-invariants.py`.

**Fix.** Extend `scan()`'s area tuple to `("include", "src", "test",
"examples")`, matching `check_legacy()`'s scope and its stated rationale.
This single change would have caught F2 as a CI failure the moment it landed.

## 4. Not found — coverage boundaries

- **Both static gates pass clean at HEAD.** `python3 tools/check-layering.py`
  and `python3 tools/check-invariants.py` were run directly against the
  working tree and both report `0 violation(s)`. Every finding above that
  the gates are supposed to catch (F2, F10, F12) is a case the gates'
  *current, as-written* scope does not reach — not a case they reach and get
  wrong.
- **No layering violation found.** `include/fuse/**`'s actual `#include
  "fuse/<layer>/..."` graph was checked (by the gate, and spot-read for
  `linalg`/`field`/`transport`/`backend`) against `LAYERS`'s rank order and
  found consistent; the link graph in `src/CMakeLists.txt` matches it
  target-for-target (`base ← model ← ir ← schedule ← {linalg beside num} ←
  num ← {backend, transport} ← runtime`).
  No `Eigen/`, `mpi.h`, `moab/`, `hdf5.h`, `KokkosSparse`/`KokkosBlas`/`KokkosBatched`
  header was found in a public FUSE header outside its own backend
  directory.
- **No un-renamed legacy code found.** No `soma::`/`KARMA::` namespace, no
  `KARMA*` type prefix, no `karma_*`/`soma_*` file prefix, and no `.C`
  extension exists anywhere under `include/fuse/`, `src/`, `test/`, or
  `examples/` (the areas `check-layering.py` scans for I11).
  `upstream/soma` and `upstream/karma` themselves were not re-verified
  byte-identical against their tagged parents (`tools/verify-import.py
  --parents` needs the parent repositories checked out locally, which is
  outside this review's file scope and this sandbox's network access) — the
  manifest-only check (`tools/verify-import.py`, no `--parents`) was read but
  not re-run, since it requires nothing this review changed.
- **No other stale mutation anchor found.** All 70 entries in
  `build-evidence/mutations.toml` were checked programmatically
  (`text.count(old) == 1`) against the current tree; F5 is the only failure.
  The mutation-log-vs-toml accounting (16+13+13+15+9+2+2 = 70 rows across the
  seven `*-mutations.md` files, matching the toml's 70 entries exactly) has
  no orphaned or missing log.
- **No other unregistered test file found.** Every `test/<layer>/*_test.cpp`
  file was cross-referenced against `test/CMakeLists.txt`'s registration
  calls; F3's two files are the only ones unmatched, across all twelve test
  layers.
- **No `FUSE_ENABLE_*`/`FUSE_HAVE_*` misuse found outside F2/F10.** Every
  `#ifdef FUSE_`/`#ifndef FUSE_` and every `FUSE_ENABLE_*`/`FUSE_LINALG_EIGEN`
  read in `include/`, `src/`, `test/`, `examples/` was enumerated by grep;
  the only two with the presence-vs-value defect are F2 and F10.
  `include/fuse/base/error.hpp:16`'s `#ifndef FUSE_ENABLE_EXCEPTIONS` is
  the one deliberate exception the checker itself special-cases (never
  flagged, and correctly so — it is inside the header that *is*
  `error.hpp`'s own fallback-definition guard, not a feature read).
- **No CI-leg/preset mismatch found beyond F1 and F14.** Every
  `configurePreset`/`buildPreset`/`testPreset`/`workflowPreset` in
  `CMakePresets.json` was checked for internal consistency (tier
  prerequisites, `-LE throws` filtering, `jobs` counts) and none besides the
  build-directory naming (F14) diverges from the ladder rules in
  `FuseDependencies.cmake`; `ci.yml`'s five jobs were each checked against
  the tiers/legs `testing.md` and `roadmap.md` describe as "running today,"
  and F1 is the only leg that cannot pass.
- **Examples were read but no defect found in them.** `mms_pair/main.cpp`
  and `dn_heat/main.cpp` use `std::cout`/`std::printf` for their `--dump`
  output; this is outside the scope of `conventions.md`'s fmt-only rule
  (which the gate itself scopes to `include/`/`src/`, i.e. library code, not
  examples) and the output is human-facing diagnostic text, not a
  byte-compared golden artifact, so the locale-pinning rule does not apply
  either. No feature-macro read, no legacy name, and no MOAB/MPI guard exists
  in either example or their component headers.
- **`FUSE_BUILD_UPSTREAM_SOMA`/`_KARMA` default mismatch (root `CMakeLists.txt`
  ON/OFF vs. `CMakePresets.json`'s `_machine` OFF/OFF) was checked and is not
  a defect** — it is documented, intentional (`CMakeLists.txt`'s own leading
  comment explains a bare `cmake -S . -B build` is meant to also build the
  frozen parents for the S0 exit criterion, while every preset and CI leg
  deliberately overrides both OFF for a fuse-only build).

## 5. Suggested new tests / mutations

1. Register `test/schedule/buffers_test.cpp` and `test/schedule/slab_test.cpp`
   in `test/CMakeLists.txt` (F3) and add at least one named mutation each to
   `build-evidence/mutations.toml` for `schedule/buffers.cpp` and
   `schedule/slab.cpp` — currently zero mutation evidence exists for either
   kernel.
2. Fix and re-run the `epoch-schedule-drops-the-answer` mutation (F5) against
   the current binomial-tree implementation and commit a fresh
   `runtime-mutations.md`.
3. A CMake-configure-time or CI smoke check that a built GoogleTest binary
   reports at least N cases via `--gtest_list_tests`, to catch a "registered
   target that silently reduces to a stub" bug the way F2 did — ctest's
   pass/fail/skip accounting cannot see this class of defect by construction.
4. Extend `check-invariants.py`'s `scan()` to `test/` and `examples/` (F17) —
   this one change is the highest-leverage fix in this review, since it would
   have caught F2 mechanically, on every future commit, for free.
5. A unit test for `tools/sync-upstream.py::lock()` against a fixture file
   using the real multi-line array syntax (F4), plus a `classify()` fixture
   case that exercises the `superseded` branch end-to-end once F4 and F6 are
   both fixed.
6. A repository-wide grep-based gate ("every CI matrix value referencing a
   CMake cache variable with a `STRINGS` property must be one of that
   property's values") would have caught F1 mechanically and would catch a
   repeat.
