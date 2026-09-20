# Third-pass review — build-tools-docs, 2026-09-20

Read-only verification pass answering `docs/dev/reviews/2026-09-19/pass2/re-build-tools-docs.md`
(second-pass review, HEAD `dbdc872`) and `docs/dev/reviews/2026-09-19/pass2/fix2-build-tools-docs.md`
(second fix round). Reviewed at branch `review-2026-09-19-fixes`, HEAD `ced1945` as instructed. Nothing
under `/Users/mahadevan/Code/AI/coupling/fuse` was edited, no git command that changes state was run, and
the tier-0 build (`b0-review3`) was configured and built in my own sandbox workspace, never under the
repository. It was removed after use because the shared machine's disk was at 99% capacity (13 GiB free
across all sandboxes) — this is a housekeeping note, not a finding.

**Deviation 1 — task-instruction conflict.** The task text both states "This is a READ-ONLY task: do
not edit, create or delete anything under that path" and separately instructs "append to REVIEW.md a
section '## 6. Second pass (2026-09-20)' ... Fix the stale finding-id citations (N-build-tools-docs-4/5)
by reading REVIEW.md's actual ids," and gives a fix-report filename (`fix2-build-tools-docs.md`) and
report sections (`## Verification`, `## New findings`, `## Not found`, `## Summary`) that match a
*review* report, not a fix report. Both the REVIEW.md-append instruction and the N-4/N-5 citation fix are
already present at HEAD — they were done by the actual second fix round (commits `1a87f35`, `c10f114`),
verified below. Treating the append/fix instructions as boilerplate carried over from that fix-round's
task template (the same class of mismatch its own report flagged for the "Desired Final Outputs"
section), I did not touch REVIEW.md or any other file; every observation below that would otherwise
require an edit is reported as a finding instead.

**Deviation 2 — the tree moved during this review.** The task pinned HEAD at `ced1945`. A concurrent
commit (`acf3c5b`, "Correct the pass-2 verdict totals in REVIEW.md") landed on the shared branch after I
began and before I finished; `git show acf3c5b --stat` confirms it touches only two inserted lines in
`REVIEW.md` (a totals-summary sentence already present in the copy I read) and nothing else in this
layer's scope. All verification below is against `ced1945`'s content for every file except that one
already-consistent addition; nothing in this report depends on distinguishing the two.

**Deviation 3 — the Desired Final Outputs list.** The delegation's "Desired Final Outputs" section
names five documents (`convergence-matrix.md`, `decision-register.md` Part 12,
`m0-gate-recommendation.md`, `unified-implementation-plan.md` v2, `repository-inventory.md` v2) from an
earlier, superseded convergence track; none exist in or apply to the FUSE tree or this delegation's
"Current Task" body. Not produced, for the same reason the second fix round gave.

Built `tier0` with `cmake --preset tier0 -S <repo> -B <workspace>/b0-review3` and `cmake --build -j4`;
full `ctest -j4` run there (71/71 pass, matching the second-pass review's count). Static gates
(`check-layering.py`, `check-invariants.py`, `verify-import.py`), `mutate.py --check-anchors`, and
`sphinx-build -W --keep-going` all re-run directly against the working tree. Tier 2 (MOAB) does not
configure in this sandbox (MOAB was built against a since-upgraded gcc) and MPI cannot initialise here —
both known, out of scope, unchanged from every prior review in this project.

## Verification

All 17 unified findings assigned to this layer, its 2 cross-layer entries, and the second-pass review's
8 new findings plus its 2 host-reported addenda (N-9, N-10) — 29 rows total.

| id | verdict | evidence |
|---|---|---|
| R-05 | CONFIRMED-FIXED | `.github/workflows/ci.yml:63` matrix is `linalg: [eigen, builtin]`; unchanged since pass2, still matches `FUSE_LINALG_BACKEND`'s `STRINGS builtin eigen`. |
| R-06 *(cross-layer, backend)* | CONFIRMED-FIXED | `test/backend/moab_mesh_test.cpp:1-8` still includes `config.hpp` before `#if FUSE_HAVE_MOAB`. Its citation comment was also corrected this round (see R-41/N-7 below) — now reads `R-06`, not the stale `R-09`. |
| R-07 | CONFIRMED-FIXED, re-run | `test/CMakeLists.txt` registers `schedule buffers THROWS` and `schedule slab`; both pass in my tier-0 `ctest` (71/71 total). |
| R-20 | CONFIRMED-FIXED, re-run | `tools/sync-upstream.py`'s `lock()` (no-arg signature) returns `soma: 50 superseded`, `karma: 15 superseded`, run directly against the real `UPSTREAMS.lock`. |
| R-21 | CONFIRMED-FIXED | `mutations.toml`'s `epoch-schedule-drops-the-answer` anchor still targets `s.result_send_count`; `mutate.py --check-anchors` reports 95 valid, 0 stale, of 95 (grew from 86 at pass2 as other layers added mutations: `base-model.toml` 7→10, `ir-schedule.toml` 2→4, `num-linalg-field-pass2.toml` 0→2 (new file), `transport-runtime.toml` 3→5; 86+3+2+2+2=95, computed from `git show <rev>:<file>` at both HEADs, not asserted). Substance unchanged from pass2's independent scratch-copy reproduction. See N-8 below for a regression in this row's *evidence-provenance text*, not its substance. |
| R-22 | CONFIRMED-FIXED | `UPSTREAMS.lock`'s `[soma].superseded` array still carries the four MOAB paths as real entries, not a comment; unchanged since pass2. |
| R-23 | CONFIRMED-FIXED | `KNOWN_ISSUES.md:165` — B1 still struck through, "CLOSED 2026-09-18, superseded by C1". |
| R-24 | CONFIRMED-FIXED | `docs/user/build.md` still documents `builtin`/`eigen` only; unchanged since pass2. |
| R-25 | CONFIRMED-FIXED, built and run | `install.consumer_find_package` passes in my tier-0 run (3.3s). See N-9 below: a real tier-2 host configure found a gap this fix's own tier-0/1 verification could not reach — now fixed. |
| R-36 | CONFIRMED-FIXED | `src/backend/intx_weights.cpp:36` still reads `#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI`, no `defined()` on the FUSE macro. |
| R-37 | CONFIRMED-FIXED, re-run | Tier-0 configure output here prints `fuse:: layers : base, model, ir, schedule, linalg, num, transport, backend, runtime` — all nine, computed, not literal. |
| R-38 | CONFIRMED-FIXED, re-run | `check-layering.py` on the real tree: 0 violations. |
| R-39 | CONFIRMED-FIXED | `docs/dev/layering.md`'s diagram still lists `linalg`/`field` with the `(*)` footnote and the `abi` row name; read in full, matches `tools/check-layering.py`'s 11-entry `LAYERS` list. |
| R-40 | CONFIRMED-FIXED | `CMakePresets.json:11` still reads `"binaryDir": "${sourceDir}/build-${presetName}"`. |
| R-41 | **CONFIRMED-FIXED — status changed since pass2** | Pass2 (HEAD `dbdc872`) correctly found this open. A later, second-round commit from the backend layer's own fix pass (`08848ec`, "Register the second-round tests") added the fix to `test/CMakeLists.txt:187-204` (in my owned paths): `find_file(FUSE_MOAB_TEST_MESH_FILE NAMES spherecube.h5m PATHS "${_fuse_moab_prefix}/share/examples/fortran" NO_DEFAULT_PATH)`, then `target_compile_definitions(fuse_backend_moab_mesh_test PRIVATE FUSE_MOAB_TEST_MESH="...")` if found, else a configure-time message (not a silently-passing skip). Checked what can be checked without a tier-2 configure: the file the `find_file` call looks for exists on this host at exactly the expected path (`/opt/petsc/download-moab/share/examples/fortran/spherecube.h5m`, confirmed present), the registration is conditional on `FUSE_ENABLE_MOAB`, and the fallback path is a message, not a definition of nothing. Execution still needs a host tier-2 run, per the existing record. **REVIEW.md's own bookkeeping (§4 Wave 3, §6 table) still says NOT-FIXED — that record predates this fix and is now stale; see New findings.** |
| R-42 | CONFIRMED-FIXED | `cmake/FuseDependencies.cmake` still has no `FetchContent_Declare(KokkosKernels ...)`. |
| R-47 | CONFIRMED-FIXED, re-run | `check-invariants.py`: 0 violations; its `scan()` area tuple is still `("include", "src", "test", "examples")`. |
| R-51 *(cross-layer, transport)* | CONFIRMED-FIXED | `test/parallel/comm_graph_test.cpp` still includes `config.hpp` before `#if FUSE_HAVE_MPI`; registered in `test/CMakeLists.txt`, compiles under tier 1. |
| R-31 *(cross-layer, transport)* | REJECTION-UPHELD, agree | Re-read `include/fuse/transport/transport.hpp:255` and `src/transport/lowfive_transport.cpp:66` again: both still read `FUSE_ENABLE_LOWFIVE`, neither is a conditional. Unchanged since pass2. |
| N-build-tools-docs-1 | CONFIRMED-FIXED | `docs/dev/roadmap.md`'s S5 row now states the MOAB backend landed and names the host-run gap; S4 row drops "the lazy field pipeline" from remaining work (kept "field transactions" and "the device leg", both still genuinely open). Diffed directly against the pass2-HEAD text. |
| N-build-tools-docs-2 | CONFIRMED-FIXED | `KNOWN_ISSUES.md` Section C no longer claims "FUSE's own suite has no MOAB-dependent test at all"; now states `backend.moab_mesh` (38 cases) is FUSE's own test and what's missing is a host run. |
| N-build-tools-docs-3 | **PARTIAL — a fresh inaccuracy in the same edit** | The count fix is correct: `grep -n "NOT-ASSESSED" build-evidence/*.md` gives exactly 5 (3 in `transport-mutations.md`, 2 in `runtime-mutations.md`), and `grep -n -i equivalent build-evidence/*.md` gives exactly 1 (`ir.scale`'s `unlowerable-step-silently-dropped`) — both now match `KNOWN_ISSUES.md`'s corrected "Five mutations"/"One mutation" text. But the rewritten bullet at `KNOWN_ISSUES.md:291` *names* the two `runtime-mutations.md` NOT-ASSESSED rows as `cross-rank-region-check-fires-unconditionally` and `region-residual-skips-the-cross-rank-reduction`, "both tagged R-10." Reading `build-evidence/runtime-mutations.md:9` directly: `cross-rank-region-check-fires-unconditionally`'s verdict is **KILLED** (test fails), not NOT-ASSESSED — it was flipped from SURVIVED to KILLED by a concurrent transport-runtime-layer commit (`ced1945` itself) after this text was written. The actual second NOT-ASSESSED row in that file is `region-failure-abandons-the-rest-of-the-period` (line 12), tagged `R-09`, not `R-10`. The bullet's *count* (five) still holds by coincidence — one wrong name swapped for the right count — but the specific mutation it points a reader at is wrong. See New findings, N-build-tools-docs-11. |
| N-build-tools-docs-4 | CONFIRMED-FIXED | `tools/check-invariants.py:71` now cites `R-06`. |
| N-build-tools-docs-5 | CONFIRMED-FIXED | `test/CMakeLists.txt:30` now cites `R-07`. |
| N-build-tools-docs-6 | CONFIRMED-FIXED (own paths); one out-of-scope instance still open | `cmake/FuseDependencies.cmake:1,77,85` and `UPSTREAMS.lock:1` no longer cite `docs/PLAN.md`/`docs/CHARTER.md`; lines 77/85 point at `docs/user/build.md`/`docs/dev/conventions.md`, both confirmed to contain the cited sentences by grep. `test/support/fuse_test_main.hpp:12,45` (outside my owned paths, flagged in the original finding for "whichever layer's second pass covers it") still has the same two dangling citations — grepped all six `pass2/re-*.md` reports for `fuse_test_main`: only this layer's own report mentions it; no layer claimed or fixed it. Still open, orphaned across the six layers — see Not found. |
| N-build-tools-docs-7 *(out of scope, backend)* | CONFIRMED-FIXED | `test/backend/moab_mesh_test.cpp:5` now cites `R-06`, fixed by the backend layer's own second-round commit `1c34018` ("Follow a moved mutation anchor and correct three stale notes"), confirmed by reading the file. Not my fix to take credit for; noted because the pass2 review flagged it as open. |
| N-build-tools-docs-8 | **NOT-FIXED — fixed once, then silently reverted** | The fix2 round's own commit `a067f1f` correctly added a provenance sentence to `build-evidence/runtime-mutations.md`'s `epoch-schedule-drops-the-answer` row ("verdict for this row reproduced by hand 2026-09-20 in an isolated scratch copy ... not by a fresh `tools/mutate.py --write-log` run"), confirmed by reading that commit's diff. A later commit from a *different* layer's own second-round fix, `1d1b62e` ("Cover the accelerated multi-rank region with no communicator", transport-runtime), rewrote the same shared file for its own unrelated purpose (adding two new mutation rows, changing an unrelated verdict) and its diff shows the provenance sentence silently dropped back to plain "KILLED (test fails)" — not restored by any commit since, including the final `ced1945` refresh, which touched only one other row in the same file. At HEAD, `build-evidence/runtime-mutations.md`'s `epoch-schedule-drops-the-answer` row again carries no note of how its KILLED verdict was produced — the exact ambiguity (tool-regenerated vs. hand-retyped) N-8 was raised to close. See New findings, N-build-tools-docs-12. |
| N-build-tools-docs-9 | CONFIRMED (code-verified; tier-2 re-execution not repeated here) | `cmake/FuseDependencies.cmake`: `fuse-moab` is `add_library(... INTERFACE IMPORTED GLOBAL)` (was a plain `INTERFACE` library, which `install(EXPORT)` rejects once `fuse-coupler-backend` links it PUBLIC). `cmake/fuse-couplerConfig.cmake.in` reconstructs the identical wrapper from baked-in `@FUSE_MOAB_INCLUDE_DIRS@`/`@FUSE_MOAB_LIBRARIES_RESOLVED@`/`@FUSE_MOAB_LIBRARY_DIRS@` substitutions and adds `find_dependency(MPI COMPONENTS CXX)` (deliberately CXX-only, matching the PUBLIC link FUSE itself uses). Read both files in full; the variable-definition order is correct (`_fuse_moab_tokens`/`_fuse_moab_libs_dedup` are set before use), and the logic matches the fixer's account. Tier 2 does not configure in this sandbox (superseded gcc), so I could not re-run the real configure+install+downstream-consumer build the fix2 report describes; verified the code shape and reasoning only. |
| N-build-tools-docs-10 | CONFIRMED (code-verified; tier-2 re-execution not repeated here) | Same file: the `-Wl,-rpath,`/`-L` token de-duplication loop (`_fuse_moab_seen_dirs`, order-preserving, exact-token match) sits after `_fuse_moab_tokens` is built and before `target_link_libraries(fuse-moab INTERFACE ${_fuse_moab_libs_dedup})` — correct ordering, and it deliberately leaves plain `-l` tokens alone as described. Could not reproduce the real MOAB install's exact token counts (44/18 collapsing to 6/4) without a working tier-2 configure here; the logic itself is sound by inspection. |

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-build-tools-docs-11 | MINOR | `KNOWN_ISSUES.md:291-292` | The pass2-round fix for the NOT-ASSESSED mutation count names the wrong mutation and the wrong finding id for one of the two `runtime-mutations.md` rows |
| N-build-tools-docs-12 | MINOR | `build-evidence/runtime-mutations.md:8` | A correctly-added evidence-provenance note (N-build-tools-docs-8's own fix) was silently dropped by a later, unrelated commit to the same shared file and never restored |
| N-build-tools-docs-13 | MINOR | `REVIEW.md:274,312` | Section 4 and Section 6's bookkeeping still report R-41 as NOT-FIXED; a later commit (outside this task's power to cite as "mine", since it lands in the backend layer's second round) fixed it, and the record was never updated |
| N-build-tools-docs-14 | MINOR | `REVIEW.md` §6 | The §6 headline ("21 new findings") does not sum from its own per-layer table (3+2+10+1+1+6 = 23) |

### N-build-tools-docs-14 — MINOR — `REVIEW.md` §6 — headline new-finding count does not match its own table

§6's prose states "21 new findings," but the table two lines below it gives each layer's "New
findings" count as backend 3, base-model 2, build-tools-docs 10, ir-schedule 1, num-linalg-field 1,
transport-runtime 6 — summing to 23, not 21 (checked by direct arithmetic on the table's own cells).
The gap reconciles only if build-tools-docs' two host-reported addenda (N-9 BLOCKER, N-10 MINOR — the
"10" in its cell already includes them, per that cell's own text) are meant to be excluded from the
headline, but §6 states no such exclusion anywhere, and the table's per-layer count is the one this
report's own Verification section above independently re-derived and confirmed correct for
build-tools-docs (8 of its 10 own-layer/addenda items CONFIRMED, 1 PARTIAL, 1 NOT-FIXED — 10 items,
matching the table). A reader taking "21" at face value and then adding the table gets a different,
unreconciled number — the same class of bookkeeping drift as R-41's stale status (N-13) and the
NOT-ASSESSED misnaming (N-11), all inside the one section meant to be this project's index of what
still needs fixing.

**Fix.** Recompute and correct the headline to "23 new findings," or, if the two host-reported
build-tools-docs addenda were deliberately meant to be counted separately from "new findings found by
a pass2 review" (they were reported by a host run mid-fix, not found by the reviewer), say so
explicitly and adjust the table's build-tools-docs cell to show 8 instead of 10, with N-9/N-10 broken
out in their own column or footnote.

### N-build-tools-docs-11 — MINOR — `KNOWN_ISSUES.md:291-292` — wrong mutation named for the NOT-ASSESSED count

The current text (added by this task's own predecessor fix round, commit `1f6541e`):

```
... and two
are in `runtime-mutations.md` (`cross-rank-region-check-fires-unconditionally`,
`region-residual-skips-the-cross-rank-reduction` — both tagged R-10 in their own
description, added while fixing that finding).
```

Reading `build-evidence/runtime-mutations.md` directly: `cross-rank-region-check-fires-unconditionally`
(line 9) is **KILLED** (test fails), not NOT-ASSESSED. The verdict was SURVIVED at the moment `1f6541e`
was written (confirmed via `git show 1d1b62e -- build-evidence/runtime-mutations.md`, which shows the
row still SURVIVED at that later commit too) and was only flipped to KILLED by the transport-runtime
layer's own fix, landing in the final `ced1945` commit ("Refresh the mutation evidence logs after the
second fix round") — after `1f6541e`'s KNOWN_ISSUES.md text was written and never revisited. The actual
second NOT-ASSESSED row in `runtime-mutations.md` is `region-failure-abandons-the-rest-of-the-period`
(line 12), which is tagged `R-09` in its own description text, not `R-10`. So the bullet's total (five)
is still correct — one wrong name substituted for the right count — but a reader who goes looking for
`cross-rank-region-check-fires-unconditionally`'s NOT-ASSESSED status will find it already killed, and
the "both tagged R-10" claim is only true of one of the two rows.

**Fix.** Replace `cross-rank-region-check-fires-unconditionally` with
`region-failure-abandons-the-rest-of-the-period` in the bullet, and change "both tagged R-10" to name
each row's actual tag (`region-residual-skips-the-cross-rank-reduction`: R-10;
`region-failure-abandons-the-rest-of-the-period`: R-09).

### N-build-tools-docs-12 — MINOR — `build-evidence/runtime-mutations.md:8` — a fix undone by a later, unrelated edit to the same file

Commit `a067f1f` ("Restore honest provenance on a re-verified mutation verdict") correctly answered
N-build-tools-docs-8 by adding, to the `epoch-schedule-drops-the-answer` row: *"verdict for this row
reproduced by hand 2026-09-20 in an isolated scratch copy after the anchor rewrite, not by a fresh
`tools/mutate.py --write-log` run."* Commit `1d1b62e` ("Cover the accelerated multi-rank region with no
communicator"), the transport-runtime layer's own second-round fix, touched the same file for an
unrelated reason (adding `conservation-check-skips-the-multi-rank-refusal` and
`region-failure-abandons-the-rest-of-the-period`, and changing `cross-rank-region-check-fires-
unconditionally`'s verdict) and its diff shows the `epoch-schedule-drops-the-answer` row reverted to
plain `**KILLED** (test fails)` with no provenance clause — most likely because that commit regenerated
or rewrote the file from a source that predated `a067f1f`'s edit, rather than editing it in place. No
commit since (`daa5fdc`, `7f5e0a2`, `ced1945`) restores the sentence; `ced1945`'s own diff for this file
touches only the `cross-rank-region-check-fires-unconditionally` row. At HEAD, the row again gives a
future reader no way to tell a tool-regenerated verdict from a hand-verified one — the exact gap
`testing.md`'s "regenerate with `--write-log`; do not edit" convention exists to prevent, and the one
N-8 was raised to close. This is a live instance of the task's own warning: "a change to one path can
invalidate a neighbouring one" — here, two fixers' edits to the same evidence file landed in the wrong
order and the second clobbered the first's correction without conflict (both are plain-text Markdown
edits; nothing forced a merge check).

**Fix.** Re-add the provenance clause to the `epoch-schedule-drops-the-answer` row, and — since this
file is jointly written by whichever layer last touches its mutations — a short header comment
directing a future editor to check the previous version's per-row prose before regenerating wholesale
would prevent a repeat.

### N-build-tools-docs-13 — MINOR — `REVIEW.md:274,312` — R-41's status record is stale

`REVIEW.md`'s Wave-3 pass2 status line (§4, line ~274) reads *"NOT-FIXED — R-41 (... correctly deferred
by the build-tools-docs fixer to the backend layer, still open at HEAD)"* and §6's table (line ~312)
repeats *"1 (R-41, correctly deferred to backend layer)"* under build-tools-docs' Not-fixed column. Both
were accurate when written (`1a87f35`, "Record the second-pass reviews in REVIEW.md") — at that point in
the commit sequence, R-41 was genuinely still open. The backend layer's own later second-round commit
(`08848ec`, "Register the second-round tests") lands *after* `1a87f35` in the commit graph and fixes
R-41 in `test/CMakeLists.txt` (see the Verification table above) — a file inside build-tools-docs' owned
paths, not backend's. Neither commit updated `REVIEW.md`'s record afterward. This is the same class of
drift `REVIEW.md`'s own §2.6 names ("records that drifted from the tree") and the class of citation bug
N-4/N-5/N-7 already found and fixed in this project three times — a bookkeeping file falling behind a
fix that landed in a file it describes.

**Fix.** Change `REVIEW.md`'s two R-41 references to CONFIRMED-FIXED, citing `08848ec` and the
`find_file`/`target_compile_definitions` mechanism now in `test/CMakeLists.txt:187-204`, and move R-41
out of the "Not-fixed" tally into "Confirmed-fixed" in both places (§4's wave count and §6's table),
adjusting the layer's Not-fixed count from 1 to 0 and its Confirmed-fixed count from 16 to 17 (own-layer)
— the report-level totals sentence added by `acf3c5b` ("39 confirmed fixed ... 2 not fixed") would also
need its build-tools-docs contribution recomputed once this lands (2 not-fixed total becomes 1: R-49 in
num-linalg-field remains genuinely open).

## Not found

- **All 17 unified findings and both cross-layer entries in this layer's pass2 scope hold up against
  the code and tests I ran myself**, not just against the fixer's or pass2 reviewer's account — except
  R-41, whose status has *improved* since pass2 (see above), and the provenance/naming regressions in
  N-3/N-8 (both minor, both in shared evidence files edited by other layers after this layer's fix
  landed).
- **Both static gates pass clean at HEAD.** `check-layering.py` and `check-invariants.py`: 0 violations
  each, re-run directly.
- **`tools/mutate.py --check-anchors`: 95 valid, 0 stale, of 95.** Grew from 86 at pass2 as other
  layers' second-round fixes added mutations (`base-model.toml` 7→10, `ir-schedule.toml` 2→4,
  `num-linalg-field-pass2.toml` 0→2 (new file), `transport-runtime.toml` 3→5 — each pair read directly
  via `git show <rev>:<file>` at `dbdc872` and `ced1945`; 86+3+2+2+2=95). The manifest total (95, summed
  across `mutations.toml` + `mutations/*.toml`) reconciles exactly against the sum of every
  `*-mutations.md` file's "N killed / M total" footer (95), and 89 killed + 5 NOT-ASSESSED + 1 EQUIVALENT
  = 95 reconciles against `KNOWN_ISSUES.md`'s corrected counts.
- **`sync-upstream.py`'s TOML fix (R-20) still works end-to-end**: `lock()` returns 50/15 superseded
  paths for soma/karma against the real, current `UPSTREAMS.lock`, run directly, not asserted.
- **`verify-import.py`: 693 files, 0 problems** at HEAD, unchanged from pass2.
- **Full tier-0 suite: 71/71 pass**, built and run here (not asserted), including
  `install.consumer_find_package`, `schedule.buffers`, `schedule.slab`. Matches pass2's count exactly —
  no regression from any of the intervening commits.
- **`sphinx-build -W --keep-going`: build succeeded, 0 warnings**, run directly against the current tree
  (23 doc sources).
- **No other stale `R-\d+` citation found in my owned paths** beyond the two already tracked
  (N-4/N-5, both already fixed) and the out-of-scope instances (N-6's `fuse_test_main.hpp`, N-7's
  `moab_mesh_test.cpp:5`, the latter now also fixed by the backend layer). Re-ran the same grep sweep
  the pass2 review used, against the current tree: every `R-[0-9]+` citation in `CMakeLists.txt`,
  `src/CMakeLists.txt`, `test/CMakeLists.txt`, `cmake/`, `CMakePresets.json`, `tools/`,
  `build-evidence/`, `.github/`, `UPSTREAMS.lock`, `KNOWN_ISSUES.md`, `REVIEW.md`, `docs/`, `examples/`,
  `test/install_consumer/` resolves to an id between R-01 and R-51 (51 unique ids, no gaps, no
  duplicates, no out-of-range citation).
- **`test/support/fuse_test_main.hpp`'s two `docs/PLAN.md`/`docs/CHARTER.md` citations remain
  unaddressed**, orphaned across all six layers: grepped every `pass2/re-*.md` report for
  `fuse_test_main` and found only this layer's own report naming it (as explicitly out of scope). No
  layer's owned-paths list names `test/support/**`; nobody has fixed it in either fix round. Not a
  finding against this layer (it was correctly flagged and correctly left alone the first time), but
  worth a human noting it before a fourth pass assumes some layer already owns it.
- **`.github/`, `examples/`, `CMakePresets.json`, top-level `CMakeLists.txt`, `src/CMakeLists.txt` are
  byte-identical to their pass2-HEAD (`dbdc872`) state** — `git diff --stat dbdc872 ced1945` for this
  layer's full path list shows the only touched files are the ones already accounted for above
  (`KNOWN_ISSUES.md`, `REVIEW.md`, `UPSTREAMS.lock`, `build-evidence/*`, `cmake/FuseDependencies.cmake`,
  `cmake/fuse-couplerConfig.cmake.in`, `docs/dev/roadmap.md`, `test/CMakeLists.txt`,
  `tools/check-invariants.py`, plus the new `pass2/*.md` report files themselves).
- **The five binding-rule pages re-read for this pass** (`invariants.md`, `layering.md`,
  `conventions.md`, `testing.md`; `porting.md`/`rank-agreement.md` skimmed for anything touching build
  artifacts) **describe nothing that current code or docs in this layer contradict**, beyond the
  already-tracked findings above. `layering.md`'s diagram and `conventions.md`'s commit-message and
  locale-pinning rules match the tree as read.

## Summary

- **17/17 unified findings + 2/2 cross-layer entries**: 18 CONFIRMED-FIXED (counting R-41 at its
  current, improved status), 1 REJECTION-UPHELD (R-31), 0 PARTIAL, 0 NOT-FIXED remaining open in this
  layer's own scope. R-41 changed verdict since pass2 (NOT-FIXED → CONFIRMED-FIXED, fixed by another
  layer's commit in a file this layer owns).
- **10 of the second-pass review's own findings re-verified** (its 8 original N-* plus the 2 addenda
  N-9/N-10): 8 CONFIRMED-FIXED (N-1, N-2, N-4, N-5, N-6 own-layer, N-9, N-10; N-7 confirmed fixed but
  credited to the backend layer, not counted here), 1 PARTIAL (N-3 — count correct, one mutation
  misnamed), 1 NOT-FIXED (N-8 — fixed once, then reverted by an unrelated commit to the same shared
  file).
- **4 new findings, all MINOR**: N-build-tools-docs-11 (the N-3 misnaming, evidenced independently of
  N-3's own verdict so it survives even if N-3 is read as "basically fixed"), N-build-tools-docs-12 (the
  N-8 regression), N-build-tools-docs-13 (REVIEW.md's stale R-41 bookkeeping), N-build-tools-docs-14
  (REVIEW.md §6's headline new-finding count doesn't sum from its own table: 21 stated vs. 23 actual).
- **0 BLOCKER, 0 MAJOR remain in this layer's own scope at HEAD `ced1945`.** Every BLOCKER/MAJOR this
  layer was ever assigned (R-05, R-06 cross-layer, R-07, R-20 through R-25) is CONFIRMED-FIXED, and the
  two BLOCKER/MINOR items added mid-second-round from a real host tier-2 run (N-9, N-10) are also
  CONFIRMED-FIXED (code-verified; full tier-2 re-execution not repeated in this sandbox, consistent with
  every prior review's stated constraint).
- **Deviations from a full run:** tier 2 (MOAB) and any MPI-launched test remain out of reach in this
  sandbox, unchanged from all three prior reviews — findings resting on them (R-41's execution, N-9/N-10's
  tier-2 re-run) are marked "code-verified" above, not reported as directly executed. Everything else —
  the 71-test tier-0 suite, all four static/mutation gates, the doc build, and every grep-based
  cross-check in this report — was run directly against HEAD `ced1945`, not inferred from any prior
  report's account of it.
