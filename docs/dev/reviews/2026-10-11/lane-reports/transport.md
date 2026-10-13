# Lane report — transport (fix round 2026-10-12/13)

Lane head at handback: `61c27d00bcd7` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- Did not reconfigure higgs slot1 to FUSE_ENABLE_LOWFIVE=ON to compile-check TR-2/TR-4's lowfive_plugin_vol.cpp, since that would require a disruptive full reconfigure+rebuild of a slot shared with other concurrent lanes; those two findings are reported as fixed-by-derivation, verified by reading only, per the task's own explicit fallback for this case.
- Did not isolate TR-16/TR-17's new tier-0 tests with a per-test gtest_filter run on higgs (ran the whole transport.transport/transport.connect binaries instead, which passed); individual test identity is inferred from the binary's aggregate pass rather than a filtered run.

## Findings

### TR-1 — fixed

Tests: `parallel.mpi_transport.2 (MpiTransportPeers.AsymmetricPhaseAFailureIsRefusedByEveryRank)`

Evidence: Pre-fix: ctest/gtest run under mpiexec -n 2 on higgs slot1 hung and was killed by `timeout -k 10 120` (exit 124) with the test stuck at [ RUN ] -- the exact pre-fix signature the brief calls for. Post-fix: same binary, same filter, [ OK ] in 0ms. Both legs ran in higgs job 64622932-0be4-4b3f-bed6-f37472086c11 (tr1_prefix_run.log / tr1_postfix_run.log). Full matrix confirmation: parallel.mpi_transport.{1..5} all Passed in job 4139130b-9dd4-4235-988d-86c87e983f8b (mpi_neighbour_ctest.log), and the full tree build was clean.

### TR-2 — fixed

Evidence: Fixed by derivation per the owner's ruling. verified by reading, not compiled: FUSE_ENABLE_LOWFIVE is OFF in slot1's existing configure (and in every slot made available to this round), consistent with KNOWN_ISSUES D14's libfmt.8 dlopen blocker, so src/CMakeLists.txt never adds lowfive_plugin_vol.cpp to the build at all in this configuration -- flipping FUSE_ENABLE_LOWFIVE ON would need a full reconfigure+rebuild of the shared slot, which was not attempted (see notes_for_lead). The full-tree build and all static gates are otherwise green at this commit.

Record text: D14 addendum (review 2026-10-11 TR-2, TR-4): fuse-lowfive-vol's ConfigureOnce() now issues MPI_Comm_split/MPI_Intercomm_create on every world rank (color MPI_UNDEFINED for a rank the LowFive channel does not name) instead of only on participants, and aborts the job on a failed LoadConfig() or a distributed config built without DistMetadataVOL instead of attaching a zero-binding connector. Fixed by derivation and unmeasured: FUSE_ENABLE_LOWFIVE is OFF on every host available to this round, so neither the fix nor its two regression scenarios (a non-participant world rank; a missing DistMetadataVOL build) has run. The fix does not close this finding's second hazard: the rendezvous still runs only on a rank whose own first H5P_DEFAULT call triggers this lazily-loaded plugin; a rank that never touches HDF5 (a pure orchestrator) still never reaches it. Moving the rendezvous into an explicit step every participant calls regardless of its own HDF5 usage (connect()'s own MPMD choreography is the natural site, since every FUSE-aware process in the launch already calls it) is the larger fix that closes that half; left to the lead.

### TR-3 — fixed

Evidence: Deleted rather than defined (nothing in the tree calls LowFiveTransport::build_plan_from_peers; runner.cpp's own peer-wise path calls it only through an MpiTransport*). transport.hpp compiles into every transport TU; the full-tree build on higgs (job 4139130b-9dd4-4235-988d-86c87e983f8b) was clean, confirming no caller was broken by the removal.

### TR-4 — fixed

Evidence: Same commit and same unmeasured status as TR-2; see TR-2's evidence and record_text.

Record text: Recorded jointly with TR-2 under the same D14 addendum; see TR-2's record_text.

### TR-5 — fixed

Evidence: Compiled cleanly (connect.cpp is part of fuse-coupler-transport, built unconditionally; full-tree build on higgs was clean). Not run: connect_pairwise() is reachable only when FUSE_ENABLE_CONNECT_ACCEPT is ON, and slot1's existing configure has it OFF (KNOWN_ISSUES D23, MPICH ch3:nemesis), so no test anywhere in the tree can reach this code on this host -- verified by reading only.

Record text: D23 addendum (review 2026-10-11 TR-5): connect_pairwise()'s acceptor-side rank 0 now broadcasts its own MPI_Open_port/publish_port_file outcome to every acceptor rank before the port-name Bcast, so a rank-0-only failure no longer leaves its peers waiting in that collective. Unmeasured on higgs: the whole connect_pairwise() path is gated behind FUSE_ENABLE_CONNECT_ACCEPT, OFF here per this same entry's finding, so the fix compiled cleanly under slot1's configure but could not be exercised by any running test.

### TR-6 — fixed

Evidence: Documentation-only fix to docs/user/multiple-executables.md; verified by reading.

### TR-7 — fixed

Tests: `parallel.comm_graph.2 (AccumulateSharedRows.AMismatchedBufferOnOneRankIsRefusedByEveryRank, CopyOwnerSharedRows.AMismatchedBufferOnOneRankIsRefusedByEveryRank)`

Evidence: Pre-fix: both new tests run under mpiexec -n 2 hung (the AccumulateSharedRows case printed [ RUN ] and was killed by `timeout -k 10 120`, exit 124) -- the surviving rank entered MPI_Allgather alone, exactly the pre-fix signature. Post-fix: all 3 new tests (both AMismatchedBufferOnOneRank cases plus TR-8's AGlobalIdNobodyOwns case) [ OK ]. Both legs ran in higgs job 64622932-0be4-4b3f-bed6-f37472086c11 (tr78_prefix_run.log / tr78_postfix_run.log). Full matrix: parallel.comm_graph.{1..5} all Passed in job 4139130b-9dd4-4235-988d-86c87e983f8b.

### TR-8 — fixed

Tests: `parallel.comm_graph.2 (CopyOwnerSharedRows.AGlobalIdNobodyOwnsIsRefusedByEveryRank)`

Evidence: Same before/after run as TR-7 (both fixes are in the same commit and the same test binary). The new AGlobalIdNobodyOwnsIsRefusedByEveryRank test constructs a gid held (unowned) by rank 0 and absent from rank 1 entirely; pre-fix this is exactly the shape that made the old per-rank-local scan return success on rank 1 and failure on rank 0 (not asserted in isolation here because the binary was killed earlier in the same pre-fix run by the AccumulateSharedRows hang ahead of it in test order -- see TR-7's evidence); post-fix it passes, agreed on both ranks.

### TR-9 — fixed

Evidence: Verified by reading for the specific failure branch: a local MPI_Comm_group/MPI_Group_incl or MPI_Comm_dup failure is resource-exhaustion-only and cannot be triggered in a test environment (no test anywhere in the tree exercises this class, consistent with the file's other abort_job windows, none of which are individually tested either). Compiled and exercised indirectly: the full parallel.comm_graph.{1..5} and parallel.runner_transport.{1..5} suites (job 4139130b-9dd4-4235-988d-86c87e983f8b) all pass, confirming the surrounding code paths these two sites sit in are unbroken.

### TR-10 — fixed

Evidence: Verified by reading for the bad_alloc trigger itself (these are small, comm-size-scaled allocations that cannot realistically fail). Compiled and exercised indirectly: parallel.mpi_transport.{1..5} all pass in job 4139130b-9dd4-4235-988d-86c87e983f8b, including through the exact code paths now wrapped in try/abort_job.

### TR-11 — fixed

Evidence: Comment-only fix to comm_graph.hpp; verified by reading.

### TR-12 — out-of-scope

Evidence: Assigned to the runtime lane per the task brief (docs/dev/rank-agreement.md is not an owned path this round).

### TR-13 — fixed

Evidence: One-line rename in lowfive_transport.cpp (the sentinel never leaves the process, so there is nothing to assert against). lowfive_transport.cpp is part of fuse-coupler-transport, built unconditionally; the full-tree build on higgs (job 4139130b-9dd4-4235-988d-86c87e983f8b) was clean.

### TR-14 — fixed

Evidence: Compiled and exercised indirectly: parallel.mpi_transport.{1..5} all pass in job 4139130b-9dd4-4235-988d-86c87e983f8b, through the exact check_peers path now carrying the guard. Verified by reading for the overflow trigger itself (needs on the order of 2^31/components local indices on one peer -- tens of GB -- impractical to construct in a test).

### TR-15 — fixed

Evidence: Documentation-only fix to transport.hpp's error lists; verified by reading against code_of()'s mapping.

### TR-16 — fixed

Tests: `transport.transport (CopyTransport.AcceptsNullPointersWhenNIsZero)`

Evidence: transport.transport Passed (0.32-0.79s, job 4139130b-9dd4-4235-988d-86c87e983f8b and its tier0_ctest.log), which runs every TEST() in test/transport/transport_test.cpp including the new AcceptsNullPointersWhenNIsZero case and the pre-existing RejectsNullPointers case (unaffected, since it uses n=4); gtest's aggregate exit code is 0 only when every case passes.

### TR-17 — fixed

Tests: `transport.connect (ConnectModeComponents.RefusesAThirdConnectModeComponent, ConnectModeComponents.RefusesJustOneConnectModeComponent, ConnectModeComponents.AcceptsExactlyTwoInGraphOrder)`

Evidence: transport.connect Passed (0.32-0.77s, job 4139130b-9dd4-4235-988d-86c87e983f8b), which runs every TEST() in test/transport/connect_test.cpp including the three new ConnectModeComponents cases -- the first tier-0-reachable test of D18's refusal in the tree.

### TR-18 — out-of-scope

Evidence: IMPROVEMENT-tier; out of scope for this round per the lane brief.

## Notes for the lead

All 16 in-scope findings (TR-1..TR-11, TR-13..TR-17) addressed with 13 commits; TR-12 (runtime lane's docs/dev/rank-agreement.md epoch tag) and TR-18 (IMPROVEMENT) left out of scope per the brief. git status --porcelain is clean on review-1011/transport at 61c27d0, 13 commits ahead of e6a92e4. Higgs evidence (slot1, FUSE_ENABLE_MPI=ON, FUSE_ENABLE_MOAB=ON, FUSE_ENABLE_LOWFIVE=OFF, FUSE_ENABLE_CONNECT_ACCEPT=OFF): TR-1's and TR-7/TR-8's hang-class fixes each have real fails-before (ctest/mpiexec hang, killed by `timeout -k 10 120`, exit 124) / passes-after (gtest [ OK ]) pairs, both in job 64622932-0be4-4b3f-bed6-f37472086c11. The whole tree builds clean (job 4139130b-9dd4-4235-988d-86c87e983f8b, fulltree_build.log, exit 0), all 4 `transport` label tests pass, the `mpi` neighbour check (parallel.{runner_transport,comm_graph,mpi_transport}.{1..5}, 15/15) passes, and all 4 static gates (check-invariants, check-layering, gen-abi --check, verify-import) report 0 violations at this commit. Full combined log saved as review-1011-transport-higgs-log.txt. Two findings (TR-2, TR-4) are fixed by derivation per the owner's explicit ruling but UNMEASURED: FUSE_ENABLE_LOWFIVE is OFF in slot1's existing configure (consistent with KNOWN_ISSUES D14's libfmt.8 dlopen blocker), and src/CMakeLists.txt never adds lowfive_plugin_vol.cpp to the build in that configuration at all -- the file was not even compiled, let alone run. I deliberately did NOT reconfigure the shared slot1 to flip FUSE_ENABLE_LOWFIVE ON: that needs a full reconfigure+rebuild of the tree slot1 holds for every other lane using it this round, which the brief's own 'only one lane uses a slot at a time' note suggests other lanes may still need in its current state. If the lead wants this compiled, a dedicated LowFive-enabled configure (fresh build dir, not slot1's) is the safer way to get it without disrupting anyone else's in-flight work. TR-5's fix is compiled (connect.cpp builds unconditionally and the full-tree build is clean) but likewise unmeasured: FUSE_ENABLE_CONNECT_ACCEPT is OFF on this host (KNOWN_ISSUES D23, MPICH ch3:nemesis), so connect_pairwise() -- and therefore this fix -- has no reachable test anywhere in the tree on higgs. D14 and D23 addendum text for both unmeasured items is in the TR-2/TR-4 and TR-5 findings' record_text above, ready to paste into KNOWN_ISSUES.md. TR-9's and TR-10's fixes (abort_job on local-MPI-call and inter-collective-allocation failures) are compiled and exercised indirectly by the full comm_graph/mpi_transport/runner_transport suites, but the specific abort branches themselves cannot be triggered by any test (resource-exhaustion-only), matching the file's own untested precedent for every other abort_job call site. No files outside the owned paths were touched. No test/CMakeLists.txt changes were needed: every new test landed in an existing gtest/mpi-test binary already registered at the rank counts needed (mpi_transport/comm_graph default to RANKS 1 2 3 4 5; transport/connect are plain tier-0 gtests).
