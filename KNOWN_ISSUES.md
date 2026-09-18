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

### A1. There is no remote, so CI has never run

`.github/workflows/ci.yml` defines five legs. `git remote -v` is empty and the
history is 79 commits local-only, so **not one of those legs has ever
executed**. Two consequences:

- The OpenMP execution-space coverage is **unverified**. The installed Kokkos
  reports `Kokkos_DEVICES = THREADS;SERIAL` (read from
  `/opt/kokkos/lib/cmake/Kokkos/KokkosConfigCommon.cmake`), so locally the
  typed kernel tests instantiate on Threads and Serial only. The CI leg is the
  only thing that would build with OpenMP, because it passes
  `-DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON -DKokkos_ENABLE_OPENMP=ON` and
  fetches Kokkos rather than using the installed one.
- The static gates, docs build and imported-suite legs are likewise only ever
  run by hand here.

**What to do.** Create the remote and push. Nothing in the tree needs to
change; the workflow is already written.

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

### A4. No external code is nominated for the conformance profile

The profile is fixed — separately built, unmodified, owns its time loop, drives
the coupler through the participant ABI, exposes fields via `CFI_cdesc_t`,
disjoint rank sets, passes the compliance harness — and **no code in the
workspace satisfies it**. Until one is named, that milestone is scheduled but
unpopulated. The interim fallback is a second separately-built FUSE
executable, which must be reported as weaker, naming which profile items went
unverified.

**What to do.** Name the code, or accept the weaker fallback explicitly.

### A5. A device performance number needs hardware

This machine has no GPU. Every kernel in `include/fuse/num/space_kernels.hpp`
is parameterised on an execution space and verified against the serial oracle
on every space the build enables, but **parameterisation is a precondition for
device performance, not evidence of it**, and the headers say so.

**What to do.** Nothing, unless you want a measured device claim — that needs
a machine with a GPU-enabled Kokkos, at which point the typed tests
instantiate on it with no code change.

---

## B. Mine, and already scheduled — please do not act on these

### B1. Tier 2 does not build: the MOAB backend is unported

`-DFUSE_ENABLE_MOAB=ON` now configures and resolves MOAB 5.6.0 correctly, then
fails to compile:

```
src/backend/intx_weights.cpp:49:10: fatal error: 'fuse/backend/moab_backend.hpp' file not found
```

This is **not** a build defect. `upstream/soma/src/mesh/moab_backend.cpp` is
2,111 lines and has not been ported yet; porting it is the mesh-backend step of
S5. The conservative remap path that does not need MOAB works today and is the
differential oracle for this one when it lands.

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

## C. Not defects — do not fix these

- **Two mutations report `NOT-ASSESSED` locally.** They need MPI, and a test
  that cannot start exits non-zero whether or not the mutation had an effect.
  Recording that as a kill would be evidence worse than none.
- **Two mutations are declared equivalent**, each with a written bound. The
  backend one — computing the conservation audit before rather than after the
  negligible-overlap floor — is indistinguishable because the floor discards
  only contributions below 1e-12 of a cell's measure. **Raising the floor would
  make it distinguishable again** and the declaration says so.
- **The four `SkippedWithoutMoab` entries are the PARENT's, not FUSE's.** They
  live in `upstream/soma/test/unit/{moab_mesh,remap,remap_order,remap_runtime}_test.cpp`
  and are run by the imported-suite CI leg. FUSE's own suite has no
  MOAB-dependent test at all, which is consistent with B1: there is no MOAB
  backend here yet to test. Do not read those four skips as FUSE coverage.
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
