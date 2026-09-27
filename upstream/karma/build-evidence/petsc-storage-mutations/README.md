# Host action: mutation-test the PETSc field storage fix

`karma_petsc_field_storage.C` cannot be exercised in the agent sandbox: its
test calls `PetscInitialize`, which hangs there. Run on a host where PETSc
and MPI initialize:

```bash
export PATH=/opt/homebrew/bin:/opt/mpich/bin:$PATH   # adjust to the host
cmake --build --preset release --target karma_field_commit_test -j 8
python3 build-evidence/mutate.py \
  src/coupling/karma_petsc_field_storage.C karma_field_commit_test \
  build/test/karma_field_commit_test \
  build-evidence/petsc-storage-mutations/mutations.json \
  build-evidence/petsc-storage-mutations/results.log
cat build-evidence/petsc-storage-mutations/results.log
```

## What each result should be

Five of the six must be **killed** (`exit=1`): `ownership-check-dropped`,
`range-query-failure-ignored`, `publish-skips-owned-check`,
`read-skips-owned-check`, and `nonfinite-check-dropped`.

`publish-skips-duplicate-safety` is **expected to survive**, and that is not
a coverage gap to close by adding an assertion. It removes the
duplicate-then-copy-back staging so `publish()` writes into the live vector
directly. Given the owned-entries-only precondition every index is local and
in range, so a successful publish produces byte-identical results either way;
the two differ only when `VecSetValues` or assembly fails partway through,
leaving some entries written. Nothing in the test suite can provoke that
without PETSc fault injection, so no test can distinguish the mutant. The
staging is kept because the all-or-nothing contract is what
`commit_petsc_values()` has always promised its callers, not because a test
proves it.

## History

The first run of this matrix (2026-09-09, on the host) returned:

- `publish-skips-duplicate-safety: INCONCLUSIVE; mutated source does not
  compile` — the mutation deleted the `staged_vector` declaration while
  leaving `staged_vector.get()` referenced below it. Rewritten as described
  above so it compiles and actually tests something.
- `nonfinite-check-dropped: exit=0; MUTATION SURVIVED` — a real gap. Both
  routes that normally reach `publish()` screen nonfinite values first
  (`commit_petsc_values()` throws before building a transaction, and
  `FieldTransaction::stage()` refuses before commit), so nothing reached the
  check inside `publish()`. Closed by `refuses_a_nonfinite_publish()` in
  `test/field_commit_test.C`, which drives the storage directly with NaN and
  both infinities, asserts the vector is left untouched, and asserts a finite
  publish through the same object still succeeds.

## Reading the output

`mutate.py` prints its results and also writes them to the log, with
`write_text`, which replaces rather than appends. Running it and then `cat`-ing
the log therefore shows the table twice - that is the print followed by the
cat, not a doubled file. An earlier commit message here claimed the host had
run the matrix twice and that a duplicate pass was trimmed; that was a
misreading. Nothing was ever duplicated and nothing was trimmed.

## Result, host run 2026-09-10

Five killed. `publish-skips-duplicate-safety` survived, as predicted above and
for the reason given: with the owned-entries-only precondition a successful
publish is byte-identical either way, and the two forms differ only under a
partway `VecSetValues` or assembly failure that no test can provoke without
fault injection. Do not close it by adding an assertion.
