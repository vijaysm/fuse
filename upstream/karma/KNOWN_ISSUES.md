# Known issues

Open defects and standing risks, each with enough detail to act on without
consulting anyone. Commit hashes are given only as a convenience and may not
survive a rebase — every entry states the change in terms of **file, function
and code**, so it can be found and reverted by hand.

Capability boundaries (what the library does not yet do) are in the README,
not here. This file is for things that are wrong, might be wrong, or will
surprise someone.

---

## 1. Distorted pyramids under the resized quadrature — UNVERIFIED RISK

**Severity:** medium. Correctness of pyramid and prism assembly on distorted
meshes. Nothing is known to be wrong; the risk is that it has not been
checked where it is hardest to check.

### What changed

`include/numerics/fe/karma_quadrature.h`, in `create_product_quadrature()`,
which serves prism and pyramid topologies. Every axis is now sized
`(_degree + 2) / 2` points, matching edge, quad and hex. It was
`(_degree + 4) / 2`. At degree 4 a pyramid dropped from 64 points to 27 and
a prism from 24 to 18.

For the pyramid, the collapsed (Duffy) map's `h^2` Jacobian moved out of the
integrand and into the quadrature weight function, via Gauss-Jacobi with
`alpha = 2` on the collapsed axis.

### What is verified

- Monomial exactness to `~2e-16` through degree 6, two independent ways: from
  first principles in NumPy, and in C against PETSc's own
  `PetscDTGaussQuadrature` and `PetscDTGaussJacobiQuadrature`.
- Gradient terms — where the chain rule introduces `1/h` factors — on the
  reference cell, to `~2e-16`, for the real rational basis.
- `karma_fe_quadrature_exactness` and `karma_fe_pyramid_gradients` pin both
  properties in the suite, for Pyramid5, Pyramid13 and Pyramid14, with Prism6
  as a control. Both pass serially and at two ranks in a clean 422/422 run.

### What is NOT verified, and is the actual risk

**Distorted pyramids.** Everything above is on reference cells. A distorted
element makes the geometric Jacobian non-constant and the integrand genuinely
rational — for the old sizing as much as the new — so this is a question of
*relative accuracy*, not of exactness, and no test currently isolates it.

**How it would present:** not as a test failure, but as a pyramid or prism MMS
**convergence rate** shifting. Watch rates, not pass/fail.

### How to revert, without needing the commit

Three edits in `create_product_quadrature()` in
`include/numerics/fe/karma_quadrature.h`. There is one commit that made all
three (`74a2e77`, "Size the prism and pyramid quadratures at the standard
point count") if it still exists; otherwise:

1. Restore the point count:

   ```cpp
   const PetscInt points_per_axis = std::max( 1, ( _degree + 4 ) / 2 );
   ```

2. Delete the `collapsed_points` / `collapsed_weights` declarations, the
   `if( _topology != moab::MBPRISM )` block that `PetscMalloc1`s them and
   calls `PetscDTGaussJacobiQuadrature( points_per_axis, 0.0, 1.0, 2.0, 0.0,
   ... )`, and the four `PetscFree` calls for them in the success and catch
   paths.

3. In the non-prism fill loop, restore the Gauss-Legendre line rule and the
   hand-applied Jacobian:

   ```cpp
   for( PetscInt k = 0; k < line_count; ++k )
       ...
       const PetscReal height = 1.0 - line_points[k];
       points[3 * q + 2] = line_points[k];
       weights[q] = 4.0 * height * height * line_weights[i] * line_weights[j] * line_weights[k];
   ```

   and change the point count back to
   `line_count * line_count * line_count`.

`karma_fe_quadrature_exactness` asserts the *reduced* point count, so it will
fail after a revert. That is intended — it is what stops the reduction being
undone silently. Update its `expected_per_axis` if the revert is meant to
stand.

### The trap, if anyone re-derives this

PETSc's Gauss-Jacobi weights **already include** the `2^alpha (1-z)^alpha`
factor: for `alpha = 2` on `[0,1]` they sum to `4/3`, not `1/3`. That is
exactly the `4 * h * h` the old loop applied by hand. The new weight is
therefore a plain product of the three axis weights — multiplying by
`4 * h * h` again applies the Jacobian twice. This was measured against the
installed PETSc, not assumed.

Full analysis: `docs/developer-guide/pyramid-quadrature-cost.md`.

---

## 2. The exceptions-disabled configuration cannot run the full test suite

**Severity:** low, and it is the documented option behaving as specified
rather than a defect. It is recorded because the symptom looks alarming and
the cause is not where it appears.

`KARMA_ENABLE_EXCEPTIONS` is documented as "Throw C++ exceptions on errors"
(`CMakeLists.txt:32`). With it OFF, `KARMA_THROW` aborts the process instead.
A test whose job is to assert that a refusal **is raised** therefore cannot
pass: the refusal kills it. Sixteen test binaries do so.

**Do not try to find them with a grep.** Two static detectors were tried and
both were wrong, in opposite directions:

- *By helper name* — `throws()`, `throws_invalid()`, `throws_with()`,
  `throws_exception()`, and the template form
  `throws< std::logic_error >( ... )`. **Under-detects.** A pattern matching
  only `\bthrows\w*\(` misses the template form, which `system_owner_test.C`
  uses exclusively — and it misses `mesh_test.C` completely, because that file
  uses no helper at all, just inline
  `try { ... } catch( const InvalidArgument& ) { flag = true; }`.
  `karma_mesh_test` consequently reached the leg unlabelled and aborted on the
  host, which is how this was found.
- *By catch site* — any `catch` of a KARMA or `std` error type.
  **Over-detects badly:** 51 targets match, but most are the single
  `catch( const std::exception& )` in `main()` that nearly every test has for
  reporting, which never fires here.

Membership is therefore **empirical**: run each binary in the no-exceptions
build and see which abort. `cmake/karma_test_helpers.cmake` carries the
regeneration loop. Most binaries need PETSc, so it is a host action.

Reproducer, needing no PETSc or MPI:

```
./build/test/karma_basis_test                  # exit 0
./build-no-exceptions/test/karma_basis_test    # exit 134 (SIGABRT)
```

**The symptom appears far from the cause.** What a full run reports first is
three *fixture* tests aborting — `karma_moab_io_write_partition_fixture`,
`karma_fem_transfer_write_source_fixture`,
`karma_fem_transfer_write_target_fixture` — and then a cascade of `Not Run`
dependents. All three invoke `karma_moab_io_test`, whose unconditional
preamble makes about ten refusal assertions before it looks at its selector,
so every one of the 30 tests invoking that binary aborts regardless of what it
was asked to do.

**Why a label alone does not fix it.** The refusal-asserting tests carry a
`refusal_asserting` CTest label, but CTest re-adds a `FIXTURES_SETUP` test
that a selected test requires whatever label it carries — eight labelled
fixture tests were still selected for exactly that reason. The
`no-exceptions` test preset therefore selects by an explicit include list of
binaries with no refusal assertions, **minus** the six
`karma_mesh_partitioned_*` tests that require fixtures built by
`karma_moab_io_test`, and without the `karma_mesh_*` tests, whose binary
asserts refusals through inline try/catch; the label is the record of why each
one is out.

Measured, not inferred: that selection is **30** tests of 425. Label-exclusion
was tried and produced a 174-test selection that still contained nine labelled
tests, the aborting `karma_moab_io_write_partition_fixture` among them. The
allowlist is narrower than the 171 tests merely lacking the label, because it
was written as a known-safe set rather than as the complement of the unsafe
one; broadening it is open work.

**What was deliberately not done.** Compiling the refusal assertions out under
this option was tried and reverted. It cannot be done uniformly: the helpers
must still be *called* for the opposite assertion, `check( !throws( ... ) )`,
which tests that something does not refuse and is a happy path worth keeping.

The attempt is worth recording precisely, because its numbers differ from the
footprint above and the difference is the point. A macro substitution over
`check( throws` rewrote 202 sites across 15 sources — and `karma_basis_test`
still aborted afterwards, because its assertion is `check( !registry.supports(
key ) && throws( ... ) )`, which that pattern does not reach. That count covers one syntactic form only; the inline try/catch
form in `mesh_test.C` is not in it at all, which is the second reason a
site-by-site rewrite is the wrong instrument. It would trade a clear
configuration boundary for a large diff across sixteen binaries and a set of
vacuous passes.

**What the leg proves, then:** that the configuration builds and its
non-refusal paths run. Verified here by running the same selection under both
presets: identical outcomes, with the only failures
(`karma_smoke`, `karma_mesh`, `karma_coupling_compatibility_oracle`)
attributable to PETSc in *both* builds and so to the sandbox, not the option.

---

## Not defects — do not "fix" these

- **`publish-skips-duplicate-safety` survives mutation testing.** With the
  owned-entries-only precondition, a successful publish is byte-identical
  whether or not the staged vector is duplicated first; the two forms differ
  only under a partway `VecSetValues` or assembly failure that no test can
  provoke without fault injection. The staging stays because all-or-nothing is
  what `commit_petsc_values()` promises, not because a test proves it. See
  `build-evidence/petsc-storage-mutations/README.md`.

- **`mutate.py` appears to print its results twice.** It writes the log with
  `write_text`, which replaces, *and* prints the same table. Running it and
  then `cat`-ing the log shows both. Nothing is duplicated on disk.

- **Per-test wall times rose after the suite was parallelized.** Tests now
  contend for cores instead of running alone. Per-test numbers go up while the
  total goes down; comparing a serial run's per-test times against a parallel
  run's reads as a regression and is not one.
