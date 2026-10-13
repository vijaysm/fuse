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

### Naming the failing ranks, not just counting them

Reducing the failure status across the communicator answers "did this test
pass", but a launcher commonly forwards only one rank's stdout, so the rank
that actually failed can be entirely absent from the log — "failed on 1 of 3
rank(s)" is then the whole evidence (measured on the host, 2026-10-08,
`parallel.window_runner_mpi.3`). `MpiFailureReducer::OnTestEnd`
(`test/support/fuse_test_main.cpp`) now `MPI_Allgather`s each rank's own
pass/fail bit — collective, since every rank runs the same tests in the same
order — and a rank that passed its own assertions but sees a peer fail adds
its own `ADD_FAILURE()` naming which rank(s) failed, so the test fails
uniformly (no rank proceeds alone into the next collective) and the report
names where to look:

```
failed on rank(s) 2 of 3; this rank (0) passed its own assertions. Reported
here so the test fails uniformly and no rank proceeds to the next
collective. For the failing rank's own output, rerun with a per-rank
capture: mpiexec -n 3 -outfile-pattern=/tmp/<name>-%r.out <binary>
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

`KNOWN_ISSUES.md` D11, CLOSED 2026-09-28: `src/runtime/`'s `execute()`/`run_region()` now call
`num/space_kernels.hpp`'s kernels over `Runner::Impl::HostExecSpace`
(`Kokkos::DefaultHostExecutionSpace`), so `example.*.omp_identical` and this leg are load-bearing
for the coupling hot loop, not merely for the partitioner, the native mesh and the typed kernels.
Elementwise operators (`Scale`, `Merge`, `Residual`, `Relax`, the remap apply's CSR matvec, and
`Interpolate`/`Sample`/`Accumulate`'s copies and finalisers) are thread-count-independent because
each output element's write touches no other; the per-sweep block value/residual norms
(`iteration.cpp`) are the one place that argument does not apply, since a plain
`Kokkos::parallel_reduce`'s summation order is the threads' — they use a fixed-chunk scheme instead
(`space_kernels.hpp`'s `deterministic_dot`/`deterministic_norm_squared`(`_diff`)), verified against
a plain-host reference with `EXPECT_EQ` under `FUSE_FP_REPRODUCIBLE` and against a derived bound
otherwise (`SpaceKernels.
DeterministicReductionsMatchTheChunkedReferenceExactlyOrWithinTheFpContractionBound` — renamed from
`...Exactly`, see "Floating-point contraction policy" below) and, at the Runner level, by a real
2048-element iteration region pinned to measured literals across every `OMP_NUM_THREADS`
(`runtime.runner_determinism`, `fuse_add_omp_legs`). See `KNOWN_ISSUES.md` D11's closing entry for
the full account, including the one label-length allocation trap this closing found and fixed, and
what remains open for a device execution space under A5.

### The device leg (`ctest -L device`)

A Kokkos built with a CUDA/HIP/SYCL backend registers every Runner-driving verification and
example row a second time as `<row>.device` (`fuse_add_device_twin()` in `test/CMakeLists.txt`,
label `device`; absent, not skipped, on a build without a device backend). The twin is the same
binary with `FUSE_EXECUTION_SPACE=device` in its environment: `RunnerConfig::execution_space`
defaults to `Default`, which `Runner::initialize()` resolves from that variable (`device`, `host`,
unset; anything else is `E-RT-SPACE`), so the graphs run under the device interpreter --
coupler arcs in device memory, host-registered fields mirrored once per component step,
Space-generic kernels on `Kokkos::DefaultExecutionSpace` -- and must reproduce the host row's
verdict (docs/dev/device-resident-arcs.md). A caller that names `Host` or `Device` explicitly is
never overridden, which is how `runtime.runner_device` compares the two spaces in one process.

Two rows skip themselves under the device leg: `Iteration.TheSweepLoopAllocatesNothing` and
`Iteration.TheAcceleratorAllocatesNothingPerUpdate`. Kokkos's `deep_copy` and `parallel_for`
construct a `std::string` from their fence/launch labels on every call, so under the device
interpreter the sweep loop allocates in Kokkos, not in FUSE; the rows say so and skip rather than
measure the wrong thing. `FUSE_DEVICE_DEBUG=1` prints the Runner's placement (one line per
instruction and per arc) and is the first thing to read when a twin fails.

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

## The derivative gate

A Newton step, a monolithic Jacobian block, or any other operator whose correctness depends on its
own adjoint or its own linearisation is **Verified** (the state above) only once it has a case
citing it in `linalg.derivative_gate` ([](derivative-gate.md)): the adjoint dot-product identity
against a bound derived from the operator's own norm, finite-difference derivative convergence with
the round-off plateau detected rather than fit through, and assembled-vs-matrix-free agreement. An
operator with an `apply` and no such case is **Implemented, untested** for that property, however
well-tested its forward `apply` alone is.

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

## Floating-point contraction policy

`FUSE_FP_REPRODUCIBLE` (`cmake/FuseFpPolicy.cmake`, default **ON**) decides whether `a*b + c` is
rounded once (a fused multiply-add) or twice (multiply, round, then add — `-ffp-contract=off`,
applied to every C, C++ and Fortran target). **Measured 2026-10-12**, reproducing `KNOWN_ISSUES.md`
D30 at the stock compiler default (no flag at all, `tier0-release`, Apple clang 21.0.0): exactly
three named cases fail by one ulp and nothing else does —
`DistVectorOneRank.ArithmeticMatchesAPlainLoopBitForBit` (`linalg.dist_vector`) and
`SpaceKernels/{0,1}.DeterministicReductionsMatchTheChunkedReferenceExactly`
(`num.space_kernels{,.omp1,.omp8}`) — 184 of 188 tier-0 rows passing. Both are renamed below
(`...ExactlyOrWithinTheFpContractionBound`) because under `FUSE_FP_REPRODUCIBLE=OFF` they no longer
claim bit-identity.

**D30's own account of *why* this happens was wrong, and the record is corrected here rather than
quietly fixed.** D30 said "Apple clang contracts `a*b + c` into a fused multiply-add at `-O2` and
not at `-O0`." That is false, confirmed directly: `double f(double a,double b,double c){return
a*b+c;}` compiled `-S` emits one `fmadd` (the fused form) at `-O0` with the compiler's **default**
flags, zero `fmadd` at `-O0 -ffp-contract=off`, and the SAME one `fmadd` at `-O2` with default flags
— `-ffp-contract`'s default is `on` at every optimization level this compiler has; it is not toggled
by `-O0` vs `-O2`. The true mechanism, found by disassembling the actual built objects (not assumed):
`DistVector::dot()`'s and `deterministic_dot()`'s reduction loops have a **runtime** trip count, so
LLVM's auto-vectorizer at `-O2`/`-O3` splits each into a NEON-vectorized main body — computing every
product with a plain, always-unfused `fmul.2d` and summing the results with a chain of scalar `fadd`
— plus a short scalar **remainder** loop (the elements left over after the vector width) that uses one
fused `fmadd` by default, or the same unfused `fmul`+`fadd` under `-ffp-contract=off`; confirmed
identical vectorized-body instructions, flag or no flag, in both `DistVector::dot` and
`deterministic_dot<Serial>`, with the remainder tail the ONLY place that differs. `dist_vector_test.cpp`'s
own reference loop, whose trip count (`n=37`) IS a compile-time constant, is instead FULLY UNROLLED
by LLVM into 37 individual scalar instructions — fused by default, unfused under the flag — never
vectorized at all. At `-O0` (Debug), nothing is vectorized or unrolled by either side: the library's
loop and the test's reference loop both compile to the textually literal sequence of fused `fmadd`
instructions, in the same ascending order, and so necessarily agree bit for bit — which is why Debug
always passed, not because `-O0` fails to contract. At `-O2`/`-O3` without the flag, the library's
loop is *partly* unfused (its vectorized main body, unconditionally) while the test's loop is
*entirely* fused (unrolled, default contraction) — a genuine disagreement over the vectorized
elements, which is the real source of the one-ulp mismatch. Adding `-ffp-contract=off` project-wide
does not "restore Debug's bits" in Release: it makes the library's already-partly-unfused loop and
the test's now-also-unfused unrolled loop agree with EACH OTHER (all unfused, ascending order) — a
DIFFERENT, equally valid rounding from Debug's own historical (all-fused) one, which happens to
satisfy the test's equality assertion because both sides of the comparison now take the same path,
not because either value matches a prior baseline. This is also why `FUSE_FP_REPRODUCIBLE=ON`
changes Debug's OWN numbers, not just Release's: the flag reaches `-O0` too, turning Debug's
previously-always-fused arithmetic into always-unfused arithmetic — a genuinely NEW numerical
configuration in both build types, which is the reason the two regressions below
(`num.accelerator`, `verification.fsi_energy_diag`) appear under `ON` in Debug as well as Release.
`deterministic_dot`'s inner chunk loop (`space_kernels.hpp`) and the test's own
`ChunkedSerialDotReference` showed the identical vectorized-main/fused-remainder split on direct
disassembly of the same compiled object, flag on vs off; the exact size of that remainder tail for a
given chunk was not re-derived from the generic trip-count arithmetic the vectorizer emits (that
depends on register-allocation and cost-model choices specific to each call site, which a minimal
standalone reproduction of the same source did not reproduce — stated rather than guessed at).

### The forward-error bound (OFF)

With the flag off, the three comparisons above are checked against the standard forward-error bound
for recursive summation (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., secs.
3.1/4.2): two IEEE 754 evaluations of `sum_i x_i` — contracted or not, since a fused multiply-add is
itself correctly rounded and so cannot make an evaluation *worse* than this bound — differ by at most

```
2 * gamma_n * sum_i |x_i|,   gamma_n = n*u / (1 - n*u),   u = 2^-53
```

`test/linalg/dist_vector_test.cpp`'s `forward_error_bound()` uses `n` = the vector length (37) for
the flat `DistVector::dot`/self-dot reduction. `test/num/space_kernels_test.cpp`'s
`chunked_forward_error_bound()` generalises `n` for `deterministic_dot`/`_norm_squared`(`_diff`)'s
two-level chunked scheme (chunks of up to `kDeterministicChunk` = 256, then a sum of chunk totals):
it uses `n_eff = b + n_chunks`, where `b = min(256, n)` is the longest chunk and `n_chunks =
deterministic_chunks(n)` the number of chunks — one term more than the exact root-to-leaf operation
count (`(b-1) + (n_chunks-1)`) needs, and strictly larger than the tighter two-level bound
(`gamma_{b-1} + gamma_{n_chunks-1}`, valid by the triangle inequality since each level's own partial
sums are bounded by the same `sum_i|x_i|`) by `gamma`'s monotonicity — a conservative, not a tuned,
choice (the test file's own comment has the full derivation). `deterministic_sum`'s own comparison,
and the single-chunk-equals-the-plain-oracle check in the same test, stay exact in both
configurations: `s += v[i]` contains no multiply, so no contraction policy can change its rounding.

**Worst observed ratio of (measured difference) / (bound), Release + `FUSE_FP_REPRODUCIBLE=OFF`,
printed by every run of these two tests (not only on failure):** 0.0060 (`DistVector::dot`, n=37);
0.0031 (`deterministic_norm_squared`, n=256) — three orders of magnitude of margin, consistent with a
bound that was derived, not tuned to the data.

### The probe test (`base.fp_policy`)

`test/base/fp_policy_test.cpp` checks that the flag actually reached the compiler, with a
hand-picked triple (`kA`/`kB`/`kC`, recorded as exact hexadecimal literals) for which `std::fma` and
a volatile-forced two-step evaluation are proven, by the test's own `ASSERT_NE` precondition, to
differ on this compiler — then asserts `a*b+c ==` the two-step value under `FUSE_FP_REPRODUCIBLE==1`
and only prints the observed difference, asserting nothing, under `==0`. Because the flag's default
(`on`, fused) applies at **every** optimization level, not only `-O2`/`-O3` (the corrected premise
above), this single, unvectorized expression is sensitive to the flag in Debug exactly as it is in
Release — **hand-proved 2026-10-12** in both: deleting the call that applies the C/CXX half of the
policy (`fuse_apply_fp_contract_policy_c_cxx()`, `CMakeLists.txt`) fails
`FpPolicy.ContractionIsOffWhenReproducible` in a Release tree (`-4.0672332501391795` vs
`-4.0672332501391804`) AND in a Debug tree (same two values), while the precondition test still
passes in both; `CMakeLists.txt` was restored byte-identically after each. `tools/mutate.py`'s
`requires = "release"` gate, added under the original (wrong) premise that only a Release tree could
see this mutation, is therefore DROPPED (`build-evidence/mutations/fp-policy.toml`'s row carries no
`requires` key) — the row is now proved and KILLED in whichever build type `mutate.py` happens to run
against, with no configuration-dependent exemption.

### Applying the flag without reaching third-party sources

The C/CXX half is applied in `CMakeLists.txt` right after `fuse_resolve_dependencies()` — once every
`FetchContent_MakeAvailable` call that function makes (r3d, yaml-cpp, on a machine with neither
installed) has already run — so it reaches `src/`, `test/`, `examples/` and the golden-descriptor
tool, never the vendored third-party sources or `upstream/soma`/`upstream/karma` (invariant I8: an
import that had to be patched to build is not an import, and injecting a new compiler flag into a
vendored build is the same category of change even without editing its own `CMakeLists.txt`). The
Fortran half is a separate function (`fuse_apply_fp_contract_policy_fortran()`), called from
`cmake/FuseFortran.cmake` right after `enable_language(Fortran)` and before `add_subdirectory(fortran)`
— `CMAKE_Fortran_COMPILER_ID` does not exist before that call, and a directory added before a compile
option is set in its parent's scope never inherits it, the same ordering hazard `FUSE_ENABLE_FORTRAN`
is already declared early to avoid. Verified in `compile_commands.json`, not assumed: every Kokkos-
kernel-instantiating TU checked (`src/linalg/dist_vector.cpp`, `dist_vector_test.cpp`,
`space_kernels_test.cpp`, `brusselator_physics_test.cpp`, `fp_policy_test.cpp`) carries
`-ffp-contract=off` under `FUSE_FP_REPRODUCIBLE=ON`, and the Fortran provider's and `ffuse`'s own
`flags.make` carry it too.

**Compiler coverage.** GNU, Clang and AppleClang accept `-ffp-contract=off` identically (confirmed on
this machine: Apple clang 21.0.0 and Homebrew GNU Fortran 16.2.0). Any other C/C++ compiler gets a
CMake `WARNING` instead of a silently-assumed flag (MSVC's `/fp:precise` and Intel's `-fp-model=
precise` are different flags, untested here, not added). **Not covered, by design:** a CUDA device
build's own contraction (`nvcc`'s `-fmad`, default `ON`) needs the matching `-fmad=false` — documented
here as follow-up for the device branch (A5), not added, since this machine has no CUDA toolchain to
test it against.

### Measured run-time cost

**NOISY — see caveat below.** `OMP_NUM_THREADS=1`, 11 repetitions per cell (median \[min, max\]),
compiled with each build's own flags, `-O3 -DNDEBUG -mcpu=native`:

| Kernel | n | ON (reproducible) | OFF (fast) |
|---|---|---|---|
| `DistVector::dot` | 5,000,000 | 4.693 \[4.511, 4.869\] ms | 4.688 \[4.443, 5.076\] ms |
| `DistVector::axpy` | 5,000,000 | 1.972 \[1.432, 3.869\] ms | 1.765 \[1.315, 2.374\] ms |
| `deterministic_dot<Serial>` | 5,000,000 | 2.702 \[2.591, 2.943\] ms | 3.514 \[3.062, 4.665\] ms |
| `deterministic_sum<Serial>` | 5,000,000 | 2.518 \[2.436, 2.854\] ms | 2.875 \[2.542, 4.466\] ms |
| `brusselator::reaction_kokkos_span` | 2,000,000 | 0.530 \[0.514, 1.143\] ms | 0.581 \[0.530, 1.529\] ms |
| `fuse_example_brusselator --driver runner --end 1.0` (end to end) | — | 1008.3 \[974.2, 1111.5\] ms | 994.7 \[990.1, 1064.9\] ms |

**Caveat, stated rather than hidden: this machine could not be made quiet.** The methodology called
for `load1 < 3` before each batch, retried up to 10 times at 60 s intervals; two other tracks were
building concurrently throughout and `load1` never dropped below 6 in ten attempts (range 6.0–13.8
across all batches). Every row above is labelled **NOISY**. `deterministic_sum` contains no multiply
at all (`s += v[i]`) and therefore cannot be affected by a contraction policy either way, yet shows a
"cost" of the same order as the kernels that do — direct evidence the differences above are
contention noise, not a measured cost attributable to `-ffp-contract`. **The lead repeats this
measurement on a quiet machine**; until then, no speedup or slowdown number from this table should be
cited as a property of the flag.

### A measured cost beyond speed: iteration-sensitive tests, restated as what they actually show

Round 2 of this track found that `FUSE_FP_REPRODUCIBLE=ON` (the chosen default) regressed three
pre-existing, unowned tests because each asserted a *specific* iteration count or energy-closure
value that turned out to be a measurement of the compiler's default (contracted) arithmetic, not a
robust property of the method. **Round 3, measured 2026-10-12 (user decision 2026-10-12: restate
each test as the property it is actually named for; default stays `ON`)**, restated two of the
three and made the third's own exemption visible rather than silent — this track edited only these
three `TEST(...)` bodies (its only edit permission this round); `examples/` and `src/` were untouched:

- **`Accelerator.LadderIsMonotoneOnADenseLinearProblem`** (`test/num/accelerator_test.cpp`) no
  longer pins `n_aitken == 56`. A 24-way perturbation of the problem's initial vector by 1 ulp of
  1.0 showed `n_aitken` spans `{55, 56, 57, 59}` under BOTH contraction policies (contracted:
  `{55:3, 56:12, 57:7, 59:2}`, mean 56.42; `ON`: `{55:3, 56:14, 57:6, 59:1}`, mean 56.25) — 56 is
  the single most common value in both, and the `56->57` shift between the two unperturbed
  baselines sits well inside this natural scatter, confirming the literal was a rounding-dependent
  number. The test's *own name* — "monotone" — is what is actually asserted now: the pre-existing
  ordering relations (`n_secant < n_picard`, `n_aitken < n_secant`, `n_ils < n_aitken`,
  `n_imvj <= n_ils`) hold for every one of the 48 measured values of `n_aitken`, under both
  policies, so this test now PASSES under `ON`.
- **`Accelerator.SteffensenIsExactOnADiagonalProblemAndBasisDependent`** (same file) no longer pins
  `n_dense == 681`. Residual histories under the two policies track each other to 5-6 significant
  figures through iteration 35, then both enter an intermittent-burst regime (k~60-165) consistent
  with a trajectory near a chaotic basin boundary; the contracted build's bursts happen to relax
  back into convergence by k=681, `ON`'s does not and diverges instead. The same 24-way, 1-ulp
  perturbation showed this is not specific to `ON`: 4 of 24 (16.7%) perturbed runs also diverge
  (non-convergent, `-1`) under the DEFAULT contracted policy, and 8 of 24 (33%) under `ON`; the
  convergent runs ranged 488-2366 iterations in both policies. 681 was one outcome of a dynamical
  system whose qualitative behaviour is controlled by perturbations far smaller than any physically
  meaningful input, not a reproducible property — so the test now asserts what it is actually named
  for, basis dependence: `n_dense == -1` or `n_dense >= 20 * n_diag` (with `n_diag` itself exact at
  `2` in all 48 measured runs). `K = 20` carries more than a 12x margin below the weakest ratio
  actually measured (488/2 = 244) while remaining far larger than any ratio a well-conditioned
  method could reach by chance; this test now also PASSES under `ON`.
- **`FsiEnergyDiag.FullBudgetOnTheDominantModeCloses`** (`test/verification/fsi_energy_diag_test.cpp`).
  Round 3's `GTEST_SKIP()` treatment (above) was itself superseded 2026-10-12 (user decision: redesign
  the check around a mode with real energy, rather than make the old mode's exemption more visible).
  The dominant eigenvalue (`lambda = -1`, Newmark's acceleration passenger) is *retained* only as a
  cheap growth-rate sanity check (`EXPECT_LE(eig.rho, 1.0 + 1e-6)`) with no energy-budget assertion —
  it carries no energy of its own in either policy (measured in round 3: `E0` is within 1-2 orders of
  magnitude of its own eigenvector-accuracy noise floor), so no closure check on it could ever be
  trustworthy. The budget is instead evaluated on **the beam's own oscillatory mode**: among the full
  spectrum's complex-conjugate eigenvalue pairs (printed once per run), the one whose argument is
  closest to `omega1*dt` (the phase a free oscillator at the beam's own fundamental frequency would
  advance in one coupling step) and whose two-part energy `Q(Re v)+Q(Im v)` exceeds `1e-6` of the full
  spectrum's total — a criterion applied in code, not chosen by trial. (Measured: `omega1*dt` = 9.00
  deg/step; the discrete beam-alone estimate is 8.09 deg/step; the selected coupled-system mode sits
  at 4.39 deg/step, roughly half of either — consistent with fluid ADDED MASS lowering the coupled
  system's effective frequency, not a selection error.) Since a real coupling period cannot start
  from a complex state, the check runs `run_one_coupled_period_with_budget` on the eigenvector's real
  and imaginary parts SEPARATELY (`x`, `x_im`) and combines them: for a real matrix `M` with
  `M v = lambda v` and ANY quadratic form `Q`, `Q(M x) + Q(M x_im) = |lambda|^2 * (Q(x) + Q(x_im))`
  exactly (the cross term between the two parts cancels in the sum) — verified numerically on this
  `M1`/mode to a relative difference of `7.3e-17` (machine precision), and confirmed to be the
  phase-independent quantity to use: `Q(x)+Q(x_im)` agreed to 7 significant figures between the two
  contraction policies (`3.871039808e-04` vs `3.871040e-04`) even though the individual real/imaginary
  energies differed between them. The tolerance is DERIVED (Cauchy-Schwarz on the energy inner
  product, full derivation in the code comment) from each part's own measured `eig_quality` (residual
  against the correct two-part eigen-prediction, not simply `lambda*x`) and a measured energy-density
  proxy, not chosen: `2|lambda| sqrt(E0) sqrt(Q(r)) + Q(r)`. A non-vacuity assertion
  (`E0/(eig_quality^2 |x|^2) > 1e6`) fails loudly, rather than passing silently, if this test ever
  lands on an energetically-empty mode again — the passenger mode's own worst measured ratio was
  `~54`; this mode measures `~1e12`, ten-plus orders of magnitude higher, so the `1e6` bar has wide
  margin on both sides.

  The pre-existing beam-side/fluid-side identity checks further down the same test function now run
  on the real part's own production period and are reachable/load-bearing again (see
  `build-evidence/mutations/fsi.toml`). Round 4 first recalibrated the beam-side identity's tolerance
  factor from `1.0` to `1.5` because factor `1.0` narrowly failed on this less-favourably-resolved
  mode (`1.029` contracted); that recalibration was reverted round 5 (a tolerance widened specifically
  because one run failed is a widened tolerance regardless of how the new number is framed), and
  `eig_quality`/`CoupledCase::tol`/the beam stiffness matrix's condition number were each measured and
  ruled out as the floor's mechanism, leaving a real `~1.05e-12` failure under Release+`OFF` as an open
  finding.

  **Root cause found and fixed 2026-10-12 (final FpPolicy round, `KNOWN_ISSUES.md` D34).** The floor
  was neither `eig_quality` nor the solver's own conditioning: it was this test's own bookkeeping.
  `build_fsi_graph()` does not hand the coupling graph `c.dt` itself — it quantizes it first, through
  this file's own `rationalize_dt()` (an 8-decimal-place rational: `Rational{llround(dt*1e8), 1e8}`),
  and the production run every downstream identity is evaluated against physically advanced on THAT
  quantized value, not on the literal `c.dt`. (`FUSE`'s core scheduler was considered and ruled out
  directly: `fuse::StepContext::dt()` and the coupling graph's own component interval already carry the
  quantized `Rational` — nothing downstream of `rationalize_dt()` ever sees the literal `c.dt` again.)
  The beam-side identity's `dE_beam_naive = c.dt * beam_wall_power` line used the literal value while
  the run it was being compared against used the delivered one; measured directly, `mismatch = |c.dt -
  dt_delivered| * |beam_wall_power|` to 7-8 significant figures in both policies (`c.dt - dt_delivered
  = -1.613339e-09` on this mode), and substituting the delivered dt into a fresh, isolated beam step
  reproduces the production run's own `w1`/`wd1` to exactly `0.0` in both policies. Fixed by computing
  `dt_delivered = rationalize_dt(c.dt).to_double()` and using it in place of `c.dt` in
  `dE_beam_naive`; the mismatch drops to round-off (`4.07e-20` ON, `6.78e-21` OFF; relative to `E0`:
  `2.78e-16` ON, `3.03e-17` OFF). The fluid-side budget's `measure_full_energy_identity_general` call
  has the same class of defect (passes the literal `c.dt` to an independently-built `cn_step`, compared
  against a production state that advanced on `dt_delivered`) but was not fixed this round — only the
  beam identity was in scope.

  The tolerance was re-derived from scratch rather than loosened: it no longer uses `eig_quality` at
  all (measured, this round, to be the wrong scale for an identity computed entirely from one
  production run's own data). It now bounds the floating-point evaluation error of the identity's own
  arithmetic directly, via Higham's recursive-summation/inner-product result (*Accuracy and Stability
  of Numerical Algorithms*, 2nd ed., Sec. 3.1: an `n`-term sum/inner product's evaluation error is
  bounded by `gamma_n * (sum of absolute products)`, `gamma_n = n*u/(1-n*u)`), plus the beam's
  `Keff2=(dt/4)*K+(rho_s h_s/dt)*I` solve's own backward-stable (direct sparse LU) forward error
  (`kappa_2(Keff2)*u`). The quadratic form `PE=0.5*w^T*K*w` behind the energy terms cancels
  substantially against its own evaluation-error scale (measured ~44x: `sum|K_ij*w_i*w_j|` versus
  `2*|PE|`), so the bound uses that larger, uncancelled sum rather than `|PE|` itself — using `|PE|`
  there would understate the true floor by the same ~44x. Both `nnz(K)` (`19`) and `kappa_2(Keff2)`
  (`1.222628`) are computed directly in the test from an independent replica of `BeamSolver`'s public,
  documented stiffness construction (confirmed byte-identical, `||diff||=0.0`, to the real, private
  `K_` for this configuration) — no private access, no cited external constant. The resulting overall
  multiplier `c` (`max` of three per-term multipliers, never fitted) is `11402.13` (ON) / `23444.91`
  (OFF); the derived tolerance is `1.031e-16` (ON) / `1.978e-16` (OFF), which is `2534.7x` / `29183.6x`
  looser than the now-round-off-level observed mismatch — a loose bound, honestly reported as such,
  not tightened to fit.

  Both `build-evidence/mutations/fsi.toml` rows for this identity were re-proved on the fixed code: the
  pre-existing `fsi-energy-diag-beam-identity-dx-beam-coefficient-dropped` row now kills the test in
  BOTH policies (previously it could only be proved under `ON`, since the `OFF` baseline itself failed
  before this round); a new row,
  `fsi-energy-diag-beam-identity-dt-literal-not-delivered`, reverts the fix (`dE_beam_naive` back to
  literal `c.dt`) and is killed in both policies by the new tolerance, with the mismatch exceeding it
  by `1.058e+04x` (ON) / `5.319e+03x` (OFF).

Full `ctest`, both targets, after the final round (measured 2026-10-12):

| `CMAKE_BUILD_TYPE` | `FUSE_FP_REPRODUCIBLE` | `num.accelerator` | `verification.fsi_energy_diag` |
|---|---|---|---|
| Release | ON  | Passed (2/2 cases OK) | **Passed** (`FullBudgetOnTheDominantModeCloses` OK, closure mismatch/tolerance `2.38e-3`, non-vacuity `1.0e12`, beam-identity mismatch/tolerance `3.95e-4`, margin `2534.7x`) |
| Release | OFF | Passed (2/2 cases OK) | **Passed** (closure mismatch/tolerance `1.36e-1`, non-vacuity `4.5e12`; beam-identity mismatch/tolerance `3.43e-5`, margin `29183.6x` — previously FAILED here at `1.029`, see above) |
| Debug   | ON  | Passed (2/2 cases OK) | **Passed** (identical numbers to Release+ON, as D30's own mechanism predicts) |

All three required configurations now pass, including the previously-failing Release+`OFF` case. The
beam-side identity's floor is no longer open: it was this test's own dt-literal/dt-delivered
bookkeeping, now fixed, with a Higham-derived (not fitted, not eig_quality-based) tolerance in its
place. Two related, same-class findings are carried forward rather than fixed this round: the
fluid-side budget's `measure_full_energy_identity_general` call (described above), and
`examples/fsi_channel/main.cpp`'s own `rationalize_dt()` (a coarser, 6-decimal-place quantization of
the same kind), whose literal, un-quantized `dt` is used in exactly two places after the graph is
built: the step count `llround(t_end / dt)` (`main.cpp:259`) and the output CSV's `time` column,
`(step + 1) * dt` (`main.cpp:279`). Neither enters a dynamics computation — the components take their
step and time from the scheduler's `StepContext`, and the inflow is a function of `y` alone — so the
consequence is that the simulated end time is `n_steps` times the quantized step rather than
`t_end`, and the CSV's time axis carries the same relative error, accumulating linearly. The example
computes no energy budget or identity of its own. Its magnitude is
NOT the `~4.6e-7` figure this round's fix measured (that number is specific to this test's own
`c.dt`/8-decimal-place grid): measured directly for `main.cpp`'s own 6-decimal-place grid, the
quantization is EXACT (`0.0` relative error) for the tool's default `--dt 0.01`, since `0.01` lands
exactly on a `1e-6` grid point; for a `--dt` that does not (e.g. `0.0035104283866610403`, this test's
own value, used here only to exercise the grid, not as a claim about what `main.cpp` is run with), the
relative error measures `~1.22e-4` — about 265x LARGER than this round's own figure, consistent with
`main.cpp`'s grid being 100x coarser (`1e6` vs `1e8`).

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
| 2026-10-08 | **MOAB, 12 of 12 rows, no failures**: `parallel.moab_placed_view_mpi.{1..5}` (0.06–0.33 s) alongside `backend.moab_mesh`'s 47 cases and the five `parallel.remap_halo_mpi` legs (`KNOWN_ISSUES.md` D5). `parallel.window_runner_mpi.{2,3}` pass (D19). `parallel.connect.mpmd.*` pass throughout; `parallel.connect.accept.*` fails on this MPICH `ch3:nemesis` build when both programs share a node — not a library defect (D23). |
| 2026-10-09 | `verification.brusselator.participants.mpmd.3` passes: `max_window_balance_error`/`global_balance_error`/`max_window_exchange_error` within threshold, final means within 2 ulp (`mean_u`) and 5 ulp (`mean_v`) of the single-process example (D25). The Schur-complement row-band solver's one-row-band refusal is found to strand a peer inside the Schur-assembly allgather when decided from local data alone, and fixed to decide from `row_ownership` (a pure function every rank already holds) instead — no collective needed for the refusal itself (`docs/dev/schur-slab-solver.md`, commit `3b43bd0`). |
| 2026-10-10 | On `main` `e76881c`, eight rows in one `ctest` invocation, no failures: `parallel.linalg_mpi.2` (0.40 s), `parallel.schur_slab_mpi.2` (0.26 s), `.3` (0.07 s), `parallel.krylov_mpi.2` (0.31 s), `parallel.brusselator_grid_mpi.2` (0.29 s), `parallel.native_halo_mpi.2` (0.37 s), `.3` (0.08 s), `verification.brusselator.participants.mpmd.3` (2.34 s). The first host run of each of the first seven, two days earlier, had found one defect per track (a refusal returned on one rank's data before a collective; crossed `MPI_Sendrecv` tags in a test's halo exchange; a test combining vectors of different ghost layouts), all fixed before this run. |
| 2026-10-10 | `verification.brusselator.participants.mpmd.5` passes (D25): U on two ranks, V on two, the coupler on one, 1.37 s. Bit-identical to the single-process example at 3 ranks (0 ulp in both means); the 5-rank row differs by 8.9e-14 in `mean_u` and 9.4e-14 in `mean_v` (relative 5.6e-14/4.7e-14) — the decomposed Schur solve's own round-off, consistent with the in-process 2-partition bound of 4.5e-14. The species-core decomposition track's own in-process harness (`test/verification/brusselator_species_core_decomposition_test.cpp`, 9 tests) measures the decomposed `Schur`/`Pcg`/`Gmres` diffusion solvers against a single-rank reference to 2.7e-15–5.9e-14. Full tier-0 suite unaffected throughout (179/179; see "Tier-0 row count" below). |

The 2026-09-16 run is what makes the rank-agreement mechanism verified rather
than asserted: `parallel.epoch_reconcile` fails one rank's component mid-step
and requires every other rank to return a failure instead of blocking. The
2026-10-08/09/10 runs are the first on this machine to exercise a THIRD
process boundary at once (U, V and the coupler each their own executable,
`verification.brusselator.participants.mpmd.{3,5}`) and the first to run the
native distributed linear-algebra containers, the Krylov solvers and the
Schur-complement row-band solver under a real `MPI_COMM_WORLD` rather than
the in-process fakes `docs/dev/linalg-distributed.md`/`linalg-krylov.md`/
`schur-slab-solver.md`/`native-halo.md` describe: `parallel.linalg_mpi.2`,
`parallel.krylov_mpi.2`, `parallel.schur_slab_mpi.{2,3}`,
`parallel.brusselator_grid_mpi.2` and `parallel.native_halo_mpi.{2,3}` all
pass (0.07–0.40 s each) — the 2026-10-10 row in the table above is that run,
on `main` `e76881c`, together with `mpmd.3`.

## Tier-0 row count, recounted

The count depends on the configuration, so two measured figures are given
rather than one. **179** is the tier-0 count in the configuration every commit
this week was verified in: a clean export, Debug, Eigen linear-algebra
backend, no MPI, no MOAB, no LowFive (`ctest` reports "100% tests passed, 0
tests failed out of 179"). A `ctest --show-only=json-v1` query against the
full host configuration (`build-all`, MPI, MOAB and LowFive all enabled)
counts **181** non-`mpi`-labelled rows out of 289 total; the difference is the
serial MOAB and LowFive rows that configuration adds, not a disagreement. The call-site count (`fuse_add_gtest`/`fuse_add_mpi_test`/
`fuse_add_upstream_main_test`/`fuse_add_layout_test`/`fuse_add_omp_legs` in
`test/CMakeLists.txt` and `examples/CMakeLists.txt`) is 110, not 90 — the
discrepancy is `fuse_add_mpi_test`/`fuse_add_layout_test` each expanding to up
to five `ctest` rows (one per rank count) from one call site, so "calls" and
"rows" are never the same number and neither figure is wrong, they are
answering different questions. Recounted from the build tree, not asserted
from either prior figure.

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
