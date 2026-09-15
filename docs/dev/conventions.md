# Conventions

## Identity

Namespace `fuse::`. CMake package `fuse-coupler`. Headers `<fuse/<layer>/<name>.hpp>`. Libraries
`libfuse-coupler-<layer>`.

No `soma::`, no `KARMA::`, no `KARMA*` type prefix, no `karma_*`/`soma_*` file prefix, no `.C`
extension. Ported code is renamed on arrival ([](porting.md)); the gate rejects the legacy forms.

The C ABI exports **one symbol per ABI** returning a versioned function table — no prefixed symbol
family, no macro-generated names, no macros in public headers. The reasoning is in
[](abi-design.md).

## Naming

Enforced by `clang-tidy`, not by review:

| Kind | Form |
|---|---|
| Namespaces, functions, variables, parameters | `lower_case` |
| Types, enums, enumerators, template parameters | `CamelCase` |
| Private and protected members | `trailing_underscore_` |
| Constants and `constexpr` | `kCamelCase` |
| Macros | `FUSE_UPPER_CASE` |

The C ABI is plain C by contract and opts out of the C++ naming rules through its own
`.clang-tidy` in `include/fuse/abi/`.

## Formatting and linting

`.clang-format` and `.clang-tidy` at the repository root are authoritative, and every deviation from
the upstream defaults carries a comment saying why. The split that matters:

- **Real-bug checks fail the build**: `bugprone-*`, `mpi-*`, `concurrency-*`, narrowing conversions,
  uninitialised variables.
- **Style and modernisation only warn.**

Several checks are disabled on purpose because they fight numerical code — magic numbers *are* the
content of a quadrature table, and `easily-swappable-parameters` is inherent to `remap(src, dst, tol,
...)`. Each exclusion states its reason in the file.

## Output and logging

`fmt` is the only formatter. No `iostream`, and no `printf` in library code. Any path that must not
allocate formats into a caller-provided buffer with `fmt::format_to`.

`fmtlog` is the logger, started by `fuse::initialize` and drained by `fuse::finalize`. Three
constraints are part of the contract:

- It is **not callable from device code.** Device-side diagnostics use `Kokkos::printf`.
- Per-thread buffers are pre-touched at initialisation, so logging never allocates after setup
  ([](invariants.md) `I5`).
- Every line carries the MPI rank, and **a log line is never evidence of a collective outcome** —
  that is `Epoch`'s job ([](invariants.md) `I2`).

## Commit messages

A single-line header, then one succinct line stating the core change. A second body line only if it
is genuinely needed. Rationale belongs in the code and in these pages, where it can be reviewed and
corrected — not in an unversioned commit body.

## Vocabulary

Fixed before code carries it, because one word meaning two things cost a day of confusion once
already.

Rendezvous
: Cross-rank matching in the communication graph.

AdaptiveWindow
: Sub-stepping within a coupling window. (Formerly also called "rendezvous"; do not.)

Provider ABI
: FUSE calls the plugin.

Participant ABI
: A foreign code with its own time loop calls FUSE.
