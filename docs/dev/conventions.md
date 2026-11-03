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

## Character encoding

Every tracked source file is pure ASCII -- code, comments, CMake, YAML, TOML, shell, and prose in files other than Markdown. `tools/check-ascii.py` enforces it: it walks `git ls-files`, skips `*.md` (Markdown prose is reviewed by eye and may carry a section sign, an em dash, or an author's own name) and skips `upstream/` (vendored parent trees staged for removal, not edited), decodes every remaining tracked file as UTF-8, and refuses with `file:line:col` and the Unicode character name for the first code point above U+007F it finds in each file. A tracked file that is not valid UTF-8 at all is refused too, unless its path matches one of the binary-fixture globs named in the script (the golden `.bin` snapshots under `test/golden/` and `python/tests/golden/`) -- an undecodable file that is not on that list is far more likely to be a newly added fixture nobody told the gate about than a deliberate binary artifact, so the default is to refuse, not to guess.

Run it the same way as the other static gates:

```
python3 tools/check-ascii.py
```

`tools/ascii-fix.py` is the mechanical fix for a violation: it replaces each non-ASCII character with the meaning-preserving ASCII spelling recorded in its own docstring (a section sign becomes the word "Section ", an em dash becomes "--", and so on), except for the SI-unit separator in `include/fuse/base/units.hpp`, whose few call sites are listed there explicitly rather than matched by a general rule, because the same character (U+00B7 MIDDLE DOT) means two different things in this tree depending on where it appears. Re-run it whenever a new non-ASCII character lands in a tracked file; it is a no-op on an already-clean file.

## Errors

`FUSE_ASSERT` for invariants (checked in every build), `FUSE_THROW(Type, …)` for recoverable errors,
`FUSE_REFUSE("E-CODE", …)` for deliberate refusals. Never a bare `assert`, never a bare `throw`, and
never the parents' `KARMA_THROW`/`SOMA_ASSERT` — the gate rejects all three. Design and the
debugger hook: [](debugging.md).

## Output and logging

`fmt` is the only formatter. No `iostream`, and no `printf` in library code. Any path that must not
allocate formats into a caller-provided buffer with `fmt::format_to`.

There is no process-wide logger. `fmtlog` was fetched (`cmake/FuseDependencies.cmake`) and a
`fuse_resolve_logging()` macro built an interface target for it, but nothing in `include/` or
`src/` ever included it or called the macro — removed rather than left as dead fetch
weight. `fuse::initialize`/`fuse::finalize` set the process-wide error policy and the rank reported
in diagnostics (`include/fuse/base/error.hpp`); they do not start or drain a logger.
Diagnostic text today goes through `FUSE_THROW`/`FUSE_REFUSE` (errors) or directly through `fmt` at
the call site (host-only CLI examples' status lines). Should a process-wide logger be added later,
it must still meet the three constraints this section used to state for `fmtlog`, because they are
properties of *any* logger this project would accept, not of that one library:

- **Not callable from device code.** Device-side diagnostics use `Kokkos::printf`.
- **No allocation after setup** — per-thread buffers pre-touched at initialisation
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

## Canonical text and the locale

Any function whose output is byte-compared — the IR dump, the plan dump, a golden file — must not
format through the host's locale. `fmt` gives that by construction and is the default. A stream-based
writer must pin `std::locale::classic()` on **every** stream it creates, and `tools/check-invariants.py`
verifies the pin rather than banning `<sstream>`: banning the header would have been satisfied by a
file formatting through `std::to_string`, which reads the global locale and would have passed silently.

Tightening that check immediately found a real defect carried in from the parent: the IR dump pinned
the locale for doubles and left the two dump builders unpinned, so their integer output would be
digit-grouped (`coupler_elements=16.384`) under a grouping locale — in a file whose bytes three golden
tests compare.
