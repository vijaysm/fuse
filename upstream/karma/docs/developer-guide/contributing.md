# Contributing

## Before Editing

1. Inspect `git status --short --branch` and preserve unrelated work.
2. Read the public contract, active source registration, and nearby tests.
3. Confirm code is active in `src/CMakeLists.txt`; do not revive a legacy subsystem accidentally.
4. Choose the smallest change that addresses the requirement.

## Implementation Workflow

1. Add or strengthen a regression test first for behavior changes.
2. Run the focused test and confirm it detects the problem.
3. Implement the minimal production change.
4. Run focused serial and MPI tests.
5. Run the complete applicable Release suite.
6. Run `git diff --check` and inspect the final diff and status.

Follow local file style. The repository mixes legacy tab-indented code with newer four-space code and has no canonical formatter. Avoid broad formatting. Prefer modern C++20 ownership and RAII patterns in new code.

## Review Checklist

- Public behavior has a regression test.
- Inputs, dimensions, mappings, communicators, and finite values are validated.
- Failure paths do not partially mutate caller-visible state.
- Distributed failures are collective-safe and deterministic.
- Empty ranks and zero-length ownership are handled.
- Numerical tolerances are scale-aware and justified.
- Boundary signs and physical units are explicit.
- New `.C` files and tests are registered explicitly.
- MPI tests have finite timeouts; file tests have fixtures/resource locks.
- Documentation reflects public capability changes.

## Commit Hygiene

Do not stage unrelated files. Recent commit subjects use concise imperative phrases. Only commit, push, or create a pull request when explicitly requested.
