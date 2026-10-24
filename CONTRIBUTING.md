# Contributing to FUSE

Thank you for improving FUSE. This page covers how to build and test a change, the rules a change
must satisfy, and how to submit it.

## Learn by example

Before changing code, see how a task is done with FUSE today:

- [Getting started](docs/user/getting-started.md): build and run a first coupling.
- [Quickstart](docs/user/quickstart.md): the five stages with real code.
- [Graph programming](docs/user/graph-programming.md), [Writing a component](docs/user/component-interface.md)
  and [Writing a driver](docs/user/drivers.md): the three things most contributions touch.
- [Cookbook](docs/user/cookbook.md): a recipe for each common task, with links to full examples.
- [Examples](docs/user/examples.md) and the programs under [`examples/`](examples/): complete
  couplings that double as tests.
- [Concepts](docs/user/concepts.md) and the [glossary](docs/user/glossary.md).

## Set up

```bash
git clone https://github.com/vijaysm/fuse.git
cd fuse
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DFUSE_ALLOW_FETCH=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench
```

See the [build guide](docs/user/build.md) for MPI, MOAB, PETSc, HDF5, Python and the other options.
If your change touches an optional feature, build with that feature on.

## Checks that must pass

The static checks need only Python and run in under a second each:

```bash
python3 tools/check-layering.py
python3 tools/check-invariants.py
python3 tools/check-ascii.py
python3 tools/check-throws-labels.py
python3 tools/verify-import.py
python3 tools/mutate.py --check-anchors
```

Then the tests, and the documentation build, in which a warning is an error:

```bash
ctest --test-dir build --output-on-failure -LE bench
pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

[Continuous integration](https://github.com/vijaysm/fuse/actions/workflows/ci.yml) runs the same
checks, plus a matrix of serial, MPI, undefined-behaviour-sanitizer and no-exceptions builds, with
both linear-algebra backends. A pull request needs a green run.

## Rules for code

- **Layering.** Libraries form layers and may include only from layers below them
  (`tools/check-layering.py` enforces it). Read [docs/dev/layering.md](docs/dev/layering.md) before
  adding an include across layers.
- **Invariants.** [docs/dev/invariants.md](docs/dev/invariants.md) lists rules that are checked
  mechanically. Do not weaken a check to make a change pass.
- **Conventions.** Follow [docs/dev/conventions.md](docs/dev/conventions.md): C++20, namespace
  `fuse::`, errors returned as `Result` or reported as `Diagnostics` with a stable `E-...` code, no
  silent fallbacks.
- **ASCII.** Source files are ASCII-only. Markdown files may use other characters.
- **Errors are tested.** A new refusal needs a test that triggers it and checks its code. Tests that
  expect an exception carry the label `throws`.
- **Distributed code fails together.** A rank that fails must agree the failure with its peers
  instead of returning alone. See [docs/dev/rank-agreement.md](docs/dev/rank-agreement.md).
- **Tests prove something.** A test should fail if the behaviour it names is broken. For a change
  to a checked operator, add evidence with `tools/mutate.py` as described in
  [docs/dev/testing.md](docs/dev/testing.md).
- **Licence.** New files are BSD 3-Clause, like the rest of the repository. Do not add code under an
  incompatible licence, and list any new third-party component in `THIRD-PARTY-NOTICES.md`.

## Adding an example

An example is a program in `examples/<name>/` that:

1. couples real components, not stubs;
2. checks its own result against a known answer, a control, or a conservation count;
3. exits non-zero when a check fails;
4. is registered with `ctest`, so it runs on every build;
5. has a page in `docs/user/` that includes its code with `literalinclude`, so the page cannot
   drift from the program, and a row in [docs/user/examples.md](docs/user/examples.md).

Then link it from the [cookbook](docs/user/cookbook.md) under the task it demonstrates.

## Documentation

User documentation lives in `docs/user/` (Sphinx with MyST) and is written for people using FUSE.
Keep it current with the code:

- Describe what FUSE does now. Do not put plans, work in progress, task or ticket names, dates, or
  to-do lists in the README or user documentation.
- Prefer a short, runnable example to a description. Include real source with `literalinclude`.
- State limitations plainly where a user would hit them.
- Say what the automated tests cover and what they do not.
- Edit `docs/`, never the wiki. The wiki is generated from `docs/` by `tools/docs-to-wiki.py`
  and republished when `main` changes. Run the script locally to preview it.

## Submitting a change

1. Open an [issue](https://github.com/vijaysm/fuse/issues) for anything larger than a small fix, so
   the approach can be agreed first.
2. Branch from the default branch and keep the change focused.
3. Run the checks above. Include the output of any new or changed test in the description.
4. Open a pull request that says what changed and why, and updates the documentation and the
   cookbook when behaviour users can see changes.

## Reporting problems

Open an issue with the smallest coupling that shows the problem, the `dump_text` / `plan.to_string()`
output, the error code, and the `fuse-resolved-dependencies.txt` from your build directory.

## License

By contributing you agree that your contribution is licensed under the BSD 3-Clause license in
[`LICENSE`](LICENSE).
