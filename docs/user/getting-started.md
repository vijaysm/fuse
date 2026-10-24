# Getting started

This page takes you from a fresh clone to a running coupling, and shows you how to read what it
prints. It assumes a C++20 compiler and CMake. It does not assume MPI, a GPU or any other
dependency. Allow 10 to 20 minutes for the first build; running the examples takes under a second.

## 1. Build the two smallest examples

```bash
git clone https://github.com/vijaysm/fuse.git
cd fuse

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFUSE_ALLOW_FETCH=ON
cmake --build build --target fuse_example_mms_pair fuse_example_dn_heat
```

`-DFUSE_ALLOW_FETCH=ON` lets CMake build pinned copies of Kokkos, fmt, GoogleTest, ArborX and Eigen
if you do not have them. If you already have Kokkos installed, leave it off and add
`-DCMAKE_PREFIX_PATH=/path/to/kokkos`. Naming two targets builds only what they need. Building
everything (`cmake --build build`) also compiles the whole test suite and takes much longer. See
[](build.md) for every option.

## 2. Run a coupling

`mms_pair` couples two components that take different time steps and checks the result against a
closed-form solution.

```bash
./build/examples/fuse_example_mms_pair
```

```
periods                   : 100
instructions              : 1600
conservation checks       : 100
violations                : 0
worst conservation error  : 0
t                         : 0.1
L2 error left  (own disc.): 1.30225e-06
L2 error right (coupled)  : 0.00114793
L2 error right (no exch.) : 0.127525
coupled / uncoupled       : 0.00900155
```

(The last digits can differ slightly between platforms.) The program exits with status 0 only if its
checks pass, so a quiet exit is a result, not an absence of one.

How to read it:

| Line | Meaning |
|---|---|
| `periods`, `instructions` | 100 coupling cycles; 16 scheduled instructions per cycle |
| `conservation checks`, `violations` | A transfer declared *extensive* is checked to conserve its integral on every cycle |
| `L2 error left` | The first component's own discretisation error. It has nothing to do with the coupling |
| `L2 error right (coupled)` | The second component's error. It comes from exchanging data across a one-cycle lag |
| `L2 error right (no exch.)` | The same component with the exchange removed, as a control |
| `coupled / uncoupled` | About 0.009: the exchange removes over 99% of the error |

The example is not only run but *gated*: it fails if either component misses its error bound, if the
exchange is not decisively better than the control, or if no conservation check ran. That is the
standard every example in the repository holds itself to ([](examples.md)).

## 3. Look at what FUSE built

A FUSE program describes the coupling, then compiles it. `--dump` prints the compiled result without
running anything:

```bash
./build/examples/fuse_example_mms_pair --dump
```

The output has two parts. `=== IR ===` is the dataflow graph: one chain of operators for the one
field transfer.

```
node 1   Source       -> %0  "left.T"
node 2   Sample       %0 -> %1  "T_link/sample"
node 3   Accumulate   %1 -> %2  "T_link/accumulate" {accum=Mean}
node 4   Transport    %2 -> %3  "T_link/transport" {0->1}
node 5   Epoch        %3 -> %4  "T_link/epoch" {reason=after-transport}
node 6   Remap        %4 -> %5  "T_link/remap"
node 7   Constrain    %5 -> %6  "T_link/constrain"
node 8   Diagnose     %6 -> %7  "T_link/diagnose"
node 9   Sink         %7 -> %8  "right.T_bc"
```

(Abridged. The real lines also carry each value's time level, units and ranks.) Reading it top to
bottom is following one field from the producing component to the consuming one:

| Operator | What it does here |
|---|---|
| `Source` | Reads the field `T` that `left` provides |
| `Sample` | Records it at each of `left`'s four sub-steps in the cycle |
| `Accumulate` | Averages those samples over the cycle (`Mean`), because `right` takes one long step |
| `Transport` | Moves the data from `left`'s ranks to `right`'s (a plain copy in a single-process run) |
| `Epoch` | Makes every rank agree the move succeeded, so a failure cannot leave peers waiting |
| `Remap` | Maps between the two meshes (here an identity, since they coincide) |
| `Constrain` | Checks that the integral is conserved |
| `Diagnose` | Records values for inspection; nothing flows back |
| `Sink` | Writes the result into the field `right` requires |

You wrote none of these operators. You wrote declarations (the `Mean` you asked for, the fact that
`T` is conserved, which ranks each component runs on) and FUSE derived the operators from them. `=== PLAN ===` lists the instructions that actually run each cycle, in
order, with their times as exact fractions (`t=1/4000`, `t=1/1000`). When a coupling does something
you did not expect, this dump is the first place to look, because it is what runs.

## 4. Run a coupling that iterates

`dn_heat` couples two heat-conduction domains through a shared interface, so each needs the other's
answer to finish its own step. FUSE iterates the pair to convergence inside one *implicit coupling
group* and compares several ways of accelerating the iteration.

```bash
./build/examples/fuse_example_dn_heat
```

The table it prints compares the same problem under no coupling at all, a plain fixed-point
iteration (Picard), Aitken relaxation, interface quasi-Newton (IQN-ILS) and Newton-Krylov:

```
             periods   sweeps  !conv    L2 err u/K  max err u/K    L2 err q     max q
  explicit      20       20     20   1.193e+02  2.105e+02   7.751e+00  1.423e+01
  picard        20    14918      0   7.910e-07  1.582e-06   1.089e-06  2.178e-06
  aitken        20      938      0   4.184e-07  8.240e-07   2.852e-07  4.300e-07
  iqn-ils       20      201      0   4.923e-14  5.684e-14   3.627e-14  4.974e-14
  ...
all gates passed
```

Read the `sweeps` column: the same converged answer costs 14918 sweeps with Picard and 201 with
IQN-ILS. The `explicit` row, which exchanges once per step without iterating, is wrong by 119 K.
[](convergence.md) and [](dn-heat.md) explain how convergence is decided and what each accelerator
does.

## 5. Run the tests

```bash
ctest --test-dir build -R "^example\.(mms_pair|dn_heat)" --output-on-failure
```

For the whole suite, build everything first and run `ctest --test-dir build -LE bench`. See
[Running the tests](build.md#running-the-tests) for labels that select subsets.

## 6. Write your own

A FUSE program has five stages. The same five appear in every example:

1. **Describe**: build a `fuse::CouplingGraph` of meshes, regions, components, fields and channels.
2. **Load**: give each mesh to a backend.
3. **Compile**: `validate()` reports every defect, then `lower()` makes the dataflow graph.
4. **Schedule**: `schedule()` turns it into a fixed instruction order and proves it cannot deadlock.
5. **Run**: bind each component's storage, `initialize()`, then `run()`.

Your physics goes in a class that implements `fuse::Component`: register your arrays once, then
advance them in place. [](quickstart.md) walks through the five stages with the real code from
`mms_pair`, and [](cookbook.md) has a recipe for each common task.

## Where next

| You want to | Read |
|---|---|
| Understand the model before writing code | [](concepts.md) |
| Copy a working snippet for a task | [](cookbook.md) |
| Look up a word | [](glossary.md) |
| Couple an existing C or Fortran code | [](c-abi.md), [](fortran.md) |
| Run on many ranks or across executables | [](placement.md), [](multiple-executables.md), [](power-users.md) |
| Fix an error message | [](troubleshooting.md) |
