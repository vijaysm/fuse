# Layering

Dependencies point down only. A layer may use the layers below it and must not name anything above.

```text
interface   provider ABI · participant ABI · registry/adapter host · Fortran · Python · YAML · dynamic loading
backend     mesh · remap · transport · solver · persisted maps
runtime     Runner · field binding · transactions · checkpoint sets · Kokkos kernels
num         waveforms · temporal bases · space-time projection · accelerators · convergence metric
schedule    clocks · slabs · expansion · acyclicity proof · epochs · ExecutionPlan
ir          operator DAG (SSA) · lowering registry · transform chain · golden IR
model       semantic model · coupling graph · canonical descriptor · compatibility · inference
base        Rational · units · TimeLevel · SemanticVersion · ids · Result/Diagnostics · byte writer
```

`physics-kit/` is a separate optional component. It **depends on FUSE and never the reverse** — that
direction is what keeps FUSE adoptable by a code that already has its own discretisation.

## Why the bottom half is dependency-light

`base` through `num` need only a C++20 compiler, Kokkos and a linear-algebra backend. That is not
minimalism for its own sake: it is what makes the fast CI leg fast, and what lets a code with neither
MOAB nor PETSc use the coupler. MPI enters at `runtime`; everything heavier enters at `backend`,
behind one directory and one build option each.

## Enforcement

```{literalinclude} ../../tools/check-layering.py
:language: python
:start-after: "# --- rules"
:end-before: "# --- end rules"
```

The script runs as a CI gate on every push. A new layer or a new permitted dependency is a one-line
change there, reviewed like any other.

## A boundary the gate moved

`model` declares WHAT was asked for; `num` computes it. The inherited graph put its iteration
region's convergence criterion and acceleration choice in the numerics layer and reached up for them,
which inverts the dependency — the layering gate rejected the port on the first run. The configuration
types (`ConvergenceMeasure`, `ConvergenceCriterion`, `AccelKind`, `AccelConfig`, and the
`fuse::Converged` named constructors) now live in `fuse/model/iteration.hpp`; the algorithms (the
`Accelerator` implementations) stay in `fuse/num`, depending on the model rather than the reverse.
