# Worked examples

Two complete couplings live in `examples/`. They are the only code in the tree
that drives the library the way you would — build a graph, lower it, run it —
rather than assembling a test fixture, and both are run by `ctest`, because an
example nobody runs rots within a release.

Neither is gated on "did not crash". Each exits non-zero when its *physics* is
wrong, which is what makes them worth running.

## `mms_pair` — explicit coupling, verified against a manufactured solution

Two heat domains exchanging an interface flux, with a manufactured solution
giving the exact answer. The binary fails when the coupled error misses its
analytic gate, when no conservation check ran, or when the coupled error is not
decisively better than a run with the exchange removed — the control that
proves the number moves when the coupling breaks.

Measured, 100 coupling cycles to t = 0.1:

| Quantity | Value |
|---|---|
| L2 error, own discretisation | 1.30e-06 |
| L2 error, coupled | 1.15e-03 |
| L2 error, exchange removed | 1.28e-01 |
| coupled / uncoupled | **0.0090** |
| conservation violations | 0 of 100 checks |

## `dn_heat` — implicit coupling, Dirichlet–Neumann, iterated to a fixed point

A classical Dirichlet–Neumann pair of conduction domains inside an implicit coupling
group, with a derived checkpoint set, a convergence criterion covering both
field transfers, and quasi-Newton acceleration. This is the example that exercises
rollback, acceleration and convergence together.

Measured over 20 coupling cycles:

| Quantity | Value |
|---|---|
| implicit vs explicit, cold start | 1.19e+02 K → 7.91e-07 K (**1.5e+08×**) |
| implicit vs explicit, asymptotic | 2.31e+00 K → 2.69e-07 K (8.6e+06×) |
| implicit vs severed exchange | 3.04e+01 K → 7.91e-07 K (3.8e+07×) |
| Picard sweeps | 14918 |
| Aitken | 938 (**15.9×** fewer) |
| IQN-ILS | 211 (**70.7×** fewer) |
| accelerators agree to | 1.6e-06 K |
| rollback verifications | 29796 |
| conservation violations | 0 of 29836 checks |
| reproducibility | bit-identical over 32 values |

The severed-exchange row is the one that matters: it is what the answer
degrades to when the coupling is broken, so it is the evidence that the
headline factor is the coupling doing work rather than the physics being easy.

## Seeing the compiled plan

Both accept `--dump`, which prints the operator IR and the flat execution plan
without running anything:

```
./examples/fuse_example_dn_heat --dump
```

The `ctest` entries for the dump path assert on its *output* rather than the
exit status, because an empty dump still exits 0.
