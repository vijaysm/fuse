# One interface for linear operations

A remap between non-matching meshes, a transport that moves values between
ranks, and an interpolation in time are the same kind of object: each takes a
number of source values, produces a number of target values, is linear, and has
an adjoint. `fuse::LinearOperator` (`include/fuse/backend/linear_operator.hpp`)
is that object.

Before it, those three were unrelated code with three shapes, so the properties
a coupled run depends on were checked for the remap and assumed for the rest.
They are now checked once, for every implementation, by
`test/backend/operator_conformance_test.cpp`.

## What an implementation promises

```cpp
class LinearOperator {
  std::size_t rows() const noexcept;   // target entities
  std::size_t cols() const noexcept;   // source entities
  Result<void> apply(x, y, components) const;             // y = A x
  Result<void> apply_transpose(y, x, components) const;   // x = A^T y
  Result<void> row_sums(out) const;
  Result<void> col_sums(out) const;
  OperatorClaims claims() const noexcept;
  std::uint64_t signature() const;
};
```

`apply_transpose` is **required**, not optional. An operator that cannot supply
its adjoint cannot appear in the monolithic solver path, which forms its
off-diagonal Jacobian blocks from these operators — and a transpose that is only
approximately the adjoint gives a Newton iteration that converges to the wrong
answer while looking healthy.

## Claims, not a fixed list of properties

An operator states what it guarantees, and the fixture holds it to exactly
that:

| Claim | Meaning | Usual holder |
|---|---|---|
| `preserves_constants` | row sums are 1 — a constant field is reproduced | interpolation |
| `conservative` | column sums weighted by measure preserve the integral | conservative remap |
| `monotone` | no new extrema | low-order, non-negative weights |
| `rank_local` | touches no remote data | everything except transport |

The first two are the classic pair a remap satisfies **one** of, which is why
they are separate claims rather than a single "accurate" flag. `WeightsOperator`
derives its claims from the audit the weight builder recorded when it built the
matrix, so a claim is a statement about *that* matrix — including one reloaded
from disk — and not about the method that produced it.

A claim survives composition only if **both** factors make it. That is the
mathematically correct rule (a conservative map composed with a
non-conservative one does not conserve), and it is the rule a test has to reach
a specific regime to check: with no conservative operator in the fixture, a
mutation that inherited the claim from *either* factor survived the whole
suite. The fixture now carries a genuinely conservative operator for that
reason.

## Signature

A 64-bit hash of everything that determines what the operator computes — shape,
triplets, their order — and nothing incidental. The audit fields are
deliberately excluded: they are measurements *about* the matrix, so hashing them
would make a map that computes exactly the same thing look different whenever
the builder's diagnostics changed, and every persisted map on disk would be
rejected.

Composition mixes the factor signatures **non-commutatively**, because `A∘B` and
`B∘A` are different operators and a plan that confused them would reuse the
wrong persisted map.

## Why the step loop does not call through this interface

The runtime's inner loop calls the sparse multiply directly rather than through
a virtual `apply`. That is deliberate. The interface earns its place at init
time — composition, the consistency and conservation checks, the signature that
matches a persisted map to its mesh pair, and the solver seam — while the step
loop runs one instruction per operator and would pay an indirection for nothing.

What matters is that there is exactly **one** sparse multiply in the library:
`WeightsOperator::apply` delegates to the same `apply_weights` the runtime
calls, rather than reimplementing it. A second implementation would be a second
thing to keep in step, and the adjoint identity would then be testing the copy.
