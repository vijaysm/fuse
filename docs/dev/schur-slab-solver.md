# The exact Schur-complement row-band solver

`fuse::linalg::SchurSlabSolver` (`include/fuse/linalg/schur_slab.hpp`,
`src/linalg/schur_slab.cpp`) is an exact distributed direct solver for the
backward-Euler periodic Laplacian the Brusselator example builds
(`examples/brusselator/brusselator_physics.hpp`'s `periodic_laplacian`,
`DirectDiffusionSolver`), when that operator's rows are partitioned into
contiguous row bands across ranks instead of being factored whole on one
rank.

A note on names before the derivation: the domain-decomposition literature
calls one rank's contiguous partition a SUBDOMAIN, and every DOC COMMENT and
every PROSE sentence below says "row band" for the same thing -- the term
`examples/brusselator/brusselator_physics.hpp`'s own `NativeGrid` already
uses for this exact rank-contiguous range of `j`. `docs/user/glossary.md`
already defines "slab" as a TIME-domain concept (an interval of the common
refinement); re-using it for a SPATIAL row partition here would recreate the
"one word meaning two things" collision `docs/dev/conventions.md`'s own
Vocabulary section names as a known hazard. `SchurSlabSolver`'s own class
name and this file's own name were fixed by the task that produced them
before this note was written, so they keep "Slab"; nothing else does.

## The operator

Let `A = I - c L_h`, `c = dt*D` the step's diffusion coefficient
(`coefficient` in the interface below), `L_h` the periodic 5-point
Laplacian on an `nx * ny` cell grid, canonical index `idx(i, j) = j*nx + i`,
`0 <= i < nx`, `0 <= j < ny`, both indices periodic. Writing
`xx = nx^2`, `yy = ny^2` (the squared inverse cell widths
`periodic_laplacian` itself uses):

```
L_h[idx(i,j), idx(i,j)]         = -2(xx + yy)
L_h[idx(i,j), idx(i+1 mod nx,j)] =  xx
L_h[idx(i,j), idx(i-1 mod nx,j)] =  xx
L_h[idx(i,j), idx(i,j+1 mod ny)] =  yy
L_h[idx(i,j), idx(i,j-1 mod ny)] =  yy
```

so `A`'s own entries are

```
A[idx(i,j), idx(i,j)]           = 1 + 2c(xx + yy)     (diagonal)
A[idx(i,j), idx(i+-1 mod nx,j)] = -c*xx               (horizontal neighbour, same row)
A[idx(i,j), idx(i,j+-1 mod ny)] = -c*yy               (vertical neighbour, adjacent row)
```

`A` is symmetric (every neighbour pair is added in both directions, with
the same value) and, for the `coefficient >= 0` this operator is actually
used at, diagonally dominant -- symmetric positive definite, the condition
an `LDLT` factorisation (sparse or dense) is unconditionally stable for.

## The partition: row bands, not cells

Rows are split into contiguous bands of `j`: rank `r` owns
`[j_lo(r), j_hi(r))`, `fuse::native::rank_slice(ny, rank, nranks)`
(`src/backend/native_generators.cpp`) -- the SAME split
`examples/brusselator/brusselator_physics.hpp`'s own `NativeGrid` already
uses for its native-mesh-backed row band. `SchurSlabSolver::factor` takes
`j_lo`/`j_hi` directly (this rank's own band) AND a `RowOwnership` carrying
every rank's own `j_lo`, re-using `fuse::linalg::RowOwnership`
(`dist_sparse_matrix.hpp`) in its already-established convention --
ascending, one entry per participant, the LAST participant's own `j_hi`
being `ny` itself rather than a further entry -- reinterpreted here over
the grid's `ny` ROWS instead of its `nx*ny` CELLS. A rank's own row band is
owned-only, in LOCAL canonical order `(j - j_lo)*nx + i`; a local row index
`k = j - j_lo` ranges over `[0, local_rows)`, `local_rows = j_hi - j_lo`.

Because `A`'s only couplings across `j` are to `j+-1`, a row's neighbours
leave its own row band only at the band's own first (`k=0`) and last
(`k=local_rows-1`) row -- `A`'s horizontal couplings (`i+-1`, same row)
never leave the row band at all, by construction. The rank owning
`j_lo - 1 (mod ny)` is the PREDECESSOR in the periodic RING of row bands;
the rank owning `j_hi (mod ny)` is the SUCCESSOR. Because `RowOwnership`'s
own convention is ascending and contiguous, these are fixed by rank
arithmetic alone, with no lookup and no communication:

```
predecessor = (rank - 1 + nranks) mod nranks
successor   = (rank + 1) mod nranks
```

which also makes the ring PERIODIC in rank space: rank 0's predecessor is
the LAST rank, and the last rank's successor is rank 0 -- the periodic wrap
in `j` becomes, after partitioning, a wrap in RANK as well.

## Interior rows and interface rows

A local row `k` is INTERIOR when `1 <= k <= local_rows-2` -- both its
neighbours (`k-1`, `k+1`) are inside `[0, local_rows)`, hence owned by this
SAME rank. It is an INTERFACE row when `k == 0` (FIRST) or
`k == local_rows-1` (LAST) -- one of its two neighbours is owned by a
DIFFERENT rank (the predecessor for `k=0`'s `k-1` direction, the successor
for `k=local_rows-1`'s `k+1` direction).

**`nranks == 1` is degenerate: there are NO interface rows at all.** The
one row band IS the whole grid (`j_lo=0`, `j_hi=ny`); `k=0`'s predecessor
neighbour `j=-1 mod ny = ny-1` and `k=local_rows-1`'s successor neighbour
`j=ny mod ny = 0` are BOTH inside `[0, ny)` -- i.e. inside this same,
only, row band -- so every row is interior and the periodic wrap closes
entirely within one rank's own matrix, exactly as
`examples/brusselator/brusselator_physics.hpp`'s own whole-grid
`periodic_laplacian` already assembles it. `SchurSlabSolver::factor` at
`nranks==1` therefore builds and factors that SAME whole-grid matrix
directly -- no interface, no Schur complement, no collective at all -- and
`solve()` is a single sparse triangular solve. This is what keeps the
in-process reference (`DirectDiffusionSolver`,
`examples/brusselator/species_core.hpp`) untouched: a caller that never
decomposes its grid sees the identical operator, assembled and factored
the identical way.

**`nranks >= 2` needs `local_rows >= 2`.** A row band's first and last row
must be 2 DISTINCT rows for the block structure below to have a 2-row-wide
(`2*nx` entries) interface at all; a 1-row band would have to use its one
row as both `k=0` and `k=local_rows-1` at once, double-counting it between
the (then zero-sized) interior block and an interface block sized for 2
rows. `factor()` refuses this, `E-LINALG-SCHUR-SLAB`
(`test/linalg/schur_slab_test.cpp`'s
`SchurSlabSolverRefusal.OneRowBandRefusesWithoutFactorizing`/
`EmptyRowBandAlsoRefuses`) -- a caller sizes `nranks` so `ny >= 2*nranks`,
which this grid and partition were always meant to allow, and silently
handling the 1-row case would hide a misconfigured decomposition instead of
surfacing it. **This refusal is LOCAL, never collective**: it is answerable
from this rank's own `j_lo`/`j_hi` alone, before the one collective
`factor()` performs (the Schur-complement allgather, below) -- so one
rank's own row band being too small cannot leave its peers waiting inside a
collective that only some of them reach, the same "never return on a
locally computed condition" discipline
`DistSparseMatrix::from_triplets`'s own header comment already states for
its `E-LINALG-FOREIGN-ROW`/`E-LINALG-COL-RANGE` refusals.

## The block system

Order this rank's own `local_rows*nx` unknowns as `[x_I; x_G]`: `x_I` the
INTERIOR rows (`n_I = (local_rows-2)*nx` entries, `k` in
`[1, local_rows-2]`), `x_G` the two INTERFACE rows (`2*nx` entries, `k=0`
then `k=local_rows-1`) -- which is already the row band's own canonical
order, `k=0` first and `k=local_rows-1` last, with the interior rows
contiguous in between. Globally, `G` is the UNION of every rank's own 2
interface rows, `|G| = 2*nx*nranks`; `I` is the union of every rank's own
interior rows. `A_II` (every rank's own interior-interior block) is
BLOCK-DIAGONAL across ranks: an interior row's two neighbours are both
inside its own row band by definition, so no interior row of one rank ever
couples to anything owned by another rank. The full system, reordered
`[I; G]`:

```
[ A_II  A_IG ] [x_I]   [b_I]
[ A_GI  A_GG ] [x_G] = [b_G]
```

`A_GG` is NOT block-diagonal across ranks: a rank's own `k=0` row couples
to its predecessor's own `k=local_rows-1` row, and its own
`k=local_rows-1` row couples to its successor's own `k=0` row -- both
COUPLINGS BETWEEN TWO DIFFERENT RANKS' INTERFACE ROWS, the periodic wrap
included (rank 0's `k=0` predecessor is the LAST rank). These couplings are
diagonal in `i` (the stencil's vertical-neighbour term never mixes columns)
with value `-c*yy`, the same constant as every other vertical coupling in
`A`.

Standard block (Schur-complement) elimination: from the first block row,
`x_I = A_II^-1 (b_I - A_IG x_G)`; substituting into the second,

```
S x_G = b_G',   S = A_GG - A_GI A_II^-1 A_IG,   b_G' = b_G - A_GI A_II^-1 b_I
```

`S`, the SCHUR COMPLEMENT, is the dense `|G| x |G|` system every rank must
agree on to recover a consistent `x_G`.

## Assembly, one pass

`A_II`'s block-diagonal structure means `A_GI A_II^-1 A_IG` is ALSO
block-diagonal across ranks: `A_GI` (`= A_IG^T` by `A`'s own symmetry) only
has nonzero columns at this rank's own interior indices, so this term only
ever corrects THIS rank's own `2*nx x 2*nx` diagonal block of `S`. Writing
`S_own = A_GG,own - A_IG^T (A_II^-1 A_IG)` (the rank's own 2-row interface
block, reduced by its own interior elimination) and the two scalar
couplings to its periodic neighbours (`-c*yy`, diagonal in `i`), `S`'s full
structure, 3 blocks of nonzero columns per 2-row block of rows:

```
S[own rows, own cols]              = S_own                 (2*nx x 2*nx dense)
S[own rows (k=0), predecessor's LAST cols]  = -c*yy * I_nx  (one diagonal block)
S[own rows (k=local_rows-1), successor's FIRST cols] = -c*yy * I_nx
```

everywhere else zero. `A_IG`, `A_GG,own` and the interior `A_II` are ALL
built in ONE pass over this rank's own `(k, i)` pairs and their up-to-4
stencil neighbours, bucketing each generated `(row, col, value)` triplet by
whether `row`'s and `col`'s own `k` are interior or interface -- diagonal
and horizontal (`i+-1`) terms never leave `k`'s own classification;
vertical (`k+-1`) terms do, exactly at the row band's own two boundaries.
The dropped `(interface row, interior col)` direction is the TRANSPOSE of
an `(interior row, interface col)` entry the loop generates from the
NEIGHBOURING `k`'s own iteration (both `(k, k+1)` and `(k+1, k)` are added
explicitly) -- redundant by `A`'s own symmetry, so `A_GI` is never
separately materialised; every use reads it as `A_IG.transpose()`.

**Assembled by allgather.** Each rank places its OWN `2*nx` rows of the
`|G| x |G|` dense matrix (`S_own` at its own column block, the two
couplings above at its neighbours' column blocks, zero elsewhere) into a
`2*nx x |G|` buffer and calls ONE `Reducer::allgather` (this track's own
addition to `comm.hpp`, below): the concatenation of every rank's own
`2*nx` rows, in RANK ORDER, IS the complete `|G| x |G|` matrix row-major,
with no further bookkeeping -- ranks are numbered `0..nranks-1` and each
owns exactly `2*nx` CONSECUTIVE rows of `G` by construction (`G`'s own
ordering is "rank 0's first row, rank 0's last row, rank 1's first row,
...").

**Solved identically on every rank.** Every rank now holds the SAME `S`
(up to the floating-point order its own `allgather` implementation
delivers bytes in, which does not change the VALUES) and factors it with
the SAME dense `Eigen::LDLT` -- not a distributed solve of its own, by
design: `|G|` is `O(nx * nranks)`, small enough that a dense, fully
replicated factorisation costs far less than the complexity of a
distributed one, and "identical on every rank" is what lets `solve()`'s
own `x_G` extraction be a pure local slice (`x_G_full.segment(this rank's
own offset, 2*nx)`) with no further communication.

## Local back-substitution

At `solve()` time, `b_G' = b_G - A_GI A_II^-1 b_I` reduces the SAME way:
each rank computes its own `A_II^-1 b_I` and `A_IG^T` correction locally,
ONE allgather assembles the full `b_G'` (`2*nx` entries per rank,
concatenated in the SAME rank order `S` uses), every rank solves the SAME
factored `S x_G = b_G'` for the full `x_G` and keeps its own `2*nx`-entry
slice, then back-substitutes LOCALLY:

```
x_I = A_II^-1 (b_I - A_IG x_G)
```

-- `A_IG` here, never `A_GI`: eq. (1)'s own off-diagonal block, the
opposite role from the `A_GI` (`= A_IG^T`) used to build `b_G'`/`S_own`
above. Swapping which one a line uses
(`build-evidence/mutations/linalg.toml`'s "transposition" row,
`impl_->a_ig.transpose()` in place of `impl_->a_ig` in the
back-substitution line) is a genuine bug -- but a NOTEWORTHY, VERIFIED
property of this specific stencil keeps it from being a clean "wrong
number" demonstration at every grid: `n_I = 2*nx` holds ONLY at
`local_rows == 4` (solving `(local_rows-2)*nx = 2*nx` for `nx > 0`), and at
`local_rows == 4` there are exactly 2 interior rows, each adjacent to
EXACTLY ONE interface row -- `A_IG` is therefore `vert_val * I_{2nx}`, a
SCALAR MULTIPLE OF THE IDENTITY, hence its own transpose, at the one shape
where the swap does not immediately abort on an Eigen dimension assertion.
Measured directly (`nx=4, ny=8`, 2 ranks): the mutated build reproduces the
UNMUTATED answer exactly at that grid (`A_IG.transpose() == A_IG` there,
bit for bit), then the very NEXT case in declaration order
(`TwoPartitions16x16AgreesWithWholeGridReference`, where `n_I=96 != 2*nx=32`)
hits `eigen_assert(lhs.cols() == rhs.rows())` and aborts the whole test
process -- `test/linalg/schur_slab_test.cpp` deliberately declares the
`nx=4,ny=8` case FIRST among the multi-partition tests so this sequence
(one unaffected pass, then a crash) is reproducible rather than order-
dependent. `build-evidence/mutations/linalg.toml`'s own row records both
halves of this measurement rather than asserting a wrong number that this
stencil's own structure does not, in fact, produce at any shape that
survives long enough to report one.

## Operation count

Per rank, per `factor()` call (`n_I = (local_rows-2)*nx` this rank's own
interior count, `nx` the stencil's own bandwidth under the natural
`k*nx+i` numbering):

* Interior `SimplicialLDLT`: `O(n_I * nx^2)` (banded sparse Cholesky,
  bandwidth `nx`).
* `A_II^-1 A_IG` (`2*nx` right-hand sides against the already-factored
  interior): `O(n_I * nx^2)`.
* Dense `LDLT` of the assembled `|G| x |G| = (2*nx*nranks)^2` system:
  `O((nx*nranks)^3)`, REPLICATED IDENTICALLY on every rank -- the one place
  this design trades scalability for an exact, bit-reproducible answer
  with no further communication at `solve()` time; acceptable for the
  small `nranks` (a handful of participants) this solver targets, and
  explicitly NOT something a larger decomposition should assume stays
  cheap (see "what this does not scale to", below).
* The one `allgather`: `O(nx^2 * nranks)` doubles sent by each rank,
  `O(nx^2 * nranks^2)` total traffic -- grows with `nranks^2` because every
  rank needs the FULL dense `S`, not just its own portion.

Per `solve()` call, with `S` already factored: `O(n_I * nx)` local interior
solve + correction, `O(nx)` sent per `allgather`, `O((nx*nranks)^2)`
dense triangular solve (REPLICATED), `O(n_I * nx)` local back-substitution
-- the dense solve, not the sparse interior one, dominates for any
`nranks` worth decomposing over at all.

**What this does not scale to.** The replicated dense factorisation and
the `O(nranks^2)` allgather traffic make this solver appropriate for the
SMALL participant counts this project's own verification targets use
(a handful of ranks, e.g. `verification.brusselator.participants.mpmd.5`),
never for a row decomposition over hundreds of ranks -- at that scale a
distributed (not replicated) Schur solve, or an iterative method
preconditioned by this one, would be the right design. Nothing here
prevents building the iterative path later: `DistVector`/`DistSparseMatrix`
(`docs/dev/linalg-distributed.md`) are the same containers a Krylov
solver would use.

## `Reducer::allgather`: the one change to the containers

`SchurSlabSolver` needs ONE collective `Reducer`/`GhostExchanger`
(`comm.hpp`, `docs/dev/linalg-distributed.md`) did not already provide:
assembling a dense matrix or vector every rank must hold in FULL, from each
rank's own CONTIGUOUS contribution, with no reduction of the values
themselves -- `sum`/`max` reduce TO one value; this concatenates INTO a
larger one. Added as `Reducer::allgather(send, n_each, recv)`:
`recv[r*n_each .. (r+1)*n_each)` becomes participant `r`'s own `send`, for
every `r`, in rank order. `make_serial_reducer()`'s own implementation is
the one-participant identity (a copy); `make_mpi_reducer`'s wraps a single
`MPI_Allgather` per call. This is the ONE change this track made to the
containers the "Containers" track committed; both implementations and a
`LinalgComm` test of each (`test/linalg/schur_slab_test.cpp`'s
`SerialReducerAllgatherIsTheOneParticipantIdentity`,
`test/parallel/schur_slab_mpi_test.cpp`'s
`MpiReducerAllgatherConcatenatesInRankOrder`) live alongside this solver's
own files rather than in `dist_vector_test.cpp`/`dist_sparse_matrix_test.cpp`,
which this track does not own.

## Measured at tier 0 with an N-way in-process fake, not only the host

MPI cannot start in this sandbox (`docs/dev/testing.md`). The existing
`FakePeerExchanger`/`TwoPartitionReducer` pattern
(`test/linalg/dist_vector_test.cpp`, `dist_sparse_matrix_test.cpp`) binds
exactly 2 peers; requirement 2 needs 2 AND 3 partitions, so
`test/linalg/schur_slab_test.cpp` uses a GENUINE N-way in-process
`allgather` fake instead (`RingAllgatherReducer`): every rank's own call
records its `send` into a shared, rank-indexed cache slot, then reads every
slot back into `recv`. Sequential execution (no threads) means a single
pass leaves later ranks' slots unpopulated for an earlier rank's own call,
so every `factor()`/`solve()` batch runs TWICE through `two_pass_collective`
-- once to populate every slot (its own, necessarily incomplete, return
values discarded), once more now that every slot holds the value this
rank's own deterministic computation would produce either way, so the
second pass's results are the true, fully-concatenated answer. Valid
because `factor()`/`solve()` are pure functions of this rank's own already-
fixed inputs (`nx`/`ny`/`j_lo`/`j_hi`/`coefficient`, and for `solve()`, this
rank's own already-fixed rhs) -- nothing a warm-up call could leave mutated
that the real call would see differently.
`test/parallel/schur_slab_mpi_test.cpp` repeats the 2- and 3-rank
properties over a REAL `MPI_COMM_WORLD`, compiled here, run on the host.

## Measured bounds (round-off, not a discretisation difference)

Every multi-partition case is checked against
`examples/brusselator/brusselator_physics.hpp`'s own `periodic_laplacian`,
assembled and solved WHOLE exactly the way `DirectDiffusionSolver::factor`
does. At `nranks==1` the two paths are BIT-IDENTICAL (`EXPECT_EQ` on every
entry) by construction (the same section above). For `nranks>=2`, measured
on this machine (Debug, AppleClang 21, arm64):

| case | max absolute | max relative | largest entry |
|---|---|---|---|
| 16x16, 2 ranks | 8.4e-16 | 6.5e-14 | 7.6e-2 |
| 5x17, 3 ranks (17 not divisible by 3) | 4.7e-15 | 2.4e-14 | 2.9e-1 |
| 4x8, 2 ranks | 2.5e-16 | 1.3e-13 | 9.3e-2 |
| 6x10, 2 ranks, periodic-wrap forcing | 1.5e-16 | 1.4e-15 | 1.9e-1 |

every one consistent with double-precision round-off propagated through a
handful of extra floating-point operations (a sparse interior solve plus a
dense `64 x 64`-to-`170 x 170`-scale `LDLT`), not a discretisation
difference -- the assertions use `1e-12` absolute, ~1000x the largest
measured value and far below the `1e-9`+ scale a genuine assembly bug (a
dropped term, a wrong sign, a mis-ordered block) would produce, the same
margin `test/linalg/dist_sparse_matrix_test.cpp`'s own periodic-Laplacian
case uses for the identical reason.
