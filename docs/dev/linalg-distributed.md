# Distributed linear algebra containers

`fuse::linalg::DistVector` and `fuse::linalg::DistSparseMatrix`
(`include/fuse/linalg/{dist_vector,dist_sparse_matrix,comm}.hpp`) are the
row-distributed vector and sparse matrix a native Krylov solver (PCG,
PGMRES(m)) and an exact Schur-complement solve are built on. Neither type
names MPI: communication is injected through two small interfaces,
`Reducer` and `GhostExchanger`, with a serial implementation of both that
needs no MPI symbol at all and an MPI implementation
(`src/linalg/comm_mpi.cpp`) behind the same two abstract classes.

## Data layout: owned, then ghost

Every participant's `DistVector` is one contiguous buffer, the OWNED block
first and the GHOST region immediately after it:

```
data()[0 .. owned_size())                 this participant's own entries
data()[owned_size() .. owned_size()+ghost_size())   copies of entries OTHER
                                                     participants own
```

The owned block is in the participant's own local order -- for a
row-distributed problem, local row `i` corresponds to global row
`row_lo + i`. The ghost region has no global-index order of its own; it is
ordered however the `GhostPattern` that built it says, which in practice is
"ascending by global column" (see below), but nothing in `DistVector`
depends on that order being anything in particular. A caller that needs a
specific ghost's value reads `data()[owned_size() + g]` for whichever `g`
its own bookkeeping (typically a `DistSparseMatrix`'s off-diagonal CSR
column indices) names.

This single-buffer layout is what lets `DistSparseMatrix::apply` read `x`
directly with no repacking: the diagonal CSR product indexes
`x.data()[0 .. owned_size())` and the off-diagonal product indexes
`x.data()[owned_size() + g)` for ghost column `g` -- two plain loops over
one buffer, nothing gathered or scattered first.

A ghost is **refreshed on request, not kept continuously in sync**:
`update_ghosts()` is the one operation that changes the ghost region, and
nothing else touches it. A caller that mutates its own owned block and
then reads a peer's ghost value without calling `update_ghosts()` again
sees whatever the LAST update left there, stale or not -- the same
contract `fuse::backend`'s mesh halos and `fuse::transport`'s copy
transport both already use, named here so a caller coming from either of
those is not surprised by a third variant.

## Communication, injected: `Reducer` and `GhostExchanger`

Both are abstract classes in `comm.hpp`, with no third-party type (not
even `mpi.h`) anywhere a public header can see it (`linalg`'s position in
docs/dev/layering.md: it depends on `base` alone). A `DistVector` holds a
`std::shared_ptr<const Reducer>` and a `std::shared_ptr<const
GhostExchanger>` for its whole lifetime, handed in at construction.

* **`Reducer`** is a global combine over whatever set of participants this
  instance was built for: `sum`/`max`, scalar and vectorised, plus
  `rank()`/`size()`. `make_serial_reducer()` is the one-participant
  identity (sum/max of one value is that value); `make_mpi_reducer(comm_f)`
  wraps an `MPI_Allreduce` per call.

* **`GhostExchanger`** answers one question -- "what are my ghosts worth
  right now" -- along a `GhostPattern` (one `(owner_rank,
  remote_local_index)` pair per ghost) that was fixed when the exchanger
  was built. `make_serial_exchanger()` is the zero-ghost case a one-rank
  `DistVector` uses. `make_mpi_exchanger(comm_f, pattern, owned_n)` derives
  the real exchange plan ONCE:

  1. Group this rank's own ghosts by owner (`req_counts`/`req_displs`,
     ascending owner rank) -- this is also the UNPACK map (`fill_index`):
     which ghost slot each grouped request answers.
  2. One `MPI_Alltoall` of those counts against every other rank's own
     tells this rank how many requests it must answer, and from whom
     (`ans_counts`/`ans_displs`).
  3. One `MPI_Alltoallv` of the request VALUES (each one simply
     `remote_local_index` itself -- the pattern already names the owner's
     own local position, no id lookup needed) tells this rank exactly
     which of its OWN owned entries every other rank wants, in the order
     that rank expects them back -- the PACK map (`ans_index`).

  `update()` repeats only the data motion fixed by that plan: pack
  `owned[ans_index[k]]` into a send buffer ordered by
  `ans_counts`/`ans_displs`, one `MPI_Alltoallv`, unpack
  `ghosts[fill_index[k]] = recv[k]`. This is the same two-phase shape
  `src/backend/moab_backend.cpp`'s halo exchange already uses for a
  different entity set; `src/linalg/comm_mpi.cpp`'s own header comment has
  the line-by-line derivation.

A `GhostExchanger::update()` call is COLLECTIVE under MPI: every
participant named anywhere in any participant's pattern must call it when
any one of them does, the same "global collective discipline"
`fuse::partition::detail`'s collectives document for exactly this reason
(docs/dev/partitioning.md). `DistSparseMatrix::apply` calls it exactly
once, at the start, before either local CSR product.

## The column split: diagonal block vs. off-process block

`DistSparseMatrix::from_triplets` takes a COO list in GLOBAL row/column
numbering -- this participant's OWN rows only -- and a `RowOwnership`
(every participant's `row_lo`, ascending). From those two it derives,
locally and with no communication:

* **Which triplets are a caller mistake.** A triplet whose row falls
  outside `[row_lo, row_hi)` is refused (`E-LINALG-FOREIGN-ROW`): a
  participant's own triplet list must be exactly its own rows, never
  "some other rank's row that will be retried over there" -- there is no
  cross-rank retry in this design, a triplet is either this participant's
  own or it is an error. A column outside `[0, global_n)` is refused
  (`E-LINALG-COL-RANGE`) for the same reason: no `RowOwnership` entry could
  resolve it.

* **Which column is a column this participant already owns.** Since the
  operators this solves for are square and partitioned identically on
  rows and columns, a triplet's column is in the DIAGONAL block exactly
  when it falls in this participant's own `[row_lo, row_hi)` -- the same
  range its rows do, checked the same way, no `RowOwnership` lookup
  needed.

* **Which column is off-process, and who owns it.** Every other column's
  owner is found with one `std::upper_bound` over `RowOwnership::row_lo`
  (ascending, contiguous ranges, so one comparison resolves it). The
  UNIQUE set of such columns, sorted ascending by global column -- which
  also sorts them ascending by owner, because the ranges are contiguous
  and monotonic -- becomes the matrix's `GhostPattern`: ghost `g`'s
  `owner_rank` and `remote_local_index` (`global_col - row_lo[owner]`) are
  fixed at this point, once, and `ghost_pattern()` is what a `DistVector`
  passed to `apply()` as `x` must be built to carry.

* **Duplicates are summed.** Two triplets naming the same `(row, col)` are
  the usual COO-to-CSR convention: their values add, in `from_triplets`
  alone, before either CSR block is assembled.

The result is two ordinary CSR blocks per participant -- a diagonal block
over LOCAL row and column indices, and an off-process block over local
rows but GHOST-local column indices (`0 .. ghost_pattern().owner_rank.size()`
), so `apply()`'s off-diagonal loop reads `x.data()[owned_size() + g]`
directly with no further translation. `diagonal()` needs no communication
at all: a participant's own diagonal entries are, by the column split's
own definition, always inside its own diagonal block.

## Measured against a tier-0 fake, not only the host

MPI cannot start in this sandbox (docs/dev/testing.md), so the ghost
pattern and reduction logic are measured at tier 0 with an **in-process
fake two-partition split**: two `DistVector`s (or two `DistSparseMatrix`
row blocks) in the SAME process, wired together by a `GhostExchanger` test
double that reads its peer's `data()` directly instead of sending a
message, and a `Reducer` test double that adds in the peer's own local
contribution (computed independently, as ground truth, not by calling the
peer's own code) instead of issuing an `MPI_Allreduce`. This is enough to
catch a wrong unpack index, a dropped off-diagonal product, or a skipped
reduction -- the three mutation rows in
`build-evidence/mutations/linalg.toml` -- without any MPI symbol in the
test binary. `test/parallel/linalg_mpi_test.cpp` repeats the same
properties over a REAL `MPI_COMM_WORLD`, compiled in a tier-1 build and
run on the host, never here.
