# Persisting a transfer map

Building a conservative remap between two unstructured meshes costs far more
than applying it, and the result depends only on the two meshes. Writing it
out turns a repeated run, a restart, or a change of rank count into a read.

```cpp
#include "fuse/backend/transfer_map_io.hpp"

// After building weights for the meshes this rank holds:
const auto map = fuse::to_persisted(weights, src_view, tgt_view, /*src_dim=*/2,
                                    /*tgt_dim=*/2);
if (!map.ok()) { /* E-MAP-SHAPE */ }
fuse::write_transfer_map("ocean-to-atmos." + std::to_string(rank) + ".fmap",
                         map.value());
```

Reading it back, at any rank count:

```cpp
std::vector<fuse::PersistedTransferMap> shards;
for (int r = 0; r < ranks_that_wrote; ++r) {
  auto s = fuse::read_transfer_map(fmt::format("ocean-to-atmos.{}.fmap", r));
  if (!s.ok()) { /* E-MAP-IO, -MAGIC, -VERSION, -TRUNCATED, -CHECKSUM */ }
  shards.push_back(std::move(s.value()));
}
const auto whole = fuse::merge_transfer_maps(shards);
const auto weights = fuse::localise(whole.value(), src_view, tgt_view, 2, 2);
```

## Why the rank count does not appear anywhere

Each entry is stored as `(target global id, source global id, weight)`. A
`MeshView` reports only what its rank holds — it has no global view by design
— so `MeshView::global_id` is the one thing two different partitions of the
same mesh agree on. A shard written by any rank is therefore readable by any
rank: `localise()` keeps the rows whose target entity the reader owns and
ignores the rest. Writing at two ranks and reading at four is not a special
case; it is the only thing the format does.

Storing the matrix as it sits in memory would not work: `RemapWeights::row`
and `::col` are *local* indices, so such a file would only be readable by a
run with the identical partition — the one case that does not need a file,
since an identical partition means an identical mesh and the map could just
be rebuilt.

## What a failed read tells you

| Code | Meaning |
|---|---|
| `E-MAP-IO` | the filesystem said no; the path and reason are in the message |
| `E-MAP-MAGIC` | not one of these files |
| `E-MAP-VERSION` | a format this build does not read |
| `E-MAP-TRUNCATED` | the header's length disagrees with the bytes present — what a killed job leaves behind |
| `E-MAP-CHECKSUM` | right length, changed contents |
| `E-MAP-DUPLICATE` | two shards claim the same entry; summing would double the weight and dropping one would lose a split contribution, so neither is chosen for you |
| `E-MAP-IDENTITY` | the map is not the kind you asked for (order, monotonicity, dimensions) |
| `E-MAP-COVERAGE` | **the distribution check** — a target entity you own has no row, or a row needs a source entity you do not hold |

`E-MAP-COVERAGE` is the one worth understanding. It is what a mismatched
partition looks like from the reader's side, and it is rejected rather than
tolerated because the alternative is a matrix with missing rows: `apply()`
would leave those targets at zero, which is indistinguishable from a field
that transferred and happened to be zero.

## What the file does not promise

That the weights are still numerically right for the meshes you have loaded.
Nothing in a file can establish that — the meshes may have moved under the
same global ids. The builder's own conservation audit travels with the map
(`max_row_defect`, `max_col_defect`, the two measure totals) and is reported
on read, but it describes the meshes the map was *built* from. If that matters
to your run, re-check it against the meshes you loaded.

## The answer does not depend on how many ranks you run

Values are bit-for-bit identical at 1, 2, 3, 4 and 5 tasks, including the
uneven splits, and `TransferMapIo.EveryRankCountGivesBitIdenticalValues`
asserts exact equality rather than a tolerance.

Two things make it hold. A target entity is owned by exactly one rank, so
every entry of a row is local and no row is ever summed in pieces. And
`localise()` orders each row by ascending **source global id**, never by local
column index — local indices depend on how many entities a rank holds and in
what order its halo arrived, so summing in that order would make the last bits
of the answer a function of the decomposition.

The test that pins this uses weights of very different magnitude (one of 1 and
three of 6e-17) so that summation order changes the bits, and gives each rank its
own source numbering, as a halo gather does. A test with similar-magnitude weights
or a replicated source cannot distinguish accumulation orders.

## See also

- [](geometry.md) for how the weights are built
- [](point-cloud-remap.md) for the point-cloud builder
