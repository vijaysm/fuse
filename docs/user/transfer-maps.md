# Persisting a transfer map

This page is for anyone who builds the same remap weights more than once:
across runs, across restarts, or at a different rank count. It covers the
write and read API in `fuse/backend/transfer_map_io.hpp`, a complete
round trip, and what a failed read means.

Building a conservative remap between two unstructured meshes costs far more
than applying it, and the result depends only on the two meshes. Writing it
out turns a repeated run, a restart, or a change of rank count into a read.

## The API

Everything is in namespace `fuse`, declared in
`fuse/backend/transfer_map_io.hpp`. The file format is FUSE's own
self-describing binary (an 8-byte magic, a format version, the payload and a
64-bit hash of it); it needs no HDF5 or any other optional library.

```cpp
struct TransferMapIdentity {        // what a map says about itself
  int  order = 1;                   // RemapWeights::order
  bool monotone = false;            // RemapWeights::monotone
  int  src_dim = 0;                 // topological dimension the source ids live on
  int  tgt_dim = 0;                 // and the target's
};

struct PersistedTransferMap {       // a map as it exists on disk: global ids
  TransferMapIdentity        identity;
  std::vector<std::uint64_t> tgt_gid;   // ascending, with src_gid ascending within it
  std::vector<std::uint64_t> src_gid;
  std::vector<double>        val;
  double src_measure_total, tgt_measure_total, max_row_defect, max_col_defect;  // the builder's audit
  std::size_t size() const noexcept;    // entries
};

// Local indices -> global ids. `src`/`tgt` must be the views `w` was built from.
Result<PersistedTransferMap> to_persisted(const RemapWeights& w, const MeshView& src,
                                          const MeshView& tgt, int src_dim, int tgt_dim);

// Bytes, for a caller that stores the map somewhere other than a file.
std::vector<std::byte>       encode_transfer_map(const PersistedTransferMap& map);
Result<PersistedTransferMap> decode_transfer_map(std::span<const std::byte> bytes);

// One shard per file.
Result<void>                 write_transfer_map(const std::string& path, const PersistedTransferMap& map);
Result<PersistedTransferMap> read_transfer_map(const std::string& path);

// Shards -> one sorted map.
Result<PersistedTransferMap> merge_transfer_maps(std::span<const PersistedTransferMap> shards);

// Global ids -> this rank's local matrix. Keeps the rows whose target gid `tgt` owns.
Result<RemapWeights> localise(const PersistedTransferMap& map, const MeshView& src,
                              const MeshView& tgt, int src_dim, int tgt_dim,
                              const TransferMapIdentity* required = nullptr);
```

`src_dim` and `tgt_dim` are the entity dimensions the matrix indexes (the
same `dim` passed to `build_supermesh_weights`, or 0 for a nodal side).
Passing `required` to `localise` makes it refuse a map that is not the kind
you expect (`E-MAP-IDENTITY`); leaving it null accepts whatever the file says
and lets you read `identity` afterwards.

## Write, then read

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

## A complete round trip

Weights between two generated line meshes, written as one shard, read back,
localised against the same two views, and compared with the direct build.
Serial build, no optional library:

```cpp
#include <fuse/backend/serial_backend.hpp>
#include <fuse/backend/supermesh.hpp>
#include <fuse/backend/transfer_map_io.hpp>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  // 1. Two meshes of different resolution.
  fuse::SerialMeshBackend backend;
  if (!backend.load(fuse::MeshDecl{.name = "ocean", .source = "generated:line:11"}).ok() ||
      !backend.load(fuse::MeshDecl{.name = "atmos", .source = "generated:line:7"}).ok()) {
    return 1;
  }
  const fuse::MeshView& src = *backend.view(backend.mesh_id("ocean").value()).value();
  const fuse::MeshView& tgt = *backend.view(backend.mesh_id("atmos").value()).value();

  // 2. Build the weights (order 1, conservative, dimension 1).
  const auto w = fuse::build_supermesh_weights(src, tgt, /*dim=*/1);
  if (!w.ok()) { std::cerr << w.error().code << ": " << w.error().message << "\n"; return 1; }

  // 3. Translate local indices to global ids and write one shard.
  const auto persisted = fuse::to_persisted(w.value(), src, tgt, /*src_dim=*/1, /*tgt_dim=*/1);
  if (!persisted.ok()) { std::cerr << persisted.error().code << "\n"; return 1; }
  const std::string path = "ocean-to-atmos.0.fmap";
  if (const auto r = fuse::write_transfer_map(path, persisted.value()); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }

  // 4. Read it back, merge the (single) shard, and localise for the meshes in hand.
  std::vector<fuse::PersistedTransferMap> shards;
  const auto shard = fuse::read_transfer_map(path);
  if (!shard.ok()) { std::cerr << shard.error().code << ": " << shard.error().message << "\n"; return 1; }
  shards.push_back(shard.value());
  const auto whole = fuse::merge_transfer_maps(shards);
  if (!whole.ok()) { std::cerr << whole.error().code << "\n"; return 1; }

  fuse::TransferMapIdentity required;   // what we expect the file to be
  required.order = 1;
  required.monotone = w.value().monotone;
  required.src_dim = 1;
  required.tgt_dim = 1;
  const auto reloaded = fuse::localise(whole.value(), src, tgt, 1, 1, &required);
  if (!reloaded.ok()) { std::cerr << reloaded.error().code << ": " << reloaded.error().message << "\n"; return 1; }

  // 5. The reload applies identically to the direct build.
  std::vector<double> f(src.entity_count(1));
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = 2.3 + 0.37 * static_cast<double>(i);
  std::vector<double> g_direct(tgt.entity_count(1)), g_reload(tgt.entity_count(1));
  (void)fuse::apply_weights(w.value(), f.data(), g_direct.data(), 1);
  (void)fuse::apply_weights(reloaded.value(), f.data(), g_reload.data(), 1);
  double max_diff = 0.0, src_total = 0.0, tgt_total = 0.0;
  for (std::size_t j = 0; j < g_direct.size(); ++j) {
    max_diff = std::max(max_diff, std::abs(g_direct[j] - g_reload[j]));
    tgt_total += g_reload[j] * tgt.measure(1, j);
  }
  for (std::size_t i = 0; i < f.size(); ++i) src_total += f[i] * src.measure(1, i);

  std::cout << "entries written           : " << persisted.value().size() << "\n"
            << "entries reloaded (local)  : " << reloaded.value().nnz() << "\n"
            << "max |direct - reload|     : " << max_diff << "\n"
            << "max_row_defect (file)     : " << whole.value().max_row_defect << "\n"
            << "max_col_defect (file)     : " << whole.value().max_col_defect << "\n"
            << "source integral           : " << src_total << "\n"
            << "target integral (reload)  : " << tgt_total << "\n";

  // 6. A wrong identity is refused by name.
  fuse::TransferMapIdentity wrong = required;
  wrong.order = 2;
  const auto bad = fuse::localise(whole.value(), src, tgt, 1, 1, &wrong);
  std::cout << "order-2 required          : " << bad.error().code << "\n";
  return 0;
}
```

Expected output (exact digits can vary by platform):

```text
entries written           : 17
entries reloaded (local)  : 17
max |direct - reload|     : 0
max_row_defect (file)     : 4.44089e-16
max_col_defect (file)     : 9.15934e-16
source integral           : 4.15
target integral (reload)  : 4.15
order-2 required          : E-MAP-IDENTITY
```

Eleven cells onto seven (coprime, so no interior boundary coincides) give 17
overlaps. The reloaded matrix applies bit-identically to the direct build
here because every row is summed in ascending source global id on both
paths; the builder's audit travels in the file and the integral is the same
on both sides.

With several writers, each rank writes its own shard (one `to_persisted` per
rank over the views that rank holds) and every reader reads all of them,
merges, and localises against its own views, exactly as the snippet at the
top of this page shows. `localise` never needs to know how many ranks wrote
or how many read.

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

On the write side, `to_persisted` refuses `E-MAP-SHAPE` when an index in the
matrix falls outside the view it should index: the weights were built from
different views than the ones passed.

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

- [](geometry.md) for how the weights are built, the method strings and the remap error codes
- [](point-cloud-remap.md) for the point-cloud builder
- [](interface-transfer.md) for the nodal interpolation builders
- [](meshes.md) for the mesh backends and their global ids
- [](restart.md) for restarting the coupled run itself
