#include "fuse/backend/transfer_map_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <unordered_map>

#include <fmt/format.h>

#include "fuse/base/bytes.hpp"

namespace fuse {
namespace {

constexpr char          kMagic[8] = {'F', 'U', 'S', 'E', 'M', 'A', 'P', '\0'};
constexpr std::uint32_t kVersion = 1;

/// A cursor over bytes. `bytes.hpp` has writers and `fnv1a` but no readers,
/// and the reader needs to fail rather than read past the end on every field,
/// which is the whole difficulty of parsing a file that may be truncated.
class Cursor {
 public:
  explicit Cursor(std::span<const std::byte> b) : b_(b) {}

  [[nodiscard]] bool take(std::size_t n) noexcept {
    if (n > b_.size() - at_) return false;
    at_ += n;
    return true;
  }
  [[nodiscard]] bool integer(std::size_t width, std::uint64_t& out) noexcept {
    if (width > b_.size() - at_) return false;
    out = 0;
    for (std::size_t i = 0; i < width; ++i) {
      out |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(b_[at_ + i]))
             << (8 * i);
    }
    at_ += width;
    return true;
  }
  [[nodiscard]] bool real(double& out) noexcept {
    std::uint64_t bits = 0;
    if (!integer(8, bits)) return false;
    // Exact round trip: put_double writes the bit pattern, not a decimal.
    std::memcpy(&out, &bits, sizeof out);
    return true;
  }
  [[nodiscard]] std::size_t at() const noexcept { return at_; }
  [[nodiscard]] std::size_t left() const noexcept { return b_.size() - at_; }

 private:
  std::span<const std::byte> b_;
  std::size_t                at_ = 0;
};

void put_real(std::vector<std::byte>& out, double v) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof bits);
  bytes::put_integer(out, bits, 8);
}

}  // namespace

Result<PersistedTransferMap> to_persisted(const RemapWeights& w,
                                          const MeshView& src,
                                          const MeshView& tgt, int src_dim,
                                          int tgt_dim) {
  const std::size_t n_src = src.entity_count(src_dim);
  const std::size_t n_tgt = tgt.entity_count(tgt_dim);

  PersistedTransferMap out;
  out.identity = TransferMapIdentity{.order = w.order,
                                     .monotone = w.monotone,
                                     .src_dim = src_dim,
                                     .tgt_dim = tgt_dim};
  out.src_measure_total = w.src_measure_total;
  out.tgt_measure_total = w.tgt_measure_total;
  out.max_row_defect = w.max_row_defect;
  out.max_col_defect = w.max_col_defect;

  const std::size_t n = w.val.size();
  if (w.row.size() != n || w.col.size() != n) {
    return Failure{"E-MAP-SHAPE",
                   fmt::format("the matrix has {} values but {} rows and {} "
                               "columns; a triplet needs all three",
                               n, w.row.size(), w.col.size())};
  }

  out.tgt_gid.reserve(n);
  out.src_gid.reserve(n);
  out.val.reserve(n);

  // Ordered by (target gid, source gid), which is NOT the same permutation as
  // the local (row, col) order: local indices and global ids need not agree.
  // Sorting here rather than at write time keeps the promise in one place.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::vector<std::uint64_t> tg(n);
  std::vector<std::uint64_t> sg(n);
  for (std::size_t k = 0; k < n; ++k) {
    const auto r = static_cast<std::size_t>(w.row[k]);
    const auto c = static_cast<std::size_t>(w.col[k]);
    if (w.row[k] < 0 || r >= n_tgt || w.col[k] < 0 || c >= n_src) {
      return Failure{
          "E-MAP-SHAPE",
          fmt::format("entry {} names target row {} of {} and source column "
                      "{} of {}; an index outside the view it indexes cannot "
                      "be given a global id",
                      k, w.row[k], n_tgt, w.col[k], n_src)};
    }
    tg[k] = static_cast<std::uint64_t>(tgt.global_id(tgt_dim, r));
    sg[k] = static_cast<std::uint64_t>(src.global_id(src_dim, c));
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) {
                     if (tg[a] != tg[b]) return tg[a] < tg[b];
                     return sg[a] < sg[b];
                   });
  for (const std::size_t k : order) {
    out.tgt_gid.push_back(tg[k]);
    out.src_gid.push_back(sg[k]);
    out.val.push_back(w.val[k]);
  }
  return out;
}

std::vector<std::byte> encode_transfer_map(const PersistedTransferMap& map) {
  std::vector<std::byte> payload;
  payload.reserve(map.size() * 24 + 64);
  bytes::put_signed(payload, map.identity.order, 4);
  bytes::put_integer(payload, map.identity.monotone ? 1u : 0u, 1);
  bytes::put_signed(payload, map.identity.src_dim, 4);
  bytes::put_signed(payload, map.identity.tgt_dim, 4);
  put_real(payload, map.src_measure_total);
  put_real(payload, map.tgt_measure_total);
  put_real(payload, map.max_row_defect);
  put_real(payload, map.max_col_defect);
  bytes::put_integer(payload, map.size(), 8);
  for (std::size_t k = 0; k < map.size(); ++k) {
    bytes::put_integer(payload, map.tgt_gid[k], 8);
    bytes::put_integer(payload, map.src_gid[k], 8);
    put_real(payload, map.val[k]);
  }

  std::vector<std::byte> out;
  out.reserve(payload.size() + 24);
  for (const char c : kMagic) out.push_back(static_cast<std::byte>(c));
  bytes::put_integer(out, kVersion, 4);
  bytes::put_integer(out, payload.size(), 8);
  // Over the payload only: the header's own length field is what the payload
  // is checked against, so hashing it too would conflate the two failures.
  bytes::put_integer(out, bytes::fnv1a(payload), 8);
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

Result<PersistedTransferMap> decode_transfer_map(
    std::span<const std::byte> raw) {
  Cursor c{raw};
  if (raw.size() < sizeof(kMagic) + 20) {
    return Failure{"E-MAP-TRUNCATED",
                   fmt::format("{} bytes is shorter than the header", raw.size())};
  }
  for (const char want : kMagic) {
    std::uint64_t got = 0;
    (void)c.integer(1, got);
    if (got != static_cast<std::uint64_t>(static_cast<unsigned char>(want))) {
      return Failure{"E-MAP-MAGIC",
                     "not a fuse transfer map: the file does not begin with "
                     "the format's magic bytes"};
    }
  }
  std::uint64_t version = 0;
  std::uint64_t payload_len = 0;
  std::uint64_t want_hash = 0;
  (void)c.integer(4, version);
  (void)c.integer(8, payload_len);
  (void)c.integer(8, want_hash);
  if (version != kVersion) {
    return Failure{"E-MAP-VERSION",
                   fmt::format("the file is format version {}; this build "
                               "reads version {} only",
                               version, kVersion)};
  }
  if (c.left() != payload_len) {
    return Failure{
        "E-MAP-TRUNCATED",
        fmt::format("the header declares a {}-byte payload and {} bytes "
                    "follow it; a write that was interrupted looks exactly "
                    "like this",
                    payload_len, c.left())};
  }
  const std::size_t payload_at = c.at();
  if (bytes::fnv1a(raw.subspan(payload_at, payload_len)) != want_hash) {
    return Failure{"E-MAP-CHECKSUM",
                   "the payload hash disagrees with the header; the file is "
                   "the declared length but its contents have changed"};
  }

  PersistedTransferMap map;
  std::uint64_t        order = 0;
  std::uint64_t        monotone = 0;
  std::uint64_t        src_dim = 0;
  std::uint64_t        tgt_dim = 0;
  std::uint64_t        n = 0;
  const bool head_ok = c.integer(4, order) && c.integer(1, monotone) &&
                       c.integer(4, src_dim) && c.integer(4, tgt_dim) &&
                       c.real(map.src_measure_total) &&
                       c.real(map.tgt_measure_total) &&
                       c.real(map.max_row_defect) &&
                       c.real(map.max_col_defect) && c.integer(8, n);
  if (!head_ok) {
    return Failure{"E-MAP-TRUNCATED",
                   "the payload ends inside the map's own description"};
  }
  map.identity = TransferMapIdentity{
      .order = static_cast<int>(static_cast<std::int32_t>(order)),
      .monotone = monotone != 0,
      .src_dim = static_cast<int>(static_cast<std::int32_t>(src_dim)),
      .tgt_dim = static_cast<int>(static_cast<std::int32_t>(tgt_dim))};

  if (n > c.left() / 24) {
    return Failure{
        "E-MAP-TRUNCATED",
        fmt::format("the map declares {} entries, which needs {} bytes, and "
                    "{} remain",
                    n, n * 24, c.left())};
  }
  map.tgt_gid.resize(n);
  map.src_gid.resize(n);
  map.val.resize(n);
  for (std::uint64_t k = 0; k < n; ++k) {
    if (!c.integer(8, map.tgt_gid[k]) || !c.integer(8, map.src_gid[k]) ||
        !c.real(map.val[k])) {
      return Failure{"E-MAP-TRUNCATED",
                     fmt::format("the payload ends inside entry {}", k)};
    }
  }
  return map;
}

Result<void> write_transfer_map(const std::string& path,
                                const PersistedTransferMap& map) {
  const std::vector<std::byte> bytes = encode_transfer_map(map);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    return Failure{"E-MAP-IO",
                   fmt::format("cannot open '{}' for writing", path)};
  }
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  f.close();
  if (!f) {
    return Failure{"E-MAP-IO",
                   fmt::format("writing '{}' failed after {} bytes; the file "
                               "on disk is incomplete and will read as "
                               "E-MAP-TRUNCATED",
                               path, bytes.size())};
  }
  return Result<void>::success();
}

Result<PersistedTransferMap> read_transfer_map(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return Failure{"E-MAP-IO", fmt::format("cannot open '{}' for reading", path)};
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  if (f.bad()) {
    return Failure{"E-MAP-IO", fmt::format("reading '{}' failed", path)};
  }
  return decode_transfer_map(
      std::span<const std::byte>{reinterpret_cast<const std::byte*>(raw.data()),
                                 raw.size()});
}

Result<PersistedTransferMap> merge_transfer_maps(
    std::span<const PersistedTransferMap> shards) {
  if (shards.empty()) {
    return Failure{"E-MAP-SHAPE", "no shards to merge"};
  }
  PersistedTransferMap out = shards.front();
  out.tgt_gid.clear();
  out.src_gid.clear();
  out.val.clear();

  std::size_t total = 0;
  for (const PersistedTransferMap& s : shards) {
    if (s.identity != out.identity) {
      return Failure{
          "E-MAP-IDENTITY",
          fmt::format("a shard describes an order-{} {} map on dims {}->{} "
                      "while the first describes order-{} {} on {}->{}; these "
                      "are not shards of one map",
                      s.identity.order, s.identity.monotone ? "monotone" : "plain",
                      s.identity.src_dim, s.identity.tgt_dim,
                      out.identity.order,
                      out.identity.monotone ? "monotone" : "plain",
                      out.identity.src_dim, out.identity.tgt_dim)};
    }
    total += s.size();
  }

  // Indices into the concatenation, sorted once: a k-way merge of already
  // sorted shards, without materialising the concatenation first.
  std::vector<std::pair<std::size_t, std::size_t>> where;
  where.reserve(total);
  for (std::size_t i = 0; i < shards.size(); ++i) {
    for (std::size_t k = 0; k < shards[i].size(); ++k) where.emplace_back(i, k);
  }
  std::stable_sort(where.begin(), where.end(), [&](auto a, auto b) {
    const auto& A = shards[a.first];
    const auto& B = shards[b.first];
    if (A.tgt_gid[a.second] != B.tgt_gid[b.second]) {
      return A.tgt_gid[a.second] < B.tgt_gid[b.second];
    }
    return A.src_gid[a.second] < B.src_gid[b.second];
  });

  out.tgt_gid.reserve(total);
  out.src_gid.reserve(total);
  out.val.reserve(total);
  for (const auto [i, k] : where) {
    const auto& s = shards[i];
    if (!out.tgt_gid.empty() && out.tgt_gid.back() == s.tgt_gid[k] &&
        out.src_gid.back() == s.src_gid[k]) {
      return Failure{
          "E-MAP-DUPLICATE",
          fmt::format("two shards both hold the entry (target {}, source {}); "
                      "summing them would double the weight and dropping one "
                      "would lose a legitimately split contribution, so "
                      "neither is chosen for you",
                      s.tgt_gid[k], s.src_gid[k])};
    }
    out.tgt_gid.push_back(s.tgt_gid[k]);
    out.src_gid.push_back(s.src_gid[k]);
    out.val.push_back(s.val[k]);
  }
  return out;
}

Result<RemapWeights> localise(const PersistedTransferMap& map,
                              const MeshView& src, const MeshView& tgt,
                              int src_dim, int tgt_dim,
                              const TransferMapIdentity* required) {
  if (required != nullptr && !(map.identity == *required)) {
    return Failure{
        "E-MAP-IDENTITY",
        fmt::format("the file holds an order-{} {} map on dims {}->{}; the "
                    "caller requires order-{} {} on {}->{}",
                    map.identity.order,
                    map.identity.monotone ? "monotone" : "plain",
                    map.identity.src_dim, map.identity.tgt_dim,
                    required->order, required->monotone ? "monotone" : "plain",
                    required->src_dim, required->tgt_dim)};
  }

  const std::size_t n_src = src.entity_count(src_dim);
  const std::size_t n_tgt = tgt.entity_count(tgt_dim);

  std::unordered_map<std::uint64_t, std::int32_t> src_local;
  src_local.reserve(n_src * 2);
  for (std::size_t i = 0; i < n_src; ++i) {
    src_local.emplace(static_cast<std::uint64_t>(src.global_id(src_dim, i)),
                      static_cast<std::int32_t>(i));
  }
  std::unordered_map<std::uint64_t, std::int32_t> tgt_local;
  tgt_local.reserve(n_tgt * 2);
  for (std::size_t j = 0; j < n_tgt; ++j) {
    tgt_local.emplace(static_cast<std::uint64_t>(tgt.global_id(tgt_dim, j)),
                      static_cast<std::int32_t>(j));
  }

  RemapWeights w;
  w.n_rows = n_tgt;
  w.n_cols = n_src;
  w.order = map.identity.order;
  w.monotone = map.identity.monotone;
  w.src_measure_total = map.src_measure_total;
  w.tgt_measure_total = map.tgt_measure_total;
  w.max_row_defect = map.max_row_defect;
  w.max_col_defect = map.max_col_defect;

  // Collected before it is emitted, because the output order is load-bearing
  // twice over and the two requirements are not the same:
  //
  //   1. `RemapWeights` documents `row` as ascending in LOCAL index. The file
  //      is sorted by global id, and local numbering need not agree with it,
  //      so emitting file order would break that invariant.
  //   2. Within one row the sum must run in ascending SOURCE GLOBAL ID, which
  //      is what makes the result bit-for-bit independent of rank count: a
  //      target entity is owned by exactly one rank, so all of its entries are
  //      local, and the global ids they carry are the same numbers whatever
  //      the partition. Local column indices are NOT: they depend on how many
  //      entities this rank holds and in which order.
  //
  // Sorting by (local row, source gid) satisfies both, and the row part is
  // free of consequence: reordering whole rows cannot change any row's sum.
  struct Entry {
    std::int32_t  row;
    std::uint64_t src_gid;
    std::int32_t  col;
    double        val;
  };
  std::vector<Entry> kept;
  kept.reserve(map.size());

  std::vector<char> row_seen(n_tgt, 0);
  for (std::size_t k = 0; k < map.size(); ++k) {
    const auto row_it = tgt_local.find(map.tgt_gid[k]);
    if (row_it == tgt_local.end()) continue;  // another rank's row
    const auto col_it = src_local.find(map.src_gid[k]);
    if (col_it == src_local.end()) {
      return Failure{
          "E-MAP-COVERAGE",
          fmt::format("target entity {} needs source entity {}, which this "
                      "rank does not hold. The map was written for a "
                      "different partition of the source mesh, and the value "
                      "it needs is not here to apply -- transfer the source "
                      "field to this rank first, or reload with the "
                      "partition the map was written for",
                      map.tgt_gid[k], map.src_gid[k])};
    }
    kept.push_back(Entry{.row = row_it->second,
                         .src_gid = map.src_gid[k],
                         .col = col_it->second,
                         .val = map.val[k]});
    row_seen[static_cast<std::size_t>(row_it->second)] = 1;
  }

  std::stable_sort(kept.begin(), kept.end(), [](const Entry& a, const Entry& b) {
    if (a.row != b.row) return a.row < b.row;
    return a.src_gid < b.src_gid;
  });
  w.row.reserve(kept.size());
  w.col.reserve(kept.size());
  w.val.reserve(kept.size());
  for (const Entry& e : kept) {
    w.row.push_back(e.row);
    w.col.push_back(e.col);
    w.val.push_back(e.val);
  }

  const auto missing = static_cast<std::size_t>(
      std::count(row_seen.begin(), row_seen.end(), 0));
  if (missing != 0) {
    const auto first =
        static_cast<std::size_t>(
            std::find(row_seen.begin(), row_seen.end(), 0) - row_seen.begin());
    return Failure{
        "E-MAP-COVERAGE",
        fmt::format("{} of this rank's {} target entities have no row in the "
                    "map, the first being local {} (global {}). Applying it "
                    "would write zero into a field the map exists to fill, so "
                    "a shard is missing or the map belongs to another mesh",
                    missing, n_tgt, first,
                    tgt.global_id(tgt_dim, first))};
  }
  return w;
}

}  // namespace fuse
