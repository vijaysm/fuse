// Acceptance criterion: a map persisted at 2 ranks that reloads at 4 and
// reproduces 1e-13, plus a distribution check that rejects a mismatched
// partition. No MPI is needed: the rank-count change is modelled the way the format
// itself sees it: a partition is nothing but a set of global ids, and two
// "ranks" are two MeshViews over disjoint id sets. The parallel run adds the
// launcher, not a new code path -- `localise()` cannot tell how many ranks
// exist and has no argument that would let it.

#include "fuse/backend/transfer_map_io.hpp"

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

/// A partition: `gids[i]` is the global id of local entity `i`. Everything
/// else is the minimum `MeshView` demands.
class Partition final : public fuse::MeshView {
 public:
  explicit Partition(std::vector<std::size_t> gids) : gids_(std::move(gids)) {}

  [[nodiscard]] int dimension() const override { return 2; }
  [[nodiscard]] std::size_t entity_count(int) const override {
    return gids_.size();
  }
  [[nodiscard]] std::size_t owned_count(int) const override {
    return gids_.size();
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] std::size_t global_id(int, std::size_t i) const override {
    return gids_[i];
  }
  [[nodiscard]] fuse::Point centroid(int, std::size_t i) const override {
    return fuse::Point{static_cast<double>(gids_[i]), 0.0, 0.0};
  }
  [[nodiscard]] double measure(int, std::size_t) const override { return 1.0; }

  // No geometry, which `MeshView` documents as legal for `simplex_count`: a
  // persisted map is global ids and weights, and nothing in this file's code
  // path asks where an entity is. A view that answered with invented corners
  // would let a geometric test pass here on coordinates that mean nothing.
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override {
    return 0;
  }
  void simplex_points(int, std::size_t, std::size_t,
                      fuse::Point*) const override {}

 private:
  std::vector<std::size_t> gids_;
};

std::vector<std::size_t> ids(std::size_t first, std::size_t count) {
  std::vector<std::size_t> v(count);
  std::iota(v.begin(), v.end(), first);
  return v;
}

/// Row `j` takes 0.25 from each of four source entities, so the row sums to
/// one and a constant transfers exactly -- the property a reload must not
/// disturb.
fuse::RemapWeights four_point_map(std::size_t n_rows, std::size_t n_cols) {
  fuse::RemapWeights w;
  w.n_rows = n_rows;
  w.n_cols = n_cols;
  w.order = 1;
  w.max_row_defect = 4.4e-16;
  w.max_col_defect = 8.8e-16;
  w.src_measure_total = static_cast<double>(n_cols);
  w.tgt_measure_total = static_cast<double>(n_rows);
  for (std::size_t j = 0; j < n_rows; ++j) {
    for (std::size_t d = 0; d < 4; ++d) {
      w.row.push_back(static_cast<std::int32_t>(j));
      w.col.push_back(static_cast<std::int32_t>((j * 3 + d) % n_cols));
      // Four DIFFERENT weights, none a power of two, summing to one only to
      // rounding. Equal weights would make every summation order give the
      // same bits, which would hide exactly what the bit-for-bit test below
      // is for.
      static constexpr double kW[4] = {0.1, 0.2, 0.3, 0.4};
      w.val.push_back(kW[d]);
    }
  }
  return w;
}

/// A map whose row sums are ORDER SENSITIVE, which the four-point map is not.
///
/// Summation is non-associative only when magnitudes differ enough to be
/// absorbed. Each row here carries one weight of 1 and three of 6e-17 against
/// a unit field: ascending, the three small terms accumulate to 1.8e-16 first
/// and then push 1 up by one ulp; descending, each is absorbed into 1 and the
/// row comes out exactly 1. The 1 is placed on the row's SMALLEST local
/// column, which -- because the source partition numbers its entities in
/// descending gid order -- is its LARGEST global id, so global-id order is the
/// ascending case and local-column order is the descending one.
///
/// Without this, `map-accumulates-in-local-column-order` survives: it did, on
/// the first attempt at the test below.
fuse::RemapWeights order_sensitive_map(std::size_t n_rows, std::size_t n_cols) {
  fuse::RemapWeights w;
  w.n_rows = n_rows;
  w.n_cols = n_cols;
  w.order = 1;
  for (std::size_t j = 0; j < n_rows; ++j) {
    std::vector<std::size_t> cols;
    for (std::size_t d = 0; d < 4; ++d) cols.push_back((j * 3 + d) % n_cols);
    const std::size_t big = *std::min_element(cols.begin(), cols.end());
    for (const std::size_t c : cols) {
      w.row.push_back(static_cast<std::int32_t>(j));
      w.col.push_back(static_cast<std::int32_t>(c));
      w.val.push_back(c == big ? 1.0 : 6e-17);
    }
  }
  return w;
}

/// Named `apply_map`, not `apply`: one argument is a std::vector, so ADL
/// finds `std::apply` and resolves to it.
std::vector<double> apply_map(const fuse::RemapWeights& w,
                          const std::vector<double>& f) {
  std::vector<double> g(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    g[static_cast<std::size_t>(w.row[k])] +=
        w.val[k] * f[static_cast<std::size_t>(w.col[k])];
  }
  return g;
}

// ---- the format ---------------------------------------------------------

TEST(TransferMapIo, ARoundTripIsBitIdentical) {
  const Partition src{ids(0, 16)};
  const Partition tgt{ids(100, 8)};
  const auto      p = fuse::to_persisted(four_point_map(8, 16), src, tgt, 2, 2);
  ASSERT_TRUE(p.ok()) << p.error().message;

  const auto back = fuse::decode_transfer_map(
      fuse::encode_transfer_map(p.value()));
  ASSERT_TRUE(back.ok()) << back.error().message;
  EXPECT_EQ(back.value().tgt_gid, p.value().tgt_gid);
  EXPECT_EQ(back.value().src_gid, p.value().src_gid);
  // EQ, not NEAR: the codec writes the bit pattern, so a weight that comes
  // back merely close is a defect, and 1e-13 would hide it.
  EXPECT_EQ(back.value().val, p.value().val);
  EXPECT_EQ(back.value().identity, p.value().identity);
  EXPECT_EQ(back.value().max_col_defect, p.value().max_col_defect);
}

TEST(TransferMapIo, EntriesComeOutSortedByTargetThenSource) {
  // Local index order and global id order disagree here: local 0 is the
  // highest gid. A codec that trusted the local order would emit unsorted.
  std::vector<std::size_t> reversed = ids(100, 8);
  std::reverse(reversed.begin(), reversed.end());
  const Partition src{ids(0, 16)};
  const Partition tgt{reversed};

  const auto p = fuse::to_persisted(four_point_map(8, 16), src, tgt, 2, 2);
  ASSERT_TRUE(p.ok()) << p.error().message;
  for (std::size_t k = 1; k < p.value().size(); ++k) {
    const bool ordered =
        p.value().tgt_gid[k - 1] < p.value().tgt_gid[k] ||
        (p.value().tgt_gid[k - 1] == p.value().tgt_gid[k] &&
         p.value().src_gid[k - 1] <= p.value().src_gid[k]);
    ASSERT_TRUE(ordered) << "entry " << k << " breaks the order the format "
                            "promises, so apply_map() would sum in a different "
                            "order than the writer did";
  }
}

TEST(TransferMapIo, EveryCorruptionHasItsOwnRefusal) {
  const Partition src{ids(0, 16)};
  const Partition tgt{ids(100, 8)};
  const auto      p = fuse::to_persisted(four_point_map(8, 16), src, tgt, 2, 2);
  ASSERT_TRUE(p.ok());
  const std::vector<std::byte> good = fuse::encode_transfer_map(p.value());

  const auto code_of = [](std::span<const std::byte> b) {
    const auto r = fuse::decode_transfer_map(b);
    return r.ok() ? std::string{"<accepted>"} : r.error().code;
  };

  std::vector<std::byte> bad = good;
  bad[2] = std::byte{'X'};
  EXPECT_EQ(code_of(bad), "E-MAP-MAGIC");

  bad = good;
  bad[8] = std::byte{99};  // version, little end
  EXPECT_EQ(code_of(bad), "E-MAP-VERSION");

  // A write killed part-way: the bytes are a prefix of a valid file.
  EXPECT_EQ(code_of(std::span<const std::byte>{good}.first(good.size() - 24)),
            "E-MAP-TRUNCATED");
  EXPECT_EQ(code_of(std::span<const std::byte>{good}.first(4)),
            "E-MAP-TRUNCATED");

  // Right length, changed contents -- the case a length check cannot see.
  bad = good;
  bad[bad.size() - 3] = static_cast<std::byte>(
      std::to_integer<std::uint8_t>(bad[bad.size() - 3]) ^ 0x40);
  EXPECT_EQ(code_of(bad), "E-MAP-CHECKSUM");
}

TEST(TransferMapIo, AFileRoundTripsThroughTheFilesystem) {
  const Partition src{ids(0, 16)};
  const Partition tgt{ids(100, 8)};
  const auto      p = fuse::to_persisted(four_point_map(8, 16), src, tgt, 2, 2);
  ASSERT_TRUE(p.ok());

  const std::string path = "transfer_map_io_test.fmap";
  ASSERT_TRUE(fuse::write_transfer_map(path, p.value()).ok());
  const auto back = fuse::read_transfer_map(path);
  ASSERT_TRUE(back.ok()) << back.error().message;
  EXPECT_EQ(back.value().val, p.value().val);
  std::remove(path.c_str());

  const auto missing = fuse::read_transfer_map("does-not-exist.fmap");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.error().code, "E-MAP-IO");
}

// ---- the point of the format: a different partition ---------------------

TEST(TransferMapIo, TwoWritersReloadAsFourReadersAndReproduceTheValues) {
  // Written by two "ranks": target gids 100..103 and 104..107, each holding
  // the whole source mesh, as a replicated source would.
  const Partition src_all{ids(0, 16)};
  std::vector<fuse::PersistedTransferMap> shards;
  for (std::size_t half = 0; half < 2; ++half) {
    const Partition tgt_half{ids(100 + half * 4, 4)};
    fuse::RemapWeights w = four_point_map(4, 16);
    const auto p = fuse::to_persisted(w, src_all, tgt_half, 2, 2);
    ASSERT_TRUE(p.ok()) << p.error().message;
    shards.push_back(p.value());
  }
  const auto whole = fuse::merge_transfer_maps(shards);
  ASSERT_TRUE(whole.ok()) << whole.error().message;
  EXPECT_EQ(whole.value().size(), shards[0].size() + shards[1].size());

  // The reference: the same coupling applied in one piece.
  std::vector<double> f(16);
  for (std::size_t i = 0; i < f.size(); ++i) {
    f[i] = 1.0 + 0.5 * static_cast<double>(i);
  }
  std::vector<double> reference;
  for (std::size_t half = 0; half < 2; ++half) {
    const Partition tgt_half{ids(100 + half * 4, 4)};
    const auto      w = fuse::localise(whole.value(), src_all, tgt_half, 2, 2);
    ASSERT_TRUE(w.ok()) << w.error().message;
    for (const double v : apply_map(w.value(), f)) reference.push_back(v);
  }

  // Reloaded by FOUR readers over the same eight target entities.
  std::vector<double> reloaded;
  for (std::size_t q = 0; q < 4; ++q) {
    const Partition tgt_quarter{ids(100 + q * 2, 2)};
    const auto      w =
        fuse::localise(whole.value(), src_all, tgt_quarter, 2, 2);
    ASSERT_TRUE(w.ok()) << "quarter " << q << ": " << w.error().message;
    EXPECT_EQ(w.value().n_rows, 2u);
    for (const double v : apply_map(w.value(), f)) reloaded.push_back(v);
  }

  ASSERT_EQ(reloaded.size(), reference.size());
  for (std::size_t j = 0; j < reference.size(); ++j) {
    EXPECT_NEAR(reloaded[j], reference[j], 1e-13) << "target " << j;
  }
  // Every row still sums to one, which is the property the map exists to
  // have; a reload that lost an entry would show up here and not in a norm.
  std::vector<double> ones(16, 1.0);
  for (std::size_t q = 0; q < 4; ++q) {
    const Partition tgt_quarter{ids(100 + q * 2, 2)};
    const auto      w =
        fuse::localise(whole.value(), src_all, tgt_quarter, 2, 2);
    ASSERT_TRUE(w.ok());
    for (const double v : apply_map(w.value(), ones)) EXPECT_NEAR(v, 1.0, 1e-15);
  }
}

// ---- bit-for-bit across rank counts ------------------------------------

// The property: 1/2/3/4/5 "tasks" must give
// bit-for-bit the same answer, with accumulation ordered by global id.
//
// EXPECT_EQ on doubles, not NEAR: a tolerance would pass for the wrong
// reason. What makes it hold is that a target entity is owned by exactly one
// rank -- so every entry of a row is local -- and `localise()` sums a row in
// ascending source GLOBAL id, which is the same sequence of numbers on every
// partition. Local column indices are not: they depend on how many entities
// the rank holds. The source partition here numbers its entities in DESCENDING
// gid order for that reason, so local-column order and global-id order
// disagree and a mutation that used the former is detectable.
TEST(TransferMapIo, EveryRankCountGivesBitIdenticalValues) {
  // Each rank's source view holds every source entity -- as it would after a
  // halo gather -- but NUMBERED DIFFERENTLY: its own share first, then the
  // rest. That is the crux. A replicated source numbered identically on every
  // reader makes local-column order partition-independent too, and then
  // nothing distinguishes it from global-id order; the first version of this
  // test did exactly that and could not kill
  // `map-accumulates-in-local-column-order`.
  const auto src_for = [](std::size_t ranks, std::size_t r) {
    std::vector<std::size_t> g = ids(0, 16);
    std::rotate(g.begin(), g.begin() + static_cast<long>((r * 5 + ranks) % 16),
                g.end());
    return Partition{g};
  };

  const std::size_t n_tgt = 8;
  const Partition   src_writer = src_for(1, 0);
  const Partition   tgt_whole{ids(100, n_tgt)};
  const auto        p = fuse::to_persisted(order_sensitive_map(n_tgt, 16),
                                           src_writer, tgt_whole, 2, 2);
  ASSERT_TRUE(p.ok()) << p.error().message;

  // Unit field: the row value is then exactly the construction above, so the
  // two orders differ by one ulp of 1 rather than by an accident of the data.
  // Being constant, it is also independent of each rank's local numbering --
  // so any difference between rank counts comes from the summation order and
  // not from which local slot a value landed in.
  const std::vector<double> f(16, 1.0);

  // Global target id -> value, so answers from different partitions are
  // comparable entity by entity rather than by local position.
  const auto run_on = [&](std::size_t ranks) {
    std::vector<std::pair<std::size_t, double>> out;
    for (std::size_t r = 0; r < ranks; ++r) {
      // Uneven on purpose for 3 and 5: the first ranks take one more entity,
      // and at 5 ranks over 8 entities some hold 2 and some hold 1.
      const std::size_t base = n_tgt / ranks;
      const std::size_t extra = n_tgt % ranks;
      const std::size_t first = r * base + std::min(r, extra);
      const std::size_t count = base + (r < extra ? 1 : 0);
      if (count == 0) continue;
      const Partition mine{ids(100 + first, count)};
      const Partition src_here = src_for(ranks, r);
      const auto      w = fuse::localise(p.value(), src_here, mine, 2, 2);
      if (!w.ok()) return std::vector<std::pair<std::size_t, double>>{};

      // The documented invariant, checked where it is produced.
      for (std::size_t k = 1; k < w.value().row.size(); ++k) {
        EXPECT_LE(w.value().row[k - 1], w.value().row[k])
            << "RemapWeights::row must be ascending; ranks=" << ranks;
      }
      const std::vector<double> g = apply_map(w.value(), f);
      for (std::size_t j = 0; j < g.size(); ++j) {
        out.emplace_back(100 + first + j, g[j]);
      }
    }
    std::sort(out.begin(), out.end());
    return out;
  };

  const auto one = run_on(1);
  ASSERT_EQ(one.size(), n_tgt) << "the single-partition run must localise";
  for (const std::size_t ranks : {2u, 3u, 4u, 5u}) {
    const auto got = run_on(ranks);
    ASSERT_EQ(got.size(), one.size()) << "ranks=" << ranks;
    for (std::size_t j = 0; j < got.size(); ++j) {
      EXPECT_EQ(got[j].first, one[j].first) << "ranks=" << ranks;
      EXPECT_EQ(got[j].second, one[j].second)
          << "target " << got[j].first << " at ranks=" << ranks
          << " differs from the 1-task answer in its bits; accumulation is "
             "no longer ordered by source global id";
    }
  }
}

TEST(TransferMapIo, AMismatchedPartitionIsRefusedNotSilentlyZeroFilled) {
  const Partition src_all{ids(0, 16)};
  const Partition tgt{ids(100, 8)};
  const auto      p = fuse::to_persisted(four_point_map(8, 16), src_all, tgt, 2, 2);
  ASSERT_TRUE(p.ok());

  // A reader whose target entities are not the ones the map was written for.
  // Applying the map would leave every value at zero, which looks like a
  // field that was transferred and happened to be zero.
  const Partition elsewhere{ids(900, 8)};
  const auto      r = fuse::localise(p.value(), src_all, elsewhere, 2, 2);
  ASSERT_FALSE(r.ok()) << "a map for other entities must not localise";
  EXPECT_EQ(r.error().code, "E-MAP-COVERAGE");
  EXPECT_NE(r.error().message.find("write zero"), std::string::npos)
      << r.error().message;

  // Half right is still rejected: coverage is per entity, not a fraction.
  std::vector<std::size_t> overlapping = ids(104, 8);  // 104..111, map has ..107
  const Partition          half{overlapping};
  const auto               h = fuse::localise(p.value(), src_all, half, 2, 2);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-MAP-COVERAGE");

  // A source entity the reader does not hold is the other direction, and a
  // different remedy: the value is not here to apply.
  const Partition src_part{ids(0, 4)};
  const auto      s = fuse::localise(p.value(), src_part, tgt, 2, 2);
  ASSERT_FALSE(s.ok());
  EXPECT_EQ(s.error().code, "E-MAP-COVERAGE");
  EXPECT_NE(s.error().message.find("does not hold"), std::string::npos)
      << s.error().message;
}

TEST(TransferMapIo, AMapOfADifferentKindIsRefused) {
  const Partition src{ids(0, 16)};
  const Partition tgt{ids(100, 8)};
  fuse::RemapWeights w = four_point_map(8, 16);
  w.order = 2;
  w.monotone = true;
  const auto p = fuse::to_persisted(w, src, tgt, 2, 2);
  ASSERT_TRUE(p.ok());
  EXPECT_EQ(p.value().identity.order, 2);
  EXPECT_TRUE(p.value().identity.monotone);

  const fuse::TransferMapIdentity want{
      .order = 1, .monotone = false, .src_dim = 2, .tgt_dim = 2};
  const auto r = fuse::localise(p.value(), src, tgt, 2, 2, &want);
  ASSERT_FALSE(r.ok()) << "an order-2 monotone map is not an order-1 plain one";
  EXPECT_EQ(r.error().code, "E-MAP-IDENTITY");

  const fuse::TransferMapIdentity actual{
      .order = 2, .monotone = true, .src_dim = 2, .tgt_dim = 2};
  EXPECT_TRUE(fuse::localise(p.value(), src, tgt, 2, 2, &actual).ok());
}

TEST(TransferMapIo, ShardsThatBothClaimAnEntryAreRefused) {
  const Partition src{ids(0, 16)};
  const Partition tgt{ids(100, 4)};
  const auto      p = fuse::to_persisted(four_point_map(4, 16), src, tgt, 2, 2);
  ASSERT_TRUE(p.ok());

  const std::vector<fuse::PersistedTransferMap> twice{p.value(), p.value()};
  const auto r = fuse::merge_transfer_maps(twice);
  ASSERT_FALSE(r.ok()) << "the same shard twice would double every weight";
  EXPECT_EQ(r.error().code, "E-MAP-DUPLICATE");
}

TEST(TransferMapIo, AnIndexOutsideItsViewCannotBeGivenAGlobalId) {
  const Partition    src{ids(0, 4)};
  const Partition    tgt{ids(100, 4)};
  fuse::RemapWeights w = four_point_map(4, 4);
  w.col[0] = 9;  // no such source entity in this view
  const auto p = fuse::to_persisted(w, src, tgt, 2, 2);
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-MAP-SHAPE");
}

}  // namespace
