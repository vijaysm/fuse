// CommGraph and the Crystal Router underneath it, under a real mpiexec.
//
// This is the regression matrix of docs/specs/2026-09-11-comm-graph-design.md
// §7. Every test here is named after what it pins rather than after the
// function it calls, and every load-bearing one was checked against a
// deliberately broken build before being trusted -- the discipline
// docs/HANDOFF.md "Lessons paid for" item 2 exists for.
//
// WHY THIS IS NOT A soma_add_test(), for the same three reasons
// parallel/mpi_transport_test.cpp gives: it calls MPI_Init, it is launched by
// mpiexec, and a GoogleTest assertion failure unwinds one rank out of a TEST
// body while its peers walk on into the next collective, which is a hang and
// not a failure. The listener in main() turns the first failed assertion on
// any rank into MPI_Abort.
//
// WHY THE FIXTURES ARE DELIBERATELY NON-CORRESPONDING. Sources hold
// CONTIGUOUS blocks of global ids; destinations hold a STRIDED, REVERSED
// selection. So no destination slot lines up positionally with a source slot,
// every destination talks to every source, and an implementation that matched
// by position -- or that only ever exchanged with its opposite number --
// produces wrong numbers instead of passing by luck. This is the same pattern
// mpi_transport_test.cpp uses, and test 14 of the matrix is precisely the
// requirement that it be driven through two independently MPI_Comm_split-built
// communicators rather than through one shared communicator with the rank sets
// handed in by the test.

#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MPI

#include <mpi.h>
#include <soma/mesh/mesh_view.hpp>
#include <soma/transport/comm_graph.hpp>
#include <soma/transport/transport.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "transport/crystal_router.hpp"

using soma::CommGraph;
using soma::CommGraphOptions;
using soma::CommGraphSide;
using soma::MatchMode;
using soma::MpiTransport;
using soma::Point;
using soma::TransportSpec;
using soma::detail::CrystalRouter;
using soma::detail::Envelope;

namespace {

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

int world_size() {
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

// ---------------------------------------------------------------------------
// A MeshView the test drives directly.
//
// Not SerialMeshBackend: these tests need a DIFFERENT partition on every rank,
// with sparse non-index global ids and hand-placed centroids, which is exactly
// the shape `MeshView`'s contract is written for ("global ids are typically
// sparse after partitioning ... and must never be used to subscript
// anything") and exactly what a structured serial backend cannot produce.
// ---------------------------------------------------------------------------
class TestMesh final : public soma::MeshView {
 public:
  TestMesh(int dim, std::vector<std::size_t> gids, std::vector<Point> pts)
      : dim_(dim), gid_(std::move(gids)), pt_(std::move(pts)),
        owned_(gid_.size(), true) {}

  void set_ghost(std::size_t i) { owned_[i] = false; }

  [[nodiscard]] int dimension() const override { return dim_; }

  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == dim_ ? gid_.size() : 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    if (dim != dim_) return 0;
    std::size_t n = 0;
    for (const bool b : owned_) n += b ? 1 : 0;
    return n;
  }
  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    check(dim, i);
    return owned_[i];
  }
  [[nodiscard]] Point centroid(int dim, std::size_t i) const override {
    check(dim, i);
    return pt_[i];
  }
  [[nodiscard]] double measure(int dim, std::size_t i) const override {
    check(dim, i);
    return 1.0;
  }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    check(dim, i);
    return gid_[i];
  }

 private:
  void check(int dim, std::size_t i) const {
    if (dim != dim_ || i >= gid_.size()) {
      throw std::out_of_range("TestMesh: dimension or index out of range");
    }
  }

  int                      dim_;
  std::vector<std::size_t> gid_;
  std::vector<Point>       pt_;
  std::vector<bool>        owned_;
};

// ---------------------------------------------------------------------------
// The three rank-overlap shapes of the matrix's rows 1-6.
// ---------------------------------------------------------------------------
enum class Shape { FullyDisjoint, FullyOverlapping, PartialOverlap };

const char* name_of(Shape s) {
  switch (s) {
    case Shape::FullyDisjoint:    return "fully-disjoint";
    case Shape::FullyOverlapping: return "fully-overlapping";
    case Shape::PartialOverlap:   return "partially-overlapping";
  }
  return "?";
}

/// Which world ranks are in each component, for `shape` at `size` ranks.
/// PartialOverlap puts every rank but the last in the source and the last two
/// in the destination, so exactly one rank is in both at any size >= 2.
void membership(Shape shape, int size, int rank, bool* in_src, bool* in_dst) {
  switch (shape) {
    case Shape::FullyDisjoint:
      *in_src = rank < size / 2;
      *in_dst = rank >= size / 2;
      break;
    case Shape::FullyOverlapping:
      *in_src = true;
      *in_dst = true;
      break;
    case Shape::PartialOverlap:
      *in_src = rank <= size - 2;
      *in_dst = rank >= size - 2;
      break;
  }
}

/// A communicator over the ranks for which `member` is true, or
/// MPI_COMM_NULL. Built with MPI_Comm_split, so `CommGraph` is handed two
/// communicators it did not create and knows nothing about -- which is what
/// test 14 of the matrix asks for.
MPI_Comm split_for(bool member) {
  MPI_Comm c = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, member ? 0 : MPI_UNDEFINED, world_rank(), &c);
  return c;
}

/// The value carried by source global id `g`. Distinct for every id, and not
/// a small integer, so a stale zero or a shifted index is visible.
double value_of(std::size_t g) { return static_cast<double>(g) * 10.0 + 1.0; }

/// Global ids are sparse and are NOT indices -- `MeshView::global_id`
/// documents that they may exceed `entity_count` by orders of magnitude, and
/// an implementation that used one to subscript would pass against a dense
/// numbering.
std::size_t gid_of(std::size_t k) { return 100000 + 7 * k; }

/// Rank `pos` of `n`, source side: a contiguous block.
std::vector<std::size_t> source_block(std::size_t total, int n, int pos) {
  const std::size_t per = total / static_cast<std::size_t>(n);
  const std::size_t lo = static_cast<std::size_t>(pos) * per;
  const std::size_t hi =
      pos + 1 == n ? total : lo + per;  // the last rank absorbs the remainder
  std::vector<std::size_t> k;
  for (std::size_t i = lo; i < hi; ++i) k.push_back(i);
  return k;
}

/// Rank `pos` of `m`, destination side: a strided, descending selection.
///
/// DESCENDING MATTERS, and mpi_transport_test.cpp records why: with ascending
/// destinations against ascending source blocks the unpack permutation is the
/// identity, and an implementation that ignored it entirely still passes.
std::vector<std::size_t> dest_stride(std::size_t total, int m, int pos) {
  std::vector<std::size_t> k;
  for (std::size_t i = static_cast<std::size_t>(pos); i < total;
       i += static_cast<std::size_t>(m)) {
    k.push_back(i);
  }
  std::reverse(k.begin(), k.end());
  return k;
}

int position_in(const std::vector<int>& v, int r) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (v[i] == r) return static_cast<int>(i);
  }
  return -1;
}

std::vector<int> members_of(Shape shape, int size, bool source) {
  std::vector<int> v;
  for (int r = 0; r < size; ++r) {
    bool s = false, d = false;
    membership(shape, size, r, &s, &d);
    if (source ? s : d) v.push_back(r);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Delivering a field through the graph. This is the end-to-end check: the
// matching is only right if the numbers that arrive are right.
// ---------------------------------------------------------------------------

/// Applies `g` to move `src_vals` into `dst_vals`. Local pairs are memcpy'd
/// and never reach a Transport; the cross-rank remainder goes through a real
/// `MpiTransport` over MPI_COMM_WORLD when the graph says that is expressible,
/// and through the per-peer plan otherwise.
///
/// COLLECTIVE over MPI_COMM_WORLD. Every rank calls it, including ranks in
/// neither component.
void deliver(const CommGraph& g, const std::vector<double>& src_vals,
             std::vector<double>* dst_vals) {
  for (const soma::LocalCopyPair& p : g.local_pairs()) {
    (*dst_vals)[p.dest_local_index] = src_vals[p.source_local_index];
  }

  const TransportSpec ss = g.source_spec();
  const TransportSpec ds = g.dest_spec();

  if (g.specs_are_disjoint()) {
    // A rank is a source, or a destination, or neither -- never both, which
    // is exactly what `specs_are_disjoint()` asserts.
    const bool    is_dst = !ds.global_ids.empty();
    TransportSpec mine = is_dst ? ds : ss;

    MpiTransport t;
    ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
    const auto plan = t.build_plan(mine);
    ASSERT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;

    std::vector<double> sbuf(mine.elements, 0.0);
    std::vector<double> rbuf(mine.elements, 0.0);
    if (!is_dst) {
      for (std::size_t i = 0; i < mine.elements; ++i) {
        sbuf[i] = src_vals[g.source_indices()[i]];
      }
    }
    ASSERT_TRUE(t.prepare(mine).ok());
    const auto mv = t.move(mine.elements == 0 ? nullptr : sbuf.data(),
                           mine.elements == 0 ? nullptr : rbuf.data(),
                           mine.elements);
    ASSERT_TRUE(mv.ok()) << mv.error().code << ": " << mv.error().message;
    if (is_dst) {
      for (std::size_t i = 0; i < mine.elements; ++i) {
        (*dst_vals)[g.dest_indices()[i]] = rbuf[i];
      }
    }
    ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
    return;
  }

  // Overlapping rank sets: `MpiTransport` cannot express them (R24), so the
  // per-peer plan is executed directly. `send_plan()`/`recv_plan()` are
  // ordered on the same key at both ends, which is what makes this line up.
  std::vector<MPI_Request> reqs;
  std::vector<std::vector<double>> sbuf(g.send_plan().size());
  std::vector<std::vector<double>> rbuf(g.recv_plan().size());
  for (std::size_t p = 0; p < g.recv_plan().size(); ++p) {
    rbuf[p].assign(g.recv_plan()[p].local_indices.size(), 0.0);
    reqs.emplace_back();
    ASSERT_EQ(MPI_Irecv(rbuf[p].data(), static_cast<int>(rbuf[p].size()),
                        MPI_DOUBLE, g.recv_plan()[p].base_rank, 991,
                        MPI_COMM_WORLD, &reqs.back()),
              MPI_SUCCESS);
  }
  for (std::size_t p = 0; p < g.send_plan().size(); ++p) {
    for (const std::size_t li : g.send_plan()[p].local_indices) {
      sbuf[p].push_back(src_vals[li]);
    }
    reqs.emplace_back();
    ASSERT_EQ(MPI_Isend(sbuf[p].data(), static_cast<int>(sbuf[p].size()),
                        MPI_DOUBLE, g.send_plan()[p].base_rank, 991,
                        MPI_COMM_WORLD, &reqs.back()),
              MPI_SUCCESS);
  }
  if (!reqs.empty()) {
    ASSERT_EQ(MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                          MPI_STATUSES_IGNORE),
              MPI_SUCCESS);
  }
  for (std::size_t p = 0; p < g.recv_plan().size(); ++p) {
    for (std::size_t i = 0; i < rbuf[p].size(); ++i) {
      (*dst_vals)[g.recv_plan()[p].local_indices[i]] = rbuf[p][i];
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ---------------------------------------------------------------------------
// The GlobalId fixture: one shared identity space, non-corresponding
// partitions.
// ---------------------------------------------------------------------------
constexpr std::size_t kGlobalEntities = 24;

struct Fixture {
  bool                     in_src = false;
  bool                     in_dst = false;
  MPI_Comm                 src_comm = MPI_COMM_NULL;
  MPI_Comm                 dst_comm = MPI_COMM_NULL;
  std::vector<std::size_t> src_k, dst_k;  // global entity indices held
  std::unique_ptr<TestMesh> src_mesh, dst_mesh;

  Fixture() = default;
  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  // MPI_Comm_free is not idempotent, so the moved-from fixture must forget its
  // communicators rather than free them a second time in its destructor.
  // `= default` would not: MPI_Comm is a handle, and copying one leaves both
  // objects believing they own it.
  Fixture(Fixture&& o) noexcept
      : in_src(o.in_src), in_dst(o.in_dst), src_comm(o.src_comm),
        dst_comm(o.dst_comm), src_k(std::move(o.src_k)),
        dst_k(std::move(o.dst_k)), src_mesh(std::move(o.src_mesh)),
        dst_mesh(std::move(o.dst_mesh)) {
    o.src_comm = MPI_COMM_NULL;
    o.dst_comm = MPI_COMM_NULL;
  }
  Fixture& operator=(Fixture&&) = delete;

  ~Fixture() {
    if (src_comm != MPI_COMM_NULL) MPI_Comm_free(&src_comm);
    if (dst_comm != MPI_COMM_NULL) MPI_Comm_free(&dst_comm);
  }
};

/// Position of the point carrying global entity index `k`, on the source side
/// and on the destination side. In NearestNeighbor mode the two lattices are
/// deliberately NOT coincident: the destination is shifted by `kShift`, well
/// inside a tolerance of `kTol`, so the nearest match is unambiguous and is a
/// non-trivial permutation of the source's own ordering.
constexpr double kSpacing = 1.0;
constexpr double kShift = 0.1;
constexpr double kTol = 0.3;

Point source_point(std::size_t k, int dim) {
  const auto d = static_cast<double>(k);
  switch (dim) {
    case 1:  return Point{d * kSpacing, 0.0, 0.0};
    case 2:  return Point{static_cast<double>(k % 6) * kSpacing,
                          static_cast<double>(k / 6) * kSpacing, 0.0};
    default: return Point{static_cast<double>(k % 3) * kSpacing,
                          static_cast<double>((k / 3) % 2) * kSpacing,
                          static_cast<double>(k / 6) * kSpacing};
  }
}

Point dest_point(std::size_t k, int dim) {
  Point p = source_point(k, dim);
  p.x += kShift;
  if (dim >= 2) p.y += kShift;
  if (dim >= 3) p.z += kShift;
  return p;
}

/// The independent oracle for NearestNeighbor mode: brute force over the whole
/// global point set, computed serially on every rank from the same closed-form
/// fixture. Deliberately not a re-derivation of the library's own search --
/// it shares no code with it.
std::size_t expected_match(std::size_t dst_k, int dim) {
  const Point q = dest_point(dst_k, dim);
  double      best = std::numeric_limits<double>::infinity();
  std::size_t win = 0;
  for (std::size_t k = 0; k < kGlobalEntities; ++k) {
    const Point s = source_point(k, dim);
    const double d = std::sqrt((s.x - q.x) * (s.x - q.x) +
                               (s.y - q.y) * (s.y - q.y) +
                               (s.z - q.z) * (s.z - q.z));
    if (d < best || (d == best && gid_of(k) < gid_of(win))) {
      best = d;
      win = k;
    }
  }
  return win;
}

/// Builds both sides for `shape`. `coincident` selects the GlobalId fixture
/// (destination entities carry the SAME ids as the source's) from the
/// NearestNeighbor one (destination entities carry ids of their own, matched
/// only by position).
Fixture make_fixture(Shape shape, int dim, bool coincident) {
  Fixture f;
  const int size = world_size();
  const int rank = world_rank();
  membership(shape, size, rank, &f.in_src, &f.in_dst);
  f.src_comm = split_for(f.in_src);
  f.dst_comm = split_for(f.in_dst);

  const std::vector<int> src_members = members_of(shape, size, true);
  const std::vector<int> dst_members = members_of(shape, size, false);

  if (f.in_src) {
    f.src_k = source_block(kGlobalEntities,
                           static_cast<int>(src_members.size()),
                           position_in(src_members, rank));
    std::vector<std::size_t> gids;
    std::vector<Point>       pts;
    for (const std::size_t k : f.src_k) {
      gids.push_back(gid_of(k));
      pts.push_back(source_point(k, dim));
    }
    f.src_mesh = std::make_unique<TestMesh>(dim, gids, pts);
  }
  if (f.in_dst) {
    f.dst_k = dest_stride(kGlobalEntities,
                          static_cast<int>(dst_members.size()),
                          position_in(dst_members, rank));
    std::vector<std::size_t> gids;
    std::vector<Point>       pts;
    for (const std::size_t k : f.dst_k) {
      // In the non-coincident fixture the destination's own ids are from a
      // disjoint numbering, so a GlobalId match on them is impossible and a
      // NearestNeighbor implementation that secretly compared ids would fail.
      gids.push_back(coincident ? gid_of(k) : 500000 + 3 * k);
      pts.push_back(dest_point(k, dim));
    }
    f.dst_mesh = std::make_unique<TestMesh>(dim, gids, pts);
  }
  return f;
}

CommGraphSide side_of(const Fixture& f, bool source, int dim) {
  CommGraphSide s;
  s.mesh = source ? f.src_mesh.get() : f.dst_mesh.get();
  s.dim = dim;
  s.comm_handle = MPI_Comm_c2f(source ? f.src_comm : f.dst_comm);
  return s;
}

/// Runs one cell of the 6-way matrix and checks the numbers that arrive.
void run_matrix_cell(Shape shape, MatchMode mode, int dim) {
  const bool coincident = mode == MatchMode::GlobalId;
  Fixture    f = make_fixture(shape, dim, coincident);

  CommGraphOptions opts;
  opts.mode = mode;
  opts.nn_tolerance = coincident ? 0.0 : kTol;

  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD),
                                      side_of(f, true, dim),
                                      side_of(f, false, dim), opts);
  ASSERT_TRUE(built.ok()) << name_of(shape) << ": " << built.error().code
                          << ": " << built.error().message;
  const CommGraph& g = built.value();

  // Every destination entity is matched exactly once -- no gaps, no doubles.
  std::vector<int> hits(f.dst_k.size(), 0);
  for (const soma::LocalCopyPair& p : g.local_pairs()) ++hits[p.dest_local_index];
  for (const soma::CommGraphPeer& p : g.recv_plan()) {
    for (const std::size_t li : p.local_indices) ++hits[li];
  }
  for (std::size_t i = 0; i < hits.size(); ++i) {
    ASSERT_EQ(hits[i], 1) << name_of(shape) << " rank " << world_rank()
                          << ": destination entity " << i << " (global entity "
                          << f.dst_k[i] << ") was matched " << hits[i]
                          << " times, expected exactly once";
  }

  std::vector<double> src_vals(f.src_k.size(), 0.0);
  for (std::size_t i = 0; i < f.src_k.size(); ++i) {
    src_vals[i] = value_of(gid_of(f.src_k[i]));
  }
  std::vector<double> dst_vals(f.dst_k.size(), -12345.0);
  deliver(g, src_vals, &dst_vals);

  for (std::size_t i = 0; i < f.dst_k.size(); ++i) {
    const std::size_t want_k =
        coincident ? f.dst_k[i] : expected_match(f.dst_k[i], dim);
    ASSERT_DOUBLE_EQ(dst_vals[i], value_of(gid_of(want_k)))
        << name_of(shape) << " rank " << world_rank() << " dim " << dim
        << ": destination entity " << i << " (global entity " << f.dst_k[i]
        << ") received the wrong value";
  }

  std::printf("[%s/%s dim %d rank %d/%d] local=%zu cross-send=%zu "
              "cross-recv=%zu bucket=%zu rounds=%d\n",
              name_of(shape), coincident ? "GlobalId" : "NearestNeighbor", dim,
              world_rank(), world_size(), g.local_pairs().size(),
              g.source_indices().size(), g.dest_indices().size(),
              g.stats().bucket_records, g.stats().router_rounds);
  std::fflush(stdout);
}

}  // namespace

// ===========================================================================
// The Crystal Router itself
// ===========================================================================

// Everything reaches the rank it names, from everywhere to everywhere. The
// all-to-all pattern is the one that catches a router that only ever ships
// across the current split correctly in one direction.
TEST(CrystalRouter, DeliversEveryEnvelopeToTheRankItNames) {
  const int size = world_size();
  const int rank = world_rank();

  CrystalRouter r(MPI_COMM_WORLD);
  std::vector<Envelope> out;
  for (int d = 0; d < size; ++d) {
    Envelope e;
    e.dest_rank = d;
    // Payload identifies the sender AND the intended receiver, so a message
    // delivered to the wrong rank is caught rather than merely counted.
    const auto tag = static_cast<std::uint32_t>(rank * 1000 + d);
    e.payload.assign(reinterpret_cast<const std::uint8_t*>(&tag),
                     reinterpret_cast<const std::uint8_t*>(&tag) + 4);
    out.push_back(std::move(e));
  }
  const auto got = r.route(std::move(out));
  ASSERT_TRUE(got.ok()) << got.error().code << ": " << got.error().message;

  std::vector<int> from(static_cast<std::size_t>(size), 0);
  for (const Envelope& e : got.value()) {
    ASSERT_EQ(e.payload.size(), 4u);
    std::uint32_t tag = 0;
    std::memcpy(&tag, e.payload.data(), 4);
    const int sender = static_cast<int>(tag) / 1000;
    const int target = static_cast<int>(tag) % 1000;
    ASSERT_EQ(target, rank) << "a message for rank " << target
                            << " was delivered to rank " << rank;
    ASSERT_GE(sender, 0);
    ASSERT_LT(sender, size);
    ++from[static_cast<std::size_t>(sender)];
  }
  for (int s = 0; s < size; ++s) {
    ASSERT_EQ(from[static_cast<std::size_t>(s)], 1)
        << "rank " << rank << " received " << from[static_cast<std::size_t>(s)]
        << " messages from rank " << s << ", expected exactly one";
  }
}

// O(log P), not O(P). The whole reason this primitive exists instead of an
// allgather is the round count, so it is asserted rather than assumed.
//
// THE BOUND IS ON THE MAXIMUM OVER RANKS, not on every rank. Each round halves
// a rank's range, and at an odd size the two halves differ by one, so a rank
// that lands in the smaller half runs out of range -- and rounds -- sooner: at
// three ranks, rank 2 finishes in one round while ranks 0 and 1 take two.
// That is correct, and asserting `== ceil(log2(P))` per rank would be
// asserting a bug. What must hold is that NOBODY exceeds the bound and that
// somebody reaches it.
TEST(CrystalRouter, NoRankExceedsCeilLogTwoPRounds) {
  const int size = world_size();
  CrystalRouter r(MPI_COMM_WORLD);
  const auto got = r.route({});
  ASSERT_TRUE(got.ok());
  ASSERT_TRUE(got.value().empty());

  int bound = 0;
  for (int n = 1; n < size; n *= 2) ++bound;

  const int mine = r.last_rounds();
  EXPECT_LE(mine, bound) << "rank " << world_rank() << " took " << mine
                         << " rounds, above ceil(log2(" << size
                         << ")) = " << bound;
  if (size > 1) EXPECT_GE(mine, 1);

  int worst = 0;
  ASSERT_EQ(MPI_Allreduce(&mine, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_EQ(worst, bound)
      << "the deepest rank took " << worst << " rounds at " << size
      << " ranks; the router's bound is ceil(log2(P)) = " << bound;
}

// A rank with nothing to send still enters every round -- the same rule
// MpiTransport states for MPI_Alltoallv. Only rank 0 sends here, so every
// other rank calls route() with an empty vector; if an empty call short
// circuited, this would hang rather than fail.
TEST(CrystalRouter, ARankWithNothingToSendStillParticipates) {
  const int size = world_size();
  const int rank = world_rank();
  CrystalRouter r(MPI_COMM_WORLD);

  std::vector<Envelope> out;
  if (rank == 0) {
    for (int d = 0; d < size; ++d) {
      Envelope e;
      e.dest_rank = d;
      e.payload.assign(3, static_cast<std::uint8_t>(d));
      out.push_back(std::move(e));
    }
  }
  const auto got = r.route(std::move(out));
  ASSERT_TRUE(got.ok()) << got.error().message;
  ASSERT_EQ(got.value().size(), 1u);
  ASSERT_EQ(got.value()[0].payload.size(), 3u);
  EXPECT_EQ(got.value()[0].payload[0], static_cast<std::uint8_t>(rank));
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Variable-length payloads survive the pack/unpack, including empty ones. A
// router that assumed a fixed record size would pass every test above.
TEST(CrystalRouter, CarriesPayloadsOfDifferentLengthsIncludingEmpty) {
  const int size = world_size();
  const int rank = world_rank();
  CrystalRouter r(MPI_COMM_WORLD);

  std::vector<Envelope> out;
  for (int d = 0; d < size; ++d) {
    Envelope e;
    e.dest_rank = d;
    const std::size_t n = static_cast<std::size_t>((rank * 5 + d * 3) % 7);
    for (std::size_t i = 0; i < n; ++i) {
      e.payload.push_back(static_cast<std::uint8_t>(rank + i));
    }
    out.push_back(std::move(e));
  }
  const auto got = r.route(std::move(out));
  ASSERT_TRUE(got.ok()) << got.error().message;
  ASSERT_EQ(got.value().size(), static_cast<std::size_t>(size));

  std::vector<bool> seen(static_cast<std::size_t>(size), false);
  for (const Envelope& e : got.value()) {
    // Reconstruct which sender this must have come from: only one length is
    // consistent with a given first byte, so check every candidate.
    bool matched = false;
    for (int s = 0; s < size && !matched; ++s) {
      const std::size_t n = static_cast<std::size_t>((s * 5 + rank * 3) % 7);
      if (e.payload.size() != n || seen[static_cast<std::size_t>(s)]) continue;
      bool all = true;
      for (std::size_t i = 0; i < n; ++i) {
        if (e.payload[i] != static_cast<std::uint8_t>(s + i)) all = false;
      }
      if (!all) continue;
      seen[static_cast<std::size_t>(s)] = true;
      matched = true;
    }
    ASSERT_TRUE(matched) << "rank " << rank
                         << " received a payload matching no sender";
  }
}

// The out-of-range check is agreed collectively before anything is sent, so
// one rank's bad envelope is an error return on EVERY rank, not a hang.
TEST(CrystalRouter, AnOutOfRangeDestinationIsRefusedOnEveryRank) {
  const int size = world_size();
  CrystalRouter r(MPI_COMM_WORLD);

  std::vector<Envelope> out;
  if (world_rank() == 0) {
    Envelope e;
    e.dest_rank = size;  // one past the end
    e.payload.assign(1, 0);
    out.push_back(std::move(e));
  }
  const auto got = r.route(std::move(out));
  ASSERT_FALSE(got.ok());
  EXPECT_EQ(got.error().code, "E-CROUTER-RANK");
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ===========================================================================
// Matrix rows 1-6: rank-overlap shape x matching mode
// ===========================================================================

TEST(CommGraph, FullyDisjointRanksMatchByGlobalId) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::FullyDisjoint, MatchMode::GlobalId, 1);
}

TEST(CommGraph, FullyOverlappingRanksMatchByGlobalId) {
  run_matrix_cell(Shape::FullyOverlapping, MatchMode::GlobalId, 1);
}

TEST(CommGraph, PartiallyOverlappingRanksMatchByGlobalId) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::PartialOverlap, MatchMode::GlobalId, 1);
}

TEST(CommGraph, FullyDisjointRanksMatchByNearestNeighbor) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::FullyDisjoint, MatchMode::NearestNeighbor, 1);
}

TEST(CommGraph, FullyOverlappingRanksMatchByNearestNeighbor) {
  run_matrix_cell(Shape::FullyOverlapping, MatchMode::NearestNeighbor, 1);
}

TEST(CommGraph, PartiallyOverlappingRanksMatchByNearestNeighbor) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::PartialOverlap, MatchMode::NearestNeighbor, 1);
}

// ===========================================================================
// 7: the same-rank shortcut is load bearing, not an optimization
// ===========================================================================

// With identical partitions on both sides, every match has both ends on one
// rank. Nothing at all may reach MpiTransport -- and the proof that it cannot
// is that MpiTransport REFUSES the spec this graph produces: the two rank sets
// are identical, which is the overlap it rejects by design (R24). If the
// shortcut were removed, those matches would have to travel through exactly
// the mechanism that will not carry them.
TEST(CommGraph, AMatchWithBothEndsOnOneRankNeverReachesMpiTransport) {
  const int  rank = world_rank();
  const int  size = world_size();
  const auto per = kGlobalEntities / static_cast<std::size_t>(size);
  const auto lo = static_cast<std::size_t>(rank) * per;
  const auto hi = rank + 1 == size ? kGlobalEntities : lo + per;

  std::vector<std::size_t> gids;
  std::vector<Point>       pts;
  for (std::size_t k = lo; k < hi; ++k) {
    gids.push_back(gid_of(k));
    pts.push_back(source_point(k, 1));
  }
  TestMesh mesh(1, gids, pts);

  CommGraphSide s;
  s.mesh = &mesh;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);

  CommGraphOptions opts;
  opts.mode = MatchMode::GlobalId;

  const auto built =
      CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, s, opts);
  ASSERT_TRUE(built.ok()) << built.error().code << ": "
                          << built.error().message;
  const CommGraph& g = built.value();

  EXPECT_EQ(g.local_pairs().size(), hi - lo);
  EXPECT_TRUE(g.send_plan().empty());
  EXPECT_TRUE(g.recv_plan().empty());
  for (const soma::LocalCopyPair& p : g.local_pairs()) {
    EXPECT_EQ(p.source_local_index, p.dest_local_index);
  }

  if (size > 1) {
    // The refusal that makes the shortcut necessary rather than merely fast.
    EXPECT_FALSE(g.specs_are_disjoint());
    MpiTransport t;
    ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
    const auto plan = t.build_plan(g.source_spec());
    ASSERT_FALSE(plan.ok())
        << "MpiTransport accepted a spec whose rank sets are identical; the "
           "same-rank shortcut would then be only an optimization";
    EXPECT_EQ(plan.error().code, "E-TRANSPORT-RANKS");
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ===========================================================================
// 8: bidirectionality via transpose()
// ===========================================================================

// transpose() performs NO collective. The test proves it by calling it a
// DIFFERENT NUMBER OF TIMES ON DIFFERENT RANKS: if it contained any
// communication at all the job would hang and the CTest entry would fail on
// its timeout. Then the reverse leg is driven end to end to show the
// relabelling is not just cheap but correct.
TEST(CommGraph, TheReverseLegIsARelabellingAndCostsNoCollective) {
  if (world_size() < 2) GTEST_SKIP();
  const int dim = 1;
  Fixture   f = make_fixture(Shape::FullyDisjoint, dim, true);

  CommGraphOptions opts;
  opts.mode = MatchMode::GlobalId;
  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD),
                                      side_of(f, true, dim),
                                      side_of(f, false, dim), opts);
  ASSERT_TRUE(built.ok()) << built.error().message;
  const CommGraph& g = built.value();

  // Rank-dependent call count: a collective here deadlocks.
  const int repeats = world_rank() == 0 ? 5 : 1;
  CommGraph back = g.transpose();
  for (int i = 1; i < repeats; ++i) back = g.transpose();

  EXPECT_EQ(back.local_pairs().size(), g.local_pairs().size());
  EXPECT_EQ(back.source_indices(), g.dest_indices());
  EXPECT_EQ(back.dest_indices(), g.source_indices());
  EXPECT_EQ(back.source_spec().src_ranks, g.dest_spec().dst_ranks);

  // And it delivers. The reverse leg moves values from what used to be the
  // destination mesh back to what used to be the source mesh.
  std::vector<double> rev_src(f.dst_k.size(), 0.0);
  for (std::size_t i = 0; i < f.dst_k.size(); ++i) {
    rev_src[i] = -value_of(gid_of(f.dst_k[i]));
  }
  std::vector<double> rev_dst(f.src_k.size(), -12345.0);
  deliver(back, rev_src, &rev_dst);
  for (std::size_t i = 0; i < f.src_k.size(); ++i) {
    ASSERT_DOUBLE_EQ(rev_dst[i], -value_of(gid_of(f.src_k[i])))
        << "rank " << world_rank() << " reverse leg, source entity " << i;
  }
}

// ===========================================================================
// 9-11: dimension independence
// ===========================================================================

TEST(CommGraph, NearestNeighborIsDimensionIndependentIn1D) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::FullyDisjoint, MatchMode::NearestNeighbor, 1);
}

TEST(CommGraph, NearestNeighborIsDimensionIndependentIn2D) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::FullyDisjoint, MatchMode::NearestNeighbor, 2);
}

TEST(CommGraph, NearestNeighborIsDimensionIndependentIn3D) {
  if (world_size() < 2) GTEST_SKIP();
  run_matrix_cell(Shape::FullyDisjoint, MatchMode::NearestNeighbor, 3);
}

// ===========================================================================
// 12: the tolerance boundary, pinned exactly
// ===========================================================================

// One source at the origin and three queries at exactly the tolerance, just
// inside it, and just outside it. The first two must match and the third must
// be refused -- which pins `<=` rather than `<`, and pins that the refusal is
// a refusal rather than a silent nearest-anything match.
TEST(CommGraph, TheNearestNeighborToleranceBoundaryIsInclusiveAndRefusesBeyond) {
  const double tol = 0.25;
  const int    rank = world_rank();

  // Only rank 0 holds the single source; every rank holds one query, so the
  // answer has to cross ranks.
  std::vector<std::size_t> sg;
  std::vector<Point>       sp;
  if (rank == 0) {
    sg.push_back(gid_of(0));
    sp.push_back(Point{0.0, 0.0, 0.0});
  }
  TestMesh smesh(1, sg, sp);

  struct Case {
    double      offset;
    const char* what;
    bool        expect_ok;
  };
  const Case cases[] = {
      {tol, "exactly at the tolerance", true},
      {std::nextafter(tol, 0.0), "just inside the tolerance", true},
      {std::nextafter(tol, 1.0), "just outside the tolerance", false},
  };

  for (const Case& c : cases) {
    TestMesh dmesh(1, {500000}, {Point{c.offset, 0.0, 0.0}});

    CommGraphSide s;
    s.mesh = &smesh;
    s.dim = 1;
    s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
    CommGraphSide d = s;
    d.mesh = &dmesh;

    CommGraphOptions opts;
    opts.mode = MatchMode::NearestNeighbor;
    opts.nn_tolerance = tol;

    const auto built =
        CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
    if (c.expect_ok) {
      ASSERT_TRUE(built.ok())
          << "a query " << c.what << " was refused: " << built.error().code
          << ": " << built.error().message;
      EXPECT_EQ(built.value().matched_count(), 1u);
    } else {
      ASSERT_FALSE(built.ok()) << "a query " << c.what << " was accepted";
      EXPECT_EQ(built.error().code, "E-COMMGRAPH-NOMATCH");
      // The refusal names the distance actually found, per spec §6 -- the
      // practice E-REMAP-COVERAGE already follows.
      EXPECT_NE(built.error().message.find("nearest one found"),
                std::string::npos)
          << "the refusal did not name the nearest distance found: "
          << built.error().message;
    }
    ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
  }
}

// The neighbourhood probe is load bearing, and nothing else here proves it.
//
// FOUND BY MUTATION, not by design. Every other NearestNeighbor fixture in
// this file happens to put a query and its match in the SAME cell, so
// restricting both the routing fan-out and the bucket-side search to the
// query's own cell -- deleting the `3^dim - 1` neighbourhood entirely -- left
// the whole suite green. The geometry below is built so that it cannot: the
// cell size is `2 * nn_tolerance = 0.5`, sources sit at `0.49 + i` and their
// matches at `0.51 + i`, so a match is 0.02 away and ALWAYS one cell over.
// The lone source at the origin is there to pin the bounding box, which is
// what fixes where the cell boundaries fall.
TEST(CommGraph, AMatchOneCellAwayIsStillFound) {
  const double tol = 0.25;  // cell size 0.5
  const int    rank = world_rank();
  const int    size = world_size();
  // Enough pairs that the Morton range spans several buckets. The two halves
  // of the neighbourhood probe fail differently, and the numbers are worth
  // recording because they say what this test does and does not pin:
  //
  //   * The BUCKET-SIDE search (comparing a query against the candidates in
  //     the cells around it, not just its own) is caught at every rank count
  //     tried -- 2, 3, 4 and 5 -- because the query and its match are always
  //     one cell apart by construction.
  //   * The ROUTING FAN-OUT (sending a query to the buckets owning those
  //     neighbouring cells) is only wrong where a neighbouring cell is in a
  //     DIFFERENT bucket, so it is caught at 3, 4 and 5 ranks but not at 2,
  //     where there is a single splitter and it is unlikely to land between a
  //     matched pair. The suite runs at 2, 3 and 4, so it is caught.
  //
  // At six pairs with uniform cell populations it was caught at neither 2 nor
  // 4: see the filler below.
  constexpr int kPairs = 32;

  std::vector<std::size_t> sg, dg;
  std::vector<Point>       sp, dp;
  if (rank == 0) {  // the bounding-box anchor, matched by nothing
    sg.push_back(gid_of(999));
    sp.push_back(Point{0.0, 0.0, 0.0});
  }
  for (int i = 0; i < kPairs; ++i) {
    if (i % size != rank) continue;
    sg.push_back(gid_of(static_cast<std::size_t>(i)));
    sp.push_back(Point{0.49 + static_cast<double>(i), 0.0, 0.0});
    // An extra, deliberately NON-winning source in every third source cell
    // (0.21 away from that cell's query, against the winner's 0.02). Its only
    // job is to make the per-cell populations irregular. With one point per
    // cell the population-balanced splitters land on even cell indices at
    // even rank counts, so a (source cell, query cell) pair never straddles a
    // bucket boundary and dropping the routing fan-out goes unnoticed at 2
    // and 4 ranks -- measured, before this filler was added.
    if (i % 3 == 0) {
      sg.push_back(gid_of(static_cast<std::size_t>(i) + 5000));
      sp.push_back(Point{0.30 + static_cast<double>(i), 0.0, 0.0});
    }
  }
  for (int i = 0; i < kPairs; ++i) {
    // Deliberately a different rank from the source of the same pair, so the
    // answer has to cross the communicator as well as the cell boundary.
    if ((i + 1) % size != rank) continue;
    dg.push_back(500000 + static_cast<std::size_t>(i));
    dp.push_back(Point{0.51 + static_cast<double>(i), 0.0, 0.0});
  }
  TestMesh smesh(1, sg, sp);
  TestMesh dmesh(1, dg, dp);

  CommGraphSide s;
  s.mesh = &smesh;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  CommGraphSide d = s;
  d.mesh = &dmesh;

  CommGraphOptions opts;
  opts.mode = MatchMode::NearestNeighbor;
  opts.nn_tolerance = tol;

  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(built.ok()) << built.error().code << ": "
                          << built.error().message;
  const CommGraph& g = built.value();
  ASSERT_EQ(g.matched_count(), dg.size());

  std::vector<double> src_vals;
  for (const std::size_t gid : sg) src_vals.push_back(value_of(gid));
  std::vector<double> dst_vals(dg.size(), -12345.0);
  deliver(g, src_vals, &dst_vals);
  for (std::size_t k = 0; k < dg.size(); ++k) {
    const auto i = static_cast<std::size_t>(dg[k] - 500000);
    ASSERT_DOUBLE_EQ(dst_vals[k], value_of(gid_of(i)))
        << "rank " << rank << ": the query at " << dp[k].x
        << " did not find the source at " << (0.49 + static_cast<double>(i))
        << " one cell away";
  }
}

// ===========================================================================
// 13: the space-filling-curve partition balances bucket population
// ===========================================================================

// A deliberately non-uniform mesh: 90% of the points are crammed into 10% of
// the domain, the shape a refined patch or a boundary layer has. A raw
// `hash(cell) % P` shard would scatter them at random, which balances counts
// but destroys locality; the failure mode it actually risks -- and the one
// flagged in design review before any code was written -- is that a
// spatially-incoherent bucket has to compare a query against candidates that
// are nowhere near it. The Morton partition with population-balanced
// splitters is asserted here to keep the worst bucket within 4x the mean.
TEST(CommGraph, TheSfcPartitionBalancesBucketPopulationOnARefinedPatch) {
  const int size = world_size();
  if (size < 2) GTEST_SKIP();
  const int rank = world_rank();

  constexpr std::size_t kTotal = 400;
  constexpr double      kTolerance = 0.004;

  // Point i sits at x = (i/kTotal)^3, so density rises steeply toward 0.
  const auto position = [](std::size_t i) {
    const double t = static_cast<double>(i) / static_cast<double>(kTotal);
    return t * t * t;
  };

  std::vector<std::size_t> sg, dg;
  std::vector<Point>       sp, dp;
  for (std::size_t i = 0; i < kTotal; ++i) {
    if (static_cast<int>(i) % size != rank) continue;
    sg.push_back(gid_of(i));
    sp.push_back(Point{position(i), 0.0, 0.0});
    dg.push_back(500000 + i);
    // Half a spacing off the source, well inside the tolerance near the dense
    // end and still inside it at the coarse end by construction of kTolerance.
    dp.push_back(Point{position(i) + 1e-9, 0.0, 0.0});
  }
  TestMesh smesh(1, sg, sp);
  TestMesh dmesh(1, dg, dp);

  CommGraphSide s;
  s.mesh = &smesh;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  CommGraphSide d = s;
  d.mesh = &dmesh;

  CommGraphOptions opts;
  opts.mode = MatchMode::NearestNeighbor;
  opts.nn_tolerance = kTolerance;

  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(built.ok()) << built.error().code << ": "
                          << built.error().message;

  const auto        mine = static_cast<long long>(built.value().stats().bucket_records);
  std::vector<long long> all(static_cast<std::size_t>(size), 0);
  ASSERT_EQ(MPI_Allgather(&mine, 1, MPI_LONG_LONG, all.data(), 1, MPI_LONG_LONG,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);

  long long total = 0;
  long long worst = 0;
  for (const long long v : all) {
    total += v;
    worst = std::max(worst, v);
  }
  const double mean = static_cast<double>(total) / static_cast<double>(size);
  ASSERT_GT(total, 0);

  if (rank == 0) {
    std::printf("[sfc-balance] buckets:");
    for (const long long v : all) std::printf(" %lld", v);
    std::printf("  mean=%.1f worst=%lld\n", mean, worst);
    std::fflush(stdout);
  }
  EXPECT_LE(static_cast<double>(worst), 4.0 * mean)
      << "the worst bucket holds " << worst << " records against a mean of "
      << mean << "; the splitters are not balancing population";
}

// 13, continued: the property a balance assertion alone does NOT pin.
//
// A CORRECTION TO THE SPEC'S OWN TEST 13, made because the test was checked
// against the design it exists to rule out. Balancing bucket POPULATION does
// not distinguish a Morton partition from the raw `hash(cell) % P` that was
// rejected in design review: a hash spreads cells uniformly, so whenever
// there are many cells of comparable weight it balances population perfectly
// well, and the balance test above passes against it. What a hash destroys is
// LOCALITY, and locality is directly observable: a query has to be probed
// against every bucket that could own a cell within `nn_tolerance` -- its own
// and the 3^dim - 1 around it. Morton-adjacent cells share a bucket, so the
// probe count per query stays near 1; a hash scatters those 27 cells over
// `min(27, P)` unrelated ranks. That ratio is asserted here.
TEST(CommGraph, TheSfcPartitionKeepsAQuerysNeighbourhoodInOneOrTwoBuckets) {
  const int size = world_size();
  if (size < 2) GTEST_SKIP();
  const int rank = world_rank();

  // A 3D lattice, where a hash has the most to scatter: 27 neighbouring cells
  // per query against a Morton curve that keeps them adjacent.
  constexpr int    kSide = 8;
  constexpr double kTolerance = 0.4;
  std::vector<std::size_t> sg, dg;
  std::vector<Point>       sp, dp;
  std::size_t              k = 0;
  for (int ix = 0; ix < kSide; ++ix) {
    for (int iy = 0; iy < kSide; ++iy) {
      for (int iz = 0; iz < kSide; ++iz, ++k) {
        if (static_cast<int>(k) % size != rank) continue;
        const Point p{static_cast<double>(ix), static_cast<double>(iy),
                      static_cast<double>(iz)};
        sg.push_back(gid_of(k));
        sp.push_back(p);
        dg.push_back(500000 + k);
        dp.push_back(Point{p.x + 0.05, p.y + 0.05, p.z + 0.05});
      }
    }
  }
  TestMesh smesh(3, sg, sp);
  TestMesh dmesh(3, dg, dp);

  CommGraphSide s;
  s.mesh = &smesh;
  s.dim = 3;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  CommGraphSide d = s;
  d.mesh = &dmesh;

  CommGraphOptions opts;
  opts.mode = MatchMode::NearestNeighbor;
  opts.nn_tolerance = kTolerance;

  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(built.ok()) << built.error().code << ": "
                          << built.error().message;

  long long mine[2] = {
      static_cast<long long>(built.value().stats().query_probes),
      static_cast<long long>(built.value().stats().query_points)};
  long long tot[2] = {0, 0};
  ASSERT_EQ(MPI_Allreduce(mine, tot, 2, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_GT(tot[1], 0);
  const double ratio =
      static_cast<double>(tot[0]) / static_cast<double>(tot[1]);
  if (rank == 0) {
    std::printf("[sfc-locality] %lld probes for %lld queries at %d ranks, "
                "%.2f buckets per query (a hash would approach min(27, %d))\n",
                tot[0], tot[1], size, ratio, size);
    std::fflush(stdout);
  }
  // MEASURED, on this fixture, by running both designs. Morton splitters:
  // 1.41 (2 ranks), 1.78 (3), 1.95 (4), 2.38 (8). The rejected raw
  // `mix64(cell) % P`, with nothing else changed: 2.00, 3.00, 3.97, 7.43 --
  // min(27, P), as predicted. The bound below separates them from three ranks
  // upward; at two ranks there are only two buckets and no metric of this
  // shape can tell the designs apart, which is why the suite runs at 3 and 4
  // as well.
  //
  // The same two runs are why the balance assertion above is documented as
  // insufficient rather than merely supplemented: under the raw hash the
  // refined-patch buckets measured 618 against a mean of 535 at two ranks and
  // 409 against 338 at four -- comfortably inside the 4x the balance test
  // allows. It passes against the design it was meant to rule out.
  EXPECT_LE(ratio, 2.6)
      << "a query is being probed against " << ratio
      << " buckets on average; the Morton partition is not keeping a "
         "neighbourhood together";
}

// ===========================================================================
// 14: the union communicator is exercised for real
// ===========================================================================

// The pattern from mpi_transport_test.cpp -- contiguous source blocks against
// a strided, reversed destination selection -- driven through two
// independently MPI_Comm_split-built communicators instead of one shared
// communicator with the rank sets handed in by the test. Nothing here tells
// CommGraph which ranks are involved; it works that out from the two
// communicators alone.
TEST(CommGraph, TwoIndependentlySplitCommunicatorsAreEnoughToBuildTheGraph) {
  if (world_size() < 2) GTEST_SKIP();
  const int dim = 1;
  Fixture   f = make_fixture(Shape::FullyDisjoint, dim, true);

  CommGraphOptions opts;
  opts.mode = MatchMode::GlobalId;
  const auto built = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD),
                                      side_of(f, true, dim),
                                      side_of(f, false, dim), opts);
  ASSERT_TRUE(built.ok()) << built.error().code << ": "
                          << built.error().message;
  const CommGraph& g = built.value();

  // The union is the whole world for this shape, and it is disjoint, so the
  // graph is expressible as one MpiTransport plan -- the R5 path that had no
  // working caller at all before this.
  EXPECT_TRUE(g.specs_are_disjoint());
  EXPECT_EQ(g.union_size(), world_size());
  EXPECT_EQ(g.source_spec().src_ranks, members_of(Shape::FullyDisjoint,
                                                  world_size(), true));
  EXPECT_EQ(g.source_spec().dst_ranks, members_of(Shape::FullyDisjoint,
                                                  world_size(), false));
  // Nothing is local: the two rank sets do not intersect.
  EXPECT_TRUE(g.local_pairs().empty());

  std::vector<double> src_vals(f.src_k.size(), 0.0);
  for (std::size_t i = 0; i < f.src_k.size(); ++i) {
    src_vals[i] = value_of(gid_of(f.src_k[i]));
  }
  std::vector<double> dst_vals(f.dst_k.size(), -12345.0);
  deliver(g, src_vals, &dst_vals);
  for (std::size_t i = 0; i < f.dst_k.size(); ++i) {
    ASSERT_DOUBLE_EQ(dst_vals[i], value_of(gid_of(f.dst_k[i])));
  }
}

// ===========================================================================
// Refusals and degenerate cases (spec §6)
// ===========================================================================

TEST(CommGraph, NearestNeighborWithoutAToleranceIsRefusedBeforeAnyCollective) {
  // Deliberately called on rank 0 only. A precondition check that reached a
  // collective would hang here instead of returning.
  if (world_rank() == 0) {
    TestMesh      mesh(1, {gid_of(0)}, {Point{0, 0, 0}});
    CommGraphSide s;
    s.mesh = &mesh;
    s.dim = 1;
    s.comm_handle = MPI_Comm_c2f(MPI_COMM_SELF);

    CommGraphOptions opts;
    opts.mode = MatchMode::NearestNeighbor;
    opts.nn_tolerance = 0.0;
    const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_SELF), s, s, opts);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-COMMGRAPH-OPTS");

    opts.nn_tolerance = -1.0;
    const auto r2 = CommGraph::build(MPI_Comm_c2f(MPI_COMM_SELF), s, s, opts);
    ASSERT_FALSE(r2.ok());
    EXPECT_EQ(r2.error().code, "E-COMMGRAPH-OPTS");

    // The mirror image: a tolerance set under GlobalId is refused rather than
    // silently ignored, because a caller who set it meant NearestNeighbor.
    opts.mode = MatchMode::GlobalId;
    opts.nn_tolerance = 0.5;
    const auto r3 = CommGraph::build(MPI_Comm_c2f(MPI_COMM_SELF), s, s, opts);
    ASSERT_FALSE(r3.ok());
    EXPECT_EQ(r3.error().code, "E-COMMGRAPH-OPTS");
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A wanted id that no source owns is refused, on EVERY rank, naming the id.
TEST(CommGraph, AWantedGlobalIdWithNoSourceIsRefusedOnEveryRank) {
  if (world_size() < 2) GTEST_SKIP();
  const int rank = world_rank();
  const int size = world_size();

  const bool in_src = rank < size / 2;
  const bool in_dst = !in_src;
  MPI_Comm   sc = split_for(in_src);
  MPI_Comm   dc = split_for(in_dst);

  std::vector<std::size_t> g;
  std::vector<Point>       p;
  if (in_src) {
    g.push_back(gid_of(static_cast<std::size_t>(rank)));
    p.push_back(Point{0, 0, 0});
  }
  if (in_dst) {
    // 999999 is owned by nobody.
    g.push_back(rank == size - 1 ? 999999 : gid_of(0));
    p.push_back(Point{0, 0, 0});
  }
  TestMesh mesh(1, g, p);

  CommGraphSide s;
  s.mesh = in_src ? &mesh : nullptr;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(sc);
  CommGraphSide d;
  d.mesh = in_dst ? &mesh : nullptr;
  d.dim = 1;
  d.comm_handle = MPI_Comm_c2f(dc);

  CommGraphOptions opts;
  const auto       r =
      CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  EXPECT_FALSE(r.ok()) << "rank " << rank << " accepted an unmatched id";
  if (!r.ok()) EXPECT_EQ(r.error().code, "E-COMMGRAPH-NOMATCH");

  if (sc != MPI_COMM_NULL) MPI_Comm_free(&sc);
  if (dc != MPI_COMM_NULL) MPI_Comm_free(&dc);
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Two source ranks claiming one id is ambiguous ownership, and is refused
// rather than resolved by picking a winner -- the two copies need not agree.
TEST(CommGraph, TwoSourceRanksOwningOneIdIsRefused) {
  if (world_size() < 2) GTEST_SKIP();

  // Every rank owns the SAME id on the source side.
  TestMesh smesh(1, {gid_of(0)}, {Point{0, 0, 0}});
  TestMesh dmesh(1, {gid_of(0)}, {Point{0, 0, 0}});

  CommGraphSide s;
  s.mesh = &smesh;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  CommGraphSide d = s;
  d.mesh = &dmesh;

  CommGraphOptions opts;
  const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-COMMGRAPH-DUPID");
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A ghost on the source side is NOT advertised as owned, so the same id being
// visible on two ranks is ordinary rather than an error. This is the positive
// control for the test above: it is the same id on two ranks, and it is
// accepted, because exactly one rank owns it.
TEST(CommGraph, AGhostedSourceEntityIsNotASecondOwner) {
  if (world_size() < 2) GTEST_SKIP();
  const int rank = world_rank();

  // Rank 0 owns id A; every other rank carries it as a ghost.
  TestMesh smesh(1, {gid_of(0)}, {Point{0, 0, 0}});
  if (rank != 0) smesh.set_ghost(0);
  TestMesh dmesh(1, {gid_of(0)}, {Point{0, 0, 0}});

  CommGraphSide s;
  s.mesh = &smesh;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  CommGraphSide d = s;
  d.mesh = &dmesh;

  CommGraphOptions opts;
  const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  // Every rank's single destination slot is matched, to rank 0's copy.
  EXPECT_EQ(r.value().matched_count(), 1u);
  if (rank == 0) {
    EXPECT_EQ(r.value().local_pairs().size(), 1u);
  } else {
    EXPECT_EQ(r.value().recv_plan().size(), 1u);
    EXPECT_EQ(r.value().recv_plan()[0].base_rank, 0);
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A rank in neither component still calls build(), gets an empty graph, and --
// this is the part that matters -- still holds the same two rank sets as
// everyone else, because it still has to enter MpiTransport's collectives.
TEST(CommGraph, ARankInNeitherComponentStillGetsTheRankSets) {
  const int size = world_size();
  if (size < 3) GTEST_SKIP();
  const int rank = world_rank();

  const bool in_src = rank == 0;
  const bool in_dst = rank == 1;
  MPI_Comm   sc = split_for(in_src);
  MPI_Comm   dc = split_for(in_dst);

  TestMesh mesh(1, {gid_of(0)}, {Point{0, 0, 0}});
  CommGraphSide s;
  s.mesh = in_src ? &mesh : nullptr;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(sc);
  CommGraphSide d;
  d.mesh = in_dst ? &mesh : nullptr;
  d.dim = 1;
  d.comm_handle = MPI_Comm_c2f(dc);

  CommGraphOptions opts;
  const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const CommGraph& g = r.value();

  EXPECT_EQ(g.participates(), in_src || in_dst);
  EXPECT_EQ(g.source_spec().src_ranks, std::vector<int>{0});
  EXPECT_EQ(g.source_spec().dst_ranks, std::vector<int>{1});
  EXPECT_TRUE(g.specs_are_disjoint());
  if (!in_src && !in_dst) {
    EXPECT_EQ(g.source_spec().elements, 0u);
    EXPECT_EQ(g.dest_spec().elements, 0u);
    EXPECT_TRUE(g.local_pairs().empty());
  }

  // And the whole world can now drive one MpiTransport plan over it, which is
  // the reason the rank sets have to be global.
  std::vector<double> src_vals(in_src ? 1 : 0, value_of(gid_of(0)));
  std::vector<double> dst_vals(in_dst ? 1 : 0, -12345.0);
  deliver(g, src_vals, &dst_vals);
  if (in_dst) EXPECT_DOUBLE_EQ(dst_vals[0], value_of(gid_of(0)));

  if (sc != MPI_COMM_NULL) MPI_Comm_free(&sc);
  if (dc != MPI_COMM_NULL) MPI_Comm_free(&dc);
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A rank contributing zero entities on either side is ordinary in parallel and
// still enters every collective -- the same rule MpiTransport already states.
TEST(CommGraph, ARankWithZeroLocalEntitiesIsOrdinary) {
  if (world_size() < 2) GTEST_SKIP();
  const int rank = world_rank();
  const int size = world_size();

  // Only the last rank holds anything on either side.
  const bool loaded = rank == size - 1;
  std::vector<std::size_t> gids;
  std::vector<Point>       pts;
  if (loaded) {
    for (std::size_t k = 0; k < 4; ++k) {
      gids.push_back(gid_of(k));
      pts.push_back(source_point(k, 1));
    }
  }
  TestMesh smesh(1, gids, pts);
  TestMesh dmesh(1, gids, pts);

  const bool in_src = rank % 2 == 0 || loaded;
  const bool in_dst = rank % 2 == 1 || loaded;
  MPI_Comm   sc = split_for(in_src);
  MPI_Comm   dc = split_for(in_dst);

  CommGraphSide s;
  s.mesh = in_src ? &smesh : nullptr;
  s.dim = 1;
  s.comm_handle = MPI_Comm_c2f(sc);
  CommGraphSide d;
  d.mesh = in_dst ? &dmesh : nullptr;
  d.dim = 1;
  d.comm_handle = MPI_Comm_c2f(dc);

  CommGraphOptions opts;
  const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_WORLD), s, d, opts);
  ASSERT_TRUE(r.ok()) << "rank " << rank << ": " << r.error().code << ": "
                      << r.error().message;
  // The only destination entities anywhere are the last rank's four, and it
  // owns their sources too, so all four are local copies there.
  EXPECT_EQ(r.value().matched_count(), loaded ? 4u : 0u);

  if (sc != MPI_COMM_NULL) MPI_Comm_free(&sc);
  if (dc != MPI_COMM_NULL) MPI_Comm_free(&dc);
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A side with a communicator but no mesh (or the reverse) is a caller bug and
// is caught before any collective.
TEST(CommGraph, ASideWithACommunicatorButNoMeshIsRefused) {
  if (world_rank() == 0) {
    TestMesh      mesh(1, {gid_of(0)}, {Point{0, 0, 0}});
    CommGraphSide s;
    s.mesh = nullptr;
    s.dim = 1;
    s.comm_handle = MPI_Comm_c2f(MPI_COMM_SELF);
    CommGraphSide d;
    d.mesh = &mesh;
    d.dim = 1;
    d.comm_handle = MPI_Comm_c2f(MPI_COMM_SELF);

    CommGraphOptions opts;
    const auto r = CommGraph::build(MPI_Comm_c2f(MPI_COMM_SELF), s, d, opts);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-COMMGRAPH-OPTS");

    // And the reverse: a mesh with no communicator.
    CommGraphSide s2 = d;
    s2.comm_handle = MPI_Comm_c2f(MPI_COMM_NULL);
    const auto r2 = CommGraph::build(MPI_Comm_c2f(MPI_COMM_SELF), s2, d, opts);
    ASSERT_FALSE(r2.ok());
    EXPECT_EQ(r2.error().code, "E-COMMGRAPH-OPTS");
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
namespace {

/// Turns the first failed assertion on any rank into an MPI_Abort. Without it
/// a failure unwinds one rank out of its TEST body while the others block in
/// the next collective, and the CTest entry hangs until its timeout instead of
/// reporting the failure.
class AbortOnFailure : public ::testing::EmptyTestEventListener {
  void OnTestPartResult(const ::testing::TestPartResult& r) override {
    if (!r.failed()) return;
    std::fprintf(stderr, "[rank %d] FAILURE at %s:%d\n%s\n", world_rank(),
                 r.file_name() != nullptr ? r.file_name() : "<unknown>",
                 r.line_number(), r.message());
    std::fflush(stderr);
    std::fflush(stdout);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
};

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::UnitTest::GetInstance()->listeners().Append(new AbortOnFailure);
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

#else
// Built without MPI: nothing to exercise. This binary is only registered as a
// CTest entry when SOMA_ENABLE_MPI is ON, so it never actually runs here.
int main() { return 0; }
#endif
