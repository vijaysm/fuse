// Task 5.4: the MPI transport, run under a real mpiexec with more than one
// rank.
//
// WHY THE PATTERNS BELOW ARE NOT THE OBVIOUS ONES. Sources hold CONTIGUOUS
// blocks of global ids; destinations hold a STRIDED selection (rank j of m
// takes every id with g % m == j). So no destination slot ever lines up
// positionally with a source slot, every destination receives from every
// source, and an implementation that copied by position, or that only
// exchanged with its opposite number, produces wrong numbers rather than
// passing by luck.
//
// WHY THIS IS NOT A soma_add_test(). It calls MPI_Init, it is launched by
// mpiexec, and -- the part that matters -- a GoogleTest assertion failure
// unwinds one rank out of a TEST body while its peers walk on into the next
// collective, which is a hang, not a failure. The listener installed in main()
// turns the first failed assertion on any rank into MPI_Abort, so a broken
// build fails the CTest entry in seconds instead of sitting on its timeout.

#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MPI

#include <mpi.h>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>
#include <soma/transport/transport.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

using soma::MpiTransport;
using soma::TransportSpec;

namespace {
/// Cells per side of the Runner-driven harness far below. One value, because
/// producer and consumer must agree for SerialMeshBackend's identity remap.
constexpr std::size_t kHookCells = 8;
}  // namespace

// ---------------------------------------------------------------------------
// Allocation counting, the same mechanism test/unit/runner_test.cpp uses to
// prove the hot loop allocates nothing. Replacing the global operator new is
// the only way to measure it rather than assert it by inspection. MPICH is C
// and allocates through malloc, so what is counted here is SOMA's own
// behaviour, which is exactly the thing under test.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_allocations = 0;
bool        g_counting = false;

class AllocationCounter {
 public:
  AllocationCounter() {
    g_allocations = 0;
    g_counting = true;
  }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;

  [[nodiscard]] std::size_t count() const { return g_allocations; }
};
}  // namespace

void* operator new(std::size_t n) {
  if (g_counting) ++g_allocations;
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  if (g_counting) ++g_allocations;
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}

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

/// The value carried by global id g. Distinct for every id, and not a small
/// integer, so a stale zero or a shifted index is visible.
double value_of(std::size_t g) { return static_cast<double>(g) * 10.0 + 1.0; }

/// Position of `rank` inside `set`, or -1.
int position_in(const std::vector<int>& set, int rank) {
  for (std::size_t i = 0; i < set.size(); ++i) {
    if (set[i] == rank) return static_cast<int>(i);
  }
  return -1;
}

/// The redistribution pattern under test. `global` ids, split into contiguous
/// blocks across `src` and strided across `dst`.
struct Pattern {
  std::vector<int> src;
  std::vector<int> dst;
  std::size_t      global = 12;
  int              components = 1;
};

std::vector<std::size_t> ids_for(const Pattern& p, int rank) {
  std::vector<std::size_t> ids;
  const int si = position_in(p.src, rank);
  if (si >= 0) {
    const std::size_t n = p.src.size();
    const std::size_t per = p.global / n;
    const std::size_t base = static_cast<std::size_t>(si) * per;
    ids.resize(per);
    std::iota(ids.begin(), ids.end(), base);
    return ids;
  }
  const int di = position_in(p.dst, rank);
  if (di >= 0) {
    const std::size_t m = p.dst.size();
    for (std::size_t g = static_cast<std::size_t>(di); g < p.global; g += m) {
      ids.push_back(g);
    }
    // DESCENDING, and this matters. With an ascending destination list and
    // ascending source blocks, "concatenate the blocks received from source 0,
    // then source 1, ..." happens to reproduce the destination's own order, so
    // the unpack permutation is the identity and an implementation that
    // ignored it entirely would still pass. Reversing the list makes the
    // permutation non-trivial in both directions: verified by mutation --
    // unpacking to slot k instead of to recv_index[k] passes with an ascending
    // list and fails with this one.
    std::reverse(ids.begin(), ids.end());
  }
  return ids;
}

TransportSpec spec_for(const Pattern& p, int rank) {
  TransportSpec s;
  s.src_rank_set = 0;
  s.dst_rank_set = 1;
  s.src_ranks = p.src;
  s.dst_ranks = p.dst;
  s.components = p.components;
  s.global_ids = ids_for(p, rank);
  s.elements = s.global_ids.size();
  return s;
}

/// Build the plan, move once, and check what arrived. Prints a per-rank line
/// so the CTest log actually shows the exchange happening.
void exchange_and_check(const Pattern& p, const char* label) {
  const int rank = world_rank();
  const TransportSpec spec = spec_for(p, rank);
  const auto nc = static_cast<std::size_t>(p.components);
  const std::size_t total = spec.elements * nc;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto plan = t.build_plan(spec);
  ASSERT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;

  std::vector<double> send(total, 0.0);
  std::vector<double> recv(total, -12345.0);
  const bool is_source = position_in(p.src, rank) >= 0;
  const bool is_dest = position_in(p.dst, rank) >= 0;
  if (is_source) {
    for (std::size_t i = 0; i < spec.elements; ++i) {
      for (std::size_t c = 0; c < nc; ++c) {
        send[i * nc + c] = value_of(spec.global_ids[i]) + static_cast<double>(c);
      }
    }
  }

  ASSERT_TRUE(t.prepare(spec).ok());
  const auto mv = t.move(total == 0 ? nullptr : send.data(),
                         total == 0 ? nullptr : recv.data(), total);
  ASSERT_TRUE(mv.ok()) << mv.error().code << ": " << mv.error().message;

  std::printf("[%s rank %d/%d] role=%s ids=%zu sent=%zu recvd=%zu\n", label,
              rank, world_size(),
              is_source ? "source" : (is_dest ? "dest" : "idle"),
              spec.elements, t.send_volume(), t.recv_volume());
  std::fflush(stdout);

  if (is_dest) {
    for (std::size_t i = 0; i < spec.elements; ++i) {
      for (std::size_t c = 0; c < nc; ++c) {
        ASSERT_DOUBLE_EQ(recv[i * nc + c],
                         value_of(spec.global_ids[i]) + static_cast<double>(c))
            << label << " rank " << rank << " element " << i << " (global id "
            << spec.global_ids[i] << ") component " << c;
      }
    }
  } else {
    // Not a destination: move() must not have written anything.
    for (std::size_t k = 0; k < total; ++k) {
      ASSERT_DOUBLE_EQ(recv[k], -12345.0)
          << label << " rank " << rank << " had its output buffer written at "
          << k << " although it is not a destination";
    }
  }

  // Every rank got here, which is the deadlock check for the idle-rank and
  // zero-element cases.
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

std::vector<int> range_of(int lo, int hi) {  // [lo, hi)
  std::vector<int> v;
  for (int i = lo; i < hi; ++i) v.push_back(i);
  return v;
}

/// The first half of the ranks produce and the rest consume. Deliberately not
/// gated on an even rank count: at -n 3 this is 1 -> 2 and at -n 5 it is
/// 2 -> 3, so the ordinary case is asymmetric on most rank counts rather than
/// being skipped there. False when `global` will not divide evenly on both
/// sides, which is the only thing that actually rules a rank count out.
bool halves(int size, Pattern& p) {
  if (size < 2) return false;
  const auto n = static_cast<std::size_t>(size / 2);
  const auto m = static_cast<std::size_t>(size) - n;
  if (n == 0 || p.global % n != 0 || p.global % m != 0) return false;
  p.src = range_of(0, size / 2);
  p.dst = range_of(size / 2, size);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

TEST(MpiTransport, IsNotLocalAndNamed) {
  const MpiTransport t;
  EXPECT_FALSE(t.is_local());
  EXPECT_EQ(t.name(), "mpi");
}

TEST(MpiTransport, RegisteredInTheTransportRegistry) {
  const auto t = soma::TransportRegistry::instance().create("mpi");
  ASSERT_TRUE(t.ok()) << t.error().message;
  EXPECT_EQ(t.value()->name(), "mpi");
  EXPECT_FALSE(t.value()->is_local());
}

// ---------------------------------------------------------------------------
// Redistribution
// ---------------------------------------------------------------------------

// The disjoint-rank-set case spec §4.1 calls the normal one: the first half of
// the ranks produce, the second half consume.
TEST(MpiTransport, RedistributesAcrossDisjointRankSets) {
  Pattern p;
  if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";
  exchange_and_check(p, "halves");
}

// n != m, the asymmetric case. Three producers, one consumer.
TEST(MpiTransport, RedistributesThreeToOne) {
  if (world_size() != 4) GTEST_SKIP() << "needs exactly 4 ranks";
  Pattern p;
  p.src = {0, 1, 2};
  p.dst = {3};
  exchange_and_check(p, "3->1");
}

// The other direction: one producer, three consumers.
TEST(MpiTransport, RedistributesOneToThree) {
  if (world_size() != 4) GTEST_SKIP() << "needs exactly 4 ranks";
  Pattern p;
  p.src = {0};
  p.dst = {1, 2, 3};
  exchange_and_check(p, "1->3");
}

TEST(MpiTransport, RedistributesVectorFields) {
  Pattern p;
  p.components = 3;
  if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";
  exchange_and_check(p, "vector");
}

// A rank in NEITHER set is ordinary: it still enters MPI_Alltoallv, with zero
// counts.
//
// HONEST LIMIT OF THIS TEST. The brief says a four-rank run with two idle
// ranks deadlocks if they skip the collective. Measured, that is not true of
// MPICH 5.0: an early `return` for ranks whose counts are all zero was
// injected into move() and this test still passed in 3 ms, because MPICH's
// MPI_Alltoallv posts point-to-point traffic only to peers with non-zero
// counts and never synchronises with the rest. So what this test really
// proves is that an idle rank succeeds and the job completes; that the
// implementation is also standard-conforming -- every rank reaches the
// collective, unconditionally -- is a property of the code, which has no
// early return, rather than something MPICH can be made to reveal.
TEST(MpiTransport, IdleRanksParticipateInsteadOfDeadlocking) {
  const int size = world_size();
  if (size < 3) GTEST_SKIP() << "needs a rank outside both sets";
  Pattern p;
  p.src = {0};
  p.dst = {1};
  exchange_and_check(p, "idle");
}

// A rank that owns zero elements of a region is ordinary in parallel, not an
// edge case: it is a member of a rank set, it has nothing to contribute, and
// `std::vector<double>{}.data()` is null. Both must work.
TEST(MpiTransport, RankOwningZeroElementsIsOrdinary) {
  const int size = world_size();
  if (size < 4) GTEST_SKIP() << "needs two sources and two destinations";
  const int rank = world_rank();

  // src = {0, 1} with rank 1 holding nothing; dst = the rest with the LAST
  // one wanting nothing.
  const std::vector<int> src{0, 1};
  const std::vector<int> dst = range_of(2, size);
  constexpr std::size_t kGlobal = 12;

  TransportSpec spec;
  spec.src_ranks = src;
  spec.dst_ranks = dst;
  spec.components = 1;
  const int di = position_in(dst, rank);
  if (rank == 0) {
    spec.global_ids.resize(kGlobal);
    std::iota(spec.global_ids.begin(), spec.global_ids.end(), std::size_t{0});
  } else if (di >= 0 && di + 1 < static_cast<int>(dst.size())) {
    const std::size_t m = dst.size() - 1;  // the last one takes nothing
    for (std::size_t g = static_cast<std::size_t>(di); g < kGlobal; g += m) {
      spec.global_ids.push_back(g);
    }
    std::reverse(spec.global_ids.begin(), spec.global_ids.end());
  }
  spec.elements = spec.global_ids.size();

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto plan = t.build_plan(spec);
  ASSERT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;

  std::vector<double> send(spec.elements, 0.0);
  std::vector<double> recv(spec.elements, -1.0);
  for (std::size_t i = 0; i < spec.elements; ++i) {
    send[i] = value_of(spec.global_ids[i]);
  }
  // Deliberately the raw vector pointers. For an empty vector libc++ and
  // libstdc++ both hand back a null pointer, which is the whole point.
  if (spec.elements == 0) {
    EXPECT_EQ(send.data(), nullptr)
        << "this platform's empty vector::data() is non-null, so this test no "
           "longer exercises the null case";
  }
  std::printf("[zero rank %d/%d] elements=%zu send.data()=%p\n", rank,
              world_size(), spec.elements,
              static_cast<const void*>(send.data()));
  std::fflush(stdout);

  ASSERT_TRUE(t.prepare(spec).ok());
  const auto mv = t.move(send.data(), recv.data(), spec.elements);
  ASSERT_TRUE(mv.ok()) << mv.error().code << ": " << mv.error().message;

  if (di >= 0) {
    for (std::size_t i = 0; i < spec.elements; ++i) {
      ASSERT_DOUBLE_EQ(recv[i], value_of(spec.global_ids[i]))
          << "rank " << rank << " element " << i;
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Two channels alive at once. Each MpiTransport duplicates the communicator,
// so their collectives run in private contexts and cannot be matched with each
// other however they interleave.
TEST(MpiTransport, ConcurrentChannelsDoNotCollide) {
  const int rank = world_rank();

  Pattern forward;
  if (!halves(world_size(), forward)) {
    GTEST_SKIP() << "rank count does not split 12";
  }
  Pattern backward;              // the opposite direction, same ranks
  backward.src = forward.dst;
  backward.dst = forward.src;

  const TransportSpec fs = spec_for(forward, rank);
  const TransportSpec bs = spec_for(backward, rank);

  MpiTransport a;
  MpiTransport b;
  ASSERT_TRUE(a.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(b.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(a.build_plan(fs).ok());
  ASSERT_TRUE(b.build_plan(bs).ok());

  std::vector<double> a_send(fs.elements), a_recv(fs.elements, -1.0);
  std::vector<double> b_send(bs.elements), b_recv(bs.elements, -1.0);
  for (std::size_t i = 0; i < fs.elements; ++i) {
    a_send[i] = value_of(fs.global_ids[i]);
  }
  for (std::size_t i = 0; i < bs.elements; ++i) {
    b_send[i] = value_of(bs.global_ids[i]) + 0.5;
  }

  for (int step = 0; step < 3; ++step) {
    ASSERT_TRUE(a.prepare(fs).ok());
    ASSERT_TRUE(a.move(a_send.data(), a_recv.data(), fs.elements).ok());
    ASSERT_TRUE(b.prepare(bs).ok());
    ASSERT_TRUE(b.move(b_send.data(), b_recv.data(), bs.elements).ok());
  }

  if (position_in(forward.dst, rank) >= 0) {
    for (std::size_t i = 0; i < fs.elements; ++i) {
      ASSERT_DOUBLE_EQ(a_recv[i], value_of(fs.global_ids[i]));
    }
  }
  if (position_in(backward.dst, rank) >= 0) {
    for (std::size_t i = 0; i < bs.elements; ++i) {
      ASSERT_DOUBLE_EQ(b_recv[i], value_of(bs.global_ids[i]) + 0.5);
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ---------------------------------------------------------------------------
// The hot loop. Task 4.6's constraint, measured.
// ---------------------------------------------------------------------------

TEST(MpiTransport, PrepareAndMoveAllocateNothingPerStep) {
  const int size = world_size();
  const int rank = world_rank();
  Pattern p;
  if (!halves(size, p)) GTEST_SKIP() << "rank count does not split 12";
  const TransportSpec spec = spec_for(p, rank);

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.build_plan(spec).ok());

  std::vector<double> send(spec.elements, 1.0);
  std::vector<double> recv(spec.elements, 0.0);

  // One warm-up step outside the counter, exactly as the runner's
  // AllocatesNothingWhileRunning does, so something allocated lazily on first
  // use is still caught by the counted steps.
  ASSERT_TRUE(t.prepare(spec).ok());
  ASSERT_TRUE(t.move(send.data(), recv.data(), spec.elements).ok());

  std::size_t allocations = 0;
  bool        ok = true;
  {
    const AllocationCounter counter;
    for (int step = 0; step < 8; ++step) {
      // Both calls unconditionally, never short-circuited: skipping a move()
      // on one rank because an earlier step failed would strand every peer in
      // MPI_Alltoallv.
      const bool prepared = t.prepare(spec).ok();
      const bool moved = t.move(send.data(), recv.data(), spec.elements).ok();
      ok = ok && prepared && moved;
    }
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  std::printf("[alloc rank %d/%d] %zu allocation(s) across 8 prepare+move\n",
              rank, size, allocations);
  std::fflush(stdout);
  EXPECT_EQ(allocations, 0u)
      << "prepare()+move() allocated " << allocations
      << " times over 8 steps; the plan must be built in build_plan(), not in "
         "the hot loop";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, AllocationCounterActuallyCounts) {
  // A check that cannot fail proves nothing.
  std::size_t allocations = 0;
  {
    const AllocationCounter counter;
    std::vector<double> v(4096, 1.0);
    allocations = counter.count();
    EXPECT_DOUBLE_EQ(v[0], 1.0);
  }
  EXPECT_GE(allocations, 1u);
}

// ---------------------------------------------------------------------------
// Error paths. Each one must be an identical error return on EVERY rank: the
// MPI_Barrier at the end of each test is what proves nobody was left in a
// collective.
// ---------------------------------------------------------------------------

TEST(MpiTransport, BuildPlanRejectsEmptyRankSets) {
  TransportSpec spec;
  spec.src_ranks = {};
  spec.dst_ranks = {world_size() - 1};
  spec.elements = 0;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RANKS") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, BuildPlanRejectsOverlappingRankSets) {
  if (world_size() < 2) GTEST_SKIP();
  TransportSpec spec;
  spec.src_ranks = {0, 1};
  spec.dst_ranks = {1};
  spec.elements = 0;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RANKS") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, BuildPlanRejectsARankOutsideTheCommunicator) {
  TransportSpec spec;
  spec.src_ranks = {0};
  spec.dst_ranks = {world_size()};  // one past the end
  spec.elements = 0;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RANKS") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// A matching error must be an error RETURN on every rank, never a hang: the
// destination wants an id no source holds. Detected on every rank at once
// because the matching runs on data every rank has.
TEST(MpiTransport, UnmatchedGlobalIdIsAnErrorOnEveryRank) {
  const int size = world_size();
  if (size < 2) GTEST_SKIP();
  const int rank = world_rank();
  const std::vector<int> src = range_of(0, size / 2 > 0 ? size / 2 : 1);
  const std::vector<int> dst = range_of(size / 2 > 0 ? size / 2 : 1, size);

  TransportSpec spec;
  spec.src_ranks = src;
  spec.dst_ranks = dst;
  if (position_in(src, rank) >= 0) {
    spec.global_ids = {static_cast<std::size_t>(rank)};
  } else if (position_in(dst, rank) >= 0) {
    spec.global_ids = {9999};  // held by nobody
  }
  spec.elements = spec.global_ids.size();

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok()) << "rank " << rank << " accepted an unmatched id";
  EXPECT_EQ(r.error().code, "E-TRANSPORT-MATCH") << r.error().message;
  std::printf("[match rank %d/%d] %s: %s\n", rank, size, r.error().code.c_str(),
              r.error().message.c_str());
  std::fflush(stdout);
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Two source ranks claiming the same id is a ghost that was not filtered out.
// Sending either copy would be a silent choice between two values.
TEST(MpiTransport, DuplicatedSourceIdIsRejected) {
  const int size = world_size();
  if (size < 3) GTEST_SKIP() << "needs two sources and a destination";
  const int rank = world_rank();
  const std::vector<int> src{0, 1};
  const std::vector<int> dst = range_of(2, size);

  TransportSpec spec;
  spec.src_ranks = src;
  spec.dst_ranks = dst;
  if (rank == 0 || rank == 1) {
    spec.global_ids = {7};  // both claim id 7
  } else if (rank == 2) {
    spec.global_ids = {7};
  }
  spec.elements = spec.global_ids.size();

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-MATCH") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, BuildPlanRejectsAnIdListThatDisagreesWithElements) {
  if (world_size() < 2) GTEST_SKIP();
  TransportSpec spec;
  spec.src_ranks = {0};
  spec.dst_ranks = {1};
  spec.global_ids = {0, 1, 2};
  spec.elements = 4;  // not 3

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Standing rule 5: sizes are std::size_t and wrap silently, but MPI counts are
// int. A buffer longer than INT_MAX doubles must be refused, not truncated.
// Reached with a huge component count so the test costs no memory.
TEST(MpiTransport, BuildPlanRejectsCountsBeyondIntMax) {
  const int size = world_size();
  if (size < 2) GTEST_SKIP();
  const int rank = world_rank();
  const std::vector<int> src = range_of(0, size / 2);
  const std::vector<int> dst = range_of(size / 2, size);

  TransportSpec spec;
  spec.src_ranks = src;
  spec.dst_ranks = dst;
  // 3 * (INT_MAX/2) overflows an int count; 2 * (INT_MAX/2) would not, so the
  // third element is what makes this a real test of the guard.
  spec.components = std::numeric_limits<int>::max() / 2;
  spec.global_ids = {static_cast<std::size_t>(rank), 100, 200};
  spec.elements = 3;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, PrepareAndMoveRefuseAnUnplannedTransport) {
  MpiTransport t;
  TransportSpec spec;
  spec.elements = 4;
  const auto pr = t.prepare(spec);
  ASSERT_FALSE(pr.ok());
  EXPECT_EQ(pr.error().code, "E-TRANSPORT-UNPLANNED");
  std::vector<double> v(4, 0.0);
  const auto mr = t.move(v.data(), v.data(), 4);
  ASSERT_FALSE(mr.ok());
  EXPECT_EQ(mr.error().code, "E-TRANSPORT-UNPLANNED");
}

TEST(MpiTransport, PrepareRefusesADifferentSpecThanWasPlanned) {
  const int rank = world_rank();
  Pattern p;
  if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";
  const TransportSpec spec = spec_for(p, rank);

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.build_plan(spec).ok());
  ASSERT_TRUE(t.prepare(spec).ok());

  TransportSpec other = spec;
  other.components = spec.components + 1;
  const auto r = t.prepare(other);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RESPEC");
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, MoveChecksItsLengthAndItsPointers) {
  const int rank = world_rank();
  Pattern p;
  if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";
  const TransportSpec spec = spec_for(p, rank);
  ASSERT_GT(spec.elements, 0u);

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.build_plan(spec).ok());
  ASSERT_TRUE(t.prepare(spec).ok());

  std::vector<double> v(spec.elements, 0.0);
  const auto wrong = t.move(v.data(), v.data(), spec.elements + 1);
  ASSERT_FALSE(wrong.ok());
  EXPECT_EQ(wrong.error().code, "E-TRANSPORT-SIZE");

  const auto null_src = t.move(nullptr, v.data(), spec.elements);
  ASSERT_FALSE(null_src.ok());
  EXPECT_EQ(null_src.error().code, "E-TRANSPORT-NULL");
  const auto null_dst = t.move(v.data(), nullptr, spec.elements);
  ASSERT_FALSE(null_dst.ok());
  EXPECT_EQ(null_dst.error().code, "E-TRANSPORT-NULL");

  // None of the three reached MPI_Alltoallv, so nobody is stranded.
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransport, SetCommunicatorRejectsCommNull) {
  MpiTransport t;
  const auto r = t.set_communicator(MPI_Comm_c2f(MPI_COMM_NULL));
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-COMM") << r.error().message;
}

TEST(MpiTransport, BuildPlanRefusesWithoutACommunicator) {
  MpiTransport t;
  TransportSpec spec;
  spec.src_ranks = {0};
  spec.dst_ranks = {1};
  const auto r = t.build_plan(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-COMM") << r.error().message;
}

// ---------------------------------------------------------------------------
// MpiTransport THROUGH RunnerConfig::transport_setup (Task 8.5's hook).
//
// Everything above drives MpiTransport directly. Task 8.5 added the hook that
// is supposed to carry `set_communicator` + `build_plan` into a transport the
// Runner constructed, and wired LowFive through it; that MpiTransport works the
// same way was argued by symmetry and never run. These three tests run it.
//
// WHAT THEY FOUND, and what they now pin: THE SYMMETRY DOES NOT HOLD. The
// Runner fills only four fields of the TransportSpec it hands prepare() --
// src_rank_set, dst_rank_set, elements, components (src/runtime/runner.cpp, the
// OpKind::Transport case of resolve_instructions). `src_ranks`, `dst_ranks` and
// `global_ids` are left empty; nothing in src/ or examples/ populates them.
// LowFive survives that because its build_plan() does not care what is in those
// three vectors. MpiTransport's build_plan() does: it REQUIRES a non-empty
// source and destination rank set, and requires global_ids.size() == elements.
// So the two specs can never be the same object, and prepare()'s equality test
// -- which compares all three sizes -- must fail whichever way the hook is
// used. Both ways are exercised below; neither moves a byte.
//
// This is a genuine end-to-end exercise of the path, not a stand-in: a real
// Runner over a real graph, the "mpi" factory out of the process-wide
// TransportRegistry, the hook, and a real MPI_COMM_WORLD. What it establishes
// is that the path is broken, and exactly where.
// ---------------------------------------------------------------------------
namespace {

/// Producer writes a constant; consumer holds what arrives. Deliberately
/// minimal -- the subject is the transport hook, not the components.
class HookProducer final : public soma::Component {
 public:
  [[nodiscard]] std::string name() const override { return "producer"; }
  soma::Result<void> register_fields(soma::FieldRegistry& reg,
                                     const soma::CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }
  soma::Result<void> advance(const soma::StepContext&) override {
    for (double& v : data_) v = 2.0;
    return soma::Result<void>::success();
  }

 private:
  std::vector<double> data_ = std::vector<double>(kHookCells, 0.0);
};

class HookConsumer final : public soma::Component {
 public:
  [[nodiscard]] std::string name() const override { return "consumer"; }
  soma::Result<void> register_fields(soma::FieldRegistry& reg,
                                     const soma::CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }
  soma::Result<void> advance(const soma::StepContext&) override {
    return soma::Result<void>::success();
  }
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_ = std::vector<double>(kHookCells, 0.0);
};

/// producer (rank 0) -> consumer (rank 1). The rank ranges are DISJOINT, which
/// is what makes lowering emit an OpKind::Transport at all -- see the
/// `ranks_differ` computation in src/ir/rules_channel.cpp. Equal cell counts on
/// both sides so the Remap that follows is the identity SerialMeshBackend can
/// do; anything else would fail for an unrelated reason.
struct HookHarness {
  soma::CouplingGraph     graph;
  soma::SerialMeshBackend backend;
  soma::OperatorDAG       dag;
  soma::ExecutionPlan     plan;
  soma::ComponentSet      components;
  soma::FieldRegistry     registry;
};

std::unique_ptr<HookHarness> make_hook_harness() {
  auto  h = std::make_unique<HookHarness>();
  auto& g = h->graph;
  namespace u = soma::units;

  const std::string src = "generated:line:" + std::to_string(kHookCells);
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1", .source = src});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2", .source = src});
  const auto r1 =
      g.add_region(soma::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(soma::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(soma::ComponentDecl{
      .name = "producer", .ranks = {.first = 0, .count = 1},
      .interval = soma::Rational{1}});
  const auto c = g.add_component(soma::ComponentDecl{
      .name = "consumer", .ranks = {.first = 1, .count = 1},
      .interval = soma::Rational{1}});
  const auto sf = g.add_field(
      p, soma::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                         .intent = soma::Intent::Provides});
  const auto tf = g.add_field(
      c, soma::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                         .intent = soma::Intent::Requires});
  g.add_channel("link", sf, tf);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = soma::lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched = soma::schedule(g, h->dag, clocks,
                              soma::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  h->components.add(std::make_unique<HookProducer>());
  h->components.add(std::make_unique<HookConsumer>());
  for (auto& comp : h->components) {
    EXPECT_TRUE(comp.register_fields(h->registry, h->graph).ok());
  }
  return h;
}

/// The Transport instruction the Runner emits for that graph carries exactly
/// one of these. Reconstructed here from the same four fields
/// resolve_instructions() copies, so the tests can hand build_plan() the very
/// spec prepare() will later be given.
TransportSpec runner_side_spec() {
  TransportSpec s;
  s.src_rank_set = 0;  // producer's component id
  s.dst_rank_set = 1;  // consumer's component id
  s.elements = kHookCells;
  s.components = 1;
  return s;  // src_ranks, dst_ranks and global_ids stay empty -- see above
}

/// A pattern MpiTransport WILL plan: rank 0 sources, rank 1 sinks, the rest
/// idle. Valid at -n 2 and -n 4 alike.
TransportSpec plannable_spec(int rank) {
  TransportSpec s = runner_side_spec();
  s.src_ranks = {0};
  s.dst_ranks = {1};
  if (rank == 0 || rank == 1) {
    s.global_ids.resize(kHookCells);
    std::iota(s.global_ids.begin(), s.global_ids.end(), std::size_t{0});
  }
  s.elements = s.global_ids.size();  // 0 on a rank in neither set
  return s;
}

}  // namespace

// The half that DOES work, and the reason the gap below is a plumbing defect
// rather than a missing feature: the hook really does hand back the very
// MpiTransport the Runner will use, a communicator really does reach it, and a
// refusal from its concrete-typed setup really does fail initialize().
TEST(MpiTransportThroughRunner, TheHookReachesTheRunnersOwnMpiTransport) {
  auto h = make_hook_harness();
  soma::Runner runner{h->graph, h->dag, h->plan, h->backend};

  int         calls = 0;
  std::string seen_name;
  soma::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [&](soma::Transport& t) -> soma::Result<void> {
    ++calls;
    seen_name = t.name();
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    EXPECT_NE(mpi, nullptr) << "the registry's \"mpi\" factory did not produce "
                               "an MpiTransport";
    if (mpi == nullptr) return soma::Error{"E-TEST", "not an MpiTransport"};
    // COLLECTIVE, and reached by every rank: the hook runs on every rank's
    // Runner::initialize(), in the same order.
    const auto comm = mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    EXPECT_TRUE(comm.ok()) << comm.error().code << ": " << comm.error().message;
    // Refuse from inside the hook, with a code only MpiTransport produces, so
    // what comes out of initialize() cannot have been manufactured anywhere
    // else. An empty spec is refused for a stated reason -- no rank sets.
    return mpi->build_plan(TransportSpec{});
  };

  const auto init = runner.initialize(h->components, h->registry, cfg);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(seen_name, "mpi");
  ASSERT_FALSE(init.ok());
  EXPECT_EQ(init.error().code, "E-TRANSPORT-RANKS") << init.error().message;

  std::printf("[hook rank %d/%d] set_communicator reached the Runner's own "
              "MpiTransport; its build_plan refusal surfaced as %s\n",
              world_rank(), world_size(), init.error().code.c_str());
  std::fflush(stdout);
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Route 1: plan the spec the Runner will actually hand prepare(). It is
// refused, because that spec names no ranks. So the hook cannot be satisfied
// by copying the Runner's own spec.
TEST(MpiTransportThroughRunner, PlanningTheRunnersOwnSpecIsRefusedForNoRanks) {
  auto h = make_hook_harness();
  soma::Runner runner{h->graph, h->dag, h->plan, h->backend};

  soma::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [](soma::Transport& t) -> soma::Result<void> {
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    if (mpi == nullptr) return soma::Error{"E-TEST", "not an MpiTransport"};
    const auto comm = mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    if (!comm.ok()) return comm;
    return mpi->build_plan(runner_side_spec());
  };

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_FALSE(init.ok()) << "the Runner's TransportSpec carries no rank sets, "
                             "so MpiTransport cannot plan it";
  EXPECT_EQ(init.error().code, "E-TRANSPORT-RANKS") << init.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Route 2: plan a pattern MpiTransport accepts. initialize() now succeeds --
// so the hook is genuinely wired -- and the run fails at the FIRST Transport
// instruction, because the spec the Runner hands prepare() is not the one that
// was planned. This is the whole of R5's answer: there is no argument to
// build_plan() for which this path moves data, and the missing piece is on the
// Runner's side, in resolve_instructions().
TEST(MpiTransportThroughRunner, APlannableSpecStillCannotSurvivePrepare) {
  auto h = make_hook_harness();
  soma::Runner runner{h->graph, h->dag, h->plan, h->backend};

  const int rank = world_rank();
  soma::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [rank](soma::Transport& t) -> soma::Result<void> {
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    if (mpi == nullptr) return soma::Error{"E-TEST", "not an MpiTransport"};
    const auto comm = mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    if (!comm.ok()) return comm;
    return mpi->build_plan(plannable_spec(rank));
  };

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const auto run = runner.run_period(h->components);
  ASSERT_FALSE(run.ok())
      << "if this ever passes, resolve_instructions() has started to populate "
         "TransportSpec::src_ranks/dst_ranks/global_ids and R5 is closed -- "
         "turn this test into the data-movement check it should have been";
  // One code, two reasons, depending on the rank. On ranks 0 and 1 the planned
  // spec named rank sets and the Runner's does not, so the src_ranks/dst_ranks
  // sizes disagree. On a rank in neither set -- 2 and 3 at -n 4 -- the planned
  // spec had elements == 0 while the Runner's says 8, so `elements` disagrees
  // first. Both are prepare()'s single equality test; neither moves anything.
  EXPECT_EQ(run.error().code, "E-TRANSPORT-RESPEC") << run.error().message;

  // Nothing arrived. Stated as data, not inferred from the error: prepare()
  // returns before move(), so no rank entered MPI_Alltoallv and the consumer's
  // storage still holds its initial zeros.
  const auto* consumer =
      dynamic_cast<const HookConsumer*>(h->components.find("consumer"));
  ASSERT_NE(consumer, nullptr);
  for (const double v : consumer->data()) EXPECT_DOUBLE_EQ(v, 0.0);

  std::printf("[hook rank %d/%d] R5: MpiTransport planned, Runner ran, "
              "prepare() rejected the Runner's spec with %s -- the hook is "
              "wired but the spec is not\n",
              rank, world_size(), run.error().code.c_str());
  std::fflush(stdout);
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
