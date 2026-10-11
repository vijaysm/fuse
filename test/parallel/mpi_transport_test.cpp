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
// WHY THIS IS NOT A fuse_add_mpi_test(). It calls MPI_Init, it is launched by
// mpiexec, and -- the part that matters -- a GoogleTest assertion failure
// unwinds one rank out of a TEST body while its peers walk on into the next
// collective, which is a hang, not a failure. The listener installed in main()
// turns the first failed assertion on any rank into MPI_Abort, so a broken
// build fails the CTest entry in seconds instead of sitting on its timeout.

// Explicit, not transitive: the feature macros are read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"

#include <fuse/transport/comm_graph.hpp>
#include <gtest/gtest.h>

// R-50 (REVIEW.md): the documented rank-requirement convention
// (docs/dev/testing.md). `halves()` below is a divisibility check the
// documented macros cannot express, so it stays hand-rolled.
#include "fuse_test_main.hpp"

#if FUSE_HAVE_MPI

#include <mpi.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

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

using fuse::MpiTransport;
using fuse::TransportSpec;

namespace {
/// Cells per side of the Runner-driven harness far below. One value, because
/// producer and consumer must agree for SerialMeshBackend's identity remap.
constexpr std::size_t kHookCells = 8;
}  // namespace

// ---------------------------------------------------------------------------
// Allocation counting, the same mechanism test/unit/runner_test.cpp uses to
// prove the hot loop allocates nothing. Replacing the global operator new is
// the only way to measure it rather than assert it by inspection. MPICH is C
// and allocates through malloc, so what is counted here is FUSE's own
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
  // #ifndef __CUDA_ARCH__, not just std::abort() instead of throw: a
  // CUDA-enabled Kokkos dual-compiles every global operator new/delete
  // override for device too (nothing here is reachable from a kernel; nvcc
  // does this unconditionally), and the device side can neither support
  // exceptions nor see g_counting/g_allocations, host globals.
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) std::abort();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
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
  const auto t = fuse::TransportRegistry::instance().create("mpi");
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
  FUSE_REQUIRE_RANKS(4);
  Pattern p;
  p.src = {0, 1, 2};
  p.dst = {3};
  exchange_and_check(p, "3->1");
}

// The other direction: one producer, three consumers.
TEST(MpiTransport, RedistributesOneToThree) {
  FUSE_REQUIRE_RANKS(4);
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
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  Pattern p;
  p.src = {0};
  p.dst = {1};
  exchange_and_check(p, "idle");
}

// A rank that owns zero elements of a mesh entity set is ordinary in parallel, not an
// edge case: it is a member of a rank set, it has nothing to contribute, and
// `std::vector<double>{}.data()` is null. Both must work.
TEST(MpiTransport, RankOwningZeroElementsIsOrdinary) {
  const int size = world_size();
  FUSE_REQUIRE_AT_LEAST_RANKS(4);
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

// Two field transfers alive at once. Each MpiTransport duplicates the communicator,
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
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
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

// R-30 (REVIEW.md): build_plan() has no cross-rank agreement on
// TransportSpec::components, unlike build_plan_from_peers()'s Phase B
// numbering check. Rank 0 calls it with components = 1, rank 1 with
// components = 2, over an otherwise valid disjoint pattern; every rank must
// reject with E-TRANSPORT-PEERMATCH rather than build a plan whose
// MPI_Alltoallv count arrays disagree between the two sides. Needs a host
// run: the sandbox this was written in cannot launch MPI at all.
TEST(MpiTransport, BuildPlanRejectsDisagreeingComponentsAcrossRanks) {
  FUSE_REQUIRE_RANKS(2);
  const int rank = world_rank();

  TransportSpec spec;
  spec.src_ranks = {0};
  spec.dst_ranks = {1};
  spec.elements = 1;
  spec.global_ids = {0};
  spec.components = (rank == 0) ? 1 : 2;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.build_plan(spec);
  // Printed on every rank, pass or fail: the 2026-09-20 host run had rank 1
  // rejecting and rank 0 building a plan, which the assertion alone does not
  // distinguish from both ranks missing the check.
  std::printf("[components rank %d/%d] components=%d outcome=%s %s\n", rank,
              world_size(), spec.components, r.ok() ? "OK" : "REFUSED",
              r.ok() ? "" : r.error().code.c_str());
  std::fflush(stdout);
  ASSERT_FALSE(r.ok()) << "rank " << rank
                       << ": ranks disagreed on components, and every rank "
                          "must reject together rather than one side sizing "
                          "MPI_Alltoallv from its own, different, count";
  EXPECT_EQ(r.error().code, "E-TRANSPORT-PEERMATCH") << r.error().message;
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
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
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
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
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
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
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
// int. A buffer longer than INT_MAX doubles must be rejected, not truncated.
// Reached with a huge component count so the test costs no memory.
TEST(MpiTransport, BuildPlanRejectsCountsBeyondIntMax) {
  const int size = world_size();
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
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
class HookProducer final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "producer"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (double& v : data_) v = 2.0;
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> data_ = std::vector<double>(kHookCells, 0.0);
};

class HookConsumer final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_ = std::vector<double>(kHookCells, 0.0);
};

/// producer (rank 0) -> consumer (rank 1). The rank ranges are DISJOINT, which
/// is what makes compilation into the dataflow graph emit an OpKind::Transport at all -- see the
/// `ranks_differ` computation in src/ir/rules_channel.cpp. Equal cell counts on
/// both sides so the Remap that follows is the identity SerialMeshBackend can
/// do; anything else would fail for an unrelated reason.
struct HookHarness {
  fuse::CouplingGraph     graph;
  fuse::SerialMeshBackend backend;
  fuse::OperatorDAG       dag;
  fuse::ExecutionPlan     plan;
  fuse::ComponentSet      components;
  fuse::FieldRegistry     registry;
};

std::unique_ptr<HookHarness> make_hook_harness() {
  auto  h = std::make_unique<HookHarness>();
  auto& g = h->graph;
  namespace u = fuse::units;

  const std::string src = "generated:line:" + std::to_string(kHookCells);
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = src});
  const auto r1 =
      g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "producer", .ranks = {0, 1},
      .interval = fuse::Rational{1}});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = {1, 1},
      .interval = fuse::Rational{1}});
  const auto sf = g.add_field(
      p, fuse::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                         .intent = fuse::Intent::Provides});
  const auto tf = g.add_field(
      c, fuse::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                         .intent = fuse::Intent::Requires});
  g.add_channel("link", sf, tf);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = fuse::lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, h->dag, clocks,
                              fuse::make_region_sizes(g, h->backend));
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
// rejection from its concrete-typed setup really does fail initialize().
TEST(MpiTransportThroughRunner, TheHookReachesTheRunnersOwnMpiTransport) {
  auto h = make_hook_harness();
  fuse::Runner runner{h->graph, h->dag, h->plan, h->backend};

  int         calls = 0;
  std::string seen_name;
  fuse::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [&](fuse::Transport& t) -> fuse::Result<void> {
    ++calls;
    seen_name = t.name();
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    EXPECT_NE(mpi, nullptr) << "the registry's \"mpi\" factory did not produce "
                               "an MpiTransport";
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "not an MpiTransport"};
    // COLLECTIVE, and reached by every rank: the hook runs on every rank's
    // Runner::initialize(), in the same order.
    const auto comm = mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    EXPECT_TRUE(comm.ok()) << comm.error().code << ": " << comm.error().message;
    // Reject from inside the hook, with a code only MpiTransport produces, so
    // what comes out of initialize() cannot have been manufactured anywhere
    // else. An empty spec is rejected for a stated reason -- no rank sets.
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
// rejected, because that spec names no ranks. So the hook cannot be satisfied
// by copying the Runner's own spec.
TEST(MpiTransportThroughRunner, PlanningTheRunnersOwnSpecIsRefusedForNoRanks) {
  auto h = make_hook_harness();
  fuse::Runner runner{h->graph, h->dag, h->plan, h->backend};

  fuse::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "not an MpiTransport"};
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
  // plannable_spec() names two rank sets, so it needs two ranks to name.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_hook_harness();
  fuse::Runner runner{h->graph, h->dag, h->plan, h->backend};

  const int rank = world_rank();
  fuse::RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.transport_setup = [rank](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "not an MpiTransport"};
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

// No main(): test/support/fuse_test_main reduces the failure status across
// MPI_COMM_WORLD, so a failure on one rank fails every rank instead of leaving
// the others to block in the next collective. A test that keeps its own
// entry point silently loses that.

// ---------------------------------------------------------------------------
// The peer-wise plan (R31): partial overlap, and the matched plan as oracle
// ---------------------------------------------------------------------------

namespace {

/// This rank's send and receive peer lists, derived from the pattern DIRECTLY
/// rather than taken from a rank graph. That independence is the point: the
/// test below requires the peer-built plan to deliver exactly what the matched
/// plan delivers, so reusing the graph's own output on both sides would only
/// prove the two agree with each other.
///
/// The orders reproduce what the matched path does, and they are not
/// interchangeable: a send block to peer `d` is in **`d`'s** id order, because
/// that is the order `d` unpacks in; a receive block from peer `s` is in **my**
/// id order, because that is the order `s` walked my list in.
void peers_for(const Pattern& p, int rank, std::vector<fuse::CommGraphPeer>& send,
               std::vector<fuse::CommGraphPeer>& recv) {
  send.clear();
  recv.clear();
  const std::vector<std::size_t> mine = ids_for(p, rank);
  const bool is_source = position_in(p.src, rank) >= 0;
  const bool is_dest = position_in(p.dst, rank) >= 0;

  // Where each id lives: source rank `si` owns the block [si*per, (si+1)*per).
  const std::size_t per = p.global / p.src.size();
  const auto owner_of = [&](std::size_t gid) {
    return p.src[std::min(gid / per, p.src.size() - 1)];
  };

  if (is_source) {
    for (const int d : p.dst) {          // ascending: p.dst is built ascending
      if (d == rank) continue;           // same-rank matches never travel
      fuse::CommGraphPeer peer;
      peer.base_rank = d;
      Pattern q = p;
      for (const std::size_t gid : ids_for(q, d)) {   // d's own id order
        if (owner_of(gid) != rank) continue;
        const auto it = std::find(mine.begin(), mine.end(), gid);
        if (it == mine.end()) continue;
        peer.local_indices.push_back(
            static_cast<std::size_t>(it - mine.begin()));
        peer.global_ids.push_back(gid);
      }
      if (!peer.local_indices.empty()) send.push_back(std::move(peer));
    }
  }

  if (is_dest) {
    for (const int s : p.src) {
      if (s == rank) continue;
      fuse::CommGraphPeer peer;
      peer.base_rank = s;
      for (std::size_t i = 0; i < mine.size(); ++i) {   // my id order
        if (owner_of(mine[i]) != s) continue;
        peer.local_indices.push_back(i);
        peer.global_ids.push_back(mine[i]);
      }
      if (!peer.local_indices.empty()) recv.push_back(std::move(peer));
    }
  }

  std::sort(send.begin(), send.end(),
            [](const auto& a, const auto& b) { return a.base_rank < b.base_rank; });
  std::sort(recv.begin(), recv.end(),
            [](const auto& a, const auto& b) { return a.base_rank < b.base_rank; });
}

/// Matches with both ends on this rank: ids it owns as a source and also wants
/// as a destination. Empty unless the pattern overlaps.
std::vector<fuse::LocalCopyPair> local_for(const Pattern& p, int rank) {
  std::vector<fuse::LocalCopyPair> out;
  if (position_in(p.src, rank) < 0 || position_in(p.dst, rank) < 0) return out;
  // With one id list per rank, a rank in both sets holds one buffer that is
  // both its source and its destination view, so a self-match is an element
  // copied to itself. The plan must carry it rather than silently drop it.
  const std::vector<std::size_t> mine = ids_for(p, rank);
  const std::size_t per = p.global / p.src.size();
  const auto owner_of = [&](std::size_t gid) {
    return p.src[std::min(gid / per, p.src.size() - 1)];
  };
  for (std::size_t i = 0; i < mine.size(); ++i) {
    if (owner_of(mine[i]) == rank) out.push_back({i, i});
  }
  return out;
}

/// Runs a pattern through the peer-wise builder and returns the delivered
/// buffer, so a caller can compare it against the matched path's.
std::vector<double> deliver_via_peers(const Pattern& p, int rank) {
  const TransportSpec spec = spec_for(p, rank);
  const auto nc = static_cast<std::size_t>(p.components);
  const std::size_t total = spec.elements * nc;

  std::vector<fuse::CommGraphPeer> send_plan;
  std::vector<fuse::CommGraphPeer> recv_plan;
  peers_for(p, rank, send_plan, recv_plan);

  MpiTransport t;
  EXPECT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto plan = t.build_plan_from_peers(send_plan, recv_plan,
                                            local_for(p, rank), spec.elements,
                                            p.components);
  EXPECT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;
  if (!plan.ok()) return {};

  std::vector<double> send(total, 0.0);
  std::vector<double> recv(total, -12345.0);
  if (position_in(p.src, rank) >= 0) {
    for (std::size_t i = 0; i < spec.elements; ++i) {
      for (std::size_t c = 0; c < nc; ++c) {
        send[i * nc + c] = value_of(spec.global_ids[i]) + static_cast<double>(c);
      }
    }
  }
  EXPECT_TRUE(t.prepare(spec).ok());
  const auto mv = t.move(total == 0 ? nullptr : send.data(),
                         total == 0 ? nullptr : recv.data(), total);
  EXPECT_TRUE(mv.ok()) << mv.error().code << ": " << mv.error().message;
  return recv;
}

}  // namespace

TEST(MpiTransportPeers, DeliversWhatTheMatchedPlanDeliversOnADisjointPattern) {
  // The differential oracle. The matched path is verified at 2-5 ranks; the
  // peer path is new. On a pattern both can express they must agree element by
  // element -- and the peer lists were derived from the pattern independently,
  // so agreement is evidence about the transport rather than about one
  // derivation.
  Pattern p;
  if (!halves(world_size(), p)) GTEST_SKIP() << "rank count does not split 12";
  const int rank = world_rank();
  const TransportSpec spec = spec_for(p, rank);
  const auto nc = static_cast<std::size_t>(p.components);
  const std::size_t total = spec.elements * nc;

  const std::vector<double> via_peers = deliver_via_peers(p, rank);
  ASSERT_EQ(via_peers.size(), total);

  // The oracle, run in the same process on the same data.
  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.build_plan(spec).ok());
  std::vector<double> send(total, 0.0);
  std::vector<double> via_match(total, -12345.0);
  if (position_in(p.src, rank) >= 0) {
    for (std::size_t i = 0; i < spec.elements; ++i) {
      for (std::size_t c = 0; c < nc; ++c) {
        send[i * nc + c] = value_of(spec.global_ids[i]) + static_cast<double>(c);
      }
    }
  }
  ASSERT_TRUE(t.prepare(spec).ok());
  ASSERT_TRUE(t.move(total == 0 ? nullptr : send.data(),
                     total == 0 ? nullptr : via_match.data(), total).ok());

  for (std::size_t k = 0; k < total; ++k) {
    ASSERT_DOUBLE_EQ(via_peers[k], via_match[k])
        << "rank " << rank << " slot " << k
        << ": the peer-wise plan and the matched plan disagree";
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransportPeers, DeliversUnderPartialOverlap) {
  // THE CASE THE MATCHED PATH CANNOT EXPRESS, and the reason R31 was open:
  // every rank is both a source and a destination, so one global_ids list per
  // rank cannot say which role an entry plays. Each rank ends up with a mix of
  // values received from peers and values copied locally.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  Pattern p;
  p.src.resize(static_cast<std::size_t>(world_size()));
  std::iota(p.src.begin(), p.src.end(), 0);
  p.dst = p.src;                       // fully shared: the extreme of overlap
  p.global = static_cast<std::size_t>(world_size()) * 3;
  const int rank = world_rank();
  const TransportSpec spec = spec_for(p, rank);
  const auto nc = static_cast<std::size_t>(p.components);
  const std::size_t total = spec.elements * nc;

  const std::vector<double> got = deliver_via_peers(p, rank);
  ASSERT_EQ(got.size(), total);
  for (std::size_t i = 0; i < spec.elements; ++i) {
    for (std::size_t c = 0; c < nc; ++c) {
      ASSERT_DOUBLE_EQ(got[i * nc + c],
                       value_of(spec.global_ids[i]) + static_cast<double>(c))
          << "rank " << rank << " element " << i << " (global id "
          << spec.global_ids[i] << ")";
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(MpiTransportPeers, DisagreeingHalvesAreRefusedBeforeTheRun) {
  // The numbering check. One rank is given a send plan its partner does not
  // expect; every rank must return E-TRANSPORT-PEERMATCH rather than one rank
  // sending into a receive nobody posted.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const int rank = world_rank();
  std::vector<fuse::CommGraphPeer> send_plan;
  std::vector<fuse::CommGraphPeer> recv_plan;
  if (rank == 0) {
    fuse::CommGraphPeer peer;
    peer.base_rank = 1;
    peer.local_indices = {0};
    peer.global_ids = {0};
    send_plan.push_back(peer);          // rank 1 posts no matching receive
  }

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto plan = t.build_plan_from_peers(send_plan, recv_plan, {}, 4, 1);
  ASSERT_FALSE(plan.ok()) << "a plan whose halves disagree was accepted";
  EXPECT_EQ(plan.error().code, "E-TRANSPORT-PEERMATCH")
      << "every rank must report the same code: " << plan.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Regression guard for a real bug in this host's Open MPI (4.1.5), not in
// fuse: its default "basic_linear" MPI_Alltoallv algorithm hangs -- rank 0
// returns, every other rank sits inside the collective forever -- whenever
// the communicator is anything other than MPI_COMM_WORLD itself (a plain
// MPI_Comm_dup of it is enough), on this exact disjoint-halves pattern.
// Measured 2026-10-10: reproduced with zero fuse or Kokkos code involved,
// independent of MPI_THREAD_MULTIPLE (still hangs under MPI_THREAD_FUNNELED
// and even plain MPI_Init), independent of BTL (vader, tcp) and PML (ob1),
// and absent on MPI_COMM_WORLD with the identical send/recv pattern. An
// MPI_Barrier on the duplicate right after MPI_Comm_dup -- the first fix
// tried -- does NOT help; the hang is inside MPI_Alltoallv itself. The only
// thing that worked: forcing the "pairwise" algorithm
// (`ompi_info --param coll tuned --level 9`'s coll_tuned_alltoallv_algorithm
// enum). test/CMakeLists.txt's fuse_add_mpi_test sets exactly that via
// OMPI_MCA_coll_tuned_use_dynamic_rules=1 and
// OMPI_MCA_coll_tuned_alltoallv_algorithm=2 on every MPI test's ENVIRONMENT,
// which is what this test actually exercises: it fails the moment that
// workaround is missing or broken, long before a real MpiTransport test
// would time out trying to tell you why.
TEST(MpiTransportDiag, RawAlltoallvDisjointHalvesOnADuplicatedCommAtFourRanks) {
  FUSE_REQUIRE_RANKS(4);
  const int rank = world_rank();

  MPI_Comm dup = MPI_COMM_NULL;
  ASSERT_EQ(MPI_Comm_dup(MPI_COMM_WORLD, &dup), MPI_SUCCESS);
  ASSERT_EQ(MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN), MPI_SUCCESS);

  int sendcounts[4] = {0, 0, 0, 0}, sdispls[4] = {0, 0, 0, 0};
  int recvcounts[4] = {0, 0, 0, 0}, rdispls[4] = {0, 0, 0, 0};
  double sendbuf[8] = {0}, recvbuf[8] = {0};
  if (rank == 0 || rank == 1) {
    sendcounts[2] = 3;
    sendcounts[3] = 3;
    sdispls[3] = 3;
    for (int i = 0; i < 6; ++i) sendbuf[i] = rank * 100 + i;
  }
  if (rank == 2 || rank == 3) {
    recvcounts[0] = 3;
    recvcounts[1] = 3;
    rdispls[1] = 3;
  }
  const int rc = MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, recvbuf,
                               recvcounts, rdispls, MPI_DOUBLE, dup);
  ASSERT_EQ(rc, MPI_SUCCESS);
  ASSERT_EQ(MPI_Barrier(dup), MPI_SUCCESS);
  MPI_Comm_free(&dup);
}

#endif  // the guard whose body the removed main() sat inside
