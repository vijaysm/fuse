// linalg.comm -- fuse::linalg::Reducer's int64_t allgather overload
// (physics-kit's own fe::DofMap/
// AlgebraicSystem review): an allgather moves bytes, not arithmetic,
// so an integer payload travels through this NATIVE overload rather
// than a std::bit_cast<double> reinterpretation at the call site.
//
// adds `alltoallv`: the point-to-point
// exchange `fe::DofMap`'s ghost resolution and `fe::AlgebraicSystem`'s
// foreign-entry exchange now route through instead of a padded
// allgather. `SerialReducer`'s own one-participant case is exercised
// here at tier 0; the real multi-rank MPI_Alltoall/MPI_Alltoallv path
// (`MpiReducer::alltoallv`) is exercised by the new
// `physics_dof_map_mpi` test row (RANKS 2 3), build-only in this
// sandbox, since a genuine point-to-point exchange has nothing to
// show across one rank.

#include "fuse/linalg/comm.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::linalg::Reducer;
using fuse::linalg::make_serial_reducer;

/// Implements ONLY Reducer's pure virtuals (one participant, trivial
/// sum/max/allgather) and overrides NOTHING else -- unlike
/// `SerialReducer` (comm_serial.cpp), which explicitly overrides
/// `alltoallv` for its own discoverability reasons (that file's own
/// header comment) and so never actually calls the BASE CLASS default
/// this test exists to exercise. A mutation of comm.hpp's own
/// `alltoallv` default survived once (round 4, caught by this
/// project's own mutation gate) specifically because every test used
/// `SerialReducer`, whose override shadows it completely -- this class
/// is the fix: a `Reducer` with no `alltoallv` override of its own, so
/// `BareReducerAlltoallv*` below is the ONLY code path that actually
/// calls comm.hpp's default implementation.
class BareReducer : public Reducer {
 public:
  [[nodiscard]] double sum(double local) const override { return local; }
  void sum(double*, std::size_t) const override {}
  [[nodiscard]] double max(double local) const override { return local; }
  [[nodiscard]] int rank() const override { return 0; }
  [[nodiscard]] int size() const override { return 1; }
  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    std::copy(send, send + n_each, recv);
  }
};

// linalg.comm.SerialReducerInt64AllgatherIsAnExactCopy -- one
// participant: the concatenation over [0, size()) is this
// participant's own send, copied once (SerialReducer's own double
// overload states the identical contract) -- EXACT, including values
// whose magnitude a round-trip through double would lose (any
// int64_t with more than 53 significant bits), which is the whole
// point of giving integers their own native channel.
TEST(CommTest, SerialReducerInt64AllgatherIsAnExactCopy) {
  const auto reducer = make_serial_reducer();
  ASSERT_EQ(reducer->size(), 1);
  const std::vector<std::int64_t> send{1, -2, 1234567890123456789LL, 0};
  std::vector<std::int64_t> recv(send.size(), 0);
  reducer->allgather(send.data(), send.size(), recv.data());
  EXPECT_EQ(recv, send);
}

// linalg.comm.SerialReducerInt64AlltoallvIsAnIdentityEcho -- one
// participant: whatever is "sent" to anyone is sent to this one rank,
// so the payload returns unchanged and recv_counts mirrors send_counts
// (comm.hpp's own header comment on the base class's default, which
// SerialReducer overrides natively but to the identical result).
TEST(CommTest, SerialReducerInt64AlltoallvIsAnIdentityEcho) {
  const auto reducer = make_serial_reducer();
  ASSERT_EQ(reducer->size(), 1);
  const std::vector<std::int64_t> send{7, -8, 9};
  const std::vector<int> send_counts{3};
  std::vector<int> recv_counts;
  const auto recv = reducer->alltoallv(send, send_counts, recv_counts);
  EXPECT_EQ(recv_counts, send_counts);
  EXPECT_EQ(recv, send);
}

// linalg.comm.SerialReducerDoubleAlltoallvIsAnIdentityEcho -- the
// double analogue of the above, the genuine VALUE channel
// AlgebraicSystem's foreign-entry exchange needs alongside the int64
// one.
TEST(CommTest, SerialReducerDoubleAlltoallvIsAnIdentityEcho) {
  const auto reducer = make_serial_reducer();
  const std::vector<double> send{1.5, -2.5};
  const std::vector<int> send_counts{2};
  std::vector<int> recv_counts;
  const auto recv = reducer->alltoallv(send, send_counts, recv_counts);
  EXPECT_EQ(recv_counts, send_counts);
  EXPECT_EQ(recv, send);
}

// linalg.comm.BareReducerAlltoallvInt64UsesTheBaseClassDefault -- the
// SAME identity-echo contract, now actually exercising comm.hpp's own
// default implementation (this file's own header comment on
// BareReducer states why SerialReducer's own tests above do not).
TEST(CommTest, BareReducerAlltoallvInt64UsesTheBaseClassDefault) {
  const BareReducer reducer;
  ASSERT_EQ(reducer.size(), 1);
  const std::vector<std::int64_t> send{11, -22, 33};
  const std::vector<int> send_counts{3};
  std::vector<int> recv_counts;
  const auto recv = reducer.alltoallv(send, send_counts, recv_counts);
  EXPECT_EQ(recv_counts, send_counts);
  EXPECT_EQ(recv, send);
}

// linalg.comm.BareReducerAlltoallvDoubleUsesTheBaseClassDefault -- the
// double analogue.
TEST(CommTest, BareReducerAlltoallvDoubleUsesTheBaseClassDefault) {
  const BareReducer reducer;
  const std::vector<double> send{4.5, -6.5};
  const std::vector<int> send_counts{2};
  std::vector<int> recv_counts;
  const auto recv = reducer.alltoallv(send, send_counts, recv_counts);
  EXPECT_EQ(recv_counts, send_counts);
  EXPECT_EQ(recv, send);
}

}  // namespace
