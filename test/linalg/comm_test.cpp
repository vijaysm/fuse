// linalg.comm -- fuse::linalg::Reducer's int64_t allgather overload
// (S8.3 wave 2 round 2 item B, physics-kit's own fe::DofMap/
// AlgebraicSystem review): an allgather moves bytes, not arithmetic,
// so an integer payload travels through this NATIVE overload rather
// than a std::bit_cast<double> reinterpretation at the call site.

#include "fuse/linalg/comm.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::linalg::make_serial_reducer;

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

}  // namespace
