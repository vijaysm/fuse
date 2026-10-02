// S6 phase 3, "MPI connect" track. Plays component "a" (the producer) of
// test/parallel/connect's two-executable field transfer. See
// connect_common.hpp for the shared graph/backend/runner driver and
// run_pair.sh / test/CMakeLists.txt for how this and connect_b.cpp are
// launched together, in both launch modes (mpmd, accept/connect).

#include "connect_common.hpp"

int main(int argc, char** argv) {
  return fuse_connect_test::run_participant(argc, argv, "a");
}
