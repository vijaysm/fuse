// KNOWN_ISSUES.md D18, round 2. Plays link 0 (component "a") of
// test/parallel/connect's N-way connect chain. See connect_chain_common.hpp
// for the shared graph/backend/runner driver and run_chain.sh for how this
// and its siblings are launched together.

#include "connect_chain_common.hpp"

int main(int argc, char** argv) {
  return fuse_connect_chain::run_chain_participant(argc, argv, 0);
}
