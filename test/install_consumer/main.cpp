// Links only fuse::base -- the layer with no PRIVATE third-party dependency
// that a fetched-not-installed package (yaml-cpp, r3d) would leave
// unresolved at final link (cmake/fuse-couplerConfig.cmake.in explains why).
// The point of this program is only that it configures, builds and runs
// against an INSTALLED fuse-coupler, proving find_package(fuse-coupler)
// works at all -- the one thing this file exists to
// verify, so it exercises the smallest real fuse:: type: base's Result<T>.
#include <fuse/base/result.hpp>

#include <cstdio>

int main() {
  fuse::Result<int> r(41);
  if (!r.ok()) return 1;
  std::printf("fuse-coupler install smoke test: base result ok, value=%d\n", r.value() + 1);
  return 0;
}
