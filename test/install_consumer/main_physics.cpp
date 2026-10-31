// Install smoke test for the physics_kit component: the installed
// fuse_physics/version.hpp is found through fuse::physics_kit's exported
// include directory, and the installed library resolves the symbol.
#include <fuse_physics/version.hpp>

#include <cstdio>

int main() {
  const auto v = fuse::physics::version();
  if (v.empty()) return 1;
  std::printf("fuse-coupler install smoke test: physics_kit version=%.*s petsc=%d\n",
              static_cast<int>(v.size()), v.data(), fuse::physics::has_petsc() ? 1 : 0);
  return 0;
}
