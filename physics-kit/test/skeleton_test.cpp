// physics.skeleton -- proves fuse_physics_kit (physics-kit/CMakeLists.txt)
// compiles, links against fuse::runtime/fuse::backend/fuse::num/
// fuse::linalg, and that FUSE_HAVE_PETSC (fuse/base/config.hpp) reaches
// physics-kit code without physics-kit adding any find_package(PETSc) of
// its own (owner decision D-B: physics-kit is tier 0 with PETSc optional).
//
// No physics code exists yet (docs/dev/physics-kit.md); this is the one
// test the skeleton stage asks for, not a stand-in for behavioural
// coverage of a component that has none yet.

#include "fuse_physics/version.hpp"

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

namespace
{

TEST(PhysicsSkeleton, VersionStringIsNonEmpty)
{
    const std::string_view version = fuse::physics::version();
    EXPECT_FALSE(version.empty());
    // The project version string (FUSE_VERSION_STRING, fuse/base/config.hpp.in),
    // not a physics-kit-specific contract -- see version.hpp's own comment on
    // why the skeleton reports that number rather than inventing a second one.
    EXPECT_EQ(version, std::string_view(FUSE_VERSION_STRING));
}

TEST(PhysicsSkeleton, HasPetscReflectsBuildOption)
{
    // Reads the SAME FUSE_HAVE_PETSC define fuse_physics_kit's own
    // src/version.cpp reads, so a build that enables FUSE_ENABLE_PETSC
    // without rebuilding this test from a stale cache cannot silently
    // disagree with itself.
#if FUSE_HAVE_PETSC
    EXPECT_TRUE(fuse::physics::has_petsc());
#else
    EXPECT_FALSE(fuse::physics::has_petsc());
#endif
}

}  // namespace
