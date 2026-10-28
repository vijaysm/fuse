#pragma once

// physics-kit, S8's optional finite-element component. Tier 0 at this
// skeleton stage (needs only what tier 0 has: Kokkos, fmt, the fuse::runtime,
// fuse::backend, fuse::num and fuse::linalg layers already built by tier 0).
// See docs/dev/physics-kit.md for what the component is and docs/dev/
// layering.md for the direction-of-dependency rule this header's own
// location proves: physics-kit depends on fuse, never the reverse.
//
// No physics code lives here yet (S8.1 and later tracks port KARMA's FEM
// core, physics modules and FEM transfer per docs/dev/physics-kit-
// disposition.md's D-A/D-B dispositions). This header exists only to give
// the skeleton target and its one gtest (physics-kit/test/skeleton_test.cpp)
// something concrete to compile, link and assert against.

#include <string_view>

namespace fuse::physics
{

/** The FUSE project version (CMake PROJECT_VERSION, fuse/base/config.hpp's
 * FUSE_VERSION_STRING) this physics-kit build was configured against. Not a
 * physics-kit-specific version number yet: the component has no released
 * contract of its own at the skeleton stage, so it reports the project
 * version it was built from rather than inventing a second number nothing
 * reads.
 */
[[nodiscard]] std::string_view version() noexcept;

/** Whether this build was configured with the PETSc option (cmake/
 * FuseDependencies.cmake). The translation unit reads the feature define from
 * fuse/base/config.hpp; this header deliberately does not name it, so the
 * invariants gate's rule (a feature macro is read only where config.hpp is
 * included) holds for every physics-kit header -- no new find_package logic
 * for physics-kit, per the owner's decision D-B (tier 0, PETSc optional).
 */
[[nodiscard]] bool has_petsc() noexcept;

}  // namespace fuse::physics
