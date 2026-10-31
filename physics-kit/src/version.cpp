// The one placeholder translation unit for the physics-kit skeleton
// (physics-kit/CMakeLists.txt's fuse_physics_kit target). Exists so the
// target compiles and links something; no physics code yet (see
// docs/dev/physics-kit.md).

#include "fuse_physics/version.hpp"

#include "fuse/base/config.hpp"

namespace fuse::physics
{

std::string_view version() noexcept
{
    return FUSE_VERSION_STRING;
}

bool has_petsc() noexcept
{
    return static_cast<bool>(FUSE_HAVE_PETSC);
}

}  // namespace fuse::physics
