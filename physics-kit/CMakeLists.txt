# physics-kit -- the optional finite-element component (docs/dev/physics-kit.md).
#
# A separate optional component, not a fuse:: layer (docs/dev/layering.md):
# it depends on FUSE and FUSE never depends on it back. tools/check-layering.py's
# physics-kit check enforces the "never back" half (nothing under include/fuse/,
# src/, python/ or examples/ may include a fuse_physics/ header or link
# fuse::physics_kit); this file enforces the "depends on FUSE" half, by
# linking only fuse:: targets below.
#
# Tier 0: needs nothing beyond what
# fuse::runtime, fuse::backend, fuse::num and fuse::linalg already need
# (Kokkos, fmt). FUSE_ENABLE_PETSC is OPTIONAL, read through the existing
# FUSE_HAVE_PETSC define (fuse/base/config.hpp) -- no new find_package(PETSc)
# call here, the same posture examples/CMakeLists.txt already takes for
# fuse_example_dn_heat and fuse_example_petsc_reaction_diffusion (link
# fuse::petsc only under FUSE_ENABLE_PETSC; a #if FUSE_HAVE_PETSC call site
# reaches the macro either way through config.hpp, which fuse::runtime's
# PUBLIC link chain already carries down to fuse::base).

add_library(fuse_physics_kit
  src/version.cpp
)
add_library(fuse::physics_kit ALIAS fuse_physics_kit)

target_compile_features(fuse_physics_kit PUBLIC cxx_std_20)

target_include_directories(fuse_physics_kit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(fuse_physics_kit PUBLIC
  fuse::runtime fuse::backend fuse::num fuse::linalg
)

if(FUSE_ENABLE_PETSC)
  # No physics-kit source calls into fuse::petsc yet (nothing here is
  # Newton-coupled). Linked ahead of need, PRIVATE, the same way the two
  # PETSc-gated examples link it unconditionally once FUSE_ENABLE_PETSC is
  # on, so a later "#if FUSE_HAVE_PETSC" call site never discovers the
  # target missing.
  target_link_libraries(fuse_physics_kit PRIVATE fuse::petsc)
endif()

set_target_properties(fuse_physics_kit PROPERTIES
  EXPORT_NAME physics_kit
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# AFTER test/'s own add_subdirectory at the root (root CMakeLists.txt orders
# add_subdirectory(physics-kit) after its FUSE_ENABLE_TESTS block for exactly
# this reason): test/CMakeLists.txt defines fuse_add_gtest and friends, and
# this is the first caller of that machinery from OUTSIDE test/'s own
# directory tree -- see test/CMakeLists.txt's fuse_add_gtest SRC_DIR
# argument and physics-kit/test/CMakeLists.txt's own comment.
if(FUSE_ENABLE_TESTS)
  add_subdirectory(test)
endif()
