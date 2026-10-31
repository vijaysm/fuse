# Standalone consumer project for the install/find_package() smoke test
# (REVIEW.md R-25/BT-F9): NOT part of the main superbuild -- test/CMakeLists.txt's
# "# ---- install" section configures this as its own independent project,
# against an install prefix, exactly the way a real downstream consumer would.
cmake_minimum_required(VERSION 3.24)
project(fuse_install_consumer LANGUAGES CXX)

# WITH_PHYSICS_KIT is passed by run-smoke-test.cmake from the installing
# build's own FUSE_ENABLE_PHYSICS_KIT: when ON, the component is REQUIRED
# and a second consumer links fuse::physics_kit (S8.2, 2026-10-31); when OFF,
# the component is asked for OPTIONALLY and must come back NOT found, so an
# install without physics-kit is proved to refuse the component cleanly
# rather than leave the target undefined.
option(WITH_PHYSICS_KIT "consume the physics_kit component" OFF)
if(WITH_PHYSICS_KIT)
  find_package(fuse-coupler CONFIG REQUIRED COMPONENTS physics_kit)
else()
  find_package(fuse-coupler CONFIG REQUIRED OPTIONAL_COMPONENTS physics_kit)
  if(fuse-coupler_physics_kit_FOUND OR TARGET fuse::physics_kit)
    message(FATAL_ERROR "install without physics-kit still exposes the physics_kit component")
  endif()
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE fuse::base)
if(WITH_PHYSICS_KIT)
  add_executable(consumer_physics main_physics.cpp)
  target_link_libraries(consumer_physics PRIVATE fuse::physics_kit)
endif()
