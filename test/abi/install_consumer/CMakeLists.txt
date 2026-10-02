# Standalone consumer project for ABI-C-CONSUMER's install/find_package()
# smoke test -- NOT part of the main superbuild, the same way
# test/install_consumer/CMakeLists.txt (fuse::base's own install check)
# isn't: test/CMakeLists.txt's "abi.install_consumer_c" entry configures
# this as its own independent project, against an install prefix, exactly
# the way a real downstream C-only consumer would.
cmake_minimum_required(VERSION 3.24)
# CXX, not just C: fuse-coupler's own dependency chain (Kokkos -> its
# KokkosConfig.cmake calls find_dependency(OpenMP), which CMake's FindOpenMP
# probes per ENABLED LANGUAGE) needs the CXX language enabled in whichever
# project calls find_package(fuse-coupler), even though every line of code
# in this consumer -- main.c, and the one target below -- is C. A real
# downstream C-only consumer hits this same requirement and works around it
# the same way: this is a property of the dependency graph fuse-coupler
# sits on top of (Kokkos), not of the ABI itself, which main.c's single
# #include <fuse/abi/participant.h> and single `fuse::abi` link line still
# prove is a pure-C surface.
project(fuse_abi_install_consumer LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(fuse-coupler CONFIG REQUIRED)

add_executable(fuse_abi_consumer main.c)
target_compile_options(fuse_abi_consumer PRIVATE -Wpedantic -Wall)
target_link_libraries(fuse_abi_consumer PRIVATE fuse::abi)
