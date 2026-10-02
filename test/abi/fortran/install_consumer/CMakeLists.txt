# Standalone consumer project for the Fortran install/find_package() smoke
# test (ABI-FORTRAN-CONSUMER, build-evidence/abi-matrix.toml), modelled on
# test/install_consumer/CMakeLists.txt: NOT part of
# the main superbuild -- test/CMakeLists.txt's "abi.install_consumer_fortran"
# registration configures this as its own independent project, against an
# install prefix, exactly the way a real downstream Fortran code would.
cmake_minimum_required(VERSION 3.24)
# CXX is enabled alongside Fortran even though this project compiles no C++
# source of its own: fuse-coupler-targets.cmake's find_dependency(Kokkos)
# probes OpenMP_CXX, which CMake's FindOpenMP cannot do in a project that
# never enabled the CXX language at all (confirmed by running this exact
# configure without it: "Could NOT find OpenMP (missing: OpenMP_CXX_FOUND
# CXX)"). fuse::abi-fortran and fuse::abi-static are the only things this
# consumer links; neither needs the consumer to WRITE any C++.
project(fuse_install_consumer_fortran LANGUAGES Fortran CXX)

# fuse::abi-fortran is STATIC, and CMake records a static target's PRIVATE
# link dependencies (fortran/CMakeLists.txt links Threads::Threads PRIVATE,
# for the shim's pthread_once) into the IMPORTED target's own interface --
# the archive cannot carry that dependency the way a .so would, so the
# importing project must be able to resolve Threads::Threads itself too.
# Confirmed by configuring without this line: "the target was not found"
# against fuse::abi-fortran's link interface.
find_package(Threads REQUIRED)
find_package(fuse-coupler CONFIG REQUIRED)

# fuse::abi-fortran's only undefined symbol, fuse_coupler_abi, is defined in
# the table library itself, not in the Fortran shim -- the Fortran binding
# is a deliberate SECOND ABI library specifically so libfuse-abi keeps
# exactly one exported symbol (docs/dev/abi-design.md, "The costs"), which
# means a Fortran participant always links two ABI libraries where a C one
# links one.
#
# fuse::abi (SHARED), not fuse::abi-static: tried static first (matching
# "no extra .so to ship", src/CMakeLists.txt's own stated reason for
# building fuse-abi-static alongside the shared fuse-abi) and it failed at
# this consumer's own final link with undefined YAML::*/_r2d_*/_r3d_*
# symbols -- yaml-cpp and r3d are linked PRIVATE several STATIC-library
# layers down (fuse-coupler-model, fuse-coupler-backend), and that chain is
# not what `install(EXPORT)` forwards to an installed STATIC consumer. The
# shared variant has no such gap because a .so bakes its own PRIVATE
# dependencies into itself at ITS build time, nothing for an installed
# consumer to re-resolve -- confirmed by test/abi/install_consumer/
# CMakeLists.txt (the C sibling) linking fuse::abi for the identical reason.
add_executable(consumer_fortran ../consumer.f90)
target_link_libraries(consumer_fortran PRIVATE fuse::abi-fortran fuse::abi)

# two_component.yaml must sit next to the built executable -- consumer.f90
# opens it by its bare relative name, the same way main.c's C sibling would
# read a fixture placed beside it (test/install_consumer has no data file of
# its own to copy, so there is no precedent to match beyond "keep the
# fixture next to the binary that reads it").
add_custom_command(TARGET consumer_fortran POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/two_component.yaml
          $<TARGET_FILE_DIR:consumer_fortran>/two_component.yaml
)
