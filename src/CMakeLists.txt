# fuse:: library targets, one per layer (docs/dev/layering.md). Each layer links
# only the layers below it; the layering gate checks the includes, this file
# checks the link graph -- both must agree.

# ---------------------------------------------------------------- base
# Exact time, units, ids, versions, diagnostics, canonical bytes. Almost
# header-only; version.cpp is the one translation unit. Depends on nothing
# above Tier 0: Kokkos is linked for the execution-space vocabulary later
# layers need in their public headers, fmt for Diagnostics formatting.
# Exceptions on by default; OFF turns every FUSE_THROW/ASSERT/REFUSE into
# report-then-abort after the same on-error policy has run. Types and call
# sites are unconditional either way (docs/dev/conventions.md).
option(FUSE_ENABLE_EXCEPTIONS "Throw C++ exceptions on errors (OFF = report and abort)" ON)
set(FUSE_HAVE_MPI    ${FUSE_ENABLE_MPI})
set(FUSE_HAVE_KOKKOS ON)
set(FUSE_HAVE_MOAB   ${FUSE_ENABLE_MOAB})
set(FUSE_HAVE_PETSC  ${FUSE_ENABLE_PETSC})
configure_file(${PROJECT_SOURCE_DIR}/include/fuse/base/config.hpp.in
               ${PROJECT_BINARY_DIR}/include/fuse/base/config.hpp @ONLY)

add_library(fuse-coupler-base
  base/backtrace.cpp
  base/error.cpp
)
add_library(fuse::base ALIAS fuse-coupler-base)
target_include_directories(fuse-coupler-base PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(fuse-coupler-base PUBLIC cxx_std_20)
# Exceptions on by default; OFF turns every FUSE_THROW/ASSERT/REFUSE into
# report-then-abort after the same on-error policy has run. The types and
# call sites are unconditional either way (docs/dev/conventions.md).
target_link_libraries(fuse-coupler-base PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(fuse-coupler-base PUBLIC Kokkos::kokkos fmt::fmt)
set_target_properties(fuse-coupler-base PROPERTIES
  CXX_VISIBILITY_PRESET hidden      # the C ABI exports one symbol per ABI; nothing else leaks
  VISIBILITY_INLINES_HIDDEN ON
)
