# FindLowFive.cmake -- LowFive is a runtime-loaded HDF5 VOL plugin, so both
# the library and a SHARED HDF5 must be present; a static HDF5 cannot back a
# VOL plugin. Neither LowFive installation on this lab's /opt
# (/opt/lowfive, /opt/lowfive-shared) ships a CMake package of its own (only
# /opt/lowfive-shared ships one at all, and that one is its vendored fmt's,
# not LowFive's) -- find_package(LowFive) has no CONFIG mode to fall back to,
# which is why this MODULE exists at all.
#
# Produces, on success:
#   LowFive_FOUND, LOWFIVE_INCLUDE_DIR, LOWFIVE_LIBRARY
#   imported target fuse::lowfive-raw (the bare library + include path; see
#   cmake/FuseDependencies.cmake for fuse::lowfive, which adds the fmt
#   dependency and HDF5 on top).
#
# Search roots, in order: LowFive_DIR / LowFive_ROOT (CMake's own
# find_package hint variables), then CMAKE_PREFIX_PATH, then this lab's
# default /opt/lowfive-shared.
#
# DOES NOT run the two-HDF5-image probe (tools/probe-hdf5-images.sh): that
# is a DYNAMIC check -- it needs to link the SAME HDF5 and MOAB the actual
# build will use -- and runs separately from cmake/FuseDependencies.cmake's
# LowFive block, after MOAB has already been resolved. This module only
# answers "does a LowFive installation exist here, and is it a shared
# library".

find_path(LOWFIVE_INCLUDE_DIR lowfive/vol-base.hpp
          HINTS ${LowFive_DIR} ${LowFive_ROOT} ${CMAKE_PREFIX_PATH} /opt/lowfive-shared
          PATH_SUFFIXES include)
find_library(LOWFIVE_LIBRARY NAMES lowfive
             HINTS ${LowFive_DIR} ${LowFive_ROOT} ${CMAKE_PREFIX_PATH} /opt/lowfive-shared
             PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LowFive
  REQUIRED_VARS LOWFIVE_INCLUDE_DIR LOWFIVE_LIBRARY)

if(NOT LowFive_FOUND)
  return()
endif()

# A .a here means the plugin cannot be dlopened -- catch it at configure
# time rather than at run time, exactly as the stale-toolchain check does
# for MOAB a few hundred lines below in FuseDependencies.cmake.
get_filename_component(_fuse_lowfive_ext "${LOWFIVE_LIBRARY}" EXT)
if(_fuse_lowfive_ext STREQUAL ".a")
  message(FATAL_ERROR
    "fuse: the LowFive found at ${LOWFIVE_LIBRARY} is a static archive. A "
    "runtime-loaded HDF5 VOL plugin must be shared (.dylib/.so); rebuild "
    "LowFive with -DBUILD_SHARED_LIBS=ON. See docs/user/lowfive.md.")
endif()

# LowFive's own headers (lowfive/vol-metadata.hpp) include <fmt/core.h>
# unconditionally. Prefer a copy co-located with the LowFive install (what
# /opt/lowfive-shared ships); fall back to DIY's vendored copy, since LowFive
# also depends on DIY and the two are built from matching fmt versions in
# this lab's installs (see tools/probe-hdf5-images.sh, which makes the same
# choice for the probe binary).
if(EXISTS "${LOWFIVE_INCLUDE_DIR}/fmt/core.h")
  set(_fuse_lowfive_fmt_include "${LOWFIVE_INCLUDE_DIR}")
elseif(EXISTS "/opt/diy/include/diy/thirdparty/fmt/core.h")
  set(_fuse_lowfive_fmt_include "/opt/diy/include/diy/thirdparty")
else()
  message(FATAL_ERROR
    "fuse: LowFive was found at ${LOWFIVE_INCLUDE_DIR}, but no fmt/core.h "
    "was found alongside it or in DIY's vendored copy "
    "(/opt/diy/include/diy/thirdparty/fmt/core.h). LowFive's own headers "
    "need it unconditionally. Point -DCMAKE_PREFIX_PATH at a prefix "
    "providing fmt, or install one next to the LowFive include directory.")
endif()
set(LOWFIVE_FMT_INCLUDE_DIR "${_fuse_lowfive_fmt_include}" CACHE PATH
    "Include directory providing fmt/core.h for LowFive's headers")

# The distributed (cross-process) API -- LowFive::DistMetadataVOL,
# lowfive/vol-dist-metadata.hpp -- is compiled into a SEPARATE static archive
# on this lab's two installs (liblowfive-dist.a, next to the main shared
# library), not into liblowfive itself. Measured directly: `nm -gU
# liblowfive.dylib | grep DistMetadataVOL` finds nothing on either
# /opt/lowfive or /opt/lowfive-shared, while `nm -gU liblowfive-dist.a` finds
# 22 symbols including create_DistMetadataVOL/serve_all/broadcast_files. A
# consumer that includes vol-dist-metadata.hpp and does not also link this
# archive gets "symbol(s) not found" at the final link of ITS OWN binary,
# not here -- which is exactly what happened before this search existed
# (fuse_transport_lowfive_test). find_library on a base name
# that already contains a hyphen needs the NAMES form spelled out; PATH_SUFFIXES
# mirrors LOWFIVE_LIBRARY's own search above.
find_library(LOWFIVE_DIST_LIBRARY NAMES lowfive-dist
             HINTS ${LowFive_DIR} ${LowFive_ROOT} ${CMAKE_PREFIX_PATH} /opt/lowfive-shared
             PATH_SUFFIXES lib)
if(LOWFIVE_DIST_LIBRARY)
  set(LOWFIVE_HAVE_DIST ON)
else()
  set(LOWFIVE_HAVE_DIST OFF)
  message(STATUS
    "fuse: LowFive found at ${LOWFIVE_INCLUDE_DIR}, but no liblowfive-dist "
    "next to it; fuse::LowFiveTransport::set_intercomm()/serve_as_producer()/"
    "receive_as_consumer() (distributed/cross-process mode, Demo B) will be "
    "compiled out even in an MPI build -- see docs/user/lowfive.md")
endif()

if(NOT TARGET fuse::lowfive-raw)
  add_library(fuse-lowfive-raw UNKNOWN IMPORTED GLOBAL)
  add_library(fuse::lowfive-raw ALIAS fuse-lowfive-raw)
  set_target_properties(fuse-lowfive-raw PROPERTIES
    IMPORTED_LOCATION "${LOWFIVE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${LOWFIVE_INCLUDE_DIR};${LOWFIVE_FMT_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1")
  if(LOWFIVE_HAVE_DIST)
    target_link_libraries(fuse-lowfive-raw INTERFACE "${LOWFIVE_DIST_LIBRARY}")
  endif()
endif()

mark_as_advanced(LOWFIVE_INCLUDE_DIR LOWFIVE_LIBRARY LOWFIVE_FMT_INCLUDE_DIR
                 LOWFIVE_DIST_LIBRARY)
