# LowFive is a runtime-loaded HDF5 VOL plugin, so both the library and a
# SHARED HDF5 must be present. A static HDF5 cannot back a VOL plugin: the
# plugin is brought in with dlopen(), and it resolves the H5* symbols it was
# linked against out of a shared libhdf5 at load time. If the only HDF5 on the
# machine is a `.a`, the dlopen fails with "Library not loaded: libhdf5.<N>"
# and there is nothing the caller can do about it at run time. See
# docs/BUILDING.md for the recorded instance of exactly that failure.
#
# Sets, in the including scope:
#   SOMA_HAVE_LOWFIVE   ON only if LowFive is usable as a VOL plugin
#   SomaLowFive_FOUND   mirror of the above, for find_package() callers
#   soma::lowfive       imported target, defined only when usable
#
# Deliberately no return() and no PARENT_SCOPE: a Find module is include()d
# into its caller's scope, so PARENT_SCOPE would write past the top-level
# scope and return() would abandon the rest of the caller's CMakeLists.

set(SOMA_HAVE_LOWFIVE OFF)

find_path(LOWFIVE_INCLUDE_DIR lowfive/vol-base.hpp
          HINTS ${SOMA_LOWFIVE_ROOT} $ENV{SOMA_LOWFIVE_ROOT} ${CMAKE_PREFIX_PATH}
          PATH_SUFFIXES include)
find_library(LOWFIVE_LIBRARY NAMES lowfive
             HINTS ${SOMA_LOWFIVE_ROOT} $ENV{SOMA_LOWFIVE_ROOT} ${CMAKE_PREFIX_PATH}
             PATH_SUFFIXES lib)
find_package(HDF5 QUIET COMPONENTS C HL)

if(NOT LOWFIVE_INCLUDE_DIR OR NOT LOWFIVE_LIBRARY)
  message(STATUS "LowFive: not found; Demo B is unavailable")
else()
  # A .a here means the plugin cannot be dlopened -- catch it at configure time
  # rather than at run time.
  get_filename_component(_lf_ext "${LOWFIVE_LIBRARY}" EXT)
  if(_lf_ext STREQUAL ".a")
    message(WARNING
      "LowFive: found a static archive (${LOWFIVE_LIBRARY}). A VOL plugin must "
      "be shared. See docs/BUILDING.md.")
  else()
    # The extension check above only proves LowFive itself is shared. It says
    # nothing about whether the libhdf5 LowFive was linked against can still be
    # resolved -- which is the failure this module exists to prevent. Actually
    # dlopen the plugin; that is the only check that covers both.
    set(_lf_dlopen_ok TRUE)
    if(NOT CMAKE_CROSSCOMPILING)
      # .cpp, not .c: this project enables only CXX, and try_compile() works
      # only for enabled languages.
      set(_lf_probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/soma_lowfive_dlopen.cpp")
      file(WRITE "${_lf_probe}"
        "#include <dlfcn.h>\n#include <cstdio>\n"
        "int main(){ void*h=dlopen(\"${LOWFIVE_LIBRARY}\",RTLD_NOW);\n"
        "  if(!h){ std::fprintf(stderr,\"%s\\n\",dlerror()); return 1; } return 0; }\n")
      try_run(_lf_run _lf_compiled
              "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/soma_lowfive_dlopen"
              "${_lf_probe}"
              LINK_LIBRARIES ${CMAKE_DL_LIBS}
              RUN_OUTPUT_VARIABLE _lf_run_out)
      if(NOT _lf_compiled OR NOT _lf_run EQUAL 0)
        set(_lf_dlopen_ok FALSE)
        message(WARNING
          "LowFive: ${LOWFIVE_LIBRARY} exists but cannot be dlopened, so it "
          "cannot serve as an HDF5 VOL plugin. This is usually a static or "
          "missing libhdf5. See docs/BUILDING.md.\n${_lf_run_out}")
      endif()
    endif()

    # The transport translation unit includes <hdf5.h> and LowFive's own
    # headers include it too, so a LowFive that cannot be *compiled* against
    # is as useless as one that cannot be dlopened. HDF5 is therefore
    # required from here on, not merely nice to have.
    if(_lf_dlopen_ok AND NOT HDF5_FOUND)
      set(_lf_dlopen_ok FALSE)
      message(WARNING
        "LowFive: found the plugin but no HDF5 development files. Add the "
        "HDF5 prefix to CMAKE_PREFIX_PATH, e.g. "
        "-DCMAKE_PREFIX_PATH=\"/opt/lowfive-shared;/opt/hdf5-shared\".")
    endif()

    # A PARALLEL HDF5's <H5public.h> does `#include <mpi.h>` unconditionally,
    # so every consumer of <hdf5.h> needs MPI's include directory even when it
    # makes no MPI call and never links libmpi. This is not hypothetical: the
    # first compile of src/transport/lowfive_transport.cpp against
    # /opt/hdf5-shared failed with "'mpi.h' file not found" and nothing else.
    # FindHDF5 does not add it, so add it here.
    set(_lf_extra_includes "")
    set(_lf_extra_libs "")
    if(_lf_dlopen_ok AND HDF5_IS_PARALLEL)
      find_package(MPI QUIET COMPONENTS C)
      if(MPI_C_FOUND)
        list(APPEND _lf_extra_includes ${MPI_C_INCLUDE_DIRS})
      else()
        set(_lf_dlopen_ok FALSE)
        message(WARNING
          "LowFive: HDF5 at ${HDF5_INCLUDE_DIRS} is parallel, so <hdf5.h> "
          "includes <mpi.h>, but no MPI was found. Point CMake at one, e.g. "
          "-DMPI_HOME=/opt/mpich.")
      endif()
    elseif(_lf_dlopen_ok)
      message(WARNING "LowFive: HDF5 is not parallel; n->m redistribution will fail")
    endif()

    # Actually compile against the headers. The dlopen probe above proves the
    # plugin loads; this proves a translation unit that uses it builds, which
    # is a different failure (the missing mpi.h, and LowFive's <fmt/core.h>
    # dependency, are both caught only here).
    if(_lf_dlopen_ok AND NOT CMAKE_CROSSCOMPILING)
      set(_lf_hdr_probe
          "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/soma_lowfive_header.cpp")
      file(WRITE "${_lf_hdr_probe}"
        "#include <hdf5.h>\n#include <lowfive/vol-metadata.hpp>\n"
        "#include <lowfive/H5VOLProperty.hpp>\n"
        "int main(){ return H5VLis_connector_registered_by_name(\"lowfive\") < 0; }\n")
      try_compile(_lf_hdr_ok
                  "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/soma_lowfive_header"
                  "${_lf_hdr_probe}"
                  CMAKE_FLAGS
                    "-DINCLUDE_DIRECTORIES=${LOWFIVE_INCLUDE_DIR};${HDF5_INCLUDE_DIRS};${_lf_extra_includes}"
                  LINK_LIBRARIES ${LOWFIVE_LIBRARY} ${HDF5_C_LIBRARIES}
                  CXX_STANDARD 20
                  OUTPUT_VARIABLE _lf_hdr_out)
      if(NOT _lf_hdr_ok)
        set(_lf_dlopen_ok FALSE)
        message(WARNING
          "LowFive: the headers at ${LOWFIVE_INCLUDE_DIR} do not compile "
          "against the HDF5 at ${HDF5_INCLUDE_DIRS}.\n${_lf_hdr_out}")
      endif()
    endif()

    if(_lf_dlopen_ok)
      set(SOMA_HAVE_LOWFIVE ON)
      if(NOT TARGET soma::lowfive)
        add_library(soma::lowfive UNKNOWN IMPORTED)
        set_target_properties(soma::lowfive PROPERTIES
          IMPORTED_LOCATION "${LOWFIVE_LIBRARY}"
          INTERFACE_INCLUDE_DIRECTORIES
            "${LOWFIVE_INCLUDE_DIR};${HDF5_INCLUDE_DIRS};${_lf_extra_includes}"
          INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES};${_lf_extra_libs}")
      endif()
    endif()
  endif()
endif()

set(SomaLowFive_FOUND ${SOMA_HAVE_LOWFIVE})
