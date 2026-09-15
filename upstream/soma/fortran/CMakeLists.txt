# The Fortran shim (spec 6, integration mode 2).
#
# Reached only from `if(SOMA_ENABLE_FORTRAN)` in the top-level CMakeLists,
# which is where enable_language(Fortran) is called and why. SOMA_ENABLE_FORTRAN
# is OFF by default, so a machine without a Fortran compiler never probes for
# one and the dependency-free default build is untouched.

add_library(soma_fortran soma_mod.f90)

# PUBLIC because a Fortran component links soma_fortran and needs soma_core's
# symbols resolved at its own link line. Nothing of soma_core's C++ interface
# leaks: the only thing that crosses is the C ABI.
target_link_libraries(soma_fortran PUBLIC soma_core)

# .mod files are compiler-generated build artefacts, not sources; put them
# somewhere predictable and export that directory so `use soma_mod` resolves
# in consumers. CMake passes include directories to Fortran as -I, which is
# where gfortran, ifx, and flang all look for modules.
set_target_properties(soma_fortran PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/modules)
target_include_directories(soma_fortran PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/modules)

# Warnings, to the same standard the C++ targets are held to. `-std=f2018`
# is the load-bearing one: this module has to compile under gfortran, ifx,
# and flang, and the fastest way to find out that it has drifted into a GNU
# extension is to forbid them here, on the machine where it is developed.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(soma_fortran PRIVATE
    -Wall -Wextra -std=f2018 -pedantic)
endif()
