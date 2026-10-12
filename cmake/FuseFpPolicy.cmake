# Floating-point contraction policy (KNOWN_ISSUES.md D30, "FpPolicy" track).
#
# Apple clang (and GNU/Clang generally) fuse `a*b + c` into a single fused
# multiply-add BY DEFAULT AT EVERY OPTIMIZATION LEVEL, including -O0 --
# confirmed directly (docs/dev/testing.md's "Floating-point contraction
# policy" section has the instruction counts): -ffp-contract's default
# value is `on` regardless of -O0 vs -O2, so "Debug never contracts" is
# false. What actually differs between a Debug and a Release build of a
# reduction loop is the CODE SHAPE an optimizing build's auto-vectorizer
# introduces: a runtime-trip-count loop gets split into a vectorized main
# body (always-unfused multiply then add, flag or no flag) plus a scalar
# remainder tail (fused by default, unfused under -ffp-contract=off) --
# and a compile-time-sized reference loop gets fully unrolled into scalar
# instructions instead, fused by default. At -O0 neither transformation
# happens, so both sides reduce to the SAME textual sequence (fused,
# ascending) and agree; at -O2/-O3 without the flag they can disagree
# over the vectorized elements. D30 measured the SYMPTOM of this
# correctly (fuse::linalg::DistVector::dot against a plain loop;
# fuse::num::deterministic_dot/deterministic_norm_squared against a
# chunked serial reference differ by one ulp in a Release build and
# nowhere else) but attributed it to the wrong cause.
#
# FUSE_FP_REPRODUCIBLE, default ON (the robustness rule: a build-wide,
# compiler-default-independent choice is preferred unless there is a
# measured reason to prefer the alternative -- docs/dev/testing.md's cost
# table has that measurement): -ffp-contract=off on every target, every
# language, in EVERY build type (including Debug, whose own default-
# contracted arithmetic this also changes -- this is a new numerical
# configuration in Debug too, not a return to its historical bits) --
# makes the three comparisons above hold bit-for-bit by forcing both
# sides of each comparison onto the SAME (unfused) path, not by
# "reproducing Debug." OFF leaves the compiler's default contraction
# policy in place (fused, faster) and the same three comparisons are
# restated as a measured forward-error bound instead of bit-identity
# (test/linalg/dist_vector_test.cpp, test/num/space_kernels_test.cpp).
#
# Declared in THIS file rather than alongside the other FUSE_ENABLE_*
# options in cmake/FuseDependencies.cmake, for the same ordering reason
# CMakeLists.txt's own comment gives for declaring FUSE_ENABLE_FORTRAN
# early: this file is include()'d (see CMakeLists.txt) BEFORE
# FuseFortran.cmake enables the Fortran language and adds fortran/ as a
# subdirectory, and a directory added before an option is declared cannot
# see that option's later effect -- a cache variable's initial value comes
# from whichever option() call reaches it first in a fresh configure.
# FuseDependencies.cmake documents this option's existence, with a pointer
# here, next to FUSE_ENABLE_EXCEPTIONS et al., for a reader scanning that
# file; it does not redeclare it.
option(FUSE_FP_REPRODUCIBLE
       "Reproducible floating point: -ffp-contract=off on every target so a \
Release build reproduces Debug's bits exactly (KNOWN_ISSUES.md D30). OFF \
keeps the compiler's default FMA contraction; the three bit-identity \
comparisons it affects are then checked against a derived forward-error \
bound instead."
       ON)

# Scope: FUSE's own targets only (src/, test/, examples/, tools/,
# fortran/) -- NOT the upstream/soma and upstream/karma imported parent
# trees (invariant I8: an import that had to be patched to build is not
# an import) and NOT the FetchContent-vendored r3d/yaml-cpp sources this
# machine builds from source (cmake/FuseDependencies.cmake): injecting a
# new compiler flag into a vendored build is the same kind of change I8
# forbids even though no file of theirs is edited. Applying the C/CXX
# half of the policy AFTER fuse_resolve_dependencies() (CMakeLists.txt),
# once every FetchContent_MakeAvailable call that vendored build will
# make has already run, keeps it out of their directory scope.
#
# Applied via a function, not inline at each call site, so the compiler-
# ID guard and its message are written once.
#
# Compiler coverage, measured on this machine (2026-10-12): GNU, Clang and
# AppleClang accept -ffp-contract=off identically (`clang --help` lists
# `-ffp-contract=<value>`; confirmed Apple clang 21.0.0 here). No other
# C/C++ compiler is probed -- MSVC and Intel's classic/oneAPI compilers
# use different flags (/fp:precise, -fp-model=precise) that are NOT added
# here because they are untested on this machine; a build configured with
# FUSE_FP_REPRODUCIBLE=ON on such a compiler gets a CMake WARNING instead
# of a silently-assumed flag.
#
# NOT covered, by design, and not guessed at: a CUDA device build's own
# contraction (nvcc's -fmad, default ON, equivalent to -ffp-contract). The
# matching flag is -fmad=false; documented as follow-up work for the
# device branch in docs/dev/testing.md, not added here because this
# machine has no CUDA toolchain to test it against.
function(fuse_apply_fp_contract_policy_c_cxx)
  if(NOT FUSE_FP_REPRODUCIBLE)
    return()
  endif()
  set(_fuse_fp_ok ON)
  foreach(_fuse_fp_lang C CXX)
    if(NOT CMAKE_${_fuse_fp_lang}_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
      set(_fuse_fp_ok OFF)
    endif()
  endforeach()
  if(_fuse_fp_ok)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:C>:-ffp-contract=off>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
    message(STATUS "fuse: FUSE_FP_REPRODUCIBLE is ON; -ffp-contract=off added for C and CXX "
                   "(${CMAKE_C_COMPILER_ID}/${CMAKE_CXX_COMPILER_ID})")
  else()
    message(WARNING "fuse: FUSE_FP_REPRODUCIBLE is ON but the C compiler "
                    "(${CMAKE_C_COMPILER_ID}) or the CXX compiler "
                    "(${CMAKE_CXX_COMPILER_ID}) is neither GNU, Clang nor AppleClang; no "
                    "contraction flag was added for that language, and a Release build may "
                    "still fuse multiply-adds (KNOWN_ISSUES.md D30).")
  endif()
endfunction()

# Fortran's compiler ID is not known until FuseFortran.cmake's own
# enable_language(Fortran) has run (CMAKE_Fortran_COMPILER_ID is undefined
# before that), so this is a SEPARATE function, called from
# FuseFortran.cmake itself right after enable_language(Fortran) and before
# its add_subdirectory(fortran) -- not from the function above, which runs
# (from CMakeLists.txt) after Fortran may already have its own
# subdirectory queued.
function(fuse_apply_fp_contract_policy_fortran)
  if(NOT FUSE_FP_REPRODUCIBLE)
    return()
  endif()
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-ffp-contract=off>)
    message(STATUS "fuse: FUSE_FP_REPRODUCIBLE is ON; -ffp-contract=off added for Fortran (GNU)")
  else()
    message(WARNING "fuse: FUSE_FP_REPRODUCIBLE is ON but the Fortran compiler "
                    "(${CMAKE_Fortran_COMPILER_ID}) is not GNU; no contraction flag was added "
                    "for ffuse (KNOWN_ISSUES.md D30).")
  endif()
endfunction()
