# The two examples named in S4's exit criteria, ported from the coupler parent.
#
# They are EXAMPLES, and they are also the only code in the tree that drives
# the library the way a user would -- building a graph with the fluent API,
# compiling it into the dataflow graph, and running it -- rather than assembling a fixture. That makes
# them worth running rather than merely compiling.
#
# add_test() here works only because the root CMakeLists.txt adds examples/
# AFTER enable_testing(); added before it, no CTestTestfile.cmake is generated
# for this directory and the examples build but are never run. The parent
# recorded that trap and it applies unchanged.

# review-2026-10-11 BC-1: FUSE_LINALG_BACKEND=builtin configures the library
# with no Eigen3::Eigen target at all (cmake/FuseDependencies.cmake creates
# it only when the backend is eigen). Several example executables below link
# Eigen3::Eigen directly (Gray-Scott, Brusselator, FSI channel -- all via
# Eigen::SparseLU/SimplicialLDLT) or through fuse::examples_common's own
# INTERFACE link, further down in this file; this one ON/OFF variable,
# set once, is what every gate below reads.
if(FUSE_LINALG_BACKEND STREQUAL "eigen")
  set(FUSE_LINALG_BACKEND_IS_EIGEN ON)
else()
  set(FUSE_LINALG_BACKEND_IS_EIGEN OFF)
endif()
set(_fuse_eigen_gated_examples "")

add_executable(fuse_example_mms_pair mms_pair/main.cpp)
target_link_libraries(fuse_example_mms_pair PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_mms_pair PRIVATE mms_pair)

add_executable(fuse_example_dn_heat dn_heat/main.cpp)
target_link_libraries(fuse_example_dn_heat PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_dn_heat PRIVATE dn_heat)
# S7.2b (PetscSelect track): dn_heat/main.cpp's own --newton-petsc path
# (guarded by #if FUSE_HAVE_PETSC, fuse/base/config.hpp) calls
# fuse::petsc::install(), which is defined only in fuse-coupler-petsc.
# Linking it unconditionally would be fine too (the symbol is simply
# unused in a build without FUSE_ENABLE_PETSC), but the target does not
# EXIST at all in that configuration (src/CMakeLists.txt's own
# `if(FUSE_ENABLE_PETSC)` guard around `add_library(fuse-coupler-petsc
# ...)`), so this link must be conditional too.
if(FUSE_ENABLE_PETSC)
  target_link_libraries(fuse_example_dn_heat PRIVATE fuse::petsc)
endif()

# S7.4 (owner decision 2026-10-21, Option C): the smallest real port of an
# ImplicitComponent onto a PETSc-object-backed LinearizationToken (real
# Mat/Vec, MatMult/MatMultTranspose, never MATMFFD), paired with an
# existing native-substrate implementation of the SAME physics in one
# Newton-coupled region -- examples/petsc_reaction_diffusion/. The native
# half of the pair needs nothing from fuse::petsc; the executable always
# links fuse::runtime/fuse::backend, and fuse::petsc only when
# FUSE_ENABLE_PETSC, the SAME conditional fuse_example_dn_heat uses just
# above (reaction_diffusion_component_petsc.hpp is never #included
# otherwise, guarded by #if FUSE_HAVE_PETSC in main.cpp).
add_executable(fuse_example_petsc_reaction_diffusion petsc_reaction_diffusion/main.cpp)
target_link_libraries(fuse_example_petsc_reaction_diffusion PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_petsc_reaction_diffusion PRIVATE petsc_reaction_diffusion)
if(FUSE_ENABLE_PETSC)
  # fuse::petsc's own <petsc.h>/MPI include path is PRIVATE
  # (src/CMakeLists.txt's own comment: "<petsc.h> appears only in this
  # library's own .cpp files, never in a public fuse/petsc/*.hpp") --
  # unlike fuse_example_dn_heat, this example's OWN header
  # (reaction_diffusion_component_petsc.hpp) includes <petsc.h> directly
  # (examples/ is not subject to tools/check-layering.py's PETSc-type-
  # free rule, which applies only to include/fuse/ public headers), so
  # it needs PkgConfig::PETSc/MPI::MPI_C linked here too, not only
  # fuse::petsc.
  target_link_libraries(fuse_example_petsc_reaction_diffusion PRIVATE
    fuse::petsc PkgConfig::PETSc MPI::MPI_C)
endif()

if(FUSE_ENABLE_TESTS)
  # Gated on the PHYSICS, not on "did not crash". mms_pair exits non-zero when
  # the coupled answer misses its analytic gate, when no conservation check
  # ran, or when the coupled error is not decisively better than a run with no
  # exchange. dn_heat exits non-zero when the converged answer misses the
  # analytic two-layer-wall solution, when the implicit run is not decisively
  # better than an explicit one or than a run whose exchange has been severed,
  # when the accelerators disagree about the fixed point, when IQN-ILS fails to
  # beat Picard, when a rollback was never verified, or when two runs of the
  # same configuration differ.
  add_test(NAME example.mms_pair COMMAND fuse_example_mms_pair)
  add_test(NAME example.dn_heat  COMMAND fuse_example_dn_heat)
  # `.device` twins (test/CMakeLists.txt's fuse_add_device_twin): the same
  # command under FUSE_EXECUTION_SPACE=device, registered only when Kokkos
  # has a device backend.
  if(COMMAND fuse_add_device_twin)
    fuse_add_device_twin(example.mms_pair fuse_example_mms_pair LABELS example)
    fuse_add_device_twin(example.dn_heat fuse_example_dn_heat LABELS example)
  endif()

  # The --dump path has its own failure mode: an empty dump still exits 0, so
  # these assert on the output rather than the status.
  add_test(NAME example.mms_pair.dump COMMAND fuse_example_mms_pair --dump)
  add_test(NAME example.dn_heat.dump  COMMAND fuse_example_dn_heat --dump)
  set_tests_properties(example.mms_pair.dump example.dn_heat.dump PROPERTIES
    PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ===")
  set_tests_properties(example.mms_pair example.dn_heat
    example.mms_pair.dump example.dn_heat.dump PROPERTIES LABELS "example")

  # S7.2b (PetscSelect track): the PETSc-substrate rows, gated the same
  # way every other PETSc-dependent ctest row in this tree is -- the
  # target does not exist, and --newton-petsc refuses at runtime with a
  # clear message, in a build without FUSE_ENABLE_PETSC (dn_heat/main.
  # cpp's own #if FUSE_HAVE_PETSC guard), so registering this row
  # unconditionally would register a row that always fails there.
  if(FUSE_ENABLE_PETSC)
    add_test(NAME example.dn_heat.newton_petsc
             COMMAND fuse_example_dn_heat --newton-petsc)
    set_tests_properties(example.dn_heat.newton_petsc PROPERTIES LABELS "example;petsc")
  endif()

  # S7.4: the native-twin leg always runs (no PETSc, no MPI) and is gated
  # on the physics the SAME way dn_heat/mms_pair are -- exits non-zero if
  # the converged answer misses the manufactured solution by more than
  # the track's own derived bound (main.cpp's own kAgreementBound
  # comment). The --petsc leg additionally builds AND RUNS the PETSc-
  # substrate half; registered ONLY when FUSE_ENABLE_PETSC, for the exact
  # reason example.dn_heat.newton_petsc is gated just above, and expected
  # to FAIL in a sandbox that cannot complete a bare MPI_Init()
  # (KNOWN_ISSUES.md D32) -- the SAME already-documented, host-pending
  # status every other petsc.*/example.dn_heat.newton_petsc row carries,
  # not a new one this track introduces.
  add_test(NAME example.petsc_reaction_diffusion
           COMMAND fuse_example_petsc_reaction_diffusion)
  set_tests_properties(example.petsc_reaction_diffusion PROPERTIES LABELS "example")
  if(FUSE_ENABLE_PETSC)
    add_test(NAME example.petsc_reaction_diffusion.petsc
             COMMAND fuse_example_petsc_reaction_diffusion --petsc)
    set_tests_properties(example.petsc_reaction_diffusion.petsc PROPERTIES LABELS "example;petsc")
  endif()

  # Thread-count determinism of the two physics examples, bit for bit: each is
  # run at one OpenMP thread and at every core and the two outputs compared
  # exactly (tools/compare-runs.cmake). Since D11 closed (2026-09-28) the
  # Runner's own hot loop runs on Kokkos' host execution space with a
  # fixed-order reduction scheme, so these legs are load-bearing: they are
  # what the components' `DeterminismLevel::Bitwise` declarations rest on.
  # (Until the same day, test/CMakeLists.txt's FUSE_KOKKOS_HAS_OPENMP was not
  # exported to this sibling directory and every leg here silently registered
  # nothing.)
  if(COMMAND fuse_add_omp_legs AND FUSE_KOKKOS_HAS_OPENMP)
    foreach(ex mms_pair dn_heat)
      add_test(NAME example.${ex}.omp_identical
               COMMAND ${CMAKE_COMMAND}
                       -DBIN=$<TARGET_FILE:fuse_example_${ex}>
                       -DTHREADS_HI=${_fuse_host_cores}
                       -DOUT=${CMAKE_CURRENT_BINARY_DIR}/omp_${ex}
                       -P ${CMAKE_SOURCE_DIR}/tools/compare-runs.cmake)
      set_tests_properties(example.${ex}.omp_identical PROPERTIES LABELS "example;omp")
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# processor-processor-layout driver (plan-track-d). MPI-only: it links fuse::transport for
# MpiTransport and needs mpiexec to mean anything, so the whole block is
# gated on FUSE_ENABLE_MPI, exactly as test/CMakeLists.txt gates its own
# parallel/ targets.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Shared by the two examples whose cost is repeated sparse LU (Gray-Scott,
# FSI channel) and by their verification tests: examples/common/
# sparse_solve_backend.hpp, a SparseLuBackend that is Eigen::SparseLU by
# default and cusolverRf on the device under FUSE_ENABLE_CUSPARSE. INTERFACE
# only -- a header and, when the flag is on, the three CUDA libraries it
# needs. test/CMakeLists.txt links it by its namespaced name; CMake resolves
# that at generate time, so test/ being added before examples/ is fine.
# ---------------------------------------------------------------------------
add_library(fuse_examples_common INTERFACE)
add_library(fuse::examples_common ALIAS fuse_examples_common)
target_include_directories(fuse_examples_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)
target_link_libraries(fuse_examples_common INTERFACE fuse::base)
# review-2026-10-11 BC-1: fuse::examples_common is an INTERFACE target, so
# this one unguarded Eigen3::Eigen link used to fail every consumer's
# configure at once under FUSE_LINALG_BACKEND=builtin (test/CMakeLists.txt's
# own comment on this target makes the same observation about its reach).
if(FUSE_LINALG_BACKEND_IS_EIGEN)
  target_link_libraries(fuse_examples_common INTERFACE Eigen3::Eigen)
endif()
if(FUSE_ENABLE_CUSPARSE)
  target_link_libraries(fuse_examples_common INTERFACE
    CUDA::cusolver CUDA::cusparse CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# review-2026-10-11 BC-1: gated on the eigen backend -- this example's
# Newton solves use Eigen::SparseLU and have no builtin-backend substitute.
if(FUSE_LINALG_BACKEND_IS_EIGEN)
  # Gray-Scott (Time windows track): U and V on one structured grid, coupled
  # implicitly at independent time steps through fuse::WindowCoupler. Needs
  # Eigen3::Eigen directly (its Newton solves use Eigen::SparseLU) -- available
  # because FUSE_LINALG_BACKEND=eigen is the default (cmake/FuseDependencies.cmake).
  # ---------------------------------------------------------------------------
  add_executable(fuse_example_gray_scott
    gray_scott/main.cpp)
  target_link_libraries(fuse_example_gray_scott PRIVATE
    fuse::runtime fuse::backend Eigen3::Eigen fuse::examples_common)
  target_include_directories(fuse_example_gray_scott PRIVATE gray_scott)

  if(FUSE_ENABLE_TESTS)
    # Gated on the physics, same convention as mms_pair/dn_heat above:
    # --check exits non-zero if the m=1 windowed result misses the monolithic
    # reference (to a TIGHT, iteration-tolerance-level bound -- see the
    # comment at that check in main.cpp) or a concentration goes negative.
    # --dt 0.1 is not incidental: the m=1 windowed/monolithic mismatch is a
    # genuine order-2-in-dt effect (gray_scott_reference_test.cpp measures it
    # exactly), not merely the iteration residual, so it is checked at a dt
    # small enough for that order-2 term to already be below the tight bound,
    # rather than by loosening the bound. The --check block's own m=1
    # sub-run is internally fixed at 5 windows of --dt regardless of --ratio/
    # --t-end (main.cpp), so --dt is what this bound actually depends on;
    # measured AT THIS EXACT COMMAND LINE: diff 3.022e-08, bound 1.0e-6.
    add_test(NAME example.gray_scott COMMAND fuse_example_gray_scott --n 24 --ratio 2 --t-end 4 --dt 0.1 --check)
    set_tests_properties(example.gray_scott PROPERTIES LABELS "example")
    # An unknown --accel name is refused before any setup (examples/common/
    # accel_names.hpp). Until 2026-10-20 this program ran Aitken for it.
    add_test(NAME example.gray_scott.accel_refused
             COMMAND fuse_example_gray_scott --accel no-such-method)
    set_tests_properties(example.gray_scott.accel_refused PROPERTIES
      LABELS "example" PASS_REGULAR_EXPRESSION "E-EXAMPLE-ACCEL: unknown --accel 'no-such-method'")
    if(COMMAND fuse_add_device_twin)
      fuse_add_device_twin(example.gray_scott fuse_example_gray_scott LABELS example
                           ARGS --n 24 --ratio 2 --t-end 4 --dt 0.1 --check)
    endif()
    # Same command line, compared bit for bit at one thread and at every core;
    # the species components declare DeterminismLevel::Bitwise on this measurement.
    if(COMMAND fuse_add_omp_legs AND FUSE_KOKKOS_HAS_OPENMP)
      add_test(NAME example.gray_scott.omp_identical
               COMMAND ${CMAKE_COMMAND}
                       -DBIN=$<TARGET_FILE:fuse_example_gray_scott>
                       "-DARGS=--n;24;--ratio;2;--t-end;4;--dt;0.1;--check"
                       -DTHREADS_HI=${_fuse_host_cores}
                       -DOUT=${CMAKE_CURRENT_BINARY_DIR}/omp_gray_scott
                       -P ${CMAKE_SOURCE_DIR}/tools/compare-runs.cmake)
      set_tests_properties(example.gray_scott.omp_identical PROPERTIES LABELS "example;omp")
    endif()
  endif()
else()
  message(STATUS "fuse: fuse_example_gray_scott not built under "
                 "FUSE_LINALG_BACKEND=builtin (needs Eigen::SparseLU)")
  list(APPEND _fuse_eigen_gated_examples fuse_example_gray_scott)
endif()

# ---------------------------------------------------------------------------
# review-2026-10-11 BC-1: gated on the eigen backend -- this example's
# periodic-diffusion backward-Euler solve uses Eigen::SimplicialLDLT and
# has no builtin-backend substitute.
if(FUSE_LINALG_BACKEND_IS_EIGEN)
  # Brusselator (Time windows track, "a code FUSE has never met"): U and V on
  # two INDEPENDENT, non-nested native meshes ("generated:quad:NxM"), coupled
  # at independent time steps through fuse::WindowCoupler and exchanged
  # through FUSE's conservative supermesh remap (fuse::build_weights /
  # fuse::apply_weights) -- ported from the user's own independently-written
  # multirate Brusselator (/Users/mahadevan/Code/AI/Toy). Needs Eigen3::Eigen
  # directly (periodic-diffusion backward-Euler solves, Eigen::SimplicialLDLT)
  # and Kokkos::kokkos directly (the reaction kernel, brusselator_physics.hpp
  # `reaction_kokkos`) -- both available unconditionally, same reasoning as
  # fuse_example_gray_scott just above.
  # ---------------------------------------------------------------------------
  add_executable(fuse_example_brusselator
    brusselator/main.cpp)
  target_link_libraries(fuse_example_brusselator PRIVATE
    fuse::runtime fuse::backend Eigen3::Eigen Kokkos::kokkos
    fuse::examples_common)
  target_include_directories(fuse_example_brusselator PRIVATE brusselator)
  # S7.2b (PetscSelect track): main.cpp's own unconditional (guarded by #if
  # FUSE_HAVE_PETSC) fuse::petsc::install() call -- same reasoning as
  # fuse_example_dn_heat's own identical conditional link, just above in
  # this file.
  if(FUSE_ENABLE_PETSC)
    target_link_libraries(fuse_example_brusselator PRIVATE fuse::petsc)
  endif()

  if(FUSE_ENABLE_TESTS)
    # Gated on the physics, same convention as the examples above: --check
    # exits non-zero unless every window converged, both species stayed
    # non-negative, the discrete mass budget (NUMERICS.md "Discrete budget")
    # held per-window and globally, and the U<->V exchange stayed conservative
    # -- see main.cpp's --check block and docs/user/brusselator.md for the
    # measured thresholds. A short run (--end 0.5) so the example itself
    # stays fast; the deep verification (weights-vs-Toy-oracle, invariants,
    # cross-code agreement, convergence order) lives in
    # test/verification/brusselator_*_test.cpp.
    add_test(NAME example.brusselator COMMAND fuse_example_brusselator --end 0.5 --check)
    set_tests_properties(example.brusselator PROPERTIES LABELS "example")
    # An unknown --accel name is refused before any setup. Until 2026-10-20
    # this program ran with no acceleration for it.
    add_test(NAME example.brusselator.accel_refused
             COMMAND fuse_example_brusselator --accel no-such-method)
    set_tests_properties(example.brusselator.accel_refused PROPERTIES
      LABELS "example" PASS_REGULAR_EXPRESSION "E-EXAMPLE-ACCEL: unknown --accel 'no-such-method'")

    # Time windows track (Conformance, Unit 1): `--driver runner` (main.cpp)
    # runs the SAME graph through fuse::Runner instead of fuse::WindowCoupler
    # (docs/user/coupling-windows.md "Through schedule() and the Runner"), at
    # this example's own default parameters (--window 0.02, --dt-u 0.002,
    # --dt-v 0.003) -- the same span example.brusselator above covers, so a
    # --check failure here means the Runner and the driver disagree on THIS
    # example, not a special-cased schedule. Four `fuse::Runner` defects
    # (test/verification/brusselator_runner_test.cpp's own header) stood
    # between this row and passing; all four are fixed as of this row's own
    # registration, each regression-tested there against the full existing
    # window_runner_* suite.
    add_test(NAME example.brusselator.runner
             COMMAND fuse_example_brusselator --driver runner --end 0.5 --check)
    set_tests_properties(example.brusselator.runner PROPERTIES LABELS "example")
    if(COMMAND fuse_add_device_twin)
      fuse_add_device_twin(example.brusselator.runner fuse_example_brusselator LABELS example
                           ARGS --driver runner --end 0.5 --check)
    endif()

    # The coupling window by Newton on BOTH substrates, through the example
    # itself rather than the one-window test rig (test/petsc/brusselator_
    # newton_test.cpp): the Eisenstat-Walker floor the window needs
    # (docs/dev/newton-krylov.md Sec. 9.9.2, 10.3) shows only on the
    # example's own first window -- the rig's window passed without it on
    # the host, the example's first window reported DivergedLinearSolve
    # (2026-10-11). Five windows, --check on.
    add_test(NAME example.brusselator.newton
             COMMAND fuse_example_brusselator --driver runner --accel newton --end 0.1 --check)
    set_tests_properties(example.brusselator.newton PROPERTIES LABELS "example")
    if(FUSE_ENABLE_PETSC)
      add_test(NAME example.brusselator.newton_petsc
               COMMAND fuse_example_brusselator --driver runner --accel newton --substrate petsc
                       --end 0.1 --check)
      set_tests_properties(example.brusselator.newton_petsc PROPERTIES LABELS "example;petsc")
    endif()

    # Thread-count determinism of the Kokkos reaction kernel, bit for bit --
    # the same convention mms_pair/dn_heat use (tools/compare-runs.cmake),
    # and, since D11 closed, load-bearing for every example: `reaction_kokkos`
    # is a genuine Kokkos::parallel_for over the OpenMP execution space on top
    # of the Runner's own kernels.
    if(COMMAND fuse_add_omp_legs AND FUSE_KOKKOS_HAS_OPENMP)
      add_test(NAME example.brusselator.omp_identical
               COMMAND ${CMAKE_COMMAND}
                       -DBIN=$<TARGET_FILE:fuse_example_brusselator>
                       -DTHREADS_HI=${_fuse_host_cores}
                       -DOUT=${CMAKE_CURRENT_BINARY_DIR}/omp_brusselator
                       -P ${CMAKE_SOURCE_DIR}/tools/compare-runs.cmake)
      set_tests_properties(example.brusselator.omp_identical PROPERTIES LABELS "example;omp")
    endif()
  endif()
else()
  message(STATUS "fuse: fuse_example_brusselator not built under "
                 "FUSE_LINALG_BACKEND=builtin (needs Eigen::SimplicialLDLT)")
  list(APPEND _fuse_eigen_gated_examples fuse_example_brusselator)
endif()

# Conformance, Unit 2 ("Brusselator as separately built participants"):
# graph AUTHORING, kept deliberately separate from PARTICIPATING (the
# lead's own Q2 ruling) -- this tool calls brusselator_graph.hpp's own
# add_window_coupling() and writes the result as text YAML
# (fuse::to_yaml(), include/fuse/model/yaml_io.hpp), so u_main.c/
# v_main.f90 (examples/brusselator/participants, built only under
# FUSE_ENABLE_MPI below) never construct a fuse::CouplingGraph
# themselves -- they call fuse_load_graph() on this tool's own output
# and nothing else. Needs no MPI itself (it only builds a graph object
# and writes text), so built unconditionally, same as
# fuse_example_brusselator above; the
# verification.brusselator.graph_tool.reproduces_reference row
# registered just below proves it reproduces the one committed
# reference YAML (examples/brusselator/participants/
# reference_mpmd_3.yaml) byte for byte.
add_executable(brusselator_graph_tool
  brusselator/participants/brusselator_graph_tool.cpp)
target_link_libraries(brusselator_graph_tool PRIVATE fuse::model)
target_include_directories(brusselator_graph_tool PRIVATE brusselator)

if(FUSE_ENABLE_TESTS)
  # --ranks-u 1 --ranks-v 1 --ranks-coupler 1 --launch mpmd is the ONE
  # committed reference layout (examples/brusselator/participants/
  # reference_mpmd_3.yaml) -- the documented example of what a user
  # writes by hand for the {3}-rank mpmd ctest row.
  add_test(NAME verification.brusselator.graph_tool.reproduces_reference
           COMMAND ${CMAKE_COMMAND}
                   -DBIN=$<TARGET_FILE:brusselator_graph_tool>
                   "-DARGS=--ranks-u;1;--ranks-v;1;--ranks-coupler;1;--launch;mpmd"
                   -DOUT=${CMAKE_CURRENT_BINARY_DIR}/graph_tool_reproduced_mpmd_3.yaml
                   -DREFERENCE=${CMAKE_SOURCE_DIR}/examples/brusselator/participants/reference_mpmd_3.yaml
                   -P ${CMAKE_SOURCE_DIR}/tools/verify-graph-tool-reference.cmake)
  set_tests_properties(verification.brusselator.graph_tool.reproduces_reference
                       PROPERTIES LABELS "verification")

  # S6 phase 3 ("LowFive" track), Unit 3: the SAME {3}-rank mpmd layout,
  # plus --transport lowfive -- byte-for-byte against examples/
  # brusselator/participants/reference_mpmd_3_lowfive.yaml, exactly the
  # same convention as the row above. Needs no MPI, no LowFive and no
  # FUSE_HAVE_LOWFIVE: the graph tool only ever writes text, regardless
  # of which transport the text describes.
  add_test(NAME verification.brusselator.graph_tool.reproduces_reference_lowfive
           COMMAND ${CMAKE_COMMAND}
                   -DBIN=$<TARGET_FILE:brusselator_graph_tool>
                   "-DARGS=--ranks-u;1;--ranks-v;1;--ranks-coupler;1;--launch;mpmd;--transport;lowfive"
                   -DOUT=${CMAKE_CURRENT_BINARY_DIR}/graph_tool_reproduced_mpmd_3_lowfive.yaml
                   -DREFERENCE=${CMAKE_SOURCE_DIR}/examples/brusselator/participants/reference_mpmd_3_lowfive.yaml
                   -P ${CMAKE_SOURCE_DIR}/tools/verify-graph-tool-reference.cmake)
  set_tests_properties(verification.brusselator.graph_tool.reproduces_reference_lowfive
                       PROPERTIES LABELS "verification")

  # KNOWN_ISSUES.md D36: the accept-mode layout had NO committed reference
  # at all before this fix (confirmed by listing this directory -- only
  # the two mpmd rows above ever had one), so nothing caught a regression
  # in brusselator_graph_tool.cpp's own accept branch the way the mpmd
  # rows above do for theirs. Added now, alongside the fix, at both world
  # sizes the tool already supports: --ranks-u 1 --ranks-v 1
  # --ranks-coupler 1 (world 3, the row that timed out on the host) and
  # --ranks-u 2 --ranks-v 2 --ranks-coupler 1 (world 5, never run --
  # KNOWN_ISSUES.md D18's own "build it but never run it" ruling for the
  # 5-rank accept row, unchanged by this track).
  add_test(NAME verification.brusselator.graph_tool.reproduces_reference_accept_3
           COMMAND ${CMAKE_COMMAND}
                   -DBIN=$<TARGET_FILE:brusselator_graph_tool>
                   "-DARGS=--ranks-u;1;--ranks-v;1;--ranks-coupler;1;--launch;accept"
                   -DOUT=${CMAKE_CURRENT_BINARY_DIR}/graph_tool_reproduced_accept_3.yaml
                   -DREFERENCE=${CMAKE_SOURCE_DIR}/examples/brusselator/participants/reference_accept_3.yaml
                   -P ${CMAKE_SOURCE_DIR}/tools/verify-graph-tool-reference.cmake)
  set_tests_properties(verification.brusselator.graph_tool.reproduces_reference_accept_3
                       PROPERTIES LABELS "verification")

  add_test(NAME verification.brusselator.graph_tool.reproduces_reference_accept_5
           COMMAND ${CMAKE_COMMAND}
                   -DBIN=$<TARGET_FILE:brusselator_graph_tool>
                   "-DARGS=--ranks-u;2;--ranks-v;2;--ranks-coupler;1;--launch;accept"
                   -DOUT=${CMAKE_CURRENT_BINARY_DIR}/graph_tool_reproduced_accept_5.yaml
                   -DREFERENCE=${CMAKE_SOURCE_DIR}/examples/brusselator/participants/reference_accept_5.yaml
                   -P ${CMAKE_SOURCE_DIR}/tools/verify-graph-tool-reference.cmake)
  set_tests_properties(verification.brusselator.graph_tool.reproduces_reference_accept_5
                       PROPERTIES LABELS "verification")

  # The SAME {3}-rank accept layout, plus --transport lowfive -- exactly the
  # combination KNOWN_ISSUES.md D36's own second defect (the LowFive
  # skip-list gap) was measured against, mirrored here the way
  # reproduces_reference_lowfive mirrors reproduces_reference for mpmd.
  add_test(NAME verification.brusselator.graph_tool.reproduces_reference_accept_3_lowfive
           COMMAND ${CMAKE_COMMAND}
                   -DBIN=$<TARGET_FILE:brusselator_graph_tool>
                   "-DARGS=--ranks-u;1;--ranks-v;1;--ranks-coupler;1;--launch;accept;--transport;lowfive"
                   -DOUT=${CMAKE_CURRENT_BINARY_DIR}/graph_tool_reproduced_accept_3_lowfive.yaml
                   -DREFERENCE=${CMAKE_SOURCE_DIR}/examples/brusselator/participants/reference_accept_3_lowfive.yaml
                   -P ${CMAKE_SOURCE_DIR}/tools/verify-graph-tool-reference.cmake)
  set_tests_properties(verification.brusselator.graph_tool.reproduces_reference_accept_3_lowfive
                       PROPERTIES LABELS "verification")
endif()

# Conformance, Unit 2 (Q3): the --check assembler, run against u_main.c's
# and v_main.f90's own printed per-window logs after a real host run
# (ctest rows registered in test/CMakeLists.txt pipe both executables'
# logs into this tool as their last step). Needs no MPI -- it is pure
# text parsing (brusselator_check_core.hpp's own log-format comment) --
# so, like brusselator_graph_tool above, built unconditionally; its own
# tier-0 test (test/verification/brusselator_check_test.cpp) exercises
# brusselator_check_core's logic directly, without a subprocess.
add_executable(brusselator_check
  brusselator/participants/brusselator_check.cpp
  brusselator/participants/brusselator_check_core.cpp)

# The three participant/coupler executables themselves (Conformance,
# Unit 2) -- tier 1 (FUSE_ENABLE_MPI) only, since each genuinely needs
# MPI (u_main.c's/coupler_main.c's own MPI_Init/MPI_Comm_c2f,
# v_main.f90's own `use mpi`), unlike the two tools above. Each is PURE
# C (u_main.c, coupler_main.c, u_loop.c) or PURE FORTRAN (v_main.f90,
# v_loop.f90) in its own coupling-facing code -- only the shared physics
# (shim.cpp) is C++ (u_main.c's own header comment has the full
# rationale). Never run in THIS sandbox (KNOWN_ISSUES.md C1: MPI cannot
# start here at all) -- build-verified only; test/CMakeLists.txt's own
# verification.brusselator.participants.{mpmd,accept}.{3,5} rows below
# register the real host invocations.
if(FUSE_ENABLE_MPI)
  add_executable(brusselator_coupler_main
    brusselator/participants/coupler_main.c)
  target_link_libraries(brusselator_coupler_main PRIVATE fuse::abi MPI::MPI_C)
  # ---- D14 Stage B (KNOWN_ISSUES.md), additive: begin --------------------
  # brusselator_coupler_main now ALSO joins the single collective split
  # bruss_lowfive_join_local_split() performs (lowfive_attach.cpp, compiles
  # to a trivial, always-local_comm_f=0 stub when FUSE_HAVE_LOWFIVE is 0,
  # exactly like u_main.c's/v_main.f90's own additive hunks below) -- never
  # bruss_lowfive_attach() itself (round 9's own ruling, unchanged: this
  # target still never constructs a real LowFive VOL object). Needs
  # fuse::model (fuse::build_lowfive_bindings(), fuse::load_yaml_file())
  # AND, corrected here after a tier-1 (-DFUSE_ENABLE_LOWFIVE=ON) build
  # caught the gap this comment's own earlier claim missed:
  # lowfive_attach.cpp's own top-of-file #include block (outside any
  # FUSE_HAVE_LOWFIVE guard, so EVERY target compiling this source needs
  # it) names <lowfive/H5VOLProperty.hpp> directly -- a HEADER search
  # path only `fuse::lowfive` provides, which KNOWN_ISSUES.md D39 Round
  # 10's own finding (liblowfive/liblowfive-dist already linked
  # transitively via fuse::abi -> fuse-coupler-transport) does NOT: a
  # transitive LINK dependency carries no transitive include directory
  # in this build graph. Mirrors brusselator_u_main's own identical
  # fuse_link_lowfive_with_fmt_fix() call below, same reason (shim.cpp's
  # own header comment on that function has the full fmt-version
  # account).
  target_sources(brusselator_coupler_main PRIVATE brusselator/participants/lowfive_attach.cpp)
  target_link_libraries(brusselator_coupler_main PRIVATE fuse::model)
  if(TARGET fuse::lowfive)
    fuse_link_lowfive_with_fmt_fix(brusselator_coupler_main)
  endif()
  if(LOWFIVE_HAVE_DIST)
    target_compile_definitions(brusselator_coupler_main PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
  endif()
  # ---- D14 Stage B: end ---------------------------------------------------

  # review-2026-10-11 BC-1: brusselator_u_main/brusselator_v_main link
  # Eigen3::Eigen directly (shim.cpp's physics), so both -- and the two
  # host rows below that are already conditioned on
  # "TARGET brusselator_u_main AND TARGET brusselator_v_main" -- are gated
  # on the eigen backend. brusselator_coupler_main above needs no linalg
  # backend and stays unconditional.
  if(FUSE_LINALG_BACKEND_IS_EIGEN)
    add_executable(brusselator_u_main
      brusselator/participants/u_main.c
      brusselator/participants/u_loop.c
      brusselator/participants/shim.cpp)
    target_link_libraries(brusselator_u_main PRIVATE
      fuse::abi MPI::MPI_C Eigen3::Eigen Kokkos::kokkos)
    target_include_directories(brusselator_u_main PRIVATE brusselator)

    # S6 phase 3 (Conformance, "LowFive" track), Unit 3b: shim.cpp's own
    # bruss_lowfive_write/read (behind FUSE_HAVE_LOWFIVE) call the real
    # HDF5 C API -- fuse::lowfive (cmake/FuseDependencies.cmake, which this
    # directory's own fuse_resolve_dependencies() call already ran before
    # add_subdirectory(examples) reached here, CMakeLists.txt) carries
    # HDF5 plus the LowFive libraries themselves. This unit first attached the
    # connector through the HDF5_VOL_CONNECTOR/HDF5_PLUGIN_PATH env-var
    # mechanism (docs/user/lowfive.md), which needed no LowFive SYMBOL in THIS
    # executable at all -- only ordinary HDF5 calls (the participants now attach
    # programmatically, see the note after the Unit 3b rows) -- but linking the one
    # `fuse::lowfive` target already resolved is simpler than teasing the
    # raw HDF5-only half back out, and costs nothing an ordinary HDF5
    # dependency does not already cost.
    # fuse_link_lowfive_with_fmt_fix (cmake/FuseDependencies.cmake): plain
    # target_link_libraries(... fuse::lowfive ...) here is not enough --
    # shim.cpp also includes fuse/base/rational.hpp (-> result.hpp -> fmt
    # 12), in the SAME translation unit as <hdf5.h>, and fuse::lowfive's
    # own include directory carries a bundled, incompatible fmt. See that
    # function's own comment and KNOWN_ISSUES.md D14's Unit 3b account for
    # the full measured record (an earlier, single-link-order version of
    # this fix broke under a different CMAKE_PREFIX_PATH ordering).
    if(TARGET fuse::lowfive)
      fuse_link_lowfive_with_fmt_fix(brusselator_u_main)
    endif()

    # ---- S6 phase 3 (Conformance, "LowFive attach" track): begin --------
    # lowfive_attach.cpp (examples/brusselator/participants) constructs the
    # VOL connector PROGRAMMATICALLY instead of through the HDF5_VOL_
    # CONNECTOR/HDF5_PLUGIN_PATH env-var route -- see that file's own
    # header comment and KNOWN_ISSUES.md D39 for why. Always added to this
    # executable's own sources (like shim.cpp, it compiles to a trivial
    # "not built with LowFive" stub when FUSE_HAVE_LOWFIVE is 0), needing
    # fuse::model (fuse/model/yaml_io.hpp, CouplingGraph) in addition to
    # shim.cpp's own fuse::lowfive -- a dependency this executable did not
    # need before this track, since shim.cpp's own bruss_lowfive_write/
    # read never parsed the graph themselves.
    target_sources(brusselator_u_main PRIVATE brusselator/participants/lowfive_attach.cpp)
    target_link_libraries(brusselator_u_main PRIVATE fuse::model)
    # FUSE_LOWFIVE_DIST_LIB_FOUND: this executable's own copy of the SAME
    # compile definition src/CMakeLists.txt's fuse-lowfive-vol-graph/-impl
    # OBJECT libraries already receive -- PRIVATE compile definitions do
    # not propagate from a linked library to a consumer, so lowfive_
    # attach.cpp needs its own.
    if(LOWFIVE_HAVE_DIST)
      target_compile_definitions(brusselator_u_main PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
    endif()
    # ---- "LowFive attach" track: end (brusselator_u_main half) ----------

    if(TARGET fuse-abi-fortran)
      add_executable(brusselator_v_main
        brusselator/participants/v_main.f90
        brusselator/participants/v_loop.f90
        brusselator/participants/shim.cpp)
      target_link_libraries(brusselator_v_main PRIVATE
        fuse::abi fuse::abi-fortran MPI::MPI_Fortran MPI::MPI_C Eigen3::Eigen Kokkos::kokkos)
      target_include_directories(brusselator_v_main PRIVATE brusselator)
      if(TARGET fuse::lowfive)
        fuse_link_lowfive_with_fmt_fix(brusselator_v_main)
      endif()
      # ---- "LowFive attach" track: begin (brusselator_v_main half) ------
      # Mirrors brusselator_u_main's own additive hunk just above exactly.
      target_sources(brusselator_v_main PRIVATE brusselator/participants/lowfive_attach.cpp)
      target_link_libraries(brusselator_v_main PRIVATE fuse::model)
      if(LOWFIVE_HAVE_DIST)
        target_compile_definitions(brusselator_v_main PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
      endif()
      # ---- "LowFive attach" track: end -----------------------------------
    else()
      message(STATUS "fuse: FUSE_ENABLE_MPI is ON but no Fortran compiler was found; "
                     "brusselator_v_main (Conformance Unit 2) will not be built")
    endif()
  else()
    message(STATUS "fuse: brusselator_u_main/brusselator_v_main (Conformance Unit 2) not "
                   "built under FUSE_LINALG_BACKEND=builtin (shim.cpp needs Eigen3::Eigen)")
    list(APPEND _fuse_eigen_gated_examples brusselator_u_main brusselator_v_main)
  endif()
endif()


# ---- Conformance Unit 2, "Brusselator as separately built participants":
# host rows begin -----------------------------------------------------------
# MPI cannot start in this sandbox at all (KNOWN_ISSUES.md C1), so --
# exactly like test/CMakeLists.txt's own parallel.connect.* rows, whose
# host_verification_needed convention this mirrors -- every row below is
# a host_verification_needed item; what IS verified here is that every
# executable involved builds at tier 1 (confirmed this session) and that
# brusselator_graph_tool reproduces its own committed reference YAML byte
# for byte (verification.brusselator.graph_tool.reproduces_reference,
# registered just above). Registered HERE, not in test/CMakeLists.txt,
# because add_subdirectory(test) runs before add_subdirectory(examples)
# at the top level (CMakeLists.txt) -- a TARGET check against these
# executables in test/CMakeLists.txt would always be false (found this
# session: the rows silently failed to register there).
#
# {3,5} name each row's total world size. mpmd: ranks_u + ranks_v +
# ranks_coupler = 3 (1+1+1) or 5 (2+2+1), three app segments. accept:
# U's own executable carries ranks_u + ranks_coupler (2 or 3), V's
# carries ranks_v (1 or 2) -- Q1's ruling, KNOWN_ISSUES.md D18.
if(FUSE_ENABLE_TESTS AND FUSE_ENABLE_MPI AND TARGET brusselator_u_main AND TARGET brusselator_v_main)
  set(_fuse_bp_world_3_u 1)
  set(_fuse_bp_world_3_v 1)
  set(_fuse_bp_world_3_c 1)
  set(_fuse_bp_world_5_u 2)
  set(_fuse_bp_world_5_v 2)
  set(_fuse_bp_world_5_c 1)
  foreach(_world 3 5)
    foreach(_mode mpmd accept)
    # The accept/mpmd rows exist only where the MPI can carry them. Skipping
    # rather than registering-and-failing: a row that cannot pass on this
    # installation is not a measurement, and a permanently red row teaches a
    # reader to ignore the colour. cmake/FuseDependencies.cmake decides
    # FUSE_ENABLE_CONNECT_ACCEPT/FUSE_ENABLE_CONNECT_MPMD by measurement, and
    # leaves each ON when it could not test, so nothing is hidden on a
    # machine that was never probed.
    if(_mode STREQUAL "accept" AND NOT FUSE_ENABLE_CONNECT_ACCEPT)
      continue()
    endif()
    if(_mode STREQUAL "mpmd" AND NOT FUSE_ENABLE_CONNECT_MPMD)
      continue()
    endif()
      set(_name verification.brusselator.participants.${_mode}.${_world})
      set(_scratch ${CMAKE_CURRENT_BINARY_DIR}/brusselator-participants-scratch/${_mode}-${_world})
      add_test(NAME ${_name}
        COMMAND bash ${CMAKE_SOURCE_DIR}/test/parallel/brusselator_participants/run_participants.sh
                ${_mode} ${MPIEXEC_EXECUTABLE}
                ${_fuse_bp_world_${_world}_u} ${_fuse_bp_world_${_world}_v} ${_fuse_bp_world_${_world}_c}
                $<TARGET_FILE:brusselator_graph_tool>
                $<TARGET_FILE:brusselator_u_main>
                $<TARGET_FILE:brusselator_v_main>
                $<TARGET_FILE:brusselator_coupler_main>
                $<TARGET_FILE:brusselator_check>
                ${_scratch} 2 60)
      # Same OMPI alltoallv-algorithm workaround as test/CMakeLists.txt's own
      # fuse_add_mpi_test ENVIRONMENT line -- see that comment for the full
      # measured account. The Runner traffic driving this run goes over an
      # MPI_Comm_dup'd communicator exactly like MpiTransport's own tests do.
      set_tests_properties(${_name} PROPERTIES
        LABELS "parallel;mpi;verification;brusselator" TIMEOUT 120
        ENVIRONMENT "OMPI_MCA_coll_tuned_use_dynamic_rules=1;OMPI_MCA_coll_tuned_alltoallv_algorithm=2;OMPI_MCA_rmaps_base_oversubscribe=1")
      # KNOWN_ISSUES.md D23's final account: U bundles the coupler's own
      # ranks and is the Connect-mode acceptor (brusselator_graph_tool.cpp's
      # own accept branch), so this row's real layout is (ranks_u +
      # ranks_coupler) accepting against ranks_v connecting -- not
      # necessarily the same rank counts the pair rows above measured, and
      # checked against the SAME per-layout probe record they read. MPMD is
      # never gated by this (cmake/FuseMpiDynamic.cmake only probes
      # accept/connect).
      if(_mode STREQUAL "accept")
        math(EXPR _fuse_bp_na "${_fuse_bp_world_${_world}_u} + ${_fuse_bp_world_${_world}_c}")
        set(_fuse_bp_nb "${_fuse_bp_world_${_world}_v}")
        fuse_mpi_dynamic_layout_record("${FUSE_MPI_DYNAMIC_LAYOUTS}" "${_fuse_bp_na}+${_fuse_bp_nb}" _fuse_row_record)
        fuse_mpi_dynamic_layout_working("${_fuse_row_record}" _fuse_row_working)
        if(NOT _fuse_row_working)
          if("${_fuse_row_record}" STREQUAL "")
            set(_fuse_row_reason
                "accept/connect layout ${_fuse_bp_na}+${_fuse_bp_nb} has no measurement on this MPI (KNOWN_ISSUES.md D23)")
          else()
            set(_fuse_row_reason
                "accept/connect layout ${_fuse_bp_na}+${_fuse_bp_nb} measured ${_fuse_row_record} on this MPI (KNOWN_ISSUES.md D23)")
          endif()
          set_tests_properties(${_name} PROPERTIES
            LABELS "parallel;mpi;verification;brusselator;accept-unreliable-on-this-mpi"
            DISABLED TRUE
            FUSE_ACCEPT_LAYOUT_REASON "${_fuse_row_reason}")
        endif()
      endif()
    endforeach()
  endforeach()
endif()
# ---- Conformance Unit 2 host rows: end -------------------------------------

# ---- S6 phase 3 (Conformance, "LowFive" track), Unit 3b host rows:
# begin -----------------------------------------------------------------
# Same host_verification_needed status as the plain rows just above.
# The host's LowFive fmt-12 rebuild (KNOWN_ISSUES.md D14 "what remains"
# item 1) is DONE -- `fuse-lowfive-vol` loads and `.lowfive.mpmd.3` passes
# (D39 Round 10). `.lowfive.accept.3` still needs a host run of D14 Stage
# B's own ABI-gap fix (D14's "what remains" item 3) before it can pass.
# World size 3 only (1+1+1): item 3's own ask names .lowfive.{mpmd,accept}.3,
# not .5; the degenerate reference graph (brusselator_graph_tool.cpp's own
# kDtU == kWindow) this track measured against never needed more than one
# rank per component to prove the wiring, and a second world size would
# only repeat it.
#
# Environment: these rows set NO HDF5_VOL_CONNECTOR, HDF5_PLUGIN_PATH or
# FUSE_LOWFIVE_BINDINGS. They were written for docs/user/lowfive.md's env-var
# route, but u_main and v_main attach LowFive programmatically
# (lowfive_attach.cpp), so nothing here reads them, and setting them is
# harmful: libMOAB's static initialiser (_GLOBAL__sub_I_WriteHDF5.cpp) calls
# H5open before main, HDF5 then reads HDF5_VOL_CONNECTOR and registers the
# LowFive connector in the coupler, and at exit that connector's teardown
# reaches an already-destroyed spdlog registry and aborts (KNOWN_ISSUES.md
# D39). Measured 2026-10-13 on the owner's machine: the mpmd row fails with the
# three variables set and passes with them unset, and also passes with only
# FUSE_LOWFIVE_BINDINGS set. The env-route plugin (fuse-lowfive-vol) stays in
# the tree; the TARGET guard below is kept so the rows are registered under
# the same conditions as before.
if(FUSE_ENABLE_TESTS AND FUSE_ENABLE_MPI AND FUSE_ENABLE_LOWFIVE
   AND TARGET brusselator_u_main AND TARGET brusselator_v_main
   AND TARGET fuse-lowfive-vol)
  foreach(_mode mpmd accept)
    # Same gates as the non-LowFive rows above.
    if(_mode STREQUAL "accept" AND NOT FUSE_ENABLE_CONNECT_ACCEPT)
      continue()
    endif()
    if(_mode STREQUAL "mpmd" AND NOT FUSE_ENABLE_CONNECT_MPMD)
      continue()
    endif()
    set(_name verification.brusselator.participants.lowfive.${_mode}.3)
    set(_scratch ${CMAKE_CURRENT_BINARY_DIR}/brusselator-participants-scratch/lowfive-${_mode}-3)
    add_test(NAME ${_name}
      COMMAND bash ${CMAKE_SOURCE_DIR}/test/parallel/brusselator_participants/run_participants.sh
              ${_mode} ${MPIEXEC_EXECUTABLE}
              1 1 1
              $<TARGET_FILE:brusselator_graph_tool>
              $<TARGET_FILE:brusselator_u_main>
              $<TARGET_FILE:brusselator_v_main>
              $<TARGET_FILE:brusselator_coupler_main>
              $<TARGET_FILE:brusselator_check>
              ${_scratch} 2 60 lowfive)
    # Same OMPI alltoallv-algorithm workaround as the non-LowFive rows above.
    set_tests_properties(${_name} PROPERTIES
      LABELS "parallel;mpi;lowfive;verification;brusselator"
      TIMEOUT 120
      ENVIRONMENT "OMPI_MCA_coll_tuned_use_dynamic_rules=1;OMPI_MCA_coll_tuned_alltoallv_algorithm=2;OMPI_MCA_rmaps_base_oversubscribe=1")
    # KNOWN_ISSUES.md D23's final account, same check as the non-LowFive
    # participants rows above: ranks_u=1, ranks_coupler=1 bundled onto U
    # (the acceptor), ranks_v=1 connecting -- layout 2+1.
    if(_mode STREQUAL "accept")
      fuse_mpi_dynamic_layout_record("${FUSE_MPI_DYNAMIC_LAYOUTS}" "2+1" _fuse_row_record)
      fuse_mpi_dynamic_layout_working("${_fuse_row_record}" _fuse_row_working)
      if(NOT _fuse_row_working)
        if("${_fuse_row_record}" STREQUAL "")
          set(_fuse_row_reason "accept/connect layout 2+1 has no measurement on this MPI (KNOWN_ISSUES.md D23)")
        else()
          set(_fuse_row_reason "accept/connect layout 2+1 measured ${_fuse_row_record} on this MPI (KNOWN_ISSUES.md D23)")
        endif()
        set_tests_properties(${_name} PROPERTIES
          LABELS "parallel;mpi;lowfive;verification;brusselator;accept-unreliable-on-this-mpi"
          DISABLED TRUE
          FUSE_ACCEPT_LAYOUT_REASON "${_fuse_row_reason}")
      endif()
    endif()
  endforeach()
endif()
# ---- S6 phase 3 Unit 3b host rows: end ---------------------------------

# ---- LowFive attach note (revised 2026-10-13) ---------------------------
# Both brusselator_u_main and brusselator_v_main attach LowFive
# programmatically (examples/brusselator/participants/lowfive_attach.cpp) for
# every --transport lowfive row, mpmd and accept alike: every
# bruss_lowfive_write()/bruss_lowfive_read() call passes an explicit fapl_id
# and refuses if it is negative, so neither row's H5Fcreate/H5Fopen reaches
# HDF5_VOL_CONNECTOR's default-VOL resolution. The three env-route variables
# were therefore removed from these two rows' ENVIRONMENT (see the comment
# above the block for the measurement). The mpmd row passes (D39 Round 10,
# 1.70 s). The accept row's own blanket E-LOWFIVE-ATTACH-ACCEPT refusal is
# replaced, D14 Stage B (2026-10-13): decide_attach_mode() now accepts a
# Connect-mode channel when the base communicator is exposed (the new
# base_communicator() ABI entry) and at most two Connect-mode components
# are declared (D18), refusing only those two narrower conditions by name.
# Host-run 2026-10-13: both LowFive rows Passed (accept 1.26 s, mpmd 2.37 s),
# KNOWN_ISSUES.md D14 item 3.
# ---- "LowFive attach" track note: end ----------------------------------

# review-2026-10-11 BC-1: gated on the eigen backend -- both FSI channel
# solvers factor through Eigen::SparseLU with no builtin-backend substitute.
if(FUSE_LINALG_BACKEND_IS_EIGEN)
  # ---------------------------------------------------------------------------
  # FSI channel (FSI track): a 2D unsteady-Stokes channel with an Euler-
  # Bernoulli wall beam, strongly coupled on a non-matching 1D interface
  # (consistent H, conservative H^T -- fuse/backend/consistent_interp.hpp).
  # Needs Eigen3::Eigen directly for both solvers' SparseLU factorisations,
  # same reasoning as fuse_example_gray_scott just above.
  # ---------------------------------------------------------------------------
  add_executable(fuse_example_fsi_channel
    fsi_channel/main.cpp)
  target_link_libraries(fuse_example_fsi_channel PRIVATE
    fuse::runtime fuse::backend Eigen3::Eigen fuse::examples_common)
  target_include_directories(fuse_example_fsi_channel PRIVATE fsi_channel)

  if(FUSE_ENABLE_TESTS)
    # Small grids and few steps so the example itself stays fast; the
    # convergence-order and physics gates live in test/verification/fsi_*_test.
    # --check exits non-zero if a window fails to converge under acceleration,
    # if the beam displacement is ever non-finite, or if the beam never moved
    # (the coupling did nothing measurable).
    add_test(NAME example.fsi_channel
             COMMAND fuse_example_fsi_channel --nx-fluid 16 --ny-fluid 8
                     --nx-beam 12 --mass-ratio 1.0 --accel iqn-ils --dt 0.01
                     --t-end 0.1 --check)
    set_tests_properties(example.fsi_channel PROPERTIES LABELS "example")
    # An unknown --accel name is refused before any output (it already was,
    # after the header line; the shared table now owns the list).
    add_test(NAME example.fsi_channel.accel_refused
             COMMAND fuse_example_fsi_channel --accel no-such-method)
    set_tests_properties(example.fsi_channel.accel_refused PROPERTIES
      LABELS "example" PASS_REGULAR_EXPRESSION "E-EXAMPLE-ACCEL: unknown --accel 'no-such-method'")
    # Same command line at one thread and at every core, bit for bit. IQN-ILS
    # goes through the Runner's reductions, so this is the example most exposed
    # to a non-deterministic sum; both FSI components declare Bitwise on it.
    if(COMMAND fuse_add_omp_legs AND FUSE_KOKKOS_HAS_OPENMP)
      add_test(NAME example.fsi_channel.omp_identical
               COMMAND ${CMAKE_COMMAND}
                       -DBIN=$<TARGET_FILE:fuse_example_fsi_channel>
                       "-DARGS=--nx-fluid;16;--ny-fluid;8;--nx-beam;12;--mass-ratio;1.0;--accel;iqn-ils;--dt;0.01;--t-end;0.1;--check"
                       -DTHREADS_HI=${_fuse_host_cores}
                       -DOUT=${CMAKE_CURRENT_BINARY_DIR}/omp_fsi_channel
                       -P ${CMAKE_SOURCE_DIR}/tools/compare-runs.cmake)
      set_tests_properties(example.fsi_channel.omp_identical PROPERTIES LABELS "example;omp")
    endif()
    # FUSE_ENABLE_CUSPARSE=ON: these two rows are DISABLED (ctest reports them
    # as "Not Run (Disabled)", not hidden), KNOWN_ISSUES.md D43 (review
    # 2026-10-11 used the pre-rebase number D36 for this finding; D36 in the
    # current KNOWN_ISSUES.md is the unrelated accept-mode RankSet fix).
    # Measured 2026-10-11: the example's IQN-ILS coupling is marginally stable
    # at the step nearest the beam's first natural period (step 7 of the
    # default run). With Eigen it passes that step in 46 sweeps, twice its
    # usual count; with the cusolverRf backend -- whose solutions agree with
    # Eigen's to <= 1.1e-12 on every one of the 730 solves before the
    # divergence, and are closer to a long-double reference than Eigen's own
    # -- the same step diverges to 2e80. The flag-off build itself diverges
    # with --accel none and --accel aitken at the same periods the flag-on
    # build does. The fsi_* verification rows (test/verification) still run
    # under the flag and pass.
    if(FUSE_ENABLE_CUSPARSE)
      set_tests_properties(example.fsi_channel PROPERTIES DISABLED TRUE)
      if(TEST example.fsi_channel.omp_identical)
        set_tests_properties(example.fsi_channel.omp_identical PROPERTIES DISABLED TRUE)
      endif()
    endif()
  endif()
else()
  message(STATUS "fuse: fuse_example_fsi_channel not built under "
                 "FUSE_LINALG_BACKEND=builtin (needs Eigen::SparseLU)")
  list(APPEND _fuse_eigen_gated_examples fuse_example_fsi_channel)
endif()

if(FUSE_ENABLE_MPI)
  add_executable(fuse_example_layout_driver layout_driver/main.cpp)
  target_link_libraries(fuse_example_layout_driver PRIVATE
    fuse::runtime fuse::backend fuse::partition fuse::transport MPI::MPI_CXX)
  # PartitionedTestBackend (test/support/partitioned_test_backend.hpp) is the
  # only MeshBackend in the tree that gives each graph component its own
  # decomposition of a shared mesh -- see that header's own comment. Sharing
  # it here, rather than a second copy, is why this include directory reaches
  # into test/.
  target_include_directories(fuse_example_layout_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/test/support)

  # fuse_add_mpi_test (test/CMakeLists.txt) always builds a NEW target from
  # test/<layer>/<name>_test.cpp -- one gtest binary per registration. This is
  # ONE binary, driven by different CLI arguments at each rank count and
  # processor set, so the mpiexec line is written out directly instead.
  function(fuse_add_layout_test name ranks)
    cmake_parse_arguments(ARG "WILL_FAIL" "" "ARGS" ${ARGN})
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                     $<TARGET_FILE:fuse_example_layout_driver> ${ARG_ARGS})
    set_tests_properties(${name} PROPERTIES TIMEOUT 120 LABELS "mpi;parallel;layout")
    if(ARG_WILL_FAIL)
      set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
    endif()
  endfunction()

  # Identical, disjoint and partially-overlapping processor sets at 2, 3 and 4
  # ranks -- partial overlap needs >= 3 ranks (2 ranks can only be identical
  # or disjoint), and at 3 and 4 ranks one arrangement of each non-identical
  # shape puts comp2 on the LOWER range, not comp1.
  fuse_add_layout_test(layout.2.c1-0-0.c2-1-1 2 ARGS --comp1 0:0 --comp2 1:1 --check)
  fuse_add_layout_test(layout.2.c1-1-1.c2-0-0 2 ARGS --comp1 1:1 --comp2 0:0 --check)
  fuse_add_layout_test(layout.2.c1-0-1.c2-0-1 2 ARGS --comp1 0:1 --comp2 0:1 --check)

  fuse_add_layout_test(layout.3.c1-0-0.c2-1-2 3 ARGS --comp1 0:0 --comp2 1:2 --check)
  fuse_add_layout_test(layout.3.c1-0-1.c2-1-2 3 ARGS --comp1 0:1 --comp2 1:2 --check)
  fuse_add_layout_test(layout.3.c1-1-2.c2-0-1 3 ARGS --comp1 1:2 --comp2 0:1 --check)
  fuse_add_layout_test(layout.3.c1-0-2.c2-0-2 3 ARGS --comp1 0:2 --comp2 0:2 --check)

  fuse_add_layout_test(layout.4.c1-0-1.c2-2-3 4 ARGS --comp1 0:1 --comp2 2:3 --check)
  fuse_add_layout_test(layout.4.c1-0-2.c2-1-3 4 ARGS --comp1 0:2 --comp2 1:3 --check)
  fuse_add_layout_test(layout.4.c1-1-3.c2-0-2 4 ARGS --comp1 1:3 --comp2 0:2 --check)
  fuse_add_layout_test(layout.4.c1-0-3.c2-0-3 4 ARGS --comp1 0:3 --comp2 0:3 --check)

  # --coupler: the coupler-placement track (design C). The field transfer always
  # gets RemapFlags::Conservative (main.cpp's own comment on why), so
  # --check still verifies the same "every destination entity holds its own
  # global id" property as the rows above, now routed source -> coupler ->
  # target.
  #   * coupler disjoint from both comp1 and comp2.
  fuse_add_layout_test(layout.coupler.3.disjoint 3
    ARGS --comp1 0:0 --comp2 1:1 --coupler 2:2 --check)
  #   * coupler IDENTICAL to comp2's own range -- the coupler -> target hop
  #     is elided (same_as), so this must reproduce the no-coupler result
  #     bit for bit; the byte-exact claim itself is pinned at the base build configuration
  #     (CouplerLowering.CouplerEqualToTargetReproducesTodaysIRExactly,
  #     Runner.ACouplerNeedsNoComponentAndDeliversTheIdenticalValue), --check
  #     here confirms delivery still holds under a real multi-rank run.
  fuse_add_layout_test(layout.coupler.3.equals-target 3
    ARGS --comp1 0:0 --comp2 1:2 --coupler 1:2 --check)
  #   * coupler overlapping comp1 (shares rank 1, not identical to either
  #     side) -- needs 4 ranks for the same reason a partial comp1/comp2
  #     overlap needs 3: the shared rank must leave room for the rest of
  #     each processor set to differ.
  fuse_add_layout_test(layout.coupler.4.overlaps-comp1 4
    ARGS --comp1 0:1 --comp2 2:3 --coupler 1:2 --check)

  # --mesh/--migrate: comp2's mesh is comp1's, migrated with RCB onto
  # comp2's processor set (Args::mesh_spec's own comment on what --check means
  # in this mode: cell count and measure conservation, not the field
  # field transfer). One box generator per rank count, on the same
  # disjoint/overlapping/identical shapes as the field-transfer rows above.
  fuse_add_layout_test(layout.migrate.2.c1-0-0.c2-1-1 2
    ARGS --comp1 0:0 --comp2 1:1 --mesh hexbox:4x4x4 --migrate --check)
  fuse_add_layout_test(layout.migrate.3.c1-0-1.c2-1-2 3
    ARGS --comp1 0:1 --comp2 1:2 --mesh hexbox:4x4x4 --migrate --check)
  fuse_add_layout_test(layout.migrate.4.c1-0-2.c2-1-3 4
    ARGS --comp1 0:2 --comp2 1:3 --mesh hexbox:4x4x4 --migrate --check)

  # Deliberate rejections (docs/dev/testing.md: counts as coverage only when a
  # test makes the rejection fire). Run at 2 ranks -- the rank count is
  # incidental to what each of these is testing.
  fuse_add_layout_test(layout.refuse.min-nonzero 2 WILL_FAIL
    ARGS --comp1 1:1 --comp2 1:1)
  fuse_add_layout_test(layout.refuse.max-not-last 2 WILL_FAIL
    ARGS --comp1 0:0 --comp2 0:0)
  fuse_add_layout_test(layout.refuse.reversed-range 2 WILL_FAIL
    ARGS --comp1 1:0 --comp2 0:1)
  # --coupler is rejected only in combination with --mesh (E-LAYOUT-COUPLER):
  # that mode never builds the field-transfer graph a coupler could attach
  # to. Plain --coupler is no longer a rejection -- see the layout.coupler.*
  # rows above -- so this replaces the row that used to pin the "not
  # implemented yet" rejection.
  fuse_add_layout_test(layout.refuse.coupler-with-mesh 2 WILL_FAIL
    ARGS --comp1 0:1 --comp2 0:1 --coupler 0:1 --mesh hexbox:4x4x4)
endif()

# ---- S6 phase 3, "LowFive" track: begin -----------------------------------
# Demo B (docs/user/lowfive.md, KNOWN_ISSUES.md section D): two SEPARATE
# executables, demo_b_producer and demo_b_consumer, coupled purely by
# LowFive's HDF5-VOL interception across an MPI intercommunicator the two
# form themselves (examples/demo_b/rendezvous.hpp) -- not launched as one
# MPMD job. Needs the distributed path (set_intercomm/serve_as_producer/
# receive_as_consumer), which itself needs FUSE_ENABLE_MPI, a LowFive build,
# AND the separate liblowfive-dist archive LOWFIVE_HAVE_DIST records
# (cmake/FindLowFive.cmake); MOAB is NOT required by this example (the
# producer/consumer link nothing from fuse::backend), so its own hazard
# probe (cmake/FuseDependencies.cmake's LowFive block) only runs, and only
# matters, when FUSE_ENABLE_MOAB is ALSO on in the same configure.
#
# Run as the ctest row example.demo_b below (examples/demo_b/run_demo.sh).
if(FUSE_ENABLE_LOWFIVE AND FUSE_ENABLE_MPI AND LOWFIVE_HAVE_DIST)
  add_library(demo_b_producer_hdf5 OBJECT demo_b/producer_hdf5.cpp)
  target_link_libraries(demo_b_producer_hdf5 PRIVATE fuse::lowfive)

  add_library(demo_b_consumer_hdf5 OBJECT demo_b/consumer_hdf5.cpp)
  target_link_libraries(demo_b_consumer_hdf5 PRIVATE fuse::lowfive)

  add_executable(demo_b_producer demo_b/producer_main.cpp $<TARGET_OBJECTS:demo_b_producer_hdf5>)
  target_link_libraries(demo_b_producer PRIVATE fuse::transport MPI::MPI_CXX)
  target_include_directories(demo_b_producer PRIVATE demo_b)

  add_executable(demo_b_consumer demo_b/consumer_main.cpp $<TARGET_OBJECTS:demo_b_consumer_hdf5>)
  target_link_libraries(demo_b_consumer PRIVATE fuse::transport MPI::MPI_CXX)
  target_include_directories(demo_b_consumer PRIVATE demo_b)

  # Two separate one-task launches joined by MPI_Comm_accept/connect, so the
  # row needs the dynamic-process capability the configure probe measures
  # (cmake/FuseMpiDynamic.cmake). Exit code 0 means both processes exited 0:
  # the consumer's values match the producer's bit for bit AND no HDF5 file
  # reached disk (consumer_main.cpp).
  if(FUSE_ENABLE_CONNECT_ACCEPT)
    set(_demo_b_scratch ${CMAKE_CURRENT_BINARY_DIR}/demo-b-scratch)
    add_test(NAME example.demo_b
      COMMAND ${CMAKE_COMMAND} -E env MPIEXEC=${MPIEXEC_EXECUTABLE}
              bash ${CMAKE_CURRENT_SOURCE_DIR}/demo_b/run_demo.sh ${CMAKE_BINARY_DIR}
              ${_demo_b_scratch}/port ${_demo_b_scratch}/demo_b_producer_out.h5 /grid/values)
    set_tests_properties(example.demo_b PROPERTIES
      LABELS "parallel;mpi;lowfive;example"
      TIMEOUT 120)
  endif()
else()
  message(STATUS "fuse: Demo B (examples/demo_b) needs "
                 "-DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON and a "
                 "LowFive install shipping liblowfive-dist; not built")
endif()
# ---- S6 phase 3, "LowFive" track: end -------------------------------------

# review-2026-10-11 BC-1: one line naming every example this configure
# skipped because FUSE_LINALG_BACKEND is not eigen, same convention as
# test/CMakeLists.txt's own skip summary at the bottom of its audit section.
if(NOT FUSE_LINALG_BACKEND_IS_EIGEN)
  list(LENGTH _fuse_eigen_gated_examples _n_eigen_gated_examples)
  list(JOIN _fuse_eigen_gated_examples ", " _fuse_eigen_gated_examples_text)
  message(STATUS "fuse: ${_n_eigen_gated_examples} example target(s) skipped under "
                 "FUSE_LINALG_BACKEND=builtin (no Eigen3::Eigen target): "
                 "${_fuse_eigen_gated_examples_text}")
endif()

# Last: the label applies to rows registered above, in this directory.
fuse_label_slow_tests()
