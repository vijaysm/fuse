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

add_executable(fuse_example_mms_pair mms_pair/main.cpp)
target_link_libraries(fuse_example_mms_pair PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_mms_pair PRIVATE mms_pair)

add_executable(fuse_example_dn_heat dn_heat/main.cpp)
target_link_libraries(fuse_example_dn_heat PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_dn_heat PRIVATE dn_heat)

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

  # The --dump path has its own failure mode: an empty dump still exits 0, so
  # these assert on the output rather than the status.
  add_test(NAME example.mms_pair.dump COMMAND fuse_example_mms_pair --dump)
  add_test(NAME example.dn_heat.dump  COMMAND fuse_example_dn_heat --dump)
  set_tests_properties(example.mms_pair.dump example.dn_heat.dump PROPERTIES
    PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ===")
  set_tests_properties(example.mms_pair example.dn_heat
    example.mms_pair.dump example.dn_heat.dump PROPERTIES LABELS "example")

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
# Gray-Scott (Time windows track): U and V on one structured grid, coupled
# implicitly at independent time steps through fuse::WindowCoupler. Needs
# Eigen3::Eigen directly (its Newton solves use Eigen::SparseLU) -- available
# because FUSE_LINALG_BACKEND=eigen is the default (cmake/FuseDependencies.cmake).
# ---------------------------------------------------------------------------
add_executable(fuse_example_gray_scott
  gray_scott/main.cpp)
target_link_libraries(fuse_example_gray_scott PRIVATE fuse::runtime fuse::backend Eigen3::Eigen)
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

# ---------------------------------------------------------------------------
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
  fuse::runtime fuse::backend Eigen3::Eigen Kokkos::kokkos)
target_include_directories(fuse_example_brusselator PRIVATE brusselator)

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

  add_executable(brusselator_u_main
    brusselator/participants/u_main.c
    brusselator/participants/u_loop.c
    brusselator/participants/shim.cpp)
  target_link_libraries(brusselator_u_main PRIVATE
    fuse::abi MPI::MPI_C Eigen3::Eigen Kokkos::kokkos)
  target_include_directories(brusselator_u_main PRIVATE brusselator)

  if(TARGET fuse-abi-fortran)
    add_executable(brusselator_v_main
      brusselator/participants/v_main.f90
      brusselator/participants/v_loop.f90
      brusselator/participants/shim.cpp)
    target_link_libraries(brusselator_v_main PRIVATE
      fuse::abi fuse::abi-fortran MPI::MPI_Fortran MPI::MPI_C Eigen3::Eigen Kokkos::kokkos)
    target_include_directories(brusselator_v_main PRIVATE brusselator)
  else()
    message(STATUS "fuse: FUSE_ENABLE_MPI is ON but no Fortran compiler was found; "
                   "brusselator_v_main (Conformance Unit 2) will not be built")
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
      set_tests_properties(${_name} PROPERTIES
        LABELS "parallel;mpi;verification;brusselator" TIMEOUT 120)
    endforeach()
  endforeach()
endif()
# ---- Conformance Unit 2 host rows: end -------------------------------------

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
  fuse::runtime fuse::backend Eigen3::Eigen)
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
# UNEXECUTED in this checkout -- MPI cannot start in the sandbox this was
# written in. Compiles against this host's real LowFive/MPI headers
# (2026-10-02); run via examples/demo_b/run_demo.sh on a host where MPI
# works, per docs/user/lowfive.md's host_verification_needed commands.
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
else()
  message(STATUS "fuse: Demo B (examples/demo_b) needs "
                 "-DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON and a "
                 "LowFive install shipping liblowfive-dist; not built")
endif()
# ---- S6 phase 3, "LowFive" track: end -------------------------------------
