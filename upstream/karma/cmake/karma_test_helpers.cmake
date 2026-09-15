## Shared registration policy for every KARMA test, in test/ and examples/.
##
## Include this once per directory that registers tests. It overrides
## add_test so that two things are true of every test without the call site
## having to remember them - which matters because the cost of forgetting is
## not a build error, it is a test that fails only under `ctest -j N`, and
## only sometimes.
##
## Overriding a built-in command is a documented CMake idiom: the original
## remains reachable with a leading underscore. The override is scoped to the
## including directory, which is why examples/ has to include this too rather
## than inheriting test/'s copy.
##
## 1. WORKING_DIRECTORY - a private scratch directory per test.
##
##    Test sources create scratch files under fixed relative names
##    (karma_checkpoint_cleanup_test/, karma_checkpoint_publish_target.h5m,
##    karma_moab_io_test.h5m, and about a dozen more). That was harmless
##    while CTest ran one test at a time, but 30 registered tests invoke
##    karma_moab_io_test alone, and some of those names are touched from an
##    unconditional preamble that runs whatever selector was passed. Two such
##    tests at once in a shared directory delete each other's scratch state
##    mid-assertion - which is what made karma_moab_io_write_partition_fixture
##    fail under -j2 with "checkpoint cleanup reports a nonempty reservation
##    directory", and, since it is a FIXTURES_SETUP test, skipped everything
##    downstream of it.
##
##    Doing this here fixes the whole class rather than the one filename that
##    happened to fail first, and needs no change to the numerical sources.
##    It is safe because tests exchange data through ABSOLUTE paths
##    (${CMAKE_CURRENT_BINARY_DIR}/...): no test passes another a relative
##    path and no test command is itself relative, both verified against the
##    generated CTestTestfile.
##
##    It is also the right granularity for MPI. Every rank of one mpiexec
##    invocation inherits this single directory, so a collective relative
##    path stays identical across ranks - which a per-process scratch name (a
##    pid suffix, say) would have broken in
##    checkpoint_failure_does_not_mutate_source(), where all ranks must agree
##    on the destination.
##
## 2. PROCESSORS - how many slots the test really occupies.
##
##    CTest counts a test as one slot unless told otherwise, so `ctest -j 12`
##    against undeclared 5-rank tests would run ~60 processes on 12 cores.
##    That does not fail cleanly; it makes MPI tests slow and then time out.
##
##    The count is read from the arguments following MPIEXEC_NUMPROC_FLAG and
##    SUMMED, not taken from the first one: the MPMD example tests launch
##    `mpiexec -n 1 exe : -n 1 exe`, which occupies two slots, and the test
##    name is not a reliable source either
##    (karma_moab_io_restore_2_checkpoint_3 runs 3 ranks while carrying a 2).
## Include guard. CMake functions are GLOBAL, not directory-scoped, so a
## single include covers test/ and examples/ both - and a second include
## would make this override's _add_test resolve to the first override
## instead of the built-in, recursing until CMake bails out at depth 1000.
if(KARMA_TEST_HELPERS_INCLUDED)
  return()
endif()
set(KARMA_TEST_HELPERS_INCLUDED TRUE)

##
## 3. LABELS refusal_asserting - tests that assert an error is REPORTED.
##
##    KARMA_ENABLE_EXCEPTIONS is documented as "Throw C++ exceptions on
##    errors". With it OFF, KARMA_THROW aborts instead, so a test whose job is
##    to check that a refusal is raised cannot pass: the refusal kills the
##    process. That is the option working as specified, not a defect, and no
##    amount of fixing the library changes it.
##
##    MEMBERSHIP IS EMPIRICAL. Two static detectors were tried and both were
##    wrong, in opposite directions, so do not trust a grep here:
##
##      - by helper name - throws(), throws_invalid(), throws_with(),
##        throws_exception(), and the template form
##        throws< std::logic_error >( ... ). UNDER-detects. It missed the
##        template form, which system_owner_test.C uses exclusively, and it
##        missed mesh_test.C entirely, which uses no helper at all - just
##        inline try { ... } catch( const InvalidArgument& ) { flag = true; }.
##        karma_mesh_test reached the no-exceptions leg unlabelled and aborted
##        there on the host, which is how this was found.
##
##      - by catch site - any catch of a KARMA or std error type.
##        OVER-detects badly: 51 targets match, but most are the single
##        catch( const std::exception& ) in main() that nearly every test has
##        for reporting, which never fires and is harmless here.
##
##    The authoritative check is to RUN each binary in the no-exceptions build;
##    an abort (exit 134, reported by CTest as "Subprocess aborted") is the
##    answer. To regenerate this list after adding or changing tests:
##
##      for t in build-no-exceptions/test/karma_*_test; do
##        "$t" >/dev/null 2>&1; [ $? -eq 134 ] && echo "$(basename "$t")"
##      done
##
##    Most binaries need PETSc, so that is a host action, not a sandbox one.
##
##    Observed rather than assumed: karma_basis_test exits 134 (SIGABRT) in
##    the no-exceptions build and 0 in the release build, from a `throws()`
##    helper catching std::logic_error, which KARMA::LogicError derives from.
##    The reach is wider than the test that reports it first - three fixture
##    tests were seen aborting, and all three run karma_moab_io_test, whose
##    unconditional preamble makes about ten such assertions before it even
##    looks at its selector, so every one of the 30 tests invoking that binary
##    aborts and their dependents are skipped.
##
##    Membership is by SOURCE, not by which assertions actually trip. The
##    test sources that assert a refusal at all are listed below, and
##    every registered test invoking one of them is labelled. That deliberately
##    over-labels: karma_operator_split_graph_test asserts refusals through
##    raw `throw std::invalid_argument`, which is unaffected because this
##    option does not pass -fno-exceptions, and it passes in both builds.
##    Over-labelling costs a test that is merely not run in one configuration;
##    under-labelling costs a red suite that is reporting the option working.
##    Classifying exactly would mean running each of the PETSc-dependent ones
##    under the no-exceptions build, which is a host action.
##
##    The label alone is NOT sufficient, and this is worth knowing before
##    trusting it: CTest re-adds a FIXTURES_SETUP test that a selected test
##    requires, whatever label it carries. Eight fixture-setup tests are
##    labelled and were still selected for that reason - which is exactly how
##    karma_moab_io_write_partition_fixture came to abort in a run that had
##    excluded its label. So the no-exceptions test preset selects by an
##    explicit include list of binaries with no refusal assertions rather than
##    by excluding this label, and the label remains the record of WHY each
##    one is out.
##
##    What that leg proves is that the configuration builds and its
##    non-refusal paths run. Proving that refusals are raised requires
##    exceptions, by construction: with them off a refusal aborts, so there is
##    nothing left to observe it.

set(KARMA_REFUSAL_ASSERTING_TARGETS
    karma_advection_flux_test
    karma_algebraic_system_test
    karma_assembly_test
    karma_basis_test
    karma_boundary_condition_test
    karma_diffusion_assembly_test
    karma_fe_face_test
    karma_fe_test
    karma_geometry_test
    karma_gray_scott_model_test
    karma_mass_assembly_test
    karma_mesh_test
    karma_moab_io_test
    karma_ode_system_test
    karma_system_owner_test
    karma_system_test
    )

function(add_test)
  _add_test(${ARGV})

  # Only the add_test(NAME <name> COMMAND ...) signature carries a name that
  # properties can be attached to; the positional form is left alone.
  list(GET ARGV 0 _karma_first_argument)
  if(NOT _karma_first_argument STREQUAL "NAME")
    return()
  endif()
  list(GET ARGV 1 _karma_test_name)

  set(_karma_test_workdir "${CMAKE_CURRENT_BINARY_DIR}/test-scratch/${_karma_test_name}")
  file(MAKE_DIRECTORY "${_karma_test_workdir}")
  set_tests_properties("${_karma_test_name}" PROPERTIES WORKING_DIRECTORY "${_karma_test_workdir}")

  # Label any test that runs a refusal-asserting executable, whatever form the
  # command takes: a bare target name or a $<TARGET_FILE:...> expression both
  # contain the target name as a substring.
  foreach(_karma_refusal_target IN LISTS KARMA_REFUSAL_ASSERTING_TARGETS)
    foreach(_karma_argument IN LISTS ARGV)
      if("${_karma_argument}" MATCHES "${_karma_refusal_target}")
        set_property(TEST "${_karma_test_name}" APPEND PROPERTY LABELS refusal_asserting)
        break()
      endif()
    endforeach()
  endforeach()

  if(NOT MPIEXEC_NUMPROC_FLAG)
    return()
  endif()

  # Sum every "<numproc flag> <count>" pair so an MPMD launch is counted whole.
  set(_karma_total_ranks 0)
  list(LENGTH ARGV _karma_argument_count)
  math(EXPR _karma_last_index "${_karma_argument_count} - 2")
  if(_karma_last_index LESS 0)
    return()
  endif()
  foreach(_karma_index RANGE ${_karma_last_index})
    list(GET ARGV ${_karma_index} _karma_argument)
    if(NOT _karma_argument STREQUAL "${MPIEXEC_NUMPROC_FLAG}")
      continue()
    endif()
    math(EXPR _karma_next "${_karma_index} + 1")
    list(GET ARGV ${_karma_next} _karma_rank_count)
    # A non-numeric value means the flag matched something that is not a rank
    # count; declaring PROCESSORS from it would be worse than leaving it unset.
    if(_karma_rank_count MATCHES "^[0-9]+$")
      math(EXPR _karma_total_ranks "${_karma_total_ranks} + ${_karma_rank_count}")
    endif()
  endforeach()

  if(_karma_total_ranks GREATER 0)
    set_tests_properties("${_karma_test_name}" PROPERTIES PROCESSORS ${_karma_total_ranks})
  endif()
endfunction()
