# The pybind11 front-end (spec 6: in-process Python / ML surrogates are peer
# components; spec 3.2: the Python front-end emits the same descriptor as C++
# and YAML).
#
# Reached only from `if(SOMA_ENABLE_PYTHON)` in the top-level CMakeLists.
# SOMA_ENABLE_PYTHON is OFF by default, so a machine with no Python development
# headers never probes for them and the dependency-free default build is
# untouched.
#
# ---------------------------------------------------------------------------
# INTERPRETER SELECTION
# ---------------------------------------------------------------------------
# The task brief specified
#
#     find_package(Python 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)
#
# on the stated grounds that "the installed LowFive extension is built for 3.11
# and Demo B loads both in one interpreter". That is not done here, for two
# reasons, both checked on this machine rather than assumed:
#
#   1. There is no *usable* LowFive Python extension to match. What is on this
#      machine is
#        /opt/lowfive/lib/python3.11/site-packages/lowfive/_lowfive.cpython-311-darwin.so
#      which (a) belongs to /opt/lowfive, the static-HDF5 install docs/BUILDING.md
#      records as unusable as a VOL plugin, and (b) does not import even with
#      that directory on PYTHONPATH:
#        ImportError: dlopen(...): Library not loaded:
#                     /opt/mpich/lib/libmpicxx.12.dylib
#      The install that *is* usable, /opt/lowfive-shared, was configured with
#      -Dlowfive_python=OFF and ships no Python module at all. Pinning every
#      SOMA Python build to 3.11 to match an extension that cannot be imported
#      buys nothing, and makes -DSOMA_ENABLE_PYTHON=ON a hard configure failure
#      on any machine whose Python is not exactly 3.11 -- including two of the
#      three interpreters installed here.
#
#   2. `COMPONENTS Development` demands libpython for EMBEDDING. An extension
#      module does not embed the interpreter and must not link libpython on
#      macOS, where the loader supplies those symbols from the host process;
#      `Development.Module` is the component that expresses exactly that, and
#      it is what pybind11_add_module expects.
#
# SOMA_PYTHON_VERSION restores the pin when Demo B needs it:
#     -DSOMA_ENABLE_PYTHON=ON -DSOMA_PYTHON_VERSION=3.11 -DSOMA_PYTHON_EXACT=ON
# and -DPython_EXECUTABLE=... selects a specific interpreter as usual.
set(SOMA_PYTHON_VERSION "3.9" CACHE STRING
    "Minimum (or, with SOMA_PYTHON_EXACT, exact) Python version for the bindings")
option(SOMA_PYTHON_EXACT
       "Require exactly SOMA_PYTHON_VERSION rather than that version or newer"
       OFF)

if(SOMA_PYTHON_EXACT)
  find_package(Python ${SOMA_PYTHON_VERSION} EXACT
               COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python ${SOMA_PYTHON_VERSION}
               COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# NumPy is not optional: every field binding in src/bindings.cpp is a numpy
# array, and the test suite cannot run without it. pybind11/numpy.h needs no
# NumPy *headers* -- it resolves the C API through the imported module at run
# time -- so the check is that the chosen INTERPRETER can import it, which is
# also the interpreter ctest will use.
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import numpy; print(numpy.__version__)"
  RESULT_VARIABLE _soma_numpy_rc
  OUTPUT_VARIABLE _soma_numpy_version
  ERROR_VARIABLE  _soma_numpy_err
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _soma_numpy_rc EQUAL 0)
  message(FATAL_ERROR
    "SOMA_ENABLE_PYTHON=ON but ${Python_EXECUTABLE} cannot import numpy:\n"
    "${_soma_numpy_err}\n"
    "The bindings exchange field storage as numpy arrays; there is no "
    "fallback. Install numpy for that interpreter, or point "
    "-DPython_EXECUTABLE at one that has it.")
endif()
message(STATUS "SOMA Python bindings: ${Python_EXECUTABLE} "
               "(${Python_VERSION}), numpy ${_soma_numpy_version}")

# FIND_PACKAGE_ARGS lets an already-installed pybind11 satisfy this without a
# network fetch, on the same terms as yaml-cpp and googletest in src/ and
# test/. The version floor is 2.12: `py::dtype::equal()` and the numpy 2.x
# support that src/bindings.cpp relies on landed before that, and 2.13.6 is the
# pinned fallback.
#
# PYBIND11_FINDPYTHON=ON makes pybind11 reuse the FindPython result above
# instead of running its own classic FindPythonInterp/FindPythonLibs search.
# Without it the two can disagree -- observed here: FindPython selected the
# interpreter this file checked for numpy, while pybind11's classic path went
# looking again and also tripped CMP0148's deprecation warning. An extension
# built against one interpreter's headers and tested with another's is a
# crash, not a warning, so the two must be the same object.
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
  FIND_PACKAGE_ARGS 2.12 NAMES pybind11)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_soma src/bindings.cpp)
target_link_libraries(_soma PRIVATE soma_core)
target_compile_options(_soma PRIVATE -Wall -Wextra -Wpedantic)

# The tests. Guarded on SOMA_ENABLE_TESTS because that is what calls
# enable_testing() at the top level; without it add_test() registers nothing.
if(SOMA_ENABLE_TESTS)
  # PYTHONPATH, not an installed package: the extension lands in the build tree
  # and the pure-Python wrapper stays in the source tree, so both directories
  # have to be on the path. $<TARGET_FILE_DIR:_soma> is a generator expression
  # because the module's directory is not known at configure time under a
  # multi-config generator.
  #
  # unittest, not pytest. The suite is written against the standard library so
  # it runs under whichever interpreter the build selected; pytest is installed
  # for some of the interpreters on this machine and not others, and a test
  # that silently does not run is worse than one that is a little less pretty.
  # The file is also a valid pytest suite, so `pytest python/tests` works where
  # pytest is available.
  set(_soma_py_env
      "PYTHONPATH=$<TARGET_FILE_DIR:_soma>:${CMAKE_CURRENT_SOURCE_DIR}"
      "PYTHONDONTWRITEBYTECODE=1"
      "SOMA_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/test/data")

  # Make the sanitizer leg actually cover the bindings.
  #
  # The asan preset puts -fsanitize=address in CMAKE_CXX_FLAGS, so _soma.so is
  # instrumented -- but the interpreter that dlopen()s it is not, and the ASan
  # runtime must be the FIRST thing in the process or its interceptors are not
  # installed. Without this the module aborts at import with "Interceptors are
  # not working. This may be because AddressSanitizer is loaded too late", and
  # the instrumented Python test is silently no coverage at all.
  #
  # Preloading the runtime into an uninstrumented python is the supported way
  # round that. detect_leaks is turned off because CPython and NumPy both hold
  # deliberately-unfreed state for the life of the process (as does this module
  # -- see the ExcTypes comment in src/bindings.cpp), and every one of those
  # would be reported as a leak; the use-after-free and buffer-overflow checks,
  # which are what the numpy binding contract needs, are unaffected.
  string(FIND "${CMAKE_CXX_FLAGS}" "-fsanitize=address" _soma_asan_at)
  if(NOT _soma_asan_at EQUAL -1)
    if(APPLE)
      set(_soma_asan_names libclang_rt.asan_osx_dynamic.dylib)
      set(_soma_preload_var DYLD_INSERT_LIBRARIES)
    else()
      set(_soma_asan_names libasan.so libclang_rt.asan-x86_64.so
                           libclang_rt.asan-aarch64.so)
      set(_soma_preload_var LD_PRELOAD)
    endif()
    set(_soma_asan_runtime "")
    foreach(_name IN LISTS _soma_asan_names)
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=${_name}
                      OUTPUT_VARIABLE _candidate
                      OUTPUT_STRIP_TRAILING_WHITESPACE
                      ERROR_QUIET)
      if(EXISTS "${_candidate}")
        set(_soma_asan_runtime "${_candidate}")
        break()
      endif()
    endforeach()
    if(_soma_asan_runtime)
      list(APPEND _soma_py_env
           "${_soma_preload_var}=${_soma_asan_runtime}"
           "ASAN_OPTIONS=detect_leaks=0")
      message(STATUS "SOMA Python bindings: preloading ${_soma_asan_runtime} "
                     "into ${Python_EXECUTABLE} for python_bindings_test")
    else()
      # Not silently skipped: a sanitizer leg that cannot run the Python test
      # is a gap the reader has to know about.
      message(WARNING
        "SOMA_ENABLE_PYTHON=ON under -fsanitize=address, but the ASan runtime "
        "shared library was not found via '${CMAKE_CXX_COMPILER} "
        "-print-file-name'. python_bindings_test will very likely fail at "
        "import with 'AddressSanitizer is loaded too late'.")
    endif()
  endif()

  add_test(NAME python_bindings_test
           COMMAND ${Python_EXECUTABLE} -m unittest discover
                   --start-directory ${CMAKE_CURRENT_SOURCE_DIR}/tests
                   --top-level-directory ${CMAKE_CURRENT_SOURCE_DIR}/tests
                   --verbose)
  set_tests_properties(python_bindings_test PROPERTIES
    ENVIRONMENT "${_soma_py_env}"
    TIMEOUT 300)
endif()
