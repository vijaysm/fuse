# pybind11 module `_pfuse`, gated by FUSE_ENABLE_PYTHON (cmake/FuseDependencies.cmake, default
# OFF). See docs/dev/python-surface.md for the design this implements.
#
# pybind11_DIR is read from the chosen interpreter's own pybind11.get_cmake_dir() -- never
# hand-searched -- so find_package(pybind11) below resolves against exactly the interpreter the
# caller built for.
find_package(pybind11 REQUIRED)

pybind11_add_module(_pfuse MODULE
  src/module.cpp
  src/base_bindings.cpp
  src/model_bindings.cpp
  src/field_bindings.cpp
  src/component_bindings.cpp
)

target_link_libraries(_pfuse PRIVATE
  fuse::model
  fuse::runtime
  fuse::backend
  fuse::ir
  fuse::schedule
)

# No MPI, no PETSc, no Fortran: v1 drives only fuse::SerialMeshBackend with
# RunnerConfig's default ("copy" transport, no base_communicator) -- see
# docs/dev/python-surface.md, "What this does not expose."

# Built in-tree, not installed: the module lands at <build>/python/_pfuse*.so, and
# pfuse/__init__.py is copied alongside it so `PYTHONPATH=<build>/python python -c "import
# pfuse"` works directly from a configured build tree without an install step.
add_custom_command(TARGET _pfuse POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/pfuse
          $<TARGET_FILE_DIR:_pfuse>/pfuse
  COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:_pfuse>
          $<TARGET_FILE_DIR:_pfuse>/pfuse/$<TARGET_FILE_NAME:_pfuse>
  COMMENT "pfuse: staging pfuse/ package next to the built extension"
)

if(FUSE_ENABLE_TESTS)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  # One ctest row per acceptance test file, so a failure names the file without the whole
  # suite needing to run to isolate it (docs/dev/python-surface.md, "Build/CMake integration").
  # PYTHONPATH points at the staged pfuse/ package directory created above.
  foreach(_pfuse_test IN ITEMS buffers graph yaml runtime newton dn_heat_rows analytic_problem)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${_pfuse_test}.py)
      add_test(NAME python.${_pfuse_test}
               COMMAND ${CMAKE_COMMAND} -E env
                       PYTHONPATH=$<TARGET_FILE_DIR:_pfuse>
                       ${Python3_EXECUTABLE} -m pytest -q
                       ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${_pfuse_test}.py)
      set_tests_properties(python.${_pfuse_test} PROPERTIES LABELS "python")
    endif()
  endforeach()
endif()
