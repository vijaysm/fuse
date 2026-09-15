include("${CMAKE_CURRENT_LIST_DIR}/PetscScalarProbe.cmake")

set(work "${CMAKE_CURRENT_BINARY_DIR}/petsc-scalar-probe-test")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

foreach(case IN ITEMS single-real double-real single-complex double-complex)
    string(REPLACE "-" ";" parts "${case}")
    list(GET parts 0 precision)
    list(GET parts 1 scalar)
    string(TOUPPER "${precision}" precision_macro)
    set(header "#define PETSC_USE_REAL_${precision_macro} 1\n")
    if(scalar STREQUAL "complex")
        string(APPEND header "#define PETSC_USE_COMPLEX 1\n")
    endif()
    set(path "${work}/${case}.h")
    file(WRITE "${path}" "${header}")
    karma_parse_petsc_scalar_config("${path}" actual)
    string(REPLACE "-" " " expected "${case}")
    if(NOT actual STREQUAL expected)
        message(FATAL_ERROR "${case}: expected '${expected}', got '${actual}'")
    endif()
endforeach()

foreach(case IN ITEMS single-real single-complex double-complex)
    string(REPLACE "-" " " mode "${case}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
                "-DPETSC_CONFIG_HEADER=${work}/${case}.h"
                "-DEXPECTED_MODE=${mode}"
                -P "${CMAKE_CURRENT_LIST_DIR}/check_petsc_scalar.cmake"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE output)
    if(result EQUAL 0 OR NOT output MATCHES "requires PETSc double real")
        message(FATAL_ERROR "configured ${mode} was not rejected clearly: '${output}'")
    endif()
endforeach()
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            "-DPETSC_CONFIG_HEADER=${work}/double-real.h"
            "-DEXPECTED_MODE=double real"
            -P "${CMAKE_CURRENT_LIST_DIR}/check_petsc_scalar.cmake"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "configured double real was rejected: '${output}'")
endif()

get_filename_component(source_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
foreach(case IN ITEMS single-real double-real)
    set(prefix "${work}/configure-${case}")
    file(MAKE_DIRECTORY "${prefix}/include" "${prefix}/lib/pkgconfig")
    if(case STREQUAL "single-real")
        set(precision SINGLE)
    else()
        set(precision DOUBLE)
    endif()
    file(WRITE "${prefix}/include/petscconf.h"
         "#define PETSC_USE_REAL_${precision} 1\n")
    file(WRITE "${prefix}/lib/pkgconfig/PETSc.pc"
         "prefix=${prefix}\nlibdir=\${prefix}/lib\nincludedir=\${prefix}/include\nName: PETSc\nDescription: synthetic PETSc\nVersion: 3.25.0\nLibs: -L\${libdir} -lpetsc\nCflags: -I\${includedir}\n")
    file(WRITE "${prefix}/lib/pkgconfig/MOAB.pc"
         "prefix=${prefix}\nlibdir=\${prefix}/lib\nincludedir=\${prefix}/include\nName: MOAB\nDescription: synthetic MOAB\nVersion: 5.6.0\nLibs: -L\${libdir} -lMOAB\nCflags: -I\${includedir}\n")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
                "PKG_CONFIG_PATH=${prefix}/lib/pkgconfig"
                "${CMAKE_COMMAND}" -S "${source_dir}" -B "${work}/build-${case}"
                "-DCMAKE_PREFIX_PATH=${prefix}" -DKARMA_ENABLE_TESTS=OFF
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE output)
    if(case STREQUAL "single-real")
        if(result EQUAL 0 OR NOT output MATCHES "Configured PETSc is unsupported: KARMA requires PETSc double real")
            message(FATAL_ERROR "top-level configure did not reject single real at the scalar gate: '${output}'")
        endif()
    elseif(NOT result EQUAL 0)
        message(FATAL_ERROR "top-level configure did not pass the double-real scalar gate: '${output}'")
    endif()
endforeach()

foreach(mode IN ITEMS "single real" "single complex" "double complex")
    karma_validate_petsc_scalar_mode("${mode}" error)
    if(NOT error MATCHES "requires PETSc double real")
        message(FATAL_ERROR "unsupported mode '${mode}' was not rejected: '${error}'")
    endif()
endforeach()
karma_validate_petsc_scalar_mode("double real" error)
if(error)
    message(FATAL_ERROR "double real was not accepted: '${error}'")
endif()
foreach(mode IN ITEMS "single real" "single complex" "double complex")
    karma_validate_petsc_candidate_report(
        "candidate: PETSc candidate ${mode}, scalar ABI: unsupported, MOAB metadata: compatible"
        error)
    if(NOT error MATCHES "requires PETSc double real")
        message(FATAL_ERROR "unsupported variant '${mode}' was advertised as testable: '${error}'")
    endif()
endforeach()
karma_validate_petsc_candidate_report(
    "candidate: PETSc candidate double real, scalar ABI: supported, MOAB metadata: compatible"
    error)
if(error)
    message(FATAL_ERROR "double-real compatible variant was rejected: '${error}'")
endif()
foreach(case IN ITEMS missing-prefix missing-header invalid-scalar)
    file(MAKE_DIRECTORY "${work}/${case}/lib/pkgconfig" "${work}/${case}/include")
endforeach()
file(WRITE "${work}/missing-prefix/lib/pkgconfig/PETSc.pc" "Name: PETSc\n")
file(WRITE "${work}/missing-header/lib/pkgconfig/PETSc.pc"
     "prefix=${work}/missing-header\n")
file(WRITE "${work}/invalid-scalar/lib/pkgconfig/PETSc.pc"
     "prefix=${work}/invalid-scalar\n")
file(WRITE "${work}/invalid-scalar/include/petscconf.h" "#define PETSC_USE_COMPLEX 1\n")
karma_probe_petsc_candidates(failed_reports
    "${work}/missing-prefix/lib/pkgconfig"
    "${work}/missing-header/lib/pkgconfig"
    "${work}/invalid-scalar/lib/pkgconfig")
foreach(report IN LISTS failed_reports)
    karma_validate_petsc_candidate_report("${report}" error)
    if(NOT error STREQUAL report)
        message(FATAL_ERROR "candidate probe failure was replaced: '${report}' became '${error}'")
    endif()
endforeach()
if(NOT failed_reports MATCHES "PETSc metadata has no prefix" OR
   NOT failed_reports MATCHES "PETSc configuration header not found" OR
   NOT failed_reports MATCHES "PETSc precision is neither supported single nor double")
    message(FATAL_ERROR "candidate probe failure coverage is incomplete: '${failed_reports}'")
endif()

file(WRITE "${work}/invalid.h" "#define PETSC_USE_COMPLEX 1\n")
karma_parse_petsc_scalar_config("${work}/invalid.h" invalid ERROR_VARIABLE error)
if(NOT error MATCHES "precision")
    message(FATAL_ERROR "missing precision was not rejected: '${error}'")
endif()

foreach(candidate IN ITEMS real complex)
    file(MAKE_DIRECTORY "${work}/${candidate}/lib/pkgconfig" "${work}/${candidate}/include")
    file(WRITE "${work}/${candidate}/lib/pkgconfig/PETSc.pc" "prefix=${work}/${candidate}\n")
endforeach()
file(WRITE "${work}/real/include/petscconf.h" "#define PETSC_USE_REAL_DOUBLE 1\n")
file(WRITE "${work}/real/lib/pkgconfig/MOAB.pc" "prefix=${work}/real\n")
file(WRITE "${work}/complex/include/petscconf.h"
     "#define PETSC_USE_REAL_DOUBLE 1\n#define PETSC_USE_COMPLEX 1\n")
karma_probe_petsc_candidates(reports
    "${work}/real/lib/pkgconfig"
    "${work}/complex/lib/pkgconfig")
list(GET reports 0 real_report)
list(GET reports 1 complex_report)
if(NOT real_report MATCHES "double real.*scalar ABI: supported.*MOAB metadata: compatible")
    message(FATAL_ERROR "real candidate report is wrong: '${real_report}'")
endif()
if(NOT complex_report MATCHES "double complex.*scalar ABI: unsupported.*MOAB metadata: missing")
    message(FATAL_ERROR "complex candidate report is wrong: '${complex_report}'")
endif()
karma_summarize_petsc_variants(summaries "double real" ${reports})
list(GET summaries 0 single_real)
list(GET summaries 1 double_real)
list(GET summaries 2 single_complex)
list(GET summaries 3 double_complex)
if(NOT single_real STREQUAL "single real: unavailable" OR
   NOT double_real MATCHES "configured" OR
   NOT single_complex STREQUAL "single complex: unavailable" OR
   NOT double_complex MATCHES "discovered but unsupported")
    message(FATAL_ERROR "variant summary is wrong: '${summaries}'")
endif()

file(MAKE_DIRECTORY
    "${work}/prefix/lib/pkgconfig"
    "${work}/prefix/include/petsc"
    "${work}/env/lib/pkgconfig"
    "${work}/explicit")
file(WRITE "${work}/prefix/lib/pkgconfig/PETSc.pc"
    "prefix=${work}/prefix\nincludedir=\${prefix}/include\nCflags: -I\${includedir}\n")
file(WRITE "${work}/prefix/include/petsc/petscconf.h"
    "#define PETSC_USE_REAL_SINGLE 1\n#define PETSC_USE_COMPLEX 1\n")
file(WRITE "${work}/prefix/lib/pkgconfig/MOAB.pc"
    "prefix=${work}/relocated-moab-build\nRequires.private: PETSc\n")
file(WRITE "${work}/prefix/lib/libMOAB.a" "")
file(WRITE "${work}/env/lib/pkgconfig/PETSc.pc"
    "prefix=${work}/env\nincludedir=\${prefix}/include\n")
file(MAKE_DIRECTORY "${work}/env/include")
file(WRITE "${work}/env/include/petscconf.h" "#define PETSC_USE_REAL_DOUBLE 1\n")
file(WRITE "${work}/env/lib/pkgconfig/MOAB.pc"
    "prefix=${work}/other\nRequires: PETSc\n")
file(WRITE "${work}/explicit/PETSc.pc"
    "prefix=${work}/prefix\nincludedir=\${prefix}/include\n")

set(CMAKE_PREFIX_PATH "${work}/prefix")
set(ENV{PKG_CONFIG_PATH} "${work}/env/lib/pkgconfig:${work}/prefix/lib/pkgconfig")
karma_collect_petsc_candidate_dirs(candidate_dirs
    EXPLICIT_DIRS "${work}/explicit" "${work}/env/lib/pkgconfig")
list(LENGTH candidate_dirs candidate_count)
if(NOT candidate_count EQUAL 3)
    message(FATAL_ERROR "candidate discovery did not deduplicate standard inputs: '${candidate_dirs}'")
endif()
karma_probe_petsc_candidates(discovered_reports ${candidate_dirs})
if(NOT discovered_reports MATCHES
   "prefix/lib/pkgconfig: PETSc candidate single complex, scalar ABI: unsupported, MOAB metadata: compatible")
    message(FATAL_ERROR "compatible candidate was not classified: '${discovered_reports}'")
endif()
if(NOT discovered_reports MATCHES
   "env/lib/pkgconfig: PETSc candidate double real, scalar ABI: supported, MOAB metadata: incompatible")
    message(FATAL_ERROR "incompatible candidate was not classified: '${discovered_reports}'")
endif()
if(NOT discovered_reports MATCHES "explicit: PETSc candidate single complex")
    message(FATAL_ERROR "explicit candidate did not expand pkg-config variables: '${discovered_reports}'")
endif()
karma_summarize_petsc_variants(discovered_summaries "double real" ${discovered_reports})
list(GET discovered_summaries 2 discovered_single_complex)
if(NOT discovered_single_complex MATCHES "discovered but unsupported")
    message(FATAL_ERROR "unsupported candidate was advertised as testable: '${discovered_summaries}'")
endif()
set(reversed_reports
    "candidate: PETSc candidate single complex, scalar ABI: unsupported, MOAB metadata: compatible"
    "candidate: PETSc candidate single complex, scalar ABI: unsupported, MOAB metadata: missing")
karma_summarize_petsc_variants(reversed_summaries "double real" ${reversed_reports})
list(GET reversed_summaries 2 reversed_single_complex)
if(NOT reversed_single_complex MATCHES "discovered but unsupported")
    message(FATAL_ERROR "summary depends on candidate order: '${reversed_summaries}'")
endif()

file(REMOVE_RECURSE "${work}")
