# A standalone pure-C project. It knows nothing about KARMA's build: it is
# given a prefix, adds one include path, and links one library.
cmake_minimum_required(VERSION 3.20)
project(karma_abi_consumer LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT DEFINED KARMA_PREFIX)
  message(FATAL_ERROR "KARMA_PREFIX must name an install prefix")
endif()

find_library(KARMA_ABI_LIBRARY NAMES karma_abi
             PATHS "${KARMA_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(karma_abi_consumer main.c)
# Deliberately the only include path: if an ABI header needed a sibling
# directory, this would fail to compile.
target_include_directories(karma_abi_consumer PRIVATE "${KARMA_PREFIX}/include")
target_link_libraries(karma_abi_consumer PRIVATE "${KARMA_ABI_LIBRARY}" m)
set_target_properties(karma_abi_consumer PROPERTIES
  BUILD_RPATH "${KARMA_PREFIX}/lib"
  INSTALL_RPATH "${KARMA_PREFIX}/lib")
