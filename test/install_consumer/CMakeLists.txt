# Standalone consumer project for the install/find_package() smoke test:
# NOT part of the main superbuild -- test/CMakeLists.txt's
# "# ---- install" section configures this as its own independent project,
# against an install prefix, exactly the way a real downstream consumer would.
cmake_minimum_required(VERSION 3.24)
project(fuse_install_consumer LANGUAGES CXX)

find_package(fuse-coupler CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE fuse::base)
