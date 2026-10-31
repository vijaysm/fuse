# physics-kit/test/ -- the skeleton's own test registration.
#
# Calls test/CMakeLists.txt's own fuse_add_gtest directly, with its new
# SRC_DIR argument (see that function's header comment for why one is
# needed here and not in test/partition/, the only other caller outside
# test/<layer>/): fuse_add_gtest is defined while test/CMakeLists.txt
# executes, and is visible here because CMake functions, unlike variables,
# are not scoped to the directory that defines them -- only ORDER matters,
# and root CMakeLists.txt's own add_subdirectory(physics-kit) runs after
# its FUSE_ENABLE_TESTS block's add_subdirectory(test) for exactly this
# reason.
#
# NO_LAYER_LIB + explicit LIBS: the ctest name the task names is
# "physics.skeleton" (fuse::physics is the component's NAMESPACE), but the
# CMake target alias is "fuse::physics_kit" (the task's own name for it) --
# the two are deliberately different, so fuse_add_gtest's default
# "fuse::${layer}" link (which would look for a nonexistent "fuse::physics"
# target) is turned off and the real target named explicitly.
#
# This test is NOT swept by test/CMakeLists.txt's own end-of-file audit
# (file(GLOB_RECURSE ... ${CMAKE_CURRENT_SOURCE_DIR}/*/*_test.cpp), which
# globs relative to test/ itself and never descends into physics-kit/) --
# correctly so: that audit exists to catch a test/<layer>/*_test.cpp file
# that nobody registered, and physics-kit/test/ is a different tree with
# exactly one source file, registered right here, in full view.
fuse_add_gtest(physics skeleton SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)

# S8.3 wave 1 (numerics/fe, numerics/spatial): the five test rows named in
# docs/dev/physics-kit-disposition.md's porting scope, plus the spatial
# layer's own domain/mesh tests. Same NO_LAYER_LIB + explicit LIBS pattern
# as physics.skeleton above, for the same reason (fuse::physics_kit is the
# CMake target; fuse::physics is only the C++ namespace).
fuse_add_gtest(physics fe_reference_topology SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_quadrature SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_basis SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_geometry_map SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_face SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics spatial_mesh SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics spatial_domain SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
