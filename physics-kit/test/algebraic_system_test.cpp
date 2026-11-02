// physics.algebraic_system -- fuse_physics::fe::AlgebraicSystem, adapted
// from KARMA's karma_algebraic_system.h/.C. See algebraic_system.hpp's
// own header comment for how this port's assembly sink diverges from
// KARMA's own PETSc Mat/Vec construction.

#include "fuse_physics/fe/algebraic_system.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::AlgebraicSystem;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::eliminate_dirichlet_rows;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Mesh;
using fuse::linalg::Triplet;

// physics.algebraic_system.EliminateDirichletRowsStaysSymmetric -- a 1-D
// 3-point Laplacian stencil, K = [[1,-1,0],[-1,2,-1],[0,-1,1]]
// (Neumann at both ends, so K is singular -- irrelevant here, only its
// SYMMETRY and the elimination's own arithmetic are under test),
// Dirichlet-eliminated at row 0 to value 5. The textbook symmetric
// elimination this wave implements (NOT KARMA's own no-op constrain(),
// algebraic_system.hpp's own header comment) must: (a) replace row 0
// with exactly {(0,0,1)}; (b) remove the (1,0) entry that named the
// eliminated COLUMN, moving its value (-1 * 5 = -5) into rhs[1]; (c)
// leave every other entry, including (1,1) and (2,*), untouched; (d)
// stay symmetric (every kept off-diagonal (r,c) has a matching (c,r)).
TEST(AlgebraicSystemTest, EliminateDirichletRowsStaysSymmetric) {
  std::vector<Triplet> triplets{
      {0, 0, 1.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 1.0},
  };
  std::vector<double> rhs{0.0, 0.0, 0.0};
  eliminate_dirichlet_rows(triplets, 3, {0}, {5.0}, rhs);

  // Exactly one triplet for row 0, the identity.
  int row0_count = 0;
  for (const auto& t : triplets)
    if (t.row == 0) {
      ++row0_count;
      EXPECT_EQ(t.col, 0);
      EXPECT_DOUBLE_EQ(t.value, 1.0);
    }
  EXPECT_EQ(row0_count, 1);
  EXPECT_DOUBLE_EQ(rhs[0], 5.0);

  // Row 1 kept its own diagonal and its (1,2) entry, lost its (1,0)
  // entry (column 0 is dirichlet). rhs[1] <- 0 - (A(1,0) * 5) =
  // 0 - (-1 * 5) = 5 (the row-1 equation -x0 + 2x1 - x2 = 0, with x0
  // fixed at 5, becomes 2x1 - x2 = 5).
  bool has_11 = false, has_12 = false, has_10 = false;
  for (const auto& t : triplets) {
    if (t.row == 1 && t.col == 1) { has_11 = true; EXPECT_DOUBLE_EQ(t.value, 2.0); }
    if (t.row == 1 && t.col == 2) { has_12 = true; EXPECT_DOUBLE_EQ(t.value, -1.0); }
    if (t.row == 1 && t.col == 0) has_10 = true;
  }
  EXPECT_TRUE(has_11);
  EXPECT_TRUE(has_12);
  EXPECT_FALSE(has_10);
  EXPECT_DOUBLE_EQ(rhs[1], 5.0);

  // Row 2 is untouched (never named column 0).
  bool has_21 = false, has_22 = false;
  for (const auto& t : triplets) {
    if (t.row == 2 && t.col == 1) { has_21 = true; EXPECT_DOUBLE_EQ(t.value, -1.0); }
    if (t.row == 2 && t.col == 2) { has_22 = true; EXPECT_DOUBLE_EQ(t.value, 1.0); }
  }
  EXPECT_TRUE(has_21);
  EXPECT_TRUE(has_22);
  EXPECT_DOUBLE_EQ(rhs[2], 0.0);

  // Symmetry: for every kept off-diagonal (r,c), a matching (c,r) with
  // the same value exists (Higham-style: compared to EXACT equality --
  // every value here is an exact dyadic double, so no rounding floor is
  // needed; a derived-tolerance version would be needed only once a
  // non-exact value entered this arithmetic).
  for (const auto& t : triplets) {
    if (t.row == t.col) continue;
    bool mirrored = false;
    for (const auto& u : triplets)
      if (u.row == t.col && u.col == t.row) {
        mirrored = true;
        EXPECT_DOUBLE_EQ(u.value, t.value);
      }
    EXPECT_TRUE(mirrored) << "(" << t.row << "," << t.col << ") has no mirror";
  }
}

TEST(AlgebraicSystemTest, EliminateDirichletRowsRefusesSizeMismatch) {
  std::vector<Triplet> triplets;
  std::vector<double> rhs{0.0, 0.0};
  // The refusal's own message, not merely its type (lead's review of
  // round 1, 2026-11-01's own rule, applied here too).
  try {
    eliminate_dirichlet_rows(triplets, 2, {0, 1}, {1.0}, rhs);
    FAIL() << "a dirichlet_rows/values size mismatch must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("dirichlet_rows and values must have the same size"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

// physics.algebraic_system.AssembleMatrixSumsElementContributionsToAGlobalLaplacian
// -- a 1x1x1 single-Hex8-element "mesh" reduces the finite-element
// assembly loop to one element; the per-element callback contributes a
// hand-picked 8x8 "stiffness-like" matrix whose diagonal is exactly 1 at
// every dof (chosen so the assembled global matrix's own diagonal,
// read back through DistSparseMatrix::diagonal(), is checkable to EXACT
// equality without needing a derived rounding bound: summing eight
// disjoint diagonal entries of value 1.0 each, with no column overlap
// between different rows' diagonal contributions, involves no floating
// addition that could round at all).
TEST(AlgebraicSystemTest, AssembleMatrixSumsElementContributionsToAGlobalLaplacian) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);

  const auto matrix = system.assemble_matrix(
      [](fuse::physics::spatial::EntityHandle /*element*/, const std::vector<std::int64_t>& indices,
         std::vector<Triplet>& out) {
        for (const auto index : indices) out.push_back(Triplet{index, index, 1.0});
      });
  ASSERT_TRUE(matrix.ok()) << matrix.error().message;
  EXPECT_EQ(matrix.value().global_rows(), 8);
  EXPECT_EQ(matrix.value().local_rows(), 8u);

  fuse::linalg::DistVector diag = system.new_vector();
  matrix.value().diagonal(diag);
  for (std::size_t i = 0; i < diag.owned_size(); ++i) EXPECT_DOUBLE_EQ(diag.data()[i], 1.0);
}

// physics.algebraic_system.AssembleVectorSumsElementContributionsAtSharedVertices
// -- a 2x1x1 two-Hex8-element mesh shares 4 vertices between its two
// elements; each element's callback contributes 1.0 to every one of ITS
// OWN 8 dofs, so a shared vertex's assembled entry must be EXACTLY 2.0
// (two disjoint elements' contributions of exactly 1.0 each, summed --
// again no rounding floor needed for an exact-dyadic sum) and an
// unshared vertex's entry must be exactly 1.0.
TEST(AlgebraicSystemTest, AssembleVectorSumsElementContributionsAtSharedVertices) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);

  auto vector = system.new_vector();
  const auto result = system.assemble_vector(
      [](fuse::physics::spatial::EntityHandle /*element*/, const std::vector<std::int64_t>& indices,
         std::vector<std::pair<std::int64_t, double>>& out) {
        for (const auto index : indices) out.emplace_back(index, 1.0);
      },
      vector);
  ASSERT_TRUE(result.ok()) << result.error().message;

  // The shared face has 4 vertices (out of 12 total in this 2x1x1 box);
  // every other vertex belongs to exactly one element.
  int shared_count = 0, unshared_count = 0;
  for (std::size_t i = 0; i < vector.owned_size(); ++i) {
    // Exact equality: each entry is a sum of one or two 1.0 contributions, an
    // exact dyadic sum in IEEE 754 (the track had a chosen 1e-15 here; lead, 2026-11-01).
    if (vector.data()[i] == 2.0)
      ++shared_count;
    else if (vector.data()[i] == 1.0)
      ++unshared_count;
    else
      ADD_FAILURE() << "dof " << i << " has unexpected value " << vector.data()[i];
  }
  EXPECT_EQ(shared_count, 4);
  EXPECT_EQ(unshared_count, 8);
}


// physics.algebraic_system.AssembleVectorRefusesMismatchedVectorSize --
// `out` must already carry this system's own row distribution
// (`new_vector()`'s own contract); a caller passing an unrelated
// DistVector (wrong owned size) is a programming mistake, refused by
// name rather than read out of bounds.
TEST(AlgebraicSystemTest, AssembleVectorRefusesMismatchedVectorSize) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);

  fuse::linalg::DistVector wrong_size(dof_map.n_local_dofs() + 1, fuse::linalg::make_serial_exchanger(),
                                       fuse::linalg::make_serial_reducer());
  try {
    const auto result = system.assemble_vector(
        [](fuse::physics::spatial::EntityHandle, const std::vector<std::int64_t>&,
           std::vector<std::pair<std::int64_t, double>>&) {},
        wrong_size);
    (void)result;
    FAIL() << "a mismatched-size vector must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("does not match this system's own row distribution"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

}  // namespace
