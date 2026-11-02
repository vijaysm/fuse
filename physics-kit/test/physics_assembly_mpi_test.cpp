// physics-kit/test/physics_assembly_mpi_test.cpp -- item H (wave 2
// round 3): the GLOBAL assembled mass matrix over a REAL, multi-rank
// partitioned 3-D box, built through fuse_physics::fe::AlgebraicSystem's
// own collective path (DofMap/AlgebraicSystem given a real
// fortran_communicator). Registered with RANKS 2 3
// (physics-kit/test/CMakeLists.txt), build-only under FUSE_ENABLE_MPI=ON
// (host_pending has the exact mpirun commands) -- MPI cannot start in
// this sandbox (KNOWN_ISSUES.md C1), matching every other *_mpi_test.cpp
// in this tree.
//
// PARTITION. The SAME device physics_mesh_mpi_test.cpp's own
// build_local_strip uses, generalised from a 1-D line of Edge2 elements
// to a 3-D box of Hex8 elements, split along z into `world_size` equal
// slabs of `kElementsPerAxis` x `kElementsPerAxis` x `kElementsPerAxis`
// elements each (so the global box is kElementsPerAxis x
// kElementsPerAxis x (kElementsPerAxis * world_size) elements). Rank r
// owns vertex LAYERS [r*kElementsPerAxis, (r+1)*kElementsPerAxis) along
// z (kElementsPerAxis*(kElementsPerAxis+1)^2 vertices), identically to
// build_local_strip's own "every rank owns its own low end, the high end
// is a ghost unless this is the last rank" rule -- the layer at z index
// (r+1)*kElementsPerAxis is a GHOST here (owned by rank r+1) for every
// rank except the last, which owns it too (there is no rank beyond it).
//
// ASSERTIONS (both never executed, build-verified only):
//  1. GlobalMassMatrixEntrySumEqualsBoxVolume -- the SAME partition-of-
//     unity identity mass_assembly_test.cpp's own serial
//     GlobalAssembledMassMatrixEntrySumEqualsDomainVolume test checks
//     (sum_i sum_j M(i,j) = integral(1) dV = box volume), now gathered
//     across ranks: each rank sums its own OWNED rows' contribution to
//     M * 1 (a cG Lagrange basis's partition-of-unity property, so M * 1
//     IS the row-sum vector -- the same identity that test's own header
//     comment states), and fuse::linalg::Reducer::sum combines the
//     partial sums -- one collective, not a gather-then-sum-on-rank-0
//     (every rank needs the same answer to assert it, by this project's
//     own standing rule on collective symmetry).
//  2. LumpedRowSumsAgreeWithMassMatrixActionOnOnes -- the SAME identity,
//     checked row-by-row rather than summed: M * 1 at owned row i must
//     equal that row's own LUMPED (diagonal, row-summed) mass entry --
//     recomputed independently from the SAME per-element triplets
//     (summed by GLOBAL row across every element and every rank that
//     contributes to it, not read back from M itself, so this is a
//     genuine cross-check against a second construction of the same
//     quantity, not a tautology).
//
// TOLERANCE. Higham (2002) Thm 8.1 over the GLOBAL term count
// (world_size * kElementsPerAxis^3 elements * 8 * 8 basis-pair terms per
// element for assertion 1's own sum; 8 * 8 terms per OWNED row for
// assertion 2), the SAME derivation mass_assembly_test.cpp's own global
// test already uses, generalised from "mesh.n_elements()" (one rank's
// whole mesh) to "the GLOBAL element count" (summed across ranks, known
// a priori from kElementsPerAxis and world_size -- no reduction needed
// to state it).

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fuse/backend/native_mesh.hpp"
#include "fuse/field/collective_agreement.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse_physics/fe/algebraic_system.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/fe/generic_operators.hpp"
#include "fuse_physics/spatial/mesh.hpp"

#include "fuse_test_main.hpp"

namespace {

using fuse::physics::fe::AlgebraicSystem;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::element_mass_matrix;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::Mesh;
using fuse::linalg::Triplet;

constexpr unsigned kElementsPerAxis = 2;  // per rank, along every axis

/// The global box: kElementsPerAxis x kElementsPerAxis x
/// (kElementsPerAxis * world_size) Hex8 elements, split along z into
/// `world_size` slabs -- rank r's own slab, with a ghost copy of the
/// NEXT rank's bottom vertex layer (its own top boundary) unless r is
/// the last rank. Mirrors build_local_strip's own ownership convention
/// (physics_mesh_mpi_test.cpp), generalised from 1 vertex/element per
/// rank to a full 3-D slab.
Mesh build_local_slab(int rank, int world_size) {
  const unsigned nx = kElementsPerAxis, ny = kElementsPerAxis, nz_local = kElementsPerAxis;
  const unsigned px = nx + 1, py = ny + 1;
  const unsigned z_lo_global = static_cast<unsigned>(rank) * nz_local;
  const bool last_rank = (rank == world_size - 1);
  // This rank holds vertex layers z_lo_global .. z_lo_global + nz_local
  // (inclusive), nz_local + 1 layers; the TOP layer is a ghost unless
  // last_rank.
  const unsigned n_layers_local = nz_local + 1;

  fuse::native::NativeMeshBuilder builder(3);
  // local_vertex[k_local][j][i] -- the builder-local index returned by
  // add_vertex, keyed by this rank's own (k_local, j, i).
  std::vector<std::vector<std::vector<int>>> local_vertex(
      n_layers_local, std::vector<std::vector<int>>(py, std::vector<int>(px, -1)));
  for (unsigned k_local = 0; k_local < n_layers_local; ++k_local) {
    const unsigned k_global = z_lo_global + k_local;
    const bool owned_layer = (k_local < nz_local) || last_rank;  // top layer: owned only if last rank
    const int owner_rank = owned_layer ? rank : rank + 1;
    for (unsigned j = 0; j < py; ++j) {
      for (unsigned i = 0; i < px; ++i) {
        const double x = static_cast<double>(i) / nx;
        const double y = static_cast<double>(j) / ny;
        const double z = static_cast<double>(k_global) / (nz_local * static_cast<unsigned>(world_size));
        const std::int64_t gid =
            static_cast<std::int64_t>(k_global) * py * px + static_cast<std::int64_t>(j) * px + i;
        local_vertex[k_local][j][i] = builder.add_vertex(gid, x, y, z, owner_rank, owned_layer);
      }
    }
  }
  for (unsigned k_local = 0; k_local < nz_local; ++k_local) {
    const std::int64_t element_gid =
        (static_cast<std::int64_t>(rank) * nz_local + k_local) * ny * nx;  // offset by slab below
    for (unsigned j = 0; j < ny; ++j) {
      for (unsigned i = 0; i < nx; ++i) {
        const std::int64_t gid = element_gid + static_cast<std::int64_t>(j) * nx + i;
        const std::vector<int> corners = {
            local_vertex[k_local][j][i],         local_vertex[k_local][j][i + 1],
            local_vertex[k_local][j + 1][i + 1],  local_vertex[k_local][j + 1][i],
            local_vertex[k_local + 1][j][i],      local_vertex[k_local + 1][j][i + 1],
            local_vertex[k_local + 1][j + 1][i + 1], local_vertex[k_local + 1][j + 1][i]};
        builder.add_cell(fuse::native::CellType::Hex, corners, gid, rank, /*is_owned=*/true);
      }
    }
  }

  const auto built = builder.build();
  if (!built.ok()) throw std::runtime_error("NativeMeshBuilder::build failed: " + built.error().message);
  const auto view = fuse::native::NativeMeshView::create(built.value(), rank);
  if (!view.ok()) throw std::runtime_error("NativeMeshView::create failed: " + view.error().message);
  return Mesh::from_mesh_view(*view.value());
}

TEST(PhysicsAssemblyMpi, GlobalMassMatrixEntrySumEqualsBoxVolume) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const Mesh mesh = build_local_slab(rank, world_size);
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry, comm_f);
  AlgebraicSystem system(mesh, dof_map, comm_f);

  const auto basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Hex8, basis, /*quadrature_degree=*/2);

  const auto matrix = system.assemble_matrix(
      [&](EntityHandle element, const std::vector<std::int64_t>& indices, std::vector<Triplet>& out) {
        const auto local = element_mass_matrix(fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) out.push_back(Triplet{indices[i], indices[j], local[i * n + j]});
      });
  ASSERT_TRUE(matrix.ok()) << matrix.error().message;

  auto ones = system.new_vector_for(matrix.value());  // D104: the matrix's own ghost pattern
  for (std::size_t i = 0; i < ones.owned_size(); ++i) ones.data()[i] = 1.0;
  auto result = system.new_vector_for(matrix.value());  // D104: the matrix's own ghost pattern
  const auto apply_result = matrix.value().apply(ones, result);
  ASSERT_TRUE(apply_result.ok()) << apply_result.error().message;

  double local_total = 0.0;
  for (std::size_t i = 0; i < result.owned_size(); ++i) local_total += result.data()[i];
  const double global_total = system.reducer()->sum(local_total);

  const double domain_volume = 1.0 * 1.0 * 1.0;  // the global box is still the unit cube
  const std::int64_t global_elements =
      static_cast<std::int64_t>(world_size) * kElementsPerAxis * kElementsPerAxis * kElementsPerAxis;
  const std::size_t n_terms = static_cast<std::size_t>(global_elements) * 8 * 8;
  const double eps = std::numeric_limits<double>::epsilon();
  const double tolerance = static_cast<double>(n_terms) * eps * domain_volume;
  EXPECT_NEAR(global_total, domain_volume, tolerance) << "rank " << rank;
}

TEST(PhysicsAssemblyMpi, LumpedRowSumsAgreeWithMassMatrixActionOnOnes) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const Mesh mesh = build_local_slab(rank, world_size);
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry, comm_f);
  AlgebraicSystem system(mesh, dof_map, comm_f);

  const auto basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Hex8, basis, /*quadrature_degree=*/2);

  // Second, independent construction of the same lumped row sums: every
  // element's own local mass-matrix row sums, scattered by GLOBAL row
  // (summed across every element that touches that row, including
  // elements owned by OTHER ranks -- exchanged the same way
  // AlgebraicSystem::assemble_vector already exchanges a foreign-row
  // vector contribution, reused directly rather than re-implemented).
  auto lumped_scratch = system.new_vector();
  const auto lumped = system.assemble_vector(
      [&](EntityHandle element, const std::vector<std::int64_t>& indices, std::vector<std::pair<std::int64_t, double>>& out) {
        const auto local = element_mass_matrix(fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i) {
          double row_sum = 0.0;
          for (std::size_t j = 0; j < n; ++j) row_sum += local[i * n + j];
          out.push_back({indices[i], row_sum});
        }
      },
      lumped_scratch);
  ASSERT_TRUE(lumped.ok()) << lumped.error().message;

  const auto matrix = system.assemble_matrix(
      [&](EntityHandle element, const std::vector<std::int64_t>& indices, std::vector<Triplet>& out) {
        const auto local = element_mass_matrix(fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) out.push_back(Triplet{indices[i], indices[j], local[i * n + j]});
      });
  ASSERT_TRUE(matrix.ok()) << matrix.error().message;
  auto ones = system.new_vector_for(matrix.value());  // D104: the matrix's own ghost pattern
  for (std::size_t i = 0; i < ones.owned_size(); ++i) ones.data()[i] = 1.0;
  auto m_times_ones = system.new_vector_for(matrix.value());  // D104: the matrix's own ghost pattern
  const auto apply_result = matrix.value().apply(ones, m_times_ones);
  ASSERT_TRUE(apply_result.ok()) << apply_result.error().message;

  // Higham (2002) Thm 8.1 over 8*8 terms per owned row (one Hex8 row's
  // own worth of basis-pair products); every mass entry on this
  // reference element family is non-negative (mass_assembly_test.cpp's
  // own closed forms), so the row sum itself already bounds the sum S
  // the theorem's own bound multiplies.
  const double eps = std::numeric_limits<double>::epsilon();
  fuse::Result<void> local_check = fuse::Result<void>::success();
  for (std::size_t i = 0; i < m_times_ones.owned_size() && local_check.ok(); ++i) {
    const double tolerance = 64.0 * eps * std::max(std::fabs(lumped_scratch.data()[i]), 1e-300);
    if (std::fabs(m_times_ones.data()[i] - lumped_scratch.data()[i]) > tolerance) {
      local_check = fuse::Failure{"E-TEST-LUMPED-ROW-MISMATCH",
                                   "rank " + std::to_string(rank) + " row " + std::to_string(i) +
                                       ": M*1 disagreed with the independently-summed lumped row"};
    }
  }
  const auto agreed = fuse::agree_outcome(MPI_Comm_c2f(MPI_COMM_WORLD), local_check, "E-TEST-LUMPED-ROW-AGREEMENT",
                                           "E-TEST-LUMPED-ROW-COMM-FAILURE", "physics_assembly_mpi_test lumped check");
  EXPECT_TRUE(agreed.ok()) << "rank " << rank << ": " << (agreed.ok() ? "" : agreed.error().message);
}

}  // namespace
