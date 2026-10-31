#pragma once

/// Shared grid, mesh-index bookkeeping, and periodic-diffusion/backward-Euler
/// pieces for the multirate Brusselator example, ported
/// from the independently-written multirate 2D Brusselator ("the Toy":
/// /Users/mahadevan/Code/AI/Toy, its NUMERICS.md/README.md are the model and
/// algorithm reference).
///
/// MODEL (unchanged from the Toy). Periodic unit square:
///   U_t = a*Laplacian(U) + B - U + q + f
///   V_t = a*Laplacian(V) - q
///   q   = U^2*V - A*U            (A = 3.4, B = 1, a = 10 by default)
/// U0 = 22*[y(1-y)]^1.5, V0 = 27*[x(1-x)]^1.5, sampled at cell midpoints.
/// f = 5 inside the radius-0.1 circle at (0.3, 0.6) for t >= 1.1, else 0.
///
/// MESH. Unlike the Toy's hand-rolled `Grid` (include/brusselator.hpp),
/// cell geometry here comes from FUSE's native mesh backend
/// (fuse::native::NativeMeshBackend, "generated:quad:NxM" --
/// include/fuse/backend/native_backend.hpp): a genuinely different quad
/// mesh generator, not a copy of the Toy's index arithmetic. Its cell
/// global id IS `ix + iy*nx` (src/backend/native_generators.cpp's
/// `Grid::cell_gid`) and its vertices sit at `(ix/nx, iy/ny)`, so a native
/// cell's centroid is `((ix+0.5)/nx, (iy+0.5)/ny)` -- the same point the
/// Toy's `Grid::x(i)`/`y(j)` name -- but this header does not ASSUME that
/// coincidence: `NativeGrid::make` recovers the (i, j) <-> native local
/// cell index correspondence by reading back each cell's own
/// `MeshView::centroid`, so it is correct even if a future native-mesh
/// change reorders cells.
///
/// DECOMPOSITION. `NativeGrid::make` loads `backend` at
/// `backend.rank()`/`backend.num_ranks()`: `fuse::native::NativeMeshBackend`
/// slabs a generated quad grid by ROWS (`fuse::native::rank_slice(ny, rank,
/// num_ranks)` over the row index), so the view this rank sees is already
/// its own contiguous row band, owned cells only -- the native mesh carries
/// no ghost cells of its own (src/backend/native_generators.cpp never
/// builds one). `NativeGrid` exposes that band (`j_lo`/`j_hi`/`owned_rows`)
/// and a LOCAL canonical index `(j - j_lo)*nx + i` over it, plus a one-row
/// periodic halo (`ghost_pattern`) that a `SlabField` (below) exchanges
/// through an injected `GridHaloExchanger` -- never MPI directly, so this
/// header stays includable by the tier-0 (no-MPI) build. At `num_ranks ==
/// 1` the row band is the whole grid (`j_lo = 0`, `j_hi = ny`), the local
/// index equals the GLOBAL canonical index `j*nx+i` this header already
/// used, and the two halo rows are the grid's own periodic wrap rows --
/// every existing whole-grid caller (`UCore`, `VDiffusionCore`,
/// `VParticipant`) is unaffected; see docs/user/brusselator.md's grid-
/// decomposition section for the full contract and what remains undone (the
/// decomposed diffusion SOLVE is not done here).
///
/// FIELD STORAGE. Every field this example solves for is kept as a plain
/// `Eigen::VectorXd` in CANONICAL order (`index(i,j) = j*nx+i`, matching
/// the Toy exactly) for the periodic-Laplacian backward-Euler solve, which
/// reuses Eigen::SparseCholesky exactly as the Toy does. The reaction rate
/// `q = U^2*V - A*U` -- the one genuinely elementwise, embarrassingly
/// parallel kernel in this model -- is instead computed on a
/// `Kokkos::View<double*>` with `Kokkos::parallel_for` over the canonical
/// cell index space (`reaction_kokkos`, below): this is the Kokkos
/// port, and it is what makes `ctest -L omp` load-
/// bearing for this example (examples/CMakeLists.txt registers an
/// `omp_identical` leg exactly as mms_pair/dn_heat do). FUSE's remap
/// weights (fuse::RemapWeights, built over the NATIVE mesh's own local
/// cell numbering) are applied through `apply_weights` after permuting a
/// canonical-order field into native order with `NativeGrid::to_native`/
/// `to_canonical` -- plain gather loops, not a Kokkos kernel; see
/// docs/user/brusselator.md "What is and is not Kokkos here" for why that
/// split was chosen and what it costs.

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"

namespace brusselator {

using Vector = Eigen::VectorXd;
using Sparse = Eigen::SparseMatrix<double>;

/// The grid ghost/halo/reduction types are the `fuse::linalg` types
/// themselves (`GhostPattern`/`GhostExchanger`/`Reducer`,
/// include/fuse/linalg/comm.hpp), aliased under the `Grid*` names used
/// throughout this directory. `NativeGrid`'s own public shape
/// (`ghost_pattern()`, `SlabField::update_ghosts`/`owned_mean`) follows
/// `comm.hpp`'s own header comment on `GhostExchanger::update`.
/// `make_serial_exchanger()`/`make_serial_reducer()` give the one-rank
/// case; an MPI pair comes from
/// `fuse::linalg::make_mpi_exchanger`/`make_mpi_reducer` in the
/// participants (examples/brusselator/participants), never from a type
/// defined here.
using GridGhostPattern = fuse::linalg::GhostPattern;
using GridHaloExchanger = fuse::linalg::GhostExchanger;
using GridReducer = fuse::linalg::Reducer;

/// A cell-centred nx*ny periodic grid backed by FUSE's native mesh, plus
/// the canonical (Toy-matching) index that lets the rest of this example's
/// Eigen solves stay a straight port of the Toy's own linear algebra.
class NativeGrid {
 public:
  NativeGrid(const NativeGrid&) = delete;
  NativeGrid& operator=(const NativeGrid&) = delete;
  NativeGrid(NativeGrid&&) = default;
  NativeGrid& operator=(NativeGrid&&) = default;

  /// Loads a "generated:quad:nxXny" native mesh named `name` onto `backend`
  /// (a SHARED backend: U and V's meshes must live on the SAME
  /// `fuse::MeshBackend` instance for `fuse::build_weights` to resolve
  /// both sides of a remap plan through it -- see main.cpp) and recovers
  /// the canonical index correspondence. Errors:
  ///   E-BRUSS-MESH-LOAD   the native backend refused the generated spec
  ///   E-BRUSS-MESH-COUNT  entity_count(2) != nx*ny
  ///   E-BRUSS-MESH-INDEX  a cell's centroid does not land on an (i, j)
  ///                       of this grid to within half a cell width, or
  ///                       two cells mapped to the same (i, j)
  [[nodiscard]] static fuse::Result<NativeGrid> make(fuse::native::NativeMeshBackend& backend,
                                                     const std::string& name, int nx, int ny);

  [[nodiscard]] int nx() const noexcept { return nx_; }
  [[nodiscard]] int ny() const noexcept { return ny_; }
  /// The GLOBAL grid size, nx*ny -- unchanged meaning from before
  /// decomposition landed (every existing caller at num_ranks()==1 reads
  /// this as "the whole grid", which it still is: j_lo()==0, j_hi()==ny()
  /// there). Not this rank's owned count; see `owned_size()` for that.
  [[nodiscard]] int size() const noexcept { return nx_ * ny_; }
  [[nodiscard]] double x(int i) const noexcept { return (i + 0.5) / static_cast<double>(nx_); }
  [[nodiscard]] double y(int j) const noexcept { return (j + 0.5) / static_cast<double>(ny_); }
  /// GLOBAL canonical index `j*nx+i`, `j` a GLOBAL row -- the whole-grid
  /// convention every existing caller (UCore, VDiffusionCore, VParticipant,
  /// the free functions below) already uses, unaffected by decomposition.
  [[nodiscard]] int index(int i, int j) const noexcept { return j * nx_ + i; }

  [[nodiscard]] fuse::MeshId mesh_id() const noexcept { return mesh_id_; }

  /// --- decomposition over this component's own ranks -----------------
  ///
  /// `backend.rank()`/`backend.num_ranks()` at `make()` time, kept so a
  /// caller can tell whether it is looking at a slab or the whole grid
  /// without re-deriving it from `j_lo`/`j_hi`.
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int num_ranks() const noexcept { return num_ranks_; }
  /// This rank's row band, GLOBAL row indices, half-open: `[j_lo(), j_hi())`.
  /// `fuse::native::rank_slice(ny, rank, num_ranks)` -- the same split the
  /// native generator itself used to decide which cells to build, so the
  /// band named here and the band the backend actually populated agree by
  /// construction, not by re-derivation.
  [[nodiscard]] int j_lo() const noexcept { return j_lo_; }
  [[nodiscard]] int j_hi() const noexcept { return j_hi_; }
  [[nodiscard]] int owned_rows() const noexcept { return j_hi_ - j_lo_; }
  /// This rank's owned cell count, nx*owned_rows() -- equals `size()` at
  /// num_ranks()==1 (`j_lo()==0`, `j_hi()==ny()`), strictly smaller than it
  /// when this component is decomposed over more than one rank.
  [[nodiscard]] int owned_size() const noexcept { return nx_ * owned_rows(); }
  /// LOCAL canonical index `(j - j_lo())*nx + i` of a cell whose GLOBAL row
  /// is `j` (`j_lo() <= j < j_hi()`) -- what `SlabField`'s owned block and
  /// `native_of_canonical_` are addressed by. Equals `index(i, j)` whenever
  /// `j_lo() == 0` (in particular at num_ranks()==1), by construction.
  [[nodiscard]] int local_index(int i, int j) const noexcept { return (j - j_lo_) * nx_ + i; }

  /// Which rank owns GLOBAL row `j` of this grid's ny rows, by the SAME
  /// `fuse::native::rank_slice` split `make()` used for its own band -- a
  /// caller never has to reverse-engineer the split by hand.
  [[nodiscard]] int owner_of_row(int j) const {
    for (int r = 0; r < num_ranks_; ++r) {
      const auto [lo, hi] = fuse::native::rank_slice(static_cast<std::size_t>(ny_), r, num_ranks_);
      if (j >= static_cast<int>(lo) && j < static_cast<int>(hi)) return r;
    }
    return num_ranks_ > 0 ? num_ranks_ - 1 : 0;
  }

  /// This grid's row decomposition, restated as a CELL-indexed
  /// `fuse::linalg::RowOwnership` -- `DistSparseMatrix::from_triplets`'s
  /// own convention (dist_sparse_matrix.hpp): `row_lo[r]` ascending, one
  /// entry per participant, ranging over the `nx()*ny()` CELL rows of the
  /// periodic-Laplacian system this grid's own diffusion solve assembles,
  /// NOT over the `ny()` GRID rows `rank_slice` itself splits. `row_lo[r]
  /// = rank_slice(ny, r, num_ranks).first * nx` for every rank `r` --
  /// derived with NO communication, the same `fuse::native::rank_slice`
  /// call every rank already makes locally for its own band in `make()`,
  /// just repeated here for every OTHER rank too (a pure function of
  /// `nx()`/`ny()`/`num_ranks()`, identical on every rank by construction,
  /// not a value one rank learns from another). For `SchurSlabSolver`'s
  /// OWN row-ownership argument, which reinterprets the SAME
  /// `fuse::linalg::RowOwnership` TYPE over `ny()` GRID rows instead
  /// (schur_slab.hpp's own doc comment: "reinterpreted here over the
  /// grid's `ny` ROWS rather than its `nx*ny` CELLS") see
  /// `row_ownership_by_grid_row()` below -- the two are NOT
  /// interchangeable despite sharing a type, a genuine convention
  /// difference between `DistSparseMatrix` and `SchurSlabSolver`, not a
  /// copy-paste duplication.
  /// `SchurSlabSolver::factor`'s own `FUSE_ASSERT` catches the mismatch
  /// (`row_ownership.row_lo[...]` disagreeing with the `j_lo` argument by
  /// exactly a factor of `nx`).
  [[nodiscard]] fuse::linalg::RowOwnership row_ownership() const {
    fuse::linalg::RowOwnership owners;
    owners.row_lo.resize(static_cast<std::size_t>(num_ranks_));
    for (int r = 0; r < num_ranks_; ++r) {
      const auto [lo, hi] = fuse::native::rank_slice(static_cast<std::size_t>(ny_), r, num_ranks_);
      (void)hi;
      owners.row_lo[static_cast<std::size_t>(r)] = static_cast<std::int64_t>(lo) * nx_;
    }
    return owners;
  }

  /// `SchurSlabSolver::factor`'s own row-ownership argument: the SAME
  /// split as `row_ownership()` above, but over `ny()` GRID rows (plain
  /// `rank_slice(ny, r, num_ranks).first`, no `* nx` scaling) -- see that
  /// method's own doc comment for why the two differ.
  [[nodiscard]] fuse::linalg::RowOwnership row_ownership_by_grid_row() const {
    fuse::linalg::RowOwnership owners;
    owners.row_lo.resize(static_cast<std::size_t>(num_ranks_));
    for (int r = 0; r < num_ranks_; ++r) {
      const auto [lo, hi] = fuse::native::rank_slice(static_cast<std::size_t>(ny_), r, num_ranks_);
      (void)hi;
      owners.row_lo[static_cast<std::size_t>(r)] = static_cast<std::int64_t>(lo);
    }
    return owners;
  }

  /// The one periodic row of ghosts above and below this rank's own band,
  /// GhostPattern-shaped (`fuse::linalg::GhostPattern`'s planned contract:
  /// one `owner_rank`/`remote_local_index` pair per ghost, in ghost order).
  /// Order: the `nx()` cells of the row BELOW this band first (GLOBAL row
  /// `(j_lo()-1+ny()) % ny()`), then the `nx()` cells of the row ABOVE it
  /// (GLOBAL row `j_hi() % ny()`) -- `SlabField::ghosts()`'s own layout.
  ///
  /// Every row-band split is contiguous, so the row immediately below this
  /// band is always the LAST row of whichever band owns it (itself, when
  /// `num_ranks()==1` and it wraps to `ny()-1`), and the row immediately
  /// above is always the FIRST row of whichever band owns it (itself,
  /// wrapping to row 0) -- which is why this needs no communication to
  /// build: each ghost's `remote_local_index` is `(owned_rows(owner)-1)*nx+i`
  /// for the row below, `i` for the row above, with no dependence on where
  /// the owner's own band starts.
  [[nodiscard]] GridGhostPattern ghost_pattern() const {
    GridGhostPattern pattern;
    pattern.owner_rank.resize(static_cast<std::size_t>(2 * nx_));
    pattern.remote_local_index.resize(static_cast<std::size_t>(2 * nx_));
    const int below_row = ((j_lo_ - 1) % ny_ + ny_) % ny_;
    const int above_row = j_hi_ % ny_;
    const int below_owner = owner_of_row(below_row);
    const int above_owner = owner_of_row(above_row);
    const auto [below_lo, below_hi] =
        fuse::native::rank_slice(static_cast<std::size_t>(ny_), below_owner, num_ranks_);
    const int below_owned_rows = static_cast<int>(below_hi - below_lo);
    for (int i = 0; i < nx_; ++i) {
      pattern.owner_rank[static_cast<std::size_t>(i)] = below_owner;
      pattern.remote_local_index[static_cast<std::size_t>(i)] =
          static_cast<std::int64_t>((below_owned_rows - 1) * nx_ + i);
      pattern.owner_rank[static_cast<std::size_t>(nx_ + i)] = above_owner;
      pattern.remote_local_index[static_cast<std::size_t>(nx_ + i)] = static_cast<std::int64_t>(i);
    }
    return pattern;
  }

  /// The native-mesh local cell index corresponding to LOCAL canonical
  /// index `c = (j - j_lo())*nx+i`. Exposed for verification code (entry-
  /// by-entry weight comparisons against an oracle built in canonical
  /// order) -- ordinary field transfer should use `to_native`/`to_canonical`
  /// below instead. Equals the GLOBAL-canonical-indexed lookup whenever
  /// `j_lo() == 0`.
  [[nodiscard]] int native_index_of(int c) const noexcept { return native_of_canonical_[static_cast<std::size_t>(c)]; }

  /// (docs/dev/restart.md Section 2.2-2.3): the
  /// `fuse::MeshView::global_id(2, i)` of the native cell at LOCAL
  /// canonical index `c = (j - j_lo())*nx+i`, i.e. the SAME `cell_gid =
  /// ix + iy*nx` (`src/backend/native_generators.cpp`'s own
  /// `Grid::cell_gid`, this file's own top comment) FUSE's quad
  /// generator assigns -- contiguous and unique over the WHOLE
  /// (un-partitioned) logical grid, independent of how many ranks this
  /// grid is sliced across (the per-field-family table's own entry for
  /// "Brusselator example"). Returned as `std::int64_t` (the id type
  /// every `fuse::RestartWriter`/`RestartReader` call uses), widened
  /// from `global_id()`'s own `std::size_t` -- exact for this generator
  /// (`cell_gid` never exceeds `nx*ny`, far inside `int64_t`'s range for
  /// any grid size this example builds). `global_id_of_canonical_` is
  /// filled once, in `make()`, from the SAME `view->global_id(2,
  /// native_idx)` call already sitting beside the `view->centroid(2,
  /// native_idx)` read that resolves `native_of_canonical_` -- no
  /// second mesh walk.
  [[nodiscard]] std::int64_t global_id(int c) const noexcept {
    return global_id_of_canonical_[static_cast<std::size_t>(c)];
  }

  /// Gathers an OWNED, LOCAL-canonical-order field (size `owned_size()`)
  /// into native-mesh local-index order (what fuse::RemapWeights
  /// addresses). At num_ranks()==1, `owned_size() == size()` and LOCAL ==
  /// GLOBAL canonical order, so this is byte-identical to before
  /// decomposition landed.
  void to_native(const Vector& canonical, std::vector<double>* native) const {
    native->resize(static_cast<std::size_t>(owned_size()));
    for (int c = 0; c < owned_size(); ++c) (*native)[static_cast<std::size_t>(native_of_canonical_[static_cast<std::size_t>(c)])] = canonical[c];
  }
  /// Scatters a native-mesh-ordered field back into OWNED, LOCAL-canonical
  /// order.
  void to_canonical(const std::vector<double>& native, Vector* canonical) const {
    canonical->resize(owned_size());
    for (int c = 0; c < owned_size(); ++c) (*canonical)[c] = native[static_cast<std::size_t>(native_of_canonical_[static_cast<std::size_t>(c)])];
  }

 private:
  NativeGrid(int nx, int ny) : nx_(nx), ny_(ny) {}

  int nx_ = 0, ny_ = 0;
  int rank_ = 0, num_ranks_ = 1;
  int j_lo_ = 0, j_hi_ = 0;  ///< this rank's GLOBAL row band, [j_lo_, j_hi_)
  fuse::MeshId mesh_id_{};
  std::vector<int> native_of_canonical_;  ///< indexed by LOCAL canonical index (j-j_lo_)*nx+i
  /// Indexed the SAME way as `native_of_canonical_`, the
  /// native mesh's own `MeshView::global_id(2, native_of_canonical_[c])`
  /// for LOCAL canonical index `c`. See `global_id()`'s own comment.
  std::vector<std::int64_t> global_id_of_canonical_;
};

inline fuse::Result<NativeGrid> NativeGrid::make(fuse::native::NativeMeshBackend& backend,
                                                 const std::string& name, int nx, int ny) {
  NativeGrid g(nx, ny);
  g.rank_ = backend.rank();
  g.num_ranks_ = backend.num_ranks();
  std::tie(g.j_lo_, g.j_hi_) = [&] {
    const auto [lo, hi] =
        fuse::native::rank_slice(static_cast<std::size_t>(ny), g.rank_, g.num_ranks_);
    return std::pair<int, int>(static_cast<int>(lo), static_cast<int>(hi));
  }();
  const std::string source =
      "generated:quad:" + std::to_string(nx) + "x" + std::to_string(ny);
  auto loaded = backend.load(fuse::MeshDecl{.name = name, .source = source});
  if (!loaded.ok()) {
    return fuse::Failure{"E-BRUSS-MESH-LOAD",
                        "NativeGrid::make('" + name + "', " + source +
                            "): backend.load failed: " + loaded.error().code + ": " +
                            loaded.error().message};
  }
  auto mid = backend.mesh_id(name);
  if (!mid.ok()) return mid.error();
  g.mesh_id_ = mid.value();
  auto view_r = backend.view(g.mesh_id_);
  if (!view_r.ok()) return view_r.error();
  const fuse::MeshView* view = view_r.value();
  const std::size_t n2 = view->entity_count(2);
  const std::size_t expected_owned =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(g.owned_rows());
  if (n2 != expected_owned) {
    return fuse::Failure{"E-BRUSS-MESH-COUNT",
                        "NativeGrid::make('" + name + "'): native mesh reports " +
                            std::to_string(n2) + " dimension-2 entities on rank " +
                            std::to_string(g.rank_) + " of " + std::to_string(g.num_ranks_) +
                            ", expected " + std::to_string(expected_owned) + " (" +
                            std::to_string(nx) + " x rows [" + std::to_string(g.j_lo_) + ", " +
                            std::to_string(g.j_hi_) + "))"};
  }
  g.native_of_canonical_.assign(expected_owned, -1);
  g.global_id_of_canonical_.assign(expected_owned, -1);
  std::vector<int> seen(expected_owned, -1);
  for (std::size_t native_idx = 0; native_idx < n2; ++native_idx) {
    const fuse::Point c = view->centroid(2, native_idx);
    const int i = static_cast<int>(std::lround(c.x * nx - 0.5));
    const int j = static_cast<int>(std::lround(c.y * ny - 0.5));
    if (i < 0 || i >= nx || j < g.j_lo_ || j >= g.j_hi_ ||
        std::abs(c.x - g.x(i)) > 0.25 / nx || std::abs(c.y - g.y(j)) > 0.25 / ny) {
      return fuse::Failure{"E-BRUSS-MESH-INDEX",
                          "NativeGrid::make('" + name + "'): native cell " +
                              std::to_string(native_idx) + " centroid (" +
                              std::to_string(c.x) + ", " + std::to_string(c.y) +
                              ") does not land on a unique (i,j) of this rank's own row band "
                              "[" + std::to_string(g.j_lo_) + ", " + std::to_string(g.j_hi_) +
                              ") of a " + std::to_string(nx) + "x" + std::to_string(ny) +
                              " periodic grid"};
    }
    const int canonical = g.local_index(i, j);
    if (seen[static_cast<std::size_t>(canonical)] != -1) {
      return fuse::Failure{"E-BRUSS-MESH-INDEX",
                          "NativeGrid::make('" + name + "'): two native cells mapped to the "
                          "same local canonical index " + std::to_string(canonical)};
    }
    seen[static_cast<std::size_t>(canonical)] = 1;
    g.native_of_canonical_[static_cast<std::size_t>(canonical)] = static_cast<int>(native_idx);
    g.global_id_of_canonical_[static_cast<std::size_t>(canonical)] =
        static_cast<std::int64_t>(view->global_id(2, native_idx));
  }
  return std::move(g);
}

/// The periodic 5-point Laplacian on a canonical-order nx*ny grid --
/// identical stencil and identical sign convention to the Toy's
/// `bruss::laplacian` (include/brusselator.hpp / src/brusselator.cpp),
/// re-derived here rather than `#include`d from a read-only sibling repo.
[[nodiscard]] inline Sparse periodic_laplacian(int nx, int ny) {
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 5);
  const double xx = double(nx) * nx, yy = double(ny) * ny;
  const auto index = [nx](int i, int j) { return j * nx + i; };
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int row = index(i, j);
      entries.emplace_back(row, row, -2 * (xx + yy));
      entries.emplace_back(row, index((i + 1) % nx, j), xx);
      entries.emplace_back(row, index((i + nx - 1) % nx, j), xx);
      entries.emplace_back(row, index(i, (j + 1) % ny), yy);
      entries.emplace_back(row, index(i, (j + ny - 1) % ny), yy);
    }
  }
  Sparse result(nx * ny, nx * ny);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

/// U0 = 22*[y(1-y)]^1.5 (is_u=true), V0 = 27*[x(1-x)]^1.5 (is_u=false),
/// sampled at cell midpoints -- the Toy's `initial_condition`.
[[nodiscard]] inline Vector initial_condition(bool is_u, int nx, int ny) {
  Vector result(nx * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double s = is_u ? (j + 0.5) / static_cast<double>(ny) : (i + 0.5) / static_cast<double>(nx);
      result[j * nx + i] = (is_u ? 22.0 : 27.0) * std::pow(s * (1 - s), 1.5);
    }
  }
  return result;
}

/// 5 inside the radius-0.1 circle centred at (0.3, 0.6), else 0 -- the
/// Toy's `forcing_mask`, evaluated on the U grid only (V has no forcing).
[[nodiscard]] inline Vector forcing_mask(int nx, int ny) {
  Vector result(nx * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) / static_cast<double>(nx);
      const double y = (j + 0.5) / static_cast<double>(ny);
      result[j * nx + i] = (std::pow(x - 0.3, 2) + std::pow(y - 0.6, 2) <= 0.01) ? 5.0 : 0.0;
    }
  }
  return result;
}

/// `initial_condition(is_u, nx, ny)` restricted to the GLOBAL row band
/// `[j_lo, j_lo+owned_rows)`, stored in LOCAL canonical order
/// `(j-j_lo)*nx+i` -- the explicit (no-solve, no-stencil) part of building
/// a decomposed component's own initial state. Same formula, same operand
/// order per row as the whole-grid function; at `j_lo==0`,
/// `owned_rows==ny` the two loops visit identical (i, j, s) triples in the
/// identical order, so the result is byte-identical to
/// `initial_condition(is_u, nx, ny)` in that case (measured, not assumed --
/// see the grid-decomposition test).
[[nodiscard]] inline Vector initial_condition_owned(bool is_u, int nx, int ny, int j_lo,
                                                    int owned_rows) {
  Vector result(nx * owned_rows);
  for (int lj = 0; lj < owned_rows; ++lj) {
    const int j = j_lo + lj;
    for (int i = 0; i < nx; ++i) {
      const double s = is_u ? (j + 0.5) / static_cast<double>(ny) : (i + 0.5) / static_cast<double>(nx);
      result[lj * nx + i] = (is_u ? 22.0 : 27.0) * std::pow(s * (1 - s), 1.5);
    }
  }
  return result;
}

/// `forcing_mask(nx, ny)` restricted the same way `initial_condition_owned`
/// restricts `initial_condition` -- same per-row formula and operand order,
/// byte-identical to the whole-grid function at `j_lo==0`, `owned_rows==ny`.
[[nodiscard]] inline Vector forcing_mask_owned(int nx, int ny, int j_lo, int owned_rows) {
  Vector result(nx * owned_rows);
  for (int lj = 0; lj < owned_rows; ++lj) {
    const int j = j_lo + lj;
    const double y = (j + 0.5) / static_cast<double>(ny);
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) / static_cast<double>(nx);
      result[lj * nx + i] = (std::pow(x - 0.3, 2) + std::pow(y - 0.6, 2) <= 0.01) ? 5.0 : 0.0;
    }
  }
  return result;
}

/// Exact activated duration of the forcing over `[t_begin, t_end)`:
/// `max(0, t_end - max(t_begin, 1.1))`, the Toy's own closed form
/// (src/brusselator.cpp `Component::advance`), so a step that straddles
/// the t=1.1 switch integrates it exactly regardless of window alignment.
[[nodiscard]] inline double forcing_active_duration(double t_begin, double t_end) noexcept {
  return std::max(0.0, t_end - std::max(t_begin, 1.1));
}

/// Core of `reaction_kokkos`: `q[m] = u[m]^2*v[m] - A*u[m]` over `n`
/// pointer-addressed entries -- the same `Kokkos::parallel_for` either way,
/// so it runs equally over a whole canonical vector (`reaction_kokkos`,
/// below) or over just a `NativeGrid`'s OWNED cells (`SlabField::owned()`,
/// size `owned_size()`): the reaction is pointwise, so it needs no ghost
/// value in either case. `u`, `v`, `q` must each have at least `n` entries.
inline void reaction_kokkos_span(const double* u, const double* v, double* q, int n) {
  Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u_view(
      u, n);
  Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_view(
      v, n);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> q_view(
      q, n);
  Kokkos::parallel_for(
      "brusselator::reaction",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const int m) {
        q_view(m) = u_view(m) * u_view(m) * v_view(m) - 3.4 * u_view(m);
      });
}

/// q = U^2*V_on_u_grid - A*U, componentwise, computed on a
/// `Kokkos::View<double*>` with `Kokkos::parallel_for` over the canonical
/// cell index space -- the reaction kernel this port moves onto Kokkos
/// (see the file header). `A = 3.4`, matching the Toy exactly
/// (src/brusselator.cpp `reaction`, which hardcodes the same constant).
/// `u` and `v_on_u_grid` must both have the same size; `q_out` is resized
/// and overwritten. A thin wrapper over `reaction_kokkos_span` (the same
/// Kokkos call, unchanged), kept so every existing whole-grid caller
/// stays byte-identical.
inline void reaction_kokkos(const Vector& u, const Vector& v_on_u_grid, Vector* q_out) {
  const int n = static_cast<int>(u.size());
  if (v_on_u_grid.size() != n) {
    FUSE_THROW(fuse::InvalidArgument, "{}",
              std::string("reaction_kokkos: size mismatch, u has ") + std::to_string(n) +
                  " entries, v_on_u_grid has " + std::to_string(v_on_u_grid.size()));
  }
  q_out->resize(n);
  reaction_kokkos_span(u.data(), v_on_u_grid.data(), q_out->data(), n);
}

/// An owned-plus-ghost field over one rank's row band: `owned_size()`
/// entries this rank is responsible for, then `ghost_size()` copies of
/// neighbouring ranks' boundary rows -- the same contiguous "owned block
/// then ghosts" layout `fuse::linalg::DistVector` uses
/// (`data()[0..owned_size())` owned, `[owned_size()..size())` ghosts), so
/// adopting that type later moves this storage, not this class's callers.
class SlabField {
 public:
  SlabField(std::size_t owned_n, std::size_t ghost_n)
      : owned_n_(owned_n), data_(Vector::Zero(static_cast<int>(owned_n + ghost_n))) {}

  [[nodiscard]] double* data() noexcept { return data_.data(); }
  [[nodiscard]] const double* data() const noexcept { return data_.data(); }
  [[nodiscard]] double* owned() noexcept { return data_.data(); }
  [[nodiscard]] const double* owned() const noexcept { return data_.data(); }
  [[nodiscard]] double* ghosts() noexcept { return data_.data() + owned_n_; }
  [[nodiscard]] const double* ghosts() const noexcept { return data_.data() + owned_n_; }
  [[nodiscard]] std::size_t owned_size() const noexcept { return owned_n_; }
  [[nodiscard]] std::size_t ghost_size() const noexcept {
    return static_cast<std::size_t>(data_.size()) - owned_n_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return static_cast<std::size_t>(data_.size()); }

  /// Copies `src`'s `n` entries into the owned block (`n` must equal
  /// `owned_size()`); the usual way to seed a `SlabField` from
  /// `initial_condition_owned`/`forcing_mask_owned`'s `Vector` result.
  void assign_owned(const Vector& src) {
    for (std::size_t k = 0; k < owned_n_; ++k) data_[static_cast<int>(k)] = src[static_cast<int>(k)];
  }

  /// Refreshes the ghost block by calling through to the injected
  /// exchanger with this field's own owned/ghost buffers -- never
  /// resolves the pattern itself (the exchanger was built from one, at
  /// construction; see `GridHaloExchanger`'s own comment).
  [[nodiscard]] fuse::Result<void> update_ghosts(const GridHaloExchanger& exchanger) {
    return exchanger.update(owned(), owned_size(), ghosts(), ghost_size());
  }

  /// Global mean over OWNED cells only: this rank's own partial sum,
  /// reduced across the component's own communicator by `reducer`, divided
  /// by the GLOBAL cell count `global_n` -- never by a locally reduced
  /// count, so a caller that passes the wrong `global_n` is a caller bug
  /// this cannot mask (the requirement this exists to satisfy: "every
  /// mean/sum as owned partial + injected global reduction").
  [[nodiscard]] double owned_mean(const GridReducer& reducer, std::size_t global_n) const {
    double local_sum = 0.0;
    for (std::size_t k = 0; k < owned_n_; ++k) local_sum += data_[static_cast<int>(k)];
    return reducer.sum(local_sum) / static_cast<double>(global_n);
  }

 private:
  std::size_t owned_n_;
  Vector data_;
};

}  // namespace brusselator
