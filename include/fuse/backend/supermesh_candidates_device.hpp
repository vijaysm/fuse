#pragma once

/// The supermesh's candidate-pair broad phase
/// (docs/dev/device-resident-arcs.md: "last the supermesh
/// intersection"), moved to a Kokkos device kernel, standing ALONGSIDE
/// `build_supermesh_weights` (src/backend/supermesh.cpp) rather than
/// replacing any of its code. The r3d/r2d device-feasibility finding is
/// summarized below.
///
/// `build_supermesh_weights`'s own brute-force box test -- "O(n_t * n_s)
/// box tests ... the honest starting point" (supermesh.cpp's own comment)
/// -- is exactly the quadratic step this header accelerates: given every
/// target simplex's and every source simplex's axis-aligned bounding box
/// (already lifted from a `MeshView` on the host, since `MeshView::
/// simplex_points` is a host-only, polymorphic, non-device-callable
/// interface -- docs/dev/native-mesh.md's own rule), `candidate_pairs`
/// returns, as a CSR, every (target, source) simplex PAIR whose boxes
/// overlap, using the IDENTICAL per-axis test `boxes_overlap` (anonymous
/// namespace, supermesh.cpp) applies: `dim` is deliberately `int`, not a
/// compile-time count, because the 2D/3D split happens the same way it
/// does there, by how many axes the caller passes in.
///
/// WHY THE CLIP ITSELF STAYS ON THE HOST (the feasibility assessment the
/// task asked for, not assumed): `build_supermesh_weights` calls exactly
/// three r3d functions for a 3D pair (`r3d_init_tet`, `r3d_clip`,
/// `r3d_reduce`) and three r2d functions for a 2D pair
/// (`r2d_init_poly`, `r2d_clip`, `r2d_reduce`). Read in full
/// (`_deps/r3d-src/src/r3d.c`/`r2d.c`) for this assessment:
///   - Every buffer on this call path is a FIXED-size stack array bounded
///     by `R3D_MAX_VERTS`/`R2D_MAX_VERTS` (compile-time constants from
///     `r3d-config.h`/`r2d.h`) -- no VLA, and the test meshes
///     (tets, quads) never approach that bound.
///   - No recursion on this call path (`r3d_clip` is a plain nested-loop
///     function; verified by reading its body, not inferred).
///   - No heap allocation on this call path: `r3d_init_brep`/`r3d_free_
///     brep` call `malloc`/`free`, but `build_supermesh_weights` never
///     calls either.
///   - The libm calls reached (`sqrt`, `pow`, `sin`, `cos`) are all in
///     functions NOT on this call path (`r3d_shift_moments`, `r3d_rotate`)
///     -- `r3d_reduce(&poly, moments, 0)` (polyorder 0, what this builder
///     asks for) does not reach them.
///   So the ALGORITHM on this call path has no structural blocker to
///   running as device code.
///
///   What DOES block it, concretely, without a patch: (1) `r3d.c`/`r2d.c`
///   are vendored, third-party, plain-C translation units with no
///   `KOKKOS_FUNCTION`/`__host__ __device__` annotation anywhere, compiled
///   today as ordinary host C -- calling them from a device kernel needs
///   every function on the path annotated, which means patching the
///   vendored copy (and keeping that patch across an `UPSTREAMS.lock`
///   bump) or hand-porting the small subset FUSE actually calls into a
///   FUSE-owned, Kokkos-native header. (2) `r3d_clip`'s debug path has an
///   `fprintf(stderr, ...)` guarded only by `#if !defined(NDEBUG)` -- a
///   Debug build (`NDEBUG` unset) is one such configuration, and `fprintf` to a
///   host `FILE*` is not device-callable, so even an annotated `r3d_clip`
///   would fail to compile for device in exactly this build configuration
///   without also stripping or reworking that guard. (3) `r3d.c`/`r2d.c`
///   are compiled as C, not C++; `nvcc`/`nvcc_wrapper` identifies device
///   translation units by language, so picking up `KOKKOS_FUNCTION` at
///   all needs the TU compiled as C++ (extern "C" linkage can stay for
///   the exported symbol names).
///
///   None of these three is a correctness or algorithmic obstacle --
///   they are packaging/build-system work on a vendored dependency, which
///   is not done here. The candidate-pair/CSR-assembly split below is
///   the fallback for this situation: accelerate what
///   does not touch r3d/r2d, leave the clip itself host-side, and report
///   the blocker precisely rather than assume it either way.

#include <cstddef>

#include <Kokkos_Core.hpp>

namespace fuse::device {

template <class Space>
KOKKOS_INLINE_FUNCTION bool boxes_overlap_on_space(const Kokkos::View<double* [3], Space>& t_lo,
                                                    const Kokkos::View<double* [3], Space>& t_hi,
                                                    int t,
                                                    const Kokkos::View<double* [3], Space>& s_lo,
                                                    const Kokkos::View<double* [3], Space>& s_hi,
                                                    int s, int dim) {
  for (int k = 0; k < dim; ++k) {
    if (t_hi(t, k) < s_lo(s, k) || s_hi(s, k) < t_lo(t, k)) return false;
  }
  return true;
}

template <class Space>
struct CandidatePairs {
  Kokkos::View<int*, Space> offsets;     ///< size n_t + 1
  Kokkos::View<int*, Space> src_index;   ///< size offsets(n_t); CSR columns
};

/// Count-then-fill CSR broad phase: `result.offsets(t)..offsets(t+1)`
/// names every source simplex index `s` (into `s_lo`/`s_hi`) whose box
/// overlaps target simplex `t`'s, dim-limited exactly as `boxes_overlap`
/// (supermesh.cpp) is. O(n_t * n_s) box tests, same complexity as the host
/// brute force -- this is a parallelization of that loop, not a different
/// (e.g. spatially indexed) algorithm, matching the task's own "the index
/// is a performance task ... replacing this before it is measured would be
/// replacing something correct with something unmeasured" posture for the
/// equivalent host code.
template <class Space>
[[nodiscard]] CandidatePairs<Space> candidate_pairs(const Kokkos::View<double* [3], Space>& t_lo,
                                                     const Kokkos::View<double* [3], Space>& t_hi,
                                                     const Kokkos::View<double* [3], Space>& s_lo,
                                                     const Kokkos::View<double* [3], Space>& s_hi,
                                                     int dim) {
  using ExecSpace = typename Space::execution_space;
  const int nt = static_cast<int>(t_lo.extent(0));
  const int ns = static_cast<int>(s_lo.extent(0));

  Kokkos::View<int*, Space> counts("fuse::device::supermesh_counts", nt);
  Kokkos::parallel_for(
      "fuse::device::supermesh_count", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const int t) {
        int cnt = 0;
        for (int s = 0; s < ns; ++s) {
          if (boxes_overlap_on_space<Space>(t_lo, t_hi, t, s_lo, s_hi, s, dim)) ++cnt;
        }
        counts(t) = cnt;
      });

  Kokkos::View<int*, Space> offsets("fuse::device::supermesh_offsets", nt + 1);
  Kokkos::parallel_scan(
      "fuse::device::supermesh_scan", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const int t, int& running, const bool final_pass) {
        if (final_pass) offsets(t) = running;
        running += counts(t);
      });
  int total = 0;
  Kokkos::parallel_reduce(
      "fuse::device::supermesh_total", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const int t, int& acc) { acc += counts(t); }, total);
  Kokkos::deep_copy(Kokkos::subview(offsets, nt), total);

  CandidatePairs<Space> out;
  out.src_index = Kokkos::View<int*, Space>(
      Kokkos::view_alloc("fuse::device::supermesh_src_index", Kokkos::WithoutInitializing), total);
  auto src_index = out.src_index;
  Kokkos::parallel_for(
      "fuse::device::supermesh_fill", Kokkos::RangePolicy<ExecSpace>(0, nt),
      KOKKOS_LAMBDA(const int t) {
        int w = offsets(t);
        for (int s = 0; s < ns; ++s) {
          if (boxes_overlap_on_space<Space>(t_lo, t_hi, t, s_lo, s_hi, s, dim)) {
            src_index(w) = s;
            ++w;
          }
        }
      });
  out.offsets = offsets;
  return out;
}

}  // namespace fuse::device
