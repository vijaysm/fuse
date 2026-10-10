#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_SHIM_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_SHIM_H_

#include <stdint.h>

/* Time windows track (Conformance, Unit 2): the tiny extern "C" shim both
 * the C main (u_main.c) and the Fortran main (v_main.f90, through
 * iso_c_binding's own bind(C) convention -- identical calling convention,
 * no separate Fortran-specific entry points needed) call. Wraps
 * UCore/VDiffusionCore/VParticipant (../species_core.hpp) -- the SAME
 * classes both fuse::Component's own in-process drivers
 * (species_component.hpp) and this shim call, so the reaction-diffusion
 * solve (and, for V, the by-hand remap -- VParticipant's own comment
 * has the full account of why V still needs one) exists exactly once.
 * Opaque handle, C-ABI-only types (no Eigen, no fuse:: types) cross this
 * boundary -- a plain error-buffer convention (bruss_*_step returns 0
 * on success, nonzero on failure with the message readable via
 * bruss_*_last_error()) rather than fuse::Result<void>, which is not
 * standard-layout and cannot cross a C ABI at all. */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bruss_u_handle bruss_u_handle;
typedef struct bruss_v_handle bruss_v_handle;

/* U_t = a*Lap(U) + B - U + q + f (species_core.hpp's own UCore). `nx`/`ny`
 * are U's own native grid shape; `initial_v_on_u` (length nx*ny) is the
 * initial P_VU(V) mean on U's own grid -- the SAME seed UComponent's own
 * constructor uses, so the very first window's own reaction has
 * something real to compute from. Returns NULL on allocation failure
 * only (never on a physics error -- step() reports those). */
/* The grid and physics constants of brusselator_grid.h, for a caller that
 * cannot include a C header (v_main.f90). One source of truth: see that
 * header's own comment for what happened when there were three. */
void bruss_grid_dims(int* nx_u, int* ny_u, int* nx_v, int* ny_v);
double bruss_diffusion(void);

/* Physics track: THIS rank's own owned cell count of an `nx`x`ny`
 * periodic grid split `fuse::native::rank_slice`'s own way over
 * `num_ranks` ranks (brusselator_physics.hpp's own `NativeGrid::
 * owned_size()`) -- what a decomposed participant must register
 * (bruss_u_create()'s own `initial_v_on_u`, and the field storage
 * u_main.c/v_main.f90 register through the ABI) instead of the GLOBAL
 * `nx*ny` every single-rank caller uses. Returns -1 if `nx`/`ny` cannot
 * be laid out as a periodic grid at this `rank`/`num_ranks` (the same
 * conditions `NativeGrid::make`'s own E-BRUSS-MESH-* refusals cover) --
 * there is no handle yet to report a message through, mirroring
 * bruss_u_create()'s own NULL-return convention. */
int bruss_owned_size(int nx, int ny, int rank, int num_ranks);

/* `whole[j*nx+i]` (GLOBAL canonical order, length nx*ny) restricted to
 * this rank's own owned band, in LOCAL canonical order (length
 * bruss_owned_size(nx, ny, rank, num_ranks)) -- what a decomposed
 * participant needs to seed its OWN `initial_v_on_u` (bruss_u_create)
 * from a whole-grid-sized value (u_main.c's own "throwaway V, built at
 * rank 0 of 1, to seed the first window's reaction" trick hands back a
 * whole U-grid-sized array regardless of U's own decomposition -- this
 * function is how each U rank then takes only its own slice of it).
 * Returns nonzero (no message: no handle exists yet) if `nx`/`ny` cannot
 * be laid out as a periodic grid at this `rank`/`num_ranks`. */
int bruss_owned_slice(const double* whole, int nx, int ny, int rank, int num_ranks,
                      double* owned_out);

/* This component's CURRENT `provided` value (length size()) -- right after
 * create(), the INITIAL one: q at t=0 for U, P_VU(V) at t=0 for V. A
 * participant copies it into the storage it registers for its export
 * BEFORE fuse_initialize(), because the Runner primes the first window's
 * t=0 sample from whatever that storage holds at initialize -- the
 * in-process components hold their constructor's own value there, and an
 * externally driven participant must put the same value there by hand
 * (test/verification/brusselator_participants_test.cpp does exactly this;
 * the standalone programs did not, and the first window's exchange was
 * then measured 40% short on the host, 2026-10-09). */
void bruss_u_provided(const bruss_u_handle* h, double* out);
void bruss_v_provided(const bruss_v_handle* h, double* out);

/* Physics track (closing the Grid track's own solver seam): `rank`/
 * `num_ranks` are THIS component's own placement -- 0/1 for every
 * EXISTING caller (the tier-0 harness, the in-process example path),
 * which must keep passing exactly that; there is no default at the C
 * ABI. `comm_f` is an `MPI_Comm_c2f` handle of this component's OWN
 * physics-only communicator (the sub-communicator a decomposed caller
 * forms by `MPI_Comm_split`ting its local world on is_physics_rank --
 * u_main.c's own header comment has the full account), consulted ONLY
 * when `num_ranks > 1`; ignored (any value, including 0) otherwise.
 * `solver`: "auto" (default: direct at num_ranks()==1, schur at >1),
 * "direct", "schur", "pcg", or "gmres" (species_core.hpp's own
 * `parse_solver_kind`) -- an unrecognised name is refused the same way
 * an allocation failure is, by returning NULL (no handle exists yet to
 * report a message through). */
bruss_u_handle* bruss_u_create(int nx, int ny, double diffusion, double A,
                               double B, int forcing,
                               const double* initial_v_on_u,
                               int rank, int num_ranks, int64_t comm_f,
                               const char* solver);
void bruss_u_destroy(bruss_u_handle* h);
int bruss_u_size(const bruss_u_handle* h);

/* One step [t_begin, t_end), given as EXACT rationals -- the same
 * fuse_exact_time{numerator, denominator} fuse_step_report carries, passed
 * straight through with no arithmetic of the caller's own. NOT two
 * doubles: UCore's own dt-scaled bookkeeping (species_core.hpp's own
 * comment on UCore::step()) needs `dt` rounded ONCE from the EXACT
 * rational t_end-t_begin, not reconstructed from two ALREADY-rounded
 * t_begin/t_end doubles subtracted a second time -- the two are not
 * always bit-equal (found this session: a Runner-driven bit-identical
 * check caught the difference in --check's round-off-scale diagnostics
 * after the Brusselator physics refactor). This shim does that exact-
 * rational subtraction once, in C++ (fuse::Rational), so u_main.c/
 * v_main.f90 never have to get rational arithmetic right themselves.
 * `required` (length size()) is P_VU(V)'s own mean over the step, on
 * U's own grid -- exactly what advance_window() hands an externally-
 * driven window member through `cfg.required` before suspending
 * (docs/user/c-abi.md's own "What imports are current actually
 * means"). On success (return 0), `provided_out` (length size())
 * receives q, valid at t_end -- exactly what advance_window() reads
 * from `cfg.provided` on resume. Nonzero: see bruss_u_last_error(). */
int bruss_u_step(bruss_u_handle* h, long long t_begin_num, long long t_begin_den,
                 long long t_end_num, long long t_end_den, const double* required,
                 double* provided_out);
void bruss_u_save_state(bruss_u_handle* h);
void bruss_u_restore_state(bruss_u_handle* h);
void bruss_u_state(const bruss_u_handle* h, double* out);
double bruss_u_external_total(const bruss_u_handle* h);
double bruss_u_exchange_out(const bruss_u_handle* h);
/* Valid only after a bruss_u_step() call that returned nonzero; the
 * pointer is owned by `h` and invalidated by the next bruss_u_step()
 * call or by bruss_u_destroy(). */
const char* bruss_u_last_error(const bruss_u_handle* h);

/* V_t = a*Lap(V) - q_on_v (species_core.hpp's own VDiffusionCore), PLUS
 * the by-hand remap (both directions) species_component.hpp's own
 * VComponent carries (species_core.hpp's own VParticipant) -- the
 * participant graph IS Unit 1's own graph plus only add_coupler()/
 * remap_on() (the lead's own ruling, Conformance Unit 2: both transfer
 * fields stay on u_region, so the coupler-routed Remap this compiles is
 * the IDENTITY and V still stands in for the real remap by hand,
 * exactly as it already did in Unit 1). `nx_u`/`ny_u` are U's own
 * native grid shape (needed here so this shim can build the SAME
 * U<->V remap weights VComponent's own constructor does); `nx_v`/`ny_v`
 * are V's own. Returns NULL on allocation failure OR on a weight-build
 * failure (an nx/ny that cannot be laid out as a periodic grid) --
 * there is no handle yet to report the error through, so a NULL return
 * alone is this function's only diagnostic, exactly like
 * bruss_u_create()'s own contract. `rank`/`num_ranks`/`comm_f`/`solver`
 * mirror bruss_u_create()'s own (this header's comment on that function
 * has the full account) -- EXCEPT that `num_ranks > 1` is itself refused
 * (NULL return) here: `VParticipant::make`'s own `E-BRUSS-V-DECOMPOSED-
 * REMAP` refusal (species_core.hpp), since V's two-sided remap is not yet
 * routed through FUSE's distributed weights (a different track's work).
 * `solver` selects V's OWN diffusion solve only (independent of U's). */
bruss_v_handle* bruss_v_create(int nx_u, int ny_u, int nx_v, int ny_v, double diffusion,
                               int rank, int num_ranks, int64_t comm_f, const char* solver);
void bruss_v_destroy(bruss_v_handle* h);
/* The length bruss_v_step()'s own `required`/`provided_out` use -- U's
 * own grid size (nx_u*ny_u), NOT V's own (nx_v*ny_v): both of V's
 * window-transfer fields live on u_region, species_component.hpp's own
 * file header explains why. */
int bruss_v_size(const bruss_v_handle* h);

/* One step from `t_begin` to `t_end`, given as exact rational
 * numerator/denominator pairs (matching fuse_exact_time) -- NOT two
 * doubles, for the same reason bruss_u_step() takes rationals rather
 * than a precomputed dt: VDiffusionCore::step() takes dt directly (it
 * has no forcing term that needs t_begin/t_end separately), but the
 * dt it takes must still be `(t_end - t_begin)` rounded to double
 * EXACTLY ONCE, in exact rational arithmetic -- the same discipline
 * fuse::StepContext::dt() follows. Computing dt as a double
 * subtraction of two already-rounded doubles here would reintroduce,
 * for V, precisely the rounding mismatch Unit 2's own item 1 found
 * and fixed for U (species_core.hpp's UCore::step() header has the
 * full account). `required` (length bruss_v_size()) is the raw q mean
 * this step, on the U grid (unremapped) -- exactly what
 * advance_window() hands an externally-driven window member through
 * `cfg.required`. On success (return 0), `provided_out` (length
 * bruss_v_size()) receives P_VU(V's own new state), on the U grid,
 * valid at t_end. Nonzero: see bruss_v_last_error(). */
int bruss_v_step(bruss_v_handle* h, long long t_begin_num, long long t_begin_den,
                 long long t_end_num, long long t_end_den, const double* required,
                 double* provided_out);
void bruss_v_save_state(bruss_v_handle* h);
void bruss_v_restore_state(bruss_v_handle* h);
/* V's own native state (length nx_v*ny_v, NOT bruss_v_size()) -- for
 * verification against the in-process reference only; no graph field
 * is this shape. */
void bruss_v_state(const bruss_v_handle* h, double* out);
double bruss_v_exchange_in(const bruss_v_handle* h);
const char* bruss_v_last_error(const bruss_v_handle* h);

/* S6 phase 3 (Conformance, "LowFive" track), Unit 3b: the ordinary,
 * no-FUSE-call-at-all HDF5 write/read a window member's own loop performs
 * directly against its export/import, with the VOL connector that makes
 * this interception rather than a real file attached process-wide by
 * HDF5's own HDF5_VOL_CONNECTOR/HDF5_PLUGIN_PATH mechanism (docs/user/
 * lowfive.md's "Unit 3" design) -- this shim exists ONLY so u_main.c
 * (pure C, already links this translation unit) and v_main.f90 (pure
 * Fortran, no HDF5 Fortran module in this install -- checked, /opt/hdf5
 * ships no Fortran bindings) can issue the SAME two HDF5 calls (H5Fcreate/
 * H5Dcreate2/H5Dwrite/H5Fclose and H5Fopen/H5Dopen2/H5Dread/H5Fclose) once
 * rather than twice, exactly as this file's own header comment already
 * does for bruss_u_step/bruss_v_step. `fapl_id` is deliberately NOT a
 * parameter here, unlike LowFiveTransport::access_plist_handle() (the
 * DIFFERENT, programmatic-attachment path examples/demo_b uses): a literal
 * H5P_DEFAULT is the whole point of the env-var attachment mechanism this
 * track's instruction specifies, so both calls always pass it.
 *
 * Returns 0 on success; nonzero on failure, with a message copied into
 * errbuf (truncated to errbuf_len - 1 bytes, always NUL-terminated) --
 * the same out-parameter convention access_plist_handle()'s own callers
 * use, rather than bruss_*_last_error()'s per-handle convention, since
 * there is no handle here. Without -DFUSE_ENABLE_LOWFIVE=ON, both
 * functions return nonzero immediately with "not built with LowFive" in
 * errbuf, never touching HDF5 at all (no `hdf5.h` on this build's own
 * include path in that configuration). */
int bruss_lowfive_write(const char* file, const char* dataset,
                        const double* data, long long n,
                        char* errbuf, long long errbuf_len);
int bruss_lowfive_read(const char* file, const char* dataset,
                       double* data, long long n,
                       char* errbuf, long long errbuf_len);

/* 1 iff this build was configured with -DFUSE_ENABLE_LOWFIVE=ON, 0
 * otherwise. v_main.f90's own early refusal check needs this as a plain
 * runtime query -- unlike u_main.c, which can read the matching build-
 * config header directly (a C-safe header), v_main.f90 is pure Fortran
 * with no way to read a C preprocessor macro, and this codebase ships no
 * HDF5 Fortran module to probe for instead (checked: /opt/hdf5 has
 * none). */
int bruss_lowfive_available(void);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_SHIM_H_ */