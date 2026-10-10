/* Time windows track (Conformance, Unit 2): the C participant for
 * component "U". PURE C -- no C++ header, no C++ call, anywhere in this
 * file (the lead's own constraint: "a foreign code never touches C++").
 * Links fuse::abi (fuse/abi/participant.h, the one-exported-symbol
 * table), the shim's own C-ABI physics wrapper (shim.h/shim.cpp -- a
 * C++ TRANSLATION UNIT, but its own header is extern "C", so this file
 * never sees a C++ type) and loop_step.h (the SAME u_loop_step() the
 * tier-0 harness, test/verification/brusselator_participants_test.cpp,
 * drives).
 *
 * GRAPH AUTHORING vs PARTICIPATING (Q2): this program never builds a
 * fuse::CouplingGraph. It calls fuse_load_graph() on a YAML path a
 * SEPARATE tool, brusselator_graph_tool, already wrote (--graph on the
 * command line), and nothing else constructs the graph.
 *
 * LAUNCH MODE FROM THE GRAPH, NOT FROM A FLAG (the brief's own "LaunchSpec
 * from the YAML, connect() per the YAML's launch mode"): this program
 * always passes `communicator = MPI_Comm_c2f(MPI_COMM_WORLD)` and
 * `my_component = "U"` to fuse_setup(); fuse_setup() itself resolves "U"'s
 * LaunchSpec from the loaded graph and only calls fuse::transport::
 * connect() when that LaunchSpec says to (fuse_setup_options::my_component
 *'s own comment in fuse/abi/types.h has the full account) -- this
 * executable's own code is IDENTICAL whether the graph says mpmd, accept,
 * or (for a tier-0 smoke run with no MPI at all) in-process.
 *
 * BUNDLED COUPLER RANKS (accept mode only, KNOWN_ISSUES.md D18 / Q1):
 * connect_pairwise() supports exactly two Connect-mode components, so in
 * accept mode the coupler's ranks are NOT a separate executable -- they
 * are EXTRA ranks of this same executable's own MPI_COMM_WORLD, declared
 * in the graph as an overlapping subset of U's own RankSet
 * (brusselator_graph_tool.cpp's own header comment has the full
 * rationale). `--ranks-u N` tells this binary how many of ITS OWN local
 * ranks (0..N-1) are real U-physics ranks; any remaining local ranks
 * (N..world_size-1, present only in accept mode with a nonzero
 * --ranks-coupler) are coupler placeholder ranks: they call fuse_setup()
 * identically (the call is collective over this executable's WHOLE local
 * world, exactly matching what MPI_Comm_accept/MPI_Intercomm_merge
 * requires of every rank of the local group passed to them) but never
 * create a bruss_u_handle, never register a field, and never call
 * u_loop_step() -- there is no "U" work for them to drive, and the
 * coupler itself has no clock, no step, and no user code of its own
 * (docs/dev/coupler-placement.md's own enumerated call sites). In mpmd
 * mode --ranks-u defaults to this executable's own full world size (no
 * bundled coupler at all -- the coupler is coupler_main.c's own separate
 * appnum segment there instead).
 *
 * NOT YET HOST-VERIFIED (MPI cannot start in this sandbox, KNOWN_ISSUES.md
 * C1): whether a coupler placeholder rank skipping begin_step/end_step
 * entirely (rather than calling them with no field registered) is what
 * the Runner's own internal remap collectives actually expect. Flagged
 * in the Unit 2 report as the one open mechanical question beyond
 * Q1-Q3; build-verified only.
 */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuse/abi/participant.h"
#include "fuse/base/config.hpp"
#include "brusselator_grid.h"
#include "u_loop.h"

/* S6 phase 3 (Conformance, "LowFive" track), Unit 3b. The graph tool's own
 * hardcoded choice (brusselator_graph_tool.cpp's own `--transport lowfive`
 * block) -- u_main.c has no ABI query for a field's own hdf5_path, so this
 * mirrors the SAME "reproduce the tool's own build-time constant" pattern
 * this file already uses for nx/ny below. */
#define U_LOWFIVE_OWN_FILE "brusselator_u.h5"
#define U_LOWFIVE_OWN_DATASET "/grid/u_out"
#define U_LOWFIVE_PEER_FILE "brusselator_v.h5"
#define U_LOWFIVE_PEER_DATASET "/grid/v_out"

/* Unit 3b, item 2(a): `fuse_step_report::delivery_pending` (entries.def,
 * docs/user/c-abi.md's own new section) replaces this file's own prior
 * window-boundary arithmetic entirely -- the ABI itself now names the
 * exact call at which a real H5Dwrite/H5Dread pair is owed, independent
 * of this binary's own dt/window constants (which item 2(b) of this same
 * round changes to the real example's own values; this file no longer
 * needs to track them at all). A degenerate one-substep-per-window graph
 * made "before every substep" and "before a sweep's first substep"
 * coincide by accident, which is what let the prior version of this file
 * get away with reading unconditionally -- recorded in KNOWN_ISSUES.md
 * D14 as a scope limit now closed. */

static void u_lowfive_fail_io(const char* what, int rc, const char* errbuf) {
  fprintf(stderr, "FAIL(u_main): %s: rc=%d %s\n", what, rc, errbuf);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void fail(fuse_ctx* ctx, const char* what, fuse_status rc);

/* view_field(u_out), NOT u_out itself: the application's job is the
 * window's own K-wide waveform buffer, not the single-sample native
 * storage register_field() was given (shim.h's own u_out is q at t_end
 * only). Shared by the cold-start write and the steady-state
 * delivery_pending-gated one below -- same call, two call sites. */
static void u_lowfive_write_export(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                                   const fuse_field_ref* out_ref) {
  fuse_field_view out_view;
  memset(&out_view, 0, sizeof(out_view));
  out_view.struct_size = sizeof(out_view);
  fuse_status rc = t->view_field(ctx, out_ref, &out_view);
  if (rc != FUSE_OK) fail(ctx, "view_field(u_out)", rc);
  if (out_view.window_samples > 0 && out_view.data) {
    const long long nvals = (long long)out_view.count * out_view.components;
    char errbuf[2048];
    const int io_rc =
        bruss_lowfive_write(U_LOWFIVE_OWN_FILE, U_LOWFIVE_OWN_DATASET,
                            (const double*)out_view.data, nvals, errbuf,
                            (long long)sizeof(errbuf));
    if (io_rc != 0) u_lowfive_fail_io("bruss_lowfive_write(u_out)", io_rc, errbuf);
  }
}

/* view_field(u_in), NOT u_in itself -- the window Sink copies from there
 * into u_in as part of the SAME mechanics it already uses for an
 * ordinary channel (docs/user/lowfive.md's own Unit 3 account). */
static void u_lowfive_read_import(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                                  const fuse_field_ref* in_ref) {
  fuse_field_view in_view;
  memset(&in_view, 0, sizeof(in_view));
  in_view.struct_size = sizeof(in_view);
  fuse_status rc = t->view_field(ctx, in_ref, &in_view);
  if (rc != FUSE_OK) fail(ctx, "view_field(u_in)", rc);
  if (in_view.window_samples > 0 && in_view.data) {
    const long long nvals = (long long)in_view.count * in_view.components;
    char errbuf[2048];
    const int io_rc =
        bruss_lowfive_read(U_LOWFIVE_PEER_FILE, U_LOWFIVE_PEER_DATASET,
                           (double*)in_view.data, nvals, errbuf,
                           (long long)sizeof(errbuf));
    if (io_rc != 0) u_lowfive_fail_io("bruss_lowfive_read(u_in)", io_rc, errbuf);
  }
}

static void fail(fuse_ctx* ctx, const char* what, fuse_status rc) {
  char buf[2048];
  fuse_error err;
  memset(&err, 0, sizeof(err));
  err.struct_size = sizeof(err);
  err.buffer = buf;
  err.capacity = sizeof(buf);
  buf[0] = '\0';
  if (ctx) {
    const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
    if (t) t->last_error(ctx, NULL, &err);
  }
  fprintf(stderr, "FAIL(u_main): %s: status=%d %s\n", what, (int)rc, buf);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char** argv) {
  const char* graph_path = NULL;
  const char* log_path = NULL;
  int ranks_u = -1; /* -1 = "all of my local world" (mpmd default) */
  int periods = 2;
  const char* transport = "fuse";
  /* Physics track: "auto" resolves to the direct (whole-grid) solver at
   * ranks_u==1 and to schur at ranks_u>1 (species_core.hpp's own
   * SolverKind::Auto doc comment) -- the task's own "--solver flag
   * defaulting to schur on >1 rank" is exactly this default, with no
   * special-casing needed here. */
  const char* solver = "auto";

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
      graph_path = argv[++i];
    } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
      log_path = argv[++i];
    } else if (strcmp(argv[i], "--ranks-u") == 0 && i + 1 < argc) {
      ranks_u = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--periods") == 0 && i + 1 < argc) {
      periods = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--transport") == 0 && i + 1 < argc) {
      transport = argv[++i];
    } else if (strcmp(argv[i], "--solver") == 0 && i + 1 < argc) {
      solver = argv[++i];
    } else {
      fprintf(stderr, "u_main: unrecognized argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (!graph_path || !log_path) {
    fprintf(stderr,
            "usage: u_main --graph graph.yaml --log u.log [--ranks-u N] "
            "[--periods N] [--transport fuse|lowfive] [--solver auto|direct|schur|pcg|gmres]\n");
    return 1;
  }
  if (strcmp(transport, "fuse") != 0 && strcmp(transport, "lowfive") != 0) {
    fprintf(stderr, "u_main: --transport must be 'fuse' or 'lowfive' (got '%s')\n",
            transport);
    return 1;
  }
  const int use_lowfive = strcmp(transport, "lowfive") == 0;
  if (use_lowfive && !FUSE_HAVE_LOWFIVE) {
    fprintf(stderr,
            "u_main: --transport lowfive requires a build configured with "
            "-DFUSE_ENABLE_LOWFIVE=ON; this binary was not (FUSE_HAVE_LOWFIVE=0)\n");
    return 1;
  }

  MPI_Init(&argc, &argv);
  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (ranks_u < 0) ranks_u = world_size; /* mpmd: every local rank is U */
  const int is_physics_rank = (world_rank < ranks_u);

  /* Physics track: THIS executable's own MPI_COMM_WORLD is NOT "U's own
   * world" in mpmd mode -- it is the SHARED world every ':'-separated
   * mpiexec segment's ranks number into one block each (U first, by the
   * graph's own component order, validate_launch_layout()'s own
   * requirement), so U's physics ranks are exactly [0, ranks_u) of it,
   * but V's and the coupler's own ranks share the SAME MPI_COMM_WORLD
   * too. In accept mode this executable's local world IS already "U
   * plus its own bundled coupler ranks" only, but the split below is
   * correct and a no-op-equivalent either way: it is what actually
   * forms "U's own physics-only communicator" regardless of which mode
   * put which ranks into this process's MPI_COMM_WORLD.
   *
   * COLOR = COMPONENT INDEX (U 0, V 1, coupler 2), not "0 or
   * MPI_UNDEFINED" as an earlier version of this file had it: v_main.f90
   * and coupler_main.c now call the IDENTICAL split
   * (MPI_Comm_split(MPI_COMM_WORLD, <their own fixed color>, world_rank,
   * ...)) at the equivalent point in their own startup, so every rank of
   * every ':'-segment in an mpmd launch names a color exactly once and
   * none ever needs MPI_UNDEFINED -- one uniform call shape across all
   * three files, rather than "the physics owner's own color plus
   * everyone else opts out", which was harder to keep synchronized
   * correctly across three separately-compiled executables (this is
   * exactly the kind of choreography mismatch this track's own host run
   * found once already: coupler_main.c's own matching call, placed AFTER
   * setup() instead of before load_graph(), met U's own split at the
   * wrong logical point and truncated a later Allgather -- the lead's own
   * fix, 2026-10-10). A non-physics (bundled coupler placeholder, accept
   * mode only) rank's own color is 2, same as a real coupler_main.c
   * process's in mpmd mode; it reads back a real (if unused)
   * communicator rather than MPI_COMM_NULL, and frees it unconditionally
   * below rather than only when physics.
   *
   * COLLECTIVE: MPI_Comm_split is called by EVERY rank of this
   * executable's own world, physics and non-physics alike (the same
   * "every rank calls every split" discipline src/runtime/runner.cpp's
   * own base-communicator formation uses, and this track's own standing
   * lesson: a collective must never be skipped on a locally computed
   * condition). `key = world_rank` preserves this process's own relative
   * rank order inside the new communicator, so a physics rank's OWN
   * position within u_comm is identical to its position within
   * MPI_COMM_WORLD's own [0, ranks_u) block -- u_rank below is that
   * position, read back rather than assumed. */
  MPI_Comm u_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, is_physics_rank ? 0 : 2, world_rank, &u_comm);
  int u_rank = 0, u_size = 1;
  if (is_physics_rank) {
    MPI_Comm_rank(u_comm, &u_rank);
    MPI_Comm_size(u_comm, &u_size);
  }

  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) {
    fprintf(stderr, "FAIL(u_main): fuse_coupler_abi() returned NULL\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  fuse_context_out co;
  memset(&co, 0, sizeof(co));
  co.struct_size = sizeof(co);
  fuse_status rc = t->create(NULL, NULL, &co);
  if (rc != FUSE_OK) fail(NULL, "create", rc);
  fuse_ctx* ctx = co.ctx;

  fuse_graph_source src;
  memset(&src, 0, sizeof(src));
  src.struct_size = sizeof(src);
  src.yaml_path = graph_path;
  rc = t->load_graph(ctx, &src, NULL);
  if (rc != FUSE_OK) fail(ctx, "load_graph", rc);

  /* The ONE field a foreign caller sets to take part in a multi-
   * executable run: my_component. communicator is this executable's
   * OWN world (already MPI_Init'd); fuse_setup() resolves "U"'s
   * LaunchSpec from the loaded graph and connects accordingly (this
   * file's own header comment has the full account). */
  fuse_setup_options setup;
  memset(&setup, 0, sizeof(setup));
  setup.struct_size = sizeof(setup);
  setup.communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
  setup.my_component = "U";
  rc = t->setup(ctx, &setup, NULL);
  if (rc != FUSE_OK) fail(ctx, "setup", rc);

  bruss_u_handle* uh = NULL;
  double* u_in = NULL;
  double* u_out = NULL;
  int nu = 0;
  FILE* log = NULL;
  /* Hoisted out of the is_physics_rank block below (unlike u_out_ref's/
   * u_in_ref's own prior scope, which never needed to survive past field
   * registration) -- the lowfive write/read block in the period loop
   * further down needs these same handles again. */
  /* struct_size on EVERY one of them, here where they are declared. An
   * out struct is negotiated exactly like an in struct: the library reads
   * the caller's struct_size to learn which layout it was compiled
   * against, and a zero is not "unset, use the current layout" but a
   * version older than anything it knows -- FUSE_ERR_VERSION. Only
   * u_out_ref used to get one (set again at its own lookup below), so
   * field_handle("u_in") failed with -6 the first time this program ran
   * far enough to reach it (host, 2026-10-09). The tier-0 conformance
   * harness drives the SAME loops through the same ABI and never saw it,
   * because it is a different caller: this is u_main.c's own bug, and the
   * negotiation check is what caught it. */
  fuse_field_ref u_out_ref, u_in_ref, v_out_peer_ref;
  memset(&u_out_ref, 0, sizeof(u_out_ref));
  u_out_ref.struct_size = sizeof(u_out_ref);
  memset(&u_in_ref, 0, sizeof(u_in_ref));
  u_in_ref.struct_size = sizeof(u_in_ref);
  memset(&v_out_peer_ref, 0, sizeof(v_out_peer_ref));
  v_out_peer_ref.struct_size = sizeof(v_out_peer_ref);
  v_out_peer_ref.handle = -1;

  if (is_physics_rank) {
    memset(&u_out_ref, 0, sizeof(u_out_ref));
    u_out_ref.struct_size = sizeof(u_out_ref);
    fuse_field_name fname;
    memset(&fname, 0, sizeof(fname));
    fname.struct_size = sizeof(fname);
    fname.component = "U";
    fname.field = "u_out";
    rc = t->field_handle(ctx, &fname, &u_out_ref);
    if (rc != FUSE_OK) fail(ctx, "field_handle(u_out)", rc);

    /* The grid is brusselator_grid.h's, the SAME header the graph tool
     * writes the meshes from. The participant ABI has no "field length"
     * query (the caller is expected to know its own grid), which is
     * exactly why the number must come from one place: this file carried
     * its own copy, the graph tool moved to 16x16 and this stayed at 3x3,
     * and a 9-element registration against a 256-cell field ran for two
     * days before the host measured a trajectory nothing could explain. */
    const int nx = BRUSS_NX_U, ny = BRUSS_NY_U;
    /* Physics track: OWNED size at (u_rank, u_size), not the GLOBAL
     * nx*ny every pre-decomposition caller used -- at u_size==1 the two
     * are equal (u_rank==0 forced above), so this is byte-identical to
     * before for every row this file's own ctest rows exercise today
     * (ranks_u 1 or 2; the latter genuinely decomposed). The registered
     * field storage below (view.count/owned_count) is this SAME nu, per
     * the Runner's own per-placement sizing (D24/D25). */
    nu = bruss_owned_size(nx, ny, u_rank, u_size);
    if (nu < 0) fail(ctx, "bruss_owned_size", FUSE_ERR_RUNTIME);
    u_out = (double*)calloc((size_t)nu, sizeof(double));
    u_in = (double*)calloc((size_t)nu, sizeof(double));

    /* INITIAL COUPLING STATE, the same way the tier-0 harness builds it
     * (test/verification/brusselator_participants_test.cpp, "U seeded
     * from V's own initial provided value"): U's initial reaction rate
     * depends on V's initial state on U's grid, and V's initial state is
     * not uniform, so "zero is a valid starting point" (what this block
     * said until 2026-10-09) is a DIFFERENT problem from the one V and the
     * single-process example solve. A throwaway V, built from the same
     * header's grid, is read for its constructor's own initial P_VU(V0)
     * and discarded -- deterministic, cheap, and bit-identical to what the
     * real V in the other program starts from. The throwaway V is always
     * SINGLE-rank (V's own decomposition is a different track's work --
     * species_core.hpp's own VParticipant::make refuses otherwise), so
     * its own bruss_v_provided() hands back the WHOLE U grid
     * (nx*ny); bruss_owned_slice() is what restricts that down to THIS
     * U rank's own owned band before it ever reaches bruss_u_create().
     *
     * This is a limitation of the current startup contract worth naming:
     * a participant must know its partner's initial export before
     * fuse_initialize(), because the Runner primes the first window from
     * the storage registered at that point. A coupler-delivered initial
     * exchange (imports delivered before exports are frozen) would remove
     * the need for U to reconstruct V at all. */
    {
      bruss_v_handle* v_seed = bruss_v_create(nx, ny, BRUSS_NX_V, BRUSS_NY_V, BRUSS_DIFFUSION,
                                              /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
      if (!v_seed) fail(ctx, "bruss_v_create (initial seed)", FUSE_ERR_RUNTIME);
      double* v_seed_whole = (double*)calloc((size_t)(nx * ny), sizeof(double));
      bruss_v_provided(v_seed, v_seed_whole); /* P_VU(V0), WHOLE U grid */
      bruss_v_destroy(v_seed);
      if (bruss_owned_slice(v_seed_whole, nx, ny, u_rank, u_size, u_in) != 0) {
        fail(ctx, "bruss_owned_slice (initial seed)", FUSE_ERR_RUNTIME);
      }
      free(v_seed_whole);
    }
    uh = bruss_u_create(nx, ny, BRUSS_DIFFUSION, BRUSS_A, BRUSS_B, BRUSS_FORCING, u_in, u_rank,
                        u_size, (int64_t)MPI_Comm_c2f(u_comm), solver);
    if (!uh) fail(ctx, "bruss_u_create", FUSE_ERR_RUNTIME);
    /* q at t=0 into the storage registered below: the Runner primes the
     * first window's t=0 sample from it at fuse_initialize(), and V
     * integrates that sample. Left at calloc's zero, V received a window
     * whose first sample was 0 while U consumed from its real q0, and the
     * two sides' exchange disagreed by exactly that half-interval. */
    bruss_u_provided(uh, u_out);

    fuse_field_view view;
    memset(&view, 0, sizeof(view));
    view.struct_size = sizeof(view);
    view.handle = u_out_ref.handle;
    view.element_type = FUSE_ELEMENT_F64;
    view.memspace = FUSE_MEMSPACE_HOST;
    view.ownership = FUSE_OWNERSHIP_OWNED;
    view.components = 1;
    view.count = (uint64_t)nu;
    view.owned_count = (uint64_t)nu;
    view.stride = 1;
    view.data = u_out;
    rc = t->register_field(ctx, &view, NULL);
    if (rc != FUSE_OK) fail(ctx, "register_field(u_out)", rc);

    fname.field = "u_in";
    rc = t->field_handle(ctx, &fname, &u_in_ref);
    if (rc != FUSE_OK) fail(ctx, "field_handle(u_in)", rc);
    view.handle = u_in_ref.handle;
    view.data = u_in;
    rc = t->register_field(ctx, &view, NULL);
    if (rc != FUSE_OK) fail(ctx, "register_field(u_in)", rc);

    /* delivery_field (docs/user/c-abi.md's own protocol) names the
     * delivering channel's own SOURCE field -- "v_out" for the channel
     * this component CONSUMES (b2a's own source, not "u_in", its own
     * target) -- so matching it needs the PEER's own field handle, not
     * just this component's own u_out/u_in. A graph-wide lookup
     * (any component's name, not just "U") -- fuse_field_handle()
     * resolves straight off the loaded graph, which every process has a
     * full copy of regardless of which components it hosts. */
    if (use_lowfive) {
      fname.component = "V";
      fname.field = "v_out";
      rc = t->field_handle(ctx, &fname, &v_out_peer_ref);
      if (rc != FUSE_OK) fail(ctx, "field_handle(v_out)", rc);
    }

    log = fopen(log_path, "w");
    if (!log) fail(ctx, "fopen(log)", FUSE_ERR_RUNTIME);
  }

  rc = t->initialize(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(ctx, "initialize", rc);

  if (is_physics_rank && use_lowfive) {
    /* Cold start (docs/user/c-abi.md's own `delivery_pending` section;
     * the harness's own precedent, test/verification/
     * brusselator_lowfive_harness_test.cpp): period 0's very first
     * ExternalDelivery suspension is hit before any step at all, inside
     * fuse_begin_step()'s own first-ever call -- too late for this
     * process to know ahead of time it must deliver first. Write this
     * component's own INITIAL (pre-any-step) u_out and read the peer's
     * -- constant over the window by construction, nothing has run
     * yet to vary either -- unconditionally, once, before the loop. */
    u_lowfive_write_export(t, ctx, &u_out_ref);
    u_lowfive_read_import(t, ctx, &u_in_ref);
  }

  if (is_physics_rank) {
    double* baseline_state = (double*)calloc((size_t)nu, sizeof(double));
    bruss_u_state(uh, baseline_state);
    double baseline_mean_u = 0.0;
    for (int i = 0; i < nu; ++i) baseline_mean_u += baseline_state[i];
    baseline_mean_u /= (double)nu;
    free(baseline_state);
    fprintf(log, "window=0 mean_u=%a external_total=%a exchange_out=%a\n", baseline_mean_u,
            bruss_u_external_total(uh), bruss_u_exchange_out(uh));

    fuse_step_request req;
    memset(&req, 0, sizeof(req));
    req.struct_size = sizeof(req);
    req.component = "U";

    for (int p = 0; p < periods; ++p) {
      fuse_ongoing ongoing;
      memset(&ongoing, 0, sizeof(ongoing));
      ongoing.struct_size = sizeof(ongoing);
      /* This process hosts only "U" (shim.h's own precedent, species_
       * component.hpp's single-component-per-rank convention) -- unlike
       * the tier-0 harness, which drives BOTH U and V in one process and
       * therefore does need next_component, u_main.c calls begin_step/
       * end_step for "U" every time it is this process's own turn, and
       * simply loops until the PERIOD closes; it never has to decide
       * which OTHER component is due, because no other component is
       * ever due on this process's own ranks. */
      fuse_step_report rep, begin_rep;
      for (;;) {
        rc = u_loop_step(ctx, uh, &req, u_in, u_out, &rep, &begin_rep);
        if (rc != FUSE_OK) fail(ctx, "u_loop_step", rc);

        /* docs/user/c-abi.md's own `delivery_pending`/`delivery_field`
         * protocol: on a report naming field F, write F if producing it
         * or read F if consuming it, then begin_step() again -- never
         * elsewhere, and never both unconditionally (a graph with more
         * than one independently-paced LowFive-bound channel can report
         * a field THIS component is neither producing nor consuming at
         * all, this round's own corrected rule). view_field(u_out), not
         * u_out itself, still holds the sweep that JUST completed
         * (nothing has overwritten it yet, since this call's own physics
         * -- if any ran -- is for the NEXT sweep, not this report's). */
        if (use_lowfive && begin_rep.delivery_pending) {
          if (begin_rep.delivery_field == u_out_ref.handle) {
            u_lowfive_write_export(t, ctx, &u_out_ref);
          } else if (begin_rep.delivery_field == v_out_peer_ref.handle) {
            u_lowfive_read_import(t, ctx, &u_in_ref);
          }
        }

        if (rep.next_component == -1) break; /* period complete */
      }
      /* u_out holds q (the reaction term), NOT U's own state -- the log
       * format's own mean_u is the STATE mean (bruss_u_state), matching
       * main.cpp's own pu->state().mean(). */
      double* state = (double*)calloc((size_t)nu, sizeof(double));
      bruss_u_state(uh, state);
      double mean_u = 0.0;
      for (int i = 0; i < nu; ++i) mean_u += state[i];
      mean_u /= (double)nu;
      free(state);

      fprintf(log, "window=%d mean_u=%a external_total=%a exchange_out=%a\n",
              p + 1, mean_u, bruss_u_external_total(uh), bruss_u_exchange_out(uh));
    }
    fclose(log);
    bruss_u_destroy(uh);
    free(u_in);
    free(u_out);
  } else {
    /* Unit 2b (D19), step (d): a bundled-coupler placeholder rank
     * (accept mode, --ranks-coupler > 0) hosts no externally driven
     * component -- fuse_run() is now legal here (Runner::hosts_
     * component() says so; fuse_run()'s own E-ABI-RUN-EXTERNAL refusal
     * only fires when one IS hosted). This is the ONLY way this rank
     * can execute its own share of the period's collectives (the
     * Transport/Remap routed through it) at all: it registers no
     * field, so begin_step()/end_step() have nothing to drive, and
     * before this fix this branch did not exist -- this rank called
     * neither, leaving U's own rank blocked in the first Transport
     * that needed this one's participation (KNOWN_ISSUES.md D19,
     * evidence 4). Same --periods every physics rank steps, so every
     * rank's own period count agrees (the schedule's own common-
     * period invariant depends on it). */
    fuse_run_request req;
    memset(&req, 0, sizeof(req));
    req.struct_size = sizeof(req);
    req.periods = periods;
    rc = t->run(ctx, &req, NULL);
    if (rc != FUSE_OK) fail(ctx, "run", rc);
  }

  rc = t->destroy(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(NULL, "destroy", rc);

  /* Unconditional: u_comm is never MPI_COMM_NULL now that every rank's
   * own color is 0 or 2 (never MPI_UNDEFINED) -- see this split's own
   * comment above. */
  MPI_Comm_free(&u_comm);
  MPI_Finalize();
  return 0;
}
