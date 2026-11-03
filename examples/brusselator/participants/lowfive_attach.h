#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_

#include <stdint.h>

/* Constructs the LowFive VOL connector PROGRAMMATICALLY -- a file-access
 * property list carrying it, built the same way the SINGLE-PROCESS path
 * already measured in this tree does (src/transport/lowfive_transport.
 * cpp's own VolSingleton, LowFive::H5VOLProperty::apply) -- instead of
 * relying on HDF5's HDF5_VOL_CONNECTOR/HDF5_PLUGIN_PATH environment
 * route the loadable plugin (src/transport/lowfive_plugin_vol.cpp,
 * target fuse-lowfive-vol) uses. See docs/user/lowfive.md's
 * programmatic-attach account for why: with the env route, U's and V's
 * own FIRST H5Fcreate (shim.cpp's own bruss_lowfive_write, called from
 * bruss_lowfive_write_export) null-dereferenced inside LowFive's own
 * VOLBase::_file_create. This header's own attach call sidesteps the
 * question of WHY (the env route's info_t::vol class-static timing)
 * entirely by never taking that route for the participants.
 *
 * Opaque-free, C-ABI-only types cross this boundary, same convention as
 * shim.h's own bruss_lowfive_write/bruss_lowfive_read (an HDF5 hid_t
 * crosses as `long long`, exactly as `fuse::LowFiveTransport::
 * access_plist_handle()` already does, so this header stays HDF5-free
 * and is includable from a build without -DFUSE_ENABLE_LOWFIVE=ON, pure
 * C u_main.c included). */

#ifdef __cplusplus
extern "C" {
#endif

/* Called by EVERY rank of the bridge
 * communicator (bruss_lowfive_attach()'s own callers below -- the real
 * physics ranks -- AND a bundled coupler/placeholder rank that will
 * never call bruss_lowfive_attach() at all), right after fuse_setup()
 * succeeds, BEFORE any bruss_lowfive_write()/bruss_lowfive_read() call.
 * `bridge_comm_f` is the Fortran handle base_communicator() (entries.def)
 * returned for this context -- MPI_COMM_WORLD itself for an Mpmd launch
 * (MPMD callers keep passing MPI_COMM_WORLD
 * literally, not this accessor's own value -- see u_main.c's own header
 * comment for why), or the merged base communicator connect() formed for
 * a Connect (accept/connect) launch.
 *
 * Loads `graph_path` itself (the SAME file every caller already parsed
 * for its own fuse_load_graph(), reloaded here exactly as
 * bruss_lowfive_attach() below already does) and, if-and-only-if that
 * graph has at least one field with hdf5_path set, performs EXACTLY ONE
 * MPI_Comm_split() of `bridge_comm_f` -- collective over the WHOLE of
 * it, every rank, coupler/placeholder included (a collective must never
 * be skipped on a locally computed condition; MPI_Comm_split() is collective over its
 * own input communicator regardless of which output group a given rank
 * lands in). A rank this attach's own qualifying channels do not name
 * (the coupler, real or bundled) is given color MPI_UNDEFINED internally
 * and gets back MPI_COMM_NULL, reported here as `*local_comm_f_out = 0`
 * -- not an error, the expected answer for a rank with nothing to attach.
 * If the graph has NO hdf5_path field at all (an ordinary, non-LowFive
 * run), nothing is collective here at all: every rank reaches that same
 * answer from the SAME loaded file, with no MPI call of any kind
 * ("collective agreement the easy way", connect.hpp's own module
 * comment) -- so this function is SAFE and CHEAP to call unconditionally
 * on every participant main, lowfive or not, mpmd or accept.
 *
 * This split's own color is keyed by which component's RankSet (from the
 * graph -- the SAME writer/reader RankSets plan_lowfive_rendezvous()
 * consumes) this rank belongs to, NEVER by reusing a pre-existing
 * communicator like u_main.c's own `u_comm`: reuse would rest on an
 * argument, from source alone, that u_comm's own relative rank order
 * happens to coincide with the bridge's (true for U, who merges "low"
 * and keeps its own numbers, per MPI_Intercomm_merge's own high/low rule
 * -- not independently confirmed against real MPI semantics). A fresh
 * split of the bridge itself needs no such argument: the numbering IS, by
 * construction, the one plan_lowfive_rendezvous() already expects.
 *
 * On success (return 0) `*local_comm_f_out` is set (0 for a
 * non-participating rank, a real Fortran handle otherwise); on failure
 * (return nonzero), `*local_comm_f_out` is left at 0 and a message is
 * copied into `errbuf`. Without -DFUSE_ENABLE_LOWFIVE=ON, always
 * succeeds with `*local_comm_f_out = 0` -- there is no HDF5-bound field
 * for ANY graph to carry in that configuration, so every caller reaches
 * the identical "nothing to join" answer the same way an ordinary,
 * non-LowFive graph does, with no MPI call. The ONE communicator this
 * function creates (when it creates one at all) is this function's own
 * to describe, not to free -- see bruss_lowfive_finalize() below for
 * where and in what order it, and the intercommunicator
 * bruss_lowfive_attach() forms from it, are released. */
int bruss_lowfive_join_local_split(const char* graph_path, long long bridge_comm_f,
                                   long long* local_comm_f_out,
                                   char* errbuf, long long errbuf_len);

/* Call EXACTLY ONCE per process, after fuse_setup() succeeds and
 * bruss_lowfive_join_local_split() (above) has returned a NONZERO
 * `*local_comm_f_out` for THIS rank, and before the first
 * bruss_lowfive_write()/bruss_lowfive_read() call. `graph_path` is the
 * SAME --graph argument this program's own main() already parsed -- the
 * graph that carries the field-to-hdf5-path bindings
 * (fuse::build_lowfive_bindings(), the SAME function src/transport/
 * lowfive_plugin_graph.cpp's own LoadConfig() calls) and the writer/
 * reader RankSets this attach needs. Unlike the env-route plugin, this
 * shim already has the graph path on its own command line and reads no
 * FUSE_LOWFIVE_BINDINGS environment variable.
 *
 * NEVER CALLED by a rank with nothing to attach: a bundled coupler/placeholder rank (accept mode)
 * or the real coupler (mpmd mode) calls bruss_lowfive_join_local_split()
 * above -- the collective every rank of the bridge must join -- and
 * stops there; it must never call this function at all, mirroring
 * coupler_main.c's own, pre-existing, explicit convention ("this process
 * never calls bruss_lowfive_attach()").
 *
 * `bridge_comm_f` is the SAME handle passed to
 * bruss_lowfive_join_local_split() just before this call -- this
 * function reads this rank's own position in it (`this_rank`) for the
 * static order check, decide_attach_mode(), and plan_lowfive_rendezvous()
 * alike, and uses it as MPI_Intercomm_create()'s own `peer_comm`
 * argument. In an accept (Connect) launch
 * this canNOT be MPI_COMM_WORLD (unlike Mpmd, where it already is one and
 * the same) -- V's own position in the merged bridge is OFFSET from its
 * own local MPI_COMM_WORLD rank by U's local world size
 * (MPI_Intercomm_merge's own high/low rule), so using MPI_COMM_WORLD
 * here for an accept-mode V would silently derive the WRONG remote
 * leader.
 *
 * `local_comm_f` is the NONZERO handle bruss_lowfive_join_local_split()
 * already produced for this rank -- this function does not split
 * anything itself; it only verifies `local_comm_f`'s own size/position
 * still match the rendezvous plan it independently derives from the
 * SAME graph (defence in depth, not re-trust), then forms the ONE
 * intercommunicator LowFive needs via MPI_Intercomm_create(local_comm_f,
 * ..., bridge_comm_f, ...).
 *
 * On success (return 0), `*fapl_id_out` receives the file-access
 * property list every subsequent bruss_lowfive_write()/
 * bruss_lowfive_read() call on THIS process must pass as `fapl_id`,
 * literally, instead of H5P_DEFAULT.
 *
 * On failure (return nonzero), `*fapl_id_out` is left at -1 and a
 * message is copied into `errbuf` (same truncate-and-NUL-terminate
 * convention as bruss_lowfive_write/read): a malformed graph, no
 * HDF5-bound field in it, more than two Connect-mode (accept/connect)
 * components graph-wide, a Connect-
 * mode channel with no bridge formed for this context, two HDF5-bound channels this rank
 * takes part in that do not partition the world the same way, or
 * (distributed mode only) an HDF5/MPI call that itself failed. A second
 * call in the same process is also refused: the VOL registration is
 * process-wide and exactly once (src/transport/lowfive_transport.cpp's
 * own VolSingleton has the full account of why re-registering is not
 * safe).
 *
 * Without -DFUSE_ENABLE_LOWFIVE=ON, returns nonzero immediately with
 * "not built with LowFive" in errbuf, mirroring bruss_lowfive_write/
 * read's own contract; never touches HDF5 at all in that configuration. */
int bruss_lowfive_attach(const char* graph_path, long long bridge_comm_f,
                         long long local_comm_f, long long* fapl_id_out,
                         char* errbuf, long long errbuf_len);

/* INTERNAL to the shim -- called only from shim.cpp's own
 * bruss_lowfive_write(), right after its own H5Fclose(), never directly
 * by u_main.c/v_main.f90. No-op success (0) when the last successful
 * bruss_lowfive_attach() chose the single-process connector (nothing to
 * serve across an intercommunicator that was never formed).
 *
 * In distributed mode, consults the pure state machine in lowfive_
 * attach_decision.hpp (serve_state_step()) and either serves now
 * (`LowFive::DistMetadataVOL::serve_all()`) or defers. See
 * lowfive_attach_decision.hpp's own "WHY THE SERVE IS ROLE-ORDERED" and
 * "RECEIVE AND RELEASE" comments for the measured stack behind the
 * design.
 * Refused (nonzero, errbuf set) if bruss_lowfive_attach() was never
 * called successfully in this process, or if the state machine itself
 * refuses (a second deferred serve before the first was flushed -- see
 * serve_state_step()'s own doc). */
int bruss_lowfive_serve_after_write(char* errbuf, long long errbuf_len);

/* INTERNAL to the shim -- called only from shim.cpp's own
 * bruss_lowfive_read(), right before its own H5Fopen(), never directly
 * by u_main.c/v_main.f90. No-op success (0) in single-process mode.
 *
 * In distributed mode, does NOTHING LowFive-level at all, regardless of
 * role or pending state -- flushing a first-reader's own deferred serve
 * HERE, before H5Fopen, would reproduce a circular wait (lowfive_
 * attach_decision.hpp's own "RECEIVE AND RELEASE" comment has the full
 * citations). The real
 * receive is `H5Fopen` itself, about to run right after this call
 * returns, as an ordinary VOL callback dispatch -- nothing here needs to
 * assist it, and nothing here calls `LowFive::DistMetadataVOL::
 * broadcast_files()` either (same comment: that call propagates an
 * already-known file registry to a MULTI-RANK component's own sibling
 * local ranks, never to the peer component, and this row places exactly
 * one rank per component). `serve_state_step()` is still called purely
 * to keep the event sequence validated. Refused (nonzero, errbuf set)
 * if bruss_lowfive_attach() was never called successfully in this
 * process. */
int bruss_lowfive_receive_before_read(char* errbuf, long long errbuf_len);

/* INTERNAL to the shim -- called only from shim.cpp's own
 * bruss_lowfive_read(), right after its own H5Fclose(), on EVERY path
 * that reached a successful H5Fopen (success or a later dataset-level
 * failure -- shim.cpp's own comment on this has the full account of
 * why every such path, not only the success path), never directly by
 * u_main.c/v_main.f90. No-op success (0) in single-process mode.
 *
 * THIS is the event a first-reader's own deferred serve is flushed at
 * (not bruss_lowfive_receive_before_read()). Read from LowFive's own
 * source (lowfive_attach_decision.hpp's own "RECEIVE AND RELEASE"
 * comment): a process's own
 * H5Fclose of a REMOTE file it opened is exactly what releases the
 * PEER's own pending `serve_all()` -- so only once THIS process has
 * itself completed that open-and-close is it safe to enter its own
 * blocking serve, which the peer's own next, symmetric open-and-close
 * will then release in turn. In distributed mode, consults serve_state_
 * step() (`kImportClosed`) and either flushes (`serve_all()`) or does
 * nothing (first-server, or a first-reader with nothing pending -- not
 * an error). Refused (nonzero, errbuf set) if bruss_lowfive_attach() was
 * never called successfully in this process. */
int bruss_lowfive_after_read_close(char* errbuf, long long errbuf_len);

/* Call EXACTLY ONCE per process, at the very end of main() -- after
 * this process's own last bruss_lowfive_write/read call, before
 * MPI_Comm_free/MPI_Finalize -- regardless of whether bruss_lowfive_
 * attach() was ever called (a plain --transport fuse run is a harmless
 * no-op here, unlike serve_after_write/receive_before_read/after_read_
 * close's own "never attached" refusal: this is a hygiene check at
 * process exit, not a per-delivery operation).
 *
 * Purpose: catch, loudly, the one failure mode the role-ordered design
 * cannot detect any other way -- a first-reader role (see
 * decide_serve_roles()) that deferred a serve at its own last write and
 * then never got another kImportClosed to flush it against (an
 * asymmetric write/read count between the two sides, which would
 * violate this design's own progress invariant I1; see lowfive_attach_
 * decision.hpp). Refused (nonzero, errbuf set) in exactly that case; a
 * clean finalize (serve already flushed, or this side is the
 * first-server and never defers) returns 0. Without
 * -DFUSE_ENABLE_LOWFIVE=ON, or when LowFive was never attached in this
 * process, always returns 0: there is nothing to have deferred.
 *
 * ALSO where the two communicators this code creates are freed, in the
 * order that cannot hang -- AFTER this rank's own H5close() (H5close()
 * must run while the process is still in ORDINARY
 * execution, before any exit()-time static-destruction unwind begins,
 * the same reasoning that makes freeing a communicator here, rather
 * than letting MPI_Finalize's own implicit cleanup do it, the safe
 * order), BEFORE the caller's own MPI_Finalize: first the
 * intercommunicator bruss_lowfive_attach() formed (MPI_Comm_free,
 * non-collective over the BRIDGE -- an intercommunicator's own free is
 * collective only over its own two local groups, already disjoint from
 * the coupler), then the local communicator
 * bruss_lowfive_join_local_split() split off (also MPI_Comm_free). A
 * rank whose own bruss_lowfive_join_local_split() call returned
 * `*local_comm_f_out == 0` (the coupler, real or bundled) created
 * nothing and frees nothing here -- this function is still harmless to
 * call unconditionally for such a rank (the existing "always returns 0"
 * contract above already covers it). */
int bruss_lowfive_finalize(char* errbuf, long long errbuf_len);

/* 1 iff FUSE_LOWFIVE_ATTACH_TRACE is
 * set to exactly "1" in this process's own environment, read ONCE
 * (cached in a function-local static the first time this is called)
 * -- so shim.cpp, u_main.c, and v_main.f90 (via a bind(C) wrapper, see
 * bruss_lowfive_trace_skip() below) all agree on the SAME decision
 * without each doing its own getenv() call. 0 when unset, empty, "0",
 * or any other value -- lowfive_attach_decision.hpp's own
 * ParseTraceEnabledEnv() is the pure half this wraps; see its own doc
 * comment for the exact grammar. When 0, every trace call site below
 * is a single cheap branch -- no fprintf, no formatting -- so the
 * trace adds no cost on a normal (untraced) run (one cached bool). */
int bruss_lowfive_trace_enabled(void);

/* Called from u_main.c's/v_main.f90's own write/read guard's ELSE
 * branch (i.e. `window_samples > 0 && data` was FALSE, so the actual
 * H5Fcreate/H5Fopen was SKIPPED) -- settles whether the cold-start
 * export/import's own window view guard passed, or the write/read was
 * silently skipped because the window wasn't primed yet. `which` is a
 * short caller-chosen label ("u_lowfive_write_export",
 * "v_lowfive_read_import", etc.); `window_samples`/`count`/
 * `components` are the view's own fields at the point the guard was
 * evaluated. A no-op (no stderr output) unless bruss_lowfive_trace_
 * enabled() is true. v_main.f90 is pure Fortran with no printf-
 * equivalent that reaches stderr with this exact shape, hence a
 * bind(C) wrapper here rather than Fortran WRITE statements at each of
 * its own two call sites -- u_main.c could use fprintf() directly but
 * calls this too, so there is exactly one place that owns the trace
 * line's own format. */
void bruss_lowfive_trace_skip(const char* which, long long window_samples,
                              long long count, long long components);

/* Returns how many files this process's
 * own LowFive VOL currently holds in its in-memory metadata (`LowFive
 * ::MetadataVOL::files.size()`, a PUBLIC member -- confirmed from the
 * installed /opt/lowfive/include/lowfive/vol-metadata.hpp, the struct
 * defaults every member public and `files` is declared before any
 * access-specifier label), or -1 if bruss_lowfive_attach() was never
 * called successfully in this process. Nothing public exposes whether
 * a SPECIFIC dataset was attached as in-memory (`Dataset::is_memory` is
 * a private/internal field, not reachable from this translation unit's
 * own public LowFive headers) -- this file count is the best available
 * proxy. Diagnostic only -- never consulted by any
 * non-diagnostic code path. */
long long bruss_lowfive_attach_file_count(void);

/* Whether `file` (not the dataset within it) is tracked as a REAL file
 * (not a DummyFile), via `LowFive::MetadataVOL::has_real_file()`, a
 * PUBLIC, FILE-level, RELIABLE check. (An H5Lexists()-based dataset check
 * is NOT reliable: H5Lexists() walks a path SEGMENT BY SEGMENT
 * (LowFive::MetadataVOL::link_specific()'s own H5VL_LINK_EXISTS case calls
 * Object::locate() -> search(), which expects each slash-separated
 * segment to be a SEPARATE child object), but dataset_create() (called
 * with no prior H5Gcreate for an intermediate group) attaches the new
 * Dataset as ONE FLAT child literally named "grid/u_out" -- search()
 * looks for a child named "grid" first, finds none, and reports
 * not-found, even though the dataset genuinely exists; dataset_create()'s
 * own is_memory check, which compares the WHOLE string via
 * Object::fullname() rather than walking it segment by segment, is
 * unaffected.) Returns 1/0, or -1 if bruss_lowfive_attach() was never
 * called successfully in this process. Deliberately FILE-level, not
 * dataset-level: nothing public exposes a per-dataset existence check at
 * all (Object/Dataset are forward-declared/opaque in the installed public
 * headers) -- this is the best available proxy, the same limitation
 * bruss_lowfive_attach_file_count() above has. */
int bruss_lowfive_attach_has_real_file(const char* file);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_ */
