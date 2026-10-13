#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_

#include <stdint.h>

/* S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track:
 * constructs the LowFive VOL connector PROGRAMMATICALLY -- a file-access
 * property list carrying it, built the same way the SINGLE-PROCESS path
 * already measured in this tree does (src/transport/lowfive_transport.
 * cpp's own VolSingleton, LowFive::H5VOLProperty::apply) -- instead of
 * relying on HDF5's HDF5_VOL_CONNECTOR/HDF5_PLUGIN_PATH environment
 * route the loadable plugin (src/transport/lowfive_plugin_vol.cpp,
 * target fuse-lowfive-vol, UNCHANGED by this track) uses. See docs/user/
 * lowfive.md's own "Unit 3" and "programmatic attach" accounts and
 * KNOWN_ISSUES.md D39 for why: with the env route, U's and V's own FIRST
 * H5Fcreate (shim.cpp's own bruss_lowfive_write, called from
 * bruss_lowfive_write_export) null-dereferenced inside LowFive's own
 * VOLBase::_file_create, measured on the host 2026-10-12 (KNOWN_ISSUES.md
 * D38's own "What the host must verify" paragraph names the exact crash
 * reports). This header's own attach call sidesteps the question of WHY
 * (the env route's info_t::vol class-static timing) entirely by never
 * taking that route for the participants.
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

/* Call EXACTLY ONCE per process, after MPI_Init() and before the first
 * bruss_lowfive_write()/bruss_lowfive_read() call. `graph_path` is the
 * SAME --graph argument this program's own main() already parsed -- the
 * graph that carries the field-to-hdf5-path bindings
 * (fuse::build_lowfive_bindings(), the SAME function src/transport/
 * lowfive_plugin_graph.cpp's own LoadConfig() calls) and the writer/
 * reader RankSets this attach needs. Unlike the env-route plugin, this
 * shim already has the graph path on its own command line and reads no
 * FUSE_LOWFIVE_BINDINGS environment variable.
 *
 * `physics_comm_f` is this process's own ALREADY-FORMED physics-only
 * communicator (an MPI_Comm_c2f handle) -- u_main.c's own `u_comm`,
 * v_main.f90's own `v_comm` (the SAME communicator bruss_u_create()/
 * bruss_v_create()'s own `comm_f` argument already takes, for the same
 * reason). FOUND WHILE IMPLEMENTING, NOT BEFORE: an earlier version of
 * this function formed a FRESH MPI_Comm_split(MPI_COMM_WORLD, ...) of
 * its own for the distributed rendezvous, exactly mirroring src/
 * transport/lowfive_plugin_vol.cpp's own ConfigureOnce() -- but
 * MPI_Comm_split on MPI_COMM_WORLD is collective over EVERY rank of
 * MPI_COMM_WORLD, including the coupler's, and `bruss_lowfive_attach()`
 * is never called from coupler_main.c at all: U and V would have
 * entered that split while the coupler never did, which hangs (the
 * standing rule this track was asked to audit for: "a collective path
 * must never return on a locally computed condition" -- the mirror
 * image of that rule, a collective ENTERED without every rank's own
 * participation, is the SAME hazard from the other side). Passing in
 * `u_comm`/`v_comm` instead needs NO new collective over MPI_COMM_WORLD
 * at all: `MPI_Intercomm_create`'s own `local_comm` argument is
 * collective only over the union of the two SIDES' own groups (U's
 * physics ranks, V's physics ranks), which the coupler was never part
 * of in the first place, and `u_comm`/`v_comm` are already formed, with
 * the coupler's own matching MPI_Comm_split call already satisfied
 * (u_main.c's own header comment on that split has the full account).
 *
 * On success (return 0), `*fapl_id_out` receives the file-access
 * property list every subsequent bruss_lowfive_write()/
 * bruss_lowfive_read() call on THIS process must pass as `fapl_id`,
 * literally, instead of H5P_DEFAULT.
 *
 * On failure (return nonzero), `*fapl_id_out` is left at -1 and a
 * message is copied into `errbuf` (same truncate-and-NUL-terminate
 * convention as bruss_lowfive_write/read): a malformed graph, no
 * HDF5-bound field in it, an accept-mode component on an HDF5-bound
 * channel this rank takes part in (refused by design -- see
 * KNOWN_ISSUES.md D39; D14's own accept-mode intercommunicator question
 * stays open), two HDF5-bound channels this rank takes part in that do
 * not partition the world the same way, or (distributed mode only) an
 * HDF5/MPI call that itself failed. A second call in the same process is
 * also refused: the VOL registration is process-wide and exactly once
 * (src/transport/lowfive_transport.cpp's own VolSingleton has the full
 * account of why re-registering is not safe).
 *
 * Without -DFUSE_ENABLE_LOWFIVE=ON, returns nonzero immediately with
 * "not built with LowFive" in errbuf, mirroring bruss_lowfive_write/
 * read's own contract; never touches HDF5 at all in that configuration. */
int bruss_lowfive_attach(const char* graph_path, long long physics_comm_f,
                         long long* fapl_id_out,
                         char* errbuf, long long errbuf_len);

/* INTERNAL to the shim -- called only from shim.cpp's own
 * bruss_lowfive_write(), right after its own H5Fclose(), never directly
 * by u_main.c/v_main.f90. No-op success (0) when the last successful
 * bruss_lowfive_attach() chose the single-process connector (nothing to
 * serve across an intercommunicator that was never formed).
 *
 * In distributed mode, consults the pure state machine in lowfive_
 * attach_decision.hpp (serve_state_step()) and either serves now
 * (`LowFive::DistMetadataVOL::serve_all()`) or defers -- UNCHANGED by
 * the round-2 review below: this WRITE-side event/action pair was never
 * the part that was wrong. See lowfive_attach_decision.hpp's own
 * "ROUND 2 REVIEW" comment and KNOWN_ISSUES.md D39 for the measured
 * stack that led here and the two rejected designs along the way.
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
 * role or pending state -- this is a CORRECTION this round, not the
 * original design: a first round-2 attempt flushed a first-reader's own
 * deferred serve HERE, before H5Fopen, and was reviewed against
 * LowFive's own source and rejected before any host run (lowfive_
 * attach_decision.hpp's own "ROUND 2 REVIEW" comment has the full
 * citations and the circular-wait it would have reproduced). The real
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
 * NEW this round, replacing the pre-open flush bruss_lowfive_receive_
 * before_read() used to perform: THIS is the event a first-reader's own
 * deferred serve is flushed at. Read from LowFive's own source (lowfive_
 * attach_decision.hpp's own "ROUND 2 REVIEW" comment): a process's own
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
 * process, always returns 0: there is nothing to have deferred. */
int bruss_lowfive_finalize(char* errbuf, long long errbuf_len);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_ */
