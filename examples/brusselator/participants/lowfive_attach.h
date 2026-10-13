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
 * serve across an intercommunicator that was never formed); in
 * distributed mode, pushes the just-closed file across the
 * intercommunicator (`LowFive::DistMetadataVOL::serve_all()`), mirroring
 * `fuse::LowFiveTransport::serve_as_producer()`'s own placement (called
 * "by the producer side after it has written ... and called its own,
 * unmodified H5Fclose", fuse/transport/transport.hpp). Refused
 * (nonzero, errbuf set) if bruss_lowfive_attach() was never called
 * successfully in this process. */
int bruss_lowfive_serve_after_write(char* errbuf, long long errbuf_len);

/* INTERNAL to the shim -- called only from shim.cpp's own
 * bruss_lowfive_read(), right before its own H5Fopen(), never directly
 * by u_main.c/v_main.f90. No-op success (0) in single-process mode; in
 * distributed mode, receives the peer's file index across the
 * intercommunicator (`LowFive::DistMetadataVOL::broadcast_files()`),
 * mirroring `fuse::LowFiveTransport::receive_as_consumer()`'s own
 * placement ("called by the CONSUMER side BEFORE its own, unmodified
 * H5Fopen"). Refused (nonzero, errbuf set) if bruss_lowfive_attach() was
 * never called successfully in this process. */
int bruss_lowfive_receive_before_read(char* errbuf, long long errbuf_len);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_LOWFIVE_ATTACH_H_ */
