#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_U_LOOP_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_U_LOOP_H_

/* The ONE place "drive one
 * externally-driven step of U" is written IN C -- fuse_begin_step(), the
 * checkpoint/restore dance fuse_step_report::requires_checkpoint/
 * requires_restore asks for, the shim's own
 * bruss_u_step(), then fuse_end_step() -- so u_main.c's own time loop and
 * test/verification/brusselator_participants_test.cpp's own tier-0
 * harness call the SAME code rather than two hand-copies that could
 * drift. PURE C, deliberately: u_main.c demonstrates a C code driving
 * FUSE through the ABI in C, so the loop body itself -- not just main() --
 * has to be C, not a thin wrapper around a C++ helper.
 * Resolves fuse_coupler_abi(FUSE_ABI_VERSION) internally; takes `ctx`
 * only, never the `fuse_participant_abi_v1*` table itself.
 */

#include "fuse/abi/participant.h"

#include "shim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Drives ONE step of the externally-driven component named by `req->
 * component` (must be "U"): fuse_begin_step(), the checkpoint/restore
 * dance, bruss_u_step() (u_in/u_out are the caller's own registered
 * field buffers -- already bound to the graph through
 * fuse_register_field() before this is ever called), then
 * fuse_end_step(). Returns the first non-FUSE_OK status encountered, in
 * call order: begin_step()'s own status; FUSE_ERR_RUNTIME if
 * bruss_u_step() itself fails (bruss_u_last_error(uh) has the message);
 * otherwise end_step()'s own status. `*out_rep` is end_step()'s own
 * report (meaningful only when this returns FUSE_OK), carrying
 * next_component for the caller's own turn-order loop. `*out_begin_rep`
 * is begin_step()'s own report (always written, even on failure) -- the
 * tier-0 harness checks its next_component == -2 invariant; u_main.c's
 * own loop has no use for it but still supplies a buffer (no optional
 * pointers at this boundary, to keep this signature identical in spirit
 * to v_loop_step's own bind(C) one, which cannot take a NULL-able
 * non-pointer dummy argument). */
fuse_status u_loop_step(fuse_ctx* ctx, bruss_u_handle* uh, fuse_step_request* req,
                        double* u_in, double* u_out,
                        fuse_step_report* out_rep,
                        fuse_step_report* out_begin_rep);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_U_LOOP_H_ */
