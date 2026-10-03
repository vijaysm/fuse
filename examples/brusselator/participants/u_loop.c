/* Pure C (no C++ anywhere in this file) -- see u_loop.h's own header
 * comment for why that matters here. */
#include "u_loop.h"

#include <string.h>

fuse_status u_loop_step(fuse_ctx* ctx, bruss_u_handle* uh, fuse_step_request* req,
                        double* u_in, double* u_out,
                        fuse_step_report* out_rep,
                        fuse_step_report* out_begin_rep) {
  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) return FUSE_ERR_RUNTIME;

  memset(out_begin_rep, 0, sizeof(*out_begin_rep));
  out_begin_rep->struct_size = sizeof(*out_begin_rep);
  const fuse_status begin_rc = t->begin_step(ctx, req, out_begin_rep);
  if (begin_rc != FUSE_OK) return begin_rc;

  if (out_begin_rep->requires_checkpoint) bruss_u_save_state(uh);
  if (out_begin_rep->requires_restore) bruss_u_restore_state(uh);

  const int step_rc = bruss_u_step(
      uh, out_begin_rep->t_begin.numerator, out_begin_rep->t_begin.denominator,
      out_begin_rep->t_end.numerator, out_begin_rep->t_end.denominator, u_in, u_out);
  if (step_rc != 0) return FUSE_ERR_RUNTIME;

  memset(out_rep, 0, sizeof(*out_rep));
  out_rep->struct_size = sizeof(*out_rep);
  return t->end_step(ctx, req, out_rep);
}
