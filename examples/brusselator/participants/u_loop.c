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

  /* fuse_step_report::step_opened's own comment (entries.def): resuming
   * past a pending delivery proxy can legitimately roll this call into a
   * NEW period whose own first request names a different component than
   * "U" -- a time-window member cannot know from outside whether the
   * coupling iteration it shares a window with has converged. No step
   * opened in that case (out_begin_rep->next_component names who IS due
   * instead): nothing to checkpoint, no physics to run, no end_step() to
   * call. The caller's own loop body is the one place that must notice
   * and call fuse_begin_step() again with that next_component -- this
   * function only reports FUSE_OK with *out_rep left zeroed. */
  if (!out_begin_rep->step_opened) {
    memset(out_rep, 0, sizeof(*out_rep));
    out_rep->struct_size = sizeof(*out_rep);
    /* KNOWN_ISSUES.md D57, Round 2. u_main.c's own loop (its header
     * comment's own account) has exactly one way to notice a period has
     * closed: `if (rep.next_component == -1) break;` -- *out_rep*, never
     * *out_begin_rep*, which this function's own doc comment above
     * already says is what names who IS due instead. Leaving next_
     * component at its zeroed default (0) here was wrong: 0 is also a
     * valid ComponentId, so u_main.c's own check can never distinguish
     * "nothing is due, the run just ended" from "component 0 is due" --
     * and in particular can never see the one case that matters most,
     * fuse_begin_step()'s own correctly-reported next_component == -1 at
     * a genuine, no-end-time-declared period boundary (KNOWN_ISSUES.md
     * D57's own fix, src/abi/participant.cpp's prime_next_request()):
     * with that signal silently zeroed to 0 here, u_main.c's loop never
     * breaks, calls fuse_begin_step() again, and that FRESH call's own
     * ordinary (never-bounded-when-!has_end_time) first resume_period()
     * call starts a period the caller's own loop never asked for --
     * measured on the host as U/V hanging in that unrequested period's
     * own base-communicator Constrain reduction while the coupler's own
     * bounded Runner::run(periods) had already exited. Copying this one
     * field across is enough: every OTHER case this branch also covers
     * (a delivery-pending stop, an ordinary rollover naming a real
     * component) already has out_begin_rep->next_component set to that
     * real, non -1 component id, so u_main.c's own check correctly keeps
     * looping for those, exactly as it already did. */
    out_rep->next_component = out_begin_rep->next_component;
    return FUSE_OK;
  }

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
