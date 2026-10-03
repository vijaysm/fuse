#ifndef FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_V_LOOP_H_
#define FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_V_LOOP_H_

/* C-callable prototype for v_loop.f90's own `v_loop_step` function
 * (module v_loop_mod), exported as `bruss_v_loop_step` via that
 * function's own `bind(C, name="bruss_v_loop_step")`. This header is
 * how test/verification/brusselator_participants_test.cpp and
 * brusselator_check_test.cpp (both C++ gtest binaries) call the SAME
 * Fortran-written loop body v_main.f90's own program calls -- the
 * harness never needs to know which half of a given call is Fortran
 * underneath. `v_in`/`v_out` cross as `void*` (Fortran's own dummy
 * arguments are `type(c_ptr), value`, interoperable with exactly this);
 * `out_begin_rep` is REQUIRED, not nullable (a Fortran bind(C) dummy of
 * non-pointer derived type cannot be OPTIONAL and still interoperate
 * cleanly with a C NULL the way u_loop_step's sibling parameter could in
 * principle -- this header keeps the two signatures symmetric by making
 * neither one nullable). */

#include "fuse/abi/participant.h"

#ifdef __cplusplus
extern "C" {
#endif

fuse_status bruss_v_loop_step(fuse_ctx* ctx, void* vh, fuse_step_request* req,
                              void* v_in, void* v_out,
                              fuse_step_report* out_rep,
                              fuse_step_report* out_begin_rep);

#ifdef __cplusplus
}
#endif

#endif /* FUSE_EXAMPLES_BRUSSELATOR_PARTICIPANTS_V_LOOP_H_ */
