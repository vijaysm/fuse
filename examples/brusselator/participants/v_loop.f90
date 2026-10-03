! Time windows track (Conformance, Unit 2): the ONE place "drive one
! externally-driven step of V" is written IN FORTRAN -- fuse_begin_step()
! (through ffuse's own Fortran wrapper), the checkpoint/restore dance,
! the shim's own bruss_v_step() (called here through a bind(C) interface
! onto shim.cpp's C-ABI export), then fuse_end_step() -- so v_main.f90's
! own time loop and test/verification/brusselator_participants_test.cpp's
! own tier-0 harness (a C++ gtest binary) call the SAME code rather than
! two hand-copies that could drift. PURE FORTRAN, deliberately: v_main.f90
! is the half of Conformance Unit 2 that demonstrates a Fortran code
! driving FUSE through the ABI in Fortran, so the loop body itself -- not
! just the program -- has to be Fortran (the lead's own correction on
! this point; an earlier version of this session's own work had this
! loop body shared as C++, which made both mains thin wrappers around a
! C++ helper and never exercised ffuse's own Fortran bindings with a real
! time loop).
!
! v_loop_step is `bind(C, name="bruss_v_loop_step")` so the C++ tier-0
! harness can call it through an ordinary `extern "C"` prototype
! (v_loop.h) the same way it calls u_loop_step -- the harness itself
! never needs to know which half of a given call is Fortran underneath.
! `ctx` crosses as the opaque `type(c_ptr)` ffuse.f90's own raw
! `ffuse_<entry>` interfaces already use for the identical argument
! (docs/user/fortran.md's own type-mapping table, "fuse_ctx* ... opaque").
module v_loop_mod
  use, intrinsic :: iso_c_binding
  use ffuse
  implicit none
  private
  public :: v_loop_step
  public :: bruss_v_create, bruss_v_destroy, bruss_v_size, bruss_v_step
  public :: bruss_v_save_state, bruss_v_restore_state, bruss_v_state
  public :: bruss_v_exchange_in, bruss_v_last_error

  ! ---- shim.h's own bruss_v_* C-ABI functions (shim.cpp, a C++
  ! translation unit exporting extern "C" -- this interface block is
  ! what lets pure Fortran call them without ever seeing a C++ type). ----
  interface
    function bruss_v_create(nx_u, ny_u, nx_v, ny_v, diffusion) result(h) &
        bind(C, name="bruss_v_create")
      import :: c_int, c_double, c_ptr
      integer(c_int), value :: nx_u, ny_u, nx_v, ny_v
      real(c_double), value :: diffusion
      type(c_ptr) :: h
    end function bruss_v_create

    subroutine bruss_v_destroy(h) bind(C, name="bruss_v_destroy")
      import :: c_ptr
      type(c_ptr), value :: h
    end subroutine bruss_v_destroy

    function bruss_v_size(h) result(n) bind(C, name="bruss_v_size")
      import :: c_ptr, c_int
      type(c_ptr), value :: h
      integer(c_int) :: n
    end function bruss_v_size

    function bruss_v_step(h, t_begin_num, t_begin_den, t_end_num, t_end_den, &
        required, provided_out) result(rc) bind(C, name="bruss_v_step")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: h
      integer(c_long_long), value :: t_begin_num, t_begin_den, t_end_num, t_end_den
      type(c_ptr), value :: required
      type(c_ptr), value :: provided_out
      integer(c_int) :: rc
    end function bruss_v_step

    subroutine bruss_v_save_state(h) bind(C, name="bruss_v_save_state")
      import :: c_ptr
      type(c_ptr), value :: h
    end subroutine bruss_v_save_state

    subroutine bruss_v_restore_state(h) bind(C, name="bruss_v_restore_state")
      import :: c_ptr
      type(c_ptr), value :: h
    end subroutine bruss_v_restore_state

    subroutine bruss_v_state(h, out) bind(C, name="bruss_v_state")
      import :: c_ptr
      type(c_ptr), value :: h
      type(c_ptr), value :: out
    end subroutine bruss_v_state

    function bruss_v_exchange_in(h) result(x) bind(C, name="bruss_v_exchange_in")
      import :: c_ptr, c_double
      type(c_ptr), value :: h
      real(c_double) :: x
    end function bruss_v_exchange_in

    function bruss_v_last_error(h) result(msg) bind(C, name="bruss_v_last_error")
      import :: c_ptr
      type(c_ptr), value :: h
      type(c_ptr) :: msg
    end function bruss_v_last_error
  end interface

contains

  ! Drives ONE step of the externally-driven component named by `req%
  ! component` (must be "V"): fuse_begin_step() (ffuse's own Fortran
  ! wrapper, which resolves fuse_coupler_abi() internally -- this module
  ! never touches that table directly, docs/user/fortran.md's own "Fortran
  ! does not get the C side's one-exported-symbol table"), the checkpoint/
  ! restore dance, bruss_v_step() (v_in/v_out are the caller's own
  ! registered field buffers, passed as raw c_ptr -- this function never
  ! needs their length, only bruss_v_step() does, through its own
  ! bruss_v_size(vh)), then fuse_end_step(). Returns the first non-FUSE_OK
  ! status encountered, in call order: begin_step()'s own status;
  ! FUSE_ERR_RUNTIME if bruss_v_step() itself fails; otherwise
  ! end_step()'s own status. `out_rep`/`out_begin_rep` mirror u_loop_step's
  ! own two report outputs exactly (out_begin_rep is always written, even
  ! on a begin_step failure).
  function v_loop_step(ctx, vh, req, v_in, v_out, out_rep, out_begin_rep) &
      result(status) bind(C, name="bruss_v_loop_step")
    type(c_ptr), value :: ctx
    type(c_ptr), value :: vh
    type(fuse_step_request), intent(inout) :: req
    type(c_ptr), value :: v_in
    type(c_ptr), value :: v_out
    type(fuse_step_report), intent(inout) :: out_rep
    type(fuse_step_report), intent(inout) :: out_begin_rep
    integer(c_int32_t) :: status

    type(fuse_ctx) :: fctx
    integer(c_int) :: step_rc

    fctx%ptr = ctx

    status = fuse_begin_step(fctx, req, out_begin_rep)
    if (status /= FUSE_OK) return

    if (out_begin_rep%requires_checkpoint /= 0) call bruss_v_save_state(vh)
    if (out_begin_rep%requires_restore /= 0) call bruss_v_restore_state(vh)

    step_rc = bruss_v_step(vh, out_begin_rep%t_begin%numerator, out_begin_rep%t_begin%denominator, &
                           out_begin_rep%t_end%numerator, out_begin_rep%t_end%denominator, &
                           v_in, v_out)
    if (step_rc /= 0) then
      status = FUSE_ERR_RUNTIME
      return
    end if

    status = fuse_end_step(fctx, req, out_rep)
  end function v_loop_step

end module v_loop_mod
