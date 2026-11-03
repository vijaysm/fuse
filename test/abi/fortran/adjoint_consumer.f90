! The reverse sweep through the Fortran participant interface. Drives the same "upstream" -> "downstream" coupling as
! consumer.f90 for a few periods, the upstream component stepping
!
!   out <- a out,
!
! then reverses it: registers the adjoint array of each field, initializes the reverse sweep, seeds the
! adjoint of the final state of the downstream field, and reverses each step between
! fuse_begin_adjoint_step and fuse_end_adjoint_step, newest period first.
!
! The coupling is a linear map G from the initial upstream values x to the final downstream values y,
! so the adjoint identity <G x, w> = <x, G^T w> must hold for the w that was seeded; the program checks
! it to rounding error, and checks the reports' next_component along the way. It stops with status 1 on
! any mismatch, so it needs no reference output.
program adjoint_consumer
  use, intrinsic :: iso_c_binding
  use ffuse
  implicit none

  integer, parameter :: ncells = 4
  integer, parameter :: nperiods = 3
  real(c_double), parameter :: a = 0.9_c_double

  type(fuse_ctx) :: ctx
  type(fuse_null) :: in_null
  type(fuse_context_out) :: made
  type(fuse_setup_options) :: options
  type(fuse_field_ref) :: up_out, down_in
  type(fuse_field_view) :: view
  type(fuse_step_report) :: report
  integer(c_int32_t) :: status
  real(c_double), target :: up(ncells), down(ncells), up_bar(ncells), down_bar(ncells)
  real(c_double) :: x(ncells), w(ncells), gx_dot_w, x_dot_gtw
  integer :: i, period

  status = fuse_create(ctx, in_null, made)
  call check(status, "create")
  ctx%ptr = made%ctx

  status = fuse_load_graph(ctx, "two_component.yaml", in_null)
  call check(status, "load_graph")

  ! the run ends after nperiods periods, so that no further period starts behind the last step
  options%has_end_time = 1_c_int32_t
  options%end_time = fuse_exact_time(int(nperiods, c_int64_t), 1_c_int64_t)
  status = fuse_setup(ctx, options, in_null)
  call check(status, "setup")

  status = fuse_field_handle(ctx, "upstream", "out", up_out)
  call check(status, "field_handle(upstream.out)")
  status = fuse_field_handle(ctx, "downstream", "in", down_in)
  call check(status, "field_handle(downstream.in)")

  view%element_type = FUSE_ELEMENT_F64
  view%memspace = FUSE_MEMSPACE_HOST
  view%ownership = FUSE_OWNERSHIP_OWNED
  view%components = 1_c_int32_t
  view%count = int(ncells, c_int64_t)
  view%owned_count = int(ncells, c_int64_t)
  view%stride = 1_c_int64_t
  view%component_stride = 1_c_int64_t

  view%handle = up_out%handle
  view%data = c_loc(up)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.out)")
  view%handle = down_in%handle
  view%data = c_loc(down)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.in)")

  status = fuse_initialize(ctx, in_null, in_null)
  call check(status, "initialize")

  ! the forward run, from x
  do i = 1, ncells
    x(i) = 1.0_c_double + 0.25_c_double * real(i * i, c_double)
    w(i) = 2.0_c_double - 0.5_c_double * real(i, c_double)
  end do
  up = x
  down = 0.0_c_double
  do period = 1, nperiods
    status = fuse_begin_step(ctx, "upstream", report)
    call check(status, "begin_step(upstream)")
    up = a * up
    status = fuse_end_step(ctx, "upstream", report)
    call check(status, "end_step(upstream)")
    status = fuse_begin_step(ctx, "downstream", report)
    call check(status, "begin_step(downstream)")
    status = fuse_end_step(ctx, "downstream", report)
    call check(status, "end_step(downstream)")
  end do
  gx_dot_w = dot_product(down, w)

  ! the adjoint arrays: the caller's own, registered with the layout of the field itself
  up_bar = 0.0_c_double
  down_bar = 0.0_c_double
  view%handle = up_out%handle
  view%data = c_loc(up_bar)
  status = fuse_register_adjoint_field(ctx, view, in_null)
  call check(status, "register_adjoint_field(upstream.out)")
  view%handle = down_in%handle
  view%data = c_loc(down_bar)
  status = fuse_register_adjoint_field(ctx, view, in_null)
  call check(status, "register_adjoint_field(downstream.in)")
  status = fuse_initialize_adjoint(ctx, in_null, in_null)
  call check(status, "initialize_adjoint")

  ! the seed: the adjoint of the final downstream values
  down_bar = w

  ! the reverse run, newest period first and each period's steps in the reverse of the forward order
  do period = nperiods, 1, -1
    status = fuse_begin_adjoint_step(ctx, "downstream", report)
    call check(status, "begin_adjoint_step(downstream)")
    ! the downstream component has no state of its own to reverse
    status = fuse_end_adjoint_step(ctx, "downstream", report)
    call check(status, "end_adjoint_step(downstream)")
    ! the upstream component is the one declared first: next_component names it
    if (report%next_component /= 0_c_int32_t) call fail("end_adjoint_step must name the upstream component next")

    status = fuse_begin_adjoint_step(ctx, "upstream", report)
    call check(status, "begin_adjoint_step(upstream)")
    up_bar = a * up_bar
    status = fuse_end_adjoint_step(ctx, "upstream", report)
    call check(status, "end_adjoint_step(upstream)")
    if (report%next_component /= -1_c_int32_t) call fail("a reversed period must report next_component = -1")
  end do

  x_dot_gtw = dot_product(x, up_bar)
  if (abs(gx_dot_w - x_dot_gtw) > 1.0e-12_c_double * max(abs(gx_dot_w), 1.0_c_double)) then
    write(0, '(A,ES24.16,A,ES24.16)') "adjoint_consumer: <Gx,w> = ", gx_dot_w, " but <x,GTw> = ", x_dot_gtw
    call fail("the adjoint identity does not hold")
  end if

  status = fuse_destroy(ctx, in_null, in_null)
  call check(status, "destroy")
  write(*, '(A)') "adjoint_consumer: ok"

contains

  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(0, '(A,A)') "adjoint_consumer: ", message
    stop 1
  end subroutine fail

  subroutine check(call_status, step)
    integer(c_int32_t), intent(in) :: call_status
    character(len=*), intent(in) :: step
    character(len=512) :: detail
    logical :: truncated
    integer(c_int32_t) :: error_status

    if (call_status /= FUSE_OK) then
      detail = ' '
      error_status = fuse_last_error(ctx, detail, truncated)
      if (error_status /= FUSE_OK) detail = '(last_error unavailable)'
      write(0, '(A,A,A,I0,A,A)') "adjoint_consumer: ", step, " failed, status=", &
        call_status, ": ", trim(detail)
      stop 1
    end if
  end subroutine check

end program adjoint_consumer
