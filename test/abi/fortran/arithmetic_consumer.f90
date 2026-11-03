! Double-double values and a forward-mode gradient through the Fortran participant interface
!. Drives the same "upstream" ->
! "downstream" coupling as consumer.f90, over two channels:
!
!   dd:   a double-double value per cell, FUSE_ELEMENT_DD. The graph declares components: 2
!         (it counts binary64 numbers), and the view counts values, so view%components is 1.
!         The array is real(c_double) :: d(2, n): (hi, lo) of each cell, limb index first.
!   grad: a value and two derivatives per cell, FUSE_ELEMENT_F64 with components: 3 and
!         gradient_size = 2, the last two of each cell's three numbers.
!
! The program checks, bit for bit, that what the upstream wrote (including the low limb of
! every double-double value) is in the downstream arrays once the downstream step has closed, and that fuse_view_field reports
! the element type, the components in values, and the gradient_size that were registered. It
! stops with status 1 on any mismatch, so it needs no reference output.
program arithmetic_consumer
  use, intrinsic :: iso_c_binding
  use ffuse
  implicit none

  integer, parameter :: ncells = 4
  integer, parameter :: nperiods = 2

  type(fuse_ctx) :: ctx
  type(fuse_null) :: in_null
  type(fuse_context_out) :: made
  type(fuse_field_ref) :: up_dd, down_dd, up_grad, down_grad
  type(fuse_field_view) :: view, back
  type(fuse_step_report) :: report
  integer(c_int32_t) :: status
  real(c_double), target :: up_d(2, ncells), down_d(2, ncells)
  real(c_double), target :: up_g(3, ncells), down_g(3, ncells)
  integer :: i, k, period

  status = fuse_create(ctx, in_null, made)
  call check(status, "create")
  ctx%ptr = made%ctx

  status = fuse_load_graph(ctx, "dd_two_component.yaml", in_null)
  call check(status, "load_graph")

  status = fuse_field_handle(ctx, "upstream", "dd", up_dd)
  call check(status, "field_handle(upstream.dd)")
  status = fuse_field_handle(ctx, "downstream", "dd", down_dd)
  call check(status, "field_handle(downstream.dd)")
  status = fuse_field_handle(ctx, "upstream", "grad", up_grad)
  call check(status, "field_handle(upstream.grad)")
  status = fuse_field_handle(ctx, "downstream", "grad", down_grad)
  call check(status, "field_handle(downstream.grad)")
  if (up_dd%components /= 2_c_int32_t) call fail("the graph counts two binary64 numbers per dd value")

  up_d = 0.0_c_double
  down_d = 0.0_c_double
  up_g = 0.0_c_double
  down_g = 0.0_c_double

  ! the double-double fields: values, not numbers, are counted by the view
  view%element_type = FUSE_ELEMENT_DD
  view%memspace = FUSE_MEMSPACE_HOST
  view%ownership = FUSE_OWNERSHIP_OWNED
  view%count = int(ncells, c_int64_t)
  view%owned_count = int(ncells, c_int64_t)
  view%stride = 1_c_int64_t
  view%component_stride = 1_c_int64_t
  view%gradient_size = 0_c_int32_t

  view%handle = up_dd%handle
  view%components = up_dd%components / 2_c_int32_t
  view%data = c_loc(up_d)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.dd)")

  view%handle = down_dd%handle
  view%components = down_dd%components / 2_c_int32_t
  view%data = c_loc(down_d)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.dd)")

  ! the gradient fields: three numbers per cell, the last two the gradient of the first
  view%element_type = FUSE_ELEMENT_F64
  view%stride = 3_c_int64_t

  view%handle = up_grad%handle
  view%components = up_grad%components
  view%gradient_size = 2_c_int32_t
  view%data = c_loc(up_g)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.grad)")

  view%handle = down_grad%handle
  view%components = down_grad%components
  view%data = c_loc(down_g)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.grad)")

  ! what the interface reports back
  status = fuse_view_field(ctx, up_dd, back)
  call check(status, "view_field(upstream.dd)")
  if (back%element_type /= FUSE_ELEMENT_DD) call fail("view_field lost the element type")
  if (back%components /= 1_c_int32_t) call fail("view_field must count double-double values")
  status = fuse_view_field(ctx, up_grad, back)
  call check(status, "view_field(upstream.grad)")
  if (back%gradient_size /= 2_c_int32_t) call fail("view_field lost the gradient_size")

  status = fuse_initialize(ctx, in_null, in_null)
  call check(status, "initialize")

  do period = 0, nperiods - 1
    status = fuse_begin_step(ctx, "upstream", report)
    call check(status, "begin_step(upstream)")
    do i = 1, ncells
      up_d(1, i) = 1.0_c_double + real(i, c_double) + real(period, c_double)
      up_d(2, i) = scale(1.0_c_double, -60) * real(i, c_double)
      up_g(1, i) = 10.0_c_double + real(i, c_double)
      up_g(2, i) = 0.5_c_double * real(i, c_double)
      up_g(3, i) = -0.25_c_double * real(i + period, c_double)
    end do
    status = fuse_end_step(ctx, "upstream", report)
    call check(status, "end_step(upstream)")

    status = fuse_begin_step(ctx, "downstream", report)
    call check(status, "begin_step(downstream)")
    status = fuse_end_step(ctx, "downstream", report)
    call check(status, "end_step(downstream)")

    ! the exchange of this period is complete when the downstream step has closed
    do i = 1, ncells
      do k = 1, 2
        if (transfer(down_d(k, i), 0_c_int64_t) /= transfer(up_d(k, i), 0_c_int64_t)) then
          write(0, '(A,I0,A,I0,A,I0)') "arithmetic_consumer: period ", period, " cell ", i, &
            " limb ", k
          call fail("a double-double limb did not arrive bit for bit")
        end if
      end do
      do k = 1, 3
        if (transfer(down_g(k, i), 0_c_int64_t) /= transfer(up_g(k, i), 0_c_int64_t)) then
          call fail("a value or gradient entry did not arrive bit for bit")
        end if
      end do
    end do
  end do

  status = fuse_destroy(ctx, in_null, in_null)
  call check(status, "destroy")
  write(*, '(A)') "arithmetic_consumer: ok"

contains

  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(0, '(A,A)') "arithmetic_consumer: ", message
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
      write(0, '(A,A,A,I0,A,A)') "arithmetic_consumer: ", step, " failed, status=", &
        call_status, ": ", trim(detail)
      stop 1
    end if
  end subroutine check

end program arithmetic_consumer
