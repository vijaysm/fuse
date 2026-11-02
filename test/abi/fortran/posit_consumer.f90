! Posit fields through the Fortran participant interface (plans/2026-10-22-universal-number-types.md,
! stage U5). Drives the same "upstream" -> "downstream" coupling as consumer.f90, over three channels:
!
!   p16:  posit<16, 2> to posit<16, 2>, FUSE_ELEMENT_POSIT16. A posit array is the integer pattern of its
!         values: integer(c_int16_t) holds posit16 and integer(c_int32_t) posit32, so Fortran
!         has no posit type to declare and the program handles patterns. The patterns are
!         arbitrary (a posit is exactly the value of its pattern), and each must arrive unchanged.
!   p32:  posit<32, 2> to posit<32, 2>, FUSE_ELEMENT_POSIT32, the same.
!   mix:  real(c_double) upstream to a posit16 downstream: the coupler rounds once on delivery, to
!         nearest, a tie to the even pattern. Four values with known patterns: 1 is 0x4000, 1 + 2^-12 is
!         the tie between 0x4000 and 0x4001 (it goes to the even 0x4000), 1 + 2^-12 + 2^-30 is above the tie
!         (0x4001), and -1 is 0xC000 (the two's complement of 0x4000).
!
! The program stops with status 1 on any mismatch, so it needs no reference output.
program posit_consumer
  use, intrinsic :: iso_c_binding
  use ffuse
  implicit none

  integer, parameter :: ncells = 4
  integer, parameter :: nperiods = 2

  type(fuse_ctx) :: ctx
  type(fuse_null) :: in_null
  type(fuse_context_out) :: made
  type(fuse_field_ref) :: up_p16, down_p16, up_p32, down_p32, up_mix, down_mix
  type(fuse_field_view) :: view, back
  type(fuse_step_report) :: report
  integer(c_int32_t) :: status
  integer(c_int16_t), target :: up_a(ncells), down_a(ncells), down_m(ncells)
  integer(c_int32_t), target :: up_b(ncells), down_b(ncells)
  real(c_double), target :: up_m(ncells)
  ! the patterns of 1, 1 + 2^-12 (a tie, to the even one), 1 + 2^-12 + 2^-30, -1
  integer(c_int16_t), parameter :: want_mix(ncells) = &
    [16384_c_int16_t, 16384_c_int16_t, 16385_c_int16_t, -16384_c_int16_t]
  integer :: i, period

  status = fuse_create(ctx, in_null, made)
  call check(status, "create")
  ctx%ptr = made%ctx

  status = fuse_load_graph(ctx, "posit_two_component.yaml", in_null)
  call check(status, "load_graph")

  status = fuse_field_handle(ctx, "upstream", "p16", up_p16)
  call check(status, "field_handle(upstream.p16)")
  status = fuse_field_handle(ctx, "downstream", "p16", down_p16)
  call check(status, "field_handle(downstream.p16)")
  status = fuse_field_handle(ctx, "upstream", "p32", up_p32)
  call check(status, "field_handle(upstream.p32)")
  status = fuse_field_handle(ctx, "downstream", "p32", down_p32)
  call check(status, "field_handle(downstream.p32)")
  status = fuse_field_handle(ctx, "upstream", "mix", up_mix)
  call check(status, "field_handle(upstream.mix)")
  status = fuse_field_handle(ctx, "downstream", "mix", down_mix)
  call check(status, "field_handle(downstream.mix)")

  up_a = 0_c_int16_t
  down_a = 0_c_int16_t
  down_m = 0_c_int16_t
  up_b = 0_c_int32_t
  down_b = 0_c_int32_t
  up_m = 0.0_c_double

  view%memspace = FUSE_MEMSPACE_HOST
  view%ownership = FUSE_OWNERSHIP_OWNED
  view%count = int(ncells, c_int64_t)
  view%owned_count = int(ncells, c_int64_t)
  view%stride = 1_c_int64_t
  view%component_stride = 1_c_int64_t
  view%gradient_size = 0_c_int32_t
  view%components = 1_c_int32_t

  view%element_type = FUSE_ELEMENT_POSIT16
  view%handle = up_p16%handle
  view%data = c_loc(up_a)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.p16)")
  view%handle = down_p16%handle
  view%data = c_loc(down_a)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.p16)")

  view%element_type = FUSE_ELEMENT_POSIT32
  view%handle = up_p32%handle
  view%data = c_loc(up_b)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.p32)")
  view%handle = down_p32%handle
  view%data = c_loc(down_b)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.p32)")

  view%element_type = FUSE_ELEMENT_F64
  view%handle = up_mix%handle
  view%data = c_loc(up_m)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.mix)")
  view%element_type = FUSE_ELEMENT_POSIT16
  view%handle = down_mix%handle
  view%data = c_loc(down_m)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.mix)")

  ! what the interface reports back
  status = fuse_view_field(ctx, up_p16, back)
  call check(status, "view_field(upstream.p16)")
  if (back%element_type /= FUSE_ELEMENT_POSIT16) call fail("view_field lost the posit16 element type")
  status = fuse_view_field(ctx, up_p32, back)
  call check(status, "view_field(upstream.p32)")
  if (back%element_type /= FUSE_ELEMENT_POSIT32) call fail("view_field lost the posit32 element type")
  status = fuse_view_field(ctx, down_mix, back)
  call check(status, "view_field(downstream.mix)")
  if (back%element_type /= FUSE_ELEMENT_POSIT16) call fail("view_field lost the element type of a converted field")

  status = fuse_initialize(ctx, in_null, in_null)
  call check(status, "initialize")

  do period = 0, nperiods - 1
    status = fuse_begin_step(ctx, "upstream", report)
    call check(status, "begin_step(upstream)")
    do i = 1, ncells
      up_a(i) = int(16384 + 37 * i + period, c_int16_t)
      up_b(i) = int(1073741824 + 123457 * i + period, c_int32_t)
    end do
    up_m = [1.0_c_double, &
            1.0_c_double + scale(1.0_c_double, -12), &
            1.0_c_double + scale(1.0_c_double, -12) + scale(1.0_c_double, -30), &
            -1.0_c_double]
    status = fuse_end_step(ctx, "upstream", report)
    call check(status, "end_step(upstream)")

    status = fuse_begin_step(ctx, "downstream", report)
    call check(status, "begin_step(downstream)")
    status = fuse_end_step(ctx, "downstream", report)
    call check(status, "end_step(downstream)")

    ! the exchange of this period is complete when the downstream step has closed
    do i = 1, ncells
      if (down_a(i) /= up_a(i)) then
        write(0, '(A,I0,A,I0)') "posit_consumer: period ", period, " cell ", i
        call fail("a posit16 pattern did not arrive unchanged")
      end if
      if (down_b(i) /= up_b(i)) then
        write(0, '(A,I0,A,I0)') "posit_consumer: period ", period, " cell ", i
        call fail("a posit32 pattern did not arrive unchanged")
      end if
      if (down_m(i) /= want_mix(i)) then
        write(0, '(A,I0,A,I0,A,I0)') "posit_consumer: period ", period, " cell ", i, " got ", int(down_m(i))
        call fail("a binary64 value was not rounded to the right posit16 pattern on delivery")
      end if
    end do
  end do

  status = fuse_destroy(ctx, in_null, in_null)
  call check(status, "destroy")
  write(*, '(A)') "posit_consumer: ok"

contains

  subroutine fail(message)
    character(len=*), intent(in) :: message
    write(0, '(A,A)') "posit_consumer: ", message
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
      write(0, '(A,A,A,I0,A,A)') "posit_consumer: ", step, " failed, status=", &
        call_status, ": ", trim(detail)
      stop 1
    end if
  end subroutine check

end program posit_consumer
