! capi_test.f90 -- the Fortran shim's test.
!
! This is a Fortran program, compiled by a Fortran compiler and linked against
! libsoma_core. Nothing less proves the shim: a C++ test that inspects
! soma_mod.f90 as text would pass against a module that does not compile, and
! a C test would exercise the C ABI rather than the binding to it.
!
! The data directory arrives as argv(1) rather than through a preprocessor
! macro. See test/CMakeLists.txt for why.
program soma_fortran_test
  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_int32_t,   &
                                        c_int64_t, c_double
  use soma_mod
  implicit none

  integer :: failures
  character(len=4096) :: data_dir

  failures = 0

  if (command_argument_count() < 1) then
     write(*,*) 'FAIL: usage: soma_fortran_test <test-data-dir>'
     stop 1
  end if
  call get_command_argument(1, data_dir)

  call test_version(failures)
  call test_lifecycle_and_errors(failures)
  call test_scalar_field_and_stepping(failures)
  call test_rank2_component_order(failures)
  call test_rank2_layout_accepted(failures)

  if (failures == 0) then
     write(*,'(a)') ' soma_fortran_test: all checks passed'
     stop
  else
     write(*,'(a,i0,a)') ' soma_fortran_test: ', failures, ' failure(s)'
     stop 1
  end if

contains

  subroutine fail(msg, count)
    character(len=*), intent(in)    :: msg
    integer,          intent(inout) :: count
    write(*,'(a,a)') ' FAIL: ', msg
    count = count + 1
  end subroutine fail

  !> Assert an exact ABI return code. `ierr /= 0` is not enough: the whole
  !> point of five distinct codes is that a caller can branch on them, and a
  !> test that accepts any of them would pass if NOTFOUND silently became
  !> RUNTIME.
  subroutine expect_rc(actual, expected, what, count)
    integer(c_int),   intent(in)    :: actual, expected
    character(len=*), intent(in)    :: what
    integer,          intent(inout) :: count
    if (actual /= expected) then
       write(*,'(a,a,a,i0,a,i0)') ' FAIL: ', what, ': expected rc ',          &
            expected, ', got ', actual
       count = count + 1
    end if
  end subroutine expect_rc

  subroutine expect_true(cond, what, count)
    logical,          intent(in)    :: cond
    character(len=*), intent(in)    :: what
    integer,          intent(inout) :: count
    if (.not. cond) call fail(what, count)
  end subroutine expect_true

  function yaml(name) result(path)
    character(len=*), intent(in)  :: name
    character(len=:), allocatable :: path
    path = trim(data_dir) // '/' // name
  end function yaml

  ! ==========================================================================

  subroutine test_version(count)
    integer, intent(inout) :: count
    character(len=64) :: v
    call expect_rc(soma_f_abi_version(), soma_abi_version_header,             &
                   'library ABI version disagrees with soma_mod', count)
    call soma_f_version(v)
    call expect_true(len_trim(v) > 0, 'version string is empty', count)
    ! The string must be NUL-stripped and blank-padded, not NUL-padded.
    call expect_true(index(v, achar(0)) == 0,                                 &
                     'version string still carries a NUL', count)
    write(*,'(a,a,a,i0)') ' soma version ', trim(v), ', ABI ',                &
         soma_f_abi_version()
  end subroutine test_version

  ! --------------------------------------------------------------------------
  ! Handle validity, the state machine, and the two error message slots.
  ! --------------------------------------------------------------------------
  subroutine test_lifecycle_and_errors(count)
    integer, intent(inout) :: count
    type(soma_context)  :: ctx, unset
    integer(c_int)      :: ierr
    integer(c_int32_t)  :: handle
    integer(c_int64_t)  :: n
    character(len=1024) :: message

    ! An unset context is a null handle. Every entry point must reject it
    ! rather than dereference it.
    call expect_true(.not. soma_f_is_open(unset),                             &
                     'a default context claims to be open', count)
    call soma_f_initialize(unset, ierr)
    call expect_rc(ierr, soma_err_invalid, 'initialize on a null handle',     &
                   count)
    ! ... and the reason is retrievable, because the null handle selects the
    ! thread-local slot rather than a context that does not exist.
    call soma_f_last_error(unset, message)
    call expect_true(len_trim(message) > 0,                                   &
                     'no thread-local message after a null-handle call', count)

    call soma_f_create(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'create', count)
    call expect_true(soma_f_is_open(ctx), 'create left the handle closed',    &
                     count)

    ! EMPTY state: no graph yet.
    call soma_f_field_handle(ctx, 'fluid', 'heat_flux', handle, ierr)
    call expect_rc(ierr, soma_err_state, 'field_handle before a graph', count)

    ! A path that does not exist. This is also the round trip that matters
    ! most: a Fortran string in, a NUL-terminated C string out, and a C
    ! message back into a blank-padded Fortran string.
    call soma_f_load_yaml(ctx, yaml('definitely_not_here.yaml'), ierr)
    call expect_rc(ierr, soma_err_invalid, 'load of a missing file', count)
    call soma_f_last_error(ctx, message)
    call expect_true(index(message, 'definitely_not_here.yaml') > 0,          &
                     'the message does not quote the path we passed', count)
    call expect_true(index(message, achar(0)) == 0,                           &
                     'the message still carries a NUL', count)
    call soma_f_last_error_length(ctx, n, ierr)
    call expect_rc(ierr, soma_ok, 'last_error_length', count)
    call expect_true(int(n) == len_trim(message),                             &
                     'last_error_length disagrees with the copied message',   &
                     count)

    ! Truncation into a short buffer must still terminate cleanly.
    block
      character(len=8) :: tiny
      call soma_f_last_error(ctx, tiny)
      call expect_true(len_trim(tiny) == 8 .and. index(tiny, achar(0)) == 0,  &
                       'a truncated message is not blank-padded', count)
    end block

    call soma_f_load_yaml(ctx, yaml('pair.yaml'), ierr)
    call expect_rc(ierr, soma_ok, 'load pair.yaml', count)
    call soma_f_load_yaml(ctx, yaml('pair.yaml'), ierr)
    call expect_rc(ierr, soma_err_state, 'a second graph was accepted', count)

    ! Rational end time. There is no real-valued overload on purpose.
    call soma_f_set_end_time(ctx, 1_c_int64_t, 3_c_int64_t, ierr)
    call expect_rc(ierr, soma_ok, 'set_end_time 1/3', count)
    call soma_f_set_end_time(ctx, 1_c_int64_t, 0_c_int64_t, ierr)
    call expect_rc(ierr, soma_err_invalid, 'a zero denominator was accepted', &
                   count)

    ! The descriptor of spec 3.2, through the two-call idiom.
    call soma_f_descriptor_size(ctx, n, ierr)
    call expect_rc(ierr, soma_ok, 'descriptor_size', count)
    call expect_true(n > 0_c_int64_t, 'descriptor size is not positive', count)
    block
      integer(c_int8_t), allocatable, target :: buf(:)
      integer(c_int64_t) :: got
      allocate(buf(n))
      call soma_f_save_descriptor(ctx, buf, got, ierr)
      call expect_rc(ierr, soma_ok, 'save_descriptor', count)
      call expect_true(got == n, 'save_descriptor size disagrees with the '// &
                       'query', count)
      ! Too small: still reports the size needed.
      got = 0_c_int64_t
      call soma_f_save_descriptor(ctx, buf(1:1), got, ierr)
      call expect_rc(ierr, soma_err_invalid, 'an undersized buffer was '   // &
                     'accepted', count)
      call expect_true(got == n, 'a failed save_descriptor did not report ' //&
                       'the size needed', count)

      ! Round-trip the descriptor -- the narrow waist of spec 3.2 -- through
      ! a second context. This is the path a Fortran front-end uses to author
      ! in YAML and hand the bytes to somebody else.
      block
        type(soma_context) :: ctx2
        integer(c_int32_t) :: h2
        call soma_f_create(ctx2, ierr)
        call expect_rc(ierr, soma_ok, 'create for the descriptor round trip', &
                       count)
        call soma_f_load_descriptor(ctx2, buf, ierr)
        call expect_rc(ierr, soma_ok, 'load_descriptor', count)
        call soma_f_field_handle(ctx2, 'fluid', 'heat_flux', h2, ierr)
        call expect_rc(ierr, soma_ok,                                         &
                       'the round-tripped graph lost fluid.heat_flux', count)
        ! An empty descriptor is refused by the shim, before SOMA sees it.
        call soma_f_load_descriptor(ctx2, buf(1:0), ierr)
        call expect_rc(ierr, soma_err_invalid,                                &
                       'an empty descriptor was accepted', count)
        call soma_f_destroy(ctx2, ierr)
        call expect_rc(ierr, soma_ok, 'destroy 2', count)
      end block
      deallocate(buf)
    end block

    call soma_f_destroy(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'destroy', count)
    call expect_true(.not. soma_f_is_open(ctx),                               &
                     'destroy left the handle open', count)
    ! Reported, not undefined.
    call soma_f_destroy(ctx, ierr)
    call expect_rc(ierr, soma_err_invalid, 'a double destroy was accepted',   &
                   count)
  end subroutine test_lifecycle_and_errors

  ! --------------------------------------------------------------------------
  ! The scalar path the brief describes, plus the begin_step/end_step loop a
  ! Fortran component actually drives. soma_run is NOT papered over: it is
  ! asserted to be unsupported.
  ! --------------------------------------------------------------------------
  subroutine test_scalar_field_and_stepping(count)
    integer, intent(inout) :: count
    type(soma_context) :: ctx
    integer(c_int)     :: ierr
    integer(c_int32_t) :: hf, hw
    integer            :: i
    logical            :: flag
    real(c_double)     :: t0, t1
    ! TARGET, module-lifetime-equivalent (they live for the whole subroutine,
    ! which outlives the context). See the ARRAYS section of soma_mod.
    real(c_double), target :: flux(16), wall(16)
    real(c_double), target :: scratch(32)
    character(len=512) :: message

    call soma_f_create(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'create', count)
    call soma_f_load_yaml(ctx, yaml('pair.yaml'), ierr)
    if (ierr /= soma_ok) then
       call soma_f_last_error(ctx, message)
       write(*,'(a,a)') ' yaml load error: ', trim(message)
       call fail('yaml load failed', count)
       return
    end if

    call soma_f_field_handle(ctx, 'fluid', 'heat_flux', hf, ierr)
    call expect_rc(ierr, soma_ok, 'handle fluid.heat_flux', count)
    call expect_true(hf >= 0, 'handle is negative', count)
    call soma_f_field_handle(ctx, 'solid', 'q_wall', hw, ierr)
    call expect_rc(ierr, soma_ok, 'handle solid.q_wall', count)

    ! A missing field is a lookup that found nothing, not a malformed call.
    call soma_f_field_handle(ctx, 'fluid', 'nope', hf, ierr)
    call expect_rc(ierr, soma_err_notfound, 'a missing field', count)
    call expect_true(hf == -1, 'a failed lookup did not clear the handle',    &
                     count)
    call soma_f_field_handle(ctx, 'nosuch', 'heat_flux', hf, ierr)
    call expect_rc(ierr, soma_err_notfound, 'a missing component', count)
    ! Trailing blanks in a name must not reach SOMA.
    call soma_f_field_handle(ctx, 'fluid   ', 'heat_flux  ', hf, ierr)
    call expect_rc(ierr, soma_ok, 'a blank-padded name was not trimmed', count)

    do i = 1, 16
       flux(i) = real(i, c_double)
       wall(i) = 0.0_c_double
    end do

    ! A non-contiguous slice must be refused. If the dummy were declared
    ! CONTIGUOUS the compiler would copy this into a temporary and SOMA would
    ! record the address of storage that dies at the closing parenthesis --
    ! silent, and undetectable afterwards.
    call soma_f_register_field(ctx, hw, scratch(1:16:2), ierr)
    call expect_rc(ierr, soma_err_invalid, 'a strided slice was accepted',    &
                   count)
    ! ... and refusing it must not have registered anything, so the real
    ! registration below still succeeds.

    call soma_f_register_field(ctx, hf, flux, ierr)
    call expect_rc(ierr, soma_ok, 'register fluid.heat_flux', count)
    call soma_f_register_field(ctx, hw, wall, ierr)
    call expect_rc(ierr, soma_ok, 'register solid.q_wall', count)
    ! Rebinding is refused, never a silent replacement.
    call soma_f_register_field(ctx, hf, flux, ierr)
    call expect_rc(ierr, soma_err_state, 'a rebind was accepted', count)

    ! Not initialised yet.
    call soma_f_begin_step(ctx, 'fluid', t0, t1, ierr)
    call expect_rc(ierr, soma_err_state, 'begin_step before initialize', count)

    call soma_f_initialize(ctx, ierr)
    if (ierr /= soma_ok) then
       call soma_f_last_error(ctx, message)
       write(*,'(a,a)') ' initialize error: ', trim(message)
    end if
    call expect_rc(ierr, soma_ok, 'initialize', count)

    ! Spec 6 mode 2: the Fortran caller IS the component. SOMA's own driver
    ! is not available through this ABI and the shim does not pretend it is.
    call soma_f_run_period(ctx, ierr)
    call expect_rc(ierr, soma_err_unsupported, 'run_period is not reserved',  &
                   count)
    call soma_f_run(ctx, 4_c_int64_t, ierr)
    call expect_rc(ierr, soma_err_unsupported, 'run is not reserved', count)
    call soma_f_run(ctx, -1_c_int64_t, ierr)
    call expect_rc(ierr, soma_err_invalid, 'run accepted negative periods',   &
                   count)

    ! fluid's interval is 1/2; the end time set here is never reached in this
    ! test, so coupling stays ongoing.
    call soma_f_is_coupling_ongoing(ctx, flag, ierr)
    call expect_rc(ierr, soma_ok, 'is_coupling_ongoing', count)
    call expect_true(flag, 'coupling is over before it began', count)

    t0 = -1.0_c_double
    t1 = -1.0_c_double
    call soma_f_begin_step(ctx, 'fluid', t0, t1, ierr)
    call expect_rc(ierr, soma_ok, 'begin fluid', count)
    call expect_true(t0 == 0.0_c_double .and. t1 == 0.5_c_double,             &
                     'fluid window 1 is not [0, 1/2)', count)
    call soma_f_requires_checkpoint(ctx, 'fluid', flag, ierr)
    call expect_rc(ierr, soma_ok, 'requires_checkpoint', count)
    call expect_true(.not. flag,                                              &
                     'a component in no iteration region wants a checkpoint', &
                     count)
    call soma_f_requires_restore(ctx, 'fluid', flag, ierr)
    call expect_rc(ierr, soma_ok, 'requires_restore', count)
    call expect_true(.not. flag, 'restore wanted outside an iteration region',&
                     count)
    call soma_f_end_step(ctx, 'fluid', ierr)
    call expect_rc(ierr, soma_ok, 'end fluid', count)

    call soma_f_begin_step(ctx, 'fluid', t0, t1, ierr)
    call expect_rc(ierr, soma_ok, 'begin fluid 2', count)
    call expect_true(t0 == 0.5_c_double .and. t1 == 1.0_c_double,             &
                     'fluid window 2 is not [1/2, 1)', count)
    call soma_f_end_step(ctx, 'fluid', ierr)
    call expect_rc(ierr, soma_ok, 'end fluid 2', count)

    ! solid's interval is 1, so its first window spans both of fluid's.
    call soma_f_begin_step(ctx, 'solid', t0, t1, ierr)
    call expect_rc(ierr, soma_ok, 'begin solid', count)
    call expect_true(t0 == 0.0_c_double .and. t1 == 1.0_c_double,             &
                     'solid window 1 is not [0, 1)', count)
    call soma_f_end_step(ctx, 'solid', ierr)
    call expect_rc(ierr, soma_ok, 'end solid', count)

    call soma_f_begin_step(ctx, 'nosuch', t0, t1, ierr)
    call expect_rc(ierr, soma_err_notfound, 'begin_step on a bad component',  &
                   count)

    call soma_f_destroy(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'destroy', count)
  end subroutine test_scalar_field_and_stepping

  ! --------------------------------------------------------------------------
  ! Column-major vs. row-major, from the negative side.
  !
  ! `array(ncomp, ncell)` must be read as ncomp components of ncell elements.
  ! Registering a 3-by-5 array against a field the graph declares with ONE
  ! component makes SOMA quote the component count the shim derived, so the
  ! message distinguishes the two readings: "3" is right, "5" is a
  ! transposition. A square array could not tell them apart, which is why the
  ! array is 3 by 5 and not 3 by 3.
  ! --------------------------------------------------------------------------
  subroutine test_rank2_component_order(count)
    integer, intent(inout) :: count
    type(soma_context) :: ctx
    integer(c_int)     :: ierr
    integer(c_int32_t) :: hf
    real(c_double), target :: a35(3, 5), a53(5, 3)
    character(len=512) :: message

    a35 = 0.0_c_double
    a53 = 0.0_c_double

    call soma_f_create(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'create', count)
    call soma_f_load_yaml(ctx, yaml('pair.yaml'), ierr)
    call expect_rc(ierr, soma_ok, 'load pair.yaml', count)
    call soma_f_field_handle(ctx, 'fluid', 'heat_flux', hf, ierr)
    call expect_rc(ierr, soma_ok, 'handle fluid.heat_flux', count)

    call soma_f_register_field(ctx, hf, a35, ierr)
    call expect_rc(ierr, soma_err_invalid,                                    &
                   'a 3-component array on a 1-component field', count)
    call soma_f_last_error(ctx, message)
    call expect_true(index(message, 'registered with 3') > 0,                 &
                     'the first extent is not being read as the component ' //&
                     'count: ' // trim(message), count)

    call soma_f_register_field(ctx, hf, a53, ierr)
    call expect_rc(ierr, soma_err_invalid,                                    &
                   'a 5-component array on a 1-component field', count)
    call soma_f_last_error(ctx, message)
    call expect_true(index(message, 'registered with 5') > 0,                 &
                     'the transposed array was not read as 5 components: '  //&
                     trim(message), count)

    call soma_f_destroy(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'destroy', count)
  end subroutine test_rank2_component_order

  ! --------------------------------------------------------------------------
  ! ... and from the positive side, against a field that really has three
  ! components.
  !
  ! This is the check that pins the STRIDES, not just the counts. For a
  ! 3-by-5 array the shim must say stride = 3 (element to element) and
  ! component_stride = 1. The opposite reading -- stride = 1,
  ! component_stride = 3 -- is what you get by treating the array as
  ! `array(ncell, ncomp)`, and Layout::well_formed() rejects it here
  ! (interleaved needs stride >= 3*3 = 9; blocked needs
  ! component_stride >= count*stride = 5). It would NOT be rejected for a
  ! square array, where 3 >= 3*1 holds. Hence 3 by 5.
  ! --------------------------------------------------------------------------
  subroutine test_rank2_layout_accepted(count)
    integer, intent(inout) :: count
    type(soma_context) :: ctx
    integer(c_int)     :: ierr
    integer(c_int32_t) :: hd
    integer            :: c, i
    real(c_double), target :: disp(3, 5)
    character(len=512) :: message

    do i = 1, 5
       do c = 1, 3
          disp(c, i) = real(100 * i + c, c_double)
       end do
    end do

    call soma_f_create(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'create', count)
    call soma_f_load_yaml(ctx, yaml('vector_field.yaml'), ierr)
    if (ierr /= soma_ok) then
       call soma_f_last_error(ctx, message)
       write(*,'(a,a)') ' vector_field.yaml load error: ', trim(message)
       call fail('vector_field.yaml did not load', count)
       return
    end if
    call soma_f_field_handle(ctx, 'shell', 'displacement', hd, ierr)
    call expect_rc(ierr, soma_ok, 'handle shell.displacement', count)

    call soma_f_register_field(ctx, hd, disp, ierr)
    if (ierr /= soma_ok) then
       call soma_f_last_error(ctx, message)
       write(*,'(a,a)') ' register error: ', trim(message)
    end if
    call expect_rc(ierr, soma_ok,                                             &
                   'a 3-by-5 array on a 3-component field', count)

    ! The transpose must not be accepted in its place.
    block
      type(soma_context) :: ctx2
      real(c_double), target :: bad(5, 3)
      bad = 0.0_c_double
      call soma_f_create(ctx2, ierr)
      call expect_rc(ierr, soma_ok, 'create 2', count)
      call soma_f_load_yaml(ctx2, yaml('vector_field.yaml'), ierr)
      call expect_rc(ierr, soma_ok, 'load vector_field.yaml 2', count)
      call soma_f_field_handle(ctx2, 'shell', 'displacement', hd, ierr)
      call expect_rc(ierr, soma_ok, 'handle shell.displacement 2', count)
      call soma_f_register_field(ctx2, hd, bad, ierr)
      call expect_rc(ierr, soma_err_invalid,                                  &
                     'a 5-by-3 array on a 3-component field was accepted',    &
                     count)
      call soma_f_destroy(ctx2, ierr)
      call expect_rc(ierr, soma_ok, 'destroy 2', count)
    end block

    call soma_f_destroy(ctx, ierr)
    call expect_rc(ierr, soma_ok, 'destroy', count)
  end subroutine test_rank2_layout_accepted

end program soma_fortran_test
