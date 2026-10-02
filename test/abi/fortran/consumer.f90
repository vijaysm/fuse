! Pure-Fortran install-tree consumer for the participant ABI
! (build-evidence/abi-matrix.toml, ABI-FORTRAN-CONSUMER), modelled on
! test/abi/install_consumer/main.c (the C sibling, the other S6-phase-2
! track). Links ONLY fuse-abi-fortran, which in turn needs libfuse-abi at
! FINAL LINK, not at this file's own compilation (fortran/CMakeLists.txt's
! header explains why a STATIC archive defers that). Drives the SAME
! "upstream" -> "downstream" graph test/abi/graph_fixture.hpp builds in C++
! and main.c spells out as YAML for itself (its own header explains why:
! that translation unit cannot link fuse::model to serialize the graph, and
! neither can this one) -- two_component.yaml here is kept textually
! identical to main.c's kGraphYaml, verified by hand against the same
! structure, specifically SO a value mismatch between this program and the
! C consumer is a real finding and not an artifact of two different
! graphs. Not lagged, matching graph_fixture.hpp's own current fixture
! (its header comment records why: lag changes type bookkeeping, not
! timing, and an earlier lagged version of this exact graph reproduced a
! real FUSE_ERR_RUNTIME/E-IR-TYPE failure at initialize() -- see
! two_component.yaml's own header for the measured error text). 4 cells,
! 2 periods, upstream.out[i] = 10.0 + period, matching main.c's values
! exactly.
!
! Prints every value FUSE handed back as an exact byte pattern (the Z edit
! descriptor on TRANSFER(x, 0_int64)), per the task's own instruction,
! rather than a decimal. OPEN QUESTION (see the task report): main.c prints
! the same IEEE double with C's "%a" hex-FLOAT notation (e.g. "0x1.4p+3"),
! which is not the same text as this program's raw hex BIT pattern (e.g.
! "4014000000000000") even when the underlying value agrees bit-for-bit --
! a literal byte-diff of the two programs' stdout will not pass as written
! even on a correct run. Reconcile the two sides' text format (or make the
! comparison script decode both into bits before comparing) at integration;
! this is a cross-track formatting mismatch, not a bug in either value.
program consumer
  use, intrinsic :: iso_c_binding
  use ffuse
  implicit none

  integer, parameter :: FUSE_CELL_COUNT = 4
  integer, parameter :: FUSE_PERIODS = 2

  type(fuse_ctx) :: ctx
  type(fuse_null) :: in_null
  type(fuse_context_out) :: made
  type(fuse_field_ref) :: upstream_out, downstream_in
  type(fuse_field_view) :: view
  type(fuse_step_report) :: report
  integer(c_int32_t) :: status
  real(c_double), target :: up_storage(FUSE_CELL_COUNT)
  real(c_double), target :: down_storage(FUSE_CELL_COUNT)
  integer :: i, period

  ! create(): ctx must be NULL, satisfied by fuse_ctx's default c_null_ptr.
  status = fuse_create(ctx, in_null, made)
  call check(status, "create")
  ctx%ptr = made%ctx

  ! load_graph(): the string convenience overload (abi-design.md, "Strings")
  ! builds the NUL-terminated yaml_path for us.
  status = fuse_load_graph(ctx, "two_component.yaml", in_null)
  call check(status, "load_graph")

  status = fuse_field_handle(ctx, "upstream", "out", upstream_out)
  call check(status, "field_handle(upstream.out)")
  status = fuse_field_handle(ctx, "downstream", "in", downstream_in)
  call check(status, "field_handle(downstream.in)")

  up_storage = 0.0_c_double
  down_storage = 0.0_c_double

  view%handle = upstream_out%handle
  view%element_type = FUSE_ELEMENT_F64
  view%memspace = FUSE_MEMSPACE_HOST
  view%ownership = FUSE_OWNERSHIP_OWNED
  view%components = upstream_out%components
  view%count = int(FUSE_CELL_COUNT, c_int64_t)
  view%owned_count = int(FUSE_CELL_COUNT, c_int64_t)
  view%stride = 1_c_int64_t
  view%component_stride = 1_c_int64_t
  view%data = c_loc(up_storage)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(upstream.out)")

  view%handle = downstream_in%handle
  view%components = downstream_in%components
  view%data = c_loc(down_storage)
  status = fuse_register_field(ctx, view, in_null)
  call check(status, "register_field(downstream.in)")

  status = fuse_initialize(ctx, in_null, in_null)
  call check(status, "initialize")

  do period = 0, FUSE_PERIODS - 1
    status = fuse_begin_step(ctx, "upstream", report)
    call check(status, "begin_step(upstream)")
    do i = 1, FUSE_CELL_COUNT
      up_storage(i) = 10.0_c_double + real(period, c_double)
    end do
    status = fuse_end_step(ctx, "upstream", report)
    call check(status, "end_step(upstream)")

    status = fuse_begin_step(ctx, "downstream", report)
    call check(status, "begin_step(downstream)")
    do i = 1, FUSE_CELL_COUNT
      ! Every exchanged value as an exact byte pattern -- see the file header.
      write(*, '(A,I0,A,I0,A,Z16)') "period ", period, " downstream.in[", i - 1, &
        "] = ", transfer(down_storage(i), 0_c_int64_t)
    end do
    status = fuse_end_step(ctx, "downstream", report)
    call check(status, "end_step(downstream)")
  end do

  status = fuse_destroy(ctx, in_null, in_null)
  call check(status, "destroy")

contains

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
      write(0, '(A,A,A,I0,A,A)') "consumer: ", step, " failed, status=", &
        call_status, ": ", trim(detail)
      stop 1
    end if
  end subroutine check

end program consumer
