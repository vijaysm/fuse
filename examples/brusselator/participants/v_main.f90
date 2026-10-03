! Time windows track (Conformance, Unit 2): the Fortran participant for
! component "V". PURE FORTRAN -- no C, no C++, anywhere in this file (the
! lead's own constraint: "a foreign code never touches C++"). Uses ffuse
! (fortran/ffuse.f90, generated from the same include/fuse/abi/entries.def
! the C header is) and v_loop_mod (v_loop.f90 -- the SAME v_loop_step the
! tier-0 harness, test/verification/brusselator_participants_test.cpp,
! drives, and which this program's own time loop calls below), plus
! v_loop_mod's own `bruss_v_*` interfaces onto shim.cpp's C-ABI physics
! wrapper.
!
! GRAPH AUTHORING vs PARTICIPATING (Q2): this program never builds a
! fuse::CouplingGraph -- it calls fuse_load_graph() on a YAML path a
! SEPARATE tool, brusselator_graph_tool, already wrote, and nothing else
! constructs the graph.
!
! LAUNCH MODE FROM THE GRAPH, NOT FROM A FLAG: this program always passes
! `communicator = MPI_COMM_WORLD` (already the Fortran integer handle --
! Fortran's own MPI binding never needs MPI_Comm_c2f(), unlike the C side,
! because the Fortran communicator IS the Fortran representation already)
! and `my_component = "V"` to fuse_setup(); fuse_setup() itself resolves
! "V"'s LaunchSpec from the loaded graph and only calls fuse::transport::
! connect() when that LaunchSpec says to (u_main.c's own header comment,
! and fuse_setup_options::my_component's own comment in entries.def, have
! the full account).
!
! UNLIKE u_main.c, this program never bundles coupler ranks: Q1's ruling
! bundles the coupler only into the ACCEPTOR side (U's executable); V's
! own executable is always "pure V", in both mpmd and accept layouts
! (brusselator_graph_tool.cpp's own header comment has the layout
! rationale). So every rank of this program's own MPI_COMM_WORLD is a
! real V-physics rank -- no --ranks-v flag is needed here.
!
! LOG FORMAT: decimal, not hexfloat -- Fortran has no printf("%a")
! equivalent, so this program writes `ES25.17E3` (17 significant digits,
! the same DBL_DECIMAL_DIG exactness guarantee hexfloat gives) rather than
! u_main.c's own `%a`. brusselator_check_core.cpp's own log parser
! (`sscanf(..., "%la", ...)`) accepts both: a scanf floating conversion
! (a/A/e/E/f/F/g/G) is ONE conversion, specified (C99/C11 7.21.6.2) to
! accept anything strtod() accepts, decimal or hex -- only printf
! distinguishes the two. No format-reconciliation work was needed on the
! C++ side for this reason.
program v_main
  use, intrinsic :: iso_c_binding
  use mpi
  use ffuse
  use v_loop_mod
  implicit none

  integer, parameter :: nx_u = 3, ny_u = 3, nx_v = 4, ny_v = 4  ! brusselator_graph_tool.cpp's own kNxU/kNyU/kNxV/kNyV
  real(c_double), parameter :: kDiffusion = 10.0d0

  character(len=512) :: graph_path, log_path, arg
  integer :: periods, ierr, world_rank, world_size, nargs, i, log_unit, p
  type(fuse_ctx) :: ctx
  type(fuse_null) :: nothing
  type(fuse_context_out) :: made
  type(fuse_setup_options) :: setup
  type(fuse_field_ref) :: v_out_ref, v_in_ref
  type(fuse_field_view) :: view
  integer(c_int32_t) :: status
  type(c_ptr) :: vh
  real(c_double), target, allocatable :: v_out(:), v_in(:)
  real(c_double), allocatable, target :: v_native_state(:)
  character(len=2), target :: my_component_name
  integer :: nu
  type(fuse_step_request) :: req
  type(fuse_step_report) :: rep, begin_rep
  real(c_double) :: mean_v

  graph_path = ""
  log_path = ""
  periods = 2
  nargs = command_argument_count()
  i = 1
  do while (i <= nargs)
    call get_command_argument(i, arg)
    if (trim(arg) == "--graph" .and. i < nargs) then
      call get_command_argument(i + 1, graph_path)
      i = i + 2
    else if (trim(arg) == "--log" .and. i < nargs) then
      call get_command_argument(i + 1, log_path)
      i = i + 2
    else if (trim(arg) == "--periods" .and. i < nargs) then
      call get_command_argument(i + 1, arg)
      read(arg, *) periods
      i = i + 2
    else
      write(0, '(A,A)') "v_main: unrecognized argument: ", trim(arg)
      call exit(1)
    end if
  end do
  if (len_trim(graph_path) == 0 .or. len_trim(log_path) == 0) then
    write(0, '(A)') "usage: v_main --graph graph.yaml --log v.log [--periods N]"
    call exit(1)
  end if

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, world_size, ierr)

  status = fuse_create(ctx, nothing, made)
  call check(status, "create")
  ctx%ptr = made%ctx

  status = fuse_load_graph(ctx, trim(graph_path), nothing)
  call check(status, "load_graph")

  ! The ONE field a foreign caller sets to take part in a multi-
  ! executable run: my_component. communicator is this executable's OWN
  ! world (already MPI_Init'd, and already the Fortran handle --
  ! fuse_setup_options%communicator's own comment in entries.def
  ! documents both conventions); fuse_setup() resolves "V"'s LaunchSpec
  ! from the loaded graph and connects accordingly.
  my_component_name = "V" // c_null_char
  setup%communicator = int(MPI_COMM_WORLD, c_int64_t)
  setup%my_component = c_loc(my_component_name)
  status = fuse_setup(ctx, setup, nothing)
  call check(status, "setup")

  nu = nx_u * ny_u  ! both of V's window-transfer fields live on U's own
                    ! grid -- species_component.hpp's own file header
                    ! explains why (brusselator_graph.hpp mirrors it)
  allocate(v_out(nu), v_in(nu))
  v_out = 0.0d0
  v_in = 0.0d0

  status = fuse_field_handle(ctx, "V", "v_out", v_out_ref)
  call check(status, "field_handle(v_out)")
  view%handle = v_out_ref%handle
  view%element_type = FUSE_ELEMENT_F64
  view%memspace = FUSE_MEMSPACE_HOST
  view%ownership = FUSE_OWNERSHIP_OWNED
  view%components = 1
  view%count = int(nu, c_int64_t)
  view%owned_count = int(nu, c_int64_t)
  view%stride = 1_c_int64_t
  view%data = c_loc(v_out)
  status = fuse_register_field(ctx, view, nothing)
  call check(status, "register_field(v_out)")

  status = fuse_field_handle(ctx, "V", "v_in", v_in_ref)
  call check(status, "field_handle(v_in)")
  view%handle = v_in_ref%handle
  view%data = c_loc(v_in)
  status = fuse_register_field(ctx, view, nothing)
  call check(status, "register_field(v_in)")

  vh = bruss_v_create(nx_u, ny_u, nx_v, ny_v, kDiffusion)
  if (.not. c_associated(vh)) then
    write(0, '(A)') "FAIL(v_main): bruss_v_create returned NULL"
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  status = fuse_initialize(ctx, nothing, nothing)
  call check(status, "initialize")

  open(newunit=log_unit, file=trim(log_path), status="replace", action="write")

  allocate(v_native_state(nx_v * ny_v))
  call bruss_v_state(vh, c_loc(v_native_state))
  mean_v = sum(v_native_state) / real(size(v_native_state), c_double)
  write(log_unit, '(A,ES25.17E3,A,ES25.17E3)') &
      "window=0 mean_v=", mean_v, " exchange_in=", bruss_v_exchange_in(vh)

  req%component = c_loc(my_component_name)  ! "V" + NUL, built above

  do p = 0, periods - 1
    ! This process hosts only "V" (shim.h's own precedent) -- like
    ! u_main.c, it never needs next_component: no other component is
    ! ever due on this process's own ranks, so it simply loops calling
    ! v_loop_step() for "V" until the PERIOD closes.
    do
      status = v_loop_step(ctx%ptr, vh, req, c_loc(v_in), c_loc(v_out), rep, begin_rep)
      call check(status, "v_loop_step")
      if (rep%next_component == -1) exit  ! period complete
    end do

    call bruss_v_state(vh, c_loc(v_native_state))
    mean_v = sum(v_native_state) / real(size(v_native_state), c_double)
    write(log_unit, '(A,I0,A,ES25.17E3,A,ES25.17E3)') &
        "window=", p + 1, " mean_v=", mean_v, " exchange_in=", bruss_v_exchange_in(vh)
  end do

  close(log_unit)
  call bruss_v_destroy(vh)
  deallocate(v_out, v_in, v_native_state)

  status = fuse_destroy(ctx, nothing, nothing)
  call check(status, "destroy")

  call MPI_Finalize(ierr)

contains

  subroutine check(status_in, what)
    integer(c_int32_t), intent(in) :: status_in
    character(len=*), intent(in) :: what
    integer :: abort_ierr
    if (status_in /= FUSE_OK) then
      write(0, '(A,A,A,I0)') "FAIL(v_main): ", what, ": status=", status_in
      call MPI_Abort(MPI_COMM_WORLD, 1, abort_ierr)
    end if
  end subroutine check

end program v_main
