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
  character(len=16) :: transport
  integer :: periods, ierr, world_rank, world_size, nargs, i, log_unit, p
  type(fuse_ctx) :: ctx
  type(fuse_null) :: nothing
  type(fuse_context_out) :: made
  type(fuse_setup_options) :: setup
  type(fuse_field_ref) :: v_out_ref, v_in_ref, u_out_peer_ref
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
  logical :: use_lowfive
  integer(c_int) :: io_rc
  character(len=2048), target :: errbuf

  ! S6 phase 3 (Conformance, "LowFive" track), Unit 3b. Mirrors u_main.c's
  ! own hardcoded hdf5_path/window constants (that file's own header
  ! comment has the full rationale: no ABI query exists for a field's own
  ! hdf5_path, or for the graph's own window length, so both participant
  ! mains reproduce brusselator_graph_tool.cpp's own build-time constants
  ! rather than query them). V's own export is v_out (brusselator_v.h5/
  ! grid/v_out); its import is u_out's own hdf5_path (brusselator_u.h5/
  ! grid/u_out) -- the two mains' own constants are each other's peer,
  ! by construction.
  ! NOT `parameter`: c_loc() (used below, where these are passed to the
  ! bruss_lowfive_* C-ABI calls) requires an actual TARGET variable, which
  ! a compile-time named constant cannot be in Fortran -- these are
  ! assigned once, just below, and never changed again.
  character(len=17), target :: kOwnFile, kPeerFile
  character(len=12), target :: kOwnDataset, kPeerDataset

  ! ---- the two bruss_lowfive_* HDF5 calls and the one build-config
  ! query, through shim.cpp's own extern "C" export (shim.h's own header
  ! comment has the full account of why this exists at all: no HDF5
  ! Fortran module ships with this host's /opt/hdf5). ----
  interface
    function bruss_lowfive_available() result(avail) &
        bind(C, name="bruss_lowfive_available")
      import :: c_int
      integer(c_int) :: avail
    end function bruss_lowfive_available

    function bruss_lowfive_write(file, dataset, data, n, errbuf, errbuf_len) &
        result(rc) bind(C, name="bruss_lowfive_write")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: file, dataset, data, errbuf
      integer(c_long_long), value :: n, errbuf_len
      integer(c_int) :: rc
    end function bruss_lowfive_write

    function bruss_lowfive_read(file, dataset, data, n, errbuf, errbuf_len) &
        result(rc) bind(C, name="bruss_lowfive_read")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: file, dataset, data, errbuf
      integer(c_long_long), value :: n, errbuf_len
      integer(c_int) :: rc
    end function bruss_lowfive_read
  end interface

  graph_path = ""
  log_path = ""
  periods = 2
  transport = "fuse"
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
    else if (trim(arg) == "--transport" .and. i < nargs) then
      call get_command_argument(i + 1, arg)
      transport = trim(arg)
      i = i + 2
    else
      write(0, '(A,A)') "v_main: unrecognized argument: ", trim(arg)
      call exit(1)
    end if
  end do
  if (len_trim(graph_path) == 0 .or. len_trim(log_path) == 0) then
    write(0, '(A)') "usage: v_main --graph graph.yaml --log v.log [--periods N] " // &
        "[--transport fuse|lowfive]"
    call exit(1)
  end if
  if (trim(transport) /= "fuse" .and. trim(transport) /= "lowfive") then
    write(0, '(A,A,A)') "v_main: --transport must be 'fuse' or 'lowfive' (got '", &
        trim(transport), "')"
    call exit(1)
  end if
  use_lowfive = (trim(transport) == "lowfive")
  if (use_lowfive .and. bruss_lowfive_available() == 0) then
    write(0, '(A)') "v_main: --transport lowfive requires a build configured with " // &
        "-DFUSE_ENABLE_LOWFIVE=ON; this binary was not"
    call exit(1)
  end if
  ! V's own export is v_out (brusselator_v.h5/grid/v_out); its import is
  ! U's own export (brusselator_u.h5/grid/u_out) -- u_main.c's own
  ! U_LOWFIVE_OWN_FILE/U_LOWFIVE_PEER_FILE constants are this pair's
  ! mirror image, by construction (both mains reproduce brusselator_
  ! graph_tool.cpp's own hardcoded hdf5_path strings).
  kOwnFile = "brusselator_v.h5" // c_null_char
  kOwnDataset = "/grid/v_out" // c_null_char
  kPeerFile = "brusselator_u.h5" // c_null_char
  kPeerDataset = "/grid/u_out" // c_null_char

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

  ! delivery_field (docs/user/c-abi.md's own protocol) names the
  ! delivering channel's own SOURCE field -- "u_out" for the channel this
  ! component CONSUMES (a2b's own source, not "v_in", its own target) --
  ! so matching it needs the PEER's own field handle, not just this
  ! component's own v_out/v_in. A graph-wide lookup (any component's
  ! name, not just "V") -- fuse_field_handle() resolves straight off the
  ! loaded graph, which every process has a full copy of regardless of
  ! which components it hosts.
  u_out_peer_ref%handle = -1
  if (use_lowfive) then
    status = fuse_field_handle(ctx, "U", "u_out", u_out_peer_ref)
    call check(status, "field_handle(u_out)")
  end if

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

  ! Cold start (docs/user/c-abi.md's own `delivery_pending` section; the
  ! harness's own precedent, test/verification/
  ! brusselator_lowfive_harness_test.cpp): period 0's very first
  ! ExternalDelivery suspension is hit before any step at all, inside
  ! fuse_begin_step()'s own first-ever call -- too late for this process
  ! to know ahead of time it must deliver first. Write this component's
  ! own INITIAL (pre-any-step) v_out and read the peer's -- constant over
  ! the window by construction, nothing has run yet to vary either --
  ! unconditionally, once, before the loop.
  if (use_lowfive) then
    call v_lowfive_write_export()
    call v_lowfive_read_import()
  end if

  do p = 0, periods - 1
    ! This process hosts only "V" (shim.h's own precedent) -- like
    ! u_main.c, it never needs next_component: no other component is
    ! ever due on this process's own ranks, so it simply loops calling
    ! v_loop_step() for "V" until the PERIOD closes.
    do
      status = v_loop_step(ctx%ptr, vh, req, c_loc(v_in), c_loc(v_out), rep, begin_rep)
      call check(status, "v_loop_step")

      ! docs/user/c-abi.md's own `delivery_pending`/`delivery_field`
      ! protocol: on a report naming field F, write F if producing it or
      ! read F if consuming it, then begin_step() again -- never
      ! elsewhere, and never both unconditionally (a graph with more
      ! than one independently-paced LowFive-bound channel can report a
      ! field THIS component is neither producing nor consuming at all,
      ! this round's own corrected rule). fuse_view_field(v_out_ref),
      ! not v_out itself, still holds the sweep that JUST completed
      ! (nothing has overwritten it yet, since this call's own physics
      ! -- if any ran -- is for the NEXT sweep, not this report's).
      if (use_lowfive .and. begin_rep%delivery_pending /= 0) then
        if (begin_rep%delivery_field == v_out_ref%handle) then
          call v_lowfive_write_export()
        else if (begin_rep%delivery_field == u_out_peer_ref%handle) then
          call v_lowfive_read_import()
        end if
      end if

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

  ! `errbuf` is a fixed-length Fortran buffer the C side NUL-terminates
  ! (shim.h's own convention) but never blank-pads -- trim() alone would
  ! print the whole 2048-byte buffer, garbage past the NUL included.
  function cstr_message(buf) result(msg)
    character(len=*), intent(in) :: buf
    character(len=:), allocatable :: msg
    integer :: nul_pos
    nul_pos = index(buf, c_null_char)
    if (nul_pos > 1) then
      msg = buf(1:nul_pos - 1)
    else
      msg = ""
    end if
  end function cstr_message

  ! view%data (fuse_view_field(v_out_ref)), NOT v_out itself: the
  ! application's job is the window's own K-wide waveform buffer, not the
  ! single-sample native storage register_field() was given. Shared by
  ! the cold-start write and the steady-state delivery_pending-gated one
  ! above -- same call, two call sites (u_main.c's own
  ! u_lowfive_write_export mirrors this).
  subroutine v_lowfive_write_export()
    status = fuse_view_field(ctx, v_out_ref, view)
    call check(status, "view_field(v_out)")
    if (view%window_samples > 0 .and. c_associated(view%data)) then
      io_rc = bruss_lowfive_write(c_loc(kOwnFile), c_loc(kOwnDataset), &
          view%data, int(view%count, c_long_long) * int(view%components, c_long_long), &
          c_loc(errbuf), int(len(errbuf), c_long_long))
      if (io_rc /= 0) then
        write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_write(v_out): ", cstr_message(errbuf)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
    end if
  end subroutine v_lowfive_write_export

  ! fuse_view_field(v_in_ref), NOT v_in itself -- the window Sink copies
  ! from there into v_in as part of the SAME mechanics it already uses
  ! for an ordinary channel (docs/user/lowfive.md's own Unit 3 account).
  subroutine v_lowfive_read_import()
    status = fuse_view_field(ctx, v_in_ref, view)
    call check(status, "view_field(v_in)")
    if (view%window_samples > 0 .and. c_associated(view%data)) then
      io_rc = bruss_lowfive_read(c_loc(kPeerFile), c_loc(kPeerDataset), &
          view%data, int(view%count, c_long_long) * int(view%components, c_long_long), &
          c_loc(errbuf), int(len(errbuf), c_long_long))
      if (io_rc /= 0) then
        write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_read(v_in): ", cstr_message(errbuf)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
    end if
  end subroutine v_lowfive_read_import

end program v_main
