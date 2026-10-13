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
! real V-physics rank. Even so, --ranks-u/--ranks-v (below) are still
! read: V's OWN physics-only communicator now comes from a REAL
! MPI_Comm_split (color 1, matching u_main.c's own color 0 and
! coupler_main.c's own color 2 -- see that split's own comment below),
! so this program could derive v_rank/v_size from it alone; the flags
! are kept anyway and cross-checked against the split's own
! MPI_Comm_size, so a mismatch between what the launcher SAID V's shape
! was and what the real communicator SAYS it is aborts loudly rather
! than running on whichever one happened to be wrong.
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

  ! brusselator_grid.h's grid and diffusion, through the shim (bruss_grid_dims):
  ! one source of truth with the graph tool and U. Set below, before use.
  integer(c_int) :: nx_u, ny_u, nx_v, ny_v
  real(c_double) :: kDiffusion

  character(len=512) :: graph_path, log_path, arg
  character(len=16) :: transport
  ! Physics track: --ranks-u is the OFFSET into this program's own
  ! MPI_COMM_WORLD at which V's own block starts (0 in accept mode,
  ! where this executable's local world contains only V's own ranks;
  ! the real ranks_u in mpmd mode, where MPI_COMM_WORLD is the SHARED
  ! world every ':'-segment of one mpiexec numbers into, U's own block
  ! first -- u_main.c's own header comment has the full account).
  ! --ranks-v is V's own component size, -1 (the default) meaning
  ! "world_size - ranks_u", correct whenever V's own block is the LAST
  ! one in this program's own world (accept mode always; mpmd mode only
  ! when there is no bundled-coupler segment AFTER V, which is this
  ! track's own layout -- brusselator_graph_tool.cpp's component order
  ! is U, V, coupler, so an mpmd run genuinely needs ranks_v passed
  ! explicitly whenever ranks_coupler > 0, which test/parallel/
  ! brusselator_participants/run_participants.sh now does). Halo track:
  ! v_size (handed to bruss_v_create below) now comes from the REAL
  ! `v_comm` (the color-1 split, below), not purely from these flags --
  ! `ranks_u_offset`/`ranks_v_declared` are kept only as a CROSS-CHECK
  ! against it (a disagreement aborts loudly) and for the diagnostic
  ! message if `bruss_v_create` itself refuses. --solver mirrors
  ! u_main.c's own flag (species_core.hpp's own SolverKind; "auto"
  ! resolves to direct at v_size==1, schur otherwise).
  integer :: ranks_u_offset, ranks_v_declared, v_rank, v_size
  character(len=16) :: solver
  integer :: periods, ierr, world_rank, world_size, nargs, i, log_unit, p
  integer :: v_comm
  type(fuse_ctx) :: ctx
  type(fuse_null) :: nothing
  type(fuse_context_out) :: made
  type(fuse_setup_options) :: setup
  type(fuse_field_ref) :: v_out_ref, v_in_ref, u_out_peer_ref
  type(fuse_field_view) :: view
  integer(c_int32_t) :: status
  type(c_ptr) :: vh
  real(c_double), target, allocatable :: v_out(:), v_in(:)
  character(len=2), target :: my_component_name
  character(len=17), target :: solver_cstr
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
  ! ROUND 5 (diagnostic trace track): the two `which` labels bruss_
  ! lowfive_trace_skip() prints, one per call site below -- same
  ! c_loc()-needs-a-target reasoning as kOwnFile/kOwnDataset above, not
  ! because the TEXT varies, only because c_loc() needs an addressable
  ! variable, not a literal.
  character(len=25), target :: kTraceWhichWriteExport
  character(len=24), target :: kTraceWhichReadImport

  ! S6 phase 3 (Conformance, "LowFive attach" track): the file-access
  ! property list bruss_lowfive_attach() hands back, set once below
  ! (right after MPI_Init()/the physics-communicator split) and read by
  ! v_lowfive_write_export()/v_lowfive_read_import() on every call
  ! thereafter -- shim.h's own updated contract (-1 means "not attached
  ! yet", refused by bruss_lowfive_write()/bruss_lowfive_read()
  ! themselves). graph_path_cstr is graph_path NUL-terminated for the
  ! bind(C) call, the same c_loc()-of-a-target-buffer convention
  ! kOwnFile/kPeerFile/solver_cstr already use just above.
  integer(c_long_long) :: fapl_id
  character(len=513), target :: graph_path_cstr
  ! D14 Stage B (KNOWN_ISSUES.md): the bridge base_communicator() (ffuse,
  ! generated) reports for this context, and the local communicator
  ! bruss_lowfive_join_local_split()'s own fresh split of it produces --
  ! see this program's own post-setup block below for the full account.
  type(fuse_base_communicator_info) :: bci
  integer(c_long_long) :: bridge_comm_f, local_comm_f
  logical :: is_mpmd_launch

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

    function bruss_lowfive_write(file, dataset, data, n, fapl_id, errbuf, errbuf_len) &
        result(rc) bind(C, name="bruss_lowfive_write")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: file, dataset, data, errbuf
      integer(c_long_long), value :: n, fapl_id, errbuf_len
      integer(c_int) :: rc
    end function bruss_lowfive_write

    function bruss_lowfive_read(file, dataset, data, n, fapl_id, errbuf, errbuf_len) &
        result(rc) bind(C, name="bruss_lowfive_read")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: file, dataset, data, errbuf
      integer(c_long_long), value :: n, fapl_id, errbuf_len
      integer(c_int) :: rc
    end function bruss_lowfive_read

    ! D14 Stage B (KNOWN_ISSUES.md): the collective EVERY rank of the
    ! bridge must join -- lowfive_attach.h's own header comment on this
    ! function has the full account. `local_comm_f_out` is `intent(out)`
    ! on the C side, spelled the same by-reference way fapl_id_out
    ! already is below.
    function bruss_lowfive_join_local_split(graph_path, bridge_comm_f, local_comm_f_out, &
        errbuf, errbuf_len) result(rc) bind(C, name="bruss_lowfive_join_local_split")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: graph_path, errbuf
      integer(c_long_long), value :: bridge_comm_f, errbuf_len
      integer(c_long_long) :: local_comm_f_out
      integer(c_int) :: rc
    end function bruss_lowfive_join_local_split

    ! S6 phase 3 (Conformance, "LowFive attach" track): programmatic VOL
    ! attach, once per process -- lowfive_attach.h's own header comment
    ! has the full account. `fapl_id_out` is `intent(out)` on the C side
    ! (a `long long*`), which Fortran's own bind(C) spells as a by-
    ! reference INTEGER argument (no `value`), identical to how ffuse's
    ! own generated interfaces already pass an ABI out-parameter. D14
    ! Stage B: `bridge_comm_f`/`local_comm_f` replace the single
    ! `physics_comm_f` this interface used to take -- never called by a
    ! rank with nothing to attach now (the companion fix); see this
    ! program's own call site below for why.
    function bruss_lowfive_attach(graph_path, bridge_comm_f, local_comm_f, fapl_id_out, &
        errbuf, errbuf_len) result(rc) bind(C, name="bruss_lowfive_attach")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: graph_path, errbuf
      integer(c_long_long), value :: bridge_comm_f, local_comm_f, errbuf_len
      integer(c_long_long) :: fapl_id_out
      integer(c_int) :: rc
    end function bruss_lowfive_attach

    ! Round 2 ("role-ordered serve"), new in this track: called
    ! unconditionally at the very end of this program (lowfive_attach.h's
    ! own header comment has the full account of what this catches --
    ! harmless no-op when LowFive was never attached in this process).
    function bruss_lowfive_finalize(errbuf, errbuf_len) &
        result(rc) bind(C, name="bruss_lowfive_finalize")
      import :: c_ptr, c_long_long, c_int
      type(c_ptr), value :: errbuf
      integer(c_long_long), value :: errbuf_len
      integer(c_int) :: rc
    end function bruss_lowfive_finalize

    ! ROUND 5 (diagnostic trace track), item 3 of the brief: a no-op
    ! unless FUSE_LOWFIVE_ATTACH_TRACE=1 -- lowfive_attach.h's own doc
    ! comment has the full account. A SUBROUTINE (void return), unlike
    ! every other interface above, because this is purely a diagnostic
    ! side effect with nothing to report back.
    subroutine bruss_lowfive_trace_skip(which, window_samples, count, components) &
        bind(C, name="bruss_lowfive_trace_skip")
      import :: c_ptr, c_long_long
      type(c_ptr), value :: which
      integer(c_long_long), value :: window_samples, count, components
    end subroutine bruss_lowfive_trace_skip
  end interface

  graph_path = ""
  log_path = ""
  periods = 2
  transport = "fuse"
  ranks_u_offset = 0
  ranks_v_declared = -1
  solver = "auto"
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
    else if (trim(arg) == "--ranks-u" .and. i < nargs) then
      call get_command_argument(i + 1, arg)
      read(arg, *) ranks_u_offset
      i = i + 2
    else if (trim(arg) == "--ranks-v" .and. i < nargs) then
      call get_command_argument(i + 1, arg)
      read(arg, *) ranks_v_declared
      i = i + 2
    else if (trim(arg) == "--solver" .and. i < nargs) then
      call get_command_argument(i + 1, arg)
      solver = trim(arg)
      i = i + 2
    else
      write(0, '(A,A)') "v_main: unrecognized argument: ", trim(arg)
      call exit(1)
    end if
  end do
  if (len_trim(graph_path) == 0 .or. len_trim(log_path) == 0) then
    write(0, '(A)') "usage: v_main --graph graph.yaml --log v.log [--periods N] " // &
        "[--transport fuse|lowfive] [--ranks-u N] [--ranks-v N] " // &
        "[--solver auto|direct|schur|pcg|gmres]"
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
  kTraceWhichWriteExport = "v_lowfive_write_export" // c_null_char
  kTraceWhichReadImport = "v_lowfive_read_import" // c_null_char

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, world_size, ierr)

  ! Physics track: COLLECTIVE over MPI_COMM_WORLD, matching u_main.c's
  ! own MPI_Comm_split and coupler_main.c's own -- in mpmd mode
  ! MPI_COMM_WORLD is the SHARED world every ':'-segment's ranks (U, V,
  ! the coupler) number into, so each program's own split is a
  ! collective over ALL of them, not just its own ranks; a rank that
  ! never called a matching split would strand the other two programs'
  ! own ranks inside theirs (this track's own standing lesson: a
  ! collective must never be skipped).
  !
  ! COLOR = COMPONENT INDEX (U 0, V 1, coupler 2), key = world rank --
  ! replacing the earlier "V calls MPI_UNDEFINED to opt out of U's own
  ! split" form with ONE uniform call shape every program makes, each
  ! naming its own fixed color (u_main.c's and coupler_main.c's own
  ! comments on their own identical-shaped calls have the full account,
  ! including the choreography mismatch this exact collective caused on
  ! the host once already when the three programs entered it in
  ! different orders). UNLIKE the old MPI_UNDEFINED form, this result IS
  ! used now: `v_comm` is V's own real physics-only communicator,
  ! handed to `bruss_v_create` below as `comm_f` so a decomposed V's own
  ! `NativeMeshBackend::set_communicator` has something real to adopt.
  ! In accept mode this program's own MPI_COMM_WORLD contains only V's
  ! own ranks (Q1's ruling, this file's own header comment), so `v_comm`
  ! there is isomorphic to MPI_COMM_WORLD itself -- a harmless, correct
  ! no-op split, not a special case.
  call MPI_Comm_split(MPI_COMM_WORLD, 1, world_rank, v_comm, ierr)
  call MPI_Comm_rank(v_comm, v_rank, ierr)
  call MPI_Comm_size(v_comm, v_size, ierr)

  ! D14 Stage B: fapl_id's own "not attached yet" default -- the real
  ! attach call moves to AFTER fuse_setup() succeeds, below (decision 5).
  fapl_id = -1_c_long_long

  ! Cross-check against --ranks-u/--ranks-v (above): the two mechanisms
  ! must agree (both describe the SAME physical fact, V's own component
  ! size, computed two different ways -- real-communicator size here,
  ! launcher-supplied arithmetic there). A mismatch means the launcher
  ! (run_participants.sh) passed a flag that does not match the actual
  ! mpiexec layout it ran, which is a launch-script bug worth aborting
  ! on loudly rather than silently trusting one source over the other.
  if (ranks_v_declared < 0) ranks_v_declared = world_size - ranks_u_offset
  if (ranks_v_declared /= v_size) then
    write(0, '(A,I0,A,I0,A)') &
        "FAIL(v_main): --ranks-v (or its world_size-derived default) says ", ranks_v_declared, &
        " but the real color-1 communicator says ", v_size, &
        " -- the launcher's own flags disagree with the actual mpiexec layout"
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

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

  ! D14 Stage B (KNOWN_ISSUES.md), decision 5: the attach call moves
  ! HERE, after fuse_setup() succeeds, uniformly for mpmd and accept (not
  ! mode-conditional) -- base_communicator() (ffuse, generated) has
  ! nothing to report before setup()'s own collective has run. Cold-start
  ! priming (this program's own first write-then-read, after
  ! bruss_v_create() below) is UNAFFECTED: it already runs after this
  ! point in program order, so moving the attach call earlier in this
  ! same post-setup region changes nothing about when priming happens,
  ! only when the VOL registration it depends on does -- strictly
  ! before, same as before.
  !
  ! is_mpmd_launch, decision 5's own ruling: mpmd keeps MPI_COMM_WORLD
  ! literally for the bridge (connect_mpmd() merely dups it -- congruent,
  ! same rank for same process, by construction, but not the SAME
  ! communicator object, and this track found no way to show the two are
  ! interchangeable for MPI_Intercomm_create's own peer_comm argument
  ! without a host measurement this sandbox cannot make; the mpmd row
  ! already PASSES on the host with MPI_COMM_WORLD, so nothing here risks
  ! that regression for zero measured benefit). ranks_u_offset is 0 in
  ! accept mode (this program's own header comment on it, above) and > 0
  ! in mpmd mode (the real U rank count, always >= 1) -- the ONE existing
  ! CLI signal this program already carries for which launch mode this
  ! is, unchanged by argument parsing between there and here.
  is_mpmd_launch = (ranks_u_offset > 0)
  if (is_mpmd_launch) then
    bridge_comm_f = int(MPI_COMM_WORLD, c_long_long)
  else
    ! struct_size is set by the generated fuse_base_communicator()
    ! module procedure itself, not by this caller (abi-design.md's own
    ! "struct_size is set by the module, not the user" -- the SAME
    ! convention fuse_setup()/fuse_create() above already rely on
    ! silently; grep this file for "struct_size" to confirm it is never
    ! set by hand anywhere else here either).
    status = fuse_base_communicator(ctx, nothing, bci)
    call check(status, "base_communicator")
    if (bci%formed == 0) then
      write(0, '(A)') "FAIL(v_main): base_communicator(): formed=0 after a " // &
          "successful setup() with a nonzero communicator -- connect() " // &
          "itself should have refused setup() already if it could not form one"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    bridge_comm_f = bci%communicator
  end if

  ! D14 Stage B, decision 3 (the companion fix's own mirror image): V's
  ! own executable has no bundled placeholder rank at all (Q1's ruling,
  ! this file's own header comment) -- every rank of it is real
  ! V-physics -- but it still calls bruss_lowfive_join_local_split()
  ! unconditionally, the same collective every rank of the bridge must
  ! join (lowfive_attach.h's own header comment on that function has the
  ! full account); safe and cheap even for an ordinary, non-LowFive run.
  graph_path_cstr = trim(graph_path) // c_null_char
  io_rc = bruss_lowfive_join_local_split(c_loc(graph_path_cstr), bridge_comm_f, &
      local_comm_f, c_loc(errbuf), int(len(errbuf), c_long_long))
  if (io_rc /= 0) then
    write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_join_local_split: ", cstr_message(errbuf)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  if (use_lowfive) then
    if (local_comm_f == 0) then
      write(0, '(A)') "FAIL(v_main): --transport lowfive but this rank has no " // &
          "qualifying HDF5-bound channel (bruss_lowfive_join_local_split " // &
          "returned local_comm_f=0)"
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
    io_rc = bruss_lowfive_attach(c_loc(graph_path_cstr), bridge_comm_f, local_comm_f, &
        fapl_id, c_loc(errbuf), int(len(errbuf), c_long_long))
    if (io_rc /= 0) then
      write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_attach: ", cstr_message(errbuf)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end if
  end if

  call bruss_grid_dims(nx_u, ny_u, nx_v, ny_v)
  kDiffusion = bruss_diffusion()
  ! Both of V's window-transfer fields live on U's own grid
  ! (species_component.hpp's file header explains why; brusselator_graph.hpp
  ! mirrors it) -- and on THIS rank's band of it. The Runner sizes every
  ! field of a placement by the placement's own decomposition (D24), so a
  ! V on two ranks owns half of U's 16x16 grid per rank, 128 cells, exactly
  ! as u_main.c sizes its own storage with bruss_owned_size. The first
  ! 5-rank host run (2026-10-10) registered the whole grid here and was
  ! refused: "registered 256 values for its export, but the plan sizes that
  ! field at 128 on this rank" (E-RT-SIZE, the window member's size check).
  ! On one rank this is nx_u*ny_u, as before.
  nu = bruss_owned_size(nx_u, ny_u, v_rank, v_size)
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

  solver_cstr = trim(solver) // c_null_char
  ! comm_f = v_comm itself: Fortran's own MPI binding never needs
  ! MPI_Comm_c2f() (this file's own header comment has the full
  ! account) -- v_comm IS already the Fortran-integer handle
  ! fuse::native::CommHandle expects.
  vh = bruss_v_create(nx_u, ny_u, nx_v, ny_v, kDiffusion, v_rank, v_size, &
                      int(v_comm, c_int64_t), c_loc(solver_cstr))
  if (.not. c_associated(vh)) then
    write(0, '(A,I0,A,I0,A)') "FAIL(v_main): bruss_v_create returned NULL (v_rank=", v_rank, &
        ", v_size=", v_size, " -- see bruss_v_last_error on a handle this call never returned; " // &
        "check the solver string and, at v_size>1, whether this build has real MPI)"
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if
  ! P_VU(V0) into the storage registered for v_out: the Runner primes the
  ! first window's t=0 sample from it at fuse_initialize (shim.h, and
  ! u_main.c's own matching block). Left at zero, U's first window started
  ! from a V that did not exist.
  call bruss_v_provided(vh, c_loc(v_out))

  status = fuse_initialize(ctx, nothing, nothing)
  call check(status, "initialize")

  ! One writer per --log path (rank 0); every rank still computes the
  ! mean below, since bruss_v_state_mean is a collective over V's ranks.
  if (v_rank == 0) then
    open(newunit=log_unit, file=trim(log_path), status="replace", action="write")
  end if

  ! The WHOLE species' mean (one reduction over V's ranks), not this rank's
  ! slab: bit-identical to the old sum/size on one rank.
  mean_v = bruss_v_state_mean(vh)
  if (v_rank == 0) write(log_unit, '(A,ES25.17E3,A,ES25.17E3)') &
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

    mean_v = bruss_v_state_mean(vh)  ! collective, see window 0
    if (v_rank == 0) write(log_unit, '(A,I0,A,ES25.17E3,A,ES25.17E3)') &
        "window=", p + 1, " mean_v=", mean_v, " exchange_in=", bruss_v_exchange_in(vh)
  end do

  if (v_rank == 0) close(log_unit)
  call bruss_v_destroy(vh)
  deallocate(v_out, v_in)

  status = fuse_destroy(ctx, nothing, nothing)
  call check(status, "destroy")

  ! Round 2 ("role-ordered serve"), new in this track: called
  ! unconditionally (harmless no-op when !use_lowfive or the attach was
  ! never reached) after this process's own last v_lowfive_write_export/
  ! v_lowfive_read_import call and before tearing down MPI --
  ! lowfive_attach.h's own header comment has the full account of what
  ! this catches (u_main.c's own mirror-image call has the same comment).
  io_rc = bruss_lowfive_finalize(c_loc(errbuf), int(len(errbuf), c_long_long))
  if (io_rc /= 0) then
    write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_finalize: ", cstr_message(errbuf)
    call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  end if

  call MPI_Comm_free(v_comm, ierr)
  call MPI_Finalize(ierr)

contains

  subroutine check(status_in, what)
    integer(c_int32_t), intent(in) :: status_in
    character(len=*), intent(in) :: what
    integer :: abort_ierr
    character(len=2048) :: detail
    integer(c_int32_t) :: detail_status
    if (status_in /= FUSE_OK) then
      ! The status alone names a class (-4 is FUSE_ERR_RUNTIME, "an exception
      ! was contained at the boundary"), not the cause. The first 5-rank host
      ! run (2026-10-10) printed exactly that and nothing else from V, so
      ! the library's own message is read back the way u_main.c's fail()
      ! does with t->last_error. The context may be unset when `what` is
      ! "create"; then there is no message to read and the status stands.
      write(0, '(A,A,A,I0)') "FAIL(v_main): ", what, ": status=", status_in
      if (c_associated(ctx%ptr)) then
        detail_status = fuse_last_error(ctx, detail)
        if (detail_status == FUSE_OK .and. len_trim(detail) > 0) then
          write(0, '(A,A)') "  ", trim(detail)
        end if
      end if
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
          fapl_id, c_loc(errbuf), int(len(errbuf), c_long_long))
      if (io_rc /= 0) then
        write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_write(v_out): ", cstr_message(errbuf)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
    else
      ! ROUND 5 (diagnostic trace track), item 3: u_main.c's own
      ! u_lowfive_write_export() mirrors this with the identical guard
      ! and the identical call, settling H1 for V's own write side too.
      call bruss_lowfive_trace_skip(c_loc(kTraceWhichWriteExport), &
          int(view%window_samples, c_long_long), int(view%count, c_long_long), &
          int(view%components, c_long_long))
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
          fapl_id, c_loc(errbuf), int(len(errbuf), c_long_long))
      if (io_rc /= 0) then
        write(0, '(A,A)') "FAIL(v_main): bruss_lowfive_read(v_in): ", cstr_message(errbuf)
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
    else
      ! ROUND 5 (diagnostic trace track), item 3: the READ side's own
      ! counterpart, same reasoning as v_lowfive_write_export()'s own
      ! else branch above.
      call bruss_lowfive_trace_skip(c_loc(kTraceWhichReadImport), &
          int(view%window_samples, c_long_long), int(view%count, c_long_long), &
          int(view%components, c_long_long))
    end if
  end subroutine v_lowfive_read_import

end program v_main
