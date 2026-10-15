! abi.fortran_sizes (build-evidence/abi-matrix.toml, ABI-FORTRAN-SIZES): prints
! "name C_SIZEOF" for every ffuse derived type, in the order of
! test/abi/struct_sizes.golden, so cmake/.../compare_sizes.cmake can diff this
! program's stdout against that committed table byte-for-byte on the numbers.
! Pins abi-design.md's "C_SIZEOF of the Fortran type must equal C sizeof".
!
! The golden's bookend lines are not one of ffuse's own argument structs:
! "sizeof(void*)" is C_SIZEOF of a C_PTR value (the pointer width every
! fuse_field_view.data / string field assumes), and
! "fuse_participant_abi_v1" is the C function-pointer TABLE fuse_coupler_abi()
! returns -- Fortran never dereferences that table (abi-design.md, "The
! costs": "Fortran cannot dereference a function-pointer struct
! ergonomically", which is exactly why ffuse exists as flat ffuse_* symbols
! instead), so this mirror type exists ONLY in this test, to pin the one
! other number struct_sizes.golden records: struct_size + abi_version
! (4 + 4 bytes) plus one C_FUNPTR per of the 17 entries (Unit 3, S6 phase
! 3 Conformance, 2026-10-04: view_field appended last; D14, 2026-10-13:
! base_communicator appended last after that).
program sizes_test
  use, intrinsic :: iso_c_binding
  use ffuse
  ! fuse_provider_abi_v1 itself is a GENERATED derived type (unlike
  ! fuse_participant_abi_v1_mirror below, hand-written only for this test):
  ! the provider table carries no function-pointer-dereferencing problem for
  ! Fortran the way the participant table does (abi-design.md, "The costs"),
  ! because a provider plugin ASSEMBLES this table itself via
  ! ffuse_provider_set_<entry>, never dereferences another library's copy of
  ! it, so ffuse_provider.f90 generates it directly.
  use ffuse_provider, only: fuse_provider_abi_v1
  implicit none

  type, bind(C) :: fuse_participant_abi_v1_mirror
    integer(c_int32_t) :: struct_size = 0_c_int32_t
    integer(c_int32_t) :: abi_version = 0_c_int32_t
    type(c_funptr) :: version = c_null_funptr
    type(c_funptr) :: capabilities = c_null_funptr
    type(c_funptr) :: create = c_null_funptr
    type(c_funptr) :: destroy = c_null_funptr
    type(c_funptr) :: last_error = c_null_funptr
    type(c_funptr) :: load_graph = c_null_funptr
    type(c_funptr) :: save_graph = c_null_funptr
    type(c_funptr) :: setup = c_null_funptr
    type(c_funptr) :: field_handle = c_null_funptr
    type(c_funptr) :: register_field = c_null_funptr
    type(c_funptr) :: initialize = c_null_funptr
    type(c_funptr) :: is_coupling_ongoing = c_null_funptr
    type(c_funptr) :: begin_step = c_null_funptr
    type(c_funptr) :: end_step = c_null_funptr
    type(c_funptr) :: run = c_null_funptr
    type(c_funptr) :: view_field = c_null_funptr
    type(c_funptr) :: base_communicator = c_null_funptr
    type(c_funptr) :: request_restart_point = c_null_funptr
  end type fuse_participant_abi_v1_mirror

  type(c_ptr) :: a_pointer
  type(fuse_null) :: v_null
  type(fuse_version_info) :: v_version_info
  type(fuse_capability_set) :: v_capabilities
  type(fuse_context_out) :: v_context_out
  type(fuse_error) :: v_error
  type(fuse_graph_source) :: v_graph_source
  type(fuse_graph_sink) :: v_graph_sink
  type(fuse_setup_options) :: v_setup
  type(fuse_field_name) :: v_field_name
  type(fuse_field_ref) :: v_field_ref
  type(fuse_field_view) :: v_field_view
  type(fuse_step_request) :: v_step_request
  type(fuse_step_report) :: v_step_report
  type(fuse_ongoing) :: v_ongoing
  type(fuse_run_request) :: v_run_request
  type(fuse_base_communicator_info) :: v_base_communicator_info
  type(fuse_participant_abi_v1_mirror) :: v_table
  type(fuse_field_spec) :: v_field_spec
  type(fuse_field_table) :: v_field_table
  type(fuse_provider_instance) :: v_provider_instance
  type(fuse_provider_abi_v1) :: v_provider_table
  type(fuse_restart_blob_sink) :: v_restart_blob_sink
  type(fuse_restart_blob_source) :: v_restart_blob_source

  a_pointer = c_null_ptr

  write(*, '(A," ",I0)') "sizeof(void*)", c_sizeof(a_pointer)
  write(*, '(A," ",I0)') "fuse_null", c_sizeof(v_null)
  write(*, '(A," ",I0)') "fuse_version_info", c_sizeof(v_version_info)
  write(*, '(A," ",I0)') "fuse_capability_set", c_sizeof(v_capabilities)
  write(*, '(A," ",I0)') "fuse_context_out", c_sizeof(v_context_out)
  write(*, '(A," ",I0)') "fuse_error", c_sizeof(v_error)
  write(*, '(A," ",I0)') "fuse_graph_source", c_sizeof(v_graph_source)
  write(*, '(A," ",I0)') "fuse_graph_sink", c_sizeof(v_graph_sink)
  write(*, '(A," ",I0)') "fuse_setup_options", c_sizeof(v_setup)
  write(*, '(A," ",I0)') "fuse_field_name", c_sizeof(v_field_name)
  write(*, '(A," ",I0)') "fuse_field_ref", c_sizeof(v_field_ref)
  write(*, '(A," ",I0)') "fuse_field_view", c_sizeof(v_field_view)
  write(*, '(A," ",I0)') "fuse_step_request", c_sizeof(v_step_request)
  write(*, '(A," ",I0)') "fuse_step_report", c_sizeof(v_step_report)
  write(*, '(A," ",I0)') "fuse_ongoing", c_sizeof(v_ongoing)
  write(*, '(A," ",I0)') "fuse_run_request", c_sizeof(v_run_request)
  write(*, '(A," ",I0)') "fuse_base_communicator_info", c_sizeof(v_base_communicator_info)
  write(*, '(A," ",I0)') "fuse_participant_abi_v1", c_sizeof(v_table)
  write(*, '(A," ",I0)') "fuse_field_spec", c_sizeof(v_field_spec)
  write(*, '(A," ",I0)') "fuse_field_table", c_sizeof(v_field_table)
  write(*, '(A," ",I0)') "fuse_provider_instance", c_sizeof(v_provider_instance)
  write(*, '(A," ",I0)') "fuse_provider_abi_v1", c_sizeof(v_provider_table)
  write(*, '(A," ",I0)') "fuse_restart_blob_sink", c_sizeof(v_restart_blob_sink)
  write(*, '(A," ",I0)') "fuse_restart_blob_source", c_sizeof(v_restart_blob_source)
end program sizes_test
