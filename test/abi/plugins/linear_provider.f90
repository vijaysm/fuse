! A provider plugin for test/abi/conformance_test.cpp, written in Fortran
! through the generated ffuse_provider module (fortran/ffuse_provider.f90):
! the SAME physics as test/abi/plugins/linear_provider.c and the native C++
! "upstream" component, so the conformance harness's EXPECT_EQ against the
! native run is a real cross-language check (docs/dev/abi-design.md,
! "Provider direction").
module linear_provider_impl
  use, intrinsic :: iso_c_binding
  use ffuse, only: fuse_null, fuse_version_info, fuse_capability_set, &
                   fuse_field_view, fuse_field_table, fuse_field_spec, &
                   fuse_provider_instance, fuse_step_request, fuse_step_report, &
                   fuse_semantic_version, FUSE_OK, FUSE_ERR_INVALID, FUSE_ERR_RUNTIME, &
                   FUSE_ABI_VERSION, FUSE_ELEMENT_F64
  use ffuse_provider, only: fuse_provider_abi_v1, ffuse_provider_bootstrap, &
                            ffuse_provider_set_version, ffuse_provider_set_create, &
                            ffuse_provider_set_destroy, ffuse_provider_set_describe, &
                            ffuse_provider_set_bind, ffuse_provider_set_advance, &
                            ffuse_provider_set_save_state, ffuse_provider_set_restore_state, &
                            ffuse_provider_set_capabilities
  implicit none
  private
  public :: linear_provider_bootstrap

  type :: instance_t
    type(c_ptr) :: out_ptr = c_null_ptr
  end type instance_t

  ! SAVE, TARGET: ffuse_provider_bootstrap() takes c_loc() of this, so its
  ! address must stay fixed for the plugin's whole lifetime
  ! (abi-design.md's own "field storage ... must stay valid" rule, applied
  ! here to the table itself).
  type(fuse_provider_abi_v1), save, target :: g_table
  logical, save :: g_table_ready = .false.

contains

  function p_version(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_version_info), intent(inout) :: out
    integer(c_int32_t) :: status
    character(len=48), target, save :: msg = "fuse test linear_provider (F)" // c_null_char

    out%library = fuse_semantic_version(1_c_int32_t, 0_c_int32_t, 0_c_int32_t)
    out%abi_version = FUSE_ABI_VERSION
    out%reserved = 0_c_int32_t
    out%string = c_loc(msg)
    status = FUSE_OK
  end function p_version

  function p_create(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_provider_instance), intent(inout) :: out
    integer(c_int32_t) :: status
    type(instance_t), pointer :: inst

    allocate(inst)
    out%self = c_loc(inst)
    status = FUSE_OK
  end function p_create

  function p_destroy(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_null), intent(inout) :: out
    integer(c_int32_t) :: status
    type(instance_t), pointer :: inst

    if (c_associated(self)) then
      call c_f_pointer(self, inst)
      deallocate(inst)
    end if
    status = FUSE_OK
  end function p_destroy

  function p_describe(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_field_table), intent(inout) :: out
    integer(c_int32_t) :: status
    type(fuse_field_spec), pointer :: rows(:)
    character(len=16), target, save :: field_name = "out" // c_null_char

    out%written = 1_c_int64_t
    if (c_associated(out%buffer) .and. out%capacity >= 1) then
      call c_f_pointer(out%buffer, rows, [1])
      rows(1)%name = c_loc(field_name)
      rows(1)%components = 1_c_int32_t
      rows(1)%intent = 0_c_int32_t
      rows(1)%element_type = FUSE_ELEMENT_F64
    end if
    status = FUSE_OK
  end function p_describe

  function p_bind(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_field_view), intent(in) :: in
    type(fuse_null), intent(inout) :: out
    integer(c_int32_t) :: status
    type(instance_t), pointer :: inst

    if (.not. c_associated(self) .or. .not. c_associated(in%data)) then
      status = FUSE_ERR_INVALID
      return
    end if
    call c_f_pointer(self, inst)
    inst%out_ptr = in%data
    status = FUSE_OK
  end function p_bind

  function p_advance(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_step_request), intent(in) :: in
    type(fuse_step_report), intent(inout) :: out
    integer(c_int32_t) :: status
    type(instance_t), pointer :: inst
    real(c_double), pointer :: out_val

    if (.not. c_associated(self)) then
      status = FUSE_ERR_INVALID
      return
    end if
    call c_f_pointer(self, inst)
    if (c_associated(inst%out_ptr)) then
      call c_f_pointer(inst%out_ptr, out_val)
      out_val = real(in%t_end%numerator, c_double) / real(in%t_end%denominator, c_double)
    end if
    out%requires_checkpoint = 0_c_int32_t
    out%requires_restore = 0_c_int32_t
    out%step = in%step
    out%iteration = in%iteration
    out%reserved = 0_c_int32_t
    out%t_begin = in%t_begin
    out%t_end = in%t_end
    status = FUSE_OK
  end function p_advance

  function p_save_state(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_null), intent(inout) :: out
    integer(c_int32_t) :: status
    status = FUSE_OK
  end function p_save_state

  function p_restore_state(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_null), intent(inout) :: out
    integer(c_int32_t) :: status
    status = FUSE_OK
  end function p_restore_state

  function p_capabilities(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_null), intent(in) :: in
    type(fuse_capability_set), intent(inout) :: out
    integer(c_int32_t) :: status
    out%flags = 0_c_int64_t
    status = FUSE_OK
  end function p_capabilities

  ! Called from the plugin's own top-level fuse_provider_abi() (below): the
  ! table is built once, lazily, then every later call reuses it.
  function linear_provider_bootstrap(abi_version) result(ptr)
    integer(c_int32_t), value :: abi_version
    type(c_ptr) :: ptr

    if (.not. g_table_ready) then
      g_table%struct_size = int(c_sizeof(g_table), c_int32_t)
      g_table%abi_version = FUSE_ABI_VERSION
      call ffuse_provider_set_version(g_table, p_version)
      call ffuse_provider_set_create(g_table, p_create)
      call ffuse_provider_set_destroy(g_table, p_destroy)
      call ffuse_provider_set_describe(g_table, p_describe)
      call ffuse_provider_set_bind(g_table, p_bind)
      call ffuse_provider_set_advance(g_table, p_advance)
      call ffuse_provider_set_save_state(g_table, p_save_state)
      call ffuse_provider_set_restore_state(g_table, p_restore_state)
      call ffuse_provider_set_capabilities(g_table, p_capabilities)
      g_table_ready = .true.
    end if
    ptr = ffuse_provider_bootstrap(g_table, abi_version)
  end function linear_provider_bootstrap

end module linear_provider_impl

! THE ONE SYMBOL THIS PLUGIN EXPORTS (docs/dev/abi-design.md, point 4): a
! one-line wrapper around linear_provider_bootstrap(), exactly the shape
! fortran/ffuse_provider.f90's own header comment describes.
function fuse_provider_abi(abi_version) result(ptr) bind(C, name="fuse_provider_abi")
  use, intrinsic :: iso_c_binding, only: c_ptr, c_int32_t
  use linear_provider_impl, only: linear_provider_bootstrap
  implicit none
  integer(c_int32_t), value :: abi_version
  type(c_ptr) :: ptr

  ptr = linear_provider_bootstrap(abi_version)
end function fuse_provider_abi
