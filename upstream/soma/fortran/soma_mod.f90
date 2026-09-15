! soma_mod.f90 -- SOMA's ISO_C_BINDING shim over the stable C ABI (spec 6,
! integration mode 2). The C ABI of include/soma/soma_c.h is the narrow waist
! (spec 3.2); this module adds nothing to it and hides nothing from it. It
! exists so that a Fortran component never has to spell out a C string, a
! void pointer, or a memory layout by hand.
!
! ---------------------------------------------------------------------------
! INTEGER KINDS
! ---------------------------------------------------------------------------
! Every integer that crosses the boundary is a fixed-width C type, so every
! dummy argument here carries an explicit ISO_C_BINDING kind. This is not
! pedantry: a Fortran default `integer` is four bytes, the ABI wants eight for
! counts and times, and passing the wrong one is silent corruption with no
! diagnostic from either compiler. In particular
!
!     integer            :: h   ! WRONG under -fdefault-integer-8
!     integer(c_int32_t) :: h   ! right, always
!
! The ABI's `uint64_t` parameters (sizes and counts) are declared here as
! `integer(c_int64_t)`, because Fortran has no unsigned integer type. The bit
! patterns agree for every value in [0, 2**63); SOMA rejects anything larger
! anyway (`fits_size()` in src/capi/soma_c.cpp), and a negative Fortran value
! arrives on the C side as a huge unsigned and is likewise rejected. No count
! this library can accept is misrepresented by the substitution.
!
! ---------------------------------------------------------------------------
! ERROR HANDLING
! ---------------------------------------------------------------------------
! Every wrapper has a final `ierr` of kind `c_int` holding the ABI return
! code: soma_ok, or one of the negative soma_err_* parameters. Nothing here
! aborts, and nothing here is reported only through an out-argument -- always
! test ierr.
!
! Two classes of failure exist and they are distinguishable:
!
!   * a failure detected by SOMA. `soma_f_last_error` retrieves the message.
!   * a failure detected by this shim before the call was made -- an empty
!     array, a non-contiguous array, a size that does not fit the ABI's
!     32-bit component count. These return soma_err_invalid and do NOT set
!     the SOMA-side message, because no call reached SOMA; `soma_f_last_error`
!     would report a stale message from an earlier call. The shim keeps no
!     module state (it must stay usable from several threads driving several
!     contexts, which the C ABI explicitly permits), so there is nowhere else
!     to put such a message. Each such case is documented at its wrapper.
!
! `soma_f_last_error` accepts a context whose handle is null, which is how the
! C ABI exposes the calling thread's message. That is the only way to learn
! why `soma_f_create` failed, so pass the unset context straight back in.
!
! ---------------------------------------------------------------------------
! STRINGS
! ---------------------------------------------------------------------------
! A Fortran `character` is blank-padded and carries no terminator; C wants a
! NUL. Conversion runs in both directions here.
!
!   Fortran -> C: `trim()` is applied, then `c_null_char` appended. Trailing
!   blanks are therefore not part of a path or of a component/field name. A
!   SOMA name that ends in a blank cannot be spelled from Fortran; nothing in
!   the schema produces one.
!
!   C -> Fortran: the message is copied up to the terminator and the
!   destination is blank-padded, so `trim(message)` is the message. A message
!   longer than the destination is truncated -- ask `soma_f_last_error_length`
!   first if that matters.
!
! ---------------------------------------------------------------------------
! ARRAYS, LAYOUT, AND INDEX ORDER
! ---------------------------------------------------------------------------
! `soma_f_register_field` hands SOMA a raw address and a layout. Three
! Fortran-specific hazards are handled explicitly.
!
! 1. CONTIGUITY. The array dummies are plain assumed-shape (`array(:)`,
!    `array(:,:)`) and NOT declared `contiguous`. That is deliberate. A
!    `contiguous` dummy does not make a non-contiguous actual contiguous: it
!    makes the compiler copy it into a temporary, and SOMA would then record
!    the address of a temporary that ceases to exist when the call returns --
!    a dangling pointer with no diagnostic anywhere. A plain assumed-shape
!    dummy is passed as a descriptor with no copy, so `is_contiguous()`
!    reports the truth about the caller's array. It is checked, and a
!    non-contiguous actual is rejected with soma_err_invalid. Slice your array
!    contiguously or copy it yourself into storage you keep alive.
!
! 2. LIFETIME AND TARGET. The actual argument must have the TARGET (or
!    POINTER) attribute and must stay alive, at a fixed address, until
!    `soma_f_destroy`. SOMA never owns field storage. Fortran cannot check
!    either property from inside the callee -- if the actual is not a TARGET
!    the standard makes the retained address undefined on return -- so this is
!    a contract, in the same way the C ABI's is.
!
! 3. ORDER. SOMA's `Layout` counts in elements and components with strides
!    measured in doubles, and is neither row- nor column-major in itself; the
!    mapping is fixed here, once, for both supported ranks:
!
!      rank 1, `array(n)`, a one-component field:
!          count = n, stride = 1, component_stride = 0 (i.e. 1),
!          components = 1.
!          Fortran element i (1-based) is SOMA element i-1 (0-based).
!
!      rank 2, `array(ncomp, ncell)`, an ncomp-component field:
!          count = ncell, stride = ncomp, component_stride = 1,
!          components = ncomp.
!          Fortran `array(c, i)` is SOMA element i-1, component c-1.
!
!    The component index is the FIRST, fastest-varying dimension. Column-major
!    storage then puts the components of one element next to each other, which
!    is precisely SOMA's interleaved/AoS case, so no transposition and no copy
!    occurs. Declaring `array(ncell, ncomp)` instead and passing it here is
!    wrong and, for a non-square array, is rejected by SOMA because the
!    derived component count will not match the graph; for a square array it
!    would be accepted and silently transposed, which is why the shim fixes
!    the order rather than guessing it.
!
!    Handles and the SOMA-side element and component indices are 0-based, as
!    in C. Only Fortran's own subscripts are 1-based; the shim never exposes a
!    SOMA index to be added to or subtracted from a Fortran one.
!
! ---------------------------------------------------------------------------
! WHAT IS NOT HERE
! ---------------------------------------------------------------------------
! `soma_f_run` and `soma_f_run_period` are wrapped, and they return
! soma_err_unsupported by design in this build: SOMA driving the loop would
! mean calling back into the component, and this ABI defines no callback. A
! Fortran caller IS the component. Drive with soma_f_begin_step /
! soma_f_end_step, and consult soma_f_requires_checkpoint /
! soma_f_requires_restore inside an iteration region. The shim does not
! emulate the missing driver.

module soma_mod
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_associated,     &
                                         c_loc, c_f_pointer, c_char,          &
                                         c_null_char, c_int, c_int8_t,        &
                                         c_int32_t, c_int64_t, c_double
  implicit none
  private

  ! -- Public API ------------------------------------------------------------
  public :: soma_context
  public :: soma_ok, soma_err_invalid, soma_err_notfound, soma_err_state,     &
            soma_err_runtime, soma_err_unsupported
  public :: soma_memspace_host, soma_memspace_device
  public :: soma_abi_version_header
  public :: soma_f_version, soma_f_abi_version
  public :: soma_f_create, soma_f_destroy, soma_f_is_open
  public :: soma_f_last_error, soma_f_last_error_length
  public :: soma_f_load_yaml, soma_f_load_descriptor, soma_f_descriptor_size, &
            soma_f_save_descriptor
  public :: soma_f_field_handle, soma_f_register_field
  public :: soma_f_set_communicator, soma_f_set_end_time
  public :: soma_f_initialize
  public :: soma_f_begin_step, soma_f_end_step
  public :: soma_f_requires_checkpoint, soma_f_requires_restore
  public :: soma_f_is_coupling_ongoing
  public :: soma_f_run, soma_f_run_period

  ! -- ABI constants ---------------------------------------------------------
  ! Mirrors of the #defines in include/soma/soma_c.h. Their values are part of
  ! the ABI and never change.
  integer(c_int), parameter :: soma_ok              =  0
  integer(c_int), parameter :: soma_err_invalid     = -1
  integer(c_int), parameter :: soma_err_notfound    = -2
  integer(c_int), parameter :: soma_err_state       = -3
  integer(c_int), parameter :: soma_err_runtime     = -4
  integer(c_int), parameter :: soma_err_unsupported = -5

  integer(c_int32_t), parameter :: soma_memspace_host   = 0
  integer(c_int32_t), parameter :: soma_memspace_device = 1

  !> The SOMA_ABI_VERSION this module was written against. Compare with
  !> soma_f_abi_version(), which reports what the loaded library was built
  !> with; a mismatch means header and library disagree.
  integer(c_int), parameter :: soma_abi_version_header = 1

  !> An opaque coupling context. The wrapped `soma_ctx` is private on purpose:
  !> it is validated by SOMA against a table of live contexts and must not be
  !> fabricated, copied out of, or dereferenced. A default-initialised
  !> soma_context holds a null handle, which every SOMA entry point rejects
  !> with soma_err_invalid -- except soma_f_last_error, where null selects the
  !> calling thread's message.
  !>
  !> Do not copy a soma_context and destroy one of the copies: soma_f_destroy
  !> clears only the handle it was given, so the other copy is left pointing
  !> at a destroyed context. SOMA still rejects it (soma_err_invalid, from the
  !> live-handle table) rather than faulting, but nothing here can clear it
  !> for you -- Fortran has no move semantics and a finaliser that destroyed
  !> on every copy's demise would be far worse. Pass the context by dummy
  !> argument; do not assign it.
  !>
  !> A single context must not be driven from two threads at once. Distinct
  !> contexts may be, and this module keeps no state of its own, so the shim
  !> adds no constraint beyond the C ABI's.
  type :: soma_context
     private
     type(c_ptr) :: handle = c_null_ptr
  end type soma_context

  !> Bind caller-owned storage to a field handle. See the ARRAYS section of
  !> the module header for the layout each rank implies.
  interface soma_f_register_field
     module procedure soma_f_register_field_1d
     module procedure soma_f_register_field_2d
  end interface soma_f_register_field

  ! ==========================================================================
  ! Raw bind(C) interfaces.
  !
  ! Private: they take C pointers and fixed-width integers with no checking,
  ! and exposing them would defeat the point of the shim. Every one names its
  ! C symbol explicitly, so nothing depends on this compiler's name mangling
  ! or on whether it appends a trailing underscore.
  ! ==========================================================================
  interface

     function c_soma_version() bind(C, name="soma_version") result(p)
       import :: c_ptr
       type(c_ptr) :: p
     end function c_soma_version

     function c_soma_abi_version() bind(C, name="soma_abi_version") result(v)
       import :: c_int
       integer(c_int) :: v
     end function c_soma_abi_version

     function c_soma_create(out) bind(C, name="soma_create") result(rc)
       import :: c_int, c_ptr
       type(c_ptr), intent(out) :: out
       integer(c_int)           :: rc
     end function c_soma_create

     function c_soma_destroy(ctx) bind(C, name="soma_destroy") result(rc)
       import :: c_int, c_ptr
       type(c_ptr), value :: ctx
       integer(c_int)     :: rc
     end function c_soma_destroy

     function c_soma_last_error(ctx, buffer, buffer_size)                     &
          bind(C, name="soma_last_error") result(rc)
       import :: c_int, c_ptr, c_char, c_int64_t
       type(c_ptr), value            :: ctx
       character(kind=c_char), intent(inout) :: buffer(*)
       integer(c_int64_t), value     :: buffer_size
       integer(c_int)                :: rc
     end function c_soma_last_error

     function c_soma_last_error_length(ctx, out_length)                       &
          bind(C, name="soma_last_error_length") result(rc)
       import :: c_int, c_ptr, c_int64_t
       type(c_ptr), value :: ctx
       integer(c_int64_t), intent(inout) :: out_length
       integer(c_int)     :: rc
     end function c_soma_last_error_length

     function c_soma_load_graph(ctx, descriptor, bytes)                       &
          bind(C, name="soma_load_graph") result(rc)
       import :: c_int, c_ptr, c_int64_t
       type(c_ptr), value        :: ctx
       type(c_ptr), value        :: descriptor
       integer(c_int64_t), value :: bytes
       integer(c_int)            :: rc
     end function c_soma_load_graph

     function c_soma_load_graph_yaml(ctx, path)                               &
          bind(C, name="soma_load_graph_yaml") result(rc)
       import :: c_int, c_ptr, c_char
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in) :: path(*)
       integer(c_int)         :: rc
     end function c_soma_load_graph_yaml

     function c_soma_save_graph(ctx, buffer, buffer_size, out_bytes)          &
          bind(C, name="soma_save_graph") result(rc)
       import :: c_int, c_ptr, c_int64_t
       type(c_ptr), value        :: ctx
       type(c_ptr), value        :: buffer
       integer(c_int64_t), value :: buffer_size
       integer(c_int64_t), intent(inout) :: out_bytes
       integer(c_int)            :: rc
     end function c_soma_save_graph

     function c_soma_field_handle(ctx, component, field, out_handle)          &
          bind(C, name="soma_field_handle") result(rc)
       import :: c_int, c_ptr, c_char, c_int32_t
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in)    :: component(*)
       character(kind=c_char), intent(in)    :: field(*)
       integer(c_int32_t),     intent(inout) :: out_handle
       integer(c_int)         :: rc
     end function c_soma_field_handle

     function c_soma_register_field(ctx, handle, data, count, stride,         &
                                    component_stride, components, memspace)   &
          bind(C, name="soma_register_field") result(rc)
       import :: c_int, c_ptr, c_int32_t, c_int64_t
       type(c_ptr), value        :: ctx
       integer(c_int32_t), value :: handle
       type(c_ptr), value        :: data
       integer(c_int64_t), value :: count
       integer(c_int64_t), value :: stride
       integer(c_int64_t), value :: component_stride
       integer(c_int32_t), value :: components
       integer(c_int32_t), value :: memspace
       integer(c_int)            :: rc
     end function c_soma_register_field

     function c_soma_set_communicator(ctx, fortran_comm)                      &
          bind(C, name="soma_set_communicator") result(rc)
       import :: c_int, c_ptr, c_int32_t
       type(c_ptr), value        :: ctx
       integer(c_int32_t), value :: fortran_comm
       integer(c_int)            :: rc
     end function c_soma_set_communicator

     function c_soma_set_end_time(ctx, numerator, denominator)                &
          bind(C, name="soma_set_end_time") result(rc)
       import :: c_int, c_ptr, c_int64_t
       type(c_ptr), value        :: ctx
       integer(c_int64_t), value :: numerator
       integer(c_int64_t), value :: denominator
       integer(c_int)            :: rc
     end function c_soma_set_end_time

     function c_soma_initialize(ctx) bind(C, name="soma_initialize") result(rc)
       import :: c_int, c_ptr
       type(c_ptr), value :: ctx
       integer(c_int)     :: rc
     end function c_soma_initialize

     function c_soma_begin_step(ctx, component, t_begin, t_end)               &
          bind(C, name="soma_begin_step") result(rc)
       import :: c_int, c_ptr, c_char, c_double
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in)    :: component(*)
       real(c_double),         intent(inout) :: t_begin
       real(c_double),         intent(inout) :: t_end
       integer(c_int)         :: rc
     end function c_soma_begin_step

     function c_soma_end_step(ctx, component)                                 &
          bind(C, name="soma_end_step") result(rc)
       import :: c_int, c_ptr, c_char
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in) :: component(*)
       integer(c_int)         :: rc
     end function c_soma_end_step

     function c_soma_requires_checkpoint(ctx, component, out_required)        &
          bind(C, name="soma_requires_checkpoint") result(rc)
       import :: c_int, c_ptr, c_char, c_int32_t
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in)    :: component(*)
       integer(c_int32_t),     intent(inout) :: out_required
       integer(c_int)         :: rc
     end function c_soma_requires_checkpoint

     function c_soma_requires_restore(ctx, component, out_required)           &
          bind(C, name="soma_requires_restore") result(rc)
       import :: c_int, c_ptr, c_char, c_int32_t
       type(c_ptr), value     :: ctx
       character(kind=c_char), intent(in)    :: component(*)
       integer(c_int32_t),     intent(inout) :: out_required
       integer(c_int)         :: rc
     end function c_soma_requires_restore

     function c_soma_is_coupling_ongoing(ctx, out_ongoing)                    &
          bind(C, name="soma_is_coupling_ongoing") result(rc)
       import :: c_int, c_ptr, c_int32_t
       type(c_ptr), value :: ctx
       integer(c_int32_t), intent(inout) :: out_ongoing
       integer(c_int)     :: rc
     end function c_soma_is_coupling_ongoing

     function c_soma_run_period(ctx) bind(C, name="soma_run_period") result(rc)
       import :: c_int, c_ptr
       type(c_ptr), value :: ctx
       integer(c_int)     :: rc
     end function c_soma_run_period

     function c_soma_run(ctx, periods) bind(C, name="soma_run") result(rc)
       import :: c_int, c_ptr, c_int64_t
       type(c_ptr), value        :: ctx
       integer(c_int64_t), value :: periods
       integer(c_int)            :: rc
     end function c_soma_run

  end interface

contains

  ! ==========================================================================
  ! String conversion
  ! ==========================================================================

  !> Fortran character -> NUL-terminated c_char array. Trailing blanks are
  !> dropped (see the STRINGS section of the module header). The result is an
  !> automatic array, so it lives for the duration of the caller's statement,
  !> which is all any of these calls need: SOMA copies every string it is
  !> given.
  pure function f_to_c(s) result(a)
    character(len=*), intent(in) :: s
    character(kind=c_char)       :: a(len_trim(s) + 1)
    integer                      :: i, n
    n = len_trim(s)
    do i = 1, n
       a(i) = s(i:i)
    end do
    a(n + 1) = c_null_char
  end function f_to_c

  !> NUL-terminated c_char array -> Fortran character, blank-padded. Stops at
  !> the terminator, so nothing beyond it is read.
  pure subroutine c_to_f(a, s)
    character(kind=c_char), intent(in)  :: a(:)
    character(len=*),       intent(out) :: s
    integer :: i
    s = ''
    do i = 1, min(size(a), len(s))
       if (a(i) == c_null_char) return
       s(i:i) = a(i)
    end do
  end subroutine c_to_f

  ! ==========================================================================
  ! Version
  ! ==========================================================================

  !> SOMA's version string, e.g. "0.1.0", blank-padded into `str`.
  subroutine soma_f_version(str)
    character(len=*), intent(out) :: str
    type(c_ptr)                          :: p
    character(kind=c_char), pointer      :: buf(:)
    integer                              :: i
    str = ''
    p = c_soma_version()
    if (.not. c_associated(p)) return
    ! soma_version() never returns NULL and its storage is static, but the
    ! length is not known here, so map at most len(str)+1 characters and stop
    ! at the terminator. Nothing past the NUL is dereferenced.
    call c_f_pointer(p, buf, [len(str) + 1])
    do i = 1, len(str)
       if (buf(i) == c_null_char) return
       str(i:i) = buf(i)
    end do
  end subroutine soma_f_version

  !> The SOMA_ABI_VERSION the loaded library was compiled with. Compare with
  !> the soma_abi_version_header parameter.
  function soma_f_abi_version() result(v)
    integer(c_int) :: v
    v = c_soma_abi_version()
  end function soma_f_abi_version

  ! ==========================================================================
  ! Lifetime
  ! ==========================================================================

  !> Create a context in the EMPTY state. On failure the handle is left null
  !> and the reason is in the calling thread's message: pass the same, unset,
  !> `ctx` to soma_f_last_error to read it.
  subroutine soma_f_create(ctx, ierr)
    type(soma_context), intent(out) :: ctx
    integer(c_int),     intent(out) :: ierr
    ctx%handle = c_null_ptr
    ierr = c_soma_create(ctx%handle)
    if (ierr /= soma_ok) ctx%handle = c_null_ptr
  end subroutine soma_f_create

  !> Destroy a context. Registered field storage is not freed; SOMA never
  !> owned it. The handle is cleared whatever the outcome, so a second call
  !> cannot present a stale pointer to SOMA -- it presents a null one, which
  !> is rejected without any lookup. (SOMA also rejects a genuinely stale
  !> handle; this just removes the chance to offer one.)
  subroutine soma_f_destroy(ctx, ierr)
    type(soma_context), intent(inout) :: ctx
    integer(c_int),     intent(out)   :: ierr
    ierr = c_soma_destroy(ctx%handle)
    ctx%handle = c_null_ptr
  end subroutine soma_f_destroy

  !> True while `ctx` holds a handle this shim has not cleared. It does not
  !> prove the handle is live inside SOMA -- only SOMA's own table can, and it
  !> is consulted on every call.
  function soma_f_is_open(ctx) result(open)
    type(soma_context), intent(in) :: ctx
    logical                        :: open
    open = c_associated(ctx%handle)
  end function soma_f_is_open

  ! ==========================================================================
  ! Errors
  ! ==========================================================================

  !> Copy SOMA's last error message into `message`, blank-padded and possibly
  !> truncated. A `ctx` whose handle is null selects the calling thread's
  !> message, which is the only place the reason for a failed soma_f_create,
  !> or for a call rejected on its handle, is recorded.
  !>
  !> Failures this shim detects itself before calling SOMA do not update
  !> either message; see the ERROR HANDLING section of the module header.
  subroutine soma_f_last_error(ctx, message, ierr)
    type(soma_context), intent(in)            :: ctx
    character(len=*),   intent(out)           :: message
    integer(c_int),     intent(out), optional :: ierr
    character(kind=c_char) :: buf(len(message) + 1)
    integer(c_int)         :: rc
    message = ''
    buf(1) = c_null_char
    rc = c_soma_last_error(ctx%handle, buf,                                   &
                           int(len(message) + 1, c_int64_t))
    if (rc == soma_ok) call c_to_f(buf, message)
    if (present(ierr)) ierr = rc
  end subroutine soma_f_last_error

  !> Length in characters of the last error message, excluding the
  !> terminator. Size a buffer with this when truncation is unacceptable.
  subroutine soma_f_last_error_length(ctx, length, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int64_t), intent(out) :: length
    integer(c_int),     intent(out) :: ierr
    length = 0_c_int64_t
    ierr = c_soma_last_error_length(ctx%handle, length)
  end subroutine soma_f_last_error_length

  ! ==========================================================================
  ! The narrow waist
  ! ==========================================================================

  !> Load a graph from a YAML file. `path` is trimmed and NUL-terminated here.
  subroutine soma_f_load_yaml(ctx, path, ierr)
    type(soma_context), intent(in)  :: ctx
    character(len=*),   intent(in)  :: path
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_load_graph_yaml(ctx%handle, f_to_c(path))
  end subroutine soma_f_load_yaml

  !> Load the flat POD graph descriptor of spec 3.2 from a byte array. SOMA
  !> copies it, so `descriptor` need not outlive the call and need not be a
  !> TARGET beyond it.
  !>
  !> Shim-detected failures (soma_err_invalid, no SOMA-side message): an empty
  !> or non-contiguous `descriptor`.
  subroutine soma_f_load_descriptor(ctx, descriptor, ierr)
    type(soma_context), intent(in)         :: ctx
    integer(c_int8_t),  intent(in), target :: descriptor(:)
    integer(c_int),     intent(out)        :: ierr
    if (size(descriptor) <= 0 .or. .not. is_contiguous(descriptor)) then
       ierr = soma_err_invalid
       return
    end if
    ierr = c_soma_load_graph(ctx%handle, c_loc(descriptor(1)),                &
                             int(size(descriptor), c_int64_t))
  end subroutine soma_f_load_descriptor

  !> Size in bytes of the descriptor the loaded graph serialises to. This is
  !> the query half of the ABI's two-call idiom.
  subroutine soma_f_descriptor_size(ctx, nbytes, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int64_t), intent(out) :: nbytes
    integer(c_int),     intent(out) :: ierr
    nbytes = 0_c_int64_t
    ierr = c_soma_save_graph(ctx%handle, c_null_ptr, 0_c_int64_t, nbytes)
  end subroutine soma_f_descriptor_size

  !> Serialise the loaded graph into `buffer`. `nbytes` is set to the exact
  !> descriptor size both on success and when the buffer was too small, so a
  !> failed guess still says what to allocate.
  !>
  !> Shim-detected failures (soma_err_invalid, no SOMA-side message): an empty
  !> or non-contiguous `buffer`.
  subroutine soma_f_save_descriptor(ctx, buffer, nbytes, ierr)
    type(soma_context), intent(in)    :: ctx
    integer(c_int8_t),  intent(inout), target :: buffer(:)
    integer(c_int64_t), intent(out)   :: nbytes
    integer(c_int),     intent(out)   :: ierr
    nbytes = 0_c_int64_t
    if (size(buffer) <= 0 .or. .not. is_contiguous(buffer)) then
       ierr = soma_err_invalid
       return
    end if
    ierr = c_soma_save_graph(ctx%handle, c_loc(buffer(1)),                    &
                             int(size(buffer), c_int64_t), nbytes)
  end subroutine soma_f_save_descriptor

  ! ==========================================================================
  ! Fields
  ! ==========================================================================

  !> Resolve (component, field) to the 0-based integer handle used by
  !> soma_f_register_field. `handle` is -1 on every failure SOMA detects; test
  !> `ierr`, never the handle.
  subroutine soma_f_field_handle(ctx, component, field, handle, ierr)
    type(soma_context), intent(in)  :: ctx
    character(len=*),   intent(in)  :: component
    character(len=*),   intent(in)  :: field
    integer(c_int32_t), intent(out) :: handle
    integer(c_int),     intent(out) :: ierr
    handle = -1_c_int32_t
    ierr = c_soma_field_handle(ctx%handle, f_to_c(component), f_to_c(field),  &
                               handle)
  end subroutine soma_f_field_handle

  !> Bind a rank-1 array to a one-component field:
  !>     count = size(array), stride = 1, component_stride = 0, components = 1.
  !>
  !> `array` must have the TARGET attribute at the call site and must stay
  !> alive at a fixed address until soma_f_destroy; see the ARRAYS section of
  !> the module header.
  !>
  !> Shim-detected failures (soma_err_invalid, no SOMA-side message): an empty
  !> or non-contiguous `array`.
  subroutine soma_f_register_field_1d(ctx, handle, array, ierr, memspace)
    type(soma_context), intent(in)              :: ctx
    integer(c_int32_t), intent(in)              :: handle
    real(c_double),     intent(inout), target   :: array(:)
    integer(c_int),     intent(out)             :: ierr
    integer(c_int32_t), intent(in),    optional :: memspace
    integer(c_int32_t) :: space
    space = soma_memspace_host
    if (present(memspace)) space = memspace
    if (size(array) <= 0 .or. .not. is_contiguous(array)) then
       ierr = soma_err_invalid
       return
    end if
    ierr = c_soma_register_field(ctx%handle, handle, c_loc(array(1)),         &
                                 int(size(array), c_int64_t),                 &
                                 1_c_int64_t, 0_c_int64_t, 1_c_int32_t, space)
  end subroutine soma_f_register_field_1d

  !> Bind a rank-2 array `array(ncomp, ncell)` to an ncomp-component field:
  !>     count = ncell, stride = ncomp, component_stride = 1,
  !>     components = ncomp.
  !>
  !> The COMPONENT index is the first, fastest-varying dimension. This is the
  !> interleaved/AoS layout and needs no transposition, because column-major
  !> storage already places one element's components together. Passing
  !> `array(ncell, ncomp)` is the classic transposition bug: SOMA is then told
  !> the field has `ncell` components and rejects the registration unless the
  !> array happens to be square.
  !>
  !> Same TARGET and lifetime contract as the rank-1 form.
  !>
  !> Shim-detected failures (soma_err_invalid, no SOMA-side message): an empty
  !> or non-contiguous `array`, or a first extent exceeding the ABI's 32-bit
  !> component count.
  subroutine soma_f_register_field_2d(ctx, handle, array, ierr, memspace)
    type(soma_context), intent(in)              :: ctx
    integer(c_int32_t), intent(in)              :: handle
    real(c_double),     intent(inout), target   :: array(:,:)
    integer(c_int),     intent(out)             :: ierr
    integer(c_int32_t), intent(in),    optional :: memspace
    integer(c_int32_t) :: space
    integer            :: ncomp, ncell
    space = soma_memspace_host
    if (present(memspace)) space = memspace
    ncomp = size(array, 1)
    ncell = size(array, 2)
    if (ncomp <= 0 .or. ncell <= 0 .or. .not. is_contiguous(array)) then
       ierr = soma_err_invalid
       return
    end if
    if (int(ncomp, c_int64_t) > int(huge(0_c_int32_t), c_int64_t)) then
       ierr = soma_err_invalid
       return
    end if
    ierr = c_soma_register_field(ctx%handle, handle, c_loc(array(1, 1)),      &
                                 int(ncell, c_int64_t),                       &
                                 int(ncomp, c_int64_t),                       &
                                 1_c_int64_t, int(ncomp, c_int32_t), space)
  end subroutine soma_f_register_field_2d

  ! ==========================================================================
  ! Configuration
  ! ==========================================================================

  !> Record the MPI communicator, as a *Fortran* MPI handle -- pass
  !> MPI_COMM_WORLD directly, not MPI_Comm_c2f of anything. Returns
  !> soma_err_unsupported on a build without MPI, rather than silently running
  !> single-rank.
  !>
  !> RECORDED, NOT USED TO MOVE DATA. Nothing in this build sends a byte over
  !> the handle: this interface moves no data on any path, and
  !> soma_f_run/soma_f_run_period, the only entry points that would, are
  !> unsupported in every build. soma_f_initialize reads it for exactly one
  !> purpose -- how many ranks will run this context -- because it refuses to
  !> test an iteration region's convergence for you above one rank; that test
  !> is rank-local and the clocks would diverge. A size-1 communicator,
  !> MPI_COMM_SELF included, is always accepted.
  !>
  !> Not calling this is NOT a way around that: with nothing recorded,
  !> soma_f_initialize asks MPI_COMM_WORLD instead.
  subroutine soma_f_set_communicator(ctx, fortran_comm, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int32_t), intent(in)  :: fortran_comm
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_set_communicator(ctx%handle, fortran_comm)
  end subroutine soma_f_set_communicator

  !> Set the end of the run to the exact rational numerator/denominator.
  !> Rational, not a real: spec 4.3 needs 1/3 and 1/7 subcycling not to drift,
  !> and a floating-point end time would make "is the run over?" a rounding
  !> question. There is deliberately no real-valued overload.
  subroutine soma_f_set_end_time(ctx, numerator, denominator, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int64_t), intent(in)  :: numerator
    integer(c_int64_t), intent(in)  :: denominator
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_set_end_time(ctx%handle, numerator, denominator)
  end subroutine soma_f_set_end_time

  ! ==========================================================================
  ! Run
  ! ==========================================================================

  !> Validate the graph, lower it, and build the schedule. One-way.
  !>
  !> Also refuses, with soma_err_invalid, an iteration region whose stopping
  !> rule is unusable -- no convergence measure at all, max_iterations below 1,
  !> a measure that names no channel of its region (or more than one), or a
  !> channel of the region that no measure names. Each of those would report a
  !> convergence the run did not achieve.
  !>
  !> And with soma_err_unsupported when this context will run on more than one
  !> rank while an iteration region asks SOMA to test convergence:
  !> soma_f_end_step decides that from this rank's own storage and reduces
  !> nothing, so the component clocks would diverge. See the note on
  !> soma_f_set_communicator. A refused call leaves the context loaded, not
  !> half-initialised.
  subroutine soma_f_initialize(ctx, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_initialize(ctx%handle)
  end subroutine soma_f_initialize

  !> Open a step on `component` and report the half-open interval
  !> [t_begin, t_end) it must integrate. Both are untouched on failure, so
  !> initialise them if you intend to look at them regardless.
  subroutine soma_f_begin_step(ctx, component, t_begin, t_end, ierr)
    type(soma_context), intent(in)    :: ctx
    character(len=*),   intent(in)    :: component
    real(c_double),     intent(inout) :: t_begin
    real(c_double),     intent(inout) :: t_end
    integer(c_int),     intent(out)   :: ierr
    ierr = c_soma_begin_step(ctx%handle, f_to_c(component), t_begin, t_end)
  end subroutine soma_f_begin_step

  !> Close the step opened on `component`.
  subroutine soma_f_end_step(ctx, component, ierr)
    type(soma_context), intent(in)  :: ctx
    character(len=*),   intent(in)  :: component
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_end_step(ctx%handle, f_to_c(component))
  end subroutine soma_f_end_step

  !> Must `component` checkpoint before the coming sweep? `required` is
  !> .false. on failure and for every component in no iteration region.
  subroutine soma_f_requires_checkpoint(ctx, component, required, ierr)
    type(soma_context), intent(in)  :: ctx
    character(len=*),   intent(in)  :: component
    logical,            intent(out) :: required
    integer(c_int),     intent(out) :: ierr
    integer(c_int32_t) :: flag
    flag = 0_c_int32_t
    ierr = c_soma_requires_checkpoint(ctx%handle, f_to_c(component), flag)
    required = (flag /= 0_c_int32_t)
  end subroutine soma_f_requires_checkpoint

  !> Must `component` restore its last checkpoint before the coming sweep?
  subroutine soma_f_requires_restore(ctx, component, required, ierr)
    type(soma_context), intent(in)  :: ctx
    character(len=*),   intent(in)  :: component
    logical,            intent(out) :: required
    integer(c_int),     intent(out) :: ierr
    integer(c_int32_t) :: flag
    flag = 0_c_int32_t
    ierr = c_soma_requires_restore(ctx%handle, f_to_c(component), flag)
    required = (flag /= 0_c_int32_t)
  end subroutine soma_f_requires_restore

  !> Is there time left in the run? Always .true. when no end time was set.
  !> `ongoing` is .false. on failure, so test `ierr` first.
  subroutine soma_f_is_coupling_ongoing(ctx, ongoing, ierr)
    type(soma_context), intent(in)  :: ctx
    logical,            intent(out) :: ongoing
    integer(c_int),     intent(out) :: ierr
    integer(c_int32_t) :: flag
    flag = 0_c_int32_t
    ierr = c_soma_is_coupling_ongoing(ctx%handle, flag)
    ongoing = (flag /= 0_c_int32_t)
  end subroutine soma_f_is_coupling_ongoing

  !> Reserved. Returns soma_err_state when the context is not initialised and
  !> soma_err_unsupported otherwise -- never soma_ok in this build. A Fortran
  !> caller is the component and drives the loop itself with soma_f_begin_step
  !> and soma_f_end_step. The shim does not emulate the absent driver.
  subroutine soma_f_run_period(ctx, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_run_period(ctx%handle)
  end subroutine soma_f_run_period

  !> Reserved; see soma_f_run_period. Also soma_err_invalid for periods < 0.
  subroutine soma_f_run(ctx, periods, ierr)
    type(soma_context), intent(in)  :: ctx
    integer(c_int64_t), intent(in)  :: periods
    integer(c_int),     intent(out) :: ierr
    ierr = c_soma_run(ctx%handle, periods)
  end subroutine soma_f_run

end module soma_mod
