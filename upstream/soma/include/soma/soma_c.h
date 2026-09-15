/* soma_c.h -- SOMA's stable C ABI.
 *
 * Spec 3.2: the serialized graph descriptor is the ABI boundary, and this
 * header is the function-level half of that same narrow waist. Everything the
 * Fortran shim (Task 8.2), the Python bindings (Task 8.3), and any external
 * adapter ever does crosses here.
 *
 * THIS INTERFACE IS FROZEN. Unlike the C++ API it cannot be changed without
 * breaking already-compiled callers. Additions must be new functions with new
 * names; no existing signature, error code, or documented behaviour may be
 * altered. SOMA_ABI_VERSION is bumped when a compatible addition is made.
 *
 * -------------------------------------------------------------------------
 * WHAT THIS ABI DOES NOT DO
 * -------------------------------------------------------------------------
 * This narrow waist provides NO automatic field exchange, remap, or
 * transport. soma_begin_step() / soma_end_step() only report the
 * [t_begin, t_end) window a component must integrate over and manage that
 * component's clock and, inside an iteration region, its sweep counter --
 * they never read or write a single byte of registered field storage. The
 * caller *is* the component: it is the caller's job to read whatever inputs
 * it needs from its own registered storage, perform any remap or exchange
 * itself (by whatever means it and its peers have agreed on outside this
 * ABI), and write its own outputs, in between the soma_begin_step() and
 * soma_end_step() calls that bracket one sweep.
 *
 * This is a design choice, not an oversight: SOMA's C++ Runner (the engine
 * behind spec 3.2's operator DAG) drives an entire coupled run by calling
 * *into* every component it advances, and this ABI has no mechanism for a C
 * or Fortran caller to register such a callback. soma_run_period() and
 * soma_run() are reserved for exactly that -- an in-process driver that would
 * remove this restriction -- and both always return SOMA_ERR_UNSUPPORTED in
 * every build today. Until one exists, do not assume this ABI silently
 * accepts a graph the Runner would reject, or that it performs any of the
 * remap/transport/accumulation machinery spec 3.2 describes: it deliberately
 * never attempts any of that, on any path, in this build.
 *
 * IN PARTICULAR: A SUCCESSFUL soma_initialize() IS NOT A CLAIM THAT SOMA COULD
 * PERFORM THE GRAPH'S REMAP. soma_initialize() validates, lowers and schedules
 * the graph, and (see its own documentation) refuses an iteration region whose
 * stopping rule is unusable -- but it never constructs a mesh backend and never
 * asks whether the remap weights a channel's `Conservative` flag calls for can
 * be built. A graph whose C++ Runner would stop at build_weights() with
 * E-REMAP-REGION, E-REMAP-BACKEND, E-REMAP-PARALLEL or E-REMAP-DIM still
 * initializes here, and correctly so: through this ABI *you* perform the
 * transfer, so SOMA's own ability to build a weight matrix for it is not the
 * question being asked. This repository's own test/data/pair.yaml is exactly
 * such a graph. If you need to know whether SOMA can build the weights, use the
 * C++ Runner, which answers that at its initialize().
 *
 * -------------------------------------------------------------------------
 * LANGUAGE
 * -------------------------------------------------------------------------
 * Pure C99. No C++ types, no references, no default arguments, no <stdbool.h>
 * dependency, no size_t: every integer that crosses the boundary is a
 * fixed-width type from <stdint.h>, so the same descriptor and the same call
 * work from an ILP32 and an LP64 caller. `int` is used only for the return
 * code, where it is the universal C convention and maps to Fortran's
 * `integer(c_int)`.
 *
 * -------------------------------------------------------------------------
 * ERRORS
 * -------------------------------------------------------------------------
 * Every entry point returns an `int`: SOMA_OK (0) on success, one of the
 * negative SOMA_ERR_* codes on failure. No C++ exception can escape any of
 * these functions -- each one wraps its body in a catch-all and converts to
 * SOMA_ERR_RUNTIME. An exception crossing `extern "C"` would be
 * std::terminate, and the C++ core throws by design in several places, so
 * this is enforced at every entry point rather than assumed.
 *
 * On failure the human-readable reason is stored in two places:
 *   - on the context, if the call had a valid one; retrieve with
 *     soma_last_error(ctx, ...);
 *   - in a thread-local slot belonging to the calling thread, always;
 *     retrieve with soma_last_error(NULL, ...).
 * The thread-local slot is the only way to see why soma_create() failed, and
 * the only way to see why a call with an invalid handle was rejected.
 *
 * OUT-PARAMETERS. Every out-parameter documented as "set to X on failure" is
 * cleared for failures detected *after* the context handle has been
 * validated. A call rejected because the handle itself was null, stale, or
 * never issued leaves its out-parameters untouched -- there is no context in
 * which to interpret them, and the caller has no result to read. Initialize
 * your out-parameters and check the return code; never infer success from an
 * out-parameter's value.
 *
 * ONE SUCCESSFUL CALL ALSO WRITES THE MESSAGE SLOT. soma_initialize() records
 * an advisory there when it accepts a multi-rank context (see its own
 * documentation). A non-empty message therefore does not mean the last call
 * failed; only the return code means that.
 *
 * MESSAGE LIFETIME. soma_last_error() copies into a caller-provided buffer
 * and NUL-terminates it. SOMA never hands out a pointer into its own storage
 * for an error message, so there is nothing to free and nothing that can
 * dangle. The stored message is overwritten by the next failing call on the
 * same context (or, for the thread-local slot, by the next failing call on
 * the same thread) and is not cleared by a subsequent successful call.
 *
 * -------------------------------------------------------------------------
 * HANDLES AND OWNERSHIP
 * -------------------------------------------------------------------------
 * `soma_ctx` is an opaque handle, not a pointer the caller may dereference,
 * do arithmetic on, or copy into persistent storage after destruction.
 *
 *   - soma_create() is the only producer. It allocates, and the caller owns
 *     the result.
 *   - soma_destroy() is the only consumer. It releases everything the context
 *     owns. SOMA never destroys a context the caller has not passed to it.
 *   - Every entry point validates its handle against an internal table of
 *     live contexts *before dereferencing it*. A handle that is not in that
 *     table is rejected with SOMA_ERR_INVALID.
 *   - Consequently: a double soma_destroy() returns SOMA_ERR_INVALID and does
 *     nothing; any call with a stale (already-destroyed) handle returns
 *     SOMA_ERR_INVALID and does nothing; a null handle returns
 *     SOMA_ERR_INVALID. None of these is undefined behaviour and none is a
 *     use-after-free.
 *   - Residual, unavoidable with pointer-shaped handles: if the allocator
 *     reuses the exact address of a destroyed context for a newly created
 *     one, a stale handle becomes indistinguishable from the new context.
 *     SOMA cannot detect that. It is still not a use-after-free -- the call
 *     lands on a live, fully constructed context.
 *
 * A field handle (soma_field_handle()) is a small non-negative integer that
 * indexes the graph loaded into *that* context. It is meaningless in any
 * other context. It stays valid for the life of the context: a context
 * accepts exactly one graph, so handles never go stale short of destruction.
 *
 * FIELD STORAGE is never owned by SOMA. soma_register_field() records a
 * pointer and a layout; the caller must keep that storage alive, at a fixed
 * address, until the context is destroyed. Reallocating the array behind a
 * registered pointer dangles it and SOMA cannot detect it.
 *
 * -------------------------------------------------------------------------
 * THREADING
 * -------------------------------------------------------------------------
 * A single soma_ctx must not be used concurrently from two threads; the
 * caller serializes. Distinct contexts may be used concurrently: the live
 * handle table is internally locked, and the error slot read by
 * soma_last_error(NULL, ...) is per-thread.
 *
 * -------------------------------------------------------------------------
 * LIFECYCLE
 * -------------------------------------------------------------------------
 *   soma_create
 *     -> EMPTY        soma_set_communicator, soma_set_end_time,
 *                     soma_load_graph / soma_load_graph_yaml
 *     -> LOADED       soma_field_handle, soma_save_graph,
 *                     soma_register_field, soma_initialize
 *     -> INITIALIZED  soma_begin_step / soma_end_step,
 *                     soma_requires_checkpoint / soma_requires_restore,
 *                     soma_is_coupling_ongoing
 *   soma_destroy      from any state
 *
 * Calling something outside its state returns SOMA_ERR_STATE. The transitions
 * are one-way: a context takes one graph and is initialized once.
 */

#ifndef SOMA_SOMA_C_H
#define SOMA_SOMA_C_H

#include <stdint.h>

#if defined(_WIN32) && defined(SOMA_C_SHARED)
#  ifdef SOMA_C_BUILD
#    define SOMA_C_API __declspec(dllexport)
#  else
#    define SOMA_C_API __declspec(dllimport)
#  endif
#else
#  define SOMA_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Error codes. Values are part of the ABI and never change.
 * ----------------------------------------------------------------------- */

/** Success. */
#define SOMA_OK               0
/** A malformed argument: null pointer, out-of-range value, bad handle,
 *  unparseable or self-inconsistent input. */
#define SOMA_ERR_INVALID     (-1)
/** A well-formed lookup that found nothing: no such component, no such
 *  field. */
#define SOMA_ERR_NOTFOUND    (-2)
/** The call is valid but not in this state: no graph loaded, already
 *  initialized, step already open, field already registered. */
#define SOMA_ERR_STATE       (-3)
/** An exception was caught at the boundary, or an internal operation failed.
 *  soma_last_error() carries the reason. */
#define SOMA_ERR_RUNTIME     (-4)
/** The operation exists in this ABI but this build cannot perform it. */
#define SOMA_ERR_UNSUPPORTED (-5)

/* --------------------------------------------------------------------------
 * Memory spaces, for soma_register_field().
 * ----------------------------------------------------------------------- */

/** Pointer is addressable by the host CPU. */
#define SOMA_MEMSPACE_HOST   0
/** Pointer lives in device memory; the transport layer stages it. */
#define SOMA_MEMSPACE_DEVICE 1

/** Version of this interface. Compare against soma_abi_version() at runtime
 *  to detect a header/library mismatch. */
#define SOMA_ABI_VERSION 1

/** Opaque coupling context. See HANDLES AND OWNERSHIP above. */
typedef struct soma_ctx_s* soma_ctx;

/* --------------------------------------------------------------------------
 * Version. Both are pure and always succeed.
 * ----------------------------------------------------------------------- */

/** Library version string, e.g. "0.1.0". Points to storage with static
 *  lifetime owned by SOMA; never free it, and it never dangles. Never
 *  returns NULL. */
SOMA_C_API const char* soma_version(void);

/** The SOMA_ABI_VERSION the library was compiled with. A caller compiled
 *  against a newer header than the library it loaded detects it here. */
SOMA_C_API int soma_abi_version(void);

/* --------------------------------------------------------------------------
 * Lifetime.
 * ----------------------------------------------------------------------- */

/** Create a context in the EMPTY state.
 *
 *  @param out  non-NULL; set to the new handle on success, to NULL on
 *              failure.
 *  @return SOMA_OK; SOMA_ERR_INVALID if @p out is NULL; SOMA_ERR_RUNTIME on
 *          allocation failure. Retrieve the message with
 *          soma_last_error(NULL, ...) -- there is no context to ask. */
SOMA_C_API int soma_create(soma_ctx* out);

/** Destroy a context and release everything it owns. Registered field
 *  storage is *not* freed; SOMA never owned it.
 *
 *  @return SOMA_OK; SOMA_ERR_INVALID if @p ctx is NULL, was never created by
 *          soma_create(), or was already destroyed. A double destroy is
 *          reported, not undefined. */
SOMA_C_API int soma_destroy(soma_ctx ctx);

/* --------------------------------------------------------------------------
 * Error retrieval.
 * ----------------------------------------------------------------------- */

/** Copy the last error message into @p buffer, truncated to fit and always
 *  NUL-terminated.
 *
 *  @param ctx          a live context, or NULL to read the calling thread's
 *                      thread-local message (the only way to see why
 *                      soma_create() or a call with a bad handle failed).
 *  @param buffer       non-NULL destination.
 *  @param buffer_size  capacity in bytes, including the NUL; must be >= 1.
 *  @return SOMA_OK, including when the message is empty or truncated;
 *          SOMA_ERR_INVALID if @p buffer is NULL, @p buffer_size is 0, or
 *          @p ctx is non-NULL but not a live handle.
 *
 *  Use soma_last_error_length() to size the buffer exactly. The message is
 *  copied, so its lifetime is entirely the caller's. */
SOMA_C_API int soma_last_error(soma_ctx ctx, char* buffer,
                               uint64_t buffer_size);

/** Length in bytes of the last error message, excluding the NUL.
 *
 *  @param ctx         a live context, or NULL for the thread-local message.
 *  @param out_length  non-NULL; set to 0 on failure.
 *  @return SOMA_OK; SOMA_ERR_INVALID if @p out_length is NULL or @p ctx is
 *          non-NULL but not live. */
SOMA_C_API int soma_last_error_length(soma_ctx ctx, uint64_t* out_length);

/* --------------------------------------------------------------------------
 * The narrow waist: getting a graph into the context.
 * ----------------------------------------------------------------------- */

/** Load the flat POD graph descriptor of spec 3.2 -- the narrow waist.
 *  C++, Fortran, Python, and YAML all produce this same byte string.
 *
 *  @param ctx         live context in the EMPTY state.
 *  @param descriptor  non-NULL; @p bytes readable bytes. Copied; SOMA does
 *                     not retain the pointer.
 *  @param bytes       length, > 0.
 *  @return SOMA_OK; SOMA_ERR_STATE if a graph is already loaded;
 *          SOMA_ERR_INVALID if @p descriptor is NULL, @p bytes is 0 or
 *          exceeds addressable memory, or the bytes are not a descriptor
 *          this build can read (wrong magic, wrong version, truncated,
 *          trailing bytes); SOMA_ERR_RUNTIME on an internal failure.
 *          Structural validation of the loaded graph happens in
 *          soma_initialize(), not here. */
SOMA_C_API int soma_load_graph(soma_ctx ctx, const void* descriptor,
                               uint64_t bytes);

/** Load a graph from a YAML file. Equivalent to reading the file and
 *  authoring the same graph, then soma_load_graph().
 *
 *  @param ctx   live context in the EMPTY state.
 *  @param path  non-NULL, NUL-terminated.
 *  @return SOMA_OK; SOMA_ERR_STATE if a graph is already loaded;
 *          SOMA_ERR_INVALID if @p path is NULL, the file cannot be opened, or
 *          the YAML is malformed or has unresolvable references;
 *          SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_load_graph_yaml(soma_ctx ctx, const char* path);

/** Serialize the loaded graph back to the descriptor of spec 3.2.
 *
 *  Two-call idiom: pass @p buffer == NULL to learn the size, then call again
 *  with a buffer at least that large. This is what lets a Fortran or Python
 *  front-end author in YAML and hand the *descriptor* to another process.
 *
 *  @param ctx          live context with a graph loaded (LOADED or
 *                      INITIALIZED).
 *  @param buffer       destination, or NULL to query the size only.
 *  @param buffer_size  capacity of @p buffer in bytes; ignored when
 *                      @p buffer is NULL.
 *  @param out_bytes    non-NULL; always set to the exact descriptor size on
 *                      SOMA_OK and on the too-small SOMA_ERR_INVALID, so a
 *                      failed guess still tells the caller what to allocate.
 *  @return SOMA_OK; SOMA_ERR_STATE if no graph is loaded; SOMA_ERR_INVALID if
 *          @p out_bytes is NULL or @p buffer is non-NULL and too small;
 *          SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_save_graph(soma_ctx ctx, void* buffer,
                               uint64_t buffer_size, uint64_t* out_bytes);

/* --------------------------------------------------------------------------
 * Fields.
 * ----------------------------------------------------------------------- */

/** Resolve (component name, field name) to the integer handle used by
 *  soma_register_field(). Resolve once at setup; nothing hot crosses this
 *  boundary (spec 3.2).
 *
 *  @param ctx         live context with a graph loaded.
 *  @param component   non-NULL component name, as spelled in the graph.
 *  @param field       non-NULL field name, as spelled in the graph.
 *  @param out_handle  non-NULL; set to a value >= 0 on success and to -1 on
 *                     every failure.
 *  @return SOMA_OK; SOMA_ERR_STATE if no graph is loaded; SOMA_ERR_INVALID if
 *          any pointer argument is NULL; SOMA_ERR_NOTFOUND if there is no
 *          such component, or no such field in that component;
 *          SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_field_handle(soma_ctx ctx, const char* component,
                                 const char* field, int32_t* out_handle);

/** Bind caller-owned storage to a field handle.
 *
 *  Strides are in units of `double`, not bytes, and describe both AoS and
 *  SoA without a separate flag:
 *      interleaved  x0 y0 z0 x1 y1 z1 : stride = 3, component_stride = 1
 *      blocked      x0 x1 y0 y1 z0 z1 : stride = 1, component_stride = count
 *  @p component_stride == 0 is shorthand for 1 (interleaved).
 *
 *  @param ctx               live context in the LOADED state -- after the
 *                           graph, before soma_initialize().
 *  @param handle            from soma_field_handle().
 *  @param data              non-NULL. NOT owned by SOMA: it must stay alive
 *                           and at a fixed address until soma_destroy().
 *  @param count             number of elements addressable through the view.
 *  @param stride            element-to-element step, in doubles.
 *  @param component_stride  component-to-component step, in doubles; 0 means
 *                           1.
 *  @param components        components per element; must be >= 1 and must
 *                           equal the count the graph declares for this
 *                           field. A mismatch is rejected rather than
 *                           reinterpreted -- it is silent wrong physics.
 *  @param memspace          SOMA_MEMSPACE_HOST or SOMA_MEMSPACE_DEVICE.
 *  @return SOMA_OK; SOMA_ERR_STATE if no graph is loaded, if the context is
 *          already initialized, or if this field is already registered
 *          (rebinding is refused, never a silent replacement);
 *          SOMA_ERR_INVALID for a null @p data, an out-of-range @p handle, a
 *          bad @p memspace, a @p components that is < 1 or disagrees with the
 *          graph, a count/stride that does not fit this platform's address
 *          space, or a layout whose element and component strides alias or
 *          overflow; SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_register_field(soma_ctx ctx, int32_t handle, double* data,
                                   uint64_t count, uint64_t stride,
                                   uint64_t component_stride,
                                   int32_t components, int32_t memspace);

/* --------------------------------------------------------------------------
 * Configuration.
 * ----------------------------------------------------------------------- */

/** Record the MPI communicator this context couples over, as a *Fortran*
 *  integer handle (what MPI_Comm_c2f() produces and what Fortran's
 *  MPI_COMM_WORLD is). An integer is used rather than MPI_Comm so that this
 *  header has no MPI dependency and can be included by a caller built
 *  without MPI.
 *
 *  RECORDED, NOT CONSUMED FOR DATA MOVEMENT. This call stores @p fortran_comm
 *  on the context. Nothing in this build moves a byte over it:
 *  soma_begin_step() and soma_end_step() never touch it (they do not move data
 *  at all -- see the note near the top of this header), and the only entry
 *  points that would, the reserved in-process driver soma_run_period() /
 *  soma_run(), answer SOMA_ERR_UNSUPPORTED in every build.
 *
 *  WHAT IT IS READ FOR. soma_initialize() asks it exactly one question: how
 *  many ranks will run this context? That decides whether SOMA may test an
 *  iteration region's convergence for you -- see soma_initialize(), which
 *  refuses that combination above one rank because the test is rank-local.
 *  Setting a communicator of size 1, MPI_COMM_SELF included, is always
 *  accepted and changes nothing.
 *
 *  SETTING NOTHING IS NOT A WAY AROUND THAT. With no communicator recorded,
 *  soma_initialize() asks MPI_COMM_WORLD instead. The hazard belongs to the
 *  process, not to this call, so declining to mention a communicator does not
 *  make a multi-rank run safe -- it only hides how many ranks there are, which
 *  is why SOMA does not let it.
 *
 *  @param ctx           live context, not yet initialized.
 *  @param fortran_comm  the Fortran handle.
 *  @return SOMA_OK; SOMA_ERR_STATE if the context is already initialized;
 *          SOMA_ERR_UNSUPPORTED if this build of SOMA has no MPI transport
 *          (configure with -DSOMA_ENABLE_MPI=ON). A build without MPI reports
 *          the communicator as unsupported rather than silently ignoring it
 *          and running single-rank. */
SOMA_C_API int soma_set_communicator(soma_ctx ctx, int32_t fortran_comm);

/** Set the time at which the run ends, as the exact rational
 *  @p numerator / @p denominator.
 *
 *  Rational, not double, because spec 4.3 requires exact coupling times: 1/3
 *  and 1/7 subcycling must not drift over 10^6 steps, and a double end time
 *  would make "is the run over?" depend on rounding.
 *
 *  With no end time set, soma_is_coupling_ongoing() always reports 1 and the
 *  caller decides when to stop.
 *
 *  @param ctx          live context, not yet initialized.
 *  @param numerator    any value.
 *  @param denominator  non-zero.
 *  @return SOMA_OK; SOMA_ERR_STATE if the context is already initialized;
 *          SOMA_ERR_INVALID if @p denominator is 0 or the fraction is not
 *          representable; SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_set_end_time(soma_ctx ctx, int64_t numerator,
                                 int64_t denominator);

/* --------------------------------------------------------------------------
 * Run.
 * ----------------------------------------------------------------------- */

/** Validate the graph, lower it, and build the schedule. One-way: EMPTY is
 *  not reachable again and a second call is SOMA_ERR_STATE.
 *
 *  Performed here, in this order: structural validation of the graph (all
 *  diagnostics are collected and reported together, not just the first); a
 *  check that every field a channel reads or writes has been registered;
 *  lowering to the operator DAG; construction of the per-component clocks and
 *  a check that their common period is representable; a check of every
 *  iteration region's stopping rule; and a check that this context's rank
 *  count is compatible with that stopping rule.
 *
 *  THE ITERATION-REGION CHECKS, and what each refuses. All are the same
 *  predicates SOMA's C++ Runner applies, and each exists because the state it
 *  refuses is a run that reports convergence it did not achieve:
 *    - a region with NO convergence measure. An empty criterion is satisfied by
 *      definition, so the region would stop after one sweep and call itself
 *      converged.
 *    - a region with max_iterations < 1, whose body would never run. It was
 *      previously clamped to one sweep, which is the same failure by a
 *      different route.
 *    - a convergence measure that names no channel of its region, or more than
 *      one; and a channel of the region that no measure names. The first is
 *      never tested against anything; the last lets the region stop while that
 *      channel is still far from its fixed point.
 *
 *  THE RANK-COUNT CHECK, and why you should read it even if you never call
 *  soma_set_communicator(). soma_end_step() decides an iteration region's
 *  convergence from THIS RANK'S registered storage and advances THIS RANK'S
 *  clock on that decision. It reduces nothing: this ABI has no collective
 *  anywhere. On more than one rank the clocks would therefore diverge as soon
 *  as two ranks' residuals differ -- a hang or a mistimed exchange, not a
 *  number you could inspect. So when this context will run on more than one
 *  rank AND some component's region asks SOMA to test convergence,
 *  soma_initialize() returns SOMA_ERR_UNSUPPORTED and names the components.
 *  Run on one rank, drop the convergence measure and sweep to max_iterations
 *  (which every rank does identically), or use the C++ Runner. The rank count
 *  comes from the communicator you set, if you set one, and from
 *  MPI_COMM_WORLD otherwise.
 *
 *  A run on more than one rank with no such region is fine and is accepted;
 *  it records an advisory retrievable with soma_last_error(). THAT IS THE ONE
 *  CASE IN WHICH A SUCCESSFUL CALL WRITES THE ERROR SLOT: as always, decide on
 *  the return code, never on the contents of the slot.
 *
 *  BEHAVIOUR CHANGE, recorded because this header is frozen: graphs whose
 *  iteration region is in any of the states above, and multi-rank contexts
 *  whose region asks SOMA to test convergence, used to reach the INITIALIZED
 *  state and now do not. No signature, error code or out-parameter meaning
 *  changed, and a graph that this ABI can actually honour is accepted exactly
 *  as before -- in particular a single-rank context is never refused on rank
 *  count, whatever communicator it recorded. A refused call leaves the context
 *  in LOADED, not half-initialized, so a caller may fix what it can fix and
 *  call again.
 *
 *  A SUCCESSFUL RETURN IS NOT A CLAIM ABOUT REMAP; see "WHAT THIS ABI DOES NOT
 *  DO" at the top of this header.
 *
 *  @param ctx  live context in the LOADED state.
 *  @return SOMA_OK; SOMA_ERR_STATE if no graph is loaded, if already
 *          initialized, or if a field a channel touches is unregistered (the
 *          message names them); SOMA_ERR_INVALID if the graph fails
 *          validation, cannot be lowered, or has an iteration region whose
 *          stopping rule is unusable (the message is the full diagnostic
 *          list); SOMA_ERR_UNSUPPORTED if this context will run on more than
 *          one rank and an iteration region asks SOMA to test convergence;
 *          SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_initialize(soma_ctx ctx);

/** Open a step on @p component and report the half-open interval
 *  [*t_begin, *t_end) it must integrate over.
 *
 *  Inside an iteration region the same interval is handed out repeatedly,
 *  once per sweep; soma_requires_restore() tells the caller when it is being
 *  re-swept. Outside one, each soma_end_step() advances to the next tick.
 *
 *  Times are reported as doubles for the caller's convenience; SOMA's own
 *  schedule is exact rational arithmetic and is never derived from these.
 *
 *  @param ctx        live, initialized context.
 *  @param component  non-NULL component name.
 *  @param t_begin    non-NULL; unmodified on failure.
 *  @param t_end      non-NULL; unmodified on failure.
 *  @return SOMA_OK; SOMA_ERR_STATE if the context is not initialized or a
 *          step is already open on this component; SOMA_ERR_INVALID if a
 *          pointer argument is NULL; SOMA_ERR_NOTFOUND if there is no such
 *          component; SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_begin_step(soma_ctx ctx, const char* component,
                               double* t_begin, double* t_end);

/** Close the step opened by soma_begin_step() on @p component.
 *
 *  If the component takes part in an iteration region this either advances
 *  the sweep counter (leaving the interval unchanged for the next
 *  soma_begin_step()) or, on the last sweep, resets it and advances the
 *  clock. Otherwise it advances the clock.
 *
 *  @param ctx        live, initialized context.
 *  @param component  non-NULL component name.
 *  @return SOMA_OK; SOMA_ERR_STATE if the context is not initialized or no
 *          step is open on this component; SOMA_ERR_INVALID if @p component
 *          is NULL; SOMA_ERR_NOTFOUND if there is no such component;
 *          SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_end_step(soma_ctx ctx, const char* component);

/** Must @p component checkpoint its state before the coming sweep?
 *
 *  The preCICE contract, exposed for components SOMA's own loop cannot drive.
 *  1 exactly when the component is in an iteration region and the next
 *  soma_begin_step() starts a fresh window; 0 otherwise, including for every
 *  component that is in no iteration region.
 *
 *  @param ctx           live, initialized context.
 *  @param component     non-NULL component name.
 *  @param out_required  non-NULL; set to 0 or 1, and to 0 on failure.
 *  @return SOMA_OK; SOMA_ERR_STATE if not initialized; SOMA_ERR_INVALID if a
 *          pointer argument is NULL; SOMA_ERR_NOTFOUND if there is no such
 *          component; SOMA_ERR_RUNTIME on an internal failure. */
SOMA_C_API int soma_requires_checkpoint(soma_ctx ctx, const char* component,
                                        int32_t* out_required);

/** Must @p component restore its last checkpoint before the coming sweep?
 *
 *  1 exactly when the component is in an iteration region and the next
 *  soma_begin_step() re-sweeps the same window; 0 otherwise. Same arguments
 *  and same return codes as soma_requires_checkpoint(). */
SOMA_C_API int soma_requires_restore(soma_ctx ctx, const char* component,
                                     int32_t* out_required);

/** Is there time left in the run?
 *
 *  @param ctx          live, initialized context.
 *  @param out_ongoing  non-NULL; 1 while the earliest component clock is
 *                      before the end time, 0 once it is not. Always 1 when
 *                      no end time was set (see soma_set_end_time()), and 0
 *                      for a graph with no components. Set to 0 on failure.
 *  @return SOMA_OK; SOMA_ERR_STATE if not initialized; SOMA_ERR_INVALID if
 *          @p out_ongoing is NULL; SOMA_ERR_RUNTIME on an internal
 *          failure. */
SOMA_C_API int soma_is_coupling_ongoing(soma_ctx ctx, int32_t* out_ongoing);

/** Run one common coupling period under SOMA's own driver.
 *
 *  Reserved. SOMA driving the loop requires it to call *into* each component,
 *  and this ABI defines no way to register such a callback -- a C or Fortran
 *  caller is itself the component and must drive with soma_begin_step() /
 *  soma_end_step(). The signature is fixed now so the in-process driver can
 *  be added later without an ABI break.
 *
 *  @return SOMA_ERR_STATE if the context is not initialized (checked first,
 *          so a mis-sequenced call is still diagnosed);
 *          SOMA_ERR_UNSUPPORTED otherwise. Never SOMA_OK in this build. */
SOMA_C_API int soma_run_period(soma_ctx ctx);

/** Run @p periods common coupling periods under SOMA's own driver.
 *
 *  Reserved; see soma_run_period(). `int64_t`, not `long`: `long` is 32 bits
 *  on LLP64 and 64 on LP64, which is not admissible in a stable ABI.
 *
 *  @return SOMA_ERR_STATE if not initialized; SOMA_ERR_INVALID if
 *          @p periods < 0; SOMA_ERR_UNSUPPORTED otherwise. */
SOMA_C_API int soma_run(soma_ctx ctx, int64_t periods);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SOMA_SOMA_C_H */
