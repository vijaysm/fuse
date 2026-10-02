# Why the C ABI is one symbol

The usage side of this is in [](../user/c-abi.md). This page is the reasoning, kept because the
decision looks arbitrary until you have hit the alternative.

## The constraint

FUSE needs a C ABI for two opposite reasons: a **participant ABI** so a foreign code with its own
time loop can drive the coupler, and a **provider ABI** so the coupler can drive a plugin. C symbols
are unmangled and global, and the natural names collide. libfuse (Filesystem in Userspace) is
installed on essentially every Linux and HPC system and owns `fuse_*`, including `fuse_semantic_version`,
`fuse_new`, `fuse_loop`, `struct fuse_args`, `struct fuse_buf` and `struct fuse_chan` — the last
three being names a coupler would plausibly want. Any process linking both is exposed, and symbol
interposition makes the failure mode a *wrong call*, not a link error.

## Two answers we rejected

**A prefix** (`fusec_begin_step`). Correct and dull. It renames roughly fifty symbols to fifty other
symbols: the collision surface is unchanged in size, only in spelling, and every entry point carries
a wart forever.

**A macro** (`CC(fuse_)` expanding to `c_fuse_*`). Rejected on four counts. `CC` is the C compiler
variable in every Makefile and a macro in several libraries, so defining it in a public header is
hostile. Users cannot grep and IDEs cannot complete. `nm`, `gdb`, `perf` and every stack trace show a
name that appears nowhere in the source. And it still produces fifty exported symbols — it solves the
spelling, not the surface.

## What we do instead

Export **one** symbol per ABI. Everything else is a member of a versioned struct of function
pointers.

```c
const fuse_participant_abi_v1* fuse_coupler_abi(uint32_t abi_version);
```

Six consequences, in rough order of how much they matter:

1. **The collision surface is one name, and it is unmistakable.** Built with
   `-fvisibility=hidden`, the library exports that one C entry point and nothing else: the whole ABI
   is auditable with a single `nm` invocation, and there is no accidental ABI to freeze by mistake.
2. **Members get short, prefix-free names**, because the struct is the namespace.
   `fuse->begin_step(...)` reads better than either rejected spelling.
3. **Versioning is the mechanism, not a convention.** A `struct_size` header plus append-only growth
   is exactly the negotiation the upstream provider ABI already implements and mutation-tested, so
   both ABIs share one mechanism instead of having two.
4. **`dlopen` becomes one `dlsym`**, which the dynamic-provider path needs anyway.
5. **No macros in public headers.** The single-source-of-truth trick lives where macros belong: an
   *internal* entry list generates the table initialiser and the `struct_size` assertion, and a golden
   test asserts that the public header's member list matches that list exactly — the same golden-file
   discipline the IR uses. The two cannot drift, and the header a user reads is plain documented C.
6. **Testability.** A test substitutes its own table to inject a fault at one entry point, with no
   linker tricks and no rebuild.

## The costs

Stated here rather than discovered later:

- One extra line to fetch the table before first use.
- **No direct linking against individual entry points.** A build system that wants
  `-Wl,--undefined=fuse_begin_step` cannot have it. A thin optional shim library forwarding real
  symbols to the table exists for codes that need it — opt-in, and never the default.
- Function pointers do not inline across the boundary. Irrelevant here: every entry point is a setup
  or step-boundary call, never per-degree-of-freedom, and `I5` restricts dispatch only in hot kernels.
- **Fortran cannot dereference a function-pointer struct ergonomically.** It is possible
  (`type(C_FUNPTR)` plus `C_F_PROCPOINTER`) but unpleasant, and Fortran is the one audience for which
  flat symbols are genuinely better. So the Fortran shim keeps a small set of exported `bind(C)`
  symbols, generated from the same entry list. A deliberate exception with a stated reason, not a
  second ABI.

C++ callers never see any of this; `fuse::Graph` and `fuse::run` are the primary surface.

## How arguments cross the boundary: structs, and the rules for growing them

Decided 2026-10-02, with the user's requirement that "any new args do not break the API in the
future". Every entry point has one shape,

```c
fuse_status name(fuse_ctx* ctx, const IN* in, OUT* out);
```

and `IN`, `OUT` are argument structs from `include/fuse/abi/types.h` whose first member is
`uint32_t struct_size`. The caller sets it to `sizeof` of the struct it was compiled against. The
rules, each of which a test enforces:

1. **Members are appended, never inserted, reordered, retyped or removed.** The order in
   `include/fuse/abi/entries.def` is the ABI. (Golden: the header is regenerated from the `.def`
   and must be byte-identical; `tools/gen-abi.py --check`.)
2. **The callee reads only the members `struct_size` covers**, and takes the documented default
   for the rest. A `struct_size` smaller than the first frozen layout, or larger than
   `FUSE_ABI_MAX_STRUCT_SIZE`, is `FUSE_ERR_VERSION`. (Mutation rows: a reader that trusts a short
   struct; a writer that sets the wrong size.)
3. **An appended member must increase `sizeof`.** This is the rule the first size table exposed:
   `fuse_version_info` has 28 bytes of members in 32 bytes of storage, so a 4-byte member appended
   there would change the layout *without* changing `struct_size`, and rule 2 would read the new
   member from an old caller's padding. So a member that would fit in trailing padding is instead
   declared together with an explicit `reserved` member that pushes `sizeof` up, or widened to
   64 bits. (Golden: `test/abi/struct_sizes.golden`, one line per struct, with
   `sizeof(void*) == 8` asserted; a change to the table is a deliberate, reviewed commit.)
4. **Value types never grow.** `fuse_exact_time` and `fuse_semantic_version` carry no `struct_size`; they are
   closed, and a new kind of time would be a new type.
5. **Nothing is allocated across the boundary.** Buffers (error messages, graph bytes) are the
   caller's, sized by a length query; strings are NUL-terminated and valid for the call only;
   field storage (`fuse_field_view.data`) is the caller's and must stay valid until `destroy`.
6. **Time is exact.** A step's bounds cross as `fuse_exact_time` (numerator, denominator), never as
   a double; a caller that needs a double computes it, and the agreement on where a step begins
   and ends is not lost in the rounding.
7. **Collective entries are marked in the `.def`** (`setup`, `initialize`, `begin_step`,
   `end_step`) and must be called by every process of the stated communicator in the same order
   relative to its other collectives — the same contract `MoabMeshBackend::set_communicator`
   states.

## Fortran: `ffuse`

Fortran gets flat `bind(C)` procedures rather than the table (the exception stated under "The
costs"). The whole module is generated by `tools/gen-abi.py` from the same `entries.def`, so it
cannot disagree with the C header. The mapping rules:

| In `entries.def` | In Fortran | Note |
|---|---|---|
| argument struct `fuse_x` | `type, bind(C) :: fuse_x`, members in the same order | `C_SIZEOF` of the Fortran type must equal C `sizeof`; tested for every struct |
| `uint32_t`, `int32_t` | `integer(c_int32_t)` | unsigned fields hold values below 2^31 by construction (`FUSE_ABI_MAX_STRUCT_SIZE`); documented |
| `int64_t`, `uint64_t` | `integer(c_int64_t)` | |
| any pointer | `type(c_ptr)` | `c_loc()` of a `target` array; the caller keeps it alive as the C rule says |
| `fuse_exact_time`, `fuse_semantic_version` | `type, bind(C)` of the same name | closed value types |
| `fuse_ctx*` | `type(fuse_ctx)` wrapping one `type(c_ptr)` | opaque |
| entry `name` | module procedure `fuse_name(ctx, in, out)`; link symbol `ffuse_name` | see below |
| a type whose name equals an entry's `fuse_name` | refused by the generator | Fortran has one namespace for types and procedures; the type is renamed in the `.def` (hence `fuse_capability_set`, `fuse_setup_options`, `fuse_semantic_version`) |
| `FUSE_OK`, `FUSE_ERR_*`, `FUSE_ELEMENT_*`, ... | `integer(c_int32_t), parameter` of the same name | |

- **Link names are `ffuse_*`, never `fuse_*`.** The Fortran-visible names *are* `fuse_*`, because a
  module procedure name is not a link symbol; the `bind(C, name="ffuse_...")` symbol that the
  Fortran interface references is exported from a separate library, `libfuse-abi-fortran`, so
  `libfuse-abi` keeps exactly one exported symbol and the `ffuse_*` family cannot collide with
  libfuse's `fuse_*`.
- **`struct_size` is set by the module, not the user.** Each generated `fuse_name` procedure sets
  `in%struct_size = int(c_sizeof(in), c_int32_t)` (and the same for `out`) before the call. A
  Fortran user never sees the mechanism, and a Fortran program compiled against an older module
  keeps running against a newer library for the same reason a C one does.
- **Defaults are default initialisation.** Every argument struct's Fortran type initialises each
  member to the documented default (`0`, `c_null_ptr`, `fuse_exact_time(0, 1)`), so a user who sets
  only the members they need passes exactly what a C caller passing a zeroed struct would.
- **Strings.** Procedures that take a name accept `character(len=*)`; the generated code copies it
  into a NUL-terminated `character(kind=c_char)` array that lives for the call and passes its
  address. The error message comes back the same way: `fuse_last_error(ctx, message)` fills a
  Fortran `character(len=*)` and reports truncation through the struct's own flag.
- **Communicators** are the Fortran MPI handle, passed as `integer(c_int64_t)`; on the C side
  `MPI_Comm_f2c` recovers it, which is the convention `set_communicator` already uses everywhere in
  this library.
- **Optional behaviour is a struct member, never an optional argument.** Fortran `optional` would
  put the argument *list* in the ABI again; a member with a default costs nothing and grows by the
  rules above.
- **Status is the function result**, `integer(c_int32_t)`, compared against the `FUSE_*` parameters.

What this costs: one more generated file and one more library to install. What it buys: a
Fortran program is written against the same fifteen names and the same structs as a C program,
and the test that proves the two agree is a `C_SIZEOF` table, not a reading.
