# Why the C ABI is one symbol

The usage side of this is in [](../user/c-abi.md). This page is the reasoning, kept because the
decision looks arbitrary until you have hit the alternative.

## The constraint

FUSE needs a C ABI for two opposite reasons: a **participant ABI** so a foreign code with its own
time loop can drive the coupler, and a **provider ABI** so the coupler can drive a plugin. C symbols
are unmangled and global, and the natural names collide. libfuse (Filesystem in Userspace) is
installed on essentially every Linux and HPC system and owns `fuse_*`, including `fuse_version`,
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
